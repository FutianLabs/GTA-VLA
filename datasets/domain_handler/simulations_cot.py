# ------------------------------------------------------------------------------
# CoT (Chain-of-Thought) Domain Handlers with Object Grounding
# ------------------------------------------------------------------------------
"""
Domain handlers that include CoT annotations with object grounding.

These handlers extend the base simulation handlers to:
1. Load CoT annotations from JSON files
2. Build CoT text with bounding boxes in Qwen3-VL format (0-1000 quantization)
3. Include CoT text in training samples

Supported tags (XML-style with explicit start/end):
- <TASK>instruction</TASK>: Main task instruction
- <SUBTASKS>task1 -> task2</SUBTASKS>: Sequence of subtasks
- <CURRENT>current subtask</CURRENT>: Current subtask with action type
- <OBJECTS>obj<|box_start|>(x1,y1),(x2,y2)<|box_end|></OBJECTS>: Key objects with bounding boxes
- <PICK>object<|box_start|>(x1,y1),(x2,y2)<|box_end|></PICK>: Pick target object with bbox
- <PLACE><|box_start|>(x1,y1),(x2,y2)<|box_end|></PLACE>: Place location
- <AFFORDANCE_2D>(x,y)</AFFORDANCE_2D>: 2D pixel coordinate
- <AFFORDANCE_3D><pos>x,y,z,rx,ry,rz</pos></AFFORDANCE_3D>: 3D position/orientation
- <GRIPPER_PATH_2D>x1,y1;x2,y2;...</GRIPPER_PATH_2D>: Future gripper 2D trajectory

Each tag is independently validated based on data availability.
"""

from __future__ import annotations

import copy
import json
import os
import io
import re
from pathlib import Path
from typing import Iterable, Dict, Optional

import numpy as np
import h5py

try:
    from .simulations import BridgeHandler, FractalHandler
    from .droid import DroidHandler
    from .robomind import RobomindHandler
    from ..cot_builder import CotBuilder, CotConfig
    from ..interaction_augmenter import InteractionAugmenter
except ImportError:
    import sys

    _cur_dir = Path(__file__).resolve().parent
    _datasets_dir = _cur_dir.parent
    _project_root = _datasets_dir.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from datasets.domain_handler.simulations import BridgeHandler, FractalHandler
    from datasets.domain_handler.droid import DroidHandler
    from datasets.domain_handler.robomind import RobomindHandler
    from datasets.cot_builder import CotBuilder, CotConfig
    from datasets.interaction_augmenter import InteractionAugmenter



def _pick_oversample_cfg_from_model(model_config) -> Optional[dict]:
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        if not model_config.get("cot_pick_keyframe_oversample", False):
            return None
        return {
            "skip_initial": int(model_config.get("cot_pick_keyframe_skip_initial", 1)),
            "num_anchors": int(model_config.get("cot_pick_keyframe_num_anchors", 2)),
            "radius": int(model_config.get("cot_pick_oversample_radius", 4)),
            "boost": float(model_config.get("cot_pick_oversample_boost", 2.0)),
            "stochastic": bool(model_config.get("cot_pick_oversample_stochastic", False)),
        }
    if not getattr(model_config, "cot_pick_keyframe_oversample", False):
        return None
    return {
        "skip_initial": int(getattr(model_config, "cot_pick_keyframe_skip_initial", 1)),
        "num_anchors": int(getattr(model_config, "cot_pick_keyframe_num_anchors", 2)),
        "radius": int(getattr(model_config, "cot_pick_oversample_radius", 4)),
        "boost": float(getattr(model_config, "cot_pick_oversample_boost", 2.0)),
        "stochastic": bool(getattr(model_config, "cot_pick_oversample_stochastic", False)),
    }


def _pick_anchor_frames_from_annotation(annotation: dict, skip_initial: int, num_anchors: int) -> list:
    ga = annotation.get("gripper_analysis") or {}
    kf = ga.get("keyframe_indices")
    if not kf:
        return []
    ordered = sorted({int(x) for x in kf})
    if skip_initial <= 0:
        rest = ordered
    else:
        if skip_initial >= len(ordered):
            return []
        rest = ordered[skip_initial:]
    if num_anchors <= 0:
        return []
    return rest[:num_anchors]


def _pick_oversample_repeat(
    frame_idx: int, anchors: list, radius: int, boost: float, stochastic: bool
) -> int:
    if not anchors:
        return 1
    if not any(abs(int(frame_idx) - int(a)) <= radius for a in anchors):
        return 1
    if stochastic:
        lam = max(float(boost), 1e-6)
        return max(1, int(np.random.poisson(lam)))
    return max(1, int(round(boost)))



class BridgeCotHandler(BridgeHandler):
    """
    Bridge dataset handler with CoT support (see cot_builder.py for tag details).
    
    Extends BridgeHandler with:
    - CoT text generation via CotBuilder (configured via set_cot_config)
    - Optional user interaction augmentation (configured via set_interaction_config)
    """
    dataset_name = "Bridge"
    
    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        
        # Annotation directory
        self.annotation_dir = meta.get("annotation_dir", None)
        self._annotation_cache: Dict[str, dict] = {}
        self.cot_builder = CotBuilder(CotConfig.from_meta(meta))
        
        # Interaction augmenter (disabled by default, initialized via set_interaction_config)
        self.interaction_augmenter = None
        
        if self.annotation_dir:
            print(f"📝 BridgeCotHandler: Loading annotations from {self.annotation_dir}")
        self._pick_oversample = None
    
    def set_cot_config(self, model_config) -> None:

        self.cot_builder = CotBuilder(CotConfig.from_model_config(model_config))
        self._pick_oversample = _pick_oversample_cfg_from_model(model_config)
        if self._pick_oversample:
            print(
                "📝 BridgeCotHandler: pick keyframe oversample "
                f"skip_initial={self._pick_oversample['skip_initial']} "
                f"num_anchors={self._pick_oversample['num_anchors']} "
                f"radius={self._pick_oversample['radius']} boost={self._pick_oversample['boost']} "
                f"stochastic={self._pick_oversample['stochastic']}"
            )
    
    def set_interaction_config(self, model_config) -> None:
        """
        Set interaction augmentation configuration from model config.
        
        Call this after handler initialization to enable user interaction augmentation.
        
        Args:
            model_config: XVLAConfig instance or dict with use_interaction_augmentation, etc.
        """
        self.interaction_augmenter = InteractionAugmenter.from_model_config(model_config)
        if self.interaction_augmenter.enabled:
            print(f"📝 BridgeCotHandler: Interaction augmentation enabled")
    
    def _load_annotation(self, episode_id: str) -> Optional[dict]:
        """Load annotation JSON for an episode."""
        if episode_id in self._annotation_cache:
            return self._annotation_cache[episode_id]
        
        
        json_path = os.path.join(self.annotation_dir, f"{episode_id}.json")
        if not os.path.exists(json_path):
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            self._annotation_cache[episode_id] = annotation
            return annotation
        except Exception as e:
            print(f"Warning: Failed to load annotation for {episode_id}: {e}")
            return None
    
    def _get_episode_id(self, traj_idx: int) -> str:
        """Extract episode_id from trajectory path."""
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]
        return Path(datapath).stem
    
    def _load_gripper_2d(self, h5_path: str) -> tuple[Optional[np.ndarray], bool]:
        """Load gripper_2d and valid flag. Returns (gripper_2d, valid_flag)."""
        if not h5_path or not os.path.exists(h5_path):
            return None, False
        try:
            with h5py.File(h5_path, 'r') as f:
                gripper_2d = f['gripper_position'][:] if 'gripper_position' in f else None
                valid_flag = f.attrs.get('gripper_2d_valid', False)
                return gripper_2d, valid_flag
        except Exception as e:
            print(f"Warning: Failed to load gripper data from {h5_path}: {e}")
        return None, False
    
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        image_processor=None,
        **kwargs
    ) -> Iterable[dict]:
        """Iterate episode samples with CoT annotations and optional interaction augmentation."""
        samples = list(super().iter_episode(
            traj_idx,
            num_actions=num_actions,
            training=training,
            image_aug=image_aug,
            lang_aug_map=lang_aug_map,
            image_processor=image_processor,
            **kwargs
        ))
        
        episode_id = self._get_episode_id(traj_idx)
        annotation = self._load_annotation(episode_id)
        
        if annotation is None:
            for sample in samples:
                sample["cot_text"] = ""
            return samples
        
        h5_path = annotation.get("h5_path")
        gripper_2d, gripper_2d_valid_flag = self._load_gripper_2d(h5_path)
        
        result_samples = []
        for sample in samples:
            frame_idx = sample.get("frame_idx", 0)
            
            # Apply interaction augmentation to instruction (only during training)
            if training and self.interaction_augmenter and self.interaction_augmenter.enabled:
                augmented_instruction, _ = self.interaction_augmenter.augment_sample(
                    sample=sample,
                    annotation=annotation,
                    frame_idx=frame_idx,
                    gripper_2d=gripper_2d,
                    gripper_2d_valid_flag=gripper_2d_valid_flag,
                    coord_scale=self.cot_builder.config.coord_scale,
                )
                sample["language_instruction"] = augmented_instruction
            
            # Build CoT (uses GT data, unaffected by instruction augmentation)
            cot_text = self.cot_builder.build_cot(
                annotation=annotation,
                frame_idx=frame_idx,
                gripper_2d=gripper_2d,
                gripper_2d_valid_flag=gripper_2d_valid_flag,
            )
            sample["cot_text"] = cot_text
            result_samples.append(sample)
        
        if (
            training
            and self._pick_oversample
            and annotation is not None
        ):
            anchors = _pick_anchor_frames_from_annotation(
                annotation,
                self._pick_oversample["skip_initial"],
                self._pick_oversample["num_anchors"],
            )
            if anchors:
                expanded = []
                r = self._pick_oversample["radius"]
                b = self._pick_oversample["boost"]
                stoch = self._pick_oversample["stochastic"]
                for sample in result_samples:
                    fi = int(sample.get("frame_idx", 0))
                    ct = sample.get("cot_text") or ""
                    n_rep = (
                        _pick_oversample_repeat(fi, anchors, r, b, stoch)
                        if ct.strip()
                        else 1
                    )
                    for _ in range(n_rep):
                        expanded.append(copy.copy(sample))
                return expanded
        
        return result_samples


class ManiSkillCotHandler(BridgeCotHandler):
    dataset_name = "ManiSkillCot"

    def _resolve_img_size_wh(self, annotation: dict, sample: dict) -> Optional[tuple[int, int]]:
        detections = annotation.get("detections", {})
        for detector in self.cot_builder.config.detector_priority:
            size = detections.get(detector, {}).get("image_size")
            if size and len(size) == 2:
                return int(size[0]), int(size[1])
        for det in detections.values():
            size = det.get("image_size") if isinstance(det, dict) else None
            if size and len(size) == 2:
                return int(size[0]), int(size[1])
        raw_images = sample.get("raw_images")
        if raw_images is not None and hasattr(raw_images, "shape") and len(raw_images.shape) >= 4:
            h, w = int(raw_images.shape[-2]), int(raw_images.shape[-1])
            return w, h
        return None

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        image_processor=None,
        **kwargs
    ) -> Iterable[dict]:
        samples = list(super(BridgeCotHandler, self).iter_episode(
            traj_idx,
            num_actions=num_actions,
            training=training,
            image_aug=image_aug,
            lang_aug_map=lang_aug_map,
            image_processor=image_processor,
            **kwargs
        ))

        episode_id = self._get_episode_id(traj_idx)
        annotation = self._load_annotation(episode_id)

        if annotation is None:
            for sample in samples:
                sample["cot_text"] = ""
            return samples

        h5_path = annotation.get("h5_path")
        gripper_2d, gripper_2d_valid_flag = self._load_gripper_2d(h5_path)

        result_samples = []
        for sample in samples:
            frame_idx = sample.get("frame_idx", 0)
            img_size_wh = self._resolve_img_size_wh(annotation, sample)

            if training and self.interaction_augmenter and self.interaction_augmenter.enabled:
                augmented_instruction, _ = self.interaction_augmenter.augment_sample(
                    sample=sample,
                    annotation=annotation,
                    frame_idx=frame_idx,
                    gripper_2d=gripper_2d,
                    gripper_2d_valid_flag=gripper_2d_valid_flag,
                    coord_scale=self.cot_builder.config.coord_scale,
                )
                sample["language_instruction"] = augmented_instruction

            cot_text = self.cot_builder.build_cot(
                annotation=annotation,
                frame_idx=frame_idx,
                gripper_2d=gripper_2d,
                gripper_2d_valid_flag=gripper_2d_valid_flag,
                img_size_wh=img_size_wh,
            )
            sample["cot_text"] = cot_text
            result_samples.append(sample)

        if (
            training
            and self._pick_oversample
            and annotation is not None
        ):
            anchors = _pick_anchor_frames_from_annotation(
                annotation,
                self._pick_oversample["skip_initial"],
                self._pick_oversample["num_anchors"],
            )
            if anchors:
                expanded = []
                r = self._pick_oversample["radius"]
                b = self._pick_oversample["boost"]
                stoch = self._pick_oversample["stochastic"]
                for sample in result_samples:
                    fi = int(sample.get("frame_idx", 0))
                    ct = sample.get("cot_text") or ""
                    n_rep = (
                        _pick_oversample_repeat(fi, anchors, r, b, stoch)
                        if ct.strip()
                        else 1
                    )
                    for _ in range(n_rep):
                        expanded.append(copy.copy(sample))
                return expanded

        return result_samples


class DroidCotHandler(DroidHandler):
    dataset_name = "Droid-*"

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        self.annotation_dir = meta.get("annotation_dir", None)
        self._annotation_cache: Dict[str, dict] = {}
        self.cot_builder = CotBuilder(CotConfig.from_meta(meta))
        self.interaction_augmenter = None
        if self.annotation_dir:
            print(f"📝 DroidCotHandler: Loading annotations from {self.annotation_dir}")

    def set_cot_config(self, model_config) -> None:
        self.cot_builder = CotBuilder(CotConfig.from_model_config(model_config))

    def set_interaction_config(self, model_config) -> None:
        self.interaction_augmenter = InteractionAugmenter.from_model_config(model_config)
        if self.interaction_augmenter.enabled:
            print("📝 DroidCotHandler: Interaction augmentation enabled")

    def _load_annotation(self, episode_id: str) -> Optional[dict]:
        if episode_id in self._annotation_cache:
            return self._annotation_cache[episode_id]
        if not self.annotation_dir:
            return None
        json_path = os.path.join(self.annotation_dir, f"{episode_id}.json")
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            self._annotation_cache[episode_id] = ann
            return ann
        except Exception:
            return None

    def _get_episode_id(self, traj_idx: int) -> str:
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]
        return Path(datapath).stem

    def _load_gripper_2d(self, h5_path: str) -> tuple[Optional[np.ndarray], bool]:
        if not h5_path or not os.path.exists(h5_path):
            return None, False
        try:
            with h5py.File(h5_path, "r") as f:
                gripper_2d = f["gripper_position"][:] if "gripper_position" in f else None
                valid_flag = f.attrs.get("gripper_2d_valid", False)
                return gripper_2d, valid_flag
        except Exception:
            return None, False

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        image_processor=None,
        **kwargs
    ) -> Iterable[dict]:
        samples = list(super().iter_episode(
            traj_idx,
            num_actions=num_actions,
            training=training,
            image_aug=image_aug,
            lang_aug_map=lang_aug_map,
            image_processor=image_processor,
            **kwargs
        ))
        episode_id = self._get_episode_id(traj_idx)
        annotation = self._load_annotation(episode_id)
        if annotation is None:
            for sample in samples:
                sample["cot_text"] = ""
            return samples
        h5_path = annotation.get("h5_path")
        gripper_2d, gripper_2d_valid_flag = self._load_gripper_2d(h5_path)
        result_samples = []
        for sample in samples:
            frame_idx = sample.get("frame_idx", 0)
            if training and self.interaction_augmenter and self.interaction_augmenter.enabled:
                aug_ins, _ = self.interaction_augmenter.augment_sample(
                    sample=sample,
                    annotation=annotation,
                    frame_idx=frame_idx,
                    gripper_2d=gripper_2d,
                    gripper_2d_valid_flag=gripper_2d_valid_flag,
                    coord_scale=self.cot_builder.config.coord_scale,
                )
                sample["language_instruction"] = aug_ins
            sample["cot_text"] = self.cot_builder.build_cot(
                annotation=annotation,
                frame_idx=frame_idx,
                gripper_2d=gripper_2d,
                gripper_2d_valid_flag=gripper_2d_valid_flag,
            )
            result_samples.append(sample)
        return result_samples


class RobomindCotHandler(RobomindHandler):
    dataset_name = "robomind-*"

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        self.annotation_dir = meta.get("annotation_dir", None)
        self._annotation_cache: Dict[str, dict] = {}
        self.cot_builder = CotBuilder(CotConfig.from_meta(meta))
        self.interaction_augmenter = None
        if self.annotation_dir:
            print(f"📝 RobomindCotHandler: Loading annotations from {self.annotation_dir}")

    def set_cot_config(self, model_config) -> None:
        self.cot_builder = CotBuilder(CotConfig.from_model_config(model_config))

    def set_interaction_config(self, model_config) -> None:
        self.interaction_augmenter = InteractionAugmenter.from_model_config(model_config)
        if self.interaction_augmenter.enabled:
            print("📝 RobomindCotHandler: Interaction augmentation enabled")

    def _load_annotation(self, episode_id: str) -> Optional[dict]:
        if episode_id in self._annotation_cache:
            return self._annotation_cache[episode_id]
        if not self.annotation_dir:
            return None
        json_path = os.path.join(self.annotation_dir, f"{episode_id}.json")
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            self._annotation_cache[episode_id] = ann
            return ann
        except Exception:
            return None

    def _get_episode_id(self, traj_idx: int) -> str:
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]
        ep_id_parts = (self.meta.get("dataset_config") or {}).get("episode_id_parts")
        if ep_id_parts:
            parts = Path(datapath).parts
            selected = []
            for i in ep_id_parts:
                idx = i if i >= 0 else len(parts) + i
                if 0 <= idx < len(parts):
                    selected.append(parts[idx])
            if selected:
                return "__".join(selected)
        return Path(datapath).stem

    def _load_gripper_2d(self, h5_path: str) -> tuple[Optional[np.ndarray], bool]:
        if not h5_path or not os.path.exists(h5_path):
            return None, False
        try:
            with h5py.File(h5_path, "r") as f:
                gripper_2d = f["gripper_position"][:] if "gripper_position" in f else None
                valid_flag = f.attrs.get("gripper_2d_valid", False)
                return gripper_2d, valid_flag
        except Exception:
            return None, False

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        image_processor=None,
        **kwargs
    ) -> Iterable[dict]:
        samples = list(super().iter_episode(
            traj_idx,
            num_actions=num_actions,
            training=training,
            image_aug=image_aug,
            lang_aug_map=lang_aug_map,
            image_processor=image_processor,
            **kwargs
        ))
        episode_id = self._get_episode_id(traj_idx)
        annotation = self._load_annotation(episode_id)
        if annotation is None:
            for sample in samples:
                sample["cot_text"] = ""
            return samples
        h5_path = annotation.get("h5_path")
        gripper_2d, gripper_2d_valid_flag = self._load_gripper_2d(h5_path)
        result_samples = []
        for sample in samples:
            frame_idx = sample.get("frame_idx", 0)
            if training and self.interaction_augmenter and self.interaction_augmenter.enabled:
                aug_ins, _ = self.interaction_augmenter.augment_sample(
                    sample=sample,
                    annotation=annotation,
                    frame_idx=frame_idx,
                    gripper_2d=gripper_2d,
                    gripper_2d_valid_flag=gripper_2d_valid_flag,
                    coord_scale=self.cot_builder.config.coord_scale,
                )
                sample["language_instruction"] = aug_ins
            sample["cot_text"] = self.cot_builder.build_cot(
                annotation=annotation,
                frame_idx=frame_idx,
                gripper_2d=gripper_2d,
                gripper_2d_valid_flag=gripper_2d_valid_flag,
            )
            result_samples.append(sample)
        return result_samples


class FractalCotHandler(FractalHandler):
    """
    Fractal dataset handler with CoT support.
    """
    dataset_name = "FractalCot"

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)

        self.annotation_dir = meta.get("annotation_dir", None)
        self._annotation_cache: Dict[str, dict] = {}
        self.cot_builder = CotBuilder(CotConfig.from_meta(meta))
        self.interaction_augmenter = None

        if self.annotation_dir:
            print(f"📝 FractalCotHandler: Loading annotations from {self.annotation_dir}")

    def set_cot_config(self, model_config) -> None:
        self.cot_builder = CotBuilder(CotConfig.from_model_config(model_config))

    def set_interaction_config(self, model_config) -> None:
        self.interaction_augmenter = InteractionAugmenter.from_model_config(model_config)
        if self.interaction_augmenter.enabled:
            print("📝 FractalCotHandler: Interaction augmentation enabled")

    def _load_annotation(self, episode_id: str) -> Optional[dict]:
        if episode_id in self._annotation_cache:
            return self._annotation_cache[episode_id]

        json_path = os.path.join(self.annotation_dir, f"{episode_id}.json")
        if not os.path.exists(json_path):
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            self._annotation_cache[episode_id] = annotation
            return annotation
        except Exception as e:
            print(f"Warning: Failed to load annotation for {episode_id}: {e}")
            return None

    def _get_episode_id(self, traj_idx: int) -> str:
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]
        return Path(datapath).stem

    def _load_gripper_2d(self, h5_path: str) -> tuple[Optional[np.ndarray], bool]:
        if not h5_path or not os.path.exists(h5_path):
            return None, False
        try:
            with h5py.File(h5_path, 'r') as f:
                gripper_2d = f['gripper_position'][:] if 'gripper_position' in f else None
                valid_flag = f.attrs.get('gripper_2d_valid', False)
                return gripper_2d, valid_flag
        except Exception as e:
            print(f"Warning: Failed to load gripper data from {h5_path}: {e}")
        return None, False

    def _resolve_img_size_wh(self, annotation: dict, sample: dict) -> Optional[tuple[int, int]]:
        detections = annotation.get("detections", {})
        for detector in self.cot_builder.config.detector_priority:
            size = detections.get(detector, {}).get("image_size")
            if size and len(size) == 2:
                return int(size[0]), int(size[1])
        for det in detections.values():
            size = det.get("image_size") if isinstance(det, dict) else None
            if size and len(size) == 2:
                return int(size[0]), int(size[1])
        raw_images = sample.get("raw_images")
        if raw_images is not None and hasattr(raw_images, "shape") and len(raw_images.shape) >= 4:
            h, w = int(raw_images.shape[-2]), int(raw_images.shape[-1])
            return w, h
        return None

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        image_processor=None,
        **kwargs
    ) -> Iterable[dict]:
        samples = list(super().iter_episode(
            traj_idx,
            num_actions=num_actions,
            training=training,
            image_aug=image_aug,
            lang_aug_map=lang_aug_map,
            image_processor=image_processor,
            **kwargs
        ))

        episode_id = self._get_episode_id(traj_idx)
        annotation = self._load_annotation(episode_id)

        if annotation is None:
            for sample in samples:
                sample["cot_text"] = ""
            return samples

        h5_path = annotation.get("h5_path")
        gripper_2d, gripper_2d_valid_flag = self._load_gripper_2d(h5_path)

        result_samples = []
        for sample in samples:
            frame_idx = sample.get("frame_idx", 0)
            img_size_wh = self._resolve_img_size_wh(annotation, sample)

            if training and self.interaction_augmenter and self.interaction_augmenter.enabled:
                augmented_instruction, _ = self.interaction_augmenter.augment_sample(
                    sample=sample,
                    annotation=annotation,
                    frame_idx=frame_idx,
                    gripper_2d=gripper_2d,
                    gripper_2d_valid_flag=gripper_2d_valid_flag,
                    coord_scale=self.cot_builder.config.coord_scale,
                )
                sample["language_instruction"] = augmented_instruction

            cot_text = self.cot_builder.build_cot(
                annotation=annotation,
                frame_idx=frame_idx,
                gripper_2d=gripper_2d,
                gripper_2d_valid_flag=gripper_2d_valid_flag,
                img_size_wh=img_size_wh,
            )
            sample["cot_text"] = cot_text
            result_samples.append(sample)

        return result_samples


    

    