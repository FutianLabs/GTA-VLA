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


if __name__ == "__main__":
    import argparse
    import shutil
    from PIL import Image, ImageDraw
    from torchvision import transforms
    import matplotlib.pyplot as plt

    def _to_numpy(value):
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _save_text(path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _save_scalar(path: Path, value):
        _save_text(path.with_suffix(".txt"), str(value))

    def _save_image_tensor(path: Path, arr: np.ndarray):
        x = np.array(arr)
        if x.ndim == 2:
            if x.dtype != np.uint8:
                x = x.astype(np.float32)
                x_min, x_max = float(np.min(x)), float(np.max(x))
                if x_max > x_min:
                    x = (x - x_min) / (x_max - x_min)
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(x).save(path)
            return

        if x.ndim == 3:
            if x.shape[0] in (1, 3):
                x = np.transpose(x, (1, 2, 0))
            if x.shape[-1] == 1:
                x = x[..., 0]
            if x.dtype != np.uint8:
                x = x.astype(np.float32)
                if np.max(x) <= 1.0 + 1e-6:
                    x = x * 255.0
                x = x.clip(0, 255).astype(np.uint8)
            Image.fromarray(x).save(path)

    def _plot_array(path_prefix: Path, arr: np.ndarray):
        arr = np.asarray(arr)
        npy_path = path_prefix.with_suffix(".npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, arr)

        if arr.ndim == 0:
            _save_scalar(path_prefix, arr.item())
            return

        if arr.ndim == 1:
            fig = plt.figure(figsize=(10, 3))
            plt.plot(arr)
            plt.title(path_prefix.name)
            plt.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(path_prefix.with_suffix(".png"), dpi=160)
            plt.close(fig)
            return

        if arr.ndim == 2:
            fig = plt.figure(figsize=(10, 4))
            plt.imshow(arr, aspect="auto", cmap="viridis")
            plt.colorbar()
            plt.title(path_prefix.name)
            fig.tight_layout()
            fig.savefig(path_prefix.with_suffix(".png"), dpi=160)
            plt.close(fig)
            return

        if arr.ndim == 3 and (arr.shape[0] in (1, 3) or arr.shape[-1] in (1, 3)):
            _save_image_tensor(path_prefix.with_suffix(".png"), arr)
            return

        if arr.ndim == 4 and arr.shape[1] in (1, 3):
            for i in range(arr.shape[0]):
                _save_image_tensor(path_prefix.parent / f"{path_prefix.name}_view{i}.png", arr[i])
            return

        flat = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr[None, ...]
        fig = plt.figure(figsize=(10, 4))
        plt.imshow(flat, aspect="auto", cmap="magma")
        plt.colorbar()
        plt.title(path_prefix.name)
        fig.tight_layout()
        fig.savefig(path_prefix.with_suffix(".png"), dpi=160)
        plt.close(fig)

    def _draw_cot_boxes(raw_img: np.ndarray, cot_text: str, out_path: Path):
        img = raw_img.copy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            x = img.astype(np.float32)
            if np.max(x) <= 1.0 + 1e-6:
                x = x * 255.0
            img = x.clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        w, h = pil.size
        boxes = re.findall(r"\((\d+),(\d+)\)\s*,\s*\((\d+),(\d+)\)", cot_text or "")
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in b]
            x1 = int(x1 * w / 1000)
            y1 = int(y1 * h / 1000)
            x2 = int(x2 * w / 1000)
            y2 = int(y2 * h / 1000)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=2)
            draw.text((x1, max(0, y1 - 12)), f"box{i}", fill=(255, 80, 80))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)

    def _read_state_arrays(datapath):
        if not isinstance(datapath, str):
            datapath = datapath[0]
        proprio = None
        action = None
        with h5py.File(datapath, "r") as f:
            if "proprio" in f:
                proprio = np.asarray(f["proprio"][()])
            if "action" in f:
                action = np.asarray(f["action"][()])
        return proprio, action

    def _to_2d(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr[:, None]
        if arr.ndim > 2:
            return arr.reshape(arr.shape[0], -1)
        return arr

    def _plot_compare_overlay(path_prefix: Path, a: np.ndarray, b: np.ndarray, name_a: str, name_b: str):
        a2 = _to_2d(a)
        b2 = _to_2d(b)
        if a2 is None or b2 is None:
            return
        T = min(a2.shape[0], b2.shape[0])
        D = min(a2.shape[1], b2.shape[1])
        if T <= 0 or D <= 0:
            return
        a2 = a2[:T, :D]
        b2 = b2[:T, :D]
        show_dims = min(D, 6)
        fig, axes = plt.subplots(show_dims, 1, figsize=(12, 2.2 * show_dims), sharex=True)
        if show_dims == 1:
            axes = [axes]
        for i in range(show_dims):
            axes[i].plot(a2[:, i], linewidth=1.0, label=name_a)
            axes[i].plot(b2[:, i], linewidth=1.0, label=name_b)
            axes[i].set_ylabel(f"d{i}")
            axes[i].grid(alpha=0.3)
            if i == 0:
                axes[i].legend(loc="upper right")
        axes[-1].set_xlabel("t")
        fig.tight_layout()
        path_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_prefix.with_suffix(".png"), dpi=180)
        plt.close(fig)

        diff = a2 - b2
        fig = plt.figure(figsize=(12, 4))
        plt.imshow(diff, aspect="auto", cmap="coolwarm")
        plt.colorbar()
        plt.title(f"{path_prefix.name}_diff ({name_a} - {name_b})")
        fig.tight_layout()
        fig.savefig(path_prefix.with_name(f"{path_prefix.name}_diff").with_suffix(".png"), dpi=180)
        plt.close(fig)

        info = {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "aligned_shape": [int(T), int(D)],
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "max_abs_diff": float(np.max(np.abs(diff))),
        }
        _save_text(path_prefix.with_suffix(".json"), json.dumps(info, ensure_ascii=False, indent=2))

    parser = argparse.ArgumentParser(description="Visualize CoT result_samples keys.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fractal",
        choices=["fractal", "bridge"],
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
    )
    parser.add_argument("--traj_idx", type=int, default=0)
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument(
        "--compare_output_root",
        type=str,
        default="/VLA-Data/scripts/lingyiran/x-vla-main/visualization/compare",
    )
    parser.add_argument(
        "--compare_with",
        type=str,
        default=None,
        choices=["fractal", "bridge"],
    )
    args = parser.parse_args()

    default_meta = {
        "fractal": "/VLA-Data/scripts/lianqing/data/xvla_metadata/fractal_meta.json",
        "bridge": "/VLA-Data/scripts/lianqing/data/xvla_metadata/bridge_meta.json",
    }
    default_annotation = {
        "fractal": "/VLA-Data/scripts/lianqing/data/xvla/cot_annotations/fractal_annotations_main",
        "bridge": "/VLA-Data/scripts/lianqing/data/xvla/cot_annotations/bridge_annotations_main",
    }
    default_output = {
        "fractal": "/VLA-Data/scripts/lingyiran/x-vla-main/visualization/fractal",
        "bridge": "/VLA-Data/scripts/lingyiran/x-vla-main/visualization/bridge",
    }

    if args.meta_path is None:
        args.meta_path = default_meta[args.dataset]
    if args.annotation_dir is None:
        args.annotation_dir = default_annotation[args.dataset]
    if args.output_root is None:
        args.output_root = default_output[args.dataset]

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["annotation_dir"] = args.annotation_dir

    if args.dataset == "bridge":
        handler = BridgeCotHandler(meta=meta, num_views=max(args.num_views, 1))
    else:
        handler = FractalCotHandler(meta=meta, num_views=max(args.num_views, 1))
    result_samples = list(
        handler.iter_episode(
            traj_idx=args.traj_idx,
            num_actions=args.num_actions,
            training=False,
            image_aug=transforms.ToTensor(),
            lang_aug_map=None,
            image_processor=None,
        )
    )

    if len(result_samples) == 0:
        raise RuntimeError("No samples generated from first episode.")

    episode_id = handler._get_episode_id(args.traj_idx)
    out_root = Path(args.output_root) / episode_id
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    key_set = sorted({k for s in result_samples for k in s.keys()})
    _save_text(out_root / "keys.txt", "\n".join(key_set))

    datapath = meta["datalist"][args.traj_idx]
    proprio_arr, action_arr = _read_state_arrays(datapath)
    if proprio_arr is not None:
        _plot_array(out_root / "proprio_trajectory_all_dims", proprio_arr)
        if proprio_arr.ndim == 2 and proprio_arr.shape[1] >= 3:
            _plot_array(out_root / "proprio_xyz", proprio_arr[:, :3])
    if action_arr is not None:
        _plot_array(out_root / "action_trajectory_all_dims", action_arr)
        if action_arr.ndim == 2 and action_arr.shape[1] >= 3:
            _plot_array(out_root / "action_xyz", action_arr[:, :3])

    if args.compare_with is not None and args.compare_with != args.dataset:
        cmp_meta_path = default_meta[args.compare_with]
        with open(cmp_meta_path, "r", encoding="utf-8") as f:
            cmp_meta = json.load(f)
        if args.traj_idx >= len(cmp_meta["datalist"]):
            raise RuntimeError(
                f"traj_idx={args.traj_idx} out of range for {args.compare_with} meta"
            )
        cmp_proprio, cmp_action = _read_state_arrays(cmp_meta["datalist"][args.traj_idx])
        compare_dir = (
            Path(args.compare_output_root)
            / f"{args.dataset}_vs_{args.compare_with}"
            / episode_id
        )
        if proprio_arr is not None and cmp_proprio is not None:
            _plot_compare_overlay(
                compare_dir / "proprio_overlay",
                proprio_arr,
                cmp_proprio,
                args.dataset,
                args.compare_with,
            )
        if action_arr is not None and cmp_action is not None:
            _plot_compare_overlay(
                compare_dir / "action_overlay",
                action_arr,
                cmp_action,
                args.dataset,
                args.compare_with,
            )

    for sample_i, sample in enumerate(result_samples):
        sample_dir = out_root / f"sample_{sample_i:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for key, value in sample.items():
            key_base = sample_dir / key
            if isinstance(value, str):
                _save_text(key_base.with_suffix(".txt"), value)
                if key == "cot_text" and "raw_images" in sample:
                    raw = _to_numpy(sample["raw_images"])
                    if raw.ndim == 4 and raw.shape[0] > 0:
                        _draw_cot_boxes(raw[0], value, key_base.with_name("cot_text_overlay.png"))
            elif isinstance(value, (int, float, np.integer, np.floating, bool)):
                _save_scalar(key_base, value)
            elif isinstance(value, (dict, list, tuple)):
                try:
                    _save_text(key_base.with_suffix(".json"), json.dumps(value, ensure_ascii=False, indent=2))
                except TypeError:
                    _save_text(key_base.with_suffix(".txt"), str(value))
            else:
                arr = _to_numpy(value)
                _plot_array(key_base, arr)

    print(f"episode_id: {episode_id}")
    print(f"samples: {len(result_samples)}")
    print(f"saved_to: {out_root}")

    