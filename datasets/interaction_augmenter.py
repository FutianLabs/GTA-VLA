# ------------------------------------------------------------------------------
# Interaction Augmenter - Data augmentation for user interaction training
# ------------------------------------------------------------------------------
"""
Augment instructions with user interaction tags for interactive VLA training.

Vision tags (same tokens as CoT builder):
- <|box_start|>(x1,y1),(x2,y2)<|box_end|>
- <|affordance_2d_start|>(x,y)<|affordance_2d_end|>
- <|gripper_path_2d_start|>(x1,y1);...;(xn,yn)<|gripper_path_2d_end|>

Coordinates normalized to 0-1000 (Qwen3-VL style).
Box/point tags replace the object name in the instruction (e.g. "pick A to B" -> "pick <box> to B").

Usage:
    augmenter = InteractionAugmenter.from_model_config(model_config)
    instruction, _ = augmenter.augment_sample(sample, annotation, frame_idx, gripper_2d)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cot_builder import sample_gripper_path


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InteractionConfig:
    """Configuration for interaction augmentation."""
    enabled: bool = False
    aug_ratio: float = 0.5
    modes: Dict[str, float] = None
    
    def __post_init__(self):
        if self.modes is None:
            self.modes = {
                "pick_box": 0.34,
                "place_box": 0.20,
                "pick_and_place": 0.20,
                "affordance_2d": 0.16,
                "gripper_path_2d": 0.10,
            }
    
    @classmethod
    def from_model_config(cls, config) -> "InteractionConfig":
        """Create from model config (XVLAConfig or dict)."""
        def _get(key, default):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)
        
        return cls(
            enabled=_get('use_interaction_augmentation', False),
            aug_ratio=_get('interaction_aug_ratio', 0.5),
            modes=_get('interaction_modes', None),
        )


# Tag token mapping
_TAG_TOKENS = {
    "box": ("<|box_start|>", "<|box_end|>"),
    "affordance_2d": ("<|affordance_2d_start|>", "<|affordance_2d_end|>"),
    "gripper_path_2d": ("<|gripper_path_2d_start|>", "<|gripper_path_2d_end|>"),
}


def _valid_box(box) -> bool:
    return box is not None and len(box) == 4 and any(v != 0 for v in box)


# =============================================================================
# Augmenter
# =============================================================================

class InteractionAugmenter:
    """Augment instructions with user interaction tags using GT data."""
    
    def __init__(self, config: InteractionConfig = None):
        self.config = config or InteractionConfig()
        self.enabled = self.config.enabled
    
    @classmethod
    def from_model_config(cls, config) -> "InteractionAugmenter":
        return cls(InteractionConfig.from_model_config(config))
    
    # --- Coordinate formatting ---
    
    def _normalize(self, val: int, size: int, scale: int) -> int:
        return max(0, min(scale, int(val * scale / size)))
    
    def _format_box(self, box: List[int], scale: int, img_size: Tuple[int, int]) -> str:
        w, h = img_size
        x1, y1, x2, y2 = box
        return (f"({self._normalize(x1, w, scale)},{self._normalize(y1, h, scale)}),"
                f"({self._normalize(x2, w, scale)},{self._normalize(y2, h, scale)})")
    
    def _format_affordance_2d(self, coord: Tuple[int, int], scale: int, img_size: Tuple[int, int]) -> str:
        w, h = img_size
        return f"({self._normalize(coord[0], w, scale)},{self._normalize(coord[1], h, scale)})"
    
    def _format_gripper_path_2d(self, path: List[Tuple[int, int]], scale: int, img_size: Tuple[int, int]) -> str:
        return ";".join(self._format_affordance_2d(p, scale, img_size) for p in path)
    
    # --- Tag insertion ---
    
    def _wrap_tag(self, tag: str, content: str) -> str:
        start, end = _TAG_TOKENS[tag]
        return f"{start}{content}{end}"
    
    def _insert_after_object(self, instruction: str, object_name: str, tag_str: str) -> str:
        """Insert tag_str after object_name in instruction. Append at end if not found."""
        if object_name:
            idx = instruction.lower().find(object_name.lower())
            if idx != -1:
                end = idx + len(object_name)
                return f"{instruction[:end]} {tag_str}{instruction[end:]}"
        return f"{instruction} {tag_str}"
    
    def _replace_object(self, instruction: str, object_name: str, tag_str: str) -> str:
        """Replace object_name with tag_str in instruction. Append at end if not found."""
        if object_name:
            idx = instruction.lower().find(object_name.lower())
            if idx != -1:
                return f"{instruction[:idx]}{tag_str}{instruction[idx + len(object_name):]}"
        return f"{instruction} {tag_str}"
    
    # --- GT data extraction ---
    
    @staticmethod
    def _get_subtask(annotation: dict, frame_idx: int) -> Optional[dict]:
        step_labels = annotation.get("step_labels", [])
        if frame_idx >= len(step_labels):
            return None
        subtask_id = step_labels[frame_idx]
        subtasks = annotation.get("subtasks", [])
        return subtasks[subtask_id - 1] if 0 < subtask_id <= len(subtasks) else None
    
    @staticmethod
    def _get_img_size(annotation: dict) -> Tuple[int, int]:
        for detector in ["seed_vl", "dino_x"]:
            size = annotation.get("detections", {}).get(detector, {}).get("image_size")
            if size and len(size) == 2:
                return tuple(size)
        return (256, 256)
    
    @staticmethod
    def _get_tracking_boxes(annotation: dict, subtask: dict) -> dict:
        return annotation.get("pick_object_tracking", {}).get(
            f"subtask_{subtask.get('id')}", {}
        ).get("boxes", {})
    
    def _get_pick_box(self, annotation: dict, subtask: dict, frame_idx: int) -> Optional[List[int]]:
        box = self._get_tracking_boxes(annotation, subtask).get(str(frame_idx))
        return box if _valid_box(box) else None
    
    def _get_place_box(self, annotation: dict, subtask: dict) -> Optional[List[int]]:
        if subtask.get("gripper_key_status", {}).get("action_type") != "place":
            return None
        boxes = self._get_tracking_boxes(annotation, subtask)
        for fidx in sorted((int(k) for k in boxes), reverse=True)[:5]:
            box = boxes.get(str(fidx))
            if _valid_box(box):
                return box
        return None
    
    @staticmethod
    def _get_affordance_2d(subtask: dict, gripper_2d_valid: bool) -> Optional[Tuple[int, int]]:
        if not gripper_2d_valid:
            return None
        pos = subtask.get("gripper_key_status", {}).get("gripper_2d")
        if pos and len(pos) == 2:
            return (int(pos[0]), int(pos[1]))
        return None
    
    @staticmethod
    def _get_gripper_path_2d(subtask: dict, frame_idx: int, gripper_2d: np.ndarray,
                             gripper_2d_valid: bool, num_points: int = 5) -> Optional[List[Tuple[int, int]]]:
        """Get gripper path using shared distance-based sampling (same as CoT builder)."""
        if not gripper_2d_valid or gripper_2d is None:
            return None
        frame_range = subtask.get("frame_range")
        if not frame_range or len(frame_range) < 2:
            return None
        end = min(frame_range[1], len(gripper_2d) - 1)
        if frame_idx >= end:
            return None
        path = sample_gripper_path(gripper_2d, frame_idx, end, num_points)
        return path if len(path) > 1 else None
    
    # --- Main augmentation ---
    
    def _sample_mode(self) -> str:
        modes, weights = zip(*self.config.modes.items())
        return random.choices(modes, weights=weights, k=1)[0]
    
    def augment_sample(
        self,
        sample: dict,
        annotation: dict,
        frame_idx: int,
        gripper_2d: np.ndarray = None,
        gripper_2d_valid_flag: bool = False,
        coord_scale: int = 1000,
    ) -> Tuple[str, Dict[str, Any]]:
        """Augment instruction with user interaction tags.
        Returns (augmented_instruction, user_interaction_dict).
        """
        instruction = sample.get("language_instruction", "")
        user_interaction = {}
        
        if not self.enabled or random.random() >= self.config.aug_ratio:
            return instruction, user_interaction
        
        mode = self._sample_mode()
        if mode == "none":  # backward compat
            return instruction, user_interaction
        
        subtask = self._get_subtask(annotation, frame_idx)
        if not subtask:
            return instruction, user_interaction
        
        img_size = self._get_img_size(annotation)
        pick_obj = subtask.get("pick_object_clean")
        place_loc = subtask.get("place_location") or subtask.get("target_location")
        
        # Box modes: replace object name with box tag
        # pick_and_place triggers both pick and place
        if mode in ("pick_box", "pick_and_place"):
            box = self._get_pick_box(annotation, subtask, frame_idx)
            if box:
                tag = self._wrap_tag("box", self._format_box(box, coord_scale, img_size))
                instruction = self._replace_object(instruction, pick_obj, tag)
                user_interaction["user_pick_box"] = box
        
        if mode in ("place_box", "pick_and_place"):
            box = self._get_place_box(annotation, subtask)
            if box:
                tag = self._wrap_tag("box", self._format_box(box, coord_scale, img_size))
                instruction = self._replace_object(instruction, place_loc, tag)
                user_interaction["user_place_box"] = box
        
        # Point/path modes: insert after object name
        if mode == "affordance_2d":
            coord = self._get_affordance_2d(subtask, gripper_2d_valid_flag)
            if coord:
                tag = self._wrap_tag("affordance_2d", self._format_affordance_2d(coord, coord_scale, img_size))
                instruction = self._insert_after_object(instruction, pick_obj, f"with affordance {tag}")
                user_interaction["user_affordance_2d"] = coord
        
        elif mode == "gripper_path_2d":
            path = self._get_gripper_path_2d(subtask, frame_idx, gripper_2d, gripper_2d_valid_flag)
            if path:
                tag = self._wrap_tag("gripper_path_2d", self._format_gripper_path_2d(path, coord_scale, img_size))
                instruction = f"{instruction}, follow the gripper path {tag}"
                user_interaction["user_gripper_path_2d"] = path
        
        return instruction, user_interaction
