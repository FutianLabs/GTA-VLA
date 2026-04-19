# ------------------------------------------------------------------------------
# VLA CoT Builder - Modular Tag-based CoT Text Generator
# ------------------------------------------------------------------------------
"""
Modular CoT (Chain-of-Thought) text builder for VLA training.

Each tag has independent validation logic, and the final CoT is composed
of all valid tags. This allows flexible handling of incomplete data.

Tags:
  Text (XML):    <TASK>, <SUBTASKS>, <CURRENT>
  Vision (special tokens): <|objects_start|>, <|pick_start|>, <|place_start|>,
                           <|affordance_2d_start|>, <|gripper_path_2d_start|>
  Boundary:      <|cot_start|>...<|cot_end|>

Coordinates: Qwen3-VL style, quantized to 0-1000.

Usage:
    config = CotConfig(coord_scale=1000)
    builder = CotBuilder(config)
    cot_text = builder.build_cot(annotation, frame_idx, gripper_2d)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import h5py
import os


@dataclass
class CotConfig:
    """
    Configuration for CoT builder.
    
    Note: img_size is NOT in config - it's read from annotation's detections.
    """
    coord_scale: int = 1000  # Qwen3-VL normalizes to 0-1000
    gripper_future_steps: int = 5  # Number of distance-sampled points from current to subtask end
    detector_priority: List[str] = field(default_factory=lambda: ["seed_vl", "dino_x"])
    
    @classmethod
    def from_meta(cls, meta: dict) -> "CotConfig":
        """Create config from meta dictionary (legacy support)."""
        cot_config = meta.get("cot_config", {})
        return cls(
            coord_scale=cot_config.get("coord_scale", 1000),
            gripper_future_steps=cot_config.get("gripper_future_steps", 5),
            detector_priority=cot_config.get("detector_priority", ["seed_vl", "dino_x"]),
        )
    
    @classmethod
    def from_model_config(cls, model_config) -> "CotConfig":
        """Create config from XVLAConfig or dict."""
        if isinstance(model_config, dict):
            return cls(
                coord_scale=model_config.get("cot_coord_scale", 1000),
                gripper_future_steps=model_config.get("cot_gripper_future_steps", 5),
                detector_priority=model_config.get("cot_detector_priority", ["seed_vl", "dino_x"]),
            )
        return cls(
            coord_scale=getattr(model_config, "cot_coord_scale", 1000),
            gripper_future_steps=getattr(model_config, "cot_gripper_future_steps", 5),
            detector_priority=getattr(model_config, "cot_detector_priority", ["seed_vl", "dino_x"]),
        )


def sample_gripper_path(
    gripper_2d: np.ndarray,
    start_frame: int,
    end_frame: int,
    num_points: int,
) -> List[Tuple[int, int]]:
    """Sample gripper 2D trajectory uniformly by cumulative distance.
    
    Args:
        gripper_2d: Array of shape (T, 2+) with gripper pixel coords per frame.
        start_frame: Start frame index (inclusive).
        end_frame: End frame index (inclusive).
        num_points: Number of output points.
    
    Returns:
        List of (x, y) tuples, length num_points. Empty list if no valid data.
    """
    pts = []
    for idx in range(start_frame, end_frame + 1):
        if idx < len(gripper_2d) and len(gripper_2d[idx]) >= 2:
            x, y = float(gripper_2d[idx][0]), float(gripper_2d[idx][1])
            if np.isnan(x) or np.isnan(y):
                continue
            pts.append([x, y])
    
    if not pts:
        return []
    
    path = np.array(pts)
    if len(path) == 1:
        return [(int(path[0, 0]), int(path[0, 1]))] * num_points
    
    cum_dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if cum_dist[-1] == 0:
        return [(int(path[0, 0]), int(path[0, 1]))] * num_points
    
    targets = np.linspace(0, cum_dist[-1], num_points)
    sx = np.interp(targets, cum_dist, path[:, 0])
    sy = np.interp(targets, cum_dist, path[:, 1])
    return [(int(x), int(y)) for x, y in zip(sx, sy)]


def _valid_box(box) -> bool:
    """Check if a box [x1,y1,x2,y2] is non-null and non-zero."""
    return box is not None and len(box) == 4 and any(v != 0 for v in box)


@dataclass
class BuildContext:
    """Build context providing data access methods for tag builders."""
    annotation: dict
    frame_idx: int
    gripper_2d: Optional[np.ndarray]
    gripper_2d_valid_flag: bool = False
    config: CotConfig = field(default_factory=CotConfig)
    _img_size_wh: Optional[Tuple[int, int]] = None

    @property
    def img_size(self) -> Tuple[int, int]:
        """Image size as (W, H). Must be set externally from actual image data."""
        if self._img_size_wh is None:
            raise ValueError(
                "img_size_wh not set. Pass img_size_wh from HDF5 image to build_cot()."
            )
        return self._img_size_wh
    
    def get_current_subtask(self) -> Optional[dict]:
        """Get current subtask based on step_labels and frame_idx."""
        step_labels = self.annotation.get("step_labels", [])
        if self.frame_idx >= len(step_labels):
            return None
        subtask_id = step_labels[self.frame_idx]  # 1-indexed
        subtasks = self.annotation.get("subtasks", [])
        if 0 < subtask_id <= len(subtasks):
            return subtasks[subtask_id - 1]
        return None
    
    def _get_tracking_boxes(self) -> dict:
        """Get pick_object_tracking boxes dict for current subtask."""
        subtask = self.get_current_subtask()
        if not subtask:
            return {}
        tracking = self.annotation.get("pick_object_tracking", {}).get(f"subtask_{subtask.get('id')}", {})
        return tracking.get("boxes", {})
    
    def get_pick_object_box(self) -> Optional[List[int]]:
        """Get pick object bbox at current frame."""
        box = self._get_tracking_boxes().get(str(self.frame_idx))
        return box if _valid_box(box) else None
    
    def get_all_object_boxes(self, obj_name: str) -> List[List[int]]:
        """Get all bounding boxes for an object at current frame, sorted by x0 (left to right)."""
        detections = self.annotation.get("detections", {})
        for detector in self.config.detector_priority:
            det_data = detections.get(detector, {}).get("detections", {})
            boxes = det_data.get(str(self.frame_idx), {}).get(obj_name, [])
            valid = [b for b in boxes if _valid_box(b)]
            if valid:
                valid.sort(key=lambda b: b[0])
                return valid[:5]
        return []
    
    def get_pick_object_last_frame_box(self) -> Optional[List[int]]:
        """Get pick object bbox at last frame (for PLACE tag)."""
        boxes = self._get_tracking_boxes()
        if not boxes:
            return None
        for fi in sorted((int(k) for k in boxes), reverse=True)[:5]:
            box = boxes.get(str(fi))
            if _valid_box(box):
                return box
        return None
    
    def get_future_gripper_path(self) -> List[Tuple[int, int]]:
        """Get gripper 2D trajectory from current frame to subtask end."""
        if self.gripper_2d is None:
            return []
        subtask = self.get_current_subtask()
        if not subtask:
            return []
        frame_range = subtask.get("frame_range")
        if not frame_range or len(frame_range) < 2:
            return []
        
        start = self.frame_idx
        end = min(frame_range[1], len(self.gripper_2d) - 1)
        N = self.config.gripper_future_steps
        
        if start >= end:
            if start < len(self.gripper_2d) and len(self.gripper_2d[start]) >= 2:
                pos = self.gripper_2d[start]
                px, py = float(pos[0]), float(pos[1])
                if np.isnan(px) or np.isnan(py):
                    return []
                return [(int(px), int(py))] * N
            return []
        
        return sample_gripper_path(self.gripper_2d, start, end, N)
    
    
    def _normalize_coord(self, x: float, y: float) -> Tuple[int, int]:
        w, h = self.img_size
        s = self.config.coord_scale
        nx = max(0, min(s, int(x * s / w)))
        ny = max(0, min(s, int(y * s / h)))
        return nx, ny
    
    def format_box(self, box: List[int]) -> str:
        """Format bbox: (x1,y1),(x2,y2) normalized to coord_scale."""
        x1, y1, x2, y2 = box
        a = self._normalize_coord(x1, y1)
        b = self._normalize_coord(x2, y2)
        return f"({a[0]},{a[1]}),({b[0]},{b[1]})"
    
    def format_point(self, x: int, y: int) -> str:
        """Format point: (x,y) normalized to coord_scale."""
        nx, ny = self._normalize_coord(x, y)
        return f"({nx},{ny})"
    
    def is_gripper_2d_valid(self) -> bool:
        return self.gripper_2d_valid_flag and self.gripper_2d is not None and len(self.gripper_2d) > 0


# =============================================================================
# Tag builder functions
# Each returns Optional[str]: content if valid, None if not.
# =============================================================================

def _build_task(ctx: BuildContext) -> Optional[str]:
    inst = ctx.annotation.get("original_instruction", "")
    if not inst or not inst.strip():
        return None
    return f"<TASK>{inst}</TASK>"


def _build_subtasks(ctx: BuildContext) -> Optional[str]:
    inst = ctx.annotation.get("original_instruction", "")
    if not inst or not inst.strip():
        return None
    subtasks = ctx.annotation.get("subtasks", [])
    subs = [st.get("sub_instruction", "") for st in subtasks if st.get("sub_instruction", "").strip()]
    if not subs:
        return None
    return f"<SUBTASKS>{' -> '.join(subs)}</SUBTASKS>"


def _build_current(ctx: BuildContext) -> Optional[str]:
    inst = ctx.annotation.get("original_instruction", "")
    if not inst or not inst.strip():
        return None
    subtask = ctx.get_current_subtask()
    if not subtask:
        return None
    sub_inst = subtask.get("sub_instruction", "").strip()
    if not sub_inst:
        return None
    return f"<CURRENT>{sub_inst}</CURRENT>"


def _build_objects(ctx: BuildContext) -> Optional[str]:
    subtask = ctx.get_current_subtask()
    if not subtask:
        return None
    key_objects = subtask.get("key_objects_clean", [])
    if not key_objects:
        return None
    
    parts = []
    for obj_name in key_objects:
        boxes = ctx.get_all_object_boxes(obj_name)
        if boxes:
            box_strs = [f"<|box_start|>{ctx.format_box(b)}<|box_end|>" for b in boxes]
            parts.append(f"{obj_name}{''.join(box_strs)}")
    
    if not parts:
        return None
    return f"<|objects_start|>{', '.join(parts)}<|objects_end|>"


def _build_pick(ctx: BuildContext) -> Optional[str]:
    subtask = ctx.get_current_subtask()
    if not subtask or not subtask.get("pick_object_clean"):
        return None
    box = ctx.get_pick_object_box()
    if not box:
        return None
    pick_obj = subtask["pick_object_clean"]
    return f"<|pick_start|>{pick_obj}<|box_start|>{ctx.format_box(box)}<|box_end|><|pick_end|>"


def _build_place(ctx: BuildContext) -> Optional[str]:
    subtask = ctx.get_current_subtask()
    if not subtask:
        return None
    action_type = subtask.get("gripper_key_status", {}).get("action_type")
    if action_type != "place":
        return None
    box = ctx.get_pick_object_last_frame_box()
    if not box:
        return None
    return f"<|place_start|><|box_start|>{ctx.format_box(box)}<|box_end|><|place_end|>"


def _build_affordance_2d(ctx: BuildContext) -> Optional[str]:
    if not ctx.is_gripper_2d_valid():
        return None
    subtask = ctx.get_current_subtask()
    if not subtask:
        return None
    gripper_key = subtask.get("gripper_key_status", {})
    gripper_2d = gripper_key.get("gripper_2d")
    if not gripper_key.get("action_type") or not gripper_2d or len(gripper_2d) != 2:
        return None
    x, y = float(gripper_2d[0]), float(gripper_2d[1])
    if np.isnan(x) or np.isnan(y):
        return None
    return f"<|affordance_2d_start|>{ctx.format_point(x, y)}<|affordance_2d_end|>"


def _build_gripper_path(ctx: BuildContext) -> Optional[str]:
    if not ctx.is_gripper_2d_valid():
        return None
    if ctx.gripper_2d is None or len(ctx.gripper_2d) <= ctx.frame_idx:
        return None
    path = ctx.get_future_gripper_path()
    if not path:
        return None
    path_str = ";".join(ctx.format_point(x, y) for x, y in path)
    return f"<|gripper_path_2d_start|>{path_str}<|gripper_path_2d_end|>"


# Ordered list of (name, builder_fn) — name is used by build_cot_with_details
_TAG_BUILDERS: List[Tuple[str, Any]] = [
    ("TASK", _build_task),
    ("SUBTASKS", _build_subtasks),
    ("CURRENT", _build_current),
    ("OBJECTS", _build_objects),
    ("PICK", _build_pick),
    ("PLACE", _build_place),
    ("AFFORDANCE_2D", _build_affordance_2d),
    # ("AFFORDANCE_3D", _build_affordance_3d),  # Disabled: not predicting 3D affordance
    ("GRIPPER_PATH_2D", _build_gripper_path),
]


# =============================================================================
# CoT Builder
# =============================================================================

class CotBuilder:
    """
    Modular CoT builder - each tag is independently validated and formatted.
    Image size is read from annotation's detections, not from config.
    """
    
    def __init__(self, config: Optional[CotConfig] = None):
        self.config = config or CotConfig()
    
    def _make_context(
        self,
        annotation: dict,
        frame_idx: int,
        gripper_2d: Optional[np.ndarray],
        gripper_2d_valid_flag: Optional[bool],
        img_size_wh: Optional[Tuple[int, int]] = None,
    ) -> BuildContext:
        if gripper_2d_valid_flag is None:
            gripper_2d_valid_flag = self._load_gripper_2d_valid_from_h5(annotation)
        return BuildContext(
            annotation=annotation,
            frame_idx=frame_idx,
            gripper_2d=gripper_2d,
            gripper_2d_valid_flag=gripper_2d_valid_flag,
            config=self.config,
            _img_size_wh=img_size_wh,
        )
    
    def _load_gripper_2d_valid_from_h5(self, annotation: dict) -> bool:
        h5_path = annotation.get('h5_path')
        if not h5_path or not os.path.exists(h5_path):
            return False
        try:
            with h5py.File(h5_path, 'r') as f:
                return f.attrs.get('gripper_2d_valid', False)
        except Exception as e:
            print(f"Warning: Could not read gripper_2d_valid from {h5_path}: {e}")
            return False
    
    def build_cot(
        self,
        annotation: dict,
        frame_idx: int,
        gripper_2d: Optional[np.ndarray] = None,
        gripper_2d_valid_flag: Optional[bool] = None,
        img_size_wh: Optional[Tuple[int, int]] = None,
    ) -> str:
        """Build CoT text containing only valid tags."""
        ctx = self._make_context(annotation, frame_idx, gripper_2d, gripper_2d_valid_flag, img_size_wh)
        
        parts = []
        for name, builder_fn in _TAG_BUILDERS:
            try:
                result = builder_fn(ctx)
                if result:
                    parts.append(result)
            except Exception as e:
                print(f"Warning: Error processing {name} tag: {e}")
        
        if not parts:
            return ""
        return "<|cot_start|>" + "\n".join(parts) + "\n<|cot_end|>"
    
    def build_cot_with_details(
        self,
        annotation: dict,
        frame_idx: int,
        gripper_2d: Optional[np.ndarray] = None,
        gripper_2d_valid_flag: Optional[bool] = None,
        img_size_wh: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build CoT text with detailed tag info for visualization/debugging."""
        ctx = self._make_context(annotation, frame_idx, gripper_2d, gripper_2d_valid_flag, img_size_wh)
        
        parts = []
        details = {}
        
        for name, builder_fn in _TAG_BUILDERS:
            try:
                content = builder_fn(ctx)
                if content:
                    parts.append(content)
                details[name] = {"valid": bool(content), "content": content}
            except Exception as e:
                details[name] = {"valid": False, "content": None, "error": str(e)}
        
        cot_text = ""
        if parts:
            cot_text = "<|cot_start|>" + "\n".join(parts) + "\n<|cot_end|>"
        
        return cot_text, details
