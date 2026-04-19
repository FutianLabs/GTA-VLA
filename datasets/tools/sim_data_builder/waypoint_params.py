"""Per-task waypoint offsets for scripted put-on policies (base frame, meters)."""

import hashlib
from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np


@dataclass
class WaypointParams:
    approach_dz: float = 0.11
    grasp_hover_dz: float = 0.022
    grasp_lower_dz: float = -0.010
    grasp_z_jitter: float = 0.003
    grasp_x_offset: float = 0.0
    grasp_y_offset: float = 0.0
    grasp_height_offset: float = 0.0
    grasp_use_vertical_extent: bool = False
    grasp_use_source_com: bool = False
    grasp_use_major_axis_yaw: bool = False
    grasp_major_axis_anisotropy_threshold: float = 1.2
    grasp_roll_offset: float = 0.0
    grasp_pitch_offset: float = 0.0
    lift_dz: float = 0.06
    post_grasp_lift_m: float = 0.08
    place_dz: float = 0.025
    pos_tol: float = 0.028
    max_steps_per_phase: int = 52
    grasp_preclose_hold_steps: int = 2
    grasp_hold_steps: int = 2
    grasp_reach_xy_tol: float = 0.004
    grasp_reach_z_tol: float = 0.004
    grasp_max_settle_steps: int = 24
    release_hold_steps: int = 12
    done_hold_steps: int = 8
    position_blend_alpha: float = 0.26
    max_position_step: float = 0.038
    euler_blend_alpha: float = 0.48
    grasp_yaw_offset: float = 0.0
    place_yaw_offset: float = 0.0
    pre_grasp_motion_steps: int = 38
    pre_grasp_segment_weights: Tuple[float, float, float] = (0.38, 0.34, 0.28)
    lift_motion_steps: int = 12
    move_motion_steps: int = 28
    place_descend_motion_steps: int = 20


DEFAULT_PARAMS = WaypointParams()


def jitter_waypoint_params(p: WaypointParams, rng: np.random.Generator) -> WaypointParams:
    def rf(x: float, rel: float, lo: float, hi: float) -> float:
        m = float(rng.uniform(1.0 - rel, 1.0 + rel))
        return float(np.clip(x * m, lo, hi))

    def ri(x: int, rel: float, lo: int, hi: int) -> int:
        m = float(rng.uniform(1.0 - rel, 1.0 + rel))
        return int(np.clip(round(x * m), lo, hi))

    speed_scale = float(rng.uniform(1.0, 2.0))

    return replace(
        p,
        approach_dz=rf(p.approach_dz, 0.10, 0.09, 0.15),
        grasp_hover_dz=rf(p.grasp_hover_dz, 0.12, 0.016, 0.032),
        grasp_z_jitter=rf(p.grasp_z_jitter, 0.15, 0.001, 0.008),
        post_grasp_lift_m=rf(p.post_grasp_lift_m, 0.14, 0.055, 0.11),
        place_dz=rf(p.place_dz, 0.12, 0.015, 0.045),
        grasp_hold_steps=ri(p.grasp_hold_steps, 0.4, 1, 4),
        pre_grasp_motion_steps=int(np.clip(round(ri(p.pre_grasp_motion_steps, 0.14, 22, 48) / speed_scale), 12, 48)),
        lift_motion_steps=int(np.clip(round(ri(p.lift_motion_steps, 0.16, 6, 22) / speed_scale), 4, 22)),
        move_motion_steps=int(np.clip(round(ri(p.move_motion_steps, 0.14, 18, 40) / speed_scale), 10, 40)),
        place_descend_motion_steps=int(np.clip(round(ri(p.place_descend_motion_steps, 0.14, 12, 30) / speed_scale), 6, 30)),
        release_hold_steps=ri(p.release_hold_steps, 0.12, 8, 16),
        position_blend_alpha=float(np.clip(rf(p.position_blend_alpha, 0.14, 0.17, 0.36) * (0.9 + 0.2 * speed_scale), 0.17, 0.55)),
        max_position_step=float(np.clip(rf(p.max_position_step, 0.12, 0.026, 0.048) * speed_scale, 0.026, 0.08)),
    )


_TASK_OVERRIDES: Dict[str, WaypointParams] = {
    "widowx_carrot_on_plate": WaypointParams(
        approach_dz=0.12,
        grasp_hover_dz=0.024,
        grasp_lower_dz=-0.009,
        grasp_z_jitter=0.003,
        lift_dz=0.06,
        place_dz=0.05,
        pos_tol=0.026,
        grasp_hold_steps=2,
        max_steps_per_phase=52,
        position_blend_alpha=0.24,
        max_position_step=0.036,
        euler_blend_alpha=0.5,
    ),
    "widowx_spoon_on_towel": WaypointParams(
        approach_dz=0.10,
        grasp_hover_dz=0.022,
        grasp_lower_dz=-0.008,
        grasp_z_jitter=0.0025,
        lift_dz=0.055,
        place_dz=0.02,
        pos_tol=0.03,
        position_blend_alpha=0.25,
        max_position_step=0.036,
        euler_blend_alpha=0.48,
    ),
    "widowx_stack_cube": WaypointParams(
        approach_dz=0.10,
        grasp_hover_dz=0.021,
        grasp_lower_dz=-0.009,
        grasp_z_jitter=0.002,
        lift_dz=0.055,
        place_dz=0.038,
        pos_tol=0.024,
        position_blend_alpha=0.25,
        euler_blend_alpha=0.48,
    ),
    "widowx_put_eggplant_in_basket": WaypointParams(
        approach_dz=0.12,
        grasp_hover_dz=0.026,
        grasp_lower_dz=-0.011,
        lift_dz=0.065,
        place_dz=0.05,
        pos_tol=0.032,
        max_steps_per_phase=55,
        position_blend_alpha=0.24,
        euler_blend_alpha=0.46,
    ),
    "widowx_put_bridge_objects_on_plate": WaypointParams(
        approach_dz=0.11,
        grasp_hover_dz=0.024,
        grasp_lower_dz=-0.010,
        lift_dz=0.06,
        place_dz=0.045,
        pos_tol=0.028,
        position_blend_alpha=0.25,
        euler_blend_alpha=0.48,
    ),
    "widowx_carrot_on_plate_multi_object": WaypointParams(
        approach_dz=0.12,
        grasp_hover_dz=0.024,
        grasp_lower_dz=-0.009,
        lift_dz=0.06,
        place_dz=0.045,
        pos_tol=0.028,
        position_blend_alpha=0.24,
        euler_blend_alpha=0.5,
    ),
    "widowx_spoon_on_towel_multi_object": WaypointParams(
        approach_dz=0.10,
        grasp_hover_dz=0.022,
        grasp_lower_dz=-0.008,
        lift_dz=0.055,
        place_dz=0.02,
        pos_tol=0.03,
        position_blend_alpha=0.25,
        euler_blend_alpha=0.48,
    ),
    "widowx_put_eggplant_in_basket_multi_object": WaypointParams(
        approach_dz=0.12,
        grasp_hover_dz=0.026,
        grasp_lower_dz=-0.011,
        lift_dz=0.065,
        place_dz=0.05,
        pos_tol=0.032,
        max_steps_per_phase=55,
        position_blend_alpha=0.24,
        euler_blend_alpha=0.46,
    ),
    "widowx_stack_cube_multi_object": WaypointParams(
        approach_dz=0.10,
        grasp_hover_dz=0.021,
        grasp_lower_dz=-0.009,
        lift_dz=0.055,
        place_dz=0.038,
        pos_tol=0.024,
        position_blend_alpha=0.25,
        euler_blend_alpha=0.48,
    ),
    "widowx_carrot_on_plate_layout_distractor": WaypointParams(
        approach_dz=0.12,
        grasp_hover_dz=0.026,
        grasp_lower_dz=-0.009,
        lift_dz=0.065,
        place_dz=0.045,
        pos_tol=0.028,
        position_blend_alpha=0.23,
        max_position_step=0.034,
        euler_blend_alpha=0.48,
    ),
    "widowx_spoon_on_towel_layout_distractor": WaypointParams(
        approach_dz=0.11,
        grasp_hover_dz=0.023,
        grasp_lower_dz=-0.008,
        lift_dz=0.06,
        place_dz=0.02,
        pos_tol=0.03,
        position_blend_alpha=0.24,
        euler_blend_alpha=0.48,
    ),
    "widowx_put_eggplant_in_basket_layout_distractor": WaypointParams(
        approach_dz=0.13,
        grasp_hover_dz=0.028,
        grasp_lower_dz=-0.011,
        lift_dz=0.07,
        place_dz=0.05,
        pos_tol=0.032,
        max_steps_per_phase=55,
        position_blend_alpha=0.23,
        euler_blend_alpha=0.45,
    ),
    "widowx_stack_cube_layout_distractor": WaypointParams(
        approach_dz=0.11,
        grasp_hover_dz=0.022,
        grasp_lower_dz=-0.009,
        lift_dz=0.06,
        place_dz=0.038,
        pos_tol=0.024,
        position_blend_alpha=0.24,
        euler_blend_alpha=0.48,
    ),
}


def get_waypoint_params(task_key: str) -> WaypointParams:
    return _TASK_OVERRIDES.get(task_key, DEFAULT_PARAMS)


def waypoint_rng(task_key: str, episode_id: int) -> np.random.Generator:
    h = hashlib.sha256(f"{task_key}:{int(episode_id)}".encode()).digest()
    seed = int.from_bytes(h[:8], "little")
    return np.random.default_rng(seed)
