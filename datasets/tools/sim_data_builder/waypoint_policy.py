"""Segmented waypoint FSM for Put-on WidowX tasks (absolute EEF + gripper)."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional, Tuple

import numpy as np

from .env_pose import (
    episode_object_names_from_info,
    source_pick_affordance_uv_from_obs,
    source_center_in_base,
    source_grasp_height_base,
    source_grasp_yaw_in_base,
    target_place_height_base,
    target_center_in_base,
    tcp_xyz_euler_from_env,
    unwrap_put_on_env,
)
from .waypoint_params import (
    WaypointParams,
    get_waypoint_params,
    jitter_waypoint_params,
    waypoint_rng,
)


class Phase(IntEnum):
    APPROACH = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    MOVE = 4
    PLACE = 5
    OPEN = 6
    DONE = 7


class WaypointPutOnPolicy:
    def __init__(self, task_key: str, params: Optional[WaypointParams] = None):
        self._task_key = task_key
        self._p_template = params or get_waypoint_params(task_key)
        self._p = self._p_template
        self._env: Any = None
        self._disabled = True
        self._euler = np.zeros(3, dtype=np.float32)
        self._phase = Phase.APPROACH
        self._phase_steps = 0
        self._open_count = 0
        self._src_name = ""
        self._tgt_name = ""
        self._hold = np.zeros(6, dtype=np.float32)
        self._pre_grasp_n: Tuple[int, int, int] = (1, 1, 1)
        self._approach_start_pos = np.zeros(3, dtype=np.float32)
        self._approach_start_yaw = 0.0
        self._fixed_roll_pitch = np.zeros(2, dtype=np.float32)
        self._lift_seg_start = np.zeros(3, dtype=np.float32)
        self._move_seg_start = np.zeros(3, dtype=np.float32)
        self._place_descend_start = np.zeros(3, dtype=np.float32)
        self._carry_z_goal: float = 0.0
        self._grasp_z_offset: float = 0.0
        self._grasp_reached_steps: int = 0
        self._grasp_close_steps: int = 0
        self._episode_id: int = 0
        self._attempt_index: int = 0

    def reset(self) -> None:
        self._env = None
        self._disabled = True
        self._phase = Phase.APPROACH
        self._phase_steps = 0
        self._open_count = 0
        self._grasp_reached_steps = 0
        self._grasp_close_steps = 0
        self._attempt_index = 0

    def setup(self, env: Any, obs: dict, info: dict) -> None:
        self._env = env
        po = unwrap_put_on_env(env)
        if po is None:
            self._disabled = True
            return
        self._disabled = False
        tcp = tcp_xyz_euler_from_env(env)
        self._euler = tcp[3:6].copy()
        self._src_name, self._tgt_name = episode_object_names_from_info(info)
        ep = 0
        if isinstance(info, dict) and info.get("episode_id") is not None:
            ep = int(info["episode_id"])
        self._episode_id = ep
        if isinstance(info, dict) and info.get("attempt_index") is not None:
            self._attempt_index = int(info["attempt_index"])
        if isinstance(info, dict) and info.get("waypoint_jitter_seed") is not None:
            sj = int(info["waypoint_jitter_seed"]) & 0xFFFFFFFFFFFFFFFF
            rng = np.random.default_rng(sj)
        else:
            rng = waypoint_rng(self._task_key, ep)
        self._p = jitter_waypoint_params(self._p_template, rng)
        self._phase = Phase.APPROACH
        self._phase_steps = 0
        self._open_count = 0
        self._pre_grasp_n = self._split_pre_grasp_steps(self._p)
        self._approach_start_pos = tcp[:3].copy()
        self._approach_start_yaw = float(tcp[5])
        self._fixed_roll_pitch = tcp[3:5].copy()
        self._carry_z_goal = 0.0
        self._grasp_reached_steps = 0
        self._grasp_close_steps = 0
        self._grasp_z_offset = float(
            rng.uniform(-self._p.grasp_z_jitter, self._p.grasp_z_jitter)
        )

    def set_init_pose(self, tcp_xyzrpy: np.ndarray) -> None:
        self._hold = np.asarray(tcp_xyzrpy, dtype=np.float32).reshape(6).copy()
        self._euler = self._hold[3:6].copy()

    def _grasp_hover_z(self, grasp_z: float) -> float:
        return float(grasp_z + self._grasp_z_offset + self._p.grasp_hover_dz)

    def _grasp_close_z(self, grasp_z: float) -> float:
        return float(grasp_z + self._grasp_z_offset + self._p.grasp_lower_dz)

    def _smooth_position(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        p = self._p
        d = goal.astype(np.float64) - pos.astype(np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return goal.copy()
        step = p.position_blend_alpha * d
        sn = float(np.linalg.norm(step))
        if sn > p.max_position_step:
            step = step * (p.max_position_step / sn)
        out = pos + step.astype(np.float64)
        return out.astype(np.float32)

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _lerp_yaw(y0: float, y1: float, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        d = WaypointPutOnPolicy._wrap_pi(y1 - y0)
        return float(y0 + t * d)

    def _eef_fixed_rp_yaw(self, yaw: float) -> np.ndarray:
        rp = self._fixed_roll_pitch + np.array(
            [self._p.grasp_roll_offset, self._p.grasp_pitch_offset], dtype=np.float32
        )
        return np.array([rp[0], rp[1], float(yaw)], dtype=np.float32)

    def _grasp_yaw(self, env: Any) -> float:
        p = self._p
        use_major_axis = p.grasp_use_major_axis_yaw or (
            self._task_key == "widowx_put_bridge_objects_on_plate"
        )
        yaw = source_grasp_yaw_in_base(
            env,
            use_major_axis=use_major_axis,
            anisotropy_threshold=p.grasp_major_axis_anisotropy_threshold,
        )
        if self._task_key == "widowx_put_bridge_objects_on_plate":
            orthogonal_yaws = (
                float(yaw),
                float(yaw) + 0.5 * np.pi,
            )
            idx = int(self._attempt_index % len(orthogonal_yaws))
            yaw = orthogonal_yaws[idx]
        return self._wrap_pi(float(yaw) + p.grasp_yaw_offset)

    def _grasp_xy(self, src: np.ndarray) -> np.ndarray:
        return np.array(
            [src[0] + self._p.grasp_x_offset, src[1] + self._p.grasp_y_offset],
            dtype=np.float32,
        )

    def _grasp_reached(self, pos: np.ndarray, close: np.ndarray) -> bool:
        xy_err = float(np.linalg.norm(pos[:2] - close[:2]))
        z_err = abs(float(pos[2] - close[2]))
        return (
            xy_err <= float(self._p.grasp_reach_xy_tol)
            and z_err <= float(self._p.grasp_reach_z_tol)
        )

    @staticmethod
    def _split_pre_grasp_steps(p: WaypointParams) -> Tuple[int, int, int]:
        total = max(12, min(50, int(p.pre_grasp_motion_steps)))
        w = np.asarray(p.pre_grasp_segment_weights, dtype=np.float64).ravel()
        if w.size != 3 or float(np.sum(w)) < 1e-9:
            w = np.array([0.38, 0.34, 0.28], dtype=np.float64)
        w = w / float(np.sum(w))
        raw = np.maximum(1, np.round(total * w).astype(int))
        s = int(raw.sum())
        if s != total:
            j = int(np.argmax(raw))
            raw[j] = max(1, int(raw[j]) + (total - s))
        return int(raw[0]), int(raw[1]), int(raw[2])

    def _update_phase(
        self,
        pos: np.ndarray,
        src: np.ndarray,
        tgt: np.ndarray,
        th: float,
        close: np.ndarray,
    ) -> None:
        p = self._p

        for _ in range(12):
            ph = self._phase
            if ph == Phase.DONE:
                return
            if ph == Phase.APPROACH:
                n1 = self._pre_grasp_n[0]
                if self._phase_steps >= n1:
                    self._phase = Phase.DESCEND
                    self._phase_steps = 0
                    continue
                return

            if ph == Phase.DESCEND:
                n2 = self._pre_grasp_n[1]
                if self._phase_steps >= n2:
                    self._phase = Phase.GRASP
                    self._phase_steps = 0
                    continue
                return

            if ph == Phase.GRASP:
                n3 = self._pre_grasp_n[2]
                if self._phase_steps < n3:
                    self._grasp_reached_steps = 0
                    self._grasp_close_steps = 0
                    return
                if self._grasp_reached(pos, close):
                    self._grasp_reached_steps += 1
                else:
                    self._grasp_reached_steps = 0
                    self._grasp_close_steps = 0
                    return
                if self._grasp_reached_steps < p.grasp_preclose_hold_steps:
                    self._grasp_close_steps = 0
                    return
                self._grasp_close_steps += 1
                if self._grasp_close_steps >= p.grasp_hold_steps:
                    self._phase = Phase.LIFT
                    self._phase_steps = 0
                    continue
                return

            if ph == Phase.LIFT:
                if self._phase_steps >= p.lift_motion_steps:
                    self._phase = Phase.MOVE
                    self._phase_steps = 0
                    continue
                return

            if ph == Phase.MOVE:
                if self._phase_steps >= p.move_motion_steps:
                    self._phase = Phase.OPEN
                    self._phase_steps = 0
                    self._open_count = 0
                    continue
                return

            if ph == Phase.PLACE:
                if self._phase_steps >= p.place_descend_motion_steps:
                    self._phase = Phase.OPEN
                    self._phase_steps = 0
                    self._open_count = 0
                    continue
                return

            if ph == Phase.OPEN:
                self._open_count += 1
                if self._open_count >= p.release_hold_steps:
                    self._phase = Phase.DONE
                    self._phase_steps = 0
                    continue
                return

            return

    def act(self, obs: dict) -> np.ndarray:
        if self._disabled or self._env is None:
            return np.concatenate([self._hold[:6], [1.0]]).astype(np.float32)

        env = self._env
        tcp = tcp_xyz_euler_from_env(env)
        pos = tcp[:3]
        p = self._p

        try:
            src = source_center_in_base(env, use_com=p.grasp_use_source_com)
            tgt = target_center_in_base(env)
            grasp_z = source_grasp_height_base(
                env,
                use_vertical_extent=p.grasp_use_vertical_extent,
                use_com=p.grasp_use_source_com,
                extra_offset=p.grasp_height_offset,
            )
            yaw_grasp = self._grasp_yaw(env)
            th = target_place_height_base(env)
        except RuntimeError:
            return np.concatenate([pos, self._euler, [1.0]]).astype(np.float32)
        g_open = 1.0
        g_close = -1.0

        k = self._phase_steps
        ph = self._phase
        hz = self._grasp_hover_z(grasp_z)
        gz = self._grasp_close_z(grasp_z)
        n1, n2, n3 = self._pre_grasp_n
        grasp_xy = self._grasp_xy(src)
        hover = np.array([grasp_xy[0], grasp_xy[1], hz], dtype=np.float32)
        close = np.array([grasp_xy[0], grasp_xy[1], gz], dtype=np.float32)

        if ph == Phase.APPROACH:
            t = min(1.0, float(k + 1) / max(1, n1))
            ga = np.array([src[0], src[1], src[2] + p.approach_dz], dtype=np.float32)
            sm = ((1.0 - t) * self._approach_start_pos + t * ga).astype(np.float32)
            y = self._lerp_yaw(self._approach_start_yaw, yaw_grasp, t)
            e_cmd = self._eef_fixed_rp_yaw(y)
            grip = g_open
        elif ph == Phase.DESCEND:
            t = min(1.0, float(k + 1) / max(1, n2))
            ga = np.array([src[0], src[1], src[2] + p.approach_dz], dtype=np.float32)
            sm = ((1.0 - t) * ga + t * hover).astype(np.float32)
            e_cmd = self._eef_fixed_rp_yaw(yaw_grasp)
            grip = g_open
        elif ph == Phase.GRASP:
            if k < n3:
                t = min(1.0, float(k + 1) / max(1, n3))
                sm = ((1.0 - t) * hover + t * close).astype(np.float32)
            else:
                sm = close.copy()
            e_cmd = self._eef_fixed_rp_yaw(yaw_grasp)
            reached_now = self._grasp_reached(pos, close)
            close_ready = reached_now and (
                self._grasp_reached_steps + 1 >= p.grasp_preclose_hold_steps
            )
            grip = g_close if close_ready else g_open
        else:
            if ph == Phase.LIFT:
                if k == 0:
                    self._lift_seg_start = pos.copy()
                    self._carry_z_goal = float(pos[2]) + float(p.post_grasp_lift_m)
                nl = max(1, p.lift_motion_steps)
                tseg = min(1.0, float(k + 1) / float(nl))
                z0 = float(self._lift_seg_start[2])
                z1 = self._carry_z_goal
                zm = (1.0 - tseg) * z0 + tseg * z1
                sm = np.array([src[0], src[1], zm], dtype=np.float32)
                e_cmd = self._eef_fixed_rp_yaw(yaw_grasp)
                grip = g_close
            elif ph == Phase.MOVE:
                nm = max(1, p.move_motion_steps)
                tseg = min(1.0, float(k + 1) / float(nm))
                cz = self._carry_z_goal
                mx = (1.0 - tseg) * float(src[0]) + tseg * float(tgt[0])
                my = (1.0 - tseg) * float(src[1]) + tseg * float(tgt[1])
                sm = np.array([mx, my, cz], dtype=np.float32)
                e_cmd = self._eef_fixed_rp_yaw(yaw_grasp)
                grip = g_close
            elif ph == Phase.PLACE:
                release_z = max(float(self._carry_z_goal), float(th + p.place_dz))
                target = np.array([tgt[0], tgt[1], release_z], dtype=np.float32)
                sm = self._smooth_position(pos, target)
                e_cmd = self._eef_fixed_rp_yaw(yaw_grasp)
                grip = g_open
            elif ph == Phase.OPEN:
                release_z = max(float(self._carry_z_goal), float(th + p.place_dz))
                target = np.array([tgt[0], tgt[1], release_z], dtype=np.float32)
                sm = self._smooth_position(pos, target)
                e_cmd = self._eef_fixed_rp_yaw(yaw_grasp)
                grip = g_open
            else:
                place_z = th + max(p.place_dz, 0.02)
                target = np.array([tgt[0], tgt[1], place_z], dtype=np.float32)
                sm = self._smooth_position(pos, target)
                e_cmd = self._eef_fixed_rp_yaw(yaw_grasp)
                grip = g_open

        self._phase_steps += 1
        self._update_phase(pos, src, tgt, th, close)

        return np.concatenate([sm, e_cmd, [grip]]).astype(np.float32)

    @property
    def source_object_name(self) -> str:
        return self._src_name

    @property
    def target_object_name(self) -> str:
        return self._tgt_name

    def pick_affordance_uv_from_obs(self, obs: dict) -> np.ndarray:
        if self._disabled or self._env is None:
            return np.array([np.nan, np.nan], dtype=np.float32)
        p = self._p
        return source_pick_affordance_uv_from_obs(
            self._env,
            obs,
            use_vertical_extent=p.grasp_use_vertical_extent,
            use_com=p.grasp_use_source_com,
            extra_offset=p.grasp_height_offset,
            xy_offset=np.array(
                [p.grasp_x_offset, p.grasp_y_offset], dtype=np.float32
            ),
        )
