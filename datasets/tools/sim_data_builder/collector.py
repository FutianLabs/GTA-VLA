"""
Collector: run rollout in SimplerEnv WidowX environments and return episode data.

Designed to be plugged with arbitrary policies. Ships with a NullPolicy
(holds initial pose) for smoke-testing the pipeline.

Usage:
    from sim_data_builder.collector import collect_episode, NullPolicy
    data = collect_episode("widowx_carrot_on_plate", episode_id=0, policy=NullPolicy())
"""

import gc
import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from .env_pose import (
    episode_object_names_from_info,
    source_pick_affordance_uv_from_obs,
    tcp_center_uv_from_obs,
    tcp_xyz_euler_from_env,
)


# ── Policy interface ────────────────────────────────────────────────────────

class Policy(Protocol):
    """Minimal policy interface: obs → action (7-D absolute EEF)."""

    def reset(self) -> None: ...

    def act(self, obs: dict) -> np.ndarray:
        """Return action [x, y, z, roll, pitch, yaw, gripper].
        Gripper: 1.0 = open, -1.0 = close (SimplerEnv convention).
        """
        ...


class NullPolicy:
    """Hold initial pose, gripper open. For smoke-testing only."""

    def __init__(self):
        self._init_pose = None

    def reset(self) -> None:
        self._init_pose = None

    def set_init_pose(self, tcp_xyzrpy: np.ndarray) -> None:
        self._init_pose = tcp_xyzrpy.copy()

    def act(self, obs: dict) -> np.ndarray:
        if self._init_pose is None:
            raise RuntimeError("NullPolicy: call set_init_pose() before act()")
        return np.concatenate([self._init_pose, [1.0]]).astype(np.float32)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _gripper_state_from_obs(obs: dict) -> float:
    """Return a binary gripper state in {0, 1}. 1 = open, 0 = closed."""
    qpos = obs["agent"]["qpos"]
    # WidowX: last 2 joints are left/right finger, range [0.015, 0.037]
    finger_pos = qpos[-2]  # left finger
    low, high = 0.015, 0.037
    openness = float(np.clip((finger_pos - low) / (high - low), 0.0, 1.0))
    return float(openness >= 0.5)


def _compute_delta_actions(
    proprio: np.ndarray, gripper_states: np.ndarray
) -> np.ndarray:
    """Compute Bridge-style delta actions from absolute proprio trajectory.

    Args:
        proprio: [T, 6] absolute tcp xyz + euler_xyz.
        gripper_states: [T] gripper openness in [0, 1].

    Returns:
        action: [T, 7] delta_xyz(3) + delta_euler(3) + gripper(1).
    """
    T_len = proprio.shape[0]
    delta = np.zeros((T_len, 6), dtype=np.float32)
    if T_len > 1:
        delta[:-1] = proprio[1:] - proprio[:-1]
        delta[-1] = delta[-2] if T_len > 1 else 0.0  # repeat last delta
    action = np.concatenate(
        [delta, gripper_states[:, None].astype(np.float32)], axis=-1
    )
    return action


def _waypoint_jitter_seed(task_key: str, episode_id: int) -> int:
    digest = hashlib.sha256(f"{task_key}:{int(episode_id)}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _pick_affordance_uv_from_obs(env: Any, obs: dict, policy: Policy) -> np.ndarray:
    if hasattr(policy, "pick_affordance_uv_from_obs"):
        uv = policy.pick_affordance_uv_from_obs(obs)
        return np.asarray(uv, dtype=np.float32).reshape(2)
    return source_pick_affordance_uv_from_obs(env, obs)


def _collect_single_attempt(
    task_key: str,
    episode_id: int,
    policy: Policy,
    cfg: "TaskConfig",
    max_steps: int,
    env_modifiers: Optional[List[str]],
    modifier_params: Optional[Dict],
    attempt_index: int,
    attempt_count: int,
) -> "EpisodeData":
    env = _make_simpler_env(task_key, cfg)
    try:
        policy.reset()

        reset_options: Dict[str, Any] = {"obj_init_options": {"episode_id": episode_id}}
        if env_modifiers:
            reset_options["env_modifiers"] = list(env_modifiers)
        if modifier_params:
            reset_options["modifier_params"] = dict(modifier_params)

        obs, info = env.reset(seed=episode_id, options=reset_options)

        instruction = ""
        unwrapped = env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "get_language_instruction"):
            instruction = unwrapped.get_language_instruction()

        all_image_0: List[np.ndarray] = []
        all_image_3: List[np.ndarray] = []
        all_tcp: List[np.ndarray] = []
        all_gripper: List[float] = []
        all_gripper_2d: List[np.ndarray] = []
        all_pick_affordance_2d: List[np.ndarray] = []
        total_reward = 0.0
        success = False

        all_image_0.append(obs["image"]["3rd_view_camera"]["rgb"].copy())
        if "wrist_camera" in obs["image"]:
            all_image_3.append(obs["image"]["wrist_camera"]["rgb"].copy())
        tcp = tcp_xyz_euler_from_env(env)
        all_tcp.append(tcp)
        all_gripper.append(_gripper_state_from_obs(obs))
        all_gripper_2d.append(tcp_center_uv_from_obs(env, obs))

        src_name, tgt_name = episode_object_names_from_info(info)
        if hasattr(policy, "set_init_pose"):
            policy.set_init_pose(tcp)
        if hasattr(policy, "setup"):
            info_policy = dict(info) if isinstance(info, dict) else info
            if isinstance(info_policy, dict):
                info_policy["episode_id"] = episode_id
                info_policy["attempt_index"] = attempt_index
                info_policy["attempt_count"] = attempt_count
                info_policy["waypoint_jitter_seed"] = _waypoint_jitter_seed(
                    task_key, episode_id
                )
            policy.setup(env, obs, info_policy)
        if hasattr(policy, "source_object_name") and hasattr(policy, "target_object_name"):
            src_name = getattr(policy, "source_object_name", "") or src_name
            tgt_name = getattr(policy, "target_object_name", "") or tgt_name
        pick_affordance_2d_ref = _pick_affordance_uv_from_obs(env, obs, policy)
        all_pick_affordance_2d.append(pick_affordance_2d_ref.copy())

        for _ in range(max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            all_image_0.append(obs["image"]["3rd_view_camera"]["rgb"].copy())
            if "wrist_camera" in obs["image"]:
                all_image_3.append(obs["image"]["wrist_camera"]["rgb"].copy())

            tcp = tcp_xyz_euler_from_env(env)
            all_tcp.append(tcp)
            all_gripper.append(_gripper_state_from_obs(obs))
            all_gripper_2d.append(tcp_center_uv_from_obs(env, obs))
            all_pick_affordance_2d.append(pick_affordance_2d_ref.copy())

            total_reward += float(reward)
            if info.get("success", False):
                success = True

            if terminated or truncated:
                break

        images_0 = np.stack(all_image_0, axis=0)
        images_3 = np.stack(all_image_3, axis=0) if all_image_3 else None
        proprio_arr = np.stack(all_tcp, axis=0).astype(np.float32)
        gripper_arr = np.array(all_gripper, dtype=np.float32)
        gripper_2d_arr = np.stack(all_gripper_2d, axis=0).astype(np.float32)
        pick_affordance_2d_arr = np.stack(all_pick_affordance_2d, axis=0).astype(
            np.float32
        )
        action_arr = _compute_delta_actions(proprio_arr, gripper_arr)

        wrist_valid = images_3 is not None and images_3.shape[0] > 0
        gripper_2d_valid = bool(np.isfinite(gripper_2d_arr).all())
        pick_affordance_2d_valid = bool(np.isfinite(pick_affordance_2d_arr).all())

        return EpisodeData(
            images_0=images_0,
            images_3=images_3,
            proprio=proprio_arr,
            action=action_arr,
            gripper_position=gripper_2d_arr,
            pick_affordance_position=pick_affordance_2d_arr,
            gripper_2d_valid=gripper_2d_valid,
            pick_affordance_2d_valid=pick_affordance_2d_valid,
            instruction=instruction,
            wrist_view_valid=wrist_valid,
            task_key=cfg.task_key,
            task_group=cfg.task_group,
            env_id=cfg.env_id,
            episode_id=episode_id,
            success=success,
            total_reward=total_reward,
            num_steps=images_0.shape[0],
            source_object_name=src_name,
            target_object_name=tgt_name,
        )
    finally:
        env.close()
        del env
        gc.collect()


# ── Episode data container ──────────────────────────────────────────────────

@dataclass
class EpisodeData:
    images_0: np.ndarray          # [T, H, W, 3] uint8
    images_3: Optional[np.ndarray]  # [T, H, W, 3] uint8 or None
    proprio: np.ndarray           # [T, 6+] float32
    action: np.ndarray            # [T, 7] float32
    gripper_position: Optional[np.ndarray]  # [T, 2] float32 or None
    pick_affordance_position: Optional[np.ndarray]  # [T, 2] float32 or None
    gripper_2d_valid: bool
    pick_affordance_2d_valid: bool
    instruction: str
    wrist_view_valid: bool
    task_key: str
    task_group: str
    env_id: str
    episode_id: int
    success: bool
    total_reward: float
    num_steps: int
    source_object_name: str = ""
    target_object_name: str = ""


# ── GPU env (before simpler_env import; use in fresh child process when switching GPUs) ──

def apply_sim_gpu_env(physical_gpu_id: int) -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu_id))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ.setdefault("DISPLAY", "")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("SAPIEN_RENDERER", "offscreen")


def ensure_simpler_env_importable() -> None:
    candidates = []
    env_root = os.environ.get("SIMPLER_DIR", "").strip()
    if env_root:
        base = Path(env_root).expanduser()
        candidates.extend([base, base / "ManiSkill2_real2sim"])
    root = Path(__file__).resolve().parents[3]
    candidates.extend([
        root / "Isaac-GR00T" / "external_dependencies" / "SimplerEnv",
        root / "Isaac-GR00T" / "external_dependencies" / "SimplerEnv" / "ManiSkill2_real2sim",
    ])
    for path in reversed(candidates):
        if path.is_dir():
            p = str(path)
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)


def _make_simpler_env(task_key: str, cfg: "TaskConfig"):
    import simpler_env

    env_name = task_key if task_key in getattr(simpler_env, "ENVIRONMENTS", []) else cfg.env_id
    return simpler_env.make(
        env_name,
        renderer_kwargs={"offscreen_only": True},
        **cfg.extra_env_kwargs,
    )


# ── Main collector ──────────────────────────────────────────────────────────

def collect_episode(
    task_key: str,
    episode_id: int,
    policy: Policy,
    max_steps: int = 120,
    renderer_kwargs: Optional[Dict] = None,
    env_modifiers: Optional[List[str]] = None,
    modifier_params: Optional[Dict] = None,
    attempt_index: Optional[int] = None,
    attempt_count: Optional[int] = None,
) -> EpisodeData:
    """Collect one episode from a WidowX SimplerEnv environment.

    Args:
        task_key: Key from task_config (e.g. "widowx_carrot_on_plate").
        episode_id: Episode seed for deterministic reset.
        policy: Policy object implementing act(obs) -> action.
        max_steps: Maximum rollout steps.
        renderer_kwargs: Reserved for compatibility; direct env creation ignores it.
        env_modifiers: Optional list of runtime environment modifiers.
        modifier_params: Optional runtime modifier parameter dict.
        attempt_index: Optional fixed grasp attempt index for same-seed probing.
        attempt_count: Optional total attempt count paired with attempt_index.

    Returns:
        EpisodeData with all collected fields.
    """
    ensure_simpler_env_importable()
    import simpler_env  # noqa: F401  # ensure custom envs are registered
    from .task_config import get_task

    cfg = get_task(task_key)

    if attempt_index is not None:
        total_attempts = (
            max(1, int(attempt_count))
            if attempt_count is not None
            else (2 if task_key == "widowx_put_bridge_objects_on_plate" else 1)
        )
        return _collect_single_attempt(
            task_key=task_key,
            episode_id=episode_id,
            policy=policy,
            cfg=cfg,
            max_steps=max_steps,
            env_modifiers=env_modifiers,
            modifier_params=modifier_params,
            attempt_index=int(attempt_index),
            attempt_count=total_attempts,
        )

    attempt_count = 2 if task_key == "widowx_put_bridge_objects_on_plate" else 1
    last_data: Optional[EpisodeData] = None
    for attempt_index in range(attempt_count):
        data = _collect_single_attempt(
            task_key=task_key,
            episode_id=episode_id,
            policy=policy,
            cfg=cfg,
            max_steps=max_steps,
            env_modifiers=env_modifiers,
            modifier_params=modifier_params,
            attempt_index=attempt_index,
            attempt_count=attempt_count,
        )
        last_data = data
        if data.success:
            return data
    if last_data is None:
        raise RuntimeError(f"failed to collect episode for {task_key} episode_id={episode_id}")
    return last_data
