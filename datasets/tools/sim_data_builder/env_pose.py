"""ManiSkill put-on env helpers: TCP poses, object positions, and 2D projections."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def _unwrap_env_chain(env: Any) -> Any:
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def unwrap_put_on_env(env: Any) -> Optional[Any]:
    e = env
    for _ in range(24):
        if (
            hasattr(e, "episode_source_obj")
            and e.episode_source_obj is not None
            and hasattr(e, "episode_target_obj")
            and e.episode_target_obj is not None
        ):
            return e
        if hasattr(e, "unwrapped"):
            e = e.unwrapped
            continue
        if hasattr(e, "env"):
            e = e.env
            continue
        break
    return None


def tcp_xyz_euler_from_env(env: Any) -> np.ndarray:
    u = _unwrap_env_chain(env)
    tcp_world = u.tcp.pose
    base_world = u.agent.robot.pose
    tcp_wrt_base = base_world.inv() * tcp_world
    q_wxyz = tcp_wrt_base.q
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    euler = R.from_quat(q_xyzw).as_euler("xyz")
    return np.concatenate([tcp_wrt_base.p, euler]).astype(np.float32)


def _camera_mats_from_obs(
    obs: dict, camera_name: str = "3rd_view_camera"
) -> Tuple[np.ndarray, np.ndarray]:
    camera_param = obs.get("camera_param", {})
    cam = camera_param.get(camera_name)
    if cam is None:
        raise KeyError(f"camera_param[{camera_name!r}] not found in observation")
    intrinsic = np.asarray(cam["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(cam["extrinsic_cv"], dtype=np.float64)
    return intrinsic, extrinsic


def project_world_point_to_image(
    obs: dict,
    point_world: np.ndarray,
    camera_name: str = "3rd_view_camera",
) -> np.ndarray:
    intrinsic, extrinsic = _camera_mats_from_obs(obs, camera_name)
    pw = np.ones(4, dtype=np.float64)
    pw[:3] = np.asarray(point_world, dtype=np.float64).reshape(3)
    pc = extrinsic @ pw
    if not np.isfinite(pc[:3]).all() or pc[2] <= 1e-8:
        return np.array([np.nan, np.nan], dtype=np.float32)
    if intrinsic.shape == (3, 3):
        uvw = intrinsic @ pc[:3]
    elif intrinsic.shape == (3, 4):
        uvw = intrinsic @ np.array([pc[0], pc[1], pc[2], 1.0], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")
    if not np.isfinite(uvw).all() or abs(float(uvw[2])) <= 1e-8:
        return np.array([np.nan, np.nan], dtype=np.float32)
    u = float(uvw[0] / uvw[2])
    v = float(uvw[1] / uvw[2])
    image = obs.get("image", {}).get(camera_name, {}).get("rgb")
    if image is None:
        return np.array([np.nan, np.nan], dtype=np.float32)
    h, w = image.shape[:2]
    if not (0.0 <= u < float(w) and 0.0 <= v < float(h)):
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([u, v], dtype=np.float32)


def tcp_center_uv_from_obs(
    env: Any,
    obs: dict,
    camera_name: str = "3rd_view_camera",
) -> np.ndarray:
    u = _unwrap_env_chain(env)
    point_world = np.asarray(u.tcp.pose.p, dtype=np.float64)
    return project_world_point_to_image(obs, point_world, camera_name=camera_name)


def _rotmat_from_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    return R.from_quat(q_xyzw).as_matrix()


def _vector_world_to_base(env: Any, vec_world: np.ndarray) -> np.ndarray:
    u = _unwrap_env_chain(env)
    rb = u.agent.robot.pose
    return _rotmat_from_wxyz(rb.q).T @ np.asarray(vec_world, dtype=np.float64).reshape(3)


def _point_base_to_world(env: Any, point_base: np.ndarray) -> np.ndarray:
    u = _unwrap_env_chain(env)
    rb = u.agent.robot.pose
    pb = np.asarray(point_base, dtype=np.float64).reshape(3)
    return (_rotmat_from_wxyz(rb.q) @ pb + np.asarray(rb.p, dtype=np.float64)).astype(
        np.float64
    )


def actor_pose_in_base(env: Any, actor: Any, use_com: bool = False):
    u = _unwrap_env_chain(env)
    rb = u.agent.robot.pose
    pose_world = actor.pose
    if use_com and getattr(actor, "cmass_local_pose", None) is not None:
        pose_world = pose_world.transform(actor.cmass_local_pose)
    return rb.inv() * pose_world


def actor_center_in_base(env: Any, actor: Any, use_com: bool = False) -> np.ndarray:
    pw = actor_pose_in_base(env, actor, use_com=use_com)
    return np.array(pw.p, dtype=np.float32)


def actor_euler_xyz_in_base(env: Any, actor: Any, use_com: bool = False) -> np.ndarray:
    pose = actor_pose_in_base(env, actor, use_com=use_com)
    q_wxyz = pose.q
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R.from_quat(q_xyzw).as_euler("xyz").astype(np.float32)


def actor_bbox_extents_in_base(
    env: Any, actor: Any, bbox_world: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    if bbox_world is None:
        return None
    b = np.asarray(bbox_world, dtype=np.float64).ravel()
    if b.size < 3:
        return None
    bbox_base = _vector_world_to_base(env, b[:3])
    rot_base = _rotmat_from_wxyz(actor_pose_in_base(env, actor).q)
    return np.abs(rot_base.T @ bbox_base)


def actor_vertical_half_extent_in_base(
    env: Any, actor: Any, bbox_world: Optional[np.ndarray]
) -> Optional[float]:
    local_extent = actor_bbox_extents_in_base(env, actor, bbox_world)
    if local_extent is None:
        return None
    rot_base = _rotmat_from_wxyz(actor_pose_in_base(env, actor).q)
    return float(0.5 * np.sum(np.abs(rot_base[2]) * local_extent))


def actor_major_axis_yaw_in_base(
    env: Any,
    actor: Any,
    bbox_world: Optional[np.ndarray],
    anisotropy_threshold: float = 1.2,
) -> Optional[float]:
    local_extent = actor_bbox_extents_in_base(env, actor, bbox_world)
    if local_extent is None:
        return None
    rot_base = _rotmat_from_wxyz(actor_pose_in_base(env, actor).q)
    horizontal_span = np.linalg.norm(rot_base[:2, :], axis=0) * local_extent
    order = np.argsort(horizontal_span)
    best = int(order[-1])
    second = float(horizontal_span[order[-2]]) if horizontal_span.size >= 2 else 0.0
    best_span = float(horizontal_span[best])
    if best_span <= 1e-8:
        return None
    if second > 1e-8 and best_span / second < float(anisotropy_threshold):
        return None
    axis = rot_base[:2, best]
    return float(np.arctan2(axis[1], axis[0]))


def source_target_euler_in_base(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    po = unwrap_put_on_env(env)
    if po is None:
        raise RuntimeError("put-on env not found")
    es = actor_euler_xyz_in_base(env, po.episode_source_obj)
    et = actor_euler_xyz_in_base(env, po.episode_target_obj)
    return es, et


def source_target_centers_in_base(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    po = unwrap_put_on_env(env)
    if po is None:
        raise RuntimeError("put-on env not found")
    s = actor_center_in_base(env, po.episode_source_obj)
    t = actor_center_in_base(env, po.episode_target_obj)
    return s, t


def source_center_in_base(env: Any, use_com: bool = False) -> np.ndarray:
    po = unwrap_put_on_env(env)
    if po is None:
        raise RuntimeError("put-on env not found")
    return actor_center_in_base(env, po.episode_source_obj, use_com=use_com)


def target_center_in_base(env: Any, use_com: bool = False) -> np.ndarray:
    po = unwrap_put_on_env(env)
    if po is None:
        raise RuntimeError("put-on env not found")
    return actor_center_in_base(env, po.episode_target_obj, use_com=use_com)


def source_grasp_yaw_in_base(
    env: Any,
    use_major_axis: bool = False,
    anisotropy_threshold: float = 1.2,
) -> float:
    po = unwrap_put_on_env(env)
    if po is None:
        raise RuntimeError("put-on env not found")
    if use_major_axis:
        yaw = actor_major_axis_yaw_in_base(
            env,
            po.episode_source_obj,
            getattr(po, "episode_source_obj_bbox_world", None),
            anisotropy_threshold=anisotropy_threshold,
        )
        if yaw is not None:
            return yaw
    return float(actor_euler_xyz_in_base(env, po.episode_source_obj)[2])


def source_grasp_height_base(
    env: Any,
    use_vertical_extent: bool = False,
    use_com: bool = False,
    extra_offset: float = 0.0,
) -> float:
    po = unwrap_put_on_env(env)
    if po is None:
        raise RuntimeError("put-on env not found")
    s_base = actor_pose_in_base(env, po.episode_source_obj, use_com=use_com)
    z0 = float(np.array(s_base.p)[2])
    bbox = getattr(po, "episode_source_obj_bbox_world", None)
    if bbox is not None:
        b = np.asarray(bbox, dtype=np.float64).ravel()
        if b.size >= 3:
            if use_vertical_extent:
                half_extent = actor_vertical_half_extent_in_base(
                    env, po.episode_source_obj, bbox
                )
                if half_extent is not None:
                    z0 = z0 + half_extent
            else:
                z0 = z0 + float(b[2]) * 0.5
    return float(z0 + extra_offset)


def source_grasp_position_base(
    env: Any,
    use_vertical_extent: bool = False,
    use_com: bool = False,
    extra_offset: float = 0.0,
    xy_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    src = source_center_in_base(env, use_com=use_com).astype(np.float32)
    out = np.array(
        [
            float(src[0]),
            float(src[1]),
            float(
                source_grasp_height_base(
                    env,
                    use_vertical_extent=use_vertical_extent,
                    use_com=use_com,
                    extra_offset=extra_offset,
                )
            ),
        ],
        dtype=np.float32,
    )
    if xy_offset is not None:
        xy = np.asarray(xy_offset, dtype=np.float32).reshape(2)
        out[:2] = out[:2] + xy
    return out


def source_pick_affordance_uv_from_obs(
    env: Any,
    obs: dict,
    camera_name: str = "3rd_view_camera",
    use_vertical_extent: bool = False,
    use_com: bool = False,
    extra_offset: float = 0.0,
    xy_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    point_base = source_grasp_position_base(
        env,
        use_vertical_extent=use_vertical_extent,
        use_com=use_com,
        extra_offset=extra_offset,
        xy_offset=xy_offset,
    )
    point_world = _point_base_to_world(env, point_base)
    return project_world_point_to_image(obs, point_world, camera_name=camera_name)


def target_place_height_base(
    env: Any,
    use_vertical_extent: bool = False,
    use_com: bool = False,
    extra_offset: float = 0.0,
) -> float:
    po = unwrap_put_on_env(env)
    if po is None:
        raise RuntimeError("put-on env not found")
    t_base = actor_pose_in_base(env, po.episode_target_obj, use_com=use_com)
    z0 = float(np.array(t_base.p)[2])
    bbox = getattr(po, "episode_target_obj_bbox_world", None)
    if bbox is not None:
        b = np.asarray(bbox, dtype=np.float64).ravel()
        if b.size >= 3:
            if use_vertical_extent:
                half_extent = actor_vertical_half_extent_in_base(
                    env, po.episode_target_obj, bbox
                )
                if half_extent is not None:
                    z0 = z0 + half_extent
            else:
                z0 = z0 + float(b[2]) * 0.5
    return float(z0 + extra_offset)


def episode_object_names_from_info(info: dict) -> Tuple[str, str]:
    src = str(info.get("episode_source_obj_name", "") or "")
    tgt = str(info.get("episode_target_obj_name", "") or "")
    return src, tgt
