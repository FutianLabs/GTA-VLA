#!/usr/bin/env python3
"""
Main entry point: collect WidowX sim episodes → Bridge-compatible HDF5 + meta.

Examples (在 x-vla-main 根目录执行；数据目录与 bridge_enhance 对齐)：
    # Single task, waypoint policy (default)
    python -m datasets.tools.sim_data_builder.run \
        --task widowx_carrot_on_plate \
        --episodes 1 \
        --output_dir ../data/openX/x-vla/bridge_enhance

    # Null policy (smoke test)
    python -m datasets.tools.sim_data_builder.run \
        --task widowx_carrot_on_plate --policy null --episodes 1 \
        --output_dir ../data/openX/x-vla/bridge_enhance

    # Default: all rollouts saved. Production: only successes:
    python -m datasets.tools.sim_data_builder.run --task ... --success_only --output_dir ../data/openX/x-vla/bridge_enhance

    # Debug: save all rollouts + third-view & wrist MP4:
    python -m datasets.tools.sim_data_builder.run --task ... --save_video --output_dir ../data/openX/x-vla/bridge_enhance

    # Parallel rollouts (use most CPUs; omit --parallel with --success_quota to auto-size):
    python -m datasets.tools.sim_data_builder.run --task widowx_carrot_on_plate \\
        --episodes 32 --parallel $(nproc) --output_dir ../data/openX/x-vla/bridge_enhance
"""

import argparse
import json
import multiprocessing as mp
from multiprocessing import get_context
import os
import subprocess
import sys
import time

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bridge_writer import save_episode_to_h5
from .collector import (
    EpisodeData,
    NullPolicy,
    apply_sim_gpu_env,
    collect_episode,
    ensure_simpler_env_importable,
)
from .gen_meta import generate_meta
from .task_config import ALL_TASKS, get_group, get_task, list_all_task_keys
from .waypoint_policy import WaypointPutOnPolicy


def _write_episode(data: EpisodeData, output_dir: Path, global_idx: int) -> str:
    """Write one EpisodeData → HDF5, return the path."""
    fname = f"episode_{global_idx:06d}.hdf5"
    path = output_dir / fname
    save_episode_to_h5(
        output_path=str(path),
        images_0=data.images_0,
        proprio=data.proprio,
        action=data.action,
        instruction=data.instruction,
        images_3=data.images_3,
        gripper_position=data.gripper_position,
        pick_affordance_position=data.pick_affordance_position,
        wrist_view_valid=data.wrist_view_valid,
        gripper_2d_valid=data.gripper_2d_valid,
        pick_affordance_2d_valid=data.pick_affordance_2d_valid,
        instruction_source="sim",
        task_key=data.task_key,
        task_group=data.task_group,
        env_id=data.env_id,
    )
    return str(path)


def _write_manifest_entry(
    data: EpisodeData,
    global_idx: int,
    h5_path: str,
    video_3rd: str = "",
    video_wrist: str = "",
    variant_label: str = "",
    env_modifiers: Optional[List[str]] = None,
    modifier_params: Optional[Dict[str, Any]] = None,
) -> dict:
    d = {
        "global_episode_id": global_idx,
        "env_episode_id": data.episode_id,
        "task_key": data.task_key,
        "task_group": data.task_group,
        "env_id": data.env_id,
        "instruction": data.instruction,
        "wrist_view_valid": data.wrist_view_valid,
        "gripper_2d_valid": data.gripper_2d_valid,
        "pick_affordance_2d_valid": data.pick_affordance_2d_valid,
        "success": data.success,
        "total_reward": data.total_reward,
        "num_steps": data.num_steps,
        "h5_path": h5_path,
        "source_object": data.source_object_name,
        "target_object": data.target_object_name,
    }
    if variant_label:
        d["variant_label"] = variant_label
    if env_modifiers:
        d["env_modifiers"] = list(env_modifiers)
    if modifier_params:
        d["modifier_params"] = dict(modifier_params)
    if video_3rd:
        d["video_3rd_view"] = video_3rd
    if video_wrist:
        d["video_wrist"] = video_wrist
    return d


def _write_mp4_cv2(path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    import cv2

    t, h, w = frames_rgb.shape[:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    for i in range(t):
        bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()


def _write_mp4_ffmpeg(path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    t, h, w = frames_rgb.shape[:3]
    if t <= 0:
        raise ValueError("empty frames")
    frames_rgb = np.ascontiguousarray(frames_rgb.astype(np.uint8, copy=False))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "veryfast",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(frames_rgb.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore")
        ret = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if ret != 0:
        raise RuntimeError(stderr.strip() or f"ffmpeg failed with code {ret}")


def _write_episode_videos(
    data: EpisodeData,
    output_dir: Path,
    stem: str,
    fps: int = 10,
) -> Tuple[str, str]:
    """Write third-view and optional wrist MP4. Returns (path_3rd, path_wrist)."""
    p3p = output_dir / f"{stem}_3rd_view.mp4"
    p3 = ""
    pw = ""

    try:
        _write_mp4_ffmpeg(p3p, data.images_0, fps)
        p3 = str(p3p)
        if data.images_3 is not None and data.images_3.shape[0] > 0:
            pwp = output_dir / f"{stem}_wrist.mp4"
            _write_mp4_ffmpeg(pwp, data.images_3, fps)
            pw = str(pwp)
        return p3, pw
    except Exception as e:
        print(f"  WARN: ffmpeg video failed ({e}), try mediapy")

    try:
        import mediapy as media

        media.write_video(str(p3p), data.images_0, fps=fps)
        p3 = str(p3p)
        if data.images_3 is not None and data.images_3.shape[0] > 0:
            pwp = output_dir / f"{stem}_wrist.mp4"
            media.write_video(str(pwp), data.images_3, fps=fps)
            pw = str(pwp)
        return p3, pw
    except ImportError:
        pass
    except Exception as e:
        print(f"  WARN: mediapy video failed ({e}), try cv2")

    try:
        _write_mp4_cv2(p3p, data.images_0, fps)
        p3 = str(p3p)
        if data.images_3 is not None and data.images_3.shape[0] > 0:
            pwp = output_dir / f"{stem}_wrist.mp4"
            _write_mp4_cv2(pwp, data.images_3, fps)
            pw = str(pwp)
        return p3, pw
    except Exception as e:
        print(f"  WARN: cv2 video export failed: {e}")
        return "", ""


def _make_policy(policy_name: str, task_key: str):
    if policy_name == "null":
        return NullPolicy()
    if policy_name == "waypoint":
        return WaypointPutOnPolicy(task_key)
    raise ValueError(f"Unknown policy {policy_name!r}; use 'waypoint' or 'null'")


def _load_modifier_params(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("modifier_params must be a JSON object")
    return data


ParallelWorkerPayload = Tuple[
    str, int, int, str, Optional[int], Optional[List[str]], Optional[Dict[str, Any]]
]
RolloutWorkerPayload = Tuple[
    str,
    int,
    int,
    str,
    int,
    Optional[int],
    Optional[List[str]],
    Optional[Dict[str, Any]],
]


def _physical_gpu(slot_in_batch: int, num_gpus: int, first_gpu: int) -> Optional[int]:
    if num_gpus <= 0:
        return None
    return int(first_gpu + (slot_in_batch % num_gpus))


def _parallel_worker(
    task_key: str,
    episode_id: int,
    max_steps: int,
    policy_name: str,
    physical_gpu_id: Optional[int],
    env_modifiers: Optional[List[str]],
    modifier_params: Optional[Dict[str, Any]],
):
    """One process: one rollout per episode_id (deterministic env → duplicate retries useless)."""
    if physical_gpu_id is not None:
        apply_sim_gpu_env(physical_gpu_id)
    try:
        policy = _make_policy(policy_name, task_key)
        return collect_episode(
            task_key=task_key,
            episode_id=episode_id,
            policy=policy,
            max_steps=max_steps,
            env_modifiers=env_modifiers,
            modifier_params=modifier_params,
        )
    except Exception:
        return None


def _rollout_worker(
    task_key: str,
    episode_id: int,
    max_steps: int,
    policy_name: str,
    max_retries: int,
    physical_gpu_id: Optional[int],
    env_modifiers: Optional[List[str]],
    modifier_params: Optional[Dict[str, Any]],
) -> Optional[EpisodeData]:
    if physical_gpu_id is not None:
        apply_sim_gpu_env(physical_gpu_id)
    data: Optional[EpisodeData] = None
    for _ in range(max(1, max_retries)):
        try:
            policy = _make_policy(policy_name, task_key)
            data = collect_episode(
                task_key=task_key,
                episode_id=episode_id,
                policy=policy,
                max_steps=max_steps,
                env_modifiers=env_modifiers,
                modifier_params=modifier_params,
            )
        except Exception:
            data = None
            continue
        if data is not None and data.success:
            break
    return data


def run_parallel_success_quota(
    task_key: str,
    output_dir: str,
    target_successes: int,
    parallel: int,
    max_steps: int = 120,
    gen_meta_flag: bool = True,
    policy_name: str = "waypoint",
    max_total_episode_ids: int = 8192,
    num_gpus: int = 1,
    first_gpu: int = 0,
    env_modifiers: Optional[List[str]] = None,
    modifier_params: Optional[Dict[str, Any]] = None,
) -> int:
    """Collect successful episodes in parallel batches until `target_successes` is reached."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list = []
    global_idx = 0
    next_base = 0
    parallel = max(1, parallel)
    use_pool = num_gpus > 1
    if parallel > 12 and num_gpus <= 1:
        print(
            "WARN: parallel>12 时 Vulkan 离屏渲染易触发 vk::DeviceLostError；"
            "单 GPU 建议 4~8；多卡请设 --num_gpus。"
        )

    print(
        f"Parallel success quota: task={task_key}  need={target_successes}  "
        f"workers={parallel}  num_gpus={num_gpus}  first_gpu={first_gpu}  "
        f"fresh_proc_per_rollout={use_pool}"
    )

    while global_idx < target_successes and next_base < max_total_episode_ids:
        ids = list(range(next_base, next_base + parallel))
        payloads = []
        for j, eid in enumerate(ids):
            gid = _physical_gpu(j, num_gpus, first_gpu)
            payloads.append(
                (
                    task_key,
                    eid,
                    max_steps,
                    policy_name,
                    gid,
                    env_modifiers,
                    modifier_params,
                )
            )
        t_batch = time.time()
        if use_pool:
            ctx = get_context("spawn")
            with ctx.Pool(processes=parallel, maxtasksperchild=1) as pool:
                batch_results = pool.starmap(_parallel_worker, payloads)
            for data in batch_results:
                if global_idx >= target_successes:
                    break
                if data is None or not getattr(data, "success", False):
                    continue
                h5_path = _write_episode(data, output_dir, global_idx)
                manifest.append(_write_manifest_entry(data, global_idx, h5_path))
                dt = time.time() - t_batch
                print(
                    f"  [{global_idx:06d}] env_ep={data.episode_id:4d}  steps={data.num_steps:3d}  "
                    f"success=True  src={data.source_object_name!r}  batch_t~{dt:.1f}s"
                )
                global_idx += 1
        else:
            with ProcessPoolExecutor(max_workers=parallel) as ex:
                futures = [ex.submit(_parallel_worker, *p) for p in payloads]
                for fut in as_completed(futures):
                    if global_idx >= target_successes:
                        break
                    data = fut.result()
                    if data is None or not getattr(data, "success", False):
                        continue
                    h5_path = _write_episode(data, output_dir, global_idx)
                    manifest.append(_write_manifest_entry(data, global_idx, h5_path))
                    dt = time.time() - t_batch
                    print(
                        f"  [{global_idx:06d}] env_ep={data.episode_id:4d}  steps={data.num_steps:3d}  "
                        f"success=True  src={data.source_object_name!r}  batch_t~{dt:.1f}s"
                    )
                    global_idx += 1
        next_base += len(ids)
        if global_idx >= target_successes:
            break

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}  ({len(manifest)} episodes)")

    if gen_meta_flag and global_idx > 0:
        meta_path = output_dir / "meta.json"
        generate_meta(data_dir=str(output_dir), output_path=str(meta_path))

    if global_idx < target_successes:
        print(
            f"\nWARNING: only {global_idx}/{target_successes} successes "
            f"(exhausted episode_id range or policy never succeeded)."
        )

    print(f"\nDone. Successful episodes saved: {global_idx}")
    return global_idx


def run_multi_task_success_quota(
    tasks: List[str],
    output_dir: str,
    target_successes_per_task: int,
    parallel: int,
    max_steps: int = 120,
    gen_meta_flag: bool = True,
    policy_name: str = "waypoint",
    max_total_episode_ids: int = 8192,
    num_gpus: int = 1,
    first_gpu: int = 0,
    env_modifiers: Optional[List[str]] = None,
    modifier_params: Optional[Dict[str, Any]] = None,
) -> int:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ensure_simpler_env_importable()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list = []
    global_idx = 0
    parallel = max(1, parallel)
    use_pool = num_gpus > 1
    target_successes_per_task = max(1, int(target_successes_per_task))
    if parallel > 12 and num_gpus <= 1:
        print(
            "WARN: parallel>12 时 Vulkan 离屏渲染易触发 vk::DeviceLostError；"
            "单 GPU 建议 4~8；多卡请设 --num_gpus。"
        )

    for task_key in tasks:
        task_successes = 0
        next_base = 0
        print(
            f"\n{'='*60}\n"
            f"Task: {task_key}  success_quota={target_successes_per_task}  "
            f"workers={parallel}  num_gpus={num_gpus}  first_gpu={first_gpu}\n"
            f"{'='*60}"
        )
        while task_successes < target_successes_per_task and next_base < max_total_episode_ids:
            ids = list(range(next_base, next_base + parallel))
            payloads = []
            for j, eid in enumerate(ids):
                gid = _physical_gpu(j, num_gpus, first_gpu)
                payloads.append(
                    (
                        task_key,
                        eid,
                        max_steps,
                        policy_name,
                        gid,
                        env_modifiers,
                        modifier_params,
                    )
                )
            t_batch = time.time()
            if parallel <= 1:
                for eid in ids:
                    if task_successes >= target_successes_per_task:
                        break
                    try:
                        policy = _make_policy(policy_name, task_key)
                        data = collect_episode(
                            task_key=task_key,
                            episode_id=eid,
                            policy=policy,
                            max_steps=max_steps,
                            env_modifiers=env_modifiers,
                            modifier_params=modifier_params,
                        )
                    except Exception as e:
                        print(f"  ERROR task={task_key} env_ep={eid}: {e}")
                        continue
                    if data is None or not getattr(data, "success", False):
                        print(f"  skip task={task_key} env_ep={eid} success=False")
                        continue
                    h5_path = _write_episode(data, output_dir, global_idx)
                    manifest.append(_write_manifest_entry(data, global_idx, h5_path))
                    dt = time.time() - t_batch
                    print(
                        f"  [{global_idx:06d}] task_success={task_successes + 1:03d}/{target_successes_per_task:03d}  "
                        f"env_ep={data.episode_id:4d}  steps={data.num_steps:3d}  "
                        f"src={data.source_object_name!r}  batch_t~{dt:.1f}s"
                    )
                    global_idx += 1
                    task_successes += 1
            elif use_pool:
                ctx = get_context("spawn")
                with ctx.Pool(processes=parallel, maxtasksperchild=1) as pool:
                    batch_results = pool.starmap(_parallel_worker, payloads)
                for data in batch_results:
                    if task_successes >= target_successes_per_task:
                        break
                    if data is None or not getattr(data, "success", False):
                        continue
                    h5_path = _write_episode(data, output_dir, global_idx)
                    manifest.append(_write_manifest_entry(data, global_idx, h5_path))
                    dt = time.time() - t_batch
                    print(
                        f"  [{global_idx:06d}] task_success={task_successes + 1:03d}/{target_successes_per_task:03d}  "
                        f"env_ep={data.episode_id:4d}  steps={data.num_steps:3d}  "
                        f"src={data.source_object_name!r}  batch_t~{dt:.1f}s"
                    )
                    global_idx += 1
                    task_successes += 1
            else:
                with ProcessPoolExecutor(max_workers=parallel) as ex:
                    futures = [ex.submit(_parallel_worker, *p) for p in payloads]
                    for fut in as_completed(futures):
                        if task_successes >= target_successes_per_task:
                            break
                        data = fut.result()
                        if data is None or not getattr(data, "success", False):
                            continue
                        h5_path = _write_episode(data, output_dir, global_idx)
                        manifest.append(_write_manifest_entry(data, global_idx, h5_path))
                        dt = time.time() - t_batch
                        print(
                            f"  [{global_idx:06d}] task_success={task_successes + 1:03d}/{target_successes_per_task:03d}  "
                            f"env_ep={data.episode_id:4d}  steps={data.num_steps:3d}  "
                            f"src={data.source_object_name!r}  batch_t~{dt:.1f}s"
                        )
                        global_idx += 1
                        task_successes += 1
            next_base += len(ids)

        if task_successes < target_successes_per_task:
            print(
                f"WARNING: task {task_key} only reached "
                f"{task_successes}/{target_successes_per_task} successes."
            )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}  ({len(manifest)} episodes)")

    if gen_meta_flag and global_idx > 0:
        meta_path = output_dir / "meta.json"
        generate_meta(data_dir=str(output_dir), output_path=str(meta_path))

    print(f"\nDone. Successful episodes saved: {global_idx}")
    return global_idx


def run(
    tasks: List[str],
    output_dir: str,
    episodes_override: Optional[int] = None,
    max_steps: int = 120,
    gen_meta_flag: bool = True,
    policy_name: str = "waypoint",
    only_save_success: bool = False,
    max_retries: int = 1,
    save_video: bool = False,
    video_fps: int = 10,
    parallel: int = 1,
    num_gpus: int = 1,
    first_gpu: int = 0,
    env_modifiers: Optional[List[str]] = None,
    modifier_params: Optional[Dict[str, Any]] = None,
):
    """Main collection loop."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    global_idx = 0
    parallel = max(1, parallel)
    use_pool = num_gpus > 1 and parallel > 1
    if parallel > 12 and num_gpus <= 1:
        print(
            "WARN: parallel>12 时 Vulkan 离屏渲染易触发 vk::DeviceLostError；"
            "单 GPU 建议 4~8；多卡请设 --num_gpus 与 --parallel。"
        )
    if parallel > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    if parallel <= 1 and num_gpus > 0:
        apply_sim_gpu_env(first_gpu)

    for task_key in tasks:
        cfg = get_task(task_key)
        n_eps = episodes_override if episodes_override is not None else cfg.default_episodes
        print(f"\n{'='*60}")
        print(f"Task: {task_key}  env: {cfg.env_id}  episodes: {n_eps}")
        if parallel > 1:
            print(
                f"parallel workers: {parallel}  num_gpus: {num_gpus}  "
                f"first_gpu: {first_gpu}  fresh_proc_per_rollout: {use_pool}"
            )
        print(f"{'='*60}")

        def _save_one(ep_id: int, data: EpisodeData, dt_s: float) -> None:
            nonlocal global_idx
            if only_save_success and not data.success:
                print(
                    f"  skip ep={ep_id}  success=False after {max(1, max_retries)} attempt(s)"
                )
                return
            h5_path = _write_episode(data, output_dir, global_idx)
            stem = f"episode_{global_idx:06d}"
            v3, vw = ("", "")
            if save_video:
                v3, vw = _write_episode_videos(
                    data, output_dir, stem, fps=video_fps
                )
            entry = _write_manifest_entry(
                data, global_idx, h5_path, video_3rd=v3, video_wrist=vw
            )
            manifest.append(entry)
            extra_v = f"  vid_3rd={stem}_3rd_view.mp4" if v3 else ""
            print(
                f"  [{global_idx:06d}] ep={ep_id:3d}  steps={data.num_steps:3d}  "
                f"success={data.success}  wrist={data.wrist_view_valid}  "
                f'src={data.source_object_name!r} tgt={data.target_object_name!r}  '
                f'instr="{data.instruction[:50]}"  {dt_s:.1f}s{extra_v}'
            )
            global_idx += 1

        if parallel <= 1:
            for ep_id in range(n_eps):
                data: Optional[EpisodeData] = None
                t0 = time.time()
                for attempt in range(max(1, max_retries)):
                    try:
                        policy = _make_policy(policy_name, task_key)
                        data = collect_episode(
                            task_key=task_key,
                            episode_id=ep_id,
                            policy=policy,
                            max_steps=max_steps,
                            env_modifiers=env_modifiers,
                            modifier_params=modifier_params,
                        )
                    except Exception as e:
                        print(f"  ERROR episode {ep_id} attempt {attempt + 1}: {e}")
                        data = None
                        continue
                    if data is not None and data.success:
                        break

                if data is None:
                    continue
                _save_one(ep_id, data, time.time() - t0)
        else:
            for wave_start in range(0, n_eps, parallel):
                batch = list(range(wave_start, min(wave_start + parallel, n_eps)))
                nw = len(batch)
                payloads: List[RolloutWorkerPayload] = []
                for j, eid in enumerate(batch):
                    gid = _physical_gpu(j, num_gpus, first_gpu)
                    payloads.append(
                        (
                            task_key,
                            eid,
                            max_steps,
                            policy_name,
                            max_retries,
                            gid,
                            env_modifiers,
                            modifier_params,
                        )
                    )
                t_batch = time.time()
                if use_pool:
                    ctx = get_context("spawn")
                    with ctx.Pool(processes=nw, maxtasksperchild=1) as pool:
                        results = pool.starmap(_rollout_worker, payloads)
                else:
                    with ProcessPoolExecutor(max_workers=nw) as ex:
                        futs = [ex.submit(_rollout_worker, *p) for p in payloads]
                        results = [fu.result() for fu in futs]
                dt_each = (time.time() - t_batch) / max(1, nw)
                for ep_id, data in zip(batch, results):
                    if data is None:
                        continue
                    _save_one(ep_id, data, dt_each)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}  ({len(manifest)} episodes)")

    if gen_meta_flag and global_idx > 0:
        meta_path = output_dir / "meta.json"
        generate_meta(
            data_dir=str(output_dir),
            output_path=str(meta_path),
        )

    print(f"\nDone. Total episodes: {global_idx}")
    return global_idx


def main():
    parser = argparse.ArgumentParser(
        description="Collect WidowX sim episodes → Bridge HDF5"
    )
    parser.add_argument(
        "--task", type=str, nargs="*", default=None,
        help="One or more task_key values"
    )
    parser.add_argument(
        "--group", type=str, nargs="*", default=None,
        help="Task group(s): base, multi_object, layout_distractor"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for HDF5 + meta"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override default episode count per task"
    )
    parser.add_argument(
        "--max_steps", type=int, default=120,
        help="Max steps per episode (default: 120)"
    )
    parser.add_argument(
        "--no_meta", action="store_true",
        help="Skip meta JSON generation"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available tasks and exit"
    )
    parser.add_argument(
        "--policy", type=str, default="waypoint",
        choices=("waypoint", "null"),
        help="Rollout policy: waypoint scripted FSM or null (hold pose)"
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="Only write HDF5 when rollout succeeded (default: save all rollouts)",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="For each saved episode, write episode_XXXXXX_3rd_view.mp4 and _wrist.mp4",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=10,
        help="FPS for --save_video (default: 10)",
    )
    parser.add_argument(
        "--max_retries", type=int, default=1,
        help="Rollout attempts per episode_id before giving up (default: 1)"
    )
    parser.add_argument(
        "--success_quota",
        type=int,
        default=None,
        metavar="N",
        help="Collect N successful episodes then stop (requires exactly one --task); uses parallel workers",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Worker processes for parallel rollouts. "
        "Default: 1 (sequential) for normal runs; if omitted with --success_quota, "
        "uses min(64, CPU count). Example: --parallel $(nproc)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        metavar="N",
        help="参与轮询的 GPU 个数（物理卡从 --first_gpu 起连续编号）。"
        "设为 0 表示不绑定 CUDA/EGL（沿用环境变量）。"
        "大于 1 时每轮 rollout 用独立子进程并轮换 GPU（maxtasksperchild=1）。",
    )
    parser.add_argument(
        "--first_gpu",
        type=int,
        default=0,
        metavar="N",
        help="起始 GPU 编号（默认 0）。与 --num_gpus 组合使用，例如 8 卡：--num_gpus 8 --first_gpu 0",
    )
    parser.add_argument(
        "--env_modifier",
        action="append",
        default=None,
        help="运行时环境 modifier，可重复传入，例如 --env_modifier camera --env_modifier table_color",
    )
    parser.add_argument(
        "--modifier_params_json",
        type=str,
        default=None,
        help="modifier 参数 JSON 字符串或 JSON 文件路径",
    )
    args = parser.parse_args()

    ncpu = max(1, min(64, os.cpu_count() or 8))
    if args.parallel is not None:
        parallel_workers = max(1, args.parallel)
    elif args.success_quota is not None:
        parallel_workers = ncpu
    else:
        parallel_workers = 1

    if args.list:
        for t in ALL_TASKS:
            print(f"  {t.task_key:50s}  group={t.task_group:20s}  eps={t.default_episodes:3d}  env={t.env_id}")
        sys.exit(0)

    tasks = []
    if args.task:
        tasks.extend(args.task)
    if args.group:
        for g in args.group:
            tasks.extend(t.task_key for t in get_group(g))

    if not tasks:
        parser.error("Specify --task or --group (or --list to see available tasks)")

    env_modifiers = args.env_modifier or None
    modifier_params = _load_modifier_params(args.modifier_params_json)

    if args.success_quota is not None:
        if len(tasks) != 1:
            parser.error("--success_quota requires exactly one task in --task (no --group)")
        rq = max(1, args.success_quota)
        run_parallel_success_quota(
            task_key=tasks[0],
            output_dir=args.output_dir,
            target_successes=rq,
            parallel=parallel_workers,
            max_steps=args.max_steps,
            gen_meta_flag=not args.no_meta,
            policy_name=args.policy,
            num_gpus=max(0, args.num_gpus),
            first_gpu=args.first_gpu,
            env_modifiers=env_modifiers,
            modifier_params=modifier_params,
        )
    else:
        run(
            tasks=tasks,
            output_dir=args.output_dir,
            episodes_override=args.episodes,
            max_steps=args.max_steps,
            gen_meta_flag=not args.no_meta,
            policy_name=args.policy,
            only_save_success=args.success_only,
            max_retries=args.max_retries,
            save_video=args.save_video,
            video_fps=args.video_fps,
            parallel=parallel_workers,
            num_gpus=max(0, args.num_gpus),
            first_gpu=args.first_gpu,
            env_modifiers=env_modifiers,
            modifier_params=modifier_params,
        )


if __name__ == "__main__":
    main()
