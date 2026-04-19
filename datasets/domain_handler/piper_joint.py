from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Iterable

import numpy as np
import h5py

from .base import BaseHDF5Handler

DEG2RAD = np.pi / 180.0


def ensure_norm_stats(meta_path: str) -> Optional[dict]:
    """
    Auto-compute and cache per-dim q01/q99 (6 joints + 1 gripper) from
    an HDF5 datalist.  Pi0.5-style: gripper is a continuous value, not
    binary.

    Returns None if the meta is not for piper_joint.
    Stats are cached to ``<meta_dir>/<dataset_name>_norm_stats.json``.
    """
    with open(meta_path) as f:
        meta = json.load(f)

    ds_name = meta.get("dataset_name", "")
    if "joint" not in ds_name.lower():
        return None

    meta_stem = os.path.splitext(os.path.basename(meta_path))[0]
    cache_path = os.path.join(
        os.path.dirname(os.path.abspath(meta_path)),
        f"{meta_stem}_norm_stats.json",
    )

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            stats = json.load(f)
        if len(stats.get("q01", [])) == 7:
            print(f"[norm] Loaded cached 7-dim stats from {cache_path}")
            return stats
        print("[norm] Old 6-dim stats found, regenerating with gripper …")

    print(f"[norm] Computing 7-dim q01/q99 from {len(meta['datalist'])} episodes …")
    all_data = []
    for p in meta["datalist"]:
        with h5py.File(p, "r") as fh:
            js = fh["joint_states"][()]          # [T, 7] joints_deg + gripper_m
        converted = np.empty_like(js)
        converted[:, :6] = js[:, :6] * DEG2RAD   # degrees → radians
        converted[:, 6] = js[:, 6]               # gripper already in meters
        all_data.append(converted)

    all_data = np.concatenate(all_data, axis=0)
    q01 = np.percentile(all_data, 1, axis=0).tolist()
    q99 = np.percentile(all_data, 99, axis=0).tolist()

    stats = {
        "q01": q01,
        "q99": q99,
        "joint_indices": [0, 1, 2, 3, 4, 5, 6],
        "num_frames": int(len(all_data)),
        "num_episodes": len(meta["datalist"]),
    }

    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(
        f"[norm] Saved 7-dim stats to {cache_path}\n"
        f"       q01={[f'{v:.4f}' for v in q01]}\n"
        f"       q99={[f'{v:.4f}' for v in q99]}"
    )
    return stats


class PiperJointHandler(BaseHDF5Handler):
    """
    Piper joint-space handler — Pi0.5-style continuous gripper.

    HDF5 layout:
      /joint_states               [T, 7]  joints_deg(6) + gripper_m(1)
      /observation/agentview_rgb  [T, 480, 640, 3]  uint8
      /observation/eye_in_hand_rgb[T, 480, 640, 3]  uint8
      attrs["instruction"]        str

    All 7 dims (6 joints in radians + 1 gripper in meters) are
    normalized to [-1, 1] via per-dim q01/q99 quantile statistics.
    Output left/right: [T, 7] = normalized(joints + gripper).
    Right arm is zeros.  Matches JointActionSpace (dim_action=14).
    """

    dataset_name = "piper_joint"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stats = self.meta.get("joint_norm_stats")
        if stats is None:
            stats = self._load_cached_stats()
        self._q01 = np.array(stats["q01"], dtype=np.float64)
        self._q99 = np.array(stats["q99"], dtype=np.float64)
        self._range = self._q99 - self._q01
        self._range[self._range < 1e-6] = 1.0

    def _load_cached_stats(self) -> dict:
        """Try loading from cache files next to the data directory."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith("_norm_stats.json"):
                path = os.path.join(data_dir, fname)
                with open(path) as f:
                    stats = json.load(f)
                if len(stats.get("q01", [])) == 7:
                    print(f"[norm] Loaded fallback 7-dim stats from {path}")
                    return stats
        raise FileNotFoundError(
            "No 7-dim joint_norm_stats in meta and no 7-dim *_norm_stats.json "
            "found in data/. Delete old cache and re-run ensure_norm_stats()."
        )

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        action = f["joint_states"][()]  # [T, 7]
        return None, action

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 3.33

        phys = np.empty_like(action, dtype=np.float64)
        phys[:, :6] = action[:, :6] * DEG2RAD   # degrees → radians
        phys[:, 6] = action[:, 6]                # gripper in meters

        left = (phys - self._q01) / self._range * 2.0 - 1.0   # [T, 7]
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 100))
