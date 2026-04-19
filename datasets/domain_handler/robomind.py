from __future__ import annotations

from typing import Optional, Tuple, Iterable

import numpy as np
import h5py

from ..utils import euler_to_rotate6d
from .base import BaseHDF5Handler


class RobomindHandler(BaseHDF5Handler):

    dataset_name = "robomind-*"

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        ds_name: str = self.meta["dataset_name"]

        if ds_name in ("robomind-franka", "robomind-ur", "robomind-franka-1rgb", "robomind-franka-3rgb",
                        "robomind-franka-cot", "robomind-ur-cot"):
            ee = f["puppet"]["end_effector"][()]
            jp = f["puppet"]["joint_position"][()]
            return ee, jp

        if ds_name in ("robomind-agilex", "robomind-agilex-cot"):
            le = f["puppet"]["end_effector_left"][()]
            re = f["puppet"]["end_effector_right"][()]
            return le, re

        if ds_name == "robomind-franka-dual":
            ee = f["puppet"]["end_effector"][()]
            jp = f["puppet"]["joint_position"][()]
            return ee, jp

        raise NotImplementedError(f"RobomindHandler: unsupported dataset '{ds_name}'")

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        ds_name: str = self.meta["dataset_name"]
        freq, qdur = 30.0, 4.0

        if ds_name in ("robomind-franka", "robomind-ur", "robomind-franka-1rgb", "robomind-franka-3rgb",
                        "robomind-franka-cot", "robomind-ur-cot"):
            ee = proprio
            grip = action[:, -1:]
            left = np.concatenate([ee[:, :3], euler_to_rotate6d(ee[:, 3:6], "xyz"), grip], axis=-1)
            right = np.zeros_like(left)
            return left, right, None, None, freq, qdur

        if ds_name in ("robomind-agilex", "robomind-agilex-cot"):
            le, re = proprio, action
            l = np.concatenate([le[:, :3], euler_to_rotate6d(le[:, 3:6], "xyz"), (le[:, -1:] > 2.5).astype(np.float64)], axis=-1)
            r = np.concatenate([re[:, :3], euler_to_rotate6d(re[:, 3:6], "xyz"), (re[:, -1:] > 2.5).astype(np.float64)], axis=-1)
            return l, r, None, None, freq, qdur

        if ds_name == "robomind-franka-dual":
            ee, jp = proprio, action
            l = np.concatenate([ee[:, 0:3], euler_to_rotate6d(ee[:, 3:6], "xyz"), jp[:, 7:8]], axis=-1)
            r = np.concatenate([ee[:, 6:9], euler_to_rotate6d(ee[:, 9:12], "xyz"), jp[:, -1:]], axis=-1)
            return l, r, None, None, freq, qdur

        raise NotImplementedError(f"RobomindHandler: unsupported dataset '{ds_name}'")

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 30))
