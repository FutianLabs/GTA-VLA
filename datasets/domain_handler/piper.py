from __future__ import annotations

from typing import Optional, Tuple, Iterable, Sequence, Any
import numpy as np
import h5py

from ..utils import euler_to_rotate6d
from .base import BaseHDF5Handler

# Piper gripper opening (meters): 0 = fully closed, ~0.084 = fully open.
# Treat opening < 10mm as "closed" (binary grip = 1).
GRIPPER_CLOSE_THRESHOLD_M = 0.01


class PiperHandler(BaseHDF5Handler):
    """
    Piper (real robot, single arm).

    HDF5 layout (from convert_lerobot_to_hdf5.py):
      /abs_action_6d              [T, 7]  xyz_m(3) + euler_xyz_rad(3) + gripper_m(1)
      /observation/agentview_rgb  [T, 480, 640, 3]  uint8
      /observation/eye_in_hand_rgb[T, 480, 640, 3]  uint8
      attrs["instruction"]        str

    Output left/right: [T, 10] = xyz(3) + rot6d(6) + grip(1).  Right is zeros.
    """

    dataset_name = "piper"

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        action = f["abs_action_6d"][()]  # [T, 7]
        return None, action

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0

        xyz = action[:, :3]
        rot6d = euler_to_rotate6d(action[:, 3:6], "xyz")
        grip_closed = (action[:, 6:] < GRIPPER_CLOSE_THRESHOLD_M).astype(np.float64)

        left = np.concatenate([xyz, rot6d, grip_closed], axis=-1)  # [T, 10]
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 30))
