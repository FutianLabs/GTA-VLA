# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Tuple, Iterable, Sequence, Any
import numpy as np
import h5py
import robosuite.utils.transform_utils as T

from ..utils import euler_to_rotate6d, quat_to_rotate6d
from .base import BaseHDF5Handler

EPS = 1e-6





class LiberoAbsActionProcessor:
    """Helpers to convert between 6D rotation (Zhou et al.) and axis-angle."""

    def Rotate6D_to_AxisAngle(self, r6d: np.ndarray) -> np.ndarray:
        """Convert 6D rotation representation to axis-angle.

        Args:
            r6d: array with shape (N, 6) or (6,)
        Returns:
            array with shape (N, 3) or (3,)
        """
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]

        # b1
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)

        # b2
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)

        # b3
        b3 = np.cross(b1, b2, axis=-1)

        R = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)

        axis_angle_list = []
        for i in range(R.shape[0]):
            quat = T.mat2quat(R[i])
            axis_angle = T.quat2axisangle(quat)
            axis_angle_list.append(axis_angle)

        axis_angle_array = np.stack(axis_angle_list, axis=0)
        return axis_angle_array[0] if single else axis_angle_array

    def Mat_to_Rotate6D(self, R: np.ndarray) -> np.ndarray:
        if R.ndim == 2:
            return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)
        elif R.ndim == 3:
            return np.concatenate([R[:, :3, 0], R[:, :3, 1]], axis=-1)
        else:
            raise ValueError("Rotation matrix must be (...,3,3)")

    def AxisAngle_to_Rotate6D(self, aa: np.ndarray) -> np.ndarray:
        # TODO support ndim==2
        if aa.ndim == 1:
            return self.Mat_to_Rotate6D(T.quat2mat(T.axisangle2quat(aa)))
        elif aa.ndim == 2:
            output = [self.Mat_to_Rotate6D(T.quat2mat(T.axisangle2quat(aa[i]))) for i in range(aa.shape[0])]
            return np.stack(output, axis=0)
        else:
            raise ValueError("Unsupported axis-angle shape")

    def action_6d_to_axisangle(self, action: np.ndarray) -> np.ndarray:
        """Convert action [..., 3(pos)+6(rot6d)+1(grip)] -> [..., 3(pos)+3(aa)+1(grip)]"""
        if action.ndim == 1:
            final_ori = self.Rotate6D_to_AxisAngle(action[3:9])
            return np.concatenate([action[0:3], final_ori, action[-1:]])
        elif action.ndim == 2:
            final_ori = self.Rotate6D_to_AxisAngle(action[:, 3:9])
            return np.concatenate([action[:, 0:3], final_ori, action[:, -1:]], axis=-1)
        else:
            raise ValueError("Unsupported action shape")


# ------------------------------- Calvin --------------------------------------
class CalvinHandler(BaseHDF5Handler):
    """Calvin (sim): proprio [T,7] -> xyz(3)+euler_xyz(3)+grip(1). Right is zeros."""
    dataset_name = "Calvin"

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        proprio = f["proprio"][()]  # [T,7]
        return proprio, None

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), proprio[:, -1:] < 0.],
            axis=-1,
        )  # [T,10]
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 20))


# --------------------------------- RT1 ---------------------------------------
class RT1Handler(BaseHDF5Handler):
    """RT1 (sim-like packaging): eef_quat_orientation [T,7], gripper [T,1]."""
    dataset_name = "RT1"

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        eefq = f["eef_quat_orientation"][()]  # [T,7] pos3 + quat4
        grip = f["gripper"][()]               # [T,1] or [T]
        return (eefq, grip), None

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 3.0, 10.0
        eefq, grip = proprio
        if grip.ndim == 1:
            grip = grip[:, None]
        left = np.concatenate([eefq[:, :3], quat_to_rotate6d(eefq[:, 3:]), grip], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 6))


# ------------------------------ Fractal (RT-1) --------------------------------
class FractalHandler(BaseHDF5Handler):
    """
    Fractal / RT-1. HDF5:
      /proprio [T, 8] -> position(3) + quaternion(4) + gripper_closed(1)
      /action  [T, 10] -> world_vector(3) + rotation_delta(3) + gripper(1)
                          + base_disp(2) + base_rot(1)
    Output left/right: [T,10] = xyz(3)+rot6d(6)+grip(1). Single arm → right zeros.
    ~3 Hz control frequency.
    """
    dataset_name = "Fractal"

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 3.0, 10.0
        # proprio: [T, 8] = pos(3) + quat(4) + gripper_closed(1)
        left = np.concatenate(
            [proprio[:, :3], quat_to_rotate6d(proprio[:, 3:7]), proprio[:, 7:8]],
            axis=-1,
        )  # [T, 10]
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 6))


# ------------------------------- Bridge --------------------------------------
class BridgeHandler(BaseHDF5Handler):
    """
    Bridge (sim). HDF5:
      /proprio [T, >=6] -> xyz(3) + euler_xyz(3) + ...
      /action  [T, ...] -> last channel is gripper (1=open), we convert to (1=closed)
    Output left/right: [T,10] = xyz(3)+rot6d(6)+grip(1). Single arm → right zeros.
    """
    dataset_name = "Bridge"
    wrist_key = "observation/image_3"

    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys: Sequence[str] = self.meta["observation_key"]
        images = []
        for k in keys:
            if k != self.wrist_key:
                images.append(f[k][()])
            else:
                if k in f and f.attrs.get('wrist_view_valid', True):
                    images.append(f[k][()])
                # else: skip — base handler pads with zeros and sets mask=False
        return images

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 5.0, 5.0

        action[:, -1] = np.clip(action[:, -1], 0, 1)
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), 1 - action[:, -1:]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ------------------------------- LIBERO --------------------------------------
class LiberoHandler(BaseHDF5Handler):
    """
    LIBERO (sim). HDF5:
      /abs_action_6d [T,10] = xyz(3)+rot6d(6)+grip_raw(1). Single arm.
    Also drops first frame for images (matches original pipeline behavior).
    """
    dataset_name = "libero"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.libero_abs_action_processor = LiberoAbsActionProcessor()
    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys = self.meta["observation_key"]
        images = [f[k] for k in keys]
        # idx = keys.index("observation/agentview_rgb")
        # images[idx] = np.flip(images[idx][:], axis=2)
        # images[idx] = np.flip(images[idx], axis=1)
        # Drop the first frame (image desync quirk in original data)
        return [img[1:] for img in images]

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        proprio = None                  # [T, >=6]
        action  = f["abs_action_6d"][()]    
        return proprio, action

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        if action.shape[1] == 7: # the rot is saved in axisangle format
            xyz = action[:, :3]
            rot = self.libero_abs_action_processor.AxisAngle_to_Rotate6D(action[:, 3:6])
            left = np.concatenate([xyz, rot, (action[:, 6:] > 0.0)], axis=-1)
        else:
            left = np.concatenate([action[:, :9], (action[:, 9:] > 0.0)], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ------------------------------ VLABench -------------------------------------
class VLABenchHandler(BaseHDF5Handler):
    """ 
    VLABench (sim). HDF5:
      /proprio [T, >=7] -> xyz(3) + euler_xyz(3) + grip(1).
    Single arm → right zeros.
    """
    dataset_name = "VLABench"

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        proprio = f["proprio"][()]
        return proprio, None

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), proprio[:, -1:]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 15))


# ------------------------------ RobotWin2 ------------------------------------
class RobotWin2Handler(BaseHDF5Handler):
    """
    robotwin2_abs_ee / robotwin2_clean (sim). HDF5:
      /endpose/left_endpose   [T,7]  xyz(3)+quat(4)
      /endpose/right_endpose  [T,7]
      /endpose/left_gripper   [T]    1=open  -> convert to 1=closed
      /endpose/right_gripper  [T]
    Output both arms. freq≈30Hz, qdur=1s.
    """
    dataset_name = "robotwin2-*"

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        l = f["endpose/left_endpose"][()]                      # [T,7]
        r = f["endpose/right_endpose"][()]                     # [T,7]
        lg = f["endpose/left_gripper"][()][:, None]            # [T,1] 1=open
        rg = f["endpose/right_gripper"][()][:, None]           # [T,1] 1=open
        return l, r, lg, rg

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        l, r, lg, rg = proprio
        lg = 1 - lg
        rg = 1 - rg
        left  = np.concatenate([l[:, :3], quat_to_rotate6d(l[:, 3:]), lg], axis=-1)
        right = np.concatenate([r[:, :3], quat_to_rotate6d(r[:, 3:]), rg], axis=-1)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ---------------------------- Robocasa-Human ---------------------------------
class RobocasaHumanHandler(BaseHDF5Handler):
    """
    robocasa-human (teleop in sim). HDF5:
      /action_dict/abs_pos     [T,3]
      /action_dict/abs_rot_6d  [T,6]
      /action_dict/gripper     [T,1]  ( >0 => closed )
    Single arm → right zeros.
    """
    dataset_name = "robocasa-human"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        left = np.concatenate(
            [
                f["action_dict/abs_pos"][()],
                f["action_dict/abs_rot_6d"][()],
                (f["action_dict/gripper"][()] > 0.0).astype(np.float32),
            ],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 30))



