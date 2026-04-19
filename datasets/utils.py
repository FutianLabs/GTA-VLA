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
import io, numpy as np, pyarrow.parquet as pq, av, cv2
from mmengine import fileio
from PIL import Image
import h5py
from typing import Sequence, Dict
import torch
from scipy.spatial.transform import Rotation as R

from models.action_hub import convert_rotate6d_to_euler

def read_bytes(path: str) -> bytes:
    return fileio.get(path)

def open_h5(path: str) -> h5py.File:
    try: return h5py.File(path, "r")
    except OSError: return h5py.File(io.BytesIO(read_bytes(path)), "r")

def read_video_to_frames(path: str) -> np.ndarray:
    buf = io.BytesIO(read_bytes(path)); container = av.open(buf, options={'threads': '2'})
    frames = []
    for packet in container.demux(video=0):
        for f in packet.decode(): frames.append(f.to_ndarray(format="rgb24"))
    return np.stack(frames, axis=0)

def read_parquet(path: str) -> dict:
    buf = io.BytesIO(read_bytes(path))
    return pq.read_table(buf).to_pydict()

_COMMON_RAW_SHAPES = [
    (720, 1280, 3), (480, 640, 3), (256, 256, 3), (224, 224, 3),
    (1080, 1920, 3), (480, 848, 3), (360, 640, 3), (240, 320, 3),
]

def decode_image_from_bytes(x, raw_image_shape=None) -> Image.Image:
    """Decode image from bytes, ndarray, or raw flat pixels.

    Args:
        x: JPEG/PNG bytes, shaped (H,W,3) ndarray, or flat uint8 ndarray.
        raw_image_shape: (H, W, 3) for flat raw pixel data. When the given
            shape doesn't match the data size, common resolutions are tried
            automatically.
    """
    if isinstance(x, (bytes, bytearray)): x = np.frombuffer(x, dtype=np.uint8)
    bgr = cv2.imdecode(x, cv2.IMREAD_COLOR)
    if bgr is not None:
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    raw = x if isinstance(x, np.ndarray) else np.frombuffer(x, dtype=np.uint8)
    sz = raw.size
    if raw_image_shape is not None and np.prod(raw_image_shape) == sz:
        return Image.fromarray(raw.reshape(raw_image_shape))
    for shape in _COMMON_RAW_SHAPES:
        if np.prod(shape) == sz:
            return Image.fromarray(raw.reshape(shape))
    if sz % 3 == 0:
        pixels = sz // 3
        for h in range(16, 2048):
            if pixels % h == 0:
                w = pixels // h
                if 0.2 < h / w < 5.0:
                    return Image.fromarray(raw.reshape(h, w, 3))
    raise ValueError(
        f"cv2.imdecode failed and cannot infer raw shape (data size={sz}). "
        f"Add \"raw_image_shape\": [H, W, 3] to the dataset meta JSON."
    )


def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    return R.from_quat(q, scalar_first = scalar_first).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz')

def rotate6d_to_quat(v6: np.ndarray, scalar_first = False) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_quat(scalar_first = scalar_first)

def action_slice(abs_traj: torch.Tensor, action_mode="ee6d") -> Dict[str, torch.Tensor]:
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    if action_mode == "ee3d":
        abs_traj = convert_rotate6d_to_euler(abs_traj)
    proprio = abs_traj[0]         # [D]
    action = abs_traj[1:].clone() # [H, D]
    return {"proprio": proprio, "action": action}