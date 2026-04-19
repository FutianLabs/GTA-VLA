"""
Bridge-compatible HDF5 writer for WidowX simulation episodes.

Writes episodes that can be directly read by BridgeHandler in
datasets/domain_handler/simulations.py with zero training-side changes.

Schema (per file):
    /observation/image_0  [T, H, W, 3] uint8   – 3rd-view camera
    /observation/image_3  [T, H, W, 3] uint8   – wrist camera (optional)
    /proprio              [T, D]       float32 – tcp_xyz(3) + euler_xyz(3) [+ extras]
    /action               [T, D]       float32 – delta_xyz(3) + delta_euler(3) + gripper(1)
    /gripper_position     [T, 2]       float32 – projected gripper/TCP center in image_0
    /pick_affordance_position [T, 2]   float32 – projected pick affordance point in image_0
    attrs:
        instruction          str
        instruction_source   str   ("sim")
        wrist_view_valid     bool
        gripper_2d_valid     bool
        pick_affordance_2d_valid  bool
        task_key             str
        task_group           str
        env_id               str
"""

from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np


def save_episode_to_h5(
    output_path: str,
    images_0: np.ndarray,
    proprio: np.ndarray,
    action: np.ndarray,
    instruction: str,
    images_3: Optional[np.ndarray] = None,
    gripper_position: Optional[np.ndarray] = None,
    pick_affordance_position: Optional[np.ndarray] = None,
    wrist_view_valid: bool = False,
    gripper_2d_valid: bool = False,
    pick_affordance_2d_valid: bool = False,
    instruction_source: str = "sim",
    task_key: str = "",
    task_group: str = "",
    env_id: str = "",
) -> None:
    """
    Save one episode as a Bridge-compatible HDF5 file.

    Args:
        output_path: Destination .hdf5 path.
        images_0: [T, H, W, 3] uint8 – primary (3rd-view) camera.
        proprio: [T, D] float32 – at least 6 dims: tcp_xyz + euler_xyz.
        action: [T, D] float32 – last dim must be gripper channel
                 (1 = open in Bridge convention).
        instruction: Language instruction string.
        images_3: Optional [T, H, W, 3] uint8 – wrist camera.
        gripper_position: Optional [T, 2] float32 – projected gripper center.
        pick_affordance_position: Optional [T, 2] float32 – projected pick affordance.
        wrist_view_valid: Whether image_3 is usable.
        gripper_2d_valid: Whether the whole projected gripper trajectory is valid.
        pick_affordance_2d_valid: Whether the whole projected pick affordance is valid.
        instruction_source: "sim" for synthetic data.
        task_key / task_group / env_id: Debug metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as f:
        obs_grp = f.create_group("observation")
        obs_grp.create_dataset(
            "image_0",
            data=images_0,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )

        if images_3 is not None:
            obs_grp.create_dataset(
                "image_3",
                data=images_3,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            )

        f.create_dataset("proprio", data=proprio, dtype=np.float32)
        f.create_dataset("action", data=action, dtype=np.float32)
        if gripper_position is not None:
            f.create_dataset("gripper_position", data=gripper_position, dtype=np.float32)
        if pick_affordance_position is not None:
            f.create_dataset(
                "pick_affordance_position",
                data=pick_affordance_position,
                dtype=np.float32,
            )

        # Required attrs for BridgeHandler
        f.attrs["instruction"] = instruction if instruction else ""
        f.attrs["instruction_source"] = instruction_source
        f.attrs["wrist_view_valid"] = wrist_view_valid
        f.attrs["gripper_2d_valid"] = gripper_2d_valid
        f.attrs["pick_affordance_2d_valid"] = pick_affordance_2d_valid

        # Debug attrs
        f.attrs["task_key"] = task_key
        f.attrs["task_group"] = task_group
        f.attrs["env_id"] = env_id
