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
from re import U
import os
import numpy as np
import torch
import random
import cv2
from mmengine import fileio
from scipy.interpolate import interp1d
from ..utils import open_h5, quat_to_rotate6d
from PIL import Image
from .base import DomainHandler
import json

SPLITFILE = None
USE_WRIST_VIEW = False

# Task instruction per domain
DOMAIN2INS = {
    "agiworld-on-site-pack": "Pick up the object and place it in the bag.",
    "agiworld-on-site-pack-extra": "Pick up the object and place it in the bag.",
    "agiworld-on-site-conveyor": "Pick objects from the conveyor belt and place them in the box.",
    "agiworld-on-site-conveyor-extra": "Pick objects from the conveyor belt and place them in the box.",
    "agiworld-on-site-restock": "Hang the snacks on the shelf.",
    "agiworld-on-site-pour": "pour the water into the cup.", # "stop and place the cup on the table."
    "agiworld-on-site-microwave": "Open the microwave, put the food in", # close and start it.
    "agiworld-on-site-cloth": "fold the clothes.",
}


# Max chunk length per domain

# shorter chunk version
DOMAIN2CHUNKSIZE = {
    "agiworld-on-site-pack": 61,
    "agiworld-on-site-pack-extra": 61,
    "agiworld-on-site-conveyor": 61,
    "agiworld-on-site-conveyor-extra": 61,
    "agiworld-on-site-restock": 61,
    "agiworld-on-site-pour": 61,
    "agiworld-on-site-microwave": 121,
    "agiworld-on-site-cloth": 121
}


import torch


class AGIWolrdHandler(DomainHandler):
    def _get_action_path(self, item: str) -> str:
        data_root = self.meta.get("data_root")
        if data_root:
            return os.path.join(data_root, "proprio_stats", item, "proprio_stats.h5")
        if 'extra' not in self.meta['dataset_name']:
            return fileio.join_path(item, 'aligned_joints.h5')
        return fileio.join_path(item, 'proprio_stats.h5')

    def _get_image_path(self, item: str, idx: int, name: str) -> str:
        data_root = self.meta.get("data_root")
        if data_root:
            return os.path.join(data_root, "observations", item, "videos", f"{name}.mp4")
        if 'extra' not in self.meta['dataset_name']:
            return fileio.join_path(item, f'camera/{idx}/{name}.jpg')
        return fileio.join_path(item, f'videos/{name}/frame_{idx}.jpg')

    def _preload_mp4_frames(self, mp4_path: str, needed_indices: set = None) -> dict:
        import av
        container = av.open(mp4_path)
        frames = {}
        for i, frame in enumerate(container.decode(video=0)):
            if needed_indices is None or i in needed_indices:
                frames[i] = frame.to_ndarray(format='rgb24')
            if needed_indices and i > max(needed_indices):
                break
        container.close()
        return frames

    def read_action(self, item: str):
        action_path = self._get_action_path(item)
        
        with open_h5(str(action_path)) as f:
            try:
                # Some versions: grippers under action/effector/position (two columns: L, R)
                gripper_left = f['action']['effector']['position'][:, 0]   # [T]
                gripper_right = f['action']['effector']['position'][:, 1]  # [T]
            except Exception:
                # Fallback: split under state/left_effector and state/right_effector
                gripper_left = f['action']['left_effector']['position'][:, 0]
                gripper_right = f['action']['right_effector']['position'][:, 0]

            joints = f['state']['joint']['position'][:]                  # [T, 14]
            assert len(gripper_left) == joints.shape[0], "gripper/joint length mismatch"

            xyz_position_left = f['state']['end']['position'][:, 0]      # [T, 3]
            xyz_position_right = f['state']['end']['position'][:, 1]     # [T, 3]
            
            orientation_left = f['state']['end']['orientation'][:, 0]    # [T, 4]
            orientation_right = f['state']['end']['orientation'][:, 1]   # [T, 4]
            
            # Concatenate joints and grippers -> [T, 16]
            abs_joint = np.concatenate([joints,
                                        gripper_left[:, None],
                                        gripper_right[:, None]], axis=-1)

            # Build 6D rotations + XYZ + gripper for both arms
            abs_ee6d = np.concatenate([
                xyz_position_left, quat_to_rotate6d(orientation_left), gripper_left[:, None],
                xyz_position_right, quat_to_rotate6d(orientation_right), gripper_right[:, None]
            ], axis=-1)

        return abs_joint, abs_ee6d

    def _resolve_dataset_name(self):
        return self.meta.get('sub_dataset_name', self.meta['dataset_name'])

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool, action_mode,
                     image_aug, lang_aug_map: dict | None):
        item = self.meta["datalist"][traj_idx]
        ds_name = self._resolve_dataset_name()

        abs_joint, abs_ee6d = self.read_action(item)

        grippers = abs_joint[:, -2:]
        chg = np.any(grippers[1:] != grippers[:-1], axis=1)
        gripper_change_idx = np.flatnonzero(chg)
        
        ins = DOMAIN2INS[ds_name]

        current_ep_idx = item.split('/')[-1]
        try:
            with open(SPLITFILE, "r") as f: 
                split_data = json.load(f) 
        except: split_data = {}
        split_list = [0]
        if current_ep_idx in split_data.keys():
            split_list.extend(split_data[current_ep_idx])
        split_list.append(len(abs_joint))
        split_list = [(a, b) for a, b in zip(split_list[:-1], split_list[1:])]
        
        random.shuffle(split_list)

        all_index_list = []
        segment_map = {}
        for traj_start_idx, traj_end_idx in split_list:
            index_list = list(range(traj_start_idx, 
                                        max(traj_start_idx + 1, 
                                        traj_end_idx - DOMAIN2CHUNKSIZE[ds_name])))
            if ds_name == 'agiworld-on-site-pour':
                for gi in gripper_change_idx: 
                    for i in range(gi-DOMAIN2CHUNKSIZE[ds_name], gi+DOMAIN2CHUNKSIZE[ds_name]):
                        if i in index_list: index_list.append(i)
            valid = [i for i in index_list if np.abs(abs_ee6d[i + 1] - abs_ee6d[i]).max() >= 5e-4]
            random.shuffle(valid)
            for i in valid:
                segment_map[i] = (traj_start_idx, traj_end_idx)
            all_index_list.extend(valid)

        use_mp4 = self.meta.get("data_root") is not None
        image_names = ['head_color', 'hand_left_color', 'hand_right_color']
        mp4_frames_cache = {}

        if use_mp4 and all_index_list:
            needed = set(all_index_list)
            for name in image_names:
                mp4_path = self._get_image_path(item, 0, name)
                mp4_frames_cache[name] = self._preload_mp4_frames(mp4_path, needed)

        for idx in all_index_list:
            traj_start_idx, traj_end_idx = segment_map[idx]

            if ds_name == 'agiworld-on-site-pour':
                if idx > len(abs_joint) // 2: 
                    if random.random() < idx / len(abs_joint): 
                        ins = "stop and place the cup on the table."
            
            if ds_name == 'agiworld-on-site-microwave':
                if idx > 500: 
                    if random.random() < idx / len(abs_joint): 
                        ins = "close and start it."
                        
            rel = min(DOMAIN2CHUNKSIZE[ds_name] + 1, traj_end_idx - idx)
            seg = abs_ee6d[idx:idx + rel] if 'ee6d' in action_mode else abs_joint[idx:idx + rel]

            t_old = np.linspace(0.0, 1.0, seg.shape[0])
            t_new = np.linspace(0.0, 1.0, num_actions + 1)
            abs_trajectory = interp1d(t_old, seg, axis=0, kind='linear', bounds_error=False)(t_new)

            def load_image(name):
                if use_mp4:
                    return Image.fromarray(mp4_frames_cache[name][idx])
                else:
                    path = self._get_image_path(item, idx, name)
                    return Image.open(path).convert('RGB')

            if random.random() < 0.5:
                image_mask = torch.tensor([1, 1, 1]).to(torch.bool)
                imgs = [image_aug(load_image(n)) for n in image_names]
            else:
                image_mask = torch.tensor([1, 0, 0]).to(torch.bool)
                imgs = [image_aug(load_image(image_names[0]))]
                while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
            
            image_input = torch.stack(imgs, dim=0)
            
            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.from_numpy(abs_trajectory).float(),
                "frame_idx": idx,
            }