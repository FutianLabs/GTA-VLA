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

import io
import random
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Optional, Sequence, Any

import numpy as np
import h5py
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d
import os.path as osp
class DomainHandler(ABC):
    """
    Minimal domain handler interface.

    Subclasses provide dataset-specific decoding by implementing an iterator
    that yields per-sample dictionaries compatible with the training loop.
    """
    dataset_name: str

    def __init__(self, meta: dict, num_views: int) -> None:
        self.meta = meta
        self.num_views = num_views

    @abstractmethod
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """Yield samples for a single episode."""
        ...


def _open_h5(path: str) -> h5py.File:
    """Open HDF5 from local FS or remote backend via mmengine.fileio."""
    try:
        return h5py.File(path, "r")
    except OSError:
        return h5py.File(io.BytesIO(fileio.get(path)), "r")


class BaseHDF5Handler(DomainHandler):
    """
    Generic HDF5 handler with resource-safe iteration.
    
    Supports unified image processing via image_processor parameter:
      - image_processor=None: Uses image_aug transforms (Florence2) → [C, H, W]
      - image_processor provided: Uses VLM processor (Qwen3-VL) → [N, 1536]

    Subclasses only implement:
      - build_left_right(f) -> (left, right, left_time, right_time, freq, qdur)
          left/right: abs_trajectory [T, C], left_time/right_time: optional time arrays [T],
          freq (Hz), qdur (seconds of future window)
      - index_candidates(T_left, training) -> Iterable[int]

    Optionally override:
      - get_image_datasets(f): sequence of image arrays/datasets
      - read_instruction(f): string instruction
    """

    # --- Optional overrides -------------------------------------------------
    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys: Sequence[str] = self.meta["observation_key"]
        return [f[k][()] for k in keys]

    def get_optional_image_datasets(self, f: h5py.File) -> Tuple[Sequence[Any], Sequence[str]]:
        if "optional_view_key" not in self.meta:
            return [], []
        valid_keys = [k for k in self.meta["optional_view_key"] if k in f]
        return [f[k][()] for k in valid_keys], valid_keys

    def read_instruction(self, f: h5py.File) -> str:
        raw_key = self.meta["language_instruction_key"]
        keys = raw_key if isinstance(raw_key, list) else [raw_key]
        for key in keys:
            if key in f.attrs:
                v = f.attrs[key]
                result = v.decode() if isinstance(v, bytes) else v
                if isinstance(result, str) and result.strip():
                    return result
            elif key in f:
                ds = f[key]
                v = ds[()]
                result = v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()
                if isinstance(result, str) and result.strip():
                    return result
        return ""

    # --- Required hooks -----------------------------------------------------
    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        raise NotImplementedError

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        raise NotImplementedError
    # -----------------------------------------------------------------------

    @staticmethod
    def _pil_from_arr(arr: Any, raw_image_shape=None) -> Image.Image:
        from ..utils import decode_image_from_bytes
        if not isinstance(arr, Image.Image):
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                arr = Image.fromarray(arr)
            else:
                arr = decode_image_from_bytes(arr, raw_image_shape=raw_image_shape)
        return arr.convert("RGB")

    def _process_single_image(self, pil_img, image_processor, image_aug):
        """Process a single PIL image into tensor + optional grid_thw metadata."""
        if image_processor is not None:
            processed = image_processor([pil_img], return_tensors="pt")
            img_tensor = processed['pixel_values']
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            grid_thw = processed.get('image_grid_thw', None)
            return img_tensor, grid_thw
        elif image_aug is not None:
            return image_aug(pil_img), None
        else:
            raise ValueError("Either image_processor or image_aug must be provided")

    def get_proprio_and_action(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        proprio = f["proprio"][()]                     # [T, >=6]
        action  = f["action"][()]    
        return proprio, action

    def get_image_and_proprio(self, traj_idx: int):
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]

        with _open_h5(datapath) as f:
            # Images and mask
            images = self.get_image_datasets(f)
            optional_images, optional_keys = self.get_optional_image_datasets(f)
            # Language
            if "language_instruction_key" in self.meta.keys():
                ins = self.read_instruction(f)
            else:
                ins = osp.basename(datapath) # the patch for libero;
            # Domain-specific kinematics and timing
            proprio, action= self.get_proprio_and_action(f)
        return images, optional_images, optional_keys, ins, proprio, action

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        image_processor=None,
        **kwargs
    ) -> Iterable[dict]:
        """
        Open once, yield many samples; file is always closed on exit.
        
        Args:
            image_processor: VLM-specific image processor
                - None: Use image_aug transforms (Florence2) → returns [C, H, W]
                - Provided: Use processor (Qwen3-VL) → returns [num_patches, embed_dim]
        """
        images, optional_images, opt_keys, ins, proprio, action = self.get_image_and_proprio(traj_idx)
        left, right, lt, rt, freq, qdur = self.build_left_right(proprio, action)
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        if lt is None: lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
        if rt is None: rt = np.arange(right.shape[0], dtype=np.float64) / float(freq)

        # Raw image shape dict: observation_key -> (H, W, 3)
        # For datasets that store flat uint8 pixels (e.g. some RoboMIND real-robot data)
        raw_shape_dict = self.meta.get("raw_image_shape")  # dict or None
        obs_keys = self.meta["observation_key"]

        def _resolve_shape(key):
            if raw_shape_dict is None:
                return None
            s = raw_shape_dict.get(key)
            return tuple(s) if s is not None else None

        # Build per-view shape lists parallel to images / optional_images
        view_shapes = [_resolve_shape(k) for k in obs_keys]
        opt_shapes = [_resolve_shape(k) for k in opt_keys]

        # Optional view augmentation: pool main view with optional views
        aug_optional = (
            training
            and getattr(self, 'aug_with_optional_view', False)
            and len(optional_images) > 0
        )
        if aug_optional:
            optional_pool = list(zip([images[0]] + list(optional_images),
                                     [view_shapes[0]] + opt_shapes))
        else:
            optional_pool = None

        # Candidate indices (optionally shuffled)
        idxs = list(self.index_candidates(left.shape[0], training))
        if training: random.shuffle(idxs)

        # Interpolators; clamp to endpoints
        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        ref = (lt + rt) / 2.0

        V = min(self.num_views, len(images))
        dual_freq = getattr(self, 'use_dual_frequency', False)
        vlm_max_offset = getattr(self, 'vlm_frame_max_offset', 10)
        output = []
        for idx in idxs:
            cur = ref[idx]
            q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
            lseq = torch.tensor(L(q))
            rseq = torch.tensor(R(q))

            # Skip static segments
            if (lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5: continue
            
            # Language augmentation
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            # Per-sample: pick which source to use for view slot 0
            if optional_pool:
                view_0_source, view_0_shape = random.choice(optional_pool)
            else:
                view_0_source, view_0_shape = images[0], view_shapes[0] if view_shapes else None
            
            imgs = []
            raw_imgs = []  # Store raw images for visualization
            grid_thw_list = []
            
            resize_short = self.meta.get("resize_short_side")

            for v in range(V):
                src = view_0_source if v == 0 else images[v]
                shape = view_0_shape if v == 0 else (view_shapes[v] if v < len(view_shapes) else None)
                pil_img = self._pil_from_arr(src[idx], raw_image_shape=shape)
                if resize_short is not None:
                    pil_img = pil_img.resize((resize_short, resize_short), Image.BILINEAR)

                raw_img = pil_img.resize((224, 224))
                raw_tensor = torch.from_numpy(np.array(raw_img)).permute(2, 0, 1).float() / 255.0
                raw_imgs.append(raw_tensor)
                
                if image_processor is not None:
                    processed = image_processor([pil_img], return_tensors="pt")
                    img_tensor = processed['pixel_values']
                    if img_tensor.dim() == 4:
                        img_tensor = img_tensor.squeeze(0)
                    imgs.append(img_tensor)
                    if 'image_grid_thw' in processed:
                        grid_thw_list.append(processed['image_grid_thw'])
                elif image_aug is not None:
                    img_tensor = image_aug(pil_img)
                    imgs.append(img_tensor)            
            # Pad to num_views
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
                if image_processor is not None and grid_thw_list:
                    grid_thw_list.append(torch.zeros(1, 3, dtype=torch.long))
            while len(raw_imgs) < self.num_views:
                raw_imgs.append(torch.zeros_like(raw_imgs[0]))
            
            # Stack images - handle different formats
            if imgs and imgs[0].dim() == 2:
                max_patches = max(img.shape[0] for img in imgs)
                padded_imgs = [
                    torch.nn.functional.pad(img, (0, 0, 0, max_patches - img.shape[0])) 
                    if img.shape[0] < max_patches else img
                    for img in imgs
                ]
                image_input = torch.stack(padded_imgs[:self.num_views], dim=0)
            else:
                image_input = torch.stack(imgs[:self.num_views], dim=0)
            
            lseq[:, -1] = torch.clamp(lseq[:, -1], -1, 1)
            sample = {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float(),
                "frame_idx": idx,
                "raw_images": torch.stack(raw_imgs[:self.num_views], dim=0),
            }
            
            if grid_thw_list:
                sample["image_grid_thw"] = torch.cat(grid_thw_list, dim=0)

            # --- Dual-frequency: sample VLM frame (previous frame, main view only) ---
            if dual_freq and training:
                vlm_offset = random.randint(0, vlm_max_offset)
                vlm_idx = max(0, idx - vlm_offset)
                vlm_src = view_0_source
                vlm_shape = view_0_shape
                vlm_pil = self._pil_from_arr(vlm_src[vlm_idx], raw_image_shape=vlm_shape)
                if resize_short is not None:
                    vlm_pil = vlm_pil.resize((resize_short, resize_short), Image.BILINEAR)

                vlm_img_tensor, vlm_grid_thw = self._process_single_image(
                    vlm_pil, image_processor, image_aug
                )
                sample["vlm_image_input"] = vlm_img_tensor.unsqueeze(0)  # [1, ...]
                sample["vlm_image_mask"] = torch.tensor([True], dtype=torch.bool)
                sample["vlm_frame_idx"] = vlm_idx
                if vlm_grid_thw is not None:
                    sample["vlm_image_grid_thw"] = vlm_grid_thw  # [1, 3]

            output.append(sample)
        return output



