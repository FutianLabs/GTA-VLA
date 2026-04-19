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

import torch
from torch.utils.data import DataLoader
import json

def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed); random.seed(base_seed); torch.manual_seed(base_seed)


def get_or_compute_action_stats(
    metas_path: str,
    num_actions: int,
    action_mode: str,
) -> dict:
    """
    Get action stats from meta file, or compute and save if not present.
    
    Args:
        metas_path: Path to meta JSON file
        num_actions: Number of actions per sample
        action_mode: Action mode (e.g., 'ee6d')
    
    Returns:
        Dict with 'mean', 'std', 'num_samples', 'action_dim'
    """
    with open(metas_path, 'r') as f:
        meta = json.load(f)
    
    if "action_stats" in meta:
        print(f"Found existing action_stats in {metas_path}")
        return meta["action_stats"]
    
    print(f"action_stats not found in {metas_path}, computing...")
    from .dataset import InfiniteDataReader
    dataset = InfiniteDataReader(
        metas_path=metas_path,
        num_actions=num_actions,
        training=False,
        action_mode=action_mode,
        image_color_jitter=False,
    )
    return dataset.compute_action_stats(metas_path)


def create_dataloader(batch_size: int, 
                      metas_path: str, 
                      num_actions: int,
                      training: bool,
                      action_mode: str,
                      image_color_jitter: bool = True,
                      num_workers: int = 4,
                      persistent_workers: bool = True,
                      image_processor = None,
                      model_config = None,
                      ):
    """
    Create dataloader with VLM-specific image processing.
    
    Args:
        image_processor: VLM-specific image processor
            - None: Default ImageNet transforms (Florence2)
                   Output: [B, V, C, H, W] raw pixels
            - Provided: VLM processor (Qwen3-VL)
                   Output: [B, V, num_patches, embed_dim] pre-embedded
        model_config: XVLAConfig for CoT support. If use_cot_training=True,
            will automatically use CoT handlers for supported datasets.
    """
    from .dataset import InfiniteDataReader
    return DataLoader(
        InfiniteDataReader(
            metas_path, 
            num_actions=num_actions, 
            training=training,
            action_mode=action_mode, 
            image_color_jitter=image_color_jitter,
            image_processor=image_processor,
            model_config=model_config,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    