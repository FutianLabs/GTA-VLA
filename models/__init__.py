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
"""
GTA-VLA Models Package

Provides Vision-Language-Action models with multiple VLM backbone support:

Supported Backbones:
  - Florence2 (default): encoder-decoder, 0.9B params
  - Qwen3-VL: decoder-only, 2B params (requires transformers >= 4.49.0)

Usage:
    # Florence2 backbone (default, backward compatible)
    from models import GTAVLA, GTAVLAConfig
    
    config = GTAVLAConfig.from_pretrained("configs/libero/from_scratch_abs_ee3d.json")
    model = GTAVLA(config)
    
    # Qwen3-VL backbone
    config = GTAVLAConfig(
        vlm_backbone_type="qwen3_vl",
        qwen3_pretrained="Qwen/Qwen3-VL-2B-Instruct",
    )
    model = GTAVLA(config)
    
    # Or load from config file
    config = GTAVLAConfig.from_pretrained("configs/libero/gtavla_qwen3vl_2b.json")
    model = GTAVLA(config)
"""

# GTA-VLA model and config
from .configuration_gtavla import GTAVLAConfig, XVLAConfig
from .modeling_gtavla import (
    GTAVLA,
    XVLA,
    prepare_batch,
    build_vla_optimizer,
    update_vla_learning_rates,
)

# XVLAActionToken (VQ-VAE based action tokenizer)
# TODO: Implement modeling_xvla_action_token.py
# from .modeling_xvla_action_token import XVLAActionToken

# Processor
from .processing_gtavla import GTAVLAProcessor, XVLAProcessor, build_gtavla_processor, build_xvla_processor

# Transformer components
from .transformer import SoftPromptedTransformer

# Action space
from .action_hub import build_action_space

__all__ = [
    # Model
    "GTAVLA",
    "XVLA",
    # "XVLAActionToken",  # TODO: implement
    "GTAVLAConfig",
    "XVLAConfig",
    # Processor
    "GTAVLAProcessor",
    "build_gtavla_processor",
    "XVLAProcessor",
    "build_xvla_processor",
    # Training helpers
    "prepare_batch",
    "build_vla_optimizer",
    "update_vla_learning_rates",
    # Components
    "SoftPromptedTransformer",
    "build_action_space",
]
