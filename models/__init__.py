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
X-VLA Models Package

Provides Vision-Language-Action models with multiple VLM backbone support:

Supported Backbones:
  - Florence2 (default): encoder-decoder, 0.9B params
  - Qwen3-VL: decoder-only, 2B params (requires transformers >= 4.49.0)

Usage:
    # Florence2 backbone (default, backward compatible)
    from models import XVLA, XVLAConfig
    
    config = XVLAConfig.from_pretrained("configs/libero/from_scratch_abs_ee3d.json")
    model = XVLA(config)
    
    # Qwen3-VL backbone
    config = XVLAConfig(
        vlm_backbone_type="qwen3_vl",
        qwen3_pretrained="Qwen/Qwen3-VL-2B-Instruct",
    )
    model = XVLA(config)
    
    # Or load from config file
    config = XVLAConfig.from_pretrained("configs/libero/xvla_qwen3vl_2b.json")
    model = XVLA(config)
"""

# XVLA model and config
from .configuration_xvla import XVLAConfig
from .modeling_xvla import (
    XVLA,
    prepare_batch,
    build_vla_optimizer,
    update_vla_learning_rates,
)

# XVLAActionToken (VQ-VAE based action tokenizer)
# TODO: Implement modeling_xvla_action_token.py
# from .modeling_xvla_action_token import XVLAActionToken

# Processor
from .processing_xvla import XVLAProcessor, build_xvla_processor

# Transformer components
from .transformer import SoftPromptedTransformer

# Action space
from .action_hub import build_action_space

__all__ = [
    # Model
    "XVLA",
    # "XVLAActionToken",  # TODO: implement
    "XVLAConfig",
    # Processor
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
