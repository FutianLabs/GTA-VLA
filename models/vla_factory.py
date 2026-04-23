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
VLA Model Factory.

Provides unified loading for different VLA architectures:
  - GTA-VLA / XVLA (with Florence2 or Qwen3-VL backbone)
  - OpenVLA family (openvla, openvla-oft, vla-adapter)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
import json
from typing import Any, Callable, Dict

import torch


@dataclass
class VLAComponents:
    model: Any
    processor: Any
    prepare_batch_fn: Callable[[Dict[str, Any], Any, Any, torch.device], Dict[str, torch.Tensor]]
    build_optimizer_fn: Callable[[Any], torch.optim.Optimizer]
    update_lr_fn: Callable[[torch.optim.Optimizer, int], None]


# ---------------------------------------------------------------------------
# Helper: Load config from file or pretrained path
# ---------------------------------------------------------------------------
def _load_config(config_cls, config_source: str | None, default_config=None):
    """Load config from JSON file, pretrained path, or return default."""
    if config_source is None:
        return default_config if default_config is not None else config_cls()
    if os.path.isfile(config_source):
        return config_cls.from_json_file(config_source)
    return config_cls.from_pretrained(config_source)


def _load_checkpoint_keep_mismatch(model, checkpoint_path: str) -> None:
    """
    Load checkpoint weights while keeping existing params for mismatched shapes.
    """
    from transformers.modeling_utils import load_state_dict as hf_load_state_dict

    model_state = model.state_dict()
    mismatched = []
    total_loaded = 0

    def _load_state_dict_file(path: str) -> None:
        nonlocal total_loaded
        state_dict = hf_load_state_dict(path)
        filtered = {}
        for key, value in state_dict.items():
            if key not in model_state:
                continue
            if value.shape != model_state[key].shape:
                mismatched.append(key)
                continue
            filtered[key] = value
        total_loaded += len(filtered)
        model.load_state_dict(filtered, strict=False)

    if os.path.isfile(checkpoint_path):
        _load_state_dict_file(checkpoint_path)
    else:
        candidates = [
            "model.safetensors",
            "pytorch_model.bin",
        ]
        found = None
        for name in candidates:
            path = os.path.join(checkpoint_path, name)
            if os.path.isfile(path):
                found = path
                break
        if found is not None:
            _load_state_dict_file(found)
        else:
            index_candidates = [
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            ]
            index_path = None
            for name in index_candidates:
                path = os.path.join(checkpoint_path, name)
                if os.path.isfile(path):
                    index_path = path
                    break
            if index_path is None:
                raise FileNotFoundError(f"No checkpoint weights found in {checkpoint_path}")
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = sorted(set(index.get("weight_map", {}).values()))
            if not shard_files:
                raise FileNotFoundError(f"No shard files listed in {index_path}")
            for shard in shard_files:
                _load_state_dict_file(os.path.join(checkpoint_path, shard))

    if mismatched:
        logging.warning("Skipped %d mismatched keys (kept existing weights).", len(mismatched))
    logging.info("Loaded %d checkpoint tensors.", total_loaded)



# ---------------------------------------------------------------------------
# GTA-VLA Loader (supports both Florence2 and Qwen3-VL backbones)
# ---------------------------------------------------------------------------
def _is_gtavla_checkpoint(path: str | None) -> bool:
    """Return True if path looks like a GTA-VLA/XVLA checkpoint directory."""
    if path is None or not os.path.isdir(path):
        return False
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    model_type = cfg.get("model_type")
    archs = cfg.get("architectures") or []
    return model_type in {"xvla", "gtavla"} or "XVLA" in archs or "GTAVLA" in archs


def _is_xvla_checkpoint(path: str | None) -> bool:
    return _is_gtavla_checkpoint(path)


_DEFAULT_FLORENCE_HUB = "microsoft/Florence-2-large"


def resolve_gtavla_vlm_path_for_processor(config) -> tuple[str, str]:
    vlm = (getattr(config, "vlm_backbone_type", "florence2") or "florence2").lower()
    if vlm == "qwen3_vl":
        return "qwen3_vl", getattr(config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct")
    florence_path = getattr(config, "florence_pretrained_name_or_path", None)
    if not florence_path:
        fc = getattr(config, "florence_config", None)
        if fc is not None:
            florence_path = getattr(fc, "_name_or_path", None) or getattr(fc, "name_or_path", None)
    if not florence_path:
        florence_path = (
            os.environ.get("GTAVLA_FLORENCE_PRETRAINED")
            or os.environ.get("XVLA_FLORENCE_PRETRAINED")
            or ""
        ).strip() or _DEFAULT_FLORENCE_HUB
    return "florence2", florence_path


def resolve_xvla_vlm_path_for_processor(config) -> tuple[str, str]:
    return resolve_gtavla_vlm_path_for_processor(config)


def load_gtavla_from_pretrained_dir(
    pretrained_model_name_or_path: str,
    *,
    num_views: int = 3,
    trust_remote_code: bool = True,
):
    from .configuration_gtavla import GTAVLAConfig
    from .modeling_gtavla import GTAVLA
    from .processing_gtavla import GTAVLAProcessor

    path = os.path.abspath(os.path.expanduser(pretrained_model_name_or_path))
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    config = GTAVLAConfig.from_pretrained(path)
    model = GTAVLA.from_pretrained(path, trust_remote_code=trust_remote_code)
    vlm_type, vlm_path = resolve_gtavla_vlm_path_for_processor(config)
    nv = int(num_views) if num_views is not None else int(getattr(config, "num_views", 3))
    if vlm_type == "qwen3_vl":
        processor = GTAVLAProcessor.from_pretrained_vlm(
            "qwen3_vl",
            vlm_path,
            num_views=nv,
            use_cot_training=getattr(config, "use_cot_training", False),
            cot_max_length=getattr(config, "cot_max_length", 768),
        )
    else:
        processor = GTAVLAProcessor.from_pretrained_vlm("florence2", vlm_path, num_views=nv)
    processor.num_views = nv
    return model, processor


def load_xvla_from_pretrained_dir(
    pretrained_model_name_or_path: str,
    *,
    num_views: int = 3,
    trust_remote_code: bool = True,
):
    return load_gtavla_from_pretrained_dir(
        pretrained_model_name_or_path,
        num_views=num_views,
        trust_remote_code=trust_remote_code,
    )


def _load_xvla(load_path: str | None, args) -> VLAComponents:
    """
    Load XVLA model with automatic backbone detection.
    
    The backbone (Florence2 or Qwen3-VL) is determined by config.vlm_backbone_type.
    
    Usage:
        # Florence2 (default)
        python train.py --model_arch gtavla \\
            --config_path configs/libero/from_scratch_abs_ee3d.json
        
        # Qwen3-VL
        python train.py --model_arch gtavla \\
            --config_path configs/libero/gtavla_qwen3vl_2b.json
    """
    from .configuration_gtavla import GTAVLAConfig
    from .modeling_gtavla import (
        GTAVLA,
        build_vla_optimizer,
        prepare_batch as gtavla_prepare_batch,
        update_vla_learning_rates,
    )
    from .processing_gtavla import GTAVLAProcessor, build_gtavla_processor

    checkpoint_path = None
    if args.resume_from_checkpoint is not None:
        checkpoint_path = args.resume_from_checkpoint
    elif _is_gtavla_checkpoint(load_path):
        checkpoint_path = load_path
    elif load_path is not None and args.config_path is None and args.vlm_pretrained is None:
        checkpoint_path = load_path
    loading_checkpoint = checkpoint_path is not None
    print("Loading checkpoint from:", checkpoint_path)
    if loading_checkpoint:
        config = None
        if args.config_path:
            config = _load_config(GTAVLAConfig, args.config_path)
        if config is None:
            config = _load_config(GTAVLAConfig, checkpoint_path)
        model = GTAVLA(config)
        _load_checkpoint_keep_mismatch(model, checkpoint_path)
        # Determine processor based on backbone type
        vlm_type = getattr(model.config, "vlm_backbone_type", "florence2")
        use_cot_training = getattr(model.config, "use_cot_training", False)
        
        if vlm_type == "qwen3_vl":
            qwen3_path = getattr(model.config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct")
            processor = build_gtavla_processor(
                vlm_backbone_type="qwen3_vl",
                pretrained_name_or_path=qwen3_path,
                num_views=getattr(model.config, 'num_views', 3),
                use_cot_training=use_cot_training,
                cot_max_length=getattr(model.config, 'cot_max_length', 768),
            )
        else:
            _, vlm_path = resolve_gtavla_vlm_path_for_processor(model.config)
            florence_path = args.vlm_pretrained or args.models or vlm_path
            processor = GTAVLAProcessor.from_pretrained_vlm(
                "florence2",
                florence_path,
                num_views=getattr(model.config, "num_views", 3),
            )
    else:
        config_source = args.config_path or load_path
        if config_source is None:
            raise ValueError("Training GTA-VLA from scratch requires --config_path pointing to a GTA-VLA config.")
        config = _load_config(GTAVLAConfig, config_source)
        
        vlm_type = getattr(config, "vlm_backbone_type", "florence2")
        use_cot_training = getattr(config, "use_cot_training", False)
        
        if vlm_type == "qwen3_vl":
            # Qwen3-VL backbone
            qwen3_path = args.vlm_pretrained or getattr(config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct")
            config.qwen3_pretrained = qwen3_path
            model = GTAVLA(config)
            processor = build_gtavla_processor(
                vlm_backbone_type="qwen3_vl",
                pretrained_name_or_path=qwen3_path,
                num_views=getattr(config, 'num_views', 3),
                use_cot_training=use_cot_training,
                cot_max_length=getattr(config, 'cot_max_length', 768),
            )
        else:
            # Florence2 backbone (default)
            florence_path = args.vlm_pretrained or args.models or getattr(config, "florence_pretrained_name_or_path", None)
            if florence_path:
                config.florence_pretrained_name_or_path = florence_path
            model = GTAVLA(config)
            processor = None
            if florence_path:
                processor = GTAVLAProcessor.from_pretrained(florence_path)

    def build_optimizer_fn(model_instance):
        return build_vla_optimizer(
            model_instance,
            base_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=tuple(args.betas),
            lr_coef_soft=args.learning_coef,
        )

    def update_lr_fn(optim: torch.optim.Optimizer, step: int):
        return update_vla_learning_rates(
            optim, step,
            learning_rate=args.learning_rate,
            learning_coef=args.learning_coef,
            freeze_steps=args.freeze_steps,
            warmup_steps=args.warmup_steps,
            total_iters=args.iters,
            use_cosine_decay=args.use_cosine_decay,
            min_lr_ratio=args.min_lr_ratio,
        )

    def prepare_fn(batch, _model_instance, processor_instance, device):
        if processor_instance is None:
            raise ValueError("GTA-VLA training requires a tokenizer/processor for language encoding.")
        return gtavla_prepare_batch(batch, processor_instance, device)

    return VLAComponents(
        model=model,
        processor=processor,
        prepare_batch_fn=prepare_fn,
        build_optimizer_fn=build_optimizer_fn,
        update_lr_fn=update_lr_fn,
    )


# ---------------------------------------------------------------------------
# OpenVLA Family Loader (openvla, openvla-oft, vla-adapter)
# ---------------------------------------------------------------------------
def _load_openvla_family(load_path: str | None, args, variant: str) -> VLAComponents:
    from .openvla.configuration_openvla import (
        OpenVLAAdapterConfig,
        OpenVLAConfig,
        OpenVLAOFTConfig,
    )
    from .openvla.modeling_openvla import (
        OpenVLA,
        OpenVLAAdapter,
        OpenVLAOFT,
        build_vla_optimizer,
        prepare_batch as openvla_prepare_batch,
        prepare_batch_adapter,
        prepare_batch_oft,
        update_vla_learning_rates,
    )

    # Variant-specific mappings
    VARIANT_MAP = {
        "openvla": (OpenVLA, OpenVLAConfig, openvla_prepare_batch, False),
        "openvla-oft": (OpenVLAOFT, OpenVLAOFTConfig, prepare_batch_oft, True),
        "vla-adapter": (OpenVLAAdapter, OpenVLAAdapterConfig, prepare_batch_adapter, True),
    }

    model_cls, config_cls, prepare_batch_fn, supports_action_chunk = VARIANT_MAP[variant]
    loading_checkpoint = args.resume_from_checkpoint is not None or load_path is not None

    if loading_checkpoint:
        model = model_cls.from_pretrained(
            load_path,
            ignore_mismatched_sizes=True,
        )
        processor = getattr(model, "processor", None)
    else:
        config = _load_config(config_cls, args.config_path or load_path)

        vlm_path = args.vlm_pretrained or args.models or getattr(config, "vlm_model_name", None)
        if vlm_path:
            config.vlm_model_name = vlm_path
            config.processor_name = getattr(config, "processor_name", None) or vlm_path
            config.tokenizer_name = getattr(config, "tokenizer_name", None) or vlm_path

        if supports_action_chunk and args.dataset_num_actions is not None:
            config.num_action_chunk = args.dataset_num_actions

        model = model_cls(config)
        processor = getattr(model, "processor", None)

    def build_optimizer_fn(model_instance):
        return build_vla_optimizer(
            model_instance,
            base_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=tuple(args.betas),
        )

    def update_lr_fn(optim: torch.optim.Optimizer, step: int):
        return update_vla_learning_rates(
            optim, step,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            total_iters=args.iters,
            use_cosine_decay=args.use_cosine_decay,
            min_lr_ratio=args.min_lr_ratio,
        )

    # Capture processor in closure for prepare_fn
    _processor = processor

    def prepare_fn(batch, model_instance, _processor_instance, device):
        return prepare_batch_fn(batch, model_instance, _processor, device)

    return VLAComponents(
        model=model,
        processor=processor,
        prepare_batch_fn=prepare_fn,
        build_optimizer_fn=build_optimizer_fn,
        update_lr_fn=update_lr_fn,
    )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
# Normalize architecture aliases to canonical names
_ARCH_ALIASES = {
    "gtavla": "gtavla",
    "gta-vla": "gtavla",
    "gta_vla": "gtavla",
    "xvla": "gtavla",
    "openvla": "openvla",
    "openvla-oft": "openvla-oft",
    "openvla_oft": "openvla-oft",
    "openvlaoft": "openvla-oft",
    "vla-adapter": "vla-adapter",
    "vla_adapter": "vla-adapter",
    "vlaadapter": "vla-adapter",
}


def load_vla_components(model_arch: str, load_path: str, args) -> VLAComponents:
    """
    Load VLA model components based on architecture type.
    
    The VLM backbone (Florence2 vs Qwen3-VL) is determined by the config file,
    not by the model_arch argument. Use config.vlm_backbone_type to specify.
    
    Supported architectures:
        - gtavla: GTA-VLA (backbone determined by config)
        - openvla: OpenVLA base model
        - openvla-oft: OpenVLA with OFT
        - vla-adapter: VLA adapter model
    
    Args:
        model_arch: Architecture identifier (see _ARCH_ALIASES for all options)
        load_path: Path to pretrained model or checkpoint
        args: Training arguments
    
    Returns:
        VLAComponents with model, processor, and training functions
    """
    arch = _ARCH_ALIASES.get(model_arch.lower())

    if arch is None:
        raise ValueError(f"Unknown model_arch '{model_arch}'. Supported: {list(_ARCH_ALIASES.keys())}")

    if arch == "gtavla":
        return _load_xvla(load_path, args)

    # OpenVLA family (openvla, openvla-oft, vla-adapter)
    return _load_openvla_family(load_path, args, arch)
