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

import os
import time
import json
import random
import os.path as osp
import argparse
import wandb
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field, fields, MISSING, asdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from datasets import create_dataloader
from models.vla_factory import load_vla_components
import logging
import sys


# ============================================================
# logger
# ============================================================
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Training Configuration
# ============================================================
@dataclass
class TrainConfig:
    # I/O
    models: Optional[str] = None  # Path or HF repo for pretrained VLA or backbone
    config_path: Optional[str] = None  # XVLA config json (required when training from scratch)
    vlm_pretrained: Optional[str] = None  # Florence checkpoint to initialize backbone weights
    output_dir: str = "runnings"  # Directory to save checkpoints
    train_metas_path: str = None  # Path to training metadata (required)
    model_arch: str = "xvla"  # Model family to train (xvla, openvla, ...)
    
    # Data
    batch_size: int = 16  # Per-device batch size
    dataset_num_actions: Optional[int] = None  # Override dataloader temporal horizon
    action_mode: str = "ee6d"  # Fallback action layout for dataloader
    image_color_jitter: str = "True"
    
    # Optimizer
    learning_rate: float = 1e-4  # Base learning rate
    learning_coef: float = 0.1  # LR multiplier for soft prompts and VLM
    weight_decay: float = 0.01  # Weight decay coefficient
    betas: Tuple[float, float] = (0.9, 0.95)  # Adam beta parameters
    max_grad_norm: float = 1.0  # Max gradient norm for clipping
    
    # Schedule
    iters: int = 1000000  # Total training iterations
    freeze_steps: int = 1000  # Steps to freeze VLM and transformer core
    warmup_steps: int = 2000  # Warmup steps after freeze period
    use_cosine_decay: bool = False  # Use cosine LR decay
    min_lr_ratio: float = 0.001  # Minimum LR ratio for cosine decay
    
    # Accelerate & Distributed Training
    grad_accum: int = 1  # Gradient accumulation steps
    mixed_precision: str = "bf16"  # Mixed precision training (no, fp16, bf16)
    deepspeed: Optional[str] = None  # Path to DeepSpeed config JSON (for deepspeed launcher)
    local_rank: int = -1  # Local rank for distributed training (set by deepspeed launcher)
    
    # Logging / Saving / Evaluation
    save_interval: int = 5000  # Save checkpoint every N steps
    log_interval: int = 20  # Log metrics every N steps
    wandb_project: str = "XVLA-Training"  # Wandb project name
    wandb_entity: Optional[str] = None  # Wandb entity (username or team)
    wandb_run_name: Optional[str] = None  # Custom wandb run name

    # System
    seed: int = 0  # Random seed
    num_workers: int = 4  # Number of data loading workers
    
    # Resume
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint to resume from
    # ------------------------
    # FSDP
    # ------------------------
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "full"               # full / shard_grad / no_shard
    fsdp_wrap_policy: str = "transformer"      # transformer / block / none
    fsdp_activation_checkpoint: bool = True

    # ------------------------
    # Gradient Checkpointing
    # ------------------------
    gradient_checkpointing: bool = False

    # ------------------------
    # Debug / Visualization
    # ------------------------
    debug_cot_interval: int = -1  # Visualize every N steps
    debug_cot_max_samples: int = 16  # Max samples to visualize per step


# ============================================================
# Argument Parser
# ============================================================
def get_args():
    """Parse command-line arguments and return TrainConfig."""
    parser = argparse.ArgumentParser("VLA Training", add_help=True)
    
    # Automatically add arguments from TrainConfig dataclass
    for f in fields(TrainConfig):
        field_name = f.name
        field_type = f.type
        default_value = f.default if f.default is not MISSING else (f.default_factory() if f.default_factory is not MISSING else None)
        
        # Extract actual type from Optional, List, Tuple, etc.
        origin_type = getattr(field_type, '__origin__', None)
        is_optional = origin_type is not None and type(None) in getattr(field_type, "__args__", [])
        
        kwargs = {'help': f.metadata.get('help', '')}
        
        # Handle different types
        if origin_type is tuple or (isinstance(default_value, tuple) and field_name == 'betas'):
            # Special handling for tuples like betas
            parser.add_argument(
                f"--{field_name}",
                type=float,
                nargs=2,
                default=list(default_value) if default_value else None,
                **kwargs
            )
        elif field_type == bool or (origin_type is None and default_value is False):
            # Boolean flags
            parser.add_argument(
                f"--{field_name}",
                action="store_true",
                default=default_value,
                **kwargs
            )
        elif origin_type is list:
            # List types
            args_type = field_type.__args__[0] if hasattr(field_type, '__args__') else str
            parser.add_argument(
                f"--{field_name}",
                type=args_type,
                nargs='+',
                default=default_value,
                **kwargs
            )
        else:
            # Simple types (str, int, float)
            base_type = field_type
            if is_optional and hasattr(field_type, "__args__"):
                non_none_args = [t for t in field_type.__args__ if t is not type(None)]
                base_type = non_none_args[0] if non_none_args else str
            actual_type = str if base_type == str else base_type
            
            parser.add_argument(
                f"--{field_name}",
                type=actual_type,
                default=default_value,
                required=(default_value is None and field_name in ['train_metas_path']),
                **kwargs
            )
    
    args = parser.parse_args()
    
    # Convert betas back to tuple if needed
    args_dict = vars(args)
    if 'betas' in args_dict and isinstance(args_dict['betas'], list):
        args_dict['betas'] = tuple(args_dict['betas'])
    
    return TrainConfig(**args_dict)


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True



def _get_dataset_image_processor(model, processor):
    """Get image_processor for dataset based on model VLM backbone."""
    if hasattr(model, 'vlm_backbone_type') and model.vlm_backbone_type == "qwen3_vl":
        if processor is not None and hasattr(processor, 'image_processor'):
            return processor.image_processor
    return None

def resolve_dataset_hparams(model, args: TrainConfig) -> Tuple[int, str]:
    num_actions = args.dataset_num_actions
    if num_actions is None:
        num_actions = getattr(model, "num_actions", 1)
    action_mode = getattr(model, "action_mode", args.action_mode)
    return int(num_actions), action_mode


def unpack_loss(outputs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(outputs, dict):
        tensor_losses = {k: v for k, v in outputs.items() if torch.is_tensor(v) and "loss" in k}
        if tensor_losses:
            # If 'loss' key exists, use it as total; otherwise sum all losses
            if "loss" in tensor_losses:
                total = tensor_losses["loss"]
            else:
                total = sum(tensor_losses.values())
            return total, tensor_losses
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss, {"loss": outputs.loss}
    if torch.is_tensor(outputs):
        return outputs, {"loss": outputs}
    raise ValueError("Unable to extract loss from model outputs.")

def return_fsdp_plugin(args):
    fsdp_plugin = None
    if args.use_fsdp:
        from accelerate.utils import FSDPPlugin
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        if args.fsdp_sharding_strategy == "full":
            sharding = ShardingStrategy.FULL_SHARD
        elif args.fsdp_sharding_strategy == "shard_grad":
            sharding = ShardingStrategy.SHARD_GRAD_OP
        elif args.fsdp_sharding_strategy == "no_shard":
            sharding = ShardingStrategy.NO_SHARD
        else:
            raise ValueError("unknown fsdp_sharding_strategy")

        # wrap policy
        def get_wrap_policy(mode):
            if mode == "transformer":
                return transformer_auto_wrap_policy(
                    transformer_layer_cls={
                        torch.nn.TransformerEncoderLayer,
                        torch.nn.TransformerDecoderLayer
                    }
                )
            elif mode == "block":
                return lambda m, *a, **kw: getattr(m, "block_type", False)
            elif mode == "none":
                return None
            else:
                raise ValueError(mode)

        wrap = get_wrap_policy(args.fsdp_wrap_policy)

        fsdp_plugin = FSDPPlugin(
            sharding_strategy=sharding,
            auto_wrap_policy=wrap,
            activation_checkpointing=args.fsdp_activation_checkpoint,
            state_dict_type="full",
            cpu_offload=False,
        )
    return fsdp_plugin
# ============================================================
# Main Training
# ============================================================
def main(args: TrainConfig):
    output_dir = Path(args.output_dir)
    args.output_dir = str(output_dir)
    fsdp_plugin = return_fsdp_plugin(args)
    
    # DeepSpeed plugin (when using deepspeed launcher with --deepspeed config)
    deepspeed_plugin = None
    if args.deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed)
    
    # DDP kwargs: find_unused_parameters=True needed for Qwen3-VL backbone
    # Some parameters (e.g., visual merger, unused layers) may not participate in loss
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # Initialize Accelerator with gradient accumulation and mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        log_with=["wandb"], 
        project_dir=output_dir,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[ddp_kwargs],
    )
    
    # Initialize trackers with configuration
    tracker_config = asdict(args)  # Convert dataclass to dict for logging
    init_kwargs = {}
    
    # Wandb configuration
    run_name = args.wandb_run_name if args.wandb_run_name else osp.basename(str(output_dir))
    if "libero" in str(output_dir):
        run_name = run_name + "_libero"
    elif "bridge" in str(output_dir):
        run_name = run_name + "_bridge"
    wandb_config = {
        "name": run_name,
        "id": run_name,  # Set id to match name for consistency
    }
    if args.wandb_entity:
        wandb_config["entity"] = args.wandb_entity
    init_kwargs["wandb"] = wandb_config
    
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=tracker_config,
        init_kwargs=init_kwargs
    )

    wandb_tracker = accelerator.get_tracker("wandb")    
    wandb_tracker.store_init_configuration(tracker_config)
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    if accelerator.is_main_process:
        try:
            run_id = wandb.run.id if wandb.run else None
        except Exception:
            run_id = None
        if run_id:
            run_id_path = output_dir / "wandb_run_id.txt"
            run_id_path.write_text(run_id + "\n", encoding="utf-8")
            logger.info("Saved wandb run id to %s", run_id_path)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")
    load_path = args.resume_from_checkpoint or args.models
    print("Loading model from:", load_path, args.resume_from_checkpoint, args.models)
    components = load_vla_components(args.model_arch, load_path, args)
    model = components.model
    processor = components.processor

    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        state_path = os.path.join(args.resume_from_checkpoint, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                resume_step = state.get("global_step", 0)
            logger.info(f"Resuming from step {resume_step}")
        else:
            resume_step = 0
            logger.warning("No state.json found, starting from step 0")
    else:
        resume_step = 0
    # ---------------------------------------------
    # Gradient Checkpointing
    # ---------------------------------------------
    if args.gradient_checkpointing:

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        transformer = getattr(model, "transformer", None)
        if transformer is not None and hasattr(transformer, "gradient_checkpointing_enable"):
            transformer.gradient_checkpointing_enable()

        # Enable gradient checkpointing on VLM backbone (Qwen3-VL, etc.)
        vlm = getattr(model, "vlm", None)
        if vlm is not None and hasattr(vlm, "gradient_checkpointing_enable"):
            vlm.gradient_checkpointing_enable()
            if hasattr(vlm, "config") and hasattr(vlm.config, "use_cache"):
                vlm.config.use_cache = False
            logger.info("🔧 VLM Gradient Checkpointing ENABLED.")

        logger.info("🔧 Gradient Checkpointing ENABLED.")
    # Dataloader
    num_actions, action_mode = resolve_dataset_hparams(model, args)

    # Auto-compute joint normalization stats before dataloader creation
    if action_mode == "joint" and not getattr(model.config, "joint_norm_stats", None):
        from datasets.domain_handler.piper_joint import ensure_norm_stats
        jstats = ensure_norm_stats(args.train_metas_path)
        if jstats:
            model.config.joint_norm_stats = jstats
            model.action_space.set_norm_stats(jstats)
            logger.info(f"📐 Joint norm stats loaded ({jstats['num_frames']} frames)")

    img_proc = _get_dataset_image_processor(model, processor)
    if img_proc:
        logger.info("📸 Using VLM processor in dataset (returns pre-embedded patches)")
    
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=num_actions,
        action_mode=action_mode,
        training=True,
        image_color_jitter=args.image_color_jitter,
        num_workers=args.num_workers,
        image_processor=img_proc,
        model_config=model.config,  # Pass config for CoT support
    )

    # Optimizer
    optim = components.build_optimizer_fn(model)
    
    # Set DeepSpeed batch size before prepare() if using DeepSpeed
    # This avoids prepare(dataloader) which fails on non-tensor batch fields (e.g. strings)
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.batch_size
        accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = args.grad_accum
    
    model, optim = accelerator.prepare(model, optim)

    # Training loop
    model.train()
    global_step, t0, micro_step = resume_step, time.time(), resume_step * args.grad_accum
    effective_batch_size = args.batch_size * args.grad_accum * accelerator.num_processes
    logger.info(
        f"🚀 Start training for {args.iters} iterations | "
        f"world_size={accelerator.num_processes} | "
        f"grad_accum={args.grad_accum} | "
        f"effective_batch_size={effective_batch_size}"
    )
    
    for batch in train_dataloader:
        # Debug: Visualize CoT data
        if accelerator.is_main_process:
            if args.debug_cot_interval > 0 and global_step % args.debug_cot_interval == 0:
                try:
                    from scripts.debug_cot import visualize_cot_batch
                    cot_output_dir = os.path.join(args.output_dir, "cot_debug")
                    visualize_cot_batch(
                        batch,
                        output_dir=cot_output_dir,
                        max_samples=args.debug_cot_max_samples,
                        step=global_step,
                    )
                except ImportError:
                    logging.warning("scripts.debug_cot not found, skipping CoT visualization")
        
        model_ref = accelerator.unwrap_model(model)
        inputs = components.prepare_batch_fn(batch, model_ref, processor, accelerator.device)
        # Update LR per group (only at the start of accumulation cycle)
        if micro_step % args.grad_accum == 0:
            components.update_lr_fn(optim, global_step)

        # Forward & backward with gradient accumulation
        with accelerator.accumulate(model):
            outputs = model(**inputs)
            loss, loss_dict = unpack_loss(outputs)
            accelerator.backward(loss)
            
            # Only step optimizer after accumulating gradients
            if accelerator.sync_gradients:
                if args.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                optim.zero_grad()
        
        micro_step += 1
        
        # Update global step only after accumulation
        if accelerator.sync_gradients:

            # Logging (only after gradient sync)
            if global_step % args.log_interval == 0:

                logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
                logs["loss_total"] = float(loss.detach().item())
                for idx, group in enumerate(optim.param_groups):
                    key = group.get("name", f"group{idx}")
                    logs[f"lr_{key}"] = group["lr"]
                logs["global_step"] = global_step
                logs["micro_step"] = micro_step
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    dt = (time.time() - t0) / args.log_interval
                    t0 = time.time()

                    # ETA -------------------------------------
                    steps_left = args.iters - global_step
                    eta_seconds = steps_left * dt
                    eta_h = int(eta_seconds // 3600)
                    eta_m = int((eta_seconds % 3600) // 60)
                    eta_s = int(eta_seconds % 60)
                    eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                    finish_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta_seconds)
                    )
                    lr_pairs = []
                    for idx, group in enumerate(optim.param_groups):
                        key = group.get("name", f"group{idx}")
                        lr_pairs.append(f"{key}={group['lr']:.2e}")
                    lr_info = " ".join(lr_pairs)
                    loss_specific_info = " | ".join([f"{k}={v:.4f}" for k, v in logs.items() if k not in ("loss_total", "loss")])
                    logger.info(
                        f"[{global_step}/{args.iters}] "
                        f"loss={logs['loss_total']:.4f} "
                        f"{loss_specific_info} "
                        f"{lr_info} "
                        f"({dt:.2f}s/it) | ETA {eta_str} | End ~ {finish_time}"
                    )
            global_step += 1
            
            # Checkpointing (only after gradient sync)
            if accelerator.is_main_process:
                if global_step == args.iters or global_step % args.save_interval == 0:
                    save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                    accelerator.print(f"💾 Saving model to {save_dir}")
                    accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                    with open(os.path.join(save_dir, "state.json"), "w") as f:
                        json.dump({"global_step": global_step, "micro_step": micro_step}, f)
            
            if global_step >= args.iters:
                break

    accelerator.end_training()

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
