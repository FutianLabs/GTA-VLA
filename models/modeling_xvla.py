"""
XVLA: Vision-Language-Action Model with Multi-Backbone Support.

Supports two VLM backbones:
  - Florence2 (default): encoder-decoder, 0.9B params
  - Qwen3-VL: decoder-only, 2B params (requires transformers >= 4.49.0)

The backbone is automatically selected based on config.vlm_backbone_type.

Usage:
    # Florence2 backbone (default, backward compatible)
    config = XVLAConfig(florence_config={...})
    model = XVLA(config)
    
    # Qwen3-VL backbone
    config = XVLAConfig(
        vlm_backbone_type="qwen3_vl",
        qwen3_pretrained="Qwen/Qwen3-VL-2B-Instruct",
    )
    model = XVLA(config)
"""

from __future__ import annotations

import logging
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import PreTrainedModel
from PIL import Image

from .modeling_florence2 import Florence2ForConditionalGeneration
from .transformer import SoftPromptedTransformer
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype_str, torch.bfloat16)


def resolve_florence_pretrained_path(name_or_path: str) -> str:
    raw = name_or_path.rstrip("/")
    expanded = os.path.expanduser(raw)
    if os.path.isdir(expanded) or os.path.isfile(expanded):
        return expanded
    if os.path.isabs(expanded) or raw.startswith(("./", "../")):
        raise FileNotFoundError(
            f"Florence 权重路径不存在: {name_or_path!r}。"
            "请把 Florence-2-large 同步到该节点，或把 florence_pretrained_name_or_path 改为 Hub 上的 microsoft/Florence-2-large。"
        )
    return raw


class XVLA(PreTrainedModel):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • VLM backbone (Florence2 or Qwen3-VL)
      • SoftPromptedTransformer (temporal/action head)
      • Action space (pre/post-processing + loss)
    
    The VLM backbone is selected automatically based on config.vlm_backbone_type:
      - "florence2" (default): Uses Florence2ForConditionalGeneration
      - "qwen3_vl": Uses transformers.Qwen3VLForConditionalGeneration
    
    Attributes:
        vlm_backbone_type (str): Current backbone type
        vlm: The VLM backbone model
        transformer: Action prediction transformer
        action_space: Action preprocessing and loss
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        # VLM backbone type
        self.vlm_backbone_type: str = getattr(config, "vlm_backbone_type", "florence2").lower()
        # Dual-frequency training
        self.use_dual_frequency: bool = getattr(config, "use_dual_frequency", False)
        
        # CoT settings (Qwen3-VL only; Florence2 does not support CoT)
        self.use_cot_training: bool = getattr(config, 'use_cot_training', False)
        self.cot_loss_weight: float = getattr(config, 'cot_loss_weight', 1.0)
        self.cot_only_pretrain: bool = getattr(config, 'cot_only_pretrain', False)
        if self.use_cot_training and self.vlm_backbone_type != "qwen3_vl":
            raise ValueError(
                f"CoT training is only supported with Qwen3-VL backbone, "
                f"got vlm_backbone_type='{self.vlm_backbone_type}'"
            )
        if self.cot_only_pretrain and not self.use_cot_training:
            raise ValueError("cot_only_pretrain=True requires use_cot_training=True")
        
        # Action space (dimensions + hooks)
        self.action_space = build_action_space(config.action_mode.lower())
        if getattr(config, "joint_norm_stats", None) and hasattr(self.action_space, "set_norm_stats"):
            self.action_space.set_norm_stats(config.joint_norm_stats)
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)
        # Initialize VLM backbone based on type
        if self.vlm_backbone_type == "qwen3_vl":
            projection_dim = self._init_qwen3vl(config)
        else:
            projection_dim = self._init_florence2(config)

        # Temporal/action head (same for all backbones)
        self.use_soft_prompt = getattr(config, 'use_soft_prompt', True)
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            use_multilayer_proj=getattr(config, "use_multilayer_proj", False),
            proj_num_layers=getattr(config, "proj_num_layers", 2),
            proj_hidden_size=getattr(config, "proj_hidden_size", None),
            proj_dropout=getattr(config, "proj_dropout", 0.1),
            use_soft_prompt=self.use_soft_prompt,
            use_main_view=self.use_dual_frequency,
        )

        # Freeze action transformer when only pre-training CoT
        if self.cot_only_pretrain:
            for p in self.transformer.parameters():
                p.requires_grad = False
            logging.info("cot_only_pretrain=True: action transformer frozen, only CoT loss will be computed")

        # Deferred FastAPI app
        self.app = None

    # ======================= Backbone Initialization =======================
    def _init_florence2(self, config: XVLAConfig) -> int:
        """Initialize Florence2 backbone (default)."""
        florence_pretrained = getattr(config, "florence_pretrained_name_or_path", "microsoft/Florence-2-large")
        if florence_pretrained:
            florence_pretrained = resolve_florence_pretrained_path(florence_pretrained)
            config.florence_pretrained_name_or_path = florence_pretrained
            self.vlm = Florence2ForConditionalGeneration.from_pretrained(
                florence_pretrained,
                config=config.florence_config,
            ).to(torch.float32)
            # Keep config in sync with the loaded backbone
            self.config.florence_config = self.vlm.config
        else:
            self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)
        
        # Remove decoder (encoder-only mode)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head
        return self.vlm.config.projection_dim
    
    def _init_qwen3vl(self, config: XVLAConfig) -> int:
        """Initialize Qwen3-VL backbone. Requires transformers >= 4.49.0."""
        from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

        pretrained_path = getattr(config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct")
        torch_dtype = _get_torch_dtype(getattr(config, "qwen3_torch_dtype", "bfloat16"))
        attn_impl = "flash_attention_2" if getattr(config, "qwen3_use_flash_attn", True) else "sdpa"

        logging.info(f"Loading Qwen3-VL from {pretrained_path}")
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_path,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.qwen3_tokenizer = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)

        if self.use_cot_training:
            # Processor adds CoT tokens; replicate here so embedding resize happens
            additional_tokens = [
                "<|cot_start|>", "<|cot_end|>",
                "<|objects_start|>", "<|objects_end|>",
                "<|pick_start|>", "<|pick_end|>",
                "<|place_start|>", "<|place_end|>",
                "<|affordance_2d_start|>", "<|affordance_2d_end|>",
                "<|gripper_path_2d_start|>", "<|gripper_path_2d_end|>",
            ]
            existing = list(self.qwen3_tokenizer.additional_special_tokens)
            new_tokens = [t for t in additional_tokens if t not in existing]
            if new_tokens:
                self.qwen3_tokenizer.add_special_tokens(
                    {"additional_special_tokens": existing + new_tokens}
                )
                self.vlm.resize_token_embeddings(len(self.qwen3_tokenizer), mean_resizing=False)
                logging.info(f"Added {len(new_tokens)} CoT special tokens")

            self.qwen3_cot_start_id = self.qwen3_tokenizer.convert_tokens_to_ids("<|cot_start|>")
            self.qwen3_cot_end_id = self.qwen3_tokenizer.convert_tokens_to_ids("<|cot_end|>")
        else:
            self.qwen3_cot_start_id = None
            self.qwen3_cot_end_id = None

        return self.vlm.config.text_config.hidden_size
    
    # ======================= VLM Encoding (backbone-agnostic) =======================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
        image_grid_thw: Optional[torch.LongTensor] = None,  # [B, V, 3] for Qwen3-VL
        labels: Optional[torch.LongTensor] = None,  # [B, L] CoT labels (Qwen3-VL only)
        return_hidden_states: bool = False,           # CoT hidden states (Qwen3-VL only)
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text + multi-view images via VLM backbone.

        Args:
          labels: CoT target labels. Qwen3-VL only, ignored for Florence2.
          return_hidden_states: Return all hidden states for CoT. Qwen3-VL only.

        Returns:
          { "vlm_features": [B, T_enc, D], "aux_visual_inputs": [B, (V-1)*N, D] }
        """
        if self.vlm_backbone_type == "qwen3_vl":
            return self._forward_vlm_qwen3(input_ids, pixel_values, image_mask, image_grid_thw, labels, return_hidden_states)
        else:
            return self._forward_vlm_florence2(input_ids, pixel_values, image_mask)
    
    def _forward_vlm_florence2(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Florence2-specific encoding."""
        B, V = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)

        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        N, D = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]

        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)
        # Per-token mask for aux views: [B, (V-1)*N]
        aux_view_mask = image_mask[:, 1:].to(torch.bool)
        aux_token_mask = aux_view_mask.unsqueeze(-1).expand(-1, -1, N).reshape(B, -1)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs, "aux_token_mask": aux_token_mask}
    
    def _forward_vlm_qwen3(
        self, input_ids, pixel_values, image_mask, image_grid_thw=None, labels=None, return_hidden_states=False
    ) -> Dict[str, torch.Tensor]:
        """Qwen3-VL forward: get_image_features + masked_scatter + M-RoPE + DeepStack."""
        B, V, N, C = pixel_values.shape
        flat_images = pixel_values.view(B * V, N, C)
        flat_mask = image_mask.view(-1).to(torch.bool)

        if flat_mask.sum().item() == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_grid_thw = image_grid_thw.view(-1, 3)[flat_mask] if image_grid_thw is not None else None

        # Extract only valid patches (remove padding) based on grid_thw.
        # patches_per_view = T*H*W — raw patch count BEFORE spatial merge.
        # spatial_merge_size is applied after the visual encoder, not here.
        if valid_grid_thw is not None:
            patches_per_view = valid_grid_thw.prod(dim=-1).tolist()
            valid_patches = []
            for i, num_patches in enumerate(patches_per_view):
                valid_patches.append(valid_images[i, :int(num_patches)])
            flat_pv = torch.cat(valid_patches, dim=0)
        else:
            flat_pv = valid_images.reshape(-1, valid_images.shape[-1])
        
        image_embeds_list, deepstack_all = self.vlm.get_image_features(flat_pv, valid_grid_thw)

        inputs_embeds = self.vlm.model.language_model.embed_tokens(input_ids)
        target_dtype = next(self.vlm.model.language_model.parameters()).dtype
        hidden_dim = inputs_embeds.shape[-1]
        pad_id = self.qwen3_tokenizer.pad_token_id
        attention_mask = (input_ids != pad_id).long()

        view0_mask = torch.zeros(B * V, dtype=torch.bool, device=flat_mask.device)
        view0_mask[0::V] = True
        view0_among_valid = view0_mask[flat_mask]
        prefix_feats_flat = torch.cat(
            [emb for emb, is_v0 in zip(image_embeds_list, view0_among_valid) if is_v0], dim=0
        )
        # [B, 3] — get_rope_index expects flat [num_images, 3], one row per image in the batch
        prefix_grid_thw = image_grid_thw[:, 0, :] if image_grid_thw is not None else None

        tokens_per_image = image_embeds_list[0].shape[0]
        prefix_deepstack = None
        if deepstack_all:
            prefix_deepstack = [
                ds.reshape(len(image_embeds_list), tokens_per_image, -1)[view0_among_valid].reshape(-1, ds.shape[-1])
                for ds in deepstack_all
            ]

        num_valid = int(flat_mask.sum().item())
        all_features = torch.zeros(B * V, tokens_per_image, hidden_dim, device=inputs_embeds.device, dtype=target_dtype)
        all_features[flat_mask] = torch.cat(image_embeds_list, dim=0).reshape(num_valid, tokens_per_image, hidden_dim).to(dtype=target_dtype)
        image_features = all_features.view(B, V, tokens_per_image, hidden_dim)
        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, hidden_dim)
        aux_token_mask = image_mask[:, 1:].to(torch.bool).unsqueeze(-1).expand(-1, -1, tokens_per_image).reshape(B, -1)

        image_pad_mask = (input_ids == self.vlm.config.image_token_id)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_pad_mask.unsqueeze(-1).expand_as(inputs_embeds),
            prefix_feats_flat.to(dtype=target_dtype, device=inputs_embeds.device),
        )

        merged_labels = None
        if labels is not None:
            merged_labels = labels.clone()
            merged_labels[image_pad_mask] = -100

        if prefix_grid_thw is not None:
            prefix_grid_thw = prefix_grid_thw.to(device=input_ids.device, dtype=torch.long)
        position_ids, _ = self.vlm.model.get_rope_index(input_ids, prefix_grid_thw, None, attention_mask)

        use_multi_layer = getattr(self.config, 'vlm_use_multi_layer', False)
        hidden_layer = getattr(self.config, 'vlm_hidden_layer', -1)
        need_all_hidden_states = use_multi_layer or hidden_layer != -1 or return_hidden_states

        outputs = self.vlm.model.language_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds.to(dtype=target_dtype),
            position_ids=position_ids,
            visual_pos_masks=image_pad_mask,
            deepstack_visual_embeds=prefix_deepstack,
            output_hidden_states=need_all_hidden_states,
            return_dict=True,
        )

        if need_all_hidden_states:
            all_hidden_states = outputs.hidden_states
            enc_out = self._extract_vlm_hidden_states(all_hidden_states)
        else:
            enc_out, all_hidden_states = outputs.last_hidden_state, None

        result = {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs, "aux_token_mask": aux_token_mask}
        if merged_labels is not None:
            result["merged_labels"] = merged_labels
        if return_hidden_states and all_hidden_states is not None:
            result["hidden_states"] = all_hidden_states
            result["image_features"] = image_features
            result["merged_embeds_len"] = inputs_embeds.size(1)
        return result

    def _extract_vlm_hidden_states(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Extract hidden states: mean of selected layers or single layer."""
        if getattr(self.config, 'vlm_use_multi_layer', False):
            indices = getattr(self.config, 'vlm_multi_layer_indices', [-1])
            return torch.stack([hidden_states[i] for i in indices], dim=0).mean(dim=0)
        return hidden_states[getattr(self.config, 'vlm_hidden_layer', -1)]

    # ======================= Vision Encoder Only (dual-frequency) =======================
    def _encode_images_only_florence2(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Florence2: run vision encoder on all views, skip language model."""
        B, V = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        N, D = valid_feats.shape[1:]

        all_features = valid_feats.new_zeros((B * V, N, D))
        all_features[flat_mask] = valid_feats
        return all_features.view(B, V, N, D)

    def _encode_images_only_qwen3(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Qwen3-VL: run visual encoder on all views, skip language model."""
        B, V, N, C = pixel_values.shape
        flat_images = pixel_values.view(B * V, N, C)
        flat_mask = image_mask.view(-1).to(torch.bool)

        valid_images = flat_images[flat_mask]
        valid_grid_thw = image_grid_thw.view(-1, 3)[flat_mask] if image_grid_thw is not None else None

        if valid_grid_thw is not None:
            patches_per_view = valid_grid_thw.prod(dim=-1).tolist()
            valid_patches = [valid_images[i, :int(n)] for i, n in enumerate(patches_per_view)]
            flat_pv = torch.cat(valid_patches, dim=0)
        else:
            flat_pv = valid_images.reshape(-1, valid_images.shape[-1])

        image_embeds_list, _ = self.vlm.get_image_features(flat_pv, valid_grid_thw)

        target_dtype = next(self.vlm.model.language_model.parameters()).dtype
        hidden_dim = image_embeds_list[0].shape[-1]
        tokens_per_image = image_embeds_list[0].shape[0]
        num_valid = int(flat_mask.sum().item())

        all_features = torch.zeros(B * V, tokens_per_image, hidden_dim,
                                   device=pixel_values.device, dtype=target_dtype)
        all_features[flat_mask] = torch.cat(image_embeds_list, dim=0).reshape(
            num_valid, tokens_per_image, hidden_dim
        ).to(dtype=target_dtype)
        return all_features.view(B, V, tokens_per_image, hidden_dim)

    def _encode_images_only(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Run vision encoder only (no LLM) on all views.

        Returns
        -------
        image_features : [B, V, tokens_per_image, D]
        """
        if self.vlm_backbone_type == "qwen3_vl":
            return self._encode_images_only_qwen3(pixel_values, image_mask, image_grid_thw)
        else:
            return self._encode_images_only_florence2(pixel_values, image_mask)

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        domain_id: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        vlm_image_input: Optional[torch.FloatTensor] = None,
        vlm_image_mask: Optional[torch.Tensor] = None,
        vlm_image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass: diffusion action denoising + optional CoT loss.

        Dual-frequency mode (use_dual_frequency=True, vlm_image_input provided):
          - VLM processes vlm_image_input (previous frame, low freq) → vlm_features + CoT loss
          - Vision encoder processes image_input (current frame, high freq) → main/aux visual inputs
          - Action head receives all three feature streams
        """
        B = input_ids.shape[0]
        device = input_ids.device
        has_cot = self.use_cot_training and labels is not None

        # === Dual-frequency path ===
        is_dual = self.use_dual_frequency and vlm_image_input is not None
        if is_dual:
            enc = self.forward_vlm(
                input_ids, vlm_image_input, vlm_image_mask,
                vlm_image_grid_thw,
                labels=labels if has_cot else None,
                return_hidden_states=has_cot,
            )

            current_features = self._encode_images_only(
                image_input, image_mask, image_grid_thw,
            )
            V = current_features.shape[1]
            D = current_features.shape[-1]
            enc["main_visual_inputs"] = current_features[:, 0]
            enc["aux_visual_inputs"] = current_features[:, 1:].reshape(B, -1, D) if V > 1 else torch.zeros(B, 0, D, device=device, dtype=current_features.dtype)
            tokens_per_view = current_features.shape[2]
            aux_view_mask = image_mask[:, 1:].to(torch.bool) if V > 1 else None
            if aux_view_mask is not None:
                enc["aux_token_mask"] = aux_view_mask.unsqueeze(-1).expand(-1, -1, tokens_per_view).reshape(B, -1)
            else:
                enc["aux_token_mask"] = None
        else:
            # === Single-frequency path (original) ===
            enc = self.forward_vlm(
                input_ids, image_input, image_mask, image_grid_thw,
                labels=labels if has_cot else None,
                return_hidden_states=has_cot,
            )

        # === Part 2: CoT loss (only when labels provided) ===
        cot_loss = None
        if has_cot:
            last_hidden = enc.pop("hidden_states")[-1]
            merged_labels = enc.pop("merged_labels", None)
            if merged_labels is None:
                raise ValueError("Missing merged_labels for CoT loss computation.")
            enc.pop("image_features", None)
            enc.pop("merged_embeds_len", None)

            logits = self.vlm.lm_head(last_hidden[:, :-1, :])
            cot_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                merged_labels[:, 1:].reshape(-1),
                ignore_index=-100, reduction='mean',
            )
        
        # === CoT-only pretrain: skip action denoising entirely ===
        if self.cot_only_pretrain:
            if cot_loss is None:
                logging.warning(
                    "cot_only_pretrain=True but no CoT loss in this batch (labels missing). "
                    "Returning zero loss — check dataset CoT coverage."
                )
                dummy = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                return {"cot_loss": dummy}
            return {"cot_loss": self.cot_loss_weight * cot_loss}

        # === Part 3: Diffusion action denoising ===
        model_dtype = next(self.transformer.parameters()).dtype
        K = getattr(self.config, 'num_diffusion_samples', 1)
        action = action.to(model_dtype)
        proprio = proprio.to(model_dtype)

        if K > 1:
            base = torch.rand(B, 1, device=device, dtype=model_dtype) / K
            strata = torch.arange(K, device=device, dtype=model_dtype) / K
            t = ((base + strata.unsqueeze(0)) % (1 - 1e-5)).reshape(B * K)

            action = action.repeat_interleave(K, dim=0)
            proprio = proprio.repeat_interleave(K, dim=0)
            if domain_id is not None:
                domain_id = domain_id.repeat_interleave(K, dim=0)

            enc = {k: v.repeat_interleave(K, dim=0) if isinstance(v, torch.Tensor) else v
                   for k, v in enc.items()}
        else:
            t = (torch.rand(1, device=device) + torch.arange(B, device=device) / B) % (1 - 1e-5)
            t = t.to(model_dtype)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(domain_id=domain_id, action_with_noise=action_noisy_m, t=t, proprio=proprio_m, **enc)
        loss_dict = self.action_space.compute_loss(pred_action, action)
        
        if cot_loss is not None:
            loss_dict["cot_loss"] = self.cot_loss_weight * cot_loss
        
        return loss_dict
    
    def _prepare_vlm_features_from_hidden_states(
        self,
        hidden_states: tuple,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        image_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract VLM features from hidden states for action transformer (CoT path)."""
        use_multi_layer = getattr(self.config, 'vlm_use_multi_layer', False)
        feats = self._extract_vlm_hidden_states(hidden_states)[:, :seq_len, :] if use_multi_layer else hidden_states[-1][:, :seq_len, :]
        if attention_mask is not None:
            feats = feats * attention_mask[:, :seq_len].unsqueeze(-1)
        hidden_dim = feats.shape[-1]
        if not self.use_dual_frequency:
            aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim) if image_features is not None and image_features.shape[1] > 1 else torch.zeros(batch_size, 0, hidden_dim, device=device, dtype=feats.dtype)
        else:
            aux_visual_inputs = torch.zeros(batch_size, 0, hidden_dim, device=device, dtype=feats.dtype)
        return {'vlm_features': feats, 'aux_visual_inputs': aux_visual_inputs}

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        domain_id: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,  # [B, V, 3] for Qwen3-VL
        steps: int = 10,
        return_cot: bool = False,  # Whether to return CoT text along with actions
        temperature: float = 0.0,  # 0 = greedy, >0 = sampling with temperature
        top_p: float = 0.9,  # nucleus sampling threshold (only used when temperature > 0)
        vlm_features_cache: Optional[Dict[str, torch.Tensor]] = None,
        vlm_image_input: Optional[torch.FloatTensor] = None,
        vlm_image_mask: Optional[torch.Tensor] = None,
        vlm_image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Generate actions.

        Dual-frequency inference:
          - Pass vlm_image_input (or vlm_features_cache) for VLM, image_input for action head.
          - When vlm_features_cache is provided, VLM is skipped (reuse cached features).
          - Returns (actions, vlm_cache_dict) when use_dual_frequency=True so caller can cache.

        Args:
            vlm_features_cache: Cached VLM output dict from a previous call (skips VLM).
            vlm_image_input: VLM frame images (dual-frequency mode).
            vlm_image_mask: VLM frame mask.
            vlm_image_grid_thw: VLM frame grid metadata.
            return_cot: If True and use_cot_training, returns (actions, cot_texts).
            temperature: CoT generation temperature. 0 = greedy argmax, >0 = sampling.
            top_p: Nucleus sampling threshold (only when temperature > 0).
        """
        self.eval()
        
        if self.use_cot_training:
            return self._generate_actions_with_cot(
                input_ids, image_input, image_mask, domain_id, proprio, image_grid_thw, steps, return_cot,
                temperature=temperature, top_p=top_p,
            )
        return self._generate_actions_diffusion(
            input_ids, image_input, image_mask, domain_id, proprio, image_grid_thw, steps,
            vlm_features_cache=vlm_features_cache,
            vlm_image_input=vlm_image_input, vlm_image_mask=vlm_image_mask,
            vlm_image_grid_thw=vlm_image_grid_thw,
        )
    
    @torch.no_grad()
    def _generate_actions_diffusion(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: Optional[torch.LongTensor],
        proprio: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        steps: int = 10,
        vlm_features_cache: Optional[Dict[str, torch.Tensor]] = None,
        vlm_image_input: Optional[torch.FloatTensor] = None,
        vlm_image_mask: Optional[torch.Tensor] = None,
        vlm_image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Generate actions via iterative denoising (linear schedule).

        In dual-frequency mode, VLM features can be cached and reused across
        multiple high-frequency action generation calls.
        """
        B = input_ids.shape[0]
        is_dual = self.use_dual_frequency

        if is_dual:
            # Reuse cached VLM features or compute them from vlm_image_input
            if vlm_features_cache is not None:
                enc = {k: v for k, v in vlm_features_cache.items()}
            else:
                vlm_img = vlm_image_input if vlm_image_input is not None else image_input
                vlm_msk = vlm_image_mask if vlm_image_mask is not None else image_mask
                vlm_gthw = vlm_image_grid_thw if vlm_image_grid_thw is not None else image_grid_thw
                enc = self.forward_vlm(input_ids, vlm_img, vlm_msk, vlm_gthw)

            # Encode current-frame views through vision encoder only
            current_features = self._encode_images_only(image_input, image_mask, image_grid_thw)
            V = current_features.shape[1]
            D_feat = current_features.shape[-1]
            enc["main_visual_inputs"] = current_features[:, 0]
            enc["aux_visual_inputs"] = (
                current_features[:, 1:].reshape(B, -1, D_feat) if V > 1
                else torch.zeros(B, 0, D_feat, device=current_features.device, dtype=current_features.dtype)
            )
            if V > 1:
                tokens_per_view = current_features.shape[2]
                aux_view_mask = image_mask[:, 1:].to(torch.bool)
                enc["aux_token_mask"] = aux_view_mask.unsqueeze(-1).expand(-1, -1, tokens_per_view).reshape(B, -1)
            else:
                enc["aux_token_mask"] = None

            vlm_cache = {"vlm_features": enc["vlm_features"]}
            if "aux_token_mask" in enc and enc.get("aux_token_mask") is not None:
                pass  # aux_token_mask is per-frame, not cacheable
        else:
            enc = self.forward_vlm(input_ids, image_input, image_mask, image_grid_thw)
            vlm_cache = None

        D = self.action_space.dim_action
        model_dtype = next(self.transformer.parameters()).dtype
        proprio = proprio.to(model_dtype)
        x1 = torch.randn(B, self.num_actions, D, device=proprio.device, dtype=model_dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=model_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        action = self.action_space.postprocess(action)

        if is_dual:
            return action, vlm_cache
        return action
    
    @torch.no_grad()
    def _generate_actions_with_cot(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: Optional[torch.LongTensor],
        proprio: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        steps: int = 10,
        return_cot: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """
        Generate actions with CoT (Chain-of-Thought) text generation.
        
        1) Build multimodal embeddings (get_image_features + masked_scatter + M-RoPE)
        2) Generate CoT text autoregressively
        3) Use CoT-conditioned hidden states for action denoising
        """
        B, V, N, C = image_input.shape
        device = input_ids.device
        flat_images = image_input.view(B * V, N, C)
        flat_mask = image_mask.view(-1).to(torch.bool)

        if flat_mask.sum().item() == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_grid_thw = image_grid_thw.view(-1, 3)[flat_mask] if image_grid_thw is not None else None
        
        # Extract only valid patches (remove padding) based on grid_thw.
        # patches_per_view = T*H*W — raw patch count BEFORE spatial merge.
        # spatial_merge_size is applied after the visual encoder, not here.
        if valid_grid_thw is not None:
            patches_per_view = valid_grid_thw.prod(dim=-1).tolist()
            valid_patches = []
            for i, num_patches in enumerate(patches_per_view):
                valid_patches.append(valid_images[i, :int(num_patches)])
            flat_pv = torch.cat(valid_patches, dim=0)
        else:
            flat_pv = valid_images.reshape(-1, valid_images.shape[-1])
        
        image_embeds_list, deepstack_all = self.vlm.get_image_features(flat_pv, valid_grid_thw)

        inputs_embeds = self.vlm.model.language_model.embed_tokens(input_ids)
        target_dtype = next(self.vlm.model.language_model.parameters()).dtype
        pad_id = self.qwen3_tokenizer.pad_token_id
        attention_mask = (input_ids != pad_id).long()

        view0_mask = torch.zeros(B * V, dtype=torch.bool, device=flat_mask.device)
        view0_mask[0::V] = True
        view0_among_valid = view0_mask[flat_mask]
        prefix_feats_flat = torch.cat(
            [emb for emb, is_v0 in zip(image_embeds_list, view0_among_valid) if is_v0], dim=0
        )
        prefix_grid_thw = image_grid_thw[:, 0, :] if image_grid_thw is not None else None

        tokens_per_image = image_embeds_list[0].shape[0]
        prefix_deepstack = None
        if deepstack_all:
            prefix_deepstack = [
                ds.reshape(len(image_embeds_list), tokens_per_image, -1)[view0_among_valid].reshape(-1, ds.shape[-1])
                for ds in deepstack_all
            ]

        hidden_dim = inputs_embeds.shape[-1]
        num_valid = int(flat_mask.sum().item())
        all_features = torch.zeros(B * V, tokens_per_image, hidden_dim, device=inputs_embeds.device, dtype=target_dtype)
        all_features[flat_mask] = torch.cat(image_embeds_list, dim=0).reshape(num_valid, tokens_per_image, hidden_dim).to(dtype=target_dtype)
        image_features = all_features.view(B, V, tokens_per_image, hidden_dim)

        image_pad_mask = (input_ids == self.vlm.config.image_token_id)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_pad_mask.unsqueeze(-1).expand_as(inputs_embeds),
            prefix_feats_flat.to(dtype=target_dtype, device=inputs_embeds.device),
        )
        merged_embeds = inputs_embeds.to(dtype=target_dtype)
        merged_attn_mask = attention_mask

        if prefix_grid_thw is not None:
            prefix_grid_thw = prefix_grid_thw.to(device=device, dtype=torch.long)
        position_ids, _ = self.vlm.model.get_rope_index(input_ids, prefix_grid_thw, None, attention_mask)
        
        # 2. Generate CoT text autoregressively
        cot_max_length = getattr(self.config, 'cot_max_length', 768)
        cot_start_id = self.qwen3_cot_start_id
        cot_end_id = self.qwen3_cot_end_id
        generated_ids = torch.full((B, 1), cot_start_id, device=device, dtype=torch.long)
        cot_embeds = self.vlm.model.language_model.embed_tokens(generated_ids)
        
        current_embeds = torch.cat([merged_embeds, cot_embeds], dim=1)
        current_attn_mask = torch.cat([
            merged_attn_mask,
            torch.ones(B, 1, device=device, dtype=torch.long),
        ], dim=1)
        # position_ids shape: [3, B, L] (M-RoPE: temporal, height, width)
        current_position_ids = torch.cat([
            position_ids,
            position_ids[:, :, -1:] + 1,
        ], dim=2)

        last_outputs = None
        for _ in range(cot_max_length - 1):
            vis_mask = torch.nn.functional.pad(image_pad_mask, (0, current_embeds.size(1) - image_pad_mask.size(1)), value=False)
            lm_kw = dict(
                inputs_embeds=current_embeds,
                attention_mask=current_attn_mask,
                position_ids=current_position_ids,
                visual_pos_masks=vis_mask,
                deepstack_visual_embeds=prefix_deepstack,
                output_hidden_states=True,
                return_dict=True,
            )
            last_outputs = self.vlm.model.language_model(**lm_kw)

            next_logits = self.vlm.lm_head(last_outputs.last_hidden_state[:, -1:, :])
            if temperature > 0:
                logits_scaled = next_logits.squeeze(1) / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits_scaled, descending=True)
                    cumprob = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprob - torch.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[mask] = float('-inf')
                    logits_scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if (next_token.squeeze(-1) == cot_end_id).all():
                break

            next_embed = self.vlm.model.language_model.embed_tokens(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            current_attn_mask = torch.cat([
                current_attn_mask,
                torch.ones(B, 1, device=device, dtype=torch.long),
            ], dim=1)
            current_position_ids = torch.cat([
                current_position_ids,
                current_position_ids[:, :, -1:] + 1,
            ], dim=2)
        
        # 3. Reuse hidden states from last loop iteration
        processed_len = last_outputs.last_hidden_state.size(1) if last_outputs is not None else current_embeds.size(1)
        enc = self._prepare_vlm_features_from_hidden_states(
            last_outputs.hidden_states, processed_len, B, device,
            image_features, attention_mask=current_attn_mask,
        )
        
        # 4. Diffusion denoising for actions
        D = self.action_space.dim_action
        model_dtype = next(self.transformer.parameters()).dtype
        proprio = proprio.to(model_dtype)
        x1 = torch.randn(B, self.num_actions, D, device=device, dtype=model_dtype)
        action = torch.zeros_like(x1)
        
        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=device, dtype=model_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        
        action = self.action_space.postprocess(action)
        
        # 5. Decode CoT text if requested
        if return_cot and generated_ids is not None:
            cot_texts = []
            for i in range(B):
                # Decode full sequence - preserve special tokens for visibility
                cot_text = self.qwen3_tokenizer.decode(generated_ids[i], skip_special_tokens=False)
                cot_texts.append(cot_text)
            return action, cot_texts
        
        return action

    # =============================== FastAPI service =============================
    def _build_app(self, processor):
        """
        Minimal FastAPI app for XVLA inference.

        Args:
            processor: callable(images, text) -> Dict[str, torch.Tensor]
                       expected keys: "input_ids", "image_input", "image_mask"
        """
        if self.app is not None:
            return

        import json_numpy
        import cv2
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.get("/config")
        def get_config():
            """Return action-relevant config for client auto-configuration."""
            cfg = {
                "action_mode": getattr(self.config, "action_mode", "ee6d"),
                "num_actions": getattr(self.config, "num_actions", 30),
                "joint_norm_stats": getattr(self.config, "joint_norm_stats", None),
            }
            return JSONResponse(cfg)

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                # Decode up to 3 image inputs
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload: continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:  # encoded bytes
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))
                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                # Warn if more images than processor can handle
                if len(images) > processor.num_views:
                    logging.warning(
                        f"Received {len(images)} images but processor.num_views={processor.num_views}. "
                        f"Extra images will be ignored. Use --num_views {len(images)} when starting server."
                    )

                # Multimodal preprocessing (images first so image_grid_thw is available for Qwen3-VL)
                inputs = processor(images=images, language_instruction=payload["language_instruction"])
                required = {"input_ids", "image_input", "image_mask"}
                if self.vlm_backbone_type == "qwen3_vl":
                    required = required | {"image_grid_thw"}
                if not required.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                # Build proprio/domain tensors
                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))

                # Align to model's device/dtype
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs.update({
                    "proprio": to_model(proprio.unsqueeze(0)),
                })
                if getattr(self, 'use_soft_prompt', True) and "domain_id" in payload:
                    domain_id = torch.tensor([int(payload["domain_id"])], dtype=torch.long)
                    inputs["domain_id"] = domain_id.to(device)

                # Inference with autocast for consistent half-precision
                steps = int(payload.get("steps", 10))
                use_cot = getattr(self, 'use_cot_training', False)
                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=dtype)
                    if device.type == "cuda" and dtype != torch.float32
                    else torch.inference_mode()
                )

                with autocast_ctx:
                    if use_cot:
                        result = self.generate_actions(**inputs, steps=steps, return_cot=True)
                        if isinstance(result, tuple):
                            action, cot_texts = result
                            action = action.squeeze(0).float().cpu().numpy()
                            cot_text = cot_texts[0] if cot_texts else ""
                            print(f"\n{'='*80}")
                            print(f"MODEL CoT OUTPUT")
                            print(f"{'='*80}")
                            print(f"Instruction: {payload['language_instruction']}")
                            print(f"CoT: {cot_text}")
                            print(f"{'='*80}\n")
                            return JSONResponse({"action": action.tolist(), "cot": cot_text})
                        else:
                            action = result.squeeze(0).float().cpu().numpy()
                            return JSONResponse({"action": action.tolist()})
                    else:
                        action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                        return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """Launch the FastAPI service."""
        self._build_app(processor)
        assert self.app is not None
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# ------------------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------------------
def prepare_batch(
    batch: Dict[str, Any],
    processor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare training batch (works for all backbone types)."""
    batch = dict(batch)
    instructions = batch.pop("language_instruction", [])
    cot_texts = batch.pop("cot_text", None)
    
    # CoT is Qwen3-VL only; skip for Florence2
    if cot_texts is not None and getattr(processor, 'vlm_backbone_type', 'florence2') == 'qwen3_vl':
        if isinstance(cot_texts, str):
            cot_texts = [cot_texts]
        if not any(t.strip() for t in cot_texts):
            cot_texts = None
    else:
        cot_texts = None
    
    # In dual-frequency mode, encode_language uses VLM frame metadata
    # (VLM only sees 1 image — the VLM frame's main view)
    if "vlm_image_input" in batch:
        lang = processor.encode_language(
            instructions,
            cot_texts=cot_texts,
            image_mask=batch.get("vlm_image_mask"),
            image_grid_thw=batch.get("vlm_image_grid_thw"),
        )
    else:
        lang = processor.encode_language(
            instructions,
            cot_texts=cot_texts,
            image_mask=batch.get("image_mask"),
            image_grid_thw=batch.get("image_grid_thw"),
        )
    batch.update(lang)
    
    # Remove non-tensor metadata before moving to device
    batch.pop("vlm_frame_idx", None)
    
    tensor_inputs = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_inputs[key] = value.to(device=device, non_blocking=True)
        elif value is not None:
            tensor_inputs[key] = torch.as_tensor(value).to(device=device)
    return tensor_inputs


def build_vla_optimizer(
    model: XVLA,
    *,
    base_lr: float,
    weight_decay: float,
    betas: tuple,
    lr_coef_soft: float,
) -> AdamW:
    """Build optimizer with grouped learning rates."""
    cot_only = getattr(model, 'cot_only_pretrain', False)

    vlm_params = [p for p in model.vlm.parameters() if p.requires_grad]

    if cot_only:
        # CoT-only pretrain: only VLM needs training, skip frozen transformer entirely
        param_groups = [
            {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        ]
        return AdamW(param_groups, betas=betas)

    has_soft_prompt = getattr(model, 'use_soft_prompt', True) and hasattr(model.transformer, 'soft_prompt_hub')
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters()) if has_soft_prompt else []
    
    action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": base_lr, "weight_decay": weight_decay},
    ]
    if has_soft_prompt:
        param_groups.append(
            {"name": "soft_prompts", "params": soft_prompt_params, "lr": base_lr * lr_coef_soft, "weight_decay": weight_decay},
        )
    
    return AdamW(param_groups, betas=betas)


def update_vla_learning_rates(
    optim: torch.optim.Optimizer,
    step: int,
    *,
    learning_rate: float,
    learning_coef: float,
    freeze_steps: int,
    warmup_steps: int,
    total_iters: int,
    use_cosine_decay: bool,
    min_lr_ratio: float,
) -> None:
    """Update learning rates with warmup and optional cosine decay."""
    def set_group_lr(name: str, lr: float):
        for g in optim.param_groups:
            if g.get("name") == name:
                g["lr"] = lr

    def schedule(base_lr: float) -> float:
        return _linear_warmup_cosine(step, freeze_steps, warmup_steps, total_iters, base_lr, min_lr_ratio)

    # Base learning rates for each parameter group
    group_names = {g["name"] for g in optim.param_groups}

    # CoT-only pretrain: optimizer only has "vlm" group
    if group_names == {"vlm"}:
        vlm_lr = learning_rate * learning_coef
        if step < freeze_steps:
            set_group_lr("vlm", 0.0)
        else:
            new_lr = schedule(vlm_lr) if use_cosine_decay else vlm_lr
            set_group_lr("vlm", new_lr)
        return

    base = {
        "vlm": learning_rate * learning_coef,
        "transformer_core": learning_rate,
        "action_heads": learning_rate,
    }
    if "soft_prompts" in group_names:
        base["soft_prompts"] = learning_rate * learning_coef

    if step < freeze_steps:
        set_group_lr("vlm", 0.0)
        set_group_lr("transformer_core", 0.0)
        if "soft_prompts" in base:
            set_group_lr("soft_prompts", base["soft_prompts"])
        set_group_lr("action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(base_lr) if use_cosine_decay else base_lr
            set_group_lr(name, new_lr)


def _linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Compute LR with linear warmup and cosine decay."""
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)
