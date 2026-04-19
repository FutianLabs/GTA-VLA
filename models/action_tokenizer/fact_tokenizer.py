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
FACT Tokenizer: Complete action tokenizer with LFQ and flow matching.

Combines encoder, LFQ quantization, and flow matching decoder into a unified
action tokenizer for discrete action representation learning.
"""

import torch
import torch.nn as nn
import math
import json
from typing import Dict, Tuple, Optional, List

from .fact_encoder import FACTEncoder
from .codebook import LookupFreeQuantizer
from .fact_decoder import FACTDecoder
from .flow_matching import linear_flow, flow_matching_loss, flow_sample

# Default position dimensions for ee6d action space (dual-arm xyz)
DEFAULT_NORMALIZED_DIMS = [0, 1, 2, 10, 11, 12]


class FACTTokenizer(nn.Module):
    """
    FACT Action Tokenizer with LFQ and flow matching.
    
    Architecture:
        1. Encoder: actions [B, T, D] -> latents [B, K, D_latent]
        2. LFQ: latents -> quantized [B, K, D_latent] + indices [B, K]
        3. Decoder: quantized + noise + timestep -> velocity -> reconstructed actions
    """
    
    def __init__(
        self,
        action_dim: int = 20,
        num_actions: int = 30,
        latent_dim: int = 256,
        num_latent_tokens: int = 8,
        codebook_size: int = 1024,
        encoder_layers: int = 4,
        decoder_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        commitment_cost: float = 0.25,
        dropout: float = 0.0,
        entropy_weight: float = 0.1,
    ):
        """
        Args:
            action_dim: Dimension of action space (e.g., 20 for ee6d)
            num_actions: Length of action chunk
            latent_dim: Dimension of latent space
            num_latent_tokens: Number of latent tokens
            codebook_size: Codebook size (must be power of 2, e.g., 1024 = 2^10)
            encoder_layers: Number of encoder MMDiT layers
            decoder_layers: Number of decoder MMDiT layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            commitment_cost: Commitment loss weight for LFQ
            dropout: Dropout rate
            entropy_weight: Weight for entropy regularization loss
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        
        # Encoder
        self.encoder = FACTEncoder(
            action_dim=action_dim,
            num_actions=num_actions,
            latent_dim=latent_dim,
            num_latent_tokens=num_latent_tokens,
            num_layers=encoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        
        # LFQ Quantizer
        # codebook_size = 2^num_codebook_dims
        num_codebook_dims = int(math.log2(codebook_size))
        if 2 ** num_codebook_dims != codebook_size:
            raise ValueError(f"codebook_size must be power of 2, got {codebook_size}")
        
        self.quantizer = LookupFreeQuantizer(
            num_codebook_dims=num_codebook_dims,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            entropy_weight=entropy_weight,
        )
        
        # Decoder
        self.decoder = FACTDecoder(
            action_dim=action_dim,
            num_actions=num_actions,
            latent_dim=latent_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        
        # Action normalization buffers (saved with model, moved with .to())
        self.register_buffer("action_mean", None)
        self.register_buffer("action_std", None)
    
    # -------------------------------------------------------------------------
    # Action Normalization Methods
    # -------------------------------------------------------------------------
    
    def set_action_stats(self, stats: Dict) -> None:
        """
        Set action normalization statistics (mean/std).
        
        Args:
            stats: Dictionary with 'mean' and 'std' lists
        """
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        # Register as buffers (saved with model, moved with .to())
        self.register_buffer("action_mean", mean)
        self.register_buffer("action_std", std)
    
    @property
    def has_normalization(self) -> bool:
        """Check if normalization is enabled."""
        return self.action_mean is not None
    
    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions using mean/std.
        
        Args:
            actions: Raw actions [B, T, D]
        
        Returns:
            Normalized actions [B, T, D]
        """
        if not self.has_normalization:
            return actions
        mean = self.action_mean.to(device=actions.device, dtype=actions.dtype)
        std = self.action_std.to(device=actions.device, dtype=actions.dtype)
        return (actions - mean) / std
    
    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Denormalize actions back to original scale.
        
        Args:
            actions: Normalized actions [B, T, D]
        
        Returns:
            Denormalized actions [B, T, D]
        """
        if not self.has_normalization:
            return actions
        mean = self.action_mean.to(device=actions.device, dtype=actions.dtype)
        std = self.action_std.to(device=actions.device, dtype=actions.dtype)
        return actions * std + mean
    
    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------
    
    def encode(
        self, 
        actions: torch.Tensor,
        skip_normalize: bool = False,
        return_stats: bool = False,
    ):
        """
        Encode actions to quantized latents.
        
        Args:
            actions: Action chunk [B, T, D] (raw or pre-normalized)
            skip_normalize: If True, skip normalization (already normalized)
            return_stats: If True, also return quantization statistics
        
        Returns:
            z_q: Quantized latents [B, K, D_latent]
            indices: Codebook indices [B, K]
            lfq_loss: LFQ loss (commitment + entropy)
            stats: (optional) Dict with quantization statistics
        """
        # Normalize actions if stats are set and not already normalized
        if not skip_normalize:
            actions = self._normalize_actions(actions)
        
        # Encode to latents
        z = self.encoder(actions)  # [B, K, D_latent]
        
        # Quantize with LFQ
        if return_stats:
            z_q, indices, lfq_loss, stats = self.quantizer(z, return_stats=True)
            return z_q, indices, lfq_loss, stats
        else:
            z_q, indices, lfq_loss = self.quantizer(z)
            return z_q, indices, lfq_loss
    
    def decode(
        self,
        z_q: torch.Tensor,
        num_steps: int = 10,
        method: str = "euler",
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        Decode quantized latents to actions via flow matching.
        
        Args:
            z_q: Quantized latents [B, K, D_latent]
            num_steps: Number of flow matching steps
            method: Integration method ('euler' or 'heun')
            denormalize: Whether to denormalize output to original range
        
        Returns:
            actions: Reconstructed actions [B, T, D] (denormalized if enabled)
        """
        B = z_q.shape[0]
        device = z_q.device
        dtype = z_q.dtype
        
        # Start from noise (match dtype)
        x0 = torch.randn(B, self.num_actions, self.action_dim, device=device, dtype=dtype)
        
        # Sample via flow matching
        def model_fn(x_t, t, cond):
            return self.decoder(x_t, t, cond)
        
        actions = flow_sample(model_fn, x0, z_q, num_steps, method)
        
        # Denormalize actions if enabled
        if denormalize:
            actions = self._denormalize_actions(actions)
        
        return actions
    
    def decode_indices(
        self,
        indices: torch.Tensor,
        num_steps: int = 10,
        method: str = "euler",
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        Decode from codebook indices to actions.
        
        Args:
            indices: Codebook indices [B, K]
            num_steps: Number of flow matching steps
            method: Integration method
            denormalize: Whether to denormalize output to original range
        
        Returns:
            actions: Reconstructed actions [B, T, D] (denormalized if enabled)
        """
        # Get quantized embeddings from indices
        z_q = self.quantizer.decode_indices(indices)
        
        # Decode to actions
        return self.decode(z_q, num_steps, method, denormalize=denormalize)
    
    def forward(
        self,
        actions: torch.Tensor,
        timestep_strategy: str = "uniform",
        inference_steps: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            actions: Ground truth actions [B, T, D] (raw, unnormalized)
            timestep_strategy: Timestep sampling strategy
            inference_steps: Number of Euler steps for reconstruction (10-20 for high accuracy)
        
        Returns:
            Dictionary with losses and reconstructed actions (denormalized)
        """
        B = actions.shape[0]
        device = actions.device
        dtype = actions.dtype
        
        # Normalize actions for training
        actions_norm = self._normalize_actions(actions)
        
        # Encode and quantize (skip normalization since we already normalized)
        z_q, indices, lfq_loss, quant_stats = self.encode(actions_norm, skip_normalize=True, return_stats=True)
        
        # Sample timesteps for flow matching (match dtype)
        if timestep_strategy == "uniform":
            t = torch.rand(B, device=device, dtype=dtype)
        elif timestep_strategy == "logit_normal":
            t_raw = torch.randn(B, device=device, dtype=dtype)
            t = torch.sigmoid(t_raw)
        else:
            t = torch.rand(B, device=device, dtype=dtype)
        
        # Create noisy interpolation in normalized space
        x0 = torch.randn_like(actions_norm, dtype=dtype)  # Noise
        x1 = actions_norm  # Normalized ground truth
        x_t, v_target = linear_flow(x0, x1, t)
        
        # Predict velocity
        v_pred = self.decoder(x_t, t, z_q)
        
        # Flow matching loss (computed in normalized space)
        fm_loss, fm_loss_dict = flow_matching_loss(v_pred, v_target)
        
        # Compute reconstruction (for monitoring) - denormalize for metrics
        with torch.no_grad():
            reconstructed = self.decode(z_q, num_steps=inference_steps, denormalize=True)
        
        # Total loss
        total_loss = fm_loss + lfq_loss
        
        # Codebook usage
        codebook_usage = self.quantizer.get_codebook_usage(indices)
        
        return {
            "total_loss": total_loss,
            "fm_loss": fm_loss,
            "lfq_loss": lfq_loss,
            "reconstructed": reconstructed,
            "indices": indices,
            "codebook_usage": codebook_usage,
            "quant_stats": quant_stats,
            **fm_loss_dict,
        }


if __name__ == "__main__":
    # Test FACTTokenizer
    batch_size = 4
    num_actions = 30
    action_dim = 20
    
    # Create tokenizer (LFQ only)
    tokenizer = FACTTokenizer(
        action_dim=action_dim,
        num_actions=num_actions,
        latent_dim=256,
        num_latent_tokens=8,
        codebook_size=1024,
        encoder_layers=4,
        decoder_layers=6,
    )
    
    # Test forward pass (training)
    actions = torch.randn(batch_size, num_actions, action_dim)
    outputs = tokenizer(actions)
    
    print("Training outputs:")
    print(f"  Total loss: {outputs['total_loss'].item():.4f}")
    print(f"  FM loss: {outputs['fm_loss'].item():.4f}")
    print(f"  LFQ loss: {outputs['lfq_loss'].item():.4f}")
    print(f"  Codebook usage: {outputs['codebook_usage']:.2f}%")
    print(f"  Reconstructed shape: {outputs['reconstructed'].shape}")
    print(f"  Indices shape: {outputs['indices'].shape}")
    
    # Test encode/decode (inference)
    z_q, indices, _ = tokenizer.encode(actions)
    reconstructed = tokenizer.decode(z_q, num_steps=10)
    
    print(f"\nInference:")
    print(f"  Quantized latents shape: {z_q.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    
    # Test decode from indices
    reconstructed_from_indices = tokenizer.decode_indices(indices, num_steps=10)
    print(f"  Reconstructed from indices shape: {reconstructed_from_indices.shape}")
    
    print("✓ FACTTokenizer (LFQ-only) test passed")
