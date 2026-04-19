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
FACT Decoder: Dual-stream flow matching decoder with MMDiTBlock.

Uses multimodal diffusion transformer blocks with AdaLN modulation
to decode quantized latents into action trajectories via flow matching.
"""

import torch
import torch.nn as nn
from .mmdit_block import MMDiTBlock, TimestepEmbedder, modulate


class FACTDecoder(nn.Module):
    """
    Dual-stream flow matching decoder with MMDiTBlock.
    
    Architecture:
        1. Noisy actions projected as x-stream
        2. Quantized latents projected as c-stream
        3. Joint attention with AdaLN time modulation via MMDiTBlock
        4. Final AdaLN modulation and output projection
    """
    
    def __init__(
        self,
        action_dim: int = 20,
        num_actions: int = 30,
        latent_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,  # unused, kept for API compatibility
        dropout: float = 0.0,    # unused, kept for API compatibility
    ):
        """
        Args:
            action_dim: Dimension of action space
            num_actions: Length of action trajectory
            latent_dim: Dimension of latent/hidden space
            num_layers: Number of MMDiTBlock layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio (unused, for API compatibility)
            dropout: Dropout rate (unused, for API compatibility)
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        
        # Action projection: action_dim -> latent_dim
        self.action_proj = nn.Linear(action_dim, latent_dim)
        
        # Latent projection: latent_dim -> latent_dim
        self.latent_proj = nn.Linear(latent_dim, latent_dim)
        
        # Timestep embedder
        self.t_embedder = TimestepEmbedder(latent_dim)
        
        # MMDiTBlock layers (dual-stream joint attention with AdaLN)
        self.blocks = nn.ModuleList([
            MMDiTBlock(latent_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Final normalization and AdaLN
        self.final_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim)
        )
        # Zero-init AdaLN: makes final modulation act as identity at initialization
        nn.init.constant_(self.adaLN_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_final[-1].bias, 0)
        
        # Output projection: latent_dim -> action_dim
        self.output_head = nn.Linear(latent_dim, action_dim)
        
        # Zero-init output layer for stable training (predicts zero velocity initially)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity for flow matching.
        
        Args:
            x_t: Noisy action at time t [B, T, D_action]
            t: Timesteps [B], values in [0, 1]
            z_q: Quantized latent conditioning [B, K, D_latent]
        
        Returns:
            v_t: Predicted velocity [B, T, D_action]
        """
        # Project noisy actions to hidden space (x-stream)
        x = self.action_proj(x_t)  # [B, T, latent_dim]
        
        # Project latent codes to hidden space (c-stream)
        c = self.latent_proj(z_q)  # [B, K, latent_dim]
        
        # Get timestep embedding
        t_emb = self.t_embedder(t)  # [B, latent_dim]
        
        # Process through MMDiTBlock layers
        for block in self.blocks:
            x, c = block(x, c, t_emb)
        
        # Final AdaLN modulation
        scale_shift = self.adaLN_final(t_emb)
        shift, scale = scale_shift.chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        
        # Output velocity
        return self.output_head(x)


if __name__ == "__main__":
    B, T, D, K, H = 4, 30, 20, 8, 256
    
    decoder = FACTDecoder(
        action_dim=D,
        num_actions=T,
        latent_dim=H,
        num_layers=6,
        num_heads=8,
    )
    
    x_t = torch.randn(B, T, D)
    z_q = torch.randn(B, K, H)
    t = torch.rand(B)
    
    v_t = decoder(x_t, t, z_q)
    
    print(f"Input x_t shape: {x_t.shape}")
    print(f"Latent z_q shape: {z_q.shape}")
    print(f"Timesteps shape: {t.shape}")
    print(f"Output v_t shape: {v_t.shape}")
    assert v_t.shape == x_t.shape
    print("✓ FACTDecoder test passed")
