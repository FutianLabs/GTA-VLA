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
MMDiT (Multimodal Diffusion Transformer) Blocks with Dual-Stream Joint Attention.

Implements:
- TimestepEmbedder: MLP-based timestep embedding
- MMBlock: Encoder block with dual-stream joint attention (no time modulation)
- MMDiTBlock: Decoder block with dual-stream joint attention and AdaLN modulation
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        t: Timestep tensor [B]
        dim: Embedding dimension
        max_period: Maximum period for sinusoids
    
    Returns:
        Timestep embeddings [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """MLP-based timestep embedder."""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        # Convert to model dtype for mixed precision compatibility
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MMBlock(nn.Module):
    """
    Encoder Block: Dual Stream, Joint Attention, Standard LayerNorm (No Time Modulation).
    
    Both x (queries) and c (context) are processed together in joint attention,
    then split back for independent MLP processing.
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # LayerNorms for x and c streams
        self.norm1_x = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm1_c = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # QKV projections for both streams
        self.qkv_x = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.qkv_c = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        
        # Output projections
        self.proj_out_x = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_out_c = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # MLP LayerNorms
        self.norm2_x = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2_c = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Independent MLPs
        self.mlp_x = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Query stream [B, Lx, D]
            c: Context stream [B, Lc, D]
        
        Returns:
            x: Updated query stream [B, Lx, D]
            c: Updated context stream [B, Lc, D]
        """
        B, Lx, _ = x.shape
        _, Lc, _ = c.shape
        
        # Joint Attention
        x_norm = self.norm1_x(x)
        c_norm = self.norm1_c(c)
        
        qkv_x = self.qkv_x(x_norm)
        qkv_c = self.qkv_c(c_norm)
        
        qx, kx, vx = qkv_x.chunk(3, dim=-1)
        qc, kc, vc = qkv_c.chunk(3, dim=-1)
        
        # Concatenate for joint attention
        q = torch.cat([qx, qc], dim=1)
        k = torch.cat([kx, kc], dim=1)
        v = torch.cat([vx, vc], dim=1)
        
        # Reshape for multi-head attention
        head_dim = self.hidden_size // self.num_heads
        q = q.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.hidden_size)
        
        # Split back
        x_attn, c_attn = attn_out[:, :Lx], attn_out[:, Lx:]
        
        # Residual connections
        x = x + self.proj_out_x(x_attn)
        c = c + self.proj_out_c(c_attn)
        
        # Independent MLPs
        x = x + self.mlp_x(self.norm2_x(x))
        c = c + self.mlp_c(self.norm2_c(c))
        
        return x, c


class MMDiTBlock(nn.Module):
    """
    Decoder Block: Dual Stream, Joint Attention, AdaLN Modulation (Time Dependent).
    
    AdaLN modulation is applied only to the x-stream (noisy actions).
    The c-stream (latent codes) uses standard LayerNorm.
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # LayerNorms (x uses elementwise_affine=False for AdaLN)
        self.norm1_x = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_c = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # QKV projections
        self.qkv_x = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.qkv_c = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        
        # Output projections
        self.proj_out_x = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_out_c = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # MLP LayerNorms
        self.norm2_x = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_c = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Independent MLPs
        self.mlp_x = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
        # AdaLN modulation: 6 outputs (shift_msa, scale_msa, shift_mlp, scale_mlp, gate_msa, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Zero-init AdaLN: makes block act as identity at initialization
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor, 
        t_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Noisy action stream [B, Lx, D]
            c: Latent code stream [B, Lc, D]
            t_emb: Timestep embedding [B, D]
        
        Returns:
            x: Updated action stream [B, Lx, D]
            c: Updated latent stream [B, Lc, D]
        """
        B, Lx, _ = x.shape
        _, Lc, _ = c.shape
        
        # AdaLN Modulation parameters
        scale_shift = self.adaLN_modulation(t_emb)
        shift_msa, scale_msa, shift_mlp, scale_mlp, gate_msa, gate_mlp = scale_shift.chunk(6, dim=1)
        
        # Joint Attention with modulation on x
        x_norm = modulate(self.norm1_x(x), shift_msa, scale_msa)
        c_norm = self.norm1_c(c)
        
        qkv_x = self.qkv_x(x_norm)
        qkv_c = self.qkv_c(c_norm)
        
        qx, kx, vx = qkv_x.chunk(3, dim=-1)
        qc, kc, vc = qkv_c.chunk(3, dim=-1)
        
        # Concatenate for joint attention
        q = torch.cat([qx, qc], dim=1)
        k = torch.cat([kx, kc], dim=1)
        v = torch.cat([vx, vc], dim=1)
        
        # Reshape for multi-head attention
        head_dim = self.hidden_size // self.num_heads
        q = q.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.hidden_size)
        
        # Split back
        x_attn, c_attn = attn_out[:, :Lx], attn_out[:, Lx:]
        
        # Residual connections
        x = x + self.proj_out_x(x_attn)
        c = c + self.proj_out_c(c_attn)
        
        # Independent MLPs with modulation on x
        x_norm2 = modulate(self.norm2_x(x), shift_mlp, scale_mlp)
        x = x + self.mlp_x(x_norm2)
        c = c + self.mlp_c(self.norm2_c(c))
        
        return x, c


if __name__ == "__main__":
    B, Lx, Lc, H = 4, 30, 8, 256
    num_heads = 8
    
    x = torch.randn(B, Lx, H)
    c = torch.randn(B, Lc, H)
    t = torch.rand(B)
    
    # Test MMBlock (encoder)
    mm_block = MMBlock(H, num_heads)
    x_out, c_out = mm_block(x, c)
    assert x_out.shape == x.shape
    assert c_out.shape == c.shape
    print("✓ MMBlock test passed")
    
    # Test TimestepEmbedder
    t_embedder = TimestepEmbedder(H)
    t_emb = t_embedder(t)
    assert t_emb.shape == (B, H)
    print("✓ TimestepEmbedder test passed")
    
    # Test MMDiTBlock (decoder)
    mmdit_block = MMDiTBlock(H, num_heads)
    x_out, c_out = mmdit_block(x, c, t_emb)
    assert x_out.shape == x.shape
    assert c_out.shape == c.shape
    print("✓ MMDiTBlock test passed")
