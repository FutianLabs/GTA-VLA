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
Vector Quantization with Exponential Moving Average (EMA) and Lookup-Free Quantization (LFQ).

Implements:
1. VQ-VAE codebook with EMA updates for stable training
   Based on "Neural Discrete Representation Learning" (van den Oord et al., 2017)
2. Lookup-Free Quantization (LFQ) for improved codebook utilization
   Based on "Magvit-2" and modern VQ approaches
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with EMA codebook updates.
    
    Discretizes continuous latent representations by finding nearest
    codebook entries. Uses EMA to update codebook for stable training.
    """
    
    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_restart: bool = True,  # Enable codebook restart for dead codes
        entropy_weight: float = 0.1,  # Weight for entropy loss
    ):
        """
        Args:
            num_embeddings: Size of the codebook (number of discrete codes)
            embedding_dim: Dimension of each code vector
            commitment_cost: Weight for commitment loss
            decay: EMA decay rate for codebook updates
            epsilon: Small constant for numerical stability
            use_restart: Whether to restart dead codes with random samples
            entropy_weight: Weight for entropy regularization loss
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_restart = use_restart
        self.entropy_weight = entropy_weight
        
        # Codebook embeddings (trainable)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA cluster size and embedding average
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding", self.embedding.weight.data.clone())
        
        # Track usage for dead code restart
        self.register_buffer("code_usage_count", torch.zeros(num_embeddings))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input latents.
        
        Args:
            z: Input latent tensor [B, K, D] where K is num latent tokens
        
        Returns:
            z_q: Quantized tensor [B, K, D] (with straight-through gradient)
            indices: Codebook indices [B, K]
            vq_loss: VQ loss (commitment + codebook)
        """
        # Flatten batch and sequence dimensions
        B, K, D = z.shape
        z_flat = z.reshape(-1, D)  # [B*K, D]
        
        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )  # [B*K, num_embeddings]
        
        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)  # [B*K]
        indices = indices.view(B, K)
        
        # Quantize
        z_q = self.embedding(indices)  # [B, K, D]
        
        # Compute losses
        # Commitment loss: MSE(z, z_q.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        
        # Codebook loss (for EMA update): MSE(z.detach(), z_q)
        codebook_loss = F.mse_loss(z.detach(), z_q)
        
        # Entropy loss: encourage uniform codebook utilization
        if self.training and self.entropy_weight > 0:
            # Flatten indices
            indices_flat = indices.view(-1)  # [B*K]
            
            # Compute histogram of code usage
            hist = torch.histc(
                indices_flat.float(),
                bins=self.num_embeddings,
                min=0,
                max=self.num_embeddings - 1
            )
            
            # Normalize to probability distribution
            probs = hist / (hist.sum() + 1e-10)
            
            # Entropy: -sum(p * log(p))
            # We want high entropy (uniform distribution)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            
            # Maximize entropy = minimize negative entropy
            entropy_loss = -entropy
        else:
            entropy_loss = torch.tensor(0.0, device=z.device)
        
        # Combined VQ loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss + self.entropy_weight * entropy_loss
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        # EMA update (only in training mode)
        if self.training:
            self._ema_update(z_flat, indices.view(-1))
            
            # Dead code restart (every 100 batches, restart codes not used)
            if self.use_restart and torch.rand(1).item() < 0.01:  # 1% chance per batch
                self._restart_dead_codes(z_flat)
        
        return z_q, indices, vq_loss
    
    def _restart_dead_codes(self, z_flat: torch.Tensor):
        """
        Restart dead codes (codes that haven't been used recently).
        Replace them with random samples from the input batch.
        """
        # Find codes with very low usage
        threshold = self.ema_cluster_size.mean() * 0.01  # 1% of average
        dead_codes = (self.ema_cluster_size < threshold).nonzero(as_tuple=True)[0]
        
        if len(dead_codes) > 0:
            # Randomly sample from input batch to restart dead codes
            n_restart = min(len(dead_codes), z_flat.size(0))
            random_indices = torch.randperm(z_flat.size(0))[:n_restart]
            
            # Reset dead codes with random samples
            with torch.no_grad():
                self.embedding.weight.data[dead_codes[:n_restart]] = z_flat[random_indices]
                self.ema_embedding[dead_codes[:n_restart]] = z_flat[random_indices]
                self.ema_cluster_size[dead_codes[:n_restart]] = self.ema_cluster_size.mean()
    
    def _ema_update(self, z_flat: torch.Tensor, indices_flat: torch.Tensor):
        """
        Update codebook with exponential moving average.
        
        Args:
            z_flat: Flattened input latents [B*K, D]
            indices_flat: Flattened codebook indices [B*K]
        """
        # One-hot encode indices (match dtype with z_flat for mixed precision)
        encodings = F.one_hot(indices_flat, self.num_embeddings).to(z_flat.dtype)  # [B*K, num_embeddings]
        
        # Track code usage
        self.code_usage_count.mul_(0.99).add_(encodings.sum(0), alpha=0.01)
        
        # Update cluster sizes with EMA
        cluster_size = encodings.sum(0)  # [num_embeddings]
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        
        # Laplace smoothing to avoid dead codes
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )
        
        # Update embedding average with EMA
        # embedding_sum = sum of all z assigned to each code
        embedding_sum = torch.matmul(encodings.t(), z_flat)  # [num_embeddings, D]
        self.ema_embedding.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
        
        # Normalize by cluster size to get new embedding
        self.embedding.weight.data.copy_(
            self.ema_embedding / cluster_size.unsqueeze(1)
        )
    
    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """
        Compute percentage of codebook entries used.
        
        Args:
            indices: Codebook indices [B, K]
        
        Returns:
            Percentage of unique codes used (0-100)
        """
        unique_codes = torch.unique(indices).numel()
        return 100.0 * unique_codes / self.num_embeddings
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook indices to embeddings.
        
        Args:
            indices: Codebook indices [B, K]
        
        Returns:
            Embeddings [B, K, D]
        """
        return self.embedding(indices)


class LookupFreeQuantizer(nn.Module):
    """
    Lookup-Free Quantization (LFQ).
    
    Quantizes each dimension independently to binary values {-1, +1}.
    No explicit codebook - the quantization is implicit via the sign function.
    Effective codebook size = 2^num_codebook_dims.
    
    Advantages:
    - No codebook collapse issue
    - Perfect uniform utilization
    - Simpler training dynamics
    - No need for EMA or commitment loss
    """
    
    def __init__(
        self,
        num_codebook_dims: int = 10,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        entropy_weight: float = 0.1,
    ):
        """
        Args:
            num_codebook_dims: Number of binary dimensions (codebook_size = 2^num_codebook_dims)
            embedding_dim: Dimension of latent embeddings
            commitment_cost: Weight for commitment loss
            entropy_weight: Weight for entropy regularization loss
        """
        super().__init__()
        self.num_codebook_dims = num_codebook_dims
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.entropy_weight = entropy_weight
        self.codebook_size = 2 ** num_codebook_dims
        
        # LayerNorm to normalize encoder output before quantization
        self.pre_quant_norm = nn.LayerNorm(embedding_dim)
        
        # Project from embedding_dim to num_codebook_dims for quantization
        self.pre_quant_proj = nn.Linear(embedding_dim, num_codebook_dims)
        # Xavier init for proper variance, zero bias for balanced output
        nn.init.xavier_uniform_(self.pre_quant_proj.weight)
        nn.init.zeros_(self.pre_quant_proj.bias)
        
        # Project back from num_codebook_dims to embedding_dim
        self.post_quant_proj = nn.Linear(num_codebook_dims, embedding_dim)
    
    def quantize(self, z_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize logits to binary {-1, +1} via sign function."""
        z_q = torch.sign(z_logits)
        z_q = torch.where(z_q == 0, torch.ones_like(z_q), z_q)
        # Convert to indices for logging
        binary_codes = ((z_q + 1) / 2).long()
        powers = 2 ** torch.arange(self.num_codebook_dims - 1, -1, -1, 
                                   device=z_q.device, dtype=torch.long)
        indices = (binary_codes * powers).sum(dim=-1)
        return z_q, indices
    
    def forward(self, z: torch.Tensor, return_stats: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_norm = self.pre_quant_norm(z)
        z_logits_raw = self.pre_quant_proj(z_norm)
        
        # 1. Quantization Path (使用 Tanh 保护数值稳定性)
        z_logits = torch.tanh(z_logits_raw)
        z_q_binary, indices = self.quantize(z_logits)
        
        # STE: Pass gradient to z_logits (bounded) or z_logits_raw (unbounded)?
        # MagViT2 推荐传给 z_logits (bounded)，即你现在的做法。
        z_q_binary = z_logits + (z_q_binary - z_logits).detach()
        z_q = self.post_quant_proj(z_q_binary)
        
        commitment_loss = F.mse_loss(z_logits, z_q_binary.detach())
        entropy_loss = torch.tensor(0.0, device=z.device, requires_grad=True)
        
        if self.training:
            # [Batch * Sequence, Dim]
            z_flat = z_logits_raw.view(-1, z_logits_raw.shape[-1])
            
            # 使用 sigmoid 计算每一维变为 1 的概率
            prob_per_dim = torch.sigmoid(z_flat)
            # 计算整个 Batch 的平均概率 p_bar
            avg_prob = prob_per_dim.mean(dim=0)
            # Clamp to avoid extreme values
            avg_prob = avg_prob.clamp(0.01, 0.99)
            
            per_dim_entropy = -(avg_prob * torch.log(avg_prob + 1e-6) + 
                            (1 - avg_prob) * torch.log(1 - avg_prob + 1e-6))
            
            # 目标是最大化这个熵 -> Loss 是负的熵
            # 归一化到 [0, 1]，并 clamp 避免极端值
            max_entropy = torch.log(torch.tensor(2.0, device=z.device))
            entropy_loss = (max_entropy - per_dim_entropy.mean()) / max_entropy
            entropy_loss = entropy_loss.clamp(-1.0, 1.0)
        
        
        lfq_loss = self.commitment_cost * commitment_loss + self.entropy_weight * entropy_loss
        
        if return_stats:
            stats = {
                "z_logits_mean": z_logits.mean().item(),
                "z_logits_std": z_logits.std().item(),
                "z_logits_abs_mean": z_logits.abs().mean().item(),
                "prob_mean": avg_prob.mean().item() if self.training else 0.0,
                "prob_min": avg_prob.min().item() if self.training else 0.0,
                "prob_max": avg_prob.max().item() if self.training else 0.0,
                "commitment_loss": commitment_loss.item(),
                "entropy_loss": entropy_loss.item() if self.training else 0.0,
            }
            return z_q, indices, lfq_loss, stats
        return z_q, indices, lfq_loss
    
    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Compute percentage of codebook entries used."""
        return 100.0 * torch.unique(indices).numel() / self.codebook_size
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to embeddings."""
        B, K = indices.shape
        binary_codes = torch.zeros(B, K, self.num_codebook_dims, 
                                   device=indices.device, dtype=torch.long)
        for i in range(self.num_codebook_dims):
            binary_codes[..., i] = (indices >> (self.num_codebook_dims - 1 - i)) & 1
        # Match dtype of post_quant_proj weights
        target_dtype = self.post_quant_proj.weight.dtype
        z_q_binary = (binary_codes.float() * 2 - 1).to(dtype=target_dtype)
        return self.post_quant_proj(z_q_binary)


if __name__ == "__main__":
    B, K, D = 4, 8, 256
    z = torch.randn(B, K, D)
    
    # Test VectorQuantizer
    vq = VectorQuantizer(1024, D)
    z_q, indices, loss = vq(z)
    assert z_q.shape == z.shape
    assert torch.allclose(vq.decode_indices(indices), vq.embedding(indices))
    print("✓ VectorQuantizer test passed")
    
    # Test LookupFreeQuantizer  
    lfq = LookupFreeQuantizer(10, D)  # 2^10 = 1024 codes
    z_q, indices, loss = lfq(z)
    assert z_q.shape == z.shape
    assert lfq.decode_indices(indices).shape == z_q.shape
    print("✓ LookupFreeQuantizer test passed")

