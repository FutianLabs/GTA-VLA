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

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- Small utils ----------------------------------

def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """
    MLP used in ViT-style blocks.

    Supports Linear or 1x1 Conv 'linear_layer' for token/channel mixing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, C] for Linear variant; caller is responsible for shapes.
        input_dtype = x.dtype
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        # Handle DeepSpeed bf16 mixed precision - cast to norm weight dtype
        if hasattr(self.norm, 'weight') and self.norm.weight is not None:
            x = self.norm(x.to(self.norm.weight.dtype)).to(input_dtype)
        else:
            x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with optional fused SDPA fallback.

    If PyTorch provides `scaled_dot_product_attention`, it will be used
    (usually faster and more stable); otherwise we use a manual implementation.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, C]
            Input sequence.
        attn_mask : Tensor, optional
            Bool mask [B, 1, 1, T] where True=keep, False=mask (key-side).

        Returns
        -------
        Tensor, shape [B, T, C]
            Output sequence after MHSA + projection.
        """
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)  # 3 x [B, H, T, Dh]
        )
        q, k, v = qkv.unbind(0)  # each: [B, H, T, Dh]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # [B, H, T, Dh]
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)        # [B, H, T, T]
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float('-inf'))
                else:
                    attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v                           # [B, H, T, Dh]

        x = x.transpose(1, 2).reshape(B, T, C)     # [B, T, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# ------------------------------- Utilities -----------------------------------

def basic_init(module: nn.Module) -> None:
    """
    Apply a conservative initialization scheme to Linear and LayerNorm layers.

    - Linear: Xavier uniform with gain=0.5 (conservative)
    - Bias: Set to zero
    - LayerNorm: weight=1, bias=0
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_action_head(module: nn.Module, use_soft_prompt: bool = False) -> None:
    """
    Initialize action decoder with small output weights for stable training.
    
    Strategy:
    1. Hidden layers: conservative Xavier (gain=0.5)
    2. LayerNorm: weight=1, bias=0
    3. Output layer: very small weights (std=1e-3), zero bias
    
    This ensures initial actions are close to 0, preventing large position_loss at step 0.
    """
    if use_soft_prompt:
        # DomainAwareLinear: weight stored in fc.weight (Embedding)
        if hasattr(module, 'fc') and isinstance(module.fc, nn.Embedding):
            # Small output initialization for action decoder
            nn.init.normal_(module.fc.weight, mean=0.0, std=1e-3)
        if hasattr(module, 'bias') and isinstance(module.bias, nn.Embedding):
            nn.init.zeros_(module.bias.weight)
    else:
        # Regular Linear layer
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t : torch.Tensor
        Shape [B]. Each element is a timestep index, may be fractional.
    dim : int
        Dimensionality of the output embedding.
    max_period : int, default=100
        Controls the minimum frequency of the sinusoids.

    Returns
    -------
    torch.Tensor
        Shape [B, dim]. Sinusoidal embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ------------------------------- Core Layers ----------------------------------

class MultiLayerProjector(nn.Module):
    """
    Multi-layer MLP projector for dimension reduction with less information loss.
    
    Useful when projecting from high-dim (e.g., 2048) to low-dim (e.g., 1024).
    Uses intermediate layers to gradually reduce dimensions.
    
    Architecture:
        2-layer: input -> hidden -> output
        3-layer: input -> hidden1 -> hidden2 -> output
    
    Example:
        >>> # Project Qwen3-VL features (2048d) to transformer (1024d)
        >>> proj = MultiLayerProjector(2048, 1024, num_layers=2)
        >>> x = torch.randn(4, 100, 2048)  # [B, T, D]
        >>> y = proj(x)  # [4, 100, 1024]
    
    Args:
        input_size: Input dimension (e.g., 2048 for Qwen3-VL)
        output_size: Output dimension (e.g., 1024 for transformer)
        hidden_size: Hidden layer dimension (default: geometric mean of input/output)
                    For 2048->1024, auto computes ~1448
        num_layers: Number of layers (2 or 3, default: 2)
        dropout: Dropout rate (default: 0.1)
    
    Usage in config:
        {
          "use_multilayer_proj": true,
          "proj_num_layers": 2,
          "proj_hidden_size": 1536,  // or null for auto
          "proj_dropout": 0.1
        }
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert num_layers in [2, 3], "num_layers must be 2 or 3"
        
        # Auto-determine hidden_size as geometric mean if not specified
        if hidden_size is None:
            hidden_size = int(math.sqrt(input_size * output_size))
        
        layers = []
        if num_layers == 2:
            # Two-layer: input -> hidden -> output
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            ])
        else:  # num_layers == 3
            # Three-layer: input -> hidden1 -> hidden2 -> output
            hidden1 = int((input_size * 2 + hidden_size) / 3)
            hidden2 = int((hidden_size * 2 + output_size) / 3)
            layers.extend([
                nn.Linear(input_size, hidden1),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(hidden2, output_size),
            ])
        
        self.proj = nn.Sequential(*layers)
        
        # Initialize with small weights for stable training
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, I]
            Input features.
        
        Returns
        -------
        Tensor, shape [B, T, O]
            Projected features.
        """
        return self.proj(x)


class DomainAwareLinear(nn.Module):
    """
    Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [B, I] or [B, T, I]
        domain_id : LongTensor
            [B], domain indices.

        Returns
        -------
        Tensor
            [B, O] or [B, T, O]
        """
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size).to(x.dtype)
        b = self.bias(domain_id).view(B, self.output_size).to(x.dtype)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y


class TransformerBlock(nn.Module):
    """Standard Transformer block (pre-LN): LN → MHSA → residual, LN → MLP → residual."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, [B, T, H]
            Input action tokens.
        attn_mask : Tensor, optional
            Bool mask [B, 1, 1, T] where True=keep, False=mask (key-side).

        Returns
        -------
        Tensor, [B, T, H]
            Output action tokens.
        """
        weight_dtype = self.norm1.weight.dtype
        input_dtype = x.dtype
        x = x + self.attn(self.norm1(x.to(weight_dtype)), attn_mask=attn_mask).to(input_dtype)
        x = x + self.mlp(self.norm2(x.to(weight_dtype))).to(input_dtype)
        return x


# --------------------------- Main Model ---------------------------------------

class SoftPromptedTransformer(nn.Module):
    """
    Multi-modal, domain-aware Transformer with optional soft prompts.

    See parameter and forward I/O descriptions inside the docstrings.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        use_multilayer_proj: bool = False,
        proj_num_layers: int = 2,
        proj_hidden_size: Optional[int] = None,
        proj_dropout: float = 0.1,
        use_soft_prompt: bool = True,
        use_main_view: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.use_soft_prompt = use_soft_prompt
        self.len_soft_prompts = len_soft_prompts if use_soft_prompt else 0
        self.use_hetero_proj = use_hetero_proj
        self.use_multilayer_proj = use_multilayer_proj
        self.use_main_view = use_main_view

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Build VLM and auxiliary visual projectors
        # Strategy:
        # 1. use_hetero_proj=True: domain-aware linear (for multi-robot scenarios)
        # 2. use_multilayer_proj=True: multi-layer MLP (for dimension mismatch)
        # 3. default: simple linear projection
        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
        elif use_multilayer_proj:
            # Use multi-layer MLP for better information preservation
            # Especially useful when multi_modal_input_size >> hidden_size (e.g., 2048 -> 1024)
            self.vlm_proj = MultiLayerProjector(
                multi_modal_input_size,
                hidden_size,
                hidden_size=proj_hidden_size,
                num_layers=proj_num_layers,
                dropout=proj_dropout,
            )
            self.aux_visual_proj = MultiLayerProjector(
                multi_modal_input_size,
                hidden_size,
                hidden_size=proj_hidden_size,
                num_layers=proj_num_layers,
                dropout=proj_dropout,
            )
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        if use_main_view:
            if use_hetero_proj:
                self.main_visual_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            elif use_multilayer_proj:
                self.main_visual_proj = MultiLayerProjector(
                    multi_modal_input_size, hidden_size,
                    hidden_size=proj_hidden_size, num_layers=proj_num_layers, dropout=proj_dropout,
                )
            else:
                self.main_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        
        if use_soft_prompt:
            self.action_encoder = DomainAwareLinear(
                dim_action + dim_time + dim_propio, hidden_size, num_domains=num_domains
            )
            self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)
            if len_soft_prompts > 0:
                self.soft_prompt_hub = nn.Embedding(num_domains, self.len_soft_prompts * hidden_size)
                nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)
        else:
            self.action_encoder = nn.Linear(dim_action + dim_time + dim_propio, hidden_size)
            self.action_decoder = nn.Linear(hidden_size, dim_action)

        self.apply(basic_init)
        
        # Apply specialized initialization to action decoder for stable training
        init_action_head(self.action_decoder, use_soft_prompt=use_soft_prompt)
    
    def forward(
        self,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: Optional[torch.Tensor],
        t: torch.Tensor,
        domain_id: Optional[torch.LongTensor] = None,
        aux_token_mask: Optional[torch.Tensor] = None,
        main_visual_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Inputs
        ------
        vlm_features : [B, T_vlm, D]
        aux_visual_inputs : [B, T_aux, D]
        action_with_noise : [B, T_action, dim_action]
        proprio : [B, dim_propio] or None (if proprio is fed to VLM instead)
        t : [B]
        domain_id : [B], optional. Required only when use_soft_prompt=True.
        aux_token_mask : [B, T_aux], optional. Mask for invalid aux tokens.
        main_visual_inputs : [B, T_main, D], optional. Raw main-view features
            from the current frame (dual-frequency mode).

        Returns
        -------
        Tensor
            Predicted actions, [B, T_action, dim_action]
        """
        B, num_actions = action_with_noise.shape[:2]

        # Encode (action + proprio + time) → tokens
        time_emb = timestep_embedding(t, self.dim_time)                     # [B, dim_time]
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        
        if self.use_soft_prompt:
            x = self.action_encoder(action_tokens, domain_id)               # [B, T_action, H]
        else:
            x = self.action_encoder(action_tokens)                          # [B, T_action, H]

        # Compute main-view token count for positional embedding budget
        T_main = main_visual_inputs.shape[1] if (self.use_main_view and main_visual_inputs is not None) else 0

        # Truncate VLM features if total sequence would exceed positional embedding capacity
        max_vlm_tokens = self.pos_emb.shape[1] - num_actions - aux_visual_inputs.shape[1] - T_main
        if vlm_features.shape[1] > max_vlm_tokens:
            vlm_features = vlm_features[:, :max_vlm_tokens, :]

        # Project visual streams
        if self.use_hetero_proj:
            vlm_proj = self.vlm_proj(vlm_features, domain_id)
            aux_proj = self.aux_visual_proj(aux_visual_inputs, domain_id)
        else:
            vlm_proj = self.vlm_proj(vlm_features)
            aux_proj = self.aux_visual_proj(aux_visual_inputs)

        # Project and insert main-view features (dual-frequency mode)
        if self.use_main_view and main_visual_inputs is not None:
            if self.use_hetero_proj:
                main_proj = self.main_visual_proj(main_visual_inputs, domain_id)
            else:
                main_proj = self.main_visual_proj(main_visual_inputs)
            x = torch.cat([x, vlm_proj, main_proj, aux_proj], dim=1)
        else:
            x = torch.cat([x, vlm_proj, aux_proj], dim=1)

        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.pos_emb[:, :seq_len, :].to(x.dtype)

        # Append soft prompts
        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(B, self.len_soft_prompts, self.hidden_size).to(x.dtype)
            x = torch.cat([x, soft_prompts], dim=1)

        # Build attention mask: prevent attending to invalid aux view tokens
        attn_mask = None
        if aux_token_mask is not None:
            T_vlm = vlm_features.shape[1]
            T_aux = aux_visual_inputs.shape[1]
            # action + vlm + (optional main_view) are always valid
            prefix_len = num_actions + T_vlm + T_main
            prefix_mask = torch.ones(B, prefix_len, device=x.device, dtype=torch.bool)
            keep_mask = torch.cat([prefix_mask, aux_token_mask], dim=1)
            if self.len_soft_prompts > 0:
                suffix_mask = torch.ones(B, self.len_soft_prompts, device=x.device, dtype=torch.bool)
                keep_mask = torch.cat([keep_mask, suffix_mask], dim=1)
            attn_mask = keep_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        # Decode only the action segment (handle DeepSpeed bf16 mixed precision)
        x_out = x[:, :num_actions]
        x_normed = self.norm(x_out.to(self.norm.weight.dtype)).to(x_out.dtype)
        if self.use_soft_prompt:
            return self.action_decoder(x_normed, domain_id)
        else:
            return self.action_decoder(x_normed)