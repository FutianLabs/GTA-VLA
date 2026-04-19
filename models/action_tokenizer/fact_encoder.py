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
FACT Encoder: Dual-stream MMBlock encoder for action chunks.

Uses learnable query tokens and joint attention with action context
to compress action trajectories into latent token representations.
"""

import torch
import torch.nn as nn
from .mmdit_block import MMBlock

class FACTEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int = 20,
        num_actions: int = 30,
        latent_dim: int = 256,
        num_latent_tokens: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        
        # 1. Action Projection
        self.input_proj = nn.Linear(action_dim, latent_dim)
        
        # 2. Positional Embedding (必须开启！Action是序列)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_actions, latent_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # 3. Learnable Query Tokens
        self.query_embed = nn.Parameter(torch.zeros(1, num_latent_tokens, latent_dim))
        nn.init.orthogonal_(self.query_embed.view(num_latent_tokens, latent_dim))
        
        # 4. MMBlock Layers
        self.blocks = nn.ModuleList([
            MMBlock(latent_dim, num_heads) for _ in range(num_layers)
        ])
        
        # 5. Output Norm & Head
        self.norm_final = nn.LayerNorm(latent_dim) # 推荐加上
        self.output_head = nn.Linear(latent_dim, latent_dim)
        
        # Init output head
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        T = actions.shape[1]
        
        # Query Stream
        x = self.query_embed.repeat(B, 1, 1)  # [B, K, D]
        
        # Context Stream (必须加 Pos Embed)
        # 做了 slice 保护，防止输入长度 T 和 init 长度不一致报错
        c = self.input_proj(actions) + self.pos_embed[:, :T, :] 
        
        # Dual-stream interaction
        for block in self.blocks:
            x, c = block(x, c)
        
        # Final Norm & Projection
        x = self.norm_final(x)
        return self.output_head(x)