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
Flow Matching utilities for continuous trajectory generation.

Implements rectified flow / flow matching for high-fidelity action reconstruction.
Based on "Flow Matching for Generative Modeling" (Lipman et al., 2023)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def sample_timesteps(batch_size: int, device: torch.device, strategy: str = "uniform") -> torch.Tensor:
    """
    Sample timesteps for flow matching training.
    
    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        strategy: Sampling strategy ('uniform' or 'logit_normal')
    
    Returns:
        Timesteps in [0, 1], shape [B]
    """
    if strategy == "uniform":
        # Uniform sampling in [0, 1]
        return torch.rand(batch_size, device=device)
    elif strategy == "logit_normal":
        # Logit-normal sampling (more timesteps near 0 and 1)
        # Better for capturing distribution at boundaries
        t = torch.randn(batch_size, device=device) * 1.0  # mean=0, std=1
        t = torch.sigmoid(t)  # Map to [0, 1]
        return t
    else:
        raise ValueError(f"Unknown timestep sampling strategy: {strategy}")


def linear_flow(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear interpolation flow (rectified flow).
    
    Flow path: x_t = (1 - t) * x0 + t * x1
    Velocity: v_t = x1 - x0
    
    Args:
        x0: Start point (noise) [B, ...]
        x1: End point (data) [B, ...]
        t: Timesteps [B] or [B, 1, 1]
    
    Returns:
        x_t: Interpolated point [B, ...]
        v_t: Target velocity [B, ...]
    """
    # Ensure t has correct shape for broadcasting
    while t.dim() < x0.dim():
        t = t.unsqueeze(-1)
    
    # Linear interpolation
    x_t = (1 - t) * x0 + t * x1
    
    # Constant velocity
    v_t = x1 - x0
    
    return x_t, v_t


def flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    gripper_idx: Tuple[int, ...] = (9, 19),
    pos_weight: float = 1.0,
    rot_weight: float = 1.0,
    gripper_weight: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute flow matching loss with weighted components.
    
    Args:
        predicted_velocity: Predicted velocity [B, T, D]
        target_velocity: Target velocity [B, T, D]
        gripper_idx: Indices of gripper dimensions
        pos_weight: Weight for position loss
        rot_weight: Weight for rotation loss
        gripper_weight: Weight for gripper loss
    
    Returns:
        total_loss: Weighted total loss
        loss_dict: Dictionary with component losses
    """
    # MSE loss for non-gripper dimensions
    non_gripper_mask = torch.ones_like(predicted_velocity, dtype=torch.bool)
    for idx in gripper_idx:
        non_gripper_mask[..., idx] = False
    
    # Position loss (first 3 dims for each arm)
    pos_idx = [0, 1, 2, 10, 11, 12]
    pos_mask = torch.zeros_like(predicted_velocity, dtype=torch.bool)
    for idx in pos_idx:
        pos_mask[..., idx] = True
    pos_loss = F.mse_loss(
        predicted_velocity[pos_mask],
        target_velocity[pos_mask]
    ) * pos_weight
    
    # Rotation loss (rotation dims)
    rot_idx = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]
    rot_mask = torch.zeros_like(predicted_velocity, dtype=torch.bool)
    for idx in rot_idx:
        rot_mask[..., idx] = True
    rot_loss = F.mse_loss(
        predicted_velocity[rot_mask],
        target_velocity[rot_mask]
    ) * rot_weight
    
    # Gripper loss (BCE for binary gripper)
    gripper_mask = torch.zeros_like(predicted_velocity, dtype=torch.bool)
    for idx in gripper_idx:
        gripper_mask[..., idx] = True
    gripper_loss = F.mse_loss(
        predicted_velocity[gripper_mask],
        target_velocity[gripper_mask]
    ) * gripper_weight
    
    # Total loss
    total_loss = pos_loss + rot_loss + gripper_loss
    
    loss_dict = {
        "pos_loss": pos_loss,
        "rot_loss": rot_loss,
        "gripper_loss": gripper_loss,
    }
    
    return total_loss, loss_dict


def flow_sample(
    model_fn,
    noise: torch.Tensor,
    condition: torch.Tensor,
    num_steps: int = 10,
    method: str = "euler",
) -> torch.Tensor:
    """
    Sample from flow model using ODE integration.
    
    Args:
        model_fn: Function (x_t, t, cond) -> v_t that predicts velocity
        noise: Initial noise x_0 [B, T, D]
        condition: Conditioning information (e.g., quantized latents)
        num_steps: Number of integration steps
        method: Integration method ('euler' or 'heun')
    
    Returns:
        x_1: Generated sample [B, T, D]
    """
    x_t = noise
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.full((noise.shape[0],), i * dt, device=noise.device, dtype=noise.dtype)
        
        if method == "euler":
            # Euler method: x_{t+dt} = x_t + dt * v_t
            v_t = model_fn(x_t, t, condition)
            x_t = x_t + dt * v_t
        
        elif method == "heun":
            # Heun's method (2nd order)
            v_t = model_fn(x_t, t, condition)
            x_next = x_t + dt * v_t
            
            t_next = torch.full((noise.shape[0],), (i + 1) * dt, device=noise.device, dtype=noise.dtype)
            v_next = model_fn(x_next, t_next, condition)
            
            x_t = x_t + dt * (v_t + v_next) / 2
        
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    return x_t


if __name__ == "__main__":
    # Test flow matching utilities
    batch_size = 4
    seq_len = 30
    action_dim = 20
    
    # Test timestep sampling
    t_uniform = sample_timesteps(batch_size, "cpu", "uniform")
    print(f"Uniform timesteps: {t_uniform}")
    assert t_uniform.min() >= 0 and t_uniform.max() <= 1
    
    # Test linear flow
    x0 = torch.randn(batch_size, seq_len, action_dim)
    x1 = torch.randn(batch_size, seq_len, action_dim)
    t = torch.rand(batch_size)
    x_t, v_t = linear_flow(x0, x1, t)
    
    print(f"x_t shape: {x_t.shape}")
    print(f"v_t shape: {v_t.shape}")
    assert x_t.shape == x0.shape
    assert v_t.shape == x0.shape
    
    # Test flow matching loss
    pred_v = torch.randn(batch_size, seq_len, action_dim)
    total_loss, loss_dict = flow_matching_loss(pred_v, v_t)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    print("✓ Flow matching utilities test passed")

