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
from typing import Iterable, Tuple, Dict, Type
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
# =============================================================================
# Registry
# =============================================================================
ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""
    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections (e.g. apply sigmoid).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------------------
    # Core supervised loss
    # ---------------------------------------------------------------------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    # ---------------------------------------------------------------------
    # Space-level hooks
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action


# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")


def _rotate6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert rotate6d (CONCATENATED format) to rotation matrix.
    
    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    This matches libero_client.Mat_to_Rotate6D format.
    """
    a1 = rot6d[..., :3]   # first column: [col1[0], col1[1], col1[2]]
    a2 = rot6d[..., 3:6]  # second column: [col2[0], col2[1], col2[2]]
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / torch.linalg.norm(a1, dim=-1, keepdim=True).clamp(min=1e-8)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2 - proj
    b2 = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True).clamp(min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack([b1, b2, b3], dim=-1)


def _matrix_to_rotate6d(mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to rotate6d (CONCATENATED format).
    
    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    """
    col1 = mat[..., :, 0]
    col2 = mat[..., :, 1]
    return torch.cat([col1, col2], dim=-1)
    

def convert_rotate6d_from_rel_to_abs(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    R_abs = R_proprio @ R_rel
    Handles broadcasting for (Batch, 6) proprio and (Batch, Time, 6) action.
    """
    R_proprio = _rotate6d_to_matrix(proprio) # Shape: (..., 3, 3)
    R_rel = _rotate6d_to_matrix(action)      # Shape: (..., 3, 3)
    
    if R_proprio.ndim == R_rel.ndim - 1:
        R_proprio = R_proprio.unsqueeze(-3)
    
    R_abs = torch.matmul(R_proprio, R_rel)
    return _matrix_to_rotate6d(R_abs)


def convert_rotate6d_from_abs_to_rel(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    R_rel = R_proprio^T @ R_abs
    """
    R_proprio = _rotate6d_to_matrix(proprio)
    R_abs = _rotate6d_to_matrix(action)
    
    if R_proprio.ndim == R_abs.ndim - 1:
        R_proprio = R_proprio.unsqueeze(-3)
    
    R_rel = torch.matmul(R_proprio.transpose(-2, -1), R_abs)
    return _matrix_to_rotate6d(R_rel)


def _euler_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (xyz convention) to rotation matrix.
    
    Args:
        euler: Tensor of shape (..., 3) with Euler angles in radians (rx, ry, rz)
    
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    
    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    cos_z, sin_z = torch.cos(rz), torch.sin(rz)
    
    # Rotation matrix for xyz (extrinsic) = Rz @ Ry @ Rx
    # Row 0
    r00 = cos_y * cos_z
    r01 = sin_x * sin_y * cos_z - cos_x * sin_z
    r02 = cos_x * sin_y * cos_z + sin_x * sin_z
    # Row 1
    r10 = cos_y * sin_z
    r11 = sin_x * sin_y * sin_z + cos_x * cos_z
    r12 = cos_x * sin_y * sin_z - sin_x * cos_z
    # Row 2
    r20 = -sin_y
    r21 = sin_x * cos_y
    r22 = cos_x * cos_y
    
    # Stack into matrix [..., 3, 3]
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def convert_euler_to_rotate6d(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to rotate6d representation.
    
    Args:
        euler: Tensor of shape (..., 3) with Euler angles in radians (xyz convention)
    
    Returns:
        Tensor of shape (..., 6) with rotate6d representation
    """
    pos1 = euler[..., 0:3]
    rot6d1 = euler[..., 3:6]
    grip1 = euler[..., 6:7]
    pos2 = euler[..., 7:10]
    rot6d2 = euler[..., 10:13]
    grip2 = euler[..., 13:14]
    mat1 = _euler_to_matrix(rot6d1)
    mat1 = _matrix_to_rotate6d(mat1)
    mat2 = _euler_to_matrix(rot6d2)
    mat2 = _matrix_to_rotate6d(mat2)
    return torch.cat([pos1, mat1, grip1, pos2, mat2, grip2], dim=-1)



def convert_rotate6d_to_euler(abs_traj: torch.Tensor) -> torch.Tensor:
    """Convert rotate6d to Euler angles at dims [3:9] and optionally [13:19].
    
    Input layout (20D dual-arm):  [pos1(3), rot6d1(6), grip1(1), pos2(3), rot6d2(6), grip2(1)]
    Output layout (14D dual-arm): [pos1(3), euler1(3), grip1(1), pos2(3), euler2(3), grip2(1)]
    
    Input layout (10D single-arm):  [pos(3), rot6d(6), grip(1)]
    Output layout (7D single-arm):  [pos(3), euler(3), grip(1)]
    """
    if abs_traj.ndim == 1:
        abs_traj = abs_traj.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    dim = abs_traj.shape[-1]
    
    # Single arm case: dims [3:9] contain rotate6d
    assert dim == 20, f"Expected dim=20 (dual-arm), got {dim}"
    
    pos1 = abs_traj[..., :3]
    euler1 = _rotate6d_to_euler_torch(abs_traj[..., 3:9])
    grip1 = abs_traj[..., 9:10]
    pos2 = abs_traj[..., 10:13]
    euler2 = _rotate6d_to_euler_torch(abs_traj[..., 13:19])
    grip2 = abs_traj[..., 19:20]
    
    result = torch.cat([pos1, euler1, grip1, pos2, euler2, grip2], dim=-1)
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result

def convert_euler_from_abs_to_rel(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Convert absolute Euler angles to relative.
    
    Simple subtraction for Euler angles (approximation valid for small angles).
    
    Args:
        proprio: Base Euler angles, shape (3,) or (..., 3)
        action: Absolute Euler angles, shape (..., 3)
    
    Returns:
        Relative Euler angles, same shape as action
    """
    return action - proprio


def convert_euler_from_rel_to_abs(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Convert relative Euler angles to absolute.
    
    Simple addition for Euler angles (approximation valid for small angles).
    
    Args:
        proprio: Base Euler angles, shape (3,) or (..., 3)
        action: Relative Euler angles, shape (..., 3)
    
    Returns:
        Absolute Euler angles, same shape as action
    """
    return action + proprio



def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    """Convert rotate6d (CONCATENATED format) to euler xyz angles.
    
    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    """
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., :3]   # first column
    a2 = v6[..., 3:6]  # second column
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz')

def rotate6d_to_quat(v6: np.ndarray, scalar_first = False) -> np.ndarray:
    """Convert rotate6d (CONCATENATED format) to quaternion.
    
    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    """
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1, a2 = v6[..., :3], v6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_quat(scalar_first = scalar_first)


def _rotate6d_to_euler_torch(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert rotate6d (CONCATENATED format) to Euler angles (xyz convention).
    
    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    
    Args:
        rot6d: Tensor of shape (..., 6) representing the first two columns of rotation matrix.
    
    Returns:
        Tensor of shape (..., 3) with Euler angles in radians.
    """
    original_shape = rot6d.shape[:-1]
    rot6d_flat = rot6d.reshape(-1, 6)
    
    # Check for degenerate cases (all zeros or very small norm)
    norm = torch.linalg.norm(rot6d_flat, dim=-1)
    valid_mask = norm > 1e-6
    
    # Initialize output with zeros (for degenerate cases)
    euler_flat = torch.zeros(rot6d_flat.shape[0], 3, dtype=rot6d.dtype, device=rot6d.device)
    
    if valid_mask.any():
        valid_rot6d = rot6d_flat[valid_mask]
        
        # Extract the two column vectors (CONCATENATED format)
        a1 = valid_rot6d[..., :3]   # first column
        a2 = valid_rot6d[..., 3:6]  # second column
        
        # Gram-Schmidt orthogonalization with clamping to avoid division by zero
        norm_a1 = torch.linalg.norm(a1, dim=-1, keepdim=True).clamp(min=1e-8)
        b1 = a1 / norm_a1
        proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = a2 - proj
        norm_b2 = torch.linalg.norm(b2, dim=-1, keepdim=True).clamp(min=1e-8)
        b2 = b2 / norm_b2
        b3 = torch.cross(b1, b2, dim=-1)
        
        # Stack to form rotation matrix [..., 3, 3]
        rot_mat = torch.stack([b1, b2, b3], dim=-1)
        
        # Convert rotation matrix to Euler angles (xyz convention)
        rot_mat_np = rot_mat.detach().cpu().numpy()
        
        # Handle any remaining NaN/Inf in rotation matrices
        valid_matrices = np.isfinite(rot_mat_np).all(axis=(1, 2))
        euler_valid = np.zeros((rot_mat_np.shape[0], 3), dtype=np.float64)
        
        if valid_matrices.any():
            euler_valid[valid_matrices] = R.from_matrix(rot_mat_np[valid_matrices]).as_euler('xyz')
        
        euler_flat[valid_mask] = torch.from_numpy(euler_valid).to(dtype=rot6d.dtype, device=rot6d.device)
    
    return euler_flat.reshape(*original_shape, 3)


# =============================================================================
# Action Space Implementations
# =============================================================================


@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        # Gripper BCE
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ position
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # Rotation 6D
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action




@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """Joint-space layout — Pi0.5-style continuous regression for ALL dims.

    Both joints and gripper are normalized to [-1, 1] using quantile
    statistics (q01/q99) and trained with a single unified MSE loss.
    No BCE, no sigmoid, no gripper zeroing.

    ``set_norm_stats()`` provides per-dim q01/q99 (7-dim: 6 joints in
    radians + 1 gripper in meters). ``postprocess`` maps [-1, 1] back
    to physical units.
    """

    dim_action = 14
    ACTION_SCALE = 500.0
    gripper_idx = (6, 13)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self._norm_q01 = None
        self._norm_q99 = None
        self._norm_indices = None

    def set_norm_stats(self, stats: dict):
        """Attach q01/q99 for inference-time denormalization.

        Args:
            stats: dict with keys ``q01`` (list[float]), ``q99`` (list[float]),
                   and optionally ``joint_indices`` (list[int]).
                   For piper_joint the lists are 7-long (6 joints + gripper).
        """
        q01 = torch.tensor(stats["q01"], dtype=torch.float32)
        q99 = torch.tensor(stats["q99"], dtype=torch.float32)
        self._norm_q01 = q01
        self._norm_q99 = q99
        self._norm_indices = stats.get("joint_indices", list(range(len(q01))))

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        return {"action_loss": self.mse(pred, target) * self.ACTION_SCALE}

    def preprocess(self, proprio, action, mode="train"):
        """Pass through — all channels (including gripper) are kept."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize from [-1, 1] back to physical units via q01/q99."""
        if self._norm_q01 is not None:
            q01 = self._norm_q01.to(action.device, action.dtype)
            q99 = self._norm_q99.to(action.device, action.dtype)
            rng = q99 - q01
            idx = self._norm_indices
            action[..., idx] = (action[..., idx] + 1.0) / 2.0 * rng + q01

        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGI-bot variant of EE6DActionSpace using MSE for all components."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        gripper_loss = self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """No preprocessing applied in AGIBOT variant."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """AGIBOT does not postprocess."""
        return action



@register_action("ee3d")
class EE3DActionSpace(EE6DActionSpace):
    """End-effector layout with xyz, 3D Euler rotation, and gripper channels.
    
    Layout (14D dual-arm): [pos1(3), euler1(3), grip1(1), pos2(3), euler2(3), grip2(1)]
    """

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (7, 8, 9)
    ROT_IDX_1 = (3, 4, 5)
    ROT_IDX_2 = (10, 11, 12)

    def __init__(self):
        super().__init__()

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits and convert euler to rotate6d.
        
        Input (14D):  [pos1(3), euler1(3), grip1(1), pos2(3), euler2(3), grip2(1)]
        Output (20D): [pos1(3), rot6d1(6), grip1(1), pos2(3), rot6d2(6), grip2(1)]
        """
        # Apply sigmoid to grippers (indices 6 and 13)
        action = action.clone()
        action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        
        # Convert euler to rotate6d - this function returns 20D output
        action = convert_euler_to_rotate6d(action)
        
        return action


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "EE6DActionSpace",
    "EE3DActionSpace",
    "JointActionSpace",
    "AGIBOTEE6DActionSpace",
    "ACTION_REGISTRY",
    "convert_rotate6d_from_rel_to_abs",
    "convert_rotate6d_from_abs_to_rel",
    "convert_euler_from_rel_to_abs",
    "convert_euler_from_abs_to_rel",
    "convert_euler_to_rotate6d",
]




if __name__ == "__main__":
    import copy
    proprio = torch.randn(1, 1, 6)
    action = torch.randn(1, 2, 6)
    rotation_matrix = _rotate6d_to_matrix(action)
    print(rotation_matrix)
    print(action)

    relative_action = convert_rotate6d_from_abs_to_rel(proprio, action)
    # print(relative_action)
    absoluate_action = convert_rotate6d_from_rel_to_abs(proprio, relative_action)
    print("================================================")
    rotation_matrix_new = _rotate6d_to_matrix(absoluate_action)
    print(rotation_matrix)
    print(absoluate_action)
    print(absoluate_action.shape)
    print(torch.allclose(rotation_matrix, rotation_matrix_new, atol=1e-5))

    # write a rotat6d to euler and convert back check
    # proprio = torch.randn(1, 1, 20)
    # action = torch.randn(1, 1, 20)
    # rotation_matrix = _rotate6d_to_matrix(action[..., 3:9])
    # print(action[:, :, 3:9], "action", rotation_matrix)

    # euler = convert_rotate6d_to_euler(action)
    # relative_euler = copy.deepcopy(euler)
    # relative_euler[..., 3:6] = convert_euler_from_abs_to_rel(proprio[..., 3:6], relative_euler[..., 3:6])
    # relative_euler[..., 10:13] = convert_euler_from_abs_to_rel(proprio[..., 10:13], relative_euler[..., 10:13])
    # abs_euler  = copy.deepcopy(relative_euler)
    # abs_euler[..., 3:6] = convert_euler_from_rel_to_abs(proprio[..., 3:6], abs_euler[..., 3:6])
    # abs_euler[..., 10:13] = convert_euler_from_rel_to_abs(proprio[..., 10:13], abs_euler[..., 10:13])
    # print(euler[..., 3:6], "euler")
    # print(abs_euler[..., 3:6], "abs_euler")
    # euler = copy.deepcopy(abs_euler)
    # # print(euler)
    # new = convert_euler_to_rotate6d(euler)
    # new_matrix = _rotate6d_to_matrix(new[..., 3:9])
    # print(new[:, :, 3:9], "new", new_matrix)
