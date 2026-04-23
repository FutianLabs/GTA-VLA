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
XVLA Configuration with Multi-Backbone Support.

Supports:
  - Florence2 (default): encoder-decoder backbone
  - Qwen3-VL: decoder-only backbone with 2B params

Usage:
    # Florence2 backbone (default)
    config = XVLAConfig(florence_config={...})
    
    # Qwen3-VL backbone
    config = XVLAConfig(
        vlm_backbone_type="qwen3_vl",
        qwen3_pretrained="Qwen/Qwen3-VL-2B-Instruct",
    )
"""

from typing import Optional, Dict, Any, Union, List, Tuple
from .configuration_florence2 import Florence2Config
from transformers.configuration_utils import PretrainedConfig


class XVLAConfig(PretrainedConfig):
    """
    Configuration class for the **XVLA (Extended Vision-Language-Action)** model.

    This configuration defines all submodules of XVLA in a single place:
      - The visual-language backbone (Florence2 or Qwen3-VL)
      - The temporal/action transformer
      - The action/proprio setup
    
    Args:
        vlm_backbone_type (str): VLM backbone type, "florence2" or "qwen3_vl"
        florence_config (dict): Florence2 config (used when vlm_backbone_type="florence2")
        qwen3_pretrained (str): Qwen3-VL pretrained path (used when vlm_backbone_type="qwen3_vl")
        qwen3_torch_dtype (str): Dtype for Qwen3-VL ("bfloat16", "float16", "float32")
        qwen3_use_flash_attn (bool): Use Flash Attention 2 for Qwen3-VL
        hidden_size (int): Hidden dimension for action transformer
        depth (int): Number of transformer layers
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio
        use_soft_prompt (bool): Enable domain-specific soft prompts and domain-aware layers
        num_domains (int): Number of robot domains
        len_soft_prompts (int): Soft prompt length per domain
        dim_time (int): Time embedding dimension
        max_len_seq (int): Maximum sequence length
        use_hetero_proj (bool): Use domain-specific projections
        num_actions (int): Number of action steps to predict
        action_mode (str): Action representation ("ee6d", "ee3d", etc.)
        use_proprio (bool): Use proprioception
    """

    model_type = "xvla"

    def __init__(
        self,
        # === VLM backbone selection ===
        vlm_backbone_type: str = "florence2",  # "florence2" or "qwen3_vl"
        
        # === Florence2 backbone (default) ===
        florence_config: Optional[Union[dict, Florence2Config]] = None,
        florence_pretrained_name_or_path: str | None = None,
        # === Qwen3-VL backbone (alternative) ===
        qwen3_pretrained: str = "Qwen/Qwen3-VL-2B-Instruct",
        qwen3_torch_dtype: str = "bfloat16",
        qwen3_use_flash_attn: bool = True,
        # === Transformer head ===
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_soft_prompt: bool = True,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        
        # === VLM Feature Projection ===
        # Use multi-layer MLP for projecting VLM features to transformer hidden_size
        # Recommended when VLM output dim (e.g., 2048) >> transformer hidden_size (e.g., 1024)
        use_multilayer_proj: bool = False,  # Enable multi-layer MLP projection
        proj_num_layers: int = 2,           # Number of layers (2 or 3)
        proj_hidden_size: Optional[int] = None,  # Hidden size (auto: geometric mean)
        proj_dropout: float = 0.1,          # Dropout rate in projector

        # === Action & proprio ===
        num_actions: int = 30,
        action_mode: str = "ee6d",
        use_proprio: bool = True,
        joint_norm_stats: Optional[Dict] = None,  # q01/q99 for joint-space normalization
        # === VLM hidden states extraction (GR00T-style) ===
        vlm_hidden_layer: int = -1,  # Which layer to use: -1=last, positive=specific layer index
        vlm_use_multi_layer: bool = False,  # Whether to fuse multiple layers
        vlm_multi_layer_indices: Optional[List[int]] = None,  # e.g., [-4, -3, -2, -1] for last 4 layers
        # === CoT (Chain-of-Thought) Training ===
        use_cot_training: bool = False,  # Enable CoT training with object grounding
        cot_loss_weight: float = 1.0,  # Weight for CoT cross-entropy loss
        cot_max_length: int = 768,  # Max length for CoT text tokens
        cot_only_pretrain: bool = False,  # Only train CoT loss, skip action loss (for VLM pre-alignment)
        # CoT Builder Configuration (image_size is read from annotation)
        cot_coord_scale: int = 1000,  # Coordinate scale (Qwen3-VL uses 0-1000)
        cot_gripper_future_steps: int = 5,  # Number of distance-sampled gripper points
        cot_detector_priority: Optional[List[str]] = None,  # Detector priority: seed_vl preferred over dino_x
        # === CoT pick keyframe oversampling (Bridge CoT handler) ===
        cot_pick_keyframe_oversample: bool = False,
        cot_pick_keyframe_skip_initial: int = 1,
        cot_pick_keyframe_num_anchors: int = 2,
        cot_pick_oversample_radius: int = 4,
        cot_pick_oversample_boost: float = 2.0,
        cot_pick_oversample_stochastic: bool = False,
        # === User Interaction Augmentation ===
        use_interaction_augmentation: bool = False,  # Enable user interaction augmentation
        interaction_aug_ratio: float = 0.5,  # Ratio of samples with interaction
        interaction_modes: Optional[Dict[str, float]] = None,  # Mode probabilities
        # === Multi-sample diffusion ===
        num_diffusion_samples: int = 1,  # K timesteps per item; >1 reuses VLM features to train action head faster
        # === Views ===
        num_views: int = 3,  # Total number of camera views (shared with dataset & processor)
        # === Optional View Augmentation ===
        aug_with_optional_view: bool = False,  # Randomly swap main view with optional views
        # === Dual-Frequency Training ===
        # VLM runs on a previous frame (low freq), action head runs on current frame (high freq)
        use_dual_frequency: bool = False,
        vlm_frame_max_offset: int = 10,  # Max frame offset for VLM frame; sampled from [0, max_offset]
        **kwargs,
    ):
        # VLM backbone type
        self.vlm_backbone_type = vlm_backbone_type.lower()
        
        # Florence2 backbone configuration
        if isinstance(florence_config, dict):
            self.florence_config = Florence2Config(**florence_config)
        elif isinstance(florence_config, Florence2Config):
            self.florence_config = florence_config
        elif self.vlm_backbone_type == "florence2":
            self.florence_config = Florence2Config()
        else:
            self.florence_config = None
        self.florence_pretrained_name_or_path = florence_pretrained_name_or_path

        # Qwen3-VL backbone configuration
        self.qwen3_pretrained = qwen3_pretrained
        self.qwen3_torch_dtype = qwen3_torch_dtype
        self.qwen3_use_flash_attn = qwen3_use_flash_attn

        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_soft_prompt = use_soft_prompt
        self.num_domains = num_domains
        self.len_soft_prompts = len_soft_prompts
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq
        self.use_hetero_proj = use_hetero_proj
        
        # VLM feature projection settings
        self.use_multilayer_proj = use_multilayer_proj
        self.proj_num_layers = proj_num_layers
        self.proj_hidden_size = proj_hidden_size
        self.proj_dropout = proj_dropout

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        self.joint_norm_stats = joint_norm_stats

        # VLM hidden states extraction (GR00T-style intermediate layers)
        self.vlm_hidden_layer = vlm_hidden_layer
        self.vlm_use_multi_layer = vlm_use_multi_layer
        self.vlm_multi_layer_indices = vlm_multi_layer_indices or [-1]

        # CoT (Chain-of-Thought) Training with Object Grounding
        self.use_cot_training = use_cot_training
        self.cot_loss_weight = cot_loss_weight
        self.cot_max_length = cot_max_length
        self.cot_only_pretrain = cot_only_pretrain
        # CoT Builder Configuration (image_size is read from annotation)
        self.cot_coord_scale = cot_coord_scale
        self.cot_gripper_future_steps = cot_gripper_future_steps
        self.cot_detector_priority = cot_detector_priority or ["seed_vl", "dino_x"]
        self.cot_pick_keyframe_oversample = cot_pick_keyframe_oversample
        self.cot_pick_keyframe_skip_initial = cot_pick_keyframe_skip_initial
        self.cot_pick_keyframe_num_anchors = cot_pick_keyframe_num_anchors
        self.cot_pick_oversample_radius = cot_pick_oversample_radius
        self.cot_pick_oversample_boost = cot_pick_oversample_boost
        self.cot_pick_oversample_stochastic = cot_pick_oversample_stochastic

        # User Interaction Augmentation
        self.use_interaction_augmentation = use_interaction_augmentation
        self.interaction_aug_ratio = interaction_aug_ratio
        # Multi-sample diffusion
        self.num_diffusion_samples = num_diffusion_samples
        # Views
        self.num_views = num_views
        # Optional View Augmentation
        self.aug_with_optional_view = aug_with_optional_view
        # Dual-Frequency Training
        self.use_dual_frequency = use_dual_frequency
        self.vlm_frame_max_offset = vlm_frame_max_offset
        # aug_ratio is the single gate; all modes here are real augmentation types
        self.interaction_modes = interaction_modes or {
            "pick_box": 0.30,
            "place_box": 0.20,
            "pick_and_place": 0.20,
            "affordance_2d": 0.20,
            "gripper_path_2d": 0.10,
        }

        # Initialize base HF config attributes (e.g. name_or_path)
        super().__init__(**kwargs)

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert this configuration (and its Florence sub-config)
        into a fully serializable dictionary for HF save/load.
        """
        output = super().to_dict()
        if self.florence_config is not None:
            output["florence_config"] = self.florence_config.to_dict()
        output["florence_pretrained_name_or_path"] = self.florence_pretrained_name_or_path

        return output


class GTAVLAConfig(XVLAConfig):
    """Primary public config name for GTA-VLA."""

    model_type = "gtavla"