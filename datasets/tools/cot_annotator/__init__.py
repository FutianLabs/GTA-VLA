"""CoT Annotation Support Modules

This package provides detection and tracking clients used by the main
annotation pipeline in tools/subtask_annotator/.

Available modules:
- DINOXClient: DINO-X based object detection
- VLMGroundingClient: Seed VL based object detection  
- SAM2Tracker: SAM2 video tracking

Note: Main annotation pipeline is in tools/subtask_annotator/unified_pipeline.py
"""

from .sam2_tracker import SAM2Tracker
from .dinox_client import DINOXClient
from .vlm_client import VLMGroundingClient

__all__ = [
    'SAM2Tracker',
    'DINOXClient', 
    'VLMGroundingClient',
]




