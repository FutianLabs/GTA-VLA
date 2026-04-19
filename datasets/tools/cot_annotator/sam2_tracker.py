"""SAM2-based object tracker with pluggable first-frame detector"""

import sys
import os
import tempfile
import numpy as np
import torch
from typing import List, Optional, Dict, Tuple, Union, TYPE_CHECKING
from PIL import Image
import supervision as sv

# Add Grounded-SAM-2 to path
sys.path.insert(0, "/VLA-Data/scripts/lianqing/projects/vla/X-VLA/datasets/tools/Grounded-SAM-2")
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Type checking imports
if TYPE_CHECKING:
    from .dinox_client import DINOXClient
    from .vlm_client import VLMGroundingClient


class SAM2Tracker:
    """Track objects across frames using first-frame detector + SAM2 video tracking"""
    
    def __init__(self, 
                 detector_client: Union['DINOXClient', 'VLMGroundingClient'],
                 sam2_checkpoint: str = "./checkpoints/sam2.1_hiera_large.pt",
                 sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = None):
        """
        Initialize SAM2 tracker
        
        Args:
            detector_client: First-frame detector (DINOXClient, VLMGroundingClient, QwenVLClient, or RexOmniClient)
            sam2_checkpoint: Path to SAM2 checkpoint (relative to Grounded-SAM-2 dir)
            sam2_config: Path to SAM2 config (relative to Grounded-SAM-2 dir)
            device: Device to use (cuda/cpu), auto-detect if None
        """
        self.detector_client = detector_client
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize SAM2 models
        self._init_sam2_models(sam2_checkpoint, sam2_config)
    
    def _init_sam2_models(self, checkpoint: str, config: str):
        """Initialize SAM2 image and video predictors"""
        # Enable optimizations for CUDA
        if self.device == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Change to Grounded-SAM-2 directory for config loading
        grounded_sam2_dir = "/VLA-Data/scripts/lianqing/projects/vla/X-VLA/datasets/tools/Grounded-SAM-2"
        original_dir = os.getcwd()
        
        try:
            os.chdir(grounded_sam2_dir)
            
            # Build models (using relative paths)
            self.sam2_video_predictor = build_sam2_video_predictor(config, checkpoint, device=self.device)
            sam2_image_model = build_sam2(config, checkpoint, device=self.device)
            self.sam2_image_predictor = SAM2ImagePredictor(sam2_image_model)
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def track_objects(self, 
                      images: np.ndarray,
                      pick_object: str,
                      place_region: str) -> Tuple[List[Optional[List[float]]], List[Optional[List[float]]]]:
        """
        Track pick object and place region across all frames (single action)
        
        Args:
            images: Array of shape (T, H, W, 3) - all frames
            pick_object: Object name to pick
            place_region: Region name to place
            
        Returns:
            Tuple of (pick_boxes, place_boxes) where each is a list of [x1,y1,x2,y2] or None per frame
        """
        # Use multi-action tracking with single action
        actions = [{'pick_object': pick_object, 'place_region': place_region}]
        results = self.track_multi_actions(images, actions)
        return results[0]['pick_boxes'], results[0]['place_boxes']
    
    def track_multi_actions(self,
                           images: np.ndarray,
                           actions: List[Dict[str, str]]) -> List[Dict]:
        """
        Track multiple pick-place actions across all frames
        
        Args:
            images: Array of shape (T, H, W, 3) - all frames
            actions: List of action dicts, each with 'pick_object' and 'place_region'
            
        Returns:
            List of dicts, each containing:
                - pick_boxes: List of [x1,y1,x2,y2] or None per frame
                - place_boxes: List of [x1,y1,x2,y2] or None per frame
                - pick_object: str
                - place_region: str
        """
        num_frames = len(images)
        
        # Initialize results for each action - now supporting multiple boxes per frame
        results = []
        for action in actions:
            results.append({
                'pick_object': action['pick_object'],
                'place_region': action['place_region'],
                'pick_boxes': [[] for _ in range(num_frames)],  # List of lists: [[box1, box2, ...], ...]
                'place_boxes': [[] for _ in range(num_frames)]  # List of lists: [[box1, box2, ...], ...]
            })
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames to disk (required by SAM2 video predictor)
            frame_paths = []
            for i, img in enumerate(images):
                frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
                Image.fromarray(img.astype(np.uint8)).save(frame_path, 'JPEG')
                frame_paths.append(frame_path)
            
            # Detect all objects in first frame
            first_frame_pil = Image.fromarray(images[0].astype(np.uint8))
            
            # Collect all boxes to track (including multiple boxes per object)
            objects_to_track = []
            object_info = []  # (action_idx, 'pick'/'place', box_idx_in_list)
            input_boxes = []
            
            for action_idx, action in enumerate(actions):
                # Detect pick object (returns list of boxes: [[x1,y1,x2,y2], ...])
                pick_boxes = self.detector_client.ground_object(
                    first_frame_pil,
                    action['pick_object']
                )
                # Track ALL detected pick boxes
                if pick_boxes and len(pick_boxes) > 0:
                    for box_idx, pick_box in enumerate(pick_boxes):
                        if len(pick_box) == 4:
                            objects_to_track.append(f"{action['pick_object']}_{box_idx}")
                            object_info.append((action_idx, 'pick', box_idx))
                            input_boxes.append(pick_box)
                
                # Detect place region (returns list of boxes: [[x1,y1,x2,y2], ...])
                place_boxes = self.detector_client.ground_object(
                    first_frame_pil,
                    action['place_region']
                )
                # Track ALL detected place boxes
                if place_boxes and len(place_boxes) > 0:
                    for box_idx, place_box in enumerate(place_boxes):
                        if len(place_box) == 4:
                            objects_to_track.append(f"{action['place_region']}_{box_idx}")
                            object_info.append((action_idx, 'place', box_idx))
                            input_boxes.append(place_box)
            
            # If no detections, return empty results
            if len(input_boxes) == 0:
                return results
            
            input_boxes = np.array(input_boxes)
            
            # Get masks for first frame using SAM2 image predictor
            self.sam2_image_predictor.set_image(images[0])
            masks, scores, logits = self.sam2_image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            # Convert mask shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            # Initialize video predictor with frame directory
            inference_state = self.sam2_video_predictor.init_state(video_path=temp_dir)
            
            # Register all objects in first frame using box prompts
            for obj_idx, (obj_name, box) in enumerate(zip(objects_to_track, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.sam2_video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_idx,
                    box=box,
                )
            
            # Propagate tracking across all frames
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # Convert masks to bounding boxes for each frame
            for frame_idx in range(num_frames):
                if frame_idx not in video_segments:
                    continue
                
                segments = video_segments[frame_idx]
                
                for obj_id in sorted(segments.keys()):
                    mask = segments[obj_id]
                    
                    # Convert mask to bounding box
                    if mask.ndim == 3:
                        mask = mask.squeeze(0)
                    
                    # Get bounding box from mask
                    xyxy = sv.mask_to_xyxy(mask[np.newaxis, :, :])
                    
                    if xyxy.shape[0] > 0:
                        box = xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        # Assign to correct action's pick/place boxes (multiple boxes per frame)
                        list_idx = obj_id - 1
                        if list_idx < len(object_info):
                            action_idx, obj_type, box_idx = object_info[list_idx]
                            if obj_type == 'pick':
                                results[action_idx]['pick_boxes'][frame_idx].append(box)
                            elif obj_type == 'place':
                                results[action_idx]['place_boxes'][frame_idx].append(box)
        
        return results
    
    def ground_object(self, image: Image.Image, object_name: str) -> Optional[List[List[float]]]:
        """
        Fallback single-frame detection (delegates to detector client)
        
        Args:
            image: PIL Image
            object_name: Name of object to locate
            
        Returns:
            List of bounding boxes as [[x1, y1, x2, y2], ...] or empty list
        """
        return self.detector_client.ground_object(image, object_name)


