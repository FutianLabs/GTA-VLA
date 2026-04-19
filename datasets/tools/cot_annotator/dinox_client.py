"""DINO-X client for object grounding"""

import re
import time
from typing import Dict, List, Optional
from PIL import Image
from .utils_annotator import get_dds_client, get_detection_result


def extract_nouns(text: str) -> str:
    """
    Extract nouns from text for DINO-X detection.
    DINO-X works best with simple noun phrases.
    
    Examples:
        "red apple" -> "red apple"
        "the red apple on the table" -> "red apple, table"
        "blue bowl" -> "blue bowl"
        "bowl of fruits" -> "bowl, fruits"
        "pot cardboard fence" -> "pot, cardboard, fence"
        "coffee mug on wooden table" -> "coffee mug, wooden table"
    
    Args:
        text: Input text (e.g., "red apple on the table" or "pot cardboard fence")
        
    Returns:
        Comma-separated nouns (e.g., "red apple, table" or "pot, cardboard, fence")
    """
    if not text:
        return text
    
    # Remove common determiners and prepositions
    stop_words = {
        'the', 'a', 'an', 'on', 'in', 'at', 'to', 'of', 'for', 'with', 
        'by', 'from', 'into', 'onto', 'upon', 'under', 'over', 'near'
    }
    
    # Common adjectives that modify nouns (color, size, material, etc.)
    common_adjectives = {
        'red', 'blue', 'green', 'yellow', 'white', 'black', 'brown', 'gray', 'grey', 
        'pink', 'purple', 'orange', 'small', 'large', 'big', 'tiny', 'wooden', 
        'metal', 'plastic', 'glass', 'paper', 'left', 'right', 'top', 'bottom',
        'front', 'back', 'old', 'new', 'clean', 'dirty', 'empty', 'full'
    }
    
    # Common compound nouns (two words that go together)
    compound_patterns = [
        r'\b(coffee|tea|wine|water)\s+(cup|mug|glass|bottle)\b',
        r'\b(cutting|chopping)\s+(board)\b',
        r'\b(kitchen|bathroom|living|dining)\s+(sink|table|room)\b',
        r'\b(trash|garbage|recycling)\s+(can|bin)\b',
    ]
    
    # First, split by prepositions to get major noun phrases
    parts = re.split(r'\s+(?:on|in|at|to|of|for|with|by|from|into|onto|upon|under|over|near)\s+', text.lower())
    
    # Process each part
    all_nouns = []
    for part in parts:
        words = part.strip().split()
        # Filter out stop words
        clean_words = [w for w in words if w not in stop_words]
        
        if not clean_words:
            continue
        
        # Check if this contains a compound noun pattern
        part_text = ' '.join(clean_words)
        is_compound = any(re.search(pattern, part_text) for pattern in compound_patterns)
        
        if is_compound:
            # Keep as compound noun
            all_nouns.append(part_text)
            continue
        
        # Check if this is likely a multi-noun phrase (no adjectives or all nouns)
        has_adjective = any(w in common_adjectives for w in clean_words)
        
        # If no adjectives and more than 2 words, likely multiple independent nouns
        # Or if exactly 2 words with no adjectives, check if first word could be adjective modifier
        if not has_adjective and len(clean_words) > 2:
            # Likely multiple independent nouns, split them
            # E.g., "pot cardboard fence" -> ["pot", "cardboard", "fence"]
            all_nouns.extend(clean_words)
        elif not has_adjective and len(clean_words) == 2:
            # Two words, no known adjective - could be compound noun or two nouns
            # Keep as compound for safety (e.g., "coffee mug", "kitchen table")
            all_nouns.append(' '.join(clean_words))
        else:
            # Keep as a phrase (adjective + noun or single noun)
            all_nouns.append(' '.join(clean_words))
    
    # Return comma-separated unique nouns
    unique_nouns = []
    for noun in all_nouns:
        if noun and noun not in unique_nouns:
            unique_nouns.append(noun)
    
    return ', '.join(unique_nouns) if unique_nouns else text


class DINOXClient:
    """Ground objects in images using DINO-X API"""
    
    def __init__(self, dds_token: str):
        """
        Args:
            dds_token: DDS API token
        """
        self.dds_token = dds_token
        self.client = get_dds_client(dds_token)
    
    def ground_object(self, 
                      image: Image.Image, 
                      object_name: str,
                      bbox_threshold: float = 0.1,
                      retry_count: int = 3) -> Optional[List[List[int]]]:
        """
        Locate object in image and return all bounding boxes
        
        Args:
            image: PIL Image
            object_name: Name of object to locate (e.g., "red apple on table")
            bbox_threshold: Detection confidence threshold
            retry_count: Number of retries on failure
            
        Returns:
            List of bounding boxes as [[x1, y1, x2, y2], ...] in pixel coordinates,
            sorted by center point (top-left to bottom-right).
            Returns None on failure or empty list if no detections.
            
        Note:
            For DINO-X, we extract nouns from the object_name.
            E.g., "red apple on table" -> "red apple, table"
        """
        if not object_name:
            return None
        
        # Extract nouns for DINO-X (works better with simple noun phrases)
        extracted_nouns = extract_nouns(object_name)
        
        # Save image temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, 'JPEG')
        
        try:
            for attempt in range(retry_count):
                try:
                    # Call DINO-X API with extracted nouns
                    text_prompt = f".{extracted_nouns}"
                    result = get_detection_result(
                        tmp_path, 
                        text_prompt, 
                        self.client,
                        bbox_threshold=bbox_threshold
                    )
                    
                    # Filter boxes with score > 0.2 and sort by center point (top-left to bottom-right)
                    if result and 'objects' in result and len(result['objects']) > 0:
                        valid_boxes = []
                        for obj in result['objects']:
                            score = obj.get('score', 0)
                            bbox = obj.get('bbox', None)
                            if bbox and len(bbox) == 4 and score > 0.3:
                                box = [int(round(x)) for x in bbox]  # [x1, y1, x2, y2] as integers
                                # Calculate center point
                                center_x = (box[0] + box[2]) / 2
                                center_y = (box[1] + box[3]) / 2
                                valid_boxes.append((box, center_x, center_y, score))
                        
                        if valid_boxes:
                            # Sort by center point: top-left to bottom-right
                            # Primary sort: y coordinate (top to bottom)
                            # Secondary sort: x coordinate (left to right)
                            valid_boxes.sort(key=lambda item: (item[2], item[1]))
                            # Return all boxes (sorted)
                            return [box[0] for box in valid_boxes]
                    
                    # No valid detection found
                    return []
                    
                except Exception as e:
                    print(f"Error in DINO-X detection (attempt {attempt+1}): {e}")
                    if attempt < retry_count - 1:
                        time.sleep(1)
                        continue
                    return []
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return []
    
    def ground_objects_batch(self,
                             image: Image.Image,
                             object_names: List[str],
                             bbox_threshold: float = 0.1,
                             retry_count: int = 3) -> Dict[str, List[List[int]]]:
        """
        Batch detect multiple objects in one API call.
        
        Args:
            image: PIL Image
            object_names: List of object names to detect
            bbox_threshold: Detection confidence threshold
            retry_count: Number of retries on failure
            
        Returns:
            Dictionary mapping object_name -> list of bboxes [[x1,y1,x2,y2], ...]
            
        Note:
            DINO-X uses "." to separate multiple objects in one query.
            E.g., ".red apple.blue bowl.table"
        """
        if not object_names:
            return {}
        
        # Initialize result dict with empty lists
        result_dict = {name: [] for name in object_names}
        
        # Build mapping from extracted nouns back to original names
        # and create batch prompt
        noun_to_original = {}
        extracted_list = []
        for name in object_names:
            extracted = extract_nouns(name)
            extracted_list.append(extracted)
            # Map each extracted noun phrase back to original
            noun_to_original[extracted.lower()] = name
            # Also map individual parts if comma-separated
            for part in extracted.split(', '):
                noun_to_original[part.strip().lower()] = name
        
        # Build batch prompt: ".object1.object2.object3"
        batch_prompt = "." + ".".join(extracted_list)
        
        # Save image temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, 'JPEG')
        
        try:
            for attempt in range(retry_count):
                try:
                    result = get_detection_result(
                        tmp_path,
                        batch_prompt,
                        self.client,
                        bbox_threshold=bbox_threshold
                    )
                    
                    if result and 'objects' in result and len(result['objects']) > 0:
                        # Group detections by category
                        for obj in result['objects']:
                            score = obj.get('score', 0)
                            bbox = obj.get('bbox', None)
                            category = obj.get('category', '').lower().strip()
                            
                            if bbox and len(bbox) == 4 and score > 0.3:
                                box = [int(round(x)) for x in bbox]  # Integer coordinates
                                
                                # Match category to original object name
                                matched_name = None
                                
                                # Try exact match first
                                if category in noun_to_original:
                                    matched_name = noun_to_original[category]
                                else:
                                    # Try partial match (category contains or is contained by)
                                    for noun, orig in noun_to_original.items():
                                        if category in noun or noun in category:
                                            matched_name = orig
                                            break
                                
                                if matched_name:
                                    result_dict[matched_name].append(box)
                    
                    # Sort boxes for each object by center point
                    for name in result_dict:
                        if result_dict[name]:
                            boxes_with_center = []
                            for box in result_dict[name]:
                                center_x = (box[0] + box[2]) / 2
                                center_y = (box[1] + box[3]) / 2
                                boxes_with_center.append((box, center_x, center_y))
                            # Sort by y then x (top-left to bottom-right)
                            boxes_with_center.sort(key=lambda item: (item[2], item[1]))
                            result_dict[name] = [b[0] for b in boxes_with_center]
                    
                    return result_dict
                    
                except Exception as e:
                    print(f"Error in DINO-X batch detection (attempt {attempt+1}): {e}")
                    if attempt < retry_count - 1:
                        time.sleep(1)
                        continue
                    return result_dict
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return result_dict
    
    def ground_pick_and_place(self,
                              pick_image: Image.Image,
                              place_image: Image.Image,
                              pick_object: str,
                              place_region: str) -> dict:
        """
        Ground both pick object and place region
        
        Args:
            pick_image: Image at pick frame
            place_image: Image at place frame
            pick_object: Object to pick
            place_region: Region to place
            
        Returns:
            Dictionary with pick_box and place_box
        """
        pick_box = self.ground_object(pick_image, pick_object)
        
        # Small delay between API calls
        time.sleep(0.5)
        
        place_box = self.ground_object(place_image, place_region)
        
        return {
            'pick_box': pick_box,
            'place_box': place_box
        }

