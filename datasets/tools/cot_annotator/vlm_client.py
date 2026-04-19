"""VLM client for object grounding using Doubao API with OpenAI SDK

Supports Seed1.5-VL style grounding with <bbox> tags.
Reference: https://github.com/ByteDance-Seed/Seed1.5-VL/tree/main/Grounding
"""

import os
import re
import time
import base64
import io
from typing import List, Optional, Tuple
from PIL import Image
from openai import OpenAI


class VLMGroundingClient:
    """Ground objects and regions in images using VLM"""
    
    def __init__(self, api_key: str = None, 
                 model: str = "doubao-1.5-vision-pro-250328"):
        """
        Args:
            api_key: Doubao API key (if None, uses env var ARK_API_KEY or default)
            model: Model endpoint ID
        """
        # Use provided api_key, or env var, or default
        if api_key is None:
            api_key = os.getenv('ARK_API_KEY', 'REMOVED_ARK_API_KEY')
        
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key,
        )
    
    def ground_object(self, 
                      image: Image.Image, 
                      object_name: str,
                      retry_count: int = 3) -> Optional[List[List[int]]]:
        """
        Locate object in image and return all bounding boxes
        
        Args:
            image: PIL Image
            object_name: Name of object to locate
            retry_count: Number of retries on failure
            
        Returns:
            List of bounding boxes as [[x1, y1, x2, y2], ...] in pixel coordinates,
            sorted by center point (top-left to bottom-right).
            Returns empty list on failure or if no detections.
        """
        if not object_name:
            return None
        
        # Convert image to base64
        image_base64 = self._image_to_base64(image)
        
        # Seed1.5-VL style grounding prompt with <bbox> tags
        prompt = f"""Locate "{object_name}" in this robot manipulation image.

For each instance found, output a bounding box in format: <bbox>x1 y1 x2 y2</bbox>
- Coordinates are integers from 0 to 999 (normalized to image dimensions)
- (x1, y1) = top-left corner, (x2, y2) = bottom-right corner
- x1 < x2 and y1 < y2

If multiple instances exist, output multiple <bbox> tags on separate lines.
If the object is not found, output: <bbox>none</bbox>"""

        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                
                # Extract text content from chat.completions API format
                content = response.choices[0].message.content
                
                if content:
                    # Try Seed1.5-VL <bbox> tag format first
                    norm_boxes = self._parse_bbox_tags(content)
                    
                    # Fallback to JSON format if no bbox tags found
                    if not norm_boxes:
                        parsed = self._extract_json(content)
                        if parsed:
                            # Handle both old format (single bbox) and new format (multiple boxes)
                            if 'boxes' in parsed and parsed['boxes']:
                                norm_boxes = parsed['boxes']
                            elif 'bbox' in parsed and parsed['bbox'] is not None:
                                norm_boxes = [parsed['bbox']]
                    
                    # Process boxes: convert from 0-999 normalized to pixel coordinates
                    if norm_boxes:
                        img_width = image.width
                        img_height = image.height
                        
                        valid_boxes = []
                        for box in norm_boxes:
                            if box and len(box) == 4:
                                # Validate coordinates are in expected range
                                x1_n, y1_n, x2_n, y2_n = box
                                if all(0 <= v <= 999 for v in box) and x1_n < x2_n and y1_n < y2_n:
                                    # Convert from 0-999 normalized coords to pixel coords (as integers)
                                    x1 = int(round((x1_n / 999.0) * img_width))
                                    y1 = int(round((y1_n / 999.0) * img_height))
                                    x2 = int(round((x2_n / 999.0) * img_width))
                                    y2 = int(round((y2_n / 999.0) * img_height))
                                    
                                    pixel_box = [x1, y1, x2, y2]
                                    
                                    # Calculate center point for sorting
                                    center_x = (x1 + x2) / 2
                                    center_y = (y1 + y2) / 2
                                    
                                    valid_boxes.append((pixel_box, center_x, center_y))
                        
                        if valid_boxes:
                            # Sort by center point: top-left to bottom-right
                            # Primary sort: y coordinate (top to bottom)
                            # Secondary sort: x coordinate (left to right)
                            valid_boxes.sort(key=lambda item: (item[2], item[1]))
                            return [box[0] for box in valid_boxes]
                    
                    return []
            except Exception as e:
                print(f"VLM API error (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)
        
        return []
    
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
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        # Resize if too large
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to JPEG
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _parse_bbox_tags(self, text: str) -> List[List[int]]:
        """
        Parse Seed1.5-VL style <bbox> tags from response text.
        
        Format: <bbox>x1 y1 x2 y2</bbox>
        - Coordinates are space-separated integers in range [0, 999]
        - (x1, y1) = top-left, (x2, y2) = bottom-right
        
        Reference: https://github.com/ByteDance-Seed/Seed1.5-VL/tree/main/Grounding
        
        Args:
            text: Response text from VLM
            
        Returns:
            List of bounding boxes as [[x1, y1, x2, y2], ...]
            Empty list if no valid boxes found
        """
        boxes = []
        
        # Pattern to match <bbox>x1 y1 x2 y2</bbox>
        # Allow flexible whitespace between coordinates
        pattern = r'<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                x1, y1, x2, y2 = map(int, match.groups())
                # Validate coordinates are in range and form valid box
                if (0 <= x1 <= 999 and 0 <= y1 <= 999 and 
                    0 <= x2 <= 999 and 0 <= y2 <= 999 and
                    x1 < x2 and y1 < y2):
                    boxes.append([x1, y1, x2, y2])
            except (ValueError, IndexError):
                continue
        
        return boxes
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from VLM response text"""
        import json
        try:
            # Try direct JSON parse
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON in markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except:
                    pass
        
        # Try to find JSON object with braces
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        
        return None

