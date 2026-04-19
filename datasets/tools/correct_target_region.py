"""Correct target region using pick object's last frame box when IoU is low

Logic:
    - If IoU(pick_box[last_frame], place_box[last_frame]) < 0.2:
      The target region selection is wrong, use pick_box as corrected_place_box
    - If IoU >= 0.2: don't add the field, use default target region

Usage:
    python correct_target_region.py --cot_dir ./cot_output_qwen_flash
    python correct_target_region.py --cot_dir ./cot_output_qwen_flash --iou_threshold 0.2 --dry_run
"""

import argparse
import json
import os
from typing import List, Optional, Tuple
from tqdm import tqdm


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def process_episode(json_path: str, iou_threshold: float = 0.2, dry_run: bool = False) -> dict:
    """Process a single episode JSON file
    
    Returns:
        dict with keys: 'episode_id', 'status', 'iou', 'corrected'
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    episode_id = data.get('episode_id', os.path.basename(json_path).replace('.json', ''))
    
    # Skip if not pick-place task
    if not data.get('is_pick_place', False):
        return {'episode_id': episode_id, 'status': 'skip_not_pick_place', 'iou': None, 'corrected': False}
    
    cot = data.get('cot', {})
    frames = cot.get('frames', [])
    
    if len(frames) == 0:
        return {'episode_id': episode_id, 'status': 'skip_no_frames', 'iou': None, 'corrected': False}
    
    # Get last frame
    last_frame = frames[-1]
    pick_box = last_frame.get('pick_box')
    place_box = last_frame.get('place_box')
    
    if pick_box is None or place_box is None:
        return {'episode_id': episode_id, 'status': 'skip_no_boxes', 'iou': None, 'corrected': False}
    
    # Compute IoU
    iou = compute_iou(pick_box, place_box)
    
    # If IoU < threshold, add corrected_place_box
    if iou < iou_threshold:
        # Add corrected place box field
        cot['corrected_place_box'] = pick_box.copy()
        cot['corrected_place_affordance'] = last_frame.get('pick_affordance', []).copy()
        cot['place_region_iou'] = iou
        cot['place_region_corrected'] = True
        
        data['cot'] = cot
        
        if not dry_run:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return {'episode_id': episode_id, 'status': 'corrected', 'iou': iou, 'corrected': True}
    else:
        # Remove corrected fields if they exist (in case of re-run)
        changed = False
        for key in ['corrected_place_box', 'corrected_place_affordance', 'place_region_iou', 'place_region_corrected']:
            if key in cot:
                del cot[key]
                changed = True
        
        if changed and not dry_run:
            data['cot'] = cot
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return {'episode_id': episode_id, 'status': 'ok', 'iou': iou, 'corrected': False}


def main():
    parser = argparse.ArgumentParser(description="Correct target region using pick object box")
    parser.add_argument('--cot_dir', type=str, required=True, help='Directory with CoT annotation JSONs')
    parser.add_argument('--iou_threshold', type=float, default=0.2, help='IoU threshold (default: 0.2)')
    parser.add_argument('--dry_run', action='store_true', help='Do not save changes, just report')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cot_dir):
        print(f"Error: Directory not found: {args.cot_dir}")
        return
    
    # Find all JSON files
    json_files = []
    for filename in os.listdir(args.cot_dir):
        if filename.endswith('.json') and filename.startswith('episode_'):
            json_files.append(os.path.join(args.cot_dir, filename))
    
    json_files = sorted(json_files)
    
    print("=" * 70)
    print("Correct Target Region Tool")
    print("=" * 70)
    print(f"CoT directory: {args.cot_dir}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Episodes found: {len(json_files)}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)
    
    # Process all episodes
    stats = {
        'total': 0,
        'corrected': 0,
        'ok': 0,
        'skipped': 0
    }
    
    corrected_episodes = []
    
    for json_path in tqdm(json_files, desc="Processing"):
        result = process_episode(json_path, args.iou_threshold, args.dry_run)
        stats['total'] += 1
        
        if result['corrected']:
            stats['corrected'] += 1
            corrected_episodes.append(result)
        elif result['status'] == 'ok':
            stats['ok'] += 1
        else:
            stats['skipped'] += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total processed: {stats['total']}")
    print(f"  ✓ OK (IoU >= {args.iou_threshold}): {stats['ok']}")
    print(f"  ⚠ Corrected (IoU < {args.iou_threshold}): {stats['corrected']}")
    print(f"  ⊘ Skipped: {stats['skipped']}")
    
    if corrected_episodes:
        print(f"\nCorrected episodes (showing first 20):")
        for ep in corrected_episodes[:20]:
            print(f"  - {ep['episode_id']}: IoU = {ep['iou']:.4f}")
        if len(corrected_episodes) > 20:
            print(f"  ... and {len(corrected_episodes) - 20} more")
    
    print("=" * 70)
    
    if args.dry_run:
        print("\n(Dry run - no changes saved)")
    else:
        print("\n✓ Done!")


if __name__ == '__main__':
    main()


