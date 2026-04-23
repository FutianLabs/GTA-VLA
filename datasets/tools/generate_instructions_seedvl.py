"""
Generate instructions for Bridge dataset episodes using SeedVL.

This script processes H5 files that don't have meaningful instructions and
uses SeedVL VLM to generate natural language instructions based on video frames.

Usage:
    python datasets/tools/generate_instructions_seedvl.py \
        --h5_dir data/openX/gtavla/bridge_wrist \
        --nproc 10 
        --dry_run

Arguments:
    --h5_dir: Directory containing H5 files
    --batch_size: Number of episodes to process in one run (default: 100)
    --nproc: Number of parallel workers (default: 4)
    --dry_run: Preview generated instructions without saving to H5 files
    --start_idx: Start processing from this episode index (default: 0)
    --provider: VLM provider - 'doubao' (SeedVL) or 'qwen' (default: doubao)

Output:
    Updates H5 files with:
    - instruction: Generated instruction string
    - instruction_source: "seedvl"
    
Note:
    Extracts 6 evenly-spaced keyframes from each episode for VLM processing.
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import time

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from mmengine.utils import track_parallel_progress

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def is_empty_or_gibberish_instruction(instruction):
    """
    Check if instruction is empty or likely gibberish.
    Same logic as in bridge_tfrecord_to_h5.py for consistency.
    """
    if instruction is None:
        return True
    
    instruction = str(instruction).strip()
    
    if len(instruction) == 0:
        return True
    
    if len(instruction) < 3:
        return True
    
    words = instruction.split()
    if len(words) == 1 and len(instruction) > 10:
        alpha_chars = sum(1 for c in instruction if c.isalpha())
        if alpha_chars / len(instruction) < 0.5:
            return True
    
    return False


def load_keyframes_from_h5(h5_path, num_frames=6):
    """
    Load keyframes from H5 file for instruction generation.
    
    Args:
        h5_path: Path to H5 file
        num_frames: Number of frames to extract evenly (default: 6)
        
    Returns:
        List of PIL Images, or None if failed
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'observation' not in f or 'image_0' not in f['observation']:
                return None
            
            images = f['observation']['image_0'][:]
            total_frames = len(images)
            
            if total_frames == 0:
                return None
            
            # Extract frames evenly across the episode
            if total_frames <= num_frames:
                # If episode has fewer frames than requested, use all frames
                indices = list(range(total_frames))
            else:
                # Extract num_frames evenly spaced frames
                # Use linspace to get evenly distributed indices
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
            
            keyframes = []
            for idx in indices:
                img_array = images[idx]
                pil_img = Image.fromarray(img_array)
                keyframes.append(pil_img)
            
            return keyframes
            
    except Exception as e:
        return None


def get_episode_info(h5_path):
    """
    Get episode metadata from H5 file.
    
    Returns:
        Dict with instruction, instruction_source, wrist_view_valid
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            info = {
                'instruction': f.attrs.get('instruction', ''),
                'instruction_source': f.attrs.get('instruction_source', 'original'),
                'wrist_view_valid': f.attrs.get('wrist_view_valid', False),
            }
            # Decode if bytes
            if isinstance(info['instruction'], bytes):
                info['instruction'] = info['instruction'].decode('utf-8')
            if isinstance(info['instruction_source'], bytes):
                info['instruction_source'] = info['instruction_source'].decode('utf-8')
            return info
    except Exception as e:
        return None


def update_h5_instruction(h5_path, instruction, instruction_source="seedvl"):
    """
    Update H5 file with new instruction.
    
    Args:
        h5_path: Path to H5 file
        instruction: New instruction string
        instruction_source: Source of instruction (default: "seedvl")
    """
    try:
        with h5py.File(h5_path, 'a') as f:
            # Update instruction
            if 'instruction' in f.attrs:
                del f.attrs['instruction']
            f.attrs['instruction'] = instruction
            
            # Update instruction source
            if 'instruction_source' in f.attrs:
                del f.attrs['instruction_source']
            f.attrs['instruction_source'] = instruction_source
            
    except Exception as e:
        raise


def collect_existing_instructions(h5_files, max_samples=50):
    """
    Collect existing meaningful instructions from the dataset.
    Used to provide style examples to the VLM.
    """
    instructions = []
    for h5_path in h5_files[:max_samples * 2]:  # Sample more to filter
        info = get_episode_info(h5_path)
        if info and not is_empty_or_gibberish_instruction(info['instruction']):
            instructions.append(info['instruction'])
            if len(instructions) >= max_samples:
                break
    return instructions


# Global variable to hold VLM client per worker process
_worker_client = None
_worker_existing_instructions = None


def init_worker(provider, existing_instructions):
    """
    Initialize VLM client for each worker process.
    Called once when worker process starts.
    """
    global _worker_client, _worker_existing_instructions
    from tools.subtask_annotator.video_vlm_client import VideoVLMClient
    _worker_client = VideoVLMClient(provider=provider)
    _worker_existing_instructions = existing_instructions


def process_single_episode(args):
    """
    Process a single episode - load keyframes, generate instruction, update H5.
    
    Args:
        args: Tuple of (h5_path, dry_run)
        
    Returns:
        Dict with episode name, instruction (or error), and success status
    """
    h5_path, dry_run = args
    episode_name = os.path.basename(h5_path)
    
    global _worker_client, _worker_existing_instructions
    
    try:
        # Load keyframes
        keyframes = load_keyframes_from_h5(h5_path)
        if keyframes is None:
            return {
                'episode': episode_name,
                'instruction': None,
                'success': False,
                'error': 'Failed to load keyframes'
            }
        
        # Generate instruction
        instruction = _worker_client.generate_instruction(
            keyframes, 
            _worker_existing_instructions
        )
        
        if instruction is None:
            return {
                'episode': episode_name,
                'instruction': None,
                'success': False,
                'error': 'Failed to generate instruction'
            }
        
        # Update H5 file if not dry run
        if not dry_run:
            try:
                update_h5_instruction(h5_path, instruction, "seedvl")
            except Exception as e:
                return {
                    'episode': episode_name,
                    'instruction': instruction,
                    'success': False,
                    'error': f'Failed to update H5: {e}'
                }
        
        return {
            'episode': episode_name,
            'instruction': instruction,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'episode': episode_name,
            'instruction': None,
            'success': False,
            'error': str(e)
        }


def _scan_episode_info(h5_path):
    info = get_episode_info(h5_path)
    if info is None:
        return None
    return h5_path, info


def main(args):
    print("=" * 80)
    print("Bridge Dataset Instruction Generator (SeedVL) - Multiprocessing")
    print("=" * 80)
    
    # Find all H5 files
    h5_pattern = os.path.join(args.h5_dir, "episode_*.hdf5")
    h5_files = sorted(glob(h5_pattern))
    
    if not h5_files:
        print(f"No H5 files found in {args.h5_dir}")
        return
    
    print(f"Found {len(h5_files)} H5 files")
    
    # Filter to episodes without instruction
    episodes_to_process = []
    episodes_with_instruction = 0
    episodes_already_seedvl = 0
    
    print("Scanning episodes...")
    scan_results = track_parallel_progress(_scan_episode_info, h5_files, nproc=args.nproc)
    for result in scan_results:
        if result is None:
            continue
        h5_path, info = result

        # Skip if already has SeedVL-generated instruction
        if info['instruction_source'] == 'seedvl':
            episodes_already_seedvl += 1
            continue

        # Check if needs instruction generation
        if is_empty_or_gibberish_instruction(info['instruction']):
            episodes_to_process.append(h5_path)
        else:
            episodes_with_instruction += 1
    
    print(f"\nEpisode statistics:")
    print(f"  With original instruction: {episodes_with_instruction}")
    print(f"  Already processed (seedvl): {episodes_already_seedvl}")
    print(f"  Need instruction generation: {len(episodes_to_process)}")
    
    if not episodes_to_process:
        print("\nNo episodes need instruction generation!")
        return
    
    # Apply start_idx and batch_size
    episodes_to_process = episodes_to_process[args.start_idx:]
    if args.batch_size > 0:
        episodes_to_process = episodes_to_process[:args.batch_size]
    
    print(f"\nProcessing {len(episodes_to_process)} episodes (start_idx={args.start_idx}, batch_size={args.batch_size})")
    print(f"Using {args.nproc} worker processes")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")
    
    # Collect existing instructions for style examples
    print("Collecting example instructions...")
    existing_instructions = collect_existing_instructions(h5_files)
    print(f"Found {len(existing_instructions)} example instructions")
    
    # Prepare arguments for workers
    worker_args = [(h5_path, args.dry_run) for h5_path in episodes_to_process]
    
    # Process with multiprocessing
    print(f"\nInitializing {args.nproc} worker processes with VLM client (provider: {args.provider})...")
    
    results = []
    success_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    with Pool(
        processes=args.nproc,
        initializer=init_worker,
        initargs=(args.provider, existing_instructions)
    ) as pool:
        # Use imap_unordered for better progress tracking
        with tqdm(total=len(worker_args), desc="Generating instructions") as pbar:
            for result in pool.imap_unordered(process_single_episode, worker_args):
                results.append(result)
                if result['success']:
                    success_count += 1
                else:
                    failed_count += 1
                    if result['error']:
                        tqdm.write(f"  {result['episode']}: {result['error']}")
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total time: {elapsed_time:.1f}s")
    print(f"  Average time per episode: {elapsed_time/len(worker_args):.2f}s")
    
    if args.dry_run:
        successful_results = [r for r in results if r['success']]
        if successful_results:
            print("\n\nGenerated instructions preview:")
            for r in successful_results[:20]:  # Show first 20
                print(f"  {r['episode']}: \"{r['instruction']}\"")
            if len(successful_results) > 20:
                print(f"  ... and {len(successful_results) - 20} more")
    
    # Show failed episodes
    failed_results = [r for r in results if not r['success']]
    if failed_results and len(failed_results) <= 20:
        print("\nFailed episodes:")
        for r in failed_results:
            print(f"  {r['episode']}: {r['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instructions for Bridge dataset episodes using SeedVL (multiprocessing)"
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Directory containing H5 files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Number of episodes to process (default: 100, 0 for all)"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start processing from this episode index (default: 0)"
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help=f"Number of parallel workers (default: 4, max: {cpu_count()})"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="doubao",
        choices=["doubao", "qwen"],
        help="VLM provider: doubao (SeedVL) or qwen (default: doubao)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview generated instructions without saving"
    )
    
    args = parser.parse_args()
    
    # Validate nproc
    if args.nproc < 1:
        args.nproc = 1
    elif args.nproc > cpu_count():
        print(f"Warning: nproc ({args.nproc}) > available CPUs ({cpu_count()}), using {cpu_count()}")
        args.nproc = cpu_count()
    
    main(args)
