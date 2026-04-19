"""
Convert Bridge dataset from TF record format to H5 format.

Usage:
    python bridge_tfrecord_to_h5.py \
        --input_dir <PATH TO TF RECORD DATASET DIR> \
        --output_dir <PATH TO OUTPUT H5 DIR> \
        --nproc <NUMBER OF PROCESSES>
    python datasets/tools/bridge_tfrecord_to_h5.py --input_dir /VLA-Data/scripts/lianqing/data/openX/convert/bridge/1.0.0 \
         --output_dir /root/data/openX/x-vla/bridge_wrist --dataset_name bridge --nproc 8

Structure:
    - Each episode is saved as one H5 file
    - Format: episode_<id>.hdf5
    - Contents (only valid/non-black cameras are saved):
        - observation/image_0: [T, H, W, 3] uint8 (main over-shoulder camera)
        - observation/image_1: [T, H, W, 3] uint8 (randomized view 1, if valid)
        - observation/image_2: [T, H, W, 3] uint8 (randomized view 2, if valid)
        - observation/image_3: [T, H, W, 3] uint8 (wrist-mounted camera, if valid)
        - instruction: str (attribute)
        - instruction_source: str (attribute, "original" or "seedvl")
        - wrist_view_valid: bool (attribute, True if image_3 is valid wrist view)
        - tfrecord_file_path: str (attribute, for ECoT mapping)
        - tfrecord_episode_id: int (attribute, for ECoT mapping)
        - proprio: [T, D] float32 (proprioceptive data)
        - action: [T, D] float32 (action data)
        
Notes:
    - Bridge actions are deltas: [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper]
    - Bridge state is absolute: [3x XYZ, 3x roll-pitch-yaw, 1x gripper]
    - Multithreading: Uses ThreadPoolExecutor for parallel H5 writing while maintaining
      consistent episode IDs (episode_000000, episode_000001, ...). Reading is sequential
      to preserve order, writing is parallel for speed.
    - Set --nproc to number of threads for faster processing (default: 8)
    - TFRecord metadata (file_path, episode_id) is stored for proper ECoT data mapping
    - Camera data validation: Black/invalid cameras (mean < 5, std < 5) are not saved
    - Wrist view validity: Episodes without instruction that have image_3 are considered
      valid wrist views. Episodes with instruction may have misattributed image_3 data.
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import robosuite.utils.transform_utils as T
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from mmengine.utils import track_progress_rich

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




def is_valid_camera_data(images):
    """
    Check if camera data is valid (not all black).
    
    Args:
        images: Array of images [T, H, W, 3]
        
    Returns:
        True if camera data is valid (not black), False otherwise
    """
    if images is None or len(images) == 0:
        return False
    
    # Check first frame - if it's all black, likely the whole sequence is invalid
    first_frame = images[0]
    mean_val = first_frame.mean()
    std_val = first_frame.std()
    
    # Consider invalid if mean is very low (< 5) and std is very low (< 5)
    # This catches completely black frames
    is_invalid = mean_val < 5 and std_val < 5
    
    return not is_invalid


def is_empty_or_gibberish_instruction(instruction):
    """
    Check if instruction is empty or likely gibberish.
    
    BridgeData V2 has some episodes with empty or random character instructions
    that indicate the episode was collected without proper annotation.
    Episodes without meaningful instruction but with image_3 are likely valid wrist views.
    
    Args:
        instruction: Instruction string
        
    Returns:
        True if instruction is empty or gibberish, False otherwise
    """
    if instruction is None:
        return True
    
    instruction = str(instruction).strip()
    
    # Empty instruction
    if len(instruction) == 0:
        return True
    
    # Very short instruction (likely gibberish)
    if len(instruction) < 3:
        return True
    
    # Check for random character patterns (no spaces, no common words)
    # Gibberish often has no spaces and unusual character distributions
    words = instruction.split()
    if len(words) == 1 and len(instruction) > 10:
        # Single long "word" with no spaces is likely gibberish
        # Check if it has unusual character distribution
        alpha_chars = sum(1 for c in instruction if c.isalpha())
        if alpha_chars / len(instruction) < 0.5:
            return True
    
    return False


def is_valid_wrist_view(images, instruction):
    """
    Check if image_3 is a valid wrist view.
    
    Based on BridgeData V2 characteristics:
    - Episodes WITHOUT meaningful instruction that have image_3 are likely valid wrist views
    - Episodes WITH instruction that have image_3 may have misattributed camera data
    
    Args:
        images: Array of images for image_3 [T, H, W, 3]
        instruction: Instruction string
        
    Returns:
        True if image_3 is considered a valid wrist view, False otherwise
    """
    # First check if camera data itself is valid (not black)
    if images is None or not is_valid_camera_data(images):
        return False
    
    # Valid wrist view = non-black data AND no meaningful instruction
    return is_empty_or_gibberish_instruction(instruction)


def parse_tfrecord_episode(episode_data):
    """
    Parse a single episode from TF record format.
    Only keeps valid (non-black) camera data.
    
    Args:
        episode_data: Episode data from TFDS dataset
        
    Returns:
        images_dict: Dict mapping camera names to image arrays [T, H, W, 3]
                    Only includes valid (non-black) cameras
        instruction: String instruction
        proprio: Array of proprioceptive data [T, D]
        action: Array of actions [T, D]
        tfrecord_file_path: Original file path from TFRecord metadata
        tfrecord_episode_id: Episode ID from TFRecord metadata
    """
    steps = episode_data['steps']
    
    # Extract episode metadata for proper ECoT mapping
    tfrecord_file_path = episode_data['episode_metadata']['file_path'].numpy()
    if isinstance(tfrecord_file_path, bytes):
        tfrecord_file_path = tfrecord_file_path.decode('utf-8')
    tfrecord_episode_id = int(episode_data['episode_metadata']['episode_id'].numpy())
    
    # Initialize lists for all 4 cameras
    images_dict = {
        'image_0': [],  # Main over-shoulder camera
        'image_1': [],  # Randomized view 1
        'image_2': [],  # Randomized view 2
        'image_3': [],  # Wrist-mounted camera
    }
    instruction = None
    proprio = []
    action = []
    
    # Convert steps dataset to numpy
    for step in tfds.as_numpy(steps):
        # Get all 4 camera images
        for cam_name in ['image_0', 'image_1', 'image_2', 'image_3']:
            if cam_name in step['observation']:
                images_dict[cam_name].append(step['observation'][cam_name])
        
        # Get state and action
        proprio.append(step['observation']['state'])
        action.append(step['action'])
        
        # Get instruction (same for all steps in episode)
        if instruction is None:
            instruction = step['language_instruction'].decode('utf-8') if isinstance(
                step['language_instruction'], bytes
            ) else step['language_instruction']
    
    # Stack images for each camera and validate
    valid_images_dict = {}
    for cam_name in images_dict:
        if images_dict[cam_name]:  # Only stack if data exists
            stacked_images = np.stack(images_dict[cam_name], axis=0)
            # Only keep valid (non-black) camera data
            if is_valid_camera_data(stacked_images):
                valid_images_dict[cam_name] = stacked_images
    
    proprio = np.stack(proprio, axis=0)
    action = np.stack(action, axis=0)
    return valid_images_dict, instruction, proprio, action, tfrecord_file_path, tfrecord_episode_id


def save_episode_to_h5(output_path, images_dict, instruction, proprio, action,
                       tfrecord_file_path=None, tfrecord_episode_id=None,
                       wrist_view_valid=False, instruction_source="original"):
    """
    Save a single episode to H5 file with valid camera views only.
    
    Args:
        output_path: Path to output H5 file
        images_dict: Dict mapping camera names to [T, H, W, 3] arrays
                    Only includes valid (non-black) cameras
        instruction: String instruction
        proprio: [T, D] array of proprioceptive data
        action: [T, D] array of actions
        tfrecord_file_path: Original file path from TFRecord metadata (for ECoT mapping)
        tfrecord_episode_id: Episode ID from TFRecord metadata (for ECoT mapping)
        wrist_view_valid: Whether image_3 is a valid wrist view (bool)
        instruction_source: Source of instruction - "original" or "seedvl"
    
    Note:
        Only valid (non-black) camera data is saved. Black/invalid cameras are skipped.
    """
    with h5py.File(output_path, 'w') as f:
        # Create observation group
        obs_grp = f.create_group('observation')
        
        # Save only valid camera images (images_dict already filtered)
        for cam_name in ['image_0', 'image_1', 'image_2', 'image_3']:
            if cam_name in images_dict and images_dict[cam_name] is not None:
                obs_grp.create_dataset(
                    cam_name,
                    data=images_dict[cam_name],
                    compression='gzip',
                    compression_opts=4
                )
        
        # Save instruction as attribute
        f.attrs['instruction'] = instruction if instruction else ""
        
        # Save instruction source (original from dataset or generated by seedvl)
        f.attrs['instruction_source'] = instruction_source
        
        # Save wrist view validity flag
        f.attrs['wrist_view_valid'] = wrist_view_valid
        
        # Store TFRecord metadata for proper ECoT mapping
        if tfrecord_file_path is not None:
            f.attrs['tfrecord_file_path'] = tfrecord_file_path
        if tfrecord_episode_id is not None:
            f.attrs['tfrecord_episode_id'] = tfrecord_episode_id
        
        # Save proprio and action
        f.create_dataset(
            'proprio',
            data=proprio,
            dtype=np.float32
        )
        f.create_dataset(
            'action',
            data=action,
            dtype=np.float32
        )


def process_single_episode(episode_tuple):
    """
    Worker function to process a single episode in parallel.
    
    Args:
        episode_tuple: Tuple containing (episode_data, episode_idx, output_dir)
        
    Returns:
        episode_idx: Index of the processed episode
    """
    episode_data, episode_idx, output_dir = episode_tuple
    
    # Parse episode data (now includes all 4 cameras and TFRecord metadata)
    images_dict, instruction, proprio, action, tfrecord_file_path, tfrecord_episode_id = parse_tfrecord_episode(episode_data)
    
    # Save to H5 file with all cameras and metadata for ECoT mapping
    output_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.hdf5")
    save_episode_to_h5(output_path, images_dict, instruction, proprio, action,
                       tfrecord_file_path, tfrecord_episode_id)
    
    return episode_idx


def process_and_save_episode(args):
    """
    Worker function to process and save a single episode.
    Called by thread pool.
    
    Args:
        args: Tuple of (episode_idx, episode_data_parsed, output_dir)
              episode_data_parsed is a dict with all parsed data
              
    Returns:
        Dict with statistics for this episode
    """
    episode_idx, parsed_data, output_dir = args
    
    images_dict = parsed_data['images_dict']
    instruction = parsed_data['instruction']
    proprio = parsed_data['proprio']
    action = parsed_data['action']
    tfrecord_file_path = parsed_data['tfrecord_file_path']
    tfrecord_episode_id = parsed_data['tfrecord_episode_id']
    
    # Calculate statistics
    stats = {
        'camera_stats': {cam: 1 for cam in images_dict},
        'has_all_cameras': len(images_dict) == 4,
        'no_instruction': is_empty_or_gibberish_instruction(instruction),
        'valid_wrist': False
    }
    
    # Determine wrist view validity
    wrist_view_valid = False
    if 'image_3' in images_dict:
        wrist_view_valid = is_valid_wrist_view(images_dict['image_3'], instruction)
        stats['valid_wrist'] = wrist_view_valid
    
    # Save to H5 file
    output_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.hdf5")
    save_episode_to_h5(
        output_path, images_dict, instruction, proprio, action,
        tfrecord_file_path, tfrecord_episode_id,
        wrist_view_valid=wrist_view_valid,
        instruction_source="original"
    )
    
    return stats


def convert_bridge_to_h5(input_dir, output_dir, dataset_name='bridge_dataset', split='train', nproc=1, builder_dir=None, dummy=False):
    """
    Convert Bridge dataset from TF records to H5 format with all 4 cameras.
    
    Uses multithreading for parallel H5 file writing while maintaining consistent episode IDs.
    
    Args:
        input_dir: Directory containing TF record dataset (for standard TFDS)
        output_dir: Directory to save H5 files
        dataset_name: Name of the TFDS dataset (ignored if builder_dir is provided)
        split: Which split to convert (e.g., 'train', 'val')
        nproc: Number of threads for parallel processing (default: 1)
        builder_dir: Path to RLDS builder directory (for custom datasets like Bridge)
        dummy: If True, only convert first 100 episodes for testing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset - use builder_from_directory for custom RLDS datasets
    if builder_dir:
        print(f"Loading custom RLDS dataset from builder_dir: {builder_dir}")
        builder = tfds.builder_from_directory(builder_dir=builder_dir)
    else:
        print(f"Loading dataset from {input_dir}...")
        print(f"Dataset name: {dataset_name}")
        builder = tfds.builder(dataset_name, data_dir=input_dir)
    
    ds = builder.as_dataset(split=split)
    
    # Get total number of episodes
    if hasattr(builder, 'info'):
        total_episodes = builder.info.splits[split].num_examples
    else:
        total_episodes = None
    
    print(f"Converting {split} split...")
    if total_episodes:
        print(f"Total episodes: {total_episodes}")
    print(f"Using {nproc} thread(s) for parallel H5 writing")
    
    # Statistics tracking
    processed_count = 0
    camera_stats = {'image_0': 0, 'image_1': 0, 'image_2': 0, 'image_3': 0}
    episodes_with_all_cameras = 0
    episodes_with_valid_wrist = 0
    episodes_without_instruction = 0
    
    # Use ThreadPoolExecutor for parallel H5 writing
    # Episode reading is sequential to maintain order, but saving is parallel
    if nproc > 1:
        print(f"Processing episodes with {nproc} parallel threads...")
        
        # Batch processing with thread pool
        batch_size = nproc * 4  # Process in batches for better throughput
        batch = []
        
        pbar = tqdm(total=total_episodes, desc="Processing episodes")
        
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            for episode_idx, episode_data in enumerate(ds):
                try:
                    # Parse episode data (sequential - maintains order)
                    images_dict, instruction, proprio, action, tfrecord_file_path, tfrecord_episode_id = parse_tfrecord_episode(episode_data)
                    
                    # Prepare parsed data for worker
                    parsed_data = {
                        'images_dict': images_dict,
                        'instruction': instruction,
                        'proprio': proprio,
                        'action': action,
                        'tfrecord_file_path': tfrecord_file_path,
                        'tfrecord_episode_id': tfrecord_episode_id
                    }
                    
                    batch.append((episode_idx, parsed_data, output_dir))
                    
                    # Process batch when full or at end
                    if len(batch) >= batch_size or (dummy and episode_idx >= 100):
                        # Submit all tasks and wait for completion
                        futures = [executor.submit(process_and_save_episode, args) for args in batch]
                        
                        for future in as_completed(futures):
                            try:
                                stats = future.result()
                                
                                # Aggregate statistics
                                for cam_name, count in stats['camera_stats'].items():
                                    camera_stats[cam_name] += count
                                if stats['has_all_cameras']:
                                    episodes_with_all_cameras += 1
                                if stats['no_instruction']:
                                    episodes_without_instruction += 1
                                if stats['valid_wrist']:
                                    episodes_with_valid_wrist += 1
                                
                                processed_count += 1
                                pbar.update(1)
                                
                            except Exception as e:
                                print(f"\n⚠️  Error in worker: {e}")
                        
                        batch = []
                    
                    if dummy and episode_idx >= 100:
                        break
                        
                except Exception as e:
                    print(f"\n⚠️  Error parsing episode {episode_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)
                    continue
            
            # Process remaining batch
            if batch:
                futures = [executor.submit(process_and_save_episode, args) for args in batch]
                for future in as_completed(futures):
                    try:
                        stats = future.result()
                        for cam_name, count in stats['camera_stats'].items():
                            camera_stats[cam_name] += count
                        if stats['has_all_cameras']:
                            episodes_with_all_cameras += 1
                        if stats['no_instruction']:
                            episodes_without_instruction += 1
                        if stats['valid_wrist']:
                            episodes_with_valid_wrist += 1
                        processed_count += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"\n⚠️  Error in worker: {e}")
        
        pbar.close()
    
    else:
        # Single-threaded processing (original logic)
        for episode_idx, episode_data in enumerate(tqdm(ds, total=total_episodes, desc="Processing episodes")):
            try:
                # Parse episode data (only valid cameras)
                images_dict, instruction, proprio, action, tfrecord_file_path, tfrecord_episode_id = parse_tfrecord_episode(episode_data)
                
                # Track camera statistics
                for cam_name in images_dict:
                    camera_stats[cam_name] += 1
                
                if len(images_dict) == 4:
                    episodes_with_all_cameras += 1
                
                # Track instruction statistics
                if is_empty_or_gibberish_instruction(instruction):
                    episodes_without_instruction += 1
                
                # Determine wrist view validity
                wrist_view_valid = False
                if 'image_3' in images_dict:
                    wrist_view_valid = is_valid_wrist_view(images_dict['image_3'], instruction)
                    if wrist_view_valid:
                        episodes_with_valid_wrist += 1
                
                # Save to H5 file with valid cameras and metadata
                output_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.hdf5")
                save_episode_to_h5(
                    output_path, images_dict, instruction, proprio, action,
                    tfrecord_file_path, tfrecord_episode_id,
                    wrist_view_valid=wrist_view_valid,
                    instruction_source="original"
                )
                
                processed_count += 1
                
                if dummy and episode_idx >= 100:
                    break
                    
            except Exception as e:
                print(f"\n⚠️  Error processing episode {episode_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nConversion complete!")
    print(f"Saved {processed_count} episodes to {output_dir}")
    print(f"\nCamera availability statistics:")
    for cam_name, count in camera_stats.items():
        percentage = count / processed_count * 100 if processed_count > 0 else 0
        print(f"  {cam_name}: {count}/{processed_count} ({percentage:.1f}%)")
    print(f"\nEpisodes with all 4 valid cameras: {episodes_with_all_cameras}/{processed_count} ({episodes_with_all_cameras/processed_count*100:.1f}%)")
    print(f"\nWrist view statistics:")
    print(f"  Episodes without instruction: {episodes_without_instruction}/{processed_count} ({episodes_without_instruction/processed_count*100:.1f}%)")
    print(f"  Episodes with valid wrist view: {episodes_with_valid_wrist}/{processed_count} ({episodes_with_valid_wrist/processed_count*100:.1f}%)")


def verify_h5_file(h5_path):
    """
    Verify the structure of an H5 file and show available cameras.
    
    Args:
        h5_path: Path to H5 file
    """
    print(f"\nVerifying {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        
        # Show instruction and metadata
        print(f"\nMetadata:")
        instruction = f.attrs.get('instruction', '')
        print(f"  Instruction: '{instruction}'")
        print(f"  Instruction source: {f.attrs.get('instruction_source', 'unknown')}")
        print(f"  Wrist view valid: {f.attrs.get('wrist_view_valid', False)}")
        
        # Show TFRecord metadata for ECoT mapping
        if 'tfrecord_file_path' in f.attrs:
            print(f"  TFRecord file_path: {f.attrs['tfrecord_file_path']}")
        if 'tfrecord_episode_id' in f.attrs:
            print(f"  TFRecord episode_id: {f.attrs['tfrecord_episode_id']}")
        
        if 'observation' in f:
            available_cameras = list(f['observation'].keys())
            print(f"\nObservation keys: {available_cameras}")
            
            # Show all camera shapes
            camera_info = {
                'image_0': 'Main over-shoulder camera',
                'image_1': 'Randomized view 1',
                'image_2': 'Randomized view 2',
                'image_3': 'Wrist-mounted camera',
            }
            
            print("\nCamera views (only valid/non-black cameras saved):")
            for cam_name, description in camera_info.items():
                if cam_name in f['observation']:
                    shape = f['observation'][cam_name].shape
                    wrist_note = ""
                    if cam_name == 'image_3':
                        wrist_valid = f.attrs.get('wrist_view_valid', False)
                        wrist_note = f" [wrist_view_valid={wrist_valid}]"
                    print(f"  ✓ {cam_name} ({description}): {shape}{wrist_note}")
                else:
                    print(f"  ✗ {cam_name} ({description}): not available (invalid/black)")
        
        if 'action' in f:
            print(f"\nAction shape: {f['action'].shape}")
            print(f"Action sample: {f['action'][0]}")


def main(args):
    """Main conversion function."""
    print("=" * 80)
    print("Bridge Dataset TF Record to H5 Converter (All 4 Cameras)")
    print("=" * 80)
    
    # Convert dataset
    convert_bridge_to_h5(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        nproc=args.nproc,
        builder_dir=args.builder_dir,
        dummy=args.dummy
    )
    
    # Verify first file
    first_file = os.path.join(args.output_dir, "episode_000000.hdf5")
    if os.path.exists(first_file):
        verify_h5_file(first_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Bridge dataset from TF records to H5 format (all 4 cameras)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/VLA-Data/scripts/lianqing/data/openX",
        help="Directory containing TF record dataset (TFDS data_dir)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save H5 files"
    )
    parser.add_argument(
        "--builder_dir",
        type=str,
        default="/VLA-Data/scripts/lianqing/data/openX/convert/bridge/1.0.0",
        help="Path to RLDS builder directory (for Bridge dataset)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bridge_dataset",
        help="Name of the TFDS dataset (ignored if builder_dir is provided)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to convert (default: train)"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Only convert first 100 episodes for testing"
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=8,
        help="Number of threads for parallel H5 writing (default: 8). Episode IDs remain consistent."
    )
    
    args = parser.parse_args()
    main(args)

    # Example usage:
    # python bridge_tfrecord_to_h5.py \
    #     --builder_dir "/VLA-Data/scripts/lianqing/data/openX/convert/bridge/1.0.0" \
    #     --output_dir "/root/data/openX/x-vla/bridge_4cam" \
    #     --nproc 8