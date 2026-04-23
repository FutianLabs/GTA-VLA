"""
Convert Fractal (RT-1) dataset from TFRecord to H5 format.

Usage:
    python datasets/tools/fractal_tfrecord_to_h5.py \
        --input_dir ~/data/openX/fractal20220817_data \
        --output_dir ~/data/openX/gtavla/fractal \
        --nproc 8

    # Quick test with first 100 episodes:
    python datasets/tools/fractal_tfrecord_to_h5.py \
        --input_dir ~/data/openX/fractal20220817_data \
        --output_dir /tmp/fractal_test \
        --nproc 4 --dummy

Structure:
    episode_XXXXXX.hdf5
    ├── observation/
    │   └── image_0: [T, 256, 320, 3] uint8 (gzip compressed)
    ├── proprio: [T, 8] float32   base_pose_tool_reached(7) + gripper_closed(1)
    ├── action: [T, 10] float32   world_vector(3) + rotation_delta(3) + gripper(1)
    │                              + base_displacement_vector(2) + base_displacement_vertical_rotation(1)
    └── attrs:
        ├── instruction: str
        ├── success: bool
        ├── task_family_name: str
        └── env_name: str

Notes:
    - Fractal actions are deltas (displacements), proprio is absolute (pos + quat + gripper)
    - Steps after terminate_episode are excluded
    - natural_language_embedding (512 floats/step) is skipped
    - 87,212 training episodes, ~111 GiB raw → ~60-70 GiB with gzip
"""

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def _decode_str(v):
    return v.decode('utf-8') if isinstance(v, bytes) else str(v)


def parse_tfrecord_episode(episode_data):
    """Parse a single Fractal episode from TFRecord."""
    steps = episode_data['steps']

    # Episode-level metadata
    meta = {}
    aspects = episode_data.get('aspects', {})
    attributes = episode_data.get('attributes', {})
    if aspects:
        for k in ['success', 'feasible', 'already_success', 'undesirable']:
            if k in aspects:
                meta[k] = bool(aspects[k].numpy())
    if attributes:
        for k in ['task_family_name', 'env_name', 'collection_mode_name',
                   'data_type_name', 'objects_family_name', 'location_name']:
            if k in attributes:
                v = attributes[k].numpy()
                meta[k] = _decode_str(v) if isinstance(v, (bytes, np.bytes_)) else str(v)

    images = []
    proprio_list = []
    action_list = []
    instruction = None

    for step in tfds.as_numpy(steps):
        # Skip the terminal step (action is meaningless there)
        if step['is_terminal']:
            break

        obs = step['observation']
        act = step['action']

        images.append(obs['image'])  # (256, 320, 3)

        # proprio: base_pose_tool_reached(7) + gripper_closed(1)
        proprio_list.append(np.concatenate([
            obs['base_pose_tool_reached'],
            obs['gripper_closed'],
        ]))

        # action: world_vector(3) + rotation_delta(3) + gripper(1) + base_disp(2) + base_rot(1)
        action_list.append(np.concatenate([
            act['world_vector'],
            act['rotation_delta'],
            act['gripper_closedness_action'],
            act['base_displacement_vector'],
            act['base_displacement_vertical_rotation'],
        ]))

        if instruction is None:
            instruction = _decode_str(obs.get('natural_language_instruction', b''))

    if not images:
        return None

    return {
        'images': np.stack(images, axis=0),
        'proprio': np.stack(proprio_list, axis=0).astype(np.float32),
        'action': np.stack(action_list, axis=0).astype(np.float32),
        'instruction': instruction or '',
        'meta': meta,
    }


def save_episode_to_h5(output_path, parsed, compression_level=1):
    """Save a parsed episode to HDF5."""
    with h5py.File(output_path, 'w') as f:
        obs_grp = f.create_group('observation')
        if compression_level > 0:
            obs_grp.create_dataset('image_0', data=parsed['images'],
                                   compression='gzip', compression_opts=compression_level)
        else:
            obs_grp.create_dataset('image_0', data=parsed['images'])

        f.create_dataset('proprio', data=parsed['proprio'], dtype=np.float32)
        f.create_dataset('action', data=parsed['action'], dtype=np.float32)

        f.attrs['instruction'] = parsed['instruction']
        for k, v in parsed['meta'].items():
            f.attrs[k] = v


def _process_and_save(args_tuple):
    idx, parsed, output_dir, compression_level = args_tuple
    save_episode_to_h5(os.path.join(output_dir, f"episode_{idx:06d}.hdf5"),
                       parsed, compression_level)
    return idx


def _resolve_builder_dir(input_dir, dataset_name='fractal20220817_data'):
    if os.path.isfile(os.path.join(input_dir, "dataset_info.json")):
        return input_dir
    name_dir = os.path.join(input_dir, dataset_name)
    if os.path.isfile(os.path.join(name_dir, "dataset_info.json")):
        return name_dir
    # Check versioned sub-dirs
    for d in [input_dir, name_dir]:
        if os.path.isdir(d):
            for entry in sorted(os.listdir(d), reverse=True):
                candidate = os.path.join(d, entry)
                if os.path.isfile(os.path.join(candidate, "dataset_info.json")):
                    return candidate
    return None


def convert_fractal_to_h5(input_dir, output_dir, split='train', nproc=1,
                          dummy=False, compression_level=1):
    os.makedirs(output_dir, exist_ok=True)

    builder_dir = _resolve_builder_dir(input_dir)
    if not builder_dir:
        raise FileNotFoundError(f"dataset_info.json not found under {input_dir}")
    print(f"Loading from builder_dir: {builder_dir}")
    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    ds = builder.as_dataset(split=split)

    total = None
    if hasattr(builder, 'info'):
        total = builder.info.splits[split].num_examples

    print(f"Split={split}, total={total}, threads={nproc}, compression={compression_level}")

    processed = 0
    skipped = 0

    if nproc > 1:
        sem = threading.Semaphore(nproc)
        lock = threading.Lock()
        pbar = tqdm(total=total, desc="Converting")

        def _done(fut):
            nonlocal processed
            sem.release()
            try:
                fut.result()
            except Exception as e:
                print(f"\nError: {e}")
            with lock:
                processed += 1
                pbar.update(1)

        with ThreadPoolExecutor(max_workers=nproc) as executor:
            for idx, episode_data in enumerate(ds):
                if dummy and idx > 100:
                    break
                parsed = parse_tfrecord_episode(episode_data)
                if parsed is None:
                    skipped += 1
                    pbar.update(1)
                    continue
                sem.acquire()
                fut = executor.submit(_process_and_save,
                                      (idx, parsed, output_dir, compression_level))
                fut.add_done_callback(_done)

        pbar.close()
    else:
        for idx, episode_data in enumerate(tqdm(ds, total=total, desc="Converting")):
            if dummy and idx > 100:
                break
            parsed = parse_tfrecord_episode(episode_data)
            if parsed is None:
                skipped += 1
                continue
            save_episode_to_h5(
                os.path.join(output_dir, f"episode_{idx:06d}.hdf5"),
                parsed, compression_level)
            processed += 1

    print(f"\nDone! Saved {processed} episodes, skipped {skipped}")


def verify_h5_file(h5_path):
    print(f"\nVerifying {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        print(f"Attributes: {dict(f.attrs)}")
        def _tree(grp, prefix=''):
            for k in grp:
                item = grp[k]
                if isinstance(item, h5py.Group):
                    print(f"  {prefix}{k}/")
                    _tree(item, prefix + '  ')
                else:
                    print(f"  {prefix}{k}: shape={item.shape}, dtype={item.dtype}")
        _tree(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Fractal (RT-1) dataset from TFRecord to H5"
    )
    parser.add_argument("--input_dir", type=str,
                        default=os.path.expanduser("~/data/openX/fractal20220817_data"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dummy", action="store_true",
                        help="Only convert first 100 episodes")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--compression", type=int, default=1,
                        help="gzip compression level (0=off, 1-9)")
    args = parser.parse_args()

    print("=" * 70)
    print("Fractal (RT-1) TFRecord -> H5 Converter")
    print("=" * 70)

    convert_fractal_to_h5(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split=args.split,
        nproc=args.nproc,
        dummy=args.dummy,
        compression_level=args.compression,
    )

    first = os.path.join(args.output_dir, "episode_000000.hdf5")
    if os.path.exists(first):
        verify_h5_file(first)
