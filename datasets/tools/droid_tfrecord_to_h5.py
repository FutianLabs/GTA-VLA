"""
Convert DROID RLDS TFRecord to H5 format.

H5 layout:
    /observation/exterior_image_1_left  [T,]  JPEG bytes (may be ext2 if ext1 has no calib)
    /observation/wrist_image_left       [T,]  JPEG bytes
    /proprio                            [T, 6]  observation/cartesian_position (xyz + euler)
    /action                             [T, 7]  action (xyz + euler + gripper)
    /gripper_position                   [T, 2]  projected 2D pixel on exterior view (if calibrated)
    attrs:
        language_instruction      - primary instruction (original or KarlP supplement)
        language_instruction_1/2/3 - KarlP annotation variants (for data augmentation)
        calib_source              - 'direct' or 'propagated'
        ext2_as_ext1              - True if exterior_image_2 was stored as ext1
        file_path, gripper_2d_valid, ...

Calibration logic:
    - ext1 calib valid → use ext1 image + ext1 calib, drop ext2
    - ext1 invalid, ext2 valid → swap ext2 image into ext1 key, use ext2 calib, flag ext2_as_ext1
    - neither valid + no language → skip episode entirely

Language priority: KarlP annotations > original language_instruction.
    droid_language_annotations.json is auto-loaded from the same dir as --calib.

Usage:
    python datasets/tools/droid_tfrecord_to_h5.py \
        --input_dir /VLA-Data/scripts/lianqing/data/openX/convert \
        --output_dir /root/data/openX/x-vla/droid \
        --calib /root/data/KarlP/droid/cam2base_all.json \
        --n_streams 16 --jpeg_workers 10

    # Dry run (first 100 episodes)
    python datasets/tools/droid_tfrecord_to_h5.py \
        --input_dir /VLA-Data/scripts/lianqing/data/openX/convert \
        --output_dir /root/data/openX/x-vla/droid --dummy \
        --calib /root/data/KarlP/droid/cam2base_all.json
"""

import argparse
import json
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from turbojpeg import TurboJPEG
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
from queue import Empty

IMAGE_KEYS = ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']
JPEG_QUALITY = 100

_tj = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so')

def _encode_one(img_arr):
    return np.frombuffer(_tj.encode(img_arr, quality=JPEG_QUALITY), dtype=np.uint8)


_jpeg_pool = None

def _get_jpeg_pool(n_workers=8):
    global _jpeg_pool
    if _jpeg_pool is None or _jpeg_pool._max_workers != n_workers:
        _jpeg_pool = ThreadPoolExecutor(max_workers=n_workers)
    return _jpeg_pool


def _encode_images_jpeg(images, n_workers=8):
    pool = _get_jpeg_pool(n_workers)
    return list(pool.map(lambda i: _encode_one(images[i]), range(len(images))))


# ---------- calibration / projection ----------

def project_3d_to_2d(positions_3d, cam2base_6d, intrinsics_4, orig_w, orig_h, img_w, img_h):
    """Project (T, 3) base-frame positions to (T, 2) pixel coordinates."""
    cam2base = np.eye(4)
    cam2base[:3, :3] = Rotation.from_euler("xyz", cam2base_6d[3:6]).as_matrix()
    cam2base[:3, 3] = cam2base_6d[:3]
    base2cam = np.linalg.inv(cam2base)

    fx, cx, fy, cy = intrinsics_4
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    T = positions_3d.shape[0]
    pts_cam = (base2cam @ np.hstack([positions_3d, np.ones((T, 1))]).T).T[:, :3]
    pts_proj = (K @ pts_cam.T).T

    pixels = np.full((T, 2), np.nan, dtype=np.float32)
    valid = pts_proj[:, 2] > 0
    pixels[valid, 0] = pts_proj[valid, 0] / pts_proj[valid, 2] * (img_w / orig_w)
    pixels[valid, 1] = pts_proj[valid, 1] / pts_proj[valid, 2] * (img_h / orig_h)
    return pixels


def _resolve_episode_id(file_path, path_to_id):
    """Extract episode_id from RLDS file_path via path_to_id mapping."""
    marker = 'r2d2-data-full/'
    idx = file_path.find(marker)
    if idx < 0:
        return None
    rel = file_path[idx + len(marker):]
    if rel.endswith('/trajectory.h5'):
        rel = rel[:-len('/trajectory.h5')]
    return path_to_id.get(rel)


def resolve_calib(file_path, calib_unified, path_to_id):
    """Look up calibration for both exterior views.

    Returns (ext1_calib, ext2_calib) where each is a dict or None.
    """
    episode_id = _resolve_episode_id(file_path, path_to_id)
    if episode_id is None:
        return None, None
    entry = calib_unified.get(episode_id)
    if entry is None:
        return None, None

    serial_to_imgkey = entry.get('serial_to_imgkey', {})
    cameras = entry.get('cameras', {})
    intrinsics = entry.get('intrinsics', {})

    common = {
        'episode_id': episode_id,
        'source': entry.get('source', ''),
        'group_stats': entry.get('group_stats', {}),
    }

    ext1_calib = None
    ext2_calib = None
    for serial, extr_6d in cameras.items():
        imgkey = serial_to_imgkey.get(serial)
        if serial not in intrinsics:
            continue
        result = {**common, 'extrinsics': extr_6d, 'intrinsics': intrinsics[serial]}
        if imgkey == 'exterior_image_1_left':
            ext1_calib = result
        elif imgkey == 'exterior_image_2_left':
            ext2_calib = result

    return ext1_calib, ext2_calib


# ---------- episode parsing ----------

def parse_tfrecord_episode(episode_data, tfds):
    steps = episode_data['steps']
    metadata = {}
    if 'episode_metadata' in episode_data:
        for k, v in episode_data['episode_metadata'].items():
            val = v.numpy()
            metadata[k] = val.decode('utf-8') if isinstance(val, bytes) else val

    obs_images = {k: [] for k in IMAGE_KEYS}
    proprio_list = []
    actions = []
    lang = ''

    for step in tfds.as_numpy(steps):
        obs = step['observation']
        for k in IMAGE_KEYS:
            if k in obs:
                obs_images[k].append(obs[k])
        if 'cartesian_position' in obs:
            proprio_list.append(obs['cartesian_position'])
        actions.append(step['action'])
        if not lang:
            raw = step.get('language_instruction', b'')
            lang = raw.decode('utf-8') if isinstance(raw, bytes) else raw

    obs_images = {k: np.stack(v, axis=0) for k, v in obs_images.items() if v}
    proprio = np.stack(proprio_list, axis=0).astype(np.float32)
    actions = np.stack(actions, axis=0).astype(np.float32)
    return obs_images, proprio, actions, lang, metadata


# ---------- H5 writing ----------

def save_episode_to_h5(output_path, obs_images_jpeg, proprio, actions,
                       metadata, gripper_2d_pixels, calib_attrs, lang_attrs):
    vlen_dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    with h5py.File(output_path, 'w') as f:
        obs_grp = f.create_group('observation')
        for k, jpeg_list in obs_images_jpeg.items():
            dset = obs_grp.create_dataset(k, (len(jpeg_list),), dtype=vlen_dt)
            for i, arr in enumerate(jpeg_list):
                dset[i] = arr

        f.create_dataset('proprio', data=proprio, dtype=np.float32)
        f.create_dataset('action', data=actions, dtype=np.float32)
        if gripper_2d_pixels is not None:
            f.create_dataset('gripper_position', data=gripper_2d_pixels, dtype=np.float32)

        for k, v in lang_attrs.items():
            f.attrs[k] = v
        for k, v in metadata.items():
            f.attrs[k] = v
        for k, v in calib_attrs.items():
            f.attrs[k] = v


# ---------- worker ----------

def _worker_main(worker_id, n_streams, builder_dir, split, output_dir,
                 jpeg_workers, done_queue, dummy, calib_path):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    calib_unified = {}
    path_to_id = {}
    karlp_lang = {}
    if calib_path:
        calib_dir = os.path.dirname(calib_path)
        with open(calib_path) as f:
            calib_unified = json.load(f)
        id_to_path_file = os.path.join(calib_dir, 'episode_id_to_path.json')
        with open(id_to_path_file) as f:
            id_to_path = json.load(f)
        path_to_id = {v: k for k, v in id_to_path.items()}
        karlp_lang_file = os.path.join(calib_dir, 'droid_language_annotations.json')
        if os.path.isfile(karlp_lang_file):
            with open(karlp_lang_file) as f:
                karlp_lang = json.load(f)

    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    ds = builder.as_dataset(split=split)

    total_ds = ds
    ds = total_ds.enumerate().filter(
        lambda i, _: tf.equal(i % n_streams, worker_id)
    ).prefetch(tf.data.AUTOTUNE)

    skipped = 0
    for elem in ds:
        episode_idx = elem[0].numpy()
        episode_data = elem[1]
        if dummy and episode_idx > 100:
            break
        try:
            obs_images, proprio, actions, lang, metadata = \
                parse_tfrecord_episode(episode_data, tfds)

            file_path = metadata.get('file_path', '')
            ext1_calib, ext2_calib = resolve_calib(file_path, calib_unified, path_to_id)

            # KarlP language annotations (3 variants per episode) — prefer KarlP over original
            ep_id = _resolve_episode_id(file_path, path_to_id)
            karlp_annot = karlp_lang.get(ep_id, {}) if ep_id else {}
            karlp_langs = [
                karlp_annot.get(f'language_instruction{i}', '').strip()
                for i in range(1, 4)
            ]
            effective_lang = karlp_langs[0] or lang.strip()

            # Determine calibration: prefer ext1, fallback to ext2
            calib = None
            use_ext2 = False
            if ext1_calib and 'exterior_image_1_left' in obs_images:
                calib = ext1_calib
            elif ext2_calib and 'exterior_image_2_left' in obs_images:
                calib = ext2_calib
                use_ext2 = True

            # Skip if no calibration AND no language
            if calib is None and not effective_lang:
                skipped += 1
                done_queue.put(1)
                continue

            # Image handling: swap ext2→ext1 if needed, always drop ext2
            if use_ext2:
                obs_images['exterior_image_1_left'] = obs_images.pop('exterior_image_2_left')
            else:
                obs_images.pop('exterior_image_2_left', None)

            # Project gripper 3D → 2D pixel on the chosen exterior view
            gripper_2d_pixels = None
            calib_attrs = {}
            if calib and 'exterior_image_1_left' in obs_images:
                intr = calib['intrinsics']
                img_h, img_w = obs_images['exterior_image_1_left'].shape[1:3]
                gripper_2d_pixels = project_3d_to_2d(
                    proprio[:, :3], calib['extrinsics'], intr['cameraMatrix'],
                    intr['width'], intr['height'], img_w, img_h)

                calib_attrs['calib_source'] = calib['source']
                calib_attrs['calib_episode_id'] = calib['episode_id']
                calib_attrs['gripper_2d_valid'] = calib['source'] == 'direct'
                calib_attrs['ext2_as_ext1'] = use_ext2
                calib_attrs['cam2base_exterior_image_1_left'] = json.dumps(calib['extrinsics'])
                calib_attrs['intrinsics_exterior_image_1_left'] = json.dumps(intr['cameraMatrix'])
                calib_attrs['orig_resolution_exterior_image_1_left'] = json.dumps([intr['width'], intr['height']])

                if calib['source'] == 'propagated' and calib['group_stats']:
                    max_std = max(gs.get('trans_std', 0.0) for gs in calib['group_stats'].values())
                    calib_attrs['calib_trans_std'] = round(max_std, 4)

            # Encode images to JPEG
            obs_images_jpeg = {k: _encode_images_jpeg(v, jpeg_workers) for k, v in obs_images.items()}

            # Build language attrs (primary + KarlP variants for augmentation)
            lang_attrs = {'language_instruction': effective_lang}
            for i, kl in enumerate(karlp_langs, 1):
                if kl:
                    lang_attrs[f'language_instruction_{i}'] = kl

            output_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.hdf5")
            save_episode_to_h5(output_path, obs_images_jpeg, proprio, actions,
                               metadata, gripper_2d_pixels, calib_attrs, lang_attrs)
            done_queue.put(1)
        except Exception as e:
            print(f"\nWorker-{worker_id} episode {episode_idx} error: {e}")
            done_queue.put(1)
    if skipped:
        print(f"Worker-{worker_id} skipped {skipped} episodes (no calib & no lang)")
    done_queue.put(None)


# ---------- main ----------

def _resolve_builder_dir(input_dir, dataset_name):
    for candidate in [
        input_dir,
        os.path.join(input_dir, dataset_name),
    ]:
        if os.path.isfile(os.path.join(candidate, "dataset_info.json")):
            return candidate
    version_dir = os.path.join(input_dir, dataset_name)
    if os.path.isdir(version_dir):
        for entry in sorted(os.listdir(version_dir), reverse=True):
            p = os.path.join(version_dir, entry)
            if os.path.isfile(os.path.join(p, "dataset_info.json")):
                return p
    raise FileNotFoundError(f"dataset_info.json not found under {input_dir}")


def _read_total_episodes(builder_dir, split):
    info_path = os.path.join(builder_dir, "dataset_info.json")
    if not os.path.isfile(info_path):
        return None
    with open(info_path) as f:
        info = json.load(f)
    for s in info.get("splits", []):
        if s.get("name") == split:
            n = (s.get("statistics") or {}).get("numExamples")
            if n:
                return n
            sl = s.get("shardLengths")
            return sum(int(x) for x in sl) if sl else None
    return None


def convert_droid_to_h5(input_dir, output_dir, dataset_name='droid', split='train',
                        n_streams=4, jpeg_workers=8, builder_dir=None, dummy=False,
                        calib_path=None):
    os.makedirs(output_dir, exist_ok=True)
    builder_dir = builder_dir or _resolve_builder_dir(input_dir, dataset_name)
    print(f"builder_dir: {builder_dir}")

    total_episodes = _read_total_episodes(builder_dir, split)
    print(f"split={split}, total={total_episodes}, n_streams={n_streams}, jpeg_workers={jpeg_workers}")
    if calib_path:
        print(f"calib: {calib_path}")
        karlp_lang_file = os.path.join(os.path.dirname(calib_path), 'droid_language_annotations.json')
        if os.path.isfile(karlp_lang_file):
            print(f"karlp_lang: {karlp_lang_file} (auto-detected)")

    done_queue = Queue()
    workers = []
    for i in range(n_streams):
        p = Process(target=_worker_main,
                    args=(i, n_streams, builder_dir, split, output_dir,
                          jpeg_workers, done_queue, dummy, calib_path),
                    daemon=True)
        p.start()
        workers.append(p)

    from tqdm import tqdm
    pbar = tqdm(total=total_episodes, desc="Processing")
    finished_workers = 0
    while finished_workers < n_streams:
        while True:
            try:
                val = done_queue.get_nowait()
                if val is None:
                    finished_workers += 1
                else:
                    pbar.update(1)
            except Empty:
                break
        if finished_workers < n_streams:
            time.sleep(0.2)
    pbar.close()

    for p in workers:
        p.join(timeout=10)
    print(f"\nDone! Saved to {output_dir}")


def verify_h5_file(h5_path):
    print(f"\nVerifying {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        def _print_tree(grp, prefix=''):
            for k in grp:
                item = grp[k]
                if isinstance(item, h5py.Group):
                    print(f"  {prefix}{k}/")
                    _print_tree(item, prefix + '  ')
                else:
                    print(f"  {prefix}{k}: shape={item.shape}, dtype={item.dtype}")
        _print_tree(f)

        if 'gripper_position' in f:
            gp = f['gripper_position'][:]
            valid = ~np.isnan(gp[:, 0])
            src = f.attrs.get('calib_source', '')
            std = f.attrs.get('calib_trans_std', None)
            print(f"  gripper_position: {valid.sum()}/{len(gp)} valid, "
                  f"calib={src}" + (f", std={std}m" if std else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default="/VLA-Data/scripts/lianqing/data/openX/convert")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="droid")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--n_streams", type=int, default=4)
    parser.add_argument("--jpeg_workers", type=int, default=8)
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to cam2base_all.json (auto-loads droid_language_annotations.json from same dir)")
    args = parser.parse_args()

    print("=" * 80)
    print("Droid TFRecord -> H5 Converter")
    print("=" * 80)

    convert_droid_to_h5(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        n_streams=args.n_streams,
        jpeg_workers=args.jpeg_workers,
        dummy=args.dummy,
        calib_path=args.calib,
    )

    first_file = os.path.join(args.output_dir, "episode_000000.hdf5")
    if os.path.exists(first_file):
        verify_h5_file(first_file)
