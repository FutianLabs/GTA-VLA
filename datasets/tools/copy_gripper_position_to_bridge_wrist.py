"""
Copy gripper_position and gripper_2d_valid from bridge/ to bridge_wrist/ HDF5 files.

bridge/ has gripper_position (from earlier conversion with camera calibration),
but bridge_wrist/ (used by pretrain) is missing it. Both have the same episode IDs.

Usage:
    python datasets/tools/copy_gripper_position_to_bridge_wrist.py \
        --src /VLA-Data/scripts/lianqing/data/openX/x-vla/bridge \
        --dst /VLA-Data/scripts/lianqing/data/openX/x-vla/bridge_wrist \
        --nproc 16
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import h5py
import numpy as np


def copy_one(args):
    src_path, dst_path = args
    if not os.path.exists(src_path) or not os.path.exists(dst_path):
        return "skip"
    try:
        with h5py.File(src_path, 'r') as fs:
            if 'gripper_position' not in fs:
                return "no_gp"
            gp = fs['gripper_position'][:]
            gp_valid = bool(fs.attrs.get('gripper_2d_valid', False))

        with h5py.File(dst_path, 'a') as fd:
            if 'gripper_position' in fd:
                return "exists"
            fd.create_dataset('gripper_position', data=gp)
            fd.attrs['gripper_2d_valid'] = gp_valid
        return "ok"
    except Exception as e:
        return f"err:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="bridge/ dir with gripper_position")
    parser.add_argument("--dst", required=True, help="bridge_wrist/ dir to patch")
    parser.add_argument("--nproc", type=int, default=16)
    args = parser.parse_args()

    dst_files = sorted(f for f in os.listdir(args.dst) if f.endswith('.hdf5'))
    pairs = [(os.path.join(args.src, f), os.path.join(args.dst, f)) for f in dst_files]
    print(f"Copying gripper_position: {len(pairs)} episodes, {args.nproc} workers")

    counts = {"ok": 0, "exists": 0, "no_gp": 0, "skip": 0, "err": 0}
    with ProcessPoolExecutor(max_workers=args.nproc) as pool:
        for result in tqdm(pool.map(copy_one, pairs, chunksize=64), total=len(pairs)):
            key = result if result in counts else "err"
            counts[key] += 1

    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
