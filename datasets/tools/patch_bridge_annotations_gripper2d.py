"""
Patch bridge CoT annotations to add gripper_2d (pick point) from gripper_position.

The annotation pipeline was originally run on bridge_wrist/ files which lacked
gripper_position, so gripper_key_status.gripper_2d was left empty. This script
reads gripper_position from bridge/ (or bridge_wrist/ after copying) and patches
the annotation JSONs.

Usage:
    # Step 1: copy gripper_position to bridge_wrist/ first
    python datasets/tools/copy_gripper_position_to_bridge_wrist.py \
        --src data/openX/gtavla/bridge \
        --dst data/openX/gtavla/bridge_wrist

    # Step 2: patch annotations
    python datasets/tools/patch_bridge_annotations_gripper2d.py \
        --annotation_dir data/cot_annotations/bridge_annotations_wrist \
        --h5_dir data/openX/gtavla/bridge \
        --nproc 16
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import h5py
import numpy as np


def patch_one(args):
    ann_path, h5_dir = args
    try:
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        ep_id = os.path.splitext(os.path.basename(ann_path))[0]
        h5_path = os.path.join(h5_dir, f"{ep_id}.hdf5")
        if not os.path.exists(h5_path):
            return "no_h5"

        with h5py.File(h5_path, 'r') as fh:
            if 'gripper_position' not in fh:
                return "no_gp"
            gp = fh['gripper_position'][:]

        modified = False
        for st in ann.get('subtasks', []):
            gks = st.get('gripper_key_status')
            if not gks:
                continue
            if gks.get('gripper_2d') is not None:
                continue
            frame_idx = gks.get('frame_idx')
            if frame_idx is not None and 0 <= frame_idx < len(gp):
                coord = gp[frame_idx]
                x, y = float(coord[0]), float(coord[1])
                if not (np.isnan(x) or np.isnan(y)):
                    gks['gripper_2d'] = [x, y]
                    modified = True

        if modified:
            with open(ann_path, 'w') as f:
                json.dump(ann, f, indent=2, ensure_ascii=False)
            return "patched"
        return "unchanged"
    except Exception as e:
        return f"err:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--h5_dir", required=True,
                        help="bridge/ dir with gripper_position (or bridge_wrist/ after copy)")
    parser.add_argument("--nproc", type=int, default=16)
    args = parser.parse_args()

    ann_files = sorted(f for f in os.listdir(args.annotation_dir) if f.endswith('.json'))
    pairs = [(os.path.join(args.annotation_dir, f), args.h5_dir) for f in ann_files]
    print(f"Patching {len(pairs)} annotations with gripper_2d from {args.h5_dir}")

    counts = {"patched": 0, "unchanged": 0, "no_h5": 0, "no_gp": 0, "err": 0}
    with ProcessPoolExecutor(max_workers=args.nproc) as pool:
        for result in tqdm(pool.map(patch_one, pairs, chunksize=64), total=len(pairs)):
            key = result if result in counts else "err"
            counts[key] += 1

    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
