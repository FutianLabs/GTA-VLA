"""
Binarize ManiSkill bridge-format HDF5 gripper values.

Usage:
    python datasets/tools/binarize_maniskill_gripper.py \
        --src data/openX/gtavla/bridge_5k_group \
        --dst data/openX/gtavla/bridge_5k_group_bin \
        --threshold 0.7 \
        --nproc 16

    python datasets/tools/binarize_maniskill_gripper.py \
        --meta_path data/meta/debug_200_per_task_train_maniskill_cot_meta.json \
        --dst data/openX/gtavla/bridge_5k_group_bin \
        --threshold 0.7 \
        --nproc 16
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from tqdm import tqdm


def process_one(args):
    src_path, dst_path, threshold, overwrite = args
    if not os.path.exists(src_path):
        return "skip"
    if os.path.exists(dst_path) and not overwrite and os.path.abspath(src_path) != os.path.abspath(dst_path):
        return "exists"
    try:
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

        with h5py.File(dst_path, "r+") as f:
            if "action" not in f:
                return "no_action"
            action = f["action"][:]
            if action.ndim != 2 or action.shape[1] < 1:
                return "bad_action"

            raw = action[:, -1].astype(np.float32)
            binary = (raw >= threshold).astype(action.dtype, copy=False)
            f["action"][:, -1] = binary

            open_count = int(binary.sum())
            close_count = int(len(binary) - open_count)
            return f"ok:{open_count}:{close_count}:{float(raw.min()):.6f}:{float(raw.max()):.6f}"
    except Exception as e:
        return f"err:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Source dir containing episode_*.hdf5")
    parser.add_argument("--meta_path", help="Only process HDF5 files listed in meta['datalist']")
    parser.add_argument("--dst", required=True, help="Output dir for binarized HDF5")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--nproc", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dst = os.path.abspath(args.dst)
    if bool(args.src) == bool(args.meta_path):
        raise ValueError("Specify exactly one of --src or --meta_path")

    if args.meta_path:
        meta_path = os.path.abspath(args.meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        src_files = [p for p in meta.get("datalist", []) if str(p).endswith(".hdf5")]
        pairs = [
            (
                os.path.abspath(src_path),
                os.path.join(dst, os.path.basename(src_path)),
                args.threshold,
                args.overwrite,
            )
            for src_path in src_files
        ]
        print(f"Using meta: {meta_path}")
    else:
        src = os.path.abspath(args.src)
        files = sorted(f for f in os.listdir(src) if f.endswith(".hdf5"))
        pairs = [
            (
                os.path.join(src, name),
                os.path.join(dst, name),
                args.threshold,
                args.overwrite,
            )
            for name in files
        ]
        print(f"Using src dir: {src}")

    print(f"Binarizing {len(pairs)} files")
    print(f"  dst={dst}")
    print(f"  threshold={args.threshold}")
    print(f"  workers={args.nproc}")

    counts = {"ok": 0, "exists": 0, "skip": 0, "no_action": 0, "bad_action": 0, "err": 0}
    open_total = 0
    close_total = 0

    with ProcessPoolExecutor(max_workers=args.nproc) as pool:
        for result in tqdm(pool.map(process_one, pairs, chunksize=32), total=len(pairs)):
            if result.startswith("ok:"):
                counts["ok"] += 1
                _, open_count, close_count, _, _ = result.split(":")
                open_total += int(open_count)
                close_total += int(close_count)
            elif result in counts:
                counts[result] += 1
            else:
                counts["err"] += 1

    print(f"Done: {counts}")
    if counts["ok"] > 0:
        print(f"Binary gripper stats: open={open_total}, closed={close_total}")


if __name__ == "__main__":
    main()
