#!/usr/bin/env python3
"""
Generate filtered RoboMIND CoT meta files, excluding episodes
whose puppet/end_effector 3D trajectory is essentially static.
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gen_cot_meta import CONFIGS, OUTPUT_DIR


def _check_one(h5_path: str, proprio_key: str, min_travel: float) -> bool:
    try:
        with h5py.File(h5_path, 'r') as f:
            if proprio_key not in f:
                return False
            ee = f[proprio_key][:]
            if ee.shape[0] < 2 or ee.shape[1] < 3:
                return False
            travel = float(np.sum(np.linalg.norm(np.diff(ee[:, :3], axis=0), axis=1)))
            return travel >= min_travel
    except Exception:
        return False


def generate_filtered(name: str, min_travel: float = 0.01, workers: int = 32):
    if name not in CONFIGS:
        print(f"Unknown dataset: {name}")
        return
    cfg = CONFIGS[name]
    proprio_key = cfg["dataset_config"].get("proprio_key", "puppet/end_effector")

    source = cfg["source_meta"]
    print(f"Reading source meta: {source}")
    with open(source, 'r') as f:
        src = json.load(f)

    datalist = src.get("datalist", [])
    total = len(datalist)
    h5_paths = [item if isinstance(item, str) else item.get("path", "") for item in datalist]

    print(f"Checking {total} episodes with {workers} workers (proprio_key={proprio_key}, min_travel={min_travel}) ...")
    fn = partial(_check_one, proprio_key=proprio_key, min_travel=min_travel)
    with Pool(workers) as pool:
        results = pool.map(fn, h5_paths, chunksize=200)

    filtered = [item for item, ok in zip(datalist, results) if ok]
    skipped = total - len(filtered)

    meta = {
        "dataset_name": cfg["dataset_name"],
        "observation_key": cfg["observation_key"],
        "language_instruction_key": cfg["language_instruction_key"],
        "annotation_dir": cfg["annotation_dir"],
        "dataset_config": cfg["dataset_config"],
        "cot_config": cfg["cot_config"],
        "datalist": filtered,
    }

    out_path = OUTPUT_DIR / cfg["output_name"]
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}  (kept {len(filtered)}/{total}, removed {skipped})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=["robomind-franka", "robomind-ur"])
    parser.add_argument("--min_travel", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        generate_filtered(name, args.min_travel, args.workers)
