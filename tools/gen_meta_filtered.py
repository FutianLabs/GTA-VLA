#!/usr/bin/env python3
"""
Filter RoboMIND meta files: remove episodes whose puppet/end_effector
3D trajectory is essentially static (arm didn't move).
"""
import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

META_DIR = Path("/VLA-Data/scripts/lianqing/data/xvla_metadata")

DATASETS = {
    "robomind-franka-1rgb": {
        "meta": META_DIR / "robomind-franka-1rgb_meta.json",
        "proprio_key": "puppet/end_effector",
    },
    "robomind-franka-3rgb": {
        "meta": META_DIR / "robomind-franka-3rgb_meta.json",
        "proprio_key": "puppet/end_effector",
    },
    "robomind-sim-franka": {
        "meta": META_DIR / "robomind-sim-franka_meta.json",
        "proprio_key": "franka/end_effector",
    },
    "robomind-ur": {
        "meta": META_DIR / "robomind-ur_meta.json",
        "proprio_key": "puppet/end_effector",
    },
}


def _check_one(h5_path: str, proprio_key: str, min_travel: float) -> bool:
    try:
        with h5py.File(h5_path, 'r') as f:
            if proprio_key not in f:
                return False
            ee = f[proprio_key][:]
            if ee.shape[0] < 2 or ee.shape[1] < 3:
                return False
            return float(np.sum(np.linalg.norm(np.diff(ee[:, :3], axis=0), axis=1))) >= min_travel
    except Exception:
        return False


def filter_one(name: str, min_travel: float, workers: int):
    cfg = DATASETS[name]
    meta_path = cfg["meta"]
    proprio_key = cfg["proprio_key"]

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    datalist = meta.get("datalist", [])
    total = len(datalist)
    h5_paths = [item if isinstance(item, str) else item.get("path", "") for item in datalist]

    print(f"[{name}] Checking {total} episodes ({workers} workers, key={proprio_key}) ...")
    fn = partial(_check_one, proprio_key=proprio_key, min_travel=min_travel)
    with Pool(workers) as pool:
        results = pool.map(fn, h5_paths, chunksize=200)

    filtered = [item for item, ok in zip(datalist, results) if ok]
    removed = total - len(filtered)

    meta["datalist"] = filtered
    out_path = meta_path.parent / f"{meta_path.stem}_filtered.json"
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[{name}] Saved {out_path.name}  (kept {len(filtered)}/{total}, removed {removed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--min_travel", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    for name in args.datasets:
        if name not in DATASETS:
            print(f"Unknown: {name}, choose from {list(DATASETS.keys())}")
            continue
        filter_one(name, args.min_travel, args.workers)
