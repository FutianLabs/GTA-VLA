#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import h5py


def _resolve_h5_path(data_dir: Path, item: dict) -> str:
    h5_path = item.get("h5_path", "")
    p = Path(h5_path)
    local = data_dir / p.name
    if local.exists():
        return str(local.resolve())
    if p.exists():
        return str(p.resolve())
    return str(local.resolve())


def _load_meta_template(data_dir: Path, selected_paths: List[str]) -> dict:
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["datalist"] = selected_paths
        return meta

    has_any_wrist = False
    for path in selected_paths:
        with h5py.File(path, "r") as f:
            if f.attrs.get("wrist_view_valid", False):
                has_any_wrist = True
                break

    observation_key = ["observation/image_0"]
    optional_view_key = ["observation/image_1", "observation/image_2"]
    if has_any_wrist:
        observation_key.append("observation/image_3")
    else:
        optional_view_key.append("observation/image_3")

    return {
        "observation_key": observation_key,
        "optional_view_key": optional_view_key,
        "dataset_name": "Bridge",
        "language_instruction_key": "instruction",
        "datalist": selected_paths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--train_per_task", type=int, default=180)
    parser.add_argument("--val_per_task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_prefix", type=str, default="debug_200_per_task")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else data_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for item in manifest:
        if item.get("success", True):
            by_task[item["task_key"]].append(item)

    counts = {
        task: {
            "success_count": len(items),
            "train_target": min(args.train_per_task, len(items)),
            "val_target": min(args.val_per_task, max(0, len(items) - min(args.train_per_task, len(items)))),
        }
        for task, items in sorted(by_task.items())
    }

    rng = random.Random(args.seed)
    train_paths: List[str] = []
    val_paths: List[str] = []

    for task, items in sorted(by_task.items()):
        picked = list(items)
        rng.shuffle(picked)
        train_n = min(args.train_per_task, len(picked))
        val_n = min(args.val_per_task, max(0, len(picked) - train_n))
        train_items = picked[:train_n]
        val_items = picked[train_n:train_n + val_n]
        train_paths.extend(_resolve_h5_path(data_dir, x) for x in train_items)
        val_paths.extend(_resolve_h5_path(data_dir, x) for x in val_items)
        counts[task]["train_count"] = len(train_items)
        counts[task]["val_count"] = len(val_items)

    train_meta = _load_meta_template(data_dir, train_paths)
    val_meta = _load_meta_template(data_dir, val_paths)

    prefix = args.output_prefix
    count_path = data_dir / f"{prefix}_task_counts.json"
    train_meta_path = data_dir / f"{prefix}_train_meta.json"
    val_meta_path = data_dir / f"{prefix}_val_meta.json"

    with open(count_path, "w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "manifest_path": str(manifest_path),
                "total_success_episodes": sum(v["success_count"] for v in counts.values()),
                "num_tasks": len(counts),
                "train_total": len(train_paths),
                "val_total": len(val_paths),
                "tasks": counts,
            },
            f,
            indent=2,
        )

    with open(train_meta_path, "w") as f:
        json.dump(train_meta, f, indent=2)
    with open(val_meta_path, "w") as f:
        json.dump(val_meta, f, indent=2)

    print("Task success counts:")
    for task, info in counts.items():
        print(
            f"{task}: success={info['success_count']} "
            f"train={info['train_count']} val={info['val_count']}"
        )
    print(f"train_total={len(train_paths)}")
    print(f"val_total={len(val_paths)}")
    print(f"count_json={count_path}")
    print(f"train_meta={train_meta_path}")
    print(f"val_meta={val_meta_path}")


if __name__ == "__main__":
    main()
