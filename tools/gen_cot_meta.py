#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COT_ROOT = os.getenv("GTA_VLA_COT_ROOT", str(PROJECT_ROOT / "data" / "cot_annotations"))
OUTPUT_DIR = PROJECT_ROOT / "data"

CONFIGS = {
    "droid": {
        "source_meta": str(OUTPUT_DIR / "droid_meta.json"),
        "output_name": "droid_meta_direct.json",
        "dataset_name": "DroidCot-Left",
        "observation_key": ["observation/exterior_image_1_left"],
        "language_instruction_key": "language_instruction",
        "annotation_dir": f"{COT_ROOT}/droid_annotations_main",
        "dataset_config": {
            "image_key": "observation/exterior_image_1_left",
            "instruction_key": "language_instruction",
            "action_key": "action",
            "gripper_idx": -1,
            "proprio_key": "proprio",
            "image_format": "jpeg",
            "gripper_inverted": False,
            "episode_id_parts": None,
        },
        "cot_config": {"coord_scale": 1000, "gripper_future_steps": 5, "detector_priority": ["seed_vl", "dino_x"]},
    },
    "robomind-franka": {
        "source_meta": str(OUTPUT_DIR / "robomind-franka_meta.json"),
        "output_name": "robomind-franka_meta_cot.json",
        "dataset_name": "robomind-franka-cot",
        "observation_key": ["observations/rgb_images/camera_top"],
        "language_instruction_key": "language_raw",
        "annotation_dir": f"{COT_ROOT}/robomind-franka_annotations_main",
        "dataset_config": {
            "image_key": "observations/rgb_images/camera_top",
            "instruction_key": "language_raw",
            "instruction_from_dataset": True,
            "action_key": "puppet/joint_position",
            "gripper_idx": -1,
            "proprio_key": "puppet/end_effector",
            "image_format": "raw",
            "raw_image_shape": [720, 1280, 3],
            "gripper_inverted": False,
            "gripper_closing_threshold": 0.1,
            "gripper_opening_threshold": -0.1,
            "gripper_state_threshold": 0.3,
            "gripper_smooth_window": 10,
            "episode_id_parts": [-6, -3],
        },
        "cot_config": {"coord_scale": 1000, "gripper_future_steps": 5, "detector_priority": ["seed_vl", "dino_x"]},
    },
    "robomind-ur": {
        "source_meta": str(OUTPUT_DIR / "robomind-ur_meta.json"),
        "output_name": "robomind-ur_meta_cot.json",
        "dataset_name": "robomind-ur-cot",
        "observation_key": ["observations/rgb_images/camera_top"],
        "language_instruction_key": "language_raw",
        "annotation_dir": f"{COT_ROOT}/robomind-ur_annotations_main",
        "dataset_config": {
            "image_key": "observations/rgb_images/camera_top",
            "instruction_key": "language_raw",
            "instruction_from_dataset": True,
            "action_key": "puppet/joint_position",
            "gripper_idx": -1,
            "proprio_key": "puppet/end_effector",
            "image_format": "raw",
            "gripper_inverted": False,
            "gripper_closing_threshold": 0.1,
            "gripper_opening_threshold": -0.1,
            "gripper_state_threshold": 0.3,
            "gripper_smooth_window": 10,
            "episode_id_parts": [-6, -3],
        },
        "cot_config": {"coord_scale": 1000, "gripper_future_steps": 5, "detector_priority": ["seed_vl", "dino_x"]},
    },
}


def generate(name: str):
    cfg = CONFIGS[name]
    source = cfg["source_meta"]
    print(f"Reading source meta: {source}")
    with open(source, 'r') as f:
        src = json.load(f)

    datalist = src.get("datalist", [])
    meta = {
        "dataset_name": cfg["dataset_name"],
        "observation_key": cfg["observation_key"],
        "language_instruction_key": cfg["language_instruction_key"],
        "annotation_dir": cfg["annotation_dir"],
        "dataset_config": cfg["dataset_config"],
        "cot_config": cfg["cot_config"],
        "datalist": datalist,
    }

    out_path = OUTPUT_DIR / cfg["output_name"]
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}  ({len(datalist)} episodes)")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS.keys())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in targets:
        if t not in CONFIGS:
            print(f"Unknown dataset: {t}, choose from {list(CONFIGS.keys())}")
            continue
        generate(t)
