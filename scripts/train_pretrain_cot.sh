#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_name> [train.py args...]"
    echo "Example: $0 pretrain_cot_v1 --batch_size 16 --iters 800000"
    exit 1
fi

RUN_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_PATH="${PROJECT_DIR}/configs/pretrain/xvla_qwen3vl_2b_cot.json"
SRC_PRETRAIN_META_DIR="/VLA-Data/scripts/lianqing/data/xvla_metadata/pretrain"
META_MIX_DIR="${PROJECT_DIR}/data/_generated/pretrain_cot_from_pretrain"

mkdir -p "$META_MIX_DIR"
rm -f "${META_MIX_DIR}"/*.json

# Keep exactly the same pretrain meta set used in raw-data pretraining
PRETRAIN_METAS=(
    "bridge_meta_wrist.json"
    "droid_meta.json"
    "robomind-franka-1rgb_meta.json"
    "robomind-franka-3rgb_meta.json"
    "robomind-ur_meta.json"
)

echo "========================================"
echo "Building CoT pretrain metas from: ${SRC_PRETRAIN_META_DIR}"
echo "Output meta directory: ${META_MIX_DIR}"

for meta_name in "${PRETRAIN_METAS[@]}"; do
    src_meta="${SRC_PRETRAIN_META_DIR}/${meta_name}"
    if [[ ! -f "$src_meta" ]]; then
        echo "[ERROR] meta not found: ${src_meta}"
        exit 1
    fi

    dst_meta="${META_MIX_DIR}/${meta_name}"

    SRC_META="$src_meta" DST_META="$dst_meta" uv run python - <<'PY'
import json
import os

src = os.environ["SRC_META"]
dst = os.environ["DST_META"]

with open(src, "r", encoding="utf-8") as f:
    meta = json.load(f)

dataset_name = meta.get("dataset_name", "")
annotation_dir_map = {
    "Bridge": "/VLA-Data/scripts/lianqing/data/xvla/cot_annotations/bridge_annotations_wrist",
    "Droid-Left": "/VLA-Data/scripts/lianqing/data/xvla/cot_annotations/droid_annotations_main",
    "robomind-franka-1rgb": "/VLA-Data/scripts/lianqing/data/xvla/cot_annotations/robomind-franka_annotations_main",
    "robomind-franka-3rgb": "/VLA-Data/scripts/lianqing/data/xvla/cot_annotations/robomind-franka_annotations_main",
    "robomind-ur": "/VLA-Data/scripts/lianqing/data/xvla/cot_annotations/robomind-ur_annotations_main",
}

if dataset_name not in annotation_dir_map:
    raise ValueError(f"Unsupported pretrain dataset_name for CoT: {dataset_name}")

meta["annotation_dir"] = annotation_dir_map[dataset_name]
meta["cot_config"] = {
    "coord_scale": 1000,
    "gripper_future_steps": 5,
    "detector_priority": ["seed_vl", "dino_x"],
}

with open(dst, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

    echo "  + ${meta_name}"
done
echo "========================================"

cd "$PROJECT_DIR"

if command -v uv >/dev/null 2>&1; then
    uv run bash scripts/train.sh \
        "$META_MIX_DIR" "$RUN_NAME" --scratch \
        --model_arch xvla \
        --config_path "$CONFIG_PATH" \
        --use_cosine_decay --learning_rate 1e-4 --learning_coef 0.1 \
        --warmup_steps 4000 --freeze_steps 2000 --batch_size 24 \
        "$@"
else
    bash scripts/train.sh \
        "$META_MIX_DIR" "$RUN_NAME" --scratch \
        --model_arch xvla \
        --config_path "$CONFIG_PATH" \
        --use_cosine_decay --learning_rate 1e-4 --learning_coef 0.1 \
        --warmup_steps 4000 --freeze_steps 2000 --batch_size 24 \
        "$@"
fi
