#!/bin/bash
# WidowX evaluation (backward compatible)
# Usage: ./eval_bridge.sh <checkpoint_dir> [final_step]

source .venv/bin/activate

CHECKPOINT_DIR="${1:-.}"
FINAL_STEP="${2:-200000}"

python auto_eval_on_checkpoint.py "$CHECKPOINT_DIR" "$FINAL_STEP" widowx