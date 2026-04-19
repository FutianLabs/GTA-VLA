#!/bin/bash

source .venv/bin/activate

CHECKPOINT_DIR="${1:-.}"
GOOGLE_SETTING="${2:-both}"   # vm | va | both
FINAL_STEP="${3:--1}"         # -1 means monitor forever
OUTPUT_ROOT="$CHECKPOINT_DIR/eval_outputs"
SUMMARY_ROOT="$CHECKPOINT_DIR/eval_outputs/summaries"
POLL_INTERVAL="${POLL_INTERVAL:-120}"
WAIT_BEFORE_EVAL="${WAIT_BEFORE_EVAL:-10}"
SIMPLER_DIR_INPUT="${SIMPLER_DIR:-/VLA-Data/scripts/lingyiran/SimplerEnv}"

if [ ! -d "$SIMPLER_DIR_INPUT" ]; then
  echo "SIMPLER_DIR not found: $SIMPLER_DIR_INPUT"
  echo "Please set SIMPLER_DIR, e.g. export SIMPLER_DIR=/path/to/SimplerEnv"
  exit 1
fi

export SIMPLER_DIR="$SIMPLER_DIR_INPUT"

run_one() {
  local setting="$1"
  SIMPLER_DIR="$SIMPLER_DIR_INPUT" python auto_eval_on_checkpoint.py "$CHECKPOINT_DIR" "$FINAL_STEP" simpler_google \
    --no_wandb \
    --output_root "$OUTPUT_ROOT" \
    --output_layout flat \
    --summary_root "$SUMMARY_ROOT" \
    --poll_interval "$POLL_INTERVAL" \
    --wait_before_eval "$WAIT_BEFORE_EVAL" \
    --eval_arg=--google_setting \
    --eval_arg="$setting" \
    --eval_arg=--save_video \
    --eval_arg=--simpler_dir \
    --eval_arg="$SIMPLER_DIR_INPUT"
}

if [ "$GOOGLE_SETTING" = "both" ]; then
  run_one vm &
  PID_VM=$!
  run_one va &
  PID_VA=$!
  wait "$PID_VM" "$PID_VA"
elif [ "$GOOGLE_SETTING" = "vm" ] || [ "$GOOGLE_SETTING" = "va" ]; then
  run_one "$GOOGLE_SETTING"
else
  echo "Invalid GOOGLE_SETTING: $GOOGLE_SETTING (use vm|va|both)"
  exit 1
fi
