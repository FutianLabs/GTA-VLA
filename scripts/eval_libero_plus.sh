#!/bin/bash

# Evaluate a model on all seven LIBERO-Plus perturbation types.
# Usage: scripts/eval_libero_plus.sh <model_path> [extra args passed to evaluate_libero.py]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [extra args...]"
    exit 1
fi

MODEL_PATH=$1
shift || true

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

MODEL_NAME="$(basename "$MODEL_PATH")"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_ROOT="$REPO_ROOT/evaluation_outputs/libero_plus/${MODEL_NAME}_${STAMP}"
mkdir -p "$LOG_ROOT"

PERTURBATIONS=(camera robot language light background noise layout)
SUMMARY_FILE="$LOG_ROOT/summary.txt"

{
    echo "Model: $MODEL_PATH"
    echo "Log root: $LOG_ROOT"
    echo "Perturbations: ${PERTURBATIONS[*]}"
    echo "Extra args: $*"
    echo ""
} > "$SUMMARY_FILE"

for p in "${PERTURBATIONS[@]}"; do
    SUITE="libero_plus_${p}"
    RUN_DIR="$LOG_ROOT/${SUITE}"
    RUN_LOG="$LOG_ROOT/${SUITE}.log"
    mkdir -p "$RUN_DIR"

    echo "=== Evaluating $SUITE ==="
    python -m evaluation.scripts.evaluate_libero \
        --sim libero_plus \
        --model_path "$MODEL_PATH" \
        --task_suites "$SUITE" \
        --episodes 1 \
        --output_dir "$RUN_DIR" \
        "$@" | tee "$RUN_LOG"

    SUMMARY_LINE=$(grep -E "Overall Average" "$RUN_LOG" | tail -n 1 || true)
    if [ -n "$SUMMARY_LINE" ]; then
        echo "$SUITE: $SUMMARY_LINE" >> "$SUMMARY_FILE"
    else
        echo "$SUITE: (no summary found, see $RUN_LOG)" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "Finished. Detailed logs saved under: $LOG_ROOT"
echo "Per-suite summary: $SUMMARY_FILE"
