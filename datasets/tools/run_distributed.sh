#!/bin/bash
# Distributed CoT Annotation Pipeline Launcher
# This script launches 8 parallel processes across 8 GPUs
# Each process handles 1/8 of the total workload using sharding
#
# Usage:
#   bash run_distributed.sh <meta_path> <output_dir> <detector> [additional_args...]
#
# Examples:
#
#   1. DINO-X (requires --dds_token):
#      bash run_distributed.sh ../../data/bridge_meta.json ./cot_output_dinox dinox \
#          --dds_token <token>
#
#   2. Rex-Omni (requires --model_path, optional --backend):
#      bash run_distributed.sh ../../data/bridge_meta.json ./cot_output_rexomni rexomni \
#          --model_path IDEA-Research/Rex-Omni --backend transformers
#
#      # With custom model path:
#      bash run_distributed.sh ../../data/bridge_meta.json ./cot_output_rexomni rexomni \
#          --model_path /root/checkpoints/IDEA-Research/Rex-Omni --backend vllm
#
#   3. Qwen3-VL-Flash (requires --api_key):
#      bash run_distributed.sh ../../data/bridge_meta.json ./cot_output_qwen_flash qwen3-vl-flash \
#          --api_key <key>
#
#   4. Qwen3-VL-Plus (requires --api_key):
#      bash run_distributed.sh ../../data/bridge_meta.json ./cot_output_qwen_plus qwen3-vl-plus \
#          --api_key <key>
#
#   5. Doubao VLM / Seed-VL-1.5 (requires --api_key):
#      bash run_distributed.sh ../../data/bridge_meta.json ./cot_output_seedvl seed-vl-1.5 \
#          --api_key <key>
#
# The script will:
#   1. Create logs directory
#   2. Launch 8 parallel processes (one per GPU)
#   3. Each process logs to logs/shard_N.log
#   4. Wait for all processes to complete
#   5. Print completion status
#
# Monitor progress:
#   tail -f <output_dir>/logs/shard_0.log        # Single shard
#   tail -f <output_dir>/logs/shard_*.log        # All shards

set -e  # Exit on error

# Check minimum required arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <meta_path> <output_dir> <detector> [additional_args...]"
    echo ""
    echo "Supported detectors:"
    echo "  dinox          - DINO-X API (requires --dds_token)"
    echo "  rexomni        - Rex-Omni local model (requires --model_path, optional --backend)"
    echo "  qwen3-vl-flash - Qwen3-VL-Flash API (requires --api_key)"
    echo "  qwen3-vl-plus  - Qwen3-VL-Plus API (requires --api_key)"
    echo "  seed-vl-1.5    - Doubao VLM API (requires --api_key)"
    echo ""
    echo "Examples:"
    echo "  # DINO-X:"
    echo "  $0 ../../data/bridge_meta.json ./cot_output_dinox dinox --dds_token <token>"
    echo ""
    echo "  # Rex-Omni:"
    echo "  $0 ../../data/bridge_meta.json ./cot_output_rexomni rexomni --model_path IDEA-Research/Rex-Omni"
    echo ""
    echo "  # Qwen3-VL-Flash:"
    echo "  $0 ../../data/bridge_meta.json ./cot_output_qwen_flash qwen3-vl-flash --api_key <key>"
    echo ""
    echo "  # Qwen3-VL-Plus:"
    echo "  $0 ../../data/bridge_meta.json ./cot_output_qwen_plus qwen3-vl-plus --api_key <key>"
    exit 1
fi

META_PATH=$1
OUTPUT_DIR=$2
DETECTOR=$3
shift 3  # Remove first 3 args, keep the rest as additional args
ADDITIONAL_ARGS="$@"

# Auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi
echo "Detected ${NUM_GPUS} GPU(s)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

# Create output and log directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "========================================================================"
echo "Distributed CoT Annotation Pipeline"
echo "========================================================================"
echo "Meta path:       ${META_PATH}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Detector:        ${DETECTOR}"
echo "Number of GPUs:  ${NUM_GPUS}"
echo "Additional args: ${ADDITIONAL_ARGS}"
echo "Log directory:   ${LOG_DIR}"
echo "========================================================================"
echo ""

# Store process IDs
declare -a PIDS

# Launch processes for each GPU with staggered start to avoid CUDA race conditions
echo "Launching ${NUM_GPUS} parallel processes (staggered start)..."
for i in $(seq 0 $((NUM_GPUS-1))); do
    echo "  Starting shard ${i} on GPU ${i}..."
    
    # Launch process in background
    CUDA_VISIBLE_DEVICES=$i python "${SCRIPT_DIR}/annotate_cot.py" \
        --meta_path "${META_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --detector "${DETECTOR}" \
        --shard_id $i \
        --num_shards ${NUM_GPUS} \
        --skip_existing True \
        ${ADDITIONAL_ARGS} \
        > "${LOG_DIR}/shard_${i}.log" 2>&1 &
    
    PIDS[$i]=$!
    echo "    PID: ${PIDS[$i]}, Log: ${LOG_DIR}/shard_${i}.log"
    
    # Staggered start: wait 30s between launches to avoid CUDA initialization race conditions
    if [ $i -lt $((NUM_GPUS-1)) ]; then
        echo "    Waiting 30s before next shard..."
        sleep 30
    fi
done

echo ""
echo "All processes launched. Waiting for completion..."
echo "You can monitor progress with:"
echo "  tail -f ${LOG_DIR}/shard_0.log"
echo "  tail -f ${LOG_DIR}/shard_*.log  # (monitor all)"
echo ""

# Wait for all processes to complete
FAILED=0
for i in $(seq 0 $((NUM_GPUS-1))); do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Shard ${i} completed successfully"
    else
        echo "✗ Shard ${i} failed with exit code ${EXIT_CODE}"
        FAILED=$((FAILED+1))
    fi
done

echo ""
echo "========================================================================"
if [ $FAILED -eq 0 ]; then
    echo "✓ All ${NUM_GPUS} shards completed successfully!"
    echo ""
    echo "Results saved to: ${OUTPUT_DIR}/episode_*.json"
    echo "Summary files:    ${OUTPUT_DIR}/summary_*.json"
    echo "Logs:             ${LOG_DIR}/shard_*.log"
else
    echo "✗ ${FAILED}/${NUM_GPUS} shards failed. Check logs for details:"
    echo "  ${LOG_DIR}/shard_*.log"
    exit 1
fi
echo "========================================================================"
