#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_name> [--scratch] [train.py args...]"
    exit 1
fi

RUN_NAME="$1"
shift

SCRATCH=false
EXTRA_ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--scratch" ]] && SCRATCH=true || EXTRA_ARGS+=("$arg")
done

META_PATH="${MANISKILL_META_PATH:-}"
CONFIG_PATH="${MANISKILL_CONFIG_PATH:-}"
DEFAULT_MODEL="${MANISKILL_BASE_MODEL:-}"

NUM_GPUS=${MLP_WORKER_GPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}

if [[ -z "$MLP_MPI_HOSTFILE" || -z "$MLP_WORKER_0_HOST" ]]; then
    IS_MULTINODE=false
else
    IS_MULTINODE=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -z "$META_PATH" ]]; then
    META_PATH="${PROJECT_DIR}/data/maniskill_meta_cot.json"
fi
if [[ -z "$CONFIG_PATH" ]]; then
    CONFIG_PATH="${PROJECT_DIR}/configs/maniskill/gtavla_qwen3vl_2b_baseline_cot.json"
fi
mkdir -p "${PROJECT_DIR}/ckpt/maniskill"
OUTPUT_DIR="${PROJECT_DIR}/ckpt/maniskill/${RUN_NAME}-$(date +%m-%d-%H-%M)"
BATCH_SIZE=16

if [ "$NUM_GPUS" -le 1 ] && [ "$IS_MULTINODE" = false ]; then
    export WANDB_MODE=disabled
    BATCH_SIZE=1
fi
DEEPSPEED_CONFIG="${PROJECT_DIR}/configs/deepspeed_zero2.json"

if [ "$IS_MULTINODE" = true ]; then
    echo "========================================"
    echo "Using DeepSpeed Multi-Node"
    echo "  hostfile=$MLP_MPI_HOSTFILE"
    echo "  master_addr=$MLP_WORKER_0_HOST"
    echo "========================================"
    LAUNCH_CMD="deepspeed --hostfile=$MLP_MPI_HOSTFILE --master_addr=${MLP_WORKER_0_HOST} --force_multi"
elif [ "$NUM_GPUS" -gt 1 ]; then
    MASTER_PORT=${MASTER_PORT:-29500}
    echo "========================================"
    echo "Using DeepSpeed (${NUM_GPUS} GPUs)"
    echo "========================================"
    LAUNCH_CMD="deepspeed --num_gpus $NUM_GPUS --master_port $MASTER_PORT"
else
    echo "========================================"
    echo "Using Standard Mode (1 GPU)"
    echo "========================================"
    LAUNCH_CMD="accelerate launch --mixed_precision bf16"
fi

echo "Meta: $META_PATH"
echo "Config: $CONFIG_PATH"
echo "Base model: $DEFAULT_MODEL"
echo "GPUs: $NUM_GPUS | Multinode: $IS_MULTINODE | Scratch: $SCRATCH | Batch size: $BATCH_SIZE | Output: $OUTPUT_DIR"

if [ "$NUM_GPUS" -gt 1 ] || [ "$IS_MULTINODE" = true ]; then
    DS_ARGS="--deepspeed $DEEPSPEED_CONFIG"
else
    DS_ARGS=""
fi

if [ "$SCRATCH" = true ]; then
    $LAUNCH_CMD train.py $DS_ARGS \
        --model_arch gtavla \
        --config_path "$CONFIG_PATH" \
        --iters 200000 --use_cosine_decay --learning_rate 1e-4 --batch_size "$BATCH_SIZE" \
        --learning_coef 0.1 --freeze_steps 1000 --warmup_steps 2000 \
        --train_metas_path "$META_PATH" --output_dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
else
    if [[ -z "$DEFAULT_MODEL" ]]; then
        echo "Error: missing base model path. Set MANISKILL_BASE_MODEL or pass --scratch."
        exit 1
    fi
    $LAUNCH_CMD train.py $DS_ARGS \
        --model_arch gtavla \
        --models "$DEFAULT_MODEL" \
        --config_path "$CONFIG_PATH" \
        --iters 200000 --use_cosine_decay --learning_rate 1e-4 --batch_size "$BATCH_SIZE" \
        --learning_coef 0.1 --freeze_steps 1000 --warmup_steps 2000 \
        --train_metas_path "$META_PATH" --output_dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
fi
