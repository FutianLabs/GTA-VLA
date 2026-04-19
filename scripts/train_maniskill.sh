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

META_PATH="/VLA-Data/scripts/lianqing/data/xvla_metadata/debug_200_per_task_train_maniskill_cot_meta.json"
CONFIG_PATH="/VLA-Data/scripts/lingyiran/x-vla-main/configs/maniskill/xvla_qwen3vl_2b_baseline_cot.json"
# 微调权重起点；要换别的 ckpt 或回到 X-VLA-Pt 可 export MANISKILL_BASE_MODEL=...
DEFAULT_MODEL="${MANISKILL_BASE_MODEL:-/VLA-Data/scripts/lianqing/logs/xvla/bridge/bridge_pretrain_finetune_interactive_aggressive-03-03-06-58/ckpt-100000}"

NUM_GPUS=${MLP_WORKER_GPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}

if [[ -z "$MLP_MPI_HOSTFILE" || -z "$MLP_WORKER_0_HOST" ]]; then
    IS_MULTINODE=false
else
    IS_MULTINODE=true
fi

export http_proxy=http://100.68.175.233:3128
export https_proxy=http://100.68.175.233:3128

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
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
        --model_arch xvla \
        --config_path "$CONFIG_PATH" \
        --iters 200000 --use_cosine_decay --learning_rate 1e-4 --batch_size "$BATCH_SIZE" \
        --learning_coef 0.1 --freeze_steps 1000 --warmup_steps 2000 \
        --train_metas_path "$META_PATH" --output_dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
else
    $LAUNCH_CMD train.py $DS_ARGS \
        --model_arch xvla \
        --models "$DEFAULT_MODEL" \
        --config_path "$CONFIG_PATH" \
        --iters 200000 --use_cosine_decay --learning_rate 1e-4 --batch_size "$BATCH_SIZE" \
        --learning_coef 0.1 --freeze_steps 1000 --warmup_steps 2000 \
        --train_metas_path "$META_PATH" --output_dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
fi
