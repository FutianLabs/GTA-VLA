#!/bin/bash
# Fractal 训练脚本

SCRATCH=false
ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--scratch" ]] && SCRATCH=true || ARGS+=("$arg")
done
set -- "${ARGS[@]}"

log_meta_path="${LOG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ckpt}"
NUM_GPUS=${MLP_WORKER_GPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}

if [[ -z "$MLP_MPI_HOSTFILE" || -z "$MLP_WORKER_0_HOST" ]]; then
    IS_MULTINODE=false
else
    IS_MULTINODE=true
fi
if [ $NUM_GPUS -le 1 ] && [ "$IS_MULTINODE" = false ]; then
    export WANDB_MODE=disabled
    logs=${log_meta_path}/fractal_dummy/$2
else
    if [ "$SCRATCH" = true ]; then
        logs=${log_meta_path}/fractal_scratch/$2
    else
        logs=${log_meta_path}/fractal/$2
    fi
fi

logs="${logs}-$(date +%m-%d-%H-%M)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEEPSPEED_CONFIG="${PROJECT_DIR}/configs/deepspeed_zero2.json"

if [ "$IS_MULTINODE" = true ]; then
    echo "========================================"
    echo "Using DeepSpeed Multi-Node"
    echo "  hostfile=$MLP_MPI_HOSTFILE"
    echo "  master_addr=$MLP_WORKER_0_HOST"
    echo "========================================"
    LAUNCH_CMD="deepspeed --hostfile=$MLP_MPI_HOSTFILE --master_addr=${MLP_WORKER_0_HOST} --force_multi"
elif [ $NUM_GPUS -gt 1 ]; then
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

echo "GPUs: $NUM_GPUS | Multinode: $IS_MULTINODE | Scratch: $SCRATCH | Output: $logs"

if [ $NUM_GPUS -gt 1 ] || [ "$IS_MULTINODE" = true ]; then
    DS_ARGS="--deepspeed $DEEPSPEED_CONFIG"
else
    DS_ARGS=""
fi

if [ "$SCRATCH" = true ]; then
    $LAUNCH_CMD train.py $DS_ARGS \
        --iters 200000 --use_cosine_decay --learning_rate 1e-4 --batch_size 32 \
        --learning_coef 0.1 --freeze_steps 1000 --warmup_steps 2000 \
        --train_metas_path $1 --output_dir $logs ${@:3}
else
    BASE_MODEL="${GTA_VLA_BASE_MODEL:-}"
    if [[ -z "$BASE_MODEL" ]]; then
        echo "Error: missing base model path. Set GTA_VLA_BASE_MODEL or pass --scratch."
        exit 1
    fi
    $LAUNCH_CMD train.py $DS_ARGS \
        --models "$BASE_MODEL" \
        --iters 200000 --use_cosine_decay --learning_rate 1e-4 --batch_size 32 \
        --learning_coef 0.1 --freeze_steps 1000 --warmup_steps 2000 \
        --train_metas_path $1 --output_dir $logs ${@:3}
fi
