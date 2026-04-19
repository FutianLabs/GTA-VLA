#!/bin/bash
# LIBERO 训练脚本
# 单卡: accelerate launch
# 多卡/多机: DeepSpeed launcher
#
# 用法:
#   ./scripts/train_libero.sh <metas_path> <run_name> [--scratch] [--batch_size N] [其他参数...]

# Parse --scratch flag
SCRATCH=false
ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--scratch" ]] && SCRATCH=true || ARGS+=("$arg")
done
set -- "${ARGS[@]}"

log_meta_path=/VLA-Data/scripts/lianqing/logs/xvla
NUM_GPUS=${MLP_WORKER_GPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}

# 检测是否是多机环境
if [[ -z "$MLP_MPI_HOSTFILE" || -z "$MLP_WORKER_0_HOST" ]]; then
    IS_MULTINODE=false
else
    IS_MULTINODE=true
fi

# Determine log directory
if [ $NUM_GPUS -le 1 ] && [ "$IS_MULTINODE" = false ]; then
    export WANDB_MODE=disabled
    logs=${log_meta_path}/libero_dummy/$2
else
    if [ "$SCRATCH" = true ]; then
        logs=${log_meta_path}/libero_scratch/$2
    else
        logs=${log_meta_path}/libero/$2
    fi
fi

logs="${logs}-$(date +%m-%d-%H-%M)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEEPSPEED_CONFIG="${PROJECT_DIR}/configs/deepspeed_zero2.json"

# Build launch command
if [ "$IS_MULTINODE" = true ]; then
    # 多机：使用 MLP hostfile
    echo "========================================"
    echo "Using DeepSpeed Multi-Node"
    echo "  hostfile=$MLP_MPI_HOSTFILE"
    echo "  master_addr=$MLP_WORKER_0_HOST"
    echo "========================================"
    LAUNCH_CMD="deepspeed --hostfile=$MLP_MPI_HOSTFILE --master_addr=${MLP_WORKER_0_HOST} --force_multi"
elif [ $NUM_GPUS -gt 1 ]; then
    # 单机多卡：DeepSpeed
    MASTER_PORT=${MASTER_PORT:-29500}
    echo "========================================"
    echo "Using DeepSpeed (${NUM_GPUS} GPUs)"
    echo "========================================"
    LAUNCH_CMD="deepspeed --num_gpus $NUM_GPUS --master_port $MASTER_PORT"
else
    # 单卡：accelerate
    echo "========================================"
    echo "Using Standard Mode (1 GPU)"
    echo "========================================"
    LAUNCH_CMD="accelerate launch --mixed_precision bf16"
fi

echo "GPUs: $NUM_GPUS | Multinode: $IS_MULTINODE | Scratch: $SCRATCH | Output: $logs"

# Common DeepSpeed args (only for multi-GPU)
if [ $NUM_GPUS -gt 1 ] || [ "$IS_MULTINODE" = true ]; then
    DS_ARGS="--deepspeed $DEEPSPEED_CONFIG"
else
    DS_ARGS=""
fi

# Training
if [ "$SCRATCH" = true ]; then
    $LAUNCH_CMD train.py $DS_ARGS \
        --iters 60000 --image_color_jitter False --batch_size 32 \
        --train_metas_path $1 --output_dir $logs ${@:3}
else
    $LAUNCH_CMD train.py $DS_ARGS \
        --models '/VLA-Data/scripts/lianqing/checkpoints/2toINF/X-VLA-Pt' \
        --iters 60000 --use_cosine_decay --image_color_jitter False \
        --train_metas_path $1 --output_dir $logs ${@:3}
fi
