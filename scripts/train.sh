#!/bin/bash
# 统一训练脚本
# 用法:
#   ./scripts/train.sh <meta_path> <run_name> [--scratch] [train.py 参数...]
#
# 示例:
#   ./scripts/train.sh data/bridge_meta.json bridge_exp1
#   ./scripts/train.sh data/bridge_meta.json bridge_exp1 --scratch
#   ./scripts/train.sh data/droid_meta.json droid_exp1 --batch_size 16
#
# log 目录自动从 meta 文件名推断:
#   data/bridge_meta.json  → logs/xvla/bridge/<run_name>-MM-DD-HH-MM
#   data/droid_meta.json   → logs/xvla/droid/<run_name>-MM-DD-HH-MM

set -euo pipefail

[[ $# -lt 2 ]] && { echo "Usage: $0 <meta_path> <run_name> [--scratch] [train.py args...]"; exit 1; }

META="$1"; RUN_NAME="$2"; shift 2

SCRATCH=false; EXTRA_ARGS=()
for arg in "$@"; do [[ "$arg" == "--scratch" ]] && SCRATCH=true || EXTRA_ARGS+=("$arg"); done

# log prefix 从 meta 文件名推断: bridge_meta.json → bridge, robomind-franka_meta.json → robomind-franka
BASENAME="$(basename "$META" .json)"
LOG_PREFIX="${BASENAME%_meta}"
[[ "$LOG_PREFIX" == "$BASENAME" ]] && LOG_PREFIX="$(basename "$(dirname "$META")")"

# ─── Environment ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_ROOT=/VLA-Data/scripts/lianqing/logs/xvla
DEFAULT_MODEL='/VLA-Data/scripts/lianqing/checkpoints/2toINF/X-VLA-Pt'
NUM_GPUS=${MLP_WORKER_GPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}

if [[ -n "${MLP_MPI_HOSTFILE:-}" && -n "${MLP_WORKER_0_HOST:-}" ]]; then
    IS_MULTINODE=true
else
    IS_MULTINODE=false
fi

# ─── Log directory ──────────────────────────────────────────────────
if [ "$NUM_GPUS" -le 1 ] && [ "$IS_MULTINODE" = false ]; then
    export WANDB_MODE=disabled
    logs=${LOG_ROOT}/${LOG_PREFIX}_dummy/${RUN_NAME}
elif [ "$SCRATCH" = true ]; then
    logs=${LOG_ROOT}/${LOG_PREFIX}_scratch/${RUN_NAME}
else
    logs=${LOG_ROOT}/${LOG_PREFIX}/${RUN_NAME}
fi
logs="${logs}-$(date +%m-%d-%H-%M)"

# ─── Launch command ─────────────────────────────────────────────────
DS_ARGS=""
if [ "$IS_MULTINODE" = true ]; then
    LAUNCH_CMD="deepspeed --hostfile=$MLP_MPI_HOSTFILE --master_addr=${MLP_WORKER_0_HOST} --force_multi"
    DS_ARGS="--deepspeed ${PROJECT_DIR}/configs/deepspeed_zero2.json"
elif [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH_CMD="deepspeed --num_gpus $NUM_GPUS --master_port ${MASTER_PORT:-29500}"
    DS_ARGS="--deepspeed ${PROJECT_DIR}/configs/deepspeed_zero2.json"
else
    LAUNCH_CMD="accelerate launch --mixed_precision bf16"
fi

echo "========================================"
echo "Meta: $META → log prefix: $LOG_PREFIX"
echo "GPUs: $NUM_GPUS | Multinode: $IS_MULTINODE | Scratch: $SCRATCH"
echo "Output: $logs"
echo "========================================"

# ─── Train ──────────────────────────────────────────────────────────
TRAIN_ARGS=(
    --train_metas_path "$META" --output_dir "$logs"
    --iters 400000 --use_cosine_decay --learning_rate 1e-4
    --batch_size 32 --learning_coef 0.1
    --freeze_steps 1000 --warmup_steps 2000
)
[[ "$SCRATCH" = false ]] && TRAIN_ARGS+=(--models "$DEFAULT_MODEL")
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

$LAUNCH_CMD train.py $DS_ARGS "${TRAIN_ARGS[@]}"
