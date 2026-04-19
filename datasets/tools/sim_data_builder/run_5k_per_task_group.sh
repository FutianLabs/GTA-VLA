#!/usr/bin/env bash
# 单卡：直接 ./run_5k_per_task_group.sh（默认并行 4，减轻 Vulkan DeviceLost）
# 多卡：NUM_GPUS=8 NPROC=16 ./run_5k_per_task_group.sh
# 单卡想更快：NPROC=8 ./run_5k_per_task_group.sh（仍可能抢显存，视情况调小）
# 用法：在 x-vla-main 根下 export PYTHONPATH=$PWD:...SimplerEnv...

set -euo pipefail
XVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$XVLA_ROOT"
export PYTHONPATH="${XVLA_ROOT}:${PYTHONPATH:-}"
LINGYIRAN="$(cd "${XVLA_ROOT}/.." && pwd)"
OUT="${1:-${LINGYIRAN}/data/openX/x-vla/bridge_5k_group}"
mkdir -p "$OUT"

NUM_GPUS="${NUM_GPUS:-1}"
FIRST_GPU="${FIRST_GPU:-0}"
_CPUS="$(nproc)"

if [ -z "${NPROC:-}" ]; then
  NPROC="$_CPUS"
  if [ "$NUM_GPUS" -le 1 ]; then
    [ "$NPROC" -gt 4 ] && NPROC=4
  else
    [ "$NPROC" -gt 8 ] && NPROC=8
  fi
fi

echo "NPROC=${NPROC} NUM_GPUS=${NUM_GPUS} FIRST_GPU=${FIRST_GPU} OUT=${OUT}"
python3 -m datasets.tools.sim_data_builder.run \
  --group base multi_object layout_distractor \
  --episodes 5000 \
  --success_only \
  --parallel "${NPROC}" \
  --num_gpus "${NUM_GPUS}" \
  --first_gpu "${FIRST_GPU}" \
  --output_dir "$OUT"

echo "Done: $OUT"
