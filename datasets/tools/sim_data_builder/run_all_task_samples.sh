#!/usr/bin/env bash
# 每个 task_key 各采 1 条（--group 三组合计 10 个；不含 eggplant 系列）。
# 默认输出与 bridge_enhance 同级：scripts/lingyiran/data/openX/x-vla/bridge_all_tasks_sample
#
# 用法（在 x-vla-main 根目录）：
#   export PYTHONPATH=$PWD:/path/to/SimplerEnv:/path/to/SimplerEnv/ManiSkill2_real2sim
#   ./datasets/tools/sim_data_builder/run_all_task_samples.sh [输出目录]
#
# 等价命令：
#   python3 -m datasets.tools.sim_data_builder.run \
#     --group base multi_object layout_distractor \
#     --episodes 1 --save_video \
#     --output_dir ../data/openX/x-vla/bridge_all_tasks_sample

set -euo pipefail
XVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$XVLA_ROOT"
export PYTHONPATH="${XVLA_ROOT}:${PYTHONPATH:-}"

LINGYIRAN="$(cd "${XVLA_ROOT}/.." && pwd)"
OUT_DIR="${1:-${LINGYIRAN}/data/openX/x-vla/bridge_all_tasks_sample}"
mkdir -p "$OUT_DIR"

python3 -m datasets.tools.sim_data_builder.run \
  --group base multi_object layout_distractor \
  --episodes 1 \
  --save_video \
  --output_dir "$OUT_DIR"

echo "Done. Output: $OUT_DIR"
