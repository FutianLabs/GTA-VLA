#!/usr/bin/env bash
# 每个 task_key 各采 1 条（--group 三组合计 10 个；不含 eggplant 系列）。
# 默认输出与 bridge_enhance 同级：data/openX/gtavla/bridge_all_tasks_sample
#
# 用法（在 GTA-VLA 根目录）：
#   export PYTHONPATH=$PWD:/path/to/SimplerEnv:/path/to/SimplerEnv/ManiSkill2_real2sim
#   ./datasets/tools/sim_data_builder/run_all_task_samples.sh [输出目录]
#
# 等价命令：
#   python3 -m datasets.tools.sim_data_builder.run \
#     --group base multi_object layout_distractor \
#     --episodes 1 --save_video \
#     --output_dir data/openX/gtavla/bridge_all_tasks_sample

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${1:-${PROJECT_ROOT}/data/openX/gtavla/bridge_all_tasks_sample}"
mkdir -p "$OUT_DIR"

python3 -m datasets.tools.sim_data_builder.run \
  --group base multi_object layout_distractor \
  --episodes 1 \
  --save_video \
  --output_dir "$OUT_DIR"

echo "Done. Output: $OUT_DIR"
