#!/usr/bin/env python3
"""每个已注册 task 各跑 1 条样例（HDF5 + 第三视角/手腕 MP4）。

默认输出与 bridge_enhance 同级：GTA-VLA/data/openX/gtavla/bridge_all_tasks_sample

在 GTA-VLA 根目录执行，并设置 PYTHONPATH（含 SimplerEnv）：

  cd .../GTA-VLA
  export PYTHONPATH=$PWD:/path/to/SimplerEnv:/path/to/SimplerEnv/ManiSkill2_real2sim
  python3 datasets/tools/sim_data_builder/run_all_task_samples.py [输出目录]
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from datasets.tools.sim_data_builder.run import run
    from datasets.tools.sim_data_builder.task_config import list_group_run_task_keys

    default_out = root / "data/openX/gtavla/bridge_all_tasks_sample"
    out = sys.argv[1] if len(sys.argv) > 1 else str(default_out)
    run(
        tasks=list_group_run_task_keys(),
        output_dir=out,
        episodes_override=1,
        save_video=True,
        video_fps=10,
        parallel=1,
    )


if __name__ == "__main__":
    main()
