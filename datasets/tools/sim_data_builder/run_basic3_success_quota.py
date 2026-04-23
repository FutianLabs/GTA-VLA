#!/usr/bin/env python3
"""Run basic3 WidowX tasks with a fixed success quota per task."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from datasets.tools.sim_data_builder.run import run_multi_task_success_quota
    from datasets.tools.sim_data_builder.task_config import list_basic3_task_keys

    default_out = root / "data/openX/gtavla/bridge_basic3_200"

    parser = argparse.ArgumentParser(
        description="Collect basic3 WidowX tasks with success quota per task."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_out),
        help="Output directory for HDF5 + manifest + meta",
    )
    parser.add_argument(
        "--success_quota",
        type=int,
        default=200,
        help="Successful episodes to save per task",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=min(8, max(1, os.cpu_count() or 8)),
        help="Parallel workers",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=120,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to rotate across",
    )
    parser.add_argument(
        "--first_gpu",
        type=int,
        default=0,
        help="First GPU index",
    )
    parser.add_argument(
        "--max_total_episode_ids",
        type=int,
        default=8192,
        help="Upper bound on env episode ids searched per task",
    )
    args = parser.parse_args()

    run_multi_task_success_quota(
        tasks=list_basic3_task_keys(),
        output_dir=args.output_dir,
        target_successes_per_task=args.success_quota,
        parallel=max(1, args.parallel),
        max_steps=args.max_steps,
        gen_meta_flag=True,
        policy_name="waypoint",
        max_total_episode_ids=args.max_total_episode_ids,
        num_gpus=max(0, args.num_gpus),
        first_gpu=args.first_gpu,
    )


if __name__ == "__main__":
    main()
