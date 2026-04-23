#!/usr/bin/env python3
"""Run evaluation across every checkpoint in an experiment directory.

Usage:
    python evaluate_all_widowx_checkpoints.py logs/bridge/<exp_dir> --task widowx [--output_root ...] -- [extra evaluator args]
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_eval_summary(output: str) -> Tuple[Dict[str, float], Optional[float]]:
    """
    Extract per-task and overall averages from evaluate_widowx.py stdout.
    """
    task_results: Dict[str, float] = {}
    overall_avg: Optional[float] = None
    pattern = re.compile(r"^\s*(?P<name>[A-Za-z0-9_ ]+):\s*(?P<pct>[0-9.]+)%", re.MULTILINE)
    for match in pattern.finditer(output):
        name = match.group("name").strip()
        pct = float(match.group("pct")) / 100.0
        if name.lower() == "overall average":
            overall_avg = pct
        else:
            task_results[name] = pct
    if overall_avg is None and task_results:
        overall_avg = sum(task_results.values()) / len(task_results)
    return task_results, overall_avg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints within an experiment directory.")
    parser.add_argument(
        "exp_dir",
        type=Path,
        help="Path to experiment directory containing ckpt-* folders (e.g., logs/bridge/reproduce_bridge-20251128-1118).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["widowx", "libero"],
        default="widowx",
        help="Which evaluation task to run.",
    )
    parser.add_argument(
        "--processor_path",
        default=None,
        help="Processor path forwarded to the evaluator (defaults depend on --task).",
    )
    parser.add_argument(
        "--evaluate_script",
        default=None,
        help="Path to the evaluator script (defaults to evaluate_widowx.py or evaluate_libero.py).",
    )

    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional root directory to store outputs; per-ckpt subfolders are appended automatically.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Directory to save evaluation logs (defaults to <exp_dir>/eval_logs).",
    )
    # Use parse_known_args to properly separate known args from extra args
    # This avoids the issue where argparse.REMAINDER captures --task before it's parsed
    args, extra = parser.parse_known_args()
    # Filter out '--' separator if present
    args.extra_args = [arg for arg in extra if arg != '--']
    return args


def find_checkpoints(exp_dir: Path) -> List[Path]:
    ckpts = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("ckpt-")]
    if not ckpts:
        raise FileNotFoundError(f"No ckpt-* folders found under {exp_dir}")

    def sort_key(path: Path) -> tuple:
        match = re.match(r"ckpt-(\d+)", path.name)
        step = int(match.group(1)) if match else None
        # Sort numeric checkpoints by descending step; fallback to mtime (newest first) otherwise.
        return (0, -step) if step is not None else (1, -path.stat().st_mtime)

    return sorted(ckpts, key=sort_key)


def run_and_capture(cmd: List[str], log_path: Path) -> Tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines: List[str] = []
    # Stream to console while writing to file and capturing for optional JSON.
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_lines.append(line)
        process.wait()
        returncode = process.returncode or 0
    return returncode, "".join(output_lines)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    default_processor = {
        "widowx": os.getenv("GTA_VLA_WIDOWX_PROCESSOR"),
        "libero": os.getenv("GTA_VLA_LIBERO_PROCESSOR"),
    }
    processor_path = os.path.expanduser(args.processor_path) if args.processor_path else default_processor[args.task]
    default_script = {
        "widowx": "evaluate_widowx",
        "libero": "evaluate_libero",
    }
    eval_script = Path(args.evaluate_script or default_script[args.task])
    if not eval_script.is_absolute():
        eval_script = Path(__file__).resolve().parent / eval_script
    extra_args = list(args.extra_args)
    # if extra_args and extra_args[0] == "--":
        # extra_args = extra_args[1:]

    log_dir_default = f"eval_logs_{args.task}"
    log_dir: Path = args.log_dir.expanduser() if args.log_dir else exp_dir / (log_dir_default + str("".join(extra_args)))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = find_checkpoints(exp_dir)
    print(f"Found {len(checkpoints)} checkpoints under {exp_dir}.")

    json_entries = []
    for ckpt in checkpoints:
        match = re.match(r"ckpt-(\d+)", ckpt.name)
        step = int(match.group(1)) if match else None
        cmd = [sys.executable, "-m", str(eval_script).replace("/", "."), "--model_path", str(ckpt)]
        if processor_path:
            cmd += ["--processor_path", processor_path]

        if args.output_root:
            out_dir = args.output_root.expanduser() / exp_dir.name / ckpt.name
            cmd += ["--output_dir", str(out_dir)]
        else:
            out_dir = exp_dir / "evaluation_outputs" / ckpt.name
            cmd += ["--output_dir", str(out_dir)]
        if extra_args:
            cmd += extra_args
        log_path = log_dir / f"{ckpt.name}.log"
        print("\n=== Evaluating", ckpt.name, "===")
        print("Running:", " ".join(shlex.quote(part) for part in cmd))
        returncode, captured = run_and_capture(cmd, log_path)

        task_results, overall_avg = parse_eval_summary(captured)
        
        # For libero, filter to only keep suite-level results (libero_spatial, libero_10, etc.)
        # and exclude individual task results
        if args.task == "libero":
            task_results = {
                name: score 
                for name, score in task_results.items() 
                if name.lower().startswith("libero")
            }
        
        json_entries.append(
            {
                "checkpoint": str(ckpt),
                "step": step,
                "task_avgs": task_results,
                "overall_avg": overall_avg,
            }
        )
        if returncode != 0:
            print(f"Checkpoint {ckpt.name} finished with non-zero exit code {returncode}.")

        if json_entries:
            json_path = log_dir / "eval_results.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(json_entries, f, indent=2)
            print(f"\nSaved JSON log to {json_path}")
        print(f"\nPer-checkpoint text logs stored in {log_dir}")

        # Plot evaluation curves
        if json_entries:
            valid_entries = sorted(
                [e for e in json_entries if e.get("step") is not None], key=lambda e: e["step"]
            )
            steps = [e["step"] for e in valid_entries]
            task_names = sorted({name for e in valid_entries for name in e.get("task_avgs", {})})

            plt.figure(figsize=(10, 6))
            for task in task_names:
                ys = [e["task_avgs"].get(task, float("nan")) for e in valid_entries]
                plt.plot(steps, ys, marker="o", label=task)

            overall_vals = [
                e["overall_avg"] if e.get("overall_avg") is not None else float("nan") for e in valid_entries
            ]
            plt.plot(steps, overall_vals, marker="o", linestyle="--", color="black", label="overall_avg")

            plt.xlabel("Training Iteration")
            plt.ylabel("Success Rate")
            plt.title(f"{args.task.capitalize()} Evaluation ({exp_dir.name})")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            plot_path = log_dir / "eval_results.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"\nSaved evaluation plot to {plot_path}")


if __name__ == "__main__":
    main()
