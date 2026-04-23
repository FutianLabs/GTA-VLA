#!/usr/bin/env python3
"""
Continuous checkpoint monitor + evaluator.

Behavior:
  - Evaluate checkpoints as they appear, from newest to oldest.
  - If final checkpoint exists: evaluate it first (regular + final eval), then older checkpoints.
  - If final checkpoint doesn't exist: evaluate from newest to oldest, then wait for new checkpoints.
  - Skip checkpoints that already have complete evaluation logs.
  
Usage:
  python auto_eval_on_checkpoint.py <checkpoint_dir> <final_step> <task> [options]

Examples:
  # Libero: evaluate each new ckpt, run final libero-plus eval at step 60000
  python auto_eval_on_checkpoint.py ~/logs/gtavla/libero_scratch/run1/ 60000 libero --max_tasks 50

  # Bridge: evaluate each new ckpt, stop at step 50000
  python auto_eval_on_checkpoint.py ~/logs/gtavla/bridge_exp/run1/ 50000 widowx
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# ==================== Configuration ====================

DEFAULT_PROCESSOR = {
    "widowx": os.getenv("GTA_VLA_WIDOWX_PROCESSOR"),
    "libero": os.getenv("GTA_VLA_LIBERO_PROCESSOR"),
    "simpler_google": os.getenv("GTA_VLA_SIMPLER_PROCESSOR"),
}

DEFAULT_EVAL_MODULE = {
    "widowx": "evaluation.scripts.evaluate_widowx",
    "libero": "evaluation.scripts.evaluate_libero",
    "simpler_google": "evaluation.scripts.evaluate_simpler_google",
}

TASK_SETTINGS = {
    "libero_regular": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    "libero_final": [
        "libero_plus_camera", "libero_plus_robot", "libero_plus_language",
        "libero_plus_light", "libero_plus_background", "libero_plus_noise", "libero_plus_layout",
    ],
    "widowx_regular": ["spoon_on_towel", "carrot_on_plate", "stack_cube", "put_eggplant_in_basket"],
    "simpler_google_regular": ["coke_can", "move_near", "open_close", "place_in"],
}


# ==================== Helper Functions ====================


def find_checkpoints(checkpoint_dir: Path) -> Dict[int, Path]:
    """Find all checkpoints in the directory."""
    ckpt_map = {}
    if checkpoint_dir.exists():
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and (match := re.match(r"ckpt-(\d+)", item.name)):
                ckpt_map[int(match.group(1))] = item
    return ckpt_map


def get_eval_dir(ckpt_path: Path) -> Path:
    return ckpt_path / "eval"


def get_log_path(ckpt_path: Path, eval_type: str = "regular") -> Path:
    """Get log file path for a checkpoint evaluation."""
    return get_eval_dir(ckpt_path) / "logs" / f"{eval_type}.log"


def is_evaluation_complete(log_path: Path) -> bool:
    """Check if an evaluation log indicates completion."""
    if not log_path.exists():
        return False
    try:
        content = log_path.read_text(encoding="utf-8")
        return bool(re.search(r"^\s*Overall Average\s*:", content, re.MULTILINE))
    except OSError:
        return False


def get_completed_evaluations(ckpt_map: Dict[int, Path], steps: List[int], eval_type: str = "regular") -> Set[int]:
    """Get set of steps that have completed evaluations."""
    return {s for s in steps if s in ckpt_map and is_evaluation_complete(get_log_path(ckpt_map[s], eval_type))}


def run_and_capture(cmd: List[str], log_path: Path) -> Tuple[int, str]:
    """Run command, stream output to console and log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines = []
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_lines.append(line)
        returncode = process.wait()
    return returncode, "".join(output_lines)


def parse_eval_summary(output: str) -> Tuple[Dict[str, float], Optional[float]]:
    """Parse evaluation output and extract task results."""
    task_results = {}
    overall_avg = None
    
    for line in output.split("\n"):
        # Match format: "  libero_spatial : 100.00% (1/1) | avg steps: 107.80"
        # or: "  widowx_spoon_on_towel : 75.00% (18/24)"
        # or simpler: "task_name: 85.5%"
        if match := re.match(r"^\s*([a-zA-Z_0-9]+)\s*:\s*([\d.]+)%", line):
            task_name = match.group(1)
            value = float(match.group(2))
            # Skip non-task lines like "steps" or other metadata
            if task_name not in ['steps', 'avg', 'Successful', 'Success']:
                # Remove widowx_ prefix for consistency with TASK_SETTINGS
                if task_name.startswith('widowx_'):
                    task_name = task_name[7:]  # Remove 'widowx_' prefix
                # Remove libero_ prefix for libero_10 -> libero_10 is OK, but widowx needs stripping
                task_results[task_name] = value
        # Match: "Overall Average : 97.50% (3/4) | avg steps: 169.28"
        elif re.match(r"^\s*Overall\s+(Average|Success\s+Rate)\s*:", line, re.IGNORECASE):
            if match := re.search(r":\s*([\d.]+)%", line):
                overall_avg = float(match.group(1))
    
    return task_results, overall_avg


def save_results_tsv(tsv_path: Path, step: int, task_results: Dict[str, float], 
                     overall_avg: Optional[float], task_order: List[str]) -> None:
    """Save results to TSV and Markdown files."""
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results
    existing = {}
    if tsv_path.exists():
        lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            header = lines[0].split("\t")
            for line in lines[1:]:
                if line.strip() and (values := line.split("\t")) and len(values) == len(header):
                    row = dict(zip(header, values))
                    if "step" in row:
                        try:
                            existing[int(row["step"])] = row
                        except ValueError:
                            pass
    
    # Add new row
    fmt = lambda v: f"{v:.2f}" if v is not None else ""
    existing[step] = {"step": str(step), "avg": fmt(overall_avg), 
                      **{t: fmt(task_results.get(t)) for t in task_order}}
    
    # Write TSV
    steps_sorted = sorted(existing.keys())
    header = ["step", "avg"] + task_order
    tsv_lines = ["\t".join(header)] + ["\t".join(existing[s].get(col, "") for col in header) for s in steps_sorted]
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    
    # Print Markdown table to console instead of saving to file
    print("\n" + "=" * 80)
    print("📊 Evaluation Results Summary")
    print("=" * 80)
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |"
    ] + ["| " + " | ".join(existing[s].get(col, "") for col in header) + " |" for s in steps_sorted]
    print("\n".join(md_lines))
    print("=" * 80 + "\n")


def build_eval_command(task: str, ckpt_path: Path, processor_path: Optional[str], output_dir: Path,
                       eval_module: str, extra_args: List[str]) -> List[str]:
    """Build evaluation command."""
    cmd = [sys.executable, "-m", eval_module, "--model_path", str(ckpt_path), "--output_dir", str(output_dir)]
    if task != "simpler_google" and processor_path:
        cmd += ["--processor_path", processor_path]
    return cmd + extra_args


def get_simpler_google_setting(eval_args: List[str]) -> str:
    for idx, arg in enumerate(eval_args):
        if arg == "--google_setting" and idx + 1 < len(eval_args):
            return eval_args[idx + 1]
    return "vm"


def build_final_eval_command(task: str, ckpt_path: Path, max_tasks: int,
                             final_eval_script: Optional[str], final_eval_args: List[str]) -> List[str]:
    """Build final evaluation command."""
    if task != "libero" and not final_eval_script:
        return []
    
    if not final_eval_script:
        return [sys.executable, "-m", "evaluation.scripts.evaluate_libero",
                "--model_path", str(ckpt_path), "--sim", "libero_plus",
                "--episodes", "1", "--max_tasks", str(max_tasks)] + final_eval_args
    else:
        cmd = ["bash", final_eval_script, str(ckpt_path)]
        if task == "libero":
            cmd += ["--max_tasks", str(max_tasks)]
        return cmd + final_eval_args


def git_commit_and_push(summary_dir: Path, exp_name: str, step: int, eval_type: str = "regular") -> bool:
    """
    Commit and push evaluation results to git.
    
    Args:
        summary_dir: Directory containing results files
        exp_name: Experiment name
        step: Checkpoint step
        eval_type: Type of evaluation ('regular' or 'final')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the git root (summary_root)
        git_root = summary_dir.parent.parent  # summary_dir is task/exp_name, so go up 2 levels
        
        if not (git_root / ".git").exists():
            print(f"⚠ Not a git repository: {git_root}")
            return False
        
        # Get relative paths for git add (only TSV files, markdown is printed to console)
        results_tsv = summary_dir / f"results_{eval_type}.tsv"
        
        # Make paths relative to git root
        rel_tsv = results_tsv.relative_to(git_root)
        
        # Git add - add each file individually for better error handling
        files_to_add = []
        for file_path in [rel_tsv]:
            add_cmd = ["git", "-C", str(git_root), "add", str(file_path)]
            result = subprocess.run(add_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                files_to_add.append(str(file_path))
            else:
                # Only warn if it's not an ignored file
                if "ignored" not in result.stderr.lower():
                    print(f"⚠ Git add failed for {file_path}: {result.stderr}")
        
        if not files_to_add:
            print(f"  ℹ No files to commit for checkpoint {step}")
            return True
        
        # Git commit
        commit_msg = f"Update {exp_name} checkpoint {step} {eval_type} evaluation"
        commit_cmd = ["git", "-C", str(git_root), "commit", "-m", commit_msg]
        result = subprocess.run(commit_cmd, capture_output=True, text=True)
        
        # Check if there was nothing to commit
        if result.returncode != 0:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                print(f"  ℹ No changes to commit for checkpoint {step}")
                return True
            else:
                print(f"⚠ Git commit failed: {result.stderr}")
                return False
        
        # Git pull before push to avoid conflicts
        pull_cmd = ["git", "-C", str(git_root), "pull", "--rebase", "origin", "main"]
        result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            # Pull failed, but we already committed locally - warn user
            print(f"⚠ Git pull failed: {result.stderr}")
            print(f"  Local commit created but not pushed. Please resolve manually.")
            return False
        
        # Git push
        push_cmd = ["git", "-C", str(git_root), "push", "origin", "main"]
        result = subprocess.run(push_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"⚠ Git push failed: {result.stderr}")
            print(f"  Local commit created but not pushed. Please resolve manually.")
            return False
        
        print(f"  ✓ Git: committed and pushed checkpoint {step} results")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⚠ Git operation timed out")
        return False
    except Exception as e:
        print(f"⚠ Git operation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_wandb(args: argparse.Namespace, exp_name: str, task: str, checkpoint_dir: Path):
    """Initialize wandb if enabled."""
    if args.no_wandb:
        return None
    
    try:
        import wandb
        
        # Load run ID from checkpoint directory
        run_id_file = checkpoint_dir / "wandb_run_id.txt"
        run_id = run_id_file.read_text(encoding="utf-8").strip() if run_id_file.exists() else None
        
        tags = list(args.wandb_tag) if args.wandb_tag else []
        tags.extend(["auto_eval", task])
        
        if run_id:
            print(f"Resuming wandb run: {run_id}")
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                            id=run_id, resume="allow", tags=tags)
        else:
            print(f"Creating new wandb run")
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                            name=exp_name, tags=tags)
            # Save the new run ID for future use
            if run and run.id:
                run_id_file.write_text(run.id, encoding="utf-8")
                print(f"Saved wandb run ID to {run_id_file}")
        
        return run
    except (ImportError, Exception) as e:
        print(f"⚠ wandb initialization failed: {e}")
        return None


# ==================== Main Evaluation Logic ====================


def run_evaluation(step: int, ckpt_path: Path, eval_type: str, task: str,
                  processor_path: str, eval_module: str, eval_args: List[str],
                  output_dir: Path, wandb_run,
                  max_tasks: int = 0, final_eval_script: Optional[str] = None, 
                  final_eval_args: List[str] = []) -> bool:
    """
    Run evaluation for a checkpoint (regular or final).
    
    Returns True if evaluation completed successfully.
    """
    summary_dir = get_eval_dir(ckpt_path)
    log_path = get_log_path(ckpt_path, eval_type)
    
    # Check if already completed
    if is_evaluation_complete(log_path):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint {step} already evaluated ({eval_type}). Skipping.")
        return True
    
    # Build command
    if eval_type == "final":
        cmd = build_final_eval_command(task, ckpt_path, max_tasks, final_eval_script, final_eval_args)
        if not cmd:
            return False
    else:
        cmd = build_eval_command(task, ckpt_path, processor_path, output_dir, eval_module, eval_args)
    
    # Run evaluation
    print(f"\n{'='*80}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Evaluating checkpoint {step} ({eval_type})")
    print(f"Command: {' '.join(shlex.quote(part) for part in cmd)}")
    print(f"{'='*80}\n")
    
    returncode, captured = run_and_capture(cmd, log_path)
    
    if returncode != 0:
        print(f"⚠ Eval command exited with code {returncode}")
        return False
    
    # Parse and save results
    task_results, overall_avg = parse_eval_summary(captured)
    if not task_results or overall_avg is None:
        print("⚠ Failed to parse evaluation results")
        return False
    
    # Save to TSV
    task_key = f"{task}_{eval_type}" if eval_type == "final" else f"{task}_regular"
    task_order = TASK_SETTINGS.get(task_key, [])
    results_tsv = summary_dir / f"results_{eval_type}.tsv"
    save_results_tsv(results_tsv, step, task_results, overall_avg, task_order)
    
    # Log to wandb
    if wandb_run:
        prefix = "eval_final" if eval_type == "final" else "eval"
        log_dict = {f"{prefix}/{k}": v for k, v in task_results.items()}
        log_dict[f"{prefix}/overall_avg"] = overall_avg
        log_dict["step"] = step
        wandb_run.log(log_dict)
    
    print(f"\n✅ Checkpoint {step} {eval_type} evaluation completed. Overall: {overall_avg:.2f}%\n")
    
    # Git commit and push
    exp_name = summary_dir.name
    git_commit_and_push(summary_dir, exp_name, step, eval_type)
    
    return True


def evaluate_checkpoint(step: int, ckpt_map: Dict[int, Path], config: dict) -> bool:
    """Evaluate a single checkpoint (regular evaluation only)."""
    ckpt_path = ckpt_map[step]
    output_dir = get_eval_dir(ckpt_path) / "videos"
    
    return run_evaluation(
        step, ckpt_path, "regular", config['task'],
        config['processor_path'], config['eval_module'], config['eval_args'],
        output_dir, config['wandb_run']
    )


def evaluate_final_checkpoint(step: int, ckpt_map: Dict[int, Path], config: dict) -> Tuple[bool, bool]:
    """
    Evaluate final checkpoint (both regular and final evaluation).
    
    Returns (regular_success, final_success).
    """
    ckpt_path = ckpt_map[step]
    output_dir = get_eval_dir(ckpt_path) / "videos"
    
    # Regular evaluation
    regular_success = run_evaluation(
        step, ckpt_path, "regular", config['task'],
        config['processor_path'], config['eval_module'], config['eval_args'],
        output_dir, config['wandb_run']
    )
    
    # Final evaluation
    final_success = False
    if not config['skip_final_eval']:
        final_success = run_evaluation(
            step, ckpt_path, "final", config['task'],
            config['processor_path'], config['eval_module'], config['eval_args'],
            output_dir, config['wandb_run'],
            config['max_tasks'], config['final_eval_script'], config['final_eval_args']
        )
    
    return regular_success, final_success


# ==================== Main ====================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor checkpoint directory and automatically run evaluation on new checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint_dir", type=str, help="Path to the checkpoint directory to monitor.")
    parser.add_argument("final_step", type=int, nargs="?", default=-1,
                        help="Final checkpoint step (omit or set -1 to monitor continuously).")
    parser.add_argument("task", type=str, choices=["libero", "widowx", "simpler_google"], help="Task type.")
    parser.add_argument("--max_tasks", type=int, default=10, help="Maximum number of tasks for final libero-plus eval.")
    parser.add_argument("--eval_module", type=str, help="Python module for evaluation (default depends on task).")
    parser.add_argument("--processor_path", type=str, help="Path to processor (default depends on task).")
    parser.add_argument("--wait_before_eval", type=int, default=10, 
                       help="Time to wait (in seconds) after checkpoint is detected before starting evaluation.")
    parser.add_argument("--poll_interval", type=int, default=300,
                       help="Time to wait (in seconds) between checking for new checkpoints.")
    parser.add_argument("--output_root", type=str,
                       help="Deprecated. Evaluation outputs are now saved under each checkpoint's eval/ folder.")
    parser.add_argument("--output_layout", type=str, default="nested", choices=["nested", "flat"],
                       help="Deprecated. Evaluation outputs are now saved under each checkpoint's eval/ folder.")
    parser.add_argument("--summary_root", type=str, default=None,
                       help="Deprecated. Evaluation summaries are now saved under each checkpoint's eval/ folder.")
    parser.add_argument("--skip_final_eval", action="store_true",
                       help="Skip final evaluation even when final checkpoint is reached.")
    parser.add_argument("--final_eval_script", type=str,
                       help="Custom script for final evaluation (default: use built-in libero_plus eval).")
    parser.add_argument("--eval_arg", action="append", default=[],
                       help="Additional argument for evaluation script (repeatable).")
    parser.add_argument("--final_eval_arg", action="append", default=[],
                       help="Additional argument for final evaluation script (repeatable).")
    parser.add_argument("--wandb_project", type=str, help="WandB project name.")
    parser.add_argument("--wandb_entity", type=str, help="WandB entity.")
    parser.add_argument("--wandb_tag", action="append", default=[], help="Wandb tag (repeatable).")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--try_run", action="store_true",
                       help="Print the evaluation commands that would run and exit.")
    
    args = parser.parse_args()
    
    # Setup configuration
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    task = args.task
    exp_name = checkpoint_dir.name
    if args.output_root or args.summary_root:
        print("⚠ --output_root / --summary_root are ignored. Results are now saved under each checkpoint's eval/ folder.")
    if args.output_layout != "nested":
        print("⚠ --output_layout is ignored. Results are now saved under each checkpoint's eval/ folder.")
    
    config = {
        'checkpoint_dir': checkpoint_dir,
        'task': task,
        'exp_name': exp_name,
        'eval_module': args.eval_module or DEFAULT_EVAL_MODULE[task],
        'processor_path': os.path.expanduser(args.processor_path) if args.processor_path else DEFAULT_PROCESSOR[task],
        'eval_args': list(args.eval_arg),
        'max_tasks': args.max_tasks,
        'final_eval_script': args.final_eval_script,
        'final_eval_args': list(args.final_eval_arg),
        'skip_final_eval': args.skip_final_eval,
        'wandb_run': None,
    }
    
    # Find existing checkpoints
    ckpt_map = find_checkpoints(checkpoint_dir)
    existing_steps = sorted(ckpt_map.keys(), reverse=True)  # Newest to oldest
    
    if not existing_steps:
        print(f"No checkpoints found in {checkpoint_dir}. Waiting for checkpoints...")
    
    final_step = args.final_step
    has_final_checkpoint = final_step >= 0 and final_step in ckpt_map
    
    # Get completed evaluations
    completed_regular = get_completed_evaluations(ckpt_map, existing_steps, "regular")
    completed_final = get_completed_evaluations(ckpt_map, [final_step], "final") if final_step >= 0 else set()
    
    # Dry run mode
    if args.try_run:
        print("DRY RUN MODE - Commands that would be executed:\n")
        for step in existing_steps:
            if step not in completed_regular:
                ckpt_path = ckpt_map[step]
                out_dir = get_eval_dir(ckpt_path) / "videos"
                cmd = build_eval_command(task, ckpt_path, config['processor_path'], out_dir, config['eval_module'], config['eval_args'])
                print(f"[Step {step} - Regular] {' '.join(shlex.quote(part) for part in cmd)}\n")
        
        if final_step >= 0 and has_final_checkpoint and final_step not in completed_final and not args.skip_final_eval:
            ckpt_path = ckpt_map[final_step]
            cmd = build_final_eval_command(task, ckpt_path, args.max_tasks, args.final_eval_script, args.final_eval_arg)
            if cmd:
                print(f"[Step {final_step} - Final] {' '.join(shlex.quote(part) for part in cmd)}\n")
        return
    
    # Initialize wandb
    config['wandb_run'] = init_wandb(args, exp_name, task, checkpoint_dir)
    
    print(f"\n{'='*80}")
    print(f"Auto Evaluation Monitor")
    print(f"{'='*80}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Task: {task}")
    print(f"Final step: {final_step if final_step >= 0 else 'disabled (monitor forever)'}")
    print(f"Eval layout: <checkpoint>/eval/")
    print(f"Existing checkpoints: {len(existing_steps)}")
    print(f"Final checkpoint exists: {has_final_checkpoint}")
    print(f"{'='*80}\n")
    
    # Main evaluation loop
    try:
        if has_final_checkpoint:
            # Case A: Final checkpoint exists - evaluate it first, then older checkpoints
            if final_step not in completed_regular or (final_step not in completed_final and not args.skip_final_eval):
                reg_success, fin_success = evaluate_final_checkpoint(final_step, ckpt_map, config)
                if reg_success:
                    completed_regular.add(final_step)
                if fin_success:
                    completed_final.add(final_step)
            
            # Evaluate older checkpoints
            for step in existing_steps:
                if step != final_step and step not in completed_regular:
                    if evaluate_checkpoint(step, ckpt_map, config):
                        completed_regular.add(step)
            
            print(f"\n{'='*80}\nAll evaluations completed!\n{'='*80}\n")
            return
        
        else:
            # Case B: Final checkpoint doesn't exist - evaluate existing, then wait for new ones
            for step in existing_steps:
                if step not in completed_regular:
                    if evaluate_checkpoint(step, ckpt_map, config):
                        completed_regular.add(step)
            
            # Wait for new checkpoints
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for new checkpoints...")
            last_check_time = time.time()
            
            while True:
                time.sleep(args.poll_interval)
                
                ckpt_map = find_checkpoints(checkpoint_dir)
                all_steps = sorted(ckpt_map.keys(), reverse=True)
                new_steps = [s for s in all_steps if s not in completed_regular]
                
                if not new_steps:
                    elapsed = time.time() - last_check_time
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No new checkpoints (checked {elapsed:.0f}s ago)...")
                    continue
                
                last_check_time = time.time()
                
                for step in new_steps:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] New checkpoint detected: {step}")
                    print(f"Waiting {args.wait_before_eval}s before starting evaluation...")
                    time.sleep(args.wait_before_eval)
                    
                    if final_step >= 0 and step == final_step:
                        reg_success, fin_success = evaluate_final_checkpoint(step, ckpt_map, config)
                        if reg_success:
                            completed_regular.add(step)
                        if fin_success:
                            completed_final.add(step)
                        print(f"\n{'='*80}\nFinal checkpoint reached and evaluated. Exiting.\n{'='*80}\n")
                        return
                    else:
                        if evaluate_checkpoint(step, ckpt_map, config):
                            completed_regular.add(step)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    finally:
        if config['wandb_run']:
            config['wandb_run'].finish()


if __name__ == "__main__":
    main()
