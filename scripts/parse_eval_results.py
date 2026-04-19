#!/usr/bin/env python3
"""
Parse evaluation log files and convert results to structured JSON format.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any


def parse_evaluation_log(log_file: Path) -> Dict[str, Any]:
    """
    Parse an evaluation log file and extract results.
    
    Expected log format:
        INFO:evaluate:Evaluation finished: {'task_name': success_rate, ...}
    """
    results = {
        "checkpoint": log_file.stem.replace("_log", ""),
        "status": "unknown",
        "tasks": {},
        "average_success_rate": 0.0
    }
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Look for the evaluation finished line
        pattern = r"Evaluation finished:\s*(\{[^}]+\})"
        match = re.search(pattern, log_content)
        
        if match:
            # Extract the dictionary string and parse it
            results_str = match.group(1)
            # Clean up the string and convert to proper JSON format
            results_str = results_str.replace("'", '"')
            task_results = json.loads(results_str)
            
            results["tasks"] = task_results
            results["status"] = "completed"
            
            # Calculate average success rate
            if task_results:
                results["average_success_rate"] = sum(task_results.values()) / len(task_results)
        else:
            results["status"] = "failed"
            
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser("Parse evaluation logs and generate JSON reports")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing evaluation logs")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file (default: log_dir/results.json)")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_file = Path(args.output) if args.output else log_dir / "results.json"
    
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return
    
    # Find all log files
    log_files = sorted(log_dir.glob("checkpoint-*_log.txt"))
    
    if not log_files:
        print(f"No evaluation log files found in {log_dir}")
        return
    
    print(f"Found {len(log_files)} evaluation log files")
    
    # Parse all logs
    all_results = []
    for log_file in log_files:
        print(f"Parsing {log_file.name}...")
        result = parse_evaluation_log(log_file)
        all_results.append(result)
    
    # Create summary
    summary = {
        "total_checkpoints": len(all_results),
        "successful_evaluations": sum(1 for r in all_results if r["status"] == "completed"),
        "checkpoints": all_results
    }
    
    # Find best checkpoint
    completed_results = [r for r in all_results if r["status"] == "completed"]
    if completed_results:
        best = max(completed_results, key=lambda x: x["average_success_rate"])
        summary["best_checkpoint"] = {
            "name": best["checkpoint"],
            "average_success_rate": best["average_success_rate"],
            "tasks": best["tasks"]
        }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total checkpoints evaluated: {summary['total_checkpoints']}")
    print(f"Successful evaluations: {summary['successful_evaluations']}")
    
    if "best_checkpoint" in summary:
        best = summary["best_checkpoint"]
        print(f"\nBest Checkpoint: {best['name']}")
        print(f"Average Success Rate: {best['average_success_rate']:.2%}")
        print("\nTask Results:")
        for task, rate in sorted(best['tasks'].items()):
            print(f"  {task}: {rate:.2%}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()


