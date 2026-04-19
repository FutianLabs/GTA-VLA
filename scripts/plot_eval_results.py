#!/usr/bin/env python3
"""
Parse checkpoint evaluation logs and plot success rate and average steps.

Usage:
    # Single experiment
    python plot_eval_results.py <log_directory>
    
    # Compare two experiments
    python plot_eval_results.py <log_dir1> <log_dir2>
    
Example:
    python plot_eval_results.py /path/to/exp1/eval_logs_libero--episodes10 /path/to/exp2/eval_logs_libero--episodes10
"""

import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """
    Parse a single checkpoint log file and extract success rates and avg steps.
    
    Returns:
        dict: {
            'libero_spatial': {'success': float, 'steps': float},
            'libero_goal': {'success': float, 'steps': float},
            'libero_object': {'success': float, 'steps': float},
            'libero_10': {'success': float, 'steps': float},
            'overall': {'success': float, 'steps': float},
        }
    """
    results = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find the summary section
    # Pattern for individual settings: libero_spatial : 74.00% (7/10) | avg steps:  306.76
    pattern = r'^\s+(libero_\w+)\s+:\s+([\d.]+)%.*?avg steps:\s+([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE)
    
    for match in matches:
        setting_name, success_rate, avg_steps = match
        results[setting_name] = {
            'success': float(success_rate),
            'steps': float(avg_steps)
        }
    
    # Pattern for overall average: Overall Average : 66.75% (25/40) | avg steps:  399.29
    overall_pattern = r'^\s+Overall Average\s+:\s+([\d.]+)%.*?avg steps:\s+([\d.]+)'
    overall_match = re.search(overall_pattern, content, re.MULTILINE)
    
    if overall_match:
        results['overall'] = {
            'success': float(overall_match.group(1)),
            'steps': float(overall_match.group(2))
        }
    
    return results


def collect_all_results(log_dir):
    """
    Collect results from all checkpoint log files.
    
    Returns:
        dict: {
            checkpoint_iter: {
                'libero_spatial': {'success': float, 'steps': float},
                ...
            },
            ...
        }
    """
    all_results = {}
    
    # Find all ckpt-*.log files
    log_files = glob.glob(os.path.join(log_dir, 'ckpt-*.log'))
    
    for log_file in log_files:
        # Extract checkpoint iteration number
        basename = os.path.basename(log_file)
        match = re.match(r'ckpt-(\d+)\.log', basename)
        if match:
            ckpt_iter = int(match.group(1))
            results = parse_log_file(log_file)
            if results:  # Only add if we successfully parsed results
                all_results[ckpt_iter] = results
    
    return all_results


def plot_results_single(all_results, output_dir, title=None):
    """
    Plot success rate and average steps line charts for a single experiment.
    """
    # Sort checkpoints by iteration number
    ckpt_iters = sorted(all_results.keys())
    
    # Define settings to plot
    settings = ['libero_spatial', 'libero_goal', 'libero_object', 'libero_10', 'overall']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', '*']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Build title prefix
    title_prefix = f"[{title}] " if title else ""
    
    # ================== Plot 1: Success Rate ==================
    ax1 = axes[0]
    for setting, color, marker in zip(settings, colors, markers):
        success_rates = []
        valid_iters = []
        for ckpt in ckpt_iters:
            if setting in all_results[ckpt]:
                success_rates.append(all_results[ckpt][setting]['success'])
                valid_iters.append(ckpt)
        
        label = setting.replace('_', ' ').title() if setting != 'overall' else 'Overall Average'
        ax1.plot(valid_iters, success_rates, marker=marker, color=color, 
                 label=label, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Checkpoint Iteration', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title(f'{title_prefix}Success Rate vs Checkpoint', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 105)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    # ================== Plot 2: Average Steps ==================
    ax2 = axes[1]
    for setting, color, marker in zip(settings, colors, markers):
        avg_steps = []
        valid_iters = []
        for ckpt in ckpt_iters:
            if setting in all_results[ckpt]:
                avg_steps.append(all_results[ckpt][setting]['steps'])
                valid_iters.append(ckpt)
        
        label = setting.replace('_', ' ').title() if setting != 'overall' else 'Overall Average'
        ax2.plot(valid_iters, avg_steps, marker=marker, color=color, 
                 label=label, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Checkpoint Iteration', fontsize=12)
    ax2.set_ylabel('Average Steps', fontsize=12)
    ax2.set_title(f'{title_prefix}Average Steps vs Checkpoint', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'eval_plots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    pdf_path = os.path.join(output_dir, 'eval_plots.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path}")
    
    plt.close()


def plot_results_comparison(results1, results2, name1, name2, output_dir):
    """
    Plot comparison of two experiments with one subplot per setting.
    
    Args:
        results1: dict of checkpoint results for experiment 1
        results2: dict of checkpoint results for experiment 2
        name1: name of experiment 1
        name2: name of experiment 2
        output_dir: directory to save plots
    """
    # Define settings to plot
    settings = ['libero_spatial', 'libero_goal', 'libero_object', 'libero_10', 'overall']
    setting_titles = {
        'libero_spatial': 'LIBERO Spatial',
        'libero_goal': 'LIBERO Goal',
        'libero_object': 'LIBERO Object',
        'libero_10': 'LIBERO 10',
        'overall': 'Overall Average'
    }
    
    # Colors and markers for two experiments
    colors = ['#1f77b4', '#ff7f0e']  # Blue for exp1, Orange for exp2
    markers = ['o', 's']
    
    # Get all checkpoint iterations from both experiments
    all_ckpts = sorted(set(results1.keys()) | set(results2.keys()))
    
    # ================== Figure 1: Success Rate ==================
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    axes1 = axes1.flatten()
    
    for idx, setting in enumerate(settings):
        ax = axes1[idx]
        
        # Plot experiment 1
        success_rates1 = []
        valid_iters1 = []
        for ckpt in sorted(results1.keys()):
            if setting in results1[ckpt]:
                success_rates1.append(results1[ckpt][setting]['success'])
                valid_iters1.append(ckpt)
        
        if valid_iters1:
            ax.plot(valid_iters1, success_rates1, marker=markers[0], color=colors[0],
                    label=name1, linewidth=2, markersize=6)
        
        # Plot experiment 2
        success_rates2 = []
        valid_iters2 = []
        for ckpt in sorted(results2.keys()):
            if setting in results2[ckpt]:
                success_rates2.append(results2[ckpt][setting]['success'])
                valid_iters2.append(ckpt)
        
        if valid_iters2:
            ax.plot(valid_iters2, success_rates2, marker=markers[1], color=colors[1],
                    label=name2, linewidth=2, markersize=6)
        
        ax.set_xlabel('Checkpoint Iteration', fontsize=10)
        ax.set_ylabel('Success Rate (%)', fontsize=10)
        ax.set_title(setting_titles[setting], fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 105)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    # Hide the 6th subplot (we only have 5 settings)
    axes1[5].set_visible(False)
    
    fig1.suptitle('Success Rate Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save success rate figure
    output_path1 = os.path.join(output_dir, 'eval_comparison_success_rate.png')
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"Saved success rate comparison to: {output_path1}")
    
    pdf_path1 = os.path.join(output_dir, 'eval_comparison_success_rate.pdf')
    fig1.savefig(pdf_path1, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path1}")
    
    plt.close(fig1)
    
    # ================== Figure 2: Average Steps ==================
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    axes2 = axes2.flatten()
    
    for idx, setting in enumerate(settings):
        ax = axes2[idx]
        
        # Plot experiment 1
        avg_steps1 = []
        valid_iters1 = []
        for ckpt in sorted(results1.keys()):
            if setting in results1[ckpt]:
                avg_steps1.append(results1[ckpt][setting]['steps'])
                valid_iters1.append(ckpt)
        
        if valid_iters1:
            ax.plot(valid_iters1, avg_steps1, marker=markers[0], color=colors[0],
                    label=name1, linewidth=2, markersize=6)
        
        # Plot experiment 2
        avg_steps2 = []
        valid_iters2 = []
        for ckpt in sorted(results2.keys()):
            if setting in results2[ckpt]:
                avg_steps2.append(results2[ckpt][setting]['steps'])
                valid_iters2.append(ckpt)
        
        if valid_iters2:
            ax.plot(valid_iters2, avg_steps2, marker=markers[1], color=colors[1],
                    label=name2, linewidth=2, markersize=6)
        
        ax.set_xlabel('Checkpoint Iteration', fontsize=10)
        ax.set_ylabel('Average Steps', fontsize=10)
        ax.set_title(setting_titles[setting], fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    # Hide the 6th subplot
    axes2[5].set_visible(False)
    
    fig2.suptitle('Average Steps Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save average steps figure
    output_path2 = os.path.join(output_dir, 'eval_comparison_avg_steps.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved average steps comparison to: {output_path2}")
    
    pdf_path2 = os.path.join(output_dir, 'eval_comparison_avg_steps.pdf')
    fig2.savefig(pdf_path2, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path2}")
    
    plt.close(fig2)


def print_summary_table(all_results, name=None):
    """
    Print a summary table of all results.
    """
    ckpt_iters = sorted(all_results.keys())
    settings = ['libero_spatial', 'libero_goal', 'libero_object', 'libero_10', 'overall']
    
    prefix = f"[{name}] " if name else ""
    
    print("\n" + "=" * 100)
    print(f"📊 {prefix}SUMMARY: Success Rate (%) by Checkpoint")
    print("=" * 100)
    
    # Header
    header = f"{'Checkpoint':<12}" + "".join([f"{s:<18}" for s in settings])
    print(header)
    print("-" * 100)
    
    # Data rows
    for ckpt in ckpt_iters:
        row = f"{ckpt:<12}"
        for setting in settings:
            if setting in all_results[ckpt]:
                row += f"{all_results[ckpt][setting]['success']:<18.2f}"
            else:
                row += f"{'N/A':<18}"
        print(row)
    
    print("\n" + "=" * 100)
    print(f"📊 {prefix}SUMMARY: Average Steps by Checkpoint")
    print("=" * 100)
    
    # Header
    print(header)
    print("-" * 100)
    
    # Data rows
    for ckpt in ckpt_iters:
        row = f"{ckpt:<12}"
        for setting in settings:
            if setting in all_results[ckpt]:
                row += f"{all_results[ckpt][setting]['steps']:<18.2f}"
            else:
                row += f"{'N/A':<18}"
        print(row)
    
    print("=" * 100)


def extract_experiment_name(log_dir):
    """
    Extract experiment name from the log directory path.
    
    For a path like:
    /VLA-Data/.../logs/libero_scratch/xvla_fromscratch-12-04-14-44/eval_logs_libero--episodes10
    
    Returns: xvla_fromscratch-12-04-14-44
    """
    # Normalize path and get parent directory
    log_dir = os.path.normpath(log_dir)
    parent_dir = os.path.dirname(log_dir)
    experiment_name = os.path.basename(parent_dir)
    return experiment_name


def main():
    parser = argparse.ArgumentParser(
        description='Parse checkpoint evaluation logs and plot success rate and average steps.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single experiment
    python plot_eval_results.py /path/to/eval_logs_libero--episodes10
    
    # Compare two experiments
    python plot_eval_results.py /path/to/exp1/eval_logs_libero--episodes10 /path/to/exp2/eval_logs_libero--episodes10
    
    # With custom output directory
    python plot_eval_results.py /path/to/exp1 /path/to/exp2 --output /path/to/output
        """
    )
    parser.add_argument(
        'log_dirs',
        type=str,
        nargs='+',
        help='Path(s) to the directory containing ckpt-*.log files (1 or 2 paths)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for plots. Defaults to the first log directory.'
    )
    parser.add_argument(
        '--names',
        type=str,
        nargs='+',
        default=None,
        help='Custom names for the experiments. If not provided, uses parent directory names.'
    )
    
    args = parser.parse_args()
    
    if len(args.log_dirs) == 1:
        # Single experiment mode
        log_dir = os.path.abspath(args.log_dirs[0])
        name = args.names[0] if args.names else extract_experiment_name(log_dir)
        output_dir = args.output if args.output else log_dir
        
        print(f"Collecting results from: {log_dir}")
        print(f"Using name: {name}")
        
        all_results = collect_all_results(log_dir)
        
        if not all_results:
            print("No valid checkpoint logs found!")
            return
        
        print(f"Found {len(all_results)} checkpoint logs: {sorted(all_results.keys())}")
        
        print_summary_table(all_results, name)
        plot_results_single(all_results, output_dir, title=name)
        
    elif len(args.log_dirs) == 2:
        # Comparison mode
        log_dir1 = os.path.abspath(args.log_dirs[0])
        log_dir2 = os.path.abspath(args.log_dirs[1])
        
        if args.names and len(args.names) >= 2:
            name1, name2 = args.names[0], args.names[1]
        else:
            name1 = extract_experiment_name(log_dir1)
            name2 = extract_experiment_name(log_dir2)
        
        output_dir = args.output if args.output else os.path.dirname(log_dir1)
        
        print(f"Comparing two experiments:")
        print(f"  1. {name1}: {log_dir1}")
        print(f"  2. {name2}: {log_dir2}")
        print(f"Output directory: {output_dir}")
        
        # Collect results from both experiments
        results1 = collect_all_results(log_dir1)
        results2 = collect_all_results(log_dir2)
        
        if not results1:
            print(f"No valid checkpoint logs found in {log_dir1}!")
            return
        if not results2:
            print(f"No valid checkpoint logs found in {log_dir2}!")
            return
        
        print(f"\n{name1}: Found {len(results1)} checkpoint logs: {sorted(results1.keys())}")
        print(f"{name2}: Found {len(results2)} checkpoint logs: {sorted(results2.keys())}")
        
        # Print summary tables
        print_summary_table(results1, name1)
        print_summary_table(results2, name2)
        
        # Plot comparison
        plot_results_comparison(results1, results2, name1, name2, output_dir)
        
    else:
        print("Error: Please provide 1 or 2 log directories.")
        return
    
    print("\nDone!")


if __name__ == "__main__":
    main()
