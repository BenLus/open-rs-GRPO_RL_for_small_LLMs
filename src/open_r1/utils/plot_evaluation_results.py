#!/usr/bin/env python3
"""
Script to plot performance figures from evaluation results.

This script loads results from evaluation_results_summary.json and creates plots
showing performance across iterations for each model and task, with baseline comparisons.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import seaborn as sns

# --- Styling Configuration -------------------------------------------------
# Force white background
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': '#222222',
    'axes.linewidth': 1.2,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.facecolor': 'white',
})

# Custom qualitative palette (exclude yellow) - curated set
CUSTOM_COLORS = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#ff7f0e',  # orange (acceptable, not bright yellow)
    '#8c564b',  # brown
    '#17becf',  # teal
    '#7f7f7f',  # gray
]

sns.set_palette(CUSTOM_COLORS)

# Baseline line style
BASELINE_COLOR = '#444444'
BASELINE_STYLE = (0, (4, 4))  # dashed
BASELINE_WIDTH = 1.6

# Result line defaults
LINE_WIDTH = 3.0
MARKER_SIZE = 8
CAP_SIZE = 5
ALPHA_LINE = 0.9

# Define baseline values for Exp3Ben (given in the request)
EXP3BEN_BASELINES = {
    'aime24': 28.8 / 100.0,  # Convert percentages to decimals to match data format
    'math_500': 82.8 / 100.0,
    'amc23': 62.9 / 100.0,
    'minerva': 26.5 / 100.0,
    'olympiadbench': 43.3 / 100.0
}

# --- Utility Functions -----------------------------------------------------

def style_axes(ax):
    """Apply consistent styling to axes (box, ticks, grid)."""
    ax.grid(True, which='both', alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color('#222222')
    ax.tick_params(axis='both', labelsize=11)


def load_evaluation_results(file_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading results: {e}")
        return {}


def extract_performance_data(model_data: Dict[str, Any]) -> Dict[str, Tuple[List[int], List[float], List[float]]]:
    """Extract performance data for plotting."""
    task_data = {}
    first_iteration = next(iter(model_data.values()))
    task_names = list(first_iteration.keys())
    for task_name in task_names:
        iterations, performances, std_errors = [], [], []
        for iteration in sorted(model_data.keys(), key=int):
            if task_name in model_data[iteration]:
                tr = model_data[iteration][task_name]
                iterations.append(int(iteration))
                performances.append(tr.get('extractive_match', 0.0))
                std_errors.append(tr.get('extractive_match_stderr', 0.0))
        task_data[task_name] = (iterations, performances, std_errors)
    return task_data


def get_baseline_value(model_name: str, task_name: str, qwen3_orig_data: Dict[str, Any]) -> float:
    if model_name == "Exp3Ben":
        return EXP3BEN_BASELINES.get(task_name, 0.0)
    elif model_name == "Qwen3_trained":
        if qwen3_orig_data and task_name in qwen3_orig_data:
            return qwen3_orig_data[task_name].get('extractive_match', 0.0)
    return 0.0


def get_display_name(model_name: str) -> str:
    if model_name == "Exp3Ben":
        return "DeepSeek-R1-Distill-Qwen-1.5B_trained"
    elif model_name == "Qwen3_trained":
        return "Qwen3-1.7B_trained"
    return model_name

# --- Plotting Functions ----------------------------------------------------

def create_task_plot(model_name: str, task_name: str, iterations: List[int], performances: List[float], std_errors: List[float], baseline: float, output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    display_name = get_display_name(model_name)

    perf_pct = [p * 100 for p in performances]
    std_pct = [s * 100 for s in std_errors]
    base_pct = baseline * 100

    # Pick a deterministic color for this model+task based on name hash
    color = CUSTOM_COLORS[hash(task_name) % len(CUSTOM_COLORS)]

    # Add baseline point at x=0 if baseline exists
    if baseline > 0:
        plot_iterations = [0] + iterations
        plot_perf_pct = [base_pct] + perf_pct
        plot_std_pct = [0] + std_pct  # No error bar for baseline point
    else:
        plot_iterations = iterations
        plot_perf_pct = perf_pct
        plot_std_pct = std_pct

    ax.errorbar(
        plot_iterations, plot_perf_pct, yerr=plot_std_pct,
        marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        capsize=CAP_SIZE, color=color, alpha=ALPHA_LINE,
        label=f'{display_name}'
    )

    if baseline > 0:
        ax.axhline(base_pct, color=BASELINE_COLOR, linestyle=BASELINE_STYLE,
                   linewidth=BASELINE_WIDTH, alpha=0.5, label=f'Baseline: {base_pct:.1f}%')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Accuracy - Pass@1 (%)', fontsize=12)
    ax.set_title(f'{display_name} | {task_name.upper()}', fontsize=14)

    if plot_iterations:
        # Set x-ticks to include 0 and all iterations
        all_ticks = sorted(set(plot_iterations))
        ax.set_xticks(all_ticks)
        ax.set_xlim(-2, max(plot_iterations) + 5)

    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(max(0, y_min - pad), y_max + pad)

    style_axes(ax)
    ax.legend(fontsize=10)

    filename = f"{model_name}_{task_name}_performance.png"
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_model_overview_plot(model_name: str, task_data: Dict[str, Tuple], qwen3_orig_data: Dict[str, Any], output_dir: Path):
    display_name = get_display_name(model_name)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    task_names = list(task_data.keys())
    for i, (task_name, (iterations, performances, std_errors)) in enumerate(task_data.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        perf_pct = [p * 100 for p in performances]
        std_pct = [s * 100 for s in std_errors]
        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]

        # Add baseline point at x=0 if baseline exists
        baseline = get_baseline_value(model_name, task_name, qwen3_orig_data)
        if baseline > 0:
            base_pct = baseline * 100
            plot_iterations = [0] + iterations
            plot_perf_pct = [base_pct] + perf_pct
            plot_std_pct = [0] + std_pct  # No error bar for baseline point
        else:
            plot_iterations = iterations
            plot_perf_pct = perf_pct
            plot_std_pct = std_pct

        ax.errorbar(
            plot_iterations, plot_perf_pct, yerr=plot_std_pct,
            marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE-2,
            capsize=CAP_SIZE-1, color=color, alpha=ALPHA_LINE,
            label=task_name.upper()
        )
        
        if baseline > 0:
            bp = baseline * 100
            ax.axhline(bp, color=BASELINE_COLOR, linestyle=BASELINE_STYLE,
                       linewidth=BASELINE_WIDTH, alpha=0.4)
            ax.text(0.02, 0.95, f'Base {bp:.1f}%', transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='#555555', boxstyle='round,pad=0.25', alpha=0.85))

        ax.set_title(task_name.upper(), fontsize=12)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Accuracy - Pass@1 (%)', fontsize=10)
        if plot_iterations:
            # Set x-ticks to include 0 and all iterations
            all_ticks = sorted(set(plot_iterations))
            ax.set_xticks(all_ticks)
            ax.set_xlim(0, max(plot_iterations) + 5)
        style_axes(ax)

    for j in range(len(task_names), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{display_name} | All Tasks', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filename = f"{model_name}_overview.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_average_performance_plot(model_name: str, task_data: Dict[str, Tuple], 
                                   qwen3_orig_data: Dict[str, Any], output_dir: Path):
    """Create a plot showing average performance across all tasks for a model."""
    
    display_name = get_display_name(model_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all iterations from the first task (assuming all tasks have same iterations)
    first_task = next(iter(task_data.values()))
    iterations = first_task[0]  # iterations list
    
    # Calculate average performance for each iteration
    avg_performances = []
    avg_std_errors = []
    
    for i, iteration in enumerate(iterations):
        iter_performances = []
        iter_std_errors = []
        
        for task_name, (task_iterations, task_performances, task_std_errors) in task_data.items():
            if i < len(task_performances):
                iter_performances.append(task_performances[i] * 100)  # Convert to percentage
                iter_std_errors.append(task_std_errors[i] * 100)  # Convert to percentage
        
        if iter_performances:
            avg_performances.append(np.mean(iter_performances))
            # Propagate error as sqrt of sum of squares divided by N
            avg_std_errors.append(np.sqrt(np.sum(np.array(iter_std_errors)**2)) / len(iter_std_errors))
    
    # Calculate baseline average
    baseline_avg = 0
    baseline_count = 0
    for task_name in task_data.keys():
        baseline = get_baseline_value(model_name, task_name, qwen3_orig_data)
        if baseline > 0:
            baseline_avg += baseline * 100
            baseline_count += 1
    
    if baseline_count > 0:
        baseline_avg = baseline_avg / baseline_count
        # Add baseline point at x=0
        plot_iterations = [0] + iterations
        plot_performances = [baseline_avg] + avg_performances
        plot_std_errors = [0] + avg_std_errors  # No error for baseline
    else:
        plot_iterations = iterations
        plot_performances = avg_performances
        plot_std_errors = avg_std_errors
    
    # Plot average performance
    color = CUSTOM_COLORS[0]  # Use first color for consistency
    ax.errorbar(
        plot_iterations, plot_performances, yerr=plot_std_errors,
        marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        capsize=CAP_SIZE, color=color, alpha=ALPHA_LINE,
        label=f'{display_name} (Average)'
    )
    
    # Add baseline line if exists
    if baseline_count > 0:
        ax.axhline(baseline_avg, color=BASELINE_COLOR, linestyle=BASELINE_STYLE,
                   linewidth=BASELINE_WIDTH, alpha=0.5, 
                   label=f'Baseline Average: {baseline_avg:.1f}%')
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Accuracy - Pass@1 (%)', fontsize=12)
    ax.set_title(f'{display_name} | Average Performance Across All Tasks', fontsize=14)
    
    if plot_iterations:
        all_ticks = sorted(set(plot_iterations))
        ax.set_xticks(all_ticks)
        ax.set_xlim(0, max(plot_iterations) + 5)
    
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(max(0, y_min - pad), y_max + pad)
    
    style_axes(ax)
    ax.legend(fontsize=11)
    
    # Save plot
    filename = f"{model_name}_average_performance.png"
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")


def create_model_comparison_plot(results: Dict[str, Any], qwen3_orig_data: Dict[str, Any], output_dir: Path):
    if "Exp3Ben" not in results or "Qwen3_trained" not in results:
        print("Cannot create comparison plot - missing model data")
        return

    exp3ben_data = extract_performance_data(results["Exp3Ben"])
    qwen3_data = extract_performance_data(results["Qwen3_trained"])
    task_names = sorted(set(exp3ben_data.keys()) & set(qwen3_data.keys()))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()
    d_exp3 = get_display_name("Exp3Ben")
    d_qwen = get_display_name("Qwen3_trained")

    model_colors = {
        d_exp3: '#1f77b4',  # blue
        d_qwen: '#d62728',  # red
    }

    for i, task_name in enumerate(task_names):
        if i >= len(axes):
            break
        ax = axes[i]

        e_it, e_perf, e_std = exp3ben_data[task_name]
        q_it, q_perf, q_std = qwen3_data[task_name]
        e_perf_pct = [p * 100 for p in e_perf]
        e_std_pct = [s * 100 for s in e_std]
        q_perf_pct = [p * 100 for p in q_perf]
        q_std_pct = [s * 100 for s in q_std]

        # Add baseline points at x=0 for both models
        eb = get_baseline_value("Exp3Ben", task_name, qwen3_orig_data) * 100
        qb = get_baseline_value("Qwen3_trained", task_name, qwen3_orig_data) * 100
        
        if eb > 0:
            e_plot_it = [0] + e_it
            e_plot_perf = [eb] + e_perf_pct
            e_plot_std = [0] + e_std_pct  # No error bar for baseline
        else:
            e_plot_it = e_it
            e_plot_perf = e_perf_pct
            e_plot_std = e_std_pct
            
        if qb > 0:
            q_plot_it = [0] + q_it
            q_plot_perf = [qb] + q_perf_pct
            q_plot_std = [0] + q_std_pct  # No error bar for baseline
        else:
            q_plot_it = q_it
            q_plot_perf = q_perf_pct
            q_plot_std = q_std_pct

        ax.errorbar(e_plot_it, e_plot_perf, yerr=e_plot_std, marker='o', linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE-2, capsize=CAP_SIZE-1, color=model_colors[d_exp3],
                    alpha=ALPHA_LINE, label=d_exp3)
        ax.errorbar(q_plot_it, q_plot_perf, yerr=q_plot_std, marker='s', linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE-2, capsize=CAP_SIZE-1, color=model_colors[d_qwen],
                    alpha=ALPHA_LINE, label=d_qwen)

        if eb > 0:
            ax.axhline(eb, color=model_colors[d_exp3], linestyle=BASELINE_STYLE,
                       linewidth=BASELINE_WIDTH, alpha=0.4)
        if qb > 0:
            ax.axhline(qb, color=model_colors[d_qwen], linestyle=BASELINE_STYLE,
                       linewidth=BASELINE_WIDTH, alpha=0.4)

        ax.set_title(task_name.upper(), fontsize=12)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Accuracy - Pass@1 (%)', fontsize=10)
        if e_plot_it or q_plot_it:
            all_iterations = sorted(set(e_plot_it + q_plot_it))
            ax.set_xticks(all_iterations)
            ax.set_xlim(0, max(all_iterations) + 5)
        style_axes(ax)
        ax.legend(fontsize=9)

    for j in range(len(task_names), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Model Comparison | Accuracy - Pass@1', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filename = "model_comparison.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_average_performance_comparison_plot(results: Dict[str, Any], qwen3_orig_data: Dict[str, Any], output_dir: Path):
    """Create a comparison plot showing average performance across all tasks for both models."""
    
    if "Exp3Ben" not in results or "Qwen3_trained" not in results:
        print("Cannot create average performance comparison plot - missing model data")
        return
    
    exp3ben_data = extract_performance_data(results["Exp3Ben"])
    qwen3_data = extract_performance_data(results["Qwen3_trained"])
    
    # Get common task names
    task_names = sorted(set(exp3ben_data.keys()) & set(qwen3_data.keys()))
    if not task_names:
        print("No common tasks found between models")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    d_exp3 = get_display_name("Exp3Ben")
    d_qwen = get_display_name("Qwen3_trained")
    
    model_colors = {
        d_exp3: '#1f77b4',  # blue
        d_qwen: '#d62728',  # red
    }
    
    # Process Exp3Ben model
    exp3_iterations = exp3ben_data[task_names[0]][0]  # Get iterations from first task
    exp3_avg_performances = []
    exp3_avg_std_errors = []
    
    for i, iteration in enumerate(exp3_iterations):
        iter_performances = []
        iter_std_errors = []
        
        for task_name in task_names:
            if task_name in exp3ben_data:
                task_iterations, task_performances, task_std_errors = exp3ben_data[task_name]
                if i < len(task_performances):
                    iter_performances.append(task_performances[i] * 100)
                    iter_std_errors.append(task_std_errors[i] * 100)
        
        if iter_performances:
            exp3_avg_performances.append(np.mean(iter_performances))
            exp3_avg_std_errors.append(np.sqrt(np.sum(np.array(iter_std_errors)**2)) / len(iter_std_errors))
    
    # Calculate Exp3Ben baseline average
    exp3_baseline_avg = 0
    exp3_baseline_count = 0
    for task_name in task_names:
        baseline = get_baseline_value("Exp3Ben", task_name, qwen3_orig_data)
        if baseline > 0:
            exp3_baseline_avg += baseline * 100
            exp3_baseline_count += 1
    
    if exp3_baseline_count > 0:
        exp3_baseline_avg = exp3_baseline_avg / exp3_baseline_count
        exp3_plot_iterations = [0] + exp3_iterations
        exp3_plot_performances = [exp3_baseline_avg] + exp3_avg_performances
        exp3_plot_std_errors = [0] + exp3_avg_std_errors
    else:
        exp3_plot_iterations = exp3_iterations
        exp3_plot_performances = exp3_avg_performances
        exp3_plot_std_errors = exp3_avg_std_errors
    
    # Process Qwen3 model
    qwen3_iterations = qwen3_data[task_names[0]][0]  # Get iterations from first task
    qwen3_avg_performances = []
    qwen3_avg_std_errors = []
    
    for i, iteration in enumerate(qwen3_iterations):
        iter_performances = []
        iter_std_errors = []
        
        for task_name in task_names:
            if task_name in qwen3_data:
                task_iterations, task_performances, task_std_errors = qwen3_data[task_name]
                if i < len(task_performances):
                    iter_performances.append(task_performances[i] * 100)
                    iter_std_errors.append(task_std_errors[i] * 100)
        
        if iter_performances:
            qwen3_avg_performances.append(np.mean(iter_performances))
            qwen3_avg_std_errors.append(np.sqrt(np.sum(np.array(iter_std_errors)**2)) / len(iter_std_errors))
    
    # Calculate Qwen3 baseline average
    qwen3_baseline_avg = 0
    qwen3_baseline_count = 0
    for task_name in task_names:
        baseline = get_baseline_value("Qwen3_trained", task_name, qwen3_orig_data)
        if baseline > 0:
            qwen3_baseline_avg += baseline * 100
            qwen3_baseline_count += 1
    
    if qwen3_baseline_count > 0:
        qwen3_baseline_avg = qwen3_baseline_avg / qwen3_baseline_count
        qwen3_plot_iterations = [0] + qwen3_iterations
        qwen3_plot_performances = [qwen3_baseline_avg] + qwen3_avg_performances
        qwen3_plot_std_errors = [0] + qwen3_avg_std_errors
    else:
        qwen3_plot_iterations = qwen3_iterations
        qwen3_plot_performances = qwen3_avg_performances
        qwen3_plot_std_errors = qwen3_avg_std_errors
    
    # Plot both models
    ax.errorbar(
        exp3_plot_iterations, exp3_plot_performances, yerr=exp3_plot_std_errors,
        marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        capsize=CAP_SIZE, color=model_colors[d_exp3], alpha=ALPHA_LINE,
        label=f'{d_exp3} (Average)'
    )
    
    ax.errorbar(
        qwen3_plot_iterations, qwen3_plot_performances, yerr=qwen3_plot_std_errors,
        marker='s', linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        capsize=CAP_SIZE, color=model_colors[d_qwen], alpha=ALPHA_LINE,
        label=f'{d_qwen} (Average)'
    )
    
    # Add baseline lines
    if exp3_baseline_count > 0:
        ax.axhline(exp3_baseline_avg, color=model_colors[d_exp3], linestyle=BASELINE_STYLE,
                   linewidth=BASELINE_WIDTH, alpha=0.5, 
                   label=f'{d_exp3} Baseline: {exp3_baseline_avg:.1f}%')
    
    if qwen3_baseline_count > 0:
        ax.axhline(qwen3_baseline_avg, color=model_colors[d_qwen], linestyle=BASELINE_STYLE,
                   linewidth=BASELINE_WIDTH, alpha=0.5,
                   label=f'{d_qwen} Baseline: {qwen3_baseline_avg:.1f}%')
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Accuracy - Pass@1 (%)', fontsize=12)
    ax.set_title('Model Comparison | Average Performance Across All Tasks', fontsize=14, fontweight='bold')
    
    # Set x-axis
    all_iterations = sorted(set(exp3_plot_iterations + qwen3_plot_iterations))
    if all_iterations:
        ax.set_xticks(all_iterations)
        ax.set_xlim(0, max(all_iterations) + 5)
    
    # Set y-axis with padding
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(max(0, y_min - pad), y_max + pad)
    
    style_axes(ax)
    ax.legend(fontsize=11, loc='best')
    
    # Save plot
    filename = "model_average_performance_comparison.png"
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

# --- Main ------------------------------------------------------------------

def main():
    results_file = Path("/home/dsi/lustigb/RL_for_small_LLM/open-rs/evaluation_results_summary.json")
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    results = load_evaluation_results(results_file)
    if not results:
        print("No results loaded")
        return

    output_dir = Path("/home/dsi/lustigb/RL_for_small_LLM/open-rs/figures")
    output_dir.mkdir(exist_ok=True)
    print(f"Creating plots in: {output_dir}")

    qwen3_orig_data = {}
    if "Qwen3_orig" in results and "0" in results["Qwen3_orig"]:
        qwen3_orig_data = results["Qwen3_orig"]["0"]

    models_to_plot = ["Exp3Ben", "Qwen3_trained"]
    for model_name in models_to_plot:
        if model_name not in results:
            print(f"Model {model_name} not found in results")
            continue
        print(f"\nProcessing {model_name}...")
        task_data = extract_performance_data(results[model_name])
        if not task_data:
            print(f"No task data found for {model_name}")
            continue
        for task_name, (iterations, performances, std_errors) in task_data.items():
            if not iterations:
                continue
            baseline = get_baseline_value(model_name, task_name, qwen3_orig_data)
            create_task_plot(model_name, task_name, iterations, performances, std_errors, baseline, output_dir)
        create_model_overview_plot(model_name, task_data, qwen3_orig_data, output_dir)
        create_average_performance_plot(model_name, task_data, qwen3_orig_data, output_dir)

    create_model_comparison_plot(results, qwen3_orig_data, output_dir)
    create_average_performance_comparison_plot(results, qwen3_orig_data, output_dir)
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()