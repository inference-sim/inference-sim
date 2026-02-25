#!/usr/bin/env python3
"""
Plot benchmark results from BLIS evaluation output.

Reads evaluation results JSON (from blis_evaluator.py) and generates:
- Per-workload E2E mean error bar charts
- Cross-workload metric comparison charts
- All-metrics heatmap across all experiments

Can be used standalone or called programmatically from blis_evaluator.py.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any


# Color palette for workload types
WORKLOAD_COLORS = {
    'code': '#FF6B6B',
    'chat': '#4ECDC4',
    'summ': '#FFD93D',
    'train': '#8B9DC3',
}
DEFAULT_COLOR = '#A8A8A8'

METRICS = ['ttft_mean_ms', 'ttft_p90_ms', 'itl_mean_ms', 'e2e_mean_ms', 'e2e_p90_ms']
METRIC_LABELS = ['TTFT Mean', 'TTFT P90', 'ITL Mean', 'E2E Mean', 'E2E P90']


def group_by_workload(results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Group evaluation experiments by workload type (app_type).

    Returns: {app_type: {experiment_name: {metric: error_pct, ...}}}
    """
    groups: Dict[str, Dict[str, Dict[str, float]]] = {}
    for exp in results.get('experiments', []):
        app_type = exp.get('app_type', 'unknown')
        exp_name = exp['experiment_name']
        if app_type not in groups:
            groups[app_type] = {}
        groups[app_type][exp_name] = exp['mean_percentage_errors']
    return groups


def _get_color(app_type: str) -> str:
    """Get color for a workload type, with fallback."""
    for key, color in WORKLOAD_COLORS.items():
        if key in app_type.lower():
            return color
    return DEFAULT_COLOR


def _shorten_label(name: str) -> str:
    """Shorten experiment name for plot labels by removing date prefixes."""
    for prefix in ['20260210-', '20260', 'jan30-', 'dec17-', 'feb']:
        name = name.replace(prefix, '')
    # Also strip workload suffix if present (already shown via grouping)
    for suffix in ['-chatsweep', '-codesweep', '-summarization']:
        name = name.replace(suffix, '')
    return name


def plot_e2e_mean_only(data: Dict[str, Dict[str, float]], title: str,
                       filename: str, color: str = '#4ECDC4'):
    """Bar chart showing E2E mean error for one workload type."""
    benchmarks = list(data.keys())
    values = [data[bench]['e2e_mean_ms'] for bench in benchmarks]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(benchmarks))
    bars = ax.bar(x, values, color=color, alpha=0.8, edgecolor='black', linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Benchmark Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('E2E Mean Error (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([_shorten_label(b) for b in benchmarks], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def plot_metric_comparison(workload_groups: Dict[str, Dict[str, Dict[str, float]]],
                           metric: str, metric_label: str, filename: str):
    """Grouped bar chart comparing a metric across workload types."""
    workload_types = sorted(workload_groups.keys())
    all_benchmarks = sorted(set(
        _shorten_label(bench)
        for wl in workload_groups.values() for bench in wl.keys()
    ))

    # Build a reverse mapping: shortened label -> original name per workload
    short_to_orig: Dict[str, Dict[str, str]] = {}
    for wl_type in workload_types:
        short_to_orig[wl_type] = {}
        for bench in workload_groups[wl_type]:
            short_to_orig[wl_type][_shorten_label(bench)] = bench

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_benchmarks))
    n_workloads = len(workload_types)
    bar_width = 0.8 / max(n_workloads, 1)

    for i, wl_type in enumerate(workload_types):
        offset = (i - (n_workloads - 1) / 2) * bar_width
        vals = []
        for short_name in all_benchmarks:
            orig = short_to_orig[wl_type].get(short_name)
            if orig and orig in workload_groups[wl_type]:
                vals.append(workload_groups[wl_type][orig].get(metric, 0))
            else:
                vals.append(0)

        color = _get_color(wl_type)
        bars = ax.bar(x + offset, vals, bar_width, label=wl_type.capitalize(),
                      color=color, alpha=0.8, edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Benchmark Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_label} - By Workload Type (Roofline v2)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_benchmarks, rotation=45, ha='right')
    ax.legend(framealpha=0.9, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def plot_all_metrics_heatmap(workload_groups: Dict[str, Dict[str, Dict[str, float]]],
                             filename: str):
    """Heatmap of all metrics across all experiments, grouped by workload."""
    experiments = []
    values = []

    for wl_type in sorted(workload_groups.keys()):
        suffix = wl_type[:4]
        for bench in sorted(workload_groups[wl_type].keys()):
            label = f'{_shorten_label(bench)} ({suffix})'
            experiments.append(label)
            values.append([workload_groups[wl_type][bench].get(m, 0) for m in METRICS])

    if not experiments:
        print("  No data for heatmap, skipping.")
        return

    data = np.array(values)
    fig, ax = plt.subplots(figsize=(10, max(6, len(experiments) * 0.8)))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(METRIC_LABELS)))
    ax.set_yticks(np.arange(len(experiments)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=11, fontweight='bold')
    ax.set_yticklabels(experiments, fontsize=10)

    for i in range(len(experiments)):
        for j in range(len(METRIC_LABELS)):
            val = data[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_title('Roofline v2 - Mean % Error by Experiment & Metric',
                 fontsize=14, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Error (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def plot_from_results(results: Dict[str, Any], output_dir: str):
    """
    Generate all plots from evaluation results.

    Args:
        results: Evaluation results dict (from blis_evaluator.evaluate_all())
        output_dir: Directory to save plot PNGs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    workload_groups = group_by_workload(results)

    if not workload_groups:
        print("No experiment data to plot.")
        return

    print(f"\nGenerating plots ({len(workload_groups)} workload types)...")

    # Per-workload E2E mean bar charts
    for wl_type, data in sorted(workload_groups.items()):
        color = _get_color(wl_type)
        plot_e2e_mean_only(
            data,
            f'{wl_type.capitalize()} - E2E Mean Error (Roofline v2)',
            str(output_path / f'{wl_type}_results.png'),
            color=color
        )

    # E2E comparison across workloads (only meaningful with 2+ types)
    if len(workload_groups) >= 2:
        plot_metric_comparison(
            workload_groups,
            'e2e_mean_ms', 'E2E Mean Error',
            str(output_path / 'e2e_comparison.png')
        )

    # Full heatmap
    plot_all_metrics_heatmap(
        workload_groups,
        str(output_path / 'all_metrics_heatmap.png')
    )

    print(f"All plots saved to {output_dir}")


def main():
    """Standalone CLI: load evaluation results JSON and generate plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot BLIS evaluation results")
    parser.add_argument(
        "results_json",
        help="Path to evaluation_results.json (from blis_evaluator.py --save)"
    )
    parser.add_argument(
        "--output-dir",
        default="eval/roofline_v2_results",
        help="Directory for output plots (default: eval/roofline_v2_results)"
    )

    args = parser.parse_args()

    with open(args.results_json, 'r') as f:
        results = json.load(f)

    plot_from_results(results, args.output_dir)


if __name__ == "__main__":
    main()
