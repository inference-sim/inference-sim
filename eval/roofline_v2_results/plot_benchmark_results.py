#!/usr/bin/env python3
"""
Plot benchmark results for roofline v2 evaluation.
Updated with latest evaluation results using hardware_config_roofline_valid.json.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- New evaluation results (roofline v2 with minimal hardware config) ---

codesweep_data = {
    '20260210-codellama-34b-tp2': {
        'ttft_mean_ms': 87.42,
        'ttft_p90_ms': 88.73,
        'itl_mean_ms': 57.01,
        'e2e_mean_ms': 45.14,
        'e2e_p90_ms': 48.21
    },
    '20260210-llama2-70b-tp4': {
        'ttft_mean_ms': 88.09,
        'ttft_p90_ms': 88.78,
        'itl_mean_ms': 54.82,
        'e2e_mean_ms': 42.60,
        'e2e_p90_ms': 45.33
    },
    '20260210-qwen3-14b-tp1': {
        'ttft_mean_ms': 83.53,
        'ttft_p90_ms': 86.25,
        'itl_mean_ms': 54.70,
        'e2e_mean_ms': 43.19,
        'e2e_p90_ms': 47.64
    },
    'jan30-llama2-7b-tp1': {
        'ttft_mean_ms': 85.36,
        'ttft_p90_ms': 85.73,
        'itl_mean_ms': 51.43,
        'e2e_mean_ms': 38.07,
        'e2e_p90_ms': 43.47
    },
    'jan30-llama2-7b-tp2': {
        'ttft_mean_ms': 90.61,
        'ttft_p90_ms': 91.18,
        'itl_mean_ms': 69.39,
        'e2e_mean_ms': 61.41,
        'e2e_p90_ms': 66.64
    },
    'jan30-llama2-7b-tp4': {
        'ttft_mean_ms': 94.89,
        'ttft_p90_ms': 95.03,
        'itl_mean_ms': 80.08,
        'e2e_mean_ms': 75.54,
        'e2e_p90_ms': 77.76
    }
}

chatsweep_data = {
    '20260210-codellama-34b-tp2': {
        'ttft_mean_ms': 19.40,
        'ttft_p90_ms': 15.76,
        'itl_mean_ms': 31.54,
        'e2e_mean_ms': 31.31,
        'e2e_p90_ms': 30.56
    },
    '20260210-llama2-70b-tp4': {
        'ttft_mean_ms': 30.93,
        'ttft_p90_ms': 19.98,
        'itl_mean_ms': 20.40,
        'e2e_mean_ms': 19.14,
        'e2e_p90_ms': 18.55
    },
    '20260210-qwen3-14b-tp2': {
        'ttft_mean_ms': 47.43,
        'ttft_p90_ms': 36.98,
        'itl_mean_ms': 20.54,
        'e2e_mean_ms': 20.35,
        'e2e_p90_ms': 21.50
    },
    'jan30-llama2-7b-tp1': {
        'ttft_mean_ms': 33.78,
        'ttft_p90_ms': 26.73,
        'itl_mean_ms': 31.82,
        'e2e_mean_ms': 31.66,
        'e2e_p90_ms': 32.55
    },
    'jan30-llama2-7b-tp2': {
        'ttft_mean_ms': 68.53,
        'ttft_p90_ms': 66.18,
        'itl_mean_ms': 56.03,
        'e2e_mean_ms': 56.07,
        'e2e_p90_ms': 56.58
    },
    'jan30-llama2-7b-tp4': {
        'ttft_mean_ms': 80.18,
        'ttft_p90_ms': 76.01,
        'itl_mean_ms': 60.96,
        'e2e_mean_ms': 61.28,
        'e2e_p90_ms': 58.77
    }
}

# Summarization experiment (standalone)
summarization_data = {
    'dec17-tp1-qwen7': {
        'ttft_mean_ms': 84.92,
        'ttft_p90_ms': 78.89,
        'itl_mean_ms': 21.67,
        'e2e_mean_ms': 27.96,
        'e2e_p90_ms': 26.88
    }
}

OUTPUT_DIR = 'eval/roofline_v2_results'


def plot_e2e_mean_only(data, title, filename, color='#4ECDC4'):
    """Create a bar chart showing only E2E mean error."""
    benchmarks = list(data.keys())
    values = [data[bench]['e2e_mean_ms'] for bench in benchmarks]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(benchmarks))
    bars = ax.bar(x, values, color=color, alpha=0.8, edgecolor='black', linewidth=1.2)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Benchmark Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('E2E Mean Error (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('20260210-', '').replace('jan30-', '').replace('dec17-', '')
                         for b in benchmarks],
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close()


def plot_metric_comparison(codesweep_data, chatsweep_data, metric, metric_label, filename):
    """Create a comparison plot for a specific metric across both workloads."""
    benchmarks = sorted(set(list(codesweep_data.keys()) + list(chatsweep_data.keys())))

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(benchmarks))
    bar_width = 0.35

    code_vals = [codesweep_data[b].get(metric, 0) if b in codesweep_data else 0 for b in benchmarks]
    chat_vals = [chatsweep_data[b].get(metric, 0) if b in chatsweep_data else 0 for b in benchmarks]

    bars_code = ax.bar(x - bar_width/2, code_vals, bar_width, label='Codesweep', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.8)
    bars_chat = ax.bar(x + bar_width/2, chat_vals, bar_width, label='Chatsweep', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.8)

    # Add value labels
    for bar, val in zip(bars_code, code_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars_chat, chat_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Benchmark Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_label} - Codesweep vs Chatsweep (Roofline v2)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('20260210-', '').replace('jan30-', '') for b in benchmarks],
                       rotation=45, ha='right')
    ax.legend(framealpha=0.9, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close()


def plot_all_metrics_heatmap(codesweep_data, chatsweep_data, summarization_data, filename):
    """Create a heatmap showing all metrics for all experiments."""
    metrics = ['ttft_mean_ms', 'ttft_p90_ms', 'itl_mean_ms', 'e2e_mean_ms', 'e2e_p90_ms']
    metric_labels = ['TTFT Mean', 'TTFT P90', 'ITL Mean', 'E2E Mean', 'E2E P90']

    # Build experiment list with workload type suffix
    experiments = []
    values = []

    for bench in sorted(chatsweep_data.keys()):
        label = bench.replace('20260210-', '').replace('jan30-', '') + ' (chat)'
        experiments.append(label)
        values.append([chatsweep_data[bench][m] for m in metrics])

    for bench in sorted(codesweep_data.keys()):
        label = bench.replace('20260210-', '').replace('jan30-', '') + ' (code)'
        experiments.append(label)
        values.append([codesweep_data[bench][m] for m in metrics])

    for bench in sorted(summarization_data.keys()):
        label = bench.replace('dec17-', '') + ' (summ)'
        experiments.append(label)
        values.append([summarization_data[bench][m] for m in metrics])

    data = np.array(values)

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(experiments)))
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    ax.set_yticklabels(experiments, fontsize=10)

    # Add text annotations
    for i in range(len(experiments)):
        for j in range(len(metric_labels)):
            val = data[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_title('Roofline v2 - Mean % Error by Experiment & Metric', fontsize=14, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Error (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close()


if __name__ == "__main__":
    # E2E mean error bar charts per workload type
    plot_e2e_mean_only(
        codesweep_data,
        'Codesweep - E2E Mean Error (Roofline v2)',
        f'{OUTPUT_DIR}/codesweep_results.png',
        color='#FF6B6B'
    )

    plot_e2e_mean_only(
        chatsweep_data,
        'Chatsweep - E2E Mean Error (Roofline v2)',
        f'{OUTPUT_DIR}/chatsweep_results.png',
        color='#4ECDC4'
    )

    # E2E comparison: codesweep vs chatsweep
    plot_metric_comparison(
        codesweep_data, chatsweep_data,
        'e2e_mean_ms', 'E2E Mean Error',
        f'{OUTPUT_DIR}/e2e_comparison.png'
    )

    # Full heatmap of all metrics x all experiments
    plot_all_metrics_heatmap(
        codesweep_data, chatsweep_data, summarization_data,
        f'{OUTPUT_DIR}/all_metrics_heatmap.png'
    )

    print("\nAll plots generated successfully!")
