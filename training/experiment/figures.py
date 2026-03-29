"""Publication figures for sim-to-real accuracy validation.

Generates 5 figures + 1 LaTeX table from error_records.csv and runtime.csv.
Spec: docs/superpowers/specs/2026-03-16-publication-figures-design.md
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDED_SIMULATORS = frozenset({"blis-blackbox", "blis-crossmodel", "vidur"})

SIMULATOR_ORDER = [
    "blis-trained-roofline",
    "blis-roofline",
    "vidur",
    "llm-optimizer-estimate",
    "aiconfigurator-estimate",
    "llmservingsim",
]

SIMULATOR_DISPLAY_NAMES = {
    "blis-trained-roofline": "BLIS-Trained",
    "blis-roofline": "BLIS-Roofline",
    "vidur": "Vidur",
    "llm-optimizer-estimate": "LLM-Optimizer",
    "aiconfigurator-estimate": "AIConfigurator",
    "llmservingsim": "LLMServingSim",
}

COLOR_PALETTE = {
    "blis-trained-roofline": "#4C72B0",
    "blis-roofline": "#64B5F6",
    "vidur": "#DD8452",
    "llm-optimizer-estimate": "#55A868",
    "aiconfigurator-estimate": "#8172B3",
    "llmservingsim": "#C44E52",
}

HATCH_PATTERNS = {
    "blis-trained-roofline": "",
    "blis-roofline": "//",
    "vidur": "\\\\",
    "llm-optimizer-estimate": "xx",
    "aiconfigurator-estimate": "..",
    "llmservingsim": "++",
}

MARKER_STYLES = {
    "blis-trained-roofline": "o",
    "blis-roofline": "s",
    "vidur": "D",
    "llm-optimizer-estimate": "^",
    "aiconfigurator-estimate": "v",
}

MODEL_ORDER = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-14B",
    "codellama/CodeLlama-34b-Instruct-hf",
    "meta-llama/Llama-2-70b-hf",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
]

MODEL_SHORT_LABELS = {
    "meta-llama/Llama-2-7b-hf": "Ll-2-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Ll-3.1-8B",
    "Qwen/Qwen3-14B": "Qw-14B",
    "codellama/CodeLlama-34b-Instruct-hf": "CL-34B",
    "meta-llama/Llama-2-70b-hf": "Ll-2-70B",
    "mistralai/Mixtral-8x7B-v0.1": "Mx-8x7B",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "Mx-8x22B",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic": "Scout-17B",
}

WORKLOAD_DISPLAY_NAMES = {
    "general": "General-Purpose",
    "codegen": "Code Generation",
    "roleplay": "Roleplay",
    "reasoning": "Reasoning",
    "general-lite": "General-Lite",
}

METRICS_GRID = [
    [("e2e_mean", "E2E Mean"), ("ttft_mean", "TTFT Mean"), ("itl_mean", "ITL Mean")],
]

# USENIX/ACM double-column = 7in, single-column = 3.33in
FIGURE_SIZES = {
    "wide": (7.0, 3.2),       # 3-panel figures (Figs 1, 2, 3, 4)
    "pareto": (3.33, 3.33),   # single-column square (Fig 5)
}

RC_PARAMS = {
    "font.family": "serif",
    "text.usetex": True,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

HARDWARE_ORDER = ["H100", "A100-80GB", "L40S"]
WORKLOAD_ORDER = ["general", "codegen", "roleplay", "reasoning"]

FIGURE3_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-14B",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

_METADATA_COLUMNS = [
    "hardware", "dp", "cpu_offload", "gpu_mem_util",
    "tp", "max_num_batched_tokens",
]

# Default TP for each model (used to detect tp-varied config experiments).
_MODEL_DEFAULT_TP = {
    "meta-llama/Llama-2-7b-hf": 1,
    "meta-llama/Llama-3.1-8B-Instruct": 1,
    "Qwen/Qwen3-14B": 1,
    "codellama/CodeLlama-34b-Instruct-hf": 2,
    "meta-llama/Llama-2-70b-hf": 4,
    "mistralai/Mixtral-8x7B-v0.1": 2,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 8,
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic": 2,
}

_MOE_MODELS = frozenset({
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
})


def _derive_config_tag(row: pd.Series) -> str:
    """Derive a config_tag from raw metadata fields in a DataFrame row.

    Priority order (first match wins): mbt, cpu_offload, gpu_mem, dp/ep, tp.
    """
    mbt = row.get("max_num_batched_tokens")
    if pd.notna(mbt) and int(mbt) != 2048:
        return f"mbt={int(mbt)}"

    cpu_offload = row.get("cpu_offload")
    if cpu_offload in (True, "True", "true", 1, "1"):
        return "cpu-offload"

    gpu_mem = row.get("gpu_mem_util")
    if pd.notna(gpu_mem) and float(gpu_mem) != 0.9:
        return f"gpu-{gpu_mem}"

    dp = row.get("dp")
    if pd.notna(dp) and dp != "" and int(float(dp)) > 1:
        model = row.get("model", "")
        if model in _MOE_MODELS:
            tp = int(float(row.get("tp", 1)))
            return f"ep={int(float(dp)) * tp}"
        return f"dp={int(float(dp))}"

    tp = row.get("tp")
    if pd.notna(tp):
        model = row.get("model", "")
        default_tp = _MODEL_DEFAULT_TP.get(model, 1)
        if int(float(tp)) != default_tp:
            return f"tp={int(float(tp))}"

    return "default"


def _add_config_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``config_tag`` column if not already present.

    Derives tags from raw metadata (tp, max_num_batched_tokens, cpu_offload,
    gpu_mem_util, dp).  If the DataFrame already has a ``config_tag`` column
    (e.g. from test fixtures or an external metadata CSV), it is kept as-is.
    """
    if "config_tag" in df.columns:
        return df
    df = df.copy()
    if "max_num_batched_tokens" in df.columns and df["max_num_batched_tokens"].notna().any():
        df["config_tag"] = df.apply(_derive_config_tag, axis=1)
    else:
        # Old CSV without raw fields — assume all experiments are default config
        df["config_tag"] = "default"
    return df


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_error_data(csv_path: str) -> pd.DataFrame:
    """Load error_records.csv, exclude blacklisted simulators, keep summary rows."""
    df = pd.read_csv(csv_path)
    df = df[~df["simulator"].isin(EXCLUDED_SIMULATORS)]
    df = df[df["stage_index"] == -1]
    return df.reset_index(drop=True)


def load_runtime_data(csv_path: str) -> pd.DataFrame:
    """Load runtime.csv, exclude blacklisted simulators."""
    df = pd.read_csv(csv_path)
    df = df[~df["simulator"].isin(EXCLUDED_SIMULATORS)]
    return df.reset_index(drop=True)


def enrich_with_metadata(
    df: pd.DataFrame,
    metadata_path: str | None = None,
) -> pd.DataFrame:
    """Left-join metadata onto DataFrame by experiment_folder."""
    if metadata_path:
        if os.path.exists(metadata_path):
            meta = pd.read_csv(metadata_path)
            return df.merge(meta, on="experiment_folder", how="left", suffixes=("", "_meta"))
        logger.warning("Metadata file not found: %s", metadata_path)
    df = df.copy()
    for col in _METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _has_metadata(df: pd.DataFrame) -> bool:
    """Check if DataFrame has non-empty metadata columns."""
    return (
        "hardware" in df.columns
        and df["hardware"].notna().any()
        and (df["hardware"] != "").any()
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_rc_params() -> None:
    """Apply publication-quality RC params. Falls back to non-LaTeX if unavailable."""
    params = dict(RC_PARAMS)
    if not shutil.which("latex"):
        params["text.usetex"] = False
    matplotlib.rcParams.update(params)


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    """Save figure as PDF + PNG at 300 DPI."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    p = Path(output_path)
    if p.suffix == ".pdf":
        fig.savefig(str(p.with_suffix(".png")), dpi=300, bbox_inches="tight")
    logger.info("Saved %s", output_path)


def _grouped_bar(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    title: str,
    output_path: str | None,
    group_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    aggregate: bool = False,
    xlabel_rotation: float = 35,
    metrics: list[tuple[str, str]] | None = None,
) -> plt.Figure | None:
    """Grouped bar chart: x = groups, bars = simulators, y = MAPE.

    Parameters
    ----------
    aggregate : bool
        If True, take the median MAPE per (group, simulator, metric).
    metrics : list of (key, label) pairs
        Which metrics to plot as subplots. Defaults to E2E Mean only.
    """
    _apply_rc_params()
    if metrics is None:
        metrics = [("e2e_mean", "E2E Mean")]

    all_sims = set(df["simulator"].unique())
    simulators = [s for s in SIMULATOR_ORDER if s in all_sims]
    present_groups = [g for g in group_order if g in df[group_col].values]

    if not simulators or not present_groups:
        return None

    n_groups = len(present_groups)
    n_sims = len(simulators)
    n_cols = len(metrics)
    figsize = figsize or FIGURE_SIZES["wide"]

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    bar_width = 0.8 / n_sims
    x = np.arange(n_groups)
    col_maxes = [0.0] * n_cols
    labeled_sims = set()  # Track which simulators have been labeled

    for col_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[col_idx]

        for sim_idx, sim in enumerate(simulators):
            offset = (sim_idx - n_sims / 2 + 0.5) * bar_width
            positions = []
            heights = []

            for g_idx, group_val in enumerate(present_groups):
                vals = df[
                    (df[group_col] == group_val)
                    & (df["simulator"] == sim)
                    & (df["metric_name"] == metric_key)
                ]["mape"]
                if vals.empty:
                    continue
                mape = vals.median() if aggregate else vals.iloc[0]
                positions.append(x[g_idx] + offset)
                heights.append(mape)

            if not positions:
                continue
            col_maxes[col_idx] = max(col_maxes[col_idx], max(heights))

            # Add label only the first time this simulator is plotted
            label = SIMULATOR_DISPLAY_NAMES[sim] if sim not in labeled_sims else ""
            if sim not in labeled_sims:
                labeled_sims.add(sim)
            ax.bar(
                positions, heights, bar_width,
                color=COLOR_PALETTE[sim],
                hatch=HATCH_PATTERNS.get(sim, ""),
                edgecolor="black", linewidth=0.5,
                label=label,
            )

        ax.set_xticks(x)
        ha = "right" if xlabel_rotation else "center"
        ax.set_xticklabels(
            [(group_labels or {}).get(g, g) for g in present_groups],
            rotation=xlabel_rotation, ha=ha,
        )
        ax.set_title(metric_label)

    # Independent y-axis per subplot with 20% headroom
    pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
    for col_idx, ax in enumerate(axes):
        y_top = col_maxes[col_idx] * 1.20 if col_maxes[col_idx] > 0 else 1.0
        ax.set_ylim(bottom=0, top=y_top)
        ax.set_ylabel(f"MAPE ({pct})")

    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Collect legend handles/labels from all axes (not just axes[0])
    # to include simulators that don't have data in the first metric column
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in all_labels:  # Deduplicate by label
                all_handles.append(handle)
                all_labels.append(label)

    if all_handles:
        fig.legend(
            all_handles, all_labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.01), ncol=n_sims,
            frameon=False, handlelength=1.5, columnspacing=1.0,
        )

    fig.tight_layout()
    # tight_layout doesn't account for suptitle — push axes down to make room
    fig.subplots_adjust(top=0.86)

    if output_path:
        _save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure Functions
# ---------------------------------------------------------------------------


def plot_aggregate_comparison_analytical(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 0a: BLIS-Roofline vs LLM-Optimizer vs AIConfigurator.

    Only includes experiments where blis-roofline, llm-optimizer-estimate,
    and aiconfigurator-estimate all have data. Filters to default configs
    (model's default TP, cpu_offload=false, gpu_mem=0.9, dp≤1, mbt=2048)
    and general/general-lite workloads (consistent with Figure 2).
    Shows median MAPE across these experiments for E2E, TTFT, and ITL
    (all three simulators).
    """
    _apply_rc_params()

    # Find experiments with data from all three target simulators
    target_sims = {"blis-roofline", "llm-optimizer-estimate", "aiconfigurator-estimate"}
    exp_sims = df.groupby("experiment_folder")["simulator"].apply(set)
    common_exps = exp_sims[exp_sims.apply(lambda s: target_sims.issubset(s))].index

    if len(common_exps) == 0:
        warnings.warn("Figure 0: no experiments with data from all simulators")
        return None

    df_filtered = df[df["experiment_folder"].isin(common_exps)]

    # Filter to default configs only (consistent with Figure 2).
    # All three simulators support multi-GPU configs, so we include experiments
    # at each model's default TP rather than restricting to tp=1.
    df_filtered = df_filtered[df_filtered["config_tag"] == "default"]
    # Also filter to general/general-lite workloads to match Figure 2
    df_filtered = df_filtered[df_filtered["workload"].isin(("general", "general-lite"))]

    if df_filtered.empty:
        warnings.warn("Figure 0: no experiments with default configs")
        return None

    common_exps = df_filtered["experiment_folder"].unique()

    # Prepare data for each metric
    metrics_data = []

    # E2E: all three simulators
    e2e_df = df_filtered[
        (df_filtered["metric_name"] == "e2e_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not e2e_df.empty:
        e2e_agg = e2e_df.groupby("simulator")["mape"].median()
        metrics_data.append(("E2E Mean", e2e_agg, list(target_sims)))

    # TTFT: all three simulators
    ttft_df = df_filtered[
        (df_filtered["metric_name"] == "ttft_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not ttft_df.empty:
        ttft_agg = ttft_df.groupby("simulator")["mape"].median()
        metrics_data.append(("TTFT Mean", ttft_agg, list(target_sims)))

    # ITL: all three simulators
    itl_df = df_filtered[
        (df_filtered["metric_name"] == "itl_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not itl_df.empty:
        itl_agg = itl_df.groupby("simulator")["mape"].median()
        metrics_data.append(("ITL Mean", itl_agg, list(target_sims)))

    if not metrics_data:
        warnings.warn("Figure 0: no metrics data after aggregation")
        return None

    # Create figure
    n_metrics = len(metrics_data)
    fig, axes = plt.subplots(1, n_metrics, figsize=(10, 4))
    if n_metrics == 1:
        axes = [axes]

    bar_width = 0.6
    labeled_sims = set()  # Track which simulators have been labeled

    for col_idx, (metric_label, agg_data, sims_for_metric) in enumerate(metrics_data):
        ax = axes[col_idx]

        # Sort simulators by SIMULATOR_ORDER
        sims_ordered = [s for s in SIMULATOR_ORDER if s in sims_for_metric and s in agg_data.index]

        x = np.arange(len(sims_ordered))
        heights = [agg_data[sim] for sim in sims_ordered]
        colors = [COLOR_PALETTE[sim] for sim in sims_ordered]
        hatches = [HATCH_PATTERNS.get(sim, "") for sim in sims_ordered]

        for i, (pos, height, color, hatch) in enumerate(zip(x, heights, colors, hatches)):
            sim = sims_ordered[i]
            # Add label only the first time this simulator is plotted
            label = SIMULATOR_DISPLAY_NAMES[sim] if sim not in labeled_sims else ""
            if sim not in labeled_sims:
                labeled_sims.add(sim)

            ax.bar(
                pos, height, bar_width,
                color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5,
                label=label,
            )

        ax.set_xticks([])
        ax.set_xlim(-0.5, len(sims_ordered) - 0.5)

        y_top = max(heights) * 1.20 if heights else 1.0
        ax.set_ylim(bottom=0, top=y_top)

        pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
        ax.set_ylabel(f"MAPE ({pct})")
        ax.set_title(metric_label, fontsize=10, fontweight="bold")

    fig.suptitle(
        f"BLIS-Roofline vs LLM-Optimizer vs AIConfigurator (n={len(common_exps)}) ↓",
        fontsize=11, fontweight="bold"
    )

    # Collect legend from all axes (not just axes[0])
    # to include simulators that don't have data in the first metric column
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in all_labels:  # Deduplicate by label
                all_handles.append(handle)
                all_labels.append(label)

    if all_handles:
        fig.legend(
            all_handles, all_labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.01), ncol=len(all_handles),
            frameon=False, handlelength=1.5, columnspacing=1.0,
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_aggregate_comparison_trace(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 0b: BLIS-Roofline vs Vidur.

    Only includes experiments where both blis-roofline and vidur have data.
    Filters to default configs (model's default TP, cpu_offload=false,
    gpu_mem=0.9, dp≤1, mbt=2048) and general/general-lite workloads.
    Shows median MAPE across these experiments for E2E, TTFT, and ITL
    (both simulators report all three metrics with full distributions).

    Note: Vidur only supports 3 models (Llama-2-7b, Llama-2-70b, CodeLlama-34b),
    so experiment count is limited by Vidur's model coverage.
    """
    _apply_rc_params()

    # Find experiments with data from both trace-replay simulators
    target_sims = {"blis-roofline", "vidur"}
    exp_sims = df.groupby("experiment_folder")["simulator"].apply(set)
    common_exps = exp_sims[exp_sims.apply(lambda s: target_sims.issubset(s))].index

    if len(common_exps) == 0:
        warnings.warn("Figure 0b: no experiments with data from both BLIS and Vidur")
        return None

    df_filtered = df[df["experiment_folder"].isin(common_exps)]

    # Filter to default configs only (consistent with Figure 0a and Figure 2)
    df_filtered = df_filtered[df_filtered["config_tag"] == "default"]
    # Also filter to general/general-lite workloads
    df_filtered = df_filtered[df_filtered["workload"].isin(("general", "general-lite"))]

    if df_filtered.empty:
        warnings.warn("Figure 0b: no experiments with default configs")
        return None

    common_exps = df_filtered["experiment_folder"].unique()

    # Prepare data for each metric (both simulators report all three)
    metrics_data = []

    # E2E: both simulators
    e2e_df = df_filtered[
        (df_filtered["metric_name"] == "e2e_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not e2e_df.empty:
        e2e_agg = e2e_df.groupby("simulator")["mape"].median()
        metrics_data.append(("E2E Mean", e2e_agg, list(target_sims)))

    # TTFT: both simulators
    ttft_df = df_filtered[
        (df_filtered["metric_name"] == "ttft_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not ttft_df.empty:
        ttft_agg = ttft_df.groupby("simulator")["mape"].median()
        metrics_data.append(("TTFT Mean", ttft_agg, list(target_sims)))

    # ITL: both simulators
    itl_df = df_filtered[
        (df_filtered["metric_name"] == "itl_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not itl_df.empty:
        itl_agg = itl_df.groupby("simulator")["mape"].median()
        metrics_data.append(("ITL Mean", itl_agg, list(target_sims)))

    if not metrics_data:
        warnings.warn("Figure 0b: no metrics data after aggregation")
        return None

    # Create figure
    n_metrics = len(metrics_data)
    fig, axes = plt.subplots(1, n_metrics, figsize=(10, 4))
    if n_metrics == 1:
        axes = [axes]

    bar_width = 0.6
    labeled_sims = set()  # Track which simulators have been labeled

    for col_idx, (metric_label, agg_data, sims_for_metric) in enumerate(metrics_data):
        ax = axes[col_idx]

        # Sort simulators by SIMULATOR_ORDER
        sims_ordered = [s for s in SIMULATOR_ORDER if s in sims_for_metric and s in agg_data.index]

        x = np.arange(len(sims_ordered))
        heights = [agg_data[sim] for sim in sims_ordered]
        colors = [COLOR_PALETTE[sim] for sim in sims_ordered]
        hatches = [HATCH_PATTERNS.get(sim, "") for sim in sims_ordered]

        for i, (pos, height, color, hatch) in enumerate(zip(x, heights, colors, hatches)):
            sim = sims_ordered[i]
            # Add label only the first time this simulator is plotted
            label = SIMULATOR_DISPLAY_NAMES[sim] if sim not in labeled_sims else ""
            if sim not in labeled_sims:
                labeled_sims.add(sim)

            ax.bar(
                pos, height, bar_width,
                color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5,
                label=label,
            )

        ax.set_xticks([])
        ax.set_xlim(-0.5, len(sims_ordered) - 0.5)

        y_top = max(heights) * 1.20 if heights else 1.0
        ax.set_ylim(bottom=0, top=y_top)

        pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
        ax.set_ylabel(f"MAPE ({pct})")
        ax.set_title(metric_label, fontsize=10, fontweight="bold")

    fig.suptitle(
        f"BLIS-Roofline vs Vidur (n={len(common_exps)}) ↓",
        fontsize=11, fontweight="bold"
    )

    # Collect legend from all axes
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in all_labels:  # Deduplicate by label
                all_handles.append(handle)
                all_labels.append(label)

    if all_handles:
        fig.legend(
            all_handles, all_labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.01), ncol=len(all_handles),
            frameon=False, handlelength=1.5, columnspacing=1.0,
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_aggregate_comparison_llmservingsim(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 0c: BLIS-Roofline vs LLM-Optimizer vs LLMServingSim for MoE.

    Compares blis-roofline, llm-optimizer-estimate, and llmservingsim on the
    single Mixtral-8x7B tp4 experiment where all three simulators have data.
    Shows MAPE for E2E, TTFT, and ITL.

    Note: LLMServingSim only has complete pre-profiled models for a limited
    set of configurations. This figure demonstrates the comparison for the
    single available MoE experiment.
    """
    _apply_rc_params()

    # Find experiments with data from all three simulators
    target_sims = {"blis-roofline", "llm-optimizer-estimate", "llmservingsim"}
    exp_sims = df.groupby("experiment_folder")["simulator"].apply(set)
    common_exps = exp_sims[exp_sims.apply(lambda s: target_sims.issubset(s))].index

    if len(common_exps) == 0:
        warnings.warn("Figure 0c: no experiments with data from all three simulators")
        return None

    df_filtered = df[df["experiment_folder"].isin(common_exps)]

    # Filter to Mixtral-8x7B, general workload, and tp=4
    df_filtered = df_filtered[
        (df_filtered["model"] == "mistralai/Mixtral-8x7B-v0.1") &
        (df_filtered["workload"] == "general") &
        (df_filtered["tp"] == 4)
    ]

    if df_filtered.empty:
        warnings.warn("Figure 0c: no Mixtral-8x7B general tp=4 experiments")
        return None

    common_exps = df_filtered["experiment_folder"].unique()

    # Extract TP for subtitle
    tp_val = 4  # We know it's tp=4 from the filter above

    # Prepare data for each metric (all three simulators report all three)
    metrics_data = []

    # E2E: all three simulators
    e2e_df = df_filtered[
        (df_filtered["metric_name"] == "e2e_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not e2e_df.empty:
        e2e_agg = e2e_df.groupby("simulator")["mape"].median()
        metrics_data.append(("E2E Mean", e2e_agg, list(target_sims)))

    # TTFT: all three simulators
    ttft_df = df_filtered[
        (df_filtered["metric_name"] == "ttft_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not ttft_df.empty:
        ttft_agg = ttft_df.groupby("simulator")["mape"].median()
        metrics_data.append(("TTFT Mean", ttft_agg, list(target_sims)))

    # ITL: all three simulators
    itl_df = df_filtered[
        (df_filtered["metric_name"] == "itl_mean") &
        (df_filtered["simulator"].isin(list(target_sims)))
    ]
    if not itl_df.empty:
        itl_agg = itl_df.groupby("simulator")["mape"].median()
        metrics_data.append(("ITL Mean", itl_agg, list(target_sims)))

    if not metrics_data:
        warnings.warn("Figure 0c: no metrics data after aggregation")
        return None

    # Create figure
    n_metrics = len(metrics_data)
    fig, axes = plt.subplots(1, n_metrics, figsize=(10, 4))
    if n_metrics == 1:
        axes = [axes]

    bar_width = 0.6
    labeled_sims = set()  # Track which simulators have been labeled

    for col_idx, (metric_label, agg_data, sims_for_metric) in enumerate(metrics_data):
        ax = axes[col_idx]

        # Sort simulators by SIMULATOR_ORDER
        sims_ordered = [s for s in SIMULATOR_ORDER if s in sims_for_metric and s in agg_data.index]

        x = np.arange(len(sims_ordered))
        heights = [agg_data[sim] for sim in sims_ordered]
        colors = [COLOR_PALETTE[sim] for sim in sims_ordered]
        hatches = [HATCH_PATTERNS.get(sim, "") for sim in sims_ordered]

        for i, (pos, height, color, hatch) in enumerate(zip(x, heights, colors, hatches)):
            sim = sims_ordered[i]
            # Add label only the first time this simulator is plotted
            label = SIMULATOR_DISPLAY_NAMES[sim] if sim not in labeled_sims else ""
            if sim not in labeled_sims:
                labeled_sims.add(sim)

            ax.bar(
                pos, height, bar_width,
                color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5,
                label=label,
            )

        ax.set_xticks([])
        ax.set_xlim(-0.5, len(sims_ordered) - 0.5)

        y_top = max(heights) * 1.20 if heights else 1.0
        ax.set_ylim(bottom=0, top=y_top)

        pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
        ax.set_ylabel(f"MAPE ({pct})")
        ax.set_title(metric_label, fontsize=10, fontweight="bold")

    # Title with subtitle
    fig.suptitle(
        f"BLIS-Roofline vs LLM-Optimizer vs LLMServingSim (n={len(common_exps)}) ↓ for MoE",
        fontsize=11, fontweight="bold"
    )
    fig.text(
        0.5, 0.91, f"Mixtral-8x7B, H100, general, TP={tp_val}",
        ha="center", fontsize=9, style="italic"
    )

    # Collect legend from all axes
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in all_labels:  # Deduplicate by label
                all_handles.append(handle)
                all_labels.append(label)

    if all_handles:
        fig.legend(
            all_handles, all_labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.01), ncol=len(all_handles),
            frameon=False, handlelength=1.5, columnspacing=1.0,
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.86)

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_model_sensitivity(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 1: MAPE across 7 model architectures on H100, default config."""
    if _has_metadata(df):
        df = df[(df["hardware"] == "H100") & (df["config_tag"] == "default")]
    df = df[df["workload"].isin(("general", "general-lite"))]
    df = df[df["model"].isin(MODEL_ORDER)]

    if df.empty:
        warnings.warn("Figure 1: no data after filtering")
        return None

    fig = _grouped_bar(
        df, group_col="model", group_order=MODEL_ORDER,
        title="Prediction Error Across Model Architectures ↓",
        output_path=output_path,
        group_labels=MODEL_SHORT_LABELS,
        metrics=[
            ("e2e_mean", "E2E Mean"),
            ("ttft_mean", "TTFT Mean"),
            ("itl_mean", "ITL Mean"),
        ],
    )
    if fig is None:
        warnings.warn("Figure 1: no models with data")
    return fig


def plot_hardware_portability(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 2: MAPE across 3 GPU types, aggregated over models."""
    if not _has_metadata(df):
        warnings.warn("Figure 2: skipped (no hardware metadata)")
        return None

    df = df[df["config_tag"] == "default"]
    df = df[df["workload"].isin(("general", "general-lite"))]
    df = df[df["dp"].replace("", 1).fillna(1).astype(float) <= 1]

    if df.empty:
        warnings.warn("Figure 2: no data after filtering")
        return None

    fig = _grouped_bar(
        df, group_col="hardware", group_order=HARDWARE_ORDER,
        group_labels={"A100-80GB": "A100"},
        xlabel_rotation=0,
        title="Prediction Error Across GPU Types ↓",
        output_path=output_path,
        aggregate=True,
        metrics=[
            ("e2e_mean", "E2E Mean"),
            ("ttft_mean", "TTFT Mean"),
            ("itl_mean", "ITL Mean"),
        ],
    )
    if fig is None:
        warnings.warn("Figure 2: no hardware groups with data")
    return fig


def plot_workload_sensitivity(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 3: MAPE across 4 workload types, aggregated over 4 models."""
    if _has_metadata(df):
        df = df[(df["hardware"] == "H100") & (df["config_tag"] == "default")]
        df = df[df["dp"].replace("", 1).fillna(1).astype(float) <= 1]
    df = df[df["model"].isin(FIGURE3_MODELS)]

    if df.empty:
        warnings.warn("Figure 3: no data after filtering")
        return None

    fig = _grouped_bar(
        df, group_col="workload", group_order=WORKLOAD_ORDER,
        title="Prediction Error Across Workload Types ↓",
        output_path=output_path,
        group_labels=WORKLOAD_DISPLAY_NAMES, aggregate=True,
        metrics=[
            ("e2e_mean", "E2E Mean"),
            ("ttft_mean", "TTFT Mean"),
            ("itl_mean", "ITL Mean"),
        ],
    )
    if fig is None:
        warnings.warn("Figure 3: no workloads with data")
    return fig


FIG4A_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"
FIG4B_MODEL_DEFAULT = "mistralai/Mixtral-8x7B-v0.1"

# Keep legacy aliases for backwards compatibility in tests
FIG4A_MODEL = FIG4A_MODEL_DEFAULT
FIG4B_MODEL = FIG4B_MODEL_DEFAULT


CONFIG_SENSITIVITY_PARAMS: list[tuple[str, str]] = [
    ("tp", "TP"),
    ("max_num_batched_tokens", "Chunk Size"),
    ("gpu_mem_util", "GPU Mem (\\%)"),
    ("cpu_offload", "KV Cache Offloading"),
    ("dp", "DP"),
]
"""Config parameters to sweep for Figures 4a/4b.  Each entry is
(column_name, display_label).  Only parameters with >1 distinct value
for a given model are shown."""


def _config_variation_score(df: pd.DataFrame, model: str) -> int:
    """Score a model by total config variation (sum of distinct values across varying params)."""
    mdf = df[df["model"] == model]
    score = 0
    for col, _ in CONFIG_SENSITIVITY_PARAMS:
        if col not in mdf.columns:
            continue
        unique = mdf[col].dropna()
        unique = unique[unique != ""]
        n = unique.nunique()
        if n > 1:
            score += n
    return score


def _pick_best_model(df: pd.DataFrame, candidates: list[str], fallback: str) -> str:
    """Pick the candidate model with the most config variation in *df*."""
    best, best_score = fallback, -1
    for model in candidates:
        s = _config_variation_score(df, model)
        if s > best_score:
            best, best_score = model, s
    return best


def _short_model_name(model: str) -> str:
    """Extract a short display name: 'meta-llama/Llama-3.1-8B-Instruct' → 'Llama-3.1-8B'."""
    name = model.split("/")[-1]
    # Strip common suffixes
    for suffix in ("-Instruct", "-hf", "-v0.1", "-FP8-dynamic"):
        name = name.replace(suffix, "")
    return name


FIG4_METRIC = "e2e_mean"
FIG4_METRIC_LABEL = "E2E Mean"


def _plot_config_sensitivity(
    df: pd.DataFrame,
    model: str,
    title: str,
    subtitle: str,
    output_path: str | None,
) -> plt.Figure | None:
    """Shared implementation for Figures 4a/4b.

    Single grouped-bar chart with all config values on the x-axis,
    grouped by parameter with gaps between groups and group labels below.
    Filters to ``FIG4_METRIC`` only.
    """
    if not _has_metadata(df):
        warnings.warn(f"{title}: skipped (no config metadata)")
        return None

    df = df[df["model"] == model]
    df = df[df["metric_name"] == FIG4_METRIC]

    # Pin to a single hardware and workload so we only vary config params
    if "hardware" in df.columns and df["hardware"].notna().any():
        df = df[df["hardware"] == "H100"]
    df = df[df["workload"].isin(("general", "general-lite"))]

    if df.empty:
        warnings.warn(f"{title}: no data for model")
        return None

    # Extract hardware and workload for subtitle
    hw_str = "H100"
    wl_vals = df["workload"].dropna()
    wl_vals = wl_vals[wl_vals != ""]
    wl_names = [WORKLOAD_DISPLAY_NAMES.get(w, w) for w in sorted(wl_vals.unique())]
    wl_str = ", ".join(wl_names) if wl_names else None

    extra_parts = [hw_str]
    if wl_str:
        extra_parts.append(wl_str)
    subtitle += " | " + ", ".join(extra_parts)

    all_sims = set(df["simulator"].unique())
    simulators = [s for s in SIMULATOR_ORDER if s in all_sims]
    n_sims = len(simulators)

    # Determine baseline (mode) value for each config parameter
    config_cols = [col for col, _ in CONFIG_SENSITIVITY_PARAMS if col in df.columns]
    baseline: dict[str, str] = {}
    for col in config_cols:
        vals = df[col].dropna()
        vals = vals[vals != ""]
        if not vals.empty:
            baseline[col] = str(vals.mode().iloc[0])

    # Collect config entries: list of (group_label, value_label, {sim: mape})
    entries: list[tuple[str, str, dict[str, float]]] = []
    group_boundaries: list[int] = []  # index of first entry in each group

    for col, label in CONFIG_SENSITIVITY_PARAMS:
        if col not in df.columns:
            continue
        unique = df[col].dropna()
        unique = unique[unique != ""]
        if unique.nunique() <= 1:
            continue

        # Controlled comparison: hold all OTHER params at baseline
        sweep_df = df.copy()
        for other_col in config_cols:
            if other_col == col or other_col not in baseline:
                continue
            sweep_df = sweep_df[sweep_df[other_col].astype(str) == baseline[other_col]]

        sweep_unique = sweep_df[col].dropna()
        sweep_unique = sweep_unique[sweep_unique != ""]
        if sweep_unique.nunique() <= 1:
            continue

        try:
            sorted_vals = sorted(sweep_unique.unique(), key=lambda v: float(v))
        except (ValueError, TypeError):
            sorted_vals = sorted(sweep_unique.unique(), key=str)

        group_boundaries.append(len(entries))
        for val in sorted_vals:
            sim_mapes: dict[str, float] = {}
            for sim in simulators:
                mask = (sweep_df["simulator"] == sim) & (sweep_df[col].astype(str) == str(val))
                vals = sweep_df.loc[mask, "mape"]
                if not vals.empty:
                    sim_mapes[sim] = vals.median()
            if col == "cpu_offload":
                val_label = "On" if str(val) in ("True", "true", "1") else "Off"
            elif col == "gpu_mem_util":
                val_label = str(int(round(float(val) * 100)))
            else:
                try:
                    val_label = str(int(float(val))) if float(val) == int(float(val)) else str(val)
                except (ValueError, TypeError):
                    val_label = str(val)
            entries.append((label, val_label, sim_mapes))

    if not entries:
        warnings.warn(f"{title}: no varying config params")
        return None

    _apply_rc_params()

    # Compute x-positions with gaps between groups
    GAP = 0.6  # extra space between parameter groups
    x_positions: list[float] = []
    pos = 0.0
    for i, (group_label, val_label, sim_mapes) in enumerate(entries):
        if i in group_boundaries and i > 0:
            pos += GAP
        x_positions.append(pos)
        pos += 1.0

    x = np.array(x_positions)
    bar_width = 0.8 / n_sims
    global_max = 0.0

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    for sim_idx, sim in enumerate(simulators):
        offset = (sim_idx - n_sims / 2 + 0.5) * bar_width
        positions = []
        heights = []
        for i, (_, _, sim_mapes) in enumerate(entries):
            if sim in sim_mapes:
                positions.append(x[i] + offset)
                heights.append(sim_mapes[sim])
        if not heights:
            continue
        global_max = max(global_max, max(heights))
        ax.bar(
            positions, heights, bar_width,
            color=COLOR_PALETTE[sim],
            hatch=HATCH_PATTERNS.get(sim, ""),
            edgecolor="black", linewidth=0.5,
            label=SIMULATOR_DISPLAY_NAMES[sim],
        )

    # X-axis: value labels
    ax.set_xticks(x)
    ax.set_xticklabels([e[1] for e in entries], rotation=0, ha="center")

    # Group labels below the value labels
    for g_idx, boundary in enumerate(group_boundaries):
        end = group_boundaries[g_idx + 1] - 1 if g_idx + 1 < len(group_boundaries) else len(entries) - 1
        mid_x = (x[boundary] + x[end]) / 2.0
        ax.text(
            mid_x, -0.15, entries[boundary][0],
            ha="center", va="top", fontsize=8, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    # Y-axis with headroom
    y_top = global_max * 1.20 if global_max > 0 else 1.0
    ax.set_ylim(bottom=0, top=y_top)
    pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
    ax.set_ylabel(f"MAPE ({pct})", labelpad=10)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.96)
    ax.set_title(subtitle, fontsize=9, fontweight="normal", pad=8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.01), ncol=n_sims,
            frameon=False, handlelength=1.5, columnspacing=1.0,
        )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.18)

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_config_sensitivity_dense(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 4a: Config sensitivity for the dense model with most variation."""
    dense_candidates = [m for m in df["model"].unique() if m not in _MOE_MODELS]
    model = _pick_best_model(df, dense_candidates, FIG4A_MODEL_DEFAULT)
    short = _short_model_name(model)
    return _plot_config_sensitivity(
        df, model=model,
        title="Config Sensitivity (Dense) ↓",
        subtitle=f"{short} — {FIG4_METRIC_LABEL} MAPE",
        output_path=output_path,
    )


def plot_config_sensitivity_moe(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 4b: Config sensitivity for the MoE model with most variation."""
    moe_candidates = [m for m in df["model"].unique() if m in _MOE_MODELS]
    model = _pick_best_model(df, moe_candidates, FIG4B_MODEL_DEFAULT)
    short = _short_model_name(model)
    return _plot_config_sensitivity(
        df, model=model,
        title="Config Sensitivity (MoE) ↓",
        subtitle=f"{short} — {FIG4_METRIC_LABEL} MAPE",
        output_path=output_path,
    )


def plot_pareto(
    error_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure | None:
    """Figure 5: Accuracy-speed Pareto frontier scatter plot."""
    if error_df.empty or runtime_df.empty:
        warnings.warn("Figure 5: insufficient data")
        return None

    _apply_rc_params()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["pareto"])

    # Per-experiment median MAPE (robust to outlier metrics like Vidur's
    # 2.4M% TTFT on Llama-2-70b), then per-simulator median + IQR
    exp_mape = error_df.groupby(["simulator", "experiment_folder"])["mape"].median()

    sim_stats: dict = {}
    for sim in SIMULATOR_ORDER:
        if sim not in exp_mape.index.get_level_values(0):
            continue
        vals = exp_mape.loc[sim]
        rt = runtime_df[runtime_df["simulator"] == sim]["wall_clock_seconds"]
        if rt.empty:
            continue
        sim_stats[sim] = {
            "mape_med": vals.median(),
            "mape_q1": vals.quantile(0.25),
            "mape_q3": vals.quantile(0.75),
            "n": len(vals),
            "rt_med": rt.median(),
            "rt_q1": rt.quantile(0.25),
            "rt_q3": rt.quantile(0.75),
        }

    if not sim_stats:
        warnings.warn("Figure 5: no simulator data")
        plt.close(fig)
        return None

    # Per-simulator annotation offsets: hand-tuned to avoid overlap in the
    # typical cluster layout (BLIS/LLM-Opt/AIC cluster low-MAPE, Vidur far right).
    _annotation_offsets = {
        "blis-trained-roofline": (-14, -20),
        "blis-roofline": (14, 16),
        "vidur": (12, -18),
        "llm-optimizer-estimate": (14, 16),
        "aiconfigurator-estimate": (-14, -20),
    }

    for sim in SIMULATOR_ORDER:
        if sim not in sim_stats:
            continue
        s = sim_stats[sim]
        marker_size = 120 if sim == "blis-trained-roofline" else 60

        # Clamp error bars to avoid negative values on log scale
        yerr_lo = min(s["rt_med"] - s["rt_q1"], s["rt_med"] * 0.9)
        yerr_hi = s["rt_q3"] - s["rt_med"]

        ax.errorbar(
            s["mape_med"], s["rt_med"],
            xerr=[[s["mape_med"] - s["mape_q1"]], [s["mape_q3"] - s["mape_med"]]],
            yerr=[[max(yerr_lo, 0)], [yerr_hi]],
            fmt="none", ecolor=COLOR_PALETTE[sim], elinewidth=1, capsize=3,
        )
        ax.scatter(
            s["mape_med"], s["rt_med"], s=marker_size,
            color=COLOR_PALETTE[sim], edgecolor="black", linewidth=0.5,
            zorder=5, label=SIMULATOR_DISPLAY_NAMES[sim],
        )
        ox, oy = _annotation_offsets.get(sim, (10, 8))
        ax.annotate(
            SIMULATOR_DISPLAY_NAMES[sim],
            (s["mape_med"], s["rt_med"]),
            textcoords="offset points", xytext=(ox, oy), fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "gray", "linewidth": 0.4},
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Tight limits for better spread
    all_mapes = [s["mape_med"] for s in sim_stats.values()]
    ax.set_xlim(min(all_mapes) * 0.5, max(all_mapes) * 3)
    all_rts = [s["rt_med"] for s in sim_stats.values()]
    ax.set_ylim(min(all_rts) * 0.4, max(all_rts) * 4)

    # Pareto-dominated shading (after limits are set)
    if "blis-trained-roofline" in sim_stats:
        bt = sim_stats["blis-trained-roofline"]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between(
            [bt["mape_med"], xlim[1]], bt["rt_med"], ylim[1],
            color="#90E0EF", alpha=0.15, zorder=0,
        )

    pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
    ax.set_xlabel(f"Median MAPE ({pct})")
    ax.set_ylabel("Median Runtime (s)")

    # Place legend below the plot to avoid overlap with data/annotations
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
            frameon=False, fontsize=7, handlelength=1.5, columnspacing=1.0,
        )
    fig.tight_layout()

    if output_path:
        _save_figure(fig, output_path)
    return fig


def format_runtime_table_latex(
    runtime_df: pd.DataFrame,
    output_path: str | None = None,
    real_experiment_seconds: float = 1200.0,
) -> str:
    """Table 1: LaTeX tabular with median runtime and speedup vs. real."""
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Simulator & Median Runtime (s) & Speedup vs.\ Real \\",
        r"\midrule",
    ]
    for sim in SIMULATOR_ORDER:
        sdf = runtime_df[runtime_df["simulator"] == sim]
        if sdf.empty:
            continue
        median_t = sdf["wall_clock_seconds"].median()
        speedup = real_experiment_seconds / median_t if median_t > 0 else float("inf")
        lines.append(
            f"{SIMULATOR_DISPLAY_NAMES[sim]} & {median_t:.1f} "
            f"& {speedup:.0f}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(tex)
        logger.info("Saved %s", output_path)
    return tex


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_figure_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from experiment CSVs.",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing error_records.csv and runtime.csv",
    )
    parser.add_argument(
        "--output-dir", default="results/figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="Path to experiment_metadata.csv for hardware/config enrichment",
    )
    parser.add_argument(
        "--exclude-simulators", nargs="+", default=[],
        help="Simulators to hide from all figures (e.g. --exclude-simulators vidur)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_figure_args(argv)

    error_csv = os.path.join(args.results_dir, "error_records.csv")
    runtime_csv = os.path.join(args.results_dir, "runtime.csv")

    if not os.path.exists(error_csv):
        msg = f"error_records.csv not found in {args.results_dir}"
        logger.error(msg)
        print(msg)
        return
    if not os.path.exists(runtime_csv):
        msg = f"runtime.csv not found in {args.results_dir}"
        logger.error(msg)
        print(msg)
        return

    error_df_full = load_error_data(error_csv)
    runtime_df_full = load_runtime_data(runtime_csv)

    if args.exclude_simulators:
        excluded = set(args.exclude_simulators)
        error_df = error_df_full[~error_df_full["simulator"].isin(excluded)].reset_index(drop=True)
        runtime_df = runtime_df_full[~runtime_df_full["simulator"].isin(excluded)].reset_index(drop=True)
        print(f"Excluding simulators from Figs 0a, 1-4: {', '.join(sorted(excluded))}")
    else:
        error_df = error_df_full
        runtime_df = runtime_df_full

    error_df = enrich_with_metadata(error_df, args.metadata)
    runtime_df = enrich_with_metadata(runtime_df, args.metadata)
    error_df = _add_config_tags(error_df)

    # Fig 5 (Pareto) uses all simulators (ignore CLI exclusions)
    error_df_all = enrich_with_metadata(error_df_full, args.metadata)
    runtime_df_all = enrich_with_metadata(runtime_df_full, args.metadata)
    error_df_all = _add_config_tags(error_df_all)

    # Fig 0b (BLIS vs Vidur) needs Vidur data, which is excluded from error_df_full
    # Load raw CSV and filter to just blis-roofline and vidur
    error_df_0b = pd.read_csv(error_csv)
    error_df_0b = error_df_0b[error_df_0b["stage_index"] == -1]  # summary rows only
    error_df_0b = error_df_0b[error_df_0b["simulator"].isin(["blis-roofline", "vidur"])]
    error_df_0b = enrich_with_metadata(error_df_0b, args.metadata)
    error_df_0b = _add_config_tags(error_df_0b)

    # Fig 0c (BLIS vs LLM-Optimizer vs LLMServingSim) needs llmservingsim data, which may be excluded
    # Load raw CSV and filter to just blis-roofline, llm-optimizer-estimate, and llmservingsim
    error_df_0c = pd.read_csv(error_csv)
    error_df_0c = error_df_0c[error_df_0c["stage_index"] == -1]  # summary rows only
    error_df_0c = error_df_0c[error_df_0c["simulator"].isin(["blis-roofline", "llm-optimizer-estimate", "llmservingsim"])]
    error_df_0c = enrich_with_metadata(error_df_0c, args.metadata)
    error_df_0c = _add_config_tags(error_df_0c)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    figures = [
        ("fig0a_aggregate_analytical.pdf",
         lambda: plot_aggregate_comparison_analytical(error_df, os.path.join(out, "fig0a_aggregate_analytical.pdf"))),
        ("fig0b_aggregate_trace.pdf",
         lambda: plot_aggregate_comparison_trace(error_df_0b, os.path.join(out, "fig0b_aggregate_trace.pdf"))),
        ("fig0c_aggregate_llmservingsim.pdf",
         lambda: plot_aggregate_comparison_llmservingsim(error_df_0c, os.path.join(out, "fig0c_aggregate_llmservingsim.pdf"))),
        ("fig1_model_sensitivity.pdf",
         lambda: plot_model_sensitivity(error_df, os.path.join(out, "fig1_model_sensitivity.pdf"))),
        ("fig2_hardware_portability.pdf",
         lambda: plot_hardware_portability(error_df, os.path.join(out, "fig2_hardware_portability.pdf"))),
        ("fig3_workload_sensitivity.pdf",
         lambda: plot_workload_sensitivity(error_df, os.path.join(out, "fig3_workload_sensitivity.pdf"))),
        ("fig4a_config_dense.pdf",
         lambda: plot_config_sensitivity_dense(error_df, os.path.join(out, "fig4a_config_dense.pdf"))),
        ("fig4b_config_moe.pdf",
         lambda: plot_config_sensitivity_moe(error_df, os.path.join(out, "fig4b_config_moe.pdf"))),
        ("fig5_pareto.pdf",
         lambda: plot_pareto(error_df_all, runtime_df_all, os.path.join(out, "fig5_pareto.pdf"))),
    ]

    for name, generate in figures:
        try:
            fig = generate()
            if fig is not None:
                plt.close(fig)
                print(f"  OK: {name}")
            else:
                print(f"  SKIP: {name} (insufficient data)")
        except Exception as e:
            print(f"  FAIL: {name} ({e})")
            logger.exception("Failed to generate %s", name)

    # Table 1
    try:
        tex = format_runtime_table_latex(
            runtime_df, output_path=os.path.join(out, "table1_runtime.tex"),
        )
        print(f"  OK: table1_runtime.tex")
    except Exception as e:
        print(f"  FAIL: table1_runtime.tex ({e})")
        logger.exception("Failed to generate table1_runtime.tex")

    print(f"\nFigures saved to {out}")


if __name__ == "__main__":
    main()
