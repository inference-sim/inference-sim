"""Sampling bias characterization for StepML ground truth data.

The ground truth dataset uses step_tracing_sample_rate=0.1, meaning only ~10%
of steps are traced. This module characterizes whether the sampling introduces
systematic bias (e.g., periodic sampling, uneven experiment coverage).
"""

import numpy as np
import pandas as pd


def characterize_sampling(df: pd.DataFrame) -> dict:
    """Characterize sampling patterns in the ground truth step data.

    Analyzes the step.id gaps within each experiment to determine whether
    the 10% sampling is periodic (every Nth step) or random/pseudo-random,
    and whether all experiments are represented uniformly.

    Args:
        df: DataFrame from load_all_experiments() with columns including
            'experiment_id' and 'step.id'.

    Returns:
        dict with keys:
            - total_steps: int -- total rows loaded
            - per_experiment_counts: dict[str, int] -- row count per experiment_id
            - step_id_gaps: dict -- per-experiment gap statistics (mean, std, min, max)
            - is_periodic: bool -- True if any experiment has gap std < 0.5
            - coverage_uniformity: float -- coefficient of variation of per-experiment counts
            - summary: str -- human-readable summary of findings
    """
    total_steps = len(df)

    # Per-experiment step counts
    per_experiment_counts = df.groupby("experiment_id").size().to_dict()

    # Per-experiment step.id gap analysis
    step_id_gaps = {}
    any_periodic = False

    for exp_id, group in df.groupby("experiment_id"):
        step_ids = group["step.id"].sort_values().reset_index(drop=True)
        gaps = step_ids.diff().dropna()

        gap_mean = float(gaps.mean())
        gap_std = float(gaps.std())
        gap_min = int(gaps.min())
        gap_max = int(gaps.max())

        step_id_gaps[exp_id] = {
            "mean": gap_mean,
            "std": gap_std,
            "min": gap_min,
            "max": gap_max,
        }

        if gap_std < 0.5:
            any_periodic = True

    # Coverage uniformity: coefficient of variation of per-experiment counts
    counts_array = np.array(list(per_experiment_counts.values()), dtype=float)
    coverage_uniformity = float(counts_array.std() / counts_array.mean())

    # Build human-readable summary
    summary_lines = [
        f"Sampling characterization for {len(per_experiment_counts)} experiments:",
        f"  Total traced steps: {total_steps:,}",
        f"  Estimated total steps (at 10% rate): ~{total_steps * 10:,}",
        f"  Steps per experiment: min={int(counts_array.min()):,}, "
        f"max={int(counts_array.max()):,}, "
        f"mean={counts_array.mean():,.0f}",
        f"  Coverage uniformity (CV): {coverage_uniformity:.4f}",
        f"  Periodic sampling detected: {any_periodic}",
    ]

    # Aggregate gap statistics across all experiments
    all_means = [v["mean"] for v in step_id_gaps.values()]
    all_stds = [v["std"] for v in step_id_gaps.values()]
    summary_lines.append(
        f"  Step ID gap (across experiments): "
        f"mean of means={np.mean(all_means):.2f}, "
        f"mean of stds={np.mean(all_stds):.2f}"
    )

    if any_periodic:
        summary_lines.append(
            "  WARNING: At least one experiment shows periodic sampling "
            "(gap std < 0.5). This may introduce systematic bias."
        )
    else:
        summary_lines.append(
            "  Sampling appears random/pseudo-random (all experiments have "
            "gap std > 0.5). No periodic bias detected."
        )

    summary = "\n".join(summary_lines)

    return {
        "total_steps": total_steps,
        "per_experiment_counts": per_experiment_counts,
        "step_id_gaps": step_id_gaps,
        "is_periodic": any_periodic,
        "coverage_uniformity": coverage_uniformity,
        "summary": summary,
    }
