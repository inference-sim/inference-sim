"""Per-request KV cache length extractor for step-level feature enrichment.

Bridges the gap between step-level batch summaries (which lack per-request KV
cache lengths) and per-request lifecycle data (which has token timestamps).
By joining these two data sources, we derive per-step KV statistics
(kv_mean, kv_max, kv_sum, kv_count) needed for accurate roofline step-time
prediction.

Background: Round 1 H8 showed a 12.96x overestimate when using batch-level
kv_max (the max across the whole experiment) instead of per-request KV lengths.
This module computes the actual per-step KV distribution from lifecycle data.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data_loader import (
    DEFAULT_DATA_ROOT,
    load_all_experiments,
    load_experiment_steps,
    load_lifecycle_data,
    parse_experiment_metadata,
)


def _estimate_kv_length(request_row: pd.Series, step_start_ns: int) -> int:
    """Estimate KV cache length for a request at a given step timestamp.

    KV length = input_tokens + number of output tokens generated before
    step_start_ns. This mirrors the simulator's Request.ProgressIndex
    (input_tokens_processed + output_tokens_generated).

    Args:
        request_row: A row from the lifecycle DataFrame with columns:
            input_tokens (int), output_token_times (list of float, epoch seconds).
        step_start_ns: Step start timestamp in nanoseconds.

    Returns:
        Estimated KV cache length (non-negative integer).
    """
    input_tokens = int(request_row["input_tokens"])
    output_token_times = request_row["output_token_times"]

    if not output_token_times:
        return input_tokens

    # Convert step_start_ns to seconds for comparison with output_token_times
    step_start_s = step_start_ns / 1e9

    # Count output tokens generated before step_start_ns
    tokens_generated = 0
    for t in output_token_times:
        if t < step_start_s:
            tokens_generated += 1
        else:
            # output_token_times are chronologically ordered; once we pass
            # the step boundary, no further tokens qualify.
            break

    return input_tokens + tokens_generated


def extract_kv_features(
    steps_df: pd.DataFrame, lifecycle_df: pd.DataFrame
) -> pd.DataFrame:
    """Join step-level data with lifecycle data to derive per-step KV features.

    For each step, identifies active requests (those whose time window overlaps
    the step window), estimates each request's KV cache length at step start,
    and computes aggregate statistics.

    Args:
        steps_df: Step-level DataFrame from data_loader.load_experiment_steps().
            Must have columns: step.id, step.ts_start_ns, step.ts_end_ns,
            experiment_id.
        lifecycle_df: Per-request DataFrame from data_loader.load_lifecycle_data().
            Must have columns: start_time, end_time, input_tokens,
            output_tokens, output_token_times. Index is request_id.

    Returns:
        DataFrame with same index as steps_df plus new columns:
        kv_mean, kv_max, kv_sum, kv_count.
    """
    # Pre-convert lifecycle timestamps from seconds to nanoseconds for
    # efficient comparison with step timestamps.
    lc_start_ns = (lifecycle_df["start_time"].values * 1e9).astype(np.int64)
    lc_end_ns = (lifecycle_df["end_time"].values * 1e9).astype(np.int64)

    kv_means = np.zeros(len(steps_df), dtype=np.float64)
    kv_maxes = np.zeros(len(steps_df), dtype=np.float64)
    kv_sums = np.zeros(len(steps_df), dtype=np.float64)
    kv_stds = np.zeros(len(steps_df), dtype=np.float64)
    kv_counts = np.zeros(len(steps_df), dtype=np.int64)

    for i, (_, step_row) in enumerate(steps_df.iterrows()):
        step_start_ns = int(step_row["step.ts_start_ns"])
        step_end_ns = int(step_row["step.ts_end_ns"])

        # Active requests: start_time < step.ts_end AND end_time > step.ts_start
        # (overlapping time windows)
        active_mask = (lc_start_ns < step_end_ns) & (lc_end_ns > step_start_ns)
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            # No active requests -> all zeros (already initialized)
            continue

        kv_lengths = np.array(
            [
                _estimate_kv_length(lifecycle_df.iloc[idx], step_start_ns)
                for idx in active_indices
            ],
            dtype=np.float64,
        )

        kv_means[i] = np.mean(kv_lengths)
        kv_maxes[i] = np.max(kv_lengths)
        kv_sums[i] = np.sum(kv_lengths)
        kv_stds[i] = np.std(kv_lengths) if len(kv_lengths) > 1 else 0.0
        kv_counts[i] = len(kv_lengths)

    result = steps_df.copy()
    result["kv_mean"] = kv_means
    result["kv_max"] = kv_maxes
    result["kv_sum"] = kv_sums
    result["kv_std"] = kv_stds
    result["kv_count"] = kv_counts

    return result


def extract_all_experiments_kv_features(
    data_root: str | None = None,
) -> pd.DataFrame:
    """Load all experiments and extract KV features for every step.

    Convenience wrapper that iterates over experiment directories, loads both
    step data and lifecycle data for each, calls extract_kv_features, and
    concatenates all results.

    Args:
        data_root: Path to the ground truth data directory. Defaults to
            eval/ground_truth/ relative to this file.

    Returns:
        Concatenated DataFrame with step data enriched with KV columns,
        plus metadata columns (model, tp, workload, timestamp).
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    frames = []
    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue

        # Skip directories that lack required files
        traces_path = os.path.join(dirpath, "traces.json")
        lifecycle_path = os.path.join(
            dirpath, "results", "per_request_lifecycle_metrics.json"
        )
        if not os.path.isfile(traces_path) or not os.path.isfile(lifecycle_path):
            continue

        steps_df = load_experiment_steps(dirpath)
        lifecycle_df = load_lifecycle_data(dirpath)

        enriched = extract_kv_features(steps_df, lifecycle_df)

        meta = parse_experiment_metadata(dirname)
        enriched["model"] = meta["model"]
        enriched["tp"] = meta["tp"]
        enriched["workload"] = meta["workload"]
        enriched["timestamp"] = meta["timestamp"]

        frames.append(enriched)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["tp"] = result["tp"].astype("Int64")

    return result
