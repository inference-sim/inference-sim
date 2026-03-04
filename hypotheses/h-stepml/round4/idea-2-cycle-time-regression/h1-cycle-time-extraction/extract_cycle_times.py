"""H1: Extract per-step cycle times from lifecycle per-token timestamps.

For each step in each experiment:
1. Load per-request lifecycle data (output_token_times)
2. For each request active during that step, compute ITI = t[i+1] - t[i]
3. Match ITIs to steps by timestamp overlap
4. Compute per-step cycle_time = median ITI across active decode requests
5. Compare cycle_time vs step.duration_us

The key insight: step.duration_us captures GPU forward-pass time only.
The real step cycle time (what determines E2E) includes CPU overhead.
cycle_time / step.duration_us should be >1 for small batches (overhead-dominated)
and ~1 for large batches (compute-dominated).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Add shared infrastructure to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

from data_loader import (
    DEFAULT_DATA_ROOT,
    load_experiment_steps,
    load_lifecycle_data,
    parse_experiment_metadata,
)
from evaluation import compute_mape, compute_pearson_r


def extract_itis_for_experiment(experiment_dir: str) -> pd.DataFrame:
    """Extract per-token inter-token intervals from lifecycle data.

    Returns DataFrame with columns:
        request_id, token_index, iti_us, timestamp_s, input_tokens, output_tokens
    """
    lifecycle_df = load_lifecycle_data(experiment_dir)

    rows = []
    for req_id, req in lifecycle_df.iterrows():
        times = req["output_token_times"]
        if not times or len(times) < 2:
            continue

        for i in range(1, len(times)):
            iti_us = (times[i] - times[i - 1]) * 1e6  # seconds -> microseconds
            rows.append({
                "request_id": req_id,
                "token_index": i,
                "iti_us": iti_us,
                "timestamp_s": times[i],
                "input_tokens": req["input_tokens"],
                "output_tokens": req["output_tokens"],
            })

    return pd.DataFrame(rows)


def match_itis_to_steps(
    iti_df: pd.DataFrame, steps_df: pd.DataFrame
) -> pd.DataFrame:
    """Match ITI observations to step windows by timestamp overlap.

    For each step, finds ITIs whose timestamp falls within the step's
    [ts_start_ns, ts_end_ns] window. Computes per-step cycle_time
    as the median ITI across matched tokens.

    Returns enriched steps_df with new columns:
        cycle_time_us, matched_iti_count, cycle_time_ratio
    """
    # Convert step timestamps to seconds for comparison with ITI timestamps
    step_starts_s = steps_df["step.ts_start_ns"].values / 1e9
    step_ends_s = steps_df["step.ts_end_ns"].values / 1e9

    cycle_times = np.full(len(steps_df), np.nan)
    matched_counts = np.zeros(len(steps_df), dtype=int)

    iti_timestamps = iti_df["timestamp_s"].values
    iti_values = iti_df["iti_us"].values

    for i in range(len(steps_df)):
        # Find ITIs that fall within this step's time window
        mask = (iti_timestamps >= step_starts_s[i]) & (
            iti_timestamps <= step_ends_s[i]
        )
        count = np.sum(mask)
        matched_counts[i] = count

        if count > 0:
            matched_itis = iti_values[mask]
            cycle_times[i] = np.median(matched_itis)

    result = steps_df.copy()
    result["cycle_time_us"] = cycle_times
    result["matched_iti_count"] = matched_counts

    # Compute ratio: cycle_time / step.duration_us
    duration = result["step.duration_us"].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["cycle_time_ratio"] = np.where(
            duration > 0, cycle_times / duration, np.nan
        )

    return result


def analyze_experiment(experiment_dir: str, experiment_id: str) -> dict:
    """Run full cycle-time extraction and analysis for one experiment."""
    steps_df = load_experiment_steps(experiment_dir)
    iti_df = extract_itis_for_experiment(experiment_dir)

    if len(iti_df) == 0:
        return {
            "experiment_id": experiment_id,
            "status": "no_iti_data",
            "total_steps": len(steps_df),
            "matched_steps": 0,
        }

    enriched = match_itis_to_steps(iti_df, steps_df)

    # Filter to steps with valid cycle times
    valid = enriched.dropna(subset=["cycle_time_us"])
    matched_fraction = len(valid) / len(steps_df) if len(steps_df) > 0 else 0

    if len(valid) == 0:
        return {
            "experiment_id": experiment_id,
            "status": "no_matches",
            "total_steps": len(steps_df),
            "matched_steps": 0,
            "total_itis": len(iti_df),
        }

    # Compute correlation between cycle_time and step.duration_us
    ct = valid["cycle_time_us"].values
    dur = valid["step.duration_us"].values.astype(float)

    # Filter out zero durations
    nonzero = dur > 0
    ct_nz = ct[nonzero]
    dur_nz = dur[nonzero]

    if len(ct_nz) > 2:
        pearson_r = compute_pearson_r(ct_nz, dur_nz)
    else:
        pearson_r = float("nan")

    # Ratio analysis by regime
    ratio = valid["cycle_time_ratio"].values
    ratio_valid = ratio[~np.isnan(ratio) & ~np.isinf(ratio)]

    # Decode-only vs mixed
    is_decode_only = valid["batch.prefill_tokens"].values == 0
    decode_ratios = ratio[is_decode_only & ~np.isnan(ratio) & ~np.isinf(ratio)]
    mixed_ratios = ratio[~is_decode_only & ~np.isnan(ratio) & ~np.isinf(ratio)]

    # Large batch vs small batch (threshold: running_depth > 32)
    running_depth = valid["queue.running_depth"].values
    large_batch = running_depth > 32
    small_batch = running_depth <= 32

    large_ratios = ratio[large_batch & ~np.isnan(ratio) & ~np.isinf(ratio)]
    small_ratios = ratio[small_batch & ~np.isnan(ratio) & ~np.isinf(ratio)]

    return {
        "experiment_id": experiment_id,
        "status": "ok",
        "total_steps": len(steps_df),
        "total_itis": len(iti_df),
        "matched_steps": len(valid),
        "matched_fraction": float(matched_fraction),
        "pearson_r": float(pearson_r),
        "cycle_time_median_us": float(np.median(ct)),
        "cycle_time_mean_us": float(np.mean(ct)),
        "duration_median_us": float(np.median(dur_nz)) if len(dur_nz) > 0 else 0,
        "duration_mean_us": float(np.mean(dur_nz)) if len(dur_nz) > 0 else 0,
        "ratio_median": float(np.median(ratio_valid)) if len(ratio_valid) > 0 else 0,
        "ratio_mean": float(np.mean(ratio_valid)) if len(ratio_valid) > 0 else 0,
        "ratio_p10": float(np.percentile(ratio_valid, 10)) if len(ratio_valid) > 0 else 0,
        "ratio_p90": float(np.percentile(ratio_valid, 90)) if len(ratio_valid) > 0 else 0,
        "decode_ratio_median": float(np.median(decode_ratios)) if len(decode_ratios) > 0 else 0,
        "mixed_ratio_median": float(np.median(mixed_ratios)) if len(mixed_ratios) > 0 else 0,
        "decode_steps": int(np.sum(is_decode_only)),
        "mixed_steps": int(np.sum(~is_decode_only)),
        "large_batch_ratio_median": float(np.median(large_ratios)) if len(large_ratios) > 0 else 0,
        "small_batch_ratio_median": float(np.median(small_ratios)) if len(small_ratios) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="H1: Cycle-Time Extraction")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results = []
    all_enriched_frames = []

    for dirname in sorted(os.listdir(args.data_root)):
        dirpath = os.path.join(args.data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        traces_path = os.path.join(dirpath, "traces.json")
        lifecycle_path = os.path.join(dirpath, "results", "per_request_lifecycle_metrics.json")
        if not os.path.isfile(traces_path) or not os.path.isfile(lifecycle_path):
            continue

        print(f"\nProcessing: {dirname}")
        meta = parse_experiment_metadata(dirname)
        result = analyze_experiment(dirpath, dirname)
        result["model"] = meta["model"]
        result["workload"] = meta["workload"]
        result["tp"] = meta["tp"]
        results.append(result)

        # Also save enriched step data for later use by H2
        if result["status"] == "ok":
            steps_df = load_experiment_steps(dirpath)
            iti_df = extract_itis_for_experiment(dirpath)
            enriched = match_itis_to_steps(iti_df, steps_df)
            enriched["model"] = meta["model"]
            enriched["workload"] = meta["workload"]
            enriched["tp"] = meta["tp"]
            all_enriched_frames.append(enriched)

        # Print summary
        if result["status"] == "ok":
            print(f"  Matched: {result['matched_steps']}/{result['total_steps']} "
                  f"({result['matched_fraction']*100:.1f}%)")
            print(f"  Pearson r (cycle_time vs duration): {result['pearson_r']:.3f}")
            print(f"  Cycle time median: {result['cycle_time_median_us']:.0f} us")
            print(f"  Duration median: {result['duration_median_us']:.0f} us")
            print(f"  Ratio median: {result['ratio_median']:.2f}")
            print(f"  Decode ratio: {result['decode_ratio_median']:.2f}, "
                  f"Mixed ratio: {result['mixed_ratio_median']:.2f}")
            print(f"  Large batch ratio: {result['large_batch_ratio_median']:.2f}, "
                  f"Small batch ratio: {result['small_batch_ratio_median']:.2f}")
        else:
            print(f"  Status: {result['status']}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "h1_results.csv"), index=False)

    with open(os.path.join(args.output_dir, "h1_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save enriched step data for H2
    if all_enriched_frames:
        all_enriched = pd.concat(all_enriched_frames, ignore_index=True)
        all_enriched.to_parquet(
            os.path.join(args.output_dir, "enriched_steps.parquet"), index=False
        )
        print(f"\nSaved {len(all_enriched)} enriched steps to enriched_steps.parquet")

    # Print summary table
    print("\n" + "=" * 80)
    print("H1 SUMMARY: Cycle-Time Extraction Results")
    print("=" * 80)

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        print(f"\n{'Experiment':<50} {'Match%':>7} {'r':>7} {'Ratio':>7} {'CT(us)':>8} {'Dur(us)':>8}")
        print("-" * 90)
        for r in ok_results:
            print(f"{r['experiment_id']:<50} "
                  f"{r['matched_fraction']*100:>6.1f}% "
                  f"{r['pearson_r']:>7.3f} "
                  f"{r['ratio_median']:>7.2f} "
                  f"{r['cycle_time_median_us']:>8.0f} "
                  f"{r['duration_median_us']:>8.0f}")

        # Aggregate
        mean_match = np.mean([r["matched_fraction"] for r in ok_results])
        mean_r = np.mean([r["pearson_r"] for r in ok_results])
        mean_ratio = np.mean([r["ratio_median"] for r in ok_results])

        print("-" * 90)
        print(f"{'MEAN':<50} {mean_match*100:>6.1f}% {mean_r:>7.3f} {mean_ratio:>7.2f}")

        print(f"\nVERDICT:")
        print(f"  Mean matched fraction: {mean_match*100:.1f}% (target: >50%)")
        print(f"  Mean Pearson r: {mean_r:.3f} (target: >0.7 for compute-dominated)")
        print(f"  Mean cycle_time/duration ratio: {mean_ratio:.2f}")
        if mean_ratio > 1.0:
            print(f"  Ratio > 1 confirms cycle time includes overhead beyond GPU compute")
        if mean_match >= 0.5:
            print(f"  PASS: >50% of steps successfully matched to lifecycle data")
        else:
            print(f"  FAIL: <50% of steps matched — data quality insufficient")
    else:
        print("No experiments produced valid results!")


if __name__ == "__main__":
    main()
