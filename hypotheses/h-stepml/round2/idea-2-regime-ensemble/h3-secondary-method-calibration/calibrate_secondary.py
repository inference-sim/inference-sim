#!/usr/bin/env python3
"""Idea 2, H3: Calibrate secondary LatencyModel methods from lifecycle data.

Extracts per-model constants for QueueingTime, OutputTokenProcessingTime,
SchedulingProcessingTime, and PreemptionProcessingTime from ground-truth
journey events, then runs an ablation study comparing StepTime-only vs
full-model BLIS predictions.

Usage:
    python3 calibrate_secondary.py \
        --artifact-dir ../h1-kv-regime-models/output/artifacts \
        [--output-dir output]
"""
import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add shared infrastructure to path
SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_loader import DEFAULT_DATA_ROOT, parse_experiment_metadata, _extract_attribute_value
from validate_blis import (
    build_workload_spec,
    extract_kv_blocks_from_vllm_log,
    load_exp_config,
    load_ground_truth_metrics,
    load_profile,
    parse_experiment_dir,
    run_blis,
)


# ---------------------------------------------------------------------------
# Journey event extraction
# ---------------------------------------------------------------------------
def load_journey_events(experiment_dir: str) -> pd.DataFrame:
    """Extract per-request journey events from traces.json.

    Returns DataFrame with columns:
        request_id, event_type, ts_ns, step_id, phase, num_preemptions
    """
    traces_path = os.path.join(experiment_dir, "traces.json")
    rows = []

    journey_events = {
        "journey.QUEUED", "journey.SCHEDULED",
        "journey.FIRST_TOKEN", "journey.FINISHED",
    }

    with open(traces_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for resource_span in data.get("resourceSpans", []):
                for scope_span in resource_span.get("scopeSpans", []):
                    for span in scope_span.get("spans", []):
                        # Request ID comes from span attributes
                        span_attrs = {}
                        for attr in span.get("attributes", []):
                            span_attrs[attr["key"]] = _extract_attribute_value(attr)

                        request_id = span_attrs.get("gen_ai.request.id", span_attrs.get("request.id", ""))

                        for event in span.get("events", []):
                            event_name = event.get("name", "")
                            if event_name not in journey_events:
                                continue

                            attrs = {}
                            for attr in event.get("attributes", []):
                                attrs[attr["key"]] = _extract_attribute_value(attr)

                            # Timestamp: prefer nanosecond field
                            ts_ns = attrs.get("ts.monotonic_ns")
                            if ts_ns is None:
                                ts_mono = attrs.get("ts.monotonic")
                                if ts_mono is not None:
                                    ts_ns = int(float(ts_mono) * 1e9)

                            rows.append({
                                "request_id": request_id,
                                "event_type": event_name,
                                "ts_ns": ts_ns,
                                "step_id": attrs.get("scheduler.step"),
                                "phase": attrs.get("phase"),
                                "num_preemptions": attrs.get("num_preemptions", 0),
                                "schedule_kind": attrs.get("schedule.kind"),
                            })

    return pd.DataFrame(rows)


def compute_secondary_constants(data_root: str) -> dict[str, dict]:
    """Compute per-model secondary method constants from journey events.

    Returns dict mapping model_key -> {
        queueing_time_us: float,          # median queue-to-schedule delay
        output_token_processing_us: float, # median per-token overhead beyond step time
        scheduling_processing_us: float,   # median per-request scheduling overhead
        preemption_processing_us: float,   # median preemption overhead (0 if no preemptions)
    }
    """
    model_queue_delays = {}  # model_key -> list of queue-to-schedule delays (µs)
    model_schedule_delays = {}  # model_key -> list of per-request scheduling overhead (µs)

    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        traces_path = os.path.join(dirpath, "traces.json")
        if not os.path.isfile(traces_path):
            continue

        try:
            meta = parse_experiment_metadata(dirname)
        except ValueError:
            continue

        model_key = f"{meta['model']}_tp{meta['tp']}"

        print(f"  Extracting journey events: {dirname} ({model_key})")
        events = load_journey_events(dirpath)
        if events.empty:
            print(f"    No journey events found")
            continue

        # Pivot to per-request timing
        pivot = events.pivot_table(
            index="request_id",
            columns="event_type",
            values="ts_ns",
            aggfunc="first",
        )

        # Queue-to-schedule delay
        if "journey.QUEUED" in pivot.columns and "journey.SCHEDULED" in pivot.columns:
            delays = (pivot["journey.SCHEDULED"] - pivot["journey.QUEUED"]) / 1000  # ns -> µs
            delays = delays.dropna()
            delays = delays[delays >= 0]
            if len(delays) > 0:
                if model_key not in model_queue_delays:
                    model_queue_delays[model_key] = []
                model_queue_delays[model_key].extend(delays.tolist())

        # Schedule-to-first-token (captures scheduling + prefill overhead)
        if "journey.SCHEDULED" in pivot.columns and "journey.FIRST_TOKEN" in pivot.columns:
            delays = (pivot["journey.FIRST_TOKEN"] - pivot["journey.SCHEDULED"]) / 1000  # ns -> µs
            delays = delays.dropna()
            delays = delays[delays >= 0]
            if len(delays) > 0:
                if model_key not in model_schedule_delays:
                    model_schedule_delays[model_key] = []
                model_schedule_delays[model_key].extend(delays.tolist())

    # Aggregate per model
    results = {}
    for model_key in set(list(model_queue_delays.keys()) + list(model_schedule_delays.keys())):
        queue_delays = model_queue_delays.get(model_key, [])
        sched_delays = model_schedule_delays.get(model_key, [])

        results[model_key] = {
            "queueing_time_us": float(np.median(queue_delays)) if queue_delays else 0.0,
            "queueing_time_p25_us": float(np.percentile(queue_delays, 25)) if queue_delays else 0.0,
            "queueing_time_p75_us": float(np.percentile(queue_delays, 75)) if queue_delays else 0.0,
            "n_requests": len(queue_delays),
            "scheduling_to_first_token_us": float(np.median(sched_delays)) if sched_delays else 0.0,
            # No preemptions in data
            "preemption_processing_us": 0.0,
            # OutputTokenProcessingTime derived separately (from ITL residual)
            "output_token_processing_us": 0.0,
        }

        print(f"\n  {model_key}:")
        print(f"    Queue-to-schedule: median={results[model_key]['queueing_time_us']:.0f}µs "
              f"(p25={results[model_key]['queueing_time_p25_us']:.0f}, "
              f"p75={results[model_key]['queueing_time_p75_us']:.0f}), "
              f"n={results[model_key]['n_requests']}")
        print(f"    Schedule-to-first-token: median={results[model_key]['scheduling_to_first_token_us']:.0f}µs")

    return results


# ---------------------------------------------------------------------------
# Artifact modification
# ---------------------------------------------------------------------------
def create_calibrated_artifact(
    base_artifact_path: str,
    constants: dict,
    output_path: str,
):
    """Copy a base StepML artifact and add non-zero secondary method constants."""
    with open(base_artifact_path) as f:
        artifact = json.load(f)

    # Set secondary method constants
    artifact["output_token_processing_time_us"] = constants.get("output_token_processing_us", 0)
    artifact["scheduling_processing_time_us"] = constants.get("scheduling_processing_us", 0)
    artifact["preemption_processing_time_us"] = constants.get("preemption_processing_us", 0)

    # QueueingTime as a simple constant model (intercept-only)
    queueing_us = constants.get("queueing_time_us", 0)
    if queueing_us > 0:
        artifact["queueing_time"] = {
            "model_type": "linear",
            "intercept": queueing_us,
            "feature_coefficients": {},
        }

    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# BLIS validation runner
# ---------------------------------------------------------------------------
def run_validation(
    binary: str,
    data_root: str,
    artifact_dir: str,
    mode: str = "regime",
) -> pd.DataFrame:
    """Run BLIS validation and return results DataFrame."""
    results = []

    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
        if not os.path.isfile(summary_path):
            continue

        try:
            meta = parse_experiment_dir(dirname)
        except ValueError:
            continue

        model_key = f"{meta['model']}_tp{meta['tp']}"
        suffix = f"_{mode}.json"
        artifact_path = os.path.join(artifact_dir, model_key + suffix)
        if not os.path.isfile(artifact_path):
            # Try normalized
            normalized = model_key.replace("-hf_", "_")
            artifact_path = os.path.join(artifact_dir, normalized + suffix)
            if not os.path.isfile(artifact_path):
                continue

        gt = load_ground_truth_metrics(dirpath)
        exp_config = load_exp_config(dirpath)
        total_kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
        if total_kv_blocks is None:
            continue

        try:
            profile = load_profile(dirpath)
        except Exception:
            continue

        workload_spec = build_workload_spec(profile, gt)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(workload_spec, f, default_flow_style=False)
            spec_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results_json_path = f.name

        try:
            blis_metrics = run_blis(
                binary=binary,
                workload_spec_path=spec_path,
                exp_config=exp_config,
                total_kv_blocks=total_kv_blocks,
                alpha_coeffs=[0, 0, 0],
                beta_coeffs=[0, 0, 0],
                results_path=results_json_path,
                stepml_model_path=artifact_path,
            )
        finally:
            os.unlink(spec_path)
            if os.path.exists(results_json_path):
                os.unlink(results_json_path)

        if blis_metrics is None:
            continue

        gt_e2e = gt["e2e_mean_s"] * 1000
        gt_ttft = gt["ttft_mean_s"] * 1000
        gt_itl = gt["itl_mean_s"] * 1000

        results.append({
            "experiment": dirname,
            "model": meta["model"],
            "workload": meta["workload"],
            "model_key": model_key,
            "gt_e2e_ms": gt_e2e,
            "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
            "e2e_error_pct": abs(blis_metrics["e2e_mean_ms"] - gt_e2e) / gt_e2e * 100,
            "gt_ttft_ms": gt_ttft,
            "blis_ttft_ms": blis_metrics["ttft_mean_ms"],
            "ttft_error_pct": abs(blis_metrics["ttft_mean_ms"] - gt_ttft) / gt_ttft * 100 if gt_ttft > 0 else 0,
            "gt_itl_ms": gt_itl,
            "blis_itl_ms": blis_metrics["itl_mean_ms"],
            "itl_error_pct": abs(blis_metrics["itl_mean_ms"] - gt_itl) / gt_itl * 100 if gt_itl > 0 else 0,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Base artifact directory")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "output"))
    parser.add_argument("--binary", default=str(Path(__file__).resolve().parents[4] / "simulation_worker"))
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Extract secondary method constants from lifecycle data
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Extracting secondary method constants")
    print("=" * 60)

    constants = compute_secondary_constants(args.data_root)

    constants_path = os.path.join(args.output_dir, "secondary_constants.json")
    with open(constants_path, "w") as f:
        json.dump(constants, f, indent=2)
    print(f"\nConstants saved to: {constants_path}")

    # -----------------------------------------------------------------------
    # Step 2: Create calibrated artifacts (non-zero secondary methods)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Creating calibrated artifacts")
    print("=" * 60)

    calibrated_dir = os.path.join(args.output_dir, "calibrated_artifacts")
    os.makedirs(calibrated_dir, exist_ok=True)

    for model_key, consts in constants.items():
        base_path = os.path.join(args.artifact_dir, f"{model_key}_regime.json")
        if not os.path.isfile(base_path):
            print(f"  {model_key}: no base artifact, skipping")
            continue

        calibrated_path = os.path.join(calibrated_dir, f"{model_key}_regime.json")

        # For SchedulingProcessingTime, use the queue-to-schedule delay
        # (this is per-request scheduling overhead, distinct from QueueingTime)
        calibrated_consts = {
            "queueing_time_us": consts["queueing_time_us"],
            "scheduling_processing_us": consts["queueing_time_us"],  # queue→schedule is scheduling overhead
            "output_token_processing_us": 0,  # keep 0 (ITL already good)
            "preemption_processing_us": 0,  # no preemptions in data
        }

        create_calibrated_artifact(base_path, calibrated_consts, calibrated_path)
        print(f"  {model_key}: queueing={consts['queueing_time_us']:.0f}µs, "
              f"scheduling={consts['queueing_time_us']:.0f}µs")

    # -----------------------------------------------------------------------
    # Step 3: Run Config A (StepTime-only, baseline = current h2 artifacts)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Config A — StepTime-only (baseline)")
    print("=" * 60)

    df_a = run_validation(
        binary=args.binary,
        data_root=args.data_root,
        artifact_dir=args.artifact_dir,
    )
    if df_a.empty:
        print("ERROR: No experiments completed for Config A")
        return

    df_a.to_csv(os.path.join(args.output_dir, "config_a_steptime_only.csv"), index=False)

    # -----------------------------------------------------------------------
    # Step 4: Config B (StepTime + calibrated secondary methods)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Config B — StepTime + calibrated secondary methods")
    print("=" * 60)

    df_b = run_validation(
        binary=args.binary,
        data_root=args.data_root,
        artifact_dir=calibrated_dir,
    )
    if df_b.empty:
        print("ERROR: No experiments completed for Config B")
        return

    df_b.to_csv(os.path.join(args.output_dir, "config_b_calibrated.csv"), index=False)

    # -----------------------------------------------------------------------
    # Step 5: Ablation comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ABLATION RESULTS: Config A (StepTime-only) vs Config B (Calibrated)")
    print("=" * 70)

    merged = pd.merge(
        df_a[["experiment", "model", "workload", "model_key",
              "e2e_error_pct", "ttft_error_pct", "itl_error_pct"]],
        df_b[["experiment", "e2e_error_pct", "ttft_error_pct", "itl_error_pct"]],
        on="experiment",
        suffixes=("_a", "_b"),
    )

    merged["e2e_diff_pp"] = merged["e2e_error_pct_a"] - merged["e2e_error_pct_b"]
    merged["ttft_diff_pp"] = merged["ttft_error_pct_a"] - merged["ttft_error_pct_b"]
    merged["itl_diff_pp"] = merged["itl_error_pct_a"] - merged["itl_error_pct_b"]

    print(f"\n  {'Experiment':<55s} {'E2E A%':>7s} {'E2E B%':>7s} {'Δpp':>6s}")
    print(f"  {'-'*55} {'-'*7} {'-'*7} {'-'*6}")
    for _, row in merged.iterrows():
        indicator = "+" if row["e2e_diff_pp"] > 0 else "-" if row["e2e_diff_pp"] < 0 else " "
        print(f"  {row['experiment']:<55s} {row['e2e_error_pct_a']:>6.1f}% {row['e2e_error_pct_b']:>6.1f}% {indicator}{abs(row['e2e_diff_pp']):>4.1f}")

    mean_e2e_a = merged["e2e_error_pct_a"].mean()
    mean_e2e_b = merged["e2e_error_pct_b"].mean()
    mean_ttft_a = merged["ttft_error_pct_a"].mean()
    mean_ttft_b = merged["ttft_error_pct_b"].mean()
    mean_itl_a = merged["itl_error_pct_a"].mean()
    mean_itl_b = merged["itl_error_pct_b"].mean()

    print(f"\n  Mean E2E:  Config A = {mean_e2e_a:.1f}%  Config B = {mean_e2e_b:.1f}%  Δ = {mean_e2e_a - mean_e2e_b:+.1f}pp")
    print(f"  Mean TTFT: Config A = {mean_ttft_a:.1f}%  Config B = {mean_ttft_b:.1f}%  Δ = {mean_ttft_a - mean_ttft_b:+.1f}pp")
    print(f"  Mean ITL:  Config A = {mean_itl_a:.1f}%  Config B = {mean_itl_b:.1f}%  Δ = {mean_itl_a - mean_itl_b:+.1f}pp")

    # Dense-only results (exclude Mixtral)
    dense = merged[~merged["model"].str.contains("mixtral")]
    if not dense.empty:
        d_e2e_a = dense["e2e_error_pct_a"].mean()
        d_e2e_b = dense["e2e_error_pct_b"].mean()
        d_ttft_a = dense["ttft_error_pct_a"].mean()
        d_ttft_b = dense["ttft_error_pct_b"].mean()
        d_itl_a = dense["itl_error_pct_a"].mean()
        d_itl_b = dense["itl_error_pct_b"].mean()

        print(f"\n  Dense only ({len(dense)} experiments):")
        print(f"    Mean E2E:  A = {d_e2e_a:.1f}%  B = {d_e2e_b:.1f}%  Δ = {d_e2e_a - d_e2e_b:+.1f}pp")
        print(f"    Mean TTFT: A = {d_ttft_a:.1f}%  B = {d_ttft_b:.1f}%  Δ = {d_ttft_a - d_ttft_b:+.1f}pp")
        print(f"    Mean ITL:  A = {d_itl_a:.1f}%  B = {d_itl_b:.1f}%  Δ = {d_itl_a - d_itl_b:+.1f}pp")

    # Hypothesis evaluation
    e2e_improvement = mean_e2e_a - mean_e2e_b
    print(f"\n  {'='*70}")
    if e2e_improvement >= 5:
        print(f"  HYPOTHESIS SUPPORTED: E2E improvement = {e2e_improvement:.1f}pp >= 5pp threshold")
    elif e2e_improvement >= 3:
        print(f"  HYPOTHESIS WEAK: E2E improvement = {e2e_improvement:.1f}pp (above noise but below 5pp)")
    elif e2e_improvement < 0:
        print(f"  HYPOTHESIS REFUTED: Secondary methods WORSENED E2E by {abs(e2e_improvement):.1f}pp")
    else:
        print(f"  HYPOTHESIS REFUTED: E2E improvement = {e2e_improvement:.1f}pp < 3pp noise threshold")
    print(f"  {'='*70}")

    # Save comparison
    merged.to_csv(os.path.join(args.output_dir, "ablation_comparison.csv"), index=False)

    summary = {
        "config_a_mean_e2e_pct": mean_e2e_a,
        "config_b_mean_e2e_pct": mean_e2e_b,
        "e2e_improvement_pp": e2e_improvement,
        "config_a_mean_ttft_pct": mean_ttft_a,
        "config_b_mean_ttft_pct": mean_ttft_b,
        "config_a_mean_itl_pct": mean_itl_a,
        "config_b_mean_itl_pct": mean_itl_b,
        "n_experiments": len(merged),
        "hypothesis_status": "supported" if e2e_improvement >= 5 else "refuted",
    }
    with open(os.path.join(args.output_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
