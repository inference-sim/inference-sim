#!/usr/bin/env python3
"""Run H3: BLIS E2E Validation + 34B Investigation.

Exports the best model from H1 (3-coeff OLS with kv_sum) as StepML artifacts,
runs BLIS E2E validation, and performs 34B-specific deep-dive analysis.

Uses trace replay (from idea-1 findings) for accurate E2E comparison.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add shared/ to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "shared"))
sys.path.insert(0, _SHARED_DIR)

from data_loader import load_all_experiments, parse_experiment_metadata
from evaluation import compute_mape, compute_mspe, compute_p99_error, compute_pearson_r
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import temporal_split

import yaml

DATA_ROOT = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/BLIS-research/eval/ground_truth"
REPO_ROOT = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/stepml-worktrees/round3-idea-2-total-context-model"
BINARY = os.path.join(REPO_ROOT, "simulation_worker")
ARTIFACT_DIR = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/BLIS-research/hypotheses/h-stepml/round2/idea-2-regime-ensemble/h3-secondary-method-calibration/output/calibrated_artifacts"

TARGET_COL = "step.duration_us"
H3_OUTPUT = os.path.join(_SCRIPT_DIR, "h3-blis-e2e-34b", "output")


def get_overhead_floor(model_tp):
    """Load overhead floor from Round 2 calibrated artifacts."""
    fname = f"{model_tp}_regime.json"
    path = os.path.join(ARTIFACT_DIR, fname)
    if os.path.isfile(path):
        with open(path) as f:
            art = json.load(f)
        return art.get("step_overhead_us", 0)
    return 0


def train_3coeff_models(df, splits):
    """Train 3-coeff OLS models (best from H1) for each model_tp."""
    models = {}
    model_tps = sorted(df["model_tp"].unique())

    for model_tp in model_tps:
        mask = df["model_tp"] == model_tp
        train_mask = mask & df.index.isin(splits["train"])
        train_df = df[train_mask]

        if len(train_df) == 0:
            continue

        # new_tokens = prefill + decode
        X = train_df[["batch.prefill_tokens", "batch.decode_tokens", "kv_sum"]].values.astype(np.float64)
        y = train_df[TARGET_COL].values.astype(np.float64)

        # Also train the combined new_tokens version
        X_3 = np.column_stack([
            train_df["batch.prefill_tokens"].values + train_df["batch.decode_tokens"].values,
            train_df["kv_sum"].values
        ]).astype(np.float64)

        ols_4 = LinearRegression().fit(X, y)
        ols_3 = LinearRegression().fit(X_3, y)

        overhead = get_overhead_floor(model_tp)

        models[model_tp] = {
            "ols_4coeff": ols_4,  # a + b*prefill + c*decode + d*kv_sum
            "ols_3coeff": ols_3,  # a + b*new_tokens + c*kv_sum
            "overhead_us": overhead,
        }

        print(f"  {model_tp}: 4-coeff = [{ols_4.intercept_:.2f}, {ols_4.coef_[0]:.6f}, {ols_4.coef_[1]:.6f}, {ols_4.coef_[2]:.6f}], overhead={overhead:.1f}")
        print(f"  {model_tp}: 3-coeff = [{ols_3.intercept_:.2f}, {ols_3.coef_[0]:.6f}, {ols_3.coef_[1]:.6f}]")

    return models


def export_stepml_artifact(model_tp, ols_model, overhead_us, variant="4coeff"):
    """Export OLS model as StepML artifact JSON."""
    if variant == "4coeff":
        art = {
            "version": 2,
            "step_time": {
                "model_type": "linear",
                "intercept": float(ols_model.intercept_),
                "feature_coefficients": {
                    "prefill_tokens": float(ols_model.coef_[0]),
                    "decode_tokens": float(ols_model.coef_[1]),
                    "kv_sum": float(ols_model.coef_[2]),
                }
            },
            "step_overhead_us": float(overhead_us),
            "output_token_processing_time_us": 0,
            "scheduling_processing_time_us": 0,
            "preemption_processing_time_us": 0,
        }
    else:
        art = {
            "version": 2,
            "step_time": {
                "model_type": "linear",
                "intercept": float(ols_model.intercept_),
                "feature_coefficients": {
                    "scheduled_tokens": float(ols_model.coef_[0]),
                    "kv_sum": float(ols_model.coef_[1]),
                }
            },
            "step_overhead_us": float(overhead_us),
            "output_token_processing_time_us": 0,
            "scheduling_processing_time_us": 0,
            "preemption_processing_time_us": 0,
        }
    return art


def parse_blis_stdout(stdout):
    """Parse BLIS stdout for metrics JSON."""
    lines = stdout.split("\n")
    json_lines = []
    in_json = False
    brace_depth = 0

    for line in lines:
        if "Simulation Metrics" in line:
            in_json = False
            json_lines = []
            continue
        if not in_json and line.strip().startswith("{"):
            in_json = True
        if in_json:
            json_lines.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                break

    if not json_lines:
        return None
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None


def load_ground_truth_metrics(experiment_dir):
    """Load ground truth metrics from experiment."""
    summary_path = os.path.join(experiment_dir, "results", "summary_lifecycle_metrics.json")
    with open(summary_path) as f:
        data = json.load(f)
    successes = data.get("successes", {})
    latency = successes.get("latency", {})
    return {
        "e2e_mean_s": latency.get("request_latency", {}).get("mean", 0),
        "ttft_mean_s": latency.get("time_to_first_token", {}).get("mean", 0),
        "itl_mean_s": latency.get("inter_token_latency", {}).get("mean", 0),
        "num_requests": data.get("load_summary", {}).get("count", 0),
    }


def load_exp_config(experiment_dir):
    """Load exp-config.yaml."""
    with open(os.path.join(experiment_dir, "exp-config.yaml")) as f:
        return yaml.safe_load(f)


def extract_kv_blocks(experiment_dir):
    """Parse vllm.log for GPU KV cache tokens."""
    vllm_log = os.path.join(experiment_dir, "vllm.log")
    if not os.path.isfile(vllm_log):
        return None
    pattern = re.compile(r"GPU KV cache size:\s+([\d,]+)\s+tokens")
    with open(vllm_log) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return int(m.group(1).replace(",", "")) // 16
    return None


def convert_lifecycle_to_csv(experiment_dir, output_csv, model_name, seed=42):
    """Convert per_request_lifecycle_metrics.json to BLIS legacy trace CSV.

    BLIS expects 5 columns: arrival_time,request_id,model,prefill_tokens,decode_tokens
    with JSON arrays of token IDs.
    """
    import random
    metrics_path = os.path.join(experiment_dir, "results", "per_request_lifecycle_metrics.json")
    with open(metrics_path) as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"Empty lifecycle data: {metrics_path}")

    rng = random.Random(seed)
    base_time = min(entry["start_time"] for entry in data)

    rows = []
    for i, entry in enumerate(data):
        info = entry.get("info", {})
        input_count = info.get("input_tokens", 0)
        output_count = info.get("output_tokens", 0)
        arrival_s = entry["start_time"] - base_time
        input_tokens = [rng.randint(0, 31999) for _ in range(input_count)]
        output_tokens = [rng.randint(0, 31999) for _ in range(output_count)]
        rows.append((arrival_s, i, model_name, input_tokens, output_tokens))

    rows.sort(key=lambda x: x[0])

    with open(output_csv, "w") as f:
        f.write("arrival_time,request_id,model,prefill_tokens,decode_tokens\n")
        for arrival_s, req_id, model, inp, out in rows:
            f.write(
                f"{arrival_s:.6f},request_{req_id},{model},"
                f'"{json.dumps(inp)}","{json.dumps(out)}"\n'
            )

    duration_s = (rows[-1][0] - rows[0][0]) if len(rows) > 1 else 0
    return len(rows), duration_s


def build_trace_workload_spec(trace_csv, num_requests, horizon_us):
    """Build a workload spec for trace replay."""
    return {
        "version": "2",
        "seed": 42,
        "num_requests": num_requests,
        "horizon": horizon_us,
        "traces_filepath": trace_csv,
    }


def run_blis_with_artifact(experiment_dir, artifact_path, use_trace_replay=True):
    """Run BLIS for one experiment with StepML artifact."""
    dirname = os.path.basename(experiment_dir)
    gt = load_ground_truth_metrics(experiment_dir)
    exp_config = load_exp_config(experiment_dir)
    kv_blocks = extract_kv_blocks(experiment_dir)

    if kv_blocks is None:
        return {"experiment": dirname, "status": "no_kv_blocks"}

    model_name = exp_config.get("model", "unknown")
    tp = exp_config.get("tensor_parallelism", 1)
    max_model_len = exp_config.get("max_model_len", 4096)
    max_num_seqs = exp_config.get("max_num_seqs", 128)
    max_num_batched_tokens = exp_config.get("max_num_batched_tokens", 2048)

    if use_trace_replay:
        # Convert lifecycle data to trace CSV
        trace_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        n_reqs, duration_s = convert_lifecycle_to_csv(experiment_dir, trace_csv, model_name)
        horizon_us = int(duration_s * 1_000_000) + 120_000_000

        cmd = [
            BINARY, "run",
            "--model", model_name,
            "--workload", "traces",
            "--workload-traces-filepath", trace_csv,
            "--tp", str(tp),
            "--max-model-len", str(max_model_len),
            "--max-num-running-reqs", str(max_num_seqs),
            "--max-num-scheduled-tokens", str(max_num_batched_tokens),
            "--total-kv-blocks", str(kv_blocks),
            "--block-size-in-tokens", "16",
            "--horizon", str(horizon_us),
            "--stepml-model", artifact_path,
            "--alpha-coeffs=1,0,0",
            "--beta-coeffs=1,0,0",
            "--log", "error",
        ]
    else:
        # Use workload-spec mode (from validate_blis.py)
        trace_csv = None

        profile_path = os.path.join(experiment_dir, "profile.yaml")
        with open(profile_path) as f:
            content = f.read()
        try:
            profile = json.loads(content)
        except json.JSONDecodeError:
            profile = yaml.safe_load(content)

        # Build workload spec
        sys.path.insert(0, _SHARED_DIR)
        from validate_blis import build_workload_spec
        workload_spec = build_workload_spec(profile, gt)

        spec_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(workload_spec, spec_file, default_flow_style=False)
        spec_file.close()

        cmd = [
            BINARY, "run",
            "--model", model_name,
            "--workload-spec", spec_file.name,
            "--tp", str(tp),
            "--max-model-len", str(max_model_len),
            "--max-num-running-reqs", str(max_num_seqs),
            "--max-num-scheduled-tokens", str(max_num_batched_tokens),
            "--total-kv-blocks", str(kv_blocks),
            "--block-size-in-tokens", "16",
            "--stepml-model", artifact_path,
            "--alpha-coeffs=1,0,0",
            "--beta-coeffs=1,0,0",
            "--log", "error",
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"experiment": dirname, "status": f"error: {e}"}
    finally:
        if trace_csv and os.path.exists(trace_csv):
            os.unlink(trace_csv)

    if result.returncode != 0:
        return {"experiment": dirname, "status": f"failed: {result.stderr[:200]}"}

    metrics = parse_blis_stdout(result.stdout)
    if not metrics:
        return {"experiment": dirname, "status": "no_metrics_in_stdout"}

    # Compute errors
    gt_e2e = gt["e2e_mean_s"] * 1000
    gt_ttft = gt["ttft_mean_s"] * 1000
    gt_itl = gt["itl_mean_s"] * 1000

    blis_e2e = metrics.get("e2e_mean_ms", 0)
    blis_ttft = metrics.get("ttft_mean_ms", 0)
    blis_itl = metrics.get("itl_mean_ms", 0)

    return {
        "experiment": dirname,
        "status": "ok",
        "gt_e2e_ms": gt_e2e,
        "gt_ttft_ms": gt_ttft,
        "gt_itl_ms": gt_itl,
        "blis_e2e_ms": blis_e2e,
        "blis_ttft_ms": blis_ttft,
        "blis_itl_ms": blis_itl,
        "e2e_error": abs(blis_e2e - gt_e2e) / gt_e2e if gt_e2e > 0 else 0,
        "ttft_error": abs(blis_ttft - gt_ttft) / gt_ttft if gt_ttft > 0 else 0,
        "itl_error": abs(blis_itl - gt_itl) / gt_itl if gt_itl > 0 else 0,
        "completed": metrics.get("completed_requests", 0),
    }


def run_34b_analysis(df, splits):
    """Deep-dive analysis of CodeLlama-34B anomaly."""
    print("\n" + "="*70)
    print("34B DEEP-DIVE ANALYSIS")
    print("="*70)

    analysis = {}
    model_tps = sorted(df["model_tp"].unique())

    for model_tp in model_tps:
        mask = df["model_tp"] == model_tp
        model_df = df[mask]

        test_mask = mask & df.index.isin(splits["test"])
        test_df = df[test_mask]

        durations = test_df[TARGET_COL].values
        prefills = test_df["batch.prefill_tokens"].values
        decodes = test_df["batch.decode_tokens"].values
        kv_sums = test_df["kv_sum"].values
        n_total = len(test_df)
        n_decode_only = int((prefills == 0).sum())
        n_mixed = n_total - n_decode_only

        analysis[model_tp] = {
            "n_steps": n_total,
            "n_decode_only": n_decode_only,
            "n_mixed": n_mixed,
            "pct_decode_only": 100 * n_decode_only / max(n_total, 1),
            "duration_mean": float(np.mean(durations)),
            "duration_p50": float(np.median(durations)),
            "duration_p99": float(np.percentile(durations, 99)),
            "kv_sum_mean": float(np.mean(kv_sums)),
            "kv_sum_max": float(np.max(kv_sums)),
            "prefill_mean": float(np.mean(prefills[prefills > 0])) if n_mixed > 0 else 0,
            "decode_mean": float(np.mean(decodes)),
        }

        print(f"\n  {model_tp}:")
        print(f"    Steps: {n_total} ({n_decode_only} decode-only, {n_mixed} mixed)")
        print(f"    Duration: mean={np.mean(durations):.0f}µs, p50={np.median(durations):.0f}µs, p99={np.percentile(durations, 99):.0f}µs")
        print(f"    KV sum: mean={np.mean(kv_sums):.0f}, max={np.max(kv_sums):.0f}")

    # Comparison table
    print("\n--- Model Comparison ---")
    print(f"{'Model':<30} {'Steps':<8} {'Decode%':<10} {'Dur µs':<10} {'KV sum':<10}")
    print("-" * 68)
    for mt, a in sorted(analysis.items()):
        print(f"  {mt:<28} {a['n_steps']:<8} {a['pct_decode_only']:<10.1f} {a['duration_mean']:<10.0f} {a['kv_sum_mean']:<10.0f}")

    return analysis


def main():
    print("Loading data...")
    df = extract_all_experiments_kv_features(DATA_ROOT)
    df["model_tp"] = df["model"] + "_tp" + df["tp"].astype(str)
    df["new_tokens"] = df["batch.prefill_tokens"].fillna(0) + df["batch.decode_tokens"].fillna(0)

    splits = temporal_split(df)

    # Train models
    print("\n--- Training 3-coeff + 4-coeff OLS models ---")
    trained = train_3coeff_models(df, splits)

    # 34B analysis
    analysis_34b = run_34b_analysis(df, splits)

    # Export artifacts and run BLIS
    print("\n" + "="*70)
    print("BLIS E2E VALIDATION — Trace Replay Mode")
    print("="*70)

    os.makedirs(H3_OUTPUT, exist_ok=True)
    artifact_dir = os.path.join(H3_OUTPUT, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # Map experiment dirs to model_tp
    dir_to_model_tp = {}
    for dirname in sorted(os.listdir(DATA_ROOT)):
        dirpath = os.path.join(DATA_ROOT, dirname)
        if not os.path.isdir(dirpath):
            continue
        try:
            meta = parse_experiment_metadata(dirname)
            model_tp = meta["model"] + "_tp" + str(meta["tp"])
            dir_to_model_tp[dirname] = model_tp
        except ValueError:
            continue

    # Export per-model artifacts with 4-coeff (prefill + decode + kv_sum)
    for model_tp, m in trained.items():
        art = export_stepml_artifact(model_tp, m["ols_4coeff"], m["overhead_us"], "4coeff")
        art_path = os.path.join(artifact_dir, f"{model_tp}_total_context.json")
        with open(art_path, "w") as f:
            json.dump(art, f, indent=2)
        print(f"  Exported: {art_path}")

    # Run BLIS for each experiment (trace replay)
    blis_results = []
    for dirname in sorted(os.listdir(DATA_ROOT)):
        dirpath = os.path.join(DATA_ROOT, dirname)
        if not os.path.isdir(dirpath):
            continue
        if dirname not in dir_to_model_tp:
            continue

        model_tp = dir_to_model_tp[dirname]
        if model_tp not in trained:
            print(f"  SKIP {dirname}: no trained model for {model_tp}")
            continue

        art_path = os.path.join(artifact_dir, f"{model_tp}_total_context.json")
        print(f"\n  Running: {dirname} (model_tp={model_tp})")

        result = run_blis_with_artifact(dirpath, art_path, use_trace_replay=True)
        blis_results.append(result)

        if result["status"] == "ok":
            print(f"    GT E2E={result['gt_e2e_ms']:.1f}ms, BLIS E2E={result['blis_e2e_ms']:.1f}ms, Error={result['e2e_error']*100:.1f}%")
            print(f"    GT ITL={result['gt_itl_ms']:.2f}ms, BLIS ITL={result['blis_itl_ms']:.2f}ms, ITL Error={result['itl_error']*100:.1f}%")
        else:
            print(f"    {result['status']}")

    # Also run workload-spec mode for comparison
    print("\n" + "="*70)
    print("BLIS E2E VALIDATION — Workload-Spec Mode (for comparison)")
    print("="*70)

    blis_results_spec = []
    for dirname in sorted(os.listdir(DATA_ROOT)):
        dirpath = os.path.join(DATA_ROOT, dirname)
        if not os.path.isdir(dirpath):
            continue
        if dirname not in dir_to_model_tp:
            continue

        model_tp = dir_to_model_tp[dirname]
        if model_tp not in trained:
            continue

        art_path = os.path.join(artifact_dir, f"{model_tp}_total_context.json")
        print(f"\n  Running: {dirname}")

        result = run_blis_with_artifact(dirpath, art_path, use_trace_replay=False)
        blis_results_spec.append(result)

        if result["status"] == "ok":
            print(f"    E2E Error={result['e2e_error']*100:.1f}%, ITL Error={result['itl_error']*100:.1f}%")
        else:
            print(f"    {result['status']}")

    # Summary
    print("\n" + "="*70)
    print("H3 SUMMARY")
    print("="*70)

    ok_trace = [r for r in blis_results if r.get("status") == "ok"]
    ok_spec = [r for r in blis_results_spec if r.get("status") == "ok"]

    if ok_trace:
        e2e_errors = [r["e2e_error"] for r in ok_trace]
        ttft_errors = [r["ttft_error"] for r in ok_trace]
        itl_errors = [r["itl_error"] for r in ok_trace]

        print(f"\n  TRACE REPLAY ({len(ok_trace)} experiments):")
        print(f"    E2E  mean error: {np.mean(e2e_errors)*100:.1f}%")
        print(f"    TTFT mean error: {np.mean(ttft_errors)*100:.1f}%")
        print(f"    ITL  mean error: {np.mean(itl_errors)*100:.1f}%")
        print(f"    E2E < 10%: {sum(1 for e in e2e_errors if e < 0.10)}/{len(ok_trace)}")
        print(f"    ITL < 10%: {sum(1 for e in itl_errors if e < 0.10)}/{len(ok_trace)}")

    if ok_spec:
        e2e_s = [r["e2e_error"] for r in ok_spec]
        itl_s = [r["itl_error"] for r in ok_spec]
        print(f"\n  WORKLOAD-SPEC ({len(ok_spec)} experiments):")
        print(f"    E2E  mean error: {np.mean(e2e_s)*100:.1f}%")
        print(f"    ITL  mean error: {np.mean(itl_s)*100:.1f}%")

    # Save results
    summary = {
        "trace_replay_results": blis_results,
        "workload_spec_results": blis_results_spec,
        "analysis_34b": analysis_34b,
    }
    if ok_trace:
        summary["trace_replay_summary"] = {
            "mean_e2e_error": float(np.mean([r["e2e_error"] for r in ok_trace])),
            "mean_ttft_error": float(np.mean([r["ttft_error"] for r in ok_trace])),
            "mean_itl_error": float(np.mean([r["itl_error"] for r in ok_trace])),
            "n_experiments": len(ok_trace),
        }
    if ok_spec:
        summary["workload_spec_summary"] = {
            "mean_e2e_error": float(np.mean([r["e2e_error"] for r in ok_spec])),
            "mean_itl_error": float(np.mean([r["itl_error"] for r in ok_spec])),
            "n_experiments": len(ok_spec),
        }

    with open(os.path.join(H3_OUTPUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {H3_OUTPUT}/summary.json")


if __name__ == "__main__":
    main()
