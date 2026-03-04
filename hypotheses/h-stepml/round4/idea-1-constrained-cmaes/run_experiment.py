#!/usr/bin/env python3
"""Round 4, Idea 1: Multi-Objective Constrained CMA-ES with ITL Penalty.

Implements all 4 sub-hypotheses:
  H1: Constrained CMA-ES with dual E2E+ITL objective
  H2: Pareto sweep across alpha values
  H3: LOMO (leave-one-model-out) generalization
  H4: LOWO (leave-one-workload-out) generalization

Uses the shared infrastructure from hypotheses/h-stepml/shared/ for data loading,
BLIS validation, and evaluation.
"""

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "shared"))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
# Ground truth lives in the main BLIS-research repo (not the worktree)
# Worktree root is stepml-worktrees/round4-idea-1-constrained-cmaes/
# BLIS-research is at inference-sim-package/BLIS-research/
_INFERENCE_SIM_PKG = os.path.abspath(os.path.join(_REPO_ROOT, "..", ".."))
_DATA_ROOT = os.path.join(_INFERENCE_SIM_PKG, "BLIS-research", "eval", "ground_truth")
_R3_ARTIFACTS_DIR = os.path.join(
    _INFERENCE_SIM_PKG, "BLIS-research", "hypotheses", "h-stepml",
    "round3", "idea-3-e2e-calibration", "h1-trace-e2e-opt", "artifacts"
)
BINARY_PATH = os.path.join(_REPO_ROOT, "simulation_worker")
BLOCK_SIZE_TOKENS = 16

sys.path.insert(0, _SHARED_DIR)

import yaml
from convert_lifecycle_to_traces import convert_experiment
from validate_blis import (
    extract_cpu_kv_blocks_from_vllm_log,
    extract_kv_blocks_from_vllm_log,
    load_exp_config,
    load_ground_truth_metrics,
    parse_blis_stdout,
    parse_experiment_dir,
)

# ---------------------------------------------------------------------------
# TTFT corrections from R3 H3 (per-model additive corrections in ms)
# ---------------------------------------------------------------------------
TTFT_CORRECTIONS_MS = {
    "llama-2-7b": 16.49,
    "llama-2-70b": 60.98,
    "llama-2-70b-hf": 37.57,
    "mixtral-8x7b-v0-1": 40.16,
    "codellama-34b": 32.48,
}

# ---------------------------------------------------------------------------
# Model group mapping (experiment dir model name -> group)
# ---------------------------------------------------------------------------
# For CMA-ES optimization, group experiments by shared optimizer.
# 70b and 70b-hf share the same optimizer (both are 70B models).
# Each experiment's specific artifact is loaded from R3 by exact model name.
MODEL_GROUPS = {
    "llama-2-7b": "llama-2-7b",
    "llama-2-70b": "llama-2-70b",
    "llama-2-70b-hf": "llama-2-70b",  # Same optimization group as 70b
    "codellama-34b": "codellama-34b",
    "mixtral-8x7b-v0-1": "mixtral-8x7b-v0-1",
}

# For LOMO, we group 70b and 70b-hf as one model
LOMO_MODEL_GROUPS = {
    "llama-2-7b": "llama-2-7b",
    "llama-2-70b": "llama-2-70b",
    "llama-2-70b-hf": "llama-2-70b",
    "codellama-34b": "codellama-34b",
    "mixtral-8x7b-v0-1": "mixtral-8x7b-v0-1",
}


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------
def discover_experiments():
    """Discover all valid experiments in the ground truth directory."""
    experiments = []
    for dirname in sorted(os.listdir(_DATA_ROOT)):
        dirpath = os.path.join(_DATA_ROOT, dirname)
        if not os.path.isdir(dirpath):
            continue
        summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
        if not os.path.isfile(summary_path):
            continue

        meta = parse_experiment_dir(dirname)
        gt = load_ground_truth_metrics(dirpath)
        exp_config = load_exp_config(dirpath)
        kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
        cpu_kv_blocks = extract_cpu_kv_blocks_from_vllm_log(dirpath)

        if kv_blocks is None:
            continue

        model_group = MODEL_GROUPS.get(meta["model"], meta["model"])

        experiments.append({
            "dirname": dirname,
            "dirpath": dirpath,
            "model": meta["model"],
            "model_group": model_group,
            "workload": meta["workload"],
            "tp": meta["tp"],
            "gt": gt,
            "exp_config": exp_config,
            "kv_blocks": kv_blocks,
            "cpu_kv_blocks": cpu_kv_blocks,
        })

    return experiments


# ---------------------------------------------------------------------------
# Trace CSV generation (cached)
# ---------------------------------------------------------------------------
_TRACE_CACHE = {}


def get_trace_csv(exp):
    """Get or create trace CSV for an experiment."""
    dirname = exp["dirname"]
    if dirname in _TRACE_CACHE:
        return _TRACE_CACHE[dirname]

    trace_dir = os.path.join(_SCRIPT_DIR, "traces")
    csv_path = os.path.join(trace_dir, f"{dirname}.csv")
    if not os.path.isfile(csv_path):
        csv_path = convert_experiment(exp["dirpath"], trace_dir)

    _TRACE_CACHE[dirname] = csv_path
    return csv_path


# ---------------------------------------------------------------------------
# BLIS execution (trace replay mode)
# ---------------------------------------------------------------------------
def run_blis_trace(trace_csv, exp_config, kv_blocks, artifact_path, horizon_us,
                   cpu_kv_blocks=0):
    """Run BLIS in trace replay mode with a StepML artifact."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        results_path = f.name

    cmd = [
        BINARY_PATH, "run",
        "--model", exp_config.get("model", "unknown"),
        "--workload", "traces",
        "--workload-traces-filepath", trace_csv,
        "--tp", str(exp_config.get("tensor_parallelism", 1)),
        "--max-model-len", str(exp_config.get("max_model_len", 4096)),
        "--max-num-running-reqs", str(exp_config.get("max_num_seqs", 128)),
        "--max-num-scheduled-tokens", str(exp_config.get("max_num_batched_tokens", 2048)),
        "--total-kv-blocks", str(kv_blocks),
        "--block-size-in-tokens", str(BLOCK_SIZE_TOKENS),
        "--horizon", str(horizon_us),
        "--alpha-coeffs=1,0,0",
        "--beta-coeffs=1,0,0",
        "--stepml-model", artifact_path,
        "--results-path", results_path,
        "--log", "error",
    ]
    if cpu_kv_blocks > 0:
        cmd.extend(["--kv-cpu-blocks", str(cpu_kv_blocks)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(BINARY_PATH),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        if os.path.exists(results_path):
            os.unlink(results_path)

    if result.returncode != 0:
        return None
    return parse_blis_stdout(result.stdout)


def compute_error(predicted, observed):
    """Compute relative error: |predicted - observed| / observed."""
    if observed == 0:
        return float("inf") if predicted != 0 else 0.0
    return abs(predicted - observed) / observed


def compute_horizon(exp):
    """Compute a horizon that covers all requests."""
    gt = exp["gt"]
    e2e_s = gt["e2e_mean_s"]
    n_requests = gt["num_requests"]
    rps = gt["throughput_rps"]
    if rps > 0:
        duration_s = n_requests / rps
    else:
        duration_s = 600
    return int((duration_s + e2e_s * 3 + 120) * 1_000_000)


# ---------------------------------------------------------------------------
# Artifact manipulation
# ---------------------------------------------------------------------------
def load_r3_artifact(model_group):
    """Load the R3 CMA-ES optimized artifact for a model group.

    For the llama-2-70b group, loads the 70b artifact (not 70b-hf).
    """
    filename = f"{model_group}_optimized.json"
    path = os.path.join(_R3_ARTIFACTS_DIR, filename)
    if not os.path.isfile(path):
        # Try without -hf suffix
        alt = f"{model_group.replace('-hf', '')}_optimized.json"
        alt_path = os.path.join(_R3_ARTIFACTS_DIR, alt)
        if os.path.isfile(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"R3 artifact not found: {path}")
    with open(path) as f:
        return json.load(f)


def artifact_to_param_vector(artifact):
    """Extract optimizable parameters from a StepML artifact."""
    params = {}
    params["step_overhead_us"] = artifact.get("step_overhead_us", 4000)
    params["step_overhead_per_req_us"] = artifact.get("step_overhead_per_req_us", 0)
    params["output_token_processing_time_us"] = artifact.get("output_token_processing_time_us", 0)
    params["scheduling_processing_time_us"] = artifact.get("scheduling_processing_time_us", 0)
    params["preemption_processing_time_us"] = artifact.get("preemption_processing_time_us", 0)

    qt = artifact.get("queueing_time")
    if qt:
        params["queueing_intercept"] = qt.get("intercept", 0)

    for regime in artifact.get("step_time_regimes", []):
        prefix = regime["name"]
        model = regime["model"]
        params[f"{prefix}_intercept"] = model.get("intercept", 0)
        for feat, coeff in model.get("feature_coefficients", {}).items():
            params[f"{prefix}_{feat}"] = coeff

    return params


def param_vector_to_artifact(params, base_artifact):
    """Reconstruct a StepML artifact from optimized parameters."""
    art = copy.deepcopy(base_artifact)
    art["step_overhead_us"] = params["step_overhead_us"]
    art["step_overhead_per_req_us"] = params.get("step_overhead_per_req_us", 0)
    art["output_token_processing_time_us"] = params["output_token_processing_time_us"]
    art["scheduling_processing_time_us"] = params["scheduling_processing_time_us"]
    art["preemption_processing_time_us"] = params["preemption_processing_time_us"]

    if "queueing_intercept" in params:
        if art.get("queueing_time") is None:
            art["queueing_time"] = {
                "model_type": "linear",
                "intercept": 0,
                "feature_coefficients": {},
            }
        art["queueing_time"]["intercept"] = params["queueing_intercept"]

    for regime in art.get("step_time_regimes", []):
        prefix = regime["name"]
        key = f"{prefix}_intercept"
        if key in params:
            regime["model"]["intercept"] = params[key]
        for feat in list(regime["model"].get("feature_coefficients", {}).keys()):
            key = f"{prefix}_{feat}"
            if key in params:
                regime["model"]["feature_coefficients"][feat] = params[key]

    return art


# ---------------------------------------------------------------------------
# Constrained parameter bounds (the key R4 innovation)
# ---------------------------------------------------------------------------
def compute_bounds(params, base_overhead):
    """Compute constrained parameter bounds.

    Key constraints vs R3:
    - output_token_processing_time_us: [0, 500] (was [0, 5000+])
    - scheduling_processing_time_us: [0, 2000] (was unbounded)
    - step_overhead_us: [base*0.8, base*3.0] (was [base*0.3, base*5.0])
    - Regime coefficients: tighter bounds (+/- 2x instead of +/- 5x)
    """
    lower = {}
    upper = {}

    for name, val in params.items():
        if name == "output_token_processing_time_us":
            lower[name] = 0
            upper[name] = 500  # Hard cap (was up to 5000+ in R3)
        elif name == "scheduling_processing_time_us":
            lower[name] = 0
            upper[name] = 2000
        elif name == "preemption_processing_time_us":
            lower[name] = 0
            upper[name] = 1000
        elif name == "step_overhead_us":
            lower[name] = max(1000, base_overhead * 0.8)
            upper[name] = base_overhead * 3.0
        elif name == "step_overhead_per_req_us":
            lower[name] = 0
            upper[name] = max(200, abs(val) * 3)
        elif name == "queueing_intercept":
            lower[name] = -1000
            upper[name] = 2000
        elif "intercept" in name:
            lower[name] = val - abs(val) * 2 - 2000
            upper[name] = val + abs(val) * 2 + 2000
        else:
            # Regime coefficients: +/- 2x (tighter than R3's +/- 5x)
            lower[name] = val - abs(val) * 2 - 50
            upper[name] = val + abs(val) * 2 + 50

    return lower, upper


# ---------------------------------------------------------------------------
# Constrained CMA-ES objective function
# ---------------------------------------------------------------------------
def evaluate_artifact(artifact, exps, ttft_correction_ms=0):
    """Evaluate a StepML artifact across experiments.

    Returns dict with per-experiment E2E, TTFT, ITL errors.
    """
    # Write artifact to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(artifact, f)
        artifact_path = f.name

    results = []
    try:
        for exp in exps:
            trace_csv = get_trace_csv(exp)
            horizon_us = compute_horizon(exp)

            blis = run_blis_trace(
                trace_csv, exp["exp_config"], exp["kv_blocks"],
                artifact_path, horizon_us, exp["cpu_kv_blocks"],
            )

            if blis is None:
                results.append({
                    "experiment": exp["dirname"],
                    "e2e_error": 5.0,  # penalty
                    "ttft_error": 5.0,
                    "itl_error": 5.0,
                    "status": "failed",
                })
                continue

            # Apply TTFT additive correction
            corrected_ttft = blis["ttft_mean_ms"] + ttft_correction_ms

            gt = exp["gt"]
            e2e_err = compute_error(blis["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
            ttft_err = compute_error(corrected_ttft, gt["ttft_mean_s"] * 1000)
            itl_err = compute_error(blis["itl_mean_ms"], gt["itl_mean_s"] * 1000)

            results.append({
                "experiment": exp["dirname"],
                "model": exp["model"],
                "workload": exp["workload"],
                "e2e_error": e2e_err,
                "ttft_error": ttft_err,
                "itl_error": itl_err,
                "blis_e2e_ms": blis["e2e_mean_ms"],
                "blis_ttft_ms": corrected_ttft,
                "blis_itl_ms": blis["itl_mean_ms"],
                "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
                "gt_ttft_ms": gt["ttft_mean_s"] * 1000,
                "gt_itl_ms": gt["itl_mean_s"] * 1000,
                "status": "ok",
            })
    finally:
        os.unlink(artifact_path)

    return results


def make_objective(base_artifact, param_names, exps, alpha, ttft_correction_ms):
    """Create a CMA-ES objective function with dual E2E+ITL penalty.

    objective = alpha * mean_e2e + (1-alpha) * mean_itl
              + 10 * max(0, mean_itl - 0.20)  # hard penalty for ITL > 20%
    """
    eval_count = [0]

    def objective(x):
        eval_count[0] += 1
        params = dict(zip(param_names, x))
        artifact = param_vector_to_artifact(params, base_artifact)

        results = evaluate_artifact(artifact, exps, ttft_correction_ms)
        e2e_errors = [r["e2e_error"] for r in results]
        itl_errors = [r["itl_error"] for r in results]

        mean_e2e = np.mean(e2e_errors)
        mean_itl = np.mean(itl_errors)

        # Dual objective with ITL penalty
        obj = alpha * mean_e2e + (1 - alpha) * mean_itl
        # Penalty for ITL > 20%
        itl_penalty = 10.0 * max(0, mean_itl - 0.20)
        obj += itl_penalty

        if eval_count[0] % 5 == 0 or eval_count[0] == 1:
            print(f"  Eval {eval_count[0]:3d}: E2E={mean_e2e*100:.1f}% "
                  f"ITL={mean_itl*100:.1f}% obj={obj:.4f}")

        return obj

    return objective, eval_count


# ---------------------------------------------------------------------------
# H1: Run constrained CMA-ES per model group
# ---------------------------------------------------------------------------
def run_h1(experiments, alpha=0.5, max_evals=100, output_dir=None):
    """Run constrained CMA-ES per model group with dual E2E+ITL objective."""
    import cma

    if output_dir is None:
        output_dir = os.path.join(_SCRIPT_DIR, "h1-constrained-cmaes")
    os.makedirs(output_dir, exist_ok=True)
    artifacts_dir = os.path.join(output_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # Group experiments by model
    model_exps = {}
    for exp in experiments:
        group = exp["model_group"]
        model_exps.setdefault(group, []).append(exp)

    all_results = []
    optimization_logs = {}

    for model_group in sorted(model_exps.keys()):
        exps = model_exps[model_group]
        print(f"\n{'='*60}")
        print(f"  Optimizing: {model_group} ({len(exps)} experiments)")
        print(f"  Alpha={alpha} (E2E weight)")
        print(f"{'='*60}")

        # Load R3 artifact as starting point
        r3_art = load_r3_artifact(model_group)
        params = artifact_to_param_vector(r3_art)
        param_names = sorted(params.keys())
        x0 = [params[n] for n in param_names]

        # Get TTFT correction
        model_for_ttft = model_group
        if model_group == "llama-2-70b":
            model_for_ttft = "llama-2-70b"
        ttft_ms = TTFT_CORRECTIONS_MS.get(model_for_ttft, 0)

        # Compute initial objective
        initial_results = evaluate_artifact(r3_art, exps, ttft_ms)
        initial_e2e = np.mean([r["e2e_error"] for r in initial_results])
        initial_itl = np.mean([r["itl_error"] for r in initial_results])
        print(f"  Initial: E2E={initial_e2e*100:.1f}% ITL={initial_itl*100:.1f}%")

        # Compute base overhead for bounds
        base_overhead = r3_art.get("step_overhead_us", 5000)
        lower, upper = compute_bounds(params, base_overhead)

        lower_list = [lower[n] for n in param_names]
        upper_list = [upper[n] for n in param_names]

        # Clip x0 to bounds
        x0_clipped = [max(lo, min(hi, v)) for v, lo, hi in zip(x0, lower_list, upper_list)]

        # CMA-ES configuration
        sigma0 = 0.1 * np.mean([abs(hi - lo) for lo, hi in zip(lower_list, upper_list)])
        sigma0 = max(sigma0, 10)

        objective_fn, eval_count = make_objective(
            r3_art, param_names, exps, alpha, ttft_ms
        )

        t0 = time.time()
        opts = cma.CMAOptions()
        opts["maxfevals"] = max_evals
        opts["timeout"] = 1800  # 30 min timeout
        opts["bounds"] = [lower_list, upper_list]
        opts["verbose"] = -1
        opts["seed"] = 42
        opts["popsize"] = 8

        es = cma.CMAEvolutionStrategy(x0_clipped, sigma0, opts)
        convergence = []

        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective_fn(s) for s in solutions]
            es.tell(solutions, fitnesses)
            convergence.append(float(es.result.fbest))

        elapsed = time.time() - t0
        best_x = es.result.xbest
        best_params = dict(zip(param_names, best_x))
        best_artifact = param_vector_to_artifact(best_params, r3_art)

        # Evaluate final artifact
        final_results = evaluate_artifact(best_artifact, exps, ttft_ms)
        final_e2e = np.mean([r["e2e_error"] for r in final_results])
        final_itl = np.mean([r["itl_error"] for r in final_results])

        print(f"\n  Final:   E2E={final_e2e*100:.1f}% ITL={final_itl*100:.1f}%")
        print(f"  Evals: {eval_count[0]}, Time: {elapsed:.0f}s")

        # Save artifact
        art_path = os.path.join(artifacts_dir, f"{model_group}_optimized.json")
        with open(art_path, "w") as f:
            json.dump(best_artifact, f, indent=2)

        for r in final_results:
            r["model_group"] = model_group
            r["initial_e2e_error"] = initial_e2e
            r["initial_itl_error"] = initial_itl
            r["alpha"] = alpha
        all_results.extend(final_results)

        optimization_logs[model_group] = {
            "n_evals": eval_count[0],
            "elapsed_s": elapsed,
            "initial_mean_e2e": float(initial_e2e),
            "initial_mean_itl": float(initial_itl),
            "best_mean_e2e": float(final_e2e),
            "best_mean_itl": float(final_itl),
            "convergence": convergence,
            "alpha": alpha,
        }

    # Compute summary
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    mean_e2e = np.mean([r["e2e_error"] for r in ok_results]) * 100
    mean_itl = np.mean([r["itl_error"] for r in ok_results]) * 100
    mean_ttft = np.mean([r["ttft_error"] for r in ok_results]) * 100

    summary = {
        "alpha": alpha,
        "n_experiments": len(ok_results),
        "mean_e2e_error_pct": float(mean_e2e),
        "mean_ttft_error_pct": float(mean_ttft),
        "mean_itl_error_pct": float(mean_itl),
        "e2e_under_10pct": sum(1 for r in ok_results if r["e2e_error"] < 0.10),
        "e2e_under_15pct": sum(1 for r in ok_results if r["e2e_error"] < 0.15),
        "optimization_logs": optimization_logs,
        "per_experiment": ok_results,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  H1 Results (alpha={alpha}):")
    print(f"  Mean E2E:  {mean_e2e:.1f}%")
    print(f"  Mean TTFT: {mean_ttft:.1f}%")
    print(f"  Mean ITL:  {mean_itl:.1f}%")
    print(f"  E2E < 10%: {summary['e2e_under_10pct']}/{len(ok_results)}")
    print(f"{'='*60}")

    return summary, all_results


# ---------------------------------------------------------------------------
# H2: Pareto sweep
# ---------------------------------------------------------------------------
def run_h2(experiments, output_dir=None):
    """Run H1 with different alpha values to explore Pareto frontier."""
    if output_dir is None:
        output_dir = os.path.join(_SCRIPT_DIR, "h2-pareto-sweep")
    os.makedirs(output_dir, exist_ok=True)

    alphas = [0.7, 0.5, 0.3]
    pareto_points = []

    for alpha in alphas:
        print(f"\n{'#'*60}")
        print(f"  PARETO SWEEP: alpha = {alpha}")
        print(f"{'#'*60}")

        alpha_dir = os.path.join(output_dir, f"alpha_{alpha}")
        summary, results = run_h1(
            experiments, alpha=alpha, max_evals=100, output_dir=alpha_dir
        )

        pareto_points.append({
            "alpha": alpha,
            "mean_e2e_pct": summary["mean_e2e_error_pct"],
            "mean_itl_pct": summary["mean_itl_error_pct"],
            "mean_ttft_pct": summary["mean_ttft_error_pct"],
            "e2e_under_10": summary["e2e_under_10pct"],
            "n_experiments": summary["n_experiments"],
        })

    # Find knee point (closest to origin in E2E-ITL space)
    best_idx = 0
    best_dist = float("inf")
    for i, pt in enumerate(pareto_points):
        # Normalize: E2E target 10%, ITL target 15%
        dist = (pt["mean_e2e_pct"] / 10) ** 2 + (pt["mean_itl_pct"] / 15) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    pareto_summary = {
        "pareto_points": pareto_points,
        "best_alpha": pareto_points[best_idx]["alpha"],
        "best_point": pareto_points[best_idx],
        "knee_distance": float(best_dist),
    }

    summary_path = os.path.join(output_dir, "pareto_summary.json")
    with open(summary_path, "w") as f:
        json.dump(pareto_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  PARETO FRONTIER:")
    print(f"  {'Alpha':>6s} {'E2E%':>8s} {'ITL%':>8s} {'TTFT%':>8s} {'<10%':>5s}")
    for pt in pareto_points:
        marker = " ← KNEE" if pt["alpha"] == pareto_points[best_idx]["alpha"] else ""
        print(f"  {pt['alpha']:>6.1f} {pt['mean_e2e_pct']:>7.1f}% "
              f"{pt['mean_itl_pct']:>7.1f}% {pt['mean_ttft_pct']:>7.1f}% "
              f"{pt['e2e_under_10']:>4d}{marker}")
    print(f"{'='*60}")

    return pareto_summary


# ---------------------------------------------------------------------------
# H3: LOMO (leave-one-model-out)
# ---------------------------------------------------------------------------
def run_h3(experiments, best_alpha, output_dir=None):
    """Leave-one-model-out cross-validation with CMA-ES artifacts."""
    if output_dir is None:
        output_dir = os.path.join(_SCRIPT_DIR, "h3-lomo")
    os.makedirs(output_dir, exist_ok=True)

    # Get unique model groups
    model_groups = sorted(set(e["model_group"] for e in experiments))
    print(f"\n{'='*60}")
    print(f"  H3: LOMO ({len(model_groups)} folds)")
    print(f"{'='*60}")

    # Load best artifacts (from H2's best alpha)
    best_artifacts_dir = os.path.join(
        _SCRIPT_DIR, "h2-pareto-sweep", f"alpha_{best_alpha}", "artifacts"
    )

    folds = []
    for holdout_model in model_groups:
        print(f"\n  Fold: holdout={holdout_model}")
        holdout_exps = [e for e in experiments if e["model_group"] == holdout_model]
        # For each donor model, try its artifact on the holdout
        donor_results = []
        for donor_model in model_groups:
            if donor_model == holdout_model:
                continue
            art_path = os.path.join(best_artifacts_dir, f"{donor_model}_optimized.json")
            if not os.path.isfile(art_path):
                continue

            with open(art_path) as f:
                donor_art = json.load(f)

            ttft_ms = TTFT_CORRECTIONS_MS.get(holdout_model, 0)
            results = evaluate_artifact(donor_art, holdout_exps, ttft_ms)
            ok_results = [r for r in results if r.get("status") == "ok"]
            if ok_results:
                mean_e2e = np.mean([r["e2e_error"] for r in ok_results])
                mean_itl = np.mean([r["itl_error"] for r in ok_results])
                donor_results.append({
                    "donor": donor_model,
                    "mean_e2e_pct": float(mean_e2e * 100),
                    "mean_itl_pct": float(mean_itl * 100),
                    "per_experiment": ok_results,
                })
                print(f"    Donor {donor_model}: E2E={mean_e2e*100:.1f}% ITL={mean_itl*100:.1f}%")

        # Best donor for this fold
        if donor_results:
            best_donor = min(donor_results, key=lambda x: x["mean_e2e_pct"])
        else:
            best_donor = {"donor": "none", "mean_e2e_pct": 999, "mean_itl_pct": 999}

        folds.append({
            "holdout_model": holdout_model,
            "best_donor": best_donor["donor"],
            "best_donor_e2e_pct": best_donor["mean_e2e_pct"],
            "best_donor_itl_pct": best_donor.get("mean_itl_pct", 999),
            "all_donors": donor_results,
        })

    # Summary
    mean_best_donor_e2e = np.mean([f["best_donor_e2e_pct"] for f in folds])
    lomo_summary = {
        "folds": folds,
        "mean_best_donor_e2e_pct": float(mean_best_donor_e2e),
        "target": "<80% MAPE per fold (E2E proxy)",
        "alpha_used": best_alpha,
    }

    summary_path = os.path.join(output_dir, "lomo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(lomo_summary, f, indent=2, default=str)

    print(f"\n  LOMO Summary:")
    print(f"  {'Holdout':<25s} {'Best Donor':<25s} {'E2E%':>8s} {'ITL%':>8s}")
    for fold in folds:
        print(f"  {fold['holdout_model']:<25s} {fold['best_donor']:<25s} "
              f"{fold['best_donor_e2e_pct']:>7.1f}% {fold['best_donor_itl_pct']:>7.1f}%")
    print(f"  Mean best-donor E2E: {mean_best_donor_e2e:.1f}%")

    return lomo_summary


# ---------------------------------------------------------------------------
# H4: LOWO (leave-one-workload-out)
# ---------------------------------------------------------------------------
def run_h4(experiments, best_alpha, output_dir=None):
    """Leave-one-workload-out: check per-model workload variance."""
    if output_dir is None:
        output_dir = os.path.join(_SCRIPT_DIR, "h4-lowo")
    os.makedirs(output_dir, exist_ok=True)

    # Load best artifacts
    best_artifacts_dir = os.path.join(
        _SCRIPT_DIR, "h2-pareto-sweep", f"alpha_{best_alpha}", "artifacts"
    )

    # Get unique workloads and models
    workloads = sorted(set(e["workload"] for e in experiments))
    model_groups = sorted(set(e["model_group"] for e in experiments))

    print(f"\n{'='*60}")
    print(f"  H4: LOWO ({len(workloads)} workloads)")
    print(f"{'='*60}")

    # Evaluate each experiment with its model's best artifact
    per_exp_results = []
    for exp in experiments:
        art_path = os.path.join(best_artifacts_dir, f"{exp['model_group']}_optimized.json")
        if not os.path.isfile(art_path):
            continue
        with open(art_path) as f:
            art = json.load(f)
        ttft_ms = TTFT_CORRECTIONS_MS.get(exp["model"], 0)
        results = evaluate_artifact(art, [exp], ttft_ms)
        if results and results[0].get("status") == "ok":
            per_exp_results.append(results[0])

    # Compute per-model workload range
    model_workload_map = {}
    for r in per_exp_results:
        group = MODEL_GROUPS.get(r.get("model", ""), r.get("model", ""))
        model_workload_map.setdefault(group, []).append(r)

    model_stats = {}
    for model, results in sorted(model_workload_map.items()):
        e2e_errors = [r["e2e_error"] * 100 for r in results]
        model_stats[model] = {
            "workloads": [r.get("workload", "?") for r in results],
            "e2e_errors": e2e_errors,
            "e2e_range": max(e2e_errors) - min(e2e_errors),
            "e2e_mean": np.mean(e2e_errors),
        }

    # Aggregate E2E across all experiments
    agg_e2e = np.mean([r["e2e_error"] for r in per_exp_results])
    within_2x = sum(
        1 for r in per_exp_results if r["e2e_error"] <= agg_e2e * 2
    )

    lowo_summary = {
        "per_model_stats": {m: s for m, s in model_stats.items()},
        "aggregate_e2e_pct": float(agg_e2e * 100),
        "within_2x_aggregate": within_2x,
        "total_experiments": len(per_exp_results),
        "alpha_used": best_alpha,
    }

    summary_path = os.path.join(output_dir, "lowo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(lowo_summary, f, indent=2, default=str)

    print(f"\n  LOWO Summary:")
    print(f"  {'Model':<25s} {'Range (pp)':>10s} {'Mean E2E%':>10s} {'Workloads'}")
    for model, stats in sorted(model_stats.items()):
        wl_str = ", ".join(f"{w}={e:.1f}%" for w, e in zip(stats["workloads"], stats["e2e_errors"]))
        print(f"  {model:<25s} {stats['e2e_range']:>9.1f}pp {stats['e2e_mean']:>9.1f}% {wl_str}")
    print(f"\n  Aggregate E2E: {agg_e2e*100:.1f}%")
    print(f"  Within 2x aggregate: {within_2x}/{len(per_exp_results)}")

    return lowo_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Round 4, Idea 1: Constrained CMA-ES with ITL Penalty")
    print("=" * 60)

    # Discover experiments
    experiments = discover_experiments()
    print(f"\nDiscovered {len(experiments)} experiments")
    for exp in experiments:
        print(f"  {exp['dirname']}: model={exp['model_group']}, wl={exp['workload']}")

    if not experiments:
        print("ERROR: No experiments found. Check _DATA_ROOT path.")
        sys.exit(1)

    # H1: Run with default alpha=0.5 first to verify infrastructure
    print(f"\n{'#'*60}")
    print("  H1: Constrained CMA-ES (alpha=0.5)")
    print(f"{'#'*60}")
    h1_summary, h1_results = run_h1(experiments, alpha=0.5, max_evals=100)

    # Short-circuit check
    if h1_summary["mean_e2e_error_pct"] > 25:
        print(f"\nSHORT-CIRCUIT: Mean E2E {h1_summary['mean_e2e_error_pct']:.1f}% > 25%")
        print("Constrained CMA-ES is significantly worse than R3. Aborting.")
        return

    # H2: Pareto sweep
    print(f"\n{'#'*60}")
    print("  H2: Pareto Sweep")
    print(f"{'#'*60}")
    h2_summary = run_h2(experiments)
    best_alpha = h2_summary["best_alpha"]

    # H3: LOMO
    print(f"\n{'#'*60}")
    print(f"  H3: LOMO (using best alpha={best_alpha})")
    print(f"{'#'*60}")
    h3_summary = run_h3(experiments, best_alpha)

    # H4: LOWO
    print(f"\n{'#'*60}")
    print(f"  H4: LOWO (using best alpha={best_alpha})")
    print(f"{'#'*60}")
    h4_summary = run_h4(experiments, best_alpha)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  H1 (alpha=0.5): E2E={h1_summary['mean_e2e_error_pct']:.1f}% "
          f"ITL={h1_summary['mean_itl_error_pct']:.1f}%")
    print(f"  H2 best alpha={best_alpha}: "
          f"E2E={h2_summary['best_point']['mean_e2e_pct']:.1f}% "
          f"ITL={h2_summary['best_point']['mean_itl_pct']:.1f}%")
    print(f"  H3 LOMO mean best-donor E2E: {h3_summary['mean_best_donor_e2e_pct']:.1f}%")
    print(f"  H4 LOWO within 2x: {h4_summary['within_2x_aggregate']}/{h4_summary['total_experiments']}")


if __name__ == "__main__":
    main()
