#!/usr/bin/env python3
"""Round 3, Idea 3: End-to-End Calibration via Direct E2E Objective.

Runs all three sub-hypotheses:
  H1: CMA-ES optimization of StepML parameters against BLIS E2E error (trace replay)
  H2: Same optimization in workload-spec mode
  H3: Additive correction factors for TTFT/ITL residuals

Uses Round 2's best StepML artifacts as starting point (regime ensemble + overheads).
CMA-ES treats BLIS as a black-box function: params → E2E error.
"""

import copy
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

import cma
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
PACKAGE_ROOT = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package"
BLIS_RESEARCH_ROOT = os.path.join(PACKAGE_ROOT, "BLIS-research")
GT_DATA_ROOT = os.path.join(BLIS_RESEARCH_ROOT, "eval", "ground_truth")
ORIG_GT_ROOT = os.path.join(PACKAGE_ROOT, "inference-sim", "eval", "ground_truth")
R2_ARTIFACTS_DIR = os.path.join(
    BLIS_RESEARCH_ROOT,
    "hypotheses", "h-stepml", "round2",
    "idea-2-regime-ensemble", "h3-secondary-method-calibration",
    "output", "calibrated_artifacts",
)
BINARY_PATH = os.path.join(REPO_ROOT, "simulation_worker")
BLOCK_SIZE_TOKENS = 16

# Model → artifact file mapping
MODEL_ARTIFACT_MAP = {
    "llama-2-7b": "llama-2-7b_tp1_regime.json",
    "llama-2-70b": "llama-2-70b_tp4_regime.json",
    "llama-2-70b-hf": "llama-2-70b-hf_tp4_regime.json",
    "codellama-34b": "codellama-34b_tp2_regime.json",
    "mixtral-8x7b-v0-1": "mixtral-8x7b-v0-1_tp2_regime.json",
}

# Models that share the same artifact (grouped for optimization)
MODEL_GROUPS = {
    "llama-2-7b": ["llama-2-7b"],
    "llama-2-70b": ["llama-2-70b", "llama-2-70b-hf"],
    "codellama-34b": ["codellama-34b"],
    "mixtral-8x7b-v0-1": ["mixtral-8x7b-v0-1"],
}


# ---------------------------------------------------------------------------
# Helpers (reused from Idea 1)
# ---------------------------------------------------------------------------
def parse_experiment_dir(dirname):
    import re
    tp_matches = list(re.finditer(r"-tp(\d+)-", dirname))
    if not tp_matches:
        raise ValueError(f"Cannot parse: {dirname}")
    last_tp = tp_matches[-1]
    if re.match(r"^\d{8}-\d{6}-", dirname):
        model = dirname[16:last_tp.start()]
    elif re.match(r"^\d{8}-", dirname):
        model = dirname[9:last_tp.start()]
    else:
        model = dirname[:last_tp.start()]
    return {
        "model": model,
        "tp": int(last_tp.group(1)),
        "workload": dirname[last_tp.end():],
    }


def load_ground_truth(experiment_dir):
    summary_path = os.path.join(
        experiment_dir, "results", "summary_lifecycle_metrics.json"
    )
    with open(summary_path) as f:
        data = json.load(f)
    successes = data.get("successes", {})
    latency = successes.get("latency", {})
    return {
        "e2e_mean_s": latency.get("request_latency", {}).get("mean", 0),
        "ttft_mean_s": latency.get("time_to_first_token", {}).get("mean", 0),
        "itl_mean_s": latency.get("inter_token_latency", {}).get("mean", 0),
        "num_requests": data.get("load_summary", {}).get("count", 0),
        "prompt_len_mean": successes.get("prompt_len", {}).get("mean", 0),
        "output_len_mean": successes.get("output_len", {}).get("mean", 0),
        "throughput_rps": successes.get("throughput", {}).get("requests_per_sec", 0),
    }


def load_exp_config(experiment_dir):
    with open(os.path.join(experiment_dir, "exp-config.yaml")) as f:
        return yaml.safe_load(f)


def extract_kv_blocks(experiment_dir):
    import re
    vllm_log = os.path.join(experiment_dir, "vllm.log")
    if not os.path.isfile(vllm_log):
        return None
    pattern = re.compile(r"GPU KV cache size:\s+([\d,]+)\s+tokens")
    with open(vllm_log) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return int(m.group(1).replace(",", "")) // BLOCK_SIZE_TOKENS
    return None


def extract_cpu_kv_blocks(experiment_dir):
    import re
    vllm_log = os.path.join(experiment_dir, "vllm.log")
    if not os.path.isfile(vllm_log):
        return 0
    cpu_bytes = None
    kv_shape = None
    with open(vllm_log) as f:
        for line in f:
            m = re.search(r"cpu_bytes_to_use['\"]?:\s*([\d.]+)", line)
            if m and cpu_bytes is None:
                cpu_bytes = float(m.group(1))
            m = re.search(
                r"cross layer KV cache of shape \((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
                line,
            )
            if m and kv_shape is None:
                kv_shape = tuple(int(m.group(i)) for i in range(1, 7))
    if cpu_bytes is None or kv_shape is None or cpu_bytes == 0:
        return 0
    _, num_layers, _, block_size, num_kv_heads, head_dim = kv_shape
    per_block_bytes = num_layers * 2 * block_size * num_kv_heads * head_dim * 2
    return int(cpu_bytes) // per_block_bytes


def convert_lifecycle_to_csv(lifecycle_path, output_csv, model_name, seed=42):
    with open(lifecycle_path) as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"Empty lifecycle data: {lifecycle_path}")
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
    return {
        "num_requests": len(rows),
        "duration_s": (rows[-1][0] - rows[0][0]) if len(rows) > 1 else 0,
    }


def compute_error(predicted, observed):
    if observed == 0:
        return float("inf") if predicted != 0 else 0.0
    return abs(predicted - observed) / observed


def parse_blis_stdout(stdout):
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
        metrics = json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None
    return {
        "e2e_mean_ms": metrics.get("e2e_mean_ms", 0),
        "ttft_mean_ms": metrics.get("ttft_mean_ms", 0),
        "itl_mean_ms": metrics.get("itl_mean_ms", 0),
        "completed_requests": metrics.get("completed_requests", 0),
    }


# ---------------------------------------------------------------------------
# Artifact manipulation
# ---------------------------------------------------------------------------
def load_artifact(model_name):
    """Load the Round 2 best artifact for a given model."""
    artifact_file = MODEL_ARTIFACT_MAP.get(model_name)
    if artifact_file is None:
        raise ValueError(f"No artifact for model: {model_name}")
    path = os.path.join(R2_ARTIFACTS_DIR, artifact_file)
    with open(path) as f:
        return json.load(f)


def artifact_to_param_vector(artifact):
    """Extract optimizable parameters from a StepML artifact.

    Parameter vector layout:
      [step_overhead_us, step_overhead_per_req_us,
       scheduling_processing_time_us, preemption_processing_time_us,
       output_token_processing_time_us,
       queueing_intercept,
       decode_only_intercept, decode_only_decode_tokens, decode_only_num_decode_reqs,
       mixed_intercept, mixed_prefill_tokens, mixed_decode_tokens,
       mixed_num_prefill_reqs, mixed_num_decode_reqs]

    Returns (param_vector, param_names) tuple.
    """
    params = []
    names = []

    # Global overhead parameters
    params.append(artifact.get("step_overhead_us", 0))
    names.append("step_overhead_us")

    params.append(artifact.get("step_overhead_per_req_us", 0))
    names.append("step_overhead_per_req_us")

    params.append(artifact.get("scheduling_processing_time_us", 0))
    names.append("scheduling_processing_time_us")

    params.append(artifact.get("preemption_processing_time_us", 0))
    names.append("preemption_processing_time_us")

    params.append(artifact.get("output_token_processing_time_us", 0))
    names.append("output_token_processing_time_us")

    # Queueing time intercept
    qt = artifact.get("queueing_time")
    if qt:
        params.append(qt.get("intercept", 0))
    else:
        params.append(0)
    names.append("queueing_intercept")

    # Step time regime parameters
    regimes = artifact.get("step_time_regimes", [])
    for regime in regimes:
        prefix = regime["name"]
        model_data = regime["model"]
        params.append(model_data.get("intercept", 0))
        names.append(f"{prefix}_intercept")

        # Extract feature coefficients in sorted order for reproducibility
        coeffs = model_data.get("feature_coefficients", {})
        for feat in sorted(coeffs.keys()):
            params.append(coeffs[feat])
            names.append(f"{prefix}_{feat}")

    return np.array(params, dtype=np.float64), names


def param_vector_to_artifact(params, names, base_artifact):
    """Reconstruct a StepML artifact from an optimized parameter vector."""
    artifact = copy.deepcopy(base_artifact)
    param_dict = dict(zip(names, params))

    artifact["step_overhead_us"] = max(0, param_dict.get("step_overhead_us", 0))
    artifact["step_overhead_per_req_us"] = max(0, param_dict.get("step_overhead_per_req_us", 0))
    artifact["scheduling_processing_time_us"] = max(0, param_dict.get("scheduling_processing_time_us", 0))
    artifact["preemption_processing_time_us"] = max(0, param_dict.get("preemption_processing_time_us", 0))
    artifact["output_token_processing_time_us"] = max(0, param_dict.get("output_token_processing_time_us", 0))

    # Queueing time
    qt_intercept = param_dict.get("queueing_intercept", 0)
    if artifact.get("queueing_time"):
        artifact["queueing_time"]["intercept"] = qt_intercept
    else:
        artifact["queueing_time"] = {
            "model_type": "linear",
            "intercept": qt_intercept,
            "feature_coefficients": {},
        }

    # Regime coefficients
    for regime in artifact.get("step_time_regimes", []):
        prefix = regime["name"]
        intercept_key = f"{prefix}_intercept"
        if intercept_key in param_dict:
            regime["model"]["intercept"] = param_dict[intercept_key]

        coeffs = regime["model"].get("feature_coefficients", {})
        for feat in list(coeffs.keys()):
            key = f"{prefix}_{feat}"
            if key in param_dict:
                coeffs[feat] = param_dict[key]

    return artifact


# ---------------------------------------------------------------------------
# BLIS execution
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


def run_blis_workload_spec(spec_path, exp_config, kv_blocks, artifact_path,
                           cpu_kv_blocks=0):
    """Run BLIS in workload-spec mode with a StepML artifact."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        results_path = f.name
    cmd = [
        BINARY_PATH, "run",
        "--model", exp_config.get("model", "unknown"),
        "--workload-spec", spec_path,
        "--tp", str(exp_config.get("tensor_parallelism", 1)),
        "--max-model-len", str(exp_config.get("max_model_len", 4096)),
        "--max-num-running-reqs", str(exp_config.get("max_num_seqs", 128)),
        "--max-num-scheduled-tokens", str(exp_config.get("max_num_batched_tokens", 2048)),
        "--total-kv-blocks", str(kv_blocks),
        "--block-size-in-tokens", str(BLOCK_SIZE_TOKENS),
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


def build_workload_spec(profile, ground_truth):
    """Build a workload-spec from inference-perf profile."""
    data = profile.get("data", {})
    load_config = profile.get("load", {})
    sp = data.get("shared_prefix", {})

    stages = []
    for stage in load_config.get("stages", []):
        stages.append({"rate": stage["rate"], "duration": stage["duration"]})
    if not stages:
        num_requests = ground_truth.get("num_requests", 1000)
        rps = ground_truth.get("throughput_rps", 10)
        duration = int(num_requests / rps) + 60
        stages.append({"rate": rps, "duration": duration})

    total_duration_s = sum(s["duration"] for s in stages)
    total_requests = sum(s["rate"] * s["duration"] for s in stages)
    horizon_us = int(total_duration_s * 1_000_000) + 60_000_000
    aggregate_rate = sum(s["rate"] * s["duration"] for s in stages) / total_duration_s

    return {
        "version": "2",
        "seed": 42,
        "num_requests": int(total_requests),
        "horizon": horizon_us,
        "aggregate_rate": aggregate_rate,
        "inference_perf": {
            "stages": stages,
            "shared_prefix": {
                "num_unique_system_prompts": sp.get("num_unique_system_prompts", 9),
                "num_users_per_system_prompt": sp.get("num_users_per_system_prompt", 5),
                "system_prompt_len": sp.get("system_prompt_len", 100),
                "question_len": sp.get("question_len", int(ground_truth.get("prompt_len_mean", 500))),
                "output_len": sp.get("output_len", int(ground_truth.get("output_len_mean", 250))),
                "enable_multi_turn_chat": sp.get("enable_multi_turn_chat", False),
            },
        },
    }


def load_profile(experiment_dir):
    profile_path = os.path.join(experiment_dir, "profile.yaml")
    if not os.path.isfile(profile_path):
        return None
    with open(profile_path) as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return yaml.safe_load(content)


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------
def discover_experiments():
    """Find all experiments with lifecycle data."""
    experiments = []
    for dirname in sorted(os.listdir(GT_DATA_ROOT)):
        dirpath = os.path.join(GT_DATA_ROOT, dirname)
        if not os.path.isdir(dirpath):
            continue
        lifecycle_path = os.path.join(dirpath, "results", "per_request_lifecycle_metrics.json")
        if not os.path.isfile(lifecycle_path):
            continue
        try:
            meta = parse_experiment_dir(dirname)
        except ValueError:
            continue

        kv_blocks = extract_kv_blocks(dirpath)
        if kv_blocks is None:
            continue
        cpu_kv_blocks = extract_cpu_kv_blocks(dirpath)
        exp_config = load_exp_config(dirpath)
        gt = load_ground_truth(dirpath)

        experiments.append({
            "dirname": dirname,
            "dirpath": dirpath,
            "meta": meta,
            "lifecycle_path": lifecycle_path,
            "kv_blocks": kv_blocks,
            "cpu_kv_blocks": cpu_kv_blocks,
            "exp_config": exp_config,
            "gt": gt,
        })
    return experiments


# ---------------------------------------------------------------------------
# H1: CMA-ES E2E optimization with trace replay
# ---------------------------------------------------------------------------
def run_h1(experiments, tmpdir):
    """H1: CMA-ES optimization of StepML parameters against BLIS E2E error."""
    print("\n" + "=" * 70)
    print("H1: CMA-ES E2E Optimization with Trace Replay")
    print("=" * 70)

    # Group experiments by model for per-model optimization
    model_experiments = {}
    for exp in experiments:
        model = exp["meta"]["model"]
        # Map to canonical model name for artifact lookup
        canonical = model
        for group_key, variants in MODEL_GROUPS.items():
            if model in variants:
                canonical = group_key
                break
        if canonical not in model_experiments:
            model_experiments[canonical] = []
        model_experiments[canonical].append(exp)

    # Prepare trace CSV files for all experiments
    trace_csvs = {}
    for exp in experiments:
        csv_path = os.path.join(tmpdir, f"{exp['dirname']}_trace.csv")
        trace_stats = convert_lifecycle_to_csv(
            exp["lifecycle_path"], csv_path,
            model_name=exp["exp_config"].get("model", exp["meta"]["model"]),
        )
        horizon_us = int((trace_stats["duration_s"] + 120) * 1_000_000)
        trace_csvs[exp["dirname"]] = (csv_path, horizon_us)

    # Run initial baseline (Round 2 coefficients)
    print("\n--- Initial Baseline (Round 2 coefficients) ---")
    initial_results = {}
    for exp in experiments:
        model = exp["meta"]["model"]
        artifact = load_artifact(model)
        art_path = os.path.join(tmpdir, f"{exp['dirname']}_initial.json")
        with open(art_path, "w") as f:
            json.dump(artifact, f)
        csv_path, horizon_us = trace_csvs[exp["dirname"]]
        blis = run_blis_trace(
            csv_path, exp["exp_config"], exp["kv_blocks"], art_path,
            horizon_us, exp["cpu_kv_blocks"],
        )
        if blis:
            e2e_err = compute_error(blis["e2e_mean_ms"], exp["gt"]["e2e_mean_s"] * 1000)
            initial_results[exp["dirname"]] = {
                "blis": blis, "e2e_error": e2e_err,
                "ttft_error": compute_error(blis["ttft_mean_ms"], exp["gt"]["ttft_mean_s"] * 1000),
                "itl_error": compute_error(blis["itl_mean_ms"], exp["gt"]["itl_mean_s"] * 1000),
            }
            print(f"  {exp['dirname']}: E2E={e2e_err*100:.1f}%")
        else:
            print(f"  {exp['dirname']}: BLIS FAILED")

    # Per-model CMA-ES optimization
    optimized_artifacts = {}
    optimization_logs = {}

    for canonical_model, exps in model_experiments.items():
        print(f"\n--- Optimizing: {canonical_model} ({len(exps)} experiments) ---")

        # Load base artifact
        base_artifact = load_artifact(canonical_model)
        initial_params, param_names = artifact_to_param_vector(base_artifact)
        n_params = len(initial_params)
        print(f"  Parameters: {n_params} ({', '.join(param_names[:5])}...)")
        print(f"  Initial: {dict(zip(param_names[:3], initial_params[:3]))}")

        # Define bounds: each param can vary between 0.1x and 10x initial
        # For zero initial values, use a reasonable range
        lower_bounds = []
        upper_bounds = []
        for i, (p, name) in enumerate(zip(initial_params, param_names)):
            if "overhead" in name and "per_req" not in name:
                # Step overhead: must stay positive, allow wide range
                lower_bounds.append(max(500, p * 0.3))
                upper_bounds.append(p * 5.0)
            elif "processing_time" in name:
                lower_bounds.append(0)
                upper_bounds.append(max(5000, abs(p) * 20))
            elif "intercept" in name:
                lower_bounds.append(p - abs(p) * 5 - 5000)
                upper_bounds.append(p + abs(p) * 5 + 5000)
            else:
                # Coefficients: allow sign flips
                lower_bounds.append(p - abs(p) * 5 - 100)
                upper_bounds.append(p + abs(p) * 5 + 100)

        # CMA-ES sigma: 30% of initial magnitude or 100 for near-zero params
        sigma0 = max(100, 0.3 * np.mean(np.abs(initial_params[initial_params != 0])))

        eval_count = 0
        convergence_log = []

        def objective(params):
            """Evaluate mean E2E error across experiments for this model."""
            nonlocal eval_count
            eval_count += 1

            # Reconstruct artifact
            artifact = param_vector_to_artifact(params, param_names, base_artifact)
            art_path = os.path.join(tmpdir, f"cma_{canonical_model}_{eval_count}.json")
            with open(art_path, "w") as f:
                json.dump(artifact, f)

            errors = []
            for exp in exps:
                csv_path, horizon_us = trace_csvs[exp["dirname"]]
                blis = run_blis_trace(
                    csv_path, exp["exp_config"], exp["kv_blocks"], art_path,
                    horizon_us, exp["cpu_kv_blocks"],
                )
                if blis is None:
                    errors.append(5.0)  # Penalty for failed runs
                else:
                    e2e_err = compute_error(
                        blis["e2e_mean_ms"], exp["gt"]["e2e_mean_s"] * 1000
                    )
                    errors.append(e2e_err)

            os.unlink(art_path)
            mean_err = np.mean(errors)
            convergence_log.append({"eval": eval_count, "mean_e2e_error": float(mean_err)})

            if eval_count % 10 == 0:
                print(f"    Eval {eval_count}: mean E2E error = {mean_err*100:.1f}%")

            return mean_err

        # Run CMA-ES
        opts = cma.CMAOptions()
        opts["maxfevals"] = 150  # Budget: 150 evaluations per model
        opts["timeout"] = 1800  # 30 min timeout per model
        opts["bounds"] = [lower_bounds, upper_bounds]
        opts["verbose"] = -1  # Suppress CMA output
        opts["seed"] = 42
        opts["popsize"] = 8  # Small population for fast iteration

        t0 = time.time()
        es = cma.CMAEvolutionStrategy(initial_params.tolist(), sigma0, opts)
        es.optimize(objective)
        elapsed = time.time() - t0

        best_params = es.result.xbest
        best_error = es.result.fbest
        print(f"  CMA-ES done in {elapsed:.0f}s, {eval_count} evals")
        print(f"  Best mean E2E error: {best_error*100:.1f}%")

        # Build optimized artifact
        optimized_art = param_vector_to_artifact(best_params, param_names, base_artifact)
        optimized_artifacts[canonical_model] = optimized_art

        # Log parameter changes
        param_changes = {}
        for name, old, new in zip(param_names, initial_params, best_params):
            pct_change = ((new - old) / abs(old) * 100) if old != 0 else (new * 100 if new != 0 else 0)
            param_changes[name] = {
                "initial": float(old), "optimized": float(new),
                "pct_change": float(pct_change),
            }

        optimization_logs[canonical_model] = {
            "n_evals": eval_count,
            "elapsed_s": elapsed,
            "initial_mean_e2e": float(np.mean([
                initial_results.get(e["dirname"], {}).get("e2e_error", 1.0)
                for e in exps
            ])),
            "best_mean_e2e": float(best_error),
            "param_changes": param_changes,
            "convergence": convergence_log,
        }

    # Final evaluation with optimized artifacts
    print("\n--- Final Evaluation (Optimized) ---")
    h1_results = []
    for exp in experiments:
        model = exp["meta"]["model"]
        # Find canonical model
        canonical = model
        for group_key, variants in MODEL_GROUPS.items():
            if model in variants:
                canonical = group_key
                break

        optimized_art = optimized_artifacts.get(canonical)
        if optimized_art is None:
            print(f"  {exp['dirname']}: NO OPTIMIZED ARTIFACT")
            continue

        # For grouped models (llama-2-70b variants), need to use the right base artifact
        if model != canonical and model in MODEL_ARTIFACT_MAP:
            # Load the variant's base artifact and apply the same parameter changes
            variant_base = load_artifact(model)
            variant_params, variant_names = artifact_to_param_vector(variant_base)
            # Apply the same relative changes from canonical optimization
            canon_base = load_artifact(canonical)
            canon_initial, canon_names = artifact_to_param_vector(canon_base)
            canon_optimized, _ = artifact_to_param_vector(optimized_art)

            # Apply proportional changes
            new_params = np.copy(variant_params)
            for i, name in enumerate(variant_names):
                if i < len(canon_names) and name == canon_names[i]:
                    if canon_initial[i] != 0:
                        ratio = canon_optimized[i] / canon_initial[i]
                        new_params[i] = variant_params[i] * ratio
                    else:
                        new_params[i] = canon_optimized[i]
            art_to_use = param_vector_to_artifact(new_params, variant_names, variant_base)
        else:
            art_to_use = optimized_art

        art_path = os.path.join(tmpdir, f"final_{exp['dirname']}.json")
        with open(art_path, "w") as f:
            json.dump(art_to_use, f)

        csv_path, horizon_us = trace_csvs[exp["dirname"]]
        blis = run_blis_trace(
            csv_path, exp["exp_config"], exp["kv_blocks"], art_path,
            horizon_us, exp["cpu_kv_blocks"],
        )

        if blis:
            e2e_err = compute_error(blis["e2e_mean_ms"], exp["gt"]["e2e_mean_s"] * 1000)
            ttft_err = compute_error(blis["ttft_mean_ms"], exp["gt"]["ttft_mean_s"] * 1000)
            itl_err = compute_error(blis["itl_mean_ms"], exp["gt"]["itl_mean_s"] * 1000)

            # Get initial for comparison
            init = initial_results.get(exp["dirname"], {})
            init_e2e = init.get("e2e_error", 1.0) * 100

            print(f"  {exp['dirname']}: E2E={e2e_err*100:.1f}% (was {init_e2e:.1f}%), "
                  f"TTFT={ttft_err*100:.1f}%, ITL={itl_err*100:.1f}%")

            h1_results.append({
                "experiment": exp["dirname"],
                "model": model,
                "workload": exp["meta"]["workload"],
                "tp": exp["meta"]["tp"],
                "gt_e2e_ms": exp["gt"]["e2e_mean_s"] * 1000,
                "gt_ttft_ms": exp["gt"]["ttft_mean_s"] * 1000,
                "gt_itl_ms": exp["gt"]["itl_mean_s"] * 1000,
                "blis_e2e_ms": blis["e2e_mean_ms"],
                "blis_ttft_ms": blis["ttft_mean_ms"],
                "blis_itl_ms": blis["itl_mean_ms"],
                "blis_completed": blis["completed_requests"],
                "e2e_error": e2e_err,
                "ttft_error": ttft_err,
                "itl_error": itl_err,
                "initial_e2e_error": init.get("e2e_error", None),
            })

        # Save the optimized artifact for this experiment
        final_art_dir = os.path.join(SCRIPT_DIR, "h1-trace-e2e-opt", "artifacts")
        os.makedirs(final_art_dir, exist_ok=True)
        final_art_path = os.path.join(final_art_dir, f"{model}_optimized.json")
        with open(final_art_path, "w") as f:
            json.dump(art_to_use, f, indent=2)

    return h1_results, initial_results, optimization_logs, optimized_artifacts


# ---------------------------------------------------------------------------
# H2: Workload-spec mode E2E optimization
# ---------------------------------------------------------------------------
def run_h2(experiments, optimized_artifacts, tmpdir):
    """H2: Run optimized artifacts in workload-spec mode."""
    print("\n" + "=" * 70)
    print("H2: Workload-Spec Mode with E2E-Optimized Coefficients")
    print("=" * 70)

    h2_results = []

    for exp in experiments:
        model = exp["meta"]["model"]
        dirname = exp["dirname"]

        # Find canonical model
        canonical = model
        for group_key, variants in MODEL_GROUPS.items():
            if model in variants:
                canonical = group_key
                break

        # Try to load profile from original eval directory
        orig_dir = os.path.join(ORIG_GT_ROOT, dirname)
        profile = load_profile(orig_dir) if os.path.isdir(orig_dir) else None
        if profile is None:
            profile = load_profile(exp["dirpath"])
        if profile is None:
            print(f"  {dirname}: SKIP (no profile)")
            continue

        # Build workload spec
        spec = build_workload_spec(profile, exp["gt"])
        spec_path = os.path.join(tmpdir, f"{dirname}_spec.yaml")
        with open(spec_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)

        # Use the same optimized artifact from H1
        optimized_art = optimized_artifacts.get(canonical)
        if optimized_art is None:
            print(f"  {dirname}: NO OPTIMIZED ARTIFACT")
            continue

        # Handle variant models
        if model != canonical and model in MODEL_ARTIFACT_MAP:
            variant_base = load_artifact(model)
            variant_params, variant_names = artifact_to_param_vector(variant_base)
            canon_base = load_artifact(canonical)
            canon_initial, canon_names = artifact_to_param_vector(canon_base)
            canon_optimized, _ = artifact_to_param_vector(optimized_art)
            new_params = np.copy(variant_params)
            for i, name in enumerate(variant_names):
                if i < len(canon_names) and name == canon_names[i]:
                    if canon_initial[i] != 0:
                        new_params[i] = variant_params[i] * (canon_optimized[i] / canon_initial[i])
                    else:
                        new_params[i] = canon_optimized[i]
            art_to_use = param_vector_to_artifact(new_params, variant_names, variant_base)
        else:
            art_to_use = optimized_art

        art_path = os.path.join(tmpdir, f"h2_{dirname}.json")
        with open(art_path, "w") as f:
            json.dump(art_to_use, f)

        blis = run_blis_workload_spec(
            spec_path, exp["exp_config"], exp["kv_blocks"], art_path,
            exp["cpu_kv_blocks"],
        )

        if blis:
            e2e_err = compute_error(blis["e2e_mean_ms"], exp["gt"]["e2e_mean_s"] * 1000)
            ttft_err = compute_error(blis["ttft_mean_ms"], exp["gt"]["ttft_mean_s"] * 1000)
            itl_err = compute_error(blis["itl_mean_ms"], exp["gt"]["itl_mean_s"] * 1000)
            print(f"  {dirname}: E2E={e2e_err*100:.1f}%, TTFT={ttft_err*100:.1f}%, ITL={itl_err*100:.1f}%")

            h2_results.append({
                "experiment": dirname,
                "model": model,
                "workload": exp["meta"]["workload"],
                "tp": exp["meta"]["tp"],
                "gt_e2e_ms": exp["gt"]["e2e_mean_s"] * 1000,
                "gt_ttft_ms": exp["gt"]["ttft_mean_s"] * 1000,
                "gt_itl_ms": exp["gt"]["itl_mean_s"] * 1000,
                "blis_e2e_ms": blis["e2e_mean_ms"],
                "blis_ttft_ms": blis["ttft_mean_ms"],
                "blis_itl_ms": blis["itl_mean_ms"],
                "blis_completed": blis["completed_requests"],
                "e2e_error": e2e_err,
                "ttft_error": ttft_err,
                "itl_error": itl_err,
            })
        else:
            print(f"  {dirname}: BLIS FAILED")

    return h2_results


# ---------------------------------------------------------------------------
# H3: Additive correction factors
# ---------------------------------------------------------------------------
def run_h3(h1_results, experiments, optimized_artifacts, tmpdir):
    """H3: Test additive/multiplicative correction factors on TTFT and ITL."""
    print("\n" + "=" * 70)
    print("H3: Additive Correction Factors")
    print("=" * 70)

    if not h1_results:
        print("  SKIP: No H1 results to compute corrections from.")
        return [], {}

    # Compute TTFT and ITL residuals per model
    model_residuals = {}
    for r in h1_results:
        model = r["model"]
        if model not in model_residuals:
            model_residuals[model] = {"ttft": [], "itl": [], "gt_ttft": [], "gt_itl": []}
        model_residuals[model]["ttft"].append(r["blis_ttft_ms"] - r["gt_ttft_ms"])
        model_residuals[model]["itl"].append(r["blis_itl_ms"] - r["gt_itl_ms"])
        model_residuals[model]["gt_ttft"].append(r["gt_ttft_ms"])
        model_residuals[model]["gt_itl"].append(r["gt_itl_ms"])

    # Compute correction factors per model
    corrections = {}
    print("\n--- Residual Analysis ---")
    for model, resid in model_residuals.items():
        ttft_residuals = np.array(resid["ttft"])
        itl_residuals = np.array(resid["itl"])
        gt_ttft = np.array(resid["gt_ttft"])
        gt_itl = np.array(resid["gt_itl"])

        # TTFT correction: blis_ttft = alpha * gt_ttft + beta
        # We want corrected = blis / alpha - beta/alpha
        # But simpler: compute additive offset = mean(gt_ttft - blis_ttft)
        ttft_offset = float(np.mean(-ttft_residuals))  # Add this to blis_ttft
        # Multiplicative: gt / blis ratio
        blis_ttft_arr = gt_ttft + ttft_residuals  # = blis_ttft
        ttft_ratio = float(np.mean(gt_ttft / np.maximum(blis_ttft_arr, 0.01)))

        # ITL correction
        itl_offset = float(np.mean(-itl_residuals))
        blis_itl_arr = gt_itl + itl_residuals
        itl_ratio = float(np.mean(gt_itl / np.maximum(blis_itl_arr, 0.01)))

        corrections[model] = {
            "ttft_additive_ms": ttft_offset,
            "ttft_multiplicative": ttft_ratio,
            "itl_additive_ms": itl_offset,
            "itl_multiplicative": itl_ratio,
            "ttft_residual_mean_ms": float(np.mean(ttft_residuals)),
            "ttft_residual_std_ms": float(np.std(ttft_residuals)),
            "itl_residual_mean_ms": float(np.mean(itl_residuals)),
            "itl_residual_std_ms": float(np.std(itl_residuals)),
        }
        print(f"  {model}: TTFT residual={np.mean(ttft_residuals):.1f}±{np.std(ttft_residuals):.1f}ms, "
              f"ITL residual={np.mean(itl_residuals):.2f}±{np.std(itl_residuals):.2f}ms")
        print(f"    TTFT correction: additive={ttft_offset:.1f}ms, multiplicative={ttft_ratio:.2f}x")
        print(f"    ITL correction: additive={itl_offset:.2f}ms, multiplicative={itl_ratio:.2f}x")

    # Apply corrections via QueueingTime and OutputTokenProcessingTime adjustments
    print("\n--- Applying Corrections ---")
    # Strategy: increase step_overhead_us (the main knob controlling E2E under-prediction)
    # and adjust queueing_time intercept (controls TTFT)
    h3_results = []

    for exp in experiments:
        model = exp["meta"]["model"]
        dirname = exp["dirname"]

        canonical = model
        for group_key, variants in MODEL_GROUPS.items():
            if model in variants:
                canonical = group_key
                break

        optimized_art = optimized_artifacts.get(canonical)
        if optimized_art is None:
            continue

        # Get the artifact to use (handling variants)
        if model != canonical and model in MODEL_ARTIFACT_MAP:
            variant_base = load_artifact(model)
            variant_params, variant_names = artifact_to_param_vector(variant_base)
            canon_base = load_artifact(canonical)
            canon_initial, canon_names = artifact_to_param_vector(canon_base)
            canon_optimized, _ = artifact_to_param_vector(optimized_art)
            new_params = np.copy(variant_params)
            for i, name in enumerate(variant_names):
                if i < len(canon_names) and name == canon_names[i]:
                    if canon_initial[i] != 0:
                        new_params[i] = variant_params[i] * (canon_optimized[i] / canon_initial[i])
                    else:
                        new_params[i] = canon_optimized[i]
            corrected_art = param_vector_to_artifact(new_params, variant_names, variant_base)
        else:
            corrected_art = copy.deepcopy(optimized_art)

        # Apply TTFT correction: increase queueing time intercept
        corr = corrections.get(model, corrections.get(canonical, {}))
        ttft_add = corr.get("ttft_additive_ms", 0) * 1000  # Convert ms → us
        if corrected_art.get("queueing_time"):
            corrected_art["queueing_time"]["intercept"] += ttft_add
        else:
            corrected_art["queueing_time"] = {
                "model_type": "linear",
                "intercept": ttft_add,
                "feature_coefficients": {},
            }

        # Apply ITL correction: adjust output_token_processing_time
        itl_add = corr.get("itl_additive_ms", 0) * 1000  # ms → us
        corrected_art["output_token_processing_time_us"] = max(
            0, corrected_art.get("output_token_processing_time_us", 0) + itl_add
        )

        art_path = os.path.join(tmpdir, f"h3_{dirname}.json")
        with open(art_path, "w") as f:
            json.dump(corrected_art, f)

        csv_path, horizon_us = os.path.join(tmpdir, f"{dirname}_trace.csv"), None
        # Recompute horizon
        with open(exp["lifecycle_path"]) as f:
            lc_data = json.load(f)
        arrivals = [e["start_time"] for e in lc_data]
        duration_s = max(arrivals) - min(arrivals)
        horizon_us = int((duration_s + 120) * 1_000_000)

        blis = run_blis_trace(
            csv_path, exp["exp_config"], exp["kv_blocks"], art_path,
            horizon_us, exp["cpu_kv_blocks"],
        )

        if blis:
            e2e_err = compute_error(blis["e2e_mean_ms"], exp["gt"]["e2e_mean_s"] * 1000)
            ttft_err = compute_error(blis["ttft_mean_ms"], exp["gt"]["ttft_mean_s"] * 1000)
            itl_err = compute_error(blis["itl_mean_ms"], exp["gt"]["itl_mean_s"] * 1000)

            # Get H1 baseline for comparison
            h1_match = [r for r in h1_results if r["experiment"] == dirname]
            h1_e2e = h1_match[0]["e2e_error"] * 100 if h1_match else None

            print(f"  {dirname}: E2E={e2e_err*100:.1f}%"
                  + (f" (H1: {h1_e2e:.1f}%)" if h1_e2e else "")
                  + f", TTFT={ttft_err*100:.1f}%, ITL={itl_err*100:.1f}%")

            h3_results.append({
                "experiment": dirname,
                "model": model,
                "workload": exp["meta"]["workload"],
                "tp": exp["meta"]["tp"],
                "gt_e2e_ms": exp["gt"]["e2e_mean_s"] * 1000,
                "gt_ttft_ms": exp["gt"]["ttft_mean_s"] * 1000,
                "gt_itl_ms": exp["gt"]["itl_mean_s"] * 1000,
                "blis_e2e_ms": blis["e2e_mean_ms"],
                "blis_ttft_ms": blis["ttft_mean_ms"],
                "blis_itl_ms": blis["itl_mean_ms"],
                "blis_completed": blis["completed_requests"],
                "e2e_error": e2e_err,
                "ttft_error": ttft_err,
                "itl_error": itl_err,
                "h1_e2e_error": h1_match[0]["e2e_error"] if h1_match else None,
            })

    return h3_results, corrections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def write_results_csv(results, path):
    if not results:
        return
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    print("=" * 70)
    print("Round 3, Idea 3: E2E Calibration via Direct E2E Objective")
    print("=" * 70)

    # Ensure binary exists
    if not os.path.isfile(BINARY_PATH):
        print("Building simulation_worker...")
        build = subprocess.run(
            ["go", "build", "-o", BINARY_PATH, "main.go"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if build.returncode != 0:
            print(f"Build failed: {build.stderr}", file=sys.stderr)
            sys.exit(1)

    experiments = discover_experiments()
    print(f"\nFound {len(experiments)} experiments with lifecycle data.")

    if not experiments:
        print("ERROR: No experiments found!", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # H1: CMA-ES with trace replay
        h1_results, initial_results, opt_logs, optimized_artifacts = run_h1(experiments, tmpdir)

        # H2: Workload-spec mode with optimized coefficients
        h2_results = run_h2(experiments, optimized_artifacts, tmpdir)

        # H3: Additive correction factors
        h3_results, corrections = run_h3(h1_results, experiments, optimized_artifacts, tmpdir)

    # ---------------------------------------------------------------------------
    # Write all results
    # ---------------------------------------------------------------------------
    h1_dir = os.path.join(SCRIPT_DIR, "h1-trace-e2e-opt")
    h2_dir = os.path.join(SCRIPT_DIR, "h2-workload-spec-opt")
    h3_dir = os.path.join(SCRIPT_DIR, "h3-additive-corrections")

    os.makedirs(h1_dir, exist_ok=True)
    os.makedirs(h2_dir, exist_ok=True)
    os.makedirs(h3_dir, exist_ok=True)

    # H1 outputs
    write_results_csv(h1_results, os.path.join(h1_dir, "e2e_opt_results.csv"))
    if h1_results:
        ok = [r for r in h1_results if "e2e_error" in r]
        h1_summary = {
            "n_experiments": len(ok),
            "mean_e2e_error_pct": np.mean([r["e2e_error"] for r in ok]) * 100 if ok else None,
            "mean_ttft_error_pct": np.mean([r["ttft_error"] for r in ok]) * 100 if ok else None,
            "mean_itl_error_pct": np.mean([r["itl_error"] for r in ok]) * 100 if ok else None,
            "e2e_under_10pct": sum(1 for r in ok if r["e2e_error"] < 0.10),
            "e2e_under_15pct": sum(1 for r in ok if r["e2e_error"] < 0.15),
            "e2e_under_25pct": sum(1 for r in ok if r["e2e_error"] < 0.25),
            "initial_mean_e2e_pct": np.mean([
                initial_results.get(r["experiment"], {}).get("e2e_error", 1.0)
                for r in ok
            ]) * 100 if ok else None,
            "optimization_logs": opt_logs,
            "per_experiment": ok,
        }
        with open(os.path.join(h1_dir, "summary.json"), "w") as f:
            json.dump(h1_summary, f, indent=2)

    # H2 outputs
    write_results_csv(h2_results, os.path.join(h2_dir, "workload_spec_results.csv"))
    if h2_results:
        ok = [r for r in h2_results if "e2e_error" in r]
        h2_summary = {
            "n_experiments": len(ok),
            "mean_e2e_error_pct": np.mean([r["e2e_error"] for r in ok]) * 100 if ok else None,
            "mean_ttft_error_pct": np.mean([r["ttft_error"] for r in ok]) * 100 if ok else None,
            "mean_itl_error_pct": np.mean([r["itl_error"] for r in ok]) * 100 if ok else None,
            "e2e_under_50pct": sum(1 for r in ok if r["e2e_error"] < 0.50),
            "per_experiment": ok,
        }
        with open(os.path.join(h2_dir, "summary.json"), "w") as f:
            json.dump(h2_summary, f, indent=2)

    # H3 outputs
    write_results_csv(h3_results, os.path.join(h3_dir, "correction_results.csv"))
    if h3_results:
        ok = [r for r in h3_results if "e2e_error" in r]
        h3_summary = {
            "n_experiments": len(ok),
            "mean_e2e_error_pct": np.mean([r["e2e_error"] for r in ok]) * 100 if ok else None,
            "mean_ttft_error_pct": np.mean([r["ttft_error"] for r in ok]) * 100 if ok else None,
            "mean_itl_error_pct": np.mean([r["itl_error"] for r in ok]) * 100 if ok else None,
            "corrections": corrections,
            "per_experiment": ok,
        }
        with open(os.path.join(h3_dir, "summary.json"), "w") as f:
            json.dump(h3_summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for label, results in [("H1 (trace+CMA-ES)", h1_results),
                           ("H2 (workload-spec)", h2_results),
                           ("H3 (corrections)", h3_results)]:
        ok = [r for r in results if "e2e_error" in r]
        if ok:
            mean_e2e = np.mean([r["e2e_error"] for r in ok]) * 100
            mean_ttft = np.mean([r["ttft_error"] for r in ok]) * 100
            mean_itl = np.mean([r["itl_error"] for r in ok]) * 100
            under_10 = sum(1 for r in ok if r["e2e_error"] < 0.10)
            print(f"\n{label}: E2E={mean_e2e:.1f}%, TTFT={mean_ttft:.1f}%, "
                  f"ITL={mean_itl:.1f}%, E2E<10%={under_10}/{len(ok)}")
        else:
            print(f"\n{label}: No results")

    print("\nDone!")


if __name__ == "__main__":
    main()
