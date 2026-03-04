#!/usr/bin/env python3
"""Round 3, Idea 1: Trace-Driven Simulation with Lifecycle Replay.

Runs all three sub-hypotheses:
  H1: Trace replay reduces TTFT error (vs workload-spec mode)
  H2: Error attribution with trace replay (TTFT vs ITL decomposition)
  H3: Workload-spec parameter diagnosis (which parameter is wrong)

Uses the legacy CSV trace path: --workload traces --workload-traces-filepath <csv>
with Round 2's best StepML artifacts (regime ensemble + calibrated overheads).
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
SHARED_DIR = os.path.join(SCRIPT_DIR, "..", "..", "shared")

# The worktree is under stepml-worktrees/ — use absolute path to package root
PACKAGE_ROOT = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package"

# Ground truth data in BLIS-research repo (has per_request_lifecycle_metrics.json)
BLIS_RESEARCH_ROOT = os.path.join(PACKAGE_ROOT, "BLIS-research")
GT_DATA_ROOT = os.path.join(BLIS_RESEARCH_ROOT, "eval", "ground_truth")

# Round 2 best artifacts (calibrated regime ensemble)
R2_ARTIFACTS_DIR = os.path.join(
    BLIS_RESEARCH_ROOT,
    "hypotheses", "h-stepml", "round2",
    "idea-2-regime-ensemble", "h3-secondary-method-calibration",
    "output", "calibrated_artifacts",
)

# Original eval data (inference-sim repo, has profile.yaml for workload-spec comparison)
ORIG_GT_ROOT = os.path.join(PACKAGE_ROOT, "inference-sim", "eval", "ground_truth")

BINARY_PATH = os.path.join(REPO_ROOT, "simulation_worker")
BLOCK_SIZE_TOKENS = 16

# Model name → artifact mapping
MODEL_ARTIFACT_MAP = {
    "llama-2-7b": "llama-2-7b_tp1_regime.json",
    "llama-2-70b": "llama-2-70b_tp4_regime.json",
    "llama-2-70b-hf": "llama-2-70b-hf_tp4_regime.json",
    "codellama-34b": "codellama-34b_tp2_regime.json",
    "mixtral-8x7b-v0-1": "mixtral-8x7b-v0-1_tp2_regime.json",
}

# Round 2 workload-spec results (from FINDINGS_ROUND2.md / e2e_summary_regime.json)
R2_WORKLOAD_SPEC_RESULTS = {
    "mean_e2e_error_pct": 427.81,
    "mean_ttft_error_pct": 31905.77,
    "mean_itl_error_pct": 33.64,
    "experiments_below_10pct_e2e": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_experiment_dir(dirname):
    """Parse model, tp, workload from directory name."""
    import re
    # Try new format: YYYYMMDD-HHMMSS-model-tpN-workload
    # or old format: YYYYMMDD-model-tpN-workload
    tp_matches = list(re.finditer(r"-tp(\d+)-", dirname))
    if not tp_matches:
        raise ValueError(f"Cannot parse experiment dir: {dirname}")
    last_tp = tp_matches[-1]
    # Check if starts with full timestamp (YYYYMMDD-HHMMSS)
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
    """Load ground-truth latency metrics."""
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
        "throughput_rps": successes.get("throughput", {}).get(
            "requests_per_sec", 0
        ),
    }


def load_exp_config(experiment_dir):
    """Load exp-config.yaml."""
    config_path = os.path.join(experiment_dir, "exp-config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_kv_blocks(experiment_dir):
    """Extract GPU KV cache blocks from vllm.log."""
    import re
    vllm_log = os.path.join(experiment_dir, "vllm.log")
    if not os.path.isfile(vllm_log):
        return None
    pattern = re.compile(r"GPU KV cache size:\s+([\d,]+)\s+tokens")
    with open(vllm_log) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                tokens = int(m.group(1).replace(",", ""))
                return tokens // BLOCK_SIZE_TOKENS
    return None


def extract_cpu_kv_blocks(experiment_dir):
    """Extract CPU KV blocks from vllm.log for tiered caching."""
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
    """Convert per_request_lifecycle_metrics.json to BLIS legacy trace CSV."""
    import random
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
        "min_arrival_s": rows[0][0] if rows else 0,
        "max_arrival_s": rows[-1][0] if rows else 0,
        "duration_s": (rows[-1][0] - rows[0][0]) if len(rows) > 1 else 0,
        "mean_input_tokens": sum(len(r[3]) for r in rows) / len(rows) if rows else 0,
        "mean_output_tokens": sum(len(r[4]) for r in rows) / len(rows) if rows else 0,
    }


def compute_error(predicted, observed):
    """Relative error: |predicted - observed| / observed."""
    if observed == 0:
        return float("inf") if predicted != 0 else 0.0
    return abs(predicted - observed) / observed


def parse_blis_stdout(stdout):
    """Parse BLIS JSON output from stdout."""
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
        "tokens_per_sec": metrics.get("tokens_per_sec", 0),
    }


def run_blis_trace_replay(
    trace_csv_path, exp_config, total_kv_blocks, stepml_artifact_path,
    horizon_us, cpu_kv_blocks=0,
):
    """Run BLIS with legacy trace CSV replay + StepML artifact."""
    model_name = exp_config.get("model", "unknown")
    tp = exp_config.get("tensor_parallelism", 1)
    max_model_len = exp_config.get("max_model_len", 4096)
    max_num_seqs = exp_config.get("max_num_seqs", 128)
    max_num_batched_tokens = exp_config.get("max_num_batched_tokens", 2048)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        results_path = f.name

    cmd = [
        BINARY_PATH, "run",
        "--model", model_name,
        "--workload", "traces",
        "--workload-traces-filepath", trace_csv_path,
        "--tp", str(tp),
        "--max-model-len", str(max_model_len),
        "--max-num-running-reqs", str(max_num_seqs),
        "--max-num-scheduled-tokens", str(max_num_batched_tokens),
        "--total-kv-blocks", str(total_kv_blocks),
        "--block-size-in-tokens", str(BLOCK_SIZE_TOKENS),
        "--horizon", str(horizon_us),
        "--alpha-coeffs=1,0,0",
        "--beta-coeffs=1,0,0",
        "--stepml-model", stepml_artifact_path,
        "--results-path", results_path,
        "--log", "error",
    ]

    if cpu_kv_blocks > 0:
        cmd.extend(["--kv-cpu-blocks", str(cpu_kv_blocks)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(BINARY_PATH),
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 600s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"  Binary not found: {BINARY_PATH}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(results_path):
            os.unlink(results_path)

    if result.returncode != 0:
        print(f"  BLIS failed (exit {result.returncode})", file=sys.stderr)
        print(f"  stderr: {result.stderr[:1000]}", file=sys.stderr)
        return None

    return parse_blis_stdout(result.stdout)


def load_profile_data(experiment_dir):
    """Load profile.yaml (JSON format) for workload-spec parameter comparison."""
    profile_path = os.path.join(experiment_dir, "profile.yaml")
    if not os.path.isfile(profile_path):
        return None
    with open(profile_path) as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            return None


def extract_lifecycle_stats(lifecycle_path):
    """Extract ground-truth workload statistics from lifecycle data."""
    with open(lifecycle_path) as f:
        data = json.load(f)
    if not data:
        return None
    arrivals = [e["start_time"] for e in data]
    input_tokens = [e.get("info", {}).get("input_tokens", 0) for e in data]
    output_tokens = [e.get("info", {}).get("output_tokens", 0) for e in data]

    import numpy as np
    arrivals_arr = np.array(arrivals)
    duration_s = arrivals_arr.max() - arrivals_arr.min()
    arrival_rate = (len(data) - 1) / duration_s if duration_s > 0 else 0

    return {
        "num_requests": len(data),
        "duration_s": duration_s,
        "arrival_rate_rps": arrival_rate,
        "input_tokens_mean": np.mean(input_tokens),
        "input_tokens_std": np.std(input_tokens),
        "output_tokens_mean": np.mean(output_tokens),
        "output_tokens_std": np.std(output_tokens),
    }


def extract_workload_spec_stats(profile):
    """Extract workload-spec parameters from profile.yaml."""
    if profile is None:
        return None
    load_config = profile.get("load", {})
    data = profile.get("data", {})
    sp = data.get("shared_prefix", {})

    stages = load_config.get("stages", [])
    if not stages:
        return None

    total_duration = sum(s["duration"] for s in stages)
    total_requests = sum(s["rate"] * s["duration"] for s in stages)
    avg_rate = total_requests / total_duration if total_duration > 0 else 0

    return {
        "num_requests": int(total_requests),
        "duration_s": total_duration,
        "arrival_rate_rps": avg_rate,
        "question_len": sp.get("question_len", 0),
        "output_len": sp.get("output_len", 0),
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Round 3, Idea 1: Trace-Driven Simulation with Lifecycle Replay")
    print("=" * 70)

    # Step 0: Build BLIS binary
    if not os.path.isfile(BINARY_PATH):
        print("\nBuilding simulation_worker...")
        build = subprocess.run(
            ["go", "build", "-o", BINARY_PATH, "main.go"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if build.returncode != 0:
            print(f"Build failed: {build.stderr}", file=sys.stderr)
            sys.exit(1)
        print("  Build successful.")
    else:
        print(f"\nUsing existing binary: {BINARY_PATH}")

    # Step 1: Discover experiments
    if not os.path.isdir(GT_DATA_ROOT):
        print(f"ERROR: Ground truth data not found at {GT_DATA_ROOT}", file=sys.stderr)
        sys.exit(1)

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
        experiments.append({
            "dirname": dirname,
            "dirpath": dirpath,
            "meta": meta,
            "lifecycle_path": lifecycle_path,
        })

    print(f"\nFound {len(experiments)} experiments with lifecycle data.")

    # Step 2: Run BLIS with trace replay for each experiment
    h1_results = []
    h2_results = []
    h3_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for exp in experiments:
            dirname = exp["dirname"]
            dirpath = exp["dirpath"]
            meta = exp["meta"]
            model = meta["model"]
            workload = meta["workload"]

            print(f"\n{'=' * 60}")
            print(f"Experiment: {dirname}")
            print(f"  Model: {model}, TP: {meta['tp']}, Workload: {workload}")
            print(f"{'=' * 60}")

            # Load ground truth
            gt = load_ground_truth(dirpath)
            print(f"  GT: E2E={gt['e2e_mean_s']*1000:.1f}ms, "
                  f"TTFT={gt['ttft_mean_s']*1000:.1f}ms, "
                  f"ITL={gt['itl_mean_s']*1000:.2f}ms, "
                  f"N={gt['num_requests']}")

            # Load experiment config
            exp_config = load_exp_config(dirpath)

            # Extract KV blocks
            total_kv_blocks = extract_kv_blocks(dirpath)
            if total_kv_blocks is None:
                print("  SKIP: Cannot parse KV blocks", file=sys.stderr)
                continue
            cpu_kv_blocks = extract_cpu_kv_blocks(dirpath)
            print(f"  KV blocks: {total_kv_blocks} GPU + {cpu_kv_blocks} CPU")

            # Find StepML artifact for this model
            artifact_file = MODEL_ARTIFACT_MAP.get(model)
            if artifact_file is None:
                print(f"  SKIP: No artifact for model '{model}'", file=sys.stderr)
                continue
            artifact_path = os.path.join(R2_ARTIFACTS_DIR, artifact_file)
            if not os.path.isfile(artifact_path):
                print(f"  SKIP: Artifact not found: {artifact_path}", file=sys.stderr)
                continue
            print(f"  Artifact: {artifact_file}")

            # Convert lifecycle data to trace CSV
            csv_path = os.path.join(tmpdir, f"{dirname}.csv")
            trace_stats = convert_lifecycle_to_csv(
                exp["lifecycle_path"], csv_path,
                model_name=exp_config.get("model", model),
            )
            print(f"  Trace: {trace_stats['num_requests']} requests, "
                  f"{trace_stats['duration_s']:.1f}s duration, "
                  f"mean input={trace_stats['mean_input_tokens']:.0f}, "
                  f"mean output={trace_stats['mean_output_tokens']:.0f}")

            # Set horizon: trace duration + 120s buffer (in microseconds)
            horizon_us = int((trace_stats["duration_s"] + 120) * 1_000_000)

            # Run BLIS with trace replay
            blis = run_blis_trace_replay(
                trace_csv_path=csv_path,
                exp_config=exp_config,
                total_kv_blocks=total_kv_blocks,
                stepml_artifact_path=artifact_path,
                horizon_us=horizon_us,
                cpu_kv_blocks=cpu_kv_blocks,
            )

            if blis is None:
                print("  BLIS run failed!")
                h1_results.append({
                    "experiment": dirname, "model": model,
                    "workload": workload, "tp": meta["tp"],
                    "status": "blis_failed",
                })
                continue

            # Compute errors
            e2e_err = compute_error(blis["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
            ttft_err = compute_error(blis["ttft_mean_ms"], gt["ttft_mean_s"] * 1000)
            itl_err = compute_error(blis["itl_mean_ms"], gt["itl_mean_s"] * 1000)

            print(f"  BLIS: E2E={blis['e2e_mean_ms']:.1f}ms, "
                  f"TTFT={blis['ttft_mean_ms']:.1f}ms, "
                  f"ITL={blis['itl_mean_ms']:.2f}ms, "
                  f"completed={blis['completed_requests']}")
            print(f"  Errors: E2E={e2e_err*100:.1f}%, "
                  f"TTFT={ttft_err*100:.1f}%, "
                  f"ITL={itl_err*100:.1f}%")

            row = {
                "experiment": dirname,
                "model": model,
                "workload": workload,
                "tp": meta["tp"],
                "status": "ok",
                "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
                "gt_ttft_ms": gt["ttft_mean_s"] * 1000,
                "gt_itl_ms": gt["itl_mean_s"] * 1000,
                "gt_requests": gt["num_requests"],
                "blis_e2e_ms": blis["e2e_mean_ms"],
                "blis_ttft_ms": blis["ttft_mean_ms"],
                "blis_itl_ms": blis["itl_mean_ms"],
                "blis_completed": blis["completed_requests"],
                "e2e_error": e2e_err,
                "ttft_error": ttft_err,
                "itl_error": itl_err,
            }
            h1_results.append(row)

            # H2: Error attribution
            if row["status"] == "ok":
                h2_row = dict(row)
                # Decompose: is E2E error TTFT-dominated or ITL-dominated?
                h2_row["ttft_contribution_ms"] = blis["ttft_mean_ms"] - gt["ttft_mean_s"] * 1000
                itl_contribution_ms = (
                    (blis["itl_mean_ms"] - gt["itl_mean_s"] * 1000) *
                    gt.get("output_len_mean", 0)
                )
                h2_row["itl_contribution_ms"] = itl_contribution_ms
                h2_row["ttft_abs_error_ms"] = abs(blis["ttft_mean_ms"] - gt["ttft_mean_s"] * 1000)
                h2_row["itl_abs_error_ms"] = abs(blis["itl_mean_ms"] - gt["itl_mean_s"] * 1000)
                h2_results.append(h2_row)

            # H3: Workload-spec parameter diagnosis
            lifecycle_stats = extract_lifecycle_stats(exp["lifecycle_path"])
            # Try to load profile from original eval directory
            orig_exp_dir = os.path.join(ORIG_GT_ROOT, dirname) if ORIG_GT_ROOT else None
            profile = None
            if orig_exp_dir and os.path.isdir(orig_exp_dir):
                profile = load_profile_data(orig_exp_dir)
            # Also try BLIS-research ground truth
            if profile is None:
                profile = load_profile_data(dirpath)

            spec_stats = extract_workload_spec_stats(profile) if profile else None

            h3_row = {
                "experiment": dirname,
                "model": model,
                "workload": workload,
            }
            if lifecycle_stats:
                h3_row["gt_num_requests"] = lifecycle_stats["num_requests"]
                h3_row["gt_duration_s"] = lifecycle_stats["duration_s"]
                h3_row["gt_arrival_rate_rps"] = lifecycle_stats["arrival_rate_rps"]
                h3_row["gt_input_mean"] = lifecycle_stats["input_tokens_mean"]
                h3_row["gt_output_mean"] = lifecycle_stats["output_tokens_mean"]

            if spec_stats:
                h3_row["spec_num_requests"] = spec_stats["num_requests"]
                h3_row["spec_duration_s"] = spec_stats["duration_s"]
                h3_row["spec_arrival_rate_rps"] = spec_stats["arrival_rate_rps"]
                h3_row["spec_question_len"] = spec_stats["question_len"]
                h3_row["spec_output_len"] = spec_stats["output_len"]

                # Compute relative errors on each parameter
                if lifecycle_stats:
                    gt_rate = lifecycle_stats["arrival_rate_rps"]
                    sp_rate = spec_stats["arrival_rate_rps"]
                    h3_row["rate_error"] = compute_error(sp_rate, gt_rate)

                    h3_row["request_count_error"] = compute_error(
                        spec_stats["num_requests"], lifecycle_stats["num_requests"]
                    )
                    h3_row["duration_error"] = compute_error(
                        spec_stats["duration_s"], lifecycle_stats["duration_s"]
                    )
            else:
                h3_row["spec_status"] = "no_profile"

            h3_results.append(h3_row)

    # ---------------------------------------------------------------------------
    # Write results
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # H1 summary
    ok_results = [r for r in h1_results if r.get("status") == "ok"]
    h1_output_dir = os.path.join(SCRIPT_DIR, "h1-trace-replay")
    os.makedirs(h1_output_dir, exist_ok=True)

    if ok_results:
        e2e_errors = [r["e2e_error"] for r in ok_results]
        ttft_errors = [r["ttft_error"] for r in ok_results]
        itl_errors = [r["itl_error"] for r in ok_results]

        mean_e2e = sum(e2e_errors) / len(e2e_errors) * 100
        mean_ttft = sum(ttft_errors) / len(ttft_errors) * 100
        mean_itl = sum(itl_errors) / len(itl_errors) * 100
        e2e_under_10 = sum(1 for e in e2e_errors if e < 0.10)

        print(f"\nH1: Trace Replay Results ({len(ok_results)} experiments)")
        print(f"  Mean E2E  error: {mean_e2e:.1f}% (was {R2_WORKLOAD_SPEC_RESULTS['mean_e2e_error_pct']:.1f}% with workload-spec)")
        print(f"  Mean TTFT error: {mean_ttft:.1f}% (was {R2_WORKLOAD_SPEC_RESULTS['mean_ttft_error_pct']:.1f}% with workload-spec)")
        print(f"  Mean ITL  error: {mean_itl:.1f}% (was {R2_WORKLOAD_SPEC_RESULTS['mean_itl_error_pct']:.1f}% with workload-spec)")
        print(f"  E2E < 10%: {e2e_under_10}/{len(ok_results)}")

        # Write CSV
        csv_path = os.path.join(h1_output_dir, "trace_replay_results.csv")
        fieldnames = sorted(set().union(*(r.keys() for r in h1_results)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(h1_results)

        # Write summary JSON
        summary = {
            "n_experiments": len(ok_results),
            "n_failed": len(h1_results) - len(ok_results),
            "mean_e2e_error_pct": mean_e2e,
            "mean_ttft_error_pct": mean_ttft,
            "mean_itl_error_pct": mean_itl,
            "e2e_under_10pct": e2e_under_10,
            "r2_comparison": R2_WORKLOAD_SPEC_RESULTS,
            "e2e_reduction_factor": R2_WORKLOAD_SPEC_RESULTS["mean_e2e_error_pct"] / mean_e2e if mean_e2e > 0 else float("inf"),
            "ttft_reduction_factor": R2_WORKLOAD_SPEC_RESULTS["mean_ttft_error_pct"] / mean_ttft if mean_ttft > 0 else float("inf"),
            "per_experiment": ok_results,
        }
        with open(os.path.join(h1_output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # H2 summary
    h2_output_dir = os.path.join(SCRIPT_DIR, "h2-error-attribution")
    os.makedirs(h2_output_dir, exist_ok=True)
    if h2_results:
        csv_path = os.path.join(h2_output_dir, "error_attribution.csv")
        fieldnames = sorted(set().union(*(r.keys() for r in h2_results)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(h2_results)

        e2e_errors = [r["e2e_error"] for r in h2_results]
        mean_e2e_h2 = sum(e2e_errors) / len(e2e_errors) * 100
        under_10 = sum(1 for e in e2e_errors if e < 0.10)
        under_25 = sum(1 for e in e2e_errors if e < 0.25)

        summary_h2 = {
            "n_experiments": len(h2_results),
            "mean_e2e_error_pct": mean_e2e_h2,
            "e2e_under_10pct": under_10,
            "e2e_under_25pct": under_25,
            "per_experiment": h2_results,
        }
        with open(os.path.join(h2_output_dir, "summary.json"), "w") as f:
            json.dump(summary_h2, f, indent=2)

        print(f"\nH2: Error Attribution ({len(h2_results)} experiments)")
        print(f"  Mean E2E error with trace replay: {mean_e2e_h2:.1f}%")
        print(f"  E2E < 10%: {under_10}/{len(h2_results)}")
        print(f"  E2E < 25%: {under_25}/{len(h2_results)}")

    # H3 summary
    h3_output_dir = os.path.join(SCRIPT_DIR, "h3-workload-diagnosis")
    os.makedirs(h3_output_dir, exist_ok=True)
    if h3_results:
        csv_path = os.path.join(h3_output_dir, "workload_diagnosis.csv")
        fieldnames = sorted(set().union(*(r.keys() for r in h3_results)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(h3_results)

        with open(os.path.join(h3_output_dir, "summary.json"), "w") as f:
            json.dump({"per_experiment": h3_results}, f, indent=2)

        print(f"\nH3: Workload-Spec Diagnosis ({len(h3_results)} experiments)")
        for r in h3_results:
            if "rate_error" in r:
                print(f"  {r['experiment']}: "
                      f"rate_err={r['rate_error']*100:.0f}%, "
                      f"count_err={r.get('request_count_error', 0)*100:.0f}%, "
                      f"dur_err={r.get('duration_error', 0)*100:.0f}%")
            else:
                print(f"  {r['experiment']}: no profile.yaml for comparison")

    print("\nDone! Results written to h1-trace-replay/, h2-error-attribution/, h3-workload-diagnosis/")


if __name__ == "__main__":
    main()
