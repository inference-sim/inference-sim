"""BLIS validation harness for StepML research.

Runs BLIS on each ground-truth experiment using the *same* vLLM args and
inference-perf profile, then compares BLIS-predicted latencies against
observed values.  Produces a per-experiment summary CSV.

All experiment-specific parameters come from the experiment directory itself:
  - exp-config.yaml   → model, tensor_parallelism, max_model_len,
                         max_num_batched_tokens, max_num_seqs
  - profile.yaml      → inference-perf stages + shared_prefix config
  - vllm.log          → GPU KV cache size (tokens → blocks)
  - results/summary_lifecycle_metrics.json → ground-truth latencies

Latency coefficients are passed explicitly (--alpha-coeffs / --beta-coeffs)
or via a StepML model artifact (--stepml-model).

Usage:
    python validate_blis.py --alpha-coeffs 0,0,0 --beta-coeffs 0,0,0
    python validate_blis.py --stepml-model path/to/model.json
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "eval", "ground_truth")
DEFAULT_BINARY = os.path.join(_REPO_ROOT, "simulation_worker")

BLOCK_SIZE_TOKENS = 16


# ---------------------------------------------------------------------------
# Experiment parsing helpers
# ---------------------------------------------------------------------------
def parse_experiment_dir(dirname: str) -> dict:
    """Parse experiment metadata from directory name.

    Expected format: <timestamp>-<model>-tp<N>-<workload>
    """
    pattern = re.compile(r"^(\d{8}-\d{6})-(.+)-tp(\d+)-(\w+)$")
    m = pattern.match(dirname)
    if m:
        return {
            "timestamp": m.group(1),
            "model": m.group(2),
            "tp": int(m.group(3)),
            "workload": m.group(4),
        }
    # Fallback: find last -tp<N>-
    tp_matches = list(re.finditer(r"-tp(\d+)-", dirname))
    if not tp_matches:
        raise ValueError(f"Cannot parse experiment dir: {dirname}")
    last_tp = tp_matches[-1]
    return {
        "timestamp": dirname[:15],
        "model": dirname[16:last_tp.start()],
        "tp": int(last_tp.group(1)),
        "workload": dirname[last_tp.end():],
    }


def load_ground_truth_metrics(experiment_dir: str) -> dict:
    """Load summary latency metrics from ground-truth experiment."""
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
        "throughput_rps": successes.get("throughput", {}).get(
            "requests_per_sec", 0
        ),
        "num_requests": data.get("load_summary", {}).get("count", 0),
        "prompt_len_mean": successes.get("prompt_len", {}).get("mean", 0),
        "output_len_mean": successes.get("output_len", {}).get("mean", 0),
    }


def load_exp_config(experiment_dir: str) -> dict:
    """Load exp-config.yaml from experiment directory."""
    config_path = os.path.join(experiment_dir, "exp-config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_profile(experiment_dir: str) -> dict:
    """Load profile.yaml (inference-perf config) from experiment directory.

    profile.yaml is JSON despite the .yaml extension.
    """
    profile_path = os.path.join(experiment_dir, "profile.yaml")
    with open(profile_path) as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return yaml.safe_load(content)


def extract_kv_blocks_from_vllm_log(experiment_dir: str) -> int | None:
    """Parse vllm.log for 'GPU KV cache size: N tokens' and convert to blocks.

    Returns total KV blocks (tokens / BLOCK_SIZE_TOKENS), or None if not found.
    """
    vllm_log_path = os.path.join(experiment_dir, "vllm.log")
    if not os.path.isfile(vllm_log_path):
        return None

    pattern = re.compile(r"GPU KV cache size:\s+([\d,]+)\s+tokens")
    with open(vllm_log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                tokens = int(m.group(1).replace(",", ""))
                return tokens // BLOCK_SIZE_TOKENS
    return None


def extract_cpu_kv_blocks_from_vllm_log(experiment_dir: str) -> int:
    """Parse vllm.log for CPU offloading config and KV cache shape.

    Computes CPU KV blocks from:
      - cpu_bytes_to_use from kv_connector_extra_config
      - per-block size derived from the KV cache shape
        shape = (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim)
        per_block_bytes = num_layers * 2 * block_size * num_kv_heads * head_dim * 2 (fp16)

    Returns 0 if no offloading config found.
    """
    vllm_log_path = os.path.join(experiment_dir, "vllm.log")
    if not os.path.isfile(vllm_log_path):
        return 0

    cpu_bytes = None
    kv_shape = None
    cpu_pattern = re.compile(r"cpu_bytes_to_use['\"]?:\s*([\d.]+)")
    shape_pattern = re.compile(
        r"cross layer KV cache of shape \((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
    )

    with open(vllm_log_path) as f:
        for line in f:
            if cpu_bytes is None:
                m = cpu_pattern.search(line)
                if m:
                    cpu_bytes = float(m.group(1))
            if kv_shape is None:
                m = shape_pattern.search(line)
                if m:
                    kv_shape = tuple(int(m.group(i)) for i in range(1, 7))
            if cpu_bytes is not None and kv_shape is not None:
                break

    if cpu_bytes is None or kv_shape is None or cpu_bytes == 0:
        return 0

    # shape = (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim)
    _, num_layers, _, block_size, num_kv_heads, head_dim = kv_shape
    per_block_bytes = num_layers * 2 * block_size * num_kv_heads * head_dim * 2  # fp16
    return int(cpu_bytes) // per_block_bytes


# ---------------------------------------------------------------------------
# Workload spec construction
# ---------------------------------------------------------------------------
def build_workload_spec(profile: dict, ground_truth: dict) -> dict:
    """Build a BLIS workload-spec YAML from inference-perf profile.

    Extracts shared prefix parameters and stage-based load from the profile,
    then constructs a WorkloadSpec with inference_perf section.
    """
    data = profile.get("data", {})
    load_config = profile.get("load", {})
    sp = data.get("shared_prefix", {})

    stages = []
    for stage in load_config.get("stages", []):
        stages.append({
            "rate": stage["rate"],
            "duration": stage["duration"],
        })

    # Fallback: synthesize a stage from ground-truth throughput
    if not stages:
        num_requests = ground_truth.get("num_requests", 1000)
        rps = ground_truth.get("throughput_rps", 10)
        duration = int(num_requests / rps) + 60
        stages.append({"rate": rps, "duration": duration})

    total_requests = sum(s["rate"] * s["duration"] for s in stages)
    total_duration_s = sum(s["duration"] for s in stages)
    horizon_us = int(total_duration_s * 1_000_000) + 60_000_000  # +60s buffer

    # Compute time-weighted average rate (needed by BLIS validation before
    # InferencePerf expansion fills it in GenerateRequests).
    aggregate_rate = sum(
        s["rate"] * s["duration"] for s in stages
    ) / total_duration_s

    return {
        "version": "2",
        "seed": 42,
        "num_requests": int(total_requests),
        "horizon": horizon_us,
        "aggregate_rate": aggregate_rate,
        "inference_perf": {
            "stages": stages,
            "shared_prefix": {
                "num_unique_system_prompts": sp.get(
                    "num_unique_system_prompts", 9
                ),
                "num_users_per_system_prompt": sp.get(
                    "num_users_per_system_prompt", 5
                ),
                "system_prompt_len": sp.get("system_prompt_len", 100),
                "question_len": sp.get(
                    "question_len",
                    int(ground_truth.get("prompt_len_mean", 500)),
                ),
                "output_len": sp.get(
                    "output_len",
                    int(ground_truth.get("output_len_mean", 250)),
                ),
                "enable_multi_turn_chat": sp.get(
                    "enable_multi_turn_chat", False
                ),
            },
        },
    }


# ---------------------------------------------------------------------------
# BLIS execution
# ---------------------------------------------------------------------------
def run_blis(
    binary: str,
    workload_spec_path: str,
    exp_config: dict,
    total_kv_blocks: int,
    alpha_coeffs: list[float],
    beta_coeffs: list[float],
    results_path: str,
    stepml_model_path: str | None = None,
    roofline: bool = False,
    hardware: str | None = None,
    cpu_kv_blocks: int = 0,
) -> dict | None:
    """Run BLIS with the same vLLM args as the ground-truth experiment.

    When roofline=True, uses the analytical roofline latency model instead of
    alpha/beta coefficients.  Requires hardware (GPU type, e.g. "H100").

    Returns dict with e2e_mean_ms, ttft_mean_ms, itl_mean_ms, or None on failure.
    """
    model_name = exp_config.get("model", "unknown")
    tp = exp_config.get("tensor_parallelism", 1)
    max_model_len = exp_config.get("max_model_len", 4096)
    max_num_seqs = exp_config.get("max_num_seqs", 128)
    max_num_batched_tokens = exp_config.get("max_num_batched_tokens", 2048)

    cmd = [
        binary,
        "run",
        "--model", model_name,
        "--workload-spec", workload_spec_path,
        # --- Match vLLM serving args exactly ---
        "--tp", str(tp),
        "--max-model-len", str(max_model_len),
        "--max-num-running-reqs", str(max_num_seqs),
        "--max-num-scheduled-tokens", str(max_num_batched_tokens),
        "--total-kv-blocks", str(total_kv_blocks),
        "--block-size-in-tokens", str(BLOCK_SIZE_TOKENS),
    ]

    if cpu_kv_blocks > 0:
        cmd.extend(["--kv-cpu-blocks", str(cpu_kv_blocks)])

    if roofline:
        if not hardware:
            raise ValueError("roofline=True requires hardware (GPU type)")
        cmd.extend(["--roofline", "--hardware", hardware])
    else:
        # --- Latency coefficients (use = to handle negative values) ---
        cmd.append("--alpha-coeffs=" + ",".join(str(c) for c in alpha_coeffs))
        cmd.append("--beta-coeffs=" + ",".join(str(c) for c in beta_coeffs))

    # --- Output ---
    cmd.extend(["--results-path", results_path, "--log", "error"])

    if stepml_model_path:
        cmd.extend(["--stepml-model", stepml_model_path])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.dirname(binary),
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 600s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"  Binary not found: {binary}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(
            f"  BLIS failed (exit {result.returncode}): {result.stderr[:500]}",
            file=sys.stderr,
        )
        return None

    return parse_blis_stdout(result.stdout)


def parse_blis_stdout(stdout: str) -> dict | None:
    """Parse BLIS stdout for the JSON block after '=== Simulation Metrics ==='."""
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


def compute_error(predicted: float, observed: float) -> float:
    """Compute relative error: |predicted - observed| / observed."""
    if observed == 0:
        return float("inf") if predicted != 0 else 0.0
    return abs(predicted - observed) / observed


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------
def validate_all_experiments(
    binary: str,
    data_root: str,
    output_path: str,
    alpha_coeffs: list[float],
    beta_coeffs: list[float],
    stepml_model_path: str | None = None,
    roofline: bool = False,
    hardware: str | None = None,
):
    """Run BLIS validation across all ground-truth experiments."""
    results = []

    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue

        summary_path = os.path.join(
            dirpath, "results", "summary_lifecycle_metrics.json"
        )
        if not os.path.isfile(summary_path):
            continue

        print(f"\n{'='*60}")
        print(f"Experiment: {dirname}")
        print(f"{'='*60}")

        # Parse metadata from directory name
        try:
            meta = parse_experiment_dir(dirname)
        except ValueError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            continue

        # Load ground truth metrics
        gt = load_ground_truth_metrics(dirpath)
        print(f"  Ground truth: E2E={gt['e2e_mean_s']*1000:.1f}ms, "
              f"TTFT={gt['ttft_mean_s']*1000:.1f}ms, "
              f"ITL={gt['itl_mean_s']*1000:.2f}ms")

        # Load experiment config (vLLM serving args)
        exp_config = load_exp_config(dirpath)

        # Extract KV cache blocks from vllm.log
        total_kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
        if total_kv_blocks is None:
            print("  SKIP: Cannot parse KV blocks from vllm.log",
                  file=sys.stderr)
            results.append({
                "experiment": dirname,
                "model": meta["model"],
                "workload": meta["workload"],
                "tp": meta["tp"],
                "status": "no_kv_blocks",
            })
            continue
        cpu_kv_blocks = extract_cpu_kv_blocks_from_vllm_log(dirpath)
        print(f"  KV blocks: {total_kv_blocks} GPU "
              f"({total_kv_blocks * BLOCK_SIZE_TOKENS:,} tokens)"
              f" + {cpu_kv_blocks} CPU")

        # Load inference-perf profile and build workload spec
        try:
            profile = load_profile(dirpath)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            print(f"  SKIP: Cannot load profile: {e}", file=sys.stderr)
            results.append({
                "experiment": dirname,
                "model": meta["model"],
                "workload": meta["workload"],
                "tp": meta["tp"],
                "status": "no_profile",
            })
            continue

        workload_spec = build_workload_spec(profile, gt)

        # Write workload spec and results path to temp files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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
                alpha_coeffs=alpha_coeffs,
                beta_coeffs=beta_coeffs,
                results_path=results_json_path,
                stepml_model_path=stepml_model_path,
                roofline=roofline,
                hardware=hardware,
                cpu_kv_blocks=cpu_kv_blocks,
            )
        finally:
            os.unlink(spec_path)
            if os.path.exists(results_json_path):
                os.unlink(results_json_path)

        if blis_metrics is None:
            results.append({
                "experiment": dirname,
                "model": meta["model"],
                "workload": meta["workload"],
                "tp": meta["tp"],
                "status": "blis_failed",
            })
            continue

        # Compute errors (ground truth is in seconds, BLIS outputs ms)
        e2e_error = compute_error(
            blis_metrics["e2e_mean_ms"], gt["e2e_mean_s"] * 1000
        )
        ttft_error = compute_error(
            blis_metrics["ttft_mean_ms"], gt["ttft_mean_s"] * 1000
        )
        itl_error = compute_error(
            blis_metrics["itl_mean_ms"], gt["itl_mean_s"] * 1000
        )

        print(f"  BLIS:         E2E={blis_metrics['e2e_mean_ms']:.1f}ms, "
              f"TTFT={blis_metrics['ttft_mean_ms']:.1f}ms, "
              f"ITL={blis_metrics['itl_mean_ms']:.2f}ms")
        print(f"  Errors:       E2E={e2e_error*100:.1f}%, "
              f"TTFT={ttft_error*100:.1f}%, "
              f"ITL={itl_error*100:.1f}%")

        results.append({
            "experiment": dirname,
            "model": meta["model"],
            "workload": meta["workload"],
            "tp": meta["tp"],
            "status": "ok",
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
            "gt_ttft_ms": gt["ttft_mean_s"] * 1000,
            "gt_itl_ms": gt["itl_mean_s"] * 1000,
            "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
            "blis_ttft_ms": blis_metrics["ttft_mean_ms"],
            "blis_itl_ms": blis_metrics["itl_mean_ms"],
            "e2e_error": e2e_error,
            "ttft_error": ttft_error,
            "itl_error": itl_error,
            "blis_completed": blis_metrics["completed_requests"],
            "gt_requests": gt["num_requests"],
        })

    # Write summary CSV
    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'='*60}")
        print(f"Summary written to: {output_path}")
        print(f"{'='*60}")

        ok_results = [r for r in results if r.get("status") == "ok"]
        if ok_results:
            e2e_errors = [r["e2e_error"] for r in ok_results]
            ttft_errors = [r["ttft_error"] for r in ok_results]
            itl_errors = [r["itl_error"] for r in ok_results]

            print(f"\nExperiments: {len(ok_results)} completed, "
                  f"{len(results) - len(ok_results)} skipped/failed")
            print(f"E2E  mean error: "
                  f"{sum(e2e_errors)/len(e2e_errors)*100:.1f}%")
            print(f"TTFT mean error: "
                  f"{sum(ttft_errors)/len(ttft_errors)*100:.1f}%")
            print(f"ITL  mean error: "
                  f"{sum(itl_errors)/len(itl_errors)*100:.1f}%")

            passing = sum(1 for e in e2e_errors if e < 0.10)
            print(f"E2E < 10%: {passing}/{len(ok_results)} experiments")


def main():
    parser = argparse.ArgumentParser(
        description="BLIS validation harness for StepML research"
    )
    parser.add_argument(
        "--binary",
        default=DEFAULT_BINARY,
        help=f"Path to simulation_worker binary (default: {DEFAULT_BINARY})",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Path to eval/ground_truth/ (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_SCRIPT_DIR, "blis_validation_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--alpha-coeffs",
        default="0,0,0",
        help="Comma-separated alpha coefficients (default: 0,0,0)",
    )
    parser.add_argument(
        "--beta-coeffs",
        default="0,0,0",
        help="Comma-separated beta coefficients (default: 0,0,0)",
    )
    parser.add_argument(
        "--stepml-model",
        default=None,
        help="Path to StepML model artifact JSON (optional)",
    )
    parser.add_argument(
        "--roofline",
        action="store_true",
        help="Use analytical roofline latency model instead of alpha/beta coefficients",
    )
    parser.add_argument(
        "--hardware",
        default=None,
        help="GPU type for roofline mode (e.g. H100). Required when --roofline is set.",
    )
    args = parser.parse_args()

    if args.roofline and not args.hardware:
        parser.error("--roofline requires --hardware (GPU type, e.g. H100)")

    alpha_coeffs = [float(x) for x in args.alpha_coeffs.split(",")]
    beta_coeffs = [float(x) for x in args.beta_coeffs.split(",")]

    if not args.roofline:
        if len(alpha_coeffs) < 3:
            parser.error("--alpha-coeffs requires at least 3 values")
        if len(beta_coeffs) < 3:
            parser.error("--beta-coeffs requires at least 3 values")

    # Build binary if needed
    if not os.path.isfile(args.binary):
        print("Building simulation_worker...", file=sys.stderr)
        build_result = subprocess.run(
            ["go", "build", "-o", args.binary, "main.go"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if build_result.returncode != 0:
            print(f"Build failed: {build_result.stderr}", file=sys.stderr)
            sys.exit(1)

    validate_all_experiments(
        binary=args.binary,
        data_root=args.data_root,
        output_path=args.output,
        alpha_coeffs=alpha_coeffs,
        beta_coeffs=beta_coeffs,
        stepml_model_path=args.stepml_model,
        roofline=args.roofline,
        hardware=args.hardware,
    )


if __name__ == "__main__":
    main()
