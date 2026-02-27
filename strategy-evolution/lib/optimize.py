#!/usr/bin/env python3
"""
Bayesian Optimization Harness for BLIS Strategy Parameter Tuning.

Usage:
    python3 optimize.py --strategy-config config.yaml --binary ./simulation_worker \
        --seeds 42,43,44 --n-calls 50 --output results.json

Each strategy defines a config.yaml with:
  - fixed_args: CLI flags that don't change (e.g., --routing-policy weighted)
  - parameters: list of {name, flag, type, low, high, [prior]} for tunable params
  - workload: workload spec YAML path or inline
  - objective: {metric, direction} for primary optimization target
  - constraints: [{metric, direction, threshold}] for secondary objectives
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Add venv site-packages
VENV_PATH = Path(__file__).parent.parent / ".venv"
if VENV_PATH.exists():
    site_packages = list((VENV_PATH / "lib").glob("python*/site-packages"))
    if site_packages:
        sys.path.insert(0, str(site_packages[0]))

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical


def load_config(path):
    """Load strategy configuration YAML."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_search_space(parameters):
    """Convert parameter definitions to skopt search space."""
    dimensions = []
    names = []
    for p in parameters:
        names.append(p["name"])
        if p["type"] == "real":
            dim = Real(p["low"], p["high"], prior=p.get("prior", "uniform"), name=p["name"])
        elif p["type"] == "integer":
            dim = Integer(p["low"], p["high"], name=p["name"])
        elif p["type"] == "categorical":
            dim = Categorical(p["categories"], name=p["name"])
        else:
            raise ValueError(f"Unknown parameter type: {p['type']}")
        dimensions.append(dim)
    return dimensions, names


def build_cli_args(config, param_values, param_names):
    """Build BLIS CLI arguments from fixed args + tuned parameters."""
    args = list(config.get("fixed_args", []))
    for name, value in zip(param_names, param_values):
        param_def = next(p for p in config["parameters"] if p["name"] == name)
        flag = param_def["flag"]

        if param_def["type"] == "real":
            args.extend([flag, f"{value:.6f}"])
        elif param_def["type"] == "integer":
            args.extend([flag, str(int(value))])
        elif param_def["type"] == "categorical":
            args.extend([flag, str(value)])

    # Add workload spec if configured
    if "workload_spec" in config:
        args.extend(["--workload-spec", config["workload_spec"]])

    return args


def run_blis(binary, cli_args, seed, timeout=300):
    """Run BLIS with given arguments and seed, return parsed metrics."""
    cmd = [binary, "run"] + cli_args + ["--seed", str(seed)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None, f"Exit code {result.returncode}: {result.stderr[:200]}"
        return parse_output(result.stdout), None
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def parse_output(stdout):
    """Parse BLIS JSON output into metrics dict."""
    metrics = {}
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("{"):
            try:
                data = json.loads(line if line.startswith("{") else stdout)
                # Extract key metrics
                metrics["ttft_p99_ms"] = data.get("ttft_p99_ms", float("inf"))
                metrics["ttft_mean_ms"] = data.get("ttft_mean_ms", float("inf"))
                metrics["e2e_p99_ms"] = data.get("e2e_p99_ms", float("inf"))
                metrics["e2e_mean_ms"] = data.get("e2e_mean_ms", float("inf"))
                metrics["tpot_p99_ms"] = data.get("tpot_p99_ms", float("inf"))
                metrics["throughput_tps"] = data.get("throughput_tokens_per_second", 0)
                metrics["completed"] = data.get("completed_requests", 0)
                metrics["dropped"] = data.get("dropped_unservable", 0)
                metrics["preemptions"] = data.get("preemption_count", 0)
                return metrics
            except json.JSONDecodeError:
                continue
    return metrics if metrics else None


def compute_objective(metrics, config, direction="minimize"):
    """Compute scalar objective from metrics.

    Multi-objective scalarization:
      score = sum(weight_i * normalized_metric_i)

    Lower is better (for minimization).
    """
    if metrics is None:
        return 1e10  # Penalty for failed runs

    obj = config.get("objective", {})
    constraints = config.get("constraints", [])

    # Primary objective
    primary_metric = obj.get("metric", "ttft_p99_ms")
    primary_value = metrics.get(primary_metric, float("inf"))

    # Check constraints — add penalties for violations
    penalty = 0.0
    for c in constraints:
        metric_val = metrics.get(c["metric"], float("inf"))
        threshold = c["threshold"]
        if c["direction"] == "max" and metric_val < threshold:
            penalty += (threshold - metric_val) * c.get("weight", 100)
        elif c["direction"] == "min" and metric_val > threshold:
            penalty += (metric_val - threshold) * c.get("weight", 100)

    # Throughput: invert if maximizing (since gp_minimize minimizes)
    if obj.get("direction") == "max":
        return -primary_value + penalty
    return primary_value + penalty


def evaluate(param_values, binary, config, param_names, seeds):
    """Evaluate a parameter setting across multiple seeds."""
    cli_args = build_cli_args(config, param_values, param_names)
    scores = []
    all_metrics = []

    for seed in seeds:
        metrics, err = run_blis(binary, cli_args, seed)
        if err:
            print(f"  [seed={seed}] ERROR: {err}", file=sys.stderr)
            scores.append(1e10)
        else:
            score = compute_objective(metrics, config)
            scores.append(score)
            all_metrics.append(metrics)
            m = metrics
            print(
                f"  [seed={seed}] TTFT_P99={m.get('ttft_p99_ms', '?'):.1f}ms "
                f"E2E_P99={m.get('e2e_p99_ms', '?'):.1f}ms "
                f"TPS={m.get('throughput_tps', '?'):.1f} "
                f"score={score:.2f}",
                file=sys.stderr
            )

    # Return mean score across seeds (robust to outliers)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description="BLIS Strategy Parameter Optimizer")
    parser.add_argument("--strategy-config", required=True, help="Strategy config YAML")
    parser.add_argument("--binary", default="./simulation_worker", help="BLIS binary path")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    parser.add_argument("--n-calls", type=int, default=50, help="Number of optimization iterations")
    parser.add_argument("--n-initial", type=int, default=10, help="Initial random exploration points")
    parser.add_argument("--output", default="optimization_results.json", help="Output JSON path")
    parser.add_argument("--baseline", action="store_true", help="Run baseline only (no optimization)")
    args = parser.parse_args()

    config = load_config(args.strategy_config)
    seeds = [int(s) for s in args.seeds.split(",")]
    dimensions, param_names = build_search_space(config["parameters"])

    print(f"Strategy: {config.get('name', 'unnamed')}", file=sys.stderr)
    print(f"Parameters: {param_names}", file=sys.stderr)
    print(f"Seeds: {seeds}", file=sys.stderr)
    print(f"Optimization budget: {args.n_calls} calls ({args.n_initial} initial)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    call_count = [0]

    def objective(param_values):
        call_count[0] += 1
        params_dict = dict(zip(param_names, param_values))
        print(f"\n[{call_count[0]}/{args.n_calls}] Trying: {params_dict}", file=sys.stderr)
        return evaluate(param_values, args.binary, config, param_names, seeds)

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        dimensions,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        random_state=42,
        verbose=False,
    )

    # Extract best parameters
    best_params = dict(zip(param_names, result.x))
    best_score = result.fun

    # Run best params again to get detailed metrics
    cli_args = build_cli_args(config, result.x, param_names)
    best_metrics = {}
    for seed in seeds:
        metrics, err = run_blis(args.binary, cli_args, seed)
        if metrics:
            for k, v in metrics.items():
                best_metrics.setdefault(k, []).append(v)

    # Average metrics across seeds
    avg_metrics = {k: float(np.mean(v)) for k, v in best_metrics.items()}

    output = {
        "strategy": config.get("name", "unnamed"),
        "best_parameters": best_params,
        "best_score": best_score,
        "best_metrics": avg_metrics,
        "cli_args": cli_args,
        "n_calls": args.n_calls,
        "seeds": seeds,
        "convergence": [float(x) for x in result.func_vals],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"BEST: {best_params}", file=sys.stderr)
    print(f"Score: {best_score:.4f}", file=sys.stderr)
    print(f"Metrics: {avg_metrics}", file=sys.stderr)
    print(f"Results saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
