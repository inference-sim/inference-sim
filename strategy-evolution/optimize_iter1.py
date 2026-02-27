#!/usr/bin/env python3
"""
Bayesian optimizer for Idea 3: SLO-Gated Priority Cascade.

Objective: minimize critical TTFT P99 while constraining:
  - sheddable TTFT P99 < 600ms (< 2.2x baseline)
  - throughput > 15000 tps
  - standard TTFT P99 < 250ms (< baseline)
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

VENV = Path(__file__).parent / ".venv"
if VENV.exists():
    sp = list((VENV / "lib").glob("python*/site-packages"))
    if sp:
        sys.path.insert(0, str(sp[0]))

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

BINARY = "./strategy-evolution/simulation_worker"
WORKLOAD = "strategy-evolution/lib/mixed-production-workload.yaml"
SEEDS = [42, 43, 44]
N_CALLS = 30
N_INITIAL = 10

# Search space (7 parameters)
SPACE = [
    Real(5.0, 20.0, name="base_critical"),
    Real(0.0, 5.0, name="base_sheddable"),
    Real(1e-6, 1e-4, prior="log-uniform", name="age_weight"),
    Integer(50000, 500000, name="threshold_sheddable"),
    Real(0.5, 1.0, name="bias_critical"),
    Real(0.0, 0.5, name="bias_sheddable"),
    Categorical(["1", "2", "3", "4"], name="slo_scorer_weight"),
]

PARAM_NAMES = [d.name for d in SPACE]

def run_blis(params, seed):
    """Run BLIS and parse per-SLO metrics."""
    base_crit, base_shed, age_w, thresh_shed, bias_crit, bias_shed, scorer_w = params

    scorers = f"prefix-affinity:3,slo-priority:{scorer_w},queue-depth:2"

    cmd = [
        BINARY, "run",
        "--model", "meta-llama/llama-3.1-8b-instruct",
        "--num-instances", "8",
        "--routing-policy", "weighted",
        "--routing-scorers", scorers,
        "--scheduler", "priority-fcfs",
        "--priority-policy", "slo-tiered",
        "--slo-priority-bridge",
        "--slo-base-critical", f"{base_crit:.4f}",
        "--slo-base-standard", "5.0",
        "--slo-base-sheddable", f"{base_shed:.4f}",
        "--slo-age-weight", f"{age_w:.10f}",
        "--slo-threshold-standard", "100000",
        "--slo-threshold-sheddable", str(int(thresh_shed)),
        "--slo-scorer-bias-critical", f"{bias_crit:.4f}",
        "--slo-scorer-bias-sheddable", f"{bias_shed:.4f}",
        "--kv-cpu-blocks", "44000",
        "--kv-offload-threshold", "0.9",
        "--long-prefill-token-threshold", "256",
        "--workload-spec", WORKLOAD,
        "--seed", str(seed),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        return parse_output(result.stdout)
    except Exception:
        return None


def parse_output(stdout):
    """Parse BLIS output for cluster metrics and per-SLO TTFT P99."""
    metrics = {}

    # Parse cluster JSON
    for block in stdout.split("{"):
        block = "{" + block
        try:
            end = block.index("}") + 1
            d = json.loads(block[:end])
            if d.get("instance_id") == "cluster":
                metrics["ttft_p99_ms"] = d.get("ttft_p99_ms", 999)
                metrics["e2e_p99_ms"] = d.get("e2e_p99_ms", 999)
                metrics["throughput_tps"] = d.get("tokens_per_sec", 0)
                metrics["completed"] = d.get("completed_requests", 0)
                metrics["dropped"] = d.get("dropped_unservable", 0)
        except (json.JSONDecodeError, ValueError):
            pass

    # Parse per-SLO metrics (format: "TTFT: mean=X p99=Y (n=Z)")
    lines = stdout.split("\n")
    current_slo = None
    for line in lines:
        line = line.strip()
        if line.endswith(":") and line.rstrip(":") in ("critical", "standard", "sheddable"):
            current_slo = line.rstrip(":")
        elif current_slo and "TTFT:" in line:
            m = re.search(r"p99=([\d.]+)", line)
            if m:
                metrics[f"{current_slo}_ttft_p99_us"] = float(m.group(1))
            current_slo = None

    return metrics if "ttft_p99_ms" in metrics else None


def objective(params):
    """Multi-objective: minimize critical TTFT P99 with constraints."""
    all_crit = []
    all_std = []
    all_shed = []
    all_tps = []

    for seed in SEEDS:
        m = run_blis(params, seed)
        if m is None:
            return 1e6  # penalty for failed run

        crit = m.get("critical_ttft_p99_us", 999999) / 1000  # to ms
        std = m.get("standard_ttft_p99_us", 999999) / 1000
        shed = m.get("sheddable_ttft_p99_us", 999999) / 1000
        tps = m.get("throughput_tps", 0)

        all_crit.append(crit)
        all_std.append(std)
        all_shed.append(shed)
        all_tps.append(tps)

    mean_crit = np.mean(all_crit)
    mean_std = np.mean(all_std)
    mean_shed = np.mean(all_shed)
    mean_tps = np.mean(all_tps)

    # Primary: minimize critical TTFT P99
    score = mean_crit

    # Penalty: sheddable too high (> 600ms)
    if mean_shed > 600:
        score += (mean_shed - 600) * 2

    # Penalty: throughput too low (< 15000 tps)
    if mean_tps < 15000:
        score += (15000 - mean_tps) * 0.01

    # Penalty: standard worse than baseline (> 270ms)
    if mean_std > 270:
        score += (mean_std - 270) * 5

    print(
        f"  crit={mean_crit:.1f}ms std={mean_std:.1f}ms shed={mean_shed:.1f}ms "
        f"tps={mean_tps:.0f} → score={score:.1f}",
        file=sys.stderr, flush=True,
    )
    return score


def main():
    print(f"Optimizing Idea 3 with {N_CALLS} calls ({N_INITIAL} initial)...", file=sys.stderr)
    print(f"Seeds: {SEEDS}", file=sys.stderr)
    print(f"Parameters: {PARAM_NAMES}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    call_count = [0]

    def tracked_objective(params):
        call_count[0] += 1
        pdict = dict(zip(PARAM_NAMES, params))
        print(f"\n[{call_count[0]}/{N_CALLS}] {pdict}", file=sys.stderr, flush=True)
        return objective(params)

    result = gp_minimize(
        tracked_objective,
        SPACE,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL,
        random_state=42,
        verbose=False,
    )

    best = dict(zip(PARAM_NAMES, result.x))
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"BEST PARAMETERS: {best}", file=sys.stderr)
    print(f"BEST SCORE: {result.fun:.2f}", file=sys.stderr)

    # Final detailed evaluation
    print("\n=== Final evaluation with best parameters ===", file=sys.stderr)
    for seed in SEEDS:
        m = run_blis(result.x, seed)
        if m:
            crit = m.get("critical_ttft_p99_us", 0) / 1000
            std = m.get("standard_ttft_p99_us", 0) / 1000
            shed = m.get("sheddable_ttft_p99_us", 0) / 1000
            tps = m.get("throughput_tps", 0)
            print(f"  Seed {seed}: crit={crit:.1f}ms std={std:.1f}ms shed={shed:.1f}ms tps={tps:.0f}", file=sys.stderr)

    output = {
        "strategy": "slo-gated-priority-cascade",
        "best_parameters": {k: (v if not isinstance(v, np.integer) else int(v)) for k, v in best.items()},
        "best_score": float(result.fun),
        "convergence": [float(x) for x in result.func_vals],
        "n_calls": N_CALLS,
        "seeds": SEEDS,
    }
    with open("strategy-evolution/iter1-optimization-results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to strategy-evolution/iter1-optimization-results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
