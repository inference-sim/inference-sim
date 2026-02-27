#!/usr/bin/env python3
"""
Iteration 8 Bayesian optimizer: Full combined strategy.

Parameters:
  - base_critical, base_sheddable (priority gap)
  - age_weight (urgency escalation)
  - threshold_sheddable (grace period)
  - max_scheduled_tokens (token budget)
  - slo_prefill_critical (0=no-chunk, or chunking threshold)

Objective: minimize critical TTFT P99, constrain sheddable < 600ms and TPS > 15k.
"""

import json
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
from skopt.space import Real, Integer

BINARY = "./strategy-evolution/simulation_worker"
WORKLOAD = "strategy-evolution/lib/mixed-production-workload.yaml"
SEEDS = [42, 43, 44]
N_CALLS = 40
N_INITIAL = 12

SPACE = [
    Real(5.0, 20.0, name="base_critical"),
    Real(0.0, 5.0, name="base_sheddable"),
    Real(1e-6, 1e-4, prior="log-uniform", name="age_weight"),
    Integer(50000, 500000, name="threshold_sheddable"),
    Integer(768, 4096, name="max_tokens"),
    Integer(0, 512, name="prefill_crit"),  # 0 = no chunking
]

PARAM_NAMES = [d.name for d in SPACE]


def run_blis(params, seed):
    base_crit, base_shed, age_w, thresh_shed, max_tok, prefill_c = params
    prefill_s = 0 if prefill_c == 0 else 256  # sheddable: match critical or use 256

    cmd = [
        BINARY, "run",
        "--model", "meta-llama/llama-3.1-8b-instruct",
        "--num-instances", "8",
        "--routing-policy", "weighted",
        "--routing-scorers", "prefix-affinity:3,queue-depth:2,kv-utilization:2",
        "--scheduler", "priority-fcfs",
        "--priority-policy", "slo-tiered",
        "--slo-priority-bridge",
        "--slo-base-critical", f"{base_crit:.4f}",
        "--slo-base-standard", "5.0",
        "--slo-base-sheddable", f"{base_shed:.4f}",
        "--slo-age-weight", f"{age_w:.10f}",
        "--slo-threshold-standard", "100000",
        "--slo-threshold-sheddable", str(int(thresh_shed)),
        "--max-num-scheduled-tokens", str(int(max_tok)),
        "--slo-prefill",
        "--slo-prefill-critical", str(int(prefill_c)),
        "--slo-prefill-sheddable", str(int(prefill_s)),
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
    metrics = {}
    for block in stdout.split("{"):
        block = "{" + block
        try:
            end = block.index("}") + 1
            d = json.loads(block[:end])
            if d.get("instance_id") == "cluster":
                metrics["ttft_p99_ms"] = d.get("ttft_p99_ms", 999)
                metrics["throughput_tps"] = d.get("tokens_per_sec", 0)
                metrics["completed"] = d.get("completed_requests", 0)
        except (json.JSONDecodeError, ValueError):
            pass

    current_slo = None
    for line in stdout.split("\n"):
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
    all_crit, all_shed, all_tps = [], [], []

    for seed in SEEDS:
        m = run_blis(params, seed)
        if m is None:
            return 1e6

        crit = m.get("critical_ttft_p99_us", 999999) / 1000
        shed = m.get("sheddable_ttft_p99_us", 999999) / 1000
        tps = m.get("throughput_tps", 0)
        all_crit.append(crit)
        all_shed.append(shed)
        all_tps.append(tps)

    mc, ms, mt = np.mean(all_crit), np.mean(all_shed), np.mean(all_tps)

    # Primary: minimize critical P99
    score = mc
    # Constraint: sheddable < 600ms
    if ms > 600:
        score += (ms - 600) * 2
    # Constraint: TPS > 15000
    if mt < 15000:
        score += (15000 - mt) * 0.01
    # Bonus: standard improvement
    std_mean = sum(
        (run_blis(params, s) or {}).get("standard_ttft_p99_us", 300000) / 1000
        for s in SEEDS
    ) / len(SEEDS)
    if std_mean > 250:
        score += (std_mean - 250) * 0.5

    print(
        f"  crit={mc:.1f} shed={ms:.1f} tps={mt:.0f} std~{std_mean:.0f} -> {score:.1f}",
        file=sys.stderr, flush=True,
    )
    return score


def main():
    print(f"Optimizing full strategy: {N_CALLS} calls, {SEEDS}", file=sys.stderr)

    n = [0]
    def tracked(params):
        n[0] += 1
        pdict = dict(zip(PARAM_NAMES, params))
        print(f"\n[{n[0]}/{N_CALLS}] {pdict}", file=sys.stderr, flush=True)
        return objective(params)

    result = gp_minimize(tracked, SPACE, n_calls=N_CALLS, n_initial_points=N_INITIAL, random_state=42)

    best = dict(zip(PARAM_NAMES, result.x))
    print(f"\nBEST: {best}", file=sys.stderr)
    print(f"SCORE: {result.fun:.2f}", file=sys.stderr)

    # Final eval
    print("\n=== Final evaluation ===", file=sys.stderr)
    for seed in SEEDS:
        m = run_blis(result.x, seed)
        if m:
            c = m.get("critical_ttft_p99_us", 0) / 1000
            s = m.get("standard_ttft_p99_us", 0) / 1000
            sh = m.get("sheddable_ttft_p99_us", 0) / 1000
            t = m.get("throughput_tps", 0)
            print(f"  Seed {seed}: crit={c:.1f} std={s:.1f} shed={sh:.1f} tps={t:.0f}", file=sys.stderr)

    with open("strategy-evolution/iter8-optimization-results.json", "w") as f:
        json.dump({
            "best_parameters": {k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in best.items()},
            "best_score": float(result.fun),
            "convergence": [float(x) for x in result.func_vals],
        }, f, indent=2)
    print(f"\nSaved to strategy-evolution/iter8-optimization-results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
