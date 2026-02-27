#!/usr/bin/env python3
"""Bayesian-style parameter optimization for the compound strategy.

Uses scipy.optimize.differential_evolution (gradient-free global optimizer)
since scikit-optimize may not be installed. Evaluates each parameter config
by running BLIS with 3 seeds and taking the mean critical TTFT P99.

Parameters optimized:
1. sheddable_queue_threshold (1-15): max QueueDepth for sheddable admission
2. standard_queue_threshold (5-25): max QueueDepth for standard admission
3. pa_weight (1-5): prefix-affinity scorer weight
4. qd_weight (1-5): queue-depth scorer weight

Fixed: scheduler=priority-fcfs, priority=slo-class, admission=slo-gated
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
BINARY = REPO / "simulation_worker"
RESULTS_DIR = Path(__file__).parent / "bayesian_results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = "meta-llama/llama-3.1-8b-instruct"
SEEDS = [42, 123, 7777]
NUM_INSTANCES = 8
NUM_REQUESTS = 500
HORIZON = 300000000
RATE = 2000

call_count = 0

def make_workload(seed):
    """Create orthogonal mixed-SLO workload YAML."""
    f = RESULTS_DIR / f"workload_{seed}.yaml"
    f.write_text(f"""version: "2"
seed: {seed}
category: language
aggregate_rate: {RATE}
clients:
  - id: critical
    slo_class: critical
    rate_fraction: 0.33
    prefix_group: shared-prompt
    prefix_length: 512
    arrival: {{process: poisson}}
    input_distribution: {{type: gaussian, params: {{mean: 256, std_dev: 100, min: 64, max: 512}}}}
    output_distribution: {{type: exponential, params: {{mean: 128, min: 16, max: 1024}}}}
  - id: standard
    slo_class: standard
    rate_fraction: 0.34
    prefix_group: shared-prompt
    prefix_length: 512
    arrival: {{process: poisson}}
    input_distribution: {{type: gaussian, params: {{mean: 256, std_dev: 100, min: 64, max: 512}}}}
    output_distribution: {{type: exponential, params: {{mean: 128, min: 16, max: 1024}}}}
  - id: sheddable
    slo_class: sheddable
    rate_fraction: 0.33
    prefix_group: shared-prompt
    prefix_length: 512
    arrival: {{process: poisson}}
    input_distribution: {{type: gaussian, params: {{mean: 256, std_dev: 100, min: 64, max: 512}}}}
    output_distribution: {{type: exponential, params: {{mean: 128, min: 16, max: 1024}}}}
""")
    return str(f)


def parse_output(filepath):
    """Parse BLIS JSON output."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        c = f.read().strip()
    if not c:
        return None
    instances = []
    in_json = False
    cur = []
    for line in c.split("\n"):
        if line.strip() == "{":
            in_json = True
            cur = [line]
        elif in_json:
            cur.append(line)
            if line.strip() == "}":
                in_json = False
                try:
                    instances.append(json.loads("\n".join(cur)))
                except:
                    pass
    if not instances:
        return None
    p99 = [i["ttft_p99_ms"] for i in instances if i.get("ttft_p99_ms", 0) > 0]
    comp = sum(i.get("completed_requests", 0) for i in instances)
    return {
        "p99": max(p99) if p99 else 9999,
        "comp": comp,
    }


def evaluate(params):
    """Run BLIS with given params and return objective (TTFT P99 to minimize)."""
    global call_count
    call_count += 1

    shed_thresh, std_thresh, pa_w, qd_w = params

    # Ensure std >= shed (constraint)
    if std_thresh < shed_thresh:
        return 9999.0

    scorers = f"prefix-affinity:{pa_w:.1f},queue-depth:{qd_w:.1f}"

    results = []
    for seed in SEEDS:
        wf = make_workload(seed)
        outfile = str(RESULTS_DIR / f"run_{call_count}_seed{seed}.json")

        cmd = [
            str(BINARY), "run",
            "--model", MODEL,
            "--num-instances", str(NUM_INSTANCES),
            "--routing-policy", "weighted",
            "--routing-scorers", scorers,
            "--scheduler", "priority-fcfs",
            "--priority-policy", "slo-class",
            "--admission-policy", "slo-gated",
            "--num-requests", str(NUM_REQUESTS),
            "--horizon", str(HORIZON),
            "--seed", str(seed),
            "--workload-spec", wf,
        ]

        try:
            with open(outfile, "w") as out, open(outfile + ".stderr", "w") as err:
                subprocess.run(cmd, stdout=out, stderr=err, timeout=300)
        except subprocess.TimeoutExpired:
            results.append({"p99": 9999, "comp": 0})
            continue

        r = parse_output(outfile)
        if r:
            results.append(r)
        else:
            results.append({"p99": 9999, "comp": 0})

    if not results:
        return 9999.0

    avg_p99 = sum(r["p99"] for r in results) / len(results)
    avg_comp = sum(r["comp"] for r in results) / len(results)

    # Penalty for low completion rate (want >= 70% of NUM_REQUESTS)
    completion_rate = avg_comp / NUM_REQUESTS
    if completion_rate < 0.7:
        penalty = (0.7 - completion_rate) * 1000
    else:
        penalty = 0

    objective = avg_p99 + penalty

    print(
        f"  Call {call_count:3d}: shed={shed_thresh:.1f} std={std_thresh:.1f} "
        f"pa={pa_w:.1f} qd={qd_w:.1f} → P99={avg_p99:.1f}ms "
        f"comp={avg_comp:.0f}/{NUM_REQUESTS} obj={objective:.1f}"
    )

    return objective


def main():
    print("=" * 70)
    print("Bayesian Parameter Optimization (differential_evolution)")
    print(f"Rate={RATE}, Instances={NUM_INSTANCES}, Requests={NUM_REQUESTS}")
    print("=" * 70)

    # Parameter bounds: [shed_thresh, std_thresh, pa_weight, qd_weight]
    bounds = [
        (1, 15),   # sheddable_queue_threshold
        (5, 25),   # standard_queue_threshold
        (1, 5),    # pa_weight
        (1, 5),    # qd_weight
    ]

    try:
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            evaluate,
            bounds,
            seed=42,
            maxiter=10,
            popsize=5,
            tol=0.01,
            disp=True,
        )

        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULT")
        print("=" * 70)
        print(f"  Best params: shed={result.x[0]:.1f} std={result.x[1]:.1f} "
              f"pa={result.x[2]:.1f} qd={result.x[3]:.1f}")
        print(f"  Best objective: {result.fun:.1f}")
        print(f"  Total evaluations: {call_count}")

    except ImportError:
        print("scipy not available — running grid search instead")
        best_obj = 9999
        best_params = None

        # Coarse grid search
        for shed in [2, 5, 8, 12]:
            for std in [8, 12, 18]:
                for pa in [2, 3, 4]:
                    for qd in [2, 3]:
                        if std < shed:
                            continue
                        obj = evaluate([shed, std, pa, qd])
                        if obj < best_obj:
                            best_obj = obj
                            best_params = (shed, std, pa, qd)

        print("\n" + "=" * 70)
        print("GRID SEARCH RESULT")
        print("=" * 70)
        if best_params:
            print(f"  Best params: shed={best_params[0]} std={best_params[1]} "
                  f"pa={best_params[2]} qd={best_params[3]}")
            print(f"  Best objective: {best_obj:.1f}")
        print(f"  Total evaluations: {call_count}")


if __name__ == "__main__":
    main()
