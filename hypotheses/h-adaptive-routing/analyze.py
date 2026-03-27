#!/usr/bin/env python3
"""Analyze H-Adaptive-Routing experiment results.

Compares adaptive-weighted routing against static policies across three
workload types: prefix-heavy, independent, and mixed.

Key metrics: TTFT p99, TTFT mean, E2E mean, completion rate.
"""

import json
import os
import sys
from pathlib import Path

# Add shared helpers
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

SEEDS = [42, 123, 7777]
WORKLOADS = ["prefix", "indep", "mixed"]
POLICIES = [
    "adaptive",
    "static-default",
    "static-cache-heavy",
    "static-load-heavy",
    "round-robin",
    "least-loaded",
]


def parse_blis_output(filepath):
    """Parse BLIS JSON output, extracting aggregate metrics."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        content = f.read().strip()
    if content.startswith("TIMEOUT") or content.startswith("ERROR"):
        return {"error": content}

    # BLIS outputs multiple JSON objects (one per instance + aggregate)
    # We need the aggregate metrics
    lines = content.split("\n")
    instances = []
    in_json = False
    current = []

    # Parse the output format: "=== Simulation Metrics ===" followed by JSON blocks
    for line in lines:
        if line.strip() == "=== Simulation Metrics ===":
            continue
        if line.strip() == "{":
            in_json = True
            current = [line]
        elif in_json:
            current.append(line)
            if line.strip() == "}":
                in_json = False
                try:
                    instances.append(json.loads("\n".join(current)))
                except json.JSONDecodeError:
                    pass
                current = []

    if not instances:
        return None

    # Aggregate across instances
    total_completed = sum(inst.get("completed_requests", 0) for inst in instances)
    total_queued = sum(inst.get("still_queued", 0) for inst in instances)
    total_running = sum(inst.get("still_running", 0) for inst in instances)
    total_dropped = sum(inst.get("dropped_unservable", 0) for inst in instances)

    # Collect all per-instance timing metrics
    ttft_p99_values = []
    ttft_mean_values = []
    e2e_mean_values = []
    e2e_p99_values = []

    for inst in instances:
        if inst.get("ttft_p99_ms", 0) > 0:
            ttft_p99_values.append(inst["ttft_p99_ms"])
        if inst.get("ttft_mean_ms", 0) > 0:
            ttft_mean_values.append(inst["ttft_mean_ms"])
        if inst.get("e2e_mean_ms", 0) > 0:
            e2e_mean_values.append(inst["e2e_mean_ms"])
        if inst.get("e2e_p99_ms", 0) > 0:
            e2e_p99_values.append(inst["e2e_p99_ms"])

    return {
        "completed": total_completed,
        "queued": total_queued,
        "running": total_running,
        "dropped": total_dropped,
        "ttft_p99_ms": max(ttft_p99_values) if ttft_p99_values else 0,
        "ttft_mean_ms": (sum(ttft_mean_values) / len(ttft_mean_values)) if ttft_mean_values else 0,
        "e2e_mean_ms": (sum(e2e_mean_values) / len(e2e_mean_values)) if e2e_mean_values else 0,
        "e2e_p99_ms": max(e2e_p99_values) if e2e_p99_values else 0,
        "instances": len(instances),
    }


def main(results_dir):
    print("=" * 80)
    print("H-Adaptive-Routing Analysis")
    print("=" * 80)

    # Collect all results
    all_results = {}
    for policy in POLICIES:
        all_results[policy] = {}
        for workload in WORKLOADS:
            all_results[policy][workload] = []
            for seed in SEEDS:
                filepath = os.path.join(results_dir, f"{policy}_{workload}_seed{seed}.json")
                result = parse_blis_output(filepath)
                if result and "error" not in result:
                    all_results[policy][workload].append(result)
                elif result and "error" in result:
                    print(f"  WARNING: {policy}/{workload}/seed{seed}: {result['error']}")

    # Print per-workload comparison tables
    for workload in WORKLOADS:
        print(f"\n{'=' * 60}")
        print(f"Workload: {workload.upper()}")
        print(f"{'=' * 60}")
        print(f"{'Policy':<22} {'TTFT p99':>10} {'TTFT mean':>10} {'E2E mean':>10} {'E2E p99':>10} {'Completed':>10}")
        print("-" * 72)

        best_ttft_p99 = float("inf")
        best_policy_p99 = ""

        for policy in POLICIES:
            results = all_results[policy][workload]
            if not results:
                print(f"  {policy:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
                continue

            avg_ttft_p99 = sum(r["ttft_p99_ms"] for r in results) / len(results)
            avg_ttft_mean = sum(r["ttft_mean_ms"] for r in results) / len(results)
            avg_e2e_mean = sum(r["e2e_mean_ms"] for r in results) / len(results)
            avg_e2e_p99 = sum(r["e2e_p99_ms"] for r in results) / len(results)
            avg_completed = sum(r["completed"] for r in results) / len(results)

            print(
                f"  {policy:<20} {avg_ttft_p99:>9.2f}ms {avg_ttft_mean:>9.2f}ms "
                f"{avg_e2e_mean:>9.2f}ms {avg_e2e_p99:>9.2f}ms {avg_completed:>9.0f}"
            )

            if avg_ttft_p99 < best_ttft_p99:
                best_ttft_p99 = avg_ttft_p99
                best_policy_p99 = policy

        print(f"\n  BEST TTFT p99: {best_policy_p99} ({best_ttft_p99:.2f}ms)")

    # Compute cross-workload combined metric (average TTFT p99 across all workloads)
    print(f"\n{'=' * 60}")
    print("CROSS-WORKLOAD COMBINED (avg TTFT p99 across all workloads)")
    print(f"{'=' * 60}")
    print(f"{'Policy':<22} {'Prefix':>10} {'Indep':>10} {'Mixed':>10} {'Combined':>10} {'vs Best Static':>15}")
    print("-" * 77)

    combined_scores = {}
    for policy in POLICIES:
        scores = []
        for workload in WORKLOADS:
            results = all_results[policy][workload]
            if results:
                avg = sum(r["ttft_p99_ms"] for r in results) / len(results)
                scores.append(avg)
            else:
                scores.append(float("inf"))
        combined = sum(scores) / len(scores)
        combined_scores[policy] = (scores, combined)

    # Find best static policy
    static_policies = [p for p in POLICIES if p != "adaptive"]
    best_static = min(static_policies, key=lambda p: combined_scores[p][1])
    best_static_combined = combined_scores[best_static][1]

    for policy in POLICIES:
        scores, combined = combined_scores[policy]
        improvement = ((best_static_combined - combined) / best_static_combined) * 100
        marker = " <-- ADAPTIVE" if policy == "adaptive" else ""
        marker = " <-- BEST STATIC" if policy == best_static else marker
        print(
            f"  {policy:<20} {scores[0]:>9.2f}ms {scores[1]:>9.2f}ms {scores[2]:>9.2f}ms "
            f"{combined:>9.2f}ms {improvement:>+13.1f}%{marker}"
        )

    print(f"\n  Best static policy: {best_static} ({best_static_combined:.2f}ms)")
    adaptive_combined = combined_scores["adaptive"][1]
    overall_improvement = ((best_static_combined - adaptive_combined) / best_static_combined) * 100
    print(f"  Adaptive combined: {adaptive_combined:.2f}ms")
    print(f"  Improvement over best static: {overall_improvement:+.1f}%")

    if overall_improvement > 20:
        print("\n  VERDICT: CONFIRMED — Adaptive routing >20% better than best static")
    elif overall_improvement > 0:
        print(f"\n  VERDICT: PARTIAL — Adaptive routing {overall_improvement:.1f}% better (below 20% threshold)")
    else:
        print(f"\n  VERDICT: REFUTED — Adaptive routing {overall_improvement:.1f}% vs best static")

    # Per-seed breakdown for statistical robustness
    print(f"\n{'=' * 60}")
    print("PER-SEED BREAKDOWN (TTFT p99 in ms)")
    print(f"{'=' * 60}")
    for workload in WORKLOADS:
        print(f"\n  {workload.upper()}:")
        print(f"  {'Policy':<22}" + "".join(f" {'Seed ' + str(s):>10}" for s in SEEDS))
        print("  " + "-" * 52)
        for policy in POLICIES:
            results = all_results[policy][workload]
            vals = []
            for seed in SEEDS:
                filepath = os.path.join(results_dir, f"{policy}_{workload}_seed{seed}.json")
                r = parse_blis_output(filepath)
                vals.append(f"{r['ttft_p99_ms']:>9.2f}ms" if r and "error" not in r else f"{'N/A':>10}")
            print(f"  {policy:<22}" + "".join(f" {v}" for v in vals))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
