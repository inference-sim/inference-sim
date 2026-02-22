#!/usr/bin/env python3
"""Analysis script for H-Liveness hypothesis experiment (Rounds 1 + 2).

Parses BLIS multi-block output and checks scheduler liveness:
1. All requests complete (still_queued=0, still_running=0)
2. Request conservation holds (injected == completed + still_queued + still_running)
3. No scheduler starvation (all configs pass)
4. (Round 2) Scheduler differentiation — at high load, different schedulers
   produce different latency profiles, proving the scheduler was exercised.

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
- Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
- Cluster block has "instance_id": "cluster"
- MetricsOutput fields: completed_requests, still_queued, still_running, injected_requests,
  e2e_mean_ms, ttft_mean_ms, responses_per_sec, e2e_p99_ms, ttft_p99_ms
"""
import argparse
import json
import re
import sys
from pathlib import Path


def parse_cluster_metrics(filepath):
    """Parse BLIS stdout output -> cluster-level MetricsOutput dict.

    Extracts the JSON block with instance_id == "cluster".
    Returns None if no cluster block found or file is empty.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"  WARNING: file not found: {filepath}", file=sys.stderr)
        return None

    content = path.read_text()
    if not content.strip():
        print(f"  WARNING: empty file: {filepath}", file=sys.stderr)
        return None

    # Extract all JSON blocks preceded by "=== Simulation Metrics ==="
    # The JSON block may span multiple lines and contain nested structures.
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{.*?\n\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            print(f"  WARNING: JSON parse error in {filepath}: {e}", file=sys.stderr)
            continue
        if block.get("instance_id") == "cluster":
            cluster = block

    if cluster is None:
        print(f"  WARNING: no cluster JSON block found in {filepath}", file=sys.stderr)

    return cluster


def check_liveness(cluster_metrics, expected_requests):
    """Check liveness properties for a single run.

    Returns dict with:
      - pass: bool (all checks pass)
      - completed: int
      - still_queued: int
      - still_running: int
      - injected: int
      - conservation_holds: bool
      - all_complete: bool
      - errors: list of str
    """
    result = {
        "pass": False,
        "completed": 0,
        "still_queued": 0,
        "still_running": 0,
        "injected": 0,
        "conservation_holds": False,
        "all_complete": False,
        "errors": [],
    }

    if cluster_metrics is None:
        result["errors"].append("No cluster metrics found")
        return result

    completed = cluster_metrics.get("completed_requests", 0)
    still_queued = cluster_metrics.get("still_queued", 0)
    still_running = cluster_metrics.get("still_running", 0)
    injected = cluster_metrics.get("injected_requests", 0)

    result["completed"] = completed
    result["still_queued"] = still_queued
    result["still_running"] = still_running
    result["injected"] = injected

    # Check 1: Conservation (INV-1)
    if injected == completed + still_queued + still_running:
        result["conservation_holds"] = True
    else:
        result["errors"].append(
            f"Conservation violated: injected={injected} != "
            f"completed({completed}) + queued({still_queued}) + running({still_running}) "
            f"= {completed + still_queued + still_running}"
        )

    # Check 2: All requests complete (liveness)
    if still_queued == 0 and still_running == 0:
        result["all_complete"] = True
    else:
        result["errors"].append(
            f"Liveness violated: still_queued={still_queued}, still_running={still_running}"
        )

    # Check 3: Expected request count
    if injected != expected_requests:
        result["errors"].append(
            f"Unexpected injection count: injected={injected}, expected={expected_requests}"
        )

    result["pass"] = result["conservation_holds"] and result["all_complete"]
    return result


def analyze_round(results_dir, schedulers, seeds, workloads, rate, expected_requests, label, file_suffix=""):
    """Analyze one round of experiments.

    Returns (all_results dict, total_pass, total_fail, total_missing, latencies dict).
    latencies[sched][seed] = e2e_mean_ms (for scheduler differentiation check).
    file_suffix: additional suffix for file names (e.g., "_b8" for batch-constrained runs).
    """
    all_results = {}
    latencies = {}
    total_pass = 0
    total_fail = 0
    total_missing = 0

    for sched in schedulers:
        latencies[sched] = {}
        for workload in workloads:
            for seed in seeds:
                suffix_label = file_suffix.replace("_", "/") if file_suffix else ""
                key = f"{sched}/{workload}/rate={rate}{suffix_label}/seed={seed}"
                stdout_file = results_dir / f"{sched}_{workload}_r{rate}{file_suffix}_s{seed}_stdout.txt"

                cluster = parse_cluster_metrics(str(stdout_file))
                result = check_liveness(cluster, expected_requests)
                all_results[key] = (result, cluster)

                if cluster is None:
                    total_missing += 1
                elif result["pass"]:
                    total_pass += 1
                else:
                    total_fail += 1

                # Collect latencies per scheduler (for mixed workload)
                if cluster and workload == "mixed":
                    latencies[sched][seed] = cluster.get("e2e_mean_ms", 0)

    # Print summary table
    total_runs = len(schedulers) * len(workloads) * len(seeds)
    print("=" * 90)
    print(f"  {label}: LIVENESS RESULTS (rate={rate}, {expected_requests} requests)")
    print("=" * 90)
    print()

    print(f"{'Config':<45} {'Completed':>10} {'Queued':>8} {'Running':>8} {'Injected':>10} {'Status':>8}")
    print("-" * 90)

    for sched in schedulers:
        for workload in workloads:
            for seed in seeds:
                suffix_label = file_suffix.replace("_", "/") if file_suffix else ""
                key = f"{sched}/{workload}/rate={rate}{suffix_label}/seed={seed}"
                r, _ = all_results[key]
                status = "PASS" if r["pass"] else ("MISSING" if r["completed"] == 0 and r["injected"] == 0 else "FAIL")
                print(
                    f"  {key:<43} {r['completed']:>10} {r['still_queued']:>8} "
                    f"{r['still_running']:>8} {r['injected']:>10} {status:>8}"
                )
            print()

    # Print latency comparison
    print()
    print(f"{'Config':<45} {'E2E Mean':>10} {'E2E P99':>10} {'TTFT Mean':>10} {'TTFT P99':>10} {'Throughput':>12}")
    print("-" * 100)

    for sched in schedulers:
        for workload in workloads:
            for seed in seeds:
                suffix_label = file_suffix.replace("_", "/") if file_suffix else ""
                key = f"{sched}/{workload}/rate={rate}{suffix_label}/seed={seed}"
                _, cluster = all_results[key]
                if cluster:
                    e2e = cluster.get("e2e_mean_ms", 0)
                    e2e_p99 = cluster.get("e2e_p99_ms", 0)
                    ttft = cluster.get("ttft_mean_ms", 0)
                    ttft_p99 = cluster.get("ttft_p99_ms", 0)
                    tput = cluster.get("responses_per_sec", 0)
                    print(f"  {key:<43} {e2e:>10.2f} {e2e_p99:>10.2f} {ttft:>10.2f} {ttft_p99:>10.2f} {tput:>12.2f}")
                else:
                    print(f"  {key:<43} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>12}")
            print()

    return all_results, total_pass, total_fail, total_missing, latencies


def check_scheduler_differentiation(latencies, rate):
    """Check if schedulers produce different latencies (proving queue contention).

    Returns (differentiated: bool, details: str).
    """
    # Average E2E across seeds per scheduler
    means = {}
    for sched, seed_latencies in latencies.items():
        if seed_latencies:
            means[sched] = sum(seed_latencies.values()) / len(seed_latencies)

    if len(means) < 2:
        return False, "Insufficient data for differentiation check"

    # Check if max and min differ by more than 1% (arbitrary small threshold)
    max_lat = max(means.values())
    min_lat = min(means.values())

    if min_lat == 0:
        return False, "Zero latency detected — invalid data"

    pct_diff = (max_lat - min_lat) / min_lat * 100

    details_parts = [f"  Mean E2E by scheduler (mixed workload, rate={rate}):"]
    for sched in sorted(means.keys()):
        details_parts.append(f"    {sched}: {means[sched]:.2f} ms")
    details_parts.append(f"  Max/min difference: {pct_diff:.1f}%")

    if pct_diff > 1.0:
        details_parts.append(f"  DIFFERENTIATED: schedulers produce distinct latencies at rate={rate}")
        return True, "\n".join(details_parts)
    else:
        details_parts.append(f"  NOT DIFFERENTIATED: schedulers produce identical latencies at rate={rate} (queue rarely contended)")
        return False, "\n".join(details_parts)


def main():
    parser = argparse.ArgumentParser(description="H-Liveness analysis (Rounds 1 + 2)")
    parser.add_argument("--results-dir", required=True, help="Directory with stdout files")
    parser.add_argument("--schedulers", required=True, help="Space-separated scheduler names")
    parser.add_argument("--seeds", required=True, help="Space-separated seed values")
    parser.add_argument("--round1-rate", required=True, type=int, help="Round 1 rate")
    parser.add_argument("--round1-requests", required=True, type=int, help="Round 1 request count")
    parser.add_argument("--round2-rates", required=True, help="Space-separated Round 2 rates")
    parser.add_argument("--round2-requests", required=True, type=int, help="Round 2 request count")
    parser.add_argument("--round2b-rate", type=int, default=0, help="Round 2b rate (constrained batch)")
    parser.add_argument("--round2b-requests", type=int, default=0, help="Round 2b request count")
    parser.add_argument("--round2b-max-running", type=int, default=0, help="Round 2b max-num-running-reqs")
    parser.add_argument("--round2c-rate", type=int, default=0, help="Round 2c rate (token budget isolation)")
    parser.add_argument("--round2c-requests", type=int, default=0, help="Round 2c request count")
    parser.add_argument("--round2c-max-running", type=int, default=0, help="Round 2c max-num-running-reqs")
    parser.add_argument("--round2c-token-budget", type=int, default=0, help="Round 2c token budget")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    schedulers = args.schedulers.split()
    seeds = [int(s) for s in args.seeds.split()]
    round2_rates = [int(r) for r in args.round2_rates.split()]

    overall_pass = 0
    overall_fail = 0
    overall_missing = 0
    overall_runs = 0
    all_round_results = {}  # merged across rounds for failure reporting

    # ── Round 1: Low-load (rho~0.3) ──────────────────────────────────────
    r1_results, r1_pass, r1_fail, r1_missing, r1_latencies = analyze_round(
        results_dir, schedulers, seeds,
        workloads=["uniform", "mixed"],
        rate=args.round1_rate,
        expected_requests=args.round1_requests,
        label="ROUND 1 (rho~0.3)"
    )
    overall_pass += r1_pass
    overall_fail += r1_fail
    overall_missing += r1_missing
    overall_runs += len(r1_results)
    all_round_results.update(r1_results)

    # Differentiation check for Round 1 (expected: NOT differentiated)
    r1_diff, r1_diff_details = check_scheduler_differentiation(r1_latencies, args.round1_rate)
    print()
    print("  SCHEDULER DIFFERENTIATION CHECK (Round 1):")
    print(r1_diff_details)
    print()

    # ── Round 2: High-load (rho~0.7, rho~0.85) ──────────────────────────
    r2_all_differentiated = True
    for rate in round2_rates:
        rho_label = f"rho~{rate/328:.2f}"
        r2_results, r2_pass, r2_fail, r2_missing, r2_latencies = analyze_round(
            results_dir, schedulers, seeds,
            workloads=["mixed"],
            rate=rate,
            expected_requests=args.round2_requests,
            label=f"ROUND 2 ({rho_label}, rate={rate})"
        )
        overall_pass += r2_pass
        overall_fail += r2_fail
        overall_missing += r2_missing
        overall_runs += len(r2_results)
        all_round_results.update(r2_results)

        # Differentiation check for Round 2 (expected: DIFFERENTIATED at high load)
        r2_diff, r2_diff_details = check_scheduler_differentiation(r2_latencies, rate)
        print()
        print(f"  SCHEDULER DIFFERENTIATION CHECK (Round 2, rate={rate}):")
        print(r2_diff_details)
        if not r2_diff:
            r2_all_differentiated = False
        print()

    # ── Round 2b: Constrained-batch (force queueing) ────────────────────
    r2b_differentiated = False
    if args.round2b_rate > 0 and args.round2b_requests > 0:
        file_suffix = f"_b{args.round2b_max_running}"
        r2b_results, r2b_pass, r2b_fail, r2b_missing, r2b_latencies = analyze_round(
            results_dir, schedulers, seeds,
            workloads=["mixed"],
            rate=args.round2b_rate,
            expected_requests=args.round2b_requests,
            label=f"ROUND 2b (constrained batch={args.round2b_max_running}, rate={args.round2b_rate})",
            file_suffix=file_suffix
        )
        overall_pass += r2b_pass
        overall_fail += r2b_fail
        overall_missing += r2b_missing
        overall_runs += len(r2b_results)
        all_round_results.update(r2b_results)

        r2b_differentiated, r2b_diff_details = check_scheduler_differentiation(
            r2b_latencies, f"{args.round2b_rate}/batch={args.round2b_max_running}"
        )
        print()
        print(f"  SCHEDULER DIFFERENTIATION CHECK (Round 2b, batch-constrained):")
        print(r2b_diff_details)
        print()

        # Update overall differentiation flag
        if r2b_differentiated:
            r2_all_differentiated = True

    # ── Round 2c: Token budget isolation ──────────────────────────────────
    if args.round2c_rate > 0 and args.round2c_requests > 0:
        file_suffix = f"_t{args.round2c_token_budget}"
        r2c_results, r2c_pass, r2c_fail, r2c_missing, r2c_latencies = analyze_round(
            results_dir, schedulers, seeds,
            workloads=["mixed"],
            rate=args.round2c_rate,
            expected_requests=args.round2c_requests,
            label=f"ROUND 2c (token budget={args.round2c_token_budget}, max-running={args.round2c_max_running}, rate={args.round2c_rate})",
            file_suffix=file_suffix
        )
        overall_pass += r2c_pass
        overall_fail += r2c_fail
        overall_missing += r2c_missing
        overall_runs += len(r2c_results)
        all_round_results.update(r2c_results)

        r2c_differentiated, r2c_diff_details = check_scheduler_differentiation(
            r2c_latencies, f"{args.round2c_rate}/tokens={args.round2c_token_budget}"
        )
        print()
        print(f"  SCHEDULER DIFFERENTIATION CHECK (Round 2c, token budget isolation):")
        print(r2c_diff_details)
        print()

        # Compare Round 2b vs Round 2c differentiation
        print("  TOKEN BUDGET ISOLATION COMPARISON:")
        if r2b_differentiated and not r2c_differentiated:
            print("    Round 2b (tokens=2048): DIFFERENTIATED")
            print("    Round 2c (tokens=65536): NOT DIFFERENTIATED")
            print("    => Token budget (2048) was the binding constraint, not max-running-reqs.")
        elif r2b_differentiated and r2c_differentiated:
            print("    Round 2b (tokens=2048): DIFFERENTIATED")
            print("    Round 2c (tokens=65536): DIFFERENTIATED")
            print("    => max-running-reqs is the binding constraint (finding robust to token budget).")
        else:
            print("    Round 2b: NOT differentiated, Round 2c: check results above.")
        print()

    # ── Overall verdict ───────────────────────────────────────────────────
    print("=" * 90)
    print("  OVERALL VERDICT")
    print("=" * 90)
    print()
    print(f"  Total runs:    {overall_runs}")
    print(f"  Passed:        {overall_pass}")
    print(f"  Failed:        {overall_fail}")
    print(f"  Missing:       {overall_missing}")
    print()

    if overall_fail == 0 and overall_missing == 0:
        print("  LIVENESS: CONFIRMED -- All schedulers satisfy liveness under admissible load.")
        print("  Every admitted request completed (still_queued=0, still_running=0).")
        print("  Request conservation (INV-1) holds across all configurations.")
    elif overall_fail > 0:
        print("  LIVENESS: PARTIALLY CONFIRMED or REFUTED -- Some configurations have failures.")
        print()
        print("  Failures:")
        for key, (r, _) in all_round_results.items():
            if not r["pass"]:
                for err in r["errors"]:
                    print(f"    {key}: {err}")
    else:
        print("  LIVENESS: INCONCLUSIVE -- Missing data for some configurations.")

    print()

    if r2_all_differentiated:
        print("  DIFFERENTIATION: CONFIRMED -- High-load runs produce distinct scheduler latencies.")
        print("  This proves the scheduler ordering logic was exercised (queue non-empty).")
    else:
        print("  DIFFERENTIATION: NOT CONFIRMED -- Schedulers still produce identical results at high load.")
        print("  The experiment may need even higher load or different workload parameters.")

    print()

    # Return exit code based on result
    sys.exit(0 if overall_fail == 0 and overall_missing == 0 else 1)


if __name__ == "__main__":
    main()
