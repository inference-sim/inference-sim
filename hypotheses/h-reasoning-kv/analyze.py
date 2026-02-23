#!/usr/bin/env python3
"""Analysis script for H-Reasoning-KV.

Parses BLIS output and produces:
1. Precondition check: verifies reasoning workload token accumulation pattern
2. KV pressure sweep: preemption cliff detection, monotonicity, conservation
3. Per-round TTFT analysis (at generous KV)
4. Cache hit comparison
5. #386 DroppedUnservable verification

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
- Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
- Cluster block has "instance_id": "cluster"
- KV cache summary lines: "Preemption Rate: 0.1750", "Cache Hit Rate: 0.0452"
- Per-request data in results-path JSON: num_prefill_tokens, ttft_ms, e2e_ms
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


def parse_stdout(filepath):
    """Parse BLIS stdout -> cluster metrics + KV summary."""
    content = Path(filepath).read_text()

    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    preemption_rate = 0.0
    m = re.search(r"Preemption Rate: ([0-9.]+)", content)
    if m:
        preemption_rate = float(m.group(1))

    cache_hit_rate = 0.0
    m = re.search(r"Cache Hit Rate: ([0-9.]+)", content)
    if m:
        cache_hit_rate = float(m.group(1))

    return cluster, preemption_rate, cache_hit_rate


def parse_json_results(filepath):
    """Parse per-request JSON results file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def check_conservation(cluster):
    """Verify INV-1: injected == completed + queued + running + dropped."""
    if not cluster:
        return False, "no cluster data"
    injected = cluster.get("injected_requests", 0)
    completed = cluster.get("completed_requests", 0)
    queued = cluster.get("still_queued", 0)
    running = cluster.get("still_running", 0)
    dropped = cluster.get("dropped_unservable", 0)
    total = completed + queued + running + dropped
    ok = injected == total
    if not ok:
        return False, f"injected={injected} != completed({completed})+queued({queued})+running({running})+dropped({dropped})={total}"
    return True, f"PASS ({injected}=={total})"


def precondition_check(json_path):
    """Verify reasoning workload generates expected token accumulation pattern."""
    data = parse_json_results(json_path)
    if not data:
        print("  FAIL: No results JSON found")
        return False

    requests = data.get("requests", [])
    if not requests:
        print("  FAIL: No per-request data in JSON")
        return False

    # Expected prefill tokens for constant input=128, output=256, 5 rounds with accumulate:
    # Round 0: 128, Round 1: 128+256+128=512, Round 2: 512+256+128=896, etc.
    expected_sizes = {128: 0, 512: 1, 896: 2, 1280: 3, 1664: 4}
    prefill_tokens = [r.get("num_prefill_tokens", 0) for r in requests]

    from collections import Counter
    counts = Counter(prefill_tokens)

    print(f"  Total requests: {len(requests)}")
    print(f"  Prefill token distribution:")
    match = True
    for size in sorted(expected_sizes.keys()):
        round_idx = expected_sizes[size]
        count = counts.get(size, 0)
        print(f"    Round {round_idx} ({size:>5} tokens): {count} requests")
        if count == 0:
            print(f"    FAIL: No requests with {size} tokens (expected for round {round_idx})")
            match = False

    if match:
        print("  PASS: All 5 round types present — context accumulation working correctly")

    n_sessions = counts.get(128, 0)
    print(f"  Sessions detected: {n_sessions} (based on round-0 count)")
    return match


def spearman_rho(x, y):
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    def rank(vals):
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for r, i in enumerate(indexed):
            ranks[i] = r + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d_sq) / (n * (n * n - 1))
    # Approximate p-value using t-distribution (large-sample)
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        # Simple approximation: p ~ 2 * e^(-0.717 * t^2 / n) for large n
        p = min(1.0, 2.0 * math.exp(-0.5 * t * t / max(1, n - 2)) * max(1, n - 2))
    return rho, p


def find_cliff(blocks_list, preemption_rates, threshold=0.05):
    """Find block count where preemption rate first exceeds threshold.

    Returns interpolated block count, or None if threshold never reached.
    """
    # Sort by blocks descending (generous -> constrained)
    pairs = sorted(zip(blocks_list, preemption_rates), reverse=True)
    for i in range(1, len(pairs)):
        b_prev, p_prev = pairs[i - 1]
        b_curr, p_curr = pairs[i]
        if p_prev <= threshold and p_curr > threshold:
            # Linear interpolation
            if p_curr == p_prev:
                return b_curr
            frac = (threshold - p_prev) / (p_curr - p_prev)
            return b_prev + frac * (b_curr - b_prev)
    # Check if last point exceeds threshold
    if pairs and pairs[-1][1] > threshold:
        return pairs[-1][0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precondition", help="Run precondition check on JSON results file")
    parser.add_argument("--results-dir", help="Directory with all results")
    parser.add_argument("--block-levels", nargs="+", type=int, help="Block levels tested")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds tested")
    args = parser.parse_args()

    if args.precondition:
        print("--- Precondition: Token Accumulation Pattern ---")
        ok = precondition_check(args.precondition)
        if not ok:
            print("\nPrecondition FAILED. Fix workload before proceeding.")
            sys.exit(1)
        print("")
        return

    if not args.results_dir:
        print("Usage: analyze.py --precondition <json> OR --results-dir <dir> --block-levels ... --seeds ...")
        sys.exit(1)

    results_dir = args.results_dir
    blocks_list = args.block_levels
    seeds = args.seeds
    workloads = ["reasoning", "standard_matched_throughput", "standard_matched_sessions"]
    workload_labels = {
        "reasoning": "Reasoning",
        "standard_matched_throughput": "Std (throughput-matched)",
        "standard_matched_sessions": "Std (session-matched)",
    }

    # ---- 1. Collect all metrics ----
    data = {}  # data[workload][blocks][seed] = {preemption_rate, completed, ...}
    conservation_violations = []

    for wl in workloads:
        data[wl] = {}
        for bl in blocks_list:
            data[wl][bl] = {}
            for seed in seeds:
                stdout_file = os.path.join(results_dir, f"{wl}_{bl}_{seed}.txt")
                json_file = os.path.join(results_dir, f"{wl}_{bl}_{seed}.json")

                cluster, preemption_rate, cache_hit_rate = parse_stdout(stdout_file)

                json_data = parse_json_results(json_file)
                dropped = 0
                ttft_mean = 0
                throughput = 0
                completed = 0
                if cluster:
                    dropped = cluster.get("dropped_unservable", 0)
                    ttft_mean = cluster.get("ttft_mean_ms", 0)
                    throughput = cluster.get("responses_per_sec", 0)
                    completed = cluster.get("completed_requests", 0)

                    ok, msg = check_conservation(cluster)
                    if not ok:
                        conservation_violations.append(f"{wl} blocks={bl} seed={seed}: {msg}")

                data[wl][bl][seed] = {
                    "preemption_rate": preemption_rate,
                    "cache_hit_rate": cache_hit_rate,
                    "dropped": dropped,
                    "ttft_mean": ttft_mean,
                    "throughput": throughput,
                    "completed": completed,
                    "json_data": json_data,
                }

    # ---- 2. Preemption Rate Table ----
    print("=" * 80)
    print("1. PREEMPTION RATE BY BLOCK COUNT (3-seed average)")
    print("=" * 80)
    header = f"{'Blocks':>8}"
    for wl in workloads:
        header += f"  {workload_labels[wl]:>26}"
    print(header)
    print("-" * 80)

    for bl in sorted(blocks_list, reverse=True):
        row = f"{bl:>8}"
        for wl in workloads:
            rates = [data[wl][bl][s]["preemption_rate"] for s in seeds]
            avg = sum(rates) / len(rates)
            row += f"  {avg:>26.4f}"
        print(row)
    print("")

    # ---- 3. Cliff Detection ----
    print("=" * 80)
    print("2. PREEMPTION CLIFF DETECTION (5% threshold)")
    print("=" * 80)

    cliffs = {}  # cliffs[workload] = [cliff_per_seed]
    for wl in workloads:
        cliffs[wl] = []
        for seed in seeds:
            bls = sorted(blocks_list, reverse=True)
            prs = [data[wl][bl][seed]["preemption_rate"] for bl in bls]
            cliff = find_cliff(bls, prs, threshold=0.05)
            cliffs[wl].append(cliff)
        cliff_vals = [c for c in cliffs[wl] if c is not None]
        if cliff_vals:
            avg_cliff = sum(cliff_vals) / len(cliff_vals)
            print(f"  {workload_labels[wl]:>30}: {avg_cliff:>8.0f} blocks (seeds: {cliffs[wl]})")
        else:
            print(f"  {workload_labels[wl]:>30}: No cliff detected (preemption rate < 5% at all levels)")

    # Cliff shift ratio
    r_cliffs = [c for c in cliffs["reasoning"] if c is not None]
    t_cliffs = [c for c in cliffs["standard_matched_throughput"] if c is not None]
    if r_cliffs and t_cliffs:
        ratios = []
        for rc, tc in zip(r_cliffs, t_cliffs):
            if tc and tc > 0:
                ratios.append(rc / tc)
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"\n  Cliff shift ratio (reasoning / std-throughput): {avg_ratio:.2f}x")
            print(f"  Per-seed ratios: {[f'{r:.2f}' for r in ratios]}")
            if all(r >= 1.20 for r in ratios):
                print("  --> CLIFF SHIFT DETECTED (>=20% in all seeds)")
            elif any(r < 1.10 for r in ratios):
                print("  --> CLIFF SHIFT NOT DETECTED (<10% in at least one seed)")
            else:
                print("  --> INCONCLUSIVE (10-20% range)")
    print("")

    # ---- 4. Monotonicity (Spearman) ----
    print("=" * 80)
    print("3. MONOTONICITY: Spearman rho(blocks, -preemption_rate)")
    print("=" * 80)

    for wl in workloads:
        rhos = []
        for seed in seeds:
            bls = sorted(blocks_list)
            prs = [data[wl][bl][seed]["preemption_rate"] for bl in bls]
            neg_prs = [-p for p in prs]
            rho, p = spearman_rho(bls, neg_prs)
            rhos.append(rho)
        avg_rho = sum(rhos) / len(rhos)
        print(f"  {workload_labels[wl]:>30}: rho={avg_rho:+.3f} (per-seed: {[f'{r:+.3f}' for r in rhos]})")
    print("")

    # ---- 5. Cache Hit Rate Comparison (at generous KV) ----
    generous_bl = max(blocks_list)
    print("=" * 80)
    print(f"4. CACHE HIT RATE (blocks={generous_bl}, 3-seed average)")
    print("=" * 80)

    for wl in workloads:
        rates = [data[wl][generous_bl][s]["cache_hit_rate"] for s in seeds]
        avg = sum(rates) / len(rates)
        print(f"  {workload_labels[wl]:>30}: {avg:.4f} ({[f'{r:.4f}' for r in rates]})")
    print("")

    # ---- 6. Per-Round TTFT (reasoning at generous KV) ----
    print("=" * 80)
    print(f"5. PER-ROUND TTFT (reasoning, blocks={generous_bl})")
    print("=" * 80)

    expected_prefill = {128: 0, 512: 1, 896: 2, 1280: 3, 1664: 4}
    for seed in seeds:
        jd = data["reasoning"][generous_bl][seed]["json_data"]
        if not jd or "requests" not in jd:
            print(f"  Seed {seed}: No per-request data")
            continue
        round_ttfts = {i: [] for i in range(5)}
        for req in jd["requests"]:
            npt = req.get("num_prefill_tokens", 0)
            ttft = req.get("ttft_ms", 0)
            if npt in expected_prefill:
                round_ttfts[expected_prefill[npt]].append(ttft)

        print(f"  Seed {seed}:")
        rounds = []
        means = []
        for r in range(5):
            if round_ttfts[r]:
                avg = sum(round_ttfts[r]) / len(round_ttfts[r])
                rounds.append(r)
                means.append(avg)
                print(f"    Round {r} (input={[128,512,896,1280,1664][r]:>5}): TTFT mean={avg:>10.2f} ms (n={len(round_ttfts[r])})")
            else:
                print(f"    Round {r}: no data")

        if len(rounds) >= 3:
            rho, p = spearman_rho(rounds, means)
            print(f"    Spearman rho(round, TTFT) = {rho:+.3f}")
    print("")

    # ---- 7. #386 Verification (blocks=100) ----
    if 100 in blocks_list:
        print("=" * 80)
        print("6. #386 VERIFICATION: DroppedUnservable at blocks=100")
        print("=" * 80)

        for wl in workloads:
            drops = [data[wl][100][s]["dropped"] for s in seeds]
            avg = sum(drops) / len(drops)
            print(f"  {workload_labels[wl]:>30}: avg dropped={avg:.1f} (per-seed: {drops})")

        r_drops = [data["reasoning"][100][s]["dropped"] for s in seeds]
        t_drops = [data["standard_matched_throughput"][100][s]["dropped"] for s in seeds]
        if all(d > 0 for d in r_drops) and all(d == 0 for d in t_drops):
            print("\n  --> #386 VERIFIED: Reasoning drops oversized requests; standard does not")
        elif all(d > 0 for d in r_drops):
            print("\n  --> Reasoning drops as expected; standard also drops (unexpected for 72-block requests)")
        else:
            print("\n  --> UNEXPECTED: Check block demand calculations")
        print("")

    # ---- 8. Conservation ----
    print("=" * 80)
    print("7. CONSERVATION INVARIANT (INV-1)")
    print("=" * 80)

    if conservation_violations:
        for v in conservation_violations:
            print(f"  FAIL: {v}")
    else:
        total_checks = len(workloads) * len(blocks_list) * len(seeds)
        print(f"  PASS: {total_checks}/{total_checks} checks passed")
    print("")

    # ---- 9. Dropped + Throughput Table ----
    print("=" * 80)
    print("8. THROUGHPUT & DROPPED (3-seed average)")
    print("=" * 80)
    print(f"{'Blocks':>8}  {'Reasoning':>12} {'tput':>8}  {'Std-tput':>12} {'tput':>8}  {'Std-sess':>12} {'tput':>8}")
    print(f"{'':>8}  {'dropped':>12} {'req/s':>8}  {'dropped':>12} {'req/s':>8}  {'dropped':>12} {'req/s':>8}")
    print("-" * 90)
    for bl in sorted(blocks_list, reverse=True):
        parts = [f"{bl:>8}"]
        for wl in workloads:
            drops = [data[wl][bl][s]["dropped"] for s in seeds]
            tputs = [data[wl][bl][s]["throughput"] for s in seeds]
            avg_d = sum(drops) / len(drops)
            avg_t = sum(tputs) / len(tputs)
            parts.append(f"  {avg_d:>12.1f} {avg_t:>8.1f}")
        print("".join(parts))


if __name__ == "__main__":
    main()
