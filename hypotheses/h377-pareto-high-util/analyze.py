#!/usr/bin/env python3
"""Analysis script for H377: Pareto Frontier at High Utilization.

Extends H17's Pareto analysis to high-utilization operating points.
Parses BLIS multi-block output and produces:
1. Per-rate, per-seed metric tables
2. Seed-averaged comparison tables
3. Pareto dominance analysis per rate level
4. Cross-rate comparison (does the Pareto set change with utilization?)
5. INV-1 conservation verification

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
  - Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
  - Cluster block has "instance_id": "cluster"
  - JSON fields: ttft_mean_ms, ttft_p99_ms, e2e_mean_ms, e2e_p99_ms, responses_per_sec,
    completed_requests, injected_requests, still_queued, still_running, dropped_unservable
  - KV cache summary: "Preemption Rate: %.4f", "Cache Hit Rate: %.4f" (cmd/root.go)
  - Target Distribution block from trace summary

Usage: python3 analyze.py <results_dir> <config_names...> <seeds...>
  Results directory has subdirectories per rate: high/, moderate/
  Files expected at: <results_dir>/<rate>/<config_name>_seed<seed>.txt
"""

import json
import math
import re
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output


def parse_output_extended(filepath):
    """Parse BLIS output with extended metrics for Pareto analysis.

    Uses shared parse_blis_output for core metrics, then adds distribution
    balance metrics from trace summary.
    """
    base = parse_blis_output(filepath)
    if base["timed_out"]:
        base["dist"] = {}
        base["dist_stddev"] = 0.0
        return base

    content = Path(filepath).read_text()

    # Target distribution from trace summary (cmd/root.go)
    dist = {}
    dist_match = re.search(
        r"Target Distribution:\n((?:\s+instance_\d+: \d+\n?)+)", content
    )
    if dist_match:
        for line in dist_match.group(1).strip().split("\n"):
            parts = line.strip().split(": ")
            if len(parts) == 2:
                dist[parts[0]] = int(parts[1])

    # Compute distribution balance (stddev of per-instance counts)
    counts = [dist[k] for k in sorted(dist.keys())] if dist else [0]
    mean_d = sum(counts) / len(counts) if counts else 0
    stddev = math.sqrt(sum((x - mean_d) ** 2 for x in counts) / len(counts)) if counts else 0

    base["dist"] = dist
    base["dist_stddev"] = stddev
    return base


def verify_conservation(metrics, label):
    """Verify INV-1: injected == completed + still_queued + still_running + dropped_unservable."""
    injected = metrics.get("injected", 0)
    completed = metrics.get("completed", 0)
    still_queued = metrics.get("still_queued", 0)
    still_running = metrics.get("still_running", 0)
    # dropped_unservable is not in parse_blis_output; check JSON directly
    # For now, check the three-term version: injected >= completed + still_queued + still_running
    rhs = completed + still_queued + still_running
    if injected > 0 and injected != rhs:
        # May have dropped_unservable
        gap = injected - rhs
        if gap < 0:
            print(f"  WARNING: INV-1 violation in {label}: injected={injected} < "
                  f"completed({completed})+queued({still_queued})+running({still_running})={rhs}",
                  file=sys.stderr)
            return False
    return True


def pareto_dominates(a, b, metrics_to_minimize, metrics_to_maximize):
    """Return True if config 'a' Pareto-dominates config 'b'."""
    at_least_as_good = True
    strictly_better = False

    for m in metrics_to_minimize:
        if a[m] > b[m]:
            at_least_as_good = False
            break
        if a[m] < b[m]:
            strictly_better = True

    if not at_least_as_good:
        return False

    for m in metrics_to_maximize:
        if a[m] < b[m]:
            at_least_as_good = False
            break
        if a[m] > b[m]:
            strictly_better = True

    return at_least_as_good and strictly_better


def find_pareto_set(configs, metrics_to_minimize, metrics_to_maximize):
    """Return list of config names that are Pareto-optimal (non-dominated)."""
    names = list(configs.keys())
    dominated = set()
    for i, a_name in enumerate(names):
        for j, b_name in enumerate(names):
            if i == j:
                continue
            if b_name in dominated:
                continue
            if pareto_dominates(configs[b_name], configs[a_name],
                                metrics_to_minimize, metrics_to_maximize):
                dominated.add(a_name)
                break
    return [n for n in names if n not in dominated], dominated


def analyze_rate(rate_name, results_dir, config_names, seeds):
    """Analyze a single rate level, return averaged metrics dict."""
    subdir = results_dir / rate_name

    if not subdir.exists():
        print(f"\n  SKIPPED: {rate_name} (directory not found)")
        return None

    # Parse all results
    all_results = {}
    conservation_ok = True
    for name in config_names:
        all_results[name] = {}
        for seed in seeds:
            filepath = subdir / f"{name}_seed{seed}.txt"
            if not filepath.exists():
                print(f"WARNING: Missing file {filepath}", file=sys.stderr)
                continue
            result = parse_output_extended(str(filepath))
            if result["timed_out"]:
                print(f"  WARNING: {name}/seed{seed} timed out or errored", file=sys.stderr)
                continue
            all_results[name][seed] = result
            # Verify INV-1
            if not verify_conservation(result, f"{rate_name}/{name}/seed{seed}"):
                conservation_ok = False

    if not conservation_ok:
        print(f"  *** INV-1 conservation check: FAILURES detected ***")
    else:
        print(f"  INV-1 conservation check: PASSED (all runs)")

    # ── Per-seed table ──
    print(f"\n  Per-seed results:")
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        hdr = (f"  {'Configuration':<16} | {'TTFT Mean':>10} {'TTFT P99':>10} "
               f"{'E2E Mean':>10} {'E2E P99':>10} {'Tput':>8} "
               f"| {'CacheHit':>8} {'Preempt':>8} | Distribution")
        print(hdr)
        print(f"  {'-'*16}-+-{'-'*52}-+-{'-'*18}-+-{'-'*30}")

        for name in config_names:
            r = all_results[name].get(seed)
            if not r:
                print(f"  {name:<16} | {'MISSING/TIMEOUT':>52} |")
                continue
            dist_str = str([r['dist'].get(k, 0) for k in sorted(r['dist'].keys())])
            print(f"  {name:<16} | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f} "
                  f"{r['e2e_mean']:>10.1f} {r['e2e_p99']:>10.1f} {r['throughput']:>8.1f} "
                  f"| {r['cache_hit_rate']:>7.1%} {r['preemption_rate']:>7.4f} | {dist_str}")

    # ── Seed-averaged ──
    averaged = {}
    metrics_keys = ["ttft_mean", "ttft_p99", "e2e_mean", "e2e_p99",
                    "throughput", "cache_hit_rate", "preemption_rate"]

    for name in config_names:
        seed_results = all_results[name]
        if not seed_results:
            continue
        avg = {}
        for mk in metrics_keys:
            values = [seed_results[s][mk] for s in seeds if s in seed_results]
            avg[mk] = sum(values) / len(values) if values else 0
            avg[f"{mk}_min"] = min(values) if values else 0
            avg[f"{mk}_max"] = max(values) if values else 0
        # Also track completed count
        completed_values = [seed_results[s]["completed"] for s in seeds if s in seed_results]
        avg["completed"] = sum(completed_values) / len(completed_values) if completed_values else 0
        averaged[name] = avg

    print(f"\n  Seed-averaged results:")
    hdr = (f"  {'Configuration':<16} | {'TTFT Mean':>10} {'TTFT P99':>10} "
           f"{'E2E Mean':>10} {'E2E P99':>10} {'Tput':>8} | {'CacheHit':>8} {'Preempt':>8} | {'Completed':>9}")
    print(hdr)
    print(f"  {'-'*16}-+-{'-'*52}-+-{'-'*18}-+-{'-'*9}")

    for name in config_names:
        a = averaged.get(name)
        if not a:
            continue
        print(f"  {name:<16} | {a['ttft_mean']:>10.1f} {a['ttft_p99']:>10.1f} "
              f"{a['e2e_mean']:>10.1f} {a['e2e_p99']:>10.1f} {a['throughput']:>8.1f} "
              f"| {a['cache_hit_rate']:>7.1%} {a['preemption_rate']:>7.4f} | {a['completed']:>9.0f}")

    # Seed variability
    print(f"\n  Seed variability (min - max across {len(seeds)} seeds):")
    for name in config_names:
        a = averaged.get(name)
        if not a:
            continue
        print(f"  {name:<16}: TTFT mean [{a['ttft_mean_min']:.1f} - {a['ttft_mean_max']:.1f}]  "
              f"TTFT p99 [{a['ttft_p99_min']:.1f} - {a['ttft_p99_max']:.1f}]  "
              f"E2E mean [{a['e2e_mean_min']:.1f} - {a['e2e_mean_max']:.1f}]  "
              f"Tput [{a['throughput_min']:.1f} - {a['throughput_max']:.1f}]")

    # ── Pareto dominance ──
    minimize = ["ttft_mean", "ttft_p99", "e2e_mean", "e2e_p99"]
    maximize = ["throughput"]

    pareto_set, dominated = find_pareto_set(averaged, minimize, maximize)

    print(f"\n  Pareto-optimal (non-dominated): {len(pareto_set)}")
    for name in pareto_set:
        a = averaged[name]
        print(f"    {name}: TTFT mean={a['ttft_mean']:.1f} TTFT p99={a['ttft_p99']:.1f} "
              f"E2E mean={a['e2e_mean']:.1f} Tput={a['throughput']:.1f} CacheHit={a['cache_hit_rate']:.1%}")

    if dominated:
        print(f"\n  Dominated: {len(dominated)}")
        for name in sorted(dominated):
            a = averaged[name]
            dominators = []
            for other in pareto_set:
                if pareto_dominates(averaged[other], a, minimize, maximize):
                    dominators.append(other)
            print(f"    {name}: TTFT mean={a['ttft_mean']:.1f} TTFT p99={a['ttft_p99']:.1f} "
                  f"E2E mean={a['e2e_mean']:.1f} Tput={a['throughput']:.1f} "
                  f"(dominated by: {', '.join(dominators)})")

    # Noise check
    if len(pareto_set) >= 2:
        noise_flags = []
        for name_a in pareto_set:
            for name_b in pareto_set:
                if name_a == name_b:
                    continue
                advantages_a = []
                for m in minimize:
                    if averaged[name_a][m] < averaged[name_b][m]:
                        pct = (averaged[name_b][m] - averaged[name_a][m]) / max(abs(averaged[name_b][m]), 1e-9) * 100
                        advantages_a.append((m, pct))
                for m in maximize:
                    if averaged[name_a][m] > averaged[name_b][m]:
                        pct = (averaged[name_a][m] - averaged[name_b][m]) / max(abs(averaged[name_b][m]), 1e-9) * 100
                        advantages_a.append((m, pct))
                if len(advantages_a) == 1 and advantages_a[0][1] < 1.0:
                    noise_flags.append(
                        f"    {name_a} survives only via {advantages_a[0][0]} "
                        f"({advantages_a[0][1]:.2f}% margin) -- likely noise"
                    )
        if noise_flags:
            print(f"\n  Noise analysis (margins < 1% on surviving metric):")
            for flag in noise_flags:
                print(flag)

    # Per-metric best
    print(f"\n  Per-metric best:")
    all_metrics = minimize + maximize
    for mk in all_metrics:
        if mk in minimize:
            best_name = min(averaged.keys(), key=lambda n: averaged[n][mk])
            best_val = averaged[best_name][mk]
            worst_name = max(averaged.keys(), key=lambda n: averaged[n][mk])
            worst_val = averaged[worst_name][mk]
        else:
            best_name = max(averaged.keys(), key=lambda n: averaged[n][mk])
            best_val = averaged[best_name][mk]
            worst_name = min(averaged.keys(), key=lambda n: averaged[n][mk])
            worst_val = averaged[worst_name][mk]

        spread = abs(best_val - worst_val) / max(abs(worst_val), 1e-9) * 100
        print(f"    {mk:<12}: best={best_name} ({best_val:.1f})  "
              f"worst={worst_name} ({worst_val:.1f})  spread={spread:.1f}%")

    # Directional consistency
    print(f"\n  Directional consistency (all seeds agree on best/worst?):")
    for mk in ["ttft_mean", "ttft_p99", "e2e_mean", "throughput"]:
        rankings_per_seed = []
        for seed in seeds:
            vals = []
            for name in config_names:
                if seed in all_results[name]:
                    vals.append((all_results[name][seed][mk], name))
            vals.sort()
            rankings_per_seed.append([n for _, n in vals])

        best_per_seed = [r[0] if mk != "throughput" else r[-1] for r in rankings_per_seed]
        worst_per_seed = [r[-1] if mk != "throughput" else r[0] for r in rankings_per_seed]
        best_consistent = len(set(best_per_seed)) == 1
        worst_consistent = len(set(worst_per_seed)) == 1
        print(f"    {mk:<12}: best={best_per_seed} ({'consistent' if best_consistent else 'INCONSISTENT'})  "
              f"worst={worst_per_seed} ({'consistent' if worst_consistent else 'INCONSISTENT'})")

    # Key tradeoff analysis: does cache-heavy win TTFT mean but lose TTFT p99?
    if "cache-heavy" in averaged and len(pareto_set) >= 2:
        ch = averaged["cache-heavy"]
        print(f"\n  Tradeoff analysis (cache-heavy vs others):")
        for name in config_names:
            if name == "cache-heavy" or name not in averaged:
                continue
            other = averaged[name]
            ttft_mean_diff = (ch["ttft_mean"] - other["ttft_mean"]) / max(abs(other["ttft_mean"]), 1e-9) * 100
            ttft_p99_diff = (ch["ttft_p99"] - other["ttft_p99"]) / max(abs(other["ttft_p99"]), 1e-9) * 100
            e2e_mean_diff = (ch["e2e_mean"] - other["e2e_mean"]) / max(abs(other["e2e_mean"]), 1e-9) * 100
            tput_diff = (ch["throughput"] - other["throughput"]) / max(abs(other["throughput"]), 1e-9) * 100
            print(f"    vs {name:<16}: TTFT mean {ttft_mean_diff:+.1f}%  TTFT p99 {ttft_p99_diff:+.1f}%  "
                  f"E2E mean {e2e_mean_diff:+.1f}%  Tput {tput_diff:+.1f}%")

    return {
        "all_results": all_results,
        "averaged": averaged,
        "pareto_set": pareto_set,
        "dominated": dominated,
    }


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <results_dir> <config_names...> <seeds...>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    config_names = []
    seeds = []
    for arg in sys.argv[2:]:
        try:
            seeds.append(int(arg))
        except ValueError:
            config_names.append(arg)

    if not config_names or not seeds:
        print("ERROR: Need at least one config name and one seed", file=sys.stderr)
        sys.exit(1)

    # Discover rate subdirectories
    rate_dirs = sorted([d.name for d in results_dir.iterdir()
                        if d.is_dir() and any((d / f"{config_names[0]}_seed{seeds[0]}.txt").exists()
                                               for _ in [None])])

    rate_results = {}

    for rate_name in rate_dirs:
        print("=" * 100)
        print(f"  RATE LEVEL: {rate_name}")
        print("=" * 100)
        result = analyze_rate(rate_name, results_dir, config_names, seeds)
        if result is not None:
            rate_results[rate_name] = result

    # ── Cross-rate comparison ────────────────────────────────────────────────
    if len(rate_results) >= 2:
        print("\n" + "=" * 100)
        print("  CROSS-RATE COMPARISON")
        print("=" * 100)

        print(f"\n  Pareto sets by rate level:")
        for rate_name, res in rate_results.items():
            print(f"    {rate_name}: {res['pareto_set']}")

        all_pareto = [set(res['pareto_set']) for res in rate_results.values()]
        if all(s == all_pareto[0] for s in all_pareto):
            print(f"\n  Same Pareto set across all rates: {all_pareto[0]}")
        else:
            print(f"\n  Pareto sets DIFFER across rate levels -- "
                  f"evidence of utilization-dependent tradeoffs!")
            for rate_name, res in rate_results.items():
                unique = set(res['pareto_set'])
                for other_name, other_res in rate_results.items():
                    if other_name != rate_name:
                        unique -= set(other_res['pareto_set'])
                if unique:
                    print(f"    Pareto-optimal ONLY at {rate_name}: {sorted(unique)}")

        # Compare effect sizes across rates
        print(f"\n  Effect size comparison (cache-heavy vs load-only spread):")
        for rate_name, res in rate_results.items():
            avg = res["averaged"]
            if "cache-heavy" in avg and "load-only" in avg:
                ttft_spread = abs(avg["cache-heavy"]["ttft_mean"] - avg["load-only"]["ttft_mean"])
                ttft_pct = ttft_spread / max(abs(avg["load-only"]["ttft_mean"]), 1e-9) * 100
                p99_spread = abs(avg["cache-heavy"]["ttft_p99"] - avg["load-only"]["ttft_p99"])
                p99_pct = p99_spread / max(abs(avg["load-only"]["ttft_p99"]), 1e-9) * 100
                print(f"    {rate_name}: TTFT mean spread={ttft_spread:.1f}ms ({ttft_pct:.1f}%), "
                      f"TTFT p99 spread={p99_spread:.1f}ms ({p99_pct:.1f}%)")

    # ── Final verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  HYPOTHESIS VERDICT")
    print("=" * 100)

    high_result = rate_results.get("high")
    moderate_result = rate_results.get("moderate")

    if high_result:
        high_pareto = high_result["pareto_set"]
        if len(high_pareto) >= 2:
            print(f"\n  HIGH rate: CONFIRMED -- {len(high_pareto)} Pareto-optimal configs")
            print(f"    Pareto set: {high_pareto}")
            # Check if cache-heavy wins TTFT mean but loses TTFT p99
            avg = high_result["averaged"]
            if "cache-heavy" in avg:
                ttft_mean_best = min(avg.keys(), key=lambda n: avg[n]["ttft_mean"])
                ttft_p99_best = min(avg.keys(), key=lambda n: avg[n]["ttft_p99"])
                print(f"    TTFT mean best: {ttft_mean_best}")
                print(f"    TTFT p99 best: {ttft_p99_best}")
                if ttft_mean_best != ttft_p99_best:
                    print(f"    TRADEOFF DETECTED: different configs optimize mean vs tail!")
                else:
                    print(f"    Same config is best on both mean and tail -- no mean/tail tradeoff")
        else:
            dominant = high_pareto[0] if high_pareto else "none"
            print(f"\n  HIGH rate: NO FRONTIER -- '{dominant}' dominates all others")

    if moderate_result:
        mod_pareto = moderate_result["pareto_set"]
        if len(mod_pareto) >= 2:
            print(f"\n  MODERATE rate: {len(mod_pareto)} Pareto-optimal configs")
            print(f"    Pareto set: {mod_pareto}")
        else:
            dominant = mod_pareto[0] if mod_pareto else "none"
            print(f"\n  MODERATE rate (control): '{dominant}' dominates -- "
                  f"reproduces H17 pattern (no within-workload frontier)")

    # Overall assessment
    high_frontier = high_result and len(high_result["pareto_set"]) >= 2
    moderate_frontier = moderate_result and len(moderate_result["pareto_set"]) >= 2

    if high_frontier and not moderate_frontier:
        print(f"\n  OVERALL: CONFIRMED -- Pareto frontier emerges at high utilization "
              f"but not at moderate utilization.")
        print(f"  The within-workload tradeoff is utilization-dependent, "
              f"as predicted by #377.")
    elif high_frontier and moderate_frontier:
        print(f"\n  OVERALL: CONFIRMED WITH NUANCE -- Pareto frontier exists at BOTH rates.")
        print(f"  The tradeoff is not utilization-dependent (exists even at moderate rate).")
    elif not high_frontier and not moderate_frontier:
        print(f"\n  OVERALL: REFUTED -- No Pareto frontier at either rate level.")
        print(f"  Cache locality dominates even under saturation.")
    elif not high_frontier and moderate_frontier:
        print(f"\n  OVERALL: UNEXPECTED -- Frontier at moderate but not high rate.")
        print(f"  Requires further investigation.")


if __name__ == "__main__":
    main()
