#!/usr/bin/env python3
"""Analysis script for H17: Multi-Scorer Pareto Frontier.

Parses BLIS multi-block output and produces:
1. Per-workload, per-seed metric tables
2. Seed-averaged comparison tables
3. Pareto dominance analysis per workload
4. Cross-workload comparison (does the Pareto set change?)

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
  - Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
  - Cluster block has "instance_id": "cluster"
  - JSON fields: ttft_mean_ms, ttft_p99_ms, e2e_mean_ms, e2e_p99_ms, responses_per_sec,
    completed_requests, scheduling_delay_p99_ms
  - KV cache summary: "Preemption Rate: %.4f", "Cache Hit Rate: %.4f" (cmd/root.go:546-547)
  - Target Distribution block from trace summary (cmd/root.go:521-530)

Usage: python3 analyze.py <results_dir> <config_names...> <seeds...>
  Results directory has subdirectories per workload: prefix-heavy/, independent/
  Files expected at: <results_dir>/<workload>/<config_name>_seed<seed>.txt
"""

import json
import math
import re
import sys
from pathlib import Path


def _warn_missing(metric_name, section_header, content, filepath):
    """Warn on stderr if a section header exists but metric regex didn't match."""
    if section_header in content:
        print(f"WARNING: '{metric_name}' not found in '{filepath}' "
              f"despite '{section_header}' section being present. "
              f"Check regex against cmd/root.go format strings.",
              file=sys.stderr)


def parse_output(filepath):
    """Parse multi-block BLIS output into cluster metrics dict."""
    content = Path(filepath).read_text()

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    if cluster is None:
        print(f"WARNING: No cluster metrics block found in '{filepath}'",
              file=sys.stderr)
        return None

    # Cache hit rate (cmd/root.go:547 — "Cache Hit Rate: %.4f")
    cache_hit_rate = 0.0
    m = re.search(r"Cache Hit Rate:\s*([\d.]+)", content)
    if m:
        cache_hit_rate = float(m.group(1))
    else:
        _warn_missing("Cache Hit Rate", "=== KV Cache Metrics ===", content, filepath)

    # Target distribution from trace summary (cmd/root.go:522-530)
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

    return {
        "ttft_mean": cluster["ttft_mean_ms"],
        "ttft_p99": cluster["ttft_p99_ms"],
        "e2e_mean": cluster["e2e_mean_ms"],
        "e2e_p99": cluster["e2e_p99_ms"],
        "throughput": cluster["responses_per_sec"],
        "completed": cluster["completed_requests"],
        "cache_hit_rate": cache_hit_rate,
        "dist": dist,
        "dist_stddev": stddev,
    }


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


def analyze_workload(workload_name, results_dir, config_names, seeds):
    """Analyze a single workload, return averaged metrics dict."""
    subdir = results_dir / workload_name

    if not subdir.exists():
        print(f"\n  SKIPPED: {workload_name} (directory not found)")
        return None

    # Parse all results for this workload
    all_results = {}
    for name in config_names:
        all_results[name] = {}
        for seed in seeds:
            filepath = subdir / f"{name}_seed{seed}.txt"
            if not filepath.exists():
                print(f"WARNING: Missing file {filepath}", file=sys.stderr)
                continue
            result = parse_output(str(filepath))
            if result is not None:
                all_results[name][seed] = result

    # ── Per-seed table ──
    print(f"\n  Per-seed results:")
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        hdr = (f"  {'Configuration':<16} | {'TTFT Mean':>10} {'TTFT P99':>10} "
               f"{'E2E Mean':>10} {'E2E P99':>10} {'Tput':>8} "
               f"| {'CacheHit':>8} | Distribution")
        print(hdr)
        print(f"  {'-'*16}-+-{'-'*52}-+-{'-'*8}-+-{'-'*30}")

        for name in config_names:
            r = all_results[name].get(seed)
            if not r:
                print(f"  {name:<16} | {'MISSING':>52} |")
                continue
            dist_str = str([r['dist'].get(k, 0) for k in sorted(r['dist'].keys())])
            print(f"  {name:<16} | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f} "
                  f"{r['e2e_mean']:>10.1f} {r['e2e_p99']:>10.1f} {r['throughput']:>8.1f} "
                  f"| {r['cache_hit_rate']:>7.1%} | {dist_str}")

    # ── Seed-averaged ──
    averaged = {}
    metrics_keys = ["ttft_mean", "ttft_p99", "e2e_mean", "e2e_p99",
                    "throughput", "cache_hit_rate"]

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
        averaged[name] = avg

    print(f"\n  Seed-averaged results:")
    hdr = (f"  {'Configuration':<16} | {'TTFT Mean':>10} {'TTFT P99':>10} "
           f"{'E2E Mean':>10} {'E2E P99':>10} {'Tput':>8} | {'CacheHit':>8}")
    print(hdr)
    print(f"  {'-'*16}-+-{'-'*52}-+-{'-'*8}")

    for name in config_names:
        a = averaged.get(name)
        if not a:
            continue
        print(f"  {name:<16} | {a['ttft_mean']:>10.1f} {a['ttft_p99']:>10.1f} "
              f"{a['e2e_mean']:>10.1f} {a['e2e_p99']:>10.1f} {a['throughput']:>8.1f} "
              f"| {a['cache_hit_rate']:>7.1%}")

    # Seed variability
    print(f"\n  Seed variability (min - max across {len(seeds)} seeds):")
    for name in config_names:
        a = averaged.get(name)
        if not a:
            continue
        print(f"  {name:<16}: TTFT mean [{a['ttft_mean_min']:.1f} - {a['ttft_mean_max']:.1f}]  "
              f"E2E mean [{a['e2e_mean_min']:.1f} - {a['e2e_mean_max']:.1f}]  "
              f"Tput [{a['throughput_min']:.1f} - {a['throughput_max']:.1f}]")

    # ── Pareto dominance ──
    minimize = ["ttft_mean", "ttft_p99", "e2e_mean", "e2e_p99"]
    maximize = ["throughput"]

    pareto_set, dominated = find_pareto_set(averaged, minimize, maximize)

    print(f"\n  Pareto-optimal (non-dominated): {len(pareto_set)}")
    for name in pareto_set:
        a = averaged[name]
        print(f"    {name}: TTFT={a['ttft_mean']:.1f} E2E={a['e2e_mean']:.1f} "
              f"Tput={a['throughput']:.1f} CacheHit={a['cache_hit_rate']:.1%}")

    if dominated:
        print(f"\n  Dominated: {len(dominated)}")
        for name in sorted(dominated):
            a = averaged[name]
            dominators = []
            for other in pareto_set:
                if pareto_dominates(averaged[other], a, minimize, maximize):
                    dominators.append(other)
            print(f"    {name}: TTFT={a['ttft_mean']:.1f} E2E={a['e2e_mean']:.1f} "
                  f"Tput={a['throughput']:.1f} (dominated by: {', '.join(dominators)})")

    # Noise check: flag if any Pareto-optimal config's advantage is < 1% on the
    # metric that prevents its domination
    if len(pareto_set) >= 2:
        noise_flags = []
        for name_a in pareto_set:
            for name_b in pareto_set:
                if name_a == name_b:
                    continue
                # Find if name_a is "barely" non-dominated by name_b
                # i.e., name_b is better on all but one metric, and that margin < 1%
                advantages_a = []  # metrics where a is strictly better than b
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
                        f"({advantages_a[0][1]:.2f}% margin) — likely noise, not a real tradeoff"
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
    for mk in ["ttft_mean", "e2e_mean", "throughput"]:
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

    # Discover workload subdirectories
    workload_dirs = sorted([d.name for d in results_dir.iterdir()
                           if d.is_dir() and any((d / f"{config_names[0]}_seed{seeds[0]}.txt").exists()
                                                  for _ in [None])])

    workload_results = {}

    for wl in workload_dirs:
        print("=" * 100)
        print(f"  WORKLOAD: {wl}")
        print("=" * 100)
        result = analyze_workload(wl, results_dir, config_names, seeds)
        if result is not None:
            workload_results[wl] = result

    # ── Cross-workload comparison ────────────────────────────────────────────
    if len(workload_results) >= 2:
        print("\n" + "=" * 100)
        print("  CROSS-WORKLOAD COMPARISON")
        print("=" * 100)

        print(f"\n  Pareto sets by workload:")
        for wl, res in workload_results.items():
            print(f"    {wl}: {res['pareto_set']}")

        # Check if different workloads have different Pareto sets
        all_pareto = [set(res['pareto_set']) for res in workload_results.values()]
        if all(s == all_pareto[0] for s in all_pareto):
            print(f"\n  Same Pareto set across all workloads: {all_pareto[0]}")
        else:
            print(f"\n  Pareto sets DIFFER across workloads — evidence of workload-dependent tradeoffs!")
            # Show what changed
            for wl, res in workload_results.items():
                unique_to_wl = set(res['pareto_set'])
                for other_wl, other_res in workload_results.items():
                    if other_wl != wl:
                        unique_to_wl -= set(other_res['pareto_set'])
                if unique_to_wl:
                    print(f"    Pareto-optimal ONLY in {wl}: {sorted(unique_to_wl)}")

    # ── Final verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  HYPOTHESIS VERDICT")
    print("=" * 100)

    # The hypothesis is about whether a Pareto frontier EXISTS across the configs.
    # Check each workload independently, then aggregate.
    any_frontier = False
    for wl, res in workload_results.items():
        if len(res['pareto_set']) >= 2:
            any_frontier = True
            print(f"\n  {wl}: CONFIRMED — {len(res['pareto_set'])} Pareto-optimal configs")
        else:
            dominant = res['pareto_set'][0] if res['pareto_set'] else "none"
            print(f"\n  {wl}: NO FRONTIER — '{dominant}' dominates all others")

    if any_frontier:
        print(f"\n  OVERALL: CONFIRMED — Pareto frontier exists on at least one workload.")
        print(f"  The composable scorer framework produces meaningful tradeoffs.")
    else:
        # Check if the dominant config is the SAME across workloads
        dominant_per_wl = {wl: res['pareto_set'][0]
                          for wl, res in workload_results.items()
                          if len(res['pareto_set']) == 1}
        if len(set(dominant_per_wl.values())) > 1:
            print(f"\n  OVERALL: CONFIRMED WITH NUANCE — no within-workload frontier,")
            print(f"  but different workloads favor different configs:")
            for wl, dom in dominant_per_wl.items():
                print(f"    {wl}: {dom}")
            print(f"  This is a cross-workload Pareto frontier.")
        elif len(set(dominant_per_wl.values())) == 1:
            dominant = list(dominant_per_wl.values())[0]
            print(f"\n  OVERALL: REFUTED — '{dominant}' dominates on ALL workloads.")
            print(f"  The scoring dimensions may be redundant.")
        else:
            print(f"\n  OVERALL: INCONCLUSIVE — no clear pattern.")


if __name__ == "__main__":
    main()
