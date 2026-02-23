#!/usr/bin/env python3
"""Analysis script for H24: Combined Pathological Anomalies.

Parses BLIS output and produces comparison tables for:
  core       — Normal vs Pathological across 3 seeds
  decomposed — Routing-only vs Scheduling-only contribution (seed 42)
  per_slo    — Per-SLO class impact analysis (seed 42)

Usage:
    python3 analyze.py core <results_dir>
    python3 analyze.py decomposed <results_dir>
    python3 analyze.py per_slo <results_dir>
"""

import json
import math
import re
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout


def parse_anomalies(filepath):
    """Parse HOL blocking and priority inversions from raw output.

    These are NOT in parse_blis_output — they only appear in the
    '=== Anomaly Counters ===' section when > 0.
    Format (cmd/root.go:511-512):
        Priority Inversions: %d
        HOL Blocking Events: %d
    """
    content = Path(filepath).read_text()

    hol = 0
    m = re.search(r"HOL Blocking Events: (\d+)", content)
    if m:
        hol = int(m.group(1))

    inversions = 0
    m = re.search(r"Priority Inversions: (\d+)", content)
    if m:
        inversions = int(m.group(1))

    return hol, inversions


def parse_distribution(filepath):
    """Parse target distribution from trace summary.

    Format (cmd/root.go):
        Target Distribution:
          instance_0: 125
          instance_1: 125
          ...
    """
    content = Path(filepath).read_text()
    dist = {}
    dist_match = re.search(
        r"Target Distribution:\n((?:\s+instance_\d+: \d+\n?)+)", content
    )
    if dist_match:
        for line in dist_match.group(1).strip().split("\n"):
            parts = line.strip().split(": ")
            dist[parts[0]] = int(parts[1])

    # Compute stddev (pad to 4 instances — missing instances have 0 requests)
    num_instances = 4
    counts = [0] * num_instances
    for i, k in enumerate(sorted(dist.keys())):
        if i < num_instances:
            counts[i] = dist[k]
    mean_d = sum(counts) / len(counts) if counts else 0
    stddev = math.sqrt(sum((x - mean_d) ** 2 for x in counts) / len(counts)) if counts else 0

    return dist, stddev


def parse_per_slo(filepath):
    """Parse per-SLO class metrics from output.

    Format (cmd/root.go:579-580):
        === Per-SLO Metrics ===
          batch:
            TTFT: mean=X.XX p99=X.XX (n=N)
            E2E:  mean=X.XX p99=X.XX (n=N)

    NOTE: Per-SLO values are in ticks (microseconds) — ComputePerSLODistributions
    uses raw RequestTTFTs/RequestE2Es which are in ticks, unlike the cluster JSON
    which divides by 1000 to produce _ms fields. We convert to ms here.
    """
    content = Path(filepath).read_text()
    slo_metrics = {}

    # Match each SLO class block
    for m in re.finditer(
        r"  (\w+):\n"
        r"    TTFT: mean=([0-9.]+) p99=([0-9.]+) \(n=(\d+)\)\n"
        r"    E2E:  mean=([0-9.]+) p99=([0-9.]+) \(n=(\d+)\)",
        content,
    ):
        cls = m.group(1)
        # Convert from ticks (microseconds) to milliseconds
        slo_metrics[cls] = {
            "ttft_mean": float(m.group(2)) / 1000.0,
            "ttft_p99": float(m.group(3)) / 1000.0,
            "ttft_n": int(m.group(4)),
            "e2e_mean": float(m.group(5)) / 1000.0,
            "e2e_p99": float(m.group(6)) / 1000.0,
            "e2e_n": int(m.group(7)),
        }

    return slo_metrics


def parse_full(filepath):
    """Parse all metrics from a BLIS output file."""
    metrics = parse_blis_output(filepath)
    hol, inversions = parse_anomalies(filepath)
    dist, stddev = parse_distribution(filepath)
    per_slo = parse_per_slo(filepath)
    return {
        **metrics,
        "hol": hol,
        "inversions": inversions,
        "dist": dist,
        "stddev": stddev,
        "per_slo": per_slo,
    }


def dist_str(dist):
    """Format distribution as compact list."""
    return str([dist[k] for k in sorted(dist.keys())])


def analyze_core(results_dir):
    """Experiment 1: Normal vs Pathological across 3 seeds."""
    results_dir = Path(results_dir)
    seeds = [42, 123, 456]

    print("  Normal vs Pathological (per seed):")
    print()
    hdr = (
        f"  {'Seed':>4} {'Config':<12} | {'TTFT Mean':>10} {'TTFT P99':>10}"
        f" | {'E2E P99':>10} | {'HOL':>4} {'Inv':>4}"
        f" | {'StdDev':>8} | Distribution"
    )
    print(hdr)
    print(
        f"  {'-' * 4} {'-' * 12}-+-{'-' * 21}"
        f"-+-{'-' * 10}-+-{'-' * 9}"
        f"-+-{'-' * 8}-+-{'-' * 30}"
    )

    all_normal = []
    all_patho = []

    for seed in seeds:
        normal_file = results_dir / f"normal_{seed}.txt"
        patho_file = results_dir / f"patho_{seed}.txt"

        n = parse_full(str(normal_file))
        p = parse_full(str(patho_file))
        all_normal.append(n)
        all_patho.append(p)

        for config, label, r in [("normal", "normal", n), ("patho", "pathological", p)]:
            if r["timed_out"]:
                print(f"  {seed:>4} {label:<12} | TIMED OUT")
                continue
            print(
                f"  {seed:>4} {label:<12} |"
                f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['e2e_p99']:>10.1f}"
                f" | {r['hol']:>4} {r['inversions']:>4}"
                f" | {r['stddev']:>8.1f} | {dist_str(r['dist'])}"
            )

        # Print ratio
        if not n["timed_out"] and not p["timed_out"] and n["ttft_p99"] > 0:
            ratio = p["ttft_p99"] / n["ttft_p99"]
            print(
                f"       {'':12} |"
                f" {'Effect:':>10} {ratio:>9.1f}x worse TTFT P99"
            )
        print()

    # Summary: consistency check
    print("  Summary across seeds:")
    valid_normal = [r for r in all_normal if not r["timed_out"]]
    valid_patho = [r for r in all_patho if not r["timed_out"]]
    if valid_normal and valid_patho:
        patho_hol_all = all(r["hol"] > 0 for r in valid_patho)
        patho_inv_all = all(r["inversions"] > 0 for r in valid_patho)
        normal_hol_all = all(r["hol"] == 0 for r in valid_normal)
        print(f"    Pathological HOL blocking > 0 in all seeds: {patho_hol_all}")
        print(f"    Pathological inversions > 0 in all seeds:   {patho_inv_all}")
        print(f"    Normal HOL blocking == 0 in all seeds:      {normal_hol_all}")
        avg_ratio = sum(
            p["ttft_p99"] / n["ttft_p99"]
            for n, p in zip(valid_normal, valid_patho)
            if n["ttft_p99"] > 0
        ) / len(valid_normal)
        print(f"    Average TTFT P99 degradation ratio:         {avg_ratio:.1f}x")
    print()


def analyze_decomposed(results_dir):
    """Experiment 2: Routing-only vs Scheduling-only contribution."""
    results_dir = Path(results_dir)

    configs = [
        ("normal_42.txt", "Normal (all correct)"),
        ("routing_only_42.txt", "Routing-only pathological"),
        ("sched_only_42.txt", "Scheduling-only pathological"),
        ("patho_42.txt", "All pathological"),
    ]

    print(
        f"  {'Configuration':<30} | {'TTFT P99':>10}"
        f" | {'E2E P99':>10}"
        f" | {'HOL':>4} {'Inv':>4}"
        f" | {'StdDev':>8} | Distribution"
    )
    print(
        f"  {'-' * 30}-+-{'-' * 10}"
        f"-+-{'-' * 10}"
        f"-+-{'-' * 9}"
        f"-+-{'-' * 8}-+-{'-' * 30}"
    )

    parsed = {}
    for filename, label in configs:
        filepath = results_dir / filename
        r = parse_full(str(filepath))
        parsed[filename] = r
        if r["timed_out"]:
            print(f"  {label:<30} | TIMED OUT")
            continue
        print(
            f"  {label:<30} |"
            f" {r['ttft_p99']:>10.1f}"
            f" | {r['e2e_p99']:>10.1f}"
            f" | {r['hol']:>4} {r['inversions']:>4}"
            f" | {r['stddev']:>8.1f} | {dist_str(r['dist'])}"
        )

    # Attribution
    normal = parsed.get("normal_42.txt")
    routing = parsed.get("routing_only_42.txt")
    sched = parsed.get("sched_only_42.txt")
    patho = parsed.get("patho_42.txt")
    if all(r and not r["timed_out"] for r in [normal, routing, sched, patho]):
        print()
        print("  Attribution (TTFT P99 delta from normal):")
        normal_p99 = normal["ttft_p99"]
        if normal_p99 > 0:
            routing_delta = routing["ttft_p99"] - normal_p99
            sched_delta = sched["ttft_p99"] - normal_p99
            combined_delta = patho["ttft_p99"] - normal_p99
            print(f"    Routing only:    +{routing_delta:>10.1f} ms ({routing_delta / normal_p99 * 100:>6.1f}%)")
            print(f"    Scheduling only: +{sched_delta:>10.1f} ms ({sched_delta / normal_p99 * 100:>6.1f}%)")
            print(f"    Combined:        +{combined_delta:>10.1f} ms ({combined_delta / normal_p99 * 100:>6.1f}%)")
            superadditivity = combined_delta - (routing_delta + sched_delta)
            print(f"    Super-additivity: {superadditivity:>+10.1f} ms (combined - sum of parts)")
    print()


def analyze_per_slo(results_dir):
    """Experiment 3: Per-SLO class impact analysis."""
    results_dir = Path(results_dir)

    normal = parse_full(str(results_dir / "normal_42.txt"))
    patho = parse_full(str(results_dir / "patho_42.txt"))

    if normal["timed_out"] or patho["timed_out"]:
        print("  SKIPPED — timeout in seed 42 runs")
        return

    slo_classes = sorted(set(list(normal["per_slo"].keys()) + list(patho["per_slo"].keys())))

    if not slo_classes:
        print("  No per-SLO metrics found (single SLO class?)")
        return

    print(
        f"  {'SLO Class':<12} | {'Normal TTFT P99':>16} {'Patho TTFT P99':>16} {'Ratio':>8}"
        f" | {'Normal E2E P99':>16} {'Patho E2E P99':>16} {'Ratio':>8}"
    )
    print(
        f"  {'-' * 12}-+-{'-' * 16} {'-' * 16} {'-' * 8}"
        f"-+-{'-' * 16} {'-' * 16} {'-' * 8}"
    )

    for cls in slo_classes:
        n_slo = normal["per_slo"].get(cls, {})
        p_slo = patho["per_slo"].get(cls, {})
        n_ttft_p99 = n_slo.get("ttft_p99", 0)
        p_ttft_p99 = p_slo.get("ttft_p99", 0)
        n_e2e_p99 = n_slo.get("e2e_p99", 0)
        p_e2e_p99 = p_slo.get("e2e_p99", 0)

        ttft_ratio = p_ttft_p99 / n_ttft_p99 if n_ttft_p99 > 0 else float("inf")
        e2e_ratio = p_e2e_p99 / n_e2e_p99 if n_e2e_p99 > 0 else float("inf")

        print(
            f"  {cls:<12} |"
            f" {n_ttft_p99:>16.1f} {p_ttft_p99:>16.1f} {ttft_ratio:>7.1f}x"
            f" | {n_e2e_p99:>16.1f} {p_e2e_p99:>16.1f} {e2e_ratio:>7.1f}x"
        )

    print()
    print("  Expected: realtime class should be hurt most (short requests,")
    print("  latency-sensitive, penalized by inverted-slo deprioritization)")
    print()


ANALYZERS = {
    "core": analyze_core,
    "decomposed": analyze_decomposed,
    "per_slo": analyze_per_slo,
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <experiment-type> <results_dir>")
        print(f"Types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    experiment_type = sys.argv[1]
    results_dir = sys.argv[2]

    analyzer = ANALYZERS.get(experiment_type)
    if not analyzer:
        print(f"Unknown experiment type: {experiment_type}")
        print(f"Valid types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    analyzer(results_dir)
