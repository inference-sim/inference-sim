#!/usr/bin/env python3
"""Analyze elastic batching generalization sweep results.

Computes elastic ratio (elastic_critical_P99 / large_batch_critical_P99) for each
of 12 workload variants across 3 seeds. Produces summary table with verdicts.

Usage: python3 analyze.py <results_dir>
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout


def parse_per_slo_metrics(filepath):
    """Extract per-SLO-class TTFT from BLIS text output.

    Returns dict: {slo_class: {"ttft_mean": float, "ttft_p99": float, ...}}
    """
    content = Path(filepath).read_text()
    slo_metrics = {}

    slo_pattern = re.compile(
        r"^\s+(\w+):\s*\n"
        r"\s+TTFT:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)\s*\n"
        r"\s+E2E:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)",
        re.MULTILINE,
    )
    for m in slo_pattern.finditer(content):
        cls = m.group(1)
        slo_metrics[cls] = {
            "ttft_mean": float(m.group(2)),
            "ttft_p99": float(m.group(3)),
            "ttft_n": int(m.group(4)),
            "e2e_mean": float(m.group(5)),
            "e2e_p99": float(m.group(6)),
            "e2e_n": int(m.group(7)),
        }
    return slo_metrics


def parse_batch_occupancy(filepath):
    """Extract batch occupancy from BLIS output."""
    content = Path(filepath).read_text()
    result = {"avg_batch_occupancy": 0.0}

    # Try text output first
    m = re.search(r"Avg Batch Occupancy:\s+([0-9.]+)", content)
    if m:
        result["avg_batch_occupancy"] = float(m.group(1))

    # Also try cluster JSON block
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
            if block.get("instance_id") == "cluster":
                if "avg_batch_occupancy" in block:
                    result["avg_batch_occupancy"] = block["avg_batch_occupancy"]
        except json.JSONDecodeError:
            continue

    return result


def parse_preemption_count(filepath):
    """Extract preemption count from BLIS output."""
    content = Path(filepath).read_text()

    # From cluster JSON block
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
            if block.get("instance_id") == "cluster":
                return block.get("preemption_count", 0)
        except json.JSONDecodeError:
            continue
    return 0


# Variant metadata for the table
VARIANTS = {
    "W1":  {"rate": 200, "load": "80%",  "arrival": "gamma2",   "session": "multi", "slo_mix": "20/40/40", "notes": "Moderate load"},
    "W2":  {"rate": 300, "load": "120%", "arrival": "gamma2",   "session": "multi", "slo_mix": "20/40/40", "notes": "Base case"},
    "W3":  {"rate": 272, "load": "80%",  "arrival": "gamma2",   "session": "single","slo_mix": "20/40/40", "notes": "Single-turn mod"},
    "W4":  {"rate": 408, "load": "120%", "arrival": "gamma2",   "session": "single","slo_mix": "20/40/40", "notes": "Single-turn OL"},
    "W5":  {"rate": 300, "load": "120%", "arrival": "gamma2",   "session": "multi", "slo_mix": "5/45/50",  "notes": "Few critical"},
    "W6":  {"rate": 300, "load": "120%", "arrival": "gamma2",   "session": "multi", "slo_mix": "50/30/20", "notes": "Many critical"},
    "W7":  {"rate": 300, "load": "120%", "arrival": "poisson",  "session": "multi", "slo_mix": "20/40/40", "notes": "Poisson"},
    "W8":  {"rate": 300, "load": "120%", "arrival": "gamma4",   "session": "multi", "slo_mix": "20/40/40", "notes": "Heavy bursts"},
    "W9":  {"rate": 500, "load": "200%", "arrival": "gamma2",   "session": "multi", "slo_mix": "20/40/40", "notes": "Extreme OL"},
    "W10": {"rate": 125, "load": "50%",  "arrival": "gamma2",   "session": "multi", "slo_mix": "20/40/40", "notes": "Light load"},
    "W11": {"rate": 300, "load": "120%", "arrival": "gamma2",   "session": "multi", "slo_mix": "10/10/80", "notes": "Sheddable-heavy"},
    "W12": {"rate": 300, "load": "120%", "arrival": "constant", "session": "multi", "slo_mix": "20/40/40", "notes": "Constant arr"},
}


def classify_ratio(ratio):
    """Classify elastic ratio into verdict."""
    if ratio < 0.80:
        return "STRONG BENEFIT"
    elif ratio < 0.95:
        return "MODERATE"
    elif ratio <= 1.05:
        return "NO EFFECT"
    else:
        return "HARMFUL"


def analyze(results_dir):
    """Main analysis."""
    results_path = Path(results_dir)
    seeds = ["42", "123", "456"]
    variant_ids = [f"W{i}" for i in range(1, 13)]
    configs = ["large", "elastic"]

    # Collect per-run metrics
    # data[variant_id][config] = list of seed dicts
    data = defaultdict(lambda: defaultdict(list))

    for vid in variant_ids:
        for config in configs:
            for seed in seeds:
                label = f"{vid}_{config}_s{seed}"
                filepath = results_path / f"{label}.txt"

                if not filepath.exists():
                    print(f"WARNING: missing {filepath}", file=sys.stderr)
                    continue

                if check_for_timeout(str(filepath)):
                    print(f"  SKIP: {label} (timeout/error)", file=sys.stderr)
                    data[vid][config].append({"timed_out": True, "seed": seed})
                    continue

                base = parse_blis_output(str(filepath))
                slo = parse_per_slo_metrics(str(filepath))
                occ = parse_batch_occupancy(str(filepath))
                preemptions = parse_preemption_count(str(filepath))

                entry = {
                    "seed": seed,
                    "timed_out": False,
                    "ttft_p99": base["ttft_p99"],
                    "e2e_p99": base["e2e_p99"],
                    "throughput": base["throughput"],
                    "completed": base["completed"],
                    "avg_batch_occupancy": occ["avg_batch_occupancy"],
                    "preemption_count": preemptions,
                }

                # Per-SLO critical TTFT
                if "critical" in slo:
                    entry["critical_ttft_p99"] = slo["critical"]["ttft_p99"]
                    entry["critical_ttft_mean"] = slo["critical"]["ttft_mean"]
                    entry["critical_n"] = slo["critical"]["ttft_n"]
                else:
                    # Fallback to cluster-wide metric
                    entry["critical_ttft_p99"] = base["ttft_p99"]
                    entry["critical_ttft_mean"] = base["ttft_mean"]
                    entry["critical_n"] = base["completed"]

                data[vid][config].append(entry)

    # Compute per-variant averages and ratios
    results = []
    for vid in variant_ids:
        large_entries = [e for e in data[vid]["large"] if not e.get("timed_out")]
        elastic_entries = [e for e in data[vid]["elastic"] if not e.get("timed_out")]

        if not large_entries or not elastic_entries:
            results.append({
                "vid": vid,
                "valid": False,
                "large_ok": len(large_entries),
                "elastic_ok": len(elastic_entries),
            })
            continue

        # Averages
        def avg(entries, key):
            vals = [e[key] for e in entries if key in e]
            return sum(vals) / len(vals) if vals else 0.0

        large_crit_p99 = avg(large_entries, "critical_ttft_p99")
        elastic_crit_p99 = avg(elastic_entries, "critical_ttft_p99")
        large_occ = avg(large_entries, "avg_batch_occupancy")
        elastic_occ = avg(elastic_entries, "avg_batch_occupancy")
        elastic_preemptions = avg(elastic_entries, "preemption_count")
        large_tput = avg(large_entries, "throughput")
        elastic_tput = avg(elastic_entries, "throughput")
        large_completed = avg(large_entries, "completed")
        elastic_completed = avg(elastic_entries, "completed")

        # Ratios
        elastic_ratio = elastic_crit_p99 / large_crit_p99 if large_crit_p99 > 0 else float("inf")
        occ_ratio = elastic_occ / large_occ if large_occ > 0 else float("inf")

        results.append({
            "vid": vid,
            "valid": True,
            "large_crit_p99": large_crit_p99,
            "elastic_crit_p99": elastic_crit_p99,
            "large_occ": large_occ,
            "elastic_occ": elastic_occ,
            "elastic_ratio": elastic_ratio,
            "occ_ratio": occ_ratio,
            "preemptions": elastic_preemptions,
            "large_tput": large_tput,
            "elastic_tput": elastic_tput,
            "large_completed": large_completed,
            "elastic_completed": elastic_completed,
            "large_seeds": len(large_entries),
            "elastic_seeds": len(elastic_entries),
        })

    # Print summary table
    print()
    print("GENERALIZATION SWEEP RESULTS")
    print("=" * 130)
    print()

    header = (
        f"{'Variant':<8} {'Load':<6} {'Arrival':<9} {'Session':<8} {'SLO Mix':<10} "
        f"{'L-Crit P99':>11} {'E-Crit P99':>11} {'Elast Ratio':>12} {'Occ Ratio':>10} "
        f"{'Preempt':>8} {'Verdict':<16}"
    )
    print(header)
    print("-" * 130)

    verdicts = {"STRONG BENEFIT": 0, "MODERATE": 0, "NO EFFECT": 0, "HARMFUL": 0}

    for r in results:
        vid = r["vid"]
        meta = VARIANTS[vid]

        if not r["valid"]:
            print(
                f"{vid:<8} {meta['load']:<6} {meta['arrival']:<9} {meta['session']:<8} {meta['slo_mix']:<10} "
                f"{'N/A':>11} {'N/A':>11} {'N/A':>12} {'N/A':>10} {'N/A':>8} {'INCOMPLETE':<16}"
            )
            continue

        verdict = classify_ratio(r["elastic_ratio"])
        verdicts[verdict] += 1

        print(
            f"{vid:<8} {meta['load']:<6} {meta['arrival']:<9} {meta['session']:<8} {meta['slo_mix']:<10} "
            f"{r['large_crit_p99']:>10.0f}ms {r['elastic_crit_p99']:>10.0f}ms "
            f"{r['elastic_ratio']:>11.3f} {r['occ_ratio']:>9.3f} "
            f"{r['preemptions']:>7.0f} {verdict:<16}"
        )

    print()
    print("SUMMARY:")
    total_valid = sum(verdicts.values())
    print(f"  Strong benefit (ratio < 0.80):   {verdicts['STRONG BENEFIT']:>2}/{total_valid} variants")
    print(f"  Moderate benefit (0.80-0.95):     {verdicts['MODERATE']:>2}/{total_valid} variants")
    print(f"  No effect (0.95-1.05):            {verdicts['NO EFFECT']:>2}/{total_valid} variants")
    print(f"  Harmful (> 1.05):                 {verdicts['HARMFUL']:>2}/{total_valid} variants")
    print()

    # Generalization boundary analysis
    print("GENERALIZATION BOUNDARIES:")
    print("-" * 60)

    # Load regime analysis
    load_groups = {"sub-saturation": [], "overload": [], "extreme": []}
    for r in results:
        if not r["valid"]:
            continue
        vid = r["vid"]
        meta = VARIANTS[vid]
        load_pct = int(meta["load"].rstrip("%"))
        if load_pct <= 80:
            load_groups["sub-saturation"].append(r)
        elif load_pct <= 150:
            load_groups["overload"].append(r)
        else:
            load_groups["extreme"].append(r)

    print()
    print("  Load regime:")
    for regime, runs in load_groups.items():
        if not runs:
            continue
        avg_ratio = sum(r["elastic_ratio"] for r in runs) / len(runs)
        ratios = [f"{r['vid']}={r['elastic_ratio']:.3f}" for r in runs]
        print(f"    {regime} ({len(runs)} variants): avg elastic_ratio={avg_ratio:.3f} ({', '.join(ratios)})")

    # Session structure analysis
    session_groups = {"multi": [], "single": []}
    for r in results:
        if not r["valid"]:
            continue
        meta = VARIANTS[r["vid"]]
        session_groups[meta["session"]].append(r)

    print()
    print("  Session structure:")
    for session, runs in session_groups.items():
        if not runs:
            continue
        avg_ratio = sum(r["elastic_ratio"] for r in runs) / len(runs)
        ratios = [f"{r['vid']}={r['elastic_ratio']:.3f}" for r in runs]
        print(f"    {session}-turn ({len(runs)} variants): avg elastic_ratio={avg_ratio:.3f} ({', '.join(ratios)})")

    # SLO mix analysis
    slo_groups = defaultdict(list)
    for r in results:
        if not r["valid"]:
            continue
        meta = VARIANTS[r["vid"]]
        slo_groups[meta["slo_mix"]].append(r)

    print()
    print("  SLO mix:")
    for mix, runs in sorted(slo_groups.items()):
        if not runs:
            continue
        avg_ratio = sum(r["elastic_ratio"] for r in runs) / len(runs)
        ratios = [f"{r['vid']}={r['elastic_ratio']:.3f}" for r in runs]
        print(f"    {mix} ({len(runs)} variants): avg elastic_ratio={avg_ratio:.3f} ({', '.join(ratios)})")

    # Arrival pattern analysis
    arrival_groups = defaultdict(list)
    for r in results:
        if not r["valid"]:
            continue
        meta = VARIANTS[r["vid"]]
        arrival_groups[meta["arrival"]].append(r)

    print()
    print("  Arrival pattern:")
    for pattern, runs in sorted(arrival_groups.items()):
        if not runs:
            continue
        avg_ratio = sum(r["elastic_ratio"] for r in runs) / len(runs)
        ratios = [f"{r['vid']}={r['elastic_ratio']:.3f}" for r in runs]
        print(f"    {pattern} ({len(runs)} variants): avg elastic_ratio={avg_ratio:.3f} ({', '.join(ratios)})")

    # Detailed per-seed breakdown
    print()
    print()
    print("PER-SEED DETAIL")
    print("=" * 100)
    print()
    for r in results:
        if not r["valid"]:
            continue
        vid = r["vid"]
        meta = VARIANTS[vid]
        large_entries = [e for e in data[vid]["large"] if not e.get("timed_out")]
        elastic_entries = [e for e in data[vid]["elastic"] if not e.get("timed_out")]

        print(f"  {vid} ({meta['notes']}):")
        for e in large_entries:
            print(
                f"    large    s{e['seed']}: crit_p99={e['critical_ttft_p99']:>10.0f}ms  "
                f"occ={e['avg_batch_occupancy']:.4f}  tput={e['throughput']:.1f}/s  "
                f"preempt={e['preemption_count']}"
            )
        for e in elastic_entries:
            print(
                f"    elastic  s{e['seed']}: crit_p99={e['critical_ttft_p99']:>10.0f}ms  "
                f"occ={e['avg_batch_occupancy']:.4f}  tput={e['throughput']:.1f}/s  "
                f"preempt={e['preemption_count']}"
            )
        print()

    # Occupancy impact analysis
    print()
    print("OCCUPANCY IMPACT")
    print("=" * 60)
    print()
    occ_deltas = []
    for r in results:
        if not r["valid"]:
            continue
        delta_pct = (r["occ_ratio"] - 1.0) * 100
        occ_deltas.append((r["vid"], r["occ_ratio"], delta_pct))
        print(f"  {r['vid']}: occ_ratio={r['occ_ratio']:.4f} ({delta_pct:+.1f}%)")

    if occ_deltas:
        avg_delta = sum(d[2] for d in occ_deltas) / len(occ_deltas)
        max_loss = min(d[2] for d in occ_deltas)
        max_gain = max(d[2] for d in occ_deltas)
        print()
        print(f"  Average occupancy change: {avg_delta:+.1f}%")
        print(f"  Range: {max_loss:+.1f}% to {max_gain:+.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)
    analyze(sys.argv[1])
