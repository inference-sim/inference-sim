#!/usr/bin/env python3
"""Analyze elastic batching stress test results.

Computes elastic ratio (elastic_critical_P99 / large_batch_critical_P99) for each
of 8 stress variants across 3 seeds. Produces summary table with scale effect,
KV pressure effect, and asymmetric-size analysis.

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

    m = re.search(r"Avg Batch Occupancy:\s+([0-9.]+)", content)
    if m:
        result["avg_batch_occupancy"] = float(m.group(1))

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


def parse_dropped_unservable(filepath):
    """Extract dropped_unservable count from BLIS output."""
    content = Path(filepath).read_text()

    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
            if block.get("instance_id") == "cluster":
                return block.get("dropped_unservable", 0)
        except json.JSONDecodeError:
            continue
    return 0


def parse_kv_allocation_failures(filepath):
    """Extract kv_allocation_failures count from BLIS output."""
    content = Path(filepath).read_text()

    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
            if block.get("instance_id") == "cluster":
                return block.get("kv_allocation_failures", 0)
        except json.JSONDecodeError:
            continue
    return 0


# Variant metadata
VARIANTS = {
    "S1": {"instances": 2,  "rate": 150,  "kv": "default", "sizes": "standard",     "notes": "Small cluster (2 inst)"},
    "S2": {"instances": 8,  "rate": 600,  "kv": "default", "sizes": "standard",     "notes": "Large cluster (8 inst)"},
    "S3": {"instances": 4,  "rate": 300,  "kv": "5000",    "sizes": "standard",     "notes": "Moderate KV (5K)"},
    "S4": {"instances": 4,  "rate": 300,  "kv": "2000",    "sizes": "standard",     "notes": "Heavy KV (2K)"},
    "S5": {"instances": 4,  "rate": 300,  "kv": "default", "sizes": "crit=short",   "notes": "Critical=short"},
    "S6": {"instances": 4,  "rate": 300,  "kv": "default", "sizes": "crit=long",    "notes": "Critical=long"},
    "S7": {"instances": 4,  "rate": 300,  "kv": "default", "sizes": "pareto",       "notes": "ParetoLogNormal"},
    "S8": {"instances": 16, "rate": 1200, "kv": "default", "sizes": "standard",     "notes": "Very large cluster (16)"},
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
    variant_ids = [f"S{i}" for i in range(1, 9)]
    configs = ["large", "elastic"]

    # Collect per-run metrics
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
                dropped = parse_dropped_unservable(str(filepath))
                kv_failures = parse_kv_allocation_failures(str(filepath))

                entry = {
                    "seed": seed,
                    "timed_out": False,
                    "ttft_p99": base["ttft_p99"],
                    "e2e_p99": base["e2e_p99"],
                    "throughput": base["throughput"],
                    "completed": base["completed"],
                    "avg_batch_occupancy": occ["avg_batch_occupancy"],
                    "preemption_count": preemptions,
                    "dropped_unservable": dropped,
                    "kv_allocation_failures": kv_failures,
                }

                # Per-SLO critical TTFT
                if "critical" in slo:
                    entry["critical_ttft_p99"] = slo["critical"]["ttft_p99"]
                    entry["critical_ttft_mean"] = slo["critical"]["ttft_mean"]
                    entry["critical_n"] = slo["critical"]["ttft_n"]
                else:
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

        def avg(entries, key):
            vals = [e[key] for e in entries if key in e]
            return sum(vals) / len(vals) if vals else 0.0

        large_crit_p99 = avg(large_entries, "critical_ttft_p99")
        elastic_crit_p99 = avg(elastic_entries, "critical_ttft_p99")
        large_occ = avg(large_entries, "avg_batch_occupancy")
        elastic_occ = avg(elastic_entries, "avg_batch_occupancy")
        elastic_preemptions = avg(elastic_entries, "preemption_count")
        large_preemptions = avg(large_entries, "preemption_count")
        elastic_dropped = avg(elastic_entries, "dropped_unservable")
        large_dropped = avg(large_entries, "dropped_unservable")
        elastic_kv_failures = avg(elastic_entries, "kv_allocation_failures")
        large_kv_failures = avg(large_entries, "kv_allocation_failures")

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
            "large_preemptions": large_preemptions,
            "elastic_dropped": elastic_dropped,
            "large_dropped": large_dropped,
            "elastic_kv_failures": elastic_kv_failures,
            "large_kv_failures": large_kv_failures,
            "large_seeds": len(large_entries),
            "elastic_seeds": len(elastic_entries),
        })

    # Print main summary table
    print()
    print("STRESS TEST RESULTS")
    print("=" * 150)
    print()

    header = (
        f"{'Variant':<8} {'Inst':>4} {'KV Blocks':<10} {'Sizes':<14} "
        f"{'L-Crit P99':>11} {'E-Crit P99':>11} {'Elast Ratio':>12} {'Occ Ratio':>10} "
        f"{'Preempt':>8} {'KV Fails':>9} {'Dropped':>8} {'Verdict':<16}"
    )
    print(header)
    print("-" * 150)

    verdicts = {"STRONG BENEFIT": 0, "MODERATE": 0, "NO EFFECT": 0, "HARMFUL": 0}

    for r in results:
        vid = r["vid"]
        meta = VARIANTS[vid]

        if not r["valid"]:
            print(
                f"{vid:<8} {meta['instances']:>4} {meta['kv']:<10} {meta['sizes']:<14} "
                f"{'N/A':>11} {'N/A':>11} {'N/A':>12} {'N/A':>10} "
                f"{'N/A':>8} {'N/A':>9} {'N/A':>8} {'INCOMPLETE':<16}"
            )
            continue

        verdict = classify_ratio(r["elastic_ratio"])
        verdicts[verdict] += 1

        print(
            f"{vid:<8} {meta['instances']:>4} {meta['kv']:<10} {meta['sizes']:<14} "
            f"{r['large_crit_p99']:>10.0f}ms {r['elastic_crit_p99']:>10.0f}ms "
            f"{r['elastic_ratio']:>11.3f} {r['occ_ratio']:>9.3f} "
            f"{r['preemptions']:>7.0f} {r['elastic_kv_failures']:>8.0f} "
            f"{r['elastic_dropped']:>7.0f} {verdict:<16}"
        )

    print()
    print("SUMMARY:")
    total_valid = sum(verdicts.values())
    print(f"  Strong benefit (ratio < 0.80):   {verdicts['STRONG BENEFIT']:>2}/{total_valid} variants")
    print(f"  Moderate benefit (0.80-0.95):     {verdicts['MODERATE']:>2}/{total_valid} variants")
    print(f"  No effect (0.95-1.05):            {verdicts['NO EFFECT']:>2}/{total_valid} variants")
    print(f"  Harmful (> 1.05):                 {verdicts['HARMFUL']:>2}/{total_valid} variants")

    # SCALE EFFECT
    print()
    print()
    print("SCALE EFFECT")
    print("=" * 80)
    print()

    scale_variants = {"S1": 2, "S2": 8, "S8": 16}
    # Also include W2 from Iter 7 reference (4 instances, ratio ~0.196)
    print("  Instance count vs elastic ratio:")
    print()
    for vid in ["S1", "S2", "S8"]:
        r = next((x for x in results if x["vid"] == vid), None)
        if r and r["valid"]:
            print(f"    {VARIANTS[vid]['instances']:>2} instances: ratio = {r['elastic_ratio']:.3f}  "
                  f"(L={r['large_crit_p99']:.0f}ms, E={r['elastic_crit_p99']:.0f}ms)  "
                  f"preemptions={r['preemptions']:.0f}")
        else:
            print(f"    {VARIANTS[vid]['instances']:>2} instances: INCOMPLETE")

    print()
    print("  Reference from Iter 7 W2 (4 instances, 120% load): ratio ~0.196")
    print()

    # Check if benefit scales with cluster size
    scale_ratios = []
    for vid in ["S1", "S2", "S8"]:
        r = next((x for x in results if x["vid"] == vid), None)
        if r and r["valid"]:
            scale_ratios.append((VARIANTS[vid]["instances"], r["elastic_ratio"]))

    if len(scale_ratios) >= 2:
        print("  Scale trend: ", end="")
        sorted_ratios = sorted(scale_ratios, key=lambda x: x[0])
        trend_parts = [f"{inst}i={ratio:.3f}" for inst, ratio in sorted_ratios]
        print(", ".join(trend_parts))

    # KV PRESSURE EFFECT
    print()
    print()
    print("KV PRESSURE EFFECT")
    print("=" * 80)
    print()

    kv_variants = ["S3", "S4"]
    print("  KV blocks vs elastic ratio:")
    print()

    # Reference: default KV (132K) from S1/S2 or Iter 7
    print(f"    default (~132K): ratio ~0.196 (Iter 7 W2 reference)")

    for vid in kv_variants:
        r = next((x for x in results if x["vid"] == vid), None)
        if r and r["valid"]:
            print(f"    {VARIANTS[vid]['kv']:>10}: ratio = {r['elastic_ratio']:.3f}  "
                  f"(L={r['large_crit_p99']:.0f}ms, E={r['elastic_crit_p99']:.0f}ms)  "
                  f"preempt={r['preemptions']:.0f}  kv_fails={r['elastic_kv_failures']:.0f}  "
                  f"dropped={r['elastic_dropped']:.0f}")
        else:
            print(f"    {VARIANTS[vid]['kv']:>10}: INCOMPLETE")

    # KV pressure comparison: large vs elastic dropped/failures
    print()
    print("  KV pressure detail (large-batch vs elastic):")
    for vid in kv_variants:
        r = next((x for x in results if x["vid"] == vid), None)
        if r and r["valid"]:
            print(f"    {vid} ({VARIANTS[vid]['kv']} blocks):")
            print(f"      Large-batch:  preempt={r['large_preemptions']:.0f}  "
                  f"kv_fails={r['large_kv_failures']:.0f}  dropped={r['large_dropped']:.0f}")
            print(f"      Elastic:      preempt={r['preemptions']:.0f}  "
                  f"kv_fails={r['elastic_kv_failures']:.0f}  dropped={r['elastic_dropped']:.0f}")

    # ASYMMETRIC SIZE EFFECT
    print()
    print()
    print("ASYMMETRIC SIZE EFFECT")
    print("=" * 80)
    print()

    asym_variants = ["S5", "S6", "S7"]
    for vid in asym_variants:
        r = next((x for x in results if x["vid"] == vid), None)
        if r and r["valid"]:
            print(f"  {vid} ({VARIANTS[vid]['notes']}): ratio = {r['elastic_ratio']:.3f}  "
                  f"(L={r['large_crit_p99']:.0f}ms, E={r['elastic_crit_p99']:.0f}ms)  "
                  f"occ_ratio={r['occ_ratio']:.3f}")
        else:
            print(f"  {vid} ({VARIANTS[vid]['notes']}): INCOMPLETE")

    print()
    print("  Reference: standard sizes (Iter 7 W2): ratio ~0.196")

    # PER-SEED DETAIL
    print()
    print()
    print("PER-SEED DETAIL")
    print("=" * 120)
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
            kv_info = ""
            if e.get("kv_allocation_failures", 0) > 0 or e.get("dropped_unservable", 0) > 0:
                kv_info = f"  kv_fails={e.get('kv_allocation_failures', 0)}  dropped={e.get('dropped_unservable', 0)}"
            print(
                f"    large    s{e['seed']}: crit_p99={e['critical_ttft_p99']:>10.0f}ms  "
                f"occ={e['avg_batch_occupancy']:.4f}  tput={e['throughput']:.1f}/s  "
                f"preempt={e['preemption_count']}{kv_info}"
            )
        for e in elastic_entries:
            kv_info = ""
            if e.get("kv_allocation_failures", 0) > 0 or e.get("dropped_unservable", 0) > 0:
                kv_info = f"  kv_fails={e.get('kv_allocation_failures', 0)}  dropped={e.get('dropped_unservable', 0)}"
            print(
                f"    elastic  s{e['seed']}: crit_p99={e['critical_ttft_p99']:>10.0f}ms  "
                f"occ={e['avg_batch_occupancy']:.4f}  tput={e['throughput']:.1f}/s  "
                f"preempt={e['preemption_count']}{kv_info}"
            )
        print()

    # OCCUPANCY IMPACT
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
