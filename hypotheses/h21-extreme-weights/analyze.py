#!/usr/bin/env python3
"""Analysis script for H21: Extreme Scorer Weights.

Compares extreme weight ratio (prefix-affinity:100,queue-depth:1) against
single-scorer routing (prefix-affinity:1 alone) to test whether 100:1 weight
ratio behaves identically to single-scorer.

Round 2: Added control analysis for configs C (weight-sensitivity) and D (zero-prefix).

Equivalence threshold: within 5% on TTFT mean and p99.
"""
import re
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output


NUM_INSTANCES = 4


def parse_target_distribution(filepath):
    """Extract target distribution from trace summary."""
    content = Path(filepath).read_text()
    dist = {}
    for m in re.finditer(r"  (instance[_-]\d+): (\d+)", content):
        dist[m.group(1)] = int(m.group(2))
    return dist


def jain_fairness(values):
    """Compute Jain's fairness index for a list of values."""
    if not values or all(v == 0 for v in values):
        return 0.0
    n = len(values)
    total = sum(values)
    sum_sq = sum(v * v for v in values)
    if sum_sq == 0:
        return 0.0
    return (total * total) / (n * sum_sq)


def padded_dist_values(dist):
    """Return list of request counts padded to NUM_INSTANCES (missing = 0)."""
    all_inst = sorted(set(list(dist.keys()) + [f"instance_{i}" for i in range(NUM_INSTANCES)]))
    return [dist.get(inst, 0) for inst in all_inst], all_inst


def print_pairwise_table(label_a, label_b, data_a, data_b, dist_a, dist_b, seeds):
    """Print TTFT/E2E/throughput comparison and target distribution for two configs."""
    # TTFT
    print(f"  {'Seed':<8} {label_a+' mean':>12} {label_b+' mean':>12} {'Diff%':>10}"
          f"   {label_a+' p99':>12} {label_b+' p99':>12} {'Diff%':>10}")
    print(f"  {'----':<8} {'------':>12} {'------':>12} {'-----':>10}"
          f"   {'-----':>12} {'-----':>12} {'-----':>10}")

    mean_diffs = []
    p99_diffs = []
    for seed in seeds:
        a = data_a[seed]
        b = data_b[seed]
        md = ((b["ttft_mean"] - a["ttft_mean"]) / a["ttft_mean"] * 100) if a["ttft_mean"] > 0 else 0
        pd = ((b["ttft_p99"] - a["ttft_p99"]) / a["ttft_p99"] * 100) if a["ttft_p99"] > 0 else 0
        mean_diffs.append(abs(md))
        p99_diffs.append(abs(pd))
        print(f"  {seed:<8} {a['ttft_mean']:>12.2f} {b['ttft_mean']:>12.2f} {md:>+10.1f}%"
              f"   {a['ttft_p99']:>12.2f} {b['ttft_p99']:>12.2f} {pd:>+10.1f}%")
    print()
    return mean_diffs, p99_diffs


def print_dist_table(configs, labels, dists, seeds):
    """Print target distribution comparison for N configs."""
    for seed in seeds:
        print(f"  Seed {seed}:")
        # Collect all instance names across all configs
        all_inst = set()
        for cfg in configs:
            all_inst.update(dists[cfg][seed].keys())
        for i in range(NUM_INSTANCES):
            all_inst.add(f"instance_{i}")
        all_inst = sorted(all_inst)

        header = f"    {'Instance':<15}"
        for label in labels:
            header += f" {label:>10}"
        print(header)
        print(f"    {'--------':<15}" + " ----------" * len(labels))

        vals_per_cfg = []
        for cfg in configs:
            vals = [dists[cfg][seed].get(inst, 0) for inst in all_inst]
            vals_per_cfg.append(vals)

        for inst in all_inst:
            row = f"    {inst:<15}"
            for cfg in configs:
                row += f" {dists[cfg][seed].get(inst, 0):>10}"
            print(row)

        jain_row = f"    {'Jain fairness:':<15}"
        for vals in vals_per_cfg:
            jain_row += f" {jain_fairness(vals):>10.4f}"
        print(jain_row)
        print()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    seeds = [42, 123, 456]

    # Config names and file prefixes
    configs = {
        "a": "config_a",   # prefix-affinity:100,queue-depth:1 + prefix workload
        "b": "config_b",   # prefix-affinity:1 + prefix workload
        "c": "config_c",   # prefix-affinity:100,queue-depth:0.001 + prefix workload (Round 2)
        "d1": "config_d1", # prefix-affinity:100,queue-depth:1 + no-prefix workload (Round 2)
        "d2": "config_d2", # prefix-affinity:1 + no-prefix workload (Round 2)
    }

    data = {}   # config -> seed -> metrics
    dists = {}  # config -> seed -> target distribution

    for cfg, prefix in configs.items():
        data[cfg] = {}
        dists[cfg] = {}
        for seed in seeds:
            fpath = results_dir / f"{prefix}_seed{seed}.txt"
            data[cfg][seed] = parse_blis_output(str(fpath))
            dists[cfg][seed] = parse_target_distribution(str(fpath))

    # Check for timeouts
    any_timeout = False
    for cfg in configs:
        for seed in seeds:
            if data[cfg][seed]["timed_out"]:
                print(f"WARNING: Config {cfg} seed {seed} timed out", file=sys.stderr)
                any_timeout = True

    if any_timeout:
        print("ERROR: Some runs timed out, cannot produce valid comparison")
        sys.exit(1)

    # =====================================================================
    # Section 1: Original A vs B comparison
    # =====================================================================
    print("=" * 78)
    print("H21: Extreme Scorer Weights — Comparison Results (Round 2)")
    print("=" * 78)
    print()
    print("Config A: prefix-affinity:100,queue-depth:1    (PA=0.990, QD=0.010)")
    print("Config B: prefix-affinity:1                    (single scorer)")
    print("Config C: prefix-affinity:100,queue-depth:0.001 (PA=0.99999, QD=0.00001)")
    print("Config D1: Config A scorers + no-prefix workload")
    print("Config D2: Config B scorers + no-prefix workload")
    print()

    print("## 1. TTFT Comparison: A vs B (ms) — primary hypothesis test")
    print()
    ab_mean_diffs, ab_p99_diffs = print_pairwise_table(
        "A", "B", data["a"], data["b"], dists["a"], dists["b"], seeds)

    print("## 2. E2E Latency: A vs B (ms)")
    print()
    print(f"  {'Seed':<8} {'A mean':>10} {'B mean':>10} {'Diff%':>10}   {'A p99':>10} {'B p99':>10} {'Diff%':>10}")
    print(f"  {'----':<8} {'------':>10} {'------':>10} {'-----':>10}   {'-----':>10} {'-----':>10} {'-----':>10}")
    for seed in seeds:
        a, b = data["a"][seed], data["b"][seed]
        md = ((b["e2e_mean"] - a["e2e_mean"]) / a["e2e_mean"] * 100) if a["e2e_mean"] > 0 else 0
        pd = ((b["e2e_p99"] - a["e2e_p99"]) / a["e2e_p99"] * 100) if a["e2e_p99"] > 0 else 0
        print(f"  {seed:<8} {a['e2e_mean']:>10.2f} {b['e2e_mean']:>10.2f} {md:>+10.1f}%"
              f"   {a['e2e_p99']:>10.2f} {b['e2e_p99']:>10.2f} {pd:>+10.1f}%")
    print()

    print("## 3. Throughput: A vs B (req/s)")
    print()
    print(f"  {'Seed':<8} {'A':>10} {'B':>10} {'Diff%':>10}")
    print(f"  {'----':<8} {'------':>10} {'------':>10} {'-----':>10}")
    for seed in seeds:
        a, b = data["a"][seed], data["b"][seed]
        d = ((b["throughput"] - a["throughput"]) / a["throughput"] * 100) if a["throughput"] > 0 else 0
        print(f"  {seed:<8} {a['throughput']:>10.2f} {b['throughput']:>10.2f} {d:>+10.1f}%")
    print()

    # =====================================================================
    # Section 4: Target distribution — all configs with prefix workload
    # =====================================================================
    print("## 4. Target Distribution: A vs B vs C (prefix workload)")
    print()
    print_dist_table(["a", "b", "c"], ["A(100:1)", "B(1)", "C(100:0.001)"], dists, seeds)

    print("## 5. Cache Hit Rate: A vs B vs C")
    print()
    print(f"  {'Seed':<8} {'A':>10} {'B':>10} {'C':>10}")
    print(f"  {'----':<8} {'------':>10} {'------':>10} {'------':>10}")
    for seed in seeds:
        print(f"  {seed:<8} {data['a'][seed]['cache_hit_rate']:>10.4f}"
              f" {data['b'][seed]['cache_hit_rate']:>10.4f}"
              f" {data['c'][seed]['cache_hit_rate']:>10.4f}")
    print()

    # =====================================================================
    # Section 6: Control A3a — weight sensitivity (A vs C)
    # =====================================================================
    print("=" * 78)
    print("## 6. Control A3a: Weight Sensitivity (A vs C)")
    print("  If mechanism is correct: C should match A (any QD weight prevents cascade)")
    print()
    ac_mean_diffs, ac_p99_diffs = print_pairwise_table(
        "A", "C", data["a"], data["c"], dists["a"], dists["c"], seeds)

    avg_ac_mean = sum(ac_mean_diffs) / len(ac_mean_diffs)
    avg_ac_p99 = sum(ac_p99_diffs) / len(ac_p99_diffs)
    print(f"  Avg |A-C TTFT mean diff|: {avg_ac_mean:.1f}%")
    print(f"  Avg |A-C TTFT p99 diff|:  {avg_ac_p99:.1f}%")
    ac_equivalent = avg_ac_mean <= 5.0 and avg_ac_p99 <= 5.0
    if ac_equivalent:
        print("  CONTROL RESULT: A ~ C (equivalent) -- confirms any QD weight prevents cascade")
    else:
        print("  CONTROL RESULT: A != C -- weight magnitude matters beyond mere presence")
    print()

    # =====================================================================
    # Section 7: Control A3b — zero-prefix-sharing (D1 vs D2)
    # =====================================================================
    print("=" * 78)
    print("## 7. Control A3b: Zero-Prefix-Sharing (D1 vs D2)")
    print("  If mechanism is correct: D1 ~ D2 (no prefix cascade, both degenerate)")
    print()
    print("### TTFT: D1 vs D2 (ms)")
    print()
    d_mean_diffs, d_p99_diffs = print_pairwise_table(
        "D1", "D2", data["d1"], data["d2"], dists["d1"], dists["d2"], seeds)

    print("### Target Distribution: D1 vs D2 (no-prefix workload)")
    print()
    print_dist_table(["d1", "d2"], ["D1(100:1)", "D2(1)"], dists, seeds)

    avg_d_mean = sum(d_mean_diffs) / len(d_mean_diffs)
    avg_d_p99 = sum(d_p99_diffs) / len(d_p99_diffs)
    print(f"  Avg |D1-D2 TTFT mean diff|: {avg_d_mean:.1f}%")
    print(f"  Avg |D1-D2 TTFT p99 diff|:  {avg_d_p99:.1f}%")
    d_equivalent = avg_d_mean <= 5.0 and avg_d_p99 <= 5.0
    if d_equivalent:
        print("  CONTROL RESULT: D1 ~ D2 (equivalent) -- no prefix sharing = no cascade difference")
    else:
        print("  CONTROL RESULT: D1 != D2 -- QD tiebreaker matters even without prefix sharing")
        print("    This isolates the tiebreaker from the cascade: with unique requests,")
        print("    prefix-affinity scores are always 0.0 for all instances (no match),")
        print("    so the queue-depth scorer is the ONLY differentiating signal.")
    print()

    # =====================================================================
    # Overall verdict
    # =====================================================================
    threshold = 5.0
    avg_ab_mean = sum(ab_mean_diffs) / len(ab_mean_diffs)
    avg_ab_p99 = sum(ab_p99_diffs) / len(ab_p99_diffs)

    print("=" * 78)
    print("## Overall Verdict")
    print()
    print(f"  Equivalence threshold: {threshold}%")
    print(f"  Avg |A-B TTFT mean diff|: {avg_ab_mean:.1f}%")
    print(f"  Avg |A-B TTFT p99 diff|:  {avg_ab_p99:.1f}%")
    print()

    if avg_ab_mean <= threshold and avg_ab_p99 <= threshold:
        print("  PRIMARY: EQUIVALENT -- hypothesis CONFIRMED")
    else:
        print("  PRIMARY: NOT EQUIVALENT -- hypothesis REFUTED")
    print()

    print(f"  Control A3a (A vs C): {'PASS' if ac_equivalent else 'FAIL'}"
          f" (mean: {avg_ac_mean:.1f}%, p99: {avg_ac_p99:.1f}%)")
    print(f"  Control A3b (D1 vs D2): {'PASS' if d_equivalent else 'FAIL'}"
          f" (mean: {avg_d_mean:.1f}%, p99: {avg_d_p99:.1f}%)")
    print()

    if not (avg_ab_mean <= threshold and avg_ab_p99 <= threshold):
        print("  Root cause: Configs A and B differ in scorer COUNT (2 vs 1), not just")
        print("  weight ratio. The queue-depth scorer at any positive weight provides a")
        print("  load-balancing tiebreaker via the observer-seeded prefix-affinity feedback")
        print("  loop. Single-scorer has NO tiebreaker, causing all-to-one concentration.")

    print("=" * 78)


if __name__ == "__main__":
    main()
