#!/usr/bin/env python3
"""Analysis script for H20: Heavy-Tailed Input Distributions.

Parses BLIS output for Gaussian and ParetoLogNormal input distributions
across multiple seeds and produces comparison tables.

Hypothesis: ParetoLogNormal (heavy-tailed) input distributions produce more
preemptions and higher TTFT p99 than Gaussian at the same mean input length.

Experiments:
  Exp 1: Core comparison -- rate=1000, 500 requests, KV=2000 blocks
  Exp 2: Sub-saturation control -- rate=200, 500 requests, KV=2000 blocks
  Exp 3: KV-abundant control -- rate=1000, 500 requests, KV=100000 blocks

Usage: python3 analyze.py <results_dir>
"""
import json
import re
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout


def extract_dropped_unservable(filepath):
    """Extract dropped_unservable from cluster JSON block.

    The shared parse_blis_output helper does not extract this field,
    so we parse it directly from the cluster-level JSON block.
    Field reference: sim/metrics_utils.go:75 — `json:"dropped_unservable"`.
    """
    path = Path(filepath)
    if not path.exists():
        return 0
    content = path.read_text()
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

SEEDS = [42, 123, 456]


def load_experiment(results_dir, prefix):
    """Load Gaussian and ParetoLogNormal results for all seeds.

    Also extracts dropped_unservable (not in shared helper) for full INV-1 check.
    """
    gaussian = {}
    pareto = {}
    for seed in SEEDS:
        g_file = Path(results_dir) / f"{prefix}_gaussian_{seed}.txt"
        p_file = Path(results_dir) / f"{prefix}_pareto_{seed}.txt"
        gaussian[seed] = parse_blis_output(str(g_file))
        gaussian[seed]["dropped_unservable"] = extract_dropped_unservable(str(g_file))
        pareto[seed] = parse_blis_output(str(p_file))
        pareto[seed]["dropped_unservable"] = extract_dropped_unservable(str(p_file))
    return gaussian, pareto


def safe_ratio(a, b, default=float("inf")):
    """Compute a/b safely, returning default if b is zero or near-zero."""
    if b < 0.001:
        return default
    return a / b


def print_comparison_table(gaussian, pareto, title):
    """Print per-seed comparison of key metrics."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()

    fmt = "{:<8} {:<14} {:>12} {:>12} {:>12} {:>12} {:>10}"
    print(fmt.format("Seed", "Distribution", "TTFT mean", "TTFT p99",
                      "E2E mean", "E2E p99", "Preempt#"))
    print("-" * 90)

    for seed in SEEDS:
        g = gaussian[seed]
        p = pareto[seed]
        if g["timed_out"]:
            print(f"  Seed {seed}: Gaussian TIMED OUT -- skipping")
            continue
        if p["timed_out"]:
            print(f"  Seed {seed}: ParetoLN TIMED OUT -- skipping")
            continue

        print(fmt.format(seed, "Gaussian",
                          f"{g['ttft_mean']:.2f}",
                          f"{g['ttft_p99']:.2f}",
                          f"{g['e2e_mean']:.2f}",
                          f"{g['e2e_p99']:.2f}",
                          str(g['preemption_count'])))
        print(fmt.format("", "ParetoLN",
                          f"{p['ttft_mean']:.2f}",
                          f"{p['ttft_p99']:.2f}",
                          f"{p['e2e_mean']:.2f}",
                          f"{p['e2e_p99']:.2f}",
                          str(p['preemption_count'])))

        ttft_ratio = safe_ratio(p["ttft_p99"], g["ttft_p99"])
        e2e_ratio = safe_ratio(p["e2e_p99"], g["e2e_p99"])
        preempt_g = g["preemption_count"]
        preempt_p = p["preemption_count"]
        if preempt_g > 0:
            preempt_ratio_str = f"{safe_ratio(preempt_p, preempt_g):.2f}x"
        elif preempt_p > 0:
            preempt_ratio_str = "inf"
        else:
            preempt_ratio_str = "N/A"

        print(fmt.format("", "Ratio(P/G)",
                          f"{safe_ratio(p['ttft_mean'], g['ttft_mean']):.2f}x",
                          f"{ttft_ratio:.2f}x",
                          f"{safe_ratio(p['e2e_mean'], g['e2e_mean']):.2f}x",
                          f"{e2e_ratio:.2f}x",
                          preempt_ratio_str))
        print()


def print_conservation_check(datasets):
    """Verify INV-1 for all runs.

    Full INV-1: injected == completed + still_queued + still_running + dropped_unservable
    (Round 2 fix: includes dropped_unservable per docs/standards/invariants.md)
    """
    print("=" * 90)
    print("  CONSERVATION CHECK (INV-1)")
    print("  Formula: injected == completed + queued + running + dropped_unservable")
    print("=" * 90)
    print()

    all_pass = True
    for exp_label, gaussian, pareto in datasets:
        for dist_label, data in [("Gaussian", gaussian), ("ParetoLN", pareto)]:
            for seed in SEEDS:
                m = data[seed]
                if m["timed_out"]:
                    print(f"  [{exp_label}] {dist_label} seed={seed}: SKIPPED (timeout)")
                    continue
                injected = m["injected"]
                completed = m["completed"]
                queued = m["still_queued"]
                running = m["still_running"]
                dropped = m["dropped_unservable"]
                conserved = (completed + queued + running + dropped) == injected
                status = "PASS" if conserved else "FAIL"
                if not conserved:
                    all_pass = False
                print(f"  [{exp_label}] {dist_label} seed={seed}: "
                      f"injected={injected}, completed={completed}, "
                      f"queued={queued}, running={running}, "
                      f"dropped={dropped} -> {status}")

    print()
    if all_pass:
        print("  OVERALL: ALL CONSERVATION CHECKS PASS")
    else:
        print("  OVERALL: SOME CONSERVATION CHECKS FAILED")
    print()


def compute_summary_stats(gaussian, pareto):
    """Compute cross-seed summary. Returns dict or None if all timed out."""
    valid_seeds = []
    ttft_p99_g = []
    ttft_p99_p = []
    e2e_p99_g = []
    e2e_p99_p = []
    ttft_mean_g = []
    ttft_mean_p = []
    preempt_g = []
    preempt_p = []

    for seed in SEEDS:
        g = gaussian[seed]
        p = pareto[seed]
        if g["timed_out"] or p["timed_out"]:
            continue
        valid_seeds.append(seed)
        ttft_p99_g.append(g["ttft_p99"])
        ttft_p99_p.append(p["ttft_p99"])
        e2e_p99_g.append(g["e2e_p99"])
        e2e_p99_p.append(p["e2e_p99"])
        ttft_mean_g.append(g["ttft_mean"])
        ttft_mean_p.append(p["ttft_mean"])
        preempt_g.append(g["preemption_count"])
        preempt_p.append(p["preemption_count"])

    if not valid_seeds:
        return None

    n = len(valid_seeds)
    per_seed_ttft_ratios = [safe_ratio(p, g) for p, g in zip(ttft_p99_p, ttft_p99_g)]
    per_seed_preempt_ratios = [
        safe_ratio(p, g) if g > 0 else (float("inf") if p > 0 else 1.0)
        for p, g in zip(preempt_p, preempt_g)
    ]
    pareto_more_preemptions = sum(1 for p, g in zip(preempt_p, preempt_g) if p > g)
    pareto_worse_ttft = sum(1 for p, g in zip(ttft_p99_p, ttft_p99_g) if p > g)

    return {
        "n": n, "total": len(SEEDS),
        "valid_seeds": valid_seeds,
        "pareto_worse_ttft": pareto_worse_ttft,
        "pareto_more_preemptions": pareto_more_preemptions,
        "avg_ttft_ratio": sum(per_seed_ttft_ratios) / n,
        "min_ttft_ratio": min(per_seed_ttft_ratios),
        "max_ttft_ratio": max(per_seed_ttft_ratios),
        "per_seed_ttft_ratios": per_seed_ttft_ratios,
        "per_seed_preempt_ratios": per_seed_preempt_ratios,
        "avg_preempt_g": sum(preempt_g) / n,
        "avg_preempt_p": sum(preempt_p) / n,
        "ttft_mean_g": sum(ttft_mean_g) / n,
        "ttft_mean_p": sum(ttft_mean_p) / n,
        "ttft_p99_g": sum(ttft_p99_g) / n,
        "ttft_p99_p": sum(ttft_p99_p) / n,
        "e2e_p99_g": sum(e2e_p99_g) / n,
        "e2e_p99_p": sum(e2e_p99_p) / n,
        "preempt_g_list": preempt_g,
        "preempt_p_list": preempt_p,
    }


def print_summary(label, stats):
    """Print cross-seed summary for one experiment."""
    print(f"  --- {label} ---")
    if stats is None:
        print("  No valid results (all timed out).")
        print()
        return

    n = stats["n"]
    print(f"  Valid seeds: {n}/{stats['total']}")
    print(f"  Seeds where ParetoLN TTFT p99 > Gaussian: {stats['pareto_worse_ttft']}/{n}")
    print(f"  Seeds where ParetoLN preemptions > Gaussian: {stats['pareto_more_preemptions']}/{n}")
    print(f"  TTFT p99 ratio (ParetoLN/Gaussian): avg={stats['avg_ttft_ratio']:.3f}x "
          f"range=[{stats['min_ttft_ratio']:.3f}x, {stats['max_ttft_ratio']:.3f}x]")
    print(f"  Avg preemption count: Gaussian={stats['avg_preempt_g']:.1f}, "
          f"ParetoLN={stats['avg_preempt_p']:.1f}")
    print(f"  Per-seed preemption counts:")
    for i, seed in enumerate(stats["valid_seeds"]):
        print(f"    Seed {seed}: Gaussian={stats['preempt_g_list'][i]}, "
              f"ParetoLN={stats['preempt_p_list'][i]}")
    print()
    print(f"  Cross-seed averages:")
    print(f"    Gaussian TTFT mean: {stats['ttft_mean_g']:.2f} ms")
    print(f"    ParetoLN TTFT mean: {stats['ttft_mean_p']:.2f} ms")
    print(f"    Gaussian TTFT p99:  {stats['ttft_p99_g']:.2f} ms")
    print(f"    ParetoLN TTFT p99:  {stats['ttft_p99_p']:.2f} ms")
    print(f"    Gaussian E2E p99:   {stats['e2e_p99_g']:.2f} ms")
    print(f"    ParetoLN E2E p99:   {stats['e2e_p99_p']:.2f} ms")
    print()


def print_preemption_detail(datasets):
    """Print preemption and cache hit details across all experiments."""
    print("=" * 90)
    print("  PREEMPTION AND CACHE DETAILS")
    print("=" * 90)
    print()

    fmt = "{:<6} {:<6} {:<14} {:>12} {:>10} {:>12} {:>12}"
    print(fmt.format("Exp", "Seed", "Distribution", "Throughput", "Completed",
                      "Preempt#", "Cache Hit"))
    print("-" * 90)

    for exp_label, gaussian, pareto in datasets:
        for seed in SEEDS:
            g = gaussian[seed]
            p = pareto[seed]
            if g["timed_out"] or p["timed_out"]:
                continue

            print(fmt.format(exp_label, seed, "Gaussian",
                              f"{g['throughput']:.2f}",
                              str(g['completed']),
                              str(g['preemption_count']),
                              f"{g['cache_hit_rate']:.4f}"))
            print(fmt.format("", "", "ParetoLN",
                              f"{p['throughput']:.2f}",
                              str(p['completed']),
                              str(p['preemption_count']),
                              f"{p['cache_hit_rate']:.4f}"))
        print()


def print_verdict(exp1_stats, exp2_stats, exp3_stats):
    """Print overall hypothesis verdict."""
    print("=" * 90)
    print("  OVERALL VERDICT")
    print("=" * 90)
    print()

    # Exp 1: Core
    if exp1_stats:
        n = exp1_stats["n"]
        ttft_wins = exp1_stats["pareto_worse_ttft"]
        preempt_wins = exp1_stats["pareto_more_preemptions"]
        avg_ttft = exp1_stats["avg_ttft_ratio"]
        mn_ttft = exp1_stats["min_ttft_ratio"]
        mx_ttft = exp1_stats["max_ttft_ratio"]
        avg_pre_g = exp1_stats["avg_preempt_g"]
        avg_pre_p = exp1_stats["avg_preempt_p"]
        print(f"  Exp 1 (rate=1000, KV=2000, 500 req):")
        print(f"    TTFT p99: ParetoLN worse in {ttft_wins}/{n} seeds, "
              f"avg ratio={avg_ttft:.3f}x, range=[{mn_ttft:.3f}x, {mx_ttft:.3f}x]")
        print(f"    Preemptions: ParetoLN more in {preempt_wins}/{n} seeds, "
              f"avg Gaussian={avg_pre_g:.1f} vs ParetoLN={avg_pre_p:.1f}")

    # Exp 2: Sub-saturation control
    if exp2_stats:
        n = exp2_stats["n"]
        avg_ttft = exp2_stats["avg_ttft_ratio"]
        avg_pre_g = exp2_stats["avg_preempt_g"]
        avg_pre_p = exp2_stats["avg_preempt_p"]
        print(f"  Exp 2 (rate=200, KV=2000, 500 req): sub-saturation control")
        print(f"    TTFT p99 ratio: {avg_ttft:.3f}x")
        print(f"    Preemptions: Gaussian={avg_pre_g:.1f} vs ParetoLN={avg_pre_p:.1f}")
        if avg_ttft < 1.05 and abs(avg_pre_p - avg_pre_g) < max(avg_pre_g * 0.1, 5):
            print("    => Effect vanishes at sub-saturation as predicted")
        else:
            print("    => Effect persists at sub-saturation (may indicate intrinsic prefill cost)")

    # Exp 3: KV-abundant control
    if exp3_stats:
        n = exp3_stats["n"]
        avg_ttft = exp3_stats["avg_ttft_ratio"]
        avg_pre_g = exp3_stats["avg_preempt_g"]
        avg_pre_p = exp3_stats["avg_preempt_p"]
        print(f"  Exp 3 (rate=1000, KV=100000, 500 req): KV-abundant control")
        print(f"    TTFT p99 ratio: {avg_ttft:.3f}x")
        print(f"    Preemptions: Gaussian={avg_pre_g:.1f} vs ParetoLN={avg_pre_p:.1f}")
        if avg_pre_g == 0 and avg_pre_p == 0:
            print("    => Preemptions vanish with abundant KV (confirms KV pressure is mechanism)")
        elif avg_pre_p <= avg_pre_g:
            print("    => ParetoLN preemptions <= Gaussian (KV pressure mechanism eliminated)")
        else:
            print("    => UNEXPECTED: preemptions persist with abundant KV")

    print()

    # Overall assessment
    # Primary refutation signal: preemption count (hypothesis predicts ParetoLN MORE,
    # data shows ParetoLN FEWER). TTFT p99 is supporting evidence but mixed (2/3 seeds).
    if exp1_stats:
        n = exp1_stats["n"]
        ttft_confirmed = (exp1_stats["pareto_worse_ttft"] == n
                          and exp1_stats["min_ttft_ratio"] >= 1.20)
        preempt_confirmed = exp1_stats["pareto_more_preemptions"] == n
        ttft_directional = exp1_stats["pareto_worse_ttft"] == n
        preempt_directional = exp1_stats["pareto_more_preemptions"] == n

        # Check if OPPOSITE is true for preemptions (Gaussian has MORE preemptions)
        gaussian_more_preemptions = sum(
            1 for pg, pp in zip(exp1_stats["preempt_g_list"], exp1_stats["preempt_p_list"])
            if pg > pp
        )
        gaussian_worse_ttft = sum(1 for r in exp1_stats["per_seed_ttft_ratios"] if r < 1.0)

        kv_control_validates = (exp3_stats and
                                exp3_stats["avg_preempt_g"] == 0 and
                                exp3_stats["avg_preempt_p"] == 0)

        if ttft_confirmed and preempt_confirmed:
            print("  VERDICT: CONFIRMED -- ParetoLogNormal produces more preemptions and")
            print("    higher TTFT p99 than Gaussian at constrained KV blocks.")
        elif ttft_directional and preempt_directional:
            print("  VERDICT: CONFIRMED WITH NUANCE -- directionally consistent but")
            print("    some per-seed effect sizes below 20% threshold.")
        elif gaussian_more_preemptions == n:
            # Preemption prediction is REVERSED in all seeds — primary refutation signal
            if gaussian_worse_ttft == n:
                print("  VERDICT: REFUTED -- Gaussian produces MORE preemptions AND worse")
                print("    TTFT p99 than ParetoLogNormal in all seeds. See Root Cause Analysis.")
            else:
                print(f"  VERDICT: REFUTED -- Gaussian produces MORE preemptions in all {n}")
                print(f"    seeds (primary metric). TTFT p99 favors ParetoLN in {gaussian_worse_ttft}/{n}")
                print(f"    seeds (mixed, but preemption reversal is the primary refutation signal).")
        elif ttft_directional or preempt_directional:
            print("  VERDICT: PARTIALLY CONFIRMED -- one metric directionally consistent,")
            print("    the other is mixed.")
        else:
            print("  VERDICT: INCONCLUSIVE -- mixed directionality across seeds.")

        if kv_control_validates:
            print("  KV-abundant control: VALIDATES that preemption differences require")
            print("    constrained KV (mechanism confirmation).")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # Load all experiments
    exp1_g, exp1_p = load_experiment(results_dir, "exp1")
    exp2_g, exp2_p = load_experiment(results_dir, "exp2")
    exp3_g, exp3_p = load_experiment(results_dir, "exp3")

    # Per-experiment comparison tables
    print_comparison_table(exp1_g, exp1_p,
                           "Exp 1: CORE (rate=1000, KV=2000, 500 req)")
    print_comparison_table(exp2_g, exp2_p,
                           "Exp 2: SUB-SATURATION CONTROL (rate=200, KV=2000, 500 req)")
    print_comparison_table(exp3_g, exp3_p,
                           "Exp 3: KV-ABUNDANT CONTROL (rate=1000, KV=100000, 500 req)")

    # Conservation check across all experiments
    datasets = [
        ("Exp1", exp1_g, exp1_p),
        ("Exp2", exp2_g, exp2_p),
        ("Exp3", exp3_g, exp3_p),
    ]
    print_conservation_check(datasets)

    # Preemption and cache detail
    print_preemption_detail(datasets)

    # Cross-seed summaries
    print("=" * 90)
    print("  CROSS-SEED SUMMARIES")
    print("=" * 90)
    print()

    exp1_stats = compute_summary_stats(exp1_g, exp1_p)
    exp2_stats = compute_summary_stats(exp2_g, exp2_p)
    exp3_stats = compute_summary_stats(exp3_g, exp3_p)

    print_summary("Exp 1: Core (rate=1000, KV=2000, 500 req)", exp1_stats)
    print_summary("Exp 2: Sub-saturation (rate=200, KV=2000, 500 req)", exp2_stats)
    print_summary("Exp 3: KV-abundant (rate=1000, KV=100000, 500 req)", exp3_stats)

    # Overall verdict
    print_verdict(exp1_stats, exp2_stats, exp3_stats)


if __name__ == "__main__":
    main()
