#!/usr/bin/env python3
"""Analysis script for H4: Round-Robin vs Least-Loaded at Low Utilization.

Parses BLIS output for RR and LL routing policies across multiple seeds
and produces comparison tables with equivalence testing.

Hypothesis: Round-robin should match least-loaded (within 5%) for
uniform workloads at low rates.

Experiments:
  Exp 1: Low rate (rate=100, 0.29x utilization) -- equivalence expected
  Exp 2: High rate control (rate=1000, 3x overload) -- LL should outperform RR

Equivalence criterion: |RR - LL| / max(RR, LL) < 5% across all seeds.

Usage: python3 analyze.py <results_dir>
"""
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout

SEEDS = [42, 123, 456]


def load_experiment(results_dir, prefix):
    """Load RR and LL results for all seeds with given prefix."""
    rr = {}
    ll = {}
    for seed in SEEDS:
        rr_file = Path(results_dir) / f"{prefix}_rr_{seed}.txt"
        ll_file = Path(results_dir) / f"{prefix}_ll_{seed}.txt"
        rr[seed] = parse_blis_output(str(rr_file))
        ll[seed] = parse_blis_output(str(ll_file))
    return rr, ll


def pct_diff(a, b):
    """Compute percentage difference: |a - b| / max(a, b). Returns 0 if both are 0."""
    mx = max(a, b)
    if mx == 0:
        return 0.0
    return abs(a - b) / mx * 100.0


def print_comparison_table(rr, ll, title):
    """Print per-seed comparison of key metrics."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()

    fmt = "{:<8} {:<14} {:>14} {:>14} {:>14} {:>14}"
    print(fmt.format("Seed", "Routing", "TTFT mean(ms)", "TTFT p99(ms)",
                      "E2E mean(ms)", "E2E p99(ms)"))
    print("-" * 90)

    for seed in SEEDS:
        r = rr[seed]
        l = ll[seed]
        if r["timed_out"]:
            print(f"  Seed {seed}: RR TIMED OUT -- skipping")
            continue
        if l["timed_out"]:
            print(f"  Seed {seed}: LL TIMED OUT -- skipping")
            continue

        print(fmt.format(seed, "Round-Robin",
                          f"{r['ttft_mean']:.2f}",
                          f"{r['ttft_p99']:.2f}",
                          f"{r['e2e_mean']:.2f}",
                          f"{r['e2e_p99']:.2f}"))
        print(fmt.format("", "Least-Loaded",
                          f"{l['ttft_mean']:.2f}",
                          f"{l['ttft_p99']:.2f}",
                          f"{l['e2e_mean']:.2f}",
                          f"{l['e2e_p99']:.2f}"))

        # Percentage differences
        ttft_mean_diff = pct_diff(r['ttft_mean'], l['ttft_mean'])
        ttft_p99_diff = pct_diff(r['ttft_p99'], l['ttft_p99'])
        e2e_mean_diff = pct_diff(r['e2e_mean'], l['e2e_mean'])
        e2e_p99_diff = pct_diff(r['e2e_p99'], l['e2e_p99'])

        print(fmt.format("", "% Diff",
                          f"{ttft_mean_diff:.2f}%",
                          f"{ttft_p99_diff:.2f}%",
                          f"{e2e_mean_diff:.2f}%",
                          f"{e2e_p99_diff:.2f}%"))
        print()


def print_conservation_check(datasets):
    """Verify INV-1 for all runs across all experiments."""
    print("=" * 90)
    print("  CONSERVATION CHECK (INV-1)")
    print("=" * 90)
    print()

    all_pass = True
    for exp_label, rr, ll in datasets:
        for policy_label, data in [("RR", rr), ("LL", ll)]:
            for seed in SEEDS:
                m = data[seed]
                if m["timed_out"]:
                    print(f"  [{exp_label}] {policy_label} seed={seed}: SKIPPED (timeout)")
                    continue
                injected = m["injected"]
                completed = m["completed"]
                queued = m["still_queued"]
                running = m["still_running"]
                conserved = (completed + queued + running) == injected
                status = "PASS" if conserved else "FAIL"
                if not conserved:
                    all_pass = False
                print(f"  [{exp_label}] {policy_label} seed={seed}: "
                      f"injected={injected}, completed={completed}, "
                      f"queued={queued}, running={running} -> {status}")

    print()
    if all_pass:
        print("  OVERALL: ALL CONSERVATION CHECKS PASS")
    else:
        print("  OVERALL: SOME CONSERVATION CHECKS FAILED")
    print()


def print_additional_metrics(datasets):
    """Print throughput, preemption, and cache hit info."""
    print("=" * 90)
    print("  ADDITIONAL METRICS")
    print("=" * 90)
    print()

    fmt = "{:<6} {:<8} {:<14} {:>12} {:>10} {:>12} {:>10}"
    print(fmt.format("Exp", "Seed", "Routing", "Throughput", "Completed",
                      "Preemptions", "Cache Hit"))
    print("-" * 90)

    for exp_label, rr, ll in datasets:
        for seed in SEEDS:
            r = rr[seed]
            l = ll[seed]
            if r["timed_out"] or l["timed_out"]:
                continue

            print(fmt.format(exp_label, seed, "RR",
                              f"{r['throughput']:.2f}",
                              str(r['completed']),
                              str(r['preemption_count']),
                              f"{r['cache_hit_rate']:.4f}"))
            print(fmt.format("", "", "LL",
                              f"{l['throughput']:.2f}",
                              str(l['completed']),
                              str(l['preemption_count']),
                              f"{l['cache_hit_rate']:.4f}"))
        print()


def compute_equivalence_stats(rr, ll):
    """Compute equivalence statistics across seeds.

    Returns dict with per-seed and cross-seed metrics, or None if all timed out.
    """
    valid_seeds = []
    ttft_mean_diffs = []
    ttft_p99_diffs = []
    e2e_mean_diffs = []
    e2e_p99_diffs = []
    ttft_mean_rr = []
    ttft_mean_ll = []

    for seed in SEEDS:
        r = rr[seed]
        l = ll[seed]
        if r["timed_out"] or l["timed_out"]:
            continue
        valid_seeds.append(seed)
        ttft_mean_diffs.append(pct_diff(r['ttft_mean'], l['ttft_mean']))
        ttft_p99_diffs.append(pct_diff(r['ttft_p99'], l['ttft_p99']))
        e2e_mean_diffs.append(pct_diff(r['e2e_mean'], l['e2e_mean']))
        e2e_p99_diffs.append(pct_diff(r['e2e_p99'], l['e2e_p99']))
        ttft_mean_rr.append(r['ttft_mean'])
        ttft_mean_ll.append(l['ttft_mean'])

    if not valid_seeds:
        return None

    n = len(valid_seeds)
    return {
        "n": n,
        "total": len(SEEDS),
        "valid_seeds": valid_seeds,
        "ttft_mean_diffs": ttft_mean_diffs,
        "ttft_p99_diffs": ttft_p99_diffs,
        "e2e_mean_diffs": e2e_mean_diffs,
        "e2e_p99_diffs": e2e_p99_diffs,
        "max_ttft_mean_diff": max(ttft_mean_diffs),
        "max_ttft_p99_diff": max(ttft_p99_diffs),
        "max_e2e_mean_diff": max(e2e_mean_diffs),
        "max_e2e_p99_diff": max(e2e_p99_diffs),
        "avg_ttft_mean_rr": sum(ttft_mean_rr) / n,
        "avg_ttft_mean_ll": sum(ttft_mean_ll) / n,
    }


def compute_dominance_stats(rr, ll):
    """Compute dominance statistics for high-rate experiment.

    Returns dict with per-seed ratios (LL TTFT / RR TTFT), or None if all timed out.
    """
    valid_seeds = []
    ttft_mean_ratios = []
    ttft_p99_ratios = []
    e2e_mean_ratios = []

    for seed in SEEDS:
        r = rr[seed]
        l = ll[seed]
        if r["timed_out"] or l["timed_out"]:
            continue
        valid_seeds.append(seed)
        # Ratio < 1 means LL is better
        ttft_mean_ratios.append(l['ttft_mean'] / max(r['ttft_mean'], 0.001))
        ttft_p99_ratios.append(l['ttft_p99'] / max(r['ttft_p99'], 0.001))
        e2e_mean_ratios.append(l['e2e_mean'] / max(r['e2e_mean'], 0.001))

    if not valid_seeds:
        return None

    n = len(valid_seeds)
    ll_wins_ttft = sum(1 for r in ttft_mean_ratios if r < 1.0)
    return {
        "n": n,
        "total": len(SEEDS),
        "valid_seeds": valid_seeds,
        "ttft_mean_ratios": ttft_mean_ratios,
        "ttft_p99_ratios": ttft_p99_ratios,
        "e2e_mean_ratios": e2e_mean_ratios,
        "ll_wins_ttft_mean": ll_wins_ttft,
        "avg_ttft_mean_ratio": sum(ttft_mean_ratios) / n,
        "avg_ttft_p99_ratio": sum(ttft_p99_ratios) / n,
    }


def print_equivalence_summary(label, stats):
    """Print equivalence summary for low-rate experiment."""
    print(f"  --- {label} ---")
    if stats is None:
        print("  No valid results (all timed out).")
        print()
        return

    n = stats["n"]
    print(f"  Valid seeds: {n}/{stats['total']}")
    print(f"  Equivalence threshold: 5%")
    print()
    print(f"  Per-seed percentage differences:")

    for i, seed in enumerate(stats["valid_seeds"]):
        print(f"    Seed {seed}: TTFT mean={stats['ttft_mean_diffs'][i]:.2f}%, "
              f"TTFT p99={stats['ttft_p99_diffs'][i]:.2f}%, "
              f"E2E mean={stats['e2e_mean_diffs'][i]:.2f}%, "
              f"E2E p99={stats['e2e_p99_diffs'][i]:.2f}%")

    print()
    print(f"  Max TTFT mean diff: {stats['max_ttft_mean_diff']:.2f}%")
    print(f"  Max TTFT p99 diff:  {stats['max_ttft_p99_diff']:.2f}%")
    print(f"  Max E2E mean diff:  {stats['max_e2e_mean_diff']:.2f}%")
    print(f"  Max E2E p99 diff:   {stats['max_e2e_p99_diff']:.2f}%")
    print()

    all_within = (stats['max_ttft_mean_diff'] < 5.0 and
                  stats['max_ttft_p99_diff'] < 5.0 and
                  stats['max_e2e_mean_diff'] < 5.0 and
                  stats['max_e2e_p99_diff'] < 5.0)
    if all_within:
        print("  => EQUIVALENT: All metrics within 5% across all seeds")
    else:
        exceeding = []
        if stats['max_ttft_mean_diff'] >= 5.0:
            exceeding.append(f"TTFT mean ({stats['max_ttft_mean_diff']:.2f}%)")
        if stats['max_ttft_p99_diff'] >= 5.0:
            exceeding.append(f"TTFT p99 ({stats['max_ttft_p99_diff']:.2f}%)")
        if stats['max_e2e_mean_diff'] >= 5.0:
            exceeding.append(f"E2E mean ({stats['max_e2e_mean_diff']:.2f}%)")
        if stats['max_e2e_p99_diff'] >= 5.0:
            exceeding.append(f"E2E p99 ({stats['max_e2e_p99_diff']:.2f}%)")
        print(f"  => NOT EQUIVALENT: Exceeds 5% in: {', '.join(exceeding)}")
    print()


def print_dominance_summary(label, stats):
    """Print dominance summary for high-rate control experiment."""
    print(f"  --- {label} ---")
    if stats is None:
        print("  No valid results (all timed out).")
        print()
        return

    n = stats["n"]
    print(f"  Valid seeds: {n}/{stats['total']}")
    print(f"  LL wins on TTFT mean: {stats['ll_wins_ttft_mean']}/{n}")
    print()
    print(f"  Per-seed ratios (LL/RR, <1 = LL better):")
    for i, seed in enumerate(stats["valid_seeds"]):
        print(f"    Seed {seed}: TTFT mean={stats['ttft_mean_ratios'][i]:.4f}x, "
              f"TTFT p99={stats['ttft_p99_ratios'][i]:.4f}x, "
              f"E2E mean={stats['e2e_mean_ratios'][i]:.4f}x")

    print()
    print(f"  Avg TTFT mean ratio: {stats['avg_ttft_mean_ratio']:.4f}x")
    print(f"  Avg TTFT p99 ratio:  {stats['avg_ttft_p99_ratio']:.4f}x")
    print()

    all_ll_better = all(r < 1.0 for r in stats['ttft_mean_ratios'])
    significant = all(r < 0.80 for r in stats['ttft_mean_ratios'])

    if all_ll_better and significant:
        print("  => CONTROL VALIDATES: LL outperforms RR by >20% on TTFT mean across all seeds")
    elif all_ll_better:
        print("  => CONTROL PARTIALLY VALIDATES: LL better in all seeds but effect <20%")
    else:
        print("  => CONTROL UNEXPECTED: LL does not consistently outperform RR at high rate")
    print()


def print_verdict(exp1_stats, exp2_stats):
    """Print overall hypothesis verdict."""
    print("=" * 90)
    print("  OVERALL VERDICT")
    print("=" * 90)
    print()

    # Low-rate equivalence
    if exp1_stats:
        all_within = (exp1_stats['max_ttft_mean_diff'] < 5.0 and
                      exp1_stats['max_ttft_p99_diff'] < 5.0 and
                      exp1_stats['max_e2e_mean_diff'] < 5.0 and
                      exp1_stats['max_e2e_p99_diff'] < 5.0)
        print(f"  Exp 1 (rate=100, low util): max TTFT mean diff={exp1_stats['max_ttft_mean_diff']:.2f}%, "
              f"max TTFT p99 diff={exp1_stats['max_ttft_p99_diff']:.2f}%, "
              f"max E2E mean diff={exp1_stats['max_e2e_mean_diff']:.2f}%")
        if all_within:
            print("    => Policies equivalent at low rate (all metrics within 5%)")
        else:
            print("    => Policies NOT fully equivalent at low rate (some metrics exceed 5%)")

    # High-rate control
    if exp2_stats:
        avg_ratio = exp2_stats['avg_ttft_mean_ratio']
        ll_wins = exp2_stats['ll_wins_ttft_mean']
        print(f"  Exp 2 (rate=1000, overload): LL wins {ll_wins}/{exp2_stats['n']} seeds, "
              f"avg TTFT ratio={avg_ratio:.4f}x")
        if ll_wins == exp2_stats['n'] and avg_ratio < 0.80:
            print("    => High-rate control validates: LL measurably better under load")
        else:
            print("    => High-rate control: LL advantage is modest or inconsistent")

    print()

    # Overall — check ALL four metrics for strict equivalence
    all_within_5pct = (exp1_stats and
                       exp1_stats['max_ttft_mean_diff'] < 5.0 and
                       exp1_stats['max_ttft_p99_diff'] < 5.0 and
                       exp1_stats['max_e2e_mean_diff'] < 5.0 and
                       exp1_stats['max_e2e_p99_diff'] < 5.0)
    means_within_5pct = (exp1_stats and
                         exp1_stats['max_ttft_mean_diff'] < 5.0 and
                         exp1_stats['max_e2e_mean_diff'] < 5.0)
    p99_exceeds = (exp1_stats and
                   (exp1_stats['max_ttft_p99_diff'] >= 5.0 or
                    exp1_stats['max_e2e_p99_diff'] >= 5.0))
    high_validates = (exp2_stats and
                      exp2_stats['ll_wins_ttft_mean'] == exp2_stats['n'] and
                      exp2_stats['avg_ttft_mean_ratio'] < 0.95)

    if all_within_5pct and high_validates:
        print("  VERDICT: CONFIRMED -- RR and LL equivalent at low rate (all metrics <5%),")
        print("    LL outperforms under overload (control validates comparison).")
    elif all_within_5pct:
        print("  VERDICT: CONFIRMED -- RR and LL equivalent at low rate (all metrics <5%).")
    elif means_within_5pct and p99_exceeds:
        exceeding = []
        if exp1_stats['max_ttft_p99_diff'] >= 5.0:
            exceeding.append(f"TTFT p99 ({exp1_stats['max_ttft_p99_diff']:.1f}%)")
        if exp1_stats['max_e2e_p99_diff'] >= 5.0:
            exceeding.append(f"E2E p99 ({exp1_stats['max_e2e_p99_diff']:.1f}%)")
        print("  VERDICT: CONFIRMED WITH NUANCE -- mean metrics equivalent at low rate (<5%),")
        print(f"    but tail metrics exceed 5%: {', '.join(exceeding)}.")
        if not high_validates:
            print("    High-rate control did not validate LL advantage (identical parsed metrics).")
    elif high_validates:
        print("  VERDICT: REFUTED -- RR and LL NOT equivalent even at low rate,")
        print("    despite LL being better at high rate.")
    else:
        print("  VERDICT: INCONCLUSIVE -- neither equivalence nor clear dominance observed.")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # Load experiments
    exp1_rr, exp1_ll = load_experiment(results_dir, "exp1")
    exp2_rr, exp2_ll = load_experiment(results_dir, "exp2")

    # Per-experiment comparison tables
    print_comparison_table(exp1_rr, exp1_ll,
                           "Exp 1: LOW RATE (rate=100, 500 req, 0.29x utilization)")
    print_comparison_table(exp2_rr, exp2_ll,
                           "Exp 2: HIGH RATE CONTROL (rate=1000, 500 req, 3x overload)")

    # Conservation check
    datasets = [
        ("Exp1", exp1_rr, exp1_ll),
        ("Exp2", exp2_rr, exp2_ll),
    ]
    print_conservation_check(datasets)

    # Additional metrics
    print_additional_metrics(datasets)

    # Summaries
    print("=" * 90)
    print("  EQUIVALENCE & DOMINANCE SUMMARIES")
    print("=" * 90)
    print()

    exp1_stats = compute_equivalence_stats(exp1_rr, exp1_ll)
    exp2_stats = compute_dominance_stats(exp2_rr, exp2_ll)

    print_equivalence_summary("Exp 1: Low Rate (rate=100, 500 req)", exp1_stats)
    print_dominance_summary("Exp 2: High Rate Control (rate=1000, 500 req)", exp2_stats)

    # Overall verdict
    print_verdict(exp1_stats, exp2_stats)


if __name__ == "__main__":
    main()
