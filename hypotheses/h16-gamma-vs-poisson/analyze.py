#!/usr/bin/env python3
"""Analysis script for H16: Gamma vs Poisson Tail Latency.

Parses BLIS output for Poisson and Gamma arrival configurations across
multiple seeds and produces comparison tables.

Hypothesis: Gamma (CV=3.5) arrivals produce worse tail latency than
Poisson at the same average rate.

Experiments:
  Exp 1:  Core comparison — rate=1000, 500 requests (Round 1)
  Exp B2: Sub-saturation control — rate=200, 500 requests (Round 2)
  Exp C1: Larger sample — rate=1000, 2000 requests (Round 2)

Statistical note: scipy is not available in this environment. Significance
thresholds are assessed by consistent directionality across seeds and
effect size magnitude, not formal hypothesis tests (legacy threshold
exemption per MEMORY.md).

Usage: python3 analyze.py <results_dir>
"""
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout

SEEDS = [42, 123, 456]


def load_experiment(results_dir, prefix):
    """Load Poisson and Gamma results for all seeds with given prefix."""
    poisson = {}
    gamma = {}
    for seed in SEEDS:
        p_file = Path(results_dir) / f"{prefix}_poisson_{seed}.txt"
        g_file = Path(results_dir) / f"{prefix}_gamma_{seed}.txt"
        poisson[seed] = parse_blis_output(str(p_file))
        gamma[seed] = parse_blis_output(str(g_file))
    return poisson, gamma


def print_comparison_table(poisson, gamma, title):
    """Print per-seed comparison of key metrics."""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()

    fmt = "{:<8} {:<12} {:>14} {:>14} {:>14} {:>14}"
    print(fmt.format("Seed", "Arrival", "TTFT mean(ms)", "TTFT p99(ms)",
                      "E2E mean(ms)", "E2E p99(ms)"))
    print("-" * 80)

    for seed in SEEDS:
        p = poisson[seed]
        g = gamma[seed]
        if p["timed_out"]:
            print(f"  Seed {seed}: Poisson TIMED OUT -- skipping")
            continue
        if g["timed_out"]:
            print(f"  Seed {seed}: Gamma TIMED OUT -- skipping")
            continue

        print(fmt.format(seed, "Poisson",
                          f"{p['ttft_mean']:.2f}",
                          f"{p['ttft_p99']:.2f}",
                          f"{p['e2e_mean']:.2f}",
                          f"{p['e2e_p99']:.2f}"))
        print(fmt.format("", "Gamma",
                          f"{g['ttft_mean']:.2f}",
                          f"{g['ttft_p99']:.2f}",
                          f"{g['e2e_mean']:.2f}",
                          f"{g['e2e_p99']:.2f}"))

        if p["ttft_p99"] > 0:
            ttft_ratio = g["ttft_p99"] / p["ttft_p99"]
        else:
            ttft_ratio = float("inf")
        if p["e2e_p99"] > 0:
            e2e_ratio = g["e2e_p99"] / p["e2e_p99"]
        else:
            e2e_ratio = float("inf")

        print(fmt.format("", "Ratio",
                          f"{g['ttft_mean']/max(p['ttft_mean'],0.001):.2f}x",
                          f"{ttft_ratio:.2f}x",
                          f"{g['e2e_mean']/max(p['e2e_mean'],0.001):.2f}x",
                          f"{e2e_ratio:.2f}x"))
        print()


def print_conservation_check(datasets):
    """Verify INV-1 for all runs across all experiments."""
    print("=" * 80)
    print("  CONSERVATION CHECK (INV-1)")
    print("=" * 80)
    print()

    all_pass = True
    for exp_label, poisson, gamma in datasets:
        for arrival_label, data in [("Poisson", poisson), ("Gamma", gamma)]:
            for seed in SEEDS:
                m = data[seed]
                if m["timed_out"]:
                    print(f"  [{exp_label}] {arrival_label} seed={seed}: SKIPPED (timeout)")
                    continue
                injected = m["injected"]
                completed = m["completed"]
                queued = m["still_queued"]
                running = m["still_running"]
                conserved = (completed + queued + running) == injected
                status = "PASS" if conserved else "FAIL"
                if not conserved:
                    all_pass = False
                print(f"  [{exp_label}] {arrival_label} seed={seed}: "
                      f"injected={injected}, completed={completed}, "
                      f"queued={queued}, running={running} -> {status}")

    print()
    if all_pass:
        print("  OVERALL: ALL CONSERVATION CHECKS PASS")
    else:
        print("  OVERALL: SOME CONSERVATION CHECKS FAILED")
    print()


def compute_summary_stats(poisson, gamma):
    """Compute cross-seed summary statistics. Returns dict or None if all timed out."""
    valid_seeds = []
    ttft_p99_p = []
    ttft_p99_g = []
    e2e_p99_p = []
    e2e_p99_g = []
    ttft_mean_p = []
    ttft_mean_g = []

    for seed in SEEDS:
        p = poisson[seed]
        g = gamma[seed]
        if p["timed_out"] or g["timed_out"]:
            continue
        valid_seeds.append(seed)
        ttft_p99_p.append(p["ttft_p99"])
        ttft_p99_g.append(g["ttft_p99"])
        e2e_p99_p.append(p["e2e_p99"])
        e2e_p99_g.append(g["e2e_p99"])
        ttft_mean_p.append(p["ttft_mean"])
        ttft_mean_g.append(g["ttft_mean"])

    if not valid_seeds:
        return None

    n = len(valid_seeds)
    gamma_wins_ttft = sum(1 for i in range(n) if ttft_p99_g[i] > ttft_p99_p[i])
    gamma_wins_e2e = sum(1 for i in range(n) if e2e_p99_g[i] > e2e_p99_p[i])
    per_seed_ratios = [g / max(p, 0.001) for g, p in zip(ttft_p99_g, ttft_p99_p)]
    avg_ttft_ratio = sum(per_seed_ratios) / n
    avg_e2e_ratio = sum(g / max(p, 0.001) for g, p in zip(e2e_p99_g, e2e_p99_p)) / n
    min_ratio = min(per_seed_ratios)
    max_ratio = max(per_seed_ratios)

    return {
        "n": n, "total": len(SEEDS),
        "gamma_wins_ttft": gamma_wins_ttft,
        "gamma_wins_e2e": gamma_wins_e2e,
        "avg_ttft_ratio": avg_ttft_ratio,
        "avg_e2e_ratio": avg_e2e_ratio,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "per_seed_ratios": per_seed_ratios,
        "ttft_mean_p": sum(ttft_mean_p) / n,
        "ttft_mean_g": sum(ttft_mean_g) / n,
        "ttft_p99_p": sum(ttft_p99_p) / n,
        "ttft_p99_g": sum(ttft_p99_g) / n,
        "e2e_p99_p": sum(e2e_p99_p) / n,
        "e2e_p99_g": sum(e2e_p99_g) / n,
    }


def print_summary(label, stats):
    """Print cross-seed summary and hypothesis verdict for one experiment."""
    print(f"  --- {label} ---")
    if stats is None:
        print("  No valid results (all timed out).")
        print()
        return

    n = stats["n"]
    print(f"  Valid seeds: {n}/{stats['total']}")
    print(f"  Seeds where Gamma TTFT p99 > Poisson: {stats['gamma_wins_ttft']}/{n}")
    print(f"  Seeds where Gamma E2E p99 > Poisson:  {stats['gamma_wins_e2e']}/{n}")
    print(f"  TTFT p99 ratio (Gamma/Poisson): avg={stats['avg_ttft_ratio']:.2f}x "
          f"range=[{stats['min_ratio']:.2f}x, {stats['max_ratio']:.2f}x]")
    print(f"  E2E p99 ratio (Gamma/Poisson):  avg={stats['avg_e2e_ratio']:.2f}x")
    print()
    print(f"  Cross-seed averages:")
    print(f"    Poisson TTFT mean: {stats['ttft_mean_p']:.2f} ms")
    print(f"    Gamma   TTFT mean: {stats['ttft_mean_g']:.2f} ms")
    print(f"    Poisson TTFT p99:  {stats['ttft_p99_p']:.2f} ms")
    print(f"    Gamma   TTFT p99:  {stats['ttft_p99_g']:.2f} ms")
    print(f"    Poisson E2E p99:   {stats['e2e_p99_p']:.2f} ms")
    print(f"    Gamma   E2E p99:   {stats['e2e_p99_g']:.2f} ms")
    print()


def print_verdict(exp1_stats, b2_stats, c1_stats):
    """Print overall hypothesis verdict incorporating all experiments."""
    print("=" * 80)
    print("  OVERALL VERDICT")
    print("=" * 80)
    print()

    # Exp 1 verdict
    if exp1_stats:
        n = exp1_stats["n"]
        wins = exp1_stats["gamma_wins_ttft"]
        ratio = exp1_stats["avg_ttft_ratio"]
        mn = exp1_stats["min_ratio"]
        mx = exp1_stats["max_ratio"]
        print(f"  Exp 1 (rate=1000, 500 req): {wins}/{n} seeds Gamma worse, "
              f"avg ratio={ratio:.2f}x, range=[{mn:.2f}x, {mx:.2f}x]")
        if mn < 1.10:
            print(f"    NOTE: minimum per-seed ratio {mn:.2f}x is below 10% threshold")

    # B2 verdict
    if b2_stats:
        n = b2_stats["n"]
        wins = b2_stats["gamma_wins_ttft"]
        ratio = b2_stats["avg_ttft_ratio"]
        mn = b2_stats["min_ratio"]
        mx = b2_stats["max_ratio"]
        print(f"  Exp B2 (rate=200, 500 req): {wins}/{n} seeds Gamma worse, "
              f"avg ratio={ratio:.2f}x, range=[{mn:.2f}x, {mx:.2f}x]")
        if ratio < 1.05:
            print("    => Sub-saturation control: effect vanishes as predicted")
        else:
            print("    => UNEXPECTED: effect persists at sub-saturation")

    # C1 verdict
    if c1_stats:
        n = c1_stats["n"]
        wins = c1_stats["gamma_wins_ttft"]
        ratio = c1_stats["avg_ttft_ratio"]
        mn = c1_stats["min_ratio"]
        mx = c1_stats["max_ratio"]
        print(f"  Exp C1 (rate=1000, 2000 req): {wins}/{n} seeds Gamma worse, "
              f"avg ratio={ratio:.2f}x, range=[{mn:.2f}x, {mx:.2f}x]")
        if mn >= 1.10:
            print("    => Larger sample eliminates anomalous sub-10% seeds")
        else:
            print(f"    NOTE: minimum per-seed ratio {mn:.2f}x still below 10%")

    print()

    # Overall
    all_directional = True
    if exp1_stats and exp1_stats["gamma_wins_ttft"] < exp1_stats["n"]:
        all_directional = False
    if c1_stats and c1_stats["gamma_wins_ttft"] < c1_stats["n"]:
        all_directional = False

    sub_sat_vanishes = b2_stats and b2_stats["avg_ttft_ratio"] < 1.05

    has_sub_threshold = (exp1_stats and exp1_stats["min_ratio"] < 1.10)

    if all_directional and sub_sat_vanishes and not has_sub_threshold:
        print("  VERDICT: CONFIRMED -- Gamma produces worse TTFT tail latency")
    elif all_directional and sub_sat_vanishes:
        print("  VERDICT: CONFIRMED WITH NUANCE -- Gamma is directionally worse across")
        print("    all seeds, sub-saturation control validates mechanism, but per-seed")
        print("    effect size varies (some seeds below 10% threshold)")
    elif all_directional:
        print("  VERDICT: PARTIALLY CONFIRMED -- directionally consistent but")
        print("    sub-saturation control did not behave as expected")
    else:
        print("  VERDICT: INCONCLUSIVE -- mixed directionality across seeds")
    print()


def print_preemption_info(datasets):
    """Print preemption and throughput info across all experiments."""
    print("=" * 80)
    print("  ADDITIONAL METRICS (confound check)")
    print("=" * 80)
    print()

    fmt = "{:<6} {:<8} {:<12} {:>12} {:>10} {:>12} {:>10}"
    print(fmt.format("Exp", "Seed", "Arrival", "Throughput", "Completed",
                      "Preemptions", "Cache Hit"))
    print("-" * 80)

    for exp_label, poisson, gamma in datasets:
        for seed in SEEDS:
            p = poisson[seed]
            g = gamma[seed]
            if p["timed_out"] or g["timed_out"]:
                continue

            print(fmt.format(exp_label, seed, "Poisson",
                              f"{p['throughput']:.2f}",
                              str(p['completed']),
                              str(p['preemption_count']),
                              f"{p['cache_hit_rate']:.4f}"))
            print(fmt.format("", "", "Gamma",
                              f"{g['throughput']:.2f}",
                              str(g['completed']),
                              str(g['preemption_count']),
                              f"{g['cache_hit_rate']:.4f}"))
        print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # Load all experiments
    exp1_p, exp1_g = load_experiment(results_dir, "exp1")
    b2_p, b2_g = load_experiment(results_dir, "b2")
    c1_p, c1_g = load_experiment(results_dir, "c1")

    # Per-experiment comparison tables
    print_comparison_table(exp1_p, exp1_g,
                           "Exp 1: CORE (rate=1000, 500 req, 3x overload)")
    print_comparison_table(b2_p, b2_g,
                           "Exp B2: SUB-SATURATION CONTROL (rate=200, 500 req, 0.59x util)")
    print_comparison_table(c1_p, c1_g,
                           "Exp C1: LARGER SAMPLE (rate=1000, 2000 req, 3x overload)")

    # Conservation check across all experiments
    datasets = [
        ("Exp1", exp1_p, exp1_g),
        ("B2", b2_p, b2_g),
        ("C1", c1_p, c1_g),
    ]
    print_conservation_check(datasets)

    # Additional metrics
    print_preemption_info(datasets)

    # Summaries
    print("=" * 80)
    print("  CROSS-SEED SUMMARIES")
    print("=" * 80)
    print()

    exp1_stats = compute_summary_stats(exp1_p, exp1_g)
    b2_stats = compute_summary_stats(b2_p, b2_g)
    c1_stats = compute_summary_stats(c1_p, c1_g)

    print_summary("Exp 1: Core (rate=1000, 500 req)", exp1_stats)
    print_summary("Exp B2: Sub-saturation (rate=200, 500 req)", b2_stats)
    print_summary("Exp C1: Larger sample (rate=1000, 2000 req)", c1_stats)

    # Overall verdict
    print_verdict(exp1_stats, b2_stats, c1_stats)


if __name__ == "__main__":
    main()
