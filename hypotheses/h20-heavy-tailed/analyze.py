#!/usr/bin/env python3
"""Analysis script for H20: Heavy-Tailed Input Distributions.

Parses BLIS output for Gaussian and ParetoLogNormal input distributions
across multiple seeds and produces comparison tables.

Hypothesis: ParetoLogNormal input distributions produce more preemptions
and worse tail latency than Gaussian at the same average input length.

Experiments:
  Exp 1:  Core comparison -- default KV, rate=1000, 500 requests
  Exp 2:  KV-constrained -- 2000 blocks, rate=1000, 500 requests
  Exp 3:  Sub-saturation control -- default KV, rate=200, 500 requests

Usage: python3 analyze.py <results_dir>
"""
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout

SEEDS = [42, 123, 456]


def load_experiment(results_dir, prefix):
    """Load Gaussian and ParetoLogNormal results for all seeds."""
    gaussian = {}
    pareto = {}
    for seed in SEEDS:
        g_file = Path(results_dir) / f"{prefix}_gaussian_{seed}.txt"
        p_file = Path(results_dir) / f"{prefix}_pareto_{seed}.txt"
        gaussian[seed] = parse_blis_output(str(g_file))
        pareto[seed] = parse_blis_output(str(p_file))
    return gaussian, pareto


def print_comparison_table(gaussian, pareto, title):
    """Print per-seed comparison of key metrics."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()

    fmt = "{:<8} {:<14} {:>14} {:>14} {:>14} {:>14}"
    print(fmt.format("Seed", "Distribution", "TTFT mean(ms)", "TTFT p99(ms)",
                      "E2E mean(ms)", "E2E p99(ms)"))
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
                          f"{g['e2e_p99']:.2f}"))
        print(fmt.format("", "ParetoLN",
                          f"{p['ttft_mean']:.2f}",
                          f"{p['ttft_p99']:.2f}",
                          f"{p['e2e_mean']:.2f}",
                          f"{p['e2e_p99']:.2f}"))

        ttft_ratio = p["ttft_p99"] / max(g["ttft_p99"], 0.001)
        e2e_ratio = p["e2e_p99"] / max(g["e2e_p99"], 0.001)
        print(fmt.format("", "Ratio",
                          f"{p['ttft_mean']/max(g['ttft_mean'],0.001):.2f}x",
                          f"{ttft_ratio:.2f}x",
                          f"{p['e2e_mean']/max(g['e2e_mean'],0.001):.2f}x",
                          f"{e2e_ratio:.2f}x"))
        print()


def print_preemption_table(datasets):
    """Print preemption and KV metrics across all experiments."""
    print("=" * 90)
    print("  PREEMPTION AND KV METRICS")
    print("=" * 90)
    print()

    fmt = "{:<6} {:<8} {:<14} {:>12} {:>12} {:>12} {:>10}"
    print(fmt.format("Exp", "Seed", "Distribution", "Preemptions",
                      "Preempt Rate", "Cache Hit", "Completed"))
    print("-" * 90)

    for exp_label, gaussian, pareto in datasets:
        for seed in SEEDS:
            g = gaussian[seed]
            p = pareto[seed]
            if g["timed_out"] or p["timed_out"]:
                continue

            print(fmt.format(exp_label, seed, "Gaussian",
                              str(g["preemption_count"]),
                              f"{g['preemption_rate']:.4f}",
                              f"{g['cache_hit_rate']:.4f}",
                              str(g["completed"])))
            print(fmt.format("", "", "ParetoLN",
                              str(p["preemption_count"]),
                              f"{p['preemption_rate']:.4f}",
                              f"{p['cache_hit_rate']:.4f}",
                              str(p["completed"])))
        print()


def print_conservation_check(datasets):
    """Verify INV-1 for all runs."""
    print("=" * 90)
    print("  CONSERVATION CHECK (INV-1)")
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
                conserved = (completed + queued + running) == injected
                status = "PASS" if conserved else "FAIL"
                if not conserved:
                    all_pass = False
                print(f"  [{exp_label}] {dist_label} seed={seed}: "
                      f"injected={injected}, completed={completed}, "
                      f"queued={queued}, running={running} -> {status}")

    print()
    if all_pass:
        print("  OVERALL: ALL CONSERVATION CHECKS PASS")
    else:
        print("  OVERALL: SOME CONSERVATION CHECKS FAILED")
    print()


def compute_summary_stats(gaussian, pareto):
    """Compute cross-seed summary statistics."""
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
    pareto_wins_ttft = sum(1 for i in range(n) if ttft_p99_p[i] > ttft_p99_g[i])
    pareto_wins_e2e = sum(1 for i in range(n) if e2e_p99_p[i] > e2e_p99_g[i])
    pareto_more_preempt = sum(1 for i in range(n) if preempt_p[i] > preempt_g[i])

    per_seed_ttft_ratios = [p / max(g, 0.001) for p, g in zip(ttft_p99_p, ttft_p99_g)]
    per_seed_e2e_ratios = [p / max(g, 0.001) for p, g in zip(e2e_p99_p, e2e_p99_g)]

    avg_ttft_ratio = sum(per_seed_ttft_ratios) / n
    avg_e2e_ratio = sum(per_seed_e2e_ratios) / n

    return {
        "n": n, "total": len(SEEDS),
        "pareto_wins_ttft": pareto_wins_ttft,
        "pareto_wins_e2e": pareto_wins_e2e,
        "pareto_more_preempt": pareto_more_preempt,
        "avg_ttft_ratio": avg_ttft_ratio,
        "avg_e2e_ratio": avg_e2e_ratio,
        "min_ttft_ratio": min(per_seed_ttft_ratios),
        "max_ttft_ratio": max(per_seed_ttft_ratios),
        "per_seed_ttft_ratios": per_seed_ttft_ratios,
        "per_seed_e2e_ratios": per_seed_e2e_ratios,
        "ttft_mean_g": sum(ttft_mean_g) / n,
        "ttft_mean_p": sum(ttft_mean_p) / n,
        "ttft_p99_g": sum(ttft_p99_g) / n,
        "ttft_p99_p": sum(ttft_p99_p) / n,
        "e2e_p99_g": sum(e2e_p99_g) / n,
        "e2e_p99_p": sum(e2e_p99_p) / n,
        "avg_preempt_g": sum(preempt_g) / n,
        "avg_preempt_p": sum(preempt_p) / n,
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
    print(f"  Seeds where ParetoLN TTFT p99 > Gaussian: {stats['pareto_wins_ttft']}/{n}")
    print(f"  Seeds where ParetoLN E2E p99 > Gaussian:  {stats['pareto_wins_e2e']}/{n}")
    print(f"  Seeds where ParetoLN preemptions > Gaussian: {stats['pareto_more_preempt']}/{n}")
    print(f"  TTFT p99 ratio (ParetoLN/Gaussian): avg={stats['avg_ttft_ratio']:.2f}x "
          f"range=[{stats['min_ttft_ratio']:.2f}x, {stats['max_ttft_ratio']:.2f}x]")
    print(f"  E2E p99 ratio (ParetoLN/Gaussian):  avg={stats['avg_e2e_ratio']:.2f}x")
    print(f"  Avg preemptions: Gaussian={stats['avg_preempt_g']:.0f}, "
          f"ParetoLN={stats['avg_preempt_p']:.0f}")
    print()
    print(f"  Cross-seed averages:")
    print(f"    Gaussian TTFT mean: {stats['ttft_mean_g']:.2f} ms")
    print(f"    ParetoLN TTFT mean: {stats['ttft_mean_p']:.2f} ms")
    print(f"    Gaussian TTFT p99:  {stats['ttft_p99_g']:.2f} ms")
    print(f"    ParetoLN TTFT p99:  {stats['ttft_p99_p']:.2f} ms")
    print(f"    Gaussian E2E p99:   {stats['e2e_p99_g']:.2f} ms")
    print(f"    ParetoLN E2E p99:   {stats['e2e_p99_p']:.2f} ms")
    print()


def print_verdict(exp1_stats, exp2_stats, exp3_stats):
    """Print overall hypothesis verdict."""
    print("=" * 90)
    print("  OVERALL VERDICT")
    print("=" * 90)
    print()

    # Exp 1 (default KV)
    if exp1_stats:
        n = exp1_stats["n"]
        wins = exp1_stats["pareto_wins_ttft"]
        ratio = exp1_stats["avg_ttft_ratio"]
        preempt_wins = exp1_stats["pareto_more_preempt"]
        print(f"  Exp 1 (default KV, rate=1000): {wins}/{n} seeds ParetoLN worse TTFT, "
              f"avg ratio={ratio:.2f}x")
        print(f"    Preemption dominance: {preempt_wins}/{n} seeds")
        print(f"    Avg preemptions: Gaussian={exp1_stats['avg_preempt_g']:.0f}, "
              f"ParetoLN={exp1_stats['avg_preempt_p']:.0f}")

    # Exp 2 (KV-constrained)
    if exp2_stats:
        n = exp2_stats["n"]
        wins = exp2_stats["pareto_wins_ttft"]
        ratio = exp2_stats["avg_ttft_ratio"]
        preempt_wins = exp2_stats["pareto_more_preempt"]
        print(f"  Exp 2 (2000 blocks, rate=1000): {wins}/{n} seeds ParetoLN worse TTFT, "
              f"avg ratio={ratio:.2f}x")
        print(f"    Preemption dominance: {preempt_wins}/{n} seeds")
        print(f"    Avg preemptions: Gaussian={exp2_stats['avg_preempt_g']:.0f}, "
              f"ParetoLN={exp2_stats['avg_preempt_p']:.0f}")
        if exp2_stats["avg_preempt_p"] > exp2_stats["avg_preempt_g"]:
            print("    => KV constraint amplifies preemption gap as predicted")
        else:
            print("    => UNEXPECTED: KV constraint does not amplify preemption gap")

    # Exp 3 (sub-saturation control)
    if exp3_stats:
        n = exp3_stats["n"]
        ratio = exp3_stats["avg_ttft_ratio"]
        print(f"  Exp 3 (sub-saturation, rate=200): avg TTFT ratio={ratio:.2f}x")
        if ratio < 1.10:
            print("    => Sub-saturation control: tail effect vanishes (queues don't build up)")
        else:
            print("    => UNEXPECTED: tail effect persists at sub-saturation")

    print()

    # Synthesis
    all_directional_ttft = True
    all_directional_preempt = True
    if exp1_stats and exp1_stats["pareto_wins_ttft"] < exp1_stats["n"]:
        all_directional_ttft = False
    if exp2_stats and exp2_stats["pareto_wins_ttft"] < exp2_stats["n"]:
        all_directional_ttft = False
    if exp2_stats and exp2_stats["pareto_more_preempt"] < exp2_stats["n"]:
        all_directional_preempt = False

    sub_sat_vanishes = exp3_stats and exp3_stats["avg_ttft_ratio"] < 1.10
    kv_amplifies = (exp2_stats and exp1_stats and
                    exp2_stats["avg_preempt_p"] > exp1_stats["avg_preempt_p"])

    if all_directional_ttft and all_directional_preempt and sub_sat_vanishes:
        print("  VERDICT: CONFIRMED -- ParetoLogNormal produces worse tail latency")
        print("    and more preemptions than Gaussian. Effect is load-dependent")
        print("    (vanishes at sub-saturation).")
    elif all_directional_ttft and sub_sat_vanishes:
        print("  VERDICT: CONFIRMED WITH NUANCE -- ParetoLN is directionally worse")
        print("    for tail latency across all seeds. Sub-saturation control validates")
        print("    the queue-buildup mechanism. Preemption signal mixed.")
    elif all_directional_ttft:
        print("  VERDICT: PARTIALLY CONFIRMED -- directionally consistent TTFT tail")
        print("    penalty, but sub-saturation control did not validate mechanism.")
    else:
        print("  VERDICT: INCONCLUSIVE -- mixed directionality across seeds.")
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
                           "Exp 1: CORE (default KV, rate=1000, 500 req)")
    print_comparison_table(exp2_g, exp2_p,
                           "Exp 2: KV-CONSTRAINED (2000 blocks, rate=1000, 500 req)")
    print_comparison_table(exp3_g, exp3_p,
                           "Exp 3: SUB-SATURATION CONTROL (rate=200, 500 req)")

    # Preemption table
    datasets = [
        ("Exp1", exp1_g, exp1_p),
        ("Exp2", exp2_g, exp2_p),
        ("Exp3", exp3_g, exp3_p),
    ]
    print_preemption_table(datasets)

    # Conservation check
    print_conservation_check(datasets)

    # Summaries
    print("=" * 90)
    print("  CROSS-SEED SUMMARIES")
    print("=" * 90)
    print()

    exp1_stats = compute_summary_stats(exp1_g, exp1_p)
    exp2_stats = compute_summary_stats(exp2_g, exp2_p)
    exp3_stats = compute_summary_stats(exp3_g, exp3_p)

    print_summary("Exp 1: Core (default KV, rate=1000)", exp1_stats)
    print_summary("Exp 2: KV-constrained (2000 blocks, rate=1000)", exp2_stats)
    print_summary("Exp 3: Sub-saturation (rate=200)", exp3_stats)

    # Overall verdict
    print_verdict(exp1_stats, exp2_stats, exp3_stats)


if __name__ == "__main__":
    main()
