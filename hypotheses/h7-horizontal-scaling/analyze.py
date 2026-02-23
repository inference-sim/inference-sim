#!/usr/bin/env python3
"""Analysis script for H7: Horizontal Scaling.

Parses BLIS output across instance counts (2, 4, 8) and seeds (42, 123, 456)
to test whether TTFT p99 decreases monotonically as instances increase.

Experiments:
  Exp 1: Scaling sweep — rate=1000, 500 requests (saturating load)
  Ctrl:  Sub-saturation control — rate=100, 500 requests

Statistical note: scipy is not available in this environment. Monotonicity is
assessed by consistent directionality across all seeds. Effect size thresholds:
>20% for dominance, <5% for equivalence (per docs/standards/experiments.md).

Usage: python3 analyze.py <results_dir>
"""
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output

INSTANCE_COUNTS = [2, 4, 8]
SEEDS = [42, 123, 456]


def load_experiment(results_dir, prefix):
    """Load results for all instance counts and seeds."""
    data = {}
    for inst in INSTANCE_COUNTS:
        data[inst] = {}
        for seed in SEEDS:
            fname = Path(results_dir) / f"{prefix}_inst{inst}_seed{seed}.txt"
            data[inst][seed] = parse_blis_output(str(fname))
    return data


def print_comparison_table(data, title):
    """Print per-seed comparison across instance counts."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()

    fmt = "{:<6} {:<8} {:>14} {:>14} {:>14} {:>14} {:>14}"
    print(fmt.format("Inst", "Seed", "TTFT mean(ms)", "TTFT p99(ms)",
                      "E2E mean(ms)", "E2E p99(ms)", "Throughput"))
    print("-" * 90)

    for seed in SEEDS:
        for inst in INSTANCE_COUNTS:
            m = data[inst][seed]
            if m["timed_out"]:
                print(f"  inst={inst} seed={seed}: TIMED OUT -- skipping")
                continue
            print(fmt.format(inst, seed,
                              f"{m['ttft_mean']:.2f}",
                              f"{m['ttft_p99']:.2f}",
                              f"{m['e2e_mean']:.2f}",
                              f"{m['e2e_p99']:.2f}",
                              f"{m['throughput']:.2f}"))
        print()


def print_scaling_ratios(data, title):
    """Print scaling ratios (4-vs-2 and 8-vs-4) for key metrics."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()

    fmt = "{:<8} {:>18} {:>18} {:>18} {:>18}"
    print(fmt.format("Seed", "TTFT p99 4/2", "TTFT p99 8/4",
                      "Thru 4/2", "Thru 8/4"))
    print("-" * 90)

    for seed in SEEDS:
        m2 = data[2][seed]
        m4 = data[4][seed]
        m8 = data[8][seed]
        if m2["timed_out"] or m4["timed_out"] or m8["timed_out"]:
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        ttft_4v2 = m4["ttft_p99"] / max(m2["ttft_p99"], 0.001)
        ttft_8v4 = m8["ttft_p99"] / max(m4["ttft_p99"], 0.001)
        thru_4v2 = m4["throughput"] / max(m2["throughput"], 0.001)
        thru_8v4 = m8["throughput"] / max(m4["throughput"], 0.001)

        print(fmt.format(seed,
                          f"{ttft_4v2:.3f}x",
                          f"{ttft_8v4:.3f}x",
                          f"{thru_4v2:.3f}x",
                          f"{thru_8v4:.3f}x"))
    print()


def print_conservation_check(datasets):
    """Verify INV-1 for all runs."""
    print("=" * 90)
    print("  CONSERVATION CHECK (INV-1)")
    print("=" * 90)
    print()

    all_pass = True
    for exp_label, data in datasets:
        for inst in INSTANCE_COUNTS:
            for seed in SEEDS:
                m = data[inst][seed]
                if m["timed_out"]:
                    print(f"  [{exp_label}] inst={inst} seed={seed}: SKIPPED (timeout)")
                    continue
                injected = m["injected"]
                completed = m["completed"]
                queued = m["still_queued"]
                running = m["still_running"]
                conserved = (completed + queued + running) == injected
                status = "PASS" if conserved else "FAIL"
                if not conserved:
                    all_pass = False
                print(f"  [{exp_label}] inst={inst} seed={seed}: "
                      f"injected={injected}, completed={completed}, "
                      f"queued={queued}, running={running} -> {status}")

    print()
    if all_pass:
        print("  OVERALL: ALL CONSERVATION CHECKS PASS")
    else:
        print("  OVERALL: SOME CONSERVATION CHECKS FAILED")
    print()


def print_instance_distribution(data, title):
    """Print per-instance request distribution from trace summary."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()

    # Show completed + preemption info per config
    fmt = "{:<6} {:<8} {:>12} {:>12} {:>12}"
    print(fmt.format("Inst", "Seed", "Completed", "Preemptions", "CacheHitRate"))
    print("-" * 90)

    for seed in SEEDS:
        for inst in INSTANCE_COUNTS:
            m = data[inst][seed]
            if m["timed_out"]:
                continue
            print(fmt.format(inst, seed,
                              str(m["completed"]),
                              str(m["preemption_count"]),
                              f"{m['cache_hit_rate']:.4f}"))
        print()


def compute_monotonicity_stats(data):
    """Compute monotonicity statistics across seeds.

    Returns dict with:
        monotonic_ttft_p99: count of seeds where TTFT p99 is strictly decreasing
        monotonic_throughput: count of seeds where throughput is strictly increasing
        per_seed: list of dicts with per-seed details
    """
    valid_seeds = []
    per_seed = []

    for seed in SEEDS:
        m2 = data[2][seed]
        m4 = data[4][seed]
        m8 = data[8][seed]
        if m2["timed_out"] or m4["timed_out"] or m8["timed_out"]:
            continue
        valid_seeds.append(seed)

        ttft_p99_vals = [m2["ttft_p99"], m4["ttft_p99"], m8["ttft_p99"]]
        thru_vals = [m2["throughput"], m4["throughput"], m8["throughput"]]

        # TTFT p99 should decrease: 2 > 4 > 8
        ttft_monotonic = (ttft_p99_vals[0] > ttft_p99_vals[1] > ttft_p99_vals[2])

        # Throughput should increase: 2 < 4 < 8
        thru_monotonic = (thru_vals[0] < thru_vals[1] < thru_vals[2])

        # Halving ratio: TTFT p99 at 8 inst / TTFT p99 at 4 inst
        halving_ratio = m8["ttft_p99"] / max(m4["ttft_p99"], 0.001)

        # Throughput scaling ratio: 8 inst / 4 inst (ideal = 2.0x)
        thru_scale = m8["throughput"] / max(m4["throughput"], 0.001)

        per_seed.append({
            "seed": seed,
            "ttft_p99": ttft_p99_vals,
            "throughput": thru_vals,
            "ttft_monotonic": ttft_monotonic,
            "thru_monotonic": thru_monotonic,
            "halving_ratio_8v4": halving_ratio,
            "thru_scale_8v4": thru_scale,
            "ttft_mean": [m2["ttft_mean"], m4["ttft_mean"], m8["ttft_mean"]],
            "e2e_p99": [m2["e2e_p99"], m4["e2e_p99"], m8["e2e_p99"]],
        })

    n = len(valid_seeds)
    mono_ttft = sum(1 for s in per_seed if s["ttft_monotonic"])
    mono_thru = sum(1 for s in per_seed if s["thru_monotonic"])

    return {
        "n": n,
        "total": len(SEEDS),
        "monotonic_ttft_p99": mono_ttft,
        "monotonic_throughput": mono_thru,
        "per_seed": per_seed,
    }


def print_summary(label, stats):
    """Print cross-seed summary for one experiment."""
    print(f"  --- {label} ---")
    if stats is None or stats["n"] == 0:
        print("  No valid results (all timed out).")
        print()
        return

    n = stats["n"]
    print(f"  Valid seeds: {n}/{stats['total']}")
    print(f"  Seeds with monotonically decreasing TTFT p99: "
          f"{stats['monotonic_ttft_p99']}/{n}")
    print(f"  Seeds with monotonically increasing throughput: "
          f"{stats['monotonic_throughput']}/{n}")
    print()

    print(f"  Per-seed details:")
    fmt = "    {:<8} {:>12} {:>12} {:>12} {:>12} {:>12}"
    print(fmt.format("Seed", "TTFT p99@2", "TTFT p99@4", "TTFT p99@8",
                      "8/4 ratio", "Thru 8/4"))
    print("    " + "-" * 72)
    for s in stats["per_seed"]:
        mono_flag = " (M)" if s["ttft_monotonic"] else ""
        print(fmt.format(
            f"{s['seed']}{mono_flag}",
            f"{s['ttft_p99'][0]:.1f}",
            f"{s['ttft_p99'][1]:.1f}",
            f"{s['ttft_p99'][2]:.1f}",
            f"{s['halving_ratio_8v4']:.3f}x",
            f"{s['thru_scale_8v4']:.3f}x"))
    print()

    # Average halving ratio
    avg_halving = sum(s["halving_ratio_8v4"] for s in stats["per_seed"]) / n
    avg_thru_scale = sum(s["thru_scale_8v4"] for s in stats["per_seed"]) / n
    print(f"  Average TTFT p99 ratio (8/4 instances): {avg_halving:.3f}x "
          f"(ideal=0.50x for halving)")
    print(f"  Average throughput scaling (8/4 instances): {avg_thru_scale:.3f}x "
          f"(ideal=2.00x for linear)")
    print()


def print_verdict(exp1_stats, ctrl_stats):
    """Print overall hypothesis verdict."""
    print("=" * 90)
    print("  OVERALL VERDICT")
    print("=" * 90)
    print()

    if exp1_stats is None or exp1_stats["n"] == 0:
        print("  VERDICT: INCONCLUSIVE -- no valid data")
        return

    n = exp1_stats["n"]
    mono_count = exp1_stats["monotonic_ttft_p99"]
    thru_count = exp1_stats["monotonic_throughput"]

    # Halving assessment
    halving_ratios = [s["halving_ratio_8v4"] for s in exp1_stats["per_seed"]]
    avg_halving = sum(halving_ratios) / n
    thru_scales = [s["thru_scale_8v4"] for s in exp1_stats["per_seed"]]
    avg_thru_scale = sum(thru_scales) / n

    print(f"  Saturating sweep (Exp 1, rate=1000):")
    print(f"    Monotonic TTFT p99 decrease: {mono_count}/{n} seeds")
    print(f"    Monotonic throughput increase: {thru_count}/{n} seeds")
    print(f"    Average TTFT p99 8/4 ratio: {avg_halving:.3f}x")
    print(f"    Average throughput 8/4 scaling: {avg_thru_scale:.3f}x")
    print()

    # Sub-saturation control assessment
    if ctrl_stats and ctrl_stats["n"] > 0:
        ctrl_halving = [s["halving_ratio_8v4"] for s in ctrl_stats["per_seed"]]
        ctrl_avg = sum(ctrl_halving) / ctrl_stats["n"]
        ctrl_ttft_range = []
        for s in ctrl_stats["per_seed"]:
            spread = max(s["ttft_p99"]) / max(min(s["ttft_p99"]), 0.001)
            ctrl_ttft_range.append(spread)
        ctrl_avg_spread = sum(ctrl_ttft_range) / ctrl_stats["n"]

        print(f"  Sub-saturation control (Ctrl, rate=100):")
        print(f"    Average TTFT p99 8/4 ratio: {ctrl_avg:.3f}x")
        print(f"    Average max/min TTFT p99 spread: {ctrl_avg_spread:.3f}x")
        sub_sat_flat = ctrl_avg_spread < 2.0
        if sub_sat_flat:
            print("    => Control: scaling effect diminished at sub-saturation (as expected)")
        else:
            print("    => UNEXPECTED: large TTFT spread even at sub-saturation")
        print()

    # Verdict
    all_monotonic = mono_count == n
    # "roughly halve" = TTFT p99 ratio < 0.65 (more lenient than exact 0.5)
    roughly_halved = avg_halving < 0.65

    if all_monotonic and roughly_halved:
        print("  VERDICT: CONFIRMED -- TTFT p99 decreases monotonically with instances,")
        print(f"    and 8 instances achieve {avg_halving:.3f}x of 4-instance TTFT p99")
        print(f"    (within the 'roughly halve' threshold of <0.65x)")
    elif all_monotonic:
        print("  VERDICT: CONFIRMED WITH NUANCE -- TTFT p99 decreases monotonically,")
        print(f"    but 8-vs-4 ratio is {avg_halving:.3f}x (not close to 0.5x halving).")
        print("    Scaling improves TTFT but sub-linearly.")
    elif mono_count > 0:
        print(f"  VERDICT: PARTIALLY CONFIRMED -- {mono_count}/{n} seeds show monotonic")
        print(f"    TTFT p99 decrease. Avg 8/4 ratio: {avg_halving:.3f}x")
    else:
        print("  VERDICT: REFUTED -- TTFT p99 does not decrease monotonically")
        print("    with increasing instance count under saturation.")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # Load experiments
    exp1 = load_experiment(results_dir, "exp1")
    ctrl = load_experiment(results_dir, "ctrl")

    # Comparison tables
    print_comparison_table(exp1, "Exp 1: Scaling sweep at rate=1000 (saturating)")
    print_comparison_table(ctrl, "Ctrl: Sub-saturation at rate=100")

    # Scaling ratios
    print_scaling_ratios(exp1, "Exp 1: Scaling Ratios (saturating)")
    print_scaling_ratios(ctrl, "Ctrl: Scaling Ratios (sub-saturation)")

    # Conservation check
    datasets = [("Exp1", exp1), ("Ctrl", ctrl)]
    print_conservation_check(datasets)

    # Additional metrics
    print_instance_distribution(exp1, "Exp 1: Additional Metrics (confound check)")

    # Summaries
    print("=" * 90)
    print("  CROSS-SEED SUMMARIES")
    print("=" * 90)
    print()

    exp1_stats = compute_monotonicity_stats(exp1)
    ctrl_stats = compute_monotonicity_stats(ctrl)

    print_summary("Exp 1: Saturating (rate=1000, 500 req)", exp1_stats)
    print_summary("Ctrl: Sub-saturation (rate=100, 500 req)", ctrl_stats)

    # Overall verdict
    print_verdict(exp1_stats, ctrl_stats)


if __name__ == "__main__":
    main()
