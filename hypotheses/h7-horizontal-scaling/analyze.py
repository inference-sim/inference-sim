#!/usr/bin/env python3
"""Analysis script for H7: Horizontal Scaling.

Parses BLIS output for different instance counts across multiple seeds
and produces comparison tables.

Hypothesis: Increasing instances from 4 to 8 should roughly halve TTFT p99
for saturated workloads.

Experiments:
  Exp 1 (sat): Saturation — rate=500, 500 requests, instances=2,4,8
  Exp 2 (sub): Sub-saturation control — rate=100, 500 requests, instances=2,4,8

Statistical note: scipy is not available in this environment. Significance
thresholds are assessed by consistent directionality across seeds and
effect size magnitude (legacy threshold per docs/standards/experiments.md).

Usage: python3 analyze.py <results_dir>
"""
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output

SEEDS = [42, 123, 456]
INSTANCE_COUNTS = [2, 4, 8]


def load_experiment(results_dir, prefix):
    """Load results for all instance counts and seeds."""
    data = {}
    for inst in INSTANCE_COUNTS:
        data[inst] = {}
        for seed in SEEDS:
            fname = Path(results_dir) / f"{prefix}_inst{inst}_seed{seed}.txt"
            data[inst][seed] = parse_blis_output(str(fname))
    return data


def print_detail_table(data, title):
    """Print per-seed comparison table for all instance counts."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()

    fmt = "{:<10} {:<8} {:>14} {:>14} {:>14} {:>14} {:>10}"
    print(fmt.format("Instances", "Seed", "TTFT mean(ms)", "TTFT p99(ms)",
                      "E2E mean(ms)", "E2E p99(ms)", "Throughput"))
    print("-" * 90)

    for inst in INSTANCE_COUNTS:
        for seed in SEEDS:
            m = data[inst][seed]
            if m["timed_out"]:
                print(f"  instances={inst} seed={seed}: TIMED OUT -- skipping")
                continue
            print(fmt.format(inst, seed,
                              f"{m['ttft_mean']:.2f}",
                              f"{m['ttft_p99']:.2f}",
                              f"{m['e2e_mean']:.2f}",
                              f"{m['e2e_p99']:.2f}",
                              f"{m['throughput']:.2f}"))
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


def compute_scaling_ratios(data):
    """Compute per-seed scaling ratios between instance counts.

    Returns dict with entries like:
        ("4_vs_8", seed) -> { "ttft_p99_ratio": ..., "e2e_p99_ratio": ..., ... }

    Ratios are (fewer_instances / more_instances), so >1 means more instances is better.
    """
    ratios = {}
    pairs = [(2, 4), (4, 8), (2, 8)]
    for fewer, more in pairs:
        key = f"{fewer}_vs_{more}"
        for seed in SEEDS:
            m_fewer = data[fewer][seed]
            m_more = data[more][seed]
            if m_fewer["timed_out"] or m_more["timed_out"]:
                continue

            def safe_ratio(a, b):
                return a / b if b > 0 else float("inf")

            ratios[(key, seed)] = {
                "ttft_p99_ratio": safe_ratio(m_fewer["ttft_p99"], m_more["ttft_p99"]),
                "ttft_mean_ratio": safe_ratio(m_fewer["ttft_mean"], m_more["ttft_mean"]),
                "e2e_p99_ratio": safe_ratio(m_fewer["e2e_p99"], m_more["e2e_p99"]),
                "e2e_mean_ratio": safe_ratio(m_fewer["e2e_mean"], m_more["e2e_mean"]),
                "throughput_ratio": safe_ratio(m_more["throughput"], m_fewer["throughput"]),
            }
    return ratios


def print_scaling_table(ratios, title):
    """Print scaling ratio table across instance count pairs."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()
    print("  Ratios: TTFT/E2E = (fewer_inst / more_inst), >1 means scaling helps.")
    print("  Throughput ratio = (more_inst / fewer_inst), >1 means scaling helps.")
    print()

    fmt = "{:<12} {:<8} {:>16} {:>16} {:>14} {:>14} {:>14}"
    print(fmt.format("Comparison", "Seed", "TTFT p99 ratio", "TTFT mean ratio",
                      "E2E p99 ratio", "E2E mean ratio", "Tput ratio"))
    print("-" * 90)

    pairs = ["2_vs_4", "4_vs_8", "2_vs_8"]
    for pair in pairs:
        seed_ratios = [(seed, ratios[(pair, seed)])
                       for seed in SEEDS if (pair, seed) in ratios]
        if not seed_ratios:
            print(f"  {pair}: no valid data")
            continue

        for seed, r in seed_ratios:
            print(fmt.format(pair, seed,
                              f"{r['ttft_p99_ratio']:.3f}x",
                              f"{r['ttft_mean_ratio']:.3f}x",
                              f"{r['e2e_p99_ratio']:.3f}x",
                              f"{r['e2e_mean_ratio']:.3f}x",
                              f"{r['throughput_ratio']:.3f}x"))

        # Cross-seed average
        n = len(seed_ratios)
        avg = {}
        for metric in ["ttft_p99_ratio", "ttft_mean_ratio", "e2e_p99_ratio",
                        "e2e_mean_ratio", "throughput_ratio"]:
            avg[metric] = sum(r[metric] for _, r in seed_ratios) / n

        print(fmt.format(pair, "AVG",
                          f"{avg['ttft_p99_ratio']:.3f}x",
                          f"{avg['ttft_mean_ratio']:.3f}x",
                          f"{avg['e2e_p99_ratio']:.3f}x",
                          f"{avg['e2e_mean_ratio']:.3f}x",
                          f"{avg['throughput_ratio']:.3f}x"))
        print()


def print_precondition_check(sat_data):
    """Verify saturation precondition: TTFT p99 at 4 instances >> bare prefill time.

    Bare prefill time for 512 tokens = beta0 + beta1*512 = 6910.42 + 17.67*512 = 15958.46 us ~ 16ms.
    If TTFT p99 is >5x this, the system is clearly saturated (queuing dominates).
    """
    bare_prefill_ms = 15.96  # beta0 + beta1*512 in ms
    print("=" * 90)
    print("  PRECONDITION CHECK: Is 4-instance config saturated?")
    print("=" * 90)
    print()
    print(f"  Bare prefill time (512 tokens): ~{bare_prefill_ms:.1f} ms")
    print(f"  Saturation threshold: TTFT p99 > 5x bare prefill = {bare_prefill_ms * 5:.1f} ms")
    print()

    all_saturated = True
    for seed in SEEDS:
        m = sat_data[4][seed]
        if m["timed_out"]:
            print(f"  Seed {seed}: TIMED OUT")
            all_saturated = False
            continue
        ttft_p99 = m["ttft_p99"]
        ratio = ttft_p99 / bare_prefill_ms
        saturated = ratio > 5.0
        status = "SATURATED" if saturated else "NOT SATURATED"
        if not saturated:
            all_saturated = False
        print(f"  Seed {seed}: TTFT p99 = {ttft_p99:.2f} ms "
              f"({ratio:.1f}x bare prefill) -> {status}")

    print()
    if all_saturated:
        print("  PRECONDITION: SATISFIED -- 4-instance config is saturated")
    else:
        print("  PRECONDITION: NOT SATISFIED -- 4-instance config may not be saturated")
    print()


def print_verdict(sat_ratios, sub_ratios):
    """Print overall hypothesis verdict."""
    print("=" * 90)
    print("  OVERALL VERDICT")
    print("=" * 90)
    print()

    # Saturation: 4 vs 8 comparison
    pair = "4_vs_8"
    sat_seeds = [(seed, sat_ratios[(pair, seed)])
                 for seed in SEEDS if (pair, seed) in sat_ratios]

    if not sat_seeds:
        print("  No valid saturation data for 4 vs 8 comparison.")
        return

    n = len(sat_seeds)
    ttft_ratios = [r["ttft_p99_ratio"] for _, r in sat_seeds]
    avg_ttft = sum(ttft_ratios) / n
    min_ttft = min(ttft_ratios)
    max_ttft = max(ttft_ratios)
    all_directional = all(r > 1.0 for r in ttft_ratios)
    all_significant = all(r > 1.20 for r in ttft_ratios)

    print(f"  Saturation (rate=500), 4 vs 8 instances:")
    print(f"    TTFT p99 scaling ratio: avg={avg_ttft:.3f}x, "
          f"range=[{min_ttft:.3f}x, {max_ttft:.3f}x]")
    print(f"    Directional ({n}/{n} seeds 8-inst better): "
          f"{'YES' if all_directional else 'NO'}")
    print(f"    Significant (>20% all seeds): "
          f"{'YES' if all_significant else 'NO'}")
    print()

    # Predicted: ~2x. Check if super-linear (>2x) or sub-linear (<2x)
    if avg_ttft > 2.0:
        print(f"    Scaling is SUPER-LINEAR ({avg_ttft:.3f}x > predicted 2.0x)")
    elif avg_ttft > 1.5:
        print(f"    Scaling is roughly as predicted ({avg_ttft:.3f}x ~ 2.0x)")
    elif avg_ttft > 1.2:
        print(f"    Scaling is SUB-LINEAR ({avg_ttft:.3f}x < predicted 2.0x)")
    else:
        print(f"    Scaling effect is SMALL ({avg_ttft:.3f}x << predicted 2.0x)")
    print()

    # Sub-saturation control
    sub_seeds = [(seed, sub_ratios[(pair, seed)])
                 for seed in SEEDS if (pair, seed) in sub_ratios]

    if sub_seeds:
        sub_ttft = [r["ttft_p99_ratio"] for _, r in sub_seeds]
        sub_avg = sum(sub_ttft) / len(sub_ttft)
        print(f"  Sub-saturation control (rate=100), 4 vs 8 instances:")
        print(f"    TTFT p99 scaling ratio: avg={sub_avg:.3f}x")
        if abs(sub_avg - 1.0) < 0.10:
            print("    => Control validates: scaling effect vanishes at sub-saturation")
            sub_vanishes = True
        else:
            print("    => UNEXPECTED: scaling effect persists at sub-saturation")
            sub_vanishes = False
    else:
        sub_vanishes = False
    print()

    # E2E scaling (expected to be less dramatic since decode dominates)
    e2e_ratios_sat = [sat_ratios[(pair, s)]["e2e_p99_ratio"]
                      for s in SEEDS if (pair, s) in sat_ratios]
    if e2e_ratios_sat:
        avg_e2e = sum(e2e_ratios_sat) / len(e2e_ratios_sat)
        print(f"  E2E p99 scaling ratio (4 vs 8, saturation): avg={avg_e2e:.3f}x")
        if avg_e2e < 1.10:
            print("    => E2E insensitive to scaling (decode-dominated, as expected)")
        elif avg_e2e < 1.50:
            print("    => E2E shows moderate scaling benefit (queue wait visible)")
        else:
            print("    => E2E shows strong scaling benefit")
    print()

    # Final verdict
    if all_significant and sub_vanishes:
        print("  VERDICT: CONFIRMED -- Horizontal scaling significantly reduces TTFT p99")
        print("    under saturation. Effect vanishes at sub-saturation, confirming")
        print("    mechanism is queue-depth reduction, not service-time change.")
    elif all_directional and sub_vanishes:
        print("  VERDICT: CONFIRMED WITH NUANCE -- Scaling is directionally consistent")
        print("    but some seeds below 20% threshold.")
    elif all_directional:
        print("  VERDICT: PARTIALLY CONFIRMED -- 8 instances better in all seeds,")
        print("    but sub-saturation control does not fully validate mechanism.")
    else:
        print("  VERDICT: INCONCLUSIVE -- mixed directionality across seeds")
    print()


def print_additional_metrics(datasets):
    """Print throughput, preemption, and cache info for confound checking."""
    print("=" * 90)
    print("  ADDITIONAL METRICS (confound check)")
    print("=" * 90)
    print()

    fmt = "{:<6} {:<10} {:<8} {:>12} {:>10} {:>12} {:>10}"
    print(fmt.format("Exp", "Instances", "Seed", "Throughput", "Completed",
                      "Preemptions", "Cache Hit"))
    print("-" * 90)

    for exp_label, data in datasets:
        for inst in INSTANCE_COUNTS:
            for seed in SEEDS:
                m = data[inst][seed]
                if m["timed_out"]:
                    continue
                print(fmt.format(exp_label, inst, seed,
                                  f"{m['throughput']:.2f}",
                                  str(m['completed']),
                                  str(m['preemption_count']),
                                  f"{m['cache_hit_rate']:.4f}"))
            print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # Load experiments
    sat_data = load_experiment(results_dir, "sat")
    sub_data = load_experiment(results_dir, "sub")

    # Detailed tables
    print_detail_table(sat_data,
                        "Exp 1: SATURATION (rate=500, 500 req, instances=2,4,8)")
    print_detail_table(sub_data,
                        "Exp 2: SUB-SATURATION CONTROL (rate=100, 500 req, instances=2,4,8)")

    # Conservation check
    print_conservation_check([
        ("Sat", sat_data),
        ("Sub", sub_data),
    ])

    # Precondition check
    print_precondition_check(sat_data)

    # Scaling ratios
    sat_ratios = compute_scaling_ratios(sat_data)
    sub_ratios = compute_scaling_ratios(sub_data)

    print_scaling_table(sat_ratios,
                         "SCALING RATIOS — SATURATION (rate=500)")
    print_scaling_table(sub_ratios,
                         "SCALING RATIOS — SUB-SATURATION (rate=100)")

    # Additional metrics
    print_additional_metrics([
        ("Sat", sat_data),
        ("Sub", sub_data),
    ])

    # Verdict
    print_verdict(sat_ratios, sub_ratios)


if __name__ == "__main__":
    main()
