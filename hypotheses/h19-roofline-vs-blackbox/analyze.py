#!/usr/bin/env python3
"""H19: Roofline vs Blackbox Mode — Policy Ranking Equivalence Analysis.

Compares TTFT rankings across routing policies between blackbox and roofline
latency modes. The hypothesis: roofline should produce different absolute
latencies but the SAME relative policy ranking as blackbox.

Usage: python3 analyze.py <results_dir>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout

SEEDS = [42, 123, 456]
POLICIES = ["round-robin", "least-loaded", "weighted"]
MODES = ["blackbox", "roofline"]
CONTROL_MODE = "alpha0"  # Round 2: blackbox with alpha=0, real beta


def load_results(results_dir):
    """Load all experiment results into a nested dict: mode -> policy -> seed -> metrics."""
    results = {}
    for mode in MODES:
        results[mode] = {}
        for policy in POLICIES:
            results[mode][policy] = {}
            for seed in SEEDS:
                fname = f"{mode}_{policy}_s{seed}.txt"
                fpath = Path(results_dir) / fname
                metrics = parse_blis_output(str(fpath))
                results[mode][policy][seed] = metrics
    return results


def load_control_results(results_dir):
    """Load alpha=0 control results: policy -> seed -> metrics."""
    results = {}
    for policy in POLICIES:
        results[policy] = {}
        for seed in SEEDS:
            fname = f"{CONTROL_MODE}_{policy}_s{seed}.txt"
            fpath = Path(results_dir) / fname
            if fpath.exists():
                metrics = parse_blis_output(str(fpath))
            else:
                metrics = {"timed_out": True}
            results[policy][seed] = metrics
    return results


def compute_ranking_from_flat(flat_results, seed, metric_key):
    """Compute ranking from flat dict (policy -> seed -> metrics)."""
    scores = []
    for policy in POLICIES:
        m = flat_results[policy][seed]
        if m["timed_out"]:
            scores.append((policy, float("inf")))
        else:
            scores.append((policy, m[metric_key]))
    scores.sort(key=lambda x: x[1])
    return scores


def compute_ranking(results, mode, seed, metric_key):
    """Compute policy ranking for a given mode/seed by a metric (lower is better).

    Returns a list of (policy, value) sorted by value ascending (best first).
    """
    scores = []
    for policy in POLICIES:
        m = results[mode][policy][seed]
        if m["timed_out"]:
            scores.append((policy, float("inf")))
        else:
            scores.append((policy, m[metric_key]))
    scores.sort(key=lambda x: x[1])
    return scores


def rankings_match(ranking_a, ranking_b):
    """Check if two rankings have the same ordering of policies."""
    return [p for p, _ in ranking_a] == [p for p, _ in ranking_b]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]
    results = load_results(results_dir)

    # Check for any failed runs
    failed = []
    for mode in MODES:
        for policy in POLICIES:
            for seed in SEEDS:
                if results[mode][policy][seed]["timed_out"]:
                    failed.append(f"{mode}/{policy}/s{seed}")
    if failed:
        print(f"WARNING: {len(failed)} runs failed: {', '.join(failed)}")
        print()

    # --- Table 1: Absolute latencies ---
    print("=" * 80)
    print("TABLE 1: Absolute Latencies (TTFT mean / TTFT p99 in ms)")
    print("=" * 80)
    header = f"{'Seed':>6} {'Policy':<15} {'Blackbox Mean':>14} {'Roofline Mean':>14} {'BB p99':>10} {'RF p99':>10}"
    print(header)
    print("-" * 80)
    for seed in SEEDS:
        for policy in POLICIES:
            bb = results["blackbox"][policy][seed]
            rf = results["roofline"][policy][seed]
            bb_mean = f"{bb['ttft_mean']:.2f}" if not bb["timed_out"] else "FAIL"
            rf_mean = f"{rf['ttft_mean']:.2f}" if not rf["timed_out"] else "FAIL"
            bb_p99 = f"{bb['ttft_p99']:.2f}" if not bb["timed_out"] else "FAIL"
            rf_p99 = f"{rf['ttft_p99']:.2f}" if not rf["timed_out"] else "FAIL"
            print(f"{seed:>6} {policy:<15} {bb_mean:>14} {rf_mean:>14} {bb_p99:>10} {rf_p99:>10}")
        print()

    # --- Table 2: E2E latencies ---
    print("=" * 80)
    print("TABLE 2: E2E Latencies (mean / p99 in ms)")
    print("=" * 80)
    header = f"{'Seed':>6} {'Policy':<15} {'Blackbox Mean':>14} {'Roofline Mean':>14} {'BB p99':>10} {'RF p99':>10}"
    print(header)
    print("-" * 80)
    for seed in SEEDS:
        for policy in POLICIES:
            bb = results["blackbox"][policy][seed]
            rf = results["roofline"][policy][seed]
            bb_mean = f"{bb['e2e_mean']:.2f}" if not bb["timed_out"] else "FAIL"
            rf_mean = f"{rf['e2e_mean']:.2f}" if not rf["timed_out"] else "FAIL"
            bb_p99 = f"{bb['e2e_p99']:.2f}" if not bb["timed_out"] else "FAIL"
            rf_p99 = f"{rf['e2e_p99']:.2f}" if not rf["timed_out"] else "FAIL"
            print(f"{seed:>6} {policy:<15} {bb_mean:>14} {rf_mean:>14} {bb_p99:>10} {rf_p99:>10}")
        print()

    # --- Table 3: Throughput ---
    print("=" * 80)
    print("TABLE 3: Throughput (responses/sec)")
    print("=" * 80)
    header = f"{'Seed':>6} {'Policy':<15} {'Blackbox':>12} {'Roofline':>12}"
    print(header)
    print("-" * 80)
    for seed in SEEDS:
        for policy in POLICIES:
            bb = results["blackbox"][policy][seed]
            rf = results["roofline"][policy][seed]
            bb_tp = f"{bb['throughput']:.2f}" if not bb["timed_out"] else "FAIL"
            rf_tp = f"{rf['throughput']:.2f}" if not rf["timed_out"] else "FAIL"
            print(f"{seed:>6} {policy:<15} {bb_tp:>12} {rf_tp:>12}")
        print()

    # --- Ranking comparison (core hypothesis test) ---
    print("=" * 80)
    print("RANKING COMPARISON: Do blackbox and roofline produce the same policy ordering?")
    print("=" * 80)

    metrics_to_compare = [
        ("ttft_mean", "TTFT Mean"),
        ("ttft_p99", "TTFT P99"),
        ("e2e_mean", "E2E Mean"),
        ("e2e_p99", "E2E P99"),
    ]

    total_comparisons = 0
    matches = 0
    mismatches = []

    for metric_key, metric_label in metrics_to_compare:
        print(f"\n--- {metric_label} Rankings (lower = better) ---")
        for seed in SEEDS:
            bb_ranking = compute_ranking(results, "blackbox", seed, metric_key)
            rf_ranking = compute_ranking(results, "roofline", seed, metric_key)

            bb_order = " < ".join(f"{p}({v:.1f})" for p, v in bb_ranking)
            rf_order = " < ".join(f"{p}({v:.1f})" for p, v in rf_ranking)

            match = rankings_match(bb_ranking, rf_ranking)
            total_comparisons += 1
            if match:
                matches += 1
                status = "MATCH"
            else:
                status = "MISMATCH"
                mismatches.append(f"{metric_label}/seed={seed}")

            print(f"  Seed {seed}: {status}")
            print(f"    Blackbox:  {bb_order}")
            print(f"    Roofline:  {rf_order}")

    # --- Verdict ---
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"Ranking matches: {matches}/{total_comparisons}")

    if total_comparisons == 0:
        print("INCONCLUSIVE: No valid comparisons (all runs failed)")
        verdict = "INCONCLUSIVE"
    elif matches == total_comparisons:
        print("SUPPORTED: Roofline and blackbox produce identical policy rankings")
        print("across all seeds and metrics.")
        verdict = "SUPPORTED"
    elif matches >= total_comparisons * 0.75:
        print("PARTIALLY SUPPORTED: Most rankings match, but some differences observed.")
        print(f"Mismatches: {', '.join(mismatches)}")
        verdict = "PARTIALLY_SUPPORTED"
    else:
        print("REFUTED: Roofline and blackbox produce different policy rankings.")
        print(f"Mismatches: {', '.join(mismatches)}")
        verdict = "REFUTED"

    # --- Check that absolute values actually differ ---
    print()
    print("=" * 80)
    print("SANITY CHECK: Do absolute values differ between modes?")
    print("=" * 80)
    mode_differs = False
    for seed in SEEDS:
        for policy in POLICIES:
            bb = results["blackbox"][policy][seed]
            rf = results["roofline"][policy][seed]
            if bb["timed_out"] or rf["timed_out"]:
                continue
            if bb["ttft_mean"] != rf["ttft_mean"]:
                mode_differs = True
                ratio = rf["ttft_mean"] / bb["ttft_mean"] if bb["ttft_mean"] > 0 else float("inf")
                print(f"  {policy}/s{seed}: blackbox TTFT={bb['ttft_mean']:.2f}ms, "
                      f"roofline TTFT={rf['ttft_mean']:.2f}ms (ratio={ratio:.3f})")

    if mode_differs:
        print("  Absolute values DO differ (expected).")
    else:
        print("  WARNING: Absolute values are IDENTICAL — modes may not be different!")
        print("  Check stderr for roofline activation messages.")

    # === Experiment 2: Alpha=0 Control (RCV-4) ===
    control = load_control_results(results_dir)
    has_control = any(
        not control[p][s]["timed_out"]
        for p in POLICIES for s in SEEDS
        if s in control.get(p, {})
    )

    if has_control:
        print()
        print("=" * 80)
        print("EXPERIMENT 2: Alpha=0 Control (RCV-4)")
        print("Blackbox with alpha=0, real beta — isolates alpha overhead effect")
        print("=" * 80)

        # Table: Alpha=0 vs Roofline absolute values
        print()
        print(f"{'Seed':>6} {'Policy':<15} {'Alpha0 TTFT':>12} {'RF TTFT':>12} "
              f"{'A0 p99':>10} {'RF p99':>10} {'A0 E2E':>10} {'RF E2E':>10}")
        print("-" * 90)
        for seed in SEEDS:
            for policy in POLICIES:
                a0 = control[policy][seed]
                rf = results["roofline"][policy][seed]
                a0_mean = f"{a0['ttft_mean']:.2f}" if not a0["timed_out"] else "FAIL"
                rf_mean = f"{rf['ttft_mean']:.2f}" if not rf["timed_out"] else "FAIL"
                a0_p99 = f"{a0['ttft_p99']:.2f}" if not a0["timed_out"] else "FAIL"
                rf_p99 = f"{rf['ttft_p99']:.2f}" if not rf["timed_out"] else "FAIL"
                a0_e2e = f"{a0['e2e_mean']:.2f}" if not a0["timed_out"] else "FAIL"
                rf_e2e = f"{rf['e2e_mean']:.2f}" if not rf["timed_out"] else "FAIL"
                print(f"{seed:>6} {policy:<15} {a0_mean:>12} {rf_mean:>12} "
                      f"{a0_p99:>10} {rf_p99:>10} {a0_e2e:>10} {rf_e2e:>10}")
            print()

        # Ranking comparison: alpha0 vs roofline
        print("--- Alpha=0 vs Roofline P99 Rankings ---")
        control_matches = 0
        control_total = 0
        control_mismatches = []

        for metric_key, metric_label in [("ttft_p99", "TTFT P99"), ("e2e_p99", "E2E P99")]:
            print(f"\n  {metric_label}:")
            for seed in SEEDS:
                a0_ranking = compute_ranking_from_flat(control, seed, metric_key)
                rf_ranking = compute_ranking(results, "roofline", seed, metric_key)

                a0_order = " < ".join(f"{p}({v:.1f})" for p, v in a0_ranking)
                rf_order = " < ".join(f"{p}({v:.1f})" for p, v in rf_ranking)

                match = rankings_match(a0_ranking, rf_ranking)
                control_total += 1
                if match:
                    control_matches += 1
                    status = "MATCH"
                else:
                    status = "MISMATCH"
                    control_mismatches.append(f"{metric_label}/seed={seed}")

                print(f"    Seed {seed}: {status}")
                print(f"      Alpha0:   {a0_order}")
                print(f"      Roofline: {rf_order}")

        # Also compare alpha0 vs blackbox (full alpha) P99 to show the shift
        print("\n--- Alpha=0 vs Full-Blackbox P99 Rankings ---")
        for metric_key, metric_label in [("ttft_p99", "TTFT P99")]:
            print(f"\n  {metric_label}:")
            for seed in SEEDS:
                a0_ranking = compute_ranking_from_flat(control, seed, metric_key)
                bb_ranking = compute_ranking(results, "blackbox", seed, metric_key)

                a0_order = " < ".join(f"{p}({v:.1f})" for p, v in a0_ranking)
                bb_order = " < ".join(f"{p}({v:.1f})" for p, v in bb_ranking)

                match = rankings_match(a0_ranking, bb_ranking)
                status = "MATCH" if match else "MISMATCH"

                print(f"    Seed {seed}: {status}")
                print(f"      Alpha0:   {a0_order}")
                print(f"      Blackbox: {bb_order}")

        print()
        print(f"Control result: alpha0 vs roofline P99 ranking matches: "
              f"{control_matches}/{control_total}")
        if control_total > 0 and control_matches == control_total:
            print("MECHANISM CONFIRMED: Removing alpha overhead makes blackbox P99 rankings")
            print("match roofline. Alpha overhead is the cause of P99 ranking divergence.")
        elif control_total > 0 and control_matches >= control_total * 0.75:
            print("MECHANISM PARTIALLY CONFIRMED: Most alpha0 P99 rankings match roofline.")
            if control_mismatches:
                print(f"Remaining mismatches: {', '.join(control_mismatches)}")
        elif control_total > 0:
            print("MECHANISM NOT CONFIRMED: Alpha0 blackbox P99 rankings still diverge from roofline.")
            print("The step-time model difference (beta regression vs roofline FLOPs) also contributes.")
            if control_mismatches:
                print(f"Mismatches: {', '.join(control_mismatches)}")
    else:
        print()
        print("(Alpha=0 control not run — skipping Experiment 2 analysis)")

    print()
    print(f"FINAL: {verdict}")
    return 0 if verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED") else 1


if __name__ == "__main__":
    sys.exit(main())
