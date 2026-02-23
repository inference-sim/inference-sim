#!/usr/bin/env python3
"""Analysis script for H23: Low-Load Routing Policy Equivalence.

Parses BLIS output for 4 routing policies at low-load (rate=1) and high-load
(rate=2000), comparing TTFT mean across policies within each seed.

Hypothesis: At very low load, all routing policies produce equivalent TTFT
(within 5%). At high load, policies diverge (>20%).

Equivalence criterion: max_deviation = (max(TTFT) - min(TTFT)) / mean(TTFT) < 5%
Divergence criterion: max_deviation > 20% at high load

Usage: python3 analyze.py <results_dir>
"""
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output

SEEDS = [42, 123, 456]
POLICIES = ["round-robin", "least-loaded", "weighted", "prefix-affinity"]
POLICY_SHORT = {"round-robin": "RR", "least-loaded": "LL", "weighted": "W", "prefix-affinity": "PA"}


def parse_dropped_unservable(filepath):
    """Extract dropped_unservable from cluster JSON block.

    The shared parse_blis_output helper does not include this field,
    so we parse it separately from the cluster JSON.
    """
    import json
    import re
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


def load_experiment(results_dir, prefix):
    """Load results for all policies and seeds with given prefix."""
    data = {}
    for policy in POLICIES:
        data[policy] = {}
        for seed in SEEDS:
            fname = Path(results_dir) / f"{prefix}_{policy}_{seed}.txt"
            metrics = parse_blis_output(str(fname))
            metrics["dropped_unservable"] = parse_dropped_unservable(str(fname))
            data[policy][seed] = metrics
    return data


def print_comparison_table(data, title, metric_key, metric_label):
    """Print per-seed comparison of a metric across all policies."""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()

    # Header
    hdr = "{:<8}".format("Seed")
    for policy in POLICIES:
        hdr += " {:>14}".format(POLICY_SHORT[policy])
    hdr += " {:>12} {:>8}".format("MaxDev%", "Status")
    print(hdr)
    print("-" * 80)

    all_pass = True
    deviations = []

    for seed in SEEDS:
        values = []
        any_timeout = False
        for policy in POLICIES:
            m = data[policy][seed]
            if m["timed_out"]:
                any_timeout = True
                break
            values.append(m[metric_key])

        if any_timeout:
            print(f"  Seed {seed}: SKIPPED (timeout)")
            continue

        mean_val = sum(values) / len(values) if values else 0
        if mean_val > 0:
            max_dev = (max(values) - min(values)) / mean_val * 100.0
        else:
            max_dev = 0.0

        deviations.append(max_dev)
        status = "PASS" if max_dev < 5.0 else "FAIL"
        if status == "FAIL":
            all_pass = False

        row = "{:<8}".format(seed)
        for v in values:
            row += " {:>14.2f}".format(v)
        row += " {:>11.2f}% {:>8}".format(max_dev, status)
        print(row)

    print()
    if deviations:
        avg_dev = sum(deviations) / len(deviations)
        max_of_devs = max(deviations)
        print(f"  Average max deviation: {avg_dev:.2f}%")
        print(f"  Worst-case deviation:  {max_of_devs:.2f}%")
        if all_pass:
            print(f"  => ALL seeds within 5% threshold: EQUIVALENT")
        else:
            print(f"  => Some seeds exceed 5% threshold: NOT EQUIVALENT")
    print()


def print_high_load_divergence(data, title, metric_key, metric_label):
    """Print high-load divergence table to validate comparison is meaningful."""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()

    hdr = "{:<8}".format("Seed")
    for policy in POLICIES:
        hdr += " {:>14}".format(POLICY_SHORT[policy])
    hdr += " {:>12} {:>8}".format("MaxDev%", "Status")
    print(hdr)
    print("-" * 80)

    any_diverge = False
    deviations = []

    for seed in SEEDS:
        values = []
        any_timeout = False
        for policy in POLICIES:
            m = data[policy][seed]
            if m["timed_out"]:
                any_timeout = True
                break
            values.append(m[metric_key])

        if any_timeout:
            print(f"  Seed {seed}: SKIPPED (timeout)")
            continue

        mean_val = sum(values) / len(values) if values else 0
        if mean_val > 0:
            max_dev = (max(values) - min(values)) / mean_val * 100.0
        else:
            max_dev = 0.0

        deviations.append(max_dev)
        status = "DIVERGE" if max_dev > 20.0 else "FLAT"
        if status == "DIVERGE":
            any_diverge = True

        row = "{:<8}".format(seed)
        for v in values:
            row += " {:>14.2f}".format(v)
        row += " {:>11.2f}% {:>8}".format(max_dev, status)
        print(row)

    print()
    if deviations:
        avg_dev = sum(deviations) / len(deviations)
        max_of_devs = max(deviations)
        print(f"  Average max deviation: {avg_dev:.2f}%")
        print(f"  Worst-case deviation:  {max_of_devs:.2f}%")
        if any_diverge:
            print(f"  => High-load control: policies DIVERGE as expected (validates comparison)")
        else:
            print(f"  => WARNING: policies do NOT diverge at high load (test may be insensitive)")
    print()


def print_conservation_check(datasets):
    """Verify INV-1 for all runs."""
    print("=" * 80)
    print("  CONSERVATION CHECK (INV-1)")
    print("=" * 80)
    print()

    all_pass = True
    for exp_label, data in datasets:
        for policy in POLICIES:
            for seed in SEEDS:
                m = data[policy][seed]
                if m["timed_out"]:
                    print(f"  [{exp_label}] {POLICY_SHORT[policy]} seed={seed}: SKIPPED (timeout)")
                    continue
                injected = m["injected"]
                completed = m["completed"]
                queued = m["still_queued"]
                running = m["still_running"]
                dropped = m.get("dropped_unservable", 0)
                conserved = (completed + queued + running + dropped) == injected
                status = "PASS" if conserved else "FAIL"
                if not conserved:
                    all_pass = False
                print(f"  [{exp_label}] {POLICY_SHORT[policy]} seed={seed}: "
                      f"injected={injected}, completed={completed}, "
                      f"queued={queued}, running={running}, dropped={dropped} -> {status}")
        print()

    if all_pass:
        print("  OVERALL: ALL CONSERVATION CHECKS PASS")
    else:
        print("  OVERALL: SOME CONSERVATION CHECKS FAILED")
    print()


def print_detailed_metrics(data, title):
    """Print full metrics table for inspection."""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()

    fmt = "{:<6} {:<5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Seed", "Pol", "TTFT mean", "TTFT p99", "E2E mean", "E2E p99",
                      "Thruput", "Completed"))
    print("-" * 80)

    for seed in SEEDS:
        for policy in POLICIES:
            m = data[policy][seed]
            if m["timed_out"]:
                print(f"  {seed} {POLICY_SHORT[policy]}: TIMED OUT")
                continue
            print(fmt.format(
                seed, POLICY_SHORT[policy],
                f"{m['ttft_mean']:.2f}",
                f"{m['ttft_p99']:.2f}",
                f"{m['e2e_mean']:.2f}",
                f"{m['e2e_p99']:.2f}",
                f"{m['throughput']:.2f}",
                str(m['completed']),
            ))
        print()


def print_verdict(low_data, high_data):
    """Print overall hypothesis verdict."""
    print("=" * 80)
    print("  OVERALL VERDICT")
    print("=" * 80)
    print()

    # Check low-load equivalence (TTFT mean)
    low_deviations = []
    for seed in SEEDS:
        values = []
        any_timeout = False
        for policy in POLICIES:
            m = low_data[policy][seed]
            if m["timed_out"]:
                any_timeout = True
                break
            values.append(m["ttft_mean"])
        if any_timeout:
            continue
        mean_val = sum(values) / len(values) if values else 0
        if mean_val > 0:
            max_dev = (max(values) - min(values)) / mean_val * 100.0
        else:
            max_dev = 0.0
        low_deviations.append(max_dev)

    # Check high-load divergence (TTFT mean)
    high_deviations = []
    for seed in SEEDS:
        values = []
        any_timeout = False
        for policy in POLICIES:
            m = high_data[policy][seed]
            if m["timed_out"]:
                any_timeout = True
                break
            values.append(m["ttft_mean"])
        if any_timeout:
            continue
        mean_val = sum(values) / len(values) if values else 0
        if mean_val > 0:
            max_dev = (max(values) - min(values)) / mean_val * 100.0
        else:
            max_dev = 0.0
        high_deviations.append(max_dev)

    low_equiv = all(d < 5.0 for d in low_deviations) if low_deviations else False
    high_diverge = any(d > 20.0 for d in high_deviations) if high_deviations else False

    if low_deviations:
        worst_low = max(low_deviations)
        print(f"  Low-load (rate=1):  worst deviation = {worst_low:.2f}%  "
              f"{'< 5% EQUIVALENT' if low_equiv else '>= 5% NOT EQUIVALENT'}")
    else:
        print("  Low-load: ALL TIMED OUT")

    if high_deviations:
        worst_high = max(high_deviations)
        print(f"  High-load (rate=2000): worst deviation = {worst_high:.2f}%  "
              f"{'> 20% DIVERGE (control validates)' if high_diverge else '<= 20% FLAT (control fails)'}")
    else:
        print("  High-load: ALL TIMED OUT")

    print()

    if low_equiv and high_diverge:
        print("  VERDICT: CONFIRMED")
        print("    All routing policies produce equivalent TTFT at near-zero load.")
        print("    High-load control confirms the comparison is meaningful.")
    elif low_equiv and not high_diverge:
        print("  VERDICT: INCONCLUSIVE")
        print("    Low-load equivalence holds, but high-load control does not diverge.")
        print("    The comparison may be insensitive to routing policy differences.")
    elif not low_equiv and high_diverge:
        print("  VERDICT: REFUTED")
        print("    Routing policies produce different TTFT even at near-zero load.")
        print("    This indicates a bug in routing or snapshot logic.")
    else:
        print("  VERDICT: INCONCLUSIVE")
        print("    Neither equivalence nor divergence criteria met.")
    print()

    # Also check E2E mean equivalence at low load
    low_e2e_devs = []
    for seed in SEEDS:
        values = []
        any_timeout = False
        for policy in POLICIES:
            m = low_data[policy][seed]
            if m["timed_out"]:
                any_timeout = True
                break
            values.append(m["e2e_mean"])
        if any_timeout:
            continue
        mean_val = sum(values) / len(values) if values else 0
        if mean_val > 0:
            max_dev = (max(values) - min(values)) / mean_val * 100.0
        else:
            max_dev = 0.0
        low_e2e_devs.append(max_dev)

    if low_e2e_devs:
        worst_e2e = max(low_e2e_devs)
        e2e_equiv = all(d < 5.0 for d in low_e2e_devs)
        print(f"  Low-load E2E mean: worst deviation = {worst_e2e:.2f}%  "
              f"{'< 5% EQUIVALENT' if e2e_equiv else '>= 5% NOT EQUIVALENT'}")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # Load all experiments
    low_data = load_experiment(results_dir, "exp1")
    high_data = load_experiment(results_dir, "exp2")

    # Detailed metrics tables
    print_detailed_metrics(low_data, "Exp 1: LOW-LOAD DETAILED METRICS (rate=1, 50 req)")
    print_detailed_metrics(high_data, "Exp 2: HIGH-LOAD DETAILED METRICS (rate=2000, 500 req)")

    # Low-load equivalence tables
    print_comparison_table(low_data,
                           "Exp 1: LOW-LOAD TTFT MEAN EQUIVALENCE (rate=1, 50 req)",
                           "ttft_mean", "TTFT mean (ms)")
    print_comparison_table(low_data,
                           "Exp 1: LOW-LOAD E2E MEAN EQUIVALENCE (rate=1, 50 req)",
                           "e2e_mean", "E2E mean (ms)")

    # High-load divergence tables
    print_high_load_divergence(high_data,
                                "Exp 2: HIGH-LOAD TTFT MEAN DIVERGENCE (rate=2000, 500 req)",
                                "ttft_mean", "TTFT mean (ms)")
    print_high_load_divergence(high_data,
                                "Exp 2: HIGH-LOAD E2E MEAN DIVERGENCE (rate=2000, 500 req)",
                                "e2e_mean", "E2E mean (ms)")

    # Conservation check
    datasets = [
        ("Low", low_data),
        ("High", high_data),
    ]
    print_conservation_check(datasets)

    # Overall verdict
    print_verdict(low_data, high_data)


if __name__ == "__main__":
    main()
