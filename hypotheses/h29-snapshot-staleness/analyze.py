#!/usr/bin/env python3
"""H29: Stale Routing Snapshots Degrade Tail Latency — Analysis

Parses BLIS output and per-request JSON results from all experiments:
  - Exp 1: kv-utilization:1 (staleness-sensitive) — fresh vs stale, 3 seeds
  - Exp 2: queue-depth:1 (negative control) — fresh vs stale, 3 seeds
  - Exp 3: composite kv-util+queue-depth — fresh vs stale, 3 seeds
  - Exp 4: interval sweep (kv-utilization:1, seed=42)

Computes:
  - TTFT p50, p99, mean for each config and seed
  - E2E p50, p99, mean for comparison
  - Per-instance request distribution (Jain fairness index) from per-request JSON
  - Comparison tables with percent change
  - Verdict based on 20% confirmation / 10% refutation thresholds

Note: --results-path produces a single JSON object (MetricsOutput) with a
"requests" array containing per-request RequestMetrics. Each request has
"ttft_ms", "e2e_ms", "handled_by" (instance ID). This is NOT JSON Lines.
"""

import json
import math
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout


def load_per_request_results(filepath):
    """Load per-request results from a BLIS --results-path JSON file.

    The file is a single JSON object (MetricsOutput) with a "requests" array.
    Each request has: ttft_ms, e2e_ms, handled_by, scheduling_delay_ms, etc.
    Returns the list of request dicts, or empty list if file missing/invalid.
    """
    path = Path(filepath)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("requests", [])
    except (json.JSONDecodeError, KeyError):
        return []


def percentile(values, p):
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_per_request_metrics(results):
    """Compute TTFT and E2E percentiles from per-request JSON data."""
    ttfts = [r["ttft_ms"] for r in results if "ttft_ms" in r]
    e2es = [r["e2e_ms"] for r in results if "e2e_ms" in r]

    if not ttfts:
        return None

    return {
        "ttft_mean": sum(ttfts) / len(ttfts),
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p99": percentile(ttfts, 99),
        "e2e_mean": sum(e2es) / len(e2es) if e2es else 0,
        "e2e_p50": percentile(e2es, 50) if e2es else 0,
        "e2e_p99": percentile(e2es, 99) if e2es else 0,
        "count": len(ttfts),
    }


def compute_instance_distribution(results):
    """Compute per-instance request counts and Jain fairness index.

    Uses "handled_by" field from RequestMetrics (sim/metrics_utils.go:30).
    """
    counts = {}
    for r in results:
        inst = r.get("handled_by", "unknown")
        if not inst:
            inst = "unknown"
        counts[inst] = counts.get(inst, 0) + 1

    if not counts:
        return None, 0.0

    values = list(counts.values())
    n = len(values)
    if n == 0:
        return counts, 0.0

    sum_x = sum(values)
    sum_x2 = sum(v * v for v in values)

    if sum_x2 == 0:
        return counts, 1.0

    # Jain's fairness index: (sum x_i)^2 / (n * sum x_i^2)
    jain = (sum_x ** 2) / (n * sum_x2)
    return counts, jain


def pct_change(base, compare):
    """Compute percent change from base to compare. Returns None if base is 0."""
    if base == 0:
        return None
    return ((compare - base) / base) * 100.0


def format_pct(val):
    """Format a percent change value."""
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


def print_comparison_table(title, fresh_data, stale_data, seeds):
    """Print a comparison table for fresh vs stale across seeds."""
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print()

    header = f"{'Seed':>6}  {'Metric':>10}  {'Fresh':>12}  {'Stale':>12}  {'Change':>10}"
    print(header)
    print("-" * len(header))

    all_ttft_p99_changes = []

    for seed in seeds:
        f = fresh_data.get(seed)
        s = stale_data.get(seed)
        if f is None or s is None:
            print(f"  {seed:>4}  {'(missing data)':>10}")
            continue

        for metric_name, f_key in [
            ("TTFT mean", "ttft_mean"),
            ("TTFT p50", "ttft_p50"),
            ("TTFT p99", "ttft_p99"),
            ("E2E mean", "e2e_mean"),
            ("E2E p50", "e2e_p50"),
            ("E2E p99", "e2e_p99"),
        ]:
            f_val = f.get(f_key, 0)
            s_val = s.get(f_key, 0)
            change = pct_change(f_val, s_val)
            print(f"  {seed:>4}  {metric_name:>10}  {f_val:>10.2f}ms  {s_val:>10.2f}ms  {format_pct(change):>10}")

            if f_key == "ttft_p99" and change is not None:
                all_ttft_p99_changes.append(change)

        print()

    return all_ttft_p99_changes


def print_fairness_table(title, fresh_data, stale_data, seeds):
    """Print Jain fairness comparison across seeds."""
    print(f"\n  {title}")
    print(f"  {'Seed':>6}  {'Fresh FI':>10}  {'Stale FI':>10}  {'Fresh Dist':>30}  {'Stale Dist':>30}")
    print(f"  {'-' * 90}")

    for seed in seeds:
        f_fi = fresh_data.get(seed, (None, 0.0))
        s_fi = stale_data.get(seed, (None, 0.0))
        f_dist_str = str(dict(sorted(f_fi[0].items()))) if f_fi[0] else "N/A"
        s_dist_str = str(dict(sorted(s_fi[0].items()))) if s_fi[0] else "N/A"
        print(f"  {seed:>4}  {f_fi[1]:>10.4f}  {s_fi[1]:>10.4f}  {f_dist_str:>30}  {s_dist_str:>30}")


def print_sweep_table(sweep_data):
    """Print interval sweep results."""
    print(f"\n{'=' * 78}")
    print(f"  Experiment 4: Interval Sweep (kv-utilization:1, seed=42)")
    print(f"{'=' * 78}")
    print()

    header = f"{'Interval':>12}  {'TTFT mean':>12}  {'TTFT p50':>12}  {'TTFT p99':>12}  {'E2E mean':>12}  {'E2E p99':>12}"
    print(header)
    print("-" * len(header))

    baseline_p99 = None
    for interval, data in sorted(sweep_data.items()):
        if data is None:
            print(f"  {interval:>10}us  {'(missing)':>12}")
            continue

        ttft_p99 = data.get("ttft_p99", 0)
        if baseline_p99 is None:
            baseline_p99 = ttft_p99
            suffix = " (baseline)"
        else:
            change = pct_change(baseline_p99, ttft_p99)
            suffix = f" ({format_pct(change)})"

        print(
            f"  {interval:>10}us  "
            f"{data.get('ttft_mean', 0):>10.2f}ms  "
            f"{data.get('ttft_p50', 0):>10.2f}ms  "
            f"{ttft_p99:>10.2f}ms{suffix:>16}  "
            f"{data.get('e2e_mean', 0):>10.2f}ms  "
            f"{data.get('e2e_p99', 0):>10.2f}ms"
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    seeds = [42, 123, 456]

    # ── Experiment 1: kv-utilization (staleness-sensitive) ──────────────────
    exp1_fresh_agg = {}
    exp1_stale_agg = {}
    exp1_fresh_pr = {}
    exp1_stale_pr = {}
    exp1_fresh_fi = {}
    exp1_stale_fi = {}

    for seed in seeds:
        # Aggregate metrics from BLIS output
        f_agg = parse_blis_output(results_dir / f"exp1_kv_fresh_s{seed}.txt")
        s_agg = parse_blis_output(results_dir / f"exp1_kv_stale_s{seed}.txt")
        exp1_fresh_agg[seed] = f_agg
        exp1_stale_agg[seed] = s_agg

        # Per-request metrics from JSON results file
        f_results = load_per_request_results(results_dir / f"exp1_kv_fresh_s{seed}.json")
        s_results = load_per_request_results(results_dir / f"exp1_kv_stale_s{seed}.json")

        f_pr = compute_per_request_metrics(f_results)
        s_pr = compute_per_request_metrics(s_results)

        if f_pr:
            exp1_fresh_pr[seed] = f_pr
        else:
            exp1_fresh_pr[seed] = f_agg  # fallback to aggregate

        if s_pr:
            exp1_stale_pr[seed] = s_pr
        else:
            exp1_stale_pr[seed] = s_agg

        # Instance distribution and fairness
        f_dist, f_fi = compute_instance_distribution(f_results)
        s_dist, s_fi = compute_instance_distribution(s_results)
        exp1_fresh_fi[seed] = (f_dist, f_fi)
        exp1_stale_fi[seed] = (s_dist, s_fi)

    # Print Exp 1 results using aggregate metrics (these have p99)
    exp1_changes = print_comparison_table(
        "Experiment 1: kv-utilization:1 (staleness-sensitive scorer)",
        exp1_fresh_agg, exp1_stale_agg, seeds
    )
    print_fairness_table("Instance Distribution (Jain Fairness Index)", exp1_fresh_fi, exp1_stale_fi, seeds)

    # Also show per-request computed metrics if available
    if any(exp1_fresh_pr.get(s, {}).get("count", 0) > 0 for s in seeds):
        print("\n  Per-request computed metrics (from JSON results file):")
        for seed in seeds:
            f = exp1_fresh_pr.get(seed, {})
            s = exp1_stale_pr.get(seed, {})
            if f.get("count", 0) > 0 and s.get("count", 0) > 0:
                change = pct_change(f.get("ttft_p99", 0), s.get("ttft_p99", 0))
                print(
                    f"    Seed {seed}: TTFT p99 fresh={f.get('ttft_p99', 0):.2f}ms "
                    f"stale={s.get('ttft_p99', 0):.2f}ms change={format_pct(change)}"
                )

    # ── Experiment 2: queue-depth (negative control) ────────────────────────
    exp2_fresh_agg = {}
    exp2_stale_agg = {}
    exp2_fresh_fi = {}
    exp2_stale_fi = {}

    for seed in seeds:
        exp2_fresh_agg[seed] = parse_blis_output(results_dir / f"exp2_qd_fresh_s{seed}.txt")
        exp2_stale_agg[seed] = parse_blis_output(results_dir / f"exp2_qd_stale_s{seed}.txt")

        f_results = load_per_request_results(results_dir / f"exp2_qd_fresh_s{seed}.json")
        s_results = load_per_request_results(results_dir / f"exp2_qd_stale_s{seed}.json")
        f_dist, f_fi = compute_instance_distribution(f_results)
        s_dist, s_fi = compute_instance_distribution(s_results)
        exp2_fresh_fi[seed] = (f_dist, f_fi)
        exp2_stale_fi[seed] = (s_dist, s_fi)

    exp2_changes = print_comparison_table(
        "Experiment 2: queue-depth:1 (NEGATIVE CONTROL — should show ~0% change)",
        exp2_fresh_agg, exp2_stale_agg, seeds
    )
    print_fairness_table("Instance Distribution (Jain Fairness Index)", exp2_fresh_fi, exp2_stale_fi, seeds)

    # ── Experiment 3: composite scorer (mitigation) ─────────────────────────
    exp3_fresh_agg = {}
    exp3_stale_agg = {}
    exp3_fresh_fi = {}
    exp3_stale_fi = {}

    for seed in seeds:
        exp3_fresh_agg[seed] = parse_blis_output(results_dir / f"exp3_combo_fresh_s{seed}.txt")
        exp3_stale_agg[seed] = parse_blis_output(results_dir / f"exp3_combo_stale_s{seed}.txt")

        f_results = load_per_request_results(results_dir / f"exp3_combo_fresh_s{seed}.json")
        s_results = load_per_request_results(results_dir / f"exp3_combo_stale_s{seed}.json")
        f_dist, f_fi = compute_instance_distribution(f_results)
        s_dist, s_fi = compute_instance_distribution(s_results)
        exp3_fresh_fi[seed] = (f_dist, f_fi)
        exp3_stale_fi[seed] = (s_dist, s_fi)

    exp3_changes = print_comparison_table(
        "Experiment 3: kv-utilization:2,queue-depth:2 (mitigation via composite)",
        exp3_fresh_agg, exp3_stale_agg, seeds
    )
    print_fairness_table("Instance Distribution (Jain Fairness Index)", exp3_fresh_fi, exp3_stale_fi, seeds)

    # ── Experiment 4: interval sweep ────────────────────────────────────────
    sweep_intervals = [0, 1000, 5000, 10000, 50000, 100000, 500000]
    sweep_data = {}
    for interval in sweep_intervals:
        agg = parse_blis_output(results_dir / f"exp4_sweep_i{interval}.txt")
        if agg.get("timed_out"):
            sweep_data[interval] = None
        else:
            sweep_data[interval] = agg

    print_sweep_table(sweep_data)

    # ── Conservation check (INV-1) ──────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"  Conservation Check (INV-1)")
    print(f"{'=' * 78}")
    print()

    conservation_ok = True
    for seed in seeds:
        for label, data in [
            (f"exp1_kv_fresh_s{seed}", exp1_fresh_agg.get(seed, {})),
            (f"exp1_kv_stale_s{seed}", exp1_stale_agg.get(seed, {})),
            (f"exp2_qd_fresh_s{seed}", exp2_fresh_agg.get(seed, {})),
            (f"exp2_qd_stale_s{seed}", exp2_stale_agg.get(seed, {})),
        ]:
            if data.get("timed_out"):
                print(f"  {label}: SKIPPED (timed out)")
                conservation_ok = False
                continue

            injected = data.get("injected", 0)
            completed = data.get("completed", 0)
            queued = data.get("still_queued", 0)
            running = data.get("still_running", 0)
            # Note: parse_blis_output does not extract dropped_unservable.
            # At 85% saturation with default KV blocks, drops should be 0.
            accounted = completed + queued + running
            if injected > 0 and injected != accounted:
                print(f"  {label}: VIOLATION injected={injected} != completed({completed}) + queued({queued}) + running({running}) = {accounted}")
                conservation_ok = False
            else:
                print(f"  {label}: OK (injected={injected}, completed={completed}, queued={queued}, running={running})")

    # Also check Experiment 3 (composite scorer)
    for seed in seeds:
        for label, prefix in [("exp3_combo_fresh", f"exp3_combo_fresh_s{seed}"),
                              ("exp3_combo_stale", f"exp3_combo_stale_s{seed}")]:
            data = parse_blis_output(results_dir / f"{prefix}.txt")
            if data.get("timed_out"):
                print(f"  {prefix}: SKIPPED (timed out)")
                conservation_ok = False
                continue
            injected = data.get("injected", 0)
            completed = data.get("completed", 0)
            queued = data.get("still_queued", 0)
            running = data.get("still_running", 0)
            accounted = completed + queued + running
            if injected > 0 and injected != accounted:
                print(f"  {prefix}: VIOLATION injected={injected} != {accounted}")
                conservation_ok = False
            else:
                print(f"  {prefix}: OK (injected={injected}, completed={completed}, queued={queued}, running={running})")

    # Also check Experiment 4 (sweep)
    for interval in [0, 1000, 5000, 10000, 50000, 100000, 500000]:
        prefix = f"exp4_sweep_i{interval}"
        data = parse_blis_output(results_dir / f"{prefix}.txt")
        if data.get("timed_out"):
            print(f"  {prefix}: SKIPPED (timed out)")
            continue
        injected = data.get("injected", 0)
        completed = data.get("completed", 0)
        queued = data.get("still_queued", 0)
        running = data.get("still_running", 0)
        accounted = completed + queued + running
        if injected > 0 and injected != accounted:
            print(f"  {prefix}: VIOLATION injected={injected} != {accounted}")
            conservation_ok = False
        else:
            print(f"  {prefix}: OK (injected={injected}, completed={completed}, queued={queued}, running={running})")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"  VERDICT")
    print(f"{'=' * 78}")
    print()

    # Check negative control first
    qd_max_change = 0.0
    if exp2_changes:
        qd_max_change = max(abs(c) for c in exp2_changes)
    print(f"  Negative control (queue-depth): max |TTFT p99 change| = {qd_max_change:.1f}%")
    if qd_max_change > 5.0:
        print(f"  WARNING: Negative control shows {qd_max_change:.1f}% change — confound detected!")
        print(f"  queue-depth should be unaffected by snapshot interval (always Immediate).")
    else:
        print(f"  Negative control PASSED (< 5% change, as expected)")

    print()

    # Primary verdict on kv-utilization
    if not exp1_changes:
        print("  INCONCLUSIVE: No TTFT p99 data available for kv-utilization experiment")
    else:
        min_change = min(exp1_changes)
        max_change = max(exp1_changes)
        mean_change = sum(exp1_changes) / len(exp1_changes)

        print(f"  kv-utilization TTFT p99 changes across seeds: {[f'{c:.1f}%' for c in exp1_changes]}")
        print(f"  Min: {min_change:.1f}%, Max: {max_change:.1f}%, Mean: {mean_change:.1f}%")
        print()

        # Hypothesis: >= 20% degradation
        # Refuted if: < 10% across ALL seeds
        all_above_20 = all(c >= 20.0 for c in exp1_changes)
        all_below_10 = all(c < 10.0 for c in exp1_changes)

        if all_above_20:
            print("  CONFIRMED: TTFT p99 degradation >= 20% across all seeds")
            print("  Stale KV-utilization snapshots cause significant tail latency degradation.")
        elif all_below_10:
            print("  REFUTED: TTFT p99 difference < 10% across all seeds")
            print("  Stale KV-utilization snapshots do NOT significantly impact tail latency at 85% saturation.")
        elif mean_change >= 20.0:
            print("  PARTIALLY CONFIRMED: Mean TTFT p99 degradation >= 20% but not consistent across all seeds")
        elif min_change >= 10.0:
            print("  INCONCLUSIVE: TTFT p99 changes between 10% and 20% (dead zone)")
            print("  Effect exists but below the 20% confirmation threshold.")
        else:
            print("  MIXED: Some seeds below 10%, some above")
            print("  Results are inconsistent across seeds — may need more investigation.")

    # Composite mitigation analysis
    print()
    if exp3_changes:
        exp3_max = max(abs(c) for c in exp3_changes)
        exp3_mean = sum(exp3_changes) / len(exp3_changes)
        print(f"  Composite scorer (kv-util + queue-depth) TTFT p99 changes: {[f'{c:.1f}%' for c in exp3_changes]}")
        print(f"  Mean change: {exp3_mean:.1f}%")
        if exp1_changes:
            exp1_mean = sum(exp1_changes) / len(exp1_changes)
            if exp1_mean > 10.0 and exp3_mean < exp1_mean * 0.5:
                print(f"  Mitigation: queue-depth compensates for stale KV signals ({exp3_mean:.1f}% vs {exp1_mean:.1f}%)")
            elif exp1_mean <= 10.0:
                print(f"  No mitigation needed — base effect is small ({exp1_mean:.1f}%)")
            else:
                print(f"  Partial/no mitigation: composite ({exp3_mean:.1f}%) vs kv-only ({exp1_mean:.1f}%)")

    if not conservation_ok:
        print()
        print("  WARNING: Conservation violations detected — results may be unreliable")

    print()


if __name__ == "__main__":
    main()
