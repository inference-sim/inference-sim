#!/usr/bin/env python3
"""H28: Chunked Prefill ITL Impact — Analysis

Compares Config A (threshold=0, chunking disabled) vs Config B (threshold=512,
chunking enabled) across 3 seeds. Separates long-input (2048 tokens) from
short-input (128 tokens) requests and measures:
  1. ITL improvement for short-input (concurrent decode) requests
  2. TTFT change for long-input requests

Hypothesis thresholds:
  - CONFIRMED if: mean ITL improvement for short-input >= 15% AND
                   TTFT increase for long-input >= 20%
  - REFUTED if: mean ITL improvement < 10% OR TTFT increase < 10%

Usage: python3 analyze.py <results_dir>
"""

import json
import sys
from pathlib import Path
from statistics import mean

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout

SEEDS = [42, 123, 456]

# Threshold to classify long vs short input requests.
# Long-input client uses 2048 tokens; short-input uses 128 tokens.
# Mid-point at 1000 cleanly separates the two groups.
LONG_INPUT_THRESHOLD = 1000


def load_per_request_data(filepath):
    """Load per-request data from results JSON file.

    Returns list of request dicts, or None if file is missing/invalid.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"WARNING: results file missing: {filepath}", file=sys.stderr)
        return None
    try:
        data = json.loads(path.read_text())
        requests = data.get("requests", [])
        if not requests:
            print(f"WARNING: no requests in {filepath}", file=sys.stderr)
            return None
        return requests
    except (json.JSONDecodeError, KeyError) as e:
        print(f"WARNING: failed to parse {filepath}: {e}", file=sys.stderr)
        return None


def split_by_input_size(requests):
    """Split requests into long-input and short-input groups.

    Uses num_prefill_tokens to classify. Long >= LONG_INPUT_THRESHOLD.
    """
    long_reqs = [r for r in requests if r.get("num_prefill_tokens", 0) >= LONG_INPUT_THRESHOLD]
    short_reqs = [r for r in requests if r.get("num_prefill_tokens", 0) < LONG_INPUT_THRESHOLD]
    return long_reqs, short_reqs


def compute_metrics(requests):
    """Compute ITL, TTFT, E2E statistics for a group of requests.

    Returns dict with mean/p50/p99 for ITL, TTFT, E2E, plus count.
    All values are in ms (matching per-request JSON fields).
    """
    if not requests:
        return {
            "count": 0,
            "itl_mean": 0, "itl_p50": 0, "itl_p99": 0,
            "ttft_mean": 0, "ttft_p50": 0, "ttft_p99": 0,
            "e2e_mean": 0, "e2e_p50": 0, "e2e_p99": 0,
        }

    itls = [r["itl_ms"] for r in requests if r.get("itl_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in requests if r.get("ttft_ms", 0) > 0]
    e2es = [r["e2e_ms"] for r in requests if r.get("e2e_ms", 0) > 0]

    def percentile(data, p):
        if not data:
            return 0.0
        data_sorted = sorted(data)
        idx = p / 100.0 * (len(data_sorted) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(data_sorted) - 1)
        frac = idx - lower
        return data_sorted[lower] * (1 - frac) + data_sorted[upper] * frac

    return {
        "count": len(requests),
        "itl_mean": mean(itls) if itls else 0,
        "itl_p50": percentile(itls, 50),
        "itl_p99": percentile(itls, 99),
        "ttft_mean": mean(ttfts) if ttfts else 0,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p99": percentile(ttfts, 99),
        "e2e_mean": mean(e2es) if e2es else 0,
        "e2e_p50": percentile(e2es, 50),
        "e2e_p99": percentile(e2es, 99),
    }


def pct_change(baseline, treatment):
    """Compute percentage change: (treatment - baseline) / baseline * 100.

    Returns 0.0 if baseline is 0 (avoid division by zero).
    """
    if baseline == 0:
        return 0.0
    return (treatment - baseline) / baseline * 100.0


def print_comparison_table(label, config_a_metrics, config_b_metrics):
    """Print a formatted comparison table for one request group."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Config A':>12} {'Config B':>12} {'Change':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

    metrics_to_show = [
        ("Count", "count", ""),
        ("ITL Mean (ms)", "itl_mean", "ms"),
        ("ITL P50 (ms)", "itl_p50", "ms"),
        ("ITL P99 (ms)", "itl_p99", "ms"),
        ("TTFT Mean (ms)", "ttft_mean", "ms"),
        ("TTFT P50 (ms)", "ttft_p50", "ms"),
        ("TTFT P99 (ms)", "ttft_p99", "ms"),
        ("E2E Mean (ms)", "e2e_mean", "ms"),
        ("E2E P50 (ms)", "e2e_p50", "ms"),
        ("E2E P99 (ms)", "e2e_p99", "ms"),
    ]

    for display_name, key, unit in metrics_to_show:
        a_val = config_a_metrics.get(key, 0)
        b_val = config_b_metrics.get(key, 0)
        if key == "count":
            print(f"  {display_name:<20} {a_val:>12d} {b_val:>12d} {'':>10}")
        else:
            change = pct_change(a_val, b_val)
            sign = "+" if change >= 0 else ""
            print(f"  {display_name:<20} {a_val:>12.2f} {b_val:>12.2f} {sign}{change:>8.1f}%")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    # =========================================================================
    # INV-1: Conservation check (from stdout output)
    # =========================================================================
    print("=" * 70)
    print("  H28: Chunked Prefill ITL Impact — Analysis")
    print("=" * 70)

    print("\n--- INV-1: Conservation Check ---")
    all_conserved = True
    for config_label, prefix in [("A (threshold=0)", "a"), ("B (threshold=512)", "b")]:
        for seed in SEEDS:
            stdout_file = results_dir / f"{prefix}_s{seed}_stdout.txt"
            if check_for_timeout(str(stdout_file)):
                print(f"  Config {config_label} seed={seed}: TIMEOUT/ERROR")
                all_conserved = False
                continue
            metrics = parse_blis_output(str(stdout_file))
            if metrics["timed_out"]:
                print(f"  Config {config_label} seed={seed}: PARSE ERROR")
                all_conserved = False
                continue
            injected = metrics["injected"]
            completed = metrics["completed"]
            queued = metrics["still_queued"]
            running = metrics["still_running"]
            # Note: parse_blis_output does NOT extract dropped_unservable.
            # For this experiment with abundant KV blocks, dropped should be 0.
            conserved = injected == completed + queued + running
            status = "PASS" if conserved else "FAIL"
            if not conserved:
                all_conserved = False
            print(f"  Config {config_label} seed={seed}: {status} "
                  f"(injected={injected}, completed={completed}, "
                  f"queued={queued}, running={running})")

    print(f"\n  Conservation: {'ALL PASS' if all_conserved else 'SOME FAILURES'}")

    # =========================================================================
    # Per-request analysis: separate long vs short input requests
    # =========================================================================
    # Accumulate per-seed metrics for averaging
    seed_results = {
        "a_long": [], "a_short": [],
        "b_long": [], "b_short": [],
    }

    for seed in SEEDS:
        for prefix, config_label in [("a", "A"), ("b", "B")]:
            results_file = results_dir / f"{prefix}_s{seed}_results.json"
            requests = load_per_request_data(str(results_file))
            if requests is None:
                print(f"\n  WARNING: No per-request data for Config {config_label} seed={seed}",
                      file=sys.stderr)
                continue

            long_reqs, short_reqs = split_by_input_size(requests)
            long_metrics = compute_metrics(long_reqs)
            short_metrics = compute_metrics(short_reqs)

            seed_results[f"{prefix}_long"].append(long_metrics)
            seed_results[f"{prefix}_short"].append(short_metrics)

    # =========================================================================
    # Aggregate across seeds (mean of per-seed means)
    # =========================================================================
    def aggregate_metrics(metrics_list):
        """Average metrics across seeds."""
        if not metrics_list:
            return compute_metrics([])  # return zeros
        result = {}
        for key in metrics_list[0]:
            vals = [m[key] for m in metrics_list]
            result[key] = int(mean(vals)) if key == "count" else mean(vals)
        return result

    a_long_avg = aggregate_metrics(seed_results["a_long"])
    a_short_avg = aggregate_metrics(seed_results["a_short"])
    b_long_avg = aggregate_metrics(seed_results["b_long"])
    b_short_avg = aggregate_metrics(seed_results["b_short"])

    # =========================================================================
    # Print per-seed detail
    # =========================================================================
    print("\n--- Per-Seed Detail ---")
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed}:")
        for prefix, config_label in [("a", "A (threshold=0)"), ("b", "B (threshold=512)")]:
            key_long = f"{prefix}_long"
            key_short = f"{prefix}_short"
            if seed_idx < len(seed_results[key_long]):
                lm = seed_results[key_long][seed_idx]
                sm = seed_results[key_short][seed_idx]
                print(f"    Config {config_label}:")
                print(f"      Long-input  (n={lm['count']:>3d}): ITL={lm['itl_mean']:>8.2f}ms  "
                      f"TTFT={lm['ttft_mean']:>8.2f}ms  E2E={lm['e2e_mean']:>8.2f}ms")
                print(f"      Short-input (n={sm['count']:>3d}): ITL={sm['itl_mean']:>8.2f}ms  "
                      f"TTFT={sm['ttft_mean']:>8.2f}ms  E2E={sm['e2e_mean']:>8.2f}ms")
            else:
                print(f"    Config {config_label}: NO DATA")

    # =========================================================================
    # Print comparison tables (averaged across seeds)
    # =========================================================================
    print_comparison_table(
        "LONG-INPUT REQUESTS (2048 tokens) — Averaged over 3 seeds",
        a_long_avg, b_long_avg
    )
    print_comparison_table(
        "SHORT-INPUT REQUESTS (128 tokens) — Averaged over 3 seeds",
        a_short_avg, b_short_avg
    )

    # =========================================================================
    # Verdict
    # =========================================================================
    # Primary metrics:
    #   1. ITL improvement for short-input requests: (A - B) / A * 100
    #      (positive = B has lower ITL = improvement)
    #   2. TTFT increase for long-input requests: (B - A) / A * 100
    #      (positive = B has higher TTFT = cost)

    itl_improvement = pct_change(a_short_avg["itl_mean"], b_short_avg["itl_mean"])
    # ITL improvement = negative change means B is lower (better)
    itl_improvement_pct = -itl_improvement  # flip sign: positive = improvement

    ttft_increase = pct_change(a_long_avg["ttft_mean"], b_long_avg["ttft_mean"])
    # TTFT increase = positive change means B is higher (cost)
    ttft_increase_pct = ttft_increase

    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    print(f"\n  Primary metrics:")
    print(f"    Short-input ITL mean: Config A = {a_short_avg['itl_mean']:.2f}ms, "
          f"Config B = {b_short_avg['itl_mean']:.2f}ms")
    print(f"    ITL improvement (B vs A): {itl_improvement_pct:+.1f}%")
    print(f"    (positive = B has lower ITL = chunking helps decode requests)")
    print(f"")
    print(f"    Long-input TTFT mean: Config A = {a_long_avg['ttft_mean']:.2f}ms, "
          f"Config B = {b_long_avg['ttft_mean']:.2f}ms")
    print(f"    TTFT increase (B vs A): {ttft_increase_pct:+.1f}%")
    print(f"    (positive = B has higher TTFT = chunking costs prefill requests)")

    # Determine verdict
    if itl_improvement_pct >= 15.0 and ttft_increase_pct >= 20.0:
        verdict = "CONFIRMED"
        explanation = (
            f"ITL improvement ({itl_improvement_pct:.1f}%) >= 15% threshold AND "
            f"TTFT increase ({ttft_increase_pct:.1f}%) >= 20% threshold"
        )
    elif itl_improvement_pct < 10.0 or ttft_increase_pct < 10.0:
        if itl_improvement_pct < 10.0 and ttft_increase_pct < 10.0:
            verdict = "REFUTED"
            explanation = (
                f"ITL improvement ({itl_improvement_pct:.1f}%) < 10% threshold AND "
                f"TTFT increase ({ttft_increase_pct:.1f}%) < 10% threshold"
            )
        elif itl_improvement_pct < 10.0:
            verdict = "REFUTED"
            explanation = (
                f"ITL improvement ({itl_improvement_pct:.1f}%) < 10% threshold "
                f"(TTFT increase was {ttft_increase_pct:.1f}%)"
            )
        else:
            verdict = "REFUTED"
            explanation = (
                f"TTFT increase ({ttft_increase_pct:.1f}%) < 10% threshold "
                f"(ITL improvement was {itl_improvement_pct:.1f}%)"
            )
    else:
        verdict = "INCONCLUSIVE"
        explanation = (
            f"ITL improvement ({itl_improvement_pct:.1f}%) is between 10-15% or "
            f"TTFT increase ({ttft_increase_pct:.1f}%) is between 10-20% "
            f"(gray zone between confirmation and refutation thresholds)"
        )

    print(f"\n  Verdict: {verdict}")
    print(f"  Reason: {explanation}")

    # =========================================================================
    # Per-seed consistency check
    # =========================================================================
    print(f"\n--- Per-Seed Consistency ---")
    for seed_idx, seed in enumerate(SEEDS):
        if (seed_idx < len(seed_results["a_short"]) and
                seed_idx < len(seed_results["b_short"]) and
                seed_idx < len(seed_results["a_long"]) and
                seed_idx < len(seed_results["b_long"])):
            a_itl = seed_results["a_short"][seed_idx]["itl_mean"]
            b_itl = seed_results["b_short"][seed_idx]["itl_mean"]
            seed_itl_imp = -pct_change(a_itl, b_itl)

            a_ttft = seed_results["a_long"][seed_idx]["ttft_mean"]
            b_ttft = seed_results["b_long"][seed_idx]["ttft_mean"]
            seed_ttft_inc = pct_change(a_ttft, b_ttft)

            print(f"  Seed {seed}: ITL improvement = {seed_itl_imp:+.1f}%, "
                  f"TTFT increase = {seed_ttft_inc:+.1f}%")
        else:
            print(f"  Seed {seed}: INCOMPLETE DATA")


if __name__ == "__main__":
    main()
