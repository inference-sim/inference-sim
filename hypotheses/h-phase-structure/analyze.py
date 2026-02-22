#!/usr/bin/env python3
"""Analysis script for H-Phase-Structure: Latency Model Phase Linearity.

Parses per-request JSON from BLIS --results-path output.
Computes linear regression (R²) for:
  A. TTFT vs input tokens (output held constant)
  B. (E2E - TTFT) vs output tokens (input held constant)

BLIS per-request JSON fields (see sim/metrics_utils.go:19-31):
  - num_prefill_tokens: int (input tokens)
  - num_decode_tokens: int (output tokens)
  - ttft_ms: float (milliseconds, converted from ticks by /1e3)
  - e2e_ms: float (milliseconds, converted from ticks by /1e3)

NOTE: scheduling_delay_ms is in TICKS (not ms) despite the field name.
      This script does not use scheduling_delay_ms.

Analytical predictions from trained coefficients (defaults.yaml):
  Alpha coefficients: [1601.35, 3.51, 1805.54]
    alpha0 = 1601.35 us (base queue delay)
    alpha1 = 3.51 us/token (per-input-token queue delay)
    alpha2 = 1805.54 us (output processing time)

  Beta coefficients: [6910.42, 17.67, 2.84]
    beta0 = 6910.42 us (base step time)
    beta1 = 17.67 us/token (per-cache-miss-token step time)
    beta2 = 2.84 us/token (per-decode-token step time)

  Expected TTFT = (alpha0 + alpha1*input) + (beta0 + beta1*input) + alpha2
                = (alpha0 + beta0 + alpha2) + (alpha1 + beta1)*input
                ≈ 10317 us + 21.18 us/token * input
                ≈ 10.317 ms + 0.02118 ms/token * input

  Expected decode_time = output_tokens * (beta0 + beta2 + alpha2)
                       = output_tokens * 8718.80 us
                       ≈ output_tokens * 8.719 ms
  Each ITL entry = currStepAdvance + getOutputTokenProcessingTime()
                 = (beta0 + beta2*1) + alpha2 = 8718.80 us  (simulator.go:627,659)
"""

import argparse
import json
import sys
from pathlib import Path


def linear_regression(xs, ys):
    """Compute least-squares linear regression.

    Returns (slope, intercept, r_squared).
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0, sum_y / n, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R² = 1 - SS_res / SS_tot
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))

    if ss_tot < 1e-15:
        r_squared = 1.0  # Perfect fit (all y values identical)
    else:
        r_squared = 1.0 - ss_res / ss_tot

    return slope, intercept, r_squared


def parse_requests(filepath):
    """Parse per-request metrics from a BLIS --results-path JSON file.

    Returns list of dicts with keys: input_tokens, output_tokens, ttft_ms, e2e_ms, decode_ms.
    Filters to completed requests only (e2e_ms > 0).
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARNING: could not parse {filepath}: {e}", file=sys.stderr)
        return []

    results = []
    for r in data.get("requests", []):
        if r.get("e2e_ms", 0) <= 0:
            continue
        ttft = r["ttft_ms"]
        e2e = r["e2e_ms"]
        # scheduling_delay_ms field is in TICKS (μs), not ms — divide by 1000
        sched_delay_ms = r.get("scheduling_delay_ms", 0) / 1000.0
        results.append({
            "input_tokens": r["num_prefill_tokens"],
            "output_tokens": r["num_decode_tokens"],
            "ttft_ms": ttft,
            "e2e_ms": e2e,
            "decode_ms": e2e - ttft,
            "sched_delay_ms": sched_delay_ms,
            "ttft_adjusted_ms": ttft - sched_delay_ms,  # Pure latency model (no queueing)
        })
    return results


def analyze_experiment_a(results_dir, input_levels, fixed_output, seeds):
    """Experiment A: TTFT vs Input Tokens (output held constant)."""
    print("=" * 78)
    print("  Experiment A: TTFT vs Input Tokens")
    print(f"  Fixed output = {fixed_output} tokens")
    print("=" * 78)
    print()

    # Collect per-seed data: both raw TTFT and adjusted (minus scheduling delay)
    per_seed_raw = {}
    per_seed_adj = {}
    for seed in seeds:
        xs = []
        ys_raw = []
        ys_adj = []
        for input_tok in input_levels:
            filepath = Path(results_dir) / f"expA_in{input_tok}_s{seed}.json"
            reqs = parse_requests(filepath)
            if not reqs:
                print(f"  WARNING: no data for input={input_tok} seed={seed}", file=sys.stderr)
                continue

            mean_ttft = sum(r["ttft_ms"] for r in reqs) / len(reqs)
            mean_adj = sum(r["ttft_adjusted_ms"] for r in reqs) / len(reqs)
            xs.append(input_tok)
            ys_raw.append(mean_ttft)
            ys_adj.append(mean_adj)

        if len(xs) >= 2:
            sl_r, ic_r, r2_r = linear_regression(xs, ys_raw)
            per_seed_raw[seed] = {
                "xs": xs, "ys": ys_raw,
                "slope": sl_r, "intercept": ic_r, "r_squared": r2_r,
            }
            sl_a, ic_a, r2_a = linear_regression(xs, ys_adj)
            per_seed_adj[seed] = {
                "xs": xs, "ys": ys_adj,
                "slope": sl_a, "intercept": ic_a, "r_squared": r2_a,
            }

    # ── Raw TTFT table ──
    print("  Per-level mean TTFT (ms) — raw (includes queueing delay):")
    print(f"  {'Input':>8s}", end="")
    for seed in seeds:
        print(f"  {'seed=' + str(seed):>12s}", end="")
    print(f"  {'Mean':>10s}")
    print("  " + "-" * (8 + 12 * len(seeds) + 10 + 2 * (len(seeds) + 1)))

    for i, input_tok in enumerate(input_levels):
        print(f"  {input_tok:8d}", end="")
        vals = []
        for seed in seeds:
            if seed in per_seed_raw and i < len(per_seed_raw[seed]["ys"]):
                v = per_seed_raw[seed]["ys"][i]
                print(f"  {v:12.3f}", end="")
                vals.append(v)
            else:
                print(f"  {'N/A':>12s}", end="")
        if vals:
            print(f"  {sum(vals)/len(vals):10.3f}", end="")
        print()

    print()

    # ── Raw regression ──
    print("  Raw Linear Regression: TTFT = slope * input_tokens + intercept")
    print(f"  {'Seed':>8s}  {'Slope (ms/tok)':>14s}  {'Intercept (ms)':>15s}  {'R²':>10s}  {'Pass':>6s}")
    print("  " + "-" * 60)

    raw_pass = True
    for seed in seeds:
        if seed not in per_seed_raw:
            print(f"  {seed:8d}  {'N/A':>14s}  {'N/A':>15s}  {'N/A':>10s}  {'N/A':>6s}")
            raw_pass = False
            continue
        r = per_seed_raw[seed]
        passed = r["r_squared"] >= 0.95
        if not passed:
            raw_pass = False
        print(f"  {seed:8d}  {r['slope']:14.6f}  {r['intercept']:15.3f}  {r['r_squared']:10.6f}  {'YES' if passed else 'NO':>6s}")

    print()

    # ── Adjusted TTFT table (TTFT - scheduling_delay) ──
    print("  Per-level mean adjusted TTFT (ms) — scheduling delay subtracted:")
    print("  (Isolates latency model: step_time + alpha2, no queueing)")
    print(f"  {'Input':>8s}", end="")
    for seed in seeds:
        print(f"  {'seed=' + str(seed):>12s}", end="")
    print(f"  {'Mean':>10s}")
    print("  " + "-" * (8 + 12 * len(seeds) + 10 + 2 * (len(seeds) + 1)))

    for i, input_tok in enumerate(input_levels):
        print(f"  {input_tok:8d}", end="")
        vals = []
        for seed in seeds:
            if seed in per_seed_adj and i < len(per_seed_adj[seed]["ys"]):
                v = per_seed_adj[seed]["ys"][i]
                print(f"  {v:12.3f}", end="")
                vals.append(v)
            else:
                print(f"  {'N/A':>12s}", end="")
        if vals:
            print(f"  {sum(vals)/len(vals):10.3f}", end="")
        print()

    print()

    # ── Adjusted regression ──
    print("  Adjusted Linear Regression: (TTFT - sched_delay) = slope * input + intercept")
    print(f"  {'Seed':>8s}  {'Slope (ms/tok)':>14s}  {'Intercept (ms)':>15s}  {'R²':>10s}  {'Pass':>6s}")
    print("  " + "-" * 60)

    adj_pass = True
    for seed in seeds:
        if seed not in per_seed_adj:
            print(f"  {seed:8d}  {'N/A':>14s}  {'N/A':>15s}  {'N/A':>10s}  {'N/A':>6s}")
            adj_pass = False
            continue
        r = per_seed_adj[seed]
        passed = r["r_squared"] >= 0.95
        if not passed:
            adj_pass = False
        print(f"  {seed:8d}  {r['slope']:14.6f}  {r['intercept']:15.3f}  {r['r_squared']:10.6f}  {'YES' if passed else 'NO':>6s}")

    print()

    # Analytical predictions
    alpha0, alpha1, alpha2 = 1601.35, 3.51, 1805.54
    beta0, beta1 = 6910.42, 17.67

    # Raw TTFT: slope = alpha1 + beta1, intercept = alpha0 + beta0 + alpha2
    raw_slope_us = alpha1 + beta1
    raw_intercept_us = alpha0 + beta0 + alpha2
    # Adjusted TTFT (minus scheduling delay): slope = beta1, intercept = beta0 + alpha2
    adj_slope_us = beta1
    adj_intercept_us = beta0 + alpha2

    print(f"  Analytical predictions:")
    print(f"    Raw TTFT:      slope = alpha1+beta1 = {raw_slope_us:.2f} μs/tok = {raw_slope_us/1000:.6f} ms/tok")
    print(f"                   intercept = alpha0+beta0+alpha2 = {raw_intercept_us:.2f} μs = {raw_intercept_us/1000:.3f} ms")
    print(f"    Adjusted TTFT: slope = beta1 = {adj_slope_us:.2f} μs/tok = {adj_slope_us/1000:.6f} ms/tok")
    print(f"                   intercept = beta0+alpha2 = {adj_intercept_us:.2f} μs = {adj_intercept_us/1000:.3f} ms")
    print()

    if per_seed_raw:
        mean_raw_slope = sum(r["slope"] for r in per_seed_raw.values()) / len(per_seed_raw)
        raw_err = abs(mean_raw_slope - raw_slope_us / 1000) / (raw_slope_us / 1000) * 100
        print(f"    Raw measured slope:      {mean_raw_slope:.6f} ms/tok (error: {raw_err:.1f}%)")
    if per_seed_adj:
        mean_adj_slope = sum(r["slope"] for r in per_seed_adj.values()) / len(per_seed_adj)
        adj_err = abs(mean_adj_slope - adj_slope_us / 1000) / (adj_slope_us / 1000) * 100
        print(f"    Adjusted measured slope: {mean_adj_slope:.6f} ms/tok (error: {adj_err:.1f}%)")
    print()

    # Verdicts
    print(f"  Raw TTFT:      {'PASS' if raw_pass else 'FAIL'} (R² ≥ 0.95 all seeds: {raw_pass})")
    print(f"  Adjusted TTFT: {'PASS' if adj_pass else 'FAIL'} (R² ≥ 0.95 all seeds: {adj_pass})")
    if not raw_pass and adj_pass:
        print("  → Queueing noise contaminates raw TTFT but the latency model is perfectly linear")
    print()

    return {"raw": per_seed_raw, "adjusted": per_seed_adj, "raw_pass": raw_pass, "adj_pass": adj_pass}


def analyze_experiment_b(results_dir, output_levels, fixed_input, seeds):
    """Experiment B: Decode Time (E2E - TTFT) vs Output Tokens (input held constant)."""
    print("=" * 78)
    print("  Experiment B: Decode Time (E2E − TTFT) vs Output Tokens")
    print(f"  Fixed input = {fixed_input} tokens")
    print("=" * 78)
    print()

    # Collect per-seed data
    per_seed_results = {}
    for seed in seeds:
        xs = []  # output token levels
        ys = []  # mean decode time at each level
        for output_tok in output_levels:
            filepath = Path(results_dir) / f"expB_out{output_tok}_s{seed}.json"
            reqs = parse_requests(filepath)
            if not reqs:
                print(f"  WARNING: no data for output={output_tok} seed={seed}", file=sys.stderr)
                continue

            mean_decode = sum(r["decode_ms"] for r in reqs) / len(reqs)
            xs.append(output_tok)
            ys.append(mean_decode)

        if len(xs) >= 2:
            slope, intercept, r2 = linear_regression(xs, ys)
            per_seed_results[seed] = {
                "xs": xs, "ys": ys,
                "slope": slope, "intercept": intercept, "r_squared": r2,
            }

    # Print per-level table
    print("  Per-level mean decode time (E2E − TTFT) (ms):")
    print(f"  {'Output':>8s}", end="")
    for seed in seeds:
        print(f"  {'seed=' + str(seed):>12s}", end="")
    print(f"  {'Mean':>10s}")
    print("  " + "-" * (8 + 12 * len(seeds) + 10 + 2 * (len(seeds) + 1)))

    for i, output_tok in enumerate(output_levels):
        print(f"  {output_tok:8d}", end="")
        vals = []
        for seed in seeds:
            if seed in per_seed_results and i < len(per_seed_results[seed]["ys"]):
                v = per_seed_results[seed]["ys"][i]
                print(f"  {v:12.3f}", end="")
                vals.append(v)
            else:
                print(f"  {'N/A':>12s}", end="")
        if vals:
            print(f"  {sum(vals)/len(vals):10.3f}", end="")
        print()

    print()

    # Print regression results per seed
    print("  Linear Regression: decode_time = slope * output_tokens + intercept")
    print(f"  {'Seed':>8s}  {'Slope (ms/tok)':>14s}  {'Intercept (ms)':>15s}  {'R²':>10s}  {'Pass':>6s}")
    print("  " + "-" * 60)

    all_pass = True
    for seed in seeds:
        if seed not in per_seed_results:
            print(f"  {seed:8d}  {'N/A':>14s}  {'N/A':>15s}  {'N/A':>10s}  {'N/A':>6s}")
            all_pass = False
            continue
        r = per_seed_results[seed]
        passed = r["r_squared"] >= 0.95
        if not passed:
            all_pass = False
        print(f"  {seed:8d}  {r['slope']:14.6f}  {r['intercept']:15.3f}  {r['r_squared']:10.6f}  {'YES' if passed else 'NO':>6s}")

    print()

    # Analytical prediction
    # Each decode token's ITL = currStepAdvance + getOutputTokenProcessingTime()
    #   currStepAdvance = beta0 + beta2*1 (batch=1, one decode token)
    #   getOutputTokenProcessingTime() = alpha2
    # So per-token decode cost = beta0 + beta2 + alpha2
    # See simulator.go:627 and simulator.go:659 — both add currStepAdvance + alpha2
    alpha2 = 1805.54
    beta0, beta2 = 6910.42, 2.84
    expected_step_us = beta0 + beta2 + alpha2  # 8718.80 us per decode token
    expected_slope_ms = expected_step_us / 1000.0  # ms per output token

    print(f"  Analytical prediction (from alpha/beta coefficients):")
    print(f"    Per-token decode cost: {expected_slope_ms:.6f} ms/token ({expected_step_us:.2f} μs/token)")
    print(f"    (beta0 + beta2*1 + alpha2 = {beta0} + {beta2} + {alpha2} = {expected_step_us:.2f} μs)")
    print(f"    Slope should ≈ {expected_slope_ms:.3f} ms/token")
    print()

    if per_seed_results:
        mean_slope = sum(r["slope"] for r in per_seed_results.values()) / len(per_seed_results)
        slope_error = abs(mean_slope - expected_slope_ms) / expected_slope_ms * 100
        mean_intercept = sum(r["intercept"] for r in per_seed_results.values()) / len(per_seed_results)
        print(f"    Measured mean slope: {mean_slope:.6f} ms/token")
        print(f"    Slope error vs analytical: {slope_error:.1f}%")
        print(f"    Measured mean intercept: {mean_intercept:.3f} ms")
        print(f"    (Non-zero intercept indicates offset from first-token / prefill step)")
    print()

    # Verdict
    if all_pass:
        print("  ✓ PASS: R² ≥ 0.95 for all seeds — decode time is linear in output tokens")
    else:
        print("  ✗ FAIL: R² < 0.95 for at least one seed — decode time is NOT linear in output tokens")
    print()

    return per_seed_results


def main():
    parser = argparse.ArgumentParser(description="H-Phase-Structure analysis")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--input-levels", required=True, help="Space-separated input token levels")
    parser.add_argument("--output-levels", required=True, help="Space-separated output token levels")
    parser.add_argument("--fixed-output", type=int, required=True)
    parser.add_argument("--fixed-input", type=int, required=True)
    parser.add_argument("--seeds", required=True, help="Space-separated seed values")
    args = parser.parse_args()

    input_levels = [int(x) for x in args.input_levels.split()]
    output_levels = [int(x) for x in args.output_levels.split()]
    seeds = [int(x) for x in args.seeds.split()]

    exp_a = analyze_experiment_a(args.results_dir, input_levels, args.fixed_output, seeds)
    exp_b = analyze_experiment_b(args.results_dir, output_levels, args.fixed_input, seeds)

    # Overall summary
    print("=" * 78)
    print("  Overall Summary")
    print("=" * 78)
    print()

    a_raw_pass = exp_a.get("raw_pass", False)
    a_adj_pass = exp_a.get("adj_pass", False)
    a_adj = exp_a.get("adjusted", {})
    b_pass = all(r["r_squared"] >= 0.95 for r in exp_b.values()) if exp_b else False

    a_adj_r2s = [r["r_squared"] for r in a_adj.values()] if a_adj else []
    b_r2s = [r["r_squared"] for r in exp_b.values()] if exp_b else []

    print(f"  Experiment A — raw TTFT vs input:      {'PASS' if a_raw_pass else 'FAIL'}")
    print(f"  Experiment A — adjusted TTFT vs input:  {'PASS' if a_adj_pass else 'FAIL'}")
    if a_adj_r2s:
        print(f"    Adjusted R² range: [{min(a_adj_r2s):.6f}, {max(a_adj_r2s):.6f}]")
    print()
    print(f"  Experiment B — decode time vs output:   {'PASS' if b_pass else 'FAIL'}")
    if b_r2s:
        print(f"    R² range: [{min(b_r2s):.6f}, {max(b_r2s):.6f}]")
    print()

    # The adjusted TTFT is the true test of the latency model's phase structure
    if a_adj_pass and b_pass:
        print("  ✓ HYPOTHESIS CONFIRMED: Both phase relationships are linear (R² ≥ 0.95)")
        print("    TTFT ∝ input_tokens and (E2E − TTFT) ∝ output_tokens")
        if not a_raw_pass:
            print("    (Raw TTFT contaminated by Poisson queueing noise; adjusted TTFT confirms linearity)")
    elif a_adj_pass or b_pass:
        print("  ~ PARTIALLY CONFIRMED: One phase is linear, the other is not")
        if not a_adj_pass:
            print("    TTFT vs input_tokens: non-linear even after removing scheduling delay")
        if not b_pass:
            print("    Decode vs output_tokens: non-linear — investigate step time model")
    else:
        print("  ✗ HYPOTHESIS REFUTED: Neither phase relationship is linear")
        print("    The alpha/beta coefficient model may have fundamental non-linearities")

    print()


if __name__ == "__main__":
    main()
