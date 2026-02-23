#!/usr/bin/env python3
"""Analysis script for H-Step-Quantum: Step-Time Quantum vs DES-M/M/1 Divergence.

Compares DES wait times against M/M/1 analytical predictions at multiple beta
coefficient scalings to test whether discrete step-time quantum causes the
DES-to-M/M/1 divergence observed in H-MMK.

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
  - Per-request JSON via --results-path
  - scheduling_delay_ms in per-request JSON is in TICKS (microseconds), not ms
    (known unit inconsistency — see sim/metrics_utils.go line 27)
  - e2e_ms is in actual milliseconds
  - Aggregate scheduling_delay_p99_ms IS in ms (goes through CalculatePercentile)

M/M/1 analytical formulas:
  rho = lambda / mu
  W_q = rho / (mu * (1 - rho))  # mean wait time in queue (seconds)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path


# -- M/M/1 Analytical Model ------------------------------------------------

def mm1_mean_wait_s(lam, mu):
    """Mean wait time in queue (W_q) for M/M/1 in seconds."""
    rho = lam / mu
    if rho >= 1.0:
        return float('inf')
    return rho / (mu * (1.0 - rho))


def mm1_mean_sojourn_s(lam, mu):
    """Mean sojourn time (W = W_q + 1/mu) for M/M/1 in seconds."""
    rho = lam / mu
    if rho >= 1.0:
        return float('inf')
    return 1.0 / (mu * (1.0 - rho))


# -- DES Output Parsing -----------------------------------------------------

def parse_results_json(filepath):
    """Parse BLIS --results-path JSON -> per-request data.

    Returns dict with wait_times_ms, e2e_times_ms, service_times_ms lists
    and conservation counts.

    JSON fields verified against sim/metrics_utils.go:
      - RequestMetrics.SchedulingDelay (json:"scheduling_delay_ms") → TICKS (us)
      - RequestMetrics.E2E (json:"e2e_ms") → actual ms
      - MetricsOutput.InjectedRequests (json:"injected_requests")
      - MetricsOutput.CompletedRequests (json:"completed_requests")
      - MetricsOutput.StillQueued (json:"still_queued")
      - MetricsOutput.StillRunning (json:"still_running")
    """
    data = json.loads(Path(filepath).read_text())
    completed = [r for r in data.get('requests', []) if r['e2e_ms'] > 0]

    # scheduling_delay_ms is in TICKS (microseconds), convert to ms
    wait_times_ms = [r['scheduling_delay_ms'] / 1000.0 for r in completed]
    e2e_times_ms = [r['e2e_ms'] for r in completed]
    service_times_ms = [
        r['e2e_ms'] - r['scheduling_delay_ms'] / 1000.0
        for r in completed
    ]

    return {
        'completed': len(completed),
        'injected': data.get('injected_requests', 0),
        'still_queued': data.get('still_queued', 0),
        'still_running': data.get('still_running', 0),
        'wait_times_ms': wait_times_ms,
        'e2e_times_ms': e2e_times_ms,
        'service_times_ms': service_times_ms,
    }


# -- Little's Law -----------------------------------------------------------

def verify_littles_law(data):
    """Verify L = lambda * W from per-request data.

    Uses e2e_times_ms for sojourn time W.
    Lambda = completed / sim_duration.
    """
    e2e_times = data['e2e_times_ms']
    if len(e2e_times) < 2:
        return None

    # Effective throughput from completion count / total time span
    total_e2e_s = sum(e2e_times) / 1000.0
    n = len(e2e_times)

    # We use the standard approach: lambda_eff * W_mean
    # For a stable system, lambda_eff ≈ arrival rate
    # W = mean sojourn time
    w_mean_s = sum(e2e_times) / n / 1000.0

    return {
        'W_mean_ms': w_mean_s * 1000.0,
        'n_completed': n,
    }


# -- Analysis ---------------------------------------------------------------

def analyze(results_dir, beta_scales, mu_values, service_ms_values,
            step_us_values, num_requests):
    """Main analysis: compare DES vs M/M/1 at each beta scaling."""
    rhos = [0.3, 0.5, 0.7, 0.9]
    seeds = [42, 123, 456]

    print("Configuration:")
    for i, scale in enumerate(beta_scales):
        print(f"  Scale {scale}x: mu={mu_values[i]:.4f} req/s, "
              f"service_time={service_ms_values[i]:.1f} ms, "
              f"step_time={step_us_values[i]:.0f} us")
    print(f"  Utilization levels: {rhos}")
    print(f"  Seeds: {seeds}")
    print(f"  Requests per run: {num_requests}")
    print()

    # Collect all results for cross-scale comparison
    all_results = {}  # (scale, rho) -> {wq_ana_ms, wq_des_ms, pct_err, ...}

    for i, scale in enumerate(beta_scales):
        mu = mu_values[i]
        svc_ms = service_ms_values[i]
        step_us = step_us_values[i]

        print(f"{'=' * 74}")
        print(f"  Beta scale {scale}x: step_time={step_us:.0f} us, "
              f"service_time={svc_ms:.1f} ms")
        print(f"{'=' * 74}")
        print()
        print(f"  {'rho':<6} {'W_q Ana (ms)':>14} {'W_q DES (ms)':>14} "
              f"{'Error':>8} {'Dir':>5} {'INV-1':>7} {'n':>6}")
        print(f"  {'-' * 6} {'-' * 14} {'-' * 14} {'-' * 8} {'-' * 5} {'-' * 7} {'-' * 6}")

        for rho in rhos:
            lam = rho * mu
            wq_ana_ms = mm1_mean_wait_s(lam, mu) * 1000.0

            wait_times_all = []
            conservation_ok = True
            for seed in seeds:
                fpath = os.path.join(results_dir, f"s{scale}_r{rho}_s{seed}.json")
                if not os.path.exists(fpath):
                    print(f"  WARNING: missing {fpath}", file=sys.stderr)
                    continue
                data = parse_results_json(fpath)
                wait_times_all.extend(data['wait_times_ms'])
                # Conservation check: injected == completed + queued + running
                total = data['completed'] + data['still_queued'] + data['still_running']
                if total != data['injected']:
                    conservation_ok = False
                    print(f"  WARNING: INV-1 FAIL for scale={scale} rho={rho} "
                          f"seed={seed}: {total} != {data['injected']}",
                          file=sys.stderr)

            if not wait_times_all:
                print(f"  {rho:<6} {'MISSING':>14} {'MISSING':>14} "
                      f"{'N/A':>8} {'N/A':>5} {'N/A':>7} {'0':>6}")
                continue

            wq_des_ms = sum(wait_times_all) / len(wait_times_all)

            if wq_ana_ms > 0:
                # Signed error: negative means DES < analytical
                signed_err = (wq_des_ms - wq_ana_ms) / wq_ana_ms * 100
            else:
                signed_err = 0.0

            direction = "LOW" if signed_err < 0 else "HIGH"
            inv1 = "OK" if conservation_ok else "FAIL"

            print(f"  {rho:<6} {wq_ana_ms:>14.2f} {wq_des_ms:>14.2f} "
                  f"{signed_err:>+7.1f}% {direction:>5} {inv1:>7} "
                  f"{len(wait_times_all):>6}")

            all_results[(scale, rho)] = {
                'wq_ana_ms': wq_ana_ms,
                'wq_des_ms': wq_des_ms,
                'signed_err': signed_err,
                'abs_err': abs(signed_err),
                'step_us': step_us,
                'svc_ms': svc_ms,
                'n_samples': len(wait_times_all),
            }

        print()

    # -- Cross-scale comparison table --
    print(f"{'=' * 74}")
    print("  Cross-Scale Comparison: |W_q error| vs step-time quantum")
    print(f"{'=' * 74}")
    print()
    print(f"  {'rho':<6}", end="")
    for scale in beta_scales:
        label = f"|err| @{scale}x"
        print(f" {label:>14}", end="")
    print(f" {'Monotonic?':>12}")
    print(f"  {'-' * 6}", end="")
    for _ in beta_scales:
        print(f" {'-' * 14}", end="")
    print(f" {'-' * 12}")

    monotonic_count = 0
    total_rows = 0
    for rho in rhos:
        print(f"  {rho:<6}", end="")
        errors = []
        for scale in beta_scales:
            key = (scale, rho)
            if key in all_results:
                err = all_results[key]['abs_err']
                errors.append(err)
                print(f" {err:>13.1f}%", end="")
            else:
                errors.append(None)
                print(f" {'N/A':>14}", end="")

        # Check monotonicity: error should decrease as scale decreases
        valid_errors = [e for e in errors if e is not None]
        if len(valid_errors) >= 2:
            total_rows += 1
            is_mono = all(valid_errors[j] >= valid_errors[j + 1]
                          for j in range(len(valid_errors) - 1))
            if is_mono:
                monotonic_count += 1
                print(f" {'YES':>12}")
            else:
                print(f" {'NO':>12}")
        else:
            print(f" {'N/A':>12}")

    print()
    if total_rows > 0:
        print(f"  Monotonicity: {monotonic_count}/{total_rows} utilization levels")
        print(f"  Hypothesis {'CONFIRMED' if monotonic_count == total_rows else 'NOT CONFIRMED'}: "
              f"reducing step quantum {'monotonically' if monotonic_count == total_rows else 'does NOT monotonically'} "
              f"reduces divergence")
    print()

    # -- Step-time ratio vs error ratio (linearity check) --
    print(f"{'=' * 74}")
    print("  Linearity Check: Does error scale proportionally with step time?")
    print(f"{'=' * 74}")
    print()
    print(f"  For linear scaling: error_ratio ≈ step_time_ratio")
    print(f"  step_time_ratio = step_time(scale) / step_time(baseline)")
    print(f"  error_ratio = |error(scale)| / |error(baseline)|")
    print()

    baseline_scale = beta_scales[0]
    baseline_step_us = step_us_values[0]

    print(f"  {'rho':<6} {'scale':>6} {'step_ratio':>12} {'err_ratio':>12} "
          f"{'linear?':>10}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 10}")

    for rho in rhos:
        baseline_key = (baseline_scale, rho)
        if baseline_key not in all_results:
            continue
        baseline_err = all_results[baseline_key]['abs_err']
        if baseline_err < 0.1:
            continue  # skip if baseline error is negligible

        for j, scale in enumerate(beta_scales[1:], 1):
            key = (scale, rho)
            if key not in all_results:
                continue
            err = all_results[key]['abs_err']
            step_ratio = step_us_values[j] / baseline_step_us
            err_ratio = err / baseline_err if baseline_err > 0 else 0

            # Linear if err_ratio ≈ step_ratio within 50% tolerance
            if step_ratio > 0:
                ratio_of_ratios = err_ratio / step_ratio
                is_linear = 0.5 <= ratio_of_ratios <= 1.5
            else:
                is_linear = False

            linear_str = "YES" if is_linear else "NO"
            print(f"  {rho:<6} {scale:>6} {step_ratio:>12.3f} {err_ratio:>12.3f} "
                  f"{linear_str:>10}")

    print()

    # -- Service time composition table --
    print(f"{'=' * 74}")
    print("  Service Time Composition: Step-time vs Alpha overhead")
    print(f"{'=' * 74}")
    print()
    print(f"  As beta decreases, alpha overhead (constant) dominates service time.")
    print(f"  step_fraction = total_step_time / total_service_time")
    print()
    print(f"  {'scale':>6} {'svc_ms':>10} {'step_total_ms':>14} {'alpha_ms':>10} "
          f"{'step_frac':>10}")
    print(f"  {'-' * 6} {'-' * 10} {'-' * 14} {'-' * 10} {'-' * 10}")

    # For constant input=1, output=128:
    # step_total = prefill_step + 128 * decode_step
    # alpha_total = alpha0 + alpha1*1 + 128*alpha2 (queueing + output processing)
    alpha0, alpha1, alpha2 = 1601.35, 3.51, 1805.54
    for j, scale in enumerate(beta_scales):
        svc_ms = service_ms_values[j]
        # Compute step time in ms for constant output=128
        prefill_step_us = 6910.42 * scale + 17.67 * scale * 1  # beta0 + beta1*input
        decode_step_us = 6910.42 * scale + 2.84 * scale * 1    # beta0 + beta2*1
        step_total_ms = (prefill_step_us + 128 * decode_step_us) / 1000.0
        alpha_total_ms = (alpha0 + alpha1 * 1 + 128 * alpha2) / 1000.0
        step_frac = step_total_ms / svc_ms if svc_ms > 0 else 0

        print(f"  {scale:>6} {svc_ms:>10.1f} {step_total_ms:>14.1f} "
              f"{alpha_total_ms:>10.1f} {step_frac:>9.1%}")

    print()

    # -- Per-seed consistency check --
    print(f"{'=' * 74}")
    print("  Per-Seed Consistency: W_q DES across seeds")
    print(f"{'=' * 74}")
    print()

    for i, scale in enumerate(beta_scales):
        mu = mu_values[i]
        print(f"  Scale {scale}x:")
        print(f"    {'rho':<6} {'seed 42':>12} {'seed 123':>12} {'seed 456':>12} "
              f"{'CV':>8}")
        print(f"    {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8}")

        for rho in rhos:
            per_seed_wq = []
            for seed in seeds:
                fpath = os.path.join(results_dir, f"s{scale}_r{rho}_s{seed}.json")
                if not os.path.exists(fpath):
                    per_seed_wq.append(None)
                    continue
                data = parse_results_json(fpath)
                wq = sum(data['wait_times_ms']) / len(data['wait_times_ms']) \
                    if data['wait_times_ms'] else 0
                per_seed_wq.append(wq)

            vals = [v for v in per_seed_wq if v is not None]
            if len(vals) >= 2:
                mean_v = sum(vals) / len(vals)
                std_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
                cv = std_v / mean_v if mean_v > 0 else 0
            else:
                cv = 0

            parts = []
            for v in per_seed_wq:
                if v is not None:
                    parts.append(f"{v:>12.1f}")
                else:
                    parts.append(f"{'N/A':>12}")

            print(f"    {rho:<6} {parts[0]} {parts[1]} {parts[2]} {cv:>7.1%}")

        print()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="H-Step-Quantum: Step-Time Quantum Divergence Analyzer")
    parser.add_argument("--results-dir", required=True,
                        help="Directory with DES results")
    parser.add_argument("--beta-scales", required=True,
                        help="Comma-separated beta scale factors")
    parser.add_argument("--mu-values", required=True,
                        help="Comma-separated mu values (req/s) per scale")
    parser.add_argument("--service-ms-values", required=True,
                        help="Comma-separated service times (ms) per scale")
    parser.add_argument("--step-us-values", required=True,
                        help="Comma-separated step times (us) per scale")
    parser.add_argument("--num-requests", type=int, required=True,
                        help="Requests per run")
    args = parser.parse_args()

    beta_scales = [float(x) for x in args.beta_scales.split(",")]
    mu_values = [float(x) for x in args.mu_values.split(",")]
    service_ms_values = [float(x) for x in args.service_ms_values.split(",")]
    step_us_values = [float(x) for x in args.step_us_values.split(",")]

    if len(beta_scales) != len(mu_values):
        print("ERROR: beta_scales and mu_values must have same length",
              file=sys.stderr)
        sys.exit(1)

    analyze(args.results_dir, beta_scales, mu_values, service_ms_values,
            step_us_values, args.num_requests)


if __name__ == "__main__":
    main()
