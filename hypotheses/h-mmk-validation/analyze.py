#!/usr/bin/env python3
"""Analysis script for H-MMK: Cross-Validate DES Against M/M/k Analytical Model.

Compares BLIS DES output against M/M/1 and M/M/k analytical queueing models.

Analytical formulas (standard queueing theory):

M/M/1:
  rho = lambda / mu
  W_q = rho / (mu * (1 - rho))           # mean wait time in queue
  W   = 1 / (mu * (1 - rho))             # mean sojourn time (wait + service)
  L_q = rho^2 / (1 - rho)                # mean queue length
  L   = rho / (1 - rho)                  # mean number in system

M/M/k:
  rho = lambda / (k * mu)                # per-server utilization
  a   = lambda / mu = k * rho            # offered load (Erlangs)
  P0  = 1 / [sum_{n=0}^{k-1} a^n/n! + a^k / (k! * (1-rho))]
  C(k,a) = (a^k / (k! * (1-rho))) * P0  # Erlang C (probability of waiting)
  L_q = C(k,a) * rho / (1 - rho)         # mean queue length
  W_q = L_q / lambda                     # mean wait time in queue
  W   = W_q + 1/mu                       # mean sojourn time
  L   = lambda * W                       # mean number in system (Little's Law)

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
  - Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
  - Per-request data in JSON when --results-path is used
  - scheduling_delay in per-request JSON is in TICKS (microseconds), not ms
  - Aggregate scheduling_delay_p99_ms IS in ms (goes through CalculatePercentile)
"""
import argparse
import json
import math
import os
from pathlib import Path


# ── M/M/1 Analytical Model ──────────────────────────────────────────────────

def mm1_mean_wait(lam, mu):
    """Mean wait time in queue (W_q) for M/M/1."""
    rho = lam / mu
    if rho >= 1.0:
        return float('inf')
    return rho / (mu * (1.0 - rho))


def mm1_mean_sojourn(lam, mu):
    """Mean sojourn time (W = W_q + 1/mu) for M/M/1."""
    rho = lam / mu
    if rho >= 1.0:
        return float('inf')
    return 1.0 / (mu * (1.0 - rho))


def mm1_mean_queue_length(lam, mu):
    """Mean queue length (L_q) for M/M/1."""
    rho = lam / mu
    if rho >= 1.0:
        return float('inf')
    return rho * rho / (1.0 - rho)


def mm1_mean_in_system(lam, mu):
    """Mean number in system (L) for M/M/1."""
    rho = lam / mu
    if rho >= 1.0:
        return float('inf')
    return rho / (1.0 - rho)


# ── M/M/k Analytical Model ──────────────────────────────────────────────────

def mmk_p0(k, lam, mu):
    """Probability of empty system (P0) for M/M/k."""
    rho = lam / (k * mu)
    if rho >= 1.0:
        return 0.0
    a = lam / mu  # offered load in Erlangs
    # Sum: sum_{n=0}^{k-1} a^n / n!
    s = sum(a ** n / math.factorial(n) for n in range(k))
    # Last term: a^k / (k! * (1-rho))
    last = (a ** k) / (math.factorial(k) * (1.0 - rho))
    return 1.0 / (s + last)


def mmk_erlang_c(k, lam, mu):
    """Erlang C formula: probability that an arriving customer must wait."""
    rho = lam / (k * mu)
    if rho >= 1.0:
        return 1.0
    a = lam / mu
    p0 = mmk_p0(k, lam, mu)
    return (a ** k / (math.factorial(k) * (1.0 - rho))) * p0


def mmk_mean_wait(k, lam, mu):
    """Mean wait time in queue (W_q) for M/M/k."""
    rho = lam / (k * mu)
    if rho >= 1.0:
        return float('inf')
    c = mmk_erlang_c(k, lam, mu)
    l_q = c * rho / (1.0 - rho)
    return l_q / lam


def mmk_mean_sojourn(k, lam, mu):
    """Mean sojourn time (W = W_q + 1/mu) for M/M/k."""
    return mmk_mean_wait(k, lam, mu) + 1.0 / mu


def mmk_mean_queue_length(k, lam, mu):
    """Mean queue length (L_q) for M/M/k."""
    rho = lam / (k * mu)
    if rho >= 1.0:
        return float('inf')
    c = mmk_erlang_c(k, lam, mu)
    return c * rho / (1.0 - rho)


def mmk_mean_in_system(k, lam, mu):
    """Mean number in system (L = lambda * W) for M/M/k."""
    return lam * mmk_mean_sojourn(k, lam, mu)


# ── Little's Law ─────────────────────────────────────────────────────────────

def verify_littles_law(requests, sim_duration_s):
    """Verify L = lambda * W from per-request data.

    Computes:
      lambda_eff = completed / sim_duration
      W = mean E2E (sojourn time)
      L_computed = lambda_eff * W
      L_empirical = time-averaged number in system (from per-request intervals)

    Returns dict with computed values and % error.
    """
    completed = [r for r in requests if r['e2e_ms'] > 0]
    if len(completed) < 2:
        return None

    # Effective arrival rate
    lam_eff = len(completed) / sim_duration_s

    # Mean sojourn time in seconds
    w_mean_s = sum(r['e2e_ms'] for r in completed) / len(completed) / 1000.0

    # L from Little's Law
    l_littles = lam_eff * w_mean_s

    # L empirical: time-average of number in system
    # Build event list: +1 at arrival, -1 at departure
    events = []
    for r in completed:
        t_arrive = r['arrived_at']  # seconds
        t_depart = t_arrive + r['e2e_ms'] / 1000.0
        events.append((t_arrive, +1))
        events.append((t_depart, -1))
    events.sort(key=lambda e: (e[0], e[1]))

    # Compute time-weighted average
    total_area = 0.0
    current_n = 0
    prev_time = events[0][0]
    for t, delta in events:
        if t > prev_time:
            total_area += current_n * (t - prev_time)
        current_n += delta
        prev_time = t

    span = events[-1][0] - events[0][0]
    l_empirical = total_area / span if span > 0 else 0

    pct_error = abs(l_littles - l_empirical) / max(l_empirical, 1e-9) * 100

    return {
        'lambda_eff': lam_eff,
        'W_mean_s': w_mean_s,
        'L_littles': l_littles,
        'L_empirical': l_empirical,
        'pct_error': pct_error,
    }


# ── DES Output Parsing ──────────────────────────────────────────────────────

def parse_results_json(filepath):
    """Parse BLIS --results-path JSON output → per-request and cluster metrics."""
    data = json.loads(Path(filepath).read_text())
    completed = [r for r in data.get('requests', []) if r['e2e_ms'] > 0]

    # scheduling_delay in per-request JSON is in TICKS (microseconds), not ms
    # (known unit inconsistency — see sim/metrics_utils.go line 142)
    wait_times_ms = [r['scheduling_delay_ms'] / 1000.0 for r in completed]
    e2e_times_ms = [r['e2e_ms'] for r in completed]

    # Service time = E2E - scheduling_delay
    service_times_ms = [
        r['e2e_ms'] - r['scheduling_delay_ms'] / 1000.0
        for r in completed
    ]

    return {
        'completed': len(completed),
        'injected': data.get('injected_requests', 0),
        'still_queued': data.get('still_queued', 0),
        'still_running': data.get('still_running', 0),
        'e2e_mean_ms': data.get('e2e_mean_ms', 0),
        'ttft_mean_ms': data.get('ttft_mean_ms', 0),
        'responses_per_sec': data.get('responses_per_sec', 0),
        'sim_duration_s': data.get('vllm_estimated_duration_s', 0),
        'wait_times_ms': wait_times_ms,
        'e2e_times_ms': e2e_times_ms,
        'service_times_ms': service_times_ms,
        'requests': completed,
    }


def parse_stdout(filepath):
    """Parse BLIS stdout for cluster-level JSON metrics."""
    content = Path(filepath).read_text()
    import re
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block
    return cluster


# ── KS Test (manual, no scipy dependency) ────────────────────────────────────

def ks_test_exponential(samples, rate):
    """One-sample KS test against Exponential(rate) distribution.

    Returns (D_statistic, approximate_p_value).
    Uses the Dvoretzky-Kiefer-Wolfowitz inequality for p-value approximation:
      P(D_n > x) <= 2 * exp(-2 * n * x^2)
    """
    n = len(samples)
    if n == 0:
        return (1.0, 0.0)

    sorted_samples = sorted(samples)
    d_max = 0.0

    for i, x in enumerate(sorted_samples):
        # CDF of Exponential(rate): F(x) = 1 - exp(-rate * x)
        cdf = 1.0 - math.exp(-rate * x) if x >= 0 else 0.0
        # Empirical CDF: F_n(x) = (i+1)/n (after), F_n(x-) = i/n (before)
        d_plus = abs((i + 1) / n - cdf)
        d_minus = abs(i / n - cdf)
        d_max = max(d_max, d_plus, d_minus)

    # DKW inequality for approximate p-value
    p_approx = 2.0 * math.exp(-2.0 * n * d_max * d_max)
    p_approx = min(p_approx, 1.0)

    return (d_max, p_approx)


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_mm1(results_dir, mu, rhos, seeds):
    """Sub-experiment 1: Compare single-instance DES against M/M/1 analytical."""
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  Sub-experiment 1: M/M/1 (k=1) — DES vs Analytical                │")
    print("├────────┬──────────────┬──────────────┬──────────┬──────────────────┤")
    print("│  rho   │ W_q Ana (ms) │ W_q DES (ms) │ Error %  │ KS p-value (svc)│")
    print("├────────┼──────────────┼──────────────┼──────────┼──────────────────┤")

    all_results = []
    for rho in rhos:
        lam = rho * mu  # arrival rate for k=1
        wq_analytical_s = mm1_mean_wait(lam, mu)
        wq_analytical_ms = wq_analytical_s * 1000.0

        wait_times_all = []
        service_times_all = []
        for seed in seeds:
            fpath = os.path.join(results_dir, f"mm1_r{rho}_s{seed}.json")
            if not os.path.exists(fpath):
                print(f"│  {rho:<5} │ {'MISSING':>12} │ {'MISSING':>12} │ {'N/A':>8} │ {'N/A':>16} │")
                continue
            data = parse_results_json(fpath)
            wait_times_all.extend(data['wait_times_ms'])
            service_times_all.extend(data['service_times_ms'])

        if not wait_times_all:
            continue

        wq_des_ms = sum(wait_times_all) / len(wait_times_all)

        if wq_analytical_ms > 0:
            pct_err = abs(wq_des_ms - wq_analytical_ms) / wq_analytical_ms * 100
        else:
            pct_err = abs(wq_des_ms)  # analytical is 0 — error is the raw DES value

        # KS test on service times against Exponential(mu)
        # Service times are in ms, mu is in req/s, so rate_ms = mu / 1000
        svc_times_s = [t / 1000.0 for t in service_times_all if t > 0]
        _, ks_p = ks_test_exponential(svc_times_s, mu)

        status = "PASS" if pct_err < 5.0 else ("WARN" if pct_err < 10.0 else "FAIL")
        ks_status = f"{ks_p:.4f}" if ks_p is not None else "N/A"

        print(f"│  {rho:<5} │ {wq_analytical_ms:>12.2f} │ {wq_des_ms:>12.2f} │ {pct_err:>6.1f}% {status} │ {ks_status:>16} │")

        all_results.append({
            'rho': rho, 'wq_ana_ms': wq_analytical_ms, 'wq_des_ms': wq_des_ms,
            'pct_err': pct_err, 'ks_p': ks_p, 'n_samples': len(wait_times_all),
        })

    print("└────────┴──────────────┴──────────────┴──────────┴──────────────────┘")
    print(f"  Tolerance: <5% = PASS, 5-10% = WARN, >10% = FAIL")
    print(f"  KS test: p > 0.05 = service times consistent with exponential")
    print()

    # Sojourn time (W) comparison
    print("  M/M/1 Sojourn Time (W = W_q + 1/mu):")
    print(f"  {'rho':<8} {'W Ana (ms)':>12} {'W DES (ms)':>12} {'Error %':>10}")
    for rho in rhos:
        lam = rho * mu
        w_ana_ms = mm1_mean_sojourn(lam, mu) * 1000.0
        fpath = os.path.join(results_dir, f"mm1_r{rho}_s{seeds[0]}.json")
        if not os.path.exists(fpath):
            continue
        data = parse_results_json(fpath)
        w_des_ms = data['e2e_mean_ms']
        pct = abs(w_des_ms - w_ana_ms) / w_ana_ms * 100 if w_ana_ms > 0 else 0
        print(f"  {rho:<8} {w_ana_ms:>12.2f} {w_des_ms:>12.2f} {pct:>8.1f}%")
    print()

    return all_results


def analyze_kxmm1(results_dir, mu, rhos, seeds):
    """Sub-experiment 2: Compare k×M/M/1 (round-robin) against per-instance M/M/1."""
    k = 4
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  Sub-experiment 2: k×M/M/1 (k=4, round-robin)                     │")
    print("│  Each instance sees Poisson(lambda/4) → compare per-instance M/M/1 │")
    print("├────────┬──────────────┬──────────────┬──────────┬──────────────────┤")
    print("│  rho   │ W_q Ana (ms) │ W_q DES (ms) │ Error %  │ Conservation    │")
    print("├────────┼──────────────┼──────────────┼──────────┼──────────────────┤")

    for rho in rhos:
        lam_total = rho * k * mu
        lam_per_instance = lam_total / k  # Poisson splitting
        wq_analytical_ms = mm1_mean_wait(lam_per_instance, mu) * 1000.0

        wait_times_all = []
        conservation_ok = True
        for seed in seeds:
            fpath = os.path.join(results_dir, f"kxmm1_r{rho}_s{seed}.json")
            if not os.path.exists(fpath):
                continue
            data = parse_results_json(fpath)
            wait_times_all.extend(data['wait_times_ms'])
            # Conservation check: injected == completed + queued + running
            total = data['completed'] + data['still_queued'] + data['still_running']
            if total != data['injected']:
                conservation_ok = False

        if not wait_times_all:
            print(f"│  {rho:<5} │ {'MISSING':>12} │ {'MISSING':>12} │ {'N/A':>8} │ {'N/A':>16} │")
            continue

        wq_des_ms = sum(wait_times_all) / len(wait_times_all)
        pct_err = abs(wq_des_ms - wq_analytical_ms) / max(wq_analytical_ms, 0.001) * 100
        status = "PASS" if pct_err < 5.0 else ("WARN" if pct_err < 10.0 else "FAIL")
        cons = "INV-1 OK" if conservation_ok else "INV-1 FAIL"

        print(f"│  {rho:<5} │ {wq_analytical_ms:>12.2f} │ {wq_des_ms:>12.2f} │ {pct_err:>6.1f}% {status} │ {cons:>16} │")

    print("└────────┴──────────────┴──────────────┴──────────┴──────────────────┘")
    print()


def analyze_mmk(results_dir, mu, rhos, seeds):
    """Sub-experiment 3: Compare least-loaded (JSQ) against M/M/k analytical."""
    k = 4
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  Sub-experiment 3: M/M/k approximation (k=4, least-loaded)         │")
    print("│  Least-loaded ≈ JSQ → compare against M/M/k shared-queue model     │")
    print("├────────┬──────────────┬──────────────┬──────────┬──────────────────┤")
    print("│  rho   │ W_q Ana (ms) │ W_q DES (ms) │ Error %  │ vs k×M/M/1 DES │")
    print("├────────┼──────────────┼──────────────┼──────────┼──────────────────┤")

    for rho in rhos:
        lam = rho * k * mu
        wq_mmk_ms = mmk_mean_wait(k, lam, mu) * 1000.0

        # M/M/k DES (least-loaded)
        wait_mmk_des = []
        for seed in seeds:
            fpath = os.path.join(results_dir, f"mmk_r{rho}_s{seed}.json")
            if not os.path.exists(fpath):
                continue
            data = parse_results_json(fpath)
            wait_mmk_des.extend(data['wait_times_ms'])

        # k×M/M/1 DES (round-robin) for comparison
        wait_kxmm1_des = []
        for seed in seeds:
            fpath = os.path.join(results_dir, f"kxmm1_r{rho}_s{seed}.json")
            if not os.path.exists(fpath):
                continue
            data = parse_results_json(fpath)
            wait_kxmm1_des.extend(data['wait_times_ms'])

        if not wait_mmk_des:
            print(f"│  {rho:<5} │ {'MISSING':>12} │ {'MISSING':>12} │ {'N/A':>8} │ {'N/A':>16} │")
            continue

        wq_des_ms = sum(wait_mmk_des) / len(wait_mmk_des)
        pct_err = abs(wq_des_ms - wq_mmk_ms) / max(wq_mmk_ms, 0.001) * 100
        status = "PASS" if pct_err < 5.0 else ("WARN" if pct_err < 10.0 else "FAIL")

        # How much better is least-loaded vs round-robin?
        if wait_kxmm1_des:
            wq_kxmm1 = sum(wait_kxmm1_des) / len(wait_kxmm1_des)
            if wq_kxmm1 > 0:
                improvement = (1.0 - wq_des_ms / wq_kxmm1) * 100
                vs_kxmm1 = f"{improvement:>+.1f}% vs RR"
            else:
                vs_kxmm1 = "~0 both"
        else:
            vs_kxmm1 = "N/A"

        print(f"│  {rho:<5} │ {wq_mmk_ms:>12.2f} │ {wq_des_ms:>12.2f} │ {pct_err:>6.1f}% {status} │ {vs_kxmm1:>16} │")

    print("└────────┴──────────────┴──────────────┴──────────┴──────────────────┘")
    print()

    # Erlang C table for reference
    print("  M/M/k Reference (Erlang C probabilities):")
    print(f"  {'rho':<8} {'P(wait)':>10} {'L_q':>10} {'W_q (ms)':>12} {'W (ms)':>12}")
    for rho in rhos:
        lam = rho * k * mu
        ec = mmk_erlang_c(k, lam, mu)
        lq = mmk_mean_queue_length(k, lam, mu)
        wq = mmk_mean_wait(k, lam, mu) * 1000
        w = mmk_mean_sojourn(k, lam, mu) * 1000
        print(f"  {rho:<8} {ec:>10.4f} {lq:>10.4f} {wq:>12.2f} {w:>12.2f}")
    print()


def analyze_littles_law(results_dir, rhos, seeds):
    """Verify Little's Law (L = lambda * W) across all sub-experiments."""
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  Little's Law Verification: L = lambda * W                         │")
    print("├───────────────┬────────┬──────────┬──────────┬──────────┬──────────┤")
    print("│  Experiment   │  rho   │ L(lit)   │ L(emp)   │ Error %  │ Status   │")
    print("├───────────────┼────────┼──────────┼──────────┼──────────┼──────────┤")

    prefixes = [
        ("M/M/1 k=1", "mm1"),
        ("k×M/M/1 RR", "kxmm1"),
        ("M/M/k LL", "mmk"),
    ]

    for label, prefix in prefixes:
        for rho in rhos:
            fpath = os.path.join(results_dir, f"{prefix}_r{rho}_s{seeds[0]}.json")
            if not os.path.exists(fpath):
                continue
            data = parse_results_json(fpath)
            result = verify_littles_law(data['requests'], data['sim_duration_s'])
            if result is None:
                continue
            pct = result['pct_error']
            status = "PASS" if pct < 5.0 else ("WARN" if pct < 10.0 else "FAIL")
            print(f"│ {label:<13} │  {rho:<5} │ {result['L_littles']:>8.3f} │ {result['L_empirical']:>8.3f} │ {pct:>6.1f}%  │ {status:<8} │")

    print("└───────────────┴────────┴──────────┴──────────┴──────────┴──────────┘")
    print(f"  L(lit) = lambda_eff * W_mean, L(emp) = time-avg number in system")
    print(f"  Tolerance: <5% = PASS (universal law, should always hold)")
    print()


def main():
    parser = argparse.ArgumentParser(description="M/M/k Cross-Validation Analyzer")
    parser.add_argument("--results-dir", required=True, help="Directory with DES results")
    parser.add_argument("--mu", type=float, required=True, help="Service rate (req/s)")
    parser.add_argument("--mean-service-ms", type=float, required=True, help="Mean service time (ms)")
    parser.add_argument("--output-token-mean", type=int, required=True, help="Mean output tokens")
    parser.add_argument("--num-requests", type=int, required=True, help="Requests per run")
    args = parser.parse_args()

    mu = args.mu
    rhos = [0.3, 0.5, 0.7, 0.85]
    seeds = [42, 123, 456]

    print(f"Configuration:")
    print(f"  mu = {mu:.6f} req/s (1/mu = {1000/mu:.2f} ms)")
    print(f"  Mean service time = {args.mean_service_ms:.2f} ms")
    print(f"  Output token mean = {args.output_token_mean}")
    print(f"  Requests per run  = {args.num_requests}")
    print(f"  Seeds: {seeds}")
    print()

    analyze_mm1(args.results_dir, mu, rhos, seeds)
    analyze_kxmm1(args.results_dir, mu, rhos, seeds)
    analyze_mmk(args.results_dir, mu, rhos, seeds)
    analyze_littles_law(args.results_dir, rhos, seeds)

    # Summary
    print("============================================================================")
    print("  Summary")
    print("============================================================================")
    print()
    print("  Sub-exp 1 (M/M/1): Cleanest validation — single instance, no routing.")
    print("    If this fails, the service-time or event-loop model diverges from M/M/1.")
    print()
    print("  Sub-exp 2 (k×M/M/1): Tests cluster decomposition with round-robin.")
    print("    Poisson splitting property: each instance should be independent M/M/1.")
    print("    If this fails, round-robin routing introduces correlation between instances.")
    print()
    print("  Sub-exp 3 (M/M/k): Tests how close least-loaded routing gets to shared queue.")
    print("    Expected: close but not identical (snapshot-based routing ≠ shared queue).")
    print("    The gap quantifies the architectural cost of per-instance queues.")
    print()
    print("  Little's Law: Universal — must hold regardless of architecture.")
    print("    If this fails, there is a measurement or conservation bug.")
    print()


if __name__ == "__main__":
    main()
