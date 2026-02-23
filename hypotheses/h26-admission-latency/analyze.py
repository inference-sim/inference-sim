#!/usr/bin/env python3
"""H26: Admission Latency Causal Ordering — Analysis

Compares three configurations:
  A: admission-latency=0 (baseline)
  B: admission-latency=10000 (10ms)
  C: admission-latency=50000 (50ms)

Expected: TTFT and E2E mean increase by exactly the admission latency.
Verification: per-request scheduling_delay should also shift by the admission latency.

Output format cross-reference:
  - Aggregate JSON: sim/metrics_utils.go MetricsOutput struct
    - "e2e_mean_ms", "ttft_mean_ms" (in ms)
  - Per-request JSON: sim/metrics_utils.go RequestMetrics struct
    - "ttft_ms" (in ms, divided by 1e3 at sim/metrics.go:138)
    - "e2e_ms" (in ms, divided by 1e3 at sim/metrics.go:139)
    - "scheduling_delay_ms" (in TICKS/us despite name, sim/metrics.go:141)
"""

import json
import os
import sys


def load_results(results_dir, label):
    """Load aggregate and per-request JSON results."""
    agg_path = os.path.join(results_dir, f"results_{label}.json")
    if not os.path.exists(agg_path):
        print(f"ERROR: {agg_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(agg_path) as f:
        data = json.load(f)

    return data


def extract_aggregate(data):
    """Extract aggregate metrics from JSON."""
    return {
        "e2e_mean_ms": data.get("e2e_mean_ms", 0.0),
        "ttft_mean_ms": data.get("ttft_mean_ms", 0.0),
        "e2e_p99_ms": data.get("e2e_p99_ms", 0.0),
        "ttft_p99_ms": data.get("ttft_p99_ms", 0.0),
        "completed": data.get("completed_requests", 0),
        "scheduling_delay_p99_ms": data.get("scheduling_delay_p99_ms", 0.0),
    }


def extract_per_request(data):
    """Extract per-request metrics."""
    requests = data.get("requests", [])
    if not requests:
        print("WARNING: no per-request data found", file=sys.stderr)
        return []
    return requests


def compute_per_request_stats(requests):
    """Compute mean TTFT, E2E, and scheduling_delay from per-request data."""
    if not requests:
        return {"ttft_ms": 0, "e2e_ms": 0, "sched_delay_us": 0, "count": 0}

    ttfts = [r["ttft_ms"] for r in requests if r.get("ttft_ms", 0) > 0]
    e2es = [r["e2e_ms"] for r in requests if r.get("e2e_ms", 0) > 0]
    # scheduling_delay_ms is actually in ticks (us) — see MEMORY.md
    sched_delays = [r["scheduling_delay_ms"] for r in requests if r.get("scheduling_delay_ms", 0) > 0]

    return {
        "ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
        "e2e_ms": sum(e2es) / len(e2es) if e2es else 0,
        "sched_delay_us": sum(sched_delays) / len(sched_delays) if sched_delays else 0,
        "count": len(e2es),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    configs = {
        "A (latency=0)": {"label": "a", "latency_us": 0},
        "B (latency=10ms)": {"label": "b", "latency_us": 10000},
        "C (latency=50ms)": {"label": "c", "latency_us": 50000},
    }

    results = {}
    for name, info in configs.items():
        data = load_results(results_dir, info["label"])
        agg = extract_aggregate(data)
        per_req = extract_per_request(data)
        per_req_stats = compute_per_request_stats(per_req)
        results[name] = {
            "agg": agg,
            "per_req": per_req_stats,
            "latency_us": info["latency_us"],
        }

    # --- Print aggregate comparison ---
    print("=" * 70)
    print("H26: Admission Latency Causal Ordering — Results")
    print("=" * 70)
    print()

    print("--- Aggregate Metrics (from JSON output) ---")
    print(f"{'Config':<25} {'TTFT mean (ms)':<18} {'E2E mean (ms)':<18} {'Completed':<12}")
    print("-" * 70)
    for name, r in results.items():
        agg = r["agg"]
        print(f"{name:<25} {agg['ttft_mean_ms']:<18.4f} {agg['e2e_mean_ms']:<18.4f} {agg['completed']:<12}")

    print()

    # --- Compute deltas ---
    baseline = results["A (latency=0)"]

    print("--- Deltas vs Baseline (Config A) ---")
    print(f"{'Config':<25} {'TTFT delta (ms)':<18} {'E2E delta (ms)':<18} {'Expected (ms)':<18} {'TTFT match?':<14} {'E2E match?':<14}")
    print("-" * 105)

    all_pass = True
    for name, r in results.items():
        if r["latency_us"] == 0:
            continue
        expected_ms = r["latency_us"] / 1000.0
        ttft_delta = r["agg"]["ttft_mean_ms"] - baseline["agg"]["ttft_mean_ms"]
        e2e_delta = r["agg"]["e2e_mean_ms"] - baseline["agg"]["e2e_mean_ms"]

        # Tolerance: 0.1ms (100 us) — rounding precision
        ttft_ok = abs(ttft_delta - expected_ms) < 0.1
        e2e_ok = abs(e2e_delta - expected_ms) < 0.1
        if not ttft_ok or not e2e_ok:
            all_pass = False

        print(f"{name:<25} {ttft_delta:<18.4f} {e2e_delta:<18.4f} {expected_ms:<18.4f} {'PASS' if ttft_ok else 'FAIL':<14} {'PASS' if e2e_ok else 'FAIL':<14}")

    print()

    # --- Per-request analysis ---
    print("--- Per-Request Statistics ---")
    print(f"{'Config':<25} {'Mean TTFT (ms)':<18} {'Mean E2E (ms)':<18} {'Mean SchedDelay (us)':<22} {'Count':<8}")
    print("-" * 90)
    for name, r in results.items():
        pr = r["per_req"]
        print(f"{name:<25} {pr['ttft_ms']:<18.4f} {pr['e2e_ms']:<18.4f} {pr['sched_delay_us']:<22.1f} {pr['count']:<8}")

    print()

    # --- Scheduling delay deltas ---
    print("--- Scheduling Delay Deltas (per-request) ---")
    print("Note: scheduling_delay_ms field is in TICKS (us), not ms")
    print(f"{'Config':<25} {'SchedDelay delta (us)':<24} {'Expected delta (us)':<22} {'Match?':<10}")
    print("-" * 80)

    baseline_sched = baseline["per_req"]["sched_delay_us"]
    for name, r in results.items():
        if r["latency_us"] == 0:
            continue
        expected_us = r["latency_us"]
        actual_delta_us = r["per_req"]["sched_delay_us"] - baseline_sched
        # Tolerance: 100 us
        ok = abs(actual_delta_us - expected_us) < 100
        if not ok:
            all_pass = False
        print(f"{name:<25} {actual_delta_us:<24.1f} {expected_us:<22.1f} {'PASS' if ok else 'FAIL':<10}")

    print()

    # --- Linearity check ---
    print("--- Linearity Check ---")
    b_delta = results["B (latency=10ms)"]["agg"]["e2e_mean_ms"] - baseline["agg"]["e2e_mean_ms"]
    c_delta = results["C (latency=50ms)"]["agg"]["e2e_mean_ms"] - baseline["agg"]["e2e_mean_ms"]
    if b_delta > 0:
        ratio = c_delta / b_delta
        expected_ratio = 5.0  # 50ms / 10ms
        print(f"E2E delta ratio C/B: {ratio:.4f} (expected: {expected_ratio:.1f})")
        linear_ok = abs(ratio - expected_ratio) < 0.1
        print(f"Linearity: {'PASS' if linear_ok else 'FAIL'}")
        if not linear_ok:
            all_pass = False
    else:
        print("Cannot compute linearity (B delta is 0)")
        all_pass = False

    print()

    # --- Verdict ---
    print("=" * 70)
    if all_pass:
        print("VERDICT: H26 CONFIRMED")
        print("Admission latency delays TTFT and E2E by exactly the configured amount.")
        print("Causal ordering: Arrival -> Admission (+latency) -> Routing -> Queue -> Batch -> Step")
    else:
        print("VERDICT: H26 REFUTED or PARTIAL")
        print("Some deltas did not match expected values. Investigate event pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
