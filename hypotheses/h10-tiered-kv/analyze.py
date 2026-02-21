#!/usr/bin/env python3
"""Analysis script for H10: Tiered KV Cache (GPU+CPU Offload).

Parses BLIS multi-block output and produces comparison tables for:
  - Core: single-tier vs tiered across 3 seeds
  - Scaling: CPU tier size effect on preemptions
  - Bandwidth: transfer bandwidth sensitivity

Usage:
    python3 analyze.py core exp1_single_42.txt exp1_tiered_42.txt ...
    python3 analyze.py scaling exp2_cpu0.txt exp2_cpu100.txt ...
    python3 analyze.py bandwidth exp3_bw10.txt exp3_bw100.txt ...
"""

import json
import math
import re
import sys
from pathlib import Path


def parse_output(filepath):
    """Parse multi-block BLIS output into cluster metrics."""
    content = Path(filepath).read_text()

    # Check for timeout/crash
    if content.strip() == "TIMEOUT_OR_CRASH":
        return None

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    if not cluster:
        return None

    # Extract preemption rate from KV cache summary section.
    # Format: "Preemption Rate: 0.1750" (float, printed by cmd/root.go:544).
    # Bug history: Round 3 used r"Preemptions?: (\d+)" which matched nothing
    # (wrong field name + integer pattern for a float), causing 0 preemptions
    # to be reported everywhere. This masked real preemptions (17.5%) for two rounds.
    preemptions = 0
    preempt_match = re.search(r"Preemption Rate: ([0-9.]+)", content)
    if preempt_match:
        preemption_rate = float(preempt_match.group(1))
        # Convert rate to count: rate * completed_requests
        completed = cluster.get("completed_requests", 0)
        preemptions = int(round(preemption_rate * completed))

    # Extract rejected count
    rejected = 0
    rejected_match = re.search(r"Rejected Requests: (\d+)", content)
    if rejected_match:
        rejected = int(rejected_match.group(1))

    # Extract cache hit rate from KV cache summary section.
    # Format: "Cache Hit Rate: 0.0452" (printed by cmd/root.go:545).
    cache_hit_rate = 0.0
    cache_match = re.search(r"Cache Hit Rate: ([0-9.]+)", content)
    if cache_match:
        cache_hit_rate = float(cache_match.group(1))

    # Conservation check
    injected = cluster.get("injected_requests", 0)
    completed = cluster.get("completed_requests", 0)
    queued = cluster.get("still_queued", 0)
    running = cluster.get("still_running", 0)
    conserved = injected == (completed + queued + running)

    return {
        "ttft_mean": cluster.get("ttft_mean_ms", 0),
        "ttft_p99": cluster.get("ttft_p99_ms", 0),
        "e2e_mean": cluster.get("e2e_mean_ms", 0),
        "e2e_p99": cluster.get("e2e_p99_ms", 0),
        "throughput": cluster.get("responses_per_sec", 0),
        "completed": completed,
        "injected": injected,
        "preemptions": preemptions,
        "rejected": rejected,
        "conserved": conserved,
        "cache_hit_rate": cache_hit_rate,
    }


def analyze_core(files):
    """Experiment 1: Core hypothesis across seeds."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    seeds = sorted(
        {name.split("_")[2] for name in results if "exp1_" in name}
    )

    print(
        f"  {'Seed':<6} {'Tier':<16}"
        f" | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'E2E Mean':>10} {'P99':>10}"
        f" | {'Preempt':>7} {'Comp':>5} {'INV-1':>5}"
    )
    print(
        f"  {'-'*6} {'-'*16}"
        f"-+-{'-'*10} {'-'*10}"
        f"-+-{'-'*10} {'-'*10}"
        f"-+-{'-'*7} {'-'*5} {'-'*5}"
    )

    preempt_ratios = []
    for seed in seeds:
        single = results.get(f"exp1_single_{seed}")
        tiered = results.get(f"exp1_tiered_{seed}")

        for label, r in [("single-tier", single), ("tiered(CPU=500)", tiered)]:
            if r is None:
                print(
                    f"  {seed:<6} {label:<16}"
                    f" | {'TIMEOUT':>10} {'':>10}"
                    f" | {'':>10} {'':>10}"
                    f" | {'':>7} {'':>5} {'':>5}"
                )
                continue
            inv1 = "OK" if r["conserved"] else "FAIL"
            print(
                f"  {seed:<6} {label:<16}"
                f" | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['e2e_mean']:>10.1f} {r['e2e_p99']:>10.1f}"
                f" | {r['preemptions']:>7} {r['completed']:>5} {inv1:>5}"
            )

        if single and tiered:
            if single["preemptions"] > 0:
                reduction = (
                    (single["preemptions"] - tiered["preemptions"])
                    / single["preemptions"]
                    * 100
                )
                preempt_ratios.append(reduction)
                print(
                    f"  {'':>6} {'Effect':>16}"
                    f" | Preemption reduction: {reduction:.1f}%"
                    f" ({single['preemptions']} -> {tiered['preemptions']})"
                )
            elif tiered["preemptions"] == 0:
                preempt_ratios.append(0)
                print(
                    f"  {'':>6} {'Effect':>16}"
                    f" | No preemptions in either tier"
                )
            else:
                preempt_ratios.append(-100)
                print(
                    f"  {'':>6} {'Effect':>16}"
                    f" | UNEXPECTED: tiered has MORE preemptions"
                )
        print()

    # Summary
    if preempt_ratios:
        all_reduced = all(r >= 0 for r in preempt_ratios)
        significant = all(r > 20 for r in preempt_ratios)
        min_r = min(preempt_ratios)
        max_r = max(preempt_ratios)

        print(f"  Summary: Preemption reduction (single→tiered):")
        print(
            f"    min={min_r:.1f}%  max={max_r:.1f}%"
            f"  all_seeds_reduced={all_reduced}"
        )
        if significant:
            print(
                f"    CONFIRMED: Tiered KV consistently reduces preemptions"
                f" (>{min_r:.0f}% reduction across all seeds)"
            )
        elif all_reduced:
            print(
                f"    INCONCLUSIVE: Tiered reduces preemptions but"
                f" effect <20% in some seeds"
            )
        else:
            print(
                f"    REFUTED: Tiered KV does not consistently reduce"
                f" preemptions"
            )


def analyze_scaling(files):
    """Experiment 2: CPU tier size scaling."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    cpu_values = sorted(
        {
            int(name.replace("exp2_cpu", ""))
            for name in results
            if name.startswith("exp2_cpu")
        }
    )

    print(
        f"  {'CPU Blocks':>10}"
        f" | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'E2E P99':>10}"
        f" | {'Preempt':>7} {'Comp':>5} {'INV-1':>5}"
    )
    print(
        f"  {'-'*10}-+-{'-'*10} {'-'*10}-+-{'-'*10}-+-{'-'*7} {'-'*5} {'-'*5}"
    )

    prev_preempt = None
    monotonic = True
    for cpu in cpu_values:
        r = results.get(f"exp2_cpu{cpu}")
        if r is None:
            print(
                f"  {cpu:>10}"
                f" | {'TIMEOUT':>10} {'':>10}"
                f" | {'':>10}"
                f" | {'':>7} {'':>5} {'':>5}"
            )
            continue

        inv1 = "OK" if r["conserved"] else "FAIL"
        label = f"{cpu}" if cpu > 0 else "0 (single)"
        print(
            f"  {label:>10}"
            f" | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
            f" | {r['e2e_p99']:>10.1f}"
            f" | {r['preemptions']:>7} {r['completed']:>5} {inv1:>5}"
        )

        if prev_preempt is not None and r["preemptions"] > prev_preempt:
            monotonic = False
        prev_preempt = r["preemptions"]

    print()
    if monotonic:
        print(
            "  Preemptions are monotonically non-increasing"
            " as CPU tier size grows."
        )
    else:
        print(
            "  WARNING: Preemptions are NOT monotonically decreasing"
            " — possible interaction effect."
        )


def analyze_bandwidth(files):
    """Experiment 3: Transfer bandwidth sensitivity."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    bw_values = sorted(
        {
            int(name.replace("exp3_bw", ""))
            for name in results
            if name.startswith("exp3_bw")
        }
    )

    print(
        f"  {'Bandwidth':>10}"
        f" | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'E2E Mean':>10} {'P99':>10}"
        f" | {'Preempt':>7} {'Comp':>5}"
    )
    print(
        f"  {'-'*10}-+-{'-'*10} {'-'*10}-+-{'-'*10} {'-'*10}-+-{'-'*7} {'-'*5}"
    )

    for bw in bw_values:
        r = results.get(f"exp3_bw{bw}")
        if r is None:
            print(
                f"  {bw:>10}"
                f" | {'TIMEOUT':>10} {'':>10}"
                f" | {'':>10} {'':>10}"
                f" | {'':>7} {'':>5}"
            )
            continue

        print(
            f"  {bw:>10}"
            f" | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
            f" | {r['e2e_mean']:>10.1f} {r['e2e_p99']:>10.1f}"
            f" | {r['preemptions']:>7} {r['completed']:>5}"
        )


def analyze_confound(files):
    """Experiment 4: Confound matrix — routing × tier isolation."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    # Map names to the 2×2 matrix + control
    matrix = {
        "rr_single": ("round-robin", "single-tier",
                       results.get("exp4_rr_single")),
        "rr_tiered": ("round-robin", "tiered(CPU=500)",
                       results.get("exp4_rr_tiered")),
        "ll_single": ("least-loaded", "single-tier",
                       results.get("exp1_single_42")),
        "ll_tiered": ("least-loaded", "tiered(CPU=500)",
                       results.get("exp1_tiered_42")),
        "ll_noop":   ("least-loaded", "tiered(offload=1.0)",
                       results.get("exp4_ll_tiered_noop")),
    }

    print("  Confound Matrix: Routing × Tier (seed=42)")
    print()
    print(
        f"  {'Routing':<14} {'Tier':<22}"
        f" | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'E2E P99':>10}"
        f" | {'Preempt':>7} {'Comp':>5} {'INV-1':>5}"
    )
    print(
        f"  {'-'*14} {'-'*22}"
        f"-+-{'-'*10} {'-'*10}"
        f"-+-{'-'*10}"
        f"-+-{'-'*7} {'-'*5} {'-'*5}"
    )

    for key in ["rr_single", "rr_tiered", "ll_single", "ll_tiered",
                "ll_noop"]:
        routing, tier, r = matrix[key]
        if r is None:
            print(
                f"  {routing:<14} {tier:<22}"
                f" | {'TIMEOUT':>10} {'':>10}"
                f" | {'':>10}"
                f" | {'':>7} {'':>5} {'':>5}"
            )
            continue
        inv1 = "OK" if r["conserved"] else "FAIL"
        print(
            f"  {routing:<14} {tier:<22}"
            f" | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
            f" | {r['e2e_p99']:>10.1f}"
            f" | {r['preemptions']:>7} {r['completed']:>5} {inv1:>5}"
        )

    # Interpretation
    rr_s = matrix["rr_single"][2]
    rr_t = matrix["rr_tiered"][2]
    ll_s = matrix["ll_single"][2]
    ll_t = matrix["ll_tiered"][2]
    ll_n = matrix["ll_noop"][2]

    print()
    if rr_s and rr_t:
        if rr_s["preemptions"] > 0 and rr_t["preemptions"] < rr_s["preemptions"]:
            pct = (
                (rr_s["preemptions"] - rr_t["preemptions"])
                / rr_s["preemptions"] * 100
            )
            print(
                f"  Round-robin: tiered reduces preemptions by {pct:.1f}%"
                f" ({rr_s['preemptions']}→{rr_t['preemptions']})"
            )
        elif rr_s and rr_s["preemptions"] > 0:
            print(
                f"  Round-robin: single-tier has {rr_s['preemptions']}"
                f" preemptions; tiered has {rr_t['preemptions'] if rr_t else '?'}"
            )
        else:
            print("  Round-robin: no preemptions in either tier")

    if ll_t and ll_n:
        delta = ll_n["ttft_mean"] - ll_t["ttft_mean"]
        if abs(delta) < 1.0:
            print(
                f"  maybeOffload control: threshold=1.0 produces"
                f" TTFT={ll_n['ttft_mean']:.1f}ms vs"
                f" threshold=0.8 TTFT={ll_t['ttft_mean']:.1f}ms"
                f" (delta={delta:.1f}ms — maybeOffload has NO effect)"
            )
        else:
            print(
                f"  maybeOffload control: threshold=1.0 produces"
                f" TTFT={ll_n['ttft_mean']:.1f}ms vs"
                f" threshold=0.8 TTFT={ll_t['ttft_mean']:.1f}ms"
                f" (delta={delta:.1f}ms — maybeOffload IS the mechanism)"
            )


ANALYZERS = {
    "core": analyze_core,
    "scaling": analyze_scaling,
    "bandwidth": analyze_bandwidth,
    "confound": analyze_confound,
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <experiment-type> <files...>")
        print(f"Types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    experiment_type = sys.argv[1]
    files = sys.argv[2:]

    analyzer = ANALYZERS.get(experiment_type)
    if not analyzer:
        print(f"Unknown experiment type: {experiment_type}")
        print(f"Valid types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    analyzer(files)
