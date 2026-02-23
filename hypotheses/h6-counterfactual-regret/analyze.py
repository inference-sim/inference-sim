#!/usr/bin/env python3
"""Analysis script for H6: Counterfactual Regret — RR vs Weighted.

Parses BLIS multi-block output including trace summary for regret metrics.
Produces comparison tables for mean/max regret, TTFT, E2E, and target distribution.

Output format references:
  - cmd/root.go:539-540 — "Mean Regret: %.6f", "Max Regret: %.6f"
  - cmd/root.go:524-537 — trace summary format
  - sim/trace/summary.go — TraceSummary struct
  - sim/cluster/counterfactual.go — computeCounterfactual regret computation
"""
import re
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout


def parse_trace_summary(filepath):
    """Parse trace summary section from BLIS output.

    Returns dict with:
        mean_regret: float
        max_regret: float
        total_decisions: int
        admitted: int
        rejected: int
        unique_targets: int
        target_distribution: dict[str, int]  (instance_id -> count)
        timed_out: bool
    """
    defaults = {
        "mean_regret": 0.0,
        "max_regret": 0.0,
        "total_decisions": 0,
        "admitted": 0,
        "rejected": 0,
        "unique_targets": 0,
        "target_distribution": {},
        "timed_out": False,
    }

    if check_for_timeout(filepath):
        defaults["timed_out"] = True
        return defaults

    content = Path(filepath).read_text()

    # Mean Regret: 0.123456 (cmd/root.go:539)
    m = re.search(r"Mean Regret: ([0-9.]+)", content)
    if m:
        defaults["mean_regret"] = float(m.group(1))
    else:
        print(f"WARNING: 'Mean Regret' not found in {filepath}", file=sys.stderr)

    # Max Regret: 0.123456 (cmd/root.go:540)
    m = re.search(r"Max Regret: ([0-9.]+)", content)
    if m:
        defaults["max_regret"] = float(m.group(1))
    else:
        print(f"WARNING: 'Max Regret' not found in {filepath}", file=sys.stderr)

    # Total Decisions: N (cmd/root.go:524)
    m = re.search(r"Total Decisions: (\d+)", content)
    if m:
        defaults["total_decisions"] = int(m.group(1))

    # Admitted: N (cmd/root.go:525)
    m = re.search(r"Admitted: (\d+)", content)
    if m:
        defaults["admitted"] = int(m.group(1))

    # Rejected: N (cmd/root.go:526)
    m = re.search(r"Rejected: (\d+)", content)
    if m:
        defaults["rejected"] = int(m.group(1))

    # Unique Targets: N (cmd/root.go:527)
    m = re.search(r"Unique Targets: (\d+)", content)
    if m:
        defaults["unique_targets"] = int(m.group(1))

    # Target Distribution: (cmd/root.go:529-537)
    #   instance-0: 125
    #   instance-1: 125
    target_dist = {}
    # Instance IDs use underscore: "instance_0", "instance_1", etc. (cluster.go:55)
    for tm in re.finditer(r"^\s+(instance_\d+): (\d+)$", content, re.MULTILINE):
        target_dist[tm.group(1)] = int(tm.group(2))
    defaults["target_distribution"] = target_dist

    return defaults


def conservation_check(metrics, label):
    """INV-1: injected == completed + still_queued + still_running."""
    injected = metrics.get("injected", 0)
    completed = metrics.get("completed", 0)
    still_queued = metrics.get("still_queued", 0)
    still_running = metrics.get("still_running", 0)
    rhs = completed + still_queued + still_running
    ok = (injected == rhs)
    if not ok:
        print(f"  CONSERVATION VIOLATION [{label}]: injected={injected} != "
              f"completed({completed}) + queued({still_queued}) + running({still_running}) = {rhs}",
              file=sys.stderr)
    return ok


def compute_distribution_uniformity(target_dist):
    """Compute Jain's fairness index for target distribution."""
    if not target_dist:
        return 0.0
    counts = list(target_dist.values())
    n = len(counts)
    if n == 0:
        return 0.0
    total = sum(counts)
    sum_sq = sum(c * c for c in counts)
    if sum_sq == 0:
        return 0.0
    return (total * total) / (n * sum_sq)


def analyze_experiment(results_dir, prefix, seeds, label, rate):
    """Analyze one experiment (RR vs Weighted) across seeds."""
    print(f"\n{'='*70}")
    print(f"  {label} (rate={rate})")
    print(f"{'='*70}")

    rr_regrets = []
    w_regrets = []
    rr_max_regrets = []
    w_max_regrets = []

    # Per-seed comparison table
    print(f"\n{'Seed':>6} | {'Policy':>12} | {'Mean Regret':>12} | {'Max Regret':>11} | "
          f"{'TTFT mean':>10} | {'TTFT p99':>9} | {'E2E mean':>9} | {'Completed':>9} | {'Jain FI':>8}")
    print("-" * 110)

    all_conservation_ok = True

    for seed in seeds:
        rr_file = f"{results_dir}/{prefix}_rr_{seed}.txt"
        w_file = f"{results_dir}/{prefix}_weighted_{seed}.txt"

        rr_metrics = parse_blis_output(rr_file)
        w_metrics = parse_blis_output(w_file)
        rr_trace = parse_trace_summary(rr_file)
        w_trace = parse_trace_summary(w_file)

        if rr_metrics["timed_out"] or w_metrics["timed_out"]:
            print(f"  {seed:>6} | SKIPPED (timeout)")
            continue

        # Conservation check
        if not conservation_check(rr_metrics, f"RR seed={seed}"):
            all_conservation_ok = False
        if not conservation_check(w_metrics, f"Weighted seed={seed}"):
            all_conservation_ok = False

        rr_jain = compute_distribution_uniformity(rr_trace["target_distribution"])
        w_jain = compute_distribution_uniformity(w_trace["target_distribution"])

        rr_regrets.append(rr_trace["mean_regret"])
        w_regrets.append(w_trace["mean_regret"])
        rr_max_regrets.append(rr_trace["max_regret"])
        w_max_regrets.append(w_trace["max_regret"])

        print(f"  {seed:>6} | {'RR':>12} | {rr_trace['mean_regret']:>12.6f} | "
              f"{rr_trace['max_regret']:>11.6f} | {rr_metrics['ttft_mean']:>10.2f} | "
              f"{rr_metrics['ttft_p99']:>9.2f} | {rr_metrics['e2e_mean']:>9.2f} | "
              f"{rr_metrics['completed']:>9d} | {rr_jain:>8.4f}")
        print(f"  {seed:>6} | {'Weighted':>12} | {w_trace['mean_regret']:>12.6f} | "
              f"{w_trace['max_regret']:>11.6f} | {w_metrics['ttft_mean']:>10.2f} | "
              f"{w_metrics['ttft_p99']:>9.2f} | {w_metrics['e2e_mean']:>9.2f} | "
              f"{w_metrics['completed']:>9d} | {w_jain:>8.4f}")

    if not rr_regrets:
        print("\n  No valid results to analyze.")
        return None

    # Summary statistics
    print(f"\n--- Summary across seeds ---")
    rr_mean = sum(rr_regrets) / len(rr_regrets)
    w_mean = sum(w_regrets) / len(w_regrets)
    print(f"  RR  mean regret:      {rr_mean:.6f} (per-seed: {', '.join(f'{r:.6f}' for r in rr_regrets)})")
    print(f"  W   mean regret:      {w_mean:.6f} (per-seed: {', '.join(f'{r:.6f}' for r in w_regrets)})")

    rr_max_avg = sum(rr_max_regrets) / len(rr_max_regrets)
    w_max_avg = sum(w_max_regrets) / len(w_max_regrets)
    print(f"  RR  avg max regret:   {rr_max_avg:.6f} (per-seed: {', '.join(f'{r:.6f}' for r in rr_max_regrets)})")
    print(f"  W   avg max regret:   {w_max_avg:.6f} (per-seed: {', '.join(f'{r:.6f}' for r in w_max_regrets)})")

    # Effect size: how much more regret does RR have?
    if w_mean > 0:
        ratio = rr_mean / w_mean
        print(f"\n  Regret ratio (RR / Weighted): {ratio:.2f}x")
    elif rr_mean > 0:
        print(f"\n  Weighted has zero mean regret; RR has {rr_mean:.6f}")
    else:
        print(f"\n  Both policies have zero mean regret at this rate")

    # Per-seed directional consistency
    all_rr_higher = all(rr > w for rr, w in zip(rr_regrets, w_regrets))
    consistent_direction = all(rr >= w for rr, w in zip(rr_regrets, w_regrets))
    print(f"  Directional consistency (RR > Weighted in all seeds): {all_rr_higher}")

    if not all_conservation_ok:
        print(f"\n  WARNING: Conservation (INV-1) violation detected!")

    return {
        "rr_mean_regrets": rr_regrets,
        "w_mean_regrets": w_regrets,
        "rr_max_regrets": rr_max_regrets,
        "w_max_regrets": w_max_regrets,
        "all_rr_higher": all_rr_higher,
        "consistent_direction": consistent_direction,
        "conservation_ok": all_conservation_ok,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]
    seeds = [42, 123, 456]

    # Experiment 1: Core comparison at rate=200
    exp1 = analyze_experiment(results_dir, "exp1", seeds, "Experiment 1: Core Comparison", 200)

    # Experiment 2: Low-rate control at rate=100
    ctrl = analyze_experiment(results_dir, "ctrl", seeds, "Experiment 2: Low-Rate Control (ED-2)", 100)

    # Final verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    if exp1 and ctrl:
        # Core: RR should have significantly higher regret
        if exp1["all_rr_higher"]:
            rr_avg = sum(exp1["rr_mean_regrets"]) / len(exp1["rr_mean_regrets"])
            w_avg = sum(exp1["w_mean_regrets"]) / len(exp1["w_mean_regrets"])
            if w_avg > 0:
                effect = rr_avg / w_avg
                if effect > 1.2:
                    print(f"  CORE: CONFIRMED — RR mean regret {effect:.1f}x higher than Weighted (all seeds)")
                else:
                    print(f"  CORE: INCONCLUSIVE — RR regret only {effect:.1f}x higher (<1.2x threshold)")
            else:
                print(f"  CORE: CONFIRMED — RR has regret ({rr_avg:.6f}), Weighted has zero regret")
        elif exp1["consistent_direction"]:
            print(f"  CORE: WEAK — RR >= Weighted in all seeds but not strictly greater in all")
        else:
            print(f"  CORE: REFUTED — Weighted has higher regret in at least one seed")

        # Control: both should have low/minimal regret
        ctrl_rr_avg = sum(ctrl["rr_mean_regrets"]) / len(ctrl["rr_mean_regrets"])
        ctrl_w_avg = sum(ctrl["w_mean_regrets"]) / len(ctrl["w_mean_regrets"])
        if ctrl_rr_avg < 0.5 and ctrl_w_avg < 0.5:
            print(f"  CONTROL: Regret is minimal at low load (RR={ctrl_rr_avg:.6f}, W={ctrl_w_avg:.6f})")
        else:
            print(f"  CONTROL: Unexpectedly high regret at low load (RR={ctrl_rr_avg:.6f}, W={ctrl_w_avg:.6f})")

        # Conservation
        if exp1["conservation_ok"] and ctrl["conservation_ok"]:
            print(f"  INV-1: Conservation holds for all runs")
        else:
            print(f"  INV-1: CONSERVATION VIOLATION detected")


if __name__ == "__main__":
    main()
