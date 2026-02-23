#!/usr/bin/env python3
"""Analysis script for H15: Fitness Evaluation Ranks Prefix-Affinity Higher.

Parses BLIS output including fitness scores and produces comparison tables.

Hypothesis: With TTFT-heavy fitness weights, prefix-affinity routing should
receive a higher fitness score than load-only routing for prefix-heavy workloads.

Experiments:
  Exp 1: Prefix workload + TTFT-heavy weights (core test)
  Exp 2: Prefix workload + throughput-heavy weights (weight sensitivity)
  Exp 3: Non-prefix workload + TTFT-heavy weights (control — effect should vanish)

Fitness output format (cmd/root.go:491-501):
  === Fitness Evaluation ===
  Score: 0.123456
    mean_e2e: 0.123456
    p99_ttft: 0.123456
    throughput: 0.123456

Normalization (sim/cluster/metrics.go:424-478):
  Throughput: value / (value + 100)           — higher is better
  Latency:   1 / (1 + value / 1000)          — lower is better (1000 ticks = 1ms)

Usage: python3 analyze.py <results_dir>
"""
import json
import re
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout

SEEDS = [42, 123, 456]


def parse_fitness(filepath):
    """Parse fitness score and components from BLIS output.

    Returns dict with 'score' (float) and 'components' (dict of str->float),
    or None if fitness section not found.

    Output format reference: cmd/root.go:491-501
      === Fitness Evaluation ===
      Score: 0.123456
        key: 0.123456
    """
    path = Path(filepath)
    if not path.exists():
        print(f"WARNING: file missing for fitness parse: {filepath}", file=sys.stderr)
        return None

    content = path.read_text()

    # Find the Fitness Evaluation block
    fitness_match = re.search(
        r"=== Fitness Evaluation ===\s*\n(.*?)(?:\n===|\Z)",
        content,
        re.DOTALL,
    )
    if not fitness_match:
        print(f"WARNING: no fitness section in {filepath}", file=sys.stderr)
        return None

    block = fitness_match.group(1)

    # Parse Score line: "Score: 0.123456"
    score_match = re.search(r"Score:\s+([0-9.]+)", block)
    if not score_match:
        print(f"WARNING: no Score line in fitness block: {filepath}", file=sys.stderr)
        return None

    score = float(score_match.group(1))

    # Parse component lines: "  key: 0.123456"
    components = {}
    for comp_match in re.finditer(r"^\s+(\S+):\s+([0-9.]+)", block, re.MULTILINE):
        key = comp_match.group(1)
        if key == "Score":
            continue
        components[key] = float(comp_match.group(2))

    return {"score": score, "components": components}


def load_experiment(results_dir, prefix):
    """Load prefix-aware and load-only results for all seeds."""
    prefix_aware = {}
    load_only = {}
    for seed in SEEDS:
        pa_file = f"{results_dir}/{prefix}_prefix_aware_{seed}.txt"
        lo_file = f"{results_dir}/{prefix}_load_only_{seed}.txt"
        prefix_aware[seed] = {
            "metrics": parse_blis_output(pa_file),
            "fitness": parse_fitness(pa_file),
        }
        load_only[seed] = {
            "metrics": parse_blis_output(lo_file),
            "fitness": parse_fitness(lo_file),
        }
    return prefix_aware, load_only


def print_fitness_comparison(prefix_aware, load_only, title):
    """Print per-seed fitness score comparison."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()

    # Header
    fmt = "{:<8} {:<20} {:>12} {:>12} {:>12} {:>12}"
    print(fmt.format("Seed", "Config", "Fitness", "p99_ttft", "mean_e2e", "throughput"))
    print("-" * 90)

    for seed in SEEDS:
        pa = prefix_aware[seed]
        lo = load_only[seed]

        if pa["metrics"]["timed_out"] or lo["metrics"]["timed_out"]:
            print(f"  Seed {seed}: TIMED OUT -- skipping")
            continue

        pa_fit = pa["fitness"]
        lo_fit = lo["fitness"]

        if pa_fit is None or lo_fit is None:
            print(f"  Seed {seed}: FITNESS PARSE FAILED -- skipping")
            continue

        # Print prefix-aware row
        print(fmt.format(
            seed, "prefix-aware",
            f"{pa_fit['score']:.6f}",
            f"{pa_fit['components'].get('p99_ttft', 0):.6f}",
            f"{pa_fit['components'].get('mean_e2e', 0):.6f}",
            f"{pa_fit['components'].get('throughput', 0):.6f}",
        ))

        # Print load-only row
        print(fmt.format(
            "", "load-only",
            f"{lo_fit['score']:.6f}",
            f"{lo_fit['components'].get('p99_ttft', 0):.6f}",
            f"{lo_fit['components'].get('mean_e2e', 0):.6f}",
            f"{lo_fit['components'].get('throughput', 0):.6f}",
        ))

        # Print difference
        score_diff = pa_fit["score"] - lo_fit["score"]
        score_pct = (score_diff / max(lo_fit["score"], 1e-9)) * 100
        winner = "prefix-aware" if score_diff > 0 else "load-only"
        print(fmt.format(
            "", f"DIFF ({winner})",
            f"{score_diff:+.6f}",
            f"{pa_fit['components'].get('p99_ttft', 0) - lo_fit['components'].get('p99_ttft', 0):+.6f}",
            f"{pa_fit['components'].get('mean_e2e', 0) - lo_fit['components'].get('mean_e2e', 0):+.6f}",
            f"{pa_fit['components'].get('throughput', 0) - lo_fit['components'].get('throughput', 0):+.6f}",
        ))
        print(f"         Score diff: {score_pct:+.2f}%")
        print()

    print()


def print_raw_metrics_comparison(prefix_aware, load_only, title):
    """Print raw metric comparison (TTFT, E2E, throughput) for context."""
    print("=" * 90)
    print(f"  {title} — Raw Metrics")
    print("=" * 90)
    print()

    fmt = "{:<8} {:<20} {:>14} {:>14} {:>14} {:>12}"
    print(fmt.format("Seed", "Config", "TTFT mean(ms)", "TTFT p99(ms)",
                      "E2E mean(ms)", "Throughput"))
    print("-" * 90)

    for seed in SEEDS:
        pa = prefix_aware[seed]["metrics"]
        lo = load_only[seed]["metrics"]

        if pa["timed_out"] or lo["timed_out"]:
            continue

        print(fmt.format(seed, "prefix-aware",
                          f"{pa['ttft_mean']:.2f}",
                          f"{pa['ttft_p99']:.2f}",
                          f"{pa['e2e_mean']:.2f}",
                          f"{pa['throughput']:.2f}"))
        print(fmt.format("", "load-only",
                          f"{lo['ttft_mean']:.2f}",
                          f"{lo['ttft_p99']:.2f}",
                          f"{lo['e2e_mean']:.2f}",
                          f"{lo['throughput']:.2f}"))

        # Ratios
        ttft_ratio = pa["ttft_p99"] / max(lo["ttft_p99"], 0.001)
        e2e_ratio = pa["e2e_mean"] / max(lo["e2e_mean"], 0.001)
        tput_ratio = pa["throughput"] / max(lo["throughput"], 0.001)
        print(fmt.format("", "Ratio (PA/LO)",
                          f"{pa['ttft_mean']/max(lo['ttft_mean'],0.001):.3f}x",
                          f"{ttft_ratio:.3f}x",
                          f"{e2e_ratio:.3f}x",
                          f"{tput_ratio:.3f}x"))
        print()

    print()


def print_conservation_check(datasets):
    """Verify INV-1 for all runs."""
    print("=" * 90)
    print("  CONSERVATION CHECK (INV-1)")
    print("=" * 90)
    print()

    all_pass = True
    for exp_label, prefix_aware, load_only in datasets:
        for config_label, data in [("prefix-aware", prefix_aware), ("load-only", load_only)]:
            for seed in SEEDS:
                m = data[seed]["metrics"]
                if m["timed_out"]:
                    print(f"  [{exp_label}] {config_label} seed={seed}: SKIPPED (timeout)")
                    continue
                injected = m["injected"]
                completed = m["completed"]
                queued = m["still_queued"]
                running = m["still_running"]
                # Full INV-1: injected == completed + queued + running + dropped_unservable
                # DroppedUnservable is 0 for this experiment (abundant KV blocks, no drops)
                dropped = m.get("dropped_unservable", 0)
                conserved = (completed + queued + running + dropped) == injected
                status = "PASS" if conserved else "FAIL"
                if not conserved:
                    all_pass = False
                print(f"  [{exp_label}] {config_label} seed={seed}: "
                      f"injected={injected}, completed={completed}, "
                      f"queued={queued}, running={running} -> {status}")

    print()
    if all_pass:
        print("  OVERALL: ALL CONSERVATION CHECKS PASS")
    else:
        print("  OVERALL: SOME CONSERVATION CHECKS FAILED")
    print()


def compute_fitness_summary(prefix_aware, load_only):
    """Compute cross-seed fitness comparison summary."""
    valid_seeds = []
    pa_scores = []
    lo_scores = []
    score_diffs = []
    pa_wins = 0

    for seed in SEEDS:
        pa = prefix_aware[seed]
        lo = load_only[seed]
        if pa["metrics"]["timed_out"] or lo["metrics"]["timed_out"]:
            continue
        if pa["fitness"] is None or lo["fitness"] is None:
            continue

        valid_seeds.append(seed)
        pa_s = pa["fitness"]["score"]
        lo_s = lo["fitness"]["score"]
        pa_scores.append(pa_s)
        lo_scores.append(lo_s)
        diff = pa_s - lo_s
        score_diffs.append(diff)
        if pa_s > lo_s:
            pa_wins += 1

    if not valid_seeds:
        return None

    n = len(valid_seeds)
    avg_pa = sum(pa_scores) / n
    avg_lo = sum(lo_scores) / n
    avg_diff = sum(score_diffs) / n
    pct_diff = (avg_diff / max(avg_lo, 1e-9)) * 100

    return {
        "n": n,
        "pa_wins": pa_wins,
        "avg_pa": avg_pa,
        "avg_lo": avg_lo,
        "avg_diff": avg_diff,
        "pct_diff": pct_diff,
        "per_seed_diffs": dict(zip(valid_seeds, score_diffs)),
        "per_seed_pa": dict(zip(valid_seeds, pa_scores)),
        "per_seed_lo": dict(zip(valid_seeds, lo_scores)),
    }


def print_summary(label, stats):
    """Print fitness summary for one experiment."""
    print(f"  --- {label} ---")
    if stats is None:
        print("  No valid results.")
        print()
        return

    n = stats["n"]
    print(f"  Valid seeds: {n}/{len(SEEDS)}")
    print(f"  Seeds where prefix-aware wins: {stats['pa_wins']}/{n}")
    print(f"  Avg fitness (prefix-aware): {stats['avg_pa']:.6f}")
    print(f"  Avg fitness (load-only):    {stats['avg_lo']:.6f}")
    print(f"  Avg difference:             {stats['avg_diff']:+.6f} ({stats['pct_diff']:+.2f}%)")
    print(f"  Per-seed scores:")
    for seed in SEEDS:
        if seed in stats["per_seed_diffs"]:
            pa = stats["per_seed_pa"][seed]
            lo = stats["per_seed_lo"][seed]
            diff = stats["per_seed_diffs"][seed]
            pct = (diff / max(lo, 1e-9)) * 100
            winner = "PA" if diff > 0 else "LO"
            print(f"    seed={seed}: PA={pa:.6f}, LO={lo:.6f}, diff={diff:+.6f} ({pct:+.2f}%) [{winner}]")
    print()


def print_verdict(exp1_stats, exp2_stats, exp3_stats):
    """Print overall hypothesis verdict."""
    print("=" * 90)
    print("  OVERALL VERDICT")
    print("=" * 90)
    print()

    # Exp 1: core test — prefix workload + TTFT-heavy weights
    if exp1_stats:
        n = exp1_stats["n"]
        wins = exp1_stats["pa_wins"]
        pct = exp1_stats["pct_diff"]
        print(f"  Exp 1 (prefix + TTFT-heavy):      {wins}/{n} seeds PA wins, "
              f"avg diff={pct:+.2f}%")
        if wins == n and abs(pct) > 5:
            print("    => Prefix-aware DOMINATES with TTFT-heavy weights")
        elif wins == n:
            print("    => Prefix-aware wins but difference is small (<5%)")
        else:
            print("    => Mixed results across seeds")

    # Exp 2: throughput-heavy weights
    if exp2_stats:
        n = exp2_stats["n"]
        wins = exp2_stats["pa_wins"]
        pct = exp2_stats["pct_diff"]
        print(f"  Exp 2 (prefix + throughput-heavy): {wins}/{n} seeds PA wins, "
              f"avg diff={pct:+.2f}%")
        if wins < n:
            print("    => Weight sensitivity confirmed: throughput focus changes ranking")
        elif abs(pct) < 5:
            print("    => Difference compressed under throughput-heavy weights")

    # Exp 3: control — non-prefix workload
    if exp3_stats:
        n = exp3_stats["n"]
        wins = exp3_stats["pa_wins"]
        pct = exp3_stats["pct_diff"]
        print(f"  Exp 3 (no-prefix + TTFT-heavy):   {wins}/{n} seeds PA wins, "
              f"avg diff={pct:+.2f}%")
        if abs(pct) < 5:
            print("    => Control validates: advantage vanishes without prefix sharing")
        else:
            print("    => UNEXPECTED: fitness difference persists without prefix sharing")

    print()

    # Overall classification
    core_confirmed = (
        exp1_stats
        and exp1_stats["pa_wins"] == exp1_stats["n"]
        and exp1_stats["pct_diff"] > 5
    )
    control_vanishes = exp3_stats and abs(exp3_stats["pct_diff"]) < 5
    weight_sensitive = exp2_stats and exp2_stats["pct_diff"] < exp1_stats["pct_diff"] if exp1_stats and exp2_stats else False

    if core_confirmed and control_vanishes:
        print("  VERDICT: CONFIRMED -- Fitness evaluation correctly ranks prefix-affinity")
        print("    routing higher for prefix-heavy workloads with TTFT-heavy weights.")
        if weight_sensitive:
            print("    Weight sensitivity validated: throughput-heavy weights compress the difference.")
    elif core_confirmed:
        print("  VERDICT: CONFIRMED WITH NUANCE -- Prefix-affinity wins on prefix workloads")
        print("    but control experiment shows unexpected behavior.")
    elif exp1_stats and exp1_stats["pa_wins"] > 0:
        print("  VERDICT: PARTIALLY CONFIRMED -- Not all seeds show prefix-aware dominance")
    else:
        print("  VERDICT: REFUTED or INCONCLUSIVE")

    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # Load all experiments
    exp1_pa, exp1_lo = load_experiment(results_dir, "exp1")
    exp2_pa, exp2_lo = load_experiment(results_dir, "exp2")
    exp3_pa, exp3_lo = load_experiment(results_dir, "exp3")

    # Fitness comparison tables
    print_fitness_comparison(exp1_pa, exp1_lo,
                             "Exp 1: Prefix workload + TTFT-heavy weights")
    print_fitness_comparison(exp2_pa, exp2_lo,
                             "Exp 2: Prefix workload + throughput-heavy weights")
    print_fitness_comparison(exp3_pa, exp3_lo,
                             "Exp 3: Non-prefix workload + TTFT-heavy weights (CONTROL)")

    # Raw metrics for context
    print_raw_metrics_comparison(exp1_pa, exp1_lo,
                                  "Exp 1: Prefix workload")
    print_raw_metrics_comparison(exp3_pa, exp3_lo,
                                  "Exp 3: Non-prefix workload (CONTROL)")

    # Conservation check
    datasets = [
        ("Exp1", exp1_pa, exp1_lo),
        ("Exp2", exp2_pa, exp2_lo),
        ("Exp3", exp3_pa, exp3_lo),
    ]
    print_conservation_check(datasets)

    # Summaries
    print("=" * 90)
    print("  CROSS-SEED SUMMARIES")
    print("=" * 90)
    print()

    exp1_stats = compute_fitness_summary(exp1_pa, exp1_lo)
    exp2_stats = compute_fitness_summary(exp2_pa, exp2_lo)
    exp3_stats = compute_fitness_summary(exp3_pa, exp3_lo)

    print_summary("Exp 1: Prefix + TTFT-heavy", exp1_stats)
    print_summary("Exp 2: Prefix + Throughput-heavy", exp2_stats)
    print_summary("Exp 3: No-prefix + TTFT-heavy (CONTROL)", exp3_stats)

    # Verdict
    print_verdict(exp1_stats, exp2_stats, exp3_stats)


if __name__ == "__main__":
    main()
