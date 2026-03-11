#!/usr/bin/env python3
"""Analyze joint KV-scheduling optimization experiment results.

Usage: python3 analyze.py <results_dir>

Produces:
  - Per-KV-level summary table with per-SLO critical TTFT P99
  - Interaction analysis (additive vs multiplicative improvement)
  - Cross-KV-level trends
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output


def parse_per_slo_metrics(filepath):
    """Parse per-SLO TTFT/E2E from BLIS output.

    Output format (cmd/root.go line 967-969):
        === Per-SLO Metrics ===
          critical:
            TTFT: mean=X.XX p99=Y.YY (n=NNN)
            E2E:  mean=X.XX p99=Y.YY (n=NNN)
          standard:
            ...

    Returns dict: {slo_class: {'ttft_mean': float, 'ttft_p99': float,
                                'e2e_mean': float, 'e2e_p99': float, 'count': int}}
    Note: Per-SLO output is in TICKS (microseconds). We convert to ms here
    to match cluster JSON output (which uses _ms suffix fields).
    """
    path = Path(filepath)
    if not path.exists():
        return {}

    content = path.read_text()
    metrics = {}
    current_class = None
    in_slo_section = False

    for line in content.split('\n'):
        # Detect the Per-SLO Metrics section
        if '=== Per-SLO Metrics ===' in line:
            in_slo_section = True
            continue

        # Stop parsing at next section header
        if in_slo_section and line.startswith('===') and 'Per-SLO' not in line:
            in_slo_section = False
            current_class = None
            continue

        if not in_slo_section:
            continue

        # Match SLO class header: "  critical:" or "  standard:" etc.
        m = re.match(r'^\s{2}(\w+):$', line)
        if m and m.group(1) in ('critical', 'standard', 'sheddable', 'batch', 'background', 'default'):
            current_class = m.group(1)
            metrics[current_class] = {}
            continue

        if current_class:
            # Match TTFT line: "    TTFT: mean=X.XX p99=Y.YY (n=NNN)"
            m = re.match(r'\s+TTFT:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)', line)
            if m:
                # Convert ticks (us) to ms
                metrics[current_class]['ttft_mean'] = float(m.group(1)) / 1000.0
                metrics[current_class]['ttft_p99'] = float(m.group(2)) / 1000.0
                metrics[current_class]['count'] = int(m.group(3))
                continue

            # Match E2E line: "    E2E:  mean=X.XX p99=Y.YY (n=NNN)"
            m = re.match(r'\s+E2E:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)', line)
            if m:
                # Convert ticks (us) to ms
                metrics[current_class]['e2e_mean'] = float(m.group(1)) / 1000.0
                metrics[current_class]['e2e_p99'] = float(m.group(2)) / 1000.0
                current_class = None  # Reset after E2E (last line per class)
                continue

    return metrics


def safe_mean(values):
    """Compute mean of a list, returning 0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def analyze_results(results_dir):
    """Main analysis function."""
    results_dir = Path(results_dir)

    kv_levels = [5000, 2000, 1500, 1200]
    configs = ["large-tail", "large-slo", "elastic-tail", "elastic-slo"]
    config_labels = {
        "large-tail": "baseline",
        "large-slo": "kv-only",
        "elastic-tail": "batch-only",
        "elastic-slo": "JOINT",
    }
    seeds = [42, 123, 456]

    # Collect metrics: results[kv][config] = list of (cluster_metrics, per_slo_metrics)
    cluster_data = defaultdict(lambda: defaultdict(list))
    slo_data = defaultdict(lambda: defaultdict(list))

    timeouts = []

    for kv in kv_levels:
        for config in configs:
            for seed in seeds:
                label = f"kv{kv}_{config}_s{seed}"
                filepath = results_dir / f"{label}.txt"

                metrics = parse_blis_output(str(filepath))
                if metrics["timed_out"]:
                    timeouts.append(label)
                    continue

                cluster_data[kv][config].append(metrics)

                per_slo = parse_per_slo_metrics(str(filepath))
                slo_data[kv][config].append(per_slo)

    if timeouts:
        print(f"\n!!! TIMEOUTS ({len(timeouts)}): {', '.join(timeouts)}")
        print()

    # -- Main Results Table --
    print("=" * 130)
    print("JOINT OPTIMIZATION RESULTS -- Per-SLO Critical TTFT P99")
    print("=" * 130)
    print()

    header = (f"{'KV Blocks':>10s}   {'Config':<12s}   {'Crit TTFT P99':>14s}   "
              f"{'Cluster TTFT P99':>17s}   {'Batch Occ':>10s}   "
              f"{'KV Preempt':>11s}   {'Throughput':>11s}   {'Completed':>10s}")
    separator = (f"{'-'*10:>10s}   {'-'*12:<12s}   {'-'*14:>14s}   "
                 f"{'-'*17:>17s}   {'-'*10:>10s}   "
                 f"{'-'*11:>11s}   {'-'*11:>11s}   {'-'*10:>10s}")

    for kv in kv_levels:
        print(f"--- KV={kv} {'---' * 20}")
        print(header)
        print(separator)

        for config in configs:
            cdata = cluster_data[kv][config]
            sdata = slo_data[kv][config]

            if not cdata:
                print(f"{kv:>10d}   {config_labels[config]:<12s}   {'N/A':>14s}")
                continue

            n = len(cdata)

            # Cluster-level metrics
            cluster_ttft_p99 = safe_mean([d["ttft_p99"] for d in cdata])
            throughput = safe_mean([d["throughput"] for d in cdata])
            preemptions = safe_mean([d["preemption_count"] for d in cdata])
            completed = safe_mean([d["completed"] for d in cdata])

            # Batch occupancy: completed / (max_running * instances) as fraction
            # Approximation: throughput / (some theoretical max)
            # Actually compute from metrics if available

            # Per-SLO critical TTFT P99 (average across seeds)
            crit_ttft_p99_values = []
            for s in sdata:
                if 'critical' in s and 'ttft_p99' in s['critical']:
                    crit_ttft_p99_values.append(s['critical']['ttft_p99'])
            crit_ttft_p99 = safe_mean(crit_ttft_p99_values)

            crit_str = f"{crit_ttft_p99:>14.1f}" if crit_ttft_p99_values else f"{'N/A':>14s}"

            print(f"{kv:>10d}   {config_labels[config]:<12s}   {crit_str}   "
                  f"{cluster_ttft_p99:>17.1f}   "
                  f"{'---':>10s}   "
                  f"{preemptions:>11.0f}   {throughput:>11.2f}   {completed:>10.0f}")

        print()

    # -- Interaction Analysis --
    print()
    print("=" * 130)
    print("INTERACTION ANALYSIS")
    print("  ratio < 1.0 = improvement (lower latency is better)")
    print("  interaction = joint_improvement / (kv_improvement + batch_improvement)")
    print("  > 1.2 = SUPER-ADDITIVE, 0.8-1.2 = ADDITIVE, < 0.8 = SUB-ADDITIVE")
    print("=" * 130)
    print()

    interaction_header = (
        f"{'KV':>6s}   "
        f"{'batch-only crit':>16s}   {'kv-only crit':>14s}   {'joint crit':>12s}   "
        f"{'interaction':>12s}   {'type':>18s}"
    )
    interaction_sep = (
        f"{'-'*6:>6s}   "
        f"{'-'*16:>16s}   {'-'*14:>14s}   {'-'*12:>12s}   "
        f"{'-'*12:>12s}   {'-'*18:>18s}"
    )

    print(interaction_header)
    print(interaction_sep)

    for kv in kv_levels:
        # Extract per-SLO critical TTFT P99 for each config
        def get_crit_ttft_p99(config):
            values = []
            for s in slo_data[kv][config]:
                if 'critical' in s and 'ttft_p99' in s['critical']:
                    values.append(s['critical']['ttft_p99'])
            return safe_mean(values) if values else None

        baseline_crit = get_crit_ttft_p99("large-tail")
        kv_only_crit = get_crit_ttft_p99("large-slo")
        batch_only_crit = get_crit_ttft_p99("elastic-tail")
        joint_crit = get_crit_ttft_p99("elastic-slo")

        if baseline_crit is None or baseline_crit == 0:
            print(f"{kv:>6d}   INCOMPLETE DATA")
            continue

        # Compute ratios (value / baseline; < 1.0 = improvement)
        batch_ratio = batch_only_crit / baseline_crit if batch_only_crit else float('inf')
        kv_ratio = kv_only_crit / baseline_crit if kv_only_crit else float('inf')
        joint_ratio = joint_crit / baseline_crit if joint_crit else float('inf')

        # Compute improvements (1 - ratio; > 0 = improvement)
        kv_improv = 1.0 - kv_ratio
        batch_improv = 1.0 - batch_ratio
        joint_improv = 1.0 - joint_ratio
        additive_pred = kv_improv + batch_improv

        if additive_pred > 0.001:
            interaction_ratio = joint_improv / additive_pred
            if interaction_ratio > 1.2:
                itype = "SUPER-ADDITIVE"
            elif interaction_ratio > 0.8:
                itype = "ADDITIVE"
            else:
                itype = "SUB-ADDITIVE"
            interaction_str = f"{interaction_ratio:>12.2f}x"
        elif joint_improv > 0.001:
            itype = "EMERGENT (indiv=0)"
            interaction_str = f"{'inf':>12s}"
        else:
            itype = "NO IMPROVEMENT"
            interaction_str = f"{'N/A':>12s}"

        print(f"{kv:>6d}   "
              f"{batch_ratio:>16.3f}   {kv_ratio:>14.3f}   {joint_ratio:>12.3f}   "
              f"{interaction_str}   {itype:>18s}")

    # -- Detailed per-KV breakdown --
    print()
    print("=" * 130)
    print("DETAILED BREAKDOWN (all metrics, mean across seeds)")
    print("=" * 130)

    for kv in kv_levels:
        baseline = cluster_data[kv]["large-tail"]
        kv_only = cluster_data[kv]["large-slo"]
        batch_only = cluster_data[kv]["elastic-tail"]
        joint = cluster_data[kv]["elastic-slo"]

        if not baseline or not kv_only or not batch_only or not joint:
            print(f"\n  KV {kv}: INCOMPLETE DATA -- skipping")
            continue

        b_ttft = safe_mean([d["ttft_p99"] for d in baseline])
        kv_ttft = safe_mean([d["ttft_p99"] for d in kv_only])
        ba_ttft = safe_mean([d["ttft_p99"] for d in batch_only])
        j_ttft = safe_mean([d["ttft_p99"] for d in joint])

        b_e2e = safe_mean([d["e2e_p99"] for d in baseline])
        kv_e2e = safe_mean([d["e2e_p99"] for d in kv_only])
        ba_e2e = safe_mean([d["e2e_p99"] for d in batch_only])
        j_e2e = safe_mean([d["e2e_p99"] for d in joint])

        b_preempt = safe_mean([d["preemption_count"] for d in baseline])
        kv_preempt = safe_mean([d["preemption_count"] for d in kv_only])
        ba_preempt = safe_mean([d["preemption_count"] for d in batch_only])
        j_preempt = safe_mean([d["preemption_count"] for d in joint])

        b_thru = safe_mean([d["throughput"] for d in baseline])
        kv_thru = safe_mean([d["throughput"] for d in kv_only])
        ba_thru = safe_mean([d["throughput"] for d in batch_only])
        j_thru = safe_mean([d["throughput"] for d in joint])

        # Per-SLO critical
        def get_slo_mean(config, cls, metric):
            values = []
            for s in slo_data[kv][config]:
                if cls in s and metric in s[cls]:
                    values.append(s[cls][metric])
            return safe_mean(values) if values else None

        b_crit = get_slo_mean("large-tail", "critical", "ttft_p99")
        kv_crit = get_slo_mean("large-slo", "critical", "ttft_p99")
        ba_crit = get_slo_mean("elastic-tail", "critical", "ttft_p99")
        j_crit = get_slo_mean("elastic-slo", "critical", "ttft_p99")

        b_shed = get_slo_mean("large-tail", "sheddable", "ttft_p99")
        kv_shed = get_slo_mean("large-slo", "sheddable", "ttft_p99")
        ba_shed = get_slo_mean("elastic-tail", "sheddable", "ttft_p99")
        j_shed = get_slo_mean("elastic-slo", "sheddable", "ttft_p99")

        print(f"\n  KV Blocks: {kv}")
        print(f"  {'Metric':<24s} {'Baseline':>12s} {'KV-only':>12s} {'Batch-only':>12s} {'Joint':>12s}")
        print(f"  {'-'*24} {'-'*12:>12s} {'-'*12:>12s} {'-'*12:>12s} {'-'*12:>12s}")

        print(f"  {'Crit TTFT P99 (ms)':<24s} "
              f"{b_crit:>12.1f} {kv_crit:>12.1f} {ba_crit:>12.1f} {j_crit:>12.1f}"
              if all(v is not None for v in [b_crit, kv_crit, ba_crit, j_crit])
              else f"  {'Crit TTFT P99 (ms)':<24s} N/A")

        print(f"  {'Shed TTFT P99 (ms)':<24s} "
              f"{b_shed:>12.1f} {kv_shed:>12.1f} {ba_shed:>12.1f} {j_shed:>12.1f}"
              if all(v is not None for v in [b_shed, kv_shed, ba_shed, j_shed])
              else f"  {'Shed TTFT P99 (ms)':<24s} N/A")

        print(f"  {'Cluster TTFT P99 (ms)':<24s} {b_ttft:>12.1f} {kv_ttft:>12.1f} {ba_ttft:>12.1f} {j_ttft:>12.1f}")
        print(f"  {'Cluster E2E P99 (ms)':<24s} {b_e2e:>12.1f} {kv_e2e:>12.1f} {ba_e2e:>12.1f} {j_e2e:>12.1f}")
        print(f"  {'KV Preemptions':<24s} {b_preempt:>12.0f} {kv_preempt:>12.0f} {ba_preempt:>12.0f} {j_preempt:>12.0f}")
        print(f"  {'Throughput (req/s)':<24s} {b_thru:>12.2f} {kv_thru:>12.2f} {ba_thru:>12.2f} {j_thru:>12.2f}")

        # Cost analysis: sheddable TTFT P99 increase
        if all(v is not None for v in [b_shed, j_shed]) and b_shed > 0:
            shed_cost = j_shed / b_shed
            print(f"\n  Critical benefit (joint/baseline): {j_crit/b_crit:.3f}x"
                  if b_crit and j_crit else "")
            print(f"  Sheddable cost (joint/baseline):   {shed_cost:.3f}x")

    # -- Cross-KV trend --
    print()
    print("=" * 130)
    print("CROSS-KV TREND: Joint vs Baseline Critical TTFT P99 ratio")
    print("=" * 130)
    print(f"  {'KV Blocks':>12s} {'Baseline Crit':>14s} {'Joint Crit':>12s} {'Joint Ratio':>12s} {'KV Preempt (B)':>16s} {'KV Preempt (J)':>16s}")
    print(f"  {'-'*12:>12s} {'-'*14:>14s} {'-'*12:>12s} {'-'*12:>12s} {'-'*16:>16s} {'-'*16:>16s}")

    for kv in kv_levels:
        def get_crit_p99(config):
            values = []
            for s in slo_data[kv][config]:
                if 'critical' in s and 'ttft_p99' in s['critical']:
                    values.append(s['critical']['ttft_p99'])
            return safe_mean(values) if values else None

        b_crit = get_crit_p99("large-tail")
        j_crit = get_crit_p99("elastic-slo")
        b_preempt = safe_mean([d["preemption_count"] for d in cluster_data[kv]["large-tail"]]) if cluster_data[kv]["large-tail"] else 0
        j_preempt = safe_mean([d["preemption_count"] for d in cluster_data[kv]["elastic-slo"]]) if cluster_data[kv]["elastic-slo"] else 0

        if b_crit and j_crit and b_crit > 0:
            ratio = j_crit / b_crit
            print(f"  {kv:>12d} {b_crit:>14.1f} {j_crit:>12.1f} {ratio:>12.3f} {b_preempt:>16.0f} {j_preempt:>16.0f}")
        else:
            print(f"  {kv:>12d} {'N/A':>14s}")

    print()
    print("=" * 130)
    print("Analysis complete.")
    print("=" * 130)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)
    analyze_results(sys.argv[1])
