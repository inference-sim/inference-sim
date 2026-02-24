#!/usr/bin/env python3
"""H5 Mixed-Batch Additive Model Validation — Analysis.

Compares three mixed-batch models (baseline weighted-average, smooth-wa, additive)
against GuideLLM ground truth for 13 experiments across all QPS sweep points.

Usage:
    python3 analyze.py <results_dir> <ground_truth_dir>

Results dir expected contents (per experiment per benchmark):
    {exp}_b{i}_baseline.json   — weighted-average (current)
    {exp}_b{i}_smooth_wa.json  — smooth weighted-average (no branches)
    {exp}_b{i}_additive.json   — additive model
"""

import json
import sys
from pathlib import Path


EXPERIMENTS = [
    "jan30-llama2-7b-tp1-chatsweep",
    "jan30-llama2-7b-tp1-codesweep",
    "jan30-llama2-7b-tp2-chatsweep",
    "jan30-llama2-7b-tp2-codesweep",
    "jan30-llama2-7b-tp4-chatsweep",
    "jan30-llama2-7b-tp4-codesweep",
    "20260210-codellama-34b-tp2-chatsweep",
    "20260210-codellama-34b-tp2-codesweep",
    "20260210-llama2-70b-tp4-chatsweep",
    "20260210-llama2-70b-tp4-codesweep",
    "20260210-qwen3-14b-tp1-codesweep",
    "20260210-qwen3-14b-tp2-chatsweep",
    "dec17-tp1-qwen7-summarization",
]


def workload_type(exp_name):
    """Classify experiment by workload type."""
    if "codesweep" in exp_name:
        return "codesweep"
    elif "chatsweep" in exp_name:
        return "chatsweep"
    elif "summarization" in exp_name:
        return "summarization"
    return "unknown"


def load_blis_results(filepath):
    """Load BLIS per-request metrics from --results-path JSON."""
    path = Path(filepath)
    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: failed to load {filepath}: {e}", file=sys.stderr)
        return None

    requests = data.get("requests", [])
    if not requests:
        return {
            "ttft_mean_ms": data.get("ttft_mean_ms", 0),
            "e2e_mean_ms": data.get("e2e_mean_ms", 0),
            "tpot_mean_ms": data.get("itl_mean_ms", 0),
            "requests": [],
        }

    ttfts, tpots, e2es = [], [], []
    for r in requests:
        ttft = r.get("ttft_ms", 0)
        e2e = r.get("e2e_ms", 0)
        out_tokens = r.get("num_decode_tokens", 0)
        ttfts.append(ttft)
        e2es.append(e2e)
        if out_tokens > 1 and e2e > ttft:
            tpots.append((e2e - ttft) / (out_tokens - 1))

    return {
        "ttft_mean_ms": mean(ttfts) if ttfts else 0,
        "tpot_mean_ms": mean(tpots) if tpots else 0,
        "e2e_mean_ms": mean(e2es) if e2es else 0,
        "requests": requests,
    }


def load_guidellm_benchmark(filepath, benchmark_index):
    """Load ground truth from a specific GuideLLM benchmark."""
    path = Path(filepath)
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: failed to load {filepath}: {e}", file=sys.stderr)
        return None

    benchmarks = data.get("benchmarks", [])
    if benchmark_index >= len(benchmarks):
        return None

    b = benchmarks[benchmark_index]
    successful = b.get("requests", {}).get("successful", [])
    if not successful:
        return None

    strategy = b.get("config", {}).get("strategy", {})
    stype = strategy.get("type_", "unknown")
    rate = strategy.get("rate", 0)
    n = len(successful)
    dur = b.get("duration", 0)
    eff_rate = n / dur if dur > 0 else 0

    ttfts, tpots, e2es = [], [], []
    for r in successful:
        ttft = r.get("time_to_first_token_ms", 0)
        tpot = r.get("time_per_output_token_ms", 0)
        e2e_s = r.get("request_latency", 0)
        ttfts.append(ttft)
        tpots.append(tpot)
        e2es.append(e2e_s * 1000)

    metrics = b.get("metrics", {})
    ttft_agg = metrics.get("time_to_first_token_ms", {}).get("successful", {})
    tpot_agg = metrics.get("time_per_output_token_ms", {}).get("successful", {})
    e2e_agg = metrics.get("request_latency", {}).get("successful", {})

    ttft_mean = ttft_agg.get("mean") if ttft_agg else None
    tpot_mean = tpot_agg.get("mean") if tpot_agg else None
    e2e_mean_s = e2e_agg.get("mean") if e2e_agg else None

    return {
        "ttft_mean_ms": ttft_mean if ttft_mean is not None else (mean(ttfts) if ttfts else 0),
        "tpot_mean_ms": tpot_mean if tpot_mean is not None else (mean(tpots) if tpots else 0),
        "e2e_mean_ms": e2e_mean_s * 1000 if e2e_mean_s is not None else (mean(e2es) if e2es else 0),
        "strategy_type": stype,
        "rate": rate if rate else eff_rate,
        "eff_rate": eff_rate,
        "n_requests": n,
    }


def count_benchmarks(filepath):
    """Count number of benchmarks in a GuideLLM results file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        return len(data.get("benchmarks", []))
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return 0


def mean(values):
    if not values:
        return 0
    return sum(values) / len(values)


def mape(predicted, actual):
    if actual == 0:
        return None
    return abs(predicted - actual) / abs(actual)


def fmt_pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def fmt_delta(base, treatment):
    if base is None or treatment is None:
        return "N/A"
    delta = base - treatment
    direction = "+" if delta > 0 else ""
    return f"{direction}{delta * 100:.1f}pp"


def main():
    if len(sys.argv) < 3:
        print("Usage: analyze.py <results_dir> <ground_truth_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])

    # Collect per-experiment per-benchmark results
    all_rows = []

    for exp_name in EXPERIMENTS:
        guidellm_path = gt_dir / exp_name / "guidellm-results.json"
        n_benchmarks = count_benchmarks(guidellm_path)
        wtype = workload_type(exp_name)

        for bi in range(n_benchmarks):
            gt = load_guidellm_benchmark(guidellm_path, bi)
            if gt is None:
                continue

            row = {
                "experiment": exp_name,
                "benchmark": bi,
                "workload_type": wtype,
                "strategy_type": gt["strategy_type"],
                "rate": gt["eff_rate"],
                "gt_ttft": gt["ttft_mean_ms"],
                "gt_tpot": gt["tpot_mean_ms"],
                "gt_e2e": gt["e2e_mean_ms"],
            }

            for arm in ["baseline", "smooth_wa", "additive"]:
                results_path = results_dir / f"{exp_name}_b{bi}_{arm}.json"
                data = load_blis_results(results_path)
                if data is not None:
                    row[f"{arm}_ttft"] = data["ttft_mean_ms"]
                    row[f"{arm}_tpot"] = data["tpot_mean_ms"]
                    row[f"{arm}_e2e"] = data["e2e_mean_ms"]
                    row[f"{arm}_ttft_mape"] = mape(data["ttft_mean_ms"], gt["ttft_mean_ms"])
                    row[f"{arm}_tpot_mape"] = mape(data["tpot_mean_ms"], gt["tpot_mean_ms"])
                    row[f"{arm}_e2e_mape"] = mape(data["e2e_mean_ms"], gt["e2e_mean_ms"])
                else:
                    for m in ["ttft", "tpot", "e2e"]:
                        row[f"{arm}_{m}"] = None
                        row[f"{arm}_{m}_mape"] = None

            all_rows.append(row)

    if not all_rows:
        print("ERROR: no experiments produced results", file=sys.stderr)
        sys.exit(1)

    # ================================================================
    # Table 1: Per-experiment aggregate (average across all benchmarks)
    # ================================================================
    print("=" * 140)
    print("  H5 Mixed-Batch Additive Model — Per-Experiment Aggregate E2E MAPE")
    print("=" * 140)
    print()

    exp_agg = {}
    for row in all_rows:
        key = row["experiment"]
        if key not in exp_agg:
            exp_agg[key] = {"wtype": row["workload_type"], "rows": []}
        exp_agg[key]["rows"].append(row)

    hdr = (f"{'Experiment':<45} {'Type':<8} {'N':>3}  "
           f"{'BL E2E':>8} {'SWA E2E':>8} {'ADD E2E':>8}  "
           f"{'BL→SWA':>7} {'BL→ADD':>7}")
    print(hdr)
    print("-" * len(hdr))

    for exp_name in EXPERIMENTS:
        if exp_name not in exp_agg:
            continue
        info = exp_agg[exp_name]
        rows = info["rows"]

        bl_e2e = [r["baseline_e2e_mape"] for r in rows if r.get("baseline_e2e_mape") is not None]
        swa_e2e = [r["smooth_wa_e2e_mape"] for r in rows if r.get("smooth_wa_e2e_mape") is not None]
        add_e2e = [r["additive_e2e_mape"] for r in rows if r.get("additive_e2e_mape") is not None]

        avg_bl = mean(bl_e2e) if bl_e2e else None
        avg_swa = mean(swa_e2e) if swa_e2e else None
        avg_add = mean(add_e2e) if add_e2e else None

        print(f"{exp_name:<45} {info['wtype']:<8} {len(rows):>3}  "
              f"{fmt_pct(avg_bl):>8} {fmt_pct(avg_swa):>8} {fmt_pct(avg_add):>8}  "
              f"{fmt_delta(avg_bl, avg_swa):>7} {fmt_delta(avg_bl, avg_add):>7}")

    # ================================================================
    # Table 2: Aggregate by workload type
    # ================================================================
    print()
    print("=" * 100)
    print("  Aggregate E2E MAPE by Workload Type")
    print("=" * 100)
    print()

    type_groups = {}
    for row in all_rows:
        wtype = row["workload_type"]
        if wtype not in type_groups:
            type_groups[wtype] = []
        type_groups[wtype].append(row)

    hdr2 = (f"{'Workload Type':<16} {'N':>4}  "
            f"{'BL E2E':>8} {'SWA E2E':>8} {'ADD E2E':>8}  "
            f"{'BL→SWA':>7} {'BL→ADD':>7}  "
            f"{'BL TPOT':>8} {'ADD TPOT':>8} {'Δ TPOT':>7}")
    print(hdr2)
    print("-" * len(hdr2))

    for wtype in ["chatsweep", "codesweep", "summarization"]:
        group = type_groups.get(wtype, [])
        if not group:
            continue

        bl_e2e = [r["baseline_e2e_mape"] for r in group if r.get("baseline_e2e_mape") is not None]
        swa_e2e = [r["smooth_wa_e2e_mape"] for r in group if r.get("smooth_wa_e2e_mape") is not None]
        add_e2e = [r["additive_e2e_mape"] for r in group if r.get("additive_e2e_mape") is not None]
        bl_tpot = [r["baseline_tpot_mape"] for r in group if r.get("baseline_tpot_mape") is not None]
        add_tpot = [r["additive_tpot_mape"] for r in group if r.get("additive_tpot_mape") is not None]

        avg_bl = mean(bl_e2e) if bl_e2e else None
        avg_swa = mean(swa_e2e) if swa_e2e else None
        avg_add = mean(add_e2e) if add_e2e else None
        avg_bl_tpot = mean(bl_tpot) if bl_tpot else None
        avg_add_tpot = mean(add_tpot) if add_tpot else None

        print(f"{wtype:<16} {len(group):>4}  "
              f"{fmt_pct(avg_bl):>8} {fmt_pct(avg_swa):>8} {fmt_pct(avg_add):>8}  "
              f"{fmt_delta(avg_bl, avg_swa):>7} {fmt_delta(avg_bl, avg_add):>7}  "
              f"{fmt_pct(avg_bl_tpot):>8} {fmt_pct(avg_add_tpot):>8} {fmt_delta(avg_bl_tpot, avg_add_tpot):>7}")

    # Overall
    all_bl_e2e = [r["baseline_e2e_mape"] for r in all_rows if r.get("baseline_e2e_mape") is not None]
    all_swa_e2e = [r["smooth_wa_e2e_mape"] for r in all_rows if r.get("smooth_wa_e2e_mape") is not None]
    all_add_e2e = [r["additive_e2e_mape"] for r in all_rows if r.get("additive_e2e_mape") is not None]
    all_bl_tpot = [r["baseline_tpot_mape"] for r in all_rows if r.get("baseline_tpot_mape") is not None]
    all_add_tpot = [r["additive_tpot_mape"] for r in all_rows if r.get("additive_tpot_mape") is not None]

    print("-" * len(hdr2))
    overall_bl = mean(all_bl_e2e) if all_bl_e2e else None
    overall_swa = mean(all_swa_e2e) if all_swa_e2e else None
    overall_add = mean(all_add_e2e) if all_add_e2e else None
    overall_bl_tpot = mean(all_bl_tpot) if all_bl_tpot else None
    overall_add_tpot = mean(all_add_tpot) if all_add_tpot else None

    print(f"{'OVERALL':<16} {len(all_rows):>4}  "
          f"{fmt_pct(overall_bl):>8} {fmt_pct(overall_swa):>8} {fmt_pct(overall_add):>8}  "
          f"{fmt_delta(overall_bl, overall_swa):>7} {fmt_delta(overall_bl, overall_add):>7}  "
          f"{fmt_pct(overall_bl_tpot):>8} {fmt_pct(overall_add_tpot):>8} {fmt_delta(overall_bl_tpot, overall_add_tpot):>7}")

    # ================================================================
    # Table 3: Breakdown by strategy type (sync vs constant-rate vs throughput)
    # ================================================================
    print()
    print("=" * 100)
    print("  E2E MAPE by Concurrency Level (strategy type)")
    print("=" * 100)
    print()

    strat_groups = {}
    for row in all_rows:
        st = row["strategy_type"]
        if st not in strat_groups:
            strat_groups[st] = []
        strat_groups[st].append(row)

    hdr3 = (f"{'Strategy':<14} {'N':>4}  "
            f"{'BL E2E':>8} {'SWA E2E':>8} {'ADD E2E':>8}  "
            f"{'BL→SWA':>7} {'BL→ADD':>7}")
    print(hdr3)
    print("-" * len(hdr3))

    for stype in ["synchronous", "constant", "throughput"]:
        group = strat_groups.get(stype, [])
        if not group:
            continue

        bl_e2e = [r["baseline_e2e_mape"] for r in group if r.get("baseline_e2e_mape") is not None]
        swa_e2e = [r["smooth_wa_e2e_mape"] for r in group if r.get("smooth_wa_e2e_mape") is not None]
        add_e2e = [r["additive_e2e_mape"] for r in group if r.get("additive_e2e_mape") is not None]

        avg_bl = mean(bl_e2e) if bl_e2e else None
        avg_swa = mean(swa_e2e) if swa_e2e else None
        avg_add = mean(add_e2e) if add_e2e else None

        print(f"{stype:<14} {len(group):>4}  "
              f"{fmt_pct(avg_bl):>8} {fmt_pct(avg_swa):>8} {fmt_pct(avg_add):>8}  "
              f"{fmt_delta(avg_bl, avg_swa):>7} {fmt_delta(avg_bl, avg_add):>7}")

    # ================================================================
    # Accept Criteria Evaluation
    # ================================================================
    print()
    print("=" * 80)
    print("  Accept Criteria Evaluation")
    print("=" * 80)
    print()

    # Criterion 1: E2E MAPE improvement >= 2pp overall (additive vs baseline)
    if overall_bl is not None and overall_add is not None:
        e2e_improvement = overall_bl - overall_add
        c1_pass = e2e_improvement >= 0.02
        print(f"  1. E2E MAPE improvement >= 2pp (additive vs baseline):")
        print(f"     Baseline: {fmt_pct(overall_bl)}, Additive: {fmt_pct(overall_add)}")
        print(f"     {'PASS' if c1_pass else 'FAIL'}: {e2e_improvement * 100:.1f}pp improvement")
    else:
        print("  1. E2E MAPE improvement >= 2pp: UNABLE TO EVALUATE (missing data)")

    # Criterion 2: Improvement larger at higher QPS (where mixed batches dominate)
    sync_rows = strat_groups.get("synchronous", [])
    high_qps_rows = strat_groups.get("throughput", []) + strat_groups.get("constant", [])

    sync_bl = [r["baseline_e2e_mape"] for r in sync_rows if r.get("baseline_e2e_mape") is not None]
    sync_add = [r["additive_e2e_mape"] for r in sync_rows if r.get("additive_e2e_mape") is not None]
    high_bl = [r["baseline_e2e_mape"] for r in high_qps_rows if r.get("baseline_e2e_mape") is not None]
    high_add = [r["additive_e2e_mape"] for r in high_qps_rows if r.get("additive_e2e_mape") is not None]

    if sync_bl and sync_add and high_bl and high_add:
        sync_improvement = mean(sync_bl) - mean(sync_add)
        high_improvement = mean(high_bl) - mean(high_add)
        c2_pass = high_improvement > sync_improvement
        print(f"\n  2. Improvement concentrated at higher QPS:")
        print(f"     Sync improvement: {sync_improvement * 100:.1f}pp")
        print(f"     High-QPS improvement: {high_improvement * 100:.1f}pp")
        print(f"     {'PASS' if c2_pass else 'FAIL'}: "
              f"{'high-QPS' if c2_pass else 'sync'} shows larger improvement")
    else:
        print("\n  2. Improvement at higher QPS: UNABLE TO EVALUATE (missing data)")

    # Criterion 3: Smooth-WA vs additive (isolate branch removal from structural change)
    if overall_swa is not None and overall_add is not None:
        additive_vs_swa = overall_swa - overall_add
        swa_vs_baseline = overall_bl - overall_swa if overall_bl is not None else None
        print(f"\n  3. Decomposition of improvement:")
        print(f"     Branch removal (BL→SWA): {fmt_delta(overall_bl, overall_swa)}")
        print(f"     Structural change (SWA→ADD): {fmt_delta(overall_swa, overall_add)}")
        print(f"     Total (BL→ADD): {fmt_delta(overall_bl, overall_add)}")
    else:
        print("\n  3. Decomposition: UNABLE TO EVALUATE (missing data)")

    # Criterion 4: Vanishing effect at synchronous rate
    if sync_bl and sync_add:
        sync_diff = abs(mean(sync_bl) - mean(sync_add))
        print(f"\n  4. Vanishing effect at synchronous rate (ED-2 control):")
        print(f"     Difference: {sync_diff * 100:.2f}pp")
        if sync_diff < 0.01:
            print(f"     PASS: < 1pp difference (no mixed batches at sync rate)")
        else:
            print(f"     NOTE: {sync_diff * 100:.1f}pp difference even at sync rate")
    else:
        print("\n  4. Vanishing effect: UNABLE TO EVALUATE (missing data)")

    print()


if __name__ == "__main__":
    main()
