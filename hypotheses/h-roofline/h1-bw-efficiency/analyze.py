#!/usr/bin/env python3
"""H1 Bandwidth Efficiency Validation — Analysis.

Compares BLIS roofline v2 predictions (baseline vs treatment) against
GuideLLM ground truth for 13 experiments.

Usage:
    python3 analyze.py <results_dir> <ground_truth_dir>

Results dir expected contents (per experiment):
    {exp_name}_baseline.json  — BLIS results with raw peak BW
    {exp_name}_treatment.json — BLIS results with bwEfficiencyFactor=0.82

Ground truth dir expected contents (per experiment):
    {exp_name}/guidellm-results.json — benchmark[0] = synchronous mode
"""

import json
import sys
from pathlib import Path


# All 13 experiments in the matrix
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
    """Classify experiment by workload type for directional analysis."""
    if "codesweep" in exp_name:
        return "codesweep"
    elif "chatsweep" in exp_name:
        return "chatsweep"
    elif "summarization" in exp_name:
        return "summarization"
    return "unknown"


def load_blis_results(filepath):
    """Load BLIS per-request metrics from --results-path JSON.

    Returns dict with:
        ttft_mean_ms, tpot_mean_ms, e2e_mean_ms — aggregate means
        requests — list of per-request dicts
    Or None if file missing/invalid.
    """
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
        # Fall back to aggregate metrics
        return {
            "ttft_mean_ms": data.get("ttft_mean_ms", 0),
            "e2e_mean_ms": data.get("e2e_mean_ms", 0),
            "tpot_mean_ms": data.get("itl_mean_ms", 0),
            "requests": [],
        }

    # Compute per-request TPOT: (E2E - TTFT) / (output_tokens - 1) where output > 1
    ttfts = []
    tpots = []
    e2es = []
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


def load_guidellm_ground_truth(filepath):
    """Load ground truth from GuideLLM benchmarks[0] (synchronous mode).

    Returns dict with:
        ttft_mean_ms, tpot_mean_ms, e2e_mean_ms — from synchronous benchmark
        requests — list of per-request dicts
    Or None if file missing/invalid.
    """
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
    if not benchmarks:
        return None

    b0 = benchmarks[0]
    successful = b0.get("requests", {}).get("successful", [])
    if not successful:
        return None

    # Per-request extraction
    ttfts = []
    tpots = []
    e2es = []
    for r in successful:
        ttft = r.get("time_to_first_token_ms", 0)
        tpot = r.get("time_per_output_token_ms", 0)
        e2e_s = r.get("request_latency", 0)  # in seconds
        ttfts.append(ttft)
        tpots.append(tpot)
        e2es.append(e2e_s * 1000)  # convert to ms

    # Aggregate metrics from the benchmark
    # TTFT and TPOT aggregates are already in ms.
    # E2E aggregate (request_latency) is in seconds — must convert to ms.
    metrics = b0.get("metrics", {})
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
        "requests": successful,
    }


def mean(values):
    """Compute mean, returning 0 for empty lists."""
    if not values:
        return 0
    return sum(values) / len(values)


def mape(predicted, actual):
    """Compute Mean Absolute Percentage Error.

    Returns MAPE as a fraction (0.10 = 10%). Returns None if actual is zero.
    """
    if actual == 0:
        return None
    return abs(predicted - actual) / abs(actual)


def signed_pct_error(predicted, actual):
    """Compute signed percentage error (positive = overprediction)."""
    if actual == 0:
        return None
    return (predicted - actual) / abs(actual)


def fmt_pct(value):
    """Format a fraction as percentage string, handling None."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def fmt_delta(baseline_mape, treatment_mape):
    """Format MAPE delta as improvement (positive = better)."""
    if baseline_mape is None or treatment_mape is None:
        return "N/A"
    delta = baseline_mape - treatment_mape
    direction = "+" if delta > 0 else ""
    return f"{direction}{delta * 100:.1f}pp"


def main():
    if len(sys.argv) < 3:
        print("Usage: analyze.py <results_dir> <ground_truth_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])

    # Collect per-experiment results
    rows = []
    for exp_name in EXPERIMENTS:
        baseline_path = results_dir / f"{exp_name}_baseline.json"
        treatment_path = results_dir / f"{exp_name}_treatment.json"
        gt_path = gt_dir / exp_name / "guidellm-results.json"

        baseline = load_blis_results(baseline_path)
        treatment = load_blis_results(treatment_path)
        gt = load_guidellm_ground_truth(gt_path)

        if gt is None:
            print(f"SKIP: no ground truth for {exp_name}", file=sys.stderr)
            continue

        wtype = workload_type(exp_name)

        row = {
            "experiment": exp_name,
            "workload_type": wtype,
            "gt_ttft": gt["ttft_mean_ms"],
            "gt_tpot": gt["tpot_mean_ms"],
            "gt_e2e": gt["e2e_mean_ms"],
        }

        # Baseline MAPE
        if baseline is not None:
            row["bl_ttft"] = baseline["ttft_mean_ms"]
            row["bl_tpot"] = baseline["tpot_mean_ms"]
            row["bl_e2e"] = baseline["e2e_mean_ms"]
            row["bl_ttft_mape"] = mape(baseline["ttft_mean_ms"], gt["ttft_mean_ms"])
            row["bl_tpot_mape"] = mape(baseline["tpot_mean_ms"], gt["tpot_mean_ms"])
            row["bl_e2e_mape"] = mape(baseline["e2e_mean_ms"], gt["e2e_mean_ms"])
        else:
            row["bl_ttft"] = row["bl_tpot"] = row["bl_e2e"] = None
            row["bl_ttft_mape"] = row["bl_tpot_mape"] = row["bl_e2e_mape"] = None

        # Treatment MAPE
        if treatment is not None:
            row["tr_ttft"] = treatment["ttft_mean_ms"]
            row["tr_tpot"] = treatment["tpot_mean_ms"]
            row["tr_e2e"] = treatment["e2e_mean_ms"]
            row["tr_ttft_mape"] = mape(treatment["ttft_mean_ms"], gt["ttft_mean_ms"])
            row["tr_tpot_mape"] = mape(treatment["tpot_mean_ms"], gt["tpot_mean_ms"])
            row["tr_e2e_mape"] = mape(treatment["e2e_mean_ms"], gt["e2e_mean_ms"])
        else:
            row["tr_ttft"] = row["tr_tpot"] = row["tr_e2e"] = None
            row["tr_ttft_mape"] = row["tr_tpot_mape"] = row["tr_e2e_mape"] = None

        rows.append(row)

    if not rows:
        print("ERROR: no experiments produced results", file=sys.stderr)
        sys.exit(1)

    # --- Print per-experiment comparison table ---
    print("=" * 120)
    print("  H1 Bandwidth Efficiency Validation — Per-Experiment MAPE Comparison")
    print("=" * 120)
    print()

    # Header
    hdr = f"{'Experiment':<45} {'Type':<8} {'BL TTFT':>8} {'TR TTFT':>8} {'Δ':>7}  {'BL TPOT':>8} {'TR TPOT':>8} {'Δ':>7}  {'BL E2E':>8} {'TR E2E':>8} {'Δ':>7}"
    print(hdr)
    print("-" * len(hdr))

    for row in rows:
        ttft_delta = fmt_delta(row["bl_ttft_mape"], row["tr_ttft_mape"])
        tpot_delta = fmt_delta(row["bl_tpot_mape"], row["tr_tpot_mape"])
        e2e_delta = fmt_delta(row["bl_e2e_mape"], row["tr_e2e_mape"])

        print(
            f"{row['experiment']:<45} {row['workload_type']:<8} "
            f"{fmt_pct(row['bl_ttft_mape']):>8} {fmt_pct(row['tr_ttft_mape']):>8} {ttft_delta:>7}  "
            f"{fmt_pct(row['bl_tpot_mape']):>8} {fmt_pct(row['tr_tpot_mape']):>8} {tpot_delta:>7}  "
            f"{fmt_pct(row['bl_e2e_mape']):>8} {fmt_pct(row['tr_e2e_mape']):>8} {e2e_delta:>7}"
        )

    # --- Aggregate by workload type ---
    print()
    print("=" * 80)
    print("  Aggregate MAPE by Workload Type")
    print("=" * 80)
    print()

    type_groups = {}
    for row in rows:
        wtype = row["workload_type"]
        if wtype not in type_groups:
            type_groups[wtype] = []
        type_groups[wtype].append(row)

    hdr2 = f"{'Workload Type':<16} {'N':>3}  {'BL TPOT':>8} {'TR TPOT':>8} {'Δ':>7}  {'BL E2E':>8} {'TR E2E':>8} {'Δ':>7}"
    print(hdr2)
    print("-" * len(hdr2))

    for wtype in ["chatsweep", "codesweep", "summarization"]:
        group = type_groups.get(wtype, [])
        if not group:
            continue

        bl_tpot_mapes = [r["bl_tpot_mape"] for r in group if r["bl_tpot_mape"] is not None]
        tr_tpot_mapes = [r["tr_tpot_mape"] for r in group if r["tr_tpot_mape"] is not None]
        bl_e2e_mapes = [r["bl_e2e_mape"] for r in group if r["bl_e2e_mape"] is not None]
        tr_e2e_mapes = [r["tr_e2e_mape"] for r in group if r["tr_e2e_mape"] is not None]

        avg_bl_tpot = mean(bl_tpot_mapes) if bl_tpot_mapes else None
        avg_tr_tpot = mean(tr_tpot_mapes) if tr_tpot_mapes else None
        avg_bl_e2e = mean(bl_e2e_mapes) if bl_e2e_mapes else None
        avg_tr_e2e = mean(tr_e2e_mapes) if tr_e2e_mapes else None

        tpot_delta = fmt_delta(avg_bl_tpot, avg_tr_tpot)
        e2e_delta = fmt_delta(avg_bl_e2e, avg_tr_e2e)

        print(
            f"{wtype:<16} {len(group):>3}  "
            f"{fmt_pct(avg_bl_tpot):>8} {fmt_pct(avg_tr_tpot):>8} {tpot_delta:>7}  "
            f"{fmt_pct(avg_bl_e2e):>8} {fmt_pct(avg_tr_e2e):>8} {e2e_delta:>7}"
        )

    # --- Overall aggregate ---
    all_bl_tpot = [r["bl_tpot_mape"] for r in rows if r["bl_tpot_mape"] is not None]
    all_tr_tpot = [r["tr_tpot_mape"] for r in rows if r["tr_tpot_mape"] is not None]
    all_bl_e2e = [r["bl_e2e_mape"] for r in rows if r["bl_e2e_mape"] is not None]
    all_tr_e2e = [r["tr_e2e_mape"] for r in rows if r["tr_e2e_mape"] is not None]

    print("-" * len(hdr2))
    overall_bl_tpot = mean(all_bl_tpot) if all_bl_tpot else None
    overall_tr_tpot = mean(all_tr_tpot) if all_tr_tpot else None
    overall_bl_e2e = mean(all_bl_e2e) if all_bl_e2e else None
    overall_tr_e2e = mean(all_tr_e2e) if all_tr_e2e else None

    print(
        f"{'OVERALL':<16} {len(rows):>3}  "
        f"{fmt_pct(overall_bl_tpot):>8} {fmt_pct(overall_tr_tpot):>8} {fmt_delta(overall_bl_tpot, overall_tr_tpot):>7}  "
        f"{fmt_pct(overall_bl_e2e):>8} {fmt_pct(overall_tr_e2e):>8} {fmt_delta(overall_bl_e2e, overall_tr_e2e):>7}"
    )

    # --- Absolute values table ---
    print()
    print("=" * 120)
    print("  Absolute Latency Values (ms) — Ground Truth vs Predicted Means")
    print("=" * 120)
    print()

    hdr3 = f"{'Experiment':<45} {'GT TTFT':>8} {'BL TTFT':>8} {'TR TTFT':>8}  {'GT TPOT':>8} {'BL TPOT':>8} {'TR TPOT':>8}  {'GT E2E':>8} {'BL E2E':>8} {'TR E2E':>8}"
    print(hdr3)
    print("-" * len(hdr3))

    for row in rows:
        def fmt_ms(v):
            return f"{v:.1f}" if v is not None else "N/A"

        print(
            f"{row['experiment']:<45} "
            f"{fmt_ms(row['gt_ttft']):>8} {fmt_ms(row.get('bl_ttft')):>8} {fmt_ms(row.get('tr_ttft')):>8}  "
            f"{fmt_ms(row['gt_tpot']):>8} {fmt_ms(row.get('bl_tpot')):>8} {fmt_ms(row.get('tr_tpot')):>8}  "
            f"{fmt_ms(row['gt_e2e']):>8} {fmt_ms(row.get('bl_e2e')):>8} {fmt_ms(row.get('tr_e2e')):>8}"
        )

    # --- Accept criteria check ---
    print()
    print("=" * 80)
    print("  Accept Criteria Evaluation")
    print("=" * 80)
    print()

    # Criterion 1: TPOT MAPE improves by >=3pp overall
    if overall_bl_tpot is not None and overall_tr_tpot is not None:
        tpot_improvement = overall_bl_tpot - overall_tr_tpot
        criterion1 = tpot_improvement >= 0.03
        print(f"  1. TPOT MAPE improvement >= 3pp: {fmt_delta(overall_bl_tpot, overall_tr_tpot)}")
        print(f"     {'PASS' if criterion1 else 'FAIL'}: {'%.1f' % (tpot_improvement * 100)}pp improvement")
    else:
        print("  1. TPOT MAPE improvement >= 3pp: UNABLE TO EVALUATE (missing data)")

    # Criterion 2: Chatsweep (decode-heavy, output=215) improves more than
    # codesweep (prefill-heavy, output=28) — directional evidence that
    # bandwidth correction benefits memory-bound decode steps most.
    code_group = type_groups.get("codesweep", [])
    chat_group = type_groups.get("chatsweep", [])

    code_bl_vals = [r["bl_tpot_mape"] for r in code_group if r["bl_tpot_mape"] is not None]
    code_tr_vals = [r["tr_tpot_mape"] for r in code_group if r["tr_tpot_mape"] is not None]
    chat_bl_vals = [r["bl_tpot_mape"] for r in chat_group if r["bl_tpot_mape"] is not None]
    chat_tr_vals = [r["tr_tpot_mape"] for r in chat_group if r["tr_tpot_mape"] is not None]

    code_bl = mean(code_bl_vals) if code_bl_vals else None
    code_tr = mean(code_tr_vals) if code_tr_vals else None
    chat_bl = mean(chat_bl_vals) if chat_bl_vals else None
    chat_tr = mean(chat_tr_vals) if chat_tr_vals else None

    if all(v is not None for v in [code_bl, code_tr, chat_bl, chat_tr]):
        code_improvement = code_bl - code_tr
        chat_improvement = chat_bl - chat_tr
        criterion2 = chat_improvement > code_improvement
        print(f"  2. Chatsweep (decode-heavy) improves more than codesweep (prefill-heavy): "
              f"chat={chat_improvement * 100:.1f}pp vs code={code_improvement * 100:.1f}pp")
        print(f"     {'PASS' if criterion2 else 'FAIL'}: "
              f"{'chatsweep' if criterion2 else 'codesweep'} shows larger improvement")
    else:
        print("  2. Chatsweep (decode-heavy) improves more than codesweep (prefill-heavy): "
              "UNABLE TO EVALUATE (missing data)")

    print()


if __name__ == "__main__":
    main()
