#!/usr/bin/env python3
"""H2 Scheduling Overhead Validation — Analysis.

Compares BLIS roofline v2 predictions:
  Baseline: H1 BW correction only (bwEfficiencyFactor=0.82)
  Treatment: H1 + InferSim-scale overheads (5ms decode, 30ms prefill)

Usage:
    python3 analyze.py <results_dir> <ground_truth_dir>
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
    if "codesweep" in exp_name:
        return "codesweep"
    elif "chatsweep" in exp_name:
        return "chatsweep"
    elif "summarization" in exp_name:
        return "summarization"
    return "unknown"


def model_family(exp_name):
    if "llama2-7b" in exp_name:
        return "llama2-7b"
    elif "codellama-34b" in exp_name:
        return "codellama-34b"
    elif "llama2-70b" in exp_name:
        return "llama2-70b"
    elif "qwen3-14b" in exp_name:
        return "qwen3-14b"
    elif "qwen7" in exp_name or "qwen2.5-7b" in exp_name:
        return "qwen2.5-7b"
    return "unknown"


def load_blis_results(filepath):
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


def load_guidellm_ground_truth(filepath):
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

    ttfts, tpots, e2es = [], [], []
    for r in successful:
        ttfts.append(r.get("time_to_first_token_ms", 0))
        tpots.append(r.get("time_per_output_token_ms", 0))
        e2es.append(r.get("request_latency", 0) * 1000)

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
    if not values:
        return 0
    return sum(values) / len(values)


def mape(predicted, actual):
    if actual == 0:
        return None
    return abs(predicted - actual) / abs(actual)


def signed_pct_error(predicted, actual):
    if actual == 0:
        return None
    return (predicted - actual) / abs(actual)


def fmt_pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def fmt_delta(baseline_mape, treatment_mape):
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
        mfamily = model_family(exp_name)

        row = {
            "experiment": exp_name,
            "workload_type": wtype,
            "model_family": mfamily,
            "gt_ttft": gt["ttft_mean_ms"],
            "gt_tpot": gt["tpot_mean_ms"],
            "gt_e2e": gt["e2e_mean_ms"],
        }

        for prefix, data in [("bl", baseline), ("tr", treatment)]:
            if data is not None:
                row[f"{prefix}_ttft"] = data["ttft_mean_ms"]
                row[f"{prefix}_tpot"] = data["tpot_mean_ms"]
                row[f"{prefix}_e2e"] = data["e2e_mean_ms"]
                row[f"{prefix}_ttft_mape"] = mape(data["ttft_mean_ms"], gt["ttft_mean_ms"])
                row[f"{prefix}_tpot_mape"] = mape(data["tpot_mean_ms"], gt["tpot_mean_ms"])
                row[f"{prefix}_e2e_mape"] = mape(data["e2e_mean_ms"], gt["e2e_mean_ms"])
                row[f"{prefix}_tpot_signed"] = signed_pct_error(data["tpot_mean_ms"], gt["tpot_mean_ms"])
                row[f"{prefix}_ttft_signed"] = signed_pct_error(data["ttft_mean_ms"], gt["ttft_mean_ms"])
            else:
                for metric in ["ttft", "tpot", "e2e"]:
                    row[f"{prefix}_{metric}"] = None
                    row[f"{prefix}_{metric}_mape"] = None
                row[f"{prefix}_tpot_signed"] = None
                row[f"{prefix}_ttft_signed"] = None

        rows.append(row)

    if not rows:
        print("ERROR: no experiments produced results", file=sys.stderr)
        sys.exit(1)

    # --- Per-experiment MAPE comparison ---
    print("=" * 130)
    print("  H2 Scheduling Overhead — Per-Experiment MAPE Comparison")
    print("  Baseline=H1 only | Treatment=H1+overhead (decode=5ms, prefill=30ms)")
    print("=" * 130)
    print()

    hdr = (f"{'Experiment':<45} {'Type':<8} "
           f"{'BL TTFT':>8} {'TR TTFT':>8} {'Δ':>7}  "
           f"{'BL TPOT':>8} {'TR TPOT':>8} {'Δ':>7}  "
           f"{'BL E2E':>8} {'TR E2E':>8} {'Δ':>7}")
    print(hdr)
    print("-" * len(hdr))

    for row in rows:
        print(
            f"{row['experiment']:<45} {row['workload_type']:<8} "
            f"{fmt_pct(row['bl_ttft_mape']):>8} {fmt_pct(row['tr_ttft_mape']):>8} "
            f"{fmt_delta(row['bl_ttft_mape'], row['tr_ttft_mape']):>7}  "
            f"{fmt_pct(row['bl_tpot_mape']):>8} {fmt_pct(row['tr_tpot_mape']):>8} "
            f"{fmt_delta(row['bl_tpot_mape'], row['tr_tpot_mape']):>7}  "
            f"{fmt_pct(row['bl_e2e_mape']):>8} {fmt_pct(row['tr_e2e_mape']):>8} "
            f"{fmt_delta(row['bl_e2e_mape'], row['tr_e2e_mape']):>7}"
        )

    # --- Absolute values ---
    print()
    print("=" * 130)
    print("  Absolute Latency Values (ms) — Ground Truth vs Predicted")
    print("=" * 130)
    print()

    hdr3 = (f"{'Experiment':<45} "
            f"{'GT TTFT':>8} {'BL TTFT':>8} {'TR TTFT':>8}  "
            f"{'GT TPOT':>8} {'BL TPOT':>8} {'TR TPOT':>8}  "
            f"{'GT E2E':>8} {'BL E2E':>8} {'TR E2E':>8}")
    print(hdr3)
    print("-" * len(hdr3))

    for row in rows:
        def fmt_ms(v):
            return f"{v:.1f}" if v is not None else "N/A"
        print(
            f"{row['experiment']:<45} "
            f"{fmt_ms(row['gt_ttft']):>8} {fmt_ms(row.get('bl_ttft')):>8} "
            f"{fmt_ms(row.get('tr_ttft')):>8}  "
            f"{fmt_ms(row['gt_tpot']):>8} {fmt_ms(row.get('bl_tpot')):>8} "
            f"{fmt_ms(row.get('tr_tpot')):>8}  "
            f"{fmt_ms(row['gt_e2e']):>8} {fmt_ms(row.get('bl_e2e')):>8} "
            f"{fmt_ms(row.get('tr_e2e')):>8}"
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

    hdr2 = (f"{'Workload Type':<16} {'N':>3}  "
            f"{'BL TPOT':>8} {'TR TPOT':>8} {'Δ':>7}  "
            f"{'BL E2E':>8} {'TR E2E':>8} {'Δ':>7}")
    print(hdr2)
    print("-" * len(hdr2))

    for wtype in ["chatsweep", "codesweep", "summarization"]:
        group = type_groups.get(wtype, [])
        if not group:
            continue
        bl_tpot = [r["bl_tpot_mape"] for r in group if r["bl_tpot_mape"] is not None]
        tr_tpot = [r["tr_tpot_mape"] for r in group if r["tr_tpot_mape"] is not None]
        bl_e2e = [r["bl_e2e_mape"] for r in group if r["bl_e2e_mape"] is not None]
        tr_e2e = [r["tr_e2e_mape"] for r in group if r["tr_e2e_mape"] is not None]

        avg_bl_tpot = mean(bl_tpot) if bl_tpot else None
        avg_tr_tpot = mean(tr_tpot) if tr_tpot else None
        avg_bl_e2e = mean(bl_e2e) if bl_e2e else None
        avg_tr_e2e = mean(tr_e2e) if tr_e2e else None

        print(
            f"{wtype:<16} {len(group):>3}  "
            f"{fmt_pct(avg_bl_tpot):>8} {fmt_pct(avg_tr_tpot):>8} "
            f"{fmt_delta(avg_bl_tpot, avg_tr_tpot):>7}  "
            f"{fmt_pct(avg_bl_e2e):>8} {fmt_pct(avg_tr_e2e):>8} "
            f"{fmt_delta(avg_bl_e2e, avg_tr_e2e):>7}"
        )

    # Overall
    all_bl_tpot = [r["bl_tpot_mape"] for r in rows if r["bl_tpot_mape"] is not None]
    all_tr_tpot = [r["tr_tpot_mape"] for r in rows if r["tr_tpot_mape"] is not None]
    all_bl_e2e = [r["bl_e2e_mape"] for r in rows if r["bl_e2e_mape"] is not None]
    all_tr_e2e = [r["tr_e2e_mape"] for r in rows if r["tr_e2e_mape"] is not None]
    all_bl_ttft = [r["bl_ttft_mape"] for r in rows if r["bl_ttft_mape"] is not None]
    all_tr_ttft = [r["tr_ttft_mape"] for r in rows if r["tr_ttft_mape"] is not None]

    overall_bl_tpot = mean(all_bl_tpot) if all_bl_tpot else None
    overall_tr_tpot = mean(all_tr_tpot) if all_tr_tpot else None
    overall_bl_e2e = mean(all_bl_e2e) if all_bl_e2e else None
    overall_tr_e2e = mean(all_tr_e2e) if all_tr_e2e else None
    overall_bl_ttft = mean(all_bl_ttft) if all_bl_ttft else None
    overall_tr_ttft = mean(all_tr_ttft) if all_tr_ttft else None

    print("-" * len(hdr2))
    print(
        f"{'OVERALL':<16} {len(rows):>3}  "
        f"{fmt_pct(overall_bl_tpot):>8} {fmt_pct(overall_tr_tpot):>8} "
        f"{fmt_delta(overall_bl_tpot, overall_tr_tpot):>7}  "
        f"{fmt_pct(overall_bl_e2e):>8} {fmt_pct(overall_tr_e2e):>8} "
        f"{fmt_delta(overall_bl_e2e, overall_tr_e2e):>7}"
    )

    # --- Signed error (bias direction) ---
    print()
    print("=" * 80)
    print("  Signed Error (positive = overprediction, negative = underprediction)")
    print("=" * 80)
    print()

    hdr4 = f"{'Experiment':<45} {'BL TPOT':>10} {'TR TPOT':>10}  {'BL TTFT':>10} {'TR TTFT':>10}"
    print(hdr4)
    print("-" * len(hdr4))

    for row in rows:
        print(
            f"{row['experiment']:<45} "
            f"{fmt_pct(row.get('bl_tpot_signed')):>10} "
            f"{fmt_pct(row.get('tr_tpot_signed')):>10}  "
            f"{fmt_pct(row.get('bl_ttft_signed')):>10} "
            f"{fmt_pct(row.get('tr_ttft_signed')):>10}"
        )

    # --- Accept criteria ---
    print()
    print("=" * 80)
    print("  Accept Criteria Evaluation")
    print("=" * 80)
    print()

    # Criterion 1: TPOT MAPE improves by >=3pp
    if overall_bl_tpot is not None and overall_tr_tpot is not None:
        tpot_improvement = overall_bl_tpot - overall_tr_tpot
        criterion1 = tpot_improvement >= 0.03
        print(f"  1. TPOT MAPE improvement >= 3pp: {fmt_delta(overall_bl_tpot, overall_tr_tpot)}")
        print(f"     {'PASS' if criterion1 else 'FAIL'}: {tpot_improvement * 100:.1f}pp improvement")
    else:
        print("  1. TPOT MAPE improvement >= 3pp: UNABLE TO EVALUATE")

    # Criterion 2: TTFT MAPE does not worsen by >1pp
    if overall_bl_ttft is not None and overall_tr_ttft is not None:
        ttft_delta = overall_tr_ttft - overall_bl_ttft  # positive = worsened
        criterion2 = ttft_delta <= 0.01
        direction = "worsened" if ttft_delta > 0 else "improved"
        print(f"  2. TTFT MAPE does not worsen by >1pp: "
              f"{direction} by {abs(ttft_delta) * 100:.1f}pp")
        print(f"     {'PASS' if criterion2 else 'FAIL'}")
    else:
        print("  2. TTFT MAPE does not worsen by >1pp: UNABLE TO EVALUATE")

    # --- Cumulative: H1+H2 vs raw baseline ---
    print()
    print("=" * 80)
    print("  Cumulative Progress: raw peak BW → H1 → H1+H2")
    print("=" * 80)
    print()
    print("  (H1-only numbers from H1 experiment for reference)")
    print(f"  Raw baseline TPOT MAPE:     ~47.7% (from H1 FINDINGS.md)")
    print(f"  After H1 (BW correction):   {fmt_pct(overall_bl_tpot)} (this experiment's baseline)")
    print(f"  After H1+H2 (+ overhead):   {fmt_pct(overall_tr_tpot)}")
    print()


if __name__ == "__main__":
    main()
