#!/usr/bin/env python3
"""H2b Model-Scaled Scheduling Overhead — Analysis.

Compares BLIS roofline v2 predictions in three configurations:
  Baseline:     H1 BW correction only (bwEfficiencyFactor=0.82)
  Fixed (H2):   H1 + fixed InferSim overheads (5ms decode, 30ms prefill)
  Scaled (H2b): H1 + model-scaled overheads (100μs/layer/tp decode, 500μs/layer/tp prefill)

Usage:
    python3 analyze.py <results_dir> <ground_truth_dir>
"""

import json
import sys
from pathlib import Path


# experiment_name | num_hidden_layers | tp
EXPERIMENTS = [
    ("jan30-llama2-7b-tp1-chatsweep", 32, 1),
    ("jan30-llama2-7b-tp1-codesweep", 32, 1),
    ("jan30-llama2-7b-tp2-chatsweep", 32, 2),
    ("jan30-llama2-7b-tp2-codesweep", 32, 2),
    ("jan30-llama2-7b-tp4-chatsweep", 32, 4),
    ("jan30-llama2-7b-tp4-codesweep", 32, 4),
    ("20260210-codellama-34b-tp2-chatsweep", 48, 2),
    ("20260210-codellama-34b-tp2-codesweep", 48, 2),
    ("20260210-llama2-70b-tp4-chatsweep", 80, 4),
    ("20260210-llama2-70b-tp4-codesweep", 80, 4),
    ("20260210-qwen3-14b-tp1-codesweep", 40, 1),
    ("20260210-qwen3-14b-tp2-chatsweep", 40, 2),
    ("dec17-tp1-qwen7-summarization", 28, 1),
]

BASE_DECODE_US = 100
BASE_PREFILL_US = 500


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
    for exp_name, num_layers, tp in EXPERIMENTS:
        baseline_path = results_dir / f"{exp_name}_baseline.json"
        fixed_path = results_dir / f"{exp_name}_fixed.json"
        scaled_path = results_dir / f"{exp_name}_scaled.json"
        gt_path = gt_dir / exp_name / "guidellm-results.json"

        baseline = load_blis_results(baseline_path)
        fixed = load_blis_results(fixed_path)
        scaled = load_blis_results(scaled_path)
        gt = load_guidellm_ground_truth(gt_path)

        if gt is None:
            print(f"SKIP: no ground truth for {exp_name}", file=sys.stderr)
            continue

        wtype = workload_type(exp_name)
        mfamily = model_family(exp_name)

        decode_overhead_us = BASE_DECODE_US * num_layers / tp
        prefill_overhead_us = BASE_PREFILL_US * num_layers / tp

        row = {
            "experiment": exp_name,
            "workload_type": wtype,
            "model_family": mfamily,
            "num_layers": num_layers,
            "tp": tp,
            "decode_overhead_us": decode_overhead_us,
            "prefill_overhead_us": prefill_overhead_us,
            "gt_ttft": gt["ttft_mean_ms"],
            "gt_tpot": gt["tpot_mean_ms"],
            "gt_e2e": gt["e2e_mean_ms"],
        }

        for prefix, data in [("bl", baseline), ("fx", fixed), ("sc", scaled)]:
            if data is not None:
                row[f"{prefix}_ttft"] = data["ttft_mean_ms"]
                row[f"{prefix}_tpot"] = data["tpot_mean_ms"]
                row[f"{prefix}_e2e"] = data["e2e_mean_ms"]
                row[f"{prefix}_ttft_mape"] = mape(data["ttft_mean_ms"], gt["ttft_mean_ms"])
                row[f"{prefix}_tpot_mape"] = mape(data["tpot_mean_ms"], gt["tpot_mean_ms"])
                row[f"{prefix}_e2e_mape"] = mape(data["e2e_mean_ms"], gt["e2e_mean_ms"])
                row[f"{prefix}_tpot_signed"] = signed_pct_error(data["tpot_mean_ms"], gt["tpot_mean_ms"])
            else:
                for metric in ["ttft", "tpot", "e2e"]:
                    row[f"{prefix}_{metric}"] = None
                    row[f"{prefix}_{metric}_mape"] = None
                row[f"{prefix}_tpot_signed"] = None

        rows.append(row)

    if not rows:
        print("ERROR: no experiments produced results", file=sys.stderr)
        sys.exit(1)

    # === Per-experiment overhead values ===
    print("=" * 90)
    print("  H2b Computed Overheads (base_decode=100μs/layer, base_prefill=500μs/layer)")
    print("=" * 90)
    print()
    hdr0 = f"{'Experiment':<45} {'Layers':>6} {'TP':>3} {'Decode OH':>10} {'Prefill OH':>11}"
    print(hdr0)
    print("-" * len(hdr0))
    for row in rows:
        print(
            f"{row['experiment']:<45} "
            f"{row['num_layers']:>6} {row['tp']:>3} "
            f"{row['decode_overhead_us']:>8.0f}μs "
            f"{row['prefill_overhead_us']:>9.0f}μs"
        )

    # === Per-experiment MAPE comparison (3-way) ===
    print()
    print("=" * 140)
    print("  H2b — Per-Experiment TPOT MAPE (Baseline vs Fixed vs Scaled)")
    print("  BL=H1 only | FX=H1+fixed(5ms) | SC=H1+scaled(100μs/layer/tp)")
    print("=" * 140)
    print()

    hdr = (f"{'Experiment':<45} {'Type':<8} "
           f"{'BL TPOT':>8} {'FX TPOT':>8} {'SC TPOT':>8}  "
           f"{'BL→SC':>7} {'FX→SC':>7}  "
           f"{'BL E2E':>8} {'FX E2E':>8} {'SC E2E':>8}  "
           f"{'BL→SC':>7}")
    print(hdr)
    print("-" * len(hdr))

    for row in rows:
        bl_sc_tpot = fmt_delta(row.get("bl_tpot_mape"), row.get("sc_tpot_mape"))
        fx_sc_tpot = fmt_delta(row.get("fx_tpot_mape"), row.get("sc_tpot_mape"))
        bl_sc_e2e = fmt_delta(row.get("bl_e2e_mape"), row.get("sc_e2e_mape"))
        print(
            f"{row['experiment']:<45} {row['workload_type']:<8} "
            f"{fmt_pct(row.get('bl_tpot_mape')):>8} "
            f"{fmt_pct(row.get('fx_tpot_mape')):>8} "
            f"{fmt_pct(row.get('sc_tpot_mape')):>8}  "
            f"{bl_sc_tpot:>7} {fx_sc_tpot:>7}  "
            f"{fmt_pct(row.get('bl_e2e_mape')):>8} "
            f"{fmt_pct(row.get('fx_e2e_mape')):>8} "
            f"{fmt_pct(row.get('sc_e2e_mape')):>8}  "
            f"{bl_sc_e2e:>7}"
        )

    # === Signed error (bias direction) ===
    print()
    print("=" * 90)
    print("  Signed TPOT Error (positive = overprediction, negative = underprediction)")
    print("=" * 90)
    print()

    hdr4 = f"{'Experiment':<45} {'BL TPOT':>10} {'FX TPOT':>10} {'SC TPOT':>10}"
    print(hdr4)
    print("-" * len(hdr4))

    for row in rows:
        print(
            f"{row['experiment']:<45} "
            f"{fmt_pct(row.get('bl_tpot_signed')):>10} "
            f"{fmt_pct(row.get('fx_tpot_signed')):>10} "
            f"{fmt_pct(row.get('sc_tpot_signed')):>10}"
        )

    # === Aggregate by model family ===
    print()
    print("=" * 100)
    print("  Aggregate TPOT MAPE by Model Family")
    print("=" * 100)
    print()

    family_groups = {}
    for row in rows:
        mf = row["model_family"]
        if mf not in family_groups:
            family_groups[mf] = []
        family_groups[mf].append(row)

    hdr5 = (f"{'Model Family':<16} {'N':>3} {'Layers':>6} {'TP':>3}  "
            f"{'BL TPOT':>8} {'FX TPOT':>8} {'SC TPOT':>8}  "
            f"{'BL→SC':>7} {'FX→SC':>7}")
    print(hdr5)
    print("-" * len(hdr5))

    for mf in ["llama2-7b", "qwen2.5-7b", "qwen3-14b", "codellama-34b", "llama2-70b"]:
        group = family_groups.get(mf, [])
        if not group:
            continue
        bl = [r["bl_tpot_mape"] for r in group if r.get("bl_tpot_mape") is not None]
        fx = [r["fx_tpot_mape"] for r in group if r.get("fx_tpot_mape") is not None]
        sc = [r["sc_tpot_mape"] for r in group if r.get("sc_tpot_mape") is not None]
        avg_bl = mean(bl) if bl else None
        avg_fx = mean(fx) if fx else None
        avg_sc = mean(sc) if sc else None
        layers = group[0]["num_layers"]
        tp_str = ",".join(sorted(set(str(r["tp"]) for r in group)))
        print(
            f"{mf:<16} {len(group):>3} {layers:>6} {tp_str:>3}  "
            f"{fmt_pct(avg_bl):>8} {fmt_pct(avg_fx):>8} {fmt_pct(avg_sc):>8}  "
            f"{fmt_delta(avg_bl, avg_sc):>7} {fmt_delta(avg_fx, avg_sc):>7}"
        )

    # === Overall aggregates ===
    all_bl_tpot = [r["bl_tpot_mape"] for r in rows if r.get("bl_tpot_mape") is not None]
    all_fx_tpot = [r["fx_tpot_mape"] for r in rows if r.get("fx_tpot_mape") is not None]
    all_sc_tpot = [r["sc_tpot_mape"] for r in rows if r.get("sc_tpot_mape") is not None]
    all_bl_e2e = [r["bl_e2e_mape"] for r in rows if r.get("bl_e2e_mape") is not None]
    all_fx_e2e = [r["fx_e2e_mape"] for r in rows if r.get("fx_e2e_mape") is not None]
    all_sc_e2e = [r["sc_e2e_mape"] for r in rows if r.get("sc_e2e_mape") is not None]

    overall_bl_tpot = mean(all_bl_tpot) if all_bl_tpot else None
    overall_fx_tpot = mean(all_fx_tpot) if all_fx_tpot else None
    overall_sc_tpot = mean(all_sc_tpot) if all_sc_tpot else None
    overall_bl_e2e = mean(all_bl_e2e) if all_bl_e2e else None
    overall_fx_e2e = mean(all_fx_e2e) if all_fx_e2e else None
    overall_sc_e2e = mean(all_sc_e2e) if all_sc_e2e else None

    print()
    print("=" * 80)
    print("  Overall Aggregates")
    print("=" * 80)
    print()
    print(f"  TPOT MAPE:  Baseline={fmt_pct(overall_bl_tpot)}  Fixed(H2)={fmt_pct(overall_fx_tpot)}  Scaled(H2b)={fmt_pct(overall_sc_tpot)}")
    print(f"  E2E MAPE:   Baseline={fmt_pct(overall_bl_e2e)}  Fixed(H2)={fmt_pct(overall_fx_e2e)}  Scaled(H2b)={fmt_pct(overall_sc_e2e)}")
    print(f"  TPOT: BL→SC {fmt_delta(overall_bl_tpot, overall_sc_tpot)}, FX→SC {fmt_delta(overall_fx_tpot, overall_sc_tpot)}")
    print(f"  E2E:  BL→SC {fmt_delta(overall_bl_e2e, overall_sc_e2e)}, FX→SC {fmt_delta(overall_fx_e2e, overall_sc_e2e)}")

    # === Accept criteria ===
    print()
    print("=" * 80)
    print("  Accept Criteria Evaluation")
    print("=" * 80)
    print()

    # Criterion 1: Scaled overhead improves TPOT MAPE by >=3pp vs H1-only baseline
    if overall_bl_tpot is not None and overall_sc_tpot is not None:
        tpot_improvement = overall_bl_tpot - overall_sc_tpot
        criterion1 = tpot_improvement >= 0.03
        print(f"  1. TPOT MAPE improvement >= 3pp vs baseline: {fmt_delta(overall_bl_tpot, overall_sc_tpot)}")
        print(f"     {'PASS' if criterion1 else 'FAIL'}: {tpot_improvement * 100:.1f}pp improvement")
    else:
        criterion1 = False
        print("  1. TPOT MAPE improvement >= 3pp vs baseline: UNABLE TO EVALUATE")

    # Criterion 2: No single experiment worsens by >5pp vs baseline
    worst_worsening = 0
    worst_exp = ""
    for row in rows:
        bl_m = row.get("bl_tpot_mape")
        sc_m = row.get("sc_tpot_mape")
        if bl_m is not None and sc_m is not None:
            worsening = sc_m - bl_m  # positive = scaled is worse
            if worsening > worst_worsening:
                worst_worsening = worsening
                worst_exp = row["experiment"]

    criterion2 = worst_worsening <= 0.05
    if worst_worsening > 0:
        print(f"  2. No experiment worsens by >5pp: worst = {worst_exp} ({worst_worsening * 100:.1f}pp)")
    else:
        print(f"  2. No experiment worsens by >5pp: no experiment worsened")
    print(f"     {'PASS' if criterion2 else 'FAIL'}")

    # Criterion 3: Scaled overhead outperforms fixed-5ms on aggregate TPOT MAPE
    if overall_fx_tpot is not None and overall_sc_tpot is not None:
        criterion3 = overall_sc_tpot < overall_fx_tpot
        print(f"  3. Scaled TPOT MAPE < Fixed TPOT MAPE: {fmt_pct(overall_sc_tpot)} vs {fmt_pct(overall_fx_tpot)}")
        print(f"     {'PASS' if criterion3 else 'FAIL'}: scaled is {'better' if criterion3 else 'worse'} by {abs(overall_fx_tpot - overall_sc_tpot) * 100:.1f}pp")
    else:
        criterion3 = False
        print("  3. Scaled TPOT MAPE < Fixed TPOT MAPE: UNABLE TO EVALUATE")

    print()
    all_pass = criterion1 and criterion2 and criterion3
    print(f"  Overall: {'ALL CRITERIA PASS' if all_pass else 'ONE OR MORE CRITERIA FAIL'}")
    print()


if __name__ == "__main__":
    main()
