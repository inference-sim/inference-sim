#!/usr/bin/env python3
"""HG1: perLayerOverhead Grid Search — Analysis.

Supports three phases:
  --phase coarse: Find coarse optimum from 0-500μs sweep (prints optimum to stdout)
  --phase fine:   Find fine optimum from ±25μs sweep (prints optimum to stdout)
  --phase final:  Full analysis with train/test comparison tables

Usage:
    # Coarse phase: returns optimum value
    python3 analyze.py --phase coarse --results-dir DIR --gt-dir DIR --step 25 --start 0 --end 500

    # Fine phase: returns optimum value
    python3 analyze.py --phase fine --results-dir DIR --gt-dir DIR --step 5 --start 75 --end 125

    # Final analysis
    python3 analyze.py --phase final --results-dir DIR --gt-dir DIR --optimum 85 --coarse-optimum 100
"""

import argparse
import json
import sys
from pathlib import Path


# --- Train/Test split (matches run.sh) ---
TRAIN_EXPERIMENTS = [
    "jan30-llama2-7b-tp1-chatsweep",
    "jan30-llama2-7b-tp2-codesweep",
    "jan30-llama2-7b-tp4-chatsweep",
    "20260210-codellama-34b-tp2-chatsweep",
    "20260210-codellama-34b-tp2-codesweep",
    "20260210-llama2-70b-tp4-chatsweep",
    "20260210-qwen3-14b-tp1-codesweep",
    "20260210-qwen3-14b-tp2-chatsweep",
    "dec17-tp1-qwen7-summarization",
]

TEST_EXPERIMENTS = [
    "jan30-llama2-7b-tp1-codesweep",
    "jan30-llama2-7b-tp2-chatsweep",
    "jan30-llama2-7b-tp4-codesweep",
    "20260210-llama2-70b-tp4-codesweep",
]

ALL_EXPERIMENTS = TRAIN_EXPERIMENTS + TEST_EXPERIMENTS

# Model family metadata for per-family analysis
EXPERIMENT_META = {
    "jan30-llama2-7b-tp1-chatsweep":          {"family": "llama2-7b", "tp": 1, "workload": "chatsweep", "layers": 32},
    "jan30-llama2-7b-tp1-codesweep":          {"family": "llama2-7b", "tp": 1, "workload": "codesweep", "layers": 32},
    "jan30-llama2-7b-tp2-chatsweep":          {"family": "llama2-7b", "tp": 2, "workload": "chatsweep", "layers": 32},
    "jan30-llama2-7b-tp2-codesweep":          {"family": "llama2-7b", "tp": 2, "workload": "codesweep", "layers": 32},
    "jan30-llama2-7b-tp4-chatsweep":          {"family": "llama2-7b", "tp": 4, "workload": "chatsweep", "layers": 32},
    "jan30-llama2-7b-tp4-codesweep":          {"family": "llama2-7b", "tp": 4, "workload": "codesweep", "layers": 32},
    "20260210-codellama-34b-tp2-chatsweep":   {"family": "codellama-34b", "tp": 2, "workload": "chatsweep", "layers": 48},
    "20260210-codellama-34b-tp2-codesweep":   {"family": "codellama-34b", "tp": 2, "workload": "codesweep", "layers": 48},
    "20260210-llama2-70b-tp4-chatsweep":      {"family": "llama2-70b", "tp": 4, "workload": "chatsweep", "layers": 80},
    "20260210-llama2-70b-tp4-codesweep":      {"family": "llama2-70b", "tp": 4, "workload": "codesweep", "layers": 80},
    "20260210-qwen3-14b-tp1-codesweep":       {"family": "qwen3-14b", "tp": 1, "workload": "codesweep", "layers": 40},
    "20260210-qwen3-14b-tp2-chatsweep":       {"family": "qwen3-14b", "tp": 2, "workload": "chatsweep", "layers": 40},
    "dec17-tp1-qwen7-summarization":          {"family": "qwen2.5-7b", "tp": 1, "workload": "summarization", "layers": 28},
}


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def load_blis_results(filepath):
    path = Path(filepath)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    requests = data.get("requests", [])
    if not requests:
        return {
            "tpot_mean_ms": data.get("itl_mean_ms", 0),
            "e2e_mean_ms": data.get("e2e_mean_ms", 0),
            "ttft_mean_ms": data.get("ttft_mean_ms", 0),
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
        "ttft_mean_ms": mean(ttfts) or 0,
        "tpot_mean_ms": mean(tpots) or 0,
        "e2e_mean_ms": mean(e2es) or 0,
    }


def load_ground_truth(gt_dir, exp_name):
    gt_path = Path(gt_dir) / exp_name / "guidellm-results.json"
    if not gt_path.exists():
        return None
    try:
        with open(gt_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
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

    return {
        "ttft_mean_ms": ttft_agg.get("mean") if ttft_agg else (mean(ttfts) or 0),
        "tpot_mean_ms": tpot_agg.get("mean") if tpot_agg else (mean(tpots) or 0),
        "e2e_mean_ms": (e2e_agg.get("mean") * 1000) if e2e_agg and e2e_agg.get("mean") else (mean(e2es) or 0),
    }


def mape(predicted, actual):
    if actual == 0:
        return None
    return abs(predicted - actual) / abs(actual)


def compute_train_tpot_mape(results_dir, gt_dir, overhead, tag_prefix):
    """Compute mean TPOT MAPE across train experiments for a given overhead value."""
    mapes = []
    for exp_name in TRAIN_EXPERIMENTS:
        tag = f"{tag_prefix}_{overhead}"
        blis = load_blis_results(Path(results_dir) / f"{exp_name}_{tag}.json")
        gt = load_ground_truth(gt_dir, exp_name)
        if blis is None or gt is None:
            continue
        m = mape(blis["tpot_mean_ms"], gt["tpot_mean_ms"])
        if m is not None:
            mapes.append(m)
    return mean(mapes) if mapes else None


def find_optimum(results_dir, gt_dir, start, end, step, tag_prefix):
    """Sweep overhead values and return the one with lowest train TPOT MAPE."""
    best_overhead = start
    best_mape = float("inf")
    sweep_data = []

    for overhead in range(start, end + 1, step):
        m = compute_train_tpot_mape(results_dir, gt_dir, overhead, tag_prefix)
        if m is not None:
            sweep_data.append((overhead, m))
            if m < best_mape:
                best_mape = m
                best_overhead = overhead

    # Print sweep curve to stderr for debugging
    for overhead, m in sweep_data:
        marker = " <<<" if overhead == best_overhead else ""
        print(f"  {overhead:>4}μs: TPOT MAPE = {m*100:.2f}%{marker}", file=sys.stderr)

    return best_overhead


def fmt_pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def fmt_delta(baseline, treatment):
    if baseline is None or treatment is None:
        return "N/A"
    delta = baseline - treatment
    direction = "+" if delta > 0 else ""
    return f"{direction}{delta * 100:.1f}pp"


def final_analysis(results_dir, gt_dir, optimum, coarse_optimum):
    """Full analysis: per-experiment comparison, train/test split, accept criteria."""
    results_dir = Path(results_dir)
    gt_dir_path = Path(gt_dir)

    print("=" * 110)
    print(f"  HG1: perLayerOverhead Grid Search — Final Analysis")
    print(f"  Coarse optimum: {coarse_optimum}μs | Fine optimum: {optimum}μs")
    print("=" * 110)
    print()

    # Collect all results
    rows = []
    for exp_name in ALL_EXPERIMENTS:
        gt = load_ground_truth(gt_dir, exp_name)
        if gt is None:
            print(f"SKIP: no ground truth for {exp_name}", file=sys.stderr)
            continue

        split = "train" if exp_name in TRAIN_EXPERIMENTS else "test"
        meta = EXPERIMENT_META.get(exp_name, {})

        # Determine tag prefixes based on split
        if split == "train":
            opt_tag = f"final_optimum"
            h2b_tag = f"final_h2b"
            nooh_tag = f"final_nooverhead"
        else:
            opt_tag = f"test_optimum"
            h2b_tag = f"test_h2b"
            nooh_tag = f"test_nooverhead"

        blis_opt = load_blis_results(results_dir / f"{exp_name}_{opt_tag}.json")
        blis_h2b = load_blis_results(results_dir / f"{exp_name}_{h2b_tag}.json")
        blis_nooh = load_blis_results(results_dir / f"{exp_name}_{nooh_tag}.json")

        row = {
            "experiment": exp_name,
            "split": split,
            **meta,
            "gt_tpot": gt["tpot_mean_ms"],
            "gt_e2e": gt["e2e_mean_ms"],
            "gt_ttft": gt["ttft_mean_ms"],
        }

        for label, blis in [("nooh", blis_nooh), ("h2b", blis_h2b), ("opt", blis_opt)]:
            if blis:
                row[f"{label}_tpot"] = blis["tpot_mean_ms"]
                row[f"{label}_e2e"] = blis["e2e_mean_ms"]
                row[f"{label}_tpot_mape"] = mape(blis["tpot_mean_ms"], gt["tpot_mean_ms"])
                row[f"{label}_e2e_mape"] = mape(blis["e2e_mean_ms"], gt["e2e_mean_ms"])
            else:
                row[f"{label}_tpot"] = row[f"{label}_e2e"] = None
                row[f"{label}_tpot_mape"] = row[f"{label}_e2e_mape"] = None

        rows.append(row)

    if not rows:
        print("ERROR: no results", file=sys.stderr)
        sys.exit(1)

    # --- Per-experiment TPOT MAPE comparison (3-way) ---
    print(f"  Per-Experiment TPOT MAPE: No Overhead vs H2b (100μs) vs Optimum ({optimum}μs)")
    print("-" * 110)
    hdr = f"{'Experiment':<45} {'Split':<6} {'NoOH':>7} {'H2b':>7} {'Opt':>7} {'H2b→Opt':>8} {'Eff OH (μs)':>11}"
    print(hdr)
    print("-" * 110)

    for row in rows:
        meta = EXPERIMENT_META.get(row["experiment"], {})
        layers = meta.get("layers", 32)
        tp = meta.get("tp", 1)
        eff_overhead = optimum * layers / tp

        print(
            f"{row['experiment']:<45} {row['split']:<6} "
            f"{fmt_pct(row.get('nooh_tpot_mape')):>7} "
            f"{fmt_pct(row.get('h2b_tpot_mape')):>7} "
            f"{fmt_pct(row.get('opt_tpot_mape')):>7} "
            f"{fmt_delta(row.get('h2b_tpot_mape'), row.get('opt_tpot_mape')):>8} "
            f"{eff_overhead:>11.0f}"
        )

    # --- Per-experiment E2E MAPE comparison ---
    print()
    print(f"  Per-Experiment E2E MAPE: No Overhead vs H2b (100μs) vs Optimum ({optimum}μs)")
    print("-" * 110)
    hdr = f"{'Experiment':<45} {'Split':<6} {'NoOH':>7} {'H2b':>7} {'Opt':>7} {'H2b→Opt':>8}"
    print(hdr)
    print("-" * 110)

    for row in rows:
        print(
            f"{row['experiment']:<45} {row['split']:<6} "
            f"{fmt_pct(row.get('nooh_e2e_mape')):>7} "
            f"{fmt_pct(row.get('h2b_e2e_mape')):>7} "
            f"{fmt_pct(row.get('opt_e2e_mape')):>7} "
            f"{fmt_delta(row.get('h2b_e2e_mape'), row.get('opt_e2e_mape')):>8}"
        )

    # --- Aggregate by split ---
    print()
    print("=" * 80)
    print("  Aggregate by Split")
    print("=" * 80)
    print()

    for split_name, split_rows in [("TRAIN", [r for r in rows if r["split"] == "train"]),
                                     ("TEST", [r for r in rows if r["split"] == "test"]),
                                     ("ALL", rows)]:
        nooh_tpot = mean([r["nooh_tpot_mape"] for r in split_rows if r.get("nooh_tpot_mape") is not None])
        h2b_tpot = mean([r["h2b_tpot_mape"] for r in split_rows if r.get("h2b_tpot_mape") is not None])
        opt_tpot = mean([r["opt_tpot_mape"] for r in split_rows if r.get("opt_tpot_mape") is not None])
        nooh_e2e = mean([r["nooh_e2e_mape"] for r in split_rows if r.get("nooh_e2e_mape") is not None])
        h2b_e2e = mean([r["h2b_e2e_mape"] for r in split_rows if r.get("h2b_e2e_mape") is not None])
        opt_e2e = mean([r["opt_e2e_mape"] for r in split_rows if r.get("opt_e2e_mape") is not None])

        print(f"  {split_name} (N={len(split_rows)}):")
        print(f"    TPOT MAPE:  NoOH={fmt_pct(nooh_tpot)}  H2b={fmt_pct(h2b_tpot)}  Opt={fmt_pct(opt_tpot)}  H2b→Opt: {fmt_delta(h2b_tpot, opt_tpot)}")
        print(f"    E2E  MAPE:  NoOH={fmt_pct(nooh_e2e)}  H2b={fmt_pct(h2b_e2e)}  Opt={fmt_pct(opt_e2e)}  H2b→Opt: {fmt_delta(h2b_e2e, opt_e2e)}")
        print()

    # --- Per-family optimum analysis ---
    print("=" * 80)
    print("  Per-Model-Family TPOT MAPE at Optimum")
    print("=" * 80)
    print()

    families = {}
    for row in rows:
        fam = row.get("family", "unknown")
        if fam not in families:
            families[fam] = []
        families[fam].append(row)

    for fam, fam_rows in sorted(families.items()):
        opt_mapes = [r["opt_tpot_mape"] for r in fam_rows if r.get("opt_tpot_mape") is not None]
        h2b_mapes = [r["h2b_tpot_mape"] for r in fam_rows if r.get("h2b_tpot_mape") is not None]
        print(f"  {fam:<20} N={len(fam_rows)}  H2b={fmt_pct(mean(h2b_mapes))}  Opt={fmt_pct(mean(opt_mapes))}  Δ: {fmt_delta(mean(h2b_mapes), mean(opt_mapes))}")

    # --- Accept criteria ---
    print()
    print("=" * 80)
    print("  Accept Criteria Evaluation")
    print("=" * 80)
    print()

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    train_opt_tpot = mean([r["opt_tpot_mape"] for r in train_rows if r.get("opt_tpot_mape") is not None])
    test_opt_tpot = mean([r["opt_tpot_mape"] for r in test_rows if r.get("opt_tpot_mape") is not None])

    # Criterion 1: Generalization (test within 3pp of train)
    if train_opt_tpot is not None and test_opt_tpot is not None:
        gap = abs(test_opt_tpot - train_opt_tpot)
        c1 = gap <= 0.03
        print(f"  1. Generalization: |test - train| TPOT MAPE = {gap*100:.1f}pp (threshold: ≤3pp)")
        print(f"     Train={fmt_pct(train_opt_tpot)} Test={fmt_pct(test_opt_tpot)}")
        print(f"     {'PASS' if c1 else 'FAIL'}")
    else:
        print("  1. Generalization: UNABLE TO EVALUATE")

    # Criterion 2: Improvement (test TPOT MAPE < 20%)
    if test_opt_tpot is not None:
        c2 = test_opt_tpot < 0.20
        print(f"  2. Improvement: Test TPOT MAPE = {fmt_pct(test_opt_tpot)} (threshold: <20%)")
        print(f"     {'PASS' if c2 else 'FAIL'}")
    else:
        print("  2. Improvement: UNABLE TO EVALUATE")

    # Criterion 3: Stability (train and test optima differ by ≤30μs)
    # We only ran the fine sweep on train. Check if the test-set optimum
    # would be close by comparing test MAPE at optimum vs at optimum±25.
    print(f"  3. Stability: Fine optimum={optimum}μs, coarse optimum={coarse_optimum}μs, delta={abs(optimum - coarse_optimum)}μs")
    print(f"     (Full test-set sweep not performed; using train-set optimum for both)")

    # Overfitting guard
    if train_opt_tpot is not None and test_opt_tpot is not None:
        overfit_gap = test_opt_tpot - train_opt_tpot
        overfit = overfit_gap > 0.05
        print(f"  Overfitting guard: test - train = {overfit_gap*100:.1f}pp (threshold: >5pp → reject)")
        print(f"     {'FAIL (overfitting detected)' if overfit else 'PASS (no overfitting)'}")

    # Comparison vs H2b (100μs)
    print()
    all_h2b_tpot = mean([r["h2b_tpot_mape"] for r in rows if r.get("h2b_tpot_mape") is not None])
    all_opt_tpot = mean([r["opt_tpot_mape"] for r in rows if r.get("opt_tpot_mape") is not None])
    print(f"  Overall: H2b (100μs) TPOT MAPE = {fmt_pct(all_h2b_tpot)} → Optimum ({optimum}μs) = {fmt_pct(all_opt_tpot)}  Δ: {fmt_delta(all_h2b_tpot, all_opt_tpot)}")

    # Worst-case experiment
    worst = max(rows, key=lambda r: r.get("opt_tpot_mape") or 0)
    print(f"  Worst experiment: {worst['experiment']} at {fmt_pct(worst.get('opt_tpot_mape'))}")

    print()


def main():
    parser = argparse.ArgumentParser(description="HG1 Grid Search Analysis")
    parser.add_argument("--phase", required=True, choices=["coarse", "fine", "final"])
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--optimum", type=int, default=100)
    parser.add_argument("--coarse-optimum", type=int, default=100)
    args = parser.parse_args()

    if args.phase == "coarse":
        opt = find_optimum(args.results_dir, args.gt_dir, args.start, args.end, args.step, "coarse")
        print(opt)  # stdout: just the number for run.sh to capture
    elif args.phase == "fine":
        opt = find_optimum(args.results_dir, args.gt_dir, args.start, args.end, args.step, "fine")
        print(opt)
    elif args.phase == "final":
        final_analysis(args.results_dir, args.gt_dir, args.optimum, args.coarse_optimum)


if __name__ == "__main__":
    main()
