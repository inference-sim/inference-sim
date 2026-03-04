"""H30: Analyze BLIS crossmodel replay vs real vLLM — request-level + aggregate.

Compares BLIS --results-path JSON output against real summary_lifecycle_metrics.json.
Since BLIS generates workloads from distributions (not exact per-request replay),
we compare:
  - Aggregate metrics: TTFT/E2E percentiles (p50, p90, p99), throughput
  - Distribution shape: mean, standard deviation overlap
  - Per-request: distribution-level comparison (not per-request matching)
"""

import argparse
import json
import os
import sys


def relative_error(predicted, actual):
    if actual == 0:
        return float("nan")
    return abs(predicted - actual) / actual * 100


def signed_re(predicted, actual):
    if actual == 0:
        return float("nan")
    return (predicted - actual) / actual * 100


def analyze_experiment(results_path: str, gt_path: str) -> dict:
    """Compare BLIS results against ground truth for one experiment."""
    with open(results_path) as f:
        blis = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    # BLIS results are in ms (from MetricsOutput)
    # Ground truth is already in ms (converted in generate_replay_specs.py)

    # Per-request TTFT/E2E from BLIS results
    blis_reqs = blis.get("requests", [])
    blis_ttfts = sorted([r["ttft_ms"] for r in blis_reqs if r.get("ttft_ms", 0) > 0])
    blis_e2es = sorted([r["e2e_ms"] for r in blis_reqs if r.get("e2e_ms", 0) > 0])

    def pctl(vals, p):
        if not vals:
            return 0
        idx = int(p / 100 * (len(vals) - 1))
        return vals[min(idx, len(vals) - 1)]

    return {
        "experiment": gt["experiment"],
        "model": gt["model_short"],
        "profile": gt["profile"],
        "split": gt["split"],
        "n_blis_completed": blis["completed_requests"],
        "n_blis_dropped": blis.get("dropped_unservable", 0),
        "n_real_success": gt["success_count"],
        "n_real_fail": gt["failure_count"],
        "ttft": {
            "blis_mean": blis.get("ttft_mean_ms", 0),
            "real_mean": gt["ttft"]["mean_ms"],
            "re_mean": signed_re(blis.get("ttft_mean_ms", 0), gt["ttft"]["mean_ms"]),
            "blis_p50": pctl(blis_ttfts, 50),
            "real_p50": gt["ttft"]["p50_ms"],
            "re_p50": signed_re(pctl(blis_ttfts, 50), gt["ttft"]["p50_ms"]),
            "blis_p90": pctl(blis_ttfts, 90),
            "real_p90": gt["ttft"]["p90_ms"],
            "re_p90": signed_re(pctl(blis_ttfts, 90), gt["ttft"]["p90_ms"]),
            "blis_p99": blis.get("ttft_p99_ms", pctl(blis_ttfts, 99)),
            "real_p99": gt["ttft"]["p99_ms"],
            "re_p99": signed_re(blis.get("ttft_p99_ms", pctl(blis_ttfts, 99)), gt["ttft"]["p99_ms"]),
        },
        "e2e": {
            "blis_mean": blis.get("e2e_mean_ms", 0),
            "real_mean": gt["e2e"]["mean_ms"],
            "re_mean": signed_re(blis.get("e2e_mean_ms", 0), gt["e2e"]["mean_ms"]),
            "blis_p50": pctl(blis_e2es, 50),
            "real_p50": gt["e2e"]["p50_ms"],
            "re_p50": signed_re(pctl(blis_e2es, 50), gt["e2e"]["p50_ms"]),
            "blis_p99": blis.get("e2e_p99_ms", pctl(blis_e2es, 99)),
            "real_p99": gt["e2e"]["p99_ms"],
            "re_p99": signed_re(blis.get("e2e_p99_ms", pctl(blis_e2es, 99)), gt["e2e"]["p99_ms"]),
        },
        "throughput": {
            "blis_rps": blis.get("responses_per_sec", 0),
            "real_rps": gt["throughput"]["requests_per_sec"],
            "re_rps": signed_re(blis.get("responses_per_sec", 0), gt["throughput"]["requests_per_sec"]),
            "blis_tps": blis.get("tokens_per_sec", 0),
            "real_tps": gt["throughput"]["output_tokens_per_sec"],
            "re_tps": signed_re(blis.get("tokens_per_sec", 0), gt["throughput"]["output_tokens_per_sec"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="H30: Request-level + aggregate fidelity")
    parser.add_argument("--output-dir", required=True, help="Dir with BLIS results")
    parser.add_argument("--ground-truth-dir", required=True, help="Dir with ground truth JSONs")
    parser.add_argument("--split", default="train", help="Which split to analyze")
    args = parser.parse_args()

    results = []
    for fname in sorted(os.listdir(args.output_dir)):
        if not fname.endswith("_results.json"):
            continue
        exp_name = fname.replace("_results.json", "")
        gt_path = os.path.join(args.ground_truth_dir, f"{exp_name}_ground_truth.json")
        if not os.path.exists(gt_path):
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        if gt["split"] != args.split:
            continue

        results_path = os.path.join(args.output_dir, fname)
        results.append(analyze_experiment(results_path, gt_path))

    if not results:
        print(f"No {args.split} experiments found in {args.output_dir}")
        sys.exit(1)

    # Print results
    print(f"\n{'='*110}")
    print(f"  H30: BLIS CrossModel vs Real vLLM — {args.split.upper()} ({len(results)} experiments)")
    print(f"{'='*110}")

    print(f"\n  {'Model':<16} {'Profile':<10} {'Compl':>6} "
          f"{'TTFT mean':>11} {'TTFT p50':>11} {'TTFT p99':>11} "
          f"{'E2E mean':>11} {'E2E p99':>11} {'Thru':>8}")
    print(f"  {'-'*16} {'-'*10} {'-'*6} "
          f"{'-'*11} {'-'*11} {'-'*11} "
          f"{'-'*11} {'-'*11} {'-'*8}")

    for r in results:
        t, e = r["ttft"], r["e2e"]
        print(f"  {r['model']:<16} {r['profile']:<10} "
              f"{r['n_blis_completed']:>6} "
              f"{t['re_mean']:>+10.1f}% {t['re_p50']:>+10.1f}% {t['re_p99']:>+10.1f}% "
              f"{e['re_mean']:>+10.1f}% {e['re_p99']:>+10.1f}% "
              f"{r['throughput']['re_rps']:>+7.1f}%")

    # Summary by model
    print(f"\n  Per-model summary:")
    models = sorted(set(r["model"] for r in results))
    for model in models:
        mrs = [r for r in results if r["model"] == model]
        avg_ttft_p99_re = sum(abs(r["ttft"]["re_p99"]) for r in mrs) / len(mrs)
        avg_e2e_p99_re = sum(abs(r["e2e"]["re_p99"]) for r in mrs) / len(mrs)
        avg_thru_re = sum(abs(r["throughput"]["re_rps"]) for r in mrs) / len(mrs)
        print(f"    {model:<16}: TTFT p99 |RE|={avg_ttft_p99_re:.1f}%  "
              f"E2E p99 |RE|={avg_e2e_p99_re:.1f}%  "
              f"Throughput |RE|={avg_thru_re:.1f}%")

    # Overall verdict — gates match updated HYPOTHESIS.md
    # Throughput RE < 10% (meaningful at saturation, tautological at sub-saturation)
    # E2E mean |RE| < 25% on ≥8 of 10 experiments
    thru_pass = all(abs(r["throughput"]["re_rps"]) < 10 for r in results)
    e2e_fail_count = sum(1 for r in results if abs(r["e2e"]["re_mean"]) > 25)
    e2e_pass = e2e_fail_count <= 2

    print(f"\n  Gates (per updated HYPOTHESIS.md):")
    print(f"    Throughput |RE| < 10% for all:   {'PASS' if thru_pass else 'FAIL'}")
    print(f"    E2E mean |RE| < 25% on ≥8/10:   {'PASS' if e2e_pass else 'FAIL'} ({e2e_fail_count} experiments exceed)")

    verdict = "PARTIALLY_CONFIRMED" if (thru_pass and e2e_pass) else "REFUTED"
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*110}")

    # Save
    output_path = os.path.join(args.output_dir, "h30_analysis.json")
    with open(output_path, "w") as f:
        json.dump({"verdict": verdict, "results": results}, f, indent=2)
    print(f"\n  Full results: {output_path}")


if __name__ == "__main__":
    main()
