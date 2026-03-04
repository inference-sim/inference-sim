"""H32: Analyze aggregate capacity planning metrics of BLIS crossmodel vs real vLLM.

Compares BLIS --results-path output against generate_replay_specs.py ground truth.
Uses the same JSON formats as H30 (top-level BLIS MetricsOutput, flat GT structure).
"""

import argparse
import json
import os
import sys


def signed_re(predicted, actual):
    if actual == 0:
        return float("nan")
    return (predicted - actual) / actual * 100


def analyze_experiment(results_path, gt_path):
    with open(results_path) as f:
        blis = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    return {
        "experiment": gt["experiment"],
        "model": gt["model_short"],
        "profile": gt["profile"],
        "split": gt["split"],
        "n_real_success": gt["success_count"],
        "n_real_fail": gt["failure_count"],
        "n_blis_completed": blis["completed_requests"],
        "n_blis_dropped": blis.get("dropped_unservable", 0),
        "ttft_mean_re": signed_re(blis["ttft_mean_ms"], gt["ttft"]["mean_ms"]),
        "ttft_p99_re": signed_re(blis.get("ttft_p99_ms", 0), gt["ttft"]["p99_ms"]),
        "e2e_mean_re": signed_re(blis["e2e_mean_ms"], gt["e2e"]["mean_ms"]),
        "thru_re": signed_re(blis["responses_per_sec"], gt["throughput"]["requests_per_sec"]),
        "blis_rps": blis["responses_per_sec"],
        "real_rps": gt["throughput"]["requests_per_sec"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    args = parser.parse_args()

    results = []
    for fname in sorted(os.listdir(args.replay_dir)):
        if not fname.endswith("_results.json"):
            continue
        exp_name = fname.replace("_results.json", "")
        gt_path = os.path.join(args.ground_truth_dir, f"{exp_name}_ground_truth.json")
        if not os.path.exists(gt_path):
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        if gt["split"] != "validate":
            continue
        results.append(analyze_experiment(
            os.path.join(args.replay_dir, fname), gt_path))

    if not results:
        print("No validation experiments found.")
        sys.exit(1)

    codellama = [r for r in results if r["model"] == "codellama-34b"]
    mixtral = [r for r in results if r["model"] == "mixtral-8x7b"]

    print(f"\n{'='*100}")
    print(f"  H32: Aggregate Capacity Planning — VALIDATE ({len(results)} experiments)")
    print(f"{'='*100}")

    print(f"\n  {'Model':<16} {'Profile':<10} {'TTFT mean':>10} {'TTFT p99':>10} "
          f"{'E2E mean':>10} {'Thru RE':>10} {'Compl':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        print(f"  {r['model']:<16} {r['profile']:<10} "
              f"{r['ttft_mean_re']:>+9.1f}% {r['ttft_p99_re']:>+9.1f}% "
              f"{r['e2e_mean_re']:>+9.1f}% {r['thru_re']:>+9.1f}% "
              f"{r['n_blis_completed']:>8}/{r['n_real_success']+r['n_real_fail']}")

    # Gates for codellama: TTFT mean |RE| < 25%, throughput |RE| < 10%
    print(f"\n  CODELLAMA VALIDATION:")
    cl_pass = True
    for r in codellama:
        ttft_ok = abs(r["ttft_mean_re"]) < 25
        thru_ok = abs(r["thru_re"]) < 10
        if not (ttft_ok and thru_ok):
            cl_pass = False
        print(f"    {r['profile']}: TTFT mean RE={r['ttft_mean_re']:+.1f}% "
              f"({'PASS' if ttft_ok else 'FAIL'}), "
              f"throughput RE={r['thru_re']:+.1f}% "
              f"({'PASS' if thru_ok else 'FAIL'})")

    # Mixtral reasoning: expected to fail
    print(f"\n  MIXTRAL REASONING (overload, ~69% failure):")
    mx_failed = False
    for r in mixtral:
        thru_ok = abs(r["thru_re"]) < 25
        if not thru_ok:
            mx_failed = True
        print(f"    throughput RE={r['thru_re']:+.1f}% "
              f"(expected >25%: {'as expected' if not thru_ok else 'UNEXPECTED PASS'})")

    confirmed = cl_pass and mx_failed
    verdict = "PARTIALLY_CONFIRMED" if confirmed else "REFUTED"
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*100}")

    out = os.path.join(args.replay_dir, "h32_analysis.json")
    with open(out, "w") as f:
        json.dump({"verdict": verdict, "results": results}, f, indent=2)
    print(f"  Full results: {out}")


if __name__ == "__main__":
    main()
