"""H31: Analyze generalization to reasoning (test-set) experiments.

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
        "n_blis_preemptions": blis.get("preemption_count", 0),
        "n_total": gt["config"]["total_requests"],
        "ttft_re": signed_re(blis["ttft_mean_ms"], gt["ttft"]["mean_ms"]),
        "e2e_re": signed_re(blis["e2e_mean_ms"], gt["e2e"]["mean_ms"]),
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
        if gt["profile"] != "reasoning":
            continue
        results.append(analyze_experiment(
            os.path.join(args.replay_dir, fname), gt_path))

    if not results:
        print("No reasoning experiments found.")
        sys.exit(1)

    primary = [r for r in results if r["model"] == "codellama-34b" and r["split"] == "test"]

    print(f"\n{'='*90}")
    print(f"  H31: Generalization to Reasoning — {len(results)} experiments")
    print(f"{'='*90}")

    print(f"\n  {'Model':<16} {'Split':<8} {'Real rps':>8} {'BLIS rps':>8} "
          f"{'Thru RE':>8} {'TTFT RE':>9} {'E2E RE':>9} {'Compl':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")

    for r in results:
        print(f"  {r['model']:<16} {r['split']:<8} {r['real_rps']:>7.2f} {r['blis_rps']:>7.2f} "
              f"{r['thru_re']:>+7.1f}% {r['ttft_re']:>+8.1f}% {r['e2e_re']:>+8.1f}% "
              f"{r['n_blis_completed']:>6}/{r['n_total']}")

    if primary:
        p = primary[0]
        print(f"\n  PRIMARY: codellama-34b-reasoning")
        print(f"    Throughput: {p['blis_rps']:.2f} vs {p['real_rps']:.2f} rps (RE: {p['thru_re']:+.1f}%)")
        print(f"    Gate: throughput |RE| < 10% -> {'PASS' if abs(p['thru_re']) < 10 else 'FAIL'}")
        verdict = "CONFIRMED" if abs(p["thru_re"]) < 10 else "REFUTED"
        print(f"\n  VERDICT: {verdict}")
    else:
        verdict = "INCONCLUSIVE"
        print("\n  codellama-34b-reasoning not found")

    print(f"{'='*90}")
    out = os.path.join(args.replay_dir, "h31_analysis.json")
    with open(out, "w") as f:
        json.dump({"verdict": verdict, "results": results}, f, indent=2)
    print(f"  Full results: {out}")


if __name__ == "__main__":
    main()
