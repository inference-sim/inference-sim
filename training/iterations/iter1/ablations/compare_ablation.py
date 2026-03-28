#!/usr/bin/env python3
"""
Compare ablation results to full model baseline.

Usage:
    python compare_ablation.py --baseline inner_loop_results.json \
                                --ablation ablations/ablation_no_chunking_results.json \
                                --output ablations/ablation_no_chunking_comparison.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def load_results(path: Path) -> Dict[str, Any]:
    """Load optimization results JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def compare_results(baseline: Dict[str, Any], ablation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare ablation results to baseline.

    Returns dictionary with:
    - baseline_loss, ablation_loss, delta_loss
    - baseline_ttft_rmse, ablation_ttft_rmse, delta_ttft_rmse
    - baseline_e2e_rmse, ablation_e2e_rmse, delta_e2e_rmse
    - verdict (critical/moderate/redundant based on delta magnitudes)
    """

    baseline_loss = baseline['loss']
    ablation_loss = ablation['loss']

    delta_overall = ablation_loss['overall_loss'] - baseline_loss['overall_loss']
    delta_ttft = ablation_loss['ttft_rmse'] - baseline_loss['ttft_rmse']
    delta_e2e = ablation_loss['e2e_rmse'] - baseline_loss['e2e_rmse']

    # Calculate percentage changes
    pct_overall = (delta_overall / baseline_loss['overall_loss'] * 100) if baseline_loss['overall_loss'] > 0 else 0
    pct_ttft = (delta_ttft / baseline_loss['ttft_rmse'] * 100) if baseline_loss['ttft_rmse'] > 0 else 0
    pct_e2e = (delta_e2e / baseline_loss['e2e_rmse'] * 100) if baseline_loss['e2e_rmse'] > 0 else 0

    # Determine verdict based on delta magnitude
    if abs(delta_overall) > 10.0 or abs(delta_ttft) > 15.0 or abs(delta_e2e) > 10.0:
        verdict = "CRITICAL"  # Term is essential
        recommendation = "KEEP"
    elif abs(delta_overall) > 5.0 or abs(delta_ttft) > 5.0 or abs(delta_e2e) > 5.0:
        verdict = "MODERATE"  # Term has noticeable impact
        recommendation = "KEEP"
    else:
        verdict = "REDUNDANT"  # Term has minimal impact
        recommendation = "REMOVE"

    return {
        "baseline": {
            "overall_loss": baseline_loss['overall_loss'],
            "ttft_rmse": baseline_loss['ttft_rmse'],
            "e2e_rmse": baseline_loss['e2e_rmse'],
        },
        "ablation": {
            "overall_loss": ablation_loss['overall_loss'],
            "ttft_rmse": ablation_loss['ttft_rmse'],
            "e2e_rmse": ablation_loss['e2e_rmse'],
        },
        "delta": {
            "overall_loss": delta_overall,
            "ttft_rmse": delta_ttft,
            "e2e_rmse": delta_e2e,
        },
        "percentage_change": {
            "overall_loss": pct_overall,
            "ttft_rmse": pct_ttft,
            "e2e_rmse": pct_e2e,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "ablation_metadata": {
            "n_trials": ablation['optimization']['n_trials'],
            "converged_early": ablation['optimization']['converged_early'],
            "num_errors": ablation['optimization']['num_errors'],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare ablation to baseline")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline results JSON")
    parser.add_argument("--ablation", type=Path, required=True, help="Ablation results JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output comparison JSON")

    args = parser.parse_args()

    # Load results
    print(f"Loading baseline: {args.baseline}")
    baseline = load_results(args.baseline)

    print(f"Loading ablation: {args.ablation}")
    ablation = load_results(args.ablation)

    # Compare
    print("Comparing results...")
    comparison = compare_results(baseline, ablation)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Baseline Overall Loss:  {comparison['baseline']['overall_loss']:.2f}%")
    print(f"Ablation Overall Loss:  {comparison['ablation']['overall_loss']:.2f}%")
    print(f"Delta:                  {comparison['delta']['overall_loss']:+.2f}% ({comparison['percentage_change']['overall_loss']:+.1f}%)")
    print()
    print(f"Baseline TTFT RMSE:     {comparison['baseline']['ttft_rmse']:.2f}%")
    print(f"Ablation TTFT RMSE:     {comparison['ablation']['ttft_rmse']:.2f}%")
    print(f"Delta:                  {comparison['delta']['ttft_rmse']:+.2f}% ({comparison['percentage_change']['ttft_rmse']:+.1f}%)")
    print()
    print(f"Baseline E2E RMSE:      {comparison['baseline']['e2e_rmse']:.2f}%")
    print(f"Ablation E2E RMSE:      {comparison['ablation']['e2e_rmse']:.2f}%")
    print(f"Delta:                  {comparison['delta']['e2e_rmse']:+.2f}% ({comparison['percentage_change']['e2e_rmse']:+.1f}%)")
    print()
    print(f"Verdict:                {comparison['verdict']}")
    print(f"Recommendation:         {comparison['recommendation']}")
    print(f"{'='*60}\n")

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
