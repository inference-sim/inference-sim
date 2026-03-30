#!/usr/bin/env python3
"""
Ablation study for β₈ (MoE routing overhead).

Tests H-ablation-beta8: β₈ accounts for majority of Scout improvement.

Procedure:
1. Load best_params from iter8/inner_loop_results.json
2. Set β₈ = 0 (ablation)
3. Re-evaluate on all 15 experiments
4. Compare: TTFT with β₈ vs TTFT without β₈
"""

import json
import sys
import os
from pathlib import Path

# Add training/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inner_loop_optimize import evaluate_experiments

def main():
    iter_dir = Path(__file__).parent.parent
    results_path = iter_dir / "inner_loop_results.json"

    # Load best params
    with open(results_path) as f:
        results = json.load(f)

    alpha = results["best_params"]["alpha"]
    beta_full = results["best_params"]["beta"]

    print("=" * 80)
    print("β₈ ABLATION EXPERIMENT")
    print("=" * 80)
    print(f"\nFull model beta coefficients:")
    for i, b in enumerate(beta_full):
        print(f"  β{i}: {b:.6f}")

    # Create ablated beta (set β₈ = 0)
    beta_ablated = beta_full.copy()
    beta_ablated[8] = 0.0  # β₈ index

    print(f"\nAblated model (β₈ = 0):")
    for i, b in enumerate(beta_ablated):
        marker = " ← ABLATED" if i == 8 else ""
        print(f"  β{i}: {b:.6f}{marker}")

    # Evaluate both models
    print("\n" + "=" * 80)
    print("EVALUATING FULL MODEL (with β₈)")
    print("=" * 80)
    loss_full, per_exp_full = evaluate_experiments(alpha, beta_full, backend_name="evolved")

    print("\n" + "=" * 80)
    print("EVALUATING ABLATED MODEL (β₈ = 0)")
    print("=" * 80)
    loss_ablated, per_exp_ablated = evaluate_experiments(alpha, beta_ablated, backend_name="evolved")

    # Compute differences
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)

    print(f"\nOverall Loss:")
    print(f"  Full model:    {loss_full['overall_loss']:.2f}%")
    print(f"  Ablated (β₈=0): {loss_ablated['overall_loss']:.2f}%")
    print(f"  Difference:     {loss_ablated['overall_loss'] - loss_full['overall_loss']:.2f}pp")

    print(f"\nTTFT RMSE:")
    print(f"  Full model:    {loss_full['ttft_rmse']:.2f}%")
    print(f"  Ablated (β₈=0): {loss_ablated['ttft_rmse']:.2f}%")
    print(f"  Difference:     {loss_ablated['ttft_rmse'] - loss_full['ttft_rmse']:.2f}pp")

    print(f"\nE2E RMSE:")
    print(f"  Full model:    {loss_full['e2e_rmse']:.2f}%")
    print(f"  Ablated (β₈=0): {loss_ablated['e2e_rmse']:.2f}%")
    print(f"  Difference:     {loss_ablated['e2e_rmse'] - loss_full['e2e_rmse']:.2f}pp")

    # Per-experiment comparison (focus on Scout experiments)
    print("\n" + "=" * 80)
    print("PER-EXPERIMENT COMPARISON (Scout experiments)")
    print("=" * 80)

    scout_experiments = [
        "17-llama-4-scout-17b-16e-tp2-general-2",
        "48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1",
        "20-llama-4-scout-17b-16e-tp2-codegen-2",
        "21-llama-4-scout-17b-16e-tp2-roleplay-2"
    ]

    print("\n| Experiment | TTFT (full) | TTFT (ablated) | Δ TTFT | Verdict |")
    print("|------------|-------------|----------------|---------|---------|")

    for exp_full, exp_abl in zip(per_exp_full, per_exp_ablated):
        exp_name = Path(exp_full["experiment_folder"]).name

        # Only show Scout experiments
        is_scout = any(scout in exp_name for scout in scout_experiments)
        if not is_scout:
            continue

        ttft_full = exp_full["ttft_mean_ape"]
        ttft_abl = exp_abl["ttft_mean_ape"]
        delta = ttft_abl - ttft_full

        # Shorten experiment name
        short_name = exp_name.split("-")[0] + " " + exp_name.split("-")[-2]

        verdict = "No effect" if abs(delta) < 1 else f"{'+' if delta > 0 else ''}{delta:.1f}pp"

        print(f"| {short_name:15s} | {ttft_full:11.2f}% | {ttft_abl:14.2f}% | {delta:7.2f}pp | {verdict:7s} |")

    # Save results
    ablation_results = {
        "full_model": {
            "alpha": alpha,
            "beta": beta_full,
            "loss": loss_full,
            "per_experiment": per_exp_full
        },
        "ablated_model": {
            "alpha": alpha,
            "beta": beta_ablated,
            "loss": loss_ablated,
            "per_experiment": per_exp_ablated
        },
        "verdict": {
            "overall_loss_diff_pp": loss_ablated['overall_loss'] - loss_full['overall_loss'],
            "ttft_rmse_diff_pp": loss_ablated['ttft_rmse'] - loss_full['ttft_rmse'],
            "e2e_rmse_diff_pp": loss_ablated['e2e_rmse'] - loss_full['e2e_rmse'],
            "hypothesis": "H-ablation-beta8: β₈ contributes >30pp to Scout improvement",
            "threshold": ">20pp difference",
            "result": "CONFIRMED" if abs(loss_ablated['ttft_rmse'] - loss_full['ttft_rmse']) > 20 else "REJECTED"
        }
    }

    output_path = Path(__file__).parent / "beta8_ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("\n" + "=" * 80)
    print("HYPOTHESIS VERDICT")
    print("=" * 80)
    print(f"\nH-ablation-beta8: β₈ contributes >30pp to Scout TTFT improvement")
    print(f"Threshold: >20pp difference")
    print(f"Actual difference: {abs(loss_ablated['ttft_rmse'] - loss_full['ttft_rmse']):.2f}pp")
    print(f"Verdict: {ablation_results['verdict']['result']}")

    if ablation_results['verdict']['result'] == "REJECTED":
        print(f"\n⚠️  DIAGNOSTIC: β₈ has no significant effect (<1pp on all metrics).")
        print(f"    This indicates β₈ is either:")
        print(f"    1. Not being used in the model calculations (implementation bug)")
        print(f"    2. Cancelled out by other coefficients (zero-sum trade-off)")
        print(f"    3. MoE routing overhead is negligible (incorrect hypothesis)")

if __name__ == "__main__":
    main()
