#!/usr/bin/env python3
"""
Ablation experiment for H-ablation-beta9: Test if β₉ contributes to Scout improvement.

This script evaluates the trained iter9 model with β₉ = 0 (ablated) to determine
if β₉ had any meaningful contribution to the results.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from scripts.inner_loop_optimize import evaluate_model

def run_ablation():
    """Run ablation: evaluate iter9 model with β₉ = 0"""

    # Load iter9 results
    with open('../inner_loop_results.json') as f:
        iter9_results = json.load(f)

    # Get best params from iter9
    best_params = iter9_results['best_params'].copy()

    print("=== Ablation: β₉ = 0 ===")
    print(f"Original β₉: {best_params['beta'][9]:.10f} seconds/token/layer")
    print(f"Original β₉: {best_params['beta'][9]*1e6:.4f} μs/token/layer")

    # Ablate β₉ (set to zero)
    ablated_params = best_params.copy()
    ablated_params['beta'] = best_params['beta'].copy()
    ablated_params['beta'][9] = 0.0

    print(f"Ablated β₉: {ablated_params['beta'][9]} (set to zero)")

    # Evaluate ablated model on all experiments
    print("\nEvaluating ablated model on all 15 experiments...")
    ablation_results = evaluate_model(
        alpha=ablated_params['alpha'],
        beta=ablated_params['beta'],
        backend_name='evolved',
        iteration=9
    )

    # Save ablation results
    with open('ablation_beta9_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    print(f"\n✓ Ablation complete")
    print(f"  Overall loss: {ablation_results['loss']['overall_loss']:.2f}%")
    print(f"  TTFT RMSE: {ablation_results['loss']['ttft_rmse']:.2f}%")
    print(f"  E2E RMSE: {ablation_results['loss']['e2e_rmse']:.2f}%")

    # Compare Scout experiments
    print("\n=== Scout TTFT Comparison (Full vs Ablated) ===")
    print("Experiment                      Full β₉      Ablated β₉   Difference")
    print("-" * 75)

    for full_exp in iter9_results['per_experiment_results']:
        if 'Scout' in full_exp['model']:
            workload = full_exp['workload']
            full_ttft = full_exp['ttft_mean_ape']

            # Find matching ablated experiment
            ablated_exp = next((e for e in ablation_results['per_experiment_results']
                               if e['workload'] == workload), None)
            if ablated_exp:
                ablated_ttft = ablated_exp['ttft_mean_ape']
                diff = ablated_ttft - full_ttft
                print(f"Scout {workload:<20}   {full_ttft:>7.2f}%    {ablated_ttft:>7.2f}%    {diff:>+7.2f}pp")

    # Calculate overall difference
    full_loss = iter9_results['loss']['overall_loss']
    ablated_loss = ablation_results['loss']['overall_loss']
    loss_diff = ablated_loss - full_loss

    print(f"\n=== Overall Loss Comparison ===")
    print(f"Full model (with β₉):     {full_loss:.2f}%")
    print(f"Ablated model (β₉ = 0):   {ablated_loss:.2f}%")
    print(f"Difference:               {loss_diff:>+.2f}pp")

    if abs(loss_diff) < 1.0:
        verdict = "✓ CONFIRMED: β₉ has negligible effect (<1pp difference)"
    else:
        verdict = f"✗ REJECTED: β₉ has {abs(loss_diff):.2f}pp effect"

    print(f"\nVerdict: {verdict}")

    return {
        'full_loss': full_loss,
        'ablated_loss': ablated_loss,
        'difference': loss_diff,
        'verdict': verdict
    }

if __name__ == '__main__':
    try:
        result = run_ablation()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
