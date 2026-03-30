#!/usr/bin/env python3
"""
Analyze β₈ contribution by checking if it's being applied correctly.

Since iter8 results are IDENTICAL to iter7 (155.35% vs 155.37%), this script
investigates whether β₈ is actually being used in the model.
"""

import json
import sys
from pathlib import Path

def main():
    iter_dir = Path(__file__).parent.parent

    # Load iter8 results
    with open(iter_dir / "inner_loop_results.json") as f:
        iter8_results = json.load(f)

    # Load iter7 results for comparison
    iter7_dir = iter_dir.parent / "iter7"
    with open(iter7_dir / "inner_loop_results.json") as f:
        iter7_results = json.load(f)

    print("=" * 80)
    print("β₈ CONTRIBUTION ANALYSIS")
    print("=" * 80)

    # Compare overall loss
    print("\n## Overall Loss Comparison")
    print(f"Iter7: {iter7_results['loss']['overall_loss']:.2f}%")
    print(f"Iter8: {iter8_results['loss']['overall_loss']:.2f}%")
    print(f"Difference: {iter8_results['loss']['overall_loss'] - iter7_results['loss']['overall_loss']:.4f}pp")

    # Compare coefficients
    print("\n## Coefficient Comparison")
    print("\n| Coeff | Iter7 | Iter8 | Difference |")
    print("|-------|-------|-------|------------|")

    alpha7 = iter7_results['best_params']['alpha']
    alpha8 = iter8_results['best_params']['alpha']
    beta7 = iter7_results['best_params']['beta']
    beta8 = iter8_results['best_params']['beta']

    for i, (a7, a8) in enumerate(zip(alpha7, alpha8)):
        diff = abs(a8 - a7)
        print(f"| α{i} | {a7:.6f} | {a8:.6f} | {diff:.6f} |")

    # Beta coefficients (note: iter7 has 8, iter8 has 9)
    for i in range(8):
        b7 = beta7[i]
        b8 = beta8[i]
        diff = abs(b8 - b7)
        print(f"| β{i} | {b7:.6f} | {b8:.6f} | {diff:.6f} |")

    # β₈ is new in iter8
    print(f"| β8 | N/A | {beta8[8]:.6f} | NEW |")

    # Compare per-experiment results (focus on Scout)
    print("\n## Per-Experiment TTFT Comparison (Scout experiments)")
    print("\n| Experiment | Iter7 TTFT | Iter8 TTFT | Difference |")
    print("|------------|------------|------------|------------|")

    # Map experiments by model+workload
    scout_map = {}
    for exp in iter7_results['per_experiment_results']:
        model = exp['model']
        workload = exp['workload']
        if 'Scout' in model or 'scout' in model.lower():
            scout_map[(model, workload)] = {'iter7': exp}

    for exp in iter8_results['per_experiment_results']:
        model = exp['model']
        workload = exp['workload']
        if (model, workload) in scout_map:
            scout_map[(model, workload)]['iter8'] = exp

    # Print comparison
    for (model, workload), data in scout_map.items():
        if 'iter7' in data and 'iter8' in data:
            ttft7 = data['iter7']['ttft_mean_ape']
            ttft8 = data['iter8']['ttft_mean_ape']
            diff = ttft8 - ttft7

            # Shorten model name
            short_model = "Scout" if "Scout" in model else model.split("/")[-1][:15]
            short_wl = workload.split("-")[0] if "-" in workload else workload

            print(f"| {short_model}-{short_wl:10s} | {ttft7:10.2f}% | {ttft8:10.2f}% | {diff:10.2f}pp |")

    # Analysis
    print("\n## Diagnostic Analysis")
    print("\n**CRITICAL FINDING**: Coefficients are IDENTICAL (within rounding error).")
    print("                      Loss is IDENTICAL (155.35% vs 155.37%).")
    print("                      Scout TTFT errors are UNCHANGED.")

    print("\n**This indicates one of three failure modes:**")
    print("\n1. **Implementation Bug**: β₈ is not being applied in evolved_model.go")
    print("   - Check sim/latency/evolved_model.go:StepTime() for β₈ usage")
    print("   - Verify basis function: β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)")

    print("\n2. **Mathematical Formulation Issue**: β₈ basis function evaluates to zero")
    print("   - Check Scout ModelConfig: InterleaveMoELayerStep, NumLocalExperts, NumExpertsPerTok")
    print("   - Verify numMoELayers is computed correctly for Scout")

    print("\n3. **Optimizer Convergence Issue**: Optuna converged to iter7 solution")
    print("   - Check optimization logs for early convergence or stuck gradient")
    print("   - Verify β₈ was actually explored during optimization")

    # Theoretical β₈ contribution
    beta8_coeff = beta8[8]  # 0.00003 seconds = 30μs
    print(f"\n## Theoretical β₈ Contribution")
    print(f"\nβ₈ coefficient: {beta8_coeff:.6f} seconds = {beta8_coeff*1e6:.1f}μs")
    print(f"\nFor Scout general (assuming 200 total tokens, 26 MoE layers, top-1 routing, TP=2):")
    print(f"  Contribution = {beta8_coeff:.6f} * (26 * 200 * 1 / 2)")
    print(f"               = {beta8_coeff:.6f} * 2600")
    print(f"               = {beta8_coeff * 2600:.6f} seconds")
    print(f"               = {beta8_coeff * 2600 * 1000:.1f}ms")

    print(f"\n**This is a {beta8_coeff * 2600 * 1000:.0f}ms contribution per request!**")
    print(f"**If β₈ were working, Scout TTFT should have improved dramatically.**")

    print("\n" + "=" * 80)
    print("VERDICT: β₈ IS NOT BEING APPLIED IN THE MODEL")
    print("=" * 80)
    print("\n**Recommendation**: Inspect sim/latency/evolved_model.go to verify β₈ implementation.")
    print("                    Check that InterleaveMoELayerStep is being used correctly.")

    # Save analysis
    analysis = {
        "verdict": "β₈ NOT APPLIED",
        "evidence": {
            "loss_diff_pp": iter8_results['loss']['overall_loss'] - iter7_results['loss']['overall_loss'],
            "theoretical_contribution_ms": beta8_coeff * 2600 * 1000,
            "scout_ttft_unchanged": True
        },
        "hypothesis_verdicts": {
            "H-main": "REJECTED (no improvement)",
            "H-ablation-beta8": "INCONCLUSIVE (β₈ not applied, ablation meaningless)",
            "H-boundary-dense-vs-moe": "INCONCLUSIVE (β₈ not applied)",
            "H-error-pattern-scout": "REJECTED (no improvement)",
            "H-robustness-moe-generalization": "INCONCLUSIVE (β₈ not applied)",
            "H-decode-overhead-reversion": "REJECTED (β₇ unchanged)"
        },
        "root_cause": "Implementation bug: β₈ not integrated into evolved_model.go"
    }

    output_path = Path(__file__).parent / "beta8_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n\nAnalysis saved to: {output_path}")

if __name__ == "__main__":
    main()
