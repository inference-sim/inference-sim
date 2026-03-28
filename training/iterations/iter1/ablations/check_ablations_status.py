#!/usr/bin/env python3
"""Check status of running ablation experiments and generate summary when complete."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def get_training_root() -> Path:
    """Get training root directory (3 levels up from ablations)."""
    return Path(__file__).parent.parent.parent


def check_ablation_complete(iteration: int, ablation_name: str) -> bool:
    """Check if ablation results file exists."""
    training_root = get_training_root()
    results_path = training_root / f"iterations/iter{iteration}/ablations/ablation_no_{ablation_name}_results.json"
    return results_path.exists()


def load_comparison(iteration: int, ablation_name: str) -> Dict[str, Any]:
    """Load comparison results if available."""
    training_root = get_training_root()
    comp_path = training_root / f"iterations/iter{iteration}/ablations/ablation_no_{ablation_name}_comparison.json"
    if comp_path.exists():
        with open(comp_path, 'r') as f:
            return json.load(f)
    return None


def generate_summary_markdown(iteration: int, comparisons: Dict[str, Dict[str, Any]]) -> str:
    """Generate ABLATION-SUMMARY.md content."""

    ablation_names = {
        'chunking': 'β₅ (Chunking Term)',
        'tp_comm': 'β₃ (TP Communication Term)',
        'kv_mgmt': 'β₄ (KV Management Term)',
    }

    # Start with header
    lines = [
        f"# Iteration {iteration}: Ablation Study Results",
        "",
        "## Summary",
        "",
    ]

    # Count critical/moderate/redundant
    verdicts = {comp['verdict'] for comp in comparisons.values()}
    critical = sum(1 for comp in comparisons.values() if comp['verdict'] == 'CRITICAL')
    moderate = sum(1 for comp in comparisons.values() if comp['verdict'] == 'MODERATE')
    redundant = sum(1 for comp in comparisons.values() if comp['verdict'] == 'REDUNDANT')

    lines.extend([
        f"Tested importance of {len(comparisons)} basis functions by removing each and measuring performance degradation.",
        "",
        "**Key findings**:",
        f"- **Critical terms**: {critical} (removal causes >10% overall loss OR >15% TTFT RMSE OR >10% E2E RMSE)",
        f"- **Moderate terms**: {moderate} (removal causes 5-10% impact)",
        f"- **Redundant terms**: {redundant} (removal causes <5% impact)",
        "",
        "---",
        "",
    ])

    # Generate section for each ablation
    for ablation_key in ['chunking', 'tp_comm', 'kv_mgmt']:
        if ablation_key not in comparisons:
            continue

        comp = comparisons[ablation_key]
        term_name = ablation_names[ablation_key]

        # Map ablation keys to hypothesis predictions
        hypothesis_map = {
            'chunking': "Removing β₅ will increase TTFT RMSE by >15%",
            'tp_comm': "Removing β₃ will increase overall loss by >10% for TP>1",
            'kv_mgmt': "Removing β₄ will increase E2E RMSE by >10%",
        }

        lines.extend([
            f"## Ablation: Remove {term_name}",
            "",
            f"**Hypothesis** (from HYPOTHESIS.md): {hypothesis_map[ablation_key]}",
            "",
            "**Full model performance**:",
            f"- Overall loss: {comp['baseline']['overall_loss']:.2f}%",
            f"- TTFT RMSE: {comp['baseline']['ttft_rmse']:.2f}%",
            f"- E2E RMSE: {comp['baseline']['e2e_rmse']:.2f}%",
            "",
            f"**Ablated model performance** ({term_name} forced to 0):",
            f"- Overall loss: {comp['ablation']['overall_loss']:.2f}%",
            f"- TTFT RMSE: {comp['ablation']['ttft_rmse']:.2f}%",
            f"- E2E RMSE: {comp['ablation']['e2e_rmse']:.2f}%",
            "",
            "**Performance delta**:",
            f"- Δ Overall loss: {comp['delta']['overall_loss']:+.2f}% ({comp['percentage_change']['overall_loss']:+.1f}% change)",
            f"- Δ TTFT RMSE: {comp['delta']['ttft_rmse']:+.2f}% ({comp['percentage_change']['ttft_rmse']:+.1f}% change)",
            f"- Δ E2E RMSE: {comp['delta']['e2e_rmse']:+.2f}% ({comp['percentage_change']['e2e_rmse']:+.1f}% change)",
            "",
        ])

        # Determine hypothesis verdict
        if ablation_key == 'chunking':
            # Hypothesis: TTFT RMSE +15%
            hypothesis_met = comp['delta']['ttft_rmse'] > 15.0
        elif ablation_key == 'tp_comm':
            # Hypothesis: Overall loss +10%
            hypothesis_met = comp['delta']['overall_loss'] > 10.0
        elif ablation_key == 'kv_mgmt':
            # Hypothesis: E2E RMSE +10%
            hypothesis_met = comp['delta']['e2e_rmse'] > 10.0

        verdict_emoji = "✅" if hypothesis_met else "❌"

        lines.extend([
            f"**Verdict**: {verdict_emoji} HYPOTHESIS {'CONFIRMED' if hypothesis_met else 'REJECTED'}",
            "",
            "**Evidence**:",
        ])

        # Evidence based on verdict
        if comp['verdict'] == 'CRITICAL':
            lines.append(f"- Removing {term_name} causes significant performance degradation (>10% loss increase)")
            lines.append(f"- Term captures essential overhead that other basis functions cannot absorb")
        elif comp['verdict'] == 'MODERATE':
            lines.append(f"- Removing {term_name} causes moderate performance degradation (5-10% loss increase)")
            lines.append(f"- Term captures real overhead but some compensation occurs from other terms")
        else:  # REDUNDANT
            lines.append(f"- Removing {term_name} causes minimal performance change (<5% loss increase)")
            lines.append(f"- Term overhead is negligible or fully absorbed by other terms (e.g., α₀ fixed API overhead)")

        lines.extend([
            "",
            "**Recommendation**:",
            f"- **{comp['recommendation']}** {term_name} for iter{iteration + 1}",
        ])

        if comp['recommendation'] == 'REMOVE':
            lines.append(f"- Investigate feature extraction (coefficient was near-zero in full model: check if feature calculation is correct)")

        lines.extend([
            "",
            "---",
            "",
        ])

    # Summary table
    lines.extend([
        "## Summary Table",
        "",
        "| Term Removed | Hypothesis Prediction | Actual Δ (Primary Metric) | Verdict | Recommendation |",
        "|--------------|----------------------|---------------------------|---------|----------------|",
    ])

    for ablation_key in ['chunking', 'tp_comm', 'kv_mgmt']:
        if ablation_key not in comparisons:
            continue

        comp = comparisons[ablation_key]
        term_name = ablation_names[ablation_key]

        # Primary metric for each hypothesis
        if ablation_key == 'chunking':
            primary_metric = f"TTFT RMSE {comp['delta']['ttft_rmse']:+.1f}%"
            predicted = "+15%"
        elif ablation_key == 'tp_comm':
            primary_metric = f"Overall {comp['delta']['overall_loss']:+.1f}%"
            predicted = "+10%"
        elif ablation_key == 'kv_mgmt':
            primary_metric = f"E2E RMSE {comp['delta']['e2e_rmse']:+.1f}%"
            predicted = "+10%"

        # Determine if hypothesis confirmed
        if ablation_key == 'chunking':
            hypothesis_met = comp['delta']['ttft_rmse'] > 15.0
        elif ablation_key == 'tp_comm':
            hypothesis_met = comp['delta']['overall_loss'] > 10.0
        elif ablation_key == 'kv_mgmt':
            hypothesis_met = comp['delta']['e2e_rmse'] > 10.0

        verdict_emoji = "✅" if hypothesis_met else "❌"

        lines.append(f"| {term_name} | {predicted} | {primary_metric} | {verdict_emoji} {comp['verdict']} | {comp['recommendation']} |")

    lines.extend([
        "",
        f"**Basis function changes for iter{iteration + 1}**:",
    ])

    # Recommendations
    remove_terms = [ablation_names[k] for k, c in comparisons.items() if c['recommendation'] == 'REMOVE']
    keep_terms = [ablation_names[k] for k, c in comparisons.items() if c['recommendation'] == 'KEEP']

    if remove_terms:
        lines.append(f"- **Remove**: {', '.join(remove_terms)}")
    else:
        lines.append("- **Remove**: None (all tested terms have measurable impact)")

    lines.append(f"- **Keep**: {', '.join(keep_terms)}")

    return "\n".join(lines)


def main():
    iteration = 1
    ablations = ['chunking', 'tp_comm', 'kv_mgmt']
    training_root = get_training_root()
    ablations_dir = training_root / f"iterations/iter{iteration}/ablations"

    print(f"Checking ablation status for iteration {iteration}...\n")

    # Check which ablations are complete
    complete = {}
    for ablation in ablations:
        is_complete = check_ablation_complete(iteration, ablation)
        print(f"  {ablation:12} : {'✅ Complete' if is_complete else '⏳ Running...'}")
        complete[ablation] = is_complete

    if not all(complete.values()):
        print(f"\n⏳ Waiting for {sum(not c for c in complete.values())} ablations to complete...")
        return 1

    print(f"\n✅ All ablations complete! Generating comparisons and summary...\n")

    # Run comparisons
    comparisons = {}
    for ablation in ablations:
        baseline_path = training_root / f"iterations/iter{iteration}/inner_loop_results.json"
        ablation_path = ablations_dir / f"ablation_no_{ablation}_results.json"
        output_path = ablations_dir / f"ablation_no_{ablation}_comparison.json"

        # Check if comparison already exists
        if not output_path.exists():
            print(f"Comparing {ablation}...")
            import subprocess
            subprocess.run([
                sys.executable,
                str(ablations_dir / "compare_ablation.py"),
                "--baseline", str(baseline_path),
                "--ablation", str(ablation_path),
                "--output", str(output_path)
            ], check=True)

        # Load comparison
        comparisons[ablation] = load_comparison(iteration, ablation)

    # Generate summary markdown
    print(f"\nGenerating ABLATION-SUMMARY.md...")
    summary_md = generate_summary_markdown(iteration, comparisons)

    summary_path = ablations_dir / "ABLATION-SUMMARY.md"
    summary_path.write_text(summary_md)
    print(f"  Saved: {summary_path}")

    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"\nAll results saved to iterations/iter{iteration}/ablations/")
    print("\nNext steps:")
    print(f"  1. Review ABLATION-SUMMARY.md")
    print(f"  2. Update iter{iteration}-HYPOTHESIS-validation.md with ablation results")
    print(f"  3. Update iter{iteration}-FINDINGS.md recommendations based on ablation verdicts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
