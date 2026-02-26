# Idea 1, H3: XGBoost Generalizes Across Unseen Models and Workloads

**Status:** Not Supported (LOMO avg 2559.7%, LOWO avg 109.7%)
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> The trained XGBoost model generalizes across unseen models and workloads, achieving <20% per-step MAPE in leave-one-model-out and leave-one-workload-out cross-validation.

## Refuted-If

Leave-one-model-out MAPE > 25% for any held-out model, or leave-one-workload-out MAPE > 25% for any held-out workload. This threshold is deliberately generous (25% vs. 20% target) because generalization to unseen configurations is inherently harder than interpolation within the training distribution. Exceeding 25% indicates the model has memorized configuration-specific patterns rather than learning transferable relationships.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. **Leave-one-model-out (LOMO):** For each of the 4 models, train XGBoost on the remaining 3 models (12 experiments) and evaluate on the held-out model's 4 experiments. Report per-step MAPE for each held-out model.
2. **Leave-one-workload-out (LOWO):** For each of the 4 workloads, train XGBoost on the remaining 3 workloads (12 experiments) and evaluate on the held-out workload's 4 experiments. Report per-step MAPE for each held-out workload.
3. **Feature importance stability:** Compare SHAP feature importance rankings across LOMO/LOWO folds. Stable rankings indicate the model has learned generalizable structure; unstable rankings indicate overfitting to configuration-specific artifacts.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Leave-one-out at the model and workload levels (no temporal split needed -- entire experiments are held out)
**Baselines:** Blackbox 2-feature model (must outperform on held-out data), Per-experiment Ridge (measures how much generalization is lost vs. per-experiment training)
**Success metric:** Per-step MAPE < 20% on every held-out model and workload

**MoE-specific evaluation:** Because there is only one MoE model (Mixtral-8x7B), the LOMO fold holding out Mixtral evaluates dense-to-MoE transfer. Per the research design, MoE validation also uses leave-one-workload-out within Mixtral's 4 workloads as a secondary check. MoE MAPE < 25% is acceptable (relaxed vs. 20% for dense).

**Diagnostic analysis:**
- Per-phase breakdown: Is generalization worse for prefill-dominated, decode-dominated, or mixed steps?
- Error distribution: Are errors symmetric (bias-free) or systematic (under/over-prediction)?
- Feature ablation: Which features contribute most to generalization vs. memorization?

## Related Work

- **Vidur** (Agrawal et al., MLSys 2024): LLM inference profiling-based simulator that uses piecewise linear models per operation type. Vidur requires per-model profiling; this hypothesis tests whether a single model can generalize without per-model calibration.
- **Splitwise** (Patel et al., ISCA 2024): Disaggregated inference with prefill/decode separation. Demonstrates that different compute phases have fundamentally different scaling behavior, motivating phase-aware feature engineering.
- **Habitat** (Yu et al., ATC 2021): Cross-GPU performance prediction via analytical scaling. Achieves 9.2% median error for cross-GPU transfer -- a useful generalization benchmark.

## Go Integration Path

If generalization succeeds, a single serialized XGBoost model ships with BLIS as the default latency backend (replacing `defaults.yaml` alpha/beta coefficients). The model accepts architecture metadata (`model_id`, `tp_degree`, `is_moe`) as input features, enabling zero-shot prediction on new model configurations. If generalization fails for MoE (>25% MAPE), the integration ships two models: one for dense architectures and one for MoE, selected automatically based on the `--model` flag's architecture type (detectable from HuggingFace `config.json`).
