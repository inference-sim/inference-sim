# Idea 2, H5: Per-Model Regime Ensemble Generalizes Across Unseen Workloads (LOWO)

**Status:** Refuted
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-27

## Hypothesis

> Per-model 3-regime Ridge ensembles (h1's approach) trained on 3 of 4 workload types achieve <25% per-step MAPE on the held-out workload, demonstrating robustness to workload distribution shifts. Since per-model training already pools all 4 workloads, removing one workload tests whether the regime decomposition captures computation structure rather than workload-specific distributional artifacts.

This is the production-critical generalization test: a deployed StepML model must handle unseen traffic mixes (e.g., a new customer with reasoning-heavy workloads) without retraining. The 3-regime structure should be inherently workload-agnostic since regime boundaries (prefill_tokens thresholds) are defined by computation type, not workload semantics.

## Refuted-If

- LOWO per-step MAPE > 25% averaged across all 4 holdout folds (for any single model)
- OR LOWO per-step MAPE > 40% for ANY single (model, held-out workload) combination
- OR "general" workload holdout MAPE > 35% for any model (production traffic must be predictable)
- OR LOWO degrades per-step MAPE by >2x compared to training on all 4 workloads (h1 baseline)
- OR no improvement over Round 1's LOWO (109.7% average)

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** 4-fold leave-one-workload-out (LOWO) cross-validation, applied independently per model.

**Folds (repeated for each of the 4 models):**
| Fold | Training Workloads (3) | Held-Out Workload | Distribution Shift |
|------|----------------------|-------------------|--------------------|
| 1 | codegen, roleplay, reasoning | **general** | Specialized→diverse (hardest) |
| 2 | general, roleplay, reasoning | **codegen** | Mixed→short-bursty |
| 3 | general, codegen, reasoning | **roleplay** | Mixed→short-bursty |
| 4 | general, codegen, roleplay | **reasoning** | Short→long-context (most OOD) |

For each (model, fold), train 3 regime-specific Ridge regressors on data from the 3 training workloads (temporal 60/20/20 within), then evaluate on the held-out workload's single experiment. Report 4 models x 4 folds = 16 per-step MAPE values.

**Held-out workloads of primary interest:**
- **general (Fold 1):** Most diverse batch compositions. Training on 3 specialized workloads may miss the full diversity of batch configurations seen in production-like traffic.
- **reasoning (Fold 4):** Mean step times 6,000-33,500 us with bimodal distributions. KV cache lengths are much longer than codegen/roleplay. Tests extrapolation in the KV feature space.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Full LOWO per model (entire workload held out). Within training workloads, temporal 60/20/20 for Ridge alpha tuning.
**Baselines:** (a) Round 1 LOWO (109.7% avg), (b) Per-model regime Ridge from h1 trained on all 4 workloads (upper bound), (c) Blackbox baseline on held-out workload, (d) Regime Ridge LOWO without KV features (ablation)
**Success metric:** LOWO per-step MAPE < 25% averaged across all folds, for each model independently

**Diagnostic analysis:**
- Per-regime MAPE on held-out workload: Does the distribution shift disproportionately affect mixed-heavy (Regime C)?
- Feature range extrapolation: For reasoning holdout, do KV features (kv_sum, kv_max) exceed the training range? Report extrapolation fraction.
- Regime distribution shift: Compare regime proportions between training and held-out workloads. Large shifts indicate regime-level OOD.

## BLIS E2E Claim

This sub-hypothesis reports per-step MAPE on held-out workload folds. Additionally, BLIS E2E error is computed on each held-out workload using the LOWO-trained model + secondary method constants from h3. This provides the strongest test of production readiness: **"If a customer deploys a workload type we haven't seen, will BLIS still produce accurate simulations?"**

**Primary diagnostic:** BLIS E2E on held-out workload < 20% is "production-ready." E2E between 20-50% is "usable with caveats." E2E > 50% is "requires workload-specific retraining."

## Related Work

- **Round 1, H3:** LOWO avg 109.7% MAPE with global XGBoost. Per-model regime training should dramatically improve this.
- **MIST** (arXiv:2504.09775): Trains on controlled workloads. This hypothesis evaluates observational workload transfer.
- **AIConfigurator** (arXiv:2601.06288): Tests serving under different arrival patterns (Poisson, bursty, time-varying). Analogous to workload generalization.
- **Splitwise** (ISCA 2024): Prefill/decode disaggregation changes the workload mix. Relevant to understanding regime-level distribution shifts.

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Reasoning workload KV features out-of-distribution | High | Check feature ranges; if extrapolation > 20% of test steps, report separately. Consider adding feature clipping or log-transform. |
| "general" workload has unique batch configurations | Medium | Analyze regime proportions in general vs. training. If mixed-heavy proportion differs by > 2x, regime distribution shift is the root cause. |
| codegen and roleplay are near-identical | Low | Report separately; if codegen↔roleplay transfer MAPE < 10%, these folds are uninformative and should not inflate the "easy" average. |
| Sample size reduction (3 workloads → ~75% training data) | Low | Ridge is low-variance; should be robust to moderate data reduction. Check if val MAPE increases > 3 percentage points vs. h1. |

## Go Integration Path

If LOWO succeeds (<25% MAPE), the per-model StepML model is deployed as-is for any workload type. No workload-specific calibration or metadata is needed. This is the simplest and most maintainable production deployment: one artifact per model, serving all workloads. If LOWO fails for specific workloads, the `StepMLArtifact` could include optional `workload_overrides` for problematic cases, selected via workload classification heuristics.

## Training Strategy and Data Split

- **Training:** For each LOWO fold, use 3 workloads for the target model (3 experiments). Within these, temporal 60/20/20 for Ridge alpha tuning.
- **Test:** Held-out workload's experiment for the target model (1 experiment, ~7,500 steps)
- **Cross-validation:** Full 4-fold LOWO x 4 models = 16 evaluations total
- **Aggregation:** Report per-model average LOWO MAPE, per-workload average LOWO MAPE, and overall grand average

## Data Integrity

- Held-out workload data is completely unseen during training for that fold
- Per-model training means no cross-model leakage
- Regime classification thresholds (prefill_tokens boundaries) are fixed a priori, not tuned per fold
- KV features from ProgressIndex (available at inference time)
- No roofline predictions used (BC-3-7)
