# Idea 1, H5: Per-Model Piecewise-Linear StepTime Generalizes Across Unseen Workloads (LOWO)

**Status:** Refuted
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-27

## Hypothesis

> Per-model piecewise-linear StepTime models (h1's approach) trained on 3 of 4 workload types achieve <40% per-step MAPE on the held-out workload, improving over Round 1's LOWO result (109.7% MAPE). Since training is per-model (pooling workloads), removing one workload tests whether the model is robust to workload distribution shifts rather than memorizing workload-specific patterns.

Workload generalization is the more practically relevant test: a deployed model must handle unseen traffic mixes without retraining. This hypothesis validates that piecewise-linear regime models are robust to the workload distribution shifts present in the 4 workload types.

## Refuted-If

- LOWO per-step MAPE > 40% averaged across all 4 holdout folds (for any single model)
- OR LOWO per-step MAPE > 60% for ANY single (model, held-out workload) combination
- OR "general" workload holdout MAPE > 50% for any model (the hardest workload must still be predictable)
- OR no improvement over Round 1's LOWO baseline (109.7% average)

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** 4-fold leave-one-workload-out (LOWO) cross-validation, applied independently per model.

**Folds (repeated per model):**
| Fold | Training Workloads (3) | Held-Out Workload |
|------|----------------------|-------------------|
| 1 | codegen, roleplay, reasoning | **general** |
| 2 | general, roleplay, reasoning | **codegen** |
| 3 | general, codegen, reasoning | **roleplay** |
| 4 | general, codegen, roleplay | **reasoning** |

For each model, train h1's piecewise-linear model on 3 workloads (temporal 60/20/20 within training data), then evaluate on the held-out workload. Report 4 models x 4 folds = 16 per-step MAPE values.

**Held-out workloads of primary interest:**
- **general (Fold 1):** Most diverse batch compositions, consistently the hardest workload in Round 1. If the model can generalize to general traffic from specialized workloads, it validates production robustness.
- **reasoning (Fold 4):** Longest sequences (mean 6,000-33,500 us), bimodal distributions. Tests whether short-context training transfers to long-context inference.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Full LOWO per model (entire workload held out). Within training workloads, temporal 60/20/20 for hyperparameter tuning.
**Baselines:** (a) Round 1 LOWO (109.7% avg), (b) Per-model piecewise-linear trained on all 4 workloads from h1 (upper bound), (c) Blackbox baseline on held-out workload
**Success metric:** LOWO per-step MAPE < 40% averaged across all folds, for each model independently

**Diagnostic analysis:**
- Per-regime MAPE on held-out workload: Does distribution shift affect decode-only and mixed-batch regimes differently?
- Feature value drift: Compare feature distributions between training and held-out workloads (KL divergence or range overlap)
- Worst-case analysis: Which (model, workload) combination has the highest MAPE? What explains it?

## BLIS E2E Claim

This sub-hypothesis reports per-step MAPE on held-out workload folds (diagnostic). Optionally, BLIS E2E error is reported on each held-out fold to assess whether LOWO degradation is acceptable for simulation. The BLIS E2E on held-out workloads provides the strongest test of production readiness.

## Related Work

- **Round 1, H3:** LOWO avg 109.7% MAPE with global XGBoost. This hypothesis tests whether per-model training + KV features reduce LOWO error.
- **MIST** (arXiv:2504.09775): Trains on controlled inputs; doesn't explicitly evaluate workload transfer. This hypothesis extends to observational workload transfer.
- **AIConfigurator** (arXiv:2601.06288): Tests serving system behavior under different arrival patterns. Analogous to workload generalization.

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| "general" workload too diverse to predict from specialized training | High | Report per-regime breakdown; if mixed-heavy regime fails, increase regime C's expressiveness (more features) |
| Reasoning workload KV lengths out-of-distribution | Medium | Check if KV feature ranges in reasoning exceed training range; if so, report extrapolation MAPE separately |
| codegen ↔ roleplay near-identical, inflating average | Low | Report per-workload breakdown, not just average. Exclude near-duplicate folds from "interesting" analysis |

## Go Integration Path

If LOWO succeeds, the per-model StepML model can be deployed as-is for any workload type without workload-specific calibration. This is the default production deployment path. If LOWO fails for specific workloads (e.g., reasoning), the `StepMLArtifact` could include workload-specific overrides for problematic cases.

## Training Strategy and Data Split

- **Training:** For each LOWO fold, use 3 workloads for the target model (3 experiments). Within these, temporal 60/20/20 for Ridge alpha tuning.
- **Test:** Held-out workload's experiment for the target model (1 experiment per fold)
- **Cross-validation:** Full 4-fold LOWO x 4 models = 16 evaluations total
- **Aggregation:** Report per-model average LOWO MAPE and overall average

## Data Integrity

- Held-out workload data is completely unseen during training for that fold
- Per-model training means each model is evaluated independently (no cross-model data leakage)
- KV features from ProgressIndex (available at inference time)
- Temporal split within training workloads prevents autocorrelation leakage within training data
- No roofline predictions used (BC-3-7)
