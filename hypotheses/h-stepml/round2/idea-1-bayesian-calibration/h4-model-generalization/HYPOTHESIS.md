# Idea 1, H4: Piecewise-Linear StepTime Generalizes Across Unseen Models (LOMO)

**Status:** Refuted
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-27

## Hypothesis

> Piecewise-linear StepTime models with ProgressIndex-derived KV features (h1's approach) trained on 3 of 4 model configurations achieve <80% per-step MAPE on the held-out model, substantially improving over Round 1's catastrophic LOMO failure (2559.7% MAPE with global XGBoost). The regime structure (decode-only vs. mixed-batch) and per-request KV features capture hardware-agnostic computation patterns that transfer across model architectures.

Round 1 demonstrated that a single global model fails to generalize across models. This hypothesis tests whether the piecewise-linear regime structure -- which separates fundamentally different compute patterns -- captures enough universal structure for zero-shot deployment to unseen model+TP configurations.

## Refuted-If

- LOMO per-step MAPE > 80% averaged across all 4 holdout folds
- OR LOMO per-step MAPE > 150% for ANY single held-out model
- OR no improvement over Round 1's LOMO baseline (2559.7% average)
- OR Mixtral (MoE) holdout MAPE > 200% (MoE-to-dense transfer completely fails)

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** 4-fold leave-one-model-out (LOMO) cross-validation using h1's piecewise-linear model.

**Folds:**
| Fold | Training Models (3) | Held-Out Model |
|------|-------------------|----------------|
| 1 | Llama-2-70B, CodeLlama-34B, Mixtral-8x7B | **Llama-2-7B** (TP1) |
| 2 | Llama-2-7B, CodeLlama-34B, Mixtral-8x7B | **Llama-2-70B** (TP4) |
| 3 | Llama-2-7B, Llama-2-70B, Mixtral-8x7B | **CodeLlama-34B** (TP2) |
| 4 | Llama-2-7B, Llama-2-70B, CodeLlama-34B | **Mixtral-8x7B** (TP2, MoE) |

For each fold, train h1's piecewise-linear model (2 regimes x ~4 features) on all data from the 3 training models (pooling all 4 workloads per model = 12 experiments), then evaluate on the held-out model's 4 experiments.

**Held-out models of primary interest:**
- **Mixtral-8x7B (Fold 4):** Only MoE architecture. Dense→MoE transfer is the hardest challenge due to fundamentally different parameter activation patterns.
- **Llama-2-7B (Fold 1):** Smallest model with TP1. Tests scale-down transfer from larger TP-parallel models.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Full LOMO (entire model held out). Within training data, temporal 60/20/20 for hyperparameter tuning.
**Baselines:** (a) Round 1 global XGBoost LOMO (2559.7% avg), (b) Per-model piecewise-linear from h1 (upper bound), (c) Blackbox baseline on held-out model
**Success metric:** LOMO per-step MAPE < 80% averaged across 4 folds

**Diagnostic analysis:**
- Per-regime breakdown: Is transfer failure concentrated in decode-only or mixed-batch?
- Feature contribution: Do KV features improve LOMO accuracy vs. a piecewise-linear without them?
- Error distribution: Systematic bias direction (over/under-prediction) per held-out model?

## BLIS E2E Claim

This sub-hypothesis reports per-step MAPE on held-out model folds (diagnostic). Optionally, BLIS E2E error is also reported on each held-out fold's 4 experiments to assess whether LOMO accuracy is sufficient for meaningful simulation. However, per-step MAPE is the primary metric here since h2's BO calibration could compensate for moderate LOMO degradation.

## Related Work

- **Round 1, H3:** LOMO avg 2559.7% MAPE with global XGBoost. Established that model generalization is the hardest challenge. This hypothesis tests whether regime-based decomposition improves transfer.
- **Habitat** (Yu et al., ATC 2021): Cross-GPU performance prediction via analytical scaling achieves 9.2% median error. Demonstrates that analytical structure aids hardware transfer.
- **MIST** (arXiv:2504.09775): Trains per-model but doesn't evaluate cross-model. This hypothesis extends MIST's approach to the transfer setting.
- **Vidur** (MLSys 2024): Requires per-model profiling data. This hypothesis tests whether profiling-free transfer is feasible.

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Mixtral MoE transfer fails completely | High | Report Mixtral separately; if MoE MAPE > 200%, conclude that MoE requires dedicated training and focus on dense-to-dense transfer |
| TP degree differences dominate | Medium | Include TP degree as a feature in the piecewise-linear model; ablation with/without TP feature |
| Training data imbalance across models | Low | Normalize sample counts per model to avoid majority-model bias |

## Go Integration Path

If LOMO succeeds (<80% MAPE), a single set of piecewise-linear coefficients could serve as a "cold-start" model for new model configurations, refined via online calibration. The `StepMLArtifact` JSON would include a `universal_fallback` model alongside per-model models.

## Training Strategy and Data Split

- **Training:** For each LOMO fold, pool all 4 workloads from the 3 training models (12 experiments). Within this pool, temporal 60/20/20 split for Ridge alpha tuning.
- **Test:** All 4 experiments from the held-out model (no temporal split needed -- entire experiments are held out)
- **Cross-validation:** Full 4-fold LOMO; report per-fold and average MAPE

## Data Integrity

- Held-out model data is completely unseen during training (no temporal overlap possible since entire model is excluded)
- KV features from ProgressIndex (available at inference time, no look-ahead)
- No model-identifying features (model name, parameter count) used as inputs -- this tests whether computation-level features alone suffice for transfer
- No roofline predictions used (BC-3-7)
