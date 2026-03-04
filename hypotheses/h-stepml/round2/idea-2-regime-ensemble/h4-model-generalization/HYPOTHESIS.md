# Idea 2, H4: Regime-Switching Ridge Ensemble Generalizes Across Unseen Models (LOMO)

**Status:** Refuted
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-27

## Hypothesis

> The 3-regime Ridge ensemble with ProgressIndex-derived KV features (h1's approach) trained on 3 of 4 model configurations achieves <80% per-step MAPE on the held-out model, representing a >30x improvement over Round 1's catastrophic LOMO failure (2559.7% MAPE). The regime decomposition (decode-only / mixed-light / mixed-heavy) captures universal computation patterns -- memory-bandwidth-bound decode, compute-bound prefill -- that transfer across model architectures and TP configurations.

Per-model training is the expected production strategy. This hypothesis tests the fallback scenario: deploying to a new model+TP configuration before per-model training data is available. If LOMO MAPE < 80%, the regime ensemble provides a viable cold-start model.

## Refuted-If

- LOMO per-step MAPE > 80% averaged across all 4 holdout folds
- OR LOMO per-step MAPE > 150% for ANY single held-out model
- OR no improvement over Round 1's LOMO baseline (2559.7% average)
- OR KV features provide <10% LOMO improvement vs. regime Ridge without KV features (KV features don't help transfer)
- OR regime-specific LOMO MAPE for decode-only (Regime A) > 60% (the dominant regime must transfer)

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** 4-fold leave-one-model-out (LOMO) cross-validation using h1's 3-regime Ridge model.

**Folds:**
| Fold | Training Models (3) | Held-Out Model | Transfer Challenge |
|------|-------------------|----------------|-------------------|
| 1 | Llama-2-70B, CodeLlama-34B, Mixtral-8x7B | **Llama-2-7B** (TP1) | Large→small, multi-TP→single-TP |
| 2 | Llama-2-7B, CodeLlama-34B, Mixtral-8x7B | **Llama-2-70B** (TP4) | Small→large, low-TP→high-TP |
| 3 | Llama-2-7B, Llama-2-70B, Mixtral-8x7B | **CodeLlama-34B** (TP2) | Llama-family→Code-family |
| 4 | Llama-2-7B, Llama-2-70B, CodeLlama-34B | **Mixtral-8x7B** (TP2, MoE) | Dense→MoE (hardest) |

For each fold, train 3 regime-specific Ridge regressors on all data from the 3 training models (12 experiments), then evaluate each regime on the held-out model's 4 experiments.

**Feature engineering:** Same as h1 (decode-only: 7 features including KV; mixed: 8 features). Additionally, test including `tp_degree` as a feature to aid cross-TP transfer.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Full LOMO (entire model held out). Within training data, temporal 60/20/20 for Ridge alpha tuning via 5-fold CV.
**Baselines:** (a) Round 1 LOMO XGBoost (2559.7% avg), (b) Per-model regime Ridge from h1 (upper bound), (c) Blackbox LOMO, (d) Regime Ridge LOMO without KV features (ablation)
**Success metric:** LOMO per-step MAPE < 80% averaged across 4 folds

**Ablation studies:**
- With/without KV features in LOMO setting
- With/without `tp_degree` as an additional feature
- 3-regime vs. 2-regime (decode-only vs. any-mixed) in LOMO setting
- Per-regime LOMO MAPE breakdown

## BLIS E2E Claim

This sub-hypothesis reports per-step MAPE on held-out model folds (diagnostic). Additionally, BLIS E2E error is computed on each held-out model's 4 experiments using the LOMO-trained model, to assess whether cross-model transfer produces usable simulation results. If LOMO BLIS E2E < 50%, the cold-start model has practical utility.

## Related Work

- **Round 1, H3:** LOMO avg 2559.7% MAPE, LOWO avg 109.7%. Established that cross-model transfer is the harder challenge by >20x. This hypothesis tests whether regime decomposition bridges the gap.
- **MIST** (arXiv:2504.09775): Per-model training; doesn't evaluate cross-model transfer. The regime decomposition is inspired by MIST but applied to the transfer setting.
- **Habitat** (Yu et al., ATC 2021): Cross-GPU prediction via analytical scaling (9.2% median error). Demonstrates that structuring the model around physical computation properties aids transfer.
- **Vidur** (MLSys 2024): Requires per-model profiling. This tests profiling-free deployment.

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Mixtral MoE has fundamentally different step-time scaling | High | Report Mixtral separately; if MoE MAPE > 200%, exclude from average and conclude MoE needs dedicated training |
| TP degree causes step-time scale differences | High | Include `tp_degree` as feature in ablation; check if normalizing step times by TP helps |
| Model size affects absolute step-time scale | Medium | Test log-transform of target variable to handle scale differences |
| Imbalanced training data across models | Low | Weight samples inversely proportional to model frequency, or subsample to equal counts |

## Go Integration Path

If LOMO succeeds, the `StepMLArtifact` JSON includes both per-model models (for known configurations) and a `universal_fallback` model (for cold-start deployment). `StepMLLatencyModel` checks whether a per-model artifact exists for the requested model; if not, falls back to the universal model. This enables zero-configuration deployment for new models.

## Training Strategy and Data Split

- **Training:** For each LOMO fold, pool all 4 workloads from 3 training models (12 experiments). Within this pool, temporal 60/20/20 for Ridge alpha tuning.
- **Test:** All 4 experiments from the held-out model
- **Cross-validation:** Full 4-fold LOMO; report per-fold and average MAPE
- **Sample counts:** ~92,000 training steps per fold (3/4 of 122,752); ~30,000 test steps

## Data Integrity

- Held-out model data is completely unseen during training
- No model-identifying features (model name, parameter count) used as inputs
- KV features from ProgressIndex (available at inference time)
- Regime classification uses only step-level features (prefill_tokens), not model metadata
- No roofline predictions used (BC-3-7)
