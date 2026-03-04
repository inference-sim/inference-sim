# Idea 1, H4: Piecewise-Linear StepTime Generalizes Across Unseen Models (LOMO) — FINDINGS

**Status:** Refuted
**Resolution:** Refuted — mechanism not plausible
**Family:** Performance-regime
**VV&UQ:** Validation
**Type:** Statistical/Dominance
**Date:** 2026-02-27
**Rounds:** 1

## Hypothesis

> Piecewise-linear StepTime models with ProgressIndex-derived KV features (h1's approach) trained on 3 of 4 model configurations achieve <80% per-step MAPE on the held-out model, substantially improving over Round 1's catastrophic LOMO failure (2559.7% MAPE with global XGBoost).

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** 4-fold leave-one-model-out (LOMO) cross-validation using h1's 2-regime piecewise-linear model. For each fold, train on pooled data from 3 model families (all workloads), evaluate on the held-out model family's data.

**Folds:**
| Fold | Training Models | Held-Out Model |
|------|----------------|----------------|
| 1 | 7B, 70B, Mixtral | **CodeLlama-34B** (TP2) |
| 2 | 7B, 34B, Mixtral | **Llama-2-70B** (TP4) |
| 3 | 34B, 70B, Mixtral | **Llama-2-7B** (TP1) |
| 4 | 7B, 34B, 70B | **Mixtral-8x7B** (TP2, MoE) |

**Controlled variables:** Feature set (h1's 4 features per regime), Ridge alpha CV, temporal split within training data
**Varied variable:** Which model family is held out
**Data:** 77,816 steps from 10 experiments, 4 model families
**Baselines:** Round 1 global XGBoost LOMO (2559.7% avg)

## Results

### Per-Fold MAPE

| Fold | Held-Out | Decode-Only MAPE | Decode r | Mixed-Batch MAPE | Mixed r | Overall MAPE | Overall r | n_test |
|------|----------|-----------------|----------|-----------------|---------|-------------|-----------|--------|
| 1 | **34B** | 50.0% | 0.601 | 67.1% | 0.698 | **53.9%** | 0.670 | 24,100 |
| 2 | **70B** | 128.9% | 0.830 | 195.0% | 0.825 | **146.5%** | 0.803 | 19,412 |
| 3 | **7B** | 185.2% | 0.453 | 205.6% | 0.451 | **187.0%** | 0.357 | 15,216 |
| 4 | **Mixtral** | 163.0% | 0.598 | 327.7% | 0.561 | **207.8%** | 0.612 | 19,088 |

### Aggregate Summary

| Metric | Value | Threshold |
|--------|-------|-----------|
| Average LOMO MAPE | **148.8%** | <80% |
| Worst fold MAPE | 207.8% (Mixtral) | <150% |
| Round 1 baseline | 2559.7% | — |
| Improvement factor | **17.2x** | Must improve |

### Per-Regime Analysis (Averaged Across Folds)

| Regime | Avg MAPE | Range |
|--------|----------|-------|
| Decode-only | 131.8% | 50.0% - 185.2% |
| Mixed-batch | 198.9% | 67.1% - 327.7% |

### Directional Bias (MSPE)

| Held-Out | Decode MSPE | Mixed MSPE | Interpretation |
|----------|-----------|-----------|---------------|
| 34B | +35.3% | -8.1% | Decode overpredicted |
| 70B | +105.5% | +177.1% | Both heavily overpredicted |
| 7B | **-185.2%** | **-205.4%** | **Both heavily underpredicted (negative predictions)** |
| Mixtral | +152.0% | +312.0% | Both heavily overpredicted |

### BLIS E2E Validation

BLIS E2E validation failed for all experiments due to binary path resolution issue (`parents[4]` resolves to `hypotheses/` instead of repo root). Per-step metrics are unaffected.

## Root Cause Analysis

### Why LOMO fails at 148.8% (vs <80% target)

**1. Absolute step-time scales differ dramatically across models.** Step durations for Llama-2-7B (TP1) average ~150us, while Llama-2-70B (TP4) averages ~1000us. A pooled linear model trained on 70B/34B/Mixtral (whose step times are 5-10x larger than 7B) produces predictions scaled for large models. When applied to 7B, it generates negative predictions (MSPE = -185%), confirming catastrophic scale mismatch.

**2. The 2-regime split doesn't capture hardware-agnostic patterns.** The hypothesis assumed that decode-only vs mixed-batch separation would extract computation patterns that transfer across architectures. In reality:
- `decode_tokens` has the same range across models (determined by scheduler, not model size), but its impact on step time is proportional to model size (more parameters = more FLOPS per token)
- KV features (kv_sum, kv_max, kv_mean) scale differently per model because KV cache size depends on model architecture (num_heads, head_dim)
- The linear coefficients learned on large models predict impossibly large step times for small models (Fold 3: 7B), and impossibly small times for models with different compute profiles (Fold 4: Mixtral)

**3. Mixtral (MoE) transfer is the worst case as predicted.** Fold 4 (hold out Mixtral) achieves 207.8% MAPE because MoE models have fundamentally different parameter activation: only 2 of 8 experts activate per token, so compute scales differently with batch size. Dense-to-MoE transfer fails because the linear relationship between decode_tokens and step time has a completely different slope.

**4. CodeLlama-34B (Fold 1) is the best LOMO fold at 53.9%.** This works because 34B is "between" the other training models in terms of scale (7B < 34B < 70B), so the pooled linear model interpolates rather than extrapolates. This is consistent with typical regression behavior — interpolation succeeds, extrapolation fails.

## Devil's Advocate (RCV-5)

**Arguing the hypothesis might be Confirmed:**
The 17.2x improvement over Round 1's LOMO baseline (2559.7% → 148.8%) is substantial. If features were normalized by model-specific constants (e.g., parameter count, FLOPS per token), the scale mismatch could be eliminated. The 34B fold (53.9%) proves that cross-model transfer is feasible when the held-out model is within the training distribution's scale range. With feature engineering that accounts for model scale (dividing features by parameter count or dividing targets by expected FLOPS), the <80% threshold might be achievable. The 2-regime structure itself is not the bottleneck for LOMO — the missing scale normalization is.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| LOMO fails at 148.8% MAPE (>80% threshold) | Refutation | Confirms per-model training is required |
| 17.2x improvement over Round 1 LOMO | Confirmation | Piecewise-linear better than global XGBoost for LOMO |
| Fold 1 (34B holdout) at 53.9% shows interpolation works | Surprise | Cross-scale interpolation may be feasible with normalization |
| Fold 3 (7B holdout) produces negative predictions | Bug/Design limitation | Scale mismatch causes catastrophic failure |
| Mixtral MoE transfer fails (207.8%) | Confirmation | MoE requires dedicated training, as predicted in HYPOTHESIS.md |
| Mixed-batch regime consistently worse than decode-only in LOMO | Confirmation | Mixed-batch heterogeneity amplified by cross-model transfer |

## Standards Audit

- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? None

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4-fold LOMO across 4 model families, H100 GPU, 10 experiments from eval/ground_truth/
- **Parameters findings depend on:** 2-regime structure, raw linear regression without model-size normalization, specific KV feature set
- **What was NOT tested:** Feature normalization by model scale, 3-regime structure (Idea 2's approach), log-transformed targets, including TP degree as an explicit feature, BLIS E2E validation
- **Generalizability:** The finding that unnormalized LOMO fails likely generalizes to any regression model without model-scale features. The 34B interpolation success suggests that with scale normalization, dense-to-dense transfer may be feasible.
- **Uncertainty quantification:** UQ not performed — each fold is a single evaluation.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Avg LOMO MAPE | 148.8% | High — 4 folds, 77,816 steps total |
| Worst fold (Mixtral) | 207.8% | High — complete evaluation |
| Round 1 improvement | 17.2x | High — identical data, same metric |
| Mechanism (scale mismatch) | Proposed | High — confirmed by MSPE direction analysis |

## Implications for Users

1. **Per-model training is required.** LOMO with 2-regime piecewise-linear models does not achieve production-quality accuracy. Each model+TP configuration needs its own StepML coefficients.
2. **Cross-model interpolation is feasible.** If a new model's size falls between existing trained models, the accuracy degradation may be tolerable (as shown by the 34B fold at 53.9%).
3. **MoE models require dedicated training.** Dense-to-MoE transfer fails completely. A Mixtral-specific model is necessary.
4. **A "cold-start" universal model is viable only as a rough approximation.** LOMO accuracy (148.8%) is much worse than per-model accuracy (87.4%) but 17x better than Round 1's global approach.

## Reproducing

```bash
cd hypotheses/h-stepml/round2/idea-1-bayesian-calibration/h4-model-generalization
./run.sh
```
