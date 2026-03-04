# FINDINGS: H5 — Leave-One-Model-Out (LOMO) Generalization

**Date:** 2026-02-28
**Status:** REFUTED

## Claim

A single OLS model trained on step-level data from 3 model groups predicts the held-out 4th model's step times with <80% per-step MAPE, improving on R2's 108.6%.

## Result

**Mean LOMO MAPE: 2,281.6%** (threshold <80%). All folds far exceed the threshold and are 21x worse than R2.

## Per-Fold LOMO Results

| Holdout Model | LOMO MAPE | In-Dist MAPE | Train Models |
|---|---|---|---|
| codellama-34b | 2,121.1% | 2,121.1% | llama-2-7b, llama-2-70b, mixtral |
| llama-2-70b | 1,408.7% | 979.2-1,838.2% | llama-2-7b, codellama-34b, mixtral |
| llama-2-7b | 2,985.2% | 2,985.2% | llama-2-70b, codellama-34b, mixtral |
| mixtral-8x7b-v0-1 | 2,611.5% | 2,611.5% | llama-2-7b, llama-2-70b, codellama-34b |

## Pooled Coefficient Comparison

| Holdout | Intercept | new_tokens coeff | kv_sum coeff |
|---|---|---|---|
| codellama-34b | 60.4 | 4.6521 | 0.027322 |
| llama-2-70b | 70.6 | 6.4408 | 0.041168 |
| llama-2-7b | 12.9 | 6.8376 | 0.028066 |
| mixtral-8x7b-v0-1 | 19.0 | 6.4959 | 0.028345 |

Coefficients vary 5.4x for intercepts and 1.5x for new_tokens across folds. The instability indicates model-specific step-time scales cannot be captured by a single pooled model.

## Root Cause

The FairBatching 3-coeff formulation lacks:
1. **Regime separation** — R2's decode-only vs mixed-batch split was critical for LOMO (achieved 108.6%)
2. **Model-scale normalization** — step times span 3+ OOM across models (7B ~100us vs 70B ~500us decode)
3. **Numerical stability** — pooling data with kv_sum=0 (llama-2-7b, mixtral) alongside kv_sum up to 64,000 causes matmul overflow

## Comparison with Prior Rounds

| Round | LOMO MAPE | Approach |
|---|---|---|
| R1 | 2,559.7% | Per-experiment XGBoost |
| R2 | 108.6% | Regime ensemble + KV features |
| **R3 Idea 2** | **2,281.6%** | FairBatching 3-coeff OLS |

R3 Idea 2 is comparable to R1 (no regime structure) and 21x worse than R2 (which had regime structure). The regime structure is the critical ingredient for LOMO, not feature engineering.

## Refutation Assessment

- **REFUTED:** 2,281.6% >> 80% threshold
- The FairBatching formulation completely loses cross-model transfer capability
- Regime separation is mandatory for both LOWO and LOMO
