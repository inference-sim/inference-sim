# FINDINGS: H2 — Feature Scaling Comparison

**Date:** 2026-02-27
**Status:** REFUTED

## Claim
Proper feature scaling (StandardScaler or log-transform of KV features) applied to the Round 2 feature set makes KV features **productive** in Ridge regression, reducing MAPE below the 43.9% no-KV baseline.

## Result

| Variant | Mean MAPE | Median | Best Model | Worst Model |
|---|---|---|---|---|
| H2: Raw Ridge+KV | 86.0% | 88.8% | 18.7% (mixtral) | 187.2% (34b) |
| H2: StandardScaler+Ridge+KV | 84.5% | 84.6% | 18.7% | 187.2% |
| H2: Log-transform+Ridge+KV | 83.6% | 80.8% | 18.7% | 181.1% |
| H2: Log+Scaler+Ridge+KV | 83.0% | 80.4% | 18.7% | 180.0% |
| Baseline: No-KV Ridge | **82.0%** | 86.8% | 18.7% | 175.8% |

## Assessment

**Refuted.** ALL scaling variants (83.0-86.0%) perform **worse** than the no-KV Ridge baseline (82.0%). Feature scaling does not fix the KV feature problem.

### Root Cause Analysis

The failure has three contributing causes:

1. **KV features are zero for 2/5 models.** For llama-2-7b_tp1 and mixtral-8x7b-v0-1_tp2, the lifecycle-based KV extractor produces kv_sum=0 for all steps. These models contribute 40% of the data (15,216 + 19,088 = 34,304 out of 77,816 steps). When KV features are all zero, StandardScaler produces NaN (zero-variance column), and Ridge treats them as noise.

2. **Multicollinearity among KV features.** The 4 KV features (kv_sum, kv_max, kv_mean, kv_std) are highly correlated because they are all derived from the same underlying per-request context lengths. Ridge regression struggles with multicollinear features even with scaling.

3. **Scaling doesn't fix the wrong number of features.** Round 2's problem was not just scale mismatch — it was using too many correlated features (8+ with interactions) in a regularized model. H1 shows that the right approach is fewer features (3 coefficients with OLS), not better scaling of many features.

### Comparison with H1

H1's 3-coeff OLS (56.2% MAPE) beats all H2 variants by 26-30pp. The mechanism:
- H1 uses ONE KV feature (kv_sum) with NO regularization → clean coefficient
- H2 uses FOUR KV features with Ridge regularization → noisy coefficients

This confirms the Round 2 diagnosis: "a formulation problem, not a feature problem." The fix is not scaling, but formulation reduction.

### Ridge Coefficient Analysis

Coefficient magnitudes show the instability:
- **Raw Ridge (codellama-34b):** kv_sum=0.0296, kv_max=-1.4024 — opposite signs indicate multicollinearity
- **StandardScaler:** kv_sum=241.82, kv_max=-457.29 — huge magnitudes, cancelling out
- **Log-transform:** kv_sum=680.43, kv_max=-513.99 — even worse

The scaling amplifies the coefficient magnitudes without resolving the underlying collinearity.
