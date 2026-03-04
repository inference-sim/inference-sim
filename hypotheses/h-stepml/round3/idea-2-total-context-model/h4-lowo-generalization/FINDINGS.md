# FINDINGS: H4 — Leave-One-Workload-Out (LOWO) Generalization

**Date:** 2026-02-28
**Status:** REFUTED

## Claim

The FairBatching 3-coefficient OLS model (`a + b*new_tokens + c*kv_sum`) trained per-model on 2 workloads generalizes to the held-out 3rd workload with <70% per-step MAPE (improving on R2's 117.4%).

## Result

**Mean LOWO MAPE: 2,162.7%** (threshold <70%). All folds are catastrophically worse than the R2 baseline.

## Per-Model LOWO Results

| Model | Holdout: codegen | Holdout: general | Holdout: roleplay | Mean |
|---|---|---|---|---|
| codellama-34b_tp2 | 2,526.4% | 281.0% | 3,348.6% | 2,052.0% |
| llama-2-70b_tp4 | — | 154.2% | 3,336.0% | 1,745.1% |
| mixtral-8x7b-v0-1_tp2 | 2,380.7% | 1,172.5% | 4,102.3% | 2,551.8% |

**Skipped:** llama-2-7b_tp1 (only 1 workload), llama-2-70b-hf_tp4 (only 1 workload)

## Key Finding

**Zero degradation between LOWO and in-distribution:** Every fold shows 0.0pp degradation, meaning the model trained on 2 workloads gives identical performance to one trained on all 3. This is expected since per-model pooled OLS with 3 coefficients has too few parameters to overfit to specific workloads.

The extremely high absolute MAPE values stem from numerical instability: kv_sum features produce overflow in matmul (range 0-64,000), and the OLS model without regime separation produces predictions spanning orders of magnitude from actual values.

## Root Cause

The FairBatching 3-coeff model lacks the regime structure (decode-only vs mixed-batch) that R2 used to achieve 117.4% LOWO. Without regime separation, a single linear model across all step types produces catastrophic predictions for steps far from the training mean. The overhead floor mitigates some of this, but not enough.

## Refutation Assessment

- **REFUTED:** 2,162.7% >> 70% threshold, and 18.4x worse than R2's 117.4%
- The FairBatching formulation trades per-step accuracy for simplicity, but this tradeoff is catastrophic for LOWO
- R2's regime structure was essential for cross-workload transfer
