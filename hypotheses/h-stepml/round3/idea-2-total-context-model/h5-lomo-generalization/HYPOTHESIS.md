# Sub-Hypothesis H5: Leave-One-Model-Out (LOMO) Generalization

## Claim

A single OLS model trained on step-level data from 3 model groups predicts the held-out 4th model's step times with <80% per-step MAPE, improving on R2's LOMO result (108.6%).

## Rationale

Cross-model generalization is the hardest test because models span 3+ orders of magnitude in step-time scale (7B: ~100us decode, 70B: ~500us decode). R2 achieved 108.6% LOMO, a 23.6x improvement over R1 (2,559%). The FairBatching formulation's simpler feature set (new_tokens + kv_sum) may generalize better than R2's regime ensemble with more features. However, without model-specific normalization (e.g., FLOPs/token), cross-model transfer remains fundamentally limited.

## Prior Round Context

- **R1 LOMO:** 2,559.7% per-step MAPE (catastrophic)
- **R2 LOMO:** 108.6% per-step MAPE (23.6x better, still >80% target)
- **Known barrier:** 3+ OOM step-time scale variation across models

## Method

1. Load step-level data with KV features for all 10 experiments
2. 4 LOMO folds (one per model group):
   - llama-2-7b held out → train on 70B + 34B + Mixtral
   - llama-2-70b held out → train on 7B + 34B + Mixtral
   - codellama-34b held out → train on 7B + 70B + Mixtral
   - mixtral-8x7b held out → train on 7B + 70B + 34B
3. For each fold: fit 3-coeff OLS on pooled train data (temporal 60% per experiment)
4. Apply held-out model's overhead floor (R2 calibrated)
5. Evaluate per-step MAPE on held-out model's test set

## Refutation Criteria

- **Supported:** Mean LOMO MAPE < 80% (improvement over R2's 108.6%)
- **Refuted:** Mean LOMO MAPE > 150% — cross-model transfer gets worse with FairBatching formulation

## Diagnostics

- Per-fold LOMO MAPE: which model is hardest to predict?
- Coefficient comparison: pooled model vs per-model coefficients
- Step-time scale analysis: does the pooled model handle the 3+ OOM variation?
- Comparison with R2 LOMO baseline (108.6%) and R1 baseline (2,559.7%)
