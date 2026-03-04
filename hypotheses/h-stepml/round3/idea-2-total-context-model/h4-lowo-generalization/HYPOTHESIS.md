# Sub-Hypothesis H4: Leave-One-Workload-Out (LOWO) Generalization

## Claim

The FairBatching 3-coefficient OLS model (`a + b*new_tokens + c*kv_sum`) trained per-model on 2 workloads generalizes to the held-out 3rd workload with <25% per-step MAPE degradation relative to the in-distribution test set.

## Rationale

Round 2 tested LOWO on the regime ensemble and found 117.4% per-step MAPE — suggesting poor cross-workload transfer. However, the FairBatching formulation (Idea 2 H1) uses a simpler, more regularized model with only 3 coefficients. Simpler models typically generalize better. If the step-time relationship is fundamentally driven by new_tokens and kv_sum regardless of workload, LOWO should succeed.

## Prior Round Context

- **R2 LOWO:** 117.4% per-step MAPE (regime ensemble with KV features)
- **R2 LOMO:** 108.6% per-step MAPE
- **Mixtral LOWO:** 19.1% in R2 (exceptional cross-workload transfer)

## Method

1. Load step-level data with KV features for all 10 experiments
2. For each model_tp with 3+ workloads:
   - 3 folds: train on 2 workloads, evaluate on held-out 3rd
   - Fit 3-coeff OLS (`a + b*new_tokens + c*kv_sum`) on train workloads
   - Also fit 4-coeff OLS (`a + b*prefill + c*decode + d*kv_sum`)
   - Apply overhead floor from R2 calibrated artifacts
   - Compute per-step MAPE on held-out workload
3. Report per-model × per-fold results + aggregate mean LOWO MAPE

## Refutation Criteria

- **Supported:** Mean LOWO MAPE < 70% (better than R2's 117.4%) across all folds
- **Refuted:** Mean LOWO MAPE > 100% — the FairBatching formulation does not generalize across workloads

## Diagnostics

- Per-fold MAPE comparison: in-distribution vs LOWO
- Coefficient stability: how much do coefficients change between workload training sets?
- Per-model LOWO analysis: which models generalize best/worst?
- Comparison with R2 LOWO baseline (117.4%)
