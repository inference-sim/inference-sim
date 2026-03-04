# Idea 1: Analytical Overhead Model with Metadata-Derived Coefficients

## Hypothesis

All four BlackboxLatencyModel coefficients (alpha0, beta0, beta1, beta2) can be predicted from model architecture metadata alone — parameter count, TP degree, and architecture type — achieving <10% mean E2E across all 10 experiments with ZERO per-model calibration data.

## Approach

Use R4's 4 calibrated coefficient sets as "training data" for a meta-regression:

1. **beta0 = a × (params/1e9)^b × f(tp)** — Power law fit from 4 known models
2. **beta2 = c × (params_per_gpu/1e9)^d** — Derived from per-GPU parameter density
3. **beta1 ≈ 0** — Negligible for all models (R4: 0.00–1.22)
4. **alpha0 = e × (params/1e9)^f** — Power law from 4 known TTFTs

## Key Data Points (R4 Production Coefficients)

| Model | Params (B) | TP | Active Params (B) | alpha0 (μs) | beta0 (μs) | beta1 | beta2 |
|-------|-----------|-----|-------------------|-------------|------------|-------|-------|
| llama-2-7b | 7 | 1 | 7 | 27,129 | 9,741 | 0.30 | 13.6 |
| codellama-34b | 34 | 2 | 34 | 47,618 | 14,196 | 0.00 | 25.8 |
| llama-2-70b | 70 | 4 | 70 | 78,888 | 17,992 | 1.22 | 35.2 |
| mixtral-8x7b | 46.7 | 2 | 12.9 | 62,767 | 18,921 | 0.69 | 8.8 |

## Sub-Hypotheses

### H1: Power Law Meta-Regression (All Models)

**Claim:** Power law fits beta0, alpha0, beta2 from model metadata with <5% coefficient prediction error for the 4 training models.

**Pass criteria:** Predicted coefficients within 20% of R4 values for all 4 models. BLIS E2E <10% mean.

### H2: LOMO (Leave-One-Model-Out)

**Claim:** Fit meta-regression on 3 models, predict held-out model's coefficients, achieve <80% E2E per fold.

**Pass criteria:** Mean LOMO E2E <80% across 4 folds.

### H3: LOWO (Leave-One-Workload-Out)

**Claim:** Since coefficients are metadata-derived (no workload info), they're workload-invariant by design.

**Pass criteria:** Per-model workload variance <50%.

## Prior Round References

- R4 Idea 3 H1: Direct calibration produced these coefficients (5.7% mean E2E)
- R4 BC-4-3: LOMO regression at 30.7% — the gap R5 aims to close
- beta0 scaling: ~O(params^0.3) — 9.7ms (7B) → 14.2ms (34B) → 18.0ms (70B) → 18.9ms (Mixtral)
- GPU compute is only 2-8% of step time; beta0 dominates

## Risk Assessment

**Primary risk:** Only 4 data points for meta-regression — underdetermined for any model with >2 parameters.
**Mitigation:** Use simple power laws with 2 parameters each, validated via LOMO.
**Mixtral risk:** MoE architecture has different compute patterns — beta2=8.8 (much lower than dense models).
