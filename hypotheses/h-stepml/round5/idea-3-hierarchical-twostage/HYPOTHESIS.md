# Idea 3: Hierarchical Two-Stage Model (Shared Physics + Model-Specific Intercept)

## Hypothesis

Separating step time prediction into (1) a metadata-derived base overhead and (2) a shared cross-model delta model for batch-composition effects can achieve <10% mean E2E with a single set of delta-model weights.

## Approach

**Stage 1 (Shared across models):** Learn how step time VARIES with batch composition using pooled step data from all models. Train on residuals: `delta = step_time - per_model_mean_step`.

**Stage 2 (Metadata-derived):** Predict the absolute base overhead from model architecture: `base_overhead = f(params, tp, is_moe)`.

**Prediction:** `step_time = base_overhead(metadata) + delta_model(decode_tokens, prefill_tokens)`

## Sub-Hypotheses

### H1: Shared Delta Model (All Models)

**Claim:** Step-time variation around the mean follows similar patterns across models — more tokens → more time, with a shared slope.

**Pass criteria:** BLIS E2E <10% mean across all 10 experiments.

### H2: LOMO (Leave-One-Model-Out)

**Claim:** Train delta model on 3 models' step residuals. For held-out model, use metadata overhead + shared delta weights. Tests whether residual patterns are truly shared.

**Pass criteria:** Mean LOMO E2E <80% across 4 folds.

### H3: LOWO (Leave-One-Workload-Out)

**Claim:** Same two-stage model for all workloads. Workload-invariant by design since metadata overhead doesn't include workload info and delta captures batch dynamics generally.

**Pass criteria:** Per-model workload variance <50%.

## Prior Round References

- R3 CMA-ES artifact transfer (14.8%) — simulation dynamics transfer better than raw coefficients
- R4: GPU compute (beta2 × batch) is tiny (2-8%) vs beta0 (92-98%) — separating them aligns with the physics
- R2: Additive overhead model had phase transitions — this approach uses residuals around mean, not additive model

## Risk Assessment

**Primary risk:** The delta (variation) is so small (2-8% of step time) that even small errors in base_overhead dominate the prediction.
**Mitigation:** Use E2E-derived mean step time for base overhead during training; for LOMO, metadata formula.
**Structural risk:** If batch-composition effects are model-specific (not shared), the delta model won't transfer.
