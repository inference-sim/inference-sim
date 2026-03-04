# Idea 2: Normalized Feature Space with Scale-Invariant Regression

## Hypothesis

A SINGLE Ridge regression trained on normalized step features across ALL models can achieve <10% mean E2E by factoring out model-specific scale (overhead) and learning shared batch-composition→performance relationships.

## Approach

1. Estimate per-model overhead from metadata: `overhead_est = base × (params/7e9)^0.3`
2. Calibrate `base` from training models' E2E-derived target step times
3. Create normalized features:
   - `norm_decode = decode_tokens / max_num_seqs` (relative batch fullness)
   - `norm_prefill = prefill_tokens / max_num_batched_tokens` (relative prefill load)
4. Train SINGLE Ridge: `step_time / overhead_est = 1 + w1*norm_decode + w2*norm_prefill`
5. Map back to BlackboxLatencyModel: `beta0 = overhead_est`, `beta2 = overhead_est*w1/max_num_seqs`, `beta1 = overhead_est*w2/max_batched_tokens`

## Sub-Hypotheses

### H1: Unified Normalized Ridge (All Models)

**Claim:** Normalizing step data by metadata-derived overhead removes 3+ OOM scale variation, enabling a single Ridge to learn cross-model batch dynamics.

**Pass criteria:** BLIS E2E <10% mean across all 10 experiments with ONE set of Ridge weights.

### H2: LOMO (Leave-One-Model-Out)

**Claim:** Train normalized Ridge on 3 models' step data. For held-out model, compute metadata overhead and apply shared Ridge weights. The normalized features (relative batch fullness) should transfer.

**Pass criteria:** Mean LOMO E2E <80% across 4 folds.

### H3: LOWO (Leave-One-Workload-Out)

**Claim:** Same model for all workloads. Normalized features are workload-invariant since they capture relative batch composition, not absolute token counts.

**Pass criteria:** Per-model workload variance <50%.

## Prior Round References

- R1: Global XGBoost failed (2,559.7%) due to 3+ OOM scale variation — this approach addresses scale via normalization
- R2: Regime ensemble partially worked (108.6%) but didn't normalize away model scale
- R4: beta0 dominates (92-98%) — if we predict overhead correctly, the rest is noise
- R4: beta0 ≈ 9741 × (params/7e9)^0.30 fits approximately

## Risk Assessment

**Primary risk:** Training on step.duration_us (GPU-only) may not capture the overhead-dominated cycle time correctly.
**Mitigation:** Use E2E-derived target step times as training targets (lifecycle data provides this).
**LOMO risk:** For held-out model, overhead estimation relies on metadata formula accuracy.
