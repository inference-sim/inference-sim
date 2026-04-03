# Iter26 Findings: T_tp All-Reduce Activation

## Summary

| Metric | Before (iter25) | After (iter26) | Delta |
|--------|-----------------|----------------|-------|
| Overall loss | 39.18% | 37.42% | -1.76 pts |
| TTFT RMSE | 24.34% | 24.34% | 0.00 pts |
| E2E RMSE | 15.05% | 13.09% | -1.96 pts |

Activated the T_tp (TP All-Reduce communication) basis function via β₄, then
re-optimized β₅ (per-layer overhead) which had been absorbing TP communication
cost. Two-phase golden section search, 23 total evaluations.

## Final Coefficients

```
α₀ = 15561.960    α₁ = 776.243    α₂ = 45.910

β₁ₐ = 0.138541    (prefill compute scaling)
β₂ₐ = 0.0         (decode compute scaling — absorbed by β₂ᵦ)
β₃  = 1.363060    (memory bandwidth scaling)
β₄  = 0.409533    (TP All-Reduce scaling)      ← NEW: was 0.0
β₅  = 49.626791   (per-layer overhead, µs)     ← was 62.29
β₆  = 2.797680    (overhead scaling)
β₇  = 169.365682  (attention scaling)
β₈  = 427.3       (MoE overhead)
β₁ᵦ = 0.0         (prefill memory scaling)
β₂ᵦ = 1.2632      (decode memory scaling)
```

## Phase 1: β₄ (TP All-Reduce)

Golden section search over β₄ ∈ [0.0, 0.5]:

- Converged to β₄ = 0.410
- Loss: 39.18% → 39.03% (-0.15 points)
- The optimal value is slightly above the predicted NVLink/HBM bandwidth ratio
  of ~0.27 (900 GB/s NVLink vs 3350 GB/s HBM3e). This suggests additional
  All-Reduce scheduling and synchronization overhead beyond pure NVLink
  bandwidth cost.

## Phase 2: β₅ (Per-Layer Overhead)

Golden section search over β₅ ∈ [40.0, 90.0], triggered because Phase 1
improvement exceeded the 0.1% threshold:

- Converged to β₅ = 49.63 µs/layer (was 62.29)
- Loss: 39.03% → 37.42% (-1.61 points)
- The 20% decrease in β₅ confirms the hypothesis: β₅ had been absorbing TP
  communication cost because T_tp was inactive (β₄ = 0). With β₄ now modeling
  TP communication explicitly, β₅ shrinks to reflect only true per-layer
  kernel launch and synchronization overhead.

## Error Decomposition

E2E RMSE improved significantly (15.05% → 13.09%) because TP communication
is a per-token cost that accumulates over the full decode phase. TTFT was
unchanged (24.34%) because prefill is compute-bound and single-step — the
All-Reduce cost is negligible relative to the large prefill compute.

## Generalization

The T_tp basis function includes a `(TP-1)/TP` factor that correctly:

- Predicts 0 communication cost for TP=1 models (e.g., Llama-3.2-3B on 1 GPU)
- Scales with batch size (All-Reduce volume grows with total tokens per step)
- Distinguishes dense vs MoE layers via the `2·numDenseLayers + numMoELayers`
  layer count (MoE layers have smaller hidden-dim All-Reduce)

## Coordinate Descent History (iter20–26)

| Iter | Coefficient | Before | After | Loss Before | Loss After |
|------|-------------|--------|-------|-------------|------------|
| 20 | β₈ (MoE overhead) | 0 | 440.0 | 60.11% | 40.58% |
| 21 | β₁ₐ (prefill compute) | 0.201 | 0.116 | 40.58% | 39.86% |
| 22 | β₂ (decode correction) | 1.611 | 1.146 | 39.86% | 39.42% |
| 23 | (β₁ₐ, β₂, β₈) joint | — | — | 39.42% | 39.24% |
| 24 | β₁ᵦ/β₂ᵦ decode split | — | — | 39.24% | 39.18% |
| 25 | β₈ moeScaling | — | — | 39.18% | 39.18% |
| 26a | β₄ (TP All-Reduce) | 0.0 | 0.410 | 39.18% | 39.03% |
| 26b | β₅ (per-layer overhead) | 62.29 | 49.63 | 39.03% | 37.42% |

Total improvement from iter20 to iter26: **60.11% → 37.42% (-22.69 points)**.

## Next Steps

Remaining coefficient candidates for further optimization:

- β₃ (memory bandwidth scaling) — currently 1.363, may benefit from re-tuning
  now that TP communication is properly modeled
- β₆ (overhead scaling) — 2.798, could interact with updated β₅
- Joint re-optimization of (β₄, β₅, β₆) as a 3D search
- TTFT-focused: β₁ₐ re-tuning or α coefficient adjustment to reduce 24% TTFT error
