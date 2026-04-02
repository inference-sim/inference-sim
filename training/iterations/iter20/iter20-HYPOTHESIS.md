# Iteration 20: β₈·nMoELayers — Per-MoE-Layer Overhead Term

## Context

Iterations 17–19 confirmed that the 7-term formula has a loss floor of ~60% on 15
experiments. Diagnostic analysis revealed the bottleneck: all 4 Scout MoE experiments
systematically under-predict by ~50% (TTFT, E2E, ITL uniformly low). The model predicts
roughly half of actual Scout latency. Dense models (11/15) are well-fit (1–35% APE).

The missing cost is per-step, per-MoE-layer: router gating computation, token scatter/
gather permutation, expert load imbalance, and EP all-to-all communication — none of
which apply to dense layers.

## H-main: Loss Drops Below 55% With β₈·nMoELayers

**Prediction**: Overall loss will drop from 60.11% (iter19 best) to below 55%, with:
- Scout mean TTFT APE < 30% (currently 42–73%)
- Dense model APE unchanged within ±1pp (β₈×0 = 0 for all dense models)
- β₈ converges to 60–120 µs/MoE-layer (comparable to β₅ ≈ 62 µs/layer)

**Causal Mechanism**: The new term `β₈ × nMoELayers` adds a per-step overhead
proportional to the number of MoE layers. For Scout (nMoELayers=24), this adds
`β₈ × 24` µs per step — directly addressing the ~50% under-prediction. For dense
models (nMoELayers=0), the term is exactly zero, preserving the existing fit.

**Diagnostic Clause**: If loss does not drop below 55%:
- β₈ < 30µs → the bound may be too tight or the optimizer hasn't converged
- β₈ > 200µs → the model is over-correcting; check whether dense predictions degraded
- Dense APE increased > 2pp → β₈ is coupling with other terms (should not happen since
  the basis function is zero for dense models, but check β₁-β₇ for drift)

## H-dense-unchanged: Dense Model Predictions Identical

**Prediction**: All 11 dense model experiment APE values are within ±0.5pp of iter19
values. The β₈·nMoELayers term is mathematically zero for dense models (nMoELayers=0),
so the optimized β₁-β₇ should remain at their iter19 values.

**Diagnostic Clause**: If dense APE changes > 1pp, the optimizer is trading dense
accuracy for Scout accuracy via β₁-β₇ shifts. This would indicate the β₈ term alone is
insufficient and the formula needs a more expressive MoE correction.

## H-beta8-physical: β₈ Is Physically Plausible

**Prediction**: β₈ converges to 60–120 µs/MoE-layer, which is physically interpretable:
- Router gating: ~10µs (small matmul per token per MoE layer)
- Token permutation (scatter/gather): ~20-30µs
- Expert load imbalance idle time: ~10-20µs
- EP all-to-all communication: ~20-40µs

For Scout with 24 MoE layers: `80 × 24 = 1920 µs/step` additional overhead, which is
roughly the ~50% of step time currently missing.

**Diagnostic Clause**: If β₈ > 200 µs, it is absorbing costs beyond MoE overhead
(possibly compensating for other model errors). If β₈ < 20 µs, the MoE overhead is
too small to explain the 50% under-prediction.
