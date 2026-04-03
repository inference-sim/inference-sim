# Iteration 26: Physics-Based T_tp (TP All-Reduce)

## Context

Iterations 20–25 reduced loss from 60.19% to 39.18% via β₈ (MoE overhead) and
prefill/decode roofline splits. Throughout, β₄·T_tp = 0 — the TP All-Reduce
communication was entirely ignored, with β₅·L absorbing any residual TP cost.

Iter26 activates T_tp with a physics-derived formula:
  T_tp = (2·numDenseLayers + numMoELayers) × totalTokens × d × 2B × 2phases × (TP-1)/TP / bwHBM

Dense layers contribute 2 All-Reduce units (attention + FFN).
MoE layers contribute 1 unit (attention only; FFN uses EP All-to-All captured by β₈).
TP=1 → tTp=0 (backward compat).

## H-main: β₄ Converges to ~0.27 (NVLink/HBM Ratio on H100)

**Prediction**: β₄ converges to approximately 0.25–0.35 (the NVLink bandwidth / HBM
bandwidth ratio for H100: 900 GB/s / 3.35 TB/s ≈ 0.27).

**Causal Mechanism**: T_tp is normalized by bwHbmUs, so β₄ absorbs the actual
interconnect/memory bandwidth ratio. If TP All-Reduce is the dominant communication
cost and NVLink bandwidth is the bottleneck, β₄ ≈ 0.27.

**Diagnostic Clause**: If β₄ converges near 0, TP communication is negligible vs
the other terms. If β₄ > 0.5, the T_tp formula undercounts All-Reduce volume or
β₅ has been over-compensating.

## H-beta5: β₅ May Decrease After T_tp Activation

**Prediction**: β₅ (per-layer overhead, currently 62.3 µs/layer) may decrease
slightly because it was previously absorbing TP communication overhead for
TP=2 and TP=4 experiments.

**Diagnostic Clause**: Run golden section on β₅ after β₄ to check for drift.

## H-loss: Loss Holds or Improves Marginally

**Prediction**: Overall loss stays within ±0.5% of 39.18%. The T_tp formula
has the right physics but the training data (all H100, fixed NVLink topology)
provides limited leverage to improve Scout/dense APE further via TP correction.
The main benefit is formula correctness for new hardware/TP configurations.
