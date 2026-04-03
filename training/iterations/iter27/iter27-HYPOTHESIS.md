# Iteration 27: Joint CMA-ES 6-Parameter Optimization

## Context

Iterations 20–26 used coordinate descent (one coefficient at a time), reaching 37.42%
loss. Each golden section search held all other coefficients fixed, missing interaction
effects between parameters.

Iter26 activated T_tp (TP All-Reduce) and found β₄=0.410 via golden section, then β₅=49.6
via a second search. These were sequential and independent — but the optimal β₄ depends on
β₅, which depends on β₇, which interacts with β₈. A joint search can capture these.

Iter27 runs CMA-ES jointly over the 6 most recently-changed or likely-to-shift parameters:
β₁ₐ, β₄, β₅, β₇, β₈, β₂ᵦ — all with iter26 as warm start.

## H-main: Joint Optimization Beats Coordinate Descent

**Prediction**: Loss drops below iter26 (37.42%) by at least 0.5 points.

**Causal mechanism**: Coordinate descent missed the β₄/β₅/β₇ interaction. When T_tp was
activated (iter26), β₄ was calibrated while β₅ and β₇ were frozen. But β₅ (per-layer) and
β₇ (per-step constant) were previously absorbing TP communication overhead — once β₄ can
jointly increase, β₅ and β₇ should jointly decrease to compensate. A joint search captures
this trade-off; sequential golden section cannot.

**Diagnostic clause**: If loss does not improve, coordinate descent had already reached the
joint optimum (unlikely given the known β₄/β₅ coupling from iter26).

## H-beta4: β₄ Shifts Higher in Joint Search

**Prediction**: β₄ converges above 0.410 (the isolated golden-section value) when optimized
jointly with β₅ and β₇.

**Causal mechanism**: When β₄ increases, β₅·L and β₇ can decrease (they were partially
compensating). The isolated search couldn't find the jointly-better higher β₄ because it
held β₅/β₇ fixed.

## H-convergence: Patience-150 Stops Before 300 Trials

**Prediction**: CMA-ES with warm start converges in fewer than 300 trials (patience=150
fires before exhausting the 500-trial budget).
