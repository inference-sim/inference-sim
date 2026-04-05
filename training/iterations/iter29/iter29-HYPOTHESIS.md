# Iteration 29: Sequential Golden Section — 5-Coefficient Re-calibration

## Context

Iter27 (CMA-ES joint search) moved β₁ₐ, β₄, β₅, β₇, β₈, β₂ᵦ simultaneously, but held
β₃ and β₆ frozen throughout. After a large joint move, previously-optimal frozen
coefficients can become misaligned — the joint step changes the loss surface in ways
that the frozen values can no longer compensate for.

Iter28 (TPE cross-check, 162 trials) confirmed iter27 is a local optimum within the
tight neighborhood, but did not explore whether β₃ and β₆ need re-calibration.

Iter29 applies sequential golden section search to 5 coefficients in order:
β₃ → β₆ → β₅ → β₈ → β₂ᵦ. Each evaluation parallelizes all 15 experiments
(--max-workers 15). Each phase seeds from the best value found by the previous phase.

## H-main: β₃ and β₆ Are Misaligned After Iter27

**Prediction**: At least one of β₃ or β₆ moves >5% from its iter27 value and yields
loss improvement > 0.05 points.

**Causal mechanism**: Iter27 shifted β₄↑ (0.41→0.75), β₅↓ (49.6→32.4), β₇↓ (169→126).
These changes alter the magnitude of the per-step prediction, which β₆ (per-request
overhead, additive) and β₃ (weight loading, multiplicative) interact with. The optimal
β₆ after these shifts is no longer the pre-iter27 value.

**Diagnostic clause**: If neither β₃ nor β₆ moves meaningfully, iter27's joint search
inadvertently found the optimum for these parameters too, which would be surprising.

## H-beta6: β₆ Under-compensates More Than β₃

**Prediction**: β₆ moves more than β₃ (larger relative shift), because per-request
overhead interacts directly with the dominant TTFT term that β₄/β₅/β₇ changed.

**Causal mechanism**: β₃ (weight loading, multiplied by weight_bytes) is decoupled from
the TP communication and per-layer structure. β₆ (per-request constant additive term)
directly offsets the prefill prediction regardless of model architecture, so it absorbs
any systematic bias introduced by the β₄/β₅/β₇ joint move.

## H-sequential: Phases 3-5 Add Diminishing Returns

**Prediction**: β₅, β₈, β₂ᵦ each improve loss by <0.05 points beyond what β₃/β₆ provide.
The main improvement comes from phases 1-2; phases 3-5 are confirmatory.

**Causal mechanism**: β₅, β₈, β₂ᵦ were all active in iter27's joint search and reached
their joint optimum. Re-searching them 1D after updating β₃/β₆ should yield at most a
small correction.
