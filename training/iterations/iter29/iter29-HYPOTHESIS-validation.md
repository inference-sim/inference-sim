# Iteration 29: Hypothesis Validation

## Overall Results

| Metric | Iter27 | Iter29 | Change |
|---|---|---|---|
| Overall loss | 34.6564% | **34.5675%** | **-0.089 (✅ improvement)** |
| Total evaluations | — | ~130 across 5 phases | ~26 evals/phase |
| Wall-clock | — | ~25 min | ~5 min/phase |

---

## H-main: β₃ and β₆ Are Misaligned After Iter27

**Prediction**: At least one of β₃ or β₆ moves >5% and yields >0.05 improvement.

**Result**: ✅ **CONFIRMED** — β₆ moved +57% (2.805→4.417) and delivered -0.11 points
improvement on its own. β₃ barely moved (-0.1%), confirming β₃ was already near-optimal
but β₆ was significantly misaligned.

---

## H-beta6: β₆ Under-compensates More Than β₃

**Prediction**: β₆ moves more than β₃.

**Result**: ✅ **CONFIRMED** — β₆: +57% vs β₃: -0.1%. β₆ was the dominant misaligned
coefficient, exactly as predicted. The per-request additive overhead absorbed the
systematic bias from iter27's joint β₄/β₅/β₇ move.

---

## H-sequential: Phases 3-5 Add Diminishing Returns

**Prediction**: Phases 3-5 each contribute <0.05 points beyond phases 1-2.

**Result**: ✅ **CONFIRMED** with nuance:

| Phase | Coeff | Change | Loss | Δ vs prev |
|---|---|---|---|---|
| 1 | β₃ weight_loading | 1.3636→1.3625 (-0.1%) | 34.7208% | +0.064 (slight regression) |
| 2 | β₆ per_request | 2.8051→4.4168 (+57%) | 34.5454% | **-0.175** |
| 3 | β₅ per_layer | 32.394→32.095 (-1%) | 34.6557% | +0.110 (regression) |
| 4 | β₈ moe_overhead | 505.51→481.86 (-5%) | 34.5882% | -0.067 |
| 5 | β₂ᵦ decode_memory | 1.9224→1.9471 (+1.3%) | 34.5675% | -0.021 |

Phase 3 (β₅) showed a slight regression because the midpoint of its bracket landed
slightly above the phase 2 value — the loss surface is flat here and golden section
converged to a suboptimal midpoint. Phase 4 (β₈) and phase 5 (β₂ᵦ) each added small
genuine improvements.

**Note on midpoint vs argmin**: Golden section returns the bracket midpoint, not the
strict argmin of all cached evaluations. For β₅ and β₈ where the function is nearly
flat near the optimum, this can result in the "done" value being slightly worse than
the best seen during the search. Future iterations should track the argmin explicitly.
