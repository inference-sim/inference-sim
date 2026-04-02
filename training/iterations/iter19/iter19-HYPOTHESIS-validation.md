# Iteration 19: Hypothesis Validation

## Overall Result

| Metric | Iter16 baseline | Iter19 best | Change |
|---|---|---|---|
| Overall loss | 60.19% | **60.11%** | **-0.08 (improved)** |
| Best trial | — | Trial 1 (λ=-0.01) | |
| Trials completed | — | 97/500 (stopped early) | |

---

## H-main: A Negative-λ Point Achieves Loss Below 60.19%

**Prediction**: At least one point in λ ∈ [-0.2, 0) achieves loss < 60.19%.

**Result**: ✅ **CONFIRMED**
- Trial 1 (λ≈-0.01): loss = **60.11%** < 60.19% ✅
- Only 1 of 21 negative-λ points beat iter16 — the improvement is narrow and marginal
- The linear gradient extrapolation predicted ~60.08 at λ=-0.02; actual was 60.46 (gradient
  reversed after the first step)

---

## H-overhead-removal: Removing Per-Request Overhead Improves Fit

**Prediction**: Points where β₆ = 0 (λ ≤ -0.025) perform comparably to or better than
iter16 (β₆ = 2.94µs).

**Result**: ❌ **NOT CONFIRMED**
- At λ=-0.029 (β₆ clamped to 0): loss = 60.46% — worse than iter16's 60.19%
- β₆ = 0 points range from 60.46% to 62.01% — all worse than iter16
- The per-request scheduling term, even at 2.94µs, carries meaningful information

The optimal β₆ is ~2.80µs (from the best trial), not zero. Removing it entirely degrades
the fit by 0.27–1.82 percentage points.

---

## Summary

| Hypothesis | Result | Status |
|---|---|---|
| H-main | 60.11% < 60.19% ✅ | ✅ CONFIRMED (marginal) |
| H-overhead-removal | β₆=0 is worse | ❌ NOT CONFIRMED |
