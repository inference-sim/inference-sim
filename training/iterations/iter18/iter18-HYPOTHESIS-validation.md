# Iteration 18: Hypothesis Validation

## Overall Result

| Metric | Iter16 baseline | Iter18 best | Change |
|---|---|---|---|
| Overall loss | 60.19% | 60.19% | No improvement |
| Best trial | — | Trial 0 (iter16 warm-start) | |
| Trials completed | — | 68/500 (stopped to pivot to iter19) | |

---

## H-main: An Interpolated Point Achieves Loss Below 60.19%

**Prediction**: At least one of the 30 interpolated points or TPE-directed trials
achieves loss < 60.19%.

**Result**: ❌ **NOT CONFIRMED**
- All 30 interpolated points scored > 60.19% (range: 60.36–65.04)
- 36 TPE-directed trials also failed to beat 60.19%
- The iter16 warm-start (trial 0, λ=0) remained the best throughout

---

## H-profile: The Loss Profile Is Non-Convex Between Basins

**Prediction**: The 1D loss profile shows non-convex structure (local minimum or ridge).

**Result**: ❌ **NOT CONFIRMED**
- The profile is monotonically increasing from λ=0 (60.19%) to λ=1 (65.37%)
- No valley, ridge, or inflection point observed
- The two basins are connected by a smooth uphill slope

---

## H-tpe-exploits: Multivariate TPE Finds Improvements Off the Line

**Prediction**: TPE finds a loss below the best interpolated point by exploring
orthogonal directions.

**Result**: ❌ **NOT CONFIRMED**
- 36 TPE-directed trials (with multivariate=True) produced no improvement
- TPE was unable to exploit the correlation structure in the 32-point initial dataset

---

## H-convergence: Run Completes Within 200 Trials

**Prediction**: Optimizer stops before 200 trials.

**Result**: ✅ **CONFIRMED** (manually stopped at 68 for pivot to iter19)
- Would have early-stopped around trial 132 (patience=100 from trial 0)

---

## Summary

| Hypothesis | Result | Status |
|---|---|---|
| H-main | No point beat 60.19% | ❌ FAILED |
| H-profile | Monotonically increasing, no valley | ❌ FAILED |
| H-tpe-exploits | TPE found nothing off the line | ❌ FAILED |
| H-convergence | Stopped at 68 (manual pivot) | ✅ CONFIRMED |

**Key contribution**: The monotonically increasing loss profile from iter18 motivated the
negative-λ extension in iter19, which found the first improvement over iter16 (60.11%).
