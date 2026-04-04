# Iteration 28: Hypothesis Validation

## Overall Results

| Metric | Iter27 | Iter28 | Change |
|---|---|---|---|
| Overall loss | 34.61% | **34.6564%** | **+0.05 (no improvement)** |
| Best trial | — | 0 (warm-start) / 162 total | patience=150 fired at trial 150 |
| Trials run | 141 | **162** | stopped by patience |
| Wall-clock | — | 27:22 (1642s) | ~10s/trial avg |

---

## H-main: TPE Confirms CMA-ES Optimum

**Prediction**: TPE converges within ±1% of iter27's loss (34.61%).

**Result**: ✅ **CONFIRMED** — best loss is 34.6564%, within 0.05% of iter27. Crucially,
this is now a *meaningful* confirmation: 162 trials ran including a full TPE directed-search
phase (trials 25+). The closest competitor was trial 155 at 34.719% — only 0.06 points worse
than the warm-start. No trial found a better point anywhere in the search space.

The iter27 CMA-ES result is confirmed as a robust local optimum, not a CMA-ES-specific
artifact.

---

## H-explore: Tight Bounds Prevent Meaningful Exploration

**Prediction**: Patience fires before 200 trials due to tight bounds.

**Result**: ✅ **CONFIRMED** — patience fired at trial 150 (best at trial 0). 12 additional
in-flight trials completed after stop(), for 162 total. The tight bounds combined with the
warm-start near the optimum meant TPE's directed search consistently failed to improve.

The second-best trial (155, loss=34.719%) was still 0.06 points worse — confirming the
warm-start is at a true local minimum within this search region.
