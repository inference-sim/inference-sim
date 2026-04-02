# Iteration 19: Findings

## Summary

Iteration 19 explored the negative-λ direction (extrapolating past iter16, away from iter17
CMA-ES) and achieved the first-ever improvement over iter16: **loss 60.11% vs 60.19%**.
The improvement is marginal (0.08 absolute, 0.13% relative) and occurs at a single point
(λ≈-0.01), with loss rising again for all more negative values.

**Best coefficients found**: Nearly identical to iter16, with α₁ (PostDecodeFixed) reduced
5% (815→776µs) and β₆ (PerRequest) reduced 5% (2.94→2.80µs). All other coefficients
unchanged within 0.1%.

**Status**: Marginal improvement confirms iter16's basin is very flat-bottomed — the minimum
is broad, not sharp. Further coefficient-level optimization in this architecture is unlikely
to yield meaningful gains. The path to lower loss runs through architectural changes (MoE
correction terms), not coefficient tuning.

---

## Negative-λ Line Profile

| Physical λ | Loss | Trial | Notes |
|---|---|---|---|
| 0.000 | 60.19 | 0 | iter16 endpoint |
| **-0.010** | **60.11** | **1** | **New best — first improvement over iter16** |
| -0.019 | 60.46 | 2 | Rises again |
| -0.029 | 60.46 | 3 | β₆ clamped to 0 here |
| -0.038 | 60.68 | 4 | |
| -0.048 | 60.58 | 5 | |
| -0.057 | 60.56 | 6 | |
| -0.067 | 60.67 | 7 | |
| -0.076 | 60.56 | 8 | |
| -0.086 | 60.76 | 9 | |
| -0.095 | 60.53 | 10 | |
| -0.105 | 60.76 | 11 | |
| -0.114 | 60.62 | 12 | |
| -0.124 | 60.56 | 13 | |
| -0.133 | 60.88 | 14 | |
| -0.143 | 61.14 | 15 | α₁ approaching 0 |
| -0.152 | 61.11 | 16 | |
| -0.162 | 61.30 | 17 | α₂ at upper bound (50) |
| -0.171 | 61.68 | 18 | |
| -0.181 | 61.04 | 19 | |
| -0.190 | 61.76 | 20 | |
| -0.200 | 62.01 | 21 | Full clamped endpoint |

**Profile shape**: The minimum is at λ≈-0.01 (60.11). For λ < -0.02, loss rises into the
60.4–60.8 range (flat noisy plateau). Beyond λ=-0.14 where α₁→0 and α₂→50, loss rises
above 61%. The negative direction is NOT a productive gradient — it's a very narrow dip
followed by a flat region and then uphill.

---

## Best Coefficients (Trial 1, loss 60.11%)

| Coeff | Description | Iter19 | Iter16 | Δ | Δ% |
|---|---|---|---|---|---|
| α₀ | QueueingTime (µs) | 15,562 | 15,569 | -8 | -0.05% |
| **α₁** | **PostDecodeFixed (µs)** | **776** | **815** | **-39** | **-4.8%** |
| α₂ | OutputToken (µs) | 45.9 | 45.7 | +0.2 | +0.4% |
| β₁ | Prefill correction | 0.201 | 0.201 | +0.0004 | +0.2% |
| β₂ | Decode correction | 1.611 | 1.617 | -0.007 | -0.4% |
| β₃ | Weight loading | 1.363 | 1.360 | +0.003 | +0.2% |
| β₄ | TP communication | 0.396 | 0.396 | +0.0003 | +0.1% |
| β₅ | Per-layer (µs/layer) | 62.3 | 62.2 | +0.1 | +0.2% |
| **β₆** | **Per-request (µs/req)** | **2.80** | **2.94** | **-0.14** | **-4.8%** |
| β₇ | Per-step (µs/step) | 169.4 | 169.4 | -0.01 | -0.01% |

The improvement is entirely driven by a ~5% reduction in per-request overhead terms (α₁, β₆).

---

## Combined Line-Search Results (Iterations 18 + 19)

Across iter18 (λ ∈ [0, +1]) and iter19 (λ ∈ [-0.2, 0]), we have evaluated the full 1D
loss profile through the iter16 point along the iter16→iter17 direction:

```
Loss
65 |                                                              *
64 |                                                        * *
63 |                                                  * * *
62 |                                          * * * *
61 |     * * * * * * * * * * * * * * * * * * *
60 | * *   (← iter16 region, best at 60.11)
   +---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  -0.2    -0.1     0     0.1    0.2   ...   0.8    0.9    1.0
                        λ (physical)
```

The iter16 point sits at the bottom of a broad, asymmetric valley: gently sloping downhill
from the positive side (gradient ~5.3/unit), with a very narrow dip at λ≈-0.01 before
rising again in the negative direction.

---

## Optimization Summary

| Phase | Trials | Best found |
|---|---|---|
| Line search (22 enqueued) | 22 | 60.11 (trial 1, λ=-0.01) |
| TPE directed | 75 | No improvement |
| **Total** | **97** | **60.11** |

97 trials completed before manual stop (patience was at 78/100). TPE with multivariate
correlations could not improve on the best interpolated point.

---

## Conclusions

1. **Iter16's coefficients are near-optimal**: The 0.08 improvement (60.19→60.11) from 4
   iterations of search (iter17–19, ~500+ total trials across TPE, CMA-ES, and line searches)
   confirms that iter16 found a point very close to the global minimum for this architecture.

2. **The loss floor is ~60% for the 7-term formula**: Scout MoE experiments account for
   42–73% TTFT APE regardless of coefficient tuning. Breaking below 60% requires architectural
   changes (MoE-specific terms, per-family corrections).

3. **Per-request overhead terms are slightly over-fitted**: The improvement came entirely from
   reducing α₁ and β₆ by ~5%. These terms absorb noise from the 9-experiment training set
   that doesn't generalize to the full 15-experiment evaluation.

4. **Line-search seeded TPE is an effective exploration technique**: The convex/affine
   combination approach between known basins provided clear landscape visualization and
   found the marginal improvement that TPE alone could not.
