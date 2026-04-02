# Iteration 18: Line-Search Seeded TPE Between Iter16 and Iter17 Basins

## Context

Iteration 17 revealed that the 15-experiment loss landscape has at least two stable local
minima: iter16's basin (60.19%, found by TPE on 9 experiments) and a CMA-ES basin (65.37%,
found by CMA-ES on 15 experiments). Neither TPE nor CMA-ES starting from iter16's warm-start
could escape the local structure to find something better.

Iter18 probes the terrain **between** these two basins by evaluating 30 linearly interpolated
coefficient sets along the line segment from iter16 to iter17 CMA-ES. These 32 evaluated
points (30 interpolated + 2 endpoints) seed a multivariate TPE sampler, giving it a diverse
initial dataset spanning both basins. If a lower valley exists along this path — or if the
interpolated evaluations reveal a gradient direction TPE can exploit — this approach will
find it.

---

## H-main: An Interpolated Point Achieves Loss Below 60.19%

**Prediction**: At least one of the 30 interpolated coefficient sets will achieve a loss
strictly below 60.19% (iter16 baseline). Alternatively, the multivariate TPE seeded with
all 32 data points will find a point below 60.19% within its subsequent exploration.

**Causal Mechanism**: The iter16 and iter17 CMA-ES basins have very different coefficient
profiles (normalized L2 distance = 0.83). The biggest differences are α₁ (815 → 6270µs)
and β₆ (3 → 122µs). These two parameters trade off per-request overhead allocation. An
intermediate allocation — say α₁ ≈ 3000µs, β₆ ≈ 60µs — may fit all 15 experiments better
than either extreme. The interpolated evaluations will test this directly.

**Diagnostic Clause**: If all 32 interpolated+endpoint evaluations show loss > 60.19%,
and the loss profile is convex (monotonically increasing away from iter16), the iter16
point is confirmed as a local minimum on this line and potentially the global minimum.
Consider exploring orthogonal directions in iter19.

---

## H-profile: The Loss Profile Is Non-Convex Between Basins

**Prediction**: The 1D loss profile (loss vs λ for λ ∈ [0,1]) will show non-convex
structure — either a local minimum between the endpoints, or a ridge followed by descent.

**Causal Mechanism**: The two basins occupy different regions of coefficient space. If the
loss landscape between them were convex, gradient-based methods would have found the global
minimum. The non-convexity is why TPE and CMA-ES converged to different basins from similar
starting points.

**Diagnostic Clause**: If the profile is convex (smooth upward curve from 60.19 at λ=0 to
65.37 at λ=1), the two basins are actually the same basin viewed from different angles, and
the line search confirms iter16 is at the bottom.

---

## H-tpe-exploits: Multivariate TPE Finds Improvements Off the Line

**Prediction**: After seeding with 32 interpolated trials, multivariate TPE will find a
loss below the best interpolated point by exploring directions orthogonal to the line.

**Causal Mechanism**: The interpolated trials give TPE a 1D slice of the 10D landscape.
With `multivariate=True`, TPE models parameter correlations — it can see, for example,
that α₁ and β₆ are anti-correlated along the line (one increases as the other decreases).
This lets it extrapolate to unexplored regions of the landscape that lie off the line but
follow the same correlation structure.

**Diagnostic Clause**: If TPE does not improve on the best interpolated point, the loss
landscape near the line has no exploitable gradient perpendicular to the interpolation
direction. The line search itself was the main contribution of iter18.

---

## H-convergence: Run Completes Within 200 Trials

**Prediction**: The optimizer stops before 200 trials (budget: 500) via patience=100.
The 32 initial enqueued trials will evaluate in ~2 batches. If the best interpolated point
is near λ=0 (iter16), TPE may not find anything better and patience fires quickly (~130
trials total).

**Diagnostic Clause**: If patience fires before trial 100, the 32-trial seed was sufficient
to map the landscape and TPE added nothing. This is a positive outcome — it means the line
search was the right tool and TPE overhead was minimal.
