# Iteration 28: TPE Cross-Check on CMA-ES Result

## Context

Iter27 used CMA-ES to jointly optimize 6 parameters, reaching 34.61% loss from a 37.42%
baseline (-2.81 points). However, CMA-ES was the only sampler used in iter27 due to SQLite
race conditions with TPE + n_jobs>1 in Optuna. The CMA-ES result may reflect a local optimum
specific to the covariance adaptation path taken.

Iter28 runs TPE (Tree-structured Parzen Estimator) on all 8 active betas from the iter27
warm start, with tighter bounds centered on iter27 values. TPE's probabilistic model is
qualitatively different from CMA-ES's covariance evolution — if both converge to the same
point, the iter27 result is likely a true local optimum in this parameter space.

## H-main: TPE Confirms CMA-ES Optimum

**Prediction**: TPE converges within ±1% of iter27's loss (34.61%), confirming that iter27
found a robust local optimum rather than a CMA-ES-specific artifact.

**Causal mechanism**: CMA-ES and TPE explore the loss surface via fundamentally different
mechanisms (covariance matrix adaptation vs. probabilistic surrogate). Agreement between
them is strong evidence that iter27's coefficient values are stable.

**Diagnostic clause**: If TPE finds loss substantially below 34.61%, CMA-ES got stuck in
a CMA-specific local basin and the interaction structure is richer than iter27 revealed.

## H-explore: Tight Bounds Prevent Meaningful Exploration

**Prediction**: With bounds width ≤30% of iter27 values, TPE's first-phase random sampling
covers a small volume and patience fires before 200 trials.

**Causal mechanism**: TPE requires ~25 random trials before its surrogate model activates.
If the warm-start initial evaluation is already near-optimal within tight bounds, the random
exploration phase is unlikely to find anything better, and patience terminates early.
