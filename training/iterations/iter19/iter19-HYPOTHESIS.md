# Iteration 19: Negative-λ Affine Extension Beyond Iter16

## Context

Iter18's line search revealed the loss profile from iter16 (λ=0, 60.19%) to iter17 CMA-ES
(λ=1, 65.37%) is monotonically increasing with gradient ≈ +5.3/unit at λ=0. This means
the direction AWAY from iter17 (negative λ) should decrease loss.

Iter19 tests 20 points along the negative-λ extension from iter16, extrapolating into
previously unexplored coefficient space. The key physical test: what happens when
per-request overhead terms (α₁ PostDecodeFixed, β₆ PerRequest) are gradually reduced
toward zero? The negative-λ direction naturally performs this removal.

## H-main: A Negative-λ Point Achieves Loss Below 60.19%

**Prediction**: At least one point in the range λ ∈ [-0.2, 0) achieves loss below 60.19%.

**Causal Mechanism**: The iter18 gradient at λ=0 is +5.3/unit λ. Extrapolating linearly,
λ=-0.02 should give ~60.08 and λ=-0.10 should give ~59.66. Even with clamping effects
(β₆ → 0 at λ=-0.025, α₁ → 0 at λ=-0.149), the direction reduces terms that may be
over-fitted to the 9-experiment training set while introducing no new complexity.

**Diagnostic Clause**: If all negative-λ points score > 60.19%, the gradient reverses
at λ=0 (iter16 is a true local minimum in this direction too). This would confirm 60.19%
as a saddle-point minimum — optimal along ALL explored 1D directions.

## H-overhead-removal: Removing Per-Request Overhead Improves Fit

**Prediction**: Points where β₆ = 0 (λ ≤ -0.025) perform comparably to or better than
the iter16 baseline (β₆ = 2.94µs). The per-request scheduling overhead is a weak term
that adds noise to the fit without improving accuracy.

**Diagnostic Clause**: If β₆ = 0 points are significantly worse (>2% degradation), the
per-request term carries meaningful information that other terms cannot absorb.
