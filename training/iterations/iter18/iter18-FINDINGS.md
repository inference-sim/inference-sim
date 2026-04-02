# Iteration 18: Findings

## Summary

Iteration 18 evaluated 32 coefficient sets along the line segment from iter16 (loss=60.19%)
to iter17 CMA-ES (loss=65.37%). The loss profile is **monotonically increasing** from λ=0
to λ=1 — no hidden valley exists between the two basins. Iter16 (λ=0) remains the best
point on this line and no multivariate TPE trial (13 directed after the line search)
improved on it. Run stopped at 68/500 trials to pivot to iter19 (negative-λ extension).

**Key result**: The 1D loss profile from λ=0 to λ=1 rises smoothly from 60.19% to 65.37%
with a gradient of ~5.7 per unit λ at the iter16 endpoint. This confirms the two basins are
on different sides of a ridge, with iter16's basin being deeper.

**Implication for iter19**: Since loss increases moving toward iter17 (positive λ), it should
*decrease* moving in the opposite direction (negative λ). Iter19 will test affine
extrapolations at λ ∈ [-0.5, 0], clamped to coefficient bounds, to search for lower-loss
regions beyond iter16.

## Line-Search Profile

| λ | Loss | Trial | Notes |
|---|---|---|---|
| 0.000 | **60.19** | 0 | iter16 endpoint (best) |
| 0.032 | 60.36 | 1 | |
| 0.065 | 60.51 | 2 | |
| 0.097 | 60.49 | 3 | |
| 0.129 | 60.58 | 4 | |
| 0.161 | 60.95 | 5 | |
| 0.194 | 60.91 | 6 | |
| 0.226 | 61.25 | 7 | |
| 0.258 | 61.20 | 8 | |
| 0.290 | 61.56 | 9 | |
| 0.323 | 61.46 | 10 | |
| 0.355 | 61.57 | 11 | |
| 0.387 | 61.82 | 12 | |
| 0.419 | 61.80 | 13 | |
| 0.452 | 61.52 | 14 | |
| 0.484 | 61.68 | 15 | |
| 0.516 | 61.77 | 16 | |
| 0.548 | 61.82 | 17 | Plateau region ~61.5-62.0 |
| 0.581 | 61.80 | 18 | |
| 0.613 | 61.83 | 19 | |
| 0.645 | 62.27 | 20 | |
| 0.677 | 62.34 | 21 | |
| 0.710 | 62.66 | 22 | Steepening |
| 0.742 | 62.87 | 23 | |
| 0.774 | 63.00 | 24 | |
| 0.806 | 63.51 | 25 | |
| 0.839 | 63.62 | 26 | |
| 0.871 | 63.78 | 27 | |
| 0.903 | 64.20 | 28 | |
| 0.935 | 64.64 | 29 | |
| 0.968 | 65.04 | 30 | |
| 1.000 | 65.37 | 31 | iter17 CMA-ES endpoint |

**Profile shape**: Monotonically increasing with two regimes — gentle slope (λ=0–0.3,
Δloss ≈ 1.4) and steeper slope (λ=0.6–1.0, Δloss ≈ 3.5). The gradient at λ=0 is
approximately +5.7/unit, suggesting negative-λ exploration could yield ~0.17 loss
improvement per 0.03 step.

**Full profile saved to**: `iter18_line_profile.csv`

## Status

| Metric | Value |
|---|---|
| Trials completed | 68/500 (stopped early to pivot to iter19) |
| Best loss | 60.19% (trial 0 = iter16 endpoint) |
| TPE improvement post-line-search | None (13 directed trials, no improvement) |
| Line profile | Monotonically increasing, no valley |
