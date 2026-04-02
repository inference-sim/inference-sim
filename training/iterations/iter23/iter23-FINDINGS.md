# Iteration 23: Findings — Joint 3D Optimization of (β₁ₐ, β₂, β₈)

## Summary

Joint 3D optimization of the three most impactful coefficients (β₁ₐ, β₂, β₈) via
coarse grid → fine grid → golden section polish. Loss improved from 39.42% → **39.24%**
(0.18-point improvement, 682 evaluations, 160 minutes).

The joint optimum is close to but distinct from the coordinate descent result:
- β₁ₐ: 0.116 → **0.139** (+20% — prefill compute correction increased)
- β₂: 1.146 → **1.242** (+8% — decode correction increased)
- β₈: 440 → **427** (−3% — MoE overhead slightly decreased)

The interaction effect was real but modest: the joint optimum improves 0.18 points over
the best coordinate descent result. The coordinate descent iterations (20–22) captured
most of the improvement; the joint search refined the coupling.

## Best Coefficients (iter23)

| Coeff | Description | Value | Change from iter22 |
|---|---|---|---|
| β₁ₐ | Prefill compute correction | **0.139** | was 0.116 (+20%) |
| β₂ | Decode correction | **1.242** | was 1.146 (+8%) |
| β₈ | Per-MoE-layer overhead | **427.3** µs | was 440 (−3%) |
| (all others) | | (iter22 values) | unchanged |

## Results

| Metric | Iter22 | Iter23 | Improvement |
|---|---|---|---|
| Overall loss | 39.42% | **39.24%** | −0.18 |
| TTFT RMSE | 24.00% | 23.97% | −0.03 |
| E2E RMSE | 15.42% | **15.27%** | −0.15 |

## Full Training Journey (iter16–23)

| Iter | Loss | Method | Key change |
|---|---|---|---|
| 16 | 60.19% | TPE 1705 trials | Trained-roofline architecture |
| 17 | 65.37% | CMA-ES | Different basin (worse) |
| 18 | 60.19% | Line search λ∈[0,1] | No valley between basins |
| 19 | 60.11% | Line search λ∈[-0.2,0] | Marginal negative-λ improvement |
| 20 | 40.58% | 1D grid β₈ | **β₈·nMoELayers breakthrough** |
| 21 | 39.86% | 2D grid + golden section | Prefill compute-only split |
| 22 | 39.42% | Golden section β₂ | Decode correction readjustment |
| **23** | **39.24%** | **3D joint (β₁ₐ,β₂,β₈)** | **Joint interaction capture** |

**Total improvement**: 60.19% → 39.24% = **20.95 points (34.8% relative)**

## Conclusion

The 3D joint search confirmed that the loss landscape near the coordinate descent
solution is smooth with weak coupling between β₁ₐ, β₂, and β₈. The 0.18-point
improvement suggests we are near the floor of the current 9-term formula on 15
experiments. Further gains require either new basis functions or additional training
data, not coefficient refinement.
