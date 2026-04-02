# Iteration 22: Findings — β₂ Decode Correction via Coordinate Descent

## Summary

Golden section search on β₂ (decode roofline correction) reduced it from 1.611 → **1.146**,
improving loss from 39.86% → **39.42%** (0.44-point improvement, 16 evaluations).

β₂ dropped 29% from its iter16 value. The previous β₂=1.611 over-corrected decode time
by 61%. With β₈=440 now handling MoE decode overhead separately, the decode correction
retreats toward the physically expected ~1.0. β₂=1.146 means only a 14.6% correction
above roofline — much more physically reasonable.

## Best Coefficients (iter22)

| Coeff | Description | Value | Change from iter21 |
|---|---|---|---|
| α₀ | QueueingTime | 15,562.0 µs | unchanged |
| α₁ | PostDecodeFixedOverhead | 776.2 µs | unchanged |
| α₂ | OutputTokenProcessingTime | 45.9 µs/token | unchanged |
| β₁ₐ | Prefill compute correction | 0.116 | unchanged |
| **β₂** | **Decode correction** | **1.146** | **was 1.611 (−29%)** |
| β₃ | Weight loading correction | 1.363 | unchanged |
| β₄ | TP communication | 0.396 | unchanged |
| β₅ | Per-layer overhead | 62.3 µs/layer | unchanged |
| β₆ | Per-request scheduling | 2.80 µs/req | unchanged |
| β₇ | Per-step constant | 169.4 µs/step | unchanged |
| β₈ | Per-MoE-layer overhead | 440.0 µs/MoE-layer | unchanged |
| β₁ᵦ | Prefill memory (dropped) | 0.0 | unchanged |

## Results

| Metric | Iter21 | Iter22 | Improvement |
|---|---|---|---|
| Overall loss | 39.86% | **39.42%** | −0.44 |
| TTFT RMSE | 23.87% | 24.00% | +0.13 (negligible) |
| E2E RMSE | 16.78% | **15.42%** | −1.36 |

The improvement is entirely in **E2E** (−1.36 RMSE), consistent with β₂ governing decode
step time which dominates E2E but not TTFT.

## Coordinate Descent Progress (iter20–22)

| Iter | Pivot | Before → After | Loss |
|---|---|---|---|
| 20 | β₈ (MoE overhead) | 0 → 440 µs | 60.11% → 40.58% |
| 21 | β₁ₐ (prefill compute) | 0.201 → 0.116 | 40.58% → 39.86% |
| **22** | **β₂ (decode correction)** | **1.611 → 1.146** | **39.86% → 39.42%** |

Total improvement from coordinate descent: **60.11% → 39.42% (−20.69 points, 34.4% relative)**.

## Next Steps

Continue coordinate descent cycle:
1. **β₈** re-optimization (was found with β₁=0.201, β₂=1.611 — both changed since)
2. **β₅** (per-layer overhead — interacts with β₈)
3. Cycle back to β₁ₐ, β₂ until convergence
