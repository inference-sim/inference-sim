# H27: Mixed-Batch Max vs Weighted-Average Combination

**Status**: Confirmed
**Date**: 2025-02-25

## Hypothesis

> The roofline model's adaptive weighted-average combination for mixed prefill+decode steps (sim/roofline_step.go lines 407-430) systematically underpredicts mixed-step latency compared to the standard roofline `max(prefillTime, decodeTime)` combination, because any convex combination with positive weights summing to 1 produces a value strictly below max(P,D) when P != D. Replacing the adaptive weighting with `max(prefillTime, decodeTime)` should improve E2E MAPE for high-QPS workloads (where mixed batches dominate) by at least 1 percentage point.

**Refuted if:** Switching the mixed-batch combination from the adaptive weighted-average to `max(prefillTime, decodeTime)` worsens overall E2E MAPE by more than 0.5 percentage points across the ground-truth experiments, OR the improvement at high-QPS sweep points (rate >= 4 req/s) is less than 1 percentage point, OR the effect at synchronous rate exceeds 0.5 percentage points (violating experimental isolation since synchronous steps have no mixed batches).
