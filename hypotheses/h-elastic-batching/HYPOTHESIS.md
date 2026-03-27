# H-Elastic-Batching: Elastic Priority Batching Achieves Dual Objective

**Status**: Confirmed
**Date**: 2026-03-10

## Hypothesis

> Large batches (maxRunningReqs=64) with aggressive priority preemption (margin=4.0, circuit breaker=10) can achieve BOTH high SLO attainment for critical requests AND high GPU utilization — the dual objective that prior iterations addressed separately. Specifically, elastic batching reduces critical TTFT P99 by >2x compared to large-batch-no-preemption while maintaining batch occupancy within 5%.

**Refuted if:** Elastic batching fails to improve critical TTFT P99 by at least 1.5x over large-batch, or batch occupancy drops by more than 10% compared to large-batch, across all 3 seeds.
