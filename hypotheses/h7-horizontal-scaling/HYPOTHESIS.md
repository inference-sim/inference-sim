# H7: Horizontal Scaling Halves TTFT Under Saturation

**Status**: Confirmed
**Date**: 2026-02-23

## Hypothesis

> Increasing instances from 4 to 8 should roughly halve TTFT p99 for saturated workloads. If the workload saturates 4 instances (long queues, high utilization), adding 4 more should absorb the excess load -- requests wait in shorter queues, reducing TTFT. The predicted ratio is ~2x improvement. E2E should be less sensitive because decode time dominates.

**Refuted if:** The 4-to-8 instance TTFT p99 ratio is less than 1.5x at rate=500 (saturated), or the sub-saturation control (rate=100) shows a similar scaling ratio (>1.5x), across all 3 seeds.
