# H-Heterogeneous-Pools: Physical Isolation via Dedicated Instance Pools

**Status**: Confirmed with nuance
**Date**: 2026-03-10

## Hypothesis

> Physical isolation of critical traffic into a dedicated "fast lane" instance will achieve critical TTFT P99 < 100ms (vs 500-1100ms in a shared 4-instance pool) at 120% capacity, because critical requests no longer compete with standard and sheddable traffic for queue positions and batch slots.

**Refuted if:** Fast lane critical TTFT P99 is worse than the shared pool baseline, or total throughput (fast lane + bulk pool) drops by more than 20% compared to the shared pool.
