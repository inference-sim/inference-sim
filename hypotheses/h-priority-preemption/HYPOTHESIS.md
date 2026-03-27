# H-Priority-Preemption: Batch-Level Priority Preemption

**Status**: Partially confirmed
**Date**: 2026-03-10

## Hypothesis

> Adding priority-based preemption (evicting lowest-priority running requests for waiting high-priority requests) to StaticClassWeight will reduce critical TTFT P99 by >50% over B2, because preemption eliminates the batch-occupancy queue wait that dominates critical TTFT.

**Refuted if:** Critical TTFT P99 improvement over StaticClassWeight (B2) is less than 10% across all 3 seeds at 120% capacity, or the mechanism triggers zero preemptions (indicating it is inert).
