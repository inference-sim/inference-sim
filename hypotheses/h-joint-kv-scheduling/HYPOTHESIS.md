# H-Joint-KV-Scheduling: Joint KV-Scheduling Optimization

**Status**: Confirmed
**Date**: 2026-03-11

## Hypothesis

> SLO-aware KV eviction (targeting the lowest-priority running request instead of the tail request) creates a multiplicative interaction with elastic priority batching under KV pressure, because the two mechanisms protect critical requests at different resource layers: elastic batching protects critical at the scheduling layer (batch slot allocation) and SLO-aware eviction protects critical at the memory layer (KV cache block allocation).

**Refuted if:** The joint optimization produces less than 1.5x interaction ratio (joint improvement / product of individual improvements) at any KV pressure level where both mechanisms are independently active, across all 3 seeds.
