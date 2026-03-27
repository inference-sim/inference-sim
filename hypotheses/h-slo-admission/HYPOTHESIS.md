# H-SLO-Admission: SLO-Gated Admission Control

**Status**: Partially confirmed
**Date**: 2026-03-10

## Hypothesis

> Adding SLO-gated admission (rejecting sheddable under load) to StaticClassWeight will reduce critical TTFT P99 by >20% over B2 at 120% capacity, because shedding sheddable requests reduces total queue depth for all remaining classes. Additionally, cluster-wide TTFT P99 for admitted requests will improve (non-zero-sum).

**Refuted if:** Critical TTFT P99 improvement over B2 is less than 10% at 120% capacity across all 3 seeds, AND cluster-wide TTFT P99 does not improve by at least 5% (indicating admission provides no benefit at any level).
