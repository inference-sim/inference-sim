# H-Compound-Strategy: Compound Strategy Reduces Critical TTFT P99

**Status**: Confirmed with nuance
**Date**: 2026-03-10

## Hypothesis

> The full compound strategy (StaticClassWeight + SLOGatedAdmission + PriorityPreemption) at 120% capacity with --max-num-running-reqs 32 reduces critical TTFT P99 by >25% over baseline (always-admit, no preemption) at 120% overload with mixed SLO classes (20% critical, 40% standard, 40% sheddable), because the three mechanisms protect critical requests at different layers (queue ordering, load shedding, batch composition).

**Refuted if:** Critical TTFT P99 improvement over baseline is less than 15% across all 3 seeds at 120% capacity.
