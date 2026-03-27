# H-Deadline-Urgency: Deadline-Aware SLO Scheduling

**Status**: Refuted
**Date**: 2026-03-10

## Hypothesis

> DeadlineAwarePriority with per-class TTFT deadlines will reduce critical TTFT P99 by >15% over StaticClassWeight (B2) at 120% capacity, because hyperbolic urgency growth creates stronger priority separation during transient overload.

**Refuted if:** Critical TTFT P99 improvement over StaticClassWeight is less than 10% across all 3 seeds at 120% capacity, or Treatment produces byte-identical results to B2.
