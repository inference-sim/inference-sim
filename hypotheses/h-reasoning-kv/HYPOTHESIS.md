# H-Reasoning-KV: Reasoning Context Accumulation Under KV Pressure

**Status**: Refuted (primary), Confirmed (supporting)
**Date**: 2026-02-23

## Hypothesis

> Under constrained KV capacity, multi-turn reasoning workloads with context accumulation trigger the preemption cliff at a block count proportional to their peak per-request demand (120 blocks for round 4), while standard workloads with uniform per-request demand (72 blocks) trigger it at a proportionally lower block count. The cliff shift ratio should be approximately 1.1-1.3x.

**Refuted if:** The cliff shift ratio is less than 1.1x (below the pre-committed 20% difference threshold) across all 3 seeds, indicating that per-request peak demand does not drive the preemption cliff location.
