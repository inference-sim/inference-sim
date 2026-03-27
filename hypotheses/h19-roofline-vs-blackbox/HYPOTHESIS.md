# H19: Roofline vs Blackbox Mode -- Policy Ranking Equivalence

**Status**: Partially confirmed
**Date**: 2026-02-22

## Hypothesis

> Roofline mode should produce different absolute latencies but same relative policy rankings as blackbox mode. The routing decisions depend on instance state (queue depth, KV utilization), not on the latency model, so policy ordering should be preserved even though absolute values differ.

**Refuted if:** Mean TTFT or mean E2E policy rankings differ between roofline and blackbox modes in 2 or more of 3 seeds.
