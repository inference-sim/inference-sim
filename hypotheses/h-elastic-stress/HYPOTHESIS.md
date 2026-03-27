# H-Elastic-Stress: Stress Testing Elastic Priority Batching

**Status**: Confirmed with boundary conditions
**Date**: 2026-03-10

## Hypothesis

> The elastic priority batching dual-objective breakthrough generalizes across dimensions that Iteration 7 held constant: cluster scale (2 to 16 instances), KV cache pressure (5000 and 2000 blocks), and asymmetric request sizes (critical-short/sheddable-long, critical-long/sheddable-short, ParetoLogNormal).

**Refuted if:** Fewer than 6 of 8 stress variants show elastic ratio below 0.80, or any variant shows batch occupancy degradation greater than 10%.
