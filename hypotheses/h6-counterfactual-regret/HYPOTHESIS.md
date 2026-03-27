# H6: Counterfactual Regret Is Higher for Round-Robin Than Weighted Routing

**Status**: Confirmed with wrong mechanism
**Date**: 2026-02-23

## Hypothesis

> Counterfactual regret should be higher for round-robin than weighted routing under variable load. Round-robin ignores load entirely, so it should frequently route to suboptimal instances when load is asymmetric. The weighted policy with queue-depth scoring actively picks the best instance. The counterfactual analysis should quantify this difference, showing significantly lower regret for weighted routing.

**Refuted if:** Round-robin mean regret is within 20% of weighted mean regret at rate=200 with a mixed workload (70% short, 30% long requests) across all 3 seeds.
