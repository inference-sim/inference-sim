# H25: Integration Stress Test -- Full Policy Stack

**Status**: Confirmed
**Date**: 2026-02-22

## Hypothesis

> The full policy stack should maintain conservation invariants under combined load. Running all modules simultaneously -- weighted routing (prefix-affinity + queue-depth + kv-utilization), token-bucket admission, tiered KV cache, priority-FCFS scheduling, decision tracing with counterfactual analysis -- should satisfy: (a) conservation (completed + queued + running + rejected == injected), (b) determinism (same seed produces byte-identical output), (c) no panics.

**Refuted if:** Any invariant check (INV-1 conservation, INV-5 causality, INV-6 determinism) fails, or the simulator panics under the full policy stack configuration.
