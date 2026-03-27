# H26: Admission Latency Causal Ordering

**Status**: Confirmed
**Date**: 2026-02-22

## Hypothesis

> Under low load (no queuing), configuring `--admission-latency L` should increase both TTFT and E2E by exactly `L` microseconds. This validates the cluster event pipeline's causal ordering: Arrival -> Admission (+latency) -> Routing -> Queue -> Batch -> Step.

**Refuted if:** The TTFT or E2E delta between baseline (latency=0) and treatment (latency=L) differs from L by more than 1%, or the relationship is non-linear across two treatment levels.
