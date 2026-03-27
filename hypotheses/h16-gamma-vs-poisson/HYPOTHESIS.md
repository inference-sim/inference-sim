# H16: Gamma vs Poisson Tail Latency

**Status**: Confirmed with nuance
**Date**: 2026-02-22

## Hypothesis

> Bursty (Gamma, CV=3.5) arrivals should produce worse tail latency than Poisson at the same average rate, because burst clusters create transient queue depth spikes that inflate TTFT p99.

**Refuted if:** Poisson TTFT p99 equals or exceeds Gamma TTFT p99 in 2 or more of 3 seeds at the core operating point (rate=1000, 500 requests).
