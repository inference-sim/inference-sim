# H27: Chunked Prefill Reduces Short-Request TTFT in Bimodal Workloads

**Status**: Confirmed
**Date**: 2026-02-25

## Hypothesis

> Enabling chunked prefill (--long-prefill-token-threshold=256) reduces TTFT p99 for short requests (64 input tokens) by at least 30% in a bimodal workload (50% short at 64 tokens, 50% long at 2048 tokens) at moderate load (4 instances, rate near 50% saturation), because per-step time drops from ~43ms to ~11ms allowing short requests to be scheduled sooner.

**Refuted if:** TTFT p99 for short requests improves by less than 15% with chunked prefill enabled versus disabled, across all 3 seeds.
