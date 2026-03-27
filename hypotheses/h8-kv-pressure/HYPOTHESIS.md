# H8: KV Cache Pressure Increases Preemptions and Worsens Tail Latency

**Status**: Confirmed
**Date**: 2026-02-20

## Hypothesis

> Reducing total KV blocks should increase preemption frequency and worsen tail latency. KV blocks are the memory currency -- each running request holds blocks proportional to its token count. With fewer blocks, the cache fills up faster, forcing preemptions (evictions of running requests to make room). Preempted requests restart from scratch, increasing tail latency. Both preemption rate and TTFT p99 should monotonically increase as KV blocks decrease.

**Refuted if:** Preemption rate or TTFT p99 is non-monotonic with decreasing KV block count (i.e., any inversion where fewer blocks produces lower preemption or better TTFT), across all 3 seeds.
