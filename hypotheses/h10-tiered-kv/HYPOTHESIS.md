# H10: Tiered KV Cache Reduces Preemptions via CPU Offload

**Status**: Confirmed
**Date**: 2026-02-20

## Hypothesis

> When GPU KV blocks are exhausted, the single-tier cache preempts requests. With a CPU tier, blocks can be offloaded to CPU instead of being evicted entirely. The tradeoff: reload incurs transfer latency, but avoids full recomputation. Tiered cache should produce fewer preemptions and lower TTFT than single-tier at the same GPU block count (2100 blocks, near the preemption cliff identified in H8).

**Refuted if:** Tiered cache (CPU=500 blocks, offload threshold=0.8) does not reduce preemption count compared to single-tier, or TTFT mean is worse with tiered cache, across all 3 seeds.
