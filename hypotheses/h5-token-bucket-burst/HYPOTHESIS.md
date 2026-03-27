# H5: Token-Bucket Admission Control Reduces Tail Latency Under Bursty Traffic

**Status**: Confirmed with nuance
**Date**: 2026-02-20

## Hypothesis

> During traffic bursts (Gamma arrivals with high CV=3.5), accepting all requests floods the queues and increases tail latency. A token bucket that rejects excess requests should cap queue depth, trading some rejected requests for much better latency for admitted ones. Token-bucket TTFT p99 should be significantly lower than always-admit TTFT p99 across all seeds.

**Refuted if:** Token-bucket TTFT p99 is within 20% of always-admit TTFT p99 across all 3 seeds at rate=2000 with Gamma CV=3.5 arrivals.
