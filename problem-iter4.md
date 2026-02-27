# Problem: Load-Adaptive Cache Scoring via Cost-Benefit Composable Scorer

## Context from Iterations 1-3

**Current champion**: Static `pa:3,qd:2,kv:2` = 127.65ms RAG TTFT p99, 74.15ms combined (40.5% better than RR).

**What failed and why**:
- P2C (iter 1): Misses cache hits on N-2 instances. Full-scan weighted scoring is better.
- Dynamic weight switching (iter 2): Unnecessary — PA already returns 0 when no cache exists.
- Scheduling co-opt (iter 3): Null effect — router already separates cache-hit from cache-miss traffic at moderate load.

**The remaining opportunity**: At HIGH utilization (ρ>0.85), the router is FORCED to mix cache-hit and cache-miss requests on the same instances. The static PA:3 weight may not be optimal at all load levels:
- At low load: PA:3 is fine (cache exploitation is free)
- At high load: PA:3 may create too much load imbalance (queue penalty > cache saving)
- At overload: PA:3 is definitely counterproductive (queues grow unboundedly)

**The existing solution**: PR #447's `cost-benefit` scorer computes `cache_saving / (cache_saving + queue_delay)` which NATURALLY adapts to load. But it's only available inside the `adaptive-weighted` policy factory, not the regular `weighted` pipeline.

## Strategy: Make Cost-Benefit a Composable Scorer

Wire `cost-benefit` into `newScorerWithObserver()` so it can be used in the regular weighted pipeline:
- `cost-benefit:3,queue-depth:2` instead of `prefix-affinity:3,queue-depth:2`

The cost-benefit scorer replaces the linear PA scorer with a nonlinear load-adaptive version. Physics:
- PA scorer: `score = matchedBlocks / totalBlocks` (constant regardless of load)
- Cost-benefit: `score = cacheSaving / (cacheSaving + queueDelay)` (adapts to load)

At low load (queueDelay≈0): cost-benefit ≈ PA (exploit cache freely)
At high load (queueDelay >> cacheSaving): cost-benefit → 0 (ignore cache, balance load)

## Test Matrix: Multi-rate Sweep
- Rates: 100, 200, 300, 400 req/s at 8 instances
- Workloads: RAG (4096-token prefix), Independent (no prefix)
- Policies: cost-benefit:3,qd:2 vs pa:3,qd:2,kv:2 vs RR

## Hypotheses
H1: Cost-benefit scorer matches or beats PA scorer at ALL rates (low through overload)
H2: At overload (rate=400, ρ>1.0), cost-benefit significantly outperforms PA (which creates persistent load imbalance)
H3: At low load, cost-benefit ≈ PA ≈ RR (all equivalent, H23 confirmed this at ρ<0.2)
H4: On independent workloads, cost-benefit matches PA (both degenerate to load-only when no cache)
