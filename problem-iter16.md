# Iteration 16: Precise KV-Aware Routing (llm-d Blog Hypothesis)

## The Opportunity

The llm-d blog ("KV-Cache Wins You Can See") showed **57x TTFT P90 improvement** from precise vs approximate KV routing:
- Precise (0.54s P90): real-time KVEvents → router knows exact block-hash-to-instance mapping
- Approximate (31.08s P90): routing-history-based estimation → diverges under dynamic workloads

BLIS models the **approximate** approach via `PrefixCacheIndex` (router-side LRU of block hashes). The precise approach corresponds to the router reading actual per-instance KV cache state.

## Key Insight from Our Experiments

Iteration 6 showed static-default loses to RR by 23-25% under KV pressure (5000 blocks). Iteration 8 proved this is because the KV-utilization scorer penalizes cached content. But there's a DEEPER issue: the `PrefixCacheIndex` itself diverges from actual KV state when blocks are evicted under pressure. The PA scorer's match ratio becomes inaccurate — phantom cache hits.

## Strategy: Model Precise Routing by Synchronizing PrefixCacheIndex

Instead of building a new routing policy, we can approximate precise routing by:
1. **Increasing PrefixCacheIndex LRU capacity** to prevent router-side eviction
2. **Testing at different snapshot refresh intervals** to model staleness
3. **Comparing fresh (interval=0) vs stale (interval=100ms) KV signals** under KV pressure

This directly tests the llm-d blog's hypothesis within BLIS.

## Hypotheses

H16-1: At `--snapshot-refresh-interval=0` (always fresh KV signals) + large LRU, weighted routing under KV pressure matches or beats normal-KV performance.

H16-2: Increasing snapshot staleness (1ms → 10ms → 100ms) progressively degrades TTFT under KV pressure, reproducing the approximate→precise gap.

H16-3: The degradation is LARGER for PA-heavy weights (pa:4,qd:3) than QD-heavy (pa:2,qd:3) because PA is more sensitive to cache-state accuracy.

## Parameters for Bayesian Optimization
1. `snapshot_refresh_interval` (0, 1, 5, 10, 50, 100 ms)
2. `lru_capacity` (1000, 10000, 100000, 1000000)
3. `pa_weight` (2-5)
4. `qd_weight` (2-5)
5. `kv_blocks` (2000, 5000, 132139)
