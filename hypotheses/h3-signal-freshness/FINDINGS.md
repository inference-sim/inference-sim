# H3: Signal Freshness — queue-depth vs kv-utilization

**Status:** Confirmed
**Tier:** 2 (high diagnostic value)
**Date:** 2026-02-20

## Hypothesis

> At high request rates, the queue-depth scorer should distribute requests more evenly than the kv-utilization scorer, because queue-depth updates synchronously (PendingRequests increments at routing time) while KV utilization only changes when batch formation allocates blocks (a lagging indicator).

## Result: Overwhelmingly Confirmed

At rate=5000 with 4 instances and 1000 requests:

| Scorer | TTFT Mean | TTFT P99 | Dist StdDev | HOL Blocking |
|--------|-----------|----------|-------------|--------------|
| queue-depth | 1290-1319ms | 2532-2604ms | 0.7-1.0 | 0 |
| kv-utilization | 2259-3644ms | 7870-12285ms | 142-226 | 1 per seed |

The effect is consistent across all 3 seeds: **1.7-2.8x worse TTFT mean, 3.0-4.7x worse P99, 200x+ worse distribution uniformity**.

## Root Cause Analysis

The staleness originates from BLIS's DES event ordering in `sim/cluster/cluster.go:160`:

```
// BC-4: Cluster events at time T processed before instance events at time T
if clusterTime <= instanceTime {
```

This means all routing decisions at tick T drain BEFORE any instance-level events (batch formation, KV allocation) at tick T.

### Signal freshness comparison

| Signal | Data Source | Updated By | Freshness |
|--------|-----------|------------|-----------|
| PendingRequests | ClusterSimulator (cluster-owned) | RoutingDecisionEvent.Execute() | **Synchronously fresh** — each routing decision sees all prior decisions |
| KVUtilization | sim.KVCache.UsedBlocks() (instance-owned) | makeRunningBatch() → AllocateKVBlocks() | **Stale** — only updates after batch formation, which is downstream of routing |

The `EffectiveLoad()` formula (`QueueDepth + BatchSize + PendingRequests`) was designed to compensate for stale instance-level QueueDepth/BatchSize by adding the synchronously-fresh PendingRequests term (issue #170). There is no analogous "pending KV blocks" estimate for KVUtilization.

### Why it's rate-dependent

At rate=100 (10ms between arrivals), instance events have time to process between routing decisions — the KV signal refreshes. At rate=5000 (200us between arrivals), hundreds of routing decisions happen between batch formation steps, all seeing the same stale KV utilization.

| Rate | KV/QD TTFT Ratio | KV Dist StdDev |
|------|------------------|----------------|
| 100 | 1.07x | 1.0 |
| 500 | 1.22x | 34.0 |
| 1000 | 1.10x | 31.1 |
| 2000 | 2.13x | 159.6 |
| 5000 | 1.71x | 142.2 |

### Snapshot refresh interval compounding

Adding periodic snapshot caching (`--snapshot-refresh-interval`) compounds the inherent DES staleness. At 2ms refresh, **830/1000 requests pile onto a single instance**:

| Interval | TTFT Mean | Dist StdDev | Worst Instance |
|----------|-----------|-------------|----------------|
| immediate | 2259ms | 142 | 423/1000 |
| 500us | 2540ms | 158 | 495/1000 |
| 1ms | 2879ms | 178 | 515/1000 |
| 2ms | 6908ms | 336 | 830/1000 |
| 5ms | 6409ms | 322 | 800/1000 |

### Combined scorers naturally mitigate

Any combination including queue-depth produces near-perfect balance — even `kv:5,qd:1`:

| Configuration | TTFT Mean | StdDev | Distribution |
|--------------|-----------|--------|--------------|
| queue-depth:1 only | 1319ms | 0.7 | [251, 250, 250, 249] |
| kv-utilization:1 only | 2259ms | 142.2 | [333, 423, 47, 197] |
| kv:5,qd:1 (KV dominant) | 1320ms | 1.2 | [252, 250, 249, 249] |
| llm-d default (pa:3,qd:2,kv:2) | 1320ms | 1.2 | [252, 250, 249, 249] |

**A stale scorer adds zero information regardless of weight.** At high rates, kv-utilization returns the same value for all instances, so it's a constant offset that doesn't affect argmax. Queue-depth's variance (from PendingRequests) determines the winner.

## Bugs Found

None. The signal freshness gap is a known architectural property of DES event ordering, and the system is correctly designed to mitigate it through PendingRequests. The llm-d default configuration is robust.

## Implications for Users

1. **Never use `kv-utilization` as the sole routing scorer** at high rates — it causes severe pile-on and HOL blocking
2. **Always include `queue-depth`** in weighted routing — the default `prefix-affinity:3,queue-depth:2,kv-utilization:2` is robust
3. **kv-utilization adds value only at lower rates** where batch formation has time to occur between routing decisions

## Reproducing

```bash
cd hypotheses/h3-signal-freshness
./run.sh
```

The `--snapshot-refresh-interval` flag (added in this PR) enables researchers to study staleness effects directly.
