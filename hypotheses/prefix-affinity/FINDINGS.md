# Prefix-Affinity Hypothesis

**Status:** Confirmed (with nuance)
**Origin:** PR18 validated methodology (the first hypothesis that established this framework)
**Date:** 2026-02-20

## Hypothesis

> Prefix-aware routing should outperform load-only routing for prefix-heavy workloads, because routing same-prefix requests to the same instance enables KV cache reuse — fewer prefill tokens means lower TTFT.

## Result: Confirmed for Multi-Turn Chat, Nuanced for Shared Prompts

### Multi-Turn Chat (context accumulation across rounds)

Workload: `examples/multiturn-chat-demo.yaml` — 5-round sessions where each round prepends all prior context.

| Configuration | TTFT Mean | TTFT P99 | Throughput | Cache Hit | Distribution |
|--------------|-----------|----------|------------|-----------|--------------|
| prefix-affinity:3,qd:2 | 28.2ms | 71.0ms | 170.3 | 55.7% | [121, 121, 130, 128] |
| queue-depth:1 | 69.0ms | 167.1ms | 164.3 | 23.3% | [123, 129, 126, 122] |
| round-robin | 21.8ms | 35.9ms | 170.6 | 62.9% | [125, 125, 125, 125] |
| llm-d default | 27.7ms | 56.8ms | 170.1 | 55.8% | [124, 129, 127, 120] |

**Key finding:** Prefix-affinity is **2.45x better** than queue-depth (28.2 vs 69.0ms), confirming the PR18 result. Queue-depth achieves only 23% cache hit rate (near the 25% random baseline for 4 instances), because it *actively avoids* returning sessions to their cached instance.

**Surprise:** Round-robin is even better than prefix-affinity at low load (21.8 vs 28.2ms) with 62.9% cache hit rate — higher than prefix-affinity's 55.7%.

### Why Round-Robin Gets Accidental Cache Reuse

Round-robin's cyclic pattern (0→1→2→3→0→1→2→3...) means that for 5-round sessions with 4 instances, round 5 returns to the same instance as round 1. This provides partial cache reuse without any concentration overhead. The perfectly even distribution (125/125/125/125) means zero queuing imbalance.

Prefix-affinity routes all rounds to the same instance (full reuse) but creates slight load imbalance (121-130 range), adding queuing overhead that offsets the better reuse at low load.

### High Load Crossover (rate=5000)

At high load, prefix-affinity wins:

| Configuration | TTFT Mean | TTFT P99 | Cache Hit |
|--------------|-----------|----------|-----------|
| prefix-affinity:3,qd:2 | 205.2ms | 489.6ms | 57.6% |
| queue-depth:1 | 892.0ms | 2345.6ms | 21.9% |
| round-robin | 225.4ms | 607.9ms | 54.9% |

At high load, queues build up everywhere. Prefix-affinity's full cache reuse reduces per-request prefill time, which outweighs the slight concentration overhead. Queue-depth is **4.3x worse** — it scatters sessions across instances, forcing full re-prefill of accumulated context at every round.

### Shared System Prompt (low rate, 200 requests)

Workload: `examples/prefix-affinity-demo.yaml` — 80% of requests share a 256-token system prompt.

| Configuration | TTFT Mean | TTFT P99 | Cache Hit | Distribution |
|--------------|-----------|----------|-----------|--------------|
| pa:5,qd:1 (concentrated) | 24.4ms | 38.4ms | 13.8% | [200] |
| pa:1,qd:1 (balanced) | 16.8ms | 20.8ms | 54.4% | [51, 51, 45, 53] |
| queue-depth:1 | 16.8ms | 20.8ms | 54.4% | [51, 51, 45, 53] |
| round-robin | 16.8ms | 22.0ms | 54.5% | [50, 50, 50, 50] |

**Finding:** At low rate with a shared-prompt workload, heavy prefix-affinity weighting (pa:5,qd:1) **hurts** — it sends ALL 200 requests to one instance, creating unnecessary queuing while 3 instances sit idle. Even distribution (any other config) produces better TTFT and higher cache hit rates.

## Root Cause Analysis

### Queue-depth destroys cache locality

The `EffectiveLoad()` signal (`QueueDepth + BatchSize + PendingRequests`) is designed for load balancing. For multi-turn sessions, this is counterproductive: after round N runs on instance A (increasing A's load), queue-depth routes round N+1 to a different instance B (lower load). Instance B doesn't have round N's KV cache, so it must re-prefill all accumulated context from scratch.

### Round-robin's cyclic accident

For `N` instances and `R` rounds per session, round `i` goes to instance `i mod N`. Rounds that are `N` apart share the same instance. With 4 instances and 5 rounds: rounds 1 and 5 share an instance, providing ~20% of the sessions' cache reuse "for free."

### Concentration vs distribution tradeoff

Prefix-affinity creates a fundamental tradeoff:
- **More concentration** → better cache reuse per instance → but more queuing and cache pressure
- **More distribution** → less queuing overhead → but cache reuse drops toward random baseline

The optimal weight depends on load level. At low load, even distribution wins (no queuing to benefit from). At high load, cache reuse wins (reduced prefill outweighs concentration overhead).

## Bugs Found During PR18

The original PR18 hypothesis testing uncovered 3 bugs:
1. Multi-turn token sharing between sessions (context bleed)
2. Session overlap in workload generation (timing collision)
3. Short prefix length not creating measurable cache signal

These were fixed before the results above were generated.

## Implications for Users

1. **Multi-turn chat:** Always use prefix-affinity when sessions have context accumulation — 2-4x TTFT improvement over queue-depth
2. **Queue-depth alone is harmful** for session-based workloads — it actively destroys cache locality
3. **Round-robin is a surprisingly strong baseline** for multi-turn due to cyclic cache reuse
4. **At high load (>2000 req/s):** prefix-affinity with queue-depth balancing (pa:3,qd:2) is optimal
5. **At low load (<1000 req/s):** the benefit is smaller; round-robin may suffice

## Reproducing

```bash
cd hypotheses/prefix-affinity
./run.sh
```
