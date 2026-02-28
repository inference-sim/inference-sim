# Routing Policies

This guide covers how BLIS distributes incoming requests across instances in cluster mode. For single-instance simulation, routing is not applicable. For instance-level request ordering, see [Scheduling & Priority](scheduling.md).

```bash
# Quick example: compare round-robin vs weighted routing
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 100 --num-requests 500 \
  --routing-policy weighted --trace-level decisions --summarize-trace
```

## Available Policies

| Policy | Flag Value | Strategy |
|--------|-----------|----------|
| **Round-robin** | `round-robin` | Cyclic assignment — request N goes to instance N % k |
| **Least-loaded** | `least-loaded` | Send to the instance with lowest `EffectiveLoad` |
| **Weighted** | `weighted` | Composable multi-scorer pipeline (default: llm-d parity) |
| **Always-busiest** | `always-busiest` | Pathological template — sends to the most loaded instance (for testing) |

## Weighted Scoring (Composable Pipeline)

The `weighted` routing policy is the most flexible. It combines multiple scoring dimensions, each evaluating instances on a `[0, 1]` scale:

```bash
--routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
```

### Available Scorers

| Scorer | What It Measures | llm-d Equivalent |
|--------|-----------------|------------------|
| `prefix-affinity` | Proportional prefix match ratio via router-side block hash cache | prefix-scorer |
| `queue-depth` | Effective load: `QueueDepth + BatchSize + InFlightRequests` (min-max normalized) | queue-scorer |
| `kv-utilization` | Inverse KV utilization: `1 - KVUtilization` | kv-cache-utilization-scorer |
| `load-balance` | Inverse transform: `1 / (1 + effectiveLoad)` | BLIS-native (no llm-d equivalent) |

!!! note "Prefix-affinity is a scorer, not a standalone policy"
    The `prefix-affinity` scorer operates within the `weighted` routing pipeline, composed with load-balancing scorers. It uses a router-side `PrefixCacheIndex` with proportional block hash matching and LRU eviction. Always pair it with at least one load-aware scorer (queue-depth or kv-utilization) to prevent cold-start pile-on.

### Default Profile

When `--routing-scorers` is not specified, the default profile is:

```
prefix-affinity:3, queue-depth:2, kv-utilization:2
```

This matches the llm-d Endpoint Picker scoring pipeline. Weights are relative — only ratios matter. `[3, 2, 2]` behaves identically to `[0.43, 0.29, 0.29]`.

## Signal Freshness

> **Canonical source:** Signal freshness tiers are specified in [`docs/contributing/standards/invariants.md`](../contributing/standards/invariants.md) (INV-7). The descriptions below provide additional user-facing context; `invariants.md` is authoritative if they diverge.

!!! warning "Not all routing signals are equally fresh"
    In production inference serving systems (e.g., llm-d), the router is a separate process from the inference engines. Some signals are maintained at the router level, while others require periodic reporting from instances. BLIS models this asymmetry.

### Why Signals Have Different Staleness

Different signals originate from different places in the system:

- **Router-local signals** are maintained by the router itself — they're always current because the router controls them directly.
- **Instance-internal signals** live on the inference engine and must be communicated to the router — they're inherently stale by the reporting interval.

BLIS models three signal freshness tiers:

| Tier | Signals | Source | Freshness |
|------|---------|--------|-----------|
| **Router-local** | InFlightRequests, prefix cache index | Router increments InFlightRequests at dispatch, decrements at completion; prefix cache updated after each routing decision | Always fresh — router owns this state |
| **Instance-reported (Immediate/Periodic)** | QueueDepth, BatchSize, KVUtilization, FreeKVBlocks, CacheHitRate | Instance-internal state (scheduler queue, running batch, KV cache) | When `--snapshot-refresh-interval=0` (default): Immediate (read from instance at routing time). When `>0`: all Prometheus-sourced signals share the same Periodic refresh interval, matching real vLLM's single `/metrics` endpoint. |

!!! info "DES semantics of 'Immediate' mode"
    "Immediate" means "re-read from the instance object at query time" — NOT "perfectly synchronized with the simulation clock." At the same clock tick, cluster events are processed before instance events (determinism rule). So a routing decision at time T sees QueueDepth that hasn't yet processed instance events at time T. This is a determinism mechanism (INV-6), not a freshness guarantee.

### Staleness Impact

At 5,000 req/s with 4 instances, ~45 routing decisions occur between KV utilization updates (~9ms step time). If using `kv-utilization:1` alone, all 45 decisions see the same stale utilization — severe load imbalance (3x worse TTFT p99).

!!! tip "Safe zone for `--snapshot-refresh-interval`"
    Below **5ms** (~1 step time): no degradation. At 10ms: 14% TTFT p99 increase. At 100ms: +354%. The default composite profile (`prefix-affinity:3, queue-depth:2, kv-utilization:2`) is inherently resilient — queue-depth's Immediate signal corrects stale KV signals, mitigating ~99% of the effect.

## When to Use Which Policy

| Workload | Recommended Policy | Why |
|----------|-------------------|-----|
| Uniform traffic, no prefix sharing | `least-loaded` or `weighted` with `queue-depth:1` | Load balance is the only signal that matters |
| RAG with shared system prompts | `weighted` with `prefix-affinity:3,queue-depth:1` | Prefix affinity maximizes KV cache reuse |
| Mixed SLO classes | `weighted` default + [priority scheduling](scheduling.md) | Routing distributes load; scheduling prioritizes critical requests |
| Low traffic (< 10 req/s) | Any | All policies produce equivalent results within 5% |

## Example: Comparing Policies

BLIS includes a routing comparison script:

```bash
chmod +x examples/routing-comparison.sh
./examples/routing-comparison.sh
```

This runs 5 configurations and shows TTFT p99, target distribution, and throughput for each. See `examples/routing-comparison.sh` for the full script.

## Further Reading

- [Scheduling & Priority](scheduling.md) — instance-level request ordering
- [Admission Control](admission.md) — the gate before routing
- [Cluster Architecture](../concepts/architecture.md) — how the routing pipeline works internally
- [Configuration Reference](../reference/configuration.md#routing-policy) — all routing flags
- [Metrics & Results](results.md) — understanding trace summaries and regret analysis
