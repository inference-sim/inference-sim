# Routing Policies

This guide covers how BLIS distributes incoming requests across instances in cluster mode. For single-instance simulation, routing is not applicable. For instance-level request ordering, see [Scheduling & Priority](scheduling.md).

```bash
# Quick example: compare round-robin vs weighted routing
./blis run --model qwen/qwen3-14b \
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
--routing-policy weighted --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1"
```

### Available Scorers

| Scorer | What It Measures | llm-d Equivalent |
|--------|-----------------|------------------|
| `prefix-affinity` | Proportional prefix match ratio via router-side block hash cache | prefix-scorer |
| `precise-prefix-cache` | Actual KV cache state query with min-max normalization; all-equal (including all-zero) → 1.0 (llm-d parity) | precise-prefix-cache-scorer |
| `no-hit-lru` | LRU positional scoring for cold requests (warm = 0.5) | no-hit-lru-scorer |
| `queue-depth` | Queue depth: `QueueDepth` only (min-max normalized) | queue-scorer |
| `kv-utilization` | Inverse KV utilization: `1 - KVUtilization` | kv-cache-utilization-scorer |
| `load-balance` | Inverse transform: `1 / (1 + effectiveLoad)` | BLIS-native (no llm-d equivalent) |
| `active-requests` | In-flight requests: `(maxCount - count) / maxCount` | active-request-scorer |
| `running-requests` | Batch size (min-max normalized) | running-requests-size-scorer (GIE) |
| `load-aware` | Queue depth (linear threshold-capped, range [0, 0.5]) | load-aware-scorer |
| `vllm-dp` | vLLM data-parallel routing: `waiting × 4 + running` (inverted min-max) | DPLBAsyncMPClient.get_core_engine_for_request |

!!! note "Prefix-affinity is a scorer, not a standalone policy"
    The `prefix-affinity` scorer operates within the `weighted` routing pipeline, composed with load-balancing scorers. It uses a router-side `PrefixCacheIndex` with proportional block hash matching and LRU eviction. Always pair it with at least one load-aware scorer (queue-depth or kv-utilization) to prevent cold-start pile-on.

### Default Profile

When `--routing-scorers` is not specified, the default profile is:

```
precise-prefix-cache:2, queue-depth:1, kv-utilization:1
```

This matches the llm-d production scoring pipeline. Weights are relative — only ratios matter. `[2, 1, 1]` behaves identically to `[0.5, 0.25, 0.25]`.

## vLLM Data-Parallel Routing

BLIS supports vLLM's internal load-balancing algorithm via the `vllm-dp` scorer:

```bash
# Oracle mode (immediate signals, default):
./blis run --model qwen/qwen3-14b --routing-scorers "vllm-dp:1"

# Realistic vLLM parity (100ms coordinator staleness):
./blis run --model qwen/qwen3-14b --routing-scorers "vllm-dp:1" \
  --snapshot-refresh-interval 100000
```

This replicates vLLM's `DPLBAsyncMPClient` behavior: selecting the instance with the lowest `waiting × 4 + running` score. The scorer applies inverted min-max normalization to convert vLLM's argmin selection to BLIS's argmax routing framework.

**Signal staleness:** vLLM's coordinator publishes instance stats every 100ms by default (`min_stats_update_interval_ms`). For realistic parity studies, use `--snapshot-refresh-interval 100000` (100ms in microseconds). The default `0` (immediate mode) represents oracle routing with no signal staleness.

**Note:** Do not compose `vllm-dp` with other scorers unless comparing hybrid strategies. For vLLM parity, use `vllm-dp:1` alone — composing with other scorers double-normalizes dimensions and breaks vLLM fidelity.

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

At high request rates, many routing decisions occur between KV utilization updates (step time varies by model — ~6ms for Qwen3-14B / H100 / TP=1 at low load, longer under batch saturation). If using `kv-utilization:1` alone, all decisions within one step see the same stale utilization — this can cause severe load imbalance.

!!! tip "Safe zone for `--snapshot-refresh-interval`"
    Below **5ms** (~1 step time): no degradation. At 10ms: 14% TTFT p99 increase. At 100ms: +354% (measured with the prior default `prefix-affinity:3, queue-depth:2, kv-utilization:2`). The default composite profile (`precise-prefix-cache:2, queue-depth:1, kv-utilization:1`) is resilient — queue-depth's QueueDepth signal complements stale KV signals. The exact mitigation percentage may differ from the prior profile due to different weight ratios and scorer staleness characteristics (`precise-prefix-cache` has its own staleness model via `--cache-signal-delay`). For staleness-critical deployments, consider adding `load-balance` which reads EffectiveLoad (includes synchronous InFlightRequests).

## When to Use Which Policy

| Workload | Recommended Policy | Why |
|----------|-------------------|-----|
| Uniform traffic, no prefix sharing | `least-loaded` or `weighted` with `queue-depth:1` | Load balance is the only signal that matters |
| RAG with shared system prompts | `weighted` default or `precise-prefix-cache:3,queue-depth:1` | Prefix-aware scoring maximizes KV cache reuse |
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
