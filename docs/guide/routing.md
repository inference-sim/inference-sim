# Routing Policies

This guide covers how BLIS distributes incoming requests across instances in cluster mode. For single-instance simulation, routing is not applicable.

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
| **Prefix-affinity** | `prefix-affinity` | Route to the instance most likely to have matching KV cache |
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
| `queue-depth` | Effective load: `QueueDepth + BatchSize + PendingRequests` (min-max normalized) | queue-scorer |
| `kv-utilization` | Inverse KV utilization: `1 - KVUtilization` | kv-cache-utilization-scorer |
| `load-balance` | Inverse transform: `1 / (1 + effectiveLoad)` | BLIS-native (no llm-d equivalent) |

### Default Profile

When `--routing-scorers` is not specified, the default profile is:

```
prefix-affinity:3, queue-depth:2, kv-utilization:2
```

This matches the llm-d Endpoint Picker scoring pipeline. Weights are relative — only ratios matter. `[3, 2, 2]` behaves identically to `[0.43, 0.29, 0.29]`.

## Signal Freshness

!!! warning "Not all signals update at the same speed"
    Different scorer signals update through different mechanisms, which matters at high request rates.

BLIS models three signal update modes (defined in `ObservabilityConfig`):

| Mode | Signals | Update Behavior |
|------|---------|----------------|
| **Immediate** | QueueDepth, BatchSize | Read from instance state at routing time — always fresh |
| **Periodic** | KVUtilization | Cached and refreshed on a timer (`--snapshot-refresh-interval`) |
| **Injected** | PendingRequests | Incremented immediately at routing time by the cluster simulator |

At 5,000 req/s with 4 instances, ~45 routing decisions occur between KV utilization updates (~9ms step time). If using `kv-utilization:1` alone, all 45 decisions see the same stale utilization → severe load imbalance (3x worse TTFT p99).

!!! tip "Safe zone for `--snapshot-refresh-interval`"
    Below **5ms** (~1 step time): no degradation. At 10ms: 14% TTFT p99 increase. At 100ms: +354%. The default composite profile (`prefix-affinity:3, queue-depth:2, kv-utilization:2`) is inherently resilient — queue-depth's Immediate signal corrects stale KV signals, mitigating ~99% of the effect.

## Instance-Level Scheduling

Routing decides *which instance* receives a request. Scheduling decides *what order* requests are processed within an instance:

| Scheduler | Strategy | Flag Value |
|-----------|----------|-----------|
| **FCFS** | First-Come-First-Served (default) | `--scheduler fcfs` |
| **Priority-FCFS** | Sort by priority (descending), then FCFS | `--scheduler priority-fcfs` |
| **SJF** | Shortest Job First — sort by input token count | `--scheduler sjf` |
| **Reverse-priority** | Pathological — lowest priority first (for testing) | `--scheduler reverse-priority` |

Scheduling interacts with routing: SJF can starve long prompts, Priority-FCFS interacts with SLO classes, and scheduler choice compounds with routing policy choice.

## When to Use Which Policy

| Workload | Recommended Policy | Why |
|----------|-------------------|-----|
| Uniform traffic, no prefix sharing | `least-loaded` or `weighted` with `queue-depth:1` | Load balance is the only signal that matters |
| RAG with shared system prompts | `weighted` with `prefix-affinity:3,queue-depth:1` | Prefix affinity maximizes KV cache reuse |
| Mixed SLO classes | `weighted` default + `--scheduler priority-fcfs` + `--priority-policy slo-based` | Routing distributes load; scheduling prioritizes critical requests |
| Low traffic (< 10 req/s) | Any | All policies produce equivalent results within 5% |

## Example: Comparing Policies

BLIS includes a routing comparison script:

```bash
chmod +x examples/routing-comparison.sh
./examples/routing-comparison.sh
```

This runs 5 configurations and shows TTFT p99, target distribution, and throughput for each. See `examples/routing-comparison.sh` for the full script.

## Further Reading

- [Cluster Architecture](../concepts/architecture.md) — how the routing pipeline works internally
- [Configuration Reference](../reference/configuration.md#routing-policy) — all routing flags
- [Interpreting Results](results.md) — understanding trace summaries and regret analysis
