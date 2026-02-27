# Problem: Joint Routing + Scheduling + KV Cache Optimization for Multi-Turn with CPU Offloading

## Context

After 5 iterations and 200+ experiments, we've proven that static `pa:3,qd:2,kv:2` is optimal for the composable scorer framework under STANDARD conditions (abundant KV, moderate load). But this leaves two major regimes UNTESTED:

1. **Multi-turn with context accumulation**: KV pressure grows with each round (512→1024→1536→2048+). As contexts grow, KV utilization approaches the offload threshold, triggering CPU offloading with reload latency penalties.

2. **CPU offloading under KV pressure**: When `--kv-offload-threshold` is exceeded, blocks move to CPU. Requests with offloaded blocks pay `PendingTransferLatency` (ms-scale delay). From H-Reasoning-KV (#388): 63.8% prefix cache hit rate in multi-turn, cascading preemptions at low KV blocks.

## The Fundamental Insight: Cache-Aware Routing CAUSES KV Pressure

The static default's strength (PA-driven cache affinity) becomes a weakness under KV pressure:
- PA concentrates requests on cache-warm instances
- Concentrated requests consume MORE KV blocks on those instances
- KV utilization rises past the offload threshold
- Offloading triggers reload latency → TTFT degrades
- Meanwhile, other instances have FREE KV blocks going unused

**This is the regime where the static default SHOULD fail and a more adaptive strategy SHOULD win.**

## Available Knobs for Joint Optimization

### KV Cache Knobs
- `--total-kv-blocks N` — total GPU KV blocks per instance (default: 132139 for llama-3.1-8b)
- `--kv-cpu-blocks N` — CPU-side blocks for offloading (default: 0 = disabled)
- `--kv-offload-threshold T` — GPU utilization threshold for offloading (0.0-1.0)
- `--kv-transfer-bandwidth B` — blocks/s transfer rate
- `--kv-transfer-base-latency L` — fixed overhead per transfer (μs)
- `--block-size S` — tokens per KV block (default: 16)

### Routing Signals for KV Awareness
- `KVUtilization` — current fraction of GPU KV blocks used (Periodic refresh)
- `FreeKVBlocks` — absolute number of free GPU blocks (Periodic)
- `CacheHitRate` — fraction of cache hits (Periodic)
- **NOT available**: `PendingTransferLatency`, `KVThrashingRate` (on KVStore only, not RoutingSnapshot)

### Scheduling and Priority
- `--scheduler priority-fcfs` or `sjf`
- `--priority-policy slo-class` — critical=10, standard=5, batch=1
- RoutingDecision.Priority one-shot hint

### Multi-Turn Workload
- `sim/workload/reasoning.go` — generates multi-turn with context accumulation
- Each round's input = [prior_context + new_input]
- Context grows: 512→1024→1536→2048+ across rounds
- `SessionID` links rounds; PrefixCacheIndex matches accumulated context

## The Joint Optimization Strategy

### Hypothesis: KV-Pressure-Aware Routing Outperforms Static Default Under Memory Stress

When KV pressure is high:
1. **Route critical requests to KV-healthy instances** (high FreeKVBlocks, low KVUtilization)
   - Critical needs fast TTFT → avoid instances near offload threshold
2. **Route batch requests to cache-warm instances UNLESS near offload threshold**
   - If KVUtilization > 0.7: switch from PA-heavy to QD-heavy routing for batch too
3. **Use KV-utilization scorer with HIGHER weight under pressure**
   - Normal: `pa:3,qd:2,kv:2` → Under KV pressure: `pa:3,qd:2,kv:5`
4. **SLO-class priority + PriorityFCFS**: Critical requests scheduled first when they co-locate with batch

### Test Scenarios
1. **Multi-turn 4 rounds** (512 tokens/round, 4 instances, reducing KV blocks to force pressure)
   - Normal KV (132K blocks): No pressure, baseline
   - Medium KV (10K blocks): Moderate pressure, some offloading
   - Low KV (3K blocks): High pressure, frequent offloading, potential preemptions
2. **Multi-turn + CPU offloading** (--kv-cpu-blocks 5000 --kv-offload-threshold 0.8)
3. **Mixed SLO + multi-turn** (critical single-turn + batch multi-turn, under KV pressure)

### Success Criteria
- Beat static-default under KV pressure (medium and low KV scenarios)
- Prevent TTFT degradation for critical requests even as batch creates KV pressure
- Demonstrate that KV-aware routing reduces offloading frequency and reload latency
