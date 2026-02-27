# Research: Joint Routing + Scheduling + KV Optimization (Iterations 6-10)

## Problem
Cache-aware routing (`pa:3,qd:2,kv:2`) dominates under abundant KV (iterations 1-5), but CAUSES KV pressure by concentrating requests on cache-warm instances. Under KV memory stress with CPU offloading:
- Reload latency is a BATCH-LEVEL cost (inflates step time for ALL co-batched requests)
- Cascading preemptions below ~500 blocks reset ProgressIndex to 0 (catastrophic for TTFT)
- Multi-turn KV pressure grows quadratically with rounds (~48 new blocks/round)

## Key Background Finding
**PendingTransferLatency and KVThrashingRate ARE on RoutingSnapshot** (snapshot.go:109-115). Both signals are available for custom scorer implementations. No RoutingSnapshot extension needed.

---

# Idea 6: Multi-Turn Baseline Under Varying KV Pressure

## Strategy
Establish the baseline: how does static `pa:3,qd:2,kv:2` perform on multi-turn workloads as KV blocks decrease? This identifies the critical KV block threshold where the static default starts failing.

## Experiment Design
Multi-turn 4-round reasoning workload (512 tokens/round, context accumulation), 4 instances, rate=100:
- **Normal KV** (132K blocks): No pressure, control baseline
- **Medium KV** (5000 blocks): Moderate pressure, some offloading expected
- **Low KV** (2000 blocks): High pressure, preemptions expected
- **Critical KV** (1000 blocks): Near cascading threshold

With CPU offloading: `--kv-cpu-blocks 3000 --kv-offload-threshold 0.8 --kv-transfer-bandwidth 10 --kv-transfer-base-latency 100`

Policies: RR, static-default (pa:3,qd:2,kv:2), static-kv-heavy (pa:2,qd:2,kv:5)

## Hypotheses
**H6-1**: Static-default performance DEGRADES non-linearly as KV blocks decrease. At normal KV, it dominates RR. At low KV, it may LOSE to RR because PA-driven concentration creates KV hot spots.
**H6-2**: A KV-heavy weight profile (`pa:2,qd:2,kv:5`) outperforms the default at low KV by spreading load across instances with more free blocks.
**H6-3**: With CPU offloading, PendingTransferLatency > 0 on some instances indicates active reloads that add batch-level step time inflation.

---

# Idea 7: KV-Pressure-Aware Routing Scorer

## Strategy
Create a new scorer `kv-pressure` that combines KVUtilization with PendingTransferLatency and KVThrashingRate into a single KV-health signal:

```go
func scoreKVPressure(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
    scores := make(map[string]float64, len(snapshots))
    for _, snap := range snapshots {
        // Base: 1 - KVUtilization (higher = more free blocks)
        kvHealth := 1.0 - snap.KVUtilization

        // Penalty for pending CPU→GPU reloads (batch-level cost)
        if snap.PendingTransferLatency > 0 {
            // Each 1000 ticks of transfer latency = ~1ms added to step time
            transferPenalty := float64(snap.PendingTransferLatency) / 10000.0
            kvHealth -= transferPenalty
        }

        // Severe penalty for thrashing (offload-reload cycles < 1000 ticks)
        if snap.KVThrashingRate > 0 {
            kvHealth -= snap.KVThrashingRate * 2.0  // thrashing rate 0.5 → -1.0 penalty
        }

        // Clamp to [0, 1]
        if kvHealth < 0 { kvHealth = 0 }
        if kvHealth > 1 { kvHealth = 1 }
        scores[snap.ID] = kvHealth
    }
    return scores
}
```

Test with `pa:3,qd:2,kv-pressure:3` instead of `pa:3,qd:2,kv:2` under KV pressure conditions.

## Hypotheses
**H7-1**: `kv-pressure` scorer outperforms plain `kv-utilization` under offloading conditions because it penalizes instances with active reloads (batch-level latency cost).
**H7-2**: The KV-pressure scorer has zero effect under normal KV (PendingTransferLatency=0, KVThrashingRate=0 → degenerates to kv-utilization).

---

# Idea 8: Joint Scheduling + KV Under Pressure

## Strategy
At KV pressure, queues ACTUALLY form (preemptions and reload delays slow processing). This is the regime where SLO-class priority should finally matter. Test the compound:
- Routing: `pa:3,qd:2,kv-pressure:3`
- Scheduler: `priority-fcfs`
- Priority: `slo-class` (critical=10, batch=1)

## Hypotheses
**H8-1**: Under KV pressure (2000 blocks), SLO-class priority + PriorityFCFS reduces critical-request TTFT p99 by 20-40% vs FCFS, because preemption-induced queue buildup creates material scheduling opportunities.
**H8-2**: The compound (KV-pressure routing + SLO priority) is super-additive — routing avoids KV-stressed instances while scheduling prioritizes critical when co-location is forced.

---

# Idea 9: CPU Offloading Avoidance with SLO Awareness

## Strategy
Route critical requests AWAY from instances with active offloading (PendingTransferLatency > 0). Batch requests can tolerate the reload penalty. This creates SLO-aware KV-pressure routing without needing per-SLO weight profiles (which fragment cache affinity per iter 5 finding).

Implementation: modify the kv-pressure scorer to be SLO-aware:

```go
func scoreKVPressureSLO(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
    // Critical requests: heavy penalty for transfer latency
    // Batch requests: mild penalty (tolerate reloads for cache benefit)
    transferWeight := 1.0  // standard
    if req != nil {
        switch req.SLOClass {
        case "critical": transferWeight = 5.0   // strongly avoid reloads
        case "batch":    transferWeight = 0.2   // tolerate reloads for cache
        case "background": transferWeight = 0.1
        }
    }
    // ... same as kv-pressure but with transferWeight multiplier on penalty
}
```

## Hypotheses
**H9-1**: SLO-aware KV-pressure routing protects critical TTFT without fragmenting cache affinity (unlike per-SLO weight profiles from iter 5 which hurt overall performance by 3-5%).
**H9-2**: Batch requests routed to offloading instances achieve HIGHER cache hit rates (reload from CPU preserves cache content) at the cost of increased reload latency (tolerable under 5s TTFT budget).

---

# Idea 10: Final Compound — Routing + Scheduling + KV + SLO

## Strategy: The Full Stack
Combine all mechanisms that survived review:
1. **Routing**: `pa:3,qd:2,kv-pressure-slo:3` — orthogonal scorers with SLO-aware KV pressure
2. **Scheduling**: `priority-fcfs` with `slo-class` priority (critical=10, batch=1)
3. **KV management**: CPU offloading with threshold=0.8, bandwidth=10, base_latency=100
4. **SLO differentiation**: At the KV-PRESSURE scorer level (NOT at the weight profile level — preserves cache affinity)

The key insight from iter 5: SLO differentiation belongs in the SCORER, not the weight profiles. Per-SLO weight profiles fragment cache affinity. A single SLO-aware scorer maintains one shared weight profile while adapting per-request.

## Component Isolation Matrix (10 configs)

| # | Config | Routing | Scheduler | Priority | KV Config | Tests |
|---|--------|---------|-----------|----------|-----------|-------|
| 1 | RR | round-robin | FCFS | constant | normal | Universal baseline |
| 2 | RR-low-KV | round-robin | FCFS | constant | low KV | RR under pressure |
| 3 | Static-default | pa:3,qd:2,kv:2 | FCFS | constant | low KV | Current champion under pressure |
| 4 | KV-heavy | pa:2,qd:2,kv:5 | FCFS | constant | low KV | Extra KV weight |
| 5 | KV-pressure | pa:3,qd:2,kv-pressure:3 | FCFS | constant | low KV | New KV-pressure scorer |
| 6 | +SLO-priority | pa:3,qd:2,kv-pressure:3 | priority-fcfs | slo-class | low KV | + scheduling |
| 7 | +SLO-KV-scorer | pa:3,qd:2,kv-pressure-slo:3 | FCFS | constant | low KV | SLO-aware scorer |
| 8 | Full compound | pa:3,qd:2,kv-pressure-slo:3 | priority-fcfs | slo-class | low KV | Everything |
| 9 | Full + offload | pa:3,qd:2,kv-pressure-slo:3 | priority-fcfs | slo-class | CPU offload | + tiered |
| 10 | Static + offload | pa:3,qd:2,kv:2 | FCFS | constant | CPU offload | Default under offload |

## Test Workloads
1. **Multi-turn 4-round**: 512 tokens/round, context accumulation, mixed SLO (30% critical + 70% batch), 4 instances
2. **KV configs**: normal (132K), medium (5000), low (2000), offload (2000 GPU + 3000 CPU)

---

## Reviews for Idea 6

(Reviews will be appended by judges)
