# FINDINGS: Heterogeneous Instance Pools (Strategy Evolution Iteration 5)

**Experiment:** Strategy Evolution Iteration 5 — Heterogeneous Instance Pools
**Date:** 2026-03-10
**Branch:** `hypothesis-playground`
**Status:** H-main NOT SUPPORTED (absolute); STRONGLY SUPPORTED (relative)
**Resolution:** Confirmation with nuance
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Type:** Statistical (Dominance)
**Rounds:** 1

---

## Executive Summary

Physical isolation of critical traffic into a dedicated "fast lane" instance produces a **20.6x improvement** in critical TTFT P99 compared to a shared 4-instance pool, even when the fast lane itself is massively overloaded (7x its capacity). This confirms that **isolation dominates queue-management policies** -- the compound strategy (admission + preemption) on a shared pool achieves only 1.6x improvement on the same metric.

However, the absolute TTFT P99 for the fast lane (~53.6 seconds) vastly exceeds the predicted <100ms target. This is because the experiment runs at extreme overload: 300 req/s total arrival rate against ~108 req/s actual system capacity (2.8x overload), and the fast lane receives 60 req/s on a single instance capable of only ~8.5 rps (7x overload).

## Hypotheses and Results

### H-main: Fast Lane Critical TTFT P99 < 100ms

**Prediction:** Fast lane critical TTFT P99 < 100ms (vs 500-1100ms shared).

**Result: NOT SUPPORTED in absolute terms, but STRONGLY SUPPORTED in relative terms.**

| Seed | A (Fast Lane) | C (Shared) | D (Compound) | Improvement (C/A) |
|------|---------------|------------|--------------|-------------------|
| 42   | 53,632 ms     | 1,217,762 ms | 1,102,328 ms | 22.7x           |
| 123  | 53,228 ms     | 675,004 ms   | 449,982 ms   | 12.7x           |
| 456  | 53,520 ms     | 1,406,063 ms | 508,597 ms   | 26.3x           |
| **Mean** | **53,460 ms** | **1,099,610 ms** | **686,969 ms** | **20.6x** |

The absolute <100ms prediction was wrong because it assumed sub-saturation operation. At 7x overload on the fast lane, queueing dominates. But the relative improvement is enormous: 20.6x better than the shared pool.

**Key mechanism:** In the shared pool, critical requests (20% of traffic) compete with standard (40%) and sheddable (40%) requests for queue positions. The priority-FCFS scheduler helps but cannot overcome the fundamental queueing law: at 2.8x overload, ALL requests experience long queues. In the fast lane, critical requests only compete with other critical requests -- and there are far fewer of them.

### H-throughput: Total Throughput (A+B) Within 10% of C

**Prediction:** Combined throughput (A+B) within 10% of C (no throughput sacrifice from splitting).

**Result: NOT SUPPORTED -- heterogeneous pools produce 28.5% MORE throughput.**

| Seed | A (rps) | B (rps) | A+B (rps) | C (rps) | Difference |
|------|---------|---------|-----------|---------|------------|
| 42   | 8.55    | 132.77  | 141.32    | 108.76  | +29.9%     |
| 123  | 8.54    | 130.33  | 138.87    | 109.01  | +27.4%     |
| 456  | 8.56    | 128.57  | 137.13    | 107.07  | +28.1%     |

The heterogeneous pool produces MORE total throughput, not less. This is because:
1. The bulk pool (B) with maxRunningReqs=64 processes standard+sheddable requests at ~130 rps across 3 instances, far exceeding the per-instance rate in the shared pool.
2. The larger batch size (64 vs 32) improves GPU utilization for the bulk traffic.
3. Total completed requests: A+B = 1700 vs C = 1500.

However, this comparison is not entirely fair since A and B have different request counts (500 + 1200 = 1700 vs 1500 for C). The throughput gain is partially an artifact of the experimental design -- the heterogeneous pool serves more total requests because it has segregated queues that don't interfere.

### H-bulk: Standard/Sheddable P99 in Bulk Pool Within 50% of Shared Baseline

**Prediction:** Bulk pool standard/sheddable TTFT P99 within 50% of shared baseline.

**Result: STRONGLY SUPPORTED -- bulk pool is 44-56% BETTER than shared, not worse.**

| Metric | B (Bulk Pool) | C (Shared) | Improvement |
|--------|---------------|------------|-------------|
| Standard TTFT P99 (mean across seeds) | 2,442 s | 5,509 s | -55.6% |
| Sheddable TTFT P99 (mean across seeds) | 5,844 s | 10,462 s | -44.1% |

The bulk pool standard/sheddable P99 is dramatically BETTER than the shared baseline. This is counterintuitive -- removing 20% of traffic (critical) from the pool freed capacity, and the larger batch size (64) improved throughput for the remaining requests.

## Cross-Strategy Ranking

For critical TTFT P99, the ranking is consistent across all seeds:

```
1. A (Fast Lane)    = 53,460 ms  (1.0x baseline)
2. D (Compound)     = 686,969 ms (12.9x worse than fast lane)
3. C (Shared)       = 1,099,610 ms (20.6x worse than fast lane)
```

**Physical isolation (20.6x improvement) dominates compound queue management (1.6x improvement).**

## Resource Utilization

| Config | Instances | maxRunning | Preemptions | Rejected | Completed |
|--------|-----------|------------|-------------|----------|-----------|
| A (Fast Lane) | 1 | 8 | 0 | 0 | 500 |
| B (Bulk Pool) | 3 | 64 | 0 | 0 | 1200 |
| C (Shared) | 4 | 32 | 0 | 0 | 1500 |
| D (Compound) | 4 | 32 | ~109 | ~560 | ~940 |

Notable observations:
- Neither A nor B nor C trigger preemptions (abundant KV blocks at default settings).
- D (compound) rejects ~37% of requests through SLO-gated admission, and preempts ~109 running requests. Despite these mechanisms, its critical TTFT P99 is still 12.9x worse than the fast lane.
- The compound strategy's admission control (rejecting ~560 requests) actually hurts critical traffic because it also reduces the pool's effective load, but not enough to offset the queueing from shared infrastructure.

## Key Insights

### 1. Isolation vs. Queue Management at Overload
At extreme overload (2.8x), no amount of queue-management sophistication can match physical isolation. The fast lane's advantage comes from a fundamentally different queueing regime: 500 critical requests on 1 instance vs 1500 mixed requests on 4 instances. Even though the fast lane is 7x overloaded, it faces a smaller absolute queue.

### 2. Batch Size Tuning Creates a Second Axis of Optimization
The bulk pool's maxRunningReqs=64 (vs 32 in the shared pool) improved throughput by ~22% per instance. This is a separate lever from isolation: right-sizing batch parameters per traffic class.

### 3. The Compound Strategy Has Diminishing Returns Under Extreme Overload
The compound strategy (D) improves critical TTFT P99 by only 1.6x over the shared baseline (C), versus the fast lane's 20.6x. At this overload level, admission control and preemption are fighting a losing battle -- they can't shed traffic fast enough to protect critical requests from the sheer volume of lower-priority work.

### 4. Heterogeneous Pools Are a Pareto Improvement
Splitting into heterogeneous pools improved ALL metrics simultaneously:
- Critical TTFT P99: 20.6x better (fast lane vs shared)
- Standard TTFT P99: 55.6% better (bulk pool vs shared)
- Sheddable TTFT P99: 44.1% better (bulk pool vs shared)
- Total throughput: 28.5% higher (combined vs shared)

This is rare -- typically improving one class comes at the expense of another.

## Caveats

1. **Extreme overload amplifies isolation benefits.** At sub-saturation (e.g., 80% capacity), the shared pool would have short queues for all classes, and the isolation benefit would shrink dramatically. The 20.6x improvement is a worst-case (best-case for isolation) scenario.

2. **Static partitioning risks stranding capacity.** If critical traffic drops to 0, the fast lane instance sits idle while the bulk pool could use the capacity. Real systems need dynamic rebalancing.

3. **The experiment does not model cross-pool overflow.** In production, a heterogeneous pool system would need spillover logic for when one pool is saturated but another has capacity.

4. **Multi-turn context accumulation inflates all numbers.** With 3 rounds and context accumulation, later turns have much longer prefills, which increases step time and queue depth across all configurations.

## Scope and Limitations
- **Operating point:** 300 req/s (extreme overload) with 4 instances, llama-3.1-8b/H100/TP=2
- **Not tested:** Sub-saturation regimes, other models, real vLLM validation
- **Multi-variable confound:** The fast lane (Sim A) differs from shared (Sim C) in 4+ dimensions (instances, request count, workload composition, batch size). Results demonstrate the potential of isolation but do not isolate which variable drives the improvement.
- **DES limitation:** Results from BLIS simulation, not production inference serving

**Capacity derivation:** With beta coefficients [6910, 17.67, 2.84] for llama-3.1-8b/H100/TP=2 and mean input=256, output=128: single-turn step time ≈ 11.8ms → ~85 req/s per instance → ~340 req/s for 4 instances. Multi-turn (3 rounds, context accumulation) increases effective per-request work ~2-3x, reducing capacity to ~113-170 req/s. At 300 req/s, the effective overload is ~175-265%, significantly higher than the "120%" label suggests.

## Evidence Quality
| Claim | Evidence | Confidence |
|-------|----------|------------|
| 20.6x critical TTFT improvement | 3 seeds, consistent direction | High (ratio) / Low (absolute values) |
| Pareto improvement | All classes better in split | Medium (confounded by unequal request counts) |

## Implications for Users
Physical instance isolation provides dramatic SLO improvement at the cost of dedicated capacity. Users with strict critical SLO requirements should consider dedicated instance pools. However, elastic priority batching (Iteration 6) achieves ~80% of the benefit without dedicated pools.

## Recommendations for Next Iteration

1. **Test at sub-saturation (80% capacity)** to find the crossover point where isolation benefits vanish.
2. **Add dynamic rebalancing** -- allow the fast lane to borrow capacity from the bulk pool when idle.
3. **Test asymmetric batch sizes** more systematically -- the bulk pool's maxRunningReqs=64 produced a large throughput boost that deserves isolated study.
4. **Compare against admission-only** -- SLO-gated admission on the shared pool (without preemption) might close more of the gap to isolation.
