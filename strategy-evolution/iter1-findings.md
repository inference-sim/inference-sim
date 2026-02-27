# Iteration 1 Findings: SLO-Gated Priority Cascade

## What Worked (use in next iteration)
1. **SLO-tiered priority is the primary lever** — 50.8% critical TTFT P99 improvement from priority alone
2. **Piecewise-linear urgency with thresholds** prevents starvation (sheddable overtakes fresh critical at ~1s)
3. **priority-fcfs scheduler** correctly honors priority each step (re-evaluates, doesn't cache)
4. **Router priority bridge** (RoutingDecision.Priority) provides first-step ordering advantage
5. **Throughput is perfectly preserved** (~0% change) — reordering doesn't discard work

## What Could Be Better
1. **Cluster-wide TTFT P99 got worse** (269ms → 437ms) because sheddable degradation dominates the P99
2. **SLO-priority scorer signal may be weak** — queue-depth already captures load; the SLO bias adds marginal value
3. **Standard tier (50% of traffic) competes with critical** — need explicit threshold_standard tuning
4. **No cache-level awareness** — KV eviction still oblivious to SLO class
5. **Sheddable degradation is steep** (+76%) — could we bound it tighter?

## Key Metrics (default params, mean over 3 seeds)
- Critical TTFT P99: 132.3ms (-50.8% vs baseline 268.8ms)
- Standard TTFT P99: 189.9ms (-29.9% vs baseline 271.0ms)
- Sheddable TTFT P99: 466.1ms (+76.0% vs baseline 264.8ms)
- SLO gap: 3.52x (vs baseline 0.99x)
- Throughput: 17,213 tps (~0% vs baseline 17,198)

## Opportunities for Iteration 2
- **Preemption-aware scheduling**: At higher load, preemption becomes a factor. Can we preempt sheddable requests to free KV for critical?
- **Cache-aware SLO routing**: Route critical requests to instances that already have their prefix cached (combine prefix-affinity with SLO awareness more tightly)
- **Adaptive thresholds**: Instead of fixed thresholds, adapt based on current load regime
- **Batch-aware priority**: Consider how priority interacts with batch formation — can we influence which requests enter the next batch?
- **Two-queue architecture**: Separate fast-lane (critical) and bulk-lane (sheddable) queues per instance
