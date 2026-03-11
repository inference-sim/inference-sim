# Strategy Evolution: Elastic Batching Generalization (Iterations 7-8)

**Goal:** Determine where elastic priority batching (S28) generalizes and where it breaks down across workload dimensions.

## The 4 Generalization Questions

1. **Load regime**: Does elastic help at moderate load (80%) or only at overload (120%)?
2. **Session structure**: Does single-turn vs multi-turn affect the benefit?
3. **SLO composition**: Does the critical request fraction affect the benefit?
4. **Arrival pattern**: Does burstiness (Gamma) vs steady (Poisson) affect the benefit?

## Experiment Design

### Fixed parameters
- 4 instances, `--tp 2 --hardware H100`, `pa:3,qd:2` routing
- StaticClassWeight(10,5,1), PriorityFCFS scheduler
- 500 requests per config (sufficient for relative comparisons)
- Seeds: 42, 123, 456
- Input: Gaussian mean=256, output: Gaussian mean=128

### Two configurations per workload variant
- **Large-batch**: maxRunningReqs=64, no preemption (GPU-utilization baseline)
- **Elastic**: maxRunningReqs=64, preemption margin=4.0, circuit breaker=10

### Workload variants (12 total)

| Variant | Load | Arrival | Sessions | SLO Mix | Tests |
|---------|------|---------|----------|---------|-------|
| W1 | 80% | Gamma CV=2 | multi-turn | 20/40/40 | Load regime (moderate) |
| W2 | 120% | Gamma CV=2 | multi-turn | 20/40/40 | Base case (replication) |
| W3 | 80% | Gamma CV=2 | single-turn | 20/40/40 | Session structure (moderate) |
| W4 | 120% | Gamma CV=2 | single-turn | 20/40/40 | Session structure (overload) |
| W5 | 120% | Gamma CV=2 | multi-turn | 5/45/50 | Few critical (5%) |
| W6 | 120% | Gamma CV=2 | multi-turn | 50/30/20 | Many critical (50%) |
| W7 | 120% | Poisson | multi-turn | 20/40/40 | Steady arrivals |
| W8 | 120% | Gamma CV=4 | multi-turn | 20/40/40 | Heavy bursts |
| W9 | 200% | Gamma CV=2 | multi-turn | 20/40/40 | Extreme overload |
| W10 | 50% | Gamma CV=2 | multi-turn | 20/40/40 | Light load |
| W11 | 120% | Gamma CV=2 | multi-turn | 10/10/80 | Sheddable-heavy |
| W12 | 120% | Constant | multi-turn | 20/40/40 | Zero burstiness |

Total: 12 variants × 2 configs × 3 seeds = 72 runs

### Key metric
For each variant: **elastic_critical_P99 / large_batch_critical_P99** (the "elastic ratio")
- Ratio < 1.0: elastic batching helps
- Ratio ≈ 1.0: no effect
- Ratio > 1.0: elastic batching hurts

Also track batch occupancy ratio to verify GPU utilization is maintained.

### Predictions

| Variant | Predicted elastic ratio | Reasoning |
|---------|------------------------|-----------|
| W1 (80% load) | ~1.0 | Batch rarely full at moderate load → preemption rarely triggers (S25) |
| W2 (120%, base) | ~0.21 | Replicates Iter 6 finding |
| W3 (single-turn, 80%) | ~1.0 | Low load + no context growth → minimal queueing |
| W4 (single-turn, 120%) | <1.0 | Queue depth still matters without context growth |
| W5 (5% critical) | <1.0 but smaller effect | Fewer critical requests → fewer preemption opportunities |
| W6 (50% critical) | >1.0 or ~1.0 | Too many critical → preemption cascade, everyone preempts everyone |
| W7 (Poisson) | <1.0 | Steady arrival → consistent queue depth → consistent benefit |
| W8 (CV=4) | <1.0 | Heavier bursts → deeper queues → MORE preemption benefit |
| W9 (200% extreme) | <1.0 | Even more overload → even more preemption benefit |
| W10 (50% light) | ~1.0 | Sub-saturation → no queueing → no benefit (S25) |
| W11 (80% sheddable) | <1.0 | More sheddable = more preemption targets |
| W12 (Constant) | ~1.0 or <1.0 | No bursts but steady overload → some benefit |
