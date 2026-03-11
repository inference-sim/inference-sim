# H-Joint-KV-Scheduling: Joint KV-Scheduling Optimization

**Status:** CONFIRMED (super-additive interaction at moderate KV pressure)
**Date:** 2026-03-11
**Seeds:** 42, 123, 456 (3 seeds per config)
**Total runs:** 48 (2x2x4 design, 3 seeds)

## Hypothesis

SLO-aware KV eviction (targeting the lowest-priority running request instead of the tail request) creates a **multiplicative interaction** with elastic priority batching under KV pressure, because the two mechanisms protect critical requests at different resource layers:
- Elastic batching protects critical at the **scheduling layer** (batch slot allocation)
- SLO-aware eviction protects critical at the **memory layer** (KV cache block allocation)

When both operate jointly, critical requests are fully shielded at both layers, and sheddable requests absorb both costs.

## Experiment Design

### Factorial design: 2 x 2 x 4

| Factor | Levels |
|--------|--------|
| Scheduling | large-batch (margin=0) vs elastic (margin=4.0, cb=10) |
| KV eviction | tail (default) vs SLO-aware |
| KV blocks | 5000 (abundant), 2000 (abundant), 1500 (moderate pressure), 1200 (heavy pressure) |

### Common parameters

- Model: meta-llama/llama-3.1-8b-instruct, TP=2, H100
- 4 instances, maxRunningReqs=64
- 300 requests at 300 req/s (120% capacity)
- Scheduler: priority-fcfs with static-class-weight
- Routing: prefix-affinity:3, queue-depth:2
- Workload: gamma CV=2.0, multi-turn (3 rounds, 500ms think, accumulate)
- SLO mix: 20% critical, 40% standard, 40% sheddable
- Input: gaussian mean=256, Output: gaussian mean=128

### KV pressure calibration

With multi-turn `accumulate` context growth (3 rounds), effective per-request KV usage grows:
- Round 1: ~24 blocks (384 tokens / 16 block_size)
- Round 2: ~48 blocks (768 tokens accumulated)
- Round 3: ~72 blocks (1152 tokens accumulated)

Average ~48 blocks/request. At maxRunningReqs=64: ~3072 blocks minimum.

| Level | Blocks | Headroom | Observed preemptions (baseline) |
|-------|--------|----------|------|
| 5000 | 5000 | +63% | 0 |
| 2000 | 2000 | -35% | 0 (no pressure in practice) |
| 1500 | 1500 | -51% | 22 |
| 1200 | 1200 | -61% | 71 |

Note: KV=5000 and KV=2000 produced **byte-identical** results across all configs and seeds, confirming both are in the "abundant" regime with no KV pressure.

## Key Results

### Primary table: Critical TTFT P99 (ms, mean across 3 seeds)

| KV Blocks | Baseline | KV-only | Batch-only | JOINT | Joint/Baseline |
|-----------|----------|---------|------------|-------|----------------|
| 5000 | 323.7 | 323.7 | 80.0 | 80.0 | 0.247x |
| 2000 | 323.7 | 323.7 | 80.0 | 80.0 | 0.247x |
| 1500 | 464.0 | 343.4 | 401.0 | **80.0** | **0.172x** |
| 1200 | 639.2 | 397.4 | 639.2 | 397.4 | 0.622x |

### Interaction analysis

| KV | Batch-only ratio | KV-only ratio | Joint ratio | Interaction | Type |
|----|-----------------|---------------|-------------|-------------|------|
| 5000 | 0.247 | 1.000 | 0.247 | 1.00x | ADDITIVE |
| 2000 | 0.247 | 1.000 | 0.247 | 1.00x | ADDITIVE |
| 1500 | 0.864 | 0.740 | 0.172 | **2.09x** | **SUPER-ADDITIVE** |
| 1200 | 1.000 | 0.622 | 0.622 | 1.00x | ADDITIVE |

### Cost to sheddable (TTFT P99 ms)

| KV | Baseline | JOINT | Cost ratio |
|----|----------|-------|-----------|
| 5000 | 651.8 | 1155.3 | 1.77x |
| 1500 | 828.6 | 1305.0 | 1.58x |
| 1200 | 1194.9 | 1637.9 | 1.37x |

### Per-seed detail at KV=1500 (the super-additive regime)

| Config | Seed 42 | Seed 123 | Seed 456 | Mean |
|--------|---------|----------|----------|------|
| baseline | 502.8 | 267.4 | 621.9 | 464.0 |
| kv-only | 350.9 | 267.4 | 411.9 | 343.4 |
| batch-only | 495.2 | 76.3 | 631.4 | 401.0 |
| JOINT | 79.1 | 76.3 | 84.5 | **80.0** |

The JOINT optimization collapses variance: standard deviation drops from ~180ms (baseline) to ~4ms (JOINT). All seeds converge to ~80ms.

## Mechanism Analysis

### Three KV pressure regimes

**Regime 1 (KV=5000, 2000): No KV pressure** -- SLO-aware eviction has no targets. Elastic batching alone provides the full benefit (4.0x critical improvement). Joint = batch-only. The two mechanisms are independent, interaction is exactly additive (1.00x).

**Regime 2 (KV=1500): Moderate KV pressure** -- Both mechanisms have targets. This is the "sweet spot" where:
1. Elastic batching preempts low-priority running requests to give critical batch slots (scheduling layer)
2. SLO-aware eviction preferentially evicts sheddable KV blocks to give critical memory (memory layer)
3. Neither mechanism alone is sufficient: batch-only still gets KV-blocked (critical wins batch slot but can't allocate KV), kv-only still gets queue-blocked (critical has KV but waits behind sheddable in the batch)
4. Together: critical bypasses both bottlenecks. **Interaction ratio: 2.09x** (super-additive).

**Regime 3 (KV=1200): Extreme KV pressure** -- KV pressure completely dominates. Elastic batching has no effect because the batch is never full in the scheduling sense -- KV exhaustion limits batch size to well below maxRunningReqs. The results show elastic-tail = large-tail (byte-identical) and elastic-slo = large-slo (byte-identical). Only SLO-aware eviction matters, and the interaction collapses to additive (1.00x).

### Why super-additive at moderate pressure?

The interaction is multiplicative (not merely additive) because the two mechanisms operate on **orthogonal resource dimensions**:

1. **Without both mechanisms**: A critical request can be blocked by sheddable at either layer. P(critical blocked) = P(batch blocked) + P(KV blocked) - P(both blocked). At moderate pressure, both probabilities are significant.

2. **With both mechanisms**: Critical is protected at both layers simultaneously. The probability of critical being blocked drops to nearly zero, even though each individual mechanism only partially reduces it.

3. **Variance collapse**: The most striking effect is variance reduction. batch-only has seed-dependent results (76ms to 631ms) because whether a critical request encounters KV pressure depends on the random sequence of arrivals. JOINT eliminates this sensitivity by ensuring critical always wins at the KV layer too.

### Throughput trade-off

The joint optimization does not come free:

| KV | Config | Throughput (req/s) |
|----|--------|-------------------|
| 1500 | baseline | 107.66 |
| 1500 | JOINT | 107.15 |
| 1200 | baseline | 97.51 |
| 1200 | JOINT | 88.04 |

At moderate pressure (KV=1500), throughput impact is negligible (-0.5%). At heavy pressure (KV=1200), throughput drops 9.7% because SLO-aware eviction increases total preemptions (103 vs 71), and preempted requests must re-prefill.

### Cluster-level vs per-SLO metrics diverge

| KV | Config | Critical TTFT P99 | Cluster TTFT P99 |
|----|--------|-------------------|-----------------|
| 1500 | baseline | 464.0 | 819.3 |
| 1500 | JOINT | 80.0 | 1235.9 |

The JOINT optimization **improves** critical TTFT P99 by 5.8x but **degrades** cluster-wide TTFT P99 by 1.5x. This confirms that cluster-level metrics alone would miss the SLO-differentiated benefit entirely. The degradation at cluster level occurs because sheddable requests absorb the cost: their TTFT P99 rises from 828.6 to 1305.0 ms.

## Conclusions

1. **Super-additive interaction confirmed** at moderate KV pressure (KV=1500, interaction ratio 2.09x). The joint optimization reduces critical TTFT P99 by 5.8x vs baseline, while each mechanism alone achieves only 1.15-1.35x.

2. **Three regimes**: The interaction strength depends on KV pressure level:
   - Abundant KV: elastic batching dominates, SLO eviction adds nothing (additive)
   - Moderate KV: both mechanisms contribute, interaction is super-additive
   - Extreme KV: KV eviction dominates, elastic batching adds nothing (additive)

3. **Variance collapse is the most robust signal**: JOINT reduces per-seed standard deviation from ~180ms to ~4ms at KV=1500. This is more operationally valuable than the mean improvement.

4. **Cluster-level metrics are misleading for SLO-differentiated policies**: Always use per-SLO metrics when evaluating priority-aware mechanisms.

5. **The cost is to sheddable, not throughput**: At moderate pressure, throughput is maintained (-0.5%) while sheddable TTFT P99 increases 1.58x. This is the designed trade-off.

## Reproducibility

```bash
cd hypotheses/h-joint-kv-scheduling
bash run.sh
python3 analyze.py results/
```

All runs deterministic (INV-6). Zero timeouts across 48 runs.
