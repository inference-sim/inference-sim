# Problem: Unified Adaptive Strategy — SLO-Aware Routing + Priority Scheduling + Workload-Dependent Weights

## Context from Iterations 1-4

**Champion**: Static `pa:3,qd:2,kv:2` — 127.65ms RAG TTFT p99 (57% better than RR), 74.15ms combined.

**What failed and WHY it failed (critical for this iteration)**:
1. **P2C (iter 1)**: 2-candidate constraint misses cache. Full N-scan is better.
2. **Dynamic weight switching (iter 2)**: PA scorer already returns 0 when no cache → self-correcting.
3. **Priority scheduling (iter 3)**: NULL at ρ=0.51 because router perfectly separates cache-hit/miss traffic. **BUT**: at HIGH load (ρ>0.85), separation is IMPOSSIBLE → scheduling should help.
4. **Cost-benefit scorer (iter 4)**: Pre-mixing signals destroys orthogonality. Independent PA+QD is better.

**CRITICAL INSIGHT for iter 5**: Priority scheduling failed because we tested at the WRONG load level and with a HOMOGENEOUS SLO workload. The scheduling benefit requires:
- **High utilization** (ρ>0.85): Router forced to mix cache-hit and cache-miss on same instances
- **Mixed SLO traffic**: Critical + batch requests on the same instance → scheduler must choose which goes first
- **SLO-aware routing**: Critical requests should go to LOW-load instances (fast TTFT), batch requests should go to CACHE-warm instances (save compute)

## Available BLIS Knobs (all must be explored)

### Routing Layer
- `--routing-policy weighted` with `--routing-scorers pa:W1,qd:W2,kv:W3`
- Weight ratios: pa:1-5, qd:1-5, kv:0-3
- The PA scorer already adapts per-request (returns 0 for no-cache, >0 for cache)

### Scheduling Layer
- `--scheduler priority-fcfs` or `--scheduler sjf`
- PriorityFCFS: sorts by Priority descending, then arrival ascending
- SJF: sorts by estimated job size (input tokens) — naturally prioritizes cache-hit requests (fewer cache-miss tokens = shorter prefill)

### Priority Layer
- `--priority-policy slo-based`: Priority = BaseScore + AgeWeight × age
- SLO class maps to different base scores → critical gets scheduled before batch
- The RoutingDecision.Priority one-shot hint can ALSO be used (but gets overwritten by PriorityPolicy.Compute each step)

### SLO Tiers
- critical, standard, sheddable, batch, background
- Each can have different TTFT budgets (50ms, 200ms, 500ms, 5000ms, ∞)
- Mixed workloads with different SLO classes are supported via workload-spec YAML

### Workload Variations
- **RAG**: 5 prefix groups × 4096 tokens (long shared prefix, high cache benefit)
- **Agentic**: 10 prefix groups × 2048 tokens (medium prefix, many groups)
- **Mixed-SLO**: 50% critical (no prefix / short prefix) + 50% batch (long prefix)
- **Independent**: No shared prefix (negative control)
- **Bursty**: Gamma CV=3.5 arrival (H16 showed 1.25x TTFT p99 degradation)

### Instance Scaling
- N=4 (baseline), N=8 (standard), N=16 (scale test)

### KV Cache Pressure
- `--kv-cpu-blocks 5000 --kv-offload-threshold 0.8` (tiered KV with CPU offloading)
- Creates PendingTransferLatency and KVThrashingRate signals (not currently on RoutingSnapshot — could add)

## The Opportunity: SLO-Differentiated Routing + Scheduling

### Why mixed-SLO workloads unlock new optimization

With homogeneous SLO traffic (all "standard"), the router has one goal: minimize overall TTFT p99. With mixed SLO traffic:
- **Critical requests** (50ms budget): Must minimize TTFT. Should go to LEAST-LOADED instance, regardless of cache.
- **Batch requests** (5000ms budget): Can tolerate high queue depth. Should go to CACHE-WARM instance even if loaded.
- **The SAME instance may serve both**: At high utilization, there aren't enough idle instances for critical requests and enough cached instances for batch requests. The scheduler must resolve the conflict.

### SJF as natural cache-priority scheduling

SJF schedules by estimated job size (input token count). Cache-hit requests have fewer cache-miss tokens → shorter effective prefill → naturally scheduled first. This is the cleanest scheduling mechanism because:
- No cross-layer dependency (scheduler uses request metadata, not routing hints)
- Already implemented in BLIS (`SJFScheduler`)
- Physically grounded: shorter jobs SHOULD go first (optimal for mean latency)

### Weight-ratio optimization across workloads

From iter 4's rate sweep, static-default's TTFT p99 grows sub-linearly: 114ms→134ms for 100→500 req/s. But the OPTIMAL weights may differ:
- Low load (ρ<0.5): `pa:5,qd:1` (cache is free, exploit aggressively)
- Moderate load (ρ=0.5-0.8): `pa:3,qd:2` (balanced, the current default)
- High load (ρ>0.8): `pa:1,qd:3` (load balance dominates)

A workload-dependent meta-strategy could detect the load regime and auto-select weights.

## Hypotheses for this Iteration

H1: Mixed SLO workload (50% critical + 50% batch) with SLO-based priority + PriorityFCFS achieves:
- Critical TTFT p99 ≤ RR's critical TTFT p99 (never worse for latency-sensitive)
- Batch TTFT p99 ≤ 50% of RR's batch TTFT p99 (aggressive cache exploitation)

H2: SJF scheduler with weighted routing (pa:3,qd:2,kv:2) outperforms FCFS at high load (rate≥400)

H3: Weight ratio `pa:5,qd:2,kv:1` outperforms `pa:3,qd:2,kv:2` on RAG at low-moderate load but underperforms at high load

H4: A workload-adaptive meta-strategy that selects weights based on detected cache hit rate outperforms any fixed weight configuration across all workloads

H5: The compound (SLO routing + SJF scheduling + optimal weights) achieves 70%+ TTFT p99 reduction vs RR on mixed-SLO RAG workloads at high utilization

## Success Criteria
- Find a strategy that beats static-default (74.15ms combined) on BOTH cache-heavy and no-cache workloads
- Demonstrate that SLO-aware routing + scheduling provides measurable benefit for mixed-SLO workloads
- Identify the optimal weight profile for each workload regime
