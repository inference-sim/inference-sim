# Discovered Principles

Principles extracted from 30 iterations of [Strategy Evolution](strategy-evolution.md) across two parallel experiment tracks: routing optimization (19 iterations) and scheduling optimization (11 iterations). Each principle is a concise, falsifiable statement grounded in experimental evidence.

!!! warning "Simulation Context"
    These principles were discovered within BLIS's discrete-event simulation model, which abstracts real inference serving systems. BLIS models vLLM-style recomputation-mode preemption (not swap-based), step-level batch formation, and queue-ordering scheduling. Principles involving preemption, chunked prefill overhead, or scheduling may not directly transfer to production systems with richer mechanisms (swap-based preemption, sub-step chunking, advanced memory management). Each principle's evidence column cites the specific experiment — consult the original findings for full context.

---

## Routing Principles (from 19 iterations, 1000+ experiments)

| # | Principle | Source | Evidence |
|---|-----------|--------|----------|
| RP-1 | **Orthogonal signals > pre-combined signals** — Independent PA+QD give the argmax more information than cost-benefit | Iter 4 | Cost-benefit scorer 29–134% worse than `pa:3,qd:2,kv:2` across all rate points |
| RP-2 | **Full N-way scan > Power-of-2-Choices at moderate scale** — At N≤16 with cheap snapshot reads, seeing all instances finds better cache+load combinations. At large N, P2C's O(1) cost may dominate. | Iter 1 | HCAR (P2C) 16% worse than static-default; misses 6/8 instances |
| RP-3 | **PA scorer self-corrects on cache miss** — Returns 0 when no cache match exists, degenerating to load-only | Iter 2 | Dynamic weight switching produces byte-identical results to static-default |
| RP-4 | **Uniform routing > SLO-differentiated routing** — Per-SLO profiles fragment cache affinity | Iter 5 | Adaptive+SLO priority 3–5% worse; fragments per-instance cache hit rate |
| RP-5 | **Routing dominates scheduling at moderate load** — Effective routing keeps queues short, leaving nothing for the scheduler to optimize | Iter 3, 5 | Priority scheduling had zero effect (byte-identical) when routing was effective |
| RP-6 | **KV-utilization as a routing scorer is counterproductive under memory pressure** — The `1-KVUtilization` formula routes away from instances with high occupancy, which are often the instances with the most valuable cached prefixes. Note: KV-utilization remains valuable as an admission/capacity signal; the finding is specific to its use as a routing preference scorer with approximate cache indexes. | Iter 6, 8 | Removing kv-util from routing scorer improved performance 4% AND made routing KV-pressure-immune (instance-level KV behavior unchanged) |
| RP-7 | **The optimal strategy is regime-dependent** — Normal KV: `pa:3,qd:2,kv:2`. Under pressure: `pa:3,qd:2`. At high load with admission: `pa:4,qd:3`. | Iter 8, 10 | Static default fails under KV pressure (23–25% worse than RR) |

!!! info "Reconciling with BLIS Defaults"
    The current BLIS default is `prefix-affinity:3,queue-depth:2,kv-utilization:2` (llm-d parity). Strategy Evolution discovered that `pa:4,qd:3` (no kv-util) performs better under KV pressure and at high load with admission control. The default is maintained for compatibility with the llm-d ecosystem. Users running Strategy Evolution experiments should consider the regime-dependent recommendation in RP-7 rather than assuming the default is optimal for all scenarios.

| # | Principle | Source | Evidence |
|---|-----------|--------|----------|
| RP-8 | **Approximate routing degrades under KV pressure** — PrefixCacheIndex diverges from actual KV state | Iter 6 | Validated by llm-d blog's 57x finding on approximate vs precise routing |
| RP-9 | **Admission control is the 3rd lever at high load** — Neither routing nor scheduling can reduce total queue depth; admission can | Iter 11 | Compound strategy beats RR by 47% at rate=2000 (admission shedding 30%) |
| RP-10 | **PA:QD ratio is the dominant parameter; empirical safety rule ≤1.33 for tested config** — Disproportionate PA without QD causes cascade failure. The 1.33 threshold was measured at 8 instances, 2000 req/s, 2x overload; the safe ratio will vary with cluster size, arrival rate, and prefix group cardinality. | Iter 13, 14 | `pa:4,qd:2` (ratio 2.0) → 3570ms cascade; `pa:4,qd:3` (ratio 1.33) → 132ms optimal |
| RP-11 | **Goodput > P99 as primary optimization metric** — Fair comparison when strategies have different completion rates | Iter 14 | GPT-4o review identified metric fairness issue |
| RP-12 | **Staleness immunity comes from signal independence** — PA reads synchronous PrefixCacheIndex, QD reads Immediate EffectiveLoad | Iter 16 | `pa:3,qd:2` produced identical 65.45ms across all staleness levels and KV pressures |
| RP-13 | **Bursty arrivals amplify admission control benefit** — Admission shedding during gamma bursts provides outsized relief | Iter 18 | Compound achieves 174ms (+65% vs RR's 496ms) under gamma CV=2.0 |
| RP-14 | **Compound advantage scales inversely with cluster size** — Smaller clusters see larger relative improvement | Iter 19 | N=4: +83.5%, N=8: +69.6%, N=16: +51.2% |

---

## Scheduling & KV Cache Principles (from 11 iterations)

| # | Principle | Source | Evidence |
|---|-----------|--------|----------|
| S1 | **Priority policy is the primary differentiator for orthogonal workloads** — Routing controls WHICH instance; priority controls ORDER within each instance | Iter 1 | 50.8% critical TTFT P99 improvement from priority alone |
| S2 | **Signal quality > mechanism complexity** — KVUtilization is degenerate at abundant blocks; prefix-affinity is the only proven high-signal scorer | Iter 1 | Three independent reviewers rated priority more impactful than routing |
| S3 | **Parameter count drives optimization difficulty** — 5–7 parameters is workable; 10+ risks local minima | Iter 1-opt, 8-opt | Bayesian search with 7 params converged in 30 calls; 10-param space did not |
| S4 | **Super-additivity must be tested, not assumed** — Pathological compounding (H24) does not guarantee the converse | Iter 1 | Ablation experiments essential for all compound strategies |
| S5 | **Starvation must be quantified with concrete crossover times** — "The age weight will prevent it" is not sufficient | Iter 1 | Piecewise-linear urgency with per-class thresholds; sheddable overtakes critical at ~1s |
| S6 | **Scheduling is zero-sum at saturation; admission is not** — In a work-conserving system at saturation, priority reordering that improves one class directly worsens another. Below saturation, this effect vanishes. Admission control is non-zero-sum: reducing total queue depth benefits all admitted requests. | Iter 1–3 | Cluster P99 degraded 62.4% despite 50.8% critical improvement (at near-saturation load) |
| S7 | **Load-adaptive gap is irrelevant at sustained near-saturation** — Load always exceeds threshold at 2000 req/s | Iter 2 | Load-regime adaptive strategy showed zero improvement over fixed-gap |
| S8 | **Admission gating breaks the "compute floor"** — Queue depth reduction benefits ALL tiers | Iter 3 | At t=20: critical TTFT 107ms, breaking through 132ms scheduling floor |
| S9 | **Chunking hurts the chunked request but helps others** — Disabling chunking for critical saves 14ms of beta0 overhead | Iter 4, 5 | Critical TTFT dropped from 132ms to 90ms with no throughput loss |
| S10 | **Per-SLO prefill thresholds are a zero-cost lever** — No throughput loss, no admission rejection | Iter 4 | Smarter allocation of prefill budget per SLO class |
| S11 | **Strategy generalizes across all workload shapes** — Non-orthogonal, asymmetric, multi-prefix all show 65–74% improvement | Iter 7 | Critical TTFT P99 always 88–96ms regardless of workload |
| S12 | **Strategy is immune to KV pressure down to ~3000 blocks** — Below that, critical degrades but still beats baseline by 50% | Iter 7 | No-chunk creates vulnerability under extreme KV pressure (large upfront allocation) |
| S13 | **Multi-prefix workloads create implicit SLO-instance specialization** — Prefix-affinity routes each SLO class to different instance clusters | Iter 7 | Critical gets "fast lane" instances; sheddable degradation worst (+84%) |
| S14 | **Asymmetric workloads (80% sheddable) show minimal sheddable degradation** — Few critical requests to jump ahead | Iter 7 | Only +5% sheddable degradation — best SLO tradeoff profile |
| S15 | **SLO-aware KV preemption has no moderate regime in BLIS's recomputation model** — Either zero preemptions (no effect) or livelock. This applies to recomputation-mode preemption (ProgressIndex reset to 0); real vLLM also supports swap-based preemption (KV blocks offloaded to CPU), which may exhibit a moderate regime that BLIS cannot observe. | Iter 9 | Negative finding: no preemption regime exists for shared-prefix workloads under recomputation preemption |
| S16 | **No-chunk benefits all tiers on multi-turn workloads** — Context accumulation amplifies prefill savings | Iter 10, 11 | Sheddable TTFT improved 49% on heavy multi-turn (5 rounds, 500ms think time) |

---

## How Principles Constrain Future Iterations

Principles function as **hard constraints** on subsequent iteration design:

- **RP-1** (orthogonality) prevented building a combined cache-load scorer in iterations 5–19
- **RP-6** (KV-util counterproductive) eliminated KV-utilization from all subsequent strategies
- **S6** (scheduling is zero-sum) redirected effort from scheduler optimization to admission control
- **RP-10** (PA:QD safety rule) prevented ratio violations in Bayesian search bounds

When a new iteration proposes a mechanism that contradicts an existing principle, it must either:

1. Provide experimental evidence that the principle doesn't hold in the new regime, or
2. Redesign to work within the principle's constraints
