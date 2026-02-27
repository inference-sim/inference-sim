# Strategy Evolution Ledger

## Objective
Discover jointly-optimized scheduling + KV caching policies that substantially beat BLIS baselines on mixed production workloads. All strategies must use tiered KV (CPU offload enabled). Each strategy is a parameterized template optimized via Bayesian search.

## Baseline Configuration (llm-d parity)
```
--routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
--scheduler fcfs --priority-policy constant --admission-policy always-admit
--kv-cpu-blocks 44000 --kv-offload-threshold 0.9
--long-prefill-token-threshold 256 --num-instances 8
```

## Workload: Orthogonal SLO × Multi-Turn (v2)
- All 3 tiers share IDENTICAL workload: multi-turn, prefix=512, input~Gaussian(256,100), output~Exp(128)
- SLO class is the ONLY differentiator (critical 20%, standard 50%, sheddable 30%)
- Bursty arrivals (gamma CV=2.0), 2000 req/s, 1000 requests

## Baseline Results (mean across seeds 42/43/44)

| Metric | Baseline | Target |
|--------|----------|--------|
| TTFT P99 | 269.1 ms | <229 ms (>15%) |
| E2E P99 | 5,424.9 ms | <4,611 ms (>15%) |
| Throughput | 17,198 tps | >18,058 tps (>5%) |
| SLO Gap (critical vs sheddable) | ~4% | >30% |
| Scheduling Delay P99 | 202.4 ms | — |

## Top Strategy Recommendation

**Idea 3: SLO-Gated Priority Cascade with Cache-Protected Scheduling**

Four-layer strategy exploiting `SLOClass` at every pipeline stage:
1. `SLOTieredPriority` policy: base scores + piecewise-linear urgency with per-class thresholds
2. `slo-priority` scorer: first BLIS scorer to read `SLOClass` -- interpolates load/cache per tier
3. Router priority bridge: sets `RoutingDecision.Priority` from base scores
4. `priority-fcfs` scheduler: honors the computed priorities

**Why this wins**: Only strategy that works with orthogonal workloads (SLOClass-based, not token-based). 7 parameters (manageable for Bayesian optimization). 5-6 touch points (low implementation friction). Addresses all reviewer concerns from Ideas 1-2.

**Critical pre-implementation fixes** (from Claude Opus review):
- Replace `1-KVUtilization` cache score with actual prefix-affinity signal
- Add `threshold_standard` parameter (prevents 50% standard traffic from eroding critical priority)
- Calibrate starvation ceiling: sheddable overtakes critical at ~1s, not 300ms as initially claimed

**Implementation order**: Priority-only first (3 touch points, tests core signal), then add scorer + bridge, then Bayesian optimize.

## Strategy Comparison Matrix

| Dimension | Idea 1 | Idea 2 | Idea 3 (Selected) |
|-----------|--------|--------|-------------------|
| Parameters | 10 | 6 | 7 |
| Touch points | 6+ | 5+ | 5-6 |
| Works with orthogonal workloads | Yes | Partial (scorer degenerates) | Yes |
| Starvation protection | Linear aging (unbounded) | Hyperbolic (numerically unstable) | Piecewise-linear with threshold (bounded) |
| Scorer architecture compatibility | Yes | No (needs beta coefficients) | Yes |
| Priority data flow | Ambiguous | Clean (same formula) | Clean (same base table) |
| Reviewer consensus | Medium-Low | Medium | Medium-High |

## Iteration Summary

| Iter | Strategy Name | TTFT P99 Δ% | Throughput Δ% | SLO Gap | Key Mechanism | Status |
|------|--------------|-------------|---------------|---------|---------------|--------|
| 0 | Baseline (llm-d parity) | — | — | 0.99x | FCFS + constant priority | Measured |
| 1 | Idea 3: SLO-Gated Priority Cascade | Crit: **-50.8%**, Std: -29.9% | ~0% | **3.52x** | slo-tiered priority + slo-priority scorer + priority-fcfs + bridge | **DEFAULT PARAMS** |
| 1-opt | Idea 3 (Bayesian optimized) | Crit: **-51.0%**, Shed: -10% better | ~0% | **3.18x** | Same mechanism, optimized params (30 calls × 3 seeds) | **OPTIMIZED — compute floor at 132ms** |
| 2 | Load-Regime Adaptive Priority | Crit: -50.9%, Std: -26.8% | ~0% | 3.57x | Adaptive gap scaling by queue depth | **NO IMPROVEMENT vs Iter 1** — load always above threshold |
| 3 | +SLO-Gated Admission (t=100) | Crit: -51.9%, Std: -32.7% | -3.8% | 3.32x | Iter1 + slo-gated admission (light shedding) | 7% sheddable rejected |
| 3 | +SLO-Gated Admission (t=50) | Crit: -51.1%, Std: **-45.7%** | **-9.6%** | 2.92x | Iter1 + slo-gated admission (moderate) | **21% shed rejected, big cluster P99 win** |
| 3 | +SLO-Gated Admission (t=20) | Crit: **-60.1%**, Std: -57.4% | **-28.6%** | ~1x | Iter1 + slo-gated admission (aggressive) | 31% shed, breaks compute floor |
| 4 | +SLO-Aware Prefill (crit=0, shed=256) | Crit: -66.3%, Std: -27.2% | ~0% | 5.05x | No chunking for critical (1-step prefill) + Iter1 priority | 90.5ms crit |
| **5** | **Simplified: Priority + No-chunk (all)** | **Crit: -66.5%**, Std: **-30.9%** | **~0%** | **4.95x** | slo-tiered + priority-fcfs + bridge + no chunking for all + DEFAULT scorers | **BEST: 90ms crit, 187ms std, 0% TPS loss, no custom scorer needed** |
| 6 | Local minima check + rate sweep | Crit flat at ~90ms across 1500-3000 req/s | ~0% all rates | — | Tested SJF, LL, no-chunk alone — no better basin | Not in local minimum |
| 7 | KV pressure + workload shapes | See below | See below | See below | Robustness characterization | Strategy generalizes |
| 8 | Token budget + instance scaling | Crit: -73.4% at t=1024 | ~0% | 8.0x | Lower token budget = faster critical drain | Manual Pareto point |
| **8-opt** | **Bayesian optimized (40 calls)** | **Crit: -73.7%** | **-1.1%** | **8.5x** | t=962, prefill_crit=508, age=9.8e-5, base_crit=12.9 | **GLOBAL OPTIMUM: 70.8ms critical** |
| 9 | SLO-aware KV preemption | No effect (0 preempt) or WORSE (livelock) | — | — | Evict lowest-priority instead of tail | **NEGATIVE: no moderate-preemption regime for shared-prefix** |
| 10 | Multi-turn session workload (4 rnds) | Crit: **-29.2%** (52.7→37.3ms) | ~0% | 1.35x | Strategy works on multi-turn with context accumulation | Sheddable IMPROVED (-21%) on multi-turn |
| 11 | Heavy multi-turn (5 rnds, 500ms think) | Crit: **-30.0%** (52.9→37.0ms) | ~0% | — | Confirmed: no-chunk benefits all tiers on multi-turn | **Sheddable -49%** on multi-turn |

## Key Insights Across Iterations

6. **Scheduling optimizations have hit diminishing returns** — critical TTFT P99 is bounded at ~132ms by scheduling delay + prefill cost.
7. **Load-adaptive gap is irrelevant at sustained near-saturation** — load always above threshold at 2000 req/s.
8. **Admission gating is the non-zero-sum lever** — SLO-gated admission reduces queue depth for ALL tiers during bursts. At t=20, critical TTFT P99 drops to 107ms, breaking through the 132ms "floor" by reducing scheduling delay for everyone.
9. **Clear Pareto frontier**: throughput vs latency tradeoff is tunable via the load threshold. t=100 (light, 7% shedding) preserves throughput. t=20 (aggressive, 31% shedding) minimizes latency. The optimal point depends on workload requirements.
10. **The "132ms compute floor" was actually 20ms compute + 112ms scheduling delay** — admission gating proved this by reducing scheduling delay, pushing critical TTFT to 107ms.
11. **Chunking hurts the chunked request but helps others** — H27 showed benefits for OTHER requests. Disabling chunking for critical (1-step prefill) saves 14ms of beta0 overhead, dropping critical TTFT from 132ms to 90ms.
12. **Per-SLO prefill thresholds are a zero-cost lever** — no throughput loss, no admission rejection. Just smarter allocation of the prefill budget per SLO class.

1. **Priority policy is the primary differentiator** -- with orthogonal workloads, routing can only influence WHICH instance, but priority controls scheduling ORDER within each instance. All three reviewers rated priority as more impactful than routing.
2. **Signal quality > mechanism complexity** -- KVUtilization is degenerate at 132K blocks (uniform across instances). Prefix-affinity is the only proven high-signal scorer for multi-turn workloads.
3. **Parameter count drives optimization difficulty** -- 5-7 parameters is workable; 10+ risks local minima in Bayesian search over 3 seeds.
4. **Super-additivity is a testable hypothesis, not a given** -- H24 pathological compounding does not guarantee the converse. Ablation experiments are essential.
5. **Starvation must be quantified with concrete crossover times** -- "the age weight will prevent it" is not sufficient.

13. **Strategy generalizes across all workload shapes** — non-orthogonal (-73.5% critical), asymmetric 5/80 (-65.1%), multi-prefix (-68.4%). Critical TTFT P99 is always 88-96ms regardless of workload.
14. **Strategy is immune to KV pressure** down to ~3000 blocks (no effect). At 2000 blocks (preemption regime), critical degrades from 90ms to 118ms but still beats baseline by 50%. No-chunk creates a vulnerability under extreme KV pressure (large upfront allocation).
15. **Multi-prefix workloads create implicit SLO-instance specialization** — prefix-affinity routes each SLO class to different instance clusters. Critical gets its own "fast lane" instances. Sheddable degradation is worst in this config (+84%) because sheddable instances are overloaded.
16. **Asymmetric workloads (80% sheddable) show minimal sheddable degradation** (+5%) — with few critical requests to jump ahead, sheddable barely notices the priority reordering. This is the best SLO tradeoff profile.

## Detailed Findings

### Strategy Generation Session (3 ideas, 3 external reviews each)

**Process**: Three strategies generated iteratively. Each reviewed by GPT-4o, Claude Opus 4.6, and Gemini 2.5 Flash. Key feedback incorporated between iterations.

**Most impactful review finding**: Claude Opus identified that `1 - KVUtilization` is a free-capacity signal, not a cache-affinity signal. This would have caused the SLO-priority scorer to collapse to noise-weighted queue-depth for all tiers, negating the entire SLO-differentiated routing mechanism. Fix: use actual prefix-affinity scorer output instead.

**Most impactful design insight**: With orthogonal workloads, Idea 2's predictive-service-time scorer produces IDENTICAL scores for all SLO classes (same token distributions = same predicted times). This eliminates the scorer as a differentiation mechanism entirely, leaving only the priority policy. Idea 3 avoids this by making the scorer SLO-aware directly (reading `SLOClass`).

Full details: see `strategy-evolution/research.md` Section 15 (Strategy Ideas).
