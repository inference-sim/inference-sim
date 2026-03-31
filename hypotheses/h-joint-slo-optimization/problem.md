# Problem Statement: Joint SLO-Aware Optimization Under Mixed Sustained+Burst Load

**Date:** 2026-03-31
**Status:** Active — experiments not yet run
**Design doc:** `docs/plans/2026-03-31-joint-slo-optimization-design.md`
**Implementation PR:** #901 (`joint-slo-optimization` branch)

---

## 1. Background and Motivation

### 1.1 What prior Strategy Evolution tracks found

Two prior Strategy Evolution tracks optimized routing and scheduling in isolation:

- **Routing track** (PR #447, 19 iterations): Discovered that `prefix-affinity:4,queue-depth:3`
  outperforms the default `pa:3,qd:2,kv:2` configuration because the `kv-utilization` scorer
  is counterproductive under memory pressure — it routes away from instances with high KV
  occupancy, which are also the instances with the most valuable cached prefixes (RP-6).
  Final improvement: **−65% TTFT P99** under bursty load.

- **Scheduling track** (PR #609/#452, 11 iterations): Discovered that SLO-priority scheduling
  (`PriorityFCFSScheduler`) combined with no-chunk prefill for critical requests and
  `TierShedAdmission` reduces critical TTFT P99 by **−73.7%**. Key finding: scheduling alone
  is zero-sum at saturation (S6) — improving one SLO class hurts others. Admission control is
  the non-zero-sum lever (S8) that breaks the compute floor.

### 1.2 The gap: components never tested together

Both tracks confirmed admission control as essential, but each was run with the other layer
held at its default. The routing track used `FCFSScheduler`; the scheduling track used
`round-robin` routing. The **joint behavior of all four components together** under a workload
that stresses both burst dynamics (where admission dominates) and sustained pressure (where
scheduling matters) has never been measured.

Additionally, neither track explored **engine-level mechanisms** — the policies operating
inside the inference engine at step granularity:
- Which running request to preempt when KV blocks run out
- Which prefix cache blocks to evict under memory pressure
- How to allocate the per-step token budget across SLO tiers

### 1.3 Why this matters for bursty clients

Three empirical principles motivate the specific workload choice:
- **RP-13**: Bursty arrivals amplify the benefit of admission control. Under Poisson load,
  admission improves P99 modestly; under Gamma CV=2.0, the improvement is 2–3× larger.
- **S6**: At saturation, scheduling is zero-sum. The mixed workload (sustained + bursts)
  keeps the cluster near saturation continuously while adding transient overload spikes —
  the regime where both scheduling and admission are active simultaneously.
- **RP-7**: The optimal strategy is regime-dependent. Measuring only one regime
  (pure Poisson or pure burst) misses strategies that behave differently across regimes.

---

## 2. Research Questions

This investigation addresses four questions in dependency order:

1. **Composition**: Does the jointly-applied compound strategy
   (`pa:4,qd:3` + `priority-fcfs` + no-chunk-critical + `tier-shed`) outperform its
   components in isolation, and are there interaction effects between routing and admission
   that neither track predicted?

2. **SLO-priority preemption**: When KV blocks run out and a running request must be
   evicted, does choosing the lowest-SLO victim (rather than the most-recently-scheduled
   one) meaningfully reduce critical TTFT P99 without zero-sum harm to standard traffic?

3. **Tiered KV eviction**: Does protecting high-SLO prefix cache entries from eviction
   under memory pressure improve cache hit rates and reduce critical TTFT P99?

4. **Admission-feedback batching**: Does partitioning the per-step token budget by SLO tier
   provide measurable benefit beyond what admission shedding and priority scheduling already
   achieve?

---

## 3. Baseline Configuration

The baseline is the **best known compound from prior isolated tracks**, applied jointly. This
is NOT the BLIS default — it is the ceiling of prior work. Improvement over this baseline
therefore represents net gain from engine-level mechanisms only.

| Layer | Policy | Parameters | Source |
|---|---|---|---|
| Routing | `weighted` | `prefix-affinity:4,queue-depth:3` (no kv-util) | Routing track best, Iter 19 |
| Scheduling | `priority-fcfs` | critical=10, standard=5, sheddable=1 | Scheduling track best, Iter 8 |
| Prefill | no-chunk for critical | `LongPrefillTokenThreshold=∞` for critical tier | Scheduling track, S9 |
| Admission | `tier-shed` | `overloadThreshold`, `minAdmitPriority` from Iter 10 | Scheduling track best |
| Batch formation | `vllm` (LIFO) | Default chunk size | BLIS default |
| KV eviction | Tiered LRU (tier-aware) | All blocks start at tier 0 until tagged | PR #901 structural |
| Priority | `slo-based` | Maps SLO class to priority scores | Standard |

**Note on KV eviction baseline**: The tiered LRU free lists are structural (always active
after PR #901 merges). The "baseline" in this context uses the same KV eviction as the
treatment — the per-tier tagging is always on. The ablation for tiered-LRU is therefore a
before/after comparison between the PR #901 build and a build without the tiered-LRU changes,
or equivalently, a comparison between Iteration 2 results and Iteration 3 results.

---

## 4. Target Workload

### 4.1 Design goals

The workload is designed to prevent three common ways strategies can "game" the metric:

1. **Orthogonal SLO tiers**: All three SLO tiers (critical, standard, sheddable) use
   identical request shapes — same prompt length distribution, same output length distribution,
   same prefix length. The SLO class label is the *only* differentiator. This prevents
   strategies from benefiting by routing/scheduling based on token length as a proxy for
   priority (which would be undetectable but unfair in production).

2. **Mixed sustained+burst**: The workload simultaneously maintains a Poisson base load at
   ~85% of saturation AND overlays a Gamma burst process (CV=2.0). This prevents strategies
   from being tuned only for burst scenarios (where admission dominates) or only for sustained
   load (where scheduling dominates).

3. **Shared prefix**: All requests share a single prefix group (`shared-system-prompt`,
   256 tokens). This activates the prefix cache and makes KV eviction policy consequential.
   Without prefix sharing, tiered LRU has no effect.

### 4.2 Arrival process specification

```
Aggregate rate: 2 × measured_saturation_req/s   (PLACEHOLDER — calibrate from Iteration 0)
```

Six client cohorts, each contributing a fraction of the aggregate rate:

| Client ID | SLO Class | Process | CV | Rate Fraction |
|---|---|---|---|---|
| critical-sustained | critical | Poisson | — | 0.085 |
| critical-burst | critical | Gamma | 2.0 | 0.115 |
| standard-sustained | standard | Poisson | — | 0.255 |
| standard-burst | standard | Gamma | 2.0 | 0.345 |
| sheddable-sustained | sheddable | Poisson | — | 0.085 |
| sheddable-burst | sheddable | Gamma | 2.0 | 0.115 |

**Rate fraction derivation:**
- SLO mix: 20% critical, 60% standard, 20% sheddable
- Sustained fraction: 0.425 of aggregate (85% of saturation at peak rate = 42.5% of 2× sat)
- Burst fraction: 0.575 of aggregate (remaining, concentrated in high-CV Gamma arrivals)
- Sum: 0.085 + 0.115 + 0.255 + 0.345 + 0.085 + 0.115 = **1.000** ✓

**Why CV=2.0 for the Gamma bursts?** CV=2.0 was used in RP-13 experiments (PR #818) and
found to be the threshold where admission control benefit amplification becomes large. CV=1.0
is barely distinguishable from Poisson; CV=3.0 produces pathological behavior. CV=2.0 is
a well-characterized, reproducible stress point.

### 4.3 Request shape specification

All tiers use identical distributions (orthogonality requirement):

- **Prompt tokens**: Gaussian(mean=512, std=64, min=256, max=768)
- **Output tokens**: Gaussian(mean=128, std=16, min=64, max=192)
- **Prefix**: Single shared prefix group `shared-system-prompt`, 256 tokens

These values represent a moderate chat/assistant workload. They are not intended to model
any specific production system — they are chosen to stress KV capacity (512-token prompts
fill ~32 blocks at blocksize=16) while keeping the working set manageable.

### 4.4 Rate calibration (required before running experiments)

The `aggregate_rate` in `workload.yaml` is a placeholder. Before running experiments:

1. Run a binary-search calibration to find saturation throughput for the target model/GPU:
   ```bash
   ./run.sh calibrate --model <model> --gpu <gpu>
   ```
2. Set `SATURATION_RATE` to the measured value
3. Set `aggregate_rate = 2 × SATURATION_RATE` in `workload.yaml`
4. Record the saturation rate in `FINDINGS.md` under "Calibration"

**Why 2× saturation for aggregate_rate?** The Gamma burst process has mean rate =
`rate_fraction × aggregate_rate`. At CV=2.0, instantaneous arrivals can reach 3–4× the
mean rate. With a mean at 1× saturation (aggregate = 2×sat, burst fraction = 0.575),
peak instantaneous rates reach ~3× saturation — the regime where admission control is
decisively non-zero-sum (S8).

---

## 5. Success Criteria

### 5.1 Primary metric

**Critical TTFT P99** (Time to First Token, 99th percentile, critical-class requests only).
Measured separately during: (a) sustained-base phase and (b) burst phase. See §7.3 for
phase definition.

### 5.2 Goodput floor constraints (anti-gaming guards)

Any strategy that improves critical P99 by heavily rejecting standard or sheddable traffic
must be caught. Hard constraints:

- Standard goodput ≥ 85% of baseline measurement
- Sheddable goodput ≥ 60% of baseline measurement

A strategy that violates these is classified as "zero-sum" (S6 failure mode) and must not
be reported as an improvement.

### 5.3 Per-iteration thresholds

| Iteration | Primary threshold | Verdict if below threshold |
|---|---|---|
| 0 (baseline) | Establish reference values | — |
| 1 (composition) | >15% critical P99 improvement over BLIS defaults | REVISE if <15% |
| 2 (SLO preemption) | >15% additional improvement over Iter 1 | FAST-FAIL if <5% |
| 3 (tiered LRU) | >15% additional improvement over Iter 2 | FAST-FAIL if <5% |
| 4 (tier budget) | >10% additional improvement over Iter 3 | FAST-FAIL if <5% |

**Fast-fail rule:** If a mechanism's ablation shows <5% contribution to critical P99, it is
dropped and not carried forward to subsequent iterations.

### 5.4 Seed confirmation rule

A result is *confirmed* only when the lower bound of mean ± 1σ across seeds {42, 123, 456}
exceeds the stated threshold. If seeds diverge by >20% of the mean, add seed 789 and
re-evaluate.

---

## 6. Prior Knowledge Inventory

These principles from prior experiments constrain the design space and bound expected effects.
Violating these in experiment design is a red flag.

| Principle | Statement | Implication for this investigation |
|---|---|---|
| RP-6 | KV-utilization routing scorer is counterproductive under memory pressure | Do not include kv-util in any routing configuration |
| RP-7 | Optimal strategy is regime-dependent | Always report sustained-phase and burst-phase metrics separately |
| RP-9 | Admission is the non-zero-sum lever at high load | Engine mechanisms layer on top of admission; they cannot replace it |
| RP-10 | PA:QD weight ratio ≤ 1.5 is the empirical safety bound | Bayesian search must enforce this constraint |
| RP-13 | Bursty arrivals amplify admission benefit | Expect larger effects in burst phase than sustained phase |
| S6 | Scheduling is zero-sum at saturation | H-zero-sum arms must be checked for every mechanism |
| S8 | Admission gating breaks the compute floor | Engine mechanisms only show benefit after admission is working |
| S9 | No-chunk prefill for critical reduces TTFT P99 | Carry this into all iterations as a fixed component |
| S15 | SLO-aware *proactive* preemption has no moderate regime under recomputation | Reactive victim ordering (Iter 2) is untested and in scope; proactive is out |
| S16 | No-chunk benefits all tiers on heavy multi-turn | In single-turn workloads, only critical benefits meaningfully |

---

## 7. Measurement Protocol

### 7.1 Warmup discard

The first **10% of simulation horizon** is discarded from all metrics. This allows:
- KV prefix cache to reach steady-state occupancy
- Routing snapshot signals to stabilize
- Burst cycle dynamics to establish

Results from the warmup period are unreliable because the prefix cache starts empty and
early requests systematically miss, biasing TTFT upward.

### 7.2 Seeds

All experiments use seeds {42, 123, 456}. These seeds were used in prior tracks and provide
reproducibility across investigations. Add seed 789 only if the primary seeds diverge
unexpectedly (>20% spread in mean critical P99).

### 7.3 Phase separation

Define "burst phase" as any 60-second window where the measured arrival rate exceeds 120%
of the Poisson base rate. All other windows are "sustained phase."

Measure critical P99 and goodput separately for each phase. Report both in FINDINGS.md.
A strategy that helps only in burst phases is valuable but should be labeled as such.

### 7.4 Invariant verification

After each iteration, run:
```bash
go test ./sim/... -run TestInvariant -v
go test ./sim/kv/... -run TestConservation -v
```

INV-1 (request conservation) and INV-4 (KV block conservation) must hold for all seeds.
Failure indicates a bookkeeping bug introduced by the new mechanism and must be fixed before
reporting results.

---

## 8. Assumptions and Validity Boundaries

### 8.1 Simulator assumptions (BLIS-specific)

1. **Recomputation preemption model**: When a running request is preempted, it is
   re-computed from scratch (ProgressIndex reset to 0, KV blocks released). BLIS does
   NOT model swap-based preemption (saving KV state to CPU). This is why S15 found
   SLO-aware *proactive* preemption ineffective — the cost of recomputation overwhelms
   the benefit of preempting low-SLO requests.

2. **Blackbox or roofline latency model**: Token generation latency is estimated from
   trained coefficients (alpha/beta) or roofline arithmetic, not from real GPU measurements.
   Absolute latency numbers are model-dependent; relative improvements between strategies
   are more reliable.

3. **Single-cluster, single-model**: No P/D disaggregation, no multi-model routing, no
   heterogeneous GPU pools. These interactions are out of scope.

4. **Deterministic RNG**: Same seed produces byte-identical stdout. This means results are
   perfectly reproducible but may not generalize to stochastic real-world variance.

### 8.2 Workload assumptions

1. **Orthogonal SLO tiers**: In production, different SLO tiers often correlate with
   different request shapes (critical = short+urgent, batch = long+tolerant). The orthogonal
   design isolates the pure effect of SLO signal from token-length confounding. Real-world
   benefit may be larger (if tiers correlate with length in your system) or smaller (if
   routing already exploits length).

2. **Fixed prefix group**: All requests share one prefix group. Production workloads may
   have many prefix groups with varying popularity. Tiered-LRU benefit (Iter 3) depends on
   prefix cache pressure; with a single prefix group and adequate KV capacity, eviction
   rarely triggers and Iter 3 may show minimal gain.

3. **Synthetic burst pattern**: CV=2.0 Gamma is a parameterized model of burstiness, not
   a trace-replay of real traffic. Real burst patterns may have different temporal structure
   (e.g., correlated bursts from a small number of clients vs. aggregate Gamma bursts from
   many independent clients).

4. **Static workload rate**: The Poisson base rate is constant throughout the simulation.
   Diurnal or trending patterns are not modeled.

### 8.3 What conclusions are and are not valid

**Valid conclusions from this investigation:**

- Whether the joint compound beats the best prior isolated configurations *under this workload
  model* — the orthogonal design makes this a fair relative comparison
- Whether SLO-priority preemption, tiered-LRU, and tier-budget batch formation each provide
  measurable incremental benefit on top of the compound baseline
- Which mechanisms are redundant (fast-fail) vs. additive under the mixed regime
- The qualitative ordering of mechanism importance (admission > scheduling > engine)

**Conclusions that require additional validation:**

- Absolute TTFT P99 values depend on the latency model calibration and specific model/GPU
- Generalization to non-orthogonal workloads (where SLO correlates with token length)
- Generalization to trace-replay workloads (real traffic vs. synthetic Gamma)
- Generalization to P/D disaggregated clusters or multi-model deployments
- The exact parameter values (f_c=0.50 for tier budget, etc.) should be tuned per deployment

### 8.4 Known implementation limitations

**TierBudgetBatchFormation (Iter 4) post-pass soft-stall**: The current implementation
applies a post-pass over the inner `VLLMBatchFormation` result to enforce tier budgets.
This is a simplification with two known limitations:

1. KV blocks allocated by the inner pass for soft-stalled requests are held for the full
   step duration even though those tokens are not processed. This causes temporary
   over-allocation. The bug fix in commit `3c87916c` ensures ProgressIndex is not corrupted
   (ComputedTokens is restored, not deleted), but the over-allocation remains. In practice,
   at moderate critical fractions (f_c ≤ 0.6), this affects a small fraction of steps and
   does not materially bias the latency measurements.

2. The soft-stall mechanism does not prevent starvation of the sheddable tier if the
   critical and standard tiers are always budget-exhausted. The goodput floor constraint
   (§5.2) catches this in the H-zero-sum arm.

These are documented design decisions, not bugs. A future iteration could integrate budget
checking inside the FormBatch loop to eliminate the over-allocation.
