# Research: Unified Adaptive Strategy (Iteration 5, 10 rounds)

## Problem Summary
Beat static `pa:3,qd:2,kv:2` (74.15ms combined, 127.65ms RAG) by combining SLO-aware routing, priority scheduling, and workload-dependent weights. Prior iterations showed: P2C is worse than full-scan, dynamic switching is unnecessary, scheduling co-opt is null at moderate load, cost-benefit mixes signals destructively.

## Background Findings
- **SLOBasedPriority has NO per-SLO differentiation** — BaseScore=0.0 for all classes
- **SJF uses raw token count**, not effective prefill cost — ignores cache state
- **AdaptiveWeightedScoring uses cost-benefit** (proven inferior to orthogonal PA+QD in iter 4)
- **Composable at CLI**: `--routing-policy adaptive-weighted --priority-policy slo-based --scheduler priority-fcfs`
- **Built-in mixed-SLO workloads** available in `scenarios.go`

---

# Idea 1: SLO-Differentiated Orthogonal Routing + SLO-Class Priority

## Core Strategy

### Layer 1: Per-SLO Routing Profiles with Orthogonal Scorers
Replace the `adaptive-weighted` profiles (which use cost-benefit) with profiles using independent PA+QD scorers:

| SLO Class | Profile | Rationale |
|-----------|---------|-----------|
| critical | `qd:5,kv:2` (NO PA) | 50ms budget → never sacrifice load balance for cache. Pure load-based routing. |
| standard | `pa:3,qd:2,kv:2` (default) | Balanced. PA self-corrects to 0 on cache miss. |
| sheddable | `pa:3,qd:2,kv:1` | Slightly less KV sensitivity. |
| batch | `pa:5,qd:1,kv:1` | 5s budget → aggressive cache exploitation. Queue penalty is negligible relative to budget. |
| background | `pa:5,qd:0.5` | Maximum cache throughput. Minimal load sensitivity. |

**Why orthogonal**: Iter 4 proved that independent PA+QD > pre-combined cost-benefit. The argmax over orthogonal dimensions gives more information for optimal instance selection.

### Layer 2: SLO-Class-Aware Priority Policy (NEW)
Create `SLOClassPriority` that maps SLO class to base scores:

```go
type SLOClassPriority struct {
    BaseScores map[string]float64  // SLO class → base priority
    AgeWeight  float64
}

func (s *SLOClassPriority) Compute(req *Request, clock int64) float64 {
    base := s.BaseScores[req.SLOClass]  // default 0.0 for unknown
    age := float64(clock - req.ArrivalTime)
    return base + s.AgeWeight * age
}
```

Base scores: `critical=10.0, standard=5.0, sheddable=3.0, batch=1.0, background=0.0`

With PriorityFCFS, this means:
- Critical requests are ALWAYS scheduled before batch requests (base 10 vs 1)
- Age-based anti-starvation: after 5s (5M ticks), a batch request gains +5.0 priority, matching a fresh standard request

### Layer 3: PriorityFCFS Scheduler
`--scheduler priority-fcfs` uses the SLO-class priority from Layer 2.

## Why This Compound Works (When Prior Didn't)

Iteration 3's scheduling co-opt failed because at ρ=0.51 with homogeneous SLO traffic, the router perfectly separated cache traffic. **With MIXED SLO traffic at HIGH load**:

1. Router sends critical requests to LOW-LOAD instances (via `qd:5,kv:2`)
2. Router sends batch requests to CACHE-WARM instances (via `pa:5,qd:1`)
3. When BOTH types arrive at the SAME instance (inevitable at high load), the scheduler puts critical FIRST (base score 10 > 1)
4. Critical request gets fast prefill (low queue wait + cache-warm instance)
5. Batch request tolerates the wait (5s budget)

## Hypotheses

**H1**: On mixed-SLO RAG workload (50% critical + 50% batch) at rate=400 (ρ≈0.82):
- Critical TTFT p99 ≤ 80% of RR critical TTFT p99
- Batch TTFT p99 ≤ 50% of RR batch TTFT p99 (aggressive cache exploitation)
- Overall: better than both static-default and RR

**H2**: SLO-class priority provides measurable critical-request acceleration
- Control: same routing profile but with `--priority-policy constant` (FCFS)
- Prediction: critical TTFT p99 20-40% lower with SLO-class priority

**H3**: Per-SLO routing profiles improve upon static-default for mixed workloads
- Static-default applies `pa:3,qd:2,kv:2` to ALL request types
- Per-SLO profiles apply `qd:5` for critical and `pa:5` for batch
- Prediction: 10-30% combined improvement from routing differentiation

**H4**: The compound (routing + scheduling) is super-additive for mixed-SLO
- Test: routing-only (with FCFS) vs scheduling-only (with static routing) vs both
- Prediction: compound > sum of individual effects by 10-20%

## Test Workload: Mixed-SLO RAG

```yaml
version: "2"
aggregate_rate: 400
clients:
  - id: critical-users
    slo_class: critical
    rate_fraction: 0.3
    prefix_group: shared-system-prompt
    prefix_length: 512
    input_distribution: {type: gaussian, params: {mean: 256, std_dev: 64, min: 64, max: 512}}
    output_distribution: {type: gaussian, params: {mean: 64, std_dev: 16, min: 16, max: 128}}
  - id: batch-rag
    slo_class: batch
    rate_fraction: 0.5
    prefix_group: retrieval-corpus
    prefix_length: 4096
    input_distribution: {type: gaussian, params: {mean: 128, std_dev: 32, min: 32, max: 256}}
    output_distribution: {type: gaussian, params: {mean: 256, std_dev: 64, min: 64, max: 512}}
  - id: standard-misc
    slo_class: standard
    rate_fraction: 0.2
    input_distribution: {type: gaussian, params: {mean: 512, std_dev: 256, min: 64, max: 2048}}
    output_distribution: {type: gaussian, params: {mean: 256, std_dev: 128, min: 32, max: 1024}}
```

High utilization (rate=400, 8 instances, ρ≈0.82) with 3 SLO classes: 30% critical (short prefix, latency-sensitive), 50% batch RAG (long prefix, throughput-sensitive), 20% standard (mixed).

## Component Isolation Matrix (9 configs)

| # | Config | Routing | Scheduler | Priority | Tests |
|---|--------|---------|-----------|----------|-------|
| 1 | RR baseline | round-robin | FCFS | constant | Universal baseline |
| 2 | Static-default | pa:3,qd:2,kv:2 | FCFS | constant | Current champion |
| 3 | Per-SLO routing only | per-SLO profiles | FCFS | constant | Routing differentiation |
| 4 | SLO priority only | pa:3,qd:2,kv:2 | priority-fcfs | slo-class | Scheduling differentiation |
| 5 | SJF only | pa:3,qd:2,kv:2 | sjf | constant | SJF baseline |
| 6 | Per-SLO + SLO priority | per-SLO profiles | priority-fcfs | slo-class | Routing + scheduling |
| 7 | Per-SLO + SJF | per-SLO profiles | sjf | constant | Routing + SJF |
| 8 | Full compound | per-SLO profiles | priority-fcfs | slo-class | All three layers |
| 9 | Adaptive-weighted (PR447) | adaptive-weighted | FCFS | constant | PR #447 baseline |

---

## Reviews for Idea 1

### Gemini Review (Idea 1)

**1. Per-request SLO-based routing in vLLM / llm-d today.**
vLLM itself has no per-request SLO routing -- it is a single-instance engine. llm-d's `epp` (Endpoint Picker Plugin) does support request-level metadata via headers (`x-priority`, model labels), but today's scoring plugins (queue-depth, KV-utilization, prefix-affinity) apply uniformly to all traffic. Per-SLO weight profiles would be a new capability. The closest production pattern is separate model deployments per SLO tier behind a gateway, which is wasteful. BLIS's per-SLO profile approach maps well to what llm-d *should* do but does not yet.

**2. Critical profile omitting prefix-affinity.**
This is the weakest design choice. In production, critical requests frequently carry system prompts (512-2048 tokens) shared across users. Dropping PA entirely means critical requests never exploit cache hits on those prompts, adding unnecessary prefill latency. The iter-4 finding that PA self-corrects to 0 on cache miss applies here: `pa:1,qd:5,kv:2` costs nothing when there is no cache hit but captures 10-40ms savings when there is one. The document's own Layer 1 rationale ("never sacrifice load balance for cache") conflates "weight cache heavily" with "include cache at all." Recommend `pa:1,qd:5,kv:2` for critical.

**3. Router-level profiles vs scheduler-level priority -- which do llm-d engineers prefer?**
Router-level, unambiguously. llm-d's architecture places all intelligence in `epp` (a Kubernetes Gateway API extension); the backend vLLM instances are treated as stateless. Changing scheduler behavior requires patching vLLM itself, which llm-d deliberately avoids. Per-SLO routing profiles are pure `epp` configuration -- deployable via ConfigMap, no engine restart. SLO-class priority scheduling (Layer 2) is interesting for BLIS research but has zero deployability path in llm-d today. **Lead with Layer 1 (routing profiles) as the primary contribution; Layer 2 is simulator-only upside.**

**4. Single most impactful result for a SIG-LLM talk.**
Config 6 vs Config 2 at high utilization: show that per-SLO routing profiles + SLO priority achieves critical-request TTFT p99 within budget while batch TTFT p99 drops 40-50% from cache exploitation -- all on the *same cluster* that static-default cannot differentiate. The headline is "mixed-SLO workloads need mixed routing policies," which is obvious in retrospect but has no published quantitative evidence. The 9-config matrix is the proof; the story is Config 2 (one-size-fits-all) vs Config 6 (differentiated).

### GPT-4o Review (Idea 1)

**1. Isolation matrix: sufficient but missing one control.** The 9 configs cleanly isolate routing (configs 2 vs 3), scheduling (configs 2 vs 4), and their combination (config 6). However, there is no config that pairs SLO-class priority with RR routing (RR + priority-fcfs + slo-class). This control is needed to answer: "Does SLO-class priority help even without intelligent routing?" Without it, config 4's improvement over config 2 conflates priority scheduling with the static-default router's non-uniform load distribution. Config 8 vs 6 is also redundant -- they differ only in that 8 is described as "all three layers" but configs 6 and 8 use identical routing + scheduler + priority. Clarify or differentiate.

**2. Per-SLO rho is not 0.82 uniformly.** The workload has radically different per-class service profiles. Critical: 768 total input (256+512 prefix), 64 output. Batch: 4224 total input (128+4096 prefix), 256 output. Using beta=[6910, 17.67, 2.84]: critical prefill ~20.5ms + ~64 decode steps at ~6.9ms = ~462ms total service. Batch prefill ~81.6ms (but ~75ms saved on cache hit) + ~256 decode steps at ~6.9ms = ~1.85s total. At per-instance rates of 15 (critical) and 25 (batch) req/s, batch requests alone demand rho_batch = 25 * 1.85 / (1/0.0069) = ~0.32 per decode slot -- but this ignores batching amortization. The point: critical requests are 4x cheaper per-request than batch. The aggregate rho=0.82 masks that batch traffic dominates server time. The experiment should report per-class effective utilization to validate that the SLO-priority mechanism is actually exercised (i.e., critical and batch requests co-exist in queues frequently enough for priority to matter).

**3. The iter-3 diagnosis is incomplete.** "Homogeneous SLO" is a necessary condition for scheduling co-opt to fail, but the primary cause was moderate load (rho=0.51). At rho=0.51, queues are nearly empty most of the time -- PriorityFCFS has no work to reorder because there is rarely more than one request waiting. The correct diagnosis is: scheduling differentiation requires queue contention (multiple requests competing for the same batch slot), which requires either high per-instance load or bursty arrivals. Mixed SLO alone at rho=0.51 would still show null effect. The experiment should include a sub-saturation control (rate=200, rho~0.41) to confirm this -- if scheduling co-opt is null at low load even with mixed SLO, the load hypothesis is validated over the SLO-homogeneity hypothesis.

**4. Implementation feasibility.** SLOClassPriority is a clean policy-template addition (1 file, follows `SLOBasedPriority` pattern exactly). The per-SLO routing profiles already exist in `AdaptiveWeightedScoring` -- the proposal just replaces cost-benefit scorers with orthogonal PA+QD in `DefaultSLOProfiles()`. This is a config change within the existing factory, not a new policy. No new interfaces needed. The `BaseScores` map in SLOClassPriority should be unexported with an `IsValid` accessor per R8, and the map iteration in `Compute` needs a default-value guard per R1 (unknown SLO class should not silently return 0.0 -- either document this as intentional or log a warning).
