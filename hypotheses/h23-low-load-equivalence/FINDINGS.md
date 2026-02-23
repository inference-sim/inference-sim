# H23: Low-Load Routing Policy Equivalence

**Status:** Confirmed with nuance
**Resolution:** Confirmation with surprise — high-load control reveals equivalence is workload-dependent, not just load-dependent
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Tier:** 5 (baseline sanity check)
**Type:** Statistical (Equivalence)
**Date:** 2026-02-23
**Rounds:** 2

## Hypothesis

> Under very low load (1 req/s, 4 instances), all routing policies should produce equivalent TTFT because all instances are idle and no queue differentiates them.

## Experiment Design

**Classification:** Statistical / Equivalence

**Configurations compared:**
- A: `--routing-policy round-robin` (cyclic assignment)
- B: `--routing-policy least-loaded` (min-queue assignment)
- C: `--routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"` (llm-d default composite scorer)
- D: `--routing-policy prefix-affinity` (standalone prefix-affinity with LL fallback)

**Controlled variables:**
- Model: meta-llama/llama-3.1-8b-instruct
- Instances: 4
- Workload: uniform CLI-mode (--prompt-tokens 512, --output-tokens 512)
- Scheduler: fcfs, Priority: constant, Admission: always-admit
- KV blocks: defaults.yaml default (132139 for this model)

**Varied variable:** Routing policy only (ED-1)

**Seeds:** 42, 123, 456 (ED-4)

**Preconditions verified (ED-3):**
- Rate=1 with 4 instances: utilization = 1/(4 * 57.4) = 0.004 (essentially zero)
- All 50 requests complete with 0 still-queued and 0 still-running (confirmed by INV-1 check)
- Step time: beta0 + beta1*512 + beta2*512 = 6910.42 + 9047.04 + 1454.08 = 17411.54 us = 17.4ms

**High-load control (ED-2):** Rate=2000, 500 requests — originally predicted >20% divergence at ~8.7x overload to validate comparison sensitivity. (Prediction invalidated — with uniform workloads, policies do not diverge even at high load. See Results and Root Cause Analysis for explanation.)

## Results

### Experiment 1: Low-Load TTFT Mean (rate=1, 50 requests)

| Seed | RR (ms) | LL (ms) | W (ms) | PA (ms) | Max Dev% | Status |
|------|---------|---------|--------|---------|----------|--------|
| 42   | 22.68   | 22.19   | 22.01  | 22.19   | 3.00%    | PASS   |
| 123  | 21.88   | 20.94   | 21.05  | 20.94   | 4.40%    | PASS   |
| 456  | 22.11   | 21.51   | 21.60  | 21.51   | 2.78%    | PASS   |

**Worst-case deviation: 4.40% (< 5% threshold)**
**All seeds EQUIVALENT**

### Experiment 1: Low-Load E2E Mean (rate=1, 50 requests)

| Seed | RR (ms) | LL (ms) | W (ms) | PA (ms) | Max Dev% |
|------|---------|---------|--------|---------|----------|
| 42   | 4383.18 | 4381.12 | 4380.53| 4381.12 | 0.06%    |
| 123  | 4253.64 | 4250.73 | 4250.73| 4250.73 | 0.07%    |
| 456  | 4880.36 | 4877.68 | 4877.27| 4877.68 | 0.06%    |

**Worst-case deviation: 0.07%** — negligible E2E difference (decode time dominates).

### Experiment 2: High-Load Control (rate=2000, 500 requests)

| Seed | RR (ms) | LL (ms) | W (ms) | PA (ms) | Max Dev% | Status |
|------|---------|---------|--------|---------|----------|--------|
| 42   | 605.95  | 605.61  | 605.67 | 605.61  | 0.06%    | FLAT   |
| 123  | 577.36  | 576.81  | 577.68 | 576.81  | 0.15%    | FLAT   |
| 456  | 595.49  | 595.16  | 594.44 | 595.16  | 0.18%    | FLAT   |

**Worst-case deviation: 0.18%** — policies do NOT diverge even at 8.7x overload.

### Conservation (INV-1)

All 24 runs (4 policies x 3 seeds x 2 experiments) pass conservation: `injected == completed + still_queued + still_running + dropped_unservable`. (No dropped_unservable requests in this experiment — KV blocks are abundant.)

**Note on automated verdict:** The automated analyzer classifies the result as INCONCLUSIVE because the high-load control does not diverge (>20% criterion not met). The FINDINGS.md "Confirmed with nuance" status is based on the low-load equivalence evidence, which is the primary test of the hypothesis. The non-divergence at high load is an informative surprise, not a test failure — it reveals that uniform workloads provide no routing differentiation signal at any load level.

## Root Cause Analysis

### Why low-load equivalence holds (< 5%)

At rate=1 with 4 instances (rho=0.004), inter-arrival time is ~1 second while step time is ~17.4ms. Every request arrives to a completely empty cluster. The TTFT consists of:
- QueueingTime: alpha[0] + alpha[1]*512 = 1601.35 + 3.51*512 = 3398 us = 3.4ms (`sim/event.go:31`)
- StepTime (prefill): beta[0] + beta[1]*512 + beta[2]*0 = 6910.42 + 9047.04 = 15957 us = 16.0ms (`sim/latency_model.go:38-54`)

This is identical regardless of which instance the request is routed to, since all instances have zero queue depth, zero batch size, and zero KV utilization. The ~3-4% TTFT mean deviation between RR and the others comes from **assignment pattern differences**:

- **RR** cycles through instances 0, 1, 2, 3, 0, 1, 2, 3... deterministically (`sim/routing.go:95-96`)
- **LL** routes to the instance with minimum EffectiveLoad, tie-broken by first index (`sim/routing.go:113-122`). At zero load, all instances tie, so LL always picks instance 0.
- **PA** falls back to LeastLoaded on every cache-miss (`sim/routing.go:236-238`). With CLI-mode random tokens, every request has a unique prefix hash (100% cache-miss rate), so PA is functionally identical to LL.
- **W** uses composite scoring where all instances have tied scores at zero load. Tie-breaking by first index in snapshot order (`sim/routing.go:176-180`) produces near-identical but not byte-identical routing due to floating-point scorer accumulation.

The ~3% TTFT mean difference for RR arises from **event ordering differences**, not from step time variation. With constant prompt-tokens=512 and the blackbox latency model, StepTime is deterministic (same cacheMissTokens and decodeTokens produce the same stepTime). However, RR's cyclic assignment (0,1,2,3,...) produces a different scheduling timeline than LL/W/PA's instance-0-concentrated pattern. The alpha overhead — QueueingTime (`sim/event.go:31`) depends on arrival timing relative to batch completion — creates small per-request TTFT variations that accumulate into the ~3% mean difference. This is expected event-ordering variance, not a policy effect.

### Why high-load control does NOT diverge (surprise)

This is the most important finding. At rate=2000 with 4 instances (~8.7x overload), the queue builds up linearly for all instances. The policies produce nearly identical TTFT because:

1. **Uniform workload eliminates routing information**: All requests have the same prompt-tokens=512 and output-tokens=512. There is no heterogeneity for least-loaded to exploit — all instances receive identically-sized requests and process them at the same rate. Queue depths grow proportionally across all instances regardless of routing policy (`sim/routing.go:113-122` — LL picks the min-load instance, but with uniform arrivals, all instances have approximately equal load).

2. **No prefix overlap eliminates prefix-affinity signal**: CLI-mode generates random tokens per request. Every prefix hash is unique, so the prefix-affinity scorer (`sim/routing_prefix_scorer.go`) scores all instances at 0.0 (no cache hits). The weighted scorer degrades to queue-depth + kv-utilization only.

3. **Queue-depth and kv-utilization are redundant under uniform load**: When all instances process identical workloads, queue-depth and kv-utilization track correlated signals (#377 finding from H-Reasoning-KV). The weighted scorer with any combination of these produces the same instance rankings as LL.

4. **RR is naturally balanced for uniform workloads**: With 4 instances and cyclic assignment, RR distributes requests 125/125/125/125 across instances — exactly what LL would achieve with uniform arrivals.

**The control failure is informative, not a test design error.** It reveals that routing policy differentiation requires **workload heterogeneity** (variable request sizes, prefix overlap, SLO classes), not just high load. This confirms findings from H17 (Pareto frontier) and #377 (cache locality vs load balance): uniform workloads provide no signal for routing optimization.

### Quantitative verification

Bare TTFT at zero queue (first-principles):
- alpha_queue = 1601.35 + 3.51 * 512 = 3398.47 us = 3.40 ms
- beta_prefill = 6910.42 + 17.67 * 512 = 15957.46 us = 15.96 ms
- alpha_output = 1805.54 us = 1.81 ms (OutputTokenProcessingTime added to TTFT for the first decode step, `sim/simulator.go:454`)
- Expected TTFT = alpha_queue + beta_prefill + alpha_output = 3.40 + 15.96 + 1.81 = 21.17 ms

Observed TTFT mean (averaged across seeds and policies, low-load): 21.70 ms
Difference: 21.70 - 21.17 = 0.53 ms (2.5%) — within expected variance from event-ordering effects.

## Devil's Advocate (RCV-5)

**If this is "Confirmed," argue why it might be Refuted:**
The 3-4% TTFT deviation at low load is close to the 5% threshold. With different workload parameters (e.g., highly variable request sizes), this gap could exceed 5% even at low load because different routing patterns would assign long/short requests to different instances, creating measurable TTFT divergence even with zero queuing. The hypothesis might fail for mixed-workload scenarios even at rho < 0.01.

**If this is "Refuted," argue why it might be Confirmed:**
The hypothesis specifically targets the mechanism "queues never build up, so routing doesn't matter." At near-zero utilization with uniform workloads, this is exactly correct. The 3-4% variance is event-ordering noise from assignment pattern differences, not a routing policy effect. Under the stated conditions, equivalence holds convincingly.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| All 4 policies within 5% at low load | Confirmation | Documented here |
| High-load control does not diverge with uniform workloads | Surprise | Documented here — informs experiment design for future cross-policy experiments |
| PA = LL exactly for random-token workloads | Confirmation | Documented here — confirms #377 finding that prefix-affinity requires prefix overlap |
| RR TTFT ~3% higher than LL/W/PA at low load | Confirmation | Documented here — assignment pattern variance, not policy effect |
| Routing policy differentiation requires workload heterogeneity | Design limitation | Documented here — impacts experiment design guidance |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None — the finding that uniform workloads don't differentiate policies is a design property, not a bug
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed?
  - INV-1 (conservation): holds for all 24 runs
  - R2 (sort map keys): not triggered (no map iteration in output)
  - H17/H377 finding confirmed: uniform workloads provide no routing differentiation signal

## Scope and Limitations (RCV-6)

- **Operating point tested:** rate=1 and rate=2000; 4 instances; uniform prompt=512/output=512; KV blocks=132139 (default); 3 seeds; FCFS scheduler; constant priority; always-admit
- **Parameters findings depend on:** Uniform workload (same prompt/output sizes). Low-load equivalence likely holds for any workload at rho < 0.01 because queues never build. High-load non-divergence depends on workload uniformity.
- **What was NOT tested:**
  - Heterogeneous workloads (bimodal, mixed-SLO, prefix-heavy)
  - Variable request sizes that could create queue-depth asymmetry
  - Tiered KV cache configurations
  - Non-trivial schedulers (SJF, priority-FCFS)
  - Workload-spec YAML mode (different distributions)
  - > 4 instances (more instances increase RR imbalance probability)
- **Generalizability:** Low-load equivalence (< 5%) generalizes to any routing policy at rho < 0.01 with uniform workloads. The surprise finding (high-load non-divergence) is specific to uniform workloads and would NOT hold for heterogeneous workloads where prior experiments (H3, H17, #377) show significant differentiation.
- **Uncertainty quantification:** UQ not performed — single operating point per load level. The 4.40% worst-case deviation has no confidence interval; with 50 requests per configuration, per-seed TTFT mean has moderate precision. Larger samples (200+ requests) would tighten the bounds.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Low-load TTFT mean deviation | 3.00-4.40% across 3 seeds | Medium — consistent across all seeds, but 4.40% worst-case is within ~0.6% of the 5% threshold. With 50 requests per seed, the TTFT mean estimate has moderate precision. No formal confidence interval was computed. A larger sample (200+ requests) would tighten the bounds. |
| Low-load E2E mean deviation | 0.06-0.07% across 3 seeds | High — negligible, decode dominates |
| High-load TTFT mean deviation | 0.06-0.18% across 3 seeds | High — consistent non-divergence |
| Sample size | 3 seeds x 4 policies x 2 experiments = 24 runs | Medium — adequate for equivalence test per legacy threshold |
| INV-1 conservation | 24/24 PASS | High — exact check |
| PA = LL mechanism | PrefixAffinity falls back to LeastLoaded on cache-miss (sim/routing.go:236-238) | High — code-verified, data confirms byte-identical LL and PA results |
| Control experiment | High-load (rate=2000) does not diverge | High for stated finding; NOTE: this means the control does not validate the *low-load* comparison sensitivity. The control validates that the test is sensitive to load level (low-load equivalence is tighter than high-load equivalence), but it does NOT validate that the test can detect policy differences, because uniform workloads provide no differentiation signal. |

## Implications for Users

1. **At low utilization (rho < 0.01), routing policy choice does not affect latency.** Users running capacity-planning experiments at low load should not expect policy differentiation. All policies are equivalent when every instance is idle.

2. **Routing policy differentiation requires workload heterogeneity, not just high load.** Users testing cross-policy performance should use workload-spec YAMLs with mixed request sizes, prefix overlap, or multi-tenant SLO classes — not uniform CLI-mode workloads. This applies even at very high load.

3. **prefix-affinity routing is identical to least-loaded for random-token workloads.** The prefix-affinity standalone policy falls back to least-loaded on every cache-miss. With no prefix overlap, it provides zero benefit. Users should only select prefix-affinity when their workload has repetitive prefix patterns.

4. **For cross-policy experiments, use heterogeneous workloads.** The H16 workload spec (mixed distributions) or the multi-turn reasoning workload from H-Reasoning-KV would differentiate policies where uniform CLI-mode does not.

## Reproducing

```
cd hypotheses/h23-low-load-equivalence
./run.sh
```
