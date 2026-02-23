# H6: Counterfactual Regret — Round-Robin vs Weighted Routing

**Status:** Confirmed with wrong mechanism
**Resolution:** Confirmation with wrong mechanism — the hypothesis predicted weighted regret would be "significantly lower"; in fact it is structurally zero (a design property of the counterfactual computation, not an empirical finding about routing quality). The directional claim (RR > Weighted) holds trivially. Additionally, higher regret does not imply worse latency.
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Tier:** 2 (behavioral comparison)
**Type:** Statistical (Dominance)
**Date:** 2026-02-23
**Rounds:** 2

## Hypothesis

> "Counterfactual regret should be higher for round-robin than weighted routing under variable load."

Intuition: Round-robin ignores load entirely, so it should frequently route to suboptimal instances when load is asymmetric. The weighted policy with queue-depth scoring actively picks the best instance. The counterfactual analysis should quantify this difference.

## Experiment Design

**Classification:** Statistical / Dominance

**Configurations compared:**
- A (Round-Robin): `--routing-policy round-robin --scheduler fcfs --priority-policy constant --admission-policy always-admit --trace-level decisions --counterfactual-k 3 --summarize-trace`
- B (Weighted/queue-depth): `--routing-policy weighted --routing-scorers "queue-depth:1" --scheduler fcfs --priority-policy constant --admission-policy always-admit --trace-level decisions --counterfactual-k 3 --summarize-trace`

**Controlled variables:** Model (llama-3.1-8b-instruct), instances (4), workload (mixed 2-client: 70% short input=128/output=64, 30% long input=1024/output=256), scheduler (fcfs), priority (constant), admission (always-admit), KV blocks (defaults.yaml default for model), counterfactual-k (3)

**Varied variable:** Routing policy (round-robin vs weighted with queue-depth:1)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- `--trace-level decisions` enables trace recording (`cmd/root.go:654`)
- `--counterfactual-k 3` enables counterfactual analysis with top-3 candidates (`cmd/root.go:655`)
- `--summarize-trace` prints trace summary with regret metrics (`cmd/root.go:656`)
- Mixed workload creates load asymmetry: long requests take ~2.75x longer per step (25.7ms vs 9.4ms)

**Rate sizing:**
- Weighted mean step time: 0.7 * 9354us + 0.3 * 25731us = 14267us = ~14.3ms
- 4 instances capacity: ~280 req/s
- Rate 200 = ~0.71x utilization (moderate load, some queueing)
- Rate 100 = ~0.36x utilization (light load, minimal queueing — ED-2 control)

## Results

### Experiment 1: Core Comparison (rate=200, moderate load)

| Seed | Policy | Mean Regret | Max Regret | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | Jain FI |
|------|--------|-------------|------------|----------------|---------------|----------------|---------|
| 42 | RR | 2.124 | 12.0 | 28.46 | 64.68 | 1440.98 | 1.0000 |
| 42 | Weighted | 0.000 | 0.0 | 30.25 | 77.36 | 1443.16 | 0.9963 |
| 123 | RR | 2.116 | 13.0 | 30.06 | 80.92 | 1474.27 | 1.0000 |
| 123 | Weighted | 0.000 | 0.0 | 33.00 | 92.29 | 1476.21 | 0.9993 |
| 456 | RR | 5.362 | 27.0 | 25.39 | 57.46 | 1336.18 | 1.0000 |
| 456 | Weighted | 0.000 | 0.0 | 27.70 | 71.64 | 1331.78 | 0.9949 |

**Summary:** RR mean regret = 3.20 (all seeds > 0), Weighted mean regret = 0.000 (all seeds exactly 0). Directional consistency: RR > Weighted in all 3 seeds.

### Experiment 2: Low-Rate Control (rate=100, ~0.36x utilization)

| Seed | Policy | Mean Regret | Max Regret | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | Jain FI |
|------|--------|-------------|------------|----------------|---------------|----------------|---------|
| 42 | RR | 2.736 | 11.0 | 23.72 | 47.38 | 1295.78 | 1.0000 |
| 42 | Weighted | 0.000 | 0.0 | 25.50 | 56.41 | 1295.29 | 0.9981 |
| 123 | RR | 1.914 | 8.0 | 23.70 | 47.72 | 1317.75 | 1.0000 |
| 123 | Weighted | 0.000 | 0.0 | 25.48 | 57.23 | 1317.72 | 0.9985 |
| 456 | RR | 3.350 | 13.0 | 22.58 | 43.47 | 1213.85 | 1.0000 |
| 456 | Weighted | 0.000 | 0.0 | 23.91 | 62.29 | 1212.40 | 0.9994 |

**Surprise:** RR regret persists at low load (mean=2.67 vs expected ~0). The control does NOT confirm that regret vanishes at low load — RR structural regret is load-independent.

**Conservation (INV-1):** Holds for all 12 runs (500 completed, 0 queued, 0 running for all).

## Root Cause Analysis

### Finding 1: Weighted routing has structurally zero regret

Weighted regret is exactly 0.000000 across all seeds and rates. This is NOT an empirical result — it is a structural property of how regret is computed for weighted policies.

**Mechanism:** `computeCounterfactual()` (`sim/cluster/counterfactual.go:32-96`) receives the `scores` map from `RoutingDecision.Scores`. For `WeightedScoring`, these scores are the composite weighted scores computed by `WeightedScoring.Route()` (`sim/routing.go:151-184`). The router picks `argmax(scores)` (`sim/routing.go:176-184`). Regret = `all[0].score - chosenScore` (`sim/cluster/counterfactual.go:90`). Since `chosen == argmax`, `best_score == chosen_score`, so regret = 0 by construction. This holds for ANY weighted scorer configuration.

### Finding 2: RR regret reflects load imbalance from service time variance

RR distributes requests in cyclic order, ignoring instance load. The counterfactual uses load-based fallback scoring: `-(QueueDepth + BatchSize + PendingRequests)` (`sim/cluster/counterfactual.go:49-51`). Mixed workload creates variable service times (short: ~9.4ms, long: ~25.7ms), so instances have different QueueDepth/BatchSize at any given moment. When RR routes to a loaded instance while an emptier one exists, regret > 0.

**First-principles calculation (RCV-2):** The counterfactual score for each instance is `-(QueueDepth + BatchSize + PendingRequests)` (`sim/cluster/counterfactual.go:51`). Regret = `best_score - chosen_score` = difference in EffectiveLoad between the chosen instance and the least-loaded instance.

At rate=200 with 4 instances, mean inter-arrival time = 5ms. Long requests (30% of traffic) have step time ~25.7ms, so a long request occupies an instance for ~25.7ms/5ms = ~5 inter-arrival periods. During that time, ~1.5 more requests arrive (5ms * 200/s * 0.3 long-fraction = 0.3 long + 5ms * 200/s * 0.7 short = 0.7 short per inter-arrival). With 4 instances and batch processing, at any routing decision the EffectiveLoad spread between busiest and least-busy instance is typically 1-4 units (one instance processing a long request has BatchSize=1 while an idle instance has 0).

RR cycles through all 4 instances regardless of load. On 3 of 4 decisions, it does not pick the least-loaded instance, producing regret equal to the load gap. Expected mean regret = (3/4) * mean_load_gap. With a mean load gap of ~2-4 units, predicted mean regret = 1.5-3.0. Observed: 2.12-5.36 across seeds (mean 3.20). The upper end (seed 456, regret=5.36) reflects higher transient load variance from the specific arrival sequence. The prediction and observation are consistent in order of magnitude and range.

### Finding 3: RR regret persists at low load (ED-2 surprise)

**Predicted:** Regret should decrease toward zero at low load because all instances would have similar (near-zero) load.

**Observed:** RR mean regret at rate=100 (2.67) is comparable to rate=200 (3.20). Only 17% lower.

**Root cause:** Even at low utilization, the mixed workload creates transient load differences. A long request (input=1024, output=256) takes ~25.7ms — while it's running, that instance has BatchSize=1 while others may have BatchSize=0. The `EffectiveLoad()` (`sim/routing.go:23-25`) = QueueDepth + BatchSize + PendingRequests captures this transient difference. Since RR cycles through all 4 instances regardless, ~3/4 of requests encounter at least one instance with lower load than the one they're assigned to, producing regret.

**Why it doesn't vanish:** Regret measures *per-decision* suboptimality, not cumulative effect. Even when overall utilization is low, individual routing decisions still differ in quality because service time heterogeneity creates instantaneous load variance. This variance depends on the workload mix (short vs long), NOT on the offered load.

### Finding 4: RR has lower TTFT than Weighted despite higher regret

**Paradox:** RR has 5-15% lower TTFT than Weighted across all seeds and rates, despite having non-zero regret (suggesting suboptimal routing).

**Root cause:** The regret metric and actual TTFT measure different things:
1. **Regret** = counterfactual score gap at the instant of routing. Measures per-decision optimality.
2. **TTFT** = time from request arrival to first token. Depends on cumulative load distribution over time.

RR achieves perfect distribution uniformity (Jain FI = 1.0000) because 500 requests / 4 instances = exactly 125 per instance. This maximizes throughput and minimizes queue buildup.

Weighted routing creates slight imbalance (Jain FI ~0.995) through greedy decisions: when multiple requests arrive close together, they all route to the same "least loaded" instance, temporarily overloading it while other instances idle. This is the well-known "thundering herd" effect of greedy load balancing.

The TTFT difference (5-15%) is small because both policies achieve near-perfect distribution at this load level. At higher overload, the difference may reverse as RR's blindness to load becomes more costly.

## Devil's Advocate (RCV-5)

**If this is "Confirmed," argue why it might be Refuted:**
The hypothesis claims weighted has "significantly lower" regret. In fact, weighted has ZERO regret by structural design, not because of intelligent routing. The regret metric for weighted is degenerate — it can never be non-zero regardless of routing quality. This means the "higher RR regret" finding is trivially true and tells us nothing about whether RR is actually worse at routing. The TTFT data suggests RR is actually BETTER at this operating point. The hypothesis is technically "confirmed" but the confirmation is vacuous.

**If this is "Refuted," argue why it might be Confirmed:**
RR regret is consistently positive across all seeds and both rates, averaging 2-5 units. This demonstrates that RR provably makes suboptimal per-decision choices — the counterfactual analysis correctly identifies moments where RR picks a loaded instance over an empty one. The fact that this doesn't translate to worse aggregate TTFT at moderate load doesn't invalidate the per-decision analysis; it just means the metric captures a different aspect of routing quality.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Weighted regret is structurally zero | Design limitation | Documented here — regret is meaningful only for non-score-based policies |
| RR regret reflects instantaneous load imbalance | Confirmation | Documented here — counterfactual correctly quantifies per-decision suboptimality |
| RR regret persists at low load | Surprise | Documented here — regret is load-independent for mixed workloads |
| Higher regret does not imply worse TTFT | Surprise | Documented here — aggregate vs per-decision metrics diverge |
| Conservation (INV-1) holds across all configs | Confirmation | Confirms INV-1 under trace-enabled runs |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None — the structural-zero-regret property is specific to the counterfactual implementation, not a general antipattern
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? INV-1 (conservation) confirmed across 12 cluster-level runs with trace-level decisions enabled. INV-6 (determinism) confirmed — re-running with same seeds produces identical output.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, rate=100 and rate=200, 500 requests, mixed workload (70% short/30% long), always-admit, fcfs scheduler, 3 seeds
- **Parameters findings depend on:** Mixed workload with variable service times (finding 2, 3). Weighted scoring with any scorer (finding 1). Load-based fallback scoring for non-weighted policies (finding 2).
- **What was NOT tested:**
  - Higher rates (overload) where RR may show worse TTFT than weighted
  - Least-loaded policy (uses score=nil like RR, but picks least loaded — would have zero counterfactual regret by construction since it IS the fallback scorer)
  - Weighted with multiple scorers (prefix-affinity + queue-depth)
  - The regret metric for weighted with non-argmax routing (e.g., epsilon-greedy) — not implemented in BLIS
  - **Uniform-workload control (RCV-4 gap):** A control with all requests having identical input/output tokens (e.g., constant input=256, output=128) was not run. This control would isolate whether RR regret arises from service time variance (the proposed mechanism in Finding 2) or from Poisson arrival burstiness alone. With uniform service times, all instances should have near-identical EffectiveLoad at any moment, so RR regret should drop to near-zero. If it does not, the mechanism is arrival burstiness rather than service time heterogeneity. Follow-up experiment recommended.
- **Generalizability:** Finding 1 (structural zero regret for weighted) generalizes to ANY weighted scorer configuration. Finding 2 (RR regret from service time variance) generalizes to any mixed workload. Finding 4 (regret-TTFT divergence) may not generalize to high overload.
- **Uncertainty quantification:** UQ not formally performed. RR mean regret ranges 1.91-5.36 across seeds, suggesting moderate variance. Weighted regret is exactly 0 with zero variance (structural).

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| RR mean regret (rate=200) | 3.20 (range: 2.12-5.36) | High — consistent direction across 3 seeds |
| Weighted mean regret | 0.000000 (all seeds) | High — structural property, not empirical |
| RR TTFT advantage | 5-15% lower than Weighted | Medium — consistent direction but moderate effect, may reverse at higher load |
| RR Jain FI | 1.0000 (all seeds) | High — structural property of cyclic routing with divisible request count |
| Sample size | 3 seeds x 2 policies x 2 rates = 12 runs | Medium — 3 seeds is minimum; effect sizes are large enough |
| Conservation (INV-1) | Holds for all 12 runs | High — exact check |

## Implications for Users

1. **Counterfactual regret is most informative for non-score-based policies** (round-robin, least-loaded). For weighted policies, regret is always zero by construction and provides no signal.

2. **Non-zero RR regret does NOT mean RR is worse than weighted.** At moderate load, RR's perfect distribution uniformity can produce lower TTFT than weighted's greedy approach. Use regret as one signal among many, not as the sole routing quality indicator.

3. **Mixed workloads create inherent load variance** even at low utilization. If per-decision routing quality matters (e.g., for SLO compliance), consider least-loaded or weighted policies that account for instantaneous load.

4. **The trace/counterfactual pipeline works correctly:** it records decisions, computes counterfactual candidates, and produces meaningful regret values that match theoretical expectations.

## Reproducing

```
cd hypotheses/h6-counterfactual-regret
./run.sh
```
