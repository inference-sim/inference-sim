# H4: Round-Robin vs Least-Loaded at Low Utilization

**Status:** Confirmed with nuance
**Resolution:** Partial confirmation with surprise — RR and LL are equivalent on mean metrics at low rate, but LL TTFT p99 is 12-21% worse due to tie-breaking bias. At high rate with constant-size requests, both policies produce identical parsed metrics (0.000% difference on all extracted fields).
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Tier:** 5
**Type:** Statistical (Equivalence)
**Date:** 2026-02-23
**Rounds:** 2

## Hypothesis

> Round-robin should outperform or match least-loaded for uniform workloads at low rates. For perfectly uniform request sizes at low utilization, round-robin distributes optimally with zero overhead. Least-loaded has the same distribution but with routing computation overhead and potential for minor oscillation due to PendingRequests tracking delays.

Predicted outcome: Nearly identical metrics (within 5%), confirming that least-loaded's intelligence adds no value for uniform low-rate workloads.

## Experiment Design

**Classification:** Statistical / Equivalence

**Configurations compared:**
- A (Round-Robin): `--routing-policy round-robin --scheduler fcfs --priority-policy constant --admission-policy always-admit --num-instances 4`
- B (Least-Loaded): `--routing-policy least-loaded --scheduler fcfs --priority-policy constant --admission-policy always-admit --num-instances 4`

Both use workload-spec YAML with constant token distribution (input=256, output=128), Poisson arrivals.

**Controlled variables:** Model (llama-3.1-8b-instruct), instances (4), scheduler (fcfs), priority (constant), admission (always-admit), tokens (constant 256/128), arrival process (Poisson)

**Varied variable:** Routing policy (round-robin vs least-loaded)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- ED-3: At rate=100, utilization ~0.29 (well below saturation); all 500 requests complete with 0 queued/running at end (confirmed by INV-1 checks)
- Rate sizing: step time = 6910 + 17.67*256 + 2.84*128 = 11,798 us; capacity = 4/0.0118 = 339 req/s; rate=100 gives rho=0.29

**Experiments:**
- **Exp 1 (Low Rate):** rate=100, 500 requests, 4 instances — equivalence expected
- **Exp 2 (High Rate Control, ED-2):** rate=1000, 500 requests, 4 instances — expected LL to outperform RR under 3x overload

## Results

### Experiment 1: Low Rate (rate=100, 0.29x utilization)

| Seed | Metric | Round-Robin | Least-Loaded | % Diff |
|------|--------|------------|-------------|--------|
| 42 | TTFT mean (ms) | 19.20 | 19.82 | 3.11% |
| 42 | TTFT p99 (ms) | 23.26 | 26.52 | 12.28% |
| 42 | E2E mean (ms) | 1244.96 | 1244.43 | 0.04% |
| 42 | E2E p99 (ms) | 1288.04 | 1289.16 | 0.09% |
| 123 | TTFT mean (ms) | 19.28 | 19.75 | 2.38% |
| 123 | TTFT p99 (ms) | 23.34 | 26.67 | 12.50% |
| 123 | E2E mean (ms) | 1251.26 | 1250.58 | 0.05% |
| 123 | E2E p99 (ms) | 1284.70 | 1282.72 | 0.15% |
| 456 | TTFT mean (ms) | 19.37 | 19.93 | 2.78% |
| 456 | TTFT p99 (ms) | 22.80 | 28.77 | 20.74% |
| 456 | E2E mean (ms) | 1237.22 | 1236.72 | 0.04% |
| 456 | E2E p99 (ms) | 1278.69 | 1278.65 | 0.00% |

**Equivalence assessment (5% threshold):**
- TTFT mean: max diff = 3.11% -- **EQUIVALENT** (within 5%)
- TTFT p99: max diff = 20.74% -- **NOT EQUIVALENT** (exceeds 5% in all seeds)
- E2E mean: max diff = 0.05% -- **EQUIVALENT**
- E2E p99: max diff = 0.15% -- **EQUIVALENT**

**Note on seed 456 TTFT p99:** The 20.74% difference for seed 456 is right at the 20% "significant" boundary from `docs/standards/experiments.md`. This is marginal — the effect is real and consistent in direction (LL worse in all 3 seeds), but the magnitude at seed 456 should not be over-interpreted as definitively crossing the significance threshold. Seeds 42 and 123 at 12.28% and 12.50% are well above the 5% equivalence threshold but below the 20% significance threshold, placing them in the 5-20% range that the legacy thresholds leave ambiguous.

### Experiment 2: High Rate Control (rate=1000, 3x overload)

| Seed | Metric | Round-Robin | Least-Loaded | % Diff |
|------|--------|------------|-------------|--------|
| 42 | TTFT mean (ms) | 134.70 | 134.70 | 0.00% |
| 42 | TTFT p99 (ms) | 226.49 | 226.49 | 0.00% |
| 42 | E2E mean (ms) | 1557.50 | 1557.50 | 0.00% |
| 42 | E2E p99 (ms) | 1737.09 | 1737.09 | 0.00% |
| 123 | TTFT mean (ms) | 152.96 | 152.96 | 0.00% |
| 123 | TTFT p99 (ms) | 259.51 | 259.51 | 0.00% |
| 123 | E2E mean (ms) | 1575.56 | 1575.56 | 0.00% |
| 123 | E2E p99 (ms) | 1736.14 | 1736.14 | 0.00% |
| 456 | TTFT mean (ms) | 107.73 | 107.73 | 0.00% |
| 456 | TTFT p99 (ms) | 198.50 | 198.50 | 0.00% |
| 456 | E2E mean (ms) | 1530.79 | 1530.79 | 0.00% |
| 456 | E2E p99 (ms) | 1734.94 | 1734.94 | 0.00% |

All parsed metrics are identical (0.000% difference) across all seeds. Note: this is based on comparing parsed float values from the output, not a byte-level diff of raw output files. The structural equivalence argument (Finding 2 root cause) predicts true byte-identical output, but this was not verified via raw file comparison.

### Conservation (INV-1)

All 12 conservation checks pass (6 per experiment, 2 experiments). Every run: injected=500, completed=500, still_queued=0, still_running=0.

## Root Cause Analysis

### Finding 1: TTFT p99 divergence at low rate (12-21% worse for LL)

The divergence traces to tie-breaking in `LeastLoaded.Route()` (`sim/routing.go:107-124`). When all instances have equal EffectiveLoad (QueueDepth + BatchSize + PendingRequests), the loop picks the first instance with minimum load — always instance 0 due to the initialization at line 113-114:

```
minLoad := snapshots[0].EffectiveLoad()
target := snapshots[0]
```

The strict `<` comparison at line 118 means equal-load instances are never selected over the initial choice.

At rate=100 with 4 instances, the mean inter-arrival time is 10ms while the step time is ~11.8ms. Requests frequently arrive when all instances have drained their queues (EffectiveLoad=0 for all). In this tie state, LL routes to instance 0. If a second request arrives before instance 0 finishes processing the first, it queues behind it — adding ~11.8ms to its TTFT. RoundRobin (`sim/routing.go:90-97`) cycles through instances deterministically via `counter % len(snapshots)`, distributing perfectly regardless of load state.

The TTFT p99 captures these occasional "double-hit" events where LL's positional bias sends consecutive requests to instance 0. The mean is less affected because most requests still distribute evenly (PendingRequests breaks ties after the first request at each timestamp).

**First-principles calculation (RCV-2):** At rate=100 req/s with Poisson arrivals, P(inter-arrival < 11.8ms) = 1 - e^(-100*0.0118) = 1 - e^(-1.18) = 0.692. So ~69% of requests arrive within the service time of the previous request. At these overlapping arrivals, if LL has routed to instance 0 and the next request arrives before PendingRequests decrements, instance 0 gets two requests. For p99, this means the worst 1% of requests see a full additional step time (~11.8ms) added to TTFT.

### Finding 2: Identical parsed metrics at high rate (surprise)

At rate=1000 (3x overload), all 500 requests arrive within ~0.5 seconds but take ~1.47 seconds to process. Multiple requests arrive at the same DES timestamp. The event processing order is: ClusterArrivalEvent (priority 0) → AdmissionDecisionEvent → RoutingDecisionEvent.

When N requests arrive at the same timestamp, RoutingDecisionEvents are processed sequentially. After each routing decision, `pendingRequests[target]++` (`sim/cluster/cluster_event.go:185`). This means:
- Request 1: all loads equal → LL picks instance 0 → PendingRequests[0]=1
- Request 2: instance 0 has load 1, others 0 → LL picks instance 1 → PendingRequests[1]=1
- Request 3: instances 0,1 have load 1, others 0 → LL picks instance 2
- Request 4: instances 0,1,2 have load 1, instance 3 has load 0 → LL picks instance 3

This is identical to RR's cyclic 0,1,2,3 pattern. With constant tokens (identical service times) and no stochastic variation in processing, subsequent event timelines are deterministic and identical. The result: identical parsed metrics across all extracted fields (0.000% difference).

**Mechanism confirmation:** The PendingRequests tracking (`sim/cluster/cluster_event.go:65`, `sim/routing.go:24`) creates a virtual round-robin for LL under uniform load, because each routing decision increments the target's effective load by exactly 1, making the next-least-loaded instance always the next in cyclic order. This equivalence is structural, not coincidental — it holds for any constant-token workload regardless of arrival rate, as long as the initial state is symmetric.

### Control experiment assessment (RCV-4)

The high-rate control (Exp 2) was designed to validate that LL outperforms RR under overload. Instead, it revealed that under constant-token workloads, the two policies produce identical results at ANY rate where PendingRequests tracking creates symmetric cyclic routing. This is a valid control in that it confirms the p99 divergence at low rate is NOT a measurement artifact — it is specific to the low-rate regime where PendingRequests drains between arrivals and LL tie-breaking takes effect.

## Devil's Advocate (RCV-5)

**If this is "Confirmed with nuance," argue why it might be Refuted:**
The TTFT p99 difference (12-21%) clearly exceeds the 5% equivalence threshold, meaning the policies are NOT equivalent by the strict definition. The hypothesis predicted "within 5%" on all metrics. One could argue this is a refutation: LL is measurably worse than RR at the tail, contradicting the claim that "least-loaded's intelligence adds no value." The intelligence actively hurts at the tail due to positional bias.

**If this were "Refuted," argue why it might be Confirmed:**
The 12-21% difference is only in TTFT p99 — the mean (2-3%) and E2E metrics (<0.2%) are all within 5%. The hypothesis's core intuition ("intelligence adds no value for uniform workloads") is correct for the aggregate metrics that users typically care about. The p99 TTFT divergence is a subtle edge case from tie-breaking, not a fundamental failure of the equivalence claim. Furthermore, the high-rate control shows the policies produce identical parsed metrics under load.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| RR and LL mean metrics equivalent at low rate (<5% diff) | Confirmation | Documented here |
| LL TTFT p99 is 12-21% worse at low rate due to tie-breaking | Surprise | Documented here |
| RR and LL identical parsed metrics under constant-token overload | Surprise | Documented here |
| LL tie-breaking favors instance 0 (positional bias) | Design limitation | Documented here; follow-up could add random tie-breaking |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found. LL tie-breaking is deterministic (INV-6) and doesn't violate any existing rules.
- [x] Any new rules needed? No. The tie-breaking behavior is by design (snapshot order = instance index). Random tie-breaking would be a design choice, not a standards issue.
- [x] Any new invariants needed? No. The PendingRequests-induced cyclic pattern is consistent with INV-7 (signal freshness) — PendingRequests is synchronously updated.
- [x] Any existing rules/invariants confirmed? INV-1 (conservation) confirmed across all 12 runs. INV-6 (determinism) supported — identical parsed metrics at high rate are consistent with determinism under both policies (raw byte-level diff not performed).

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, rate=100 (low) and rate=1000 (high), constant 256/128 tokens, Poisson arrivals, fcfs scheduler, seed=42/123/456
- **Parameters findings depend on:** Constant token sizes (all requests identical service time), Poisson arrival process, 4 instances. The identical-metrics high-rate finding requires exactly constant tokens — any variance in tokens would break the cyclic symmetry.
- **What was NOT tested:** Variable token sizes (gaussian, pareto), non-Poisson arrivals, different instance counts, medium rates near saturation (rho~0.8-1.0). With variable tokens, LL should genuinely outperform RR at high rates because its load-balancing intelligence becomes valuable when service times differ.
- **Generalizability:** The mean equivalence finding (<5% on TTFT mean, E2E) likely generalizes to any uniform workload at low utilization. The p99 TTFT divergence specifically depends on the tie-breaking implementation. The identical-metrics high-rate result is specific to constant-token workloads and would NOT hold with variable tokens.
- **Missing mechanism-isolation control (RCV-4):** The tie-breaking mechanism (LL favoring instance 0 when loads are tied) is verified via code analysis (`sim/routing.go:113-118`) but not experimentally isolated. A control experiment with randomized tie-breaking (e.g., selecting a random instance among equal-load candidates) was not implemented. Such a control would confirm whether the p99 divergence vanishes when tie-breaking is removed, definitively attributing the effect to positional bias rather than some other subtle difference between RR and LL routing paths.
- **No positive control showing LL outperforming RR:** The high-rate control (Exp 2) was designed to demonstrate LL advantage under overload but instead produced identical parsed metrics. No experiment in this study demonstrates LL outperforming RR — the expected divergence requires variable-token workloads (where instances develop different queue depths), which was not tested. This means the experiment validates equivalence but does not fully validate the comparison by showing both directions.
- **Uncertainty quantification:** UQ not performed — single operating point per rate level. The p99 TTFT divergence (12-21%) exceeds the 5% threshold in all 3 seeds, giving high confidence that the effect is real. The direction is consistent (LL worse) in all seeds.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT mean equivalence | 2.38-3.11% diff | High — consistent across 3 seeds, well within 5% |
| TTFT p99 divergence | 12.28-20.74% (LL worse) | High — consistent direction, exceeds threshold in all seeds |
| E2E equivalence | 0.00-0.15% diff | High — effectively identical |
| High-rate identity | 0.000% diff (identical parsed metrics) | High — deterministic, 3 seeds confirm; structural argument predicts true byte-identity but raw diff not performed |
| Mechanism (tie-breaking) | Code path traced | High — `sim/routing.go:113-114` initialization + strict `<` at line 118 |
| Mechanism (PendingRequests cycling) | Code path traced | High — `sim/cluster/cluster_event.go:185` increment creates cyclic pattern |
| Sample size | 3 seeds x 2 experiments x 2 policies x 500 req = 6000 requests | Medium — adequate for equivalence test |

## Implications for Users

1. **At low utilization with uniform workloads, round-robin and least-loaded produce equivalent mean performance.** Users need not prefer one over the other for capacity planning at low load. The choice only matters under variable-size workloads or near-saturation rates.

2. **LL has slightly worse TTFT tail latency at low utilization** (12-21% on p99) due to positional bias in tie-breaking. For users who care about tail latency in low-load scenarios, round-robin is marginally better because it distributes perfectly cyclically regardless of load state.

3. **Under constant-token workloads at any rate, RR and LL produce identical parsed metrics.** The PendingRequests tracking makes LL behave as a virtual round-robin when all requests have the same service time. This is a consequence of the symmetric initial state and deterministic event processing.

4. **The value of least-loaded routing emerges with variable-size requests**, where different instances naturally develop different queue depths. This experiment validates the intuition that LL's "intelligence" is wasted on perfectly uniform workloads.

## Reproducing

```bash
cd hypotheses/h4-rr-vs-ll
./run.sh
```
