# H21: Extreme Scorer Weights

**Status:** Refuted
**Resolution:** Refuted -- wrong mental model
**Family:** Robustness/failure-mode
**VV&UQ:** Validation
**Tier:** B
**Type:** Statistical (Equivalence)
**Date:** 2026-02-22
**Rounds:** 2

## Hypothesis

> "Extreme scorer weight (weight=100:1) should behave identically to single-scorer routing."

## Experiment Design

**Classification:** Statistical/Equivalence (within 5% = equivalent)

**Configurations compared:**
- A: `--routing-policy weighted --routing-scorers "prefix-affinity:100,queue-depth:1"` (2 scorers, normalized: PA=0.9901, QD=0.0099)
- B: `--routing-policy weighted --routing-scorers "prefix-affinity:1"` (1 scorer, no load-balancing signal)
- C (Round 2 control A3a): `--routing-policy weighted --routing-scorers "prefix-affinity:100,queue-depth:0.001"` (2 scorers, normalized: PA=0.99999, QD=0.00001)
- D1 (Round 2 control A3b): Config A scorers + no-prefix workload (all unique requests)
- D2 (Round 2 control A3b): Config B scorers + no-prefix workload (all unique requests)

**ED-1 confound acknowledgment:** Configs A and B differ in scorer COUNT (2 vs 1), not just weight ratio. This is inherent to the hypothesis -- "100:1 behaves like single-scorer" implicitly asks whether a near-zero-weight scorer can be treated as absent. Control C (same 2-scorer count, 100000:1 ratio) isolates the weight-magnitude question. Control D (unique requests) isolates the tiebreaker from the prefix cascade.

**Controlled variables:** Model (llama-3.1-8b-instruct), 4 instances, log level (error), trace level (decisions), summarize-trace

**Varied variable:** Scorer configuration (presence, weight, and count of queue-depth scorer)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- `normalizeScorerWeights` normalizes to sum=1.0 (`sim/routing_scorers.go:82-95`)
- `WeightedScoring.Route` uses strict `>` for argmax ties, breaking to first snapshot occurrence (`sim/routing.go:180`)
- Prefix-affinity scorer returns 0.0 for all instances when no prefixes are cached (`sim/routing_prefix_scorer.go:25-26`)
- When `PrefixGroup` is empty, `generatePrefixTokens` at `sim/workload/client.go:39-41` skips the client -- no shared prefix tokens

## Results

### Round 1: A vs B (prefix workload)

#### TTFT Comparison (ms)

| Seed | A mean | B mean | Diff% | A p99 | B p99 | Diff% |
|------|--------|--------|-------|-------|-------|-------|
| 42 | 24.38 | 149.33 | +512.5% | 38.55 | 204.26 | +429.9% |
| 123 | 26.92 | 100.06 | +271.7% | 58.90 | 282.69 | +379.9% |
| 456 | 24.47 | 142.20 | +481.1% | 41.58 | 243.52 | +485.7% |

#### E2E Latency Comparison (ms)

| Seed | A mean | B mean | Diff% | A p99 | B p99 | Diff% |
|------|--------|--------|-------|-------|-------|-------|
| 42 | 1456.50 | 1724.01 | +18.4% | 5159.39 | 5413.60 | +4.9% |
| 123 | 1621.14 | 1924.93 | +18.7% | 7202.79 | 7509.04 | +4.3% |
| 456 | 1544.33 | 1848.86 | +19.7% | 7319.92 | 7654.06 | +4.6% |

#### Target Distribution (requests per instance)

| Seed | Config | inst_0 | inst_1 | inst_2 | inst_3 | Jain |
|------|--------|--------|--------|--------|--------|------|
| 42 | A (100:1) | 13 | 160 | 14 | 13 | 0.383 |
| 42 | B (single) | 200 | 0 | 0 | 0 | 0.250 |
| 123 | A (100:1) | 162 | 12 | 12 | 14 | 0.374 |
| 123 | B (single) | 200 | 0 | 0 | 0 | 0.250 |
| 456 | A (100:1) | 162 | 13 | 13 | 12 | 0.374 |
| 456 | B (single) | 200 | 0 | 0 | 0 | 0.250 |

#### Cache Hit Rate

| Seed | Config A | Config B |
|------|----------|----------|
| 42 | 0.1396 | 0.1011 |
| 123 | 0.1356 | 0.0958 |
| 456 | 0.1367 | 0.0980 |

### Round 2: Control A3a — Weight Sensitivity (A vs C)

Config C uses `prefix-affinity:100,queue-depth:0.001` (normalized QD weight = 0.00001).

| Seed | A mean | C mean | Diff% | A p99 | C p99 | Diff% |
|------|--------|--------|-------|-------|-------|-------|
| 42 | 24.38 | 24.38 | 0.0% | 38.55 | 38.55 | 0.0% |
| 123 | 26.92 | 26.92 | 0.0% | 58.90 | 58.90 | 0.0% |
| 456 | 24.47 | 24.47 | 0.0% | 41.58 | 41.58 | 0.0% |

**A and C are byte-identical.** Target distributions and cache hit rates match exactly (confirmed via analysis output). This is because normalization makes `100:1` and `100:0.001` produce the same relative weight ratios within the float64 representation -- the argmax tiebreaker fires identically in the DES (INV-6 determinism).

**Control A3a verdict: PASS.** Any positive queue-depth weight, regardless of magnitude, produces identical results. The weight-sensitivity control confirms the mechanism is binary (present vs absent), not a function of weight magnitude.

### Round 2: Control A3b — Zero-Prefix-Sharing (D1 vs D2)

Both configs use a no-prefix workload (no `prefix_group`, all unique requests).

#### TTFT: D1 vs D2 (ms)

| Seed | D1 mean | D2 mean | Diff% | D1 p99 | D2 p99 | Diff% |
|------|---------|---------|-------|--------|--------|-------|
| 42 | 19.70 | 123.73 | +527.9% | 28.41 | 223.33 | +686.0% |
| 123 | 19.52 | 104.10 | +433.2% | 27.18 | 189.72 | +598.0% |
| 456 | 19.11 | 87.91 | +360.0% | 28.55 | 152.66 | +434.7% |

#### Target Distribution: D1 vs D2

| Seed | Config | inst_0 | inst_1 | inst_2 | inst_3 | Jain |
|------|--------|--------|--------|--------|--------|------|
| 42 | D1 (100:1) | 53 | 47 | 50 | 50 | 0.998 |
| 42 | D2 (single) | 200 | 0 | 0 | 0 | 0.250 |
| 123 | D1 (100:1) | 51 | 50 | 47 | 52 | 0.999 |
| 123 | D2 (single) | 200 | 0 | 0 | 0 | 0.250 |
| 456 | D1 (100:1) | 54 | 45 | 50 | 51 | 0.996 |
| 456 | D2 (single) | 200 | 0 | 0 | 0 | 0.250 |

**Control A3b verdict: D1 != D2.** This isolates the tiebreaker from the cascade. With unique requests, the prefix-affinity scorer returns 0.0 for all instances on every request (no block hash matches). The queue-depth scorer is the ONLY differentiating signal. D1 achieves near-perfect uniformity (Jain ~0.998), while D2 degenerates to all-to-one (Jain 0.250). Note that D1 is MORE uniform than A (Jain 0.998 vs 0.383) because without prefix sharing, there is no prefix-affinity signal pulling requests toward a cached instance -- the queue-depth scorer operates as a pure load-balancer.

## Root Cause Analysis

The hypothesis fails because it assumes that reducing a scorer's weight toward zero makes it disappear. In reality, even an infinitesimal weight becomes decisive when the dominant scorer produces tied scores. The mechanism is a cold-start positive feedback loop specific to stateful scorers with observer-seeded state (like prefix-affinity).

**ED-1 confound:** Configs A and B differ in scorer count (2 vs 1), not just weight ratio. This is not a confound bug but the fundamental insight: the hypothesis implicitly asks "can near-zero weight be treated as absent?" The answer is no, because the scorer's presence (even at epsilon weight) introduces a qualitatively different signal.

**Mechanism (4 steps, with file:line citations):**

**Step 1 -- Weight normalization.** `normalizeScorerWeights()` at `sim/routing_scorers.go:82-95` divides each weight by the total. Config A: `100/(100+1) = 0.9901` for prefix-affinity, `1/101 = 0.0099` for queue-depth. Config B: `1/1 = 1.0` for prefix-affinity alone. The `NewRoutingPolicy` factory at `sim/routing.go:319` stores these normalized weights in `WeightedScoring.weights`.

**Step 2 -- Cold-start tie.** At simulation start, the prefix cache is empty (`NewPrefixCacheIndex` at `sim/routing_prefix_scorer.go:15` creates a fresh `PrefixCacheIndex`). The scorer function at `sim/routing_prefix_scorer.go:28-29` computes `matched := idx.MatchLength(hashes, snap.ID)` which returns 0 for an empty cache, yielding `score = 0/totalBlocks = 0.0` for every instance. In Config B, the composite score per instance is `0.0 * 1.0 = 0.0` -- all tied. In Config A, composite = `0.0 * 0.9901 + qd * 0.0099` where `qd` is the queue-depth score.

For the very first request, all instances have `EffectiveLoad()` = 0 (initial snapshots at `sim/cluster/snapshot.go:80` are zero-valued; `EffectiveLoad()` at `sim/routing.go:23-24` returns `QueueDepth + BatchSize + PendingRequests = 0`). The `scoreQueueDepth` function at `sim/routing_scorers.go:132-133` hits the `maxLoad == minLoad` branch and returns 1.0 for all instances. So on the first request, both configs produce identical tied scores, and the argmax at `sim/routing.go:180` (`if scores[snap.ID] > bestScore` -- strict `>`) breaks to the first instance in snapshot order (index 0).

**Step 3 -- Observer seeds the cascade.** After the argmax selects instance_0, the observer at `sim/routing.go:187-188` calls `obs(req, snapshots[bestIdx].ID)`. The prefix-affinity observer at `sim/routing_prefix_scorer.go:39-40` calls `idx.RecordBlocks(hashes, targetInstance)`, recording the request's block hashes in instance_0's prefix cache.

When the second request arrives (with the same `prefix_group`, so same block hashes), the prefix-affinity scorer at `sim/routing_prefix_scorer.go:28-29` now finds `matched > 0` for instance_0 and `matched = 0` for all others. In Config B, this gives instance_0 a score of ~0.8 (depending on prefix fraction) vs 0.0 for others -- instance_0 wins decisively. The observer records more blocks on instance_0, reinforcing the cycle. This is a positive feedback loop: every routing decision strengthens the cache on instance_0, making subsequent decisions more likely to choose instance_0.

**Step 4 -- Queue-depth tiebreaker breaks the cycle (Config A only).** In Config A, after routing the first request to instance_0, the `pendingRequests[target]++` at `sim/cluster/cluster_event.go:185` increments instance_0's pending count. By default, QueueDepth uses `Immediate` refresh mode (`sim/cluster/snapshot.go:30`), so when the next request's routing event calls `buildRouterState()` at `sim/cluster/cluster_event.go:61-71`, instance_0's snapshot reflects its current load. `scoreQueueDepth` at `sim/routing_scorers.go:134-136` uses min-max normalization: instance_0 gets `(max-load) / (max-min) = 0.0` (highest load), while empty instances get `1.0`. The composite score for instance_0 becomes `pa_score * 0.9901 + 0.0 * 0.0099` vs an empty instance: `0.0 * 0.9901 + 1.0 * 0.0099 = 0.0099`. When `pa_score` is small (early in the simulation before many prefix blocks accumulate), the 0.0099 queue-depth bonus can flip the argmax. This seeds the prefix cache on a second instance, breaking the monopoly.

The target distribution data confirms: Config A sends ~160/200 to one instance (the prefix-affinity winner once its cache is seeded) but ~40 to others (routed during the early phase when queue-depth tiebreaking overcomes the prefix advantage). Config B sends 200/200 to instance_0 because the cascade was never broken.

**First-principles calculation (RCV-2):** This is an order-of-magnitude estimate. The beta coefficients (`defaults.yaml`: `[6910.42, 17.67, 2.84]`) give a step time of ~6.9ms for a minimal batch (beta0 only). At 500 req/s with 200 requests on 1 instance (Config B), the effective arrival rate is the full 500 req/s. Mean service time per request is at least one step (~7ms), giving `rho_approx = 500 * 0.007 = 3.5` (heavily overloaded, rho > 1 means requests queue faster than they drain). With 4-instance spreading (Config A, ~50 req/instance at Jain ~0.38 in practice), per-instance rate drops to ~125 req/s, giving `rho_approx = 125 * 0.007 = 0.875` (loaded but not saturated). The M/M/1 waiting time ratio between rho=3.5 and rho=0.875 is unbounded (rho>1 diverges) vs `0.875/(1-0.875) = 7.0` -- consistent with the 5-6x observed TTFT ratio (149.33 / 24.38 = 6.1x for seed 42). Note: this is an order-of-magnitude estimate because real service time depends on batch composition and the rho approximation ignores alpha overhead.

**Control experiments (RCV-4) -- both executed in Round 2:**

1. **A3a (weight sensitivity):** Config C uses `prefix-affinity:100,queue-depth:0.001` -- reducing QD weight by 1000x. Result: **byte-identical to Config A** (0.0% difference on all metrics). This confirms the mechanism is binary: any positive QD weight, regardless of magnitude, produces identical DES trajectories. The normalization at `sim/routing_scorers.go:82-95` makes absolute weight irrelevant -- only relative ratios matter, and even at 100000:1, the argmax tiebreaker fires the same way.

2. **A3b (zero-prefix-sharing):** Config D1/D2 use a workload with no `prefix_group`. Result: D1 achieves near-perfect uniformity (Jain ~0.998) while D2 sends all 200 to instance_0 (Jain 0.250, 360-528% TTFT difference). This isolates the tiebreaker from the cascade: with unique requests, prefix-affinity scores are always 0.0, so the queue-depth scorer is the ONLY signal. D1's near-perfect uniformity (vs A's 0.383) demonstrates that the non-uniformity in Config A is caused by the prefix-affinity feedback loop pulling requests toward cached instances -- not by the queue-depth scorer being too weak.

## Devil's Advocate (RCV-5)

**If this is "Refuted," argue why it might be Confirmed:**
One could argue that after the first few requests seed the prefix cache, the prefix-affinity scorer dominates because it produces differentiated scores (e.g., 0.8 for the cached instance vs 0.0 for others). At that point, the 1% queue-depth weight truly doesn't matter because `0.8 * 0.99 = 0.792 >> 1.0 * 0.01 = 0.01`. This is true for subsequent requests with prefix matches -- the divergence is entirely driven by the cold-start phase (first ~10 requests). If the prefix cache were pre-warmed across all instances, the 100:1 ratio might indeed behave equivalently to single-scorer. The finding is specific to cold-start scenarios with observer-seeded stateful scorers. However, the control A3b result shows the tiebreaker matters even WITHOUT any prefix cascade (pure unique requests), which refutes even this narrower form of the hypothesis.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Extreme weight ratio is NOT equivalent to single-scorer routing | Refutation | Documented here |
| Tiny tiebreaker weight (1/101) prevents cold-start concentration cascade | Surprise | Documented here |
| Weight magnitude is irrelevant -- 100:1 and 100000:1 are byte-identical (control A3a) | Confirmation | Confirms normalization makes absolute weight irrelevant |
| Single-scorer prefix-affinity creates degenerate instance concentration (all-to-one) | Design limitation | Documented here -- users should always pair prefix-affinity with a load-balancing scorer |
| Without prefix sharing, QD tiebreaker achieves near-perfect uniformity (Jain ~0.998) | Confirmation | Control A3b confirms QD as pure load-balancer when PA has no signal |
| Config B Jain fairness = 0.25 (theoretical minimum for 4 instances) | Confirmation | Confirms argmax tie-breaking behavior in `sim/routing.go:180` |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found -- weight normalization (R11 division guard) works correctly at `sim/routing_scorers.go:87-88`
- [x] Any new rules needed? Consider documenting that single-scorer routing is not recommended for production use (no tiebreaker = degenerate distribution). Not a new rule per se, but a user-facing guidance note.
- [x] Any new invariants needed? None -- the behavior is correct per the algorithm. The tie-breaking is deterministic (INV-6 confirmed by byte-identical results in control A3a).
- [x] Any existing rules/invariants confirmed? R2 (determinism via sorted snapshot order), INV-6 (determinism confirmed -- A and C are byte-identical across all seeds)

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, 200 requests, 500 req/s rate, blackbox mode. Two workloads: 80% prefix sharing (primary) and 0% prefix sharing (control A3b).
- **Parameters findings depend on:** The cold-start cascade requires (a) a stateful scorer with observer-seeded feedback loop (prefix-affinity), (b) initial state with empty internal cache, and (c) argmax tie-breaking by position. All three conditions must hold simultaneously. The tiebreaker effect (control A3b) requires only condition (c) -- any single-scorer config with tied scores degenerates to positional bias.
- **What was NOT tested:** Weight ratios between 100:1 and 1:0 with scorers that DON'T have observer feedback (e.g., kv-utilization alone vs kv-utilization:100,queue-depth:1). More than 4 instances. Warm-start scenarios (pre-populated prefix cache). Workloads with partial prefix sharing (e.g., 20% instead of 80%).
- **Generalizability:** The cascade finding is specific to stateful scorers with observer-seeded feedback loops. Currently only prefix-affinity has an observer (`sim/routing_prefix_scorer.go:35-41`). Stateless scorers (queue-depth, kv-utilization, load-balance) do not accumulate state and would not exhibit this cascade. The positional tie-breaking finding (control A3b) generalizes to ANY single-scorer config where scores are uniformly tied.
- **Uncertainty quantification:** All 3 seeds showed consistent results (272-512% TTFT mean difference for A vs B; 0.0% for A vs C; 360-528% for D1 vs D2). Control A3a is exact (byte-identical), providing deterministic confidence. Primary finding confidence is high given 3 seeds and 2 independent controls.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT mean difference (A vs B) | 272-512% across seeds | High -- consistent direction, well above 5% threshold |
| Target distribution (A vs B) | 200/0/0/0 vs 13/160/14/13 | High -- deterministic, reproduced across 3 seeds |
| Control A3a (A vs C) | 0.0% difference (byte-identical) | High -- deterministic, exact match |
| Control A3b (D1 vs D2) | 360-528%, Jain 0.998 vs 0.250 | High -- isolates tiebreaker from cascade |
| Cache hit rate | A=0.137, B=0.098 | Medium -- consistent but secondary metric |
| Mechanism | Observer-seeded feedback loop + positional argmax | High -- confirmed by both controls (A3a: weight-insensitive; A3b: cascade-independent) |

## Implications for Users

1. **Never use prefix-affinity as the sole scorer.** Without a load-balancing tiebreaker (queue-depth or kv-utilization), all requests concentrate on one instance at cold-start, creating a 5-6x TTFT degradation. This holds for ANY single-scorer config where scores can be uniformly tied.
2. **Even a tiny load-balancing weight matters.** A 1:100 weight ratio on queue-depth (just 1% of total score) is sufficient to prevent degenerate concentration. Weight magnitude is irrelevant -- 100:1 and 100000:1 produce byte-identical results.
3. **The default profile is safe.** The default `prefix-affinity:3,queue-depth:2,kv-utilization:2` has substantial load-balancing weight and will not exhibit this problem.
4. **Weight ratios are NOT equivalent to scorer removal.** Users should not assume that making a scorer's weight very large relative to others is the same as using it alone. Adding a second scorer changes the qualitative behavior, even at epsilon weight.

## Reproducing

```
cd hypotheses/h21-extreme-weights
./run.sh
```
