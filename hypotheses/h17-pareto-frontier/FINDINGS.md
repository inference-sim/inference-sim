# H17: Multi-Scorer Pareto Frontier

**Status:** Reclassified to Statistical/Dominance
**Resolution:** No within-workload Pareto frontier found on any workload (prefix-heavy, independent, or mixed). Cache-heavy dominates when prefix overlap exists; configs are equivalent when it does not. The "Pareto frontier" exists only across workloads (cross-workload dominance).
**Family:** Cross-policy comparative
**VV&UQ:** Validation (statistical — dominance)
**Tier:** 2
**Type:** Statistical (Dominance) -- reclassified from Statistical (Pareto) in Round 2 per Reviewer B
**Date:** 2026-02-22
**Rounds:** 2

## Hypothesis

> Multi-scorer weights should produce a Pareto frontier: no single configuration dominates all metrics. Different weight combinations optimize for different objectives — cache-heavy weights maximize locality (good TTFT), load-balance weights maximize fairness (good tail latency). No single weight combination should be best on ALL metrics simultaneously.

## Experiment Design

**Classification:** Statistical/Pareto

**Configurations compared:**
- C1 (cache-heavy): `--routing-policy weighted --routing-scorers "prefix-affinity:5,queue-depth:1"`
- C2 (llmd-default): `--routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"`
- C3 (load-balance): `--routing-policy weighted --routing-scorers "queue-depth:3,kv-utilization:3"`
- C4 (queue-heavy): `--routing-policy weighted --routing-scorers "prefix-affinity:1,queue-depth:5"`
- C5 (kv-heavy): `--routing-policy weighted --routing-scorers "kv-utilization:5,prefix-affinity:1"`

**Workloads compared (ED-2):**
- Workload A (prefix-heavy): `examples/multiturn-chat-demo.yaml` — multi-turn chat with 5 rounds, context accumulation, rate=500, 500 requests
- Workload B (independent): Poisson, gaussian input (mean=128), gaussian output (mean=64), rate=500, 500 requests, no multi-turn, no prefix groups
- Workload C (mixed, Round 2): 50% prefix-heavy (multi-turn chat, 5 rounds, context accumulate) + 50% independent (no multi-turn, no prefix), rate=500, 500 requests -- added per Reviewer B to test within-workload Pareto frontier

**Controlled variables:** model (llama-3.1-8b-instruct), instances (4), admission (always-admit), scheduler (fcfs), KV blocks (default), seeds
**Varied variables:** routing-scorers weights, workload type
**Seeds:** 42, 123, 456
**Preconditions verified:** All configs produce completed_requests == 500 (no admission rejection, no starvation)

## Results

### Workload A: Prefix-heavy (multi-turn chat)

| Configuration | TTFT Mean | TTFT P99 | E2E Mean | E2E P99 | Throughput | Cache Hit |
|---|---|---|---|---|---|---|
| cache-heavy | **21.2** | **31.2** | **702.2** | **1192.4** | **260.4** | 39.7% |
| kv-heavy | 27.1 | 44.1 | 706.1 | 1199.6 | 259.3 | 39.7% |
| llmd-default | 26.5 | 59.8 | 750.2 | 1270.1 | 257.4 | 29.4% |
| queue-heavy | 27.8 | 69.6 | 766.5 | 1306.3 | 255.8 | 26.2% |
| load-balance | 38.8 | 115.0 | 853.4 | 1437.9 | 250.1 | 10.7% |

**Pareto set:** {cache-heavy} — **dominates all other configurations on every metric**
**Ranking consistent across all 3 seeds:** Yes

### Workload B: Independent requests (no prefix reuse)

| Configuration | TTFT Mean | TTFT P99 | E2E Mean | E2E P99 | Throughput | Cache Hit |
|---|---|---|---|---|---|---|
| llmd-default | **20.1** | **30.6** | **697.1** | 1150.1 | **268.5** | 0.0% |
| load-balance | **20.1** | **30.6** | **697.1** | 1150.1 | **268.5** | 0.0% |
| cache-heavy | 20.3 | 31.2 | 697.3 | **1147.9** | 268.0 | 0.0% |
| queue-heavy | 20.3 | 31.2 | 697.3 | **1147.9** | 268.0 | 0.0% |
| kv-heavy | 26.8 | 49.1 | 700.8 | 1150.0 | 267.0 | 0.0% |

**Pareto set:** {llmd-default, load-balance, cache-heavy, queue-heavy} — 4 of 5 non-dominated
**Only kv-heavy is dominated** (33% worse TTFT than best, consistently across all seeds)

### Workload C: Mixed (50% prefix-heavy + 50% independent) -- Round 2

| Configuration | TTFT Mean | TTFT P99 | E2E Mean | E2E P99 | Throughput | Cache Hit |
|---|---|---|---|---|---|---|
| cache-heavy | **22.5** | **38.9** | **720.3** | 1183.3 | **310.9** | 13.5% |
| llmd-default | 24.4 | 51.6 | 739.5 | 1206.7 | 308.0 | 8.6% |
| queue-heavy | 24.7 | 52.3 | 741.1 | 1206.1 | 309.3 | 8.3% |
| kv-heavy | 28.6 | 57.0 | 723.8 | **1183.0** | 309.0 | 13.5% |
| load-balance | 26.9 | 59.1 | 761.3 | 1237.8 | 307.5 | 3.5% |

**Formal Pareto set:** {cache-heavy, kv-heavy} — 2 non-dominated configs
**However:** kv-heavy survives Pareto analysis only because its averaged E2E P99 (1183.0) is 0.3ms better than cache-heavy (1183.3) — a **0.03% margin**. Per-seed breakdown:
- Seed 42: cache-heavy wins (1189.4 vs 1192.8)
- Seed 123: cache-heavy wins (1171.9 vs 1174.0)
- Seed 456: kv-heavy wins (1182.1 vs 1188.6)

This is averaging noise, not a real tradeoff. **cache-heavy effectively dominates on the mixed workload** too: it wins TTFT by 27%, E2E mean by 0.5%, throughput by 0.6%, while the only metric kv-heavy "wins" is E2E P99 by 0.03%.

**Effect size compared to prefix-heavy:** The cache-heavy advantage is attenuated (TTFT spread: 21% on mixed vs 45% on prefix-heavy), consistent with 50% of traffic being independent.

**Ranking consistent across all 3 seeds:** cache-heavy is always best on TTFT mean. Best on E2E mean across all seeds.

### Cross-workload comparison

| Workload | Effective dominant | Within-workload frontier? |
|---|---|---|
| prefix-heavy | cache-heavy (strict dominance) | No |
| independent | llmd-default/load-balance (trivially, noise-level margins) | No (4 of 5 equivalent within <1%) |
| mixed | cache-heavy (effective dominance, kv-heavy non-dominated only by 0.03% noise) | No |

The Pareto set **changes across workloads**, but no workload produces a genuine within-workload Pareto frontier with meaningful tradeoffs.

## Round 2: Mixed Workload (Reviewer B feedback)

**Reviewer B concern:** "Add a mixed-workload (50% prefix-heavy + 50% independent) to test whether a genuine within-workload Pareto frontier exists. If mixed workload also fails to produce a within-workload frontier, consider reclassifying as Statistical/Dominance rather than Statistical/Pareto."

**Experiment:** Workload C combines both traffic patterns: one client generates multi-turn chat (5 rounds, context accumulation, rate_fraction=0.5) and another generates independent requests (no multi-turn, no prefix, rate_fraction=0.5), both at aggregate_rate=500, num_requests=500.

**Result:** No genuine within-workload Pareto frontier. cache-heavy dominates all metrics by meaningful margins (TTFT: 27%, E2E: 0.5-5.4%, throughput: 0.6%). The only non-dominated survivor (kv-heavy) survives by a 0.03% E2E P99 margin -- well below noise (seed variability for E2E P99 is 1.5% for cache-heavy). This pattern matches the prefix-heavy workload result (cache-heavy dominates) with attenuated effect sizes, as expected from 50% prefix overlap.

**Why the mixed workload does not produce a frontier:** The prefix-heavy half of the traffic still creates enough prefix overlap for cache-heavy routing to win on cache locality. The independent half dilutes the advantage (13.5% cache hit rate vs 39.7% on pure prefix-heavy), but does not create a counterbalancing load-balance advantage strong enough to produce a genuine tradeoff. At this utilization level (well below saturation), load imbalance from cache-affinity routing does not cause enough queueing delay to offset the prefill savings.

**Reclassification:** Per Reviewer B's suggestion, reclassifying from Statistical/Pareto to **Statistical/Dominance**. The experiment reveals a dominance pattern (cache-heavy dominates when any prefix overlap exists; configs are equivalent when it does not), not a Pareto frontier. The composable scorer framework's value is in cross-workload configuration selection, not within-workload tradeoffs.

## Root Cause Analysis

### Why cache-heavy dominates on prefix-heavy workloads

The multi-turn chat workload (`examples/multiturn-chat-demo.yaml`) uses `context_growth: accumulate` with 5 rounds. Each round prepends all prior context, creating strong prefix overlap between rounds of the same session.

1. **Prefix-affinity scorer routes session rounds to the same instance** (`sim/routing_prefix_scorer.go:17-33`): `MatchLength()` at line 28 returns high scores for instances that processed prior rounds of the same session, because the `PrefixCacheIndex` (LRU cache of block hashes) retains hashes from prior routing decisions via the observer (`sim/routing_prefix_scorer.go:35-41`).

2. **Cache hits reduce prefill tokens** (`sim/simulator.go:426-427`): When a request arrives at an instance that already has its prefix cached, `GetCachedBlocks(next.InputTokens)` finds cached blocks. `numNewTokens` is computed as `Len64(next.InputTokens) - Len64(cachedBlocks)*BlockSize()`, so cache hits directly reduce `numNewTokens`. This feeds into `StepTime` at `sim/latency_model.go:40,48` as `totalCacheMissTokens`, where `beta1=17.67` per cache-miss token is 6.2x more expensive than `beta2=2.84` per decode token.

3. **This improves ALL metrics simultaneously**: Lower prefill time reduces TTFT directly. Lower total step time reduces E2E. Faster completion frees capacity, increasing throughput. There is no tradeoff — cache locality is a pure win when prefix overlap is high.

4. **With weight 5 on prefix-affinity**, the argmax in `WeightedScoring.Route()` (`sim/routing.go:152-196`, composite score at line 171: `scores[snap.ID] += s * ws.weights[i]`, argmax at lines 177-184) is dominated by the prefix score, concentrating sessions on their cached instance. Lower weights dilute this signal, causing some rounds to scatter across instances, destroying cache locality.

### Why configs are near-equivalent on independent workloads

Without multi-turn or prefix groups, every request has unique input tokens.

1. **Prefix-affinity scorer returns 0.0 for all instances** (`sim/routing_prefix_scorer.go:28-29`): `MatchLength()` at line 28 returns 0 because no prior request's block hashes match the current request's hashes, so `float64(0)/float64(totalBlocks) == 0.0` at line 29.

2. **The weighted sum degrades to queue-depth and kv-utilization only**: Since prefix-affinity contributes 0 to the weighted sum, configurations C1-C4 produce routing decisions based on queue-depth (which is near-uniform at this rate) and kv-utilization (also near-uniform).

3. **Identical distributions observed**: llmd-default and load-balance produce byte-identical results because both effectively route by queue-depth + kv-utilization (prefix-affinity contributes 0). Similarly, cache-heavy and queue-heavy are near-identical because both have one scorer contributing 0 signal.

### Why kv-heavy is consistently worse on independent workloads

With `kv-utilization:5,prefix-affinity:1`, the dominant signal is `1 - KVUtilization` (`sim/routing_scorers.go:150-155`, specifically line 153: `scores[snap.ID] = 1.0 - snap.KVUtilization`). Despite producing similar total distributions ([128,125,127,120] for kv-heavy vs [126,125,124,125] for load-balance), the kv-heavy routing causes 2.2x worse scheduling delay (P99: 30.8ms vs 14.1ms). The mechanism is **temporal micro-bursting**: KV utilization changes slowly relative to queue depth (block allocation/deallocation spans multi-step lifetimes), so consecutive requests see the same KV scores and route to the same instance, creating transient bursts that queue-depth routing would have distributed. This 2.2x scheduling delay increase maps directly to the 33% TTFT penalty (27.2 vs 20.2ms), since TTFT = scheduling_delay + prefill_time.

Note: KV utilization uses `Immediate` refresh mode by default (`sim/cluster/snapshot.go:32`), so this is NOT an INV-7 staleness issue. The signal is fresh but slowly-varying by nature (KV blocks persist for the request lifetime), creating correlated routing decisions.

**Control experiment (RCV-4):** The `load-balance` config (C3: `queue-depth:3,kv-utilization:3`) serves as the control for the prefix-affinity mechanism on the prefix-heavy workload — it uses zero prefix-affinity weight. On the prefix-heavy workload, C3 produces the worst metrics (TTFT 38.8 vs 21.2, cache hit 10.7% vs 39.7%), confirming that the prefix-affinity scorer is the mechanism. On the independent workload, C3 produces metrics equivalent to C1/C2/C4, confirming the mechanism vanishes when prefix overlap is absent.

## Devil's Advocate (RCV-5)

**If this is "Confirmed," argue why it might be Refuted:**
On the prefix-heavy workload, one config dominates everything. This suggests the hypothesis is WRONG for the workload it was originally proposed for — there is no tradeoff when cache locality provides a pure win. The "Pareto frontier" on the independent workload is trivial (4 of 5 configs are equivalent within noise, not genuinely trading off different objectives).

**If this is "Refuted," argue why it might be Confirmed:**
The cross-workload analysis shows the OPTIMAL config changes with workload: cache-heavy is best for prefix-heavy, while llmd-default/load-balance are best for independent. This IS a Pareto frontier — just across workloads rather than within a single workload. The composable scorer framework lets users tune for their specific workload, which is exactly the design intent.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Cache-heavy dominates on prefix-heavy workloads (no within-workload frontier) | Design limitation | The hypothesis assumed tradeoffs within a single workload, but cache locality is a pure win on prefix-heavy workloads -- the beta1/beta2 ratio (17.67/2.84 = 6.2x) makes cache misses far more expensive than any load-balance gain. |
| Cache-heavy dominates on mixed workloads too (Round 2) | Confirmation of R1 finding | Even at 50% prefix overlap, cache locality wins all metrics. The 0.03% E2E P99 margin for kv-heavy is noise. |
| No within-workload Pareto frontier on any tested workload | Reclassification | Changed type from Statistical/Pareto to Statistical/Dominance per Reviewer B |
| Different workloads favor different weight configs (cross-workload dominance) | Confirmation | Documented here -- composable scorers add value for workload-specific tuning |
| kv-heavy routing causes 33% TTFT penalty on independent workloads | Design limitation | Document in scorer usage guidance |
| llmd-default and load-balance produce identical results on non-prefix workloads | Confirmation | Expected -- prefix scorer contributes 0 when no prefix overlap |
| kv-utilization-dominant routing causes temporal micro-bursting (slowly-varying signal) | Design limitation | KV utilization is slowly-varying by nature (block lifetimes), not a staleness issue. When weighted heavily, consecutive requests route to the same instance. Document in scorer usage guidance. |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None — existing R17 (signal freshness) already covers the kv-utilization staleness finding
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? None directly. The kv-utilization micro-bursting finding is related to R17/INV-7 conceptually (signal dynamics affecting routing quality) but is NOT a staleness issue — the signal is fresh but slowly-varying by nature.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, rate=500 req/s, 500 requests, llama-3.1-8b-instruct, default KV blocks, 3 seeds
- **Parameters findings depend on:** prefix-heavy finding depends on multi-turn context accumulation with 5 rounds; independent finding depends on low utilization (rate well below saturation); mixed finding depends on 50/50 split and same utilization level
- **What was NOT tested:** High utilization (near saturation where load balance matters more); different mix ratios (e.g., 10% prefix / 90% independent); different numbers of instances; different KV block counts; heterogeneous instance sizes
- **Generalizability:** The "cache-heavy dominates" finding applies whenever any significant prefix overlap exists (tested at 50% and 100% prefix traffic). For workloads without prefix reuse, weight configuration is nearly irrelevant. A within-workload Pareto frontier might emerge at high utilization where load-balance routing prevents queueing overload -- this was not tested.
- **Uncertainty quantification:** Seed variability is low (TTFT mean ranges: 1-3ms spread within each config). Effect sizes are large enough (21-45% TTFT spread between best and worst) that the findings are robust to seed variation. The noise analysis threshold (1% margin) correctly identifies spurious Pareto survivors.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Pareto dominance (prefix-heavy) | cache-heavy dominates all 4 others | High -- consistent across 3 seeds, large effect sizes |
| Effective dominance (mixed, Round 2) | cache-heavy dominates; kv-heavy non-dominated only by 0.03% E2E P99 noise | High -- per-seed breakdown shows 2 of 3 seeds favor cache-heavy on the "surviving" metric |
| Near-equivalence (independent) | 4 of 5 configs within 1% | High -- differences within noise, consistent across seeds |
| kv-heavy penalty | 33% worse TTFT on independent | High -- consistent 26-27ms vs 20ms across all seeds |
| No within-workload Pareto frontier (all 3 workloads) | Zero workloads produce genuine tradeoffs | High -- tested pure prefix, pure independent, and 50/50 mixed |
| Cross-workload dominance | Different optimal configs per workload | High -- mechanism well-understood (prefix scorer contributes 0 without prefix overlap) |
| Sample size | 3 seeds x 5 configs x 3 workloads = 45 simulations | Medium -- adequate for dominance analysis at this operating point |
| Mechanism | Cache locality (prefix-affinity scorer) | High -- traced through code, 13.5% cache hits on mixed vs 39.7% on prefix-heavy vs 0% on independent confirms graduated effect |

## Implications for Users

1. **For prefix-heavy workloads** (multi-turn chat, shared system prompts): Use high prefix-affinity weight (e.g., `prefix-affinity:5,queue-depth:1`). Cache locality dominates all other routing concerns. The default llm-d profile (`prefix-affinity:3,queue-depth:2,kv-utilization:2`) leaves ~25% TTFT improvement on the table by diluting the cache signal.

2. **For mixed workloads** (some prefix-heavy, some independent): cache-heavy routing still wins. Even at 50% prefix traffic, the cache locality benefit (13.5% cache hit rate, 27% TTFT improvement) outweighs any load-balance penalty. No configuration produces a better tradeoff.

3. **For independent workloads** (no prefix reuse): Weight configuration barely matters -- most configs produce equivalent results. Avoid kv-utilization-heavy weights, which can cause 33% TTFT penalty via stale-signal-driven load imbalance.

4. **The composable scorer framework's value** is in letting users match weights to their workload type, not in producing within-workload tradeoffs. There is no single weight configuration that is optimal across all workload types, but within a workload type, cache-heavy is never worse than alternatives when any prefix overlap exists.

## Reproducing

```
cd hypotheses/h17-pareto-frontier
./run.sh
```
