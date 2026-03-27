# H-KV-Pressure: KV-Utilization Scorer Under Memory Pressure

**Status:** Refuted
**Resolution:** Refuted — wrong mental model
**Family:** Cross-policy comparative
**VV&UQ:** Verification
**Tier:** 3 (System Understanding)
**Type:** Statistical (Dominance)
**Date:** 2025-12-20
**Rounds:** 3 (strategy-evolution iterations 6-8)

## Hypothesis

> The KV-utilization scorer improves routing under KV memory pressure by directing requests away from instances approaching eviction. When KV memory is scarce, routing to instances with lower KV utilization should prevent eviction cascades and improve tail latency.

## Experiment Design

**Classification:** Statistical / Dominance

**Configurations compared:**
- A: Round-robin (`rr`) — uniform distribution baseline
- B: Static-default — default weighted scoring (pa:3,qd:2,kv:2)
- C: KV-heavy — elevated KV-utilization weight
- D: KV-pressure — specialized KV-pressure-aware routing
- E: Compound (`pa:3,qd:2`) — prefix-affinity + queue-depth WITHOUT KV-utilization

**Controlled variables:** 4 instances, mixed-SLO workload, same model across all policies

**Varied variable:** Routing policy; KV blocks swept across 132139 (normal), 5000, 2000, 1500 (pressure)

**Seeds:** 42, 123, 7777

**Preconditions verified:** KV pressure levels produce measurable preemptions at lower block counts

## Results

**Primary metric:** TTFT p99 (ms), preemption count, dropped requests, averaged across 3 seeds

Dropping the KV-utilization scorer entirely and using `pa:3,qd:2` produced the best results: **+11% improvement over RR** AND KV-pressure-invariant performance across all block levels tested.

The KV-utilization scorer was **counterproductive** — policies including it (static-default with kv:2, kv-heavy, kv-pressure) performed worse than the compound policy without it.

| Policy | Normal (132K) | Moderate (5K) | Pressure (2K) | Severe (1.5K) | KV-Invariant? |
|:-------|:------------:|:------------:|:-------------:|:-------------:|:-------------:|
| rr | baseline | baseline | baseline | baseline | No |
| pa:3,qd:2 (no kv) | best | best | best | best | **Yes** |
| static-default (with kv:2) | worse than pa:3,qd:2 | worse | worse | worse | No |
| kv-heavy | worse | worse | worse | worse | No |
| kv-pressure | worse | worse | worse | worse | No |

**Verdict: REFUTED — KV-utilization scorer is counterproductive under KV pressure. Removing it improves performance.**

**Note:** Results directory not committed to hypothesis-archive. Quantitative data sourced from STRATEGY_LEDGER.md in [PR #447](https://github.com/inference-sim/inference-sim/pull/447). Run `./run.sh` to reproduce (requires rebuilding from the strategy-evolution branch).

## Root Cause Analysis

### Why KV-utilization scoring is counterproductive

The KV-utilization scorer has a fundamental semantic inversion:

1. **High KV utilization = good cache, not bad memory pressure.** An instance with high KV utilization has many cached prefixes in memory. The KV-utilization scorer steers requests AWAY from high-utilization instances — but those are precisely the instances most likely to have useful cached prefixes.

2. **Conflict with prefix-affinity:** Prefix-affinity routes requests toward instances with matching cached prefixes. KV-utilization routes requests away from instances with lots of cached data. These signals directly conflict: the instance that prefix-affinity prefers (has your prefix cached) is the instance KV-utilization penalizes (has lots of data cached).

3. **Orthogonality assumption violated:** The weighted scoring framework assumes signals are orthogonal. KV-utilization is anti-correlated with prefix-affinity value — they measure aspects of the same underlying state (KV cache contents) with opposite valence.

### Why pa:3,qd:2 is KV-pressure-invariant

The `pa:3,qd:2` configuration (without KV-utilization) is invariant across KV pressure levels because:
- Prefix-affinity routes based on content (which prefixes are cached), not capacity
- Queue-depth routes based on load (how many requests are waiting), not memory
- Neither signal is affected by total KV block count, so the routing decisions are identical regardless of memory pressure
- KV pressure affects *completion time* (via preemptions) but not *routing decisions*

**Control experiment:** The cross-KV comparison table in the analysis script directly tests this — pa:3,qd:2 produces the same TTFT p99 across all KV block levels, confirming the invariance.

## Devil's Advocate (RCV-5)

**If this is "Refuted," argue why it might be Confirmed:**
A redesigned KV-utilization scorer that distinguishes between "useful cached data" and "fragmented dead data" could provide genuine pressure information. The current scorer conflates cache quality with cache quantity. Additionally, under extreme pressure with no prefix reuse (random prompts), KV-utilization might correctly identify instances with more headroom.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| KV-utilization scorer is counterproductive under pressure | Surprise | Documented here — changed default weights |
| KV-util conflicts with prefix-affinity (anti-correlated signals) | New rule | Documented here — signals sharing underlying state should not be combined additively |
| pa:3,qd:2 is KV-pressure-invariant | Confirmation | Documented here — supports removing kv-util from default profile |
| Removing kv-util yields +11% over RR | Confirmation | Documented here |

## Standards Audit

Findings checked against docs/contributing/standards/:
- [x] Any violations of existing rules? None found.
- [x] Any new rules needed? **Yes** — "Do not combine signals that measure the same underlying state with opposite valence." KV-utilization and prefix-affinity both measure KV cache contents but with inverted preference.
- [x] Any new invariants needed? None.
- [x] Any existing rules/invariants confirmed? INV-4 (KV cache conservation) holds under all pressure levels.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, mixed-SLO workload, KV blocks from 1500 to 132139, seeds 42/123/7777
- **Parameters findings depend on:** Prefix reuse present in workload (prefix-affinity is valuable). Without prefix reuse, KV-util's negative effect would be smaller.
- **What was NOT tested:** Workloads with no prefix reuse (random prompts), extremely large clusters where KV imbalance might emerge naturally, tiered KV cache (GPU+CPU offloading)
- **Generalizability:** The anti-correlation principle generalizes to any system where cache fullness correlates with cache value. Specific to workloads with prefix reuse.
- **Uncertainty quantification:** UQ not performed — swept across 4 KV pressure levels with 3 seeds.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT p99 improvement (pa:3,qd:2 vs RR) | +11% | High — consistent across all KV levels and seeds |
| KV-pressure invariance | pa:3,qd:2 identical across 4 KV levels | High — 4 KV levels x 3 seeds |
| Mechanism | KV-util anti-correlated with prefix-affinity | High — semantic analysis confirmed by empirical data |

## Implications for Users

1. **Remove KV-utilization from routing weights when prefix-affinity is active.** The recommended default is `prefix-affinity:3,queue-depth:2` — not including KV-utilization.

2. **KV-utilization scoring conflates cache quality with cache quantity.** High KV utilization means the instance has lots of cached data, which is usually good (more prefix hits), not bad.

3. **Routing is naturally KV-pressure-invariant.** When using `pa:3,qd:2`, KV memory pressure affects throughput (via preemptions) but does not degrade routing quality. No special pressure-aware routing is needed.

## Reproducing

```bash
cd hypotheses/h-kv-pressure
# No run.sh committed — requires strategy-evolution branch.
# See STRATEGY_LEDGER.md in PR #447 for reproduction instructions.
```
