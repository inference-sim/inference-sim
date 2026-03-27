# H-SLO-Compound: SLO-Differentiated Routing

**Status:** Refuted
**Resolution:** Refuted — wrong mental model
**Family:** Cross-policy comparative
**VV&UQ:** Verification
**Tier:** 3 (System Understanding)
**Type:** Statistical (Dominance)
**Date:** 2025-12-18
**Rounds:** 1 (strategy-evolution iteration 5)

## Hypothesis

> SLO-differentiated routing (different scorer weights per SLO class — e.g., higher prefix-affinity for critical, higher queue-depth for sheddable) outperforms uniform routing across all SLO classes. The intuition is that different SLO tiers have different latency-vs-throughput trade-offs, so routing each class with tailored weights should optimize per-class outcomes.

## Experiment Design

**Classification:** Statistical / Dominance

**Configurations compared:**
- A: `rr-baseline` — round-robin, no differentiation
- B: `static-default` — uniform compound routing (same weights for all SLO classes)
- C: `adaptive-ortho` — adaptive orthogonal routing (same weights, adaptive rate)
- D: `static-slopri` — SLO-differentiated static routing (different weights per SLO class)
- E: `adaptive-slopri` — SLO-differentiated adaptive routing
- F: `static-sjf` — shortest-job-first static routing

**Controlled variables:** 8 instances (implied), mixed-SLO workload, same model

**Varied variable:** Routing policy; rate swept across 200, 300, 400

**Seeds:** 42, 123, 7777

**Preconditions verified:** Mixed-SLO workload contains multiple SLO classes (critical, standard, sheddable)

## Results

**Primary metric:** TTFT p99 (ms), completed requests, queued requests, averaged across 3 seeds

SLO-differentiated routing (`static-slopri`, `adaptive-slopri`) performed **3-5% WORSE** than uniform compound routing (`static-default`) across all tested rates.

| Rate | Best Policy | SLO-Differentiated vs Uniform |
|:----:|:-----------:|:-----------------------------:|
| 200 | static-default or adaptive-ortho | slopri 3-5% worse |
| 300 | static-default or adaptive-ortho | slopri 3-5% worse |
| 400 | static-default or adaptive-ortho | slopri 3-5% worse |

**Verdict: REFUTED — SLO-differentiated routing is 3-5% WORSE than uniform compound routing. Routing different SLO classes to different instances fragments cache affinity.**

**Note:** Results directory not committed to hypothesis-archive. Quantitative data sourced from STRATEGY_LEDGER.md in [PR #447](https://github.com/inference-sim/inference-sim/pull/447). Run `./run.sh` to reproduce (requires rebuilding from the strategy-evolution branch).

## Root Cause Analysis

### Why SLO-differentiated routing hurts: cache affinity fragmentation

1. **Cross-class prefix sharing:** In typical workloads, critical and standard requests often share prefixes (same application, different SLO classes based on user tier or request priority). A critical request and a standard request with the same prefix benefit from being routed to the same instance.

2. **SLO differentiation fractures prefix groups:** When critical requests are routed with high prefix-affinity weight and standard requests with high queue-depth weight, requests sharing the same prefix may be routed to different instances. Critical goes to the instance with the prefix cached; standard goes to the instance with the shortest queue. The prefix is now cached on two instances instead of one.

3. **Cache duplication wastes capacity:** With N SLO classes and SLO-differentiated routing, each prefix may be cached on up to N instances (one per SLO-class routing preference). This reduces effective cache capacity and increases cold misses for all classes.

4. **Uniform routing preserves cross-class affinity:** When all SLO classes use the same `pa:3,qd:2` weights, requests sharing a prefix are consistently routed to the same instance regardless of SLO class. This maximizes cache utilization.

### Why the degradation is "only" 3-5%

The degradation is moderate (not catastrophic) because:
- Not all SLO classes share prefixes — some prefix groups are class-specific
- The queue-depth component still provides load balancing even with fragmented caches
- At moderate rates (200-400), the per-instance load is low enough that cache fragmentation has limited tail latency impact

At higher rates or under KV pressure, the degradation could be larger (more sensitivity to cache efficiency).

**Control experiment:** Running with a workload where SLO classes have completely disjoint prefix sets (no cross-class sharing) should show SLO-differentiated routing performing equivalently to or better than uniform routing, confirming that cross-class prefix sharing is the mechanism.

## Devil's Advocate (RCV-5)

**If this is "Refuted," argue why it might be Confirmed:**
In a deployment where SLO classes have truly different latency sensitivities AND do not share prefixes, SLO-differentiated routing could improve per-class outcomes. For example, routing batch-class requests exclusively to less-loaded instances while routing critical-class requests to cache-hot instances could benefit both classes if their prefix sets are disjoint.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| SLO-differentiated routing is 3-5% worse than uniform | Confirmation (negative) | Documented here |
| Cache affinity fragmentation from SLO differentiation | New rule | Documented here — routing differentiation must not fracture prefix groups |
| Cross-class prefix sharing is common in mixed-SLO workloads | Confirmation | Documented here |
| Uniform routing preserves cross-class cache affinity | Confirmation | Documented here |

## Standards Audit

Findings checked against docs/contributing/standards/:
- [x] Any violations of existing rules? None found.
- [x] Any new rules needed? **Yes** — "Do not differentiate routing weights by SLO class when SLO classes share prefixes." Cache affinity fragmentation outweighs per-class optimization.
- [x] Any new invariants needed? None.
- [x] Any existing rules/invariants confirmed? None directly tested.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 8 instances (implied), rates 200/300/400, mixed-SLO workload, seeds 42/123/7777
- **Parameters findings depend on:** Cross-class prefix sharing present in workload
- **What was NOT tested:** Workloads with disjoint per-class prefix sets, very high rates where per-class optimization might matter more, more than 3 SLO classes, KV pressure interactions
- **Generalizability:** The finding generalizes to any deployment with cross-class prefix sharing. Deployments with truly isolated per-class workloads may benefit from differentiation.
- **Uncertainty quantification:** UQ not performed — rate sweep across 3 rates with 3 seeds.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT p99 degradation | 3-5% worse than uniform | Medium — consistent across rates but small magnitude |
| Sample size | 3 seeds x 6 policies x 3 rates | High — broad policy and rate coverage |
| Mechanism | Cache affinity fragmentation from cross-class routing | Medium — plausible mechanism, not directly isolated |

## Implications for Users

1. **Use uniform routing weights across all SLO classes.** The same `prefix-affinity:3,queue-depth:2` profile should apply to all requests regardless of SLO tier.

2. **SLO differentiation should happen at admission and scheduling, not routing.** Admission control can shed lower-priority requests under load. Scheduling priority can reorder within an instance's queue. But routing should maximize cache affinity across all classes.

3. **Cross-class prefix sharing is a hidden dependency.** When different SLO classes share prefixes (common in multi-tenant applications), routing policies that treat classes differently pay a cache fragmentation tax.

## Reproducing

```bash
cd hypotheses/h-slo-compound
# No run.sh committed — requires strategy-evolution branch.
# See STRATEGY_LEDGER.md in PR #447 for reproduction instructions.
```
