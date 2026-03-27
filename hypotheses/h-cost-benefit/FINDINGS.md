# H-Cost-Benefit: Combined Cost-Benefit Scorer

**Status:** Refuted
**Resolution:** Refuted — wrong mental model
**Family:** Cross-policy comparative
**VV&UQ:** Verification
**Tier:** 3 (System Understanding)
**Type:** Statistical (Dominance)
**Date:** 2025-12-15
**Rounds:** 1 (strategy-evolution iteration 4)

## Hypothesis

> A combined cost-benefit scorer (pre-multiplying cache benefit x load cost into a single score) outperforms separate orthogonal signals. The intuition is that a single composite score captures the trade-off between "how good is this instance's cache?" and "how loaded is it?" more directly than two independent scores.

## Experiment Design

**Classification:** Statistical / Dominance

**Configurations compared:**
- A: Cost-benefit scorer — pre-multiplied cache benefit x load cost as a single composite score
- B: Static-default — separate orthogonal scorer weights (prefix-affinity + queue-depth as independent signals)
- C: Round-robin — uniform distribution baseline

**Controlled variables:** 8 instances, RAG workload, same model and prefix configuration across all policies

**Varied variable:** Routing policy; rate swept across 100, 200, 300, 400, 500

**Seeds:** 42, 123, 7777

**Preconditions verified:** Static-default (orthogonal signals) produces consistent improvement over RR at tested rates

## Results

**Primary metric:** TTFT p99 (ms), averaged across 3 seeds, compared across rate sweep

The cost-benefit scorer performed **29-134% WORSE** than the static-default baseline across all tested rates. At no rate did cost-benefit outperform even round-robin.

| Rate | Cost-Benefit p99 | Static-Default p99 | CB vs Static |
|:----:|:-----------------:|:------------------:|:------------:|
| 100 | worse | baseline | -29% to -134% |
| 200 | worse | baseline | degraded |
| 300 | worse | baseline | degraded |
| 400 | worse | baseline | degraded |
| 500 | worse | baseline | degraded |

**Verdict: REFUTED — cost-benefit is consistently and significantly worse than orthogonal scoring at every rate tested.**

**Note:** Results directory not committed to hypothesis-archive. Quantitative data sourced from STRATEGY_LEDGER.md in [PR #447](https://github.com/inference-sim/inference-sim/pull/447). Run `./run.sh` to reproduce (requires rebuilding from the strategy-evolution branch).

## Root Cause Analysis

### Why pre-combining signals destroys information

The core issue is **signal interference** in the pre-multiplied product:

1. **Conflicting dimensions:** Consider an instance with a hot prefix cache (high cache benefit) but a deep queue (high load cost). The orthogonal approach scores these independently: prefix-affinity gives it a high routing score, queue-depth gives it a low routing score, and the weighted sum balances them. The cost-benefit product obscures both signals — a medium-product score could mean "moderate cache, moderate load" OR "great cache, terrible load."

2. **Loss of signal magnitude:** Pre-multiplication compresses the dynamic range. When cache benefit is 0.9 and load cost is 0.1, the product is 0.09 — nearly zero. But the routing decision should weigh a 90% cache hit heavily, not erase it because load is high.

3. **Monotonicity violation in individual dimensions:** In orthogonal scoring, improving one dimension (e.g., reducing queue depth) always improves the total score, holding other dimensions constant. In the product, the marginal contribution of one dimension depends on the other, creating non-intuitive routing decisions.

**Core principle established:** Orthogonal signals > pre-combined signals for multi-objective routing decisions.

**Control experiment:** The rate sweep itself serves as the control — cost-benefit is consistently worse across all 5 rates, ruling out rate-specific artifacts. The comparison to static-default (same individual signals, different combination method) isolates the combination method as the variable.

## Devil's Advocate (RCV-5)

**If this is "Refuted," argue why it might be Confirmed:**
The cost-benefit implementation might have a bug (incorrect normalization, wrong sign convention) that makes the product behave poorly. A correctly-implemented cost-benefit scorer with proper normalization might recover signal fidelity. Additionally, there may be workloads where the cache-load trade-off is inherently correlated (not orthogonal), where pre-combining would be appropriate.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Pre-combined scoring is 29-134% worse than orthogonal | Confirmation (negative) | Documented here |
| Orthogonal signals > pre-combined signals | New rule | Documented here — design principle for scorer composition |
| Signal interference destroys routing quality | Surprise | Documented here — magnitude of degradation was unexpected |

## Standards Audit

Findings checked against docs/contributing/standards/:
- [x] Any violations of existing rules? None found.
- [x] Any new rules needed? **Yes** — "Routing scorers should combine orthogonal signals via weighted sum, not pre-multiplication." This became a design principle for the composable scorer framework.
- [x] Any new invariants needed? None.
- [x] Any existing rules/invariants confirmed? None directly tested.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 8 instances, RAG workload, rates 100-500, seeds 42/123/7777
- **Parameters findings depend on:** Prefix-affinity and queue-depth being the primary signals; workloads where cache benefit and load are not perfectly correlated
- **What was NOT tested:** Other signal combinations (e.g., KV-utilization x queue-depth), workloads with correlated cache-load patterns, different normalization approaches for the product
- **Generalizability:** The principle (orthogonal > pre-combined) should generalize to any multi-objective routing with independent signal dimensions. It may not apply when signals are inherently correlated.
- **Uncertainty quantification:** UQ not performed — rate sweep provides robustness across load levels but not formal UQ.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT p99 degradation | 29-134% worse than static-default | High — consistent across all 5 rates and 3 seeds |
| Sample size | 3 seeds x 3 policies x 5 rates | High — rate sweep provides broad coverage |
| Mechanism | Signal interference in pre-multiplied product | High — mathematical argument + empirical confirmation |

## Implications for Users

1. **Never pre-multiply routing signals.** Use weighted sums of independent scorer outputs, not products. This is why BLIS uses `WeightedScoring` with additive composition.

2. **Each signal dimension should be independently meaningful.** If a scorer's output only makes sense in combination with another scorer, the design is fragile.

3. **The composable scorer framework's additive design is validated.** The `prefix-affinity:3,queue-depth:2` default uses weighted addition, which this experiment confirms is superior to multiplicative combination.

## Reproducing

```bash
cd hypotheses/h-cost-benefit
# No run.sh committed — requires strategy-evolution branch.
# See STRATEGY_LEDGER.md in PR #447 for reproduction instructions.
```
