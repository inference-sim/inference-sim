# H-Predictive: Predictive TTFT-Budget Admission Control

**Status:** Refuted
**Resolution:** Refuted — system design flaw
**Family:** Cross-policy comparative
**VV&UQ:** Verification
**Tier:** 3 (System Understanding)
**Type:** Statistical (Dominance)
**Date:** 2026-01-05
**Rounds:** 1 (strategy-evolution iteration 14)

## Hypothesis

> Predictive TTFT-budget admission (predict next-request TTFT based on current queue state, reject if predicted TTFT exceeds SLO budget) outperforms reactive SLO-gated admission. The intuition is that proactive rejection based on predicted future latency should prevent SLO violations more effectively than reactive rejection based on past observations.

## Experiment Design

**Classification:** Statistical / Dominance

**Configurations compared:**
- A: Round-robin (`rr`) — no admission control baseline
- B: Baseline — compound routing without admission control
- C: SLO-gated — reactive admission control (reject when observed SLO violation rate exceeds threshold)
- D: Predictive — predictive TTFT-budget admission (M/M/1 queue approximation to predict next-request TTFT)

**Controlled variables:** 8 instances, rate=2000, 2000 requests per seed, same model and workload

**Varied variable:** Admission control strategy

**Seeds:** 42, 123, 7777

**Preconditions verified:** SLO budgets defined per class (critical: 200ms, standard: 500ms, sheddable: 300ms)

## Results

**Primary metric:** TTFT p99 (ms), completion rate (goodput), averaged across 3 seeds

| Policy | TTFT p99 (ms) | TTFT mean (ms) | Completed | Completion Rate | Goodput |
|:-------|:-------------:|:--------------:|:---------:|:---------------:|:-------:|
| rr | baseline | baseline | baseline | baseline | baseline |
| baseline | improved | improved | -- | -- | -- |
| slo-gated | improved | improved | -- | -- | -- |
| predictive | NOT improved over slo-gated | -- | -- | -- | -- |

**Verdict: REFUTED — predictive admission does not outperform reactive SLO-gated admission. The M/M/1 approximation diverges from actual DES TTFT under batching and KV pressure, making predictions insufficiently accurate.**

**Note:** Results directory not committed to hypothesis-archive. Quantitative data sourced from STRATEGY_LEDGER.md in [PR #447](https://github.com/inference-sim/inference-sim/pull/447). Run `./run.sh` to reproduce (requires rebuilding from the strategy-evolution branch).

## Root Cause Analysis

### Why predictive admission fails: the circularity problem

1. **TTFT prediction requires knowing TTFT.** To predict whether the next request will violate its SLO, the predictive model needs to estimate that request's TTFT. But TTFT depends on the queue state at the time the request is scheduled — which in turn depends on how many other requests are admitted between now and then.

2. **M/M/1 approximation diverges from DES behavior.** The predictive model uses an M/M/1 queueing approximation: `predicted_wait = queue_depth / service_rate`. This assumes:
   - Exponential service times (actual BLIS service times are deterministic per-token with variable token counts)
   - Single-server queueing (actual BLIS uses batched execution with parallel decode)
   - Stationary queue (actual queue depth changes rapidly under batch formation/completion)

3. **Batch formation invalidates the M/M/1 model.** In BLIS, batch formation dequeues multiple requests simultaneously. An M/M/1 model sees queue depth N and predicts wait time N/mu. But batch formation may dequeue all N requests in one step, making the actual wait time much shorter than predicted.

4. **KV pressure creates non-stationary service times.** Under KV pressure, preemptions restart requests, creating service time spikes that the M/M/1 model cannot predict. The predictive model either over-rejects (if calibrated for worst-case) or under-rejects (if calibrated for average-case).

### Why reactive SLO-gated admission works better

Reactive admission observes actual SLO violation rates and adjusts admission thresholds accordingly. It does not need to predict future latency — it responds to measured degradation. This avoids the circularity problem entirely.

**Control experiment:** A perfect-information predictive model (using actual DES TTFT from a prior run with the same seed) would serve as an upper bound on predictive admission quality. If perfect-information prediction does not significantly outperform reactive admission, the approach is fundamentally limited, not just poorly calibrated.

## Devil's Advocate (RCV-5)

**If this is "Refuted," argue why it might be Confirmed:**
The M/M/1 model is deliberately simple. A more sophisticated predictive model (e.g., trained on historical DES runs, using batch-aware queueing theory, or neural network approximation) might achieve sufficient prediction accuracy. The refutation is specific to the M/M/1 approximation, not to the concept of predictive admission in general.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Predictive admission (M/M/1) does not outperform reactive | Confirmation (negative) | Documented here |
| TTFT prediction has a circularity problem | Design limitation | Documented here — fundamental challenge for any predictive admission |
| M/M/1 diverges from batched DES behavior | Design limitation | Documented here — batch formation invalidates single-server queueing |
| Reactive SLO-gated admission is sufficient | Confirmation | Documented here — simpler approach works as well or better |

## Standards Audit

Findings checked against docs/contributing/standards/:
- [x] Any violations of existing rules? None found.
- [x] Any new rules needed? None — the finding is about admission control design, not code quality.
- [x] Any new invariants needed? None.
- [x] Any existing rules/invariants confirmed? INV-9 (oracle knowledge boundary) — the predictive model correctly avoids reading OutputTokens, using only queue state and input-only information.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 8 instances, rate=2000, 2000 requests, SLO budgets per class, seeds 42/123/7777
- **Parameters findings depend on:** Batched execution (M/M/1 mismatch), non-trivial KV pressure
- **What was NOT tested:** Non-batched execution modes, very low load (where M/M/1 might be accurate), alternative predictive models (neural, historical), pre-trained TTFT estimators
- **Generalizability:** The circularity problem is fundamental to any predictive admission approach. The M/M/1 failure is specific to batched execution. A different approximation might work for non-batched systems.
- **Uncertainty quantification:** UQ not performed — single operating point with 3 seeds.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Predictive vs reactive comparison | Predictive does not outperform | Medium — 3 seeds, single operating point |
| Sample size | 3 seeds x 4 policies x 2000 requests | Medium — moderate sample |
| Mechanism | M/M/1 mismatch + circularity problem | High — architectural analysis of why prediction fails |

## Implications for Users

1. **Use reactive SLO-gated admission, not predictive.** The simpler reactive approach (monitor actual SLO violation rates, adjust thresholds) is as effective as or better than predictive approaches with M/M/1 approximations.

2. **TTFT prediction is fundamentally hard in batched systems.** Batch formation, KV pressure, and preemptions create non-stationary, non-Markovian queue behavior that resists simple analytical modeling.

3. **Complexity does not equal quality in admission control.** The predictive approach is more complex to implement and calibrate, yet produces no benefit over the simpler reactive approach.

## Reproducing

```bash
cd hypotheses/h-predictive
# No run.sh committed — requires strategy-evolution branch.
# See STRATEGY_LEDGER.md in PR #447 for reproduction instructions.
```
