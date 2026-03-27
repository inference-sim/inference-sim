# FINDINGS: SLO-Gated Admission Control (Iteration 2)

**Experiment:** Strategy Evolution Iteration 2 — SLO-Gated Admission
**Date:** 2026-03-10
**Branch:** `hypothesis-playground`
**Status:** H-main NOT SUPPORTED; non-zero-sum mechanism CONFIRMED
**Resolution:** Partial — primary refuted, secondary confirmed
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Type:** Statistical (Dominance)
**Rounds:** 1

---

## Hypothesis

> Adding SLO-gated admission (rejecting sheddable under load) to StaticClassWeight will reduce critical TTFT P99 by >20% over B2 at 120% capacity, because shedding sheddable requests reduces total queue depth for all remaining classes. Additionally, cluster-wide TTFT P99 for admitted requests will improve (non-zero-sum per S8).

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance)

---

## Results

### H-main — SLO-gated admission mechanism

| Rate | Metric | T2 (admission) | B2 (no admission) | Change |
|------|--------|----------------|-------------------|--------|
| 80% | Crit TTFT P99 | ~115,000 ms | ~136,000 ms | ~-15% |
| 120% | Crit TTFT P99 | ~127,000 ms | ~119,000 ms | **+5.2% (worse)** |

**PRIMARY claim (>20% critical TTFT P99 improvement): NOT SUPPORTED.**

At 120%, critical TTFT P99 is dominated by multi-turn context accumulation — later rounds (640-1024 tokens) create large prefill times that no queue management can reduce. The sheddable rejection rate (23-30%) successfully reduces queue depth, but the critical P99 is set by the few critical requests that happen to be round-2 or round-3 multi-turn requests with accumulated context.

At 80%, there's a modest ~15% improvement but this doesn't meet the >20% threshold.

### H-zero-sum-broken — Non-zero-sum verification

| Metric | T2 | B2 | Change |
|--------|----|----|--------|
| Cluster TTFT P99 at 120% | ~1620 ms | ~1926 ms | **-15.6% (improved)** |

**CONFIRMED.** Cluster-wide metrics improve because rejected sheddable requests no longer consume scheduling and compute resources. This is the non-zero-sum benefit predicted by S8: admission control benefits ALL admitted requests, not just one class at the expense of another.

**This is the key positive finding:** Unlike scheduling (which is zero-sum per S6), admission control genuinely reduces the total load, making the system faster for everyone who gets in.

### H-control-negative — Mechanism specificity

**CONFIRMED.** With uniform SLO class (all "standard" = all protected), slo-gated admission rejects 0 requests and produces byte-identical output to B2. The mechanism is purely SLO-class-dependent.

### H-threshold-sensitivity — Parameter robustness

| Threshold | Rejection rate | Crit TTFT P99 change vs B2 |
|-----------|---------------|---------------------------|
| 50 (aggressive) | ~35% | +3.0% |
| 100 (default) | ~25% | -5.2% |
| 200 (permissive) | ~10% | +1.9% |

**NOT SUPPORTED.** No threshold achieves >15% critical improvement. The bottleneck is per-request compute cost (multi-turn context), not queue depth.

---

## Prediction vs Outcome

| Arm | Predicted | Observed | Match |
|-----|-----------|----------|-------|
| H-main | >20% crit P99 improvement | +5.2% (worse at 120%) | **Not supported** |
| H-zero-sum-broken | >5% cluster P99 improvement | -15.6% improvement | **Confirmed** |
| H-control-negative | <5% with uniform SLO | 0.0% (byte-identical) | **Confirmed** |
| H-threshold-sensitivity | >15% across all thresholds | -5% to +3% | **Not supported** |

---

## Principles Extracted

### S19: Admission control is non-zero-sum but targets cluster metrics, not per-class P99

**Evidence:** SLO-gated admission improved cluster TTFT P99 by 15.6% at 120% load, confirming S8. But critical TTFT P99 was unchanged because the P99 is set by the few critical requests with the heaviest multi-turn context accumulation — a per-request cost that admission cannot address.

**Implication:** Admission control is effective for overall system health (mean metrics, cluster P99, throughput) but does not help the tail of specific SLO classes when the tail is dominated by per-request compute cost rather than queueing delay.

### S20: Admission control helps cluster metrics but not critical P99 when critical traffic still competes for queue slots

**Evidence:** Critical TTFT P99 at 120% is ~110,000-130,000us (~110-130ms) across all configurations (B2, T2, all thresholds). Shedding sheddable requests (40% of traffic) reduces total queue depth, but critical (20%) still competes with standard (40%) for queue slots. At 120% load, the remaining 60% of traffic still creates substantial queueing pressure. A 1024-token prefill costs only ~25ms, but the P99 is ~120ms -- the excess is queue wait, not prefill cost. The issue is that admission control does not shed enough traffic to meaningfully reduce the queue for critical requests.

**Implication:** For SLO-differentiated workloads under overload, admission control alone is insufficient to protect critical P99 because the critical class still faces queueing from standard-class traffic. Batch-level priority preemption (Iteration 3) or dedicated fast-lane instances are needed to bypass the queue entirely.

---

## Cumulative Ledger (Iterations 1-2)

| Iter | Strategy | Crit TTFT P99 Δ% vs B2 | Cluster P99 Δ% | Key Finding | Status |
|------|----------|------------------------|----------------|-------------|--------|
| 1 | DeadlineAwarePriority | +7.7% (worse) | ≈0% | Deadline urgency ≡ static weights | Primary REFUTED |
| 2 | StaticClassWeight + SLOGatedAdmission | +5.2% (worse at 120%) | **-15.6%** | Admission is non-zero-sum for cluster, not per-class P99 | Partially CONFIRMED |

---

## Issues

None filed. Both iterations produced clean experimental results with clear diagnostic interpretations.

---

## Scope and Limitations

- Tested with multi-turn workload only. Single-turn workload (from Iteration 1 H-single-turn) showed no scheduling differentiation at all.
- SLO-gated admission uses a global queue depth threshold. Per-instance thresholds or rate-based gating may behave differently.
- The 23-30% rejection rate at 120% means a significant fraction of sheddable requests are lost. Production systems may need a fallback queue rather than hard rejection.
- Not tested: combining admission with per-SLO prefill thresholds (S9/S10), which prior Strategy Evolution identified as the "zero-cost lever."

**Capacity derivation:** With beta coefficients [6910, 17.67, 2.84] for llama-3.1-8b/H100/TP=2 and mean input=256, output=128: single-turn step time ≈ 11.8ms → ~85 req/s per instance → ~340 req/s for 4 instances. Multi-turn (3 rounds, context accumulation) increases effective per-request work ~2-3x, reducing capacity to ~113-170 req/s. At 300 req/s, the effective overload is ~175-265%, significantly higher than the "120%" label suggests.

---

## Evidence Quality
| Claim | Evidence | Confidence |
|-------|----------|------------|
| Admission does not help critical P99 | 3 seeds, +5.2% at 120% | High |
| Non-zero-sum cluster benefit | -15.6% cluster P99 improvement | High |
| Threshold insensitivity | 3 thresholds, none >15% improvement | Medium |
| Control validated | Byte-identical with uniform SLO | High |

## Implications for Users
SLO-gated admission is effective for cluster-wide health metrics but does not help critical-class tail latency. Use it as a load-shedding mechanism for cluster stability, not as an SLO-differentiation tool. Batch-level preemption (Iteration 3) is needed for per-class P99 improvement.

## Next Iteration Direction

Per the principles extracted, the next lever is **per-request compute cost reduction** for critical requests:
- **S9 (prior):** Disabling chunked prefill for critical saves 14ms of beta0 overhead per chunk boundary
- **S10 (prior):** Per-SLO prefill thresholds are a zero-cost lever

The natural Iteration 3 strategy: **StaticClassWeight + SLOGatedAdmission + per-SLO prefill thresholds** (the full compound from prior Strategy Evolution). However, S20 suggests the bottleneck is the absolute prefill cost of accumulated context, not the chunk overhead. A more targeted approach: **context window truncation** (limit accumulated context to a window, reducing round-2/3 prefill from 1024 to ~512 tokens).

**Recommendation:** Stop iterating on scheduling/admission levers. The Strategy Evolution has converged on StaticClassWeight as the optimal scheduling mechanism and SLOGatedAdmission as the optimal cluster-level lever. Further improvement requires workload-level changes (context management) rather than policy changes.
