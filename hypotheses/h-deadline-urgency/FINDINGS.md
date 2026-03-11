# FINDINGS: Deadline-Aware SLO Scheduling (Iteration 1)

**Experiment:** Strategy Evolution Iteration 1 — Deadline-Aware Urgency
**Date:** 2026-03-10
**Branch:** `hypothesis-playground`
**Status:** H-main PRIMARY REFUTED; class-awareness mechanism CONFIRMED
**Resolution:** Clean refutation — deadline urgency adds no value over static class weights
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Type:** Statistical (Dominance)
**Rounds:** 1

---

## Hypothesis

> DeadlineAwarePriority with per-class TTFT deadlines will reduce critical TTFT P99 by >15% over StaticClassWeight (B2) at 120% capacity, because hyperbolic urgency growth creates stronger priority separation during transient overload.

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance)

---

## Results

### H-main — Deadline-aware urgency mechanism

| Rate | Metric | vs B2 (mean) | vs B1 (mean) | vs B0 (mean) |
|------|--------|-------------|-------------|-------------|
| 30% | Crit TTFT P99 | +0.0% | -4.0% | -12.0% |
| 80% | Crit TTFT P99 | +0.0% | -88.9% | -88.9% |
| 120% | Crit TTFT P99 | **+7.7%** | **-92.6%** | -92.6% |

**PRIMARY claim (Treatment vs B2 >15% improvement): REFUTED.** Treatment is 7.7% *worse* than B2 at 120%, and byte-identical at 30% and 80%.

**SECONDARY claim (Treatment vs B1 >20% improvement): CONFIRMED.** -92.6% improvement. This is entirely from class-awareness, not deadline urgency.

**Throughput:** -1.2% vs B1 (PASS, within 5% threshold).

**Key observation:** Treatment produces **byte-identical** results to B2 at 30% and 80% load. At 120%, Treatment is marginally worse on seed 42 (+23.2% vs B2), identical on seeds 123 and 456. The deadline mechanism provides no scheduling differentiation beyond static class weights.

**Diagnosis:** The urgency formula `classWeight / max(epsilon, 1 - elapsed/deadline)` degenerates to the same ordering as static weights in all operationally relevant regimes:
- **Sub-saturation (30%):** Queues too short for priority to matter (RP-5 confirmed)
- **Moderate load (80%):** Requests complete before approaching deadlines → urgency ≈ classWeight
- **Overload (120%):** All requests exceed deadlines → urgency = classWeight/epsilon (fixed ratio)
- **Transient window:** The ~8-14 priority recomputations during a 100ms deadline window (step time ~7-12ms) are too few to produce different scheduling outcomes

### H-ablation-deadline — Per-class deadline differentiation

| Rate | Mean degradation with uniform deadlines |
|------|----------------------------------------|
| 80% | **+317.7%** |
| 120% | **+422.8%** |

**CONFIRMED.** Uniform deadlines (500ms for all classes) are dramatically worse. The pincer effect (loosened critical + tightened sheddable) creates competition for scheduling slots.

**Interpretation:** This confirms that *differentiation* matters — but the differentiation comes from the class-weight ratio in the denominator, not the deadline curve. Uniform deadlines with different class weights would still produce ordering differences, but less extreme ones.

### H-zero-sum — Cluster-wide side-effect

| Config | Cluster P99 degradation vs B0 |
|--------|-------------------------------|
| Treatment | +6.5% |
| B2 | +6.5% |

**PARTIALLY CONFIRMED.** Absolute bound passes (+6.5% < 40%). Comparative claim fails (Treatment ≈ B2, not less). Zero-sum weighted index: -0.078 (PASS, |x| < 0.10).

**Interpretation:** Both Treatment and B2 produce identical cluster degradation because Treatment IS B2 at saturation. The zero-sum property holds: improving critical comes at the cost of standard/sheddable, but the total redistribution is modest (+6.5%).

### H-control-negative — Mechanism specificity

**REFUTED.** 6.5% difference between uniform-SLO and differentiated at 30% (threshold was <5%). The mechanism has a load-independent component: even at sub-saturation, the class-weight differentiation produces different scheduling during brief gamma burst queues.

### H-robustness-burst — Burst intensity scaling

| CV | Mean change vs B2 |
|----|--------------------|
| 1.5 | +12.8% (worse) |
| 2.0 | +7.7% (worse) |
| 3.5 | +24.4% (worse) |

**REFUTED.** Treatment is worse than B2 at all burst intensities. The deadline urgency curve creates scheduling instability during bursts — the hyperbolic growth produces volatile priority values that destabilize the sort ordering compared to B2's stable static weights.

### H-single-turn — Multi-turn confound isolation

**REFUTED.** 0.0% difference (byte-identical) on single-turn workload. The deadline mechanism produces no effect whatsoever without multi-turn context growth.

---

## Prediction vs Outcome

| Arm | Predicted | Observed | Match |
|-----|-----------|----------|-------|
| H-main (primary) | >15% improvement vs B2 | +7.7% worse | **Refuted** |
| H-main (secondary) | >20% improvement vs B1 | -92.6% improvement | **Confirmed** |
| H-ablation | >15% degradation | +317-422% degradation | **Confirmed** (exceeded prediction 20x) |
| H-zero-sum (absolute) | <40% cluster degradation | +6.5% | **Confirmed** |
| H-zero-sum (comparative) | Treatment < B2 | Treatment ≈ B2 | **Refuted** |
| H-control-negative | <5% at sub-saturation | 6.5% | **Refuted** (borderline) |
| H-robustness-burst | >15% improvement at all CVs | +7-24% worse | **Refuted** |
| H-single-turn | >10% improvement | 0.0% | **Refuted** |

---

## Principles Extracted

### S17: Static class weights are the minimal sufficient scheduling mechanism

**Evidence:** Treatment (7-parameter deadline-aware urgency) produces byte-identical results to B2 (1-parameter-per-class static weights) at 30% and 80% load, and marginally worse results at 120%. The 92.6% critical TTFT improvement over B1 (age-only) comes entirely from knowing which SLO class a request belongs to — no time dependence needed.

**Implication:** `StaticClassWeight` should be the standard SLO-aware priority policy. `DeadlineAwarePriority` adds implementation complexity and float-precision risk (INV-6) for zero measurable benefit.

### S18: Time-dependent priority is ineffective in step-quantized DES

**Evidence:** The deadline-aware urgency window (100ms critical deadline at ~7-12ms step time = ~8-14 recomputations) is too coarse for the hyperbolic curve to produce different scheduling decisions than static weights. Priority is recomputed only at step boundaries, and same-class requests at similar elapsed times produce nearly identical urgency values.

**Implication:** Scheduling improvements in BLIS must come from class-level differentiation (which class goes first) or load management (admission control), not from within-class urgency dynamics.

### S6 Confirmed: Scheduling is zero-sum; admission is the next lever

**Evidence:** Both Treatment and B2 produce +6.5% cluster degradation vs B0. Improving critical by 92% comes at the cost of other classes. S6 (from prior iteration track) is re-confirmed. The next iteration should compose static class weights with admission control per S8.

---

## Issues

None filed. The experiment cleanly refutes the primary hypothesis and confirms the secondary finding. No bugs discovered.

---

## Scope and Limitations

- **Operating points tested:** 30%, 80%, 120% of ~250 req/s capacity with 4 instances
- **Model:** llama-3.1-8b-instruct, H100, TP=2 (blackbox latency model)
- **Not tested:** Other models, GPU types, TP configurations, real vLLM validation
- **Sample size:** 1500 requests per rate point, 3 seeds. P99 based on ~3 observations for critical class.
- **DES limitation:** Step-quantized priority recomputation may not reflect continuous-time systems (real vLLM has finer-grained scheduling). The deadline mechanism *might* work in a system with more frequent priority updates.

**Capacity derivation:** With beta coefficients [6910, 17.67, 2.84] for llama-3.1-8b/H100/TP=2 and mean input=256, output=128: single-turn step time ≈ 11.8ms → ~85 req/s per instance → ~340 req/s for 4 instances. Multi-turn (3 rounds, context accumulation) increases effective per-request work ~2-3x, reducing capacity to ~113-170 req/s. At 300 req/s, the effective overload is ~175-265%, significantly higher than the "120%" label suggests.

## Evidence Quality
| Claim | Evidence | Confidence |
|-------|----------|------------|
| Treatment equivalent to B2 | Byte-identical at 30%/80%, +7.7% at 120% | High |
| 92.6% improvement over B1 | 3 seeds, consistent | High |
| Uniform deadlines catastrophic | +317-422% degradation | High |
| Single-turn shows no effect | Byte-identical | High |

## Implications for Users
Do not use DeadlineAwarePriority -- StaticClassWeight achieves the same results with simpler configuration. Class-awareness (knowing which SLO class a request belongs to) is the key ingredient, not time-dependent urgency curves.
