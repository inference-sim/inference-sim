# Findings: H-Joint-SLO-Optimization

**Status:** Pending — experiments not yet run
**Date started:** 2026-03-31

---

## Calibration

| | Value |
|---|---|
| Target model | |
| GPU type | |
| Measured saturation throughput | req/s |
| `aggregate_rate` set to | req/s (= 2 × saturation) |
| KV blocks (default) | |
| Block size (tokens) | |
| Warmup period discarded | first 10% of horizon |

---

## Iteration 0: Baseline Measurement

**Configuration:** `pa:4,qd:3` + `priority-fcfs` + `tier-shed` + `vllm` batch formation

### Raw Results

| Seed | Critical TTFT P99 (ms) | Standard Goodput | Sheddable Goodput | Preemption Count | KV Hit Rate |
|------|----------------------|-----------------|-------------------|-----------------|------------|
| 42 | | | | | |
| 123 | | | | | |
| 456 | | | | | |
| **Mean ± 1σ** | | | | | |

### Phase-Separated

| Seed | Critical P99 (sustained) | Critical P99 (burst) | Burst amplification ratio |
|------|------------------------|---------------------|--------------------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |

### Notes
<!-- Record observations about KV pressure, preemption frequency, etc. -->

---

## Iteration 1: Joint Composition Validation

**Configuration:** Same as Iter 0 (this IS the joint compound — Iter 0 establishes the
reference against which BLIS defaults would be compared)

**BLIS defaults run** (for H-main comparison): `round-robin` + `fcfs` + `always-admit`

### H-main Results

| Seed | Compound P99 | BLIS-default P99 | Improvement |
|------|-------------|-----------------|-------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |
| **Mean ± 1σ** | | | |

**Threshold:** > 40% improvement. **Confirmed:** [ ] Yes [ ] No

### H-ablation Results

| Arm | Mean P99 | vs. Compound | Threshold | Pass? |
|-----|----------|-------------|-----------|-------|
| abl-routing (round-robin) | | | > 15% degradation | |
| abl-scheduling (fcfs) | | | > 20% degradation | |
| abl-nochunk | | | > 10% degradation | |
| abl-admission (always-admit) | | | > 30% degradation | |

**Fast-failed components:** <!-- list any with < 5% contribution -->

### H-super-additivity (routing × admission)

| Routing alone Δ | Admission alone Δ | Sum | Compound Δ | Interaction term |
|----------------|-------------------|-----|------------|-----------------|
| | | | | |

**Threshold:** Interaction > 10%. **Confirmed:** [ ] Yes [ ] No

### H-control-negative (uniform SLO)

| Mean P99 (uniform-standard) | Mean P99 (BLIS default, uniform) | Improvement |
|----------------------------|--------------------------------|-------------|
| | | |

**Threshold:** < 5% improvement. **Confirmed:** [ ] Yes [ ] No

### Decision

[ ] **PROCEED** to Iter 2
[ ] **REVISE:** _______________________
[ ] **RESTART:** _______________________

**Rationale:**

---

## Iteration 2: SLO-Priority Preemption Ordering

**Configuration:** Iter 1 compound + `--batch-formation slo-priority-preemption`
**Ablation:** `--batch-formation vllm` (LIFO)

### H-main Results

| Seed | SLO-priority P99 | LIFO (ablation) P99 | Improvement |
|------|-----------------|---------------------|-------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |
| **Mean ± 1σ** | | | |

**Threshold:** > 15% improvement. **Confirmed:** [ ] Yes [ ] No

### H-zero-sum Results

| Seed | Standard Goodput (treatment) | Standard Goodput (Iter 1) | Degradation |
|------|----------------------------|--------------------------|-------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |

**Threshold:** Degradation ≤ 20%. **Confirmed:** [ ] Yes [ ] No

### Phase-Separated

| Seed | Improvement (sustained) | Improvement (burst) |
|------|------------------------|---------------------|
| 42 | | |
| 123 | | |
| 456 | | |

### H-control-negative (abundant KV)

| KV blocks | Treatment P99 | Ablation P99 | Improvement |
|-----------|-------------|-------------|-------------|
| abundant | | | |

**Threshold:** < 3% improvement. **Confirmed:** [ ] Yes [ ] No

### Decision

[ ] **PROCEED** to Iter 3
[ ] **FAST-FAIL** (contribution < 5% — drop mechanism)
[ ] **REVISE:** _______________________

---

## Iteration 3: SLO-Aware Tiered KV Prefix Cache Eviction

**Configuration:** Iter 2 compound (or Iter 1 if Iter 2 fast-failed) with tiered-LRU build
**Ablation:** Pre-PR #901 build (single-list LRU) with same CLI flags

### H-main Results

| Seed | Tiered-LRU P99 | Single-list P99 | Improvement |
|------|---------------|-----------------|-------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |
| **Mean ± 1σ** | | | |

**Threshold:** > 15% improvement. **Confirmed:** [ ] Yes [ ] No

### Cache Hit Rate

| Seed | Hit rate (tiered-LRU) | Hit rate (single-list) | Improvement |
|------|----------------------|----------------------|-------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |

**Threshold:** > 10% hit rate improvement. **Confirmed:** [ ] Yes [ ] No

### H-super-additivity (Iter 2 × Iter 3)

| Iter 2 alone Δ | Iter 3 alone Δ | Sum | Compound Δ | Interaction |
|---------------|----------------|-----|------------|-------------|
| | | | | |

**Threshold:** Interaction > 5%. **Confirmed:** [ ] Yes [ ] No

### Decision

[ ] **PROCEED** to Iter 4
[ ] **FAST-FAIL**
[ ] **REVISE:** _______________________

---

## Iteration 4: Admission-Feedback Batch Formation

**Configuration:** Iter 3 compound + `--batch-formation tier-budget --tier-budget-critical-frac 0.50`
**Ablation:** `--tier-budget-critical-frac 0.333` (equal-share)

### H-main Results

| Seed | f_c=0.50 P99 | Iter 3 compound P99 | Improvement |
|------|-------------|---------------------|-------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |
| **Mean ± 1σ** | | | |

**Threshold:** > 10% improvement. **Confirmed:** [ ] Yes [ ] No

### H-ablation (fraction sensitivity)

| Seed | f_c=0.50 P99 | f_c=0.333 P99 | Degradation |
|------|-------------|--------------|-------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |

**Threshold:** > 8% degradation from equal-share. **Confirmed:** [ ] Yes [ ] No

### H-robustness (fraction sweep)

| f_c | Mean P99 | Standard Goodput | Monotone? |
|-----|----------|-----------------|-----------|
| 0.20 | | | |
| 0.30 | | | |
| 0.40 | | | |
| 0.50 | | | |
| 0.60 | | | |
| 0.70 | | | |

Knee location: _____ Goodput floor violated at f_c: _____

### Phase-Separated

| Seed | Improvement (burst) | Improvement (sustained) | Burst > sustained? |
|------|---------------------|------------------------|-------------------|
| 42 | | | |
| 123 | | | |
| 456 | | | |

### Decision

[ ] **PROCEED** to Bayesian optimization
[ ] **FAST-FAIL**
[ ] **REVISE:** _______________________

---

## Bayesian Optimization Results

**Parameters optimized:** `w_pa`, `w_qd`, `overloadThreshold`, `minAdmitPriority`, `f_c`,
critical prefill threshold

### Top 5 Feasible Points (Pareto: critical P99 vs. standard goodput)

| Rank | `w_pa` | `w_qd` | `overload` | `minAdmit` | `f_c` | prefill thresh | Critical P99 | Std Goodput |
|------|--------|--------|------------|------------|-------|----------------|-------------|------------|
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |
| 4 | | | | | | | | |
| 5 | | | | | | | | |

**Recommended operating point:** Rank __ (highest standard goodput within top critical P99 cluster)

---

## Summary and Principles Extracted

### Overall improvement (confirmed compound vs. BLIS defaults)

| Metric | BLIS defaults | Confirmed compound | Improvement |
|--------|-------------|-------------------|-------------|
| Critical TTFT P99 | | | |
| Standard goodput | | | |
| Sheddable goodput | | | |

### New principles (to add to principles catalog)

<!-- After experiments, record new RP-N or S-N principles here -->

### Refuted predictions

<!-- List any H-main predictions that were refuted, with diagnostic conclusions -->

### Open questions for future work

<!-- List any unexpected findings that warrant further investigation -->
