# HYPOTHESIS: Multi-Objective Constrained CMA-ES with ITL Penalty

**Round:** 4
**Idea:** 1 — Constrained CMA-ES
**Date:** 2026-03-02
**Status:** Pending

## Context and Prior Findings

Round 3's CMA-ES (Idea 3 H1) achieved 15.1% mean E2E with 4/10 experiments below 10% — the best E2E result in the research program. However, it sacrificed ITL accuracy: 87.4% mean ITL (was 9.5% in the trace replay baseline). The root cause is the unconstrained single-objective formulation — CMA-ES inflated `output_token_processing_time_us` to implausible values (e.g., 7B: 0 → 1,899μs) as a proxy for missing simulation dynamics.

This idea adds three modifications to the R3 CMA-ES approach:
1. **ITL penalty term** in the objective function
2. **Parameter bounds** constraining physically implausible values
3. **Integrated TTFT corrections** (from R3 H3) applied during optimization rather than post-hoc

### Related Work

- Deb et al., 2002 — NSGA-II for multi-objective evolutionary optimization
- Igel et al., 2007 — MO-CMA-ES for multi-objective optimization with CMA-ES
- R3 Idea 3 — unconstrained CMA-ES achieving 15.1% E2E / 87.4% ITL

## Sub-Hypotheses

### H1: Constrained CMA-ES with ITL Penalty → Joint E2E + ITL

**Claim:** A penalty-augmented CMA-ES objective (`objective = 0.5*e2e + 0.5*itl + penalty`) with constrained parameter bounds achieves **mean E2E < 15% AND mean ITL < 25%** simultaneously across 10 experiments.

**Refutation criteria:** Mean E2E > 18% (worse than R3 by >3pp) OR mean ITL > 30%.

**Method:**
1. Start from R3's optimized artifacts as initial CMA-ES population
2. Add parameter bounds: `output_token_processing_time_us ∈ [0, 500]`, `scheduling_processing_time_us ∈ [0, 2000]`, `step_overhead_us ∈ [model_baseline × 0.8, model_baseline × 3.0]`
3. Dual-metric objective: `f(x) = 0.5 * mean_e2e_error + 0.5 * mean_itl_error + 10.0 * max(0, mean_itl_error - 0.20)`
4. Apply TTFT additive corrections (per-model, from R3 H3) inside the objective evaluation loop
5. Run per-model optimization with CMA-ES (σ₀ = 0.1, ~100 evaluations per model)
6. Run with three α values (0.7, 0.5, 0.3 on the E2E weight) to explore the Pareto frontier

**Data split:** No training/test split for CMA-ES — it optimizes against BLIS E2E on all 10 experiments simultaneously. Generalization tested via LOMO/LOWO.

### H2: Pareto Sweep Identifies Knee Point

**Claim:** Running CMA-ES with different E2E/ITL weight ratios produces a Pareto frontier where at least one configuration achieves **E2E < 12% AND ITL < 20%** — a result that R3's single-objective could not achieve.

**Refutation criteria:** No Pareto point achieves both E2E < 15% and ITL < 25%.

**Method:**
1. Run H1 with α ∈ {0.7, 0.5, 0.3} (weighting E2E vs ITL)
2. Collect per-experiment E2E and ITL errors for each α
3. Plot Pareto frontier: E2E vs ITL
4. Select knee point (closest to the origin) as the recommended configuration

### H3: LOMO Generalization (Cross-Model Artifact Transfer)

**Claim:** Constrained CMA-ES artifacts transfer across models with **mean best-donor E2E < 20%** (vs R3's 14.8% with unconstrained CMA-ES). Constraining the parameter space should maintain cross-model transferability while improving ITL.

**Refutation criteria:** Mean best-donor E2E > 30% (2× worse than R3 LOMO).

**Method:**
1. For each model group, apply the CMA-ES-optimized artifacts from other models
2. Evaluate BLIS E2E and ITL per experiment
3. Build cross-model transfer matrix (same as R3 H5)
4. Use `hypotheses/h-stepml/shared/splits.py:leave_one_model_out()` for data splitting

### H4: LOWO Generalization (Cross-Workload Stability)

**Claim:** Constrained CMA-ES artifacts show per-model workload variance **≤ 20pp** (range across workloads) for dense models, with **mean E2E within 2× of aggregate** for all 10 experiments.

**Refutation criteria:** More than 2/10 experiments exceed 2× aggregate E2E error.

**Method:**
1. Evaluate per-experiment E2E/ITL from H1's best configuration
2. Group by model, compute per-workload breakdown
3. Measure range (max-min) per model
4. Use `hypotheses/h-stepml/shared/splits.py:leave_one_workload_out()` for data splitting

## Execution Plan

1. H1 first — constrained CMA-ES per model with dual objective
2. H2 — Pareto sweep (runs H1 with different α values)
3. H3 — LOMO using H1's best artifacts
4. H4 — LOWO from H1's results

## Short-Circuit Criteria

If H1 produces mean E2E > 25% (significantly worse than R3's 15.1%), the constrained approach is harming more than helping — abort remaining hypotheses and document root cause.
