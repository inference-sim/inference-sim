# HYPOTHESIS: Two-Stage Calibration — Principled Cycle-Time Base + Constrained CMA-ES Residual

**Round:** 4
**Idea:** 3 — Hybrid Calibration
**Date:** 2026-03-02
**Status:** Partially Confirmed (H1 record 5.7% E2E; H2 refuted; H3 refuted; H4 confirmed)

## Context and Prior Findings

Round 3 demonstrated two complementary results:
1. **CMA-ES** achieves 15.1% mean E2E but 87.4% ITL (E2E-focused, ITL-blind)
2. **Trace replay baseline** achieves 56.2% mean E2E but 9.5% ITL (correctly calibrated per-step but poor E2E)

The gap exists because CMA-ES compensates for a poorly calibrated base model by distorting secondary coefficients (`output_token_processing_time_us` inflated to 1,899μs). If the base model already produces correct step cycle times (via Idea 2's cycle-time regression), CMA-ES only needs small residual adjustments — the E2E ↔ ITL tradeoff softens because there's less to compensate for.

This idea chains Idea 2 (principled base) → Idea 1 (constrained CMA-ES) with tight bounds (±20% of base values).

### Related Work

- Kennedy & O'Hagan, 2001 — Bayesian calibration with discrepancy term (physics model + statistical residual)
- Revati [2025] — extends Vidur's analytical models with profiling, <5% error
- R3 Idea 3 — unconstrained CMA-ES achieving 15.1% E2E / 87.4% ITL
- R3 Finding: CMA-ES cross-model transfer achieves 14.8% LOMO E2E — transferable dynamics

## Sub-Hypotheses

### H1: Stage 1 Principled Base (Cycle-Time Regression + TTFT Corrections)

**Claim:** Cycle-time regression (from Idea 2) combined with TTFT additive corrections produces a base model with **mean E2E < 30%** and **mean ITL < 20%** across 10 experiments.

**Refutation criteria:** Mean E2E > 50% (worse than R3 trace replay baseline) OR mean ITL > 30%.

**Method:**
1. Use Idea 2's cycle-time extraction (H1) and regression (H2) artifacts
2. Apply per-model TTFT additive corrections from R3 H3 (16–61ms) to QueueingTime
3. Set `output_token_processing_time_us = 0` (absorbed into cycle time model)
4. Run BLIS validation via trace replay
5. Report per-experiment E2E, TTFT, ITL errors
6. This is the "principled base" — the starting point for Stage 2

**Note:** If Idea 2 H2 already reports BLIS E2E results, H1 here simply adds TTFT corrections. The key test is whether TTFT corrections + cycle-time base together produce a balanced model.

**Data split:** Same as Idea 2 H2 (temporal split for regression). BLIS E2E on all experiments.

### H2: Stage 2 Constrained CMA-ES Residual Tuning

**Claim:** Constrained CMA-ES (±20% bounds around Stage 1 values, dual-objective E2E+ITL) improves the principled base to **mean E2E < 12%** and **mean ITL < 18%**, with at least **6/10 experiments below 10% E2E**.

**Refutation criteria:** Mean E2E > Stage 1 E2E + 5pp (CMA-ES makes base model worse) OR mean ITL > 25%.

**Method:**
1. Start from H1's principled base coefficients as CMA-ES initial vector
2. **Tight bounds:**
   - `step_overhead_us ∈ [base × 0.85, base × 1.15]`
   - `output_token_processing_time_us ∈ [0, 200]`
   - `scheduling_processing_time_us ∈ [0, 1000]`
   - Step-time regime coefficients ∈ [base × 0.8, base × 1.2]
3. **Dual objective:** `f(x) = 0.5 * mean_e2e + 0.5 * mean_itl + 5.0 * max(0, itl - 0.20)`
4. Per-model optimization, σ₀ = 0.05 (smaller than Idea 1's 0.1 — tighter search)
5. Max 80 evaluations per model (vs 96-152 in R3 — starting closer to optimum)
6. **Safety net:** If any model's ITL worsens by >5pp vs Stage 1, discard CMA-ES for that model and use Stage 1 directly

### H3: LOMO Generalization (Cross-Model Transfer)

**Claim:** The hybrid approach achieves **mean best-donor E2E < 20%** in LOMO cross-model transfer, with **mean best-donor ITL < 30%** (R3 did not report LOMO ITL).

**Refutation criteria:** Mean best-donor E2E > 30% (2× worse than R3's 14.8%).

**Method:**
1. For Stage 1 LOMO: Train cycle-time regression on 3 models, evaluate BLIS E2E/ITL on held-out 4th
2. For Stage 2 LOMO: Apply CMA-ES artifacts (optimized on 3 models) to held-out 4th
3. Build cross-model transfer matrix for both E2E and ITL (extends R3 H5)
4. Use `hypotheses/h-stepml/shared/splits.py:leave_one_model_out()` for splitting
5. Report which stage contributes most to generalization: principled base (Stage 1) or CMA-ES residuals (Stage 2)

### H4: LOWO Generalization (Cross-Workload Stability)

**Claim:** The hybrid approach shows per-model workload variance **≤ 20pp** for dense models and **mean E2E within 2× of aggregate** for all 10 experiments.

**Refutation criteria:** More than 3/10 experiments exceed 2× aggregate E2E.

**Method:**
1. Evaluate per-experiment E2E/ITL from H2's best configuration
2. Group by model, compute per-workload breakdown
3. Measure range (max-min) per model
4. Compare against R3 LOWO results (8/10 within 2×)
5. Use `hypotheses/h-stepml/shared/splits.py:leave_one_workload_out()` for data splitting

## Execution Plan

1. **Depends on Idea 2 H1+H2:** Stage 1 uses Idea 2's cycle-time regression artifacts. If Idea 2 is executed first, Stage 1 reuses those artifacts; if not, Stage 1 re-derives them.
2. H1 — evaluate Stage 1 (principled base + TTFT corrections)
3. H2 — run Stage 2 CMA-ES on top of Stage 1
4. H3 + H4 — LOMO/LOWO generalization (can run in parallel after H2)

## Short-Circuit Criteria

- If Idea 2 H1 fails (cannot extract cycle times), this idea is blocked. Fall back to using R3's regime ensemble as the Stage 1 base.
- If H1 (Stage 1) produces mean E2E > 50% AND mean ITL > 25%, the principled base is not better than R3's trace replay — skip Stage 2 and document why.
- If H2 (Stage 2) produces E2E > 18% AND ITL > 25% (worse than both Idea 1 and R3 on both metrics), the hybrid approach is not synergistic — abort H3/H4.

## Expected Outcome

This idea represents the highest probability of achieving the research target: **E2E < 10% AND ITL < 15%**. By combining a correctly calibrated base (modeling what is measured) with constrained optimization (small residual corrections), we avoid the catastrophic E2E ↔ ITL tradeoff that plagued R3.
