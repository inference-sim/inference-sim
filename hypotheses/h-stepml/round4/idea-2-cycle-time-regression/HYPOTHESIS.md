# HYPOTHESIS: Data-Driven Step Cycle Time Model with Explicit CPU Overhead Separation

**Round:** 4
**Idea:** 2 — Cycle-Time Regression
**Date:** 2026-03-02
**Status:** Pending

## Context and Prior Findings

All prior rounds modeled `step.duration_us` — the GPU forward pass time (~70–7,000μs) — and added overhead as a post-hoc floor via `max(overhead, compute)`. This creates the "faster universe" problem (BC-NEW-2): BLIS completes requests in ~40% of real time because the overhead floor is calibrated from ITL residuals, not from direct measurement.

The real step cycle time (what matters for E2E/ITL) is the inter-token interval (ITI) between consecutive output tokens, which includes:
- GPU forward pass (`step.duration_us`)
- CPU scheduling overhead (queue scanning, KV allocation, priority sorting)
- CUDA synchronization time
- Memory management (block allocation, eviction decisions)

This idea trains the step-time model on **actual ITI from lifecycle data** instead of `step.duration_us`, eliminating the need for a separate overhead floor.

### Related Work

- FairBatching [Patel et al., 2025] — `batch_time = a + b*new_tokens + c*total_context`, ±1.3% with continuous calibration
- AIConfigurator [2026] — 7.8% TPOT MAPE via operator-level decomposition + summing
- Vidur [Agrawal et al., 2024] — operator-level RF models, <5% P95 error, random forest for attention ops
- R2 discovery: overhead floor (`max(overhead, compute)`) handles 77.9% of steps
- R3 discovery: per-step MAPE improvements have zero impact on BLIS E2E due to overhead floor dominance

## Sub-Hypotheses

### H1: Cycle-Time Extraction from Lifecycle Data

**Claim:** Per-step cycle times can be reliably extracted from lifecycle per-token timestamps, with Pearson r > 0.7 between extracted cycle times and corresponding `step.duration_us` values for the compute-dominated regime (large batches).

**Refutation criteria:** Pearson r < 0.3 (no meaningful correlation) or >50% of steps cannot be matched to lifecycle timestamps.

**Method:**
1. Load per-request lifecycle data (per_request_lifecycle_metrics.json) for each experiment
2. For each request, compute ITI = timestamp[i+1] - timestamp[i] for consecutive output tokens
3. Join ITI values with step-level data by timestamp matching (within step window)
4. Compute cycle_time per step as mean or median ITI across requests in that step
5. Compare cycle_time vs step.duration_us: the ratio (cycle_time / step.duration_us) should be >1 for small batches (overhead-dominated) and ≈1 for large batches (compute-dominated)
6. Report distribution of cycle_time / step.duration_us across regimes

### H2: FairBatching Cycle-Time Regression → BLIS E2E

**Claim:** A per-model FairBatching-style regression on cycle time (`cycle_time = a + b*new_tokens + c*kv_sum`) achieves **mean BLIS E2E < 25%** with **mean ITL < 20%** across 10 experiments, without any separate overhead floor.

**Refutation criteria:** Mean BLIS E2E > 40% (worse than R3 trace replay baseline of 56.2%) OR mean ITL > 25%.

**Method:**
1. Use H1's extracted cycle times as the training target
2. Train per-model OLS: `cycle_time = a + b*new_tokens + c*kv_sum` (FairBatching formulation)
3. Also train regime-separated models: decode-only (prefill=0) and mixed-batch (prefill>0)
4. Export coefficients to StepML JSON artifact:
   - `step_overhead_us` = regression intercept for decode-only regime (data-derived)
   - `output_token_processing_time_us` = 0 (absorbed into cycle time)
   - TTFT correction from R3 H3 in QueueingTime
5. Run BLIS validation via `validate_blis.py` with trace replay
6. Report per-experiment E2E, TTFT, ITL errors

**Data split:** Temporal split (60/20/20) for per-step regression training/validation. BLIS E2E evaluated on all experiments (trace replay mode).

### H3: LOMO Generalization (Leave-One-Model-Out)

**Claim:** Cycle-time regression achieves **LOMO per-step MAPE < 80%** per fold (train on 3 models, predict 4th).

**Refutation criteria:** Any fold MAPE > 150% (worse than R2's 108.6% regime ensemble LOMO).

**Method:**
1. Use `hypotheses/h-stepml/shared/splits.py:leave_one_model_out()` for 4-fold splitting
2. For each fold: train per-model cycle-time regression on 3 models, predict held-out 4th
3. Report per-fold MAPE and aggregate LOMO MAPE
4. Analyze which model is hardest to predict (expected: Mixtral due to MoE architecture)

### H4: LOWO Generalization (Leave-One-Workload-Out)

**Claim:** Cycle-time regression achieves **LOWO per-step MAPE < 50%** per fold (train on 2 workloads, predict 3rd).

**Refutation criteria:** Any fold MAPE > 100%.

**Method:**
1. Use `hypotheses/h-stepml/shared/splits.py:leave_one_workload_out()` for 3-fold splitting
2. For each fold: train per-model cycle-time regression on 2 workloads, predict held-out 3rd
3. Report per-fold MAPE and aggregate LOWO MAPE
4. Analyze whether general/codegen/roleplay workloads have different cycle-time characteristics

## Execution Plan

1. H1 first — extract cycle times (prerequisite for all other hypotheses)
2. H2 — train and validate cycle-time regression
3. H3 + H4 — LOMO/LOWO generalization (can run in parallel after H2)

## Short-Circuit Criteria

If H1 fails (cannot reliably extract cycle times from lifecycle data, <50% join rate), the entire idea is infeasible — abort and document the data quality gap. If H2 produces mean E2E > 56.2% (worse than the R3 trace replay baseline without CMA-ES), the cycle-time model adds no value — abort H3/H4.
