# StepML Round 4: Research Ideas

**Date:** 2026-03-02
**Branch:** `stepml-experiments`
**Primary Binding Constraint:** BC-NEW-1 — E2E ↔ ITL fundamental tradeoff (CMA-ES achieves 15.1% E2E but 87.4% ITL)
**Secondary Constraints:** BC-NEW-2 (step cycle under-prediction), BC-6 (vLLM args sensitivity never analyzed)
**Number to Beat:** 15.1% mean E2E (R3 CMA-ES), 9.5% mean ITL (R3 trace replay baseline)
**Target:** E2E < 10% AND ITL < 15% simultaneously

---

## Idea 1: Multi-Objective Constrained CMA-ES with ITL Penalty

### Title
Multi-Objective CMA-ES with Explicit ITL Constraint for Joint E2E + ITL Optimization

### Rationale

**Directly addresses BC-NEW-1 (E2E ↔ ITL tradeoff).** Round 3 showed CMA-ES is the only approach to achieve <16% E2E (15.1%), but it sacrifices ITL (87.4%) by inflating `output_token_processing_time_us` to implausible values (e.g., 0 → 1,899μs for 7B). The root cause is the single-objective formulation: CMA-ES minimizes E2E without knowledge of ITL consequences.

**Literature grounding:** Multi-objective evolutionary optimization (NSGA-II, MO-CMA-ES) is well-established for Pareto frontier discovery in noisy simulation tuning [Deb et al., 2002; Igel et al., 2007]. FairBatching [Patel et al., 2025] demonstrates that step-time accuracy ±1.3% is achievable with `a + b*new_tokens + c*total_context`, suggesting that ITL accuracy ≤15% is feasible if the optimizer is constrained.

**Codebase grounding:** The existing R3 CMA-ES infrastructure (`round3/idea-3-e2e-calibration/run_experiment.py`) already supports per-model optimization with trace replay. The change is to the objective function: instead of `f(params) = mean_e2e_error(experiments)`, use `f(params) = mean_e2e_error + λ * max(0, mean_itl_error - 0.15)` where λ is a penalty weight. This preserves the entire CMA-ES pipeline while adding ITL awareness.

**Why it differs from R3 Idea 3:** R3 used unconstrained single-objective CMA-ES. This idea adds an explicit ITL penalty term and constrains `output_token_processing_time_us` to physically plausible ranges (0–500μs based on real vLLM measurements). The constrained search space prevents the optimizer from using OTPT as a proxy for missing simulation dynamics.

### Method Sketch

1. **Constrained parameter space:** Bound `output_token_processing_time_us` ∈ [0, 500], `scheduling_processing_time_us` ∈ [0, 2000], `step_overhead_us` ∈ [model_baseline × 0.8, model_baseline × 3.0]. This prevents physically implausible solutions.
2. **Penalty-augmented objective:** `objective = α * mean_e2e_error + (1-α) * mean_itl_error + λ * constraint_violation`, where constraint violations include ITL > 20% or OTPT > 500μs.
3. **TTFT additive corrections (from R3 H3):** Per-model TTFT corrections (16–61ms) are applied *within the objective evaluation*, not post-hoc. CMA-ES sees the corrected E2E and optimizes around it, avoiding the R3 H3 double-counting issue.
4. **Per-model optimization:** Same as R3 — separate CMA-ES run per model group.
5. **Pareto sweep:** Run with multiple α values (0.7, 0.5, 0.3) to explore the Pareto frontier, then select the knee point.

### Expected Outcome

E2E < 12% mean with ITL < 20% mean, with at least 6/10 experiments below 10% E2E. The constraint prevents the catastrophic ITL degradation seen in R3.

### Go Integration Path

Coefficient export — same StepML JSON artifact format as R3. No new Go code needed.

### LatencyModel Methods Targeted

All 5 methods via CMA-ES parameter tuning: StepTime (regime coefficients + overhead floor), QueueingTime (TTFT corrections), OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime.

### Generalization Plans

#### LOMO (Leave-One-Model-Out)
Same protocol as R3 H5: train CMA-ES on 3 models, apply artifact to held-out 4th. R3 showed 14.8% LOMO E2E — the constrained version should maintain or improve this since the constraints prevent overfitting to simulation artifacts. Model features enabling transfer: overhead floor values scale with model size/TP, regime coefficients capture batch-composition physics.

#### LOWO (Leave-One-Workload-Out)
R3 showed 8/10 within 2× aggregate for LOWO. Re-run with constrained CMA-ES and verify workload variance doesn't increase. Dense models (70B, 34B) showed 9-16pp range — target ≤20pp with constrained optimization.

#### vLLM Args Sensitivity Analysis

| Parameter | Dependent? | Would predictions break? | Recalibration needed |
|---|---|---|---|
| `max_num_seqs` (128) | **Yes** — controls batch size distribution → affects overhead floor regime | Overhead floor would be wrong for different batch size regime | Re-optimize `step_overhead_us` only (~5 min CMA-ES) |
| `max_num_batched_tokens` (2048) | **Yes** — controls chunked prefill budget → changes mixed-batch frequency | Regime boundaries (prefill=0 threshold) might shift | Re-optimize regime coefficients (~15 min CMA-ES) |
| `max_model_len` (4096) | **Weak** — changes KV capacity → may shift batch sizes under pressure | Predictions stable unless KV pressure forces evictions | None for moderate changes; re-optimize for >2× change |
| `chunked_prefill` (on) | **Strong** — fundamentally changes batch composition | Model assumes mixed batches; pure-prefill regime doesn't exist in training data | Full retrain required |
| `prefix_caching` (on) | **Moderate** — changes effective prefill tokens | Prefill regime coefficients would over-predict | Re-optimize prefill regime coefficients only |
| `gpu_memory_utilization` (0.9) | **Weak** — changes KV capacity | Same as max_model_len: only matters under extreme memory pressure | None for ≥0.7; re-optimize for <0.5 |
| `tensor_parallel_size` (varies) | **Strong** — changes per-GPU compute/memory | Overhead floor and all coefficients change | Full retrain for new TP degree |

**Recalibration story:** CMA-ES re-optimization for a single parameter change takes ~5-15 minutes (50-100 evaluations). For `chunked_prefill` or `TP` changes, full retrain (~30 min/model) is required. This is acceptable for production: new serving configs are deployed infrequently and calibration is automated.

---

## Idea 2: Principled Overhead Floor Calibration via Cycle-Time Regression

### Title
Data-Driven Step Cycle Time Model with Explicit CPU Overhead Separation

### Rationale

**Directly addresses BC-NEW-2 (step cycle under-prediction) and the root cause of BC-NEW-1.** The "faster universe" problem (BLIS at ~40% real speed) exists because:
1. `step.duration_us` in training data captures GPU forward pass only (~70–7,000μs)
2. Real step cycle time = GPU forward pass + CPU overhead (scheduling, sync, memory management)
3. The overhead floor (`max(overhead, compute)`) is a heuristic — its value was hand-tuned from ITL residuals, not derived from the data itself

**Literature grounding:** AIConfigurator [2026] achieves 7.8% TPOT MAPE by decomposing iteration time into profiled operator primitives and summing them. Vidur [Agrawal et al., 2024] achieves <5% P95 error with operator-level RF models. FairBatching [Patel et al., 2025] achieves ±1.3% with continuous calibration of `a + b*new_tokens + c*total_context`. The common thread: **model what you measure, then add overhead separately**.

**Codebase grounding:** Examining `sim/latency/stepml.go:232-268`, the overhead is applied as `max(overhead, compute)` with a hard cap at 3×overhead. The overhead constant comes from R2's manual calibration (ground-truth ITL - mean compute). Instead, we should derive the overhead from the *ground-truth step cycle times* directly.

**Key insight:** We have per-request lifecycle data with per-token timestamps. The inter-token interval (ITI) between consecutive output tokens of the same request represents the actual step cycle time that request experienced. By regressing ITI against batch composition features, we can build a **cycle time model** that inherently includes CPU overhead — no separate overhead floor needed.

**Why it differs from prior rounds:** R1-R3 all modeled `step.duration_us` (GPU forward pass) and added overhead as a post-hoc floor. This idea models the **actual cycle time** (ITI from lifecycle data), which naturally includes all overhead. The overhead floor becomes unnecessary if the cycle time model is accurate.

### Method Sketch

1. **Extract per-step cycle times from lifecycle data:** For each request's output tokens, compute ITI = timestamp[i+1] - timestamp[i]. These represent real step cycle times.
2. **Join cycle times with step-level batch features:** Match lifecycle timestamps to step windows to assign batch composition features (prefill_tokens, decode_tokens, kv_sum) to each cycle time.
3. **Train per-model FairBatching-style regression on cycle times:** `cycle_time = a + b*new_tokens + c*kv_sum` where the target is ITI (real cycle time), not step.duration_us (GPU-only). The intercept `a` naturally absorbs CPU overhead.
4. **Regime separation:** Decode-only (prefill=0) gets its own model where `a_decode` represents the full cycle overhead; mixed batches get `a_mixed + b*prefill + c*decode + d*kv_sum`.
5. **BLIS integration:** Export coefficients to StepML artifact. The overhead floor is set to `a_decode` (the regression intercept for decode-only steps), which is data-derived rather than hand-tuned. No cap needed since the model already predicts cycle time, not compute time.
6. **Combine with TTFT additive corrections:** Apply per-model TTFT corrections from R3 H3 (16–61ms) directly in the artifact's QueueingTime model.

### Expected Outcome

By modeling the *right target variable* (cycle time vs GPU-only time), the "faster universe" problem should largely disappear. Target: E2E < 15% with ITL < 15% (both without CMA-ES). If successful, CMA-ES can be applied on top for further E2E refinement without the E2E ↔ ITL tradeoff (since the baseline is already correctly calibrated).

### Go Integration Path

Coefficient export — same StepML JSON artifact format. The key difference is the overhead floor value changes (data-derived vs hand-tuned), and the regression targets cycle time instead of compute time.

### LatencyModel Methods Targeted

StepTime (primary — cycle time regression), QueueingTime (TTFT corrections). OutputTokenProcessingTime set to 0 (absorbed into cycle time model). SchedulingProcessingTime and PreemptionProcessingTime remain 0 (absorbed into intercept).

### Generalization Plans

#### LOMO (Leave-One-Model-Out)
4-fold LOMO on cycle time regression. Since the intercept `a` absorbs per-model CPU overhead, cross-model transfer depends on whether overhead scales predictably with model size/TP. Expected: intercept scales roughly with model parameter count (more parameters → more memory management overhead). Target: LOMO per-step MAPE < 80%.

#### LOWO (Leave-One-Workload-Out)
3-fold LOWO. The FairBatching formulation (`new_tokens + kv_sum`) captures workload-invariant compute physics — workload differences affect only the *distribution* of these features, not the *relationship* between features and cycle time. Expected: LOWO per-step MAPE < 50%.

#### vLLM Args Sensitivity Analysis

| Parameter | Dependent? | Would predictions break? | Recalibration needed |
|---|---|---|---|
| `max_num_seqs` (128) | **Moderate** — affects batch size distribution but regression captures the relationship | Predictions shift within trained range | None if batch sizes stay within training range; retrain for >2× change |
| `max_num_batched_tokens` (2048) | **Moderate** — changes mixed-batch frequency | Regime proportions shift, but per-regime models are independent | None for moderate changes |
| `max_model_len` (4096) | **Weak** — KV capacity affects eviction, not step time | No impact on cycle time model | None |
| `chunked_prefill` (on) | **Strong** — model trained on chunked-prefill data exclusively | Without chunked prefill, pure-prefill steps would appear; model has no data for these | Full retrain (need data with chunked_prefill=off) |
| `prefix_caching` (on) | **Moderate** — affects effective prefill tokens | Prefill coefficient might over-predict; but cycle time intercept still anchors predictions | Retrain prefill regime coefficients |
| `gpu_memory_utilization` (0.9) | **Weak** — no direct effect on cycle time per step | No impact | None |
| `tensor_parallel_size` (varies) | **Strong** — changes GPU communication overhead → changes intercept | Cycle time intercept (CPU overhead) changes with TP | Full retrain for new TP |

**Recalibration story:** The formulation is structurally robust because it models *measured cycle time*, not derived quantities. For TP or chunked_prefill changes, collect ~1000 steps of profiling data and retrain (OLS, <1 second). The key advantage over CMA-ES: recalibration requires only step-level data, not full BLIS simulation runs.

---

## Idea 3: Hybrid Cycle-Time CMA-ES — Principled Base + Black-Box Residual

### Title
Two-Stage Calibration: Data-Derived Cycle-Time Base + Constrained CMA-ES Residual Tuning

### Rationale

**Combines the strengths of Ideas 1 and 2 to address all binding constraints simultaneously.** The E2E ↔ ITL tradeoff (BC-NEW-1) arises because CMA-ES compensates for a poorly calibrated base model by distorting secondary coefficients. If the base model already runs at the correct speed (Idea 2 fixes BC-NEW-2), CMA-ES only needs to make *small residual adjustments* rather than large compensatory distortions.

**Literature grounding:** Hybrid approaches combining physics-informed base models with learned residual corrections are standard in scientific simulation calibration [Kennedy & O'Hagan, 2001]. Revati [2025] uses Vidur's analytical models enhanced with profiling for <5% error. The principle: **start with a principled model, then calibrate residuals**.

**Key insight from R3:** CMA-ES inflated `output_token_processing_time_us` to 1,899μs (llama-2-7b) because the base model under-predicted step cycle time by ~2×. If the base model is already correct (via cycle-time regression from Idea 2), CMA-ES's search space shrinks dramatically and the E2E ↔ ITL tradeoff softens.

**Why it differs from prior approaches:** R3 Idea 3 applied CMA-ES to a poorly calibrated base (overhead floor from R2 hand-tuning). This idea first establishes a principled cycle-time base (Idea 2), then applies *tightly constrained* CMA-ES to fine-tune residuals. The two-stage approach means CMA-ES corrections are ≤20% of base values rather than ≥100%.

### Method Sketch

1. **Stage 1 — Principled base (from Idea 2):**
   - Train per-model cycle-time regression on ITI (inter-token interval) from lifecycle data
   - Derive overhead floor from regression intercept
   - Apply TTFT additive corrections from R3 H3
   - Evaluate BLIS E2E and ITL → this is the "principled base" result

2. **Stage 2 — Constrained CMA-ES residual tuning:**
   - Starting point: Idea 2's principled coefficients
   - **Tight bounds:** Each parameter bounded within ±20% of Stage 1 value (not the ±200% that R3 effectively allowed)
   - **Dual-objective penalty:** `objective = 0.5 * mean_e2e_error + 0.5 * mean_itl_error + λ * constraint_violations`
   - **Hard constraints:** `output_token_processing_time_us ∈ [0, 200]`, `step_overhead_us ∈ [base ± 15%]`
   - **Fewer evaluations:** Since the starting point is already good, CMA-ES converges faster (~50-80 evaluations vs 96-152 in R3)

3. **Fallback logic:** If CMA-ES Stage 2 worsens ITL by >5pp relative to Stage 1, discard CMA-ES results and use Stage 1 directly. This provides a safety net.

### Expected Outcome

Stage 1 alone: E2E ~20-25% with ITL ~12-15% (correctly calibrated base).
After Stage 2: E2E < 10% with ITL < 15% for ≥7/10 experiments. The tight bounds prevent CMA-ES from distorting the principled base.

### Go Integration Path

Coefficient export — same StepML JSON artifact format. Stage 2 produces the final artifact.

### LatencyModel Methods Targeted

All 5 methods:
- StepTime: Cycle-time regression (Stage 1) + residual tuning (Stage 2)
- QueueingTime: TTFT additive corrections (Stage 1) + fine-tuning (Stage 2)
- OutputTokenProcessingTime: Data-derived (Stage 1, ≤200μs) + bounded tuning (Stage 2)
- SchedulingProcessingTime: Small constant (Stage 1) + bounded tuning (Stage 2)
- PreemptionProcessingTime: Remains 0 (negligible in training data)

### Generalization Plans

#### LOMO (Leave-One-Model-Out)
Two-level LOMO test:
- **Stage 1 LOMO:** Train cycle-time regression on 3 models, predict 4th. Target: <80% per-step MAPE.
- **Stage 2 LOMO:** Apply CMA-ES artifacts (trained on 3 models) to held-out 4th via cross-model transfer. R3 showed 14.8% LOMO E2E — with a better base model, this should improve or maintain.
- **Combined LOMO:** The principled base provides structure; CMA-ES residuals provide per-model fine-tuning. For a truly new model, Stage 1 (cycle-time regression with lifecycle data) is all that's needed — CMA-ES Stage 2 is optional polish.

#### LOWO (Leave-One-Workload-Out)
3-fold LOWO on both stages. The cycle-time regression (Stage 1) is workload-invariant by construction (models batch-composition → cycle-time physics). CMA-ES residuals (Stage 2) are trained across all workloads per model, so they average workload effects. Target: <50% per-step MAPE for Stage 1; <30% E2E for combined.

#### vLLM Args Sensitivity Analysis

| Parameter | Dependent? | Would predictions break? | Recalibration needed |
|---|---|---|---|
| `max_num_seqs` (128) | **Moderate** — Stage 1 captures batch-level physics; Stage 2 residuals are small | Stage 1 robust within training range; Stage 2 residuals may be slightly wrong | Re-run Stage 2 CMA-ES only (~10 min) |
| `max_num_batched_tokens` (2048) | **Moderate** — affects mixed-batch frequency | Stage 1 regime boundaries may shift | Re-estimate regime thresholds from new data; Stage 2 optional |
| `max_model_len` (4096) | **Weak** — no direct impact on per-step cycle time | Stable | None |
| `chunked_prefill` (on) | **Strong** — both stages trained on chunked-prefill data | Without it, pure-prefill regime appears; Stage 1 has no data | Full retrain of Stage 1; Stage 2 optional re-run |
| `prefix_caching` (on) | **Moderate** — affects effective prefill tokens | Stage 1 prefill coefficients may over-predict | Retrain Stage 1 prefill regime; Stage 2 absorbs residual |
| `gpu_memory_utilization` (0.9) | **Weak** — no direct impact | Stable | None |
| `tensor_parallel_size` (varies) | **Strong** — changes communication overhead → Stage 1 intercept | Both stages wrong for new TP | Full retrain of both stages |

**Recalibration story:** Two-level: (1) Stage 1 recalibration requires only ~1000 steps of profiling data + OLS fit (<1 second). (2) Stage 2 requires ~50 BLIS runs (~10 minutes). For routine changes (new max_num_seqs), only Stage 2 needs re-running. For fundamental changes (new TP, chunked_prefill off), both stages retrain from scratch. Total: ~35 minutes max, fully automated.

---

## Summary: How Ideas Address Binding Constraints

| Constraint | Idea 1 (Constrained CMA-ES) | Idea 2 (Cycle-Time Regression) | Idea 3 (Hybrid) |
|---|---|---|---|
| **BC-NEW-1: E2E ↔ ITL tradeoff** | ITL penalty + parameter bounds | Eliminated at source (cycle-time model) | Both: principled base + bounded CMA-ES |
| **BC-NEW-2: Step cycle under-prediction** | Indirectly via overhead tuning | Directly (model cycle time, not compute time) | Directly via Stage 1 |
| **BC-6: vLLM args sensitivity** | Analyzed above | Analyzed above | Analyzed above |
| **BC-4: Mixtral MoE variance** | Better via constrained optimization | Uncertain (linear model may not capture MoE) | Best chance: Stage 1 + MoE-aware Stage 2 |
| **BC-5: LOMO generalization** | Inherits R3's 14.8% LOMO E2E | New — cycle time LOMO | Two-level LOMO |

## Distinctiveness of Ideas

- **Idea 1** improves the *optimization* (multi-objective + constraints) while keeping the same base model
- **Idea 2** improves the *base model* (cycle time target) without black-box optimization
- **Idea 3** combines both — principled base + constrained optimization

All three are distinct and attack different aspects of the problem. Idea 3 subsumes Ideas 1 and 2, but testing them independently reveals which component contributes most.
