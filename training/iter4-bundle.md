# Iter 4 Hypothesis Bundle: Two-Phase Simulator-Faithful Learning

**Strategy:** Analytical warm-start + coordinate-wise simulator-in-the-loop refinement
**Status:** Pending (Design Review Round 1 complete, fixes applied)
**Date:** 2026-03-03
**Family:** Validation (comparing BLIS model output to real vLLM behavior)
**VV&UQ Category:** Validation — system-level comparison against empirical ground truth
**Type:** Statistical/Dominance (H-main, H-ablation-δ), Statistical/Equivalence (H-control-neg)
**Motivation:** H30-H32 (#480) proved the dominant BLIS replay error is zero inter-step overhead, not coefficient inaccuracy. Both crossmodel and per-model blackbox produce identical -17% to -56% TTFT underprediction.
**Prior iteration:** Iter 3 (8 global parameters, PARTIAL — analytical TTFT passes, BLIS replay fails)

---

## Parameter Space (8 parameters, all global)

```
StepTime(batch)             = β₀·L + β₁·dc·kv_dim·1e-6 + β₂·(pf+dc)·I(MoE) + β₃·I(TP>1)
QueueingTime(req)           = α₀ + α₁·inputTokens     [α₁ expected 0]
OutputTokenProcessingTime() = α₂                       [γ₁ in training notation]
InterStepOverhead()         = δ₀                        [NEW method — see Architecture section]
```

| Param | Warm-Start | Search Range | Physical Meaning |
|-------|-----------|-------------|-----------------|
| β₀ | 116.1 µs/layer | [81, 151] | CUDA kernel dispatch per transformer layer |
| β₁ | 1226.9 µs/kv_unit | [859, 1595] | Decode KV cache HBM read bandwidth |
| β₂ | 19.9 µs/MoE_tok | [14, 26] | MoE expert routing + dispatch/gather |
| β₃ | 9445.2 µs/step | [6612, 12279] | NCCL all-reduce barrier (TP>1) |
| α₀ | 13,732 µs | [9612, 17852] | Pre-scheduling CPU (tokenization, HTTP) |
| α₂ | 860.6 µs/tok | [0, 1291] | Output processing (physical: ~5 µs/tok; Iter 3 absorbs β error) |
| δ₀ | **model-dependent (see below)** | [5000, 25000] | Fixed per-step inter-step overhead |

β bounded ±30% of Iter 3. α₂ lower bound extended to 0 to allow discovery of physical value if δ absorbs all inter-step compensation.

**Phase 1 diagnostic extraction (completed):** Measured inter-step gaps from 8,690 truly consecutive BATCH_SUMMARY step pairs across 10 training experiments:

| Model | TP | Count | Median δ (µs) | P5 (µs) | P95 (µs) | Batch-size corr |
|-------|-----|-------|---------------|---------|----------|-----------------|
| llama-2-7b | 1 | 4,150 | 7,774 | 6,533 | 10,567 | +0.80 |
| codellama-34b | 2 | 716 | 14,275 | 8,164 | 16,139 | -0.47 |
| llama-2-70b | 4 | 1,898 | 17,439 | 14,766 | 18,729 | -0.18 |
| mixtral-8x7b | 2 | 1,926 | 18,291 | 16,862 | 19,959 | +0.62 |
| **Global** | — | **8,690** | **13,159** | **6,720** | **19,269** | — |

The gap is 14× larger than GPU step compute time (median step_duration_us: 143-530 µs). The preliminary 1,500 µs estimate in the initial bundle was wildly wrong. Search range updated to [5000, 25000].

**Model dependence:** δ correlates with TP degree and model size (7b/TP=1: 8ms; 70b/TP=4: 17ms; mixtral/TP=2: 18ms). A single global δ₀ will be a compromise. If the global δ₀ produces > 5pp cross-model TTFT variance, a model-aware δ₀ = δ_base + δ_tp · I(TP>1) should be considered in Stage C or D.

**Batch-size correlation is inconsistent across models** (7b: +0.80, codellama: -0.47). δ₁ (per-batch-size term) remains deferred — no consistent signal to anchor it. The search space is 7 free parameters (α₁ fixed at 0, δ₁ deferred).

---

## Architecture: InterStepOverhead vs SchedulingProcessingTime

**DR-5 Round 1 identified a critical architectural conflict.** The existing `SchedulingProcessingTime()` method is called per-request at `batch_formation.go:131` and produces a `ScheduledDelay` for each newly admitted request. The proposed δ is per-STEP overhead applied between steps. These are two distinct concepts:

| Concept | Fires | Affects | Current Method |
|---------|-------|---------|----------------|
| Per-request scheduling delay | Once per newly admitted request | When a request is recorded as "scheduled" | `SchedulingProcessingTime()` at `batch_formation.go:131` |
| Per-step inter-step overhead | Every step | When the next `StepEvent` fires (global clock) | **NEW: `InterStepOverhead()`** |

**Solution: Add a new `InterStepOverhead() int64` method to the `LatencyModel` interface.** This cleanly separates the two concepts, avoids double-counting, and does not modify the existing `SchedulingProcessingTime()` semantics. The existing method continues to return 0 for all backends (per-request scheduling delay is already modeled by α₀ in `QueueingTime`).

**Interface evolution:** problem.md Section 6a describes the interface as "frozen." This is a Phase C interface evolution (adding a method, not modifying existing ones). All 3 implementations (Blackbox, Roofline, CrossModel) gain a default returning 0 (backward-compatible). Only CrossModel's return value changes during Iter 4. R13 compliance: the new method works for all backends (returns 0 for Blackbox/Roofline, δ₀ for CrossModel).

**Application points in simulator.go:** `InterStepOverhead()` is called in `scheduleNextStep()` at BOTH step-continuation paths:
- Line 432: `time: now + currStepAdvance + sim.latencyModel.InterStepOverhead()` (batch continues)
- Line 444: `time: now + currStepAdvance + sim.latencyModel.InterStepOverhead()` (work-conserving new batch)

**TTFT effect pathways:** δ delays when the NEXT step fires but does NOT inflate TTFT for requests completing prefill in the CURRENT step (TTFT is recorded at `now + currStepAdvance + OutputTokenProcessingTime()`, line 365). δ affects TTFT through three pathways:

- **(a) Scheduling-delay (load-independent, dominant at low load):** Requests arriving DURING a step must wait for `currStepAdvance + δ` before the next step starts (vs `currStepAdvance` without δ). This is a constant per-step increase in scheduling latency even for a single waiting request with no queue buildup. At low load (ρ < 0.3), this is the dominant mechanism. H-control-neg's "bounded by δ₀ × num_prefill_steps" refers to this pathway.
- **(b) Queue-depth amplification (load-dependent, dominant at high load):** Longer inter-step gaps mean more arrivals accumulate between steps → higher ρ → deeper queues → non-linear TTFT growth. This is the cascade mechanism.
- **(c) Batch-admission delay:** Fewer steps per unit time means new requests wait longer to enter the running batch, even if the queue itself is short.

---

## Phase 1: Analytical Warm-Start

### β (unchanged from Iter 3)

NNLS with Block A (step observations) + Block B (journey constraints):
- Block A: 88K training steps, features `[L, dc·kv_dim·1e-6, (pf+dc)·I(MoE), I(TP>1)]`, target `duration_us`
- Block B: 231K journey segments, features summed over step ranges, weight `w = sqrt(N_A / N_B)`
- Solve: `β = argmin_{β≥0} ||X·β - y||²` via `scipy.optimize.nnls`

### α (unchanged from Iter 3)

- α₀: 5th percentile of QUEUED→SCHEDULED intervals for requests scheduled within 1 step of arrival at lowest load stage
- α₁: NNLS on TTFT residuals (consistently zeroed)
- α₂/γ₁: NNLS on E2E residuals `= E2E_real - (α + prefill_pred + decode_pred)`

### δ₀ (new — inter-step timing gaps)

Extract from BATCH_SUMMARY step pairs where `step_id[k+1] == step_id[k] + 1` (truly consecutive, not just adjacent in the 10% sample):
```
gap_k = (ts_start_ns[k+1] - ts_end_ns[k]) / 1000    (microseconds)
```

**Preconditions (verified in script, not just prose):**
1. Filter to truly consecutive step IDs only (`step_id` gap == 1); discard non-consecutive pairs from 10% sampling
2. Require ≥ 50 consecutive pairs per experiment (abort with diagnostic if fewer)
3. Filter negative gaps (clock skew) and outliers > 50ms (GC pauses, context switches)
4. Log the number of valid consecutive pairs per experiment

**Expected yield:** With 10% independent sampling, P(two consecutive both sampled) = 1%. At ~10,000 steps/experiment, expect ~100 pairs. Across 10 experiments: ~1,000 total pairs. Adequate for a 1-parameter NNLS.

Fit: `δ₀ = median(positive_gaps)` (robust to outliers). Regression against batch size deferred — if residual analysis shows batch-size correlation, introduce δ₁ later.

**Back-of-envelope check (codellama-34b):** The H30 measured throughput is 3.93 rps with step-only compute. Adding measured δ₀ ≈ 14,275 µs (codellama median from diagnostic): effective per-step wall time = step_compute + δ₀. Since throughput ∝ 1/(step_compute + δ₀), and δ₀ is ~14× the median step_duration_us (~530 µs), the inter-step overhead dominates wall time. The H30 throughput of 3.93 rps was achieved because BLIS only modeled step_compute; adding δ₀=14,275 would roughly halve throughput — overshooting the real 3.22 rps target. **Note:** The simplified β₀·48 + β₃ formula (~15,018 µs) understates actual step time because it omits the β₁ decode term (which adds ~2,000-10,000 µs depending on batch decode tokens). The grid search will find the δ₀ that empirically matches throughput; the back-of-envelope serves only to validate the search range [5000, 25000] is physically reasonable.

---

## Phase 2: Coordinate-Wise Simulator-in-the-Loop Refinement

### Why coordinate-wise (not joint 7D search)

Joint optimization of 7 parameters with a simulator objective is expensive. Coordinate-wise search exploits the natural leverage hierarchy: δ₀ has 10× more leverage than β adjustments. Each stage fixes low-leverage parameters while searching high-leverage ones.

### Stage A: δ₀ sweep (highest leverage, 1 param)

Fix β, α from Phase 1. Grid search:
- δ₀ ∈ {500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000} (10 values)
- 10 grid points × 4 fast-subset experiments × ~15s each = ~10 min
- Then validate best δ₀ on ALL 10 training experiments (~2.5 min)

Fast subset: llama-2-7b-general (TP=1), llama-2-70b-general (TP=4), mixtral-8x7b-general (MoE), codellama-34b-codegen (cross-profile).

**Validation gate:** If any of the 6 non-fast-subset experiments shows TTFT |RE| > 25% after Stage A, re-run grid with all 10 experiments.

**Per-stage checkpoint:** Record coefficients and evaluate all 10 experiments at the end of Stage A. This checkpoint is the reference for H-ablation-SIL (comparing Stage A alone vs full refinement).

### Stage B: α₀, α₂ refinement (next highest, 2 params)

Fix β from Phase 1, fix δ₀ from Stage A. Bayesian optimization:
- `gp_minimize(loss, [(9612, 17852), (0, 1291)], n_calls=30, x0=[α₀_init, α₂_init], random_state=42)`
- Loss: per-model-weighted composite (see loss function below) on all 10 training experiments
- ~30 evals × 3 min = ~1.5 hrs

Expected: α₀ stays near 13,732. α₂ decreases from 860.6 — how much depends on whether γ₁ was compensating inter-step overhead (now captured by δ) or purely β decode error.

**Causality control:** Also run Stage B with δ₀=0 (Phase 1 β, no delta) to produce a "re-optimized α without δ" reference. If α₂ changes even without δ, the decrease is not caused by δ. This is 1 additional Bayesian run (~1.5 hrs).

**Per-stage checkpoint:** Record and evaluate after Stage B.

### Stage C: β fine-tuning (already close, 4 params)

Fix δ₀ from Stage A, fix α from Stage B. CMA-ES:
- σ₀ = 0.05 (5% of parameter value — tight search)
- Population size 10, max 50 generations = 500 evaluations, `random_state=42`
- ~500 evals × 3 min = ~25 hrs (run overnight) OR use fast subset (~6 hrs)
- Bounds: ±15% of Iter 3 (tighter than ±30% since β is already good)

Expected: small adjustments (< 10%). May be skippable if Stage A+B meet all targets.

**ED-1 exception (justified):** Stage C jointly searches 4 β parameters, not one. This is an optimization procedure (finding the best β in a 4D neighborhood), not a hypothesis test (isolating one variable's effect). The tight search radius (σ₀=5%, bounds ±15%) constrains the search to a local neighborhood of the analytically-derived warm-start. Per-stage checkpoints (Stage A → A+B → A+B+C) enable after-the-fact attribution of which stage contributed how much improvement.

**Per-stage checkpoint:** Record and evaluate after Stage C.

### Stage D: Global polish (optional, all 7 params)

Only if Stages A-C leave individual experiments failing gates. CMA-ES:
- σ₀ = 0.03 (3% — very tight)
- Warm-start from Stages A-C output
- Population size 12, max 30 generations = 360 evaluations, `random_state=42`
- ~360 evals × 3 min = ~18 hrs

### Seed Strategy

- **During optimization (Stages A-D):** Fix BLIS seed=42 for all runs. Single-seed evaluation is acceptable because BLIS is deterministic and each experiment uses its own workload spec (the comparison is BLIS-vs-real, not BLIS-vs-BLIS).
- **Post-optimization validation:** Run the final coefficients on all 10 training experiments with seeds {42, 123, 456}. Report mean ± std across seeds. Require directional consistency (same sign of RE) across all 3 seeds for each experiment.
- **Optimizer determinism:** All optimizers use `random_state=42` (CMA-ES, gp_minimize). The full pipeline is reproducible from a single `run.sh`.

### Multi-Signal Loss Function

```
L(params) = Σ_exp [ w_model(exp) × (w_ttft · L3(exp) + w_e2e · L4(exp)
                    + w_queue · L2(exp) + w_step · L1(exp) + w_thru · L5(exp)) ] / N_exp

L1 (step, 15%):  Σ_bin squared_bias(blis_steps[bin], real_steps[bin])
L2 (queue, 15%): ||resample_by_time(blis_qd, 100) - resample_by_time(real_qd, 100)||² / ||real||²
L3 (TTFT, 40%):  mean(|sorted_blis_ttft - sorted_real_ttft| / sorted_real_ttft)  [rank-matched]
L4 (E2E, 20%):   mean(|sorted_blis_e2e - sorted_real_e2e| / sorted_real_e2e)    [rank-matched]
L5 (thru, 10%):  ((blis_rps - real_rps) / real_rps)²
```

**Changes from Round 1:**
- TTFT weight increased to 40% (from 30%) — primary target per H30.
- Queue weight reduced to 15% (from 25%) — no baseline measurement exists; risk of artifact domination.
- L2 resampling is time-based (wall-clock), not step-index-based, to normalize for different BLIS/vLLM step rates.
- L3/L4 use rank-matching (k-th fastest in BLIS vs k-th fastest in real) instead of arrival-index matching, since BLIS generates different Poisson arrivals than real data.
- Per-model weighting `w_model(exp)`: codellama experiments weighted 3× (only 1 training experiment vs 3 for other models) to prevent model-count imbalance from biasing α₀.

### Python Dependencies

```
# requirements.txt for Iter 4 optimization
scipy>=1.10           # nnls, optimization
scikit-optimize>=0.9  # gp_minimize (Stage B)
cma>=3.3              # CMA-ES (Stages C-D)
numpy>=1.24
pandas>=2.0
```

`run.sh` installs dependencies: `pip install -r training/requirements.txt`.

---

## Hypothesis Bundle

### H-main

> When inter-step scheduling overhead (δ₀) is modeled in BLIS's event loop, the simulator's TTFT prediction error decreases from the current -17% to -56% range to within ±15% for training experiments, because the dominant scheduling model gap identified in H30 is captured by δ₀ compounding over ~150 steps per request.

**Confirmed if:** TTFT mean |RE| < 15% across all 10 training experiments AND TTFT mean |RE| < 20% on codellama validation experiments.
**Inconclusive if:** TTFT mean |RE| between 15-20% on any training experiment.
**Refuted if:** Any training experiment has TTFT mean |RE| > 20%, or any codellama validation experiment has TTFT mean |RE| > 25%.

### H-ablation-δ (two conditions for proper isolation)

> (a) Phase 1 analytical β/α with δ₀=0 reproduces the H30-level TTFT underprediction (-17% to -56%), establishing the "before" baseline on identical infrastructure.
> (b) Phase 2 refined β/α with δ₀=0 also shows TTFT underprediction ≥ -15%, confirming δ₀ is necessary even with optimized β/α.
> If (a) ≈ (b), β/α refinement provides no benefit without δ₀. If (b) is significantly better than (a), β/α refinement alone partially closes the gap.

**If this fails, it would indicate** that the TTFT gap has a source other than inter-step overhead — batch composition divergence, α extraction error, or VLLMBatchFormation structural differences.

### H-ablation-SIL

> Phase 2 SIL refinement improves TTFT mean |RE| by at least 5pp over Phase 1 analytical-only coefficients (Phase 1 β/α + Phase 1 δ₀) on training experiments. If this fails, it would indicate the analytical warm-start is already sufficient and SIL adds complexity without value — accept Phase 1 results.

### H-control-neg (revised: accounts for per-step constant shift)

> At ρ < 0.3 (dedicated single-rate low-load workload, not extracted from multi-stage), δ₀'s TOTAL contribution to TTFT is bounded by δ₀ × num_prefill_steps (constant, load-independent). The queueing-amplification component of δ₀'s effect (the part that compounds non-linearly with load) should be < 5% of total TTFT change attributable to δ₀.

**If this fails, it would indicate** that even at low load, queueing dynamics amplify δ₀ beyond the constant per-step overhead — suggesting BLIS's batch formation creates queueing even when the real system does not.

**Low-load workload construction:** Use codellama-34b (best-calibrated model in H30: -23% TTFT RE). Generate a single-stage workload at 2 RPS (ρ ≈ 0.2 with δ₀, well below saturation) with constant tokens matching the general profile (system_prompt_len=512, question_len=547, output_len=512). Run for 200 requests (sufficient for stable mean TTFT, short enough for fast iteration). Same KV blocks (26,602) and server config as training. Alternatively, extract Stage 0 (first 600s, ~8 RPS) from the training experiments and measure δ₀'s effect on that stage only — but this is less clean due to Stage 1 follow-on effects.

**Matched positive control:** At ρ > 0.8 (high-load stage of training experiments), the same δ₀ should produce > 15% TTFT change. This creates a dose-response pair: (low load: δ₀ effect bounded by constant term) vs (high load: δ₀ effect amplified by queueing).

### H-gamma-decrease

> After δ₀ is introduced, the Phase 2 optimized α₂ decreases by at least 30% from its Iter 3 value of 860.6 µs/tok. If this fails, it would indicate γ₁'s 860.6 value was entirely β decode error absorption, not inter-step compensation, and δ₀ provides additional overhead rather than replacing γ₁'s role.

**Causality control:** Run Stage B α optimization both WITH and WITHOUT δ₀. If α₂ changes even without δ₀, the decrease is from the Bayesian optimizer finding a different basin than Phase 1 NNLS, not from δ₀ absorbing inter-step overhead. The causal claim requires α₂ to stay near 860 without δ₀ and decrease with δ₀.

### H-cross-profile (renamed from H-robustness)

> Phase 2 refined coefficients achieve TTFT mean RE < ±20% on codellama-34b codegen and roleplay (validate set, unseen profiles but same architecture) without per-profile tuning. If this fails, it would indicate coordinate-wise optimization overfits to training profiles' batch-size distributions.

**Scope note:** This tests cross-PROFILE generalization only. Cross-architecture and cross-hardware robustness are out of scope (all 4 architectures are in the training set).

### H-saturation

> The refined δ₀ reduces BLIS's maximum sustainable throughput for codellama-34b from 3.93 rps toward real vLLM's 3.22 rps, with the throughput overestimate decreasing from +22% to < +15%.

**Back-of-envelope:** At codellama-34b (48 layers, TP=2), step time ≈ 15,018 µs. With δ₀=1,500 µs, effective step time = 16,518 µs (+10%). Throughput drops from 3.93 to ~3.57 rps (+11% overestimate). At δ₀=2,500: effective 17,518 µs, throughput ~3.37 rps (+5%). The grid search will find the δ₀ that best matches.

**Saturation-specific controls:**
- (a) KV-unlimited control: Set `--total-kv-blocks` very high (1,000,000) for the codellama-34b-reasoning experiment. If throughput gap persists without KV pressure, inter-step overhead is the cause. If it shrinks, KV dynamics contribute.
- (b) Note: H-saturation tests whether δ₀'s magnitude happens to correct the throughput — it does NOT prove δ₀ physically represents real vLLM overhead. A throughput match at saturation is necessary but not sufficient for validation.

### H-ceiling (perfect-beta)

> When real vLLM step durations are injected directly into BLIS (bypassing the latency model entirely), BLIS's TTFT prediction error establishes a structural ceiling. If the ceiling is > ±10% TTFT |RE|, the ±15% target for H-main may be structurally unachievable due to VLLMBatchFormation divergence.

**If ceiling < ±5%:** Coefficient optimization has full room to succeed — the target is achievable.
**If ceiling ±5-15%:** Target is marginally achievable; coefficients must be very accurate.
**If ceiling > ±15%:** H-main target is structurally unachievable; revise targets or improve VLLMBatchFormation.

---

## Fast-Fail Rules

1. **H-ceiling shows > ±15%:** Stop coefficient optimization. The bottleneck is VLLMBatchFormation, not coefficients. File issue for scheduler fidelity improvement.
2. **H-ablation-δ condition (a) fails** (δ=0 doesn't reproduce H30 gap): Infrastructure change introduced a confound. Investigate binary diff.
3. **H-main fails after all 4 stages with ceiling < ±5%:** Coefficients are the bottleneck but the search didn't converge. Consider larger search budget or alternative optimizer.
4. **Stage A alone meets all targets:** Skip Stages B-D. δ₀ was the dominant error; analytical β/α are sufficient.

---

## BLIS Implementation Prerequisites

1. **New `InterStepOverhead() int64` method on `LatencyModel` interface:** Returns 0 for Blackbox/Roofline, δ₀ for CrossModel. Keeps `SchedulingProcessingTime()` unchanged for its existing per-request role (avoids double-counting). R13: works for all 3 backends.
2. **Add `InterStepCoeffs []float64` to `LatencyCoeffs`:** R4 canonical constructor `NewLatencyCoeffs` gains the field — compiler error at every call site.
3. **Apply δ₀ every step in both paths of `scheduleNextStep()`:** Line 432 (batch continues) and line 444 (work-conserving new batch) both become `time: now + currStepAdvance + sim.latencyModel.InterStepOverhead()`.
4. **Per-step instrumentation on `Metrics`:** Add `StepDurations []int64`, `StepPrefillTokens []int64`, `StepDecodeTokens []int64`. Append in `executeBatchStep()`. `NumWaitQRequests` already captured (no change needed for L2).
5. **Parameterize replay binary:** Accept `--beta`, `--alpha`, `--delta` as CLI flags. Add `--emit-step-metrics` flag. `--seed` flag for reproducibility.
6. **Per-experiment KV blocks from ground truth:** All BLIS replay runs MUST use `kv_blocks_total_gpu` from the experiment's ground truth JSON, not defaults.yaml or CLI defaults.
7. **Config diff with H30:** Iter 4 uses a modified BLIS binary (prerequisites 1-4). H-ablation-δ condition (a) re-runs on the new binary with δ₀=0 to establish the "before" baseline, ensuring infrastructure changes are isolated.

---

## Data Sources (from `schemas.py`)

| Source | Phase 1 Use | Phase 2 Use |
|--------|-------------|-------------|
| `StepBatchSummary.duration_us` | β target (Block A) | L1 step comparison |
| `StepBatchSummary.ts_start_ns/ts_end_ns` | δ₀ extraction (consecutive step gaps) | — |
| `StepBatchSummary.queue_waiting_depth` | — | L2 real queue depth |
| `StepBatchSummary.batch_num_*_reqs` | β stratification | L1 operating-point bins |
| `StepBatchSummary.step_id` | δ₀ consecutive pair filtering | — |
| `JourneyEventBase` subclasses | β Block B + α extraction | — |
| `PerRequestMetrics` | α₂/γ₁ fitting | L3/L4 ground truth |
| `PerRequestMetrics.output_token_times[]` | — | ITL validation (γ independent check) |
| `ExperimentConfig` | BLIS replay config | BLIS replay config |
| `LifecycleMetrics` (summary) | — | L5 throughput comparison |
| `KVEventBatch` | — | Preemption/eviction validation |

---

## Stopping Criterion

Stop when:
1. All 10 training experiments pass: TTFT mean |RE| < 15%, E2E mean |RE| < 12%, throughput |RE| < 5%, OR
2. Two consecutive stages produce < 2pp improvement on worst-case experiment, OR
3. H-ceiling shows the target is structurally unachievable (VLLMBatchFormation divergence)

Validate on held-out experiments (H31, H32) only after training convergence. Test set (`--evaluate-test` flag) touched exactly once.
