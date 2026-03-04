# Iter 4 Hypothesis Bundle: Two-Phase Simulator-Faithful Learning

**Strategy:** Analytical warm-start + coordinate-wise simulator-in-the-loop refinement
**Status:** Pending
**Date:** 2026-03-03
**Motivation:** H30-H32 (#480) proved the dominant BLIS replay error is zero inter-step overhead (δ), not coefficient inaccuracy. Both crossmodel and per-model blackbox produce identical -17% to -56% TTFT underprediction.
**Prior iteration:** Iter 3 (8 global parameters, PARTIAL — analytical TTFT passes, BLIS replay fails)

---

## Parameter Space (8 parameters, all global)

```
StepTime(batch)               = β₀·L + β₁·dc·kv_dim·1e-6 + β₂·(pf+dc)·I(MoE) + β₃·I(TP>1)
QueueingTime(req)             = α₀ + α₁·inputTokens     [α₁ expected 0]
OutputTokenProcessingTime()   = α₂                       [γ₁ in training notation]
SchedulingProcessingTime(bs)  = δ₀ + δ₁·batchSize        [NEW — currently returns 0]
```

| Param | Warm-Start | Search Range | Physical Meaning |
|-------|-----------|-------------|-----------------|
| β₀ | 116.1 µs/layer | [81, 151] | CUDA kernel dispatch per transformer layer |
| β₁ | 1226.9 µs/kv_unit | [859, 1595] | Decode KV cache HBM read bandwidth |
| β₂ | 19.9 µs/MoE_tok | [14, 26] | MoE expert routing + dispatch/gather |
| β₃ | 9445.2 µs/step | [6612, 12279] | NCCL all-reduce barrier (TP>1) |
| α₀ | 13,732 µs | [9612, 17852] | Pre-scheduling CPU (tokenization, HTTP) |
| α₂ | 860.6 µs/tok | [430, 1291] | Output processing (absorbs β decode error) |
| δ₀ | ~1,500 µs/step | [500, 5000] | Fixed per-step scheduler overhead |
| δ₁ | ~10 µs/req | [0, 100] | Per-request metadata preparation cost |

β bounded ±30% of Iter 3. α₂ bounded ±50% (expected to decrease with δ). δ ranges from H30 vLLM analysis.

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

### δ (new — inter-step timing gaps)

Extract from BATCH_SUMMARY consecutive step pairs:
```
gap_k = ts_start_ns[k+1] - ts_end_ns[k]    (microseconds)
features = [1, batch_num_prefill_reqs[k] + batch_num_decode_reqs[k]]
δ = nnls(features, max(gap, 0))
```

10% step sampling limits consecutive pair count but distribution should be representative.

---

## Phase 2: Coordinate-Wise Simulator-in-the-Loop Refinement

### Why coordinate-wise (not joint 8D search)

Joint optimization of 8 parameters with a simulator objective is expensive: each evaluation requires 10 BLIS runs (~3 min). CMA-ES with population 12 × 80 generations = 960 evaluations × 3 min = ~48 hours. Coordinate-wise search exploits the natural hierarchy: δ has 10× more leverage than β adjustments. By fixing low-leverage parameters while searching high-leverage ones, total budget drops to ~6-8 hours.

### Stage A: δ sweep (highest leverage, 2 params)

Fix β from Phase 1, fix α from Phase 1. Grid search:
- δ₀ ∈ {500, 1000, 1500, 2000, 2500, 3000, 4000, 5000} (8 values)
- δ₁ ∈ {0, 10, 25, 50, 100} (5 values)
- 40 grid points × 4 fast-subset experiments × ~15s each = ~40 min

Fast subset: llama-2-7b-general (TP=1), llama-2-70b-general (TP=4), mixtral-8x7b-general (MoE), codellama-34b-codegen (cross-profile).

Select (δ₀, δ₁) minimizing TTFT MAPE across fast subset. Validate on all 10 training experiments.

### Stage B: α₀, α₂ refinement (next highest, 2 params)

Fix β from Phase 1, fix δ from Stage A. Bayesian optimization:
- `gp_minimize(loss, [(9612, 17852), (430, 1291)], n_calls=30, x0=[α₀_init, α₂_init])`
- Loss: 0.5 × TTFT_MAPE + 0.3 × E2E_MAPE + 0.2 × throughput_RE on all 10 training experiments
- ~30 evals × 3 min = ~1.5 hrs

Expected: α₀ stays near 13,732 (pre-scheduling overhead is real). α₂ decreases from 860.6 toward ~200-600 as δ absorbs part of its compensating role.

### Stage C: β fine-tuning (already close, 4 params)

Fix δ from Stage A, fix α from Stage B. CMA-ES:
- σ₀ = 0.05 (5% of parameter value — tight search)
- Population size 10, max 50 generations = 500 evaluations
- ~500 evals × 3 min = ~25 hrs (run overnight) OR use fast subset (~6 hrs)
- Bounds: ±15% of Iter 3 (tighter than ±30% since β is already good)

Expected: small adjustments (< 10%). May be skippable if Stage A+B meet all targets.

### Stage D: Global polish (optional, all 8 params)

Only if Stages A-C leave individual experiments failing gates. CMA-ES:
- σ₀ = 0.03 (3% — very tight)
- Warm-start from Stages A-C output
- Population size 12, max 30 generations = 360 evaluations
- ~360 evals × 3 min = ~18 hrs

### Multi-Signal Loss Function

```
L(params) = Σ_exp [ w_step · L1(exp) + w_queue · L2(exp) + w_ttft · L3(exp)
                   + w_e2e · L4(exp) + w_thru · L5(exp) ] / N_exp

L1 (step, 15%):  Σ_bin squared_bias(blis_steps[bin], real_steps[bin])
L2 (queue, 25%): ||resample(blis_queue_depth, 100) - resample(real_queue_depth, 100)||² / ||real||²
L3 (TTFT, 30%):  mean(|blis_ttft - real_ttft| / real_ttft)  [matched by arrival index]
L4 (E2E, 20%):   mean(|blis_e2e - real_e2e| / real_e2e)    [matched by arrival index]
L5 (thru, 10%):  ((blis_rps - real_rps) / real_rps)²
```

Queue dynamics (L2) gets highest weight after TTFT because it is the leading indicator of cascade divergence.

---

## Hypothesis Bundle

### H-main

> Two-phase fitting with δ achieves TTFT mean RE < ±15% and E2E mean RE < ±12% across all 10 training experiments in BLIS replay, and TTFT mean RE < ±20% on codellama validation experiments.

**Refuted if:** Any training experiment has TTFT mean |RE| > 20%, or any codellama validation experiment has TTFT mean |RE| > 25%.

### H-ablation-δ

> Removing δ (setting δ₀=δ₁=0, keeping Phase 2 refined β/α) collapses back to H30-level TTFT underprediction (-17% to -56%).

**Diagnostic if fails:** δ is NOT the dominant error — the TTFT gap has another source (batch composition divergence, α extraction error, etc.).

### H-ablation-SIL

> Simulator-in-the-loop refinement (Phase 2) improves TTFT mean |RE| by at least 5pp over analytical-only (Phase 1) coefficients on training experiments.

**Diagnostic if fails:** Analytical warm-start is already good enough — SIL refinement adds complexity without value.

### H-control-neg

> At ρ < 0.3 (lowest load stage, ~6 RPS arrival), δ has negligible effect (TTFT change < 5% relative to δ=0). Queue wait is near-zero, so inter-step overhead does not compound.

**Diagnostic if fails:** Even at low load, δ matters — inter-step overhead is per-step, not per-request, so it shifts TTFT by δ₀ × prefill_steps even without queueing.

### H-gamma-decrease

> After δ is introduced, the Phase 2 optimized α₂ (γ₁) decreases by at least 30% from its Iter 3 value of 860.6 µs/tok, because δ absorbs the inter-step component that γ₁ was previously compensating.

**Diagnostic if fails:** γ₁'s 860.6 value was entirely β decode error absorption (not inter-step compensation). δ provides additional overhead rather than replacing γ₁'s role.

### H-robustness

> Phase 2 refined coefficients achieve TTFT mean RE < ±20% on codellama-34b codegen and roleplay (validate set, cross-profile generalization) without per-profile tuning.

**Diagnostic if fails:** Cross-profile generalization breaks because different profiles have different batch-size distributions, and the coordinate-wise optimization overfits to training profiles' batch characteristics.

### H-saturation

> The refined δ reduces the codellama-34b-reasoning throughput overestimate from +22% to < +10%, shifting BLIS's ρ estimate closer to the real ρ=1.24.

**Diagnostic if fails:** The throughput gap at saturation is NOT from inter-step overhead alone — it involves KV allocation dynamics, preemption timing, or other scheduler mechanisms that δ does not capture.

---

## Fast-Fail Rules

1. **H-ablation-δ fails** (δ removal doesn't restore H30 gap): Stop. Re-diagnose the TTFT error source with a "perfect-beta" control (inject real step durations into BLIS).
2. **H-control-neg fails** (δ matters even at ρ < 0.3): Not a failure — refine the H-main targets. δ adds constant overhead per request (δ₀ × num_steps), which is load-independent.
3. **H-main fails after all 4 stages**: The error is in BLIS's `VLLMBatchFormation`, not the latency model. File issue for scheduler fidelity improvement.
4. **Stage A alone meets all targets**: Skip Stages B-D. The dominant error was δ, and analytical β/α are sufficient.

---

## BLIS Implementation Prerequisites

1. **Learn `SchedulingProcessingTime()`**: Add `SchedulingCoeffs []float64` to `LatencyCoeffs`. Implement `δ₀ + δ₁·batchSize` in crossmodel backend. For now, the interface takes no args — use fixed δ₀ only, or add a batch-size parameter.
2. **Apply δ every step**: In `simulator.go:scheduleNextStep()`, change `time: now + currStepAdvance` to `time: now + currStepAdvance + schedulingOverhead`.
3. **Per-step instrumentation**: Add `StepDurations []int64`, `StepPrefillTokens []int64`, `StepDecodeTokens []int64` to `Metrics`. Append in `executeBatchStep()`.
4. **Parameterize replay binary**: Accept `--beta`, `--alpha`, `--delta` as CLI flags instead of hardcoded values. Add `--emit-step-metrics` flag.

---

## Data Sources (from `schemas.py`)

| Source | Phase 1 Use | Phase 2 Use |
|--------|-------------|-------------|
| `StepBatchSummary.duration_us` | β target (Block A) | L1 step comparison |
| `StepBatchSummary.ts_start_ns/ts_end_ns` | δ extraction (inter-step gap) | — |
| `StepBatchSummary.queue_waiting_depth` | — | L2 real queue depth |
| `StepBatchSummary.batch_num_*_reqs` | β stratification | L1 operating-point bins |
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
3. Dominant error identified as irreducible (VLLMBatchFormation structural divergence)

Validate on held-out experiments (H31, H32) only after training convergence. Test set touched exactly once.
