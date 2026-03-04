# Round 3 Research Ideas — StepML Latency Model Fidelity

**Date:** 2026-02-27
**Input:** `hypotheses/h-stepml/problem.md` (sole input, contains all Round 1+2 findings)
**Target:** Workload-level E2E mean error < 10% per experiment
**Current best:** 427.8% mean E2E (Round 2, Idea 2 regime ensemble); ITL 33.6% mean (5/10 < 10%); TTFT 31,906% mean

## Context Summary

Round 2 demonstrated that **ITL is nearly solved** (5/10 experiments < 10%) via the overhead floor mechanism, but **TTFT errors of 31,906% dominate E2E error**. The TTFT mismatch is a simulation-fidelity problem, not a step-time modeling problem. Meanwhile, step-time prediction has a known formulation gap: the model lacks a total-context-length feature that captures memory-bandwidth-bound attention costs (validated by FairBatching [arXiv 2025] at ±1.3% per-step error with `a + b*new_tokens + c*total_context`).

### Binding Constraints Being Addressed

| Constraint | Addressed By |
|---|---|
| BC-1: TTFT/simulation fidelity mismatch (PRIMARY) | **Idea 1** (trace replay) and **Idea 3** (E2E calibration) |
| BC-2: KV feature scaling | **Idea 2** (total-context formulation) |
| BC-3: CodeLlama-34B anomaly | **Idea 2** (per-model investigation) and **Idea 3** (joint tuning) |
| BC-4: Mixed-heavy regime sparsity | **Idea 2** (threshold analysis) |

---

## Idea 1: Trace-Driven Simulation with Lifecycle Replay

### Title
Bypass workload-spec generation by replaying ground-truth request traces directly in BLIS, eliminating the arrival-pattern mismatch that causes 31,906% TTFT error.

### Rationale

The catastrophic TTFT error is not a step-time problem — it is a simulation-fidelity problem where BLIS's generated workload (from inference-perf workload specs) produces different arrival patterns, request sizes, and load levels than the real system. This is consistent with findings from:

- **Revati** [arXiv 2601.00397, Microsoft/Georgia Tech 2025]: Showed that reimplemented schedulers inevitably diverge from real vLLM, and proposed "time-warp" simulation using the *actual* vLLM control plane. The key finding: "GPU computation accounts for 90-95% of total execution time; CPU overhead is minimal."
- **DistServe** [arXiv 2401.09670, 2024]: M/D/1 queuing model for TTFT: `TTFT = D + RD²/(2(1-RD))`. Near capacity, even a 10% mismatch in arrival rate (R) or service time (D) causes exponential TTFT divergence. This explains why TTFT errors are 31,906% while ITL errors are 33.6%.
- **TokenSim** [arXiv 2503.08415, 2025]: "Existing simulators yield highly inaccurate metrics for dynamic workloads due to their lack of batching support" — batching/scheduling simulation, not step-time prediction, is the dominant error source.

The existing infrastructure already has `convert_lifecycle_to_traces.py` which converts ground-truth lifecycle data to BLIS trace CSV format. BLIS supports trace replay via `TracesWorkloadFilePath` (TraceV2 format: YAML header + CSV data). By replaying the **exact** ground-truth arrival times and token lengths, we eliminate the workload specification as an error source and isolate the step-time model's contribution to E2E error.

### Method Sketch

1. **Convert ground-truth lifecycle data to BLIS TraceV2 format** using the existing `convert_lifecycle_to_traces.py` infrastructure
2. **Run BLIS with trace replay** (`--workload-spec trace.yaml`) instead of inference-perf workload generation, using the best Round 2 StepML coefficients
3. **Compare E2E/TTFT/ITL** with trace replay vs. workload-spec generation to quantify the workload-specification contribution to error
4. **If trace replay reduces E2E dramatically**, the binding constraint is the workload spec generator — investigate which spec parameters (arrival rate, token distributions, horizon) cause divergence
5. **If trace replay does NOT reduce E2E**, the binding constraint is in BLIS's scheduling/batch-formation logic — investigate batch composition differences

### Expected Outcome

We expect trace replay to reduce TTFT error from 31,906% to < 50%, confirming that the workload-spec generator is the dominant error source. This would reframe the problem from "improve step-time prediction" to "improve workload specification fidelity" — a much more tractable calibration target.

### Why It Differs from Prior Attempts

Round 2 used the inference-perf workload pipeline exclusively. This idea uses trace replay to bypass workload generation entirely, which has never been tested in the StepML research context. It attacks BC-1 (the primary blocker) directly at its source.

### Go Integration Path

No new Go code needed — uses existing TraceV2 replay infrastructure (`sim/workload/replay.go`). The step-time model remains the Round 2 regime ensemble with overhead floor.

### LatencyModel Methods Covered

All 5 methods inherited from Round 2's best model. This idea's contribution is isolating simulation-level error from step-time error.

---

## Idea 2: Total-Context Linear Model with Feature Scaling

### Title
Replace the 2-feature step-time model with a 3-feature formulation (`new_tokens + total_context`) using proper feature scaling, informed by FairBatching's validated ±1.3% per-step approach.

### Rationale

Round 2's central finding was that KV features (kv_sum, kv_max, kv_mean) are **counter-productive in raw linear regression** (+20.5pp worse). The root cause was identified: kv_sum ranges 0–64,000+, causing Ridge coefficient instability. However, problem.md explicitly states this is "a formulation problem, not a feature problem" — the features have signal but raw linear regression cannot handle their dynamic range.

The literature strongly validates this diagnosis:

- **FairBatching** [arXiv 2510.14392, 2025]: Uses a 3-coefficient model: `batch_time = a + b * total_new_tokens + c * total_context`, where `total_context = sum of context lengths across all requests in the batch`. This is functionally equivalent to BLIS's `kv_sum` feature (sum of ProgressIndex across batch). FairBatching achieves ±1.3% per-step error with this formulation — dramatically better than Round 2's 64.4%.
- **BiScale** [arXiv 2602.18755, 2025]: Uses **summary statistics** (sum, mean, std) of sequence lengths as features for histogram gradient boosting trees, achieving 2.7–2.9% MAPE.
- **HERMES/MIST** [arXiv 2504.09775, 2025]: Uses quadratic terms (`prefill_tokens²`) in polynomial regression, achieving 2.5% avg error. The quadratic term captures compute-scaling nonlinearity.

The key insight is that the problem is not about *which* features to use, but *how* to formulate them. FairBatching's `total_context` is essentially `kv_sum` but fit with a proper formulation (direct linear coefficient, not Ridge regularization over a 64K range). The solution is:

1. **Feature normalization**: StandardScaler or per-model normalization constants
2. **Proper formulation**: `a + b*new_tokens + c*total_context` (3 coefficients, no regularization needed)
3. **Nonlinear fallback**: If linear fails, gradient boosting trees (BiScale approach) with summary statistics

### Method Sketch

1. **Baseline**: Refit the Round 2 regime ensemble WITHOUT KV features (the 43.9% MAPE version that was actually better than the 64.4% KV version)
2. **FairBatching formulation**: Fit `step_time = a + b*(prefill_tokens + decode_tokens) + c*kv_sum` per model, with NO regularization (OLS) — kv_sum is the total context. Compare vs. Round 2
3. **StandardScaler formulation**: Normalize all features to zero-mean unit-variance per model, then fit Ridge. This should fix the coefficient instability without changing the feature set
4. **Log-feature transform**: Transform kv_sum → log1p(kv_sum) before fitting, keeping target in raw space. This compresses the 0–64K range while preserving linearity in the target
5. **Gradient boosting trees**: XGBoost with features = {prefill_tokens, decode_tokens, kv_sum, kv_mean, kv_std} per model (Round 1 showed 34% MAPE with XGBoost). Use summary statistics per BiScale
6. **34B deep-dive**: Investigate CodeLlama-34B specifically — examine step-time distribution, batch composition, and KV patterns to understand why it's the worst model
7. **Mixed-heavy threshold sweep**: Test prefill thresholds at 64, 128, 256 to optimize regime boundaries
8. **BLIS E2E validation**: Run BLIS with the best formulation using overhead floor

### Expected Outcome

We expect the FairBatching formulation or StandardScaler to reduce per-step MAPE from 64.4% (with KV) / 43.9% (without KV) to < 15%, and ITL error from 33.6% to < 10% mean. The 3-coefficient model adds exactly one coefficient to the current blackbox, keeping Go integration trivial.

### Why It Differs from Prior Attempts

Round 2 tried raw KV features in Ridge regression (failed due to dynamic range). This idea:
- Uses the **FairBatching formulation** (total_context as a single linear term) instead of multiple KV features
- Uses **feature scaling** (StandardScaler/log-transform) instead of raw features
- Uses **XGBoost with summary statistics** (BiScale approach) as a nonlinear fallback
- None of these were tested in Round 2

### Go Integration Path

**Coefficient export** for the 3-coefficient linear model (trivial — add one coefficient to the existing artifact JSON). For XGBoost: the existing Go tree evaluator in `sim/latency/stepml.go` already supports future tree ensemble loading.

### LatencyModel Methods Covered

Primary: **StepTime** (3-feature linear or XGBoost with overhead floor).
Secondary: All other methods inherited from Round 2.

---

## Idea 3: End-to-End Calibration via Direct E2E Objective

### Title
Jointly calibrate all LatencyModel coefficients by directly minimizing BLIS E2E mean error using black-box optimization, bypassing per-component accuracy entirely.

### Rationale

Round 2's key finding was that per-step MAPE is a poor proxy for E2E error: 64.4% per-step MAPE → 427.8% E2E error, but 33.6% ITL. The relationship between per-component accuracy and E2E accuracy is nonlinear and mediated by simulation dynamics (queueing, batch formation, work-conservation). This suggests optimizing coefficients directly against the E2E objective.

Literature support:

- **Trace-driven simulation calibration** is standard practice in DES research. The key principle: calibrate against the metrics you care about (E2E, TTFT, ITL), not intermediate metrics (per-step MAPE).
- **AIConfigurator** [arXiv 2601.06288, NVIDIA 2025]: Uses "empirical correction factors" and "piecewise linear functions" for TTFT, achieving 3.35% error in practical regions. The correction factors are calibrated against end-to-end metrics, not per-component accuracy.
- **Block** [arXiv 2508.03611, Salesforce 2025]: Uses "online calibration" — when actual decode lengths exceed predictions, dynamically adjusts by "using the monitored decode length plus an additional 10 steps." Simple multiplicative/additive corrections calibrated against observed error.

The approach: treat the BLIS simulator as a black-box function mapping coefficients → E2E error, and use derivative-free optimization (Nelder-Mead, CMA-ES, or Bayesian optimization) to find the coefficient set that minimizes total E2E error across all experiments.

This differs from Round 2's attempted joint Bayesian optimization (Idea 1, H2) in two critical ways:
1. Round 2's BO was blocked because the *base model* (87.4% per-step MAPE) was too inaccurate. We now have a much better base (43.9% without KV, plus overhead floor producing 33.6% ITL) — close enough for E2E calibration to be effective.
2. Round 2 attempted to optimize per-step accuracy then validate E2E. This idea optimizes E2E *directly*, skipping per-step accuracy as an intermediate goal.

### Method Sketch

1. **Start from Round 2's best model**: Regime ensemble + overhead floor (the configuration producing 33.6% mean ITL)
2. **Define the objective**: `f(params) = mean over experiments of |BLIS_E2E(params) - GT_E2E| / GT_E2E`, where params includes all StepML artifact fields (step_time coefficients, overhead, secondary method constants)
3. **Augment with trace replay** (from Idea 1): Run E2E calibration using trace replay to isolate step-time/scheduling contributions from workload-spec errors
4. **Optimize with CMA-ES** (Covariance Matrix Adaptation Evolution Strategy): Derivative-free, handles noisy objectives, scales to 10–50 parameters. Budget: ~100–200 BLIS evaluations per experiment (each takes 2–5 seconds)
5. **Per-model calibration**: Optimize separate coefficient sets per model (mandatory per prior findings)
6. **Additive correction exploration**: Test if simple additive/multiplicative corrections to TTFT and ITL (like AIConfigurator's correction factors) can close the remaining gap
7. **Cross-validate**: Hold out one workload, calibrate on two, validate on the third — prevent overfitting to specific workload patterns
8. **Robustness check**: Verify calibrated coefficients generalize to unseen request patterns

### Expected Outcome

With trace replay (eliminating workload-spec error) + E2E optimization starting from the 33.6% ITL base, we expect to achieve < 15% mean E2E error. If combined with Idea 2's improved step-time model (providing a better starting point), < 10% is achievable.

### Why It Differs from Prior Attempts

Round 2's joint BO (Idea 1, H2) was never run because the base model was too inaccurate (87.4% MAPE). This idea:
- Starts from a **much better base** (43.9% per-step, 33.6% ITL with overhead floor)
- Uses **trace replay** to eliminate workload-spec error (never tested)
- Optimizes the **E2E objective directly** (not per-step MAPE as a proxy)
- Uses **CMA-ES** instead of Bayesian optimization (better for noisy multi-experiment objectives)
- Includes **additive correction factors** (inspired by AIConfigurator) for TTFT/ITL

### Go Integration Path

**Coefficient export** — the calibrated coefficients are written to the same StepML artifact JSON format. The Go evaluator (`sim/latency/stepml.go`) loads them unchanged. Any new correction parameters (e.g., TTFT multiplier) would be simple additions to the artifact schema.

### LatencyModel Methods Covered

All 5 methods: **StepTime** (calibrated coefficients), **QueueingTime** (calibrated), **OutputTokenProcessingTime** (calibrated), **SchedulingProcessingTime** (calibrated from E2E objective), **PreemptionProcessingTime** (calibrated from E2E objective).

---

## Idea Comparison Matrix

| Dimension | Idea 1: Trace Replay | Idea 2: Total-Context Model | Idea 3: E2E Calibration |
|---|---|---|---|
| Primary target | BC-1 (TTFT mismatch) | BC-2 (KV scaling) + BC-3 (34B) | BC-1 (E2E objective) + all BCs |
| Attack vector | Simulation fidelity | Step-time model formulation | Joint coefficient optimization |
| Novel technique | Lifecycle trace replay | FairBatching 3-coeff formulation | CMA-ES on BLIS E2E objective |
| Risk level | Low (infrastructure exists) | Medium (formulation untested) | High (many BLIS runs, overfitting risk) |
| Expected E2E impact | Isolates error source; may reduce TTFT 100x | Reduces ITL further; no direct TTFT impact | Directly optimizes E2E; highest ceiling |
| Go integration | None needed (trace replay) | 1 new coefficient | Same artifact format |
| Complementarity | Provides trace-replay baseline for Ideas 2+3 | Provides better starting point for Idea 3 | Consumes improvements from Ideas 1+2 |

## Recommended Execution Order

Ideas 1 and 2 are **independent and complementary** — they should run in parallel (WP3). Idea 3 should consume the best outputs from Ideas 1 and 2 as its starting point. In practice, Idea 3's sub-hypotheses should use Idea 1's trace replay and Idea 2's improved step-time model.

## Literature References

1. Agrawal et al. "Revati: Efficiently Serving LLM Reasoning and Non-Reasoning Workloads via Time-Warp Simulation." arXiv 2601.00397 (2025).
2. Agrawal et al. "Vidur: A Large-Scale Simulation Framework For LLM Inference." MLSys 2024, arXiv 2405.05465.
3. Kumbhare et al. "HERMES: Holistic Evaluation and Reasoning for Multi-stage Enterprise LLM Serving." arXiv 2504.09775 (2025).
4. Zha et al. "FairBatching: Fair Scheduling for LLM Serving with Budget Pacing." arXiv 2510.14392 (2025).
5. Zhong et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." OSDI 2024, arXiv 2401.09670.
6. Patel et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." arXiv 2311.18677 (2023).
7. Lee et al. "BiScale: A Bi-Level Scaling Framework for LLM Inference." arXiv 2602.18755 (2025).
8. Yang et al. "Block: An Efficient Simulator for Scalable LLM Inference." arXiv 2508.03611 (2025).
9. Agrawal et al. "Sarathi-Serve: Efficient LLM Inference with Chunked Prefills and Stall-Free Scheduling." OSDI 2024, arXiv 2403.02310.
10. Yu et al. "LLMServingSim 2.0: Operator-Level LLM Serving Simulator." arXiv 2602.23036 (2025).
11. Liu et al. "TokenSim: Simulating LLM Inference Using Token-Level Metrics." arXiv 2503.08415 (2025).
12. Agarwal et al. "AIConfigurator: Automated Configuration for LLM Inference." arXiv 2601.06288 (2025).
