# Idea 3: End-to-End Calibration via Direct E2E Objective

## Overview

Jointly calibrate all LatencyModel coefficients by directly minimizing BLIS E2E mean error using black-box optimization, bypassing per-component accuracy entirely. Attacks all binding constraints by optimizing the metric that matters (E2E error) rather than intermediate metrics (per-step MAPE).

**LatencyModel methods covered:** All 5: StepTime (calibrated coefficients), QueueingTime (calibrated), OutputTokenProcessingTime (calibrated), SchedulingProcessingTime (calibrated), PreemptionProcessingTime (calibrated).

**Go integration path:** Coefficient export — calibrated values written to the same StepML artifact JSON format. No new Go code.

## Prior Round Context

- **Round 2 (Idea 1, H2):** Joint Bayesian optimization was **never run** because the base model (87.4% per-step MAPE) was too inaccurate. This blocked the whole idea.
- **Round 2 (Idea 2, H3):** Secondary method calibration contributed **0.0pp** improvement — 200-400us corrections invisible against 100%+ errors.
- **Round 2 key finding:** Per-step MAPE is a poor proxy for E2E error. 64.4% per-step → 427.8% E2E but 33.6% ITL. The relationship is nonlinear and mediated by simulation dynamics.
- **Literature:** AIConfigurator [arXiv 2601.06288] uses "empirical correction factors" calibrated against E2E for 3.35% TTFT error. Block [arXiv 2508.03611] uses "online calibration" with simple additive corrections ("+10 steps"). Trace-driven DES calibration is standard practice: calibrate against metrics you care about, not intermediates.
- **Key difference from Round 2:** We now have a **much better starting point** — Round 2's best model produces 33.6% mean ITL (5/10 < 10%) with the overhead floor. Round 2's BO attempt started from 87.4% per-step MAPE (too poor for calibration).

## Training Strategy

Per-model optimization (mandatory). Each model's coefficients optimized independently. Cross-workload validation: optimize on 2 workloads, validate on the held-out 3rd. Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for derivative-free optimization.

**Data split:** Hold-one-workload-out cross-validation. For each model: train on 2 workloads, validate on 1. Rotate to get 3 folds.

---

## Sub-Hypothesis H1: Trace Replay + E2E Optimization Baseline

### Claim

Starting from Round 2's best coefficients, CMA-ES optimization of StepML artifact parameters against BLIS E2E mean error (using trace replay from Idea 1) achieves < 15% mean E2E error across all experiments.

### Rationale

This hypothesis combines Idea 1's trace replay (eliminating workload-spec error) with direct E2E optimization. By removing the workload-spec mismatch and optimizing directly against E2E, we attack both the PRIMARY (BC-1) and SECONDARY (BC-2, BC-3) binding constraints simultaneously. The starting point (33.6% mean ITL with overhead floor) is close enough to the target that derivative-free optimization should converge.

The parameter space is small enough for CMA-ES: per model, we optimize ~8–12 parameters (step_time regime coefficients, overhead, secondary method constants). With 10 experiments × 3 workloads per model, each evaluation takes ~5 seconds (BLIS run). Budget: 200 evaluations per model × 4 models = ~67 minutes total.

### Method

1. **Setup:** Generate trace replay files for all experiments (from Idea 1). Define the objective: `f(params) = mean_i |BLIS_E2E_i(params) - GT_E2E_i| / GT_E2E_i` where i indexes experiments for this model
2. **Parameter vector:** For each model: [regime coefficients (decode intercept, decode slopes, mixed intercept, mixed slopes), step_overhead_us, step_overhead_per_req_us, queueing_time_intercept, queueing_time_slope, output_token_processing_time_us, scheduling_processing_time_us, preemption_processing_time_us]
3. **Initial point:** Round 2's best coefficients per model
4. **Bounds:** Each parameter bounded to [0.1× initial, 10× initial] (or [0, 10× initial] for non-negative params). Overhead floor always >= 1000us.
5. **Optimize with CMA-ES:** sigma0 = 0.3 × initial values, population size 10, budget 200 evaluations per model
6. **Evaluate:** Full per-experiment E2E/TTFT/ITL error table with optimized coefficients

### Refutation Criteria

- **Supported:** Mean E2E error < 15% with trace replay. At least 5/10 experiments < 10%.
- **Refuted:** Mean E2E error > 30% even with trace replay + optimization — BLIS scheduling divergence is too large for coefficient tuning to overcome.

### Diagnostics

- Convergence plot: E2E error vs evaluation count per model
- Optimized vs initial coefficient values (identify which parameters moved most)
- Per-experiment E2E/TTFT/ITL error before and after optimization
- Sensitivity analysis: which parameters have largest marginal impact on E2E

---

## Sub-Hypothesis H2: Workload-Spec Mode E2E Optimization

### Claim

CMA-ES optimization in workload-spec mode (not trace replay) achieves < 50% mean E2E error, a 8.5× improvement over Round 2's 427.8%.

### Rationale

Even without trace replay, E2E optimization might partially compensate for workload-spec mismatch by finding coefficients that produce the correct *mean* E2E despite incorrect intermediate behavior (arrival patterns, TTFT). The optimizer can effectively "absorb" workload-spec errors into the coefficient values. This is useful because: (a) trace replay requires ground-truth data, while workload specs are always available; (b) if the optimizer can partially compensate, it provides a practical deployment path.

However, we expect this to be less effective than H1 because the optimizer cannot correct for exponentially divergent TTFT from workload-spec arrival rate mismatch (per DistServe's M/D/1 model).

### Method

1. Same CMA-ES setup as H1, but using workload-spec mode (inference-perf profiles from Round 2)
2. Objective: `f(params) = mean_i |BLIS_E2E_i(params) - GT_E2E_i| / GT_E2E_i` with workload-spec BLIS runs
3. Same parameter vector and bounds as H1
4. Budget: 200 evaluations per model
5. Cross-validate: optimize on 2 workloads, validate on the held-out 3rd

### Refutation Criteria

- **Supported:** Mean E2E error < 50% with workload-spec mode (8.5× improvement over 427.8%).
- **Refuted:** Mean E2E error > 200% — E2E optimization cannot compensate for workload-spec mismatch.

### Diagnostics

- Comparison: H1 (trace replay) vs H2 (workload spec) per experiment
- Cross-validation results: do optimized coefficients generalize to held-out workloads?
- Coefficient drift analysis: how much do coefficients change from the initial point? Large drift suggests overfitting to workload-spec artifacts.

---

## Sub-Hypothesis H3: Additive Correction Factors

### Claim

Simple additive or multiplicative correction factors applied to TTFT and ITL (calibrated from trace replay results) close at least 5pp of remaining E2E error beyond what H1 achieves.

### Rationale

AIConfigurator uses "empirical correction factors modeled as piecewise linear functions" for TTFT, achieving 3.35% error. Block uses "+10 steps" as a simple additive correction. Even if BLIS's scheduling differs from vLLM in systematic ways (e.g., consistently scheduling prefills one step later), a simple correction factor can compensate.

The correction approach is: `corrected_TTFT = TTFT_BLIS × alpha_ttft + beta_ttft`, where alpha_ttft and beta_ttft are calibrated per model. Similarly for ITL. These corrections are applied as part of the LatencyModel's QueueingTime and OutputTokenProcessingTime methods.

### Method

1. From H1 results, compute per-experiment TTFT and ITL residuals: `residual = BLIS_metric - GT_metric`
2. Fit linear correction models per model: `corrected = alpha * predicted + beta`
3. Implement corrections as adjustments to QueueingTime (for TTFT) and OutputTokenProcessingTime (for ITL)
4. Re-run BLIS with corrected LatencyModel
5. Measure improvement over H1's uncorrected results

### Refutation Criteria

- **Supported:** Corrections reduce mean E2E error by at least 5pp vs H1's baseline.
- **Refuted:** Corrections reduce mean E2E error by < 2pp — systematic bias is not the dominant residual, and remaining error is stochastic/per-request variance.

### Diagnostics

- TTFT and ITL residual distributions per model (are they systematic or random?)
- Correction factor values (alpha, beta) per model
- Pre- and post-correction per-experiment E2E/TTFT/ITL errors
- Scatter plot: BLIS-predicted vs ground-truth TTFT and ITL
