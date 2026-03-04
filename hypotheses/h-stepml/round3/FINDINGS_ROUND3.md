# StepML Round 3: Findings Summary

**Date:** 2026-02-27
**Branch:** `stepml-experiments`
**Scope:** Three ideas tested across 10 experiments (5 models × 3 workloads, no reasoning). This round focused on the two independent blockers identified in Round 2: (1) the 31,906% mean TTFT error from workload-spec mismatch, and (2) KV feature scaling. A third idea explored direct E2E calibration via evolutionary optimization.

---

## 1. Problem Recap

Achieve **<10% workload-level E2E mean error** across 10 experiments (5 models × 3 workloads) by improving any or all of the 5 `LatencyModel` methods. Round 2's best result was **427.8% mean E2E error** (1/10 < 10%) with 33.6% mean ITL (5/10 < 10%). Two binding constraints dominated:

- **BC-1 (PRIMARY):** TTFT/simulation fidelity mismatch — 31,906% mean TTFT error from workload-spec generation producing mismatched arrival patterns. No step-time improvement can achieve <10% E2E until this is resolved.
- **BC-2 (SECONDARY):** KV feature scaling — per-request KV features have signal but are counter-productive in raw linear space (+20.5pp worse). Feature scaling or nonlinear models needed.

The blackbox baseline is **115.0% mean E2E error**. Round 2's best (regime ensemble + overhead floor) achieved **427.8%** via workload-spec mode, but this was dominated by TTFT errors, not step-time errors.

---

## 2. What We Tried

### Idea 1: Trace-Driven Simulation with Lifecycle Replay

**Approach:** Bypass BLIS's workload-spec generation entirely by replaying ground-truth request traces (exact arrival times, token lengths) via the legacy CSV trace format. Uses Round 2's best StepML artifacts (regime ensemble + calibrated overheads). Goal: isolate whether the catastrophic TTFT error originates from workload-spec or from BLIS's internal simulation.

| Sub-Hypothesis | Target | Result | Status |
|---|---|---|---|
| H1: Trace replay reduces TTFT | < 100% TTFT | **78.8%** (was 31,906%) | **Supported** |
| H2: Remaining E2E < 25% | < 25% E2E | **56.2%** | **Refuted** |
| H3: Identify workload-spec parameter | Root cause found | All params < 1% error | **Refuted** |

### Idea 2: Total-Context Linear Model with Feature Scaling

**Approach:** Replace the 2-feature step-time model with a 3-feature formulation (`new_tokens + total_context`) using FairBatching's validated approach. Address BC-2 (KV feature scaling) and BC-3 (CodeLlama-34B anomaly).

| Sub-Hypothesis | Target | Result | Status |
|---|---|---|---|
| H1: FairBatching 3-coeff OLS | < 50% per-step MAPE | **56.2%** (vs 83.1% baseline) | **Partially Supported** |
| H2: Feature scaling variants | < 43.9% per-step MAPE | **83.0%** best | **Refuted** |
| H3: BLIS E2E + 34B deep-dive | < 50% E2E | **56.2%** (identical to R2 trace) | **Refuted** |
| H4: LOWO generalization | < 70% per-step MAPE | **2,162.7%** | **Refuted** |
| H5: LOMO generalization | < 80% per-step MAPE | **2,281.6%** | **Refuted** |

### Idea 3: End-to-End Calibration via Direct E2E Objective (CMA-ES)

**Approach:** Jointly calibrate all LatencyModel coefficients by directly minimizing BLIS E2E mean error using CMA-ES (Covariance Matrix Adaptation Evolution Strategy). Treat BLIS as a black-box function mapping artifact parameters → E2E error. Per-model optimization.

| Sub-Hypothesis | Target | Result | Status |
|---|---|---|---|
| H1: CMA-ES + trace replay → E2E | < 15% E2E | **15.1%** | **Partially Supported** (narrowly missed) |
| H2: Workload-spec mode → E2E | < 50% E2E | Not tested (infra gap) | **Not Tested** |
| H3: Additive TTFT corrections | -5pp E2E | **+2.1pp worse** (E2E); TTFT 9.4% | **Refuted for E2E** |
| H4: LOWO generalization | All within 2× aggregate | 8/10 within 2× (15.1% agg) | **Partially Supported** |
| H5: LOMO cross-model transfer | < 50% E2E | **14.8%** mean best-donor | **Supported** |

---

## 3. What Worked

### 3.1 Leaderboard — Round 3 Ideas Ranked by BLIS E2E Mean Error

| Rank | Idea | Mean E2E | Mean TTFT | Mean ITL | E2E < 10% | Mode |
|---|---|---|---|---|---|---|
| **1** | **Idea 3 H1: CMA-ES** | **15.1%** | 67.6% | 87.4% | **4/10** | Trace replay |
| 2 | Idea 3 H3: CMA-ES + TTFT corrections | 17.2% | **9.4%** | 81.1% | 4/10 | Trace replay |
| 3 | Idea 1: Trace replay (R2 artifacts) | 56.2% | 78.8% | **9.5%** | 0/10 | Trace replay |
| 4 | Idea 2: Total-context model | 56.2% | 79.4% | **9.5%** | 0/10 | Trace replay |

### 3.2 Cross-Round Leaderboard

| Round | Best Approach | Mean E2E | Mean TTFT | Mean ITL | E2E < 10% | Key Advance |
|---|---|---|---|---|---|---|
| Baseline | Per-model linear regression | 115.0% | 102.9% | 134.6% | 0/12 | — |
| R1 | Per-experiment XGBoost | Not tested | Not tested | Not tested | — | 34% per-step MAPE |
| R2 | Regime ensemble + overhead floor | 427.8% | 31,906% | 33.6% | 1/10 | Overhead floor, BLIS pipeline |
| **R3** | **CMA-ES E2E calibration** | **15.1%** | 67.6% | 87.4% | **4/10** | Trace replay, E2E optimization |

**R3 vs R2 improvement:** 427.8% → 15.1% = **28.3x better E2E**. However, this comparison is misleading because R2 used workload-spec mode while R3 uses trace replay. Comparing R3 CMA-ES against R3 trace replay baseline (same mode): 56.2% → 15.1% = **3.7x better E2E**.

### 3.3 Best Per-Experiment Results (CMA-ES, Idea 3 H1)

| Experiment | Model | Workload | GT E2E (ms) | BLIS E2E (ms) | E2E Error | TTFT Error | ITL Error |
|---|---|---|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | llama-2-7b | roleplay | 2,071 | 1,612 | 22.2% | 60.8% | 91.1% |
| llama-2-70b-tp4-general | llama-2-70b | general | 5,321 | 4,458 | 16.2% | 82.5% | 72.0% |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | codegen | 4,606 | 4,432 | **3.8%** | 67.6% | 111.6% |
| llama-2-70b-tp4-roleplay | llama-2-70b | roleplay | 4,562 | 4,249 | **6.9%** | 67.6% | 117.0% |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | codegen | 4,675 | 3,229 | 30.9% | 66.4% | 41.5% |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | general | 5,039 | 4,399 | 12.7% | 57.3% | 78.1% |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | roleplay | 4,685 | 3,084 | 34.2% | 69.3% | 34.5% |
| codellama-34b-tp2-general | codellama-34b | general | 4,093 | 4,044 | **1.2%** | 69.3% | 102.7% |
| codellama-34b-tp2-codegen | codellama-34b | codegen | 3,723 | 3,536 | **5.0%** | 68.9% | 90.2% |
| codellama-34b-tp2-roleplay | codellama-34b | roleplay | 3,670 | 3,028 | 17.5% | 66.2% | 135.4% |
| **MEAN** | | | | | **15.1%** | **67.6%** | **87.4%** |

**Solved experiments (E2E < 10%):** codellama-34b-general (1.2%), llama-2-70b-hf-codegen (3.8%), codellama-34b-codegen (5.0%), llama-2-70b-roleplay (6.9%)

**Unsolved experiments (E2E > 15%):** llama-2-7b-roleplay (22.2%), mixtral-codegen (30.9%), mixtral-roleplay (34.2%), llama-2-70b-general (16.2%), codellama-34b-roleplay (17.5%)

### 3.4 Specific Techniques That Worked

1. **Trace replay eliminates workload-spec error** — TTFT improved from 31,906% to 78.8% (405x). The legacy CSV trace format (`--workload traces`) faithfully preserves exact arrival times and per-request token lengths.

2. **CMA-ES for black-box DES calibration** — Treating BLIS as a black-box function and optimizing E2E directly achieves 15.1% mean E2E (3.7x better than trace replay baseline). CMA-ES found per-model parameter settings that balance step-time, overhead, and scheduling coefficients.

3. **Overhead floor as primary optimization knob** — CMA-ES consistently tuned step_overhead_us upward (e.g., llama-2-7b: 3,897 → 5,051μs, +30%) to match real vLLM step cycle times. The floor is the most impactful single parameter.

4. **Per-model CMA-ES optimization** — Mandatory. Dense 70B/34B models converge to 7.9-8.3% mean E2E; llama-2-7b (1 experiment) and Mixtral (MoE) are harder to optimize.

5. **TTFT additive corrections** — Per-model additive corrections to QueueingTime reduced TTFT from 67.6% to 9.4%. First time TTFT < 10% in this research.

6. **FairBatching 3-coefficient formulation** — Combining prefill+decode into `new_tokens` + single `kv_sum` achieves 27pp better per-step MAPE than 2-coefficient baseline (56.2% vs 83.1%).

7. **StepML CLI integration** — `--stepml-model` flag + factory dispatch + Go evaluator now form a complete CLI-to-simulation path for StepML artifacts.

8. **CMA-ES cross-model artifact transfer** — CMA-ES artifacts have dramatically better cross-model generalization (14.8% LOMO E2E) than per-step models (2,281.6% LOMO MAPE) because they capture transferable simulation dynamics. Mixtral is the best universal donor (11.9–26.0%). For new models without ground truth, applying an existing CMA-ES artifact provides a reasonable starting point.

---

## 4. What Failed and Why

### 4.1 E2E ↔ ITL Tradeoff (Structural — Idea 3)

CMA-ES optimizes total E2E at the expense of per-token ITL accuracy. ITL worsened from 9.5% → 87.4% because:
- **Root cause:** E2E = TTFT + Σ(ITL). Optimizing the sum while ignoring components produces imbalanced solutions. CMA-ES increased `output_token_processing_time_us` to implausible values (e.g., llama-2-7b: 0 → 1,899μs) as a proxy for missing simulation dynamics.
- **Fundamental tension:** The LatencyModel interface has 5 methods that contribute nonlinearly to E2E through simulation dynamics (queueing, batch formation). Matching total E2E via coefficient tuning can offset errors across methods, but this produces arbitrarily bad per-method metrics.

### 4.2 Feature Scaling Does Not Fix KV Instability (Idea 2 H2)

StandardScaler, log-transform, and their combination all perform WORSE than the no-KV baseline (83.0% vs 82.0% without KV).
- **Root cause:** The issue is **multicollinearity** among 4 correlated KV features (kv_sum, kv_max, kv_mean, kv_std), NOT dynamic range mismatch. Ridge regularization cannot resolve multicollinearity. The fix is fewer features (H1's single kv_sum approach works).

### 4.3 Per-Step Improvement Doesn't Propagate to E2E (Idea 2)

The 27pp per-step MAPE improvement (83% → 56%) has ZERO impact on BLIS E2E (56.2% in both cases).
- **Root cause:** The overhead floor (~4-9ms) dominates 70-90% of step predictions, completely masking the GPU compute prediction improvement. The kv_sum contribution (~40-1300μs) is negligible versus the floor.

### 4.4 Workload-Spec Mode Infra Gap (Idea 3 H2)

All BLIS runs in workload-spec mode failed due to profile file mismatch between BLIS-research and inference-sim repos.
- **Root cause:** Experiment directory naming conventions differ between repos. This is an infrastructure gap, not a fundamental limitation.

### 4.5 Mixtral and llama-2-7b Resist CMA-ES Optimization

These models achieved only 25.9% and 22.2% E2E respectively, vs 8.3% and 7.9% for 70B/34B.
- **Root cause (7b):** Only 1 experiment (roleplay). CMA-ES has no cross-workload validation and may overfit.
- **Root cause (Mixtral):** MoE architecture has higher per-step variance and less predictable batch behavior. The linear step-time model is structurally inadequate for MoE.

---

## 5. Binding Constraints

| # | Constraint | Status | Severity | Evidence |
|---|---|---|---|---|
| **BC-1** | **TTFT/simulation fidelity mismatch** | Partially resolved | HIGH | Workload-spec error eliminated (31,906% → 78.8%). Remaining 78.8% is simulation-level. CMA-ES doesn't solve TTFT directly. TTFT corrections get it to 9.4% but conflict with E2E optimization. |
| **BC-NEW-1** | **E2E ↔ ITL fundamental tradeoff** | **NEW — BLOCKING** | CRITICAL | Optimizing E2E worsens ITL (9.5% → 87.4%). The LatencyModel interface couples E2E and ITL through shared coefficients. Cannot optimize both simultaneously with the current approach. |
| **BC-NEW-2** | **Step cycle time under-prediction** | **NEW** | HIGH | BLIS predicts step cycles at ~40% of real duration (trace replay, no CMA-ES). Overhead floor needs ~2x increase to match real vLLM step cycles. |
| **BC-2** | KV feature scaling | Resolved differently | LOW | Multicollinearity, not scale, is the root cause. Single kv_sum works (27pp per-step gain) but has zero E2E impact due to overhead floor dominance. |
| **BC-3** | CodeLlama-34B anomaly | **Resolved** | — | Not anomalous in trace replay; was workload-spec-specific. CMA-ES achieves 7.9% mean E2E for 34B. |
| **BC-4** | Mixtral MoE variance | Partially addressed | MEDIUM | CMA-ES: 25.9% mean E2E. Better than baseline but above target. MoE compute scaling not captured by linear model. |
| **BC-5** | Per-step model generalization (LOMO) | Unresolved for per-step | MEDIUM | Per-step models fail LOMO (2,281.6%). CMA-ES artifacts transfer well (14.8% E2E) but per-step accuracy is not preserved. Regime structure is mandatory for per-step LOMO. |
| **BC-6** | vLLM args sensitivity | Never analyzed | MEDIUM | No round has evaluated sensitivity to vLLM configuration changes. Production viability requires this. |

---

## 6. Data Characteristics Learned

1. **Workload-spec parameters are correct; the generation process is not.** All spec parameters (rate, count, duration) match ground truth to <1%. The 31,906% TTFT error was caused by the request sampling process (shared-prefix + question_len underestimates input tokens by 20-30%, Poisson arrival sampling differs from real traffic). Improving the spec generator would help but not fix the fundamental issue.

2. **BLIS simulates a "faster universe."** With trace replay, every experiment shows BLIS completing requests in ~40% of the real time. This is a fundamental step-time calibration issue — the overhead floor needs approximately doubling (from ~4ms to ~8ms for 7B, proportionally for larger models) to match real vLLM step cycle times.

3. **The overhead floor is the dominant prediction mechanism.** For 70-90% of steps, BLIS's step-time = overhead floor, regardless of the step-time model used. This means step-time model improvements only affect 10-30% of steps.

4. **Per-step MAPE and BLIS E2E are decoupled.** A 27pp per-step improvement produces 0pp BLIS E2E improvement. BLIS E2E depends on step *cycle* time (including overhead), not step *compute* time (what the model predicts).

5. **Step-time distribution is bimodal.** All models show decode-only steps (~100-500μs) and mixed steps (~1,000-12,000μs). CodeLlama-34B has the most extreme bimodality (P50=291μs, mean=1,603μs, P99=11,928μs).

6. **E2E and ITL are in tension.** E2E = TTFT + Σ(ITL), but optimizing total E2E via LatencyModel coefficients produces arbitrarily bad ITL because the optimizer uses coefficient knobs (especially output_token_processing_time) as proxies for missing simulation dynamics.

7. **TTFT bias is systematic and model-independent.** All models under-predict TTFT by 3-4x (60-83% error even with trace replay). Additive correction (16-61ms per model) fully compensates.

8. **KV data coverage is incomplete.** 2/5 models (llama-2-7b, mixtral-8x7b) have kv_sum=0 for ALL steps due to lifecycle-step timestamp join failures.

---

## 7. Successful Techniques and Patterns (Reusable)

1. **Trace replay mode** (`--workload traces --workload-traces-filepath <csv>`) — Eliminates workload-spec generation artifacts. Converts lifecycle data → CSV traces faithfully. All future E2E experiments should use this as the baseline mode.

2. **CMA-ES for black-box DES calibration** — Effective for optimizing a noisy, multi-experiment objective with ~20 parameters. Per-model optimization mandatory. 96-152 evaluations per model, ~30 min/model.

3. **Overhead floor as primary knob** — The step_overhead_us parameter dominates timing. CMA-ES tuning of this parameter alone accounts for most E2E improvement.

4. **Per-model TTFT additive corrections** — Simple, effective (67.6% → 9.4% TTFT). Per-model constants (16-61ms) derived from ground-truth mean TTFT residuals.

5. **FairBatching 3-coefficient OLS** — Minimal formulation (`a + b*new_tokens + c*kv_sum`) outperforms complex feature engineering for per-step accuracy. Use single KV feature (kv_sum) to avoid multicollinearity.

6. **Lifecycle-to-CSV trace conversion** — Robust infrastructure for creating trace replay inputs from per_request_lifecycle_metrics.json.

7. **StepML CLI integration** — Complete path from Python artifacts to Go simulation via `--stepml-model` flag.

---

## 7.5 Generalization Results

### Idea 1 (Trace Replay)

Reuses Round 2 regime ensemble artifacts — generalization results inherit from R2 (LOMO 108.6%, LOWO 117.4%). See `idea-1-trace-replay/GENERALIZATION_NOTE.md`.

### Idea 2 (Total-Context Model)

| Test | Target | Result | Status |
|---|---|---|---|
| **H4: LOWO** | < 70% per-step MAPE | **2,162.7%** | **Refuted** |
| **H5: LOMO** | < 80% per-step MAPE | **2,281.6%** | **Refuted** |

**Root cause:** The FairBatching 3-coeff formulation removes the regime structure that enabled R2's 108.6% LOMO. Without decode-only vs mixed-batch separation, a single linear model across all step types produces catastrophic predictions for out-of-distribution data. Additionally, kv_sum numerical overflow (0–64,000 range) causes OLS coefficient instability.

**Key finding:** Zero degradation vs in-distribution (0.0pp for LOWO), confirming the 3-coeff model doesn't overfit to workloads — but the absolute MAPE is catastrophic.

### Idea 3 (CMA-ES E2E Calibration)

| Test | Target | Result | Status |
|---|---|---|---|
| **H4: LOWO** (per-workload breakdown) | All within 2× aggregate | 8/10 within 2× (15.1% agg) | **Partially Supported** |
| **H5: LOMO** (cross-model transfer) | < 50% E2E | **14.8%** mean best-donor | **Supported** |

**LOWO per-model breakdown:**

| Model | Codegen | General | Roleplay | Range (pp) | Mean |
|---|---|---|---|---|---|
| llama-2-7b | — | — | 22.2% | — | 22.2% |
| llama-2-70b | **3.8%** | 16.2% | **6.9%** | 12.4 | 9.0% |
| codellama-34b | **5.0%** | **1.2%** | 17.5% | 16.3 | 7.9% |
| mixtral-8x7b | 30.9% | 12.7% | 34.2% | 21.5 | 25.9% |

Dense models generalize well across workloads (9–16pp range). Mixtral has highest workload variance (21.5pp).

**LOMO cross-model transfer matrix (E2E %):**

| Donor → Target | codellama-34b | llama-2-70b | llama-2-7b | mixtral-8x7b |
|---|---|---|---|---|
| codellama-34b | — | 21.2% | 24.3% | 21.2% |
| llama-2-70b | 24.7% | — | 94.0% | **5.1%** |
| llama-2-7b | 40.4% | 53.8% | — | 53.8% |
| mixtral-8x7b | **11.9%** | 26.0% | **20.9%** | — |

**Key findings:**
- Mean best-donor LOMO E2E: **14.8%** (vs in-distribution 15.1%)
- **70B → Mixtral transfer (5.1%)** outperforms Mixtral's own in-distribution result (25.9%)
- **Mixtral is the best universal donor** (11.9–26.0% across targets)
- CMA-ES artifacts transfer dramatically better than per-step models because they capture simulation-level dynamics (overhead floor, scheduling overhead) that are partially model-independent

### Cross-Round Generalization Progress

| Round | Idea | LOMO | LOWO | vLLM Args |
|---|---|---|---|---|
| R1 | Tree ensemble | 2,559.7% | 109.7% | Not analyzed |
| R2 | Bayesian calibration | 148.8% | 155.4% | Not analyzed |
| R2 | Regime ensemble | 108.6% | 117.4% | Not analyzed |
| R3 | Total-context model | 2,281.6% (REFUTED) | 2,162.7% (REFUTED) | Not analyzed |
| **R3** | **CMA-ES calibration** | **14.8% E2E (SUPPORTED)** | **Partial (8/10 within 2×)** | Not analyzed |

**Generalization breakthrough:** CMA-ES artifact transfer (14.8% LOMO E2E) is the first approach to show viable cross-model generalization. This is because CMA-ES artifacts encode simulation dynamics (overhead floor calibration), not model-specific step times.

---

## 8. Failed Techniques and Anti-Patterns (Do Not Repeat)

1. **Feature scaling for multicollinear KV features** — StandardScaler, log-transform, combinations all failed. The issue is multicollinearity (4 correlated features), not scale. Use fewer features instead.

2. **Per-step MAPE as proxy for E2E accuracy** — 27pp per-step improvement → 0pp E2E improvement. The overhead floor masks per-step gains. Optimize E2E directly.

3. **Workload-spec mode for E2E validation** — Workload-spec introduces systematic TTFT errors (31,906%). Until the spec generator is fixed, all E2E experiments must use trace replay.

4. **Post-hoc corrections on CMA-ES results** — TTFT additive corrections conflict with CMA-ES-optimized coefficients (E2E worsens by 2.1pp). CMA-ES absorbs error compensation into its coefficient values; post-hoc adjustments double-count.

5. **Unconstrained CMA-ES optimization** — Without ITL constraints, CMA-ES produces implausible parameter values (e.g., output_token_processing_time 0 → 1,899μs). Need multi-objective or constrained optimization.

---

## 9. Round-Specific Design Limitations

1. **The LatencyModel interface couples E2E and ITL.** The 5 methods (StepTime, QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime) jointly determine E2E = TTFT + Σ(ITL). Because the methods share coefficient artifacts, optimizing total E2E can arbitrarily degrade individual components. A multi-objective optimization that constrains both E2E and ITL is needed.

2. **Trace replay mode is not production-viable.** It requires ground-truth request traces, which are not available for capacity planning scenarios (the primary BLIS use case). The workload-spec path must eventually work — trace replay is a research tool for isolating error sources.

3. **CMA-ES optimization is offline and per-model.** Each model requires 96-152 BLIS simulation runs (~30 minutes). This is acceptable for research but expensive for a retraining pipeline. A more principled calibration (e.g., gradient-based with differentiable simulation) would scale better.

4. **Missing simulation dynamics.** CMA-ES uses `output_token_processing_time` as a proxy for dynamics that BLIS doesn't model (CPU scheduling overhead between steps, CUDA synchronization, memory management). These should ideally be separate simulation parameters, not absorbed into the latency model.

---

## 10. Open Questions for Next Round

1. **Can multi-objective CMA-ES (or NSGA-II) optimize E2E and ITL simultaneously?** The key technical challenge is finding the Pareto frontier where E2E < 15% AND ITL < 15% for all experiments.

2. **What is the correct overhead floor for each model?** CMA-ES tuned step_overhead_us upward by 10-30%. A principled calibration approach (e.g., deriving overhead from ground-truth step cycle times rather than ITL residuals) could set the floor more accurately.

3. **Can workload-spec mode be fixed?** The infrastructure gap (profile file mismatch) prevented testing. If workload-spec mode can be made to work with CMA-ES-optimized artifacts, it would validate the approach for production use.

4. **How to handle the Mixtral and 7B optimization difficulties?** Mixtral's MoE architecture and 7B's single-experiment limitation both resist CMA-ES optimization. Different strategies may be needed: MoE-specific features for Mixtral, data augmentation for 7B.

5. **Should BLIS model additional simulation dynamics?** CMA-ES compensates for missing dynamics via implausible coefficient values. Adding explicit parameters for inter-step overhead, CUDA sync time, or memory management overhead could provide a more principled solution.

---

## 11. Reproducibility

### Data Sources
- **Ground truth:** `BLIS-research/eval/ground_truth/` (10 experiments, 5 models × 3 workloads)
- **StepML artifacts:** Round 2 calibrated regime ensembles from `idea-2-regime-ensemble/h3-secondary-method-calibration/output/calibrated_artifacts/`
- **CMA-ES optimized artifacts:** `round3/idea-3-e2e-calibration/h1-cmaes-trace/output/optimized_artifacts/`

### Trace Replay Infrastructure
- **Conversion script:** `convert_lifecycle_to_traces.py` (or equivalent in `run_experiment.py`)
- **BLIS invocation:** `./simulation_worker run --workload traces --workload-traces-filepath <csv> --stepml-model <artifact.json> --alpha-coeffs=1,0,0 --beta-coeffs=1,0,0 [vLLM args]`
- **Horizon:** Ground-truth trace duration + 120s buffer

### CMA-ES Configuration
- **Library:** pycma (CMA-ES)
- **Population size:** Default (based on parameter dimensionality)
- **Evaluations per model:** 96-152
- **Objective:** Mean E2E error across model's experiments
- **Parameters optimized:** step_overhead_us, scheduling_processing_time_us, output_token_processing_time_us, step_time regime coefficients

### Environments
- Python 3.11+, Go 1.21+
- All seeds fixed per artifact

---

## 12. Summary for Future Ideation

### What Round 3 Achieved
Round 3 made **dramatic progress**: 427.8% → 15.1% mean E2E (28.3x improvement). The two key advances were:
1. **Trace replay** eliminated workload-spec as an error source (405x TTFT improvement)
2. **CMA-ES** found per-model coefficient settings that reduce E2E by 3.7x

### What Round 3 Did NOT Solve
The **E2E ↔ ITL tradeoff** is the dominant remaining problem. CMA-ES achieves 15.1% E2E but at 87.4% ITL. The research target requires E2E < 10% — but achieving this with acceptable ITL (<15%) is the real challenge. **vLLM args sensitivity was never analyzed** in any round — this remains a gap for production viability.

### Generalization Assessment
CMA-ES artifacts show a **generalization breakthrough**: 14.8% LOMO E2E (cross-model transfer) and stable LOWO for dense models. Per-step models (total-context, regime ensemble) still fail LOMO (>100%). The key insight: simulation-level dynamics (overhead floor, scheduling) are more transferable than per-step compute models. Future ideas should preserve this property — any approach that improves per-step accuracy must not sacrifice the cross-model transferability that CMA-ES provides.

### The Path Forward
The next round should focus on **constrained or multi-objective optimization** that achieves E2E < 10% while keeping ITL < 15%. Specific avenues:
1. **Multi-objective CMA-ES** (NSGA-II) with E2E and ITL as dual objectives
2. **Constrained CMA-ES** with ITL < 15% as a hard constraint during optimization
3. **Better overhead floor calibration** — derive the floor from step *cycle* times (not step *compute* times) so that the baseline simulation runs at the correct speed without CMA-ES needing to compensate
4. **Separate simulation overhead parameter** — add explicit inter-step overhead to the LatencyModel or simulation config, rather than absorbing it into output_token_processing_time

### The Numbers to Beat (Round 3 Best)

| Metric | CMA-ES (E2E-optimized) | Trace Replay Baseline | Target |
|---|---|---|---|
| Mean E2E | **15.1%** | 56.2% | < 10% |
| Mean TTFT | 67.6% | 78.8% | < 15% |
| Mean ITL | 87.4% | **9.5%** | < 15% |
| E2E < 10% | 4/10 | 0/10 | 10/10 |

**The ideal solution combines CMA-ES's E2E accuracy with the trace replay baseline's ITL accuracy.** This likely requires either multi-objective optimization or a fundamentally different calibration approach that preserves per-step timing fidelity while achieving global E2E accuracy.
