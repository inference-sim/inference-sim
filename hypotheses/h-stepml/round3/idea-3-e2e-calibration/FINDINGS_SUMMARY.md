# FINDINGS SUMMARY: Idea 3 — End-to-End Calibration via Direct E2E Objective

**Date:** 2026-02-27
**Status:** Partially successful — CMA-ES achieves 15.1% mean E2E with 4/10 < 10%; specific models (codellama-34b, llama-2-70b) reach production quality

## 1. Idea Recap

This idea jointly calibrates all LatencyModel coefficients by directly minimizing BLIS E2E mean error using CMA-ES (Covariance Matrix Adaptation Evolution Strategy), bypassing per-component accuracy. The approach treats BLIS as a black-box function mapping artifact parameters → E2E error and optimizes per-model coefficient vectors (~16-23 parameters) against trace-replay experiments.

**Starting point:** Round 2's best regime ensemble artifacts with the overhead floor mechanism (56.2% mean E2E with trace replay, 9.5% mean ITL).

**Key insight:** Per-step MAPE is a poor proxy for E2E error (Round 2: 64.4% step → 427.8% E2E). This idea optimizes E2E *directly* rather than using step accuracy as an intermediate.

## 2. Sub-Hypothesis Results Table

| Sub-Hypothesis | Status | Key Metric | Takeaway |
|---|---|---|---|
| **H1:** CMA-ES + trace replay → < 15% E2E | **PARTIALLY SUPPORTED** | E2E 15.1% (target < 15%) | 3.7x improvement; 4/10 < 10%; narrowly missed target |
| **H2:** Workload-spec mode → < 50% E2E | **NOT TESTED** | All BLIS runs failed | Profile file mismatch between repos; infrastructure gap |
| **H3:** Additive corrections → -5pp E2E | **REFUTED for E2E** | E2E 17.2% (+2.1pp worse) | Corrections conflict with CMA-ES absorption; TTFT improved to 9.4% |

## 3. Best BLIS E2E Result — Full Per-Experiment Error Table

**H1 results (CMA-ES + trace replay):**

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

**Best per-model CMA-ES results (optimization across workloads):**

| Model | Initial E2E | Optimized E2E | Improvement |
|---|---|---|---|
| llama-2-7b | 60.9% | 22.2% | 2.7x |
| llama-2-70b | 58.2% avg | **8.3%** | 7.0x |
| mixtral-8x7b-v0-1 | 53.0% avg | 25.9% | 2.0x |
| codellama-34b | 55.8% avg | **7.9%** | 7.1x |

## 4. What Worked (Specific Techniques)

1. **CMA-ES for black-box DES calibration** — Treating BLIS as a black-box function and optimizing against E2E directly was effective. CMA-ES found parameter settings that reduce E2E error 2-7x per model, despite noisy multi-experiment objectives and ~20 parameters.

2. **Per-model optimization** — Mandatory. Each model has fundamentally different step-time characteristics (7B: 3.9ms overhead, Mixtral: 9.1ms overhead). A single global optimization would fail.

3. **Overhead floor as the primary knob** — The step_overhead_us parameter (overhead floor) controls the dominant timing characteristic. CMA-ES consistently tuned this upward (e.g., llama-2-7b: 3,897 → 5,051μs, +30%) to match real vLLM step cycle times.

4. **Scheduling time as secondary knob** — For larger models (llama-2-70b), scheduling_processing_time_us doubled (415 → 821μs), capturing CPU overhead that the step-time model misses.

5. **TTFT additive corrections (H3)** — While not helpful for E2E, applying per-model additive corrections to QueueingTime reduced TTFT error from 67.6% → 9.4%. This is the first time TTFT has been below 10% in this research.

## 5. What Failed and Why (Root Causes)

1. **E2E optimization sacrifices ITL accuracy** — CMA-ES optimizes the E2E objective, which can increase per-token overhead (output_token_processing_time_us) to match total E2E at the expense of ITL accuracy. ITL worsened from 9.5% → 87.4%. This is the fundamental tension: E2E = TTFT + Σ(ITL), so optimizing total while ignoring components produces imbalanced solutions.

2. **Mixtral and llama-2-7b resist optimization** — These models achieved only 25.9% and 22.2% E2E respectively, vs. 8.3% and 7.9% for the others. Root causes:
   - **llama-2-7b:** Only 1 experiment (roleplay). CMA-ES has no cross-workload validation and may overfit.
   - **Mixtral:** MoE architecture has higher per-step variance and less predictable batch behavior.

3. **H2 (workload-spec mode) completely failed** — Profile file mismatch between BLIS-research and inference-sim repos. The experiment directory naming conventions differ, preventing automatic matching.

4. **H3 corrections conflict with H1 optimization** — CMA-ES already "absorbed" the TTFT deficit into its coefficient values. Applying post-hoc TTFT corrections double-counts the adjustment, worsening E2E by 2.1pp.

## 6. Binding Constraints

| Constraint | Status | Evidence |
|---|---|---|
| **BC-1: TTFT/simulation fidelity mismatch** | Partially resolved by H3 | TTFT 9.4% with corrections; 67.6% without |
| **BC-NEW: E2E ↔ ITL tradeoff** | **NEW BINDING CONSTRAINT** | Optimizing E2E worsens ITL (9.5% → 87.4%). Cannot optimize both with the same coefficient set. |
| **BC-2: KV feature scaling** | Not addressed | KV features are in the artifact but CMA-ES doesn't specifically target them |
| **BC-3: CodeLlama-34B anomaly** | Resolved | E2E 7.9% mean — CMA-ES found good coefficients for this model |
| **BC-4: Mixtral MoE variance** | Partially addressed | 25.9% mean — better than baseline but above target |

## 7. Data Insights Discovered

1. **E2E and ITL are in tension.** The LatencyModel interface has 5 methods that all contribute to E2E, but their contributions are nonlinearly coupled through simulation dynamics (queueing, batch formation). Optimizing total E2E via coefficient tuning can achieve good E2E by offsetting errors across methods — but this produces arbitrarily bad per-method metrics.

2. **CMA-ES convergence is model-dependent.** Larger models (70B, 34B) converge faster to low E2E because they have more experiments (3 each) providing a smoother objective landscape. llama-2-7b (1 experiment) converges slowly and to a worse optimum.

3. **The overhead floor works bidirectionally.** CMA-ES sometimes *decreased* step_overhead_us (codellama-34b: 6,673 → 5,998, -10%) while increasing scheduling time. This suggests the initial overhead floor was already close to optimal for some models, and the remaining error is in secondary method timing.

4. **TTFT bias is systematic and model-independent.** All models under-predict TTFT by 3-4x (60-83% error). This is a simulation-level characteristic, not a step-time model issue. The additive correction (16-61ms per model) fully compensates.

5. **Output token processing time is the hidden parameter.** CMA-ES increased output_token_processing_time_us dramatically for llama-2-7b (0 → 1,899μs). This effectively adds ~1.9ms per output token, which is physically implausible but mathematically optimal for E2E. This reveals that the LatencyModel interface lacks a "simulation overhead" parameter — the optimizer uses OutputTokenProcessingTime as a proxy for missing dynamics.

## 8. Comparison to Baseline

| Metric | R2 Workload-Spec | R3 Idea 1 (Trace) | R3 Idea 3 H1 (CMA-ES) | R3 Idea 3 H3 (Corrections) |
|---|---|---|---|---|
| Mean E2E error | 427.8% | 56.2% | **15.1%** | 17.2% |
| Mean TTFT error | 31,906% | 78.8% | 67.6% | **9.4%** |
| Mean ITL error | 33.6% | **9.5%** | 87.4% | 81.1% |
| E2E < 10% | 1/10 | 0/10 | **4/10** | 4/10 |
| E2E < 15% | — | 0/10 | **5/10** | 4/10 |

**Net assessment:** CMA-ES E2E optimization achieves the best E2E accuracy in the research program (15.1% mean, 4/10 < 10%). However, it sacrifices ITL accuracy and TTFT accuracy to achieve this. The combination of H1 (E2E optimization) + H3 (TTFT correction) produces the most balanced result: 17.2% E2E + 9.4% TTFT, but ITL remains at 81.1%.

## 9. Go Integration Feasibility

**Already integrated.** The Go changes needed for this experiment were:

| File | Change | Lines |
|---|---|---|
| `sim/config.go` | Added `StepMLModelPath` field + `WithStepMLModel` builder | ~12 |
| `sim/latency/latency.go` | Added StepML dispatch in factory (after roofline, before blackbox) | ~7 |
| `cmd/root.go` | Added `--stepml-model` CLI flag + wiring | ~8 |

Total: ~27 lines of Go changes. The optimized artifacts use the same JSON schema as the Round 2 artifacts and load through the existing `sim/latency/stepml.go` evaluator unchanged.

**Artifact format:** Standard StepML JSON with version 2, step_time_regimes, and calibrated overhead/secondary method values. No new fields needed.

**Performance:** Each BLIS simulation run with StepML takes 2-10 seconds (depending on experiment size). The CMA-ES optimization requires 96-152 evaluations per model, totaling ~30 minutes per model.

## 10. Generalization Results

### H4: Leave-One-Workload-Out (LOWO) — Part A Per-Workload Breakdown

**Status: PARTIALLY SUPPORTED**

CMA-ES-optimized artifacts evaluated per-workload:

| Model Group | Codegen | General | Roleplay | Range (pp) | Mean |
|---|---|---|---|---|---|
| llama-2-7b | — | — | 22.2% | — | 22.2% |
| llama-2-70b | **3.8%** | 16.2% | **6.9%** | 12.4 | 9.0% |
| codellama-34b | **5.0%** | **1.2%** | 17.5% | 16.3 | 7.9% |
| mixtral-8x7b-v0-1 | 30.9% | 12.7% | 34.2% | 21.5 | 25.9% |

Dense models generalize well across workloads (9-16pp range). Mixtral shows highest workload variance (21.5pp). 8/10 experiments within 2x of aggregate (15.1%), but mixtral-roleplay (34.2%) slightly exceeds. Part B (LOWO CMA-ES re-optimization) not tested (requires ~4.5 hours of CMA-ES runs).

### H5: Leave-One-Model-Out (LOMO) — Cross-Model Artifact Transfer

**Status: SUPPORTED** — Mean best-donor E2E = **14.8%** (threshold <50%)

Cross-model transfer matrix (E2E %):

| Donor → Target | codellama-34b | llama-2-70b | llama-2-7b | mixtral-8x7b |
|---|---|---|---|---|
| codellama-34b | — | 21.2% | 24.3% | 21.2% |
| llama-2-70b | 24.7% | — | 94.0% | **5.1%** |
| llama-2-7b | 40.4% | 53.8% | — | 53.8% |
| mixtral-8x7b | **11.9%** | 26.0% | **20.9%** | — |

**Best donor per target:**
- codellama-34b ← mixtral (11.9%)
- llama-2-70b ← codellama-34b (21.2%)
- llama-2-7b ← mixtral (20.9%)
- mixtral ← llama-2-70b (**5.1%**)

**Key insight:** CMA-ES artifacts have dramatically better cross-model transfer than per-step models because they capture simulation-level dynamics (overhead floor, scheduling overhead) that are partially model-independent. Mixtral is the best universal donor; 70B→Mixtral transfer (5.1%) actually outperforms Mixtral's own in-distribution result (25.9%).
