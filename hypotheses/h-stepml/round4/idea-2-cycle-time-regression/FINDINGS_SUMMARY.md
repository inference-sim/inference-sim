# FINDINGS_SUMMARY: Idea 2 — Data-Driven Step Cycle Time Model with Explicit CPU Overhead Separation

**Round:** 4
**Date:** 2026-03-02
**Status:** Refuted — All sub-hypotheses refuted or partially refuted

---

## 1. Idea Recap

This idea proposed training the step-time model on **actual inter-token intervals (ITI) from lifecycle data** instead of `step.duration_us` (GPU forward-pass only), eliminating the need for a separate overhead floor. The core insight: real step cycle times include CPU overhead (scheduling, sync, memory management) that `step.duration_us` doesn't capture, causing BLIS's "faster universe" problem (completing requests in ~40% of real time).

The approach used a FairBatching-style 3-coefficient OLS regression (`cycle_time = a + b*new_tokens + c*kv_sum`) per model, with overhead floors calibrated from H1's ITI-derived cycle time measurements.

---

## 2. Sub-Hypothesis Results Table

| # | Sub-Hypothesis | Status | Key Metric | Takeaway |
|---|---|---|---|---|
| H1 | Cycle-time extraction from lifecycle data | **Partially Refuted** | 19.9% match rate (target >50%) | ITI-to-step matching fails due to dual 10% sampling; but cycle_time/duration ratio of 1.64x validates overhead hypothesis |
| H2 | FairBatching cycle-time regression → BLIS E2E | **Refuted** | 452.2% mean E2E (target <25%) | TTFT errors from workload-spec mode dominate; ITL 29.4% is within R2 range |
| H3 | LOMO generalization | **Refuted** | 132.7% mean MAPE (target <80%) | 70B fold 295.5% MAPE; Mixtral best at 40.6% |
| H4 | LOWO generalization | **Refuted** | 163.5% mean MAPE (target <50%) | All folds fail; roleplay worst at 341.6% |

---

## 3. Best BLIS E2E Result — Full Per-Experiment Error Table

**Mode:** Workload-spec (inference_perf expansion) — NOT trace replay

| Experiment | E2E Error | TTFT Error | ITL Error | Status |
|---|---|---|---|---|
| codellama-34b-tp2-roleplay | 27.2% | 17.3% | 45.8% | Best E2E |
| llama-2-70b-tp4-roleplay | 28.1% | 164.2% | 39.5% | |
| llama-2-70b-hf-tp4-codegen | 50.5% | 6,520.8% | 43.3% | |
| mixtral-8x7b-tp2-roleplay | 50.8% | 77.2% | 0.7% | Best ITL |
| mixtral-8x7b-tp2-codegen | 51.5% | 76.4% | 2.1% | |
| llama-2-7b-tp1-roleplay | 52.5% | 78.5% | 4.0% | |
| codellama-34b-tp2-codegen | 101.8% | 9,223.9% | 77.9% | |
| mixtral-8x7b-tp2-general | 554.9% | 44,487.4% | 8.7% | |
| llama-2-70b-tp4-general | 1,676.6% | 88,557.1% | 22.6% | |
| codellama-34b-tp2-general | 1,927.8% | 154,921.4% | 49.5% | Worst |
| **MEAN** | **452.2%** | **30,412.4%** | **29.4%** | |

---

## 4. What Worked (Specific Techniques)

1. **H1-confirmed overhead floor values**: ITI-derived cycle times for 70B (9,670us vs R2's 8,029-8,203us) and 34B (7,149us vs R2's 6,673us) are more accurate overhead estimates. The 70B value is 17-20% higher than R2, consistent with R3's "faster universe" finding that BLIS needs ~doubling of overhead.

2. **Mixtral universality confirmed again**: Mixtral achieves 40.6% LOMO (best fold), 25-34% per-model LOWO across all workloads. MoE architecture is inherently more workload-universal, consistent with R2's finding.

3. **Per-model overhead floor mechanism**: ITL mean of 29.4% (3/10 below 10%) confirms the overhead floor is the dominant driver of ITL accuracy. This is consistent with R2's 33.6% mean ITL.

4. **FairBatching kv_sum feature**: Contributes positively for 34B (r=0.725) and 70B (r=0.838) on validation data. The 3-coefficient formulation is a sound per-step model.

5. **StepML Go integration pipeline**: Successfully wired `--stepml-model` CLI flag → factory dispatch → StepMLLatencyModel with regime-based artifacts. This infrastructure is reusable across ideas.

---

## 5. What Failed and Why (Root Causes)

1. **Per-step cycle-time extraction (H1)**: Dual 10% sampling (traces.json + lifecycle data) means joint coverage of ~1-3%, insufficient for per-step matching. Only 3/10 experiments yielded any matches. **Root cause:** Data collection methodology — would need 100% trace sampling or synchronized timestamps to enable this approach.

2. **BLIS E2E via workload-spec mode (H2)**: 452.2% mean E2E is dominated by workload-spec TTFT errors (30,412% mean). This is the same bottleneck as R2 (427.8% / 31,906%). Without trace replay, step-time improvements are invisible in E2E. **Root cause:** Workload-spec generation creates fundamentally different request arrival patterns than real experiments. R3 solved this with trace replay.

3. **LOMO generalization (H3)**: 132.7% mean, 295.5% worst fold (70B). Cross-model transfer fails because step-time scales differ by 10-100x across models, and the FairBatching regression has no model-level normalization. **Root cause:** Per-step models learn model-specific absolute scales; need model-agnostic features (FLOPs/token, parameter count) for transfer.

4. **LOWO generalization (H4)**: 163.5% mean, 341.6% worst fold (roleplay). Workload type changes batch composition patterns, which the linear model can't capture. **Root cause:** Workload-dependent batch distributions shift the overhead/compute balance point non-linearly.

---

## 6. Binding Constraints

| ID | Constraint | Impact | Mitigation |
|---|---|---|---|
| BC-DATA-1 | Dual 10% sampling prevents per-step cycle-time extraction | Cannot train on real cycle times; must use step.duration_us + overhead floor proxy | Need synchronized timestamps or higher sampling rate |
| BC-TTFT-1 | Workload-spec mode TTFT errors (30K%) dominate E2E | Step-time model quality is irrelevant without trace replay | Use trace replay mode (as R3 demonstrated) |
| BC-SCALE-1 | 10-100x step-time scale variation across models | Linear regression cannot transfer across model scales | Need model-level normalization or per-model training |
| BC-WKLD-1 | Workload-dependent batch distributions | Linear model cannot capture overhead/compute balance shifts | Need nonlinear models or workload-aware features |

---

## 7. Data Insights Discovered

1. **Cycle-time ratio 1.64x**: Real step cycles are 64% longer than GPU-only `step.duration_us` (median across 3 matched experiments). This is lower than R3's "~2x faster universe" claim, suggesting the overhead varies significantly by model and workload.

2. **Decode-only overhead ratio 1.72 > mixed ratio 1.53**: CPU overhead is proportionally larger for decode-only steps (small batch, overhead-dominated) than mixed-batch steps (prefill computation fills the gap). This validates the `max(overhead, compute)` floor mechanism.

3. **H1-confirmed overhead values**: 70B overhead 9,670us (was 8,029-8,203 in R2), 34B overhead 7,149us (was 6,673 in R2). H1's ITI-derived values are 7-20% higher than R2's ground-truth ITL-derived values, suggesting R2's values slightly underestimate real overhead.

4. **kv_sum is zero-weighted for 7B and Mixtral**: The FairBatching regression gives kv_sum zero coefficient for these two models, suggesting that for small models (7B) and MoE models (Mixtral), KV cache pressure doesn't meaningfully affect step time at the observed batch sizes.

---

## 8. Comparison to Baseline

| Metric | WP0 Baseline | R2 Best | R3 Best (CMA-ES) | This Idea (H2) |
|---|---|---|---|---|
| Mean E2E | 115.0% | 427.8% | **15.1%** | 452.2% |
| Mean TTFT | 102.9% | 31,906% | **67.6%** | 30,412% |
| Mean ITL | 134.6% | **33.6%** | 87.4% | 29.4% |
| Mode | blackbox | workload-spec | **trace replay** | workload-spec |
| LOMO | — | 108.6% | **14.8% (E2E)** | 132.7% |
| LOWO | — | 117.4% | partial | 163.5% |

**Key takeaway:** This idea produces results comparable to R2's workload-spec baseline but is not competitive with R3's CMA-ES + trace replay (15.1% E2E). The approach is fundamentally limited by using workload-spec mode instead of trace replay.

---

## 9. Go Integration Feasibility

**Fully feasible.** During this experiment, we:
1. Added `StepMLModelPath` field to `ModelHardwareConfig` with `WithStepMLModel()` builder
2. Added `--stepml-model` CLI flag to `cmd/root.go`
3. Modified factory dispatch in `sim/latency/latency.go` to check StepML before alpha/beta validation
4. Fixed workload spec validation to accept `inference_perf` section without clients
5. All Go tests pass (`go test ./sim/latency/... -run TestStepML` — 9 tests pass)

The StepML artifact format supports regime-based models (decode-only vs mixed) with per-model overhead floors. Integration path is coefficient export — zero new Go dependencies.

---

## 10. Generalization Results

### LOMO Table: Leave-One-Model-Out Cross-Validation (4-fold)

| Fold (Holdout) | Test MAPE | Pearson r | Test Samples | Target <80% |
|---|---|---|---|---|
| CodeLlama-34B | 70.0% | 0.657 | 24,100 | PASS |
| Llama-2-70B | 295.5% | 0.811 | 19,412 | FAIL |
| Llama-2-7B | 124.6% | 0.169 | 15,216 | FAIL |
| Mixtral-8x7B | 40.6% | 0.575 | 19,088 | PASS |
| **MEAN** | **132.7%** | | | **2/4 PASS** |

**Assessment:** REFUTED — 70B fold (295.5%) exceeds 150% threshold. Only 2/4 folds pass. Similar to R2's 108.6% (same order of magnitude). The FairBatching formulation provides no LOMO improvement over R2's regime Ridge. Cross-model transfer remains structurally limited by per-step absolute scale differences.

### LOWO Table: Leave-One-Workload-Out Cross-Validation (3-fold)

| Fold (Holdout) | Per-Model MAPE | Global MAPE | Pearson r | Target <50% |
|---|---|---|---|---|
| codegen | 92.0% | 143.8% | 0.705 | FAIL |
| general | 56.8% | 110.7% | 0.686 | FAIL |
| roleplay | 341.6% | 177.5% | 0.293 | FAIL |
| **MEAN** | **163.5%** | | | **0/3 PASS** |

**Assessment:** REFUTED — All folds exceed 50% target, roleplay fold (341.6%) exceeds 100% threshold. Worse than R2's 117.4%. Mixtral is the only model that generalizes across workloads (25-34% per-model MAPE), consistent with its MoE architecture capturing workload-invariant compute physics.
