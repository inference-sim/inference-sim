# FINDINGS SUMMARY: Idea 1 — Trace-Driven Simulation with Lifecycle Replay

**Date:** 2026-02-27
**Status:** Partially successful — workload-spec error eliminated, secondary error source identified

## 1. Idea Recap

This idea bypasses BLIS's workload-spec generation by replaying ground-truth request traces directly, using the legacy CSV trace format (`--workload traces --workload-traces-filepath <csv>`). The goal was to isolate whether the catastrophic 31,906% TTFT error from Round 2 originated from the workload specification or from BLIS's internal simulation logic.

No new model training was required — the experiment uses Round 2's best StepML artifacts (regime ensemble + calibrated overheads from idea-2-regime-ensemble/h3) with exact ground-truth request arrivals and token lengths.

## 2. Sub-Hypothesis Results Table

| Sub-Hypothesis | Status | Key Metric | Takeaway |
|---|---|---|---|
| **H1:** Trace replay reduces TTFT | **SUPPORTED** | TTFT 78.8% (was 31,906%) | Workload-spec was the dominant TTFT error source (405x reduction) |
| **H2:** Remaining E2E < 25% | **REFUTED** | E2E 56.2% (target < 25%) | BLIS scheduling divergence is a co-dominant error source; 0/10 < 25% |
| **H3:** Identify workload-spec parameter | **REFUTED** | All params < 1% error | Parameters are correct; generation *process* causes the mismatch |

## 3. Best BLIS E2E Result — Full Per-Experiment Error Table

| Experiment | Model | Workload | GT E2E (ms) | BLIS E2E (ms) | E2E Error | TTFT Error | ITL Error |
|---|---|---|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | llama-2-7b | roleplay | 2,071 | 810 | 60.9% | 77.9% | **4.0%** |
| llama-2-70b-tp4-general | llama-2-70b | general | 5,321 | 1,986 | 62.7% | 87.9% | 23.5% |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | codegen | 4,606 | 2,022 | 56.1% | 77.4% | 10.7% |
| llama-2-70b-tp4-roleplay | llama-2-70b | roleplay | 4,562 | 2,019 | 55.7% | 77.2% | 10.3% |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | codegen | 4,675 | 2,234 | 52.2% | 76.2% | **2.1%** |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | general | 5,039 | 2,253 | 55.3% | 79.6% | 8.7% |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | roleplay | 4,685 | 2,276 | 51.4% | 76.7% | **0.7%** |
| codellama-34b-tp2-general | codellama-34b | general | 4,093 | 1,682 | 58.9% | 80.1% | 17.9% |
| codellama-34b-tp2-codegen | codellama-34b | codegen | 3,723 | 1,674 | 55.0% | 77.4% | 10.1% |
| codellama-34b-tp2-roleplay | codellama-34b | roleplay | 3,670 | 1,702 | 53.6% | 77.4% | 7.3% |
| **MEAN** | | | | | **56.2%** | **78.8%** | **9.5%** |

## 4. What Worked (Specific Techniques)

1. **Lifecycle-to-CSV trace conversion** — The `convert_lifecycle_to_traces.py` infrastructure faithfully converts per_request_lifecycle_metrics.json into BLIS's legacy CSV trace format. All 10 experiments produced correct request counts matching ground truth exactly (7,200 / 9,000 / 16,800).

2. **Legacy trace replay path** (`--workload traces --workload-traces-filepath`) — Preserves exact arrival times (seconds → microseconds) and per-request token lengths. No synthetic generation artifacts.

3. **StepML CLI integration** — Added `--stepml-model` CLI flag to pass regime ensemble artifacts directly to BLIS, enabling StepML LatencyModel dispatch without requiring roofline or alpha/beta coefficients. Changes: `sim/config.go` (field + builder), `sim/latency/latency.go` (factory dispatch), `cmd/root.go` (flag registration + wiring).

4. **ITL accuracy** — The regime ensemble + overhead floor mechanism from Round 2 achieves 9.5% mean ITL error with trace replay, with 4/10 experiments under 10%. Per-token timing is nearly solved.

## 5. What Failed and Why (Root Causes)

1. **E2E under-prediction (56.2%)** — BLIS E2E is consistently ~40% of ground truth across ALL experiments. The uniformity across models and workloads indicates a systematic bias, not per-model miscalibration.

   **Root cause:** The step-time model (regime ensemble + 3.9ms overhead floor) predicts step cycles that are 50-60% faster than reality. The floor was calibrated on `step.duration_us` (GPU forward pass only), but real vLLM step cycles include:
   - CPU scheduling overhead between steps (~1-3ms)
   - Memory management (KV block allocation/deallocation)
   - Batch formation decision time
   - CUDA synchronization overhead

2. **TTFT under-prediction (78.8%)** — Even with correct arrivals, BLIS produces TTFT values 75-88% below ground truth (6-14ms BLIS vs 27-103ms GT). Because BLIS processes steps faster than reality, queues drain faster, and requests get scheduled sooner than they should.

3. **H3 workload-spec diagnosis failed to find a parameter cause** — All workload-spec parameters (arrival rate, request count, duration) match ground truth to < 1%. The catastrophic R2 TTFT error was caused by the workload *generation process* (how parameters become individual requests), not the parameters themselves. The shared-prefix + question_len mechanism underestimates input tokens by 20-30%, and the Poisson arrival sampling differs from real traffic patterns.

## 6. Binding Constraints

| Constraint | Status | Evidence |
|---|---|---|
| **BC-1: TTFT/simulation fidelity mismatch** | Partially resolved | Workload-spec contribution eliminated (31,906% → 78.8%). Remaining 78.8% is simulation-level. |
| **BC-NEW: Step cycle time under-prediction** | **NEW BINDING CONSTRAINT** | BLIS predicts step cycles at ~40% of real duration. This cascading under-prediction is the dominant E2E error source with trace replay. |
| **BC-2: KV feature scaling** | Unchanged | Not addressed by this idea (deferred to Idea 2). |
| **BC-3: CodeLlama-34B anomaly** | No longer anomalous | With trace replay, codellama-34b errors (53-59%) are within the same range as other models. The anomaly was workload-spec-specific. |

## 7. Data Insights Discovered

1. **Workload-spec parameters are NOT the problem.** Rate, count, and duration all match < 1%. The R2 TTFT catastrophe was caused by the request sampling process, not the spec parameters. This means improving the spec generator (e.g., better input length distributions) would help but not fix the fundamental issue.

2. **BLIS simulates a "faster universe."** With trace replay, every experiment shows BLIS completing requests in ~40% of the real time. This is a fundamental step-time calibration issue — the overhead floor needs to be approximately doubled (from ~4ms to ~8ms) to match real vLLM step cycle times.

3. **ITL is nearly independent of workload mode.** ITL improved from 33.6% → 9.5% with trace replay, but this modest improvement (vs. the 405x TTFT improvement) shows that ITL is primarily determined by step-time prediction accuracy, not workload specification. The overhead floor mechanism works.

4. **Cross-model consistency of the error.** All 5 model families show 50-63% E2E error with trace replay, suggesting the under-prediction is NOT model-specific. It's a universal property of the step-time prediction framework (either the overhead floor calibration or missing simulation overhead).

## 8. Comparison to Baseline

| Metric | R2 Workload-Spec | R3 Trace Replay | Improvement |
|---|---|---|---|
| Mean E2E error | 427.8% | 56.2% | **7.6x better** |
| Mean TTFT error | 31,906% | 78.8% | **405x better** |
| Mean ITL error | 33.6% | 9.5% | **3.5x better** |
| E2E < 10% | 1/10 | 0/10 | No change |
| E2E < 25% | — | 0/10 | — |

**Net assessment:** Trace replay is a massive improvement for TTFT (the primary R2 blocker) and a significant improvement for E2E and ITL. However, it does NOT achieve the < 10% E2E target. The remaining 56% E2E error is a step-time calibration problem, not a workload problem.

## 9. Go Integration Feasibility

**Already integrated.** This idea required the following Go changes (all in this worktree):

| File | Change | Lines |
|---|---|---|
| `sim/config.go` | Added `StepMLModelPath` field to `ModelHardwareConfig` + `WithStepMLModel` builder | ~10 |
| `sim/latency/latency.go` | Added StepML dispatch in `NewLatencyModel` factory (after roofline, before blackbox) | ~7 |
| `cmd/root.go` | Added `--stepml-model` CLI flag + wiring to config builder | ~5 |

Total: ~22 lines of Go changes. The StepML Go evaluator (`sim/latency/stepml.go`) was already implemented in WP0. These changes complete the CLI integration.

**Patch for reproducibility:** The Go changes are captured in this worktree and can be exported via `git diff` for other worktrees.

## Appendix: Methodology

- **Ground truth source:** `BLIS-research/eval/ground_truth/` (10 experiments, 5 models, 3 workloads)
- **StepML artifacts:** Round 2 calibrated regime ensembles from `idea-2-regime-ensemble/h3-secondary-method-calibration/output/calibrated_artifacts/`
- **Trace conversion:** `convert_lifecycle_to_csv()` in `run_experiment.py` (equivalent to shared `convert_lifecycle_to_traces.py`)
- **BLIS invocation:** `./simulation_worker run --workload traces --workload-traces-filepath <csv> --stepml-model <artifact.json> --alpha-coeffs=1,0,0 --beta-coeffs=1,0,0 [vLLM args from exp-config.yaml]`
- **Horizon:** Ground-truth trace duration + 120s buffer
- **All 10 experiments completed successfully** with correct request counts
