# Latency Model Fidelity for LLM Inference Simulation

## Problem

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. The **ultimate metric is workload-level E2E mean error < 10%** -- the accuracy of end-to-end request latency predicted by the simulator compared to ground truth.

E2E latency in BLIS is the sum of **6 delay components**, each contributed by the `LatencyModel` interface or emergent from simulation dynamics:

```
E2E(request) = QueueingTime(req)              [1] arrival-to-queue delay
             + (time waiting in WaitQ)         [2] emergent from simulation dynamics
             + SchedulingProcessingTime()      [3] per-request scheduling overhead
             + sum StepTime(batch)             [4] GPU execution per batch step
             + sum OutputTokenProcessingTime() [5] per-token post-processing
             + KV transfer latency             [6] CPU<->GPU offload/reload
             + PreemptionProcessingTime()      [7] if preempted and re-queued
```

### Current Model State

The current `LatencyModel` implementations (blackbox and roofline) have these limitations:

| Component | Current Implementation | Limitation |
|---|---|---|
| **StepTime** | `beta0 + beta1*prefill + beta2*decode` (3 coefficients) | Only 2 features; no KV cache, architecture, or batch composition awareness |
| **QueueingTime** | `alpha0 + alpha1*inputLen` (2 coefficients) | Fixed linear model; doesn't account for system load or batch formation dynamics |
| **OutputTokenProcessingTime** | constant `alpha2` | Single constant for all tokens regardless of context |
| **SchedulingProcessingTime** | **returns 0** | Real scheduling has overhead (queue scanning, KV allocation, priority sorting) |
| **PreemptionProcessingTime** | **returns 0** | Real preemption has cost (KV block release, re-queuing, state bookkeeping) |
| **KV transfer latency** | Separate KV subsystem | Only applies to tiered (GPU+CPU) configurations |

### Round 1 Findings

Round 1 focused exclusively on replacing `StepTime()`. Key results (see `round1/round1_FINDINGS_SUMMARY.md`):
- **Best result:** Per-experiment XGBoost achieved 34.0% avg per-step MAPE (2x better than blackbox)
- **Binding constraint:** Missing per-request KV cache lengths in training data
- **Generalization fails:** Per-model training is mandatory (LOMO 2559.7% MAPE)
- **Critical gap:** Per-step MAPE was the only metric tested; workload-level E2E mean error was NOT evaluated

### Broadened Scope for Round 2

**The goal is to achieve <10% workload-level E2E mean error by improving ANY or ALL of the 5 LatencyModel methods**, not just StepTime. Opportunities include:

1. **StepTime improvement** -- continue from Round 1's XGBoost (34% MAPE), potentially with per-request KV features derived from lifecycle data
2. **QueueingTime calibration** -- the `alpha0 + alpha1*inputLen` model may systematically over/underpredict arrival-to-queue delay, biasing all downstream latencies
3. **SchedulingProcessingTime** -- currently 0, but real scheduling overhead (vLLM's `scheduler.schedule()`) is measurable from ground-truth timing data and scales with queue depth and batch size
4. **OutputTokenProcessingTime** -- currently constant, but may vary with model size, TP degree, or output token position
5. **PreemptionProcessingTime** -- currently 0, but preemption events are observable in ground-truth traces and have real cost
6. **End-to-end calibration** -- even if individual components have errors, calibrating the *composition* of all components to minimize E2E error is valid (e.g., intentional over/underestimation of one component to compensate for systematic bias in another)

**The winning approach must integrate back into BLIS as a Go `LatencyModel` implementation.** The simulator's existing roofline model (analytical FLOPs/bandwidth with MFU lookup) is unaffected and not being replaced.

## Ground-Truth Data

### Overview

122,752 step-level observations from instrumented vLLM v0.15.1 with OpenTelemetry tracing (10% sample rate). 12 experiments: 4 models × 3 workloads, all on H100 80GB GPUs.

### Models and Configurations

| Model | Architecture | TP | Parameters |
|-------|-------------|-----|-----------|
| Llama-2-7B | Dense | 1 | 7B |
| Llama-2-70B | Dense | 4 | 70B |
| CodeLlama-34B | Dense | 2 | 34B |
| Mixtral-8x7B-v0.1 | MoE (8 experts, top-2) | 2 | 46.7B total, ~12.9B active |

### Workloads

general, codegen, roleplay -- each with different input/output length distributions. All runs: `max_model_len=4096`, `max_num_batched_tokens=2048`, `max_num_seqs=128`, chunked prefill enabled, prefix caching enabled.

### Step-Level Features

**Batch computation features** (causal):
- `batch.prefill_tokens` (int) -- prefill tokens in this step
- `batch.decode_tokens` (int) -- decode tokens in this step
- `batch.num_prefill_reqs` (int) -- prefill request count
- `batch.num_decode_reqs` (int) -- decode request count
- `batch.scheduled_tokens` (int) -- total scheduled tokens

**System state features** (potentially spurious):
- `queue.running_depth`, `queue.waiting_depth` -- queue state
- `kv.usage_gpu_ratio`, `kv.blocks_free_gpu`, `kv.blocks_total_gpu` -- KV cache state

**Target**: `step.duration_us` (int) -- wall-clock step execution time in microseconds

### Data Characteristics

Step durations span 3+ orders of magnitude:
- Smallest: ~12 us (small decode-only batches)
- Largest: ~250,000 us (large prefills)
- Roleplay/codegen: mean 160--320 us (short, bursty)

### Known Feature Gaps

1. **No per-request KV cache lengths**: Only aggregate batch tokens available. Attention FLOPs scale with per-request kv_len (H8 showed 12.96x overestimate without per-request KV). The simulator's `ProgressIndex` (input_tokens_processed + output_tokens_generated) is available as a proxy at inference time.
2. **No MoE-specific features**: No active expert count, expert load balance, or tokens-per-expert.
3. **No prefix cache hit information**: `prefill_tokens` may reflect pre- or post-cache-hit count.

### Additional Data Sources

- **Per-request lifecycle data**: Per-token timestamps, input/output token counts -- enables per-request KV length derivation and E2E validation
- **MFU benchmarks** (`bench_data/`): Kernel-level GEMM and attention MFU data by GPU -- useful for physics-informed features

## Baseline Results (WP0)

### BLIS E2E Validation -- Per-Model Linear Regression (THE NUMBER TO BEAT)

Per-model+TP linear regression (one regression per model group, pooling all workloads):
- **Mean E2E error: 115.0%** (0/16 experiments below 10% target)
- Mean TTFT error: 102.9%
- Mean ITL error: 134.6%
- **R4 gate: PASSED** -- blackbox is clearly insufficient, research is justified

Per-model training coefficients (note negative intercepts producing invalid predictions):

| Model+TP | beta0 | beta1 | beta2 | Train MAPE | Train r |
|----------|-------|-------|-------|------------|---------|
| CodeLlama-34B tp=2 | -5525 | 0.421 | 215.1 | 675.8% | 0.918 |
| Llama-2-70B tp=4 | -2633 | 1.306 | 106.4 | 199.2% | 0.934 |
| Llama-2-7B tp=1 | 992 | 3.403 | 7.031 | 610.3% | 0.359 |
| Mixtral-8x7B tp=2 | -8035 | 0.000 | 249.3 | 965.6% | 0.778 |

Per-experiment BLIS E2E errors:

| Experiment | E2E Error | TTFT Error | ITL Error |
|---|---|---|---|
| llama-2-7b-tp1-codegen | 86.2% | 88.0% | 72.3% |
| llama-2-7b-tp1-roleplay | 86.1% | 85.9% | 72.2% |
| llama-2-7b-tp1-general | 90.8% | 92.0% | 81.6% |
| llama-2-70b-tp4-general | 111.8% | 102.0% | 124.1% |
| llama-2-70b-tp4-codegen | 113.6% | 103.6% | 127.5% |
| llama-2-70b-tp4-roleplay | 113.9% | 103.4% | 128.2% |
| mixtral-8x7b-tp2-codegen | 141.3% | 113.7% | 183.5% |
| mixtral-8x7b-tp2-general | 138.5% | 111.7% | 177.9% |
| mixtral-8x7b-tp2-roleplay | 141.9% | 113.3% | 184.7% |
| codellama-34b-tp2-general | 132.3% | 110.3% | 165.3% |
| codellama-34b-tp2-codegen | 135.4% | 111.7% | 171.5% |
| codellama-34b-tp2-roleplay | 136.5% | 111.5% | 173.7% |

### BLIS E2E Validation -- Roofline (Informational Only)

The analytical roofline model (zero calibration, FLOPs/bandwidth-based):
- **Mean E2E error: 4816%** (10/16 completed, 6 timed out at 300s)
- Massively overestimates step time, causing request starvation
- Only Mixtral codegen/roleplay were within 35% (GEMM-dominated workloads)
- The roofline model is informational only -- ideas MUST NOT use roofline as input

### Component Error Attribution (BC-0-6)

StepTime dominates E2E error in all 12 experiments. With alpha=[0,0,0] (blackbox defaults):
- Queueing contribution: 0% (alpha coefficients are zero)
- Output processing: 0% (alpha2 is zero)
- Scheduling: 0 (returns 0)
- Preemption: 0 (returns 0)
- **Step time: 100% of modeled error**. Per-step MAPE ranges from 28.5% to 1287% (codegen, high batch variance)

**Recommendation:** All improvement effort should focus on StepTime prediction. Any non-zero contribution to queueing/scheduling/preemption is additive benefit since the blackbox currently returns 0 for all of them.

### Round 1 Per-Step Baselines (Historical Reference)

Round 1 tested per-step MAPE only (no BLIS E2E validation):
- **Per-experiment XGBoost (30 features):** 34.0% avg MAPE -- 2x better than blackbox (70.4%)
- **Global blackbox:** 670% MAPE -- catastrophic failure across model scales
- **Key insight:** Per-step MAPE does not equal E2E error. The 34% per-step MAPE was never validated through BLIS.

### Validation Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| R1 (ProgressIndex as KV proxy) | FLAGGED | Pearson r ~ 0 between total_tokens and E2E time; ProgressIndex is valid for KV cache length but not for E2E prediction. Lifecycle KV extractor available as fallback. |
| R2 (Sampling bias) | PASS | CV=0.44, max/min ratio=4.2, random sampling confirmed (~10% rate) |
| R4 (Blackbox sufficient?) | PASS | 115% E2E mean error -- clearly beatable |

## Evaluation Framework

### Primary Metric

**Workload-level E2E mean error < 10%** on each of the 16 experiments individually. For each experiment:
```
predicted_mean_e2e = mean over requests of (sum of predicted step times along request's path)
observed_mean_e2e  = mean over requests of (observed request-level E2E latency)
E2E_mean_error     = |predicted_mean_e2e - observed_mean_e2e| / observed_mean_e2e
```

### All Metrics (ordered by priority)

| Priority | Metric | Target |
|----------|--------|--------|
| P1 | Workload-level E2E mean error | < 10% per experiment |
| P1 | Per-step MAPE, Pearson r | Diagnostic (no hard target) |
| P2 | TTFT mean fidelity | < 15% per experiment |
| P2 | ITL mean fidelity | < 15% per experiment |
| P3 | Tail behavior (p99) | No ranking inversions vs baseline |
| P4 | Generalization (workloads, models, hardware) | Cross-validation |
| P4 | Ease of use, retraining story, reproducibility | Qualitative |
| P5 | Hardware generalization (H100 -> A100) | Informational |
| P6 | Quantization transferability | Informational |

### Data Split Strategy

- **Primary**: Temporal split (60/20/20) within each experiment -- prevents autocorrelation leakage
- **Generalization**: Leave-one-model-out (4-fold) + leave-one-workload-out (4-fold)
- **Short-circuit**: Ideas with h1 per-step MAPE > 30% are dropped (threshold calibrated to blackbox_MAPE + 10% if blackbox exceeds 25%)

## Constraints

### LatencyModel Interface (Frozen)

The winning model must implement this 5-method Go interface:

```go
type LatencyModel interface {
    StepTime(batch []*Request) int64          // Primary target
    QueueingTime(req *Request) int64          // Currently alpha0 + alpha1*inputLen
    OutputTokenProcessingTime() int64          // Currently alpha2
    SchedulingProcessingTime() int64           // Currently returns 0
    PreemptionProcessingTime() int64           // Currently returns 0
}
```

Step-time estimation is the primary research target. The other 4 methods are secondary targets that contribute to E2E fidelity. At a minimum, improve StepTime while retaining current implementations for the other 4 methods.

### Feature Availability at Prediction Time

At inference time in the simulator, each Request in the batch provides:
- `InputTokens` (token sequence)
- `OutputTokens` (generated tokens so far)
- `ProgressIndex` (cumulative: input_processed + output_generated -- KV cache length proxy)
- `NumNewTokens` (tokens to generate this step)

Research models must use only features derivable from these fields plus experiment-level metadata (model name, TP degree).

### Inference Latency

Prediction must complete in <1ms per step (128 requests max). Linear models, tree ensembles (~100 trees), and small neural networks all qualify. Only large neural networks (>10M parameters) or GPU-inference-requiring approaches are excluded.

### Go Integration Path

The winning model must eventually run in Go. Viable paths:
1. **Coefficient export** -- for parametric models (regression, piecewise linear)
2. **Go-native reimplementation** -- for tree ensembles (Go libraries available)
3. **ONNX export** -- for neural networks
4. **Evolved code translation** -- for evolutionary approaches producing interpretable code

Each idea must specify which integration path it would use.

## Algorithm Scope

**Not restricted to ML or to StepTime.** Research ideas may propose:
- Statistical regression (Ridge, Lasso, polynomial, piecewise)
- Tree ensembles (XGBoost, LightGBM, random forest)
- Neural networks (small MLPs, attention-based)
- Physics-informed models (analytical compute model + learned residuals)
- Evolutionary program synthesis (OpenEvolve, GEPA -- evolved prediction functions)
- Hybrid approaches (analytical backbone + ML residual correction)
- **Multi-component calibration** (jointly optimize all 5 LatencyModel methods for E2E fidelity)
- **Scheduling/preemption overhead models** (data-driven models for the currently-zero methods)
- **End-to-end calibration** (tune any/all components with E2E mean error as the objective, not per-component accuracy)

Each approach must:
1. Cite relevant prior work from systems/ML literature
2. Address all evaluation dimensions (P1--P6)
3. Specify which LatencyModel methods it covers (minimum: StepTime; **bonus: additional methods**)
4. Document its Go integration path
5. Be distinct from other proposed approaches

## Key Questions for Ideas to Address

### From Round 1 (still open)
1. How to handle the 3-order-of-magnitude step time range across models?
2. How to capture KV-cache-length effects using only ProgressIndex as proxy?
3. How to handle dense vs MoE architecture differences (25% active parameters for MoE)?
4. How to handle the non-additive prefill/decode interaction in mixed batches (chunked prefill)?
5. What features beyond the existing schema would improve predictions, and are they derivable from Request objects?

### New for Round 2 (broadened scope)
6. **Which LatencyModel component contributes most to E2E error?** Is StepTime the dominant error source, or do QueueingTime/SchedulingTime/OutputTokenProcessingTime introduce comparable systematic bias?
7. **Can we derive per-request KV features from lifecycle data?** The `request_metrics/` directories contain per-token timestamps -- can these reconstruct ProgressIndex at each step?
8. **Does per-step MAPE translate to E2E error?** Round 1 showed 34% per-step MAPE. If errors are symmetric, E2E mean error could be much lower. This must be measured via BLIS simulation.
9. **Can non-zero SchedulingProcessingTime and PreemptionProcessingTime improve E2E fidelity?** These are currently hardcoded to 0. Ground-truth traces contain timing data for these events.
10. **Is end-to-end calibration more practical than per-component accuracy?** Instead of optimizing each component independently, calibrate the composition to minimize E2E error directly (e.g., grid search over alpha/beta coefficients with E2E error as the objective).
11. **Can simpler models with better features match XGBoost?** Round 1's XGBoost underfits (shallow trees) -- better features (per-request KV) might enable Ridge or piecewise linear models that are trivial to integrate in Go.

---

## Background

This section synthesizes context from the BLIS codebase, the Round 1 research findings, the StepML design document, and the academic literature on LLM inference latency modeling. The purpose is to ground the problem in both the internal engineering context and the state of the art in the broader research community.

### 1. BLIS Simulator Architecture and Latency Model Contract

BLIS is a CPU-only, deterministic discrete-event simulator (DES) for LLM inference serving systems. Its architecture follows a two-layer design: a domain-agnostic simulation kernel (event queue, clock, RNG, statistics) and domain-specific modules (router, scheduler, KV cache manager, latency model, batch formation). The simulator is designed for capacity planning and policy optimization without requiring real GPUs.

The `LatencyModel` interface is the central prediction contract for timing estimation in BLIS. It is defined in `sim/latency_model.go` and consists of 5 methods, each returning a time estimate in microseconds:

```go
type LatencyModel interface {
    StepTime(batch []*Request) int64
    QueueingTime(req *Request) int64
    OutputTokenProcessingTime() int64
    SchedulingProcessingTime() int64
    PreemptionProcessingTime() int64
}
```

Three implementations currently exist:

1. **BlackboxLatencyModel** (`sim/latency/latency.go`): Uses trained alpha/beta regression coefficients. StepTime computes `beta0 + beta1*cacheMissTokens + beta2*decodeTokens` by iterating over the batch and classifying each request as prefill or decode based on its `ProgressIndex` vs `InputTokens` length. QueueingTime uses `alpha0 + alpha1*inputLen`. OutputTokenProcessingTime returns constant `alpha2`. SchedulingProcessingTime and PreemptionProcessingTime both return 0.

2. **RooflineLatencyModel** (`sim/latency/roofline.go`): An analytical FLOPs/bandwidth roofline model that computes step time from transformer architecture parameters and measured GPU performance. It uses per-request `ProgressIndex` for decode attention FLOPs (FLOPs-weighted MFU across heterogeneous KV lengths), power-of-2 bucketing for prefill, GEMM time computation with memory-bandwidth floor, and a `max(compute, memory)` roofline combination for each phase. Mixed batches use `max(prefill_time, decode_time)`. This model is NOT being replaced -- it serves users who provide hardware specs and model config files.

3. **StepMLLatencyModel** (`sim/latency/stepml.go`): A newly created lightweight research-phase implementation that loads trained model artifacts from a JSON file. It supports linear coefficient models and extracts 8 batch features including per-request KV proxy features (`kv_sum`, `kv_max`, `kv_mean`) derived from `ProgressIndex`. All 5 interface methods are implemented with configurable constants for scheduling, preemption, and output token processing time.

The factory dispatch in `NewLatencyModel()` follows a priority chain: roofline (if `hw.Roofline` is true) > StepML (if `hw.StepMLModelPath` is set) > blackbox (fallback). This three-way dispatch ensures backward compatibility while enabling the new model.

**E2E Latency Decomposition through the Simulator:**

A request's lifecycle in BLIS proceeds through a specific event chain (documented in the design doc, Section "LatencyModel Behavioral Contract Compatibility"):

- `[1]` ArrivalEvent: Adds `QueueingTime(req)` delay
- `[2]` QueuedEvent: Request waits in WaitQ (emergent duration, depends on all other components)
- `[3]` FormBatch: Adds `SchedulingProcessingTime()` per request admitted to batch
- `[3b]` preemptForTokens: Adds `PreemptionProcessingTime()` if KV pressure forces eviction
- `[4]` executeBatchStep: Adds `StepTime(batch)` for GPU execution
- `[5]` per-token completion: Adds `OutputTokenProcessingTime()` to each decode token's ITL

The critical insight is that improving only StepTime `[4]` leaves 4 other controllable delays at their current values. Errors in `[1]`, `[3]`, and `[5]` compound into E2E error even if `[4]` is perfect. The component error attribution from WP0 confirmed that StepTime currently accounts for 100% of modeled error because all other alpha coefficients are zero -- but this also means that calibrating non-zero values for the other methods is "free" additional fidelity.

### 2. Round 1 Findings: What Worked and What Failed

Round 1 (documented in `hypotheses/h-stepml/round1/round1_FINDINGS_SUMMARY.md`) conducted a thorough investigation of StepTime-only prediction using two main approaches and one deferred approach:

**Idea 1 -- Physics-Informed Tree Ensemble:** Engineered 30 features across 6 groups (raw batch features, physics-informed FLOPs estimates, KV cache proxies, batch shape features, interaction terms, architecture metadata). Per-experiment XGBoost achieved 34.0% avg MAPE -- a 2x improvement over the blackbox's 70.4%. However, per-experiment Ridge regression with the same 30 features achieved only 92.1% avg MAPE, and global training across all experiments failed catastrophically (301.7% for Ridge, 670% for blackbox).

**Idea 2 -- Analytical Decomposition + Learned Corrections:** Decomposed step time into 4 FLOPs components (prefill GEMM, prefill attention, decode GEMM, decode attention), then attempted to learn multiplicative correction factors. This failed structurally: decode attention FLOPs were zero for all steps because per-request KV lengths were unavailable in the training data. The resulting 36-parameter model (78.7% MAPE) performed worse than the 3-parameter blackbox because correction factors compensated pathologically (factor=39.5 on zero signal).

**Key findings that carry forward:**

- **Feature quality is the bottleneck, not model capacity.** XGBoost consistently selected shallow trees (max_depth=4, n_estimators=100) for 13/16 experiments, suggesting underfitting. The top features by importance were KV state proxies (`kv_blocks_used`, `running_depth`, `kv_blocks_free`) rather than physics features -- the model was learning to predict step time from indirect system state because per-request KV information was missing.

- **Per-model training is mandatory.** Step times span 3+ orders of magnitude across model configurations (12 us for Llama-7B decode to 250,000 us for Llama-70B prefill). Leave-one-model-out XGBoost produced 9908% MAPE when holding out Llama-7B (tp=1 model trained on tp=2/4 data). This is a representation issue, not a model capacity issue.

- **"General" workloads are consistently hardest.** Across all models and approaches, the 4 general workload experiments had the worst accuracy (50--121% MAPE for XGBoost) due to the highest batch composition diversity.

- **Chunked prefill invalidates pure-phase assumptions.** Only 6 of 122,752 steps were pure-prefill. vLLM's chunked prefill makes nearly every batch mixed or pure-decode (80.6% pure decode, 19.4% mixed, 0.005% pure prefill).

- **Per-step MAPE was never validated against E2E error.** The entire Round 1 operated in Python without BLIS integration. The 34% per-step MAPE might translate to much lower E2E mean error if per-step errors are symmetric and cancel across a request's lifetime, or it might translate to even higher E2E error if errors are systematically biased.

### 3. StepML Go Integration (Already Built)

A key gap from Round 1 -- lack of Go integration -- has already been addressed. The `StepMLLatencyModel` in `sim/latency/stepml.go` provides:

- A JSON artifact schema (`StepMLArtifact`) with version, step_time model, optional queueing_time model, and constant values for output token processing, scheduling processing, and preemption processing time.
- Feature extraction (`extractBatchFeatures`) that computes 8 features from a batch of Request objects: `prefill_tokens`, `decode_tokens`, `num_prefill_reqs`, `num_decode_reqs`, `scheduled_tokens`, `kv_sum`, `kv_max`, `kv_mean`. The KV features are derived from each Request's `ProgressIndex`.
- Linear prediction (`predictLinear`) that evaluates `intercept + sum(coeff_i * feature_i)` with missing-feature tolerance.
- INV-M-1 enforcement: step time predictions are floored at 1 us.

The validation harness (`hypotheses/h-stepml/shared/validate_blis.py`) runs BLIS on each ground-truth experiment by: parsing experiment config from `exp-config.yaml`, extracting KV block counts from `vllm.log`, building a workload spec from the inference-perf profile, executing the BLIS binary with the candidate model's coefficients, and comparing predicted vs observed E2E/TTFT/ITL means.

### 4. Related Work: LLM Inference Latency Modeling in the Literature

The problem of predicting GPU inference step time has been addressed by several recent systems, each with a different methodological approach. The following survey covers the most relevant prior work, organized by methodology.

#### 4.1 Profiling-Based Simulation: Vidur (MLSys 2024)

Vidur [Agrawal et al., 2024] is the closest prior art to BLIS's latency modeling challenge. Vidur uses a simulation approach with **operator triaging** to classify operations by their input dependencies:

- **Token-level operators** (linear projections, activations): Runtime depends only on total tokens in the batch.
- **Sequence-level operators** (attention): Runtime depends on context length and token count. For prefill attention, runtime is proportional to the sum of squared prefill lengths. For decode attention, which is memory-bound, the model is based on total KV-cache reads rather than individual request characteristics.
- **Communication operators**: Runtime depends only on data transfer volume.

Vidur collects limited profiling data for each operator type and trains small ML models (random forests) for runtime interpolation. It achieves <5% error at 85% system capacity for dynamic workloads and can predict P95 tail latency with up to 3.33% error for static workloads.

**Relevance to BLIS:** Vidur's operator-level decomposition is more granular than BLIS's single `StepTime()` call, which predicts the full step duration as an opaque quantity. However, MIST (see below) found that Vidur's operator-level approach accumulates errors across layers and operators. BLIS's single-call design may actually be advantageous if the model can learn aggregate step-level patterns without the error accumulation of operator-level decomposition.

**Key insight:** Vidur's random forest approach for interpolating profiled operator times is conceptually similar to Round 1's XGBoost approach but operates at a finer granularity. The fact that Vidur achieves <5% error with operator-level random forests suggests that ML-based step prediction is viable -- the question is whether step-level (rather than operator-level) prediction can achieve comparable accuracy with the right features.

*Citation: Agrawal, A., et al. "Vidur: A Large-Scale Simulation Framework For LLM Inference." MLSys 2024.*

#### 4.2 Engine-Level ML Prediction: MIST (2025)

MIST [2025] takes a fundamentally different approach from Vidur by training ML regressors directly on **complete engine execution times** rather than individual operators. MIST uses an ensemble of regressors, each trained on a distinct pre-specified subset of the data. The key training features include input size, batch size, chunk size (for chunked batching), and tensor parallelism configuration.

MIST achieves **average error of 2.5% with median error < 1%** -- significantly outperforming Vidur. The paper identifies two key weaknesses in Vidur's operator-level approach: (1) using operator-level ML predictors results in error accumulating across layers and operators, and (2) ignoring kernel launch overheads and smaller operations.

MIST collects over 200K datapoints on running vLLM, varying input size, batch size, chunk size, and TP configuration across H100, A100, and L40S GPUs.

**Relevance to BLIS:** MIST's approach -- training directly on step-level engine execution times rather than decomposing into operators -- validates the approach taken in BLIS Round 1. MIST's 2.5% average error demonstrates that step-level prediction with the right features and sufficient training data can far exceed the 34% MAPE achieved by Round 1's XGBoost. The critical difference is that MIST profiles on the actual hardware with controlled inputs, while BLIS must predict from production trace data with observational features only.

**Key insight for Round 2:** MIST's ensemble of regressors trained on distinct data subsets is a direct precedent for per-model or per-regime training. MIST's feature set (input size, batch size, chunk size, TP) is simpler than Round 1's 30 features but achieves much better accuracy -- suggesting that fewer, higher-quality features (particularly those capturing chunked prefill semantics) may outperform many noisy features.

*Citation: MIST, 2025. arXiv:2504.09775.*

#### 4.3 Analytical Decomposition: RAPID-LLM (2025)

RAPID-LLM [2025] uses a purely analytical approach that models inference through operator-level execution traces with tile-based roofline models. For each attention and feedforward operation, RAPID-LLM searches candidate tiling strategies to determine latency, accounting for SM occupancy and execution waves. It estimates SRAM/L2/HBM traffic from tile sizes and tensor dimensions, then maps this onto a multi-level roofline model.

RAPID-LLM achieves **at most 10.4% prediction error** for Llama-2 inference under tensor parallelism relative to published runtimes across 7B, 13B, and 70B model sizes.

**Relevance to BLIS:** RAPID-LLM's approach is similar in spirit to BLIS's existing roofline model but with significantly more hardware detail (multi-level memory hierarchy, tiling analysis, SM occupancy). The 10.4% error is notably better than BLIS's roofline model (4816% E2E error), but RAPID-LLM is evaluated against published benchmark times (not E2E simulation with continuous batching dynamics). The large gap between RAPID-LLM's 10.4% and BLIS's roofline error suggests that BLIS's roofline model may be missing critical hardware-level details, but also that E2E simulation introduces error sources beyond step-time prediction alone.

*Citation: RAPID-LLM, 2025. arXiv:2512.19606.*

#### 4.4 Kernel Database Decomposition: AIConfigurator (2025)

AIConfigurator [2025] takes a data-driven decomposition approach without ML models. It maintains a calibrated kernel-level performance database with real hardware measurements, decomposes each inference iteration into constituent operators (GEMM, attention, communication, MoE operations), queries the database for individual operator latencies, and composes them to estimate total step latency.

AIConfigurator achieves **6--12% MAPE for TPOT (time per output token)** across tested models including Qwen3-32B (8.2% TPOT error) and Qwen3-235B MoE (6.8% TPOT error). It explicitly models three serving modes: static batching, aggregated (continuous batching with mixed prefill/decode), and disaggregated (separate prefill/decode pools).

**Relevance to BLIS:** AIConfigurator's kernel database approach is a middle ground between BLIS's roofline (analytical, no calibration data) and MIST's ML approach (pure statistical fitting). The 6--12% TPOT accuracy demonstrates that decomposition-with-database can work well, but it requires hardware profiling infrastructure that BLIS does not have. More importantly, AIConfigurator's explicit handling of MoE (separate expert routing and tokens-per-expert modeling) addresses a known gap in BLIS's data.

*Citation: AIConfigurator, 2025. arXiv:2601.06288.*

#### 4.5 GPU-Free Emulation: Revati (2025)

Revati [2025] takes an entirely different approach by intercepting CUDA calls in actual serving frameworks (vLLM, SGLang) and advancing virtual time instead of executing kernels. Rather than predicting step time from features, Revati uses a pluggable interface supporting both analytical and profiling-based runtime predictors for individual GPU operations.

Revati demonstrates **<5% prediction error** across multiple models and parallelism configurations while maintaining fidelity to complex scheduling behaviors like adaptive batching and prefix caching.

**Relevance to BLIS:** Revati's approach eliminates the "semantic gap" between simulator and real system by running the actual serving framework code. While not directly applicable to BLIS (which is a standalone simulator), Revati's architecture validates that accurate step-time prediction enables high-fidelity simulation. Revati's key innovation -- intercepting at the CUDA level rather than reimplementing scheduling logic -- highlights the maintenance burden that simulators like BLIS face when vLLM's scheduler changes across versions.

*Citation: Revati, 2025. arXiv:2601.00397.*

#### 4.6 Hardware-Agnostic Analytical Forecasting: LIFE (2025)

LIFE (LLM Inference Forecast Engine) [2025] provides an analytical framework that predicts inference performance using only abstract hardware specifications (TOPS and bandwidth) combined with measured operator efficiency percentages. For each operator, execution time is computed as `max(compute_time, memory_time) + dispatch_overhead`, where compute time is `operations / (efficiency * TOPS)` and memory time is `data_movement / (efficiency * bandwidth)`.

LIFE validates across CPU, NPU, iGPU, and NVIDIA V100 GPU with accuracy matching measured performance (e.g., 186ms measured vs 180ms forecast for 2048-token prefill on CPU at 50% efficiency).

**Relevance to BLIS:** LIFE's formulation is structurally similar to BLIS's roofline model but operates at the operator level with explicit efficiency factors. The key insight for BLIS is LIFE's treatment of dispatch overhead as a separate additive term -- analogous to the currently-zero `SchedulingProcessingTime()` in BLIS's LatencyModel.

*Citation: LIFE, 2025. arXiv:2508.00904.*

### 5. Synthesis: Landscape of Approaches and Gaps

The literature reveals a spectrum of methodological approaches for step-time prediction, each with distinct accuracy-generalization tradeoffs:

| Approach | Representative System | Step-Level Accuracy | Calibration Data Needed | Generalization |
|----------|----------------------|--------------------|-----------------------|----------------|
| Pure analytical (roofline) | BLIS roofline, LIFE | 10--4816% | Hardware specs only | Cross-hardware but low accuracy for batched serving |
| Analytical + kernel DB | AIConfigurator, RAPID-LLM | 6--12% TPOT | Hardware profiling runs | Per-hardware calibration |
| Operator-level ML | Vidur | <5% (static), ~10% (dynamic) | Per-operator profiling | Per-model profiling |
| Step-level ML | MIST, BLIS Round 1 | 2.5% (MIST), 34% (BLIS R1) | Engine-level trace data | Per-model/config training |
| CUDA interception | Revati | <5% | Runtime prediction models | Any vLLM/SGLang workload |

**Key gap in BLIS relative to the state of the art:** MIST achieves 2.5% average error using step-level ML with controlled profiling data (input size, batch size, chunk size, TP). BLIS Round 1 achieved only 34% MAPE despite using 30 features. The gap is attributable to three factors:

1. **Training data quality.** MIST profiles with controlled, systematic variation of inputs. BLIS Round 1 trained on observational production traces where batch composition is determined by the scheduler, not by experimental design. This means the training distribution may not uniformly cover the input space.

2. **Missing per-request KV features.** MIST's chunk size feature directly captures the prefill/decode split within a step. BLIS Round 1 lacked per-request KV lengths, forcing the model to use indirect proxies (`kv_blocks_used`, `running_depth`). The ProgressIndex is now available via the StepML Go integration.

3. **Evaluation granularity.** MIST measures per-step accuracy directly. BLIS's goal is workload-level E2E mean error, which may be more forgiving (error cancellation across steps) but also more demanding (errors compound through emergent queueing dynamics).

**What the literature suggests for Round 2:**

- **Step-level ML is the most promising direction.** MIST's 2.5% error demonstrates the feasibility. The key is feature quality (particularly per-request KV and chunk-size information), not model complexity.
- **Per-model calibration is standard practice.** Every system in the survey (Vidur, MIST, AIConfigurator) calibrates per model configuration. This aligns with Round 1's finding that global models fail catastrophically.
- **Ensemble methods outperform individual regressors.** Both MIST (ensemble of regressors) and Vidur (random forests) use ensemble approaches. Round 1's XGBoost (a boosted ensemble) was the best performer.
- **Operator decomposition is a double-edged sword.** While physically motivated, Vidur's operator-level approach accumulates errors across layers. MIST's step-level approach avoids this. BLIS Round 1's analytical decomposition (Idea 2) failed for the same reason -- structural incompleteness in one component (decode attention) contaminated the entire decomposition.
- **End-to-end calibration is underexplored.** None of the surveyed systems optimize for workload-level E2E mean error directly. They all optimize per-step or per-operator accuracy. BLIS's unique contribution could be an end-to-end calibration approach that tunes all 5 LatencyModel components jointly to minimize E2E error.
- **Physics-informed features add value where GEMM dominates.** Round 1 confirmed that analytical FLOPs estimates correlate well with step time for GEMM-dominated workloads (r > 0.85) but poorly for attention-dominated workloads (r < 0.25). A hybrid approach that uses physics for the GEMM component and ML for the attention residual could capture both regimes.

### 6. BLIS-Specific Design Constraints from the Design Document

The StepML research design document (`docs/plans/2026-02-26-stepml-research-design.md`) establishes several constraints and decisions that shape the solution space:

**Research-specific invariants (INV-M-1 through INV-M-6):**
- INV-M-1: Step-time estimate must be positive for any non-empty batch.
- INV-M-2: Prediction must be deterministic (same batch, same result).
- INV-M-3: Prediction must be side-effect-free (pure function).
- INV-M-4: Prediction latency < 1ms at max batch size (128 requests).
- INV-M-5 (soft): Monotonicity in total tokens.
- INV-M-6: Bounded systematic bias |MSPE| < 5%.

**Pipeline phases:** The research follows a 4-phase pipeline: (1) Ideation, (2) Hypothesis Selection, (3) Experimentation, (4) Comparison and Selection. All ideas must validate via BLIS simulation runs (Tier 1b) -- per-step MAPE alone is insufficient.

**Baseline fairness:** The blackbox baseline uses per-model+TP fitting (pooling all workloads), not per-experiment fitting. This is a harder regression problem than Round 1's per-experiment baseline.

**Roofline isolation guarantee:** Research ideas MUST NOT use the roofline model or its predictions as a component, feature, building block, or calibration target. Ideas may use physics-informed features derived independently from first principles.

**Three optimization strategies are identified:**
1. Per-component accuracy: Improve each method independently.
2. End-to-end calibration: Jointly optimize all 5 methods to minimize E2E error.
3. Error attribution first: Measure which components contribute most before optimizing.

**MoE validation requirement:** Because there is only one MoE model (Mixtral-8x7B), MoE generalization uses leave-one-workload-out within Mixtral experiments (4-fold) and comparison of dense-trained vs unified model predictions. Success criterion: E2E mean error < 15% on all 4 Mixtral experiments.

### 7. Available Infrastructure for Round 2

The shared infrastructure under `hypotheses/h-stepml/shared/` provides several key tools:

- **`baselines.py`**: BlackboxBaseline (re-trained 3-coefficient linear regression) and NaiveMeanBaseline, with R4 gate checking and short-circuit threshold calibration.
- **`e2e_decomposition.py`**: Component-level error attribution that computes each LatencyModel method's contribution to E2E error. Provides `component_error_attribution()` for single experiments and `component_error_attribution_all_experiments()` for batch analysis.
- **`validate_blis.py`**: Full BLIS validation harness that runs the simulator on ground-truth experiments, parsing experiment config, KV blocks from vllm.log, and inference-perf profiles. Supports both coefficient-based (`--alpha-coeffs`/`--beta-coeffs`) and StepML artifact (`--stepml-model`) modes.
- **`convert_lifecycle_to_traces.py`**: Converts per-request lifecycle data to trace format.
- **`lifecycle_kv_extractor.py`**: Derives per-request KV features from lifecycle data.
- **`mfu_benchmarks.py`**: Loads kernel-level GEMM and attention MFU data.
- **`establish_baseline.py`**: Establishes the blackbox E2E baseline (THE NUMBER TO BEAT).

### 8. Summary of the Research Landscape

The problem of achieving <10% workload-level E2E mean error in BLIS's latency model is well-positioned relative to the academic literature:

- **The target is achievable.** MIST demonstrates 2.5% step-level error using step-level ML with controlled data. BLIS's observational data and feature limitations make the problem harder, but the 10% E2E target (which benefits from error cancellation) is more forgiving than a 2.5% per-step target.

- **The primary bottleneck is feature quality.** Round 1's XGBoost underfits because the feature set lacks per-request KV information. The StepML Go integration now provides ProgressIndex-based KV features (`kv_sum`, `kv_max`, `kv_mean`) which partially close this gap.

- **Per-model calibration is universal.** Every system in the literature calibrates per model configuration. Global models fail because step times span 3+ orders of magnitude.

- **End-to-end calibration is an unexplored opportunity.** No surveyed system optimizes jointly for workload-level simulation fidelity. BLIS's 5-method LatencyModel interface with 3 currently-zero methods provides unique headroom for joint calibration.

- **The validation infrastructure is ready.** Unlike Round 1, which had no BLIS integration, Round 2 has a Go StepML model, a validation harness, and component error attribution tooling. The research-to-validation loop is closed.

The path from 115% E2E mean error to <10% requires addressing three gaps: (1) improving StepTime with per-request KV features, (2) calibrating the currently-zero LatencyModel methods, and (3) validating through BLIS simulation rather than per-step metrics alone.

---

## Round 2 Research Ideas

### Idea 1: Multi-Component Bayesian Calibration with E2E Objective

**Date:** 2026-02-26
**Iteration:** 1 of 3

#### Core Concept

Instead of optimizing each LatencyModel component independently (as Round 1 did with StepTime-only per-step MAPE), jointly calibrate ALL 5 LatencyModel methods using Bayesian optimization (BO) with the BLIS workload-level E2E mean error as the direct objective function. The key insight is that E2E error is the actual target metric, and per-component accuracy is a proxy that may not align -- compensating errors between components are acceptable and even desirable if they reduce E2E error.

#### LatencyModel Methods Covered

| Method | Coverage | Approach |
|--------|----------|----------|
| **StepTime** | Primary | Piecewise linear model with ProgressIndex-derived KV features; coefficients optimized via BO |
| **QueueingTime** | Secondary | Parameterized as `q0 + q1*input_len + q2*input_len^2`; coefficients optimized via BO |
| **OutputTokenProcessingTime** | Secondary | Single constant optimized via BO (not fixed at zero) |
| **SchedulingProcessingTime** | Secondary | Parameterized as `s0 + s1*log(queue_depth_proxy)`; optimized via BO |
| **PreemptionProcessingTime** | Secondary | Single constant optimized via BO (not fixed at zero) |

Total free parameters: ~12-15 per model configuration (StepTime: 6-8, QueueingTime: 3, OutputToken: 1, Scheduling: 2, Preemption: 1).

#### How It Differs from Round 1 and Other Ideas

- **Round 1** optimized StepTime per-step MAPE in isolation, never validated through BLIS, and left 4 methods at zero/default. This idea optimizes ALL methods jointly against E2E error through actual BLIS simulation runs.
- **Idea 2** (below) focuses on improving StepTime feature quality with regime-switching. This idea focuses on the *optimization objective* -- even a modest StepTime model can achieve low E2E error if the other 4 methods compensate correctly.
- **Idea 3** (below) uses evolutionary synthesis to discover functional forms. This idea uses fixed functional forms but optimizes their parameters end-to-end.

#### Algorithm

1. **StepTime Model:** Per-model piecewise linear with regime detection:
   - Regime 1 (pure decode): `b0 + b1*decode_tokens + b2*kv_mean + b3*kv_max`
   - Regime 2 (mixed batch): `b4 + b5*prefill_tokens + b6*decode_tokens + b7*prefill_x_decode`
   - Regime classification: `prefill_tokens > 0`
   - KV features (`kv_mean`, `kv_max`) derived from ProgressIndex per request

2. **Secondary Methods:** Parameterized as above (QueueingTime quadratic, Scheduling log-linear, OutputToken/Preemption constants)

3. **Bayesian Optimization Loop:**
   - Search space: all ~15 parameters simultaneously
   - Objective: mean E2E error across all 4 workloads for a given model
   - Inner loop: for each parameter proposal, generate StepML JSON artifact, run BLIS validation harness (`validate_blis.py`) on all 4 workloads, average E2E error
   - BO library: scikit-optimize (Gaussian process surrogate with expected improvement acquisition)
   - Budget: 200-500 evaluations per model (each evaluation = 4 BLIS runs ~ 4 minutes total)

4. **Cross-validation:** Leave-one-workload-out within each model; report best parameters on 3 workloads, test on held-out workload.

#### Go Integration Path

**Coefficient export** -- the optimized parameters are stored in the existing `StepMLArtifact` JSON schema (linear model coefficients for StepTime, constants for the other 4 methods). The `StepMLLatencyModel` in `sim/latency/stepml.go` already supports this format. The only extension needed is adding a `regime` field to support piecewise dispatch.

#### Key Features (ProgressIndex-Derived)

Per-request KV proxy features computed from `ProgressIndex`:
- `kv_sum` = sum of ProgressIndex across batch (total context length proxy)
- `kv_mean` = mean ProgressIndex (average context length)
- `kv_max` = max ProgressIndex (longest context in batch -- determines attention memory pressure)
- `prefill_x_decode` = prefill_tokens * decode_tokens (mixed-batch interaction)

These are already implemented in `extractBatchFeatures()` in `sim/latency/stepml.go`.

#### Literature Citations

- **MIST** (arXiv:2504.09775): Validates that step-level ML with simple features achieves 2.5% error. This idea builds on MIST's insight but adds multi-component calibration.
- **Vidur** (MLSys 2024): Demonstrates per-model calibration with random forests. This idea uses the same per-model calibration strategy but optimizes against E2E error rather than per-operator error.
- **Bayesian Optimization for simulation calibration:** Kennedy & O'Hagan (2001), "Bayesian calibration of computer models" -- foundational work on using BO to calibrate simulator parameters against real-world observations. This is the direct methodological precedent.
- **LIFE** (arXiv:2508.00904): Treats dispatch overhead as a separate additive term -- analogous to our non-zero SchedulingProcessingTime calibration.

#### Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| BO converges to local optimum | Medium | Multiple restarts (5 random seeds); warm-start from blackbox coefficients |
| 200-500 BLIS evaluations too expensive | Low | Each BLIS run takes ~60s; 500 runs = ~8 hours per model (parallelizable) |
| Overfitting to 4 workloads per model | Medium | Leave-one-workload-out CV; compare BO-optimized vs per-step-fitted coefficients |
| Compensating errors fragile to new workloads | High | Report both per-step MAPE and E2E error; flag cases where per-step MAPE is high but E2E is low |

#### Expected Outcome

- **Conservative estimate:** 30-50% E2E mean error (improvement from non-zero secondary methods alone)
- **Optimistic estimate:** <10% E2E mean error (joint calibration finds compensating parameter settings)
- **Key diagnostic:** The gap between per-step MAPE and E2E error reveals how much "free" improvement comes from error cancellation vs. genuine prediction accuracy

---

#### Idea 1 Reviews

**Note on model attribution:** Reviews were requested from aws/claude-opus-4-6, Azure/gpt-4o, and GCP/gemini-2.5-flash. Only the Anthropic model (aws/claude-opus-4-6) was available in this environment (no Azure OpenAI or GCP Vertex AI API credentials configured). The Claude Opus 4.6 review is the actual model output. The GPT-4o and Gemini reviews could not be obtained and are marked as UNAVAILABLE.

##### Review 1 (aws/claude-opus-4-6) -- ACTUAL

*Model: aws/claude-opus-4-6 (self-review, as this agent is running on Claude Opus 4.6)*

**1. NOVELTY (3/5):** The end-to-end Bayesian calibration concept is moderately novel in this context. Kennedy & O'Hagan (2001) established simulator calibration with BO, but applying it specifically to multi-component LLM inference latency models with a DES-in-the-loop is a new application. The novelty is in the *composition* (5-method joint calibration) rather than the algorithmic technique (BO is standard). None of the surveyed systems (MIST, Vidur, AIConfigurator) optimize jointly for E2E simulation fidelity -- they all optimize per-component accuracy. This is a genuine gap in the literature.

**2. FEASIBILITY (3/5):** Concerning. 15 parameters with 200-500 evaluations gives a ratio of ~13-33 evaluations per parameter. Gaussian process BO in 15 dimensions is at the edge of effectiveness -- the GP surrogate becomes unreliable above ~15-20 dimensions. Mitigation: use structured search (e.g., optimize StepTime coefficients first with per-step data, then fine-tune all 15 jointly via BO). The 8-hour runtime per model is acceptable for research but makes rapid iteration difficult.

**3. RISK (4/5 -- HIGH RISK):** The compensating-errors approach is the biggest weakness. If StepTime systematically overestimates by 20% and QueueingTime compensates by underestimating by 20%, the E2E might look correct but the simulation dynamics (queue buildup, batch formation patterns) will be wrong. This means TTFT/ITL distributions will be distorted even if E2E mean is correct. The idea should explicitly track TTFT and ITL errors, not just E2E mean. Also, the compensation is workload-dependent -- parameters tuned to compensate on one workload will not compensate correctly on another because the error structure differs.

**4. GAP COVERAGE (2/5):** The StepTime model uses ProgressIndex-derived KV features (kv_mean, kv_max), which partially addresses the missing per-request KV problem. However, the piecewise linear model with 4 coefficients per regime is too simple to capture the nonlinear relationship between KV features and step time (attention cost scales quadratically with kv_len). Also, the SchedulingProcessingTime parameterization `s0 + s1*log(queue_depth_proxy)` is problematic -- queue_depth is not available as a feature in the LatencyModel interface. `SchedulingProcessingTime()` takes no arguments; it returns a constant. The idea incorrectly assumes it can be parameterized on runtime state.

**5. INTEGRATION (4/5):** Mostly clear. The StepML artifact format already supports linear coefficients and constants for all 5 methods. The regime-switching extension is minor. However, the SchedulingProcessingTime parameterization issue (above) needs resolution -- it must be a constant, not state-dependent.

**6. OVERALL: 3/5 (Weak Accept)**

**Strengths:**
- Directly optimizes the target metric (E2E error) rather than a proxy (per-step MAPE)
- Addresses all 5 LatencyModel methods, not just StepTime
- Uses existing StepML infrastructure for Go integration
- Could provide valuable diagnostic: the gap between per-step and E2E error

**Weaknesses:**
- 15-dimensional BO is borderline feasible; risk of poor convergence
- Compensating errors are fragile and workload-dependent
- StepTime piecewise linear model is too simple for the nonlinear KV-attention relationship
- SchedulingProcessingTime parameterization is inconsistent with the interface (takes no arguments)
- The "conservative estimate" of 30-50% E2E error is still far from the 10% target

**Actionable suggestions:**
1. Reduce the BO dimensionality: fix secondary methods at data-derived constants first, then optimize StepTime coefficients only (8 parameters)
2. Use a two-phase approach: Phase 1 fit StepTime against per-step data, Phase 2 fine-tune all parameters jointly via BO
3. Fix the SchedulingProcessingTime to be a constant (as the interface requires)
4. Add quadratic kv_mean^2 or kv_max^2 terms to capture attention cost scaling

##### Review 2 (Azure/gpt-4o) -- UNAVAILABLE

*Requested from: Azure/gpt-4o*
*Status: UNAVAILABLE -- No Azure OpenAI API credentials configured in this environment. Review could not be obtained.*

##### Review 3 (GCP/gemini-2.5-flash) -- UNAVAILABLE

*Requested from: GCP/gemini-2.5-flash*
*Status: UNAVAILABLE -- No GCP Vertex AI API credentials configured in this environment. Review could not be obtained.*

---

### Idea 2: Regime-Switching Ensemble with Per-Request KV Features (MIST-Inspired)

**Date:** 2026-02-26
**Iteration:** 2 of 3

#### Core Concept

Inspired by MIST's ensemble approach (multiple regressors trained on distinct data subsets achieving 2.5% error), build a regime-switching ensemble of lightweight models where each sub-model specializes in a specific batch composition regime. The key insight from MIST is that *fewer, higher-quality features on regime-specific data* outperforms *many features on heterogeneous data*. Combined with ProgressIndex-derived per-request KV features (the primary binding constraint from Round 1), this approach directly addresses both the feature quality bottleneck and the regime heterogeneity problem.

#### LatencyModel Methods Covered

| Method | Coverage | Approach |
|--------|----------|----------|
| **StepTime** | Primary | 3-regime ensemble: (a) decode-only, (b) mixed-light (prefill < 256 tokens), (c) mixed-heavy (prefill >= 256 tokens). Each regime uses a per-model Ridge/Lasso regression with ~6-8 KV-enriched features. |
| **QueueingTime** | Secondary | Per-model linear regression calibrated from lifecycle data: `q0 + q1*input_len`. Coefficients fit to minimize mean queueing time error (not E2E). |
| **OutputTokenProcessingTime** | Secondary | Per-model constant derived as median per-token output processing time from lifecycle data (measurable as ITL minus step contribution). |
| **SchedulingProcessingTime** | Tertiary | Per-model constant derived from ground-truth traces: median time between scheduler invocation and step start. |
| **PreemptionProcessingTime** | Tertiary | Per-model constant derived from ground-truth preemption event timing data. |

#### How It Differs from Round 1 and Other Ideas

- **Round 1 XGBoost:** Used 30 features on all data (no regime switching), achieved 34% per-step MAPE. Features included noisy system-state proxies. This idea uses 6-8 clean features per regime with ProgressIndex KV features replacing system-state proxies.
- **Idea 1 (Bayesian Calibration):** Optimizes parameters end-to-end against E2E error with a simple StepTime model. This idea focuses on *StepTime feature and model quality* using regime-specific training, then validates through BLIS.
- **Idea 3 (Evolutionary Synthesis):** Discovers novel functional forms automatically. This idea uses known functional forms (linear regression) but with regime-specific specialization.
- **MIST:** MIST profiles with controlled inputs (input size, batch size, chunk size). This idea adapts MIST's ensemble philosophy to observational data with ProgressIndex as the KV proxy.

#### Algorithm

1. **Regime Classification:**
   - Regime A (decode-only): `prefill_tokens == 0` (~80.6% of steps)
   - Regime B (mixed-light): `0 < prefill_tokens < 256` (~15% of steps)
   - Regime C (mixed-heavy): `prefill_tokens >= 256` (~4.4% of steps)
   - The 256-token threshold is motivated by vLLM's `max_num_batched_tokens=2048` -- prefills larger than ~256 consume significant batch budget.

2. **Feature Engineering (per-request KV features):**

   For each Request in batch, ProgressIndex gives cumulative (input_processed + output_generated), which equals the request's current KV cache length.

   **Decode-only features (Regime A):**
   - `decode_tokens` -- number of decode tokens
   - `num_decode_reqs` -- batch size
   - `kv_sum` -- total KV across all requests (attention FLOPs proxy)
   - `kv_max` -- max KV length (memory access bottleneck)
   - `kv_mean` -- mean KV length
   - `kv_std` -- std deviation of KV lengths (heterogeneity indicator)
   - `num_decode_reqs * kv_mean` -- interaction: batch_size x context_length

   **Mixed-batch features (Regimes B and C):**
   - `prefill_tokens`, `decode_tokens` -- token counts
   - `num_prefill_reqs`, `num_decode_reqs` -- request counts
   - `kv_sum`, `kv_max` -- KV features
   - `prefill_tokens * decode_tokens` -- interaction term
   - `prefill_tokens^2` -- quadratic prefill (attention cost is O(seq_len^2))

3. **Per-Model Training:**
   - Separate Ridge/Lasso regression for each (model, regime) pair
   - 4 models x 3 regimes = 12 regressors
   - Training data: 60% temporal split, pooling all 4 workloads per model
   - Regularization (Ridge alpha) tuned via 5-fold CV within training set

4. **Secondary Method Calibration:**
   - QueueingTime: linear regression on per-request lifecycle data
   - OutputTokenProcessingTime: median (ITL - step_contribution_per_token) from lifecycle data
   - SchedulingProcessingTime: median scheduling overhead from trace timing data
   - PreemptionProcessingTime: median preemption cost from KV event timing data

5. **Validation:**
   - Per-step MAPE and Pearson r (diagnostic)
   - BLIS E2E validation via `validate_blis.py` on all 16 experiments
   - LOWO cross-validation within each model

#### Go Integration Path

**Coefficient export** -- each regime's Ridge coefficients exported as a separate `LinearModel` entry in the `StepMLArtifact` JSON. The `StepMLLatencyModel` extended with:
1. An array of `LinearModel` entries (one per regime)
2. A regime classification function in `extractBatchFeatures()` that routes to the appropriate model
3. Constants for the 4 secondary methods

This requires a minor extension to the `StepMLArtifact` schema (adding a `regimes` array with `condition` and `model` per entry) but no changes to the LatencyModel interface.

New feature `kv_std` needs to be added to `extractBatchFeatures()` in `stepml.go` -- a simple one-pass computation over the batch.

#### Literature Citations

- **MIST** (arXiv:2504.09775): Direct inspiration for the ensemble-of-regressors approach. MIST achieves 2.5% error with per-regime training on distinct data subsets. This idea adapts MIST's methodology to observational data.
- **Vidur** (MLSys 2024): Operator triaging (classifying operations by input dependencies) is analogous to regime classification. Vidur's token-level vs sequence-level operator distinction maps to our decode-only vs mixed-batch regimes.
- **AIConfigurator** (arXiv:2601.06288): Explicitly models three serving modes (static, aggregated, disaggregated) -- validates the regime-switching concept for inference latency prediction.
- **BLIS H8** (internal hypothesis): Showed 12.96x attention FLOPs overestimate without per-request KV lengths. ProgressIndex-derived features directly address this finding.
- **Round 1 findings:** XGBoost feature importance showed kv_blocks_used (#1), running_depth (#3), kv_blocks_free (#10) as top features -- confirming that KV information is the primary signal. ProgressIndex provides this information per-request rather than as aggregate system state.

#### Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Regime boundaries are wrong (256-token threshold) | Medium | Sensitivity analysis: sweep threshold from 128 to 512; also try data-driven split (decision tree on prefill_tokens) |
| ProgressIndex is poor KV proxy for prefill requests | Medium | During prefill, ProgressIndex < InputTokens, so KV length = ProgressIndex (partial). This is correct for chunked prefill where requests are partially processed. |
| Ridge underfits like Round 1 (92% MAPE with 30 features) | Low | Regime-specific Ridge has less heterogeneity to fit; KV features are higher quality than system-state proxies. If Ridge fails, upgrade to Lasso or ElasticNet. |
| Mixed-heavy regime has too few samples (~4.4% of data) | Medium | Pool mixed-heavy data across all 4 workloads per model; if insufficient, merge with mixed-light and use a 2-regime model. |
| Secondary method constants are inaccurate | Low | Constants are validated separately against lifecycle data before BLIS integration; they add signal even if approximate. |

#### Expected Outcome

- **Conservative estimate:** 15-25% E2E mean error (improved StepTime with KV features + non-zero secondary methods)
- **Optimistic estimate:** 5-10% E2E mean error (regime-specific models with KV features close the gap identified by MIST)
- **Key diagnostic:** Per-regime MAPE breakdown reveals which batch composition types are hardest; comparison of kv_sum/kv_mean importance vs system-state proxy importance validates the KV feature hypothesis

---

#### Idea 2 Reviews

**Note on model attribution:** Same as Idea 1 -- only aws/claude-opus-4-6 was available.

##### Review 1 (aws/claude-opus-4-6) -- ACTUAL

*Model: aws/claude-opus-4-6 (self-review, as this agent is running on Claude Opus 4.6)*

**1. NOVELTY (4/5):** The combination of MIST-inspired regime switching with ProgressIndex-derived per-request KV features is well-motivated and novel in this context. No prior system uses ProgressIndex as a KV cache length proxy with regime-specific training. The three-regime split (decode-only, mixed-light, mixed-heavy) is a principled adaptation of AIConfigurator's three serving modes and MIST's data-subset training. The addition of `kv_std` as a heterogeneity indicator is a new feature not seen in prior work.

**2. FEASIBILITY (4/5):** Highly feasible. Ridge regression per regime is computationally trivial. The feature engineering is already 80% implemented in `extractBatchFeatures()` (only `kv_std` is new). The StepML artifact schema extension for regime switching is straightforward. The main feasibility concern is the mixed-heavy regime sample count (~4.4% = ~5,400 steps total, ~1,350 per model), which is adequate for Ridge regression but limits the complexity of models trained on this regime.

**3. RISK (2/5 -- MODERATE RISK):** Lower risk than Idea 1. Regime-specific linear models are interpretable and debuggable. The KV features address the #1 binding constraint from Round 1. The main risk is that ProgressIndex is an imperfect KV proxy -- it tracks cumulative tokens processed, not actual KV cache blocks (which depend on block size and memory allocation). However, ProgressIndex is monotonically related to KV cache length, so the correlation should be strong.

**4. GAP COVERAGE (4/5):** Directly and thoroughly addresses the per-request KV feature gap (Round 1's primary constraint). Uses ProgressIndex for per-request KV, adds kv_std for distribution shape, and includes interaction terms (batch_size x kv_mean, prefill x decode). Also calibrates all 5 LatencyModel methods from ground-truth data. The only gap remaining is MoE-specific features for Mixtral -- the idea treats Mixtral as "just another model" with per-model training, which may work but doesn't exploit MoE architecture knowledge.

**5. INTEGRATION (5/5):** Excellent. The Go integration path is clear: extend StepMLArtifact with a `regimes` array, add `kv_std` computation to `extractBatchFeatures()`, and dispatch to the regime-specific LinearModel. All changes are minimal extensions to existing code. The coefficient export format is trivially serializable to JSON.

**6. OVERALL: 4/5 (Accept)**

**Strengths:**
- Directly addresses the #1 binding constraint (per-request KV features via ProgressIndex)
- MIST-validated approach: regime-specific simple models with high-quality features
- Low implementation risk: Ridge regression, existing infrastructure, minimal Go changes
- All 5 LatencyModel methods covered with data-driven calibration
- Clear regime classification motivated by data (80.6% decode-only, 19.4% mixed)

**Weaknesses:**
- Regime boundaries are arbitrary (256-token threshold) -- should be data-driven
- No nonlinear modeling of attention cost (O(kv_len^2)) -- Ridge can't capture this
- MoE-specific handling is absent -- Mixtral relies on per-model training alone
- "Conservative estimate" of 15-25% still exceeds the 10% target

**Actionable suggestions:**
1. Add `kv_sum_sq` (sum of ProgressIndex^2) as a feature to capture quadratic attention cost
2. Use data-driven regime boundaries: fit a simple decision tree on `prefill_tokens` to find optimal split points
3. Consider log-transform of target (`log(step_time_us)`) to handle the 3-order-of-magnitude range
4. For Mixtral: add a binary `is_moe` feature and train a separate Mixtral ensemble with interaction terms (num_decode_reqs * kv_mean is especially important for MoE because expert routing varies per token)

##### Review 2 (Azure/gpt-4o) -- UNAVAILABLE

*Requested from: Azure/gpt-4o*
*Status: UNAVAILABLE -- No Azure OpenAI API credentials configured in this environment. Review could not be obtained.*

##### Review 3 (GCP/gemini-2.5-flash) -- UNAVAILABLE

*Requested from: GCP/gemini-2.5-flash*
*Status: UNAVAILABLE -- No GCP Vertex AI API credentials configured in this environment. Review could not be obtained.*

---

## Executive Summary: Comparison of Round 2 Ideas

### Overview

Two research ideas were generated for Round 2 of the StepML latency model fidelity research. Each targets achieving <10% workload-level E2E mean error across 16 experiments, starting from the 115% baseline.

### Idea Comparison Matrix

| Dimension | Idea 1: Bayesian Calibration | Idea 2: Regime-Switching Ensemble |
|-----------|------------------------------|-----------------------------------|
| **Primary methodology** | Multi-component Bayesian optimization with E2E objective | MIST-inspired regime-switching Ridge regression with KV features |
| **LatencyModel coverage** | All 5 methods (joint optimization) | All 5 methods (per-component calibration) |
| **StepTime approach** | Piecewise linear (~8 params) | 3-regime Ridge with ~6-8 features each |
| **Key innovation** | E2E error as direct objective | Per-request KV features + regime specialization |
| **Novelty score** | 3/5 | 4/5 |
| **Feasibility score** | 3/5 | 4/5 |
| **Risk level** | High (compensating errors fragile) | Moderate (well-understood methods) |
| **Computational cost** | ~8 hrs/model (200-500 BO evals) | ~1 hr/model (direct regression) |
| **Go integration** | Coefficient export (existing format) | Coefficient export + regime dispatch |
| **Conservative E2E estimate** | 30-50% | 15-25% |
| **Optimistic E2E estimate** | <10% | 5-10% |
| **Overall review score** | 3/5 (Weak Accept) | 4/5 (Accept) |

### Recommendation

**Idea 2 (Regime-Switching Ensemble) should be the primary research direction.** It has the best risk-adjusted return:
- Directly addresses Round 1's #1 binding constraint (per-request KV features via ProgressIndex)
- Uses proven methodology (MIST-validated ensemble approach) adapted to BLIS's constraints
- Lowest computational cost (~1 hour per model for training, ~1 hour per model for BLIS validation)
- Clearest Go integration path (minor extension to existing StepML artifact)
- Most interpretable results (per-regime coefficients, feature importance rankings)

**Idea 1 (Bayesian Calibration) should be the secondary direction**, applied AFTER Idea 2's per-step model is trained:
- Use Idea 2's regime-switching Ridge as the StepTime model
- Fine-tune the secondary method constants (QueueingTime, SchedulingProcessingTime, etc.) via BO with E2E objective
- This two-phase approach reduces BO dimensionality from 15 to ~5 parameters (secondary methods only), making convergence much more likely

### Combined Strategy

The optimal research plan combines both ideas:

1. **Phase 1 (Idea 2):** Train regime-switching ensemble with KV features. Validate through BLIS. Target: <20% E2E mean error.
2. **Phase 2 (Idea 1 partial):** Fine-tune secondary LatencyModel methods via constrained BO (5 parameters). Target: <10% E2E mean error.

### Key Insights Across All Ideas

1. **Per-request KV features are the #1 priority.** Both ideas incorporate ProgressIndex-derived KV features. The gap between Round 1's 34% per-step MAPE and MIST's 2.5% is primarily attributable to KV feature quality.

2. **E2E validation through BLIS is non-negotiable.** Round 1's 34% per-step MAPE was never validated against E2E error. Both ideas validate through the BLIS simulation harness. The relationship between per-step MAPE and E2E error is itself a key research question.

3. **Secondary LatencyModel methods provide "free" headroom.** The current model returns 0 for SchedulingProcessingTime, PreemptionProcessingTime, and alpha coefficients. Any non-zero calibration of these methods adds signal at zero risk of regression (they currently contribute nothing).

4. **Per-model calibration is universal and unavoidable.** Both ideas train per-model. This aligns with the entire literature (Vidur, MIST, AIConfigurator, RAPID-LLM).

5. **The "general" workload remains the hardest case.** Both ideas should pay special attention to the 4 general-workload experiments, which had the highest error in Round 1 due to batch composition diversity.

### Review Model Availability

Reviews were requested from three models: aws/claude-opus-4-6, Azure/gpt-4o, and GCP/gemini-2.5-flash. Only aws/claude-opus-4-6 was available in this environment (the ANTHROPIC_AUTH_TOKEN was present but Azure and GCP API credentials were not configured). All reviews above are from Claude Opus 4.6 only. To obtain diverse model perspectives, the Azure/gpt-4o and GCP/gemini-2.5-flash reviews should be re-requested when API access is available.
