# Latency Model Fidelity for LLM Inference Simulation

## Problem

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. The **ultimate metric is workload-level E2E mean error < 10%** — the accuracy of end-to-end request latency predicted by the simulator compared to ground truth.

E2E latency in BLIS is the sum of **6 delay components**, each contributed by the `LatencyModel` interface or emergent from simulation dynamics:

```
E2E(request) = QueueingTime(req)              [1] arrival-to-queue delay
             + (time waiting in WaitQ)         [2] emergent from simulation dynamics
             + SchedulingProcessingTime()      [3] per-request scheduling overhead
             + Σ StepTime(batch)               [4] GPU execution per batch step
             + Σ OutputTokenProcessingTime()   [5] per-token post-processing
             + KV transfer latency             [6] CPU↔GPU offload/reload
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

### Round 2 Findings

Round 2 tested 2 ideas on 10 experiments (4 models x 3 workloads, no reasoning). Both ideas added ProgressIndex-derived KV features, regime-based step-time splitting, and BLIS E2E validation. **All hypotheses across both ideas were refuted or not met.** Full details: `round2/FINDINGS_ROUND2.md`.

#### Idea 1: Bayesian Calibration (2-Regime Piecewise Linear)

2 regimes (decode-only vs mixed-batch), Ridge per regime, KV features (kv_sum/kv_max/kv_mean).

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: Piecewise StepTime | <30% MAPE | 87.4% | Refuted |
| H2: Joint BO | <15% E2E | Not run (H1 too inaccurate) | Blocked |
| H4: LOMO | <80% MAPE | 148.8% | Refuted |
| H5: LOWO | <40% MAPE | 155.4% | Refuted |

#### Idea 2: Regime-Switching Ensemble (3-Regime Ridge) — Best Round 2 Result

3 regimes (decode-only / mixed-light<256 / mixed-heavy>=256), Ridge per regime, KV features, BLIS E2E validation, secondary method calibration.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: Regime Ridge MAPE | <15% | 64.4% | Not met |
| H2: BLIS E2E | <10% mean | 427.8% | Not met |
| H3: Secondary calibration | >=5pp E2E gain | 0.0pp | Refuted |
| H4: LOMO | <80% MAPE | 108.6% | Refuted |
| H5: LOWO | <25% MAPE | 117.4% | Refuted |

#### Round 2 Per-Experiment BLIS E2E Results (Idea 2)

| Experiment | E2E Error | TTFT Error | ITL Error | Status |
|---|---|---|---|---|
| 34b-roleplay | **3.3%** | 282.7% | 100.1% | E2E solved (likely coincidental) |
| mixtral-roleplay | 50.8% | 77.2% | **0.7%** | ITL solved |
| mixtral-codegen | 51.5% | 76.5% | **2.1%** | ITL solved |
| 7b-roleplay | 52.5% | 78.4% | **4.0%** | ITL solved |
| mixtral-general | 554.6% | 44,464% | **8.7%** | ITL solved |
| 70b-hf-codegen | 55.7% | 77.7% | 10.7% | ITL close |
| 70b-roleplay | 55.6% | 77.9% | 10.3% | ITL close |
| 70b-general | 182.7% | 12,567% | 23.5% | Unsolved |
| 34b-codegen | 370.2% | 30,425% | 95.9% | Unsolved |
| 34b-general | 2,901.1% | 230,931% | 80.3% | Unsolved |

**Key observations:**
- **ITL is nearly solved** — 33.6% mean, 5/10 <10%, driven by the overhead floor mechanism
- **TTFT errors dominate E2E** — 31,906% mean TTFT makes step-time improvements irrelevant until fixed
- **CodeLlama-34B is a consistent outlier** — worst across all metrics
- **"General" workloads are hardest** — largest TTFT errors for every model

#### Round 2 Critical Discovery: Overhead Floor Mechanism

The `max(overhead, compute)` step-time floor is the single most impactful technique. Per-model overhead constants (derived from ground-truth ITL):

| Model | Overhead (us) |
|---|---|
| Llama-2-7B (TP1) | 3,897 |
| CodeLlama-34B (TP2) | 6,673 |
| Llama-2-70B (TP4) | 8,029–8,203 |
| Mixtral-8x7B (TP2) | 9,125 |

`step.duration_us` captures GPU forward pass only (~70–7,000us). Real step cycle time includes CPU overhead (scheduling, sync, memory management). The floor handles 77.9% of steps (decode-only) by clamping small-batch predictions to real cycle time.

#### Round 2 Critical Discovery: KV Features Counter-Productive

The central thesis of Round 2 — that ProgressIndex-derived KV features would improve prediction — was **refuted in raw linear space**:
- Idea 1: KV features hurt by +3.6pp (87.4% vs 83.8% without)
- Idea 2: KV features hurt by +20.5pp (64.4% vs 43.9% without)

**Root cause:** kv_sum ranges 0–64,000+, causing Ridge coefficient instability. Only Mixtral benefited (-5.4pp). This is a **formulation problem, not a feature problem** — the features have signal but raw linear regression cannot handle their dynamic range. Feature scaling (StandardScaler), log-transform of features, or nonlinear models would likely fix this.

#### Round 2 Overhead Integration Approaches (All Tested)

| Approach | Result |
|---|---|
| Additive (compute + overhead) | Phase transition — under-capacity to overload, no stable point |
| Log-space cycle time | Exponential amplification — coefficients get multiplied, not added |
| **Max floor (best)** | `step_time = max(overhead, compute)` — handles memory→compute crossover |
| Floor + cap | `max(overhead, min(compute, 3*overhead))` prevents exp blowup |

Log-space expm1 pitfalls: kv_sum coefficient 0.0001 × 64,000 = 6.4 → 601x multiplier. Prefill coefficient 0.06 × 500 = 30 → exp(30) = absurd. Use `use_log_target=False` for all BLIS-destined regimes.

#### Round 2 Secondary Methods Finding

Secondary LatencyModel methods (QueueingTime, SchedulingProcessingTime) contribute **exactly 0.0pp** to E2E improvement. Extracted constants are 200–400us — invisible against 100%+ E2E errors. Do not invest in secondary methods until the dominant error source (TTFT mismatch) is resolved.

#### Round 2 Generalization Progress

| Metric | Round 1 | Round 2 | Change |
|---|---|---|---|
| LOMO avg MAPE | 2,559.7% | 108.6% | **23.6x better** |
| LOWO avg MAPE | 109.7% | 117.4% | 0.9x (no improvement) |

LOMO improved dramatically due to regime structure. LOWO regressed slightly — the regime structure did not help cross-workload transfer.

**Mixtral generalizes exceptionally:** LOWO 19.1% avg (below 25% target), ITL 0.7%/2.1%/8.7%. A single Mixtral model works across all workloads.

### Scope for Round 3

The E2E target (<10%) requires addressing **two independent problems**:

1. **TTFT/simulation fidelity** (PRIMARY BLOCKER) — TTFT errors of 31,906% mean indicate a systematic mismatch between BLIS's request lifecycle and the real system. This is not a step-time problem. Until TTFT is fixed, no step-time improvement can achieve the E2E target.

2. **Step-time accuracy for remaining experiments** — ITL is solved for 5/10 experiments via the overhead floor. The remaining 5 need either better step-time models (feature-scaled KV, nonlinear models) or 34B-specific investigation.

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

general, codegen, roleplay — each with different input/output length distributions. All runs: `max_model_len=4096`, `max_num_batched_tokens=2048`, `max_num_seqs=128`, chunked prefill enabled, prefix caching enabled.

### Step-Level Features

**Batch computation features** (causal):
- `batch.prefill_tokens` (int) — prefill tokens in this step
- `batch.decode_tokens` (int) — decode tokens in this step
- `batch.num_prefill_reqs` (int) — prefill request count
- `batch.num_decode_reqs` (int) — decode request count
- `batch.scheduled_tokens` (int) — total scheduled tokens

**System state features** (potentially spurious):
- `queue.running_depth`, `queue.waiting_depth` — queue state
- `kv.usage_gpu_ratio`, `kv.blocks_free_gpu`, `kv.blocks_total_gpu` — KV cache state

**Target**: `step.duration_us` (int) — wall-clock step execution time in microseconds

### Data Characteristics

Step durations span 3+ orders of magnitude:
- Smallest: ~12 μs (small decode-only batches)
- Largest: ~250,000 μs (large prefills)
- Roleplay/codegen: mean 160–320 μs (short, bursty)

**Phase distribution** (from Round 2 10-experiment dataset, 77,816 steps):

| Regime | Steps | Share |
|---|---|---|
| Decode-only (prefill = 0) | 60,613 | 77.9% |
| Mixed-light (0 < prefill < 256) | 17,129 | 22.0% |
| Mixed-heavy (prefill >= 256) | 74 | 0.1% |

vLLM's chunked prefill makes nearly every batch mixed or pure-decode. Only 6 pure-prefill steps exist in the full 16-experiment dataset. The mixed-heavy regime (prefill>=256) is effectively empty without reasoning workloads.

**Step data is ~10% sampled** — not every step is logged (OpenTelemetry tracing at ~10% rate).

### Known Feature Gaps

1. **No per-request KV cache lengths**: Only aggregate batch tokens available. Attention FLOPs scale with per-request kv_len (H8 showed 12.96× overestimate without per-request KV). The simulator's `ProgressIndex` (input_tokens_processed + output_tokens_generated) is available as a proxy at inference time.
2. **No MoE-specific features**: No active expert count, expert load balance, or tokens-per-expert.
3. **No prefix cache hit information**: `prefill_tokens` may reflect pre- or post-cache-hit count.

### Additional Data Sources

- **Per-request lifecycle data**: Per-token timestamps, input/output token counts — enables per-request KV length derivation and E2E validation
- **MFU benchmarks** (`bench_data/`): Kernel-level GEMM and attention MFU data by GPU — useful for physics-informed features

## Baseline Results (WP0)

### BLIS E2E Validation — Per-Model Linear Regression (THE NUMBER TO BEAT)

Per-model+TP linear regression (one regression per model group, pooling all workloads):
- **Mean E2E error: 115.0%** (0/12 experiments below 10% target)
- Mean TTFT error: 102.9%
- Mean ITL error: 134.6%
- **R4 gate: PASSED** — blackbox is clearly insufficient, research is justified

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

### BLIS E2E Validation — Roofline (Informational Only)

The analytical roofline model (zero calibration, FLOPs/bandwidth-based):
- **Mean E2E error: 4816%** (10/16 completed, 6 timed out at 300s)
- Massively overestimates step time, causing request starvation
- Only Mixtral codegen/roleplay were within 35% (GEMM-dominated workloads)
- The roofline model is informational only — ideas MUST NOT use roofline as input

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
- **Per-experiment XGBoost (30 features):** 34.0% avg MAPE — 2x better than blackbox (70.4%)
- **Global blackbox:** 670% MAPE — catastrophic failure across model scales
- **Key insight:** Per-step MAPE ≠ E2E error. The 34% per-step MAPE was never validated through BLIS.

### Validation Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| R1 (ProgressIndex as KV proxy) | FLAGGED | Pearson r ≈ 0 between total_tokens and E2E time; ProgressIndex is valid for KV cache length but not for E2E prediction. Lifecycle KV extractor available as fallback. |
| R2 (Sampling bias) | PASS | CV=0.44, max/min ratio=4.2, random sampling confirmed (~10% rate) |
| R4 (Blackbox sufficient?) | PASS | 115% E2E mean error — clearly beatable |

## Evaluation Framework

### Primary Metric

**Workload-level E2E mean error < 10%** on each of the 12 experiments individually. For each experiment:
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
| **P2** | **Generalization: model (LOMO)** | **LOMO MAPE < 80% per fold** |
| **P2** | **Generalization: workload (LOWO)** | **LOWO MAPE < 50% per fold** |
| **P2** | **Generalization: vLLM args sensitivity** | **Qualitative robustness analysis per idea** |
| P2 | TTFT mean fidelity | < 15% per experiment |
| P2 | ITL mean fidelity | < 15% per experiment |
| P3 | Tail behavior (p99) | No ranking inversions vs baseline |
| P3 | Ease of use, retraining story, reproducibility | Qualitative |
| P4 | Hardware generalization (H100 → A100) | Informational |
| P5 | Quantization transferability | Informational |

> **Generalization is P2, not optional.** An idea that achieves <10% E2E but fails LOMO/LOWO/vLLM-args analysis is **not a viable winner** — it would be useless in production where new models, workloads, and vLLM configurations are deployed routinely. Reviewers MUST evaluate generalization with the same rigor as accuracy.

### Data Split Strategy

- **Primary**: Temporal split (60/20/20) within each experiment — prevents autocorrelation leakage
- **Generalization**: Leave-one-model-out (4-fold) + leave-one-workload-out (3-fold)
- **Short-circuit**: Ideas with h1 per-step MAPE > 30% are dropped (threshold calibrated to blackbox_MAPE + 10% if blackbox exceeds 25%)

### Idea Review Criteria

Beyond quantitative accuracy, idea reviewers evaluate each approach against the following dimensions. The standard is that **the approach must be defensible to a community of reviewers at a top-tier systems/ML conference or journal** (e.g., OSDI, SOSP, NSDI, MLSys, EuroSys).

**BLOCKING dimensions** — reviewers MUST reject ideas that do not address these. An idea missing any blocking dimension cannot proceed to experimentation:

| Dimension | Blocking? | What reviewers check |
|---|:---:|---|
| **Accuracy** | **YES** | Does the approach meet the P1 target (<10% E2E mean error)? Are per-step and E2E metrics both reported? |
| **Generalization across LLMs (LOMO)** | **YES** | Does the idea include a concrete LOMO experimental plan? Does its formulation use model-agnostic features (FLOPs, parameter count, TP degree) or model-specific memorization? Would it work on a model not in the training set (e.g., Llama-3, Qwen)? Reviewers should flag ideas where the formulation structurally prevents cross-model transfer. |
| **Generalization across workloads (LOWO)** | **YES** | Does the idea include a concrete LOWO experimental plan? Does its formulation depend on specific input/output length distributions, or does it capture workload-invariant compute physics? Reviewers should flag ideas that would fail on an unseen workload type (e.g., summarization, RAG). |
| **Generalization across vLLM config** | **YES** | Does the idea include a sensitivity analysis plan for the 7 key vLLM parameters (max_num_seqs, max_num_batched_tokens, max_model_len, chunked_prefill, prefix_caching, gpu_memory_utilization, tensor_parallel_size)? Does the formulation structurally depend on batch distribution shapes determined by these parameters? Reviewers should flag ideas where changing `max_num_seqs` from 128→256 would invalidate the model without retraining. |
| **Generalization across hardware** | No | Does the approach transfer across GPU generations (H100 → A100 → B200)? What changes when memory bandwidth, compute FLOPs, or GPU memory capacity differ? |
| **Ease of use** | No | How much expertise is required to deploy the model for a new model/GPU/config combination? Is the workflow self-service or does it require ML expertise? |
| **Retraining story** | No | Is retraining or fine-tuning required? Under what circumstances (new model, new GPU, new vLLM version, new workload)? How much data and compute does retraining require? How easy is the retraining process end-to-end? |
| **vLLM version sensitivity** | No | What happens when the serving engine (vLLM) changes versions? Do internal scheduling changes (e.g., new chunked prefill heuristics, changed batch formation logic) invalidate the model's assumptions? How is this detected and corrected? |
| **Overheads** | No | What are the training-time costs (data collection, compute, human effort)? What are the inference-time costs (latency per prediction, memory footprint, complexity of the Go integration)? |
| **Reproducibility** | No | Can results be reproduced from the provided code, data, and instructions? Are random seeds fixed? Are all hyperparameters documented? Is the training pipeline deterministic? |

> **Reviewer instruction:** When reviewing ideas generated by `/research-ideas`, evaluate the top 4 blocking dimensions FIRST. If an idea has no LOMO plan, no LOWO plan, or no vLLM-args sensitivity plan, flag it as incomplete regardless of how good the accuracy approach looks. The purpose of this research is to find a **production-viable** latency model, not a benchmark-winning one.

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
- `ProgressIndex` (cumulative: input_processed + output_generated — KV cache length proxy)
- `NumNewTokens` (tokens to generate this step)

Research models must use only features derivable from these fields plus experiment-level metadata (model name, TP degree).

### Inference Latency

Prediction must complete in <1ms per step (128 requests max). Linear models, tree ensembles (~100 trees), and small neural networks all qualify. Only large neural networks (>10M parameters) or GPU-inference-requiring approaches are excluded.

### Go Integration Path

The winning model must eventually run in Go. Viable paths:
1. **Coefficient export** — for parametric models (regression, piecewise linear)
2. **Go-native reimplementation** — for tree ensembles (Go libraries available)
3. **ONNX export** — for neural networks

Each idea must specify which integration path it would use.

## Generalization Requirements (MANDATORY — P2 PRIORITY)

**Every idea must explicitly design for and test generalization across three dimensions.** An idea that achieves <10% E2E on training experiments but cannot generalize is not a viable solution — production deployments routinely introduce new models, new workload patterns, and changed vLLM serving configurations. Reviewers treat generalization failures as seriously as accuracy failures.

### Dimension 1: Model Generalization (LOMO)

**Test:** Leave-one-model-out cross-validation (4-fold: hold out each of llama-2-7b, llama-2-70b, codellama-34b, mixtral-8x7b).
**Target:** LOMO per-step MAPE < 80% per fold.
**Why it matters:** Users deploy BLIS for capacity planning across model families. A latency model that requires retraining from scratch for every new model is impractical. At minimum, the model should transfer between models of similar architecture/scale (e.g., 70B → 34B dense), and degrade gracefully for architecture shifts (dense → MoE).
**What ideas must specify:** (a) Which model features enable transfer (parameter count, architecture type, TP degree)? (b) What retraining is needed for a truly new model (e.g., Llama-3, Qwen)? (c) How much data from the new model is required?

### Dimension 2: Workload Generalization (LOWO)

**Test:** Leave-one-workload-out cross-validation (3-fold: hold out each of general, codegen, roleplay).
**Target:** LOWO per-step MAPE < 50% per fold.
**Why it matters:** Real deployments serve diverse traffic mixes. A model trained only on codegen traffic that fails on conversational traffic is brittle. The workloads in our dataset (general, codegen, roleplay) differ primarily in input/output length distributions and burstiness — the model should capture the underlying compute physics, not memorize specific length distributions.
**What ideas must specify:** (a) Which features are workload-invariant vs workload-specific? (b) How does the approach handle unseen input/output length distributions? (c) Would a new workload type (e.g., summarization, RAG) require retraining?

### Dimension 3: vLLM Configuration Sensitivity (MANDATORY ANALYSIS)

**Test:** Qualitative sensitivity analysis — for each tunable vLLM serving parameter, describe whether the model's predictions would change and whether retraining/recalibration would be needed.
**Target:** The idea must document its sensitivity to each parameter below and propose a recalibration strategy that does NOT require full retraining from step-level traces.

| vLLM Parameter | Default in Data | Range in Practice | Why It Matters |
|---|---|---|---|
| `max_num_seqs` | 128 | 32–512 | Controls maximum batch size → changes batch composition distributions |
| `max_num_batched_tokens` | 2048 | 512–8192 | Controls chunked prefill budget → changes mixed-batch frequency |
| `max_model_len` | 4096 | 2048–131072 | Controls KV cache capacity → changes eviction/preemption behavior |
| `chunked_prefill` | enabled | on/off | Fundamentally changes batch composition (pure-prefill vs mixed batches) |
| `prefix_caching` | enabled | on/off | Changes effective prefill tokens per step |
| `gpu_memory_utilization` | 0.9 | 0.5–0.95 | Changes KV cache capacity → changes batch sizes under memory pressure |
| `tensor_parallel_size` | varies | 1–8 | Changes per-GPU compute/memory → changes step time scaling |

**What ideas must specify for EACH parameter:** (a) Does the model implicitly depend on this parameter's value? (b) If the parameter changes, would the model's predictions be wrong? (c) What is the minimum recalibration needed (nothing / update one constant / retrain one component / full retrain)?

**Reviewers must evaluate:** Does the idea's formulation *structurally* depend on the training data's vLLM config, or does it capture generalizable compute physics? A model that learns `step_time = f(batch_size)` where the relationship shape depends on `max_num_seqs=128` is fragile. A model that learns `step_time = f(FLOPs, memory_bandwidth)` is robust.

### Historical Generalization Results (for calibrating targets)

| Round | Idea | LOMO | LOWO | vLLM Args |
|---|---|---|---|---|
| R1 | Tree ensemble | 2,559.7% | 109.7% | Not analyzed |
| R2 | Bayesian calibration | 148.8% | 155.4% | Not analyzed |
| R2 | Regime ensemble | 108.6% | 117.4% | Not analyzed |
| R3 | Total-context model | 2,281.6% (REFUTED) | 2,162.7% (REFUTED) | Not analyzed |
| R3 | CMA-ES calibration | **14.8% E2E** (SUPPORTED) | Partial (8/10 within 2×) | Not analyzed |
| R4 | Constrained CMA-ES | Not run (short-circuited) | Not run | Not analyzed |
| R4 | Cycle-time regression | 2/4 pass | 0/3 pass | Not analyzed |
| R4 | Direct calibration (H1) | **30.7% E2E** (3/4 pass) | 4/4 pass (5-8pp range) | Not analyzed |

**Trend:** LOMO saw breakthroughs in R3-R4 — CMA-ES artifact transfer (R3: 14.8%) transfers better than direct calibration coefficients (R4: 30.7%) because CMA-ES captures simulation dynamics while direct calibration produces model-specific constants. LOWO is confirmed solved for dense models (5-8pp range in R4). vLLM args sensitivity has never been experimentally tested in any round, though R4 documents the structural dependence and notes low recalibration cost.

## Algorithm Scope

**Not restricted to ML or to StepTime.** Research ideas may propose:
- Statistical regression (Ridge, Lasso, polynomial, piecewise)
- Tree ensembles (XGBoost, LightGBM, random forest)
- Neural networks (small MLPs, attention-based)
- Physics-informed models (analytical compute model + learned residuals)
- Hybrid approaches (analytical backbone + ML residual correction)
- **Multi-component calibration** (jointly optimize all 5 LatencyModel methods for E2E fidelity)
- **Scheduling/preemption overhead models** (data-driven models for the currently-zero methods)
- **End-to-end calibration** (tune any/all components with E2E mean error as the objective, not per-component accuracy)

Each approach must:
1. Cite relevant prior work from systems/ML literature
2. Address all evaluation dimensions (P1–P6)
3. Specify which LatencyModel methods it covers (minimum: StepTime; **bonus: additional methods**)
4. Document its Go integration path
5. Be distinct from other proposed approaches
6. Include a **model-generalization hypothesis (LOMO — MANDATORY, MUST EXECUTE)** — a leave-one-model-out experiment (train on 3 models, predict on the held-out 4th) that evaluates whether the approach transfers to unseen model architectures and scales. This hypothesis must be **actually executed** with results reported in FINDINGS_SUMMARY.md section 9, not just scaffolded. If blocked by prior sub-hypothesis failure, document "Not executed — blocked by [reason]" with root cause.
7. Include a **workload-generalization hypothesis (LOWO — MANDATORY, MUST EXECUTE)** — a leave-one-workload-out experiment (train on 2 workloads, predict on the held-out 3rd) that evaluates whether the approach transfers to unseen workload distributions. Same execution requirement as item 6.
8. **Address vLLM-args sensitivity at ideation time (MANDATORY — checked during idea review, NOT a hypothesis).** Each idea's description in research.md must include a section analyzing its structural dependence on the 7 key vLLM parameters listed in the "Generalization Requirements" section above. For each parameter: (a) does the formulation depend on it, (b) would predictions break if it changed, (c) what recalibration is needed. This is a **design-time check**, not an experiment — reviewers evaluate it during WP1 idea review and reject ideas that are structurally fragile without a recalibration story.
   > **Historical note:** Rounds 1-3 showed a pattern of generalization hypotheses being scaffolded but never run (R1 idea-2 h3, R3 idea-2 h4/h5, R3 idea-3 h4/h5), and vLLM args sensitivity was never analyzed in any round. The macro plan now enforces LOMO/LOWO execution via WP4 Step 0 gate. If an idea is infrastructure-only (not a new model), provide a GENERALIZATION_NOTE.md with cross-experiment evidence.

## Key Questions for Ideas to Address

### Answered by Round 1+2 (DO NOT re-investigate)

| # | Question | Answer | Round |
|---|---|---|---|
| Q1 | Which LatencyModel component contributes most to E2E error? | **StepTime is 100% of modeled error** (alpha coeffs are all zero). But TTFT/simulation mismatch dominates actual E2E error. | R2 |
| Q2 | Does per-step MAPE translate to E2E error? | **No simple relationship.** 64.4% per-step MAPE → 427.8% E2E error, but 33.6% ITL error. TTFT errors dominate E2E, not step-time errors. | R2 |
| Q3 | Can non-zero SchedulingProcessingTime/PreemptionProcessingTime improve E2E? | **No.** They contribute 0.0pp improvement — 200-400us corrections are invisible against 100%+ errors. | R2 |
| Q4 | Can we derive per-request KV features from lifecycle data? | **Yes, ProgressIndex proxy is available.** But raw KV features in linear regression are counter-productive (+20.5pp worse). Need feature scaling. | R2 |
| Q5 | Per-model vs global training? | **Per-model mandatory.** 3+ OOM step-time scale variation. LOMO >100% even after 23.6x improvement. | R1+R2 |

### Open questions for Round 3

1. **What causes the 31,906% mean TTFT error?** Is it the workload spec generator (request arrival patterns), BLIS request injection timing, BLIS prefill scheduling, or some other simulation-level mismatch with real vLLM? This is the PRIMARY blocker for <10% E2E.

2. **Can feature scaling (StandardScaler, log-transform) make KV features productive?** KV features have signal but their 0–64,000 range causes Ridge instability. The right normalization or a nonlinear model should unlock them.

3. **Why is CodeLlama-34B consistently the worst model?** Is it a data quality issue, architecture-specific batch dynamics, or a coefficient problem? 34B accounts for disproportionate error (99.2% per-step MAPE, 2,901% E2E for general).

4. **Can the mixed-heavy regime be made useful?** Only 74/77,816 steps (0.1%) have prefill>=256. Should the threshold lower to 64/128, or should reasoning workloads be included in the dataset?

5. **How to handle 3+ OOM step time range across models for cross-model transfer?** LOMO improved 23.6x but remains >80%. Feature normalization by model-specific constants (parameter count, FLOPS/token) might close the gap. The 34B interpolation result (63.4% LOMO fold) suggests interpolation between scales is feasible.

6. **Is end-to-end calibration (E2E objective) more practical than per-component accuracy?** Per-step MAPE is a poor proxy for E2E error. Directly optimizing coefficients against BLIS E2E might find better trade-offs, but only after the TTFT mismatch is resolved.

## Successful Techniques to Build On

1. **Overhead floor (`max(overhead, compute)`)** — the most impactful technique, produces 5/10 ITL <10%
2. **Per-model overhead constants** — stable across workloads: 7B=3,897us, 34B=6,673us, 70B=8,029-8,203us, Mixtral=9,125us
3. **Per-model training** — mandatory, non-negotiable
4. **Regime separation** (decode-only vs mixed) — distinct modeling contexts, 23.6x LOMO improvement
5. **Temporal train/test split** — prevents autocorrelation leakage
6. **BLIS E2E validation pipeline** — `validate_e2e.py` + `validate_blis.py` infrastructure for running BLIS with custom coefficients
7. **Lifecycle KV extractor** — derives per-request KV proxy from lifecycle data
8. **Mixtral universal model** — MoE architecture generalizes across all workloads (LOWO 19.1%)
9. **Trace replay mode** (`--workload traces --workload-traces-filepath <csv>`) — eliminates workload-spec generation artifacts; all E2E experiments should use this as default mode
10. **CMA-ES for black-box DES calibration** — effective for optimizing noisy multi-experiment objective. Per-model, 96-152 evaluations, ~30 min/model
11. **Per-model TTFT additive corrections** — simple per-model constants (16-61ms) reduce TTFT from 67.6% to 9.4%
12. **FairBatching 3-coefficient OLS** — `a + b*new_tokens + c*kv_sum` outperforms complex feature engineering (56.2% vs 83.1% per-step MAPE)
13. **StepML CLI integration** — complete `--stepml-model` flag + factory dispatch + Go evaluator path
14. **CMA-ES cross-model artifact transfer** — CMA-ES artifacts generalize dramatically better than per-step models (14.8% LOMO E2E vs 2,281.6% LOMO MAPE) because they capture simulation dynamics (overhead floor, scheduling), not model-specific step times. Mixtral is the best universal donor (11.9–26.0%). 70B→Mixtral transfer (5.1%) outperforms Mixtral's in-distribution result (25.9%).
15. **Direct E2E calibration** — `target_step = (E2E_mean - TTFT_mean) / output_len_mean`, then `beta0 = target_step - beta2 * avg_batch`. Achieves 5.7% mean E2E (9/10 <10%). The single most impactful technique across all 4 rounds.
16. **Large additive intercept (beta0)** — beta0 accounts for 92-98% of step time. GPU forward pass is only 2-8%. The intercept absorbs CPU scheduling, CUDA sync, memory management, and all other non-GPU overhead.
17. **Per-model TTFT as alpha0** — simple lifecycle mean TTFT (27-89ms) as a constant per model. No input-length dependence needed when TTFT is <2% of E2E.
18. **LOWO stability confirmed** — workload variation is a non-issue for dense models (5-8pp range across codegen/general/roleplay).
19. **Power law meta-regression for beta0** — `beta0 = 5629 × (params_B)^0.285` predicts the dominant overhead component (92-98% of step time) from model metadata with <5% coefficient error. Enables LOMO 14.3% (vs 30.7% with coefficient transfer).
20. **Hybrid metadata + step-data approach** — metadata for overhead (beta0, alpha0), per-model step data for GPU compute (beta2). Best accuracy–generalization tradeoff: 8.6% E2E, 14.3% LOMO.
21. **BLIS validation with `clients` workload-spec format** — explicit clients with constant distributions (not `inference_perf` format). Produces accurate results and runs 7× faster.

## Eliminated Approaches (Do NOT Repeat)

| Approach | Why It Failed | Round |
|---|---|---|
| Global models (all experiments) | 3+ OOM step-time scale variation | R1 |
| Analytical FLOPs decomposition without per-request KV | decode_attention = 0, structurally incomplete | R1 |
| Correction factors on incomplete backbone | Pathological compensation (factor=39.5 on zero signal) | R1 |
| Pure-phase analysis | Only 6 pure-prefill steps under chunked prefill | R1 |
| Raw KV features in linear regression | kv_sum up to 64K causes Ridge coefficient instability (+20.5pp worse) | R2 |
| 2-regime piecewise linear | Too coarse — mixed-batch heterogeneity too high for one regime | R2 |
| Joint Bayesian optimization on inaccurate base model | Cannot compensate for structural model errors | R2 |
| Secondary method calibration alone | 200-400us corrections invisible against 100%+ E2E errors | R2 |
| Log-space target with large features (expm1) | Exponential blowup: coeff×64K → exp(6.4) = 601x multiplier | R2 |
| Additive overhead (compute + overhead) | Phase transition from under-capacity to overload, no stable point | R2 |
| Feature scaling for multicollinear KV features | StandardScaler, log-transform, combinations all failed — issue is multicollinearity (4 correlated features), not scale | R3 |
| Per-step MAPE as proxy for E2E accuracy | 27pp per-step improvement → 0pp E2E improvement; overhead floor masks per-step gains | R3 |
| Workload-spec mode for E2E validation | Workload-spec introduces systematic 31,906% TTFT errors; use trace replay instead | R3 |
| Post-hoc corrections on CMA-ES results | TTFT additive corrections conflict with CMA-ES-optimized coefficients (+2.1pp worse) | R3 |
| Unconstrained CMA-ES optimization | Without ITL constraints, produces implausible parameter values (e.g., output_token_processing_time 0→1,899µs) | R3 |
| Constrained CMA-ES (physical bounds) | "Physical" bounds prevent compensation for unmodeled dynamics (51.2% E2E vs 15.1% unconstrained) | R4 |
| Workload-spec mode for E2E (3rd attempt) | 30,412% TTFT error confirmed dead; use trace replay | R4 |
| CMA-ES residual on well-calibrated base | Starting at E2E optimum, any CMA-ES change degrades E2E (5.7%→27.5%) | R4 |
| Pareto α sweep (E2E vs ITL) | No Pareto knee within constrained parameter space; E2E↔ITL tradeoff is architectural | R4 |
| Hierarchical two-stage (cross-model residuals) | Residuals systematically negative → Ridge intercept = -22,254, decode_coeff = 665; predictions 20-100× too high. Cross-model step-level training confirmed dead for 4th time. | R5 |

### Round 3 Findings

Round 3 tested 3 ideas on 10 experiments (5 models × 3 workloads). **Key achievement: 427.8% → 15.1% mean E2E (28.3x improvement)** via two advances: trace replay eliminated workload-spec errors, CMA-ES found per-model coefficient settings. Full details: `round3/FINDINGS_ROUND3.md`.

#### Idea 1: Trace-Driven Simulation with Lifecycle Replay

Bypass BLIS's workload-spec generation by replaying ground-truth request traces via legacy CSV format.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: Trace replay reduces TTFT | <100% TTFT | 78.8% (was 31,906%) | Supported |
| H2: Remaining E2E <25% | <25% E2E | 56.2% | Refuted |
| H3: Root cause workload-spec param | Identify param | All params <1% error | Refuted |

#### Idea 2: Total-Context Linear Model with Feature Scaling

Replace 2-feature model with 3-feature (`new_tokens + total_context`) using FairBatching formulation.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: FairBatching 3-coeff OLS | <50% per-step MAPE | 56.2% (vs 83.1% baseline) | Partially Supported |
| H2: Feature scaling variants | <43.9% per-step MAPE | 83.0% best | Refuted |
| H3: BLIS E2E + 34B deep-dive | <50% E2E | 56.2% (identical to R2 trace) | Refuted |
| H4: LOWO generalization | <70% per-step MAPE | 2,162.7% | Refuted |
| H5: LOMO generalization | <80% per-step MAPE | 2,281.6% | Refuted |

#### Idea 3: End-to-End Calibration via CMA-ES — Best Round 3 Result

Jointly calibrate all LatencyModel coefficients by directly minimizing BLIS E2E mean error using CMA-ES.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: CMA-ES + trace replay → E2E | <15% E2E | **15.1%** | Partially Supported |
| H2: Workload-spec mode | <50% E2E | Not tested (infra gap) | Not Tested |
| H3: Additive TTFT corrections | -5pp E2E | +2.1pp worse; TTFT 9.4% | Refuted for E2E |
| H4: LOWO generalization | All within 2× aggregate | 8/10 within 2× (15.1% agg) | Partially Supported |
| H5: LOMO cross-model transfer | <50% E2E | **14.8%** mean best-donor | Supported |

#### Round 3 Per-Experiment BLIS E2E Results (CMA-ES, Best)

| Experiment | E2E Error | TTFT Error | ITL Error | Status |
|---|---|---|---|---|
| codellama-34b-general | **1.2%** | 69.3% | 102.7% | E2E solved |
| llama-2-70b-hf-codegen | **3.8%** | 67.6% | 111.6% | E2E solved |
| codellama-34b-codegen | **5.0%** | 68.9% | 90.2% | E2E solved |
| llama-2-70b-roleplay | **6.9%** | 67.6% | 117.0% | E2E solved |
| mixtral-general | 12.7% | 57.3% | 78.1% | Close |
| llama-2-70b-general | 16.2% | 82.5% | 72.0% | Unsolved |
| codellama-34b-roleplay | 17.5% | 66.2% | 135.4% | Unsolved |
| llama-2-7b-roleplay | 22.2% | 60.8% | 91.1% | Unsolved |
| mixtral-codegen | 30.9% | 66.4% | 41.5% | Unsolved |
| mixtral-roleplay | 34.2% | 69.3% | 34.5% | Unsolved |
| **MEAN** | **15.1%** | **67.6%** | **87.4%** | **4/10 solved** |

#### Round 3 Critical Discovery: E2E ↔ ITL Tradeoff

CMA-ES achieves 15.1% E2E but at 87.4% ITL (was 9.5% with trace replay baseline). **Root cause:** E2E = TTFT + Σ(ITL). Optimizing total E2E via shared coefficients produces arbitrarily bad ITL. CMA-ES uses `output_token_processing_time_us` as a proxy for missing simulation dynamics (e.g., llama-2-7b: 0→1,899µs). Need multi-objective or constrained optimization.

#### Round 3 Critical Discovery: BLIS Simulates a "Faster Universe"

With trace replay, every experiment shows BLIS completing requests in ~40% of real time. The overhead floor needs approximately doubling (from ~4ms to ~8ms for 7B, proportionally for larger models) to match real vLLM step cycle times.

#### Round 3 Critical Discovery: Per-Step MAPE ↛ E2E

A 27pp per-step improvement (83%→56%) has **ZERO impact** on BLIS E2E. The overhead floor dominates 70-90% of steps, completely masking GPU compute prediction improvements. **Optimize E2E directly, not per-step metrics.**

### Round 4 Findings

Round 4 tested 3 ideas on 10 experiments (4 models × 3 workloads, minus 2 missing llama-2-7b workloads). **Key achievement: 15.1% → 5.7% mean E2E (2.6× improvement)** via direct calibration from E2E ground truth. 9/10 experiments below 10%. Full details: `round4/FINDINGS_ROUND4.md`.

#### Idea 1: Constrained CMA-ES — REFUTED

Constrain CMA-ES parameter bounds to physically plausible ranges and add ITL penalty term.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: Constrained CMA-ES (α=0.5) | <15% E2E, <25% ITL | 51.2% E2E, 2.7% ITL | Refuted |
| H2: Pareto sweep (α=0.7) | E2E<15% AND ITL<25% | 42.9% E2E, 17.5% ITL | Refuted |
| H3: LOMO | Not run | Short-circuited (51.2% >> 25%) | Not Run |
| H4: LOWO | Not run | Short-circuited | Not Run |

**Root cause:** "Physical" parameter bounds prevent CMA-ES from compensating for unmodeled simulation dynamics. The parameters are regression coefficients, not physical quantities — they must absorb scheduling, sync, memory overhead not explicitly modeled in BLIS.

#### Idea 2: Cycle-Time Regression — REFUTED

Train step-time model from FairBatching 3-coefficient OLS on cycle-time targets using lifecycle data.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: FairBatching OLS | <15% E2E | 452.2% E2E | Refuted |
| H2: LOMO | <80% E2E | 2/4 pass | Partially Supported |
| H3: LOWO | <50% E2E | 0/3 pass | Refuted |

**Root cause:** Workload-spec mode introduces systematic TTFT errors (30,412% mean TTFT). Confirmed dead for 3rd time.

#### Idea 3: Hybrid Calibration — Best Round 4 Result

Two-stage approach: principled base model from E2E ground truth, then optional CMA-ES residual tuning.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: Principled base | <10% E2E | **5.7% E2E** (9/10 <10%) | Confirmed (E2E) |
| H2: CMA-ES residual | <15% E2E AND <25% ITL | 27.5% E2E, 43.7% ITL | Refuted |
| H3: LOMO | <80% E2E per fold | 30.7% mean (3/4 pass) | Partially Supported |
| H4: LOWO | <50% E2E per fold | 4/4 pass (5-8pp range) | Confirmed |

#### Round 4 Per-Experiment BLIS E2E Results (Idea 3 H1, Best)

| Experiment | E2E Error | TTFT Error | ITL Error | Pred (ms) | GT (ms) | Status |
|---|---|---|---|---|---|---|
| codellama-34b-codegen | **1.0%** | 53.6% | 101.2% | 3,760 | 3,723 | E2E solved |
| llama-2-70b-general | **1.6%** | 7.1% | 96.9% | 5,235 | 5,321 | E2E solved |
| mixtral-general | **1.7%** | 34.2% | 96.1% | 4,954 | 5,039 | E2E solved |
| codellama-34b-roleplay | **3.2%** | 52.9% | 105.7% | 3,787 | 3,670 | E2E solved |
| codellama-34b-general | **3.8%** | 37.6% | 91.9% | 3,939 | 4,093 | E2E solved |
| mixtral-codegen | **3.9%** | 56.2% | 106.9% | 4,857 | 4,675 | E2E solved |
| mixtral-roleplay | **5.0%** | 51.9% | 109.4% | 4,921 | 4,685 | E2E solved |
| llama-2-70b-hf-codegen | **6.3%** | 95.1% | 110.9% | 4,894 | 4,605 | E2E solved |
| llama-2-70b-roleplay | **7.7%** | 97.9% | 113.8% | 4,915 | 4,562 | E2E solved |
| llama-2-7b-roleplay | 22.9% | 55.8% | 145.7% | 2,546 | 2,071 | Unsolved |
| **MEAN** | **5.7%** | **54.2%** | **107.8%** | | | **9/10 solved** |

**Evaluation note:** The 5.7% is an in-sample result — coefficients derived from same E2E/TTFT data used for evaluation. Generalization validated via LOMO (3/4 pass) and LOWO (4/4 pass).

#### Round 4 Key Technique: Direct Calibration from E2E Ground Truth

The winning approach derives per-model coefficients analytically:
1. `target_step = (E2E_mean - TTFT_mean) / output_len_mean`
2. `beta0 = target_step - beta2 × avg_decode_batch` (additive intercept absorbs all non-GPU overhead)
3. `beta2 = mean(step.duration_us) / avg_decode_batch` (marginal per-token GPU cost)
4. `alpha0 = mean(TTFT)` from lifecycle data (per-model constant)

**Production coefficients (microseconds):**

| Model | alpha0 (TTFT) | beta0 (intercept) | beta1 (prefill) | beta2 (decode) |
|-------|--------------|-------------------|------------------|----------------|
| llama-2-7b | 27,129 | 9,741 | 0.30 | 13.6 |
| codellama-34b | 47,618 | 14,196 | 0.00 | 25.8 |
| llama-2-70b | 78,888 | 17,992 | 1.22 | 35.2 |
| llama-2-70b-hf | 78,888† | 17,590 | 0.00 | 29.8 |
| mixtral-8x7b-v0-1 | 62,767 | 18,921 | 0.69 | 8.8 |

†llama-2-70b-hf inherits llama-2-70b's alpha0 due to model name normalization.

**Go integration:** Zero new code — existing `BlackboxLatencyModel` with updated `defaults.yaml` coefficients. The model computes `step_time = beta0 + beta1*prefill + beta2*decode` (additive intercept, not max-floor).

#### Round 4 Critical Discovery: Additive Intercept >> GPU Compute

R4 beta0 values are ~2.1-2.5× R3's overhead estimates:

| Model | R4 beta0 (μs) | R3 overhead (μs) | Ratio |
|-------|---------------|-------------------|-------|
| llama-2-7b | 9,741 | 3,897 | 2.50× |
| codellama-34b | 14,196 | 6,700 | 2.12× |
| llama-2-70b | 17,992 | 8,000 | 2.25× |
| mixtral-8x7b | 18,921 | 9,125 | 2.07× |

GPU forward pass is only 2-8% of step time (beta2 × avg_batch / total_step). The additive intercept dominates.

#### Round 4 Critical Discovery: E2E ↔ ITL Tradeoff is Architectural

ITL remains ~100% error (107.8%) because BLIS reports ITL ≈ step_time/batch_size (300-600μs), while vLLM lifecycle ITL has median ~30-60μs. This is a measurement mismatch, not a calibration error. CMA-ES residual tuning (H2) confirmed: any E2E sacrifice produces only modest ITL improvement (5.7%→27.5% E2E for 107.8%→43.7% ITL).

#### Round 4 Binding Constraints

| ID | Constraint | Impact | Mitigatable? |
|----|-----------|--------|-------------|
| BC-4-1 | E2E ↔ ITL fundamental tradeoff | Cannot achieve <10% E2E AND <20% ITL | NO — architectural |
| BC-4-2 | 7B model outlier (22.9% E2E) | Only unsolved experiment | YES — batch-size-aware intercept |
| BC-4-3 | LOMO regression (30.7% vs R3's 14.8%) | Coefficients are model-specific | PARTIALLY — large models transfer |
| BC-4-4 | TTFT at 54.2% | Not independently accurate | YES — per-workload TTFT |
| BC-4-5 | In-sample evaluation | 5.7% is not predictive accuracy | YES — temporal holdout |
| BC-4-6 | No baseline control | Can't isolate coefficient vs pipeline improvement | YES — old coeffs in trace-replay |
| BC-4-7 | vLLM-args untested | Coefficients specific to one config | PARTIALLY — low recalibration cost |

#### Round 4 Generalization Results

| Fold (holdout) | LOMO Best-Donor E2E | Pass (<80%) |
|---------------|---------------------|-------------|
| codellama-34b | 28.5% | PASS |
| llama-2-70b | 6.7% | PASS |
| llama-2-7b | 82.9% | FAIL |
| mixtral-8x7b | 4.5% | PASS |
| **MEAN** | **30.7%** | **3/4 pass** |

LOWO: All 4 models pass <50%, per-model range 5-8pp (tested with H2 CMA-ES coefficients).
vLLM-args: Not explicitly tested. Structural dependence on batch dynamics; low recalibration cost.

### Round 5 Findings

Round 5 tested unified cross-model training — whether a SINGLE model formulation could replace R4's per-model calibrated coefficients across all 4 LLM architectures. Three ideas tested. Full details: `round5/FINDINGS_ROUND5.md`.

#### Idea 1: Analytical Overhead Model (Power Law Meta-Regression)

Predict ALL coefficients from model metadata: `beta0 = 5629 × (params_B)^0.285`, similar for alpha0, beta2.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: Full validation | <10% E2E | 10.0% (5/10 <10%) | Partially Supported |
| H2: LOMO | <80% E2E per fold | 16.6% mean (4/4 pass) | Confirmed |
| H3: LOWO | Workload-invariant | By design (metadata-only) | Confirmed |

#### Idea 2: Normalized Features (Hybrid) — Best Round 5 Result

Metadata-derived beta0/alpha0 + step-data-derived beta2 per model.

| Hypothesis | Target | Actual | Status |
|---|---|---|---|
| H1: Full validation | <10% E2E | **8.6%** (6/10 <10%) | Partially Supported |
| H2: LOMO | <80% E2E per fold | **14.3%** mean (4/4 pass) | Confirmed |
| H3: LOWO | <50% E2E per fold | By design + 5-8pp range | Confirmed |

#### Idea 3: Hierarchical Two-Stage — REFUTED (Catastrophic)

Single Ridge on cross-model step residuals. Intercept = -22,254, producing 4,792% mean E2E. Cross-model step-level training confirmed failed for the 4th time.

#### Round 5 Per-Experiment BLIS E2E Results (Idea 2, Best)

| Experiment | E2E Error | R4 Error | Delta |
|---|---|---|---|
| llama-2-7b-roleplay | 22.9% | 22.9% | +0.0pp |
| llama-2-70b-general | 0.2% | 1.6% | -1.4pp |
| llama-2-70b-hf-codegen | 10.0% | 6.3% | +3.7pp |
| llama-2-70b-roleplay | 11.7% | 7.7% | +4.0pp |
| mixtral-codegen | 5.4% | 3.9% | +1.6pp |
| mixtral-general | 10.4% | 1.7% | +8.7pp |
| mixtral-roleplay | 4.4% | 5.0% | -0.6pp |
| codellama-34b-general | 3.1% | 3.8% | -0.6pp |
| codellama-34b-codegen | 7.9% | 1.0% | +6.9pp |
| codellama-34b-roleplay | 10.2% | 3.2% | +7.0pp |
| **MEAN** | **8.6%** | **5.7%** | **+2.9pp** |

#### Round 5 Critical Discoveries

1. **Metadata-derived overhead is viable**: `beta0 = 5629 × (params_B)^0.285` captures 92-98% of step time from model metadata alone with <5% coefficient error.
2. **LOMO breakthrough via functional form**: Power law interpolation produces 2.1× better LOMO than R4's donor coefficient transfer (14.3% vs 30.7%). The 7B fold improved from 82.9% (FAIL) → 24.0% (PASS).
3. **Cross-model step-level training is dead**: Idea 3 confirmed (4th time) that pooling step data across models with 3+ OOM scale variation fails catastrophically. Only coefficient-space transfer works.
4. **Accuracy–generalization tradeoff quantified**: 2.9pp E2E sacrifice buys 2.1× LOMO improvement. This is the production-optimal tradeoff.
5. **Workload-spec format matters**: `inference_perf` format produces ~10× errors vs `clients` format with constant distributions.

#### Round 5 Generalization Results

| Metric | R4 | R5 Idea 1 | R5 Idea 2 |
|---|---|---|---|
| LOMO mean | 30.7% (3/4 pass) | 16.6% (4/4 pass) | **14.3%** (4/4 pass) |
| LOMO 7B fold | 82.9% (FAIL) | 25.5% (PASS) | **24.0%** (PASS) |
| LOWO | 5-8pp (4/4 pass) | By design | By design |
| vLLM args | Not tested | Not tested | Not tested |

#### Round 5 Binding Constraints

| ID | Constraint | Impact | Mitigatable? |
|----|-----------|--------|-------------|
| BC-5-1 | 7B outlier persists at 22.9% | Same as R4, structural | NO |
| BC-5-2 | 4/10 experiments >10% | Aggregate 8.6% but individual failures | PARTIALLY |
| BC-5-3 | Power law from 4 points | Underdetermined | PARTIALLY — more models needed |
| BC-5-4 | Metadata beta0 vs R4 direct beta0 | 3-10pp accuracy sacrifice | TRADEOFF |

## Cumulative Round History

| Round | Ideas | Best E2E | Best ITL | Key Achievement | Primary Blocker |
|---|---|---|---|---|---|
| R1 | Tree ensemble, analytical decomposition, evolutionary (deferred) | Not tested | Not tested | 34% per-step MAPE (2x blackbox) | No BLIS integration |
| R2 | Bayesian calibration, regime ensemble | 427.8% mean (1/10 <10%) | 33.6% mean (5/10 <10%) | Overhead floor + BLIS pipeline + regime structure | TTFT mismatch (31,906%) |
| R3 | Trace replay, total-context model, CMA-ES calibration | **15.1%** mean (4/10 <10%) | 87.4% mean (0/10 <10%) | Trace replay + CMA-ES (28.3x E2E improvement) | E2E ↔ ITL tradeoff |
| R4 | Constrained CMA-ES, cycle-time regression, hybrid calibration | **5.7%** mean (9/10 <10%) | 107.8% mean | Direct calibration from E2E GT (2.6× improvement) | ITL architectural mismatch, 7B outlier |
| **R5** | **Analytical overhead, normalized features, hierarchical two-stage** | **8.6%** mean (6/10 <10%) | ~100% mean | **Metadata power law for LOMO (2.1× improvement)** | **Accuracy–generalization tradeoff** |
