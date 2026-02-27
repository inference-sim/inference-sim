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
| P2 | TTFT mean fidelity | < 15% per experiment |
| P2 | ITL mean fidelity | < 15% per experiment |
| P3 | Tail behavior (p99) | No ranking inversions vs baseline |
| P4 | Generalization (workloads, models, hardware) | Cross-validation |
| P4 | Ease of use, retraining story, reproducibility | Qualitative |
| P5 | Hardware generalization (H100 → A100) | Informational |
| P6 | Quantization transferability | Informational |

### Data Split Strategy

- **Primary**: Temporal split (60/20/20) within each experiment — prevents autocorrelation leakage
- **Generalization**: Leave-one-model-out (4-fold) + leave-one-workload-out (3-fold)
- **Short-circuit**: Ideas with h1 per-step MAPE > 30% are dropped (threshold calibrated to blackbox_MAPE + 10% if blackbox exceeds 25%)

### Idea Review Criteria

Beyond quantitative accuracy, idea reviewers evaluate each approach against the following dimensions. The standard is that **the approach must be defensible to a community of reviewers at a top-tier systems/ML conference or journal** (e.g., OSDI, SOSP, NSDI, MLSys, EuroSys).

| Dimension | What reviewers check |
|---|---|
| **Accuracy** | Does the approach meet the P1 target (<10% E2E mean error)? Are per-step and E2E metrics both reported? |
| **Generalization across workloads** | Does the model hold on unseen workload distributions (leave-one-workload-out)? Does it degrade gracefully on out-of-distribution input/output length profiles? |
| **Generalization across LLMs** | Does the model transfer to unseen model architectures and scales (leave-one-model-out)? Does it handle dense vs MoE, small vs large parameter counts? |
| **Generalization across vLLM config** | How sensitive is the model to vLLM serving parameters (`max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, chunked prefill on/off, prefix caching on/off)? Would a different config require full retraining? |
| **Generalization across hardware** | Does the approach transfer across GPU generations (H100 → A100 → B200)? What changes when memory bandwidth, compute FLOPs, or GPU memory capacity differ? |
| **Ease of use** | How much expertise is required to deploy the model for a new model/GPU/config combination? Is the workflow self-service or does it require ML expertise? |
| **Retraining story** | Is retraining or fine-tuning required? Under what circumstances (new model, new GPU, new vLLM version, new workload)? How much data and compute does retraining require? How easy is the retraining process end-to-end? |
| **vLLM version sensitivity** | What happens when the serving engine (vLLM) changes versions? Do internal scheduling changes (e.g., new chunked prefill heuristics, changed batch formation logic) invalidate the model's assumptions? How is this detected and corrected? |
| **Overheads** | What are the training-time costs (data collection, compute, human effort)? What are the inference-time costs (latency per prediction, memory footprint, complexity of the Go integration)? |
| **Reproducibility** | Can results be reproduced from the provided code, data, and instructions? Are random seeds fixed? Are all hyperparameters documented? Is the training pipeline deterministic? |

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
6. Include a **model-generalization hypothesis** — a leave-one-model-out experiment (train on 3 models, predict on the held-out 4th) that evaluates whether the approach transfers to unseen model architectures and scales
7. Include a **workload-generalization hypothesis** — a leave-one-workload-out experiment (train on 3 workloads, predict on the held-out 4th) that evaluates whether the approach transfers to unseen workload distributions

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

## Binding Constraints for Round 3

### BC-1: TTFT/Simulation Fidelity Mismatch (PRIMARY — Blocking E2E Target)

TTFT errors of 31,906% mean dominate E2E error. This is NOT a step-time problem — it indicates systematic mismatch between BLIS's request lifecycle simulation and the real vLLM system. **No step-time improvement can achieve <10% E2E until this is fixed.**

Likely causes: workload spec generation producing mismatched arrival patterns, BLIS prefill/queueing behavior diverging from real vLLM scheduling, request injection timing not matching ground-truth arrival distributions.

### BC-2: KV Feature Scaling (SECONDARY — Blocking Per-Step Improvement)

Per-request KV features derived from ProgressIndex have signal but are counter-productive in raw linear space. Feature scaling (StandardScaler), log-transform of features (not target), or nonlinear models needed to unlock their value.

### BC-3: CodeLlama-34B Anomaly (SECONDARY)

34B is the worst model across all metrics by a wide margin. May be data quality, architecture-specific dynamics, or coefficient instability. Needs targeted investigation.

### BC-4: Mixed-Heavy Regime Sparsity (TERTIARY)

The 3-regime design collapses to 2 effective regimes because only 0.1% of steps have prefill>=256. Either lower the threshold, add reasoning workloads, or accept a 2-regime model.

## Cumulative Round History

| Round | Ideas | Best E2E | Best ITL | Key Achievement | Primary Blocker |
|---|---|---|---|---|---|
| R1 | Tree ensemble, analytical decomposition, evolutionary (deferred) | Not tested | Not tested | 34% per-step MAPE (2x blackbox) | No BLIS integration |
| R2 | Bayesian calibration, regime ensemble | 427.8% mean (1/10 <10%) | 33.6% mean (5/10 <10%) | Overhead floor + BLIS pipeline + regime structure | TTFT mismatch (31,906%) |
