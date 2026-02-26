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

### Broadened Scope for Round 2

**The goal is to achieve <10% workload-level E2E mean error by improving ANY or ALL of the 5 LatencyModel methods**, not just StepTime. Opportunities include:

1. **StepTime improvement** — continue from Round 1's XGBoost (34% MAPE), potentially with per-request KV features derived from lifecycle data
2. **QueueingTime calibration** — the `alpha0 + alpha1*inputLen` model may systematically over/underpredict arrival-to-queue delay, biasing all downstream latencies
3. **SchedulingProcessingTime** — currently 0, but real scheduling overhead (vLLM's `scheduler.schedule()`) is measurable from ground-truth timing data and scales with queue depth and batch size
4. **OutputTokenProcessingTime** — currently constant, but may vary with model size, TP degree, or output token position
5. **PreemptionProcessingTime** — currently 0, but preemption events are observable in ground-truth traces and have real cost
6. **End-to-end calibration** — even if individual components have errors, calibrating the *composition* of all components to minimize E2E error is valid (e.g., intentional over/underestimation of one component to compensate for systematic bias in another)

**The winning approach must integrate back into BLIS as a Go `LatencyModel` implementation.** The simulator's existing roofline model (analytical FLOPs/bandwidth with MFU lookup) is unaffected and not being replaced.

## Ground-Truth Data

### Overview

122,752 step-level observations from instrumented vLLM v0.15.1 with OpenTelemetry tracing (10% sample rate). 16 experiments: 4 models × 4 workloads, all on H100 80GB GPUs.

### Models and Configurations

| Model | Architecture | TP | Parameters |
|-------|-------------|-----|-----------|
| Llama-2-7B | Dense | 1 | 7B |
| Llama-2-70B | Dense | 4 | 70B |
| CodeLlama-34B | Dense | 2 | 34B |
| Mixtral-8x7B-v0.1 | MoE (8 experts, top-2) | 2 | 46.7B total, ~12.9B active |

### Workloads

general, codegen, roleplay, reasoning — each with different input/output length distributions. All runs: `max_model_len=4096`, `max_num_batched_tokens=2048`, `max_num_seqs=128`, chunked prefill enabled, prefix caching enabled.

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
- Largest: ~250,000 μs (large reasoning prefills)
- Reasoning workloads: mean 6,000–33,500 μs (long contexts)
- Roleplay/codegen: mean 160–320 μs (short, bursty)

### Known Feature Gaps

1. **No per-request KV cache lengths**: Only aggregate batch tokens available. Attention FLOPs scale with per-request kv_len (H8 showed 12.96× overestimate without per-request KV). The simulator's `ProgressIndex` (input_tokens_processed + output_tokens_generated) is available as a proxy at inference time.
2. **No MoE-specific features**: No active expert count, expert load balance, or tokens-per-expert.
3. **No prefix cache hit information**: `prefill_tokens` may reflect pre- or post-cache-hit count.

### Additional Data Sources

- **Per-request lifecycle data**: Per-token timestamps, input/output token counts — enables per-request KV length derivation and E2E validation
- **MFU benchmarks** (`bench_data/`): Kernel-level GEMM and attention MFU data by GPU — useful for physics-informed features

## Baseline Results (WP0)

### Global Blackbox Baseline (single regression across all experiments)
- **MAPE: 670%** — catastrophically bad because a single linear model can't handle 3-order-of-magnitude step time range across models
- Pearson r: 0.41
- Naive mean baseline: 861% MAPE

### Per-Model+Workload Blackbox Baseline (16 separate regressions)

| Model × Workload | MAPE | Assessment |
|---|---|---|
| Mixtral-general | 9.2% | Excellent — nearly at target |
| Llama-70B-reasoning | 14.2% | Good |
| Mixtral-codegen | 19.0% | Fair |
| CodeLlama-codegen | 21.6% | Fair |
| CodeLlama-roleplay | 30.8% | Poor |
| Mixtral-roleplay | 33.6% | Poor |
| CodeLlama-reasoning | 37.3% | Poor |
| Llama-7B-roleplay | 40.3% | Poor |
| Llama-70B-general | 61.2% | Bad |
| Llama-7B-codegen | 69.7% | Bad |
| Llama-7B-general | 72.9% | Bad |
| Llama-70B-codegen | 90.8% | Bad |
| Llama-7B-reasoning | 123.5% | Terrible |
| Llama-70B-roleplay | 128.6% | Terrible |
| Mixtral-reasoning | 222.8% | Terrible |
| CodeLlama-general | 151.1% | Terrible |

**Key insight**: The 3-coefficient model fails hardest on:
1. Reasoning workloads (high variance, long contexts)
2. Experiments where step time is dominated by attention (KV-length-dependent)
3. Cross-model prediction (different parameter counts = different compute)

A unified model that handles all 16 experiments with <10% workload-level E2E mean error would be a major improvement.

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
| P5 | Hardware generalization (H100 → A100) | Informational |
| P6 | Quantization transferability | Informational |

### Data Split Strategy

- **Primary**: Temporal split (60/20/20) within each experiment — prevents autocorrelation leakage
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
4. **Evolved code translation** — for evolutionary approaches producing interpretable code

Each idea must specify which integration path it would use.

## Algorithm Scope

**Not restricted to ML or to StepTime.** Research ideas may propose:
- Statistical regression (Ridge, Lasso, polynomial, piecewise)
- Tree ensembles (XGBoost, LightGBM, random forest)
- Neural networks (small MLPs, attention-based)
- Physics-informed models (analytical compute model + learned residuals)
- Evolutionary program synthesis (OpenEvolve, GEPA — evolved prediction functions)
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

## Key Questions for Ideas to Address

### From Round 1 (still open)
1. How to handle the 3-order-of-magnitude step time range across models?
2. How to capture KV-cache-length effects using only ProgressIndex as proxy?
3. How to handle dense vs MoE architecture differences (25% active parameters for MoE)?
4. How to handle the non-additive prefill/decode interaction in mixed batches (chunked prefill)?
5. What features beyond the existing schema would improve predictions, and are they derivable from Request objects?

### New for Round 2 (broadened scope)
6. **Which LatencyModel component contributes most to E2E error?** Is StepTime the dominant error source, or do QueueingTime/SchedulingTime/OutputTokenProcessingTime introduce comparable systematic bias?
7. **Can we derive per-request KV features from lifecycle data?** The `request_metrics/` directories contain per-token timestamps — can these reconstruct ProgressIndex at each step?
8. **Does per-step MAPE translate to E2E error?** Round 1 showed 34% per-step MAPE. If errors are symmetric, E2E mean error could be much lower. This must be measured via BLIS simulation.
9. **Can non-zero SchedulingProcessingTime and PreemptionProcessingTime improve E2E fidelity?** These are currently hardcoded to 0. Ground-truth traces contain timing data for these events.
10. **Is end-to-end calibration more practical than per-component accuracy?** Instead of optimizing each component independently, calibrate the composition to minimize E2E error directly (e.g., grid search over alpha/beta coefficients with E2E error as the objective).
11. **Can simpler models with better features match XGBoost?** Round 1's XGBoost underfits (shallow trees) — better features (per-request KV) might enable Ridge or piecewise linear models that are trivial to integrate in Go.
