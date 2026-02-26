# Step-Time Prediction for LLM Inference Simulation

## Problem

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It predicts per-step execution time for transformer inference batches. The current blackbox latency model uses a 3-coefficient linear regression:

```
step_time = beta0 + beta1 * prefill_tokens + beta2 * decode_tokens
```

This model has fundamental structural limitations:
- **Only 2 features**: It reduces the entire batch to two scalar sums (total prefill tokens, total decode tokens)
- **No KV cache awareness**: A batch with one long-context request (KV length 4096) and one short-context request (KV length 128) produces identical predictions to two medium-context requests (KV length 2112 each), despite dramatically different attention FLOPs
- **No architecture awareness**: The same formula is used for dense (Llama) and MoE (Mixtral) models, despite fundamentally different compute patterns (Mixtral activates only ~25% of parameters per token via top-2 expert routing)
- **No batch composition detail**: Number of prefill vs decode requests, mixed-batch interactions, and request count are all ignored

**The goal is to replace this blackbox model with a data-driven alternative achieving <10% workload-level E2E mean error across all 16 experiments, while the winning model must integrate back into BLIS as a Go LatencyModel implementation.**

The simulator's existing roofline model (analytical FLOPs/bandwidth with MFU lookup) is unaffected and not being replaced.

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

**Not restricted to ML.** Research ideas may propose:
- Statistical regression (Ridge, Lasso, polynomial, piecewise)
- Tree ensembles (XGBoost, LightGBM, random forest)
- Neural networks (small MLPs, attention-based)
- Physics-informed models (analytical compute model + learned residuals)
- Evolutionary program synthesis (OpenEvolve, GEPA — evolved prediction functions)
- Hybrid approaches (analytical backbone + ML residual correction)

Each approach must:
1. Cite relevant prior work from systems/ML literature
2. Address all evaluation dimensions (P1–P6)
3. Specify which LatencyModel methods it covers (minimum: StepTime)
4. Document its Go integration path
5. Be distinct from other proposed approaches

## Key Questions for Ideas to Address

1. How to handle the 3-order-of-magnitude step time range across models?
2. How to capture KV-cache-length effects using only ProgressIndex as proxy?
3. How to handle dense vs MoE architecture differences (25% active parameters for MoE)?
4. How to achieve good per-step accuracy that also translates to <10% E2E mean error?
5. How to handle the non-additive prefill/decode interaction in mixed batches (chunked prefill)?
6. What features beyond the existing schema would improve predictions, and are they derivable from Request objects?
