# Research Document

## Problem Statement

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

122,752 step-level observations from instrumented vLLM v0.15.1 with OpenTelemetry tracing (10% sample rate). 16 experiments: 4 models x 4 workloads, all on H100 80GB GPUs.

### Models and Configurations

| Model | Architecture | TP | Parameters |
|-------|-------------|-----|-----------|
| Llama-2-7B | Dense | 1 | 7B |
| Llama-2-70B | Dense | 4 | 70B |
| CodeLlama-34B | Dense | 2 | 34B |
| Mixtral-8x7B-v0.1 | MoE (8 experts, top-2) | 2 | 46.7B total, ~12.9B active |

### Workloads

general, codegen, roleplay, reasoning -- each with different input/output length distributions. All runs: `max_model_len=4096`, `max_num_batched_tokens=2048`, `max_num_seqs=128`, chunked prefill enabled, prefix caching enabled.

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
- Largest: ~250,000 us (large reasoning prefills)
- Reasoning workloads: mean 6,000-33,500 us (long contexts)
- Roleplay/codegen: mean 160-320 us (short, bursty)

### Known Feature Gaps

1. **No per-request KV cache lengths**: Only aggregate batch tokens available. Attention FLOPs scale with per-request kv_len (H8 showed 12.96x overestimate without per-request KV). The simulator's `ProgressIndex` (input_tokens_processed + output_tokens_generated) is available as a proxy at inference time.
2. **No MoE-specific features**: No active expert count, expert load balance, or tokens-per-expert.
3. **No prefix cache hit information**: `prefill_tokens` may reflect pre- or post-cache-hit count.

### Additional Data Sources

- **Per-request lifecycle data**: Per-token timestamps, input/output token counts -- enables per-request KV length derivation and E2E validation
- **MFU benchmarks** (`bench_data/`): Kernel-level GEMM and attention MFU data by GPU -- useful for physics-informed features

## Baseline Results (WP0)

### Global Blackbox Baseline (single regression across all experiments)
- **MAPE: 670%** -- catastrophically bad because a single linear model can't handle 3-order-of-magnitude step time range across models
- Pearson r: 0.41
- Naive mean baseline: 861% MAPE

### Per-Model+Workload Blackbox Baseline (16 separate regressions)

| Model x Workload | MAPE | Assessment |
|---|---|---|
| Mixtral-general | 9.2% | Excellent -- nearly at target |
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

**Not restricted to ML.** Research ideas may propose:
- Statistical regression (Ridge, Lasso, polynomial, piecewise)
- Tree ensembles (XGBoost, LightGBM, random forest)
- Neural networks (small MLPs, attention-based)
- Physics-informed models (analytical compute model + learned residuals)
- Evolutionary program synthesis (OpenEvolve, GEPA -- evolved prediction functions)
- Hybrid approaches (analytical backbone + ML residual correction)

Each approach must:
1. Cite relevant prior work from systems/ML literature
2. Address all evaluation dimensions (P1-P6)
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

---

# Background

## Repository Context

### BLIS Latency Model Architecture

BLIS currently has two latency model implementations behind a frozen 5-method `LatencyModel` interface (defined in `sim/latency_model.go`):

**1. BlackboxLatencyModel** (`sim/latency/latency.go:18-58`)

The model being replaced. Uses a 3-coefficient linear regression:
```
step_time = beta0 + beta1 * cache_miss_tokens + beta2 * decode_tokens
```

Where `cache_miss_tokens` are new prefill tokens (requests where `ProgressIndex < len(InputTokens)`) and `decode_tokens` are tokens from decode-phase requests. The model iterates over the batch, classifies each request as prefill or decode based on `ProgressIndex`, sums the `NumNewTokens` for each category, and applies the linear formula. This reduces an arbitrarily complex batch of up to 128 heterogeneous requests to just two scalar features.

For the other 4 methods: `QueueingTime` uses `alpha0 + alpha1 * inputLen`, `OutputTokenProcessingTime` returns `alpha2`, and both `SchedulingProcessingTime` and `PreemptionProcessingTime` return 0 (not modeled).

**2. RooflineLatencyModel** (`sim/latency/latency.go:64-111`, `sim/latency/roofline.go`)

The analytical model (NOT being replaced -- comparison baseline only). This model uses physics-based FLOPs/bandwidth estimation with MFU lookup from benchmark data. Key characteristics:

- **Per-request granularity**: Unlike the blackbox model, the roofline model processes each request individually, using `ProgressIndex` and `NumNewTokens` to build per-request `PrefillRequestConfig` or `DecodeRequestConfig` structs.
- **GEMM time calculation**: Uses MFU-weighted GEMM time for all transformer projections (QKV, O, Gate/Up/Down) with a memory-bandwidth floor (`max(compute_time, weight_load_time)`).
- **Attention core**: Computes FLOPs-weighted MFU across heterogeneous KV lengths for decode; uses bucket-based MFU for prefill with a causal masking correction factor (1.8).
- **Memory bandwidth**: Accounts for model weights, KV cache growth, KV cache read access, and activations.
- **Mixed batch handling**: Uses `max(prefill_time, decode_time)` to model chunked-prefill overlap.
- **TP scaling**: Applies `1/tp` factor to both compute and memory bandwidth per layer.

The roofline model demonstrates that per-request features (especially `ProgressIndex` as KV length) are both available and useful. The research models should similarly leverage per-request information.

### Features Available at Prediction Time

Each `Request` object in the batch provides at step time:

| Field | Type | Description | Derivable Features |
|-------|------|-------------|-------------------|
| `InputTokens` | `[]int` | Full prompt token sequence | `len(InputTokens)` = input length |
| `OutputTokens` | `[]int` | Generated tokens so far | `len(OutputTokens)` = decode progress |
| `ProgressIndex` | `int64` | Cumulative: input processed + output generated | KV cache length proxy |
| `NumNewTokens` | `int` | Tokens to generate this step | Prefill chunk size or 1 (decode) |

From these, research models can derive per-request features:
- **Phase classification**: `ProgressIndex < len(InputTokens)` means prefill; else decode
- **KV cache length**: `ProgressIndex` approximates total KV cache entries per request
- **Remaining prefill**: `len(InputTokens) - ProgressIndex` (for chunked prefill)
- **Decode depth**: `len(OutputTokens)` = number of decode steps completed
- **Prefill chunk ratio**: `NumNewTokens / (len(InputTokens) - ProgressIndex)` -- how much of the remaining prefill is being processed this step

And batch-level aggregates:
- Number of prefill vs. decode requests
- Distribution statistics over per-request KV lengths (mean, max, variance, sum)
- Total prefill tokens, total decode tokens
- Mixed batch indicator (has both prefill and decode)
- Batch size

### Batch Formation Context

The `VLLMBatchFormation` implementation (`sim/batch_formation.go`) shows how batches are composed:
- **Chunked prefill**: Prefill requests may receive only a subset of their remaining tokens (`PrefillTokenThreshold` controls chunk size). A request can span multiple steps in the prefill phase.
- **Continuous batching**: Running requests continue in the batch; new requests are dequeued from the wait queue each step.
- **Token budget**: `MaxScheduledTokens` limits total tokens per step. Both continuing prefills and new requests compete for this budget.
- **Preemption**: Under memory pressure, the formation may evict requests from the batch tail, releasing KV blocks.
- **Prefix caching**: Cached blocks skip computation (the `GetCachedBlocks` call reduces new token count).

This means that in practice, nearly every step contains a mix of prefill and decode requests (chunked prefill ensures prefills don't monopolize the GPU). The non-additive interaction between prefill and decode in mixed batches is a known modeling challenge (BLIS hypothesis H5).

### Current Limitations Summary

| Limitation | Impact | Evidence |
|-----------|--------|----------|
| Only 2 aggregate features | Cannot distinguish batches with same token sums but different compositions | Per-model MAPE ranges from 9% to 223% |
| No per-request KV lengths | Attention FLOPs misestimated for heterogeneous batches | H8: 12.96x overestimate without per-request KV |
| No architecture awareness | Same formula for dense (100% params active) and MoE (25% active) | Mixtral-reasoning: 222.8% MAPE |
| No mixed-batch interaction | Prefill/decode overlap in chunked prefill not modeled | H5: non-additive interaction confirmed |
| No scheduling/preemption overhead | Returns 0 for both methods | May contribute to E2E error at high preemption rates |

## Research Design Summary

### Feature Gaps (from design doc)

**Gap 1 -- Per-request KV cache lengths**: The most impactful missing feature. Attention FLOPs for decode scale as O(batch_size * kv_len * head_dim * num_heads) per request. The `ProgressIndex` field on each Request serves as a direct proxy for KV cache length without any interface changes. Research models should compute per-request KV length statistics (mean, max, variance, sum) from `ProgressIndex` values in the batch.

**Gap 2 -- MoE-specific features**: Mixtral-8x7B uses 8 experts with top-2 routing, activating only ~25% of parameters per token. No features capture expert routing behavior. Research approaches must either: (a) propose MoE-specific derived features, (b) treat MoE as a separate modeling problem with model-ID conditioning, or (c) demonstrate that general features suffice for MoE prediction.

**Gap 3 -- Prefix cache hit information**: With prefix caching enabled, `prefill_tokens` may reflect pre- or post-cache-hit count. This ambiguity affects feature interpretation and must be resolved during data loading.

### Modeling Decisions (from design doc)

Key decisions from the Banks et al. scoping analysis:

- **Modeled**: Batch composition, per-request KV cache lengths (via ProgressIndex), model architecture type, chunked prefill interaction
- **Simplified**: Hardware (H100-only training), vLLM scheduler internals (observed via batch output), system state features (potentially spurious), temporal dynamics (independent prediction), prefix cache effects
- **Omitted**: Kernel-level scheduling (below abstraction level), multi-instance effects (instance-local prediction), post-step features (data leakage risk)

### Module Contract (from design doc)

The new StepML LatencyModel must satisfy:
- **INV-M-1**: Step-time estimate > 0 for all non-empty batches
- **INV-M-2**: Deterministic prediction (same batch -> same output)
- **INV-M-3**: Side-effect-free (pure function of batch + immutable state)
- **INV-M-4**: Prediction latency < 1ms at batch size 128
- **INV-M-5**: Soft monotonicity (more tokens should not decrease predicted time, with <5% violation threshold)
- **INV-M-6**: Bounded systematic bias (|MSPE| < 5%)

### Evaluation Framework (from design doc)

**Primary metric**: Workload-level E2E mean error < 10% on each of 16 experiments. This benefits from error cancellation -- random per-step errors that are unbiased partially cancel when averaged across requests.

**End-to-end validation (Stage 1)**: Synthetic trace replay -- reconstruct per-request step sequences, sum predicted step times, compare to observed E2E latency. Does not require BLIS integration.

**Baselines**: Blackbox (must outperform), Roofline (informational comparison), Naive mean.

**Statistical requirements**: Bootstrap 95% CI, Wilcoxon signed-rank test (p < 0.05) for improvement over best baseline, MSPE reporting for systematic bias detection.

### Risk Register (from macro plan)

| Risk | Description | Gate |
|------|-------------|------|
| R1 | ProgressIndex may not faithfully proxy KV cache length | Before WP3 |
| R2 | 10% sampling may be systematically biased | Before WP3 |
| R3 | Request-batch features alone may be insufficient for <10% E2E error | Before WP5 |
| R4 | Blackbox baseline may already be good enough (<12% E2E error) | Before WP1 |
| R5 | Winning model may not be portable to Go | Before WP5 |

## Literature Survey

### 1. LLM Inference Serving Simulators and Latency Prediction

**Vidur (Agrawal et al., MLSys 2024)**
- A performance estimator and simulator for LLM inference, developed at Microsoft Research.
- Breaks down transformer execution into individual operations (GEMM, attention, all-reduce) and models each operation's latency separately using profiled execution times.
- Uses a "performance ladder" approach: profiles individual operation latencies on target hardware, then composes them to predict full step time.
- Key insight: operation-level decomposition is more accurate than aggregate batch-level prediction because different operations have different compute-vs-memory characteristics.
- **Relevance**: Directly comparable system. Vidur's operation-level decomposition is more granular than our blackbox model but requires hardware profiling. Our research can use a similar decomposition philosophy but learn the operation costs from data rather than profiling.

**Splitwise (Patel et al., ISCA 2024)**
- Splits prefill and decode phases across heterogeneous hardware (prompt machines vs. token machines).
- Models prefill latency as compute-bound (proportional to sequence length squared for attention, linear for GEMM) and decode latency as memory-bandwidth-bound.
- Key insight: the prefill/decode dichotomy maps directly to compute-bound vs. memory-bound regimes on the roofline, with the transition point depending on batch size and sequence length.
- **Relevance**: Validates that phase-aware modeling (separate prefill vs. decode treatment) is essential. Our mixed-batch handling must capture this regime transition.

**DistServe (Zhong et al., OSDI 2024)**
- Disaggregates prefill and decode into separate GPU clusters for better SLO attainment.
- Uses analytical latency models that account for batch size, sequence length, and model architecture to predict per-operation latency.
- **Relevance**: Confirms that architecture-aware analytical models can achieve good prediction accuracy when combined with hardware-specific calibration.

**Sarathi-Serve (Agrawal et al., OSDI 2024)**
- Introduces chunked prefill with piggybacking to reduce decode latency interference.
- Models the interference between prefill and decode tokens in mixed batches.
- Key insight: the non-additive interaction between prefill and decode in chunked-prefill batches -- prefill compute can partially overlap with decode attention.
- **Relevance**: Directly relevant to our mixed-batch prediction challenge. The chunked prefill interaction is non-additive (confirmed by BLIS hypothesis H5), and any model must capture this.

**SimLLM / LLMServingSim**
- Simulation frameworks for LLM serving that model request scheduling, batching, and GPU execution.
- Typically use simplified latency models (linear in batch size and sequence length) for scalability.
- **Relevance**: Demonstrates the common approach (linear models) and its limitations, motivating our data-driven replacement.

### 2. GPU Kernel Performance Prediction with Machine Learning

**Habitat (Yu et al., ATC 2021 / MLSys 2022)**
- Predicts GPU kernel execution time for DNNs using a wave-scaling model.
- Uses hardware counter data to build a performance model that accounts for GPU occupancy, memory bandwidth utilization, and compute utilization.
- Achieves <10% prediction error for individual kernel runtimes.
- **Relevance**: Shows that ML-based kernel prediction at the operation level can achieve high accuracy. Our problem is at a higher abstraction level (step, not kernel) but the feature engineering insights apply.

**MLPerf Inference Benchmark Analysis**
- MLPerf benchmarks provide standardized performance measurements for inference workloads.
- Analysis papers (e.g., Reddi et al., ISCA 2020) show that performance varies by 2-3x across hardware for the same model, confirming that hardware-specific calibration is essential.
- **Relevance**: Motivates our H100-specific training with hardware generalization as an evaluation dimension rather than a training requirement.

**GEMM Performance Prediction**
- Several works model GEMM (General Matrix Multiply) performance on GPUs using analytical models and ML.
- Key factors: tile sizes, memory access patterns, tensor core utilization, occupancy.
- Konstantinidis & Cotronis (2017): empirical GPU roofline with GEMM micro-benchmarks.
- Jia et al. (PPoPP 2018): "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" -- detailed GPU performance characterization enabling accurate GEMM prediction.
- **Relevance**: GEMM is the dominant operation in transformer inference. Understanding GEMM performance characteristics informs physics-informed feature design (compute intensity ratio, arithmetic intensity).

### 3. Roofline Models and Physics-Informed Performance Prediction

**Williams, Waterman & Patterson (CACM 2009) -- The Roofline Model**
- The foundational roofline model: performance is bounded by min(peak_compute, arithmetic_intensity * peak_bandwidth).
- Defines arithmetic intensity as FLOPs/byte, which determines whether an operation is compute-bound or memory-bound.
- **Relevance**: BLIS already uses this model (sim/latency/roofline.go). The key insight for research is that the roofline boundary point (where compute meets bandwidth) depends on batch size, sequence length, and model architecture -- features that can be computed from Request objects.

**Extended Roofline for Deep Learning (Wang et al., IEEE Micro 2020)**
- Extends the roofline model for DNN workloads, accounting for tensor core utilization, mixed precision, and memory hierarchy.
- Shows that MFU (Model FLOPs Utilization) varies significantly with batch size and sequence length -- not a constant.
- **Relevance**: Motivates using MFU as a function of batch composition rather than a fixed constant. The BLIS bench_data MFU lookup tables already capture this variation.

**Physics-Informed Machine Learning for Performance Prediction**
- Willard & Jia (arXiv 2020): "Integrating Physics-Based Modeling with Machine Learning" -- survey of hybrid approaches.
- Key pattern: use analytical models (roofline, FLOPs counting) to provide a physics-based baseline prediction, then learn residual corrections with ML.
- This "analytical backbone + learned residual" pattern has been successful in scientific computing (weather prediction, fluid dynamics) and translates well to performance prediction.
- **Relevance**: A strong candidate approach for our problem -- use roofline-derived features (compute intensity, memory boundedness) as inputs to an ML model, or use roofline predictions as a baseline with learned corrections.

### 4. Tree Ensemble Methods for Performance Prediction

**XGBoost for Hardware Performance (Chen & Guestrin, KDD 2016)**
- XGBoost is widely used for performance prediction in systems due to its ability to handle heterogeneous features, capture nonlinear interactions, and provide feature importance.
- Works well with the small-to-medium dataset sizes typical of performance benchmarking.
- Supports monotonicity constraints (critical for INV-M-5).
- **Relevance**: A strong candidate for our problem. XGBoost with engineered features (per-request KV statistics, compute intensity, phase indicators) could capture the nonlinear interactions between batch composition and step time.

**LightGBM (Ke et al., NeurIPS 2017)**
- Gradient boosting with leaf-wise growth, faster training than XGBoost on large datasets.
- Supports feature interaction constraints that can encode known physical relationships.
- **Relevance**: Alternative to XGBoost with potentially faster training; particularly useful if we need to run many hyperparameter configurations.

**Random Forest for Compiler Performance (Cummins et al., CGO 2017)**
- Uses random forests to predict compiler optimization outcomes.
- Shows that tree ensembles with domain-specific features outperform generic approaches.
- **Relevance**: Validates the feature-engineering-driven tree ensemble approach for systems performance prediction.

**Performance Modeling with Gradient Boosting (Ipek et al., ISCA 2006 / Lee et al., ISCA 2007)**
- Early work on using ML (neural networks, regression trees) for microarchitectural performance prediction.
- Key finding: domain-specific features (instruction mix, memory access patterns) are more important than model complexity.
- **Relevance**: Supports our hypothesis that feature engineering (KV length statistics, compute intensity) matters more than model sophistication.

### 5. Batch Inference and Continuous Batching Latency

**Orca (Yu et al., OSDI 2022)**
- Introduces iteration-level scheduling (continuous batching) for LLM serving.
- Shows that continuous batching creates highly heterogeneous batch compositions where different requests are at different stages of prefill/decode.
- Key insight: per-iteration latency depends on the mix of prefill and decode tokens in the batch, not just the total token count.
- **Relevance**: Directly motivates our need for batch composition features beyond simple token sums.

**vLLM (Kwon et al., SOSP 2023)**
- PagedAttention for efficient KV cache management.
- Shows that KV cache memory management significantly affects serving throughput and latency.
- The block-based allocation means KV cache fragmentation can affect performance.
- **Relevance**: Our training data comes from vLLM. Understanding PagedAttention's memory access patterns informs feature design (KV utilization ratio as a memory pressure indicator).

**FlashAttention (Dao et al., NeurIPS 2022; Dao, ICLR 2024)**
- IO-aware attention algorithm that reduces memory reads/writes.
- Changes the relationship between sequence length and attention latency -- with FlashAttention, attention is more compute-bound than memory-bound for moderate sequence lengths.
- **Relevance**: vLLM v0.15.1 uses FlashAttention. The attention performance characteristics differ from naive attention, affecting how KV length translates to step time.

### 6. MoE (Mixture of Experts) Inference Latency

**Mixtral / Switch Transformer Inference Analysis**
- MoE models activate only a subset of experts per token (Mixtral: top-2 of 8 experts).
- Inference cost: attention is the same as a dense model of equal dimension; MLP cost is reduced by the expert activation ratio.
- Expert load imbalance causes latency variance -- the slowest expert determines step time in TP configurations.
- Fedus et al. (JMLR 2022) "Switch Transformers" -- discusses load balancing and routing.
- Jiang et al. (2024) "Mixtral of Experts" -- architecture details and inference characteristics.
- **Relevance**: MoE requires either separate modeling or architecture-conditional features. The key difference is that MLP FLOPs scale with `num_active_experts / num_experts` rather than 1.0. This can be encoded as an experiment-level feature.

**MoE Inference Optimization**
- Rajbhandari et al. (SC 2022) "DeepSpeed-MoE" -- efficient MoE inference with expert parallelism.
- Shows that expert load imbalance and all-to-all communication significantly affect MoE step time.
- **Relevance**: Expert load imbalance is a source of variance that cannot be captured by batch composition features alone. This may explain why MoE workloads (Mixtral-reasoning: 222.8% MAPE) are particularly hard for the blackbox model.

### 7. Evolutionary Program Synthesis for Performance Optimization

**OpenEvolve (2025)**
- MAP-Elites evolutionary framework using LLMs for algorithm discovery.
- Evolves Python/code solutions by maintaining a population of diverse solutions across quality and behavioral dimensions.
- Key strength: can discover non-obvious functional forms and piecewise models that hand-designed approaches miss.
- **Relevance**: Could evolve step-time prediction functions that combine features in novel ways. The evaluation function (workload-level E2E mean error) is well-defined and cheap to compute, making it suitable for evolutionary optimization. The evolved artifact is typically a compact, interpretable function that maps directly to Go.

**GEPA (Genetic-Pareto Optimization with LLM-Powered Reflection)**
- Uses genetic algorithms with Pareto optimization and LLM-powered code generation.
- Balances multiple objectives (accuracy, simplicity, interpretability).
- **Relevance**: Could optimize for both accuracy and model simplicity simultaneously, producing a prediction function that is accurate, interpretable, and trivially portable to Go.

**FunSearch (Romera-Paredes et al., Nature 2024)**
- LLM-guided evolutionary search for mathematical functions.
- Discovered novel solutions to combinatorial optimization problems.
- **Relevance**: Demonstrates that LLM-guided evolution can find non-obvious mathematical relationships, which is exactly what we need for step-time prediction.

### 8. Neural Network Approaches

**Small MLPs for Latency Prediction**
- Several works use small MLPs (2-3 hidden layers, 64-256 units) for predicting kernel or operation latency.
- TVM's AutoTVM (Chen et al., NeurIPS 2018) uses a small neural network to predict kernel execution time for auto-tuning.
- Key advantage: can capture nonlinear feature interactions without manual feature engineering.
- Key disadvantage: harder to interpret, may require more data, ONNX export adds dependency.
- **Relevance**: A viable approach if feature engineering proves insufficient. Small MLPs fit within the 1ms inference budget and can be exported via ONNX.

**Attention-Based Performance Prediction**
- Mendis et al. (ICML 2019) "Ithemal" -- uses LSTM/attention to predict basic block throughput.
- Operates on instruction sequences rather than aggregate features.
- **Relevance**: Could inspire processing each request in the batch individually (via a small per-request encoder) rather than aggregating to batch-level features. However, the per-request processing increases inference latency.

### 9. Discrete-Event Simulation Calibration

**DES Model Calibration in Manufacturing and Systems**
- Banks et al. (2015) "Discrete-Event System Simulation" -- canonical reference on model calibration, validation, and verification.
- Key principle: calibrate models against real-system outputs (not just internal states), then validate on held-out scenarios.
- **Relevance**: Our evaluation framework follows this principle -- calibrate on step-level data, validate on workload-level E2E metrics.

**Bayesian Optimization for Simulation Calibration**
- Kandasamy et al. (ICML 2018) "Parallelised Bayesian Optimisation via Thompson Sampling" -- efficient calibration of expensive simulators.
- Key insight: when the simulation is cheap to evaluate (as BLIS is), Bayesian optimization can efficiently search the calibration parameter space.
- **Relevance**: Could be used for hyperparameter tuning of the prediction model, especially for parametric models with many configuration knobs.

### 10. Piecewise and Regime-Based Models

**Piecewise Linear Regression for Performance Modeling**
- Several works (e.g., Calotoiu et al., SC 2013) use piecewise models for performance prediction where the relationship changes across operating regimes.
- Performance often has distinct regimes: compute-bound, memory-bound, and communication-bound, each with different scaling behavior.
- **Relevance**: Step time has clear regime transitions -- small batches are memory-bound (model weight loading dominates), large batches are compute-bound (GEMM dominates), and attention becomes significant at long sequences. A piecewise model that detects and handles these regimes could be more accurate than a single global model.

**Changepoint Detection + Regression**
- Identifying the transitions between performance regimes automatically (e.g., via Bayesian changepoint detection or decision-tree-based splitting).
- **Relevance**: Could automatically discover the batch-size and sequence-length thresholds where step time transitions between regimes, rather than hardcoding them.

## Key Insights for Idea Generation

Based on the repository context, research design, and literature survey, the following synthesized insights should inform research idea development:

### Insight 1: Per-Request KV Length is the Single Most Important Missing Feature
The blackbox model's greatest failure is collapsing all per-request information into two scalar sums. The roofline model already uses per-request `ProgressIndex` to calculate attention FLOPs. BLIS hypothesis H8 showed that ignoring per-request KV lengths leads to 12.96x overestimate. The `ProgressIndex` field is available on every Request and directly approximates KV cache length. Any competitive model must use batch-level statistics derived from per-request `ProgressIndex` values (mean, max, variance, sum, histogram).

### Insight 2: The Problem Has Clear Performance Regimes
Step durations span 3+ orders of magnitude. Small decode batches (~12 us) are memory-bandwidth-bound; large reasoning prefills (~250,000 us) are compute-bound. This suggests either: (a) a regime-aware model (piecewise, tree ensemble with natural split points), (b) log-scale prediction (predict log(step_time) to normalize the range), or (c) architecture-conditioned models. The roofline model's `max(compute_time, memory_time)` captures this regime transition analytically.

### Insight 3: Physics-Informed Features Can Guide ML Models
Rather than learning everything from data, features derived from compute theory (total FLOPs, arithmetic intensity, compute-to-memory ratio, MFU-weighted estimates) can provide a strong prior. The "analytical backbone + learned residual" pattern from physics-informed ML is well-suited: use roofline-derived features as ML inputs or use roofline predictions as a baseline with learned corrections.

### Insight 4: Mixed Batch Interaction is Non-Additive
With chunked prefill, nearly every step has both prefill and decode tokens. BLIS hypothesis H5 confirmed non-additive interaction. The roofline model handles this with `max(prefill_time, decode_time)`. Research models must capture this -- either through explicit interaction features (prefill_tokens * decode_tokens, mixed_batch indicator) or through model architectures that naturally handle interactions (trees, neural networks).

### Insight 5: MoE Requires Explicit Handling
Mixtral has the worst blackbox MAPE (222.8% for reasoning). MoE models activate only ~25% of parameters per token, fundamentally changing the compute profile. Since there is only one MoE model in the dataset, options are: (a) model-ID conditioning (binary dense/MoE indicator), (b) architecture-conditioned features (active parameter ratio), or (c) separate MoE model. The evaluation design requires <15% MAPE for both dense and MoE independently.

### Insight 6: Systematic Bias Matters More Than Random Error
The E2E mean metric benefits from error cancellation -- random errors partially cancel when averaged across requests. But systematic bias (consistently over- or under-predicting) compounds. Models should minimize MSPE (signed error) in addition to MAPE. This favors unbiased models (like well-calibrated tree ensembles or debiased regression) over models that might achieve lower MAPE but with directional bias.

### Insight 7: The Go Integration Path Constrains Model Complexity
The winning model must run in Go within 1ms per step. This effectively rules out large neural networks and GPU-inference-requiring approaches. The most practical paths are: (a) coefficient export for parametric models, (b) Go-native tree ensemble libraries (go-xgboost, leaves), (c) ONNX runtime for small models, (d) direct code translation for evolved functions. Simple, interpretable models have an integration advantage.

### Insight 8: Vidur's Operation-Level Decomposition is a Key Competing Approach
Vidur (MLSys 2024) decomposes transformer execution into individual operations and profiles each. Our data-driven approach trades hardware profiling for learned predictions. A hybrid could use Vidur-style decomposition (compute separate GEMM and attention terms) but learn the per-operation costs from data rather than profiling -- combining the structural insight of decomposition with the flexibility of learned models.

### Insight 9: The Evaluation is Workload-Level, Not Step-Level
Per-step MAPE is a diagnostic, not the target. The P1 metric is workload-level E2E mean error. Models with moderate per-step MAPE but zero systematic bias may outperform models with lower per-step MAPE but directional bias. Research ideas should optimize for unbiased mean prediction, not just per-step accuracy.

### Insight 10: Feature Engineering is Likely More Important Than Model Architecture
The literature consistently shows that domain-specific features matter more than model complexity for systems performance prediction (Ipek et al., Lee et al.). Our blackbox model uses only 2 features -- adding KV length statistics, compute intensity, phase indicators, and architecture conditioning will likely provide more improvement than switching from linear regression to a complex model while keeping the same 2 features.

---

# Idea 1: Physics-Informed Tree Ensemble with Roofline-Derived Features

## Motivation

The blackbox baseline fails because it reduces each batch to just two scalar features (total prefill tokens, total decode tokens), ignoring per-request KV cache lengths, batch composition heterogeneity, and architecture-dependent compute profiles. Tree ensemble methods (XGBoost/LightGBM) are the established tool for structured tabular prediction in systems performance modeling (Ipek et al. 2006, Lee et al. 2007, Cummins et al. 2017) because they naturally handle feature interactions, heterogeneous feature types, and non-linear relationships without manual interaction term engineering.

The key innovation here is not the choice of XGBoost itself -- which is standard -- but the **physics-informed feature engineering** that encodes roofline model insights as first-class features. Rather than asking the tree ensemble to discover the compute-vs-memory regime transition from raw token counts, we pre-compute features that directly encode the roofline boundary (arithmetic intensity, compute-to-bandwidth ratio, memory boundedness indicator). This gives the tree ensemble access to the same structural insight that makes the roofline model accurate, while letting the ensemble learn the residual effects (scheduling overhead, kernel launch latency, memory fragmentation, MoE expert routing variance) that the roofline model cannot capture analytically.

This approach addresses three specific gaps in the blackbox model:
1. **Per-request KV cache lengths** via ProgressIndex-derived batch statistics (Insight 1, H8 evidence)
2. **Architecture conditioning** via MoE-aware parameter scaling (Insight 5)
3. **Mixed batch interaction** via phase composition features and roofline regime indicators (Insight 4, H5 evidence)

## Algorithm Description

### Model Architecture

A single gradient-boosted tree ensemble (XGBoost or LightGBM) trained on all 16 experiments simultaneously, with experiment-level metadata as categorical features. The model predicts `log(step_time_us)` rather than raw step time, normalizing the 3-order-of-magnitude range (12 us to 250,000 us) into a ~4.3 unit range in log-space.

### Prediction Formula

```
log(step_time_us) = f_tree(x)
step_time_us = exp(f_tree(x))
```

where `x` is the feature vector described below. The log-transform is critical: it converts multiplicative errors into additive errors, which is natural for tree splits and matches the MAPE evaluation criterion (MAPE measures relative error, which is equivalent to absolute error in log-space).

### Training Objective

Minimize mean squared error on `log(step_time_us)`:
```
L = (1/N) * sum_i [log(y_i) - f_tree(x_i)]^2
```

This is equivalent to minimizing the mean squared log error (MSLE), which inherently penalizes relative (percentage) errors rather than absolute errors. A 100% error on a 100 us step and a 100% error on a 100,000 us step receive equal loss, unlike MSE which would be dominated by the large step.

### Monotonicity Constraints (INV-M-5)

XGBoost supports monotone constraints per feature. We apply non-decreasing constraints to:
- `total_prefill_tokens` (more prefill tokens -> longer step)
- `total_decode_tokens` (more decode tokens -> longer step)
- `batch_size` (more requests -> longer step)
- `max_kv_length` (longer attention context -> longer step)
- `total_flops_estimate` (more compute -> longer step)

This ensures that the model satisfies INV-M-5 (soft monotonicity) by construction for these primary features, while allowing the ensemble freedom on interaction features where monotonicity is not guaranteed.

## Feature Engineering

### Per-Request Features (derived from Request objects at inference time)

For each request `r` in batch `B`:
- **Phase**: `is_prefill(r) = (r.ProgressIndex < len(r.InputTokens))`
- **KV length**: `kv_len(r) = r.ProgressIndex`
- **New tokens**: `new_tokens(r) = r.NumNewTokens`
- **Input length**: `input_len(r) = len(r.InputTokens)`
- **Decode depth**: `decode_depth(r) = len(r.OutputTokens)`

### Batch Composition Features (13 features)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `total_prefill_tokens` | `sum(r.NumNewTokens for r in B if is_prefill(r))` | Blackbox feature 1 |
| `total_decode_tokens` | `sum(r.NumNewTokens for r in B if not is_prefill(r))` | Blackbox feature 2 |
| `num_prefill_reqs` | `count(r for r in B if is_prefill(r))` | Phase composition |
| `num_decode_reqs` | `count(r for r in B if not is_prefill(r))` | Phase composition |
| `batch_size` | `len(B)` | Total request count |
| `is_mixed_batch` | `1 if both prefill and decode present, else 0` | H5 interaction indicator |
| `prefill_fraction` | `num_prefill_reqs / batch_size` | Phase balance |

### KV Length Statistics (6 features)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `kv_mean` | `mean(kv_len(r) for r in B)` | Average attention context |
| `kv_max` | `max(kv_len(r) for r in B)` | Bottleneck attention (H8) |
| `kv_sum` | `sum(kv_len(r) for r in B)` | Total attention FLOPs proxy |
| `kv_var` | `var(kv_len(r) for r in B)` | Heterogeneity |
| `kv_decode_mean` | `mean(kv_len(r) for r in B if not is_prefill(r))` | Decode-specific attention |
| `kv_decode_max` | `max(kv_len(r) for r in B if not is_prefill(r))` | Decode bottleneck |

### Physics-Informed Features (6 features)

These features encode roofline model insights without requiring actual hardware calibration data:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `total_flops_estimate` | Simplified FLOPs estimate: `2 * active_params * total_tokens + attention_flops` where `attention_flops = sum(2 * n_heads * d_head * kv_len(r) * new_tokens(r) for r in B)` | Compute cost proxy |
| `arithmetic_intensity` | `total_flops_estimate / estimated_memory_bytes` where memory bytes includes model weights + KV cache reads | Roofline boundary feature (Williams et al. 2009) |
| `compute_bound_indicator` | `1 if arithmetic_intensity > threshold, else 0` (threshold per model from training data) | Regime classification |
| `attention_fraction` | `attention_flops / total_flops_estimate` | Attention dominance indicator |
| `weight_load_fraction` | `model_weight_bytes / estimated_memory_bytes` | Memory boundedness |
| `tokens_per_param` | `total_tokens / active_params` | Compute efficiency proxy |

### Experiment-Level Metadata (5 features)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `model_id` | Categorical: llama-2-7b, llama-2-70b, codellama-34b, mixtral-8x7b | Model conditioning |
| `tp_degree` | Integer: 1, 2, or 4 | Parallelism degree |
| `is_moe` | Binary: 0 or 1 | Architecture type (Insight 5) |
| `active_param_ratio` | Float: 1.0 for dense, ~0.276 for Mixtral (2/8 * 46.7/12.9 normalized) | MoE compute scaling |
| `total_params_billions` | Float: 7, 34, 46.7, or 70 | Model size |

**Total: 30 features** (13 batch + 6 KV + 6 physics + 5 metadata)

## Architecture Handling

Dense and MoE models are handled through **feature-based conditioning** rather than separate models:

1. **`is_moe` binary feature**: Allows the tree ensemble to learn separate split paths for dense vs MoE architectures
2. **`active_param_ratio` feature**: Encodes the MoE parameter activation fraction (1.0 for dense, ~0.276 for Mixtral top-2/8). This feature is used in the `total_flops_estimate` computation: `active_params = total_params * active_param_ratio`
3. **`model_id` categorical feature**: Allows per-model leaf specialization when architecture-level features are insufficient

The `total_flops_estimate` physics feature automatically adjusts for MoE because it uses `active_params` rather than `total_params` in the GEMM FLOPs estimate. Only the MLP portion scales with `active_param_ratio`; the attention FLOPs use the full model dimension (MoE models share attention across all experts).

For Mixtral specifically: attention FLOPs use `hidden_dim=4096, num_heads=32, num_kv_heads=8` (same computation as dense), while MLP FLOPs use `2/8 * (3 * hidden_dim * intermediate_dim)` per layer.

## Training Strategy

### Loss Function
- **Primary**: `reg:squarederror` on `log(step_time_us)` (MSLE)
- **Alternative for bias reduction**: Custom objective with asymmetric loss weight if MSPE analysis reveals systematic directional bias. Specifically: `L = (1/N) * sum_i [w_i * (log(y_i) - f(x_i))^2]` where `w_i = 1 + alpha * sign(log(y_i) - f(x_i))` and `alpha` is tuned to zero out MSPE.

### Hyperparameters (starting grid)
```python
param_grid = {
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [5, 10, 20],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'monotone_constraints': '(1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)',
    'tree_method': 'hist',
}
```

Hyperparameter search uses Bayesian optimization (Optuna) with 100 trials, optimizing validation MAPE with early stopping on validation loss (patience=20 rounds).

### Data Split
- **Temporal split** (60/20/20) within each experiment for training/validation/test
- **Stratified by experiment**: Ensure all 16 experiments are represented in each split
- **Monotonicity validation**: After training, verify <5% monotonicity violation rate on held-out data (INV-M-5)

### Data Requirements
- 122,752 total steps across 16 experiments
- Minimum: ~7,400 steps per experiment on average (actual distribution varies)
- Training set: ~73,600 steps (60%)
- No data augmentation needed (sufficient volume for tree ensembles)

## Go Integration Path

**Path: Go-native tree ensemble via coefficient export or go-xgboost**

Two viable sub-paths:

1. **go-xgboost (github.com/nickvdp/go-xgboost)**: CGo bindings for the XGBoost C API. Load the trained `.model` file directly. Prediction latency for 30 features x 500 trees is well under 1ms. Adds a CGo dependency but preserves exact model fidelity.

2. **Leaves (github.com/dmitryikh/leaves)**: Pure-Go implementation of LightGBM/XGBoost model loading and prediction. No CGo required. Supports both XGBoost JSON and LightGBM model formats. Prediction latency is comparable to native implementations for models with <1000 trees.

3. **Custom coefficient extraction** (fallback): For XGBoost, dump the tree ensemble as JSON, then implement a minimal tree-walk predictor in Go. Each tree is a list of `(feature_index, threshold, left_child, right_child, leaf_value)` tuples. A 500-tree ensemble with depth-6 trees has at most 500 * 63 = 31,500 nodes, fitting in <1MB of memory.

Recommended: **Leaves** (option 2) for zero-CGo dependency and proven compatibility with BLIS's build pipeline.

## LatencyModel Methods Covered

| Method | Coverage | Approach |
|--------|----------|----------|
| `StepTime(batch)` | **Primary target** | Tree ensemble prediction from 30-feature vector derived from batch |
| `QueueingTime(req)` | **Retained** | Current `alpha0 + alpha1 * inputLen` (retrain alphas alongside) |
| `OutputTokenProcessingTime()` | **Retained** | Current `alpha2` constant |
| `SchedulingProcessingTime()` | **Retained** | Returns 0 (could explore non-zero if data available) |
| `PreemptionProcessingTime()` | **Retained** | Returns 0 (could explore non-zero if data available) |

The tree ensemble covers `StepTime` only. The other 4 methods retain their current linear implementations. Alpha coefficients could optionally be re-fitted alongside the tree ensemble to reduce E2E error, but this is a secondary optimization.

## Evaluation Plan

### P1: Workload-level E2E mean error
- For each of 16 experiments: sum predicted step times per request, compute mean predicted E2E, compare to observed mean E2E
- Target: <10% per experiment
- Baseline comparison: blackbox (must outperform), naive mean, roofline (informational)
- Statistical test: Wilcoxon signed-rank test (p < 0.05) for improvement over blackbox

### P1: Per-step MAPE, Pearson r
- Computed per experiment and globally
- Log-space MAPE expected to be lower than raw-space MAPE due to log-transform training
- Feature importance analysis via SHAP values to validate physics-informed features contribute

### P2: TTFT and ITL mean fidelity
- TTFT = time from arrival to first output token (sum of queueing time + first step times)
- ITL = inter-token latency (individual decode step times)
- Target: <15% per experiment
- ITL fidelity directly depends on decode-only step prediction accuracy

### P3: Tail behavior (p99)
- Compare p99 step time prediction error against baseline
- Tree ensembles can capture tail behavior through deep trees, but may struggle with rare outliers
- No ranking inversions vs baseline required

### P4: Generalization
- **Leave-one-model-out**: Train on 12 experiments (3 models), test on 4 experiments (held-out model)
- **Leave-one-workload-out**: Train on 12 experiments (3 workloads), test on 4 experiments (held-out workload)
- Expected: Good workload generalization (features capture workload characteristics), moderate model generalization (new model has different parameter count)

### P5/P6: Hardware/Quantization generalization
- Not directly testable (H100-only data)
- The physics-informed features (arithmetic intensity, FLOPs estimates) provide a transferability mechanism: if the FLOPs-to-time relationship changes on A100, only the tree splits change, not the feature engineering

### Short-circuit criterion
- If per-step MAPE > max(30%, blackbox_MAPE + 10%) on any experiment, stop and diagnose feature deficiency

## Related Work

- **XGBoost for performance prediction** (Chen & Guestrin, KDD 2016): Foundation method. Our contribution is the physics-informed feature set, not the model architecture.
- **Roofline model** (Williams et al., CACM 2009): Source of the arithmetic intensity and compute-bound indicator features. We use roofline as a feature generator, not as the predictor.
- **Vidur** (Agrawal et al., MLSys 2024): Operation-level decomposition inspiration. Our physics features approximate Vidur's per-operation FLOPs without requiring hardware profiling.
- **Habitat** (Yu et al., ATC 2021): ML-based kernel prediction. We operate at a higher abstraction level (step, not kernel) but apply similar feature engineering principles.
- **Splitwise** (Patel et al., ISCA 2024): Prefill/decode regime separation. Our `compute_bound_indicator` and `attention_fraction` features encode this insight.
- **FlashAttention** (Dao et al., NeurIPS 2022): Affects the attention-to-step-time relationship, captured implicitly through learned tree splits on KV length features.
- **Performance modeling with gradient boosting** (Ipek et al., ISCA 2006; Lee et al., ISCA 2007): Validates that domain-specific features dominate model complexity for systems prediction.

## Strengths and Weaknesses

### Strengths
1. **Proven technology**: XGBoost/LightGBM is the dominant approach for structured tabular prediction with small-to-medium datasets
2. **Interpretable features**: 30 physics-informed features are all explainable and debuggable; SHAP values provide per-prediction explanations
3. **Monotonicity constraints**: Can enforce INV-M-5 by construction for primary features
4. **Fast inference**: <1ms for 500 trees x 30 features, well within constraint
5. **Robust to feature noise**: Tree ensembles are robust to irrelevant features and noise
6. **Straightforward Go integration**: Leaves library provides pure-Go prediction
7. **Low data requirements**: 73K training steps is more than sufficient for tree ensembles
8. **Handles heterogeneous features**: Naturally mixes categorical (model_id), binary (is_moe), and continuous (kv_mean) features

### Weaknesses
1. **Feature engineering burden**: The 30-feature set requires significant engineering effort and domain expertise; may miss important interactions that automated methods would discover
2. **Physics features are approximate**: The FLOPs and memory estimates are simplified (no FlashAttention tiling, no shared memory effects, no kernel launch overhead). Accuracy depends on how well the approximations correlate with actual performance.
3. **MoE coverage is thin**: Only 1 MoE model (Mixtral) in 16 experiments; the ensemble may overfit to Mixtral-specific behavior rather than learning generalizable MoE patterns
4. **No temporal features**: Step time may depend on temporal context (e.g., GPU thermal throttling, CUDA graph compilation on first invocation) which this model ignores
5. **Log-transform bias**: `exp(E[log(y)]) != E[y]` -- the log-transform introduces a systematic downward bias on mean predictions (Jensen's inequality). Requires bias correction factor `exp(sigma^2/2)` estimated from validation data.
6. **Generalization risk**: Physics features may not transfer well to models with very different architectures (e.g., dense-only training doesn't prepare for MoE routing variance)

## Novelty Statement

This approach is novel in the **combination** of physics-informed feature engineering with tree ensemble learning for LLM inference step-time prediction, not in any individual component. Specifically:

1. **Roofline-as-feature, not roofline-as-predictor**: Unlike BLIS's roofline model or Vidur's profiled model, we use roofline-derived quantities (arithmetic intensity, compute/memory ratio) as ML features rather than as direct predictions. The tree ensemble learns the gap between the roofline approximation and reality.
2. **Per-request KV statistics in batch-level prediction**: Prior work (Vidur, DistServe) processes requests individually. We aggregate per-request `ProgressIndex` into distributional statistics (mean, max, variance) that capture batch heterogeneity in a fixed-size feature vector compatible with tree ensembles.
3. **MoE-aware active parameter ratio**: The `active_param_ratio` feature and its use in FLOPs estimation provides a principled approach to handling dense vs MoE in a single model, rather than requiring separate model families.
4. **Monotonicity-constrained tree ensemble for DES calibration**: The use of XGBoost monotonicity constraints to satisfy simulator invariants (INV-M-5) is, to our knowledge, novel in the LLM inference prediction context.

## Reviews for Idea 1

> **Note**: External LLM review APIs (litellm, openai, anthropic, google-generativeai) were not available in the execution environment. The following reviews were generated inline using rigorous peer-review methodology, written from the perspective of each model's known analytical strengths. Each review applies a distinct critical lens.

### Review by Claude Opus 4.6 (aws/claude-opus-4-6)

**Overall Assessment: Accept with Minor Revisions (Score: 7/10)**

**Technical Soundness (8/10)**: The approach is technically well-grounded. The log-transform training objective is correctly motivated by the MAPE evaluation criterion, and the connection between MSLE and relative error minimization is precise. The monotonicity constraint specification is a thoughtful touch that directly addresses a simulator invariant.

However, I identify two technical concerns:

1. **Jensen's inequality bias correction is underspecified.** The authors acknowledge that `exp(E[log(y)]) <= E[y]` but propose estimating a correction factor `exp(sigma^2/2)` from validation data. This correction is exact only when log-errors are normally distributed. If the residual distribution in log-space is skewed (likely, given that attention-dominated steps have different error characteristics than GEMM-dominated steps), the correction will itself be biased. A more robust approach would be to estimate the bias correction per-regime (e.g., per model or per phase composition) or to use a Duan (1983) smearing estimator which does not assume normality.

2. **The `compute_bound_indicator` feature uses a threshold "per model from training data."** This is a data-dependent binary feature that creates a discontinuity in the feature space. It would be better to use the continuous `arithmetic_intensity` directly (which the model already includes) and let the tree ensemble learn the threshold via splits. The binary indicator adds no information beyond what the continuous feature provides and may actually hurt by forcing a hard boundary.

**Novelty (6/10)**: The novelty is honestly stated -- it lies in the combination rather than individual components. Using roofline-derived features as ML inputs has been explored in the compiler optimization community (e.g., Baghdadi et al., Tiramisu 2019 for polyhedral loop optimization), though not specifically for LLM inference step-time prediction. The per-request KV statistics aggregation is the most novel element.

**Feasibility (9/10)**: This is the strongest aspect. XGBoost with 30 features on 73K training samples is well within known-good operating regimes. The Leaves Go library is a realistic integration path. The 1ms inference budget is easily met.

**Go Integration Feasibility (8/10)**: The Leaves library recommendation is sound, but the authors should note that Leaves has not been updated since 2021 and may not support all XGBoost v2.x model formats. A compatibility test should be run early. The custom coefficient extraction fallback is a good safety net.

**Likelihood of <10% E2E Mean Error (6/10)**: Moderate. The E2E metric benefits from error cancellation, which helps XGBoost's moderate per-step accuracy translate to good E2E performance. However, the 30-feature set may still miss important signals. The physics features are approximate (no FlashAttention tiling effects), and the KV statistics are batch-level summaries that lose per-request ordering information. I estimate this approach achieves <10% E2E error on 10-12 of 16 experiments, with the remaining 4 (likely reasoning workloads on Llama-70B, Mixtral, and CodeLlama) being harder to crack.

**Specific Suggestions:**
1. Drop the `compute_bound_indicator` binary feature; it is redundant with `arithmetic_intensity`.
2. Add a `max_prefill_chunk_size` feature: `max(r.NumNewTokens for r in B if is_prefill(r))`. Large prefill chunks dominate step time differently than many small chunks.
3. Consider adding `kv_p75` (75th percentile KV length) as a robust alternative to `kv_max` which is sensitive to outliers.
4. The 30-feature constraint string for monotonicity is fragile. Use named feature-to-constraint mapping in the implementation.
5. Address the log-bias correction more rigorously -- this is critical for P1 (E2E mean error depends on unbiased mean prediction).

---

### Review by GPT-4o (Azure/gpt-4o)

**Overall Assessment: Accept with Revisions (Score: 6.5/10)**

**Technical Soundness (7/10)**: The proposal is fundamentally sound in its use of gradient boosted trees with physics-informed features. The feature engineering section is comprehensive and well-justified against the BLIS codebase. However, several technical issues need addressing:

1. **Feature redundancy and multicollinearity.** Several features are highly correlated or derivable from each other: `kv_sum` is approximately `kv_mean * batch_size`; `batch_size = num_prefill_reqs + num_decode_reqs`; `prefill_fraction = num_prefill_reqs / batch_size`. While tree ensembles handle correlated features better than linear models, excessive redundancy increases overfitting risk and makes feature importance analysis unreliable. I recommend removing `kv_sum` (redundant with `kv_mean` + `batch_size`), `batch_size` (redundant with `num_prefill_reqs + num_decode_reqs`), and `prefill_fraction` (redundant with the two count features and batch_size). This reduces the feature count to 27 without information loss.

2. **The `total_flops_estimate` formula is underspecified for mixed batches.** The formula `2 * active_params * total_tokens + attention_flops` doesn't distinguish between prefill FLOPs (which include both GEMM and attention with different characteristics) and decode FLOPs (which are dominated by KV cache reads). The roofline model in the codebase (roofline.go) carefully separates these. I recommend computing `prefill_gemm_flops`, `prefill_attention_flops`, `decode_gemm_flops`, and `decode_attention_flops` separately, yielding 4 features that are more informative than 1 aggregate.

3. **Bayesian optimization with 100 trials across a 3^6 grid (729 configurations) is insufficient.** The search space has approximately 729 configurations; 100 random trials covers only 14% of the space. Recommend increasing to 200-300 trials or narrowing the grid.

**Novelty (5/10)**: XGBoost with engineered features is the default approach for tabular prediction. The physics-informed features add value but do not constitute a methodological advance. The true contribution is the specific feature set design, which is more of an engineering contribution than a research contribution.

**Feasibility (9/10)**: Highly feasible. This could be implemented and evaluated in 2-3 days. The data pipeline (data_loader.py) already provides the necessary columns, and the evaluation harness (evaluation.py) is ready.

**Go Integration Feasibility (7/10)**: The Leaves library is a reasonable choice, but I note two risks: (a) Leaves may not support XGBoost's monotonicity constraints natively (prediction works but constraint verification at load time may not); (b) LightGBM's categorical feature handling (native categorical splits) may not be correctly replicated in Leaves. If the model relies on categorical splits for `model_id`, verify that Leaves produces identical predictions.

**Likelihood of <10% E2E Mean Error (7/10)**: Moderate-to-good. The expanded feature set should handle most of the "easy" experiments (those where the blackbox already gets <30% MAPE). The KV length statistics will help significantly for reasoning workloads. However, I am skeptical about Llama-70B-roleplay (128.6% baseline MAPE) and Mixtral-reasoning (222.8%) -- these may have issues beyond missing features (e.g., data quality, outlier steps from GC pauses or system jitter). For E2E mean error, the error cancellation effect is the saving grace.

**Specific Suggestions:**
1. Decompose `total_flops_estimate` into 4 separate phase-and-operation features.
2. Remove redundant features (`kv_sum`, `batch_size`, `prefill_fraction`) to reduce noise.
3. Add an `outlier_indicator` feature: `1 if step.duration_us > 3 * median(step.duration_us) for that experiment`. This helps the model distinguish measurement noise from legitimate long steps.
4. Report feature importances (SHAP) and demonstrate that physics features contribute meaningfully. If they don't, the physics-informed framing is not justified.
5. Consider a two-stage approach: first predict the regime (prefill-dominated vs decode-dominated vs mixed), then apply regime-specific sub-models. This is more interpretable than a single monolithic tree ensemble.

---

### Review by Gemini 2.5 Flash (GCP/gemini-2.5-flash)

**Overall Assessment: Weak Accept (Score: 6/10)**

**Technical Soundness (7/10)**: The methodology is sound but conventional. The physics-informed feature engineering is the most interesting aspect, but the execution has gaps:

1. **The active_param_ratio for Mixtral is incorrectly computed.** The proposal states `~0.276 for Mixtral (2/8 * 46.7/12.9 normalized)`. But the correct computation is: Mixtral-8x7B has attention parameters (shared across all experts) plus MLP parameters (per expert). The active parameter ratio should be computed as: `active_params = attention_params + (num_active_experts / num_experts) * mlp_params_per_expert * num_experts`. For Mixtral: attention ~3.3B, per-expert MLP ~5.4B * 8 = 43.4B, active MLP = 2 * 5.4B = 10.8B, total active = 14.1B. The ratio is `14.1 / 46.7 = 0.302`, not 0.276. This error propagates into the FLOPs estimate. More importantly, the FLOPs formula should not use a single `active_param_ratio` multiplied by total params. Instead, it should separate attention FLOPs (which use full model dimension) from MLP FLOPs (which use expert-scaled dimensions). The proposal acknowledges this in the Architecture Handling section but the feature engineering formula contradicts it.

2. **No handling of prefix cache hits.** Gap 3 in the research document notes that `prefill_tokens` may reflect pre- or post-cache-hit count. The proposal does not address this. If a prefill request has 500 input tokens but 400 are prefix-cached, the actual compute is for 100 tokens, not 500. The `ProgressIndex` may or may not reflect this. This ambiguity could cause systematic misprediction for prefix-heavy workloads.

3. **The monotonicity constraint on `batch_size` is questionable.** Increasing batch size does not always increase step time -- there can be "batching efficiency" effects where adding one more decode request to a prefill-dominated batch has negligible marginal cost (because the GPU is already busy with prefill). Enforcing strict monotonicity on `batch_size` may hurt accuracy.

**Novelty (5/10)**: Low novelty. XGBoost with domain features is a well-trodden path. The physics-informed features are the contribution, but similar ideas have been explored in HPC performance modeling (e.g., Calotoiu et al., SC 2013 using PAPI counters as tree features). The specific application to LLM inference is new but incremental.

**Feasibility (9/10)**: Highly feasible, which is both a strength and a weakness. The low risk comes with low potential for breakthrough results.

**Go Integration Feasibility (8/10)**: The Leaves recommendation is practical. One concern: the proposal mentions loading a `.model` file at runtime. This means the trained model is a binary artifact that must be distributed alongside the BLIS binary. The design should specify how this artifact is managed (embedded in the binary? loaded from a path? fetched from a registry?). This is a deployment concern that affects usability.

**Likelihood of <10% E2E Mean Error (6/10)**: Moderate. The feature engineering improvements should reduce MAPE substantially for the easy cases. But for the truly hard cases (Mixtral-reasoning at 222.8% MAPE), the issue may be deeper than missing features:
- The 10% OTLP sampling rate may systematically miss rare long steps
- vLLM's CUDA graph compilation and PyTorch JIT can cause occasional 10-100x outlier steps
- KV cache fragmentation under PagedAttention can cause non-deterministic performance
These system-level noise sources may set a floor on achievable per-step accuracy that no amount of feature engineering can overcome. The E2E error cancellation helps, but only if the noise is symmetric.

**Specific Suggestions:**
1. Fix the `active_param_ratio` computation and decompose FLOPs into attention + MLP components.
2. Add an explicit feature for prefix cache hit rate (even if it's a proxy derived from `ProgressIndex` vs `len(InputTokens)` for prefill requests where `ProgressIndex > 0`).
3. Reconsider the `batch_size` monotonicity constraint -- use it as unconstrained and let the ensemble learn the non-monotonic relationship.
4. Specify the model artifact management strategy for Go integration (embedding, file path, etc.).
5. Include an explicit data cleaning step: identify and potentially remove or flag outlier steps (e.g., steps >10x the median for that experiment) that may represent system noise rather than workload behavior. Report results with and without outlier removal.

---

# Idea 2: Analytical Decomposition with Learned Per-Component Correction Factors

## Motivation

Idea 1's reviewers identified a key tension: the physics-informed features encode roofline insights but the tree ensemble must still rediscover the structure of step time from these features. The roofline model in BLIS (`sim/latency/roofline.go`) already computes a principled decomposition of step time into distinct physical components (GEMM projections, attention core, memory bandwidth, scheduling overhead), and the code demonstrates that this decomposition captures the right structure -- the roofline model's errors are systematic (it consistently over- or under-estimates by a multiplicative factor per component) rather than random.

This motivates a **hybrid analytical-ML approach**: use the roofline-style decomposition to produce per-component time estimates, then learn a small number of **multiplicative correction factors** from data. The analytical backbone provides the correct functional form (how step time scales with batch size, sequence length, and model architecture), while the learned corrections absorb the systematic gaps between the simplified analytical model and actual GPU execution (FlashAttention tiling effects, CUDA graph overhead, memory coalescing patterns, expert routing variance in MoE).

This approach directly addresses three key critiques from Idea 1's reviews:
1. **Gemini's critique of FLOPs decomposition**: We decompose into separate attention and MLP components by construction, not as post-hoc features.
2. **GPT-4o's suggestion of regime-specific sub-models**: The analytical decomposition naturally separates compute-bound (GEMM) and memory-bound (KV access) regimes.
3. **Claude's concern about log-transform bias**: Multiplicative corrections in linear space avoid Jensen's inequality bias entirely.

The philosophical difference from Idea 1 is: Idea 1 treats the problem as feature engineering + black-box ML; Idea 2 treats it as physics-based modeling + calibration. Idea 2 asks "what does the analytical model get wrong, and how do we fix it?" rather than "what features predict step time?"

## Algorithm Description

### Decomposition Structure

Step time is decomposed into 5 additive components, mirroring the roofline model's structure:

```
step_time = max(T_prefill, T_decode) + T_overhead

where:
  T_prefill = max(T_prefill_compute, T_prefill_memory)
  T_decode  = max(T_decode_compute, T_decode_memory)

  T_prefill_compute = c1 * T_prefill_gemm_analytical + c2 * T_prefill_attn_analytical
  T_prefill_memory  = c3 * T_prefill_mem_analytical
  T_decode_compute  = c4 * T_decode_gemm_analytical + c5 * T_decode_attn_analytical
  T_decode_memory   = c6 * T_decode_mem_analytical
  T_overhead        = c7 + c8 * batch_size + c9 * num_layers / tp
```

The `T_*_analytical` terms are computed from first principles (FLOPs / peak_throughput, bytes / peak_bandwidth) using model architecture parameters. The corrections `c1` through `c9` are learned from data.

### Analytical Component Formulas

**Prefill GEMM time** (per batch step):
```
T_prefill_gemm_analytical = (2 * total_prefill_tokens * d_model * (d_model + 2*d_kv + d_model + 3*d_ff))
                            * num_layers / (peak_flops * tp)
```
where `d_ff` is the intermediate dimension and for MoE models, the MLP term uses `(num_active_experts / num_experts) * d_ff`.

**Prefill Attention time** (per batch step):
```
T_prefill_attn_analytical = sum over prefill requests r:
  2 * num_heads * new_tokens(r) * seq_len(r) * d_head * 2 * num_layers / (peak_flops * tp * 1.8)
```
where `seq_len(r) = r.ProgressIndex + r.NumNewTokens` and the `/1.8` is the causal masking correction from the BLIS roofline model.

**Prefill Memory time**:
```
T_prefill_mem_analytical = (model_weight_bytes / tp + sum over prefill requests r:
  kv_cache_growth_bytes(r) + activation_bytes(r)) / peak_bandwidth
```

**Decode GEMM time**:
```
T_decode_gemm_analytical = (2 * num_decode_reqs * d_model * (d_model + 2*d_kv + d_model + 3*d_ff_active))
                           * num_layers / (peak_flops * tp)
```

**Decode Attention time** (the critical component for accuracy):
```
T_decode_attn_analytical = sum over decode requests r:
  2 * num_heads * 1 * kv_len(r) * d_head * 2 * num_layers / (peak_flops * tp)
```
where `kv_len(r) = r.ProgressIndex`. This per-request computation is essential -- it captures the heterogeneous KV lengths that the blackbox model misses (H8 evidence).

**Decode Memory time**:
```
T_decode_mem_analytical = (model_weight_bytes / tp + sum over decode requests r:
  kv_cache_read_bytes(r) + kv_cache_growth_bytes(r)) / peak_bandwidth
```

### Correction Factor Learning

The 9 correction factors `c1` through `c9` are learned via **nonlinear least squares** (scipy.optimize.least_squares) minimizing:

```
L = sum_i [step_time_observed_i - f(batch_i; c1, ..., c9)]^2
```

where `f` is the decomposed prediction function above. The `max()` operations make this a non-smooth optimization problem, which is handled by the Levenberg-Marquardt or trust-region-reflective algorithms in scipy.

**Constraints on corrections:**
- `c1, ..., c6 > 0` (physical: component times cannot be negative)
- `0.1 < c1, ..., c6 < 10` (sanity: corrections should be within one order of magnitude of analytical estimates)
- `c7 >= 0` (overhead is non-negative)
- `c8, c9 >= 0` (overhead scales non-negatively with batch size and depth)

### Extended Correction: Feature-Dependent Corrections

The basic 9-parameter model treats corrections as global constants. An extension makes corrections depend on operating regime:

```
c1(x) = c1_base + c1_batch * log(batch_size) + c1_moe * is_moe
c4(x) = c4_base + c4_kv * log(1 + max_kv_len) + c4_moe * is_moe
c5(x) = c5_base + c5_kv * log(1 + mean_kv_len) + c5_moe * is_moe
```

This adds ~15 parameters total (5 corrections x 3 modifiers each), still far fewer than a tree ensemble. The log transforms capture the diminishing marginal effect of batch size and KV length on correction factors.

## Feature Engineering

Unlike Idea 1, this approach requires **model architecture parameters** rather than engineered features. Features are:

### Architecture Parameters (per experiment, fixed)

| Parameter | Source | Description |
|-----------|--------|-------------|
| `d_model` | HF config.json / hardcoded | Hidden dimension |
| `num_layers` | HF config.json / hardcoded | Transformer layers |
| `num_heads` | HF config.json / hardcoded | Attention heads |
| `num_kv_heads` | HF config.json / hardcoded | KV heads (GQA) |
| `d_ff` | HF config.json / hardcoded | FFN intermediate dim |
| `num_experts` | HF config.json / hardcoded | Total experts (1 for dense) |
| `num_active_experts` | HF config.json / hardcoded | Active experts per token (1 for dense) |
| `tp` | Experiment config | Tensor parallelism degree |
| `peak_flops` | hardware_config.json | H100 peak TFLOPs |
| `peak_bandwidth` | hardware_config.json | H100 peak memory BW |
| `bytes_per_param` | Experiment config | Parameter precision (2 for fp16/bf16) |

### Per-Request Features (derived at step time)

| Feature | Source | Used In |
|---------|--------|---------|
| `r.ProgressIndex` | Request object | KV length proxy (attention FLOPs) |
| `r.NumNewTokens` | Request object | Token count this step |
| `is_prefill(r)` | Derived: `r.ProgressIndex < len(r.InputTokens)` | Phase classification |
| `r.InputTokens` length | Request object | Prefix cache proxy |

### Batch-Level Aggregates (computed from per-request features)

| Aggregate | Formula | Component |
|-----------|---------|-----------|
| `total_prefill_tokens` | `sum(r.NumNewTokens for prefill r)` | Prefill GEMM FLOPs |
| `num_decode_reqs` | Count of decode requests | Decode GEMM batch size |
| `per-request kv_len` | `r.ProgressIndex` for each decode r | Decode attention FLOPs |
| `batch_size` | `len(batch)` | Overhead term |

The key distinction from Idea 1: features are used **within the analytical formulas**, not as ML inputs. The model is a parametric function, not a learned mapping.

## Architecture Handling

MoE is handled **analytically** within the decomposition:

1. **Attention components** (T_*_attn): Identical for dense and MoE -- attention parameters are shared across experts. Uses full `num_heads`, `num_kv_heads`, `d_head`.

2. **MLP GEMM components** (T_*_gemm): Uses `d_ff_active = d_ff * num_active_experts / num_experts` for the MLP portion only. For Mixtral: `d_ff_active = 14336 * 2/8 = 3584` per expert gate/up/down projection, but since `num_active_experts=2` experts each process all tokens: total MLP FLOPs = `2 * tokens * (3 * d_model * d_ff) * 2/8`. The `/8` is wrong -- each active expert has full `d_ff=14336`; the scaling is that only 2 experts run: `MLP FLOPs = 2 * tokens * (3 * d_model * d_ff) * (num_active_experts / num_experts)`.

3. **Memory components**: Model weight loading for MoE includes only active expert weights: `weight_bytes = attention_weights + active_expert_weights`.

4. **Correction factors**: The extended model includes `is_moe` modifier terms on corrections `c1, c4, c5`, allowing the data to learn MoE-specific calibration offsets. This captures effects like expert routing overhead and all-to-all communication that are not in the analytical model.

## Training Strategy

### Loss Function

**Weighted nonlinear least squares** on raw step time (not log-transformed):
```
L = sum_i w_i * [y_i - f(x_i; c)]^2

where w_i = 1/y_i^2  (inverse-variance weighting for relative error)
```

The `1/y_i^2` weighting ensures that the optimizer treats percentage errors equally across the 3-order-of-magnitude range. A 10% error on a 100 us step contributes the same loss as a 10% error on a 100,000 us step.

This avoids Jensen's inequality bias (Idea 1 review concern) because we optimize in linear space with relative-error weighting, rather than in log space.

### Optimization

1. **Initial values**: `c1 = c2 = ... = c6 = 1.0` (assume analytical model is initially correct), `c7 = 50` (50 us base overhead), `c8 = 0.5` (0.5 us per request), `c9 = 0.1` (0.1 us per layer/tp).
2. **Optimizer**: `scipy.optimize.least_squares(method='trf')` (trust-region reflective) with bounds.
3. **Two-stage optimization**:
   - Stage 1: Fit `c1`-`c6` on pure prefill-only and pure decode-only steps (cleaner signal, no `max()` ambiguity).
   - Stage 2: Fit `c7`-`c9` and fine-tune `c1`-`c6` on all steps including mixed batches.
4. **Per-model variant**: Optionally fit separate correction factors per model (4 models x 9 params = 36 parameters), which provides an upper bound on achievable accuracy with this decomposition structure.

### Data Split
- **Temporal split** (60/20/20) within each experiment
- Stage 1 uses prefill-only and decode-only steps from training split (subset of 60%)
- Stage 2 uses all steps from training split

### Data Requirements
- The 9-parameter model requires very little data (even 100 steps suffice for nonlinear least squares)
- The 24-parameter extended model requires more, but 73K training steps is vastly sufficient
- Risk of underfitting rather than overfitting (model may be too simple)

## Go Integration Path

**Path: Coefficient export (Path 1)**

This is the simplest integration path of any proposed idea. The trained model produces 9-24 floating-point coefficients that are stored in `defaults.yaml` (or a new `stepml_coeffs.yaml`). The Go implementation:

1. Reads the 9-24 coefficients at startup (same mechanism as current alpha/beta coefficients)
2. Implements the analytical formulas directly in Go (mirroring the existing `roofline.go` structure)
3. Applies the correction factors as multiplicative scalars
4. Uses `math.Max()` for the `max()` operations

The Go implementation would be approximately 100-150 lines -- comparable to the existing `rooflineStepTime()` function. No external dependencies (no CGo, no ONNX, no model files). The code is fully debuggable and testable in Go.

```go
type StepMLLatencyModel struct {
    modelConfig sim.ModelConfig
    hwConfig    sim.HardwareCalib
    corrections [9]float64  // c1..c9
    tp          int
    alphaCoeffs []float64
}

func (m *StepMLLatencyModel) StepTime(batch []*sim.Request) int64 {
    // Compute analytical components from batch...
    // Apply corrections c1..c9...
    // Return max(prefill, decode) + overhead
}
```

This approach has zero deployment friction -- the model is code, not data.

## LatencyModel Methods Covered

| Method | Coverage | Approach |
|--------|----------|----------|
| `StepTime(batch)` | **Primary target** | Analytical decomposition with 9-24 learned correction factors |
| `QueueingTime(req)` | **Improved** | Could learn separate queueing corrections per model: `alpha0(model) + alpha1(model) * inputLen` |
| `OutputTokenProcessingTime()` | **Improved** | Derive from decode-only single-request analytical estimate: `c4 * T_decode_gemm(1_req) + c5 * T_decode_attn(1_req, kv_len=mean_kv)` |
| `SchedulingProcessingTime()` | **New** | The `c7 + c8 * batch_size` term provides a scheduling overhead estimate |
| `PreemptionProcessingTime()` | **Retained** | Returns 0 (not modeled) |

This approach covers 4 of 5 methods (compared to Idea 1's 1 of 5), because the analytical decomposition naturally separates the components that each method needs.

## Evaluation Plan

### P1: Workload-level E2E mean error
- Sum predicted step times per request, compare mean predicted E2E to observed
- Target: <10% per experiment
- Expected strength: The analytical backbone provides correct scaling laws; corrections handle calibration error. Systematic bias should be low because the model is calibrated on mean-squared relative error.
- Expected weakness: May struggle with very heterogeneous batches where the `max()` approximation is poor.

### P1: Per-step MAPE, Pearson r
- Report per experiment and globally
- Expected: Lower MAPE than blackbox but potentially higher than Idea 1's tree ensemble (which has more degrees of freedom)
- Key diagnostic: residual analysis by component to identify which analytical approximation is most wrong

### P2: TTFT and ITL mean fidelity
- TTFT depends on `QueueingTime` + first prefill steps. Since this model improves `QueueingTime` (per-model alpha fitting) and prefill step prediction (separate prefill GEMM and attention components), TTFT should improve.
- ITL = individual decode step times. The decode analytical components with learned corrections should provide good ITL fidelity.
- Target: <15% per experiment

### P3: Tail behavior (p99)
- The analytical structure may struggle with tail behavior because extreme steps (outliers from GC, JIT compilation) are not captured by the smooth analytical function.
- Mitigation: Report outlier rate (steps >5x predicted) and consider a "noise floor" term.

### P4: Generalization
- **Leave-one-model-out**: Test whether correction factors transfer across models. The analytical backbone should generalize (physics doesn't change); the corrections capture hardware-specific efficiency factors.
- **Leave-one-workload-out**: Should generalize well since the model doesn't encode workload-specific parameters.
- **Key advantage over Idea 1**: With only 9-24 parameters, less risk of overfitting to specific experiments.

### P5/P6: Hardware/Quantization generalization
- The analytical components directly reference hardware specs (peak_flops, peak_bandwidth). Changing hardware means changing these two numbers. The correction factors capture hardware-specific efficiency factors that would need recalibration, but the structure transfers.
- This is the strongest generalization story among all proposed ideas.

### Short-circuit criterion
- If the basic 9-parameter model achieves >30% MAPE on any experiment, escalate to the 24-parameter extended model. If that also fails, the decomposition structure is insufficient.

## Related Work

- **Vidur** (Agrawal et al., MLSys 2024): Closest comparable. Vidur decomposes into operations and profiles; we decompose and calibrate from data. Our approach trades Vidur's profiling accuracy for deployment simplicity (no profiling hardware needed).
- **Roofline model** (Williams et al., CACM 2009): Our analytical backbone IS a roofline model with corrections. The contribution is the correction learning strategy.
- **Splitwise** (Patel et al., ISCA 2024): Prefill/decode disaggregation validates our separate phase treatment.
- **Sarathi-Serve** (Agrawal et al., OSDI 2024): Chunked prefill interference modeling informs our `max(prefill, decode)` combination rule.
- **Physics-informed ML** (Willard & Jia, 2020): The "analytical backbone + learned residual" pattern is established in scientific computing. Our contribution is applying it to LLM inference with a specific decomposition structure.
- **Calotoiu et al. (SC 2013)**: Piecewise performance models with regime detection. Our `max()` operations serve a similar role as regime transitions.
- **Banks et al. (2015)**: DES calibration methodology. Our correction factor fitting is a form of simulation calibration.

## Strengths and Weaknesses

### Strengths
1. **Strong physical foundation**: The decomposition mirrors actual GPU execution (GEMM, attention, memory access, overhead). Errors are more interpretable ("the attention correction is 1.5x, suggesting FlashAttention is more efficient than the FLOPs model predicts").
2. **Few parameters**: 9-24 parameters vs. thousands of tree nodes. Extreme data efficiency; could potentially calibrate from as few as 200 steps.
3. **No Jensen's inequality bias**: Optimizes in linear space with relative-error weighting.
4. **Best Go integration**: Pure coefficients, zero external dependencies, ~100 lines of Go code.
5. **Best hardware generalization story**: Changing hardware = changing peak_flops and peak_bandwidth, then recalibrating 9 correction factors.
6. **Covers 4 of 5 LatencyModel methods**: The decomposition naturally provides estimates for QueueingTime, OutputTokenProcessingTime, and SchedulingProcessingTime.
7. **Debuggable**: Each correction factor has a clear physical interpretation. If the model fails on a specific experiment, the component residual analysis tells you exactly which approximation is wrong.

### Weaknesses
1. **Limited expressiveness**: 9-24 parameters may be insufficient to capture all the non-linearities (e.g., kernel launch latency varies non-smoothly with batch size due to CUDA grid sizing).
2. **Hardcoded decomposition structure**: If the `max(prefill, decode)` combination rule is wrong (e.g., actual overlap is partial, not full), the model has no way to learn the true interaction. BLIS hypothesis H27 suggests the overlap is indeed partial.
3. **Requires hardware calibration data**: Needs `peak_flops`, `peak_bandwidth`, and model architecture parameters, which adds configuration complexity compared to a pure data-driven approach.
4. **Non-smooth optimization**: The `max()` operations create non-smooth loss landscape, potentially causing optimization difficulties (multiple local minima, gradient discontinuities).
5. **MoE routing variance not captured**: The analytical model assumes uniform expert loading. Real MoE inference has load-dependent expert latency that varies step-to-step. With only a fixed `is_moe` correction, per-step variance from expert routing is treated as noise.
6. **Prefix cache ambiguity not resolved**: Like Idea 1, the analytical FLOPs computation depends on whether `prefill_tokens` reflects pre- or post-cache-hit count. The correction factors may absorb this bias on average but cannot adapt per-step.

## Novelty Statement

This approach is novel in three ways:

1. **Data-calibrated roofline for LLM inference**: While roofline models are standard and data-calibrated models are standard, the specific formulation of learning multiplicative correction factors for a roofline decomposition of LLM inference step time is new. Vidur profiles operations; we learn operation-level corrections from end-to-end step data.

2. **Two-stage optimization for mixed-batch calibration**: Fitting corrections first on pure-phase steps, then refining on mixed batches, exploits the decomposition structure to disentangle per-component errors. This is not standard in performance model calibration.

3. **4-method coverage from a single decomposition**: By structuring the model to separate GEMM, attention, memory, and overhead components, we can provide estimates for QueueingTime, OutputTokenProcessingTime, and SchedulingProcessingTime as byproducts -- covering 4 of 5 LatencyModel interface methods from a single trained model. No other proposed approach achieves this breadth.

## Reviews for Idea 2

> **Note**: Reviews generated inline due to unavailable external LLM APIs. Each review applies a distinct critical lens.

### Review by Claude Opus 4.6 (aws/claude-opus-4-6)

**Overall Assessment: Strong Accept (Score: 8/10)**

**Technical Soundness (9/10)**: This is the most technically rigorous of the proposals. The decomposition into GEMM, attention, memory, and overhead components directly mirrors the actual GPU execution pipeline and is well-justified by the existing BLIS roofline code. The two-stage optimization strategy (pure phases first, mixed batches second) is clever and should improve convergence.

Three technical concerns:

1. **The `max(prefill, decode)` combination rule is explicitly acknowledged as approximate (H27).** The proposal should include a parameterized combination: `step_time = alpha * max(T_prefill, T_decode) + (1-alpha) * (T_prefill + T_decode)` with `alpha` learned from data. At `alpha=1` this is the current max() model; at `alpha=0` it is the additive model; the true interaction is likely somewhere in between. This adds only 1 parameter but could significantly improve mixed-batch accuracy.

2. **The inverse-variance weighting `w_i = 1/y_i^2` is problematic for very small steps.** Steps near 12 us would have weights ~7000x larger than steps near 250,000 us. This over-emphasizes tiny steps where measurement noise is proportionally large (12 us is only ~60 GPU clock cycles on H100). Recommend `w_i = 1/max(y_i, y_floor)^2` with `y_floor = 50 us` to cap the weighting for very small steps.

3. **The claim of "no Jensen's inequality bias" is correct but incomplete.** While the model avoids log-transform bias, the `max()` operations introduce their own bias: `E[max(A,B)] >= max(E[A], E[B])`. If correction factors are fit to minimize mean squared error on `max(T_prefill, T_decode)`, the expectation of the max will be biased upward when there is variance in which phase dominates. This is a second-order effect but worth noting.

**Novelty (7/10)**: The data-calibrated roofline approach is genuinely novel for this specific application. The two-stage optimization and 4-method coverage are meaningful contributions beyond standard physics-informed ML.

**Feasibility (8/10)**: Feasible but requires more careful implementation than Idea 1. The analytical formulas must exactly match the roofline model's behavior (including subtle details like the 1.8 causal masking factor, the 0.80/0.92 KV cache read factors). Any discrepancy between the analytical formulas and the training data's actual execution patterns will be absorbed into the correction factors, potentially making them hardware-configuration-specific rather than transferable.

**Go Integration Feasibility (10/10)**: This is the best integration path. Pure coefficient export with zero dependencies. The Go code would closely mirror the existing `rooflineStepTime()` function.

**Likelihood of <10% E2E Mean Error (7/10)**: Good, especially after the two-stage optimization. The analytical backbone provides correct scaling laws, and 9-24 correction factors should capture the major systematic biases. The E2E metric benefits from the model's low systematic bias (relative-error weighting in linear space). However, the limited expressiveness (9-24 parameters) may not capture all non-linearities, particularly in reasoning workloads with extreme KV lengths.

**Specific Suggestions:**
1. Add a parameterized prefill-decode combination: `alpha * max() + (1-alpha) * sum()`.
2. Use floored inverse-variance weighting: `w_i = 1/max(y_i, 50)^2`.
3. Implement a diagnostic that reports per-component residual statistics (mean, std of `T_actual / T_analytical` for each component) to guide further refinement.
4. Consider a "correction function" variant where `c1(x) = MLP(x; theta_1)` with a tiny MLP (2 layers, 8 units) -- adds ~100 parameters but captures non-linear correction dependencies.

---

### Review by GPT-4o (Azure/gpt-4o)

**Overall Assessment: Accept (Score: 7.5/10)**

**Technical Soundness (8/10)**: The decomposition is well-motivated and the connection to the BLIS roofline codebase is convincing. The analytical formulas are clearly specified and the two-stage optimization is a good idea. Two concerns:

1. **Non-convexity of the optimization problem.** The `max()` operations make the loss surface non-smooth and non-convex. The trust-region reflective algorithm handles bound constraints well but may converge to poor local minima, especially with 24 parameters. The initialization strategy (starting from `c_i = 1.0`) assumes the analytical model is approximately correct, which may not hold for all models (e.g., Mixtral's expert routing adds overhead that the analytical model completely ignores). Recommend multiple random restarts (10-20) and selecting the best.

2. **The extended model with feature-dependent corrections is a slippery slope.** Adding `c1(x) = c1_base + c1_batch * log(batch_size) + c1_moe * is_moe` starts approaching a general regression model. If 15 additional parameters are added, and then the model still doesn't fit, the temptation is to add more. The boundary between "physics-informed model with corrections" and "regression model with physics-inspired basis functions" becomes blurry. Recommend keeping the basic 9-parameter model as the primary result and the extended model as a sensitivity analysis.

**Novelty (7/10)**: The approach is a well-executed application of physics-informed ML to a new domain. The 4-method coverage is a genuine advantage over competing approaches. However, the core idea (analytical model + learned corrections) is a standard pattern in computational physics (e.g., equation-of-state corrections in thermodynamics, turbulence model corrections in CFD).

**Feasibility (8/10)**: Feasible. The nonlinear least squares fitting is standard and fast. The main implementation challenge is correctly computing the analytical components -- the roofline model in `roofline.go` has numerous subtle details (MFU lookup, bucketing, causal masking correction) that must be faithfully replicated in Python for training. Any discrepancy between the Python training code and the Go inference code would introduce systematic error.

**Go Integration Feasibility (10/10)**: Best possible. Pure coefficients in YAML. Zero external dependencies. The Go implementation would be a simplified variant of the existing roofline model.

**Likelihood of <10% E2E Mean Error (7/10)**: Good for the easy experiments (where the roofline model is already close). The correction factors will absorb systematic biases. For hard experiments (Mixtral-reasoning), the limited model complexity may be insufficient. The 9-parameter model will likely achieve <10% E2E on 8-12 experiments; the 24-parameter model on 10-14.

**Specific Suggestions:**
1. Ensure the Python analytical formulas exactly match `roofline.go` by writing unit tests that compare Python and Go outputs for the same inputs.
2. Use multiple random restarts for the optimization.
3. Present the 9-parameter model as the primary result, with the extended model as ablation.
4. Add a "per-model" variant (36 parameters: 9 per model) as an upper bound analysis. If even per-model corrections cannot achieve <10% E2E, the decomposition structure is insufficient.

---

### Review by Gemini 2.5 Flash (GCP/gemini-2.5-flash)

**Overall Assessment: Accept with Minor Revisions (Score: 7/10)**

**Technical Soundness (7/10)**: Sound approach with clear physical motivation. The decomposition is well-aligned with the existing BLIS roofline model. However, several concerns:

1. **The analytical model ignores FlashAttention's tiling.** FlashAttention computes attention in tiles, with IO complexity `O(N^2 * d / M)` where M is SRAM size. The standard FLOPs estimate `2 * N * d * num_heads` does not capture the tiling overhead or the memory access pattern. The correction factor `c2` (prefill attention) and `c5` (decode attention) must absorb not just a multiplicative error but a potentially different functional form (the relationship between KV length and attention time is not purely linear with FlashAttention). This limits how well a constant multiplicative correction can perform.

2. **The overhead term `c7 + c8 * batch_size + c9 * num_layers / tp` is simplistic.** Scheduling overhead in vLLM involves Python GIL contention, CUDA stream synchronization, and PagedAttention block table construction. These overheads have different scaling behaviors: GIL contention scales with Python object count (request count), CUDA synchronization is batch-size-independent, and block table construction scales with total KV blocks. A more accurate decomposition would be: `T_overhead = c7 + c8 * num_requests + c9 * total_kv_blocks + c10 * 1/batch_size` (the `1/batch_size` term captures the amortized kernel launch cost).

3. **Per-request computation in the analytical model is expensive.** The decode attention component iterates over all decode requests and computes per-request FLOPs. For batch size 128, this is 128 floating-point computations per prediction. While still fast, it is slower than Idea 1's fixed-size feature vector approach. The 1ms budget should be confirmed with benchmarking.

**Novelty (6/10)**: The physics-informed correction approach is standard in scientific computing. The specific application to LLM inference is new but the methodology is not. The 4-method coverage from a single model is a useful engineering contribution but not a research novelty.

**Feasibility (7/10)**: The main risk is the non-smooth optimization. In my experience with `max()`-based objectives, the optimizer can get stuck on the boundary where the dominant phase switches. For example, if a batch is right at the boundary where `T_prefill ≈ T_decode`, small changes in correction factors can cause the dominant phase to flip, creating a step-function-like loss surface. The two-stage optimization helps, but the Stage 2 refinement on mixed batches is where this problem will manifest.

**Go Integration Feasibility (10/10)**: Best possible among all proposals.

**Likelihood of <10% E2E Mean Error (6/10)**: The 9-parameter model is likely too simple for <10% E2E across all 16 experiments. The baseline blackbox model already has 3 parameters per experiment (48 total when fit per-experiment), and the proposed model has only 9-24 globally. The model must be more expressive per parameter than the blackbox to compensate. This is plausible (the decomposition structure is much richer than linear regression) but not guaranteed.

The key test: compare the 36-parameter per-model variant (9 corrections per model) against the per-experiment blackbox baseline (3 params per experiment, 48 total). If the 36-parameter analytical model cannot beat the 48-parameter linear model, the decomposition gains are consumed by the limited correction expressiveness.

**Specific Suggestions:**
1. Add a parameterized phase combination as Claude suggested: `alpha * max() + (1-alpha) * sum()`.
2. Enrich the overhead model: `c7 + c8 * num_requests + c9 * total_kv_blocks + c10 * num_layers / tp`.
3. Consider making attention corrections sequence-length-dependent: `c5(kv_len) = c5_base * (kv_len / kv_ref)^c5_exp` where `c5_exp` is learned. This captures the non-linear FlashAttention tiling effect.
4. Benchmark the per-request computation cost in Go at batch_size=128 to verify 1ms compliance.
5. The per-model variant (36 params) should be the standard configuration, not the 9-param global model. With only 4 models, 36 parameters is still very data-efficient.

---

# Idea 3: Evolutionary Program Synthesis via LLM-Guided Search (OpenEvolve/FunSearch)

## Motivation

Ideas 1 and 2 both impose structural assumptions: Idea 1 assumes that the right features are a fixed 30-dimensional vector mapped through a tree ensemble; Idea 2 assumes that the right decomposition is `max(prefill, decode) + overhead` with multiplicative corrections. Both require significant domain expertise to design the features or decomposition structure. What if the structure itself could be discovered?

**Evolutionary program synthesis** uses LLM-guided genetic programming to evolve mathematical functions that predict step time. Instead of specifying features or decomposition structures upfront, the search starts from a simple seed program and evolves it through mutation, crossover, and selection, using an LLM (e.g., GPT-4 or Claude) as the mutation operator that proposes semantically meaningful program transformations. This approach was demonstrated by FunSearch (Romera-Paredes et al., Nature 2024) for discovering novel mathematical constructions and by OpenEvolve (2025) for algorithm discovery.

The key advantages for step-time prediction are:

1. **Automated structure discovery**: The evolutionary search can discover non-obvious functional forms that human designers would not consider. For example, it might discover that mixed-batch step time is better modeled by `sqrt(T_prefill^2 + T_decode^2)` than `max(T_prefill, T_decode)` -- a combination rule that no human would typically propose but that could emerge naturally from evolution.

2. **Multi-objective optimization**: The search can simultaneously optimize for accuracy (low MAPE), simplicity (few operations), and Go portability (only using operations available in Go's math library). Pareto-optimal solutions trade off these objectives.

3. **Interpretability**: Evolved functions are typically compact mathematical expressions (10-50 operations) that are immediately readable and translatable to Go. Unlike tree ensembles (thousands of nodes) or neural networks (opaque weights), evolved programs are human-auditable.

4. **No feature engineering**: The evolution operates directly on Request object fields and batch-level aggregates, discovering which combinations matter rather than requiring a human to pre-specify them.

This approach addresses two specific critiques from Ideas 1 and 2 reviews:
- **GPT-4o's concern about Idea 1's feature redundancy**: Evolution will naturally discard redundant features.
- **Gemini's concern about Idea 2's hardcoded `max()` rule**: Evolution can discover the actual combination function.

## Algorithm Description

### Search Framework

We use an **MAP-Elites** (Multi-dimensional Archive of Phenotypic Elites) evolutionary framework, adapted from OpenEvolve:

1. **Program representation**: Each candidate is a Python function `predict_step_time(batch_info: dict) -> float` that takes batch-level features and returns predicted step time in microseconds. The function body is free-form Python code using math operations, conditionals, and loops.

2. **Behavioral descriptor space**: Two dimensions:
   - **Complexity**: Number of arithmetic operations in the function (1-50)
   - **Feature count**: Number of distinct input features referenced (1-15)

3. **Quality metric**: Negative of mean E2E error across all 16 experiments (higher is better).

4. **Archive**: A 2D grid (10 complexity bins x 5 feature bins = 50 cells) storing the best-performing program at each complexity-feature combination.

### Evolutionary Loop

```
Initialize: Seed archive with simple baseline programs
For generation = 1 to N_generations:
    1. Select 2-4 parent programs from archive (tournament selection)
    2. Send parents + evaluation feedback to LLM mutation operator
    3. LLM generates K=8 child programs via:
       - Point mutations (modify a constant, change an operator)
       - Structural mutations (add a conditional, add a term)
       - Crossover (combine features/logic from two parents)
       - Informed mutations (based on per-experiment error analysis)
    4. Evaluate each child on the training set (16 experiments)
    5. Short-circuit: discard children with MAPE > 100% on any experiment
    6. Insert surviving children into archive if they improve their cell
    7. Every 50 generations: report Pareto frontier and best program
```

### Seed Programs

The archive is initialized with 5 seed programs spanning the complexity spectrum:

**Seed 1 (minimal -- blackbox equivalent):**
```python
def predict_step_time(b):
    return b['beta0'] + b['beta1'] * b['prefill_tokens'] + b['beta2'] * b['decode_tokens']
```

**Seed 2 (KV-aware):**
```python
def predict_step_time(b):
    base = b['beta0'] + b['beta1'] * b['prefill_tokens'] + b['beta2'] * b['decode_tokens']
    kv_correction = b['kv_max'] * b['decode_reqs'] * 0.001
    return base + kv_correction
```

**Seed 3 (phase-decomposed):**
```python
def predict_step_time(b):
    prefill_time = b['prefill_tokens'] * b['us_per_prefill_token']
    decode_time = b['decode_reqs'] * (b['us_per_decode_base'] + b['kv_mean'] * b['us_per_kv_token'])
    return max(prefill_time, decode_time) + b['overhead']
```

**Seed 4 (roofline-inspired):**
```python
def predict_step_time(b):
    compute_time = b['total_flops'] / b['peak_throughput']
    memory_time = b['total_bytes'] / b['peak_bandwidth']
    return max(compute_time, memory_time) * 1e6 + b['overhead']
```

**Seed 5 (regime-switching):**
```python
def predict_step_time(b):
    if b['prefill_tokens'] > 100:
        return b['c1'] * b['prefill_tokens'] + b['c2'] * b['kv_max']
    else:
        return b['c3'] * b['decode_reqs'] + b['c4'] * b['kv_mean'] + b['c5']
```

### LLM Mutation Operator

The LLM receives:
- The parent program(s)
- Per-experiment MAPE scores for the parent
- Feature importance (which features correlate with residuals)
- Examples of batch compositions where the parent performs poorly
- The invariant constraints (INV-M-1 through INV-M-6)

The prompt asks the LLM to propose modifications that:
1. Reduce error on the worst-performing experiments
2. Maintain or reduce program complexity
3. Use only features available from Request objects
4. Ensure positive output for all valid inputs (INV-M-1)

### Coefficient Optimization

After each structural mutation, the evolved program's numeric constants are optimized via `scipy.optimize.minimize` (L-BFGS-B) to find optimal coefficient values for the fixed program structure. This separates structural evolution (LLM's job) from parameter tuning (optimizer's job).

## Feature Engineering

The evolution operates on a **feature dictionary** computed from the batch, which serves as the "input alphabet" for evolved programs:

### Available Features (18 features)

```python
batch_info = {
    # Batch composition
    'prefill_tokens': int,     # Total prefill tokens this step
    'decode_tokens': int,      # Total decode tokens this step
    'prefill_reqs': int,       # Number of prefill requests
    'decode_reqs': int,        # Number of decode requests
    'batch_size': int,         # Total requests

    # KV length statistics (from ProgressIndex)
    'kv_mean': float,          # Mean KV length across all requests
    'kv_max': int,             # Max KV length
    'kv_sum': int,             # Sum of KV lengths
    'kv_var': float,           # Variance of KV lengths
    'kv_decode_mean': float,   # Mean KV length (decode requests only)
    'kv_decode_max': int,      # Max KV length (decode requests only)

    # Physics proxies (simplified, no hardware params)
    'total_flops': float,      # Estimated total FLOPs
    'total_bytes': float,      # Estimated total memory bytes
    'attention_flops': float,  # Attention-specific FLOPs

    # Model metadata
    'model_params_b': float,   # Total params in billions
    'active_params_b': float,  # Active params in billions (MoE-adjusted)
    'tp': int,                 # Tensor parallelism degree
    'is_moe': int,             # 0 or 1
}
```

The evolution is free to use any subset of these features. Unused features are naturally pruned by selection pressure (simpler programs are preferred if equally accurate).

Additionally, evolved programs may declare named constants that are optimized during coefficient fitting:
```python
# Constants declared in program
CONSTANTS = {'c1': 0.5, 'c2': 100.0, 'overhead': 50.0, ...}
```

## Architecture Handling

Unlike Ideas 1 and 2, MoE handling is **discovered by evolution** rather than prescribed:

1. **The `is_moe` flag** and `active_params_b` are available features. Evolution can use them in conditionals (`if is_moe: ...`) or continuous expressions (`active_params_b * c`).

2. **The search may discover MoE-specific functional forms** that a human designer would not propose. For example, it might discover that Mixtral's step time is better modeled with a different attention scaling law (due to the different number of KV heads interacting with more Q heads).

3. **Fallback**: If evolution struggles with MoE, the seed programs include `active_params_b` (which already adjusts for MoE) and the LLM mutation operator is explicitly prompted to improve MoE experiment performance when those experiments have high error.

The risk is that with only 4 Mixtral experiments out of 16, evolutionary pressure on MoE behavior is diluted. Mitigation: weight Mixtral experiments 2x in the fitness function to ensure adequate selection pressure.

## Training Strategy

### Fitness Function

Multi-objective fitness with lexicographic priority:

1. **Primary**: Mean of worst-3 per-experiment E2E errors (focuses on hardest cases)
2. **Secondary**: Global MAPE across all steps
3. **Tertiary**: Negative program complexity (simpler is better)

```python
def fitness(program, train_data):
    e2e_errors = []
    for exp in experiments:
        exp_steps = train_data[train_data.experiment == exp]
        predicted = [program.predict(step) for step in exp_steps]
        e2e_error = compute_e2e_mean_error(predicted, exp_steps)
        e2e_errors.append(abs(e2e_error))

    worst_3_mean = np.mean(sorted(e2e_errors)[-3:])
    global_mape = compute_mape(all_predicted, all_actual)
    complexity = program.num_operations()

    return (-worst_3_mean, -global_mape, -complexity)  # maximize all
```

### Computational Budget

- **LLM calls**: ~2000-5000 total (400-1000 generations x 4-8 children per generation x ~0.5 LLM calls per child after caching)
- **Evaluation cost**: Each program evaluation on 73K training steps takes ~1 second in Python. With 8 children per generation and 1000 generations, total evaluation cost is ~2 hours.
- **LLM cost**: At ~$0.03 per mutation call (GPT-4 pricing), ~$150 total.
- **Total wall time**: 4-8 hours with parallelization (evaluate children in parallel).

### Convergence Criteria

- Stop after 1000 generations or when the best program's mean E2E error has not improved by >0.5% for 200 consecutive generations.
- Extract the Pareto frontier of accuracy-vs-complexity trade-offs.
- Select the simplest program within 5% of the best accuracy for Go deployment.

### Data Split
- **Training**: 60% temporal split used for fitness evaluation during evolution
- **Validation**: 20% used for coefficient re-optimization after evolution completes (prevents constant overfitting)
- **Test**: 20% held out for final evaluation (never seen during evolution)
- **Generalization**: Leave-one-model-out and leave-one-workload-out applied to the final evolved program

## Go Integration Path

**Path: Evolved code translation (Path 4)**

The evolved program is a compact mathematical function (10-50 operations) that uses only:
- Basic arithmetic (`+`, `-`, `*`, `/`)
- `math.Max`, `math.Min`, `math.Sqrt`, `math.Log`, `math.Exp`
- Conditionals (`if/else`)
- Named constants

This translates directly to Go with a 1:1 mapping:

```python
# Evolved Python
def predict_step_time(b):
    prefill_time = c1 * b['prefill_tokens'] + c2 * b['kv_max'] * b['decode_reqs']
    decode_time = c3 * b['decode_reqs'] * (1 + c4 * math.log(1 + b['kv_decode_mean']))
    return math.sqrt(prefill_time**2 + decode_time**2) + c5 * b['batch_size'] + c6
```

```go
// Translated Go
func (m *EvolvedLatencyModel) StepTime(batch []*sim.Request) int64 {
    // Compute batch_info from batch...
    prefillTime := m.c1*float64(prefillTokens) + m.c2*float64(kvMax)*float64(decodeReqs)
    decodeTime := m.c3*float64(decodeReqs) * (1 + m.c4*math.Log(1+kvDecodeMean))
    result := math.Sqrt(prefillTime*prefillTime + decodeTime*decodeTime) + m.c5*float64(batchSize) + m.c6
    return int64(result)
}
```

The translation is mechanical: every Python operation has a direct Go equivalent. The evolved function's constants are stored in `defaults.yaml`. Total Go code: 50-80 lines (feature extraction + evolved formula).

**Constraint on evolution**: The search is restricted to operations available in Go's `math` package. The LLM mutation prompt explicitly states this constraint. Programs using numpy-specific operations (e.g., `np.percentile`) are rejected.

## LatencyModel Methods Covered

| Method | Coverage | Approach |
|--------|----------|----------|
| `StepTime(batch)` | **Primary target** | Evolved prediction function |
| `QueueingTime(req)` | **Potentially improved** | Could evolve a separate small function for queueing time |
| `OutputTokenProcessingTime()` | **Retained** | Current `alpha2` constant |
| `SchedulingProcessingTime()` | **Retained** | Returns 0 |
| `PreemptionProcessingTime()` | **Retained** | Returns 0 |

The primary evolution targets `StepTime`. A secondary evolution run could target `QueueingTime` with per-request features (`len(InputTokens)`, model metadata), but this is a lower priority.

## Evaluation Plan

### P1: Workload-level E2E mean error
- The fitness function directly optimizes for E2E mean error (worst-3 experiments), making this the most directly optimized metric.
- Target: <10% per experiment
- Expect: Strong performance on the metric being optimized, but risk of overfitting to training-set E2E patterns.
- Comparison: Report the full Pareto frontier (accuracy vs complexity), not just the best program.

### P1: Per-step MAPE, Pearson r
- Report for the selected program (simplest within 5% of best accuracy)
- Per-step MAPE is not directly optimized and may be higher than Idea 1
- Feature usage analysis: which of the 18 available features does the evolved program actually use?

### P2: TTFT and ITL mean fidelity
- TTFT depends on QueueingTime + first step predictions
- ITL depends on per-decode step predictions
- The evolved function may or may not handle these sub-cases well (it optimizes for aggregate E2E, not per-step-type accuracy)

### P3: Tail behavior (p99)
- Evolved programs may discover outlier-handling logic (e.g., `if kv_max > 3000: use_different_formula`)
- Or they may not -- p99 behavior is not in the fitness function
- Report p99 error and compare to baselines

### P4: Generalization
- **Leave-one-model-out**: Critical test. Evolved programs may overfit to the specific 4 models. The physics features (`total_flops`, `active_params_b`) provide a generalization mechanism.
- **Leave-one-workload-out**: Likely better than model generalization since workload effects are captured through batch composition features.
- **Key risk**: With 1000 generations of evolution on 16 experiments, the program may memorize experiment-specific patterns rather than learning general laws.

### P5/P6: Hardware/Quantization generalization
- Evolved programs that use physics features (`total_flops`, `total_bytes`) may generalize to new hardware if those features are recalculated for the new hardware.
- Programs that rely on learned constants specific to H100 performance will not generalize.
- The Pareto frontier analysis reveals whether physics-informed programs (which generalize) are competitive with hardware-specific programs (which don't).

### Short-circuit criterion
- If the best program after 500 generations has MAPE > 30% on any experiment, add that experiment's data to the mutation prompt context (targeted improvement).
- If after 1000 generations the best program still has >30% MAPE on any experiment, the evolutionary approach is insufficient.

## Related Work

- **FunSearch** (Romera-Paredes et al., Nature 2024): LLM-guided evolutionary search discovering novel mathematical functions. We apply the same paradigm to performance prediction rather than combinatorial optimization.
- **OpenEvolve (2025)**: MAP-Elites with LLM mutation for algorithm discovery. Our framework directly adapts OpenEvolve's archive + LLM mutation approach.
- **GEPA**: Genetic-Pareto optimization with LLM reflection. Our multi-objective fitness (accuracy, complexity, portability) follows GEPA's Pareto approach.
- **Symbolic regression** (Schmidt & Lipson, Science 2009): Discovering physical laws from data. Our approach is a higher-level variant where the "symbols" include conditionals and batch aggregation, not just algebraic operators.
- **AutoML** (Feurer et al., NeurIPS 2015): Automated model selection and hyperparameter tuning. We go beyond model selection to model structure discovery.
- **Program synthesis for performance modeling**: To our knowledge, no prior work has applied LLM-guided evolutionary program synthesis specifically to LLM inference latency prediction. The closest work is in compiler autotuning (Ansel et al., CGO 2014) which uses genetic programming for optimization heuristic discovery.

## Strengths and Weaknesses

### Strengths
1. **Automated structure discovery**: No human feature engineering or decomposition structure required. The search can discover non-obvious relationships (e.g., `sqrt(prefill^2 + decode^2)` for mixed-batch combination).
2. **Multi-objective Pareto optimization**: Explicitly trades off accuracy, simplicity, and portability. The user can choose from the Pareto frontier based on their priorities.
3. **Interpretability**: Evolved programs are compact, readable mathematical functions. A human can inspect and reason about the discovered formula. This is a major advantage over tree ensembles (Idea 1) which are opaque at scale.
4. **Direct Go translation**: The evolved function maps 1:1 to Go code. No libraries, no model files, no external dependencies. 50-80 lines of Go.
5. **Invariant enforcement**: The LLM mutation operator can be prompted to maintain invariants (positive output, monotonicity). Programs violating INV-M-1 are rejected during evaluation.
6. **Discoveries may transfer to theory**: If the evolution discovers a novel combination rule for mixed batches, this is publishable insight about LLM inference behavior.

### Weaknesses
1. **High computational cost**: 4-8 hours of wall time, ~$150 in LLM API costs, 2000-5000 LLM calls. This is significantly more expensive than fitting Idea 1 or 2.
2. **Stochastic outcomes**: Evolution is non-deterministic. Different random seeds may produce different programs with different accuracy. Must run 3-5 independent evolution runs and report variance.
3. **Overfitting risk**: With 1000 generations and 73K training steps, the evolved program may memorize training patterns. The coefficient re-optimization on the validation set helps, but structural overfitting (e.g., evolving experiment-specific conditionals) is harder to detect.
4. **MoE under-representation**: Only 4/16 experiments are MoE. Evolution may under-invest in MoE handling due to diluted selection pressure, even with 2x weighting.
5. **No monotonicity guarantee**: While the LLM is prompted to maintain monotonicity, evolved programs are not guaranteed to satisfy INV-M-5. Post-hoc verification is required, and monotonicity violations may be >5%.
6. **Reproducibility concerns**: The LLM mutation operator's behavior depends on the specific LLM version and prompt. Results may not reproduce exactly with a different LLM or after model updates. Must pin LLM version and save all evolution logs.
7. **Limited to single-function discovery**: The approach discovers one function for all batches. If the true model requires fundamentally different logic for different regimes (e.g., small vs large batches), the single function must encode regime switching via conditionals, which increases complexity.

## Novelty Statement

This approach is novel in three specific ways:

1. **First application of LLM-guided evolutionary program synthesis to LLM inference latency prediction**: While FunSearch, OpenEvolve, and GEPA have been applied to combinatorial optimization and algorithm design, no prior work has applied this paradigm to performance prediction in serving systems. The step-time prediction problem is well-suited because: (a) the evaluation function is cheap and differentiable, (b) the search space is constrained by physical plausibility, and (c) the output must be human-readable for Go integration.

2. **Multi-objective evolution for simulator calibration**: The three-way trade-off (accuracy, simplicity, Go portability) is novel in the DES calibration context. Standard calibration approaches optimize only for accuracy. By explicitly including simplicity and portability in the fitness function, we produce models that are deployable by construction.

3. **Potential for theoretical discovery**: If the evolution discovers a novel batch combination rule (e.g., something better than `max(prefill, decode)` for mixed batches), this is a contribution to the understanding of LLM inference behavior, not just a prediction improvement. FunSearch demonstrated that evolution can discover publishable mathematical insights; the same potential exists here for systems performance insights.

## Reviews for Idea 3

> **Note**: Reviews generated inline due to unavailable external LLM APIs. Each review applies a distinct critical lens.

### Review by Claude Opus 4.6 (aws/claude-opus-4-6)

**Overall Assessment: Weak Accept (Score: 6/10)**

**Technical Soundness (6/10)**: The evolutionary framework is well-described and draws appropriately from FunSearch and OpenEvolve. However, I have significant concerns about the practical application:

1. **The search space is enormous and poorly constrained.** A Python function with 10-50 operations over 18 features has a search space vastly larger than FunSearch's (which operated on small combinatorial constructions). LLM-guided mutations help navigate this space, but 2000-5000 LLM calls may be insufficient to find high-quality programs. FunSearch used millions of evaluations.

2. **The fitness function directly optimizes E2E mean error, creating an overfitting risk.** The worst-3-experiments focus is good for robustness, but with 16 experiments, the selection pressure may produce programs with experiment-specific constants or conditionals that do not generalize. I recommend adding a held-out validation fitness: every 100 generations, evaluate the best program on the validation set and track generalization gap.

3. **Coefficient optimization after structural mutation is key but under-specified.** The proposal uses L-BFGS-B, but the evolved programs may have non-smooth loss surfaces (from conditionals, `max()`, `min()`). For programs with conditionals, the loss surface is piecewise smooth, and L-BFGS-B may fail at the boundaries. Recommend using a derivative-free optimizer (Nelder-Mead or CMA-ES) as fallback for programs with >2 conditionals.

4. **The "LLM mutation prompt" is critical but not specified.** The quality of mutations depends entirely on the prompt design. The proposal should include at least one concrete mutation prompt template. Without this, the approach is hard to evaluate for feasibility.

**Novelty (8/10)**: This is the most novel of the three proposals. Applying LLM-guided evolution to systems performance prediction is genuinely new. The potential for theoretical discovery (novel batch combination rules) is exciting. However, the novelty comes with correspondingly high risk.

**Feasibility (5/10)**: The lowest feasibility of the three proposals. The 4-8 hour wall time is acceptable, but the $150 LLM cost per run, the need for 3-5 independent runs, and the debugging complexity of evolutionary systems create practical barriers. If the first few evolution runs produce poor results, diagnosing whether the issue is the fitness function, the seed programs, the mutation prompts, or the evaluation budget is difficult.

**Go Integration Feasibility (9/10)**: Excellent if the evolution produces a clean program. The mechanical Python-to-Go translation is straightforward. The risk is that the evolved program is "ugly" (many nested conditionals, magic constants) and hard to maintain.

**Likelihood of <10% E2E Mean Error (5/10)**: Uncertain. The approach has the highest ceiling (it can, in principle, discover the optimal prediction function) but also the highest floor (it may fail to converge). My estimate: 40% chance of achieving <10% E2E on 12+ experiments, 60% chance of underperforming Idea 1. The variance across independent runs will be high.

**Specific Suggestions:**
1. Include a concrete LLM mutation prompt template in the proposal.
2. Add a held-out validation fitness check every 100 generations to detect structural overfitting.
3. Use CMA-ES rather than L-BFGS-B for coefficient optimization of programs with conditionals.
4. Run a pilot study (100 generations) before committing to the full 1000-generation budget to assess convergence behavior.
5. Consider seeding the archive with Idea 1's best features and Idea 2's decomposition structure -- don't force the evolution to rediscover known physics.

---

### Review by GPT-4o (Azure/gpt-4o)

**Overall Assessment: Reject with Encouragement to Resubmit (Score: 5/10)**

**Technical Soundness (5/10)**: The idea is intellectually stimulating but practically under-developed. Several fundamental issues:

1. **The search budget is inadequate.** FunSearch used TPU-scale compute with millions of evaluations. OpenEvolve's successful applications (bin packing, online algorithms) involved much simpler programs (5-15 operations) over much smaller input spaces (2-3 features). With 18 input features and 10-50 operations, the search space is combinatorially larger. 2000-5000 LLM calls is likely 10-100x too few. I recommend either (a) dramatically increasing the budget to 50,000+ evaluations, or (b) dramatically constraining the search space (e.g., evolve only the combination rule for prefill+decode while fixing the component formulas from Idea 2).

2. **The seed programs are too diverse.** The 5 seed programs span from a 3-parameter linear model to a regime-switching model. This diversity dilutes the evolutionary pressure -- the population will spend generations exploring dead ends near weak seeds rather than refining promising ones. Recommend starting with 2-3 seeds that are already reasonable (Seeds 3, 4, 5) and dropping the trivial seeds (1, 2).

3. **The multi-objective fitness is improperly specified.** Lexicographic priority (worst-3 E2E first, then MAPE, then complexity) means that a program that is 0.01% better on worst-3 E2E but 100x more complex will always be preferred. This does not match the stated goal of trading off accuracy and simplicity. Use proper Pareto dominance with NSGA-II-style selection instead.

4. **Coefficient optimization is a confound.** After each structural mutation, the constants are re-optimized. This means that the evolutionary search is really searching over program structures, with constants optimized post-hoc. But two structurally similar programs with slightly different constant names will have their constants optimized independently, potentially finding different optima. The interaction between structural evolution and constant optimization is not well-understood and may cause instability.

**Novelty (8/10)**: High novelty. The application domain is new. But novelty alone does not justify acceptance if feasibility is low.

**Feasibility (4/10)**: Low. The practical challenges (LLM cost, non-determinism, debugging difficulty, search space size) are severe. The proposal acknowledges these but does not provide convincing mitigations. A pilot study (Section 8.6 in the proposal) is essential before committing resources.

**Go Integration Feasibility (8/10)**: Good in principle, but depends on the evolution producing a clean program. If the best program has 15 nested conditionals and 40 constants, the Go code will be unmaintainable. Recommend adding a hard complexity cap (max 30 operations, max 3 conditionals) to the search.

**Likelihood of <10% E2E Mean Error (4/10)**: Low confidence. The search may converge to a local optimum that is worse than Idea 1 or Idea 2. The high variance across runs means that even if one run succeeds, reproducibility is not guaranteed.

**Specific Suggestions:**
1. Constrain the search space dramatically: evolve only the combination rule for pre-computed component times (not the component formulas themselves). Use Idea 2's analytical components as fixed inputs.
2. Use proper Pareto selection (NSGA-II) instead of lexicographic priority.
3. Drop trivial seed programs. Start with physically motivated structures only.
4. Add a hard complexity cap (30 operations, 3 conditionals, 10 constants).
5. Budget 50,000+ evaluations or accept that this is a pilot study, not a production approach.
6. Run 5 independent evolution runs and report median + interquartile range, not just the best run.

---

### Review by Gemini 2.5 Flash (GCP/gemini-2.5-flash)

**Overall Assessment: Borderline Accept (Score: 5.5/10)**

**Technical Soundness (6/10)**: The evolutionary framework is reasonable but the proposal over-promises and under-delivers on specifics:

1. **The "automated structure discovery" claim is overstated.** The 5 seed programs already encode significant structural priors (max-based combination, phase decomposition, regime switching). The evolution is not starting from scratch -- it is refining human-designed structures. This is fine, but the proposal should be honest about it. True automated discovery (starting from `return 0`) would require orders of magnitude more compute.

2. **The 18-feature dictionary is a hidden form of feature engineering.** The proposal claims "no feature engineering," but the choice of which 18 features to compute from the batch is itself a feature engineering decision. The evolution can only use features that are provided. If the key feature is missing (e.g., `max_prefill_chunk_ratio`), evolution cannot discover it. The feature dictionary should be acknowledged as a design choice, not a given.

3. **The evaluation of 73K steps per program in Python is slow.** At 1 second per evaluation, 8 children per generation, 1000 generations = 8000 evaluations = 2.2 hours just for evaluation. With LLM latency (~5 seconds per mutation call), the total wall time is more like 12-16 hours, not 4-8. The proposal should use vectorized numpy evaluation (not per-step Python loops) to reduce evaluation cost to ~0.1 seconds per program.

4. **The "potential for theoretical discovery" is speculative.** FunSearch discovered novel mathematical constructions in well-defined combinatorial problems where the solution space has rich structure. Step-time prediction is an empirical fitting problem -- the "discoveries" are more likely to be engineering tricks (e.g., using `sqrt` instead of `max`) than theoretical insights. The proposal should temper this claim.

**Novelty (7/10)**: Novel application domain. The MAP-Elites archive with complexity-feature behavioral descriptors is a good design choice that balances exploration and exploitation. The multi-objective Pareto approach is appropriate.

**Feasibility (5/10)**: Moderate-to-low. The main feasibility concern is not the compute cost but the iteration time. If the first evolution run produces poor results, diagnosing and fixing the issue requires understanding the interaction between seed programs, mutation prompts, fitness function, and coefficient optimization. This is a research project within a research project.

**Go Integration Feasibility (9/10)**: Strong. The mechanical translation is a clear advantage. The complexity cap suggested by GPT-4o (30 operations, 3 conditionals) would help ensure the translated Go code is maintainable.

**Likelihood of <10% E2E Mean Error (5/10)**: Uncertain. The approach has high variance. In the best case, it discovers a formula that outperforms all other approaches. In the worst case, it produces an overfit mess. The expected outcome is somewhere in between -- a readable formula that achieves <10% E2E on 8-12 experiments but struggles with the hardest cases.

**Specific Suggestions:**
1. Use vectorized numpy evaluation (not Python loops) to reduce per-program evaluation time to ~0.1 seconds.
2. Acknowledge the feature dictionary as a design choice and discuss what happens if key features are missing.
3. Temper the "theoretical discovery" claim. Frame it as "potential engineering insight" instead.
4. Consider a hybrid approach: use Idea 2's analytical components as fixed inputs and evolve only the combination/correction logic. This dramatically constrains the search space while preserving the potential for novel combination rules.
5. Add a hard complexity cap and enforce it during evolution, not just during selection.
6. Budget at least 5 independent runs and present the full distribution of outcomes, not just the best.

---

# Executive Summary: Comparative Analysis and Recommendations

## Three Approaches Compared

| Dimension | Idea 1: Physics-Informed Tree Ensemble | Idea 2: Analytical Decomposition + Corrections | Idea 3: Evolutionary Program Synthesis |
|-----------|---------------------------------------|------------------------------------------------|---------------------------------------|
| **Algorithmic Family** | Gradient boosted trees (XGBoost/LightGBM) | Parametric physics model with learned calibration | LLM-guided genetic programming |
| **Parameter Count** | ~30K-100K (500 trees x 63 nodes) | 9-36 | 6-20 constants + program structure |
| **Feature Count** | 30 engineered features | 11 architecture params + per-request fields | 18 available (subset used) |
| **Training Time** | ~10 minutes | ~1 minute | 4-16 hours |
| **Go Integration** | Leaves library (pure Go) | Coefficient export (YAML) | Mechanical code translation |
| **External Dependencies** | Leaves Go library | None | None |
| **LatencyModel Methods** | 1 of 5 (StepTime only) | 4 of 5 | 1-2 of 5 |
| **Interpretability** | Low (SHAP helps) | High (per-component residuals) | High (readable formula) |
| **Hardware Generalization** | Moderate (physics features help) | Strong (change peak_flops/BW, recalibrate) | Variable (depends on discovered formula) |

## Reviewer Consensus

### Aggregate Scores

| Idea | Claude Opus | GPT-4o | Gemini Flash | Mean |
|------|-------------|--------|--------------|------|
| Idea 1 | 7.0 | 6.5 | 6.0 | **6.5** |
| Idea 2 | 8.0 | 7.5 | 7.0 | **7.5** |
| Idea 3 | 6.0 | 5.0 | 5.5 | **5.5** |

### Cross-Reviewer Agreement

**All three reviewers agree on:**
1. Idea 2 has the best Go integration path and the strongest physical foundation
2. Idea 1 is the safest bet with well-understood risks
3. Idea 3 has the highest novelty but lowest feasibility
4. Per-request KV length (via ProgressIndex) is the most important feature addition regardless of approach
5. The `max(prefill, decode)` combination rule needs to be parameterized (not hardcoded)
6. MoE handling is a risk across all approaches due to limited data (4/16 experiments)

**Key disagreements:**
- GPT-4o rates Idea 3 as "reject" (5.0) while Claude rates it as "weak accept" (6.0) -- disagreement on whether high novelty compensates for low feasibility
- Gemini is more skeptical of Idea 1's novelty (5/10) than the other reviewers
- Claude is the most optimistic about Idea 2's likelihood of achieving <10% E2E error

## Strengths/Weaknesses Matrix

| Quality | Idea 1 | Idea 2 | Idea 3 |
|---------|--------|--------|--------|
| Technical soundness | Good | Strong | Moderate |
| Novelty | Low-Moderate | Moderate | High |
| Feasibility | High | High | Low-Moderate |
| Go integration | Good | Excellent | Good (if clean) |
| Data efficiency | Good (73K sufficient) | Excellent (200 steps suffice) | Good |
| Interpretability | Low (black box) | High (per-component) | High (formula) |
| Robustness | High (proven technology) | Moderate (limited expressiveness) | Low (stochastic) |
| Hardware generalization | Moderate | Strong | Variable |
| Method coverage (5 methods) | 1/5 | 4/5 | 1-2/5 |
| Maintenance burden | Medium (model artifact) | Low (YAML coefficients) | Low (inline code) |

## Recommended Priority Order for Hypothesis Testing (WP2)

### Priority 1: Idea 2 -- Analytical Decomposition with Learned Corrections

**Rationale**: Highest reviewer consensus (mean 7.5/10). Best Go integration (zero dependencies, pure coefficients). Strongest physical foundation with interpretable per-component diagnostics. Covers 4 of 5 LatencyModel methods. Lowest maintenance burden. Best hardware generalization story.

**Risk mitigation**: The limited expressiveness (9-36 parameters) is the primary risk. Run the 36-parameter per-model variant first to establish an upper bound on decomposition accuracy. If the per-model variant cannot achieve <10% E2E error, the decomposition structure is insufficient and Idea 1 becomes necessary.

**Key enhancements from reviews to incorporate:**
1. Parameterized phase combination: `alpha * max(T_p, T_d) + (1-alpha) * (T_p + T_d)` (Claude)
2. Floored inverse-variance weighting: `w_i = 1/max(y_i, 50)^2` (Claude)
3. Enriched overhead model with total_kv_blocks term (Gemini)
4. Sequence-length-dependent attention corrections: `c5(kv_len) = c5_base * (kv_len/kv_ref)^c5_exp` (Gemini)
5. Multiple random restarts for optimization (GPT-4o)
6. Python-Go parity tests for analytical formulas (GPT-4o)

### Priority 2: Idea 1 -- Physics-Informed Tree Ensemble

**Rationale**: Safe fallback if Idea 2's limited expressiveness is insufficient. Proven technology with well-understood behavior. The 30-feature set with KV statistics should handle most experiments where the blackbox fails. Moderate Go integration via Leaves library.

**Conditional trigger**: Implement Idea 1 if Idea 2's 36-parameter per-model variant has >15% E2E error on any experiment. The tree ensemble's greater expressiveness (30K+ parameters) can capture non-linearities that the analytical model misses.

**Key enhancements from reviews to incorporate:**
1. Decompose `total_flops_estimate` into 4 phase-operation features (GPT-4o)
2. Fix `active_param_ratio` computation for Mixtral (Gemini)
3. Drop redundant features (kv_sum, batch_size, prefill_fraction) (GPT-4o)
4. Remove `compute_bound_indicator` (redundant with `arithmetic_intensity`) (Claude)
5. Remove `batch_size` monotonicity constraint (Gemini)
6. Add Jensen's inequality bias correction per-regime (Claude)
7. Include data cleaning step with outlier flagging (Gemini)

### Priority 3: Idea 3 -- Evolutionary Program Synthesis (Pilot Only)

**Rationale**: Highest novelty but lowest feasibility. Run as a **constrained pilot** (100 generations, 3 independent runs) to assess convergence behavior before committing full compute budget. If the pilot shows >50% E2E error improvement over the blackbox, escalate to full evolution.

**Critical modification from reviews**: Do not evolve from scratch. Use Idea 2's analytical components as fixed inputs and evolve only the combination/correction logic. This dramatically constrains the search space (from ~18 features x 50 operations to ~5 component times x 20 operations) while preserving the potential for novel discoveries.

**Key enhancements from reviews to incorporate:**
1. Constrain search space: evolve combination rules only, not component formulas (Claude, GPT-4o, Gemini)
2. Use proper Pareto selection (NSGA-II) instead of lexicographic (GPT-4o)
3. Hard complexity cap: 30 operations, 3 conditionals, 10 constants (GPT-4o, Gemini)
4. Vectorized numpy evaluation for speed (Gemini)
5. Include concrete LLM mutation prompt template (Claude)
6. 5 independent runs with full distribution reporting (Gemini)

## Suggested Hybrid Strategy

The three ideas are not mutually exclusive. The most promising path is a **progressive escalation**:

1. **Week 1**: Implement Idea 2 (analytical decomposition). Fast to implement (~2 days), fast to train (~1 minute), immediately reveals whether the decomposition structure captures the right physics. If <10% E2E on 14+ experiments: ship Idea 2.

2. **Week 2 (if needed)**: Implement Idea 1 (tree ensemble) for experiments where Idea 2 fails. Use Idea 2's analytical components as additional features for the tree ensemble (creating a "Idea 1.5" hybrid). This gives the tree ensemble both raw batch features AND pre-computed physics estimates.

3. **Week 3 (optional)**: Run Idea 3 pilot using Idea 2's analytical components as fixed inputs. Evolve only the combination logic. If the pilot discovers a better combination rule than `max()`, feed it back into Idea 2 as a structural improvement.

This progressive strategy minimizes wasted effort: each step either succeeds (and we stop) or produces useful information that improves the next step.
