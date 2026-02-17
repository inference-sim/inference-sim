# PROBLEM (verbatim from problem.md)

Problem statement:
- objective: currently blis contains two modeling techniques for inference performance:
  - blackbox optimization approach as documented in @docs/approach.md and
  - roofline approach in @docs/roofline.md
- come up with a third approach that can simulate diverse settings such as
  - model type/architecture: i.e. dense vs. MoE
  - different workload types: i.e. prefill- and decode-heavy and mixed/balanced workloads
  - different hardware: i.e. A100, H100
  - different tensor parallelism sizes and expert parallel settings
  - different vLLM knobs: i.e. chunk size, max-model-len, and --cpu-offloading
- constraints
  - We still want alpha (used to compute the delay between request arrival and queuing) and beta (used to compute the vLLM busy-loop step time) coefficients, but you have freedom to determine what alpha and beta coefficients need to be to achieve objective.
  - We can heavily featurize each setting. You can derive any new features using a model's config.json, the hardware specs (will be provided through data sheets in JSON), vLLM configuration specs, and request characteristics. These are known for each simulation.
  - carefully look into the request journey tracing, step tracing, and KV event streams documented in @vllm.md. Make sure the coefficient vectors alpha and beta can be learned using the tracing and KV event stream data. Provide a short description of the training pipeline. It can include anything from simple linear regression to advanced techniques like expectation maximization, convex optimization, or anything else that is relevant
  - The arrival to queuing latency is alpha * feature_vec_1 and the step-time latency is beta * feature_vec_2 (the `*` represents dot product). Feel free to derive the features in any way you think is appropriate. Show your reasoning and explain why the features meet the constraints and objectives.
  - we want the training procedure to not overfit but be robust

# BACKGROUND: blis.md (verbatim)

# BLIS: Blackbox Inference Simulator

## Overview

BLIS is a **Discrete Event Simulator (DES)** for LLM inference platforms (vLLM, SGLang). It enables performance prediction, capacity planning, and what-if analysis **without requiring physical GPUs** by simulating request arrival, KV-cache dynamics, scheduling, and token generation.

---

## Core Architecture

### Components

1. **Event Engine**: Priority-queue-based orchestrator managing global simulation clock and triggering callbacks at precise timestamps.

2. **Virtual Scheduler**: High-fidelity mirror of vLLM's `Scheduler` class, managing request state transitions across `Waiting`, `Running`, and `Preempted` queues.

3. **Virtual KV-Cache Manager**: Simulates PagedAttention block allocation, tracking logical-to-physical mappings, reference counts for prefix sharing, and memory fragmentation.

4. **Latency Model**: Predicts iteration duration based on batch composition, model architecture, and hardware topology.

### Main Simulation Loop

```python
while time < horizon and active_requests:
    scheduled_batch = scheduler.schedule()
    step_time = gpu_latency_model.predict(scheduled_batch)
    scheduler.update_processing_indices(scheduled_batch)
    sim.Clock.advance(step_time)
```

---

## Existing Modeling Approaches

### Approach 1: Blackbox Optimization

Uses trained alpha/beta coefficients to predict latencies:

**GPU Latency Model:**
$$L_{\text{gpu},k} = \beta_0 + \beta_1 X_k + \beta_2 Y_k$$

Where:
- $X_k$: Total uncached prefill tokens in iteration $k$
- $Y_k$: Total decode tokens in iteration $k$
- $\beta_i$: Learned coefficients from hardware-specific profiling

**CPU/System Overhead Model:**
$$L_{\text{cpu}} = \alpha_0 + \alpha_1 M + \alpha_2 N$$

Where:
- $M$: Input sequence length
- $N$: Generated output length

**Training**: Bayesian optimization minimizing multi-objective loss over TTFT, ITL, E2E metrics (mean and P90).

**Limitation**: Requires hours of profiling per (Model, GPU, TP, vLLM version) configuration.

### Approach 2: Roofline Model

Analytical approach using roofline model principles:

$$\text{Phase Time} = \max\left( \frac{\text{Total FLOPS}}{\text{Peak Performance}}, \frac{\text{Total Bytes}}{\text{Memory Bandwidth}} \right)$$

**FLOP Calculation**: GEMM ops (QKV projections, attention, MLP) + Vector ops (Softmax, RoPE, normalization)

**Memory Access**: Weight loading, KV cache growth, KV cache reads

**Step Time** = Prefill Phase + Decode Phase + Communication Overhead + Hardware Overheads

**Limitation**: Requires tuning efficiency factors (MFU, bandwidth efficiency) that may not generalize.

---

## Metrics Captured

| Metric | Definition |
|--------|------------|
| **TTFT** | Time to First Token: $L_{\text{cpu}} + \sum_{k \in P} L_{\text{gpu},k} - T_{\text{arrival}}$ |
| **ITL** | Inter-Token Latency: $\Delta_t$ between consecutive decode iterations |
| **E2E** | End-to-End: $L_{\text{cpu}} + \sum_{k \in \{P \cup D\}} L_{\text{gpu},k} - T_{\text{arrival}}$ |

---

## Configuration Inputs

### Model Configuration (config.json)

```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "vocab_size": 32000
}
```

Derived features: model type (dense/MoE), attention pattern (MHA/GQA/MQA), model size proxy, KV cache size per token.

### Hardware Configuration

```json
{
  "H100": {
    "TFlopsEff": 989,
    "BwEffTBs": 3.35,
    "memory_gb": 80
  },
  "A100": {
    "TFlopsEff": 312,
    "BwEffTBs": 2.039,
    "memory_gb": 80
  }
}
```

### vLLM Configuration

| Parameter | Description |
|-----------|-------------|
| `--max-num-batched-tokens` | Maximum tokens per batch |
| `--max-num-seqs` | Maximum concurrent sequences |
| `--gpu-memory-utilization` | KV cache memory fraction |
| `--enable-chunked-prefill` | Allow mixed prefill/decode batches |
| `--enable-prefix-caching` | Enable KV cache prefix sharing |
| `--tensor-parallel-size` | TP degree |

---

## Capabilities Required for Third Approach

The new approach must handle:

1. **Model diversity**: Dense vs MoE architectures
2. **Workload diversity**: Prefill-heavy, decode-heavy, mixed
3. **Hardware diversity**: A100, H100, different interconnects
4. **Parallelism**: Tensor parallelism, expert parallelism
5. **vLLM knobs**: Chunk size, max-model-len, CPU offloading

While maintaining the coefficient structure:
- **Alpha coefficients**: `arrival_to_queuing_latency = alpha · feature_vec_1`
- **Beta coefficients**: `step_time_latency = beta · feature_vec_2`

---

## Available Training Data

From vLLM tracing infrastructure:

### Request Journey Events
- ARRIVED, QUEUED, SCHEDULED, FIRST_TOKEN, FINISHED timestamps
- Per-request: prompt_tokens, output_tokens, num_preemptions

### Step-Level Events
- queue.running_depth, queue.waiting_depth
- batch.prefill_tokens, batch.decode_tokens
- step.duration_us
- kv.usage_gpu_ratio

### KV Cache Events
- Block allocations, transfers, evictions
- Prefix cache hits

---

## Robustness Requirements

Training must:
- Not overfit to specific workload patterns
- Generalize across hardware/model combinations
- Handle edge cases (high KV pressure, preemption)
- Be validated on held-out temporal test sets

# BACKGROUND: vllm.md (verbatim)

# vLLM Tracing and Performance Modeling Summary

This document summarizes vLLM's tracing capabilities and internal mechanisms relevant to building a simulation model with learnable alpha (arrival-to-queuing latency) and beta (step time) coefficients.

## Overview

vLLM provides three complementary observability streams that can be used to learn performance coefficients:

1. **Request Journey Tracing** - Per-request lifecycle events (arrival → completion)
2. **Step Tracing** - Per-scheduler-iteration aggregate metrics
3. **KV Cache Events** - Block allocation, transfer, and eviction events

---

## 1. Request Journey Tracing

### Event Sequence

Each request flows through two layers with distinct events:

```
┌─────────────────────────────────────────────────────────────┐
│ API Layer (llm_request span)                                 │
│                                                              │
│  ARRIVED → HANDOFF_TO_CORE → FIRST_RESPONSE → DEPARTED      │
│               │                                              │
│               └──┐                                           │
│  ┌───────────────▼──────────────────────────────────────┐   │
│  │ Core Layer (llm_core span)                            │   │
│  │                                                       │   │
│  │  QUEUED → SCHEDULED → FIRST_TOKEN → FINISHED         │   │
│  │              ↓                                        │   │
│  │         [PREEMPTED → SCHEDULED (resume)]              │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Events and Timing Points

| Event | Layer | Description | Timing Significance |
|-------|-------|-------------|---------------------|
| **ARRIVED** | API | Request received by server | `t_arrival` |
| **HANDOFF_TO_CORE** | API | Request sent to scheduler | API overhead measured |
| **QUEUED** | Core | Added to wait queue | `t_queued` (start of alpha interval) |
| **SCHEDULED** | Core | Resources allocated, execution starts | `t_scheduled` (end of alpha interval) |
| **FIRST_TOKEN** | Core | First output token generated | Prefill complete |
| **PREEMPTED** | Core | Resources reclaimed (if any) | Preemption tracking |
| **FINISHED** | Core | Request completed | `t_completion` |
| **DEPARTED** | API | Response sent to client | End-to-end complete |

### Event Attributes (Per-Request)

Each event carries rich attributes useful for feature extraction:

**Progress Tracking:**
- `phase`: Current phase - "WAITING", "PREFILL", or "DECODE"
- `prefill_done_tokens` / `prefill_total_tokens`: Prompt processing progress
- `decode_done_tokens` / `decode_max_tokens`: Output generation progress

**Timing:**
- `ts.monotonic`: High-precision timestamp (nanoseconds)
- `scheduler.step`: Scheduler iteration number

**Lifecycle:**
- `num_preemptions`: Preemption count for this request
- `schedule.kind`: "FIRST" or "RESUME" (after preemption)
- `finish.status`: "stopped", "length", "aborted", "ignored", "error"

### Alpha Coefficient Learning

**Alpha models the delay between request arrival and scheduling:**

```
alpha_latency = alpha · feature_vec_1
```

**Learnable from journey tracing:**
- Time from `QUEUED` to `SCHEDULED` events
- Attributes at queue time: `prefill_total_tokens`, queue position (from step tracing)

**Feature candidates for alpha:**
- Queue depth at arrival (from concurrent step tracing)
- Request prompt length (prefill_total_tokens)
- Batch fullness (running requests)
- KV cache pressure (usage ratio)
- Time since last scheduling decision

---

## 2. Step-Level Tracing

### Batch Summary Events

Each sampled scheduler step emits `step.BATCH_SUMMARY` with:

**Queue State:**
- `queue.running_depth`: Actively processing requests
- `queue.waiting_depth`: Requests waiting in queue

**Batch Composition:**
- `batch.num_prefill_reqs`: Requests in prefill phase
- `batch.num_decode_reqs`: Requests in decode phase
- `batch.scheduled_tokens`: Total tokens this step

**Token Distribution:**
- `batch.prefill_tokens`: Tokens for prompt processing
- `batch.decode_tokens`: Tokens for generation

**Lifecycle Counts:**
- `batch.num_finished`: Requests completed this step
- `batch.num_preempted`: Requests preempted this step

**KV Cache Health:**
- `kv.usage_gpu_ratio`: Cache utilization [0.0, 1.0]
- `kv.blocks_total_gpu`: Total GPU blocks available
- `kv.blocks_free_gpu`: Free GPU blocks remaining

**Timing:**
- `step.id`: Monotonic step counter
- `step.ts_start_ns` / `step.ts_end_ns`: Step timestamps
- `step.duration_us`: Step duration in microseconds

### Request Snapshot Events (Rich Mode)

When enabled, `step.REQUEST_SNAPSHOT` provides per-request state:

**Per-Request Progress:**
- `request.id`: Unique identifier
- `request.phase`: "PREFILL" or "DECODE"
- `request.num_prompt_tokens`: Total prompt tokens
- `request.num_computed_tokens`: Tokens computed so far
- `request.num_output_tokens`: Output tokens generated
- `request.num_preemptions`: Preemption count
- `request.scheduled_tokens_this_step`: Tokens scheduled this step

**Per-Request KV Cache:**
- `kv.blocks_allocated_gpu`: GPU blocks for this request
- `kv.blocks_cached_gpu`: Blocks from prefix cache hits
- `request.effective_prompt_len`: Prompt length after cache reduction

### Beta Coefficient Learning

**Beta models the step execution time:**

```
beta_latency = beta · feature_vec_2
```

**Learnable from step tracing:**
- `step.duration_us` as target variable
- Batch composition and token counts as features

**Feature candidates for beta:**
- `batch.num_prefill_reqs` and `batch.num_decode_reqs`
- `batch.prefill_tokens` and `batch.decode_tokens`
- `batch.scheduled_tokens` (total batch size)
- `queue.running_depth` (batch size in requests)
- `kv.usage_gpu_ratio` (memory pressure)
- Tensor parallelism degree (from config)
- Model architecture features (derived from config.json)

---

## 3. KV Cache Events

### Event Types

From the KV offloading connector:

| Event | Description |
|-------|-------------|
| `CacheStoreCommitted` | Block stored to CPU cache |
| `TransferInitiated` | GPU→CPU transfer started |
| `TransferCompleted` | Transfer finished |

### Block Allocation Mechanics

**Key concepts:**
- Blocks are allocated incrementally during chunked prefill
- `req.block_hashes` reflects token IDs (known at request creation)
- `_request_block_ids` contains only allocated blocks
- During chunked prefill: `len(block_hashes) > len(block_ids)`

**Store deferral mechanism:**
- Stores prepared in step N are submitted in step N+1
- Ensures KV is computed before transfer

**Preemption handling:**
- `_request_block_ids[req_id]` reset on preemption
- Rebuilt on resume

---

## 4. Configuration Sources for Feature Engineering

### Model Configuration (config.json)

From HuggingFace model config:

```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "max_position_embeddings": 4096
}
```

**Derived features:**
- Model type: dense vs MoE (from architecture)
- Attention pattern: MHA vs GQA vs MQA (from num_kv_heads ratio)
- Model size proxy: hidden_size × num_layers
- KV cache size per token: hidden_size × num_kv_heads / num_attn_heads × 2 × dtype_bytes

### Hardware Configuration

GPU specifications:

```json
{
  "H100": {
    "memory_bandwidth_gb_s": 3350,
    "compute_tflops_fp16": 989,
    "memory_gb": 80,
    "sm_count": 132
  },
  "A100": {
    "memory_bandwidth_gb_s": 2039,
    "compute_tflops_fp16": 312,
    "memory_gb": 80,
    "sm_count": 108
  }
}
```

**Derived features:**
- Compute-to-bandwidth ratio
- Memory capacity (affects max batch size)
- Relative performance factor

### vLLM Configuration

Key knobs affecting performance:

| Parameter | Effect on Alpha | Effect on Beta |
|-----------|----------------|----------------|
| `--max-num-batched-tokens` | Affects queue wait time | Caps batch size |
| `--max-num-seqs` | Affects scheduling capacity | Caps concurrent requests |
| `--gpu-memory-utilization` | Affects KV cache size | Affects preemption rate |
| `--enable-chunked-prefill` | May increase queue time | Allows mixed batches |
| `--enable-prefix-caching` | Reduces effective queue time | Reduces prefill work |
| `--tensor-parallel-size` | - | Affects compute efficiency |
| `--enforce-eager` | - | May affect step latency |

### Request Characteristics

Per-request features:

| Feature | Source | Use |
|---------|--------|-----|
| `prompt_length` | Request input | Alpha, Beta |
| `max_output_tokens` | Request params | Lifetime prediction |
| `prefix_hit_ratio` | KV events | Alpha reduction |
| `arrival_rate` | Request timestamps | Queue modeling |

---

## 5. Training Data Pipeline

### Data Collection

1. **Enable tracing:**
   ```bash
   vllm serve MODEL \
     --enable-journey-tracing \
     --journey-tracing-sample-rate 1.0 \
     --step-tracing-enabled \
     --step-tracing-sample-rate 1.0 \
     --step-tracing-rich-subsample-rate 1.0 \
     --otlp-traces-endpoint http://collector:4317
   ```

2. **Collect traces to file:**
   ```yaml
   # OTEL collector config
   exporters:
     file:
       path: /data/traces.json
   ```

3. **Parse traces:**
   - Extract journey events by scope `vllm.api` and `vllm.scheduler`
   - Extract step events by scope `vllm.scheduler.step`
   - Join by `scheduler.step` / `step.id`

### Alpha Training Data

**Target:** `t_scheduled - t_queued` (from journey tracing)

**Features:**
- From request: `prefill_total_tokens`
- From step trace at queue time: `queue.waiting_depth`, `queue.running_depth`, `kv.usage_gpu_ratio`
- From config: model size proxy, hardware type, TP degree

**Sample row:**
```
alpha_latency_ms, prompt_tokens, queue_depth, running_depth, kv_ratio, model_size, hw_type, tp
150.3, 512, 8, 24, 0.72, 7B, H100, 1
```

### Beta Training Data

**Target:** `step.duration_us` (from step tracing)

**Features:**
- From step trace: `batch.num_prefill_reqs`, `batch.num_decode_reqs`, `batch.prefill_tokens`, `batch.decode_tokens`, `kv.usage_gpu_ratio`
- From config: model type, hardware, TP degree

**Sample row:**
```
step_duration_us, num_prefill, num_decode, prefill_tokens, decode_tokens, kv_ratio, model_type, hw, tp
2350, 2, 18, 256, 18, 0.65, dense, H100, 1
```

### Recommended Training Approach

1. **Data cleaning:**
   - Remove outliers (warmup steps, preemption-heavy steps)
   - Normalize features by hardware/model

2. **Model selection:**
   - Start with linear regression for interpretability
   - Consider polynomial features for non-linear effects (batch size squared)
   - Ridge/Lasso for regularization and feature selection

3. **Validation:**
   - Hold-out test set by time (not random)
   - Cross-validate across different model sizes/hardware

4. **Robustness:**
   - Train on diverse workloads (prefill-heavy, decode-heavy, mixed)
   - Include edge cases (near-full KV cache, high preemption)

---

## 6. Key Formulas

### Time-to-First-Token (TTFT)

```
TTFT = alpha_latency + prefill_time
     = (alpha · F_queue) + (beta_prefill · F_prefill)
```

Where:
- `F_queue`: Queue state features at arrival
- `F_prefill`: Prefill batch features

### Inter-Token Latency (ITL)

```
ITL = beta_decode · F_decode / tokens_per_step
```

### End-to-End Latency

```
E2E = TTFT + (output_tokens - 1) × ITL
```

---

## 7. Feature Engineering Recommendations

### For Alpha (Queue Latency)

**Strong predictors:**
1. Queue depth at arrival (`queue.waiting_depth`)
2. KV cache pressure (`kv.usage_gpu_ratio`)
3. Request prompt length (affects scheduling priority)
4. Running batch size (`queue.running_depth`)

**Interaction terms:**
- `queue_depth × kv_ratio`: High queue + cache pressure = longer wait
- `prompt_length × batch_fullness`: Long prompts harder to schedule when busy

### For Beta (Step Time)

**Strong predictors:**
1. Total scheduled tokens (`batch.scheduled_tokens`)
2. Prefill vs decode ratio (`batch.prefill_tokens / batch.decode_tokens`)
3. Number of requests (`queue.running_depth`)
4. Hardware compute/bandwidth ratio

**Architecture-specific:**
- Dense: Linear in tokens
- MoE: Add expert activation factor
- GQA: Reduce attention compute estimate

**Interaction terms:**
- `prefill_tokens × model_layers`: Prefill is compute-bound
- `decode_tokens × kv_blocks`: Decode is memory-bound

---

## 8. Summary

| Coefficient | Target Variable | Primary Features |
|-------------|-----------------|------------------|
| **Alpha** | `SCHEDULED.ts - QUEUED.ts` | queue_depth, kv_ratio, prompt_len |
| **Beta** | `step.duration_us` | batch_tokens, prefill_ratio, hw_type |

Both can be learned via linear regression on traced data, with features derived from:
- vLLM tracing (journey + step)
- Model config.json
- Hardware specs
- vLLM runtime parameters

# IDEA 3 (verbatim from idea-3.md)

# Idea 3: Hierarchical Feature Decomposition with Transfer Learning (HFD-TL)

## Core Insight

Reviewer feedback on Ideas 1 and 2 highlighted two recurring themes: (1) the difficulty of pre-specifying the right features or regime boundaries, and (2) the substantial data requirements for training accurate models across diverse configurations. This idea proposes a **hierarchical decomposition** where:

1. A **base model** is trained on rich data from a few representative configurations (the "donor" configurations)
2. **Configuration-specific residuals** are learned with minimal data from new configurations (the "target" configurations)

This enables rapid calibration to new (model, hardware, vLLM) settings while preserving the interpretable alpha/beta coefficient structure.

## Approach Overview

### Phase 1: Hierarchical Feature Decomposition

Decompose latency into three additive components:

```
alpha_latency = alpha_base · F_base + alpha_config · F_config + alpha_residual · F_residual
beta_latency = beta_base · F_base + beta_config · F_config + beta_residual · F_residual
```

**Base Features (F_base)** - Universal, hardware-normalized physics:
```python
F_alpha_base = [
    queue_depth / max_num_seqs,  # Normalized queue load
    kv_usage_ratio,  # Cache pressure (dimensionless)
    prompt_tokens / max_context_length,  # Normalized request size
]

F_beta_base = [
    prefill_tokens * model_flops_per_token / hardware_tflops,  # Normalized prefill work
    decode_tokens * kv_bytes_per_token / hardware_bandwidth,  # Normalized decode work
    batch_tokens / max_batch_tokens,  # Normalized batch load
]
```

**Configuration Features (F_config)** - Configuration-specific modifiers:
```python
F_alpha_config = [
    1 / tp_degree,  # TP affects scheduling
    prefix_caching_enabled,  # Binary: affects queue time
    chunked_prefill_enabled,  # Binary: affects scheduling
]

F_beta_config = [
    1 / tp_degree,  # TP reduces per-GPU work
    moe_indicator * num_experts / 8,  # MoE complexity (normalized)
    cpu_offload_enabled,  # Binary: adds transfer overhead
]
```

**Residual Features (F_residual)** - Capture unexplained variance:
```python
F_alpha_residual = [
    1,  # Constant bias term
    (queue_depth / max_num_seqs) ** 2,  # Non-linear queue effect (precomputed)
]

F_beta_residual = [
    1,  # Constant bias term
    kv_usage_ratio ** 2,  # Non-linear saturation effect (precomputed)
]
```

**Note**: The quadratic terms are **precomputed** before the dot product, so they remain scalar features that multiply against scalar coefficients. This satisfies the linear constraint: `alpha · [x, x^2]` is still a dot product.

### Phase 2: Two-Stage Training Pipeline

**Stage 1: Train Base Model on Donor Configurations**

Select 2-3 representative configurations spanning the diversity space:
- Dense model on H100 (e.g., Llama-3-8B)
- MoE model on H100 (e.g., Mixtral-8x7B)
- Dense model on A100 (e.g., Llama-2-7B)

Train the base coefficients on pooled data from all donors:

```python
from sklearn.linear_model import RidgeCV

# Pool data from all donor configurations
X_base = np.vstack([compute_base_features(traces, config) for config in donors])
X_config = np.vstack([compute_config_features(traces, config) for config in donors])
X_combined = np.hstack([X_base, X_config])

y_alpha = np.concatenate([t['scheduled_ts'] - t['queued_ts'] for t in donor_traces])

# Train combined base + config model
alpha_combined_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X_combined, y_alpha)

# Extract base and config coefficients
alpha_base_coef = alpha_combined_model.coef_[:len(F_alpha_base)]
alpha_config_coef = alpha_combined_model.coef_[len(F_alpha_base):]
```

**Stage 2: Calibrate Residuals on Target Configuration**

For a new configuration, collect minimal calibration data (~500-1000 observations) and fit only the residual coefficients:

```python
def calibrate_residuals(target_traces, target_config, base_coef, config_coef):
    # Compute base prediction using frozen coefficients
    F_base = compute_base_features(target_traces, target_config)
    F_config = compute_config_features(target_traces, target_config)
    base_prediction = F_base @ base_coef + F_config @ config_coef

    # Compute residuals
    y_actual = target_traces['scheduled_ts'] - target_traces['queued_ts']
    residuals = y_actual - base_prediction

    # Fit residual coefficients
    F_residual = compute_residual_features(target_traces, target_config)
    residual_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(F_residual, residuals)

    return residual_model.coef_
```

### Phase 3: Inference in BLIS

```go
// Hierarchical alpha computation
func ComputeAlphaLatency(req *Request, state *SimState, config *Config) float64 {
    // Base features (hardware-normalized)
    fBase := []float64{
        float64(state.QueueDepth) / float64(config.MaxNumSeqs),
        state.KVUsageRatio,
        float64(req.PromptTokens) / float64(config.MaxContextLength),
    }

    // Config features
    fConfig := []float64{
        1.0 / float64(config.TPDegree),
        boolToFloat(config.PrefixCaching),
        boolToFloat(config.ChunkedPrefill),
    }

    // Residual features (precomputed non-linear terms)
    normalizedQueue := float64(state.QueueDepth) / float64(config.MaxNumSeqs)
    fResidual := []float64{
        1.0,
        normalizedQueue * normalizedQueue,
    }

    return DotProduct(config.AlphaBase, fBase) +
           DotProduct(config.AlphaConfig, fConfig) +
           DotProduct(config.AlphaResidual, fResidual)
}
```

### Phase 4: Active Learning for Efficient Calibration

To minimize calibration data requirements, use uncertainty-guided sampling:

```python
def select_calibration_points(candidate_traces, base_model, n_samples=500):
    """Select traces that maximize expected information gain."""

    # Compute base predictions and uncertainty estimates
    predictions = base_model.predict(compute_features(candidate_traces))

    # Use residual variance as uncertainty proxy
    # (high residual = high uncertainty = high information value)
    estimated_residuals = np.abs(predictions - candidate_traces['actual_latency'])

    # Stratified sampling: ensure coverage of different operating regions
    strata = assign_strata(candidate_traces)  # By queue depth, KV ratio bins

    selected_indices = []
    for stratum in strata:
        stratum_indices = np.where(strata == stratum)[0]
        stratum_uncertainty = estimated_residuals[stratum_indices]
        # Select top-k highest uncertainty within each stratum
        top_k = stratum_indices[np.argsort(stratum_uncertainty)[-n_samples//len(strata):]]
        selected_indices.extend(top_k)

    return selected_indices
```

## Why This Addresses Reviewer Feedback

### Addresses "Hardcoded Thresholds" (from Idea 2)
- No regime boundaries to specify
- The base model learns smooth relationships across the entire operating range
- Residual coefficients adapt to configuration-specific quirks

### Addresses "Boundary Discontinuities" (from Idea 2)
- Single continuous model, no regime switching
- Non-linear effects captured via precomputed polynomial features
- Smooth predictions across all operating conditions

### Addresses "Quadratic Constraint Violation" (from Ideas 1 & 2)
- Quadratic terms are **features**, not coefficients
- `alpha · [x, x^2, ...]` is still a valid dot product
- Coefficients remain linear and interpretable

### Addresses "Training Data Requirements" (from Ideas 1 & 2)
- Base model trained once on rich donor data (can be expensive)
- New configurations require only ~500-1000 calibration observations
- Active learning further reduces calibration data needs

### Addresses "Feature Engineering Rigidity" (from Idea 1)
- Residual features explicitly capture unexplained variance
- Can add more residual features if systematic patterns emerge
- Hierarchical structure separates universal physics from config-specific effects

## Expected Benefits

| Aspect | PICF (Idea 1) | MoLE-RD (Idea 2) | HFD-TL (This Idea) |
|--------|---------------|------------------|---------------------|
| Calibration data | ~10,000 obs | ~2,000/regime | ~500-1,000 total |
| Regime boundaries | N/A | Hardcoded | None (continuous) |
| Non-linear effects | Limited | Regime-based | Precomputed features |
| New config time | Hours | ~30 min | ~5-10 min |
| Interpretability | High | High (per-regime) | High (per-hierarchy) |

## Training Data Requirements

**Donor configurations (one-time):**
- 3-5 representative (model, hardware, vLLM) combinations
- ~50,000 total step observations across donors
- Should span: dense/MoE, A100/H100, various TP degrees

**Target configurations (per new config):**
- 500-1,000 step observations for residual calibration
- ~100 request journey observations for alpha calibration
- Can be collected in ~5-10 minutes of traffic

## Limitations and Mitigations

**Limitation 1: Donor selection affects transfer quality**
- Mitigation: Include diverse donors spanning model types and hardware
- Validation: Test transfer to held-out configurations before deployment

**Limitation 2: Residual features may be insufficient for highly novel configurations**
- Mitigation: Expand residual feature set if transfer error exceeds threshold
- Fallback: Retrain as a new donor if configuration is fundamentally different

**Limitation 3: Assumes additive decomposition of latency**
- Mitigation: The additive structure is validated by the physics (queueing + processing)
- Empirical: If multiplicative effects emerge, can add interaction terms to residuals
