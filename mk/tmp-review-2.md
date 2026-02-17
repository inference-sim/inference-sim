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

# IDEA 2 (verbatim from idea-2.md)

# Idea 2: Mixture of Linear Experts with Regime Detection (MoLE-RD)

## Core Insight

Reviewer feedback on Idea 1 (PICF) consistently identified the **linear model limitation** as a key weakness: real LLM inference systems exhibit non-linear behaviors such as threshold effects under KV pressure, discontinuities at batch size boundaries, and regime-dependent performance characteristics. Rather than adding polynomial features (which explode combinatorially), this idea proposes **learning multiple linear models, each specialized for a distinct operational regime**, with a lightweight regime detector that routes observations to the appropriate expert.

This maintains the interpretability and constraint-compatibility of linear alpha/beta coefficients while capturing non-linear system behavior through mixture composition.

## Approach Overview

### Phase 1: Regime Identification

LLM inference systems operate in distinct regimes with different performance characteristics:

**Identified Regimes for Alpha (Queueing Latency):**
1. **Low-load regime**: Queue empty or near-empty, immediate scheduling, alpha ≈ constant
2. **Normal-load regime**: Steady-state queueing, alpha ∝ queue_depth × service_time
3. **High-pressure regime**: KV cache near-full, scheduling constrained by memory, alpha dominated by kv_pressure
4. **Preemption regime**: Active preemptions occurring, alpha includes preemption overhead

**Identified Regimes for Beta (Step Latency):**
1. **Prefill-dominated**: batch_prefill_tokens >> batch_decode_tokens, compute-bound
2. **Decode-dominated**: batch_decode_tokens >> batch_prefill_tokens, memory-bound
3. **Mixed-balanced**: Comparable prefill and decode, competition for resources
4. **Saturation regime**: KV cache > 90%, performance degradation due to fragmentation

### Phase 2: Regime Detector (Gating Function)

A simple decision tree or logistic regression classifier determines which regime applies:

**Alpha Regime Detector:**
```python
def detect_alpha_regime(state):
    if state.queue_depth <= 2:
        return "low_load"
    elif state.kv_usage_ratio > 0.85:
        return "high_pressure"
    elif state.num_recent_preemptions > 0:
        return "preemption"
    else:
        return "normal_load"
```

**Beta Regime Detector:**
```python
def detect_beta_regime(batch):
    prefill_ratio = batch.prefill_tokens / (batch.prefill_tokens + batch.decode_tokens + 1e-6)
    if prefill_ratio > 0.7:
        return "prefill_dominated"
    elif prefill_ratio < 0.3:
        return "decode_dominated"
    elif batch.kv_usage_ratio > 0.9:
        return "saturation"
    else:
        return "mixed_balanced"
```

### Phase 3: Regime-Specific Feature Vectors

Each regime has its own feature vector optimized for that regime's dynamics:

**Alpha Feature Vectors:**

```python
# Low-load regime: minimal queueing, focus on API overhead
F_alpha_low = [
    1,  # constant (API baseline)
    prompt_tokens / 1000,  # tokenization overhead
    prefix_cache_miss_ratio,  # cold-start overhead
]

# Normal-load regime: queueing theory applies
F_alpha_normal = [
    queue_depth,
    queue_depth * avg_service_time,  # Little's Law
    prompt_tokens / max_batch_tokens,
    running_depth / max_num_seqs,
]

# High-pressure regime: KV constraints dominate
F_alpha_high = [
    1 / (1 - kv_usage_ratio + 0.01),  # Divergent pressure
    queue_depth * kv_usage_ratio,
    prompt_tokens * kv_usage_ratio,  # Large prompts harder to schedule
]

# Preemption regime: includes preemption overhead
F_alpha_preempt = [
    queue_depth,
    num_preemptions,
    preempted_tokens / 1000,  # Wasted work
    kv_usage_ratio,
]
```

**Beta Feature Vectors:**

```python
# Prefill-dominated: compute-bound behavior
F_beta_prefill = [
    prefill_tokens * flops_per_token / hardware_tflops,
    prefill_tokens**2 / (tp_degree * hardware_tflops),  # Attention
    num_prefill_reqs,  # Per-request overhead
    moe_active_experts * prefill_tokens / hardware_tflops,  # MoE compute
]

# Decode-dominated: memory-bound behavior
F_beta_decode = [
    decode_tokens * kv_bytes_per_token / hardware_bandwidth,
    decode_tokens * running_depth,  # KV reads scale with batch
    decode_tokens * moe_expert_count / hardware_bandwidth,  # MoE memory
    cpu_offload_enabled * offload_transfer_time,  # CPU offload overhead
]

# Mixed-balanced: both matter
F_beta_mixed = [
    prefill_tokens / hardware_tflops,
    decode_tokens / hardware_bandwidth,
    prefill_tokens * decode_tokens / batch_tokens**2,  # Competition
    running_depth * kv_usage_ratio,
]

# Saturation regime: fragmentation and overhead dominate
F_beta_saturation = [
    batch_tokens / hardware_bandwidth,  # Baseline
    kv_fragmentation_ratio * batch_tokens,  # Fragmentation overhead
    num_evictions * block_transfer_time,  # Eviction overhead
    1 / (1 - kv_usage_ratio + 0.01),  # Pressure term
]
```

### Phase 4: Training Pipeline

**Step 1: Regime Labeling**
Label each trace observation with its regime based on the regime detector.

**Step 2: Per-Regime Linear Regression**
Train separate Ridge regression models for each regime:

```python
from sklearn.linear_model import RidgeCV

# Group observations by regime
alpha_regimes = group_by_regime(alpha_traces, detect_alpha_regime)
beta_regimes = group_by_regime(beta_traces, detect_beta_regime)

# Train per-regime models
alpha_models = {}
for regime, traces in alpha_regimes.items():
    X = compute_alpha_features(traces, regime)
    y = traces['scheduled_ts'] - traces['queued_ts']
    alpha_models[regime] = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X, y)

beta_models = {}
for regime, traces in beta_regimes.items():
    X = compute_beta_features(traces, regime)
    y = traces['step_duration_us']
    beta_models[regime] = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X, y)
```

**Step 3: Regime Detector Refinement (Optional)**
If rule-based detection is insufficient, train a classifier on the same data:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Features for regime classification (subset of full features)
regime_features = ['queue_depth', 'kv_usage_ratio', 'prefill_ratio', 'num_preemptions']

alpha_regime_clf = GradientBoostingClassifier(n_estimators=50, max_depth=3).fit(
    alpha_traces[regime_features], alpha_traces['true_regime']
)
```

### Phase 5: Inference in BLIS

```go
// Alpha computation with regime detection
func ComputeAlphaLatency(req *Request, state *SimState, config *Config) float64 {
    regime := DetectAlphaRegime(state)
    features := ComputeAlphaFeatures(req, state, config, regime)
    coefficients := alphaCoefficients[regime]
    return DotProduct(coefficients, features)
}

func DetectAlphaRegime(state *SimState) string {
    if state.QueueDepth <= 2 {
        return "low_load"
    } else if state.KVUsageRatio > 0.85 {
        return "high_pressure"
    } else if state.RecentPreemptions > 0 {
        return "preemption"
    }
    return "normal_load"
}
```

## Why This Addresses Reviewer Feedback

### Addresses "Linear Model Limitation"
- Non-linear behavior captured through **regime switching**, not polynomial expansion
- Each regime uses the most appropriate linear approximation for that operating region
- Threshold effects (e.g., KV saturation) handled by dedicated saturation regime

### Addresses "Feature Completeness Gap"
- **CPU offloading**: Explicit `cpu_offload_enabled * offload_transfer_time` feature in decode regime
- **Expert parallelism**: `moe_active_experts` and `moe_expert_count` features in both prefill and decode
- **Prefix caching**: `prefix_cache_miss_ratio` feature in low-load alpha regime

### Addresses "Alpha Model Scope Confusion"
- Low-load regime explicitly models **arrival-to-queuing** (API overhead + tokenization)
- Other regimes model **queuing-to-scheduling** as originally intended
- Separation is natural because these regimes have different dominant factors

### Addresses "Feature Engineering Rigidity"
- Regimes can be refined based on data; more regimes added as needed
- Per-regime features are smaller and more targeted, easier to validate
- Optional classifier-based regime detection learns boundaries from data

## Expected Benefits

| Aspect | PICF (Idea 1) | MoLE-RD (This Idea) |
|--------|---------------|---------------------|
| Non-linear effects | Limited (linear only) | Captured via regime switching |
| Interpretability | High | High (per-regime) |
| Edge cases (saturation, preemption) | Underspecified | Explicit regimes |
| Training complexity | Single model | Multiple smaller models |
| Feature count | 8+ features | 3-4 features per regime |

## Training Data Requirements

- Same as PICF: 3-5 model sizes, 2+ hardware types, diverse workloads
- Additional: Ensure coverage of all regimes (especially edge cases like saturation)
- ~2,000-3,000 observations per regime (less per regime than single-model approach)

## Limitations

**Limitation 1: Regime boundaries may be discontinuous**
- Mitigation: Use soft gating (weighted average of regime predictions) near boundaries

**Limitation 2: More coefficients to maintain (4 alpha + 4 beta sets)**
- Mitigation: Still interpretable and can be stored compactly; training is parallelizable

**Limitation 3: Regime detector errors propagate to latency errors**
- Mitigation: Classifier accuracy can be validated independently; boundary cases can use ensemble
