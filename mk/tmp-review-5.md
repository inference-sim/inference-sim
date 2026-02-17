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

# IDEA 5 (verbatim from idea-5.md)

# Idea 5: Robust Ensemble with Configuration Embeddings (RECE)

## Core Insight

Reviewer feedback across all previous ideas revealed a common tension: physics-informed features provide interpretability and generalization, but struggle with (1) heavy-tailed latency distributions, (2) configuration-specific non-linearities, and (3) unspecified donor selection. This idea proposes a **robust ensemble** approach that:

1. Uses **quantile regression** to handle heavy-tailed distributions (median prediction + confidence intervals)
2. Learns **configuration embeddings** that capture configuration similarity without manual donor selection
3. Combines **multiple specialist models** via learned weights based on configuration proximity

This provides robustness, automatic configuration similarity, and maintains linear coefficient interpretability.

## Approach Overview

### Phase 1: Quantile Regression for Robust Predictions

Instead of predicting mean latency (which is sensitive to outliers), predict multiple quantiles:

**Quantile Loss Function:**
```python
def quantile_loss(y_true, y_pred, quantile):
    """Asymmetric loss for quantile regression."""
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
```

**Predict Three Quantiles:**
- `tau=0.5` (median): Primary prediction, robust to outliers
- `tau=0.1` (P10): Lower bound for confidence interval
- `tau=0.9` (P90): Upper bound for confidence interval

```python
from sklearn.linear_model import QuantileRegressor

# Train three models per coefficient
alpha_models = {
    'median': QuantileRegressor(quantile=0.5, alpha=1.0).fit(X, y),
    'lower': QuantileRegressor(quantile=0.1, alpha=1.0).fit(X, y),
    'upper': QuantileRegressor(quantile=0.9, alpha=1.0).fit(X, y),
}

# Prediction with confidence interval
def predict_alpha(features, models):
    return {
        'prediction': models['median'].predict(features),
        'ci_lower': models['lower'].predict(features),
        'ci_upper': models['upper'].predict(features),
    }
```

### Phase 2: Configuration Embeddings

Replace manual donor selection with learned embeddings that capture configuration similarity:

**Configuration Vector:**
```python
def compute_config_vector(config):
    """Extract configuration features for embedding."""
    return np.array([
        # Model features
        config.hidden_size / 4096,  # Normalized
        config.num_layers / 32,
        config.num_kv_heads / config.num_heads,  # GQA ratio
        float(config.is_moe),
        config.num_experts / 8 if config.is_moe else 0,

        # Hardware features
        config.hardware_tflops / 1000,
        config.hardware_bandwidth / 3000,
        config.memory_gb / 80,

        # vLLM features
        config.max_batch_tokens / 8192,
        config.max_num_seqs / 256,
        1 / config.tp_degree,
        float(config.chunked_prefill),
        float(config.prefix_caching),
        float(config.cpu_offload),
    ])
```

**Embedding Similarity:**
```python
from scipy.spatial.distance import cosine

def config_similarity(config_a, config_b):
    """Cosine similarity between configuration embeddings."""
    vec_a = compute_config_vector(config_a)
    vec_b = compute_config_vector(config_b)
    return 1 - cosine(vec_a, vec_b)
```

### Phase 3: Ensemble of Specialist Models

Train specialist models on clusters of similar configurations, then ensemble based on similarity:

**Cluster Training:**
```python
from sklearn.cluster import KMeans

def train_specialist_ensemble(all_traces, all_configs, n_specialists=5):
    """Train specialist models on configuration clusters."""

    # Compute configuration embeddings
    config_vectors = np.array([compute_config_vector(c) for c in all_configs])

    # Cluster configurations
    kmeans = KMeans(n_clusters=n_specialists, random_state=42)
    cluster_labels = kmeans.fit_predict(config_vectors)

    # Train specialist per cluster
    specialists = {}
    for cluster_id in range(n_specialists):
        cluster_mask = cluster_labels == cluster_id
        cluster_traces = [t for t, m in zip(all_traces, cluster_mask) if m]
        cluster_configs = [c for c, m in zip(all_configs, cluster_mask) if m]

        X = np.vstack([compute_features(t, c) for t, c in zip(cluster_traces, cluster_configs)])
        y = np.concatenate([t['latency'] for t in cluster_traces])

        specialists[cluster_id] = {
            'centroid': kmeans.cluster_centers_[cluster_id],
            'model': QuantileRegressor(quantile=0.5, alpha=1.0).fit(X, y),
        }

    return specialists, kmeans
```

**Weighted Ensemble Prediction:**
```python
def ensemble_predict(features, config, specialists, temperature=0.5):
    """Predict using similarity-weighted ensemble."""

    config_vec = compute_config_vector(config)
    weights = []
    predictions = []

    for cluster_id, specialist in specialists.items():
        # Compute similarity to cluster centroid
        similarity = 1 - cosine(config_vec, specialist['centroid'])
        weight = np.exp(similarity / temperature)  # Softmax-like weighting
        weights.append(weight)
        predictions.append(specialist['model'].predict(features))

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Weighted average prediction
    ensemble_pred = sum(w * p for w, p in zip(weights, predictions))

    return ensemble_pred, weights
```

### Phase 4: Expanded Feature Set with Communication Overhead

Address the missing communication features identified in Idea 4 feedback:

**Complete Alpha Features (20 features):**
```python
F_alpha_complete = [
    # Base queueing (from Ideas 1-4)
    1,  # Bias
    queue_depth / max_num_seqs,
    (queue_depth / max_num_seqs) ** 2,
    kv_usage_ratio,
    kv_usage_ratio ** 2,

    # Request features
    prompt_tokens / max_context_length,
    (prompt_tokens / max_context_length) ** 2,

    # Interaction terms
    queue_depth * kv_usage_ratio / max_num_seqs,
    prompt_tokens * kv_usage_ratio / max_context_length,

    # Configuration features
    1 / tp_degree,
    prefix_caching_enabled,
    chunked_prefill_enabled,

    # NEW: Communication overhead features
    tp_degree > 1,  # Binary: communication exists
    (tp_degree - 1) * prompt_tokens / interconnect_bandwidth,  # TP comm cost

    # Running batch features
    running_depth / max_num_seqs,
    running_depth * kv_usage_ratio / max_num_seqs,
]
```

**Complete Beta Features (22 features):**
```python
F_beta_complete = [
    # Base workload (from Ideas 1-4)
    1,  # Bias
    prefill_tokens * flops_per_token / hardware_tflops,
    decode_tokens * kv_bytes_per_token / hardware_bandwidth,
    batch_tokens / max_batch_tokens,

    # Non-linear workload
    (prefill_tokens / max_batch_tokens) ** 2,
    (decode_tokens / max_batch_tokens) ** 2,

    # Interaction terms
    prefill_tokens * decode_tokens / max_batch_tokens ** 2,
    prefill_tokens * kv_usage_ratio / max_batch_tokens,
    decode_tokens * running_depth / max_batch_tokens,

    # KV cache features
    kv_usage_ratio,
    kv_usage_ratio ** 2,

    # Configuration features
    1 / tp_degree,
    moe_indicator * num_active_experts / 8,
    cpu_offload_enabled,

    # NEW: Communication overhead features
    (tp_degree - 1) * batch_tokens / interconnect_bandwidth,  # All-reduce cost
    (tp_degree - 1) * running_depth / interconnect_latency,  # Sync cost

    # NEW: CPU offload features
    cpu_offload_enabled * kv_blocks_to_offload / pcie_bandwidth,
    cpu_offload_enabled * kv_blocks_to_restore / pcie_bandwidth,

    # MoE-specific features
    moe_indicator * expert_parallel_degree,
    moe_indicator * tokens_per_expert * expert_flops / hardware_tflops,
]
```

### Phase 5: Automatic Donor Diversity Metric

Define coverage criteria to guide configuration collection:

```python
def compute_coverage_score(existing_configs, candidate_config):
    """Score how much a candidate config adds to diversity."""

    candidate_vec = compute_config_vector(candidate_config)
    existing_vecs = np.array([compute_config_vector(c) for c in existing_configs])

    # Minimum distance to any existing config
    distances = [np.linalg.norm(candidate_vec - ev) for ev in existing_vecs]
    min_distance = min(distances)

    # High score = candidate is far from all existing configs = adds diversity
    return min_distance

def select_donors(candidate_configs, n_donors=5):
    """Greedy selection of diverse donors."""

    selected = [candidate_configs[0]]  # Start with first candidate

    while len(selected) < n_donors:
        scores = [compute_coverage_score(selected, c) for c in candidate_configs
                  if c not in selected]
        best_idx = np.argmax(scores)
        selected.append([c for c in candidate_configs if c not in selected][best_idx])

    return selected
```

### Phase 6: Inference in BLIS

```go
// Robust ensemble alpha computation
func ComputeAlphaLatency(req *Request, state *SimState, config *Config) AlphaResult {
    features := ComputeAlphaFeatures(req, state, config)

    // Compute similarity weights to specialist centroids
    configVec := ComputeConfigVector(config)
    weights := make([]float64, len(config.Specialists))
    totalWeight := 0.0

    for i, specialist := range config.Specialists {
        similarity := CosineSimilarity(configVec, specialist.Centroid)
        weights[i] = math.Exp(similarity / config.Temperature)
        totalWeight += weights[i]
    }

    // Normalize weights and compute ensemble prediction
    prediction := 0.0
    for i, specialist := range config.Specialists {
        weights[i] /= totalWeight
        prediction += weights[i] * DotProduct(specialist.Coefficients, features)
    }

    // Confidence interval from quantile models
    ciLower := DotProduct(config.QuantileLower, features)
    ciUpper := DotProduct(config.QuantileUpper, features)

    return AlphaResult{
        Prediction: prediction,
        CILower:    ciLower,
        CIUpper:    ciUpper,
        Weights:    weights,
    }
}
```

## Why This Addresses All Previous Feedback

### Addresses "Gaussian Assumption" (from Idea 4)
- Quantile regression is distribution-agnostic
- Median prediction (tau=0.5) is robust to outliers
- Confidence intervals capture heavy-tailed uncertainty

### Addresses "Donor Selection Manual" (from Ideas 3-4)
- Configuration embeddings automatically capture similarity
- Greedy diversity selection algorithm for donor collection
- Coverage metric guides data collection priorities

### Addresses "Missing Communication Features" (from Idea 4)
- Explicit TP communication features: `(tp_degree - 1) * tokens / bandwidth`
- CPU offload features: `kv_blocks_to_offload / pcie_bandwidth`
- Expert parallelism features for MoE

### Addresses "Shallow Calibration" (from Ideas 3-4)
- Ensemble of specialists adapts to configuration
- Similarity-based weighting provides soft interpolation
- No hard regime boundaries

### Maintains Linear Constraint
- Each specialist uses linear regression: `alpha · features`
- Ensemble combines linear predictions linearly
- Coefficients remain interpretable per specialist

## Expected Benefits

| Aspect | BSR-ARD (Idea 4) | RECE (This Idea) |
|--------|------------------|------------------|
| Distribution assumption | Gaussian | None (quantile) |
| Donor selection | Manual | Automatic (embeddings) |
| Transfer adaptation | Bias + scale | Weighted ensemble |
| Confidence intervals | z-score based | Quantile-based |
| Communication features | Partial | Complete |

## Training Data Requirements

**Configuration Collection (guided by diversity):**
- Use `select_donors()` to identify high-value configurations
- Target 5-8 diverse configurations across hardware/model/vLLM space
- ~10,000 observations per configuration (50,000-80,000 total)

**Specialist Training:**
- Cluster into 5 specialists via K-means
- Train quantile models (median, P10, P90) per specialist
- ~15 models total (5 specialists × 3 quantiles)

**New Configuration:**
- Compute similarity to specialist centroids
- No explicit calibration needed; ensemble adapts automatically
- Optionally fine-tune specialist weights with small calibration data

## Limitations and Mitigations

**Limitation 1: Ensemble complexity increases inference cost**
- Mitigation: At inference, only evaluate 2-3 highest-weight specialists
- Pre-compute weights for known configurations

**Limitation 2: K-means clustering may not be optimal**
- Mitigation: Use hierarchical clustering or DBSCAN if K-means fails
- Validate cluster quality via silhouette score

**Limitation 3: Quantile regression is slower to train**
- Mitigation: Use approximation algorithms (e.g., linear programming)
- Cache trained models; retraining is infrequent
