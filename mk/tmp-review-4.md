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

# IDEA 4 (verbatim from idea-4.md)

# Idea 4: Bayesian Sparse Regression with Automatic Relevance Determination (BSR-ARD)

## Core Insight

Reviewer feedback on Idea 3 consistently identified three gaps: (1) limited residual features, (2) no principled uncertainty quantification, and (3) no validation protocol for detecting transfer failures. This idea proposes using **Bayesian sparse regression with Automatic Relevance Determination (ARD)** which addresses all three issues:

1. **Sparse feature selection**: ARD automatically prunes irrelevant features, allowing us to include a large candidate set without overfitting
2. **Uncertainty quantification**: Bayesian inference provides predictive distributions, not just point estimates
3. **Transfer failure detection**: High predictive uncertainty signals when the model is extrapolating beyond training data

This maintains the linear dot-product structure while providing principled regularization and uncertainty bounds.

## Approach Overview

### Phase 1: Expanded Candidate Feature Set

Instead of hand-selecting a minimal feature set, include a comprehensive candidate set and let ARD prune irrelevant features:

**Alpha Candidate Features (18 features):**
```python
F_alpha_candidates = [
    # Base queueing features
    1,  # Constant/bias
    queue_depth,
    queue_depth / max_num_seqs,
    queue_depth ** 2 / max_num_seqs ** 2,  # Precomputed

    # KV cache features
    kv_usage_ratio,
    kv_usage_ratio ** 2,  # Saturation effect
    1 / (1 - kv_usage_ratio + 0.05),  # Pressure term (bounded)

    # Request features
    prompt_tokens / max_context_length,
    (prompt_tokens / max_context_length) ** 2,

    # Interaction terms
    queue_depth * kv_usage_ratio / max_num_seqs,
    prompt_tokens * kv_usage_ratio / max_context_length,
    prompt_tokens * queue_depth / (max_context_length * max_num_seqs),

    # Configuration features
    1 / tp_degree,
    prefix_caching_enabled,
    chunked_prefill_enabled,
    prefix_caching_enabled * kv_usage_ratio,  # Cache hit effect under pressure
    chunked_prefill_enabled * queue_depth / max_num_seqs,

    # Running batch features
    running_depth / max_num_seqs,
]

F_beta_candidates = [
    # Base workload features
    1,  # Constant/bias
    prefill_tokens * flops_per_token / hardware_tflops,
    decode_tokens * kv_bytes_per_token / hardware_bandwidth,
    batch_tokens / max_batch_tokens,

    # Non-linear workload features
    (prefill_tokens / max_batch_tokens) ** 2,  # Attention scaling
    (decode_tokens / max_batch_tokens) ** 2,

    # Interaction terms
    prefill_tokens * decode_tokens / max_batch_tokens ** 2,  # Mixed batch effect
    prefill_tokens * kv_usage_ratio / max_batch_tokens,
    decode_tokens * running_depth / max_batch_tokens,

    # KV cache features
    kv_usage_ratio,
    kv_usage_ratio ** 2,  # Saturation

    # Configuration features
    1 / tp_degree,
    moe_indicator * num_active_experts / 8,
    cpu_offload_enabled,
    moe_indicator * decode_tokens / hardware_bandwidth,  # MoE decode

    # Request count features
    num_prefill_reqs,
    num_decode_reqs,
    running_depth / max_num_seqs,
]
```

### Phase 2: Bayesian Sparse Regression with ARD

ARD places independent precision hyperparameters on each coefficient, allowing the model to learn which features are relevant:

```python
import numpy as np
from scipy.optimize import minimize

class BayesianARDRegression:
    """
    Bayesian Linear Regression with Automatic Relevance Determination.
    Implements the ARD prior: p(w_i | α_i) = N(0, 1/α_i)
    where α_i is the precision (inverse variance) for each weight.
    """

    def __init__(self, n_features, prior_alpha=1.0, noise_precision=1.0):
        self.n_features = n_features
        self.alpha = np.ones(n_features) * prior_alpha  # Per-feature precision
        self.beta = noise_precision  # Observation noise precision

    def fit(self, X, y, max_iter=100, tol=1e-4):
        """Fit using Evidence Maximization (Type-II ML)."""
        n_samples = X.shape[0]

        for iteration in range(max_iter):
            # Compute posterior covariance and mean
            A = np.diag(self.alpha)
            Sigma_inv = A + self.beta * X.T @ X
            Sigma = np.linalg.inv(Sigma_inv)
            mu = self.beta * Sigma @ X.T @ y

            # Update hyperparameters using evidence maximization
            gamma = 1 - self.alpha * np.diag(Sigma)  # Effective number of params
            alpha_new = gamma / (mu ** 2 + 1e-10)
            alpha_new = np.clip(alpha_new, 1e-6, 1e6)  # Numerical stability

            residuals = y - X @ mu
            beta_new = (n_samples - np.sum(gamma)) / (residuals @ residuals + 1e-10)
            beta_new = np.clip(beta_new, 1e-6, 1e6)

            # Check convergence
            if np.max(np.abs(alpha_new - self.alpha)) < tol:
                break

            self.alpha = alpha_new
            self.beta = beta_new

        self.mean = mu
        self.covariance = Sigma
        self.relevant_features = self.alpha < 1e3  # Features with small alpha are relevant

    def predict(self, X, return_std=False):
        """Predict with optional uncertainty."""
        mu = X @ self.mean

        if return_std:
            # Predictive variance = noise + model uncertainty
            var = 1/self.beta + np.sum((X @ self.covariance) * X, axis=1)
            return mu, np.sqrt(var)
        return mu

    def get_sparse_coefficients(self):
        """Return coefficients, zeroing out irrelevant features."""
        coef = self.mean.copy()
        coef[~self.relevant_features] = 0.0
        return coef
```

### Phase 3: Training Pipeline with Uncertainty-Aware Validation

**Stage 1: Train on Donor Configurations**

```python
def train_bayesian_model(donor_traces, configs):
    """Train ARD model on pooled donor data."""

    # Compute all candidate features
    X = np.vstack([compute_alpha_candidates(t, c) for t, c in zip(donor_traces, configs)])
    y = np.concatenate([t['scheduled_ts'] - t['queued_ts'] for t in donor_traces])

    # Fit Bayesian ARD regression
    model = BayesianARDRegression(n_features=X.shape[1])
    model.fit(X, y, max_iter=200)

    # Report which features were selected
    selected = np.where(model.relevant_features)[0]
    print(f"ARD selected {len(selected)} of {X.shape[1]} features: {selected}")

    return model
```

**Stage 2: Validate Transfer with Uncertainty**

```python
def validate_transfer(model, target_traces, target_config, threshold_z=2.0):
    """Check if transfer is valid using predictive uncertainty."""

    X = compute_alpha_candidates(target_traces, target_config)
    y_actual = target_traces['scheduled_ts'] - target_traces['queued_ts']

    # Get predictions with uncertainty
    y_pred, y_std = model.predict(X, return_std=True)

    # Compute z-scores: how many std deviations from prediction?
    z_scores = np.abs(y_actual - y_pred) / y_std

    # Flag transfer failure if too many outliers
    outlier_rate = np.mean(z_scores > threshold_z)
    if outlier_rate > 0.1:  # More than 10% outliers
        return False, outlier_rate

    return True, outlier_rate
```

**Stage 3: Calibrate or Retrain**

```python
def calibrate_or_retrain(model, target_traces, target_config):
    """Calibrate residuals or trigger full retraining."""

    transfer_valid, outlier_rate = validate_transfer(model, target_traces, target_config)

    if transfer_valid:
        # Calibrate: fit residual bias and variance scaling
        X = compute_alpha_candidates(target_traces, target_config)
        y_actual = target_traces['scheduled_ts'] - target_traces['queued_ts']
        y_pred, _ = model.predict(X, return_std=True)

        residuals = y_actual - y_pred
        bias = np.mean(residuals)
        scale = np.std(residuals) / np.std(y_pred) if np.std(y_pred) > 0 else 1.0

        return {"calibration": "bias_shift", "bias": bias, "scale": scale}
    else:
        # Transfer failed: retrain with target data added to donors
        return {"calibration": "retrain_required", "outlier_rate": outlier_rate}
```

### Phase 4: Inference in BLIS with Uncertainty

```go
// Alpha computation with uncertainty
func ComputeAlphaLatency(req *Request, state *SimState, config *Config) (float64, float64) {
    features := ComputeAlphaCandidates(req, state, config)

    // Sparse dot product (only relevant features)
    prediction := 0.0
    for i, coef := range config.AlphaCoefficients {
        if config.AlphaRelevant[i] {
            prediction += coef * features[i]
        }
    }

    // Apply calibration
    prediction = prediction * config.AlphaScale + config.AlphaBias

    // Compute uncertainty (simplified: use precomputed average variance)
    uncertainty := config.AlphaBaseUncertainty * math.Sqrt(ComputeFeatureVariance(features, config))

    return prediction, uncertainty
}
```

## Why This Addresses Reviewer Feedback

### Addresses "Limited Residual Features" (from Idea 3)
- Start with 18+ candidate features; ARD prunes irrelevant ones
- No manual feature selection required
- Can easily add more candidates without risk of overfitting

### Addresses "No Uncertainty Quantification" (from Idea 3)
- Bayesian inference provides predictive variance, not just point estimates
- Uncertainty increases when extrapolating beyond training data
- Principled probabilistic framework

### Addresses "No Validation Protocol" (from Idea 3)
- Z-score-based outlier detection identifies transfer failures
- Quantitative threshold (>10% outliers → retrain) is explicit
- Automatic decision between calibration and retraining

### Addresses "Donor Selection Underspecified" (from Idea 3)
- Uncertainty quantification flags when donors are insufficient
- Leave-one-donor-out cross-validation can validate donor coverage
- High uncertainty on new configs triggers retraining

### Maintains Linear Constraint
- ARD is still linear regression: `alpha · features`
- Coefficients are learned, features are precomputed (including polynomials)
- Sparse coefficients set irrelevant features to zero

## Expected Benefits

| Aspect | HFD-TL (Idea 3) | BSR-ARD (This Idea) |
|--------|------------------|---------------------|
| Feature selection | Manual (3-8 features) | Automatic (18+ candidates) |
| Uncertainty | None | Per-prediction std |
| Transfer validation | None | Z-score outlier detection |
| Overfitting risk | Low (few features) | Low (ARD regularization) |
| Calibration decision | Manual | Automatic (threshold-based) |

## Training Data Requirements

**Donor configurations (one-time):**
- Same as Idea 3: 3-5 representative configurations
- ~50,000 total step observations
- ARD fitting takes ~1 minute (200 iterations)

**Target configurations (per new config):**
- 500-1000 observations for transfer validation
- If transfer passes: only bias/scale calibration needed
- If transfer fails: add to donor pool and retrain

## Limitations and Mitigations

**Limitation 1: ARD may be computationally expensive at scale**
- Mitigation: Pre-compute covariance matrix offline; inference is just matrix-vector product
- Alternative: Use approximate inference (variational Bayes) if needed

**Limitation 2: Gaussian assumption may not hold for heavy-tailed latency distributions**
- Mitigation: Log-transform latency to normalize distribution
- Alternative: Use robust regression (Student-t likelihood) if Gaussian fails

**Limitation 3: Feature engineering still required for candidate set**
- Mitigation: Include exhaustive physics-informed candidates; ARD will prune
- The candidate set is a superset of all features from Ideas 1-3
