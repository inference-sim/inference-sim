# Background

## vLLM Tracing and Performance Modeling Summary

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

## Background on BLIS

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

# Problem Statement

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

# Iteration 1

## Idea 1: Hierarchical Mixture of Experts Coefficient Model (HMoE-Coef)

### Core Concept
Instead of learning a single alpha and beta coefficient vector, learn a **hierarchical mixture model** where the top level gates on hardware/model class, and leaf experts specialize in workload patterns.

### Architecture
```
                    ┌─────────────────┐
                    │  Gating Network │ ← (hw_type, model_arch, tp_degree)
                    │   (top level)   │
                    └────────┬────────┘
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Expert_A │  │ Expert_B │  │ Expert_C │
        │(prefill) │  │(decode)  │  │(mixed)   │
        └──────────┘  └──────────┘  └──────────┘
               │             │             │
               └─────────────┴─────────────┘
                             ▼
                    Weighted α, β vectors
```

### Alpha Features (arrival-to-queuing)
- **F1**: `queue_depth / max_num_seqs` (normalized queue pressure)
- **F2**: `kv_usage_ratio` (memory pressure indicator)
- **F3**: `prompt_tokens / chunk_size` (chunked prefill iterations needed)
- **F4**: `running_batch_tokens / max_batched_tokens` (batch headroom)
- **F5**: `is_prefix_cacheable` (binary: can leverage cached KV)

### Beta Features (step time)
- **F1**: `prefill_tokens * num_layers / (compute_tflops * tp_degree)` (compute-bound estimate)
- **F2**: `decode_tokens * kv_bytes_per_token / memory_bandwidth` (memory-bound estimate)
- **F3**: `batch_size` (attention overhead scaling)
- **F4**: `is_moe * num_active_experts / total_experts` (MoE activation sparsity)
- **F5**: `1.0 if chunked_prefill_enabled else 0.0` (mixed batch indicator)

### Training Pipeline
1. **Data Collection**: Parse OTEL traces into (features, latency) pairs
2. **Expert Assignment**: EM algorithm assigns samples to experts based on workload signature
3. **Gating Training**: Softmax classifier on (hw, model, tp) to predict expert weights
4. **Expert Training**: Ridge regression per expert for robustness
5. **Validation**: Temporal holdout + cross-hardware validation

### Why This Meets Constraints
- Maintains `alpha · F1` and `beta · F2` structure (each expert is a linear model)
- Handles diversity via gating (different experts for A100 vs H100, dense vs MoE)
- Training from traces: all features derivable from step/journey events + configs
- Robustness via: (a) EM prevents hard clustering mistakes, (b) Ridge regularization, (c) multiple experts reduce per-expert variance

## Iteration 1 Review Summary

### Scores
| Judge | Model | Score | Verdict |
|-------|-------|-------|---------|
| Judge 1 (Opus) | claude-opus-4-6 | 6/10 | Promising with structural flaws |
| Judge 2 (Sonnet) | claude-sonnet | 3.8/5 (7.6/10) | Strong foundation, needs refinement |
| Judge 3 (Haiku) | claude-haiku | Approve with contingencies | Technically viable |

### Consensus Strengths
1. **Sound core intuition**: Mixture of experts is principled for capturing hardware/model/workload diversity
2. **Maintains linearity constraint**: `alpha · F` and `beta · F` structure preserved within each expert
3. **Features mostly traceable**: Alpha and beta features largely derivable from vLLM OTEL traces
4. **Ridge regularization**: Prevents overfitting within each expert

### Critical Issues Identified (All Judges)

**1. Gating/Expert Axis Mismatch (Judge 1 - CRITICAL)**
- Gating inputs are deployment-level (hw_type, model_arch, tp_degree)
- Experts specialize by workload pattern (prefill/decode/mixed)
- These are orthogonal dimensions — deployment features cannot determine workload pattern
- **Fix**: Gate on workload-level features OR restructure expert specialization

**2. Alpha F5 Data Leakage (Judges 1, 3)**
- `is_prefix_cacheable` unknown at queue time (determined at scheduling)
- Using post-hoc cache hit information leaks scheduling decisions
- **Fix**: Replace with proxy (e.g., `prefix_length / prompt_length`, historical cache hit rate)

**3. Deployment-Constant Beta Features (Judges 1, 2)**
- F4 (MoE sparsity) and F5 (chunked_prefill) are constants per deployment
- Provide no discriminative power within a single deployment's traces
- **Fix**: Require multi-deployment training corpus explicitly, or add per-step features

**4. Missing Diversity Dimensions (All Judges)**
- Expert parallelism (EP) — not mentioned
- CPU offloading — not modeled
- `max-model-len` effects — not captured
- `gpu-memory-utilization` — not mentioned
- Preemption effects on alpha latency — not modeled

**5. EM Pipeline Underspecified (Judges 2, 3)**
- No initialization strategy (risk of expert collapse)
- No convergence criteria
- No hyperparameter tuning methodology
- No handling of sample imbalance across experts

**6. Cold Start Problem (Judges 1, 2)**
- No fallback for unseen hardware/model combinations
- Need transfer learning or default expert strategy

### Recommendations for Next Iteration
1. Restructure to two-level hierarchy: deployment-level gating → workload-level gating
2. Replace problematic features (Alpha F5, add per-step varying features)
3. Add missing dimensions: EP, CPU offloading, preemption count, gpu-memory-utilization
4. Specify EM initialization (K-means++), convergence criteria, and model selection
5. Add cold-start fallback mechanism
6. Make roofline-embedding in Beta F1/F2 explicit (learning correction factors)

---

# Iteration 2

## Idea 2: Roofline-Corrected Regime Model (RCRM)

### Core Concept
Instead of learning coefficients from scratch, **use the roofline model as a physics-informed prior** and learn **multiplicative correction factors** that capture regime-specific deviations. The model explicitly acknowledges that step time is fundamentally roofline-bounded, and alpha latency follows queuing theory principles.

### Key Insight from Iteration 1 Feedback
The gating/expert mismatch arose from conflating deployment-level and workload-level features. RCRM separates these cleanly:
- **Deployment regime** (hw, model, tp, ep) → selects correction factor lookup table
- **Per-step features** (batch composition, KV pressure) → inputs to corrected roofline formula

### Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                     ALPHA MODEL                               │
│  α_latency = γ_α(regime) × QueueTheory(F_queue)              │
│                                                               │
│  where QueueTheory = M/M/1 approx:                           │
│    E[wait] ∝ ρ/(1-ρ), ρ = arrival_rate × service_time       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                     BETA MODEL                                │
│  β_latency = γ_β(regime) × Roofline(F_step) + Δ(F_step)     │
│                                                               │
│  where:                                                       │
│    γ_β(regime) = learned correction factor per regime        │
│    Roofline(F_step) = max(compute_time, memory_time)         │
│    Δ(F_step) = linear residual for unmodeled effects         │
└──────────────────────────────────────────────────────────────┘
```

### Regime Definition
A regime is a tuple: `(hardware_class, model_class, parallelism_config, vllm_mode)`

| Dimension | Values | Source |
|-----------|--------|--------|
| hardware_class | {A100, H100, B200} | hardware_config.json |
| model_class | {dense_small, dense_large, moe} | config.json (architecture + param count) |
| parallelism_config | {tp1, tp2, tp4, tp8} × {ep1, ep2, ep4} | vLLM args |
| vllm_mode | {eager, chunked, offload} | vLLM args |

**Cold-start handling**: For unseen regimes, interpolate γ from nearest neighbors in feature space (using hardware FLOPS/BW ratio, model params, parallelism degree as continuous coordinates).

### Alpha Features (Queue-Time Observable)

| Feature | Formula | Trace Source |
|---------|---------|--------------|
| F1 | `queue_depth / max_num_seqs` | step.queue.waiting_depth / config |
| F2 | `kv_usage_ratio` | step.kv.usage_gpu_ratio |
| F3 | `prompt_tokens / max_model_len` | request.prefill_total_tokens / config |
| F4 | `recent_service_rate` | 1 / moving_avg(step.duration_us) from last 10 steps |
| F5 | `preemption_pressure` | count(preemptions in last 10 steps) / 10 |
| F6 | `effective_arrival_rate` | requests_arrived_last_1s |

**Queuing theory formulation:**
```
ρ = F6 × (1/F4)  # utilization = arrival_rate × avg_service_time
base_wait = ρ / (1 - ρ) × (1/F4)  # M/M/1 expected wait
α_latency = γ_α(regime) × base_wait × (1 + w1*F1 + w2*F2 + w3*F3 + w4*F5)
```

### Beta Features (Per-Step)

| Feature | Formula | Trace Source |
|---------|---------|--------------|
| F1 | `prefill_tokens` | step.batch.prefill_tokens |
| F2 | `decode_tokens` | step.batch.decode_tokens |
| F3 | `batch_size` | step.queue.running_depth |
| F4 | `kv_blocks_accessed` | sum(request.kv.blocks_allocated_gpu) from rich trace |
| F5 | `num_preempted_this_step` | step.batch.num_preempted |
| F6 | `chunked_prefill_active` | 1 if any request in partial prefill, else 0 |

**Roofline formulation:**
```
compute_time = (prefill_flops(F1) + decode_flops(F2, F3)) / (hardware.tflops × tp × mfu)
memory_time = (weight_bytes + kv_bytes(F4)) / (hardware.bandwidth × bw_eff)
roofline_estimate = max(compute_time, memory_time)

β_latency = γ_β(regime) × roofline_estimate + δ · [F3, F5, F6]
```
Where δ is a small learned residual vector for batch overhead, preemption overhead, and chunked-prefill scheduling overhead.

### Training Pipeline

**Stage 1: Roofline Calibration (per hardware)**
- Run microbenchmarks to measure actual MFU and bandwidth efficiency
- Produces hardware-specific `mfu` and `bw_eff` constants

**Stage 2: Regime Factor Learning**
- For each regime with sufficient data (>500 step samples):
  - Compute roofline predictions for all samples
  - Learn γ_β = median(actual_step_time / roofline_prediction)
  - Learn γ_α similarly from queue time samples
- For regimes with sparse data:
  - Use Gaussian Process regression on regime features to interpolate γ

**Stage 3: Residual Learning**
- Pool all samples, fit linear model for δ coefficients
- Use Lasso regularization to select only significant residual features

**Validation:**
- Temporal holdout (train on first 80% of trace timeline, test on last 20%)
- Leave-one-regime-out cross-validation for interpolation quality
- Stress test on high-preemption and high-KV-pressure scenarios

### Why This Addresses Iteration 1 Feedback

| Issue | How Addressed |
|-------|---------------|
| Gating/expert mismatch | No gating — regime is determined by deployment config, not learned |
| Alpha F5 data leakage | Removed; replaced with `preemption_pressure` (historical, observable) |
| Deployment-constant features | Separated into regime lookup; per-step features are all dynamic |
| Missing EP | Included in regime definition |
| Missing CPU offload | Included in vllm_mode dimension |
| Missing preemption | Added F5 (alpha), F5 (beta) for preemption effects |
| EM underspecification | No EM needed — regime assignment is deterministic from config |
| Cold-start | GP interpolation on regime feature space |
| Roofline connection | Made explicit — learning correction factors on physics prior |

### Maintains Constraint Structure
- Alpha: `γ_α × f(queue_features)` — still a function of feature vector, multiplicative form
- Beta: `γ_β × roofline + δ · F` — affine in features, roofline provides structure

## Iteration 2 Review Summary

### Scores
| Judge | Model | Score | Verdict |
|-------|-------|-------|---------|
| Judge 1 (Opus) | claude-opus-4-6 | 7.5/10 | Strong advance, constraint compliance concerns |
| Judge 2 (Sonnet) | claude-sonnet | 8.4/10 | APPROVE WITH CONDITIONS |
| Judge 3 (Haiku) | claude-haiku | STRONG APPROVE | Contingencies on calibration details |

**Average: 7.9/10** (vs 6.5/10 for Iteration 1) — Significant improvement

### Consensus Strengths
1. **Physics-informed prior**: Roofline as baseline is principled, data-efficient, interpretable
2. **Deterministic regime assignment**: Eliminates gating/expert mismatch completely
3. **Clean separation**: Deployment-level (regime) vs per-step (dynamic) features
4. **Iteration 1 feedback**: Addresses 9/9 critical issues comprehensively
5. **Multiplicative + residual decomposition**: Correctly separates efficiency (γ) from overhead (δ)

### Critical Issues Identified

**1. Dot-Product Constraint Deviation (Judge 1 - HIGH PRIORITY)**
- Alpha uses M/M/1 formula `ρ/(1-ρ)` — rational function, not dot product
- Beta uses `max(compute, memory)` — piecewise linear, not dot product
- **Fix**: Provide linearized variant: precompute `base_wait` and `roofline_est` as derived features, then use `α · [base_wait, base_wait×F1, ...]` and `β · [roofline_est, F3, F5, F6]`

**2. M/M/1 Queuing Assumptions (All Judges - MEDIUM)**
- Assumes Poisson arrivals and exponential service times
- Real LLM workloads: bursty arrivals, bimodal service times
- Edge case: `ρ ≥ 1` causes divergence — needs clamping
- **Fix**: Either validate M/M/1 fit empirically, use M/G/1 with measured variance, or add `ρ = min(ρ, 0.99)` clamping

**3. Regime Space Sparsity (Judges 1, 2 - MEDIUM)**
- ~324 potential regimes, need >500 samples each = 162,000+ steps
- Many regimes will be data-sparse, relying heavily on GP interpolation
- **Fix**: Quantify expected regime coverage, collapse low-impact dimensions, consider transfer learning

**4. GP Kernel Underspecified (All Judges - MEDIUM)**
- Mixed categorical (hw, vllm_mode) + continuous (FLOPS, params, TP) features
- Need explicit kernel: `k_categorical × k_continuous`
- **Fix**: Specify kernel (RBF + Matern for continuous, Hamming for categorical)

**5. Stage 1 Microbenchmarking Details (Judges 2, 3 - MEDIUM)**
- No specification of which benchmarks, measurement duration, drift handling
- **Fix**: Provide concrete playbook (vLLM profiler or custom GEMM benchmarks)

**6. Rolling-Window State in Simulation (Judge 1 - LOW)**
- Alpha F4, F5, F6 require temporal state (moving averages, rolling counts)
- Current BLIS architecture doesn't support this
- **Fix**: Document integration into RouterState/InstanceSnapshot

### Recommendations for Next Iteration
1. **Linearized formulation**: Provide dot-product-compliant variant as primary, nonlinear as optional
2. **Sparse regime handling**: Use simpler interpolation (IDW) or hierarchical regimes
3. **Precomputed derived features**: `base_wait` and `roofline_est` as first-class features
4. **Explicit GP kernel**: Specify for mixed feature types
5. **ρ clamping**: Add saturation behavior for overloaded scenarios
6. **Per-regime-group residuals**: Instead of single pooled δ

---

# Iteration 3

## Idea 3: Linearized Physics-Informed Coefficients (LPIC)

### Core Concept
Address the dot-product constraint deviation by **precomputing physics-based derived features** and using them in a strictly linear model. The roofline and queuing theory insights from RCRM are preserved as **feature engineering**, not as inline computation.

### Key Insight from Iteration 2 Feedback
The constraint `α · F` requires a fixed coefficient vector and fixed feature vector. RCRM violated this with M/M/1 rational functions and max() operators. LPIC moves these computations into **derived features** computed before prediction, making the final model a pure dot product.

### Architecture
```
┌────────────────────────────────────────────────────────────────────┐
│                    FEATURE DERIVATION LAYER                         │
│  (Computed once per prediction, not part of learned model)          │
│                                                                     │
│  Derived Alpha Features:                                            │
│    D1 = base_wait(ρ) = min(ρ, 0.95) / (1 - min(ρ, 0.95)) / svc_rate│
│    D2 = D1 × queue_pressure                                         │
│    D3 = D1 × kv_pressure                                            │
│    D4 = D1 × prompt_ratio                                           │
│    D5 = preemption_pressure                                         │
│                                                                     │
│  Derived Beta Features:                                             │
│    D1 = roofline_prefill = prefill_tokens × layers / (tflops × tp) │
│    D2 = roofline_decode = decode_tokens × kv_bytes / bandwidth      │
│    D3 = roofline_max = max(D1, D2)                                  │
│    D4 = batch_overhead = batch_size                                 │
│    D5 = preemption_overhead = num_preempted                         │
│    D6 = chunked_overhead = chunked_active                           │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                    LINEAR COEFFICIENT MODEL                         │
│                                                                     │
│  α_latency = α · [D1, D2, D3, D4, D5]         (dot product)        │
│  β_latency = β · [D1, D2, D3, D4, D5, D6]     (dot product)        │
│                                                                     │
│  where α, β are regime-specific coefficient vectors                 │
└────────────────────────────────────────────────────────────────────┘
```

### Regime Hierarchy (Simplified)
To address regime sparsity, use a **two-level hierarchy** instead of flat enumeration:

**Level 1: Hardware Class** (3 values)
- Determines base compute/memory characteristics
- Values: {GPU_Ampere, GPU_Hopper, GPU_Blackwell}

**Level 2: Workload Class** (4 values)
- Determined at runtime from batch composition
- Values: {prefill_dominant, decode_dominant, mixed, memory_bound}

**Total: 12 regime combinations** (vs 324 in RCRM)

Classification rule for workload class:
```python
def classify_workload(prefill_tokens, decode_tokens, kv_usage):
    if kv_usage > 0.85:
        return "memory_bound"
    ratio = prefill_tokens / max(decode_tokens, 1)
    if ratio > 5:
        return "prefill_dominant"
    elif ratio < 0.2:
        return "decode_dominant"
    else:
        return "mixed"
```

### Alpha Feature Derivation (Queue-Time)

**Raw features** (directly observable):
| Feature | Source |
|---------|--------|
| R1: queue_depth | step.queue.waiting_depth |
| R2: max_num_seqs | config |
| R3: kv_usage | step.kv.usage_gpu_ratio |
| R4: prompt_tokens | request.prefill_total_tokens |
| R5: max_model_len | config |
| R6: recent_svc_rate | 1 / EMA(step.duration_us, α=0.1) |
| R7: arrival_rate | EMA(arrivals_per_second, α=0.1) |
| R8: preemption_count | count(preemptions, last 10 steps) |

**Derived features** (physics-informed transforms):
```python
# Utilization with clamping (prevents M/M/1 divergence)
rho = min(R7 / R6, 0.95)  # clamp at 0.95

# Base wait from queuing theory (precomputed, not inline)
D1_base_wait = (rho / (1 - rho)) / R6  # seconds

# Interaction terms (all linear in D1)
D2_queue_interaction = D1_base_wait * (R1 / R2)
D3_kv_interaction = D1_base_wait * R3
D4_prompt_interaction = D1_base_wait * (R4 / R5)

# Direct feature (no interaction)
D5_preemption = R8 / 10.0
```

**Final alpha model:**
```
α_latency = α₁×D1 + α₂×D2 + α₃×D3 + α₄×D4 + α₅×D5
```
This is a proper dot product. The α vector is regime-specific.

### Beta Feature Derivation (Per-Step)

**Raw features:**
| Feature | Source |
|---------|--------|
| R1: prefill_tokens | step.batch.prefill_tokens |
| R2: decode_tokens | step.batch.decode_tokens |
| R3: batch_size | step.queue.running_depth |
| R4: num_preempted | step.batch.num_preempted |
| R5: chunked_active | 1 if partial prefill in batch else 0 |
| R6: kv_blocks | step.kv.blocks_total - step.kv.blocks_free |

**Hardware constants** (from calibration):
```python
tflops = hardware.compute_tflops  # e.g., 989 for H100
bandwidth = hardware.memory_bandwidth  # e.g., 3.35 TB/s for H100
layers = model.num_hidden_layers
kv_bytes_per_token = model.hidden_size * model.num_kv_heads / model.num_attn_heads * 2 * 2
tp = config.tensor_parallel_size
```

**Derived features:**
```python
# Roofline estimates (precomputed)
D1_roofline_prefill = R1 * layers * flops_per_token / (tflops * tp * 1e12)
D2_roofline_decode = R2 * kv_bytes_per_token * R6 / (bandwidth * 1e12)
D3_roofline_max = max(D1_roofline_prefill, D2_roofline_decode)

# Overhead terms (linear)
D4_batch_overhead = R3 / 100.0  # normalized
D5_preemption_overhead = R4
D6_chunked_overhead = R5
```

**Final beta model:**
```
β_latency = β₁×D1 + β₂×D2 + β₃×D3 + β₄×D4 + β₅×D5 + β₆×D6
```
This is a proper dot product. The β vector is regime-specific.

### Training Pipeline

**Stage 1: Hardware Calibration** (unchanged from RCRM)
- Microbenchmark MFU and bandwidth efficiency
- One-time per hardware class

**Stage 2: Regime Coefficient Learning**
For each of 12 regimes:
```python
# Collect samples assigned to this regime
samples = filter_by_regime(all_samples, hw_class, workload_class)

if len(samples) >= 100:
    # Direct estimation with Ridge regression
    alpha_coeffs = Ridge(alpha=0.1).fit(alpha_derived_features, alpha_targets)
    beta_coeffs = Ridge(alpha=0.1).fit(beta_derived_features, beta_targets)
else:
    # Fallback: use parent regime (hardware class only)
    alpha_coeffs = hw_class_defaults[hw_class].alpha
    beta_coeffs = hw_class_defaults[hw_class].beta
```

**Stage 3: Hierarchical Fallback Learning**
Train hardware-class-level defaults for sparse workload classes:
```python
for hw_class in [Ampere, Hopper, Blackwell]:
    hw_samples = filter_by_hardware(all_samples, hw_class)
    hw_class_defaults[hw_class].alpha = Ridge().fit(...)
    hw_class_defaults[hw_class].beta = Ridge().fit(...)
```

**Validation:**
- Temporal holdout (80/20 split by time)
- Leave-one-hardware-out for transfer assessment
- Per-regime error analysis

### Why This Addresses Iteration 2 Feedback

| Issue | How Addressed |
|-------|---------------|
| Dot-product constraint | Derived features are precomputed; final model is pure `α·D` and `β·D` |
| M/M/1 divergence | ρ clamped at 0.95 in derived feature computation |
| Regime sparsity (324→12) | Two-level hierarchy with runtime workload classification |
| GP kernel complexity | Eliminated — simple Ridge + hierarchical fallback |
| Rolling-window state | EMA with α=0.1 is simple to implement in simulator |
| Roofline as feature | D1-D3 in beta explicitly use roofline as derived features |

### Integration with BLIS

**SimConfig additions:**
```go
type LPICConfig struct {
    HardwareClass   string           // "Ampere", "Hopper", "Blackwell"
    AlphaCoeffs     map[string][]float64  // workload_class -> [α1..α5]
    BetaCoeffs      map[string][]float64  // workload_class -> [β1..β6]
    HWDefaults      LPICCoeffs       // fallback coefficients
}
```

**RouterState additions** (for rolling statistics):
```go
type LPICState struct {
    SvcRateEMA      float64  // exponential moving average of 1/step_duration
    ArrivalRateEMA  float64  // EMA of arrivals per second
    PreemptWindow   []int    // circular buffer of last 10 preemption counts
}
```

**Prediction flow:**
1. Classify workload from current batch composition
2. Look up regime coefficients (or use HW defaults)
3. Compute derived features from raw observables + calibration constants
4. Return `α · D_alpha` and `β · D_beta`

### Complexity Comparison

| Aspect | HMoE-Coef | RCRM | LPIC |
|--------|-----------|------|------|
| Constraint compliant | Yes (by design) | No (max, rational) | **Yes** |
| Regimes | 3 experts | ~324 | **12** |
| Training algorithm | EM + Ridge | Median + GP + Lasso | **Ridge only** |
| Cold-start handling | None | GP interpolation | **Hierarchical fallback** |
| Interpretability | Low (mixture weights) | High (γ corrections) | **High** (coefficients) |
| Implementation complexity | Medium-High | High | **Low-Medium** |

## Iteration 3 Review Summary

### Scores
| Judge | Model | Score | Verdict |
|-------|-------|-------|---------|
| Judge 1 (Opus) | claude-opus-4-6 | 8/10 | Most implementation-ready proposal |
| Judge 2 (Sonnet) | claude-sonnet | 7.8/10 | APPROVE WITH CONDITIONS |
| Judge 3 (Haiku) | claude-haiku | STRONG APPROVE | Excellent constraint compliance |

**Average: 7.9/10** (Iteration 1: 6.5, Iteration 2: 7.9) — Best constraint compliance yet

### Consensus Strengths
1. **Constraint compliant**: Pure dot product via precomputed derived features
2. **Dramatic regime reduction**: 324→12 regimes (27× simpler)
3. **Simple training**: Ridge only, no EM/GP complexity
4. **Hierarchical fallback**: Elegant cold-start handling
5. **Implementation-ready**: Concrete Go structs and integration flow

### Critical Issues Identified

**1. Lost Diversity Dimensions (Judges 1, 2 - HIGH PRIORITY)**
- TP/EP degree collapsed into derived features only
- MoE-specific effects not captured (routing overhead, expert imbalance)
- vLLM mode (eager/chunked/offload) partially captured
- **Fix**: Either expand to 24 regimes (add model_type) or add MoE-specific features

**2. Beta Multicollinearity (Judges 1, 3 - HIGH)**
- D3 = max(D1, D2) creates perfect collinearity
- Ridge handles numerics but interpretation suffers
- **Fix**: Replace [D1, D2, D3] with [max(D1,D2), min(D1,D2)] or drop D3

**3. Missing Intercept Terms (Judge 1 - HIGH)**
- No constant/bias term in alpha or beta
- Low-load predictions degenerate (D1→0 makes D2-D4→0)
- **Fix**: Add D0=1.0 to both feature vectors

**4. Workload Thresholds Arbitrary (All Judges - MEDIUM)**
- ratio>5, ratio<0.2, kv>0.85 not empirically validated
- **Fix**: Data-driven threshold tuning via clustering

**5. Feature Standardization Missing (Judges 1, 2 - MEDIUM)**
- Derived features span vastly different scales
- Ridge sensitive to scaling
- **Fix**: Add z-score standardization, store scaler params

**6. Ridge Hyperparameter Fixed (Judges 2, 3 - MEDIUM)**
- α=0.1 hardcoded without justification
- **Fix**: Cross-validate per regime

### Recommendations for Next Iteration
1. Add intercept (D0=1.0) to both alpha and beta
2. Handle MoE models explicitly (regime dimension or features)
3. Fix multicollinearity: [roofline_max, roofline_min] or drop D3
4. Add TP communication overhead feature
5. Feature standardization + CV for Ridge alpha
6. Consider hybrid: LPIC simplicity + RCRM's cross-hardware GP interpolation

---

# Iteration 4

## Idea 4: LPIC+ (Enhanced Linearized Physics-Informed Coefficients)

### Core Concept
Refine LPIC by addressing all critical feedback: add intercepts, fix multicollinearity, handle MoE explicitly, and add cross-hardware interpolation. Maintains constraint compliance while recovering lost diversity dimensions.

### Key Changes from LPIC

| Issue | LPIC | LPIC+ |
|-------|------|-------|
| Intercept | Missing | D0=1.0 added to both α and β |
| Multicollinearity | D3=max(D1,D2) overlaps | Replaced with [roofline_bottleneck, roofline_ratio] |
| MoE handling | Implicit in roofline | Explicit regime dimension + MoE features |
| TP communication | Missing | D7_comm added to beta |
| Standardization | Not specified | Explicit z-score + stored scalers |
| Cross-hardware | Hierarchical fallback only | Fallback + TFLOPs-ratio interpolation |

### Regime Hierarchy (24 Regimes)

**Level 1: Hardware Class** (3 values)
- {GPU_Ampere, GPU_Hopper, GPU_Blackwell}

**Level 2: Model Class** (2 values)
- {Dense, MoE}

**Level 3: Workload Class** (4 values, runtime-determined)
- {prefill_dominant, decode_dominant, mixed, memory_bound}

**Total: 3 × 2 × 4 = 24 regimes** (vs 12 in LPIC, vs 324 in RCRM)

### Alpha Feature Vector (7 features)

```python
# Raw observables
R1 = queue_depth                    # step.queue.waiting_depth
R2 = max_num_seqs                   # config
R3 = kv_usage                       # step.kv.usage_gpu_ratio
R4 = prompt_tokens                  # request.prefill_total_tokens
R5 = max_model_len                  # config
R6 = svc_rate                       # EMA(1/step.duration_us)
R7 = arrival_rate                   # EMA(arrivals/second)
R8 = preemption_count               # rolling window, last 10 steps

# Derived features (standardized)
rho = min(R7 / R6, 0.95)
base_wait = (rho / (1 - rho)) / R6

D0 = 1.0                            # INTERCEPT (new)
D1 = standardize(base_wait)
D2 = standardize(base_wait * (R1 / R2))  # queue interaction
D3 = standardize(base_wait * R3)          # KV interaction
D4 = standardize(base_wait * (R4 / R5))   # prompt interaction
D5 = standardize(R8 / 10.0)               # preemption (direct)
D6 = standardize(R1 / R2)                 # queue pressure (uninteracted, new)

α_latency = [α₀..α₆] · [D0..D6]
```

**Key improvements:**
- D0 (intercept): captures minimum scheduling delay at low load
- D6 (raw queue pressure): provides signal when D1→0

### Beta Feature Vector (9 features)

```python
# Raw observables
R1 = prefill_tokens                 # step.batch.prefill_tokens
R2 = decode_tokens                  # step.batch.decode_tokens
R3 = batch_size                     # step.queue.running_depth
R4 = num_preempted                  # step.batch.num_preempted
R5 = chunked_active                 # 1 if partial prefill in batch
R6 = kv_blocks_used                 # step.kv.blocks_total - blocks_free
R7 = tp_degree                      # config.tensor_parallel_size
R8 = is_moe                         # 1 if MoE model else 0
R9 = active_experts                 # num active experts (MoE only)

# Hardware/model constants (from calibration)
tflops = hardware.compute_tflops
bandwidth = hardware.memory_bandwidth
layers = model.num_hidden_layers
kv_bytes = model.kv_bytes_per_token
total_experts = model.num_experts   # 1 for dense

# Roofline estimates
roofline_prefill = R1 * layers * flops_per_token / (tflops * R7 * 1e12)
roofline_decode = R2 * kv_bytes * R6 / (bandwidth * 1e12)

# Derived features (no multicollinearity)
D0 = 1.0                                    # INTERCEPT (new)
D1 = standardize(max(roofline_prefill, roofline_decode))  # bottleneck
D2 = standardize(min(roofline_prefill, roofline_decode) /
                 max(roofline_prefill, roofline_decode, 1e-9))  # ratio (new)
D3 = standardize(R3 / max_num_seqs)         # normalized batch size
D4 = standardize(R4)                         # preemption overhead
D5 = standardize(R5)                         # chunked overhead
D6 = standardize((R7 - 1) * layers * msg_size / interconnect_bw)  # TP comm (new)
D7 = standardize(R8 * R9 / total_experts * R3)  # MoE routing (new)
D8 = standardize(R3 * R3 / 10000)            # attention quadratic (new)

β_latency = [β₀..β₈] · [D0..D8]
```

**Key improvements:**
- D0 (intercept): captures minimum kernel launch / scheduler overhead
- D1, D2 (bottleneck + ratio): no multicollinearity, captures both "which bound" and "how close"
- D6 (TP communication): captures AllReduce overhead missing from LPIC
- D7 (MoE routing): per-batch expert activation cost
- D8 (attention quadratic): captures O(batch²) attention scaling

### Standardization Protocol

```python
class FeatureScaler:
    def __init__(self):
        self.mean = {}    # feature_name -> mean
        self.std = {}     # feature_name -> std

    def fit(self, regime_samples):
        """Compute mean/std from training data per regime."""
        for feature in features:
            self.mean[feature] = np.mean(regime_samples[feature])
            self.std[feature] = np.std(regime_samples[feature]) + 1e-8

    def transform(self, x, feature):
        """Standardize a feature value."""
        return (x - self.mean[feature]) / self.std[feature]
```

**Storage:** Scaler parameters stored alongside coefficients in regime config.

### Training Pipeline

**Stage 1: Hardware Calibration** (same as LPIC)
- Microbenchmark MFU and bandwidth efficiency
- Measure AllReduce latency for interconnect_bw

**Stage 2: Feature Scaler Fitting**
```python
for regime in regimes:
    samples = filter_by_regime(all_samples, regime)
    regime.scaler = FeatureScaler()
    regime.scaler.fit(samples)
```

**Stage 3: Regime Coefficient Learning**
```python
for regime in regimes:
    samples = filter_by_regime(all_samples, regime)
    X = regime.scaler.transform(raw_features)  # standardized

    if len(samples) >= 100:
        # Cross-validate Ridge alpha
        alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
        best_alpha = cross_val_ridge(X, targets, alphas, cv=5)
        coeffs = Ridge(alpha=best_alpha).fit(X, targets)
    else:
        coeffs = fallback_coeffs(regime)

    regime.alpha_coeffs = coeffs.alpha
    regime.beta_coeffs = coeffs.beta
```

**Stage 4: Hierarchical Fallback with Interpolation**
```python
def fallback_coeffs(regime):
    hw_class = regime.hardware_class

    # First: try hardware-class average
    hw_samples = filter_by_hardware(all_samples, hw_class)
    if len(hw_samples) >= 50:
        return Ridge().fit(hw_samples)

    # Second: interpolate from nearest hardware class
    nearest_hw = find_nearest_by_tflops_bw_ratio(hw_class)
    scale = hw_class.tflops / nearest_hw.tflops  # scaling factor
    nearest_coeffs = get_coeffs(nearest_hw)
    return scale_coefficients(nearest_coeffs, scale)
```

**Validation:**
- Temporal holdout (80/20 by time)
- Leave-one-hardware-out
- Leave-one-workload-class-out
- Per-regime MAPE targets: β < 15%, α < 25%

### Workload Classification (Refined)

```python
def classify_workload(batch, kv_usage, training_thresholds=None):
    """
    Classify workload with empirically-tuned or default thresholds.
    """
    if training_thresholds:
        th = training_thresholds  # from k-means on training data
    else:
        th = DEFAULT_THRESHOLDS  # kv=0.85, prefill_ratio=5, decode_ratio=0.2

    if kv_usage > th.kv_bound:
        return "memory_bound"

    ratio = batch.prefill_tokens / max(batch.decode_tokens, 1)
    if ratio > th.prefill_dominant:
        return "prefill_dominant"
    elif ratio < th.decode_dominant:
        return "decode_dominant"
    else:
        return "mixed"

# Threshold tuning (run once on training data)
def tune_thresholds(training_samples):
    features = [sample.prefill_ratio, sample.kv_usage for sample in training_samples]
    kmeans = KMeans(n_clusters=4).fit(features)
    return derive_thresholds_from_centroids(kmeans.cluster_centers_)
```

### Integration with BLIS

```go
type LPICPlusConfig struct {
    HardwareClass   string
    ModelClass      string                   // "Dense" or "MoE" (new)
    AlphaCoeffs     map[string][]float64     // workload_class -> [α₀..α₆]
    BetaCoeffs      map[string][]float64     // workload_class -> [β₀..β₈]
    AlphaScaler     map[string]FeatureScaler // per-regime scalers (new)
    BetaScaler      map[string]FeatureScaler
    Thresholds      WorkloadThresholds       // tuned thresholds (new)
    HWDefaults      Coefficients             // fallback
    InterconnectBW  float64                  // for TP communication (new)
}

type LPICPlusState struct {
    SvcRateEMA      float64
    ArrivalRateEMA  float64
    PreemptWindow   CircularBuffer[int]
}
```

### Complexity vs Coverage Tradeoff

| Approach | Regimes | Features | Training | Constraint | Coverage |
|----------|---------|----------|----------|------------|----------|
| HMoE-Coef | 3 experts | 5+5 | EM+Ridge | ✅ | Limited |
| RCRM | ~324 | 6+6 | Median+GP+Lasso | ❌ | Full |
| LPIC | 12 | 5+6 | Ridge | ✅ | Partial |
| **LPIC+** | **24** | **7+9** | **Ridge+CV** | **✅** | **Good** |

### Why This Addresses Iteration 3 Feedback

| Issue | How Addressed |
|-------|---------------|
| Missing intercept | D0=1.0 in both α and β |
| Beta multicollinearity | [bottleneck, ratio] instead of [D1, D2, max(D1,D2)] |
| MoE not captured | model_class regime dimension + D7_moe_routing feature |
| TP comm missing | D6 for AllReduce overhead |
| Standardization | Explicit FeatureScaler with stored params |
| Ridge α hardcoded | Cross-validation from [0.01, 0.05, 0.1, 0.5, 1.0] |
| Arbitrary thresholds | k-means-based threshold tuning option |
| Cross-hardware transfer | TFLOPs-ratio interpolation in fallback |

## Iteration 4 Review Summary

### Scores
| Judge | Model | Score | Verdict |
|-------|-------|-------|---------|
| Judge 1 (Opus) | claude-opus-4-6 | 8.5/10 | Nearing convergence, minor refinements needed |
| Judge 2 (Sonnet) | claude-sonnet | 9.2/10 | STRONG APPROVE, best synthesis |
| Judge 3 (Haiku) | claude-haiku | APPROVE | Blocking clarifications on MoE observability |

**Average: 8.9/10** (Iter 1: 6.5, Iter 2: 7.9, Iter 3: 7.9, Iter 4: 8.9) — **Best score yet, approaching convergence**

### Consensus Strengths
1. **Perfect constraint compliance**: Pure dot product with elegant [bottleneck, ratio] multicollinearity fix
2. **Sweet spot complexity**: 24 regimes captures Dense/MoE distinction with 2,400 samples (vs 162k for RCRM)
3. **Comprehensive feature coverage**: D6 (TP comm), D7 (MoE routing), D8 (attention quadratic) address all gaps
4. **Robust training**: Cross-validated Ridge alpha, explicit standardization, TFLOPs interpolation
5. **All Iteration 3 feedback addressed**: 8/8 issues resolved, 7 perfectly

### Remaining Issues (Minor)

**1. D8 Physical Justification (Judges 1, 2 - MEDIUM)**
- `batch_size²/10000` claimed as "attention quadratic" but attention is O(seq_len²), not O(batch²)
- **Fix**: Replace with `batch_size × avg_seq_len²` OR validate empirically

**2. Coefficient-Specific Cross-Hardware Scaling (Judges 1, 2 - MEDIUM)**
- Single TFLOPs ratio scales all coefficients uniformly
- **Fix**: Scale compute-bound by TFLOPs, memory-bound by bandwidth, overhead by 1.0

**3. MoE active_experts Observability (Judge 3 - HIGH)**
- D7 requires `active_experts` which may not be in standard vLLM traces
- **Fix**: Clarify source or use static average

**4. Alpha D4 Interpretation (Judges 1, 2 - LOW)**
- Validate sign on real traces, document hypothesis

**5. Outlier Handling Before Scaler Fit (Judge 2 - LOW)**
- Remove outliers (>3σ) before fitting standardization scaler

### Convergence Assessment
- Improvement from Iter 3→4 (+1.0) shows continued progress
- Remaining issues are refinements, not architectural changes
- **Approaching final proposal**

---

# Iteration 5

## Idea 5: LPIC-Final (Production-Ready Linearized Physics-Informed Coefficients)

### Core Concept
Final refinement of LPIC+ addressing all remaining minor issues: fix D8 physics, add coefficient-specific interpolation, clarify MoE observability, document all feature interpretations, and add robust preprocessing.

### Key Changes from LPIC+

| Issue | LPIC+ | LPIC-Final |
|-------|-------|------------|
| D8 formula | `batch_size²/10000` (misattributed) | `batch_size × avg_seq_len² / normalizer` (correct) |
| Cross-HW interpolation | Uniform TFLOPs scaling | **Coefficient-specific scaling** |
| MoE observability | `active_experts` (unclear source) | **Static expert count from config** |
| Feature interpretation | Some unclear | **All documented with expected signs** |
| Preprocessing | Implicit | **Explicit outlier removal + warmup skip** |

### Beta Feature D8 Correction

**Problem**: LPIC+ claimed `batch_size²` captures "attention quadratic" but:
- Attention complexity is O(seq_len²) per request, not O(batch²)
- Total batch attention is Σᵢ O(seq_lenᵢ²), linear in batch size

**Corrected D8**:
```python
# Correct attention quadratic scaling
avg_seq_len_squared = mean([seq_len**2 for seq_len in batch_seq_lens])
D8 = standardize(batch_size * avg_seq_len_squared / 1e8)
```

**Interpretation**: Captures actual O(N²) attention compute per token, summed across batch.

**Alternative if seq_lens unavailable**:
```python
# Approximation using prefill/decode tokens
estimated_avg_seq_len = (prefill_tokens + decode_tokens) / batch_size
D8 = standardize(batch_size * estimated_avg_seq_len**2 / 1e8)
```

### Coefficient-Specific Cross-Hardware Interpolation

**Problem**: LPIC+ scaled all coefficients by uniform TFLOPs ratio, but:
- Compute-bound features should scale by TFLOPs ratio
- Memory-bound features should scale by bandwidth ratio
- Overhead features should not scale (hardware-independent)

**Solution**:
```python
def interpolate_coefficients(source_coeffs, source_hw, target_hw):
    tflops_ratio = target_hw.tflops / source_hw.tflops
    bw_ratio = target_hw.bandwidth / source_hw.bandwidth
    interconnect_ratio = target_hw.interconnect_bw / source_hw.interconnect_bw

    # Alpha coefficients (queue latency - mostly hardware-independent)
    alpha_interp = source_coeffs.alpha.copy()
    # α₀ (intercept): scheduler overhead, ~constant across hardware
    # α₁-α₄ (queue interactions): scale by inverse service rate improvement
    alpha_interp[1:5] *= 1.0 / max(tflops_ratio, bw_ratio)
    # α₅, α₆ (preemption, raw queue): hardware-independent

    # Beta coefficients (step latency - hardware-dependent)
    beta_interp = source_coeffs.beta.copy()
    # β₀ (intercept): kernel overhead, ~constant
    # β₁ (bottleneck): scale by max(compute, memory) improvement
    beta_interp[1] *= 1.0 / max(tflops_ratio, bw_ratio)
    # β₂ (ratio): dimensionless, no scaling
    # β₃-β₅ (batch, preempt, chunked overhead): hardware-independent
    # β₆ (TP comm): scale inversely with interconnect bandwidth
    beta_interp[6] *= 1.0 / interconnect_ratio
    # β₇ (MoE routing): hardware-independent (sparse compute)
    # β₈ (attention): scale by max(compute, memory)
    beta_interp[8] *= 1.0 / max(tflops_ratio, bw_ratio)

    return Coefficients(alpha=alpha_interp, beta=beta_interp)
```

### MoE Feature Observability Clarification

**Problem**: D7 used `active_experts` which isn't in standard vLLM traces.

**Solution**: Use static expert configuration instead of per-step activation:
```python
# D7 uses config values, not runtime trace
total_experts = model.num_experts        # from config.json (e.g., 8 for Mixtral)
top_k = model.num_experts_per_tok        # from config.json (e.g., 2 for top-2 routing)

D7 = standardize(is_moe * (top_k / total_experts) * batch_size)
```

**Rationale**:
- `top_k/total_experts` is constant per model (e.g., 2/8 = 0.25 for Mixtral)
- Captures expected activation sparsity without runtime instrumentation
- Scales with batch_size (more requests = more routing decisions)

**Limitation**: Doesn't capture load imbalance (all experts equally weighted). Acceptable for v1; can add variance term in future if data supports.

### Alpha Feature D4 Interpretation

**Hypothesis**: `prompt_ratio × base_wait` increases queue latency because:
1. Longer prompts require more contiguous KV blocks
2. Under memory pressure, scheduler delays long-prompt requests until blocks free
3. Effect is multiplicative with base_wait (congestion amplifies the effect)

**Expected coefficient sign**: **Positive** (α₄ > 0)
- Higher prompt_ratio × higher base_wait → longer queue time

**Validation procedure**:
```python
# Check coefficient sign during training
if trained_alpha[4] < 0:
    logging.warning("α₄ negative - hypothesis may be wrong, investigate")
```

### Robust Preprocessing Pipeline

**Stage 0: Data Cleaning** (NEW explicit stage)
```python
def preprocess_samples(raw_samples):
    # Step 1: Remove warmup steps
    samples = raw_samples[WARMUP_STEPS:]  # default: 10 steps

    # Step 2: Remove outliers using robust statistics
    for feature in FEATURES:
        median = np.median(samples[feature])
        mad = np.median(np.abs(samples[feature] - median))  # median absolute deviation
        threshold = 3.0 * mad * 1.4826  # scale factor for normal approximation
        samples = samples[np.abs(samples[feature] - median) < threshold]

    # Step 3: Remove high-preemption anomalies
    samples = samples[samples['num_preempted'] < MAX_PREEMPTIONS]  # default: 5

    return samples
```

**Why MAD instead of standard deviation**:
- MAD is robust to outliers (outliers don't inflate the scale estimate)
- Standard deviation is sensitive to extreme values
- 1.4826 scale factor makes MAD comparable to std for normal distributions

### Complete Feature Documentation

**Alpha Features (7 total)**:
| Feature | Formula | Expected Sign | Interpretation |
|---------|---------|---------------|----------------|
| α₀ (D0) | 1.0 | + | Minimum scheduling delay (kernel launch, context switch) |
| α₁ (D1) | base_wait | + | M/M/1 queue time; larger when utilization high |
| α₂ (D2) | base_wait × queue_pressure | + | Queue backlog amplifies wait |
| α₃ (D3) | base_wait × kv_usage | + | Memory pressure delays scheduling |
| α₄ (D4) | base_wait × prompt_ratio | + | Long prompts harder to schedule |
| α₅ (D5) | preemption_pressure | + | Preemption causes re-queuing |
| α₆ (D6) | queue_pressure | + | Raw queue signal for low-load |

**Beta Features (9 total)**:
| Feature | Formula | Expected Sign | Interpretation |
|---------|---------|---------------|----------------|
| β₀ (D0) | 1.0 | + | Minimum step overhead (kernel launch) |
| β₁ (D1) | roofline_bottleneck | + | Dominant compute/memory time |
| β₂ (D2) | roofline_ratio | +/- | Balance indicator (model-dependent) |
| β₃ (D3) | batch_size_norm | + | Attention overhead scales with batch |
| β₄ (D4) | num_preempted | + | Preemption has scheduling cost |
| β₅ (D5) | chunked_active | + | Mixed batches have overhead |
| β₆ (D6) | tp_comm | + | AllReduce latency |
| β₇ (D7) | moe_routing | + | Expert selection overhead |
| β₈ (D8) | attention_quadratic | + | O(seq_len²) attention compute |

### Training Pipeline Summary

```
Stage 0: Preprocessing
├── Remove warmup steps (first 10)
├── Remove outliers (MAD-based, 3σ)
└── Remove high-preemption anomalies

Stage 1: Hardware Calibration
├── Measure MFU (microbenchmark)
├── Measure bandwidth efficiency
└── Measure AllReduce latency

Stage 2: Feature Scaler Fitting
├── Fit per-regime standardization scaler
└── Store mean/std in config

Stage 3: Regime Coefficient Learning
├── Cross-validate Ridge α (5-fold)
├── Fit Ridge regression per regime
└── Validate coefficient signs

Stage 4: Hierarchical Fallback
├── Hardware-class average (if regime sparse)
└── Coefficient-specific interpolation (if HW class sparse)

Stage 5: Validation
├── Temporal holdout (80/20)
├── Leave-one-hardware-out
├── Leave-one-workload-class-out
└── Per-regime MAPE targets (β<15%, α<25%)
```

### Final Specification Summary

| Aspect | Value |
|--------|-------|
| Regimes | 24 (3 hw × 2 model × 4 workload) |
| Alpha features | 7 (including intercept) |
| Beta features | 9 (including intercept) |
| Constraint | ✅ Pure dot product |
| Training | Ridge + 5-fold CV |
| Fallback | Hierarchical + coefficient-specific interpolation |
| Data requirement | ~2,400 steps minimum |
| Implementation | ~4-5 weeks |

## Iteration 5 Review Summary

### Scores
| Judge | Model | Score | Verdict |
|-------|-------|-------|---------|
| Judge 1 (Opus) | claude-opus-4-6 | 9/10 | APPROVE for implementation |
| Judge 2 (Sonnet) | claude-sonnet | 9.6/10 | APPROVE FOR IMPLEMENTATION |
| Judge 3 (Haiku) | claude-haiku | 9.5+/10 | PROCEED IMMEDIATELY |

**Average: 9.4/10** — **UNANIMOUS APPROVAL, DESIGN CONVERGED**

### All Iteration 4 Concerns Resolved

| Issue | Status | Solution |
|-------|--------|----------|
| D8 physics (batch² → seq_len²) | ✅ RESOLVED | `batch_size × avg_seq_len² / 1e8` |
| Uniform cross-HW scaling | ✅ RESOLVED | Coefficient-specific scaling per category |
| MoE active_experts observability | ✅ RESOLVED | Static `top_k/total_experts` from config |
| Feature interpretation | ✅ RESOLVED | All 16 features documented with expected signs |
| Outlier handling | ✅ RESOLVED | MAD-based + warmup removal + preemption filter |

### Remaining Items (LOW Priority, Implementation Details)

1. **EP coverage**: Deferred to v2 (uncommon outside large-scale MoE)
2. **flops_per_token derivation**: Specify once during implementation
3. **D8 prefill vs decode**: Note O(seq_len²) applies mainly to prefill
4. **msg_size for D6**: Specify as `hidden_size × dtype_bytes × 2 / tp`
5. **EMA decay validation**: Verify α=0.1 is optimal window

### Convergence Assessment

| Iteration | Score | Delta | Status |
|-----------|-------|-------|--------|
| 1: HMoE-Coef | 6.5/10 | — | Foundational |
| 2: RCRM | 7.9/10 | +1.4 | Physics-grounded |
| 3: LPIC | 7.9/10 | +0.0 | Constraint-compliant |
| 4: LPIC+ | 8.9/10 | +1.0 | Comprehensive |
| **5: LPIC-Final** | **9.4/10** | **+0.5** | **Production-ready** |

**Convergence confirmed**: Delta shrinking (1.4 → 0 → 1.0 → 0.5), remaining items are polish-level, all judges recommend proceeding to implementation.

---

# Final Recommendation

## LPIC-Final is the recommended third modeling approach for BLIS

### Design Summary
- **24 regimes** (3 hardware × 2 model_class × 4 workload)
- **7 alpha features** + **9 beta features** (all documented with expected signs)
- **Pure dot product** constraint: `α·D`, `β·D`
- **Ridge regression** with 5-fold CV for hyperparameter tuning
- **Hierarchical fallback** with coefficient-specific cross-hardware interpolation
- **~2,400 samples** minimum for full coverage

### Implementation Roadmap

**Phase 1 (MVP - 3 weeks):**
- Dense models only (12 regimes)
- Core features (D0-D6 for alpha, D0-D6,D8 for beta)
- Basic fallback (hardware-class average)

**Phase 2 (Full - 2 weeks):**
- Add MoE support (24 regimes)
- Enable D7_moe_routing
- Coefficient-specific interpolation

**Phase 3 (Production - 2 weeks):**
- Monitoring + drift detection
- A/B testing framework
- Documentation

**Total: ~7 weeks** to production

### Success Metrics
- β MAPE < 15% per regime
- α MAPE < 25% per regime
- E2E MAPE < 20% on held-out test
- 95% coefficient signs match expected

---

**ITERATIONS COMPLETE.** No further design iterations needed. Proceed to implementation.


