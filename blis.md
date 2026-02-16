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
