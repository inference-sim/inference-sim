# vLLM Architecture and Performance Analysis for Roofline Modeling

## Executive Summary

vLLM is a high-performance inference engine that achieves efficient GPU utilization through unified token-based scheduling, sophisticated KV cache management, and optimized attention kernels. This analysis focuses on aspects relevant to roofline modeling: scheduler architecture, step time breakdown, GPU compute vs. memory behavior, and arithmetic intensity characteristics.

---

## 1. Scheduler Architecture

### Scheduling Algorithm

vLLM's v1 engine implements a **unified token-based scheduling approach** that treats all inference work uniformly:

- **Token-Centric Model**: Each request has `num_computed_tokens` and `num_tokens_with_spec`, advancing toward completion. This handles prefills, decode, chunked prefills, prefix caching, and speculative decoding uniformly.

- **Two-Stage Batch Formation**:
  1. Running requests first (prioritize ongoing work)
  2. Waiting requests second (schedule new/preempted work)

- **Resource Constraints Enforced**:
  - Maximum concurrent requests (`max_num_seqs`)
  - Total token budget per batch (`max_num_batched_tokens`)
  - LoRA adapter limits

### Request Queuing

Two queue implementations:
- **FCFS**: First-come-first-served ordering
- **Priority Queue**: Heap-based priority with arrival time as tiebreaker

### Key Advantages

- **Interleaving**: Prefill and decode work simultaneously in same batch
- **Chunked Prefills**: Long prompts split across multiple steps
- **Prefix Caching**: Reuse previously computed token sequences
- **Efficient Preemption**: Lower-priority work can be preempted when needed

---

## 2. Step Execution Time Breakdown

### Main Polling Loop (`vllm/v1/engine/core.py`)

The engine follows this cycle:

1. **Input Processing** (~μs-ms): Poll ZMQ sockets for incoming requests
2. **Scheduling** (~ms): Invoke scheduler to batch pending requests
3. **Model Execution** (~ms-seconds): Execute model with scheduled batch asynchronously
4. **Output Generation** (~μs-ms): Collect results and transmit via sockets

### GPU Model Runner Phases (`execute_model()`)

The forward pass follows these stages:

1. **State Updates**: Process scheduler output, manage request states
2. **Input Preparation**: Prepare GPU tensors (input IDs, positions, attention metadata, cache slots)
3. **Model Forward Pass**: Neural network computation generating logits
4. **Speculative Decoding** (optional): Generate draft tokens for verification
5. **Token Sampling**: Sample final tokens based on logits and sampling parameters

### CUDA Graph Optimization

- **CudagraphDispatcher**: Captures and replays optimized execution patterns
- **Modes**: PIECEWISE and FULL graph strategies
- **Performance Benefit**: Kernel fusion, reduced CPU overhead, optimized memory operations
- **Cost**: Graph capture overhead amortized over many replays

### Profiling Infrastructure

- **Encoder Timing Registry**: Thread-safe timing statistics
- **CUDA Graph Statistics**: Integrated with CUDAGraphStat
- **NVTx Hooks**: NVIDIA Tools Extension for timeline profiling
- **Layer-wise Granularity**: Available from layer to full inference

---

## 3. Attention Mechanisms and Memory Layout

### Paged Attention Implementation

**Memory Layout**:
- Key cache: `[num_blocks, num_heads, head_size/x, block_size, x]`
- Value cache: `[num_blocks, num_heads, head_size, block_size]`
- Block tables: `[num_seqs, max_num_blocks_per_seq]` (maps sequences to cache blocks)

**Paged Attention V2 Two-Stage Computation**:
1. **Partition Phase**: Processes query-key interactions across 512-token partitions
2. **Reduction Phase**: Combines results using accumulated exponentials and max logits

**Supported Configurations**:
- Head sizes: 32-256
- Block sizes: 8, 16, 32
- Block-sparse patterns
- FP8 KV cache quantization

### Attention Backend Abstraction

Multiple optimized backends implemented:
- Flash Attention (Standard and DiffKV)
- FlashInfer
- Flex Attention
- CPU Attention fallback
- Linear Attention
- Mamba/State-space models
- ROCm (AMD GPU)
- Triton-compiled kernels
- Specialized (Tree, MLA)

---

## 4. GPU Compute vs. Memory Behavior

### KV Cache Management

**Block-based Allocation**:
- Memory divided into discrete blocks enabling sharing across requests
- Reference counting tracks block usage
- Null block (id=0) serves as placeholder

**Prefix Caching**:
- Identifies and reuses previously computed KV values
- Block hashing with `hash_block_tokens()`
- Parent block hashes for hierarchical matching
- Reduces redundant computation for shared prefixes

**Advanced Techniques**:
- Sliding window management (frees out-of-window blocks)
- Lazy cache commitment (defer caching decisions)
- Multi-type KV cache support (full attention, sliding window, cross-attention)
- Hybrid KV cache coordination with LCM-based alignment

### Cache Operations (`csrc/cache_kernels.cu`)

Key CUDA kernels:
- **swap_blocks**: CPU↔GPU memory transfers
- **copy_blocks_kernel**: Standard KV cache transfers
- **reshape_and_cache_kernel**: Reformats incoming tokens with optional FP8
- **Quantization kernels**: FP8 conversion with per-tile scaling

**Memory Characteristics**:
- Vectorized element copying with alignment
- Warp-level reductions for scale computation
- Support for 8, 16, 32-bit data types
- Optional FP8 conversion

### Quantization Support

Comprehensive implementations:
- AWQ (Activation-aware Weight Quantization)
- FP4/W4A8 variants
- GPTQ and AllSpark
- W8A8 (8-bit weight and activation)
- Marlin format
- Hadamard/Hadacore
- CUTLASS integration

### Device Communication

- **Custom All-Reduce**: Hardware detection (NVLink vs. PCIe)
- **Zero-copy IPC Buffers**: Pre-allocated buffers for exchange
- **World Size**: Supports 2, 4, 6, 8 GPUs
- **Fallback**: NCCL for unsupported topologies

---

## 5. Microbatching and Parallelization

### Ubatching (Micro-batching)

Implemented in `gpu_ubatch_wrapper.py`:

- **Parallel Execution**: Multiple micro-batches on different CUDA streams
- **Threading Model**: Ubatch threads + main thread coordinated via barrier sync
- **CUDA Graph Capture**: Captures and replays optimized ubatch patterns
- **SM Allocation**: Controls Streaming Multiprocessor allocation
- **Throughput**: Concurrent sequence processing while maintaining batch benefits

---

## 6. Roofline-Relevant Insights

### Arithmetic Intensity Characteristics

**Prefill Phase**:
- **High arithmetic intensity**: Query @ Key^T, attention @ Value operations
- **Compute-bound for longer sequences**: Better GPU utilization
- **Token batch size effect**: Larger batches enable higher arithmetic intensity

**Decode Phase**:
- **Low arithmetic intensity**: Single token generation per request
- **Memory-bound**: Dominated by KV cache lookup and scattered access
- **Limited computation**: Activation functions, normalization, small matrix multiplies

**Arithmetic Intensity Formula**:
```
Arithmetic Intensity = FLOPs per batch / Bytes transferred per batch
- Prefill: Higher (process token sequences)
- Decode: Lower (single token with large memory footprint)
```

### Bottleneck Analysis

**Compute-Bound Scenarios**:
- Large prefill batches (100s-1000s of tokens)
- High-dimensional models (large d_model, many heads)
- Long sequences exceeding partition size (512 tokens for Paged Attention V2)

**Memory-Bound Scenarios**:
- Decode phase (single token per request)
- Small batch sizes (1-8 sequences)
- KV cache lookups and scattered access patterns
- Attention softmax with full-length key cache

### Batch Size Impact on Performance

1. **Token Budget Constraints** (`max_num_batched_tokens`):
   - Prefill typically dominant token consumer
   - Decode: many requests, fewer tokens each

2. **Batching Efficiency**:
   - Small batches (1-8): Memory-bound, poor utilization
   - Medium batches (32-128): Sweet spot for most workloads
   - Large batches (256+): Better arithmetic intensity, increased memory pressure

3. **KV Cache Memory Trade-off**:
   - Larger batches → more requests → larger cache footprint
   - Cache grows linearly with sequence length per request
   - FP8 quantization reduces footprint by 50% with compute overhead

4. **Concurrent Requests** (`max_num_seqs`):
   - Limits in-flight requests simultaneously
   - Affects KV cache allocation patterns
   - Influences preemption decisions

### Computation-to-Memory Ratio Factors

1. **Sequence Length**:
   - Longer sequences: More arithmetic per cache line
   - Extremely long sequences: May exceed GPU memory, requiring paging

2. **Model Dimensions**:
   - Larger head_size: More computation per fetched cache line
   - More heads: Parallelization across GPU threads

3. **Token Pattern**:
   - Prefill-heavy: Higher compute utilization
   - Decode-heavy: Memory bandwidth limited

4. **Prefix Caching Efficiency**:
   - Reused prefixes: Eliminate redundant computation
   - Memory savings: Free blocks for new requests
   - Improves arithmetic intensity for continuation requests

### Hardware Utilization Factors

1. **CUDA Graph Impact**:
   - Reduces CPU overhead for kernel launches
   - Enables kernel fusion and memory coalescing
   - Overhead amortized over many replays

2. **Paged Attention Memory Pattern**:
   - Block-based layout enables scattered KV cache reads
   - 512-token partitions balance shared memory vs. global bandwidth
   - Two-stage reduction combines partial results efficiently

3. **Quantization Benefits**:
   - FP8 KV cache: 50% memory bandwidth reduction
   - Per-tile scaling: Maintains accuracy
   - Trade-off: Quantization/dequantization compute overhead

---

## 7. Performance Metrics and Monitoring

### KV Cache Metrics

- **Lifetime**: Duration from allocation to eviction
- **Idle Time**: Time since last access
- **Reuse Gaps**: Time between consecutive accesses
- **Collection**: Sampling-based (default 1%)
- **Events**: Allocation, access, eviction

### Model Executor Interface

Abstract methods for execution backends (Ray, multiprocess, uniprocess):
- `execute_model()`: Process scheduler output and run model
- `sample_tokens()`: Generate tokens from outputs
- `execute_dummy_batch()`: Initialization
- `collective_rpc()`: Cross-worker communication
- `initialize_from_config()`: KV cache setup
- `determine_available_memory()`: Device capacity
- `get_kv_cache_specs()`: Cache specifications

---

## 8. Layer Implementations

Available optimized layers in `model_executor/layers`:

- **Attention**: Multiple optimized backends with automatic selection
- **Rotary Embeddings**: RoPE position encoding
- **Fused MOE**: Mixture-of-experts kernels
- **Mamba/Mamba2**: State-space model layers
- **Quantization**: Multi-precision support
- **Batch Invariant**: Batch-independent operations
- **Convolution**: Convolutional operations
- **Activation Functions**: Fused kernels
- **Layernorm**: Optimized normalization
- **Linear**: Weight matrices with quantization
- **Embeddings**: Vocabulary and position embeddings
- **Pooling**: Specialized pooling operations
- **FLA**: Efficient attention variants

---

## 9. Mixed Batch Execution: Detailed Kernel-Level Analysis

### Forward Pass Execution Flow

**Key Finding**: vLLM processes prefill and decode tokens **in a single unified forward pass**, not sequentially.

#### Entry Point: `execute_model()` Method

**File**: `vllm/v1/worker/gpu_model_runner.py:3280-3600`

The execution flow:

1. **Preprocessing** (lines 3304-3355):
   - Update persistent batch states
   - Collect all scheduled tokens from scheduler output
   - Calculate `num_tokens_unpadded` = total tokens across all requests
   - `max_num_scheduled_tokens` = maximum tokens in any single request

2. **Unified Batch Preparation** (lines 3346-3400):
```python
# All tokens (prefill + decode) collected together
tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
```

3. **Single Batch Descriptor** (lines 3372-3379):
   - One `BatchDescriptor` for entire mixed batch
   - No separate prefill/decode descriptors
   - CUDA graph mode determined once for the whole batch

4. **Single Model Forward** (line 3506):
```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

### Token Organization: Variable-Length Sequence Packing

**File**: `vllm/v1/worker/gpu_model_runner.py:1544-1549`

vLLM uses a **flat concatenated tensor** with cumulative indices:

```python
# Example: 3 requests with [1, 20, 1] tokens (decode, prefill, decode)
# num_scheduled_tokens: [1, 20, 1]
# cu_num_tokens (cumsum): [1, 21, 22]

self.query_start_loc.np[0] = 0
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens  # [0, 1, 21, 22]
```

**Structure**:
- All tokens packed into single flat tensor: `[T0, P0...P19, T1]` (total 22 tokens)
- `query_start_loc` = `[0, 1, 21, 22]` marks boundaries
- Request 0 (decode): tokens `[0:1]`
- Request 1 (prefill): tokens `[1:21]`
- Request 2 (decode): tokens `[21:22]`

This enables **variable-length sequence batching** where:
- Each request can have different query lengths
- Prefill requests (many tokens) and decode requests (1 token) coexist
- No padding required within the packed tensor

### FlashAttention Backend: Single Kernel Launch

**File**: `vllm/v1/attention/backends/flash_attn.py:607-742`

The attention computation uses **one kernel call** for all tokens:

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, ...):
    # Single FlashAttention kernel processes entire mixed batch
    flash_attn_varlen_func(
        q=query[:num_actual_tokens],           # All queries (prefill + decode)
        k=key_cache,                            # KV cache for all requests
        v=value_cache,
        out=output[:num_actual_tokens],
        cu_seqlens_q=cu_seqlens_q,             # query_start_loc: [0, 1, 21, 22]
        max_seqlen_q=max_seqlen_q,             # Max query length (e.g., 20)
        seqused_k=seqused_k,                   # seq_lens: [100, 50, 200] per request
        max_seqlen_k=max_seqlen_k,             # Max context (e.g., 200)
        block_table=block_table,               # Paged KV cache block mappings
        scheduler_metadata=scheduler_metadata,
        ...
    )
```

**Key Parameters**:
- **`cu_seqlens_q`** (`query_start_loc`): Cumulative query positions `[0, 1, 21, 22]`
- **`max_seqlen_q`**: Maximum query tokens in any request (20 in example)
- **`seqused_k`** (`seq_lens`): Per-request total context length `[100, 50, 200]`
- **`max_seqlen_k`**: Maximum context length across all requests (200)

**No Separate Kernel Launches**:
- FlashAttention v2/v3 kernel internally handles variable-length sequences
- Prefill requests with 20 tokens and decode requests with 1 token processed in same kernel
- Kernel uses `cu_seqlens_q` to identify each request's query range

### Attention Metadata: Unified Structure

**File**: `vllm/v1/attention/backend.py:287-410`

Single `CommonAttentionMetadata` for entire batch:

```python
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor      # [0, 1, 21, 22] - cumulative indices
    seq_lens: torch.Tensor             # [100, 50, 200] - per-request context
    num_reqs: int                      # 3
    num_actual_tokens: int             # 22 total tokens
    max_query_len: int                 # 20 (max query in batch)
    max_seq_len: int                   # 200 (max context in batch)
    block_table: torch.Tensor          # Paged KV cache mappings
    slot_mapping: torch.Tensor         # Token → cache slot mapping
```

**Characteristics**:
- Single metadata structure for mixed batch
- No prefill-specific or decode-specific metadata
- All requests described by same structure with variable lengths

### Layer-by-Layer Processing

**Critical Observation**: While the forward pass is unified, **each layer processes tokens sequentially**:

```
Layer 0: Input [T0, P0...P19, T1] → QKV projection → Attention → MLP → Output
Layer 1: Previous output → QKV projection → Attention → MLP → Output
...
Layer N: Previous output → Final projection
```

**Within Each Layer**:
1. **QKV Projection**: Single GEMM for all tokens (prefill + decode)
   - Matrix multiply: `[num_tokens, hidden_dim] @ [hidden_dim, 3*hidden_dim]`
   - Processes all 22 tokens together

2. **Attention Computation**: Single FlashAttention kernel
   - Variable-length sequence handling via `cu_seqlens_q`
   - Prefill tokens attend to their prompt context
   - Decode tokens attend to full KV cache

3. **Output Projection**: Single GEMM for all tokens
   - Matrix multiply: `[num_tokens, hidden_dim] @ [hidden_dim, hidden_dim]`

4. **MLP**: Single FFN computation for all tokens
   - Gate/Up/Down projections on all tokens together

**Memory Access Pattern**:
- **Prefill requests**: High arithmetic intensity (Q@K^T on large matrices)
- **Decode requests**: Low arithmetic intensity (single query vector)
- **Combined**: Heterogeneous workload in same kernel

### Why Sequential Processing May Occur

Despite unified batching, **effective sequential behavior** can occur due to:

#### 1. **Memory Hierarchy Effects**

**L2 Cache Behavior**:
- Prefill tokens generate large intermediate tensors
- Decode tokens generate small intermediate tensors
- L2 cache may not effectively share data between heterogeneous token types

**HBM Access Patterns**:
- Prefill: Mostly streaming reads/writes (high bandwidth utilization)
- Decode: Scattered KV cache reads (low bandwidth utilization)
- Mixed access patterns may serialize at memory controller

#### 2. **Kernel Launch Overhead**

**Per-Layer Operations**:
- 4-5 kernel launches per layer (QKV, Attn, Proj, Gate, Up, Down)
- Each kernel has fixed overhead (~5-20μs)
- For models with 28-48 layers: 112-240 kernel launches per step
- Overhead: ~0.5-5ms total per forward pass

#### 3. **CUDA Stream Serialization**

**Default Stream Behavior**:
- Operations in same CUDA stream execute sequentially
- Mixed batches may not benefit from kernel-level parallelism
- Ubatching (DBO) can parallelize across streams, but disabled for mixed batches in some configurations

#### 4. **Warp Divergence**

**FlashAttention Kernel Internals**:
- Warps processing prefill tokens: High compute utilization
- Warps processing decode tokens: Low compute utilization (more memory waits)
- Kernel execution time determined by slowest warp
- Mixed workload may reduce overall efficiency

#### 5. **Block Table Lookup Overhead**

**Paged Attention Memory Access**:
- Each token needs block table lookup
- Decode tokens: Lookup full KV cache across many blocks
- Prefill tokens: Minimal lookups (fresh tokens)
- Unbalanced lookup overhead adds to decode latency

### CUDA Graph Impact on Mixed Batches

**File**: `vllm/forward_context.py:29-58`

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False      # False for mixed batches
    has_lora: bool = False

    def relax_for_mixed_batch_cudagraphs(self) -> "BatchDescriptor":
        # Used for PIECEWISE cudagraphs or mixed prefill-decode
        return BatchDescriptor(
            self.num_tokens, num_reqs=None, uniform=False, has_lora=self.has_lora
        )
```

**CUDA Graph Modes**:
1. **FULL**: Captures entire forward pass with fixed dimensions
   - Requires padding to match graph shape
   - Not efficient for highly variable mixed batches

2. **PIECEWISE**: Captures individual operations with variable shapes
   - Better for mixed batches
   - Still requires shape matching within tolerance

3. **DISABLED**: No graph capture
   - Full flexibility for variable batches
   - Higher kernel launch overhead

**Mixed Batch Challenge**:
- Prefill-heavy: Large token count, few requests
- Decode-heavy: Small token count, many requests
- Graph capture requires similar shapes → limited reuse across workload types

### Roofline Implications for Mixed Batches

#### Scenario 1: Prefill-Heavy Batch (e.g., CodeLlama prefillheavy)
- **Characteristics**: Few large prefills (3000 tokens), minimal decode
- **Bottleneck**: Compute-bound (large matrix operations)
- **Time Estimation**: Dominated by prefill compute
- **Decode Impact**: Minimal (1-2 decode requests negligible)

#### Scenario 2: Decode-Heavy Batch
- **Characteristics**: Many decode requests (50-100), no prefill
- **Bottleneck**: Memory-bound (weight loading + KV cache access)
- **Time Estimation**: Weight loading amortized across batch
- **Prefill Impact**: None

#### Scenario 3: Mixed Batch (e.g., mid-workload phase)
- **Characteristics**: 5-10 prefills (500 tokens each), 10-20 decode
- **Bottleneck**: Both compute (prefill) and memory (decode)
- **Time Estimation**: **Complex interaction**
  - If fully overlapped: `max(prefill_time, decode_time)`
  - If sequential: `prefill_time + decode_time`
  - Reality: **Partial overlap** due to memory hierarchy effects

**Empirical Finding** (from roofline_step.go testing):
- Additive model (`prefill_time + decode_time`) gives **better predictions** than fully overlapped model
- Suggests effective sequential processing through layers
- Likely due to:
  1. Layer-by-layer execution (not inter-layer parallelism)
  2. Memory bandwidth sharing between prefill and decode
  3. Kernel launch overhead accumulation
  4. Warp divergence in mixed workloads

---

## 10. Summary: Roofline Modeling Implications

| Aspect | Characteristic | Roofline Implication |
|--------|----------------|-------------------|
| **Memory Hierarchy** | Paged KV cache with block-based allocation | Enables cache reuse, scattered access patterns |
| **Compute Kernel** | Partition-based attention (512-token partitions) | Balances shared memory vs. global bandwidth |
| **Prefill Workload** | High arithmetic intensity | Compute-bound, high GPU utilization |
| **Decode Workload** | Low arithmetic intensity | Memory-bound, limited by memory bandwidth |
| **Batch Size Scaling** | Token-based budgeting | Interleaves prefill/decode for better utilization |
| **Memory Footprint** | Quantizable KV cache (FP8) | 50% bandwidth reduction with compute overhead |
| **Parallelization** | Ubatching with concurrent streams | Improved throughput for multi-sequence workloads |
| **Latency Optimization** | CUDA graph capture/replay | Reduced kernel launch overhead |

### Key Insights for Performance Modeling

1. **Unified Forward Pass with Effective Sequential Execution**: vLLM processes prefill and decode tokens in a single forward pass through the model, but layer-by-layer execution, memory hierarchy effects, and warp divergence create **effectively sequential behavior**. Roofline models should use **additive time estimation** (`prefill_time + decode_time`) rather than full overlap for mixed batches.

2. **Variable-Length Sequence Packing**: Tokens from different requests (prefill with many tokens, decode with 1 token) are packed into a flat tensor using `query_start_loc` cumulative indices. FlashAttention handles variable-length sequences in a single kernel, but heterogeneous workloads reduce kernel efficiency.

3. **Memory-Bound Decode**: Decode phase typically bottlenecked by memory bandwidth due to:
   - Weight loading (amortized across batch)
   - Scattered KV cache accesses (per-token block table lookups)
   - Performance scales with device memory bandwidth, not peak compute
   - Mixed batches share bandwidth between prefill and decode operations

4. **Prefill Scalability with Batch Size**: Prefill arithmetic intensity increases with:
   - Total tokens in batch (multiple concurrent prefills)
   - Sequence length per prefill
   - Enables compute utilization up to roofline peak when sufficient tokens batched
   - MFU (Model FLOPs Utilization) should scale with batch characteristics

5. **Kernel Launch Overhead**: Each layer requires 4-5 kernel launches (QKV, Attn, Proj, FFN):
   - 28-layer model: ~112 kernel launches per step
   - 48-layer model: ~240 kernel launches per step
   - Overhead: ~0.5-5ms total per forward pass
   - CUDA graph capture amortizes overhead but requires shape consistency

6. **KV Cache Impact**:
   - Cache size determines maximum batch size and sequence length
   - Quantization (FP8) reduces bandwidth pressure by 50% but adds dequantization compute overhead
   - Block table lookups add latency for long-context decode
   - Paged memory enables sharing but increases memory access indirection

7. **Partition-Based Efficiency**: 512-token partition strategy in Paged Attention V2:
   - Balances shared memory occupancy with register pressure
   - Partition-based reduction may serialize attention computation
   - Affects achieved bandwidth for very long contexts

8. **Prefix Caching Trade-off**:
   - Reused prefixes reduce computation (eliminate redundant prefill)
   - Adds hashing and lookup overhead
   - Net benefit depends on prefix reuse frequency
   - Cache hit enables chunked prefill to start from cached position

9. **Mixed Batch Complexity**:
   - Prefill-heavy workloads: Compute-bound, high MFU achievable
   - Decode-heavy workloads: Memory-bound, low MFU typical
   - Mixed workloads: Heterogeneous arithmetic intensity creates kernel inefficiency
   - **Time estimation**: Additive model more accurate than overlap model

---

## References

### vLLM GitHub
https://github.com/vllm-project/vllm

### Key Files Analyzed for Roofline Modeling

**Execution and Scheduling**:
- `vllm/v1/engine/core.py`: Main execution loop and orchestration
- `vllm/v1/worker/gpu_model_runner.py`: GPU model runner with `execute_model()` method
  - Lines 3280-3600: Mixed batch preparation and forward pass
  - Lines 1544-1549: Token organization using `query_start_loc`
  - Lines 3372-3379: Batch descriptor and CUDA graph mode determination
- `vllm/v1/scheduler.py`: Token-based unified scheduling logic

**Attention Implementation**:
- `vllm/v1/attention/backends/flash_attn.py`: FlashAttention v2/v3 backend
  - Lines 607-742: Forward method with single kernel launch
  - Lines 719-742: `flash_attn_varlen_func()` call with variable-length sequences
- `vllm/v1/attention/backend.py`: Attention backend abstraction
  - Lines 287-410: `CommonAttentionMetadata` structure for mixed batches
- `vllm/forward_context.py`: Batch descriptor for CUDA graph handling
  - Lines 29-58: `BatchDescriptor` with mixed batch support

**GPU Kernels**:
- `csrc/attention.cu`: CUDA attention kernels (FlashAttention integration)
- `csrc/cache_kernels.cu`: KV cache management kernels
  - Swap, copy, reshape operations
  - FP8 quantization kernels

**Model Layers**:
- `vllm/model_executor/layers/attention.py`: Attention layer abstractions
- `vllm/model_executor/layers/rotary_embedding.py`: RoPE position encoding
- `vllm/model_executor/layers/fused_moe`: Mixture-of-experts kernels

### Research Papers Referenced
- FlashAttention v2: Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
- Paged Attention: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- vLLM System: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (MLSys 2023)
