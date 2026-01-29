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

## 9. Summary: Roofline Modeling Implications

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

1. **Prefill vs. Decode Split**: The unified scheduler enables simultaneous prefill and decode processing, creating heterogeneous arithmetic intensity within batches.

2. **Memory-Bound Decode**: Decode phase typically bottlenecked by memory bandwidth due to scattered KV cache accesses. Performance scales with device memory bandwidth, not peak compute.

3. **Prefill Scalability**: Prefill arithmetic intensity increases with batch size and sequence length, enabling compute utilization up to roofline peak when sufficient tokens are batched.

4. **KV Cache Impact**: Cache size determines maximum batch size and sequence length. Quantization reduces bandwidth pressure but adds compute overhead.

5. **Partition-Based Efficiency**: 512-token partition strategy in Paged Attention V2 balances shared memory occupancy with register pressure, affecting achieved bandwidth.

6. **Prefix Caching Trade-off**: Reused prefixes reduce computation but add hashing and lookup overhead. Net benefit depends on prefix reuse frequency.

7. **CUDA Graph Overhead**: Graph capture amortization depends on batch consistency and replay frequency. Highly variable batches may reduce graph efficiency.

---

## References

- vLLM GitHub: https://github.com/vllm-project/vllm
- Key Files:
  - `vllm/v1/engine/core.py`: Main execution loop
  - `vllm/v1/engine/gpu_model_runner.py`: GPU execution
  - `vllm/v1/scheduler.py`: Scheduling logic
  - `csrc/attention.cu`, `csrc/cache_kernels.cu`: GPU kernels
  - `vllm/model_executor/layers/attention.py`: Attention abstractions
