# BLIS Performance Modeling and Roofline Analysis

## Overview

**BLIS** (Blackbox Inference Simulator) implements an analytical roofline model for LLM inference performance prediction in Go. Despite its name, it does NOT integrate with the BLIS linear algebra library—it implements its own performance modeling from scratch.

## Roofline Model Implementation

### Core Formula
```
Phase_Time = max(FLOPs / Peak_Performance, Bytes / Memory_Bandwidth)
Step_Time = Prefill_Time + Decode_Time + Communication_Overhead + Kernel_Overhead
```

The model uses the classical roofline approach, computing the maximum of compute-bound and memory-bound times for each phase. Key implementation: `sim/roofline_step.go:137-208`

### Hardware Characterization

For H100 (from `hardware_config.json`):
- **Peak Compute**: 989.5 TFLOPS (tensor cores)
- **Peak Bandwidth**: 3.35 TB/s
- **Effective Bandwidth**: 72% of peak (2.41 TB/s)
- **MFU Prefill**: 65% (compute-bound workload)
- **MFU Decode**: 12% (memory-bound workload)

## Roofline Assumptions: Explicit and Implicit

### Explicit Assumptions
- Tensor cores for GEMM operations (QKV, attention, MLP)
- FlashAttention eliminates O(seq²) memory traffic for self-attention
- Fixed bandwidth efficiency: 72% of theoretical peak
- Phase-specific Model FLOPs Utilization (MFU) values calibrated empirically

### Implicit Assumptions
- **Flat memory hierarchy**: No L1/L2 cache modeling; treats HBM as primary memory
- **Constant all-reduce latency**: ~20 µs per layer regardless of message size
- **No arithmetic intensity metric**: Uses max(compute, memory) instead of explicit intensity calculation
- **Fixed overheads**: Kernel launch and communication costs are uniform
- **No DVFS/throttling**: Hardware performance assumed constant

## FLOPs Calculation

Two operation categories: `sim/roofline_step.go:8-79`

1. **GEMM Operations** (Tensor Cores at full peak)
   - QKV projections
   - Q·K^T attention scores
   - Attention·V output
   - MLP feed-forward gates

2. **Vector Operations** (10% of peak FLOPS)
   - Softmax, RoPE, layer normalization

Critical insight: Attention score matrices are treated as GEMMs when sufficiently large, enabling tensor core utilization rather than scalar operations.

## Memory Access Modeling

Four distinct access patterns:
1. **Model Weights**: Loaded once per step, divided by tensor parallelism factor
2. **KV Cache Growth**: Writing new token embeddings
3. **KV Cache Access**: Reading historical context for attention
4. **Activation Memory**: Intermediate tensors during forward pass

**FlashAttention Impact**: New prefill tokens attending to each other are processed in fast SRAM, generating zero HBM traffic for self-attention operations. This is a critical roofline assumption enabling compute-bound prefill performance.

## Compute-Bound vs Memory-Bound Dynamics

### Prefill Phase (Typically Compute-Bound)
- Arithmetic intensity >> 410 FLOPs/byte (H100 roofline threshold)
- MFU = 65%
- Weight amortization across large batch sizes
- Time dominated by peak compute capacity

### Decode Phase (Typically Memory-Bound)
- Arithmetic intensity ≈ 1-2 FLOPs/byte
- MFU = 12%
- Time dominated by memory bandwidth
- Repeated weight reloading per token

## Tensor Parallelism Integration

- **FLOPs divided by TP factor**: Reduced per-GPU compute
- **Memory divided by TP factor**: Reduced per-GPU bandwidth
- **All-reduce communication**: 2× per layer, ~20 µs fixed latency, no overlap with compute (conservative model)

## Dual Modeling Modes

1. **Regression Mode**: Learns α/β coefficients from calibration data (requires GPU measurements)
2. **Roofline Mode**: Analytical derivation (zero training, generalizable to new hardware/models)

The roofline approach enables zero-shot performance prediction for any LLM with known architecture and hardware specs.

## Key Architectural Insights

**Performance Prediction Pipeline**:
- Roofline calculations integrate directly into discrete event scheduler logic
- Supports complex inference patterns: chunked prefill, prefix caching, continuous batching
- Production-grade model optimized for fast, training-free prediction

**Design Trade-offs**:
- **Prioritizes simplicity** over absolute accuracy
- **Empirically-calibrated MFU values** serve as tuning parameters rather than first-principles derivation
- **No cache hierarchy modeling** simplifies analysis and improves portability
- **Uniform efficiency factors** reduce parameter count for generalization

## Implementation Details

- **Main roofline logic**: `sim/roofline_step.go`
  - FLOPs calculation: lines 8-79
  - Memory bytes calculation: lines 81-135
  - Roofline formula application: lines 137-208
- **Hardware configuration**: `sim/model_hardware_config.go:24-33`
- **Scheduler integration**: `sim/simulator.go:422-428`

## Conclusion

BLIS implements a **production-grade analytical roofline model** balancing accuracy, generalization, and computational efficiency. By combining empirically-calibrated MFU values with theoretical roofline analysis, it enables fast, training-free LLM inference performance prediction across diverse hardware and model configurations. The architecture prioritizes capacity planning and what-if analysis use cases over cycle-accurate simulation.

### Roofline Strengths
✓ Training-free performance prediction
✓ Generalizes to new LLMs and hardware
✓ Fast execution for scheduling decisions
✓ Captures prefill vs decode asymmetry

### Known Limitations
✗ Flat memory hierarchy assumption
✗ Fixed communication costs
✗ Empirically-calibrated rather than first-principles
✗ No adaptive MFU values
✗ Simplified kernel overhead model
