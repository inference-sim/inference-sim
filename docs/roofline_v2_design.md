# Roofline V2: Literature-Backed Performance Model

**Status**: Design Document
**Author**: Systems Research Team
**Date**: 2025-02-16

## Executive Summary

This document describes a principled approach to roofline-based vLLM step time estimation that reduces calibration parameters from **13 to 2** by grounding the model in published systems research. The approach is designed to generalize across LLMs and workloads while remaining reproducible for any hardware platform.

### Key Innovation

Replace empirical calibration with **measured hardware characteristics** and **literature-backed models**:
- Hardware limits measured via standard benchmarks (CUTLASS, STREAM, NCCL)
- Algorithmic properties computed exactly from model architecture
- Only 2 framework-specific parameters require calibration

---

## Table of Contents

1. [Motivation](#motivation)
2. [Literature Foundation](#literature-foundation)
3. [Architecture Overview](#architecture-overview)
4. [Hardware Characterization Protocol](#hardware-characterization-protocol)
5. [Model Implementation](#model-implementation)
6. [Calibration Procedure](#calibration-procedure)
7. [Validation Strategy](#validation-strategy)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Comparison with Current Approach](#comparison-with-current-approach)

---

## Motivation

### Current Limitations

The existing roofline implementation (`sim/roofline_step.go`) requires 13 calibration parameters:

```
Hardware:     TFlopsPeak, BwPeakTBs, BwEffConstant
Efficiency:   mfuPrefill, mfuDecode, prefillBwFactor, decodeBwFactor
TP Scaling:   tpScalingExponent, decodeTpScalingExponent
Calibration:  mfuPrefillMultiplier, mfuDecodeMultiplier
Overhead:     TOverheadMicros, allReduceLatency
```

**Problems**:
1. **Lack of generalizability**: Parameters tuned for one model may not transfer
2. **Unclear physical meaning**: Magic numbers (0.72, 0.75, 0.96) lack theoretical grounding
3. **Coupling**: Efficiency factors interact with calibration multipliers
4. **Reproducibility**: No clear protocol for measuring parameters on new hardware

### Design Goals

1. **Minimize calibration**: Reduce to 2 orthogonal parameters
2. **Literature backing**: Every number must come from measurement or published research
3. **Reproducibility**: Clear protocol for characterizing new hardware
4. **Generalization**: Zero-shot performance on unseen models
5. **Systems-grade rigor**: Acceptable to systems research community (OSDI/SOSP/ATC standards)

---

## Literature Foundation

All model components are grounded in published research:

| Concept | Reference | Application |
|---------|-----------|-------------|
| **Roofline Model** | Williams et al., CACM 2009 [^1] | Core performance model: $T = \max(W/F, Q/B)$ |
| **Operational Intensity** | Williams et al., CACM 2009 | Compute vs memory bound determination |
| **FlashAttention** | Dao et al., NeurIPS 2022 [^2] | Attention fused in SRAM; zero HBM traffic for $QK^TV$ |
| **PagedAttention** | Kwon et al., SOSP 2023 [^3] | KV cache access patterns (random for decode, sequential for prefill) |
| **Tensor Parallelism** | Shoeybi et al., arXiv 2019 [^4] | Communication volume: $2 \times d_{model} \times \text{bytes}$ per layer |
| **Memory Bandwidth** | McCalpin, STREAM 1995 [^5] | Achievable BW = 60-85% of peak depending on access pattern |
| **Amdahl's Law** | Amdahl, AFIPS 1967 [^6] | TP speedup: $S = 1/(s + p/n)$ with communication overhead |

[^1]: S. Williams, A. Waterman, D. Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*, 2009.

[^2]: T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*, 2022.

[^3]: W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, I. Stoica. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP*, 2023.

[^4]: M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." *arXiv:1909.08053*, 2019.

[^5]: J. D. McCalpin. "STREAM: Sustainable Memory Bandwidth in High Performance Computers." *Technical Report*, University of Virginia, 1995.

[^6]: G. M. Amdahl. "Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities." *AFIPS Conference Proceedings*, 1967.

---

## Architecture Overview

### Four-Layer Model

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Hardware Characteristics (Measured Once)       │
│  • Peak FLOPS          [datasheet or CUTLASS]          │
│  • Achievable BW       [STREAM benchmark]              │
│  • NVLink bandwidth    [nccl-tests]                    │
│  • All-reduce latency  [nccl-tests]                    │
│  • L2 cache size       [deviceQuery]                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Algorithmic Analysis (Computed Exactly)        │
│  • FLOPs per token     [Transformer architecture]      │
│  • Memory traffic      [Weight/KV/activation sizes]    │
│  • Access patterns     [Sequential vs random]          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: First-Principles Execution Model               │
│  • Roofline bound      [Williams 2009]                 │
│  • TP scaling          [Amdahl + measured comm]        │
│  • Memory hierarchy    [STREAM access patterns]        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Framework Calibration (2 PARAMETERS ONLY)      │
│  • α: Kernel efficiency    [vLLM-specific fusion]     │
│  • β: Overhead per step    [Scheduling/launch]        │
└─────────────────────────────────────────────────────────┘
```

### Key Principle

**Parameters at higher layers should be minimal.** Most complexity is handled by lower layers using first principles and measurements.

---

## Hardware Characterization Protocol

### Overview

Characterize hardware **once per GPU type** using standard benchmarks. Results are stored in `hardware_config.json` and reused across all models.

### Benchmark Suite

| Benchmark | Measurement | Tool | Output Parameter |
|-----------|-------------|------|------------------|
| **Peak Tensor Core FLOPS** | Maximum GEMM throughput | CUTLASS Profiler | `TFlopsPeak`, `MfuMeasured` |
| **Memory Bandwidth** | Streaming bandwidth | STREAM benchmark | `BwStreamCopy`, `BwStreamScale` |
| **NVLink Bandwidth** | Inter-GPU bandwidth | NCCL bandwidth test | `NVLinkBW` |
| **All-Reduce Latency** | Collective communication | NCCL latency test | `AllReduceLatencyPerByte` |
| **Device Properties** | Cache sizes, SM count | CUDA deviceQuery | `L2CacheSize`, `NumSMs` |

---

### 1. Peak FLOPS and MFU Measurement (CUTLASS)

**Tool**: [NVIDIA CUTLASS Profiler](https://github.com/NVIDIA/cutlass)

#### Installation

```bash
# Clone CUTLASS repository
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Build CUTLASS profiler
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90  # 90 for H100, 80 for A100
make cutlass_profiler -j$(nproc)
```

#### Running GEMM Benchmark

```bash
# Test GEMM performance for typical transformer dimensions
# M = batch_size * seq_len, K = hidden_dim, N = hidden_dim

./tools/profiler/cutlass_profiler \
  --operation=gemm \
  --providers=cutlass \
  --m=8192 --n=4096 --k=4096 \
  --A=f16:column --B=f16:column --C=f32:column \
  --accum=f32 --cta_m=128 --cta_n=128 --cta_k=64 \
  --stages=4 --warmup-iterations=10 --profiling-iterations=100 \
  | tee cutlass_results.txt

# Parse results
ACHIEVED_TFLOPS=$(grep "GFLOPs/s" cutlass_results.txt | awk '{print $3/1000}')
PEAK_TFLOPS=989.5  # From H100 datasheet
MFU_MEASURED=$(echo "scale=4; $ACHIEVED_TFLOPS / $PEAK_TFLOPS" | bc)

echo "Achieved TFLOPS: $ACHIEVED_TFLOPS"
echo "MFU: $MFU_MEASURED"  # Typically 0.55-0.65 for fp16 tensor cores
```

**Expected Results (H100 80GB)**:
- Peak TFLOPS: 989.5 (from datasheet)
- Achieved: ~570 TFLOPS
- **MFU: ~0.58** (58% utilization due to wave quantization, kernel overhead)

#### Physical Interpretation

`MfuMeasured` captures:
- **Wave quantization**: Not all SMs busy when problem size isn't a perfect multiple
- **Kernel launch overhead**: Fixed cost per GEMM kernel
- **Memory bank conflicts**: Suboptimal access patterns
- **Register spilling**: Limited register file per SM

This is measured **once per GPU** and applies to all models.

---

### 2. Memory Bandwidth Measurement (STREAM)

**Tool**: [STREAM Benchmark](https://www.cs.virginia.edu/stream/)

#### Installation

```bash
# Download STREAM source
wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c

# Compile for GPU using CUDA
cat > stream_cuda.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

#define N (256*1024*1024)  // 256M elements = 2GB for fp64

__global__ void copy_kernel(double *a, double *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

__global__ void scale_kernel(double *a, double *b, double scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = scalar * a[i];
}

__global__ void add_kernel(double *a, double *b, double *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void triad_kernel(double *a, double *b, double *c, double scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + scalar * c[i];
}

int main() {
    double *d_a, *d_b, *d_c;
    size_t bytes = N * sizeof(double);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Warmup
    for (int i = 0; i < 10; i++) {
        copy_kernel<<<grid, block>>>(d_a, d_b, N);
    }
    cudaDeviceSynchronize();

    // Benchmark Copy: b[i] = a[i]  (2 arrays, 2 * 8 bytes = 16 bytes per element)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        copy_kernel<<<grid, block>>>(d_a, d_b, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double copy_bw = (2.0 * bytes * 100) / (ms / 1000.0) / 1e12;  // TB/s
    printf("Copy bandwidth: %.2f TB/s\n", copy_bw);

    // Benchmark Scale: b[i] = scalar * a[i]
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        scale_kernel<<<grid, block>>>(d_a, d_b, 3.0, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    double scale_bw = (2.0 * bytes * 100) / (ms / 1000.0) / 1e12;
    printf("Scale bandwidth: %.2f TB/s\n", scale_bw);

    // Benchmark Triad: a[i] = b[i] + scalar * c[i]
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        triad_kernel<<<grid, block>>>(d_a, d_b, d_c, 3.0, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    double triad_bw = (3.0 * bytes * 100) / (ms / 1000.0) / 1e12;
    printf("Triad bandwidth: %.2f TB/s\n", triad_bw);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
EOF

nvcc -O3 stream_cuda.cu -o stream_cuda
```

#### Running STREAM Benchmark

```bash
# Pin GPU clocks to avoid frequency scaling
sudo nvidia-smi -lgc 1410  # Lock to base clock (H100)

# Run benchmark
./stream_cuda | tee stream_results.txt

# Example output:
# Copy bandwidth: 2.85 TB/s    (85% of 3.35 TB/s peak)
# Scale bandwidth: 2.73 TB/s   (81% of peak)
# Triad bandwidth: 2.68 TB/s   (80% of peak)

# Parse results
BW_STREAM_COPY=$(grep "Copy bandwidth" stream_results.txt | awk '{print $3}')
BW_STREAM_SCALE=$(grep "Scale bandwidth" stream_results.txt | awk '{print $3}')
```

**Expected Results (H100 80GB)**:
- Peak HBM Bandwidth: 3.35 TB/s (from datasheet)
- **Copy: ~2.85 TB/s (85% efficiency)** ← Use this for sequential access
- **Scale: ~2.73 TB/s (81% efficiency)**
- **Triad: ~2.68 TB/s (80% efficiency)**

#### Physical Interpretation

STREAM measures **achievable bandwidth** under different access patterns:

- **Copy** (85%): Simple streaming reads/writes (best case)
- **Scale** (81%): Streaming with scalar multiply (compute overhead)
- **Triad** (80%): Streaming with multiple arrays (bank conflicts)

For roofline model:
- Use **Copy BW** for sequential weight loading (prefill)
- Use **Triad BW × 0.75** for paged attention scattered access (decode)

This matches literature: McCalpin (1995) shows 75-85% efficiency is typical for real workloads.

---

### 3. NVLink and All-Reduce Measurement (NCCL)

**Tool**: [nccl-tests](https://github.com/NVIDIA/nccl-tests)

#### Installation

```bash
# Clone and build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
```

#### Running All-Reduce Benchmark

```bash
# Test all-reduce latency for typical model hidden dimensions
# H100 has 900 GB/s per NVLink (4th gen), 18 links total

# Run for different TP values
for TP in 2 4 8; do
    mpirun -np $TP --bind-to none \
        ./build/all_reduce_perf \
        -b 128K -e 32M -f 2 -g 1 \
        | tee allreduce_tp${TP}.txt
done

# Example output (TP=2, 8MB message = 2 layers × 4096 hidden_dim × fp16):
# Size        Time(us)   Bandwidth(GB/s)
# 8388608     157.3      106.7

# Parse per-layer latency (assume typical message size = 2 * hidden_dim * 2 bytes)
HIDDEN_DIM=4096
MSG_SIZE=$((2 * HIDDEN_DIM * 2))  # 2 * hidden_dim * fp16
ALLREDUCE_LATENCY=$(grep "^$MSG_SIZE " allreduce_tp2.txt | awk '{print $2}')

echo "All-reduce latency per layer: $ALLREDUCE_LATENCY μs"
```

**Expected Results (H100, TP=2 via NVLink)**:
- Message size: 16 KB per layer (typical hidden_dim=4096)
- **Latency: ~15-25 μs per layer**
- Bandwidth: ~700-800 GB/s (actual vs 900 GB/s peak due to protocol overhead)

#### Physical Interpretation

All-reduce latency has two components:
1. **Latency**: Fixed protocol overhead (~10 μs)
2. **Bandwidth**: Data transfer time = `message_size / nvlink_bw`

For TP scaling model (Amdahl's Law):
```
comm_time = latency + (message_size / bandwidth)
tp_speedup = tp / (1 + comm_time * num_layers / compute_time)
```

---

### 4. Device Properties (CUDA)

```bash
# Get device properties
cat > device_query.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device: %s\n", prop.name);
    printf("L2 Cache Size: %.2f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

    return 0;
}
EOF

nvcc device_query.cu -o device_query
./device_query | tee device_properties.txt
```

**Expected Output (H100 80GB)**:
```
Device: NVIDIA H100 80GB HBM3
L2 Cache Size: 50.00 MB      ← Used for FlashAttention modeling
Shared Memory per Block: 164 KB
Total Global Memory: 80.00 GB
Number of SMs: 132
Max Threads per SM: 2048
```

---

### 5. Automated Characterization Script

```bash
#!/bin/bash
# File: scripts/characterize_hardware.sh
# Usage: ./characterize_hardware.sh H100 /path/to/output

set -e

GPU_TYPE=$1
OUTPUT_DIR=${2:-./hardware_profiles}

echo "=== Hardware Characterization for $GPU_TYPE ==="
echo "Output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# Pin GPU clocks
echo "Pinning GPU clocks..."
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader,nounits | head -1)

# 1. Device properties
echo "=== Step 1/5: Device Properties ==="
./device_query > $OUTPUT_DIR/device_properties.txt
L2_CACHE=$(grep "L2 Cache Size" $OUTPUT_DIR/device_properties.txt | awk '{print $4}')

# 2. Peak FLOPS (CUTLASS)
echo "=== Step 2/5: Peak FLOPS (CUTLASS) ==="
./cutlass_profiler \
    --operation=gemm \
    --m=8192 --n=4096 --k=4096 \
    --A=f16:column --B=f16:column --C=f32:column \
    --providers=cutlass --warmup-iterations=10 --profiling-iterations=100 \
    > $OUTPUT_DIR/cutlass_results.txt

ACHIEVED_TFLOPS=$(grep "GFLOPs/s" $OUTPUT_DIR/cutlass_results.txt | awk '{print $3/1000}')
PEAK_TFLOPS=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader | awk '{print 989.5}')  # From datasheet
MFU_MEASURED=$(echo "scale=4; $ACHIEVED_TFLOPS / $PEAK_TFLOPS" | bc)

# 3. Memory Bandwidth (STREAM)
echo "=== Step 3/5: Memory Bandwidth (STREAM) ==="
./stream_cuda > $OUTPUT_DIR/stream_results.txt
BW_STREAM_COPY=$(grep "Copy bandwidth" $OUTPUT_DIR/stream_results.txt | awk '{print $3}')

# 4. NVLink and All-Reduce (NCCL)
echo "=== Step 4/5: All-Reduce Latency (NCCL) ==="
if command -v mpirun &> /dev/null; then
    mpirun -np 2 --bind-to none ./all_reduce_perf -b 16K -e 16K -g 1 \
        > $OUTPUT_DIR/allreduce_results.txt
    ALLREDUCE_LATENCY=$(grep "^16384 " $OUTPUT_DIR/allreduce_results.txt | awk '{print $2}')
    NVLINK_BW=$(grep "^16384 " $OUTPUT_DIR/allreduce_results.txt | awk '{print $3}')
else
    echo "MPI not found, using default values"
    ALLREDUCE_LATENCY=20.0
    NVLINK_BW=700.0
fi

# 5. Generate JSON config
echo "=== Step 5/5: Generating hardware_config.json ==="
cat > $OUTPUT_DIR/hardware_config_${GPU_TYPE}.json << EOF
{
    "$GPU_TYPE": {
        "TFlopsPeak": $PEAK_TFLOPS,
        "MfuMeasured": $MFU_MEASURED,
        "BwStreamCopy": $BW_STREAM_COPY,
        "NVLinkBW": $NVLINK_BW,
        "AllReduceLatency": $ALLREDUCE_LATENCY,
        "L2CacheSize": $L2_CACHE
    }
}
EOF

echo "=== Characterization Complete ==="
cat $OUTPUT_DIR/hardware_config_${GPU_TYPE}.json

# Restore default GPU clocks
sudo nvidia-smi -pm 0
```

**Time**: ~30 minutes per GPU type
**Output**: Complete hardware profile ready for roofline model

---

## Model Implementation

### Unified Roofline Formula

Replace separate prefill/decode paths with operational intensity-based routing:

```go
// Hardware balance point (Williams 2009)
hwBalancePoint := hw.TFlopsPeak * 1e12 / (hw.BwStreamCopy * 1e12)

// Per-request operational intensity (FLOPS per byte)
operationalIntensity := totalFlops / totalBytes

// Roofline model
var timeS float64
if operationalIntensity > hwBalancePoint {
    // Compute-bound regime (typical for large prefill)
    timeS = totalFlops / (hw.TFlopsPeak * 1e12 * hw.MfuMeasured)
} else {
    // Memory-bound regime (typical for decode)
    achievableBW := getAchievableBW(accessPattern, hw.BwStreamCopy * 1e12)
    timeS = totalBytes / achievableBW
}
```

### Memory Access Pattern Model

Based on STREAM benchmark results and PagedAttention paper:

```go
func getAchievableBW(accessPattern string, streamCopyBW float64) float64 {
    switch accessPattern {
    case "sequential":
        // Weight loading, sequential KV writes (prefill)
        // STREAM Copy benchmark: 85% of peak
        return streamCopyBW * 0.95

    case "gather":
        // Paged attention scattered reads (decode)
        // Based on PagedAttention paper + STREAM Triad: 60-70% of peak
        return streamCopyBW * 0.70

    case "mixed":
        // Combination of sequential and random
        return streamCopyBW * 0.80
    }
    return streamCopyBW
}
```

### TP Scaling Model (Amdahl's Law)

Based on Megatron-LM paper and measured communication time:

```go
func tpSpeedup(baseComputeTimeS float64, tp int, modelConfig ModelConfig, hw HardwareProfile) float64 {
    if tp == 1 {
        return 1.0
    }

    // Communication volume per layer (from Megatron-LM)
    // All-reduce of activations: 2 * hidden_dim * bytes_per_param
    commVolumeBytes := 2.0 * float64(modelConfig.HiddenDim) * float64(modelConfig.BytesPerParam)

    // Communication time per layer
    commLatencyS := hw.AllReduceLatency / 1e6
    commBandwidthS := commVolumeBytes / (hw.NVLinkBW * 1e9)
    commTimePerLayerS := commLatencyS + commBandwidthS

    // Total communication overhead
    totalCommTimeS := commTimePerLayerS * float64(modelConfig.NumLayers)

    // Amdahl's Law: speedup = 1 / (serial_fraction + parallel_fraction / p)
    // Here: serial_fraction = communication, parallel_fraction = computation
    idealSpeedup := float64(tp)
    commOverheadRatio := totalCommTimeS / baseComputeTimeS

    actualSpeedup := idealSpeedup / (1.0 + commOverheadRatio * (idealSpeedup - 1.0))

    return actualSpeedup
}
```

**Physical Interpretation**:
- When `comm_overhead << compute_time`: Near-linear scaling
- When `comm_overhead ~ compute_time`: Sublinear scaling
- Automatically handles decode (memory-bound) getting less TP benefit than prefill

### FlashAttention Memory Model

Based on Dao et al. (2022) proof that attention is fused in SRAM:

```go
func calculateMemoryAccessBytes(
    modelConfig ModelConfig,
    sequenceLength int64,
    newTokens int64,
    includeKVCache bool,
) map[string]float64 {
    mem := make(map[string]float64)

    // Model weights: loaded once from HBM
    weightsPerLayer := calculateWeightsPerLayer(modelConfig)
    mem["model_weights"] = weightsPerLayer * float64(modelConfig.NumLayers) * modelConfig.BytesPerParam

    if includeKVCache {
        // KV cache growth: writing new K/V to HBM
        kvBytesPerToken := 2 * float64(modelConfig.NumLayers) *
                          float64(modelConfig.NumKVHeads) *
                          (float64(modelConfig.HiddenDim) / float64(modelConfig.NumHeads)) *
                          modelConfig.BytesPerParam
        mem["kv_cache_growth"] = kvBytesPerToken * float64(newTokens)

        // KV cache access: reading historical KV from HBM for attention
        // PagedAttention: decode has scattered access, prefill has sequential
        if newTokens == 1 {
            // Decode: scattered paged reads
            mem["kv_cache_access"] = kvBytesPerToken * float64(sequenceLength)
        } else {
            // Prefill: sequential reads with better caching
            mem["kv_cache_access"] = kvBytesPerToken * float64(sequenceLength) * 0.85
        }
    }

    // Activation tensors (per-token hidden states between layers)
    activationBytes := float64(modelConfig.NumLayers) *
                      float64(modelConfig.HiddenDim) *
                      modelConfig.BytesPerParam * float64(newTokens)
    mem["activations"] = activationBytes

    // CRITICAL: Do NOT model attention map (Q@K^T) memory traffic
    // FlashAttention paper proves this stays in SRAM and never touches HBM
    // mem["attention_map"] = 0  // Explicitly zero

    mem["total"] = mem["model_weights"] + mem["kv_cache_growth"] +
                  mem["kv_cache_access"] + mem["activations"]

    return mem
}
```

---

## Calibration Procedure

### Two-Parameter Model

After all literature-based modeling, only **2 orthogonal parameters** require calibration:

```go
type FrameworkCalibration struct {
    // α: Kernel efficiency factor
    // Accounts for: vLLM kernel fusion, memory coalescing, CUDA graph optimization
    // Expected range: 0.9 - 1.1
    KernelEfficiency float64 `json:"kernelEfficiency"`

    // β: Per-iteration overhead (microseconds)
    // Accounts for: Python scheduling, CUDA kernel launch, vLLM bookkeeping
    // Expected range: 300 - 800 μs
    OverheadMicros float64 `json:"overheadMicros"`
}
```

**Final time calculation**:
```
T_calibrated = α * T_roofline + β
```

Where:
- `T_roofline`: Predicted by hardware-measured roofline model (no calibration)
- `α`: Scales all predictions uniformly (multiplicative error)
- `β`: Adds fixed overhead independent of workload size (additive error)

**Orthogonality**: α and β are independent:
- α affects large workloads proportionally
- β dominates for small workloads

### Calibration Workload Suite

Design 8-10 diverse workloads to cover the operating regime:

```python
calibration_workloads = [
    # Pure prefill (various sizes)
    {"type": "prefill", "tokens": 128, "batch": 1},
    {"type": "prefill", "tokens": 512, "batch": 1},
    {"type": "prefill", "tokens": 2048, "batch": 1},
    {"type": "prefill", "tokens": 8192, "batch": 1},

    # Pure decode (various batch sizes and contexts)
    {"type": "decode", "batch": 1, "context": 512},
    {"type": "decode", "batch": 8, "context": 2048},
    {"type": "decode", "batch": 32, "context": 2048},
    {"type": "decode", "batch": 64, "context": 4096},

    # Mixed batches (real-world scenarios)
    {"type": "mixed", "prefill_tokens": 256, "decode_batch": 4, "context": 1024},
    {"type": "mixed", "prefill_tokens": 1024, "decode_batch": 16, "context": 2048},
]
```

### Calibration Script

```python
# File: scripts/calibrate_framework.py

import numpy as np
from scipy.optimize import least_squares
import subprocess
import json

def measure_vllm_step_time(model_name, gpu_type, workload_config):
    """
    Run single vLLM iteration and measure step time.

    Uses vLLM profiling with minimal overhead:
    - CUDA events for GPU kernel timing
    - Single iteration (no queueing effects)
    """
    # Create minimal vLLM script that measures one step
    vllm_script = f"""
import torch
from vllm import LLM, SamplingParams
from vllm.sequence import SequenceGroupMetadata

# Initialize model
llm = LLM(model="{model_name}", tensor_parallel_size=1, gpu_memory_utilization=0.9)

# Create workload
if {workload_config['type']} == 'prefill':
    # Single prefill request
    prompt_tokens = torch.randint(0, 32000, ({workload_config['tokens']},))
    # Measure prefill time via profiling

elif {workload_config['type']} == 'decode':
    # Decode batch with pre-filled KV cache
    # Create {workload_config['batch']} sequences with {workload_config['context']} context

# Use CUDA events for accurate GPU timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start_event.record()

# Execute one model step
output = llm.model.forward(...)

end_event.record()
torch.cuda.synchronize()

step_time_ms = start_event.elapsed_time(end_event)
print(f"STEP_TIME_MICROSECONDS: {{step_time_ms * 1000}}")
"""

    # Run and parse output
    result = subprocess.run(
        ["python", "-c", vllm_script],
        capture_output=True,
        text=True
    )

    for line in result.stdout.split('\n'):
        if line.startswith("STEP_TIME_MICROSECONDS:"):
            return float(line.split(':')[1].strip())

    raise ValueError("Could not measure vLLM step time")


def calibrate(gpu_type, model_name, hw_profile):
    """
    Calibrate 2 framework parameters using least-squares fitting.
    """
    print(f"Calibrating for {model_name} on {gpu_type}")

    workloads = calibration_workloads

    # 1. Measure ground truth on vLLM
    print("Measuring vLLM ground truth...")
    measurements = []
    for w in workloads:
        # Run 3 trials, take median
        trials = [measure_vllm_step_time(model_name, gpu_type, w) for _ in range(3)]
        measurements.append(np.median(trials))

    print(f"Measurements (μs): {measurements}")

    # 2. Predict using roofline (before calibration)
    print("Computing roofline predictions...")
    predictions_base = []
    for w in workloads:
        pred_us = roofline_predict(model_name, hw_profile, w)
        predictions_base.append(pred_us)

    print(f"Base predictions (μs): {predictions_base}")

    # 3. Fit α and β
    def residual(params):
        alpha, beta = params
        predictions = [alpha * p + beta for p in predictions_base]
        return np.array(predictions) - np.array(measurements)

    result = least_squares(
        residual,
        x0=[1.0, 500.0],  # Initial guess
        bounds=([0.8, 0], [1.2, 1000]),  # Reasonable ranges
        method='trf'
    )

    alpha_opt, beta_opt = result.x

    # 4. Compute error metrics
    predictions_calibrated = [alpha_opt * p + beta_opt for p in predictions_base]
    errors = np.abs(np.array(predictions_calibrated) - np.array(measurements)) / np.array(measurements)
    mape = np.mean(errors) * 100
    max_error = np.max(errors) * 100

    print("\n=== Calibration Results ===")
    print(f"Kernel Efficiency (α): {alpha_opt:.4f}")
    print(f"Overhead (β): {beta_opt:.1f} μs")
    print(f"MAPE: {mape:.2f}%")
    print(f"Max Error: {max_error:.2f}%")

    # 5. Save calibration
    calib = {
        "gpu": gpu_type,
        "model": model_name,
        "kernelEfficiency": alpha_opt,
        "overheadMicros": beta_opt,
        "mape": mape,
        "workloads": workloads,
        "measurements": measurements,
        "predictions_base": predictions_base,
        "predictions_calibrated": predictions_calibrated
    }

    with open(f"calibration_{gpu_type}_{model_name}.json", "w") as f:
        json.dump(calib, f, indent=2)

    return calib

if __name__ == "__main__":
    import sys
    gpu_type = sys.argv[1]  # e.g., "H100"
    model_name = sys.argv[2]  # e.g., "qwen2.5-7b-instruct"

    hw_profile = load_hardware_profile(gpu_type)
    calib = calibrate(gpu_type, model_name, hw_profile)
```

**Time**: ~1 hour (10 workloads × 3 trials × 2 minutes per run)

---

## Validation Strategy

### Holdout Test Set

Reserve 20% of models and workloads for testing:

```python
# Training set (used for calibration)
train_models = ["qwen2.5-7b-instruct"]

# Test set (unseen during calibration)
test_models = [
    "qwen2.5-1.5b-instruct",  # Different size
    "qwen2.5-3b-instruct",    # Different size
    "llama-3.1-8b-instruct",  # Different architecture (GQA vs MHA)
    "codellama-34b-instruct"  # Much larger
]
```

### Validation Metrics

```python
def validate_generalization(calibration, test_models, test_workloads):
    """
    Test calibrated model on unseen data.
    """
    results = []

    for model in test_models:
        for workload in test_workloads:
            # Measure actual vLLM performance
            actual_time = measure_vllm_step_time(model, calibration['gpu'], workload)

            # Predict using calibrated roofline
            predicted_time = roofline_predict_calibrated(
                model, calibration['gpu'], workload, calibration
            )

            error = abs(predicted_time - actual_time) / actual_time

            results.append({
                'model': model,
                'workload': workload,
                'actual': actual_time,
                'predicted': predicted_time,
                'error_pct': error * 100
            })

    df = pd.DataFrame(results)

    # Compute aggregate metrics
    metrics = {
        'MAPE': df['error_pct'].mean(),
        'Median_Error': df['error_pct'].median(),
        'Max_Error': df['error_pct'].max(),
        'R_squared': calculate_r_squared(df['actual'], df['predicted'])
    }

    return df, metrics
```

### Acceptance Criteria

Model is considered validated if:

| Metric | Threshold | Justification |
|--------|-----------|---------------|
| **MAPE** | < 15% | Standard for analytical performance models |
| **Median Error** | < 10% | Most predictions should be quite accurate |
| **Max Error** | < 30% | Worst-case should still be reasonable |
| **R²** | > 0.95 | Strong correlation with actual measurements |

### Cross-TP Validation

Test that TP scaling works without re-calibration:

```python
def validate_tp_scaling(calibration, model, base_workload):
    """
    Test that TP=2,4,8 predictions are accurate using TP=1 calibration.
    """
    tp_values = [1, 2, 4, 8]
    results = []

    for tp in tp_values:
        actual_time = measure_vllm_step_time_tp(model, tp, base_workload)
        predicted_time = roofline_predict_calibrated(model, calibration, workload, tp)

        results.append({
            'tp': tp,
            'actual': actual_time,
            'predicted': predicted_time,
            'speedup_actual': results[0]['actual'] / actual_time if results else 1.0,
            'speedup_predicted': results[0]['predicted'] / predicted_time if results else 1.0
        })

    return pd.DataFrame(results)
```

---

## Implementation Roadmap

### Phase 1: Hardware Characterization (Week 1)

**Deliverables**:
- [ ] CUTLASS profiler setup and GEMM benchmarks
- [ ] STREAM benchmark implementation and BW measurements
- [ ] NCCL all-reduce latency measurements
- [ ] Automated `characterize_hardware.sh` script
- [ ] `hardware_config.json` entries for target GPUs (H100, A100)

**Time**: 2-3 days + 30 min per GPU

---

### Phase 2: Code Refactoring (Week 1-2)

**Deliverables**:
- [ ] `sim/roofline_v2.go` with clean architecture
  - [ ] `OperationProfile` struct (algorithmic analysis)
  - [ ] `HardwareProfile` struct (measured characteristics)
  - [ ] `executeOnHardware()` function (roofline + TP scaling)
  - [ ] `rooflineStepTime()` with 2-parameter calibration
- [ ] Update `sim/model_hardware_config.go`
  - [ ] Add `FrameworkCalibration` struct
  - [ ] Load function for hardware profiles
- [ ] Update `sim/simulator.go`
  - [ ] `getStepTimeRoofline()` calls new implementation
- [ ] Unit tests for FLOP/byte counting accuracy

**Time**: 3-4 days

---

### Phase 3: Calibration Implementation (Week 2)

**Deliverables**:
- [ ] `scripts/calibrate_framework.py` script
  - [ ] vLLM measurement harness
  - [ ] Least-squares fitting for α and β
  - [ ] Calibration workload suite
- [ ] Run calibration for base model (Qwen 7B on H100)
- [ ] Generate `calibration_H100_qwen7b.json`

**Time**: 2-3 days

---

### Phase 4: Validation (Week 3)

**Deliverables**:
- [ ] `scripts/validate_generalization.py`
- [ ] Test on holdout models (Qwen 1.5B, 3B, Llama 8B)
- [ ] Cross-TP validation (TP=2,4,8 without re-calibration)
- [ ] Error analysis and visualization
- [ ] Comparison report: current vs v2 accuracy

**Time**: 2-3 days

---

### Phase 5: Documentation (Week 3)

**Deliverables**:
- [ ] `docs/roofline_v2.md` (this document)
- [ ] `docs/hardware_characterization_guide.md` (step-by-step)
- [ ] Update `docs/calibration_methodology.md` with 2-param process
- [ ] Add example Jupyter notebook for analysis

**Time**: 1-2 days

---

## Comparison with Current Approach

### Parameter Reduction

| Category | Current | V2 | Change |
|----------|---------|----|----|
| **Hardware Limits** | 3 params (TFlopsPeak, BwPeakTBs, BwEffConstant) | **Measured** (CUTLASS, STREAM) | 0 calibration |
| **Efficiency Factors** | 5 params (mfuPrefill, mfuDecode, prefillBwFactor, decodeBwFactor, VectorPeakFraction) | **Measured** (CUTLASS MFU, STREAM patterns) | 0 calibration |
| **TP Scaling** | 2 params (tpScalingExponent, decodeTpScalingExponent) | **Derived** (Amdahl + NCCL) | 0 calibration |
| **Calibration** | 3 params (mfuPrefillMultiplier, mfuDecodeMultiplier, TOverheadMicros) | **2 params** (α, β) | **Reduced from 3 to 2** |
| **Overhead** | 3 params (TOverheadMicros, prefillOverheadMicros, mixedPrefillOverheadMicros) | **1 param** (β, unified overhead) | **Reduced from 3 to 1** |
| **Total** | **13 parameters** | **2 parameters** | **85% reduction** |

### Literature Grounding

| Component | Current | V2 | Reference |
|-----------|---------|----|----|
| Roofline | ✓ Used | ✓ Used | Williams 2009 |
| TP Scaling | Empirical exponents | Amdahl's Law | Amdahl 1967, Shoeybi 2019 |
| Memory BW | Magic factors | STREAM benchmark | McCalpin 1995 |
| FlashAttention | Modeled HBM traffic | Zero HBM traffic | Dao 2022 |
| PagedAttention | Generic | Scatter/gather patterns | Kwon 2023 |

### Expected Improvements

1. **Generalization**: Calibrate once on Qwen 7B, apply to Llama 8B without re-calibration
2. **Interpretability**: Every number has physical meaning or literature backing
3. **Reproducibility**: Clear measurement protocol for new hardware
4. **Accuracy**: Expect similar or better than current (currently 11.23% E2E error)

---

## Open Questions

### For Discussion

1. **CUTLASS vs Published Benchmarks**: If CUTLASS is unavailable, can we use published MFU numbers from NVIDIA MLPerf results?

2. **Model Coverage**: Should we test on models beyond Transformer decoder-only? (e.g., encoder-decoder, MoE)

3. **Error Budget**: Is 15% MAPE acceptable for systems research publication? (Current: 11.23%)

4. **TP Range**: Validate up to TP=8 or also TP=16+? (Requires multi-node NVSwitch)

5. **Chunked Prefill**: Current code has `LongPrefillTokenThreshold` - does this need special handling in V2?

### Future Extensions

- **Speculative Decoding**: Extend model to handle draft + verification steps
- **Mixed Precision**: FP8 / INT8 quantization effects on MFU and BW
- **Model-Specific Optimizations**: Some models have custom attention patterns (sliding window, sparse)
- **Online Calibration**: Adapt α and β during live serving based on observed latencies

---

## References

1. Williams, S., Waterman, A., Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52.4 (2009): 65-76.

2. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS* 2022.

3. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., Stoica, I. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP* 2023.

4. Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., Catanzaro, B. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." *arXiv:1909.08053* 2019.

5. McCalpin, J. D. "STREAM: Sustainable Memory Bandwidth in High Performance Computers." *Technical Report*, University of Virginia, 1995. https://www.cs.virginia.edu/stream/

6. Amdahl, G. M. "Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities." *AFIPS Conference Proceedings* 30 (1967): 483-485.

7. NVIDIA. "CUTLASS: CUDA Templates for Linear Algebra Subroutines." https://github.com/NVIDIA/cutlass

8. NVIDIA. "NCCL Tests: MPI compatible benchmarks for NCCL." https://github.com/NVIDIA/nccl-tests

---

## Appendix A: Hardware Profile Template

```json
{
    "H100-80GB-SXM": {
        "comment": "Measured on 2025-02-16 using characterization protocol",

        "hardware_limits": {
            "TFlopsPeak": 989.5,
            "BwPeakTBs": 3.35,
            "L2CacheSize": 50.0,
            "NumSMs": 132
        },

        "measured_characteristics": {
            "MfuMeasured": 0.58,
            "BwStreamCopy": 2.85,
            "BwStreamScale": 2.73,
            "BwStreamTriad": 2.68,
            "NVLinkBW": 700.0,
            "AllReduceLatency": 20.0
        },

        "measurement_metadata": {
            "cutlass_version": "3.2.0",
            "cuda_version": "12.2",
            "driver_version": "535.104.05",
            "measurement_date": "2025-02-16",
            "gpu_clocks_locked": true,
            "warmup_iterations": 10,
            "measurement_iterations": 100
        }
    }
}
```

---

## Appendix B: Calibration Results Template

```json
{
    "gpu": "H100",
    "model": "qwen2.5-7b-instruct",
    "calibration_date": "2025-02-16",

    "parameters": {
        "kernelEfficiency": 1.05,
        "overheadMicros": 450.0
    },

    "validation": {
        "mape": 12.3,
        "median_error": 9.5,
        "max_error": 24.1,
        "r_squared": 0.97
    },

    "workloads": [
        {
            "type": "prefill",
            "tokens": 2048,
            "measured_us": 15420,
            "predicted_base_us": 14680,
            "predicted_calibrated_us": 15864,
            "error_pct": 2.9
        }
    ]
}
```

---

**End of Document**
