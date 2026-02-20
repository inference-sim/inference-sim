# Implementation Plan Review: MFU-Based Roofline Model

## Review Update (2026-02-20)

**Sections Removed After Investigation:**
- ❌ Memory Bandwidth Over-Simplification (MEDIUM)
- ❌ Decode Aggregation Design Decision (MEDIUM)
- ❌ Prefill Bucketing Granularity (LOW-MEDIUM)

**Rationale:**
After examining InferSim technical report (page 3) and source code (`InferSim/layers/attn.py`), confirmed that **MFU values already capture memory bottlenecks** within kernel execution. InferSim states: "for the computing bottleneck and the memory access bottleneck, they are both reflected in the MFU." The V2 implementation correctly follows InferSim's design by:
1. Using MFU-based timing for kernel execution (memory-bound behavior already captured)
2. Using separate bandwidth calculation only for KV cache loading (not in kernel benchmarks)

These issues should be **validated first, fixed only if proven wrong**:
- Validate with `blis_evaluator.py` against vLLM ground truth
- Only add bandwidth efficiency factors if decode error >20%
- Max KV aggregation likely correct (vLLM PagedAttention pads to max)
- Bucketing acceptable (MFU scaling offsets approximation errors)

**What Was Fixed:**
✅ Zero MFU fallback - Changed from "first non-zero" to "nearest by distance" (`sim/mfu_database.go`)

---

## Problem Statement

### Context

Large Language Model (LLM) inference simulation is critical for:
- **Model-system co-design**: Predicting performance before training/deployment
- **Hardware planning**: Determining GPU requirements and parallelization strategies
- **Performance optimization**: Identifying compute vs memory bottlenecks

Current state-of-the-art simulators like **InferSim** (Alibaba, 2025) achieve 4-15% prediction error by using **MFU (Model FLOPs Utilization)** lookup tables from benchmarked kernels rather than theoretical peak performance. We want to modify the current roofline implementation in (`sim/roofline_step.go`) to accurately estimate vLLM step times using InferSim's benchmarked MFU values.

### Relevant Sources
* InferSim codebase (https://github.com/inference-sim/InferSim/tree/main)
* InferSim technical report (`docs/infersim_tech_report (1).pdf`)
* Benchmarked H100 InferSim coefficients (`bench_data`)
* vLLM CUDA codebase (https://github.com/vllm-project/vllm/tree/main/csrc)

### Current State: Roofline V1 Limitations

The existing Go implementation (`sim/roofline_step.go`) uses a **calibrated roofline model** with the following limitations:

#### 1. Fixed MFU Values
```go
adjustedPrefillMFU := hwConfig.MfuPrefill * hwConfig.MfuPrefillMultiplier
adjustedDecodeMFU := hwConfig.MfuDecode * hwConfig.MfuDecodeMultiplier
```
- Single MFU value for all prefill operations (regardless of sequence length)
- Single MFU value for all decode operations (regardless of batch size or KV length)
- Does not account for shape-dependent kernel efficiency

#### 2. Empirical Calibration Factors
```go
prefillEffBW := effBW * hwConfig.PrefillBwFactor
decodeEffBW := effBW * hwConfig.DecodeBwFactor
effectiveTpPrefill := math.Pow(tpFactor, hwConfig.TpScalingExponent)
```
- 15+ tunable parameters per hardware configuration
- Requires extensive vLLM profiling to calibrate
- Not generalizable across models or workload patterns

#### 3. Aggregate Computation
- Lumps all GEMM operations into single FLOPs calculation
- Cannot distinguish between Q/K/V projections, MLP layers with different shapes
- Misses per-operation efficiency variations

#### 4. No Attention-Specific Modeling
```go
decodeComputeS = (dGemmFlops / (peakFlops * adjustedDecodeMFU)) + (dVectorFlops / vectorPeak)
```
- Attention core operations use `vectorPeak` (artificial 10% of peak FLOPs)
- Doesn't reflect actual FlashAttention/FlashInfer MFU characteristics
- Ignores sequence length and batch size impact on attention efficiency

---

## Implementation Plan Summary

The implementation plan proposes a comprehensive transition from calibrated roofline to **MFU-based roofline** by:

### Architecture Overview
- **Load pre-computed H100 MFU benchmark data** from CSV files at simulator initialization
- **Replace fixed MFU constants** with dynamic lookups based on operation shapes
- **Integrate with existing roofline model** by aggregating decode requests and bucketing prefill requests
- **Port InferSim's lookup logic** to Go using standard library (encoding/csv, math, fmt)

### Key Components

#### 1. MFU Database (`sim/mfu_database.go`)
- **Data Structures**: MHAPrefillRow, MHADecodeRow, GEMMRow, AttentionShape
- **CSV Loaders**: loadPrefillCSV, loadDecodeCSV, loadGEMMCSV
- **Lookup Functions**:
  - `GetAttnPrefillMFU(seqLen)`: Floor lookup on seq_len
  - `GetAttnDecodeMFU(batchSize, kvLen, tp)`: 2D floor lookup with TP-specific data
  - `GetGEMMmfu(m, k, n)`: Two-stage lookup (ceiling on k,n then floor on m)
- **Fallback Strategy**: Nearest neighbor using Euclidean distance for missing attention configs

#### 2. Per-Operation GEMM Calculation (`sim/roofline_step.go`)
- **computeGEMMTime**: Calculate time for individual GEMM with MFU lookup
- **computeTransformerGEMMTimes**: Aggregate Q/K/V/O projections + MLP Gate/Up/Down
- Separate lookups for each projection type to capture shape-dependent efficiency

#### 3. Decode Phase Modifications
- **Aggregate batch size** across all decode requests
- **Find max KV length** for attention core lookup
- **Single GEMM lookup** for aggregated batch (not per-request)
- **Attention core MFU** from decode CSV: `GetAttnDecodeMFU(totalBatchSize, maxKVLen, tp)`
- **TP scaling**: Simple linear division (MFU data captures efficiency)

#### 4. Prefill Phase Modifications
- **Bucket requests by seq_len** (not exact length, bucketed)
- **Process each bucket independently** with separate MFU lookups
- **Per-bucket GEMM lookups** using batch size × seq_len
- **Attention core MFU** from prefill CSV: `GetAttnPrefillMFU(seqLen)`
- **Hardware-specific factor**: InferSim divides prefill attention by 1.8

#### 5. Simplified Hardware Config
- **Remove calibration factors**: MfuPrefillMultiplier, MfuDecodeMultiplier, PrefillBwFactor, DecodeBwFactor, VectorPeakFraction, TpScalingExponent
- **Keep only**: TFlopsPeak, BwPeakTBs, overhead parameters
- MFU database replaces ~10+ calibration parameters with benchmark data

### Implementation Tasks (12 Total)
1. Create MFU data structures
2. Implement attention config computation
3. Implement CSV loading functions
4. Implement database initialization
5. Implement lookup functions
6. Add helper function for individual GEMM calculations
7. Integrate MFU database into roofline (decode phase)
8. Integrate MFU database into roofline (prefill phase)
9. Initialize MFU database in simulator
10. Test end-to-end integration
11. Add documentation
12. Final verification and cleanup

---

## Background Context

### Current Codebase Analysis

#### Roofline Model V1 (`sim/roofline_step.go`)

**Current Architecture:**
```go
func rooflineStepTime(modelConfig, hwConfig, stepConfig, tp) int64 {
    // 1. PREFILL PHASE
    for req in stepConfig.PrefillRequests {
        pGemmFlops += f["gemm_ops"] / effectiveTpPrefill
        pVectorFlops += f["sram_ops"] / effectiveTpPrefill
    }
    adjustedPrefillMFU := hwConfig.MfuPrefill * hwConfig.MfuPrefillMultiplier
    prefillComputeS = (pGemmFlops / (peakFlops * adjustedPrefillMFU)) + (pVectorFlops / vectorPeak)

    // 2. DECODE PHASE
    for req in stepConfig.DecodeRequests {
        dGemmFlops += f["gemm_ops"] / effectiveTpDecode
        dVectorFlops += f["sram_ops"] / effectiveTpDecode
    }
    adjustedDecodeMFU := hwConfig.MfuDecode * hwConfig.MfuDecodeMultiplier
    decodeComputeS = (dGemmFlops / (peakFlops * adjustedDecodeMFU)) + (dVectorFlops / vectorPeak)

    // 3. COMBINE PHASES
    stepHardwareS = math.Max(prefillComputeS, prefillMemoryS) or math.Max(decodeComputeS, decodeMemoryS)
}
```

**Key Observations:**
- Aggregates all FLOPs before applying single MFU value
- Uses `calculateTransformerFlops()` which returns `gemm_ops` (Q/K/V/O + MLP) and `sram_ops` (attention core)
- Separates compute-bound (FLOPs / MFU) from memory-bound (bytes / bandwidth)
- Applies roofline principle: `max(compute_time, memory_time)`
- Has complex TP scaling: sublinear for prefill (`tpFactor^0.68`), linear for decode
- Mixed batch handling: token-weighted average with heuristics

**FLOP Calculation (`calculateTransformerFlops`):**
```go
// Attention QKV projections
qkvFlops := 2 * newT * (dModel*dModel + 2*dModel*dKV)
projFlops := 2 * newT * dModel * dModel
flops["gemm_ops"] = (qkvFlops + projFlops) * nLayers

// FlashAttention core
qkMatMul := 2 * nHeads * newT * effectiveCtx * dHead
softmaxOps := 4 * nHeads * newT * effectiveCtx
avMatMul := 2 * nHeads * newT * effectiveCtx * dHead
flops["sram_ops"] = (qkMatMul + softmaxOps + avMatMul) * nLayers

// MLP (SwiGLU)
flops["gemm_ops"] += 2 * newT * (3 * dModel * dFF) * nLayers
```

**Memory Calculation (`calculateMemoryAccessBytes`):**
- Model weights: `weightsPerLayer * nLayers * BytesPerParam`
- KV cache growth: `2 * nLayers * nKVHeads * dHead * newT`
- KV cache access: Different factors for prefill (0.92) vs decode (0.80)
- Activation tokens: Different factors for prefill (0.85) vs decode (0.75)

#### Existing Roofline V2 Implementation (`sim/roofline_step_v2.go`)

**Status:** Already partially implemented on current branch!

**Architecture Changes:**
```go
func rooflineStepTimeV2(modelConfig, hwConfig, stepConfig, tp, mfuDB) int64 {
    // 1. DECODE PHASE (Aggregate)
    totalBatchSize := len(stepConfig.DecodeRequests)
    maxKVLen := max(req.ProgressIndex)

    gemmTimeS := computeTransformerGEMMTimes(modelConfig, totalBatchSize, peakFlops, mfuDB, tpScaling)
    attnCoreFLOPs := calculateAttentionCoreFLOPs(...)
    attnMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, maxKVLen, tp)
    attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

    // 2. PREFILL PHASE (Bucket by seq_len)
    bucketMap := groupByPowerOf2Bucket(stepConfig.PrefillRequests)
    for bucketSeqLen, requests := range bucketMap {
        gemmTimeS := computeTransformerGEMMTimes(...)
        attnMFU := mfuDB.GetAttnPrefillMFU(bucketSeqLen)
        attnCoreTimeS := attnCoreFLOPs / 1.8 / (peakFlops * attnMFU) * tpScaling
    }
}
```

**Key Differences from V1:**
- Per-operation GEMM lookups instead of aggregate FLOPs
- Separate attention core calculation with dedicated MFU
- Power-of-2 bucketing for prefill (512, 1024, 2048, 4096, etc.)
- Simplified TP scaling (1/tp instead of power law)
- No vectorPeak - attention uses same peakFlops with different MFU
- Hardware factor: `/1.8` for prefill attention (from InferSim)

**Helper Functions:**
```go
computeGEMMTime(m, k, n, peakFlops, mfuDB) float64 {
    flops := 2.0 * m * k * n
    mfu := mfuDB.GetGEMMmfu(m, k, n)
    return flops / (peakFlops * mfu)
}

computeTransformerGEMMTimes(modelConfig, batchSize, peakFlops, mfuDB, tpScaling) float64 {
    // Per layer:
    qTime := computeGEMMTime(batchSize, dModel, dModel, peakFlops, mfuDB)
    kTime := computeGEMMTime(batchSize, dModel, dKV, peakFlops, mfuDB)
    vTime := computeGEMMTime(batchSize, dModel, dKV, peakFlops, mfuDB)
    oTime := computeGEMMTime(batchSize, dModel, dModel, peakFlops, mfuDB)
    gateTime := computeGEMMTime(batchSize, dModel, dFF, peakFlops, mfuDB)
    upTime := computeGEMMTime(batchSize, dModel, dFF, peakFlops, mfuDB)
    downTime := computeGEMMTime(batchSize, dFF, dModel, peakFlops, mfuDB)

    return (qTime + kTime + vTime + oTime + gateTime + upTime + downTime) * nLayers * tpScaling
}
```

#### MFU Database Implementation (`sim/mfu_database.go`)

**Status:** Fully implemented!

**Data Loading:**
- Loads from `bench_data/mha/prefill/{gpu}/*.csv` (e.g., `32-32-128.csv`)
- Loads from `bench_data/mha/decode/{gpu}/*.csv` (e.g., `32-32-128-tp1.csv`)
- Loads from `bench_data/gemm/{gpu}/data.csv`

**Attention Config Matching:**
```go
computeAttentionConfig(config) -> "32-32-128" (NumHeads-NumKVHeads-HeadDim)
findNearestConfig(target, available) -> Uses Euclidean distance if exact match missing
```

**Lookup Strategies:**

1. **Prefill Attention** (`GetAttnPrefillMFU`):
   - Prefers floor matches (seq_len <= target)
   - Falls back to ceiling if no floor available
   - Uses nearest neighbor with distance metric
   - Zero-MFU protection with fallback

2. **Decode Attention** (`GetAttnDecodeMFU`):
   - 2D lookup on (batch_size, kv_len)
   - Prefers floor in both dimensions
   - Euclidean distance for nearest neighbor
   - TP-specific files (`-tp1`, `-tp2`, `-tp4`)
   - Falls back to TP=1 if specific TP unavailable

3. **GEMM** (`GetGEMMmfu`):
   - Stage 1: Find smallest (k, n) where k >= target_k AND n >= target_n (ceiling)
   - Stage 2: Within that (k, n), find largest m <= target_m (floor)
   - Falls back to largest available (k, n) if no ceiling match
   - Falls back to smallest m if no floor match

**Error Handling:**
- Fatal errors for missing CSV files (database corruption)
- Informative warnings for nearest neighbor fallbacks
- Zero-MFU protection to prevent division by zero

#### Simulator Integration (`sim/simulator.go`)

**Current Integration:**
```go
type Simulator struct {
    ModelConfig  ModelConfig
    HWConfig     HardwareCalib
    MFUDatabase  *MFUDatabase  // Field already exists!
    // ... other fields
}

func NewSimulator(..., mfuDB *MFUDatabase) *Simulator {
    s := &Simulator{
        MFUDatabase: mfuDB,
        // ...
    }
}

func (sim *Simulator) getStepTimeRoofline() int64 {
    // Use MFU-based roofline if database is available
    if sim.MFUDatabase != nil {
        stepTime = rooflineStepTimeV2(sim.GPU, sim.ModelConfig, sim.HWConfig, stepConfig, sim.TP, sim.MFUDatabase)
    } else {
        stepTime = rooflineStepTime(sim.GPU, sim.ModelConfig, sim.HWConfig, stepConfig, sim.TP)
    }
    return stepTime
}
```

**Observation:** The codebase already has conditional logic to use V2 if MFU database is provided!

#### Hardware Configuration Structure

**Current Hardware Config (`hardware_config.json`):**
```json
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35
    },
    "A100-SXM": {
        "TFlopsPeak": 312,
        "BwPeakTBs": 2.039
    }
}
```

**HardwareCalib Struct:**
```go
type HardwareCalib struct {
    // Core hardware specs
    TFlopsPeak       float64 `json:"TFlopsPeak"`
    BwPeakTBs        float64 `json:"BwPeakTBs"`

    // Overheads (still needed)
    TOverheadMicros  float64 `json:"TOverheadMicros"`
    AllReduceLatency float64 `json:"allReduceLatency"`
    PrefillOverheadMicros      float64 `json:"prefillOverheadMicros"`
    MixedPrefillOverheadMicros float64 `json:"mixedPrefillOverheadMicros"`

    // Calibration factors (deprecated with MFU database)
    BwEffConstant              float64 `json:"BwEffConstant"`
    MfuPrefill                 float64 `json:"mfuPrefill"`
    MfuDecode                  float64 `json:"mfuDecode"`
    TpScalingExponent          float64 `json:"tpScalingExponent"`
    DecodeTpScalingExponent    float64 `json:"decodeTpScalingExponent"`
    MfuPrefillMultiplier       float64 `json:"mfuPrefillMultiplier"`
    MfuDecodeMultiplier        float64 `json:"mfuDecodeMultiplier"`
    PrefillBwFactor            float64 `json:"prefillBwFactor"`
    DecodeBwFactor             float64 `json:"decodeBwFactor"`
    VectorPeakFraction         float64 `json:"vectorPeakFraction"`
}
```

**Observation:** The simplified hardware config only needs TFlopsPeak and BwPeakTBs. Many calibration fields become unused with MFU database.

---

### Benchmark Data Structure

**Directory Layout:**
```
bench_data/
├── gemm/
│   └── h100/
│       └── data.csv
└── mha/
    ├── prefill/
    │   └── h100/
    │       ├── 32-32-128.csv
    │       ├── 32-8-128.csv
    │       ├── 64-8-128.csv
    │       ├── 28-4-128.csv
    │       ├── 40-40-128.csv
    │       └── 56-8-128.csv
    └── decode/
        └── h100/
            ├── 32-32-128-tp1.csv
            ├── 32-32-128-tp2.csv
            ├── 32-32-128-tp4.csv
            ├── 32-8-128-tp1.csv
            ├── 32-8-128-tp2.csv
            ├── 32-8-128-tp4.csv
            └── ... (similar for other configs)
```

**CSV Formats:**

1. **Prefill MHA** (`mha/prefill/h100/32-32-128.csv`):
```csv
dtype,seq_len,latency_us,mfu
bf16,1024,45.85,1.266
bf16,4096,219.951,4.222
bf16,8192,771.621,4.814
bf16,16384,3107.445,4.782
bf16,32768,12628.475,4.706
```

**Key Observations:**
- MFU values are > 1.0 (not normalized to 0-1 range like decode)
- Likely represents attention core throughput relative to peak
- Varies with sequence length (efficiency changes)

2. **Decode MHA** (`mha/decode/h100/32-32-128-tp1.csv`):
```csv
dtype,kv_dtype,batch_size,kv_len,latency_us,mfu
bf16,bf16,1,1024,36.041,0.0
bf16,bf16,1,4096,33.06,0.002
bf16,bf16,1,8192,55.547,0.002
bf16,bf16,1,16384,96.921,0.003
bf16,bf16,1,32768,180.529,0.003
bf16,bf16,1,65536,351.278,0.003
bf16,bf16,1,131072,688.56,0.003
bf16,bf16,16,1024,96.576,0.003
bf16,bf16,16,4096,346.871,0.003
```

**Key Observations:**
- MFU values very low (~0.002-0.003) - memory-bound operation
- TP-specific files (tp1, tp2, tp4) capture scaling behavior
- 2D grid: batch_size × kv_len
- Some zero MFU values (e.g., bs=1, kv=1024) - database has fallback handling

3. **GEMM** (`gemm/h100/data.csv`):
```csv
m,k,n,latency_us,mfu
8,2048,6144,8.878,0.023
16,2048,6144,8.813,0.046
32,2048,6144,8.819,0.092
64,2048,6144,8.682,0.187
128,2048,6144,10.113,0.322
256,2048,6144,11.402,0.571
512,2048,6144,15.605,0.834
1024,2048,6144,25.552,1.019
4096,2048,6144,90.913,1.146
```

**Key Observations:**
- MFU increases with m (batch size) - better utilization
- MFU can exceed 1.0 (similar to prefill) - likely denominator issue
- Fixed (k, n) pairs with varying m
- Two-stage lookup makes sense: find (k, n) bucket, then m within bucket

**Data Characteristics:**
- **Coverage**: Limited attention configs (6 configs for H100)
- **Granularity**: Power-of-2 seq_len for prefill, sparse grid for decode
- **TP Scaling**: Separate files per TP level (explicit measurement)
- **Quality**: Some zero/near-zero MFU values require fallback handling

---

### Architecture Considerations

#### Integration Points

**1. Simulator Initialization Flow:**
```
main.go (CLI)
  └─> NewSimulator(modelConfig, hwConfig, mfuDB)
       ├─> Load model_configs/{model}/config.json -> ModelConfig
       ├─> Load hardware_config.json -> HardwareCalib
       └─> NewMFUDatabase(modelConfig, "bench_data", gpu) -> MFUDatabase
            ├─> Compute attention config: "32-32-128"
            ├─> Load all CSV files (prefill, decode, gemm)
            ├─> Find nearest config if exact match missing
            └─> Return initialized database
```

**2. Step Execution Flow:**
```
Simulator.Step(now)
  ├─> makeRunningBatch(now)  // Fill batch from wait queue
  ├─> getStepTimeRoofline()  // Estimate step time
  │    ├─> Build StepConfig from RunningBatch
  │    │    ├─> PrefillRequests: {ProgressIndex, NumNewPrefillTokens}
  │    │    └─> DecodeRequests: {ProgressIndex, NumNewDecodeTokens}
  │    │
  │    └─> rooflineStepTimeV2(modelConfig, hwConfig, stepConfig, tp, mfuDB)
  │         ├─> DECODE PHASE
  │         │    ├─> Aggregate totalBatchSize, maxKVLen
  │         │    ├─> computeTransformerGEMMTimes()
  │         │    │    └─> For each layer:
  │         │    │         ├─> computeGEMMTime(bs, dModel, dModel, peakFlops, mfuDB)
  │         │    │         │    └─> mfuDB.GetGEMMmfu(bs, dModel, dModel)
  │         │    │         └─> ... (7 GEMMs total per layer)
  │         │    │
  │         │    └─> mfuDB.GetAttnDecodeMFU(totalBatchSize, maxKVLen, tp)
  │         │
  │         ├─> PREFILL PHASE
  │         │    ├─> Bucket requests by power-of-2 seq_len
  │         │    ├─> For each bucket:
  │         │    │    ├─> computeTransformerGEMMTimes()
  │         │    │    └─> mfuDB.GetAttnPrefillMFU(bucketSeqLen)
  │         │    └─> Sum across buckets
  │         │
  │         └─> COMBINE: max(compute, memory) with mixed batch heuristics
  │
  └─> Update request progress and schedule next step
```

**3. MFU Lookup Flow:**
```
GetGEMMmfu(m=128, k=4096, n=4096)
  ├─> Stage 1: Find smallest (k, n) >= (4096, 4096)
  │    └─> Scan gemmData, compute Euclidean distance
  │         └─> Result: (k=4096, n=4096) or (k=8192, n=8192)
  │
  ├─> Stage 2: Within (k, n), find largest m <= 128
  │    └─> Scan filtered rows: m in {8, 16, 32, 64, 128, 256, 512, 1024}
  │         └─> Result: m=128
  │
  └─> Return: MFU for (m=128, k=4096, n=4096)
```

#### Compatibility Considerations

**1. Backward Compatibility:**
- V1 roofline (`rooflineStepTime`) still exists for non-MFU mode
- Simulator checks `if sim.MFUDatabase != nil` to decide which version to use
- No breaking changes to existing calibrated workflows

**2. Configuration Files:**
- `hardware_config.json` needs only TFlopsPeak and BwPeakTBs for V2
- Existing calibration fields (MfuPrefill, etc.) ignored when MFU database present
- Could maintain both for A/B testing

**3. Model Compatibility:**
- Requires models with standard transformer architecture
- Needs NumHeads, NumKVHeads, HiddenDim to compute attention config
- Fallback to nearest config if exact match unavailable

**4. Hardware Compatibility:**
- Currently only H100 benchmark data available
- A100 directory exists but needs data population
- New GPUs require benchmark data generation

#### Design Decisions from Implementation Plan

**1. Decode Aggregation:**
- **Rationale**: vLLM batches decode requests together in single kernel launch
- **Implication**: Single MFU lookup for aggregate batch, not per-request
- **Trade-off**: Loses per-request KV length variation, uses max KV length

**2. Prefill Bucketing:**
- **Rationale**: Cannot aggregate prefill with different seq_len (separate kernel launches)
- **Strategy**: Power-of-2 bucketing (512, 1024, 2048, ..., 65536)
- **Implication**: Approximates actual seq_len with bucket ceiling
- **Trade-off**: Over-estimates time for requests not exactly at bucket boundary

**3. TP Scaling Simplification:**
- **V1**: `effectiveTp = tp^0.68` (sublinear, calibrated)
- **V2**: `tpScaling = 1.0 / tp` (linear)
- **Rationale**: MFU data from TP-specific benchmarks already captures efficiency loss
- **Implication**: No need for empirical TP scaling exponents

**4. Attention Core Handling:**
- **V1**: Uses `vectorPeak = 0.1 * peakFlops` (artificial construct)
- **V2**: Uses same `peakFlops` but separate MFU lookup for attention
- **Hardware Factor**: `/1.8` for prefill attention (from InferSim)
- **Implication**: More realistic modeling of FlashAttention efficiency

**5. Memory Bandwidth:**
- **V1**: Calibrated factors (PrefillBwFactor, DecodeBwFactor)
- **V2**: Uses raw peak bandwidth
- **Rationale**: Roofline principle - memory time rarely dominates compute in modern LLMs
- **Potential Issue**: May under-estimate memory-bound scenarios

---

## Critical Analysis

### Strengths of the Plan

1. **Empirical Foundation**:
   - Uses real H100 benchmark data instead of theoretical estimates
   - Captures hardware-specific quirks (e.g., FlashAttention efficiency)
   - Eliminates need for extensive calibration per model

2. **Granular Modeling**:
   - Per-operation GEMM lookups capture shape-dependent efficiency
   - Separate attention core modeling with dedicated MFU
   - TP-specific benchmark data for accurate scaling

3. **Maintainability**:
   - Reduces calibration parameters from 15+ to 2 (TFlopsPeak, BwPeakTBs)
   - Clear separation: hardware specs vs benchmark data
   - Extensible to new GPUs by adding benchmark CSVs

4. **Implementation Quality**:
   - Comprehensive error handling (missing data, zero MFU)
   - Fallback strategies (nearest neighbor, TP=1 fallback)
   - Already partially implemented in codebase

### Potential Issues and Gaps

#### 1. Decode Aggregation Limitation

**Issue**: Uses max KV length across all decode requests
```go
maxKVLen := int64(0)
for _, req := range stepConfig.DecodeRequests {
    if req.ProgressIndex > maxKVLen {
        maxKVLen = req.ProgressIndex
    }
}
attnMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, int(maxKVLen), tp)
```

**Problem**:
- If batch has requests at KV lengths [1000, 2000, 3000, 32000], uses 32000 for all
- Over-estimates attention time for shorter context requests
- InferSim likely handles this differently (weighted average or bucketing?)

**Impact**: Decode step time over-estimation, especially with heterogeneous context lengths

**Potential Fix**: Bucket decode requests by KV length ranges, similar to prefill bucketing

#### 2. Prefill Bucketing Granularity

**Issue**: Power-of-2 bucketing approximates actual seq_len
```go
bucket := 512
for bucket < seqLen && bucket < 65536 {
    bucket *= 2
}
```

**Problem**:
- Request with seq_len=1500 uses bucket=2048 (37% over-estimation)
- Request with seq_len=2047 uses bucket=2048 (accurate)
- Non-uniform error distribution across bucket range

**Impact**: Variable accuracy depending on actual seq_len distribution

**Potential Fix**: Interpolation between bucket boundaries or finer bucketing

#### 3. Memory Bandwidth Simplification

**Issue**: V2 uses raw peak bandwidth without calibration factors
```go
decodeMemoryS = (dWeightBytes + dDynamicBytes) / peakBW
prefillMemoryS = (pWeightBytes + pDynamicBytes) / peakBW
```

**Problem**:
- Ignores memory access patterns (scattered vs sequential)
- KV cache access in decode is scattered (paged attention)
- V1 had `DecodeBwFactor = 0.6` to account for this

**Impact**: May under-estimate memory-bound decode scenarios

**Potential Fix**: Retain minimal bandwidth scaling factors or model memory access patterns

#### 4. MFU > 1.0 Semantics

**Observation**: Prefill and GEMM MFU values exceed 1.0
```csv
bf16,1024,45.85,1.266  # Prefill MFU
1024,2048,6144,25.552,1.019  # GEMM MFU
```

**Concern**:
- MFU typically means "fraction of peak FLOPs achieved"
- Values > 1.0 suggest different denominator (theoretical vs marketing peak?)
- Not documented in implementation plan

**Impact**: Unclear if formula `flops / (peakFlops * mfu)` is correct

**Potential Fix**: Verify MFU definition with InferSim codebase, adjust formula if needed

#### 5. Attention Core Hardware Factor

**Issue**: Hard-coded `/1.8` factor for prefill attention
```go
attnCoreTimeS := attnCoreFLOPs / 1.8 / (peakFlops * attnMFU) * tpScaling
```

**Concern**:
- Factor appears to be H100-specific
- No documentation on where 1.8 comes from
- Not applied to decode attention

**Impact**: May not generalize to A100 or other GPUs

**Potential Fix**: Make hardware-specific factor configurable in HardwareCalib

#### 6. Zero MFU Handling

**Observation**: Database has zero-MFU protection but still finds fallbacks
```go
if bestRow.MFU < 0.0001 {
    for i := range rows {
        if rows[i].MFU >= 0.0001 {
            return rows[i].MFU
        }
    }
    logrus.Fatalf("All MFU values are zero")
}
```

**Concern**:
- Decode data has `mfu=0.0` for (bs=1, kv=1024)
- Fallback uses first non-zero MFU, which may be very different shape
- Could lead to inaccurate predictions for edge cases

**Impact**: Unreliable for small batch sizes or short contexts

**Potential Fix**: Interpolate or extrapolate from neighboring data points

#### 7. TP Scaling Assumption

**Issue**: Assumes linear TP scaling with 1/tp
```go
tpScaling := 1.0 / tpFactor
totalTime += qTime * tpScaling
```

**Concern**:
- Communication overhead not fully captured in MFU data
- All-reduce still modeled separately but may not be sufficient
- TP=4 may not be exactly 4× faster due to interconnect bandwidth

**Impact**: May over-estimate TP scaling efficiency for TP > 2

**Potential Fix**: Add TP-specific efficiency multipliers or measure more TP configs

#### 8. Model Architecture Assumptions

**Issue**: Assumes standard transformer with SwiGLU
```go
// Hard-coded 7 GEMMs per layer
qTime, kTime, vTime, oTime, gateTime, upTime, downTime
```

**Concern**:
- Doesn't handle architectural variations (GLU vs SwiGLU, MoE, different MLP structures)
- Model-specific deviations not accounted for

**Impact**: Limited to Llama/Mistral-style architectures

**Potential Fix**: Make GEMM structure configurable per model family

---

## Questions for External Review

1. **Decode Aggregation**: Should decode requests be bucketed by KV length like prefill, or is max KV length aggregation acceptable?

2. **MFU Semantics**: Why are MFU values > 1.0? Should the formula be adjusted or is the benchmark data using a different peak FLOP definition?

3. **Memory Bandwidth**: Is it safe to remove bandwidth calibration factors? How significant is memory bottleneck in modern LLM inference?

4. **Hardware Factor**: Is the `/1.8` factor for prefill attention H100-specific? Should it be configurable?

5. **Bucketing Strategy**: Is power-of-2 bucketing optimal? Would linear interpolation between buckets improve accuracy?

6. **TP Scaling**: Is linear scaling (1/tp) sufficient, or should we retain some calibration factor for communication overhead?

7. **Zero MFU Handling**: What's the best fallback strategy for missing or zero MFU data points?

8. **Model Generalization**: How should we handle architectural variations beyond standard transformers?

---

## Implementation Status

Based on code analysis:

**Completed:**
- ✅ MFU database structure (`sim/mfu_database.go`) - fully implemented
- ✅ CSV loading functions - all three types (prefill, decode, GEMM)
- ✅ Lookup functions with nearest neighbor fallback
- ✅ Roofline V2 function (`sim/roofline_step_v2.go`) - fully implemented
- ✅ Helper functions (computeGEMMTime, computeTransformerGEMMTimes)
- ✅ Simulator integration - conditional V2 usage

**Partially Completed:**
- ⚠️ Hardware config simplification - struct has old fields but JSON only has new ones
- ⚠️ Database initialization - field exists in Simulator but CLI may not pass it

**Remaining Work:**
- ❌ CLI integration - ensure `mfuDB` is created and passed to NewSimulator
- ❌ End-to-end testing with real workloads
- ❌ Validation against vLLM ground truth
- ❌ Documentation and examples
- ❌ A100 benchmark data population

**Critical Path:**
1. Verify CLI creates MFU database at startup
2. Test with various models (Llama-2-7B, Llama-2-70B, Mixtral)
3. Compare predictions against vLLM measurements
4. Iterate on accuracy gaps (bucketing, TP scaling, etc.)

---

## Recommendations

### For Implementation Plan Review:

1. **Validate MFU Semantics**: Cross-reference with InferSim source code to confirm MFU > 1.0 handling

2. **Consider Decode Bucketing**: Evaluate whether max KV aggregation causes significant error vs bucketing approach

3. **Retain Minimal Bandwidth Factors**: Consider keeping decode bandwidth factor to model scattered access patterns

4. **Document Hardware-Specific Constants**: Make `/1.8` factor and similar constants configurable and documented

5. **Add Interpolation**: Consider linear interpolation for bucket boundaries instead of ceiling lookup

6. **Expand Test Coverage**: Test with diverse context length distributions to identify edge cases

7. **Benchmark Validation**: Measure actual vLLM performance and compare with simulator predictions

8. **Documentation**: Add design rationale document explaining all modeling choices and assumptions

---

## Appendix: Code References

### Key Files
- `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/sim/roofline_step.go` - Roofline V1 (calibrated)
- `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/sim/roofline_step_v2.go` - Roofline V2 (MFU-based)
- `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/sim/mfu_database.go` - MFU data structures and loading
- `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/sim/simulator.go` - Main simulator with V2 integration
- `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/hardware_config.json` - Simplified hardware specs
- `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/bench_data/` - H100 benchmark data

### Benchmark Data Sample Sizes
- **H100 Prefill**: 6 attention configs × ~5 seq_len points = ~30 rows
- **H100 Decode**: 6 attention configs × 3 TP levels × ~20 (bs, kv) points = ~360 rows
- **H100 GEMM**: ~9 m values × ~10 (k, n) pairs = ~90 rows

### Model Configs Tested
- Llama-2-7B: 32 heads, 32 KV heads, 128 head_dim → `32-32-128`
- Llama-2-70B: 64 heads, 8 KV heads (GQA), 128 head_dim → `64-8-128`
- Mixtral-8x7B: 32 heads, 8 KV heads (GQA), 128 head_dim → `32-8-128`

---

# Critique 1: Implementation Plan Analysis

## Overview

This critique examines the MFU-based roofline model implementation plan against the actual codebase state. The most striking finding is a **fundamental plan-reality mismatch**: the implementation described as "planned work" has been **nearly completed** already. The code analysis reveals that `sim/roofline_step_v2.go` (339 lines) and `sim/mfu_database.go` (574 lines) are fully functional, with CLI integration in `cmd/root.go` already operational.

The implementation quality is excellent - comprehensive error handling, fallback strategies, and well-structured code. However, this critique identifies several design decisions that warrant scrutiny, particularly around decode aggregation, MFU semantics (values > 1.0), memory bandwidth simplification, and validation gaps.

## Critical Findings

### 1. **Implementation Status Mismatch** (CRITICAL)

**Finding**: The "Implementation Plan Summary" describes 12 tasks as future work, but code inspection reveals:

- ✅ **COMPLETE**: `sim/mfu_database.go` - All data structures, CSV loaders, lookup functions fully implemented (574 lines)
- ✅ **COMPLETE**: `sim/roofline_step_v2.go` - Entire V2 roofline with GEMM helpers, attention core calculation, bucketing logic (339 lines)
- ✅ **COMPLETE**: CLI integration in `cmd/root.go` (lines 154-162) - MFU database initialization conditional on roofline mode
- ✅ **COMPLETE**: Simulator integration in `sim/simulator.go` (lines 247-254) - Conditional V2 usage when MFUDatabase != nil
- ⚠️ **PARTIAL**: Hardware config simplification - JSON only contains TFlopsPeak/BwPeakTBs, but struct still has deprecated fields

**Impact**: The "plan" is actually a **post-hoc description** of existing code. Treating this as future work misrepresents project status and obscures the real task: **validation and testing** of an already-implemented system.

**Actual Remaining Work**:
- End-to-end validation against vLLM ground truth
- Performance benchmarking across diverse workloads
- Edge case testing (zero MFU, extreme batch sizes, TP > 4)
- A100 benchmark data collection
- Documentation of design rationale

### 2. **MFU Semantics Issue** (HIGH SEVERITY)

**Finding**: Benchmark data contains MFU values > 1.0, contradicting standard MFU definition (fraction of peak, 0.0-1.0 range):

```csv
# Prefill MHA (mha/prefill/h100/32-32-128.csv)
bf16,1024,45.85,1.266    # MFU = 126.6% of peak?
bf16,4096,219.951,4.222  # MFU = 422.2% of peak?

# GEMM (gemm/h100/data.csv)
1024,2048,6144,25.552,1.019  # MFU = 101.9% of peak
4096,2048,6144,90.913,1.146  # MFU = 114.6% of peak
```

**Code using MFU values**:
```go
// roofline_step_v2.go:19
return flops / (peakFlops * mfu)  // If mfu > 1.0, this reduces time estimate
```

**Problem Analysis**:
- If MFU > 1.0 means "throughput exceeds marketing peak", formula is correct
- If MFU is a scaling factor (not fraction), formula needs adjustment
- InferSim technical report is silent on MFU > 1.0 semantics
- No validation that predictions match vLLM measurements

**Potential Causes**:
1. **Peak FLOPs definition mismatch**: Benchmark uses theoretical peak (e.g., 2000 TFLOPs), `hardware_config.json` uses achievable peak (989.5 TFLOPs)
2. **Denominator inconsistency**: Benchmark measures throughput vs. FP32 peak, config uses BF16 peak
3. **Scaling factor interpretation**: MFU might be `(measured_time / theoretical_time)^-1` rather than fraction

**Impact**: Unclear if predictions are accurate. Could lead to systematic under-estimation of latency if MFU > 1.0 is misinterpreted.

**Recommendation**:
- Cross-reference InferSim source code for MFU calculation methodology
- Validate end-to-end predictions against vLLM for known workloads
- Document MFU semantics explicitly in code comments
- Consider normalizing MFU to 0-1 range if semantics are confirmed

### 3. **Decode Aggregation Design Decision** (MEDIUM SEVERITY)

**Finding**: Decode phase uses **max KV length** across all requests (lines 145-152 in roofline_step_v2.go):

```go
maxKVLen := int64(0)
for _, req := range stepConfig.DecodeRequests {
    if req.ProgressIndex > maxKVLen {
        maxKVLen = req.ProgressIndex
    }
}
attnMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, int(maxKVLen), tp)
```

**Problem**: If batch contains requests at KV lengths [1024, 2048, 4096, 32768], attention MFU lookup uses 32768 for all requests.

**Theoretical Impact Analysis**:
- **Scenario**: 8 requests at KV=2048, 1 request at KV=32768
- **Current**: Uses MFU for (bs=9, kv=32768) - pessimistic for 8 requests
- **Alternative**: Bucket by KV ranges, weight by request count
- **vLLM Reality**: Decode kernel processes variable-length batch, but FlashAttention-2 efficiency depends on max sequence length in batch

**Why This Might Be Correct**:
- PagedAttention in vLLM pads to max sequence length in batch for efficient kernel execution
- Hardware occupancy determined by longest sequence (even if most are short)
- Simpler implementation matches vLLM's actual batching behavior

**Validation Needed**: Compare predictions against vLLM with heterogeneous context lengths to determine if bucketing is necessary.

### 4. ~~**Memory Bandwidth Over-Simplification**~~ (REMOVED - See Review Update)

### 5. **Prefill Bucketing Granularity** (LOW-MEDIUM SEVERITY)

**Finding**: Power-of-2 bucketing introduces variable approximation error:

```go
bucket := 512
for bucket < seqLen && bucket < 65536 {
    bucket *= 2  // 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
}
```

**Error Distribution**:
- seq_len=513 → bucket=1024 (99% over-estimate)
- seq_len=1000 → bucket=1024 (2% over-estimate)
- seq_len=1500 → bucket=2048 (37% over-estimate)
- seq_len=2047 → bucket=2048 (0% over-estimate)

**Average Error**: ~25-30% over-estimation when seq_len is not near bucket boundary.

**Mitigation**: Prefill MFU increases sub-linearly with seq_len (1.266 @ 1024 → 4.222 @ 4096), so over-estimating seq_len may partially cancel with lower MFU.

**Why Not Linear Interpolation?**: Benchmark data is sparse (only 5-6 seq_len points per config). Interpolation between 1024 and 4096 may not capture non-linear efficiency curves.

**Validation Needed**: Test with seq_len distribution centered between buckets (e.g., 1500-1800) to quantify impact.

### 6. **Zero MFU Handling Fragility** (LOW SEVERITY)

**Finding**: Database has zero-MFU protection but fallback strategy is simplistic:

```go
// mfu_database.go:405-414
if bestRow.MFU < 0.0001 {
    for i := range rows {
        if rows[i].MFU >= 0.0001 {
            return rows[i].MFU  // Returns FIRST non-zero, not nearest
        }
    }
    logrus.Fatalf("All MFU values are zero")
}
```

**Data Evidence**: Decode CSV has zero MFU for (bs=1, kv=1024):
```csv
bf16,bf16,1,1024,36.041,0.0  # Zero MFU
```

**Problem**: Fallback returns first non-zero MFU in array, which may be from vastly different shape (e.g., bs=256, kv=65536).

**Impact**: Rare, but could cause wildly inaccurate predictions for edge cases (single-request, short-context decode).

**Recommendation**: Find nearest non-zero MFU by Euclidean distance, not first in array.

### 7. **TP Scaling Assumption** (LOW SEVERITY)

**Finding**: V2 uses linear TP scaling (1/tp), justified by "MFU data captures efficiency":

```go
tpScaling := 1.0 / tpFactor
totalTime += qTime * tpScaling  // Linear scaling
```

**V1 Used Non-Linear Scaling**:
```go
effectiveTpPrefill := math.Pow(tpFactor, hwConfig.TpScalingExponent)  // tp^0.68
effectiveTpDecode := math.Pow(tpFactor, hwConfig.DecodeTpScalingExponent)  // tp^1.0
```

**Justification**: Decode benchmark data has TP-specific files (`-tp1`, `-tp2`, `-tp4`), capturing real efficiency loss.

**Gap**: Prefill benchmark data has **no TP-specific files**. If prefill at TP=4 achieves only 75% efficiency (not 25% as linear scaling suggests), predictions will be optimistic.

**Evidence Missing**: No validation that prefill TP scaling is linear. InferSim report may contain this data.

**Recommendation**:
- Check if InferSim has TP-specific prefill benchmarks
- Test predictions at TP=2 and TP=4 against vLLM
- Consider adding TP-specific scaling factor to HardwareCalib if needed

### 8. **Hard-Coded Hardware Factor** (LOW SEVERITY)

**Finding**: Prefill attention has hard-coded `/1.8` factor:

```go
attnCoreTimeS := attnCoreFLOPs / 1.8 / (peakFlops * attnMFU) * tpScaling
```

**Concern**:
- Factor appears H100-specific (no documentation)
- Not applied to decode attention or GEMM operations
- Not configurable per GPU type

**Source**: Likely from InferSim codebase, but not referenced in plan.

**Impact**: May not generalize to A100 or other architectures.

**Recommendation**: Move to HardwareCalib struct as `PrefillAttnScalingFactor` with default 1.8 for H100.

## Design Review

### Architectural Strengths

1. **Empirical Foundation**: Using real benchmark data eliminates calibration overhead
2. **Modular Design**: Clean separation of concerns (MFU database, roofline calculation, simulator integration)
3. **Fallback Strategies**: Robust handling of missing data (nearest config, TP fallback, zero-MFU protection)
4. **Backward Compatibility**: V1 coexists with V2, enabling A/B testing
5. **Code Quality**: Comprehensive error handling, informative logging, well-documented functions

### Architectural Weaknesses

1. **Benchmark Data Sparsity**: Limited coverage (6 attention configs, 5-6 seq_len points, only H100)
2. **Implicit Assumptions**: MFU semantics, TP scaling linearity, memory bandwidth not validated
3. **Hard-Coded Constants**: Hardware-specific factors embedded in code rather than configuration
4. **Missing Observability**: No metrics on lookup accuracy (floor vs ceiling, Euclidean distance magnitude)

### Design Decision Assessment

#### ✅ **Good Decisions**:
- **Per-operation GEMM lookups**: Captures shape-dependent efficiency correctly
- **Separate attention core modeling**: More accurate than lumping into GEMM FLOPs
- **Power-of-2 bucketing**: Practical given sparse benchmark data
- **Nearest neighbor fallback**: Handles model configs not in benchmark data gracefully

#### ⚠️ **Questionable Decisions**:
- **Decode max KV aggregation**: May over-estimate for heterogeneous batches (needs validation)
- **No bandwidth calibration**: Ignores memory access patterns; may under-estimate decode latency
- **Linear TP scaling for prefill**: No TP-specific prefill benchmarks to validate this assumption

#### ❌ **Problematic Decisions**:
- **MFU > 1.0 not explained**: Fundamental ambiguity in semantics could invalidate predictions
- **Hard-coded /1.8 factor**: Not portable across hardware types
- **No validation infrastructure**: Cannot verify accuracy of design decisions

## Implementation Status Mismatch

### What the Plan Says:
> "Implementation Tasks (12 Total)"
> 1. Create MFU data structures
> 2. Implement attention config computation
> 3. Implement CSV loading functions
> ...
> 12. Final verification and cleanup

### What Actually Exists:

**Complete Implementation** (`sim/mfu_database.go`):
- ✅ Lines 14-42: Data structures (MHAPrefillRow, MHADecodeRow, GEMMRow, AttentionShape, MFUDatabase)
- ✅ Lines 54-73: Attention config computation (computeAttentionConfig, parseAttentionConfig)
- ✅ Lines 84-291: CSV loading (loadPrefillCSV, loadAllPrefillCSVs, loadDecodeCSV, loadAllDecodeCSVs, loadGEMMCSV)
- ✅ Lines 314-360: Database initialization (NewMFUDatabase with nearest config matching)
- ✅ Lines 362-498: Lookup functions with nearest neighbor fallback
- ✅ Lines 500-573: GEMM two-stage lookup

**Complete Implementation** (`sim/roofline_step_v2.go`):
- ✅ Lines 11-20: computeGEMMTime helper
- ✅ Lines 25-86: computeTransformerGEMMTimes (7 GEMMs per layer)
- ✅ Lines 92-116: calculateAttentionCoreFLOPs
- ✅ Lines 120-338: rooflineStepTimeV2 with decode aggregation, prefill bucketing, mixed batch heuristics

**Complete Integration** (`cmd/root.go`):
- ✅ Lines 131-162: Roofline mode detection, MFU database initialization with error handling

**Complete Integration** (`sim/simulator.go`):
- ✅ Lines 116-146: NewSimulator accepts mfuDB parameter
- ✅ Lines 247-254: Conditional V2 usage when MFUDatabase != nil

### Reality Check:

**The "Implementation Plan" is a misnomer.** It's actually a **technical specification of already-completed work**. The codebase has:
- 913 lines of production code (mfu_database.go: 574, roofline_step_v2.go: 339)
- Full CLI integration with error handling
- Simulator integration with backward compatibility
- Comprehensive logging and fallback strategies

**What's Actually Missing**:
1. **Validation**: No tests comparing predictions to vLLM measurements
2. **Benchmarking**: No performance evaluation across workload types
3. **Documentation**: Design rationale not captured in code or separate docs
4. **A100 Data**: Only H100 benchmarks populated
5. **Edge Case Testing**: Zero MFU, extreme batch sizes, TP > 4 not validated

## Recommendations

### Immediate Actions (Critical Path)

1. **Validate MFU Semantics** (HIGH PRIORITY)
   - Cross-reference InferSim source code at https://github.com/inference-sim/InferSim
   - Document MFU > 1.0 interpretation with concrete formula
   - Add validation tests: run simulator + vLLM on identical workload, compare latencies

2. **Validation Framework Available** (OPTIONAL)

   **Tool:** `python_scripts/blis_evaluator.py` is available for validation if needed.

   **Usage:** Run `./python_scripts/blis_evaluator.py --ground-truth eval/combined_ground_truth.json`

   This can be used to validate accuracy against real vLLM deployments when desired.

3. **Memory Bandwidth Validation** (MEDIUM PRIORITY)
   - Test decode-heavy workload: bs=1-4, kv=16K-32K, model=Llama-2-70B
   - Compare simulator predictions to vLLM measurements
   - If error > 20%, reintroduce DecodeBwFactor calibration parameter

### Correctness Improvements

4. **Fix Zero-MFU Fallback** (MEDIUM PRIORITY)
   ```go
   // Instead of returning first non-zero MFU
   if bestRow.MFU < 0.0001 {
       // Find nearest non-zero by distance, not array order
       minDist := math.MaxFloat64
       var fallbackRow *MHAPrefillRow
       for i := range rows {
           if rows[i].MFU >= 0.0001 {
               dist := math.Abs(float64(rows[i].SeqLen - seqLen))
               if dist < minDist {
                   minDist = dist
                   fallbackRow = &rows[i]
               }
           }
       }
       return fallbackRow.MFU
   }
   ```

5. **Make Hardware Factors Configurable** (LOW PRIORITY)
   ```go
   // hardware_config.json
   {
       "H100": {
           "TFlopsPeak": 989.5,
           "BwPeakTBs": 3.35,
           "PrefillAttnScalingFactor": 1.8,
           "DecodeBwEfficiency": 0.6,
           "PrefillBwEfficiency": 0.85
       }
   }
   ```

6. **Improve Decode Aggregation** (OPTIONAL - pending validation)
   - If heterogeneous context length causes >20% error:
     - Bucket decode requests by KV ranges (e.g., 0-2K, 2K-8K, 8K-32K, 32K+)
     - Lookup MFU per bucket, weight by request count
     - More accurate but complex; only implement if validation shows necessity

### Documentation and Observability

7. **Add Design Rationale Document** (`docs/roofline_v2_design.md`)
   - MFU semantics and formula derivation
   - Decode aggregation rationale (vLLM kernel behavior)
   - Prefill bucketing trade-offs
   - TP scaling assumptions and validation
   - Memory bandwidth simplification justification

8. **Enhance Logging for Analysis**
   ```go
   // Log lookup accuracy metrics
   logrus.Debugf("GEMM lookup: target (m=%d, k=%d, n=%d) → actual (m=%d, k=%d, n=%d), MFU=%.4f",
       targetM, targetK, targetN, actualM, actualK, actualN, mfu)

   // Log roofline bottleneck
   logrus.Infof("Step %d: compute=%.2fms, memory=%.2fms, bottleneck=%s",
       stepID, computeS*1000, memoryS*1000, bottleneckType)
   ```

9. **Add Validation Metrics to Output**
   - Per-step breakdown: GEMM time, attention time, memory time, overhead
   - Lookup statistics: nearest neighbor distance, fallback frequency
   - TP scaling efficiency: compare predicted vs measured speedup

### Testing and Data Expansion

10. **Expand Test Coverage**
    - Unit tests for lookup functions with edge cases
    - Integration tests with synthetic workloads
    - Regression tests comparing V1 vs V2 predictions
    - Benchmark tests measuring lookup performance

11. **A100 Data Collection**
    - Run InferSim benchmarking scripts on A100 hardware
    - Populate `bench_data/mha/{prefill,decode}/a100/` and `bench_data/gemm/a100/`
    - Validate predictions against vLLM on A100

12. **Expand Benchmark Coverage**
    - More attention configs: 40-40-128 (Llama-3), 56-8-128 (Qwen), 28-4-128 (Mistral-Nemo)
    - Finer seq_len granularity: 512, 768, 1024, 1536, 2048, 3072, 4096, ...
    - Higher TP levels: tp8 for 70B+ models

### Process Improvements

13. **Update Project Status**
    - Rename "Implementation Plan" to "Technical Specification"
    - Create new document: "Validation and Testing Plan"
    - Update README with current status: implementation complete, validation in progress

14. **Establish Validation Workflow**
    ```bash
    # Standard validation protocol
    1. Generate vLLM trace: python scripts/benchmark_vllm.py --model llama-2-7b --workload chatbot
    2. Run simulator: inference-sim run --model llama-2-7b --workload chatbot --roofline
    3. Compare results: python scripts/compare_predictions.py vllm_trace.json sim_results.json
    4. Analyze errors: python scripts/analyze_errors.py --breakdown-by phase,batch-size,context-length
    ```

## Questions for External Review

The following questions remain unresolved and should be addressed by Claude Opus review or InferSim maintainers:

1. **MFU Semantics**: Why are MFU values > 1.0 in benchmark data? What is the correct interpretation and formula?

2. **Decode Aggregation Accuracy**: Is using max KV length for heterogeneous decode batches accurate? Should we bucket by KV ranges?

3. **Memory Bandwidth Necessity**: Is removing bandwidth calibration factors safe? How significant is memory bottleneck in decode at small batch sizes?

4. **TP Scaling Validation**: Is linear TP scaling (1/tp) for prefill accurate? Do we need TP-specific prefill benchmarks?

5. **Hardware Factor Portability**: Is the `/1.8` prefill attention factor H100-specific? What should A100 use?

6. **Bucketing Optimality**: Would linear interpolation between seq_len buckets improve accuracy vs power-of-2 bucketing?

7. **Validation Threshold**: What prediction error is acceptable? InferSim paper reports 4-15% - is this per-step or aggregate?

8. **Benchmark Data Quality**: Are there known issues with zero MFU values? Should we filter/interpolate during data loading?

---

## External Review by Claude Opus 4.6

### Review Metadata
- **Model**: aws/claude-opus-4-6
- **Date**: 2026-02-20
- **Review Type**: Independent external critique
- **API Endpoint**: IBM Research Internal LiteLLM Gateway

---

# Implementation Plan Review

## 1. OVERALL ASSESSMENT

**Rating: Strong** (with caveats)

This is an exceptionally thorough technical document — arguably *too* thorough for what it claims to be. The plan demonstrates deep understanding of both the problem domain (LLM inference simulation) and the existing codebase. The critical analysis section is genuinely excellent, identifying real issues with concrete code references and data evidence.

**However, the document has a fundamental identity crisis.** It presents itself as an "Implementation Plan" but is actually a **post-implementation technical audit** of already-completed work. This mismatch is the single most important finding and it distorts everything downstream: task prioritization, resource allocation, and risk assessment are all oriented around building something that already exists, rather than validating and hardening it.

**Core requirements coverage:**
- ✅ Replacing fixed MFU with benchmark-driven lookups — implemented
- ✅ Per-operation GEMM granularity — implemented
- ✅ Attention-specific modeling — implemented
- ✅ Backward compatibility — implemented
- ❌ Validation against ground truth — **completely absent**
- ❌ Correctness verification of MFU semantics — unresolved
- ❌ Production readiness (testing, documentation, multi-GPU support) — not started

The plan addresses the *construction* requirements comprehensively but almost entirely neglects the *verification* requirements, which are now the actual critical path.

---

## 2. POTENTIAL ISSUES

### 2.1 Showstopper: MFU > 1.0 Semantic Ambiguity

This is the single most dangerous issue in the entire plan, and it's treated as a discussion point rather than a blocking concern.

**The problem is concrete.** The core formula is:

```go
time = flops / (peakFlops * mfu)
```

If `peakFlops = 989.5 TFLOPs` (from `hardware_config.json`) and `mfu = 4.222` (from prefill CSV at seq_len=4096), then:

```
time = flops / (989.5e12 * 4.222) = flops / 4.176e13
```

This implies the hardware achieves **4.176 PFLOPs** — over 4× the stated peak. Either:

1. **The peak in `hardware_config.json` is wrong** (should be ~200 TFLOPs to make MFU=4.2 sensible as a ratio against a lower base), or
2. **MFU is not "Model FLOPs Utilization" in the traditional sense** — it's a different scaling factor, possibly `achieved_throughput / some_reference_throughput`, or
3. **The formula should not multiply peakFlops × mfu** — perhaps MFU is already an absolute throughput measure

**Without resolving this, every prediction the simulator produces is untrustworthy.** The plan acknowledges this ("Potential Fix: Verify MFU definition with InferSim codebase") but doesn't flag it as a prerequisite for any other work.

**Concrete action:** Before any validation or testing, someone must:
```python
# In InferSim source, find the MFU calculation
# Expected location: InferSim/benchmark/compute_mfu.py or similar
# Verify: mfu = measured_flops / (peak_flops * time)
# Check: what is "peak_flops" in their calculation?
# Compare: their peak_flops vs our hardware_config.json TFlopsPeak
```

### 2.2 Decode Aggregation: Correct Intuition, Wrong Justification

The plan states decode uses max KV length because "vLLM batches decode requests together in single kernel launch." This is partially correct but mischaracterizes how PagedAttention works.

**What actually happens in vLLM:**
- PagedAttention does **not** pad to max sequence length. That's the entire point of paged attention — it processes variable-length sequences without padding.
- The kernel iterates over each sequence's page table independently.
- Execution time scales roughly with `sum(kv_lengths)`, not `batch_size × max(kv_lengths)`.

**What InferSim likely does:** Uses `(batch_size, max_kv_len)` as a proxy because their benchmark grid is 2D. This is a modeling approximation, not a reflection of kernel behavior.

**The real question:** How much error does this approximation introduce? Consider:

| Scenario | Actual Cost (proportional) | Modeled Cost (proportional) | Error |
|----------|---------------------------|----------------------------|-------|
| 8 reqs @ kv=4096 | 8 × 4096 = 32,768 | lookup(8, 4096) | Low |
| 7 reqs @ kv=1024, 1 req @ kv=32768 | 7×1024 + 32768 = 39,936 | lookup(8, 32768) | **High** |
| 4 reqs @ kv=16384, 4 reqs @ kv=32768 | 4×16384 + 4×32768 = 196,608 | lookup(8, 32768) | Moderate |

The error is worst when there's high variance in KV lengths within a batch — which is common in production serving with continuous batching.

### 2.3 Memory Bandwidth: The Silent Regression

The plan removes bandwidth calibration factors, arguing "memory time rarely dominates compute." This assumption is **incorrect for decode at small batch sizes**, which is precisely the most latency-sensitive scenario (single-user interactive inference).

**Concrete example with H100:**
```
Model: Llama-2-70B, TP=4
Decode batch_size=1, kv_len=8192

Weight access per step: ~17.5 GB (70B params / 4 TP, BF16)
KV cache access: ~0.5 GB (2 × 80 layers × 8 KV heads × 128 dim × 8192 × 2 bytes / 4 TP)
Total memory: ~18 GB

Memory time (peak BW=3.35 TB/s): 18 / 3350 = 5.37 ms
Memory time (effective BW=0.6×3.35=2.01 TB/s): 18 / 2010 = 8.96 ms

GEMM compute (MFU ~0.05 at m=1):
  FLOPs ≈ 2 × 1 × 4096 × 4096 × 7 ops × 80 layers / 4 TP ≈ 37.6 GFLOPs
  Time = 37.6e9 / (989.5e12 × 0.05) = 0.76 ms

Roofline: max(compute, memory) → memory-bound by 7-12×
```

Removing the bandwidth factor changes the prediction from 8.96ms to 5.37ms — a **40% under-estimation** in a memory-bound regime. This is not a minor calibration issue; it's a regime change.

### 2.4 Assumptions That May Be Incorrect

| Assumption | Evidence For | Evidence Against | Risk |
|-----------|-------------|-----------------|------|
| Linear TP scaling for GEMM | MFU data has TP-specific decode files | No TP-specific prefill files; all-reduce overhead not in MFU | Medium |
| Power-of-2 bucketing is sufficient | Matches InferSim approach | Up to 99% over-estimation at bucket boundaries | Medium |
| `/1.8` factor is universal for H100 | Appears in InferSim code | No documentation; may vary with FlashAttention version | Low-Medium |
| SwiGLU architecture (7 GEMMs/layer) | Matches Llama/Mistral | Doesn't handle MoE (Mixtral), different MLP widths | Low |
| Nearest-neighbor fallback is adequate | Reasonable for similar configs | Euclidean distance on (heads, kv_heads, head_dim) is arbitrary | Low |

### 2.5 Missing Edge Cases

1. **Chunked prefill**: The simulator has `LongPrefillTokenThreshold` but roofline_step_v2.go doesn't appear to handle this. If a 8192-token prefill is chunked into 2×4096, does it use bucket=8192 or bucket=4096? This could be a significant error source.

2. **Mixed precision**: All CSV data is BF16. What happens if the model uses FP16 or FP8 (increasingly common for inference)? No fallback or scaling factor is mentioned.

3. **Batch size = 0**: What if DecodeRequests is empty? The code would compute maxKVLen=0 and totalBatchSize=0. Does GetAttnDecodeMFU(0, 0, tp) handle this gracefully?

4. **Very large batch sizes**: Benchmark data likely maxes out at bs=256 or bs=512. What happens when continuous batching creates bs=1024 in decode? Nearest neighbor will extrapolate, but is that accurate?

5. **TP > 4**: No benchmark data for tp=8. For 70B+ models, this is a common configuration. Fallback to tp=1 data is a poor proxy.

---

## 3. ALTERNATIVE APPROACHES

### 3.1 Decode Aggregation Alternative

Instead of `max(kv_len)`, use a **weighted average** that better reflects PagedAttention's actual behavior:

```go
// Current approach
maxKVLen := int64(0)
for _, req := range stepConfig.DecodeRequests {
    if req.ProgressIndex > maxKVLen {
        maxKVLen = req.ProgressIndex
    }
}
attnMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, int(maxKVLen), tp)

// Alternative: KV-weighted bucketing
kvBuckets := make(map[int]int) // bucket_kv -> count
for _, req := range stepConfig.DecodeRequests {
    bucket := roundToNearestBucket(req.ProgressIndex)  // e.g., 1024, 2048, 4096, ...
    kvBuckets[bucket]++
}

totalAttnTime := 0.0
for bucket, count := range kvBuckets {
    attnMFU := mfuDB.GetAttnDecodeMFU(count, bucket, tp)
    attnCoreFLOPs := calculateAttentionCoreFLOPs(..., count, bucket)
    totalAttnTime += attnCoreFLOPs / (peakFlops * attnMFU)
}
```

This approach:
- Respects the heterogeneous nature of decode batches
- Mirrors how vLLM's PagedAttention actually processes sequences
- Requires minimal changes (bucketing function + loop)
- Uses existing MFU data (just changes the lookup keys)

**Trade-off:** Slightly more complex, and if the batch is homogeneous (all requests at similar KV lengths), it's equivalent to the current approach anyway.

### 3.2 Memory Bandwidth Hybrid Model

Instead of choosing between V1 (calibrated factors) and V2 (raw peak), use a **dynamic efficiency model** based on access patterns:

```go
// Sequential weight access (high efficiency)
weightBytes := baseMem["model_weights"] * tpScaling
weightTime := weightBytes / (peakBW * 0.9)  // 90% efficiency

// Scattered KV cache access (low efficiency, batch-size-dependent)
kvBytes := calculateKVAccessBytes(...)
kvEfficiency := 0.4 + 0.3 * min(batchSize / 64.0, 1.0)  // 40-70% efficient
kvTime := kvBytes / (peakBW * kvEfficiency)

memoryTime := weightTime + kvTime
```

This model:
- Acknowledges that different memory accesses have different characteristics
- Scales KV access efficiency with batch size (better coalescing at higher BS)
- Retains predictability without needing per-GPU calibration
- Can be validated against memory profiling tools (Nsight Compute)

### 3.3 MFU Interpolation for Prefill

Instead of power-of-2 bucketing with ceiling, use **linear interpolation** when seq_len falls between benchmark points:

```go
// Current: seq_len=1500 → bucket=2048, MFU=mfu[2048]
// Proposed: seq_len=1500 → interpolate between mfu[1024] and mfu[2048]

func (db *MFUDatabase) GetAttnPrefillMFU(seqLen int) float64 {
    rows := db.prefillData[db.attentionConfig]

    // Find floor and ceiling
    var floor, ceil *MHAPrefillRow
    for i := range rows {
        if rows[i].SeqLen <= seqLen {
            if floor == nil || rows[i].SeqLen > floor.SeqLen {
                floor = &rows[i]
            }
        }
        if rows[i].SeqLen >= seqLen {
            if ceil == nil || rows[i].SeqLen < ceil.SeqLen {
                ceil = &rows[i]
            }
        }
    }

    // Exact match or boundary cases
    if floor != nil && floor.SeqLen == seqLen { return floor.MFU }
    if ceil != nil && ceil.SeqLen == seqLen { return ceil.MFU }
    if floor == nil { return ceil.MFU }
    if ceil == nil { return floor.MFU }

    // Linear interpolation
    t := float64(seqLen - floor.SeqLen) / float64(ceil.SeqLen - floor.SeqLen)
    return floor.MFU + t * (ceil.MFU - floor.MFU)
}
```

**Caveat:** MFU may not be linear in seq_len. But given the sparse data (5-6 points), any interpolation is better than ceiling-only bucketing. Could also use log-linear interpolation if MFU scales with log(seq_len).

---

## 4. RISKS

### 4.1 Critical Risks (Blocking)

| Risk | Impact | Likelihood | Mitigation Status |
|------|--------|-----------|-------------------|
| **MFU semantics ambiguity** | Predictions systematically wrong by 2-4× | High | ❌ Not addressed |
| **Memory bandwidth under-estimation** | 40%+ error in decode-heavy workloads | Medium | ❌ Not addressed |

### 4.2 High Risks (Should Address Before Production)

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Decode aggregation with heterogeneous batches** | 20-50% error in mixed workloads | Test with real trace data; implement bucketing if needed |
| **Sparse benchmark data** | Poor generalization to unseen configs | Expand dataset; add more attention configs and finer granularity |
| **A100 data missing** | V2 unusable for A100 deployments | Run InferSim benchmarks on A100 hardware |
| **TP > 4 extrapolation** | Inaccurate for 70B+ model configs | Benchmark tp=8; add validation for TP scaling linearity |

### 4.3 Medium Risks (Accept or Monitor)

| Risk | Mitigation Complexity |
|------|----------------------|
| **Prefill bucketing 99% over-estimation** | Medium (add interpolation) |
| **Zero MFU fallback inadequacy** | Low (fix distance metric) |
| **Hard-coded /1.8 factor** | Low (move to config) |
| **Chunked prefill not handled** | Medium (needs roofline V2 changes) |

### 4.4 Low Risks (Accept)

- Euclidean distance for attention config matching (works well enough in practice)
- SwiGLU architecture assumption (most modern LLMs use this)
- TP scaling overhead not in MFU (small effect, likely <5%)

---

## 5. MISSING ELEMENTS

### 5.1 Note: Validation Framework Available

**UPDATE:** A comprehensive validation framework exists at `python_scripts/blis_evaluator.py` that provides automated BLIS evaluation when needed. The tool includes:

**1. Ground Truth Collection:**
```bash
# Benchmark vLLM with known workloads
python scripts/benchmark_vllm.py \
    --model llama-2-7b \
    --gpu h100 \
    --tp 1 \
    --workload chatbot \
    --num-requests 100 \
    --output vllm_trace_llama2_7b_h100_tp1_chatbot.json

# Repeat for:
# - Models: llama-2-7b, llama-2-13b, llama-2-70b, mixtral-8x7b
# - Workloads: chatbot, summarization, long-context
# - Configs: (h100, tp1), (h100, tp2), (h100, tp4), (a100, tp1)
```

**2. Simulator Execution:**
```bash
inference-sim run \
    --model llama-2-7b \
    --hardware h100 \
    --tp 1 \
    --workload chatbot \
    --model-config-folder model_configs/llama-2-7b \
    --hardware-config hardware_config.json \
    --bench-data-path bench_data \
    --roofline \
    --output sim_results_llama2_7b_h100_tp1_chatbot.json
```

**3. Comparison and Analysis:**
```python
# scripts/validate_predictions.py
def validate(vllm_trace, sim_results):
    errors = []
    for i, (vllm_step, sim_step) in enumerate(zip(vllm_trace, sim_results)):
        error_pct = abs(sim_step.latency - vllm_step.latency) / vllm_step.latency * 100
        errors.append(error_pct)

        if error_pct > 20:  # Flag outliers
            print(f"Step {i}: error={error_pct:.1f}%, "
                  f"vLLM={vllm_step.latency:.2f}ms, sim={sim_step.latency:.2f}ms")
            print(f"  Batch: {vllm_step.batch_size} reqs, "
                  f"{vllm_step.num_prefill} prefill, {vllm_step.num_decode} decode")

    print(f"Mean error: {np.mean(errors):.2f}%")
    print(f"P50 error: {np.percentile(errors, 50):.2f}%")
    print(f"P95 error: {np.percentile(errors, 95):.2f}%")
    print(f"P99 error: {np.percentile(errors, 99):.2f}%")

    # InferSim paper reports 4-15% error → target P95 < 15%
    assert np.percentile(errors, 95) < 15, "P95 error exceeds InferSim baseline"
```

**4. Error Analysis:**
```python
# Break down error by scenario
df = pd.DataFrame({
    'error': errors,
    'phase': [step.phase for step in sim_results],  # prefill, decode, mixed
    'batch_size': [step.batch_size for step in sim_results],
    'prefill_tokens': [step.num_prefill_tokens for step in sim_results],
})

print("Error by phase:")
print(df.groupby('phase')['error'].describe())

print("\nError by batch size:")
df['bs_bucket'] = pd.cut(df['batch_size'], bins=[0, 4, 16, 64, 256])
print(df.groupby('bs_bucket')['error'].describe())
```

The existing `python_scripts/blis_evaluator.py` tool provides:
- ✅ Automated BLIS execution for experiments
- ✅ Comparison against ground truth (TTFT, ITL, E2E latency)
- ✅ Percentage error calculation and aggregation
- ✅ Results breakdown by workload type and model

This tool can be used for validation when comparing against real vLLM deployments is needed.

### 5.2 Missing: Performance/Scalability Analysis

The plan doesn't discuss lookup performance. With thousands of requests per second:

```go
// Every step calls these lookups:
// - 7 GEMM lookups × num_layers per request in prefill
// - 7 GEMM lookups × num_layers for aggregate decode batch
// - 1 attention lookup per prefill bucket
// - 1 attention lookup for decode

// For Llama-2-70B (80 layers), single prefill request:
// 7 × 80 = 560 GEMM lookups per step
```

**Questions:**
- What's the lookup latency? Is it O(n) scan or O(log n) with indexing?
- Should we cache recent lookups (likely high locality)?
- Does this become a bottleneck at scale (e.g., 10K RPS simulation)?

Current implementation uses linear scan (`for i := range rows`). Consider:
```go
// Build index at initialization
type GEMMIndex struct {
    byKN map[[2]int][]GEMMRow  // Map (k,n) → sorted list of rows by m
}

func (db *MFUDatabase) GetGEMMmfu(m, k, n int) float64 {
    rows := db.gemmIndex.byKN[[2]int{k, n}]  // O(1) map lookup
    // Binary search in sorted list (O(log n))
    idx := sort.Search(len(rows), func(i int) bool { return rows[i].M >= m })
    // ...
}
```

### 5.3 Missing: Observability and Debugging

When predictions are wrong, how do you debug? Need instrumentation:

```go
type StepDebugInfo struct {
    StepID         int
    Phase          string  // "prefill", "decode", "mixed"
    BatchSize      int

    // Compute breakdown
    GEMMTimeS      float64
    AttnTimeS      float64
    ComputeTotalS  float64

    // Memory breakdown
    WeightBytesGB  float64
    KVBytesGB      float64
    MemoryTimeS    float64

    // Bottleneck
    Bottleneck     string  // "compute", "memory"

    // Lookup stats
    GEMMLookupsMiss int  // How many nearest-neighbor fallbacks?
    AttnLookupDist  float64  // Euclidean distance for attention lookup

    // Overheads
    CommOverheadS   float64
    SchedOverheadS  float64
}

func rooflineStepTimeV2(...) (int64, StepDebugInfo) {
    debug := StepDebugInfo{}
    // ... populate throughout function ...
    return totalMicros, debug
}
```

This allows:
```go
logrus.WithFields(logrus.Fields{
    "step": debug.StepID,
    "phase": debug.Phase,
    "gemm_ms": debug.GEMMTimeS * 1000,
    "attn_ms": debug.AttnTimeS * 1000,
    "memory_ms": debug.MemoryTimeS * 1000,
    "bottleneck": debug.Bottleneck,
}).Info("Step complete")
```

And post-hoc analysis:
```python
# Which steps are memory-bound vs compute-bound?
df = pd.read_json("sim_debug.log", lines=True)
print(df['bottleneck'].value_counts())

# Are prefill steps compute-bound and decode memory-bound (as expected)?
print(df.groupby('phase')['bottleneck'].value_counts())

# Are GEMM lookups frequently missing (suggesting sparse data)?
print(f"GEMM miss rate: {df['gemm_lookups_miss'].sum() / len(df) * 100:.1f}%")
```

### 5.4 Missing: Multi-GPU Support Beyond TP

Plan assumes TP (tensor parallelism) but doesn't mention:
- **Pipeline parallelism (PP)**: Common for 70B+ models (e.g., TP=4, PP=2 for 8 GPUs)
- **Expert parallelism (EP)**: Required for MoE models like Mixtral-8x22B
- **Data parallelism (DP)**: For high-throughput serving

Each has different communication patterns and MFU characteristics. This might be out of scope, but should be documented as a limitation.

### 5.5 Missing: Regression Test Suite

To prevent regressions when modifying roofline V2:

```go
// sim/roofline_test.go
func TestRooflineV2_Llama2_7B_H100_TP1(t *testing.T) {
    // Fixture: known configuration
    modelConfig := loadModelConfig("testdata/llama-2-7b/config.json")
    hwConfig := HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35}
    mfuDB := loadTestMFUDatabase("testdata/bench_data")

    // Test case: Single prefill, seq_len=2048
    stepConfig := StepConfig{
        PrefillRequests: []PrefillRequestConfig{
            {ProgressIndex: 0, NumNewPrefillTokens: 2048},
        },
    }

    result := rooflineStepTimeV2("h100", modelConfig, hwConfig, stepConfig, 1, mfuDB)

    // Expected: ~15ms (hand-calculated or from vLLM measurement)
    expectedMicros := int64(15000)
    tolerance := float64(expectedMicros) * 0.1  // 10% tolerance

    assert.InDelta(t, expectedMicros, result, tolerance,
        "Prefill time mismatch: expected %d±10%%, got %d", expectedMicros, result)
}

func TestRooflineV2_MixedBatch(t *testing.T) {
    // Test: 4 decode + 1 prefill (common mixed batch scenario)
    stepConfig := StepConfig{
        DecodeRequests: []DecodeRequestConfig{
            {ProgressIndex: 1024}, {ProgressIndex: 2048},
            {ProgressIndex: 4096}, {ProgressIndex: 8192},
        },
        PrefillRequests: []PrefillRequestConfig{
            {ProgressIndex: 0, NumNewPrefillTokens: 512},
        },
    }

    result := rooflineStepTimeV2(...)
    // Assert mixed batch heuristics work as expected
    // (This would require vLLM measurement for ground truth)
}
```

Run on every commit to catch unintended changes.

---

## 6. RECOMMENDATIONS

### 6.1 Immediate Actions (Week 1)

**Priority 0: Resolve MFU Semantics (Blocker)**
1. Clone InferSim repository
2. Find MFU calculation code (likely in `benchmark/` or `simulator/`)
3. Document exact formula: `mfu = ? / ?`
4. Verify: Run one InferSim benchmark manually, recompute MFU from latency, check it matches CSV
5. Update docs/research.md with findings
6. Adjust `hardware_config.json` or formula in `roofline_step_v2.go` if needed

**Priority 1: Minimal Validation (Sanity Check)**
1. Run vLLM benchmark: Llama-2-7B, H100, TP=1, chatbot workload (100 requests)
2. Run simulator with same workload
3. Compare aggregate metrics: mean latency, throughput, P99 latency
4. If error > 20%, stop and debug. If error < 20%, proceed to broader validation.

**Priority 2: Memory Bandwidth Reality Check**
1. Run simulator for decode-only workload: bs=1, model=Llama-2-70B, TP=4, kv_len=8192
2. Check log output: is bottleneck "compute" or "memory"?
3. If "compute" → wrong! Should be memory-bound.
4. Add back `DecodeBwEfficiency` factor (0.5-0.6) to `HardwareCalib`
5. Rerun and verify bottleneck switches to "memory"

### 6.2 Short-Term (Weeks 2-4)

**Validation Campaign:**
- Expand to 3 models: Llama-2-7B, Llama-2-13B, Llama-2-70B
- Test 3 workloads: chatbot, summarization, long-context
- Test 2 TP configs: TP=1, TP=4
- Target: P95 error < 15% (match InferSim paper)

**Critical Bug Fixes:**
- Fix zero-MFU fallback (use nearest by distance, not first in array)
- Add MFU=0 handling for batch_size=0 edge case
- Make `/1.8` factor configurable in `hardware_config.json`

**Observability:**
- Add `StepDebugInfo` return from `rooflineStepTimeV2`
- Log compute vs memory bottleneck per step
- Add `--debug` CLI flag for verbose output

### 6.3 Medium-Term (Months 2-3)

**Improve Accuracy:**
- Implement decode KV-length bucketing (if heterogeneous batch error > 20%)
- Add prefill MFU interpolation (if bucketing error > 15%)
- Collect more benchmark data: finer granularity (10 seq_len points instead of 5)

**Expand Coverage:**
- Run InferSim benchmarks on A100
- Add TP=8 benchmarks for 70B models
- Add attention configs: 40-40-128 (Llama-3), 56-8-128 (Qwen)

**Productionization:**
- Add regression test suite (10-15 test cases)
- Write developer documentation (how to add new GPU, new model, debug predictions)
- Performance optimization: index GEMM lookups, cache repeated queries

### 6.4 Long-Term (Month 4+)

**Advanced Features:**
- Pipeline parallelism support (multi-stage models)
- MoE support (Mixtral, DeepSeek)
- Chunked prefill handling
- Multi-GPU data parallelism

**Research:**
- Investigate learned MFU models (predict MFU from shape features, reduce benchmark data dependency)
- Compare against other simulators (Vidur, Sarathi, InferSim)
- Publish validation results (blog post, paper)

---

## 7. QUESTIONS

### 7.1 For InferSim Authors

1. **MFU semantics**: What is the exact formula for MFU in your benchmark data? Why are values > 1.0?
2. **Decode attention**: Do you use max(kv_len) or weighted average for heterogeneous batches?
3. **Memory bandwidth**: Do you model effective bandwidth or peak? What factors do you use?
4. **TP scaling**: Is prefill TP scaling linear (1/tp) or sublinear? Do you have TP-specific prefill benchmarks?
5. **Hardware factor**: What is the `/1.8` factor for prefill attention? Is it hardware-specific?

### 7.2 For vLLM Developers

1. **PagedAttention batching**: Does decode attention processing time scale with `sum(kv_lengths)` or `max(kv_length) × batch_size`?
2. **Mixed batches**: How does vLLM schedule prefill vs decode in mixed batches? Sequential or overlapped?
3. **Chunked prefill**: What's the chunking strategy for long prefills? Fixed size (e.g., 2048) or adaptive?

### 7.3 For Simulator Users

1. **Use cases**: What are the primary use cases? (Capacity planning, cost estimation, architecture exploration?)
2. **Accuracy requirements**: What prediction error is acceptable? 10%? 20%? 50%?
3. **Workload characteristics**: What's the distribution of batch sizes, sequence lengths, prefill/decode ratios?
4. **Hardware targets**: Which GPUs are most important? H100, A100, L40S, B200?

---

## 8. CONCLUSION

**Summary**: This is a well-executed implementation of a research idea, but it's being presented as a plan when it's actually a retrospective. The code quality is high, the design is generally sound, but there are three showstopper issues:

1. **MFU > 1.0 ambiguity** — fundamental correctness concern
2. **No validation** — how do we know it works?
3. **Memory bandwidth regression** — likely under-estimates decode latency

The document spends 90% of its effort describing construction (already done) and 10% on verification (not started). The recommendation priorities need to flip immediately.

**Final verdict**: The implementation is **not production-ready** until:
- ✅ MFU semantics verified and documented
- ✅ At least one end-to-end validation against vLLM (P95 error < 15%)
- ✅ Memory bandwidth modeling validated for decode workloads
- ✅ Observability added (bottleneck logging, debug output)

After these are addressed, it's a strong foundation for an LLM inference simulator. The existing code is ~80% of the way there — the remaining 20% is testing, validation, and hardening.

**Recommendation for authors**: Retitle this document as "MFU-Based Roofline Implementation: Technical Specification and Validation Plan", split into two sections (implemented vs to-be-validated), and focus all immediate effort on the validation campaign outlined in Section 6.1-6.2.

---

# Executive Summary

**Document Type:** Implementation Plan Review  
**Subject:** MFU-Based Roofline Model for LLM Inference Simulation  
**Review Date:** 2026-02-20  
**Reviewers:** Internal Analysis + Claude Opus 4.6 (External Expert)

## TL;DR: Critical Findings

**🚨 SHOWSTOPPER:** The implementation plan has a **fundamental identity crisis**. It presents itself as a "plan" for future work, but **913 lines of production-quality code already exist** (Tasks 1-10 complete). The actual remaining work is CLI integration and resolving MFU semantics.

**🔴 CRITICAL ISSUE:** MFU values > 1.0 in benchmark data contradict the standard definition of "Model FLOPs Utilization" (fraction of peak ≤ 1.0). This semantic ambiguity undermines the entire approach and requires immediate clarification from InferSim authors.

**📊 OVERALL ASSESSMENT:** Strong implementation with excellent code quality. Validation framework available (`blis_evaluator.py`). Not production-ready until MFU semantics resolved.

---

## Executive Overview

### What Was Reviewed

We reviewed the "MFU-Based Roofline Model: Go Implementation Plan" (12 tasks) against:
- Current codebase (`sim/roofline_step_v2.go`, `sim/mfu_database.go`)
- InferSim reference implementation (`InferSim/layers/attn.py`)
- InferSim technical report (September 2025)
- H100 benchmark data (`bench_data/h100/`)

### Review Methodology

Two independent critiques:
1. **Internal Analysis:** Comprehensive codebase exploration, design review, gap analysis
2. **External Expert Review:** Claude Opus 4.6 technical critique focusing on architecture, risks, and alternatives

Both reviews **independently converged** on the same critical issues, increasing confidence in findings.

---

### 🟡 HIGH: Undocumented Magic Numbers

**Issue:** Implementation blindly copies InferSim's `/1.8` factor without understanding or documenting its purpose.

**Location in InferSim:**
```python
# InferSim/layers/attn.py:86
attn_core_time = (
    seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)
)
```

**Location in Plan:**
```go
// docs/plans/2026-02-20-mfu-roofline-go-implementation.md:862
// Note: InferSim divides prefill attention by 1.8 (hardware-specific factor)
attnCoreTimeS := bucketVectorFlops / 1.8 / (peakFlops * attnMFU)
```

**What We Know:**
- ❌ Not documented in InferSim technical report
- ❌ No explanation in InferSim code comments
- ❌ No mention in git history
- ❌ Not in InferSim README

**What We Don't Know:**
- What does 1.8 represent? (Warp efficiency? Memory access overhead? FlashAttention scheduling?)
- Is it H100-specific or universal?
- Does it apply to all attention configs or only certain head dimensions?
- Should it be part of MFU measurement or separate correction factor?

**Impact:**
- May cause 80% over-estimation of prefill attention time if applied incorrectly
- Portability issues (is this factor valid for H20, A100, etc.?)
- Risk of double-counting if MFU already accounts for this overhead

**Recommendation:**
1. **Investigate through profiling:** Compare predicted vs actual prefill attention time with/without factor
2. **Make configurable:** Add `AttentionPrefillCorrectionFactor` to hardware config
3. **Document uncertainty:** Add comment acknowledging this is empirical, not understood
4. **Contact InferSim authors:** Request explanation or point to relevant paper

---

### 🟡 ~~MEDIUM: Decode Aggregation Design Decision~~ (REMOVED - See Review Update)

### 🟢 ~~LOW-MEDIUM: Prefill Bucketing Granularity~~ (REMOVED - See Review Update)

---

## Design Strengths

Despite critical issues, the implementation has significant strengths:

### ✅ Excellent Code Quality
- Clean separation of concerns (database, lookup, roofline logic)
- Comprehensive error handling with informative messages
- Well-structured CSV loading with validation
- Good use of Go idioms (deferred file closes, error wrapping)

### ✅ Strong Architectural Decisions
- **Nearest neighbor fallback:** Handles missing attention configs gracefully
- **TP-specific files:** Properly models tensor parallelism effects
- **Two-stage GEMM lookup:** Faithful port of InferSim algorithm
- **Backward compatibility:** V1/V2 coexist, conditional usage based on MFU database availability

### ✅ Maintainability
- No new calibration parameters (simplified from 15+ down to 2)
- Centralized MFU data in CSV files (easy to update/extend)
- Clear logging helps debugging ("Loaded MFU database: H100, attention config 32-32-128...")
- Modular design enables independent testing

---

## Recommendations (Actionable Roadmap)

### Phase 1: Hardening (Optional, Months 2-3)

**Robustness:**
- [ ] Handle zero MFU values with better strategy (warn + use V1 fallback)
- [ ] Add bounds checking (seqLen, batchSize, kvLen within CSV range)
- [ ] Test with corrupted CSV files (missing headers, wrong column count)
- [ ] Add unit tests for lookup functions (edge cases, interpolation logic)

**Observability:**
- [ ] Add debug logging: per-operation MFU, GEMM shapes, bottleneck identification
- [ ] Expose metrics: MFU cache hit rate, nearest neighbor fallback frequency
- [ ] Create profiling mode that outputs detailed breakdown (per-layer, per-operation)

**Performance:**
- [ ] Profile CSV loading overhead (consider caching or pre-processing)
- [ ] Optimize lookup algorithms (consider indexing or binary search for large CSVs)
- [ ] Benchmark simulator performance (ensure < 100ms latency for typical workloads)

---

### Phase 2: Long-Term Improvements (Month 4+)

**Data Collection:**
- [ ] Benchmark more GPUs (A100, H20, H200)
- [ ] Collect MFU data for more attention configs (cover Llama-3, Qwen, Mistral)
- [ ] Add FP8, INT8 quantization support
- [ ] Measure MFU variation with different context lengths (32K, 128K, 1M)

**Advanced Features:**
- [ ] Chunked prefill modeling (multi-stage prefill with intermediate decodes)
- [ ] Speculative decoding support (draft model + verification)
- [ ] Pipeline parallelism (PP) support beyond current TP modeling
- [ ] Disaggregated inference (separate prefill/decode servers)

**Validation Expansion:**
- [ ] Validate against multiple serving frameworks (SGLang, TensorRT-LLM)
- [ ] Test on diverse hardware (cloud, edge, multi-tenant)
- [ ] Long-running tests (multi-hour workloads, memory leaks, accuracy drift)

---

## Conclusion

### Summary of Findings

This implementation plan describes a **nearly complete MFU-based roofline model** with excellent code quality and architectural design. However, it suffers from two critical issues:

1. **Identity Crisis:** Presents as "plan" but is actually "specification of implemented system"
2. **Semantic Ambiguity:** MFU > 1.0 values undermine credibility and require immediate clarification

### Overall Assessment

**Technical Merit:** ⭐⭐⭐⭐ (Strong)
- Clean implementation, good design decisions, maintainable codebase

**Readiness:** 🔴 NOT PRODUCTION-READY
- Blocked on MFU semantics resolution
- Missing critical edge case handling

**Path Forward:** ✅ ACHIEVABLE
- 80-90% complete, remaining work is CLI integration and MFU semantics clarification
- Clear action items, well-scoped phases
- Strong foundation to build on

### Final Recommendation

**DO NOT merge or deploy** until:
1. ✅ MFU semantics clarified (contact InferSim authors)
2. ✅ CLI integration completed (`--roofline-version v2-mfu`)
3. ✅ Memory bandwidth modeling validated for decode workloads (optional but recommended)

**After resolving blockers:** This is a strong foundation for an accurate LLM inference simulator. The empirical MFU approach is sound, the implementation is clean, and the architecture is extensible.

---

## Appendix A: Questions for InferSim Authors

If contacting InferSim team, ask:

1. **MFU Semantics:** Why do benchmark CSV files contain MFU values > 1.0? What is the denominator in your MFU calculation?
2. **1.8 Factor:** What does the `/1.8` correction factor represent in prefill attention core time calculation?
3. **Benchmark Methodology:** How were MFU values collected? (Kernel timing? FLOP counting? Hardware counters?)
4. **Zero MFU Values:** Why do some GEMM benchmarks have mfu=0.0? Measurement error or actual zero performance?
5. **TP Scaling:** Are decode MFU values measured on TP=1 then scaled, or independently measured per TP degree?
6. **H100 Specifics:** Are any of the MFU characteristics specific to H100 architecture (e.g., Hopper TMA, FP8 Tensor Cores)?
7. **Validation:** What accuracy did you achieve when comparing InferSim predictions to vLLM/SGLang actual performance?
8. **Edge Cases:** How does InferSim handle chunked prefill, mixed precision, or very long contexts (>100K tokens)?

---

## Appendix B: Code Review Highlights

### Excellent Patterns (To Keep)

**Nearest Neighbor Fallback:**
```go
func findNearestConfig(target string, available []AttentionShape) string {
    // Euclidean distance in (num_heads, num_kv_heads, head_dim) space
    // Gracefully handles missing configs without crashing
}
```

**Informative Logging:**
```go
logrus.Infof("Loaded MFU database: %s, attention config %s (model: %s), %d prefill rows",
    gpu, attentionConfig, originalConfig, len(prefillData[attentionConfig]))
```

**Proper Error Handling:**
```go
if len(rows) == 0 {
    logrus.Fatalf("No prefill MFU data for config %s - database corrupted", db.attentionConfig)
}
```

### Areas for Improvement

**Magic Number (1.8):**
```go
// BEFORE: Undocumented
attnCoreTimeS := bucketVectorFlops / 1.8 / (peakFlops * attnMFU)

// AFTER: Configurable + documented
attnCorrectionFactor := hwConfig.AttentionPrefillCorrectionFactor  // 1.8 for H100
if attnCorrectionFactor == 0 {
    attnCorrectionFactor = 1.0  // Default: no correction
}
attnCoreTimeS := bucketVectorFlops / attnCorrectionFactor / (peakFlops * attnMFU)
```

**Bandwidth Regression:**
```go
// BEFORE: Naive (no adjustment)
decodeEffBW := peakBW

// AFTER: Realistic effective bandwidth
decodeEffBW := peakBW * hwConfig.DecodeBwEfficiency  // e.g., 0.60 for scattered access
if decodeEffBW == 0 {
    decodeEffBW = peakBW  // Fallback if not configured
}
```

**Zero MFU Handling:**
```go
// BEFORE: Simplistic
if mfu == 0.0 {
    logrus.Fatalf("Zero MFU found")
}

// AFTER: Graceful fallback
if mfu == 0.0 {
    logrus.Warnf("Zero MFU for GEMM(%d,%d,%d), falling back to V1 estimation", m, k, n)
    return computeGEMMTimeV1(m, k, n, peakFlops, fallbackMFU)
}
```

---

**End of Executive Summary**

---

**Generated by:** Claude Code (Sonnet 4.5) + Claude Opus 4.6 (External Reviewer)  
**Review Session:** 2026-02-20  
**Total Analysis:** 1939 lines (Problem + Plan + Background + Critique + Review + Summary)

