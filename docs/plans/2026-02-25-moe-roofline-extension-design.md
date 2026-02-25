# MoE Roofline Extension Design

**Date**: 2026-02-25
**Status**: Approved
**Type**: Specification
**Scope**: Extend BLIS roofline latency model for Mixture-of-Experts (MoE) architectures

## 1. Problem Statement

BLIS's roofline latency model (`sim/roofline_step.go`) assumes dense SwiGLU FFN layers — 3 GEMM projections (gate, up, down) per transformer layer with standard GEMM MFU. Modern LLMs increasingly use MoE architectures (DeepSeek-V3: 256 experts, Qwen3-MoE: 128 experts) where the FFN layer is replaced by sparse expert routing. The current model cannot predict MoE inference latency.

**Reference implementation**: InferSim (Alibaba) provides a validated MoE roofline model achieving 4-15% accuracy against real hardware (DeepSeek-V3, Qwen3-30B-A3B, Qwen3-8B on SGLang). This design adapts InferSim's approach into BLIS's Go-based discrete-event simulator.

## 2. Design Decisions

### 2.1 Target Models
Generic MoE support for any HuggingFace MoE config, including:
- DeepSeek-V3 (256 routed experts, top-8, 1 shared expert, MLA attention)
- Qwen3-MoE (128 routed experts, top-8, GQA attention)
- Any model with `num_routed_experts` / `num_experts` in its HF config

### 2.2 Parallelism
Full EP (Expert Parallelism) support alongside existing TP (Tensor Parallelism). Includes all-to-all dispatch/combine communication latency modeling.

### 2.3 MFU Strategy
Grouped GEMM MFU lookup from benchmark CSV data (InferSim format) with analytical fallback when data is missing. Requires extending InferSim benchmark scripts for H100.

### 2.4 Compute-Communication Overlap
Both micro-batch pipelining (for prefill EP) and DeepEP free-decode-communication mode.

## 3. ModelConfig Extension

Add 5 new fields to `ModelConfig` in `sim/model_hardware_config.go`:

```go
type ModelConfig struct {
    // Existing fields (unchanged)
    NumLayers       int     `json:"num_hidden_layers"`
    HiddenDim       int     `json:"hidden_size"`
    NumHeads        int     `json:"num_attention_heads"`
    NumKVHeads      int     `json:"num_key_value_heads"`
    VocabSize       int     `json:"vocab_size"`
    BytesPerParam   float64 `json:"bytes_per_param"`
    IntermediateDim int     `json:"intermediate_size"`

    // New MoE fields
    IsMoE              bool `json:"-"`                   // derived, not from JSON
    NumRoutedExperts   int  `json:"num_routed_experts"`  // total routed experts
    NumExpertsPerToken int  `json:"num_experts_per_tok"` // top-k activated per token
    MoEIntermediateDim int  `json:"moe_intermediate_size"` // per-expert FFN dim
    NumSharedExperts   int  `json:"num_shared_experts"`  // shared experts (default 0)
}
```

### Extraction Logic in `GetModelConfig()`

```
1. Check HF config for "num_routed_experts" or "num_experts"
2. If found → IsMoE = true
   - MoEIntermediateDim = hf["moe_intermediate_size"]
   - NumExpertsPerToken = hf["num_experts_per_tok"]
   - NumSharedExperts = hf["num_shared_experts"] (default 0)
   - IntermediateDim = hf["intermediate_size"] (kept for shared experts / dense layers)
3. If not found → IsMoE = false, all MoE fields = 0
   - Dense FFN uses IntermediateDim as before
```

**Design rationale**: For MoE models, `IntermediateDim` holds the dense FFN intermediate size (used for shared experts). `MoEIntermediateDim` holds the per-expert size. This matches HuggingFace config conventions where both fields coexist.

## 4. MoE Compute Time Model

### 4.1 Per-Layer MoE Formula

From InferSim tech report (Section 2, MoE/FFN):

```
FLOPs_MoE = 6 × S_q × d_hidden × d_inter × n_act
t_MoE     = FLOPs_MoE / (FLOPS_gpu × MFU_MoE)
```

Where:
- `S_q` = number of tokens (batch size for decode, total prefill tokens for prefill)
- `d_hidden` = model hidden dimension
- `d_inter` = per-expert intermediate dimension (`MoEIntermediateDim`)
- `n_act` = number of activated experts per token (`NumExpertsPerToken`)
- `MFU_MoE` = grouped GEMM MFU from benchmark data

The `6` factor = 3 SwiGLU projections × 2 (multiply-add per FLOP).

### 4.2 Expert Weight Loading Floor

At small batch sizes, MoE becomes memory-bound — loading all expert weights from HBM dominates compute:

```
expert_weight_bytes = 3 × hidden × moe_inter × bytesPerParam × numRoutedExperts / ep
load_time = expert_weight_bytes / peakBW
```

Per-MoE-layer time: `moe_time = max(routed_compute_time, load_time)`

### 4.3 Shared Expert Compute

If `NumSharedExperts > 0`:

```
shared_up_time   = computeGEMMTime(tokens, hidden, sharedInter * 2 * numShared, ...)
shared_down_time = computeGEMMTime(tokens, sharedInter * numShared, hidden, ...)
shared_time = shared_up_time + shared_down_time
```

Shared experts use standard (dense) GEMM MFU, not grouped GEMM MFU. They are **additive** to the routed expert time (not max'd), because they run sequentially after routed experts:

```
total_moe_time = max(routed_time, load_time) + shared_time
```

### 4.4 New Functions

| Function | Responsibility |
|----------|---------------|
| `computeAttentionGEMMTimes()` | Q,K,V,O projections only (extracted from `computeTransformerGEMMTimes`) |
| `computeMoELayerTime()` | Routed expert FLOPs + grouped GEMM MFU + weight loading floor + shared experts |
| `computeEPCommunication()` | Dispatch + combine all-to-all bytes / bandwidth |
| `pipelineOverlap()` | Micro-batch pipeline scheduling of compute and communication |

## 5. Expert Parallelism (EP) and Communication

### 5.1 EP Model

EP and TP are independent parallelism dimensions. EP distributes experts across GPUs:

- Expert weights per GPU: `numRoutedExperts / ep`
- Weight loading bytes scale by `1/ep`
- Grouped GEMM MFU lookup uses `ep` as `num_gpus` parameter
- Tokens per GPU for MoE: `totalTokens / ep`
- Attention uses TP only (replicated across EP ranks)

### 5.2 Communication Model

For EP > 1, each MoE layer incurs dispatch + combine all-to-all:

**Dispatch** (before MoE, tokens → expert GPUs):
```
dispatch_bytes = tokens × top_k × hidden × bytesPerElem
dispatch_time  = dispatch_bytes / comm_bandwidth
```

**Combine** (after MoE, results → token GPUs):
```
combine_bytes = tokens × top_k × hidden × bytesPerElem
combine_time  = combine_bytes / comm_bandwidth
```

**Bandwidth selection**:
- Intra-node (all EP ranks on same node): `NVLinkBWTBs`
- Inter-node (EP spans multiple nodes): `RDMABWTBs`
- Selection based on `ep > gpusPerNode` (requires `NumNodes` config)

### 5.3 Compute-Communication Overlap

Two overlap modes:

**Mode 1: Micro-batch pipelining** (for prefill with EP):
With `numMicroBatches` (default 2 for MoE with EP):

```
mb_tokens    = tokens / numMicroBatches
mb_compute   = moe_compute(mb_tokens)
mb_dispatch  = dispatch_comm(mb_tokens)
mb_combine   = combine_comm(mb_tokens)

// Pipeline: steady-state per-layer
total = max(mb_compute, mb_dispatch + mb_combine) × numMicroBatches + startup_drain
// Simplified for 2 micro-batches:
total = mb_dispatch + max(mb_compute, mb_dispatch) + max(mb_compute, mb_combine) + mb_combine
```

**Mode 2: DeepEP free decode communication**:
For decode with DeepEP low-latency mode, communication doesn't occupy GPU SMs:
```
total = moe_compute_time  // communication fully hidden
```

### 5.4 New Config Fields

```go
// In HardwareCalib:
NVLinkBWTBs  float64 `json:"nvlinkBwTBs"`   // intra-node NVLink BW (TB/s)
RDMABWTBs    float64 `json:"rdmaBwTBs"`      // inter-node RDMA BW (TB/s)

// New roofline parameters (passed to rooflineStepTime or via config):
EP              int   // expert parallelism degree (default 1)
NumNodes        int   // number of nodes (for bandwidth selection)
NumMicroBatches int   // micro-batches for overlap (default 1, typically 2 for MoE EP)
DeepEPMode      bool  // decode communication fully overlapped
```

## 6. MFU Database Extension

### 6.1 Grouped GEMM MFU Lookup

New method on `MFUDatabase`:

```go
func (db *MFUDatabase) GetGroupedGEMMMFU(
    numExperts, numGPUs, topK, hidden, inter, batchOrSeqLen int,
    phase string, // "prefill" or "decode"
) (upMFU float64, downMFU float64)
```

**CSV format** (same as InferSim `bench_data/grouped_gemm/`):
```csv
num_experts,num_gpus,num_local_experts,topk,hidden_size,intermediate_size,
batch_size_per_gpu,tokens_per_expert,up_proj_us,up_mfu,down_proj_us,down_mfu
```

**Lookup logic**:
1. Filter rows by `(num_experts, num_gpus, topk, hidden, inter)`
2. Find largest `batch_size_per_gpu ≤ target` (decode) or `seq_len_per_gpu ≤ target` (prefill)
3. Return `(up_mfu, down_mfu)` — caller uses `max(up_mfu, down_mfu)` as effective MFU

### 6.2 Analytical Fallback

When no CSV data matches the configuration:

```
sparsity = topK / numExperts
AI = 2 × batchSize × topK           // arithmetic intensity proxy
R  = peakFlops / peakBW              // hardware roofline point
estimatedMFU = min(AI / R, 0.8)     // capped at observed maximum
```

This captures the key insight from InferSim's tech report: `B_MoE = R / (2S)` where S = sparsity ratio. Below this batch size, MoE is memory-bound; above it, compute-bound.

### 6.3 InferSim Benchmark Script Changes

1. **Add H100 to `hardware/gpu.py`**:
```python
"H100": GPU(fp16_tflops=989.5, fp8_tflops=1979, mfu=0.6,
            mem=80, mem_bw=3350*0.8, nvlink_bw=450, rdma_bw=50)
```

2. **Run existing grouped GEMM benchmark on H100**:
```bash
python kernel_benchmark/deepgemm_grouped_gemm_contiguous.py \
  --config-path hf_configs/deepseek_v3_config.json \
  --gpu-tflops 1979
```

3. **Output**: `bench_data/grouped_gemm/{prefill,decode}/h100/data.csv`

No changes to benchmark script logic — the scripts are already GPU-agnostic. Only hardware definition and execution environment change.

## 7. Memory Access Model for MoE

### 7.1 Weight Loading

Update `calculateMemoryAccessBytes()` for MoE layers:

```
// Attention weights (unchanged):
attnWeightsPerLayer = dModel*(dModel + 2*dKV) + dModel*dModel

// MoE replaces MLP:
moeWeightsPerLayer  = 3 × hidden × moeInter × numRoutedExperts / ep

// Shared experts (always loaded):
sharedWeightsPerLayer = 3 × hidden × sharedInter × numSharedExperts

// Total:
weightsPerLayer = (attnWeights + moeWeights + sharedWeights) × bytesPerParam
```

### 7.2 KV Cache

Unchanged — MoE does not affect KV cache structure.

### 7.3 Activations

Activation memory scales with `tokens × hidden × bytesPerParam × numLayers`. The existing calculation applies — MoE doesn't change activation memory patterns significantly (the routing dispatch uses hidden-dim activations, same as dense FFN).

## 8. Integration into `rooflineStepTime()`

### 8.1 Modified Flow

```
rooflineStepTime(modelConfig, hwConfig, stepConfig, tp, ep, mfuDB):

  // DECODE PHASE
  if hasDecode:
    attnGEMMTime = computeAttentionGEMMTimes(...)       // Q,K,V,O only
    attnCoreTime = ... (unchanged per-request weighted MFU)

    if modelConfig.IsMoE:
      moeTime = computeMoELayerTime(totalBatchSize, ...)
      commTime = computeEPCommunication(totalBatchSize, ...)
      if deepEPMode:
        decodeComputeS = attnGEMMTime + attnCoreTime + moeTime  // comm free
      else:
        decodeComputeS = pipelineOverlap(
          attnGEMMTime + attnCoreTime + moeTime, commTime, numMicroBatches)
    else:
      mlpGEMMTime = computeMLPGEMMTimes(...)               // existing path
      decodeComputeS = attnGEMMTime + attnCoreTime + mlpGEMMTime

  // PREFILL PHASE (similar structure with bucketing)

  // COMBINE PHASES (unchanged)
  stepHardwareS = max(prefillTime, decodeTime)

  // CPU OVERHEAD (unchanged)
  totalS = stepHardwareS + overheadMicros/1e6
```

### 8.2 Backward Compatibility

- Dense models (IsMoE = false): existing code path, no behavioral change
- No changes to attention computation
- No changes to CPU overhead calculation
- No changes to phase combining logic

## 9. Validation Plan

### 9.1 Unit Tests

- `TestMoEModelConfig`: Parse DeepSeek-V3 and Qwen3-MoE HF configs, verify all MoE fields extracted correctly
- `TestMoEComputeTime`: Verify routed expert FLOPs match InferSim's formula: `6 × tokens × hidden × inter × topK`
- `TestExpertWeightFloor`: At small batch size, verify `moe_time == load_time > compute_time`
- `TestSharedExperts`: Verify shared expert time is additive
- `TestGroupedGEMMMFULookup`: Verify CSV parsing and interpolation logic
- `TestEPCommunication`: Verify dispatch/combine bytes formula
- `TestPipelineOverlap`: Verify micro-batch scheduling produces correct overlap
- `TestDenseModelUnchanged`: Golden test ensuring dense model outputs are byte-identical

### 9.2 Integration Tests

- Compare BLIS MoE roofline output against InferSim Python output for:
  - DeepSeek-V3 (EP32, TP1, prefill 4K, 16K tokens/GPU)
  - DeepSeek-V3 (EP128, TP1, decode 4K context, 128 req/GPU)
  - Qwen3-30B-A3B (TP1, EP4, decode)
- Target: within 20% of InferSim's predictions (which are within 15% of real hardware)

## 10. Files Modified

| File | Change |
|------|--------|
| `sim/model_hardware_config.go` | Add 5 MoE fields to ModelConfig, update GetModelConfig(), update ValidateRooflineConfig() |
| `sim/roofline_step.go` | Add computeMoELayerTime(), computeAttentionGEMMTimes(), computeEPCommunication(), pipelineOverlap(); modify rooflineStepTime() |
| `sim/config.go` | Add EP, NumNodes, NumMicroBatches, DeepEPMode to relevant config structs |
| `sim/model_hardware_config.go` | Add NVLinkBWTBs, RDMABWTBs to HardwareCalib |
| `sim/mfu_database.go` (or equivalent) | Add GetGroupedGEMMMFU() method, CSV parsing for grouped GEMM data |
| `hardware_config.json` | Add NVLink/RDMA bandwidth fields |
| `InferSim/hardware/gpu.py` | Add H100 GPU definition |
| `InferSim/bench_data/grouped_gemm/` | New h100/ directories (after benchmarking) |
| Tests | New test files for MoE roofline paths |
