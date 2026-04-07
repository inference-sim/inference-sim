# Kernel-Lookup Latency Backend Design

**Date:** 2026-04-07
**Type:** Specification
**Status:** Draft

## Problem

BLIS's evolved model (iter29, 34.57% loss) computes step-time basis functions from
**analytical roofline** — peak FLOPS and peak HBM bandwidth. Learned β coefficients
correct for the gap between theoretical peak and real kernel performance. This gap
can be large (β₁≈0.77, β₃≈1.36) because the roofline ignores kernel-level effects:
CUDA graph overhead, attention kernel fusion, memory access patterns, NCCL topology.

NVIDIA's aiconfigurator SDK has a **measured GPU operations database** with actual
kernel latencies for GEMM, attention, MoE, NCCL, and custom AllReduce operations
across multiple GPUs (H100, A100, B200, GB200, L40S) and backends (vLLM, SGLang,
TRT-LLM). These measurements capture the real kernel behavior that the roofline
abstracts away.

## Solution

A new latency backend (`kernel-lookup`) that replaces analytical roofline basis
functions with interpolated lookups from aiconfigurator's measured kernel database.

### Architecture

```
OFFLINE (Python, once per model/GPU/TP combo)

  HF config.json → Model shapes → Aiconfigurator DB queries → kernel_profile.yaml

RUNTIME (Go, per step in DES)

  kernel_profile.yaml → KernelLookupModel → StepTime(batch) with γ-weighted sum
```

**Key insight:** Most GEMM shapes are static (determined by model architecture + TP).
Only the token dimension (`m`) varies per step. Attention also varies by sequence
length. We pre-compute per-operation latency as a function of these dynamic dimensions,
store as lookup tables, and interpolate at runtime.

### Step-Time Formula

```
StepTime = γ₁·T_pf_gemm + γ₂·T_pf_attn + γ₃·T_dc_gemm + γ₄·T_dc_attn
         + γ₆·T_allreduce + γ₇·T_moe
         + γ₈·numLayers + γ₉·batchSize + γ₁₀
```

Note: γ₅ (weight loading) is dropped — GEMM measurements already capture weight
memory access, so a separate T_weight term would double-count.

**Coefficient semantics:**

| Coeff | Type | Expected range | Physical meaning |
|---|---|---|---|
| γ₁-γ₄, γ₆-γ₇ | Dimensionless | ~0.8-1.5 | Correction for step-level effects not captured by per-op measurements (kernel overlap, scheduling, pipeline bubbles) |
| γ₅ | Fixed = 0 | 0 | Weight loading removed (double-counts GEMM memory access) |
| γ₈ | µs/layer | ~30-50 | Per-layer software overhead (kernel launch, sync, layer norm) |
| γ₉ | µs/request | ~2-5 | Per-request overhead in batch (token management) |
| γ₁₀ | µs/step | ~50-200 | Fixed per-step overhead (CUDA graph launch, sampler, Python GIL) |

### Basis Functions from Aiconfigurator

| Term | Source | Query dimensions | Multiplier |
|---|---|---|---|
| T_pf_gemm | Sum of `query_gemm()` for QKV, proj, gate+up, down | m=totalPrefillTokens, n/k from model dims/TP | numLayers |
| T_pf_attn | `query_context_attention()` | b=numPrefillRequests, s=avgISL, heads, kv_heads, head_dim | numLayers |
| T_dc_gemm | Sum of `query_gemm()` for QKV, proj, gate+up, down | m=totalDecodeTokens (=decode batch size) | numLayers |
| T_dc_attn | `query_generation_attention()` | b=totalDecodeTokens, s=avgDecodeContext | numLayers |
| T_allreduce | `query_custom_allreduce()` (table axis: token count) | msg_size=tokens×hidden; 0 when TP=1 | 2·numDenseLayers + 1·numMoELayers |
| T_moe | `query_moe()` (FFN/expert only; attention GEMMs in T_pf/dc_gemm) | tokens, hidden, inter, topk, experts, tp | numMoELayers |

**Critical convention:** All YAML tables store **per-layer latencies**. The Python
script divides query results by `numLayers` before writing. The Go runtime multiplies
by the appropriate layer count for each term.

### Model-to-Ops Mapping

BLIS's `ModelConfig` already contains all fields needed to derive aiconfigurator
GEMM shapes. The mapping replicates aiconfigurator's `LLAMAModel.__init__()` and
`MOEModel.__init__()` dimension math:

**Dense (LLAMA family):** Per layer has 4 GEMMs + attention + 2 AllReduces (TP>1)
- QKV: `(tokens, H*d_h/TP + 2*KV*d_h/TP, hidden_size)` — fused Q+K+V projection
- Proj: `(tokens, hidden_size, H*d_h/TP)` — attention output projection
- Gate+Up: `(tokens, 2*inter/TP, hidden_size)` — SwiGLU fused gate and up
- Down: `(tokens, hidden_size, inter/TP)` — FFN down projection

**MoE (HybridMoE/MOE families):** Dense layers same as above; MoE layers replace
FFN GEMMs with `query_moe()` and have 1 AllReduce (attention only).

**Architecture detection:** Uses BLIS `ModelConfig.NumLocalExperts > 0` for MoE,
`InterleaveMoELayerStep > 0` for interleaved MoE/dense (Scout).

### Pre-computation Pipeline

A Python script (`training/scripts/generate_kernel_profile.py`) produces
`kernel_profile.yaml` for each model/GPU/TP combination:

1. Parse HF config.json (same path BLIS uses via `--model-config-folder`)
2. Compute GEMM shapes from architectural parameters
3. Query aiconfigurator database at a token-count grid [1..4096]
4. Output per-op lookup tables as YAML

**Lookup table format:**

```yaml
gpu: h100_sxm
backend: vllm
version: "0.14.0"
model: meta-llama/Llama-3.1-70B-Instruct
tp: 4

context_gemm:  # tokens → µs per layer (QKV+proj+gate_up+down summed)
  tokens: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
  latency_us: [...]

context_attention:  # (tokens, isl) → µs per layer
  tokens: [1, 2, 4, ...]
  isl: [128, 256, 512, 1024, 2048, 4096]
  latency_us: [[...], ...]  # 2D grid

generation_gemm:  # tokens → µs per layer
  tokens: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  latency_us: [...]

generation_attention:  # (tokens, context) → µs per layer
  tokens: [1, 2, 4, ...]
  context: [128, 256, 512, 1024, 2048, 4096]
  latency_us: [[...], ...]

allreduce:  # message_size → µs per invocation
  message_size: [1024, 4096, 16384, 65536, 262144, 1048576]
  latency_us: [...]

moe_compute:  # tokens → µs per MoE layer (null for dense models)
  tokens: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  latency_us: [...]

weight_loading_us: 1234.5  # constant per step
```

### Go Runtime

New file `sim/latency/kernel_lookup.go`:

```go
type KernelLookupModel struct {
    Gamma [10]float64

    contextGemm      LookupTable1D
    contextAttn      LookupTable2D
    generationGemm   LookupTable1D
    generationAttn   LookupTable2D
    allreduce        LookupTable1D
    moeCompute       *LookupTable1D  // nil for dense models

    weightLoadingUs  float64
    numLayers        int
    numMoELayers     int
    numDenseLayers   int
    hiddenDim        int
    tp               int
    arUnitsPerStep   int  // 2*dense + 1*moe
}
```

`StepTime(batch)` follows the same single-pass O(batch) scan as the evolved model,
accumulating prefill/decode tokens and context lengths, then interpolates lookup
tables instead of computing analytical roofline.

### Graceful Degradation

If no kernel profile is provided, fall back to the evolved model. The CLI flag
`--latency-model kernel-lookup` requires `--kernel-profile <path>`.

### Training Integration

1. Generate kernel profiles for all 15 training experiments (one-time)
2. Optimize γ₁-γ₁₀ using the same training loop (`run_blis_and_compute_loss.py`)
3. Initial γ values: `[1, 1, 1, 1, 1, 1, 1, 40, 3, 100]`
4. Bounds: γ₁-γ₇ in [0.5, 2.0], γ₈ in [0, 200], γ₉ in [0, 50], γ₁₀ in [0, 500]

### Expected Improvement

The analytical roofline's β corrections absorb kernel-level effects. With measured
kernels, γ corrections only need to absorb step-level effects (scheduling overlap,
Python overhead). Expected benefits:

- Lower loss than 34.57% (current evolved model)
- Tighter γ corrections (closer to 1.0 for γ₁-γ₇)
- Better generalization across batch sizes (the token-count dimension where
  roofline diverges most from reality)
- Natural extension to other GPUs/backends by swapping kernel_profile.yaml

## Scope

- **In scope:** H100/vLLM kernel profiles for all 15 training models, Go runtime,
  γ optimization for iter30
- **Out of scope:** A100/L40S profiles (future), TRT-LLM/SGLang backends (future),
  automatic profile generation from BLIS CLI (future)

## Risks

1. **LFS data not checked out:** Aiconfigurator CSV data files are Git LFS stubs in
   this repo. The Python script needs the actual data. Mitigation: run on a machine
   with LFS data, or use aiconfigurator's `EMPIRICAL` mode as fallback.

2. **Interpolation accuracy:** Aiconfigurator's scipy interpolation may not match Go's
   linear interpolation for the same grid. Mitigation: use dense enough grids and
   compare offline.

3. **Scout model coverage:** Scout (HybridMoE) may need MoE-specific queries that
   require the exact expert configuration in the database. Mitigation: check database
   coverage for Scout's dimensions before generating profiles.
