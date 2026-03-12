# Roofline Physics Port from llm-optimizer

**Status:** Approved
**Date:** 2026-03-09
**Species:** Decision record
**Motivation:** BLIS roofline has 929% E2E MAPE and 215% ITL MAPE against ground-truth vLLM experiments (discussion #522). llm-optimizer achieves 32.4% E2E and 36.5% ITL with a simpler roofline. Port its physics into BLIS's existing step-time structure.

## Problem

Three physics issues in `rooflineStepTime()` cause systematic over-prediction:

1. **Dual ceiling model** — BLIS splits compute into GEMM ops and vector ops with separate ceilings (`peakFlops * mfu` and `peakFlops * 0.10`). The vector ceiling is 10x lower than tensor core peak, adding a large serial term even when vector ops are a small fraction. llm-optimizer uses a single `peakFlops * mfu` ceiling for all ops.

2. **Bandwidth haircut** — BLIS applies `BwEffConstant = 0.72` to peak HBM bandwidth, predicting 39% slower than peak for memory-bound phases. llm-optimizer uses raw peak bandwidth. Modern GPU memory controllers achieve near-peak bandwidth for sequential weight/KV reads.

3. **Overhead stacking** — BLIS adds `TOverheadMicros` (500µs) + `PerLayerOverhead` (20µs × layers) + `AllReduceLatency` (20µs × layers × 2 for TP>1) per step. For Llama-2-7B (32 layers, TP=1), this is 1140µs — 28% of a ~4ms decode step. llm-optimizer has no overhead terms. These constants were not derived from measurement and over-compensate.

Combined, these produce ~3x over-prediction for decode (memory-bound) steps.

## Decision

Port llm-optimizer's single-crossover roofline into `rooflineStepTime()`. Keep BLIS's step-time prediction structure (StepConfig in, µs out) and BLIS's superior model-awareness (actual IntermediateDim, SwiGLU 3-matrix MLP, MoE, FlashAttention-aware memory).

### What changes

**In `rooflineStepTime()` (4 changes):**

1. Remove dual ceiling — replace `gemmFlops/(peak*mfu) + vectorFlops/vectorPeak` with `totalFlops/(peak*mfu)`. The `calculateTransformerFlops()` return map already has `"total"` = gemm_ops + sram_ops; use that directly.

2. Remove bandwidth haircut — replace `peakBW * BwEffConstant` with `peakBW`.

3. Remove all overhead terms — delete `layerFloorS`, `commOverheadS`, `TOverheadMicros` from the step time calculation.

4. Use llm-optimizer MFU defaults — update `hardware_config.json`: `MfuPrefill: 0.45`, `MfuDecode: 0.30`.

**What stays the same:**
- Function signature and return type
- Phase separation (prefill loop + decode loop)
- Phase addition (`prefillTime + decodeTime`)
- Per-request FLOPs/memory accumulation
- Weight loading once per phase
- `calculateTransformerFlops()` — unchanged
- `calculateMemoryAccessBytes()` — unchanged
- LatencyModel interface — unchanged
- KV capacity calculation — unchanged

### What we keep from BLIS (better than llm-optimizer)

| Aspect | llm-optimizer | BLIS (kept) |
|--------|--------------|-------------|
| MLP dims | Hardcoded `4 * d_model` | Actual `IntermediateDim` from HF config |
| MLP matrices | 2-matrix (up/down) | 2-matrix (matching llm-optimizer; HF `intermediate_size` already SwiGLU-scaled). KV capacity uses 3-matrix for conservative weight estimation |
| MoE | None | Routed expert FLOPs (top_k) and weights (all E); shared expert and gate weights modeled in KV capacity only |
| Attention memory | Quadratic `[B,H,T,T]` in HBM | FlashAttention-aware (SRAM-local) |
| Prefill context | Full `T²` | Effective context averaging |

## Risks

- **MFU values are per-GPU constants.** Changing from 0.65/0.12 to 0.45/0.30 affects all models on all GPUs. If the evaluation set (H100 only, 4 models) is not representative, accuracy may degrade on other hardware.
- **Removing overheads entirely** means small-batch/short-sequence steps have no floor. If real GPU scheduling overhead is ~100-200µs, the roofline will under-predict for very fast steps.
- **BwEffConstant becomes unused** but remains in `HardwareCalib` struct. It still has meaning for other potential consumers; leave the field, just stop using it in `rooflineStepTime()`.
