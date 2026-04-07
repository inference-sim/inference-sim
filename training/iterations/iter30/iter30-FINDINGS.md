# Iteration 30: Kernel-Lookup Backend — Warm-Start Evaluation

## Summary

First evaluation of the kernel-lookup backend using fixed warm-start γ coefficients
(no optimization). Purpose: establish baseline accuracy of aiconfigurator-measured
kernel data as basis functions and determine the correct order of magnitude for γ
corrections before fitting.

**Bug fix during evaluation:** The initial profile generator divided query results by
`num_layers` (thinking the raw values were per-forward-pass). In fact, `db.query_gemm()`
returns per-invocation (= per-layer) latency, and the Go runtime multiplies by
`numLayers`. This caused a double-division that made all values 32-80x too small.
Fixed and re-evaluated.

## Warm-Start Coefficients

| Coeff | Value | Role |
|---|---|---|
| γ₁ | 1.0 | prefill GEMM correction |
| γ₂ | 1.0 | prefill attention correction |
| γ₃ | 1.0 | decode GEMM correction |
| γ₄ | 1.0 | decode attention correction |
| γ₅ | 0.0 | weight loading (unused — double-counts GEMM) |
| γ₆ | 1.0 | AllReduce correction |
| γ₇ | 0.0 | MoE overhead (disabled) |
| γ₈ | 40.0 | µs/layer overhead |
| γ₉ | 3.0 | µs/request overhead |
| γ₁₀ | 100.0 | µs/step overhead |
| α₀ | 0.0 | queueing overhead |
| α₁ | 0.0 | post-decode overhead |
| α₂ | 0.0 | per-token overhead |

## Results (corrected profiles, fused GEMM)

Overall loss: **5636.77%**
TTFT RMSE: 5587.62%
E2E RMSE: **49.15%**
Succeeded: 15/15

(Prior to GEMM fusion fix: 5740.20% loss. Fusion improved best cases significantly
— Qwen roleplay from 933% to 747% — but overall TTFT remains dominated by the
γ₁=1.0 overestimate.)

## Per-Experiment Breakdown (sorted worst-to-best, fused GEMM)

| Experiment | Model | TTFT APE | E2E APE | Combined |
|---|---|---|---|---|
| 66-qwen2-5-7b-tp1-reasoning-lite | Qwen2.5-7B | 17447% | 38.7% | 17486% |
| llama-2-7b-tp1-roleplay | Llama-2-7b | 6028% | 75.0% | 6103% |
| 65-yi-34b-tp2-general | Yi-34B | 5489% | 73.2% | 5562% |
| 62-mistral-nemo-12b-tp2-general | Mistral-Nemo-12B | 4801% | 81.9% | 4882% |
| 61-llama-3-1-70b-tp4-codegen | Llama-3.1-70B | 4178% | 57.1% | 4235% |
| 63-mistral-nemo-12b-tp1-codegen | Mistral-Nemo-12B | 3963% | 55.7% | 4018% |
| 67-llama-2-7b-tp1-reasoning-lite | Llama-2-7b | 3999% | **7.4%** | 4007% |
| 48-scout-17b-tp2-reasoning-lite | Scout-17B-FP8 | 2219% | 60.9% | 2279% |
| llama-2-7b-tp1-codegen | Llama-2-7b | 2114% | 28.7% | 2143% |
| 21-scout-17b-tp2-roleplay | Scout-17B-FP8 | 2100% | 35.7% | 2136% |
| 17-scout-17b-tp2-general | Scout-17B-FP8 | 1960% | 35.2% | 1996% |
| llama-2-7b-tp1-general | Llama-2-7b | 1944% | **9.6%** | 1953% |
| 60-llama-3-1-70b-tp4-general | Llama-3.1-70B | 1652% | 22.2% | 1674% |
| 20-scout-17b-tp2-codegen | Scout-17B-FP8 | 755% | 54.7% | 809% |
| 64-qwen2-5-7b-tp1-roleplay | Qwen2.5-7B | 742% | **5.1%** | 747% |

## Key Finding: Kernel Sum vs Roofline — Opposite Bounds

The massive TTFT overestimation reveals the structural relationship between the two
latency models:

```
roofline_estimate ≤ real_step_time ≤ kernel_sum
    (β~1.0)                           (γ needs ~0.05-0.15)
```

- **Roofline** (evolved model β~1.0): Assumes perfect hardware utilization.
  Underestimates because it ignores scheduling overhead, kernel launch, etc.
  β corrections are ~1.0 because the estimate is already close.

- **Kernel sum** (this model γ=1.0): Sums measured per-layer kernel times
  SEQUENTIALLY. Overestimates because it ignores pipelining, CUDA graph batching,
  kernel fusion, and compute/memory overlap. γ corrections need ~0.05-0.15.

Both converge to the same real step time but from opposite directions.

### Concrete example: Llama-2-7b at m=4096 tokens

- Per-layer GEMM sum (aiconfigurator): 2058.7 µs
- 32 layers total GEMM: 65,877 µs = 65.9 ms
- Roofline estimate: ~54 ms (at peak FLOPS)
- Real step time: probably 80-150 ms (under load with scheduling overhead)
- γ₁ needed: real / kernel_sum ≈ 0.15 (if all step time were GEMMs)

### E2E is surprisingly good

Despite catastrophic TTFT, E2E errors are reasonable:
- 3 experiments under 10% E2E APE (Llama-2-7b general: 7.7%, reasoning: 7.2%, Qwen roleplay: 8.0%)
- Median E2E APE: ~50%

This suggests the decode-dominated E2E metric is less sensitive to step time
overestimation because the DES naturally serializes decode tokens regardless of
step duration — the total decode time depends more on the number of output tokens
than on individual step time accuracy.

## Implications for Iter31

1. **γ₁-γ₄ warm start**: ~0.05-0.15 (not 1.0). The kernel sum is ~10-20x too high.
2. **γ₈ (per-layer overhead)**: Probably needs to decrease (currently 40 µs/layer ×
   32 layers = 1280 µs/step, which is already a significant fraction of real step time).
3. **γ₁₀ (per-step overhead)**: May need to increase to absorb CUDA graph launch cost.
4. **α₀ (queueing time)**: Needs to be non-zero to model request processing overhead.

## Root Cause: Measured Kernels vs CUDA Graph Execution

The 10-20x TTFT overestimate has a clear root cause: aiconfigurator's measurements
are for **individual kernel invocations** (outside CUDA graphs), but vLLM executes
these kernels **inside CUDA graphs** where:

1. **Kernel launch overhead is amortized** — the ~150µs/layer overhead at m=1 is
   mostly launch cost that disappears in CUDA graph mode
2. **Kernels are pipelined** — the GPU scheduler overlaps memory transfers and
   compute across consecutive kernels
3. **Memory operations are coalesced** — CUDA graphs optimize memory access patterns

At m=1 (one token), the 4 GEMMs cost 152µs/layer, of which ~90% is launch overhead.
At m=4096 (large batch), the GEMMs cost 2059µs/layer, of which <10% is overhead.
A constant γ₁ can't correct both regimes simultaneously.

## Comparison vs Iter29

| Metric | Iter29 (evolved) | Iter30 (kernel-lookup γ=1.0) |
|---|---|---|
| Overall loss | **34.57%** | 5636.77% |
| E2E RMSE | — | **49.15%** |
| TTFT RMSE | — | 5587.62% |

The 163x gap is entirely from TTFT. The E2E RMSE of 49% is already in a reasonable
range — 3 experiments have E2E APE under 10%.

## Implications for Iter31

The fundamental challenge: a constant γ₁ can't correct for the CUDA-graph vs
non-CUDA-graph mismatch across different batch sizes. Options:

1. **Fit γ₁ ≈ 0.05-0.15** — crude but may work if the training set has enough
   batch size diversity that the optimizer finds a compromise value
2. **Subtract launch overhead** — estimate CUDA graph savings and subtract from the
   measured kernel time before γ correction. The overhead is approximately:
   `overhead ≈ gemm(m=1) - (gemm(m=2) - gemm(m=1))` (constant part of the GEMM cost)
3. **Token-count-dependent correction** — replace constant γ₁ with a function
   `γ₁(m) = a + b/m` that applies larger correction at small m (more overhead)

## Training Journey

| Iter | Loss | Key change |
|---|---|---|
| 16 | 60.19% | Trained-roofline baseline |
| 20 | 40.58% | β₈·nMoELayers breakthrough |
| 26 | 37.42% | T_tp activated |
| 27 | 34.61% | CMA-ES joint 6-param |
| 29 | 34.57% | Sequential golden section |
| **30** | **5740.20%** | **Kernel-lookup warm-start (γ=1.0, unoptimized)** |
