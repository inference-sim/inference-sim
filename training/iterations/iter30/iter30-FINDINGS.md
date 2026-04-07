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

## Results (corrected profiles)

Overall loss: **5740.20%**
TTFT RMSE: 5689.98%
E2E RMSE: **50.21%**
Succeeded: 15/15

## Per-Experiment Breakdown (sorted worst-to-best by combined loss)

| Experiment | Model | TTFT APE | E2E APE | Combined |
|---|---|---|---|---|
| 66-qwen2-5-7b-tp1-reasoning-lite | Qwen2.5-7B | 17691% | 39.5% | 17731% |
| llama-2-7b-tp1-roleplay | Llama-2-7b | 6216% | 77.7% | 6294% |
| 62-mistral-nemo-12b-tp2-general | Mistral-Nemo-12B | 5280% | 89.0% | 5369% |
| 65-yi-34b-tp2-general | Yi-34B | 5152% | 69.5% | 5221% |
| 61-llama-3-1-70b-tp4-codegen | Llama-3.1-70B | 4500% | 60.5% | 4561% |
| 67-llama-2-7b-tp1-reasoning-lite | Llama-2-7b | 4061% | **7.2%** | 4068% |
| 63-mistral-nemo-12b-tp1-codegen | Mistral-Nemo-12B | 3949% | 56.1% | 4005% |
| 48-scout-17b-tp2-reasoning-lite | Scout-17B-FP8 | 2245% | 60.8% | 2306% |
| llama-2-7b-tp1-general | Llama-2-7b | 2234% | **7.7%** | 2242% |
| 21-scout-17b-tp2-roleplay | Scout-17B-FP8 | 2185% | 34.6% | 2219% |
| llama-2-7b-tp1-codegen | Llama-2-7b | 2152% | 30.3% | 2183% |
| 17-scout-17b-tp2-general | Scout-17B-FP8 | 1885% | 36.0% | 1921% |
| 60-llama-3-1-70b-tp4-general | Llama-3.1-70B | 1661% | 22.3% | 1683% |
| 64-qwen2-5-7b-tp1-roleplay | Qwen2.5-7B | 925% | **8.0%** | 933% |
| 20-scout-17b-tp2-codegen | Scout-17B-FP8 | 817% | 53.8% | 871% |

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

## Comparison vs Iter29

| Metric | Iter29 (evolved) | Iter30 (kernel-lookup γ=1.0) |
|---|---|---|
| Overall loss | **34.57%** | 5740.20% |
| E2E RMSE | — | **50.21%** |
| TTFT RMSE | — | 5689.98% |

The 166x gap is entirely from TTFT. With γ₁-γ₄ fitted to ~0.1, the TTFT errors
would collapse dramatically and overall loss would likely be competitive with the
evolved model.

## Training Journey

| Iter | Loss | Key change |
|---|---|---|
| 16 | 60.19% | Trained-roofline baseline |
| 20 | 40.58% | β₈·nMoELayers breakthrough |
| 26 | 37.42% | T_tp activated |
| 27 | 34.61% | CMA-ES joint 6-param |
| 29 | 34.57% | Sequential golden section |
| **30** | **5740.20%** | **Kernel-lookup warm-start (γ=1.0, unoptimized)** |
