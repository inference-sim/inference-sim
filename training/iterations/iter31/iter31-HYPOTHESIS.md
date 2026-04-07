# Iteration 31: Offset-Subtracted Kernel-Lookup — Beat Trained-Physics

## Context

Iter30 built the kernel-lookup backend but discovered that measured kernel latencies
include ~150µs/layer of CUDA graph launch overhead that doesn't exist in real vLLM.
A constant γ multiplier can't correct this because the overhead is additive (fixed
regardless of token count), not multiplicative. Best manual warm-start: 180% loss
(vs trained-physics 34.57%).

Iter31 adds offset subtraction: subtract estimated launch overhead per layer before
applying γ corrections. This removes the additive bias, letting γ correct only the
residual compute-vs-reality gap. The structural advantage over trained-physics is
that measured GEMM cost curves capture shape-dependent GPU efficiency (which varies
non-linearly with batch size), while the roofline uses a single peak throughput.

## H-main: Offset Subtraction Beats Trained-Physics

**Prediction**: Overall loss < 32% (improvement > 2.5 points from 34.57%).

**Causal mechanism**: The trained-physics model's β corrections are constant across
all batch sizes. At m=128 tokens, the GPU runs at ~30% peak FLOPS; at m=4096, ~85%.
The roofline's single β₁ must compromise across this range. The kernel-lookup model
with offset subtraction uses the **actual measured cost curve** — naturally capturing
the batch-size-dependent efficiency — and only needs γ to correct for step-level
scheduling effects. This is a strictly more informative basis function.

**Diagnostic clause**: If loss does not improve, either (a) the offset subtraction
doesn't accurately model CUDA graph savings (overhead is not purely additive), or
(b) the attention measurements don't improve over the analytical KV-bandwidth model.

## H-offset: δ_gemm Converges Near gemm(m=1)

**Prediction**: The optimized δ_gemm is within 30% of the warm-start value
(gemm(m=1) from each model's profile, ~150µs for Llama-2-7b).

**Causal mechanism**: At m=1, GEMM compute is negligible (~2µs at peak FLOPS) and
the measured 152µs is almost entirely launch overhead. CUDA graphs eliminate this
launch cost, so δ_gemm ≈ gemm(m=1) is the right subtraction.

**Diagnostic clause**: If δ_gemm converges far from gemm(m=1), the overhead is not
purely launch cost — it may include memory allocation, synchronization, or other
effects that CUDA graphs only partially eliminate.

## Formula

```
StepTime = γ₁·max(0, T_gemm - δ_gemm)·L
         + γ₂·max(0, T_pf_attn - δ_attn)·L
         + γ₃·max(0, T_dc_attn - δ_attn)·L
         + γ₄·T_allreduce·allReduceUnits
         + γ₅·T_moe·numMoELayers
         + γ₆·L + γ₇·batchSize + γ₈
```

## Coefficient Mapping (10 params via --beta-coeffs)

| Index | Coeff | Role | Bounds | Warm-start |
|-------|-------|------|--------|------------|
| 0 | γ₁ | GEMM correction | [0.5, 2.0] | 1.0 |
| 1 | γ₂ | prefill attention | [0.5, 2.0] | 1.0 |
| 2 | γ₃ | decode attention | [0.5, 2.0] | 1.0 |
| 3 | γ₄ | AllReduce | [0.0, 2.0] | 1.0 |
| 4 | γ₅ | MoE | [0.0, 5.0] | 0.0 |
| 5 | δ_gemm | GEMM overhead (µs) | [0, 300] | from profile |
| 6 | δ_attn | attention overhead (µs) | [0, 300] | from profile |
| 7 | γ₆ | per-layer (µs) | [0, 100] | 20.0 |
| 8 | γ₇ | per-request (µs) | [0, 20] | 1.0 |
| 9 | γ₈ | per-step (µs) | [0, 200] | 50.0 |

## Implementation Plan

1. Update `kernel_lookup.go` StepTime to subtract δ_gemm and δ_attn before γ multiply
2. Update `kernel_profile.go` / YAML to store per-model overhead estimates (gemm(m=1), attn(b=1,smallest_ctx))
3. Update tests
4. Fix evaluation environment (defaults.yaml path, HF auth) for reliable trained-physics baseline
5. Run warm-start evaluation with offset subtraction
6. Optimize all 10 coefficients with CMA-ES or golden section
7. Compare per-experiment against trained-physics

## Success Criteria

- Overall loss < 32% (vs trained-physics 34.57%)
- γ₁-γ₃ within [0.7, 1.3] (near 1.0 after overhead subtraction)
- δ_gemm within 30% of gemm(m=1)
- No experiment > 2x worse than trained-physics
