# Iteration 30: Kernel-Lookup Backend — Aiconfigurator-Measured Basis Functions

## Context

Iter29 reached 34.57% loss with the evolved model (10 β coefficients). The evolved
model computes basis functions from analytical roofline (peak FLOPS, peak HBM BW) and
uses learned β corrections to bridge the gap to reality. Key observation: β corrections
are large (β₁≈0.77 for prefill, β₃≈1.36 for weight loading) because the roofline
ignores kernel-level effects — CUDA graph overhead, attention fusion, actual memory
access patterns, NCCL topology-aware routing.

NVIDIA's aiconfigurator SDK includes measured GPU kernel latencies for the exact
GPU/backend combinations in our training set (H100/vLLM). By replacing analytical
roofline basis functions with measured kernel lookups, we can make the basis functions
more accurate from the start, requiring only small γ corrections for step-level
scheduling effects.

Design doc: `docs/plans/2026-04-07-kernel-lookup-backend-design.md`

## H-main: Measured Kernels Reduce Overall Loss

**Prediction**: The kernel-lookup backend with optimized γ₁-γ₁₀ achieves overall loss
< 33% (improvement > 1.5 points from 34.57%).

**Causal mechanism**: The analytical roofline's β corrections absorb two categories of
error: (A) kernel-level effects (actual GEMM efficiency, attention fusion, NCCL
topology) and (B) step-level effects (scheduling overlap, Python overhead, CUDA graph
launch). The evolved model's βs must compensate for both A and B simultaneously.
The kernel-lookup backend eliminates category A by using measured data, so γ corrections
only need to handle category B — a smaller, more uniform error source that should
generalize better across batch compositions.

**Diagnostic clause**: If loss does not improve, the analytical roofline basis functions
were already capturing kernel-level effects accurately through the β corrections, and
the remaining 34.57% error is dominated by step-level effects that measured kernels
cannot address.

## H-correction: γ₁-γ₇ Are Tighter Than Corresponding β₁-β₄

**Prediction**: The optimized γ₁-γ₇ (dimensionless kernel corrections) have a narrower
spread around 1.0 than the evolved model's β₁-β₄. Specifically, max(|γᵢ - 1|) < 0.3
for i=1..7 (vs β₁=0.77, β₃=1.36 in evolved model).

**Causal mechanism**: β corrections absorb both kernel inaccuracy and step-level effects.
γ corrections only absorb step-level effects (scheduling overlap between ops within a
step). Since kernel overlap within a step is typically small (vLLM processes ops mostly
sequentially within a CUDA graph), γ should be close to 1.0.

**Diagnostic clause**: If γ corrections are as large as β corrections, the aiconfigurator
measurements do not accurately represent vLLM's actual kernel execution (possibly due to
version mismatch, CUDA graph batching, or measurement methodology differences).

## H-scout: Scout MoE Error Specifically Improves

**Prediction**: Scout experiments (currently ~67.6% loss, the dominant hard case) show
> 5 point improvement with the kernel-lookup backend.

**Causal mechanism**: Scout's interleaved MoE/dense architecture has the largest gap
between analytical roofline and reality. The roofline treats MoE experts as independent
GEMMs, but real execution has router gating, token permutation, and EP communication
overhead. Aiconfigurator's `query_moe()` captures these effects from measured data.
The evolved model's β₈ (MoE overhead, 481 µs/layer) is a single additive constant
that cannot vary with batch size or token distribution — measured MoE data can.

**Diagnostic clause**: If Scout does not improve, the MoE measurement grid in
aiconfigurator does not cover Scout's specific expert configuration (16 experts,
top-2, interleaved with dense layers), and we may need custom profiling.

## Implementation Plan

1. Write `training/scripts/generate_kernel_profile.py` — Python script that queries
   aiconfigurator DB and produces kernel_profile.yaml
2. Generate profiles for all 15 training experiments
3. Implement `sim/latency/kernel_lookup.go` — Go runtime with lookup table interpolation
4. Wire up CLI: `--latency-model kernel-lookup --kernel-profile <path>`
5. Run training loop with initial γ = [1,1,1,1,1,1,1,40,3,100]
6. Optimize γ₁-γ₁₀ (bounds: γ₁-γ₇ ∈ [0.5,2.0], γ₈ ∈ [0,200], γ₉ ∈ [0,50], γ₁₀ ∈ [0,500])

## Success Criteria

- Overall loss < 33% (vs 34.57% baseline)
- γ₁-γ₇ within [0.7, 1.3] (tighter than evolved β₁-β₄)
- Scout loss < 60% (vs ~67.6% baseline)

---

## Outcome (post-evaluation)

### H-main: REFUTED

Best warm-start loss: 180% (γ₁=0.06, γ₂=γ₃=0.1). No γ configuration achieves
< 33%. The kernel-lookup backend with constant γ corrections cannot compete with the
trained-physics model. Root cause: aiconfigurator measures kernels outside CUDA graphs.
The ~150µs/layer launch overhead is additive (not multiplicative), so a constant γ
multiplier cannot correct it — it removes a fraction of both overhead AND compute.

### H-correction: REFUTED

γ₁-γ₃ must be 0.05-0.10 to achieve reasonable TTFT — far from 1.0, and farther from
1.0 than the trained-physics β₁-β₄ (~0.77-1.36). The measured kernels are a worse
starting point than the analytical roofline for this DES use case because the
measurement methodology (outside CUDA graphs) introduces systematic overhead that
roofline estimates don't have.

### H-scout: NOT TESTED

Scout MoE was not separately evaluated because the overall loss was too high for
any model to be meaningful.

### Key Learning

Aiconfigurator's kernel measurements are accurate for **static single-request
predictions** (no DES, no queueing). Using them as basis functions in a DES requires
either: (a) offset subtraction to remove CUDA graph overhead before applying γ,
(b) using aiconfigurator's own static prediction as a compound basis function, or
(c) using the measurements only for cross-model/hardware transfer learning rather
than as primary step-time predictors.
