# Iter33 Findings

## Summary

| Metric | Value |
|--------|-------|
| **Best loss (CMA-ES)** | **57.48%** |
| Iter32 baseline | 56.49% |
| Iter29 target | 34.57% |
| Gap to target | 22.9pp |

Iter33 introduced two structural changes and ran a full CMA-ES search. The best loss
(57.48%) is 0.99pp worse than iter32, despite the FlashAttention fix being physically
correct. The kernel-lookup model has a structural ceiling around 57-60% with the
current basis functions.

---

## Changes Introduced

### 1. γ₇ Split (Overhead Amortization)

**Problem**: γ₇ = 187µs/layer was a constant per-step overhead regardless of batch
size. At 8 concurrent decode requests × 28 layers = 5.2ms constant overhead dominated
step time and inflated TTFT at high load.

**Fix**: Split into two phase-specific terms:
- `gamma[6]` = γ₇_pf: µs/layer per prefill sequence (scales with #prefill seqs)
- `gamma[9]` = γ₇_dc: µs/layer decode overhead, amortized as `base / √decodeReqs`

**Effect**: Properly models that decode overhead amortizes across the batch.

### 2. FlashAttention Prefix Cache Fix (Correctness Bug)

**Problem**: For a request with KV prefix cache hits (e.g., 150-token system prompt
already cached from a previous request), `StepTime` computed:
```
fullS = ProgressIndex + NumNewTokens = 0 + (ISL - cached) = ISL - cached
prefix = 0
```
This made the FlashAttention lookup at context=(ISL - cached) with correction=1.0,
*underestimating* FlashAttention by ~19% for cache-hit requests. The roleplay
training experiment had 16-35% cache hit rates.

**Fix**: When `ProgressIndex == 0` (first prefill step), use `len(req.InputTokens)`
as `fullS` (the full ISL), since BLIS has already allocated the cached prefix blocks:
```go
if req.ProgressIndex == 0 {
    fullS  = float64(util.Len64(req.InputTokens))  // full KV context
    prefix = fullS - newT                            // = cachedLen (0 if cold start)
} else {
    prefix = float64(req.ProgressIndex)              // chunked prefill
    fullS  = prefix + newT
}
```

This mirrors `trained_physics` which uses `si = len(req.InputTokens)` for the same
reason.

**Test**: `TestKernelLookupModel_PrefixCacheHit_EqualToChunkedSecondStep` verifies
the invariant: a prefix-cache-hit first step and a chunked second step with equal
`(full_s=512, prefix=256, newT=256)` must produce identical step times.

---

## NM Phase Results (Gamma7 Split Only)

Before CMA-ES, ran 2D Nelder-Mead over (gamma[6], gamma[9]) with all other
coefficients held at iter32 values.

| Result | Value |
|--------|-------|
| γ₇_pf (gamma[6]) | 0.0 (prefill overhead negligible) |
| γ₇_dc (gamma[9]) | 484.03 µs/layer |
| Loss | 64.26% |

The γ₇_pf converging to 0 confirms the prefill sequence overhead is not the
dominant term — the decode amortization was the key insight.

---

## CMA-ES Phase Results

**Configuration**:
- 9 free parameters: α₀, γ₁, γ₂, γ₃, γ₇_pf, γ₈, γ₉, γ₇_dc, α₂
- Warm start: iter33-NM best (64.26%)
- Population: 10 per generation
- Parallelism: 2 concurrent evals × 8 BLIS workers = 16 total processes
- ~130s per generation

**Convergence trajectory**:

| Gen | Best Loss | Sigma |
|-----|-----------|-------|
| 1 | 99.2% | 0.198 |
| 3 | 69.6% | 0.201 |
| 5 | 62.3% | 0.217 |
| 6 | 60.2% | 0.229 |
| 12 | 58.7% | 0.196 |
| 25 | 58.7% | 0.158 |
| 46 | **57.5%** | ~0.14 |

Sigma slowly declining but not converged at stop (would need ~200+ more gens).

**Best coefficients (gen 46)**:

| Param | Value | vs Iter32 |
|-------|-------|-----------|
| α₀ (queueing) | 14692 µs | 17658 µs |
| γ₁ (GEMM) | 0.046 | 0.086 — much lower |
| γ₂ (ctx attn) | 0.083 | 0.302 — much lower |
| γ₃ (gen attn) | 0.607 | 0.939 — lower |
| γ₇_pf (pf overhead) | 15.4 µs/layer | 0 (new) |
| γ₈ (per-req) | 445 µs | 86 µs — much higher |
| γ₉ (per-step) | 52 µs | 214 µs — lower |
| γ₇_dc (dc overhead) | 282 µs/layer | 484 (NM) |
| α₂ (per-token) | 14.1 µs | 3.1 µs — much higher |

The most notable shift: γ₁ and γ₂ dropped significantly (lower GEMM and context-attn
corrections), while γ₈ and α₂ increased sharply. This compensates for the prefix-cache
fix making FlashAttention larger — the model needs smaller attention corrections to avoid
over-predicting TTFT.

---

## Analysis: Why We Didn't Beat Iter32

**Root cause**: The FlashAttention prefix-cache fix increases predicted step times for
cache-hit requests, but the kernel-lookup model's other coefficients were calibrated
under the wrong basis. Re-optimizing 9 coefficients together partially compensates,
but the structural floor is around 57-58% with the current feature set.

**The 22.9pp gap to iter29 (34.57%)** comes from two remaining issues identified in
the handoff:
1. ~~γ₇ constant overhead~~ (fixed, iter33)
2. ~~FlashAttention prefix cache~~ (fixed, iter33)
3. **Compute-heavy large models** (70B TP=4, Yi-34B TP=2): profile overestimates step
   time because AIC measurements all show ~14.5ms TTFT at ISL=256 regardless of model
   size, but roofline arithmetic over-predicts for large models where CUDA graph
   execution benefits from tensor core pipelining and NVLink amortization.

---

## Next Ideas (for Iter34+)

1. **Per-model-class γ₁ correction** (Option B from handoff): Add a secondary 1D table
   indexed by numLayers to apply per-model-size corrections on top of γ₁. Directly
   addresses the 70B/34B overestimation.

2. **Hybrid trained-physics + kernel-lookup** (Option C): Use kernel-lookup profiles
   for T_gemm/T_attn basis functions but trained-physics queueing/overhead model.
   Combines measured kernel accuracy with calibrated queueing quality.

3. **Larger CMA-ES run**: Run for 200+ gens — sigma=0.14 at gen 47, still has room.
   Estimated ~4h more compute. Not guaranteed to beat iter32.

4. **Separate prefill-heavy vs decode-heavy experiments**: The 22pp gap may be driven
   by a small subset of experiments. Per-experiment loss breakdown would identify
   which model/workload combinations are hardest.
