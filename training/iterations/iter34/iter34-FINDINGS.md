# Iter34 Findings

## Summary

| Metric | Value |
|--------|-------|
| **Best loss (9D CMA-ES)** | **65.99%** |
| Iter33 baseline | 57.48% |
| Iter32 baseline | 56.49% |
| Iter29 target | 34.57% |

Iter34 introduced GEMM floor subtraction (same physics as AllReduce overhead subtraction
already in the code) and retired the unphysical γ₇_dc term. The best found was 65.99%,
which is **worse than iter33 (57.48%)** and significantly worse than iter32 (56.49%).

---

## Changes Introduced

### 1. GEMM Floor Subtraction

**Motivation**: AIC single-kernel-per-CUDA-graph measurements have a flat plateau at
m=1..8 dominated by weight-read memory bandwidth cost (~167µs/layer for Qwen-7B).
In vLLM's full-model CUDA graph, consecutive-layer weight reads partially pipeline
with current-layer compute, amortizing this floor. Same physics as AllReduce overhead
subtraction (already in code since iter30).

**Change**:
```go
// Before:
tGemm = clampPositive(m.gemm.Interp1D(totalTokens)) * L

// After:
tGemm = clampPositive(m.gemm.Interp1D(totalTokens) - m.gemmFloor) * L
tFloor = m.gamma[3] * (m.gemmFloor*L + m.logitsFloor)  // γ_wt term
```

### 2. γ₇_dc Retired (gamma[9] = 0)

**Motivation**: The `γ₇_dc × L / √decodeTokens` term was unphysical — total step-time
overhead DECREASED as decode batch grew. The optimizer was exploiting this incorrectly.

---

## What Went Wrong: Root Cause Analysis

### The Pathological Regime

The 9D CMA-ES drove `γ₁ → 0.012, γ_wt → 0.198`. At decode batch=8:

| Model | GEMM/floor contribution |
|-------|------------------------|
| iter32 | `0.086 × 168µs × 28L = 403µs` |
| iter34 | `0 + 0.198 × 5052µs = 1000µs` |

iter34 over-predicts decode overhead by **2.5×**, inflating TTFT for high-concurrency scenarios.

### Why This Happened

Retiring γ₇_dc removed ~2789µs of overhead at decode=8 (iter33 value). The floor term
(γ_wt × T_floor) partially filled this gap but at the WRONG shape — constant regardless
of batch size rather than per-step per-layer. The optimizer found a local minimum that
fits some experiments but over-predicts decode overhead globally.

The floor subtraction physics is correct but the implementation needs:
1. A replacement for γ₇_dc's functional role (constant per-step per-layer decode overhead)
2. Tighter bounds on γ_wt (≤ 0.08) to prevent the degenerate γ₁→0 regime

---

## Search Results

### 9D CMA-ES (26 gens, 260 evals, ~45 min)

| Gen | Best Loss | σ |
|-----|-----------|---|
| 1 | 110.9% | 0.193 |
| 5 | 79.2% | 0.262 |
| 9 | 68.3% | 0.239 |
| 17 | 65.9% | 0.201 |
| 21 | 65.9% | 0.156 |
| 26 | 65.9% | ~0.130 |

Best coefficients (gen 17):
- α₀=13737, γ₁=0.012, γ₂=0.491, γ₃=1.257, γ_wt=0.198, γ₇_pf=38.3, γ₈=314.9, γ₉=267.8, α₂=14.1

### 4D CMA-ES (3 gens, ~80 evals, focused on γ₂/γ₃/γ₈/γ₉)

Best: 71.8% — worse than 9D. Fixed dimensions (γ₁=0.012, γ_wt=0.198) may not be
globally optimal, limiting the 4D search.

---

## Lessons Learned

1. **Floor subtraction is physically correct but needs constraints**: The γ₁→0
   pathology is preventable by bounding γ₁ ≥ 0.04 and γ_wt ≤ 0.08.

2. **γ₇_dc played a functional role**: Even though unphysical (1/√batch), it provided
   a large per-step overhead floor that prevented under-prediction for decode scenarios.
   Need a physically correct replacement before retiring it.

3. **Structural gap remains**: The 22pp gap to iter29 is not from floor subtraction.
   The iter34 and iter33 models both plateau around 56-66%, suggesting the kernel-lookup
   basis functions need additional terms to match trained_physics accuracy.

---

## Next Steps (Iter35)

**Option B (recommended)**: Revert floor subtraction, fix γ₇_dc correctly.

- Change `γ₇_dc × L / √batch` to `γ₇_dc × L` (constant per-step per-layer overhead)
- This is physically sensible: a fixed scheduling cost per layer per step
- Keep all other iter33 improvements (prefix-cache fix, γ₇_pf)
- Warm-start from iter32 best (56.49%) with new γ₇_dc = constant
- Run CMA-ES + TPE + golden section
- Expected: should beat iter32 (56.49%) and approach iter33 (57.48%) or better
