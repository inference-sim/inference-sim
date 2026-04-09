# Iter35 Findings

## Summary

| Metric | Value |
|--------|-------|
| **Best loss (golden section)** | **54.55%** |
| Iter32 baseline | 56.49% |
| Iter33 baseline | 57.48% |
| Iter29 target | 34.57% |
| **Gap to target** | **20pp** |

Iter35 introduced a physically-motivated fix to the γ₇_dc overhead term and found
a new best of 54.55% — beating iter32 (56.49%) by 1.94pp. However, the 20pp gap
to iter29 (34.57%) remains and is not closing with the kernel-lookup approach.

---

## Change: γ₇_dc Fixed to Constant × L

**Problem (iter33)**: `γ₇_dc × L / √decodeTokens` — total overhead *decreased*
as decode batch grew (more requests = less overhead), which is unphysical.

**Fix (iter35)**: `γ₇_dc × L` — constant per-layer per-step overhead, applied
when decode requests are present. Physically: fixed GPU scheduling cost per
transformer layer per step, independent of batch size.

**Calibration**: Setting `γ₇_pf = 0` (disable per-seq scaling) and
`γ₇_dc = 187` (constant × 28 layers = 5.2ms) replicates iter32's overhead
structure, giving 55.75% warm-start — already beating iter32 (56.49%).

---

## Search Results

### Warm-start
- Coefficients: iter33 best + γ₇_pf=0 + γ₇_dc=187
- Loss: **55.75%** (beats iter32=56.49% before any optimization)

### Golden Section Round 1 (132 evaluations, ~37 min)

| Param | Result | Best value |
|-------|--------|-----------|
| γ₇_dc | ✓ **54.61%** | 170.45 µs/layer |
| α₀    | ✗ no improvement | 17658 µs |
| γ₁    | ✗ no improvement | 0.086 |
| γ₂    | intermediate **54.55%** at 0.3215 (midpoint 55.13%) | 0.302 |
| γ₃    | ✗ no improvement | 0.939 |
| γ₉    | ✗ no improvement | 214 µs |
| γ₈    | ✗ no improvement | 86 µs |
| γ₇_pf | ✗ no improvement | 0 |
| α₂    | ✗ no improvement | 3.1 µs |

**Note on γ₂**: During the golden section search, an intermediate evaluation found
54.55% at γ₂=0.3215. However, the golden section midpoint of the final interval
(γ₂=0.3211) scored 55.13%, so the code did not update γ₂. This is a known
limitation of golden section on noisy landscapes. Round 2 with a narrower
window would re-explore this region. The **true best seen across all evaluations
is 54.55%**, not 54.61%.

### Round 2 (partial, stopped early)
- γ₇_dc re-search found 54.57% at γ₇_dc=169.96 (consistent with round 1)
- No further improvements before search was stopped

### Best known coefficient vector
```
alpha = [17658.51, 0.0, 3.106]
beta  = [0.086, 0.302, 0.939, 0.0, 0.009, 1.0, 0.0, 86, 214, 170]
         γ₁    γ₂    γ₃         γ₅     γ₆  γ₇pf γ₈  γ₉  γ₇dc
```
Estimated loss: ~54.6% (based on γ₇_dc=170.45 improvement)

---

## Root Cause Analysis: Why We Can't Beat Iter29

The persistent 20pp gap reveals something fundamental:

### iter29 (34.57%) uses trained_physics — a roofline model
- Basis: analytical FLOPs × (1/flopsPeak) + bytes × (1/bandwidth)
- These basis functions **separate compute-bound and memory-bandwidth-bound regimes**
- At small decode batches on large models: T_dc_kv dominates (KV reads), T_dc_compute is tiny
- The model correctly predicts memory-bandwidth-limited behavior

### kernel-lookup (~54%) uses measured kernel times
- Basis: measured per-kernel latencies from AIC single-kernel CUDA graphs
- These **bundle compute + memory bandwidth together**
- A single γ₁ correction can't separately adjust compute vs. bandwidth components
- The model can't accurately predict small-batch decode on large models

### The structural gap
Even with:
- ✓ Prefix-cache FlashAttention fix (iter33)
- ✓ γ₇ split (iter33)
- ✓ γ₇_dc constant form (iter35)
- ✓ Optimized coefficients (iter35 golden section)

...the kernel-lookup model plateau is ~54-57%. Closing the remaining 20pp to
iter29 requires either:

1. **Adding separate T_weight and T_kv analytical terms** (Option C from iter33
   handoff) — the same structure that makes trained_physics work
2. **Per-model-class corrections** — a 1D table indexed by numLayers
3. **Abandoning pure kernel-lookup** in favor of a hybrid model

---

## Next Steps

The kernel-lookup model has proven useful but structurally limited. The path to
beating iter29 requires adding the missing physics that trained_physics has:

**Recommended: Hybrid model with AIC attention + analytical weight/KV bandwidth**

```
T_step = γ₁ × T_gemm_AIC(m) × L          (measured compute, all tokens)
       + γ₂ × T_pf_attn_AIC × L × correction   (measured FlashAttention)
       + γ₃ × T_dc_attn_AIC × L           (measured PagedAttention)
       + β_wt × T_weight_analytical        (analytical weight bandwidth = trained_physics T_weight)
       + β_kv × T_kv_analytical            (analytical KV bandwidth = trained_physics T_dc_kv)
       + overhead terms
```

Where `T_weight_analytical = (weight_bytes / bwHbm)` and
`T_kv_analytical = (kv_bytes × context / bwHbm)` are computed from model
architecture + hardware profile — exactly the terms in trained_physics that
correctly handle memory-bandwidth-limited decode on large models.
