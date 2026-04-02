# Iteration 21: Findings — Prefill Compute/Memory Split

## Summary

Split the prefill roofline term from `β₁·max(T_compute, T_kv)` into `β₁ₐ·T_compute + β₁ᵦ·T_kv`
to allow independent correction of compute (FlashAttention-discounted) and memory bandwidth.

**Result**: Loss improved from 40.58% → **39.84%** (0.74-point improvement). The improvement
came entirely from adjusting β₁ₐ from 0.20 → 0.15. **β₁ᵦ is nearly irrelevant** — the top 15
results span β₁ᵦ = 0.05 to 2.00 with loss variation < 0.12%.

**Key discovery**: Prefill is **firmly compute-dominated** across all 15 experiments.
`T_pf_compute >> T_pf_kv`, so `max(compute, kv) ≈ compute` and the memory term contributes
negligibly. The split did not unlock a structural breakthrough — the improvement is a 1D
refinement of the compute correction coefficient.

## Best Coefficients

| Coeff | Description | Value | Change from iter20 |
|---|---|---|---|
| **β₁ₐ** | **Prefill COMPUTE correction** | **0.116** | **was β₁=0.201 (max mode); 8.6× discount** |
| **β₁ᵦ** | **Prefill MEMORY correction** | **0.0 (dropped)** | **memory term negligible** |
| β₂ | Decode correction | 1.611 | unchanged |
| β₃–β₇ | (all others) | (iter20 values) | unchanged |
| β₈ | Per-MoE-layer overhead | 440.0 | unchanged |

**Final model**: β₁ₐ=0.116 with β₁ᵦ=0 found via golden section search (11 evaluations,
tolerance 0.005). Dropping the memory term entirely costs only 0.05 loss points vs the
best split (39.86% vs 39.84%).

## 2D Grid Search Results

144 points evaluated (12 × 12 grid, 12 parallel workers).

**β₁ₐ sensitivity** (strong): Loss drops from ~42% at β₁ₐ=0.02 to ~39.8% at β₁ₐ=0.12–0.15,
then rises again beyond 0.20. Clear optimum at β₁ₐ ≈ 0.15.

**β₁ᵦ sensitivity** (negligible): At β₁ₐ=0.15, loss varies < 0.12% across the full β₁ᵦ
range [0.02, 2.00]. The prefill memory term is swamped by the compute term.

Full grid saved in `iter21_2d_grid.csv`.

## Comparison

| Iteration | Loss | Key change |
|---|---|---|
| iter19 | 60.11% | 7-term, best β₁-β₇ |
| iter20 | 40.58% | +β₈·nMoELayers (440µs) |
| **iter21** | **39.86%** | Prefill compute-only: β₁ₐ=0.116, β₁ᵦ=0 |

## Code Change

`sim/latency/evolved_model.go`: When 9 betas provided, `prefillSplit=true` activates
`β₁ₐ·T_pf_compute + β₁ᵦ·T_pf_kv` instead of `β₁·max(T_pf_compute, T_pf_kv)`.
Backward compatible: 7-8 betas use max mode as before.
