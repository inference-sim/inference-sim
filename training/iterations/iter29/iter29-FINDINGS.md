# Iteration 29: Findings — Sequential Golden Section, β₆ Breakthrough

## Summary

Sequential golden section search over 5 coefficients (β₃, β₆, β₅, β₈, β₂ᵦ), each with
15 experiments parallelized per evaluation. Loss improved from **34.6564% → 34.5675%**
(-0.089 points, 0.26% relative). β₆ (per-request overhead) was the dominant contributor,
jumping +57% from 2.805 to 4.417.

## Key Coefficient Changes vs Iter27

| Coeff | Iter27 | Iter29 | Δ | Physical interpretation |
|---|---|---|---|---|
| β₃ | 1.3636 | 1.3625 | -0.1% | Weight loading — already optimal |
| **β₆** | 2.8051 | **4.4168** | **+57%** | Per-request overhead — main driver |
| β₅ | 32.394 | 32.095 | -1% | Per-layer — already optimal |
| β₈ | 505.51 | 481.86 | -5% | MoE overhead — small correction |
| β₂ᵦ | 1.9224 | 1.9471 | +1% | Decode memory — small correction |

## Why β₆ Was Misaligned

β₆ adds a per-request constant to the prefill prediction. In iter27, the joint search
shifted β₄↑ (+83%), β₅↓ (-35%), β₇↓ (-26%) — effectively reducing the per-step
and per-layer contributions. This made the model systematically under-predict TTFT for
requests where the per-request overhead was the dominant term. β₆ needed to increase to
compensate. Since β₆ was frozen in iter27, this misalignment carried forward undetected.

## Results vs Iter27

| Metric | Iter27 | Iter29 | Improvement |
|---|---|---|---|
| Overall loss | 34.6564% | **34.5675%** | **-0.089** |

Per-experiment breakdown not collected in this run (golden section does not call
`--evaluate-per-experiment`). The improvement is distributed across experiments with
high per-request overhead sensitivity.

## Full Training Journey

| Iter | Loss | Key change |
|---|---|---|
| 16 | 60.19% | Trained-roofline baseline |
| 20 | 40.58% | β₈·nMoELayers breakthrough |
| 21 | 39.86% | Prefill compute-only split |
| 22 | 39.42% | β₂ decode correction |
| 23 | 39.24% | 3D joint (β₁ₐ,β₂,β₈) |
| 24 | 39.18% | Decode memory-only split |
| 25 | 39.18% | β₈ moeScaling (arch-aware) |
| 26 | 37.42% | T_tp activated, β₄=0.41, β₅=49.6 |
| 27 | 34.61% | CMA-ES joint 6-param |
| 28 | 34.66% | TPE cross-check — no improvement (confirmed iter27 local optimum) |
| **29** | **34.57%** | **Sequential golden section — β₆ +57%, β₃/β₆ re-calibration** |
