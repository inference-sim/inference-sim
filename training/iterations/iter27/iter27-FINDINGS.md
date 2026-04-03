# Iteration 27: Findings — CMA-ES Joint 6-Parameter Optimization

## Summary

CMA-ES joint optimization of 6 parameters (β₁ₐ, β₄, β₅, β₇, β₈, β₂ᵦ) from iter26 warm
start. Loss improved from **37.42% → 34.61%** (-2.81 points, 7.5% relative).

Best at trial 62 of 141. Patience stopped the run 79 trials after the last improvement.

## Key Coefficient Changes vs Iter26

| Coeff | Iter26 | Iter27 | Δ | Physical interpretation |
|---|---|---|---|---|
| β₁ₐ | 0.139 | 0.152 | +9% | Prefill compute — slight increase |
| β₄ | 0.410 | **0.752** | **+83%** | TP All-Reduce — higher than expected |
| β₅ | 49.6 | **32.4** | **-35%** | Per-layer — β₄ increase absorbed part of β₅ |
| β₇ | 169.4 | **126.0** | **-26%** | Per-step constant — reduced |
| β₈ | 427.3 | **505.5** | **+18%** | MoE interleaving overhead — increased |
| β₂ᵦ | 1.263 | **1.922** | **+52%** | Decode memory — significant increase |

The joint optimization found strong interactions between β₄↑ and β₅↓, β₇↓: activating
T_tp with β₄=0.752 (nearly 3× the NVLink/HBM ratio) allowed β₅ and β₇ to retreat,
suggesting they were previously compensating for unmodeled TP communication overhead.

## Results vs Iter26

| Metric | Iter26 | Iter27 | Improvement |
|---|---|---|---|
| Overall loss | 37.42% | **34.61%** | **-2.81** |
| TTFT RMSE | 24.34% | **22.81%** | -1.53 |
| E2E RMSE | 13.09% | **11.79%** | -1.30 |

## Per-Experiment Highlights

- **Mistral TP=2**: 27.4% → 20.0% TTFT — direct benefit from T_tp activation
- **Llama-3.1 TP=4 E2E**: 9.4% → 3.2% — major E2E improvement from T_tp
- **Scout reasoning-lite**: 60.3% → 59.2% TTFT — still the hardest experiment

## Notes

TPE was excluded this iteration due to SQLite race conditions with n_jobs>1 in Optuna.
TPE will be run in iter28 as a cross-check on the CMA-ES result.

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
| **27** | **34.61%** | **CMA-ES joint 6-param** |
