# H3 FINDINGS: LOMO Generalization (Leave-One-Model-Out)

**Status:** Refuted
**Date:** 2026-03-02

## Hypothesis

> Cycle-time regression achieves LOMO per-step MAPE < 80% per fold (train on 3 models, predict 4th).

**Refutation criteria:** Any fold MAPE > 150%.

## Results

| Holdout Model | MAPE | Pearson r | Test Samples | Status |
|---|---|---|---|---|
| CodeLlama-34B | 70.0% | 0.657 | 24,100 | PASS |
| Llama-2-70B | **295.5%** | 0.811 | 19,412 | FAIL |
| Llama-2-7B | 124.6% | 0.169 | 15,216 | FAIL |
| Mixtral-8x7B | **40.6%** | 0.575 | 19,088 | PASS |
| **MEAN** | **132.7%** | | | |

## Analysis

**REFUTED:** Llama-2-70B fold MAPE 295.5% > 150% threshold.

- **Mixtral generalizes best** (40.6% MAPE) — consistent with R2's finding that MoE architecture is the most workload-universal model
- **CodeLlama-34B passes** (70.0%) — benefits from interpolation between 7B and 70B training data
- **Llama-2-7B fails** (124.6%) — the 7B model has step durations 10-100x smaller than training models, making cross-model prediction structurally difficult
- **Llama-2-70B fails catastrophically** (295.5%) — although it has high Pearson r (0.811), the absolute scale mismatch causes huge MAPE

**Root cause:** FairBatching regression learns model-specific step-time scales. Without model-level normalization features (parameter count, FLOPS/token, TP degree), cross-model transfer is limited to models with similar step-time distributions.

**Comparison to R2:** LOMO improved from R1's 2,559.7% to R2's 108.6% via regime structure. Our 132.7% is similar to R2's regime ensemble — the FairBatching formulation adds no LOMO benefit over Ridge with regime structure.
