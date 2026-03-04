# H4 FINDINGS: LOWO Generalization (Leave-One-Workload-Out)

**Status:** Refuted
**Date:** 2026-03-02

## Hypothesis

> Cycle-time regression achieves LOWO per-step MAPE < 50% per fold (train on 2 workloads, predict 3rd).

**Refutation criteria:** Any fold MAPE > 100%.

## Results

| Holdout Workload | Global MAPE | Per-Model MAPE | Pearson r | Status |
|---|---|---|---|---|
| codegen | 143.8% | 92.0% | 0.705 | FAIL |
| general | 110.7% | 56.8% | 0.686 | FAIL |
| roleplay | 177.5% | **341.6%** | 0.293 | FAIL |
| **MEAN** | | **163.5%** | | |

### Per-Model Breakdown (Roleplay Fold — Worst)

| Model | MAPE | Samples |
|---|---|---|
| CodeLlama-34B | 110.0% | 8,443 |
| Llama-2-70B | **936.6%** | 6,741 |
| Mixtral-8x7B | 33.9% | 6,677 |

## Analysis

**REFUTED:** All folds exceed 50% target, roleplay fold MAPE 341.6% > 100% threshold.

- **Mixtral generalizes well across workloads** (25-34% per-model MAPE across all folds) — consistent with R2's finding of Mixtral LOWO 19.1%
- **Llama-2-70B is the worst offender** (90.9-936.6% across folds) — roleplay workload has fundamentally different step-time characteristics for 70B
- **"General" workload is easiest to predict** (56.8% weighted) — likely because it has the broadest input/output distribution coverage
- **"Roleplay" is hardest** (341.6%) — Llama-2-70B roleplay has a step-time distribution very different from codegen/general

**Root cause:** Workload type changes input/output length distributions, which changes batch composition patterns. Roleplay has shorter, burstier patterns that shift the overhead/compute balance point. The linear regression cannot capture these non-linear workload effects.

**Comparison to R2:** LOWO degraded from R2's 117.4% (all models) to 163.5% here. The FairBatching formulation with kv_sum feature does not improve cross-workload transfer — in fact it slightly hurts, likely due to kv_sum's workload-dependent distribution.
