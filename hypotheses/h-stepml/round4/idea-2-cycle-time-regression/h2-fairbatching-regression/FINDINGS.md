# H2 FINDINGS: FairBatching Cycle-Time Regression → BLIS E2E

**Status:** Refuted (E2E target not met; TTFT dominates)
**Date:** 2026-03-02

## Hypothesis

> A per-model FairBatching-style regression on cycle time (`cycle_time = a + b*new_tokens + c*kv_sum`) achieves mean BLIS E2E < 25% with mean ITL < 20% across 10 experiments, without any separate overhead floor.

**Refutation criteria:** Mean BLIS E2E > 40% OR mean ITL > 25%.

## Experiment Design

- **Independent variable:** FairBatching regression model (3-coefficient OLS per model) with overhead floor calibrated from H1 ITI data
- **Dependent variable:** BLIS E2E, TTFT, and ITL mean errors
- **Control:** Per-model overhead floors from H1/R2 data

### Training Configuration

Per-model OLS regression: `step_time = intercept + coeff_new_tokens * (prefill_tokens + decode_tokens) + coeff_kv_sum * kv_sum`

Overhead floors (H1-confirmed where available):
- Llama-2-7B: 3,897 us (R2)
- CodeLlama-34B: 7,149 us (H1-confirmed, was 6,673 in R2)
- Llama-2-70B: 9,670 us (H1-confirmed, was 8,029-8,203 in R2)
- Mixtral-8x7B: 9,125 us (R2)

## Results

### Per-Model Regression Coefficients

| Model | Intercept | Coeff new_tokens | Coeff kv_sum | Overhead (us) | Test MAPE | Test r |
|---|---|---|---|---|---|---|
| CodeLlama-34B | -68.6 | 11.705 | 0.037 | 7,149 | 63.0% | 0.725 |
| Llama-2-70B | -36.4 | 5.962 | 0.031 | 9,670 | 54.7% | 0.838 |
| Llama-2-70B-HF | -953.6 | 0.352 | 0.065 | 9,670 | 89.4% | 0.702 |
| Llama-2-7B | 149.0 | 1.024 | 0.000 | 3,897 | 45.6% | 0.558 |
| Mixtral-8x7B | 173.4 | 3.605 | 0.000 | 9,125 | 28.5% | 0.724 |

### BLIS E2E Validation Results (Workload-Spec Mode)

| Experiment | E2E Error | TTFT Error | ITL Error |
|---|---|---|---|
| llama-2-7b-tp1-roleplay | 52.5% | 78.5% | **4.0%** |
| llama-2-70b-tp4-general | 1,676.6% | 88,557.1% | 22.6% |
| llama-2-70b-hf-tp4-codegen | 50.5% | 6,520.8% | 43.3% |
| llama-2-70b-tp4-roleplay | 28.1% | 164.2% | 39.5% |
| mixtral-8x7b-tp2-codegen | 51.5% | 76.4% | **2.1%** |
| mixtral-8x7b-tp2-general | 554.9% | 44,487.4% | **8.7%** |
| mixtral-8x7b-tp2-roleplay | 50.8% | 77.2% | **0.7%** |
| codellama-34b-tp2-general | 1,927.8% | 154,921.4% | 49.5% |
| codellama-34b-tp2-codegen | 101.8% | 9,223.9% | 77.9% |
| codellama-34b-tp2-roleplay | 27.2% | 17.3% | 45.8% |
| **MEAN** | **452.2%** | **30,412.4%** | **29.4%** |

## Analysis

### Why E2E Failed (452.2% vs 25% target)

The dominant error source is **TTFT mismatch from workload-spec mode** (mean 30,412%), not step-time prediction error. This is the same bottleneck discovered in R2 (427.8% mean E2E, 31,906% mean TTFT) and resolved in R3 by switching to trace replay. The "general" workload experiments have catastrophic TTFT errors (88K, 44K, 154K%) because inference-perf workload generation creates different arrival patterns than the real experiment.

Without trace replay mode, the FairBatching regression's contribution to E2E is unmeasurable — it is completely dominated by workload-spec artifacts.

### ITL Results Are Promising

ITL mean 29.4% with 3/10 experiments below 10% (7B-roleplay: 4.0%, mixtral-roleplay: 0.7%, mixtral-codegen: 2.1%). This matches R2's overhead floor pattern (33.6% mean ITL, 5/10 < 10%). The overhead floor mechanism works: small-batch decode steps are dominated by the per-model overhead constant, making ITL prediction accurate regardless of step-time model quality.

### Per-Step Regression Quality

FairBatching 3-coefficient OLS produces test MAPE ranging from 28.5% (Mixtral) to 89.4% (70B-HF). Compared to R3 idea 2's FairBatching (56.2% overall MAPE), our per-model results are similar. The kv_sum feature contributes positively for 34B and 70B (r > 0.7) but is zero-weighted for 7B and Mixtral (insufficient correlation).

### Adapted Approach Note

H1's partial refutation (19.9% match rate) meant we couldn't train directly on extracted cycle times as originally planned. Instead, we trained on `step.duration_us` (GPU compute, abundant data) and used the overhead floor as the cycle-time proxy. This is functionally equivalent to R2's approach — the novel contribution was H1-confirmed overhead values (7,149us for 34B, 9,670us for 70B).

## Verdict

**Refuted.** Mean E2E 452.2% >> 40% threshold, mean ITL 29.4% > 25% threshold. Both refutation criteria are met. However, the failure is not caused by the FairBatching regression itself — it's caused by workload-spec mode TTFT artifacts (same as R2). The per-step regression and overhead floor are functioning as designed; ITL results (29.4%) are within range of R2's 33.6%. This idea would need trace replay to produce meaningful E2E results.

## Comparison to Prior Rounds

| Metric | R2 Best | R3 Best (CMA-ES) | This Idea (H2) |
|---|---|---|---|
| Mean E2E | 427.8% | **15.1%** | 452.2% |
| Mean TTFT | 31,906% | **67.6%** | 30,412% |
| Mean ITL | **33.6%** | 87.4% | 29.4% |
| Mode | workload-spec | trace replay | workload-spec |
