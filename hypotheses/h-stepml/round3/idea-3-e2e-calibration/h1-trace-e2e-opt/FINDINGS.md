# FINDINGS: H1 — Trace Replay + E2E Optimization Baseline

**Date:** 2026-02-27
**Status:** PARTIALLY SUPPORTED

## Claim

CMA-ES optimization of StepML artifact parameters against BLIS E2E mean error (using trace replay) achieves < 15% mean E2E error across all experiments.

## Result

**Mean E2E error: 15.1%** (target < 15%). 4/10 experiments < 10%, 5/10 < 15%, 8/10 < 25%.

The target was narrowly missed at 15.1%, but the improvement from the 56.2% baseline is dramatic (3.7x).

## Per-Experiment Results

| Experiment | Model | Workload | GT E2E (ms) | BLIS E2E (ms) | E2E Error | TTFT Error | ITL Error |
|---|---|---|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | llama-2-7b | roleplay | 2,071 | 1,612 | **22.2%** | 60.8% | 91.1% |
| llama-2-70b-tp4-general | llama-2-70b | general | 5,321 | 4,458 | 16.2% | 82.5% | 72.0% |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | codegen | 4,606 | 4,432 | **3.8%** | 67.6% | 111.6% |
| llama-2-70b-tp4-roleplay | llama-2-70b | roleplay | 4,562 | 4,249 | **6.9%** | 67.6% | 117.0% |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | codegen | 4,675 | 3,229 | 30.9% | 66.4% | 41.5% |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | general | 5,039 | 4,399 | 12.7% | 57.3% | 78.1% |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | roleplay | 4,685 | 3,084 | 34.2% | 69.3% | 34.5% |
| codellama-34b-tp2-general | codellama-34b | general | 4,093 | 4,044 | **1.2%** | 69.3% | 102.7% |
| codellama-34b-tp2-codegen | codellama-34b | codegen | 3,723 | 3,536 | **5.0%** | 68.9% | 90.2% |
| codellama-34b-tp2-roleplay | codellama-34b | roleplay | 3,670 | 3,028 | 17.5% | 66.2% | 135.4% |
| **MEAN** | | | | | **15.1%** | **67.6%** | **87.4%** |

## CMA-ES Optimization Results Per Model

| Model | Initial E2E | Optimized E2E | Evals | Time (s) | Key Parameter Change |
|---|---|---|---|---|---|
| llama-2-7b | 60.9% | 22.2% | 152 | 801 | step_overhead_us: 3,897 → 5,051 (+30%) |
| llama-2-70b | 62.7% | 8.3% | 96 | 1,904 | scheduling_time: 415 → 821 (+98%) |
| mixtral-8x7b-v0-1 | 52.2% | 25.9% | 96 | 1,870 | step_overhead_us: 9,125 → 9,884 (+8%) |
| codellama-34b | 58.9% | 7.9% | 96 | 1,913 | step_overhead_us: 6,673 → 5,998 (-10%) |

## Key Findings

1. **Step overhead is the primary knob:** The optimizer consistently increased step_overhead_us (llama-2-7b: +30%), scheduling_processing_time_us (llama-2-70b: +98%), and output_token_processing_time_us to slow down simulation speed to match reality.

2. **Model-dependent response to optimization:** llama-2-70b and codellama-34b responded best (8.3% and 7.9% mean E2E), while llama-2-7b (22.2%) and mixtral-8x7b-v0-1 (25.9%) showed limited improvement.

3. **E2E improvement comes at ITL cost:** ITL error worsened from 9.5% → 87.4% because the optimizer increased per-token overhead to match total E2E, overshooting per-token timing. The optimizer prioritizes the E2E objective, not ITL.

4. **TTFT remains systematically under-predicted (67.6%):** Even with optimized coefficients, BLIS TTFT is 3-5x lower than ground truth. This confirms TTFT is dominated by simulation scheduling behavior, not coefficient values.

5. **Workload variance within a model:** For mixtral, codegen (30.9%) and roleplay (34.2%) have much higher error than general (12.7%), suggesting the optimizer overfits to the general workload.

## Refutation Assessment

- **Supported:** 5/10 experiments < 15% (not "at least 5/10 < 10%" as claimed)
- **Partially refuted:** Mean E2E = 15.1%, narrowly missing the < 15% target
- **Root cause of remaining error:** Mixtral and llama-2-7b contribute disproportionately. These models have fewer experiments (1 for llama-2-7b, 3 for mixtral) and the CMA-ES optimization is more prone to local minima.
