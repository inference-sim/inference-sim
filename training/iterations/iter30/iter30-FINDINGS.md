# Iteration 30: Kernel-Lookup Backend -- Warm-Start Evaluation

## Summary

First evaluation of the kernel-lookup backend using fixed warm-start gamma coefficients
(no optimization). Purpose: establish baseline accuracy of aiconfigurator-measured
kernel data as basis functions before any coefficient fitting.

## Warm-Start Coefficients

| Coeff | Value | Role |
|---|---|---|
| gamma1 | 1.0 | prefill GEMM correction |
| gamma2 | 1.0 | prefill attention correction |
| gamma3 | 1.0 | decode GEMM correction |
| gamma4 | 1.0 | decode attention correction |
| gamma5 | 0.0 | weight loading (unused) |
| gamma6 | 1.0 | AllReduce correction |
| gamma7 | 0.0 | MoE overhead (disabled) |
| gamma8 | 40.0 | us/layer overhead |
| gamma9 | 3.0 | us/request overhead |
| gamma10 | 100.0 | us/step overhead |
| alpha0 | 0.0 | queueing overhead |
| alpha1 | 0.0 | post-decode overhead |
| alpha2 | 0.0 | per-token overhead |

## Results

Overall loss: 1619.92%
TTFT RMSE: 1543.32%
E2E RMSE: 76.60%
Succeeded: 15/15

## Per-Experiment Breakdown

| Experiment | Model | Workload | TTFT APE | E2E APE | Combined |
|---|---|---|---|---|---|
| 66-qwen2-5-7b-instruct-tp1-reasoning-lite-1-1 | Qwen/Qwen2.5-7B-Instruct | reasoning-lite-1-1 | 4520.87% | 67.54% | 4588.40% |
| 67-llama-2-7b-hf-tp1-reasoning-lite-1-1 | meta-llama/Llama-2-7b-hf | reasoning-lite-1-1 | 3259.63% | 75.87% | 3335.49% |
| 48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1 | Scout-17B-16E-FP8 | reasoning-lite-2-1 | 1474.90% | 85.88% | 1560.78% |
| 61-llama-3-1-70b-tp4-codegen-4-1 | meta-llama/Llama-3.1-70B-Instruct | codegen-4-1 | 897.06% | 65.61% | 962.67% |
| 20260217-162547-llama-2-7b-tp1-roleplay | meta-llama/Llama-2-7b-hf | roleplay | 869.70% | 68.43% | 938.14% |
| 21-llama-4-scout-17b-16e-tp2-roleplay-2 | Scout-17B-16E-FP8 | roleplay-2 | 602.89% | 81.16% | 684.05% |
| 20-llama-4-scout-17b-16e-tp2-codegen-2 | Scout-17B-16E-FP8 | codegen-2 | 491.92% | 83.26% | 575.18% |
| 64-qwen2-5-7b-instruct-tp1-roleplay-1-1 | Qwen/Qwen2.5-7B-Instruct | roleplay-1-1 | 372.23% | 73.00% | 445.23% |
| 62-mistral-nemo-12b-tp2-general-lite-2-1 | mistralai/Mistral-Nemo-Instruct-2407 | general-lite-2-1 | 319.12% | 58.97% | 378.10% |
| 63-mistral-nemo-12b-tp1-codegen-1-1 | mistralai/Mistral-Nemo-Instruct-2407 | codegen-1-1 | 205.05% | 77.70% | 282.75% |
| 20260217-231439-llama-2-7b-tp1-general | meta-llama/Llama-2-7b-hf | general | 157.39% | 85.85% | 243.25% |
| 65-01-ai-yi-34b-tp2-general-lite-2-1 | 01-ai/Yi-34B | general-lite-2-1 | 92.08% | 76.92% | 168.99% |
| 60-llama-3-1-70b-tp4-general-lite-4-1 | meta-llama/Llama-3.1-70B-Instruct | general-lite-4-1 | 58.49% | 74.25% | 132.74% |
| 20260217-155451-llama-2-7b-tp1-codegen | meta-llama/Llama-2-7b-hf | codegen | 51.74% | 80.86% | 132.61% |
| 17-llama-4-scout-17b-16e-tp2-general-lite-2-1 | Scout-17B-16E-FP8 | general-lite-2-1 | 17.76% | 87.42% | 105.18% |

## Analysis

### TTFT Errors (dominant error source)

The TTFT errors are extremely high for reasoning-lite workloads (1475--4521% APE) and
moderate-to-high for other workloads. This pattern suggests the kernel-lookup backend
is significantly underestimating prefill latency, particularly for workloads with
long input sequences (reasoning-lite uses ~2048 input tokens).

Key observations:
- **Reasoning-lite workloads dominate**: 3 experiments with >1000% TTFT APE are all reasoning-lite
- **General-lite workloads are best**: Scout general-lite at 17.76% TTFT APE suggests
  the kernel data is closest to reality for shorter prefill sequences
- **All TTFT errors are positive**: Simulator predicts much lower TTFT than ground truth,
  indicating the kernel-measured times underestimate real-world prefill latency under load

### E2E Errors (relatively stable)

E2E APE is in the 59--88% range across all experiments. This is more stable than TTFT
but still high. The E2E metric includes decode time which dominates for long-output
workloads, suggesting decode kernel lookup is also underestimating.

### Root Cause Hypotheses

1. **Missing batching effects**: Kernel profiles measure single-request latency, but
   under load, batching increases compute per step. The gamma corrections (all 1.0)
   need fitting to capture this gap.
2. **Missing scheduling overhead**: alpha coefficients are all 0.0, meaning no
   queueing delay or per-token overhead is modeled. Real servers have scheduling latency.
3. **KV cache pressure**: Under high load, KV cache management adds latency not
   captured by isolated kernel measurements.

## Comparison vs Iter29 (evolved model)

| Metric | Iter29 (evolved) | Iter30 (kernel-lookup warm-start) |
|---|---|---|
| Overall loss | 34.57% | 1619.92% |
| TTFT RMSE | -- | 1543.32% |
| E2E RMSE | -- | 76.60% |
| Succeeded | -- | 15/15 |

The warm-start kernel-lookup backend is 47x worse than the fully-optimized iter29 model.
This is expected: iter29 had 29 rounds of coefficient optimization while this is the
untuned baseline. The purpose of this evaluation is to establish the starting point
for gamma coefficient fitting in subsequent iterations.
