# H31: CrossModel Generalization to Near-Saturation Reasoning

**Status**: Refuted
**Date**: 2026-03-03
**Control**: Iter 2 per-model blackbox (same result — not a crossmodel-specific issue)

## Hypothesis

> For codellama-34b-reasoning (0.1% real failure), BLIS will complete ≥90% of requests with TTFT MAPE < 30% and E2E MAPE < 25%.

## Results

### Primary: codellama-34b-reasoning

| Metric | Blackbox | CrossModel | Real vLLM |
|--------|----------|------------|-----------|
| Completed | 4,720/4,800 | 4,720/4,800 | 4,796/4,800 |
| TTFT mean | 45.3 ms | 64.2 ms | **120,171 ms** |
| E2E mean | 25,208 ms | 27,719 ms | **158,991 ms** |
| Throughput | 3.94 rps | 3.93 rps | **3.22 rps** |

### Root Cause: Utilization Regime Mismatch

| | Real vLLM | BLIS (both backends) |
|-|-----------|---------------------|
| Throughput (µ) | 3.22 rps | 3.93 rps |
| Utilization (ρ = λ/µ) | **124%** | **102%** |
| Queue regime | Heavily overloaded | Barely saturated |
| TTFT mean | 120 seconds | 45-64 ms |

Both backends produce identical throughput (3.93 rps) and completion rates (~98%). The 22% throughput overestimate shifts ρ from 124% → 102%, collapsing the queue from 120s TTFT to ~50ms.

### Informational: Other reasoning experiments

| Experiment | Backend | Throughput | Real rps | Completion |
|-----------|---------|------------|----------|------------|
| llama-2-7b-reasoning | BB | 0.08 rps | 0.49 rps | 85% fail both |
| llama-2-70b-reasoning | BB | 3.87 rps | 2.13 rps | BB completes ~all, real fails 33% |
| mixtral-8x7b-reasoning | Both | 3.93 rps | 1.00 rps | BB/CM complete ~all, real fails 69% |

## Verdict

**Refuted.** Completion rate technically passes (98% ≥ 90%), but TTFT/E2E fail by 2000×. The three-way comparison confirms this is a BLIS scheduling model issue (both backends produce the same result), not a crossmodel coefficient issue. The 22% throughput overestimate at ρ>1.0 collapses queueing dynamics.

## Implication

BLIS should not be used for saturation-regime analysis without calibrating the scheduler model. The crossmodel coefficients are not the bottleneck — the scheduling model's systematic speed advantage over real vLLM is.
