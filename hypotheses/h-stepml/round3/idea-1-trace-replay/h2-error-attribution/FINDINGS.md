# H2 FINDINGS: Error Attribution with Trace Replay

**Status:** REFUTED
**Date:** 2026-02-27

## Claim

With trace replay eliminating workload-spec error, the remaining E2E error is < 25% mean, attributable primarily to step-time prediction inaccuracy rather than simulation-level mismatch.

## Result

| Metric | Value | Target |
|--------|-------|--------|
| Mean E2E error | 56.2% | < 25% |
| E2E < 10% | 0/10 | >= 3/10 |
| E2E < 25% | 0/10 | — |
| Mean TTFT error | 78.8% | — |
| Mean ITL error | 9.5% | — |

**Verdict:** REFUTED. Mean E2E error (56.2%) exceeds the 25% threshold. Zero experiments achieve < 10% or even < 25%.

## Error Decomposition

| Experiment | E2E Error | TTFT Error | ITL Error | TTFT Abs Error (ms) | ITL Abs Error (ms) | Dominant Error Source |
|---|---|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | 60.9% | 77.9% | 4.0% | 21.1 | 0.16 | TTFT (queue drain) |
| llama-2-70b-tp4-general | 62.7% | 87.9% | 23.5% | 90.6 | 2.47 | TTFT + ITL |
| llama-2-70b-hf-tp4-codegen | 56.1% | 77.4% | 10.7% | 43.0 | 0.99 | TTFT |
| llama-2-70b-tp4-roleplay | 55.7% | 77.2% | 10.3% | 42.3 | 0.93 | TTFT |
| mixtral-8x7b-v0-1-codegen | 52.2% | 76.2% | 2.1% | 44.8 | 0.20 | TTFT (queue drain) |
| mixtral-8x7b-v0-1-general | 55.3% | 79.6% | 8.7% | 54.9 | 0.87 | TTFT |
| mixtral-8x7b-v0-1-roleplay | 51.4% | 76.7% | 0.7% | 46.4 | 0.07 | TTFT (queue drain) |
| codellama-34b-tp2-general | 58.9% | 80.1% | 17.9% | 41.3 | 1.46 | TTFT + ITL |
| codellama-34b-tp2-codegen | 55.0% | 77.4% | 10.1% | 35.3 | 0.75 | TTFT |
| codellama-34b-tp2-roleplay | 53.6% | 77.4% | 7.3% | 35.4 | 0.53 | TTFT |

## Key Findings

### 1. TTFT Dominates E2E Error (Not ITL)

Even with correct arrival times from trace replay, BLIS produces TTFT values 75-88% below ground truth. This means BLIS's internal scheduling/batch-formation processes requests faster than real vLLM, causing:
- Requests enter the batch sooner (lower TTFT)
- Queues drain faster (lower queueing)
- Less contention = faster E2E

### 2. ITL Is Nearly Solved

ITL error is 9.5% mean with trace replay, and 4 experiments achieve < 10%. The per-token step-time model (regime ensemble + overhead floor) produces accurate inter-token timing. The remaining ITL error is concentrated in larger models (llama-2-70b, codellama-34b) where batch sizes and KV cache pressures are higher.

### 3. The "Step-Time Improvement Headroom" Is Small

The HYPOTHESIS.md defined "step-time improvement headroom" = E2E_error - ITL_error. This quantity is large (56.2% - 9.5% = 46.7pp), but it's NOT because step-time is wrong. It's because **BLIS processes the workload too fast overall**, creating a compounding queue-drain effect where each step's slight under-prediction cascades through the entire simulation.

### 4. Root Cause: Systematic Under-Prediction of Step Cycle Time

BLIS E2E is consistently ~40% of ground truth across ALL experiments. This uniformity rules out per-model or per-workload effects. The root cause is that the step-time model (regime ensemble + 3.9ms overhead floor) predicts steps that are ~50-60% faster than reality. This could be due to:

- **Overhead floor too low:** The 3.9ms floor was calibrated on step.duration_us (GPU compute time only). Real vLLM step cycles include additional CPU scheduling overhead (~1-3ms) that occurs between steps.
- **Missing queuing effects in BLIS:** Real vLLM experiences memory allocation delays, KV block management overhead, and scheduler decision time that BLIS's work-conserving model doesn't capture.
- **Batch formation differences:** BLIS's FCFS batch formation may produce different batch compositions than vLLM's actual scheduler, leading to different step durations.

## Refutation Criteria Assessment

- **Refutation criteria met:** Mean E2E error 56.2% > 50% even with trace replay.
- **Co-dominant sources confirmed:** Both scheduling divergence (TTFT under-prediction) and step-time under-prediction contribute to E2E error. BLIS scheduling is NOT a faithful reproduction of vLLM scheduling.
