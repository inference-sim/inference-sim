# H3 FINDINGS: Workload-Spec Parameter Diagnosis

**Status:** REFUTED (unexpected finding)
**Date:** 2026-02-27

## Claim

The workload-spec generator's dominant error source can be identified as one of: (a) arrival rate mismatch, (b) token length distribution mismatch, or (c) horizon/duration mismatch.

## Result

| Experiment | Rate Error | Count Error | Duration Error |
|---|---|---|---|
| llama-2-7b-tp1-roleplay | 0.01% | 0% | 0.03% |
| llama-2-70b-tp4-general | 0.59% | 0% | 0.58% |
| llama-2-70b-hf-tp4-codegen | 0.52% | 0% | 0.51% |
| llama-2-70b-tp4-roleplay | 0.01% | 0% | 0.01% |
| mixtral-8x7b-v0-1-tp2-codegen | 0.49% | 0% | 0.48% |
| mixtral-8x7b-v0-1-tp2-general | 0.57% | 0% | 0.56% |
| mixtral-8x7b-v0-1-tp2-roleplay | 0.00% | 0% | 0.01% |
| codellama-34b-tp2-general | 0.50% | 0% | 0.49% |
| codellama-34b-tp2-codegen | 0.50% | 0% | 0.48% |
| codellama-34b-tp2-roleplay | 0.00% | 0% | 0.02% |

**All parameter errors are < 1%.** The workload-spec parameters (arrival rate, request count, duration) match ground truth almost exactly.

## Verdict

REFUTED. No single workload-spec parameter explains the 31,906% TTFT error in Round 2. The parameters themselves are nearly identical to ground truth.

## Key Insight

The workload-spec *parameters* are correct, but the *generation process* is the error source. Specifically:

1. **Arrival rate and count match perfectly.** The inference-perf profile.yaml stages produce the correct average rate and total request count.

2. **The error is in request-level sampling.** The workload-spec generator samples token lengths from distributions (shared_prefix + question_len parameters) rather than replaying exact per-request values. While the *mean* token length is close, the *distribution shape* and *per-request correlation* differ from reality.

3. **Token length distribution mismatch analysis:**

| Experiment | GT Mean Input | Spec Question Len | GT Mean Output | Spec Output Len |
|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | 786 | 600 | 206 | 251 |
| llama-2-70b-tp4-general | 575 | 447 | 246 | 248 |
| llama-2-70b-hf-tp4-codegen | 593 | 466 | 245 | 247 |
| llama-2-70b-tp4-roleplay | 787 | 600 | 250 | 251 |
| mixtral-codegen | 589 | 466 | 243 | 247 |
| mixtral-general | 571 | 447 | 245 | 248 |
| mixtral-roleplay | 782 | 600 | 248 | 251 |
| codellama-general | 574 | 447 | 250 | 248 |
| codellama-codegen | 593 | 466 | 249 | 247 |
| codellama-roleplay | 787 | 600 | 254 | 251 |

The `question_len` parameter consistently underestimates GT input tokens by 20-30% for roleplay (600 vs 786) and by ~22% for general/codegen (447-466 vs 571-593). This is because the inference-perf shared_prefix mechanism adds a system_prompt_len (100 tokens) plus a question_len, but the total doesn't match the full ground-truth input length.

4. **However, this doesn't explain the 31,906% TTFT.** Even a 20-30% input length mismatch cannot cause a 31,906% TTFT error. The catastrophic TTFT divergence in Round 2 must have been caused by BLIS's internal workload generation process interacting with the simulator's scheduling in a pathological way — likely creating bursty arrival patterns or mismatched request sizes that caused queue buildup.

## Why Trace Replay Helps

Trace replay bypasses the workload-spec generation entirely, providing:
- **Exact per-request arrival times** (not sampled from a Poisson process)
- **Exact per-request token lengths** (not sampled from distributions)
- **Correct temporal ordering** (preserving any correlation between arrival times and request sizes)

This eliminates the generation-level error (31,906% TTFT → 78.8% TTFT) while revealing the simulation-level error (56.2% E2E under-prediction).

## Conclusion

The workload-spec *parameters* are not wrong. The error is in the *generation process* — how parameters are converted to individual requests. This is a workload-generation fidelity problem, not a parameter calibration problem. Fixing it requires either:
1. Always using trace replay for validation (simple, already works)
2. Improving the workload-spec generator to better reproduce ground-truth request-level statistics (complex, low priority)
