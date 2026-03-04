# H1 FINDINGS: Trace Replay Reduces TTFT Error

**Status:** SUPPORTED (with caveats)
**Date:** 2026-02-27

## Claim

Replaying ground-truth request arrival times and token lengths via BLIS's legacy trace CSV format reduces mean TTFT error from 31,906% to < 100%, confirming that the workload-spec generator is the dominant source of TTFT mismatch.

## Result

| Metric | Workload-Spec (R2) | Trace Replay (R3) | Reduction Factor |
|--------|--------------------|--------------------|------------------|
| Mean E2E error | 427.8% | 56.2% | **7.6x** |
| Mean TTFT error | 31,906% | 78.8% | **405x** |
| Mean ITL error | 33.6% | 9.5% | **3.5x** |
| E2E < 10% | 1/10 | 0/10 | — |

**Verdict:** SUPPORTED. Mean TTFT error dropped from 31,906% to 78.8% (< 100% threshold). However, E2E remains at 56.2%, indicating a secondary error source beyond workload specification.

## Per-Experiment Results

| Experiment | Model | Workload | GT E2E (ms) | BLIS E2E (ms) | E2E Error | TTFT Error | ITL Error |
|---|---|---|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | llama-2-7b | roleplay | 2071 | 810 | 60.9% | 77.9% | 4.0% |
| llama-2-70b-tp4-general | llama-2-70b | general | 5321 | 1986 | 62.7% | 87.9% | 23.5% |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | codegen | 4606 | 2022 | 56.1% | 77.4% | 10.7% |
| llama-2-70b-tp4-roleplay | llama-2-70b | roleplay | 4562 | 2019 | 55.7% | 77.2% | 10.3% |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | codegen | 4675 | 2234 | 52.2% | 76.2% | 2.1% |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | general | 5039 | 2253 | 55.3% | 79.6% | 8.7% |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | roleplay | 4685 | 2276 | 51.4% | 76.7% | 0.7% |
| codellama-34b-tp2-general | codellama-34b | general | 4093 | 1682 | 58.9% | 80.1% | 17.9% |
| codellama-34b-tp2-codegen | codellama-34b | codegen | 3723 | 1674 | 55.0% | 77.4% | 10.1% |
| codellama-34b-tp2-roleplay | codellama-34b | roleplay | 3670 | 1702 | 53.6% | 77.4% | 7.3% |

## Key Observations

1. **BLIS systematically under-predicts E2E by ~50-60%** across all experiments. BLIS E2E is consistently ~40% of ground truth. This is NOT random error; it's a systematic bias.

2. **TTFT is still 78.8% error even with trace replay.** BLIS TTFT predictions (6-14ms) are ~75-88% lower than ground truth (27-103ms). This means BLIS's scheduling simulation processes requests faster than real vLLM, producing artificially low TTFT.

3. **ITL improved to 9.5% mean** (from 33.6%), with 4/10 experiments below 10%. The per-token step time is reasonably accurate; the cumulative E2E error comes from queue/scheduling dynamics, not step-time prediction.

4. **All 10 experiments completed with correct request counts**, confirming the trace conversion is faithful.

## Interpretation

The 405x TTFT improvement confirms H1's core claim: the workload-spec generator was the dominant TTFT error source. However, the remaining 56% E2E error reveals a secondary problem: **BLIS's step-time model consistently under-predicts, causing requests to complete faster than reality.** With faster completion, queues drain faster, TTFT stays low, and E2E accumulates a systematic under-prediction bias.

The root cause is likely the `step_overhead_us` floor (3.9ms for llama-2-7b) being too low — real vLLM step cycles may include additional CPU overhead not captured by the current model.

## Refutation Criteria Assessment

- **SUPPORTED criteria met:** Mean TTFT error 78.8% < 100%. 10/10 experiments have TTFT < 100%.
- **Additional finding:** E2E error (56.2%) is much better than R2 (427.8%) but still far from <10% target.
