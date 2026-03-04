# FINDINGS: H3 — BLIS E2E Validation + 34B Investigation

**Date:** 2026-02-27
**Status:** REFUTED (no ITL improvement; 34B root cause identified)

## Claim
The best step-time formulation from H1/H2, combined with overhead floor, achieves < 20% mean ITL error on BLIS E2E validation (improving from Round 2's 33.6%), and the CodeLlama-34B anomaly root cause is identified.

## BLIS E2E Results — Trace Replay Mode

| Experiment | Model | Workload | GT E2E (ms) | BLIS E2E (ms) | E2E Error | TTFT Error | ITL Error |
|---|---|---|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | llama-2-7b | roleplay | 2,071 | 810 | 60.9% | 78.6% | **4.0%** |
| llama-2-70b-tp4-general | llama-2-70b | general | 5,321 | 1,986 | 62.7% | 88.3% | 23.5% |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | codegen | 4,606 | 2,022 | 56.1% | 77.9% | 10.7% |
| llama-2-70b-tp4-roleplay | llama-2-70b | roleplay | 4,562 | 2,019 | 55.7% | 78.0% | 10.3% |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | codegen | 4,675 | 2,234 | 52.2% | 76.8% | **2.1%** |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | general | 5,039 | 2,253 | 55.3% | 80.1% | 8.7% |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | roleplay | 4,685 | 2,276 | 51.4% | 77.3% | **0.7%** |
| codellama-34b-tp2-general | codellama-34b | general | 4,093 | 1,682 | 58.9% | 80.6% | 17.9% |
| codellama-34b-tp2-codegen | codellama-34b | codegen | 3,723 | 1,674 | 55.0% | 78.1% | 10.1% |
| codellama-34b-tp2-roleplay | codellama-34b | roleplay | 3,670 | 1,702 | 53.6% | 78.1% | 7.3% |
| **MEAN** | | | | | **56.2%** | **79.4%** | **9.5%** |

- **E2E < 10%:** 0/10
- **ITL < 10%:** 5/10 (same as Round 2 idea-1 trace replay)

## Assessment

**Refuted.** The total-context model produces identical BLIS E2E results to Round 2's regime ensemble (idea-1 trace replay: 56.2% E2E, 9.5% ITL). The kv_sum coefficient has **zero impact** on BLIS E2E because:

1. **Overhead floor dominates.** For 70-90% of steps (decode-only), the overhead floor (~4-9ms) is applied. The predicted GPU compute (~100-500µs) plus kv_sum contribution (~0-200µs) is far below the floor and gets clamped. The kv_sum coefficient only affects the 10-30% of mixed steps.

2. **Mixed-step kv_sum contribution is small.** Even for mixed steps where the floor doesn't dominate, the kv_sum coefficient (0.002-0.065) on typical kv_sum values (~20,000) adds ~40-1300µs. But the step overhead floor is already 4,000-9,000µs, so this addition is within BLIS's step-cycle time and doesn't significantly change E2E.

3. **E2E is dominated by step count × overhead floor.** The 56.2% E2E under-prediction comes from BLIS simulating steps that take ~4-9ms versus real steps taking ~8-15ms. Improving the GPU compute prediction by a few hundred microseconds doesn't close this gap.

## 34B Deep-Dive Analysis

### Step-Time Distribution Comparison

| Model | Steps (test) | Decode-Only % | Mean Duration (µs) | P50 (µs) | P99 (µs) | KV Sum Mean |
|---|---|---|---|---|---|---|
| llama-2-7b_tp1 | 3,043 | 90.8% | 177 | 147 | 1,383 | 0 |
| mixtral-8x7b-v0-1_tp2 | 3,819 | 66.2% | 468 | 397 | 1,070 | 0 |
| codellama-34b_tp2 | 4,819 | 70.7% | 1,603 | 291 | 11,928 | 18,856 |
| llama-2-70b-hf_tp4 | 1,336 | 69.2% | 1,818 | 1,676 | 5,375 | 39,169 |
| llama-2-70b_tp4 | 2,547 | 64.2% | 2,978 | 1,132 | 10,401 | 41,641 |

### 34B Root Cause

**CodeLlama-34B is NOT anomalous in trace replay mode.** With trace replay, 34B's E2E errors (53.6-58.9%) are in the same range as other models (51.4-62.7%). The Round 2 "anomaly" (99.2% per-step MAPE, 2,901% E2E for general) was caused by the workload specification generator, not the step-time model.

However, 34B has distinctive characteristics:
1. **Bimodal step-time distribution.** P50=291µs but mean=1,603µs — the mean is pulled up by heavy-tail mixed steps (p99=11,928µs). This is the most bimodal of all models.
2. **Lower decode-only percentage** (70.7%) than 7b (90.8%) — more mixed batches → more variable step times.
3. **Moderate KV sum** (18,856 mean vs 40,000+ for 70b) — 34B has shorter context lengths per request.

### KV Data Quality Issue

Two models (llama-2-7b_tp1, mixtral-8x7b-v0-1_tp2) have kv_sum=0 for ALL steps. This means the lifecycle extractor found no active requests overlapping the step time windows. Likely cause: these experiments have only 1 experiment each (roleplay) with potentially different timing characteristics, or the step timestamps in traces.json don't overlap with lifecycle timestamps in per_request_lifecycle_metrics.json. This limits the kv_sum feature's coverage to 3/5 models.
