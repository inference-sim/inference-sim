# Idea 2, H2: Regime Ensemble Achieves <10% BLIS E2E Mean Error

**Status:** Pending
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-26

## Hypothesis

> The full Idea 2 system -- h1's 3-regime StepTime ensemble combined with data-calibrated secondary LatencyModel methods (QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime) -- achieves <10% BLIS workload-level E2E mean error across all 16 experiments (4 models x 4 workloads on H100 GPUs).

This is the primary success criterion for Idea 2 and the P1 metric for the WP4 leaderboard.

## Refuted-If

- BLIS E2E mean error > 15% across all 16 experiments
- OR BLIS E2E mean error > 25% on ANY single experiment
- OR E2E error is worse than the blackbox baseline (115%) on ANY experiment
- OR TTFT mean error > 30% (indicating StepTime or scheduling calibration failure)
- OR ITL mean error > 40% (indicating decode step prediction failure)

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. **StepTime:** Use h1's 3-regime Ridge ensemble (12 regressors)
2. **Secondary methods calibration:**
   - **QueueingTime:** Per-model constant `q0` -- median queueing delay from lifecycle data (time between arrival and first step scheduling)
   - **OutputTokenProcessingTime:** Per-model constant `o0` -- median per-token output overhead from lifecycle data
   - **SchedulingProcessingTime:** Per-model constant `s0` -- median scheduling overhead from trace data (time between batch formation and step execution start)
   - **PreemptionProcessingTime:** Per-model constant `p0` -- median preemption cost from KV event data (time spent on KV cache eviction/reload)
3. **BLIS validation:** Run `validate_blis.py --mode stepml` on all 16 experiments
4. **Comparison:** Report E2E, TTFT, ITL vs. blackbox baseline and per-step-only StepTime model

**Data:** 16 ground-truth experiments replayed through BLIS with the candidate StepML model
**Split:** For secondary method constants, use the same temporal 60% training split as h1 (no new data). BLIS validation covers all 16 experiments (train + eval workloads).

**Baselines:**
- Blackbox per-model linear regression: 115% E2E mean error
- StepTime-only model (h1 StepTime + zero secondary methods): expected ~80-90% E2E error
- Per-step MAPE (diagnostic from h1)

**Success metric:** BLIS E2E mean error < 10% across all 16 experiments

## BLIS E2E Claim

**Primary:** Mean |E2E_predicted - E2E_actual| / E2E_actual < 10% across 16 experiments.
**Secondary:** E2E error < 15% on at least 12/16 experiments.
**Diagnostic:** Report per-experiment breakdown (E2E, TTFT, ITL) to identify model-workload combinations that remain challenging.

## Related Work

- **MIST** (arXiv:2504.09775): 2.5% per-step error translates to accurate simulation-level predictions in their evaluation. Sets the accuracy target for step-level models.
- **Vidur** (MLSys 2024): <5% per-operator error when validated against real vLLM. Demonstrates that per-component accuracy translates to system-level fidelity.
- **LIFE** (arXiv:2508.00904): Treats dispatch overhead separately from computation -- analogous to non-zero SchedulingProcessingTime calibration.
- **Round 1 findings**: Per-step models never validated through BLIS. This is the first BLIS E2E validation of a data-driven step-time model.

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Secondary method constants are inaccurate | Medium | Validate individually: compare QueueingTime constant vs. observed queueing delays distribution. If median is poor, try trimmed mean or workload-specific constants. |
| Step-level accuracy doesn't translate to E2E | Medium | This is a fundamental research question. Per-step MAPE and E2E error are both reported; their relationship is itself a finding. |
| "General" workload has high error (most diverse batches) | High | Report per-workload breakdown. If general exceeds 25% E2E, consider workload-specific adjustments or more expressive model for mixed-heavy regime. |
| Reasoning workload dominates errors (long sequences) | Medium | Reasoning experiments have very long token counts. Check if ProgressIndex-derived KV features correctly handle very long contexts (>4K tokens). |

## Go Integration Path

Same as h1's path with additional constants for secondary methods in the `StepMLArtifact` JSON:
```json
{
  "queueing_time_constant": 150.0,
  "output_token_processing_constant": 25.0,
  "scheduling_processing_constant": 80.0,
  "preemption_processing_constant": 0.0,
  "regimes": [...]
}
```
`StepMLLatencyModel` reads these constants in its constructor and returns them from the corresponding methods.

## Training Strategy and Data Split

- **StepTime:** h1's 60% temporal training split, pooled per model (4 workloads)
- **Secondary methods:** Median constants from same 60% training split of lifecycle data
- **BLIS validation:** All 16 experiments (includes training workloads) -- this is acceptable because the metric is E2E error through the full simulation pipeline, not per-step test accuracy
- **Leave-one-workload-out (LOWO):** Also report LOWO E2E error to assess generalization to unseen workload types

## Data Integrity

- StepTime and secondary method constants trained on non-overlapping temporal split
- BLIS validation uses the official harness (`validate_blis.py`) -- identical to how the blackbox baseline was measured
- No model selection on BLIS E2E (the model is fixed from h1; this sub-hypothesis only evaluates)
- No roofline predictions used (BC-3-7)
