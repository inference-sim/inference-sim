# Idea 2, H3: Non-Zero Secondary Method Calibration Improves BLIS E2E by >= 5 Percentage Points

**Status:** Refuted
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-26

## Hypothesis

> Calibrating the 4 secondary LatencyModel methods (QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime) with non-zero per-model constants improves BLIS E2E mean error by at least 5 percentage points compared to a StepTime-only model (which returns 0 for all secondary methods).

This sub-hypothesis isolates the contribution of secondary method calibration -- providing "free" headroom from methods that currently return 0. Even if StepTime is imperfect, non-zero secondary methods should improve E2E by correcting the systematic underestimation of non-step latency components.

## Refuted-If

- E2E improvement from secondary methods < 3 percentage points (below noise threshold)
- OR secondary methods make E2E error WORSE (indicating that constants introduce more error than they correct)
- OR the improvement is concentrated in only 1 of the 4 model configurations (not generalizable)

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** Ablation study. Compare two BLIS configurations:
- **Config A (StepTime-only):** h1's StepTime model + all 4 secondary methods return 0
- **Config B (Full model):** h1's StepTime model + calibrated secondary method constants

Run both configurations through `validate_blis.py` on all 16 experiments. Report the E2E, TTFT, and ITL difference.

**Secondary method derivation:**
- QueueingTime: per-model median from lifecycle data (request arrival to first step)
- OutputTokenProcessingTime: per-model median from lifecycle data
- SchedulingProcessingTime: per-model median from trace timing data
- PreemptionProcessingTime: per-model median from KV event data

**Data:** Same 16 experiments. Secondary method constants derived from 60% training split.
**Baselines:** Config A (StepTime-only) is the baseline.
**Success metric:** |E2E_A - E2E_B| >= 5 percentage points, with Config B being better.

**Additional diagnostics:**
- Per-method ablation: test each secondary method individually (turn on one at a time)
- Report which method contributes the most E2E improvement
- Check if any secondary method worsens E2E when added (interaction effects)

## BLIS E2E Claim

**Primary:** Non-zero secondary methods reduce BLIS E2E mean error by >= 5 percentage points.
**Secondary:** At least 2 of the 4 secondary methods individually contribute >= 1 percentage point improvement.
**Diagnostic:** Per-method contribution breakdown reveals which non-step latency components matter most for E2E fidelity.

## Related Work

- **LIFE** (arXiv:2508.00904): Explicitly separates dispatch overhead from computation time. Validates the importance of modeling non-compute latency.
- **WP0 component attribution:** Found StepTime dominates 100% of modeled error because other methods return 0. This hypothesis tests whether correcting that zero produces measurable E2E improvement.
- **BLIS design doc (Section F):** BlackboxLatencyModel returns 0 for SchedulingProcessingTime and PreemptionProcessingTime. These zeros represent systematic model incompleteness.

## Go Integration Path

No additional Go changes beyond h2. This hypothesis is an ablation study using the same artifact format with constants set to 0 vs. non-zero.

## Training Strategy and Data Split

- **Secondary method constants:** Derived from 60% temporal training split of lifecycle data, per model
- **BLIS validation:** All 16 experiments (same as h2)
- **No data leakage:** Constants are aggregated statistics (medians), not per-step predictions

## Data Integrity

- Ablation uses identical BLIS configuration except for secondary method values
- Both configs use the same StepTime model (h1's output) -- isolating secondary method contribution
- No roofline predictions used (BC-3-7)
