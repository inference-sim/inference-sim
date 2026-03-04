# FINDINGS: H2 — Workload-Spec Mode E2E Optimization

**Date:** 2026-02-27
**Status:** NOT TESTED (infrastructure failure)

## Claim

CMA-ES optimization in workload-spec mode (not trace replay) achieves < 50% mean E2E error, a 8.5× improvement over Round 2's 427.8%.

## Result

**All 10 BLIS runs failed.** No E2E error data was collected.

## Root Cause

The BLIS workload-spec mode requires profile.yaml files from the original inference-sim repository. These profile files were located in `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/eval/ground_truth/`, but the experiment directory names differ between the BLIS-research and inference-sim repositories:

- **BLIS-research:** `20260217-162547-llama-2-7b-tp1-roleplay` (has lifecycle data, no profile)
- **inference-sim:** `jan30-llama2-7b-tp1-codesweep` (has profile, no lifecycle data)

The experiment name mismatch means the profile lookup fails silently, producing `None`, and then the workload spec construction fails.

Additionally, the CMA-ES optimization produced artifacts with negative parameter values (e.g., negative queueing_intercept) that may cause BLIS configuration validation failures in workload-spec mode.

## Refutation Assessment

- **INCONCLUSIVE:** The hypothesis was not tested due to infrastructure limitations
- **Expected outcome if tested:** Based on Idea 1 (R2 → R3), workload-spec mode produces 427.8% E2E even with correct coefficients. E2E optimization would partially compensate, but the workload-spec generation process (not parameters) is the dominant error source. We estimate < 50% would be achievable but the improvement would be modest compared to trace replay.

## Recommendation

To properly test H2, one would need to:
1. Create a mapping between BLIS-research and inference-sim experiment names
2. Or generate workload specs from the lifecycle data itself (synthesize profiles from ground-truth statistics)
