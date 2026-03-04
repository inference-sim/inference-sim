# Idea 1, H2: Joint Bayesian Optimization of All 5 LatencyModel Methods Achieves <15% BLIS E2E Mean Error

**Status:** Not Run (blocked by H1)
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-26

## Hypothesis

> Joint Bayesian optimization (BO) of all 5 LatencyModel method parameters -- using h1's piecewise-linear StepTime as the base plus parameterized secondary methods (QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime) -- with BLIS workload-level E2E mean error as the direct objective achieves <15% E2E mean error across all 16 experiments.

The key claim is that optimizing against the actual BLIS simulator output (E2E latency) rather than a proxy (per-step MAPE) unlocks additional accuracy. Compensating errors between the 5 methods are acceptable if E2E is correct, but TTFT and ITL must also be tracked to detect pathological compensation.

## Refuted-If

- BLIS E2E mean error > 15% after 500 BO evaluations per model
- OR TTFT mean error > 50% (indicating compensating errors distort simulation dynamics)
- OR ITL mean error > 50% (same check)
- OR BO fails to converge (best E2E error shows no improvement after first 100 evaluations)

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** Bayesian optimization with Gaussian process surrogate (scikit-optimize). For each BO iteration:
1. Propose parameter values for all 5 methods (~12-15 total: h1's StepTime coefficients + secondary method constants)
2. Generate StepML JSON artifact with proposed values
3. Run BLIS validation harness (`validate_blis.py`) on all 4 workloads for the target model
4. Score: mean E2E error across 4 workloads
5. Update GP surrogate and select next proposal

**Search space per model:**
- StepTime: h1's 8 piecewise-linear coefficients (warm-started from h1's fit)
- QueueingTime: `q0` (constant, per model) -- `[0, 10000]` microseconds
- OutputTokenProcessingTime: `o0` (constant, per model) -- `[0, 5000]` microseconds
- SchedulingProcessingTime: `s0` (constant, per model) -- `[0, 5000]` microseconds
- PreemptionProcessingTime: `p0` (constant, per model) -- `[0, 5000]` microseconds
Total: ~12 parameters per model

**Budget:** 300-500 BO evaluations per model (each = 4 BLIS runs ~ 4 min). Total: ~20-33 hours per model.
**Warm start:** Initialize StepTime coefficients from h1's fit; secondary methods at 0 (current default).

**Data:** Same 16 experiments as h1. BLIS validation uses `validate_blis.py` which replays ground-truth traces.
**Split:** Leave-one-workload-out cross-validation within each model. BO trains on 3 workloads, reports E2E on held-out workload.

## BLIS E2E Claim

**Primary claim:** BLIS E2E mean error < 15% across all 16 experiments (4 models x 4 workloads).
**Secondary claim:** BLIS E2E mean error < 10% on at least 8/16 experiments.

## Related Work

- **Kennedy & O'Hagan (2001):** "Bayesian calibration of computer models" -- foundational work on using BO for simulator calibration against real-world observations. Direct methodological precedent.
- **MIST** (arXiv:2504.09775): Achieves 2.5% step-level error but validates only per-operator, not through end-to-end simulation. This hypothesis validates through the full BLIS DES pipeline.
- **LIFE** (arXiv:2508.00904): Treats dispatch overhead as separate additive term. Analogous to calibrating SchedulingProcessingTime.
- **Vidur** (MLSys 2024): Per-model calibration with random forests achieving <5% per-operator error. This hypothesis takes the per-model strategy further by optimizing against E2E output.

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 12-dimensional BO has poor convergence | Medium | Warm-start from h1; reduce to 4-dimensional (secondary methods only) if StepTime coefficients stable from h1 |
| Compensating errors fragile across workloads | High | Track TTFT and ITL separately; reject solutions where TTFT or ITL error > 50% |
| 500 evaluations insufficient for 12 parameters | Medium | Use structured BO: fix StepTime at h1 values for first 200 evals, then unfreeze all |
| BLIS run failures during BO | Low | Catch BLIS errors, return penalty score (200% E2E) to BO |

## Go Integration Path

BO-optimized coefficients exported in the same `StepMLArtifact` JSON format as h1. StepTime uses piecewise-linear coefficients; secondary methods use per-model constants. No additional Go code changes beyond h1's regime dispatch.

## Training Strategy and Data Split

- **Training:** BO uses 3 workloads per model (12 experiments) as the optimization objective
- **Evaluation:** Held-out workload per model (4 experiments) -- reported separately
- **Aggregate:** Mean E2E error across all 16 experiments (train + eval) reported for leaderboard
- **No data leakage:** BO sees only E2E error (aggregated metric), never raw step traces

## Data Integrity

- BLIS validation harness produces E2E metrics identically to production
- BO parameter proposals are bounded to physically meaningful ranges
- Leave-one-workload-out ensures generalization is tested per model
- Secondary methods are constants (no runtime-state dependency), consistent with the LatencyModel interface where these methods take no arguments beyond the batch
