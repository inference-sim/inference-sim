# Idea 1, H2: Joint Bayesian Optimization of All 5 LatencyModel Methods — FINDINGS

**Status:** Not Run (Blocked by H1)
**Resolution:** Converged to open question
**Family:** Performance-regime
**VV&UQ:** Validation
**Type:** Statistical/Dominance
**Date:** 2026-02-27
**Rounds:** 0

## Hypothesis

> Joint Bayesian optimization (BO) of all 5 LatencyModel method parameters — using h1's piecewise-linear StepTime as the base plus parameterized secondary methods (QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime) — with BLIS workload-level E2E mean error as the direct objective achieves <15% E2E mean error across all 16 experiments.

## Decision Not to Run

H2 was not executed for the following reasons:

**1. H1 base model quality is insufficient.** H1's piecewise-linear model achieved 87.4% per-step MAPE — far above the <30% target. BO optimizes secondary parameters (overhead, output token processing, scheduling, preemption) to minimize E2E error, but it cannot compensate for a fundamentally inaccurate StepTime model. The 4-dimensional BO search space (overhead + 3 secondary constants) has limited degrees of freedom — it can shift E2E predictions by a constant offset but cannot fix batch-composition-dependent errors.

**2. Expected compute cost is disproportionate.** Each BO evaluation requires 4 BLIS runs (one per workload). At 200 evaluations per model × 5 models = 1,000 evaluations × 4 BLIS runs = 4,000 BLIS runs. With each BLIS run taking ~1-2 minutes, the total runtime would be ~66-133 hours. This is not justified given H1's results.

**3. Idea 2's comparable test (H3 secondary calibration) was already refuted.** Idea 2 tested secondary method calibration (h3) and found that non-zero secondary methods do NOT improve E2E by >=5 percentage points. The secondary methods in BLIS (QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime) contribute minimally to E2E error compared to StepTime accuracy + overhead calibration.

**4. The overhead parameter alone provides most of the E2E benefit.** From Round 2 experience (documented in MEMORY.md), the `max(overhead, compute)` floor is the primary mechanism for achieving good E2E accuracy. BO over just the overhead parameter would be a 1-dimensional search that can be done analytically from ground-truth ITL data — no BO needed.

## What Would Be Needed to Revisit

H2 could be reconsidered if:
1. A stronger StepTime model achieves <30% per-step MAPE (e.g., Idea 2's 3-regime model, or a nonlinear model)
2. The BLIS binary path issue is fixed (currently `parents[4]` resolves to `hypotheses/` instead of repo root)
3. The BO search space is reduced to just the overhead parameter (1-dimensional), which can be done much faster

## Implementation Status

The `bo_calibrate.py` script is fully implemented and ready to run. It:
- Loads H1's piecewise artifacts as warm start
- Defines 4-dimensional search space (overhead + 3 secondary constants)
- Uses scikit-optimize's `gp_minimize` with GP surrogate
- Evaluates BLIS E2E error on all experiments per model
- Exports optimized artifacts

The script can be invoked with `--max-evals` to control compute budget:
```bash
python3 bo_calibrate.py --h1-artifact-dir ../h1-piecewise-steptime/output/artifacts --max-evals 50
```

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| H2 blocked by H1's 87.4% MAPE | Design limitation | Need stronger base model before BO is worthwhile |
| BO cannot compensate for structural model errors | Open question | Would reduced search space (overhead only) change the calculus? |
| Secondary methods contribute minimally to E2E | Confirmation (from Idea 2 H3) | No need to BO-optimize secondary constants |

## Reproducing

```bash
# Not recommended — requires ~66+ hours of compute
cd hypotheses/h-stepml/round2/idea-1-bayesian-calibration/h2-joint-bo-calibration
./run.sh
```
