# Idea 3, H2: Evolved Function Achieves <10% E2E Mean Error Using Analytical Components

**Status:** Deferred (depends on Idea 2 analytical components which are structurally incomplete)
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> The best evolved prediction function achieves <10% workload-level E2E mean error using Idea 2's analytical components as fixed inputs, evolving only the combination/correction logic.

## Refuted-If

After 500 generations x 5 independent runs, the best evolved function achieves >15% E2E mean error on more than 4 experiments. The extended budget (500 generations vs. 100 in h1) and relaxed threshold (15% vs. 10%) account for the increased difficulty of E2E optimization (where per-step errors propagate through the simulation). If evolution cannot reach 15% with analytical components as inputs, the combination of evolutionary search + analytical decomposition does not synergize.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. **Input representation:** The evolved program receives Idea 2's 4 analytical component estimates (prefill_gemm_time, prefill_attn_time, decode_gemm_time, decode_attn_time) plus 6 contextual features (prefill_tokens, decode_tokens, num_prefill_reqs, num_decode_reqs, kv_mean, kv_max) as inputs. Total: 10 inputs. The analytical components encode the physics; the contextual features provide information for regime detection.
2. **Search space:** The evolved function can use arithmetic operations (+, -, *, /), comparisons (<, >), min/max, conditional expressions (if/else), and constants. It outputs a single float (predicted step duration). The search is constrained to <50 operations to prevent overfitting.
3. **Two-stage fitness:** Stage 1 (generations 1-200): Optimize per-step MAPE on training data. Stage 2 (generations 200-500): Optimize workload-level E2E mean error via fast BLIS simulation runs. This two-stage approach first finds good per-step predictors, then refines them for the downstream metric that matters.
4. **Population management:** MAP-Elites archive indexed by (program_complexity, E2E_error_regime). This maintains diversity across simple-but-biased and complex-but-accurate programs.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Temporal 60/20/20 -- stage 1 uses 60% for fitness, stage 2 uses BLIS simulation on full workloads with the validation split for program selection
**Baselines:** Idea 2 h2 nonlinear least squares (the hand-designed correction approach), Idea 1 h2 XGBoost (the data-driven approach), Raw analytical sum (sum of 4 components without correction)
**Success metric:** Workload-level E2E mean error < 10% on at least 12 of 16 experiments

**Computational budget:** 500 generations x population 50 x 5 runs = 125,000 evaluations. Stage 1 evaluations are fast (~1 second each, total ~14 hours for stage 1). Stage 2 evaluations require BLIS simulation runs (~30 seconds each), but only the top 10% of stage 1 programs are evaluated in stage 2, so ~12,500 simulation runs total (~100 hours, parallelizable across 16 cores to ~6 hours).

**Novelty detection:** After evolution completes, analyze the best programs for structural novelty: (a) Did evolution discover interaction terms not present in the hand-designed feature set? (b) Did it discover regime transitions (piecewise functions) at non-obvious thresholds? (c) Does the formula simplify to a known analytical form? Document any discoveries as potential contributions to the systems/ML literature.

## Related Work

- **OpenEvolve** (2025): MAP-Elites + LLM-guided evolution. The two-stage fitness function (per-step then E2E) is a novel application of multi-stage evolutionary optimization.
- **FunSearch** (Romera-Paredes et al., Nature 2024): Demonstrated that evolved programs can outperform hand-designed algorithms on combinatorial problems. Our application tests whether this transfers to continuous prediction problems with physics-informed inputs.

## Go Integration Path

Same as Idea 3 h1 -- the evolved function translates to a compact Go expression. The key difference from h1 is that this function receives pre-computed analytical components as inputs (computed by the `DecomposeStepTime()` function from Idea 2). The integration would compose both: `StepMLEvolvedModel` calls `DecomposeStepTime()` internally, then applies the evolved combination logic. This composition is natural in Go: the evolved formula is a method body, not a separate model file. Total code addition: ~50 lines of Go (the formula + the composition plumbing).
