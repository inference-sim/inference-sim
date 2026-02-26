# Idea 3, H1: Evolutionary Search Discovers Non-Obvious Feature Combinations

**Status:** Pending
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> An LLM-guided evolutionary search over the feature space discovers non-obvious feature combinations that a human expert would not design, achieving <25% per-step MAPE with a compact formula (<30 operations).

## Refuted-If

After 100 generations x 3 independent runs, the best evolved program achieves >30% per-step MAPE (worse than tree ensemble's feature set with Ridge regression from Idea 1 h1). This threshold ensures that evolutionary search adds value beyond human-designed feature engineering. If evolution cannot match Ridge regression on human-curated features, the search space or fitness function is misconfigured.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. **Search framework:** Use OpenEvolve (MAP-Elites + LLM-guided mutation) to evolve Python programs that predict step time from a dictionary of 18 raw features. The LLM proposes mutations (new feature combinations, mathematical transformations, conditional logic) and MAP-Elites maintains a diversity archive indexed by program complexity (number of operations) and error regime (prefill-dominated vs. decode-dominated vs. mixed).
2. **Fitness function:** Negative per-step MAPE on the 20% validation split, penalized by program complexity: `fitness = -MAPE - 0.01 * num_operations`. This encourages compact, interpretable formulas.
3. **Seed programs:** Initialize the population with 3 seeds: (a) the blackbox 2-feature linear model, (b) the analytical decomposition from Idea 2, (c) a random combination of 3 features. This ensures the search starts from known baselines rather than random noise.
4. **Budget:** 100 generations x population size 50 x 3 independent runs = 15,000 program evaluations per run. Each evaluation is fast (~1 second on the precomputed feature matrix).

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Temporal 60/20/20 -- evolution uses 60% for fitness evaluation, 20% for validation (early stopping / best program selection), 20% for final test
**Baselines:** Ridge regression with 30-feature set from Idea 1 h1, Raw analytical decomposition from Idea 2 h1, Blackbox 2-feature model
**Success metric:** Per-step MAPE < 25% with a program of <30 operations (arithmetic ops, comparisons, min/max)

## Feature Set

All 18 available features provided as a dictionary to the evolved program -- the search discovers which to use:

**Batch features (5):**
- `prefill_tokens`, `decode_tokens`, `scheduled_tokens`, `num_prefill_reqs`, `num_decode_reqs`

**KV statistics (5):**
- `kv_mean`, `kv_max`, `kv_min`, `kv_std`, `kv_sum` (derived from per-request ProgressIndex)

**Architecture features (4):**
- `hidden_dim`, `num_layers`, `num_heads`, `intermediate_dim`

**Hardware features (2):**
- `peak_flops_bf16`, `peak_bandwidth`

**MoE features (2):**
- `num_experts` (1 for dense), `active_experts` (1 for dense)

The evolved program receives these as a flat dictionary and outputs a single float (predicted step duration in microseconds). The search can discover arbitrary combinations, including conditional expressions (if/else), non-linear transformations (sqrt, log, power), and piecewise functions.

## Related Work

- **OpenEvolve** (2025): Open-source implementation of MAP-Elites + LLM-guided evolution for algorithm discovery. Demonstrated on bin packing and TSP; this would be a novel application to performance modeling.
- **FunSearch** (Romera-Paredes et al., Nature 2024): LLM-guided evolutionary search that discovered new mathematical constructions for the cap set problem. Demonstrates that LLMs can propose non-trivial algorithmic innovations when guided by a well-defined fitness function.
- **GEPA** (Genetic-Pareto optimization): LLM-powered multi-objective optimization that maintains a Pareto frontier of accuracy vs. complexity. Directly applicable to our accuracy-vs-interpretability tradeoff.

## Go Integration Path

The evolved Python program would be manually translated to Go (the program is <30 operations, so translation is trivial). The translated function becomes the core of a `StepMLEvolvedModel` that implements `LatencyModel.StepTime()`. Because the evolved formula is a compact mathematical expression (not a tree ensemble or neural network), it compiles to a few dozen arithmetic instructions with zero allocation overhead. The formula is stored as Go source code (not a serialized model), making it fully auditable and version-controllable. If the formula uses conditional logic (if/else), it maps directly to Go control flow.
