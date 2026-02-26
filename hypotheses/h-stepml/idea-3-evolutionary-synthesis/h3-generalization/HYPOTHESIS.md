# Idea 3, H3: Evolved Formula Generalizes via Physically Meaningful Structure

**Status:** Deferred (depends on h1 and h2 which are deferred)
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> The evolved formula generalizes because it discovers physically meaningful structure (e.g., regime transitions, scaling laws) rather than overfitting to training data.

## Refuted-If

Leave-one-model-out MAPE > 35% for any held-out model, or the formula has >20 parameters (indicating overfitting rather than structural discovery). The relaxed threshold (35% vs. 25% for tree ensembles, 30% for analytical corrections) reflects that evolutionary search has a higher variance in generalization. However, if the formula is compact (<20 parameters) and still exceeds 35%, the evolved structure does not encode transferable physics.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. **Leave-one-model-out (LOMO) evolution:** For each of the 4 models, run the full evolutionary search (500 generations x 5 runs) on the remaining 3 models and evaluate the best program on the held-out model. This is expensive (4 folds x 5 runs x 500 generations) but necessary to test true generalization.
2. **Structural analysis:** For each LOMO-evolved program, analyze its structure:
   - **Regime detection:** Does the formula contain conditional branches (if/else)? If so, what thresholds do they encode? Do they correspond to known compute-bound vs. memory-bound transitions?
   - **Scaling laws:** Does the formula contain power-law terms (x^a for non-integer a)? Do the exponents correspond to known complexity scaling (linear in tokens, quadratic in sequence length for attention)?
   - **Feature selection:** Which of the 10 input features does the formula actually use? Is there convergence across LOMO folds (same features selected regardless of held-out model)?
3. **Physical interpretability scoring:** Rate each evolved formula on a 3-point scale: (a) fully interpretable -- maps to a known analytical model, (b) partially interpretable -- contains recognizable sub-expressions but with unexpected combinations, (c) opaque -- no recognizable physical structure. Report the distribution across all runs.
4. **Comparison with hand-designed models:** Compare LOMO MAPE of the evolved formula against Idea 1 h3 (XGBoost LOMO) and Idea 2 h3 (analytical corrections LOMO). This three-way comparison determines whether evolutionary search provides a generalization advantage.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Leave-one-model-out and leave-one-workload-out (entire experiments held out)
**Baselines:** XGBoost LOMO from Idea 1 h3, Analytical corrections LOMO from Idea 2 h3, Raw analytical sum (no corrections)
**Success metric:** Per-step MAPE < 20% on every held-out model with a formula of <20 parameters

**Overfitting detection:** Compare training MAPE vs. LOMO MAPE for each evolved formula. If the gap exceeds 15 percentage points (e.g., 10% training, 25% LOMO), the formula has overfit despite its compact size. Report the generalization gap for each run.

**Ablation study:** For the best-generalizing formula, systematically remove each sub-expression and measure MAPE degradation. Sub-expressions whose removal causes >5% MAPE increase are essential for generalization; others are potentially overfitting artifacts.

## Related Work

- **FunSearch** (Romera-Paredes et al., Nature 2024): The FunSearch paper demonstrated that evolved programs can discover novel mathematical constructions (cap set bounds) that were not known to human experts. Our hypothesis tests the analogous claim for performance modeling: can evolution discover non-obvious physical relationships?
- **GEPA** (Genetic-Pareto optimization with LLM-powered reflection): Maintains a Pareto frontier of accuracy vs. complexity, which directly maps to our MAPE vs. parameter-count tradeoff. The reflection mechanism (LLM analyzes why programs fail) could identify architecture-specific weaknesses.
- **Symbolic regression** (Schmidt & Lipson, Science 2009): Automated discovery of physical laws from data using genetic programming. Their method rediscovered Newton's laws from pendulum data -- a direct precedent for discovering physics-informed formulas from system performance data.

## Go Integration Path

If the evolved formula generalizes and is physically interpretable (score (a) or (b)), it ships as a documented Go function with comments explaining the physical interpretation of each sub-expression. This is the highest-value outcome: a compact, interpretable, physics-informed formula that requires no model files, no deserialization, and no external dependencies. The formula is versioned as Go source code and can be audited by anyone reading `sim/latency/stepml_evolved.go`. If the formula is opaque (score (c)) but generalizes, it still ships but with a prominent comment noting that interpretability was not achieved and the formula should be treated as a black box.
