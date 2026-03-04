# Idea 4: Leave-Two-Out Robustness of Power Law Meta-Regression

## Hypothesis

The power law meta-regression formulas from Ideas 1 and 2 (`beta0 = a × params^b`, `alpha0 = a × params^b`, `beta2 = a × params_per_gpu^b`) remain viable when trained on only 2 of the 4 known models, achieving <80% holdout E2E across all C(4,2) = 6 training folds.

## Motivation

LOMO (leave-one-model-out) trains on 3 models and predicts 1 — but a 2-parameter power law fit from 3 points has only 1 degree of freedom. This is nearly interpolation and may overstate the formula's generalization. Leave-two-out (L2O) is a harder test: 2 training points exactly determine the 2 power law parameters (0 degrees of freedom), revealing whether the formula captures genuine physical scaling or is merely curve-fitting noise.

This matters for production: if a user has data from only 2 models, can they still use the power law to predict coefficients for a 3rd unseen model?

## Prior Round References

- R5 Idea 1: Power law meta-regression on R4 coefficients → 10.0% mean E2E, 16.6% LOMO (4/4 pass)
- R5 Idea 2: Hybrid metadata + step data → 8.6% mean E2E, 14.3% LOMO (4/4 pass)
- R4 BC-4-3: LOMO regression at 30.7% — R5 improved to 14.3% via power law interpolation
- R1: Global cross-model training fails catastrophically (2,559.7% MAPE) due to 3+ OOM scale variation
- Known risk: Only 4 data points for meta-regression (BC-5-4)

## Sub-Hypotheses

### H1: Dense-Model Pairs Generalize

**Claim:** When both training models are dense architectures with well-separated parameter counts (>3× ratio), the power law extrapolates to held-out models with <80% E2E.

**Relevant folds:** [7b, 34b], [7b, 70b], [34b, 70b]

**Pass criteria:** All 6 holdout predictions (3 folds × 2 holdout models) achieve <80% mean E2E.

**Rationale:** Dense models follow a consistent params→overhead scaling. With log-separated training points, the power law slope is well-determined.

### H2: MoE Training Pairs Degrade Gracefully

**Claim:** When one training model is MoE (Mixtral), the power law degrades but remains <80% for at least 50% of holdout predictions.

**Relevant folds:** [7b, mixtral], [34b, mixtral], [70b, mixtral]

**Pass criteria:** At least 3/6 holdout predictions achieve <80% mean E2E.

**Rationale:** Mixtral's dual scaling (46.7B total params for overhead, 12.9B active for compute) means `params_per_gpu` = 6.45B, which is close to 7B's 7.0B. Folds pairing Mixtral with 7B have near-identical x-values for beta2 fitting, making the power law slope indeterminate.

### H3: Catastrophic Failure Predictable from Training Point Spacing

**Claim:** L2O failures (>100% holdout E2E) occur exclusively when the two training models have `params_per_gpu` values within 2× of each other, making the power law exponent numerically unstable.

**Pass criteria:** All catastrophic failures (>100% E2E) can be explained by `max(ppg_train) / min(ppg_train) < 2`.

**Rationale:** Power law `y = a × x^b` requires log-separated x-values to determine b. When `log(x2/x1)` is small, the exponent `b = log(y2/y1) / log(x2/x1)` amplifies noise.

## Risk Assessment

**Primary risk:** With only 4 models in the dataset, C(4,2) = 6 folds is a small sample. Patterns observed may not generalize to a larger model zoo.

**Mitigation:** Focus on structural explanations (training point spacing, dense vs MoE) rather than aggregate statistics.

**Known limitation:** 2-parameter power law from 2 points is exactly determined (0 DOF). This is not a statistical fit — it's pure interpolation/extrapolation. Any noise in the 2 training coefficients propagates directly into the predicted coefficients.

## Experimental Design

For each of 6 training folds (pairs of models):
1. Fit `beta0 = a × params^b` from the 2 training models' R4 coefficients
2. Fit `alpha0 = a × params^b` from the 2 training models' R4 alpha0 values
3. Fit `beta2 = a × params_per_gpu^b` from the 2 training models' R4 beta2 values
4. Predict coefficients for both held-out models
5. Run BLIS E2E validation on all 10 experiments using predicted coefficients
6. Report per-holdout-model mean E2E

Repeat for both Idea 1 (pure metadata) and Idea 2 (metadata overhead + step-derived beta2).

## vLLM Args Sensitivity

Same as Ideas 1 and 2 — beta0 and alpha0 are metadata-derived (vLLM-invariant). beta2 depends on batch dynamics shaped by `max_num_seqs` and `max_num_batched_tokens`. Low recalibration cost: retrain beta2 from step data under new config.

## Go Integration Path

Same as Ideas 1 and 2: coefficient export to `defaults.yaml`. The power law formula itself could be implemented in Go for runtime prediction of unseen models: `beta0 = a * math.Pow(paramsB, b)`.
