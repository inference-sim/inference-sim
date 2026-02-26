# Idea 1, H2: XGBoost Achieves <15% Per-Step MAPE and <10% Workload-Level E2E Mean Error

**Status:** Pending
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> An XGBoost model with the validated feature set from h1 (after feature selection) achieves <15% per-step MAPE and <10% workload-level E2E mean error on each of the 16 experiments, significantly outperforming the blackbox baseline.

## Refuted-If

The best XGBoost configuration achieves >10% workload-level E2E mean error on more than 4 of 16 experiments. This threshold allows for MoE or extreme-workload outliers while requiring strong performance on the majority of configurations.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. Take the validated feature set from h1 (features that survived importance analysis and achieved <25% MAPE with Ridge).
2. Train XGBoost with hyperparameter search (max_depth: [4, 6, 8], n_estimators: [100, 300, 500], learning_rate: [0.01, 0.05, 0.1], min_child_weight: [1, 5, 10]).
3. Evaluate per-step MAPE on 20% temporal test split.
4. Run BLIS simulation with the trained model as the latency backend and compare workload-level E2E mean latency against ground truth.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Temporal 60/20/20 within each experiment
**Baselines:** Blackbox 2-feature model (must outperform), Ridge with h1 features (quantifies tree benefit), Roofline analytical model (comparison only -- not a replacement target)
**Success metric:** Per-step MAPE < 15% AND workload-level E2E mean error < 10% on at least 12 of 16 experiments

**Hyperparameter search:** Grid search with 5-fold temporal cross-validation on the 60% training split. The validation split (20%) is used for early stopping and model selection. The test split (20%) is used only for final evaluation.

**Error propagation analysis:** Per-step prediction errors may cancel (symmetric errors around zero) or compound (systematic bias) when aggregated into workload-level E2E mean. This experiment explicitly measures both per-step MAPE and workload-level E2E mean error to characterize the propagation relationship.

## Related Work

- **LightGBM** (Ke et al., NeurIPS 2017): Gradient boosting with histogram-based splitting -- a direct alternative to XGBoost with potentially faster training. Should be included as a secondary model comparison.
- **vLLM** (Kwon et al., SOSP 2023): The serving system whose step-time behavior we are predicting. Understanding vLLM's batch formation and memory management is essential for interpreting model predictions.
- **XGBoost** (Chen & Guestrin, KDD 2016): Regularized gradient boosting with column subsampling -- the primary model architecture for this hypothesis.

## Go Integration Path

The trained XGBoost model would be exported as a set of decision trees in a custom JSON format. A pure-Go tree evaluator in `sim/latency/stepml.go` would load the serialized model and implement the `LatencyModel` interface. No CGo or Python dependencies -- the tree evaluation is a series of if/else comparisons on the feature vector. Model files would be stored alongside `defaults.yaml` and loaded via the existing `--model-config-folder` mechanism or a new `--stepml-model` flag.
