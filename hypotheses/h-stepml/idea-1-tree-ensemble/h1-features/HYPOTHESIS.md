# Idea 1, H1: Physics-Informed Feature Set Achieves <25% MAPE with Ridge Regression

**Status:** Weakly Supported (per-experiment); Short-circuited (global)
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> A feature set combining roofline-derived physics features (compute intensity, arithmetic intensity, memory boundedness) with per-request KV cache statistics (mean, max, variance from ProgressIndex) achieves <25% per-step MAPE with Ridge regression, significantly outperforming the 2-feature blackbox baseline.

## Refuted-If

Ridge regression with the proposed 30-feature set achieves MAPE > 30% (the short-circuit threshold). If the full feature set cannot beat 30% MAPE with a linear model, tree ensembles cannot be justified -- the features themselves are insufficient.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** Train Ridge regression on the 30-feature set using temporal 60/20/20 split within each of the 16 experiments. Compare per-step MAPE against the blackbox 2-feature baseline (prefill_tokens, decode_tokens) and a naive mean baseline.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Temporal 60/20/20 within each experiment (preserves time-series structure)
**Baselines:** Blackbox 2-feature Ridge (must outperform), Naive per-experiment mean
**Success metric:** Per-step MAPE < 25% on the held-out 20% test set, averaged across all 16 experiments

## Feature Set

30 features organized into 6 groups:

**Batch tokens (5):**
- `prefill_tokens` -- total prefill tokens in this step
- `decode_tokens` -- total decode tokens in this step
- `scheduled_tokens` -- total scheduled tokens (prefill + decode)
- `num_prefill_reqs` -- number of prefill requests in batch
- `num_decode_reqs` -- number of decode requests in batch

**KV statistics (5):**
- `kv_mean` -- mean KV cache length across decode requests (from ProgressIndex)
- `kv_max` -- maximum KV cache length in batch
- `kv_min` -- minimum KV cache length in batch
- `kv_std` -- standard deviation of KV cache lengths in batch
- `kv_sum` -- sum of KV cache lengths (total attention memory access)

**Phase indicators (4):**
- `prefill_fraction` -- prefill_tokens / scheduled_tokens
- `decode_fraction` -- decode_tokens / scheduled_tokens
- `is_mixed_batch` -- 1 if both prefill and decode requests present, 0 otherwise
- `is_pure_prefill` -- 1 if no decode requests, 0 otherwise

**Physics features (8):**
- `total_flops_estimate` -- roofline-derived total FLOPs for this step
- `arithmetic_intensity` -- FLOPs / memory_bytes (operational intensity)
- `compute_bound_indicator` -- 1 if arithmetic_intensity > machine_balance_point
- `prefill_compute_intensity` -- prefill FLOPs / prefill memory bytes
- `decode_memory_intensity` -- decode memory bytes / decode FLOPs
- `attention_flops_ratio` -- attention FLOPs / total FLOPs
- `gemm_flops_ratio` -- GEMM FLOPs / total FLOPs
- `active_param_ratio` -- active parameters / total parameters (1.0 for dense, ~0.25 for MoE)

**Architecture (3):**
- `model_id` -- one-hot encoded model identifier
- `tp_degree` -- tensor parallelism degree
- `is_moe` -- 1 if MoE architecture, 0 if dense

**Interaction terms (5):**
- `prefill_tokens x decode_tokens` -- mixed-batch interaction (non-additive, per BLIS H5)
- `kv_max x num_decode_reqs` -- attention memory pressure interaction
- `arithmetic_intensity x is_moe` -- MoE compute pattern interaction
- `prefill_tokens x kv_mean` -- prefill-decode overlap interaction
- `batch_size x kv_max` -- batch scaling interaction (batch_size = num_prefill_reqs + num_decode_reqs)

## Related Work

- **XGBoost** (Chen & Guestrin, KDD 2016): Gradient boosted trees with regularization -- the target model for h2 if features validate here.
- **Habitat** (Yu et al., ATC 2021): Predicts DNN execution time using analytical + learned models; demonstrates that physics-informed features improve generalization.
- **Ipek et al.** (ISCA 2006): Neural network performance prediction for memory systems; showed that feature engineering matters more than model complexity for hardware performance prediction.

## Go Integration Path

Features would be computed in a new `stepml_features.go` file within `sim/latency/`, called during batch formation to construct the feature vector from `BatchContext` (which contains the batch's requests with their ProgressIndex values). The feature computation is a pure function: `ComputeFeatures(batch []Request, modelConfig ModelConfig) []float64`. No interface changes needed -- this feeds into the existing `LatencyModel.StepTime()` method.
