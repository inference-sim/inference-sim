# Idea 1, H3: XGBoost Cross-Configuration Generalization — FINDINGS

**Status:** Not Supported
**Date:** 2026-02-26

## Summary

A single XGBoost model does not generalize across unseen model configurations (LOMO avg 2559.7% MAPE) or workload types (LOWO avg 109.7% MAPE). The model learns configuration-specific patterns (step time scale, KV cache state distribution) that do not transfer. This confirms that **per-model XGBoost models are necessary** — consistent with how production inference simulators (Vidur, Splitwise) operate with per-model profiling.

## Experimental Setup

- **Model:** XGBoost with max_depth=6, n_estimators=300, learning_rate=0.05
- **Features:** 30 physics-informed features from h1
- **Validation:** 25% of training data held out temporally for early stopping (larger than h2's 20%)
- **CV schemes:**
  - LOMO: 4 folds, one model held out per fold (train on 12 experiments, test on 4)
  - LOWO: 4 folds, one workload held out per fold (train on 12 experiments, test on 4)
- **Data:** 122,752 steps from 16 experiments

## Results

### Leave-One-Model-Out (LOMO)

| Holdout Model | Train Size | Test Size | Valid MAPE | Test MAPE | MSPE | Pearson r |
|---------------|-----------|----------|------------|-----------|------|-----------|
| llama-2-70b | 98,892 | 23,860 | 42.7% | 65.5% | +33.1% | 0.947 |
| codellama-34b | 93,658 | 29,094 | 209.2% | 78.9% | +48.3% | 0.965 |
| mixtral-8x7b-v0-1 | 99,988 | 22,764 | 57.3% | 186.6% | +154.5% | 0.905 |
| llama-2-7b | 75,718 | 47,034 | 36.5% | 9908.0% | +9907.9% | 0.519 |

**LOMO Average: 2559.7% MAPE**

### LOMO Per-Experiment Breakdown

| Holdout | Model | Workload | MAPE | MSPE |
|---------|-------|----------|------|------|
| codellama-34b | codellama-34b | roleplay | 30.0% | +6.0% |
| llama-2-70b | llama-2-70b | roleplay | 32.0% | -0.8% |
| llama-2-70b | llama-2-70b | codegen | 49.4% | -11.7% |
| codellama-34b | codellama-34b | reasoning | 53.8% | -40.6% |
| llama-2-70b | llama-2-70b | general | 60.2% | +38.2% |
| mixtral-8x7b-v0-1 | mixtral-8x7b-v0-1 | reasoning | 77.5% | -42.3% |
| mixtral-8x7b-v0-1 | mixtral-8x7b-v0-1 | roleplay | 95.9% | +74.0% |
| codellama-34b | codellama-34b | general | 105.2% | +82.9% |
| codellama-34b | codellama-34b | codegen | 120.5% | +114.4% |
| llama-2-70b | llama-2-70b | reasoning | 147.3% | +145.0% |
| mixtral-8x7b-v0-1 | mixtral-8x7b-v0-1 | codegen | 185.4% | +169.6% |
| mixtral-8x7b-v0-1 | mixtral-8x7b-v0-1 | general | 359.5% | +352.4% |
| llama-2-7b | llama-2-7b | reasoning | 461.7% | +461.3% |
| llama-2-7b | llama-2-7b | general | 6781.0% | +6781.0% |
| llama-2-7b | llama-2-7b | codegen | 12471.2% | +12471.2% |
| llama-2-7b | llama-2-7b | roleplay | 12680.0% | +12680.0% |

### Leave-One-Workload-Out (LOWO)

| Holdout Workload | Train Size | Test Size | Valid MAPE | Test MAPE | MSPE | Pearson r |
|------------------|-----------|----------|------------|-----------|------|-----------|
| roleplay | 85,675 | 37,077 | 87.5% | 30.6% | +11.0% | 0.737 |
| reasoning | 104,668 | 18,084 | 103.1% | 53.2% | -37.6% | -0.411 |
| codegen | 85,725 | 37,027 | 57.0% | 53.4% | +42.9% | 0.737 |
| general | 92,188 | 30,564 | 96.6% | 301.6% | +286.0% | 0.236 |

**LOWO Average: 109.7% MAPE**

### LOWO Per-Experiment Breakdown (Top/Bottom 5)

| Holdout | Model | Workload | MAPE | MSPE |
|---------|-------|----------|------|------|
| codegen | mixtral-8x7b-v0-1 | codegen | 20.4% | +4.2% |
| general | llama-2-7b | general | 23.1% | -0.3% |
| reasoning | llama-2-7b | reasoning | 24.5% | +21.9% |
| roleplay | llama-2-70b | roleplay | 25.5% | -2.4% |
| roleplay | codellama-34b | roleplay | 27.4% | +6.8% |
| ... | ... | ... | ... | ... |
| codegen | codellama-34b | codegen | 121.4% | +118.1% |
| general | codellama-34b | general | 131.0% | +114.2% |
| general | mixtral-8x7b-v0-1 | general | 1299.0% | +1292.4% |

### Feature Importance Stability

**LOMO top features (by avg importance):**

| Feature | Avg Importance | Avg Rank | Rank StdDev |
|---------|---------------|----------|-------------|
| f_kv_blocks_used | 0.4280 | 2.2 | 2.5 |
| f_kv_blocks_free | 0.1096 | 7.0 | 4.1 |
| f_num_decode_reqs | 0.1043 | 7.5 | 8.4 |
| f_decode_tokens | 0.0599 | 8.0 | 5.9 |
| f_decode_memory_intensity | 0.0505 | 6.0 | 2.9 |

Top-10 average rank StdDev: **4.1** (moderate stability)

**LOWO top features:**

| Feature | Avg Importance | Avg Rank | Rank StdDev |
|---------|---------------|----------|-------------|
| f_intensity_x_moe | 0.3985 | 6.8 | 11.5 |
| f_decode_memory_intensity | 0.2409 | 3.2 | 3.2 |
| f_arithmetic_intensity | 0.1221 | 3.0 | 0.8 |
| f_running_depth | 0.0508 | 7.5 | 3.8 |
| f_kv_blocks_used | 0.0526 | 5.0 | 1.8 |

Top-10 average rank StdDev: **3.4** (slightly more stable than LOMO)

## Analysis

### Why LOMO Fails Catastrophically

**Llama-2-7B held out (9908% MAPE):** This is the most extreme failure. Llama-2-7B runs on 1 GPU (tp=1) with step times in the 12–500μs range. The model trained on 70B (tp=4), 34B (tp=2), and Mixtral (tp=2) learns a step time scale 10–100x larger. The features encode absolute values (total FLOPs, kv_blocks_used) that scale with model size, so the trained model predicts step times orders of magnitude too high for 7B.

**Mixtral held out (186.6% MAPE):** MoE architecture has fundamentally different compute patterns. The MoE-specific features (`f_is_moe`, `f_active_param_ratio`, `f_intensity_x_moe`) provide metadata but don't capture expert routing dynamics. Dense models can't predict MoE behavior.

### Why LOWO Fails Less Severely

Workload generalization is more feasible because the same model configuration produces step times at the same scale regardless of workload. The failures are concentrated in "general" workloads (301.6%) which have the most diverse batch compositions — the model trained on codegen/roleplay/reasoning doesn't encounter the specific batch patterns of general workloads.

### Feature Stability Interpretation

- **LOMO:** `f_kv_blocks_used` dominates (0.43 importance) with high stability (rank StdDev 2.5) — this is a system-state feature that captures step time scale per model configuration. It's informative but not transferable.
- **LOWO:** Physics features (`f_arithmetic_intensity`, `f_decode_memory_intensity`) are more important and stable — these capture computation structure that transfers across workloads better than system state.

## Verdict

**The hypothesis is not supported.** A single XGBoost model cannot generalize across unseen model configurations. This is a fundamental structural limitation, not a model capacity issue:

1. Step times span 3+ orders of magnitude across model sizes
2. The features encode absolute values (FLOPs, KV blocks) that are model-size-specific
3. Architecture metadata features (`f_model_id`, `f_tp_degree`, `f_is_moe`) are insufficient to bridge the scale gap

### Implications for Go Integration

The failed generalization confirms that **per-model (or per-experiment) XGBoost models are the practical deployment path**. This is consistent with existing practice:
- **Vidur** requires per-model profiling runs
- **Splitwise** uses per-model analytical models
- **BLIS blackbox** already uses per-model alpha/beta coefficients from `defaults.yaml`

The recommended Go integration ships one serialized XGBoost model per model configuration, selected via `--model` flag — analogous to how `defaults.yaml` currently selects alpha/beta coefficients per model.
