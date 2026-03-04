# Research Document: Round 5 — Unified Cross-Model Latency Prediction

## Problem Statement

Achieve <10% mean E2E MAPE across 10 experiments (4 models × 3 workloads) using a **single unified model** trained across all LLMs — eliminating the per-model calibration requirement from R4.

**R4 baseline (per-model):** 5.7% mean E2E, 9/10 <10%. But requires per-model lifecycle data for calibration.
**R5 goal:** Match or approach R4 accuracy with a single model that generalizes to unseen models without calibration data.

## User Directive

Round 5 is specifically focused on training a SINGLE model that works across all LLMs. This addresses the LOMO gap (R4: 30.7%, R3: 14.8% CMA-ES transfer). The target is <10% mean E2E with ONE set of coefficients or one model applied to all 4 model architectures.

## Key Context from Prior Rounds

### Why cross-model has failed (R1-R4)

| Round | Cross-Model Approach | LOMO Result | Root Cause |
|-------|---------------------|-------------|------------|
| R1 | Global XGBoost | 2,559.7% | 3+ OOM step-time scale variation |
| R2 | Regime ensemble | 108.6% | Regime structure helps but not enough |
| R3 | Total-context model | 2,281.6% | No regime structure → R1-level failure |
| R3 | CMA-ES artifact transfer | **14.8%** | Simulation dynamics transfer, not coefficients |
| R4 | Direct calibration coefficients | 30.7% | Additive intercepts are model-specific |

### What makes cross-model hard

1. **Step times span 3+ OOM:** 7B steps ~12-500μs, 70B steps ~500-250,000μs
2. **Overhead floor (beta0) scales with model:** 9.7ms (7B) → 14.2ms (34B) → 18.0ms (70B) → 18.9ms (Mixtral)
3. **Batch sizes differ:** 7B avg 12 tokens/batch, others 33-46
4. **GPU compute is only 2-8% of step time** — the model-specific overhead dominates
5. **Architecture diversity:** Dense (7B, 34B, 70B) vs MoE (Mixtral) have different compute patterns

### What partially worked for cross-model

1. **R3 CMA-ES artifact transfer (14.8%)** — simulation dynamics (overhead + scheduling interactions) are more transferable than model-specific constants
2. **R4 large-model transfer (4.5-6.7%)** — 70B↔Mixtral transfer works well because similar overhead scales
3. **Mixtral as universal donor (11.9-26.0%)** — best single-model donor for all targets
4. **Overhead scaling is predictable:** beta0 ∝ O(sqrt(params)), roughly: 9.7ms, 14.2ms, 18.0ms, 18.9ms

### R4 Production Coefficients (the per-model baseline)

| Model | Params | TP | alpha0 (TTFT μs) | beta0 (intercept μs) | beta1 (prefill) | beta2 (decode) |
|-------|--------|-----|-------------------|---------------------|------------------|----------------|
| llama-2-7b | 7B | 1 | 27,129 | 9,741 | 0.30 | 13.6 |
| codellama-34b | 34B | 2 | 47,618 | 14,196 | 0.00 | 25.8 |
| llama-2-70b | 70B | 4 | 78,888 | 17,992 | 1.22 | 35.2 |
| mixtral-8x7b | 46.7B (12.9B active) | 2 | 62,767 | 18,921 | 0.69 | 8.8 |

### Key Physics Insight

GPU compute (beta2 × batch) is tiny (2-8% of step time). The additive intercept (beta0) dominates at 92-98%. If we can predict beta0 from model metadata alone, the rest is noise.

**beta0 vs model parameters:**
- 7B (TP1): 9,741μs
- 34B (TP2): 14,196μs → 14,196/9,741 = 1.46×
- 70B (TP4): 17,992μs → 17,992/9,741 = 1.85×
- Mixtral (TP2): 18,921μs → 18,921/9,741 = 1.94×

Scaling: beta0 ≈ 9741 × (params/7B)^0.30 fits approximately (7B→1.0, 34B→1.50, 70B→1.86, 47B→1.63). The actual Mixtral value (1.94) is higher due to MoE routing overhead.

## Background (Literature)

### Vidur (OSDI 2024)
Uses per-model profiling with execution time prediction based on transformer layer structure. Models attention and GEMM separately. Requires per-model kernel benchmarking — NOT zero-shot cross-model.

### FairBatching (EuroSys 2025)
Per-model OLS with 3 coefficients: `step_time = a + b*new_tokens + c*kv_sum`. Simple and effective but strictly per-model.

### Splitwise (ISCA 2024)
Analytical model based on FLOPs and memory bandwidth. Uses roofline analysis per model. Cross-model via physics, but needs hardware profiling.

### Key Insight from Literature
No existing system achieves zero-shot cross-model latency prediction. All require either (a) per-model profiling, (b) per-model training data, or (c) analytical models with per-model architecture parameters. The closest to "cross-model" is using analytical FLOPs + measured MFU, but this requires per-model hardware benchmarks.

## Constraints for Round 5

1. Must use **trace replay mode** (workload-spec is dead — confirmed R2, R3, R4)
2. Must achieve <10% mean E2E across ALL 10 experiments with a SINGLE model/formula
3. The model CAN use model metadata (parameter count, TP degree, architecture type, num_layers, hidden_size, num_heads) as features
4. The model CAN use per-model constants IF derivable from metadata (not from calibration data)
5. Must implement via existing BlackboxLatencyModel or a simple Go extension
6. **Per-model CALIBRATION DATA is what we're trying to eliminate** — the model should work for a new model without running requests first

## Eliminated Approaches (Do Not Repeat)

- Global models without scale normalization (R1: 670%)
- Raw per-model training only (defeats the purpose of R5)
- Workload-spec mode (dead — R2, R3, R4)
- Log-space targets with large features (exponential blowup — R2)
- Per-step MAPE as proxy for E2E (broken — R3)

---

# Idea 1: Analytical Overhead Model with Metadata-Derived Coefficients

## Approach

Derive ALL four coefficients (alpha0, beta0, beta1, beta2) from model architecture metadata alone, using the R4 data as training points for a meta-regression:

1. **beta0 = f(params, tp_degree, is_moe)** — fit a power law from the 4 known models
2. **beta2 = f(params_per_gpu, flops_per_token)** — derive from analytical FLOPs/token
3. **beta1 ≈ 0** (negligible for all models — R4 showed 0.00-1.22)
4. **alpha0 = g(params, tp_degree)** — fit power law from the 4 known TTFTs

This produces a formula that, given HuggingFace config.json metadata, predicts all coefficients. Zero calibration data needed.

### Meta-Regression Training

Use R4's 4 model data points as training data for the meta-model:
- Input: (param_count, tp_degree, is_moe, num_layers, hidden_size)
- Output: (beta0, beta2, alpha0)

### LOMO Plan

Leave one model out of the meta-regression, predict its coefficients from the remaining 3, run BLIS with predicted coefficients.

### LOWO Plan

Same coefficients across all workloads (workload-invariant by design since metadata doesn't include workload info).

### vLLM-args Sensitivity

- **max_num_seqs**: Affects avg_decode_batch → changes the effective beta0 (overhead depends on batch formation cost). Recalibration: run a few batches to measure new overhead constant.
- **chunked_prefill**: Changes step composition. beta1 would change. Minimal impact since beta1 ≈ 0.
- **tensor_parallel_size**: Changes params_per_gpu → changes both beta0 and beta2. The meta-model takes TP as input, so it should adapt.
- **Other params**: Low sensitivity (beta0 dominates and is relatively stable).

### Go Integration

Coefficient export — compute (alpha0, beta0, beta1, beta2) from HuggingFace config.json at startup.

---

# Idea 2: Normalized Feature Space with Scale-Invariant Regression

## Approach

Instead of predicting raw step_time (which spans 3+ OOM), predict **normalized step time**:

```
normalized_step = step_time / model_overhead_estimate
```

Where `model_overhead_estimate` is derived analytically from model metadata (e.g., `base_overhead × (params/7B)^0.3`).

Train a SINGLE Ridge regression across ALL models on normalized features:
- `normalized_decode = decode_tokens / max_num_seqs` (relative batch fullness)
- `normalized_prefill = prefill_tokens / max_num_batched_tokens`
- `model_scale = params_per_gpu / 7B` (relative model size)
- `tp_factor = 1 / tp_degree` (parallelism discount)

The regression predicts a **multiplier** on the base overhead:
```
step_time = model_overhead × (1 + w1*norm_decode + w2*norm_prefill + w3*model_scale)
```

### Why This Might Work

R1-R4 showed that step time = 92-98% overhead + 2-8% compute. If we normalize away the overhead (which is model-specific but predictable from metadata), the remaining variation is small and potentially shared across models.

### LOMO Plan

Train Ridge on 3 models, predict on held-out 4th. The normalized features should transfer because relative batch fullness and relative compute intensity are model-invariant concepts.

### LOWO Plan

Same model for all workloads. Normalized features (batch fullness) should be workload-invariant.

### vLLM-args Sensitivity

- **max_num_seqs**: Changes the normalization denominator for decode_tokens. The model captures relative batch fullness, so it should adapt.
- **max_num_batched_tokens**: Changes prefill normalization.
- **chunked_prefill**: Changes the distribution of prefill tokens but not the normalized feature definition.
- **tensor_parallel_size**: Changes params_per_gpu → changes model_scale feature.

### Go Integration

Coefficient export — small number of weights for the Ridge regression, plus the metadata-derived overhead formula.

---

# Idea 3: Hierarchical Two-Stage Model (Shared Physics + Model-Specific Intercept)

## Approach

Split the prediction into two stages that separate transferable physics from model-specific constants:

**Stage 1 (Shared across models):** Learn the relationship between batch composition and *relative* step time variation using a single global model:
```
delta_time = shared_model(decode_tokens, prefill_tokens, batch_features)
```

**Stage 2 (Model-specific but metadata-derived):** Add a model-specific base overhead predicted from architecture metadata:
```
step_time = metadata_overhead(params, tp, is_moe) + delta_time
```

The key insight: Stage 1 captures HOW step time varies with batch composition (this is shared physics — more tokens = more compute, the shape is similar across models). Stage 2 captures the ABSOLUTE scale (which is model-specific but predictable from metadata).

### Training

1. For each model, compute the residual: `delta = step_time - mean_step_time_for_model`
2. Pool ALL residuals across models into one training set
3. Train a single Ridge/XGBoost on (decode_tokens, prefill_tokens) → delta
4. Fit `metadata_overhead = mean_step_time` as a function of model metadata

### Why This Might Work

R4 showed that beta0 (the constant) accounts for 92-98% of step time. The variation (beta2 × batch) is small and driven by the same physics (more tokens → more FLOPs → more time). If the variation pattern is similar across models (just scaled), a shared model of the variation should transfer.

### LOMO Plan

Hold out one model. Use Stage 1 shared model (trained on 3 models) + metadata-derived overhead for the held-out model. This tests whether the residual pattern (step time variation around the mean) is truly shared.

### LOWO Plan

Same two-stage model for all workloads. Workload-invariant by design.

### vLLM-args Sensitivity

- **max_num_seqs**: Changes batch composition distributions → Stage 1 should handle via batch features.
- **chunked_prefill**: Changes mixed-batch frequency → affects Stage 1 training distribution. May need retraining of Stage 1 if disabled.
- **tensor_parallel_size**: Changes model_scale → Stage 2 metadata formula adapts.

### Go Integration

Two components: (1) metadata overhead formula (computed at startup from config.json), (2) small Ridge/tree model for delta prediction. Both lightweight.
