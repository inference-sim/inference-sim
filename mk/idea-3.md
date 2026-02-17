# Idea 3: Hierarchical Feature Decomposition with Transfer Learning (HFD-TL)

## Core Insight

Reviewer feedback on Ideas 1 and 2 highlighted two recurring themes: (1) the difficulty of pre-specifying the right features or regime boundaries, and (2) the substantial data requirements for training accurate models across diverse configurations. This idea proposes a **hierarchical decomposition** where:

1. A **base model** is trained on rich data from a few representative configurations (the "donor" configurations)
2. **Configuration-specific residuals** are learned with minimal data from new configurations (the "target" configurations)

This enables rapid calibration to new (model, hardware, vLLM) settings while preserving the interpretable alpha/beta coefficient structure.

## Approach Overview

### Phase 1: Hierarchical Feature Decomposition

Decompose latency into three additive components:

```
alpha_latency = alpha_base · F_base + alpha_config · F_config + alpha_residual · F_residual
beta_latency = beta_base · F_base + beta_config · F_config + beta_residual · F_residual
```

**Base Features (F_base)** - Universal, hardware-normalized physics:
```python
F_alpha_base = [
    queue_depth / max_num_seqs,  # Normalized queue load
    kv_usage_ratio,  # Cache pressure (dimensionless)
    prompt_tokens / max_context_length,  # Normalized request size
]

F_beta_base = [
    prefill_tokens * model_flops_per_token / hardware_tflops,  # Normalized prefill work
    decode_tokens * kv_bytes_per_token / hardware_bandwidth,  # Normalized decode work
    batch_tokens / max_batch_tokens,  # Normalized batch load
]
```

**Configuration Features (F_config)** - Configuration-specific modifiers:
```python
F_alpha_config = [
    1 / tp_degree,  # TP affects scheduling
    prefix_caching_enabled,  # Binary: affects queue time
    chunked_prefill_enabled,  # Binary: affects scheduling
]

F_beta_config = [
    1 / tp_degree,  # TP reduces per-GPU work
    moe_indicator * num_experts / 8,  # MoE complexity (normalized)
    cpu_offload_enabled,  # Binary: adds transfer overhead
]
```

**Residual Features (F_residual)** - Capture unexplained variance:
```python
F_alpha_residual = [
    1,  # Constant bias term
    (queue_depth / max_num_seqs) ** 2,  # Non-linear queue effect (precomputed)
]

F_beta_residual = [
    1,  # Constant bias term
    kv_usage_ratio ** 2,  # Non-linear saturation effect (precomputed)
]
```

**Note**: The quadratic terms are **precomputed** before the dot product, so they remain scalar features that multiply against scalar coefficients. This satisfies the linear constraint: `alpha · [x, x^2]` is still a dot product.

### Phase 2: Two-Stage Training Pipeline

**Stage 1: Train Base Model on Donor Configurations**

Select 2-3 representative configurations spanning the diversity space:
- Dense model on H100 (e.g., Llama-3-8B)
- MoE model on H100 (e.g., Mixtral-8x7B)
- Dense model on A100 (e.g., Llama-2-7B)

Train the base coefficients on pooled data from all donors:

```python
from sklearn.linear_model import RidgeCV

# Pool data from all donor configurations
X_base = np.vstack([compute_base_features(traces, config) for config in donors])
X_config = np.vstack([compute_config_features(traces, config) for config in donors])
X_combined = np.hstack([X_base, X_config])

y_alpha = np.concatenate([t['scheduled_ts'] - t['queued_ts'] for t in donor_traces])

# Train combined base + config model
alpha_combined_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X_combined, y_alpha)

# Extract base and config coefficients
alpha_base_coef = alpha_combined_model.coef_[:len(F_alpha_base)]
alpha_config_coef = alpha_combined_model.coef_[len(F_alpha_base):]
```

**Stage 2: Calibrate Residuals on Target Configuration**

For a new configuration, collect minimal calibration data (~500-1000 observations) and fit only the residual coefficients:

```python
def calibrate_residuals(target_traces, target_config, base_coef, config_coef):
    # Compute base prediction using frozen coefficients
    F_base = compute_base_features(target_traces, target_config)
    F_config = compute_config_features(target_traces, target_config)
    base_prediction = F_base @ base_coef + F_config @ config_coef

    # Compute residuals
    y_actual = target_traces['scheduled_ts'] - target_traces['queued_ts']
    residuals = y_actual - base_prediction

    # Fit residual coefficients
    F_residual = compute_residual_features(target_traces, target_config)
    residual_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(F_residual, residuals)

    return residual_model.coef_
```

### Phase 3: Inference in BLIS

```go
// Hierarchical alpha computation
func ComputeAlphaLatency(req *Request, state *SimState, config *Config) float64 {
    // Base features (hardware-normalized)
    fBase := []float64{
        float64(state.QueueDepth) / float64(config.MaxNumSeqs),
        state.KVUsageRatio,
        float64(req.PromptTokens) / float64(config.MaxContextLength),
    }

    // Config features
    fConfig := []float64{
        1.0 / float64(config.TPDegree),
        boolToFloat(config.PrefixCaching),
        boolToFloat(config.ChunkedPrefill),
    }

    // Residual features (precomputed non-linear terms)
    normalizedQueue := float64(state.QueueDepth) / float64(config.MaxNumSeqs)
    fResidual := []float64{
        1.0,
        normalizedQueue * normalizedQueue,
    }

    return DotProduct(config.AlphaBase, fBase) +
           DotProduct(config.AlphaConfig, fConfig) +
           DotProduct(config.AlphaResidual, fResidual)
}
```

### Phase 4: Active Learning for Efficient Calibration

To minimize calibration data requirements, use uncertainty-guided sampling:

```python
def select_calibration_points(candidate_traces, base_model, n_samples=500):
    """Select traces that maximize expected information gain."""

    # Compute base predictions and uncertainty estimates
    predictions = base_model.predict(compute_features(candidate_traces))

    # Use residual variance as uncertainty proxy
    # (high residual = high uncertainty = high information value)
    estimated_residuals = np.abs(predictions - candidate_traces['actual_latency'])

    # Stratified sampling: ensure coverage of different operating regions
    strata = assign_strata(candidate_traces)  # By queue depth, KV ratio bins

    selected_indices = []
    for stratum in strata:
        stratum_indices = np.where(strata == stratum)[0]
        stratum_uncertainty = estimated_residuals[stratum_indices]
        # Select top-k highest uncertainty within each stratum
        top_k = stratum_indices[np.argsort(stratum_uncertainty)[-n_samples//len(strata):]]
        selected_indices.extend(top_k)

    return selected_indices
```

## Why This Addresses Reviewer Feedback

### Addresses "Hardcoded Thresholds" (from Idea 2)
- No regime boundaries to specify
- The base model learns smooth relationships across the entire operating range
- Residual coefficients adapt to configuration-specific quirks

### Addresses "Boundary Discontinuities" (from Idea 2)
- Single continuous model, no regime switching
- Non-linear effects captured via precomputed polynomial features
- Smooth predictions across all operating conditions

### Addresses "Quadratic Constraint Violation" (from Ideas 1 & 2)
- Quadratic terms are **features**, not coefficients
- `alpha · [x, x^2, ...]` is still a valid dot product
- Coefficients remain linear and interpretable

### Addresses "Training Data Requirements" (from Ideas 1 & 2)
- Base model trained once on rich donor data (can be expensive)
- New configurations require only ~500-1000 calibration observations
- Active learning further reduces calibration data needs

### Addresses "Feature Engineering Rigidity" (from Idea 1)
- Residual features explicitly capture unexplained variance
- Can add more residual features if systematic patterns emerge
- Hierarchical structure separates universal physics from config-specific effects

## Expected Benefits

| Aspect | PICF (Idea 1) | MoLE-RD (Idea 2) | HFD-TL (This Idea) |
|--------|---------------|------------------|---------------------|
| Calibration data | ~10,000 obs | ~2,000/regime | ~500-1,000 total |
| Regime boundaries | N/A | Hardcoded | None (continuous) |
| Non-linear effects | Limited | Regime-based | Precomputed features |
| New config time | Hours | ~30 min | ~5-10 min |
| Interpretability | High | High (per-regime) | High (per-hierarchy) |

## Training Data Requirements

**Donor configurations (one-time):**
- 3-5 representative (model, hardware, vLLM) combinations
- ~50,000 total step observations across donors
- Should span: dense/MoE, A100/H100, various TP degrees

**Target configurations (per new config):**
- 500-1,000 step observations for residual calibration
- ~100 request journey observations for alpha calibration
- Can be collected in ~5-10 minutes of traffic

## Limitations and Mitigations

**Limitation 1: Donor selection affects transfer quality**
- Mitigation: Include diverse donors spanning model types and hardware
- Validation: Test transfer to held-out configurations before deployment

**Limitation 2: Residual features may be insufficient for highly novel configurations**
- Mitigation: Expand residual feature set if transfer error exceeds threshold
- Fallback: Retrain as a new donor if configuration is fundamentally different

**Limitation 3: Assumes additive decomposition of latency**
- Mitigation: The additive structure is validated by the physics (queueing + processing)
- Empirical: If multiplicative effects emerge, can add interaction terms to residuals

---

## Reviewer Feedback

### Reviewer A (ML Systems Researcher)

**Scores:** Novelty: 6/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 7/10

**Strengths:**
- Low calibration overhead: Two-stage approach reduces data requirements from hours to minutes.
- Continuous predictions: Eliminates regime boundaries and discontinuity problems.
- Interpretable structure: Hierarchical decomposition preserves explainability.

**Weaknesses:**
- Donor selection is critical but underspecified: If target differs significantly from all donors, transfer may fail silently.
- Limited non-linearity: Only quadratic terms may be insufficient for complex interactions.
- Active learning is superficial: Residual magnitude as uncertainty proxy is naive.

**Suggestions:**
1. Add cross-validation across donors to quantify expected transfer error.
2. Use bootstrap ensembles or Gaussian process regression for better uncertainty estimation.

---

### Reviewer B (Systems Performance Engineer)

**Scores:** Novelty: 7/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 7/10

**Strengths:**
- Efficient calibration: 5-10 minute calibration time is a significant practical advantage.
- Graceful degradation with fallback: Retraining as new donor for novel configurations shows mature thinking.
- Maintains interpretability: Hierarchical coefficient structure preserves explainability.

**Weaknesses:**
- Donor selection is underspecified: Transfer quality could degrade for novel architectures.
- Residual features are minimal: Only constant and one quadratic term may be insufficient.
- No validation protocol: Lacks concrete holdout validation or error bounds.

**Suggestions:**
1. Define explicit transfer failure detection with RMSE thresholds triggering fallback.
2. Expand residual features with interaction terms like `queue_depth * kv_ratio`.

---

### Reviewer C (LLM Inference Optimization Researcher)

**Scores:** Novelty: 7/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 7/10

**Strengths:**
- Practical calibration efficiency: 500-1000 observation requirement is significant improvement.
- Maintains constraint compliance: Precomputed polynomial features solve quadratic constraint elegantly.
- Clean separation of concerns: Base/config/residual hierarchy provides principled isolation.

**Weaknesses:**
- Donor selection risk: Accuracy depends heavily on donor representativeness.
- Underspecified feature discovery: Lacks systematic method to identify insufficient residual features.
- Additive assumption fragility: Real-world latency often exhibits multiplicative interactions.

**Suggestions:**
1. Add quantitative thresholds (e.g., MAPE < 10%) for transfer acceptance vs. donor retraining.
2. Include interaction terms like `prefill_tokens * kv_usage_ratio` in base features to capture saturation effects.
