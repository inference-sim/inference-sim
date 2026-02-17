# Idea 5: Robust Ensemble with Configuration Embeddings (RECE)

## Core Insight

Reviewer feedback across all previous ideas revealed a common tension: physics-informed features provide interpretability and generalization, but struggle with (1) heavy-tailed latency distributions, (2) configuration-specific non-linearities, and (3) unspecified donor selection. This idea proposes a **robust ensemble** approach that:

1. Uses **quantile regression** to handle heavy-tailed distributions (median prediction + confidence intervals)
2. Learns **configuration embeddings** that capture configuration similarity without manual donor selection
3. Combines **multiple specialist models** via learned weights based on configuration proximity

This provides robustness, automatic configuration similarity, and maintains linear coefficient interpretability.

## Approach Overview

### Phase 1: Quantile Regression for Robust Predictions

Instead of predicting mean latency (which is sensitive to outliers), predict multiple quantiles:

**Quantile Loss Function:**
```python
def quantile_loss(y_true, y_pred, quantile):
    """Asymmetric loss for quantile regression."""
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
```

**Predict Three Quantiles:**
- `tau=0.5` (median): Primary prediction, robust to outliers
- `tau=0.1` (P10): Lower bound for confidence interval
- `tau=0.9` (P90): Upper bound for confidence interval

```python
from sklearn.linear_model import QuantileRegressor

# Train three models per coefficient
alpha_models = {
    'median': QuantileRegressor(quantile=0.5, alpha=1.0).fit(X, y),
    'lower': QuantileRegressor(quantile=0.1, alpha=1.0).fit(X, y),
    'upper': QuantileRegressor(quantile=0.9, alpha=1.0).fit(X, y),
}

# Prediction with confidence interval
def predict_alpha(features, models):
    return {
        'prediction': models['median'].predict(features),
        'ci_lower': models['lower'].predict(features),
        'ci_upper': models['upper'].predict(features),
    }
```

### Phase 2: Configuration Embeddings

Replace manual donor selection with learned embeddings that capture configuration similarity:

**Configuration Vector:**
```python
def compute_config_vector(config):
    """Extract configuration features for embedding."""
    return np.array([
        # Model features
        config.hidden_size / 4096,  # Normalized
        config.num_layers / 32,
        config.num_kv_heads / config.num_heads,  # GQA ratio
        float(config.is_moe),
        config.num_experts / 8 if config.is_moe else 0,

        # Hardware features
        config.hardware_tflops / 1000,
        config.hardware_bandwidth / 3000,
        config.memory_gb / 80,

        # vLLM features
        config.max_batch_tokens / 8192,
        config.max_num_seqs / 256,
        1 / config.tp_degree,
        float(config.chunked_prefill),
        float(config.prefix_caching),
        float(config.cpu_offload),
    ])
```

**Embedding Similarity:**
```python
from scipy.spatial.distance import cosine

def config_similarity(config_a, config_b):
    """Cosine similarity between configuration embeddings."""
    vec_a = compute_config_vector(config_a)
    vec_b = compute_config_vector(config_b)
    return 1 - cosine(vec_a, vec_b)
```

### Phase 3: Ensemble of Specialist Models

Train specialist models on clusters of similar configurations, then ensemble based on similarity:

**Cluster Training:**
```python
from sklearn.cluster import KMeans

def train_specialist_ensemble(all_traces, all_configs, n_specialists=5):
    """Train specialist models on configuration clusters."""

    # Compute configuration embeddings
    config_vectors = np.array([compute_config_vector(c) for c in all_configs])

    # Cluster configurations
    kmeans = KMeans(n_clusters=n_specialists, random_state=42)
    cluster_labels = kmeans.fit_predict(config_vectors)

    # Train specialist per cluster
    specialists = {}
    for cluster_id in range(n_specialists):
        cluster_mask = cluster_labels == cluster_id
        cluster_traces = [t for t, m in zip(all_traces, cluster_mask) if m]
        cluster_configs = [c for c, m in zip(all_configs, cluster_mask) if m]

        X = np.vstack([compute_features(t, c) for t, c in zip(cluster_traces, cluster_configs)])
        y = np.concatenate([t['latency'] for t in cluster_traces])

        specialists[cluster_id] = {
            'centroid': kmeans.cluster_centers_[cluster_id],
            'model': QuantileRegressor(quantile=0.5, alpha=1.0).fit(X, y),
        }

    return specialists, kmeans
```

**Weighted Ensemble Prediction:**
```python
def ensemble_predict(features, config, specialists, temperature=0.5):
    """Predict using similarity-weighted ensemble."""

    config_vec = compute_config_vector(config)
    weights = []
    predictions = []

    for cluster_id, specialist in specialists.items():
        # Compute similarity to cluster centroid
        similarity = 1 - cosine(config_vec, specialist['centroid'])
        weight = np.exp(similarity / temperature)  # Softmax-like weighting
        weights.append(weight)
        predictions.append(specialist['model'].predict(features))

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Weighted average prediction
    ensemble_pred = sum(w * p for w, p in zip(weights, predictions))

    return ensemble_pred, weights
```

### Phase 4: Expanded Feature Set with Communication Overhead

Address the missing communication features identified in Idea 4 feedback:

**Complete Alpha Features (20 features):**
```python
F_alpha_complete = [
    # Base queueing (from Ideas 1-4)
    1,  # Bias
    queue_depth / max_num_seqs,
    (queue_depth / max_num_seqs) ** 2,
    kv_usage_ratio,
    kv_usage_ratio ** 2,

    # Request features
    prompt_tokens / max_context_length,
    (prompt_tokens / max_context_length) ** 2,

    # Interaction terms
    queue_depth * kv_usage_ratio / max_num_seqs,
    prompt_tokens * kv_usage_ratio / max_context_length,

    # Configuration features
    1 / tp_degree,
    prefix_caching_enabled,
    chunked_prefill_enabled,

    # NEW: Communication overhead features
    tp_degree > 1,  # Binary: communication exists
    (tp_degree - 1) * prompt_tokens / interconnect_bandwidth,  # TP comm cost

    # Running batch features
    running_depth / max_num_seqs,
    running_depth * kv_usage_ratio / max_num_seqs,
]
```

**Complete Beta Features (22 features):**
```python
F_beta_complete = [
    # Base workload (from Ideas 1-4)
    1,  # Bias
    prefill_tokens * flops_per_token / hardware_tflops,
    decode_tokens * kv_bytes_per_token / hardware_bandwidth,
    batch_tokens / max_batch_tokens,

    # Non-linear workload
    (prefill_tokens / max_batch_tokens) ** 2,
    (decode_tokens / max_batch_tokens) ** 2,

    # Interaction terms
    prefill_tokens * decode_tokens / max_batch_tokens ** 2,
    prefill_tokens * kv_usage_ratio / max_batch_tokens,
    decode_tokens * running_depth / max_batch_tokens,

    # KV cache features
    kv_usage_ratio,
    kv_usage_ratio ** 2,

    # Configuration features
    1 / tp_degree,
    moe_indicator * num_active_experts / 8,
    cpu_offload_enabled,

    # NEW: Communication overhead features
    (tp_degree - 1) * batch_tokens / interconnect_bandwidth,  # All-reduce cost
    (tp_degree - 1) * running_depth / interconnect_latency,  # Sync cost

    # NEW: CPU offload features
    cpu_offload_enabled * kv_blocks_to_offload / pcie_bandwidth,
    cpu_offload_enabled * kv_blocks_to_restore / pcie_bandwidth,

    # MoE-specific features
    moe_indicator * expert_parallel_degree,
    moe_indicator * tokens_per_expert * expert_flops / hardware_tflops,
]
```

### Phase 5: Automatic Donor Diversity Metric

Define coverage criteria to guide configuration collection:

```python
def compute_coverage_score(existing_configs, candidate_config):
    """Score how much a candidate config adds to diversity."""

    candidate_vec = compute_config_vector(candidate_config)
    existing_vecs = np.array([compute_config_vector(c) for c in existing_configs])

    # Minimum distance to any existing config
    distances = [np.linalg.norm(candidate_vec - ev) for ev in existing_vecs]
    min_distance = min(distances)

    # High score = candidate is far from all existing configs = adds diversity
    return min_distance

def select_donors(candidate_configs, n_donors=5):
    """Greedy selection of diverse donors."""

    selected = [candidate_configs[0]]  # Start with first candidate

    while len(selected) < n_donors:
        scores = [compute_coverage_score(selected, c) for c in candidate_configs
                  if c not in selected]
        best_idx = np.argmax(scores)
        selected.append([c for c in candidate_configs if c not in selected][best_idx])

    return selected
```

### Phase 6: Inference in BLIS

```go
// Robust ensemble alpha computation
func ComputeAlphaLatency(req *Request, state *SimState, config *Config) AlphaResult {
    features := ComputeAlphaFeatures(req, state, config)

    // Compute similarity weights to specialist centroids
    configVec := ComputeConfigVector(config)
    weights := make([]float64, len(config.Specialists))
    totalWeight := 0.0

    for i, specialist := range config.Specialists {
        similarity := CosineSimilarity(configVec, specialist.Centroid)
        weights[i] = math.Exp(similarity / config.Temperature)
        totalWeight += weights[i]
    }

    // Normalize weights and compute ensemble prediction
    prediction := 0.0
    for i, specialist := range config.Specialists {
        weights[i] /= totalWeight
        prediction += weights[i] * DotProduct(specialist.Coefficients, features)
    }

    // Confidence interval from quantile models
    ciLower := DotProduct(config.QuantileLower, features)
    ciUpper := DotProduct(config.QuantileUpper, features)

    return AlphaResult{
        Prediction: prediction,
        CILower:    ciLower,
        CIUpper:    ciUpper,
        Weights:    weights,
    }
}
```

## Why This Addresses All Previous Feedback

### Addresses "Gaussian Assumption" (from Idea 4)
- Quantile regression is distribution-agnostic
- Median prediction (tau=0.5) is robust to outliers
- Confidence intervals capture heavy-tailed uncertainty

### Addresses "Donor Selection Manual" (from Ideas 3-4)
- Configuration embeddings automatically capture similarity
- Greedy diversity selection algorithm for donor collection
- Coverage metric guides data collection priorities

### Addresses "Missing Communication Features" (from Idea 4)
- Explicit TP communication features: `(tp_degree - 1) * tokens / bandwidth`
- CPU offload features: `kv_blocks_to_offload / pcie_bandwidth`
- Expert parallelism features for MoE

### Addresses "Shallow Calibration" (from Ideas 3-4)
- Ensemble of specialists adapts to configuration
- Similarity-based weighting provides soft interpolation
- No hard regime boundaries

### Maintains Linear Constraint
- Each specialist uses linear regression: `alpha · features`
- Ensemble combines linear predictions linearly
- Coefficients remain interpretable per specialist

## Expected Benefits

| Aspect | BSR-ARD (Idea 4) | RECE (This Idea) |
|--------|------------------|------------------|
| Distribution assumption | Gaussian | None (quantile) |
| Donor selection | Manual | Automatic (embeddings) |
| Transfer adaptation | Bias + scale | Weighted ensemble |
| Confidence intervals | z-score based | Quantile-based |
| Communication features | Partial | Complete |

## Training Data Requirements

**Configuration Collection (guided by diversity):**
- Use `select_donors()` to identify high-value configurations
- Target 5-8 diverse configurations across hardware/model/vLLM space
- ~10,000 observations per configuration (50,000-80,000 total)

**Specialist Training:**
- Cluster into 5 specialists via K-means
- Train quantile models (median, P10, P90) per specialist
- ~15 models total (5 specialists × 3 quantiles)

**New Configuration:**
- Compute similarity to specialist centroids
- No explicit calibration needed; ensemble adapts automatically
- Optionally fine-tune specialist weights with small calibration data

## Limitations and Mitigations

**Limitation 1: Ensemble complexity increases inference cost**
- Mitigation: At inference, only evaluate 2-3 highest-weight specialists
- Pre-compute weights for known configurations

**Limitation 2: K-means clustering may not be optimal**
- Mitigation: Use hierarchical clustering or DBSCAN if K-means fails
- Validate cluster quality via silhouette score

**Limitation 3: Quantile regression is slower to train**
- Mitigation: Use approximation algorithms (e.g., linear programming)
- Cache trained models; retraining is infrequent

---

## Reviewer Feedback

### Reviewer A (ML Systems Researcher)

**Scores:** Novelty: 6/10 | Feasibility: 7/10 | Technical Soundness: 7/10 | Completeness: 8/10

**Strengths:**
- Distribution-agnostic quantile regression eliminates Gaussian assumption weakness.
- Automatic configuration embedding removes manual donor selection burden.
- Comprehensive feature set explicitly addresses communication and offload overhead.

**Weaknesses:**
- Ensemble weighting violates the stated linear coefficient constraint at the system level.
- K-means clustering is sensitive to initialization and assumes spherical clusters.
- Data collection requirements (50K-80K samples) may not be practical for rare configurations.

**Suggestions:**
1. Replace K-means with simpler approach: use configuration embeddings as additional features in single quantile model.
2. Add explicit temporal validation and model drift detection.

---

### Reviewer B (Systems Performance Engineer)

**Scores:** Novelty: 7/10 | Feasibility: 7/10 | Technical Soundness: 6/10 | Completeness: 7/10

**Strengths:**
- Robustness via quantile regression addresses heavy-tailed latency distributions.
- Automatic donor selection provides principled guidance for data collection.
- Soft interpolation via similarity-weighted ensembles avoids hard regime boundaries.

**Weaknesses:**
- Euclidean clustering over heterogeneous features is methodologically weak.
- No handling of quantile crossing: independent training can produce inconsistent predictions.
- Training data volume (50K-80K) is substantial and may limit practical adoption.

**Suggestions:**
1. Use monotonic quantile regression or joint quantile networks to guarantee non-crossing.
2. Replace K-means with learned configuration representations via contrastive learning.

---

### Reviewer C (LLM Inference Optimization Researcher)

**Scores:** Novelty: 6/10 | Feasibility: 7/10 | Technical Soundness: 7/10 | Completeness: 8/10

**Strengths:**
- Robustness to outliers: Quantile regression provides meaningful confidence intervals.
- Automatic donor selection via configuration embedding approach.
- Comprehensive feature engineering with 20 alpha and 22 beta features.

**Weaknesses:**
- Model complexity vs. interpretability tradeoff: ensemble weighting undermines interpretability goal.
- K-means sensitivity to normalization choices; poor clusters may not correspond to meaningful regimes.
- Missing concrete cross-validation methodology for ensemble validation.

**Suggestions:**
1. Add embedding validation: verify configuration embeddings correlate with measured latency differences.
2. Consider sparse ensemble: use hard assignment to nearest 1-2 specialists to preserve interpretability.
