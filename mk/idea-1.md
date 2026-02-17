# Idea 1: Physics-Informed Compositional Features (PICF)

## Core Insight

The key limitation of both existing approaches is that they treat the mapping from configuration to latency as either purely empirical (blackbox) or purely analytical (roofline). This idea proposes a **middle ground**: use analytical formulas to derive physics-informed features, but learn the coefficients empirically. This gives the best of both worlds: the generalization of physics-based reasoning with the accuracy of data-driven learning.

## Approach Overview

### Phase 1: Feature Engineering via Analytical Decomposition

Instead of learning raw alpha/beta coefficients directly, we decompose latency into physically-meaningful components and create features that capture the underlying mechanics.

**For Alpha (Queue Latency):**

The queuing delay is fundamentally driven by:
1. **Service rate**: How fast are requests being processed?
2. **Queue depth**: How many requests are ahead?
3. **Resource contention**: Is KV cache constraining scheduling?

Feature vector `F_alpha`:
```
F_alpha = [
    # Queueing theory features
    queue_depth,
    queue_depth^2 / service_rate_proxy,  # Little's Law inspired

    # Resource contention features
    kv_pressure = 1 / (1 - kv_usage_ratio + ε),  # Diverges as cache fills
    kv_pressure × queue_depth,  # Interaction term

    # Request-specific scheduling features
    prompt_tokens / max_batch_tokens,  # Fraction of batch capacity
    prompt_tokens × running_depth / max_batch_tokens,  # Schedulability

    # Configuration features (normalized)
    1 / max_num_seqs,  # Inverse capacity
    chunked_prefill_enabled,  # Binary: allows interleaving
]
```

The `service_rate_proxy` is derived analytically from hardware specs:
```
service_rate_proxy = hardware_bandwidth / (model_kv_per_token × avg_batch_size_estimate)
```

**For Beta (Step Latency):**

Step time is governed by compute vs. memory trade-offs that vary by phase:

Feature vector `F_beta`:
```
F_beta = [
    # Prefill features (compute-bound)
    prefill_tokens × model_flops_per_token / hardware_tflops,
    prefill_tokens^2 / (tp_degree × hardware_tflops),  # Attention quadratic term

    # Decode features (memory-bound)
    decode_tokens × kv_blocks_per_token / hardware_bandwidth,
    decode_tokens × running_depth,  # More requests = more KV reads

    # Mixed workload features
    prefill_tokens × decode_tokens / batch_tokens^2,  # Competition for resources

    # MoE-specific features (0 for dense models)
    moe_experts_per_token × decode_tokens / hardware_bandwidth,

    # Hardware interaction terms
    batch_tokens / (tp_degree × hardware_sm_count),  # Parallelism efficiency

    # KV cache features
    kv_usage_ratio × decode_tokens,  # Cache access patterns
]
```

Physics-derived feature helpers:
```
model_flops_per_token = 2 × hidden_size × (3 × intermediate_size + vocab_size)
model_kv_per_token = 2 × num_layers × (hidden_size / num_heads) × num_kv_heads × dtype_bytes
```

### Phase 2: Training Pipeline

**Data Preparation:**
1. Collect traces from vLLM with full tracing enabled
2. Join journey traces (for alpha targets) with step traces (for beta targets)
3. Compute derived features from config.json + hardware_config.json
4. Normalize features per hardware type to handle scale differences

**Model Training (Ridge Regression with Cross-Validation):**

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Alpha model
X_alpha = compute_alpha_features(traces, configs)
y_alpha = traces['scheduled_ts'] - traces['queued_ts']
alpha_scaler = StandardScaler().fit(X_alpha)
alpha_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(
    alpha_scaler.transform(X_alpha), y_alpha
)
alpha_coefficients = alpha_model.coef_ / alpha_scaler.scale_

# Beta model
X_beta = compute_beta_features(step_traces, configs)
y_beta = step_traces['step_duration_us']
beta_scaler = StandardScaler().fit(X_beta)
beta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(
    beta_scaler.transform(X_beta), y_beta
)
beta_coefficients = beta_model.coef_ / beta_scaler.scale_
```

**Robustness Measures:**
- Ridge regularization prevents overfitting to noisy observations
- Cross-validation selects regularization strength automatically
- Temporal train/test split ensures generalization to future workloads
- Feature normalization per hardware type allows single model across hardware

### Phase 3: Coefficient Interpretation

The learned coefficients have interpretable meanings:
- Large coefficient for `queue_depth^2 / service_rate_proxy` indicates queueing effects dominate
- Large coefficient for `prefill_tokens^2` indicates attention compute is bottleneck
- Ratio of prefill vs decode coefficients indicates workload sensitivity

## Why This Approach Works

### Handles Model Diversity
- Dense vs MoE: `moe_experts_per_token` feature is zero for dense, non-zero for MoE
- GQA/MQA: Captured in `model_kv_per_token` calculation via `num_kv_heads`

### Handles Workload Diversity
- Prefill-heavy: High `prefill_tokens` features dominate
- Decode-heavy: High `decode_tokens` features dominate
- Mixed: Interaction term `prefill_tokens × decode_tokens` captures competition

### Handles Hardware Diversity
- Features are normalized by hardware specs (tflops, bandwidth)
- Same coefficients work across A100/H100 because physics is embedded in features

### Handles vLLM Knobs
- `max_batch_tokens`, `max_num_seqs` appear directly in features
- `chunked_prefill_enabled` is a binary feature
- `tensor_parallel_size` normalizes compute features

## Training Data Requirements

Minimum for robust training:
- 3-5 different model sizes (7B, 13B, 70B range)
- 2+ hardware types (A100, H100)
- 3+ workload patterns (prefill-heavy, decode-heavy, balanced)
- ~10,000 step observations per configuration

## Implementation in BLIS

```go
// Alpha feature computation
func ComputeAlphaFeatures(req *Request, state *SimState, config *Config) []float64 {
    serviceRateProxy := config.Hardware.Bandwidth /
        (config.Model.KVPerToken * state.AvgBatchSize)
    kvPressure := 1.0 / (1.0 - state.KVUsageRatio + 1e-6)

    return []float64{
        float64(state.QueueDepth),
        math.Pow(float64(state.QueueDepth), 2) / serviceRateProxy,
        kvPressure,
        kvPressure * float64(state.QueueDepth),
        float64(req.PromptTokens) / float64(config.MaxBatchTokens),
        float64(req.PromptTokens * state.RunningDepth) / float64(config.MaxBatchTokens),
        1.0 / float64(config.MaxNumSeqs),
        boolToFloat(config.ChunkedPrefill),
    }
}
```

## Expected Improvements Over Existing Approaches

| Aspect | Blackbox | Roofline | PICF (This Approach) |
|--------|----------|----------|----------------------|
| New config calibration | Hours | None | Minutes (feature computation only) |
| Accuracy | High (on trained configs) | Medium | High |
| Interpretability | Low | High | High |
| Generalization | Poor | Good | Good |
| MoE support | Requires retraining | Manual | Automatic |

## Limitations and Mitigations

**Limitation 1: Requires diverse training data**
- Mitigation: Start with available traces, expand coverage incrementally

**Limitation 2: Physics features may miss non-obvious effects**
- Mitigation: Include residual term or add learned features discovered via exploration

**Limitation 3: Feature engineering requires domain expertise**
- Mitigation: Features are derived from published roofline principles; well-documented

---

## Reviewer Feedback

### Reviewer A (ML Systems Researcher)

**Scores:** Novelty: 6/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 7/10

**Strengths:**
- Interpretable coefficients: Unlike blackbox, the learned coefficients have physical meaning, aiding debugging and trust-building.
- Hardware-agnostic via normalization: Features scaled by hardware specs should generalize across GPUs without retraining.
- Practical training pipeline: Clear, implementable specification with standard ML tools.

**Weaknesses:**
- Linear model limitation: Many real effects (kernel launch latencies, memory hierarchy, batching thresholds) are non-linear; may need polynomial features.
- Feature engineering burden: Requires manual derivation of features; misses non-obvious effects by design.
- Incomplete coverage: CPU offloading, expert parallelism, and prefix caching interactions are underspecified.

**Suggestions:**
1. Add a residual modeling layer (e.g., small gradient-boosted tree on residuals) to capture non-linear effects the physics features miss.
2. Explicitly specify how prefix cache hits reduce alpha latency and how preemption events should be handled in feature computation.

---

### Reviewer B (Systems Performance Engineer)

**Scores:** Novelty: 6/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 7/10

**Strengths:**
- Interpretability: The physics-grounded features provide meaningful coefficients that can be inspected and debugged, unlike pure blackbox approaches.
- Transfer efficiency: Hardware normalization allows a single trained model to generalize across GPU types without retraining.
- Practical training pipeline: The sklearn-based approach with RidgeCV is production-ready and robust against overfitting via regularization.

**Weaknesses:**
- Feature completeness gap: Missing features for CPU offloading, expert parallelism, and prefix caching effects. The MoE features are simplistic.
- Alpha model scope confusion: The feature vector models scheduling delay, not the arrival-to-queuing delay explicitly requested in the problem statement.
- Training data burden: Requiring 10,000 observations across 3-5 models and 2+ hardware types is substantial.

**Suggestions:**
1. Separate alpha into two components: Model arrival-to-queuing (API overhead) separately from queuing-to-scheduling (the features currently proposed).
2. Add prefix caching features: Include `prefix_hit_ratio` and `effective_prompt_length` in both alpha and beta feature vectors.

---

### Reviewer C (LLM Inference Optimization Researcher)

**Scores:** Novelty: 7/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 8/10

**Strengths:**
- Strong generalization principle: By embedding hardware/model physics into features rather than learning them, the approach should transfer to new configurations with minimal retraining.
- Interpretable coefficients: The learned coefficients have physical meaning, aiding debugging and validation.
- Practical training pipeline: Ridge regression with temporal cross-validation is robust and well-understood.

**Weaknesses:**
- Linear model limitation: The dot-product formulation cannot capture all non-linear interactions (e.g., threshold effects under high KV pressure).
- Feature engineering rigidity: Hand-crafted features may miss platform-specific behaviors (CUDA graph overhead, prefix caching warm-up).
- Training data burden understated: The claim of "minutes" for new config calibration assumes pre-trained coefficients generalize.

**Suggestions:**
1. Add non-linear transformations: Consider polynomial feature expansion or piecewise-linear terms for KV pressure.
2. Include a small residual network or additive bias per configuration to capture platform-specific effects.
