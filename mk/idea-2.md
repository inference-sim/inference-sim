# Idea 2: Mixture of Linear Experts with Regime Detection (MoLE-RD)

## Core Insight

Reviewer feedback on Idea 1 (PICF) consistently identified the **linear model limitation** as a key weakness: real LLM inference systems exhibit non-linear behaviors such as threshold effects under KV pressure, discontinuities at batch size boundaries, and regime-dependent performance characteristics. Rather than adding polynomial features (which explode combinatorially), this idea proposes **learning multiple linear models, each specialized for a distinct operational regime**, with a lightweight regime detector that routes observations to the appropriate expert.

This maintains the interpretability and constraint-compatibility of linear alpha/beta coefficients while capturing non-linear system behavior through mixture composition.

## Approach Overview

### Phase 1: Regime Identification

LLM inference systems operate in distinct regimes with different performance characteristics:

**Identified Regimes for Alpha (Queueing Latency):**
1. **Low-load regime**: Queue empty or near-empty, immediate scheduling, alpha ≈ constant
2. **Normal-load regime**: Steady-state queueing, alpha ∝ queue_depth × service_time
3. **High-pressure regime**: KV cache near-full, scheduling constrained by memory, alpha dominated by kv_pressure
4. **Preemption regime**: Active preemptions occurring, alpha includes preemption overhead

**Identified Regimes for Beta (Step Latency):**
1. **Prefill-dominated**: batch_prefill_tokens >> batch_decode_tokens, compute-bound
2. **Decode-dominated**: batch_decode_tokens >> batch_prefill_tokens, memory-bound
3. **Mixed-balanced**: Comparable prefill and decode, competition for resources
4. **Saturation regime**: KV cache > 90%, performance degradation due to fragmentation

### Phase 2: Regime Detector (Gating Function)

A simple decision tree or logistic regression classifier determines which regime applies:

**Alpha Regime Detector:**
```python
def detect_alpha_regime(state):
    if state.queue_depth <= 2:
        return "low_load"
    elif state.kv_usage_ratio > 0.85:
        return "high_pressure"
    elif state.num_recent_preemptions > 0:
        return "preemption"
    else:
        return "normal_load"
```

**Beta Regime Detector:**
```python
def detect_beta_regime(batch):
    prefill_ratio = batch.prefill_tokens / (batch.prefill_tokens + batch.decode_tokens + 1e-6)
    if prefill_ratio > 0.7:
        return "prefill_dominated"
    elif prefill_ratio < 0.3:
        return "decode_dominated"
    elif batch.kv_usage_ratio > 0.9:
        return "saturation"
    else:
        return "mixed_balanced"
```

### Phase 3: Regime-Specific Feature Vectors

Each regime has its own feature vector optimized for that regime's dynamics:

**Alpha Feature Vectors:**

```python
# Low-load regime: minimal queueing, focus on API overhead
F_alpha_low = [
    1,  # constant (API baseline)
    prompt_tokens / 1000,  # tokenization overhead
    prefix_cache_miss_ratio,  # cold-start overhead
]

# Normal-load regime: queueing theory applies
F_alpha_normal = [
    queue_depth,
    queue_depth * avg_service_time,  # Little's Law
    prompt_tokens / max_batch_tokens,
    running_depth / max_num_seqs,
]

# High-pressure regime: KV constraints dominate
F_alpha_high = [
    1 / (1 - kv_usage_ratio + 0.01),  # Divergent pressure
    queue_depth * kv_usage_ratio,
    prompt_tokens * kv_usage_ratio,  # Large prompts harder to schedule
]

# Preemption regime: includes preemption overhead
F_alpha_preempt = [
    queue_depth,
    num_preemptions,
    preempted_tokens / 1000,  # Wasted work
    kv_usage_ratio,
]
```

**Beta Feature Vectors:**

```python
# Prefill-dominated: compute-bound behavior
F_beta_prefill = [
    prefill_tokens * flops_per_token / hardware_tflops,
    prefill_tokens**2 / (tp_degree * hardware_tflops),  # Attention
    num_prefill_reqs,  # Per-request overhead
    moe_active_experts * prefill_tokens / hardware_tflops,  # MoE compute
]

# Decode-dominated: memory-bound behavior
F_beta_decode = [
    decode_tokens * kv_bytes_per_token / hardware_bandwidth,
    decode_tokens * running_depth,  # KV reads scale with batch
    decode_tokens * moe_expert_count / hardware_bandwidth,  # MoE memory
    cpu_offload_enabled * offload_transfer_time,  # CPU offload overhead
]

# Mixed-balanced: both matter
F_beta_mixed = [
    prefill_tokens / hardware_tflops,
    decode_tokens / hardware_bandwidth,
    prefill_tokens * decode_tokens / batch_tokens**2,  # Competition
    running_depth * kv_usage_ratio,
]

# Saturation regime: fragmentation and overhead dominate
F_beta_saturation = [
    batch_tokens / hardware_bandwidth,  # Baseline
    kv_fragmentation_ratio * batch_tokens,  # Fragmentation overhead
    num_evictions * block_transfer_time,  # Eviction overhead
    1 / (1 - kv_usage_ratio + 0.01),  # Pressure term
]
```

### Phase 4: Training Pipeline

**Step 1: Regime Labeling**
Label each trace observation with its regime based on the regime detector.

**Step 2: Per-Regime Linear Regression**
Train separate Ridge regression models for each regime:

```python
from sklearn.linear_model import RidgeCV

# Group observations by regime
alpha_regimes = group_by_regime(alpha_traces, detect_alpha_regime)
beta_regimes = group_by_regime(beta_traces, detect_beta_regime)

# Train per-regime models
alpha_models = {}
for regime, traces in alpha_regimes.items():
    X = compute_alpha_features(traces, regime)
    y = traces['scheduled_ts'] - traces['queued_ts']
    alpha_models[regime] = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X, y)

beta_models = {}
for regime, traces in beta_regimes.items():
    X = compute_beta_features(traces, regime)
    y = traces['step_duration_us']
    beta_models[regime] = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X, y)
```

**Step 3: Regime Detector Refinement (Optional)**
If rule-based detection is insufficient, train a classifier on the same data:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Features for regime classification (subset of full features)
regime_features = ['queue_depth', 'kv_usage_ratio', 'prefill_ratio', 'num_preemptions']

alpha_regime_clf = GradientBoostingClassifier(n_estimators=50, max_depth=3).fit(
    alpha_traces[regime_features], alpha_traces['true_regime']
)
```

### Phase 5: Inference in BLIS

```go
// Alpha computation with regime detection
func ComputeAlphaLatency(req *Request, state *SimState, config *Config) float64 {
    regime := DetectAlphaRegime(state)
    features := ComputeAlphaFeatures(req, state, config, regime)
    coefficients := alphaCoefficients[regime]
    return DotProduct(coefficients, features)
}

func DetectAlphaRegime(state *SimState) string {
    if state.QueueDepth <= 2 {
        return "low_load"
    } else if state.KVUsageRatio > 0.85 {
        return "high_pressure"
    } else if state.RecentPreemptions > 0 {
        return "preemption"
    }
    return "normal_load"
}
```

## Why This Addresses Reviewer Feedback

### Addresses "Linear Model Limitation"
- Non-linear behavior captured through **regime switching**, not polynomial expansion
- Each regime uses the most appropriate linear approximation for that operating region
- Threshold effects (e.g., KV saturation) handled by dedicated saturation regime

### Addresses "Feature Completeness Gap"
- **CPU offloading**: Explicit `cpu_offload_enabled * offload_transfer_time` feature in decode regime
- **Expert parallelism**: `moe_active_experts` and `moe_expert_count` features in both prefill and decode
- **Prefix caching**: `prefix_cache_miss_ratio` feature in low-load alpha regime

### Addresses "Alpha Model Scope Confusion"
- Low-load regime explicitly models **arrival-to-queuing** (API overhead + tokenization)
- Other regimes model **queuing-to-scheduling** as originally intended
- Separation is natural because these regimes have different dominant factors

### Addresses "Feature Engineering Rigidity"
- Regimes can be refined based on data; more regimes added as needed
- Per-regime features are smaller and more targeted, easier to validate
- Optional classifier-based regime detection learns boundaries from data

## Expected Benefits

| Aspect | PICF (Idea 1) | MoLE-RD (This Idea) |
|--------|---------------|---------------------|
| Non-linear effects | Limited (linear only) | Captured via regime switching |
| Interpretability | High | High (per-regime) |
| Edge cases (saturation, preemption) | Underspecified | Explicit regimes |
| Training complexity | Single model | Multiple smaller models |
| Feature count | 8+ features | 3-4 features per regime |

## Training Data Requirements

- Same as PICF: 3-5 model sizes, 2+ hardware types, diverse workloads
- Additional: Ensure coverage of all regimes (especially edge cases like saturation)
- ~2,000-3,000 observations per regime (less per regime than single-model approach)

## Limitations

**Limitation 1: Regime boundaries may be discontinuous**
- Mitigation: Use soft gating (weighted average of regime predictions) near boundaries

**Limitation 2: More coefficients to maintain (4 alpha + 4 beta sets)**
- Mitigation: Still interpretable and can be stored compactly; training is parallelizable

**Limitation 3: Regime detector errors propagate to latency errors**
- Mitigation: Classifier accuracy can be validated independently; boundary cases can use ensemble

---

## Reviewer Feedback

### Reviewer A (ML Systems Researcher)

**Scores:** Novelty: 6/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 7/10

**Strengths:**
- Interpretability preserved: Each regime has 3-4 targeted features that map to physical system behaviors.
- Explicit edge case handling: Saturation and preemption regimes directly address failure modes.
- Practical training: Smaller per-regime datasets are easier to collect and validate.

**Weaknesses:**
- Discontinuity at regime boundaries: Hard switching can produce prediction jumps.
- Threshold selection problem: The thresholds (0.85, 0.9, 0.7, 0.3) appear heuristic and may vary across configurations.
- Feature constraint violation: The prefill_tokens^2 term breaks the linear dot-product requirement.

**Suggestions:**
1. Implement soft gating explicitly with sigmoid weighting for boundary transitions.
2. Remove or justify the quadratic term to satisfy dot-product constraints.

---

### Reviewer B (Systems Performance Engineer)

**Scores:** Novelty: 6/10 | Feasibility: 7/10 | Technical Soundness: 7/10 | Completeness: 7/10

**Strengths:**
- Interpretability preserved: Each regime has 3-4 targeted features with clear physical meanings.
- Natural handling of edge cases: Saturation and preemption regimes explicitly model failure modes.
- Pragmatic complexity: Avoids polynomial feature explosion while capturing non-linearities.

**Weaknesses:**
- Discontinuity at boundaries: Hard regime transitions can cause prediction jumps; soft-gating may violate linear constraints.
- Data imbalance risk: Rare regimes (saturation, preemption) may have insufficient training data.
- Regime detector coupling: Errors in regime classification directly corrupt latency predictions.

**Suggestions:**
1. Implement boundary smoothing with explicit interpolation weights in overlap regions.
2. Add synthetic data augmentation by intentionally inducing saturation/preemption during profiling.

---

### Reviewer C (LLM Inference Optimization Researcher)

**Scores:** Novelty: 7/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 8/10

**Strengths:**
- Maintains interpretability: Each regime has 3-4 features with clear physical meanings.
- Principled handling of nonlinearity: Regime switching captures threshold effects without combinatorial complexity.
- Modular training and maintenance: Per-regime models can be updated independently.

**Weaknesses:**
- Hardcoded regime thresholds: Thresholds are presented without empirical justification and may not transfer across configurations.
- Regime boundary discontinuities: The soft gating mitigation is underspecified.
- Increased validation burden: 8 separate linear models require more careful experimental design.

**Suggestions:**
1. Learn regime boundaries using clustering or breakpoint detection, then validate across hardware types.
2. Implement weighted ensemble at boundaries with explicit transition zones (e.g., 0.80-0.90 for KV pressure).
