# Idea 4: Bayesian Sparse Regression with Automatic Relevance Determination (BSR-ARD)

## Core Insight

Reviewer feedback on Idea 3 consistently identified three gaps: (1) limited residual features, (2) no principled uncertainty quantification, and (3) no validation protocol for detecting transfer failures. This idea proposes using **Bayesian sparse regression with Automatic Relevance Determination (ARD)** which addresses all three issues:

1. **Sparse feature selection**: ARD automatically prunes irrelevant features, allowing us to include a large candidate set without overfitting
2. **Uncertainty quantification**: Bayesian inference provides predictive distributions, not just point estimates
3. **Transfer failure detection**: High predictive uncertainty signals when the model is extrapolating beyond training data

This maintains the linear dot-product structure while providing principled regularization and uncertainty bounds.

## Approach Overview

### Phase 1: Expanded Candidate Feature Set

Instead of hand-selecting a minimal feature set, include a comprehensive candidate set and let ARD prune irrelevant features:

**Alpha Candidate Features (18 features):**
```python
F_alpha_candidates = [
    # Base queueing features
    1,  # Constant/bias
    queue_depth,
    queue_depth / max_num_seqs,
    queue_depth ** 2 / max_num_seqs ** 2,  # Precomputed

    # KV cache features
    kv_usage_ratio,
    kv_usage_ratio ** 2,  # Saturation effect
    1 / (1 - kv_usage_ratio + 0.05),  # Pressure term (bounded)

    # Request features
    prompt_tokens / max_context_length,
    (prompt_tokens / max_context_length) ** 2,

    # Interaction terms
    queue_depth * kv_usage_ratio / max_num_seqs,
    prompt_tokens * kv_usage_ratio / max_context_length,
    prompt_tokens * queue_depth / (max_context_length * max_num_seqs),

    # Configuration features
    1 / tp_degree,
    prefix_caching_enabled,
    chunked_prefill_enabled,
    prefix_caching_enabled * kv_usage_ratio,  # Cache hit effect under pressure
    chunked_prefill_enabled * queue_depth / max_num_seqs,

    # Running batch features
    running_depth / max_num_seqs,
]

F_beta_candidates = [
    # Base workload features
    1,  # Constant/bias
    prefill_tokens * flops_per_token / hardware_tflops,
    decode_tokens * kv_bytes_per_token / hardware_bandwidth,
    batch_tokens / max_batch_tokens,

    # Non-linear workload features
    (prefill_tokens / max_batch_tokens) ** 2,  # Attention scaling
    (decode_tokens / max_batch_tokens) ** 2,

    # Interaction terms
    prefill_tokens * decode_tokens / max_batch_tokens ** 2,  # Mixed batch effect
    prefill_tokens * kv_usage_ratio / max_batch_tokens,
    decode_tokens * running_depth / max_batch_tokens,

    # KV cache features
    kv_usage_ratio,
    kv_usage_ratio ** 2,  # Saturation

    # Configuration features
    1 / tp_degree,
    moe_indicator * num_active_experts / 8,
    cpu_offload_enabled,
    moe_indicator * decode_tokens / hardware_bandwidth,  # MoE decode

    # Request count features
    num_prefill_reqs,
    num_decode_reqs,
    running_depth / max_num_seqs,
]
```

### Phase 2: Bayesian Sparse Regression with ARD

ARD places independent precision hyperparameters on each coefficient, allowing the model to learn which features are relevant:

```python
import numpy as np
from scipy.optimize import minimize

class BayesianARDRegression:
    """
    Bayesian Linear Regression with Automatic Relevance Determination.
    Implements the ARD prior: p(w_i | α_i) = N(0, 1/α_i)
    where α_i is the precision (inverse variance) for each weight.
    """

    def __init__(self, n_features, prior_alpha=1.0, noise_precision=1.0):
        self.n_features = n_features
        self.alpha = np.ones(n_features) * prior_alpha  # Per-feature precision
        self.beta = noise_precision  # Observation noise precision

    def fit(self, X, y, max_iter=100, tol=1e-4):
        """Fit using Evidence Maximization (Type-II ML)."""
        n_samples = X.shape[0]

        for iteration in range(max_iter):
            # Compute posterior covariance and mean
            A = np.diag(self.alpha)
            Sigma_inv = A + self.beta * X.T @ X
            Sigma = np.linalg.inv(Sigma_inv)
            mu = self.beta * Sigma @ X.T @ y

            # Update hyperparameters using evidence maximization
            gamma = 1 - self.alpha * np.diag(Sigma)  # Effective number of params
            alpha_new = gamma / (mu ** 2 + 1e-10)
            alpha_new = np.clip(alpha_new, 1e-6, 1e6)  # Numerical stability

            residuals = y - X @ mu
            beta_new = (n_samples - np.sum(gamma)) / (residuals @ residuals + 1e-10)
            beta_new = np.clip(beta_new, 1e-6, 1e6)

            # Check convergence
            if np.max(np.abs(alpha_new - self.alpha)) < tol:
                break

            self.alpha = alpha_new
            self.beta = beta_new

        self.mean = mu
        self.covariance = Sigma
        self.relevant_features = self.alpha < 1e3  # Features with small alpha are relevant

    def predict(self, X, return_std=False):
        """Predict with optional uncertainty."""
        mu = X @ self.mean

        if return_std:
            # Predictive variance = noise + model uncertainty
            var = 1/self.beta + np.sum((X @ self.covariance) * X, axis=1)
            return mu, np.sqrt(var)
        return mu

    def get_sparse_coefficients(self):
        """Return coefficients, zeroing out irrelevant features."""
        coef = self.mean.copy()
        coef[~self.relevant_features] = 0.0
        return coef
```

### Phase 3: Training Pipeline with Uncertainty-Aware Validation

**Stage 1: Train on Donor Configurations**

```python
def train_bayesian_model(donor_traces, configs):
    """Train ARD model on pooled donor data."""

    # Compute all candidate features
    X = np.vstack([compute_alpha_candidates(t, c) for t, c in zip(donor_traces, configs)])
    y = np.concatenate([t['scheduled_ts'] - t['queued_ts'] for t in donor_traces])

    # Fit Bayesian ARD regression
    model = BayesianARDRegression(n_features=X.shape[1])
    model.fit(X, y, max_iter=200)

    # Report which features were selected
    selected = np.where(model.relevant_features)[0]
    print(f"ARD selected {len(selected)} of {X.shape[1]} features: {selected}")

    return model
```

**Stage 2: Validate Transfer with Uncertainty**

```python
def validate_transfer(model, target_traces, target_config, threshold_z=2.0):
    """Check if transfer is valid using predictive uncertainty."""

    X = compute_alpha_candidates(target_traces, target_config)
    y_actual = target_traces['scheduled_ts'] - target_traces['queued_ts']

    # Get predictions with uncertainty
    y_pred, y_std = model.predict(X, return_std=True)

    # Compute z-scores: how many std deviations from prediction?
    z_scores = np.abs(y_actual - y_pred) / y_std

    # Flag transfer failure if too many outliers
    outlier_rate = np.mean(z_scores > threshold_z)
    if outlier_rate > 0.1:  # More than 10% outliers
        return False, outlier_rate

    return True, outlier_rate
```

**Stage 3: Calibrate or Retrain**

```python
def calibrate_or_retrain(model, target_traces, target_config):
    """Calibrate residuals or trigger full retraining."""

    transfer_valid, outlier_rate = validate_transfer(model, target_traces, target_config)

    if transfer_valid:
        # Calibrate: fit residual bias and variance scaling
        X = compute_alpha_candidates(target_traces, target_config)
        y_actual = target_traces['scheduled_ts'] - target_traces['queued_ts']
        y_pred, _ = model.predict(X, return_std=True)

        residuals = y_actual - y_pred
        bias = np.mean(residuals)
        scale = np.std(residuals) / np.std(y_pred) if np.std(y_pred) > 0 else 1.0

        return {"calibration": "bias_shift", "bias": bias, "scale": scale}
    else:
        # Transfer failed: retrain with target data added to donors
        return {"calibration": "retrain_required", "outlier_rate": outlier_rate}
```

### Phase 4: Inference in BLIS with Uncertainty

```go
// Alpha computation with uncertainty
func ComputeAlphaLatency(req *Request, state *SimState, config *Config) (float64, float64) {
    features := ComputeAlphaCandidates(req, state, config)

    // Sparse dot product (only relevant features)
    prediction := 0.0
    for i, coef := range config.AlphaCoefficients {
        if config.AlphaRelevant[i] {
            prediction += coef * features[i]
        }
    }

    // Apply calibration
    prediction = prediction * config.AlphaScale + config.AlphaBias

    // Compute uncertainty (simplified: use precomputed average variance)
    uncertainty := config.AlphaBaseUncertainty * math.Sqrt(ComputeFeatureVariance(features, config))

    return prediction, uncertainty
}
```

## Why This Addresses Reviewer Feedback

### Addresses "Limited Residual Features" (from Idea 3)
- Start with 18+ candidate features; ARD prunes irrelevant ones
- No manual feature selection required
- Can easily add more candidates without risk of overfitting

### Addresses "No Uncertainty Quantification" (from Idea 3)
- Bayesian inference provides predictive variance, not just point estimates
- Uncertainty increases when extrapolating beyond training data
- Principled probabilistic framework

### Addresses "No Validation Protocol" (from Idea 3)
- Z-score-based outlier detection identifies transfer failures
- Quantitative threshold (>10% outliers → retrain) is explicit
- Automatic decision between calibration and retraining

### Addresses "Donor Selection Underspecified" (from Idea 3)
- Uncertainty quantification flags when donors are insufficient
- Leave-one-donor-out cross-validation can validate donor coverage
- High uncertainty on new configs triggers retraining

### Maintains Linear Constraint
- ARD is still linear regression: `alpha · features`
- Coefficients are learned, features are precomputed (including polynomials)
- Sparse coefficients set irrelevant features to zero

## Expected Benefits

| Aspect | HFD-TL (Idea 3) | BSR-ARD (This Idea) |
|--------|------------------|---------------------|
| Feature selection | Manual (3-8 features) | Automatic (18+ candidates) |
| Uncertainty | None | Per-prediction std |
| Transfer validation | None | Z-score outlier detection |
| Overfitting risk | Low (few features) | Low (ARD regularization) |
| Calibration decision | Manual | Automatic (threshold-based) |

## Training Data Requirements

**Donor configurations (one-time):**
- Same as Idea 3: 3-5 representative configurations
- ~50,000 total step observations
- ARD fitting takes ~1 minute (200 iterations)

**Target configurations (per new config):**
- 500-1000 observations for transfer validation
- If transfer passes: only bias/scale calibration needed
- If transfer fails: add to donor pool and retrain

## Limitations and Mitigations

**Limitation 1: ARD may be computationally expensive at scale**
- Mitigation: Pre-compute covariance matrix offline; inference is just matrix-vector product
- Alternative: Use approximate inference (variational Bayes) if needed

**Limitation 2: Gaussian assumption may not hold for heavy-tailed latency distributions**
- Mitigation: Log-transform latency to normalize distribution
- Alternative: Use robust regression (Student-t likelihood) if Gaussian fails

**Limitation 3: Feature engineering still required for candidate set**
- Mitigation: Include exhaustive physics-informed candidates; ARD will prune
- The candidate set is a superset of all features from Ideas 1-3

---

## Reviewer Feedback

### Reviewer A (ML Systems Researcher)

**Scores:** Novelty: 6/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 8/10

**Strengths:**
- Principled feature selection: ARD elegantly solves feature explosion by automatically shrinking irrelevant coefficients.
- Uncertainty quantification: Predictive uncertainty provides actionable signals for transfer failure detection.
- Backward compatibility: Maintains required linear dot-product structure.

**Weaknesses:**
- Gaussian likelihood mismatch: Latency distributions are typically right-skewed and heavy-tailed.
- Shallow calibration: Bias + scale may be insufficient when feature interactions differ across hardware.
- Transfer validation threshold is arbitrary: 10% outlier threshold and z=2.0 cutoff lack justification.

**Suggestions:**
1. Use log-transformed latency by default; alternatively, implement Student-t likelihood.
2. Add leave-one-hardware-out cross-validation during donor training.

---

### Reviewer B (Systems Performance Engineer)

**Scores:** Novelty: 6/10 | Feasibility: 8/10 | Technical Soundness: 7/10 | Completeness: 7/10

**Strengths:**
- Principled regularization: ARD provides automatic feature selection without manual tuning.
- Actionable uncertainty: Predictive variance gives operators quantitative signal for retraining.
- Backward compatible: Maintains linear coefficient structure required by BLIS.

**Weaknesses:**
- Gaussian assumption fragility: Latency distributions are heavy-tailed with occasional outliers.
- Donor selection still manual: Choice of donor configurations remains unspecified.
- Calibration is shallow: Cannot correct non-linear transfer errors or feature-dependent biases.

**Suggestions:**
1. Use log-normal or Student-t likelihood for heavy-tailed latencies.
2. Add donor diversity metric (feature space coverage) to guide donor selection.

---

### Reviewer C (LLM Inference Optimization Researcher)

**Scores:** Novelty: 7/10 | Feasibility: 8/10 | Technical Soundness: 8/10 | Completeness: 7/10

**Strengths:**
- Principled feature selection: ARD provides automatic, data-driven pruning.
- Actionable uncertainty: Enables automatic detection of out-of-distribution scenarios.
- Clean transfer protocol: 10% outlier threshold provides quantitative decision boundary.

**Weaknesses:**
- Gaussian assumption fragility: Multi-modal or preemption-induced outliers not well-handled.
- Missing communication/memory features: No explicit features for TP communication overhead or PCIe transfer latency.
- Uncertainty approximation in inference: Production Go code uses simplified variance computation.

**Suggestions:**
1. Add robust regression variant with Student-t likelihood (df=4-5).
2. Expand hardware interaction features: `tp_degree * batch_tokens / interconnect_bandwidth` and CPU offload features.
