# Executive Summary: Third Approach for BLIS Latency Modeling

## Problem Statement

BLIS currently uses two approaches for LLM inference performance modeling:
1. **Blackbox Optimization**: Accurate but requires hours of profiling per configuration
2. **Roofline Model**: Analytical but requires manual tuning of efficiency factors

The goal is to develop a **third approach** that:
- Handles diverse settings (dense/MoE models, A100/H100 hardware, various vLLM knobs)
- Maintains the linear coefficient structure: `alpha · features` and `beta · features`
- Is learnable from vLLM tracing data
- Is robust and does not overfit

---

## Ideas Overview

### Idea 1: Physics-Informed Compositional Features (PICF)
Uses analytical physics-based feature engineering (FLOPs/TFLOPs, bytes/bandwidth) with learned coefficients via Ridge regression. Features are derived from roofline principles, enabling hardware-agnostic normalization. Simple to implement with standard ML tools.

### Idea 2: Mixture of Linear Experts with Regime Detection (MoLE-RD)
Identifies distinct operational regimes (low-load, normal, high-pressure, saturation) and trains separate linear models for each. A rule-based regime detector routes observations to the appropriate specialist. Captures non-linearities through piecewise-linear approximation.

### Idea 3: Hierarchical Feature Decomposition with Transfer Learning (HFD-TL)
Decomposes latency into base features (universal physics), config features, and residual features. Trains a base model on "donor" configurations and calibrates residuals on new "target" configurations with minimal data (~500-1000 observations).

### Idea 4: Bayesian Sparse Regression with Automatic Relevance Determination (BSR-ARD)
Uses Bayesian linear regression with ARD priors to automatically prune irrelevant features from a large candidate set (18+ features). Provides uncertainty quantification and z-score-based transfer validation protocol.

### Idea 5: Robust Ensemble with Configuration Embeddings (RECE)
Combines quantile regression (for heavy-tailed distributions), configuration embeddings (for automatic similarity), and specialist ensembles (for configuration-aware adaptation). Includes comprehensive feature set with communication and CPU offload overhead.

---

## Comparison Table

| Aspect | PICF | MoLE-RD | HFD-TL | BSR-ARD | RECE |
|--------|------|---------|--------|---------|------|
| **Average Novelty** | 6.3 | 6.3 | 6.7 | 6.3 | 6.3 |
| **Average Feasibility** | 8.0 | 7.3 | 8.0 | 8.0 | 7.0 |
| **Average Soundness** | 7.0 | 7.0 | 7.0 | 7.3 | 6.7 |
| **Average Completeness** | 7.3 | 7.0 | 7.0 | 7.3 | 7.7 |
| **Calibration Data** | ~10,000 | ~2,000/regime | ~500-1,000 | ~500-1,000 | ~10,000/config |
| **Non-linear Effects** | Limited | Regime-based | Precomputed features | Precomputed features | Quantile + ensemble |
| **Distribution Handling** | Gaussian | Per-regime | Gaussian | Gaussian (can log) | Quantile (robust) |
| **Feature Selection** | Manual | Manual per-regime | Manual hierarchical | Automatic (ARD) | Manual comprehensive |
| **Transfer Validation** | None | None | None | Z-score based | Embedding-based |
| **Implementation Complexity** | Low | Medium | Medium | Medium | High |

---

## Reviewer Consensus

**Consistent Strengths Across Ideas:**
1. **Interpretability**: All ideas maintain the linear coefficient structure, enabling debugging and validation
2. **Physics-grounded features**: Hardware normalization (FLOPs/TFLOPs, bytes/bandwidth) enables cross-hardware generalization
3. **Practical training pipelines**: All use standard ML tools (sklearn, scipy) with clear implementation paths

**Consistent Weaknesses:**
1. **Linear model limitations**: Reviewers consistently noted that linear models may miss complex non-linear interactions
2. **Distribution assumptions**: Gaussian assumptions are problematic for heavy-tailed latency distributions
3. **Data requirements**: All approaches require substantial training data (10K-50K+ observations)

**Key Debates:**
- **Feature selection**: Manual (Ideas 1-3) vs. automatic (Idea 4) - ARD received praise for reducing practitioner burden
- **Regime handling**: Discrete regimes (Idea 2) vs. continuous models - discrete regimes have discontinuity issues at boundaries
- **Robustness**: Quantile regression (Idea 5) vs. Gaussian assumptions - quantile approach is more principled for latency data

---

## Recommendation

### Primary Recommendation: **Hybrid of Ideas 3 + 4**

Combine the **transfer learning framework** from Idea 3 (HFD-TL) with the **Bayesian ARD feature selection** from Idea 4 (BSR-ARD):

1. **Base Model with ARD**: Train a Bayesian ARD model on pooled donor configurations to automatically select relevant features from the comprehensive candidate set
2. **Calibration via Residuals**: For new configurations, fit only residual coefficients with minimal data
3. **Uncertainty-Based Validation**: Use ARD's predictive uncertainty to detect transfer failures
4. **Optional Quantile Enhancement**: If latency distributions are heavy-tailed, use quantile regression instead of mean regression (from Idea 5)

**Rationale:**
- HFD-TL provides the lowest calibration data requirement (~500-1000 observations)
- BSR-ARD provides automatic feature selection, reducing manual feature engineering burden
- Combining them addresses the "limited residual features" weakness of Idea 3
- This hybrid scored highest on the combination of feasibility (8.0) + soundness (7.0-7.3)

### Secondary Recommendation: **Start with Idea 1 (PICF)**

If implementation speed is critical, start with PICF as a baseline:
- Simplest to implement (single Ridge regression)
- Provides immediate, interpretable results
- Can be incrementally enhanced with ARD (Idea 4) or transfer learning (Idea 3)

---

## Next Steps

1. **Data Collection**: Identify 3-5 diverse donor configurations (dense/MoE on A100/H100)
   - Collect ~10,000 step traces per configuration
   - Ensure coverage of edge cases (high KV pressure, preemption)

2. **Feature Engineering**: Implement the comprehensive feature set from Idea 5
   - Include communication overhead features: `(tp_degree - 1) * tokens / bandwidth`
   - Include CPU offload features: `kv_blocks_to_offload / pcie_bandwidth`

3. **Baseline Implementation**: Start with PICF (Idea 1) as baseline
   - Validate on held-out temporal test set
   - Measure MAPE on TTFT, ITL, E2E metrics

4. **Enhancement**: Add ARD feature selection (Idea 4)
   - Prune irrelevant features automatically
   - Implement z-score transfer validation

5. **Production Deployment**: Integrate with BLIS
   - Store coefficients in `defaults.yaml`
   - Implement Go inference code
   - Validate against golden dataset

6. **Iteration**: Based on production feedback
   - Consider quantile regression if outliers are problematic
   - Consider ensemble if single model is insufficient
