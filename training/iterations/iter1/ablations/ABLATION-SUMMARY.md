# Ablation Experiments Summary - Iteration 1

**Baseline Model Performance:**
- Overall Loss: 134.54%
- TTFT RMSE: 69.29%
- E2E RMSE: 65.24%

**Experiment Date:** 2026-03-28
**Method:** Bayesian optimization with 50 trials per ablation

---

## 1. Chunking Overhead (β₅) - REDUNDANT ⚪

**Hypothesis:** H-ablation-chunking
"Removing chunking overhead (β₅=0) will increase loss by >5%, confirming it captures real prefill chunking cost for sequences >2048 tokens"

**Results:**
- Overall Loss: 135.967% (Δ **+1.06%**)
- TTFT RMSE: 69.340% (Δ **+0.07%**)
- E2E RMSE: 66.628% (Δ **+2.13%**)

**Verdict:** ⚪ **REDUNDANT**
Removing β₅ causes minimal performance degradation (<5% overall, <5% TTFT, <10% E2E). The chunking overhead term provides negligible benefit and can be safely removed in iter2.

**Interpretation:**
The near-zero degradation suggests that either:
1. Chunking overhead is not a significant factor in the training data
2. The β₅ term is redundant with other basis functions (e.g., β₀ prefill already captures chunking)
3. The optimizer assigned β₅ a near-zero coefficient anyway

**Recommendation:** Remove β₅ in iter2 model evolution.

---

## 2. TP Communication Overhead (β₃) - MODERATE 🟡

**Hypothesis:** H-ablation-tp-comm
"Removing TP communication overhead (β₃=0) will increase loss by >5%, confirming it captures real all-reduce costs"

**Results:**
- Overall Loss: 138.420% (Δ **+2.88%**)
- TTFT RMSE: 68.764% (Δ **-0.76%**)
- E2E RMSE: 69.656% (Δ **+6.77%**)

**Verdict:** 🟡 **MODERATE**
Removing β₃ causes moderate E2E degradation (6.77%) but minimal overall loss increase (2.88%). The term provides measurable benefit but is not critical.

**Interpretation:**
The +6.77% E2E degradation confirms β₃ captures real TP communication overhead for distributed models (TP>1). However, the small overall loss increase suggests this overhead is only significant for a subset of experiments (likely TP2/TP4 configurations).

**Recommendation:** Keep β₃ in iter2. It captures a real phenomenon with moderate impact.

---

## 3. KV Cache Management Overhead (β₄) - CRITICAL 🔴

**Hypothesis:** H-ablation-kv-mgmt
"Removing KV cache management overhead (β₄=0) will increase loss by >5%, confirming it captures real block allocation/deallocation costs"

**Results:**
- Overall Loss: 161.733% (Δ **+20.21%**)
- TTFT RMSE: 76.739% (Δ **+10.75%**)
- E2E RMSE: 84.994% (Δ **+30.28%**)

**Verdict:** 🔴 **CRITICAL**
Removing β₄ causes severe degradation across all metrics (>10% overall, >10% TTFT, >30% E2E). This term is **essential** and captures a fundamental cost component.

**Interpretation:**
The massive degradation (30.28% E2E increase) confirms β₄ captures critical per-request KV cache management overhead. This is the single most important additive overhead term in the model. Without it, the model cannot accurately predict request-level latency variance.

**Recommendation:** **Definitely keep** β₄ in iter2. Highest-priority term.

---

## Summary Table

| Term | Coefficient | Δ Overall Loss | Δ TTFT | Δ E2E | Verdict |
|------|-------------|----------------|--------|-------|---------|
| β₅ (chunking) | 0.0 | +1.06% | +0.07% | +2.13% | ⚪ REDUNDANT |
| β₃ (TP comm) | 1.940 | +2.88% | -0.76% | +6.77% | 🟡 MODERATE |
| β₄ (KV mgmt) | 0.00035 | +20.21% | +10.75% | +30.28% | 🔴 CRITICAL |

---

## Recommendations for Iteration 2

1. **Remove β₅ (chunking)**: Minimal impact, reduces model complexity
2. **Keep β₃ (TP comm)**: Moderate benefit for distributed models (TP>1)
3. **Keep β₄ (KV mgmt)**: Essential term with highest impact
4. **Investigate β₄ scaling**: Small coefficient (0.00035) but massive ablation impact suggests this term may need better normalization or a different functional form

---

**Methodology Note:**
All ablations forced the target coefficient to [0.0, 0.0] bounds while allowing other coefficients to re-optimize. This tests whether the optimizer can compensate for the missing term by adjusting other coefficients. The inability to compensate (high degradation) confirms the term captures unique variance not covered by other basis functions.
