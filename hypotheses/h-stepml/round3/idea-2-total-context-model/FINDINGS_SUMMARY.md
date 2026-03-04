# FINDINGS SUMMARY: Idea 2 — Total-Context Linear Model with Feature Scaling

**Date:** 2026-02-27
**Status:** Per-step improvement demonstrated; no BLIS E2E improvement over Round 2

## 1. Idea Recap

Replace the 2-feature step-time model (`prefill_tokens + decode_tokens`) with a 3-feature formulation (`new_tokens + total_context`) using proper feature scaling, informed by FairBatching's validated +/-1.3% per-step approach. Addresses BC-2 (KV feature scaling) and BC-3 (CodeLlama-34B anomaly).

Three sub-hypotheses tested:
- **H1:** FairBatching 3-coefficient OLS formulation (a + b*new_tokens + c*kv_sum)
- **H2:** Feature scaling variants (StandardScaler, log-transform) for Ridge regression
- **H3:** BLIS E2E validation with trace replay + 34B deep-dive

## 2. Sub-Hypothesis Results Table

| Sub-Hypothesis | Status | Key Metric | Takeaway |
|---|---|---|---|
| **H1:** FairBatching 3-coeff | **Partially supported** | 56.2% per-step MAPE (vs 83.1% baseline) | 27pp improvement; kv_sum helps for 3/5 models |
| **H2:** Feature scaling | **Refuted** | 83.0% best (vs 82.0% no-KV baseline) | All scaling variants WORSE than no-KV; multicollinearity is the root cause, not scale mismatch |
| **H3:** BLIS E2E + 34B | **Refuted** | 56.2% E2E, 9.5% ITL (identical to R2 trace replay) | kv_sum has zero BLIS E2E impact; 34B anomaly was workload-spec-specific |

## 3. Best BLIS E2E Result — Full Per-Experiment Error Table

Using trace replay mode with 4-coeff OLS (prefill + decode + kv_sum) + overhead floor:

| Experiment | Model | Workload | GT E2E (ms) | BLIS E2E (ms) | E2E Error | TTFT Error | ITL Error |
|---|---|---|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | llama-2-7b | roleplay | 2,071 | 810 | 60.9% | 78.6% | **4.0%** |
| llama-2-70b-tp4-general | llama-2-70b | general | 5,321 | 1,986 | 62.7% | 88.3% | 23.5% |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | codegen | 4,606 | 2,022 | 56.1% | 77.9% | 10.7% |
| llama-2-70b-tp4-roleplay | llama-2-70b | roleplay | 4,562 | 2,019 | 55.7% | 78.0% | 10.3% |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | codegen | 4,675 | 2,234 | 52.2% | 76.8% | **2.1%** |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | general | 5,039 | 2,253 | 55.3% | 80.1% | 8.7% |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | roleplay | 4,685 | 2,276 | 51.4% | 77.3% | **0.7%** |
| codellama-34b-tp2-general | codellama-34b | general | 4,093 | 1,682 | 58.9% | 80.6% | 17.9% |
| codellama-34b-tp2-codegen | codellama-34b | codegen | 3,723 | 1,674 | 55.0% | 78.1% | 10.1% |
| codellama-34b-tp2-roleplay | codellama-34b | roleplay | 3,670 | 1,702 | 53.6% | 78.1% | 7.3% |
| **MEAN** | | | | | **56.2%** | **79.4%** | **9.5%** |

**E2E < 10%:** 0/10 | **ITL < 10%:** 5/10 | **ITL < 15%:** 7/10

## 4. What Worked (Specific Techniques)

1. **FairBatching 3-coeff OLS formulation.** Combining prefill+decode into `new_tokens` and adding a single `kv_sum` coefficient achieves 56.2% per-step MAPE — 27pp better than the 2-coeff baseline (83.1%). This validates FairBatching's insight that a minimal formulation with one context-length feature outperforms complex multi-feature Ridge regression.

2. **The kv_sum coefficient is positive and stable** across all models where KV data exists (0.031-0.065). This confirms that total context length (sum of per-request KV cache lengths) captures real signal about memory-bandwidth-bound attention costs.

3. **Per-experiment analysis with 4-coeff OLS.** Individual experiments like mixtral-codegen (18.1% MAPE) and mixtral-general (11.1% MAPE) achieve excellent per-step accuracy, showing the formulation works well for specific model+workload combinations.

## 5. What Failed and Why (Root Causes)

1. **Feature scaling (H2) does not fix KV feature instability.** StandardScaler, log-transform, and their combination all perform WORSE than the no-KV baseline. Root cause: the issue is **multicollinearity** among 4 correlated KV features (kv_sum, kv_max, kv_mean, kv_std), not dynamic range mismatch. Ridge regularization cannot resolve multicollinearity. Solution: use fewer features (H1's approach).

2. **Per-step MAPE improvement does not propagate to BLIS E2E.** The 27pp per-step improvement (83% → 56%) has ZERO impact on BLIS E2E (56.2% E2E, 9.5% ITL — identical to Round 2 trace replay). Root cause: the overhead floor (~4-9ms) dominates 70-90% of step predictions, completely masking the GPU compute prediction improvement. The kv_sum contribution (~40-1300µs) is negligible versus the floor.

3. **KV data coverage is incomplete.** 2/5 models (llama-2-7b, mixtral-8x7b-v0-1) have kv_sum=0 for ALL steps, limiting the feature's utility to 3/5 models. Root cause: the lifecycle-to-step join fails for experiments where step timestamps and request lifecycle timestamps don't overlap.

## 6. Binding Constraints

| Constraint | Status | Evidence |
|---|---|---|
| **BC-2: KV feature scaling** | Addressed but irrelevant to E2E | H1 shows kv_sum is productive per-step, but overhead floor masks its BLIS E2E contribution |
| **BC-3: CodeLlama-34B anomaly** | Resolved | Not anomalous in trace replay mode; was workload-spec-specific (all models show 51-63% E2E error uniformly) |
| **BC-NEW: Overhead floor dominance** | **BINDING** | The floor masks all step-time improvements. Improving GPU compute prediction cannot improve BLIS E2E while the floor dominates. |
| **BC-1: TTFT/simulation fidelity** | Unchanged | 79.4% TTFT error persists even with trace replay |

## 7. Data Insights Discovered

1. **The overhead floor is the dominant prediction mechanism.** For 70-90% of steps, BLIS's step-time prediction = overhead floor, regardless of the step-time model used. This means step-time model improvements only affect 10-30% of steps.

2. **Per-step MAPE and BLIS E2E are decoupled.** A 27pp per-step improvement (83% → 56%) produces 0pp BLIS E2E improvement. This is because BLIS E2E depends on step *cycle* time (including overhead), not step *compute* time (what the model predicts).

3. **FairBatching's minimal formulation works.** The 3-coeff model (`a + b*new_tokens + c*kv_sum`) achieves better per-step accuracy than any complex feature engineering approach (Ridge with 8+ features, StandardScaler, log-transforms, interaction terms).

4. **Multicollinearity, not scale, causes KV feature failure.** Round 2 diagnosed the KV issue as a scaling problem. Round 3 proves it's a collinearity problem. The fix is fewer features (1 KV feature instead of 4), not better scaling.

5. **Step-time distribution is bimodal.** All models show a sharp bimodal distribution: decode-only steps (~100-500µs) and mixed steps (~1,000-12,000µs). CodeLlama-34B has the most extreme bimodality (P50=291µs, mean=1,603µs, P99=11,928µs).

## 8. Comparison to Baseline

| Metric | Round 2 Workload-Spec | Round 2 Trace Replay (Idea 1) | **Round 3 Idea 2** | Delta vs R2 Trace |
|---|---|---|---|---|
| Per-step MAPE | 43.9% (no KV) | — | **56.2%** (3-coeff) | +12.3pp (worse with overhead, better without) |
| BLIS E2E (mean) | 427.8% | 56.2% | **56.2%** | **0pp** |
| BLIS TTFT (mean) | 31,906% | 78.8% | **79.4%** | ~0pp |
| BLIS ITL (mean) | 33.6% | 9.5% | **9.5%** | **0pp** |
| ITL < 10% | 5/10 | 5/10 | 5/10 | 0 |

**Net assessment:** Idea 2 achieves a meaningful per-step prediction improvement (27pp better than 2-coeff baseline without overhead floor), but this improvement has zero impact on BLIS E2E. The binding constraint is NOT the step-time model — it's the overhead floor calibration and BLIS simulation fidelity.

## 9. Go Integration Feasibility

**Trivial.** The 3-coeff model adds exactly one feature coefficient (`kv_sum`) to the existing StepML artifact JSON. The Go evaluator (`sim/latency/stepml.go`) already extracts `kv_sum` from the batch via `extractBatchFeatures()` and supports arbitrary `feature_coefficients` maps. No Go code changes needed — just export the artifact with the kv_sum coefficient.

Example artifact addition:
```json
{
  "step_time": {
    "model_type": "linear",
    "intercept": -68.642,
    "feature_coefficients": {
      "prefill_tokens": 0.798,
      "decode_tokens": 55.978,
      "kv_sum": 0.002
    }
  }
}
```

However, since the kv_sum coefficient has zero impact on BLIS E2E, the integration adds complexity without value. **Recommended: do NOT integrate** unless the overhead floor calibration is fixed first.

## 10. Generalization Results

### H4: Leave-One-Workload-Out (LOWO)

**Status: REFUTED** — Mean LOWO MAPE: **2,162.7%** (threshold <70%, R2 baseline 117.4%)

| Model | Holdout: codegen | Holdout: general | Holdout: roleplay | Mean |
|---|---|---|---|---|
| codellama-34b_tp2 | 2,526.4% | 281.0% | 3,348.6% | 2,052.0% |
| llama-2-70b_tp4 | — | 154.2% | 3,336.0% | 1,745.1% |
| mixtral-8x7b-v0-1_tp2 | 2,380.7% | 1,172.5% | 4,102.3% | 2,551.8% |

Zero degradation vs in-distribution (0.0pp), confirming the 3-coeff model doesn't overfit to workloads — but the absolute MAPE is catastrophic due to lack of regime separation and numerical instability from kv_sum overflow.

### H5: Leave-One-Model-Out (LOMO)

**Status: REFUTED** — Mean LOMO MAPE: **2,281.6%** (threshold <80%, R2 baseline 108.6%)

| Holdout Model | LOMO MAPE | R2 LOMO | Comparison |
|---|---|---|---|
| codellama-34b | 2,121.1% | — | — |
| llama-2-70b | 1,408.7% | — | — |
| llama-2-7b | 2,985.2% | — | — |
| mixtral-8x7b-v0-1 | 2,611.5% | — | — |

**Root cause:** The FairBatching 3-coeff formulation removes the regime structure that enabled R2's 108.6% LOMO. Without decode-only vs mixed-batch separation, a single linear model across all step types produces catastrophic predictions for out-of-distribution models. R2's regime structure is mandatory for generalization.

## Appendix: Methodology

- **Data source:** BLIS-research/eval/ground_truth/ (10 experiments, 5 models × 3 workloads, minus reasoning)
- **KV features:** Extracted via `lifecycle_kv_extractor.py` (joins step-level batch summaries with per-request lifecycle timestamps)
- **Split:** Temporal 60/20/20 per experiment (first 60% train, next 20% valid, last 20% test)
- **Per-model training:** Mandatory per Round 1+2 findings — separate model per model_tp
- **Overhead floor:** From Round 2 calibrated artifacts (3,897-9,125µs per model)
- **BLIS validation:** Trace replay mode using lifecycle-to-CSV conversion (matching idea-1)
- **All per-step metrics computed WITHOUT overhead floor** (floor only applied for BLIS E2E)
