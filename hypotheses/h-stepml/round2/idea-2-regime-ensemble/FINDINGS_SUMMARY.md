# Idea 2: Regime-Switching Ensemble — Findings Summary

**Date:** 2026-02-27
**Data:** 77,816 steps from 10 experiments (4 models x 3 workloads; no reasoning workloads in this dataset)
**Models:** Llama-2-7B (TP1), Llama-2-70B (TP4), CodeLlama-34B (TP2), Mixtral-8x7B (TP2)
**Workloads:** general, codegen, roleplay

---

## Executive Summary

Idea 2's regime-switching Ridge ensemble was designed to exploit the insight that batch computation falls into distinct regimes (decode-only, mixed-light, mixed-heavy) with different feature-performance relationships. The approach uses ProgressIndex-derived KV features to capture per-request memory footprint — a key missing signal identified in Round 1.

**Bottom line:** On the current ground-truth data (10 experiments, 77K steps), Idea 2 fails to meet its primary targets. Per-step MAPE is 64.4% (target was <15%), BLIS E2E error is 427.8% (target was <10%), and generalization tests both fail. However, ITL prediction remains reasonable at 33.6%, and LOMO shows a 23.6x improvement over Round 1's catastrophic baseline. The results point to specific fixable problems (KV feature instability, workload spec generation for TTFT) rather than a fundamental flaw in the approach.

---

## Hypothesis Results

| # | Hypothesis | Status | Key Metric | Target | Actual |
|---|-----------|--------|-----------|--------|--------|
| H1 | Regime Ridge per-step MAPE | **Not met** | Weighted MAPE | <15% | 64.4% |
| H2 | BLIS E2E validation | **Not met** | Mean E2E error | <10% | 427.8% |
| H3 | Secondary method calibration | **Refuted** | E2E improvement | >=5pp | 0.0pp |
| H4 | LOMO (cross-model) | **Refuted** | Avg MAPE | <80% | 108.6% |
| H5 | LOWO (cross-workload) | **Refuted** | Avg MAPE | <25% | 117.4% |

---

## H1: Regime-Specific Ridge Training

### Per-Model Step Overhead

Step duration in training data captures GPU forward pass only. The real step cycle time includes CPU overhead (scheduling, sync, memory management). Overhead is derived from ground-truth ITL:

| Model | Overhead (us) |
|-------|--------------|
| Llama-2-7B (TP1) | 3,897 |
| CodeLlama-34B (TP2) | 6,673 |
| Llama-2-70B (TP4) | 8,029-8,203 |
| Mixtral-8x7B (TP2) | 9,125 |

### Regime Distribution

| Regime | Steps | Share |
|--------|-------|-------|
| decode_only (prefill == 0) | 60,613 | 77.9% |
| mixed_light (0 < prefill < 256) | 17,129 | 22.0% |
| mixed_heavy (prefill >= 256) | 74 | 0.1% |

Mixed-heavy has only 74 steps — effectively untestable. This is a key difference from the HYPOTHESIS.md expectation of ~4.4% (~5,400 steps), indicating the new ground-truth data has fundamentally different prefill characteristics.

### Per-Model Regime MAPE (Test Set)

| Model | Weighted MAPE | decode_only | mixed_light | Best alpha |
|-------|--------------|-------------|-------------|------------|
| Llama-2-7B (TP1) | **36.2%** | 36.6% (r=0.530) | 32.4% (r=0.590) | 1000/0.01 |
| Mixtral-8x7B (TP2) | **16.6%** | 18.3% (r=0.610) | 13.4% (r=0.872) | 0.01/0.01 |
| CodeLlama-34B (TP2) | **99.2%** | 65.8% (r=0.785) | 179.9% (r=0.714) | 1000/0.01 |
| Llama-2-70B-HF (TP4) | **84.4%** | 91.7% (r=0.702) | 67.9% (r=0.697) | 0.01/1000 |
| Llama-2-70B (TP4) | **93.3%** | 112.2% (r=0.854) | 59.3% (r=0.758) | 1000/1000 |

### Model Comparison

| Model Type | Weighted MAPE |
|-----------|--------------|
| Regime-specific Ridge (with KV) | **64.4%** |
| Single global Ridge (all features) | 93.5% |
| Blackbox baseline (2 features) | **46.9%** |
| Regime Ridge (no KV features) | **43.9%** |

### KV Feature Ablation: Negative Contribution

KV features **hurt** per-step MAPE by 20.5 percentage points (64.4% with KV vs. 43.9% without). Per-model breakdown:

| Model | With KV | Without KV | Delta |
|-------|---------|-----------|-------|
| Llama-2-7B | 36.2% | 30.1% | +6.1pp worse |
| Mixtral-8x7B | 16.6% | 22.0% | -5.4pp better |
| CodeLlama-34B | 99.2% | 55.5% | +43.7pp worse |
| Llama-2-70B-HF | 84.4% | 48.2% | +36.2pp worse |
| Llama-2-70B | 93.3% | 68.3% | +25.0pp worse |

KV features only help Mixtral. For all dense models, KV features degrade performance. The numerical overflow warnings during Ridge training (divide-by-zero, overflow in matmul) confirm that KV feature magnitudes (kv_sum up to 64,000+) cause coefficient instability in raw linear space.

### Diagnosis

1. **Mixed-heavy regime is empty** (74 steps vs. expected ~5,400). The 256-token prefill threshold sees almost no steps. This means the 3-regime split collapses to 2 effective regimes.
2. **KV features cause numerical instability.** Large kv_sum values produce overflow during Ridge CV grid search. Alpha=1000 often wins for large models, indicating severe regularization is needed to tame unstable coefficients.
3. **The blackbox 2-feature baseline outperforms all KV-augmented models.** This suggests the KV features as implemented add noise rather than signal — possibly because ProgressIndex KV proxies are poorly correlated with actual batch-level compute cost in this dataset.

---

## H2: BLIS E2E Validation

### Per-Experiment Results

| Experiment | E2E Error | TTFT Error | ITL Error |
|-----------|-----------|------------|-----------|
| 7b-roleplay | 52.5% | 78.4% | **4.0%** |
| 70b-general | 182.7% | 12,567% | 23.5% |
| 70b-hf-codegen | 55.7% | 77.7% | **10.7%** |
| 70b-roleplay | 55.6% | 77.9% | **10.3%** |
| mixtral-codegen | 51.5% | 76.5% | **2.1%** |
| mixtral-general | 554.6% | 44,464% | **8.7%** |
| mixtral-roleplay | 50.8% | 77.2% | **0.7%** |
| 34b-general | 2,901.1% | 230,931% | 80.3% |
| 34b-codegen | 370.2% | 30,425% | 95.9% |
| 34b-roleplay | **3.3%** | 282.7% | 100.1% |

### Aggregate Metrics

| Metric | Mean | Experiments <10% |
|--------|------|-----------------|
| E2E error | 427.8% | 1/10 (34b-roleplay) |
| TTFT error | 31,906% | 0/10 |
| ITL error | 33.6% | 5/10 |

### Pattern Analysis

- **ITL is the best metric** (33.6% mean, 5/10 under 10%). ITL directly reflects step-time prediction quality + overhead floor, confirming the core modeling approach works for decode-dominant workloads.
- **TTFT errors are catastrophic** (thousands to hundreds-of-thousands percent). This indicates the BLIS simulation's prefill + queueing timing is fundamentally mismatched with the ground truth. The workload spec generation or request injection timing may be wrong.
- **"general" workloads consistently worst** (70b-general 182.7%, mixtral-general 554.6%, 34b-general 2901.1%). The "general" workload likely has longer prefill sequences and more variable batch dynamics.
- **CodeLlama-34B is the worst model** across all metrics. Per-step MAPE for 34b was already 99.2% (worst in H1).

---

## H3: Secondary Method Calibration

### Extracted Constants

| Model | Queue-to-Schedule (us) | Schedule-to-First-Token (us) |
|-------|----------------------|---------------------------|
| Llama-2-7B (TP1) | 202 (p25=187, p75=219) | 15,884 |
| Llama-2-70B (TP4) | 415 (p25=265, p75=550) | 40,869 |
| Llama-2-70B-HF (TP4) | 281 (p25=220, p75=316) | 37,372 |
| CodeLlama-34B (TP2) | 291 (p25=246, p75=510) | 30,100 |
| Mixtral-8x7B (TP2) | 353 (p25=275, p75=577) | 37,952 |

### Ablation Result

| Config | Mean E2E | Mean TTFT | Mean ITL |
|--------|----------|-----------|----------|
| A: StepTime-only | 427.8% | 31,905.8% | 33.6% |
| B: + Secondary methods | 427.8% | 31,905.7% | 33.6% |
| **Delta** | **+0.0pp** | **+0.0pp** | **+0.0pp** |

**Verdict: REFUTED.** The secondary method constants (200-400 us) are negligible compared to E2E errors of hundreds to thousands of percent. When the dominant error source (StepTime or workload spec) is off by orders of magnitude, microsecond-level corrections cannot matter.

---

## H4: LOMO (Leave-One-Model-Out)

### Per-Fold Results

| Fold | Held-Out Model | Per-Step MAPE | BLIS E2E Mean |
|------|---------------|---------------|---------------|
| 1 | CodeLlama-34B | 63.4% | 657.0% |
| 2 | Llama-2-70B | 197.6% | 797.6% |
| 3 | Llama-2-7B | 41.2% | BLIS failed |
| 4 | Mixtral-8x7B | 132.3% | 253.7% |

**Average LOMO MAPE: 108.6%** (target: <80%) — **REFUTED**

### Per-Regime MAPE (Across Folds)

| Regime | Average MAPE | Range |
|--------|-------------|-------|
| decode_only | 88.9% | 27.8-200.4% |
| mixed_light | 185.7% | 55.0-320.6% |
| mixed_heavy | 243.1% | 67.5-473.9% |

### Comparison to Round 1

- Round 1 LOMO: 2,559.7% average MAPE
- Round 2 LOMO: 108.6% average MAPE
- **Improvement: 23.6x** (significant, but still above 80% threshold)

### Diagnosis

The 70B fold is the worst (197.6%) because 70B is the largest model with highest absolute step times. Cross-model transfer fails because step-time scale is model-dependent (determined by parameter count and TP degree). The Mixtral fold (132.3%) confirms that dense-to-MoE transfer remains challenging due to fundamentally different compute scaling.

---

## H5: LOWO (Leave-One-Workload-Out)

### Per-Model LOWO MAPE

| Model | Avg MAPE | Worst Fold |
|-------|----------|-----------|
| 7b | 25.6% | roleplay (31.1%) |
| 34b | 131.0% | roleplay (183.5%) |
| 70b | 293.8% | roleplay (420.0%) |
| Mixtral | 19.1% | roleplay (23.9%) |

**Grand average LOWO MAPE: 117.4%** (target: <25%) — **REFUTED**

### Per-Workload MAPE (Across Models)

| Held-Out Workload | Avg MAPE | Range |
|-------------------|----------|-------|
| codegen | 123.1% | 18.7-265.1% |
| general | 67.4% | 14.8-145.4% |
| roleplay | 161.7% | 23.9-420.0% |

### BLIS E2E on Held-Out Workloads

| Held-Out Workload | Mean E2E Error | Mean ITL Error |
|-------------------|---------------|----------------|
| codegen | 163.5% | 36.8% |
| general | 273.0% | 16.4% |
| roleplay | 36.4% | 36.7% |

### Comparison to Round 1

- Round 1 LOWO: 109.7% average MAPE
- Round 2 LOWO: 117.4% average MAPE
- **Improvement: 0.9x** (no improvement — slightly worse)

### Diagnosis

Workload generalization is comparable to Round 1. The regime structure was expected to provide workload-agnostic computation modeling, but LOWO MAPE actually regressed slightly. The 70B model dominates the error average (293.8%), suggesting that for large models with complex batch dynamics, workload distribution shifts have outsized impact on predictions.

---

## Cross-Cutting Findings

### 1. KV Features Are Counter-Productive in Raw Linear Space

The central thesis of Idea 2 was that ProgressIndex-derived KV features would improve per-step prediction. On this ground-truth data, the opposite is true: KV features degrade per-step MAPE from 43.9% to 64.4%. The root cause is numerical instability — kv_sum values reaching 64,000+ cause Ridge coefficient overflow when alpha is small. Only Mixtral benefits from KV features (-5.4pp), possibly because its MoE architecture has a different KV-to-compute relationship.

**Implication:** KV features need either (a) normalization/scaling before regression, (b) log-transform of features (not target), or (c) a non-linear model that handles large feature ranges gracefully.

### 2. Mixed-Heavy Regime Is Empty

Only 74 of 77,816 steps (0.1%) fall in the mixed-heavy regime (prefill >= 256). The original hypothesis expected ~4.4% based on 16 experiments with 4 workloads. The current 10-experiment dataset (3 workloads, no reasoning) has very few long-prefill batches. The 3-regime design is effectively a 2-regime design on this data.

### 3. TTFT Errors Indicate Systematic Workload Spec or Simulation Mismatch

TTFT errors of 31,906% cannot be explained by per-step model inaccuracy. These errors point to a mismatch in how the BLIS simulation generates/schedules requests vs. how the real system operates. Likely causes include:
- Workload spec generation (`build_workload_spec`) producing request arrival patterns that don't match the ground truth
- BLIS request queueing behavior differing from real vLLM queueing
- Prefill scheduling in BLIS (chunked prefill, request admission) not matching the real system's behavior

### 4. ITL Remains the Most Accurate Metric

Despite the per-step model limitations, ITL error is 33.6% mean with 5/10 experiments under 10%. This confirms that the overhead floor mechanism (`max(overhead, compute)`) effectively handles the decode-dominated continuous batching regime. The step-time model's accuracy matters less when the overhead floor dominates — which it does for small-to-medium batch sizes.

### 5. CodeLlama-34B Is Consistently the Worst Model

34B shows the highest per-step MAPE (99.2%), worst BLIS E2E (2,901% for general), and high generalization errors. This model may have batch dynamics that differ significantly from the Llama-2 family. Investigation into 34B-specific step-time distributions and batch size patterns is warranted.

### 6. "General" Workloads Are Consistently Hard

Across H2, the "general" workload produces the worst E2E errors for every model (182.7%, 554.6%, 2,901.1%). The "general" workload likely has more diverse batch compositions and longer prefill sequences, pushing the model into poorly-covered regions of the feature space.

---

## Recommendations for Next Steps

1. **Fix KV feature scaling.** Normalize KV features (e.g., kv_sum / max_kv_blocks, kv_mean / max_seq_len) before regression, or use StandardScaler within the Ridge pipeline.

2. **Investigate TTFT errors.** The 31,906% mean TTFT error is the dominant contributor to E2E error. Diagnose whether the workload spec, BLIS request injection, or prefill scheduling is the root cause.

3. **Lower the mixed-heavy threshold or drop it.** With only 74 steps in mixed-heavy, consider a 2-regime model (decode-only vs. any-mixed) or lower the threshold to 64 or 128 tokens.

4. **Investigate 34B-specific dynamics.** CodeLlama-34B accounts for disproportionate error. Check if its batch size distribution, step-time distribution, or KV utilization patterns differ qualitatively from other models.

5. **Consider feature scaling for large models.** The 70B and 34B models show much higher MAPE than 7B and Mixtral. This may be because absolute feature values (kv_sum, decode_tokens) scale with model capacity, and raw linear regression cannot handle the scale differences without normalization.

---

## Output Artifacts

| Hypothesis | Output Directory | Key Files |
|-----------|-----------------|-----------|
| H1 | `h1-kv-regime-models/output/` | `regime_ridge_results.csv`, `artifacts/*.json` |
| H2 | `h2-blis-e2e-validation/output/` | `e2e_validation_regime.csv`, `e2e_summary_regime.json` |
| H3 | `h3-secondary-method-calibration/output/` | `ablation_comparison.csv`, `ablation_summary.json`, `secondary_constants.json` |
| H4 | `h4-model-generalization/output/` | `lomo_per_step_results.csv`, `lomo_blis_e2e_results.csv`, `lomo_summary.json` |
| H5 | `h5-workload-generalization/output/` | `lowo_per_step_results.csv`, `lowo_blis_e2e_results.csv`, `lowo_summary.json` |
