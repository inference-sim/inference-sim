# Idea 2: Total-Context Linear Model with Feature Scaling

## Overview

Replace the 2-feature step-time model (`prefill_tokens + decode_tokens`) with a 3-feature formulation (`new_tokens + total_context`) using proper feature scaling, informed by FairBatching's validated ±1.3% per-step approach. Addresses BC-2 (KV feature scaling) and BC-3 (CodeLlama-34B anomaly).

**LatencyModel methods covered:** Primary: StepTime. Secondary: All others inherited from Round 2.

**Go integration path:** Coefficient export — add one coefficient (`total_context`) to the existing StepML artifact JSON. The Go evaluator already supports arbitrary `feature_coefficients` maps.

## Prior Round Context

- **Round 2 (Idea 2, H1):** Raw KV features (kv_sum, kv_max, kv_mean) in Ridge regression were **counter-productive** (+20.5pp worse: 64.4% with KV vs 43.9% without KV). Root cause: kv_sum ranges 0–64,000+ causing Ridge coefficient instability.
- **Round 2 diagnosis (problem.md):** "This is a formulation problem, not a feature problem — the features have signal but raw linear regression cannot handle their dynamic range. Feature scaling (StandardScaler), log-transform of features, or nonlinear models would likely fix this."
- **Literature:** FairBatching [arXiv 2510.14392] achieves ±1.3% per-step error with `a + b*total_new_tokens + c*total_context`. BiScale [arXiv 2602.18755] uses summary statistics (sum/mean/std) of sequence lengths with gradient boosting trees for 2.7–2.9% MAPE. HERMES [arXiv 2504.09775] adds quadratic terms for 2.5% avg error.

## Training Strategy

Per-model training (mandatory per Round 1+2 findings). Temporal split (60/20/20) within each experiment. Regime separation (decode-only vs mixed) inherited from Round 2. Feature scaling applied per-model training set statistics.

**Data split:** Temporal 60/20/20 per experiment. Training: first 60% of steps. Validation: next 20%. Test: final 20%.

---

## Sub-Hypothesis H1: FairBatching 3-Coefficient Formulation

### Claim

A 3-coefficient linear model `step_time = a + b*(prefill_tokens + decode_tokens) + c*kv_sum` per model achieves < 30% mean per-step MAPE on the test set, outperforming both the Round 2 regime ensemble with KV (64.4%) and without KV (43.9%).

### Rationale

FairBatching's formulation treats `total_context` (= sum of all request context lengths in the batch, equivalent to BLIS's `kv_sum`) as a single linear feature with its own coefficient, rather than including it alongside multiple KV features in a regularized Ridge regression. The key difference: OLS (ordinary least squares) with 3 coefficients avoids Ridge's regularization penalty that distorts the kv_sum coefficient. With only 3 parameters, there is no multicollinearity to regularize away.

### Method

1. Load step-level data + KV features for all 10 experiments (4 models × 3 workloads, excluding reasoning)
2. For each model, fit OLS: `step_time = a + b*(prefill_tokens + decode_tokens) + c*kv_sum` on training set (60%)
3. Also fit variants: (a) separate prefill/decode coefficients: `a + b*prefill + c*decode + d*kv_sum`, (b) with regime split
4. Apply overhead floor: `step_cycle = max(overhead, prediction)` with per-model overhead constants from Round 2
5. Evaluate per-step MAPE on test set per experiment
6. Compare against Round 2 baselines: 43.9% (no KV) and 64.4% (KV + Ridge)

### Refutation Criteria

- **Supported:** Mean per-step MAPE < 30% on test set. KV feature (kv_sum) has positive, stable coefficient across all models.
- **Refuted:** Mean per-step MAPE > 43.9% (worse than Round 2 without KV) — the FairBatching formulation does not work for this dataset.

### Diagnostics

- Per-experiment MAPE, Pearson r, p99 error
- Coefficient values and stability across models
- Comparison: OLS vs Ridge, 3-coeff vs 4-coeff, regime vs no-regime
- Feature importance: contribution of kv_sum coefficient to total prediction

---

## Sub-Hypothesis H2: Feature Scaling Comparison

### Claim

Proper feature scaling (StandardScaler or log-transform of kv features) applied to the Round 2 feature set makes KV features **productive** in Ridge regression, reducing MAPE below the 43.9% no-KV baseline.

### Rationale

Round 2's KV feature failure was diagnosed as a scaling problem: kv_sum ranges 0–64,000 while prefill/decode tokens range 0–2048. StandardScaler normalizes all features to zero-mean, unit-variance, eliminating the dynamic range mismatch. Log-transform compresses the kv_sum range from [0, 64K] to [0, 11]. Either approach should allow Ridge to find stable coefficients.

### Method

1. **StandardScaler:** Fit StandardScaler on training set per model, transform train/test. Fit Ridge on scaled features (same feature set as Round 2: prefill, decode, num_prefill_reqs, num_decode_reqs, kv_sum, kv_max, kv_mean, kv_std + interactions)
2. **Log-transform:** Apply `log1p()` to kv_sum, kv_max, kv_mean, kv_std before fitting Ridge. Keep target in raw space (`use_log_target=False`)
3. **Combined:** StandardScaler + log-transform of KV features
4. Compare all variants against Round 2's 43.9% (no KV) and 64.4% (raw KV + Ridge)
5. Apply overhead floor and evaluate per-step MAPE

### Refutation Criteria

- **Supported:** At least one scaling variant achieves MAPE < 43.9% (proving KV features become productive with proper scaling).
- **Refuted:** All scaling variants are worse than 43.9% — the KV features from ProgressIndex genuinely lack signal for step-time prediction (not just a scaling issue).

### Diagnostics

- Per-variant MAPE comparison table
- Ridge coefficient magnitudes: scaled vs unscaled
- Per-regime analysis: does scaling help more in decode-only or mixed regime?
- CodeLlama-34B specific: does scaling help the 34B anomaly?

---

## Sub-Hypothesis H3: BLIS E2E Validation + 34B Investigation

### Claim

The best step-time formulation from H1/H2, combined with overhead floor, achieves < 20% mean ITL error on BLIS E2E validation (improving from Round 2's 33.6%), and the CodeLlama-34B anomaly root cause is identified.

### Rationale

If H1 or H2 significantly reduces per-step MAPE, the ITL improvement should propagate to BLIS E2E. The overhead floor already handles 77.9% of steps (decode-only), so step-time improvements primarily affect the 22.1% mixed-batch steps — which are exactly where the unsolved experiments fail. CodeLlama-34B is the worst model (99.2% per-step MAPE, 2,901% E2E for general) and deserves targeted investigation.

### Method

1. Export the best model from H1/H2 as StepML artifacts (per model)
2. Run BLIS E2E validation using `validate_e2e.py` with workload-spec mode (same as Round 2 H2)
3. Compare per-experiment E2E/TTFT/ITL errors against Round 2's results
4. **34B deep-dive:** (a) Plot step-time distribution for 34B vs other models. (b) Examine batch composition patterns (more mixed batches? different KV patterns?). (c) Check coefficient stability for 34B. (d) Test whether a 34B-specific feature (e.g., batch_size × kv_mean interaction) helps. (e) Check data quality (outliers, sampling artifacts).

### Refutation Criteria

- **Supported:** Mean ITL error < 20% (improvement from 33.6%). At least 7/10 experiments have ITL < 15%. 34B root cause identified with actionable fix.
- **Refuted:** ITL error does not improve significantly (< 3pp reduction) — the overhead floor already captures most ITL signal, and step-time improvements have diminishing returns for ITL.

### Diagnostics

- Full per-experiment E2E/TTFT/ITL error table (this round vs Round 2)
- 34B-specific analysis: step-time distribution, batch composition, coefficient stability
- Feature importance analysis per model: which features contribute most per regime
- Mixed-heavy threshold sweep results (64, 128, 256 prefill tokens)
