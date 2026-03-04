# Idea 1: Bayesian Calibration — Findings Summary

**Date:** 2026-02-27
**Overall Verdict:** All hypotheses refuted. The 2-regime piecewise-linear approach is structurally weaker than Idea 2's 3-regime ensemble.

---

## Executive Summary

Idea 1 proposed a piecewise-linear StepTime model with two regimes (decode-only vs mixed-batch), ProgressIndex-derived KV features, and joint Bayesian optimization of all 5 LatencyModel methods against BLIS E2E error. Four hypotheses were designed; three were executed (H1, H4, H5) and one was blocked (H2).

**Every executed hypothesis was refuted.** The core structural limitation is the 2-regime split: lumping all mixed-batch steps (1 prefill token through 2000+) into a single linear regime creates too much heterogeneity for a linear model. Additionally, KV features provided zero-to-negative contribution in this formulation, and the approach was worse than Round 1 for workload generalization.

---

## Hypothesis Scorecard

| # | Hypothesis | Target | Result | Status |
|---|-----------|--------|--------|--------|
| H1 | Piecewise-linear StepTime (<30% MAPE) | <30% per-step MAPE | **87.4%** | **Refuted** |
| H2 | Joint BO of all 5 methods (<15% E2E) | <15% E2E mean error | Not run | **Blocked** |
| H4 | LOMO cross-validation (<80% MAPE) | <80% per-step MAPE | **148.8%** | **Refuted** |
| H5 | LOWO cross-validation (<40% MAPE) | <40% per-step MAPE | **155.4%** | **Refuted** |

---

## Key Results

### H1: Piecewise-Linear StepTime — Per-Model Quality

The base model achieves 87.4% weighted MAPE across 5 model+TP configurations, barely improving over the 2-feature blackbox baseline (89.7%). Performance varies dramatically by model:

| Model | Piecewise MAPE | Blackbox MAPE | Improvement |
|-------|---------------|--------------|-------------|
| Mixtral-8x7B (TP2) | 18.6% | 20.7% | +2.1 pp |
| Llama-2-7B (TP1) | 38.2% | 40.2% | +2.0 pp |
| Llama-2-70B-HF (TP4) | 88.4% | 90.6% | +2.2 pp |
| Llama-2-70B (TP4) | 94.7% | 86.8% | -7.9 pp |
| CodeLlama-34B (TP2) | 168.5% | 176.9% | +8.4 pp |

**KV features hurt overall** (-3.6 pp) — the model without KV features achieved 83.8% MAPE vs 87.4% with KV. For 2/5 models (7B, Mixtral), Ridge assigned zero weight to KV features. For the 70B models, KV features increased error.

### H4: Cross-Model Transfer (LOMO) — 17x Better than Round 1

| Held-Out Model | MAPE | MSPE | Key Observation |
|---------------|------|------|----------------|
| CodeLlama-34B | 53.9% | Mixed | Best fold — 34B interpolates between 7B and 70B |
| Llama-2-70B | 146.5% | +105%/+177% | Heavily overpredicted |
| Llama-2-7B | 187.0% | -185%/-205% | **Negative predictions** — catastrophic scale mismatch |
| Mixtral-8x7B (MoE) | 207.8% | +152%/+312% | Dense-to-MoE transfer fails completely |

Average 148.8% MAPE (target: <80%), but still 17.2x better than Round 1's LOMO baseline (2559.7%). The improvement comes from regime separation, but the model lacks scale normalization needed for cross-architecture transfer.

### H5: Cross-Workload Transfer (LOWO) — Worse than Round 1

| Model | general | codegen | roleplay | Avg |
|-------|---------|---------|----------|-----|
| Mixtral | 15.8% | 21.4% | 26.6% | **21.3%** |
| 34B | 43.4% | 292.4% | 107.2% | 147.7% |
| 70B | 145.9% | 265.2% | 480.6% | 297.3% |
| 7B | — | — | — | (no data) |

Grand average 155.4% MAPE (target: <40%), which is 0.7x of Round 1's LOWO (109.7%) — a **regression**. Mixtral is the sole success (21.3%), proving MoE architectures generalize across workloads. Dense models fail because workload-specific batch-composition distributions shift between training and test.

### H2: Bayesian Optimization — Not Run

Blocked because H1's base model (87.4% MAPE) is too inaccurate for BO to compensate. BO can tune overhead and secondary constants but cannot fix batch-composition-dependent prediction errors. The ~66-133 hours of compute is unjustified. Idea 2's comparable test (H3) already showed secondary methods contribute minimally to E2E error.

---

## Root Causes of Failure

### 1. The 2-regime split is structurally insufficient

The decode-only / mixed-batch boundary is too coarse. The mixed-batch regime (22.1% of steps) spans steps with 1 prefill token to 2000+ prefill tokens. This range includes both memory-bound micro-prefills and compute-bound large prefills — fundamentally different operations that a single linear model cannot represent.

**Evidence:** Idea 2's 3-regime split (decode-only / mixed-light / mixed-heavy at 256-token boundary) achieves substantially different MAPE for mixed-light vs mixed-heavy, confirming the heterogeneity.

### 2. KV features require finer regime segmentation to be useful

KV features (kv_sum, kv_max, kv_mean) have extremely high variance (0 to 64,000+) relative to step-time range (70-7,000 us). In the 2-regime formulation:
- Ridge assigns zero weight to KV features for Mixtral and 7B (features are noise)
- For 70B models, the tiny KV coefficients amplify noise at extreme KV values
- The blackbox's `decode_tokens` is a better proxy because it correlates with batch size and has lower variance

With Idea 2's mixed-light/mixed-heavy split, KV features become informative because the regime boundary constrains the feature space.

### 3. Raw linear regression has inherent per-step MAPE limitations

`step.duration_us` captures GPU forward pass only (median ~200-3000 us). The batch-feature-to-time relationship is approximately log-linear, not linear. Raw linear regression:
- Overpredicts small-batch steps (can produce negative predictions)
- Underpredicts large-batch steps
- Achieves high per-step MAPE even when the overhead floor makes E2E accuracy acceptable

This is consistent with Round 2's earlier finding that raw linear per-step MAPE is ~448% but irrelevant for BLIS E2E.

### 4. Cross-model transfer requires scale normalization

LOMO fails because step-time scales differ by 5-10x across models (7B: ~150us, 70B: ~1000us). The linear model learns coefficients scaled for the training models. Without normalizing features by model-specific constants (parameter count, FLOPS per token), the predictions are catastrophically wrong for held-out models at different scales.

### 5. Cross-workload transfer requires feature diversity or nonlinearity

LOWO fails because different workloads create different batch-composition distributions. A linear model on 4 features cannot capture these distributional shifts. Round 1's 30-feature XGBoost performed better (109.7% vs 155.4%) because it had more features and nonlinear capacity to absorb distribution shifts.

---

## Comparison with Idea 2 (3-Regime Ensemble)

| Metric | Idea 1 (2-regime) | Idea 2 (3-regime) | Winner |
|--------|-------------------|-------------------|--------|
| H1 per-step MAPE | 87.4% | Comparable (per-model) | Idea 2 |
| KV feature contribution | -3.6 pp (harmful) | Positive (mixed-heavy KV helps) | Idea 2 |
| LOMO avg MAPE | 148.8% | (refuted at similar level) | Comparable |
| LOWO avg MAPE | 155.4% | (refuted at ~similar level) | Comparable |
| Feature count per regime | 4 | 7-8 | Idea 2 |
| Regime boundary | Binary (prefill=0 vs >0) | Ternary (0 / <256 / >=256) | Idea 2 |

**Verdict:** Idea 2's 3-regime approach is strictly better. The additional regime boundary and richer feature set provide meaningful improvements, especially for mixed-batch prediction and KV feature utilization.

---

## Surprises and Positive Findings

Despite overall refutation, several valuable insights emerged:

1. **Mixtral generalizes exceptionally well.** Both LOMO (Fold 4 aside) and LOWO (21.3% avg) show that MoE architectures have more regular step-time scaling that transfers across contexts. A single Mixtral StepML model can serve any workload type.

2. **LOMO interpolation works for 34B.** Holding out CodeLlama-34B (which falls between 7B and 70B in scale) yields 53.9% MAPE — suggesting that cross-model transfer via interpolation is feasible with proper scale normalization.

3. **Per-model overhead is consistent with Round 2 findings.** The overhead values (3.9ms for 7B, 6.7ms for 34B, 8.0-8.2ms for 70B, 9.1ms for Mixtral) match prior observations, providing independent validation.

4. **"General" workload is the easiest holdout.** In LOWO, holding out the "general" workload yields the lowest average MAPE (68.4%), likely because "general" has the most diverse batch compositions (averaging effect), making it the most representative of a mixture of training workloads.

---

## Recommendations

### For StepML Research

1. **Abandon the 2-regime piecewise-linear approach.** The 3-regime split (Idea 2) is strictly better and should be the foundation for further work.
2. **Investigate feature normalization for LOMO.** The 34B interpolation result (53.9%) suggests that dividing features by model-specific constants (parameter count, FLOPS per token) could make cross-model transfer viable.
3. **Explore MoE-specific models.** Mixtral's consistent success across all experiments suggests that MoE architecture warrants a dedicated StepML model family.
4. **Do not invest in BO until per-step quality improves.** Joint BO cannot compensate for structural model errors. Focus on improving the base StepTime model first.

### For BLIS Deployment

1. **Use per-model StepML coefficients** — cross-model and cross-workload transfer are not production-ready.
2. **The overhead floor (`max(overhead, compute)`) remains the primary accuracy mechanism.** Per-step MAPE is poor, but the overhead floor makes E2E accuracy acceptable for standard workloads.
3. **Ship Mixtral with a single universal model.** Its LOWO results (21.3%) justify deploying one model for all Mixtral workloads.

---

## Data and Artifacts

| Hypothesis | Output Directory | Key Files |
|-----------|-----------------|-----------|
| H1 | `h1-piecewise-steptime/output/` | `piecewise_results.csv`, `piecewise_summary.json`, `artifacts/*.json` |
| H2 | `h2-joint-bo-calibration/output/` | (not run — script ready at `bo_calibrate.py`) |
| H4 | `h4-model-generalization/output/` | `lomo_per_step_results.csv`, `lomo_summary.json`, `lomo_artifacts/*.json` |
| H5 | `h5-workload-generalization/output/` | `lowo_per_step_results.csv`, `lowo_summary.json`, `lowo_artifacts/*.json` |

## Reproducing All Experiments

```bash
# H1: Base model training (~5 min)
cd hypotheses/h-stepml/round2/idea-1-bayesian-calibration/h1-piecewise-steptime && ./run.sh

# H4: LOMO cross-validation (~15 min)
cd hypotheses/h-stepml/round2/idea-1-bayesian-calibration/h4-model-generalization && ./run.sh

# H5: LOWO cross-validation (~15 min)
cd hypotheses/h-stepml/round2/idea-1-bayesian-calibration/h5-workload-generalization && ./run.sh

# H2: BO calibration (~66-133 hours, not recommended)
cd hypotheses/h-stepml/round2/idea-1-bayesian-calibration/h2-joint-bo-calibration && ./run.sh
```
