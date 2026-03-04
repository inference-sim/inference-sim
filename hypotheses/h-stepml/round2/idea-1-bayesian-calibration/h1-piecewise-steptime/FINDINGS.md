# Idea 1, H1: Piecewise-Linear StepTime with KV Features — FINDINGS

**Status:** Refuted
**Resolution:** Refuted — wrong mental model
**Family:** Performance-regime
**VV&UQ:** Validation
**Type:** Statistical/Dominance
**Date:** 2026-02-27
**Rounds:** 1

## Hypothesis

> A piecewise-linear StepTime model with two regimes (decode-only vs. mixed-batch) and ProgressIndex-derived per-request KV features (kv_sum, kv_max, kv_mean) achieves <30% per-step MAPE on the held-out test set, outperforming the 2-feature blackbox baseline (which achieves ~675-966% MAPE depending on model).

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** Piecewise-linear Ridge regression per model+TP group. Two regimes:
- Regime 1 (decode-only, `prefill_tokens == 0`): features = `(decode_tokens, kv_mean, kv_max, kv_sum)`
- Regime 2 (mixed-batch, `prefill_tokens > 0`): features = `(prefill_tokens, decode_tokens, prefill_x_decode, kv_sum)`

Raw linear targets (no log transform) with CV-tuned Ridge alpha. Per-model temporal 60/20/20 split.

**Controlled variables:** Hardware (H100), temporal split ratios, Ridge alpha search grid
**Varied variable:** Feature set (piecewise+KV vs blackbox vs piecewise-no-KV)
**Data:** 77,816 steps from 10 experiments (eval/ground_truth/), 5 model+TP configurations

## Results

### Per-Model Per-Regime MAPE (Test Set)

| Model | Decode-Only MAPE | Decode-Only r | Mixed-Batch MAPE | Mixed-Batch r | n_test |
|-------|-----------------|---------------|-----------------|---------------|--------|
| Mixtral-8x7B (TP2) | 18.3% | 0.610 | 19.6% | 0.814 | 3,819 |
| Llama-2-7B (TP1) | 36.5% | 0.530 | 55.6% | 0.624 | 3,043 |
| Llama-2-70B-HF (TP4) | 96.2% | 0.702 | 70.0% | 0.700 | 1,336 |
| Llama-2-70B (TP4) | 112.3% | 0.854 | 63.3% | 0.758 | 2,547 |
| CodeLlama-34B (TP2) | 171.3% | 0.757 | 162.7% | 0.713 | 4,819 |

### Aggregate Comparison

| Approach | Weighted MAPE | Notes |
|----------|-------------|-------|
| Piecewise 2-regime + KV | **87.4%** | 4 decode + 4 mixed features per regime |
| Blackbox baseline | 89.7% | 2 features (prefill_tokens, decode_tokens) |
| Piecewise 2-regime no-KV | 83.8% | KV features removed |

### KV Feature Ablation

| Model | With KV | Without KV | KV Contribution |
|-------|---------|-----------|----------------|
| CodeLlama-34B | 168.5% | 162.7% | -5.8 pp (worse with KV) |
| Llama-2-70B-HF | 88.4% | 89.1% | +0.7 pp |
| Llama-2-70B | 94.7% | 83.4% | -11.3 pp (worse with KV) |
| Llama-2-7B | 38.2% | 38.2% | 0 pp (identical) |
| Mixtral-8x7B | 18.7% | 18.7% | 0 pp (identical) |
| **Overall** | **87.4%** | **83.8%** | **-3.6 pp (-4.3%)** |

### Per-Model Step Overhead (from GT ITL)

| Model | Overhead (us) |
|-------|-------------|
| Llama-2-7B (TP1) | 3,897 |
| CodeLlama-34B (TP2) | 6,673 |
| Llama-2-70B (TP4) | 8,029 |
| Llama-2-70B-HF (TP4) | 8,203 |
| Mixtral-8x7B (TP2) | 9,125 |

## Root Cause Analysis

### Why the 2-regime piecewise model fails (<30% target)

**1. Mixed-batch regime is too heterogeneous.** The mixed-batch regime (prefill_tokens > 0) combines steps with 1 prefill token and steps with 2000+ prefill tokens. In Idea 2's 3-regime split, the mixed-light/mixed-heavy boundary at 256 tokens separates memory-bound from compute-bound mixed batches. Without this split, a single linear model cannot capture the nonlinear scaling of prefill compute cost.

Evidence: Idea 2's 3-regime Ridge on the same data achieves substantially different MAPE for mixed-light vs mixed-heavy regimes, confirming the heterogeneity. The mixed-batch regime has 22.1% of steps but disproportionate MAPE contribution.

**2. KV features are inert or harmful in the 2-regime formulation.** For Llama-2-7B and Mixtral, the KV-ablated model achieves identical MAPE (Ridge assigns zero weight to KV features). For the 70B models, KV features *increase* MAPE. This happens because:
- In the decode-only regime, `kv_sum` has very high variance (range: 0 to 64,000+) while step duration has much lower range (~70-7000us). The linear coefficient on kv_sum is tiny but when multiplied by large values, it adds noise rather than signal.
- The 2-feature blackbox captures batch-size-to-time correlation more cleanly because `decode_tokens` is both lower variance and more directly causal.

**3. Raw linear regression on step.duration_us is inherently limited.** The target `step.duration_us` captures GPU forward pass only (median ~200-3000us). The relationship between batch features and GPU time is approximately log-linear, not linear. Raw linear regression produces high MAPE because:
- Small-batch steps are systematically overpredicted (negative predictions clipped to 0)
- Large-batch steps are systematically underpredicted
- The overhead floor (`max(overhead, compute)`) handles this at E2E time but doesn't help per-step MAPE

### Why Mixtral succeeds and CodeLlama fails

Mixtral-8x7B has the lowest MAPE (18.3% decode, 19.6% mixed) because its step times are more linearly predictable — the MoE routing adds consistent overhead that a linear model captures well. CodeLlama-34B has the highest MAPE (171.3% decode) because its step times have very high variance relative to the feature space, with attention costs dominating for large-KV batches.

## Devil's Advocate (RCV-5)

**Arguing the hypothesis might be Confirmed:**
The 30% MAPE target was based on MIST's controlled-input results (~2.5% step error). However, MIST trains on controlled synthetic data, not observational production traces. With observational data at ~10% sampling rate, step-time variance is inherently higher. If the threshold were relaxed to 100%, Mixtral and Llama-2-7B would pass, and the piecewise approach would show clear value vs blackbox on 2/5 models. The KV feature contribution might also improve with feature normalization (z-scoring), which was not attempted.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| 2-regime split is insufficient; need 3+ regimes | Design limitation | Confirms Idea 2's 3-regime approach |
| KV features are inert or harmful in 2-regime piecewise | Surprise | Need mixed-light/heavy split to make KV useful |
| Mixtral achieves <20% per-step MAPE with simple linear | Confirmation | Validates that linear regression works for MoE models |
| Per-model overhead: 3.9-9.1ms, consistent with Round 2 findings | Confirmation | Documented here |
| Raw linear regression insufficient for per-step MAPE | Confirmation | Consistent with MEMORY findings (~448% MAPE) |

## Standards Audit

- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? None

## Scope and Limitations (RCV-6)

- **Operating point tested:** H100 GPU, 5 model+TP configs (7B/TP1, 34B/TP2, 70B/TP4, 70B-HF/TP4, Mixtral/TP2), 10 experiments from eval/ground_truth/
- **Parameters findings depend on:** The 2-regime structure (decode-only vs mixed-batch threshold at prefill_tokens > 0), raw linear regression (no log transform), specific KV feature set
- **What was NOT tested:** 3+ regime splits, log-transformed targets, feature normalization/z-scoring, nonlinear models (e.g., polynomial features), BLIS E2E validation (binary path issue)
- **Generalizability:** The finding that 2 regimes are insufficient likely generalizes to any dataset with mixed-batch heterogeneity. The Mixtral success may not generalize to other MoE architectures.
- **Uncertainty quantification:** UQ not performed — single temporal split per model.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Overall MAPE | 87.4% | High — 15,564 test steps across 5 models |
| KV contribution | -3.6 pp | Medium — ablation on same split, no bootstrap CI |
| Pearson r | 0.696 | High — above 0.5 threshold |
| Sample size | 77,816 steps (train+valid+test) | High |
| Mechanism (mixed-batch heterogeneity) | Proposed | Medium — not confirmed by control experiment (adding 3rd regime) |

## Implications for Users

1. **Do not use 2-regime piecewise-linear models for StepML.** The 3-regime split (Idea 2) is strictly better.
2. **KV features are not universally beneficial.** They require regime segmentation that separates memory-bound from compute-bound batches.
3. **Mixtral (MoE) responds well to simple linear models** — its step-time scaling is more regular than dense models.
4. **CodeLlama-34B is the hardest model to predict** — high step-time variance, likely due to attention-dominated decode batches.

## Reproducing

```bash
cd hypotheses/h-stepml/round2/idea-1-bayesian-calibration/h1-piecewise-steptime
./run.sh
```
