# Idea 2, H1: Regime-Specific Ridge Regression with KV Features Achieves <15% Per-Step MAPE

**Status:** Pending
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-26

## Hypothesis

> A 3-regime ensemble of per-model Ridge regressors -- (A) decode-only, (B) mixed-light (prefill < 256 tokens), (C) mixed-heavy (prefill >= 256 tokens) -- trained with ProgressIndex-derived KV features (kv_sum, kv_max, kv_mean, kv_std) achieves <15% per-step MAPE, outperforming both Round 1's global XGBoost (34% MAPE) and the blackbox baseline (675-966% MAPE).

The key insight from MIST is that fewer, higher-quality features on regime-specific data outperforms many features on heterogeneous data. This sub-hypothesis validates the regime+KV combination before BLIS E2E validation in h2.

## Refuted-If

- Per-step MAPE > 20% averaged across all 4 model configurations
- OR Per-step MAPE > 30% for ANY single model configuration
- OR KV features (kv_sum, kv_max, kv_mean, kv_std) contribute <15% improvement in MAPE vs. a regime-specific model without KV features
- OR Regime B or C has MAPE > 50% (indicating insufficient sample size for mixed-batch regimes)

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** Train Ridge regression per (model, regime) pair. 4 models x 3 regimes = 12 regressors.

**Regime classification (from step features):**
- Regime A (decode-only): `prefill_tokens == 0` (~80.6% of steps)
- Regime B (mixed-light): `0 < prefill_tokens < 256` (~15% of steps)
- Regime C (mixed-heavy): `prefill_tokens >= 256` (~4.4% of steps)

**Feature engineering per regime:**

Decode-only (Regime A, 7 features):
- `decode_tokens`, `num_decode_reqs`
- `kv_sum`, `kv_max`, `kv_mean`, `kv_std` (from ProgressIndex)
- `num_decode_reqs * kv_mean` (interaction: batch_size x context_length)

Mixed-batch (Regimes B and C, 8 features):
- `prefill_tokens`, `decode_tokens`, `num_prefill_reqs`, `num_decode_reqs`
- `kv_sum`, `kv_max`
- `prefill_tokens * decode_tokens` (interaction)
- `prefill_tokens^2` (quadratic prefill -- attention cost is O(seq_len^2))

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Temporal 60/20/20 within each experiment. Pool all 4 workloads per model for training.
**Baselines:** (a) Blackbox 2-feature regression, (b) Round 1 global XGBoost (34% MAPE), (c) Regime-specific Ridge WITHOUT KV features (ablation)
**Success metric:** Per-step MAPE < 15% averaged across 4 model configurations

**Regularization:** Ridge alpha tuned via 5-fold CV within training set per (model, regime) pair.

## Sensitivity Analysis

Sweep the Regime B/C boundary threshold from 128 to 512 tokens in 64-token increments. Report MAPE at each threshold. Also test a 2-regime model (decode-only vs. any-mixed) as a simpler alternative.

## Related Work

- **MIST** (arXiv:2504.09775): Ensemble of regressors trained on distinct data subsets achieves 2.5% step-level error. Direct inspiration for regime-switching approach.
- **Vidur** (MLSys 2024): Operator triaging classifies operations by input dependencies. Decode-only vs. mixed-batch is analogous to Vidur's token-level vs. sequence-level distinction.
- **AIConfigurator** (arXiv:2601.06288): Explicitly models three serving modes (static, aggregated, disaggregated), validating the 3-regime concept.
- **BLIS H8** (internal): 12.96x overestimate without per-request KV lengths. ProgressIndex KV features directly address this.
- **Round 1 findings**: XGBoost feature importance showed kv_blocks_used (#1) and running_depth (#3) as top features -- but as system-state proxies, not per-request. ProgressIndex provides per-request KV.

## BLIS E2E Claim

This sub-hypothesis reports per-step MAPE (diagnostic). BLIS E2E validation is h2's scope. However, per-step MAPE < 15% combined with MIST's evidence (2.5% step error maps to accurate simulation) gives high confidence that h2 will succeed.

## Go Integration Path

12 Ridge coefficient vectors (4 models x 3 regimes) exported as `StepMLArtifact` JSON with a `regimes` array. Each entry has a `condition` field (regime classification rule) and a `LinearModel` (coefficients + intercept). `extractBatchFeatures()` in `stepml.go` extended with `kv_std` computation (one-pass: track sum and sum-of-squares). Regime dispatch added to `StepTime()`.

## Training Strategy and Data Split

- **Training:** 60% temporal split from each experiment, pooled per model across 4 workloads
- **Validation:** 20% temporal split (hyperparameter tuning for Ridge alpha)
- **Test:** 20% temporal split (final MAPE reporting)
- **Regime-specific sample sizes (estimated):** Regime A: ~99,000 steps, Regime B: ~18,400 steps, Regime C: ~5,400 steps
- **Minimum sample requirement:** >500 steps per (model, regime) pair for Ridge. Regime C may need workload pooling.

## Data Integrity

- Temporal split ensures no future data leaks into training
- KV features from ProgressIndex (available at inference time, no look-ahead)
- Regime classification uses only step-level features (prefill_tokens), no oracle information
- No roofline predictions used (BC-3-7)
