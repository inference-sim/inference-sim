# Idea 1, H1: Piecewise-Linear StepTime with KV Features Achieves <30% Per-Step MAPE

**Status:** Refuted
**Family:** Performance-regime
**Type:** Statistical/Dominance
**Date:** 2026-02-26

## Hypothesis

> A piecewise-linear StepTime model with two regimes (decode-only vs. mixed-batch) and ProgressIndex-derived per-request KV features (kv_sum, kv_max, kv_mean) achieves <30% per-step MAPE on the held-out test set, outperforming the 2-feature blackbox baseline (which achieves ~675-966% MAPE depending on model).

This sub-hypothesis validates that the StepTime model itself has sufficient quality before investing in Bayesian optimization of all 5 LatencyModel methods in h2. If the piecewise-linear model fails here, BO cannot compensate.

## Refuted-If

- Per-step MAPE > 30% averaged across 4 model configurations
- OR Pearson r < 0.5 between predicted and actual step times
- OR KV features (kv_sum, kv_max, kv_mean) collectively contribute <10% to prediction accuracy vs. a model without them

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** Train piecewise-linear regression per model+TP group. Two regimes:
- Regime 1 (decode-only, `prefill_tokens == 0`): `b0 + b1*decode_tokens + b2*kv_mean + b3*kv_max`
- Regime 2 (mixed-batch, `prefill_tokens > 0`): `b4 + b5*prefill_tokens + b6*decode_tokens + b7*prefill_x_decode`

Compare against blackbox 2-feature baseline (prefill_tokens, decode_tokens only).

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Temporal 60/20/20 within each experiment. Pool all 4 workloads per model for training. Report per-model and aggregate MAPE.
**Baselines:** Blackbox 2-feature regression (current production model)
**Success metric:** Per-step MAPE < 30% averaged across 4 model configurations

**Training strategy:** Ordinary least squares per (model, regime) pair. No regularization initially; add Ridge (alpha tuned via 5-fold CV) if overfitting detected on validation set.

## Feature Set

8 features organized per regime:

**Decode-only regime (4):** decode_tokens, kv_mean, kv_max, kv_sum
**Mixed-batch regime (4):** prefill_tokens, decode_tokens, prefill_x_decode (interaction), kv_sum

KV features derived from ProgressIndex: for each Request in batch, `ProgressIndex = input_processed + output_generated`, which approximates the request's current KV cache length.

## BLIS E2E Claim

This sub-hypothesis does NOT report BLIS E2E directly (that is h2's scope). However, per-step MAPE < 30% is a necessary (but not sufficient) precondition for achieving <15% BLIS E2E mean error in h2.

## Related Work

- **MIST** (arXiv:2504.09775): Step-level ML ensemble achieves 2.5% per-step error with regime-specific training on controlled inputs. This hypothesis adapts MIST's insight (regime-specific + KV features) to observational data.
- **Vidur** (MLSys 2024): Per-model calibration with operator triaging. Validates per-model training strategy.
- **BLIS H8** (internal): 12.96x overestimate without per-request KV lengths. KV features directly address this.
- **Kennedy & O'Hagan (2001):** Bayesian calibration of computer models. Motivates the E2E optimization in h2 but requires a reasonable base model first.

## Go Integration Path

Piecewise-linear coefficients exported as JSON matching `StepMLArtifact` schema. `StepMLLatencyModel` extended with regime dispatch: check `prefill_tokens > 0` to select regime, then apply the corresponding linear model. Minimal change to `sim/latency/stepml.go`.

## Data Integrity

- Training and test sets are temporally non-overlapping (no leakage)
- KV features derived solely from ProgressIndex (available at inference time in the LatencyModel interface)
- No roofline predictions used as features or targets (BC-3-7)
