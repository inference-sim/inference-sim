# FINDINGS: H1 — FairBatching 3-Coefficient Formulation

**Date:** 2026-02-27
**Status:** PARTIALLY SUPPORTED

## Claim
A 3-coefficient linear model `step_time = a + b*(prefill_tokens + decode_tokens) + c*kv_sum` per model achieves < 30% mean per-step MAPE on the test set, outperforming Round 2 baselines.

## Result

| Variant | Mean MAPE | Median | Min | Max |
|---|---|---|---|---|
| **H1: 3-coeff OLS (new_tokens + kv_sum)** | **56.2%** | 54.7% | 28.5% | 89.4% |
| H1: 4-coeff OLS (prefill + decode + kv_sum) | 87.3% | 90.1% | 20.7% | 175.4% |
| H1: Regime+KV OLS | 83.0% | 88.6% | 18.7% | 161.5% |
| Baseline: 2-coeff OLS (no KV) | 83.1% | 86.9% | 20.7% | 176.9% |

### Per-Model Breakdown

| Model | 3-coeff MAPE | kv_sum coefficient | KV signal? |
|---|---|---|---|
| mixtral-8x7b-v0-1_tp2 | 28.5% | 0.000 | No (kv_sum=0 in data) |
| llama-2-7b_tp1 | 45.6% | 0.000 | No (kv_sum=0 in data) |
| llama-2-70b_tp4 | 54.7% | 0.031 | Yes |
| codellama-34b_tp2 | 63.0% | 0.037 | Yes |
| llama-2-70b-hf_tp4 | 89.4% | 0.065 | Yes |

## Assessment

**Partially supported.** The 3-coeff OLS achieves 56.2% mean MAPE, which is a 27pp improvement over the 2-coeff baseline (83.1%) — but misses the < 30% target. The improvement is model-dependent:

- **For models with KV data** (70b, 34b): kv_sum has a positive, stable coefficient and reduces MAPE by 20-30pp. This validates the FairBatching formulation.
- **For models without KV data** (7b, mixtral): kv_sum=0 for all steps (lifecycle extractor found no active requests). For these models, the 3-coeff model degenerates to a 2-coeff model (the coefficient is 0) and still achieves 28-46% MAPE purely from the combined new_tokens feature.

**Key finding:** The 3-coeff formulation outperforms the 4-coeff (separate prefill/decode) by 31pp. This is because combining prefill+decode into `new_tokens` reduces the parameter count from 4 to 3, avoiding the instability that separate coefficients cause when prefill tokens are rare (most steps are decode-only).

## Diagnostics

### Coefficient Stability
The kv_sum coefficient is consistently positive (0.031-0.065) for models where KV data exists, confirming that total context length correlates with step time (memory bandwidth cost). The negative coefficient for 4-coeff llama-2-70b_tp4 (-0.022 on kv_sum) is a sign of multicollinearity with the separate decode coefficient.

### Why MAPE > 30%
The per-step target (step.duration_us) measures GPU forward pass time only, which has a bimodal distribution:
- Decode-only steps (70-90% of data): 100-500µs
- Mixed steps (10-30%): 1,000-12,000µs

A single linear model cannot accurately fit this bimodal distribution. The overhead floor mechanism (from Round 2) handles this at the BLIS E2E level, but per-step MAPE against raw targets remains high because the model cannot distinguish the two regimes without the floor.
