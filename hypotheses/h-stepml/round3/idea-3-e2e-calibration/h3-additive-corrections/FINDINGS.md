# FINDINGS: H3 — Additive Correction Factors

**Date:** 2026-02-27
**Status:** REFUTED for E2E, SUPPORTED for TTFT

## Claim

Simple additive or multiplicative correction factors applied to TTFT and ITL close at least 5pp of remaining E2E error beyond what H1 achieves.

## Result

**Mean E2E error: 17.2%** (H1 was 15.1%). Corrections **worsened** E2E by +2.1pp.

**Mean TTFT error: 9.4%** (H1 was 67.6%). TTFT improved by **58.2pp** — a massive correction.

**Mean ITL error: 81.1%** (H1 was 87.4%). ITL improved marginally by 6.3pp.

## Per-Experiment Results

| Experiment | Model | H1 E2E | H3 E2E | H3 TTFT | H3 ITL |
|---|---|---|---|---|---|
| llama-2-7b-tp1-roleplay | llama-2-7b | 22.2% | 40.4% | 7.9% | 44.3% |
| llama-2-70b-tp4-general | llama-2-70b | 16.2% | 15.8% | 23.4% | 70.6% |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | 3.8% | 3.8% | **0.3%** | 110.0% |
| llama-2-70b-tp4-roleplay | llama-2-70b | 6.9% | 7.3% | 43.5% | 115.2% |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | 30.9% | 31.7% | **1.4%** | 38.2% |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | 12.7% | 13.4% | **0.5%** | 75.1% |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | 34.2% | 34.9% | **3.4%** | 31.2% |
| codellama-34b-tp2-general | codellama-34b | 1.2% | 1.7% | **6.5%** | 102.1% |
| codellama-34b-tp2-codegen | codellama-34b | 5.0% | 4.5% | **2.2%** | 89.5% |
| codellama-34b-tp2-roleplay | codellama-34b | 17.5% | 18.0% | **4.8%** | 134.7% |
| **MEAN** | | **15.1%** | **17.2%** | **9.4%** | **81.1%** |

## Correction Factors Per Model

| Model | TTFT Additive (ms) | TTFT Multiplicative | ITL Additive (ms) | ITL Multiplicative |
|---|---|---|---|---|
| llama-2-7b | +16.5 | 2.55× | -3.70 | 0.52× |
| llama-2-70b | +61.0 | 4.39× | -9.02 | 0.52× |
| llama-2-70b-hf | +37.6 | 3.08× | -10.26 | 0.47× |
| mixtral-8x7b-v0-1 | +40.2 | 2.86× | -4.95 | 0.67× |
| codellama-34b | +32.5 | 3.15× | -8.27 | 0.48× |

## Key Findings

1. **TTFT corrections work spectacularly:** Adding 16-61ms per model to QueueingTime produced 9.4% mean TTFT error (from 67.6%). 7/10 experiments < 10% TTFT.

2. **ITL corrections backfire:** The optimizer made ITL worse by overshooting step times. Additive ITL corrections (subtracting the overshoot) help marginally but can't fix the fundamental issue that the E2E optimizer sacrificed ITL accuracy.

3. **TTFT correction is NOT an E2E correction:** Fixing TTFT worsened E2E because the optimizer had already compensated for the TTFT under-prediction by *over-predicting* later in the request lifecycle. Adding TTFT correction on top double-counts.

4. **Residuals are systematic, not random:** TTFT residuals have near-zero std within a model (e.g., llama-2-7b: 0.0ms std), confirming this is a systematic simulation-level bias, not stochastic noise.

5. **The correction mechanism is too crude:** Applying corrections as constant offsets to QueueingTime and OutputTokenProcessingTime interacts with the simulation dynamics in unpredictable ways. A proper calibration would need to jointly optimize these corrections with the step-time coefficients.

## Refutation Assessment

- **REFUTED for E2E:** Corrections increased E2E error by 2.1pp (not decreased by 5pp as claimed)
- **SUPPORTED for TTFT:** TTFT corrections reduced TTFT error by 58.2pp
- **Root cause:** The H1 optimizer already "absorbed" the TTFT deficit into its E2E-optimal coefficients. Post-hoc corrections conflict with this absorption.
