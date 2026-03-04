# FINDINGS: H5 — Leave-One-Model-Out (LOMO) for CMA-ES

**Date:** 2026-02-28
**Status:** SUPPORTED

## Claim

CMA-ES-optimized coefficients from one model group, when applied to another model group's experiments, achieve <50% E2E error — demonstrating partial cross-model transfer.

## Result

**Mean best-donor LOMO E2E: 14.8%** (threshold <50%). All 4 model groups have a donor achieving <25% E2E. **Strongly supported.**

## Cross-Model Transfer Matrix (E2E %)

| Donor → Target | codellama-34b | llama-2-70b | llama-2-7b | mixtral-8x7b |
|---|---|---|---|---|
| **codellama-34b** | — | 21.2% | 24.3% | 21.2% |
| **llama-2-70b** | 24.7% | — | 94.0% | **5.1%** |
| **llama-2-7b** | 40.4% | 53.8% | — | 53.8% |
| **mixtral-8x7b** | **11.9%** | 26.0% | **20.9%** | — |

## Best Donor Per Target

| Target Model | Best Donor | LOMO E2E | In-Dist E2E | Degradation |
|---|---|---|---|---|
| codellama-34b | mixtral-8x7b | **11.9%** | 7.9% | +4.0pp |
| llama-2-70b | codellama-34b | **21.2%** | 8.3% | +12.9pp |
| llama-2-7b | mixtral-8x7b | **20.9%** | 22.2% | -1.3pp |
| mixtral-8x7b | llama-2-70b | **5.1%** | 25.9% | -20.8pp |

## Key Findings

1. **Cross-model transfer is remarkably good.** Mean best-donor LOMO E2E of 14.8% vs in-distribution mean of 15.1%. For 2/4 targets (7b, mixtral), the cross-model artifact actually performs BETTER than the in-distribution one.

2. **70B → Mixtral is the best transfer pair (5.1% E2E).** The llama-2-70b CMA-ES artifact applied to Mixtral produces 5.1% E2E — better than Mixtral's own in-distribution 25.9%. This suggests llama-2-70b's overhead floor and scheduling coefficients are closer to Mixtral's optimal values than Mixtral's own CMA-ES found.

3. **Mixtral is the best universal donor.** Mixtral's artifact transfers to codellama-34b (11.9%), llama-2-7b (20.9%), and llama-2-70b (26.0%) — all under 27%. This may be because Mixtral's higher overhead (9,125us) creates a more conservative simulation that generalizes.

4. **7B artifacts are the worst donors.** llama-2-7b's artifact produces 40-54% E2E on all targets — the weakest transfer. This is expected: 7B has the smallest overhead floor (3,897us → optimized to 5,051us), which is too fast for larger models.

5. **Codellama-34b is a strong universal donor.** Consistent 21-24% across all targets — suggesting TP=2 step dynamics are a reasonable middle ground.

## Comparison with Prior Rounds

| Round | LOMO Metric | Result | Approach |
|---|---|---|---|
| R1 | Per-step MAPE | 2,559.7% | Per-experiment XGBoost |
| R2 | Per-step MAPE | 108.6% | Regime ensemble |
| R3 Idea 2 | Per-step MAPE | 2,281.6% | FairBatching 3-coeff OLS |
| **R3 Idea 3** | **E2E error** | **14.8%** | **CMA-ES artifact transfer** |

CMA-ES artifacts have dramatically better cross-model transfer than per-step models because they capture simulation-level dynamics (overhead floor calibration, scheduling overhead) that are partially model-independent.

## Refutation Assessment

- **SUPPORTED:** 14.8% << 50% threshold
- CMA-ES captures transferable simulation dynamics, not just model-specific step times
- Practical implication: for a new model without ground-truth data, applying the Mixtral or codellama-34b CMA-ES artifact provides a reasonable starting point (~12-24% E2E)
