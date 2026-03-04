# H2: Constrained CMA-ES Residual Tuning

**Status:** REFUTED

## Key Numbers

| Metric | Result | Target | Verdict |
|--------|--------|--------|---------|
| Mean E2E error | 27.5% | <12% | FAIL |
| Mean ITL error | 43.7% | <18% | FAIL |
| Mean TTFT error | 33.8% | - | - |
| E2E <10% | 0/10 | 6/10 | FAIL |
| E2E <20% | 1/10 | - | - |

## Method

CMA-ES optimization with +-30% bounds around H1 base coefficients, searching
over 5 parameters per model:
- scale_beta0, scale_beta1, scale_beta2 (multiplicative around H1 values)
- scale_alpha0 (multiplicative around H1 TTFT)
- alpha2 (output token processing time, absolute, range [0, 2000]us)

Objective: `f(x) = 0.5 * mean_E2E + 0.5 * mean_ITL + 5.0 * max(0, ITL - 0.20)`

Settings: sigma0=0.08, maxfevals=60 per model, seed=42.

### Optimized Coefficients

| Model | beta0 | beta2 | alpha0 | Direction |
|-------|-------|-------|--------|-----------|
| codellama-34b | 9,937 (0.70x) | 20.1 (0.78x) | 49,899 (1.05x) | Reduced step time |
| llama-2-70b | 12,594 (0.70x) | 27.3 (0.78x) | 82,667 (1.05x) | Reduced step time |
| llama-2-7b | 6,819 (0.70x) | 9.7 (0.71x) | 21,002 (0.77x) | Reduced step time |
| mixtral-8x7b-v0-1 | 13,245 (0.70x) | 6.2 (0.71x) | 48,592 (0.77x) | Reduced step time |

## Key Findings

1. **CMA-ES made H1 worse, not better.** H1 achieved 5.7% mean E2E; CMA-ES
   degraded it to 27.5%. The ITL improvement (from 107.8% to 43.7%) came at
   the cost of massive E2E degradation.

2. **The E2E/ITL tradeoff is fundamental.** CMA-ES consistently pushed beta0
   down to the lower bound (0.70x) for all models, trying to reduce the step
   time to improve ITL. But this directly degrades E2E since the step time
   was precisely calibrated in H1.

3. **The dual objective penalizes the sweet spot.** The ITL penalty term
   (`5.0 * max(0, ITL - 0.20)`) dominates the objective when ITL > 20%,
   forcing CMA-ES to sacrifice E2E to bring ITL down. But ITL ~100% is
   structural (see H1 findings), so CMA-ES fights against the simulator's
   inherent behavior.

4. **All models hit the lower bound on beta0** (0.70x), suggesting that
   even wider bounds would produce even worse E2E while continuing to
   reduce ITL toward the penalty threshold.

## Comparison to H1 and R3

| Metric | H1 | H2 (CMA-ES) | R3 CMA-ES |
|--------|-----|-------------|-----------|
| Mean E2E | 5.7% | 27.5% | 15.1% |
| Mean ITL | 107.8% | 43.7% | 87.4% |
| E2E <10% | 9/10 | 0/10 | 4/10 |

The hybrid calibration hypothesis is partially refuted: the principled base
(H1) is excellent for E2E, but CMA-ES residual tuning cannot improve it
without destroying E2E accuracy. The E2E/ITL tradeoff is not softened by
a better base -- it is hardened because the base is already optimal for E2E.

## Root Cause

The constrained CMA-ES fails because:
1. H1's calibration already minimizes E2E error by construction
2. ITL error is structural (BLIS reports per-step-time as ITL, which is 10-19ms,
   while ground truth ITL is ~4-10ms mean due to bimodal distribution)
3. The only way to reduce ITL is to reduce step time, which directly increases
   E2E error
4. The +-30% bounds cannot escape this local optimum
