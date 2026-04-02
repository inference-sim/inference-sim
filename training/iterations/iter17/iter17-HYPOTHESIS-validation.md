# Iteration 17: Hypothesis Validation

## Overall Results

| Metric | Iter16 (baseline) | Iter17 CMA-ES best | Change |
|--------|-------------------|--------------------|--------|
| Overall loss | 60.19% | **65.37%** | +5.18% worse |
| TTFT RMSE | 31.36% | 32.31% | worse |
| E2E RMSE | 28.83% | 33.06% | worse |
| Experiments | 15/15 | 15/15 | ✅ |

**Note**: Iter16 coefficients score exactly 60.19% on the iter17 simulator (same as original).
All comparisons use the corrected simulator with all 15 experiments.

---

## H-main: Re-Optimization Strictly Reduces Loss Below 60.19%

**Prediction**: Overall loss < 60.19%, Scout mean TTFT APE < 39%, β₁ recovers to 0.4–0.9,
α₂ retreats from 45.7µs toward 1–10µs.

**Result**: ❌ **NOT CONFIRMED**
- Overall loss: 65.37% (prediction: < 60.19%) ❌
- Scout mean TTFT APE: ~52% (prediction: < 39%) ❌
- β₁: 0.164 (prediction: 0.4–0.9) ❌ still collapsed
- α₂: 20.4µs (prediction: 1–10µs) — retreated from 45.7 ✅ but not to <10

**Root cause**: The core hypothesis was flawed. The `#877` fix was already in effect during
iter16's evaluations (the `interleave_moe_layer_step` field was already in Scout's config.json).
The loss landscape with 15 experiments has the iter16 point (60.19%) as one local minimum;
CMA-ES converged to a different local minimum (65.37%) rather than finding the iter16 basin
or anything better.

---

## H-scout: All Four Scout Experiments Improve Over Iter16

**Prediction**: Scout TTFT APE < iter16 values for all 4 experiments.

**Result**: ❌ **NOT CONFIRMED** — All four Scout experiments are worse in iter17 CMA-ES best

| Experiment | Iter16 TTFT APE | Iter17 TTFT APE | Change |
|---|---|---|---|
| reasoning-lite | 72.5% | **73.4%** | ❌ worse |
| codegen | 47.1% | **48.0%** | ❌ worse |
| general-lite | 41.1% | **45.3%** | ❌ worse |
| roleplay | 39.6% | **42.5%** | ❌ worse |

**Evidence**: CMA-ES found a coefficient set that trades Scout accuracy for better dense model
fit (several dense models improved). The fundamental Scout MoE prediction problem remains
unsolved regardless of training set size or search strategy.

---

## H-dense-stable: Dense Model Predictions Within ±5pp of Iter16

**Prediction**: Dense experiments (11/15) within ±5 TTFT APE percentage points of iter16.

**Result**: ⚠️ **MIXED** — Some stable, some shifted significantly

| Experiment | Iter16 TTFT APE | Iter17 TTFT APE | Within ±5pp? |
|---|---|---|---|
| Llama-2 codegen | 12.2% | 22.6% | ❌ +10.4pp |
| Llama-2 roleplay | 25.2% | 34.5% | ❌ +9.3pp |
| Llama-2 general | 22.6% | 8.6% | ❌ -14.0pp (improved) |
| Llama-2 reasoning-lite | 34.5% | 28.0% | ❌ -6.5pp (improved) |
| Llama-3.1 general-lite | 7.1% | 13.3% | ❌ +6.2pp |
| Llama-3.1 codegen | 10.7% | 8.5% | ✅ -2.2pp |
| Mistral general-lite | 29.4% | 28.1% | ✅ -1.3pp |
| Mistral codegen | 18.5% | 17.6% | ✅ -0.9pp |
| Qwen roleplay | 12.4% | 12.5% | ✅ +0.1pp |
| Qwen reasoning-lite | 3.9% | 4.0% | ✅ +0.1pp |
| Yi-34B general-lite | 0.1% | 5.7% | ❌ +5.6pp |

5 of 11 within ±5pp; 6 shifted more than 5pp. CMA-ES found a different region of coefficient
space where some dense experiments improved and others worsened.

---

## H-cv: All Three Cross-Validation Tests Pass at <15% MAPE

**Prediction**: CV-1, CV-2, CV-3 all pass with MAPE < 15%.

**Result**: ⬛ **NOT RUN**

Cross-validation tests were not executed for iter17. Given that the best iter17 coefficients
(CMA-ES, 65.37%) are worse than iter16's (60.19%), and the CV tests are designed to validate
generalization of the *best* coefficients, the CV tests would reflect worse generalization than
iter16 for the CMA-ES result.

**Recommendation for iter18**: Run CV tests only against the best available coefficients
(iter16's, at 60.19%) to establish the generalization baseline, or wait until iter18 finds
genuinely better coefficients before investing CV test compute.

---

## H-full-dataset: Training on All 15 Experiments Tightens Dense Model Fit

**Prediction**: Dense APE decreases for at least 3 of 6 previously-unseen Llama experiments.

**Result**: ⚠️ **MINIMALLY CONFIRMED** (3/6 improved, 3/6 worsened)

| Experiment | Iter16 TTFT APE | Iter17 TTFT APE | Improved? |
|---|---|---|---|
| Llama-2 general | 22.6% | 8.6% | ✅ -14.0pp |
| Llama-2 reasoning-lite | 34.5% | 28.0% | ✅ -6.5pp |
| Llama-3.1 codegen | 10.7% | 8.5% | ✅ -2.2pp |
| Llama-2 codegen | 12.2% | 22.6% | ❌ +10.4pp |
| Llama-2 roleplay | 25.2% | 34.5% | ❌ +9.3pp |
| Llama-3.1 general-lite | 7.1% | 13.3% | ❌ +6.2pp |

Exactly 3/6 improved (threshold was ≥3). The improvement was real but at the cost of
worsening 3 other Llama experiments. Adding Llama experiments to the training objective
caused the optimizer to make tradeoffs within the Llama family.

---

## H-convergence: Patience-Based Stopping Terminates Before 500 Trials

**Prediction**: Optimizer stops before 500 trials via `--patience 100` criterion.

**Result**: ✅ **CONFIRMED**
- Stopped at trial 188 / 500 budget (best at trial 178, patience fired at trial 278)
- CMA-ES run: 188/500 = 37.6% of budget used
- Wall-clock time: 35.8 minutes

The patience criterion worked correctly for both TPE (60 trials) and CMA-ES (188 trials).
The early stopping mechanism proved valuable in preventing wasted compute on converged searches.

---

## Summary

| Hypothesis | Prediction | Result | Status |
|---|---|---|---|
| H-main | Loss < 60.19% | 65.37% (worse) | ❌ FAILED |
| H-scout | Scout TTFT improves | All 4 Scout worse | ❌ FAILED |
| H-dense-stable | Dense within ±5pp | 5/11 stable, 6/11 shifted | ⚠️ PARTIAL |
| H-cv | All CV tests pass | Not run | ⬛ SKIPPED |
| H-full-dataset | ≥3/6 Llama experiments improve | Exactly 3/6 improved | ⚠️ PARTIAL |
| H-convergence | Stops before 500 trials | Stopped at 188 | ✅ CONFIRMED |

**Conclusion**: Iter17's core hypotheses were based on an incorrect assumption (that #877
changed the loss landscape). The actual optimization confirmed that iter16's coefficients
represent the deepest local minimum found so far on the 15-experiment landscape. No amount
of search strategy variation (TPE, CMA-ES, warm-start, parallel workers) produced better
coefficients than iter16.

**Iter17 practical outcome**: Confirmed that iter16's coefficients (60.19% loss, TTFT RMSE
31.4%, E2E RMSE 28.8%) correctly evaluate on all 15 experiments with the current simulator.
These remain the coefficients to deploy.
