# Idea 2, H2: Learned Correction Factors Achieve <10% E2E Mean Error

**Status:** Not Supported (0/16 < 15% MAPE; avg 78.7% worse than blackbox 70.4%)
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> Learning 9-36 multiplicative correction factors for the analytical decomposition via nonlinear least squares achieves <10% workload-level E2E mean error on each of 16 experiments, with |MSPE| < 5% (low systematic bias).

## Refuted-If

The 36-parameter per-model variant achieves >10% E2E mean error on more than 4 of 16 experiments, OR |MSPE| > 10% (indicating the analytical backbone has a systematic bias that multiplicative corrections cannot fix). If even per-model correction factors cannot reach the target, the decomposition structure itself is flawed.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. **Model variants:** Test 3 correction factor configurations:
   - **9-parameter global:** One multiplicative factor per component (prefill_gemm_factor, prefill_attn_factor, decode_gemm_factor, decode_attn_factor) + one overhead term, shared across all models. 4 component factors + 1 overhead + 4 MFU discount factors = 9 parameters.
   - **16-parameter per-phase:** Separate correction factors for compute-bound vs. memory-bound regimes within each component (determined by arithmetic intensity threshold). 4 components x 2 regimes + 4 MFU + 4 regime thresholds = 16 parameters.
   - **36-parameter per-model:** 9 parameters per model family (4 models x 9 = 36). This is the ceiling of what per-model calibration can achieve.
2. **Fitting procedure:** Nonlinear least squares (scipy.optimize.least_squares with 'trf' method) on the 60% training split. The objective minimizes squared log-ratio: `sum((log(predicted/actual))^2)` to ensure symmetric relative errors.
3. **Evaluation:** Per-step MAPE on 20% test split, workload-level E2E mean error via BLIS simulation, and MSPE (mean signed percentage error) to detect systematic bias.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Temporal 60/20/20 within each experiment
**Baselines:** Raw analytical decomposition from h1 (no correction factors), Blackbox 2-feature model, XGBoost from Idea 1 h2 (if available)
**Success metric:** Workload-level E2E mean error < 10% on at least 12 of 16 experiments AND |MSPE| < 5%

**Bias analysis:** MSPE (Mean Signed Percentage Error) distinguishes systematic over/under-prediction from symmetric noise. |MSPE| < 5% means the model is not systematically biased. This is critical because E2E mean latency is sensitive to bias direction -- systematic overprediction inflates E2E mean while systematic underprediction deflates it.

**Parameter count vs. accuracy tradeoff:** Plot per-step MAPE vs. number of parameters (9, 16, 36) to characterize the diminishing returns curve. If 9 parameters achieve within 2% MAPE of 36 parameters, the simpler model is preferred for generalization.

## Related Work

- **Vidur** (Agrawal et al., MLSys 2024): Uses per-operation piecewise linear models calibrated from profiling data. Our approach is similar in spirit but uses multiplicative corrections on an analytical backbone rather than profiled breakpoints.
- **Calotoiu et al.** (SC 2013): Automated performance modeling for HPC applications using regression on analytical models. Demonstrates that small parameter counts (5-20) suffice when the analytical structure is correct.
- **Extended Roofline** (Wang et al., IEEE Micro 2020): Extends the roofline model with cache hierarchy effects via learned correction factors. Directly analogous to our approach of correcting analytical estimates.

## Go Integration Path

The correction factors would be stored in a `stepml_coefficients.json` file (or embedded in `defaults.yaml` alongside existing alpha/beta coefficients). The `StepMLLatencyModel` implementation would call `DecomposeStepTime()` from h1, then multiply each component by its learned factor: `total = prefill_gemm_factor * prefill_gemm_time + prefill_attn_factor * prefill_attn_time + ... + overhead`. This is a pure-Go implementation with no ML runtime dependency -- the correction factors are constants loaded at initialization. The 9-parameter global model ships as the default; per-model 36-parameter variants are available via `--stepml-model <model-specific-coefficients.json>`.
