# Idea 2, H3: Correction Factors Transfer Across Models

**Status:** Pending
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> The correction factors transfer across models: a globally-trained 9-parameter model achieves <20% per-step MAPE on held-out models in leave-one-model-out cross-validation. The analytical backbone provides structural generalization that pure data-driven approaches lack.

## Refuted-If

Leave-one-model-out MAPE > 30% for any held-out model with the 9-parameter global model. The 30% threshold (rather than 25% as in Idea 1) reflects that 9 parameters provide less flexibility than a tree ensemble -- but the analytical backbone should compensate by encoding physical relationships that transfer across architectures. Exceeding 30% means the correction factors are architecture-specific and the analytical backbone does not generalize as hypothesized.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. **Leave-one-model-out (LOMO) with 9-parameter global model:** For each of the 4 models, fit 9 correction factors on the remaining 3 models (12 experiments) and evaluate on the held-out model's 4 experiments.
2. **Comparison with Idea 1 LOMO:** Compare the 9-parameter analytical model's LOMO MAPE against XGBoost's LOMO MAPE from Idea 1 h3. This directly tests whether the analytical backbone provides better generalization than a purely data-driven approach with the same held-out evaluation.
3. **Factor stability analysis:** Compare the fitted correction factors across LOMO folds. If factors are stable (coefficient of variation < 0.3), the analytical decomposition has captured the right structure and corrections are architecture-independent. If factors vary widely, the decomposition is missing architecture-specific effects.
4. **Residual analysis:** For the worst-performing LOMO fold, decompose the prediction error into (a) analytical backbone error and (b) correction factor error. This identifies whether failures come from wrong physics or wrong calibration.

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** Leave-one-model-out (entire model held out) and leave-one-workload-out (entire workload held out)
**Baselines:** XGBoost LOMO from Idea 1 h3 (head-to-head comparison), Raw analytical decomposition without corrections (measures correction factor value), Per-model 36-parameter variant (measures generalization gap)
**Success metric:** Per-step MAPE < 20% on every held-out model with the 9-parameter global model

**MoE transfer test:** The Mixtral LOMO fold is particularly important -- it tests whether correction factors calibrated on 3 dense models transfer to an MoE architecture. The analytical decomposition explicitly handles MoE via `active_experts/num_experts` scaling, so the hypothesis is that the correction factors capture non-MoE-specific discrepancies (kernel launch overhead, memory controller effects, etc.).

**Structural advantage test:** If the 9-parameter analytical model matches or outperforms XGBoost's LOMO MAPE despite having orders of magnitude fewer parameters, this constitutes strong evidence that the analytical backbone provides structural generalization. Report the parameter efficiency ratio: `XGBoost_LOMO_MAPE / Analytical_LOMO_MAPE * (XGBoost_params / 9)`.

## Related Work

- **DistServe** (Zhong et al., OSDI 2024): Disaggregated inference serving that models prefill and decode separately. Their performance model uses per-phase analytical estimates, supporting the decomposition approach.
- **Splitwise** (Patel et al., ISCA 2024): Demonstrates that prefill and decode have different scaling behaviors across hardware, motivating separate correction factors per component.
- **Calotoiu et al.** (SC 2013): Shows that analytical + learned correction models generalize better than purely empirical models across HPC configurations -- direct precedent for our generalization hypothesis.

## Go Integration Path

If generalization succeeds, a single `stepml_coefficients.json` with 9 correction factors ships as the default, replacing the per-model alpha/beta coefficients in `defaults.yaml`. The key advantage over Idea 1 is that no serialized model file is needed -- just 9 floating-point constants. This means: (a) the Go binary size does not increase, (b) no model loading or deserialization at startup, (c) the coefficients are human-interpretable and auditable. If MoE transfer fails (>30% MAPE on Mixtral), ship 18 parameters: 9 for dense architectures + 9 for MoE, selected based on `config.json` architecture type.
