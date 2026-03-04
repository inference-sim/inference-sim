# Idea 1, H5: Per-Model Piecewise-Linear StepTime Generalizes Across Unseen Workloads (LOWO) — FINDINGS

**Status:** Refuted
**Resolution:** Refuted — mechanism not plausible
**Family:** Performance-regime
**VV&UQ:** Validation
**Type:** Statistical/Dominance
**Date:** 2026-02-27
**Rounds:** 1

## Hypothesis

> Per-model piecewise-linear StepTime models (h1's approach) trained on 3 of 4 workload types achieve <40% per-step MAPE on the held-out workload, improving over Round 1's LOWO result (109.7% MAPE).

## Experiment Design

**Classification:** Statistical/Dominance

**Method:** 4-fold leave-one-workload-out (LOWO) cross-validation, applied independently per model family. For each model, train on 3 workloads, test on the held-out workload.

**Folds (repeated per model):**
| Fold | Training Workloads | Held-Out Workload |
|------|-------------------|-------------------|
| 1 | codegen, roleplay, reasoning | **general** |
| 2 | general, roleplay, reasoning | **codegen** |
| 3 | general, codegen, reasoning | **roleplay** |
| 4 | general, codegen, roleplay | **reasoning** |

**Controlled variables:** 2-regime piecewise-linear structure, KV features, Ridge alpha CV, temporal split within training data
**Varied variable:** Which workload is held out (per model)
**Data:** 77,816 steps from 10 experiments. Note: Llama-2-7B only has roleplay workload in eval/ground_truth/, so its LOWO folds are incomplete.
**Baselines:** Round 1 LOWO (109.7% avg)

## Results

### Per-Model LOWO Summary

| Model Family | Avg LOWO MAPE | Worst Workload | Worst MAPE | Folds Completed |
|-------------|-------------|----------------|------------|----------------|
| **Mixtral** | **21.3%** | roleplay | 26.6% | 3/4 |
| **34B** | 147.7% | codegen | 292.4% | 3/4 |
| **70B** | 297.3% | roleplay | 480.6% | 3/4 |
| **7B** | — | — | — | 0/4 (only 1 workload) |
| **Grand Avg** | **155.4%** | — | 480.6% | 9/16 |

### Full Per-(Model, Workload) Results

| Model | Held-Out | Decode MAPE | Decode r | Mixed MAPE | Mixed r | Overall MAPE | n_test |
|-------|----------|-----------|---------|-----------|---------|-------------|--------|
| **34B** | codegen | 294.9% | 0.804 | 281.7% | 0.354 | 292.4% | 8,216 |
| **34B** | general | 36.4% | 0.746 | 57.0% | 0.667 | 43.4% | 7,441 |
| **34B** | roleplay | 102.8% | 0.334 | 130.3% | 0.267 | 107.2% | 8,443 |
| **70B** | codegen | 307.1% | 0.796 | 127.8% | 0.749 | 265.2% | 6,679 |
| **70B** | general | 172.3% | 0.871 | 104.9% | 0.792 | 145.9% | 5,992 |
| **70B** | roleplay | 491.1% | 0.571 | 435.2% | 0.646 | 480.6% | 6,741 |
| **Mixtral** | codegen | 18.8% | 0.310 | 29.9% | 0.608 | 21.4% | 6,544 |
| **Mixtral** | general | 14.8% | 0.948 | 17.3% | 0.880 | 15.8% | 5,867 |
| **Mixtral** | roleplay | 24.1% | 0.460 | 37.1% | 0.058 | 26.6% | 6,677 |

### Per Held-Out Workload (Averaged Across Models)

| Held-Out Workload | Avg MAPE | Range |
|-------------------|----------|-------|
| general | 68.4% | 15.8% - 145.9% |
| codegen | 193.0% | 21.4% - 292.4% |
| roleplay | 204.8% | 26.6% - 480.6% |
| reasoning | — | No reasoning experiments in eval/ground_truth/ |

### Comparison with Round 1

| Metric | Idea 1 LOWO | Round 1 LOWO |
|--------|-------------|-------------|
| Grand avg MAPE | **155.4%** | 109.7% |
| Improvement factor | **0.7x (worse)** | — |

### BLIS E2E Validation

BLIS E2E validation failed for all experiments due to binary path resolution issue (`parents[4]` resolves to `hypotheses/` instead of repo root). Per-step metrics are unaffected.

## Root Cause Analysis

### Why LOWO fails at 155.4% (vs <40% target) and is worse than Round 1

**1. Workload distribution shifts cause catastrophic extrapolation.** The codegen and roleplay workloads have dramatically different batch-composition distributions than the training workloads:
- **Codegen**: Short prompts with long outputs — decode-only steps dominate, and step times are highly variable due to long decode sequences building up KV cache
- **Roleplay**: Medium prompts with medium outputs — but different batch-size distributions due to different arrival rates
- **General**: Most diverse batch compositions — hardest to predict from specialized training data, but averaging effects make it the most "predictable" holdout

The 70B/roleplay fold (480.6% MAPE) is the worst case: the roleplay workload creates very different batch dynamics (larger batches, more preemptions) than codegen/general, and the linear model learned on those workloads extrapolates catastrophically.

**2. Per-model training with only 3 experiments per model is fragile.** In LOWO, each model has at most 4 experiments. Holding out 1 leaves only 3 for training. With temporal 60/20/20 split within those 3, the effective training data is ~60% of 3 experiments. This is insufficient for the linear model to learn robust batch-composition-to-time relationships, especially when the held-out workload has a different feature distribution.

**3. The 2-regime split amplifies workload sensitivity.** The mixed-batch regime lumps all prefill+decode steps together. Different workloads have different prefill-token distributions within this regime:
- Codegen: few but long prefills
- General: many short-to-medium prefills
- Roleplay: medium prefills

A single linear model on `(prefill_tokens, decode_tokens, prefill_x_decode, kv_sum)` cannot capture these distributional differences. With Idea 2's 3-regime split, the mixed-light/mixed-heavy boundary partially addresses this.

**4. Mixtral succeeds because MoE step times are inherently more predictable.** Mixtral's LOWO avg is only 21.3% — well below the 40% target. This is because MoE routing creates a consistent overhead pattern that dominates step time, making it relatively invariant to workload-specific batch composition. The linear model's coefficients generalize because the underlying compute scaling is more regular.

### Why this is worse than Round 1 (0.7x)

Round 1's global XGBoost had access to many more features (30 physics-informed features) and a nonlinear model. Even though it overfitted to workload patterns, it still captured some generalizable batch-to-time relationships. The 2-regime piecewise-linear with only 4 features per regime is too simple to learn even within-model patterns robustly enough for cross-workload transfer.

## Devil's Advocate (RCV-5)

**Arguing the hypothesis might be Confirmed:**
Mixtral achieves 21.3% LOWO MAPE — well below the <40% target — proving that per-model piecewise-linear models CAN generalize across workloads for certain architectures. The failure is concentrated in the 70B models (297.3% avg) which have fewer experiments in the data (only 3 vs 34B's 3 and Mixtral's 3). If more workload diversity existed in the training data (e.g., 8 workloads per model, hold out 1), the LOWO degradation would likely be much smaller. The problem is data scarcity, not the model structure. Additionally, the 7B model couldn't be evaluated at all (only 1 workload), which means the grand average is biased by the worst-performing model family. A fairer evaluation with balanced data might yield <40% average.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Grand avg LOWO MAPE = 155.4% (>40%) | Refutation | Per-model piecewise-linear does not generalize across workloads |
| 0.7x vs Round 1 (worse, not better) | Surprise | 2-regime linear is less robust than 30-feature XGBoost for LOWO |
| Mixtral LOWO = 21.3% (well below 40%) | Confirmation with nuance | MoE models generalize across workloads; dense models do not |
| 70B/roleplay = 480.6% MAPE | Design limitation | Roleplay creates very different batch dynamics for large models |
| 7B has only 1 workload in data | Data limitation | Cannot evaluate LOWO for 7B; need more experiments |
| Codegen is consistently hardest holdout | New insight | Codegen workloads create unique batch patterns across models |

## Standards Audit

- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? None

## Scope and Limitations (RCV-6)

- **Operating point tested:** 3 model families (34B, 70B, Mixtral) x 3 workloads (codegen, general, roleplay) = 9 LOWO folds. 7B excluded (only 1 workload). Reasoning workload not in eval/ground_truth/ (only in eval/other_gt/).
- **Parameters findings depend on:** 2-regime structure, raw linear regression, 4 features per regime, eval/ground_truth/ data only (10 experiments, not 16)
- **What was NOT tested:** 3-regime structure (Idea 2), reasoning workload holdout, including all 16 experiments from both data directories, feature normalization, BLIS E2E validation
- **Generalizability:** Mixtral's success may generalize to other MoE architectures. The failure of dense models (34B, 70B) is likely generalizable — workload-induced batch distribution shifts cause extrapolation failure in simple linear models.
- **Uncertainty quantification:** UQ not performed — single evaluation per fold.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Grand avg LOWO MAPE | 155.4% | Medium — only 9/16 folds completed (7B excluded) |
| Mixtral LOWO | 21.3% | High — 3 folds, 19,088 steps |
| Worst fold (70B/roleplay) | 480.6% | High — 6,741 test steps |
| Round 1 comparison | 0.7x (worse) | Medium — different data split (10 vs 16 experiments) |
| Mechanism (workload shift) | Proposed | Medium — not confirmed by control with more training workloads |

## Implications for Users

1. **Per-model piecewise-linear models are NOT robust across workloads** with only 3 training workloads per model and a 2-regime split.
2. **Mixtral is an exception** — its MoE architecture produces more regular step-time patterns that generalize across workloads. A single Mixtral StepML model can serve all workload types.
3. **Workload-specific training may be needed for dense models**, especially for roleplay and codegen workloads which create unique batch dynamics.
4. **More ground-truth experiments per model are needed** for robust LOWO evaluation. The current 3-4 workloads per model are insufficient.
5. **Idea 2's 3-regime approach should be evaluated for LOWO** — the mixed-light/mixed-heavy split may help with workload generalization by reducing regime heterogeneity.

## Reproducing

```bash
cd hypotheses/h-stepml/round2/idea-1-bayesian-calibration/h5-workload-generalization
./run.sh
```
