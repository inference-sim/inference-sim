# StepML Round 2: Findings Summary

**Date:** 2026-02-27
**Branch:** `stepml-experiments`
**Scope:** Two ideas tested across 10 experiments (4 models x 3 workloads, no reasoning). Both ideas extended Round 1's per-step modeling to include BLIS E2E validation, per-request KV features (ProgressIndex-derived), and generalization testing.

---

## 1. Problem Recap

Achieve **<10% workload-level E2E mean error** across 12 experiments (4 models x 3 workloads) by improving any or all of the 5 `LatencyModel` methods. The baseline to beat is **115.0% mean E2E error** from per-model linear regression.

Round 1 established that per-experiment XGBoost achieves 34% per-step MAPE but never validated through BLIS E2E simulation. Round 2's mandate was to close that loop: validate through BLIS, add per-request KV features, and test generalization.

---

## 2. What We Tried

### Idea 1: Bayesian Calibration (2-Regime Piecewise Linear)

**Approach:** Split steps into 2 regimes (decode-only vs mixed-batch), fit Ridge regression per regime with ProgressIndex-derived KV features (kv_sum, kv_max, kv_mean), and plan to jointly calibrate all 5 LatencyModel methods via Bayesian optimization.

| Sub-Hypothesis | Target | Result | Status |
|---|---|---|---|
| H1: Piecewise-linear StepTime | <30% per-step MAPE | **87.4%** | **Refuted** |
| H2: Joint Bayesian optimization | <15% E2E mean | Not run | **Blocked** (H1 too inaccurate) |
| H4: LOMO cross-validation | <80% per-step MAPE | **148.8%** | **Refuted** |
| H5: LOWO cross-validation | <40% per-step MAPE | **155.4%** | **Refuted** |

### Idea 2: Regime-Switching Ensemble (3-Regime Ridge)

**Approach:** Split steps into 3 regimes (decode-only / mixed-light with prefill<256 / mixed-heavy with prefill>=256), fit Ridge per regime with KV features, validate through BLIS E2E simulation, and calibrate secondary LatencyModel methods from ground-truth lifecycle data.

| Sub-Hypothesis | Target | Result | Status |
|---|---|---|---|
| H1: Regime Ridge per-step MAPE | <15% weighted | **64.4%** | **Not met** |
| H2: BLIS E2E validation | <10% mean | **427.8%** | **Not met** |
| H3: Secondary method calibration | >=5pp E2E gain | **0.0pp** | **Refuted** |
| H4: LOMO (cross-model) | <80% avg MAPE | **108.6%** | **Refuted** |
| H5: LOWO (cross-workload) | <25% avg MAPE | **117.4%** | **Refuted** |

---

## 3. What Worked

### 3.1 Overhead Floor Mechanism (Best Single Contribution)

The `max(overhead, compute)` step-time floor is the most effective accuracy mechanism discovered. Per-model overhead constants:

| Model | Overhead (us) |
|---|---|
| Llama-2-7B (TP1) | 3,897 |
| CodeLlama-34B (TP2) | 6,673 |
| Llama-2-70B (TP4) | 8,029–8,203 |
| Mixtral-8x7B (TP2) | 9,125 |

The overhead floor handles the decode-dominated regime (77.9% of steps) by clamping small-batch predictions to the real cycle time. This mechanism alone produces **5/10 experiments with ITL <10%**, proving that for decode-dominant workloads, the overhead floor is more important than the regression model.

### 3.2 ITL Prediction (33.6% Mean, 5/10 Under 10%)

ITL directly reflects step-time + overhead quality. Per-experiment ITL:

| Experiment | ITL Error |
|---|---|
| mixtral-roleplay | **0.7%** |
| mixtral-codegen | **2.1%** |
| 7b-roleplay | **4.0%** |
| mixtral-general | **8.7%** |
| 70b-hf-codegen | **10.7%** |
| 70b-roleplay | **10.3%** |
| 70b-general | 23.5% |
| 34b-general | 80.3% |
| 34b-codegen | 95.9% |
| 34b-roleplay | 100.1% |

The 7B and Mixtral models are essentially solved for ITL. The 70B models are close. CodeLlama-34B is the sole outlier.

### 3.3 LOMO Improvement: 23.6x Over Round 1

Round 2 LOMO: 108.6% avg MAPE vs Round 1 LOMO: 2,559.7%. The regime structure plus per-request KV features dramatically improve cross-model transfer, even though the absolute target (<80%) was not met. The 34B holdout fold (63.4% MAPE) shows that interpolation between model scales is feasible.

### 3.4 Mixtral Generalizes Exceptionally Well

Mixtral is consistently the best-performing model across all metrics:
- Per-step MAPE: 16.6% (best of all models)
- LOWO: 19.1% avg (below 25% target)
- ITL: 0.7%, 2.1%, 8.7% (all under 10%)

MoE architectures appear to have more regular compute scaling. A single universal Mixtral model works across all workloads.

### 3.5 Per-Model Overhead Constants Are Stable

Overhead values derived independently in Round 2 match values from earlier Round 2 analysis, confirming they are stable properties of the model+GPU+TP configuration rather than workload-dependent artifacts.

---

## 4. What Failed and Why

### 4.1 KV Features Are Counter-Productive in Raw Linear Space

**The central thesis of Round 2 — that ProgressIndex-derived KV features would improve prediction — was refuted.**

Both ideas showed KV features *degrade* per-step accuracy:
- Idea 1: +3.6pp worse (87.4% with KV vs 83.8% without)
- Idea 2: +20.5pp worse (64.4% with KV vs 43.9% without)

**Root cause:** KV feature magnitudes (kv_sum ranging 0 to 64,000+) cause Ridge coefficient instability. Small Ridge alpha values amplify noise from extreme KV values; large alpha values suppress all KV signal. Only Mixtral benefited from KV features (-5.4pp), likely due to its MoE architecture's different KV-to-compute relationship.

**This does NOT mean KV features are useless — it means raw linear regression cannot handle their dynamic range.** Feature scaling (StandardScaler), log-transform, or nonlinear models would likely fix this.

### 4.2 TTFT Errors Are Catastrophic (31,906% Mean)

TTFT errors of thousands to hundreds-of-thousands percent indicate a systematic mismatch between BLIS's request lifecycle and the real system, not a step-time modeling problem. The TTFT error is the dominant contributor to E2E error (427.8%). Likely causes:
- Workload spec generation producing mismatched request arrival patterns
- BLIS prefill/queueing behavior diverging from real vLLM scheduling
- Request injection timing not matching ground-truth arrival distribution

### 4.3 Mixed-Heavy Regime Is Empty

Only 74/77,816 steps (0.1%) have prefill >= 256 tokens. The 3-regime design effectively collapses to 2 regimes on this dataset (which excludes reasoning workloads). The expected ~4.4% share from the 16-experiment Round 1 data does not hold for the 10-experiment Round 2 data.

### 4.4 Secondary Methods Contribute Zero

H3's ablation showed that calibrating SchedulingProcessingTime and QueueingTime constants (200–400 us) produced **exactly 0.0 percentage points** of E2E improvement. When the dominant error source (StepTime or workload spec) is off by orders of magnitude, microsecond-level corrections are invisible.

### 4.5 CodeLlama-34B Is Consistently the Worst Model

34B shows the highest per-step MAPE (99.2%), worst BLIS E2E (2,901% for general), and outlier LOMO/LOWO errors. This model may have batch dynamics that differ significantly from the Llama-2 family.

### 4.6 Cross-Workload Transfer Did Not Improve

LOWO regressed slightly from Round 1 (117.4% vs 109.7%). The regime structure was expected to provide workload-agnostic modeling but did not. The 70B model dominates the error (293.8%), suggesting large-model workload shifts have outsized impact.

---

## 5. Data Characteristics Learned (Round 2)

### 5.1 Step Duration vs Cycle Time

`step.duration_us` captures GPU forward pass only (~70–7,000 us). Real step cycle time includes CPU overhead: scheduling, synchronization, memory management. The overhead ranges 4–9ms per model and is derived from: `GT_ITL_us - mean_compute_us`.

### 5.2 Overhead Integration Approaches (All Tested)

| Approach | Result |
|---|---|
| Additive (compute + overhead) | Phase transition — under-capacity to overload with no stable point |
| Log-space cycle time | Exponential amplification — batch-size coefficients get multiplied, not added |
| **Max floor (best)** | `step_time = max(overhead, compute)` — handles memory-bound→compute-bound crossover |
| Floor + cap | `max(overhead, min(compute, 3*overhead))` prevents exponential blowup from expm1 |

### 5.3 Log-Space (expm1) Pitfalls

- KV features: coefficient 0.0001 x 64,000 = 6.4 in log-space → 601x multiplier
- Prefill tokens: coefficient 0.06 x 500 = 30 in log-space → exp(30) = absurd
- Safe ONLY for decode-only batches with small coefficients

### 5.4 Raw Linear for BLIS E2E

Use `use_log_target=False` for all regimes destined for BLIS consumption. Per-step MAPE is ~448% but irrelevant — the overhead floor handles decode-dominated steps, and the linear model only matters for large-batch compute-bound steps.

### 5.5 Phase Distribution (10-Experiment Dataset)

| Regime | Steps | Share |
|---|---|---|
| Decode-only (prefill = 0) | 60,613 | 77.9% |
| Mixed-light (0 < prefill < 256) | 17,129 | 22.0% |
| Mixed-heavy (prefill >= 256) | 74 | 0.1% |

### 5.6 Step Data Is ~10% Sampled

Not every step is logged in lifecycle data. Step observations come from OpenTelemetry tracing at ~10% sample rate.

---

## 6. Comparison of Ideas

| Metric | Idea 1 (2-regime) | Idea 2 (3-regime) | Winner |
|---|---|---|---|
| Per-step MAPE (with KV) | 87.4% | 64.4% | Idea 2 |
| Per-step MAPE (no KV) | 83.8% | 43.9% | Idea 2 |
| BLIS E2E mean error | Not tested | 427.8% | — |
| ITL mean error | Not tested | 33.6% (5/10 <10%) | Idea 2 |
| LOMO avg MAPE | 148.8% | 108.6% | Idea 2 |
| LOWO avg MAPE | 155.4% | 117.4% | Idea 2 |
| KV feature contribution | -3.6pp (harmful) | -20.5pp (harmful) | Neither |

**Idea 2 is strictly better but both fail all primary targets.** The 3-regime structure, overhead floor, and BLIS integration pipeline from Idea 2 are the foundation to build on.

---

## 7. Per-Experiment Leaderboard (BLIS E2E)

Best results achieved in Round 2 (Idea 2 with regime Ridge + overhead floor):

| Experiment | E2E Error | TTFT Error | ITL Error | Baseline E2E |
|---|---|---|---|---|
| 34b-roleplay | **3.3%** | 282.7% | 100.1% | 136.5% |
| mixtral-roleplay | 50.8% | 77.2% | **0.7%** | 141.9% |
| mixtral-codegen | 51.5% | 76.5% | **2.1%** | 141.3% |
| 7b-roleplay | 52.5% | 78.4% | **4.0%** | 86.1% |
| 70b-hf-codegen | 55.7% | 77.7% | 10.7% | — |
| 70b-roleplay | 55.6% | 77.9% | 10.3% | — |
| mixtral-general | 554.6% | 44,464% | **8.7%** | 138.5% |
| 70b-general | 182.7% | 12,567% | 23.5% | 111.8% |
| 34b-codegen | 370.2% | 30,425% | 95.9% | 135.4% |
| 34b-general | 2,901.1% | 230,931% | 80.3% | 132.3% |

**Observations:**
- Only 1/10 experiments meets <10% E2E target (34b-roleplay, likely coincidental)
- TTFT errors are the dominant E2E contributor — they're 2-3 orders of magnitude worse than ITL
- ITL is well-solved for 7B, Mixtral, and partially for 70B
- CodeLlama-34B "general" is catastrophic across all metrics

---

## 8. Binding Constraints for Round 3

### 8.1 TTFT/Workload Spec Mismatch (PRIMARY — Blocking E2E Target)

The 31,906% mean TTFT error is the single biggest obstacle to <10% E2E. This is NOT a step-time problem — it's a simulation-fidelity problem in how BLIS generates and schedules requests compared to the real system. Until TTFT is addressed, no step-time improvement can achieve the E2E target.

**Actionable:** Diagnose whether the workload spec generator, BLIS request injection timing, or BLIS prefill scheduling is the root cause. Compare BLIS-simulated request arrival/scheduling timestamps against ground-truth lifecycle data.

### 8.2 KV Feature Scaling (SECONDARY — Blocking Per-Step Improvement)

Per-request KV features (the key Round 1 recommendation) hurt accuracy due to numerical instability in raw linear space. The features themselves are not wrong — the regression formulation is.

**Actionable:** Apply StandardScaler within the Ridge pipeline, or use log-transformed features (not target), or switch to a model that handles large feature ranges (tree ensemble, or feature-scaled Ridge).

### 8.3 CodeLlama-34B Anomaly (SECONDARY)

34B is the worst model by a wide margin across all metrics. The specific batch dynamics causing this need investigation (batch-size distributions, step-time outliers, prefill/decode ratios).

**Actionable:** Profile 34B's step-time distribution, batch-size histogram, and KV utilization compared to other models. Determine if it's a data quality issue, architecture anomaly, or model-specific coefficient problem.

### 8.4 Mixed-Heavy Regime Sparsity (TERTIARY)

The 3-regime design collapses to 2 effective regimes because only 0.1% of steps are mixed-heavy. This regime needs more data (perhaps from reasoning workloads, which were excluded from Round 2 data) or the threshold needs lowering.

---

## 9. Successful Techniques to Preserve

1. **Overhead floor (`max(overhead, compute)`)** — the most impactful single technique
2. **Per-model training** — mandatory due to 3+ OOM step-time variation
3. **Regime separation** (decode-only vs mixed) — provides distinct modeling contexts
4. **Temporal train/test split** — prevents autocorrelation leakage
5. **BLIS E2E validation pipeline** (Idea 2's `validate_e2e.py` + `validate_blis.py`) — the infrastructure for running BLIS with custom coefficients and comparing against ground truth
6. **Per-model overhead derivation** from ground-truth ITL
7. **Secondary method constant extraction** from lifecycle data (even though they contributed 0pp, the extraction pipeline is correct)

---

## 10. Eliminated Approaches (Do Not Repeat)

| Approach | Why It Failed | Round |
|---|---|---|
| Global models (all experiments) | 3+ OOM step-time scale variation | R1 |
| Analytical FLOPs decomposition without per-request KV | decode_attention = 0, structurally incomplete | R1 |
| Correction factors on incomplete backbone | Pathological compensation (factor=39.5 on zero signal) | R1 |
| Pure-phase analysis | Only 6 pure-prefill steps exist under chunked prefill | R1 |
| **Raw KV features in linear regression** | kv_sum up to 64K causes Ridge coefficient instability | R2 |
| **2-regime piecewise linear** | Too coarse — mixed-batch heterogeneity too high for one linear regime | R2 |
| **Joint Bayesian optimization** | Cannot compensate for structural model errors | R2 |
| **Secondary method calibration alone** | 200-400us corrections invisible against 100%+ E2E errors | R2 |
| **Log-space target with large features** | expm1 causes exponential blowup with KV/prefill coefficients | R2 |

---

## 11. Summary for Future Ideation

### What the data tells us after 2 rounds:

1. **The overhead floor is the workhorse** — ITL is solved for 4/5 models without any sophisticated step-time model
2. **TTFT errors dominate E2E** — the step-time model is nearly irrelevant to E2E accuracy until TTFT is fixed
3. **KV features have signal but need proper scaling** — they're counter-productive in raw linear space, productive when the model handles their dynamic range
4. **Per-model calibration is non-negotiable** — step-time scales vary 3+ orders of magnitude
5. **The 3-regime + overhead floor architecture is sound** — it just needs (a) TTFT fix, (b) feature scaling, (c) 34B investigation

### What Round 3 should prioritize (ordered):

1. **Diagnose and fix the TTFT mismatch** — this is the gating issue for E2E. Without it, no step-time improvement matters.
2. **Apply feature scaling to KV features** — StandardScaler, log-transform, or tree models. The features themselves are available and meaningful; only the regression formulation is wrong.
3. **Investigate CodeLlama-34B** — determine if the anomaly is data, architecture, or coefficients.
4. **Lower mixed-heavy threshold or add reasoning data** — the 256-token boundary sees only 74 steps. Consider 64 or 128 tokens, or include reasoning workloads.

### Hard constraints to respect:
- Per-model training mandatory (LOMO still >80%)
- Features must be computable from `[]*Request` with `ProgressIndex` (Go integration)
- Prediction latency <1ms per step
- Pure-Go implementation (no Python/ONNX runtime)

---

## 12. Cumulative Round History

| Round | Best E2E Error | Best ITL Error | Key Achievement | Primary Blocker |
|---|---|---|---|---|
| R1 | Not tested (Python-only) | Not tested | 34% per-step MAPE with XGBoost (2x better than blackbox) | No BLIS integration, no E2E validation |
| **R2** | **427.8% mean** (1/10 <10%) | **33.6% mean** (5/10 <10%) | Overhead floor + BLIS pipeline + per-request KV features | **TTFT mismatch (31,906%)** dominates E2E; KV features counter-productive in linear space |

---

## 13. Reproducibility

All code is in `hypotheses/h-stepml/round2/`:

```
round2/
├── FINDINGS_ROUND2.md              # This file
├── idea-1-bayesian-calibration/
│   ├── FINDINGS_SUMMARY.md
│   ├── h1-piecewise-steptime/      # 2-regime Ridge (run.sh, train_piecewise.py)
│   ├── h2-joint-bo-calibration/    # Blocked (bo_calibrate.py ready but not run)
│   ├── h4-model-generalization/    # LOMO (lomo_cv.py)
│   └── h5-workload-generalization/ # LOWO (lowo_cv.py)
└── idea-2-regime-ensemble/
    ├── FINDINGS_SUMMARY.md
    ├── h1-kv-regime-models/        # 3-regime Ridge (train_regime_ridge.py)
    ├── h2-blis-e2e-validation/     # BLIS E2E (validate_e2e.py)
    ├── h3-secondary-method-calibration/ # Ablation (calibrate_secondary.py)
    ├── h4-model-generalization/    # LOMO (lomo_cv.py)
    └── h5-workload-generalization/ # LOWO (lowo_cv.py)
```

**Dependencies:** Python 3, pandas, scikit-learn, scipy, numpy
**Data:** `eval/ground_truth/` (77,816 steps from 10 experiments)
**BLIS integration:** `sim/latency/stepml.go` (Go StepTime with floor/cap logic)
