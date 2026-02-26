# StepML Round 1: Findings Summary

**Date:** 2026-02-26
**Branch:** `stepml-experiments`
**Purpose:** Comprehensive summary of all Round 1 experiments to inform future rounds of ideation and testing.

---

## 1. Problem Recap

Replace BLIS's 3-coefficient blackbox latency model (`step_time = beta0 + beta1*prefill_tokens + beta2*decode_tokens`) with a data-driven predictor achieving **<10% workload-level E2E mean error** across 16 experiments (4 models × 4 workloads, H100 GPUs). The winning model must integrate into BLIS as a Go `LatencyModel` implementation.

**Data:** 122,752 step-level observations from instrumented vLLM v0.15.1 with OpenTelemetry tracing (10% sample rate).

---

## 2. What We Tried

### Idea 1: Physics-Informed Tree Ensemble

**Approach:** Engineer 30 physics-informed features from batch composition and architecture parameters, train XGBoost per-experiment.

| Sub-Hypothesis | Result | Key Number |
|---|---|---|
| h1: 30 features + Ridge | Weakly Supported (per-exp) | 92.1% avg MAPE per-exp; global 301.7% short-circuited |
| h2: XGBoost per-experiment | **Weakly Supported** | **34.0% avg MAPE** (2x better than blackbox 70.4%) |
| h3: Cross-config generalization | Not Supported | LOMO 2559.7%, LOWO 109.7% |

### Idea 2: Analytical Decomposition + Learned Corrections

**Approach:** Decompose step time into 4 analytical FLOPs components (prefill GEMM, prefill attention, decode GEMM, decode attention), then learn multiplicative correction factors.

| Sub-Hypothesis | Result | Key Number |
|---|---|---|
| h1: Component correlation | Not Supported | 0/4 components > 0.8 Pearson r |
| h2: Correction factors | Not Supported | 78.7% avg MAPE (worse than blackbox 70.4%) |

### Idea 3: Evolutionary Synthesis (OpenEvolve)

**Status:** Deferred — h2 depends on Idea 2's analytical components (which are structurally incomplete), and OpenEvolve infrastructure was not set up.

---

## 3. What Worked

### 3.1 Per-Experiment XGBoost (Best Result)

XGBoost with 30 physics-informed features, trained separately per experiment:
- **Average MAPE: 34.0%** (vs blackbox 70.4%) — **2x improvement**
- **5/16 experiments < 15% MAPE**, 9/16 < 30%
- **Zero regressions** — beats or matches blackbox on all 16 experiments
- Biggest wins: reasoning workloads (Llama-7B-reasoning: 123.5% → 13.0%, Mixtral-reasoning: 222.8% → 62.4%)

**Full per-experiment leaderboard (XGBoost vs Blackbox):**

| Model | Workload | XGB MAPE | BB MAPE | Improvement |
|-------|----------|----------|---------|-------------|
| mixtral-8x7b-v0-1 | general | 9.1% | 9.2% | +0.1 pp |
| llama-2-70b-hf | reasoning | 9.8% | 14.2% | +4.4 pp |
| llama-2-7b | reasoning | 13.0% | 123.5% | +110.5 pp |
| codellama-34b | reasoning | 14.0% | 37.3% | +23.3 pp |
| mixtral-8x7b-v0-1 | codegen | 14.7% | 19.0% | +4.3 pp |
| codellama-34b | codegen | 16.1% | 21.6% | +5.5 pp |
| codellama-34b | roleplay | 25.7% | 30.8% | +5.1 pp |
| mixtral-8x7b-v0-1 | roleplay | 28.3% | 33.6% | +5.3 pp |
| llama-2-70b | roleplay | 28.3% | 128.6% | +100.2 pp |
| llama-2-7b | roleplay | 30.9% | 40.3% | +9.4 pp |
| llama-2-70b-hf | codegen | 31.1% | 90.8% | +59.7 pp |
| llama-2-7b | codegen | 33.4% | 69.7% | +36.3 pp |
| llama-2-7b | general | 50.9% | 72.9% | +22.0 pp |
| llama-2-70b | general | 55.4% | 61.2% | +5.7 pp |
| mixtral-8x7b-v0-1 | reasoning | 62.4% | 222.8% | +160.5 pp |
| codellama-34b | general | 121.5% | 151.1% | +29.6 pp |

### 3.2 Feature Engineering Insights

The 30 features in 6 groups provided clear value over the 2-feature blackbox. Most informative features by XGBoost importance:

| Rank | Feature | Description | Why It Helps |
|------|---------|-------------|-------------|
| 1 | `f_kv_blocks_used` | GPU KV cache blocks occupied | Best available proxy for total context length across all running requests |
| 2 | `f_decode_tokens` | Decode tokens in batch | Core batch computation feature (already in blackbox) |
| 3 | `f_running_depth` | Number of running requests | Proxy for batch complexity and attention fan-out |
| 4 | `f_prefill_tokens` | Prefill tokens in batch | Core batch computation feature (already in blackbox) |
| 5 | `f_num_decode_reqs` | Decode request count | Captures batch shape vs just token count |
| 6 | `f_total_flops_estimate` | Analytical FLOPs estimate | Physics-informed feature — strong for Mixtral |
| 7 | `f_prefill_x_decode` | Interaction: prefill × decode | Captures non-additive mixed-batch overhead |
| 8 | `f_decode_memory_intensity` | Memory bytes / decode FLOPs | Compute vs memory bound indicator |
| 9 | `f_batch_size_x_kv_pressure` | Interaction: batch_size × kv_max | Attention memory pressure |
| 10 | `f_kv_blocks_free` | Free GPU KV cache blocks | Inverse proxy for system load |

**Key observation:** The top features are dominated by **KV cache state proxies** (`kv_blocks_used`, `running_depth`, `kv_blocks_free`) rather than physics features. The model is learning to predict step time from system state because per-request KV information is missing.

### 3.3 Analytical Decomposition Has Signal Where GEMM Dominates

Per-experiment correlation of total analytical FLOPs estimate with step duration:

| Correlation Range | Workload Type | Example |
|---|---|---|
| r > 0.85 | General (GEMM-dominated) | Mixtral-general r=0.94, Llama-7B-general r=0.86 |
| r = 0.35–0.65 | Codegen (mixed) | CodeLlama-codegen r=0.43 |
| r < 0.25 | Reasoning/roleplay (attention-dominated) | Mixtral-reasoning r=0.02, Llama-7B-reasoning r=0.12 |

The physics is correct where GEMM dominates — but attention cost (which depends on per-request KV lengths) is the dominant component for reasoning/roleplay workloads.

---

## 4. What Failed and Why

### 4.1 Global Models (Catastrophic Failure)

Any model trained across all 16 experiments fails catastrophically:
- **Global Ridge 30f:** 301.7% MAPE
- **Global blackbox:** 670% MAPE

**Root cause:** Step times span 3+ orders of magnitude across model configurations:
- Llama-2-7B (tp=1): 12–500 μs per step
- Llama-2-70B (tp=4): 500–250,000 μs per step

A single model cannot fit both scales. Per-experiment (or per-model) training is mandatory.

### 4.2 Cross-Model Generalization (Structural Failure)

LOMO (Leave-One-Model-Out) XGBoost:

| Holdout Model | Test MAPE | Why |
|---|---|---|
| llama-2-7b | **9908%** | tp=1 model with μs-scale steps; trained on tp=2/4 models with ms-scale steps |
| mixtral-8x7b-v0-1 | 186.6% | MoE architecture fundamentally different from dense training models |
| codellama-34b | 78.9% | Less extreme but still 5x worse than per-experiment |
| llama-2-70b | 65.5% | Closest to training distribution (similar scale) |

**Root cause:** Features encode absolute values (FLOPs, KV blocks) that scale with model size. Architecture metadata (`model_id`, `tp_degree`, `is_moe`) is insufficient to bridge the scale gap. This is **not a model capacity issue** — it's a representation issue.

### 4.3 Cross-Workload Generalization (Moderate Failure)

LOWO (Leave-One-Workload-Out) XGBoost:

| Holdout Workload | Test MAPE | Why |
|---|---|---|
| roleplay | 30.6% | Similar batch patterns to codegen — some transfer |
| codegen | 53.4% | Moderate transfer from other workloads |
| reasoning | 53.2% | Long contexts not well represented in other workloads |
| general | **301.6%** | Most diverse batch compositions — unpredictable from other workloads |

**Root cause:** Less severe than LOMO because same model config → same step time scale. Failures concentrate in "general" workloads (most diverse batch compositions) and "reasoning" (long contexts with extreme KV lengths).

### 4.4 Analytical Decomposition (Structural Incompleteness)

The 4-component FLOPs decomposition fails because:

1. **decode_attention_flops = 0** for all steps. Per-request KV cache lengths (`ProgressIndex`) are unavailable in step-level data. Decode attention is the dominant cost for 80.6% of steps (pure decode), making the decomposition structurally incomplete.

2. **Only 6 pure-prefill steps** exist (out of 122,752). vLLM's chunked prefill makes nearly every batch mixed. The experimental design assumed large pure-phase subsets for correlation analysis — this assumption was wrong.

3. **Correction factors compensate pathologically:** The optimizer set `decode_attn_factor = 39.5` trying to amplify a zero signal. The 36-parameter model (78.7% MAPE) actually performed **worse than the 3-parameter blackbox** (70.4%) because the analytical backbone with missing attention is less informative than raw `decode_tokens`.

### 4.5 "General" Workloads Are Consistently Hardest

Across all models and approaches, the 4 "general" workload experiments are the worst:

| Model | General MAPE (XGB) | Next-Worst MAPE (XGB) |
|-------|--------------------|-----------------------|
| codellama-34b | **121.5%** | 25.7% (roleplay) |
| llama-2-70b | **55.4%** | 28.3% (roleplay) |
| llama-2-7b | **50.9%** | 30.9% (roleplay) |
| mixtral-8x7b-v0-1 | 9.1% (exception — Mixtral-general is easy) | 28.3% (roleplay) |

**Root cause:** "General" workloads have the most diverse batch compositions — mixing short decode-only batches with occasional large prefill+decode batches. High step time variance within a single experiment makes prediction difficult even with 30 features.

---

## 5. Binding Constraints (What Blocks Further Progress)

### 5.1 Missing Per-Request KV Cache Lengths (PRIMARY)

The single most impactful limitation. The step-level training data contains only aggregate batch features (`batch.prefill_tokens`, `batch.decode_tokens`), not per-request KV cache lengths.

**Impact on each approach:**
- **XGBoost:** Uses system-state proxies (`kv_blocks_used`, `running_depth`) instead of per-request statistics (`kv_mean`, `kv_max`, `kv_sum`). These proxies capture occupancy but not per-request distribution. Two batches with identical total KV blocks but different per-request distributions (many short vs few long) have very different attention costs.
- **Analytical decomposition:** Decode attention FLOPs are literally uncomputable. This makes the decomposition incomplete for the dominant cost component.
- **Feature importance confirms it:** XGBoost's top features are all KV state proxies — the model is begging for better KV information.

**Available in simulator:** BLIS's `LatencyModel.StepTime()` receives `[]*Request`, each with `ProgressIndex` (cumulative input_processed + output_generated). This is the per-request KV cache length proxy. It's available at prediction time but was NOT available in the training data.

**Possible solutions for Round 2:**
1. Re-instrument vLLM to export per-request KV lengths in step-level traces
2. Derive approximate per-request KV lengths from per-request lifecycle data (already available in `request_metrics/`)
3. Design features that work without per-request KV (the approach we took, but with a ceiling)

### 5.2 Step Time Scale Variation (SECONDARY)

3+ orders of magnitude across model configurations. Forces per-model training. Not addressable without normalized/relative features.

**Possible solutions for Round 2:**
1. **Log-scale prediction:** Predict `log(step_time)` instead of `step_time` — compresses the range
2. **Normalized features:** Divide FLOPs by model-specific capacity (peak_flops × tp_degree) to produce utilization features that are scale-independent
3. **Relative prediction:** Predict `step_time / baseline_step_time` where baseline is a reference batch configuration
4. **Hierarchical models:** Model-specific first stage (captures scale) + shared second stage (captures batch composition effects)

### 5.3 Hyperparameter Underfitting (TERTIARY)

XGBoost consistently selected shallow trees (max_depth=4, n_estimators=100) for 13/16 experiments. This suggests the model is **underfitting, not overfitting** — more expressive features would help more than deeper trees. The feature set, not the model capacity, is the bottleneck.

---

## 6. Data Characteristics Learned

### 6.1 Phase Distribution

| Phase | Steps | % of Total |
|-------|-------|-----------|
| Pure decode (prefill_tokens = 0) | 98,961 | 80.6% |
| Mixed batch (prefill + decode) | 23,789 | 19.4% |
| Pure prefill (decode_tokens = 0) | 6 | 0.005% |

**Implication:** vLLM's chunked prefill makes nearly every batch mixed or pure-decode. Experimental designs assuming large pure-prefill subsets (like Idea 2) are invalid. Any step-time model must primarily handle decode-dominated and mixed batches.

### 6.2 Step Time Distribution by Workload Type

| Workload | Mean Duration | Median Duration | Std Dev | Coefficient of Variation |
|----------|--------------|-----------------|---------|------------------------|
| reasoning | 6,000–33,500 μs | High variance | Very high | >1.0 (bimodal) |
| general | 200–2,000 μs | Moderate | High | >0.8 |
| codegen | 160–320 μs | Low | Low-moderate | 0.3–0.6 |
| roleplay | 160–320 μs | Low | Low-moderate | 0.3–0.6 |

**Implication:** Reasoning workloads have extremely high variance and bimodal distributions (long prefills interspersed with short decodes). This drives high MAPE even for models that capture the mean well.

### 6.3 Feature Correlation Structure

- `batch.prefill_tokens` and `batch.decode_tokens` are weakly correlated (r ≈ 0.1–0.3) — they provide mostly independent information
- `kv_blocks_used` and `running_depth` are moderately correlated (r ≈ 0.4–0.7) — both capture system load
- Physics features (`total_flops_estimate`, `arithmetic_intensity`) are highly correlated with `prefill_tokens` on prefill-heavy workloads but nearly orthogonal on decode-heavy workloads
- Interaction terms (`prefill_x_decode`, `batch_size_x_kv_pressure`) add unique variance that individual features don't capture

### 6.4 Experiments Per Model

| Model | TP | Steps | Avg Step Time | Experiments |
|-------|-----|-------|---------------|-------------|
| llama-2-7b | 1 | 47,034 | ~160–5,000 μs | 4 |
| llama-2-70b / 70b-hf | 4 | 23,860 | ~500–33,500 μs | 4 |
| codellama-34b | 2 | 29,094 | ~200–6,000 μs | 4 |
| mixtral-8x7b-v0-1 | 2 | 22,764 | ~160–8,000 μs | 4 |

**Implication:** Llama-7B has the most steps (47K) — heavily represented. Data is not balanced across models.

---

## 7. Successful Techniques and Patterns

### 7.1 Temporal Train/Test Split

The 60/20/20 temporal split (earlier steps for training, later for testing) prevents autocorrelation leakage. This is critical because consecutive steps share system state (KV cache occupancy, queue depth) which could inflate accuracy artificially.

### 7.2 Per-Experiment Training

The only viable approach — step time scale varies 3+ orders of magnitude across model configurations. All production inference simulators (Vidur, Splitwise) use per-model calibration.

### 7.3 System-State KV Proxies

While imperfect, `kv_blocks_used`, `kv_blocks_free`, and `running_depth` provide meaningful signal about KV cache pressure. They correlate with step time because they capture total context length indirectly. However, they cannot distinguish per-request KV distribution.

### 7.4 Physics-Informed Interaction Terms

`prefill_x_decode` (prefill_tokens × decode_tokens) and `batch_size_x_kv_pressure` (batch_size × kv_max) capture non-additive mixed-batch effects that individual features miss. These were among the top 10 features for several experiments.

---

## 8. Failed Techniques and Anti-Patterns

### 8.1 Global Models

Any model trained across all experiments fails catastrophically due to step time scale variation. Do not attempt global training without scale normalization.

### 8.2 Analytical Decomposition Without Per-Request KV

The 4-component FLOPs decomposition is physically correct but computationally incomplete without per-request KV lengths. The `decode_attention = 0` problem makes the decomposition worse than the simple blackbox.

### 8.3 Correction Factors on Incomplete Backbone

Learned multiplicative corrections cannot fix a structurally incomplete model. The optimizer compensates pathologically (factor=39.5 on zero signal). If the backbone is missing a component, add the component — don't amplify the existing ones.

### 8.4 Pure-Phase Analysis with Chunked Prefill

vLLM's chunked prefill produces only 6 pure-prefill steps. Experimental designs relying on pure-phase subsets are invalid under continuous batching with chunked prefill.

---

## 9. Open Questions for Round 2

### 9.1 Can Per-Request KV Features Close the Gap?

The `ProgressIndex` field is available per-request in the BLIS simulator. If we can either:
- (a) Derive per-request KV lengths from the per-request lifecycle data in `eval/ground_truth/*/request_metrics/`, or
- (b) Extend the training data to include per-request KV lengths

...then both XGBoost and the analytical decomposition should improve substantially. The feature importance analysis shows the model is bottlenecked by KV information quality.

### 9.2 Can Log-Scale or Normalized Prediction Enable Cross-Model Models?

Per-model training works but requires N separate calibration runs. If features were normalized to model-specific capacity (e.g., FLOPs / peak_throughput, step_time / reference_step_time), a single model might generalize across configurations.

### 9.3 Is Per-Step MAPE the Right Diagnostic?

The primary metric is workload-level E2E mean error (not per-step MAPE). Per-step errors may cancel in aggregate. The 34% per-step MAPE might translate to <10% E2E mean error if errors are symmetric. This was NOT tested in Round 1. Round 2 should include BLIS E2E validation.

### 9.4 Would More Training Data Help?

Current per-experiment training sizes are 2,000–9,000 steps. The consistently shallow XGBoost trees suggest underfitting due to insufficient features, not insufficient data. But more diverse batch compositions (especially for "general" workloads) might help.

### 9.5 Can We Leverage the Per-Request Lifecycle Data?

The `eval/ground_truth/*/request_metrics/` directories contain per-token timestamps, input/output token counts, and E2E latencies. This data could:
- Derive approximate per-request KV lengths (ProgressIndex at each step)
- Enable workload-level E2E validation (compare predicted vs observed per-request E2E)
- Provide TTFT/ITL breakdown for Priority 2 metrics

### 9.6 What About Simpler Models with Better Features?

XGBoost with 30 features beats Ridge with 30 features by 2.7x (34% vs 92% avg MAPE). But the 30 features include several that are imprecise (KV proxies) or constant (attention ratios set to 0/1). A smaller set of high-quality features might enable simpler models (e.g., Ridge or piecewise linear) that are easier to integrate in Go.

### 9.7 Can We Handle "General" Workloads Better?

The 4 "general" experiments are consistently the hardest (50–121% MAPE for XGBoost). These have the most diverse batch compositions. Potential approaches:
- Cluster steps by batch composition type, train separate models per cluster
- Add batch composition entropy/diversity features
- Use quantile regression to handle bimodal step time distributions

### 9.8 Idea 3 Feasibility Without Idea 2

Evolutionary synthesis (OpenEvolve) was deferred because h2 depends on Idea 2's components. But h1 (evolving raw feature combinations) has no dependency on Idea 2 and could still discover non-obvious feature interactions. Worth revisiting if OpenEvolve infrastructure becomes available.

---

## 10. Reproducibility

All experiment code is in `hypotheses/h-stepml/`:

```
hypotheses/h-stepml/
├── shared/                      # Infrastructure (data_loader, splits, evaluation)
├── idea-1-tree-ensemble/
│   ├── h1-features/             # 30-feature Ridge regression
│   │   ├── run.sh               # Orchestrator
│   │   ├── engineer_features.py # Feature engineering (30 features, 6 groups)
│   │   ├── train_ridge.py       # Ridge training + prediction
│   │   ├── analyze.py           # Evaluation against baselines
│   │   └── FINDINGS.md          # Results and analysis
│   ├── h2-model/                # XGBoost per-experiment
│   │   ├── run.sh
│   │   ├── train_xgboost.py     # XGBoost grid search + training
│   │   ├── analyze.py
│   │   └── FINDINGS.md
│   └── h3-generalization/       # LOMO + LOWO cross-validation
│       ├── run.sh
│       ├── cross_validate.py    # Leave-one-out cross-validation
│       ├── analyze.py
│       └── FINDINGS.md
├── idea-2-analytical-decomposition/
│   ├── h1-features/             # 4-component FLOPs decomposition
│   │   ├── run.sh
│   │   ├── compute_components.py # Analytical FLOPs computation
│   │   ├── analyze.py
│   │   └── FINDINGS.md
│   └── h2-model/                # Learned correction factors
│       ├── run.sh
│       ├── fit_corrections.py   # Nonlinear least squares fitting
│       ├── analyze.py
│       └── FINDINGS.md
├── idea-3-evolutionary-synthesis/ # Deferred (HYPOTHESIS.md only)
├── problem.md                   # Problem statement
├── research.md                  # Full research document (WP1)
└── DECISION_GATE.md             # WP4/5 decision gate
```

**Dependencies:** Python 3, pandas, scikit-learn, scipy, xgboost
**Data:** `eval/ground_truth/` (122,752 steps from 16 experiments)
**Hardware configs:** `model_configs/` (HuggingFace config.json), `hardware_config.json` (H100 specs)

---

## 11. Summary for Future Ideation

**What the data tells us:**
1. Per-experiment XGBoost with 30 features is 2x better than blackbox (34% vs 70% avg MAPE)
2. The bottleneck is **feature quality** (missing per-request KV), not model capacity
3. Global models are structurally impossible without scale normalization
4. The analytical decomposition is physically correct but data-incomplete
5. "General" workloads are the hardest — high batch composition diversity

**What Round 2 should explore:**
1. Deriving per-request KV features from lifecycle data (`request_metrics/`)
2. Log-scale or normalized prediction for cross-model generalization
3. BLIS E2E validation (does 34% per-step MAPE → <10% E2E mean error?)
4. Simpler models with better features (Ridge/piecewise with per-request KV)
5. Batch composition clustering for "general" workloads
6. Evolutionary synthesis (Idea 3 h1) if OpenEvolve becomes available

**Hard constraints to respect:**
- Per-model training is mandatory (LOMO fails at 2559%)
- Features must be computable from `[]*Request` with `ProgressIndex` (Go integration)
- Prediction latency < 1ms per step
- Pure-Go implementation (no Python/ONNX runtime)
