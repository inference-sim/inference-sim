# StepML Research: WP4/5 Decision Gate

**Date:** 2026-02-26
**Branch:** `stepml-experiments`
**Status:** Research complete. Integration decision required.

---

## Executive Summary

Three research ideas were explored to replace BLIS's blackbox latency model (3-coefficient linear regression on `prefill_tokens` and `decode_tokens`) with a data-driven step-time predictor achieving <10% workload-level E2E mean error:

| Idea | Approach | Best Result | Verdict |
|------|----------|-------------|---------|
| 1. Tree Ensemble | XGBoost with 30 physics-informed features | **34.0% avg MAPE** (2x better than blackbox) | **Best available** |
| 2. Analytical Decomposition | FLOPs decomposition + learned corrections | 78.7% avg MAPE (worse than blackbox) | **Abandoned** |
| 3. Evolutionary Synthesis | LLM-guided MAP-Elites | Not attempted | **Deferred** |

**Bottom line:** No approach meets the original <10% E2E mean error target on all 16 experiments. Per-experiment XGBoost (Idea 1) is the strongest result — a 2x improvement over blackbox with zero regressions. The binding constraint is **missing per-request KV cache lengths** in step-level training data.

---

## Leaderboard

### Per-Step MAPE (sorted best to worst)

| Rank | Model | Avg MAPE | Exps <15% | Exps <25% | Exps <30% | vs Blackbox |
|------|-------|----------|-----------|-----------|-----------|-------------|
| 1 | **XGBoost per-experiment (Idea 1 h2)** | **34.0%** | **5/16** | **6/16** | **9/16** | **+2x better** |
| 2 | Blackbox 2-feature baseline | 70.4% | 1/16 | 1/16 | 2/16 | — |
| 3 | Correction factors 36-param (Idea 2 h2) | 78.7% | 0/16 | 0/16 | 0/16 | -12% worse |
| 4 | Ridge 30-feature per-experiment (Idea 1 h1) | 92.1% | 2/16 | 3/16 | 6/16 | -31% worse |
| 5 | Correction factors 9-param (Idea 2 h2) | 96.0% | 0/16 | 0/16 | 0/16 | -36% worse |
| 6 | Ridge 30-feature global (Idea 1 h1) | 301.7% | — | — | — | Short-circuited |

### Per-Experiment Detail (XGBoost — the winning model)

| Model | Workload | XGB MAPE | BB MAPE | Improvement |
|-------|----------|----------|---------|-------------|
| mixtral-8x7b-v0-1 | general | 9.1% | 9.2% | Match |
| llama-2-70b-hf | reasoning | 9.8% | 14.2% | +4.4 pp |
| llama-2-7b | reasoning | 13.0% | 123.5% | **+110.5 pp** |
| codellama-34b | reasoning | 14.0% | 37.3% | +23.3 pp |
| mixtral-8x7b-v0-1 | codegen | 14.7% | 19.0% | +4.3 pp |
| codellama-34b | codegen | 16.1% | 21.6% | +5.5 pp |
| codellama-34b | roleplay | 25.7% | 30.8% | +5.1 pp |
| mixtral-8x7b-v0-1 | roleplay | 28.3% | 33.6% | +5.3 pp |
| llama-2-70b | roleplay | 28.3% | 128.6% | **+100.2 pp** |
| llama-2-7b | roleplay | 30.9% | 40.3% | +9.4 pp |
| llama-2-70b-hf | codegen | 31.1% | 90.8% | **+59.7 pp** |
| llama-2-7b | codegen | 33.4% | 69.7% | +36.3 pp |
| llama-2-7b | general | 50.9% | 72.9% | +22.0 pp |
| llama-2-70b | general | 55.4% | 61.2% | +5.7 pp |
| mixtral-8x7b-v0-1 | reasoning | 62.4% | 222.8% | **+160.5 pp** |
| codellama-34b | general | 121.5% | 151.1% | +29.6 pp |

**XGBoost beats or matches blackbox on all 16 experiments — zero regressions.**

---

## Success Criteria Evaluation

### Priority 1: Workload-Level E2E Mean Fidelity

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| E2E mean error <10% on all 16 experiments | <10% | **Not tested** (per-step MAPE only) | UNKNOWN |
| Statistically significant improvement | p < 0.05 | XGB beats BB on 16/16 experiments | LIKELY MET |
| Per-step MAPE as diagnostic | Informational | 34.0% avg (5/16 < 15%) | REPORTED |

**Gap:** Workload-level E2E mean error was not computed because that requires running BLIS simulation with the trained model as the latency backend. Per-step MAPE is a diagnostic, not the primary metric. E2E mean error could be better (if per-step errors cancel) or worse (if they compound) than per-step MAPE.

### Priority 2: Per-Metric Accuracy (TTFT, ITL)

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| TTFT mean error <15% | <15% | Not tested | UNKNOWN |
| ITL mean error <15% | <15% | Not tested | UNKNOWN |
| Outperforms blackbox on E2E, TTFT, ITL | All three | Not tested | UNKNOWN |

### Priority 3: Tail Latency Behavior

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| No p99 ranking inversions | No inversions | Not tested | UNKNOWN |
| Extreme errors <10% of steps | <10% | Not tested | UNKNOWN |

### Priority 4: Model Quality and Practicality

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Dense + MoE <15% E2E error | <15% | Mixtral best MAPE = 9.1% (general) | PARTIALLY |
| Features derivable from Request batch | Yes | **Partially** — KV proxies from system state, not Request batch | GAP |
| Retraining documented | Yes | Not yet documented | TODO |
| Reproducible | Yes | Fixed seeds, saved models | MET |
| Prediction <1ms | <1ms | Not benchmarked | UNKNOWN |
| Go integration path identified | Yes | Serialized XGBoost trees in JSON | **MET** |

### Priority 5: Generalization

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Hardware generalization | Informational | H100 only (no cross-GPU data) | REPORTED |
| Quantization generalization | Informational | BF16 only | REPORTED |
| Cross-model generalization | <20% MAPE | LOMO avg 2559.7% | **FAIL** |
| Cross-workload generalization | <20% MAPE | LOWO avg 109.7% | **FAIL** |

---

## Root Cause Analysis

### Binding Constraint: Missing Per-Request KV Cache Lengths

The single most impactful limitation across all ideas is that **step-level training data lacks per-request KV cache lengths** (ProgressIndex). This affects:

1. **Idea 1:** XGBoost uses system-state KV proxies (`kv_blocks_used`, `running_depth`) instead of per-request statistics (`kv_mean`, `kv_max`, `kv_sum`). These proxies capture occupancy but not distribution.

2. **Idea 2:** Decode attention FLOPs are literally zero — uncomputable without per-request KV lengths. This makes the analytical decomposition structurally incomplete for 80.6% of steps (pure decode).

3. **Feature importance confirms it:** XGBoost's top feature is `f_kv_blocks_used` — the best available KV proxy. The model is begging for better KV information.

### Secondary Constraint: Step Time Scale Variation

Step times span 3+ orders of magnitude across model configurations (12μs for Llama-7B decode to 250,000μs for Llama-70B mixed batches). This makes global models fail catastrophically and forces per-model training.

### Why "General" Workloads Are Hardest

The 4 worst experiments (50-121% MAPE) are all "general" workloads. These have the most diverse batch compositions — mixing short decode-only batches with occasional large prefill+decode batches. The high variance requires features that capture per-batch compute structure, not just aggregate token counts.

---

## Convergence Review: Findings Quality

### Idea 1 Findings Assessment

| Dimension | Assessment |
|-----------|------------|
| Results match code | Verified — CSVs match FINDINGS.md tables |
| Statistical rigor | MAPE correctly computed; no significance tests (would strengthen) |
| Baselines compared | Blackbox baseline compared in every experiment |
| Conclusions follow evidence | Yes — "weakly supported" is accurate for 5/16 < 15% |
| Reproducible | `run.sh` is self-contained, seeds fixed |
| Actionable | Clear recommendation: per-experiment XGBoost, extend LatencyModel for KV |

### Idea 2 Findings Assessment

| Dimension | Assessment |
|-----------|------------|
| Results match code | Verified — decode_attn_factor = 39.5 amplifying zero signal |
| Root cause identified | Yes — missing per-request KV is the structural blocker |
| Abandonment justified | Yes — 36 params worse than 3-param blackbox is definitive |
| Salvage path documented | Yes — revisit if LatencyModel gains ProgressIndex |

### Idea 3 Assessment

| Dimension | Assessment |
|-----------|------------|
| Not attempted | Justified — h2 depends on Idea 2 (failed), h1/h3 require OpenEvolve infrastructure |
| Deferred, not abandoned | Correct — evolutionary search could still discover useful formulas if data improves |

---

## Decision

### Recommendation: Conditional Integration of Per-Experiment XGBoost

**Integrate Idea 1's per-experiment XGBoost as a `StepMLLatencyModel`** behind the existing `LatencyModel` interface, with these conditions:

1. **Ship as opt-in** (`--stepml` flag), not as the default. The blackbox model remains default until E2E validation completes.

2. **Per-model coefficient files** (serialized XGBoost trees in JSON), analogous to existing `defaults.yaml` alpha/beta coefficients. One file per model configuration.

3. **Feature computation in Go** — the 30-feature vector is computed from batch composition at step time. The ~20 features that depend on system-state KV proxies use the existing `KVStore` observation methods.

4. **Pure-Go tree evaluator** — XGBoost trees are if/else comparisons on feature thresholds. No CGo, no Python runtime, no ONNX. ~200 lines of Go.

### Deferred Work (Not Part of Initial Integration)

| Item | Reason |
|------|--------|
| Workload-level E2E validation | Requires BLIS simulation loop with StepML backend — do as Go integration test |
| TTFT/ITL mean accuracy | Requires mapping step predictions to per-request metrics — do during integration |
| Per-request KV features | Requires extending `LatencyModel.StepTime()` to receive `[]Request` with ProgressIndex |
| Cross-model generalization | Per-model training is the permanent approach (consistent with Vidur/Splitwise) |
| Idea 3 evolutionary search | Deferred until data quality improves (per-request KV available) |

### Go Integration Path (WP6 Scope)

```
sim/latency/
├── latency.go          # Existing: BlackboxLatencyModel, RooflineLatencyModel
├── stepml.go           # NEW: StepMLLatencyModel (implements LatencyModel interface)
├── stepml_features.go  # NEW: ComputeFeatures(batch, modelConfig) []float64
├── stepml_tree.go      # NEW: Pure-Go XGBoost tree evaluator
├── stepml_models/      # NEW: Per-model JSON coefficient files
│   ├── llama-2-7b.json
│   ├── llama-2-70b.json
│   ├── codellama-34b.json
│   └── mixtral-8x7b.json
└── register.go         # Modified: register StepML alongside Blackbox and Roofline
```

**Feature contract:** `ComputeFeatures()` receives the same `BatchContext` that `BlackboxLatencyModel.StepTime()` already receives — no interface changes needed for the initial integration. The system-state KV features (`kv_blocks_used`, `running_depth`) are available from the `InstanceSimulator` observation methods.

---

## Open Questions for WP6

1. **Should the XGBoost models be re-trained with more data?** Current training uses 60% of ~7,600 steps per experiment (~4,500 training steps). More calibration data might improve the 11 experiments above 15% MAPE.

2. **Feature availability at inference time:** The 30-feature set includes some features (`kv.blocks_free_gpu`, `queue.running_depth`) that come from simulator state, not the `BatchContext`. How should these be plumbed into the `LatencyModel.StepTime()` call?

3. **Model file distribution:** Should the JSON coefficient files be embedded in the Go binary (via `//go:embed`), loaded from disk (like `defaults.yaml`), or fetched alongside HuggingFace configs (like `--roofline`)?

4. **Fallback behavior:** If no StepML model exists for a given `--model`, should BLIS fall back to blackbox or error?
