# StepML Research: Macro-Level Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the blackbox latency model with a data-driven alternative achieving <10% workload-level E2E mean error, then integrate the winner into BLIS as a third LatencyModel.

**Architecture:** Python research pipeline (data loading → ideation → parallel experimentation → leaderboard) followed by Go integration via the existing LatencyModel policy template extension recipe.

**Tech Stack:** Python (scikit-learn, XGBoost, pandas, scipy), Go (sim/latency/ package)

**Date:** 2026-02-26
**Status:** Draft
**Based on:** [Design Document](2026-02-26-stepml-research-design.md)

---

## A) Executive Summary

The blackbox latency model uses 2 features / 3 beta coefficients and ignores batch composition, KV cache lengths, and MoE compute patterns. This plan replaces it with a data-driven model trained on ~165K step-level ground-truth observations from instrumented vLLM.

**What changes:** A new StepML LatencyModel backend joins blackbox and roofline. The roofline model is unaffected.

**Implementation:** 7 ordered work packages (WP0–WP6), spanning Python research experimentation and Go production integration.

- **WP0** — Shared Python infrastructure: data pipeline, evaluation harness, baselines, sampling bias characterization, multi-strategy data splits (temporal + model-wise + workload-wise)
- **WP1** — Research ideation via `/research-ideas` (3+ literature-grounded ideas)
- **WP2** — Hypothesis scaffolding (3 sub-hypotheses per idea: features → model → generalization)
- **WP3** — Wave-parallel experimentation across ideas (3 waves: h1→h2→h3, with short-circuit at 30% MAPE)
- **WP4** — Leaderboard comparison and winner selection
- **WP5** — Go integration: StepML LatencyModel in `sim/latency/`, factory extension, CLI flags
- **WP6** — BLIS validation: calibration pairs, golden dataset update, CLAUDE.md update

**Key constraint:** The frozen LatencyModel interface (`sim/latency_model.go:7-23`) is not modified. StepML registers via the existing factory pattern (`sim/latency/register.go:10-12`).

---

## B) Repository Recon Summary

### B.1 Affected Packages and Files

| Package | Files Affected | Change Type |
|---|---|---|
| `hypotheses/h-stepml/` | New directory tree | New — all Python research artifacts |
| `sim/latency/` | New `stepml.go` + modified `latency.go` factory | New + Moderate — StepML implementation + factory three-way dispatch |
| `sim/` | `latency_model.go` unchanged; `config.go` minor extension | Minor — add StepML artifact path to config |
| `cmd/` | `root.go` | Minor — new CLI flag for StepML artifact path |
| `defaults.yaml` | Minor | Add StepML default config section |
| `testdata/` | `goldendataset.json` | Regenerate — blackbox replacement changes outputs |

### B.2 Core Data Structures (Frozen — facts from merged code)

- **LatencyModel** (`sim/latency_model.go:7-23`) — 5-method interface. Frozen. The StepML model implements this interface without modification.
- **NewLatencyModelFunc** (`sim/latency_model.go:31`) — Registration variable. The factory in `sim/latency/latency.go:129-167` dispatches on `hw.Roofline`; StepML adds a second dispatch branch.
- **Request.ProgressIndex** (`sim/request.go:34`) — `int64`, tracks cumulative token progress. Used as KV cache length proxy (design doc Gap 1 bridge).
- **BlackboxLatencyModel** (`sim/latency/latency.go:18-58`) — The model being replaced. Uses `betaCoeffs[0..2]` (intercept + cacheMissTokens + decodeTokens) and `alphaCoeffs[0..2]` (queueing + output processing).
- **RooflineLatencyModel** (`sim/latency/latency.go:64-111`) — Unaffected. Uses per-request `ProgressIndex` and `NumNewTokens` to build `StepConfig`.

### B.3 Current Factory Dispatch (Confirmed)

`sim/latency/latency.go:129-167`: `NewLatencyModel(coeffs, hw)` checks `hw.Roofline` → true returns `RooflineLatencyModel`, false returns `BlackboxLatencyModel`. StepML adds a third branch: if StepML artifact exists at configured path, return `StepMLLatencyModel`; else fall back to blackbox.

### B.4 Call Sites (Confirmed)

| Method | Call Site | Context |
|---|---|---|
| `StepTime` | `sim/simulator.go:415` | Per-step batch duration |
| `QueueingTime` | `sim/event.go:31` | Arrival-to-queue delay |
| `OutputTokenProcessingTime` | `sim/simulator.go:433,437,466` | Per-token post-processing |
| `SchedulingProcessingTime` | `sim/batch_formation.go:131` | Scheduling overhead |
| `PreemptionProcessingTime` | `sim/batch_formation.go:160` | Preemption overhead |

### B.5 Hypothesis Infrastructure (Confirmed)

- Shared harness: `hypotheses/lib/harness.sh` (bash), `hypotheses/lib/analyze_helpers.py` (Python)
- Template: `docs/templates/hypothesis.md` — FINDINGS.md, run.sh, analyze.py structure
- 40 existing hypothesis directories under `hypotheses/`
- No `hypotheses/h-stepml/` directory exists yet

### B.6 Existing Latency Model Tests

- `sim/latency/latency_test.go` — Unit tests for blackbox and roofline StepTime
- `sim/latency/roofline_test.go` — Roofline-specific calculation tests
- `testdata/goldendataset.json` — Regression golden dataset (will need regeneration after StepML becomes default)

---

## C) High-Level Objectives + Non-Goals + Model Scoping

### Objectives

1. **Build shared Python infrastructure** for loading, splitting, evaluating, and comparing step-time prediction models across 20 experiments (16 main + 4 sweep)
2. **Generate 3+ research ideas** via `/research-ideas` with literature citations and LLM judge reviews
3. **Execute parallel experiments** (feature engineering → model training → generalization) with convergence gates at each stage
4. **Select a winner** achieving <10% workload-level E2E mean error on all 16 experiments
5. **Integrate into BLIS** as a third LatencyModel implementation via the policy template extension recipe
6. **Validate end-to-end** using BLIS calibration infrastructure on standard workload scenarios

### Non-Goals

- Modifying the roofline latency model in any way — roofline must produce **byte-identical output** before and after this work. No roofline code, configuration, alpha coefficients, or dispatch path is touched. StepML overhead findings (scheduling, preemption, queueing) are NOT propagated to roofline; roofline continues to return 0 for scheduling/preemption and use its existing alpha coefficients for queueing/output processing.
- Multi-instance cluster-level effects (step time is instance-local)
- Non-H100 hardware training (evaluation dimension only)
- vLLM versions other than v0.15.1 (evaluation dimension only)
- Automated retraining pipeline (future work)

### Model Scoping Table

| Component | Modeled | Simplified | Omitted | Justification |
|-----------|---------|------------|---------|---------------|
| Batch composition (prefill/decode tokens, request counts) | Yes | — | — | Direct causal inputs to step FLOPs |
| Per-request KV cache lengths | Yes (via ProgressIndex proxy) | — | — | H8 showed 12.96× overestimate without per-request KV lengths |
| Model architecture (dense vs. MoE) | Yes (experiment metadata) | — | — | Fundamentally different compute patterns |
| Hardware characteristics | — | H100-only training | — | Single GPU generation; cross-hardware is P5 evaluation |
| vLLM scheduler internals | — | Observed via batch composition | — | Step time depends on what's in the batch, not how the batch was formed. Lost: preemption overhead (<5% of step time) |
| Kernel-level scheduling | — | — | Yes | Below per-step abstraction; captured in target variable |
| Temporal dynamics | — | Independent prediction | — | Simplest approach answering AQ-1. Lost: CUDA cache warming, thermal throttling |
| Quantization effects | — | BF16-only | — | P6 lowest priority; informational only |

---

## D) Concept Model

### Building Blocks (5)

**1. Data Pipeline** — Parses 20 experiments into a unified dataset; computes reproducible splits.
- OBSERVES: Raw `traces.json` and `per_request_lifecycle_metrics.json` files
- CONTROLS: Unified Parquet dataset, split indices (temporal, model-wise, workload-wise)
- OWNS: Parsed feature matrix, split assignment arrays
- INVARIANTS: Row count matches source; temporal ordering within experiments; all 16 combinations represented per split
- EVENTS: None (offline processing)
- EXTENSION FRICTION: 1 file to add a new data source or split strategy

**2. Evaluation Harness** — Computes all metrics (MAPE, MSPE, Pearson r, p99 error, workload-level E2E mean error) and baseline comparisons.
- OBSERVES: Predicted step times, ground-truth step times, per-request lifecycle data
- CONTROLS: Metric values, pass/fail decisions, leaderboard rankings
- OWNS: Metric computation functions, baseline model implementations (blackbox, roofline, naive mean)
- INVARIANTS: Metrics are deterministic; baselines re-trained on the same training split as research models
- EVENTS: None (pure computation)
- EXTENSION FRICTION: 1 file to add a new metric or baseline

**3. Experiment Engine** — Per-idea hypothesis chain (h1→h2→h3) with convergence gates.
- OBSERVES: Shared dataset and split indices (from Data Pipeline), evaluation metrics (from Evaluation Harness)
- CONTROLS: Feature extraction, model training, generalization evaluation, short-circuit decisions
- OWNS: Per-idea model artifacts, feature extractors, FINDINGS.md
- INVARIANTS: h2 depends on h1; h3 depends on h2; ideas with h1 MAPE > 30% are dropped
- EVENTS: None (offline processing)
- EXTENSION FRICTION: 3 files per new idea (h1/h2/h3 directories with run.sh + analyze.py + FINDINGS.md)

**4. StepML LatencyModel** (Go) — Third LatencyModel implementation behind the frozen interface.
- OBSERVES: Batch of `*Request` (per-request InputTokens, OutputTokens, ProgressIndex, NumNewTokens)
- CONTROLS: 5 latency estimates (step time, queueing, output processing, scheduling, preemption)
- OWNS: Trained model weights/coefficients (immutable after construction)
- INVARIANTS: INV-M-1 (positive output), INV-M-2 (deterministic), INV-M-3 (side-effect-free), INV-M-4 (< 1ms), INV-M-5 (soft monotonicity), INV-M-6 (|MSPE| < 5%)
- EVENTS: None (synchronous query in event loop)
- EXTENSION FRICTION: ~3-4 files (implementation, factory branch, config type, CLI flag)

**5. Leaderboard** — Cross-idea comparison on held-out test set, ordered by metrics priority (P1→P5).
- OBSERVES: All ideas' FINDINGS.md and evaluation results
- CONTROLS: Winner selection, final README.md
- OWNS: Leaderboard table, comparison analysis
- INVARIANTS: Test set never seen during training/tuning; rankings use P1 metric first (E2E mean)
- EVENTS: None
- EXTENSION FRICTION: 1 row per new idea

### Interaction Model

```
[Data Pipeline] ──read-only──→ [Experiment Engine (per idea, parallel)]
                                      │
[Evaluation Harness] ◄──metrics──────┘
                                      │
[Leaderboard] ◄──────results──────────┘
       │
       ▼ (winner's artifacts)
[StepML LatencyModel (Go)]
```

### Real-System Correspondence

| Building Block | vLLM v0.15.1 | BLIS |
|---|---|---|
| Data Pipeline | `traces.json` step.BATCH_SUMMARY events | `hypotheses/h-stepml/shared/` |
| Evaluation Harness | Offline metric computation | `hypotheses/h-stepml/shared/evaluation.py` |
| StepML LatencyModel | Wall-clock `model_execute()` duration | `sim/latency/stepml.go` (new) |

---

## E) Architectural Risk Register

| # | Decision | Assumption | Validation Method | Cost if Wrong | Gate |
|---|----------|------------|-------------------|---------------|------|
| R1 | ProgressIndex as KV length proxy (Gap 1 bridge) | ProgressIndex ≈ total KV cache length per request | Compare ProgressIndex values with ground-truth per-request KV lengths from lifecycle data during WP0 | WP3–WP6 rework (all features based on wrong proxy) | Before WP3 start |
| R2 | 10% sampling is sufficient for training | Sampling is approximately random, not periodic | Characterize sampling distribution in WP0: step_id uniformity, autocorrelation, per-experiment representation | WP3–WP4 rework (models trained on biased data) | Before WP3 start |
| R3 | Request-batch features suffice (no interface extension) | Per-step prediction accuracy from Request fields alone can achieve <10% E2E mean error | Best model on Request-only features must pass E2E <15% on majority of experiments | WP5–WP6 rework (LatencyModel interface needs extension) | Before WP5 start |
| R4 | Blackbox baseline is beatable | The 2-feature model has significant room for improvement | Compute blackbox per-step MAPE and E2E mean error on training split in WP0. If blackbox E2E mean < 12%, reassess research justification | All WP1–WP6 wasted | Before WP1 start |
| R5 | Winning model portable to Go | The winning model can be expressed as coefficients, ONNX, or compact code | Each idea must document its Go integration path during WP3 | WP5 rework (need different integration strategy) | Before WP5 start |

**Abort plans:**
- R1 fails → derive per-request KV lengths from lifecycle data instead of ProgressIndex; may require richer feature extraction
- R2 fails → request full-trace re-collection or apply importance weighting to correct sampling bias
- R3 fails → propose LatencyModel interface extension (adds batch-level metadata parameter); requires separate design doc
- R4 fails → cancel research; blackbox is already good enough
- R5 fails → use Python subprocess or ONNX runtime from Go (adds dependency; acceptable for research prototype)

---

## F) Architectural Evolution

### Current State

Two LatencyModel backends (`sim/latency/latency.go:129-167`):
- `hw.Roofline == true` → RooflineLatencyModel (analytical FLOPs/bandwidth)
- `hw.Roofline == false` → BlackboxLatencyModel (2-feature, 3-coefficient regression)

Factory dispatch is binary. No fallback chain. No StepML concept.

### Target State

Three LatencyModel backends with ordered dispatch:
1. `hw.Roofline == true` → RooflineLatencyModel (unchanged)
2. StepML artifact exists at configured path → StepMLLatencyModel (new)
3. Fallback → BlackboxLatencyModel (preserved for backward compatibility)

New CLI flag `--stepml-model-path` specifies the artifact location. When absent, falls back to blackbox (no behavior change for existing users).

### What Remains Unchanged

- LatencyModel interface (frozen at `sim/latency_model.go:7-23`)
- **RooflineLatencyModel: zero code changes.** All 5 method implementations are unchanged. Alpha coefficients passed to roofline are unchanged. The `--roofline` code path returns before the new StepML dispatch branch is reached (`sim/latency/latency.go:137`). Roofline output is byte-identical before and after this plan.
- All call sites (`sim/simulator.go`, `sim/event.go`, `sim/batch_formation.go`)
- Batch formation, scheduling, routing, KV cache modules
- Cluster-level configuration (LatencyModel is instance-local)

### Roofline Isolation Guarantee

The StepML research may discover improved estimates for scheduling overhead, preemption overhead, and queueing delay. These findings apply **only to the new StepML LatencyModel**. They are NOT propagated to the roofline model because:

1. **Factory dispatch order:** `hw.Roofline == true` is checked first (`sim/latency/latency.go:137`). The StepML branch is unreachable when roofline is active.
2. **No shared mutable state:** Roofline's alpha coefficients are loaded from the existing `defaults.yaml` / `--roofline` config path. StepML uses its own separate artifact. No coefficient sharing.
3. **No defaults.yaml changes for roofline:** StepML config fields are additive (new section). Existing roofline/blackbox defaults are untouched.
4. **Verification (WP5, BC-5-4):** The Go integration PR must include a regression test that runs the same simulation with `--roofline` before and after StepML integration and asserts byte-identical stdout (INV-6).

---

## G) Frozen Interface Reference

The following interface is frozen (merged code, not aspirational):

```go
// sim/latency_model.go:7-23
type LatencyModel interface {
    StepTime(batch []*Request) int64
    QueueingTime(req *Request) int64
    OutputTokenProcessingTime() int64
    SchedulingProcessingTime() int64
    PreemptionProcessingTime() int64
}
```

Registration variable (`sim/latency_model.go:31`):
```go
var NewLatencyModelFunc func(coeffs LatencyCoeffs, hw ModelHardwareConfig) (LatencyModel, error)
```

Factory (`sim/latency/latency.go:129`):
```go
func NewLatencyModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error)
```

---

## H) Cross-Cutting Infrastructure Plan

### H.1 Shared Test Infrastructure

- **Existing:** `hypotheses/lib/harness.sh` (bash experiment harness), `hypotheses/lib/analyze_helpers.py` (Python analysis helpers)
- **New (WP0):** `hypotheses/h-stepml/shared/` — data loader, split computation, evaluation harness, baselines. All WP3 experiments import from here.
- **New (WP5):** `sim/latency/stepml_test.go` — Go unit tests for StepML LatencyModel
- **Invariant tests (WP5):** INV-M-1 through INV-M-6 companion tests for every golden test

### H.2 Documentation Maintenance

| Trigger | Owner | What Updates |
|---|---|---|
| WP0 creates `hypotheses/h-stepml/` | WP0 | CLAUDE.md file organization section |
| WP5 adds `sim/latency/stepml.go` | WP5 | CLAUDE.md file organization + latency estimation section |
| WP5 adds `--stepml-model-path` flag | WP5 | CLAUDE.md CLI flags section |
| WP6 regenerates golden dataset | WP6 | CLAUDE.md golden dataset note |

### H.3 CI Pipeline Changes

- **WP0–WP4:** No CI changes (Python research code is not CI-gated; reproducibility via `run.sh`)
- **WP5:** New Go test file added to existing `go test ./sim/latency/...`
- **WP6:** Golden dataset regeneration + `go test ./...` verification

### H.4 Dependency Management

- **Python (WP0–WP4):** `requirements.txt` in `hypotheses/h-stepml/shared/` with pinned versions (pandas, scikit-learn, scipy, pyarrow). No Go dependency changes during research.
- **Go (WP5):** Depends on winning model's integration path. Coefficient export requires zero new dependencies. ONNX requires `onnxruntime-go`. Decision deferred to WP5 micro-planning.

### H.5 Interface Freeze Schedule

- **LatencyModel interface:** Already frozen (`sim/latency_model.go:7-23`). No changes planned.
- **NewLatencyModelFunc signature:** Already frozen (`sim/latency_model.go:31`). StepML uses the existing `ModelHardwareConfig` parameter with optional new fields.
- **No new interfaces introduced** by this plan. StepML is a policy template (new algorithm behind existing interface).

---

## I) Work Package Plan

### WP0: Shared Python Infrastructure

**Building Block Change:** Adds Data Pipeline + Evaluation Harness
**Extension Type:** Subsystem module (new infrastructure)
**Motivation:** All subsequent work packages depend on consistent data loading, splitting, evaluation, and baseline comparison. Building this once prevents per-idea reimplementation.

**Scope:**
- In: Data parsing (20 experiments), multi-strategy splits (temporal 60/20/20, leave-one-model-out 4-fold, leave-one-workload-out 4-fold), evaluation harness (MAPE, MSPE, Pearson r, p99, workload-level E2E mean error via synthetic trace replay), baselines (blackbox re-trained on training split, roofline, naive mean), sampling bias characterization, prefix cache semantics resolution (D-8)
- Out: Any model training, any idea-specific code

**Behavioral Guarantees:**
- BC-0-1: Row count of parsed dataset matches sum of source file step counts
- BC-0-2: Within each experiment, all training step_ids < all validation step_ids < all test step_ids
- BC-0-3: Each split contains data from all 16 model×workload combinations
- BC-0-4: Blackbox baseline coefficients are re-trained on the training split (not defaults.yaml values)
- BC-0-5: Sampling bias report documents: step_id uniformity, autocorrelation, per-experiment representation

**Risks:**
1. Sampling is systematically biased → characterization in this WP detects it; abort plan per R2
2. Blackbox baseline already achieves <12% E2E mean → reassess research justification per R4

**Cross-Cutting:** Creates `hypotheses/h-stepml/shared/` consumed by all WP3 experiments. Updates CLAUDE.md file organization.

**Validation Gate:** R1 (ProgressIndex proxy), R2 (sampling bias), R4 (blackbox beatable) must all pass before WP1.

**Tasks:**

#### Task 1: Create directory structure

**Step 1:** Create the shared infrastructure directory.
```bash
mkdir -p hypotheses/h-stepml/shared
```

**Step 2:** Create `requirements.txt` with pinned Python dependencies.

**Files:**
- Create: `hypotheses/h-stepml/shared/requirements.txt`

#### Task 2: Implement data loader

**Step 1: Write the failing test**
Create `hypotheses/h-stepml/shared/test_data_loader.py` with tests asserting:
- All 20 experiments are loaded
- Row count matches source files
- All expected columns present
- `step.duration_us` equals `ts_end_ns - ts_start_ns` (verify semantics)

**Step 2: Run test to verify it fails**
```bash
cd hypotheses/h-stepml/shared && python3 -m pytest test_data_loader.py -v
```
Expected: FAIL (module not found)

**Step 3: Write implementation**
Create `hypotheses/h-stepml/shared/data_loader.py`:
- Parse `traces.json` step.BATCH_SUMMARY events from each experiment
- Join with per-request lifecycle data to derive per-request KV lengths (via progress index)
- Output: unified Parquet dataset with experiment metadata columns

**Step 4: Run test to verify it passes**
```bash
cd hypotheses/h-stepml/shared && python3 -m pytest test_data_loader.py -v
```

**Step 5: Commit**
```bash
git add hypotheses/h-stepml/shared/data_loader.py hypotheses/h-stepml/shared/test_data_loader.py
git commit -m "feat(stepml): add data loader for step-level ground truth"
```

**Files:**
- Create: `hypotheses/h-stepml/shared/data_loader.py`
- Create: `hypotheses/h-stepml/shared/test_data_loader.py`

#### Task 3: Implement sampling bias characterization

**Step 1: Write the failing test**
Create `hypotheses/h-stepml/shared/test_sampling.py`:
- Test step_id distribution uniformity (KS test against uniform)
- Test per-experiment representation proportionality
- Test autocorrelation of consecutive step IDs

**Step 2: Write implementation**
Create `hypotheses/h-stepml/shared/sampling_analysis.py`:
- Compute step_id distribution vs. uniform random
- Check for periodicity (autocorrelation at lag 10)
- Verify 16 model×workload proportional representation
- Output: sampling bias characterization report

**Step 3: Run and verify**
```bash
cd hypotheses/h-stepml/shared && python3 -m pytest test_sampling.py -v
```

**Step 4: Commit**

**Files:**
- Create: `hypotheses/h-stepml/shared/sampling_analysis.py`
- Create: `hypotheses/h-stepml/shared/test_sampling.py`

#### Task 4: Implement data splits

**Step 1: Write the failing test**
Create `hypotheses/h-stepml/shared/test_splits.py`:
- Test temporal ordering within each experiment (BC-0-2)
- Test stratification across experiments (BC-0-3)
- Test leave-one-model-out produces 4 folds with correct holdout
- Test leave-one-workload-out produces 4 folds with correct holdout
- Test split indices are reproducible (same seed → same indices)

**Step 2: Write implementation**
Create `hypotheses/h-stepml/shared/splits.py`:
- Temporal split (60/20/20 within each experiment by step_id order)
- Leave-one-model-out split (4 folds: each model held out once)
- Leave-one-workload-out split (4 folds: each workload held out once)
- Save splits as index arrays (not data copies) for reproducibility

**Step 3: Run and verify**

**Step 4: Commit**

**Files:**
- Create: `hypotheses/h-stepml/shared/splits.py`
- Create: `hypotheses/h-stepml/shared/test_splits.py`

#### Task 5: Implement evaluation harness

**Step 1: Write the failing test**
Create `hypotheses/h-stepml/shared/test_evaluation.py`:
- Test MAPE computation on known inputs
- Test MSPE computation (signed error)
- Test Pearson r computation
- Test p99 error computation
- Test workload-level E2E mean error via synthetic trace replay
- Test bootstrap 95% confidence interval

**Step 2: Write implementation**
Create `hypotheses/h-stepml/shared/evaluation.py`:
- `compute_mape(predicted, actual)` → float
- `compute_mspe(predicted, actual)` → float (signed)
- `compute_pearson_r(predicted, actual)` → float
- `compute_p99_error(predicted, actual)` → float
- `compute_e2e_mean_error(predicted_steps, lifecycle_data, experiment_id)` → float
- `compute_ttft_mean_error(predicted_steps, lifecycle_data, experiment_id)` → float
- `compute_itl_mean_error(predicted_steps, lifecycle_data, experiment_id)` → float
- `bootstrap_ci(metric_fn, predicted, actual, n_resamples=1000)` → (lower, upper)

**Step 3: Run and verify**

**Step 4: Commit**

**Files:**
- Create: `hypotheses/h-stepml/shared/evaluation.py`
- Create: `hypotheses/h-stepml/shared/test_evaluation.py`

#### Task 6: Implement baselines

**Step 1: Write the failing test**
Create `hypotheses/h-stepml/shared/test_baselines.py`:
- Test blackbox baseline re-trained on training split (not defaults.yaml)
- Test roofline baseline produces predictions for all experiments with model configs
- Test naive mean baseline returns training set mean for all predictions
- Test all baselines produce positive predictions

**Step 2: Write implementation**
Create `hypotheses/h-stepml/shared/baselines.py`:
- `BlackboxBaseline`: Re-train 3-coefficient regression on training split (BC-0-4)
- `RooflineBaseline`: Wrap BLIS roofline predictions (if model configs available)
- `NaiveMeanBaseline`: Always predict training set mean `step_duration_us`
- Compute and report all baselines' MAPE, MSPE, Pearson r, workload-level E2E mean error

**Step 3: Run and verify. Calibrate short-circuit threshold:**
- If blackbox per-step MAPE > 25%, set threshold to `blackbox_MAPE + 10%`
- If blackbox E2E mean error < 12%, flag for research justification review (R4)

**Step 4: Commit**

**Files:**
- Create: `hypotheses/h-stepml/shared/baselines.py`
- Create: `hypotheses/h-stepml/shared/test_baselines.py`

#### Task 7: Validate temporal split effectiveness

**Step 1:** Train a simple model (Ridge regression on batch features) with random split and temporal split. Compare MAPE. If random split MAPE is much lower (>5% gap), temporal split is correctly preventing leakage. Document the finding.

**Step 2: Commit**

#### Task 8: Validate ProgressIndex as KV proxy (R1 gate)

**Step 1:** For experiments with per-request lifecycle data, compare ProgressIndex values with actual cumulative token counts. Compute correlation. Document whether ProgressIndex is a faithful proxy.

**Step 2:** If correlation < 0.9, flag R1 and derive KV lengths from lifecycle data instead.

**Step 3: Commit**

---

### WP1: Research Ideation

**Building Block Change:** Populates Experiment Engine inputs (research ideas)
**Extension Type:** Process (skill invocation, not code change)
**Motivation:** Generate 3+ literature-grounded research ideas for step-time prediction, iteratively refined by external LLM judges.

**Scope:**
- In: Write `problem.md` from design doc, invoke `/research-ideas` with 12 targeted web search queries, 3 iterations × 3 LLM judges
- Out: Model training, experiment execution

**Behavioral Guarantees:**
- BC-1-1: `research.md` contains 3+ ideas, each with literature citations
- BC-1-2: Each idea addresses all evaluation dimensions (P1–P5)
- BC-1-3: Each idea documents its Go integration path
- BC-1-4: Each idea specifies which LatencyModel methods it covers (minimum: StepTime)

**Risks:**
1. Ideas are too similar → explicitly request diverse algorithmic families in problem.md
2. Literature search misses key papers → 12 targeted queries covering simulators, GPU prediction, MoE, evolutionary methods

**Cross-Cutting:** None (produces `research.md` consumed by WP2)

**Validation Gate:** None (quality enforced by LLM review iterations)

**Tasks:**

#### Task 1: Write problem.md

Derive from design doc sections: Problem Statement, Ground-Truth Data, Evaluation Framework, Baselines, Algorithm Scope. Include WP0 baseline results as context.

#### Task 2: Invoke `/research-ideas`

Use pre-written `problem.md`. Fresh start. Background sources: this repo, data collection repo, InferSim bench data, roofline design doc. 12 web search queries. 3 LLM judges, 3 iterations.

#### Task 3: Review and commit research.md

Verify BC-1-1 through BC-1-4. Commit to `hypotheses/h-stepml/`.

---

### WP2: Hypothesis Scaffolding

**Building Block Change:** Adds Experiment Engine structure (per-idea directories)
**Extension Type:** Process (directory creation + HYPOTHESIS.md authoring)
**Motivation:** Map research ideas to testable hypotheses following BLIS experiment standards.

**Scope:**
- In: Extract top 3-5 ideas from `research.md`, create per-idea directories with 3 sub-hypotheses each, write HYPOTHESIS.md files with literature citations from research.md
- Out: Experiment code, running experiments

**Behavioral Guarantees:**
- BC-2-1: Each idea has exactly 3 sub-hypotheses: h1-features, h2-model, h3-generalization
- BC-2-2: Each HYPOTHESIS.md includes Related Work section with citations from research.md
- BC-2-3: Each h1-features HYPOTHESIS.md specifies the exact feature set to test
- BC-2-4: Each idea's directory follows `hypotheses/h-stepml/idea-N-<name>/` naming

**Risks:**
1. Ideas don't map cleanly to hypothesis structure → each idea must specify a falsifiable claim

**Cross-Cutting:** Creates directory structure consumed by WP3 experiments.

**Validation Gate:** Convergence review (h-design, 5 perspectives) on each idea's HYPOTHESIS.md files before WP3.

**Tasks:**

#### Task 1: Extract ideas and create directories

For each of the top 3-5 ideas from research.md:
```bash
mkdir -p hypotheses/h-stepml/idea-N-<name>/{h1-features,h2-model,h3-generalization}
```

#### Task 2: Write HYPOTHESIS.md files

Per idea, write 3 HYPOTHESIS.md files following `docs/templates/hypothesis.md`. Each must include:
- Status: Pending
- Family: Performance-regime
- Type: Type 2 (Statistical — Dominance)
- Claim, Refuted-If, Related Work, Algorithmic Justification

#### Task 3: Run convergence review

```
/convergence-review h-design --model sonnet
```
Zero CRITICAL, zero IMPORTANT across 5 perspectives. Iterate until converged.

#### Task 4: Commit all scaffolding

---

### WP3: Wave-Parallel Experimentation

**Building Block Change:** Executes Experiment Engine (per-idea h1→h2→h3 chains)
**Extension Type:** Policy template (new algorithms following hypothesis standard)
**Motivation:** Test each research idea's feature engineering, model training, and generalization in parallel across ideas, sequential within each idea.

**Scope:**
- In: All 3 sub-hypotheses per idea, using WP0's shared infrastructure
- Out: Go integration (that's WP5)

**Behavioral Guarantees:**
- BC-3-1: Every `run.sh` imports from `hypotheses/h-stepml/shared/` (no custom data loading)
- BC-3-2: Every `analyze.py` uses the shared evaluation harness
- BC-3-3: Every FINDINGS.md includes all evaluation dimensions + MSPE + p99 analysis
- BC-3-4: Ideas with h1 per-step MAPE > 30% (or adjusted threshold from WP0) are dropped before h2
- BC-3-5: Baselines (blackbox [must outperform], roofline [informational], naive mean) are compared in every analyze.py

**Risks:**
1. All ideas fail the 30% threshold → re-examine feature assumptions; revisit R1 (ProgressIndex proxy)
2. Convergence review stalls → max 10 rounds per gate; fall back to self-audit

**Cross-Cutting:** Consumes WP0 shared infrastructure. Each idea writes only to its own directory.

**Validation Gate:** R3 (Request-batch features suffice) evaluated after WP3 completes. If best model on Request-only features achieves E2E > 15% on majority of experiments, R3 fails.

**Tasks:**

#### Wave 1: Feature Engineering (parallel across ideas)

Per idea, scaffold and run h1-features:

**Task 1: Scaffold h1-features experiments**
- Write `run.sh` sourcing shared harness
- Write `analyze.py` importing shared evaluation
- Run convergence review (h-code, 5 perspectives)

**Task 2: Execute h1-features**
- Run all ideas' h1-features in parallel (independent: same dataset, separate directories)
- Apply short-circuit: drop ideas with MAPE > threshold

**Task 3: Document h1-features findings**
- Write FINDINGS.md for each idea
- Run convergence review (h-findings, 10 perspectives, opus)

#### Wave 2: Model Training (parallel across surviving ideas)

Same pattern as Wave 1, for h2-model. Uses h1's best feature set.

#### Wave 3: Generalization (parallel across surviving ideas)

Same pattern, for h3-generalization. Uses all split strategies:
- Temporal validation accuracy (baseline)
- Leave-one-model-out accuracy (hardest)
- Leave-one-workload-out accuracy (medium)
- Sweep experiment holdout accuracy

---

### WP4: Leaderboard & Selection

**Building Block Change:** Populates Leaderboard
**Extension Type:** Process (analysis + documentation)
**Motivation:** Compare all surviving ideas on the held-out test set and select the winner.

**Scope:**
- In: All ideas' FINDINGS.md, test-set evaluation
- Out: Winner selection, README.md with leaderboard table

**Behavioral Guarantees:**
- BC-4-1: Test set used for final evaluation only (never seen during training/tuning)
- BC-4-2: Rankings ordered by P1 metric first (workload-level E2E mean error)
- BC-4-3: Winner identified with Go integration path documented
- BC-4-4: Falsification criteria checked (design doc section)

**Risks:**
1. No idea achieves <10% E2E mean → evaluate falsification criteria; consider abandoning StepML

**Cross-Cutting:** Produces README.md in `hypotheses/h-stepml/`. Inputs to WP5.

**Validation Gate:** R3 (Request-batch features) and R5 (Go portability) confirmed for the winner.

**Tasks:**

#### Task 1: Run test-set evaluation

Evaluate all surviving ideas on the held-out test set using the shared evaluation harness. Compute all metrics from the leaderboard format (design doc).

#### Task 2: Write leaderboard README.md

Populate the leaderboard table (design doc format). Document winner selection rationale.

#### Task 3: Check falsification criteria

Verify none of the 6 falsification conditions are met. If any are, document and reassess.

#### Task 4: Commit

---

### WP5: Go Integration

**Building Block Change:** Adds StepML LatencyModel
**Extension Type:** Policy template (new algorithm behind existing LatencyModel interface)
**Motivation:** Integrate the winning research model into BLIS as a third latency model backend.

**Scope:**
- In: Winning model's artifacts, coefficients/weights, and Go integration path (from WP4)
- Out: Full BLIS validation (that's WP6)

**Behavioral Guarantees:**
- BC-5-1: StepML implements all 5 LatencyModel methods
- BC-5-2: Factory dispatch: roofline → stepml (if artifact exists) → blackbox (fallback)
- BC-5-3: INV-M-1 through INV-M-6 all pass
- BC-5-4: All existing tests pass (roofline and blackbox unchanged)
- BC-5-5: New `--stepml-model-path` CLI flag, validated per R3/R10/R18
- BC-5-6: **Roofline isolation regression test** — same `--roofline` simulation produces byte-identical stdout before and after StepML integration (INV-6)

**Risks:**
1. Model too complex for Go → use ONNX runtime (adds dependency) or coefficient export
2. Prediction latency > 1ms → benchmark test fails; simplify model or use approximate inference

**Cross-Cutting:** Updates CLAUDE.md (file organization, latency estimation, CLI flags). Creates `sim/latency/stepml.go` and `sim/latency/stepml_test.go`.

**Validation Gate:** None (all risk gates passed before WP5 start).

**Tasks:**

#### Task 1: Write failing Go tests

Create `sim/latency/stepml_test.go`:
- INV-M-1: Positive output for all non-empty batches; 0 for empty batch
- INV-M-2: Deterministic (same input → same output)
- INV-M-3: Side-effect-free (Request state unchanged after call)
- INV-M-4: Benchmark test: prediction < 1ms at batch size 128
- INV-M-5: Monotonicity check (increasing tokens → non-decreasing time)
- INV-M-6: |MSPE| < 5% on validation data
- Degenerate inputs from design doc table (R20)

Run: `go test ./sim/latency/... -run TestStepML -v`
Expected: FAIL (type not found)

#### Task 2: Implement StepML LatencyModel

Create `sim/latency/stepml.go`:
- Implementation depends on winning model's integration path:
  - **Coefficient export:** Go struct with weight arrays, prediction formula
  - **ONNX:** Load ONNX model, run inference via Go ONNX runtime
  - **Evolved code:** Translate Python prediction function to Go directly
- All 5 methods implemented

Run: `go test ./sim/latency/... -run TestStepML -v`
Expected: PASS

#### Task 3: Extend factory dispatch

Modify `sim/latency/latency.go`:
- Add StepML config fields to `sim.ModelHardwareConfig` (artifact path)
- Three-way dispatch: roofline → stepml (if artifact path set and file exists) → blackbox
- Explicit logged fallback (not silent per R1)

#### Task 4: Add CLI flag

Modify `cmd/root.go`:
- Add `--stepml-model-path` flag (string, default empty)
- Validate per R3 (file exists if set), R10 (strict parsing), R18 (CLI precedence)

#### Task 5: Run all tests

```bash
go test ./... && golangci-lint run ./...
```
Expected: All pass, no lint errors.

#### Task 6: Commit

---

### WP6: BLIS Validation & Promotion

**Building Block Change:** Validates StepML end-to-end in BLIS; promotes to default
**Extension Type:** Tier composition (validation wrapper over existing simulation)
**Motivation:** Verify that per-step prediction errors don't compound in full BLIS simulation.

**Scope:**
- In: StepML LatencyModel from WP5, standard workload scenarios
- Out: Updated golden dataset, CLAUDE.md updates

**Behavioral Guarantees:**
- BC-6-1: Workload-level E2E mean error < 10% on standard workload scenarios (Stage 2 validation)
- BC-6-2: TTFT mean error < 15%, ITL mean error < 15%
- BC-6-3: No P99 ranking inversions vs. blackbox baseline
- BC-6-4: Throughput prediction error < 15%
- BC-6-5: Golden dataset regenerated and documented (R12)
- BC-6-6: INV-6 (determinism) verified: two runs with same seed produce byte-identical stdout

**Risks:**
1. Per-step errors compound in cluster mode → run cluster-level validation; may need bias correction
2. Golden dataset changes break downstream consumers → document changes, regenerate with verification

**Cross-Cutting:** Updates CLAUDE.md (implementation focus, defaults). Regenerates `testdata/goldendataset.json`.

**Validation Gate:** Stage 2 integration gate (design doc) must pass before StepML becomes default.

**Tasks:**

#### Task 1: Run BLIS calibration with StepML

Use BLIS calibration infrastructure (`sim/workload/calibrate.go`) to generate calibration pairs for TTFT, ITL, and E2E with StepML vs. blackbox.

#### Task 2: Run cluster-level validation

Multi-instance BLIS runs with StepML. Verify per-step errors don't interact with routing/scheduling.

#### Task 3: Regenerate golden dataset

```bash
go test ./... -update-golden
```
Document changes. Verify with `go test ./...`.

#### Task 4: Update CLAUDE.md

- File organization: add `sim/latency/stepml.go`
- Latency estimation: document three-way dispatch
- CLI: document `--stepml-model-path`
- Implementation focus: update active development section

#### Task 5: Final commit

---

## J) Dependency DAG

```
WP0 (Shared Infrastructure)
 │
 ├── R1, R2, R4 validation gates
 │
 ▼
WP1 (Research Ideation)
 │
 ▼
WP2 (Hypothesis Scaffolding)
 │
 ├─────────┬─────────┐
 ▼         ▼         ▼
WP3-a    WP3-b    WP3-c    (parallel across ideas; sequential h1→h2→h3 within)
 │         │         │
 └─────────┴─────────┘
           │
           ├── R3, R5 validation gates
           ▼
         WP4 (Leaderboard)
           │
           ▼
         WP5 (Go Integration)
           │
           ▼
         WP6 (Validation & Promotion)
```

**Parallelizable workstreams:**
- WP3-a, WP3-b, WP3-c are fully parallel (independent ideas, shared read-only data)
- Within WP3-x: h1→h2→h3 are sequential (dependency chain)
- WP5 and WP6 are strictly sequential (integration before validation)

**Merge sequencing:**
- WP0 must complete and pass R1/R2/R4 before WP1 starts
- WP2 must complete before any WP3 starts
- All WP3 must complete before WP4
- WP4 must confirm R3/R5 before WP5
- WP5 must pass all tests before WP6

---

## K) Design Bug Prevention Checklist

### General

- [ ] **Scaffolding creep:** Every Python module in WP0 has tests. Every Go type in WP5 has tests. No dead code.
- [ ] **Documentation drift:** CLAUDE.md updated in WP5 (Go changes) and WP6 (golden dataset). Not deferred.
- [ ] **Test infrastructure duplication:** All experiments use `hypotheses/h-stepml/shared/` (BC-3-1, BC-3-2). No per-idea custom parsers.
- [ ] **Golden dataset staleness:** WP6 regenerates golden dataset with verification (R12).
- [ ] **Interface over-specification:** LatencyModel interface is already frozen with 2 existing implementations. StepML is the third — no over-specification risk.

### DES-Specific

- [ ] **Type catalog trap:** No Go struct definitions in this macro plan (Section F describes evolution behaviorally).
- [ ] **Fidelity for its own sake:** Every modeled component traces to AQ-1 (E2E mean fidelity). Temporal dynamics, kernel-level scheduling, and quantization are deliberately omitted/simplified.
- [ ] **Golden without invariant:** WP5 includes companion invariant tests (INV-M-1 through INV-M-6) for every golden test.
- [ ] **Mixing exogenous and endogenous:** The StepML model is a pure query (no event generation, no state mutation). Exogenous inputs (batch composition) are cleanly separated from endogenous simulation logic.

### Module Architecture

- [ ] **Shotgun surgery:** StepML constructed in exactly one place (factory in `sim/latency/latency.go`). No multiple construction sites.
- [ ] **Config mixing:** StepML config fields are module-scoped (added to `ModelHardwareConfig`, not a new top-level struct). Follows R16.
- [ ] **No silent data loss (R1):** Factory fallback from StepML to blackbox is logged, not silent.
- [ ] **Guard division (R11):** Any division in StepML prediction code checks for zero denominator.
- [ ] **Strict YAML parsing (R10):** Any new YAML fields use `yaml.KnownFields(true)`.
- [ ] **CLI flag precedence (R18):** `--stepml-model-path` does not silently override defaults.yaml.

### Roofline Isolation

- [ ] **Roofline byte-identical output:** WP5 includes regression test asserting `--roofline` output is unchanged (BC-5-6).
- [ ] **No roofline code touched:** StepML branch inserted after `if hw.Roofline` check — roofline path returns before StepML dispatch is reached.
- [ ] **No alpha coefficient sharing:** StepML uses its own artifact; roofline alpha coefficients in defaults.yaml are untouched.
- [ ] **No defaults.yaml roofline changes:** StepML config is an additive section; existing roofline/blackbox defaults preserved.

### Research-Specific

- [ ] **Reproducibility:** All random seeds fixed. Split indices saved. Python dependencies pinned. Trained model artifacts saved.
- [ ] **Baseline fairness:** Blackbox baseline re-trained on same training split as research models (BC-0-4).
- [ ] **Data leakage prevention:** Temporal splits verified (BC-0-2). Test set never seen during training.
- [ ] **Short-circuit discipline:** 30% MAPE threshold validated against blackbox baseline in WP0 before any idea uses it.
