# Latency Model Fidelity Research: Macro-Level Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve <10% BLIS E2E mean error by improving any or all of the 5 LatencyModel methods, then integrate the winner into BLIS as a third LatencyModel backend.

**Architecture:** Python research pipeline (data loading → ideation → parallel experimentation → BLIS validation → leaderboard) with lightweight Go integration during research for BLIS validation runs, followed by production-quality Go integration.

**Tech Stack:** Python (scikit-learn, XGBoost, pandas, scipy), Go (sim/latency/ package), BLIS calibration infrastructure (sim/workload/calibrate.go)

**Date:** 2026-02-26
**Status:** Draft
**Based on:** [Design Document](2026-02-26-stepml-research-design.md)

---

## A) Executive Summary

The blackbox latency model uses 2 features / 3 beta coefficients for step time and returns 0 for scheduling and preemption overhead. This plan replaces it with a data-driven model covering all 5 LatencyModel methods, trained on ~165K step-level observations plus per-request lifecycle data from instrumented vLLM. The roofline model is unaffected.

**What changes:** A new StepML LatencyModel backend joins blackbox and roofline. Ideas may improve any combination of the 5 methods — step time, queueing time, output token processing, scheduling overhead, preemption cost.

**Primary metric:** BLIS E2E mean error < 10% per experiment, measured via BLIS simulation runs (not per-step MAPE).

**Implementation:** 7 ordered work packages (WP0–WP6):

- **WP0** — Shared infrastructure: data pipeline, BLIS validation harness, Go tree evaluator, blackbox E2E baseline, MFU benchmark access, per-request KV length extraction, component-level error attribution
- **WP1** — Research ideation via `/research-ideas` (3+ ideas targeting any LatencyModel methods)
- **WP2** — Hypothesis scaffolding (idea-specific sub-hypothesis decomposition — not fixed h1/h2/h3)
- **WP3** — Wave-parallel experimentation with BLIS validation runs required per idea
- **WP4** — Leaderboard comparison ranked by BLIS E2E mean error
- **WP5** — Production Go integration: StepML LatencyModel in `sim/latency/`
- **WP6** — Production BLIS validation, golden dataset update, CLAUDE.md update

**Key constraints:** Frozen LatencyModel interface (`sim/latency_model.go:7-23`). Data integrity: train/valid/test must be strictly non-overlapping; model selection must not use BLIS E2E on training experiments. Ideas define their own splits and training strategies.

---

## B) Repository Recon Summary

### B.1 Affected Packages and Files

| Package | Files Affected | Change Type |
|---|---|---|
| `hypotheses/h-stepml/` | New directory tree | New — Python research artifacts + BLIS validation harness |
| `sim/latency/` | New `stepml.go` (lightweight in WP0, production in WP5) + modified `latency.go` factory | New + Moderate |
| `sim/` | `latency_model.go` unchanged; `config.go` minor extension | Minor — add StepML artifact path to config |
| `cmd/` | `root.go` | Minor — new CLI flag for StepML artifact path |
| `defaults.yaml` | Minor | Add StepML default config section |
| `testdata/` | `goldendataset.json` | Regenerate (WP6) |

### B.2 Core Data Structures (Frozen — facts from merged code)

- **LatencyModel** (`sim/latency_model.go:7-23`) — 5-method interface. Frozen.
- **NewLatencyModelFunc** (`sim/latency_model.go:31`) — Registration variable. Factory at `sim/latency/latency.go:129-167` dispatches on `hw.Roofline`.
- **Request.ProgressIndex** (`sim/request.go:34`) — `int64`, cumulative token progress. KV cache length proxy (design doc Gap 1 bridge).
- **BlackboxLatencyModel** (`sim/latency/latency.go:18-58`) — Model being replaced. `betaCoeffs[0..2]` (step time), `alphaCoeffs[0..2]` (queueing + output processing). Returns 0 for scheduling and preemption.
- **RooflineLatencyModel** (`sim/latency/latency.go:64-111`) — Unaffected. Uses `ProgressIndex` and `NumNewTokens`.
- **CalibrationReport** (`sim/workload/calibrate.go`) — `PrepareCalibrationPairs()` + `ComputeCalibration()` produce E2E/TTFT/ITL comparison metrics.
- **TraceV2 replay** (`sim/workload/replay.go`) — `LoadTraceV2Requests()` converts ground-truth traces into sim.Request for replay.

### B.3 Call Sites (Confirmed)

| Method | Call Site | Current Value |
|---|---|---|
| `StepTime` | `sim/simulator.go:415` | `beta0 + beta1*prefill + beta2*decode` |
| `QueueingTime` | `sim/event.go:31` | `alpha0 + alpha1*inputLen` |
| `OutputTokenProcessingTime` | `sim/simulator.go:433,437,466` | constant `alpha2` |
| `SchedulingProcessingTime` | `sim/batch_formation.go:131` | **returns 0** |
| `PreemptionProcessingTime` | `sim/batch_formation.go:160` | **returns 0** |

### B.4 Additional Data Sources

- **MFU benchmarks** (`InferSim/bench_data/`): Kernel-level GEMM and attention MFU data by GPU/shape/config. Available for physics-informed approaches.
- **KV events** (`kv_events.jsonl`): Block-level storage and transfer events.
- **Per-request lifecycle** (`per_request_lifecycle_metrics.json`): Per-token timestamps, input/output counts.

---

## C) High-Level Objectives + Non-Goals + Model Scoping

### Objectives

1. **Build shared infrastructure** for data loading, BLIS validation, and baseline establishment across all research ideas
2. **Build a lightweight Go tree evaluator** during Phase 0 to enable BLIS validation runs during research (not deferred to post-research)
3. **Generate 3+ research ideas** targeting any combination of the 5 LatencyModel methods — not limited to step-time-only approaches
4. **Execute parallel experiments** with idea-specific sub-hypothesis decomposition, training strategies, and data splits — subject to mandatory data integrity requirements
5. **Select a winner** achieving <10% BLIS E2E mean error, measured by BLIS simulation runs on all 16 experiments
6. **Integrate into BLIS** as a third LatencyModel implementation via the policy template extension recipe
7. **Validate in production** using BLIS calibration infrastructure on standard workload scenarios

### Non-Goals

- Modifying the roofline latency model — roofline must produce **byte-identical output** before and after this work
- Multi-instance cluster-level effects (latency model is instance-local)
- Non-H100 hardware training (evaluation dimension only)
- vLLM versions other than v0.15.1 (evaluation dimension only)
- Prescribing a specific training methodology or data split — ideas choose their own approach
- Automated retraining pipeline (future work)

### Model Scoping Table

| Component | Modeled | Simplified | Omitted | Justification |
|-----------|---------|------------|---------|---------------|
| Batch composition (prefill/decode tokens, request counts) | Yes | — | — | Direct causal inputs to step FLOPs |
| Per-request KV cache lengths | Yes (via ProgressIndex proxy + lifecycle extraction) | — | — | H8: 12.96× overestimate without per-request KV |
| All 5 LatencyModel methods | Yes (any may be improved) | — | — | E2E latency = sum of all 5 method contributions + emergent queueing |
| Model architecture (dense vs. MoE) | Yes (experiment metadata) | — | — | Fundamentally different compute patterns |
| MFU benchmarks | Yes (available from InferSim bench_data) | — | — | Enables physics-informed approaches grounding predictions in measured hardware performance |
| Hardware characteristics | — | H100-only training | — | Single GPU generation; cross-hardware is P5 |
| vLLM scheduler internals | — | Observed via batch composition + timing data | — | Scheduling/preemption overhead measurable from traces |
| Kernel-level scheduling | — | — | Yes | Below per-step abstraction; captured in target |
| Temporal dynamics | — | Independent prediction | — | Simplest approach answering AQ-1 |
| Quantization effects | — | BF16-only | — | P6 lowest priority |

---

## D) Concept Model

### Building Blocks (5)

**1. Data Pipeline** — Loads ground-truth data at multiple granularities; provides MFU benchmarks.
- OBSERVES: Raw step traces, per-request lifecycle data, KV events, InferSim bench_data MFU benchmarks
- CONTROLS: Parsed datasets at step-level, request-level, and experiment-level granularities
- OWNS: Parsed feature matrices, experiment metadata catalog
- INVARIANTS: Row counts match source; all 16 experiments present; MFU benchmarks indexed by GPU+shape
- EVENTS: None (offline)
- EXTENSION FRICTION: 1 file per new data source

**2. BLIS Validation Harness** — Runs BLIS with candidate models on ground-truth traces; produces E2E/TTFT/ITL metrics.
- OBSERVES: Exported model artifacts (coefficients or tree JSON), ground-truth trace v2 files
- CONTROLS: Per-experiment E2E mean error, TTFT mean error, ITL mean error (the P1/P2 metrics)
- OWNS: Validation scripts, summary CSV output
- INVARIANTS: Uses existing calibrate.go infrastructure; identical metric computation as production BLIS
- EVENTS: None (offline orchestration)
- EXTENSION FRICTION: 0 files (accepts any LatencyModel via exported artifacts)

**3. Experiment Engine** — Per-idea hypothesis chain with idea-specific decomposition.
- OBSERVES: Datasets (from Data Pipeline), BLIS metrics (from Validation Harness)
- CONTROLS: Model training, feature extraction, BLIS E2E validation, short-circuit decisions
- OWNS: Per-idea model artifacts, FINDINGS.md, idea-specific splits and training strategy
- INVARIANTS: Data integrity requirements (no leakage, non-overlapping splits); BLIS E2E reported for every idea
- EVENTS: None (offline)
- EXTENSION FRICTION: 2-3 directories per new idea

**4. StepML LatencyModel** (Go) — Third LatencyModel behind the frozen interface.
- OBSERVES: Batch of `*Request` (InputTokens, OutputTokens, ProgressIndex, NumNewTokens)
- CONTROLS: 5 latency estimates (step time, queueing, output processing, scheduling, preemption)
- OWNS: Trained model weights/coefficients (immutable after construction)
- INVARIANTS: INV-M-1 (positive), INV-M-2 (deterministic), INV-M-3 (pure), INV-M-4 (<1ms), INV-M-5 (monotonic), INV-M-6 (|MSPE|<5%)
- EVENTS: None (synchronous query)
- EXTENSION FRICTION: ~3-4 files (impl, factory, config, CLI flag)

**5. Leaderboard** — Cross-idea comparison ranked by BLIS E2E mean error.
- OBSERVES: All ideas' BLIS validation results and FINDINGS.md
- CONTROLS: Winner selection, final README.md
- OWNS: Leaderboard table
- INVARIANTS: Rankings use P1 (BLIS E2E) first; data integrity documented per idea
- EVENTS: None
- EXTENSION FRICTION: 1 row per new idea

### Interaction Model

```
[Data Pipeline] ──read-only──→ [Experiment Engine (per idea, parallel)]
                                      │
[BLIS Validation Harness] ◄──artifacts──┘
       │                              │
       └──BLIS E2E metrics──────────→ │
                                      │
[Leaderboard] ◄─────results──────────┘
       │
       ▼ (winner's artifacts)
[StepML LatencyModel (Go)]
```

### Real-System Correspondence

| Building Block | vLLM v0.15.1 | BLIS |
|---|---|---|
| Data Pipeline | step traces + lifecycle + KV events | `hypotheses/h-stepml/shared/` |
| BLIS Validation Harness | Offline comparison | `validate_blis.sh` + `sim/workload/calibrate.go` |
| StepML LatencyModel | Wall-clock model_execute() + scheduler overhead | `sim/latency/stepml.go` (new) |

---

## E) Architectural Risk Register

| # | Decision | Assumption | Validation Method | Cost if Wrong | Gate |
|---|----------|------------|-------------------|---------------|------|
| R1 | ProgressIndex as KV length proxy | ProgressIndex ≈ total KV cache length per request | Compare ProgressIndex with ground-truth per-request KV from lifecycle data in WP0 | WP3–WP6 rework | Before WP3 |
| R2 | 10% sampling sufficient | Sampling is approximately random | Characterize sampling distribution in WP0 | WP3–WP4 rework | Before WP3 |
| R3 | Request-batch features suffice | Per-step prediction from Request fields achieves <10% E2E | Best model on Request-only features must pass BLIS E2E <15% on majority of experiments | WP5–WP6 rework (interface extension) | Before WP5 |
| R4 | Blackbox is beatable | Current blackbox has significant BLIS E2E error | Compute blackbox BLIS E2E baseline in WP0. If E2E mean < 12%, reassess | All WP1–WP6 wasted | Before WP1 |
| R5 | Winning model portable to Go | Model expressible as coefficients, ONNX, or compact code | Each idea documents Go integration path during WP3 | WP5 rework | Before WP5 |
| R6 | Lightweight Go evaluator is accurate | Research evaluator matches production behavior | Compare lightweight and production Go implementations on same inputs in WP5 | WP3 results invalid | Before WP5 merge |

**Abort plans:**
- R1 fails → derive per-request KV lengths from lifecycle data instead; may need richer feature extraction
- R2 fails → request full-trace re-collection or apply importance weighting
- R3 fails → propose LatencyModel interface extension (separate design doc)
- R4 fails → cancel research; blackbox is already good enough
- R5 fails → use ONNX runtime from Go (adds dependency)
- R6 fails → rerun BLIS validation with production implementation; update leaderboard

---

## F) Architectural Evolution

### Current State

Two LatencyModel backends (`sim/latency/latency.go:129-167`):
- `hw.Roofline == true` → RooflineLatencyModel (analytical FLOPs/bandwidth)
- `hw.Roofline == false` → BlackboxLatencyModel (2-feature regression; returns 0 for scheduling/preemption)

Factory dispatch is binary. No fallback chain. No data-driven multi-method calibration.

### Target State

Three LatencyModel backends with ordered dispatch:
1. `hw.Roofline == true` → RooflineLatencyModel (unchanged)
2. StepML artifact exists at configured path → StepMLLatencyModel (new — all 5 methods data-driven)
3. Fallback → BlackboxLatencyModel (preserved for backward compatibility)

New CLI flag `--stepml-model <path>` specifies the artifact location. When absent, falls back to blackbox.

### Evolution Path

1. **WP0 (lightweight):** Add `sim/latency/stepml.go` with minimal tree evaluator + coefficient loader. Registers via existing factory. Enables BLIS validation during research. Not production quality.
2. **WP5 (production):** Replace lightweight implementation with production-quality code — proper error handling, comprehensive tests (INV-M-1 through INV-M-6), R20 degenerate input handling.

### What Remains Unchanged

- LatencyModel interface (frozen at `sim/latency_model.go:7-23`)
- **RooflineLatencyModel: zero code changes.** The `--roofline` path returns before StepML dispatch.
- All call sites (`sim/simulator.go`, `sim/event.go`, `sim/batch_formation.go`)
- Batch formation, scheduling, routing, KV cache modules
- Cluster-level configuration

### Roofline Isolation Guarantee

StepML findings for scheduling overhead, preemption, and queueing apply ONLY to the StepML LatencyModel. Not propagated to roofline because:
1. Factory dispatch order: `hw.Roofline == true` checked first — StepML unreachable when roofline active
2. No shared mutable state: separate artifacts and coefficients
3. No defaults.yaml changes for roofline
4. Verification: regression test asserting byte-identical `--roofline` stdout (INV-6)

---

## G) Frozen Interface Reference

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

- **Existing:** `hypotheses/lib/harness.sh`, `hypotheses/lib/analyze_helpers.py`
- **New (WP0):** `hypotheses/h-stepml/shared/` — data loader, evaluation harness, BLIS validation harness, baselines, feature extraction tools. All ideas use these.
- **New (WP0):** `sim/latency/stepml.go` — lightweight Go tree evaluator for BLIS validation during research
- **New (WP5):** `sim/latency/stepml_test.go` — production Go unit tests with INV-M-1 through INV-M-6

### H.2 Documentation Maintenance

| Trigger | Owner | What Updates |
|---|---|---|
| WP0 creates `hypotheses/h-stepml/` and `sim/latency/stepml.go` | WP0 | CLAUDE.md file organization + latency estimation section |
| WP0 adds `--stepml-model` flag | WP0 | CLAUDE.md CLI flags |
| WP5 replaces lightweight with production implementation | WP5 | CLAUDE.md latency estimation section |
| WP6 regenerates golden dataset | WP6 | CLAUDE.md golden dataset note |

### H.3 CI Pipeline Changes

- **WP0:** New Go file added to `go test ./sim/latency/...`. Lightweight tests only.
- **WP0–WP4:** Python research code not CI-gated; reproducibility via run.sh
- **WP5:** Production Go tests + benchmarks
- **WP6:** Golden dataset regeneration + `go test ./...` verification

### H.4 Dependency Management

- **Python (WP0–WP4):** `requirements.txt` in `hypotheses/h-stepml/shared/` with pinned versions
- **Go (WP0):** Zero new dependencies for lightweight evaluator (pure Go tree traversal + JSON parsing)
- **Go (WP5):** Depends on winning model. Coefficient export = zero new deps. ONNX = adds `onnxruntime-go`.

### H.5 Interface Freeze Schedule

- **LatencyModel interface:** Already frozen. No changes.
- **NewLatencyModelFunc signature:** Already frozen. StepML uses existing `ModelHardwareConfig` with optional new fields.
- **No new interfaces introduced.** StepML is a policy template.

---

## I) Work Package Plan

### WP0: Shared Infrastructure + BLIS Validation Loop

**Building Block Change:** Adds Data Pipeline + BLIS Validation Harness + lightweight StepML LatencyModel
**Extension Type:** Subsystem module (new infrastructure) + policy template (lightweight Go evaluator)
**Motivation:** All subsequent WPs depend on data loading, BLIS validation capability, and the blackbox E2E baseline. The BLIS validation harness is the primary evaluation tool — without it, ideas cannot measure the P1 metric.

**Scope:**
- In: Data parsing (step-level + request-level + MFU benchmarks from InferSim bench_data), Python evaluation harness, Go tree evaluator, BLIS validation harness, blackbox E2E baseline, component-level error attribution, per-request KV length extraction, sampling bias characterization, prefix cache semantics (D-8)
- Out: Any model training, any idea-specific code, production-quality Go implementation

**Behavioral Guarantees:**
- BC-0-1: Row count of parsed datasets matches source file counts
- BC-0-2: MFU benchmark data indexed and queryable by GPU type, matrix shape, attention config
- BC-0-3: Go tree evaluator implements all 5 LatencyModel methods and loads exported artifacts
- BC-0-4: BLIS validation harness runs all 16 experiments and outputs E2E/TTFT/ITL mean errors
- BC-0-5: Blackbox E2E baseline established (the number Round 2 must beat)
- BC-0-6: Component-level error attribution identifies which of the 5 methods dominates E2E error
- BC-0-7: Sampling bias report documents step_id uniformity, autocorrelation, per-experiment representation

**Risks:**
1. Sampling is systematically biased → characterization detects it; abort per R2
2. Blackbox already achieves <12% E2E mean → reassess research justification per R4
3. Go tree evaluator has accuracy issues → validated against Python predictions before WP3

**Cross-Cutting:** Creates `hypotheses/h-stepml/shared/` and `sim/latency/stepml.go`. Updates CLAUDE.md.

**Validation Gate:** R1, R2, R4 must pass before WP1.

---

**Implementation Guide:**

**Architectural Impact:** Adds `sim/latency/stepml.go` as a third LatencyModel backend (lightweight, research-only). Adds Python infrastructure under `hypotheses/h-stepml/shared/`.

**Tasks (grouped by subsystem):**

*Data loading and characterization:*
1. Create directory structure and requirements.txt
2. Implement data loader: parse step-level traces from all 20 experiments into unified Parquet
3. Implement request-level data loader: parse per-request lifecycle metrics
4. Index MFU benchmarks from InferSim bench_data (make queryable by GPU+shape+config)
5. Characterize sampling distribution (bias detection per R2 gate)
6. Verify step.duration_us semantics + resolve prefix cache semantics (D-8)

*Python evaluation (diagnostics):*
7. Implement per-step evaluation harness (MAPE, MSPE, Pearson r, p99 error, bootstrap CI)
8. Implement per-step baselines (blackbox re-trained, roofline, naive mean)

*Feature extraction tools:*
9. Component-level error attribution tool (e2e_decomposition.py): ablate each of 5 methods to measure marginal E2E error contribution
10. Per-request KV length extractor (lifecycle_kv_extractor.py): derive per-step kv_mean, kv_max, kv_sum from lifecycle data

*BLIS validation infrastructure:*
11. Go tree evaluator (sim/latency/stepml.go): lightweight LatencyModel loading exported artifacts, ~200 lines
12. BLIS validation harness (validate_blis.sh): run BLIS on 16 traces, output E2E/TTFT/ITL per experiment
13. Blackbox E2E baseline: run validation harness with current blackbox coefficients — establish the baseline
14. Validate ProgressIndex as KV proxy (R1 gate)

---

### WP1: Research Ideation

**Building Block Change:** Populates Experiment Engine inputs
**Extension Type:** Process (skill invocation)
**Motivation:** Generate 3+ literature-grounded ideas for achieving <10% BLIS E2E mean error via any combination of LatencyModel method improvements.

**Scope:**
- In: Write `problem.md` from design doc (including WP0 baseline results + component error attribution), invoke `/research-ideas`
- Out: Model training, experiment execution

**Behavioral Guarantees:**
- BC-1-1: `research.md` contains 3+ ideas with literature citations
- BC-1-2: Each idea specifies which of the 5 LatencyModel methods it targets
- BC-1-3: Each idea documents its Go integration path
- BC-1-4: Ideas are diverse — not all limited to step-time-only ML models

**Risks:**
1. Ideas are too similar → problem.md explicitly requests diverse approaches (e.g., step-time ML, end-to-end calibration, multi-component optimization)
2. Ideas ignore the non-StepTime methods → problem.md includes WP0 component error attribution showing which methods contribute most to E2E error

**Cross-Cutting:** None. Produces `research.md` consumed by WP2.

---

### WP2: Hypothesis Scaffolding

**Building Block Change:** Adds Experiment Engine structure
**Extension Type:** Process (directory creation + HYPOTHESIS.md authoring)
**Motivation:** Map research ideas to testable hypotheses with idea-specific sub-hypothesis decomposition.

**Scope:**
- In: Extract top 3-5 ideas from research.md, create per-idea directories, write HYPOTHESIS.md files
- Out: Experiment code, running experiments

**Behavioral Guarantees:**
- BC-2-1: Each idea has 2-3 sub-hypotheses with idea-specific decomposition (NOT fixed h1-features/h2-model/h3-generalization)
- BC-2-2: Each HYPOTHESIS.md includes Related Work with citations from research.md
- BC-2-3: At least one sub-hypothesis per idea reports BLIS E2E mean error
- BC-2-4: Claims and refutation criteria expressed in terms of BLIS E2E mean error where possible
- BC-2-5: Each idea documents its training strategy and data split approach (subject to data integrity requirements)

**Risks:**
1. Ideas don't map to hypotheses → each idea must specify a falsifiable BLIS E2E claim

**Cross-Cutting:** Creates directory structure consumed by WP3.

**Validation Gate:** Convergence review (h-design, 5 perspectives) on each idea's HYPOTHESIS.md.

---

### WP3: Wave-Parallel Experimentation

**Building Block Change:** Executes Experiment Engine
**Extension Type:** Policy template (new algorithms per idea)
**Motivation:** Test each idea's approach and measure BLIS E2E mean error.

**Scope:**
- In: All sub-hypotheses per idea, using WP0 shared infrastructure
- Out: Production Go integration (that's WP5)

**Behavioral Guarantees:**
- BC-3-1: Every idea uses WP0's data loader (no custom parsing)
- BC-3-2: Every idea's final sub-hypothesis reports BLIS E2E mean error via validate_blis.sh
- BC-3-3: Every FINDINGS.md includes BLIS E2E per experiment + data integrity documentation (which data was training vs. evaluation)
- BC-3-4: Data integrity requirements satisfied: non-overlapping splits, no feature leakage, no model selection on training experiments' BLIS E2E
- BC-3-5: Ideas with BLIS E2E worse than blackbox baseline are dropped
- BC-3-6: Each idea's training strategy and split documented and justified

**Risks:**
1. All ideas fail → re-examine feature assumptions; revisit R1, R3
2. BLIS validation harness bottleneck → each BLIS run takes ~2-5 min; 16 experiments × multiple ideas is manageable

**Cross-Cutting:** Consumes WP0 shared infrastructure. Each idea writes only to its own directory.

**Validation Gate:** R3 evaluated after WP3. R5 confirmed for surviving ideas.

---

### WP4: Leaderboard & Selection

**Building Block Change:** Populates Leaderboard
**Extension Type:** Process (analysis + documentation)
**Motivation:** Compare all ideas on BLIS E2E mean error and select the winner.

**Scope:**
- In: All ideas' BLIS validation results and FINDINGS.md
- Out: Winner selection, README.md with leaderboard

**Behavioral Guarantees:**
- BC-4-1: Rankings ordered by P1 metric (BLIS E2E mean error) first
- BC-4-2: Data integrity verified: each idea's training/evaluation boundary documented
- BC-4-3: Winner identified with Go integration path documented
- BC-4-4: Falsification criteria checked (design doc section)

**Risks:**
1. No idea achieves <10% BLIS E2E → evaluate falsification criteria; consider abandoning StepML

**Cross-Cutting:** Produces README.md. Inputs to WP5.

---

### WP5: Production Go Integration

**Building Block Change:** Replaces lightweight StepML with production implementation
**Extension Type:** Policy template (production-quality algorithm behind LatencyModel interface)
**Motivation:** Replace WP0's lightweight Go evaluator with production-quality code.

**Scope:**
- In: Winning model's artifacts and integration path from WP4
- Out: Production BLIS validation (that's WP6)

**Behavioral Guarantees:**
- BC-5-1: StepML implements all 5 LatencyModel methods (production quality)
- BC-5-2: Factory dispatch: roofline → stepml (if artifact) → blackbox (fallback)
- BC-5-3: INV-M-1 through INV-M-6 all pass with comprehensive tests
- BC-5-4: All existing tests pass (roofline and blackbox unchanged)
- BC-5-5: CLI flag validated per R3/R10/R18
- BC-5-6: Roofline isolation regression test — byte-identical stdout (INV-6)
- BC-5-7: R20 degenerate inputs handled (all cases from design doc table)
- BC-5-8: Production predictions match lightweight evaluator results on same inputs (R6 validation)

**Risks:**
1. Model too complex for Go → ONNX runtime or coefficient export
2. Prediction latency > 1ms → simplify or approximate

**Cross-Cutting:** Updates CLAUDE.md. Creates production `sim/latency/stepml.go` and `sim/latency/stepml_test.go`.

---

### WP6: Production Validation & Promotion

**Building Block Change:** Validates StepML end-to-end; promotes to default
**Extension Type:** Tier composition (validation over existing simulation)
**Motivation:** Verify production Go implementation matches research results and passes Stage 2 integration gate.

**Scope:**
- In: Production StepML from WP5, standard workload scenarios
- Out: Updated golden dataset, CLAUDE.md

**Behavioral Guarantees:**
- BC-6-1: BLIS E2E mean error < 10% on standard workloads (production Go, not lightweight evaluator)
- BC-6-2: TTFT mean error < 15%, ITL mean error < 15%
- BC-6-3: No P99 ranking inversions vs. blackbox
- BC-6-4: Throughput prediction error < 15%
- BC-6-5: Golden dataset regenerated and documented (R12)
- BC-6-6: INV-6 determinism: two runs with same seed produce byte-identical stdout

**Risks:**
1. Production implementation diverges from lightweight evaluator → R6 gate catches this
2. Golden dataset changes break tests → regenerate with verification

**Cross-Cutting:** Updates CLAUDE.md (defaults, implementation focus). Regenerates `testdata/goldendataset.json`.

---

## J) Dependency DAG

```
WP0 (Shared Infrastructure + BLIS Validation Loop + Go Evaluator)
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
WP3-a    WP3-b    WP3-c    (parallel across ideas; sequential sub-hypotheses within)
 │         │         │
 └─────────┴─────────┘
           │
           ├── R3, R5 validation gates
           ▼
         WP4 (Leaderboard — ranked by BLIS E2E)
           │
           ▼
         WP5 (Production Go Integration)
           │
           ├── R6 validation gate
           ▼
         WP6 (Production Validation & Promotion)
```

**Parallelizable workstreams:**
- WP3-a, WP3-b, WP3-c are fully parallel (independent ideas, shared read-only data)
- Within WP3-x: sub-hypotheses are sequential (idea-specific dependency chain)
- WP5 and WP6 are strictly sequential

**Merge sequencing:**
- WP0 must complete and pass R1/R2/R4 before WP1
- WP2 must complete before any WP3
- All WP3 must complete before WP4
- WP4 must confirm R3/R5 before WP5
- WP5 must pass R6 before WP6

---

## K) Design Bug Prevention Checklist

### General

- [ ] **Scaffolding creep:** Every Python module in WP0 has tests. Go evaluator in WP0 has basic tests. No dead code.
- [ ] **Documentation drift:** CLAUDE.md updated in WP0 (Go evaluator, shared infra), WP5 (production), WP6 (golden dataset).
- [ ] **Test infrastructure duplication:** All experiments use `hypotheses/h-stepml/shared/` and `validate_blis.sh`.
- [ ] **Golden dataset staleness:** WP6 regenerates with verification (R12).

### DES-Specific

- [ ] **Type catalog trap:** No Go struct definitions in this macro plan. Evolution described behaviorally.
- [ ] **Fidelity for its own sake:** Every modeled component traces to AQ-1 (E2E mean fidelity).
- [ ] **Golden without invariant:** WP5 includes INV-M-1 through INV-M-6 companion tests.
- [ ] **Mixing exogenous and endogenous:** StepML is a pure query (no events, no state mutation).

### Module Architecture

- [ ] **Shotgun surgery:** StepML constructed in one place (factory in `sim/latency/latency.go`).
- [ ] **Config mixing:** StepML config fields module-scoped (R16).
- [ ] **No silent data loss (R1):** Factory fallback logged, not silent.
- [ ] **Guard division (R11):** Zero denominators checked in prediction code.
- [ ] **Strict YAML parsing (R10):** New fields use `yaml.KnownFields(true)`.
- [ ] **CLI flag precedence (R18):** `--stepml-model` does not silently override defaults.yaml.

### Roofline Isolation

- [ ] **Roofline byte-identical output:** WP5 regression test (BC-5-6).
- [ ] **No roofline code touched:** StepML branch after `if hw.Roofline` check.
- [ ] **No alpha coefficient sharing:** Separate artifacts.
- [ ] **No defaults.yaml roofline changes:** Additive StepML section only.

### Data Integrity (Research-Specific)

- [ ] **No data leakage:** Every idea's train/valid/test sets are strictly non-overlapping (Integrity Req #1).
- [ ] **No evaluation on training data:** Metrics clearly labeled as training vs. validation (Integrity Req #2).
- [ ] **BLIS E2E boundary:** Model selection/tuning not done on BLIS E2E of training experiments (Integrity Req #3).
- [ ] **Temporal leakage awareness:** Ideas using step-level splits document their leakage prevention (Integrity Req #4).
- [ ] **Feature leakage:** No post-execution features used as inputs (Integrity Req #5).
- [ ] **Reproducibility:** Seeds fixed, splits saved, dependencies pinned (Integrity Req #6).
