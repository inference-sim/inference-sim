# Latency Model Fidelity Research: Macro-Level Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve <10% BLIS E2E mean error by improving any or all of the 5 LatencyModel methods, then integrate the winner into BLIS as a third LatencyModel backend.

**Architecture:** Python research pipeline (data loading → ideation → parallel experimentation → BLIS validation → leaderboard) with lightweight Go integration during research for BLIS validation runs.

**Tech Stack:** Python (scikit-learn, XGBoost, pandas, scipy), Go (sim/latency/ package), BLIS calibration infrastructure (sim/workload/calibrate.go)

**Date:** 2026-02-26
**Status:** Draft
**Based on:** [Design Document](2026-02-26-stepml-research-design.md)

---

## A) Executive Summary

The blackbox latency model uses 2 features / 3 beta coefficients for step time and returns 0 for scheduling and preemption overhead. This plan replaces it with a data-driven model covering all 5 LatencyModel methods, trained on ~165K step-level observations plus per-request lifecycle data from instrumented vLLM. The roofline model is unaffected — it serves only as an informational comparison baseline (ideas must not use roofline as a component or feature).

**What changes:** A new StepML LatencyModel backend joins blackbox and roofline. Ideas may improve any combination of the 5 methods — step time, queueing time, output token processing, scheduling overhead, preemption cost.

**Primary metric:** BLIS E2E mean error < 10% per experiment, measured via BLIS simulation runs (not per-step MAPE).

**Implementation:** WP0 (infrastructure, once) + iterative research rounds (WP1–WP4 loop):

- **WP0** — Shared infrastructure: data pipeline, BLIS validation harness, Go tree evaluator, blackbox E2E baseline, MFU benchmark access, per-request KV length extraction, component-level error attribution
- **Research Round Loop (WP1→WP2→WP3→WP4)** — Iterates until <10% E2E target met or max rounds exhausted:
  - **WP1** — Research ideation via `/research-ideas` using updated `problem.md` (incorporates prior round findings)
  - **WP2** — Hypothesis scaffolding (idea-specific sub-hypothesis decomposition)
  - **WP3** — Wave-parallel experimentation with BLIS validation runs required per idea
  - **WP4** — Leaderboard comparison + `/convergence-review` feedback → produces `FINDINGS_ROUND<N>.md` + updated `problem.md`

**Iterative loop protocol:** Each round saves `hypotheses/h-stepml/round<N>/FINDINGS_ROUND<N>.md` capturing what worked, what failed, binding constraints, and open questions. The problem statement (`problem.md`) is updated with new baselines, narrowed scope, and refined questions before the next round begins. A `/convergence-review` (hypothesis FINDINGS gate, 10 perspectives) at the end of each round determines whether to declare convergence (target met), iterate (progress but not converged), or abort (no improvement path).

**Key constraints:** Frozen LatencyModel interface (`sim/latency_model.go:7-23`). Data integrity: train/valid/test must be strictly non-overlapping; model selection must not use BLIS E2E on training experiments. Ideas define their own splits and training strategies. Max rounds: 5 (hard cap to prevent unbounded iteration).

---

## B) Repository Recon Summary

### B.1 Affected Packages and Files

| Package | Files Affected | Change Type |
|---|---|---|
| `hypotheses/h-stepml/` | New directory tree | New — Python research artifacts + BLIS validation harness |
| `sim/latency/` | New `stepml.go` (lightweight in WP0) + modified `latency.go` factory | New + Moderate |
| `sim/` | `latency_model.go` unchanged; `config.go` minor extension | Minor — add StepML artifact path to config |
| `cmd/` | `root.go` | Minor — new CLI flag for StepML artifact path |
| `defaults.yaml` | Minor | Add StepML default config section |
| `testdata/` | `goldendataset.json` | Unchanged (no production integration) |

### B.2 Core Data Structures (Frozen — facts from merged code)

- **LatencyModel** (`sim/latency_model.go:7-23`) — 5-method interface. Frozen.
- **NewLatencyModelFunc** (`sim/latency_model.go:31`) — Registration variable. Factory at `sim/latency/latency.go:129-167` dispatches on `hw.Roofline`.
- **Request.ProgressIndex** (`sim/request.go:34`) — `int64`, cumulative token progress. KV cache length proxy (design doc Gap 1 bridge).
- **BlackboxLatencyModel** (`sim/latency/latency.go:18-58`) — Model being replaced. `betaCoeffs[0..2]` (step time), `alphaCoeffs[0..2]` (queueing + output processing). Returns 0 for scheduling and preemption.
- **RooflineLatencyModel** (`sim/latency/latency.go:64-111`) — Unaffected. Uses `ProgressIndex` and `NumNewTokens`.
- **CalibrationReport** (`sim/workload/calibrate.go`) — `PrepareCalibrationPairs()` + `ComputeCalibration()` produce E2E/TTFT/ITL comparison metrics.
- **Workload generation** (`sim/workload/`) — generates requests from inference-perf workload specs (stages, distributions, shared prefix config). Also supports TraceV2 replay for direct trace-based runs.

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
3. **Iterate through research rounds** (WP1→WP2→WP3→WP4) with updated problem statements incorporating prior findings, convergence review feedback, and progressively refined approaches
4. **Generate 3+ research ideas per round** targeting any combination of the 5 LatencyModel methods — not limited to step-time-only approaches; later rounds build on earlier findings
5. **Execute parallel experiments** with idea-specific sub-hypothesis decomposition, training strategies, and data splits — subject to mandatory data integrity requirements
6. **Converge on a winner** achieving <10% BLIS E2E mean error within ≤5 rounds, measured by BLIS simulation runs on all experiments

### Non-Goals

- Modifying the roofline latency model — roofline must produce **byte-identical output** before and after this work
- Using the roofline model as a component, feature, or building block in any research idea — roofline is a comparison baseline only (ideas may use independent physics-informed features derived from first principles)
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
- INVARIANTS: Row counts match source; all 20 experiments (16 main + 4 sweep) present; MFU benchmarks indexed by GPU+shape
- EVENTS: None (offline)
- EXTENSION FRICTION: 1 file per new data source

**2. BLIS Validation Harness** — Runs BLIS with candidate models using the same inference-perf workload profiles as ground-truth experiments; compares BLIS-produced latencies against observed means to produce E2E/TTFT/ITL error metrics.
- OBSERVES: Exported model artifacts (coefficients or tree JSON), experiment configs and inference-perf profiles
- CONTROLS: Per-experiment E2E mean error, TTFT mean error, ITL mean error (the P1/P2 metrics)
- OWNS: Validation scripts, summary CSV output
- INVARIANTS: Uses existing calibrate.go infrastructure; identical metric computation as production BLIS
- EVENTS: None (offline orchestration)
- EXTENSION FRICTION: 0 files (accepts any LatencyModel via exported artifacts)

**3. Experiment Engine** — Per-idea hypothesis chain with idea-specific decomposition.
- OBSERVES: Datasets (from Data Pipeline), BLIS metrics (from Validation Harness)
- CONTROLS: Model training, feature extraction, BLIS E2E validation, short-circuit decisions
- OWNS: Per-idea model artifacts, per-sub-hypothesis FINDINGS.md, idea-level FINDINGS_SUMMARY.md, idea-specific splits and training strategy
- INVARIANTS: Data integrity requirements (no leakage, non-overlapping splits); BLIS E2E reported for every idea
- EVENTS: None (offline)
- EXTENSION FRICTION: 2-3 directories per new idea

**4. StepML LatencyModel** (Go, lightweight) — Lightweight third LatencyModel behind the frozen interface, used for BLIS validation during research.
- OBSERVES: Batch of `*Request` (InputTokens, OutputTokens, ProgressIndex, NumNewTokens)
- CONTROLS: 5 latency estimates (step time, queueing, output processing, scheduling, preemption)
- OWNS: Trained model weights/coefficients (immutable after construction)
- INVARIANTS: INV-M-1 (positive), INV-M-2 (deterministic), INV-M-3 (pure), INV-M-4 (<1ms)
- EVENTS: None (synchronous query)
- EXTENSION FRICTION: ~2-3 files (impl, factory update, config)

**5. Leaderboard** — Cross-idea comparison ranked by BLIS E2E mean error.
- OBSERVES: All ideas' BLIS validation results and FINDINGS.md
- CONTROLS: Winner selection, final README.md
- OWNS: Leaderboard table
- INVARIANTS: Rankings use P1 (BLIS E2E) first; data integrity documented per idea
- EVENTS: None
- EXTENSION FRICTION: 1 row per new idea

**6. Round Controller** — Manages the iterative WP1→WP4 loop with convergence feedback.
- OBSERVES: Current round's leaderboard results, convergence review feedback, prior rounds' FINDINGS
- CONTROLS: Loop continuation decision (iterate / converge / abort), problem.md updates, round artifact archival
- OWNS: `FINDINGS_ROUND<N>.md` per round, updated `problem.md`, round history
- INVARIANTS: Each round's findings are immutable after archival; problem.md reflects all prior rounds' learnings; max rounds enforced by orchestrator
- EVENTS: None (offline orchestration)
- EXTENSION FRICTION: 0 files (process, not code)

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
       ▼
[Round Controller]
       │
       ├── target met? ──yes──→ Research complete (winner documented)
       │
       └── iterate? ──yes──→ update problem.md ──→ [WP1 next round]
                                                    ↑
              FINDINGS_ROUND<N>.md + /convergence-review
```

### Real-System Correspondence

| Building Block | vLLM v0.15.1 | BLIS |
|---|---|---|
| Data Pipeline | step traces + lifecycle + KV events | `hypotheses/h-stepml/shared/` |
| BLIS Validation Harness | Offline comparison | `validate_blis.py` + `sim/workload/calibrate.go` |
| StepML LatencyModel | Wall-clock model_execute() + scheduler overhead | `sim/latency/stepml.go` (new) |

---

## E) Architectural Risk Register

| # | Decision | Assumption | Validation Method | Cost if Wrong | Gate |
|---|----------|------------|-------------------|---------------|------|
| R1 | ProgressIndex as KV length proxy | ProgressIndex ≈ total KV cache length per request | Compare ProgressIndex with ground-truth per-request KV from lifecycle data in WP0 | WP3–WP4 rework | Before WP3 |
| R2 | 10% sampling sufficient | Sampling is approximately random | Characterize sampling distribution in WP0 | WP3–WP4 rework | Before WP3 |
| R3 | Request-batch features suffice | Per-step prediction from Request fields achieves <10% E2E | Best model on Request-only features must pass BLIS E2E <15% on majority of experiments | Research capped at current best | After final round |
| R4 | Blackbox is beatable | Current blackbox has significant BLIS E2E error | Compute blackbox BLIS E2E baseline in WP0. If E2E mean < 12%, reassess | All WP1–WP4 wasted | Before WP1 |
| R5 | Iterative rounds converge | Each round makes measurable progress toward <10% E2E | Best E2E error improves by ≥10% relative each round; 2 consecutive stagnant rounds triggers abort | All remaining rounds wasted | After each WP4 |

**Abort plans:**
- R1 fails → derive per-request KV lengths from lifecycle data instead; may need richer feature extraction
- R2 fails → request full-trace re-collection or apply importance weighting
- R3 fails → propose LatencyModel interface extension (separate design doc)
- R4 fails → cancel research; blackbox is already good enough
- R5 fails → 2 consecutive rounds without improvement: abort research loop, document best achievable result

---

## F) Architectural Evolution

### Current State

Two LatencyModel backends (`sim/latency/latency.go:129-167`):
- `hw.Roofline == true` → RooflineLatencyModel (analytical FLOPs/bandwidth)
- `hw.Roofline == false` → BlackboxLatencyModel (2-feature regression; returns 0 for scheduling/preemption)

Factory dispatch is binary. No fallback chain. No data-driven multi-method calibration.

### Target State (Research)

Three LatencyModel backends with ordered dispatch (lightweight, research-only):
1. `hw.Roofline == true` → RooflineLatencyModel (unchanged)
2. StepML artifact exists at configured path → StepMLLatencyModel (lightweight — for BLIS validation during research)
3. Fallback → BlackboxLatencyModel (preserved for backward compatibility)

New CLI flag `--stepml-model <path>` specifies the artifact location. When absent, falls back to blackbox. Production-quality implementation is future work outside this plan.

### Evolution Path

1. **WP0 (lightweight):** Add `sim/latency/stepml.go` with minimal tree evaluator + coefficient loader. Registers via existing factory. Enables BLIS validation during research. Not production quality.
2. **Research rounds (iterative):** WP1→WP4 loop refines the model through multiple rounds. Each round's best model is validated via the lightweight Go evaluator. The Go evaluator may be updated between rounds if new model types require different artifact formats.

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
4. Verification: BLIS validation runs confirm `--roofline` path is unaffected

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
- **New (WP0):** `hypotheses/h-stepml/shared/` — data loading, evaluation harness, baselines, data splits, split validation, sampling analysis, BLIS validation harness, E2E baseline establishment, component-level error attribution, per-request KV length extraction, MFU benchmark indexing, lifecycle-to-trace conversion. All ideas across all rounds use these shared modules.
- **New (WP0):** `sim/latency/stepml.go` — lightweight Go tree evaluator for BLIS validation during research
- **New (per round):** `hypotheses/h-stepml/round<N>/FINDINGS_ROUND<N>.md` — immutable round findings summary
- **New (per round):** `hypotheses/h-stepml/round<N>/idea-*/` — per-idea experiment artifacts within each round
- **New (per idea):** `hypotheses/h-stepml/round<N>/idea-*/FINDINGS_SUMMARY.md` — idea-level synthesis across sub-hypotheses
- **Evolving:** `hypotheses/h-stepml/problem.md` — updated after each round with accumulated findings

### H.2 Documentation Maintenance

| Trigger | Owner | What Updates |
|---|---|---|
| WP0 creates `hypotheses/h-stepml/` and `sim/latency/stepml.go` | WP0 | CLAUDE.md file organization + latency estimation section |
| WP0 adds `--stepml-model` flag | WP0 | CLAUDE.md CLI flags |
| WP4 completes a round | WP4 | `FINDINGS_ROUND<N>.md` created, `problem.md` updated |

### H.3 CI Pipeline Changes

- **WP0:** New Go file added to `go test ./sim/latency/...`. Lightweight tests only.
- **WP0–WP4:** Python research code not CI-gated; reproducibility via run.sh

### H.4 Dependency Management

- **Python (WP0–WP4):** `requirements.txt` in `hypotheses/h-stepml/shared/` with pinned versions
- **Go (WP0):** Zero new dependencies for lightweight evaluator (pure Go tree traversal + JSON parsing)

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
- In: Data parsing (step-level + request-level + MFU benchmarks from InferSim bench_data), Python evaluation harness, Go tree evaluator, BLIS validation harness, blackbox E2E baseline, component-level error attribution, per-request KV length extraction, sampling bias characterization, split strategies and validation, lifecycle-to-trace conversion, prefix cache semantics (D-8)
- Out: Any model training, any idea-specific code

**Behavioral Guarantees:**
- BC-0-1: Row count of parsed datasets matches source file counts- BC-0-2: MFU benchmark data indexed and queryable by GPU type, matrix shape, attention config- BC-0-3: Go tree evaluator implements all 5 LatencyModel methods and loads exported artifacts- BC-0-4: BLIS validation harness runs all experiments and outputs E2E/TTFT/ITL mean errors- BC-0-5: E2E baselines established — blackbox (the number Round 2 must beat) and roofline (informational comparison, not available to ideas)- BC-0-6: Component-level error attribution identifies which of the 5 methods dominates E2E error- BC-0-7: Sampling bias report documents step_id uniformity, autocorrelation, per-experiment representation- BC-0-8: Data split strategies available for all three paradigms (temporal, leave-one-model-out, leave-one-workload-out)- BC-0-9: ProgressIndex validated as KV proxy (R1 gate)- BC-0-10: Per-request KV length features extractable at step level
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
1. Create directory structure and requirements.txt2. Implement data loader: parse step-level traces from all 20 experiments into unified dataset; parse per-request lifecycle metrics; parse experiment metadata from directory names3. Implement data split strategies: temporal (60/20/20), leave-one-model-out, leave-one-workload-out4. Index MFU benchmarks from InferSim bench_data: queryable by GPU type, matrix shape, attention config, with nearest-neighbor interpolation5. Characterize sampling distribution: detect periodic vs. random sampling, report per-experiment coverage uniformity (bias detection per R2 gate)6. Verify step.duration_us semantics + resolve prefix cache semantics (D-8)

*Python evaluation (diagnostics):*
7. Implement per-step evaluation harness (MAPE, MSPE, Pearson r, p99 error, bootstrap CI)8. Implement per-step baselines: blackbox re-trained per model+TP, naive mean, R4 gate check, short-circuit threshold. Roofline is an informational baseline only — ideas MUST NOT use roofline predictions as inputs, features, or building blocks.9. Implement split validation: confirm temporal splits prevent autocorrelation leakage; validate ProgressIndex as KV proxy (R1 gate)
*Feature extraction tools:*
10. Component-level error attribution tool (e2e_decomposition.py): decomposes E2E into 7 components, ablates each LatencyModel method to measure marginal E2E error contribution, recommends where to focus improvement11. Per-request KV length extractor (lifecycle_kv_extractor.py): derive per-step kv_mean, kv_max, kv_sum, kv_count from lifecycle data
*BLIS validation infrastructure:*
12. Go tree evaluator (sim/latency/stepml.go): lightweight LatencyModel loading exported artifacts, ~200 lines
13. BLIS validation harness (validate_blis.py): for each experiment, construct inference-perf workload spec from experiment's original profile, run BLIS with same serving parameters, compare BLIS-produced latencies against ground-truth observed means; supports blackbox coefficient mode, stepml model mode, and roofline mode; outputs per-experiment E2E/TTFT/ITL mean errors14. E2E baseline establishment (establish_baseline.py): orchestrates two baselines through full BLIS E2E validation with parallel execution. (a) Roofline baseline — zero-calibration analytical, informational only. (b) Per-model linear regression baseline — trains per model+TP using canonical model names from exp-config.yaml, then runs BLIS. Includes R4 gate check.15. Lifecycle-to-trace converter (convert_lifecycle_to_traces.py): converts ground-truth lifecycle data to BLIS trace CSV format, available for trace-based BLIS runs if needed alongside the primary inference-perf workload approach
---

### Research Round Loop (WP1→WP2→WP3→WP4)

**This section describes WP1–WP4 as an iterative loop.** Each "research round" executes WP1→WP2→WP3→WP4 in sequence. At the end of each round, a convergence review determines whether to iterate (with updated problem statement), declare convergence (target met), or abort.

**Round artifacts are stored under `hypotheses/h-stepml/round<N>/`** (e.g., `round1/`, `round2/`, `round3/`). Each round produces:
- `FINDINGS_ROUND<N>.md` — comprehensive findings summary (what worked, what failed, binding constraints, open questions)
- Updated `problem.md` — incorporates round's findings, narrows scope, refines baselines, adds new questions
- Per-idea `FINDINGS_SUMMARY.md` — synthesizes results across the idea's sub-hypotheses (best E2E result, what worked, what failed, binding constraints)
- Per-idea directories with HYPOTHESIS.md, per-sub-hypothesis FINDINGS.md, and experiment artifacts

**Loop termination conditions:**
1. **Converge:** Best idea achieves <10% BLIS E2E mean error on all experiments → research complete, document winner
2. **Iterate:** Progress made (improvement over prior round's best) but target not met → update problem.md, start next round
3. **Abort:** No improvement over prior round AND no viable new approaches identified → evaluate falsification criteria
4. **Hard cap:** Configurable via orchestrator (default 5, up to 25). Abort after 2 consecutive rounds without improvement regardless of cap.

**Autonomous execution protocol:**

Each round (WP1→WP4) runs **without halting for confirmation** between work packages. Do not ask "should I proceed?" — execute the full pipeline. The only valid halt reasons are:

1. **Unrecoverable error** — a script fails in a way that requires user decision (not a retry)
2. **Context window approaching limits** — write a `CHECKPOINT.md` to `hypotheses/h-stepml/round<N>/` and stop cleanly

**Context management strategy for WP3:**

WP3 (experimentation) is the heaviest phase — multiple ideas with multiple sub-hypotheses. To prevent context overflow:

- **Dispatch each idea to a subagent** using the Task tool. The subagent receives: (a) the idea's HYPOTHESIS.md, (b) the shared infrastructure location, (c) instructions to run all sub-hypotheses and write FINDINGS_SUMMARY.md. The main agent collects the summary and moves on.
- If subagents are not available, execute ideas **sequentially** and use `/compact` between ideas to reclaim context.
- Each idea's FINDINGS_SUMMARY.md serves as the **compression boundary** — once written, the sub-hypothesis details can be forgotten by the main agent.

**Checkpoint/resume protocol (`CHECKPOINT.md`):**

If a round must span multiple sessions (context limit, timeout, or error), write `hypotheses/h-stepml/round<N>/CHECKPOINT.md` containing:
- Current WP step (e.g., "WP3, idea-2, sub-hypothesis h2")
- Completed work (which ideas/hypotheses are done, with FINDINGS_SUMMARY.md locations)
- Remaining work (what still needs to run)
- Any partial results or state needed to continue

The next session reads CHECKPOINT.md to resume exactly where the prior session left off.

**Unattended multi-round execution (orchestrator mode):**

For fully autonomous multi-round execution without human intervention, use the orchestrator script `hypotheses/h-stepml/run_research_loop.sh`. The orchestrator breaks each round into **separate phases**, each with its own fresh `claude -p` session and context window. This prevents context overflow — no single session needs to hold an entire round.

**Phase decomposition per round:**

```
Phase 1 (WP1+WP2):  problem.md → /research-ideas → research.md → HYPOTHESIS.md files
   ↓ on-disk artifacts bridge to next phase
Phase 2.1 (WP3):    idea-1/HYPOTHESIS.md → experiments → idea-1/FINDINGS_SUMMARY.md
Phase 2.2 (WP3):    idea-2/HYPOTHESIS.md → experiments → idea-2/FINDINGS_SUMMARY.md
Phase 2.3 (WP3):    idea-3/HYPOTHESIS.md → experiments → idea-3/FINDINGS_SUMMARY.md
   ↓ on-disk artifacts bridge to next phase
Phase final (WP4):  all FINDINGS_SUMMARY.md → leaderboard → convergence-review → problem.md → STATUS
```

Each phase is a **separate `claude -p` invocation** with ~200 max turns. On-disk artifacts (research.md, HYPOTHESIS.md, FINDINGS_SUMMARY.md, problem.md) are the communication channel between phases. No conversation history is carried — each session reads CLAUDE.md + MEMORY.md automatically and reads specific artifacts as instructed by the orchestrator prompt.

**Resumability:** Every phase checks for its completion marker before running. If the orchestrator is killed and restarted, it skips completed phases and resumes from where it left off. Markers:
- Phase 1: `.phase1_done` marker file + idea directories exist
- Phase 2.x: `FINDINGS_SUMMARY.md` exists in idea directory
- Phase final: `STATUS` file exists

**STATUS file contract:** After completing WP4 (or declaring convergence/abort), write `hypotheses/h-stepml/round<N>/STATUS` as a single line: `CONVERGED`, `ITERATE`, or `ABORT`. This file is the **sole communication channel** between rounds — the orchestrator does not parse FINDINGS or problem.md.

**Why this works:** We've designed problem.md to be self-contained with all prior round learnings. Each fresh `claude -p` session reads CLAUDE.md (which says "follow the macro plan") and reads the specific artifacts it needs. No conversation history or `/compact` is required because each phase fits comfortably in a single context window.

---

#### WP1: Research Ideation (per round)

**Building Block Change:** Populates Experiment Engine inputs
**Extension Type:** Process (skill invocation)
**Motivation:** Generate 3+ literature-grounded ideas for achieving <10% BLIS E2E mean error via any combination of LatencyModel method improvements. In rounds >1, ideas are informed by prior rounds' findings and focus on identified gaps.

**Scope:**
- In: `problem.md` (**the sole input** — must be self-contained with all prior round learnings baked in), invoke `/research-ideas`
- Out: Model training, experiment execution

**Round-Specific Behavior:**
- **Round 1:** `problem.md` written from design doc + WP0 baselines + component error attribution. Broad exploration across all 5 LatencyModel methods.
- **Round N (N>1):** `problem.md` has been updated by the prior round's WP4 Step 5 with all accumulated knowledge: per-experiment results, solved vs unsolved experiments, successful techniques to build on, data characteristics, eliminated approaches, binding constraints, narrowed questions, convergence review prescriptions, and cumulative round history. Ideas must address at least one binding constraint and should build on successful techniques from prior rounds.

**Behavioral Guarantees:**
- BC-1-1: `research.md` contains 3+ ideas with literature citations
- BC-1-2: Each idea specifies which of the 5 LatencyModel methods it targets
- BC-1-3: Each idea documents its Go integration path
- BC-1-4: Ideas are diverse — not all limited to step-time-only ML models
- BC-1-5: (Round N>1) At least one idea directly addresses the top binding constraint from FINDINGS_ROUND<N-1>.md
- BC-1-6: (Round N>1) Ideas do not repeat approaches already tried and failed in prior rounds unless with a fundamentally different variation
- BC-1-7: **Each idea in research.md includes a generalization plan** covering all three P2 dimensions: (a) LOMO experimental design (which model features enable transfer), (b) LOWO experimental design (which features are workload-invariant), (c) vLLM-args sensitivity analysis plan (structural dependence on each of the 7 key vLLM parameters). Ideas without all three plans are rejected during WP2 scaffolding.
- BC-1-8: **Reviewers evaluate generalization as a blocking criterion.** The `/research-ideas` review prompt inherits from problem.md's "Idea Review Criteria" table where LOMO, LOWO, and vLLM-args are marked as blocking dimensions. Reviewers must flag ideas that lack concrete generalization plans.

**Risks:**
1. Ideas are too similar → problem.md explicitly requests diverse approaches (e.g., step-time ML, end-to-end calibration, multi-component optimization)
2. Ideas ignore the non-StepTime methods → problem.md includes WP0 component error attribution showing which methods contribute most to E2E error
3. (Round N>1) Ideas rehash prior failures → problem.md "Eliminated Approaches" section prevents repetition

**Cross-Cutting:** None. Produces `research.md` consumed by WP2.

---

#### WP2: Hypothesis Scaffolding (per round)

**Building Block Change:** Adds Experiment Engine structure
**Extension Type:** Process (directory creation + HYPOTHESIS.md authoring)
**Motivation:** Map research ideas to testable hypotheses with idea-specific sub-hypothesis decomposition.

**Scope:**
- In: Extract top 3-5 ideas from research.md, create per-idea directories under `round<N>/`, write HYPOTHESIS.md files
- Out: Experiment code, running experiments

**Behavioral Guarantees:**
- BC-2-1: Each idea has 2-3 sub-hypotheses with idea-specific decomposition (NOT fixed h1-features/h2-model/h3-generalization)
- BC-2-2: Each HYPOTHESIS.md includes Related Work with citations from research.md
- BC-2-3: At least one sub-hypothesis per idea reports BLIS E2E mean error
- BC-2-4: Claims and refutation criteria expressed in terms of BLIS E2E mean error where possible
- BC-2-5: Each idea documents its training strategy and data split approach (subject to data integrity requirements)
- BC-2-6: (Round N>1) Each HYPOTHESIS.md references relevant prior round findings and explains how this approach differs
- BC-2-7: **Every idea MUST include a model-generalization sub-hypothesis (LOMO)** and a **workload-generalization sub-hypothesis (LOWO)**. These may be combined into one sub-hypothesis or separate. (vLLM-args sensitivity is NOT a hypothesis — it is checked during WP1 idea review per BC-1-7.) If an idea does not train a new model (e.g., infrastructure-only changes), generalization hypotheses must either (a) test the underlying model in the new execution mode, or (b) include a written `GENERALIZATION_NOTE.md` documenting why generalization testing is not applicable, with explicit evidence (e.g., cross-workload error table) showing implicit generalization coverage. The note must be reviewed during WP4.

**Risks:**
1. Ideas don't map to hypotheses → each idea must specify a falsifiable BLIS E2E claim

**Cross-Cutting:** Creates directory structure consumed by WP3.

**Validation Gate:** Convergence review (h-design, 5 perspectives) on each idea's HYPOTHESIS.md.

---

#### WP3: Wave-Parallel Experimentation (per round)

**Building Block Change:** Executes Experiment Engine
**Extension Type:** Policy template (new algorithms per idea)
**Motivation:** Test each idea's approach and measure BLIS E2E mean error.

**Scope:**
- In: All sub-hypotheses per idea, using WP0 shared infrastructure
- Out: Production Go integration (future work, outside this plan)

**Behavioral Guarantees:**
- BC-3-1: Every idea uses WP0's data loader (no custom parsing)
- BC-3-2: Every idea's final sub-hypothesis reports BLIS E2E mean error via validate_blis.py
- BC-3-3: Every sub-hypothesis FINDINGS.md includes BLIS E2E per experiment + data integrity documentation (which data was training vs. evaluation)
- BC-3-4: Data integrity requirements satisfied: non-overlapping splits, no feature leakage, no model selection on training experiments' BLIS E2E
- BC-3-5: Ideas with BLIS E2E worse than the **current best baseline** (blackbox for round 1, best prior round result for round N>1) are dropped
- BC-3-6: Each idea's training strategy and split documented and justified
- BC-3-7: No idea uses roofline predictions as inputs, features, correction targets, or building blocks — roofline is a comparison baseline only
- BC-3-8: After all sub-hypotheses for an idea are tested, an **idea-level `FINDINGS_SUMMARY.md`** is written in the idea's root directory (e.g., `round<N>/idea-<X>-<name>/FINDINGS_SUMMARY.md`)
- BC-3-9: **Generalization hypotheses MUST be executed**, not just scaffolded. FINDINGS_SUMMARY.md cannot be written until LOMO and LOWO results (or a GENERALIZATION_NOTE.md with evidence) exist. If a prior sub-hypothesis fails badly enough to invalidate generalization testing, the FINDINGS_SUMMARY.md must explicitly document "LOMO/LOWO: Not executed — blocked by [sub-hypothesis] failure" with root cause.
- BC-3-10: Every FINDINGS_SUMMARY.md must contain a **"Generalization Results"** section with a LOMO table and a LOWO table (or reference to GENERALIZATION_NOTE.md). This section is mandatory — omission is a WP4 review gate failure. (vLLM-args sensitivity is validated at WP1 ideation review, not during WP3 experimentation.)

**Idea-Level Findings Summary (`FINDINGS_SUMMARY.md`):**

After all sub-hypotheses for a given idea are complete, write a `FINDINGS_SUMMARY.md` in the idea's root directory that synthesizes results across its sub-hypotheses. This summary is the primary input to WP4's leaderboard and the round-level `FINDINGS_ROUND<N>.md`. Structure:

1. **Idea recap** — one-paragraph description of the approach and which LatencyModel methods it targets
2. **Sub-hypothesis results table** — status (supported/refuted/partial), key metric, one-line takeaway per sub-hypothesis
3. **Best BLIS E2E result** — the idea's best **full per-experiment E2E/TTFT/ITL error table** and overall mean, with the exact model configuration (features, model type, hyperparameters, overhead handling) that produced it
4. **What worked** — techniques, features, or design choices that contributed to good results. Be specific enough that the next round can build on these without re-reading sub-hypothesis code.
5. **What failed and why** — root causes (not just symptoms) for sub-hypotheses that were refuted or underperformed
6. **Binding constraints** — what limits this idea from reaching the <10% target (if it didn't reach it). Distinguish addressable vs structural.
7. **Data insights** — any new understanding of the training data, feature importance, or system behavior discovered during experimentation
8. **Comparison to baseline** — side-by-side vs blackbox and vs prior round's best (if round >1)
9. **Generalization results (MANDATORY):**
   - **(a) LOMO results** — per-fold MAPE table (train on 3 models, predict 4th). Target: <80% per fold. If not executed: "Not executed — blocked by [reason]" with root cause.
   - **(b) LOWO results** — per-fold MAPE table (train on 2 workloads, predict 3rd). Target: <50% per fold. If not executed: "Not executed — blocked by [reason]" with root cause.
   - If not applicable (infrastructure-only idea), reference GENERALIZATION_NOTE.md with evidence of implicit coverage.
   - Note: vLLM-args sensitivity is checked at ideation time (WP1 BC-1-7), not here.
10. **Go integration feasibility** — artifact format, estimated complexity, any blockers

**Risks:**
1. All ideas fail → re-examine feature assumptions; revisit R1, R3; feed failure analysis into next round's problem.md
2. BLIS validation harness bottleneck → each BLIS run takes ~2-5 min; experiments × multiple ideas is manageable

**Cross-Cutting:** Consumes WP0 shared infrastructure. Each idea writes only to its own directory under `round<N>/`.

**Validation Gate:** R3 evaluated after WP3. R5 confirmed for surviving ideas.

---

#### WP4: Leaderboard, Convergence Review & Round Wrap-up (per round)

**Building Block Change:** Populates Leaderboard + drives loop decision
**Extension Type:** Process (analysis + documentation + convergence review)
**Motivation:** Compare all ideas on BLIS E2E mean error, run convergence review, decide whether to iterate or declare convergence.

**Scope:**
- In: All ideas' `FINDINGS_SUMMARY.md` and BLIS validation results from current round
- Out: `FINDINGS_ROUND<N>.md`, updated `problem.md`, loop decision (iterate/converge/abort)

**WP4 Procedure:**

**Step 0 — Generalization Gate (pre-leaderboard):** Before ranking, verify that EVERY idea has either (a) executed LOMO + LOWO generalization hypotheses with results in FINDINGS_SUMMARY.md section 9, or (b) a GENERALIZATION_NOTE.md with evidence. Any idea missing both is flagged as incomplete — the WP3 subagent must be resumed to run the missing generalization tests before WP4 can proceed. This gate exists because Rounds 1-3 showed a pattern of generalization tests being scaffolded but never executed.

**Step 1 — Leaderboard:** Rank all ideas by P1 metric (BLIS E2E mean error). Compare against prior round's best and blackbox baseline. Produce leaderboard table.

**Step 2 — Round Findings (`FINDINGS_ROUND<N>.md`):** Write comprehensive findings summary:
- Section 1: Problem recap (what this round was trying to solve)
- Section 2: What we tried (all ideas, brief)
- Section 3: What worked (ranked results, feature insights, successful techniques)
- Section 4: What failed and why (root causes, not just symptoms)
- Section 5: Binding constraints (what blocks further progress)
- Section 6: Data characteristics learned
- Section 7: Successful techniques and patterns (reusable across rounds)
- Section 8: Failed techniques and anti-patterns (do not repeat)
- Section 9: Round-specific design limitations
- Section 10: Open questions for next round
- Section 11: Reproducibility
- Section 12: Summary for future ideation

**Step 3 — Convergence Review:** Run `/convergence-review` (hypothesis FINDINGS gate, 10 perspectives) on `FINDINGS_ROUND<N>.md`. The review evaluates:
- Are findings well-supported by evidence?
- Are binding constraints correctly identified?
- Are failure root causes (not just symptoms) documented?
- Is the updated problem statement well-scoped for the next round?
- Is there a plausible path to <10% E2E from the current best result?

**Step 4 — Loop Decision:**
- **Converge** (best idea <10% E2E on all experiments): Research complete — document winner and artifacts
- **Iterate** (improvement over prior round, plausible path forward): Update `problem.md`, proceed to WP1 for round N+1
- **Abort** (no improvement AND no viable new approaches): Document in final FINDINGS, evaluate falsification criteria

**Step 5 — Update `problem.md`** (if iterating, skip if converging/aborting):

**Step 6 — Write STATUS file (MANDATORY, always the last action):**

Write `hypotheses/h-stepml/round<N>/STATUS` containing exactly one word: `CONVERGED`, `ITERATE`, or `ABORT`. No other content. This file is read by the orchestrator script (`run_research_loop.sh`) to decide whether to launch the next round. **If this file is missing, the orchestrator halts and requires manual intervention.**

**Step 5 detail — Update `problem.md`** (if iterating):

The updated problem.md is the **sole input** to the next round's `/research-ideas` invocation. Everything the next round needs to know must be in this file — do not assume the next session will read FINDINGS_ROUND<N>.md or any idea-level artifacts. Update the following sections (each clearly labeled `### Updated after Round <N>`):

- **Baseline Results:** Replace with this round's best numbers as the new bar to beat. Include the **full per-experiment E2E/TTFT/ITL error table** (not just the mean) so the next round knows which experiments are solved (<10%) and which remain problematic. Include the best model's configuration (features used, model type, overhead handling approach) — not just the error number.
- **Solved vs Unsolved Experiments:** Explicit table: which experiments are below target, which are above, and by how much. This focuses the next round's effort on the remaining gaps.
- **Successful Techniques (build on these):** Carry forward from FINDINGS_ROUND<N>.md Section 7 — techniques, features, and design patterns that worked. The next round should extend these, not reinvent them.
- **Data Characteristics Learned:** Carry forward from FINDINGS_ROUND<N>.md Section 6 — insights about the data that any future approach must respect (e.g., step.duration_us semantics, overhead ranges, MoE vs dense differences, phase distributions).
- **Eliminated Approaches (do not repeat):** Failed ideas with root causes (not just "didn't work" but why). Must be specific enough that the next round can avoid the same pitfall with a different surface-level approach.
- **Binding Constraints:** Carry forward from FINDINGS_ROUND<N>.md Section 5 — what specifically blocks further progress. Distinguish between constraints that are addressable (next round should target) vs structural (accept and work around).
- **Key Questions:** Narrow to reflect what's still unknown. Remove questions answered by this round. Add new questions raised by this round's findings.
- **Prescribed Focus Areas:** Convergence review feedback translated into specific directives for the next round's ideation.
- **Cumulative Round History:** One-paragraph summary per completed round: best result, key insight, why it didn't converge. Provides context without requiring the reader to open prior FINDINGS files.

**Behavioral Guarantees:**
- BC-4-1: Rankings ordered by P1 metric (BLIS E2E mean error) first
- BC-4-2: Data integrity verified: each idea's training/evaluation boundary documented
- BC-4-3: Winner identified with Go integration path documented (if converging)
- BC-4-4: Falsification criteria checked (design doc section)
- BC-4-5: `FINDINGS_ROUND<N>.md` saved to `hypotheses/h-stepml/round<N>/` — immutable after creation
- BC-4-6: `problem.md` updated with round findings before next WP1 (if iterating)
- BC-4-7: `/convergence-review` feedback documented and addressed in updated problem.md
- BC-4-8: Cross-round leaderboard maintained showing progression across all rounds
- BC-4-9: `STATUS` file written to `hypotheses/h-stepml/round<N>/STATUS` as the **last action** — exactly one word: `CONVERGED`, `ITERATE`, or `ABORT`

**Risks:**
1. No idea achieves <10% BLIS E2E → evaluate falsification criteria; iterate if binding constraint is addressable
2. Convergence review identifies issues not caught in experimentation → feed back into problem.md
3. Rounds plateau (no improvement) → abort after 2 consecutive rounds without improvement

**Cross-Cutting:** Produces FINDINGS_ROUND<N>.md, updated problem.md. Inputs to WP1 (if iterating).

---

---

## J) Dependency DAG

```
WP0 (Shared Infrastructure + BLIS Validation Loop + Go Evaluator)
 │
 ├── R1, R2, R4 validation gates
 │
 ▼
┌─────────────────────────────────────────────────────────────┐
│                 RESEARCH ROUND LOOP (max 5)                 │
│                                                             │
│  WP1 (Research Ideation — problem.md with prior findings)   │
│   │                                                         │
│   ▼                                                         │
│  WP2 (Hypothesis Scaffolding)                               │
│   │                                                         │
│   ├─────────┬─────────┐                                     │
│   ▼         ▼         ▼                                     │
│  WP3-a    WP3-b    WP3-c  (parallel; sequential within)    │
│   │         │         │                                     │
│   └─────────┴─────────┘                                     │
│             │                                               │
│             ├── R3, R5 validation gates                     │
│             ▼                                               │
│  WP4 (Leaderboard + /convergence-review)                    │
│   │                                                         │
│   ├── FINDINGS_ROUND<N>.md saved                            │
│   │                                                         │
│   ├── target met? ───yes───→ EXIT LOOP (research complete)  │
│   │                                                         │
│   ├── progress + path? ──→ update problem.md ──→ WP1 (N+1) │
│   │                                                         │
│   └── no progress? ──→ ABORT                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Parallelizable workstreams:**
- WP3-a, WP3-b, WP3-c are fully parallel within each round (independent ideas, shared read-only data)
- Within WP3-x: sub-hypotheses are sequential (idea-specific dependency chain)
- Rounds are strictly sequential (round N+1 depends on round N's findings and updated problem.md)

**Merge sequencing:**
- WP0 must complete and pass R1/R2/R4 before round 1's WP1
- WP2 must complete before any WP3 (within each round)
- All WP3 must complete before WP4 (within each round)
- WP4 convergence review must complete before loop decision

**Round progression:**
- Round N's `FINDINGS_ROUND<N>.md` is immutable after creation
- Round N's findings feed into round N+1's `problem.md` update
- Cross-round leaderboard tracks best result per round — the bar to beat rises monotonically
- Convergence review feedback from round N becomes "Prescribed Focus Areas" in round N+1's problem.md

---

## K) Design Bug Prevention Checklist

### General

- [ ] **Scaffolding creep:** Every Python module in WP0 has tests. Go evaluator in WP0 has basic tests. No dead code.
- [ ] **Documentation drift:** CLAUDE.md updated in WP0 (Go evaluator, shared infra).
- [ ] **Test infrastructure duplication:** All experiments use `hypotheses/h-stepml/shared/` and `validate_blis.py`.
- [ ] **Golden dataset staleness:** No changes to golden dataset (no production integration in scope).

### DES-Specific

- [ ] **Type catalog trap:** No Go struct definitions in this macro plan. Evolution described behaviorally.
- [ ] **Fidelity for its own sake:** Every modeled component traces to AQ-1 (E2E mean fidelity).
- [ ] **Golden without invariant:** Lightweight Go evaluator in WP0 has basic tests. Production invariant tests deferred.
- [ ] **Mixing exogenous and endogenous:** StepML is a pure query (no events, no state mutation).

### Module Architecture

- [ ] **Shotgun surgery:** StepML constructed in one place (factory in `sim/latency/latency.go`).
- [ ] **Config mixing:** StepML config fields module-scoped (R16).
- [ ] **No silent data loss (R1):** Factory fallback logged, not silent.
- [ ] **Guard division (R11):** Zero denominators checked in prediction code.
- [ ] **Strict YAML parsing (R10):** New fields use `yaml.KnownFields(true)`.
- [ ] **CLI flag precedence (R18):** `--stepml-model` does not silently override defaults.yaml.

### Roofline Isolation

- [ ] **Roofline byte-identical output:** Verified during BLIS validation runs (WP0 lightweight evaluator does not modify roofline path).
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

### Iterative Round Integrity

- [ ] **Round findings immutability:** `FINDINGS_ROUND<N>.md` is never modified after creation — subsequent rounds reference but do not edit prior findings.
- [ ] **Problem statement accumulation:** `problem.md` clearly labels which sections were added/updated in which round (e.g., `### Updated after Round 2`).
- [ ] **Problem statement self-containedness:** `problem.md` is the **sole input** to the next round's `/research-ideas`. Everything the next round needs — per-experiment results, successful techniques, data characteristics, eliminated approaches, binding constraints — must be in this file. Verify by asking: "Could someone read only problem.md and understand what to try next?"
- [ ] **Successful techniques carried forward:** `problem.md` "Successful Techniques" section includes specific, actionable techniques from prior rounds that the next round should build on (not just "XGBoost worked").
- [ ] **Per-experiment status carried forward:** `problem.md` includes the full per-experiment E2E error table showing which experiments are solved (<10%) and which remain, so the next round can focus effort.
- [ ] **Data characteristics carried forward:** `problem.md` "Data Characteristics Learned" section includes insights any future approach must respect (overhead semantics, MoE vs dense, phase distributions).
- [ ] **Eliminated approach tracking:** Failed approaches from prior rounds are listed in `problem.md` "Eliminated Approaches" section with root causes — prevents repetition.
- [ ] **Baseline monotonicity:** The "number to beat" in `problem.md` is updated to the best result from any prior round — never regresses to the original blackbox baseline.
- [ ] **Convergence review traceability:** Each round's convergence review feedback is documented in `FINDINGS_ROUND<N>.md` and addressed in the updated `problem.md`.
- [ ] **Max rounds enforced:** Hard cap set by orchestrator (default 5, up to 25). Abort after 2 consecutive stagnant rounds regardless.
- [ ] **Cross-round leaderboard:** Maintained showing best result per round, enabling trend analysis (are we converging?).

### Generalization Enforcement (P2 — Three Dimensions)

**Ideation-time checks (WP1 — before any code is written):**
- [ ] **vLLM-args sensitivity in research.md:** Each idea's description includes analysis of structural dependence on 7 vLLM parameters. Reviewers reject ideas that are structurally fragile without a recalibration story.
- [ ] **LOMO + LOWO plans in research.md:** Each idea describes its LOMO and LOWO experimental design. Reviewers reject ideas without concrete generalization plans.

**Experimentation-time checks (WP3 — must be executed, not just scaffolded):**
- [ ] **LOMO + LOWO executed per idea:** Every idea's FINDINGS_SUMMARY.md §9 has (a) LOMO table and (b) LOWO table with actual results.
- [ ] **No scaffolded-but-unexecuted generalization:** Before WP4, verify output directories for LOMO/LOWO sub-hypotheses are non-empty.
- [ ] **Infrastructure ideas not exempt:** Ideas that don't train new models still need either LOMO/LOWO in new execution mode, or GENERALIZATION_NOTE.md with cross-experiment evidence.

**Gate checks (WP4 — before leaderboard):**
- [ ] **WP4 Step 0 gate:** LOMO + LOWO completeness verified before leaderboard ranking.
