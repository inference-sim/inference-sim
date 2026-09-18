# Implementation Plan: LoRA Control-Plane Subsystem

**Branch**: `007-lora-control-plane` | **Date**: 2026-07-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/007-lora-control-plane/spec.md`

## Summary

Give BLIS enough LoRA awareness to research adapter placement, migration, and
adapter-aware routing in a deterministic, GPU-free harness. The work grafts an
**adapter identity** onto the existing `Model`-tag plumbing, adds a **per-instance
resident-adapter set** (capacity + LRU eviction), ports the Digital Twin's **three
adapter cost terms** (cold-load latency by rank, per-step compute overhead by unique
adapters × max rank, per-adapter HBM footprint) as *deltas* onto BLIS's calibrated
engine, exposes a **LoRA-aware routing scorer** as the placement-policy extension
point, and surfaces **per-adapter metrics**.

The subsystem is a **Subsystem Module** (design doc required, no-op default
mandatory) plus one **Policy Template** (the scorer). Technical approach, grounded in
the code as read on 2026-07-15:

- Adapters are a **pre-declared registry** (`id → rank`); requests reference an adapter by string id (clarified 2026-07-15). Rank is the single source of truth for both load latency and HBM footprint.
- Cold-load latency uses the tiered-KV transfer shape `baseLatency + ceil(size/bw)` (`sim/kv/tiered.go`) and is applied as a **pre-admission gate** (request cannot join a batch until its adapter is resident; loads serialized per instance).
- The resident-adapter set reuses the `cpuTier` hash-keyed LRU pattern (`store`/`touch`/`unlink`/`appendToTail`, `sim/kv/tiered.go`).
- The compute-overhead term folds into the existing `LatencyModel.StepTime(batch []*Request)` (which already receives requests) — no new single-impl interface (R13).
- HBM accounting subtracts per-adapter bytes in `CalculateKVBlocks` (`sim/latency/kv_capacity.go`), extending INV-4.
- The scorer is a new `scorerFunc` entry reading a resident-adapter-set field added to `RoutingSnapshot` — extending an existing type, not minting an interface (R13).

## Technical Context

**Language/Version**: Go 1.22+ (single implementation language; no CGO, no Python — constitution tech-stack constraint)
**Primary Dependencies**: `cobra` (CLI), `logrus` (diagnostics→stderr), `gopkg.in/yaml.v3` (strict config), `gonum` (stats). **No new dependencies.**
**Storage**: N/A — in-memory adapter registry and per-instance resident sets; no external storage.
**Testing**: `go test ./...` (table-driven; every golden test paired with an invariant test per Principle IV), `golangci-lint run ./...` (v2.9.0, zero tolerance). Full suite MUST stay < 60s.
**Target Platform**: CPU-only deterministic discrete-event simulator (Linux/macOS dev).
**Project Type**: Single Go project — CLI (`cmd/`) over a simulation library (`sim/`, `sim/cluster/`, `sim/kv/`, `sim/latency/`, `sim/workload/`, `sim/trace/`).
**Performance Goals**: (1) No-op default: adapter-blind runs byte-identical to pre-feature output (INV-6, SC-001, SC-006). (2) No measurable slowdown when unconfigured. (3) Test suite < 60s.
**Constraints**: `sim/` never terminates (errors/panic only); unidirectional deps `cmd/ → sim/cluster/ → sim/`; subpackages register into `sim/` via `init()`; determinism via `PartitionedRNG` (this subsystem introduces **no randomness** — LRU and cost terms are deterministic, so no new RNG subsystem).
**Scale/Scope**: Research regimes of N adapters (heterogeneous rank) contending for M slots across K instances; skewed popularity; bursty per-adapter arrivals. Delivery decomposes into 7 small, individually no-op-safe PRs (see Project Structure).

**No unresolved NEEDS CLARIFICATION.** The three spec-level ambiguities (cold-load gating, memory source, adapter identity) were resolved by `/speckit.clarify` (Session 2026-07-15). Two design-doc-level items remain and are resolved in `research.md`: the concrete mechanism realizing the pre-admission gate, and static-vs-dynamic memory (decided: static first).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment | Status |
|---|---|---|
| **I. Architecture & Layering** | New state lives in `sim/` (instance resident-set) and `sim/cluster/` (routing); registry threads from `sim/workload/`. Bridge type `RoutingSnapshot` (already in `sim/`) is extended, not moved. No kernel changes. Dep direction preserved. | ✅ PASS |
| **II. Determinism** | No new randomness — resident-set LRU, cost terms, and eviction are deterministic functions of arrival order. No new `PartitionedRNG` subsystem. Per-adapter metric maps sorted before output (R2). No-op default preserves INV-6. | ✅ PASS |
| **III. Interface & Module Design** | Extend `LatencyModel.StepTime` (already takes `[]*Request`) and `RoutingSnapshot` (add resident-set field) rather than mint single-impl interfaces (R13). Scorer is a `scorerFunc`, registered via existing switch + `validScorerNames` (R8). Adapter cost model behind a queryable, pure accessor. | ✅ PASS |
| **IV. Test-First (BDD/TDD)** | Every PR: GIVEN/WHEN/THEN contracts → failing tests → code. Golden tests (no-op byte-identity, per-adapter metrics) each paired with invariant tests (capacity ≤ N, INV-1, INV-4, INV-6). THEN clauses describe observable behavior (TTFT gap, resident-set size), not internal fields. | ✅ PASS |
| **V. Error Handling** | Config validation in `cmd/` → `logrus.Fatalf`; library constructors → `panic` on invalid (negative rank/capacity); runtime adapter-fit failure → `bool`/counter, never silent `continue` (R1). Eviction loop bounded by capacity (R19). | ✅ PASS |
| **VI. Configuration Discipline** | LoRA config grouped as a module sub-config, strictly parsed (R10), `*int`/`*float64` where zero is meaningful (R9), `Flags().Changed` before defaults (R18). **⚠ Adds a 7th `SimConfig` sub-config** — constitution states "exactly 6." Justified in Complexity Tracking (new Subsystem Module). **C1: to be ratified in the design-doc convergence review — preferred resolution is amending Principle VI to "one sub-config per module" (constitution v1.1.0, separate governed PR).** | ⚠ JUSTIFY (deferred to design doc) |
| **VII. System Invariants** | Extends INV-4 (`allocated + free + adapter-reserved = total`); preserves INV-1, INV-6; adds new invariant `resident ≤ capacity`; cold-load gating respects INV-5 (`enqueue ≤ schedule`) and must not violate INV-8 (work-conserving) or INV-9 (oracle boundary — routing reads adapter id/rank, never `OutputTokens`). | ✅ PASS |
| **VIII. Antipattern Prevention** | Load-bearing rules: **R4** (audit ALL `Request` literal sites when adding `Adapter`), **R13** (no single-impl interface), **R16** (module sub-config), **R17** (scorer documents snapshot fields + freshness), **R22** (adapter-fit pre-check ≥ actual op), **R23** (roofline + trained-physics apply the same adapter delta). | ✅ PASS (tracked per-PR) |

**Extension classification** (Extension Framework): **Subsystem Module** (adapter identity + resident set + cost terms + events) — design doc required, no-op default mandatory. Plus one **Policy Template** (the scorer, ≤3 files, no design doc). Touch-point budget for a Subsystem Module legitimately exceeds the ≤3-file target; footprint justified in Complexity Tracking.

**DES design requirements** addressed in the design doc / `research.md`:
- **Model scoping**: models adapter placement cost (load/compute/memory); omits adapter training and cross-GPU solver.
- **Event classification**: adapter load is **endogenous** (state-driven, triggered by a cold request reaching batch formation), not exogenous.
- **State vs statistics**: resident-adapter set is *state*; per-adapter load counts / TTFT are *statistics* — separated.
- **Verification**: `resident ≤ capacity`, INV-1, INV-4, INV-6. **Validation**: the Digital Twin (two pre-fitted configs).
- **Randomness**: none introduced.

**Gate result: PASS** with one justified complexity item (7th sub-config). No unjustified violations.

## Project Structure

### Documentation (this feature)

```text
specs/007-lora-control-plane/
├── plan.md              # This file (/speckit.plan)
├── spec.md              # Feature spec (+ Clarifications)
├── research.md          # Phase 0 output (/speckit.plan)
├── data-model.md        # Phase 1 output (/speckit.plan)
├── quickstart.md        # Phase 1 output (/speckit.plan)
├── contracts/           # Phase 1 output (/speckit.plan)
│   ├── config-schema.md         # LoRAConfig sub-config + workload adapter fields
│   ├── cli-flags.md             # New `blis run`/`replay` flags
│   ├── latency-model.md         # StepTime adapter-overhead delta contract
│   ├── routing-snapshot.md      # Resident-adapter-set field + scorer contract
│   └── metrics.md               # Per-adapter metric fields
├── checklists/
│   └── requirements.md  # Spec quality checklist (from /speckit.specify)
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

Existing tree; the feature touches these packages (no new top-level dirs):

```text
cmd/
└── run.go, replay.go        # New LoRA CLI flags; Flags().Changed guards (R18)

sim/
├── request.go               # + Adapter string field (R4: audit all construction sites)
├── simulator.go             # + LoRAConfig sub-config on SimConfig; pre-admission gate hook near FormBatch/step
├── latency_model.go         # StepTime contract extended (adapter compute-overhead delta); model construction gains an adapter-cost accessor for id→rank (T030a)
├── routing.go               # RoutingSnapshot + resident-adapter-set field
├── routing_scorers.go       # + "lora-affinity" scorerFunc + validScorerNames entry (R8)
├── lora/                     # NEW subpackage: registry, resident-set (LRU), cost model, load-gate state
│   └── (registers into sim/ via init() — no reverse import)
├── kv/tiered.go             # Pattern source (LRU + baseLatency+ceil(size/bw)); reused, not modified
├── latency/kv_capacity.go   # CalculateKVBlocks: subtract per-adapter HBM bytes (INV-4)
└── workload/spec.go         # ClientSpec/CohortSpec + Adapter field; registry declaration; threading

sim/cluster/
└── cluster_event.go         # Adapter rides per-model routing filter; resident-set into snapshot build
```

**Structure Decision**: Single Go project, no new top-level directories. A new
`sim/lora/` subpackage owns the adapter registry, resident-set (LRU), and cost
model, registering into `sim/` via `init()` to preserve the unidirectional
dependency rule (Principle I). Cross-cutting fields (`Request.Adapter`,
`RoutingSnapshot` resident set, `SimConfig.LoRAConfig`) live in `sim/` where the
bridge types already are.

### PR decomposition (each Small-tier, no-op-safe; from design doc §8)

| PR | Scope | Primary invariant / rule |
|---|---|---|
| **1** | Adapter identity plumbing: `Request.Adapter`, `ClientSpec/CohortSpec.Adapter`, workload threading, `RequestMetrics.Adapter`, adapter registry | R4 (all construction sites); INV-6 byte-identity golden |
| **2** | Per-instance resident-adapter set + capacity sub-config + LRU eviction | New inv `resident ≤ capacity`; R16, R3/R9/R10/R18 |
| **3** | Cold-load latency via pre-admission gate | Inv: load latency ≥ 0, charged only on cold; INV-5/INV-8 |
| **4** | Per-batch compute-overhead in `StepTime` (unique × max rank) | R13, R23 (roofline + trained-physics parity) |
| **5** | Adapter HBM accounting in `CalculateKVBlocks` (static) | Extends INV-4 |
| **6** | LoRA-aware routing scorer (Policy Template) | R17 freshness; INV-7; INV-9 |
| **7** *(opt)* | Calibration: import DT rank tables + comparison | R12 if output format changes |

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| **7th `SimConfig` sub-config (`LoRAConfig`)** — constitution VI states "exactly 6 embedded sub-configs" | The Extension Framework classifies this as a **Subsystem Module**, which by definition owns its own module-scoped config; capacity, cost coefficients, and the registry are one cohesive module (R16 "group by module"). | *Spreading LoRA fields across existing sub-configs* (capacity→KVCacheConfig, coeffs→LatencyCoeffs, registry→WorkloadConfig) rejected: it fragments one module across three configs, breaks R16 cohesion, and makes independent validation impossible. The "exactly 6" reflects today's module count; adding a module adds its config. |
| **Subsystem Module touch-point footprint > 3 files** | Adapter identity is genuinely cross-cutting (request, workload, latency, routing, memory, metrics); a Subsystem Module's ≤3-file target does not apply (that target is for policy templates / config params). | *Confining to ≤3 files* rejected: impossible without hiding cross-cutting state behind a god-object, which would violate Principle I layering and R14 single-responsibility. Footprint is minimized by extending existing bridge types rather than adding new ones. |
