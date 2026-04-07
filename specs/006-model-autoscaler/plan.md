# Implementation Plan: Phase 1C Model Autoscaler

**Branch**: `006-model-autoscaler` | **Date**: 2026-04-01 | **Spec**: [spec.md](./spec.md)  
**Design Doc**: [docs/plans/2026-04-01-phase1c-autoscaling-design.md](../../docs/plans/2026-04-01-phase1c-autoscaling-design.md)  
**Tracking Issue**: [#696](https://github.com/inference-sim/inference-sim/issues/696)

## Summary

Add a model-level autoscaling pipeline to BLIS's cluster simulator, mirroring the llm-d WVA V2 architecture. The pipeline fires on a configurable `ScalingTickEvent`, collects per-replica metrics from `RouterState`, runs an `Analyzer` per model to compute supply/demand signals in token units, runs an `Engine` across all models to produce variant-aware scale decisions, and applies those decisions via an `Actuator` after a configurable actuation delay. The feature is decomposed into three PRs: 1C-1a (interfaces+wiring, merged in #934), 1C-1b (V2SaturationAnalyzer+DefaultCollector+UnlimitedEngine+DirectActuator — end-to-end WVA loop), and 1C-1d (GreedyEngine — inventory-aware allocation). The minimal viable pipeline for WVA team validation is `DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator`. A future QueueingModelAnalyzer (#954) will add M/M/1/K-SD with online parameter learning.

## Technical Context

**Language/Version**: Go 1.22+  
**Primary Dependencies**: No new external dependencies. All work is within `sim/cluster/` and `sim/` (existing packages).  
**Storage**: N/A (in-memory simulation state)  
**Testing**: `go test ./...` with table-driven tests; `golangci-lint run ./...` (zero tolerance)  
**Target Platform**: Library (`sim/cluster/` package); no CLI changes needed  
**Project Type**: Subsystem module (new module with own interfaces + events) per Extension Framework  
**Performance Goals**: Autoscaler pipeline executes in O(models × replicas) per tick with no simulated-time overhead; zero impact on existing test output when `ModelAutoscalerIntervalUs = 0`  
**Constraints**: `ActuationDelay = 0` must preserve INV-6 (byte-identical stdout). New config fields use `DeploymentConfig`, not `SimConfig`.  
**Scale/Scope**: 5 new files (~100–160 LOC each), 5 test files (~100–180 LOC each), ~7 modified files (additive changes only)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Architecture & Layering** | ✅ PASS | New types go in `sim/cluster/`. `RoutingSnapshot` extension goes in `sim/router_state.go` (no import cycle: `sim/` does not import `cluster/`). Autoscaler interfaces live in `sim/cluster/autoscaler.go` — cluster-level module boundary. |
| **II. Determinism** | ✅ PASS | Actuation delay uses existing `DelaySpec.Sample(rng)`. New RNG subsystem `"autoscaler"` declared in `infra_config.go`. All map iteration over `GPUInventory.ByVariant` must sort keys first (R2). |
| **III. Interface & Module Design** | ✅ PASS | Four single-method interfaces (Collector, Analyzer, Engine, Actuator). Each has ≥2 implementations shipped in this feature: Collector(Default+future), Analyzer(Saturation+Utilization+Queue), Engine(Greedy+Unlimited), Actuator(Direct+future). Query methods are pure. |
| **IV. BDD/TDD** | ✅ PASS | Each PR begins with GIVEN/WHEN/THEN behavioral contracts. Tests written before implementation. Table-driven test files per PR. No test exceeds 5s. |
| **V. Error Handling** | ✅ PASS | `Analyze()` guards all denominators against zero (R11). `PlaceInstance()` failure produces logged entry — not silently dropped (R1). `DirectActuator.Apply()` returns void; errors logged to stderr. |
| **VI. Configuration Discipline** | ✅ PASS | New fields (`ModelAutoscalerIntervalUs`, `ActuationDelay`, `ScaleUpCooldownUs`, `ScaleDownCooldownUs`) added to `DeploymentConfig` (R16), not `SimConfig`. `CostPerHour` added to `NodePoolConfig`. `DelaySpec` (existing) used for `ActuationDelay`. |
| **VII. System Invariants** | ✅ PASS | INV-6 preserved with default zero actuation delay. INV-1 extended to include `drained_dropped` terminal state. INV-A1–A7 (autoscaler-specific) documented. INV-4 preserved: GPU accounting in `gpuInventory()` counts free = total − running − loading. |
| **VIII. Antipattern Prevention** | ✅ PASS | R2: sort `ByVariant` keys. R4: grep `DeploymentConfig` and `NodePoolConfig` construction sites before adding fields. R11: Analyze zero-replica guard. R13: each interface has ≥2 implementations. R16: config in module sub-config. R19: no unbounded retry loops in Actuator. |

## Project Structure

### Documentation (this feature)

```text
specs/006-model-autoscaler/
├── plan.md              ← this file
├── research.md          ← Phase 0: resolved decisions
├── data-model.md        ← Phase 1: all types + relationships
├── contracts/
│   └── autoscaler-interfaces.md  ← Phase 1: interface contracts
├── quickstart.md        ← Phase 1: wiring guide
├── checklists/
│   └── requirements.md  ← spec quality checklist
└── tasks.md             ← Phase 2 output (next: /speckit.tasks)
```

### Source Code

```text
sim/
├── router_state.go            ← MODIFIED: add GPUType, TPDegree, CostPerHour to RoutingSnapshot

sim/cluster/
├── autoscaler.go              ← NEW (1C-1a): interfaces + types + constants
├── saturation_analyzer.go     ← NEW (1C-1b): V2SaturationAnalyzer
├── default_collector.go       ← NEW (1C-1b): DefaultCollector
├── direct_actuator.go         ← NEW (1C-1b): DirectActuator
├── engine.go                  ← NEW (1C-1b): UnlimitedEngine; (1C-1d): GreedyEngine
├── cluster_event.go           ← MODIFIED: add ScalingTickEvent + ScaleActuationEvent
├── cluster.go                 ← MODIFIED: tick handler, actuation handler, cooldown tracking, gpuInventory()
├── deployment.go              ← MODIFIED: 4 autoscaler config fields
└── infra_config.go            ← MODIFIED: add CostPerHour to NodePoolConfig

sim/cluster/ (tests)
├── autoscaler_test.go         ← NEW (1C-1a): tick scheduling, no-op determinism, INV-6
├── saturation_analyzer_test.go ← NEW (1C-1b): V2SaturationAnalyzer
├── collector_test.go           ← NEW (1C-1b): DefaultCollector
├── actuator_test.go            ← NEW (1C-1b): DirectActuator
├── engine_test.go              ← NEW (1C-1b): UnlimitedEngine; (1C-1d): GreedyEngine
└── pipeline_integration_test.go ← NEW (1C-1b): end-to-end pipeline test
```

**Structure Decision**: Single-project Go library. All new code lands in `sim/cluster/` (cluster-level domain module) following the established package layout. `sim/router_state.go` receives additive fields only. No new packages. No CLI changes.

## Complexity Tracking

No Constitution violations requiring justification.
