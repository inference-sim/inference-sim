# Implementation Plan: Per-Tenant Jain Fairness Index

**Branch**: `005-tenant-jain-fairness` | **Date**: 2026-03-30 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/005-tenant-jain-fairness/spec.md`

## Summary

Add `ComputePerTenantMetrics` to `sim/cluster/metrics.go` following the existing `ComputePerModelMetrics` pattern, then wire `printPerTenantMetrics` into both `cmd/root.go` and `cmd/replay.go`. `JainFairnessIndex` is already implemented — this PR only adds the aggregation function and the print call. Expected change: ~60 lines across 3 production files.

## Technical Context

**Language/Version**: Go 1.22+
**Primary Dependencies**: `sim/cluster` (metrics), `sim` (Metrics, RequestMetrics types), `cobra`, `logrus`
**Storage**: N/A — all in-memory, post-simulation aggregation
**Testing**: `go test ./...`, table-driven tests, BDD behavioral contracts
**Target Platform**: Linux/macOS, CPU-only
**Project Type**: CLI tool (`cmd/`) wrapping library (`sim/cluster/`)
**Performance Goals**: No new performance requirements — pure aggregation over already-materialized `Requests` map
**Constraints**:
- Dependency direction: `cmd/ → sim/cluster/ → sim/`. Embedding per-tenant data in `MetricsOutput` JSON is architecturally blocked (`sim/` cannot import `sim/cluster/`).
- INV-6 determinism: keys MUST be sorted before iteration (R2).
- Zero-value safe: nil input or empty Requests map → nil return (no section printed).
**Scale/Scope**: ~40-60 lines new code; 3 files touched (1 new test file)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Architecture & Layering** | ✅ PASS | `ComputePerTenantMetrics` lives in `sim/cluster/` (Layer 2). Printing in `cmd/` (Layer 3). No import cycle introduced. |
| **II. Determinism** | ✅ PASS | Tenant keys sorted before iteration (R2). Same seed → same output. |
| **III. Interface Design** | ✅ PASS | Pure function returning a value map — no new interface needed. No hidden state. |
| **IV. BDD/TDD** | ✅ PASS | All TDD tasks below have GIVEN/WHEN/THEN contracts. Tests written before implementation. |
| **V. Error Handling** | ✅ PASS | No error paths: returns nil for empty input. `JainFairnessIndex` division guard already present (returns 1.0 for all-zero). |
| **VI. Configuration Discipline** | ✅ PASS | No new config params. No `SimConfig` changes. |
| **VII. System Invariants** | ✅ PASS | INV-6: sorted keys. No other invariants touched. |
| **VIII. Antipattern Prevention** | ✅ PASS | R2 (sort keys), R11 (zero denominator guard already in `JainFairnessIndex`), R4 (no new struct construction sites beyond the new function itself), R20 (handle empty input explicitly). |

No violations. Complexity Tracking table omitted.

## Project Structure

### Documentation (this feature)

```text
specs/005-tenant-jain-fairness/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/
│   └── per-tenant-output.md   ← Phase 1 output
└── tasks.md             ← Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code

```text
sim/cluster/
├── metrics.go                  MODIFY — add TenantMetrics struct + ComputePerTenantMetrics (~40 lines)
├── metrics_tenant_test.go      NEW    — behavioral tests for ComputePerTenantMetrics (BC-T1–BC-T5)

cmd/
├── root.go                     MODIFY — add printPerTenantMetrics + wire after printPerModelMetrics (~20 lines)
├── replay.go                   MODIFY — wire printPerTenantMetrics after printPerModelMetrics (~5 lines)
├── kv_metrics_output_test.go   MODIFY — add printPerTenantMetrics nil/empty guard tests (BC-T6–BC-T7)
```

**Structure Decision**: Single-project Go repo. New code slots into existing `sim/cluster/metrics.go` and `cmd/root.go` following the established `ComputePerModelMetrics` / `printPerModelMetrics` pattern verbatim.
