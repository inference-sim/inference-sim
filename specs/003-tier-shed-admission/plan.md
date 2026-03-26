# Implementation Plan: Phase 1B-1a — Tier-Ordered Admission Shedding

**Branch**: `003-tier-shed-admission` | **Date**: 2026-03-25 | **Spec**: [spec.md](spec.md)
**Tracking issue**: #696 | **Sub-issue**: #809

## Summary

Wire the 5 existing SLO priority tiers into the admission path so that under overload, lower-priority requests are shed before higher-priority ones. Delivered as a single PR touching 5 Go files (~61 lines of production code) and 2 new test files (~90 lines). No new interfaces are created — `TierShedAdmission` implements the existing `AdmissionPolicy` interface. Batch and Background pass through (return admitted=true) to preserve the composition point for the deferred-queue PR (#810).

## Technical Context

**Language/Version**: Go 1.22+
**Primary Dependencies**: `gopkg.in/yaml.v3` (strict YAML), `github.com/sirupsen/logrus`
**Storage**: N/A (in-memory simulation)
**Testing**: `go test ./...`, table-driven BDD/TDD per `docs/contributing/standards/principles.md`
**Target Platform**: Linux/macOS CLI
**Project Type**: Library (`sim/`) + CLI (`cmd/`)
**Performance Goals**: CPU-only, deterministic; no latency targets
**Constraints**: INV-6 (determinism), INV-9 (oracle boundary: no OutputTokens in admission), INV-1 (request conservation), R8 (unexported validation maps)

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Architecture & Layering | ✅ | `SLOTierPriority()` in `sim/`; tenant/cluster logic in `sim/cluster/`. Dependency direction correct: `sim/cluster/ → sim/`. |
| II. Determinism | ✅ | No new randomness. `shedByTier` map iteration not used for output ordering (metrics only). |
| III. Interface & Module Design | ✅ | `TierShedAdmission` satisfies existing `AdmissionPolicy` interface. No new interfaces. Single-method, stateless. |
| IV. BDD/TDD | ✅ | Tests written first and must FAIL before implementation. Table-driven unit tests + invariant test for monotonic shedding. |
| V. Error Handling | ✅ | `TierShedAdmission.Admit()` is panic-free: empty snapshots return admitted=true, unknown SLOClass returns priority 3. |
| VI. Configuration Discipline | ✅ | New fields added to `DeploymentConfig` alongside existing admission config (R16). |
| VII. System Invariants | ✅ | INV-9 explicitly enforced (SLOTierPriority receives string, not Request). INV-1 unaffected (rejected requests already counted). INV-6 unaffected (no randomness). |
| VIII. Antipattern Prevention | ✅ | R8: `validAdmissionPolicies` stays unexported. R4: `shedByTier` field added — grep confirms only one `ClusterSimulator` literal. R3: `OverloadThreshold` and `MinAdmitPriority` are non-negative int; no NaN/Inf risk. |

## Project Structure

### Documentation (this feature)

```text
specs/003-tier-shed-admission/
├── plan.md          ← this file
├── research.md      ← Phase 0: design decisions
├── data-model.md    ← Phase 1: entities
├── contracts/
│   └── tier-shed-admission.md
├── quickstart.md
├── checklists/
│   └── requirements.md
└── tasks.md         ← Phase 2 output (/speckit.tasks)
```

### Source Code

```text
sim/
├── admission.go          +35 lines: SLOTierPriority(), TierShedAdmission struct + Admit()
└── bundle.go             +2 lines: register "tier-shed" in validAdmissionPolicies

sim/cluster/
├── deployment.go         +4 lines: TierShedThreshold, TierShedMinPriority fields
├── cluster.go            +12 lines: shedByTier field + init + conditional construction
└── cluster_event.go      +8 lines: per-tier shed counter on rejection

Test files (new):
sim/
└── admission_tier_test.go        (new) unit + behavior tests for SLOTierPriority + TierShedAdmission

sim/cluster/
└── cluster_tier_test.go          (new) invariant test: monotonic shedding order
```

---

## PR: Tier-Ordered Admission Shedding (Issue #809)

**Goal**: Under overload, shed Sheddable requests before Standard, Standard before Critical. Batch/Background pass through. Per-tier shed counter emitted for capacity-planning visibility.

### Changes

#### `sim/admission.go` (~35 lines added)

```go
// SLOTierPriority maps SLOClass string to an integer priority.
// Higher = more important. Background=0 … Critical=4.
// Empty or unknown string maps to Standard (3) for backward compatibility.
// Exported so sim/cluster/ can call it without a circular import.
func SLOTierPriority(class string) int {
    switch class {
    case "critical":   return 4
    case "standard":   return 3
    case "sheddable":  return 2
    case "batch":      return 1
    case "background": return 0
    default:           return 3  // empty or unknown → Standard
    }
}

// TierShedAdmission sheds lower-priority requests under overload.
// Stateless: all decisions computed from RouterState at call time.
// Batch and Background always pass through (deferred queue PR handles them).
type TierShedAdmission struct {
    OverloadThreshold int // max per-instance effective load before shedding; 0 = any load triggers
    MinAdmitPriority  int // minimum tier priority admitted under overload (default: 3 = Standard)
}

func (t *TierShedAdmission) Admit(req *Request, state *RouterState) (bool, string) {
    // Batch/Background bypass tier-shed (deferred queue handles them in PR-2).
    class := req.SLOClass
    if class == "batch" || class == "background" {
        return true, ""
    }
    // Compute max effective load across all instance snapshots.
    maxLoad := 0
    for _, snap := range state.Snapshots {
        if l := snap.EffectiveLoad(); l > maxLoad {
            maxLoad = l
        }
    }
    if maxLoad <= t.OverloadThreshold {
        return true, "" // under threshold: admit all
    }
    // Under overload: reject tiers below MinAdmitPriority.
    priority := SLOTierPriority(class)
    if priority < t.MinAdmitPriority {
        return false, fmt.Sprintf("tier-shed: class=%s priority=%d < min=%d load=%d",
            class, priority, t.MinAdmitPriority, maxLoad)
    }
    return true, ""
}
```

#### `sim/bundle.go` (~2 lines changed)

Add `"tier-shed": true` to `validAdmissionPolicies`. Do NOT add a `case "tier-shed":` to `NewAdmissionPolicy()` — the factory signature `(name string, capacity, refillRate float64)` cannot carry the int parameters needed by `TierShedAdmission`. The struct is constructed directly in `cluster.go` instead.

#### `sim/cluster/deployment.go` (~4 lines added)

```go
// Phase 1B-1a: tier-ordered admission shedding config (issue #809).
// Zero value is safe: TierShedMinPriority=0 admits all tiers (same as AlwaysAdmit),
// but callers should explicitly set 3 (Standard) for meaningful protection.
TierShedThreshold   int `yaml:"tier_shed_threshold,omitempty"`
TierShedMinPriority int `yaml:"tier_shed_min_priority,omitempty"`
```

#### `sim/cluster/cluster.go` (~12 lines)

Add `shedByTier map[string]int` to `ClusterSimulator` struct (alongside `rejectedRequests`):

```go
shedByTier map[string]int // per-SLOClass rejection counts (Phase 1B-1a)
```

In `NewClusterSimulator()`, before the struct literal, detect `"tier-shed"` and build the policy directly:

```go
// Bypass generic factory for "tier-shed": factory signature is float64-only.
var admissionPolicy sim.AdmissionPolicy
if config.AdmissionPolicy == "tier-shed" {
    admissionPolicy = &sim.TierShedAdmission{
        OverloadThreshold: config.TierShedThreshold,
        MinAdmitPriority:  config.TierShedMinPriority,
    }
} else {
    admissionPolicy = sim.NewAdmissionPolicy(
        config.AdmissionPolicy,
        config.TokenBucketCapacity,
        config.TokenBucketRefillRate,
    )
}
```

In the struct literal, use `admissionPolicy` variable and add `shedByTier: make(map[string]int)`.

#### `sim/cluster/cluster_event.go` (~8 lines)

In `AdmissionDecisionEvent.Execute()`, after `cs.rejectedRequests++`:

```go
if cs.shedByTier != nil {
    tier := e.request.SLOClass
    if tier == "" {
        tier = "standard" // normalize empty → standard (matches SLOTierPriority default)
    }
    cs.shedByTier[tier]++
}
```

### Tests (BDD/TDD — write first, must FAIL before implementation)

#### `sim/admission_tier_test.go` (new file)

| Test | Type | Scenario |
|------|------|----------|
| `TestSLOTierPriority` | Unit/table | All 5 canonical classes + empty + unknown string |
| `TestTierShedAdmission_CriticalAndStandardAlwaysAdmitted` | Behavior | Critical and Standard return `(true,"")` under overload |
| `TestTierShedAdmission_SheddableRejectedUnderOverload` | Behavior | Sheddable returns `(false, reason)` when maxLoad > threshold |
| `TestTierShedAdmission_BatchAlwaysAdmitted` | Behavior | Batch returns `(true,"")` regardless of load |
| `TestTierShedAdmission_BackgroundAlwaysAdmitted` | Behavior | Background returns `(true,"")` regardless of load |
| `TestTierShedAdmission_UnderThreshold` | Behavior | No request shed when maxLoad ≤ threshold |
| `TestTierShedAdmission_EmptySnapshots` | Behavior | Zero instances → no panic, all admitted |
| `TestTierShedAdmission_EmptySLOClass` | Behavior | Empty SLOClass treated as Standard, never shed below Standard |

#### `sim/cluster/cluster_tier_test.go` (new file)

| Test | Type | Invariant |
|------|------|-----------|
| `TestTierShedMonotonicOrder` | Invariant | Under load ramp: `shed(Sheddable) ≥ shed(Standard) ≥ shed(Critical)` at every measurement point |
| `TestTierShedNoRegressionWithoutPolicy` | Invariant | Simulation without `tier-shed` produces byte-identical output to pre-feature baseline (INV-6) |

---

## PR Ordering and Dependencies

This is PR-1 in the Phase 1B sequence. It has no dependencies on other Phase 1B PRs. PRs #810, #811, and #812 all depend on this PR being merged first (they use `SLOTierPriority()` and the `shedByTier` field).

## Complexity Tracking

No constitution violations. `TierShedAdmission` is a Policy Template extension (new algorithm behind frozen `AdmissionPolicy` interface). All changes are additive — no existing behavior is modified.
