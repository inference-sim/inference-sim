# Implementation Plan: Phase 1B — Service Tiers & Tenant Fairness

**Branch**: `002-tier-tenant-fairness` | **Date**: 2026-03-23 | **Spec**: [spec.md](spec.md)

## Summary

Wire the existing `SLOClass` and `TenantID` fields on every `Request` into the admission path. Delivered as 4 sequential PRs, each touching 1–3 Go files and resolving one GitHub issue. No new interfaces are created — the existing `AdmissionPolicy` interface is extended with one new policy, and tenant enforcement is added as cluster-level logic in the event handler.

## Technical Context

**Language/Version**: Go 1.22+
**Primary Dependencies**: `gopkg.in/yaml.v3` (strict YAML), `github.com/sirupsen/logrus`
**Storage**: N/A (in-memory simulation)
**Testing**: `go test ./...`, table-driven BDD/TDD per `docs/contributing/standards/principles.md`
**Target Platform**: Linux/macOS CLI
**Project Type**: Library (`sim/`) + CLI (`cmd/`)
**Performance Goals**: CPU-only, deterministic; no latency targets
**Constraints**: INV-6 (determinism), INV-9 (oracle boundary), INV-1 (conservation), R8 (no exported mutable maps)

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Architecture & Layering | ✅ | `sim/` never imports `sim/cluster/`; tier logic stays in `sim/`; tenant logic stays in `sim/cluster/` |
| II. Determinism | ✅ | No new randomness; map iterations use sorted keys (existing pattern) |
| III. Interface Design | ✅ | `TierShedAdmission` satisfies existing `AdmissionPolicy` interface; no new interfaces |
| IV. BDD/TDD | ✅ | Every PR has invariant tests (monotonic shedding, conservation) alongside behavior tests |
| V. Error Handling | ✅ | New config fields validated in constructors; zero-value safe throughout |
| VI. Configuration Discipline | ✅ | New fields added to `DeploymentConfig` and `AdmissionConfig` — grouped by module (R16) |
| VII. System Invariants | ✅ | INV-1 extended to include deferred queue; INV-9 explicitly enforced in TierShedAdmission |
| VIII. Antipattern Prevention | ✅ | R8: no exported mutable map fields; R4: single canonical constructor per new struct |

## Project Structure

### Documentation (this feature)

```text
specs/002-tier-tenant-fairness/
├── plan.md          ← this file
├── research.md      ← Phase 0: design decisions
├── data-model.md    ← Phase 1: entities and state transitions
├── contracts/       ← Phase 1: interface contracts
│   ├── tier-shed-admission.md
│   └── tenant-tracker.md
├── quickstart.md    ← Phase 1: how to use the new features
└── tasks.md         ← Phase 2 output (/speckit.tasks — not created here)
```

### Source Code

```text
sim/
├── admission.go          PR-1: add SLOTierPriority(), TierShedAdmission
└── bundle.go             PR-1: register "tier-shed" policy name

sim/cluster/
├── cluster.go            PR-2: deferredQueue field + promoteDeferred()
│                         PR-3: tenantTracker field + wiring
├── cluster_event.go      PR-1: per-tier shed counter
│                         PR-2: Batch/Background pre-admission intercept
│                         PR-3: tenant budget override after admission
├── deployment.go         PR-1: TierShedThreshold, TierShedMinPriority config fields
│                         PR-3: TenantBudgets config field
├── metrics.go            PR-2: DeferredHorizonInterrupted in RawMetrics
│                         PR-4: ComputePerTenantMetrics() + TenantMetrics struct
└── tenant.go             PR-3: TenantTracker (new file)

cmd/root.go               PR-4: call ComputePerTenantMetrics + printPerTenantMetrics
```

---

## PR-1: Tier-Ordered Admission Shedding (Issue #809)

**Goal**: Under overload, shed Sheddable requests before Standard, Standard before Critical. Batch/Background pass through (handled by PR-2).

### Changes

#### `sim/admission.go` (~40 lines added)

```go
// SLOTierPriority maps SLOClass string to an integer priority.
// Higher = more important. Background=0 … Critical=4.
// Empty string maps to Standard (3) for backward compatibility.
// Exported so sim/cluster/ can call it without a circular import.
func SLOTierPriority(class string) int {
    switch class {
    case "critical":   return 4
    case "standard":   return 3
    case "sheddable":  return 2
    case "batch":      return 1
    case "background": return 0
    default:           return 3  // treat unknown/empty as Standard
    }
}

// TierShedAdmission sheds lower-priority requests under overload.
// When max effective load across instances >= OverloadThreshold,
// requests with tier priority < MinAdmitPriority are rejected.
// Batch and Background are always passed through (deferred queue handles them).
// OverloadThreshold=0 means: shed when any instance has load > 0.
type TierShedAdmission struct {
    OverloadThreshold int    // max effective load before shedding kicks in
    MinAdmitPriority  int    // minimum tier priority admitted under overload (default: 3 = Standard)
}

func (t *TierShedAdmission) Admit(req *sim.Request, state *sim.RouterState) (bool, string) {
    // Batch and Background bypass tier-shed (handled separately as deferred).
    class := req.SLOClass
    if class == "batch" || class == "background" {
        return true, ""
    }
    // Check overload signal.
    maxLoad := 0
    for _, snap := range state.Snapshots {
        if l := snap.EffectiveLoad(); l > maxLoad {
            maxLoad = l
        }
    }
    if maxLoad <= t.OverloadThreshold {
        return true, ""  // under threshold: admit all
    }
    // Under overload: reject tiers below MinAdmitPriority.
    if SLOTierPriority(class) < t.MinAdmitPriority {
        return false, fmt.Sprintf("tier-shed: tier=%s priority=%d < min=%d",
            class, SLOTierPriority(class), t.MinAdmitPriority)
    }
    return true, ""
}
```

#### `sim/bundle.go` (~2 lines changed)

- Add `"tier-shed": true` to `validAdmissionPolicies` (validation only)
- Do NOT add a `case "tier-shed":` to `NewAdmissionPolicy()` — the factory signature `(name string, capacity, refillRate float64)` cannot carry the int parameters needed by `TierShedAdmission`. The policy is constructed directly in `cluster.go` instead (see below).

#### `sim/cluster/deployment.go` (~8 lines changed)

```go
// Phase 1B: tier-ordered admission shedding config.
TierShedThreshold    int `yaml:"tier_shed_threshold,omitempty"`    // default 0
TierShedMinPriority  int `yaml:"tier_shed_min_priority,omitempty"` // default 3 (Standard)
```

#### `sim/cluster/cluster.go` (PR-1 addition, ~8 lines in `NewClusterSimulator()`)

```go
// Bypass the generic factory for tier-shed: it needs int params the factory doesn't carry.
if config.AdmissionPolicy == "tier-shed" {
    cs.admissionPolicy = &sim.TierShedAdmission{
        OverloadThreshold: config.TierShedThreshold,
        MinAdmitPriority:  config.TierShedMinPriority,
    }
}
```

This block runs after the existing `sim.NewAdmissionPolicy(...)` call and overwrites the result when the policy is `"tier-shed"`.

#### `sim/cluster/cluster_event.go` (~12 lines changed)

In `AdmissionDecisionEvent.Execute()`, after `cs.rejectedRequests++`:
```go
if cs.shedByTier != nil {
    tier := e.request.SLOClass
    if tier == "" { tier = "standard" }
    cs.shedByTier[tier]++
}
```
Add `shedByTier map[string]int` field to `ClusterSimulator` in `cluster.go`.

### Tests (BDD/TDD)

| Scenario | Type | Invariant |
|----------|------|-----------|
| Critical admitted when Sheddable shed | Behavior | Monotonic shedding order |
| Same-tier requests: no extra shedding at threshold=0 | Behavior | Zero-value safe |
| `sloTierPriority("")` returns Standard (3) | Unit | Backward compat |
| Empty SLOClass → treated as Standard | Behavior | R/W parity |
| INV-9: `req.OutputTokens` never read | Structural | Oracle boundary |

---

## PR-2: Deferred Queue for Batch/Background (Issue #810)

**Goal**: Batch and Background requests park in a deferred queue when the cluster is busy. They are promoted when all instance queues drain.

**Depends on**: PR-1 merged.

### Changes

#### `sim/cluster/cluster.go` (~45 lines)

```go
// New fields on ClusterSimulator:
deferredQueue []*sim.Request  // Batch/Background requests awaiting idle capacity

// New method:
// isBusy returns true when any instance has non-zero effective load.
// Matches the three-component definition: QueueDepth + BatchSize + InFlightRequests > 0.
func (c *ClusterSimulator) isBusy() bool {
    for _, inst := range c.instances {
        if inst.QueueDepth()+inst.BatchSize()+c.inFlightRequests[string(inst.ID())] > 0 {
            return true
        }
    }
    return false
}

// promoteDeferred injects all deferred requests as ClusterArrivalEvents
// at the current clock. Called when isBusy() transitions from true to false.
func (c *ClusterSimulator) promoteDeferred() {
    for _, req := range c.deferredQueue {
        heap.Push(&c.clusterEvents, clusterEventEntry{
            event: &ClusterArrivalEvent{time: c.clock, request: req},
            seqID: c.nextSeqID(),
        })
    }
    c.deferredQueue = c.deferredQueue[:0]
}
```

In `Run()`, after processing all events for a tick, add idle-capacity check:
```go
if len(c.deferredQueue) > 0 && !c.isBusy() {
    c.promoteDeferred()
}
```

#### `sim/cluster/cluster_event.go` (~20 lines)

In `AdmissionDecisionEvent.Execute()`, before calling `admissionPolicy.Admit()`:
```go
if (e.request.SLOClass == "batch" || e.request.SLOClass == "background") && cs.isBusy() {
    cs.deferredQueue = append(cs.deferredQueue, e.request)
    return  // deferred, not admitted, not rejected
}
```

#### `sim/cluster/metrics.go` (~15 lines)

Add `DeferredHorizonInterrupted int` to `RawMetrics`. Populate in `CollectRawMetrics()` from `len(cs.deferredQueue)` at horizon (passed as a new parameter or accessor).

Add invariant comment:
```
// INV-1 extended: injected == completed + running + queued + shed + dropped + deferred_horizon_interrupted
```

### Tests (BDD/TDD)

| Scenario | Type | Invariant |
|----------|------|-----------|
| Batch request during busy cluster → enters deferred queue | Behavior | |
| Deferred request promoted after all queues drain | Behavior | |
| Deferred request NOT promoted while queues non-empty | Behavior | |
| INV-1 holds with non-empty deferred queue at horizon | Invariant | Conservation |
| No deferred logic when SLOClass != batch/background | Behavior | Zero-value safe |
| INV-8 preserved: promoting deferred doesn't stall running work | Invariant | Work-conserving |

---

## PR-3: Per-Tenant Fair-Share Tracking and Enforcement (Issue #811)

**Goal**: Track per-tenant in-flight request counts. When a tenant exceeds their configured fair-share, preferentially shed their Sheddable-and-below requests.

**Depends on**: PR-1 merged.

### Changes

#### `sim/cluster/tenant.go` (new file, ~60 lines)

```go
// TenantTracker tracks in-flight request counts per tenant and enforces fair-share budgets.
// Zero-value is safe: when no budgets are configured, IsOverBudget always returns false.
type TenantTracker struct {
    budgets      map[string]float64  // tenantID → fraction of totalCapacity (0 = unlimited)
    inFlight     map[string]int      // tenantID → current in-flight count
    totalCapacity int                // max in-flight slots across cluster
}

// NewTenantTracker creates a TenantTracker from a budget map and cluster capacity.
// budgets may be nil (unlimited for all tenants).
func NewTenantTracker(budgets map[string]float64, totalCapacity int) *TenantTracker

// IsOverBudget returns true when tenantID has a configured budget and currently exceeds it.
// Always returns false when tenantID is empty or has no configured budget.
func (t *TenantTracker) IsOverBudget(tenantID string) bool

// OnStart increments the in-flight count for tenantID.
func (t *TenantTracker) OnStart(tenantID string)

// OnComplete decrements the in-flight count for tenantID (floor 0).
func (t *TenantTracker) OnComplete(tenantID string)
```

#### `sim/cluster/deployment.go` (~10 lines)

```go
// Phase 1B: per-tenant fair-share budgets (optional — nil = unlimited for all tenants).
// Key: TenantID string. Value: fraction of total cluster capacity (0.0–1.0).
// Zero-value safe: no enforcement when nil.
TenantBudgets map[string]float64 `yaml:"tenant_budgets,omitempty"`
```

#### `sim/cluster/cluster.go` (~25 lines)

- Add `tenantTracker *TenantTracker` field to `ClusterSimulator`
- In `NewClusterSimulator()`, when `config.TenantBudgets != nil`: `cs.tenantTracker = NewTenantTracker(config.TenantBudgets, totalCapacity)`
- Call `cs.tenantTracker.OnStart(req.TenantID)` when request is dispatched to an instance
- Call `cs.tenantTracker.OnComplete(req.TenantID)` in the step-completion callback

#### `sim/cluster/cluster_event.go` (~15 lines)

In `AdmissionDecisionEvent.Execute()`, after admission policy returns `admitted=true`:
```go
if cs.tenantTracker != nil && cs.tenantTracker.IsOverBudget(e.request.TenantID) {
    tier := sim.SLOTierPriority(e.request.SLOClass)
    if tier < 3 {  // below Standard
        admitted = false
        reason = "tenant-budget-exceeded"
        cs.rejectedRequests++
        if cs.shedByTier != nil { cs.shedByTier[e.request.SLOClass]++ }
        return
    }
}
```

### Tests (BDD/TDD)

| Scenario | Type | Invariant |
|----------|------|-----------|
| Over-budget tenant's Sheddable shed; on-budget tenant's Sheddable admitted | Behavior | |
| Over-budget tenant's Critical/Standard NOT shed | Behavior | Tier protection |
| Empty TenantID → never over-budget | Behavior | Zero-value safe |
| `TenantBudgets: nil` → byte-identical output vs baseline | Invariant | INV-6 determinism |
| Two tenants, equal budget, equal load → both admitted | Behavior | |

---

## PR-4: Per-Tenant Jain Fairness Metrics (Issue #812)

**Goal**: Emit `per_tenant` section in simulation output with completed count, tokens served, and Jain fairness index.

**Depends on**: PR-3 merged.

### Changes

#### `sim/cluster/metrics.go` (~40 lines)

```go
// TenantMetrics holds per-tenant throughput metrics for JSON output.
type TenantMetrics struct {
    TenantID          string  `json:"tenant_id"`
    CompletedRequests int     `json:"completed_requests"`
    TotalOutputTokens int     `json:"total_output_tokens"`
}

// ComputePerTenantMetrics groups completed requests by TenantID and computes token totals.
// Returns nil when no requests carry a TenantID (zero-value safe — no per_tenant section).
// Uses JainFairnessIndex() from this package for the cluster-level fairness index.
func ComputePerTenantMetrics(aggregated *sim.Metrics) (map[string]*TenantMetrics, float64)
```

Implementation: walk `aggregated.Requests`, group by `TenantID`; for each group sum `NumDecodeTokens`; compute Jain index over token-total distribution. Return `nil, 0` when all TenantIDs are empty.

#### `cmd/root.go` (~25 lines)

Follow the `printPerModelMetrics` pattern:
```go
perTenantMetrics, jainIndex := cluster.ComputePerTenantMetrics(cs.AggregatedMetrics())
printPerTenantMetrics(os.Stdout, perTenantMetrics, jainIndex)
```

`printPerTenantMetrics` is a no-op when `perTenantMetrics == nil`.

### Tests (BDD/TDD)

| Scenario | Type | Invariant |
|----------|------|-----------|
| Two tenants, balanced load → Jain ≥ 0.99 | Behavior | |
| Two tenants, 10:1 skew → Jain < 0.70 | Behavior | |
| No TenantID set → `ComputePerTenantMetrics` returns nil | Behavior | Zero-value safe |
| Jain index matches manual calculation within 0.001 | Invariant | Metric accuracy |

---

## PR Ordering and Dependencies

```
PR-1 (#809) ──► PR-2 (#810)
              ──► PR-3 (#811) ──► PR-4 (#812)
```

PR-2 and PR-3 can be merged in either order once PR-1 is merged (they touch different code paths).

## Complexity Tracking

No constitution violations. All changes follow existing patterns (policy template extension, cluster field addition, metrics computation). No new interfaces, no new packages.
