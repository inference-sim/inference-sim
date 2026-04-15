# PreemptionCount Routing Signal — Implementation Plan

**Goal:** Expose per-instance preemption activity as a first-class routing signal so custom scorers can factor in instance health.

**Source:** inference-sim/inference-sim#1044

**Closes:** Fixes #1044

## Behavioral Contracts

BC-1: Accessor surfaces instance metric
- GIVEN an InstanceSimulator whose cumulative preemption count is N
- WHEN `PreemptionCount()` is called on it
- THEN it returns N

BC-2: Snapshot injection is always-Immediate
- GIVEN a CachedSnapshotProvider with any ObservabilityConfig
- WHEN `Snapshot(id, clock)` is called for any clock value
- THEN `snap.PreemptionCount` equals `inst.PreemptionCount()` — unconditional, not gated by any refresh interval

## Deviation Log

No clarifications needed. Issue is unambiguous.

## Tasks

---

### Task 1 — Signal wiring (BC-1, BC-2)

**Files:**
- create `sim/cluster/snapshot_preemption_test.go`
- modify `sim/routing.go`
- modify `sim/cluster/instance.go`
- modify `sim/cluster/snapshot.go`

**Test** (`sim/cluster/snapshot_preemption_test.go`):

```go
package cluster

import (
	"testing"
)

// TestPreemptionCount_Accessor_SurfacesMetric verifies BC-1:
// PreemptionCount() returns the instance's cumulative preemption count.
func TestPreemptionCount_Accessor_SurfacesMetric(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 7

	if got := inst.PreemptionCount(); got != 7 {
		t.Errorf("PreemptionCount() = %d, want 7", got)
	}
}

// TestPreemptionCount_Snapshot_AlwaysImmediate verifies BC-2:
// Snapshot() injects PreemptionCount unconditionally regardless of ObservabilityConfig.
func TestPreemptionCount_Snapshot_AlwaysImmediate(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 5

	instances := map[InstanceID]*InstanceSimulator{"inst_0": inst}
	provider := NewCachedSnapshotProvider(instances, DefaultObservabilityConfig())

	snap := provider.Snapshot("inst_0", 0)
	if snap.PreemptionCount != 5 {
		t.Errorf("Snapshot PreemptionCount = %d, want 5", snap.PreemptionCount)
	}

	// Advance count — must reflect on next call regardless of clock
	inst.sim.Metrics.PreemptionCount = 12
	snap2 := provider.Snapshot("inst_0", 0) // same clock, Periodic fields would be stale
	if snap2.PreemptionCount != 12 {
		t.Errorf("Snapshot PreemptionCount after increment = %d, want 12 (must be Immediate)", snap2.PreemptionCount)
	}
}

```

**Verify fails:**
```bash
cd .worktrees/pr-preemption-signal
go test ./sim/cluster/... -run TestPreemptionCount
# FAIL — PreemptionCount() undefined
```

**Impl:**

`sim/routing.go` — add after `InFlightRequests`:
```go
PreemptionCount  int64  // Cumulative preemption events since instance start (monotonically increasing, always Immediate)
```

`sim/cluster/instance.go` — add after `KvTokensInUse()`:
```go
// PreemptionCount returns the cumulative number of preemption events on this instance.
func (i *InstanceSimulator) PreemptionCount() int64 {
	return i.sim.Metrics.PreemptionCount
}
```

`sim/cluster/snapshot.go` — in `Snapshot()`, add immediately after `snap.ID = string(id)`:
```go
snap.PreemptionCount = inst.PreemptionCount() // always Immediate — monotonically increasing counter
```

**Verify passes:**
```bash
go test ./sim/cluster/... -run TestPreemptionCount
# PASS
go test ./sim/... ./sim/cluster/... -count=1
# all PASS
```

**Lint:**
```bash
golangci-lint run ./sim/... ./sim/cluster/...
```

---

### Task 2 — INV-7 table update (no test)

**Files:** modify `docs/contributing/standards/invariants.md`

Add row to the INV-7 signal freshness table after the `InFlightRequests` row:

```
| PreemptionCount | Instance (`InstanceSimulator.sim.Metrics.PreemptionCount`) | Always Immediate | Always Immediate | `CachedSnapshotProvider.Snapshot()` and `RefreshAll()` |
```

**Verify:**
```bash
grep -n "PreemptionCount" docs/contributing/standards/invariants.md
# shows the new row
```

---

### Final — Single commit + push + PR

```bash
git add sim/routing.go sim/cluster/instance.go sim/cluster/snapshot.go \
        sim/cluster/snapshot_preemption_test.go \
        docs/contributing/standards/invariants.md \
        docs/plans/preemption-signal-plan.md
go build ./... && go test ./... -count=1 && golangci-lint run ./...
git commit -m "feat(routing): expose PreemptionCount as a routing signal

- Add PreemptionCount() accessor on InstanceSimulator (BC-1)
- Inject PreemptionCount unconditionally in Snapshot() — always Immediate (BC-2)
- Update INV-7 signal freshness table

Closes #1044
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push -u origin pr/preemption-signal
gh pr create --title "feat(routing): expose PreemptionCount as a routing signal" \
  --body "..."
```

---

## Sanity Checklist

- [x] No unnecessary abstractions — no new interface, no new config
- [x] No feature creep — scorer excluded per scope decision
- [x] No breaking changes — `RoutingSnapshot` field addition uses named fields throughout
- [x] No hidden global state
- [x] R1: No silent continue/return
- [x] R4: Construction sites audited — `Snapshot()` (sets PreemptionCount unconditionally); `RefreshAll()` and `AddInstance()` intentionally omit it, consistent with `InFlightRequests` which is also always-Immediate and injected at a higher level
- [x] R6: No logrus.Fatalf in sim/ packages
- [x] R7: Invariant test — `TestPreemptionCount_Snapshot_AlwaysImmediate` verifies Immediate contract
- [x] R8: No exported mutable maps
- [x] R11: No runtime-derived denominators
- [x] R17: Signal freshness documented in field comment and INV-7 table
- [x] INV-7 table updated
- [x] INV-9: `PreemptionCount` is an aggregate counter — does not reveal `OutputTokens`
- [x] INV-6: `Metrics.PreemptionCount` is deterministic (driven by DES events, same seed → same count)
- [x] Task dependencies correct — Task 2 can proceed independently of Task 1
- [x] All contracts mapped to tests
