# PreemptionCount Routing Signal — Implementation Plan

**Goal:** Expose per-instance preemption activity as a first-class routing signal so custom scorers can factor in instance health.

**Source:** inference-sim/inference-sim#1044

**Closes:** Fixes #1044

## Behavioral Contracts

BC-1: Accessor surfaces instance metric
- GIVEN an InstanceSimulator whose cumulative preemption count is N
- WHEN `PreemptionCount()` is called on it
- THEN it returns N

BC-2: Snapshot injection respects ObservabilityConfig
- GIVEN a CachedSnapshotProvider with `ObservabilityConfig.PreemptionCount = Immediate`
- WHEN `Snapshot(id, clock)` is called at the same clock tick where other Periodic fields are stale
- THEN `snap.PreemptionCount` equals `inst.PreemptionCount()` — updated because its own FieldConfig is Immediate

BC-2b: Periodic mode goes stale correctly
- GIVEN a CachedSnapshotProvider with `ObservabilityConfig.PreemptionCount = Periodic`
- WHEN `Snapshot(id, clock)` is called before the interval elapses
- THEN `snap.PreemptionCount` returns the cached (stale) value

## Deviation Log

**DEV-1 (post-review):** BC-2 behavioral contract revised. Original design had `PreemptionCount` hardcoded as "always Immediate" outside `ObservabilityConfig`. Post-review feedback required full integration into the `ObservabilityConfig`/`fieldTimestamps`/`shouldRefresh()` framework so the signal gets the same treatment as `QueueDepth`, `BatchSize`, and `KVUtilization`. When `--snapshot-refresh-interval > 0`, `PreemptionCount` is now Periodic (matching production Prometheus scrape behavior). `DefaultObservabilityConfig()` and `newObservabilityConfig(0)` preserve Immediate behavior. The invariants.md INV-7 table and architecture.md freshness tier table updated accordingly.

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
// When ObservabilityConfig.PreemptionCount is configured as Immediate,
// Snapshot() re-reads it on every call — even at the same clock tick
// where other Periodic fields would not refresh.
// NOTE (DEV-1): original design was unconditional; post-fix it is routed through
// shouldRefresh() like other signals, with Immediate as the default.
func TestPreemptionCount_Snapshot_AlwaysImmediate(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 5

	instances := map[InstanceID]*InstanceSimulator{"inst_0": inst}
	config := ObservabilityConfig{
		QueueDepth:      FieldConfig{Mode: Periodic, Interval: 1_000_000},
		BatchSize:       FieldConfig{Mode: Periodic, Interval: 1_000_000},
		KVUtilization:   FieldConfig{Mode: Periodic, Interval: 1_000_000},
		PreemptionCount: FieldConfig{Mode: Immediate},
	}
	provider := NewCachedSnapshotProvider(instances, config)

	snap := provider.Snapshot("inst_0", 0)
	if snap.PreemptionCount != 5 {
		t.Errorf("Snapshot PreemptionCount = %d, want 5", snap.PreemptionCount)
	}

	// Advance count — must reflect on next call regardless of clock (other Periodic fields are stale)
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
PreemptionCount  int64  // Cumulative preemption events since instance start (monotonically increasing; Immediate by default, Periodic when --snapshot-refresh-interval > 0)
```

`sim/cluster/instance.go` — add after `KvTokensInUse()`:
```go
// PreemptionCount returns the cumulative number of preemption events on this instance.
func (i *InstanceSimulator) PreemptionCount() int64 {
	return i.sim.Metrics.PreemptionCount
}
```

`sim/cluster/snapshot.go` — add `PreemptionCount FieldConfig` to `ObservabilityConfig` and `fieldTimestamps`; route through `shouldRefresh()` in `Snapshot()`; set in `RefreshAll()`. See DEV-1 in Deviation Log.

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
| PreemptionCount | Instance (`InstanceSimulator.sim.Metrics.PreemptionCount`) | Immediate | Periodic | `CachedSnapshotProvider.Snapshot()` and `RefreshAll()` |
```

*(Updated per DEV-1: row now shows Immediate/Periodic tiers, not "Always Immediate".)*

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
- Integrate PreemptionCount into ObservabilityConfig/shouldRefresh() — Immediate by default, Periodic when --snapshot-refresh-interval > 0 (BC-2, DEV-1)
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
- [x] R4: Construction sites audited — `Snapshot()` routes PreemptionCount through `shouldRefresh()`; `RefreshAll()` sets and timestamps it; `AddInstance()` zero-initializes it (recovered on first `Snapshot()` call)
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
