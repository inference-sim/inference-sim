# Simulation Progress Hook Implementation Plan

**Goal:** Add an opt-in, read-only progress hook to `Simulator` and `ClusterSimulator` that delivers periodic state snapshots during simulation execution, enabling interactive tool integration.

**The problem today:** BLIS exports simulation results only after completion. There is no way to observe internal state mid-run, which blocks integration with interactive tools (dashboards, live monitoring).

**What this PR adds:**
1. `ProgressHook` — a single-method interface in `sim/` that receives `ProgressSnapshot` values at configurable clock intervals
2. Clock-interval triggering — snapshots fire when the simulation clock crosses the next interval boundary, after all events at that timestamp have drained
3. Final-snapshot delivery — a terminal snapshot (`IsFinal: true`) fires immediately before `Run()` returns, regardless of whether the simulation ran to completion or was interrupted by the horizon
4. Cluster-mode support — `ClusterSimulator` delivers snapshots with per-instance detail populated from existing `InstanceSimulator` query methods

**Why this matters:** Enables real-time simulation monitoring for interactive capacity planning tools without post-hoc replay.

**Architecture:** New types in `sim/progress_hook.go` (interface + snapshot structs). Hook wiring in `sim/simulator.go` (single-instance `Run()` loop) and `sim/cluster/cluster.go` (cluster `Run()` loop). Snapshot construction reuses existing `InstanceSimulator` query methods (same data as `RoutingSnapshot` but different audience — progress monitoring vs routing decisions).

**Source:** GitHub issue #1162 + follow-up review comment addressing 8 design gaps.
**Closes:** Fixes #1162

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** New subsystem module — a progress observation hook injected into the simulation event loop.
2. **Adjacent blocks:** `Simulator` (event loop owner), `ClusterSimulator` (cluster event loop), `InstanceSimulator` (per-instance state queries), `Metrics` (completion/token counters).
3. **Invariants touched:** INV-6 (determinism) — the hook must not affect event ordering or stdout. INV-3 (clock monotonicity) — snapshot clock values must be non-decreasing.
4. **Construction site audit:**
   - `Simulator` struct (`sim/simulator.go:147`): single construction site in `NewSimulator()`. New fields added here.
   - `ClusterSimulator` struct (`sim/cluster/cluster.go`): single construction site in `NewClusterSimulator()`. New fields added here.
   - No new struct fields on `InstanceSimulator` — we use existing query methods.

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds a `ProgressHook` interface to BLIS that delivers periodic `ProgressSnapshot` values during simulation execution. The hook fires at configurable clock-interval boundaries (checked after all events at a given timestamp drain), plus a mandatory final snapshot before `Run()` returns. The interface lives in `sim/` alongside existing callback patterns (`OnRequestDone`). Cluster mode populates per-instance snapshots using existing `InstanceSimulator` query methods — the same data exposed by `CachedSnapshotProvider`/`RoutingSnapshot` but served through a different mechanism (push-based progress vs pull-based routing). The hook is strictly read-only and opt-in (`nil` = zero overhead).

### B) Behavioral Contracts

**Positive contracts (what MUST happen):**

BC-1: Clock-interval snapshot delivery
- GIVEN a `ProgressHook` is registered with `intervalUs > 0`
- WHEN the simulation clock crosses the next interval boundary
- THEN `OnProgress` is called with a snapshot whose `Clock` matches the current simulation time
- MECHANISM: After processing each event/step, check `clock >= nextSnapshotTime`; if so, build and deliver snapshot, advance `nextSnapshotTime` by `intervalUs`

BC-2: Final snapshot delivery
- GIVEN a `ProgressHook` is registered
- WHEN `Run()` is about to return (normal completion or horizon break)
- THEN `OnProgress` is called exactly once with `IsFinal: true`
- MECHANISM: Unconditional final snapshot call after event loop exits, before `Finalize()` returns / `Run()` returns

BC-3: Cluster-mode per-instance population
- GIVEN a `ProgressHook` is registered on `ClusterSimulator`
- WHEN a snapshot fires
- THEN `InstanceSnapshots` contains one entry per active instance with correct queue depth, batch size, KV utilization, and lifecycle state
- MECHANISM: Iterate `cs.instances`, call existing query methods (`QueueDepth()`, `BatchSize()`, `KVUtilization()`, etc.)

BC-4: Single-instance snapshot population
- GIVEN a `ProgressHook` is registered on `Simulator` (single-instance mode)
- WHEN a snapshot fires
- THEN `InstanceSnapshots` contains exactly one entry reflecting the single simulator's state
- MECHANISM: Build one `InstanceSnapshot` from `sim.QueueDepth()`, `sim.BatchSize()`, `sim.KVCache` methods

**Negative contracts (what MUST NOT happen):**

BC-5: Determinism preservation (INV-6)
- GIVEN a simulation with seed S and a nil `ProgressHook`
- WHEN the same simulation runs with seed S and a non-nil `ProgressHook`
- THEN stdout is byte-identical between both runs
- MECHANISM: Hook is read-only, called synchronously, does not modify simulation state, does not enqueue events

BC-6: Zero overhead when disabled
- GIVEN no `ProgressHook` is registered (nil)
- WHEN the simulation runs
- THEN no snapshot construction or callback invocation occurs — zero performance impact
- MECHANISM: `nil` guard before any snapshot-related logic in the event loop

BC-7: Read-only snapshot contract
- GIVEN a `ProgressHook` is registered
- WHEN `OnProgress` is called
- THEN the snapshot is a value type (fully copied) — modifications by the consumer cannot affect simulation state
- MECHANISM: `ProgressSnapshot` and `InstanceSnapshot` are structs with no pointer fields; slice of `InstanceSnapshot` is freshly allocated per call

**Error handling contracts:**

BC-8: Final snapshot on all exit paths
- GIVEN a `ProgressHook` is registered
- WHEN the event loop exits (empty queue, horizon break, or error in cluster mode)
- THEN the final snapshot fires with `IsFinal: true` exactly once
- MECHANISM: Single call site after event loop, guarded by `hook != nil`

### C) Component Interaction

```
cmd/root.go / cmd/replay.go
    │
    ├─ SetProgressHook(hook, intervalUs)  [optional, before Run()]
    │
    ▼
ClusterSimulator.Run()
    │
    ├─ event loop: after each cluster-event or instance-event
    │   └─ if hook != nil && clock >= nextSnapshotTime:
    │       ├─ build ProgressSnapshot from cluster state
    │       ├─ for each instance: build InstanceSnapshot via query methods
    │       └─ hook.OnProgress(snapshot)
    │
    └─ after loop: deliver final snapshot (IsFinal: true)

Simulator.Run()  [single-instance path]
    │
    ├─ event loop: after each ProcessNextEvent()
    │   └─ if hook != nil && clock >= nextSnapshotTime:
    │       ├─ build ProgressSnapshot from sim state
    │       └─ hook.OnProgress(snapshot)
    │
    └─ after loop: deliver final snapshot (IsFinal: true)
```

**Data flow:** Snapshot construction reads (never writes) existing metrics counters and instance query methods. The `InstanceSnapshot` fields map 1:1 to existing `InstanceSimulator` accessor methods.

**Relationship to CachedSnapshotProvider:** Both read the same underlying instance state, but serve different purposes. `CachedSnapshotProvider` is a pull-based cache for routing decisions (with configurable staleness). `ProgressHook` is a push-based notification for external consumers (always fresh at delivery time). They share no state and operate independently.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| `SimulationObserver` interface name | `ProgressHook` interface name | CLARIFICATION — review comment #1 flagged collision with `DisaggregationObserver`. `ProgressHook` avoids the "Observer" suffix entirely. |
| "Snapshot boundaries" (undefined) | Clock-interval triggering (`intervalUs`) | CLARIFICATION — review comment #2. Interval-based rather than per-step or per-event; configurable granularity avoids perf issues. |
| `StateSnapshot` type name | `ProgressSnapshot` type name | CLARIFICATION — aligns with `ProgressHook` naming; avoids collision with `RoutingSnapshot`. |
| `SetObserver(obs, intervalUs)` | `SetProgressHook(hook ProgressHook, intervalUs int64)` | CLARIFICATION — follows renaming. |
| No explicit nil contract | BC-6: `nil` = zero overhead, zero behavior change | ADDITION — review comment #4. |
| "Blocking callbacks are consumer issues" | BC-5/BC-7: hook must be read-only, no event enqueuing | CORRECTION — review comment #5. INV-6 proof requires explicit contract. |
| `IsFinal` semantics undefined | BC-8: fires once before `Run()` returns, all exit paths | CLARIFICATION — review comment #6. |
| No cross-path coverage statement | Plan notes: covers `blis run` and `blis replay` (both use `ClusterSimulator.Run()`); not `blis observe` | CLARIFICATION — review comment #7. |
| New package suggested in issue | Types in `sim/progress_hook.go`, no new package | SIMPLIFICATION — single file in existing `sim/` package sufficient; avoids import cycle risk. |

### E) Review Guide

**Tricky part:** The snapshot timing in the cluster event loop — the check must fire after both cluster-event and instance-event processing branches, and must not fire for cancelled/orphaned `TimeoutEvent` events (where clock was restored). The clock-restore pattern at cluster.go:580-581 means we must check `c.clock >= nextSnapshotTime` after the event processing, not before.

**Scrutinize:** BC-5 (determinism) — verify no state mutation in snapshot construction. BC-6 (zero overhead) — verify nil guard placement catches all paths.

**Safe to skim:** Snapshot struct field population — it's mechanical reads from existing accessors.

**Known debt:** The snapshot does not include per-request detail (deferred per issue #1162 alternative #3).

---

## Part 2: Executable Implementation

### F) Implementation Overview

| Action | File |
|--------|------|
| Create | `sim/progress_hook.go` — interface, snapshot types, builder helpers |
| Create | `sim/progress_hook_test.go` — all tests |
| Modify | `sim/simulator.go` — `SetProgressHook()`, hook check in `Run()`, final snapshot in `Finalize()` path |
| Modify | `sim/cluster/cluster.go` — `SetProgressHook()`, hook check in `Run()`, final snapshot after loop |

No dead code. No new CLI flags (library-level hook only).

### G) Task Breakdown

#### Task 1: Define ProgressHook interface and snapshot types (BC-7)

**Files:** create `sim/progress_hook.go`, test `sim/progress_hook_test.go`

**Test:**
```go
// sim/progress_hook_test.go
package sim

import "testing"

func TestProgressSnapshot_IsValueType(t *testing.T) {
    // BC-7: Verify snapshot is a value type — modifications don't affect source
    snap := ProgressSnapshot{
        Clock:          1000,
        TotalCompleted: 5,
        InstanceSnapshots: []InstanceSnapshot{
            {ID: "inst-0", QueueDepth: 3, BatchSize: 2},
        },
    }
    // Copy and mutate
    copy := snap
    copy.TotalCompleted = 99
    copy.InstanceSnapshots[0].QueueDepth = 999

    if snap.TotalCompleted == 99 {
        t.Error("ProgressSnapshot scalar fields are not value-copied")
    }
    // Note: slice header is copied but underlying array is shared — this is expected.
    // The contract requires fresh slice allocation per OnProgress call (tested in Task 3).
}
```

**Impl:**
```go
// sim/progress_hook.go
package sim

// ProgressHook receives periodic state snapshots during simulation execution.
// Implementations must treat snapshots as read-only and must not enqueue new
// simulation events or modify request state. A read-only, synchronous callback
// with no side-effects on simulation state cannot affect event ordering,
// therefore it cannot affect stdout (INV-6).
type ProgressHook interface {
    OnProgress(snapshot ProgressSnapshot)
}

// ProgressSnapshot captures simulation state at a point in time.
// Value type — fully copied, safe to hold indefinitely.
type ProgressSnapshot struct {
    Clock             int64
    TotalCompleted    int
    TotalTimedOut     int
    TotalDropped      int
    TotalInputTokens  int
    TotalOutputTokens int
    TotalPreemptions  int64
    InstanceSnapshots []InstanceSnapshot
    RejectedRequests  int
    RoutingRejections int
    GatewayQueueDepth int
    GatewayQueueShed  int
    ActivePDTransfers int
    ActiveInstances   int
    TotalInstances    int
    IsFinal           bool
}

// InstanceSnapshot captures per-instance state at a point in time.
// Value type — fully copied, safe to hold indefinitely.
type InstanceSnapshot struct {
    ID                string
    QueueDepth        int
    BatchSize         int
    KVUtilization     float64
    KVFreeBlocks      int64
    KVTotalBlocks     int64
    CacheHitRate      float64
    PreemptionCount   int64
    CompletedRequests int
    InFlightRequests  int
    TimedOutRequests  int
    State             string
    Model             string
}
```

**Verify:** `go test ./sim/... -run TestProgressSnapshot -count=1`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `feat(sim): define ProgressHook interface and snapshot types (BC-7)`

---

#### Task 2: Wire ProgressHook into single-instance Simulator (BC-1, BC-2, BC-4, BC-6)

**Files:** modify `sim/simulator.go`, test `sim/progress_hook_test.go`

**Test:**
```go
func TestSimulator_ProgressHook_FiresAtInterval(t *testing.T) {
    // BC-1: Hook fires when clock crosses interval boundary
    // BC-2: Final snapshot with IsFinal=true
    // BC-4: Single-instance snapshot has exactly one InstanceSnapshot
    // BC-6: nil hook produces no calls (implicit — we test non-nil path)
    sim := newTestSimulator(t) // helper that creates a minimal Simulator
    sim.InjectArrival(newTestRequest("req-1", 0, 100, 50))

    var snapshots []ProgressSnapshot
    sim.SetProgressHook(&collectingHook{snapshots: &snapshots}, 500_000) // 500ms intervals

    sim.Run()

    if len(snapshots) == 0 {
        t.Fatal("expected at least one snapshot")
    }
    // Final snapshot
    last := snapshots[len(snapshots)-1]
    if !last.IsFinal {
        t.Error("last snapshot should have IsFinal=true")
    }
    // Only one IsFinal
    finalCount := 0
    for _, s := range snapshots {
        if s.IsFinal {
            finalCount++
        }
    }
    if finalCount != 1 {
        t.Errorf("expected exactly 1 IsFinal snapshot, got %d", finalCount)
    }
    // BC-4: each snapshot has exactly 1 instance
    for i, s := range snapshots {
        if len(s.InstanceSnapshots) != 1 {
            t.Errorf("snapshot %d: expected 1 InstanceSnapshot, got %d", i, len(s.InstanceSnapshots))
        }
    }
}

func TestSimulator_ProgressHook_NilHookNoImpact(t *testing.T) {
    // BC-6: nil hook means zero overhead — simulation runs identically
    sim := newTestSimulator(t)
    sim.InjectArrival(newTestRequest("req-1", 0, 100, 50))
    sim.Run() // no SetProgressHook — should work fine
    if sim.Metrics.CompletedRequests != 1 {
        t.Errorf("expected 1 completed request, got %d", sim.Metrics.CompletedRequests)
    }
}
```

**Impl:** Add to `Simulator` struct:
```go
progressHook       ProgressHook
progressIntervalUs int64
nextSnapshotTime   int64
```

Add `SetProgressHook` method:
```go
func (sim *Simulator) SetProgressHook(hook ProgressHook, intervalUs int64) {
    sim.progressHook = hook
    if intervalUs > 0 {
        sim.progressIntervalUs = intervalUs
        sim.nextSnapshotTime = intervalUs
    }
}
```

Modify `Run()` to check after each `ProcessNextEvent()`:
```go
func (sim *Simulator) Run() {
    for sim.HasPendingEvents() {
        sim.ProcessNextEvent()
        if sim.Clock > sim.Horizon {
            break
        }
        sim.maybeDeliverProgressSnapshot(false)
    }
    sim.maybeDeliverProgressSnapshot(true)
    sim.Finalize()
}
```

Add snapshot builder:
```go
func (sim *Simulator) maybeDeliverProgressSnapshot(isFinal bool) {
    if sim.progressHook == nil {
        return
    }
    if !isFinal && (sim.progressIntervalUs <= 0 || sim.Clock < sim.nextSnapshotTime) {
        return
    }
    clock := sim.Clock
    if isFinal {
        clock = min(sim.Clock, sim.Horizon)
    }
    snap := ProgressSnapshot{
        Clock:             clock,
        TotalCompleted:    sim.Metrics.CompletedRequests,
        TotalTimedOut:     sim.Metrics.TimedOutRequests,
        TotalDropped:      sim.Metrics.DroppedUnservable,
        TotalInputTokens:  sim.Metrics.TotalInputTokens,
        TotalOutputTokens: sim.Metrics.TotalOutputTokens,
        TotalPreemptions:  sim.Metrics.PreemptionCount,
        InstanceSnapshots: []InstanceSnapshot{sim.buildInstanceSnapshot()},
        IsFinal:           isFinal,
    }
    sim.progressHook.OnProgress(snap)
    if !isFinal {
        sim.nextSnapshotTime += sim.progressIntervalUs
    }
}

func (sim *Simulator) buildInstanceSnapshot() InstanceSnapshot {
    return InstanceSnapshot{
        ID:                "instance-0",
        Model:             sim.model,
        QueueDepth:        sim.QueueDepth(),
        BatchSize:         sim.BatchSize(),
        KVUtilization:     float64(sim.KVCache.UsedBlocks()) / float64(max(sim.KVCache.TotalCapacity(), 1)),
        KVFreeBlocks:      sim.KVCache.TotalCapacity() - sim.KVCache.UsedBlocks(),
        KVTotalBlocks:     sim.KVCache.TotalCapacity(),
        CacheHitRate:      sim.KVCache.CacheHitRate(),
        PreemptionCount:   sim.Metrics.PreemptionCount,
        CompletedRequests: sim.Metrics.CompletedRequests,
        TimedOutRequests:  sim.Metrics.TimedOutRequests,
    }
}
```

**Verify:** `go test ./sim/... -run TestSimulator_ProgressHook -count=1`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `feat(sim): wire ProgressHook into single-instance Simulator (BC-1, BC-2, BC-4, BC-6)`

---

#### Task 3: Wire ProgressHook into ClusterSimulator (BC-1, BC-2, BC-3, BC-6, BC-8)

**Files:** modify `sim/cluster/cluster.go`, modify `sim/cluster/instance.go` (add `TotalKVBlocks()` accessor), test `sim/progress_hook_test.go` (or `sim/cluster/cluster_test.go` if test helpers exist there)

**Test:**
```go
// In sim/progress_hook_test.go (or cluster test file)
func TestClusterSimulator_ProgressHook_FiresWithInstances(t *testing.T) {
    // BC-3: cluster snapshots contain per-instance detail
    // BC-8: final snapshot fires on all exit paths
    cs := newTestClusterSimulator(t, 2) // 2 instances
    var snapshots []ProgressSnapshot
    cs.SetProgressHook(&collectingHook{snapshots: &snapshots}, 500_000)

    // inject requests and run
    cs.Run()

    if len(snapshots) == 0 {
        t.Fatal("expected at least one snapshot")
    }
    last := snapshots[len(snapshots)-1]
    if !last.IsFinal {
        t.Error("final snapshot should have IsFinal=true")
    }
    if last.TotalInstances != 2 {
        t.Errorf("expected TotalInstances=2, got %d", last.TotalInstances)
    }
    if len(last.InstanceSnapshots) != 2 {
        t.Errorf("expected 2 InstanceSnapshots, got %d", len(last.InstanceSnapshots))
    }
}
```

**Impl:** Add to `ClusterSimulator` struct:
```go
progressHook       sim.ProgressHook
progressIntervalUs int64
nextSnapshotTime   int64
```

Add `SetProgressHook` method:
```go
func (c *ClusterSimulator) SetProgressHook(hook sim.ProgressHook, intervalUs int64) {
    c.progressHook = hook
    if intervalUs > 0 {
        c.progressIntervalUs = intervalUs
        c.nextSnapshotTime = intervalUs
    }
}
```

In `Run()`, add the snapshot check at the **end of the for-loop body** (outside both `if`/`else` branches), after all event processing, drain/PD-detection, and flow-control logic. This ensures:
- For cluster-event branch: fires after event execution
- For instance-event branch: fires after clock-restore (orphaned timeout), completion accounting, drain, PD detection
- Uses `c.clock` which has already been corrected for orphaned timeouts

```go
    // ... end of instance-event else block (PD detection, drain, etc.)
    }

    // Progress hook: check after ALL event processing in this iteration
    c.maybeDeliverProgressSnapshot(false)
}
```

After the event loop exits (before finalization), deliver the final snapshot:
```go
c.maybeDeliverProgressSnapshot(true)
```

Snapshot builder:
```go
func (c *ClusterSimulator) maybeDeliverProgressSnapshot(isFinal bool) {
    if c.progressHook == nil {
        return
    }
    if !isFinal && (c.progressIntervalUs <= 0 || c.clock < c.nextSnapshotTime) {
        return
    }
    activeCount := 0
    instanceSnaps := make([]sim.InstanceSnapshot, 0, len(c.instances))
    for _, inst := range c.instances {
        if inst.State == InstanceStateTerminated {
            continue
        }
        if inst.State == InstanceStateActive || inst.State == InstanceStateWarmingUp {
            activeCount++
        }
        instanceSnaps = append(instanceSnaps, sim.InstanceSnapshot{
            ID:                string(inst.ID()),
            QueueDepth:        inst.QueueDepth(),
            BatchSize:         inst.BatchSize(),
            KVUtilization:     inst.KVUtilization(),
            KVFreeBlocks:      inst.FreeKVBlocks(),
            KVTotalBlocks:     inst.TotalKVBlocks(),
            CacheHitRate:      inst.CacheHitRate(),
            PreemptionCount:   inst.PreemptionCount(),
            CompletedRequests: inst.Metrics().CompletedRequests,
            InFlightRequests:  c.inFlightRequests[string(inst.ID())],
            TimedOutRequests:  inst.Metrics().TimedOutRequests,
            State:             string(inst.State),
            Model:             inst.Model,
        })
    }

    var gatewayQueueDepth, gatewayQueueShed int
    if c.gatewayQueue != nil {
        gatewayQueueDepth = c.gatewayQueue.Len()
        gatewayQueueShed = c.gatewayQueue.ShedCount()
    }

    snap := sim.ProgressSnapshot{
        Clock:             c.clock,
        TotalCompleted:    c.completedRequestsTotal(),
        TotalTimedOut:     c.timedOutRequestsTotal(),
        TotalDropped:      c.droppedRequestsTotal(),
        TotalInputTokens:  c.inputTokensTotal(),
        TotalOutputTokens: c.outputTokensTotal(),
        TotalPreemptions:  c.preemptionsTotal(),
        InstanceSnapshots: instanceSnaps,
        RejectedRequests:  c.rejectedRequests,
        RoutingRejections: c.routingRejections,
        GatewayQueueDepth: gatewayQueueDepth,
        GatewayQueueShed:  gatewayQueueShed,
        ActivePDTransfers: c.activeTransfers,
        ActiveInstances:   activeCount,
        TotalInstances:    len(c.instances),
        IsFinal:           isFinal,
    }
    c.progressHook.OnProgress(snap)
    if !isFinal {
        c.nextSnapshotTime += c.progressIntervalUs
    }
}
```

The following private aggregate helpers must be added to `ClusterSimulator` (they do not currently exist):

```go
func (c *ClusterSimulator) completedRequestsTotal() int {
    total := 0
    for _, inst := range c.instances {
        total += inst.Metrics().CompletedRequests
    }
    return total
}

func (c *ClusterSimulator) timedOutRequestsTotal() int {
    total := 0
    for _, inst := range c.instances {
        total += inst.Metrics().TimedOutRequests
    }
    return total
}

func (c *ClusterSimulator) droppedRequestsTotal() int {
    total := 0
    for _, inst := range c.instances {
        total += inst.Metrics().DroppedUnservable
    }
    return total
}

func (c *ClusterSimulator) inputTokensTotal() int {
    total := 0
    for _, inst := range c.instances {
        total += inst.Metrics().TotalInputTokens
    }
    return total
}

func (c *ClusterSimulator) outputTokensTotal() int {
    total := 0
    for _, inst := range c.instances {
        total += inst.Metrics().TotalOutputTokens
    }
    return total
}

func (c *ClusterSimulator) preemptionsTotal() int64 {
    var total int64
    for _, inst := range c.instances {
        total += inst.Metrics().PreemptionCount
    }
    return total
}
```

**Verify:** `go test ./sim/cluster/... -run TestClusterSimulator_ProgressHook -count=1`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): wire ProgressHook into ClusterSimulator (BC-1, BC-2, BC-3, BC-6, BC-8)`

---

#### Task 4: Determinism test (BC-5)

**Files:** test `sim/progress_hook_test.go`

**Test:**
```go
func TestSimulator_ProgressHook_Determinism(t *testing.T) {
    // BC-5: a non-nil hook must not change simulation output
    runSim := func(withHook bool) *Metrics {
        sim := newTestSimulator(t)
        sim.InjectArrival(newTestRequest("req-1", 0, 100, 50))
        sim.InjectArrival(newTestRequest("req-2", 100_000, 80, 30))
        if withHook {
            sim.SetProgressHook(&collectingHook{snapshots: &[]ProgressSnapshot{}}, 100_000)
        }
        sim.Run()
        return sim.Metrics
    }
    without := runSim(false)
    with := runSim(true)

    if without.CompletedRequests != with.CompletedRequests {
        t.Errorf("CompletedRequests differs: %d vs %d", without.CompletedRequests, with.CompletedRequests)
    }
    if without.TotalInputTokens != with.TotalInputTokens {
        t.Errorf("TotalInputTokens differs: %d vs %d", without.TotalInputTokens, with.TotalInputTokens)
    }
    if without.TotalOutputTokens != with.TotalOutputTokens {
        t.Errorf("TotalOutputTokens differs: %d vs %d", without.TotalOutputTokens, with.TotalOutputTokens)
    }
    if without.SimEndedTime != with.SimEndedTime {
        t.Errorf("SimEndedTime differs: %d vs %d", without.SimEndedTime, with.SimEndedTime)
    }
}
```

**Impl:** No new production code — this test validates BC-5 using existing infrastructure.

**Verify:** `go test ./sim/... -run TestSimulator_ProgressHook_Determinism -count=1`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `test(sim): verify ProgressHook determinism preservation (BC-5)`

---

#### Task 5: Interval boundary and snapshot count tests (BC-1)

**Files:** test `sim/progress_hook_test.go`

**Test:**
```go
func TestSimulator_ProgressHook_IntervalBoundaries(t *testing.T) {
    // BC-1: snapshots fire at interval boundaries, not every event
    sim := newTestSimulator(t) // horizon = 10_000_000 (10s)
    // Inject multiple requests to ensure many events fire
    for i := 0; i < 10; i++ {
        sim.InjectArrival(newTestRequest(
            fmt.Sprintf("req-%d", i),
            int64(i)*500_000, // every 500ms
            100, 50,
        ))
    }
    var snapshots []ProgressSnapshot
    sim.SetProgressHook(&collectingHook{snapshots: &snapshots}, 2_000_000) // 2s intervals

    sim.Run()

    // Non-final snapshots should have increasing clock values aligned to intervals
    for i := 0; i < len(snapshots)-1; i++ {
        if snapshots[i].IsFinal {
            t.Errorf("snapshot %d should not be final", i)
        }
        if i > 0 && snapshots[i].Clock <= snapshots[i-1].Clock {
            t.Errorf("snapshot clocks not monotonically increasing: %d <= %d",
                snapshots[i].Clock, snapshots[i-1].Clock)
        }
    }
    // Number of non-final snapshots should be reasonable (not one per event)
    nonFinal := len(snapshots) - 1
    if nonFinal > 10 {
        t.Errorf("too many non-final snapshots (%d) for 2s interval — suggests per-event firing", nonFinal)
    }
}
```

**Impl:** No new production code.

**Verify:** `go test ./sim/... -run TestSimulator_ProgressHook_IntervalBoundaries -count=1`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `test(sim): verify interval boundary snapshot delivery (BC-1)`

---

#### Task 6: Fresh slice allocation per callback (BC-7 defense-in-depth)

**Files:** test `sim/progress_hook_test.go`

**Test:**
```go
func TestSimulator_ProgressHook_FreshSlicePerCall(t *testing.T) {
    // BC-7: each OnProgress call gets a fresh InstanceSnapshots slice
    sim := newTestSimulator(t)
    sim.InjectArrival(newTestRequest("req-1", 0, 100, 50))

    var snapshots []ProgressSnapshot
    hook := &collectingHook{snapshots: &snapshots}
    sim.SetProgressHook(hook, 100_000) // frequent snapshots

    sim.Run()

    if len(snapshots) < 2 {
        t.Skip("need at least 2 snapshots to test slice independence")
    }
    // Mutate the first snapshot's slice — second must be unaffected
    snapshots[0].InstanceSnapshots[0].QueueDepth = 999999
    if snapshots[1].InstanceSnapshots[0].QueueDepth == 999999 {
        t.Error("InstanceSnapshots slices are shared between callbacks — BC-7 violated")
    }
}
```

**Impl:** No new production code — the `[]InstanceSnapshot{sim.buildInstanceSnapshot()}` allocation in `maybeDeliverProgressSnapshot` already creates a fresh slice per call.

**Verify:** `go test ./sim/... -run TestSimulator_ProgressHook_FreshSlice -count=1`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `test(sim): verify fresh slice allocation per progress callback (BC-7)`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-7 | Task 1 | Unit | `TestProgressSnapshot_IsValueType` |
| BC-1, BC-2, BC-4, BC-6 | Task 2 | Integration | `TestSimulator_ProgressHook_FiresAtInterval`, `TestSimulator_ProgressHook_NilHookNoImpact` |
| BC-1, BC-2, BC-3, BC-6, BC-8 | Task 3 | Integration | `TestClusterSimulator_ProgressHook_FiresWithInstances` |
| BC-5 | Task 4 | Invariant | `TestSimulator_ProgressHook_Determinism` |
| BC-1 | Task 5 | Behavioral | `TestSimulator_ProgressHook_IntervalBoundaries` |
| BC-7 | Task 6 | Behavioral | `TestSimulator_ProgressHook_FreshSlicePerCall` |

Key invariants verified:
- **INV-6 (Determinism):** BC-5 test compares metrics between hooked and unhooked runs
- **INV-3 (Clock monotonicity):** BC-1 test verifies snapshot clock values are non-decreasing

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Snapshot construction affects event ordering | Low | High | BC-5 determinism test; all snapshot reads use existing pure query methods | Task 4 |
| Performance regression from per-event nil check | Low | Low | Single nil pointer comparison — effectively free; no allocation when nil | Task 2 |
| Cluster aggregate helpers missing | Medium | Low | Verify during implementation; add small private aggregation helpers if needed | Task 3 |
| Snapshot fires during clock-restore (orphaned TimeoutEvent) | Medium | Medium | Check fires after event processing (including clock restore), using `c.clock` not event timestamp | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — single interface, two struct types, direct wiring
- [x] No feature creep — no CLI flags, no per-request detail, no async delivery
- [x] No unexercised flags or interfaces — `ProgressHook` is exercised by tests
- [x] No partial implementations — complete coverage of single-instance and cluster paths
- [x] No breaking changes — additive only (new interface, new optional method)
- [x] No hidden global state impact — all state is per-simulator instance
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing package
- [x] CLAUDE.md does not need updating — no new CLI flags, no file organization changes, no new packages
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY — no canonical sources modified
- [x] Deviation log reviewed — all deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 → Task 2 → Task 3 → Tasks 4-6)
- [x] All contracts mapped to specific tasks
- [x] Construction site audit completed — `Simulator` and `ClusterSimulator` each have single construction sites

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — nil guard returns immediately (expected, not data loss)
- [x] R2: No map iteration for ordered output in snapshot construction
- [x] R3: `intervalUs <= 0` validated: `SetProgressHook` only sets interval fields when `intervalUs > 0`; `maybeDeliverProgressSnapshot` short-circuits non-final calls when `progressIntervalUs <= 0`
- [x] R4: Construction sites audited — `Simulator` (NewSimulator), `ClusterSimulator` (NewClusterSimulator)
- [x] R5: No resource allocation loops
- [x] R6: No `logrus.Fatalf` in `sim/` — hook code is library code
- [x] R7: Invariant test (BC-5 determinism) alongside behavioral tests
- [x] R8: No exported mutable maps
- [x] R9: No YAML fields
- [x] R10: No YAML parsing
- [x] R11: Division guarded (`max(sim.KVCache.TotalCapacity(), 1)`)
- [x] R12: No golden dataset changes
- [x] R13: `ProgressHook` interface has single method — appropriate for extensibility (multiple callers can implement differently)
- [x] R14: No method spans multiple module responsibilities
- [x] R15: No stale PR references
- [x] R16: No new config params (hook is set programmatically, not via config)
- [x] R17: N/A — no routing scorer signals
- [x] R18: N/A — no CLI flags
- [x] R19: No retry loops
- [x] R20: N/A — no detectors/analyzers
- [x] R21: No range over shrinking slices
- [x] R22: N/A — no pre-check estimates
- [x] R23: Single-instance and cluster paths apply equivalent snapshot construction

---

## Appendix: File-Level Implementation Details

### File: `sim/progress_hook.go`

- **Purpose:** Define `ProgressHook` interface and `ProgressSnapshot`/`InstanceSnapshot` value types
- **Complete implementation:** As shown in Task 1
- **Key notes:**
  - No event ordering impact — these are pure data types
  - No RNG usage
  - No metrics collection
  - No state mutation
  - Error handling: N/A — pure types

### File: `sim/progress_hook_test.go`

- **Purpose:** All tests for progress hook behavior
- **Key notes:**
  - Test helper: `collectingHook` — implements `ProgressHook`, appends snapshots to a slice
  - Test helper: `newTestSimulator` — creates a minimal `Simulator` with default config (reuse existing test helpers if available)
  - Test helper: `newTestRequest` — creates a `Request` with specified ID, arrival, input/output lengths

### File: `sim/simulator.go`

- **Purpose:** Wire `ProgressHook` into single-instance event loop
- **Changes:**
  - Add 3 fields to `Simulator` struct: `progressHook`, `progressIntervalUs`, `nextSnapshotTime`
  - Add `SetProgressHook()` method
  - Add `maybeDeliverProgressSnapshot()` private method
  - Add `buildInstanceSnapshot()` private method
  - Modify `Run()`: add `maybeDeliverProgressSnapshot(false)` after each event, `maybeDeliverProgressSnapshot(true)` before `Finalize()`
- **Key notes:**
  - Fields are zero-value safe (nil hook = no overhead)
  - `SetProgressHook` must be called before `Run()` — not enforced (matching `OnRequestDone` pattern)

### File: `sim/cluster/instance.go`

- **Purpose:** Add `TotalKVBlocks()` accessor for snapshot construction
- **Changes:**
  - Add `TotalKVBlocks()` method returning `i.sim.KVCache.TotalCapacity()`
- **Key notes:** Matches pattern of existing accessors (`FreeKVBlocks()`, `KVUtilization()`)

### File: `sim/cluster/cluster.go`

- **Purpose:** Wire `ProgressHook` into cluster event loop
- **Changes:**
  - Add 3 fields to `ClusterSimulator` struct: `progressHook`, `progressIntervalUs`, `nextSnapshotTime`
  - Add `SetProgressHook()` method
  - Add `maybeDeliverProgressSnapshot()` private method
  - Add 6 private aggregate helpers: `completedRequestsTotal()`, `timedOutRequestsTotal()`, `droppedRequestsTotal()`, `inputTokensTotal()`, `outputTokensTotal()`, `preemptionsTotal()`
  - Modify `Run()`: add snapshot check at end of for-loop body (after all event processing), final snapshot after loop
- **Key notes:**
  - Aggregate metrics across instances via iteration (not from `aggregatedMetrics` which is only populated post-finalization)
  - `InFlightRequests` per instance read from `c.inFlightRequests` map
  - Gateway queue state read from `c.gatewayQueue` (nil-safe)
  - PD transfer count from `c.activeTransfers`
  - Snapshot check placed at end of loop body (outside if/else) to correctly handle clock-restore for orphaned timeouts
