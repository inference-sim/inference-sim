# Phase 6: Modularity Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce the "touch-point multiplier" for future extensions by adding accessor methods, eliminating type duplication, fixing destructive-read semantics, and removing library code that terminates the process.

**The problem today:** Adding a single new observable metric to the cluster simulation requires changing 6 files because `InstanceSimulator` reaches directly into `Simulator` exported fields and `InstanceSnapshot` duplicates `RoutingSnapshot` field-by-field. The `KVStore` interface requires type assertions for `SetClock` and has destructive-read semantics on `PendingTransferLatency`. Library code in `sim/` calls `logrus.Fatalf` (which calls `os.Exit`), making it impossible to embed as a library. CLI error messages hardcode policy names that can drift from the authoritative registry.

**What this PR adds:**
1. **Simulator observation methods** — `QueueDepth()`, `BatchSize()`, `CurrentClock()`, `SimHorizon()` methods that decouple `InstanceSimulator` from exported `Simulator` fields
2. **Snapshot unification** — `RoutingSnapshot` becomes the canonical type, eliminating the `InstanceSnapshot` duplicate and field-by-field translation
3. **KVStore interface improvements** — `SetClock()` added to interface (eliminates type assertion), `PendingTransferLatency()` becomes a pure query with separate `ConsumePendingTransferLatency()` for mutation
4. **Library-safe error handling** — `logrus.Fatalf` calls in `sim/` replaced with error returns; `SavetoFile` (dead code) removed
5. **CLI-bundle alignment** — `ValidAdmissionPolicyNames()` etc. derive from authoritative maps, eliminating hardcoded policy lists
6. **Type-safe request state** — `RequestState` typed constants replace raw string assignments
7. **NewKVStore validation** — factory validates inputs matching policy factory pattern

**Why this matters:** These changes reduce extension friction for PR11 (autoscaling), PR14 (P/D disaggregation), PR15 (framework adapters), and future features. Adding a new observable becomes a 3-4 file change instead of 6. Library code becomes embeddable.

**Architecture:** All changes are in `sim/`, `sim/cluster/`, and `cmd/`. No new packages. The `KVStore` interface gains one method (`SetClock`). The `SnapshotProvider` interface changes return type from `InstanceSnapshot` to `sim.RoutingSnapshot`. `NewSimulator` gains an `error` return for CSV parsing failures. All changes preserve existing behavior — this is a pure refactoring PR.

**Source:** GitHub issue #213 (Phase 6 of Hardening PR #214), design doc `docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md` Phase 6

**Closes:** Fixes #213

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR performs 8 modularity improvements (6a-6h from the hardening design doc) that reduce extension friction without changing external behavior. The changes fall into three categories: (1) decoupling `InstanceSimulator` from `Simulator` internals via accessor methods and snapshot unification, (2) fixing KVStore interface deficiencies (type assertion, destructive read), and (3) eliminating library code that terminates the process.

**Adjacent blocks:** `sim/simulator.go` (observation methods), `sim/cluster/instance.go` (uses new accessors), `sim/cluster/snapshot.go` (SnapshotProvider return type change), `sim/kv_store.go` (interface expansion), `cmd/root.go` (error handling boundary).

**DEVIATION flags:** See Section D. Key deviations: `SavetoFile` is dead code and will be removed rather than refactored; `NewSimulator` signature change requires updating 24 call sites.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Observation Method Equivalence
- GIVEN a Simulator with requests in the wait queue and a running batch
- WHEN `QueueDepth()`, `BatchSize()`, `CurrentClock()`, `SimHorizon()` are called
- THEN they return values identical to direct field access (`WaitQ.Len()`, `len(RunningBatch.Requests)`, `Clock`, `Horizon`)
- MECHANISM: Simple delegation methods on Simulator

BC-2: Snapshot Unification
- GIVEN a cluster with instances producing snapshots via CachedSnapshotProvider
- WHEN `buildRouterState()` constructs a `RouterState`
- THEN the snapshots are `RoutingSnapshot` values directly (no intermediate `InstanceSnapshot` translation)
- MECHANISM: SnapshotProvider returns `sim.RoutingSnapshot`; PendingRequests injected at buildRouterState

BC-3: PendingTransferLatency Pure Query
- GIVEN a TieredKVCache with accumulated transfer latency
- WHEN `PendingTransferLatency()` is called twice consecutively
- THEN both calls return the same value (no side effects)
- MECHANISM: Read-only accessor; mutation moved to `ConsumePendingTransferLatency()`

BC-4: ConsumePendingTransferLatency Clears
- GIVEN a TieredKVCache with accumulated transfer latency > 0
- WHEN `ConsumePendingTransferLatency()` is called
- THEN it returns the accumulated value AND resets to zero
- MECHANISM: Read-and-clear method, called by Simulator.Step()

BC-5: SetClock on KVStore Interface
- GIVEN any KVStore implementation (single-tier or tiered)
- WHEN `SetClock(clock)` is called
- THEN single-tier is a no-op; tiered updates internal clock for thrashing detection
- MECHANISM: Interface method; eliminates type assertion in Simulator.Step()

BC-6: RequestState Type Safety
- GIVEN production code that sets `Request.State`
- WHEN a state transition occurs (queued → running → completed)
- THEN the assignment uses typed constants (`StateQueued`, `StateRunning`, `StateCompleted`)
- MECHANISM: `RequestState` type with const values; `Request.State` field type changed

BC-7: ValidPolicyNames Derivation
- GIVEN CLI error messages for invalid policy names
- WHEN a user provides an unknown policy name
- THEN the error message lists valid names derived from the authoritative bundle.go maps
- MECHANISM: `ValidAdmissionPolicyNames()` etc. return sorted names from unexported maps

BC-8: NewKVStore Validation
- GIVEN a SimConfig with `TotalKVBlocks <= 0` or `BlockSizeTokens <= 0`
- WHEN `NewKVStore(cfg)` is called
- THEN it panics with a descriptive message (matching policy factory pattern)
- MECHANISM: Input validation at factory entry

BC-9: Library Code Error Returns
- GIVEN `sim/workload_config.go` CSV loading encounters an error
- WHEN the error occurs (file not found, parse error, etc.)
- THEN it returns an `error` to the caller instead of calling `logrus.Fatalf`
- MECHANISM: `generateWorkloadFromCSV() error`; `NewSimulator` propagates as `(*Simulator, error)`

**Negative Contracts:**

BC-10: No Behavior Change
- GIVEN any existing simulation configuration
- WHEN the simulation runs end-to-end
- THEN output is byte-identical to before this PR (same seed → same results)
- MECHANISM: All changes are structural, not behavioral; golden dataset unchanged

BC-11: No New Library Fatals
- GIVEN any code path in `sim/` or `sim/cluster/`
- WHEN an error occurs during CSV loading or file I/O
- THEN the code MUST NOT call `logrus.Fatalf`, `os.Exit`, or terminate the process
- MECHANISM: Error returns propagated to `cmd/root.go` boundary

**Error Handling Contracts:**

BC-12: NewKVStore Panic on Invalid Config
- GIVEN `TotalKVBlocks <= 0` OR `BlockSizeTokens <= 0`
- WHEN `NewKVStore(cfg)` is called directly (not via NewSimulator)
- THEN panic with message including the invalid value
- MECHANISM: Consistent with policy factory pattern (programming error, not user error)

BC-13: CSV Error Propagation
- GIVEN a CSV trace file with malformed data
- WHEN `NewSimulator(cfg)` processes the file
- THEN it returns `(nil, error)` with a descriptive message including row number and field
- MECHANISM: Error wrapping through `generateWorkloadFromCSV` → `NewSimulator`

### C) Component Interaction

```
cmd/root.go
  │ logrus.Fatalf for user errors
  │ handles NewSimulator error return
  │ uses ValidAdmissionPolicyNames() for error messages
  ▼
sim/simulator.go
  │ NewSimulator returns (*Simulator, error)
  │ QueueDepth(), BatchSize(), CurrentClock(), SimHorizon() accessors
  │ KVCache.SetClock(now) — no type assertion
  │ KVCache.ConsumePendingTransferLatency() — explicit mutation
  ▼
sim/kv_store.go (KVStore interface)
  │ +SetClock(clock int64)
  │ PendingTransferLatency() — now pure query
  ├──▶ sim/kvcache.go (KVCacheState) — SetClock no-op
  └──▶ sim/kvcache_tiered.go (TieredKVCache) — SetClock updates clock
       +ConsumePendingTransferLatency()

sim/cluster/instance.go
  │ Uses sim.Simulator accessor methods
  │ No longer reaches through exported fields
  ▼
sim/cluster/snapshot.go
  │ SnapshotProvider returns sim.RoutingSnapshot (not InstanceSnapshot)
  │ CachedSnapshotProvider produces RoutingSnapshot directly
  ▼
sim/cluster/cluster_event.go
  │ buildRouterState injects PendingRequests into RoutingSnapshot
  │ No InstanceSnapshot → RoutingSnapshot translation
```

**Extension Friction Assessment:**
- Adding a new observable field: 6 files → 3-4 files (Simulator method + RoutingSnapshot field + snapshot wiring)
- Adding a new policy template: 3 files → 2 files (implementation + bundle registration; CLI auto-derives)

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| 6e lists `sim/metrics_utils.go` `SavetoFile` as having 4 logrus.Fatalf calls | Remove `SavetoFile` entirely | SIMPLIFICATION: `SavetoFile` is dead code — defined but never called anywhere in the codebase. Removing it is cleaner than refactoring dead code. |
| 6e says "return error from these functions" generically | Change `NewSimulator` to return `(*Simulator, error)` | ADDITION: The design doc doesn't specify the exact propagation mechanism. `NewSimulator` must change signature because `generateWorkloadFromCSV` is called from the constructor. |
| 6b says "have `CachedSnapshotProvider.Snapshot()` return `sim.RoutingSnapshot` directly" | Also change `SnapshotProvider` interface and `RefreshAll` | ADDITION: Interface return type must change too; design doc implied but didn't explicitly state this. |
| Design doc mentions "Timestamp" field on InstanceSnapshot | Drop Timestamp field from RoutingSnapshot | SIMPLIFICATION: Timestamp was only on InstanceSnapshot, not RoutingSnapshot. Since we're unifying to RoutingSnapshot, no field addition needed. The clock is already in RouterState. |
| 6c says "The `KVStore` interface method `PendingTransferLatency()` remains a pure query" | Add `ConsumePendingTransferLatency()` to KVStore interface too | ADDITION: Design doc only specifies a concrete method on TieredKVCache, but the Simulator calls through the KVStore interface. Adding to interface is cleaner than a type assertion. KVCacheState implements as no-op (returns 0). |
| 6e does not specify cluster error propagation | `ClusterSimulator.Run()` returns error; `NewInstanceSimulator` handles `NewSimulator` error via panic (unreachable in cluster mode) | ADDITION: `DeploymentConfig.ToSimConfig()` omits `TracesWorkloadFilePath`, so `NewSimulator` error is unreachable in cluster mode. `NewInstanceSimulator` uses panic-on-error (matching existing factory pattern) to satisfy the compiler. `ClusterSimulator.Run()` returns error from `generateRequestsFromCSV`. |
| 6e mentions `sim/cluster/workload.go` logrus.Fatalf | Change `generateRequestsFromCSV` to return `([]*sim.Request, error)`, propagate through `generateRequests` | ADDITION: Error propagation requires changing `generateRequests` and `Run` signatures. |

### E) Review Guide

**The tricky part:** The `NewSimulator` signature change from `*Simulator` to `(*Simulator, error)` touches 24 call sites. Verify that test call sites use the helper or properly handle the error — a missing error check would silently ignore CSV parse failures.

**What to scrutinize:** BC-2 (snapshot unification) — verify that `PendingRequests` is still correctly injected at the `buildRouterState` level, not lost during the InstanceSnapshot elimination. BC-3/BC-4 — verify the Simulator.Step() call site switches from `PendingTransferLatency()` to `ConsumePendingTransferLatency()`.

**What's safe to skim:** BC-6 (RequestState constants) is purely mechanical — changing string literals to typed constants. BC-7 (ValidPolicyNames) is a simple accessor pattern. BC-8 (NewKVStore validation) is copy-paste from NewSimulator's existing validation.

**Known debt:** `Simulator.Step()` remains a 134-line monolith mixing multiple concerns — deferred to SGLang engine work. `SimConfig` still mixes concerns — deferred as Go-idiomatic.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/request.go` — Add `RequestState` type + constants, change `State` field type
- `sim/kv_store.go` — Add `SetClock` to KVStore interface, add NewKVStore validation
- `sim/kvcache.go` — Add `SetClock` no-op to KVCacheState
- `sim/kvcache_tiered.go` — Fix PendingTransferLatency, add ConsumePendingTransferLatency
- `sim/simulator.go` — Add observation methods, use interface SetClock, change NewSimulator signature
- `sim/bundle.go` — Add ValidAdmissionPolicyNames() etc.
- `sim/workload_config.go` — Return error from generateWorkloadFromCSV
- `sim/metrics_utils.go` — Remove dead SavetoFile
- `sim/cluster/instance.go` — Use Simulator observation methods
- `sim/cluster/snapshot.go` — Change to RoutingSnapshot, eliminate InstanceSnapshot
- `sim/cluster/cluster_event.go` — Simplify buildRouterState
- `sim/cluster/workload.go` — Return error from generateRequestsFromCSV
- `cmd/root.go` — Use ValidPolicyNames, handle NewSimulator error

**Key decisions:**
- `NewSimulator` returns `(*Simulator, error)` — the only clean way to propagate CSV errors
- `SavetoFile` removed (dead code) rather than refactored
- `SnapshotProvider` interface changes return type — breaking change within cluster package
- Test files use `StateQueued` constant — mechanical but necessary for type safety

**Task dependency order:**
- Tasks 1-5 are independent (can run in any order)
- Task 6 MUST complete before Task 7 (snapshot unification depends on observation methods)
- Task 8 is independent but placed last because it touches the most files
- Task 9 (docs) runs last

**Confirmation:** No dead code. All paths exercisable. Every method used by end of plan.

### G) Task Breakdown

---

#### Task 1: RequestState typed constants (BC-6)

**Contracts Implemented:** BC-6

**Files:**
- Modify: `sim/request.go`
- Modify: `sim/simulator.go` (3 assignment sites)
- Modify: `sim/workload_config.go` (2 construction sites)
- Modify: `sim/cluster/workload.go` (2 construction sites)
- Modify: `sim/workload/generator.go`, `sim/workload/replay.go`, `sim/workload/reasoning.go`
- Modify: All test files with `State: "queued"` or `State: "running"`
- Test: `sim/request_test.go` (new)

**Step 1: Write failing test**

Context: Verify that RequestState constants have the expected string representation (used in Request.String() output).

In `sim/request_test.go`:
```go
package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRequestState_Constants_HaveExpectedStringValues(t *testing.T) {
	// BC-6: Typed constants replace raw strings
	assert.Equal(t, RequestState("queued"), StateQueued)
	assert.Equal(t, RequestState("running"), StateRunning)
	assert.Equal(t, RequestState("completed"), StateCompleted)
}

func TestRequest_String_IncludesState(t *testing.T) {
	req := Request{ID: "test-1", State: StateQueued}
	s := req.String()
	assert.Contains(t, s, "queued")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestRequestState -v`
Expected: FAIL — `StateQueued` undefined

**Step 3: Implement RequestState type and constants**

In `sim/request.go`, add before the Request struct:
```go
// RequestState represents the lifecycle state of a request.
type RequestState string

const (
	StateQueued    RequestState = "queued"
	StateRunning   RequestState = "running"
	StateCompleted RequestState = "completed"
)
```

Change `State` field:
```go
State         RequestState // queued, running, completed
```

Update `String()` to use `string(req.State)` if needed (it already uses `%s` which works).

**Step 4: Update all assignment sites**

In `sim/simulator.go`:
- Line 360: `preemptedRequest.State = StateQueued`
- Line 486: `next.State = StateRunning`
- Line 585: `req.State = StateCompleted`

In `sim/workload_config.go`:
- Lines 77, 159: `State: StateQueued,`

In `sim/cluster/workload.go`:
- Lines 58, 125: `State: sim.StateQueued,`

In `sim/workload/generator.go`:
- Line 128: `State: sim.StateQueued,`

In `sim/workload/replay.go`:
- Line 49: `State: sim.StateQueued,`

In `sim/workload/reasoning.go`:
- Line 70: `State: sim.StateQueued,`

Update all test files (mechanical — change `"queued"` to `StateQueued` or `sim.StateQueued`):
- `sim/simulator_test.go` (~10 sites)
- `sim/simulator_decode_test.go` (1 site: `State: StateRunning`)
- `sim/scheduler_test.go` (~6 sites)
- `sim/priority_test.go` (1 site)
- `sim/cluster/instance_test.go` (~3 sites)
- `sim/cluster/cluster_test.go` (1 site)
- `sim/cluster/snapshot_test.go` (~3 sites)
- `sim/cluster/pending_requests_test.go` (~3 sites)
- `sim/workload/generator_test.go` (1 site)

**Step 5: Run tests**

Run: `go test ./sim/... ./sim/cluster/... ./sim/workload/... -count=1`
Expected: PASS (all existing tests pass with typed constants)

**Step 6: Lint**

Run: `golangci-lint run ./sim/... ./sim/cluster/... ./sim/workload/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/request.go sim/request_test.go sim/simulator.go sim/workload_config.go sim/cluster/workload.go sim/workload/generator.go sim/workload/replay.go sim/workload/reasoning.go sim/simulator_test.go sim/simulator_decode_test.go sim/scheduler_test.go sim/priority_test.go sim/cluster/instance_test.go sim/cluster/cluster_test.go sim/cluster/snapshot_test.go sim/cluster/pending_requests_test.go sim/workload/generator_test.go
git commit -m "refactor(sim): add RequestState typed constants (BC-6)

- Define RequestState type with StateQueued, StateRunning, StateCompleted
- Change Request.State field from string to RequestState
- Update all assignment and construction sites

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: NewKVStore input validation (BC-8, BC-12)

**Contracts Implemented:** BC-8, BC-12

**Files:**
- Modify: `sim/kv_store.go`
- Test: `sim/kv_store_test.go` (new)

**Step 1: Write failing test**

In `sim/kv_store_test.go`:
```go
package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewKVStore_ZeroTotalBlocks_Panics(t *testing.T) {
	// BC-8: NewKVStore validates TotalKVBlocks > 0
	assert.PanicsWithValue(t,
		"KVStore: TotalKVBlocks must be > 0, got 0",
		func() {
			NewKVStore(SimConfig{TotalKVBlocks: 0, BlockSizeTokens: 16})
		})
}

func TestNewKVStore_ZeroBlockSize_Panics(t *testing.T) {
	// BC-8: NewKVStore validates BlockSizeTokens > 0
	assert.PanicsWithValue(t,
		"KVStore: BlockSizeTokens must be > 0, got 0",
		func() {
			NewKVStore(SimConfig{TotalKVBlocks: 100, BlockSizeTokens: 0})
		})
}

func TestNewKVStore_NegativeTotalBlocks_Panics(t *testing.T) {
	assert.Panics(t, func() {
		NewKVStore(SimConfig{TotalKVBlocks: -1, BlockSizeTokens: 16})
	})
}

func TestNewKVStore_ValidConfig_SingleTier_Succeeds(t *testing.T) {
	// BC-8: Valid config produces a working KVStore
	store := NewKVStore(SimConfig{TotalKVBlocks: 100, BlockSizeTokens: 16})
	assert.Equal(t, int64(100), store.TotalCapacity())
	assert.Equal(t, int64(0), store.UsedBlocks())
}

func TestNewKVStore_ValidConfig_Tiered_Succeeds(t *testing.T) {
	store := NewKVStore(SimConfig{
		TotalKVBlocks:        100,
		BlockSizeTokens:      16,
		KVCPUBlocks:          50,
		KVOffloadThreshold:   0.8,
		KVTransferBandwidth:  1.0,
		KVTransferBaseLatency: 10,
	})
	assert.Equal(t, int64(100), store.TotalCapacity())
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestNewKVStore -v`
Expected: FAIL — no panic for zero TotalKVBlocks (factory doesn't validate)

**Step 3: Add validation to NewKVStore**

In `sim/kv_store.go`, at the top of `NewKVStore`:
```go
func NewKVStore(cfg SimConfig) KVStore {
	if cfg.TotalKVBlocks <= 0 {
		panic(fmt.Sprintf("KVStore: TotalKVBlocks must be > 0, got %d", cfg.TotalKVBlocks))
	}
	if cfg.BlockSizeTokens <= 0 {
		panic(fmt.Sprintf("KVStore: BlockSizeTokens must be > 0, got %d", cfg.BlockSizeTokens))
	}
	// ... existing factory logic
}
```

Add `"fmt"` to imports.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestNewKVStore -v`
Expected: PASS

**Step 5: Lint**

Run: `golangci-lint run ./sim/...`

**Step 6: Commit**

```bash
git add sim/kv_store.go sim/kv_store_test.go
git commit -m "refactor(sim): add NewKVStore input validation (BC-8, BC-12)

- Validate TotalKVBlocks > 0 and BlockSizeTokens > 0
- Panic with descriptive message matching policy factory pattern

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: SetClock on KVStore interface (BC-5)

**Contracts Implemented:** BC-5

**Files:**
- Modify: `sim/kv_store.go` (add to interface)
- Modify: `sim/kvcache.go` (add no-op)
- Modify: `sim/simulator.go` (remove type assertion)
- Test: `sim/kvcache_test.go` (extend)

**Step 1: Write failing test**

In `sim/kvcache_test.go`, add:
```go
func TestKVCacheState_SetClock_IsNoOp(t *testing.T) {
	// BC-5: Single-tier SetClock is a no-op (no observable effect)
	kv := NewKVCacheState(100, 16)
	kv.SetClock(1000) // should not panic or change behavior
	assert.Equal(t, int64(100), kv.TotalCapacity())
}

func TestKVStore_SetClock_InterfaceSatisfied(t *testing.T) {
	// BC-5: Both implementations satisfy KVStore interface including SetClock
	var store KVStore
	store = NewKVCacheState(100, 16)
	store.SetClock(0) // compiles and runs without error

	store = NewTieredKVCache(NewKVCacheState(100, 16), 50, 0.8, 1.0, 10)
	store.SetClock(500)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestKVCacheState_SetClock -v`
Expected: FAIL — `SetClock` not defined on KVCacheState

**Step 3: Implement**

In `sim/kv_store.go`, add to interface:
```go
type KVStore interface {
	// ... existing 9 methods ...
	SetClock(clock int64) // Synchronize clock for time-dependent operations. No-op for single-tier.
}
```

In `sim/kvcache.go`, add method:
```go
// SetClock is a no-op for single-tier KV cache (no time-dependent behavior).
func (kvc *KVCacheState) SetClock(_ int64) {}
```

In `sim/simulator.go`, replace the type assertion (line ~513):
```go
// Before:
// if tiered, ok := sim.KVCache.(*TieredKVCache); ok {
//     tiered.SetClock(now)
// }
// After:
sim.KVCache.SetClock(now)
```

**Step 4: Run tests**

Run: `go test ./sim/... -run TestKV -v`
Expected: PASS

Run: `go test ./sim/... ./sim/cluster/... -count=1`
Expected: PASS (full suite)

**Step 5: Lint**

Run: `golangci-lint run ./sim/...`

**Step 6: Commit**

```bash
git add sim/kv_store.go sim/kvcache.go sim/kvcache_tiered.go sim/simulator.go sim/kvcache_test.go
git commit -m "refactor(sim): add SetClock to KVStore interface (BC-5)

- Add SetClock(int64) to KVStore interface
- KVCacheState implements as no-op
- TieredKVCache already had SetClock (now satisfies interface)
- Remove type assertion in Simulator.Step()

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: PendingTransferLatency non-destructive read (BC-3, BC-4)

**Contracts Implemented:** BC-3, BC-4

**Files:**
- Modify: `sim/kvcache_tiered.go`
- Modify: `sim/simulator.go` (call ConsumePendingTransferLatency)
- Test: `sim/kvcache_tiered_test.go` (extend)

**Step 1: Write failing test**

In `sim/kvcache_tiered_test.go`, add:
```go
func TestTieredKVCache_PendingTransferLatency_PureQuery(t *testing.T) {
	// BC-3: PendingTransferLatency is a pure query (no side effects)
	gpu := NewKVCacheState(100, 16)
	tiered := NewTieredKVCache(gpu, 50, 0.5, 1.0, 10)

	// Force some transfer latency by triggering offload + reload
	// We'll directly set pendingLatency for this test
	tiered.pendingLatency = 42

	first := tiered.PendingTransferLatency()
	second := tiered.PendingTransferLatency()
	assert.Equal(t, int64(42), first)
	assert.Equal(t, int64(42), second, "PendingTransferLatency must be idempotent (BC-3)")
}

func TestTieredKVCache_ConsumePendingTransferLatency_ClearsValue(t *testing.T) {
	// BC-4: ConsumePendingTransferLatency returns value and clears
	gpu := NewKVCacheState(100, 16)
	tiered := NewTieredKVCache(gpu, 50, 0.5, 1.0, 10)
	tiered.pendingLatency = 42

	consumed := tiered.ConsumePendingTransferLatency()
	assert.Equal(t, int64(42), consumed)
	assert.Equal(t, int64(0), tiered.PendingTransferLatency(), "After consume, latency must be 0")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestTieredKVCache_Pending -v`
Expected: FAIL — `ConsumePendingTransferLatency` undefined; `PendingTransferLatency` clears on read

**Step 3: Implement**

In `sim/kvcache_tiered.go`, replace `PendingTransferLatency`:
```go
// PendingTransferLatency returns the accumulated transfer latency without clearing it.
// This is a pure query — no side effects. Use ConsumePendingTransferLatency to read and clear.
func (t *TieredKVCache) PendingTransferLatency() int64 {
	return t.pendingLatency
}

// ConsumePendingTransferLatency returns the accumulated transfer latency and resets it to zero.
// Called by Simulator.Step() to apply latency to the current step.
func (t *TieredKVCache) ConsumePendingTransferLatency() int64 {
	lat := t.pendingLatency
	t.pendingLatency = 0
	return lat
}
```

In `sim/simulator.go`, find where `PendingTransferLatency()` is called and change to `ConsumePendingTransferLatency()`. Search for the call site:
```go
// The call site in Step() that reads and uses pending latency:
// Change: lat := sim.KVCache.PendingTransferLatency()
// To: use ConsumePendingTransferLatency via type assertion since it's not on the interface
```

Note: `ConsumePendingTransferLatency` is NOT on the `KVStore` interface — it's a concrete method on `TieredKVCache` only. The interface method `PendingTransferLatency()` remains a pure query. In `Simulator.Step()`, we need a type assertion to call `ConsumePendingTransferLatency`, OR we can add it to the interface. Since `KVCacheState` has no pending latency (always returns 0), a no-op Consume would work too.

Decision: Add `ConsumePendingTransferLatency()` to the `KVStore` interface for clean usage. `KVCacheState` returns 0 (no-op).

In `sim/kv_store.go`:
```go
type KVStore interface {
	// ... existing methods + SetClock ...
	ConsumePendingTransferLatency() int64 // Read and clear pending transfer latency (0 for single-tier)
}
```

In `sim/kvcache.go`:
```go
func (kvc *KVCacheState) ConsumePendingTransferLatency() int64 { return 0 }
```

Add test for single-tier no-op behavior in `sim/kvcache_test.go`:
```go
func TestKVCacheState_ConsumePendingTransferLatency_AlwaysZero(t *testing.T) {
	kv := NewKVCacheState(100, 16)
	assert.Equal(t, int64(0), kv.ConsumePendingTransferLatency())
	assert.Equal(t, int64(0), kv.ConsumePendingTransferLatency()) // idempotent
}
```

In `sim/simulator.go`, update the call site to use `ConsumePendingTransferLatency()` instead of `PendingTransferLatency()`.

**Step 4: Run tests**

Run: `go test ./sim/... ./sim/cluster/... -count=1`
Expected: PASS

**Step 5: Lint**

Run: `golangci-lint run ./sim/...`

**Step 6: Commit**

```bash
git add sim/kvcache_tiered.go sim/kvcache.go sim/kv_store.go sim/simulator.go sim/kvcache_tiered_test.go
git commit -m "refactor(sim): fix PendingTransferLatency destructive-read (BC-3, BC-4)

- PendingTransferLatency() is now a pure query (no side effects)
- Add ConsumePendingTransferLatency() for read-and-clear
- Add ConsumePendingTransferLatency to KVStore interface
- Simulator.Step() uses ConsumePendingTransferLatency

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: ValidPolicyNames derivation (BC-7)

**Contracts Implemented:** BC-7

**Files:**
- Modify: `sim/bundle.go`
- Modify: `cmd/root.go`
- Test: `sim/bundle_test.go` (extend)

**Step 1: Write failing test**

In `sim/bundle_test.go`, add:
```go
func TestValidAdmissionPolicyNames_ReturnsAllNames(t *testing.T) {
	// BC-7: Names derived from authoritative map
	names := ValidAdmissionPolicyNames()
	assert.Contains(t, names, "always-admit")
	assert.Contains(t, names, "token-bucket")
	assert.Contains(t, names, "reject-all")
	assert.NotContains(t, names, "") // empty string excluded
}

func TestValidRoutingPolicyNames_Sorted(t *testing.T) {
	names := ValidRoutingPolicyNames()
	for i := 1; i < len(names); i++ {
		assert.True(t, names[i-1] < names[i], "names must be sorted: %q >= %q", names[i-1], names[i])
	}
}

func TestValidPriorityPolicyNames_ReturnsAllNames(t *testing.T) {
	names := ValidPriorityPolicyNames()
	assert.Contains(t, names, "constant")
	assert.Contains(t, names, "slo-based")
	assert.Contains(t, names, "inverted-slo")
}

func TestValidSchedulerNames_ReturnsAllNames(t *testing.T) {
	names := ValidSchedulerNames()
	assert.Contains(t, names, "fcfs")
	assert.Contains(t, names, "priority-fcfs")
	assert.Contains(t, names, "sjf")
	assert.Contains(t, names, "reverse-priority")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestValid.*Names -v`
Expected: FAIL — `ValidAdmissionPolicyNames` undefined

**Step 3: Implement**

In `sim/bundle.go`, add after the `IsValid*` functions:
```go
// ValidAdmissionPolicyNames returns sorted valid admission policy names (excluding empty).
func ValidAdmissionPolicyNames() []string { return validNamesList(validAdmissionPolicies) }

// ValidRoutingPolicyNames returns sorted valid routing policy names (excluding empty).
func ValidRoutingPolicyNames() []string { return validNamesList(validRoutingPolicies) }

// ValidPriorityPolicyNames returns sorted valid priority policy names (excluding empty).
func ValidPriorityPolicyNames() []string { return validNamesList(validPriorityPolicies) }

// ValidSchedulerNames returns sorted valid scheduler names (excluding empty).
func ValidSchedulerNames() []string { return validNamesList(validSchedulers) }

// validNamesList returns sorted non-empty keys from a validity map.
func validNamesList(m map[string]bool) []string {
	names := make([]string, 0, len(m))
	for k := range m {
		if k != "" {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	return names
}
```

In `cmd/root.go`, replace hardcoded policy lists (lines ~296-306):
```go
if !sim.IsValidAdmissionPolicy(admissionPolicy) {
	logrus.Fatalf("Unknown admission policy %q. Valid: %s", admissionPolicy, strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
}
if !sim.IsValidRoutingPolicy(routingPolicy) {
	logrus.Fatalf("Unknown routing policy %q. Valid: %s", routingPolicy, strings.Join(sim.ValidRoutingPolicyNames(), ", "))
}
if !sim.IsValidPriorityPolicy(priorityPolicy) {
	logrus.Fatalf("Unknown priority policy %q. Valid: %s", priorityPolicy, strings.Join(sim.ValidPriorityPolicyNames(), ", "))
}
if !sim.IsValidScheduler(scheduler) {
	logrus.Fatalf("Unknown scheduler %q. Valid: %s", scheduler, strings.Join(sim.ValidSchedulerNames(), ", "))
}
```

Ensure `"strings"` is in cmd/root.go imports.

**Step 4: Run tests**

Run: `go test ./sim/... ./cmd/... -count=1`
Expected: PASS

**Step 5: Lint**

Run: `golangci-lint run ./sim/... ./cmd/...`

**Step 6: Commit**

```bash
git add sim/bundle.go sim/bundle_test.go cmd/root.go
git commit -m "refactor(cmd): derive CLI valid-name lists from bundle.go (BC-7)

- Add ValidAdmissionPolicyNames(), ValidRoutingPolicyNames(), etc.
- CLI error messages now auto-derive from authoritative maps
- Adding a new policy template is now a 2-file change (not 3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 6: Simulator observation methods (BC-1)

**Contracts Implemented:** BC-1

**Files:**
- Modify: `sim/simulator.go` (add methods)
- Modify: `sim/cluster/instance.go` (use new methods)
- Test: `sim/simulator_test.go` (extend)

**Step 1: Write failing test**

In `sim/simulator_test.go`, add:
```go
func TestSimulator_ObservationMethods_MatchDirectAccess(t *testing.T) {
	// BC-1: Observation methods return same values as direct field access
	cfg := newTestSimConfig()
	sim := NewSimulator(cfg)

	// Before any events: queue empty, batch empty
	assert.Equal(t, 0, sim.QueueDepth())
	assert.Equal(t, 0, sim.BatchSize())
	assert.Equal(t, int64(0), sim.CurrentClock())
	assert.Equal(t, cfg.Horizon, sim.SimHorizon())

	// Inject a request and verify QueueDepth
	req := &Request{
		ID: "obs-test-1", ArrivalTime: 0,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
		State: StateQueued,
	}
	sim.InjectArrival(req)
	sim.ProcessNextEvent() // process arrival → queued
	assert.Equal(t, 1, sim.QueueDepth())
	assert.Equal(t, sim.WaitQ.Len(), sim.QueueDepth())
}
```

Note: This test needs `NewSimulator` to work. Since Task 8 changes the signature, and this task comes first, we use the current signature. Task 8 will update.

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestSimulator_ObservationMethods -v`
Expected: FAIL — `QueueDepth` undefined on Simulator

**Step 3: Implement observation methods**

In `sim/simulator.go`, add after the `Run()` method:
```go
// QueueDepth returns the number of requests in the wait queue.
func (sim *Simulator) QueueDepth() int { return sim.WaitQ.Len() }

// BatchSize returns the number of requests in the running batch, or 0 if nil.
func (sim *Simulator) BatchSize() int {
	if sim.RunningBatch == nil {
		return 0
	}
	return len(sim.RunningBatch.Requests)
}

// CurrentClock returns the current simulation clock (in ticks).
func (sim *Simulator) CurrentClock() int64 { return sim.Clock }

// SimHorizon returns the simulation horizon (in ticks).
func (sim *Simulator) SimHorizon() int64 { return sim.Horizon }
```

In `sim/cluster/instance.go`, update methods to use new accessors:
```go
func (i *InstanceSimulator) Clock() int64 {
	return i.sim.CurrentClock()
}

func (i *InstanceSimulator) Horizon() int64 {
	return i.sim.SimHorizon()
}

func (i *InstanceSimulator) QueueDepth() int {
	return i.sim.QueueDepth()
}

func (i *InstanceSimulator) BatchSize() int {
	return i.sim.BatchSize()
}
```

Also update `Finalize()`:
```go
func (i *InstanceSimulator) Finalize() {
	i.sim.Finalize()
	i.sim.Metrics.CacheHitRate = i.sim.KVCache.CacheHitRate()
	i.sim.Metrics.KVThrashingRate = i.sim.KVCache.KVThrashingRate()
}
```
(Note: `Finalize` still accesses `i.sim.Metrics` and `i.sim.KVCache` directly — these are deep accessors not covered by the simple observation methods. We keep these as-is; the 4 observation methods cover the high-frequency access patterns.)

**Step 4: Run tests**

Run: `go test ./sim/... ./sim/cluster/... -count=1`
Expected: PASS

**Step 5: Lint**

Run: `golangci-lint run ./sim/... ./sim/cluster/...`

**Step 6: Commit**

```bash
git add sim/simulator.go sim/cluster/instance.go sim/simulator_test.go
git commit -m "refactor(sim): add Simulator observation methods (BC-1)

- Add QueueDepth(), BatchSize(), CurrentClock(), SimHorizon() to Simulator
- InstanceSimulator delegates to these instead of direct field access
- Decouples cluster layer from Simulator internal field layout

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 7: Snapshot unification — RoutingSnapshot as canonical type (BC-2)

**Contracts Implemented:** BC-2

**Files:**
- Modify: `sim/cluster/snapshot.go` (eliminate InstanceSnapshot)
- Modify: `sim/cluster/cluster_event.go` (simplify buildRouterState)
- Modify: `sim/cluster/snapshot_test.go`
- Modify: `sim/cluster/cluster_test.go` (if InstanceSnapshot referenced)

**Step 1: Write failing test**

In `sim/cluster/snapshot_test.go`, add a test that verifies SnapshotProvider returns `sim.RoutingSnapshot`:

```go
func TestCachedSnapshotProvider_ReturnsRoutingSnapshot(t *testing.T) {
	// BC-2: SnapshotProvider returns RoutingSnapshot directly
	cfg := sim.SimConfig{
		Horizon: 1000000, TotalKVBlocks: 100, BlockSizeTokens: 16,
		BetaCoeffs: []float64{1, 1, 1}, AlphaCoeffs: []float64{1, 1, 1},
	}
	inst := NewInstanceSimulator("inst-0", cfg)
	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}

	provider := NewCachedSnapshotProvider(instances, DefaultObservabilityConfig())
	snap := provider.Snapshot("inst-0", 100)

	// Verify it's a RoutingSnapshot with expected fields
	assert.Equal(t, "inst-0", snap.ID)
	assert.Equal(t, 0, snap.QueueDepth)
	assert.Equal(t, 0, snap.BatchSize)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestCachedSnapshotProvider_ReturnsRouting -v`
Expected: FAIL — Snapshot returns InstanceSnapshot, not RoutingSnapshot

**Step 3: Implement snapshot unification**

In `sim/cluster/snapshot.go`:

1. Remove `InstanceSnapshot` struct entirely
2. Change `SnapshotProvider` interface:
```go
type SnapshotProvider interface {
	Snapshot(id InstanceID, clock int64) sim.RoutingSnapshot
	RefreshAll(clock int64)
}
```

3. Update `CachedSnapshotProvider`:
```go
type CachedSnapshotProvider struct {
	instances   map[InstanceID]*InstanceSimulator
	config      ObservabilityConfig
	cache       map[InstanceID]sim.RoutingSnapshot
	lastRefresh map[InstanceID]fieldTimestamps
}

func NewCachedSnapshotProvider(instances map[InstanceID]*InstanceSimulator, config ObservabilityConfig) *CachedSnapshotProvider {
	cache := make(map[InstanceID]sim.RoutingSnapshot, len(instances))
	lastRefresh := make(map[InstanceID]fieldTimestamps, len(instances))
	for id := range instances {
		cache[id] = sim.RoutingSnapshot{ID: string(id)}
		lastRefresh[id] = fieldTimestamps{}
	}
	return &CachedSnapshotProvider{
		instances:   instances,
		config:      config,
		cache:       cache,
		lastRefresh: lastRefresh,
	}
}

func (p *CachedSnapshotProvider) Snapshot(id InstanceID, clock int64) sim.RoutingSnapshot {
	inst := p.instances[id]
	snap := p.cache[id]
	lr := p.lastRefresh[id]

	snap.ID = string(id)

	if p.shouldRefresh(p.config.QueueDepth, lr.QueueDepth, clock) {
		snap.QueueDepth = inst.QueueDepth()
		lr.QueueDepth = clock
	}
	if p.shouldRefresh(p.config.BatchSize, lr.BatchSize, clock) {
		snap.BatchSize = inst.BatchSize()
		lr.BatchSize = clock
	}
	if p.shouldRefresh(p.config.KVUtilization, lr.KVUtilization, clock) {
		snap.KVUtilization = inst.KVUtilization()
		snap.FreeKVBlocks = inst.FreeKVBlocks()
		snap.CacheHitRate = inst.CacheHitRate()
		lr.KVUtilization = clock
	}

	p.cache[id] = snap
	p.lastRefresh[id] = lr
	return snap
}

func (p *CachedSnapshotProvider) RefreshAll(clock int64) {
	for id, inst := range p.instances {
		snap := sim.RoutingSnapshot{
			ID:            string(id),
			QueueDepth:    inst.QueueDepth(),
			BatchSize:     inst.BatchSize(),
			KVUtilization: inst.KVUtilization(),
			FreeKVBlocks:  inst.FreeKVBlocks(),
			CacheHitRate:  inst.CacheHitRate(),
		}
		p.cache[id] = snap
		p.lastRefresh[id] = fieldTimestamps{
			QueueDepth:    clock,
			BatchSize:     clock,
			KVUtilization: clock,
		}
	}
}
```

4. In `sim/cluster/cluster_event.go`, simplify `buildRouterState`:
```go
func buildRouterState(cs *ClusterSimulator) *sim.RouterState {
	snapshots := make([]sim.RoutingSnapshot, len(cs.instances))
	for i, inst := range cs.instances {
		snap := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
		snap.PendingRequests = cs.pendingRequests[string(inst.ID())]
		snapshots[i] = snap
	}
	return &sim.RouterState{
		Snapshots: snapshots,
		Clock:     cs.clock,
	}
}
```

5. Update `sim/cluster/snapshot_test.go` — change all `InstanceSnapshot` references to `sim.RoutingSnapshot`.

**Step 4: Run tests**

Run: `go test ./sim/cluster/... -count=1`
Expected: PASS

**Step 5: Lint**

Run: `golangci-lint run ./sim/cluster/...`

**Step 6: Commit**

```bash
git add sim/cluster/snapshot.go sim/cluster/cluster_event.go sim/cluster/snapshot_test.go
git commit -m "refactor(cluster): unify snapshots — RoutingSnapshot is canonical (BC-2)

- Remove InstanceSnapshot type
- SnapshotProvider returns sim.RoutingSnapshot directly
- buildRouterState injects PendingRequests without translation
- Adding a new observable field is now 3-4 files instead of 6

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 8: Error returns in library code (BC-9, BC-11, BC-13)

**Contracts Implemented:** BC-9, BC-11, BC-13

**Files:**
- Modify: `sim/workload_config.go` (return error from CSV loading)
- Modify: `sim/simulator.go` (NewSimulator returns error)
- Modify: `sim/metrics_utils.go` (remove dead SavetoFile)
- Modify: `sim/cluster/workload.go` (return error from CSV loading)
- Modify: `sim/cluster/cluster.go` (Run returns error from generateRequests)
- Modify: `cmd/root.go` (handle errors from NewSimulator and ClusterSimulator.Run)
- Modify: All test files calling NewSimulator (in sim/ package only)
- Test: `sim/workload_config_test.go` (extend)

**Step 1: Write failing test**

In `sim/workload_config_test.go` (or `sim/simulator_test.go`), add:
```go
func TestNewSimulator_InvalidCSVPath_ReturnsError(t *testing.T) {
	// BC-13: CSV errors propagated as error return
	_, err := NewSimulator(SimConfig{
		Horizon:                1000000,
		TotalKVBlocks:          100,
		BlockSizeTokens:        16,
		BetaCoeffs:             []float64{1, 1, 1},
		AlphaCoeffs:            []float64{1, 1, 1},
		TracesWorkloadFilePath: "/nonexistent/path.csv",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "csv")
}

func TestNewSimulator_ValidConfig_NoError(t *testing.T) {
	// BC-9: Valid config returns nil error
	sim, err := NewSimulator(SimConfig{
		Horizon:         1000000,
		TotalKVBlocks:   100,
		BlockSizeTokens: 16,
		BetaCoeffs:      []float64{1, 1, 1},
		AlphaCoeffs:     []float64{1, 1, 1},
	})
	assert.NoError(t, err)
	assert.NotNil(t, sim)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestNewSimulator_InvalidCSV -v`
Expected: FAIL — NewSimulator doesn't return error (currently calls logrus.Fatalf)

**Step 3: Implement error returns**

This is the largest task. Changes in order:

**3a. `sim/workload_config.go`** — Change `generateWorkloadFromCSV()` to return error:
```go
func (sim *Simulator) generateWorkloadFromCSV() error {
	file, err := os.Open(sim.tracesWorkloadFilePath)
	if err != nil {
		return fmt.Errorf("failed to open csv file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		return fmt.Errorf("failed to read csv header: %w", err)
	}

	reqIdx := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("error reading csv at row %d: %w", reqIdx, err)
		}
		if len(record) < 5 {
			return fmt.Errorf("csv row %d has %d columns, expected at least 5", reqIdx, len(record))
		}

		arrivalFloat, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return fmt.Errorf("invalid arrival time at row %d: %w", reqIdx, err)
		}
		arrivalTime := int64(arrivalFloat * 1e6)
		if arrivalTime > sim.Horizon {
			break
		}

		var inputTokens []int
		var outputTokens []int
		if err := json.Unmarshal([]byte(record[3]), &inputTokens); err != nil {
			return fmt.Errorf("failed to parse prefill_tokens at row %d: %w", reqIdx, err)
		}
		if err := json.Unmarshal([]byte(record[4]), &outputTokens); err != nil {
			return fmt.Errorf("failed to parse decode_tokens at row %d: %w", reqIdx, err)
		}

		reqID := fmt.Sprintf("request_%d", reqIdx)
		req := &Request{
			ID: reqID, ArrivalTime: arrivalTime,
			InputTokens: inputTokens, OutputTokens: outputTokens,
			State: StateQueued, ScheduledStepIdx: 0, FinishedStepIdx: 0,
		}
		sim.InjectArrival(req)
		reqIdx++
	}
	return nil
}
```

Remove `logrus` import from workload_config.go (only needed for Warnf in close, which can use fmt or be removed).

**3b. `sim/simulator.go`** — Change `NewSimulator` signature:
```go
func NewSimulator(cfg SimConfig) (*Simulator, error) {
	// ... existing validation (panics are OK for programming errors) ...

	s := &Simulator{ /* existing fields */ }
	// ...

	if cfg.TracesWorkloadFilePath != "" && cfg.GuideLLMConfig == nil {
		s.Metrics.RequestRate = 0.0
		if err := s.generateWorkloadFromCSV(); err != nil {
			return nil, fmt.Errorf("loading CSV workload: %w", err)
		}
	} else if cfg.GuideLLMConfig != nil {
		s.Metrics.RequestRate = cfg.GuideLLMConfig.Rate
		s.generateWorkloadDistribution()
	}

	return s, nil
}
```

**3c. Update all 24 call sites of NewSimulator:**

For test files in `sim/` package, add a helper at the top of the test file:
```go
// newTestSimulator creates a Simulator for testing, panicking on error.
// Test-only: production code MUST handle the error.
func newTestSimulator(cfg SimConfig) *Simulator {
	s, err := NewSimulator(cfg)
	if err != nil {
		panic(fmt.Sprintf("newTestSimulator: %v", err))
	}
	return s
}
```

Then replace `NewSimulator(cfg)` with `newTestSimulator(cfg)` in test files. Since none of the tests use CSV file paths, this is safe.

Note: `NewInstanceSimulator` must handle the new `NewSimulator` error return to compile, but `DeploymentConfig.ToSimConfig()` intentionally omits `TracesWorkloadFilePath`, so the error path is unreachable in cluster mode. Use panic-on-error inside `NewInstanceSimulator` (matching existing factory panic pattern):

In `sim/cluster/instance.go`:
```go
func NewInstanceSimulator(id InstanceID, cfg sim.SimConfig) *InstanceSimulator {
	s, err := sim.NewSimulator(cfg)
	if err != nil {
		// Unreachable in cluster mode: DeploymentConfig.ToSimConfig() omits TracesWorkloadFilePath.
		// Panic matches existing factory pattern (NewInstanceSimulator already panics on invariant violations).
		panic(fmt.Sprintf("NewInstanceSimulator(%s): %v", id, err))
	}
	return &InstanceSimulator{id: id, sim: s}
}
```

This keeps the `NewInstanceSimulator` signature unchanged (returns `*InstanceSimulator`, not error).

For `cmd/root.go` (line 403-407):
- `cmd/root.go` does NOT call `NewSimulator` directly — all simulation goes through `ClusterSimulator`
- Change `cs.Run()` (line 407) to handle error: `if err := cs.Run(); err != nil { logrus.Fatalf("Simulation failed: %v", err) }`
- `NewSimulator` errors are handled inside `NewInstanceSimulator` via panic (unreachable in cluster mode)

**3d. `sim/cluster/workload.go`** — Change `generateRequestsFromCSV` to return error:
```go
func (c *ClusterSimulator) generateRequestsFromCSV() ([]*sim.Request, error) {
	// ... same pattern as sim/workload_config.go: return error instead of logrus.Fatalf
}

func (c *ClusterSimulator) generateRequests() ([]*sim.Request, error) {
	if len(c.preGeneratedRequests) > 0 {
		return c.preGeneratedRequests, nil
	}
	if c.tracesPath != "" && c.workload == nil {
		return c.generateRequestsFromCSV()
	}
	return c.generateRequestsFromDistribution(), nil
}
```

**3e. `sim/metrics_utils.go`** — Remove dead `SavetoFile` method:
Delete the entire `SavetoFile` method (lines 118-147) and remove `"bufio"` import if unused.

**Step 4: Run tests**

Run: `go test ./sim/... ./sim/cluster/... ./cmd/... -count=1`
Expected: PASS

**Step 5: Lint**

Run: `golangci-lint run ./sim/... ./sim/cluster/... ./cmd/...`

**Step 6: Commit**

```bash
git add sim/workload_config.go sim/simulator.go sim/metrics_utils.go sim/cluster/workload.go sim/cluster/cluster.go sim/cluster/instance.go cmd/root.go sim/simulator_test.go sim/scheduler_test.go sim/simulator_decode_test.go sim/cluster/cluster_test.go sim/cluster/instance_test.go sim/cluster/snapshot_test.go sim/cluster/pending_requests_test.go
git commit -m "refactor(sim): replace logrus.Fatalf with error returns (BC-9, BC-11, BC-13)

- generateWorkloadFromCSV returns error (not logrus.Fatalf)
- NewSimulator returns (*Simulator, error)
- NewInstanceSimulator returns (*InstanceSimulator, error)
- generateRequestsFromCSV returns error
- Remove dead SavetoFile method
- cmd/root.go handles all errors at CLI boundary

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 9: Documentation updates and golden dataset verification (BC-10)

**Contracts Implemented:** BC-10

**Files:**
- Modify: `CLAUDE.md` (update for new methods, interface changes)

**Step 1: Update CLAUDE.md**

Update the following sections:
- File organization: Note `SetClock` + `ConsumePendingTransferLatency` on KVStore interface (now 11 methods)
- `sim/kv_store.go` description: "KVStore interface (11 methods)" instead of 9
- `sim/simulator.go` description: Add observation methods, note `NewSimulator` returns error
- `sim/cluster/snapshot.go` description: Remove InstanceSnapshot mention, note RoutingSnapshot is canonical
- `sim/cluster/instance.go` description: Note `NewInstanceSimulator` returns error
- `sim/request.go` description: Note `RequestState` typed constants
- Adding New Policy Templates: Note step 2 is now 2-file (not 3) since CLI auto-derives names
- Extending KV Cache Tiers: Note `SetClock` is now on the interface
- Implementation Plan Status: Update Phase 6 as completed

**Step 2: Run full test suite**

Run: `go test ./... -count=1`
Expected: PASS (all tests, unchanged golden dataset)

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: No issues

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 6 modularity improvements

- KVStore interface now 11 methods (SetClock, ConsumePendingTransferLatency)
- NewSimulator and NewInstanceSimulator return error
- InstanceSnapshot eliminated, RoutingSnapshot is canonical
- RequestState typed constants
- Adding policy templates is now 2-file change

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 6 | Unit | TestSimulator_ObservationMethods_MatchDirectAccess |
| BC-2 | Task 7 | Unit | TestCachedSnapshotProvider_ReturnsRoutingSnapshot |
| BC-3 | Task 4 | Unit | TestTieredKVCache_PendingTransferLatency_PureQuery |
| BC-4 | Task 4 | Unit | TestTieredKVCache_ConsumePendingTransferLatency_ClearsValue |
| BC-5 | Task 3 | Unit | TestKVCacheState_SetClock_IsNoOp, TestKVStore_SetClock_InterfaceSatisfied |
| BC-6 | Task 1 | Unit | TestRequestState_Constants_HaveExpectedStringValues |
| BC-7 | Task 5 | Unit | TestValidAdmissionPolicyNames_ReturnsAllNames |
| BC-8 | Task 2 | Failure | TestNewKVStore_ZeroTotalBlocks_Panics |
| BC-9 | Task 8 | Unit | TestNewSimulator_ValidConfig_NoError |
| BC-10 | Task 9 | Golden | Existing golden dataset tests (unchanged) |
| BC-11 | Task 8 | — | Enforced by removing logrus import from sim/ files |
| BC-12 | Task 2 | Failure | TestNewKVStore_NegativeTotalBlocks_Panics |
| BC-13 | Task 8 | Unit | TestNewSimulator_InvalidCSVPath_ReturnsError |

**Golden dataset:** NOT updated. Phase 6 is behavioral no-op. Existing golden tests serve as regression guard.

**Invariant tests:** Already exist (Phase 4, PR #222). They continue to pass unchanged.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| NewSimulator signature change breaks callers | Medium | High | Test helper `newTestSimulator` for test files; comprehensive grep | Task 8 |
| SnapshotProvider interface change breaks mocks | Low | Medium | Only one implementation (CachedSnapshotProvider) | Task 7 |
| PendingRequests lost during snapshot unification | Low | High | buildRouterState still injects; test with cluster routing | Task 7 |
| ConsumePendingTransferLatency call site missed | Low | High | Grep for PendingTransferLatency, verify Step() updated | Task 4 |
| RequestState breaks JSON serialization | Low | Medium | Go's fmt.Sprintf with %s works for named string types | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates (BC-9 changes NewSimulator)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing test package
- [x] CLAUDE.md updated (Task 9)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — all deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (6→7, rest independent)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset unchanged (no regeneration needed)
- [x] Construction site audit: Request struct has ~30 construction sites; all updated for RequestState in Task 1
- [x] No new CLI flags (no validation needed)
- [x] Every error path returns error (no silent continue) — Task 8 specifically eliminates logrus.Fatalf
- [x] No map iteration feeds float accumulation without sorted keys
- [x] Library code never calls logrus.Fatalf — enforced by Task 8
- [x] No resource allocation loops in new code
- [x] No exported mutable maps — existing pattern maintained
- [x] No YAML config changes
- [x] No division by zero risks in new code
- [x] KVStore interface accommodates both implementations
- [x] No method spans multiple concerns
- [x] Configuration not modified
- [x] Grepped for "Phase 6" and "PR 213" references — will resolve
- [x] Macro plan status will be updated in CLAUDE.md

---

## Appendix: File-Level Details

### sim/request.go

Add before Request struct:
```go
type RequestState string

const (
	StateQueued    RequestState = "queued"
	StateRunning   RequestState = "running"
	StateCompleted RequestState = "completed"
)
```

Change field: `State RequestState`

### sim/kv_store.go

Interface becomes 11 methods:
```go
type KVStore interface {
	AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool
	GetCachedBlocks(tokens []int) []int64
	ReleaseKVBlocks(req *Request)
	BlockSize() int64
	UsedBlocks() int64
	TotalCapacity() int64
	CacheHitRate() float64
	PendingTransferLatency() int64
	KVThrashingRate() float64
	SetClock(clock int64)
	ConsumePendingTransferLatency() int64
}
```

Factory adds validation:
```go
func NewKVStore(cfg SimConfig) KVStore {
	if cfg.TotalKVBlocks <= 0 {
		panic(fmt.Sprintf("KVStore: TotalKVBlocks must be > 0, got %d", cfg.TotalKVBlocks))
	}
	if cfg.BlockSizeTokens <= 0 {
		panic(fmt.Sprintf("KVStore: BlockSizeTokens must be > 0, got %d", cfg.BlockSizeTokens))
	}
	// ... existing logic
}
```

### sim/kvcache.go

Add two no-op methods:
```go
func (kvc *KVCacheState) SetClock(_ int64) {}
func (kvc *KVCacheState) ConsumePendingTransferLatency() int64 { return 0 }
```

### sim/kvcache_tiered.go

PendingTransferLatency becomes pure:
```go
func (t *TieredKVCache) PendingTransferLatency() int64 { return t.pendingLatency }

func (t *TieredKVCache) ConsumePendingTransferLatency() int64 {
	lat := t.pendingLatency
	t.pendingLatency = 0
	return lat
}
```

### sim/simulator.go

Observation methods + signature change:
```go
func NewSimulator(cfg SimConfig) (*Simulator, error) { ... }
func (sim *Simulator) QueueDepth() int { return sim.WaitQ.Len() }
func (sim *Simulator) BatchSize() int { ... }
func (sim *Simulator) CurrentClock() int64 { return sim.Clock }
func (sim *Simulator) SimHorizon() int64 { return sim.Horizon }
```

In Step(), replace type assertion:
```go
sim.KVCache.SetClock(now)
```

Replace PendingTransferLatency call with ConsumePendingTransferLatency.

### sim/bundle.go

Add 4 exported name accessors + 1 unexported helper:
```go
func ValidAdmissionPolicyNames() []string { return validNamesList(validAdmissionPolicies) }
func ValidRoutingPolicyNames() []string   { return validNamesList(validRoutingPolicies) }
func ValidPriorityPolicyNames() []string  { return validNamesList(validPriorityPolicies) }
func ValidSchedulerNames() []string       { return validNamesList(validSchedulers) }

func validNamesList(m map[string]bool) []string { ... }
```

### sim/cluster/snapshot.go

Remove `InstanceSnapshot`. `SnapshotProvider` returns `sim.RoutingSnapshot`.
`CachedSnapshotProvider` caches `sim.RoutingSnapshot`.

### sim/cluster/cluster_event.go

`buildRouterState` uses snapshots directly, injects PendingRequests:
```go
snap := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
snap.PendingRequests = cs.pendingRequests[string(inst.ID())]
```
