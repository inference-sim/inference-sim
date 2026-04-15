# Event-Driven Cache Signal Propagation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the global periodic poll for prefix cache signal propagation with per-instance event-driven updates, matching how real llm-d propagates KV cache state.

**The problem today:** BLIS refreshes ALL instances' cache snapshots every 2 seconds via a global timer (`StaleCacheIndex.RefreshIfNeeded`). Between refreshes, the PPC scorer is blind to any cache mutations. This creates a ~13-second warm-up artifact where the router thrashes because it can't see recently-cached prefixes — an artifact that doesn't exist in production llm-d. The 2s default was set to match llm-d's `defaultSpeculativeTTL`, but that constant is only used when speculative indexing is enabled (off by default), making it irrelevant.

**What this PR adds:**
1. **Per-instance event-driven refresh** — when an instance completes a step that allocates KV blocks (any step where `AllocateKVBlocks` ran — both prefill and decode), the cluster schedules a `CacheEventArrivalEvent` for that instance at `now + propagation_delay`. When it fires, only that instance's stale snapshot is refreshed. Note: decode steps also trigger events (they allocate 1 block for the new output token), but only prefill allocations change `HashToBlock` entries that PPC cares about. The decode events are harmless — they refresh a snapshot that hasn't semantically changed, at negligible cost (one lightweight event per step).
2. **`RefreshInstance(id)` on StaleCacheIndex** — refreshes one instance's snapshot instead of all.
3. **`AllocationEpoch()` on KVStore** — a lightweight monotonic counter that increments on each successful `AllocateKVBlocks` call. This is the **detection mechanism** that lets the cluster know "something changed in this instance's cache" without violating the `sim/` → `sim/cluster/` layering boundary. See "Why AllocationEpoch?" below for the design rationale.
4. **Renamed flag `--cache-event-delay`** — semantics change from "global poll interval" to "per-event propagation delay." Default changes from 2s to 50ms (50,000µs). Old flag `--cache-signal-delay` kept as deprecated alias.

**Why this matters:** Short simulation runs (< 2K requests) are currently dominated by the warm-up artifact, making them unreliable for cache-aware routing evaluation. This fix brings BLIS's cache signal model closer to production llm-d's event-driven ZMQ propagation.

**Architecture:** Changes span `sim/kv_store.go` (interface: +1 method), `sim/kv/cache.go` and `sim/kv/tiered.go` (implement epoch counter), `sim/cluster/stale_cache.go` (+`RefreshInstance`), `sim/cluster/cluster_event.go` (+`CacheEventArrivalEvent`), `sim/cluster/cluster.go` (detection in main loop, remove `RefreshIfNeeded` call from `buildRouterState`), `cmd/root.go` (flag rename + default change). Documentation updates in invariants, routing guide, CLAUDE.md.

**Why AllocationEpoch? (detection mechanism and layering rationale)**

The cluster needs to know when an instance's KV cache state changed so it can schedule a `CacheEventArrivalEvent`. But `inst.ProcessNextEvent()` is a black box — the cluster calls it, and *something* happens inside (could be a `StepEvent` that allocates blocks, an `ArrivalEvent`, a `TimeoutEvent`, etc.). The cluster has no visibility into what happened.

The `AllocationEpoch()` counter solves this cleanly: compare before/after `ProcessNextEvent()`. If the counter changed, KV blocks were allocated → schedule the delayed cache event. If unchanged, nothing happened → skip.

**Why not alternatives?**

| Alternative | Problem |
|---|---|
| **Callback from `sim/kv/` to `sim/cluster/`** | Violates layering. `sim/kv/` is a pure library with zero knowledge of the cluster. Adding a callback would thread cluster concerns (event scheduling, instance IDs) into the KV cache layer. |
| **Check the event type returned by `ProcessNextEvent()`** | Unreliable. A `StepEvent` doesn't guarantee allocation happened (orphaned steps, empty batches). Non-StepEvents can also trigger allocation (PD transfer path calls `AllocateKVBlocks`). |
| **Add a bool to `BatchResult`** | Invasive. Would require threading allocation awareness through `FormBatch` → `BatchResult` → `Step` → the cluster layer. Touches many files in `sim/` for a concern that only `sim/cluster/` cares about. |
| **Compare `UsedBlockCnt` before/after** | Fragile. Block count can change from both allocation AND release (completions). Would detect false positives from block releases that don't change `HashToBlock`. |

The epoch counter is the least invasive option: one `int64` field in `KVCacheState`, one `++` at the end of successful `AllocateKVBlocks`, one pure query method on the `KVStore` interface. The KV layer remains completely unaware of the cluster.

**Source:** GitHub issue #1029

**Closes:** Fixes #1029

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR replaces the global periodic cache signal refresh with per-instance event-driven refresh. Today, `StaleCacheIndex.RefreshIfNeeded()` is called in `buildRouterState()` and refreshes ALL instances' snapshots every `CacheSignalDelay` µs. After this PR, each instance's snapshot is refreshed individually via a `CacheEventArrivalEvent` that fires `CacheEventDelay` µs after any step that allocated KV blocks. The `RefreshIfNeeded` call in `buildRouterState` is removed.

**Key design decision — per-step-allocation, not per-block-mutation:** Real llm-d emits a ZMQ event for every individual block allocation/eviction. BLIS instead refreshes the entire instance snapshot once after each step that ran `AllocateKVBlocks`. This fires on both prefill steps (where it matters — new prefix hashes appear in `HashToBlock`) and decode steps (where it's a no-op semantically — decode block allocations don't add new prefix hashes, so the refreshed snapshot is identical). This is a deliberate simplification: (1) per-block events would require threading cluster concerns into `sim/kv/`, violating the layering where `sim/` is a pure library; (2) all block events from one prefill step would arrive within the same ~50ms window anyway; (3) evictions only happen inside `AllocateKVBlocks`, so one snapshot after prefill captures both new hashes and any evictions. The behavioral difference from per-block events is negligible. The extra decode-triggered events add minimal overhead (they are lightweight struct pushes to the event heap).

**Adjacent blocks:** `StaleCacheIndex` (owns stale snapshots), `ClusterSimulator` main loop (detection site), `buildRouterState` (consumer of stale cache), `InstanceSimulator` (exposes `AllocationEpoch`), `KVCacheState`/`TieredKVCache` (source of epoch counter).

**Invariants touched:** INV-7 (signal freshness — cacheQueryFn row changes from "demand-triggered interval" to "event-driven per-instance delay").

### B) Behavioral Contracts

**Positive contracts:**

**BC-1: Event-driven refresh trigger**
- GIVEN a cluster with `CacheEventDelay > 0` and `staleCache` active
- WHEN an instance processes a `StepEvent` that causes `AllocateKVBlocks` to run (i.e., the instance's `AllocationEpoch` changes)
- THEN a `CacheEventArrivalEvent` is scheduled for that instance at `clock + CacheEventDelay`
- MECHANISM: The cluster main loop compares `AllocationEpoch()` before/after `ProcessNextEvent()`. If changed, it pushes a `CacheEventArrivalEvent` onto the cluster event queue.

**BC-2: Per-instance snapshot refresh**
- GIVEN a `CacheEventArrivalEvent` for instance X fires
- WHEN the event's `Execute` method runs
- THEN only instance X's snapshot in `StaleCacheIndex` is refreshed; all other instances' snapshots remain unchanged
- MECHANISM: `CacheEventArrivalEvent.Execute` calls `StaleCacheIndex.RefreshInstance(instanceID)`.

**BC-3: No refresh in buildRouterState**
- GIVEN a cluster with `CacheEventDelay > 0`
- WHEN `buildRouterState` is called
- THEN `RefreshIfNeeded` is NOT called (the periodic refresh path is removed)
- MECHANISM: The `RefreshIfNeeded` call in `buildRouterState` is removed. All refresh is now event-driven.

**BC-4: Oracle mode unchanged**
- GIVEN `CacheEventDelay == 0`
- WHEN the PPC scorer queries cache state
- THEN it reads live instance state directly (no stale cache, no events)
- MECHANISM: When delay=0, `staleCache` is nil. No `CacheEventArrivalEvent`s are scheduled. Identical to current behavior.

**BC-5: AllocationEpoch monotonicity**
- GIVEN a KVStore instance
- WHEN `AllocateKVBlocks` is called and returns `true` (successful allocation)
- THEN `AllocationEpoch()` increments by exactly 1
- MECHANISM: An `allocationEpoch` counter in `KVCacheState` increments at the end of a successful `AllocateKVBlocks` call.

**BC-6: Default delay is 50ms**
- GIVEN a user runs `blis run` or `blis replay` without specifying `--cache-event-delay`
- WHEN the cluster is configured
- THEN `CacheEventDelay` is 50,000 µs (50ms)
- MECHANISM: `DefaultCacheEventDelay` constant set to 50,000.

**Negative contracts:**

**BC-7: No stale cache refresh before event fires**
- GIVEN a `CacheEventArrivalEvent` is pending (not yet fired)
- WHEN the router makes a routing decision for instance X
- THEN it sees the OLD snapshot for X (from before the allocation that triggered the event)
- MECHANISM: The snapshot is only updated when `CacheEventArrivalEvent.Execute` runs. The propagation delay models real ZMQ transport time.

**BC-8: No cluster→sim layering violation**
- GIVEN the `AllocationEpoch()` method on `KVStore`
- WHEN the cluster detects cache changes
- THEN it uses only the public `KVStore` interface; no cluster types or events leak into `sim/kv/`
- MECHANISM: `AllocationEpoch()` is a pure query method on `KVStore`. The counter is incremented inside `AllocateKVBlocks` — no callback, no event, no cluster awareness.

**Backward compatibility:**

**BC-9: Deprecated flag alias**
- GIVEN a user passes `--cache-signal-delay 100000`
- WHEN the CLI parses flags
- THEN `CacheEventDelay` is set to 100,000 µs (the old flag still works)
- MECHANISM: `--cache-signal-delay` registered as a deprecated alias for `--cache-event-delay` via Cobra's `MarkDeprecated`.

### C) Component Interaction

```
┌─────────────────────────────────────────────────────┐
│                  ClusterSimulator                     │
│                                                       │
│  Main Loop                                            │
│  ┌───────────────────────────────────────────────┐   │
│  │ for each instance event:                       │   │
│  │   epochBefore = inst.AllocationEpoch()         │   │
│  │   inst.ProcessNextEvent()                      │   │
│  │   epochAfter = inst.AllocationEpoch()          │   │
│  │   if epochAfter > epochBefore && staleCache:   │   │
│  │     push CacheEventArrivalEvent(inst, delay)   │   │
│  └───────────────────────────────────────────────┘   │
│                                                       │
│  buildRouterState()                                   │
│  ┌───────────────────────────────────────────────┐   │
│  │ [REMOVED: staleCache.RefreshIfNeeded(clock)]   │   │
│  │ snapshots := ...                               │   │
│  └───────────────────────────────────────────────┘   │
│                                                       │
│  CacheEventArrivalEvent.Execute()                     │
│  ┌───────────────────────────────────────────────┐   │
│  │ staleCache.RefreshInstance(instanceID)          │   │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

┌────────────────────────┐    ┌─────────────────────────┐
│   StaleCacheIndex       │    │    InstanceSimulator     │
│                         │    │                          │
│ RefreshInstance(id)     │◄───│ AllocationEpoch() int64  │
│ RefreshIfNeeded(clock)  │    │ SnapshotCacheQueryFn()   │
│   [KEPT for tests/      │    │                          │
│    backward compat]     │    └──────────┬───────────────┘
│ Query(id, tokens)       │               │
│ BuildCacheQueryFn()     │               ▼
└─────────────────────────┘    ┌─────────────────────────┐
                               │   KVStore (interface)    │
                               │                          │
                               │ + AllocationEpoch() int64│
                               │   AllocateKVBlocks(...)  │
                               └─────────────────────────┘
```

**State ownership:**
- `allocationEpoch int64` — owned by `KVCacheState` / `TieredKVCache`, incremented inside `AllocateKVBlocks`
- `StaleCacheIndex` — same ownership as today, but `RefreshInstance(id)` added
- `CacheEventArrivalEvent` — transient cluster event, no persistent state

**Extension friction:** Adding one more KVStore implementation requires adding `AllocationEpoch() int64` (1 method, 1 counter field). Touch-point: 1 file per implementation. Acceptable.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| Rename flag to `--cache-event-delay` | Keep `--cache-signal-delay` as deprecated alias | ADDITION — backward compatibility for existing scripts/experiments |
| "Remove `RefreshIfNeeded()`" | Keep method, remove its call from `buildRouterState` | SIMPLIFICATION — method still useful for tests and may serve as fallback; removing the call achieves the behavioral goal without breaking test infrastructure |
| Default 50ms | Default 50,000 µs (50ms) | CLARIFICATION — same value, explicit unit |
| "Add `HadPrefillAllocation` to BatchResult" | Use `AllocationEpoch()` counter on KVStore instead | CORRECTION — BatchResult modifications would thread allocation awareness through FormBatch, which is more invasive. Epoch counter is a pure query at the KVStore interface level, cleaner and more testable |
| No mention of `AllocationEpoch` rollback on failed allocation | Epoch only increments on successful `AllocateKVBlocks` (returns true) | ADDITION — failed allocations don't change `HashToBlock`, so no PPC visibility change |

### E) Review Guide

**The tricky part:** The detection mechanism in the cluster main loop — comparing `AllocationEpoch()` before/after `ProcessNextEvent()`. Verify it handles: (1) multiple StepEvents at the same tick (orphaned StepEvent guard), (2) non-StepEvents that don't touch KV cache, (3) instances with staleCache disabled (oracle mode).

**What to scrutinize:** The `CacheEventArrivalEvent` priority value (10) — it fires AFTER all other cluster events at the same tick (0-9). This means if a `RoutingDecisionEvent` (priority=2) and a `CacheEventArrivalEvent` (priority=10) land on the same tick, routing sees the OLD snapshot. This is fine in practice because the cache event is scheduled at `clock + CacheEventDelay` (future tick), so they rarely coincide. But verify that the delay is always > 0 when staleCache is active (it is: `NewStaleCacheIndex` panics on interval <= 0).

**What's safe to skim:** The `AllocationEpoch` implementation in `KVCacheState` and `TieredKVCache` — trivial counter increment.

**Known debt:** `RefreshIfNeeded` is kept but no longer called in production paths. Could be removed in a future cleanup PR.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/kv_store.go` — add `AllocationEpoch() int64` to `KVStore` interface
- `sim/kv/cache.go` — add `allocationEpoch` field, increment on successful `AllocateKVBlocks`, implement accessor
- `sim/kv/tiered.go` — delegate `AllocationEpoch()` to GPU tier
- `sim/cluster/stale_cache.go` — add `RefreshInstance(id)` method
- `sim/cluster/cluster_event.go` — add `CacheEventArrivalEvent` type
- `sim/cluster/cluster.go` — detection in main loop, remove `RefreshIfNeeded` from `buildRouterState`
- `sim/cluster/instance.go` — add `AllocationEpoch()` accessor
- `sim/cluster/deployment.go` — rename constant, add `CacheEventDelay` field
- `cmd/root.go` — rename flag, change default, add deprecated alias

**Files to create:**
- `sim/cluster/cache_event.go` — `CacheEventArrivalEvent` (small, separate file for clarity)

**Documentation to update:**
- `docs/contributing/standards/invariants.md` — INV-7 cacheQueryFn row
- `docs/guide/routing.md` — cache signal delay section
- `CLAUDE.md` — cache signal delay references

**Key decisions:**
1. `AllocationEpoch` on KVStore interface rather than BatchResult flag — respects layering
2. `CacheEventArrivalEvent` priority = 10 — fires after all existing event types (0-9) at the same timestamp, ensuring the instance has finished its step before the snapshot is taken
3. Keep `RefreshIfNeeded` method (not called in production) — useful for test infrastructure

**Confirmation:** No dead code — `AllocationEpoch` is called in cluster main loop, `RefreshInstance` is called by event, all paths exercised by tests.

### G) Task Breakdown

---

#### Task 1: Add `AllocationEpoch()` to KVStore interface and implementations

**Contracts Implemented:** BC-5, BC-8

**Files:**
- Modify: `sim/kv_store.go:5-20` (interface)
- Modify: `sim/kv/cache.go:36-47` (struct + AllocateKVBlocks)
- Modify: `sim/kv/tiered.go` (delegation)
- Test: `sim/kv/cache_test.go`

**Step 1: Write failing test**

Context: We need a monotonic counter that increments on each successful KV allocation, so the cluster can detect when cache state changed.

```go
// In sim/kv/cache_test.go
func TestAllocationEpoch_IncrementsOnSuccessfulAllocate(t *testing.T) {
	// GIVEN a KV cache with enough capacity
	kvc := NewKVCacheState(10, 4)

	// WHEN no allocations have occurred
	// THEN epoch is 0
	assert.Equal(t, int64(0), kvc.AllocationEpoch())

	// WHEN a successful allocation occurs
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	ok := kvc.AllocateKVBlocks(req, 0, 4, nil)
	assert.True(t, ok)

	// THEN epoch increments to 1
	assert.Equal(t, int64(1), kvc.AllocationEpoch())

	// WHEN another allocation occurs (decode token)
	req.ProgressIndex = 4
	req.OutputTokens = []int{99}
	ok = kvc.AllocateKVBlocks(req, 4, 5, nil)
	assert.True(t, ok)

	// THEN epoch increments to 2
	assert.Equal(t, int64(2), kvc.AllocationEpoch())
}

func TestAllocationEpoch_DoesNotIncrementOnFailedAllocate(t *testing.T) {
	// GIVEN a KV cache with only 1 block of size 2
	kvc := NewKVCacheState(1, 2)

	// Fill the cache
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2}}
	kvc.AllocateKVBlocks(req1, 0, 2, nil)
	epochAfterFill := kvc.AllocationEpoch()

	// WHEN an allocation fails (no free blocks)
	req2 := &sim.Request{ID: "r2", InputTokens: []int{3, 4, 5, 6}}
	ok := kvc.AllocateKVBlocks(req2, 0, 4, nil)

	// THEN epoch does NOT increment
	assert.False(t, ok)
	assert.Equal(t, epochAfterFill, kvc.AllocationEpoch())
}
```

**Step 2: Run test — expect compilation failure (method doesn't exist)**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/kv/... -run TestAllocationEpoch -count=1
```

**Step 3: Implement**

In `sim/kv_store.go`, add to `KVStore` interface:
```go
AllocationEpoch() int64 // Monotonic counter incremented on each successful AllocateKVBlocks call.
```

In `sim/kv/cache.go`, add field to `KVCacheState`:
```go
allocationEpoch int64 // Monotonic counter incremented on each successful AllocateKVBlocks call.
```

At the end of `AllocateKVBlocks`, just before `return true`:
```go
kvc.allocationEpoch++
```

Add accessor:
```go
// AllocationEpoch returns a monotonic counter incremented on each successful AllocateKVBlocks call.
// Used by the cluster layer to detect when an instance's cache state has changed.
func (kvc *KVCacheState) AllocationEpoch() int64 { return kvc.allocationEpoch }
```

In `sim/kv/tiered.go`, add delegation:
```go
// AllocationEpoch delegates to the GPU tier's allocation epoch counter.
func (t *TieredKVCache) AllocationEpoch() int64 { return t.gpu.AllocationEpoch() }
```

**Step 4: Run test — expect pass**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/kv/... -run TestAllocationEpoch -count=1 -v
```

**Step 5: Run lint**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && golangci-lint run ./sim/kv/... ./sim/...
```

**Step 6: Commit**
```bash
git add sim/kv_store.go sim/kv/cache.go sim/kv/tiered.go sim/kv/cache_test.go
git commit -m "feat(kv): add AllocationEpoch() to KVStore interface (BC-5, BC-8)

Monotonic counter incremented on each successful AllocateKVBlocks call.
Enables cluster-layer detection of cache state changes without
violating sim/kv layering boundaries.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Add `RefreshInstance(id)` to StaleCacheIndex

**Contracts Implemented:** BC-2

**Files:**
- Modify: `sim/cluster/stale_cache.go`
- Test: `sim/cluster/stale_cache_test.go`

**Step 1: Write failing test**

Context: The stale cache needs to refresh a single instance's snapshot without touching others.

```go
// In sim/cluster/stale_cache_test.go
func TestStaleCacheIndex_RefreshInstance_OnlyRefreshesTargetInstance(t *testing.T) {
	// GIVEN two instances with stale cache
	inst1 := newTestInstanceWithKV(t, "inst-1")
	inst2 := newTestInstanceWithKV(t, "inst-2")
	instances := map[InstanceID]*InstanceSimulator{
		"inst-1": inst1,
		"inst-2": inst2,
	}
	idx := NewStaleCacheIndex(instances, 2_000_000)

	// Allocate blocks on inst-1 (changes its HashToBlock)
	tokens := []int{1, 2, 3, 4}
	req := &sim.Request{ID: "r1", InputTokens: tokens}
	inst1.sim.KVCache.AllocateKVBlocks(req, 0, 4, nil)

	// WHEN RefreshInstance is called for inst-1 only
	idx.RefreshInstance("inst-1")

	// THEN inst-1's snapshot sees the new blocks
	result1 := idx.Query("inst-1", tokens)
	assert.Greater(t, result1, 0, "inst-1 should see cached blocks after refresh")

	// AND inst-2's snapshot is unchanged (still sees initial empty state)
	result2 := idx.Query("inst-2", tokens)
	assert.Equal(t, 0, result2, "inst-2 should NOT see any blocks (not refreshed)")
}
```

Note: `newTestInstanceWithKV` is a test helper. If it doesn't exist, create a minimal one that constructs an `InstanceSimulator` with a real `KVCacheState`. Check existing test helpers in `stale_cache_test.go` first and reuse them.

**Step 2: Run test — expect compilation failure**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestStaleCacheIndex_RefreshInstance -count=1
```

**Step 3: Implement**

In `sim/cluster/stale_cache.go`, add:
```go
// RefreshInstance updates the stale snapshot for a single instance.
// No-op if the instance ID is not registered. Does not affect other instances' snapshots.
func (s *StaleCacheIndex) RefreshInstance(id InstanceID) {
	e, ok := s.entries[id]
	if !ok {
		return
	}
	e.staleFn = e.inst.SnapshotCacheQueryFn()
	s.entries[id] = e
}
```

**Step 4: Run test — expect pass**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestStaleCacheIndex_RefreshInstance -count=1 -v
```

**Step 5: Run lint**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && golangci-lint run ./sim/cluster/...
```

**Step 6: Commit**
```bash
git add sim/cluster/stale_cache.go sim/cluster/stale_cache_test.go
git commit -m "feat(cluster): add RefreshInstance(id) to StaleCacheIndex (BC-2)

Per-instance snapshot refresh without touching other instances.
Foundation for event-driven cache signal propagation.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Add `AllocationEpoch()` accessor to InstanceSimulator

**Contracts Implemented:** BC-1 (prerequisite)

**Files:**
- Modify: `sim/cluster/instance.go`
- Test: `sim/cluster/instance_test.go`

**Step 1: Write failing test**

```go
// In sim/cluster/instance_test.go
func TestInstanceSimulator_AllocationEpoch_DelegatesToKVStore(t *testing.T) {
	// GIVEN an instance with a KV cache
	cfg := sim.SimConfig{
		KVCacheConfig: sim.KVCacheConfig{
			TotalKVBlocks: 100,
			BlockSizeTokens: 16,
		},
		// Minimal config for construction — add other required fields
	}
	inst := NewInstanceSimulator("test-inst", cfg)

	// WHEN no allocations have occurred
	// THEN epoch is 0
	assert.Equal(t, int64(0), inst.AllocationEpoch())
}
```

Note: Check existing test patterns in `instance_test.go` for how to construct an `InstanceSimulator` for testing. Use the same config patterns.

**Step 2: Run test — expect compilation failure**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestInstanceSimulator_AllocationEpoch -count=1
```

**Step 3: Implement**

In `sim/cluster/instance.go`, add:
```go
// AllocationEpoch returns the KV cache allocation epoch counter.
// Increments on each successful AllocateKVBlocks call. Used by the cluster
// main loop to detect when this instance's cache state changed.
func (i *InstanceSimulator) AllocationEpoch() int64 {
	if i.sim == nil || i.sim.KVCache == nil {
		logrus.Warnf("[cluster] instance %s: nil sim or KVCache in AllocationEpoch — cache events disabled for this instance", i.id)
		return 0
	}
	return i.sim.KVCache.AllocationEpoch()
}
```

**Step 4: Run test — expect pass**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestInstanceSimulator_AllocationEpoch -count=1 -v
```

**Step 5: Run lint**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && golangci-lint run ./sim/cluster/...
```

**Step 6: Commit**
```bash
git add sim/cluster/instance.go sim/cluster/instance_test.go
git commit -m "feat(cluster): add AllocationEpoch() accessor to InstanceSimulator (BC-1 prereq)

Delegates to KVStore.AllocationEpoch() for cluster-layer detection
of cache state changes.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Add CacheEventArrivalEvent and rename config constants

**Contracts Implemented:** BC-2, BC-6

**Files:**
- Create: `sim/cluster/cache_event.go`
- Modify: `sim/cluster/deployment.go`
- Test: `sim/cluster/cache_event_test.go` (new)

**Step 1: Write failing test**

Context: The `CacheEventArrivalEvent` should refresh one instance's stale snapshot when it fires.

```go
// In sim/cluster/cache_event_test.go
package cluster

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCacheEventArrivalEvent_RefreshesTargetInstance(t *testing.T) {
	// GIVEN a cluster with stale cache and 2 instances
	// Build a minimal ClusterSimulator with staleCache set up
	// (Reuse test helpers from stale_cache_test.go)

	// Allocate blocks on instance 1 after initial snapshot

	// WHEN CacheEventArrivalEvent fires for instance 1
	event := &CacheEventArrivalEvent{
		time:       100_000,
		instanceID: "inst-1",
	}
	// Verify timestamp and priority
	assert.Equal(t, int64(100_000), event.Timestamp())
	assert.Equal(t, PriorityCacheEvent, event.Priority())
}
```

Note: The full integration test (event actually refreshing via Execute) will be in Task 6 where we have the complete cluster. This task tests the event structure and its interaction with StaleCacheIndex in isolation.

**Step 2: Run test — expect compilation failure**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestCacheEventArrivalEvent -count=1
```

**Step 3: Implement**

Create `sim/cluster/cache_event.go`:
```go
package cluster

import "github.com/sirupsen/logrus"

// PriorityCacheEvent is the cluster event priority for cache signal propagation events.
// Set to 10 — after all existing event types (0=Arrival through 9=ScaleActuation).
// This ensures the cache snapshot is taken after the instance has fully processed its step.
const PriorityCacheEvent = 10

// CacheEventArrivalEvent models the arrival of a KV cache state change signal
// from an instance to the router's cache index. In production llm-d, this corresponds
// to a ZMQ event traveling from vLLM to the router's KVBlockIndex.
//
// BLIS simplification: one event per step that ran AllocateKVBlocks, not per individual
// block mutation. This is deliberate — all block mutations from one step would arrive
// within the same propagation window, and evictions only happen inside AllocateKVBlocks.
// The behavioral difference from per-block events is negligible.
//
// Scheduled by ClusterSimulator's main loop when it detects AllocationEpoch() changed
// after ProcessNextEvent(). Fires CacheEventDelay µs later.
type CacheEventArrivalEvent struct {
	time       int64
	instanceID InstanceID
}

func (e *CacheEventArrivalEvent) Timestamp() int64 { return e.time }
func (e *CacheEventArrivalEvent) Priority() int     { return PriorityCacheEvent }

// Execute refreshes the stale cache snapshot for the target instance.
func (e *CacheEventArrivalEvent) Execute(cs *ClusterSimulator) {
	if cs.staleCache == nil {
		return // oracle mode — should not happen, but defensive
	}
	logrus.Debugf("[cluster] cache event arrival for instance %s at tick %d", e.instanceID, e.time)
	cs.staleCache.RefreshInstance(e.instanceID)
}
```

In `sim/cluster/deployment.go`, add the new constant and keep the old as deprecated alias:
```go
// DefaultCacheEventDelay is the default per-event propagation delay for prefix cache
// signals in microseconds (50 milliseconds). Models ZMQ transport from vLLM to the
// router's KVBlockIndex in production llm-d.
//
// This differs from the old DefaultCacheSignalDelay (2s) which modeled a global periodic
// poll. The new event-driven model fires per-instance after each step that allocates
// KV blocks, matching llm-d's event-driven ZMQ propagation.
//
// Set to 0 for oracle mode (live cache state).
const DefaultCacheEventDelay int64 = 50_000

// Deprecated: Use DefaultCacheEventDelay instead. Kept for backward compatibility.
const DefaultCacheSignalDelay = DefaultCacheEventDelay
```

**Do NOT rename the `CacheSignalDelay` field in `DeploymentConfig` in this task.** The field rename is a cross-cutting change that must be done atomically in Task 5 alongside all its consumers. See Task 5 for the full rename and construction site enumeration.

**Step 4: Run test — expect pass**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestCacheEventArrivalEvent -count=1 -v
```

**Step 5: Run lint**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && golangci-lint run ./sim/cluster/...
```

**Step 6: Commit**
```bash
git add sim/cluster/cache_event.go sim/cluster/cache_event_test.go sim/cluster/deployment.go
git commit -m "feat(cluster): add CacheEventArrivalEvent and rename to CacheEventDelay (BC-2, BC-6)

New cluster event type for per-instance cache signal propagation.
DefaultCacheEventDelay = 50ms (was 2s global poll).
DefaultCacheSignalDelay kept as deprecated alias.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Wire detection into cluster main loop, remove RefreshIfNeeded from buildRouterState, and rename CacheSignalDelay field

**Contracts Implemented:** BC-1, BC-3, BC-4, BC-7

**Files:**
- Modify: `sim/cluster/cluster.go` (main loop + buildRouterState + staleCache construction)
- Modify: `sim/cluster/cluster_event.go` (remove RefreshIfNeeded call from buildRouterState, update priority comment)
- Modify: `sim/cluster/deployment.go` (rename `CacheSignalDelay` field → `CacheEventDelay`)
- Test: `sim/cluster/stale_cache_test.go` (integration test)

**IMPORTANT — Atomic field rename (R4 construction site audit):**

The `CacheSignalDelay` field in `DeploymentConfig` must be renamed to `CacheEventDelay` in this task atomically with all its consumers. Grep results for all references:

1. `sim/cluster/deployment.go:49` — field declaration
2. `sim/cluster/cluster.go:335,337` — reads `config.CacheSignalDelay` in `newClusterSimulator`
3. `cmd/root.go:1599` — sets `CacheSignalDelay: cacheSignalDelay`
4. `cmd/replay.go:204` — sets `CacheSignalDelay: cacheSignalDelay`
5. `sim/cluster/stale_cache_test.go:221` — test config struct literal
6. `cmd/replay_test.go:423,628,847,1000+` — test variable `origCacheSignalDelay` / `cacheSignalDelay`

All of these must be updated in this task to maintain compilation. The cmd/ references (items 3, 4, 6) use the local variable `cacheSignalDelay` which will be renamed in Task 6 — for now, just update the struct field name in cmd/ construction sites from `.CacheSignalDelay` to `.CacheEventDelay`.

**Step 1: Write failing test**

Context: Integration test — run a minimal cluster with stale cache enabled, inject a request, process events, and verify that the cache signal arrives after the configured delay. Adapt from the existing `TestCluster_CacheSignalDelay_StaleRouting` test pattern.

```go
// In sim/cluster/stale_cache_test.go
func TestCluster_EventDrivenCacheRefresh_RefreshesAfterDelay(t *testing.T) {
	// GIVEN a 2-instance cluster with CacheEventDelay = 100_000 µs (100ms)
	// Use the same config pattern as TestCluster_CacheSignalDelay_StaleRouting:
	// - weighted routing with precise-prefix-cache scorer
	// - KV cache with enough blocks for prefix tokens
	// - roofline latency model
	delay := int64(100_000)
	cfg := minimalStaleTestConfig(t) // reuse/adapt existing helper
	cfg.CacheEventDelay = delay
	cfg.NumInstances = 2

	// Build cluster, inject request with known prefix tokens to instance 0
	// Use the same injection pattern as the existing stale cache test

	// Process events step-by-step using a manual event loop (not cluster.Run())
	// to control timing precisely:

	// PHASE 1: Process until prefill completes on instance 0
	// (advance until StepEvent fires and AllocateKVBlocks runs)
	prefillCompleteTime := /* clock after StepEvent */

	// PHASE 2: Check stale cache BEFORE delay elapses
	// Query staleCache directly for instance 0's tokens
	// ASSERT: stale cache does NOT see the new prefix (BC-7)

	// PHASE 3: Process until CacheEventArrivalEvent fires
	// (advance to prefillCompleteTime + delay)
	// ASSERT: stale cache DOES see the new prefix for instance 0 (BC-1, BC-2)
	// ASSERT: stale cache for instance 1 is unchanged (BC-2)
}
```

Note: The exact helper functions depend on the existing test infrastructure in `stale_cache_test.go`. Read that file and adapt. The key is the temporal assertion: snapshot updates AFTER the delay, not before. If the existing test uses `cluster.Run()`, you may need to use a manual event loop or add clock checkpoints.

**Step 2: Run test — expect failure (detection not wired)**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestCluster_EventDrivenCacheRefresh -count=1
```

**Step 3: Implement**

**3a. Rename field in deployment.go:**

In `sim/cluster/deployment.go`, rename `CacheSignalDelay` to `CacheEventDelay` in the struct definition. Update the doc comment:
```go
// CacheEventDelay is the per-event propagation delay for prefix cache signals
// in microseconds. When > 0, after each step that allocates KV blocks, a
// CacheEventArrivalEvent is scheduled at now + CacheEventDelay to refresh
// that instance's stale snapshot. Models ZMQ transport delay from vLLM to
// the router's KVBlockIndex in production llm-d.
// Default: DefaultCacheEventDelay (50ms).
// 0 = oracle mode (scorers read live cache state with zero delay).
CacheEventDelay int64
```

**3b. Update all construction sites (R4):**

Update struct field references from `CacheSignalDelay` to `CacheEventDelay` in:
- `sim/cluster/cluster.go:335,337` — `config.CacheSignalDelay` → `config.CacheEventDelay`
- `cmd/root.go:1599` — `CacheSignalDelay:` → `CacheEventDelay:`
- `cmd/replay.go:204` — `CacheSignalDelay:` → `CacheEventDelay:`
- `sim/cluster/stale_cache_test.go:221` — test struct literal
- Any other test files that construct `DeploymentConfig` with this field

**3c. Remove RefreshIfNeeded from buildRouterState:**

In `sim/cluster/cluster_event.go`, function `buildRouterState`, **remove** the RefreshIfNeeded call:
```go
// REMOVE these lines (around line 66-69):
// if cs.staleCache != nil {
//     cs.staleCache.RefreshIfNeeded(cs.clock)
// }
```

Update the priority comment on the `ClusterEvent` interface to include `10=CacheEvent`.

**3d. Wire epoch detection in main loop:**

In `sim/cluster/cluster.go`, in the main loop's instance event processing block (after `inst.ProcessNextEvent()`, around line 577), add epoch detection:

Before `ProcessNextEvent()` (before the existing `completedBefore` snapshot), add:
```go
epochBefore := inst.AllocationEpoch()
```

After `ProcessNextEvent()` and after the existing completion-based decrement block, add:
```go
// BC-1: event-driven cache signal propagation.
// When an instance's KV cache allocation epoch changes, a step ran AllocateKVBlocks
// (prefill or decode). Schedule a CacheEventArrivalEvent to refresh that instance's
// stale snapshot after the configured propagation delay.
if c.staleCache != nil {
    epochAfter := inst.AllocationEpoch()
    if epochAfter > epochBefore {
        heap.Push(&c.clusterEvents, clusterEventEntry{
            event: &CacheEventArrivalEvent{
                time:       c.clock + c.config.CacheEventDelay,
                instanceID: inst.ID(),
            },
            seqID: c.nextSeqID(),
        })
    }
}
```

**3e. Update staleCache construction:**

In `newClusterSimulator` (around line 337), update from `CacheSignalDelay` to `CacheEventDelay`:
```go
if config.CacheEventDelay > 0 {
    cs.staleCache = NewStaleCacheIndex(instanceMap, config.CacheEventDelay)
}
```

**3f. Add comment to StaleCacheIndex.interval:**

In `sim/cluster/stale_cache.go`, update the `interval` field comment:
```go
interval    int64 // refresh interval (microseconds). Used by RefreshIfNeeded() only (deprecated path). Event-driven refresh via RefreshInstance() does not use this value.
```

**Step 4: Run test — expect pass**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestCluster_EventDrivenCacheRefresh -count=1 -v
```

**Step 5: Run all cluster tests to verify no regressions**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -count=1 -v
```

**Step 6: Run lint**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && golangci-lint run ./sim/cluster/...
```

**Step 7: Commit**
```bash
git add sim/cluster/cluster.go sim/cluster/cluster_event.go sim/cluster/stale_cache_test.go
git commit -m "feat(cluster): wire event-driven cache signal detection in main loop (BC-1, BC-3, BC-4, BC-7)

Replace global periodic RefreshIfNeeded in buildRouterState with
per-instance CacheEventArrivalEvent scheduled after each step that
allocates KV blocks. Detection via AllocationEpoch before/after
comparison.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 6: Update CLI flags and cmd/root.go

**Contracts Implemented:** BC-6, BC-9

**Files:**
- Modify: `cmd/root.go`
- Modify: `cmd/replay.go` (if it has separate CacheSignalDelay wiring)
- Test: `cmd/simconfig_shared_test.go` or `cmd/replay_test.go`

**Step 1: Write failing test**

Context: Verify the new flag name works and the deprecated alias still works.

```go
// Check existing test patterns in cmd/simconfig_shared_test.go or cmd/replay_test.go.
// The test should verify:
// 1. --cache-event-delay 100000 sets CacheEventDelay to 100000
// 2. --cache-signal-delay 200000 still works (deprecated alias)
// 3. Default value is 50000 (50ms)
```

Note: Adapt existing flag tests in `cmd/simconfig_shared_test.go` or `cmd/replay_test.go`. The exact test pattern depends on how other flags are tested there.

**Step 2: Run test — expect failure**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./cmd/... -run TestCacheEventDelay -count=1
```

**Step 3: Implement**

In `cmd/root.go`, find the `cacheSignalDelay` variable declaration and rename to `cacheEventDelay`. Update the flag registration:

```go
cmd.Flags().Int64Var(&cacheEventDelay, "cache-event-delay", cluster.DefaultCacheEventDelay,
    "Per-event propagation delay for prefix cache signals in microseconds. "+
        "After each step that allocates KV blocks, the instance's cache snapshot refreshes "+
        "after this delay, modeling ZMQ transport in production llm-d. "+
        "Only affects precise-prefix-cache and no-hit-lru scorers. "+
        "Default 50ms. Set to 0 for oracle mode (live cache state).")

// Deprecated alias for backward compatibility
cmd.Flags().Int64Var(&cacheEventDelay, "cache-signal-delay", cluster.DefaultCacheEventDelay,
    "Deprecated: use --cache-event-delay instead.")
cmd.Flags().MarkDeprecated("cache-signal-delay", "use --cache-event-delay instead")
```

Update all references from `cacheSignalDelay` to `cacheEventDelay` and from `CacheSignalDelay` to `CacheEventDelay` in the config struct construction.

Update validation:
```go
if cacheEventDelay < 0 {
    logrus.Fatalf("--cache-event-delay must be >= 0, got %d", cacheEventDelay)
}
```

**Step 4: Run test — expect pass**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./cmd/... -run TestCacheEventDelay -count=1 -v
```

**Step 5: Run all tests to check for regressions**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./... -count=1
```

**Step 6: Run lint**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && golangci-lint run ./...
```

**Step 7: Commit**
```bash
git add cmd/root.go cmd/replay.go cmd/simconfig_shared_test.go cmd/replay_test.go
git commit -m "feat(cmd): rename --cache-signal-delay to --cache-event-delay, default 50ms (BC-6, BC-9)

Semantics change from global poll interval to per-event propagation
delay. Default 50ms (was 2s). Old flag kept as deprecated alias.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 7: Update existing stale cache tests for new semantics

**Contracts Implemented:** BC-1, BC-3, BC-7 (regression verification)

**Files:**
- Modify: `sim/cluster/stale_cache_test.go`

**Step 1: Review and update existing tests**

Context: `TestCluster_CacheSignalDelay_StaleRouting` and `TestCluster_CacheSignalDelay_Zero_OracleBehavior` test the old periodic refresh behavior. Update them:

1. `TestCluster_CacheSignalDelay_StaleRouting` → rename to `TestCluster_CacheEventDelay_StaleRouting`. Update to verify that staleness is now bounded by `CacheEventDelay` per-event, not by a global poll interval.

2. `TestCluster_CacheSignalDelay_Zero_OracleBehavior` → rename to `TestCluster_CacheEventDelay_Zero_OracleBehavior`. Behavior should be identical (oracle mode unchanged, BC-4).

**Step 2: Run updated tests**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./sim/cluster/... -run TestCluster_CacheEventDelay -count=1 -v
```

**Step 3: Run full test suite**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go test ./... -count=1
```

**Step 4: Run lint**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && golangci-lint run ./sim/cluster/...
```

**Step 5: Commit**
```bash
git add sim/cluster/stale_cache_test.go
git commit -m "test(cluster): update stale cache tests for event-driven semantics (BC-1, BC-3, BC-7)

Rename and update tests from periodic poll to event-driven refresh.
Oracle mode behavior unchanged (BC-4).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 8: Update documentation

**Contracts Implemented:** All (documentation)

**Files:**
- Modify: `docs/contributing/standards/invariants.md`
- Modify: `docs/guide/routing.md`
- Modify: `CLAUDE.md`
- Modify: `sim/cluster/stale_cache.go` (doc comments)
- Modify: `sim/routing_precise_prefix_scorer.go` (doc comments)
- Modify: `sim/routing_nohit_lru_scorer.go` (doc comments)

**Step 1: Update invariants.md**

In the INV-7 signal freshness table, update the `cacheQueryFn` row:

Old:
```
| cacheQueryFn (precise-prefix-cache, no-hit-lru) ¹ | Instance (via StaleCacheIndex) | Ground truth (synchronous) | Demand-triggered (CacheSignalDelay interval, checked at routing decisions) | `StaleCacheIndex.RefreshIfNeeded()` in `buildRouterState()` |
```

New:
```
| cacheQueryFn (precise-prefix-cache, no-hit-lru) ¹ | Instance (via StaleCacheIndex) | Ground truth (synchronous) | Event-driven (CacheEventArrivalEvent, per-instance after KV allocation + CacheEventDelay) | `CacheEventArrivalEvent.Execute()` → `StaleCacheIndex.RefreshInstance()` |
```

Update footnote ¹:
```
¹ `cacheQueryFn` freshness is governed by `--cache-event-delay` (default 50ms). When > 0, each instance's snapshot refreshes via CacheEventArrivalEvent after any step that allocates KV blocks plus CacheEventDelay. This models per-instance ZMQ event propagation from vLLM to the router's KVBlockIndex in production llm-d. The "interval=0" column uses live state (oracle mode). BLIS models per-prefill-completion events (one snapshot refresh per step that ran AllocateKVBlocks), not per-block-mutation like llm-d's ZMQ stream — a deliberate simplification that preserves sim/kv layering boundaries with negligible behavioral difference.
```

**Step 2: Update routing.md**

Update the cache-signal-delay tip box to reference the new flag and semantics.

**Step 3: Update CLAUDE.md**

In the "Recent work" or "Cache signal propagation delay" section, update references from `--cache-signal-delay` to `--cache-event-delay` and from "global periodic poll" to "per-instance event-driven". Update the default from 2s to 50ms.

**IMPORTANT — Behavioral change notice:** The default delay changes from 2s (global poll) to 50ms (per-event). This is a behavioral change that affects existing experiment scripts using `--cache-signal-delay` without an explicit value. Add to the CLAUDE.md "Recent Changes" section:

```
- Event-driven cache signal propagation (#1029): `--cache-event-delay` (renamed from `--cache-signal-delay`)
  replaces global periodic poll with per-instance event-driven refresh. Default changed from 2s to 50ms.
  **Breaking change:** Existing scripts using `--cache-signal-delay` without an explicit value will now
  use 50ms delay instead of 2s. To restore old behavior, pass `--cache-event-delay 2000000` explicitly.
  Old flag `--cache-signal-delay` kept as deprecated alias.
```

**Step 4: Update doc comments in stale_cache.go**

Update the `StaleCacheIndex` type doc comment to reflect event-driven refresh as the primary mechanism. Note that `RefreshIfNeeded` is retained for backward compatibility but not used in the production path.

**Step 5: Update doc comments in scorer files**

Update `sim/routing_precise_prefix_scorer.go` and `sim/routing_nohit_lru_scorer.go` comments that reference `--cache-signal-delay` and "speculative TTL".

**Step 6: Run lint and build**
```bash
cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork3-/inference-sim/.worktrees/pr1029-event-driven-cache-signal && go build ./... && golangci-lint run ./...
```

**Step 7: Commit**
```bash
git add docs/contributing/standards/invariants.md docs/guide/routing.md CLAUDE.md \
    sim/cluster/stale_cache.go sim/routing_precise_prefix_scorer.go sim/routing_nohit_lru_scorer.go
git commit -m "docs: update cache signal propagation documentation for event-driven model

Update INV-7 table, routing guide, CLAUDE.md, and code comments to
reflect per-instance event-driven refresh (CacheEventArrivalEvent)
replacing global periodic poll (RefreshIfNeeded). Default changed
from 2s to 50ms. Flag renamed --cache-event-delay.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Test | Type |
|---|---|---|
| BC-1 | `TestCluster_EventDrivenCacheRefresh_RefreshesAfterDelay` | Integration |
| BC-2 | `TestStaleCacheIndex_RefreshInstance_OnlyRefreshesTargetInstance` | Unit |
| BC-3 | `TestCluster_EventDrivenCacheRefresh_RefreshesAfterDelay` (no RefreshIfNeeded in buildRouterState) | Integration |
| BC-4 | `TestCluster_CacheEventDelay_Zero_OracleBehavior` | Integration |
| BC-5 | `TestAllocationEpoch_IncrementsOnSuccessfulAllocate`, `TestAllocationEpoch_DoesNotIncrementOnFailedAllocate` | Unit |
| BC-6 | CLI flag test (default = 50000) | Unit |
| BC-7 | `TestCluster_EventDrivenCacheRefresh_RefreshesAfterDelay` (temporal assertion) | Integration |
| BC-8 | Structural — no cluster imports in sim/kv/ | Compile-time |
| BC-9 | CLI flag test (deprecated alias works) | Unit |

**Golden dataset:** No golden dataset changes. The trained-physics golden dataset is not affected — it tests latency prediction, not routing.

**Shared test infrastructure:** Reuse existing test helpers from `stale_cache_test.go` (instance construction, config patterns).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Existing tests depend on periodic refresh timing | Medium | Medium | Task 7 explicitly updates existing stale cache tests |
| `CacheEventArrivalEvent` priority ordering conflict | Low | High | Priority 10 is after all existing priorities (0-9), verified in code |
| Breaking experiment scripts that use `--cache-signal-delay` | Medium | Low | Deprecated alias keeps old flag working (BC-9) |
| Epoch detection misses non-StepEvent allocations (e.g., PD transfer) | Low | Low | `AllocateTransferredKV` in PD path also calls `AllocateKVBlocks`, so epoch increments correctly |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [ ] All behavioral contracts have corresponding tests
- [ ] No dead code — every new method/field is exercised
- [ ] `AllocationEpoch()` only increments on successful allocation (return true)
- [ ] `RefreshIfNeeded` is NOT called anywhere in production paths (only tests)
- [ ] `CacheEventArrivalEvent` priority > all existing cluster event priorities
- [ ] Oracle mode (delay=0) behavior is unchanged
- [ ] `--cache-signal-delay` deprecated alias works
- [ ] Default changed from 2,000,000 to 50,000 in `DefaultCacheEventDelay`
- [ ] INV-7 table updated in invariants.md
- [ ] CLAUDE.md updated
- [ ] All construction sites for `DeploymentConfig` updated for field rename
- [ ] `go test ./... -count=1` passes
- [ ] `golangci-lint run ./...` passes

---

## Appendix: File-Level Implementation Details

### sim/kv_store.go
Add to `KVStore` interface:
```go
AllocationEpoch() int64
```

### sim/kv/cache.go
Add field `allocationEpoch int64` to `KVCacheState`. Increment `kvc.allocationEpoch++` at the end of `AllocateKVBlocks` just before `return true` (line ~338). Add accessor `func (kvc *KVCacheState) AllocationEpoch() int64`.

### sim/kv/tiered.go
Add `func (t *TieredKVCache) AllocationEpoch() int64 { return t.gpu.AllocationEpoch() }`.

### sim/cluster/instance.go
Add `func (i *InstanceSimulator) AllocationEpoch() int64` delegating to `i.sim.KVCache.AllocationEpoch()`.

### sim/cluster/cache_event.go (new)
`CacheEventArrivalEvent` struct with `time int64`, `instanceID InstanceID`. Priority `PriorityCacheEvent = 10`. `Execute` calls `cs.staleCache.RefreshInstance(e.instanceID)`.

### sim/cluster/stale_cache.go
Add `func (s *StaleCacheIndex) RefreshInstance(id InstanceID)` — refreshes one entry's `staleFn`.

### sim/cluster/deployment.go
Add `DefaultCacheEventDelay` (50,000). Keep `DefaultCacheSignalDelay` as deprecated alias. Rename `CacheSignalDelay` field → `CacheEventDelay` in `DeploymentConfig` (done in Task 5 atomically with all consumers).

### sim/cluster/cluster.go
1. In main loop (line ~577): add `epochBefore`/`epochAfter` detection around `ProcessNextEvent()`, push `CacheEventArrivalEvent` when epoch changes.
2. In `newClusterSimulator` (line ~337): update `CacheSignalDelay` → `CacheEventDelay`.

### sim/cluster/cluster_event.go
Remove `RefreshIfNeeded` call from `buildRouterState` (lines 67-69). Update priority comment to include `10=CacheEvent`.

### cmd/root.go
Rename variable, flag, validation. Add deprecated alias via `MarkDeprecated`.
