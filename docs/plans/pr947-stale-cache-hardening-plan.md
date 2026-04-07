# StaleCacheIndex Post-#932 Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close six hardening gaps in the `--cache-signal-delay` feature introduced in PR #932: a structural fragility (two maps that must stay in sync), a R23 code-path duplication, three missing tests, and two inaccurate comments.

**The problem today:** `StaleCacheIndex` maintains two parallel maps (`instances` and `staleFns`) with different key types that must always be mutated together — a maintenance trap that grows riskier as dynamic instance management (autoscaler) is added. The oracle/stale branching logic for building `cacheQueryFn` entries is duplicated across two construction sites (`NewClusterSimulator` and `NodeReadyEvent.Execute`), violating R23. Three defensive guards have no test coverage, and two doc comments are inaccurate.

**What this PR adds:**
1. **Single-map consolidation** — `StaleCacheIndex` uses one `map[InstanceID]instanceCacheEntry` struct, making `instances`/`staleFns` divergence structurally impossible.
2. **`registerInstanceCacheQueryFn` helper** — eliminates the R23 dual-path oracle/stale branching by centralising single-instance registration in one method on `ClusterSimulator`.
3. **Three new tests** — `AddInstance` duplicate-panic, post-`RemoveInstance` query returning 0, and `RefreshIfNeeded` fencepost at `clock = interval − 1`.
4. **Comment fixes** — `cluster.go:294` ("Oracle mode (default)" → "Zero-delay / oracle mode") and a co-change guard linking `GetCachedBlocks` to `SnapshotCachedBlocksFn`.

**Why this matters:** The dual-map and R23 patterns are low-risk today but will be the first place a bug appears when the autoscaler PR adds dynamic instance removal at high frequency. Fixing them now keeps the invariant surface clean for downstream work.

**Architecture:** All changes are confined to `sim/cluster/stale_cache.go` (data structure refactor), `sim/cluster/cluster.go` + `sim/cluster/infra_lifecycle_event.go` (helper extraction), `sim/cluster/stale_cache_test.go` (new tests), and `sim/kv/cache.go` (comment). No interface changes, no CLI changes, no behavioral changes.

**Source:** GitHub issue #947

**Closes:** Fixes #947

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This is a pure hardening PR: **no behavioral changes to the simulator**. Routing decisions, request metrics, and all outputs remain byte-identical for every existing configuration. The three changes that touch production code (single-map refactor, helper extraction, comment fixes) are structurally equivalent to the current code; all existing tests act as regression guards.

**What comes before this PR:** PR #932 (merged) added `StaleCacheIndex` and `--cache-signal-delay`. This PR cleans up the known post-merge debt identified in issue #947.

**What depends on this PR:** The autoscaler PR (PR11 in the macro plan) will add high-frequency `AddInstance`/`RemoveInstance` calls. The single-map consolidation is the most important hardening step to have in place before that work begins.

**Adjacent blocks touched:**
- `StaleCacheIndex` → `InstanceSimulator` (via `inst.SnapshotCacheQueryFn()`)
- `ClusterSimulator` → `StaleCacheIndex` (construction, refresh, registration)
- `KVCacheState.GetCachedBlocks` / `SnapshotCachedBlocksFn` (comment co-change guard)

**DEVIATION flags from Step 1.5 audit:**
- Issue #947 item 6 offers two options (extract shared helper OR add co-change comment). This plan chooses the comment option (CLARIFICATION — shared helper would require changing `GetCachedBlocks` return semantics: nil vs empty slice). See Deviation Log.

---

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: AddInstance duplicate protection**
- GIVEN a `StaleCacheIndex` with instance ID "X" already registered
- WHEN `AddInstance` is called again with the same ID "X"
- THEN the call panics with a message containing "already registered"
- MECHANISM: Existence check in `AddInstance` panics before any mutation

**BC-2: Post-removal Query returns 0**
- GIVEN a `StaleCacheIndex` from which instance "X" has been removed via `RemoveInstance`
- WHEN `Query("X", tokens)` is called
- THEN the call returns 0 (with a logged warning)
- MECHANISM: The entry is absent from the map; the `Query` fallback returns 0

**BC-3: RefreshIfNeeded boundary precision**
- GIVEN a `StaleCacheIndex` with interval I and `lastRefresh = 0`
- WHEN `RefreshIfNeeded(I - 1)` is called
- THEN the stale snapshot is NOT refreshed (Query returns values from the snapshot taken at construction)
- MECHANISM: Strict `<` comparison: `clock - lastRefresh < interval` — at `clock = I−1`, `(I−1) − 0 = I−1 < I` is true, so return without refresh

**BC-4: Single-map structural consistency**
- GIVEN any sequence of `AddInstance` / `RemoveInstance` operations on a `StaleCacheIndex`
- WHEN `Query(id, tokens)` and `BuildCacheQueryFn()[id](tokens)` are both called for the same `id`
- THEN both calls return the same value (both read from the same underlying `instanceCacheEntry`)
- MECHANISM: Single `map[InstanceID]instanceCacheEntry` replaces the dual `instances`/`staleFns` maps

**BC-5: registerInstanceCacheQueryFn unified path (R23)**
- GIVEN a `ClusterSimulator` in either oracle or stale mode
- WHEN a new instance is registered via the constructor (oracle path) OR via `NodeReadyEvent`
- THEN the resulting `cacheQueryFn[id]` closure behaves identically to what was produced by the old per-site code
- MECHANISM: `registerInstanceCacheQueryFn` encapsulates the oracle/stale branch, called at both sites

#### Negative Contracts

**BC-6: No behavioral change to simulation output**
- GIVEN any simulation configuration that currently runs correctly
- WHEN this PR is applied
- THEN all metrics (completed requests, per-instance counts, latencies) are byte-identical to pre-PR values
- MECHANISM: Pure refactoring — no changes to routing logic, DES event processing, or metric accumulation

**BC-7: No regression in staleness semantics**
- GIVEN a cluster with `CacheSignalDelay > 0` and `precise-prefix-cache` scorer
- WHEN the simulation runs
- THEN oracle mode (delay=0) still concentrates more requests on the cache-warm instance than stale mode (large delay)
- MECHANISM: Existing `TestCluster_CacheSignalDelay_StaleRouting` confirms this; refactoring preserves the behavior

#### Error Handling Contracts

**BC-8: Comment accuracy (non-behavioral)**
- GIVEN `cluster.go:294` comment
- THEN it reads "Zero delay (CacheSignalDelay=0) — oracle mode" rather than "Oracle mode (default)"

**BC-9: Algorithm co-change guard**
- GIVEN `KVCacheState.GetCachedBlocks`
- THEN it has a doc comment noting that `SnapshotCachedBlocksFn` replicates the same algorithm and must be updated when this method changes

---

### C) Component Interaction

```
ClusterSimulator
  ├── cacheQueryFn map[string]func([]int)int ◄──── built by registerInstanceCacheQueryFn (NEW)
  │     (oracle mode: wraps inst.GetCachedBlockCount)
  │     (stale mode: delegates to StaleCacheIndex.Query)
  │
  └── staleCache *StaleCacheIndex (nil in oracle mode)
        └── entries map[InstanceID]instanceCacheEntry (NEW — replaces two maps)
              ├── inst  *InstanceSimulator  (for snapshot refresh)
              └── staleFn func([]int)int    (frozen query closure)

NodeReadyEvent.Execute → cs.registerInstanceCacheQueryFn(id, inst) (NEW)
NewClusterSimulator   → cs.registerInstanceCacheQueryFn(id, inst) (oracle path, NEW)
                      → NewStaleCacheIndex + BuildCacheQueryFn    (stale path, unchanged)
```

**New types/methods:**
- `instanceCacheEntry{inst *InstanceSimulator; staleFn func([]int) int}` — unexported, in `stale_cache.go`
- `(*ClusterSimulator).registerInstanceCacheQueryFn(id InstanceID, inst *InstanceSimulator)` — unexported, in `cluster.go`

**State changes:** `StaleCacheIndex.instances` and `StaleCacheIndex.staleFns` are replaced by `StaleCacheIndex.entries`. Lifecycle (created, destroyed, accessed) is unchanged — the same three sites mutate it.

**Extension friction:** To add one more field to `instanceCacheEntry` in the future: 1 file (`stale_cache.go`). Adding a field to `StaleCacheIndex` itself: 1 file (`stale_cache.go`). ✅ Friction is 1 file.

---

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| Item 6: "Extract shared `countPrefixBlocks` helper OR add co-change comment" | Adds co-change comment only | CLARIFICATION: `GetCachedBlocks` returns `[]int64` (block IDs); the helper would return only a count. Unifying them requires either changing return semantics (nil vs empty slice regression risk) or producing two return values. The comment option guards against divergence with zero regression risk. The refactor option can be a follow-up. |

---

### E) Review Guide

**The tricky part:** Task 2 (single-map consolidation) replaces two maps with different key types (`InstanceID` vs `string`). After the refactor, `Query(instanceID string, tokens []int) int` must convert `instanceID` to `InstanceID` internally. Verify the type conversion is consistent everywhere.

**What to scrutinize:** BC-4 (single-map consistency) and BC-5 (R23 unified path). Confirm `registerInstanceCacheQueryFn` is called at both registration sites and that the stale-mode constructor path (`NewStaleCacheIndex` + `BuildCacheQueryFn`) is left untouched (it's a bulk API, not the same as single-instance registration).

**What's safe to skim:** Task 1 (pure test additions — no production code changes), Task 4 (comment-only edits).

**Known debt:** The `SnapshotCachedBlocksFn` algorithm duplication is documented by the new co-change comment but not eliminated. Full deduplication (shared helper) is deferred due to nil-vs-empty-slice return semantics.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Change |
|------|--------|
| `sim/cluster/stale_cache.go` | Replace dual maps with `instanceCacheEntry`; extract `warnIfNotSnapshotCapable` helper |
| `sim/cluster/stale_cache_test.go` | Add 3 new tests; one existing test extended |
| `sim/cluster/cluster.go` | Add `registerInstanceCacheQueryFn` method; fix comment at line 294; use helper in oracle constructor loop |
| `sim/cluster/infra_lifecycle_event.go` | Replace 15-line oracle/stale if/else with `cs.registerInstanceCacheQueryFn(p.id, inst)` |
| `sim/kv/cache.go` | Add co-change comment to `GetCachedBlocks` |

**Key decisions:**
- `instanceCacheEntry` is unexported (only used within `stale_cache.go`).
- `registerInstanceCacheQueryFn` is a method on `*ClusterSimulator` (needs access to both `staleCache` and `cacheQueryFn`), unexported.
- The stale-mode constructor path (`NewStaleCacheIndex` + `BuildCacheQueryFn`) is NOT changed — it's a bulk initialization API distinct from single-instance registration.
- No new exported symbols.

**No dead code:** Every new method/type is used immediately in the same PR.

---

### G) Task Breakdown

---

### Task 1: Add missing test coverage for AddInstance panic and post-removal query (BC-1, BC-2)

**Contracts implemented:** BC-1, BC-2

**Files:**
- Modify (tests only): `sim/cluster/stale_cache_test.go`

**Step 1: Write failing tests**

These tests test existing, correct code — they will pass immediately. Their purpose is to lock in the defensive guards.

In `sim/cluster/stale_cache_test.go`, after `TestStaleCacheIndex_AddInstance` (line ~68), add:

```go
func TestStaleCacheIndex_AddInstance_DuplicateID_Panics(t *testing.T) {
	// GIVEN a StaleCacheIndex with instance "inst-0" already registered
	cfg := newTestSimConfig()
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)
	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	idx := NewStaleCacheIndex(instances, 1000)

	// WHEN AddInstance is called with the same ID again
	// THEN it panics with a message containing "already registered"
	defer func() {
		r := recover()
		assert.NotNil(t, r, "expected panic for duplicate instance ID")
		assert.Contains(t, fmt.Sprintf("%v", r), "already registered")
	}()
	idx.AddInstance("inst-0", inst)
}
```

In `TestStaleCacheIndex_RemoveInstance` (line ~228), after `idx.RefreshIfNeeded(2000)`, add:

```go
	// AND Query for the removed instance returns 0 (warn-and-return-0 path)
	assert.Equal(t, 0, idx.Query("inst-0", tokens), "query for removed instance should return 0")
```

**Step 2: Run tests to verify they pass**

```
go test ./sim/cluster/... -run "TestStaleCacheIndex_AddInstance_DuplicateID_Panics|TestStaleCacheIndex_RemoveInstance" -v
```
Expected: PASS for both.

**Step 3: No implementation change** (tests document existing behavior)

**Step 4: Run lint**

```
golangci-lint run ./sim/cluster/...
```
Expected: No new issues.

**Step 5: Commit**

```bash
git add sim/cluster/stale_cache_test.go
git commit -m "test(cluster): add AddInstance duplicate-panic and post-RemoveInstance query tests (BC-1, BC-2)

- TestStaleCacheIndex_AddInstance_DuplicateID_Panics: locks in the defensive
  panic guard for duplicate instance registration
- TestStaleCacheIndex_RemoveInstance: extended with post-removal Query assertion
  to verify the warn-and-return-0 path

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add RefreshIfNeeded fencepost test at clock = interval − 1 (BC-3)

**Contracts implemented:** BC-3

**Files:**
- Modify (tests only): `sim/cluster/stale_cache_test.go`

**Step 1: Write the fencepost test**

This is a precision test for the `<` vs `<=` boundary in `RefreshIfNeeded`. Add after `TestStaleCacheIndex_StaleUntilRefresh`:

```go
func TestStaleCacheIndex_RefreshIfNeeded_BoundaryAtIntervalMinusOne(t *testing.T) {
	// GIVEN a StaleCacheIndex with interval=1000 and a populated cache
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	idx := NewStaleCacheIndex(instances, 1000) // lastRefresh=0

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Populate cache
	req := &sim.Request{
		ID:           "r1",
		ArrivalTime:  0,
		InputTokens:  tokens,
		OutputTokens: []int{100},
		State:        sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()
	require.Greater(t, inst.GetCachedBlockCount(tokens), 0, "live cache must have blocks")

	// WHEN RefreshIfNeeded is called at exactly clock = interval - 1 = 999
	// THEN the snapshot is NOT refreshed (999 - 0 = 999 < 1000)
	idx.RefreshIfNeeded(999)
	assert.Equal(t, 0, idx.Query("inst-0", tokens),
		"snapshot must NOT be refreshed at clock=interval-1 (strict < boundary)")

	// WHEN RefreshIfNeeded is called at clock = interval = 1000
	// THEN the snapshot IS refreshed (1000 - 0 = 1000 >= 1000)
	idx.RefreshIfNeeded(1000)
	assert.Greater(t, idx.Query("inst-0", tokens), 0,
		"snapshot must be refreshed at clock=interval (>= threshold)")
}
```

**Step 2: Run test**

```
go test ./sim/cluster/... -run TestStaleCacheIndex_RefreshIfNeeded_BoundaryAtIntervalMinusOne -v
```
Expected: PASS.

**Step 3: No implementation change**

**Step 4: Run lint**

```
golangci-lint run ./sim/cluster/...
```
Expected: No new issues.

**Step 5: Commit**

```bash
git add sim/cluster/stale_cache_test.go
git commit -m "test(cluster): add RefreshIfNeeded fencepost test at clock=interval-1 (BC-3)

Verifies strict < boundary: clock=interval-1 must NOT refresh, clock=interval must.
Guards against fencepost errors (< vs <=) in RefreshIfNeeded.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Consolidate StaleCacheIndex to single map (BC-4)

**Contracts implemented:** BC-4, BC-6

**Files:**
- Modify: `sim/cluster/stale_cache.go`
- Modify (tests): `sim/cluster/stale_cache_test.go` (existing tests act as regression suite)

**Step 1: Run existing tests to confirm baseline**

```
go test ./sim/cluster/... -count=1 -v 2>&1 | grep -E "^(=== RUN|--- PASS|--- FAIL|FAIL|ok)"
```
Expected: All pass.

**Step 2: Rewrite `stale_cache.go`**

Replace the entire file contents with:

```go
package cluster

import (
	"fmt"

	"github.com/sirupsen/logrus"
)

// instanceCacheEntry holds a live instance reference and its current stale snapshot closure.
// The entry is owned exclusively by StaleCacheIndex.
type instanceCacheEntry struct {
	inst    *InstanceSimulator
	staleFn func([]int) int
}

// StaleCacheIndex manages per-instance frozen snapshots of KV cache hash maps.
// When cache-signal-delay > 0, the cacheQueryFn closures delegate to this index
// instead of querying live instance state, simulating asynchronous KV event
// propagation from production llm-d (issue #919).
//
// Signal freshness (R17, INV-7):
//
//	Reads: InstanceSimulator.SnapshotCacheQueryFn() snapshots (delegates to
//	KVCacheState.SnapshotCachedBlocksFn via cacheSnapshotCapable) — demand-triggered staleness.
//	Refresh checked at routing-decision boundaries (buildRouterState), not by an
//	independent timer. During idle simulation periods, staleness can exceed the
//	nominal CacheSignalDelay interval. Controlled by DeploymentConfig.CacheSignalDelay.
//	When delay=0, this type is not used (oracle mode).
//	Default delay is 2s (DefaultCacheSignalDelay), matching llm-d's speculative TTL.
type StaleCacheIndex struct {
	entries     map[InstanceID]instanceCacheEntry
	interval    int64 // refresh interval (microseconds)
	lastRefresh int64 // sim clock at last refresh
}

// NewStaleCacheIndex creates a StaleCacheIndex and takes an initial snapshot of all instances.
// interval is the refresh interval in simulated microseconds. Panics if interval <= 0.
// instances may be nil or empty (valid for testing).
func NewStaleCacheIndex(instances map[InstanceID]*InstanceSimulator, interval int64) *StaleCacheIndex {
	if interval <= 0 {
		panic(fmt.Sprintf("NewStaleCacheIndex: interval must be > 0, got %d", interval))
	}
	idx := &StaleCacheIndex{
		entries:     make(map[InstanceID]instanceCacheEntry, len(instances)),
		interval:    interval,
		lastRefresh: 0,
	}
	for id, inst := range instances {
		warnIfNotSnapshotCapable(id, inst)
		idx.entries[id] = instanceCacheEntry{
			inst:    inst,
			staleFn: inst.SnapshotCacheQueryFn(),
		}
	}
	return idx
}

// RefreshIfNeeded updates all stale snapshots if the refresh interval has elapsed.
// No-op if clock - lastRefresh < interval.
func (s *StaleCacheIndex) RefreshIfNeeded(clock int64) {
	if clock-s.lastRefresh < s.interval {
		return
	}
	for id, e := range s.entries {
		e.staleFn = e.inst.SnapshotCacheQueryFn()
		s.entries[id] = e
	}
	s.lastRefresh = clock
}

// Query returns the cached block count for the given instance and tokens,
// using the stale snapshot. Returns 0 if the instance is unknown.
func (s *StaleCacheIndex) Query(instanceID string, tokens []int) int {
	if e, ok := s.entries[InstanceID(instanceID)]; ok {
		return e.staleFn(tokens)
	}
	logrus.Warnf("[stale-cache] Query for unknown instance %q — returning 0", instanceID)
	return 0
}

// RemoveInstance unregisters an instance (e.g., on termination) and frees its
// snapshot closure. No-op if the instance is not registered.
func (s *StaleCacheIndex) RemoveInstance(id InstanceID) {
	delete(s.entries, id)
}

// AddInstance registers a new instance (e.g., from NodeReadyEvent) and takes
// an initial snapshot. Panics if the instance ID is already registered.
func (s *StaleCacheIndex) AddInstance(id InstanceID, inst *InstanceSimulator) {
	if _, exists := s.entries[id]; exists {
		panic("StaleCacheIndex.AddInstance: instance " + string(id) + " already registered")
	}
	warnIfNotSnapshotCapable(id, inst)
	s.entries[id] = instanceCacheEntry{
		inst:    inst,
		staleFn: inst.SnapshotCacheQueryFn(),
	}
}

// BuildCacheQueryFn returns a cacheQueryFn map where each closure delegates to the
// stale snapshot. The returned closures read the current entry.staleFn at call time
// (not a captured copy), so they automatically use the latest snapshot after refresh.
func (s *StaleCacheIndex) BuildCacheQueryFn() map[string]func([]int) int {
	result := make(map[string]func([]int) int, len(s.entries))
	for id := range s.entries {
		idStr := string(id)
		result[idStr] = func(tokens []int) int {
			return s.Query(idStr, tokens)
		}
	}
	return result
}

// warnIfNotSnapshotCapable logs a warning if inst's KVCache does not implement
// cacheSnapshotCapable. Called once per instance at registration (not on every refresh)
// to avoid log spam.
func warnIfNotSnapshotCapable(id InstanceID, inst *InstanceSimulator) {
	if inst.sim == nil {
		return
	}
	if _, ok := inst.sim.KVCache.(cacheSnapshotCapable); !ok {
		logrus.Warnf("[stale-cache] instance %s: KVCache does not implement cacheSnapshotCapable — falling back to live query; stale-cache semantics not honored", id)
	}
}
```

**Step 3: Run all cluster tests**

```
go test ./sim/cluster/... -count=1 -v 2>&1 | grep -E "^(=== RUN|--- PASS|--- FAIL|FAIL|ok)"
```
Expected: All tests pass (same count as baseline).

**Step 4: Run lint**

```
golangci-lint run ./sim/cluster/...
```
Expected: No new issues.

**Step 5: Commit**

```bash
git add sim/cluster/stale_cache.go
git commit -m "refactor(cluster): consolidate StaleCacheIndex to single map (BC-4)

Replace dual instances+staleFns maps (different key types, must stay in sync)
with single entries map[InstanceID]instanceCacheEntry. Eliminates structural
fragility before autoscaler (PR11) adds high-frequency add/remove calls.

Also extracts warnIfNotSnapshotCapable helper to deduplicate the R1 guard
that previously appeared in both NewStaleCacheIndex and AddInstance.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Extract registerInstanceCacheQueryFn helper (BC-5)

**Contracts implemented:** BC-5, BC-6

**Files:**
- Modify: `sim/cluster/cluster.go`
- Modify: `sim/cluster/infra_lifecycle_event.go`

**Step 1: Run tests to confirm baseline (oracle path will be refactored)**

```
go test ./sim/cluster/... -count=1
```
Expected: All pass.

**Step 2: Add helper method to `cluster.go`**

After the `buildRouterState` function definition (around line 64 in `cluster_event.go` — but the helper belongs in `cluster.go` near the `cacheQueryFn` field). Add this method anywhere in `cluster.go` outside existing functions, after the `NewClusterSimulator` function:

```go
// registerInstanceCacheQueryFn adds a cacheQueryFn entry for a single instance,
// choosing between stale (snapshot) and oracle (live) modes (R23).
// Called from two sites: oracle-mode constructor loop (NewClusterSimulator) and
// NodeReadyEvent.Execute (deferred instances). NOT called from the stale-mode
// constructor — that path uses the bulk NewStaleCacheIndex + BuildCacheQueryFn API.
// Precondition: cs.cacheQueryFn must be non-nil (initialised before calling).
func (cs *ClusterSimulator) registerInstanceCacheQueryFn(id InstanceID, inst *InstanceSimulator) {
	if cs.staleCache != nil {
		// Stale mode: register with StaleCacheIndex; the closure reads the current staleFn at call
		// time, so it picks up refreshed snapshots automatically after RefreshIfNeeded.
		cs.staleCache.AddInstance(id, inst)
		idStr := string(id)
		cs.cacheQueryFn[idStr] = func(tokens []int) int {
			return cs.staleCache.Query(idStr, tokens)
		}
	} else {
		// Oracle mode: closure captures inst directly for live-state queries.
		idStr := string(id)
		cs.cacheQueryFn[idStr] = func(tokens []int) int {
			return inst.GetCachedBlockCount(tokens)
		}
	}
}
```

**Step 3: Update oracle constructor loop in `NewClusterSimulator` (cluster.go ~line 294)**

Replace:
```go
	} else {
		// Oracle mode (default): scorers query live KV cache state.
		cs.cacheQueryFn = make(map[string]func([]int) int, len(cs.instances))
		for _, inst := range cs.instances {
			id := string(inst.ID())
			inst := inst // capture for closure
			cs.cacheQueryFn[id] = func(tokens []int) int {
				return inst.GetCachedBlockCount(tokens)
			}
		}
	}
```

With:
```go
	} else {
		// Zero delay (CacheSignalDelay=0) — oracle mode: scorers query live KV cache state.
		cs.cacheQueryFn = make(map[string]func([]int) int, len(cs.instances))
		for _, inst := range cs.instances {
			cs.registerInstanceCacheQueryFn(inst.ID(), inst)
		}
	}
```

Note: The comment is also fixed here (item 5 in issue #947). The `inst := inst` capture is no longer needed because `registerInstanceCacheQueryFn` receives `inst` as a parameter (each call frame owns its own `inst` variable).

**Step 4: Update `NodeReadyEvent.Execute` (infra_lifecycle_event.go ~line 85)**

Replace:
```go
		// Register with cacheQueryFn for precise prefix scoring (deferred instances).
		if cs.cacheQueryFn != nil {
			if cs.staleCache != nil {
				// Stale mode: register with StaleCacheIndex and delegate to stale queries (issue #919).
				cs.staleCache.AddInstance(p.id, inst)
				idStr := string(p.id)
				cs.cacheQueryFn[idStr] = func(tokens []int) int {
					return cs.staleCache.Query(idStr, tokens)
				}
			} else {
				// Oracle mode: direct live query.
				inst := inst // capture for closure
				cs.cacheQueryFn[string(p.id)] = func(tokens []int) int {
					return inst.GetCachedBlockCount(tokens)
				}
			}
		}
```

With:
```go
		// Register with cacheQueryFn for precise prefix scoring (deferred instances).
		// registerInstanceCacheQueryFn handles both oracle and stale modes (R23).
		if cs.cacheQueryFn != nil {
			cs.registerInstanceCacheQueryFn(p.id, inst)
		}
```

**Step 5: Run all cluster and full tests**

```
go test ./sim/cluster/... -count=1 -v 2>&1 | grep -E "^(=== RUN|--- PASS|--- FAIL|FAIL|ok)"
go test ./... -count=1 2>&1 | tail -10
```
Expected: All pass.

**Step 6: Run lint**

```
golangci-lint run ./sim/cluster/...
```
Expected: No new issues.

**Step 7: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/infra_lifecycle_event.go
git commit -m "refactor(cluster): extract registerInstanceCacheQueryFn helper, fix oracle comment (BC-5)

Eliminate R23 oracle/stale dual-path duplication: both NewClusterSimulator
(oracle loop) and NodeReadyEvent.Execute now delegate to a single
registerInstanceCacheQueryFn method. The stale-mode bulk constructor
(NewStaleCacheIndex + BuildCacheQueryFn) is unchanged — it is a different
initialization path, not the same as single-instance registration.

Also fixes cluster.go:294 comment: 'Oracle mode (default)' →
'Zero delay (CacheSignalDelay=0) — oracle mode' (oracle is not the CLI default).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Add co-change guard comment on GetCachedBlocks (BC-9)

**Contracts implemented:** BC-9

**Files:**
- Modify: `sim/kv/cache.go`

**Step 1: Locate `GetCachedBlocks` in `sim/kv/cache.go` (currently around line 125)**

**Step 2: Add co-change comment**

Change the existing doc comment block for `GetCachedBlocks` from:
```go
// GetCachedBlocks returns the cached blocks for the given token sequence.
// It returns block IDs for the longest contiguous cached prefix.
// This is a pure query — it does not modify any state.
// CacheHits are counted by AllocateKVBlocks when cached blocks are committed.
//
// Uses hierarchical block hashing: each block's hash chains the previous
// block's hash, so each iteration hashes only blockSize tokens plus a
// fixed-length prev hash (O(K * blockSize) total, down from O(K^2 * blockSize)
// with flat-prefix hashing). Breaks on first miss.
func (kvc *KVCacheState) GetCachedBlocks(tokens []int) (blockIDs []int64) {
```

To (add the co-change note at the end of the existing comment):
```go
// GetCachedBlocks returns the cached blocks for the given token sequence.
// It returns block IDs for the longest contiguous cached prefix.
// This is a pure query — it does not modify any state.
// CacheHits are counted by AllocateKVBlocks when cached blocks are committed.
//
// Uses hierarchical block hashing: each block's hash chains the previous
// block's hash, so each iteration hashes only blockSize tokens plus a
// fixed-length prev hash (O(K * blockSize) total, down from O(K^2 * blockSize)
// with flat-prefix hashing). Breaks on first miss.
//
// CO-CHANGE: SnapshotCachedBlocksFn (same file) replicates this algorithm on a
// frozen map[string]int64 snapshot. If this loop, the hash chain logic, or the
// break condition changes, update SnapshotCachedBlocksFn to match.
func (kvc *KVCacheState) GetCachedBlocks(tokens []int) (blockIDs []int64) {
```

**Step 3: Verify build passes**

```
go build ./sim/kv/...
go test ./sim/kv/... -count=1
```
Expected: Build passes, all tests pass.

**Step 4: Run lint**

```
golangci-lint run ./sim/kv/...
```
Expected: No new issues.

**Step 5: Commit**

```bash
git add sim/kv/cache.go
git commit -m "docs(kv): add co-change guard linking GetCachedBlocks to SnapshotCachedBlocksFn (BC-9)

Both methods implement the same hierarchical prefix-counting algorithm. Without
a cross-reference, a change to GetCachedBlocks could silently leave
SnapshotCachedBlocksFn diverged, causing stale-mode routing to count prefix
blocks differently from live-mode. The CO-CHANGE comment makes this explicit.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit/Failure | `TestStaleCacheIndex_AddInstance_DuplicateID_Panics` |
| BC-2 | Task 1 | Unit | `TestStaleCacheIndex_RemoveInstance` (extended) |
| BC-3 | Task 2 | Unit | `TestStaleCacheIndex_RefreshIfNeeded_BoundaryAtIntervalMinusOne` |
| BC-4 | Task 3 | Regression | All existing `TestStaleCacheIndex_*` tests (refactor survival) |
| BC-5 | Task 4 | Integration | `TestCluster_CacheSignalDelay_StaleRouting`, `TestCluster_CacheSignalDelay_Zero_OracleBehavior` |
| BC-6 | Tasks 3–5 | Regression | All existing cluster + kv tests |
| BC-7 | Task 4 | N/A | Comment accuracy — no runtime test possible |
| BC-8 | Task 3 | Regression | `TestCluster_CacheSignalDelay_StaleRouting` (behavioral) |
| BC-9 | Task 5 | N/A | Comment — no runtime test possible |

**No golden dataset updates:** This PR makes no changes to output metrics or format.

**Invariant tests:** BC-4 and BC-6 are verified by the full existing test suite (all existing stale cache + cluster tests act as invariant guards for the refactoring).

---

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| `itoa` helper in stale_cache.go unnecessarily avoids `fmt` | Low | Low | Alternative: keep `fmt.Sprintf` — both compile, behavior identical | Task 3 |
| `registerInstanceCacheQueryFn` called in stale-mode constructor loop (would double-register, panic) | Low (plan explicitly says do NOT call it in stale-mode constructor) | High | Plan specifies that stale-mode constructor keeps existing `NewStaleCacheIndex` + `BuildCacheQueryFn` path | Task 4 |
| `RefreshIfNeeded` mutation of struct in map (value type copy) | Low | High | Plan uses `s.entries[id] = e` assignment after mutation — verified | Task 3 |
| `warnIfNotSnapshotCapable` extraction introduces `fmt` + `logrus` imports | Low | Low | Both `"fmt"` and `"github.com/sirupsen/logrus"` are in the import block | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `instanceCacheEntry` is the minimal struct; `itoa` is optional (can use fmt)
- [x] No feature creep beyond PR scope — pure refactoring + tests + comments
- [x] No unexercised flags or interfaces
- [x] No partial implementations — all 5 tasks produce compilable, testable code
- [x] No breaking changes — same public API for `StaleCacheIndex`
- [x] No hidden global state impact — `StaleCacheIndex` is owned by `ClusterSimulator`
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (`newTestSimConfig`, `NewInstanceSimulator`)
- [x] CLAUDE.md: no new files/packages; issue #947 will be noted as Completed in CLAUDE.md in the commit
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: changes do not modify canonical sources (rules.md, invariants.md, principles.md)
- [x] Deviation log reviewed — one clarification (item 6 comment vs helper)
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (tests → single-map refactor → helper extraction → comments)
- [x] All contracts mapped to specific tasks
- [x] No golden dataset update needed
- [x] Construction site audit: `StaleCacheIndex{}` is constructed only in `NewStaleCacheIndex` (1 site). `instanceCacheEntry{}` is constructed only inside `stale_cache.go`. ✅

**Antipattern rules:**
- [x] R1: No silent continue/return — `RemoveInstance` no-op is intentional (documented)
- [x] R2: No map iteration for ordered output
- [x] R3: `interval <= 0` validated in constructor
- [x] R4: `StaleCacheIndex` constructed in one place (`NewStaleCacheIndex`); `instanceCacheEntry` constructed inside `stale_cache.go` only
- [x] R6: No `logrus.Fatalf` in `sim/` packages
- [x] R7: Existing tests act as invariant guards for the refactor
- [x] R8: No exported mutable maps — `entries` is unexported
- [x] R13: No new interfaces
- [x] R14: `registerInstanceCacheQueryFn` is single-concern (registration only)
- [x] R23: The helper **eliminates** the R23 violation

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/stale_cache.go` (complete rewrite)

**Purpose:** Replace dual `instances`/`staleFns` maps with single `entries map[InstanceID]instanceCacheEntry`. Extract `warnIfNotSnapshotCapable` to deduplicate the R1 guard.

**Key implementation notes:**
- `instanceCacheEntry` is a value type (struct), stored by value in the map. `RefreshIfNeeded` must read, mutate the `staleFn` field, then re-assign: `e.staleFn = ...; s.entries[id] = e`.
- `Query` converts the `string` argument to `InstanceID` for the map lookup: `s.entries[InstanceID(instanceID)]`.
- Import block: `"fmt"` (for panic message) + `"github.com/sirupsen/logrus"` (for Warnf).
- Public API (method signatures) is unchanged.

**Complete implementation:** See Task 3 Step 2 above.

---

### File: `sim/cluster/cluster.go` (two changes)

**Change 1 — Add `registerInstanceCacheQueryFn` method** (anywhere after `NewClusterSimulator`):

```go
func (cs *ClusterSimulator) registerInstanceCacheQueryFn(id InstanceID, inst *InstanceSimulator) {
	if cs.staleCache != nil {
		cs.staleCache.AddInstance(id, inst)
		idStr := string(id)
		cs.cacheQueryFn[idStr] = func(tokens []int) int {
			return cs.staleCache.Query(idStr, tokens)
		}
	} else {
		idStr := string(id)
		cs.cacheQueryFn[idStr] = func(tokens []int) int {
			return inst.GetCachedBlockCount(tokens)
		}
	}
}
```

**Change 2 — Oracle constructor loop** (around line 293-302):

Old:
```go
	} else {
		// Oracle mode (default): scorers query live KV cache state.
		cs.cacheQueryFn = make(map[string]func([]int) int, len(cs.instances))
		for _, inst := range cs.instances {
			id := string(inst.ID())
			inst := inst // capture for closure
			cs.cacheQueryFn[id] = func(tokens []int) int {
				return inst.GetCachedBlockCount(tokens)
			}
		}
	}
```

New:
```go
	} else {
		// Zero delay (CacheSignalDelay=0) — oracle mode: scorers query live KV cache state.
		cs.cacheQueryFn = make(map[string]func([]int) int, len(cs.instances))
		for _, inst := range cs.instances {
			cs.registerInstanceCacheQueryFn(inst.ID(), inst)
		}
	}
```

---

### File: `sim/cluster/infra_lifecycle_event.go` (one change)

**Change — NodeReadyEvent cacheQueryFn registration** (around lines 84-100):

Old (15 lines):
```go
		if cs.cacheQueryFn != nil {
			if cs.staleCache != nil {
				cs.staleCache.AddInstance(p.id, inst)
				idStr := string(p.id)
				cs.cacheQueryFn[idStr] = func(tokens []int) int {
					return cs.staleCache.Query(idStr, tokens)
				}
			} else {
				inst := inst // capture for closure
				cs.cacheQueryFn[string(p.id)] = func(tokens []int) int {
					return inst.GetCachedBlockCount(tokens)
				}
			}
		}
```

New (4 lines):
```go
		// Register with cacheQueryFn for precise prefix scoring (deferred instances).
		// registerInstanceCacheQueryFn handles both oracle and stale modes (R23).
		if cs.cacheQueryFn != nil {
			cs.registerInstanceCacheQueryFn(p.id, inst)
		}
```

---

### File: `sim/kv/cache.go` (comment addition)

**Change:** Append CO-CHANGE comment to `GetCachedBlocks` doc block. See Task 5 Step 2 above.

---

### File: `sim/cluster/stale_cache_test.go` (additions only)

**New tests:**
1. `TestStaleCacheIndex_AddInstance_DuplicateID_Panics` — after line ~68 (`TestStaleCacheIndex_AddInstance`)
2. Extension of `TestStaleCacheIndex_RemoveInstance` — add `assert.Equal(t, 0, idx.Query(...))` after `idx.RemoveInstance`
3. `TestStaleCacheIndex_RefreshIfNeeded_BoundaryAtIntervalMinusOne` — after `TestStaleCacheIndex_StaleUntilRefresh`

See Task 1 and Task 2 for complete test code.
