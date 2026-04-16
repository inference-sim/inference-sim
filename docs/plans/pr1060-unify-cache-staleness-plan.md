# Unify Cache Signal Staleness into ObservabilityConfig

**Goal:** Eliminate the duplicated `StaleCacheIndex` staleness system by folding cache block refresh timing into the existing `ObservabilityConfig` framework, and change the default cache staleness from 2s to 50ms.

**The problem today:** The codebase has two parallel implementations of periodic stale snapshots for the router: `ObservabilityConfig` + `CachedSnapshotProvider` (for QueueDepth, BatchSize, KVUtilization) and `StaleCacheIndex` (for cache block hash maps). Both model the same concept — signal propagation delay from instance to router — but use completely different plumbing. The 2s default for cache staleness creates a ~13-second warm-up artifact where the PPC scorer is blind to cached prefixes.

**What this PR adds:**
1. A `CacheBlocks` field in `ObservabilityConfig` so all signal staleness is configured in one place.
2. Cache block refresh managed by `CachedSnapshotProvider` instead of a separate `StaleCacheIndex` struct.
3. Default cache staleness changed from 2s to 50ms (eliminating the warm-up artifact).
4. The `--cache-signal-delay` CLI flag is retained (backward compat) with default changed from 2s to 50ms; internally, its value feeds into `ObservabilityConfig.CacheBlocks`.

**Why this matters:** Reduces code duplication (~150 lines), fixes the R16 violation (config not grouped by module), and eliminates the 13s warm-up artifact.

**Architecture:** `CachedSnapshotProvider` gains cache block snapshot management (previously in `StaleCacheIndex`). The `ClusterSimulator` no longer needs a separate `staleCache` field. `ObservabilityConfig` gains a `CacheBlocks FieldConfig` entry. The `--snapshot-refresh-interval` flag now controls all periodic signals uniformly; a new `--cache-signal-delay` override allows independent cache staleness tuning (backward-compat for scripts).

**Source:** [Issue #1060](https://github.com/inference-sim/inference-sim/issues/1060)

**Closes:** Fixes #1060

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** Signal freshness subsystem (ObservabilityConfig, CachedSnapshotProvider, StaleCacheIndex).
2. **Adjacent blocks:** Cluster initialization (`NewClusterSimulator`), routing (`buildRouterState`), instance lifecycle (`addLiveInstance`, drain policies), precise-prefix-cache/no-hit-lru scorers.
3. **Invariants touched:** INV-7 (Signal Freshness).
4. **Construction Site Audit:**
   - `ObservabilityConfig{}`: `DefaultObservabilityConfig()` (snapshot.go:33), `newObservabilityConfig()` (snapshot.go:50)
   - `CachedSnapshotProvider{}`: `NewCachedSnapshotProvider()` (snapshot.go:85) — single canonical constructor
   - `DeploymentConfig{}`: Many sites in tests and CLI — the `CacheSignalDelay` field is removed, all construction sites must be updated
   - `StaleCacheIndex{}`: `NewStaleCacheIndex()` (stale_cache.go:39) — will be deleted entirely

---

## Part 1: Design Validation

### A) Executive Summary

This PR unifies two parallel staleness systems into one. Currently, `ObservabilityConfig` + `CachedSnapshotProvider` handles periodic refresh of scalar signals (QueueDepth, BatchSize, KVUtilization), while `StaleCacheIndex` handles periodic refresh of cache block hash map snapshots. Both use the same pattern (time-gated refresh), but with completely separate code.

After this PR, `CachedSnapshotProvider` manages cache block snapshots alongside the scalar signals. `ObservabilityConfig` gains a `CacheBlocks FieldConfig` entry. The `StaleCacheIndex` type is deleted. The default cache staleness changes from 2s to 50ms.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Unified cache refresh timing
- GIVEN a cluster with `SnapshotRefreshInterval > 0` or `CacheSignalDelay > 0`
- WHEN `CachedSnapshotProvider.Snapshot()` is called
- THEN cache block query functions use the staleness interval from `ObservabilityConfig.CacheBlocks`, following the same `Immediate/Periodic` mode system as QueueDepth/BatchSize/KVUtilization

BC-2: Default 50ms cache staleness
- GIVEN a cluster created with default configuration (no explicit `--cache-signal-delay`)
- WHEN the `ObservabilityConfig` is constructed
- THEN `CacheBlocks` uses `Periodic` mode with interval `50_000` microseconds (50ms)

BC-3: Oracle mode preserved
- GIVEN `--cache-signal-delay 0`
- WHEN a routing decision occurs
- THEN cache block queries read live instance state (no snapshot delay), identical behavior to the previous `CacheSignalDelay=0` oracle mode

BC-4: Instance lifecycle integration
- GIVEN a deferred instance that becomes ready (NodeReadyEvent) or an instance that is terminated/drained
- WHEN `CachedSnapshotProvider.AddInstance()` is called (add) or the instance is removed
- THEN the cache block query function for that instance is correctly added/removed, maintaining the same lifecycle semantics as before

BC-5: Stale routing behavior preserved
- GIVEN two identical clusters — one oracle (delay=0), one stale (delay=very large)
- WHEN both process the same shared-prefix workload
- THEN oracle concentrates requests on the cache-warm instance while stale spreads them (same behavioral test as existing `TestCluster_CacheSignalDelay_StaleRouting`)

BC-6: CLI backward compatibility
- GIVEN `--cache-signal-delay N` is passed on the CLI
- WHEN the cluster is configured
- THEN cache block staleness uses interval `N` microseconds, overriding the 50ms default

**Negative contracts:**

BC-7: No StaleCacheIndex type
- GIVEN the codebase after this PR
- WHEN searching for `StaleCacheIndex`
- THEN no Go type, constructor, or method with that name exists (fully removed)

### C) Component Interaction

```
CLI (cmd/root.go, cmd/replay.go)
  │
  │ --cache-signal-delay N (default 50000)
  │ --snapshot-refresh-interval M (default 0)
  │
  ▼
DeploymentConfig
  │  .CacheSignalDelay int64  (preserved for CLI→config plumbing)
  │  .SnapshotRefreshInterval int64
  │
  ▼
newObservabilityConfig(snapshotRefreshInterval, cacheSignalDelay)
  │  → ObservabilityConfig{
  │      QueueDepth:    {Periodic, M} or {Immediate},
  │      BatchSize:     {Periodic, M} or {Immediate},
  │      KVUtilization: {Periodic, M} or {Immediate},
  │      CacheBlocks:   {Periodic, N} or {Immediate},  ← NEW
  │    }
  │
  ▼
CachedSnapshotProvider
  │  .cacheEntries map[InstanceID]cacheEntry         ← NEW (replaces StaleCacheIndex.entries)
  │  .cacheLastRefresh int64                          ← NEW (replaces StaleCacheIndex.lastRefresh)
  │
  │  Snapshot(id, clock) → RoutingSnapshot (unchanged)
  │  RefreshCacheIfNeeded(clock)                      ← NEW (replaces StaleCacheIndex.RefreshIfNeeded)
  │  CacheQuery(instanceID, tokens) → int             ← NEW (replaces StaleCacheIndex.Query)
  │  BuildCacheQueryFn() → map[string]func([]int)int  ← NEW (replaces StaleCacheIndex.BuildCacheQueryFn)
  │  AddCacheInstance(id, inst)                        ← NEW (replaces StaleCacheIndex.AddInstance)
  │  RemoveCacheInstance(id)                           ← NEW (replaces StaleCacheIndex.RemoveInstance)
  │
  ▼
ClusterSimulator
  │  .staleCache *StaleCacheIndex  ← REMOVED
  │  .snapshotProvider still owns cache refresh
  │
  ▼
buildRouterState() calls provider.RefreshCacheIfNeeded(clock) instead of staleCache.RefreshIfNeeded(clock)
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Remove `--cache-signal-delay` flag" | Keeps `--cache-signal-delay` as CLI flag, changes default to 50000 | CLARIFICATION: Removing the flag would break existing scripts. Keep it with new default. |
| "Remove `CacheSignalDelay` from `DeploymentConfig`" | Keeps `CacheSignalDelay` in `DeploymentConfig` | CLARIFICATION: The field plumbs CLI→config; removing it would break the replay path. Instead, `newObservabilityConfig` now consumes both `SnapshotRefreshInterval` and `CacheSignalDelay`. |
| Issue mentions 50ms default | Uses 50_000 µs (50ms) | DIRECT: issue's proposed value |

### E) Review Guide

**Tricky part:** The `CachedSnapshotProvider` must now manage two kinds of state: scalar snapshot values (existing) and closure-based cache query functions (new). The closure lifecycle (add/remove on instance lifecycle events) is the most delicate area — verify all 3 removal sites and 2 addition sites are updated.

**Scrutinize:** The `RefreshCacheIfNeeded` timing — it must use the same `>=` boundary as the old `StaleCacheIndex.RefreshIfNeeded` and the existing `shouldRefresh` for scalars.

**Safe to skim:** The `DeploymentConfig` field changes and CLI flag default change are mechanical.

**Known debt:** The `--cache-signal-delay` flag and `CacheSignalDelay` config field are preserved for backward compatibility. A future PR could fold them into `--snapshot-refresh-interval` if full unification is desired.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to modify:
- `sim/cluster/snapshot.go` — Add `CacheBlocks` to `ObservabilityConfig`, add cache management methods to `CachedSnapshotProvider`, update `newObservabilityConfig` signature
- `sim/cluster/snapshot_test.go` — Add tests for new cache management on `CachedSnapshotProvider`
- `sim/cluster/stale_cache.go` — DELETE entirely
- `sim/cluster/stale_cache_test.go` — Rewrite: unit tests target `CachedSnapshotProvider` cache methods; integration tests preserved
- `sim/cluster/deployment.go` — Change `DefaultCacheSignalDelay` to `50_000`
- `sim/cluster/cluster.go` — Remove `staleCache` field, use `snapshotProvider` for cache operations
- `sim/cluster/cluster_event.go` — Update `buildRouterState` to use `snapshotProvider` instead of `staleCache`
- `sim/cluster/infra_lifecycle_event.go` — Update drain policies to use `snapshotProvider` instead of `staleCache`
- `sim/cluster/instance.go` — No changes (SnapshotCacheQueryFn, GetCachedBlockCount unchanged)
- `cmd/root.go` — Update flag help text for `--cache-signal-delay` (new default)
- `cmd/replay.go` — No structural changes needed (CacheSignalDelay still flows through)
- `docs/contributing/standards/invariants.md` — Update INV-7 table
- `docs/guide/routing.md` — Update staleness documentation
- `CLAUDE.md` — Update Recent Changes section

Key decisions:
- Cache closures stored as a new `cacheEntry` struct inside `CachedSnapshotProvider` (parallel to the existing per-instance scalar cache)
- `RefreshCacheIfNeeded(clock)` is a separate method (not folded into `Snapshot()`) because cache refresh is all-instances-at-once (same as old `StaleCacheIndex.RefreshIfNeeded`), while `Snapshot()` is per-instance
- The `SnapshotProvider` interface is NOT changed — cache query functions are accessed via concrete `*CachedSnapshotProvider` type, same pattern as existing `AddInstance`

### G) Task Breakdown

#### Task 1: Add CacheBlocks to ObservabilityConfig and update newObservabilityConfig (BC-1, BC-2, BC-3)

**Files:** modify `sim/cluster/snapshot.go`, test `sim/cluster/snapshot_test.go`

**Test (write first, expect FAIL):**

In `sim/cluster/snapshot_test.go`, add:

```go
func TestObservabilityConfig_CacheBlocks_DefaultImmediate(t *testing.T) {
	// BC-3: When cache delay is 0, CacheBlocks uses Immediate mode.
	config := newObservabilityConfig(0, 0)
	assert.Equal(t, Immediate, config.CacheBlocks.Mode)
}

func TestObservabilityConfig_CacheBlocks_Periodic(t *testing.T) {
	// BC-1: When cache delay > 0, CacheBlocks uses Periodic mode with given interval.
	config := newObservabilityConfig(0, 50_000)
	assert.Equal(t, Periodic, config.CacheBlocks.Mode)
	assert.Equal(t, int64(50_000), config.CacheBlocks.Interval)
}

func TestObservabilityConfig_CacheBlocks_IndependentOfSnapshot(t *testing.T) {
	// BC-1: CacheBlocks interval is independent of snapshot refresh interval.
	config := newObservabilityConfig(10_000, 50_000)
	assert.Equal(t, Periodic, config.QueueDepth.Mode)
	assert.Equal(t, int64(10_000), config.QueueDepth.Interval)
	assert.Equal(t, Periodic, config.CacheBlocks.Mode)
	assert.Equal(t, int64(50_000), config.CacheBlocks.Interval)
}
```

**Verify:** `go test ./sim/cluster/... -run TestObservabilityConfig_CacheBlocks -v` → FAIL (CacheBlocks field doesn't exist)

**Impl:**

In `sim/cluster/snapshot.go`:

1. Add `CacheBlocks FieldConfig` to `ObservabilityConfig` struct (after `KVUtilization`).
2. Update `DefaultObservabilityConfig()` to include `CacheBlocks: FieldConfig{Mode: Immediate}`.
3. Change `newObservabilityConfig` signature to `newObservabilityConfig(refreshInterval int64, cacheDelay int64) ObservabilityConfig`.
4. In `newObservabilityConfig`: set `CacheBlocks` to `Periodic` with interval `cacheDelay` when `cacheDelay > 0`, otherwise `Immediate`.

**Verify:** `go test ./sim/cluster/... -run TestObservabilityConfig_CacheBlocks -v` → PASS
**Lint:** `golangci-lint run ./sim/cluster/...`

Also update all callers of `newObservabilityConfig` (currently 2 sites in cluster.go and their tests):
- `cluster.go:332`: `newObservabilityConfig(config.SnapshotRefreshInterval)` → `newObservabilityConfig(config.SnapshotRefreshInterval, config.CacheSignalDelay)`
- Fix existing tests: `TestNewObservabilityConfig_ZeroAndNegativeInterval_AllImmediate` and `TestNewObservabilityConfig_NonZeroInterval_AllFieldsPeriodic` need updated calls.

**Commit:** `refactor(cluster): add CacheBlocks to ObservabilityConfig (BC-1, BC-2, BC-3)`

---

#### Task 2: Add cache management methods to CachedSnapshotProvider (BC-1, BC-4)

**Files:** modify `sim/cluster/snapshot.go`, test `sim/cluster/snapshot_test.go`

**Test (write first, expect FAIL):**

```go
func TestCachedSnapshotProvider_CacheQuery_StaleUntilRefresh(t *testing.T) {
	// BC-1: Cache queries return stale data until refresh interval elapses.
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)
	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}

	obsConfig := newObservabilityConfig(0, 1000) // cache delay 1000µs
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Initial snapshot — empty cache
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens))

	// Populate cache via request
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()
	require.Greater(t, inst.GetCachedBlockCount(tokens), 0)

	// Before refresh interval: still stale
	provider.RefreshCacheIfNeeded(500) // 500 < 1000
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens))

	// After refresh interval: sees new data
	provider.RefreshCacheIfNeeded(1000) // 1000 >= 1000
	assert.Greater(t, provider.CacheQuery("inst-0", tokens), 0)
}

func TestCachedSnapshotProvider_CacheQuery_OracleMode(t *testing.T) {
	// BC-3: When CacheBlocks is Immediate, queries return live data.
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)
	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}

	obsConfig := newObservabilityConfig(0, 0) // oracle mode
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()

	// Oracle mode: sees live data immediately without refresh
	assert.Greater(t, provider.CacheQuery("inst-0", tokens), 0)
}

func TestCachedSnapshotProvider_AddRemoveCacheInstance(t *testing.T) {
	// BC-4: Dynamic instance add/remove for cache queries.
	obsConfig := newObservabilityConfig(0, 1000)
	provider := NewCachedSnapshotProvider(nil, obsConfig)

	cfg := newTestSimConfig()
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-new", cfg)

	provider.AddInstance("inst-new", inst)
	tokens := []int{1, 2, 3, 4}
	assert.Equal(t, 0, provider.CacheQuery("inst-new", tokens))

	provider.RemoveCacheInstance("inst-new")
	assert.Equal(t, 0, provider.CacheQuery("inst-new", tokens)) // returns 0 for unknown
}
```

**Verify:** `go test ./sim/cluster/... -run TestCachedSnapshotProvider_Cache -v` → FAIL

**Impl:**

In `sim/cluster/snapshot.go`, add to `CachedSnapshotProvider`:

```go
// cacheEntry holds a live instance reference and its current stale snapshot closure.
type cacheEntry struct {
	inst    *InstanceSimulator
	staleFn func([]int) int
}
```

Add fields to `CachedSnapshotProvider`:
```go
cacheEntries    map[InstanceID]cacheEntry
cacheLastRefresh int64
```

Update `NewCachedSnapshotProvider` to initialize `cacheEntries` from instances (take initial snapshots, same logic as old `NewStaleCacheIndex`).

Add methods:
- `RefreshCacheIfNeeded(clock int64)` — same logic as old `StaleCacheIndex.RefreshIfNeeded`, using `config.CacheBlocks` for timing. No-op when `config.CacheBlocks.Mode == Immediate`.
- `CacheQuery(instanceID string, tokens []int) int` — returns stale or live result based on mode. For `Immediate` mode, delegates directly to `inst.GetCachedBlockCount`.
- `BuildCacheQueryFn() map[string]func([]int) int` — same as old `StaleCacheIndex.BuildCacheQueryFn`.
- `AddCacheInstance(id InstanceID, inst *InstanceSimulator)` — registers a new cache entry (panics on duplicate).
- `RemoveCacheInstance(id InstanceID)` — removes a cache entry.

**Verify:** `go test ./sim/cluster/... -run TestCachedSnapshotProvider_Cache -v` → PASS
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `refactor(cluster): add cache management to CachedSnapshotProvider (BC-1, BC-4)`

---

#### Task 3: Wire cluster to use CachedSnapshotProvider for cache operations (BC-1, BC-4, BC-5, BC-6)

**Files:** modify `sim/cluster/cluster.go`, `sim/cluster/cluster_event.go`, `sim/cluster/infra_lifecycle_event.go`

**Test (write first, expect FAIL):**

The existing integration tests (`TestCluster_CacheSignalDelay_StaleRouting`, `TestCluster_CacheSignalDelay_Zero_OracleBehavior`) should pass with the new wiring. Run them first to establish the baseline — they should currently FAIL because `cluster.go` still references the old `staleCache` field which we haven't removed yet, and `newObservabilityConfig` signature changed in Task 1.

**Verify:** `go test ./sim/cluster/... -run TestCluster_CacheSignalDelay -v` → should fail due to compilation errors from Task 1's signature change

**Impl:**

In `sim/cluster/cluster.go`:

1. **Remove the `staleCache *StaleCacheIndex` field** from `ClusterSimulator` struct.

2. **Update `NewClusterSimulator`** (lines 332-345):
   Replace the dual oracle/stale initialization with:
   ```go
   cs.snapshotProvider = NewCachedSnapshotProvider(instanceMap,
       newObservabilityConfig(config.SnapshotRefreshInterval, config.CacheSignalDelay))

   // Build cacheQueryFn from the unified provider.
   if config.CacheSignalDelay > 0 {
       cs.cacheQueryFn = cs.snapshotProvider.(*CachedSnapshotProvider).BuildCacheQueryFn()
   } else {
       // Oracle mode: closures query live instance state.
       cs.cacheQueryFn = make(map[string]func([]int) int, len(cs.instances))
       for _, inst := range cs.instances {
           cs.registerInstanceCacheQueryFn(inst.ID(), inst)
       }
   }
   ```
   Note: We access `snapshotProvider` as `*CachedSnapshotProvider` via type assertion for cache-specific methods. This is safe because `NewClusterSimulator` always creates a `CachedSnapshotProvider`.

3. **Update `registerInstanceCacheQueryFn`** (lines 462-486):
   Replace `cs.staleCache != nil` check with checking the concrete provider:
   ```go
   func (cs *ClusterSimulator) registerInstanceCacheQueryFn(id InstanceID, inst *InstanceSimulator) {
       csp, ok := cs.snapshotProvider.(*CachedSnapshotProvider)
       if ok && csp.HasCacheEntries() {
           csp.AddCacheInstance(id, inst)
           idStr := string(id)
           cs.cacheQueryFn[idStr] = func(tokens []int) int {
               return csp.CacheQuery(idStr, tokens)
           }
       } else {
           idStr := string(id)
           cs.cacheQueryFn[idStr] = func(tokens []int) int {
               return inst.GetCachedBlockCount(tokens)
           }
       }
   }
   ```
   Note: `HasCacheEntries()` is a new method that returns `true` when `CacheBlocks.Mode == Periodic` (stale mode is active). This replaces the `staleCache != nil` check.

4. **Update all 3 cache removal sites** — replace `if c.staleCache != nil { c.staleCache.RemoveInstance(...) }` with:
   ```go
   if csp, ok := cs.snapshotProvider.(*CachedSnapshotProvider); ok {
       csp.RemoveCacheInstance(inst.ID())
   }
   ```
   The 3 sites are:
   - **Site A:** `cluster.go:641` — drain completion check in main event loop (T042 marker)
   - **Site B:** `infra_lifecycle_event.go:215` — `drainImmediate.Drain()` immediate termination path
   - **Site C:** `infra_lifecycle_event.go:237` — `drainWait.Drain()` idle-instance fast path

In `sim/cluster/cluster_event.go`:

5. **Update `buildRouterState`** (line 67-69):
   Replace:
   ```go
   if cs.staleCache != nil {
       cs.staleCache.RefreshIfNeeded(cs.clock)
   }
   ```
   With:
   ```go
   if csp, ok := cs.snapshotProvider.(*CachedSnapshotProvider); ok {
       csp.RefreshCacheIfNeeded(cs.clock)
   }
   ```

**Verify:** `go test ./sim/cluster/... -run TestCluster_CacheSignalDelay -v` → PASS
**Also:** `go test ./sim/cluster/... -v` → all tests PASS
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `refactor(cluster): wire CachedSnapshotProvider for cache operations (BC-1, BC-4, BC-5, BC-6)`

---

#### Task 4: Delete StaleCacheIndex and migrate tests (BC-7)

**Files:** DELETE `sim/cluster/stale_cache.go`, rewrite `sim/cluster/stale_cache_test.go`

**Impl:**

1. **Delete `sim/cluster/stale_cache.go`** entirely.

2. **Rewrite `sim/cluster/stale_cache_test.go`** — rename to keep existing test names where behavioral contracts match, but target `CachedSnapshotProvider`:
   - `TestStaleCacheIndex_StaleUntilRefresh` → already covered by `TestCachedSnapshotProvider_CacheQuery_StaleUntilRefresh` (Task 2)
   - `TestStaleCacheIndex_AddInstance` → already covered by `TestCachedSnapshotProvider_AddRemoveCacheInstance` (Task 2)
   - `TestStaleCacheIndex_AddInstance_HonorsStaleBoundary` → port to `CachedSnapshotProvider`
   - `TestStaleCacheIndex_RefreshIfNeeded_BoundaryAtIntervalMinusOne` → port to `CachedSnapshotProvider`
   - `TestStaleCacheIndex_AddInstance_DuplicateID_Panics` → port (AddCacheInstance should also panic)
   - `TestStaleCacheIndex_BuildCacheQueryFn_DelegatesToStale` → port
   - `TestCluster_CacheSignalDelay_StaleRouting` → already in the file, should pass as-is (BC-5)
   - `TestCluster_CacheSignalDelay_Zero_OracleBehavior` → should pass as-is (BC-3)
   - `TestStaleCacheIndex_RemoveInstance` → port
   - `TestStaleCacheIndex_RemoveInstance_Idempotent` → port
   - `TestNewStaleCacheIndex_ZeroInterval_Panics` → REMOVE (no longer applicable — `Periodic` with interval=0 just becomes `Immediate`)
   - `TestNewStaleCacheIndex_NegativeInterval_Panics` → REMOVE (same reason)

**Verify:** `go test ./sim/cluster/... -v` → all tests PASS
**Also:** `go build ./...` → no compilation errors (confirms no dangling references to StaleCacheIndex)
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `refactor(cluster): delete StaleCacheIndex, migrate tests to CachedSnapshotProvider (BC-7)`

---

#### Task 5: Change default to 50ms and update CLI (BC-2, BC-6)

**Files:** modify `sim/cluster/deployment.go`, `cmd/root.go`

**Test (write first, expect FAIL):**

```go
func TestDefaultCacheSignalDelay_Is50ms(t *testing.T) {
	// BC-2: Default cache signal delay is 50ms (50,000 µs).
	assert.Equal(t, int64(50_000), DefaultCacheSignalDelay)
}
```

**Verify:** `go test ./sim/cluster/... -run TestDefaultCacheSignalDelay -v` → FAIL (currently 2_000_000)

**Impl:**

1. In `sim/cluster/deployment.go`:
   - Change `const DefaultCacheSignalDelay int64 = 2_000_000` to `const DefaultCacheSignalDelay int64 = 50_000`
   - Update the comment to reflect 50ms and corrected rationale.

2. In `cmd/root.go`:
   - Update `--cache-signal-delay` flag help text: change "Default 2s matches production llm-d speculative TTL" to "Default 50ms. Set to 0 for oracle mode (live cache state)."

**Verify:** `go test ./sim/cluster/... -run TestDefaultCacheSignalDelay -v` → PASS
**Also:** `go test ./... -count=1` → all tests pass
**Lint:** `golangci-lint run ./...`
**Commit:** `fix(cluster): change DefaultCacheSignalDelay from 2s to 50ms (BC-2, BC-6)`

---

#### Task 6: Update documentation (INV-7, CLAUDE.md, routing guide)

**Files:** modify `docs/contributing/standards/invariants.md`, `docs/guide/routing.md`, `CLAUDE.md`

**Impl:**

1. In `docs/contributing/standards/invariants.md` (INV-7 table, line 86):

   **Replace the cacheQueryFn row (line 86) with:**
   ```
   | cacheQueryFn (precise-prefix-cache, no-hit-lru) ¹ | Instance (via CachedSnapshotProvider) | Ground truth (synchronous) | Periodic (CacheBlocks interval, default 50ms) | `CachedSnapshotProvider.RefreshCacheIfNeeded()` in `buildRouterState()` |
   ```

   **Replace footnote ¹ (line 88) with:**
   ```
   ¹ `cacheQueryFn` freshness is governed by `--cache-signal-delay` (default 50ms), which maps to `ObservabilityConfig.CacheBlocks`. The "interval=0" / "interval>0" columns for this row refer to `--cache-signal-delay`. Cache block staleness is now managed by `CachedSnapshotProvider` alongside other signals (#1060).
   ```

   **Replace paragraph at line 90 — change:**
   `When --cache-signal-delay > 0 (default: 2s), prefix cache query closures use a separate periodic snapshot of each instance's HashToBlock map, modeling asynchronous KV event propagation from production llm-d. The 2s default matches llm-d's defaultSpeculativeTTL — the blind spot between routing decision and KV event arrival via ZMQ. Set --cache-signal-delay 0 for oracle mode (live cache state).`
   **To:**
   `When --cache-signal-delay > 0 (default: 50ms), prefix cache query closures use periodic snapshots of each instance's HashToBlock map, managed by CachedSnapshotProvider alongside other signal snapshots. The 50ms default models aggregate signal staleness from production llm-d. Set --cache-signal-delay 0 for oracle mode (live cache state).`

2. In `docs/guide/routing.md`:
   - Update the staleness tip to mention that `--cache-signal-delay` now defaults to 50ms.

3. In `CLAUDE.md`:
   - Update "Cache signal propagation delay (#919)" entry to note it was unified into `ObservabilityConfig` by #1060.
   - Add a new entry for this PR in Recent Changes.

**Verify:** Visual review of documentation changes.
**Lint:** No Go lint needed for docs.
**Commit:** `docs: update INV-7, routing guide, and CLAUDE.md for unified cache staleness (#1060)`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestObservabilityConfig_CacheBlocks_Periodic` |
| BC-1 | Task 2 | Unit | `TestCachedSnapshotProvider_CacheQuery_StaleUntilRefresh` |
| BC-2 | Task 1 | Unit | `TestObservabilityConfig_CacheBlocks_DefaultImmediate` → verifies default wiring |
| BC-2 | Task 5 | Unit | `TestDefaultCacheSignalDelay_Is50ms` |
| BC-3 | Task 1 | Unit | `TestObservabilityConfig_CacheBlocks_DefaultImmediate` |
| BC-3 | Task 2 | Unit | `TestCachedSnapshotProvider_CacheQuery_OracleMode` |
| BC-3 | Task 3 | Integration | `TestCluster_CacheSignalDelay_Zero_OracleBehavior` (existing) |
| BC-4 | Task 2 | Unit | `TestCachedSnapshotProvider_AddRemoveCacheInstance` |
| BC-5 | Task 3 | Integration | `TestCluster_CacheSignalDelay_StaleRouting` (existing) |
| BC-6 | Task 5 | Unit | CLI flag wiring (existing `TestRunCmd_SnapshotRefreshInterval_FlagRegistered` pattern) |
| BC-7 | Task 4 | Build | `go build ./...` succeeds with no `StaleCacheIndex` references |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Cache closure lifecycle mismatch during instance add/remove | Medium | High | Port all 3 removal sites and 2 addition sites; integration tests cover lifecycle | Task 3 |
| `newObservabilityConfig` signature change breaks callers | Low | Medium | Only 2 call sites; both updated in Task 1 | Task 1 |
| Default change from 2s→50ms changes golden test outputs | Low | Medium | No golden tests depend on cache staleness timing directly | Task 5 |
| Type assertion `snapshotProvider.(*CachedSnapshotProvider)` could fail | Low | High | Only one concrete implementation is ever created; panic on failure is correct | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — reusing existing `ObservabilityConfig`/`CachedSnapshotProvider`
- [x] No feature creep — strictly unification + default change
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without contract updates — `--cache-signal-delay` preserved
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md updated
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: INV-7 (canonical source) updated; CLAUDE.md working copy updated
- [x] Deviation log reviewed — 2 CLARIFICATIONs documented
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5→6)
- [x] All contracts mapped to specific tasks
- [x] Construction site audit completed

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — `CacheQuery` returns 0 with warning for unknown instance (same as old `StaleCacheIndex.Query`)
- [x] R2: No map-key float accumulation
- [x] R3: `--cache-signal-delay` validated >= 0 (existing validation preserved)
- [x] R4: All `ObservabilityConfig` construction sites updated (2 sites)
- [x] R5: No resource allocation loops
- [x] R6: No `logrus.Fatalf` in `sim/` — existing `logrus.Warnf` preserved for unknown instance query
- [x] R7: Integration tests alongside unit tests
- [x] R8: No exported mutable maps
- [x] R9: No new YAML fields
- [x] R10: No new YAML parsing
- [x] R11: No division by runtime denominators
- [x] R12: No golden dataset changes
- [x] R13: No new interfaces — reusing existing `SnapshotProvider`
- [x] R14: No multi-module methods
- [x] R15: Old plan references updated (stale_cache.go deleted)
- [x] R16: Config now grouped properly — CacheBlocks in ObservabilityConfig
- [x] R17: Signal freshness documented (INV-7 updated)
- [x] R18: CLI flag default changed explicitly
- [x] R19: No retry loops
- [x] R20: No new detectors/analyzers
- [x] R21: No range over shrinking slices
- [x] R22: No pre-check estimates
- [x] R23: Oracle/stale paths both updated consistently

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/snapshot.go`

**Purpose:** Add `CacheBlocks` field to `ObservabilityConfig` and cache management methods to `CachedSnapshotProvider`.

**Key changes:**

1. `ObservabilityConfig` gains `CacheBlocks FieldConfig`.

2. `newObservabilityConfig(refreshInterval, cacheDelay int64)` — new signature accepts both intervals.

3. `CachedSnapshotProvider` gains:
   - `cacheEntries map[InstanceID]cacheEntry` — per-instance cache snapshots
   - `cacheLastRefresh int64` — last refresh timestamp
   - `RefreshCacheIfNeeded(clock int64)` — refreshes all cache snapshots when interval elapsed
   - `CacheQuery(instanceID string, tokens []int) int` — queries stale or live cache
   - `BuildCacheQueryFn() map[string]func([]int) int` — builds closure map
   - `AddCacheInstance(id InstanceID, inst *InstanceSimulator)` — registers new instance for cache tracking
   - `RemoveCacheInstance(id InstanceID)` — unregisters instance
   - `HasCacheEntries() bool` — returns true when stale cache mode is active (CacheBlocks.Mode == Periodic)

4. `cacheEntry` struct (private) — identical to old `instanceCacheEntry`: holds `inst *InstanceSimulator` and `staleFn func([]int) int`.

**State mutation:** `cacheEntries` and `cacheLastRefresh` are mutated by `RefreshCacheIfNeeded`, `AddCacheInstance`, `RemoveCacheInstance`. Owned exclusively by `CachedSnapshotProvider`.

**Error handling:** `AddCacheInstance` panics on duplicate ID (same as old `StaleCacheIndex.AddInstance`). `CacheQuery` for unknown instance logs warning and returns 0.

### File: `sim/cluster/stale_cache.go`

**Purpose:** DELETED. All functionality moved to `CachedSnapshotProvider` in `snapshot.go`.

### File: `sim/cluster/deployment.go`

**Purpose:** Change `DefaultCacheSignalDelay` from `2_000_000` to `50_000`. Update comment.

### File: `sim/cluster/cluster.go`

**Purpose:** Remove `staleCache` field. Use `snapshotProvider` for all cache operations.

**Key changes:**
- Remove `staleCache *StaleCacheIndex` field from `ClusterSimulator`
- `NewClusterSimulator`: pass `config.CacheSignalDelay` to `newObservabilityConfig`; build `cacheQueryFn` via `snapshotProvider`
- `registerInstanceCacheQueryFn`: check `snapshotProvider.(*CachedSnapshotProvider).HasCacheEntries()` instead of `staleCache != nil`
- Drain/termination cleanup: use `snapshotProvider.(*CachedSnapshotProvider).RemoveCacheInstance()` instead of `staleCache.RemoveInstance()`

### File: `sim/cluster/cluster_event.go`

**Purpose:** `buildRouterState` calls `snapshotProvider.(*CachedSnapshotProvider).RefreshCacheIfNeeded(clock)` instead of `staleCache.RefreshIfNeeded(clock)`.

### File: `sim/cluster/infra_lifecycle_event.go`

**Purpose:** Drain policies call `snapshotProvider.(*CachedSnapshotProvider).RemoveCacheInstance()` instead of `staleCache.RemoveInstance()`.

### File: `cmd/root.go`

**Purpose:** Update `--cache-signal-delay` flag help text to reflect 50ms default.

### File: `docs/contributing/standards/invariants.md`

**Purpose:** Update INV-7 table to reflect unified `CachedSnapshotProvider` model.

### File: `docs/guide/routing.md`

**Purpose:** Update staleness tip to mention 50ms default.

### File: `CLAUDE.md`

**Purpose:** Add Recent Changes entry for #1060.
