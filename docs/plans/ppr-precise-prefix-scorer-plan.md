# Precise Prefix Cache & No-Hit-LRU Scorers — Implementation Plan

**Goal:** Add two new routing scorers (`precise-prefix-cache` and `no-hit-lru`) that query actual instance KV cache state, matching llm-d production implementation.

**The problem today:** BLIS's prefix routing uses a router-side approximate approach (`PrefixCacheIndex` with LRU). The production llm-d system uses precise KV cache state. When translating BLIS routing to llm-d, prefix affinity is dropped entirely (hardcoded 0.0), causing simulation routing to diverge from production.

**What this PR adds:**
1. `precise-prefix-cache` scorer — queries actual instance KV cache state, uses min-max normalization across candidates (matching llm-d)
2. `no-hit-lru` scorer — distributes cold requests (cache misses) to least-recently-used endpoints (matching llm-d)
3. `GetCachedBlockCount` public method on `InstanceSimulator` — accessor following existing pattern
4. `cacheQueryFn` threading through scorer factory chain — enables scorers to query per-instance KV state

**Why this matters:** Enables sim-to-production parity for prefix-aware routing. The weighted scoring algorithm can be validated against llm-d since both use the same precise prefix mechanism.

**Architecture:** New scorers in `sim/` package, new public accessor in `sim/cluster/instance.go`. Factory chain (`newScorerWithObserver` + `NewRoutingPolicy`) gains a `cacheQueryFn` parameter. Cluster layer constructs the function map from `InstanceSimulator` references.

**Source:** [Issue #883](https://github.com/inference-sim/inference-sim/issues/883)
**Closes:** `Fixes #883`

---

## Phase 0: Component Context

1. **Building block:** Two new scoring policies behind the existing `scorerFunc` interface
2. **Adjacent blocks:** `WeightedScoring` (composition), `KVStore.GetCachedBlocks` (data source), `InstanceSimulator` (bridge), `CachedSnapshotProvider` (snapshot delivery)
3. **Invariants touched:** None — routing is pure scoring, read-only KV access
4. **Construction site audit:**
   - `newScorerWithObserver` — sole factory for scorer creation (`routing_scorers.go:101`)
   - `NewRoutingPolicy` — sole construction site for `WeightedScoring` (`routing.go:268`), called at `cluster.go:159`, `cluster.go:182`, `cluster.go:185`, and 48 test call sites (30 in `routing_test.go`, 10 in `routing_prefix_scorer_test.go`, 2 in `routing_scorers_test.go`, 6 in `examples_test.go`) = **51 total**

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds two new scorers to the composable weighted routing pipeline. The `precise-prefix-cache` scorer queries actual per-instance KV cache state (via a new `cacheQueryFn` map) and normalizes scores using min-max normalization across candidates. The `no-hit-lru` scorer tracks routing history and distributes cold requests (no cache hits) to least-recently-used instances. Both scorers follow the existing `scorerFunc`/`observerFunc` pattern. The factory chain gains a `cacheQueryFn` parameter that stateless scorers ignore. The cluster layer constructs the function map from `InstanceSimulator.GetCachedBlockCount()`.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Precise prefix cache scoring with min-max normalization
- GIVEN a weighted routing policy with `precise-prefix-cache` scorer and instances with varying cached prefix blocks
- WHEN a request is routed
- THEN the instance with the most matching cached blocks scores 1.0, the least scores 0.0, and intermediate instances score proportionally within [0, 1] using min-max normalization

BC-2: All-equal precise prefix scores
- GIVEN instances that all have the same number of cached blocks for a request (including zero)
- WHEN a request is routed with `precise-prefix-cache` scorer
- THEN all instances score 1.0 (all-equal case)

BC-3: No-hit-LRU cold request distribution
- GIVEN a weighted routing policy with `no-hit-lru` scorer and a request that has zero cached blocks on all instances
- WHEN the request is routed
- THEN never-used instances score highest, and used instances score by LRU position (oldest-used scores higher)

BC-4: No-hit-LRU warm request neutrality
- GIVEN a request that has cached blocks on at least one instance
- WHEN routed with `no-hit-lru` scorer
- THEN all instances score 0.5 (neutral — defers to other scorers)

BC-5: No-hit-LRU state update on cold routing only
- GIVEN the `no-hit-lru` scorer's observer
- WHEN called after routing a warm request
- THEN the LRU order is not updated

BC-6: Cache query function threading
- GIVEN `NewRoutingPolicy` called with a non-nil `cacheQueryFn`
- WHEN `precise-prefix-cache` or `no-hit-lru` scorers are configured
- THEN the scorers use the provided functions to query per-instance KV cache state

BC-7: Backward compatibility
- GIVEN `NewRoutingPolicy` called with nil `cacheQueryFn` (all existing call sites)
- WHEN any existing scorer is configured
- THEN behavior is identical to before this PR

**Negative contracts:**

BC-8: No observer for precise-prefix-cache
- GIVEN the `precise-prefix-cache` scorer
- WHEN created by the factory
- THEN the observer is nil (stateless — reads ground truth, no approximation)

### C) Component Interaction

```
ClusterSimulator
  │
  ├── instances []*InstanceSimulator
  │       │
  │       └── GetCachedBlockCount(tokens []int) int  ← NEW
  │               └── delegates to i.sim.KVCache.GetCachedBlocks()
  │
  └── NewRoutingPolicy(..., cacheQueryFn)  ← MODIFIED signature
          │
          └── WeightedScoring
                  ├── precise-prefix-cache scorer  ← NEW
                  │       └── calls cacheQueryFn[id](req.InputTokens)
                  │       └── min-max normalization (highest→1.0, lowest→0.0)
                  │
                  └── no-hit-lru scorer  ← NEW
                          ├── calls cacheQueryFn[id](req.InputTokens)
                          ├── warm requests → 0.5 for all
                          └── cold requests → LRU positional scoring
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue proposes `GetCachedBlockCount` on InstanceSimulator | Same | — |
| Issue says "Requests shorter than one block: raw score 0 for all → all-equal → all score 1.0" | Handled by all-equal case in min-max | SIMPLIFICATION — no special case needed; zero raw scores are all-equal by definition |
| Issue specifies a named scorer preset (`precise-prefix-cache:2,kv-utilization:1,queue-depth:1`) | Deferred — no named preset mechanism exists; users pass `--scorer-config` directly | DEFERRAL — adding a preset registry is a separate concern; CLI example in issue works as-is |
| Issue marks `CLI (cmd/)` as affected | No `cmd/` changes needed | CLARIFICATION — scorer name validation is in `sim/routing_scorers.go` via `validScorerNames`; `ParseScorerConfigs` validates against it, so adding names to the map is sufficient |

### E) Review Guide

**Tricky part:** The `cacheQueryFn` threading — adding a parameter to `newScorerWithObserver` and `NewRoutingPolicy` touches 48 test call sites + 3 production sites = 51 total. Each existing call must pass `nil`. The `no-hit-lru` LRU state management (observer shares warm/cold flag with scorer via closure).

**Safe to skim:** `GetCachedBlockCount` accessor (trivial delegation). `validScorerNames` registration (mechanical).

**Scrutinize:** Min-max normalization edge cases (single instance, all-zero). LRU cold-vs-warm detection. Factory parameter threading.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/routing_scorers.go` — add `cacheQueryFn` param to `newScorerWithObserver`, register 2 new scorers in `validScorerNames`
- `sim/routing.go` — add `cacheQueryFn` param to `NewRoutingPolicy`, thread to factory
- `sim/cluster/instance.go` — add `GetCachedBlockCount` method
- `sim/cluster/cluster.go` — construct `cacheQueryFn` map, pass to `NewRoutingPolicy`

**Files to create:**
- `sim/routing_precise_prefix_scorer.go` — precise-prefix-cache scorer
- `sim/routing_precise_prefix_scorer_test.go` — tests
- `sim/routing_nohit_lru_scorer.go` — no-hit-lru scorer + LRU state
- `sim/routing_nohit_lru_scorer_test.go` — tests

**Test files to update (add nil cacheQueryFn param):**
- `sim/routing_test.go` (30 call sites)
- `sim/routing_prefix_scorer_test.go` (10 call sites)
- `sim/routing_scorers_test.go` (2 call sites)
- `sim/examples_test.go` (6 call sites)

### G) Task Breakdown

#### Task 1: Add `GetCachedBlockCount` to InstanceSimulator (BC-6)

**Files:** modify `sim/cluster/instance.go`, test via existing `sim/cluster/` test patterns

**Test:** Verify `GetCachedBlockCount` returns `len(KVCache.GetCachedBlocks(tokens))`. Since this is a trivial delegation following the exact pattern of `KVUtilization()`, `FreeKVBlocks()`, `CacheHitRate()` — test via integration in Task 4.

**Impl:**
```go
// GetCachedBlockCount returns the number of consecutive cached prefix blocks
// matching the given token sequence. Used by precise prefix cache scoring.
func (i *InstanceSimulator) GetCachedBlockCount(tokens []int) int {
    return len(i.sim.KVCache.GetCachedBlocks(tokens))
}
```

**Verify:** `go build ./sim/cluster/...`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): add GetCachedBlockCount accessor to InstanceSimulator (BC-6)`

---

#### Task 2: Thread `cacheQueryFn` through factory chain (BC-6, BC-7)

**Files:** modify `sim/routing_scorers.go`, `sim/routing.go`

**Type definition (in routing_scorers.go):**
```go
// CacheQueryFn maps instance IDs to functions that return the count of
// consecutive cached prefix blocks for given tokens. Used by precise
// prefix cache scoring. Nil for sim-level tests without cluster instances.
type CacheQueryFn map[string]func([]int) int
```

**Changes to `newScorerWithObserver`:**
```go
func newScorerWithObserver(name string, blockSize int, cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc) {
```
All existing cases pass through unchanged — they ignore `cacheQueryFn`.

**Important:** Task 2 does NOT add new `case` entries to the switch or new entries to `validScorerNames`. Those are added in Tasks 3 and 4 respectively, alongside the implementation functions they call. This ensures Task 2 compiles and passes tests independently.

**Changes to `NewRoutingPolicy`:**
```go
func NewRoutingPolicy(name string, scorerConfigs []ScorerConfig, blockSize int64, rng *rand.Rand, cacheQueryFn CacheQueryFn) RoutingPolicy {
```
Thread `cacheQueryFn` to `newScorerWithObserver`.

**Update all existing call sites** to pass `nil` as the last argument. This includes:
- `sim/cluster/cluster.go:159, 182, 185` (3 production sites)
- 48 test call sites across 4 test files (30 + 10 + 2 + 6)

**Verification step:** Before mass-updating, run `grep -rn 'NewRoutingPolicy' ./sim/ --include='*.go'` to confirm all 51 call sites.

**Test:** All existing tests pass with `nil` cacheQueryFn (BC-7 backward compat).

**Verify:** `go test ./sim/... -count=1`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `refactor(sim): thread cacheQueryFn through scorer factory chain (BC-6, BC-7)`

---

#### Task 3: Implement `precise-prefix-cache` scorer (BC-1, BC-2, BC-8)

**Files:** create `sim/routing_precise_prefix_scorer.go`, create `sim/routing_precise_prefix_scorer_test.go`

**Scorer implementation:**
```go
// newPrecisePrefixCacheScorer creates a scorer that queries actual per-instance
// KV cache state for prefix match counts, then applies min-max normalization.
//
// Signal freshness (R17, INV-7):
//
//	Reads: KVCache.GetCachedBlocks via cacheQueryFn — ground truth (synchronous,
//	no staleness). Each routing decision queries the current KV cache state
//	at the moment of routing.
func newPrecisePrefixCacheScorer(cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc) {
    scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
        scores := make(map[string]float64, len(snapshots))
        if req == nil || cacheQueryFn == nil {
            for _, snap := range snapshots {
                scores[snap.ID] = 1.0
            }
            return scores
        }
        // Pass 1: compute raw scores and find min/max
        raw := make(map[string]int, len(snapshots))
        minRaw, maxRaw := math.MaxInt, 0
        for _, snap := range snapshots {
            count := 0
            if fn, ok := cacheQueryFn[snap.ID]; ok {
                count = fn(req.InputTokens)
            }
            raw[snap.ID] = count
            if count < minRaw { minRaw = count }
            if count > maxRaw { maxRaw = count }
        }
        // Pass 2: min-max normalize (higher cached → higher score)
        for _, snap := range snapshots {
            if maxRaw == minRaw {
                scores[snap.ID] = 1.0
            } else {
                scores[snap.ID] = float64(raw[snap.ID]-minRaw) / float64(maxRaw-minRaw)
            }
        }
        return scores
    }
    return scorer, nil // no observer (BC-8: stateless ground truth)
}
```

**Register in `newScorerWithObserver` (added in this task, not Task 2):**
```go
case "precise-prefix-cache":
    return newPrecisePrefixCacheScorer(cacheQueryFn)
```

**Register in `validScorerNames` (added in this task, not Task 2):**
```go
"precise-prefix-cache": true,
```

**Tests (behavioral, table-driven):**
- BC-1: Three instances with 5, 3, 0 cached blocks → scores 1.0, 0.6, 0.0
- BC-2: Three instances with 0, 0, 0 cached blocks → all 1.0
- BC-2: Three instances with 4, 4, 4 cached blocks → all 1.0
- BC-8: Observer is nil
- Single instance → score 1.0
- Nil cacheQueryFn → all 1.0

**Verify:** `go test ./sim/... -run TestPrecise -count=1`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `feat(sim): add precise-prefix-cache scorer with min-max normalization (BC-1, BC-2, BC-8)`

---

#### Task 4: Implement `no-hit-lru` scorer (BC-3, BC-4, BC-5)

**Files:** create `sim/routing_nohit_lru_scorer.go`, create `sim/routing_nohit_lru_scorer_test.go`

**Scorer implementation:**
```go
// newNoHitLRUScorer creates a scorer that distributes cold requests (no cache
// hits on any instance) to least-recently-used endpoints. Warm requests (at
// least one instance has cached blocks) score 0.5 (neutral, defers to other
// scorers).
//
// Signal freshness (R17, INV-7):
//
//	Reads: KVCache.GetCachedBlocks via cacheQueryFn — ground truth (synchronous,
//	no staleness). Used only for warm/cold detection, not scoring magnitude.
//	LRU state is deterministic (updated by observer on cold routing only).
func newNoHitLRUScorer(cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc) {
    // LRU tracking: ordered list of instance IDs, most-recently-used first.
    // Only updated on cold request routing.
    var lruOrder []string // most-recent first
    lruSet := make(map[string]bool)

    // Shared warm/cold flag between scorer and observer (same pattern as
    // cachedHashes/cachedReqID in prefix-affinity scorer). The scorer sets
    // lastWarm; the observer reads it. Safe because the DES is single-threaded
    // and scorer is always called before observer for the same request.
    lastWarm := false
    lastReqID := ""

    scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
        scores := make(map[string]float64, len(snapshots))

        // Nil cacheQueryFn → neutral (cannot determine hit status)
        if req == nil || cacheQueryFn == nil {
            for _, snap := range snapshots {
                scores[snap.ID] = 0.5
            }
            lastWarm = true // prevent observer from updating LRU
            lastReqID = ""
            return scores
        }

        // Check if any instance has cached blocks (warm detection)
        lastWarm = false
        lastReqID = req.ID
        for _, snap := range snapshots {
            if fn, ok := cacheQueryFn[snap.ID]; ok {
                if fn(req.InputTokens) > 0 {
                    lastWarm = true
                    break
                }
            }
        }

        if lastWarm {
            // BC-4: warm request → neutral 0.5 for all
            for _, snap := range snapshots {
                scores[snap.ID] = 0.5
            }
            return scores
        }
        // BC-3: cold request → LRU positional scoring
        total := len(snapshots)
        if total == 1 {
            scores[snapshots[0].ID] = 1.0
            return scores
        }
        // Build rank: never-used first (rank 0), then oldest-used to newest-used
        rank := 0
        // Never-used instances (not in lruSet) get lowest ranks (= highest scores)
        var neverUsed []string
        for _, snap := range snapshots {
            if !lruSet[snap.ID] {
                neverUsed = append(neverUsed, snap.ID)
            }
        }
        for _, id := range neverUsed {
            scores[id] = 1.0 - float64(rank)/float64(total-1)
            rank++
        }
        // Used instances: oldest first (end of lruOrder) to newest (start)
        for i := len(lruOrder) - 1; i >= 0; i-- {
            id := lruOrder[i]
            // Only score if this instance is in the current snapshot set
            found := false
            for _, snap := range snapshots {
                if snap.ID == id { found = true; break }
            }
            if found {
                scores[id] = 1.0 - float64(rank)/float64(total-1)
                rank++
            }
        }
        return scores
    }

    observer := func(req *Request, targetInstance string) {
        if req == nil {
            return
        }
        // BC-5: use scorer's warm/cold determination (not re-derived).
        // This avoids disagreement between scorer (checks all instances)
        // and observer (would only check target instance).
        if lastWarm || req.ID != lastReqID {
            return
        }
        // Move targetInstance to front of LRU (most-recently-used)
        if lruSet[targetInstance] {
            // Remove from current position
            for i, id := range lruOrder {
                if id == targetInstance {
                    lruOrder = append(lruOrder[:i], lruOrder[i+1:]...)
                    break
                }
            }
        }
        lruOrder = append([]string{targetInstance}, lruOrder...)
        lruSet[targetInstance] = true
    }

    return scorer, observer
}
```

**Register in `newScorerWithObserver` (added in this task, not Task 2):**
```go
case "no-hit-lru":
    return newNoHitLRUScorer(cacheQueryFn)
```

**Register in `validScorerNames` (added in this task, not Task 2):**
```go
"no-hit-lru": true,
```

**Tests (behavioral, table-driven):**
- BC-3: Cold request with 3 instances, 2 never-used → never-used score highest
- BC-3: Cold request after prior cold routing → LRU ordering
- BC-4: Warm request → all score 0.5
- BC-5: Route warm request, verify LRU order unchanged (uses shared warm flag)
- Single instance cold → score 1.0
- Nil cacheQueryFn → all score 0.5 (neutral, not cold-mode)
- INV-6 determinism: two identical routing sequences produce identical LRU state and scores

**Verify:** `go test ./sim/... -run TestNoHitLRU -count=1`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `feat(sim): add no-hit-lru scorer for cold request distribution (BC-3, BC-4, BC-5)`

---

#### Task 5: Wire `cacheQueryFn` in cluster layer (BC-6)

**Files:** modify `sim/cluster/cluster.go`

**Changes:** After instances are constructed (after the unified construction+placement loop and snapshot provider init), build the `cacheQueryFn` map and pass it to `NewRoutingPolicy`.

**Struct field addition:** Add `cacheQueryFn sim.CacheQueryFn` field to `ClusterSimulator` (unexported, alongside `inFlightRequests`).

**Restructuring:** Move `routingPolicy`, `prefillRoutingPolicy`, and `decodeRoutingPolicy` creation from the struct literal / PD block to AFTER the instance construction loop (same location as `snapshotProvider` init at line 266). Steps:
1. Remove `routingPolicy: sim.NewRoutingPolicy(...)` from the `cs := &ClusterSimulator{...}` struct literal (line 159) — set to nil initially
2. Move the PD routing policy creation (lines 181-185) to after the construction loop
3. After the construction loop, build `cacheQueryFn` and create all routing policies:

```go
// Build cacheQueryFn from constructed instances.
cs.cacheQueryFn = make(sim.CacheQueryFn, len(cs.instances))
for _, inst := range cs.instances {
    id := string(inst.ID())
    inst := inst // capture for closure
    cs.cacheQueryFn[id] = func(tokens []int) int {
        return inst.GetCachedBlockCount(tokens)
    }
}
cs.routingPolicy = sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem(sim.SubsystemRouter), cs.cacheQueryFn)
// PD per-pool routing policies (moved from PD block above)
if len(config.PrefillScorerConfigs) > 0 {
    cs.prefillRoutingPolicy = sim.NewRoutingPolicy("weighted", config.PrefillScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("prefill-router"), cs.cacheQueryFn)
}
if len(config.DecodeScorerConfigs) > 0 {
    cs.decodeRoutingPolicy = sim.NewRoutingPolicy("weighted", config.DecodeScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("decode-router"), cs.cacheQueryFn)
}
```

**Deferred instances (NodePools path):** In `NodeReadyEvent.Execute` (infra_lifecycle_event.go:73, after `cs.instances = append(...)`), add the new instance to `cs.cacheQueryFn`:
```go
// Register with cacheQueryFn for precise prefix scoring (deferred instances).
if cs.cacheQueryFn != nil {
    inst := inst // capture
    cs.cacheQueryFn[string(p.id)] = func(tokens []int) int {
        return inst.GetCachedBlockCount(tokens)
    }
}
```
Since Go maps are reference types, the scorer closures (which captured the map at `NewRoutingPolicy` time) will see the new entries automatically.

**Verify:** `go test ./sim/cluster/... -count=1`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): wire cacheQueryFn to routing policy from instance KV state (BC-6)`

---

#### Task 6: Update CLAUDE.md

**Files:** modify `CLAUDE.md`

Update `validScorerNames` reference, add to Recent Changes, update scorer documentation.

**Verify:** visual inspection
**Commit:** `docs: update CLAUDE.md with precise-prefix-cache and no-hit-lru scorers`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 3 | Unit | TestPrecisePrefixCache_MinMaxNormalization |
| BC-2 | Task 3 | Unit | TestPrecisePrefixCache_AllEqual |
| BC-3 | Task 4 | Unit | TestNoHitLRU_ColdRequestDistribution |
| BC-4 | Task 4 | Unit | TestNoHitLRU_WarmRequestNeutral |
| BC-5 | Task 4 | Unit | TestNoHitLRU_ObserverColdOnly |
| INV-6 | Task 4 | Unit | TestNoHitLRU_Determinism |
| BC-6 | Task 5 | Integration | TestCacheQueryFnWiring — build cacheQueryFn from mock instances, pass to NewRoutingPolicy with precise-prefix-cache scorer, verify scorer returns expected scores |
| BC-7 | Task 2 | Regression | All existing tests pass with nil cacheQueryFn |
| BC-8 | Task 3 | Unit | TestPrecisePrefixCache_ObserverIsNil |

**Key invariants:** INV-6 (determinism) — scorers use deterministic maps, no float accumulation order issues. No new events, no clock changes, no KV mutations.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| 48 test call site updates introduce typos | Medium | Low | Mechanical find-replace, `go build` catches immediately | Task 2 |
| Deferred instances (NodePools) not added to cacheQueryFn | Low | Medium | Explicit `cs.cacheQueryFn[id]` update in NodeReadyEvent handler | Task 5 |
| LRU state not deterministic across runs | Low | High | LRU order modified only by observer (deterministic routing), never by map iteration; explicit determinism test added | Task 4 |
| LRU state grows unboundedly with instance churn | Low | Low | Acceptable for current scale (static or low-churn instance counts); O(n) observer scan bounded by active instances. Future: prune on NodeDrainedEvent if churn grows. | Task 4 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — scorers follow exact existing pattern
- [x] No feature creep — only what issue #883 specifies
- [x] No unexercised interfaces — both scorers registered and testable via `--scorer-config`
- [x] No breaking changes — nil cacheQueryFn preserves all existing behavior
- [x] R1: No silent continue/return
- [x] R2: No map iteration for ordered output (LRU uses slice)
- [x] R3: No new numeric CLI params (scorer names registered in existing validation)
- [x] R4: Construction sites audited — `NewRoutingPolicy` sole constructor
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: Behavioral tests, not golden
- [x] R8: validScorerNames stays unexported
- [x] R11: Min-max division guarded by maxRaw==minRaw check
- [x] R13: No new interfaces (uses existing scorerFunc)
- [x] R17: Signal freshness documented — precise-prefix-cache reads ground truth (no staleness)
- [x] CLAUDE.md updated in Task 6

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/instance.go`
- **Purpose:** Add `GetCachedBlockCount` public accessor
- **1 new method:** `GetCachedBlockCount(tokens []int) int` — delegates to `i.sim.KVCache.GetCachedBlocks(tokens)`

### File: `sim/routing_scorers.go`
- **Purpose:** Add `CacheQueryFn` type, update `newScorerWithObserver` signature, register new scorers
- **Type:** `CacheQueryFn map[string]func([]int) int`
- **Modified:** `newScorerWithObserver` gains `cacheQueryFn CacheQueryFn` parameter
- **Modified:** `validScorerNames` gains 2 entries

### File: `sim/routing.go`
- **Purpose:** Thread `cacheQueryFn` through `NewRoutingPolicy`
- **Modified:** `NewRoutingPolicy` signature gains `cacheQueryFn CacheQueryFn` parameter

### File: `sim/routing_precise_prefix_scorer.go`
- **Purpose:** Precise prefix cache scorer implementation
- **1 factory:** `newPrecisePrefixCacheScorer(cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc)`
- **Algorithm:** Two-pass min-max normalization on `cacheQueryFn[id](req.InputTokens)` counts

### File: `sim/routing_nohit_lru_scorer.go`
- **Purpose:** No-hit-LRU scorer implementation
- **1 factory:** `newNoHitLRUScorer(cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc)`
- **State:** `lruOrder []string` + `lruSet map[string]bool` + `lastWarm bool` + `lastReqID string` (closure-owned)
- **Algorithm:** Warm (or nil cacheQueryFn) → 0.5 all. Cold → positional scoring by LRU rank. Observer reads shared `lastWarm` flag (not re-derived).

### File: `sim/cluster/cluster.go`
- **Purpose:** Construct `cacheQueryFn` map from instances, pass to routing policy
- **New field:** `cacheQueryFn sim.CacheQueryFn` on `ClusterSimulator`
- **Modified:** Remove `routingPolicy` from struct literal. Move all routing policy creation (`routingPolicy`, `prefillRoutingPolicy`, `decodeRoutingPolicy`) to after instance construction loop. Build `cacheQueryFn` map from instances.

### File: `sim/cluster/infra_lifecycle_event.go`
- **Purpose:** Register deferred instances in `cacheQueryFn`
- **Modified:** `NodeReadyEvent.Execute` — add `cs.cacheQueryFn[id]` entry for each newly constructed instance.
