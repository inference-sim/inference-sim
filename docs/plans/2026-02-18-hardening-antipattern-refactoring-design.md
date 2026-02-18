# Hardening PR: Antipattern Elimination, Invariant Tests, and Modularity Improvements

**Date:** 2026-02-18
**Status:** Approved design (v3 — expanded scope + audit findings)
**Scope:** Dedicated hardening PR before PR11 (AutoScaler)
**Issues addressed:** #175, #183, #189, #190, #191, #192, #195, #196, #197, #198, #199, #200, #201, #202, #203
**Architectural issues addressed:** snapshot duplication, observation coupling, destructive-read semantics, library code calling os.Exit, CLI/bundle name drift, unvalidated KVStore factory, request state as raw string

## Motivation

A systematic audit of the codebase uncovered 20+ issues that cluster into six antipattern categories, plus a parallel architectural analysis identified structural friction that compounds when adding new components.

**Bug-level motivation:** The codebase tests outputs (golden values) rather than laws (conservation, causality). The golden dataset test encoded a wrong value (499 completions instead of 500) because one request was silently dropped (#183). Conservation invariant tests would have caught this on day one.

**Extensibility motivation:** Tracing extension scenarios through the current architecture reveals high "touch-point multipliers" — adding a single new observable metric requires changes in 6 files; adding a config field requires 5-6 files. These friction points will compound across PR11 (autoscaling), PR14 (P/D disaggregation), PR15 (framework adapters), and future features like LMCache, heterogeneous accelerators, or SGLang engine support.

This PR fixes all known correctness bugs, adds structural defenses to prevent recurrence, AND reduces friction for future extensions — establishing a clean foundation before PR11-PR16.

## Extension Scenario Analysis

The design is motivated by tracing these scenarios through the current architecture:

### Autoscaling (PR11)
- **Blocked by:** Concrete `[]*InstanceSimulator` slice (no lifecycle abstraction), inline `pendingRequests` bookkeeping, constructor panics on static instance count
- **Friction:** 5-6 file touch per new config field (cooldown, min/max instances)
- **Works well:** ClusterEvent polymorphic dispatch (no switch/case to modify)

### LMCache / Distributed KV Cache
- **Blocked by:** `PendingTransferLatency()` destructive-read semantics, `SetClock` only reachable via type assertion
- **Friction:** `KVStore` interface leaks block-level semantics, `hashTokens` crosses routing/KV boundary, no eviction policy abstraction
- **Works well:** Tier composition pattern (delegation) from TieredKVCache

### New Storage Subsystem (NVMe tier)
- **Blocked by:** `NewKVStore` factory has zero validation (unlike policy factories)
- **Friction:** 5-6 file config touch points, SimConfig mixes concerns
- **Works well:** KVStore interface tier composition

### Heterogeneous Accelerators / AIUs
- **Blocked by:** `DeploymentConfig.ToSimConfig()` creates identical config for ALL instances
- **Friction:** SimConfig mixes hardware identity with simulation params, InstanceSimulator reaches through 7 exported Simulator fields
- **Works well:** Per-instance SimConfig is structurally possible (DeploymentConfig already has ToSimConfig)

### New Engine (SGLang alongside vLLM)
- **Blocked by:** Latency model hardcoded into `Simulator.Step()` (calls `getStepTimeBB()` or `getStepTimeRoofline()` directly — no interface)
- **Friction:** Step() is a 134-line monolith mixing scheduling, latency, token generation, completion, and metrics. Batching semantics (continuous vs chunked prefill) are implicit in code flow, not parameterized.
- **Works well:** Roofline mode already provides an alternative latency model, showing the pattern

## Antipattern Summary

| Pattern | Count | Root Cause |
|---|---|---|
| Shotgun surgery (multiple construction sites) | 3 sites | No canonical constructors |
| Golden tests encoding bugs | systemic | Testing values, not laws |
| Silent `continue` dropping data | 4 paths | Error paths that skip cleanup |
| Non-deterministic map iteration | 5 sites | Unsorted map keys in float accumulation |
| Missing input validation | 2 | No guards on Rate=0, NaN/Inf weights |
| No transactional rollback | 1 | Mid-loop exit leaks allocated blocks |
| High touch-point multiplier (observables) | 6 files/metric | InstanceSnapshot/RoutingSnapshot duplication |
| High touch-point multiplier (config) | 5-6 files/field | CLI → DeploymentConfig → SimConfig manual chain |
| Destructive read on KVStore interface | 1 | PendingTransferLatency() clears on read |
| Type assertion through interface | 1 | SetClock not on KVStore interface |
| Library code calls os.Exit | 3 files | logrus.Fatalf inside sim/ and sim/cluster/ |
| CLI/bundle name drift | 1 | Error messages hardcode policy names |
| Request state not enforced | 1 | State is raw string, written 3 times, never read |

## Design

### Phase 1: Structural Helpers

These foundational changes make bug fixes simple and prevent recurrence.

#### 1a. `EffectiveLoad()` method on RoutingSnapshot

**File:** `sim/routing.go`

Add a method to `RoutingSnapshot` that centralizes the load formula:

```go
func (s RoutingSnapshot) EffectiveLoad() int64 {
    return s.QueueDepth + s.BatchSize + s.PendingRequests
}
```

Replace all 3 inline load calculations:
- `LeastLoaded` (routing.go:70-74): `snap.QueueDepth + snap.BatchSize` -> `snap.EffectiveLoad()`
- `WeightedScoring` (routing.go:140): `snap.QueueDepth + snap.BatchSize + snap.PendingRequests` -> `snap.EffectiveLoad()`
- `AlwaysBusiest` (routing.go:216-220): `snap.QueueDepth + snap.BatchSize` -> `snap.EffectiveLoad()`

This fixes #175 structurally. Future routing policies automatically get the correct formula.

**Design decision:** Method on struct (not standalone function) so autocomplete makes the right way obvious.

#### 1b. `NewRequestMetrics()` factory function

**File:** `sim/metrics_utils.go`

Add a canonical constructor that replaces all 3 struct-literal sites:

```go
func NewRequestMetrics(req *Request, arrivedAt float64) RequestMetrics {
    return RequestMetrics{
        ID:               req.ID,
        ArrivedAt:        arrivedAt,
        NumPrefillTokens: len(req.InputTokens),
        NumDecodeTokens:  len(req.OutputTokens),
        SLOClass:         req.SLOClass,
        TenantID:         req.TenantID,
    }
}
```

Update:
- `InjectArrival` (simulator.go:254): use `NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)`
- `InjectArrivalAt` (simulator.go:269): same
- `generateWorkloadFromCSV` (workload_config.go:88): refactor to call `InjectArrival` directly (fixes #189)

Any new field added to `RequestMetrics` only needs updating in one place.

#### 1c. Sorted map iteration helper

**File:** `sim/cluster/metrics.go`

```go
func sortedKeys(m map[string]float64) []string {
    keys := make([]string, 0, len(m))
    for k := range m {
        keys = append(keys, k)
    }
    sort.Strings(keys)
    return keys
}
```

Apply to `ComputeFitness`, `JainFairnessIndex`, and `mapValues`. Fixes #195.

#### 1d. Complete `aggregateMetrics` field coverage

**File:** `sim/cluster/cluster.go`

Add the 3 missing fields to `aggregateMetrics()`:
- Sum `PreemptionCount`
- Average `CacheHitRate` across instances
- Average `KVThrashingRate` across instances

Fixes #191.

### Phase 2: Correctness Bug Fixes (Tier 1)

#### 2a. KV allocation failure silently drops request (#183)

**File:** `sim/simulator.go` ~line 589

**Current:** `continue` after failed allocation -- request vanishes from simulation.

**Fix:** Convert to panic with context including block accounting state for debugging. This path is an internal consistency violation (blocks are pre-allocated for running requests). Continuing in a corrupted state is worse than crashing for a simulator that needs deterministic, accountable results:

```go
if !ok {
    panic(fmt.Sprintf("[tick %07d] KV allocation failed for completing request %s -- cache accounting invariant violated (used=%d, free=%d, total=%d)",
        now, req.ID, sim.KVCache.UsedBlocks(), sim.KVCache.TotalCapacity()-sim.KVCache.UsedBlocks(), sim.KVCache.TotalCapacity()))
}
```

#### 2b. Wrong prefix cache hashes during chunked prefill (#196)

**File:** `sim/kvcache.go` ~line 293

**Current:** Uses `newTokens`-relative `end` offset to slice `req.InputTokens[:end]`.

**Fix:** Use absolute offset:
```go
absoluteEnd := startIndex + end
fullPrefix := req.InputTokens[:absoluteEnd]
```

#### 2c. Stale `newTokenProgressIndex` after partial block fill (#197)

**File:** `sim/kvcache.go` ~line 254

**Current:** Uses post-append block length to compute advance, causing under-advancement and duplicate token writes.

**Fix:** Capture remaining capacity before append, advance by actual tokens appended:
```go
remaining := kvc.BlockSizeTokens - Len64(latestBlk.Tokens)
toksToAppend := newTokens[:min(Len64(newTokens), remaining)]
latestBlk.Tokens = append(latestBlk.Tokens, toksToAppend...)
newTokenProgressIndex += Len64(toksToAppend)
```

#### 2d. Negative `numNewTokens` in decode path (#198)

**File:** `sim/simulator.go` ~line 430

**Current:** Decode branch reuses `numNewTokens` computed for prefill, which is negative during decode (`len(InputTokens) - ProgressIndex` where `ProgressIndex > len(InputTokens)`).

**Fix:** Compute decode tokens explicitly:
```go
if req.ProgressIndex >= Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
    decodeTokens := int64(1)
    if can_schedule := sim.preempt(req, now, decodeTokens); !can_schedule {
        break
    }
```

#### 2e. Partial KV allocation leak -- transactional rollback (#200)

**File:** `sim/kvcache.go` ~line 273

**Current:** Mid-loop `return false` leaks already-allocated blocks, violating `allocated + free == total`.

**Fix:** Two-phase commit pattern -- collect blocks first, commit only on success:
```go
var newlyAllocated []*KVBlock
for i := int64(0); i < numNewBlocks; i++ {
    blk := kvc.popFreeBlock()
    if blk == nil {
        // Rollback: return blocks to free list
        for _, a := range newlyAllocated {
            a.InUse = false
            a.RefCount = 0
            kvc.UsedBlockCnt--
            kvc.pushFreeBlock(a)
        }
        return false
    }
    blk.RefCount = 1
    blk.InUse = true
    kvc.UsedBlockCnt++
    newlyAllocated = append(newlyAllocated, blk)
}
// Commit: apply token data and hashes to committed blocks
```

**Design note — rollback scope (critical):** The rollback must also undo cached-block RefCount increments from lines 240-252. If cached blocks were found and their RefCount was incremented + removed from the free list, and then the new-block loop fails, those cached-block mutations must also be rolled back. The rollback function should track both `newlyAllocated` (new blocks) and `cachedBlocksUsed` (cached blocks with incremented RefCount) and undo both on failure.

**Design note — deprecated function:** `AllocateKVBlocksPrefill` (kvcache.go:144-201) has the SAME partial allocation leak at lines 174-177. Since this function appears to be deprecated (superseded by the unified `AllocateKVBlocks`), it should be **removed entirely** in this PR rather than fixed. If it is still referenced, fix it with the same rollback pattern.

#### 2f. PendingRequests false decrement on preemption (#192)

**File:** `sim/cluster/cluster.go` ~line 160

**Current:** QueueDepth delta heuristic confuses preemption-caused QD increase with pending request absorption.

**Fix:** Track pending requests by request ID:
```go
// Replace map[string]int64 with map[string]map[string]bool
pendingByID map[string]map[string]bool
```

When routing dispatches: `pendingByID[instID][reqID] = true`.

**Implementation mechanism (critical detail):** The current `ProcessNextEvent` is opaque — the cluster doesn't know which event type was processed or which request was involved. Two options:

1. **Callback approach (recommended):** Add a `QueuedCallback func(reqID string)` field to `Simulator` or `InstanceSimulator`. When `QueuedEvent.Execute` fires, it invokes the callback. The cluster sets this callback during construction to remove the request ID from `pendingByID`. This is minimally invasive — one new field, one callback invocation.

2. **Return-value approach:** Change `ProcessNextEvent` to return metadata about what was processed (event type, request ID). More informative but changes the method signature.

Preemption does not invoke the queued callback (preemption re-enqueues via direct slice manipulation, not via `QueuedEvent`), so it cannot cause false decrements.

#### 2g. Completion condition fragility for zero-output requests

**File:** `sim/simulator.go` ~line 586

**Current:** The completion check `req.ProgressIndex == Len64(req.InputTokens)+max(Len64(req.OutputTokens), 1)-1` is accidentally correct — it works only because the completion loop (lines 584-629) runs AFTER the prefill execution loop (lines 553-569) in separate passes over `RunningBatch.Requests`. For a request with 0 output tokens, both "prefill just completed" (line 564) and "request completed" (line 586) are true in the same step. If these loops were ever consolidated into a single pass, both branches would fire.

**Fix:** Add a documenting comment explaining the two-pass dependency:
```go
// IMPORTANT: This completion loop MUST run as a separate pass after the
// prefill/decode execution loop (lines 553-569). For zero-output-token
// requests, both "prefill completed" and "request completed" conditions
// are true in the same step. The two-pass design ensures prefill metrics
// (TTFT) are recorded before completion metrics (E2E).
```

This is a documentation fix, not a code change. Full decomposition is deferred to the SGLang engine work.

### Phase 3: Data Loss / Metric Distortion Fixes (Tier 2)

#### 3a. SaveResults drops incomplete requests (#190)

**File:** `sim/metrics.go` ~line 130

**Fix:** Iterate over `m.Requests` (all registered) instead of `m.RequestTTFTs` (only completed prefill). Incomplete requests get zero-valued TTFT/E2E/ITL fields (distinguishable from real measurements, which are always positive).

#### 3b. SLO attainment inflation (#201)

**File:** `sim/cluster/metrics.go` ~lines 229-282

**Fix:** Replace silent `continue` with a counter. Log dropped count at Warn level. For `SLOAttainment`, include dropped requests in the denominator (conservative: treat missing data as SLO violation).

#### 3c. CacheHits double-counted in tiered mode

**File:** `sim/kvcache.go` ~line 134, `sim/kvcache_tiered.go` ~line 73

**Current:** `GetCachedBlocks()` increments `CacheHits++` at line 134 despite its doc comment claiming it is "pure" and does not modify state. In tiered mode, `TieredKVCache.AllocateKVBlocks` calls `t.gpu.GetCachedBlocks(req.InputTokens)` a second time after a CPU-to-GPU reload (line 73), inflating the hit counter. This distorts the `CacheHitRate()` metric.

**Fix:** Move the `CacheHits++` increment out of `GetCachedBlocks` to the allocation call site that actually uses the cached blocks for allocation. `GetCachedBlocks` becomes a true query (matching its doc comment). The caller in `AllocateKVBlocks` increments `CacheHits` once per block actually reused.

Alternatively, add a `countHits bool` parameter to `GetCachedBlocks`, but this pollutes the interface. The move-to-caller approach is cleaner.

**Also fix:** Update the doc comment on `GetCachedBlocks` to remove the false "pure method" claim, or make it actually pure (preferred).

#### 3d. ComputeFitness unknown keys fail-fast (#203)

**File:** `sim/cluster/metrics.go` ~line 355

**Fix:** Validate all keys before computing. Return error for unknown keys:
```go
func ComputeFitness(metrics *RawMetrics, weights map[string]float64) (FitnessResult, error) {
    for key := range weights {
        if _, ok := extractMetric(metrics, key); !ok {
            return FitnessResult{}, fmt.Errorf("unknown fitness metric key %q", key)
        }
    }
    // ... compute with sorted keys
}
```

The CLI caller in `cmd/root.go` converts the error to `logrus.Fatalf`.

**Signature change note:** Adding `error` return to `ComputeFitness` requires updating all call sites. There are few callers (cmd/root.go evaluation path, metrics.go fitness computation).

### Phase 4: Invariant Tests (#199)

These tests prevent the entire class of bugs from recurring.

#### 4a. Request Conservation

After every simulation run: `injected == completed + queued + running`.

```go
func TestSimulator_RequestConservation_AllRequestsAccountedFor(t *testing.T) {
    // Run simulation with each golden dataset model
    // Assert: len(sim.Metrics.Requests) == sim.Metrics.CompletedRequests + queuedCount + runningCount
}
```

Would have caught #183 on day one.

#### 4b. KV Block Conservation

After every allocate/release cycle: `UsedBlocks() + freeBlocks == TotalCapacity()`.

```go
func TestKVCache_BlockConservation_AllocatedPlusFreeEqualsTotal(t *testing.T) {
    // Allocate, partially release, verify conservation holds
    // Run a full simulation, verify at end
}
```

Would catch #200 if the guard condition ever fails.

#### 4c. Causality

For every completed request: `ArrivalTime <= TTFT_time <= CompletionTime`. All ITL values >= 0.

#### 4d. Clock Monotonicity

After every event processed: `clock_after >= clock_before`. Both single-instance (`sim.Clock`) and cluster-level (`c.clock`).

```go
func TestSimulator_ClockMonotonicity_NeverDecreases(t *testing.T) {
    // Run simulation, instrument event processing to record clock at each step
    // Assert: clock values form a non-decreasing sequence
}
```

Currently masked by heap ordering of event queues, but a bug in event scheduling or timestamp computation could violate it. This is documented in CLAUDE.md as a key invariant but has no test.

#### 4e. Determinism

Run the same simulation twice with the same seed, assert byte-identical JSON output. Catches #195 and any future non-determinism.

#### 4f. Golden Dataset Regeneration

After fixing #183, golden dataset values will change. Regenerate with fixed code. Document the regeneration command so future contributors can update golden values when intentional behavior changes occur.

**Validation of regenerated values:** The regenerated golden dataset must pass ALL invariant tests (4a-4e) before being committed. This ensures the new golden values are correct, not just different. The invariant tests serve as the acceptance criteria for golden dataset regeneration.

### Phase 5: Input Validation

#### 5a. Rate > 0 (#202)

**File:** `cmd/root.go`

```go
if rate <= 0 {
    logrus.Fatalf("--rate must be > 0, got %f", rate)
}
```

#### 5b. NaN/Inf/negative fitness weights

**File:** `sim/cluster/metrics.go` `ParseFitnessWeights`

```go
if math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
    return nil, fmt.Errorf("invalid weight value for %q: %f", key, val)
}
```

#### 5c. CalculatePercentile panics on empty input

**File:** `sim/metrics_utils.go` ~line 67

**Current:** `CalculatePercentile` computes `rank = p/100.0 * float64(n-1)`. When `data` is empty (`n=0`), `rank` becomes negative, producing `lowerIdx = -1` and an array index out-of-bounds panic. The cluster-side `percentile` function handles empty input with an early return, but `CalculatePercentile` does not.

**Trigger path:** If all completed requests have 0 output tokens, `m.AllITLs` could be empty while `m.CompletedRequests > 0`, bypassing the `CompletedRequests > 0` guard in `SaveResults`.

**Fix:** Add an empty-input guard:
```go
func CalculatePercentile(data []float64, p float64) float64 {
    if len(data) == 0 {
        return 0
    }
    // ... existing logic
}
```

### Phase 6: Modularity Improvements (Reduce Extension Friction)

These changes reduce the "touch-point multiplier" for future extensions without changing external behavior.

#### 6a. Observation methods on Simulator

**File:** `sim/simulator.go`

Add accessor methods so `InstanceSimulator` doesn't reach through exported fields:

```go
func (s *Simulator) QueueDepth() int    { return s.WaitQ.Len() }
func (s *Simulator) BatchSize() int     { if s.RunningBatch == nil { return 0 }; return len(s.RunningBatch.Requests) }
func (s *Simulator) CurrentClock() int64 { return s.Clock }
func (s *Simulator) SimHorizon() int64  { return s.Horizon }
```

Then update `sim/cluster/instance.go` to call these methods instead of reaching through `i.sim.Clock`, `i.sim.WaitQ.Len()`, `i.sim.RunningBatch.Requests`, etc.

**Why:** Adding a new observable (e.g., PrefillQueueDepth for P/D) becomes: add method to Simulator + add to InstanceSnapshot + wire through snapshot provider. Currently it's: understand which exported field to reach through + hope the access pattern is right.

**Impact on heterogeneous accelerators:** Per-instance observation works through the same method interface regardless of underlying hardware config.

#### 6b. Eliminate InstanceSnapshot/RoutingSnapshot duplication

**File:** `sim/cluster/snapshot.go`, `sim/cluster/cluster_event.go`, `sim/routing.go`

**Current:** `InstanceSnapshot` (cluster pkg, 7 fields) and `RoutingSnapshot` (sim pkg, 7 fields) are near-identical. `buildRouterState()` manually copies field-by-field, silently dropping any field that exists in one but not the other.

**Option A (recommended): RoutingSnapshot becomes the canonical type.** Remove `InstanceSnapshot`, have `CachedSnapshotProvider.Snapshot()` return `sim.RoutingSnapshot` directly. Eliminates the translation step entirely.

This is safe because `RoutingSnapshot` lives in `sim/` (no import cycle) and `snapshot.go` already imports `sim/`. The `PendingRequests` field is currently injected during translation — it would be set directly by the snapshot provider (which already has access to `cs.pendingRequests` via its closure/reference).

**Impact:** Adding a new observable field becomes 3-4 files instead of 6: Simulator method + RoutingSnapshot field + SnapshotProvider wiring + (optionally) ObservabilityConfig entry.

#### 6c. Fix PendingTransferLatency destructive-read semantics

**File:** `sim/kvcache_tiered.go`

**Current:** `PendingTransferLatency()` clears `pendingLatency` on read (line 168-172). A second call returns 0.

**Fix:** Separate query from mutation:
```go
func (t *TieredKVCache) PendingTransferLatency() int64 {
    return t.pendingLatency  // pure query, no side effect
}

func (t *TieredKVCache) ConsumePendingTransferLatency() int64 {
    lat := t.pendingLatency
    t.pendingLatency = 0
    return lat
}
```

Update `Simulator.Step()` to call `ConsumePendingTransferLatency()`. The `KVStore` interface method `PendingTransferLatency()` remains a pure query.

**Why:** Any future code that reads transfer latency for tracing, monitoring, or routing decisions won't accidentally clear the value. This is critical for LMCache integration where transfer latency affects routing decisions.

#### 6d. Add SetClock to KVStore interface

**File:** `sim/kv_store.go`, `sim/kvcache.go`, `sim/simulator.go`

**Current:** `Simulator.Step()` type-asserts `KVCache` to `*TieredKVCache` to call `SetClock()`.

**Fix:** Add `SetClock(clock int64)` to the `KVStore` interface. Single-tier `KVCacheState` implements it as a no-op. `TieredKVCache` implements it with actual clock tracking.

```go
type KVStore interface {
    // ... existing 9 methods ...
    SetClock(clock int64) // called each step; no-op for non-tiered implementations
}
```

**Why:** Eliminates the type assertion. Any new tiered KV implementation (NVMe, distributed) that needs clock synchronization gets it automatically. This is a prerequisite for LMCache and P/D disaggregation.

**Interface freeze note:** This adds a method to a frozen interface. This is justified because it replaces a type assertion that already couples the simulator to a concrete type — the interface becomes more honest about the actual contract.

#### 6e. Replace logrus.Fatalf in library code with error returns

**Files:** `sim/workload_config.go`, `sim/cluster/workload.go`, `sim/metrics_utils.go`

**Current:** These files call `logrus.Fatalf` (which calls `os.Exit(1)`) for CSV parsing errors and file I/O errors deep inside the `sim/` package.

**Fix:** Return `error` from these functions. The caller in `cmd/root.go` converts errors to `logrus.Fatalf`.

```go
// Before (sim/workload_config.go)
func (sim *Simulator) generateWorkloadFromCSV() {
    logrus.Fatalf("failed to open csv: %v", err)  // kills process
}

// After
func (sim *Simulator) generateWorkloadFromCSV() error {
    return fmt.Errorf("failed to open csv: %w", err)
}
```

**Why:** Makes `sim/` usable as a library (for PR15 framework adapters, for test harnesses, for embedding in larger tools). Currently impossible to catch CSV parse errors programmatically.

#### 6f. Derive CLI valid-name lists from bundle.go

**File:** `cmd/root.go`, `sim/bundle.go`

**Current:** `cmd/root.go:294` hardcodes "Valid: always-admit, token-bucket, reject-all" while `bundle.go` has the authoritative `validAdmissionPolicies` map. These can drift.

**Fix:** Add `ValidPolicyNames(policyType string) []string` to `bundle.go` that returns sorted names from the maps. CLI error messages use this function instead of hardcoded strings:

```go
logrus.Fatalf("Unknown admission policy %q. Valid: %s", name,
    strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
```

**Why:** Adding a new policy template becomes a 2-file change (implementation + bundle registration) instead of 3. The CLI error messages are always correct.

#### 6g. Request state constants

**File:** `sim/request.go`

**Current:** `State` is a raw `string` field written 3 times, never read for control flow.

**Fix:** Define typed constants:
```go
type RequestState string

const (
    StateQueued    RequestState = "queued"
    StateRunning   RequestState = "running"
    StateCompleted RequestState = "completed"
)
```

Change `Request.State` from `string` to `RequestState`. Update the 3 assignment sites.

**Why:** Compiler catches typos. IDE autocomplete shows valid states. Future invariant tests can check valid transitions. Not a full state machine (no transition enforcement) — just type safety and discoverability.

#### 6h. NewKVStore input validation

**File:** `sim/kv_store.go`

**Current:** `NewKVStore` performs zero validation. `NewSimulator` validates KV fields, but `NewKVStore` called independently accepts `TotalKVBlocks <= 0`.

**Fix:** Add validation matching the policy factory pattern:
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

**Why:** Consistent factory behavior. Any future caller (including PR15 adapter constructors) gets immediate feedback on invalid config.

## File Impact Summary

| File | Changes | Issues Fixed |
|---|---|---|
| `sim/routing.go` | Add `EffectiveLoad()`, update 3 policies | #175 |
| `sim/metrics_utils.go` | Add `NewRequestMetrics()`, error returns for I/O, CalculatePercentile empty guard | #189, panic fix |
| `sim/simulator.go` | Fix decode numNewTokens, KV alloc panic (with accounting state), add completion condition comment, add observation methods, remove KV type assertion | #183, #198, 2g |
| `sim/kvcache.go` | Fix prefix hash, stale index, add rollback (with cached-block rollback), move CacheHits to caller, fix GetCachedBlocks purity, remove deprecated AllocateKVBlocksPrefill | #196, #197, #200, CacheHits fix |
| `sim/kvcache_tiered.go` | Non-destructive PendingTransferLatency, add SetClock | destructive-read fix |
| `sim/kv_store.go` | Add SetClock to interface, add NewKVStore validation | type assertion, validation |
| `sim/request.go` | RequestState typed constants | state enforcement |
| `sim/metrics.go` | Iterate Requests not RequestTTFTs | #190 |
| `sim/bundle.go` | Add ValidPolicyNames helpers | CLI name drift |
| `sim/workload_config.go` | Return errors instead of logrus.Fatalf | library os.Exit |
| `sim/cluster/cluster.go` | Complete aggregation, ID-based pending tracking (with QueuedCallback mechanism) | #191, #192 |
| `sim/cluster/metrics.go` | Sorted keys, fail-fast fitness, SLO counter | #195, #201, #203 |
| `sim/cluster/instance.go` | Use Simulator observation methods | coupling reduction |
| `sim/cluster/snapshot.go` | Use RoutingSnapshot directly (eliminate InstanceSnapshot) | snapshot duplication |
| `sim/cluster/cluster_event.go` | Simplify buildRouterState (no translation) | snapshot duplication |
| `sim/cluster/workload.go` | Return errors instead of logrus.Fatalf | library os.Exit |
| `cmd/root.go` | Rate>0, NaN/Inf validation, derive valid names | #202, name drift |
| `sim/simulator_test.go` | Conservation, causality, determinism tests | #199 |
| `sim/kvcache_test.go` | Block conservation test | #199 |
| `testdata/goldendataset.json` | Regenerate after fixes | -- |

**Estimated scope:** ~20 files, ~800-1100 LOC changes, ~200-300 LOC new tests.

## Dependency Order

```
Phase 1 (helpers) -> Phase 2 (Tier 1 fixes) -> Phase 3 (Tier 2 fixes) -> Phase 4 (invariant tests) -> Phase 5 (validation) -> Phase 6 (modularity)
```

Phase 1 must come first because Phase 2 fixes depend on the helpers (e.g., `EffectiveLoad()` *is* the fix for the routing policy inconsistency). Phase 4 comes after Phase 2-3 because the invariant tests should pass on the fixed code. Phase 5 is independent but logically last (input boundary, not internal correctness). Phase 6 is independent of Phases 2-5 but should come after Phase 1 because it builds on the structural helpers.

Within Phase 6, the dependency order is:
```
6a (observation methods) -> 6b (snapshot unification, depends on 6a)
6c (PendingTransferLatency) + 6d (SetClock) can be parallel
6e (error returns) + 6f (CLI names) + 6g (request state) + 6h (KVStore validation) are all independent
```

## Risks

1. **Golden dataset values will change.** Fixing #183 changes completion counts. This is expected and correct -- the new invariant tests provide a stronger correctness guarantee.

2. **`ComputeFitness` signature change.** Adding `error` return requires updating all callers. There are few (~2-3 call sites).

3. **PendingRequests tracking change (#192).** Switching from QD-delta heuristic to ID-based tracking is a behavioral change in the cluster event loop. Needs careful testing with preemption-heavy workloads.

4. **KVStore interface change (6d).** Adding `SetClock()` to the frozen interface. Justified because it replaces an existing type assertion — the actual contract is unchanged, just made explicit.

5. **InstanceSnapshot removal (6b).** `CachedSnapshotProvider` return type changes from `InstanceSnapshot` to `sim.RoutingSnapshot`. All snapshot consumers must be updated. The `SnapshotProvider` interface signature changes.

6. **Error return propagation (6e).** Changing `generateWorkloadFromCSV()` to return `error` requires updating the call chain through `NewSimulator` or `Run()`. May touch `Simulator.Run()` signature.

7. **CacheHits relocation (3c).** Moving `CacheHits++` from `GetCachedBlocks` to the caller changes the counting semantics. Must verify that single-tier mode (which doesn't double-call `GetCachedBlocks`) still counts correctly.

8. **QueuedCallback mechanism (2f).** Adding a callback field to `Simulator` or `InstanceSimulator` is a new coupling point. Must ensure the callback is nil-safe (no-op when cluster layer is not present, i.e., single-instance mode).

9. **Deprecated function removal (2e).** Removing `AllocateKVBlocksPrefill` requires verifying no code path still calls it. If it is referenced by tests or legacy code, those references must be updated.

## Out of Scope

- Documentation-only issues (#182, #184, #194) -- separate PR
- Horizon warning (#193) -- nice-to-have, not a correctness issue
- PendingRequests trace completeness (#176) -- enhancement, not a bug
- Stale comment cleanup (#194) -- trivial, can be folded into any PR
- SimConfig decomposition into sub-configs -- nice-to-have, Go-idiomatic as-is
- Eviction policy abstraction in KVCacheState -- defer to when a non-LRU strategy is needed
- WaitQueue encapsulation (Simulator bypasses WaitQ methods) -- intra-package, low external impact
- Simulator.Step() decomposition -- large refactor, defer to when a second engine (SGLang) is actually added
- Instance lifecycle abstraction (AddInstance/RemoveInstance) -- PR11 scope
- Per-instance SimConfig for heterogeneous hardware -- PR11 or separate

## Relationship to Remaining PRs

This PR ships before PR11 (AutoScaler). The bug fixes, invariant tests, and modularity improvements act as a foundation:

- **PR11 (Autoscaling):** Observation methods (6a) and snapshot unification (6b) reduce the touch-point count for new observables. ID-based pending tracking (2f) is robust against instance addition/removal. Instance lifecycle abstraction is deferred to PR11 itself.
- **PR14 (P/D Disaggregation):** KV block conservation and rollback (2e) protect against allocation bugs. SetClock on interface (6d) and non-destructive PendingTransferLatency (6c) enable clean cross-instance KV transfer. Snapshot unification (6b) makes adding PrefillQueueDepth/DecodeQueueDepth a 3-file change.
- **PR15 (Framework Adapters):** Error returns in library code (6e) make `sim/` embeddable. ValidPolicyNames (6f) enables adapter-level config validation. NewKVStore validation (6h) catches invalid configs from adapter constructors.
- **PR16 (Integration Tests):** Building on a correct, modular foundation with invariant tests avoids encoding bugs as golden values.
- **Future (LMCache):** Non-destructive PendingTransferLatency (6c) enables routing-aware cache placement. SetClock on interface (6d) enables distributed cache coherence.
- **Future (SGLang):** Step() decomposition is explicitly deferred, but observation methods (6a) and request state constants (6g) make the current engine boundary more visible for future extraction.
- **Future (Heterogeneous accelerators):** Observation methods (6a) decouple instance monitoring from Simulator field layout. Per-instance SimConfig is deferred but enabled by the cleaner observation layer.
