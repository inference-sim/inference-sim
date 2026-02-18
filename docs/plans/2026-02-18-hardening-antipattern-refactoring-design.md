# Hardening PR: Antipattern Elimination and Invariant Tests

**Date:** 2026-02-18
**Status:** Approved design
**Scope:** Dedicated hardening PR before PR11 (AutoScaler)
**Issues addressed:** #175, #183, #189, #190, #191, #192, #195, #196, #197, #198, #199, #200, #201, #202, #203

## Motivation

A systematic audit of the codebase uncovered 20+ issues that cluster into six antipattern categories. Most share a root cause: the codebase tests outputs (golden values, metric snapshots) rather than laws (conservation, monotonicity, causality). Output tests answer "did the number change?" but cannot answer "is the number right?"

The golden dataset test encoded a wrong value (499 completions instead of 500 for codellama) because one request was silently dropped by a bug (#183). The test perpetuated the bug rather than catching it. Conservation invariant tests would have caught this on day one.

This PR fixes all known correctness bugs and adds structural defenses to prevent recurrence, establishing a clean foundation before PR11-PR16.

## Antipattern Summary

| Pattern | Count | Root Cause |
|---|---|---|
| Shotgun surgery (multiple construction sites) | 3 sites | No canonical constructors |
| Golden tests encoding bugs | systemic | Testing values, not laws |
| Silent `continue` dropping data | 4 paths | Error paths that skip cleanup |
| Non-deterministic map iteration | 5 sites | Unsorted map keys in float accumulation |
| Missing input validation | 2 | No guards on Rate=0, NaN/Inf weights |
| No transactional rollback | 1 | Mid-loop exit leaks allocated blocks |

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

**Fix:** Convert to panic with context. This path is an internal consistency violation (blocks are pre-allocated for running requests). Continuing in a corrupted state is worse than crashing for a simulator that needs deterministic, accountable results:

```go
if !ok {
    panic(fmt.Sprintf("[tick %07d] KV allocation failed for completing request %s -- cache accounting invariant violated", now, req.ID))
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

**Design note:** Also roll back cached-block RefCount increments from earlier in the function (lines 240-252) if the new-block loop fails.

#### 2f. PendingRequests false decrement on preemption (#192)

**File:** `sim/cluster/cluster.go` ~line 160

**Current:** QueueDepth delta heuristic confuses preemption-caused QD increase with pending request absorption.

**Fix:** Track pending requests by request ID:
```go
// Replace map[string]int64 with map[string]map[string]bool
pendingByID map[string]map[string]bool
```

When routing dispatches: `pendingByID[instID][reqID] = true`.
When QueuedEvent fires for a tracked request: delete from set.
Preemption does not add request IDs to the pending set, so it cannot cause false decrements.

### Phase 3: Data Loss / Metric Distortion Fixes (Tier 2)

#### 3a. SaveResults drops incomplete requests (#190)

**File:** `sim/metrics.go` ~line 130

**Fix:** Iterate over `m.Requests` (all registered) instead of `m.RequestTTFTs` (only completed prefill). Incomplete requests get zero-valued TTFT/E2E/ITL fields (distinguishable from real measurements, which are always positive).

#### 3b. SLO attainment inflation (#201)

**File:** `sim/cluster/metrics.go` ~lines 229-282

**Fix:** Replace silent `continue` with a counter. Log dropped count at Warn level. For `SLOAttainment`, include dropped requests in the denominator (conservative: treat missing data as SLO violation).

#### 3c. ComputeFitness unknown keys fail-fast (#203)

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

#### 4d. Determinism

Run the same simulation twice with the same seed, assert byte-identical JSON output. Catches #195 and any future non-determinism.

#### 4e. Golden Dataset Regeneration

After fixing #183, golden dataset values will change. Regenerate with fixed code. Document the regeneration command so future contributors can update golden values when intentional behavior changes occur.

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

## File Impact Summary

| File | Changes | Issues Fixed |
|---|---|---|
| `sim/routing.go` | Add `EffectiveLoad()`, update 3 policies | #175 |
| `sim/metrics_utils.go` | Add `NewRequestMetrics()` | #189 |
| `sim/simulator.go` | Fix decode numNewTokens, KV alloc panic, use factory | #183, #198 |
| `sim/kvcache.go` | Fix prefix hash, stale index, add rollback | #196, #197, #200 |
| `sim/metrics.go` | Iterate Requests not RequestTTFTs | #190 |
| `sim/cluster/cluster.go` | Complete aggregation, ID-based pending tracking | #191, #192 |
| `sim/cluster/metrics.go` | Sorted keys, fail-fast fitness, SLO counter | #195, #201, #203 |
| `cmd/root.go` | Rate>0, NaN/Inf validation | #202 |
| `sim/simulator_test.go` | Conservation, causality, determinism tests | #199 |
| `sim/kvcache_test.go` | Block conservation test | #199 |
| `testdata/goldendataset.json` | Regenerate after fixes | -- |

**Estimated scope:** ~11 files, ~500-700 LOC changes, ~200-300 LOC new tests.

## Dependency Order

```
Phase 1 (helpers) -> Phase 2 (Tier 1 fixes) -> Phase 3 (Tier 2 fixes) -> Phase 4 (invariant tests) -> Phase 5 (validation)
```

Phase 1 must come first because Phase 2 fixes depend on the helpers (e.g., `EffectiveLoad()` *is* the fix for the routing policy inconsistency). Phase 4 comes after Phase 2-3 because the invariant tests should pass on the fixed code. Phase 5 is independent but logically last (input boundary, not internal correctness).

## Risks

1. **Golden dataset values will change.** Fixing #183 changes completion counts. This is expected and correct -- the new invariant tests provide a stronger correctness guarantee.

2. **`ComputeFitness` signature change.** Adding `error` return requires updating all callers. There are few (~2-3 call sites).

3. **PendingRequests tracking change (#192).** Switching from QD-delta heuristic to ID-based tracking is a behavioral change in the cluster event loop. Needs careful testing with preemption-heavy workloads.

## Out of Scope

- Documentation-only issues (#182, #184, #194) -- separate PR
- Horizon warning (#193) -- nice-to-have, not a correctness issue
- PendingRequests trace completeness (#176) -- enhancement, not a bug
- Stale comment cleanup (#194) -- trivial, can be folded into any PR

## Relationship to Remaining PRs

This PR ships before PR11 (AutoScaler). The invariant tests act as a safety net for all subsequent PRs:
- **PR11** adds new event types and metrics -- conservation tests catch any accounting gaps
- **PR14** touches kvcache.go heavily for P/D transfer -- KV block conservation and rollback protect against allocation bugs
- **PR16** adds integration tests -- building on a correct foundation avoids encoding bugs as golden values
