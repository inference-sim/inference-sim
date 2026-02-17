# PR12 Architectural Pre-Design: Tiered KV Cache + Transfer

**Purpose:** Resolve all architectural ambiguities that block the micro-planning process for PR12.
This document provides **binding design decisions** — the micro-plan should consume these directly,
not re-derive them.

**Status:** Draft (pending human review)

---

## Decision 1: No `sim/kv/` Package — Keep Everything in `sim/`

**Problem:** The macro plan places new code in `sim/kv/tiered.go` and `sim/kv/transfer.go`. But:
- `KVCacheState`, `Request`, and `Simulator` all live in `sim/`
- `sim/kv/` importing `sim/` is fine, but `sim/simulator.go` would also need to import `sim/kv/` for `TieredKVCache` → **import cycle**
- Solving this requires either a bridge type (like `RouterState`) or an interface in `sim/`

**Decision:** Keep all PR12 code in `sim/`. Create two new files:
- `sim/kv_store.go` — `KVStore` interface + factory
- `sim/kvcache_tiered.go` — `TieredKVCache` implementation (~200 LOC)

**Rationale:**
- No import cycle to solve
- `TieredKVCache` composes `*KVCacheState` (GPU tier) — same-package access is natural
- The `sim/kv/` package is better created in PR14 (P/D disaggregation) when cross-instance transfers require cluster-level coordination
- ~300 LOC of new code in `sim/` is not excessive (package is ~2,700 LOC non-test)

**Deviation from macro plan:** `sim/kv/tiered.go` and `sim/kv/transfer.go` become `sim/kv_store.go` and `sim/kvcache_tiered.go`. Log in deviation table.

---

## Decision 2: Introduce `KVStore` Interface

**Problem:** `Simulator.KVCache` is a concrete `*KVCacheState` with direct field access. The tiered cache needs a different implementation.

**Decision:** Define a `KVStore` interface in `sim/kv_store.go`:

```go
// KVStore abstracts KV cache operations for the simulator.
// KVCacheState (single-tier GPU) and TieredKVCache (GPU+CPU) both implement this.
type KVStore interface {
    AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool
    GetCachedBlocks(tokens []int) []int64
    ReleaseKVBlocks(req *Request)
    BlockSize() int64
    UsedBlocks() int64
    TotalCapacity() int64
    CacheHitRate() float64           // cumulative hit rate: hits / (hits + misses); 0 if no lookups
    PendingTransferLatency() int64   // transfer latency accumulated since last query; resets to 0 on read
}
```

**Changes to `KVCacheState`:**
Add five accessor methods (thin wrappers over public fields + counters):
```go
func (kvc *KVCacheState) BlockSize() int64              { return kvc.BlockSizeTokens }
func (kvc *KVCacheState) UsedBlocks() int64              { return kvc.UsedBlockCnt }
func (kvc *KVCacheState) TotalCapacity() int64           { return kvc.TotalBlocks }
func (kvc *KVCacheState) CacheHitRate() float64 {
    total := kvc.CacheHits + kvc.CacheMisses
    if total == 0 { return 0 }
    return float64(kvc.CacheHits) / float64(total)
}
func (kvc *KVCacheState) PendingTransferLatency() int64  { return 0 } // single-tier: no transfers
```
`KVCacheState` satisfies `KVStore` with all 8 methods.

**Changes to `Simulator`:**
```go
// Before: KVCache *KVCacheState
// After:
KVCache KVStore
```

**Changes to `InstanceSimulator`:**
```go
// Before: i.sim.KVCache.UsedBlockCnt / i.sim.KVCache.TotalBlocks
// After:
func (i *InstanceSimulator) KVUtilization() float64 {
    return float64(i.sim.KVCache.UsedBlocks()) / float64(i.sim.KVCache.TotalCapacity())
}
func (i *InstanceSimulator) FreeKVBlocks() int64 {
    return i.sim.KVCache.TotalCapacity() - i.sim.KVCache.UsedBlocks()
}
```

**Changes in `simulator.go` — 6 field-access sites become method calls:**
| Line | Before | After |
|------|--------|-------|
| 446 | `sim.KVCache.BlockSizeTokens` | `sim.KVCache.BlockSize()` |
| 453 | `sim.KVCache.BlockSizeTokens` | `sim.KVCache.BlockSize()` |
| 492 | `sim.KVCache.BlockSizeTokens` | `sim.KVCache.BlockSize()` |
| 559 | `sim.KVCache.UsedBlockCnt` | `sim.KVCache.UsedBlocks()` |
| 560 | `sim.KVCache.UsedBlockCnt` | `sim.KVCache.UsedBlocks()` |
| 562 | `sim.KVCache.UsedBlockCnt` | `sim.KVCache.UsedBlocks()` |

`NewSimulator` construction changes:
```go
// Before: KVCache: NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens),
// After:  KVCache: NewKVStore(cfg),
```

**Backward compatibility:** `NewKVStore(cfg)` returns `*KVCacheState` when `cfg.KVCPUBlocks == 0` (default), returns `*TieredKVCache` otherwise. Identical behavior for all existing tests.

---

## Decision 3: CPU Tier is a Simple Capacity Store (Not a `KVCacheState`)

**Problem:** Should the CPU tier reuse `KVCacheState` (with all its LRU, prefix caching, hashing)?

**Decision:** No. The CPU tier is a simple capacity-tracked block store:

```go
type OffloadedBlock struct {
    OriginalID int64   // block ID on GPU tier
    Tokens     []int   // token content (for reload)
    Hash       string  // prefix hash (for cache hit detection)
}

type cpuTier struct {
    blocks   map[int64]*OffloadedBlock // keyed by original GPU block ID
    capacity int64
    used     int64
}
```

**Rationale:**
- CPU tier doesn't need prefix caching (blocks are offloaded, not allocated fresh)
- CPU tier doesn't need LRU eviction (offloaded blocks are evicted FIFO or by age)
- CPU tier doesn't need the complex `AllocateKVBlocks` allocation logic
- Reusing `KVCacheState` for CPU would require two full KV cache instances with all their state — overkill for a capacity buffer

---

## Decision 4: `TieredKVCache` Composition Structure

```go
type TieredKVCache struct {
    gpu              *KVCacheState       // GPU tier — delegates all normal operations here
    cpu              cpuTier             // CPU tier — simple offload storage
    offloadThreshold float64            // GPU utilization that triggers offload (e.g., 0.9)
    transferBandwidth float64           // blocks per tick
    baseLatency      int64              // fixed cost per transfer operation (ticks)

    // Metrics counters
    offloadCount     int64              // total blocks offloaded GPU→CPU
    reloadCount      int64              // total blocks reloaded CPU→GPU
    cpuHitCount      int64              // cache hits found on CPU tier
    cpuMissCount     int64              // lookups that found nothing on CPU
    thrashingCount   int64              // offload followed by reload within N ticks
}
```

**Delegation pattern:**
- `AllocateKVBlocks` → delegates to `gpu.AllocateKVBlocks`. If that fails and CPU has the block, triggers reload (accumulates transfer latency internally).
- `GetCachedBlocks` → delegates to `gpu.GetCachedBlocks`. If partial match, checks CPU tier for additional blocks.
- `ReleaseKVBlocks` → delegates to `gpu.ReleaseKVBlocks`. Also cleans up any CPU-tier copies.
- `BlockSize()` → delegates to `gpu.BlockSize()`
- `UsedBlocks()` → returns `gpu.UsedBlocks()` (GPU is the active tier)
- `TotalCapacity()` → returns `gpu.TotalCapacity()` (GPU only — CPU tier is an overflow buffer, not part of "active" capacity visible to routing policies. This matches vLLM behavior where CPU offload is transparent to scheduling logic.)
- `CacheHitRate()` → combines GPU prefix cache hits + CPU tier hits: `float64(gpu.CacheHits + cpuHitCount) / float64(gpu.CacheHits + gpu.CacheMisses + cpuHitCount + cpuMissCount)`
- `PendingTransferLatency()` → returns accumulated transfer latency and resets to 0 (query-and-clear)

**Offload trigger:** After each `ReleaseKVBlocks` or at end of step, if `gpu.UsedBlocks()/gpu.TotalCapacity() > offloadThreshold`, offload LRU free blocks from GPU to CPU until below threshold.

---

## Decision 5: Transfer Latency is Modeled as Synchronous Delay (No Events)

**Problem:** The macro plan says "adds KVTransfer events to event queue." But:
- A new event type requires integration with the min-heap event queue
- Transfer events need to block the requesting process (request can't proceed while block is in transit)
- The existing event types (`StepEvent`, `ArrivalEvent`, etc.) are relatively coarse-grained

**Decision:** Model transfer latency as **synchronous delay added to step time**, not as separate events.

**Mechanism:**
- When `TieredKVCache.AllocateKVBlocks` encounters a GPU allocation failure but finds the block on CPU tier:
  1. Reload the block from CPU to GPU (update internal state instantly)
  2. Return `true` (allocation succeeded) and accumulate the transfer time internally
  3. The accumulated transfer time is accessible via `PendingTransferLatency() int64`
- `PendingTransferLatency()` is **query-and-clear**: returns the accumulated latency and resets to 0. This prevents double-counting across steps.
- Transfer latency formula: `baseLatency + ceil(numBlocks * blockSize / bandwidth)`

**Integration point in `simulator.go`:**
In the `Step()` method, **after** `makeRunningBatch()` returns and **before** computing step time:
```go
// In Step(), after makeRunningBatch(now) completes:
transferLatency := sim.KVCache.PendingTransferLatency()  // query-and-clear
// ... existing step time computation ...
currStepAdvance += transferLatency  // add transfer delay to step duration
```
This ensures CPU→GPU reload latency is reflected in the step that triggered it.
For `KVCacheState` (single-tier), `PendingTransferLatency()` always returns 0 — no overhead.

**Rationale:**
- Simpler implementation (~50 LOC vs ~150 LOC for event-based)
- Transfer latency is on the critical path anyway (request blocks on it)
- Avoids complex event ordering between transfer completion and step execution
- Matches how vLLM handles CPU→GPU copies in practice (synchronous in the step loop)
- A future PR can upgrade to async events if needed for P/D disaggregation

**Deviation from macro plan:** "KVTransfer events" becomes "synchronous transfer latency in step." Log in deviation table with rationale.

---

## Decision 6: Metrics Integration

**New fields on `RawMetrics` (in `sim/cluster/metrics.go`):**
```go
CacheHitRate    float64  // fraction of block lookups that found cached data (GPU or CPU)
PreemptionRate  float64  // fraction of requests that experienced at least one preemption
KVThrashingRate float64  // offloads followed by reload within 1000 ticks / total offloads
```

These were deferred from PR9 (needed KV hit/miss tracking). PR12 adds the tracking.

**New field on `RoutingSnapshot` (in `sim/routing.go`):**
```go
CacheHitRate float64  // cumulative cache hit rate for this instance (0.0-1.0)
```

**Wire-up path for `CacheHitRate` on `RoutingSnapshot`:**
1. `KVStore.CacheHitRate()` returns cumulative hit rate (part of the interface)
2. `InstanceSimulator` gets a new accessor: `CacheHitRate() float64 { return i.sim.KVCache.CacheHitRate() }`
3. `CachedSnapshotProvider` calls `inst.CacheHitRate()` when building `RoutingSnapshot` (same pattern as `KVUtilization()`)
4. `RoutingSnapshot.CacheHitRate` is populated and available to routing policies
5. `buildRouterState()` in `sim/cluster/cluster_event.go:63-69` must also copy `CacheHitRate` when constructing `RoutingSnapshot` from `InstanceSnapshot` (PR13 added trace recording that reads from `RoutingSnapshot` — the field must be populated there too)

**PR13 interaction note (post-merge):** PR13 added `CandidateScore` in `sim/trace/record.go` which captures `KVUtilization` and `FreeKVBlocks` from `RoutingSnapshot`. When PR12 adds `CacheHitRate` to `RoutingSnapshot`, consider adding it to `CandidateScore` too for trace completeness. This is optional (not a correctness issue) and can be done as a follow-up.

**Tracking counters added to `TieredKVCache`:**
- `cpuHitCount` / `cpuMissCount` — for CacheHitRate (combined with GPU counters)
- `thrashingCount` / `offloadCount` — for KVThrashingRate

**PreemptionRate tracking — NEW counter needed:**
The existing code only has a per-step `preemptionHappened` bool (`simulator.go:128`), not a counter. Add:
```go
// In sim/metrics.go (Metrics struct):
PreemptionCount int64  // total number of preemption events
```
Increment in `simulator.go:preempt()` function (line 352, alongside existing `sim.preemptionHappened = true`):
```go
sim.Metrics.PreemptionCount++
```
Formula for `RawMetrics.PreemptionRate`:
```go
PreemptionRate = float64(sum(instance.Metrics.PreemptionCount)) / float64(sum(instance.Metrics.CompletedRequests))
```

**For single-tier (`KVCacheState`):** `CacheHitRate` is computed from existing `GetCachedBlocks` calls. Add a hit/miss counter to `KVCacheState`:
```go
CacheHits   int64  // blocks found via prefix cache
CacheMisses int64  // blocks not found, allocated fresh
```
Increment `CacheHits` in `GetCachedBlocks` for each matched block, `CacheMisses` for each block request that found no match in `AllocateKVBlocks`.

---

## Decision 7: `SimConfig` Extension

New fields:
```go
type SimConfig struct {
    // ... existing fields ...
    KVCPUBlocks       int64    // CPU tier capacity (0 = no CPU tier, single-tier mode)
    KVOffloadThreshold float64 // GPU utilization threshold for offload (default 0.9)
    KVTransferBandwidth float64 // blocks/tick transfer rate
    KVTransferBaseLatency int64 // base latency per transfer operation (ticks)
}
```

**Factory logic:**
```go
func NewKVStore(cfg SimConfig) KVStore {
    gpu := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
    if cfg.KVCPUBlocks <= 0 {
        return gpu  // single-tier: backward compatible
    }
    return NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold,
        cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
}
```

---

## Decision 8: CLI Flags (in `cmd/root.go`)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--kv-cpu-blocks` | int64 | 0 | CPU tier KV blocks (0 = disabled, single-tier mode) |
| `--kv-offload-threshold` | float64 | 0.9 | GPU utilization trigger for offload |
| `--kv-transfer-bandwidth` | float64 | 100.0 | Blocks per tick transfer rate |

Note: The existing `--total-kv-blocks` flag (`cmd/root.go:369`) continues to control GPU tier capacity (`SimConfig.TotalKVBlocks`). No rename or alias needed.

---

## Decision 9: Task Ordering for Micro-Plan

Recommended task sequence (6-8 tasks):

1. **KVStore interface + accessor methods on KVCacheState** — Pure refactor, no behavior change. All existing tests must pass.
2. **Simulator uses KVStore interface** — Change `Simulator.KVCache` type, update field accesses in `simulator.go` and `instance.go`. All existing tests must pass.
3. **Cache hit/miss counters on KVCacheState** — Add counters, wire into `GetCachedBlocks` and `AllocateKVBlocks`. Foundation for CacheHitRate metric.
4. **TieredKVCache core** — `cpuTier` struct, `TieredKVCache` struct, delegation for `AllocateKVBlocks`/`GetCachedBlocks`/`ReleaseKVBlocks`. Unit test: offload trigger.
5. **Transfer latency model** — `PendingTransferLatency()`, reload-on-CPU-hit logic, latency formula. Unit test: reload adds latency.
6. **NewKVStore factory + SimConfig extension** — Factory function, `KVCPUBlocks` config field. Integration test: `--kv-cpu-blocks 0` produces identical output.
7. **CLI flags + metrics** — Wire `CacheHitRate`, `PreemptionRate`, `KVThrashingRate` into `RawMetrics`. CLI flags in `cmd/root.go`. End-to-end test.
8. **Documentation** — CLAUDE.md, README, example YAML.

Tasks 1-2 are pure refactoring (zero behavior change). Tasks 3-5 build the feature incrementally. Tasks 6-7 wire it together. Task 8 documents.

---

## Summary of Deviations from Macro Plan

| Macro Plan Says | This Pre-Design Does | Reason |
|-----------------|----------------------|--------|
| New files in `sim/kv/tiered.go`, `sim/kv/transfer.go` | New files in `sim/kv_store.go`, `sim/kvcache_tiered.go` (same package) | Import cycle avoidance; `sim/kv/` deferred to PR14 |
| `KVTransfer` events added to event queue | Synchronous transfer latency in step loop | Simpler (~100 LOC saved), equivalent fidelity for single-instance; event-based needed only for P/D cross-instance transfer (PR14) |
| ~30 LOC modification to `sim/kvcache.go` | ~15 LOC modification (3 accessor methods + 2 counter fields) | Interface introduction moves most changes to new file |
| `CacheHitRate` on `InstanceSnapshot` | `CacheHitRate` on `RoutingSnapshot` | `RoutingSnapshot` is the type used by routing policies; `InstanceSnapshot` is the cluster-internal type |

---

## How to Use This Document

The macro plan's PR12 entry contains a `**Pre-Design**` row marked `REQUIRED READING` that points to this file. When the `writing-plans` skill reads the macro plan during Phase 0, it will see this directive and read the pre-design automatically. **No manual `@` reference is needed.**

Standard invocation:
```
/superpowers:writing-plans for PR12 in @docs/plans/pr12-tiered-kv-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md
```

The planner should treat all 9 decisions as **resolved** — not re-derive them. The micro-plan's deviation log should reference this document for justification.
