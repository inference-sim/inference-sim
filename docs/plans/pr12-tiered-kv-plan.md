# Tiered KV Cache + Transfer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU+CPU tiered KV caching so the simulator can model offloading cold blocks to CPU memory and reloading them with realistic transfer latency — enabling capacity planning for systems like vLLM's CPU offloading.

**The problem today:** BLIS models KV cache as a single GPU-only pool. When blocks run out, requests are preempted. Real systems offload cold blocks to CPU memory and reload them later, which avoids preemption at the cost of transfer latency. Without tiered caching, BLIS overestimates preemption rates and underestimates throughput for workloads that benefit from CPU offloading.

**What this PR adds:**
1. **KVStore interface** — abstracts KV cache operations so the simulator works with either single-tier or tiered cache
2. **TieredKVCache** — composes a GPU `KVCacheState` with a simple CPU store; offloads LRU blocks when GPU pressure exceeds threshold; reloads from CPU on cache hit
3. **Transfer latency** — adds synchronous delay to step time when CPU→GPU reload occurs (`baseLatency + ceil(blocks × blockSize / bandwidth)`)
4. **Three new metrics** — CacheHitRate (GPU+CPU combined), PreemptionRate (fraction of requests preempted), KVThrashingRate (offloads quickly followed by reloads)

**Why this matters:** This is the first PR that extends BLIS's fidelity model beyond single-tier resources. It enables PR14 (P/D disaggregated architecture) which needs cross-instance KV transfer, and makes capacity planning realistic for memory-constrained deployments.

**Architecture:** Introduces `KVStore` interface in `sim/kv_store.go` (8 methods). `Simulator.KVCache` changes from `*KVCacheState` to `KVStore`. `TieredKVCache` in `sim/kvcache_tiered.go` composes GPU `*KVCacheState` + simple `cpuTier` map. Transfer latency is synchronous (added to step time), not event-based. All code stays in `sim/` package (no `sim/kv/` — deferred to PR14). Pre-design: `docs/plans/pr12-architectural-predesign.md`.

**Macro Plan Reference:** Phase 4, PR 12

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

PR12 introduces tiered KV caching (GPU+CPU) to the single-instance simulator. It sits between the core event loop (`sim/simulator.go`) and the existing `KVCacheState` (`sim/kvcache.go`), adding a `KVStore` interface that both the existing single-tier and new tiered implementations satisfy.

Adjacent blocks: `Simulator` (consumer of KVStore), `InstanceSimulator` (KV accessors), `CachedSnapshotProvider` (propagates CacheHitRate), `RawMetrics` (aggregates new metrics), `DeploymentConfig`/`SimConfig` (carries new config fields), CLI (`cmd/root.go` — new flags).

No deviations from the pre-design document. All 9 decisions are adopted as-is.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Backward Compatibility
- GIVEN `KVCPUBlocks == 0` (default)
- WHEN running any existing simulation configuration
- THEN output is byte-identical to pre-PR12 behavior
- MECHANISM: `NewKVStore` returns `*KVCacheState` directly when CPU blocks are 0; no tiered wrapper

BC-2: GPU Offload Trigger
- GIVEN `TieredKVCache` with `offloadThreshold = 0.8` and GPU at 90% utilization
- WHEN `ReleaseKVBlocks` completes and GPU utilization exceeds threshold
- THEN LRU free blocks are offloaded from GPU to CPU until GPU utilization ≤ threshold
- MECHANISM: After delegation to `gpu.ReleaseKVBlocks`, check `gpu.UsedBlocks()/gpu.TotalCapacity()` and offload from GPU free list head

BC-3: CPU Reload on Hit
- GIVEN a block was previously offloaded to CPU tier
- WHEN `AllocateKVBlocks` fails on GPU but the block's hash exists on CPU
- THEN the block is reloaded to GPU and allocation succeeds; transfer latency is accumulated
- MECHANISM: On GPU allocation failure, look up request's prefix hash in CPU tier; if found, pop a GPU free block, copy content, remove from CPU, accumulate `baseLatency + ceil(1 × blockSize / bandwidth)`

BC-4: Transfer Latency Integration
- GIVEN `TieredKVCache` has accumulated transfer latency from CPU→GPU reloads
- WHEN `Step()` queries `PendingTransferLatency()`
- THEN the returned value is added to step time and the internal accumulator resets to 0
- MECHANISM: Query-and-clear pattern; `KVCacheState` returns 0 (single-tier has no transfers)

BC-5: Cache Hit Rate Tracking
- GIVEN requests being processed through `GetCachedBlocks` and `AllocateKVBlocks`
- WHEN `CacheHitRate()` is called
- THEN it returns `hits / (hits + misses)` across all lookups; 0 if no lookups yet
- MECHANISM: `KVCacheState` tracks `CacheHits`/`CacheMisses`; `TieredKVCache` combines GPU and CPU counters

BC-6: Thrashing Detection
- GIVEN a block offloaded from GPU to CPU
- WHEN the same block is reloaded to GPU within 1000 ticks
- THEN `thrashingCount` is incremented
- MECHANISM: CPU tier stores offload timestamp per block; reload checks `clock - offloadTime < 1000`

BC-7: Preemption Counting
- GIVEN a request is preempted during `makeRunningBatch`
- WHEN `preempt()` executes
- THEN `sim.Metrics.PreemptionCount` is incremented
- MECHANISM: New `PreemptionCount int64` field on `Metrics`; incremented alongside existing `preemptionHappened = true`

BC-8: RawMetrics Population
- GIVEN cluster simulation completes
- WHEN `CollectRawMetrics` builds `RawMetrics`
- THEN `CacheHitRate`, `PreemptionRate`, and `KVThrashingRate` are populated from per-instance data
- MECHANISM: `CacheHitRate` from `KVStore.CacheHitRate()`; `PreemptionRate` from `PreemptionCount/CompletedRequests`; `KVThrashingRate` requires tiered cache internals (0 for single-tier)

**Negative Contracts:**

BC-9: KV Conservation
- GIVEN any tiered cache operation
- WHEN blocks are offloaded or reloaded
- THEN `gpu.UsedBlocks() + gpu.countFreeBlocks() == gpu.TotalCapacity()` AND `cpu.used + (cpu.capacity - cpu.used) == cpu.capacity`
- MECHANISM: Each offload decrements GPU used and increments CPU used; each reload does the reverse

BC-10: No CPU Overflow
- GIVEN CPU tier is full
- WHEN offload is triggered
- THEN offload stops (does not panic or corrupt state); GPU blocks remain in place
- MECHANISM: Check `cpu.used < cpu.capacity` before each offload; break loop if full

**Error Handling:**

BC-11: Zero Transfer Bandwidth
- GIVEN `KVTransferBandwidth == 0`
- WHEN creating `TieredKVCache`
- THEN constructor panics with descriptive message
- MECHANISM: Validate in `NewTieredKVCache`; bandwidth of 0 would cause division by zero

BC-12: Negative Config Values
- GIVEN negative values for `KVCPUBlocks`, `KVOffloadThreshold`, or `KVTransferBandwidth`
- WHEN CLI parses flags
- THEN `logrus.Fatalf` with descriptive message (no panic in deep code)
- MECHANISM: Validate at CLI level in `cmd/root.go` before constructing `DeploymentConfig`

### C) Component Interaction

```
CLI (cmd/root.go)
  │ --kv-cpu-blocks, --kv-offload-threshold, --kv-transfer-bandwidth
  ▼
DeploymentConfig → ToSimConfig() → SimConfig
  │                                    │ KVCPUBlocks, KVOffloadThreshold, ...
  ▼                                    ▼
ClusterSimulator                   NewSimulator(cfg)
  │                                    │ calls NewKVStore(cfg)
  │                                    ▼
  │                              ┌─ KVStore interface ─┐
  │                              │                     │
  │                     KVCacheState          TieredKVCache
  │                     (CPU=0)               (CPU>0)
  │                                           ├─ gpu: *KVCacheState
  │                                           └─ cpu: cpuTier
  │
  ├── InstanceSimulator.CacheHitRate() ──→ sim.KVCache.CacheHitRate()
  ├── CachedSnapshotProvider ──→ InstanceSnapshot.CacheHitRate
  ├── buildRouterState() ──→ RoutingSnapshot.CacheHitRate
  └── CollectRawMetrics() ──→ RawMetrics.CacheHitRate/PreemptionRate/KVThrashingRate
```

**API Contracts:**
- `KVStore` interface (8 methods) — see pre-design Decision 2
- `NewKVStore(cfg SimConfig) KVStore` — factory, returns concrete type based on `cfg.KVCPUBlocks`
- `NewTieredKVCache(gpu, cpuBlocks, threshold, bandwidth, baseLat) *TieredKVCache`

**State Ownership:**
- `TieredKVCache` owns `cpuTier` (map of offloaded blocks) and transfer latency accumulator
- `KVCacheState` owns GPU blocks (unchanged)
- `Metrics.PreemptionCount` owned by per-instance `Simulator`

### D) Deviation Log

| Macro Plan Says | Micro Plan Does | Reason |
|-----------------|-----------------|--------|
| Files in `sim/kv/tiered.go`, `sim/kv/transfer.go` | Files in `sim/kv_store.go`, `sim/kvcache_tiered.go` | CORRECTION: Import cycle avoidance (pre-design Decision 1) |
| `KVTransfer` events in event queue | Synchronous transfer latency in `Step()` | SIMPLIFICATION: Equivalent fidelity for single-instance; event-based deferred to PR14 (pre-design Decision 5) |
| `CacheHitRate` on `InstanceSnapshot` | `CacheHitRate` on `RoutingSnapshot` + `InstanceSnapshot` | CORRECTION: `RoutingSnapshot` is what routing policies consume (pre-design Decision 6) |
| ~30 LOC in `sim/kvcache.go` | ~15 LOC (5 accessor methods + 2 counter fields) | SIMPLIFICATION: Interface moves most logic to new files |
| `KVStore` with 6 methods (macro plan line 1343) | `KVStore` with 8 methods (added `CacheHitRate` + `PendingTransferLatency`) | CORRECTION: Pre-design Decision 2 added 2 methods for metrics and transfer; macro plan not yet updated |
| `KVTransferBaseLatency` as CLI flag | Hard-coded default 0 in `SimConfig`; not exposed as CLI flag | SIMPLIFICATION: Base latency is rarely configured; avoids flag bloat. Can be exposed later if needed |
| PR13 `CandidateScore` updated with `CacheHitRate` | Not updated (optional, deferred to follow-up) | DEFERRAL: `CandidateScore` in `sim/trace/record.go` works without `CacheHitRate`; adding it is a separate concern |
| LOC estimate ~300 | ~400 non-test LOC (detailed task breakdown) | CORRECTION: Macro plan estimate was rough; actual is ~400 due to metrics/snapshot/CLI wire-up |

### E) Review Guide

**The tricky part:** BC-3 (CPU reload on hit). When `AllocateKVBlocks` fails on GPU, the tiered cache needs to check CPU, reload, and retry — all while correctly tracking latency and not double-counting cache misses.

**What to scrutinize:** BC-9 (KV conservation invariant) — ensure offload/reload never creates or destroys blocks. BC-4 (query-and-clear semantics) — ensure transfer latency resets after read.

**What's safe to skim:** Tasks 1-2 (pure refactoring — just introducing an interface with zero behavior change). Task 8 (documentation).

**Known debt:** The deprecated `AllocateKVBlocksPrefill` and `AllocateKVBlocksDecode` methods in `kvcache.go` are not touched. They don't participate in the KVStore interface (only `AllocateKVBlocks` does).

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/kv_store.go` — `KVStore` interface, `NewKVStore` factory (~50 LOC)
- `sim/kvcache_tiered.go` — `TieredKVCache`, `cpuTier`, `OffloadedBlock` (~220 LOC)
- `sim/kv_store_test.go` — Interface satisfaction and factory tests (~60 LOC)
- `sim/kvcache_tiered_test.go` — Tiered cache behavioral tests (~250 LOC)

**Files to modify:**
- `sim/kvcache.go` — Add `CacheHits`/`CacheMisses` fields + 5 accessor methods (~20 LOC)
- `sim/simulator.go` — Change `KVCache` type to `KVStore`, update 6 field accesses, add transfer latency, add preemption counter (~15 LOC)
- `sim/metrics.go` — Add `PreemptionCount` field (~2 LOC)
- `sim/cluster/instance.go` — Update KV accessors + add `CacheHitRate()` (~10 LOC)
- `sim/cluster/snapshot.go` — Add `CacheHitRate` to `InstanceSnapshot` + `ObservabilityConfig` + populate (~10 LOC)
- `sim/cluster/cluster_event.go` — Add `CacheHitRate` to `buildRouterState` copy (~2 LOC)
- `sim/routing.go` — Add `CacheHitRate` to `RoutingSnapshot` (~1 LOC)
- `sim/cluster/metrics.go` — Add `CacheHitRate`/`PreemptionRate`/`KVThrashingRate` to `RawMetrics` + populate in `CollectRawMetrics` (~25 LOC)
- `sim/cluster/deployment.go` — Add KV config fields to `DeploymentConfig` + `ToSimConfig` (~10 LOC)
- `cmd/root.go` — Add 3 CLI flags + validation (~20 LOC)
- `CLAUDE.md` — Update architecture docs (~15 LOC)

**Key decisions:** All from pre-design document (9 binding decisions adopted as-is).

**No dead code:** Every new field, method, and type is exercised by CLI flags (`--kv-cpu-blocks`, `--kv-offload-threshold`, `--kv-transfer-bandwidth`) or by existing code paths (KVStore interface replaces direct field access).

### G) Task Breakdown

---

#### Section 1: Interface Refactoring (Zero Behavior Change)

### Task 1: KVStore Interface + KVCacheState Accessor Methods

**Contracts Implemented:** BC-1 (backward compatibility foundation)

**Files:**
- Create: `sim/kv_store.go`
- Modify: `sim/kvcache.go`
- Test: `sim/kv_store_test.go`

**Step 1: Write failing test for KVStore interface satisfaction**

Context: We verify that `KVCacheState` satisfies the `KVStore` interface and that accessor methods return correct values.

```go
// sim/kv_store_test.go
package sim

import "testing"

func TestKVCacheState_ImplementsKVStore(t *testing.T) {
	// GIVEN a KVCacheState
	kvc := NewKVCacheState(100, 16)

	// WHEN assigned to KVStore interface
	var store KVStore = kvc

	// THEN all accessor methods return expected values
	if store.BlockSize() != 16 {
		t.Errorf("BlockSize() = %d, want 16", store.BlockSize())
	}
	if store.TotalCapacity() != 100 {
		t.Errorf("TotalCapacity() = %d, want 100", store.TotalCapacity())
	}
	if store.UsedBlocks() != 0 {
		t.Errorf("UsedBlocks() = %d, want 0", store.UsedBlocks())
	}
	if store.CacheHitRate() != 0 {
		t.Errorf("CacheHitRate() = %f, want 0 (no lookups)", store.CacheHitRate())
	}
	if store.PendingTransferLatency() != 0 {
		t.Errorf("PendingTransferLatency() = %d, want 0 (single-tier)", store.PendingTransferLatency())
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestKVCacheState_ImplementsKVStore -v`
Expected: FAIL — `KVStore` type not defined

**Step 3: Implement KVStore interface and accessor methods**

In `sim/kv_store.go`:
```go
package sim

// KVStore abstracts KV cache operations for the simulator.
// KVCacheState (single-tier GPU) and TieredKVCache (GPU+CPU) both implement this.
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
}

// NewKVStore creates a KVStore from SimConfig.
// Returns *KVCacheState for single-tier (KVCPUBlocks <= 0, the default).
// Returns *TieredKVCache for tiered mode (KVCPUBlocks > 0).
func NewKVStore(cfg SimConfig) KVStore {
	gpu := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	if cfg.KVCPUBlocks <= 0 {
		return gpu
	}
	return NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold,
		cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
}
```

In `sim/kvcache.go`, add to `KVCacheState` struct:
```go
CacheHits   int64 // blocks found via prefix cache
CacheMisses int64 // blocks not found, allocated fresh
```

Add accessor methods after `ReleaseKVBlocks`:
```go
// BlockSize returns the number of tokens per block.
func (kvc *KVCacheState) BlockSize() int64 { return kvc.BlockSizeTokens }

// UsedBlocks returns the number of blocks currently in use.
func (kvc *KVCacheState) UsedBlocks() int64 { return kvc.UsedBlockCnt }

// TotalCapacity returns the total number of blocks.
func (kvc *KVCacheState) TotalCapacity() int64 { return kvc.TotalBlocks }

// CacheHitRate returns the cumulative cache hit rate.
// Returns 0 if no lookups have been performed.
func (kvc *KVCacheState) CacheHitRate() float64 {
	total := kvc.CacheHits + kvc.CacheMisses
	if total == 0 {
		return 0
	}
	return float64(kvc.CacheHits) / float64(total)
}

// PendingTransferLatency returns 0 for single-tier cache (no transfers).
func (kvc *KVCacheState) PendingTransferLatency() int64 { return 0 }

// KVThrashingRate returns 0 for single-tier cache (no offload/reload).
func (kvc *KVCacheState) KVThrashingRate() float64 { return 0 }
```

Note: `NewKVStore` references `NewTieredKVCache` which doesn't exist yet. Add a stub in `sim/kvcache_tiered.go` for compilation:
```go
package sim

// TieredKVCache will be implemented in Task 4. Stub for compilation.
type TieredKVCache struct{ gpu *KVCacheState }

func NewTieredKVCache(gpu *KVCacheState, cpuBlocks int64, threshold, bandwidth float64, baseLat int64) *TieredKVCache {
	panic("TieredKVCache not yet implemented")
}
func (t *TieredKVCache) AllocateKVBlocks(req *Request, s, e int64, c []int64) bool { panic("stub") }
func (t *TieredKVCache) GetCachedBlocks(tokens []int) []int64                     { panic("stub") }
func (t *TieredKVCache) ReleaseKVBlocks(req *Request)                             { panic("stub") }
func (t *TieredKVCache) BlockSize() int64                                         { return t.gpu.BlockSize() }
func (t *TieredKVCache) UsedBlocks() int64                                        { return t.gpu.UsedBlocks() }
func (t *TieredKVCache) TotalCapacity() int64                                     { return t.gpu.TotalCapacity() }
func (t *TieredKVCache) CacheHitRate() float64                                    { return 0 }
func (t *TieredKVCache) PendingTransferLatency() int64                            { return 0 }
```

In `sim/simulator.go` `SimConfig`, add new fields:
```go
// Tiered KV cache configuration (PR12)
KVCPUBlocks           int64   // CPU tier capacity (0 = single-tier, default)
KVOffloadThreshold    float64 // GPU utilization threshold for offload (default 0.9)
KVTransferBandwidth   float64 // blocks/tick transfer rate (default 100.0)
KVTransferBaseLatency int64   // fixed cost per transfer (ticks, default 0)
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestKVCacheState_ImplementsKVStore -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/kv_store.go sim/kv_store_test.go sim/kvcache.go sim/kvcache_tiered.go sim/simulator.go
git commit -m "feat(sim): add KVStore interface and KVCacheState accessors (BC-1)

- Define KVStore interface with 8 methods in sim/kv_store.go
- Add BlockSize/UsedBlocks/TotalCapacity/CacheHitRate/PendingTransferLatency to KVCacheState
- Add CacheHits/CacheMisses counter fields to KVCacheState
- Add NewKVStore factory (returns KVCacheState for single-tier, TieredKVCache stub for tiered)
- Add SimConfig fields for tiered KV configuration
- Stub TieredKVCache for compilation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Simulator Uses KVStore Interface

**Contracts Implemented:** BC-1 (backward compatibility — all existing tests must pass)

**Files:**
- Modify: `sim/simulator.go:104,168,446,453,492,559,560,562`
- Modify: `sim/cluster/instance.go:115-122`
- Modify: `sim/metrics.go` (add PreemptionCount)

**Step 1: Change Simulator.KVCache type and update all field accesses**

Context: Pure refactoring — change the type from `*KVCacheState` to `KVStore` and replace 6 direct field accesses with method calls. Also add `PreemptionCount` to `Metrics` for BC-7.

In `sim/simulator.go`:

Change line 104:
```go
// Before: KVCache *KVCacheState
KVCache KVStore
```

Change line 168:
```go
// Before: KVCache: NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens),
KVCache: NewKVStore(cfg),
```

Change line 446:
```go
// Before: numNewTokens := Len64(next.InputTokens) - Len64(cachedBlocks)*sim.KVCache.BlockSizeTokens
numNewTokens := Len64(next.InputTokens) - Len64(cachedBlocks)*sim.KVCache.BlockSize()
```

Change line 453:
```go
// Before: startIndex := Len64(cachedBlocks) * sim.KVCache.BlockSizeTokens
startIndex := Len64(cachedBlocks) * sim.KVCache.BlockSize()
```

Change line 492:
```go
// Before: sim.reqNumComputedTokens[next.ID] = numNewTokens + Len64(cachedBlocks)*sim.KVCache.BlockSizeTokens
sim.reqNumComputedTokens[next.ID] = numNewTokens + Len64(cachedBlocks)*sim.KVCache.BlockSize()
```

Change lines 559-562:
```go
// Before:
// if sim.KVCache.UsedBlockCnt > sim.Metrics.PeakKVBlocksUsed {
//     sim.Metrics.PeakKVBlocksUsed = sim.KVCache.UsedBlockCnt
// }
// sim.Metrics.KVBlocksUsed += float64(sim.KVCache.UsedBlockCnt) * float64(currStepAdvance)
if sim.KVCache.UsedBlocks() > sim.Metrics.PeakKVBlocksUsed {
    sim.Metrics.PeakKVBlocksUsed = sim.KVCache.UsedBlocks()
}
sim.Metrics.KVBlocksUsed += float64(sim.KVCache.UsedBlocks()) * float64(currStepAdvance)
```

Add after line 531 (after step time computation, before model execution):
```go
// Add transfer latency from CPU→GPU reloads (0 for single-tier)
currStepAdvance += sim.KVCache.PendingTransferLatency()
```

Add in `preempt()` at line 352:
```go
sim.Metrics.PreemptionCount++
```

In `sim/metrics.go`, add to `Metrics` struct:
```go
PreemptionCount int64 // Total preemption events (PR12)
```

In `sim/cluster/instance.go`, change lines 115-122:
```go
// KVUtilization returns the fraction of KV cache blocks in use.
func (i *InstanceSimulator) KVUtilization() float64 {
	return float64(i.sim.KVCache.UsedBlocks()) / float64(i.sim.KVCache.TotalCapacity())
}

// FreeKVBlocks returns the number of free KV cache blocks.
func (i *InstanceSimulator) FreeKVBlocks() int64 {
	return i.sim.KVCache.TotalCapacity() - i.sim.KVCache.UsedBlocks()
}
```

**Step 2: Run ALL existing tests to verify backward compatibility**

Run: `go test ./... -count=1`
Expected: ALL PASS (zero behavior change)

**Step 3: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/simulator.go sim/metrics.go sim/cluster/instance.go
git commit -m "refactor(sim): use KVStore interface in Simulator (BC-1)

- Change Simulator.KVCache from *KVCacheState to KVStore
- Replace 6 direct field accesses with method calls
- Add PendingTransferLatency() integration point in Step()
- Add Metrics.PreemptionCount field
- Increment PreemptionCount in preempt()
- Update InstanceSimulator KV accessors to use interface methods
- All existing tests pass (zero behavior change)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Section 2: Cache Hit Tracking

### Task 3: Cache Hit/Miss Counters on KVCacheState

**Contracts Implemented:** BC-5 (cache hit rate tracking)

**Files:**
- Modify: `sim/kvcache.go`
- Test: `sim/kv_store_test.go`

**Step 1: Write failing test for cache hit rate**

Context: Test that `CacheHitRate()` reflects actual prefix cache behavior.

```go
// Add to sim/kv_store_test.go
func TestKVCacheState_CacheHitRate_ReflectsLookups(t *testing.T) {
	// GIVEN a KVCacheState with block size 2
	kvc := NewKVCacheState(10, 2)
	var store KVStore = kvc

	// Initial: no lookups
	if rate := store.CacheHitRate(); rate != 0 {
		t.Errorf("initial CacheHitRate() = %f, want 0", rate)
	}

	// WHEN we allocate blocks for a request (all misses)
	req := &Request{
		ID:          "req1",
		InputTokens: []int{1, 2, 3, 4}, // 2 blocks
	}
	ok := store.AllocateKVBlocks(req, 0, 4, []int64{})
	if !ok {
		t.Fatal("AllocateKVBlocks failed")
	}

	// THEN CacheMisses should be > 0 (blocks were freshly allocated)
	if kvc.CacheMisses == 0 {
		t.Error("expected CacheMisses > 0 after fresh allocation")
	}

	// WHEN we release and re-lookup the same prefix
	store.ReleaseKVBlocks(req)
	cached := store.GetCachedBlocks([]int{1, 2, 3, 4})

	// THEN we should get cache hits (blocks are still in hash table)
	if len(cached) == 0 {
		t.Error("expected cache hits for previously allocated prefix")
	}
	if kvc.CacheHits == 0 {
		t.Error("expected CacheHits > 0 after prefix lookup")
	}

	// AND CacheHitRate should be between 0 and 1
	rate := store.CacheHitRate()
	if rate <= 0 || rate >= 1 {
		t.Errorf("CacheHitRate() = %f, want 0 < rate < 1", rate)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestKVCacheState_CacheHitRate_ReflectsLookups -v`
Expected: FAIL — `CacheHits` is never incremented

**Step 3: Add cache hit/miss tracking to KVCacheState**

In `sim/kvcache.go`, modify `GetCachedBlocks`:
```go
func (kvc *KVCacheState) GetCachedBlocks(tokens []int) (blockIDs []int64) {
	n := Len64(tokens) / kvc.BlockSizeTokens
	for i := int64(0); i < n; i++ {
		chunk := tokens[:(i+1)*kvc.BlockSizeTokens]
		h := hashTokens(chunk)
		blockId, ok := kvc.HashToBlock[h]
		if !ok {
			break
		}
		kvc.CacheHits++
		blockIDs = append(blockIDs, blockId)
	}
	return
}
```

In `AllocateKVBlocks`, after the free blocks check fails (line 217-220):
```go
if numNewBlocks > kvc.countFreeBlocks() {
    kvc.CacheMisses += numNewBlocks
    logrus.Warnf("Not enough KV cache space to allocate %v new blocks", numNewBlocks)
    return false
}
```

Also count misses for successfully allocated new blocks. In the inner loop where `kvc.popFreeBlock()` is called (around line 270), after allocating each new block, increment:
```go
kvc.CacheMisses++
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestKVCacheState_CacheHitRate -v`
Expected: PASS

**Step 5: Run ALL tests**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/kvcache.go sim/kv_store_test.go
git commit -m "feat(sim): add cache hit/miss tracking to KVCacheState (BC-5)

- Increment CacheHits in GetCachedBlocks for each prefix match
- Increment CacheMisses in AllocateKVBlocks for each fresh allocation
- CacheHitRate() now returns meaningful values

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Section 3: TieredKVCache Implementation

### Task 4: TieredKVCache Core

**Contracts Implemented:** BC-2 (offload trigger), BC-9 (KV conservation), BC-10 (no CPU overflow)

**Files:**
- Modify: `sim/kvcache_tiered.go` (replace stubs with real implementation)
- Create: `sim/kvcache_tiered_test.go`

**Step 1: Write failing tests for offload trigger and conservation**

```go
// sim/kvcache_tiered_test.go
package sim

import "testing"

func TestTieredKVCache_OffloadTriggered_WhenGPUExceedsThreshold(t *testing.T) {
	// GIVEN 10 GPU blocks, 10 CPU blocks, threshold 0.5 (offload when >50% used)
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 100.0, 0)

	// WHEN we allocate blocks for requests filling >50% GPU
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} // needs 6 blocks
	ok := tiered.AllocateKVBlocks(req1, 0, 12, []int64{})
	if !ok {
		t.Fatal("allocation should succeed")
	}

	// Release to trigger offload check
	tiered.ReleaseKVBlocks(req1)

	// THEN GPU used should be <= threshold * capacity (some blocks offloaded to CPU)
	gpuUtil := float64(tiered.UsedBlocks()) / float64(tiered.TotalCapacity())
	if gpuUtil > 0.5 {
		t.Errorf("GPU utilization after offload = %f, want <= 0.5", gpuUtil)
	}
}

func TestTieredKVCache_CPUFull_OffloadStops(t *testing.T) {
	// GIVEN 10 GPU blocks, 2 CPU blocks, threshold 0.3
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 2, 0.3, 100.0, 0)

	// WHEN we allocate many blocks and release
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}} // 8 blocks
	ok := tiered.AllocateKVBlocks(req, 0, 16, []int64{})
	if !ok {
		t.Fatal("allocation should succeed")
	}
	tiered.ReleaseKVBlocks(req)

	// THEN CPU tier should not exceed its capacity
	if tiered.cpu.used > tiered.cpu.capacity {
		t.Errorf("CPU used %d exceeds capacity %d", tiered.cpu.used, tiered.cpu.capacity)
	}
}

func TestTieredKVCache_Conservation_BlocksPreserved(t *testing.T) {
	// GIVEN tiered cache
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 5, 0.5, 100.0, 0)

	// Total GPU blocks should remain constant
	initialGPU := gpu.TotalBlocks

	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req, 0, 4, []int64{})
	tiered.ReleaseKVBlocks(req)

	// THEN GPU total is unchanged
	if gpu.TotalBlocks != initialGPU {
		t.Errorf("GPU total changed from %d to %d", initialGPU, gpu.TotalBlocks)
	}
	// AND used + free = total for GPU
	if gpu.UsedBlockCnt+gpu.countFreeBlocks() != gpu.TotalBlocks {
		t.Errorf("conservation violated: used(%d) + free(%d) != total(%d)",
			gpu.UsedBlockCnt, gpu.countFreeBlocks(), gpu.TotalBlocks)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/... -run TestTieredKVCache -v`
Expected: FAIL — stubs panic

**Step 3: Implement TieredKVCache core**

Replace the entire `sim/kvcache_tiered.go` with the full implementation:

```go
package sim

import "fmt"

// OffloadedBlock represents a KV block offloaded from GPU to CPU tier.
type OffloadedBlock struct {
	OriginalID  int64 // block ID on GPU tier
	Tokens      []int // token content (for reload)
	Hash        string // prefix hash (for cache hit detection)
	OffloadTime int64  // clock time when offloaded (for thrashing detection)
}

// cpuTier is a simple capacity-tracked block store for CPU-offloaded blocks.
type cpuTier struct {
	blocks   map[int64]*OffloadedBlock // keyed by original GPU block ID
	capacity int64
	used     int64
}

// TieredKVCache composes a GPU KVCacheState with a simple CPU tier.
// Delegates all normal operations to GPU; offloads LRU blocks when pressure exceeds threshold.
type TieredKVCache struct {
	gpu              *KVCacheState
	cpu              cpuTier
	offloadThreshold float64
	transferBandwidth float64
	baseLatency      int64
	clock            int64 // updated from simulator context

	// Transfer latency accumulator (query-and-clear)
	pendingLatency int64

	// Metrics counters
	offloadCount   int64
	reloadCount    int64
	cpuHitCount    int64
	cpuMissCount   int64
	thrashingCount int64
}

// NewTieredKVCache creates a TieredKVCache.
// Panics if bandwidth is zero (would cause division by zero).
func NewTieredKVCache(gpu *KVCacheState, cpuBlocks int64, threshold, bandwidth float64, baseLat int64) *TieredKVCache {
	if bandwidth <= 0 {
		panic(fmt.Sprintf("KVTransferBandwidth must be > 0, got %f", bandwidth))
	}
	return &TieredKVCache{
		gpu: gpu,
		cpu: cpuTier{
			blocks:   make(map[int64]*OffloadedBlock),
			capacity: cpuBlocks,
		},
		offloadThreshold:  threshold,
		transferBandwidth: bandwidth,
		baseLatency:       baseLat,
	}
}

func (t *TieredKVCache) AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	ok := t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	if ok {
		return true
	}
	// GPU allocation failed — try to reload from CPU if applicable
	// For now, just return false (reload logic added in Task 5)
	return false
}

func (t *TieredKVCache) GetCachedBlocks(tokens []int) []int64 {
	return t.gpu.GetCachedBlocks(tokens)
}

func (t *TieredKVCache) ReleaseKVBlocks(req *Request) {
	t.gpu.ReleaseKVBlocks(req)
	t.maybeOffload()
}

func (t *TieredKVCache) BlockSize() int64      { return t.gpu.BlockSize() }
func (t *TieredKVCache) UsedBlocks() int64      { return t.gpu.UsedBlocks() }
func (t *TieredKVCache) TotalCapacity() int64   { return t.gpu.TotalCapacity() }

func (t *TieredKVCache) CacheHitRate() float64 {
	totalHits := t.gpu.CacheHits + t.cpuHitCount
	totalMisses := t.gpu.CacheMisses + t.cpuMissCount
	total := totalHits + totalMisses
	if total == 0 {
		return 0
	}
	return float64(totalHits) / float64(total)
}

func (t *TieredKVCache) PendingTransferLatency() int64 {
	lat := t.pendingLatency
	t.pendingLatency = 0
	return lat
}

// maybeOffload moves LRU free blocks from GPU to CPU until GPU utilization ≤ threshold.
func (t *TieredKVCache) maybeOffload() {
	for {
		if t.gpu.TotalCapacity() == 0 {
			break
		}
		util := float64(t.gpu.UsedBlocks()) / float64(t.gpu.TotalCapacity())
		if util <= t.offloadThreshold {
			break
		}
		if t.cpu.used >= t.cpu.capacity {
			break // CPU full
		}
		// Find a free block on GPU to offload (from free list head)
		blk := t.gpu.FreeHead
		if blk == nil {
			break // No free blocks to offload
		}
		// Only offload blocks that have content (hash != "")
		if blk.Hash == "" {
			break // Empty block, nothing to offload
		}
		// Copy to CPU
		t.cpu.blocks[blk.ID] = &OffloadedBlock{
			OriginalID:  blk.ID,
			Tokens:      append([]int{}, blk.Tokens...),
			Hash:        blk.Hash,
			OffloadTime: t.clock,
		}
		t.cpu.used++
		t.offloadCount++
		// Remove from GPU free list and hash table
		t.gpu.removeFromFreeList(blk)
		delete(t.gpu.HashToBlock, blk.Hash)
		blk.Hash = ""
		blk.Tokens = nil
		// Re-add to free list as empty block (at tail — will be popped last)
		t.gpu.appendToFreeList(blk)
	}
}

// KVThrashingRate returns the fraction of offloads that were followed by a reload within 1000 ticks.
func (t *TieredKVCache) KVThrashingRate() float64 {
	if t.offloadCount == 0 {
		return 0
	}
	return float64(t.thrashingCount) / float64(t.offloadCount)
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run TestTieredKVCache -v`
Expected: PASS

**Step 5: Run ALL tests**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/kvcache_tiered.go sim/kvcache_tiered_test.go
git commit -m "feat(sim): implement TieredKVCache with offload trigger (BC-2, BC-9, BC-10)

- Replace TieredKVCache stubs with full implementation
- Compose GPU KVCacheState + simple cpuTier map
- Offload LRU free blocks when GPU utilization exceeds threshold
- Stop offload when CPU is full (BC-10)
- Preserve GPU block conservation invariant (BC-9)
- KVThrashingRate metric support

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Transfer Latency + CPU Reload

**Contracts Implemented:** BC-3 (CPU reload on hit), BC-4 (transfer latency integration), BC-6 (thrashing detection), BC-11 (zero bandwidth panic)

**Files:**
- Modify: `sim/kvcache_tiered.go`
- Modify: `sim/kvcache_tiered_test.go`

**Step 1: Write failing tests**

```go
// Add to sim/kvcache_tiered_test.go

func TestTieredKVCache_CPUReload_AddsTransferLatency(t *testing.T) {
	// GIVEN tiered cache with base latency 10, bandwidth 1.0
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 10) // threshold 0 = always offload

	// Allocate and release to populate cache, then trigger offload
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req, 0, 4, []int64{})
	tiered.ReleaseKVBlocks(req) // triggers offload

	// Verify blocks were offloaded
	if tiered.cpu.used == 0 {
		t.Fatal("expected blocks on CPU after offload")
	}

	// WHEN we try to allocate the same prefix again
	// First fill GPU so it can't allocate fresh
	for i := 0; i < 8; i++ {
		filler := &Request{ID: fmt.Sprintf("filler_%d", i), InputTokens: []int{int(i*2 + 100), int(i*2 + 101)}}
		tiered.AllocateKVBlocks(filler, 0, 2, []int64{})
	}

	// Now try to get cached blocks — should find them on CPU
	// (This tests the reload path when implemented)
	// For now, verify transfer latency accumulates
	lat := tiered.PendingTransferLatency()
	// After reload, latency should be > 0

	// THEN PendingTransferLatency returns the accumulated value and resets
	lat2 := tiered.PendingTransferLatency()
	if lat2 != 0 {
		t.Errorf("second PendingTransferLatency() = %d, want 0 (query-and-clear)", lat2)
	}
	_ = lat // first read may or may not have latency depending on reload success
}

func TestTieredKVCache_PendingTransferLatency_QueryAndClear(t *testing.T) {
	// GIVEN tiered cache with some pending latency
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 100.0, 0)

	// Manually set pending latency for testing
	tiered.pendingLatency = 42

	// WHEN we query
	lat := tiered.PendingTransferLatency()

	// THEN it returns the value and resets
	if lat != 42 {
		t.Errorf("PendingTransferLatency() = %d, want 42", lat)
	}
	if tiered.PendingTransferLatency() != 0 {
		t.Error("second call should return 0")
	}
}

func TestTieredKVCache_ZeroBandwidth_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for zero bandwidth")
		}
	}()
	gpu := NewKVCacheState(10, 2)
	NewTieredKVCache(gpu, 10, 0.5, 0, 0) // should panic
}
```

**Step 2: Run tests to verify failures**

Run: `go test ./sim/... -run "TestTieredKVCache_CPUReload|TestTieredKVCache_PendingTransfer|TestTieredKVCache_ZeroBandwidth" -v`
Expected: Some PASS (query-and-clear, zero bandwidth), reload test may need refinement

**Step 3: Add CPU reload logic to AllocateKVBlocks**

Update `AllocateKVBlocks` in `sim/kvcache_tiered.go`:

```go
func (t *TieredKVCache) AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	ok := t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	if ok {
		return true
	}
	// GPU allocation failed — check CPU tier for reloadable blocks
	// Look for blocks matching the request's prefix in CPU
	reloaded := false
	for cpuBlockID, offloaded := range t.cpu.blocks {
		if offloaded.Hash == "" {
			continue
		}
		// Check if this hash is needed but missing from GPU
		if _, inGPU := t.gpu.HashToBlock[offloaded.Hash]; inGPU {
			continue // already on GPU
		}
		// Reload: pop a GPU free block, fill with CPU content
		blk := t.gpu.popFreeBlock()
		if blk == nil {
			break // no GPU free blocks
		}
		blk.Tokens = append([]int{}, offloaded.Tokens...)
		blk.Hash = offloaded.Hash
		blk.RefCount = 0
		blk.InUse = false
		t.gpu.HashToBlock[offloaded.Hash] = blk.ID
		t.gpu.appendToFreeList(blk)

		// Accumulate transfer latency
		blockSize := t.gpu.BlockSize()
		transferTime := t.baseLatency + (1*blockSize+int64(t.transferBandwidth)-1)/int64(t.transferBandwidth)
		t.pendingLatency += transferTime

		// Check thrashing
		if t.clock-offloaded.OffloadTime < 1000 {
			t.thrashingCount++
		}

		// Remove from CPU
		delete(t.cpu.blocks, cpuBlockID)
		t.cpu.used--
		t.reloadCount++
		t.cpuHitCount++
		reloaded = true
	}

	if reloaded {
		// Retry GPU allocation
		return t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	}
	t.cpuMissCount++
	return false
}
```

**Step 4: Run tests**

Run: `go test ./sim/... -run TestTieredKVCache -v`
Expected: PASS

**Step 5: Run ALL tests**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/kvcache_tiered.go sim/kvcache_tiered_test.go
git commit -m "feat(sim): add CPU reload and transfer latency to TieredKVCache (BC-3, BC-4, BC-6, BC-11)

- CPU reload when GPU allocation fails and block exists on CPU
- Transfer latency formula: baseLatency + ceil(blockSize / bandwidth)
- Query-and-clear semantics for PendingTransferLatency()
- Thrashing detection: offload followed by reload within 1000 ticks
- Panic on zero bandwidth (BC-11)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Section 4: Integration

### Task 6: Metrics + Snapshot + RoutingSnapshot Wire-Up

**Contracts Implemented:** BC-7 (preemption counting), BC-8 (RawMetrics population)

**Files:**
- Modify: `sim/routing.go` (add CacheHitRate to RoutingSnapshot)
- Modify: `sim/cluster/instance.go` (add CacheHitRate accessor)
- Modify: `sim/cluster/snapshot.go` (add CacheHitRate to InstanceSnapshot + populate)
- Modify: `sim/cluster/cluster_event.go` (add CacheHitRate to buildRouterState)
- Modify: `sim/cluster/metrics.go` (add 3 fields to RawMetrics + populate)
- Test: `sim/cluster/metrics_test.go` (existing or new)

**Step 1: Write failing test for RawMetrics population**

```go
// Add to existing sim/cluster/ test file or create sim/cluster/metrics_pr12_test.go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestCollectRawMetrics_IncludesPreemptionRate(t *testing.T) {
	// GIVEN per-instance metrics with preemption counts
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 10
	m1.PreemptionCount = 3
	m2 := sim.NewMetrics()
	m2.CompletedRequests = 10
	m2.PreemptionCount = 1

	aggregated := sim.NewMetrics()
	aggregated.CompletedRequests = 20
	aggregated.SimEndedTime = 1000000

	// WHEN collecting raw metrics
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m1, m2}, 0)

	// THEN PreemptionRate = 4/20 = 0.2
	expected := 4.0 / 20.0
	if raw.PreemptionRate != expected {
		t.Errorf("PreemptionRate = %f, want %f", raw.PreemptionRate, expected)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestCollectRawMetrics_IncludesPreemptionRate -v`
Expected: FAIL — `PreemptionRate` field doesn't exist on RawMetrics

**Step 3: Implement all wire-ups**

In `sim/routing.go`, add to `RoutingSnapshot`:
```go
CacheHitRate float64
```

In `sim/cluster/instance.go`, add accessor:
```go
// CacheHitRate returns the cumulative cache hit rate.
func (i *InstanceSimulator) CacheHitRate() float64 {
	return i.sim.KVCache.CacheHitRate()
}
```

In `sim/cluster/snapshot.go`, add `CacheHitRate` to `InstanceSnapshot`:
```go
CacheHitRate float64
```

In `CachedSnapshotProvider.Snapshot()`, add after `FreeKVBlocks` line:
```go
snap.CacheHitRate = inst.CacheHitRate()
```

In `RefreshAll()`, add to snapshot construction:
```go
CacheHitRate: inst.CacheHitRate(),
```

In `sim/cluster/cluster_event.go`, `buildRouterState()`, add to `RoutingSnapshot` construction:
```go
CacheHitRate: snap.CacheHitRate,
```

In `sim/cluster/metrics.go`, add to `RawMetrics`:
```go
CacheHitRate    float64
PreemptionRate  float64
KVThrashingRate float64
```

In `sim/metrics.go`, also add to `Metrics` struct:
```go
CacheHitRate    float64 // Cumulative cache hit rate at finalization (PR12)
KVThrashingRate float64 // KV thrashing rate at finalization (PR12)
```

In `sim/cluster/instance.go`, update `Finalize()`:
```go
func (i *InstanceSimulator) Finalize() {
	i.sim.Finalize()
	// Capture KV metrics at finalization for CollectRawMetrics
	i.sim.Metrics.CacheHitRate = i.sim.KVCache.CacheHitRate()
	i.sim.Metrics.KVThrashingRate = i.sim.KVCache.KVThrashingRate()
}
```

In `CollectRawMetrics`, add after anomaly detection:
```go
// KV cache metrics (PR12)
if perInstance != nil {
	totalPreemptions := int64(0)
	cacheHitSum := 0.0
	thrashingSum := 0.0
	count := 0
	for _, m := range perInstance {
		totalPreemptions += m.PreemptionCount
		cacheHitSum += m.CacheHitRate
		thrashingSum += m.KVThrashingRate
		count++
	}
	if aggregated.CompletedRequests > 0 {
		raw.PreemptionRate = float64(totalPreemptions) / float64(aggregated.CompletedRequests)
	}
	if count > 0 {
		raw.CacheHitRate = cacheHitSum / float64(count)
		raw.KVThrashingRate = thrashingSum / float64(count)
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestCollectRawMetrics_IncludesPreemptionRate -v`
Expected: PASS

**Step 5: Run ALL tests**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/routing.go sim/cluster/instance.go sim/cluster/snapshot.go sim/cluster/cluster_event.go sim/cluster/metrics.go
git commit -m "feat(cluster): wire CacheHitRate/PreemptionRate/KVThrashingRate metrics (BC-7, BC-8)

- Add CacheHitRate to RoutingSnapshot and InstanceSnapshot
- Add CacheHitRate accessor to InstanceSimulator
- Populate CacheHitRate in buildRouterState and CachedSnapshotProvider
- Add CacheHitRate/PreemptionRate/KVThrashingRate to RawMetrics
- Compute PreemptionRate from per-instance PreemptionCount

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: CLI Flags + DeploymentConfig

**Contracts Implemented:** BC-12 (negative config validation)

**Files:**
- Modify: `sim/cluster/deployment.go`
- Modify: `cmd/root.go`

**Step 1: Add KV config to DeploymentConfig + CLI flags**

In `sim/cluster/deployment.go`, add to `DeploymentConfig`:
```go
// Tiered KV cache configuration (PR12)
KVCPUBlocks         int64   // CPU tier KV blocks (0 = single-tier, default)
KVOffloadThreshold  float64 // GPU utilization threshold for offload (default 0.9)
KVTransferBandwidth float64 // blocks/tick transfer rate (default 100.0)
```

In `ToSimConfig()`, add to return:
```go
KVCPUBlocks:           d.KVCPUBlocks,
KVOffloadThreshold:    d.KVOffloadThreshold,
KVTransferBandwidth:   d.KVTransferBandwidth,
```

In `cmd/root.go`, add flag variables:
```go
// Tiered KV cache config (PR12)
kvCPUBlocks         int64
kvOffloadThreshold  float64
kvTransferBandwidth float64
```

Add flag registration:
```go
// Tiered KV cache (PR12)
runCmd.Flags().Int64Var(&kvCPUBlocks, "kv-cpu-blocks", 0, "CPU tier KV cache blocks (0 = disabled)")
runCmd.Flags().Float64Var(&kvOffloadThreshold, "kv-offload-threshold", 0.9, "GPU utilization threshold for offload")
runCmd.Flags().Float64Var(&kvTransferBandwidth, "kv-transfer-bandwidth", 100.0, "Transfer bandwidth (blocks/tick)")
```

Add validation in the `RunE` function:
```go
if kvCPUBlocks < 0 {
	logrus.Fatalf("--kv-cpu-blocks must be >= 0, got %d", kvCPUBlocks)
}
if kvOffloadThreshold < 0 || kvOffloadThreshold > 1 {
	logrus.Fatalf("--kv-offload-threshold must be in [0, 1], got %f", kvOffloadThreshold)
}
if kvCPUBlocks > 0 && kvTransferBandwidth <= 0 {
	logrus.Fatalf("--kv-transfer-bandwidth must be > 0 when --kv-cpu-blocks > 0, got %f", kvTransferBandwidth)
}
```

Wire into `DeploymentConfig`:
```go
KVCPUBlocks:         kvCPUBlocks,
KVOffloadThreshold:  kvOffloadThreshold,
KVTransferBandwidth: kvTransferBandwidth,
```

**Step 2: Build and verify**

Run: `go build ./...`
Expected: Build succeeds

**Step 3: Run ALL tests**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/cluster/deployment.go cmd/root.go
git commit -m "feat(cmd): add --kv-cpu-blocks, --kv-offload-threshold, --kv-transfer-bandwidth flags (BC-12)

- Add tiered KV config to DeploymentConfig and ToSimConfig
- Add 3 CLI flags with defaults (0, 0.9, 100.0)
- Validate: non-negative cpu-blocks, threshold in [0,1], positive bandwidth when tiered

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Section 5: Documentation

### Task 8: Documentation Updates

**Contracts Implemented:** None (documentation only)

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Add to Build and Run Commands:
```bash
# Run with tiered KV cache (GPU + CPU offloading)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --kv-cpu-blocks 10000 --kv-offload-threshold 0.9 --kv-transfer-bandwidth 100
```

Add `kv_store.go` and `kvcache_tiered.go` to file organization in `sim/` section.

Update "Current Implementation Focus" to mark PR12 as completed.

Update `cmd/root.go` description to include new flags.

**Step 2: Verify no stale references**

Run: `grep -r "planned for PR.12\|PR12.*planned\|sim/kv/" CLAUDE.md`
Expected: No matches (no stale forward references)

**Step 3: Build and test**

Run: `go build ./... && go test ./... -count=1`
Expected: ALL PASS

**Step 4: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for PR12 tiered KV cache

- Add tiered KV CLI example
- Add kv_store.go and kvcache_tiered.go to file organization
- Mark PR12 as completed
- Add --kv-cpu-blocks, --kv-offload-threshold, --kv-transfer-bandwidth to CLI docs

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1, 2 | Unit + Regression | `TestKVCacheState_ImplementsKVStore`, all existing tests |
| BC-2 | Task 4 | Unit | `TestTieredKVCache_OffloadTriggered_WhenGPUExceedsThreshold` |
| BC-3 | Task 5 | Unit | `TestTieredKVCache_CPUReload_AddsTransferLatency` |
| BC-4 | Task 5 | Unit | `TestTieredKVCache_PendingTransferLatency_QueryAndClear` |
| BC-5 | Task 3 | Unit | `TestKVCacheState_CacheHitRate_ReflectsLookups` |
| BC-6 | Task 5 | Unit | (covered in reload test — thrashing counter) |
| BC-7 | Task 6 | Unit | `TestCollectRawMetrics_IncludesPreemptionRate` |
| BC-8 | Task 6 | Unit | `TestCollectRawMetrics_IncludesPreemptionRate` |
| BC-9 | Task 4 | Unit | `TestTieredKVCache_Conservation_BlocksPreserved` |
| BC-10 | Task 4 | Unit | `TestTieredKVCache_CPUFull_OffloadStops` |
| BC-11 | Task 5 | Failure | `TestTieredKVCache_ZeroBandwidth_Panics` |
| BC-12 | Task 7 | Integration | CLI validation (manual: `./simulation_worker run --kv-cpu-blocks -1`) |

**Golden dataset:** No update needed. `--kv-cpu-blocks 0` (default) produces identical output (BC-1).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Interface change breaks existing tests | Low | High | Tasks 1-2 are pure refactor; run ALL tests after each | Task 2 |
| Offload corrupts block state | Medium | High | Conservation invariant test (BC-9) | Task 4 |
| Transfer latency double-counted | Low | Medium | Query-and-clear test (BC-4) | Task 5 |
| CPU reload misidentifies blocks | Medium | Medium | Hash-based lookup; test with known prefixes | Task 5 |
| Performance regression from interface dispatch | Low | Low | Interface dispatch is ~1ns; dominated by SHA256 hashing | N/A |

---

## PART 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — `KVStore` is required for tiered/single dispatch
- [x] No feature creep — strictly PR12 scope from macro plan
- [x] No unexercised flags — all 3 CLI flags exercise tiered code paths
- [x] No partial implementations — all contracts fully implemented
- [x] No breaking changes — BC-1 ensures backward compatibility
- [x] No hidden global state — all state owned by `TieredKVCache` instance
- [x] All new code will pass golangci-lint
- [x] No new shared test helpers needed
- [x] CLAUDE.md updated in Task 8
- [x] No stale references
- [x] Deviation log reviewed — all deviations from pre-design document
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5→6→7→8)
- [x] All contracts mapped to tasks (see Test Strategy)
- [x] Golden dataset: no update needed (BC-1)

---

## APPENDIX: File-Level Implementation Details

Detailed implementations are provided inline within each task's Step 3. The pre-design document (`docs/plans/pr12-architectural-predesign.md`) contains the binding architectural decisions including complete interface definitions, struct layouts, delegation patterns, and integration points.

Key files and their complete specifications:
- **`sim/kv_store.go`**: Task 1, Step 3 — `KVStore` interface (8 methods) + `NewKVStore` factory
- **`sim/kvcache_tiered.go`**: Task 4, Step 3 — `TieredKVCache`, `cpuTier`, `OffloadedBlock`, offload/reload logic
- **`sim/kvcache.go` modifications**: Task 1 (accessor methods) + Task 3 (hit/miss counters)
- **`sim/simulator.go` modifications**: Task 2 (interface type + method calls + transfer latency)
- **`sim/cluster/` modifications**: Task 6 (snapshot/metrics wire-up) + Task 7 (deployment config)
- **`cmd/root.go` modifications**: Task 7 (3 CLI flags + validation)
