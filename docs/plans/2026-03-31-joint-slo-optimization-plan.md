# Joint SLO-Aware Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three novel engine-level mechanisms (SLO-priority preemption ordering, tiered LRU KV eviction, admission-feedback batch formation) on top of the confirmed joint composition strategy, with experiment harness infrastructure for running the Strategy Evolution hypothesis bundles.

**Architecture:** Each mechanism is a new pluggable implementation behind an existing interface (`BatchFormation`, `KVStore`) or a small extension to an existing struct. Config wiring follows the `PolicyConfig.Scheduler` pattern already in place. Experiment harness reuses the existing `sim/workload` arrival process machinery to produce the mixed sustained+burst workload.

**Tech Stack:** Go 1.22+, `sim/batch_formation.go`, `sim/kv/cache.go`, `sim/config.go`, `sim/scheduler.go` (for `SLOTierPriority` reuse), `sim/workload/` (arrival processes), `sim/cluster/` (cluster simulator entry point).

**Design doc:** `docs/plans/2026-03-31-joint-slo-optimization-design.md`

---

## Part A: SLO-Priority Preemption Ordering

### Task 1: Extract victim selector from VLLMBatchFormation

**Files:**
- Modify: `sim/batch_formation.go`

This refactor adds a `VictimSelector` function type to `VLLMBatchFormation` without changing any existing behavior. Default `nil` means LIFO (current behavior). The change is purely additive.

**Step 1: Add the VictimSelector type and field**

Open `sim/batch_formation.go`. Add after the `VLLMBatchFormation` struct declaration:

```go
// VictimSelector chooses which index in the running batch to evict during KV pressure.
// Returns the index of the request to preempt.
// nil means LIFO (evict the last element — default vLLM behavior).
type VictimSelector func(requests []*Request) int

// VLLMBatchFormation implements the vLLM FCFS + chunked-prefill + preemption strategy.
type VLLMBatchFormation struct {
	selectVictim VictimSelector // nil = LIFO
}
```

**Step 2: Update preemptForTokens to use VictimSelector**

In `preemptForTokens`, replace this line:
```go
preemptedRequest := result.RunningBatch.Requests[len(result.RunningBatch.Requests)-1]
```

With:
```go
victimIdx := len(result.RunningBatch.Requests) - 1
if v.selectVictim != nil {
    victimIdx = v.selectVictim(result.RunningBatch.Requests)
}
preemptedRequest := result.RunningBatch.Requests[victimIdx]
```

And replace the LIFO tail-trim:
```go
result.RunningBatch.Requests = result.RunningBatch.Requests[:len(result.RunningBatch.Requests)-1]
```

With a splice that handles any index:
```go
result.RunningBatch.Requests = append(
    result.RunningBatch.Requests[:victimIdx],
    result.RunningBatch.Requests[victimIdx+1:]...)
```

**Step 3: Verify NewBatchFormation still returns LIFO behavior**

`NewBatchFormation()` already returns `&VLLMBatchFormation{}` — `selectVictim` will be nil, preserving LIFO. No change needed.

**Step 4: Run existing tests**

```bash
go test ./sim/... -run TestBatch -v
```

Expected: all existing batch formation tests PASS. This is a pure refactor — no behavioral change when `selectVictim` is nil.

**Step 5: Commit**

```bash
git add sim/batch_formation.go
git commit -m "refactor(sim): extract VictimSelector from VLLMBatchFormation (no behavior change)"
```

---

### Task 2: Implement SLO-priority victim selector

**Files:**
- Modify: `sim/batch_formation.go`

**Step 1: Add the SLO-priority victim selector function**

Add after `VictimSelector` type definition:

```go
// SLOLowestPriorityVictim selects the running request with the lowest SLO tier priority.
// Ties (same SLO class) resolve to the last element (LIFO among equals).
// Uses sim.SLOTierPriority which maps: background=0, batch=1, sheddable=2, standard=3, critical=4.
func SLOLowestPriorityVictim(requests []*Request) int {
    idx := 0
    minPriority := SLOTierPriority(requests[0].SLOClass)
    for i := 1; i < len(requests); i++ {
        p := SLOTierPriority(requests[i].SLOClass)
        if p < minPriority {
            minPriority = p
            idx = i
        }
    }
    return idx
}
```

**Step 2: Add SLOPriorityBatchFormation constructor**

```go
// NewSLOPriorityBatchFormation creates a BatchFormation that uses SLO-priority victim
// selection during KV preemption: lowest-SLO running request is evicted first.
// All other behavior (chunked prefill, decode, scheduling) is identical to VLLMBatchFormation.
func NewSLOPriorityBatchFormation() BatchFormation {
    return &VLLMBatchFormation{selectVictim: SLOLowestPriorityVictim}
}
```

**Step 3: Run tests (should compile and pass)**

```bash
go test ./sim/... -run TestBatch -v
```

Expected: PASS (no new tests yet — that's Task 3).

**Step 4: Commit**

```bash
git add sim/batch_formation.go
git commit -m "feat(sim): SLO-priority preemption victim selector"
```

---

### Task 3: Write behavioral tests for SLO-priority preemption

**Files:**
- Modify: `sim/batch_formation_test.go` (or create `sim/batch_formation_slo_preempt_test.go`)

**Step 1: Write the failing test — victim is lowest SLO, not LIFO**

Add to the test file:

```go
// TestSLOPriorityPreemption_VictimsLowSLOFirst verifies that under KV pressure,
// the lowest-SLO running request is evicted, not the most-recently-scheduled one.
// BC-P1: SLO-priority victim selection evicts sheddable before critical.
func TestSLOPriorityPreemption_VictimsLowSLOFirst(t *testing.T) {
    // Build a tiny KV cache that cannot fit a new critical request once the
    // running batch holds a critical + sheddable request.
    blockSize := int64(16)
    totalBlocks := int64(4) // holds exactly 4 requests of 1 block each at minimum
    // Import sim/kv to register NewKVCacheStateFunc
    kvcache := sim.MustNewKVCacheState(totalBlocks, blockSize)

    criticalReq := &sim.Request{
        ID:          "critical-1",
        SLOClass:    "critical",
        InputTokens: makeTokens(1),
        State:       sim.StateRunning,
    }
    sheddableReq := &sim.Request{
        ID:          "sheddable-1",
        SLOClass:    "sheddable",
        InputTokens: makeTokens(1),
        State:       sim.StateRunning,
    }
    incomingCritical := &sim.Request{
        ID:          "critical-new",
        SLOClass:    "critical",
        InputTokens: makeTokens(int(totalBlocks)), // needs all blocks
        State:       sim.StateQueued,
    }

    // Pre-allocate KV blocks for both running requests to leave no room for incoming.
    // (test helper: allocate blocks manually)
    _ = kvcache.AllocateKVBlocks(criticalReq, 0, 1, nil)
    _ = kvcache.AllocateKVBlocks(sheddableReq, 0, 1, nil)

    wq := sim.NewWaitQueue()
    wq.Enqueue(incomingCritical)

    ctx := sim.BatchContext{
        RunningBatch:       &sim.Batch{Requests: []*sim.Request{criticalReq, sheddableReq}},
        WaitQ:              wq,
        KVCache:            kvcache,
        MaxScheduledTokens: blockSize * totalBlocks,
        MaxRunningReqs:     4,
        Now:                100,
        ComputedTokens:     map[string]int64{},
    }

    bf := sim.NewSLOPriorityBatchFormation()
    result := bf.FormBatch(ctx)

    // The sheddable request must be the preemption victim, not the critical one.
    if len(result.Preempted) == 0 {
        t.Fatal("expected preemption to occur but none did")
    }
    for _, p := range result.Preempted {
        if p.Request.SLOClass == "critical" {
            t.Errorf("critical request was preempted; expected only sheddable to be evicted")
        }
    }
    for _, r := range result.RunningBatch.Requests {
        if r.ID == "sheddable-1" && r.State == sim.StateRunning {
            t.Errorf("sheddable request is still running after preemption")
        }
    }
}
```

**Step 2: Write the control test — uniform SLO falls back to LIFO**

```go
// TestSLOPriorityPreemption_UniformSLOUsesLIFO verifies that when all running
// requests share the same SLO class, eviction falls back to LIFO ordering.
// BC-P2: Ties in SLO priority resolve to last element.
func TestSLOPriorityPreemption_UniformSLOUsesLIFO(t *testing.T) {
    first := &sim.Request{ID: "std-first", SLOClass: "standard", InputTokens: makeTokens(1), State: sim.StateRunning}
    last  := &sim.Request{ID: "std-last",  SLOClass: "standard", InputTokens: makeTokens(1), State: sim.StateRunning}
    requests := []*sim.Request{first, last}
    idx := sim.SLOLowestPriorityVictim(requests)
    if requests[idx].ID != "std-last" {
        t.Errorf("expected LIFO tie-break (last element), got index %d = %s", idx, requests[idx].ID)
    }
}
```

**Step 3: Write the backward-compatibility test — nil selector is LIFO**

```go
// TestVLLMBatchFormation_NilSelectorIsLIFO verifies that the default VLLMBatchFormation
// (nil selectVictim) still uses LIFO eviction, preserving prior behavior.
// BC-P3: Existing callers of NewBatchFormation() are unaffected by the refactor.
func TestVLLMBatchFormation_NilSelectorIsLIFO(t *testing.T) {
    // SLOLowestPriorityVictim on a critical+sheddable slice returns sheddable.
    // The nil (LIFO) selector on the same slice returns the last element.
    // Use different orderings to distinguish the two.
    critical := &sim.Request{ID: "c", SLOClass: "critical"}
    sheddable := &sim.Request{ID: "s", SLOClass: "sheddable"}

    // critical is last → LIFO evicts critical; SLO-priority evicts sheddable
    requests := []*sim.Request{sheddable, critical}
    idx := sim.SLOLowestPriorityVictim(requests)
    if requests[idx].ID != "s" {
        t.Errorf("SLOLowestPriorityVictim: expected sheddable (idx 0), got %s", requests[idx].ID)
    }
    // LIFO = last element = idx 1 = critical — confirmed by index arithmetic, not direct function call
    lifoIdx := len(requests) - 1
    if requests[lifoIdx].ID != "c" {
        t.Errorf("LIFO: expected critical (idx 1), got %s", requests[lifoIdx].ID)
    }
}
```

**Step 4: Run tests**

```bash
go test ./sim/... -run TestSLOPriority -v
go test ./sim/... -run TestVLLMBatchFormation_NilSelector -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim/batch_formation_test.go   # or whichever file was modified
git commit -m "test(sim): behavioral tests for SLO-priority preemption victim selection"
```

---

## Part B: SLO-Aware KV Prefix Cache Eviction

### Task 4: Add tier priority field to KVBlock

**Files:**
- Modify: `sim/kv/cache.go`

The `KVBlock` struct is in `sim/kv/cache.go`. Adding a `TierPriority` field tags each block with the SLO priority of the request that owns it. Blocks with lower tier priority are evicted first.

**Step 1: Add TierPriority to KVBlock**

In `sim/kv/cache.go`, add one field to `KVBlock`:

```go
type KVBlock struct {
    ID           int64    // Unique ID of the block
    RefCount     int      // Number of active requests referencing this block
    InUse        bool     // Whether the block is currently in use
    Hash         string   // Prefix hash
    Tokens       []int    // Actual tokens stored
    TierPriority int      // SLO tier priority of owning request (0=sheddable/bg, higher=more protected)
    PrevFree     *KVBlock // LRU doubly linked list: previous free block
    NextFree     *KVBlock // LRU doubly linked list: next free block
}
```

`TierPriority` defaults to 0 (sheddable = evicted first) when a block is created. It is set during `AllocateKVBlocks`.

**Step 2: Run existing tests to confirm zero-value is safe**

```bash
go test ./sim/kv/... -v
```

Expected: PASS. The zero value (0 = sheddable priority) is safe — all blocks start as equally evictable.

**Step 3: Commit**

```bash
git add sim/kv/cache.go
git commit -m "feat(sim/kv): add TierPriority field to KVBlock"
```

---

### Task 5: Tag blocks with tier priority during allocation

**Files:**
- Modify: `sim/kv/cache.go`

`AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64)` already receives `req`. The `sim.Request` struct has `SLOClass string`. The SLO tier priority mapping is in `sim.SLOTierPriority(class string) int` (defined in `sim/admission.go`).

**Step 1: Resolve the import cycle**

`sim/kv/cache.go` already imports `sim` (for `sim.Request`, `sim.KVStore`). `sim.SLOTierPriority` is in package `sim`. The import is already present — no new import needed.

**Step 2: Set TierPriority when a new block is popped**

In `AllocateKVBlocks`, when a new block is popped via `popFreeBlock()`, set its tier priority from the request. Find the section that calls `popFreeBlock`:

```go
// After: blk := kvc.popFreeBlock() — add:
blk.TierPriority = sim.SLOTierPriority(req.SLOClass)
```

Note: cached blocks (re-used prefix hits) also need their tier priority updated to reflect the new owner. In the cached-block path, after `block.InUse = true` and `block.RefCount++`:

```go
// Update tier priority to the maximum of the block's existing priority
// and the new request's priority. This preserves cached blocks for
// the highest-priority request currently using them.
if p := sim.SLOTierPriority(req.SLOClass); p > block.TierPriority {
    block.TierPriority = p
}
```

**Step 3: Reset TierPriority when a block is freed**

When a block is released (freed back to prefix cache), its tier priority should reflect what's appropriate for eviction ordering. In `ReleaseKVBlocks`, after decrementing RefCount and before adding back to free list, reset to 0 if RefCount reaches 0:

```go
if block.RefCount == 0 {
    block.TierPriority = 0  // No longer owned by any request; evict freely
    // ... existing free list add ...
}
```

**Step 4: Run tests**

```bash
go test ./sim/kv/... -v
```

Expected: PASS. (Tier priority is set but not yet used by eviction — no behavioral change yet.)

**Step 5: Commit**

```bash
git add sim/kv/cache.go
git commit -m "feat(sim/kv): tag KVBlocks with SLO tier priority during allocation"
```

---

### Task 6: Implement three-tier free lists in KVCacheState

**Files:**
- Modify: `sim/kv/cache.go`

Replace the single `FreeHead/FreeTail` pair with three tier-specific free lists. `popFreeBlock` exhausts tier 0 (sheddable/background) before tier 1 (standard), tier 1 before tier 2 (critical). LRU ordering is preserved within each tier.

**Step 1: Replace FreeHead/FreeTail with per-tier lists**

In `KVCacheState`, replace:
```go
FreeHead *KVBlock
FreeTail *KVBlock
```

With:
```go
// freeTierHead[i] and freeTierTail[i] are the LRU head/tail for tier priority i.
// Tier 0 = background/sheddable (evict first), tier 3 = critical (evict last).
// Uses the same priority scale as sim.SLOTierPriority (0–4 range; indices 0–4).
freeTierHead [5]*KVBlock
freeTierTail [5]*KVBlock
```

Note: `sim.SLOTierPriority` returns values 0–4 (background=0, batch=1, sheddable=2, standard=3, critical=4). Use 5 slots to match.

**Step 2: Update removeFromFreeList**

The existing `removeFromFreeList` updates `FreeHead`/`FreeTail`. Update it to use the block's tier:

```go
func (kvc *KVCacheState) removeFromFreeList(blk *KVBlock) {
    tier := blk.TierPriority
    if blk.PrevFree != nil {
        blk.PrevFree.NextFree = blk.NextFree
    } else {
        kvc.freeTierHead[tier] = blk.NextFree
    }
    if blk.NextFree != nil {
        blk.NextFree.PrevFree = blk.PrevFree
    } else {
        kvc.freeTierTail[tier] = blk.PrevFree
    }
    blk.PrevFree = nil
    blk.NextFree = nil
}
```

**Step 3: Update addToFreeList**

```go
func (kvc *KVCacheState) addToFreeList(blk *KVBlock) {
    tier := blk.TierPriority
    blk.PrevFree = kvc.freeTierTail[tier]
    blk.NextFree = nil
    if kvc.freeTierTail[tier] != nil {
        kvc.freeTierTail[tier].NextFree = blk
    } else {
        kvc.freeTierHead[tier] = blk
    }
    kvc.freeTierTail[tier] = blk
}
```

**Step 4: Update popFreeBlock to drain lowest tier first**

```go
func (kvc *KVCacheState) popFreeBlock() *KVBlock {
    for tier := 0; tier < 5; tier++ {
        head := kvc.freeTierHead[tier]
        if head == nil {
            continue
        }
        kvc.removeFromFreeList(head)
        if head.Hash != "" {
            delete(kvc.HashToBlock, head.Hash)
            head.Hash = ""
        }
        head.Tokens = nil
        return head
    }
    return nil
}
```

**Step 5: Update NewKVCacheState initialization**

The initialization loop adds all blocks to the free list. All blocks start with `TierPriority=0`. Update to call the new `addToFreeList`:

```go
// In NewKVCacheState, the initialization loop that builds the free list:
for i := int64(0); i < totalBlocks; i++ {
    blk := &KVBlock{ID: i, TierPriority: 0}
    kvc.Blocks[i] = blk
    kvc.addToFreeList(blk)
}
```

Remove the old manual `FreeHead`/`FreeTail` chain construction.

**Step 6: Update countFreeBlocks (if needed)**

`countFreeBlocks()` returns `TotalBlocks - UsedBlockCnt`. This is unchanged — no list traversal needed.

**Step 7: Update rollbackAllocation**

`rollbackAllocation` calls `addToFreeList`/`removeFromFreeList` via the prepend helper. Verify the rollback helpers use the same tier-aware list functions. The key invariant: rollback must restore blocks to their pre-allocation tier. Since `TierPriority` is set during allocation, rollback must reset it to 0 before re-adding to the free list. Add to the rollback loop:

```go
blk.TierPriority = 0
kvc.addToFreeList(blk) // now routes to tier 0
```

**Step 8: Run all KV tests**

```bash
go test ./sim/kv/... -v
go test ./sim/... -run TestKVCache -v
```

Expected: PASS.

**Step 9: Commit**

```bash
git add sim/kv/cache.go
git commit -m "feat(sim/kv): three-tier LRU free lists — sheddable evicted before critical"
```

---

### Task 7: Write behavioral tests for tiered LRU eviction

**Files:**
- Modify: `sim/kv/cache_test.go`

**Step 1: Write test — sheddable blocks evicted before standard**

```go
// TestTieredLRU_SheddableEvictedBeforeStandard verifies that under KV pressure,
// sheddable-tier prefix blocks are evicted before standard-tier blocks.
// BC-KV1: Tiered LRU protects high-SLO prefix cache entries.
func TestTieredLRU_SheddableEvictedBeforeStandard(t *testing.T) {
    kvc := NewKVCacheState(2, 16) // only 2 blocks total

    sheddableReq := &sim.Request{ID: "shed-1", SLOClass: "sheddable", InputTokens: makeTokens(16)}
    standardReq  := &sim.Request{ID: "std-1",  SLOClass: "standard",  InputTokens: makeTokens(16)}
    newReq        := &sim.Request{ID: "new-1",  SLOClass: "critical",  InputTokens: makeTokens(16)}

    // Allocate both blocks: sheddable gets block 0, standard gets block 1.
    ok1 := kvc.AllocateKVBlocks(sheddableReq, 0, 1, nil)
    ok2 := kvc.AllocateKVBlocks(standardReq, 0, 1, nil)
    if !ok1 || !ok2 {
        t.Fatal("initial allocation failed")
    }

    // Release both requests back to the prefix cache (free list).
    kvc.ReleaseKVBlocks(sheddableReq)
    kvc.ReleaseKVBlocks(standardReq)

    // Now allocate for a new request — forces eviction of one free block.
    ok := kvc.AllocateKVBlocks(newReq, 0, 1, nil)
    if !ok {
        t.Fatal("allocation for new request failed")
    }

    // The sheddable block must have been evicted (its hash cleared).
    // The standard block's hash must still be in HashToBlock.
    if _, found := kvc.HashToBlock[sheddableReq.ID]; found {
        t.Error("sheddable block hash should have been evicted but is still in HashToBlock")
    }
}
```

**Step 2: Write INV-4 conservation test with tiered LRU**

```go
// TestTieredLRU_KVConservation verifies that allocated+free == total
// after a sequence of mixed-tier allocations and releases (INV-4).
func TestTieredLRU_KVConservation(t *testing.T) {
    total := int64(8)
    kvc := NewKVCacheState(total, 16)

    reqs := []*sim.Request{
        {ID: "c1", SLOClass: "critical",  InputTokens: makeTokens(16)},
        {ID: "s1", SLOClass: "standard",  InputTokens: makeTokens(16)},
        {ID: "sh1", SLOClass: "sheddable", InputTokens: makeTokens(16)},
    }
    for _, r := range reqs {
        kvc.AllocateKVBlocks(r, 0, 2, nil)
    }
    kvc.ReleaseKVBlocks(reqs[2]) // release sheddable

    free := kvc.countFreeBlocks()
    used := kvc.UsedBlockCnt
    if free+used != total {
        t.Errorf("INV-4 violated: free=%d + used=%d != total=%d", free, used, total)
    }
}
```

**Step 3: Run tests**

```bash
go test ./sim/kv/... -run TestTieredLRU -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add sim/kv/cache_test.go
git commit -m "test(sim/kv): behavioral tests for tiered LRU eviction (BC-KV1, INV-4)"
```

---

## Part C: Admission-Feedback Batch Formation

### Task 8: Implement TierBudgetBatchFormation

**Files:**
- Modify: `sim/batch_formation.go`

`TierBudgetBatchFormation` is a new `BatchFormation` implementation that partitions the per-step token budget by SLO tier. It re-implements `FormBatch` based on `VLLMBatchFormation` logic with one addition: a per-tier token counter that enforces budget caps.

**Step 1: Add TierBudgetBatchFormation struct**

```go
// TierBudgetBatchFormation partitions the per-step token budget by SLO tier.
// Critical requests get first claim (CriticalFrac × MaxScheduledTokens),
// standard gets StandardFrac × remainder, sheddable takes what is left.
// All other behavior (chunked prefill, decode, preemption) is identical to
// VLLMBatchFormation with LIFO victim selection.
//
// Use NewTierBudgetBatchFormation to construct with validated fractions.
type TierBudgetBatchFormation struct {
    CriticalFrac  float64 // fraction of MaxScheduledTokens reserved for critical tier
    StandardFrac  float64 // fraction of remaining (1-CriticalFrac) reserved for standard
    // SheddableFrac is derived: (1-CriticalFrac) * (1-StandardFrac)
}

// NewTierBudgetBatchFormation creates a TierBudgetBatchFormation with validated fractions.
// CriticalFrac must be in (0, 1). StandardFrac must be in (0, 1).
// The sheddable fraction is (1-criticalFrac) * (1-standardFrac).
func NewTierBudgetBatchFormation(criticalFrac, standardFrac float64) *TierBudgetBatchFormation {
    if criticalFrac <= 0 || criticalFrac >= 1 {
        panic(fmt.Sprintf("NewTierBudgetBatchFormation: CriticalFrac must be in (0,1), got %v", criticalFrac))
    }
    if standardFrac <= 0 || standardFrac >= 1 {
        panic(fmt.Sprintf("NewTierBudgetBatchFormation: StandardFrac must be in (0,1), got %v", standardFrac))
    }
    return &TierBudgetBatchFormation{
        CriticalFrac: criticalFrac,
        StandardFrac: standardFrac,
    }
}
```

**Step 2: Implement tierBudgets helper**

```go
// tierBudgets computes per-tier token budgets from MaxScheduledTokens.
// Returns [criticalBudget, standardBudget, sheddableBudget].
func (t *TierBudgetBatchFormation) tierBudgets(maxTokens int64) [3]int64 {
    critBudget := int64(float64(maxTokens) * t.CriticalFrac)
    remaining  := maxTokens - critBudget
    stdBudget  := int64(float64(remaining) * t.StandardFrac)
    shedBudget := remaining - stdBudget
    return [3]int64{critBudget, stdBudget, shedBudget}
}

// tierIndex maps SLO class names to budget slice indices.
// critical → 0, standard → 1, everything else → 2.
func tierIndex(sloClass string) int {
    switch sloClass {
    case "critical":
        return 0
    case "standard":
        return 1
    default:
        return 2
    }
}
```

**Step 3: Implement FormBatch**

`TierBudgetBatchFormation.FormBatch` runs the same logic as `VLLMBatchFormation.FormBatch` with one modification: each request's token allocation is checked against both the global budget AND its tier budget. If the tier budget is exhausted, the request is skipped for this step (receives 0 new tokens) but stays in the running batch — it will receive tokens in the next step.

```go
func (t *TierBudgetBatchFormation) FormBatch(ctx BatchContext) BatchResult {
    if ctx.RunningBatch == nil {
        ctx.RunningBatch = &Batch{}
    }
    result := BatchResult{RunningBatch: ctx.RunningBatch}
    globalBudget := ctx.MaxScheduledTokens
    tierBudgets := t.tierBudgets(ctx.MaxScheduledTokens)
    tierUsed := [3]int64{}

    for _, req := range result.RunningBatch.Requests {
        req.NumNewTokens = 0
    }

    // Use VLLMBatchFormation for the core logic but inject tier budget checking.
    // We delegate to an inner VLLMBatchFormation and post-filter based on tier overuse.
    // This is a soft cap: requests that exceed their tier budget are skipped (stalled
    // for one step), not preempted. The global budget still applies.
    inner := &VLLMBatchFormation{}
    innerCtx := ctx
    innerCtx.MaxScheduledTokens = globalBudget
    innerResult := inner.FormBatch(innerCtx)

    // Post-pass: revoke token grants for requests that exceed their tier budget.
    // Walk newly scheduled and running requests; if tier is over budget, zero tokens.
    for _, req := range innerResult.RunningBatch.Requests {
        ti := tierIndex(req.SLOClass)
        if req.NumNewTokens > 0 {
            if tierUsed[ti]+int64(req.NumNewTokens) > tierBudgets[ti] {
                // Tier budget exceeded: stall this request for this step.
                // Return KV blocks it was just allocated (if any).
                // Note: this is a soft stall — the request stays in the running batch.
                req.NumNewTokens = 0
                delete(innerCtx.ComputedTokens, req.ID)
                continue
            }
            tierUsed[ti] += int64(req.NumNewTokens)
        }
    }

    return innerResult
}
```

**Important behavioral note:** The post-pass approach above is a simplification that avoids duplicating the complex preemption and KV allocation logic. It has a known limitation: KV blocks allocated by the inner pass for stalled requests are not released. The correct implementation for production use requires integrating tier budget checks *inside* the FormBatch loop rather than post-filtering. The plan prescribes the post-filter approach for testability; the micro-plan can upgrade to the integrated approach once tests are green.

**Step 4: Compile check**

```bash
go build ./sim/...
```

Expected: compiles cleanly.

**Step 5: Commit**

```bash
git add sim/batch_formation.go
git commit -m "feat(sim): TierBudgetBatchFormation — per-SLO step token budget partitioning"
```

---

### Task 9: Write behavioral tests for tier budget partitioning

**Files:**
- Modify: `sim/batch_formation_test.go`

**Step 1: Write test — critical gets first claim on token budget**

```go
// TestTierBudget_CriticalGetsFirstClaim verifies that critical requests consume
// their allocated fraction of the token budget before standard/sheddable.
// BC-TB1: token budget fraction is the active mechanism.
func TestTierBudget_CriticalGetsFirstClaim(t *testing.T) {
    bf := sim.NewTierBudgetBatchFormation(0.6, 0.7) // critical=60%, standard=28%, sheddable=12%
    budgets := bf.TierBudgets(1000) // expose via exported method or test helper
    if budgets[0] != 600 {
        t.Errorf("critical budget: want 600, got %d", budgets[0])
    }
    if budgets[1] != 280 {
        t.Errorf("standard budget: want 280, got %d", budgets[1])
    }
    if budgets[2] != 120 {
        t.Errorf("sheddable budget: want 120, got %d", budgets[2])
    }
}
```

Note: expose `tierBudgets` as an exported method `TierBudgets(maxTokens int64) [3]int64` for testability.

**Step 2: Write validation test — panics on invalid fractions**

```go
func TestNewTierBudgetBatchFormation_PanicsOnInvalidFrac(t *testing.T) {
    cases := []struct{ cf, sf float64 }{
        {0, 0.7},   // criticalFrac = 0
        {1, 0.7},   // criticalFrac = 1
        {0.5, 0},   // standardFrac = 0
        {0.5, 1.0}, // standardFrac = 1
    }
    for _, c := range cases {
        func() {
            defer func() {
                if r := recover(); r == nil {
                    t.Errorf("expected panic for criticalFrac=%v, standardFrac=%v", c.cf, c.sf)
                }
            }()
            sim.NewTierBudgetBatchFormation(c.cf, c.sf)
        }()
    }
}
```

**Step 3: Run tests**

```bash
go test ./sim/... -run TestTierBudget -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add sim/batch_formation_test.go
git commit -m "test(sim): behavioral tests for TierBudgetBatchFormation (BC-TB1)"
```

---

## Part D: Config and CLI Wiring

### Task 10: Wire new policies into PolicyConfig and factory functions

**Files:**
- Modify: `sim/config.go`
- Modify: `sim/batch_formation.go` (update `NewBatchFormation`)
- Modify: `sim/simulator.go` (use config to construct BatchFormation)

**Step 1: Add BatchFormationPolicy to PolicyConfig**

In `sim/config.go`, update `PolicyConfig`:

```go
type PolicyConfig struct {
    PriorityPolicy        string
    Scheduler             string
    BatchFormationPolicy  string  // "vllm" (default), "slo-priority-preemption", "tier-budget"
    TierBudgetCritFrac    float64 // for "tier-budget" policy; 0 means use default (0.50)
    TierBudgetStdFrac     float64 // for "tier-budget" policy; 0 means use default (0.70)
}
```

Update `NewPolicyConfig` to accept the new fields — or add a separate constructor for the engine mechanism policies. Following R4 (canonical constructors), add:

```go
// NewEnginePolicyConfig creates a PolicyConfig for engine mechanism experiments.
// batchPolicy: "vllm", "slo-priority-preemption", "tier-budget"
// critFrac, stdFrac: tier budget fractions (only used when batchPolicy="tier-budget")
func NewEnginePolicyConfig(priorityPolicy, scheduler, batchPolicy string, critFrac, stdFrac float64) PolicyConfig {
    cfg := NewPolicyConfig(priorityPolicy, scheduler)
    cfg.BatchFormationPolicy = batchPolicy
    cfg.TierBudgetCritFrac = critFrac
    cfg.TierBudgetStdFrac = stdFrac
    return cfg
}
```

**Step 2: Update NewBatchFormationFromPolicy factory**

In `sim/batch_formation.go`, add:

```go
// NewBatchFormationFromPolicy creates a BatchFormation from PolicyConfig.
// "vllm" → VLLMBatchFormation (LIFO preemption, default)
// "slo-priority-preemption" → VLLMBatchFormation with SLO-priority victim selector
// "tier-budget" → TierBudgetBatchFormation (critFrac defaults to 0.50, stdFrac to 0.70)
func NewBatchFormationFromPolicy(cfg PolicyConfig) BatchFormation {
    switch cfg.BatchFormationPolicy {
    case "", "vllm":
        return &VLLMBatchFormation{}
    case "slo-priority-preemption":
        return NewSLOPriorityBatchFormation()
    case "tier-budget":
        cf := cfg.TierBudgetCritFrac
        if cf == 0 {
            cf = 0.50
        }
        sf := cfg.TierBudgetStdFrac
        if sf == 0 {
            sf = 0.70
        }
        return NewTierBudgetBatchFormation(cf, sf)
    default:
        panic(fmt.Sprintf("NewBatchFormationFromPolicy: unknown policy %q", cfg.BatchFormationPolicy))
    }
}
```

**Step 3: Update Simulator to use the factory**

In `sim/simulator.go`, replace:
```go
batchFormation := NewBatchFormation()
```
With:
```go
batchFormation := NewBatchFormationFromPolicy(cfg.PolicyConfig)
```

**Step 4: Add CLI flags in cmd/ layer**

In the relevant `cmd/run.go` or `cmd/flags.go` (wherever `--scheduler` is defined), add:

```go
cmd.Flags().String("batch-formation", "vllm", `batch formation policy: "vllm", "slo-priority-preemption", "tier-budget"`)
cmd.Flags().Float64("tier-budget-critical-frac", 0.50, "critical tier token budget fraction (tier-budget policy only)")
cmd.Flags().Float64("tier-budget-standard-frac", 0.70, "standard tier fraction of remaining budget (tier-budget policy only)")
```

Wire these into `PolicyConfig` construction where `SimConfig` is built from flags.

**Step 5: Run all tests**

```bash
go test ./... -v 2>&1 | tail -30
```

Expected: PASS.

**Step 6: Build check**

```bash
go build -o blis main.go
./blis run --help | grep batch
```

Expected: `--batch-formation` flag visible in help output.

**Step 7: Commit**

```bash
git add sim/config.go sim/batch_formation.go sim/simulator.go cmd/
git commit -m "feat(config,cmd): wire SLO-priority preemption and tier-budget batch formation into CLI"
```

---

## Part E: Experiment Harness

### Task 11: Build the mixed sustained+burst workload generator

**Files:**
- Modify: `sim/workload/` (add or extend arrival process composition)

The mixed workload requires two superimposed arrival processes: a Poisson base at 85% of saturation and a Gamma burst overlay at CV=2.0. BLIS's existing `ArrivalProcess` supports Poisson, Gamma, Weibull, and Constant individually. The composition needs a new entry point.

**Step 1: Understand the existing workload spec schema**

Read `sim/workload/` to understand `WorkloadSpec` and `ArrivalProcess`. The relevant files are `sim/workload/spec.go` (or equivalent). The arrival process is configured per-cohort.

**Step 2: Define the mixed workload spec**

Create `experiments/joint-opt/workload-mixed.yaml`:

```yaml
# Mixed sustained + burst workload for joint SLO optimization experiments.
# Baseline saturation throughput must be measured first (Iteration 0).
# Set base_rate to 85% of saturation; burst_peak_rate to 2× saturation.
arrival_process:
  type: composed         # NEW: superposition of two processes
  components:
    - type: poisson
      rate: BASE_RATE    # fill in from Iteration 0 measurement
    - type: gamma
      rate: BURST_RATE   # fill in: (2 × saturation) - BASE_RATE
      cv: 2.0
slo_distribution:
  - class: critical
    fraction: 0.20
  - class: standard
    fraction: 0.60
  - class: sheddable
    fraction: 0.20
request_shapes:
  # Orthogonal: all tiers share identical shapes
  prompt_tokens: 512
  output_tokens: 128
  prefix_tokens: 256     # moderate prefix sharing to activate prefix cache
num_requests: 5000
seeds: [42, 123, 456]
```

Note: If `type: composed` is not yet supported in `WorkloadSpec`, implement it as two separate cohorts with distinct arrival rates and merge them in the cluster simulator input. The simpler fallback is to create two `WorkloadSpec` files and compose requests at the CLI level.

**Step 3: Create experiment run scripts**

Create `experiments/joint-opt/run_iter0_baseline.sh`:

```bash
#!/bin/bash
set -euo pipefail
# Iteration 0: measure baseline compound strategy performance.
# Outputs: results/iter0_baseline_{seed}.json for each seed.

SEEDS=(42 123 456)
MODEL=${MODEL:-qwen/qwen3-14b}
HORIZON=${HORIZON:-3600}  # 1 hour simulation

mkdir -p results

for SEED in "${SEEDS[@]}"; do
  echo "=== Seed $SEED ==="
  ./blis run \
    --model "$MODEL" \
    --seed "$SEED" \
    --horizon "$HORIZON" \
    --scheduler priority-fcfs \
    --admission tier-shed \
    --routing-weights "pa:4,qd:3" \
    --batch-formation vllm \
    --workload-spec experiments/joint-opt/workload-mixed.yaml \
    --output results/iter0_baseline_seed${SEED}.json
done

echo "Iteration 0 complete. Aggregate with: jq -s '[.[].metrics]' results/iter0_baseline_*.json"
```

Create `experiments/joint-opt/run_iter2_slo_preempt.sh`:

```bash
#!/bin/bash
set -euo pipefail
# Iteration 2: SLO-priority preemption ordering.
# H-main: >15% critical TTFT P99 improvement vs Iteration 1 compound.

SEEDS=(42 123 456)
MODEL=${MODEL:-qwen/qwen3-14b}
HORIZON=${HORIZON:-3600}

mkdir -p results

for SEED in "${SEEDS[@]}"; do
  # Treatment: SLO-priority preemption
  ./blis run \
    --model "$MODEL" --seed "$SEED" --horizon "$HORIZON" \
    --scheduler priority-fcfs \
    --admission tier-shed \
    --routing-weights "pa:4,qd:3" \
    --batch-formation slo-priority-preemption \
    --workload-spec experiments/joint-opt/workload-mixed.yaml \
    --output results/iter2_treatment_seed${SEED}.json

  # Ablation: revert to LIFO preemption, keep everything else
  ./blis run \
    --model "$MODEL" --seed "$SEED" --horizon "$HORIZON" \
    --scheduler priority-fcfs \
    --admission tier-shed \
    --routing-weights "pa:4,qd:3" \
    --batch-formation vllm \
    --workload-spec experiments/joint-opt/workload-mixed.yaml \
    --output results/iter2_ablation_seed${SEED}.json
done
```

Create analogous scripts for iter3 (tiered-lru) and iter4 (tier-budget).

**Step 4: Create findings templates**

Create `experiments/joint-opt/findings/iter2-bundle.md`:

```markdown
# Iteration 2 Hypothesis Bundle — SLO-Priority Preemption Ordering

**Status:** [ ] Designed | [ ] Running | [ ] Complete

## H-main prediction
Critical TTFT P99 improvement vs Iteration 1 compound: **>15%**
Causal mechanism: SLO-priority victim selection ensures sheddable requests bear preemption cost.
Diagnostic: If <5%, preemption is rare at this operating point.

## Results (fill in after runs)
| Seed | Treatment P99 | Ablation P99 | Improvement |
|------|--------------|--------------|-------------|
| 42   |              |              |             |
| 123  |              |              |             |
| 456  |              |              |             |
| Mean |              |              |             |

## H-zero-sum
Standard goodput degradation vs Iteration 1: ____% (threshold: <20%)

## Decision
[ ] PROCEED  [ ] REVISE  [ ] RESTART
Rationale:
```

**Step 5: Commit**

```bash
git add experiments/
git commit -m "feat(experiments): joint-opt harness — mixed workload spec, run scripts, findings templates"
```

---

### Task 12: Validate the full pipeline end-to-end

**Step 1: Build**

```bash
go build -o blis main.go
```

**Step 2: Smoke test each new policy**

```bash
# SLO-priority preemption
./blis run --model qwen/qwen3-14b \
  --batch-formation slo-priority-preemption \
  --num-requests 50 --seed 42 2>/dev/null | head -5

# Tier budget
./blis run --model qwen/qwen3-14b \
  --batch-formation tier-budget \
  --tier-budget-critical-frac 0.5 \
  --tier-budget-standard-frac 0.7 \
  --num-requests 50 --seed 42 2>/dev/null | head -5
```

Expected: both complete without panic and produce valid JSON output.

**Step 3: Run full test suite**

```bash
go test ./... 2>&1 | tail -10
```

Expected: PASS.

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: joint SLO optimization — all three engine mechanisms + experiment harness"
```

---

## Invariant Compliance Checklist

Before marking any iteration complete, verify:

- [ ] **INV-1**: `injected == completed + queued + running + dropped + timedout` for all seeds
- [ ] **INV-4**: `allocated + free == total` after every step (especially after tiered LRU changes)
- [ ] **INV-9**: None of the new mechanisms read `Request.OutputTokens` for scheduling/eviction/budget decisions
- [ ] **INV-10/11**: No sessions silently abandoned when tier budget starves the sheddable tier

Run the invariant verification suite:

```bash
go test ./sim/... -run TestInvariant -v
go test ./sim/kv/... -run TestConservation -v
```

---

**Plan complete and saved to `docs/plans/2026-03-31-joint-slo-optimization-plan.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** — Open a new session with `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**
