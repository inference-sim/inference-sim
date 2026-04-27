# Priority Preemption Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `selectPriorityVictim()` victim selection, fix Phase 1 index drift for non-tail eviction, and document INV-12 — making `--preemption-policy priority` functional.

**The problem today:** #1168 added the `PreemptionPolicy` type and `--preemption-policy` CLI flag, but `"priority"` is a registered-but-no-op value. `VLLMBatchFormation` always evicts the batch tail (`self.running.pop()` analog). There is no SLO-aware victim selection, and the Phase 1 loop has no index correction for non-tail eviction.

**What this PR adds:**
1. `preemptionPolicy` and `sloMap` fields on `VLLMBatchFormation` — configured via `NewBatchFormation(preemptionPolicy string, sloMap *SLOPriorityMap)`.
2. `selectPriorityVictim()` — finds `min(SLOPriority)` with `max(ArrivalTime)` tiebreak (analog of vLLM `scheduler.py:827-829` `max(priority, arrival_time)` with inverted convention).
3. `preemptForTokens` signature change: `(req, numNewTokens, result, ctx, tokenBudget, reqIndex) → (bool, int)` — returns `reqAdjustment` counting evictions at indices below `reqIndex`.
4. Phase 1 index correction: `reqIndex -= adj` after each `preemptForTokens` call (analog of vLLM `scheduler.py:853` `req_index -= 1`).
5. `NewSimulator` wired to use `cfg.PreemptionPolicy`.
6. INV-12 (Phase 1 Completeness Under Priority Preemption) documented.

**Why this matters:** With `--preemption-policy priority`, BLIS evicts the least-urgent request under KV pressure — matching vLLM's `--scheduling-policy priority` preemption behavior. This enables capacity planners to study how SLO-tier-aware eviction affects tail latency for high-priority requests.

**Architecture:** The change is entirely within `sim/batch_formation.go` (algorithm) and `sim/simulator.go` (1-line wiring). `VLLMBatchFormation` gains two fields; `preemptForTokens` gains a parameter and a second return value; a new `selectPriorityVictim` method does the priority lookup. `SLOPriorityMap` (from `sim/admission.go`, merged in #1013) provides the priority mapping. `nil` sloMap → `DefaultSLOPriorityMap()`.

**Source:** GitHub issue #1169 (part 2 of #1086 decomposition: #1168 → #1169 → #1170).
**Closes:** Fixes #1169.
**Behavioral Contracts:** See Part 1, Section B.

**vLLM reference:** `vllm/vllm/v1/core/sched/scheduler.py` — lines 826–860 (preemption path), line 853 (`req_index -= 1`).

---

## Phase 0: Component Context

1. **Building block modified:** `VLLMBatchFormation` in `sim/batch_formation.go` (preemption strategy), `NewBatchFormation()` constructor, `NewSimulator()` wiring.
2. **Adjacent blocks:** `SLOPriorityMap` in `sim/admission.go` (read-only dependency); `PolicyConfig.PreemptionPolicy` (read from `SimConfig` in `NewSimulator`). `cmd/` is untouched — #1168 already wired the flag.
3. **Invariants touched:** INV-4 (KV conservation — preemption must free blocks), INV-8 (work-conserving — preemption must not leave work idle). New: INV-12 (Phase 1 Completeness).
4. **Construction Site Audit — `NewBatchFormation()`:**
   - Definition: `sim/batch_formation.go:265`
   - Production: `sim/simulator.go:145`
   - Tests: `sim/batch_formation_test.go` (12 call sites: lines 19, 55, 109, 165, 232, 288, 334, 444, 615, 651, 687, 726)
   - Tests: `sim/simulator_preempt_test.go` (2 call sites: lines 18, 73)
   - Direct struct literal: `sim/batch_formation_test.go:420` (`&VLLMBatchFormation{}`)
   
   Total: 15 call sites + 1 struct literal + 1 definition = 17 sites.

---

## Part 1: Design Validation

### A) Executive Summary

This PR makes `--preemption-policy priority` functional by adding SLO-tier-aware victim selection to `VLLMBatchFormation`. When KV cache pressure triggers preemption, the `priority` policy selects the running request with the lowest `SLOPriorityMap` value (BLIS convention: lower = less urgent, e.g., background=-3) rather than blindly evicting the tail.

The critical subtlety is the Phase 1 index correction: when a victim at index `i < reqIndex` is removed, all subsequent elements shift left by one. Without `reqIndex -= adj`, the loop skips the element that shifted into the victim's slot. This is exactly vLLM `scheduler.py:853` (`req_index -= 1`), adapted for BLIS's adjustment-counter approach (which handles cascading evictions within a single `preemptForTokens` call).

For FCFS, the victim is always at the tail (`>= reqIndex`), so `adj` is always 0 — zero behavioral change in default mode.

### B) Behavioral Contracts

**Positive contracts:**

**BC-1: FCFS tail eviction preserved**
- GIVEN `--preemption-policy fcfs` (default) and 3 running requests [first, second, third]
- WHEN KV pressure triggers preemption during Phase 1
- THEN `third` (tail) is evicted, matching the pre-PR behavior exactly

**BC-2: Priority evicts least-urgent regardless of position**
- GIVEN `--preemption-policy priority` and 3 running requests with different SLO tiers
- WHEN KV pressure triggers preemption
- THEN the request with the lowest `SLOPriorityMap` value is evicted, regardless of its position in the batch (head, middle, or tail)

**BC-3: Priority tiebreak by latest arrival**
- GIVEN `--preemption-policy priority` and 3 running requests with equal SLO tier
- WHEN KV pressure triggers preemption
- THEN the most recently arrived request (`max(ArrivalTime)`) is evicted first

**BC-4: KV conservation under priority preemption**
- GIVEN `--preemption-policy priority` and preemption occurs
- WHEN the evicted request's KV blocks are released
- THEN `used_blocks <= total_capacity` at all times (INV-4)

**BC-5: Phase 1 completeness (INV-12)**
- GIVEN `--preemption-policy priority` and a non-tail eviction where `victimIdx < reqIndex`
- WHEN Phase 1 continues after the eviction
- THEN ALL remaining non-preempted running requests receive their decode tokens (`NumNewTokens > 0`), with no request silently skipped due to index drift

**BC-6: Multi-eviction ordering**
- GIVEN `--preemption-policy priority` and cascading preemptions needed
- WHEN multiple victims are evicted in one `preemptForTokens` call
- THEN victims are evicted in non-decreasing urgency order (least urgent first)

**Negative contracts:**

**NC-1: Empty batch no-panic**
- GIVEN `--preemption-policy priority` and an empty running batch
- WHEN a request too large for cache is dequeued in Phase 2
- THEN the circuit breaker fires (returns false), no panic

**NC-2: Self-preemption guard**
- GIVEN `--preemption-policy priority` and the current request IS the least urgent
- WHEN `selectPriorityVictim` selects the current request as victim
- THEN the `preemptedRequest == req` guard fires, the request goes to WaitQ, and Phase 1 breaks

### C) Component Interaction

```
sim/admission.go (SLOPriorityMap — read-only dependency)
  └─ .Priority(class) → int

sim/batch_formation.go (this PR modifies)
  ├─ VLLMBatchFormation struct { preemptionPolicy, sloMap }
  ├─ selectPriorityVictim(requests) → victimIdx
  ├─ preemptForTokens(req, numNewTokens, result, ctx, budget, reqIndex) → (bool, int)
  │   └─ switch preemptionPolicy:
  │       case priority: victimIdx = selectPriorityVictim()
  │       default:       victimIdx = len-1 (tail)
  │   └─ if victimIdx < reqIndex-adjustment: adjustment++
  ├─ FormBatch Phase 1: reqIndex -= adj after each preemptForTokens call
  └─ NewBatchFormation(preemptionPolicy string, sloMap *SLOPriorityMap) → BatchFormation

sim/simulator.go
  └─ NewSimulator: batchFormation = NewBatchFormation(cfg.PreemptionPolicy, nil)
     (nil sloMap → DefaultSLOPriorityMap(); #1170 threads the real map)
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #1169: "7 tests" | Plan has 8 new tests (BC-1 through BC-6, NC-1, NC-2) | ADDITION — BC-1 (FCFS tail test) was counted separately in earlier discussion but is part of this PR's scope |
| vLLM `scheduler.py:831`: `self.running.remove(preempted_req)` | BLIS uses `append([:victimIdx], [victimIdx+1:]...)` | CLARIFICATION — Go slice idiom for remove-by-index; semantically identical |
| vLLM `scheduler.py:853`: `req_index -= 1` (single decrement) | BLIS uses cumulative `adjustment` counter | CLARIFICATION — BLIS's preemptForTokens may evict multiple victims in one call (cascading); the counter accumulates, whereas vLLM's loop re-enters the while loop with the decremented index |
| vLLM `scheduler.py:853`: `req_index -= 1` conditional on `preempted_req in scheduled_running_reqs` | BLIS `adjustment++` fires unconditionally for any victim below `reqIndex` | CLARIFICATION — different loop structures make different conditionality correct. vLLM's `continue` paths always increment `req_index` first (no unscheduled request below cursor). BLIS allows zero-token requests below `reqIndex` (MaxModelLen cap skips `preemptForTokens` call). Unconditional adjustment is correct for BLIS. |
| #1169: "sloMap *SLOPriorityMap" | `NewBatchFormation(preemptionPolicy, nil)` in simulator.go | DEFERRAL — #1170 threads the real `SLOPriorityMap` from `DeploymentConfig`; this PR uses `nil` → `DefaultSLOPriorityMap()` |

### E) Review Guide

**Tricky part:** The `adjustment` counter logic in `preemptForTokens`. The condition `if victimIdx < reqIndex-adjustment` must account for prior adjustments within the same call (cascading evictions). Trace through `TestPreemption_Priority_Phase1Completeness` mentally.

**Scrutinize:** The `selectPriorityVictim` inversion: vLLM uses `max()` (higher=less urgent), BLIS uses iteration with `pri < victimPri` (lower=less urgent). Verify the comparison direction by checking GAIE defaults: `background=-3 < standard=3`, so `min()` selects background — correct.

**Safe to skim:** The 15 `NewBatchFormation()` → `NewBatchFormation("", nil)` call site updates. These are mechanical; the compiler catches misses.

**Known debt:** `NewBatchFormation(cfg.PreemptionPolicy, nil)` — the `nil` is replaced by a real `SLOPriorityMap` in #1170.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to **modify** (no new files):
- `sim/batch_formation.go` — `VLLMBatchFormation` fields, `selectPriorityVictim()`, `preemptForTokens` signature + adj logic, Phase 1 `reqIndex -= adj`, `NewBatchFormation(policy, sloMap)`, comments
- `sim/batch_formation_test.go` — 12 `NewBatchFormation()` → `("", nil)` call sites + 1 `&VLLMBatchFormation{}` struct literal + `makeRunningRequest` helper + 8 new tests
- `sim/simulator.go` — 1 `NewBatchFormation()` → `NewBatchFormation(cfg.PreemptionPolicy, nil)`
- `sim/simulator_preempt_test.go` — 2 `NewBatchFormation()` → `("", nil)` call sites
- `docs/contributing/standards/invariants.md` — INV-12 section
- `docs/contributing/extension-recipes.md` — update `NewBatchFormation` factory signature + preemption variant recipe
- `CLAUDE.md` — Recent Changes entry + INV-12

### G) Task Breakdown

---

#### Task 1: Core algorithm + NewBatchFormation signature change + all call sites (BC-1, BC-5)

This is the largest task — it changes `VLLMBatchFormation`, `preemptForTokens`, the Phase 1 loop, `NewBatchFormation()`, and all 17 construction sites. It must be done atomically because `NewBatchFormation()` signature change causes compile errors until all sites are updated.

**Contracts:** BC-1, BC-5 (structurally enabled), NC-1 (circuit breaker unchanged)
**Files:** modify `sim/batch_formation.go`, `sim/batch_formation_test.go`, `sim/simulator.go`, `sim/simulator_preempt_test.go`

**Step 1: Write the failing test — `TestPreemption_FCFS_EvictsTail`**

Add at end of `sim/batch_formation_test.go`:

```go
func TestPreemption_FCFS_EvictsTail(t *testing.T) {
	// 10 blocks × 16 tokens/block = 160 token capacity.
	// 3 running requests × 3 blocks each = 9 blocks used, 1 block free.
	// Phase 1: "first" gets its decode block (uses the 1 free block).
	// Phase 1: "second" needs decode but cache is now full → preemption.
	// FCFS evicts tail = "third" (BC-1).
	kvCache := MustNewKVCacheState(10, 16)
	running := []*Request{
		{ID: "first", SLOClass: "critical", ArrivalTime: 100, State: StateRunning,
			InputTokens: make([]int, 48), OutputTokens: make([]int, 10)},
		{ID: "second", SLOClass: "standard", ArrivalTime: 200, State: StateRunning,
			InputTokens: make([]int, 48), OutputTokens: make([]int, 10)},
		{ID: "third", SLOClass: "background", ArrivalTime: 300, State: StateRunning,
			InputTokens: make([]int, 48), OutputTokens: make([]int, 10)},
	}
	for _, req := range running {
		kvCache.AllocateKVBlocks(req, 0, 48, nil)
		req.ProgressIndex = 48
	}
	newReq := &Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued}
	wq := &WaitQueue{}
	wq.Enqueue(newReq)

	bf := NewBatchFormation("fcfs", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: running},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx)

	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	if result.Preempted[0].Request.ID != "third" {
		t.Errorf("FCFS preemption: evicted %q, want \"third\" (tail)", result.Preempted[0].Request.ID)
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./sim/ -run TestPreemption_FCFS_EvictsTail -v 2>&1 | tail -10
```

Expected: FAIL — `NewBatchFormation` takes 0 args, not 2.

**Step 3: Implement all changes atomically**

**3a.** In `sim/batch_formation.go`, update `VLLMBatchFormation` struct (line ~70):

```go
type VLLMBatchFormation struct {
	preemptionPolicy PreemptionPolicy
	sloMap           *SLOPriorityMap
}
```

**3b.** Update Phase 1 loop comment (line ~90-95), replace existing comment block:

```go
	// Phase 1: Process continuing requests (chunked prefill + decode).
	// Index-based loop: re-evaluates len() each iteration so evicted requests
	// are never visited. In priority mode, non-tail eviction shifts elements
	// left; the reqAdjustment returned by preemptForTokens compensates by
	// decrementing reqIndex, preventing element skipping.
	// Analog of vLLM v1 req_index -= 1 (scheduler.py:853).
```

**3c.** Update both `preemptForTokens` call sites in Phase 1 (chunked prefill ~line 120, decode ~line 141):

For chunked prefill:
```go
			canSchedule, adj := v.preemptForTokens(req, numNewTokens, &result, ctx, &tokenBudget, reqIndex)
			reqIndex -= adj
			if !canSchedule {
				break
			}
```

For decode:
```go
				canSchedule, adj := v.preemptForTokens(req, decodeTokens, &result, ctx, &tokenBudget, reqIndex)
				reqIndex -= adj
				if !canSchedule {
					break
				}
```

**3d.** Rewrite `preemptForTokens` (starting at line ~213):

```go
// preemptForTokens tries to allocate numNewTokens of KV blocks for req,
// evicting victims if needed. Returns (canSchedule, reqAdjustment) where
// reqAdjustment counts evictions at indices below reqIndex.
// The caller must apply reqIndex -= reqAdjustment after each call to prevent
// element skipping when non-tail removal shifts elements left.
// Analog of vLLM scheduler.py:853 (req_index -= 1).
func (v *VLLMBatchFormation) preemptForTokens(req *Request, numNewTokens int64, result *BatchResult, ctx BatchContext, tokenBudget *int64, reqIndex int) (bool, int) {
	adjustment := 0
	for {
		if ok := ctx.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, nil); !ok {
			// Circuit breaker: empty batch means cache is too small (R19)
			if len(result.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					ctx.Now, req.ID, numNewTokens)
				return false, adjustment
			}

			result.PreemptionHappened = true

			var victimIdx int
			switch v.preemptionPolicy {
			case PreemptionPriority:
				victimIdx = v.selectPriorityVictim(result.RunningBatch.Requests)
			default:
				victimIdx = len(result.RunningBatch.Requests) - 1
			}

			preemptedRequest := result.RunningBatch.Requests[victimIdx]
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room", ctx.Now, preemptedRequest.ID)

			// Remove by index (supports non-tail eviction in priority mode).
			result.RunningBatch.Requests = append(
				result.RunningBatch.Requests[:victimIdx],
				result.RunningBatch.Requests[victimIdx+1:]...,
			)

			// Track Phase 1 index adjustment: if the victim was before the
			// caller's current position, elements shifted left under the cursor.
			// The caller must decrement reqIndex by the returned adjustment.
			// Analog of vLLM scheduler.py:853 (req_index -= 1 when preempted
			// request was in scheduled_running_reqs).
			// For FCFS, victimIdx is always the tail (>= reqIndex), so adjustment
			// is always 0 — preserving current FCFS behavior exactly.
			//
			// Divergence from vLLM: vLLM's req_index -= 1 (scheduler.py:853) is
			// conditional on `preempted_req in scheduled_running_reqs`. BLIS's
			// adjustment fires unconditionally for any victim below reqIndex.
			// This is correct for BLIS because a MaxModelLen-capped request
			// (decodeTokens=0, never calls preemptForTokens) can remain in the
			// batch below reqIndex without being "scheduled." If evicted as a
			// priority victim, the index shift still needs compensation. In vLLM,
			// this scenario cannot arise because all skip paths (lines 743/759/808)
			// increment req_index before continue, so unscheduled requests are
			// never below req_index when preemption fires.
			if victimIdx < reqIndex-adjustment {
				adjustment++
			}

			result.Preempted = append(result.Preempted, PreemptedRequest{
				Request: preemptedRequest,
			})

			// Restore token budget if preempted request was already scheduled
			// in this step (visited earlier in Phase 1, NumNewTokens > 0).
			// Reachable in priority mode when victim was at index < reqIndex
			// (already visited and allocated tokens this step).
			// With FCFS (tail-only eviction), unreachable because evicted
			// requests are always unvisited (beyond reqIndex).
			if preemptedRequest.NumNewTokens > 0 {
				*tokenBudget += int64(preemptedRequest.NumNewTokens)
				preemptedRequest.NumNewTokens = 0
			}

			preemptedRequest.State = StateQueued
			preemptedRequest.ProgressIndex = 0
			preemptedRequest.ITL = nil
			preemptedRequest.TTFTSet = false
			ctx.KVCache.ReleaseKVBlocks(preemptedRequest)
			delete(ctx.ComputedTokens, preemptedRequest.ID)
			ctx.WaitQ.PrependFront(preemptedRequest)

			if preemptedRequest == req {
				return false, adjustment
			}
		} else {
			return true, adjustment
		}
	}
}
```

**3e.** Add `selectPriorityVictim` method after `preemptForTokens`:

```go
// selectPriorityVictim returns the index of the least-urgent running request.
// Least urgent = lowest SLOPriorityMap value (BLIS convention: higher = more urgent).
// Ties broken by latest ArrivalTime (most recently arrived evicted first).
//
// This is the BLIS equivalent of vLLM's max(priority, arrival_time) (scheduler.py:827-829).
// The conventions are inverted: vLLM uses lower=more-urgent so max() selects least urgent;
// BLIS uses higher=more-urgent so min() selects least urgent. Both tiebreak by max(ArrivalTime).
func (v *VLLMBatchFormation) selectPriorityVictim(requests []*Request) int {
	victimIdx := len(requests) - 1
	victimPri := v.sloMap.Priority(requests[victimIdx].SLOClass)
	victimArrival := requests[victimIdx].ArrivalTime

	for i := len(requests) - 2; i >= 0; i-- {
		pri := v.sloMap.Priority(requests[i].SLOClass)
		if pri < victimPri || (pri == victimPri && requests[i].ArrivalTime > victimArrival) {
			victimIdx = i
			victimPri = pri
			victimArrival = requests[i].ArrivalTime
		}
	}
	return victimIdx
}
```

**3f.** Update `NewBatchFormation()` (line ~263):

```go
// NewBatchFormation creates the default BatchFormation.
// preemptionPolicy selects victim strategy: "fcfs" (tail-of-batch) or "priority" (least-urgent SLO tier).
// sloMap provides SLO class → priority mapping for "priority" mode; nil uses GAIE defaults.
func NewBatchFormation(preemptionPolicy string, sloMap *SLOPriorityMap) BatchFormation {
	policy := PreemptionPolicy(preemptionPolicy)
	if policy == "" {
		policy = PreemptionFCFS
	}
	if sloMap == nil {
		sloMap = DefaultSLOPriorityMap()
	}
	return &VLLMBatchFormation{
		preemptionPolicy: policy,
		sloMap:           sloMap,
	}
}
```

**3g.** Update `sim/simulator.go:145`:
```go
	batchFormation := NewBatchFormation(cfg.PreemptionPolicy, nil)
```

**3h.** Update all 12 test call sites in `sim/batch_formation_test.go`:
Replace `NewBatchFormation()` with `NewBatchFormation("", nil)` at lines: 19, 55, 109, 165, 232, 288, 334, 444, 615, 651, 687, 726.

**3i.** Update the struct literal at `sim/batch_formation_test.go:420`:
```go
	bf := &VLLMBatchFormation{preemptionPolicy: PreemptionFCFS, sloMap: DefaultSLOPriorityMap()}
```

Also update the `preemptForTokens` call at line ~430 to pass `reqIndex`:
```go
	bf.preemptForTokens(newReq, 16, &result, ctx, &budget, 0)
```

**3j.** Update 2 call sites in `sim/simulator_preempt_test.go` (lines 18, 73):
Replace `NewBatchFormation()` with `NewBatchFormation("", nil)`.

**Step 4: Run tests to verify they pass**

```bash
go test ./sim/ -run "TestPreemption_FCFS_EvictsTail|TestVLLMBatchFormation|TestPreemptForTokens" -v 2>&1 | tail -30
go build ./... 2>&1
```

Expected: all PASS, clean build.

**Step 5: Run lint**

```bash
golangci-lint run ./sim/... 2>&1 | head -20
```

**Step 6: Commit**

```bash
git add sim/batch_formation.go sim/batch_formation_test.go sim/simulator.go sim/simulator_preempt_test.go
git commit -m "feat(sim): wire VLLMBatchFormation with preemption policy + index adjustment (BC-1, BC-5)

- Add preemptionPolicy and sloMap fields to VLLMBatchFormation
- Add selectPriorityVictim(): min(SLOPriority) with max(ArrivalTime) tiebreak
- Update NewBatchFormation() to accept policy and SLO map (17 sites updated)
- Change preemptForTokens to return (bool, int) with reqIndex parameter
  - Returns reqAdjustment counting evictions at indices below reqIndex
  - Phase 1 callers apply reqIndex -= adj (prevents element skipping)
  - Analog of vLLM scheduler.py:853 req_index -= 1
- FCFS adj always 0 (tail >= reqIndex); existing behavior preserved

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Priority preemption behavioral tests (BC-2, BC-3, BC-4, BC-6, NC-1, NC-2)

**Contracts:** BC-2, BC-3, BC-4, BC-5 (explicit test), BC-6, NC-1, NC-2
**Files:** modify `sim/batch_formation_test.go`

**Step 1: Add `makeRunningRequest` helper and 7 new tests**

Add at end of `sim/batch_formation_test.go` (after `TestPreemption_FCFS_EvictsTail`):

```go
// makeRunningRequest creates a running request with prefill fully allocated.
// Allocates inputLen tokens (ceil(inputLen/blockSize) blocks) then sets ProgressIndex.
// blockSize matches the kvCache blockSize (16 for MustNewKVCacheState(N, 16)).
func makeRunningRequest(id, sloClass string, arrival int64, inputLen int, kvCache KVStore) *Request {
	req := &Request{
		ID:           id,
		SLOClass:     sloClass,
		ArrivalTime:  arrival,
		State:        StateRunning,
		InputTokens:  make([]int, inputLen),
		OutputTokens: make([]int, 10),
	}
	kvCache.AllocateKVBlocks(req, 0, int64(inputLen), nil)
	req.ProgressIndex = int64(inputLen)
	return req
}

func TestPreemption_Priority_EvictsLeastUrgent(t *testing.T) {
	tests := []struct {
		name    string
		order   []string
		wantID  string
	}{
		{"victim at head", []string{"background", "critical", "standard"}, "bg"},
		{"victim in middle", []string{"critical", "background", "standard"}, "bg"},
		{"victim at tail", []string{"critical", "standard", "background"}, "bg"},
	}

	ids := map[string]string{
		"background": "bg",
		"critical":   "crit",
		"standard":   "std",
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kvCache := MustNewKVCacheState(10, 16)
			var running []*Request
			for i, slo := range tt.order {
				req := makeRunningRequest(ids[slo], slo, int64(100*(i+1)), 48, kvCache)
				running = append(running, req)
			}

			wq := &WaitQueue{}
			wq.Enqueue(&Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

			bf := NewBatchFormation("priority", nil)
			ctx := BatchContext{
				RunningBatch:       &Batch{Requests: running},
				WaitQ:              wq,
				KVCache:            kvCache,
				MaxScheduledTokens: 10000,
				MaxRunningReqs:     10,
				Now:                1000,
				ComputedTokens:     make(map[string]int64),
			}

			result := bf.FormBatch(ctx)

			if len(result.Preempted) == 0 {
				t.Fatal("expected preemption but got none")
			}
			if result.Preempted[0].Request.ID != tt.wantID {
				t.Errorf("priority preemption: evicted %q, want %q (background = least urgent, SLO=-3)",
					result.Preempted[0].Request.ID, tt.wantID)
			}
		})
	}
}

func TestPreemption_Priority_SelfPreemption(t *testing.T) {
	kvCache := MustNewKVCacheState(10, 16)
	bg := makeRunningRequest("bg", "background", 100, 64, kvCache)
	crit := makeRunningRequest("crit", "critical", 200, 48, kvCache)
	dummy := makeRunningRequest("dummy", "standard", 300, 48, kvCache)

	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{bg, crit, dummy}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx)

	preemptedIDs := make(map[string]bool)
	for _, p := range result.Preempted {
		preemptedIDs[p.Request.ID] = true
	}
	if !preemptedIDs["bg"] {
		t.Errorf("expected bg to be preempted; got preempted: %v", preemptedIDs)
	}
	found := false
	for _, r := range ctx.WaitQ.Items() {
		if r.ID == "bg" {
			found = true
			break
		}
	}
	if !found {
		t.Error("bg was preempted but not found in WaitQ")
	}
}

func TestPreemption_Priority_TiebreakByLatestArrival(t *testing.T) {
	kvCache := MustNewKVCacheState(10, 16)
	old := makeRunningRequest("old", "standard", 100, 48, kvCache)
	mid := makeRunningRequest("mid", "standard", 200, 48, kvCache)
	new_ := makeRunningRequest("new", "standard", 300, 48, kvCache)

	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "trigger", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{old, mid, new_}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx)

	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	if result.Preempted[0].Request.ID != "new" {
		t.Errorf("tiebreak: evicted %q, want \"new\" (latest arrival ArrivalTime=300)", result.Preempted[0].Request.ID)
	}
}

func TestPreemption_Priority_KVConservation(t *testing.T) {
	kvCache := MustNewKVCacheState(10, 16)
	bg := makeRunningRequest("bg", "background", 100, 96, kvCache)
	crit := makeRunningRequest("crit", "critical", 200, 48, kvCache)

	usedBefore := kvCache.UsedBlocks()

	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "new", InputTokens: make([]int, 48), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{bg, crit}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	bf.FormBatch(ctx)

	usedAfter := kvCache.UsedBlocks()

	if usedAfter > kvCache.TotalCapacity() {
		t.Errorf("INV-4 violated: used=%d > capacity=%d", usedAfter, kvCache.TotalCapacity())
	}
	if usedAfter >= usedBefore {
		t.Errorf("expected blocks freed after priority preemption: before=%d, after=%d",
			usedBefore, usedAfter)
	}
}

func TestPreemption_Priority_EmptyBatch_NoPanic(t *testing.T) {
	kvCache := MustNewKVCacheState(2, 16)
	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "large", InputTokens: make([]int, 200), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                0,
		ComputedTokens:     make(map[string]int64),
	}

	// Must not panic — circuit breaker handles empty batch.
	bf.FormBatch(ctx)
}

func TestPreemption_Priority_Phase1Completeness(t *testing.T) {
	kvCache := MustNewKVCacheState(10, 16)
	bg := makeRunningRequest("bg", "background", 100, 48, kvCache)
	crit := makeRunningRequest("crit", "critical", 200, 48, kvCache)
	std := makeRunningRequest("std", "standard", 300, 48, kvCache)

	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{bg, crit, std}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx)

	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	if result.Preempted[0].Request.ID != "bg" {
		t.Fatalf("wrong victim: got %q, want bg", result.Preempted[0].Request.ID)
	}

	for _, req := range result.RunningBatch.Requests {
		if req.NumNewTokens == 0 {
			t.Errorf("INV-12 violated: running request %q has NumNewTokens=0 — was it skipped after index adjustment?", req.ID)
		}
	}

	runningIDs := make(map[string]bool)
	for _, req := range result.RunningBatch.Requests {
		runningIDs[req.ID] = true
	}
	if !runningIDs["crit"] {
		t.Error("crit should still be in running batch")
	}
	if !runningIDs["std"] {
		t.Error("std should still be in running batch (must NOT be skipped by index drift)")
	}
}

func TestPreemption_Priority_MultiEvictionOrdering(t *testing.T) {
	kvCache := MustNewKVCacheState(4, 16)
	bg := makeRunningRequest("bg", "background", 100, 16, kvCache)
	shed := makeRunningRequest("shed", "sheddable", 200, 16, kvCache)
	crit := &Request{
		ID: "crit", SLOClass: "critical", ArrivalTime: 300, State: StateRunning,
		InputTokens: make([]int, 64), OutputTokens: make([]int, 10),
		ProgressIndex: 0,
	}

	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{crit, bg, shed}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx)

	if len(result.Preempted) < 2 {
		t.Fatalf("expected at least 2 preemptions, got %d", len(result.Preempted))
	}
	if result.Preempted[0].Request.ID != "bg" {
		t.Errorf("1st preemption: got %q, want bg (background=-3)", result.Preempted[0].Request.ID)
	}
	if result.Preempted[1].Request.ID != "shed" {
		t.Errorf("2nd preemption: got %q, want shed (sheddable=-2)", result.Preempted[1].Request.ID)
	}

	if len(result.RunningBatch.Requests) != 1 || result.RunningBatch.Requests[0].ID != "crit" {
		ids := make([]string, len(result.RunningBatch.Requests))
		for i, r := range result.RunningBatch.Requests {
			ids[i] = r.ID
		}
		t.Errorf("expected [crit] in running batch, got %v", ids)
	}
}
```

**Step 2: Run all new tests**

```bash
go test ./sim/ -run "TestPreemption_" -v 2>&1 | tail -30
```

Expected: all PASS.

**Step 3: Run full test suite**

```bash
go test ./... -count=1 2>&1 | tail -15
```

Expected: all PASS.

**Step 4: Run lint**

```bash
golangci-lint run ./sim/... 2>&1 | head -10
```

**Step 5: Commit**

```bash
git add sim/batch_formation_test.go
git commit -m "test(sim): add priority preemption behavioral tests (BC-2 through BC-6, NC-1, NC-2)

- Add makeRunningRequest helper for test setup
- TestPreemption_Priority_EvictsLeastUrgent: table-driven, victim at head/middle/tail (BC-2)
- TestPreemption_Priority_SelfPreemption: self-eviction guard (NC-2)
- TestPreemption_Priority_TiebreakByLatestArrival: equal-priority tiebreak (BC-3)
- TestPreemption_Priority_KVConservation: INV-4 after priority eviction (BC-4)
- TestPreemption_Priority_EmptyBatch_NoPanic: circuit breaker (NC-1)
- TestPreemption_Priority_Phase1Completeness: INV-12, index adjustment (BC-5)
- TestPreemption_Priority_MultiEvictionOrdering: cascading eviction order (BC-6)

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: INV-12 documentation + CLAUDE.md update (BC-5 documentation)

**Contracts:** BC-5 (documentation)
**Files:** modify `docs/contributing/standards/invariants.md`, `CLAUDE.md`, `docs/contributing/extension-recipes.md`

**Step 1: Add INV-12 to `docs/contributing/standards/invariants.md`**

Append before the final `---` or at the end of the invariants list:

```markdown
## INV-12: Phase 1 Completeness Under Priority Preemption

**Statement:** After Phase 1 of `FormBatch` completes, every non-preempted running request in decode phase has `NumNewTokens > 0`, provided the token budget was not exhausted and `MaxModelLen` did not cap the request. No running request is silently skipped due to index drift from non-tail eviction.

**Context:** With `--preemption-policy priority`, the preemption victim may be at any index in the running batch (not just the tail). Removing an element at index `i < reqIndex` shifts subsequent elements left by one. Without the `reqIndex -= adjustment` correction (analog of vLLM `scheduler.py:853` `req_index -= 1`), the Phase 1 loop skips the shifted element.

**Verification:** `sim/batch_formation_test.go` — `TestPreemption_Priority_Phase1Completeness`: verifies that after non-tail eviction where `victimIdx < reqIndex`, ALL remaining running requests receive decode tokens (NumNewTokens > 0). The index adjustment is tested with [bg, crit, std] batch where bg is evicted at index 0 while processing crit at index 1.

**Trivially satisfied for FCFS:** With `--preemption-policy fcfs` (default), victims are always at the batch tail (`victimIdx == len-1 >= reqIndex`), so `adjustment == 0` and no element skipping is possible.

**Hypothesis family:** Structural model (same as INV-4, INV-7, INV-8, INV-9).
```

**Step 2: Update CLAUDE.md Recent Changes**

Add at the top of the Recent Changes section:

```
- Priority preemption mode (#1169): `--preemption-policy priority` evicts the least-urgent running request (min SLOPriorityMap value) under KV pressure instead of the batch tail. `selectPriorityVictim()`: `min(SLOPriority)` with `max(ArrivalTime)` tiebreak — analog of vLLM `scheduler.py:827-829` `max(priority, arrival_time)` with inverted convention. Phase 1 index adjustment (`reqIndex -= adj`) prevents element skipping after non-tail eviction (INV-12, analog of vLLM `scheduler.py:853` `req_index -= 1`). `NewBatchFormation(preemptionPolicy, sloMap)` 2-arg constructor; `nil` sloMap → `DefaultSLOPriorityMap()`.
```

**Step 2b: Update `docs/contributing/extension-recipes.md`**

Update the batch formation recipe (around line 113). Replace the stale reference:

Before:
```
2. **Register in `NewBatchFormation` factory** in `sim/batch_formation.go`: add a selection branch. The factory signature is `NewBatchFormation()` — a future PR will add a strategy selection parameter (e.g., a string field in `PolicyConfig` or `BatchConfig`)
```

After:
```
2. **Register in `NewBatchFormation` factory** in `sim/batch_formation.go`: add a selection branch. The factory signature is `NewBatchFormation(preemptionPolicy string, sloMap *SLOPriorityMap)`. For a new batch formation *strategy* (not just a preemption variant), add a `BatchFormation string` field to `PolicyConfig` and a selection branch in `NewBatchFormation`
```

Also update the Note block (line ~117) to reflect that `PreemptionPolicy` now exists:

Before:
```
**Note:** Currently only `VLLMBatchFormation` exists. Adding a second strategy will also require: (a) a `BatchFormation string` field in `PolicyConfig` or `BatchConfig` (in `sim/config.go`), (b) a CLI flag in `cmd/root.go`, (c) validation in `sim/bundle.go`, (d) selection logic in `NewBatchFormation`.
```

After:
```
**Note:** Currently only `VLLMBatchFormation` exists (with configurable preemption via `--preemption-policy fcfs|priority`). Adding a second batch formation strategy will also require: (a) a `BatchFormation string` field in `PolicyConfig` or `BatchConfig` (in `sim/config.go`), (b) a CLI flag in `cmd/root.go`, (c) validation in `sim/bundle.go`, (d) selection logic in `NewBatchFormation`. For adding a new *preemption* variant (not a new strategy), add a constant to `batch_formation.go`, a case to the `switch` in `preemptForTokens`, and an entry in `validPreemptionPolicies` in `bundle.go`.
```

**Step 2c: Update CLAUDE.md Key Invariants section**

Add INV-12 to the invariants list:

```
- **INV-12 Phase 1 Completeness**: After Phase 1 of `FormBatch`, every non-preempted running request in decode phase has `NumNewTokens > 0`. No request silently skipped due to index drift from non-tail eviction. Trivially satisfied for FCFS. See `docs/contributing/standards/invariants.md`.
```

**Step 3: Commit**

```bash
git add docs/contributing/standards/invariants.md docs/contributing/extension-recipes.md CLAUDE.md
git commit -m "docs: add INV-12 (Phase 1 Completeness) + CLAUDE.md update for #1169

- Add INV-12 to invariants.md: Phase 1 Completeness Under Priority Preemption
- Update CLAUDE.md Recent Changes with priority preemption details
- Add INV-12 to CLAUDE.md Key Invariants section

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Behavioral | `TestPreemption_FCFS_EvictsTail` |
| BC-2 | Task 2 | Behavioral (table) | `TestPreemption_Priority_EvictsLeastUrgent` |
| BC-3 | Task 2 | Behavioral | `TestPreemption_Priority_TiebreakByLatestArrival` |
| BC-4 | Task 2 | Invariant | `TestPreemption_Priority_KVConservation` |
| BC-5 | Task 2 | Invariant | `TestPreemption_Priority_Phase1Completeness` |
| BC-6 | Task 2 | Behavioral | `TestPreemption_Priority_MultiEvictionOrdering` |
| NC-1 | Task 2 | Negative | `TestPreemption_Priority_EmptyBatch_NoPanic` |
| NC-2 | Task 2 | Negative | `TestPreemption_Priority_SelfPreemption` |

All tests are behavioral: they assert observable outcomes (which request was evicted, which remain running, whether blocks were freed) — not internal structure. They would survive a complete reimplementation of the eviction algorithm as long as the behavioral contracts hold.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|-----------|------|
| `selectPriorityVictim` inversion (min vs max) | Medium | High | Explicit comment documenting the inversion; BC-2 table test covers all positions; GAIE default values make the direction verifiable (`background=-3 < standard=3` → `min` selects background) | Task 1 |
| Phase 1 index skipping without `adj` | High (by design) | High | BC-5 `TestPreemption_Priority_Phase1Completeness` specifically tests the [bg,crit,std] scenario where skipping occurs without the fix | Task 2 |
| Cascading eviction: `adjustment` counter off-by-one | Low | High | BC-6 `TestPreemption_Priority_MultiEvictionOrdering` cascades 2 evictions and verifies ordering | Task 2 |
| `NewBatchFormation` call site missed | Low | High (compile error) | Compiler enforces signature change; Phase 0 audit lists all 17 sites | Task 1 |
| `nil` sloMap causes panic in `selectPriorityVictim` | Low | High | `NewBatchFormation` replaces `nil` with `DefaultSLOPriorityMap()` before storing | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `selectPriorityVictim` is a private method, not an interface
- [x] No feature creep — `sloMap` override deferred to #1170
- [x] No unexercised code — every new code path tested by BC-1 through BC-6 and NC-1/NC-2
- [x] No partial implementations — `"priority"` is fully functional after this PR
- [x] No breaking changes — `NewBatchFormation("", nil)` preserves FCFS behavior exactly
- [x] CLAUDE.md updated with INV-12 + Recent Changes in Task 3
- [x] Documentation DRY — `invariants.md` is the canonical source; CLAUDE.md is a working copy
- [x] Deviation log complete — 4 deviations documented
- [x] Task dependencies: Task 2 depends on Task 1 (uses `NewBatchFormation("priority", nil)`); Task 3 standalone
- [x] All contracts mapped to tasks
- [x] Construction site audit: 15 call sites + 1 struct literal + 1 definition = 17, all covered

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — all paths return values or break with logs
- [x] R4: All 17 `NewBatchFormation` sites updated (compiler enforced)
- [x] R5: `preemptForTokens` loop handles mid-loop failure (circuit breaker returns)
- [x] R6: No `logrus.Fatalf` in `sim/` — only `logrus.Warnf`
- [x] R8: No exported mutable maps
- [x] R14: `selectPriorityVictim` is a single-concern method (victim selection only)
- [x] R19: Circuit breaker for empty batch preserved
- [x] R21: No range over mutable slices — `append([:i], [i+1:]...)` creates a new slice header

---

## Appendix: File-Level Implementation Details

### `sim/batch_formation.go`

**Purpose:** Core algorithm changes.

**Key implementation notes:**
- **`selectPriorityVictim`:** Iterates from tail to head. Uses `pri < victimPri` (not `>`) because BLIS convention is higher=more-urgent. Iterating from tail means the first-found lowest priority at the highest index wins ties on arrival time (latest arrival, since we use `>` for arrival comparison).
- **`preemptForTokens` adjustment counter:** `if victimIdx < reqIndex-adjustment` accounts for prior adjustments within the same call. The `adjustment` variable accumulates; the caller reads the final value once.
- **Remove-by-index:** `append([:victimIdx], [victimIdx+1:]...)` instead of `[:len-1]` truncation. This supports non-tail eviction while being a no-op for tail eviction (append of empty slice).
- **Token budget restoration:** Now reachable in priority mode (victim at index < reqIndex was already visited and had NumNewTokens > 0). The comment is updated from "unreachable" to "reachable in priority mode."

### `sim/simulator.go`

**Purpose:** Wire `cfg.PreemptionPolicy` into `NewBatchFormation`.

**One-line change:** `NewBatchFormation()` → `NewBatchFormation(cfg.PreemptionPolicy, nil)`. The `nil` sloMap is replaced by the real map in #1170.

### `docs/contributing/standards/invariants.md`

**Purpose:** Document INV-12 (Phase 1 Completeness Under Priority Preemption).

**Key points:** States the invariant, explains the index drift mechanism, names the verification test, notes FCFS trivially satisfies it, classifies as structural-model family.
