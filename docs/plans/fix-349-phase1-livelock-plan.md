# Fix Phase 1 Range Loop Livelock (#349) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix a cascading preemption livelock in the batch formation scheduler that prevents any request from completing under KV cache pressure.

**The problem today:** When KV cache is constrained and running requests trigger preemption during Phase 1 of batch formation, Go's `range` loop continues visiting requests that were already evicted. Evicted requests have `ProgressIndex` reset to 0, causing them to attempt full re-prefill allocation — triggering further preemptions of still-running requests. This cascading amplification produces 100K+ preemptions with 0 completions over a 120s horizon. Real vLLM avoids this because evicted requests are removed from the `running` deque and never revisited — achieving the behavioral property that preempted requests don't cascade within a single scheduling pass.

**What this PR adds:**

1. **Index-based Phase 1 loop** — replaces `for _, req := range` with `for reqIndex < len(...)` that re-evaluates slice length each iteration, achieving the same behavioral property as vLLM: evicted requests are never revisited. (Note: vLLM v1 uses `deque.popleft()/pop()` internally, not index-based iteration. BLIS achieves the same *behavior* via a different mechanism.)
2. **`NumNewTokens` zeroing at FormBatch entry** — clears stale per-step scheduling state from the previous step to prevent phantom budget restoration.

**Why this matters:** This is a correctness bug (R5 state corruption, R19 livelock, INV-1/INV-4 violations) that makes BLIS unusable for any workload where KV pressure triggers preemption in Phase 1. The fix restores correct scheduling behavior.

**Architecture:** Changes are entirely within `VLLMBatchFormation.FormBatch()` and `preemptForTokens()` in `sim/batch_formation.go`. No interface changes, no new types, no kernel changes. The `BatchFormation` interface, `BatchContext`, and `BatchResult` are untouched.

**Source:** GitHub issue #349 (bug: Phase 1 range loop visits evicted requests, causing cascading preemption livelock under KV pressure). Design comment: https://github.com/inference-sim/inference-sim/issues/349#issuecomment-4005101679

**Closes:** Fixes #349

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a bug in `VLLMBatchFormation.FormBatch()` Phase 1 where Go's `range` loop captures the slice at entry and continues iterating over requests that have been evicted by `preemptForTokens`. Evicted requests (with `ProgressIndex` reset to 0) attempt full re-prefill, cascading preemptions across the entire running batch. The fix replaces `range` with an index-based loop that re-evaluates `len(result.RunningBatch.Requests)` each iteration — achieving the same behavioral property as vLLM v1 (evicted requests are never revisited within a scheduling pass), though through a different mechanism (vLLM uses deque popleft/pop, BLIS uses index bounds re-evaluation).

The fix also zeros `NumNewTokens` for all running requests at `FormBatch` entry to prevent stale per-step state from causing phantom budget restoration. No interfaces or types change. Adjacent components (`Simulator.scheduleBatch`, `KVStore`, `WaitQueue`) are unaffected.

Golden dataset has 0 entries with preemptions, so no golden regeneration is needed.

### B) Behavioral Contracts

**Positive Contracts:**

FIX-1: Index-Based Phase 1 Iteration
- GIVEN a running batch of N requests where preemption evicts the tail request
- WHEN Phase 1 processes the running batch
- THEN only the remaining (N-1) requests are visited; the evicted request is NOT processed by Phase 1
- MECHANISM: `for reqIndex < len(result.RunningBatch.Requests)` re-evaluates length after each preemption. Tail eviction shortens the slice; `reqIndex` is never incremented past the new length.

FIX-2: Stale NumNewTokens Prevention
- GIVEN running requests that may carry `NumNewTokens > 0` from the previous step
- WHEN FormBatch begins a new scheduling pass
- THEN all running requests have `NumNewTokens = 0` before Phase 1 iteration starts
- MECHANISM: Explicit zeroing loop at FormBatch entry, before Phase 1.

FIX-3: KV Pressure Completion (Livelock Resolution)
- GIVEN 30 requests with output tokens 3200-3596, 7463 KV blocks (block size 16), seed 7
- WHEN the simulation runs for 120s horizon
- THEN completed_requests > 0 AND preemption_count < 1000
- MECHANISM: Index-based loop prevents cascading amplification; running batch stabilizes at ~31 concurrent requests

**Negative Contracts:**

FIX-4: No State Corruption
- GIVEN a request evicted by preemptForTokens (State=Queued, ProgressIndex=0, in WaitQ)
- WHEN Phase 1 continues iterating
- THEN the evicted request MUST NOT have KV blocks allocated by Phase 1 (no dual allocation)
- MECHANISM: Index-based loop terminates before reaching evicted indices

FIX-5: Backward Compatibility (No Preemption Path)
- GIVEN workloads where no preemption occurs (e.g., all 5 golden dataset entries)
- WHEN FormBatch is called
- THEN results are byte-identical to the current implementation
- MECHANISM: Index-based loop visits the same requests in the same order when no evictions occur. NumNewTokens zeroing has no effect when values are immediately overwritten.

**Error Handling Contracts:**

FIX-6: Circuit Breaker Preserved
- GIVEN a request that needs more KV blocks than total capacity (existing R19 guard)
- WHEN preemptForTokens evicts all running requests and the batch is empty
- THEN preemptForTokens returns false without panic
- MECHANISM: Existing `len(result.RunningBatch.Requests) == 0` check unchanged

### C) Component Interaction

```
Simulator.scheduleBatch()
    │
    ▼
BatchFormation.FormBatch(BatchContext) → BatchResult
    │
    ├── Zero NumNewTokens for all running requests ← NEW
    │
    ├── Phase 1: Process running requests (index-based loop) ← CHANGED
    │       │
    │       └── preemptForTokens(req, tokens, result, ctx, tokenBudget)  ← CHANGED
    │               │
    │               ├── KVStore.AllocateKVBlocks()
    │               ├── KVStore.ReleaseKVBlocks()
    │               └── WaitQueue.PrependFront()
    │
    └── Phase 2: Dequeue from WaitQ (unchanged)
```

**API contracts:** No interface changes. `FormBatch` signature, `BatchContext`, `BatchResult` all unchanged. `preemptForTokens` is a private method — signature changes are internal.

**State changes:** `preemptForTokens` now receives `*int64` for token budget (was not accessible before). `NumNewTokens` explicitly zeroed at FormBatch entry. No new mutable state.

**Extension friction:** 0 new files. This fix does not add fields to any struct, so R4 construction-site audit is N/A.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue #349 proposes "preemption backoff" and "admission backpressure" as options | Fix addresses root cause (range loop) instead | CORRECTION: The proposed options were workarounds for a symptom; the root cause is the Go range loop visiting evicted requests |
| Design comment suggests matching "vLLM's `while req_index < len(self.running)` semantics" | Plan describes achieving the same behavioral property via index-based loop | CORRECTION: vLLM v1 actually uses `deque.popleft()/pop()`, not while-loop. BLIS achieves the same behavior (evicted not revisited) via different mechanism |
| Design comment proposes `*int64` tokenBudget for budget restoration | Plan adds `*int64` plus zeroing of `NumNewTokens` at FormBatch entry | ADDITION: Zeroing prevents stale per-step state from prior steps causing phantom restoration |
| `preemptForTokens` signature change from 4 to 5 args | Must also update existing `TestPreemptForTokens_CleansUpComputedTokens` caller | ADDITION: Existing direct caller at `batch_formation_test.go:450` must be updated |
| Issue reproducer uses 300 requests | Integration test uses 30 requests | SIMPLIFICATION: 30 requests are sufficient to trigger and verify the livelock fix without 120s test runtime |

### E) Review Guide

**The tricky part:** The interaction between Go's `range` semantics and slice mutation. The `range` captures `(pointer, length)` at loop entry — shortening the slice via `result.RunningBatch.Requests = ...[:len-1]` does NOT affect the range. The index-based `for reqIndex < len(...)` re-evaluates `len()` each iteration, which IS affected by the shortening. Additionally, `NumNewTokens` stale state from previous steps: without zeroing, preempting an unvisited request that carries `NumNewTokens=1` from the prior step would incorrectly inflate the token budget.

**What to scrutinize:** FIX-3 (the integration test) — does the reproducer actually complete requests now? The `NumNewTokens` zeroing loop — is it placed correctly (before Phase 1, after nil check)?

**What's safe to skim:** Phase 2 is completely unchanged. The circuit breaker (FIX-6) is unchanged. Golden dataset is unaffected (0 preemption entries).

**Known debt:** #518 (BatchContext/BatchResult R13 generalization for multi-engine support). Pre-existing: `PrependFront` is O(n) per call (full slice copy); this PR reduces how often it's called. Pre-existing: preemption always uses recompute mode (ProgressIndex=0), not swap mode; vLLM supports both.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files:**
- Modify: `sim/batch_formation.go` — zero NumNewTokens at entry, replace range with index loop, add `*int64` to preemptForTokens
- Modify: `sim/batch_formation_test.go` — add FIX-1/FIX-2/FIX-3 tests, update existing preemptForTokens caller

**Key decisions:**
- Token budget passed as `*int64` to `preemptForTokens` (simplest plumbing, defensive budget restoration for future eviction policy changes)
- `NumNewTokens` zeroed at FormBatch entry (prevents stale state from prior steps)
- Integration test uses `mustNewSimulator` helper (existing pattern in test suite)
- Remove stale "do NOT fix this" comment
- Preserve "token budget exhausted" warning inside loop body

**Confirmation:** No dead code. All changes exercised by existing + new tests.

### G) Task Breakdown

---

### Task 1: Test + Fix — Phase 1 must not visit evicted requests (FIX-1, FIX-2, FIX-4)

**Contracts Implemented:** FIX-1, FIX-2, FIX-4

**Files:**
- Modify: `sim/batch_formation.go`
- Modify: `sim/batch_formation_test.go`

**Step 1: Write failing test**

Context: We construct a scenario where Phase 1 must preempt the tail request. With the current `range` loop, the evicted request gets revisited and triggers further preemptions. The test verifies that after one preemption, only the remaining (non-evicted) requests are in the result batch — and no state corruption occurs.

```go
// TestVLLMBatchFormation_Phase1_EvictedNotRevisited verifies FIX-1 and FIX-4:
// Phase 1 must not visit requests that were evicted by preemptForTokens.
// The old range-based loop continued iterating over evicted requests (ProgressIndex=0),
// causing cascading re-prefill allocations and state corruption.
func TestVLLMBatchFormation_Phase1_EvictedNotRevisited(t *testing.T) {
	// 6 blocks * 16 tokens = 96 token capacity
	cfg := SimConfig{
		KVCacheConfig:       NewKVCacheConfig(6, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{0, 0, 0}, []float64{100, 1, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, ""),
	}
	lm, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(lm)
	kvCache := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)

	// GIVEN 3 running requests, all in decode phase with KV fully allocated:
	// r1 uses 3 blocks (48 tokens, exact multiple of 16 → last block full)
	// r2 uses 2 blocks (31 tokens, partial last block: 15/16 filled)
	// r3 uses 1 block (16 tokens, exact multiple → last block full)
	// Total: 6 blocks = full cache.
	// r1's decode needs a NEW block (last block full) → triggers preemption of r3 (tail).
	// r2's decode fills its partial block (15→16) → NO new block needed, r2 survives.
	r1 := &Request{ID: "r1", InputTokens: make([]int, 48), OutputTokens: make([]int, 100), State: StateRunning}
	r2 := &Request{ID: "r2", InputTokens: make([]int, 31), OutputTokens: make([]int, 100), State: StateRunning}
	r3 := &Request{ID: "r3", InputTokens: make([]int, 16), OutputTokens: make([]int, 100), State: StateRunning}

	if ok := kvCache.AllocateKVBlocks(r1, 0, 48, []int64{}); !ok {
		t.Fatal("setup: allocate r1")
	}
	r1.ProgressIndex = 48

	if ok := kvCache.AllocateKVBlocks(r2, 0, 31, []int64{}); !ok {
		t.Fatal("setup: allocate r2")
	}
	r2.ProgressIndex = 31

	if ok := kvCache.AllocateKVBlocks(r3, 0, 16, []int64{}); !ok {
		t.Fatal("setup: allocate r3")
	}
	r3.ProgressIndex = 16
	r3.NumNewTokens = 5 // Stale value from prior step — FIX-2 zeroing must clear this

	if kvCache.UsedBlocks() != 6 {
		t.Fatalf("setup: expected 6 used blocks, got %d", kvCache.UsedBlocks())
	}

	computedTokens := map[string]int64{"r1": 48, "r2": 31, "r3": 16}
	ctx := BatchContext{
		RunningBatch:          &Batch{Requests: []*Request{r1, r2, r3}},
		WaitQ:                 &WaitQueue{},
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   5000,
		StepCount:             5,
		ComputedTokens:        computedTokens,
	}

	result := bf.FormBatch(ctx)

	// THEN r3 must be preempted (tail eviction to make room for r1's decode)
	if !result.PreemptionHappened {
		t.Fatal("expected preemption to occur")
	}

	// AND preemption count must be exactly 1 (only r3 evicted — no cascading)
	if len(result.Preempted) != 1 {
		t.Errorf("FIX-1: expected exactly 1 preemption (r3), got %d", len(result.Preempted))
	}

	// AND FIX-2: r3's stale NumNewTokens (5) must have been zeroed at FormBatch entry,
	// so preemption did NOT inflate the budget by 5.
	if len(result.Preempted) == 1 && result.Preempted[0].Request.NumNewTokens != 0 {
		t.Errorf("FIX-2: preempted r3 should have NumNewTokens=0 (zeroed at entry), got %d",
			result.Preempted[0].Request.NumNewTokens)
	}

	// AND r1 and r2 must still be in the running batch, r3 must not
	batchIDs := make(map[string]bool)
	for _, req := range result.RunningBatch.Requests {
		batchIDs[req.ID] = true
	}
	if !batchIDs["r1"] || !batchIDs["r2"] {
		t.Errorf("FIX-1: r1 and r2 must remain in batch, got %v", batchIDs)
	}
	if batchIDs["r3"] {
		t.Error("FIX-4: evicted request r3 must not be in running batch")
	}

	// AND KV conservation must hold (INV-4):
	// r1: 3 blocks (48 tokens) + 1 new block (decode at block boundary) = 4 blocks
	// r2: 2 blocks (31 tokens, decode fills partial block 15→16, no new block)
	// r3: freed (1 block released, used by r1's decode allocation)
	expectedUsed := int64(4 + 2) // r1=4, r2=2
	if kvCache.UsedBlocks() != expectedUsed {
		t.Errorf("INV-4: expected %d used blocks after preemption, got %d", expectedUsed, kvCache.UsedBlocks())
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd .worktrees/fix-349-phase1-livelock && go test ./sim/... -run TestVLLMBatchFormation_Phase1_EvictedNotRevisited -v`
Expected: FAIL — with `range` loop, r3 is revisited after eviction, causing cascading preemptions (preemption count > 1)

**Step 3: Implement the fix**

Context: Three changes in `sim/batch_formation.go`:
1. Zero `NumNewTokens` at FormBatch entry (FIX-2)
2. Replace range with index-based loop (FIX-1)
3. Add `*int64` tokenBudget to preemptForTokens with budget restoration
4. Preserve "token budget exhausted" warning

Also update `TestPreemptForTokens_CleansUpComputedTokens` at line 450 to pass the new 5th argument.

In `sim/batch_formation.go`, after `tokenBudget := ctx.MaxScheduledTokens` (line 69), add:

```go
	// Zero NumNewTokens for all running requests at the start of each scheduling pass.
	// This prevents stale values from the prior step from causing phantom budget
	// restoration when a request is preempted before being visited in this pass.
	for _, req := range ctx.RunningBatch.Requests {
		req.NumNewTokens = 0
	}
```

Replace lines 71-107 (Phase 1 loop) with:

```go
	// Phase 1: Process continuing requests (chunked prefill + decode).
	// Index-based loop: re-evaluates len() each iteration so evicted requests
	// (removed by preemptForTokens tail eviction) are never visited.
	// This achieves the same behavioral property as vLLM v1 (evicted requests
	// are never revisited within a scheduling pass), though through a different
	// mechanism (vLLM uses deque popleft/pop; BLIS uses index bounds re-evaluation).
	reqIndex := 0
	for reqIndex < len(result.RunningBatch.Requests) {
		if tokenBudget <= 0 {
			logrus.Warnf("[tick %07d] token budget exhausted, deferring remaining requests to next step", ctx.Now)
			break
		}
		req := result.RunningBatch.Requests[reqIndex]

		numNewTokens := util.Len64(req.InputTokens) - req.ProgressIndex
		// Chunked prefill for running requests
		if numNewTokens > 0 {
			if 0 < ctx.PrefillTokenThreshold && ctx.PrefillTokenThreshold < numNewTokens {
				numNewTokens = ctx.PrefillTokenThreshold
			}
			numNewTokens = min(numNewTokens, tokenBudget)

			if canSchedule := v.preemptForTokens(req, numNewTokens, &result, ctx, &tokenBudget); !canSchedule {
				break
			}

			tokenBudget -= numNewTokens
			req.NumNewTokens = int(numNewTokens)
			ctx.ComputedTokens[req.ID] += numNewTokens
		}
		// Decode phase: allocate 1 token
		if req.ProgressIndex >= util.Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
			decodeTokens := int64(1)
			if canSchedule := v.preemptForTokens(req, decodeTokens, &result, ctx, &tokenBudget); !canSchedule {
				break
			}
			tokenBudget--
			req.NumNewTokens = 1
			ctx.ComputedTokens[req.ID] += 1
		}
		reqIndex++
	}
```

Update `preemptForTokens` signature to add `tokenBudget *int64`, with defensive budget restoration:

```go
func (v *VLLMBatchFormation) preemptForTokens(req *Request, numNewTokens int64, result *BatchResult, ctx BatchContext, tokenBudget *int64) bool {
	for {
		if ok := ctx.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, []int64{}); !ok {
			// Circuit breaker: empty batch means cache is too small (R19)
			if len(result.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					ctx.Now, req.ID, numNewTokens)
				return false
			}

			result.PreemptionHappened = true
			preemptionDelay := v.latencyModel.PreemptionProcessingTime()
			preemptedRequest := result.RunningBatch.Requests[len(result.RunningBatch.Requests)-1]
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room", ctx.Now, preemptedRequest.ID)
			result.RunningBatch.Requests = result.RunningBatch.Requests[:len(result.RunningBatch.Requests)-1]

			result.Preempted = append(result.Preempted, PreemptedRequest{
				Request:         preemptedRequest,
				PreemptionDelay: preemptionDelay,
			})

			// Defensive: restore token budget if preempted request was already
			// scheduled in this step. With tail-only eviction and head-to-tail
			// iteration, this fires when a request earlier in the batch (already
			// visited and scheduled) is evicted. The NumNewTokens zeroing at
			// FormBatch entry prevents stale values from prior steps.
			if preemptedRequest.NumNewTokens > 0 {
				*tokenBudget += int64(preemptedRequest.NumNewTokens)
				preemptedRequest.NumNewTokens = 0
			}

			preemptedRequest.State = StateQueued
			preemptedRequest.ProgressIndex = 0
			ctx.KVCache.ReleaseKVBlocks(preemptedRequest)
			delete(ctx.ComputedTokens, preemptedRequest.ID)
			ctx.WaitQ.PrependFront(preemptedRequest)

			if preemptedRequest == req {
				return false
			}
		} else {
			return true
		}
	}
}
```

In `sim/batch_formation_test.go`, update the existing `TestPreemptForTokens_CleansUpComputedTokens` at line 450:

Change:
```go
bf.preemptForTokens(newReq, 16, &result, ctx)
```
To:
```go
var budget int64 = 10000
bf.preemptForTokens(newReq, 16, &result, ctx, &budget)
```

**Step 4: Run test to verify it passes**

Run: `cd .worktrees/fix-349-phase1-livelock && go test ./sim/... -run TestVLLMBatchFormation_Phase1_EvictedNotRevisited -v`
Expected: PASS

**Step 5: Run full test suite** (catches existing test breakage from signature change)

Run: `cd .worktrees/fix-349-phase1-livelock && go test ./sim/... -v -count=1`
Expected: All PASS

**Step 6: Run lint**

Run: `cd .worktrees/fix-349-phase1-livelock && golangci-lint run ./sim/...`
Expected: No new issues

**Step 7: Commit**

```bash
cd .worktrees/fix-349-phase1-livelock
git add sim/batch_formation.go sim/batch_formation_test.go
git commit -m "fix(batch): replace range loop with index-based iteration in Phase 1 (#349)

- Phase 1 now uses index-based loop: evicted requests never revisited
- Zero NumNewTokens at FormBatch entry to prevent stale state
- Defensive token budget restoration in preemptForTokens
- Update existing TestPreemptForTokens caller for new signature
- Remove stale 'do NOT fix this' comment

Fixes #349

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Livelock resolution integration test (FIX-3)

**Contracts Implemented:** FIX-3

**Files:**
- Modify: `sim/batch_formation_test.go`

**Step 1: Write integration test**

Context: Run parameters matching the `tmp/run.sh` reproducer through BLIS and verify the livelock is resolved. Uses `mustNewSimulator` helper (existing pattern in `sim/simulator_test.go`).

```go
// TestVLLMBatchFormation_LivelockResolution verifies FIX-3:
// The pathological workload from #349 (seed=7, 7463 blocks, output 3200-3596 tokens)
// must complete requests instead of livelocking with 100K+ preemptions.
func TestVLLMBatchFormation_LivelockResolution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	// GIVEN parameters matching tmp/run.sh reproducer
	// Note: NewLatencyCoeffs takes (betaCoeffs, alphaCoeffs)
	cfg := SimConfig{
		Horizon: 120000000,
		Seed:    7,
		KVCacheConfig: NewKVCacheConfig(
			7463, // total blocks
			16,   // block size
			0, 0, 0, 0,
		),
		BatchConfig: NewBatchConfig(
			256,  // max running reqs
			2048, // max scheduled tokens
			0,    // long prefill threshold (disabled)
		),
		LatencyCoeffs: NewLatencyCoeffs(
			[]float64{5752.705191348184, 17.25086436834028, 5.999143920128404},   // beta
			[]float64{232.46191091038054, 1.752360364195244, 3357.4400353290152}, // alpha
		),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, ""),
		PolicyConfig:        NewPolicyConfig("constant", "fcfs"),
	}

	sim := mustNewSimulator(t, cfg)

	// Inject 30 requests (subset of 300 for test speed) with the workload profile.
	// Uses its own RNG for workload generation (independent from simulator's RNG).
	rng := NewPartitionedRNG(NewSimulationKey(7))
	wlRng := rng.ForSubsystem(SubsystemWorkload)
	arrivalTime := int64(0)
	for i := 0; i < 30; i++ {
		inputLen := 200 + wlRng.Intn(201)  // 200-400
		outputLen := 3200 + wlRng.Intn(397) // 3200-3596
		req := &Request{
			ID:           fmt.Sprintf("req_%d", i),
			InputTokens:  make([]int, inputLen),
			OutputTokens: make([]int, outputLen),
			ArrivalTime:  arrivalTime,
			State:        StateQueued,
		}
		sim.InjectArrival(req)
		arrivalTime += 100000 // 100ms between arrivals (rate=10/s)
	}

	sim.Run()

	// THEN some requests must complete (livelock resolved)
	if sim.Metrics.CompletedRequests == 0 {
		t.Errorf("FIX-3: expected completed_requests > 0, got 0 (livelock not resolved)")
	}

	// AND preemption count must be dramatically reduced (not 100K+)
	if sim.Metrics.PreemptionCount > 1000 {
		t.Errorf("FIX-3: expected preemption_count < 1000, got %d (cascading preemption not resolved)",
			sim.Metrics.PreemptionCount)
	}

	// AND request conservation must hold (INV-1)
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 30 {
		t.Errorf("INV-1: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, expected 30",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning,
			sim.Metrics.DroppedUnservable, total)
	}
}
```

**Step 2: Run test**

Run: `cd .worktrees/fix-349-phase1-livelock && go test ./sim/... -run TestVLLMBatchFormation_LivelockResolution -v -timeout 120s`
Expected: PASS — completed_requests > 0, preemption_count < 1000

**Step 3: Run full test suite**

Run: `cd .worktrees/fix-349-phase1-livelock && go test ./... -timeout 120s`
Expected: All PASS

**Step 4: Run lint**

Run: `cd .worktrees/fix-349-phase1-livelock && golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
cd .worktrees/fix-349-phase1-livelock
git add sim/batch_formation_test.go
git commit -m "test(batch): add livelock resolution integration test (FIX-3)

- Reproduce #349 pathological workload (seed=7, 7463 blocks, output 3200-3596)
- Verify completed_requests > 0 and preemption_count < 1000
- Verify INV-1 request conservation holds

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Backward compatibility verification (FIX-5, FIX-6)

**Contracts Implemented:** FIX-5, FIX-6

**Files:** None (verification only)

**Step 1: Run existing test suite**

Run: `cd .worktrees/fix-349-phase1-livelock && go test ./... -count=1 -timeout 120s`
Expected: All PASS. Golden dataset tests pass with identical values (FIX-5). Circuit breaker tests pass (FIX-6).

**Step 2: Run lint**

Run: `cd .worktrees/fix-349-phase1-livelock && golangci-lint run ./...`
Expected: Zero issues

**Step 3: Build**

Run: `cd .worktrees/fix-349-phase1-livelock && go build ./...`
Expected: Clean build

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| FIX-1 | Task 1 | Unit | TestVLLMBatchFormation_Phase1_EvictedNotRevisited |
| FIX-2 | Task 1 | Unit | TestVLLMBatchFormation_Phase1_EvictedNotRevisited (r3.NumNewTokens=5 stale, verifies zeroing) |
| FIX-3 | Task 2 | Integration | TestVLLMBatchFormation_LivelockResolution |
| FIX-4 | Task 1 | Unit | TestVLLMBatchFormation_Phase1_EvictedNotRevisited (dual allocation check) |
| FIX-5 | Task 3 | Regression | Existing golden dataset tests (5 entries, 0 preemptions) |
| FIX-6 | Task 3 | Existing | TestVLLMBatchFormation_CircuitBreaker, TestPreempt_EmptyBatch_ReturnsFalse |

**Golden dataset update:** Not needed — 0 golden entries have preemptions, and the fix only changes preemption behavior.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Index loop visits request at wrong index after eviction | Low | High | `reqIndex` only increments after successful iteration; eviction shortens from tail only, so reqIndex always points at a valid non-evicted request | Task 1 |
| Stale NumNewTokens from prior step causes phantom budget restoration | Low | Medium | Explicit zeroing loop at FormBatch entry clears all stale values | Task 1 |
| Existing tests break from preemptForTokens signature change | High | High | Explicitly update `TestPreemptForTokens_CleansUpComputedTokens` in Task 1 | Task 1 |
| Integration test passes vacuously (30 requests don't trigger preemption) | Low | Medium | KV math: 30 × ~237 blocks/req = 7110 blocks vs 7463 total → pressure guaranteed at full occupancy | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions (fix is minimal: loop change + zeroing + budget restore)
- [x] No feature creep beyond PR scope (only fixes #349)
- [x] No unexercised flags or interfaces (no new flags/interfaces)
- [x] No partial implementations (fix is complete in Task 1)
- [x] No breaking changes (FIX-5: golden dataset unaffected)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md update not needed (no new files/packages/flags)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: no canonical sources modified
- [x] Deviation log reviewed — all deviations documented and justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 → Task 2 → Task 3)
- [x] All contracts mapped to tasks
- [x] Golden dataset regeneration not needed
- [x] Construction site audit: `preemptForTokens` signature change → existing caller at `batch_formation_test.go:450` explicitly updated in Task 1

**Antipattern rules:**
- [x] R1: No silent data loss (token budget restored defensively, warning preserved)
- [x] R2: N/A (no map iteration for float accumulation)
- [x] R3: N/A (no new CLI flags)
- [x] R4: preemptForTokens signature change — sole additional caller updated in Task 1
- [x] R5: Fix resolves existing R5 violation (evicted request state corruption)
- [x] R6: No logrus.Fatalf in sim/ (only logrus.Warnf, which is existing)
- [x] R7: Integration test (Task 2) verifies INV-1 conservation invariant
- [x] R8-R18: N/A
- [x] R19: Fix resolves existing R19 violation (cascading preemption livelock)
- [x] R20: N/A

---

## Appendix: File-Level Implementation Details

### File: `sim/batch_formation.go`

**Purpose:** Fix Phase 1 range loop, add NumNewTokens zeroing, add token budget restoration to preemptForTokens.

**Changes:**

1. **After line 69 (after `tokenBudget := ctx.MaxScheduledTokens`):** Add NumNewTokens zeroing loop for all running requests.

2. **Lines 71-107 (Phase 1 loop):** Replace `for _, req := range ctx.RunningBatch.Requests` with index-based `for reqIndex < len(result.RunningBatch.Requests)`. Preserve `tokenBudget <= 0` warning inside loop body. Remove stale "do NOT fix" comment.

3. **Line 149 (preemptForTokens signature):** Add `tokenBudget *int64` parameter.

4. **Inside preemptForTokens, after appending to `result.Preempted`:** Add defensive token budget restoration with `NumNewTokens` check and zeroing.

**Key implementation notes:**
- The `reqIndex++` is at the END of the loop body, matching vLLM's advancement pattern.
- Tail eviction only: when `preemptForTokens` evicts the tail, `len(result.RunningBatch.Requests)` decreases. Since eviction only removes from indices >= reqIndex+1 (or == reqIndex for self-eviction), reqIndex is always valid.
- `NumNewTokens` zeroing at FormBatch entry ensures budget restoration uses current-step data only.
- Budget restoration is defensive: with tail-only eviction and head-to-tail iteration, the evicted request is typically unvisited (NumNewTokens=0). Restoration fires when the batch is re-ordered or when a previously-visited request ends up at the tail.

### File: `sim/batch_formation_test.go`

**Purpose:** Add 2 new tests (FIX-1/FIX-4, FIX-3), update existing preemptForTokens caller.

**Tests added:**
- `TestVLLMBatchFormation_Phase1_EvictedNotRevisited` — 3 running requests, tight KV, verifies exactly 1 preemption
- `TestVLLMBatchFormation_LivelockResolution` — 30 requests with #349 params, verifies completions > 0

**Existing test updated:**
- `TestPreemptForTokens_CleansUpComputedTokens` — add `&budget` as 5th arg to `preemptForTokens` call
