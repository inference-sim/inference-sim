# Hardening Phase 3: Data Loss & Metric Distortion Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix four metric-reporting bugs that silently lose data or inflate measurements, ensuring simulator output is trustworthy for capacity planning decisions.

**The problem today:** (1) SaveResults silently drops incomplete requests from JSON output — any request not completing prefill vanishes from the per-request log. (2) SLO attainment computation silently skips requests missing from the Requests map, inflating the reported fraction. (3) In tiered KV cache mode, GetCachedBlocks increments CacheHits as a side effect, and the tiered reload path calls it twice per allocation — double-counting cache hits and distorting CacheHitRate. (4) ComputeFitness silently ignores unknown metric keys with a log warning, making typos in `--fitness-weights` invisible instead of failing fast.

**What this PR adds:**
1. Complete per-request output — all injected requests appear in JSON output, with zero-valued metrics for incomplete ones (distinguishable from real measurements which are always positive)
2. Conservative SLO attainment — dropped/missing requests count as SLO violations, with a warning counter for observability
3. Pure GetCachedBlocks — CacheHits counted at allocation commit (not lookup), eliminating tiered mode inflation and enabling clean rollback
4. Fail-fast ComputeFitness — unknown metric keys return an error instead of being silently ignored

**Why this matters:** These fixes ensure metrics used for capacity planning and policy optimization are correct. Downstream consumers (fitness evaluation, SLO reports, cache efficiency analysis) depend on accurate data. This completes the data-correctness layer before PR11-PR16.

**Architecture:** All changes are in the metrics/reporting layer: `sim/metrics.go` (SaveResults), `sim/cluster/metrics.go` (SLOAttainment, ComputeFitness), `sim/kvcache.go` (GetCachedBlocks purity), and `cmd/root.go` (error propagation). No new types or interfaces.

**Source:** Phase 3 of `docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md`. Issues: #190, #201, #203. CacheHits impurity documented in CLAUDE.md.

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes four metric-reporting bugs in the data loss / metric distortion category:

- **SaveResults** iterates only over completed-prefill requests, losing incomplete ones (#190)
- **SLOAttainment** silently drops unmatched requests, inflating the attainment fraction (#201)
- **GetCachedBlocks** increments CacheHits as a side effect, double-counted in tiered mode
- **ComputeFitness** silently ignores unknown keys instead of failing fast (#203)

**Where it fits:** After Phase 1 (structural helpers, PR #216) and Phase 2 (correctness fixes, PR #217). Before Phase 4 (invariant tests), Phase 5 (input validation), and Phase 6 (modularity).

**Adjacent blocks:** `sim/kvcache.go` rollback system (Phase 2), `sim/kvcache_tiered.go` allocation path, `cmd/root.go` CLI error handling.

**DEVIATION flags:** See Section D.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: SaveResults Includes All Registered Requests
- GIVEN a simulation with N injected requests where M < N complete prefill
- WHEN SaveResults writes the per-request JSON output
- THEN all N requests appear in the output, sorted by ArrivedAt
- MECHANISM: Iterate `m.Requests` (all registered) instead of `m.RequestTTFTs` (completed prefill only). Incomplete requests have zero-valued TTFT/E2E/ITL.

BC-2: SLO Attainment Conservative Denominator
- GIVEN aggregated metrics where some request IDs in RequestE2Es have no entry in Requests map
- WHEN SLOAttainment is computed
- THEN dropped requests are counted in the denominator as SLO violations (conservative)
- MECHANISM: Replace silent `continue` with a `droppedCount` counter. Add `droppedCount` to `total`. Log at Warn level if any dropped.

BC-3: GetCachedBlocks Is Pure Query
- GIVEN a KVCacheState with cached prefix blocks
- WHEN GetCachedBlocks is called N times with the same tokens
- THEN CacheHits is unchanged after all N calls
- MECHANISM: Remove `kvc.CacheHits++` from GetCachedBlocks. Move increment to AllocateKVBlocks cached-block processing loop.

BC-4: CacheHits Counted at Allocation Commit
- GIVEN a request with K cached prefix blocks
- WHEN AllocateKVBlocks succeeds using those cached blocks
- THEN CacheHits is incremented by exactly K
- MECHANISM: `kvc.CacheHits++` in the cached-block loop of AllocateKVBlocks (lines 182-194).

BC-5: CacheHits Rolled Back on Allocation Failure
- GIVEN a request with K cached blocks where allocation fails mid-loop
- WHEN rollbackAllocation undoes the partial allocation
- THEN CacheHits is decremented by K (returning to pre-allocation value)
- MECHANISM: `kvc.CacheHits--` in the cached-block rollback loop of rollbackAllocation.

BC-6: Tiered Mode No Double-Count
- GIVEN a TieredKVCache where CPU-to-GPU reload triggers a second GetCachedBlocks call
- WHEN the tiered allocation completes
- THEN CacheHits reflects only the blocks actually used, not the lookup count
- MECHANISM: Since GetCachedBlocks is now pure (BC-3), the second call at kvcache_tiered.go:73 has no side effect.

BC-7: ComputeFitness Fails on Unknown Keys
- GIVEN a weights map containing at least one key not recognized by extractMetric
- WHEN ComputeFitness is called
- THEN it returns a non-nil error and zero-valued FitnessResult
- MECHANISM: Validate all keys before computing. Return `(FitnessResult{}, error)`.

**Negative Contracts:**

BC-8: Single-Tier CacheHits Unchanged
- GIVEN a single-tier KVCacheState (no tiering)
- WHEN GetCachedBlocks is called once followed by a successful AllocateKVBlocks
- THEN the resulting CacheHits count MUST be identical to the old behavior
- MECHANISM: One GetCachedBlocks → one AllocateKVBlocks → same K blocks counted.

BC-9: Golden Dataset Not Affected
- GIVEN the existing golden dataset test scenarios
- WHEN running with fixed code
- THEN all golden values MUST remain byte-identical
- MECHANISM: Golden tests use single-instance mode (no tiering), and all requests complete (no incomplete requests in output). The CacheHits purity fix doesn't change single-tier counts.

**Error Handling Contracts:**

BC-10: ComputeFitness Error Propagation
- GIVEN the CLI caller in cmd/root.go
- WHEN ComputeFitness returns an error (unknown key)
- THEN the CLI calls logrus.Fatalf with the error message
- MECHANISM: `if err != nil { logrus.Fatalf(...) }` at the call site.

### C) Component Interaction

```
cmd/root.go
    │
    ├── ComputeFitness(raw, weights) → (FitnessResult, error)  [3d: new error return]
    │       ↓
    │   cluster/metrics.go: validate keys → compute
    │
    ├── SLOAttainment(aggregated, targets) → float64  [3b: conservative denominator]
    │       ↓
    │   cluster/metrics.go: count dropped → include in total
    │
sim/simulator.go
    │
    ├── GetCachedBlocks(tokens) → []int64  [3c: now pure, no side effects]
    │       ↓
    │   kvcache.go: lookup only, no CacheHits mutation
    │
    ├── AllocateKVBlocks(req, start, end, cached) → bool  [3c: CacheHits here]
    │       ↓ on failure
    │   rollbackAllocation: undo CacheHits along with other counters
    │
sim/metrics.go
    │
    └── SaveResults → JSON  [3a: iterate m.Requests not m.RequestTTFTs]
```

**State changes:**
- `CacheHits` ownership moves from GetCachedBlocks → AllocateKVBlocks (same struct, different method)
- `ComputeFitness` signature: `(*FitnessResult)` → `(FitnessResult, error)` (value return, not pointer)

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Log dropped count at Warn level" for 3b | Use `logrus.Warnf` when droppedCount > 0 at end of function | SIMPLIFICATION: One log at end is less noisy than per-request logging |
| ComputeFitness returns `(FitnessResult, error)` (value) | Same | MATCH |
| "Iterate over m.Requests instead of m.RequestTTFTs" for 3a | Same, but also handle missing TTFT/E2E/ITL with zero values | MATCH with explicit zero-value handling |
| Design doc mentions "CacheHits once per block actually reused" | Plan increments CacheHits in cached-block loop of AllocateKVBlocks, rolls back on failure | ADDITION: Rollback integration not in design doc but required for transactional consistency |
| Design doc says "no golden dataset regeneration needed" | Plan explicitly verifies golden dataset is NOT affected (BC-9) | ADDITION: Explicit verification prevents a repeat of Phase 2 golden surprise |
| Design doc 3d shows `for key := range weights` (unsorted) | Plan uses `sortedKeys(weights)` for deterministic error messages | IMPROVEMENT: Sorted iteration per project antipattern rules |
| Design doc 3d shows single-pass validate+compute | Plan validates all keys first, then computes in second pass | IMPROVEMENT: Fail-fast before any computation, preventing partial results |
| Current code returns `*FitnessResult` (pointer) | Plan returns `FitnessResult` (value) per error-return convention | CORRECTION: Value return with error is idiomatic Go; `evaluation.go:14` stores `*FitnessResult` but callers can take address if needed |
| Design doc Phase 3 does not mention CLAUDE.md | Plan updates CLAUDE.md section 8 to remove stale impurity warning | ADDITION: Stale docs are a bug — fix alongside the code change |
| EvaluationResult.Fitness is `*FitnessResult` | Not changed — ComputeFitness now returns value, callers take address if needed | DEFERRAL: No current code path constructs EvaluationResult with a fitness result from ComputeFitness; future callers use `f := result; NewEvaluationResult(..., &f, ...)` |

### E) Review Guide

**The tricky part:** BC-5 (CacheHits rollback). Moving CacheHits into AllocateKVBlocks means the rollback function must now also undo CacheHits. The existing `rollbackAllocation` comment says "CacheHits is NOT rolled back" — this comment becomes stale and must be updated.

**What to scrutinize:** The interaction between 3c (CacheHits move) and Phase 2's rollback system. Verify that `cachedBlockMutation` loop in rollback decrements CacheHits. Verify single-tier behavior is identical to before.

**What's safe to skim:** 3a (SaveResults iteration change) is mechanical. 3d (ComputeFitness error) is straightforward signature change.

**Known debt:** `SavetoFile` in metrics_utils.go still calls `logrus.Fatalf` (library code calling os.Exit) — this is Phase 6 scope, not Phase 3. Also, `ComputePerSLODistributions` has the same silent `continue` pattern as SLOAttainment (line 242-243) — not in Phase 3 scope but should be noted for future hardening.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/kvcache.go` — Remove CacheHits from GetCachedBlocks, add to AllocateKVBlocks, update rollback
- `sim/kvcache_tiered.go` — No code change needed (GetCachedBlocks delegation is now pure)
- `sim/metrics.go` — Iterate m.Requests instead of m.RequestTTFTs in SaveResults
- `sim/cluster/metrics.go` — SLOAttainment conservative denominator; ComputeFitness error return
- `cmd/root.go` — Handle ComputeFitness error
- `sim/kvcache_test.go` — Test CacheHits purity and rollback
- `sim/kvcache_tiered_test.go` — Test tiered mode no double-count
- `sim/cluster/metrics_test.go` — Update ComputeFitness tests for error return
- `sim/cluster/metrics_slo_test.go` — Test SLO attainment with missing requests
- `sim/metrics_test.go` — Test SaveResults with incomplete requests (new file or appended)
- `CLAUDE.md` — Update CacheHits warning to reflect fix

**Key decisions:**
- CacheHits moves INTO the cached-block loop of AllocateKVBlocks (not before or after)
- Rollback decrements CacheHits per cached mutation (symmetric with increment)
- ComputeFitness returns value `FitnessResult` not pointer `*FitnessResult` (error return convention)
- No golden dataset regeneration needed (verified by BC-9)

### G) Task Breakdown

---

### Task 1: Make GetCachedBlocks Pure — Move CacheHits to AllocateKVBlocks

**Contracts Implemented:** BC-3, BC-4, BC-5, BC-6, BC-8

**Files:**
- Modify: `sim/kvcache.go` (GetCachedBlocks, AllocateKVBlocks, rollbackAllocation)
- Test: `sim/kvcache_test.go`
- Test: `sim/kvcache_tiered_test.go`

**Step 1: Write failing tests for CacheHits purity and rollback**

Context: We need to verify that GetCachedBlocks no longer increments CacheHits, that AllocateKVBlocks counts hits correctly, and that rollback undoes CacheHits.

In `sim/kvcache_test.go`, add:

```go
func TestGetCachedBlocks_IsPureQuery_NoCacheHitsSideEffect(t *testing.T) {
	// GIVEN a KV cache with cached prefix blocks
	kvc := NewKVCacheState(4, 2)
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req)
	// Blocks have hashes now; reset CacheHits for clean measurement
	kvc.CacheHits = 0

	// WHEN GetCachedBlocks is called multiple times
	_ = kvc.GetCachedBlocks([]int{1, 2, 3, 4})
	_ = kvc.GetCachedBlocks([]int{1, 2, 3, 4})
	_ = kvc.GetCachedBlocks([]int{1, 2, 3, 4})

	// THEN CacheHits is unchanged (BC-3)
	if kvc.CacheHits != 0 {
		t.Errorf("CacheHits = %d after 3 GetCachedBlocks calls, want 0 (pure query)", kvc.CacheHits)
	}
}

func TestAllocateKVBlocks_CacheHitsCountedAtCommit(t *testing.T) {
	// GIVEN a KV cache with 2 cached prefix blocks
	kvc := NewKVCacheState(8, 2)
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)
	kvc.CacheHits = 0

	// WHEN allocating with 2 cached blocks
	cached := kvc.GetCachedBlocks([]int{1, 2, 3, 4, 5, 6})
	if len(cached) != 2 {
		t.Fatalf("expected 2 cached blocks, got %d", len(cached))
	}
	req2 := &Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	ok := kvc.AllocateKVBlocks(req2, 4, 6, cached)
	if !ok {
		t.Fatal("allocation should succeed")
	}

	// THEN CacheHits is exactly 2 (BC-4)
	if kvc.CacheHits != 2 {
		t.Errorf("CacheHits = %d, want 2 (one per cached block)", kvc.CacheHits)
	}
}

func TestAllocateKVBlocks_CacheHitsRolledBackOnFailure(t *testing.T) {
	// GIVEN a KV cache with 2 cached prefix blocks and tight free-block budget
	kvc := NewKVCacheState(4, 2)
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)

	// Consume 1 block to make allocation fail
	filler := &Request{ID: "filler", InputTokens: []int{90, 91}}
	kvc.AllocateKVBlocks(filler, 0, 2, []int64{})

	hitsBefore := kvc.CacheHits

	// WHEN allocating with cached prefix + new tokens that exceed capacity
	req2 := &Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)
	ok := kvc.AllocateKVBlocks(req2, 4, 8, cached)

	// THEN allocation fails AND CacheHits is rolled back (BC-5)
	if ok {
		t.Fatal("allocation should fail")
	}
	if kvc.CacheHits != hitsBefore {
		t.Errorf("CacheHits = %d, want %d (should be rolled back)", kvc.CacheHits, hitsBefore)
	}
}
```

In `sim/kvcache_tiered_test.go`, add:

```go
func TestTieredKVCache_NoCacheHitsDoubleCount(t *testing.T) {
	// GIVEN a tiered cache where GPU allocation will initially fail, triggering CPU reload
	gpu := NewKVCacheState(4, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 1.0, 100)

	// Populate prefix cache
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req1, 0, 4, []int64{})
	tiered.ReleaseKVBlocks(req1)
	gpu.CacheHits = 0

	// WHEN calling GetCachedBlocks multiple times (simulating tiered retry path)
	_ = tiered.GetCachedBlocks([]int{1, 2, 3, 4})
	_ = tiered.GetCachedBlocks([]int{1, 2, 3, 4})

	// THEN CacheHits is 0 — GetCachedBlocks is pure (BC-6)
	if gpu.CacheHits != 0 {
		t.Errorf("GPU CacheHits = %d after 2 GetCachedBlocks calls, want 0", gpu.CacheHits)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/... -run "TestGetCachedBlocks_IsPureQuery|TestAllocateKVBlocks_CacheHitsCountedAtCommit|TestAllocateKVBlocks_CacheHitsRolledBackOnFailure|TestTieredKVCache_NoCacheHitsDoubleCount" -v`
Expected: FAIL — `TestGetCachedBlocks_IsPureQuery` fails because GetCachedBlocks still increments CacheHits. `TestAllocateKVBlocks_CacheHitsCountedAtCommit` fails because AllocateKVBlocks doesn't increment CacheHits.

**Step 3: Implement GetCachedBlocks purity + AllocateKVBlocks CacheHits + rollback**

In `sim/kvcache.go`:

1. Remove `kvc.CacheHits++` from `GetCachedBlocks` (line 134). Update doc comment:

```go
// GetCachedBlocks attempts to reuse previously cached full blocks.
// It returns block IDs for the longest contiguous cached prefix.
// This is a pure query — it does not modify any state.
// CacheHits are counted by AllocateKVBlocks when cached blocks are committed.
func (kvc *KVCacheState) GetCachedBlocks(tokens []int) (blockIDs []int64) {
	n := Len64(tokens) / kvc.BlockSizeTokens
	for i := int64(0); i < n; i++ {
		chunk := tokens[:(i+1)*kvc.BlockSizeTokens]
		h := hashTokens(chunk)
		blockId, ok := kvc.HashToBlock[h]
		if !ok {
			break
		}
		blockIDs = append(blockIDs, blockId)
	}
	return
}
```

2. Add `kvc.CacheHits++` in the cached-block loop of `AllocateKVBlocks` (inside the `for _, blockId := range cachedBlocks` loop, after line 191):

```go
			cachedMutations = append(cachedMutations, cachedBlockMutation{block: blk, wasInUse: wasInUse})
			kvc.CacheHits++
```

3. Add `kvc.CacheHits--` in `rollbackAllocation` cached-block undo loop:

```go
	// Undo cached block mutations
	for _, cm := range cachedMutations {
		cm.block.RefCount--
		kvc.CacheHits--
		if !cm.wasInUse && cm.block.RefCount == 0 {
			cm.block.InUse = false
			kvc.UsedBlockCnt--
			kvc.appendToFreeList(cm.block)
		}
	}
```

4. Update the `rollbackAllocation` doc comment to remove the stale "CacheHits is NOT rolled back" note:

```go
// rollbackAllocation undoes all mutations from a failed AllocateKVBlocks call.
// Restores UsedBlockCnt, CacheMisses, CacheHits, RefCount, InUse, free list, HashToBlock, and RequestMap.
// Also restores prefix hashes that were destroyed by popFreeBlock during allocation.
func (kvc *KVCacheState) rollbackAllocation(reqID string, cachedMutations []cachedBlockMutation, newlyAllocated []newBlockMutation) {
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestGetCachedBlocks_IsPureQuery|TestAllocateKVBlocks_CacheHitsCountedAtCommit|TestAllocateKVBlocks_CacheHitsRolledBackOnFailure|TestTieredKVCache_NoCacheHitsDoubleCount|TestAllocateKVBlocks_CachedBlockRollback" -v`
Expected: PASS — all new tests pass, and existing rollback test still passes.

**Step 5: Run full test suite**

Run: `go test ./sim/... -count=1`
Expected: PASS (golden dataset unaffected — BC-9)

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues.

**Step 7: Commit**

```bash
git add sim/kvcache.go sim/kvcache_test.go sim/kvcache_tiered_test.go
git commit -m "fix(kvcache): make GetCachedBlocks pure, count CacheHits at allocation commit (BC-3..BC-6)

- Remove CacheHits side effect from GetCachedBlocks (now a pure query)
- Increment CacheHits in AllocateKVBlocks cached-block loop
- Roll back CacheHits in rollbackAllocation on failure
- Eliminates tiered mode double-counting via reload path

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: SaveResults Includes All Registered Requests

**Contracts Implemented:** BC-1

**Files:**
- Modify: `sim/metrics.go:130-143`
- Test: `sim/metrics_test.go` (append to existing file — already has 3 InstanceID tests)

**Step 1: Write failing test**

Context: SaveResults currently iterates `m.RequestTTFTs` which only contains requests that completed prefill. We need it to include all requests from `m.Requests`.

Add to existing `sim/metrics_test.go` (file already has `package sim` and necessary imports including `encoding/json`, `os`, `path/filepath`, `testing`, `time`):

```go
func TestSaveResults_IncludesIncompleteRequests(t *testing.T) {
	// GIVEN metrics where 2 of 3 requests completed prefill
	m := NewMetrics()
	// All 3 registered
	m.Requests["r1"] = RequestMetrics{ID: "r1", ArrivedAt: 1.0, NumPrefillTokens: 10, NumDecodeTokens: 5}
	m.Requests["r2"] = RequestMetrics{ID: "r2", ArrivedAt: 2.0, NumPrefillTokens: 20, NumDecodeTokens: 10}
	m.Requests["r3"] = RequestMetrics{ID: "r3", ArrivedAt: 3.0, NumPrefillTokens: 30, NumDecodeTokens: 0} // incomplete

	// Only r1 and r2 completed prefill
	m.RequestTTFTs["r1"] = 100.0
	m.RequestTTFTs["r2"] = 200.0
	m.RequestE2Es["r1"] = 500.0
	m.RequestE2Es["r2"] = 1000.0
	m.RequestITLs["r1"] = 50.0
	m.RequestITLs["r2"] = 100.0
	m.RequestSchedulingDelays["r1"] = 10
	m.RequestSchedulingDelays["r2"] = 20

	m.CompletedRequests = 2
	m.TotalOutputTokens = 15
	m.SimEndedTime = 1_000_000

	// WHEN SaveResults writes to a temp file
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "results.json")
	m.SaveResults("test-instance", 10_000_000, 100, time.Now(), outPath)

	// THEN the output file contains all 3 requests
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}
	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("failed to parse output: %v", err)
	}

	if len(output.Requests) != 3 {
		t.Errorf("output.Requests count = %d, want 3 (all registered, including incomplete)", len(output.Requests))
	}

	// Verify incomplete request r3 has zero-valued metrics
	for _, req := range output.Requests {
		if req.ID == "r3" {
			if req.TTFT != 0 || req.E2E != 0 || req.ITL != 0 {
				t.Errorf("incomplete request r3 should have zero metrics, got TTFT=%f E2E=%f ITL=%f",
					req.TTFT, req.E2E, req.ITL)
			}
			return
		}
	}
	t.Error("incomplete request r3 not found in output")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestSaveResults_IncludesIncompleteRequests -v`
Expected: FAIL — `output.Requests count = 2, want 3`

**Step 3: Implement SaveResults fix**

In `sim/metrics.go`, replace the per-request output loop (lines 130-143):

Replace:
```go
	// --- Write to JSON File ---
	if outputFilePath != "" {
		// request-level metrics for detailed output in file
		for id, ttft := range m.RequestTTFTs {
			detail := m.Requests[id]
			detail.TTFT = ttft / 1e3
			detail.E2E = m.RequestE2Es[id] / 1e3
			detail.ITL = m.RequestITLs[id]
			detail.SchedulingDelay = float64(m.RequestSchedulingDelays[id])
			output.Requests = append(output.Requests, detail)
		}
```

With:
```go
	// --- Write to JSON File ---
	if outputFilePath != "" {
		// request-level metrics for detailed output in file
		// Iterate over all registered requests (not just completed prefill)
		// so incomplete requests appear with zero-valued metrics.
		for _, id := range sortedRequestIDs(m.Requests) {
			detail := m.Requests[id]
			detail.TTFT = m.RequestTTFTs[id] / 1e3   // zero if not in map
			detail.E2E = m.RequestE2Es[id] / 1e3      // zero if not in map
			detail.ITL = m.RequestITLs[id]             // zero if not in map
			detail.SchedulingDelay = float64(m.RequestSchedulingDelays[id])
			output.Requests = append(output.Requests, detail)
		}
```

Add a helper function (deterministic iteration order):
```go
// sortedRequestIDs returns request IDs from the Requests map in sorted order.
// Ensures deterministic output ordering for JSON serialization.
func sortedRequestIDs(requests map[string]RequestMetrics) []string {
	ids := make([]string, 0, len(requests))
	for id := range requests {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}
```

Note: The existing `sort.Slice` by ArrivedAt (lines 142-144) still follows, so final output order is by arrival time. The sorted iteration is for deterministic map traversal, not final output order.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestSaveResults_IncludesIncompleteRequests -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `go test ./sim/... -count=1`
Expected: PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues.

**Step 7: Commit**

```bash
git add sim/metrics.go sim/metrics_test.go
git commit -m "fix(metrics): SaveResults includes all registered requests, not just completed (#190, BC-1)

- Iterate m.Requests instead of m.RequestTTFTs for per-request output
- Incomplete requests appear with zero-valued TTFT/E2E/ITL metrics
- Add sortedRequestIDs helper for deterministic map iteration

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: SLO Attainment Conservative Denominator

**Contracts Implemented:** BC-2

**Files:**
- Modify: `sim/cluster/metrics.go:284-310`
- Test: `sim/cluster/metrics_slo_test.go`

**Step 1: Write failing test**

Context: SLOAttainment silently skips requests not found in the Requests map. We need it to count them as SLO violations.

In `sim/cluster/metrics_slo_test.go`, add:

```go
func TestSLOAttainment_MissingRequests_CountedAsViolation(t *testing.T) {
	// GIVEN 10 requests in RequestE2Es but only 7 in Requests map
	m := sim.NewMetrics()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.RequestE2Es[id] = 100 // all would meet SLO
	}
	// Only register 7 in Requests map
	for i := 0; i < 7; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "batch"}
	}
	targets := map[string]float64{"batch": 200}

	// WHEN computing SLO attainment
	attainment := SLOAttainment(m, targets)

	// THEN attainment should be 7/10 = 0.7 (dropped requests are violations)
	if math.Abs(attainment-0.7) > 0.01 {
		t.Errorf("attainment = %f, want 0.7 (3 missing requests should count as violations)", attainment)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestSLOAttainment_MissingRequests -v`
Expected: FAIL — attainment = 1.0 (missing requests skipped, 7/7 = 1.0)

**Step 3: Implement SLO attainment fix**

In `sim/cluster/metrics.go`, replace the SLOAttainment function:

```go
// SLOAttainment computes the fraction of requests meeting their SLO target.
// targets maps SLO class to max acceptable E2E latency (in ticks).
// Returns a value in [0.0, 1.0].
// Requests in RequestE2Es that are missing from Requests map are counted
// as SLO violations (conservative: missing data = violation).
func SLOAttainment(aggregated *sim.Metrics, targets map[string]float64) float64 {
	if len(aggregated.RequestE2Es) == 0 {
		return 0
	}
	met := 0
	total := 0
	droppedCount := 0
	for reqID, e2e := range aggregated.RequestE2Es {
		total++
		req, ok := aggregated.Requests[reqID]
		if !ok {
			droppedCount++
			continue // counted in total but not in met (= violation)
		}
		sloClass := req.SLOClass
		if target, ok := targets[sloClass]; ok {
			if e2e <= target {
				met++
			}
		} else {
			// No target for this class = always meets SLO
			met++
		}
	}
	if droppedCount > 0 {
		logrus.Warnf("SLOAttainment: %d requests in RequestE2Es missing from Requests map (counted as violations)", droppedCount)
	}
	if total == 0 {
		return 0
	}
	return float64(met) / float64(total)
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -run "TestSLOAttainment" -v`
Expected: PASS — all SLO attainment tests pass (existing + new).

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues.

**Step 6: Commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_slo_test.go
git commit -m "fix(metrics): SLO attainment counts missing requests as violations (#201, BC-2)

- Requests in RequestE2Es but missing from Requests map are counted
  in the denominator as SLO violations (conservative estimate)
- Log warning when dropped requests are detected

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: ComputeFitness Fails on Unknown Keys

**Contracts Implemented:** BC-7, BC-10

**Files:**
- Modify: `sim/cluster/metrics.go:362-379`
- Modify: `cmd/root.go:422`
- Test: `sim/cluster/metrics_test.go`

**Step 1: Write failing test**

Context: ComputeFitness currently warns and ignores unknown keys. We need it to return an error.

In `sim/cluster/metrics_test.go`, update the existing unknown-key test and add error test:

```go
// TestComputeFitness_UnknownKey_ReturnsError verifies BC-7.
func TestComputeFitness_UnknownKey_ReturnsError(t *testing.T) {
	raw := &RawMetrics{RequestsPerSec: 100.0}
	weights := map[string]float64{"nonexistent": 1.0}

	_, err := ComputeFitness(raw, weights)
	if err == nil {
		t.Error("expected error for unknown key, got nil")
	}
}

func TestComputeFitness_MixedKnownUnknown_ReturnsError(t *testing.T) {
	raw := &RawMetrics{RequestsPerSec: 100.0}
	weights := map[string]float64{"throughput": 0.5, "invalid_key": 0.5}

	_, err := ComputeFitness(raw, weights)
	if err == nil {
		t.Error("expected error when any key is unknown")
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/cluster/... -run "TestComputeFitness_UnknownKey_ReturnsError|TestComputeFitness_MixedKnownUnknown" -v`
Expected: FAIL — tests don't compile because ComputeFitness doesn't return error yet.

**Step 3: Implement ComputeFitness error return**

In `sim/cluster/metrics.go`, change ComputeFitness signature and add validation:

```go
// ComputeFitness computes a weighted fitness score from RawMetrics.
// All metrics are normalized to [0,1] range before weighting:
// - Throughput: value / (value + referenceRPS) — higher is better, saturates at 1.0
// - Latency: 1.0 / (1.0 + value/referenceTicks) — lower is better, 1ms → 0.5
// Returns error for unknown weight keys (BC-7).
func ComputeFitness(metrics *RawMetrics, weights map[string]float64) (FitnessResult, error) {
	// Validate all keys before computing
	for _, key := range sortedKeys(weights) {
		if _, ok := extractMetric(metrics, key); !ok {
			return FitnessResult{}, fmt.Errorf("unknown fitness metric key %q", key)
		}
	}

	result := FitnessResult{
		Components: make(map[string]float64, len(weights)),
	}

	for _, key := range sortedKeys(weights) {
		weight := weights[key]
		value, _ := extractMetric(metrics, key) // already validated
		result.Components[key] = value
		result.Score += value * weight
	}

	return result, nil
}
```

Update all existing tests in `sim/cluster/metrics_test.go` to handle the new `(FitnessResult, error)` return:

- `TestComputeFitness_WeightedScore`: change `result := ComputeFitness(raw, weights)` to `result, err := ComputeFitness(raw, weights)` and add `if err != nil { t.Fatal(err) }`.
- `TestComputeFitness_LatencyInversion`: same pattern for both calls.
- `TestComputeFitness_MultiObjective`: same pattern.
- Delete `TestComputeFitness_UnknownKey_Ignored` (replaced by `TestComputeFitness_UnknownKey_ReturnsError`).

Update `cmd/root.go` (line 422):

```go
			fitness, err := cluster.ComputeFitness(rawMetrics, weights)
			if err != nil {
				logrus.Fatalf("Fitness evaluation failed: %v", err)
			}
```

Also update `EvaluationResult.Fitness` field type from `*FitnessResult` to `*FitnessResult` — wait, since ComputeFitness now returns value `FitnessResult`, the cmd/root.go caller needs to take address:

Actually, looking at `evaluation.go:14`, `Fitness` is `*FitnessResult`. The caller in cmd/root.go doesn't use EvaluationResult. The only direct caller is cmd/root.go:422 which prints the result inline. So we just need to update cmd/root.go.

Since ComputeFitness now returns `FitnessResult` (value, not pointer), update the cmd/root.go usage:

```go
			fitness, err := cluster.ComputeFitness(rawMetrics, weights)
			if err != nil {
				logrus.Fatalf("Fitness evaluation failed: %v", err)
			}
			logrus.Infof("=== Fitness Evaluation ===")
			logrus.Infof("Score: %.6f", fitness.Score)
```

This works because `fitness.Score` and `fitness.Components` work on value receiver too.

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -run "TestComputeFitness" -v`
Expected: PASS

Run: `go test ./cmd/... -count=1`
Expected: PASS (cmd tests still compile)

**Step 5: Run full test suite**

Run: `go test ./... -count=1`
Expected: PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues.

**Step 7: Commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go cmd/root.go
git commit -m "fix(metrics): ComputeFitness returns error for unknown keys (#203, BC-7, BC-10)

- Change signature from *FitnessResult to (FitnessResult, error)
- Validate all weight keys before computing
- CLI caller converts error to logrus.Fatalf
- Replace silent warn+ignore with fail-fast behavior

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Update Documentation

**Contracts Implemented:** BC-9 (verification)

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/plans/hardening-phase3-metric-fixes-plan.md` (this file — mark as executed)

**Step 1: Verify golden dataset is unaffected**

Run: `go test ./sim/... -run "Golden|golden|TestSingleInstance" -v`
Expected: PASS (BC-9: golden dataset unchanged)

**Step 2: Update CLAUDE.md**

In CLAUDE.md, update the KV cache section. Find the line:

```
8. **Avoid calling `GetCachedBlocks` multiple times** — it increments `CacheHits` as a side effect (not a pure query). `TieredKVCache.AllocateKVBlocks` calls it twice on reload; this inflates CacheHits and is tracked as a known issue (design doc Phase 3, 3c).
```

Replace with:

```
8. **`GetCachedBlocks` is a pure query** — it returns cached block IDs without side effects. `CacheHits` are counted by `AllocateKVBlocks` when cached blocks are committed to an allocation (and rolled back on failure). This was fixed in the Phase 3 hardening PR; the previous implementation incremented CacheHits in GetCachedBlocks, causing double-counting in tiered mode.
```

**Step 3: Run full verification**

Run: `go build ./... && go test ./... -count=1 && golangci-lint run ./...`
Expected: All pass.

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 3 CacheHits purity fix

- GetCachedBlocks is now documented as a pure query
- CacheHits counted at AllocateKVBlocks commit, rolled back on failure

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | TestSaveResults_IncludesIncompleteRequests |
| BC-2 | Task 3 | Unit | TestSLOAttainment_MissingRequests_CountedAsViolation |
| BC-3 | Task 1 | Unit | TestGetCachedBlocks_IsPureQuery_NoCacheHitsSideEffect |
| BC-4 | Task 1 | Unit | TestAllocateKVBlocks_CacheHitsCountedAtCommit |
| BC-5 | Task 1 | Unit | TestAllocateKVBlocks_CacheHitsRolledBackOnFailure |
| BC-6 | Task 1 | Unit | TestTieredKVCache_NoCacheHitsDoubleCount |
| BC-7 | Task 4 | Unit | TestComputeFitness_UnknownKey_ReturnsError |
| BC-7 | Task 4 | Unit | TestComputeFitness_MixedKnownUnknown_ReturnsError |
| BC-8 | Task 1 | Regression | Existing golden tests (single-tier CacheHits unchanged) |
| BC-9 | Task 5 | Regression | Golden dataset tests remain passing |
| BC-10 | Task 4 | Integration | cmd tests compile with error handling |

**Golden dataset:** No regeneration needed. Phase 3 fixes don't change simulation behavior — only metric reporting and validation.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| CacheHits rollback breaks existing rollback test | Low | High | Existing `TestAllocateKVBlocks_CachedBlockRollback` runs in Task 1 step 4 | Task 1 |
| ComputeFitness error return breaks downstream code | Medium | Medium | Grep for all call sites (only cmd/root.go:422) | Task 4 |
| SaveResults change affects golden dataset output | Low | Medium | Golden tests verified in Task 5 step 1; single-instance runs complete all requests | Task 5 |
| SLOAttainment log spam in production | Low | Low | Warn logged once per call, not per request | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — all changes are direct fixes
- [x] No feature creep beyond PR scope — 4 items from design doc Phase 3
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates — ComputeFitness signature change documented (BC-7, BC-10)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing shared test package
- [x] CLAUDE.md updated for CacheHits purity fix
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 first due to KV cache interaction)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration NOT needed (verified)
- [x] Construction site audit: FitnessResult constructed at metrics.go:363 (single site, now returns value not pointer)
- [x] No new CLI flags → no numeric validation needed
- [x] Every error path: ComputeFitness returns error; SLOAttainment logs warning + counts in denominator; no silent `continue` that drops data
- [x] No map iteration feeds float accumulation without sorted keys — SaveResults uses sortedRequestIDs, SLOAttainment uses integer addition (associative, order-independent)
- [x] Library code never calls logrus.Fatalf — SLOAttainment uses logrus.Warnf (intentional, not Fatalf)
- [x] No resource allocation loops to worry about (no new allocation patterns)

---

## Appendix: File-Level Implementation Details

### File: `sim/kvcache.go`

**Purpose:** Make GetCachedBlocks a pure query; count CacheHits in AllocateKVBlocks; roll back CacheHits on failure.

**Changes:**
1. `GetCachedBlocks` (line 122-138): Remove `kvc.CacheHits++` at line 134. Update doc comment.
2. `AllocateKVBlocks` (line 144-261): Add `kvc.CacheHits++` after line 191 (inside cached-block loop).
3. `rollbackAllocation` (line 301-333): Add `kvc.CacheHits--` in cached-block undo loop (after line 324). Update doc comment.

**Behavioral notes:**
- Single-tier: CacheHits count is identical to before (one GetCachedBlocks → one AllocateKVBlocks, same K blocks)
- Tiered: CacheHits no longer inflated by double GetCachedBlocks call in reload path
- Rollback: CacheHits decremented per cached mutation, symmetric with increment

### File: `sim/metrics.go`

**Purpose:** Include all registered requests in JSON output, not just completed ones.

**Changes:**
1. Lines 130-143: Replace `for id, ttft := range m.RequestTTFTs` with `for _, id := range sortedRequestIDs(m.Requests)`
2. Add `sortedRequestIDs` helper function

**Behavioral notes:**
- Incomplete requests get zero-valued TTFT/E2E/ITL (Go map lookup returns zero for missing keys)
- Zero values are distinguishable from real measurements (which are always positive in ticks)
- Final output still sorted by ArrivedAt (existing sort.Slice preserved)

### File: `sim/cluster/metrics.go`

**Purpose:** Fix SLO attainment inflation and ComputeFitness silent failure.

**Changes:**
1. `SLOAttainment` (lines 284-310): Add `droppedCount` counter; count dropped requests in `total` but not `met`; log warning.
2. `ComputeFitness` (lines 362-379): Change return type to `(FitnessResult, error)`; validate all keys before computing; return error for unknown keys.

**Behavioral notes:**
- SLOAttainment: `droppedCount > 0` triggers one logrus.Warnf call (not per-request)
- ComputeFitness: returns value `FitnessResult` not pointer `*FitnessResult` (Go convention for error returns)
- `sortedKeys` already used for iteration (from Phase 1)

### File: `cmd/root.go`

**Purpose:** Handle ComputeFitness error at CLI boundary.

**Changes:**
1. Line 422: Change `fitness := cluster.ComputeFitness(rawMetrics, weights)` to `fitness, err := cluster.ComputeFitness(rawMetrics, weights)` with `logrus.Fatalf` on error.
2. Update field access from pointer to value receiver (`.Score`, `.Components`).
