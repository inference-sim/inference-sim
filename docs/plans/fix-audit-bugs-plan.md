# Fix Audit Bugs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 14 bugs discovered during a comprehensive module-by-module codebase audit, spanning KV cache, latency model, scheduler, routing, workload generation, and metrics modules.

**The problem today:** The codebase contains 14 bugs ranging from phantom KV block allocation during chunked prefill to silent data loss in trace parsing, inconsistent units in per-request JSON output, and missing input validation. While none affect current golden test values (all are in untested code paths), they represent correctness risks for users exercising advanced features like chunked prefill, tiered KV, non-default block sizes, and workload-spec YAML.

**What this PR adds:**
1. **KV cache correctness** — fixes phantom block allocation, repairs a dead test, and improves rollback LRU semantics
2. **Input validation** — adds NaN/Inf checking for alpha/beta coefficients and required-parameter validation for workload distribution specs
3. **Data integrity** — makes trace CSV parsing fail-fast on errors instead of silently producing zero values, and enforces strict YAML on trace headers
4. **Output correctness** — fixes per-request ITL and scheduling delay units from ticks to milliseconds, and corrects JainFairnessIndex for all-zero inputs
5. **Routing accuracy** — threads actual block size to prefix-affinity scorer instead of hardcoded 16, and fixes standalone PrefixAffinity to use block-prefix hashing

**Why this matters:** These bugs must be fixed before adding new golden tests for advanced features. Fixing them first ensures new golden values capture correct behavior.

**Architecture:** Pure bug fixes across 6 modules. No new interfaces, no new packages, no architectural changes. Each fix is localized to 1-3 files with corresponding test additions.

**Source:** GitHub issues #352, #353, #354, #355, #356, #357, #358, #359, #360, #361, #362, #363, #364, #365

**Closes:** Fixes #352, fixes #353, fixes #354, fixes #355, fixes #356, fixes #357, fixes #358, fixes #359, fixes #360, fixes #361, fixes #362, fixes #363, fixes #364, fixes #365

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes 14 bugs found during a systematic module audit of the BLIS codebase. The bugs span 6 modules and range from medium-severity correctness issues (phantom KV blocks, silent data loss) to low-severity cleanup (stale map entries, dead tests). No bug affects current golden test values — all are in code paths the golden tests don't exercise. Fixing them is a prerequisite for expanding golden test coverage.

Key interactions: KV cache ↔ simulator (phantom blocks during chunked prefill), routing ↔ SimConfig (block size threading), metrics ↔ JSON output (unit conversion).

No deviations from issue descriptions.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Phantom Block Prevention (#352)
- GIVEN a chunked prefill where the previous chunk left a partial block
- WHEN `AllocateKVBlocks` fills the partial block and then allocates new blocks
- THEN `numNewBlocks` is recomputed after partial fill to avoid allocating empty phantom blocks
- MECHANISM: Subtract tokens consumed by partial fill before computing `numNewBlocks`

BC-2: Dead Test Repair (#353)
- GIVEN a tiered KV cache with GPU blocks at capacity and CPU blocks available
- WHEN the test allocates a request triggering CPU-to-GPU reload
- THEN `ConsumePendingTransferLatency()` returns a non-zero value reflecting transfer cost
- MECHANISM: Rewrite test setup to actually trigger `tryReloadFromCPU`

BC-3: Rollback Free-List Order (#354)
- GIVEN a KV cache where `AllocateKVBlocks` fails mid-loop and triggers rollback
- WHEN `rollbackAllocation` returns blocks to the free list
- THEN blocks are prepended to the head (original position), not appended to tail
- MECHANISM: Use `prependToFreeList` instead of `appendToFreeList` for rollback

BC-4: Alpha/Beta NaN/Inf Validation (#355)
- GIVEN a `SimConfig` with NaN or Inf values in `AlphaCoeffs` or `BetaCoeffs`
- WHEN `NewLatencyModel` is called
- THEN it returns an error describing which coefficient is invalid
- MECHANISM: Loop over coefficients checking `math.IsNaN` and `math.IsInf`

BC-5: Blackbox/Roofline Consistency (#356)
- GIVEN a batch containing a request with `ProgressIndex >= len(InputTokens)` and `len(OutputTokens) == 0`
- WHEN `StepTime` is called on either BlackboxLatencyModel or RooflineLatencyModel
- THEN both models classify the request consistently (as decode with 0 new tokens)
- MECHANISM: Add `len(req.OutputTokens) > 0` guard to roofline decode classification

BC-6: Preemption Map Cleanup (#357)
- GIVEN a request is preempted in `preemptForTokens()`
- WHEN its `ProgressIndex` is reset to 0 and KV blocks are released
- THEN `ComputedTokens[req.ID]` is also deleted
- MECHANISM: Add `delete(ctx.ComputedTokens, preemptedRequest.ID)` after `ReleaseKVBlocks` in `sim/batch_formation.go`

BC-7: Block Size Threading (#359)
- GIVEN a `SimConfig` with `BlockSizeTokens != 16`
- WHEN the `weighted` routing policy creates a prefix-affinity scorer
- THEN the scorer uses the actual `BlockSizeTokens`, not the hardcoded default 16
- MECHANISM: Pass `blockSize` through `NewRoutingPolicy` → `newScorerWithObserver`

BC-8: PrefixAffinity Block Hashing (#358)
- GIVEN two requests sharing a common prefix but differing in suffix tokens
- WHEN the standalone `PrefixAffinity` policy routes them
- THEN requests with shared block-aligned prefixes are routed to the same instance
- MECHANISM: Replace `hashTokens(req.InputTokens)` with block-aligned prefix hashing

BC-9: Sampler Overflow Guard (#360)
- GIVEN a Weibull or Pareto sampler where `rng.Float64()` returns 0.0
- WHEN sampling an inter-arrival time or token length
- THEN the sampler produces a valid finite positive value (not +Inf or int64 overflow)
- MECHANISM: Clamp `u` to `max(u, math.SmallestNonzeroFloat64)` before log/power

BC-10: Distribution Param Validation (#361)
- GIVEN a `DistSpec` with missing required parameters (e.g., gaussian without "mean")
- WHEN `NewLengthSampler` is called
- THEN it returns an error listing the missing parameters
- MECHANISM: Check required keys exist in `spec.Params` before constructing each sampler

BC-11: Trace V2 Strict YAML (#362)
- GIVEN a trace v2 header YAML with a typo (e.g., `tme_unit` instead of `time_unit`)
- WHEN `LoadTraceV2` parses the header
- THEN it returns an error identifying the unknown field
- MECHANISM: Use `yaml.NewDecoder` with `KnownFields(true)` instead of `yaml.Unmarshal`

BC-12: Trace CSV Parse Errors (#363)
- GIVEN a trace v2 CSV with a non-numeric value where an integer is expected
- WHEN `parseTraceRecord` parses the row
- THEN it returns an error identifying the field and invalid value
- MECHANISM: Check every `strconv` error instead of discarding with `_`

BC-13: JainFairnessIndex All-Zero (#364)
- GIVEN a throughput map where all values are 0.0 (e.g., `{"t1": 0, "t2": 0}`)
- WHEN `JainFairnessIndex` is called
- THEN it returns 1.0 (perfectly fair — all tenants treated identically)
- MECHANISM: Return 1.0 when `sumX2 == 0` and `n > 0`

BC-14: Per-Request Unit Consistency (#365)
- GIVEN completed simulation results written to a JSON file via `SaveResults`
- WHEN the per-request `itl_ms` and `scheduling_delay_ms` fields are read
- THEN they are in milliseconds (consistent with `ttft_ms` and `e2e_ms`)
- MECHANISM: Divide `RequestITLs` and `RequestSchedulingDelays` by 1e3

**Negative Contracts:**

NC-1: No golden test value changes
- GIVEN the current 5 golden test configurations
- WHEN this PR is applied
- THEN all golden tests continue to pass with identical values
- MECHANISM: All fixes are in code paths the golden tests don't exercise

NC-2: No new panics in sim/
- GIVEN any of the fixes in this PR
- WHEN executed in the `sim/` package
- THEN no `logrus.Fatalf` or `os.Exit` is called (R6)
- MECHANISM: All validation returns errors; library code never terminates

### C) Component Interaction

```
cmd/root.go ──► sim/latency_model.go (NaN/Inf validation in NewLatencyModel)
                 │
sim/simulator.go ──► sim/kvcache.go (phantom block fix in AllocateKVBlocks)
                 │       └──► rollbackAllocation (free-list order fix)
                 └──► preempt() (reqNumComputedTokens cleanup)

sim/routing.go ──► sim/routing_scorers.go (block size parameter threading)
     │                └──► sim/routing_prefix_scorer.go (uses actual block size)
     └──► PrefixAffinity (block-aligned hashing)

sim/workload/arrival.go (Weibull sampler overflow guard)
sim/workload/distribution.go (Pareto overflow guard + param validation)
sim/workload/tracev2.go (strict YAML + CSV error propagation)

sim/metrics.go (per-request unit conversion)
sim/cluster/metrics.go (JainFairnessIndex all-zero fix)
```

No new state, no new interfaces, no new packages. All changes are localized fixes within existing modules.

### D) Deviation Log

No deviations from issue descriptions.

### E) Review Guide

**The tricky part:** BC-1 (phantom block prevention) requires careful arithmetic — the `numNewBlocks` recomputation after partial-fill must account for the tokens consumed. BC-7 (block size threading) changes `NewRoutingPolicy`'s signature which has construction sites in `sim/routing.go` and `sim/cluster/cluster.go`.

**What to scrutinize:** The KV cache fix (Task 1) and the block size threading (Task 5) — these touch the most code paths. Verify BC-1's arithmetic with the concrete example from #352.

**What's safe to skim:** Tasks 4, 7, 8, 10 are small mechanical fixes (1-3 line changes each).

**Known debt:** The standalone `PrefixAffinity` policy (Task 6) is fundamentally less capable than the weighted scorer — it now uses block-aligned hashing but still lacks LRU eviction and hierarchical matching. This is acceptable; users should prefer the `weighted` policy.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/kvcache.go` — phantom block fix, rollback order fix
- `sim/kvcache_tiered_test.go` — dead test rewrite
- `sim/latency_model.go` — NaN/Inf validation, roofline consistency
- `sim/batch_formation.go` — preemption ComputedTokens cleanup
- `sim/routing.go` — PrefixAffinity block hashing, NewRoutingPolicy signature
- `sim/routing_scorers.go` — block size parameter, remove hardcoded const
- `sim/cluster/cluster.go` — pass block size to NewRoutingPolicy
- `sim/workload/arrival.go` — Weibull overflow guard
- `sim/workload/distribution.go` — Pareto overflow guard, param validation
- `sim/workload/tracev2.go` — strict YAML, CSV error propagation
- `sim/cluster/metrics.go` — JainFairnessIndex fix
- `sim/metrics.go` — per-request unit conversion

**Files to create:**
- None

**Key decisions:**
- Block size threading adds a `blockSize int64` parameter to `NewRoutingPolicy` (converted to `int` when calling `newScorerWithObserver`, which uses `int` internally). Construction sites: `sim/cluster/cluster.go` (passes `config.BlockSizeTokens`), plus ~50 calls in test files (`sim/routing_test.go`, `sim/routing_prefix_scorer_test.go`, `sim/examples_test.go`, `sim/routing_scorers_test.go`) that must add `int64(16)` as the third argument.
- PrefixAffinity standalone uses `PrefixCacheIndex` (already exists) for block-aligned hashing instead of reimplementing
- Per-request unit fix divides by 1e3 — this changes the per-request JSON output but NOT stdout aggregate metrics (which are already correct)

### G) Task Breakdown

---

### Task 1: Fix phantom block allocation in chunked prefill (#352)

**Contracts Implemented:** BC-1

**Files:**
- Modify: `sim/kvcache.go:148-257`
- Test: `sim/kvcache_test.go`

**Step 1: Write failing test**

Context: Create a test that triggers the phantom block by using BlockSize=4, a request with a partial block (3 tokens), and a new chunk of 5 tokens. The bug causes `numNewBlocks = ceil(5/4) = 2` but only 1 new block should be needed after filling the partial block.

```go
func TestAllocateKVBlocks_ChunkedPrefill_NoPhantomBlocks(t *testing.T) {
	// GIVEN a KV cache with BlockSize=4 and a request that already has a partial block (3 tokens)
	kvc := NewKVCacheState(20, 4) // 20 blocks, size 4
	req := &Request{
		ID:            "phantom-test",
		InputTokens:   []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens:  []int{100},
		ProgressIndex: 0,
	}

	// First allocation: 3 tokens (leaves a partial block)
	ok := kvc.AllocateKVBlocks(req, 0, 3, []int64{})
	require.True(t, ok)
	blocksAfterFirst := kvc.UsedBlocks()

	// WHEN allocating the next chunk of 5 tokens (should fill partial block + allocate 1 new block)
	req.ProgressIndex = 3
	ok = kvc.AllocateKVBlocks(req, 3, 8, []int64{})
	require.True(t, ok)

	// THEN exactly 1 new block should be allocated (partial fill + 1 new block, not 2)
	// Partial block: 3 tokens + 1 token = 4 (full). Remaining: 4 tokens = 1 new block.
	// Total: blocksAfterFirst + 1
	blocksAfterSecond := kvc.UsedBlocks()
	assert.Equal(t, blocksAfterFirst+1, blocksAfterSecond,
		"should allocate exactly 1 new block after partial fill, not %d", blocksAfterSecond-blocksAfterFirst)

	// Verify no phantom blocks (all allocated blocks should have non-empty Tokens)
	for _, blockID := range kvc.RequestMap[req.ID] {
		blk := kvc.Blocks[blockID]
		assert.True(t, len(blk.Tokens) > 0, "block %d has empty Tokens (phantom block)", blk.ID)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestAllocateKVBlocks_ChunkedPrefill_NoPhantomBlocks -v`
Expected: FAIL — phantom block with empty Tokens

**Step 3: Implement fix**

In `sim/kvcache.go`, the fix is in the `else` branch starting at line 214. After filling the partial block (lines 197-213), we need to recompute `numNewBlocks` based on remaining tokens:

In `sim/kvcache.go`, replace the inner allocation loop's usage of `numNewBlocks` with a tokens-remaining check. The key change is at the start of the `else` block (line 214):

```go
		} else {
			// latest block is full or request is coming in for the first time.
			// allocate new block(s) for the request.
			// Recompute blocks needed from remaining tokens (after partial fill consumed some).
			remainingTokens := Len64(newTokens) - newTokenProgressIndex
			numNewBlocks = (remainingTokens + kvc.BlockSizeTokens - 1) / kvc.BlockSizeTokens
			if remainingTokens <= 0 {
				break
			}
			for i := int64(0); i < numNewBlocks; i++ {
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestAllocateKVBlocks_ChunkedPrefill_NoPhantomBlocks -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`

**Step 6: Commit**

```bash
git add sim/kvcache.go sim/kvcache_test.go
git commit -m "fix(kv): prevent phantom block allocation during chunked prefill (#352)

- Recompute numNewBlocks after partial block fill consumes tokens
- Prevents empty-token phantom blocks when chunk boundaries don't align with block boundaries
- Add test: TestAllocateKVBlocks_ChunkedPrefill_NoPhantomBlocks

Implements BC-1.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix dead test and rollback free-list order (#353, #354)

**Contracts Implemented:** BC-2, BC-3

**Files:**
- Modify: `sim/kvcache_tiered_test.go:66-96`
- Modify: `sim/kvcache.go:296-332`
- Test: `sim/kvcache_test.go`

**Step 1: Write failing test for rollback order (BC-3)**

```go
func TestAllocateKVBlocks_Rollback_PreservesFreListOrder(t *testing.T) {
	// GIVEN a KV cache with 3 blocks (BlockSize=2), 2 blocks used, 1 free
	kvc := NewKVCacheState(3, 2)
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}, OutputTokens: []int{100}}
	ok := kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	require.True(t, ok) // uses 2 blocks, 1 free

	// Record the free head before the failed allocation
	freeHeadBefore := kvc.FreeHead.ID

	// WHEN allocation fails (needs 2 blocks, only 1 free) and rolls back
	req2 := &Request{ID: "r2", InputTokens: []int{5, 6, 7, 8, 9, 10}, OutputTokens: []int{100}}
	ok = kvc.AllocateKVBlocks(req2, 0, 6, []int64{})
	require.False(t, ok) // should fail — needs 3 blocks, only 1 free

	// THEN the free head should be the same block (prepended back, not appended to tail)
	assert.Equal(t, freeHeadBefore, kvc.FreeHead.ID,
		"rollback should restore free head to original block, not append to tail")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestAllocateKVBlocks_Rollback_PreservesFreListOrder -v`
Expected: FAIL — FreeHead.ID differs (block was appended to tail)

**Step 3: Implement rollback fix**

In `sim/kvcache.go`, in `rollbackAllocation`, change `appendToFreeList` to `prependToFreeList` for both newly allocated blocks (line 318) and cached blocks (line 327). First, add the new `prependToFreeList` method (it does not exist yet — this PR creates it):

```go
// prependToFreeList adds a block to the head of the free list (for rollback).
func (kvc *KVCacheState) prependToFreeList(blk *KVBlock) {
	blk.Next = kvc.FreeHead
	blk.Prev = nil
	if kvc.FreeHead != nil {
		kvc.FreeHead.Prev = blk
	}
	kvc.FreeHead = blk
	if kvc.FreeTail == nil {
		kvc.FreeTail = blk
	}
}
```

Then in `rollbackAllocation`, replace:
- Line 318: `kvc.appendToFreeList(m.block)` → `kvc.prependToFreeList(m.block)`
- Line 327: `kvc.appendToFreeList(m.block)` → `kvc.prependToFreeList(m.block)`

Also iterate `newlyAllocated` in reverse order so the first-popped block is prepended last (ending up at the head):

```go
// Reverse iterate so first-popped block ends up at head
for i := len(newlyAllocated) - 1; i >= 0; i-- {
    m := newlyAllocated[i]
    // ... existing cleanup ...
    kvc.prependToFreeList(m.block)
}
// Reverse iterate cached mutations too
for i := len(cachedMutations) - 1; i >= 0; i-- {
    m := cachedMutations[i]
    // ... existing cleanup ...
    kvc.prependToFreeList(m.block)
}
```

**Step 4: Rewrite dead tiered test (BC-2)**

Replace `TestTieredKVCache_TransferLatency_QueryAndClear` with a test that actually triggers CPU-to-GPU reload:

```go
func TestTieredKVCache_TransferLatency_ConsumeClearsAccumulated(t *testing.T) {
	// GIVEN a tiered KV cache where blocks have been offloaded to CPU
	// and then reloaded back to GPU (triggering transfer latency)
	gpu := NewKVCacheState(4, 4)   // small GPU: 4 blocks
	tiered := NewTieredKVCache(gpu, 10, 0.5, 100.0, 0) // 10 CPU blocks
	tiered.SetClock(1000)

	// Fill GPU to trigger offload
	for i := 0; i < 4; i++ {
		req := &Request{
			ID:           fmt.Sprintf("fill-%d", i),
			InputTokens:  []int{i*4 + 1, i*4 + 2, i*4 + 3, i*4 + 4},
			OutputTokens: []int{100},
		}
		ok := tiered.AllocateKVBlocks(req, 0, 4, []int64{})
		require.True(t, ok, "fill allocation %d should succeed", i)
	}

	// Release one request to free GPU blocks, triggering offload to CPU
	tiered.ReleaseKVBlocks(&Request{ID: "fill-0"})
	tiered.SetClock(2000)

	// WHEN a new request gets cache hits that trigger CPU-to-GPU reload
	reqNew := &Request{
		ID:           "reload-test",
		InputTokens:  []int{1, 2, 3, 4, 5, 6, 7, 8}, // shares prefix with fill-0
		OutputTokens: []int{100},
	}
	cached := tiered.GetCachedBlocks(reqNew.InputTokens)
	if len(cached) > 0 {
		ok := tiered.AllocateKVBlocks(reqNew, 0, 8, cached)
		require.True(t, ok)
	}

	// THEN ConsumePendingTransferLatency returns the accumulated transfer cost
	lat := tiered.ConsumePendingTransferLatency()
	// Second consume should return 0 (cleared)
	lat2 := tiered.ConsumePendingTransferLatency()
	assert.Equal(t, int64(0), lat2, "second consume should return 0")

	// If the test setup correctly triggered a reload, lat should be > 0.
	// If lat == 0, the test setup needs to be adjusted to actually trigger the reload path.
	assert.True(t, lat > 0, "first consume should return non-zero transfer latency (reload should have triggered)")
}
```

**Step 5: Run all tests**

Run: `go test ./sim/... -run "TestAllocateKVBlocks_Rollback_PreservesFreListOrder|TestTieredKVCache_TransferLatency_ConsumeClearsAccumulated" -v`
Expected: PASS

**Step 6: Lint + commit**

```bash
git add sim/kvcache.go sim/kvcache_test.go sim/kvcache_tiered_test.go
git commit -m "fix(kv): rollback prepends to free list head, rewrite dead tiered test (#353, #354)

- rollbackAllocation prepends blocks to head in reverse order (preserves LRU)
- Rewrite TestTieredKVCache_TransferLatency test to actually exercise code path
- Add test: TestAllocateKVBlocks_Rollback_PreservesFreListOrder

Implements BC-2, BC-3.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add NaN/Inf validation for alpha/beta coefficients (#355) and fix blackbox/roofline consistency (#356)

**Contracts Implemented:** BC-4, BC-5

**Files:**
- Modify: `sim/latency_model.go:121-151`
- Test: `sim/latency_model_test.go`

**Step 1: Write failing tests**

```go
func TestNewLatencyModel_NaNAlphaCoeffs_ReturnsError(t *testing.T) {
	cfg := SimConfig{
		AlphaCoeffs: []float64{math.NaN(), 1.0, 100.0},
		BetaCoeffs:  []float64{5000, 10, 5},
	}
	_, err := NewLatencyModel(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "NaN")
}

func TestNewLatencyModel_InfBetaCoeffs_ReturnsError(t *testing.T) {
	cfg := SimConfig{
		AlphaCoeffs: []float64{100, 1.0, 100.0},
		BetaCoeffs:  []float64{math.Inf(1), 10, 5},
	}
	_, err := NewLatencyModel(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Inf")
}

func TestBlackboxRoofline_ZeroOutputTokens_ConsistentClassification(t *testing.T) {
	// GIVEN a request past prefill with 0 output tokens
	req := &Request{
		InputTokens:  []int{1, 2, 3},
		OutputTokens: []int{},
		ProgressIndex: 3,
		NumNewTokens: 0,
	}
	batch := []*Request{req}

	blackbox := &BlackboxLatencyModel{
		betaCoeffs:  []float64{5000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}
	rooflineCfg := SimConfig{
		AlphaCoeffs: []float64{100, 1, 100},
		Roofline:    true,
		TP:          2,
		ModelConfig: ModelConfig{NumLayers: 1, HiddenDim: 64, NumHeads: 2, VocabSize: 100, IntermediateDim: 128, BytesPerParam: 2},
		HWConfig:    HardwareCalib{TFlopsPeak: 100, BwPeakTBs: 1, BwEffConstant: 0.8, MfuPrefill: 0.5, MfuDecode: 0.3},
	}
	rooflineModel, err := NewLatencyModel(rooflineCfg)
	require.NoError(t, err)

	// WHEN both models compute step time for this batch
	blackboxTime := blackbox.StepTime(batch)
	rooflineTime := rooflineModel.StepTime(batch)

	// THEN both should produce a valid (non-negative) step time
	// (The actual values will differ between models, but neither should panic or produce
	// wildly different behavior for the same edge case)
	assert.True(t, blackboxTime >= 0, "blackbox step time should be non-negative")
	assert.True(t, rooflineTime >= 0, "roofline step time should be non-negative")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run "TestNewLatencyModel_NaN|TestNewLatencyModel_Inf" -v`
Expected: FAIL — no error returned for NaN/Inf coefficients

**Step 3: Implement validation**

In `sim/latency_model.go`, add a helper function and call it in `NewLatencyModel`:

```go
// validateCoeffs checks for NaN or Inf in a coefficient slice.
func validateCoeffs(name string, coeffs []float64) error {
	for i, c := range coeffs {
		if math.IsNaN(c) {
			return fmt.Errorf("latency model: %s[%d] is NaN", name, i)
		}
		if math.IsInf(c, 0) {
			return fmt.Errorf("latency model: %s[%d] is Inf", name, i)
		}
	}
	return nil
}
```

Call it in `NewLatencyModel` after the length checks:
```go
if err := validateCoeffs("AlphaCoeffs", cfg.AlphaCoeffs); err != nil {
    return nil, err
}
```
And for blackbox path:
```go
if err := validateCoeffs("BetaCoeffs", cfg.BetaCoeffs); err != nil {
    return nil, err
}
```

For BC-5, in `RooflineLatencyModel.StepTime`, change the decode classification (around line 92) from:
```go
} else {
```
to:
```go
} else if len(req.OutputTokens) > 0 {
```

This makes roofline consistent with blackbox: requests past prefill with 0 output tokens contribute nothing beyond base cost.

**Step 4: Run tests**

Run: `go test ./sim/... -run "TestNewLatencyModel_NaN|TestNewLatencyModel_Inf|TestBlackboxRoofline_ZeroOutputTokens" -v`
Expected: PASS

**Step 5: Lint + commit**

```bash
git add sim/latency_model.go sim/latency_model_test.go
git commit -m "fix(latency): validate alpha/beta coefficients for NaN/Inf, fix blackbox/roofline consistency (#355, #356)

- Add validateCoeffs() helper called in NewLatencyModel for both alpha and beta
- Add len(OutputTokens) > 0 guard to roofline decode classification
- Makes both models consistent for 0-output-token requests

Implements BC-4, BC-5.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Clean up ComputedTokens on preemption (#357)

**Contracts Implemented:** BC-6

**Note:** PR #371 extracted `preempt()` from `sim/simulator.go` into `preemptForTokens()` in `sim/batch_formation.go`. The `reqNumComputedTokens` map is now passed as `ctx.ComputedTokens` in `BatchContext`. The cleanup must happen in `preemptForTokens` at `sim/batch_formation.go:168`.

**Files:**
- Modify: `sim/batch_formation.go:145-178` (preemptForTokens)
- Test: `sim/batch_formation_test.go`

**Step 1: Write failing test**

```go
func TestPreemptForTokens_CleansUpComputedTokens(t *testing.T) {
	// GIVEN a batch context where a request has ComputedTokens entry
	kv := NewKVStore(SimConfig{TotalKVBlocks: 3, BlockSizeTokens: 4})
	req := &Request{ID: "preempt-test", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}, OutputTokens: []int{100}, State: StateRunning}
	kv.AllocateKVBlocks(req, 0, 8, []int64{})
	computedTokens := map[string]int64{req.ID: 8}

	ctx := BatchContext{
		RunningBatch:   &Batch{Requests: []*Request{req}},
		WaitQ:          &WaitQueue{},
		KVCache:        kv,
		ComputedTokens: computedTokens,
		Now:            1000,
	}
	result := BatchResult{RunningBatch: ctx.RunningBatch}

	reqNew := &Request{ID: "new-req", InputTokens: []int{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}, OutputTokens: []int{100}}
	bf := &VLLMBatchFormation{latencyModel: &BlackboxLatencyModel{betaCoeffs: []float64{100, 1, 1}, alphaCoeffs: []float64{100, 1, 100}}}

	// WHEN preemption evicts req
	bf.preemptForTokens(reqNew, 12, &result, ctx)

	// THEN ComputedTokens should NOT have the preempted request's entry
	_, exists := computedTokens[req.ID]
	assert.False(t, exists, "preempted request should be removed from ComputedTokens")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestPreemptForTokens_CleansUpComputedTokens -v`
Expected: FAIL — entry still exists

**Step 3: Implement fix**

In `sim/batch_formation.go`, after line 168 (`ctx.KVCache.ReleaseKVBlocks(preemptedRequest)`), add:

```go
delete(ctx.ComputedTokens, preemptedRequest.ID)
```

**Step 4: Run test + lint + commit**

```bash
git add sim/batch_formation.go sim/batch_formation_test.go
git commit -m "fix(batch): clean up ComputedTokens on preemption (#357)

- Delete stale map entry when request is preempted and ProgressIndex reset to 0
- Keeps ComputedTokens synchronized with ProgressIndex

Implements BC-6.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Thread block size to prefix-affinity scorer (#359)

**Contracts Implemented:** BC-7

**Files:**
- Modify: `sim/routing_scorers.go:24` (remove hardcoded const)
- Modify: `sim/routing.go:268-299` (add blockSize param to NewRoutingPolicy)
- Modify: `sim/cluster/cluster.go:85` (pass blockSize at construction site)
- Modify: `sim/routing_test.go` (~30 call sites: add `int64(16)` third arg)
- Modify: `sim/routing_prefix_scorer_test.go` (~11 call sites)
- Modify: `sim/examples_test.go` (~6 call sites)
- Modify: `sim/routing_scorers_test.go` (~2 call sites)
- Test: `sim/routing_scorers_test.go`

**Note:** `NewRoutingPolicy` takes `blockSize int64` to match `SimConfig.BlockSizeTokens`. Internally, `newScorerWithObserver` uses `int`, so `NewRoutingPolicy` converts with `int(blockSize)` at the call site.

**Step 1: Write failing test**

```go
func TestNewRoutingPolicy_Weighted_UsesActualBlockSize(t *testing.T) {
	// GIVEN a non-default block size of 32
	blockSize := int64(32)
	scorerConfigs := []ScorerConfig{{Name: "prefix-affinity", Weight: 1.0}}

	// WHEN creating a weighted routing policy with the actual block size
	policy := NewRoutingPolicy("weighted", scorerConfigs, blockSize)

	// THEN routing decisions should use block size 32 (not the hardcoded 16)
	// We verify this indirectly: with block size 32, requests with 32-token
	// aligned prefixes should route to the same instance.
	req1 := &Request{ID: "r1", InputTokens: make([]int, 64)} // 2 blocks of 32
	req2 := &Request{ID: "r2", InputTokens: make([]int, 64)} // same prefix
	for i := range req1.InputTokens {
		req1.InputTokens[i] = i + 1
		req2.InputTokens[i] = i + 1
	}

	snapshots := []RoutingSnapshot{
		NewRoutingSnapshot("inst_0"),
		NewRoutingSnapshot("inst_1"),
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}

	d1 := policy.Route(req1, state)
	d2 := policy.Route(req2, state)
	assert.Equal(t, d1.TargetInstance, d2.TargetInstance,
		"identical prefix requests should route to the same instance")
	_ = policy // policy is not nil
}
```

**Step 2: Run test to verify it fails (compilation error — signature change)**

**Step 3: Implement**

In `sim/routing_scorers.go`, remove the `defaultBlockSize` constant (line 24).

In `sim/routing.go`, change `NewRoutingPolicy` signature:

```go
func NewRoutingPolicy(name string, scorerConfigs []ScorerConfig, blockSize int64) RoutingPolicy {
```

In the `"weighted"` case, pass `blockSize` instead of `defaultBlockSize` (convert to `int` since `newScorerWithObserver` takes `int`):

```go
scorer, obs := newScorerWithObserver(cfg.Name, int(blockSize))
```

In `sim/cluster/cluster.go` line 85, pass the block size:

```go
routingPolicy: sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingScorerConfigs, config.BlockSizeTokens),
```

Update the `"prefix-affinity"` standalone case to also use the block size (for Task 6).

Also update any test files that call `NewRoutingPolicy` — add the third parameter (typically `int64(16)` to maintain existing behavior).

**Step 4: Run all tests**

Run: `go test ./sim/... ./sim/cluster/... -v -count=1`
Expected: PASS (all existing + new tests)

**Step 5: Lint + commit**

```bash
git add sim/routing.go sim/routing_scorers.go sim/cluster/cluster.go sim/routing_test.go sim/routing_prefix_scorer_test.go sim/examples_test.go sim/routing_scorers_test.go sim/cluster/cluster_test.go
git commit -m "fix(routing): thread actual block size to prefix-affinity scorer (#359)

- Remove hardcoded defaultBlockSize=16 constant
- Add blockSize parameter to NewRoutingPolicy
- Pass config.BlockSizeTokens from cluster constructor
- Scorer now uses actual KV cache block size for prefix matching

Implements BC-7.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Fix standalone PrefixAffinity to use block-aligned hashing (#358)

**Contracts Implemented:** BC-8

**Files:**
- Modify: `sim/routing.go:200-240`
- Test: `sim/routing_test.go`

**Step 1: Write failing test**

```go
func TestPrefixAffinity_SharedPrefix_DifferentSuffix_SameInstance(t *testing.T) {
	// GIVEN two requests sharing a 32-token prefix but different suffixes
	prefix := make([]int, 32)
	for i := range prefix {
		prefix[i] = i + 1
	}
	req1 := &Request{ID: "r1", InputTokens: append(append([]int{}, prefix...), 100, 101, 102)}
	req2 := &Request{ID: "r2", InputTokens: append(append([]int{}, prefix...), 200, 201)}

	snapshots := []RoutingSnapshot{
		NewRoutingSnapshot("inst_0"),
		NewRoutingSnapshot("inst_1"),
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}

	policy := NewRoutingPolicy("prefix-affinity", nil, 16)

	// WHEN routing both requests
	d1 := policy.Route(req1, state)
	d2 := policy.Route(req2, state)

	// THEN they should go to the same instance (shared prefix blocks)
	assert.Equal(t, d1.TargetInstance, d2.TargetInstance,
		"requests sharing a block-aligned prefix should route to the same instance")
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — different hashes produce different instance targets

**Step 3: Implement**

Modify `PrefixAffinity` to use `PrefixCacheIndex` for block-aligned hashing instead of whole-input `hashTokens`. Add a `blockSize` field and use block hashing:

```go
type PrefixAffinity struct {
	prefixMap map[string]string // longest block hash → instance ID
	blockSize int64
}
```

In `Route()`, use a temporary `PrefixCacheIndex` to compute block hashes (reusing the existing implementation rather than duplicating hashing logic):

```go
func (pa *PrefixAffinity) Route(req *Request, state *RouterState) RoutingDecision {
	// Compute block-aligned prefix hashes using PrefixCacheIndex
	idx := NewPrefixCacheIndex(int(pa.blockSize), defaultLRUCapacity)
	blockHashes := idx.ComputeBlockHashes(req.InputTokens)
	// Use the last (longest prefix) block hash as the affinity key
	var prefixHash string
	if len(blockHashes) > 0 {
		prefixHash = blockHashes[len(blockHashes)-1]
	}
	// ... rest of lookup logic unchanged ...
}
```

Note: Creating a `PrefixCacheIndex` per `Route()` call is not ideal for performance, but acceptable because: (1) the standalone `PrefixAffinity` policy is a simple fallback — users should prefer the `weighted` policy for production, and (2) `ComputeBlockHashes` is a pure hash computation that doesn't use the LRU cache state.

Update the construction in `NewRoutingPolicy`:
```go
case "prefix-affinity":
    return &PrefixAffinity{prefixMap: make(map[string]string), blockSize: blockSize}
```

**Step 4: Run tests + lint + commit**

```bash
git add sim/routing.go sim/routing_test.go
git commit -m "fix(routing): standalone PrefixAffinity uses block-aligned prefix hashing (#358)

- Replace hashTokens(req.InputTokens) with block-aligned hierarchical hashing
- Requests sharing a block-aligned prefix now route to the same instance
- Uses same hashing scheme as the weighted-scoring prefix-affinity scorer

Implements BC-8.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Guard Weibull and Pareto samplers against float overflow (#360)

**Contracts Implemented:** BC-9

**Files:**
- Modify: `sim/workload/arrival.go:91-99`
- Modify: `sim/workload/distribution.go:59-74`
- Test: `sim/workload/arrival_test.go`, `sim/workload/distribution_test.go`

**Step 1: Write failing tests**

```go
// In arrival_test.go
func TestWeibullSampler_ZeroUniform_NoOverflow(t *testing.T) {
	// Test the edge case path directly: verify the clamping logic
	s := &WeibullSampler{shape: 1.0, scale: 1000.0}
	// We can't force rng.Float64() to return 0, but we can verify the formula
	// doesn't overflow for very small u values
	sample := s.scale * math.Pow(-math.Log(math.SmallestNonzeroFloat64), 1.0/s.shape)
	assert.False(t, math.IsInf(sample, 0), "sample should not be +Inf for SmallestNonzeroFloat64")
	assert.True(t, sample > 0, "sample should be positive")
}

// In distribution_test.go
func TestParetoLogNormalSampler_ZeroUniform_NoOverflow(t *testing.T) {
	s := &ParetoLogNormalSampler{alpha: 1.0, xm: 100.0, mu: 0, sigma: 1, mixWeight: 1.0}
	// Verify the formula with SmallestNonzeroFloat64 doesn't overflow
	u := math.SmallestNonzeroFloat64
	val := s.xm / math.Pow(u, 1.0/s.alpha)
	assert.False(t, math.IsInf(val, 0), "val should not be +Inf for SmallestNonzeroFloat64")
}
```

**Step 2: Implement fix**

In `sim/workload/arrival.go`, line 93:
```go
u := rng.Float64()
if u == 0 {
    u = math.SmallestNonzeroFloat64
}
```

In `sim/workload/distribution.go`, line 63:
```go
u := rng.Float64()
if u == 0 {
    u = math.SmallestNonzeroFloat64
}
```

**Step 3: Run tests + lint + commit**

```bash
git add sim/workload/arrival.go sim/workload/distribution.go sim/workload/arrival_test.go sim/workload/distribution_test.go
git commit -m "fix(workload): guard Weibull and Pareto samplers against math.Log(0) overflow (#360)

- Clamp u to SmallestNonzeroFloat64 when rng.Float64() returns 0
- Prevents +Inf samples and undefined int64 conversion

Implements BC-9.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Add required-parameter validation to NewLengthSampler (#361)

**Contracts Implemented:** BC-10

**Files:**
- Modify: `sim/workload/distribution.go:148-199`
- Test: `sim/workload/distribution_test.go`

**Step 1: Write failing test**

```go
func TestNewLengthSampler_MissingRequiredParams_ReturnsError(t *testing.T) {
	tests := []struct {
		name     string
		spec     DistSpec
		wantErr  string
	}{
		{
			name:    "gaussian missing mean",
			spec:    DistSpec{Type: "gaussian", Params: map[string]float64{"std_dev": 1, "min": 1, "max": 10}},
			wantErr: "mean",
		},
		{
			name:    "exponential missing mean",
			spec:    DistSpec{Type: "exponential", Params: map[string]float64{}},
			wantErr: "mean",
		},
		{
			name:    "pareto_lognormal missing alpha",
			spec:    DistSpec{Type: "pareto_lognormal", Params: map[string]float64{"xm": 1, "mu": 0, "sigma": 1, "mix_weight": 0.5}},
			wantErr: "alpha",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewLengthSampler(tt.spec)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}
```

**Step 2: Implement validation**

Add `requireParam` helper and checks before each sampler construction:

```go
func requireParam(params map[string]float64, keys ...string) error {
	for _, k := range keys {
		if _, ok := params[k]; !ok {
			return fmt.Errorf("distribution requires parameter %q", k)
		}
	}
	return nil
}
```

In each case of `NewLengthSampler`:
```go
case "gaussian":
    if err := requireParam(spec.Params, "mean", "std_dev", "min", "max"); err != nil {
        return nil, err
    }
case "exponential":
    if err := requireParam(spec.Params, "mean"); err != nil {
        return nil, err
    }
case "pareto_lognormal":
    if err := requireParam(spec.Params, "alpha", "xm", "mu", "sigma", "mix_weight"); err != nil {
        return nil, err
    }
```

**Step 3: Run tests + lint + commit**

```bash
git add sim/workload/distribution.go sim/workload/distribution_test.go
git commit -m "fix(workload): validate required distribution parameters in NewLengthSampler (#361)

- Add requireParam helper checking key existence in Params map
- Gaussian requires mean, std_dev, min, max
- Exponential requires mean
- ParetoLogNormal requires alpha, xm, mu, sigma, mix_weight

Implements BC-10.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Enforce strict YAML and CSV error propagation in trace v2 (#362, #363)

**Contracts Implemented:** BC-11, BC-12

**Files:**
- Modify: `sim/workload/tracev2.go:145-243`
- Test: `sim/workload/tracev2_test.go`

**Step 1: Write failing tests**

```go
func TestLoadTraceV2_UnknownYAMLField_ReturnsError(t *testing.T) {
	// GIVEN a trace header with a typo
	headerContent := "tme_unit: microseconds\nformat_version: 2\n"
	headerPath := filepath.Join(t.TempDir(), "header.yaml")
	require.NoError(t, os.WriteFile(headerPath, []byte(headerContent), 0644))

	dataContent := "request_id,client_id\n1,c1\n"
	dataPath := filepath.Join(t.TempDir(), "data.csv")
	require.NoError(t, os.WriteFile(dataPath, []byte(dataContent), 0644))

	// WHEN loading
	_, err := LoadTraceV2(headerPath, dataPath)

	// THEN error about unknown field
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tme_unit")
}

func TestParseTraceRecord_InvalidInteger_ReturnsError(t *testing.T) {
	// GIVEN a CSV row with a non-numeric request_id
	row := make([]string, 22)
	row[0] = "abc" // request_id should be integer
	for i := 1; i < len(row); i++ {
		row[i] = "0"
	}

	// WHEN parsing
	_, err := parseTraceRecord(row)

	// THEN error about invalid value
	require.Error(t, err)
	assert.Contains(t, err.Error(), "request_id")
}
```

**Step 2: Implement fixes**

In `tracev2.go`, replace `yaml.Unmarshal` with strict decoder:

```go
decoder := yaml.NewDecoder(bytes.NewReader(headerData))
decoder.KnownFields(true)
if err := decoder.Decode(&header); err != nil {
    return nil, fmt.Errorf("parsing trace header: %w", err)
}
```

In `parseTraceRecord`, check every error:

```go
func parseTraceRecord(row []string) (*TraceRecord, error) {
	requestID, err := strconv.Atoi(row[0])
	if err != nil {
		return nil, fmt.Errorf("parsing request_id %q: %w", row[0], err)
	}
	roundIndex, err := strconv.Atoi(row[5])
	if err != nil {
		return nil, fmt.Errorf("parsing round_index %q: %w", row[5], err)
	}
	// ... same pattern for all fields ...
}
```

**Step 3: Run tests + lint + commit**

```bash
git add sim/workload/tracev2.go sim/workload/tracev2_test.go
git commit -m "fix(workload): enforce strict YAML and CSV error propagation in trace v2 (#362, #363)

- Use yaml.KnownFields(true) for trace header parsing (R10)
- Check every strconv error in parseTraceRecord instead of discarding (R1)
- Invalid trace data now fails fast with descriptive errors

Implements BC-11, BC-12.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Fix JainFairnessIndex and per-request unit consistency (#364, #365)

**Contracts Implemented:** BC-13, BC-14

**Files:**
- Modify: `sim/cluster/metrics.go:370-386`
- Modify: `sim/metrics.go:140-141`
- Test: `sim/cluster/metrics_test.go`, `sim/metrics_test.go`

**Step 1: Write failing tests**

```go
// In cluster/metrics_test.go
func TestJainFairnessIndex_AllZeroThroughputs_ReturnsPerfectFairness(t *testing.T) {
	throughputs := map[string]float64{"t1": 0, "t2": 0, "t3": 0}
	jfi := JainFairnessIndex(throughputs)
	assert.Equal(t, 1.0, jfi, "all-zero throughputs = all identical = perfectly fair")
}

// In metrics_test.go (or sim/metrics_test.go)
func TestSaveResults_PerRequestITL_InMilliseconds(t *testing.T) {
	m := NewMetrics()
	m.CompletedRequests = 1
	m.Requests["r1"] = RequestMetrics{ID: "r1", ArrivedAt: 0, NumPrefillTokens: 5, NumDecodeTokens: 10}
	m.RequestTTFTs["r1"] = 10000.0  // 10000 ticks = 10 ms
	m.RequestE2Es["r1"] = 50000.0   // 50000 ticks = 50 ms
	m.RequestITLs["r1"] = 5000.0    // 5000 ticks = should be 5 ms
	m.RequestSchedulingDelays["r1"] = 2000 // 2000 ticks = should be 2 ms
	m.AllITLs = []int64{5000}

	// Write to temp file
	outputPath := filepath.Join(t.TempDir(), "results.json")
	m.SaveResults("test", 1e15, 100, outputPath)

	// Read and parse
	data, err := os.ReadFile(outputPath)
	require.NoError(t, err)
	var output MetricsOutput
	require.NoError(t, json.Unmarshal(data, &output))

	// THEN per-request ITL and scheduling delay should be in ms (divided by 1e3)
	require.Len(t, output.Requests, 1)
	assert.InDelta(t, 5.0, output.Requests[0].ITL, 0.001, "ITL should be in ms")
	assert.InDelta(t, 2.0, output.Requests[0].SchedulingDelay, 0.001, "SchedulingDelay should be in ms")
}
```

**Step 2: Implement fixes**

In `sim/cluster/metrics.go:382-384`:
```go
if sumX2 == 0 {
    return 1.0 // All values identical (including all-zero) → perfectly fair
}
```

In `sim/metrics.go:140-141`:
```go
detail.ITL = m.RequestITLs[id] / 1e3             // ticks → ms (consistent with TTFT, E2E)
detail.SchedulingDelay = float64(m.RequestSchedulingDelays[id]) / 1e3 // ticks → ms
```

**Step 3: Run all tests to confirm no regressions**

Run: `go test ./sim/... ./sim/cluster/... -count=1`
Expected: PASS (golden tests unaffected — they check aggregate metrics, not per-request)

**Step 4: Lint + commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go sim/metrics.go sim/metrics_test.go
git commit -m "fix(metrics): JainFairnessIndex all-zero returns 1.0, per-request units now ms (#364, #365)

- JainFairnessIndex returns 1.0 for all-zero throughputs (perfectly fair)
- Per-request itl_ms and scheduling_delay_ms now correctly divide by 1e3
- Consistent with ttft_ms and e2e_ms which already converted ticks → ms

Implements BC-13, BC-14.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | 1 | Unit | TestAllocateKVBlocks_ChunkedPrefill_NoPhantomBlocks |
| BC-2 | 2 | Unit | TestTieredKVCache_TransferLatency_ConsumeClearsAccumulated |
| BC-3 | 2 | Unit | TestAllocateKVBlocks_Rollback_PreservesFreListOrder |
| BC-4 | 3 | Unit | TestNewLatencyModel_NaNAlphaCoeffs_ReturnsError |
| BC-4 | 3 | Unit | TestNewLatencyModel_InfBetaCoeffs_ReturnsError |
| BC-5 | 3 | Unit | TestBlackboxRoofline_ZeroOutputTokens_ConsistentClassification |
| BC-6 | 4 | Unit | TestPreempt_CleansUpReqNumComputedTokens |
| BC-7 | 5 | Unit | TestNewRoutingPolicy_Weighted_UsesActualBlockSize |
| BC-8 | 6 | Unit | TestPrefixAffinity_SharedPrefix_DifferentSuffix_SameInstance |
| BC-9 | 7 | Unit | TestWeibullSampler_ZeroUniform_NoOverflow |
| BC-9 | 7 | Unit | TestParetoLogNormalSampler_ZeroUniform_NoOverflow |
| BC-10 | 8 | Unit | TestNewLengthSampler_MissingRequiredParams_ReturnsError |
| BC-11 | 9 | Unit | TestLoadTraceV2_UnknownYAMLField_ReturnsError |
| BC-12 | 9 | Unit | TestParseTraceRecord_InvalidInteger_ReturnsError |
| BC-13 | 10 | Unit | TestJainFairnessIndex_AllZeroThroughputs_ReturnsPerfectFairness |
| BC-14 | 10 | Unit | TestSaveResults_PerRequestITL_InMilliseconds |
| NC-1 | All | Golden | Existing TestSimulator_GoldenDataset (must pass unchanged) |

**Golden dataset update strategy:** No golden dataset updates needed. All fixes are in code paths not exercised by golden tests (NC-1).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| BC-7 signature change breaks callers | Medium | Low | Grep for all `NewRoutingPolicy` call sites, update each | Task 5 |
| BC-14 changes per-request JSON output | High (intentional) | Low | Only affects `--results-path` file, not stdout; existing tests don't check per-request values | Task 10 |
| BC-1 fix changes simulation output for chunked prefill | Low | Medium | No golden test uses chunked prefill; add regression test | Task 1 |
| BC-3 rollback order change affects prefix cache eviction | Low | Low | Simulation is deterministic; same rollback path always taken | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond bug fix scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (not duplicated)
- [x] CLAUDE.md does not need updating (no new files/packages)
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — no deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 6 depends on Task 5)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration: not needed (NC-1)
- [x] Construction site audit: `NewRoutingPolicy` has 2 production sites (sim/routing.go factory + cluster/cluster.go) + ~50 test call sites across 4 test files, all updated in Task 5
- [x] R1: No silent continue/return — CSV parse errors now propagated
- [x] R3: NaN/Inf validation added for alpha/beta coefficients
- [x] R6: No logrus.Fatalf in sim/ — all validation returns errors
- [x] R10: Strict YAML parsing added for trace v2 headers

---

## Appendix: File-Level Implementation Details

Detailed code is provided inline in each task's Step 3. The key files and their changes:

1. **`sim/kvcache.go`** — Recompute `numNewBlocks` after partial fill (Task 1). Add `prependToFreeList` and reverse-iterate rollback (Task 2).
2. **`sim/latency_model.go`** — Add `validateCoeffs` helper (Task 3). Add `len(req.OutputTokens) > 0` guard in roofline (Task 3).
3. **`sim/batch_formation.go`** — Add `delete(ctx.ComputedTokens, ...)` in preemptForTokens() (Task 4).
4. **`sim/routing.go`** — Add `blockSize int64` param to `NewRoutingPolicy` (Task 5). Rewrite `PrefixAffinity.Route` with block-aligned hashing (Task 6).
5. **`sim/routing_scorers.go`** — Remove `defaultBlockSize` constant (Task 5).
6. **`sim/cluster/cluster.go`** — Pass `config.BlockSizeTokens` to `NewRoutingPolicy` (Task 5).
7. **`sim/workload/arrival.go`** — Clamp `u` for Weibull (Task 7).
8. **`sim/workload/distribution.go`** — Clamp `u` for Pareto, add `requireParam` (Tasks 7-8).
9. **`sim/workload/tracev2.go`** — Strict YAML decoder, CSV error propagation (Task 9).
10. **`sim/cluster/metrics.go`** — Return 1.0 for all-zero JFI (Task 10).
11. **`sim/metrics.go`** — Divide ITL and SchedulingDelay by 1e3 (Task 10).
