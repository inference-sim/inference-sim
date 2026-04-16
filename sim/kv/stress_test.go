package kv

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
)

// assertFullConservation verifies INV-4 via independent free-list walk + InUse scan.
// Delegates to assertBlockConservation which calls verifyBlockConservation internally.
func assertFullConservation(t *testing.T, kvc *KVCacheState) {
	t.Helper()
	assertBlockConservation(t, kvc)
}

// assertAllBlocksFree verifies that UsedBlocks == 0 and countFreeBlocks == TotalBlocks.
func assertAllBlocksFree(t *testing.T, kvc *KVCacheState) {
	t.Helper()
	assert.Equal(t, int64(0), kvc.UsedBlocks(), "all blocks should be free (UsedBlocks == 0)")
	assert.Equal(t, kvc.TotalBlocks, kvc.countFreeBlocks(),
		"countFreeBlocks should equal TotalBlocks when all released")
	assertFullConservation(t, kvc)
}

// makeRequest creates a request with the given ID and token counts.
// Input tokens are sequential integers starting from baseToken.
// Output tokens (if outputLen > 0) are sequential integers starting from baseToken+inputLen.
func makeRequest(id string, inputLen, outputLen int, baseToken int) *sim.Request {
	input := make([]int, inputLen)
	for i := range input {
		input[i] = baseToken + i
	}
	output := make([]int, outputLen)
	for i := range output {
		output[i] = baseToken + inputLen + i
	}
	return &sim.Request{
		ID:           id,
		InputTokens:  input,
		OutputTokens: output,
	}
}

// TestStress_SustainedKVPressureWithPreemptionCycles simulates #963: many
// requests competing for limited KV blocks. Requests arrive faster than they
// complete, forcing repeated preemption (evict + retry) cycles. Verifies that
// all requests eventually complete and block conservation holds throughout.
func TestStress_SustainedKVPressureWithPreemptionCycles(t *testing.T) {
	// Tight cache: 20 blocks, blockSize=4 (80 tokens total capacity)
	const totalBlocks, blockSize = int64(20), int64(4)
	const numRequests = 10
	const tokensPerReq = 8 // 2 blocks per request prefill

	kvc := NewKVCacheState(totalBlocks, blockSize)

	// Phase 1: Allocate prefill for all requests, filling the cache.
	// 10 requests * 2 blocks = 20 blocks = full cache.
	reqs := make([]*sim.Request, numRequests)
	for i := 0; i < numRequests; i++ {
		reqs[i] = makeRequest(fmt.Sprintf("r%d", i), tokensPerReq, 4, i*100)
		ok := kvc.AllocateKVBlocks(reqs[i], 0, int64(tokensPerReq), []int64{})
		require.True(t, ok, "prefill for r%d should succeed", i)
		reqs[i].ProgressIndex = int64(tokensPerReq)
	}
	require.Equal(t, int64(0), kvc.countFreeBlocks(), "cache should be full after prefill")
	assertFullConservation(t, kvc)

	// Phase 2: For each request, attempt decode (fails), evict another request
	// to free blocks, then retry decode (succeeds). This is the preemption cycle.
	evictIdx := 0
	for i := 0; i < numRequests; i++ {
		req := reqs[i]
		decodeStart := int64(tokensPerReq) + int64(i) // progressive decode index

		// Attempt decode — should fail (cache full, last block may be full)
		if kvc.countFreeBlocks() == 0 {
			ids := kvc.RequestMap[req.ID]
			lastBlk := kvc.Blocks[ids[len(ids)-1]]
			if int64(len(lastBlk.Tokens)) == blockSize {
				ok := kvc.AllocateKVBlocks(req, decodeStart, decodeStart+1, []int64{})
				assert.False(t, ok, "decode for r%d should fail when cache is full", i)
				// RequestMap must be preserved after decode failure (#1061)
				assert.NotEmpty(t, kvc.RequestMap[req.ID],
					"RequestMap must survive decode failure for r%d", i)
			}
		}

		// Evict: release another request to free blocks
		// Find a request to evict (one we haven't evicted yet)
		for evictIdx < numRequests {
			if _, exists := kvc.RequestMap[reqs[evictIdx].ID]; exists && evictIdx != i {
				kvc.ReleaseKVBlocks(reqs[evictIdx])
				break
			}
			evictIdx++
		}
		assertFullConservation(t, kvc)

		// Retry decode — should now succeed
		if _, exists := kvc.RequestMap[req.ID]; exists {
			ok := kvc.AllocateKVBlocks(req, decodeStart, decodeStart+1, []int64{})
			assert.True(t, ok, "decode retry for r%d should succeed after eviction", i)
		}
		assertFullConservation(t, kvc)
	}

	// Phase 3: Release all remaining requests.
	for i := 0; i < numRequests; i++ {
		if _, exists := kvc.RequestMap[reqs[i].ID]; exists {
			kvc.ReleaseKVBlocks(reqs[i])
		}
	}

	assertAllBlocksFree(t, kvc)
}

// TestStress_DecodeFailRetryPreservesBlocksThroughManyCycles is a scaled-up
// version of TestPreemptForTokens_RetryPreservesRequestMap. It exercises #1061
// leak path 1 at scale: multiple rounds of decode-fail -> evict -> retry. After
// N cycles, verifies that NO blocks are orphaned.
func TestStress_DecodeFailRetryPreservesBlocksThroughManyCycles(t *testing.T) {
	const totalBlocks, blockSize = int64(30), int64(4)
	const numRequests = 10
	const tokensPerReq = 8 // 2 blocks per request

	kvc := NewKVCacheState(totalBlocks, blockSize)

	// Prefill all requests (10 * 2 = 20 blocks used, 10 free)
	reqs := make([]*sim.Request, numRequests)
	for i := 0; i < numRequests; i++ {
		reqs[i] = makeRequest(fmt.Sprintf("r%d", i), tokensPerReq, 10, i*1000)
		ok := kvc.AllocateKVBlocks(reqs[i], 0, int64(tokensPerReq), []int64{})
		require.True(t, ok, "prefill for r%d should succeed", i)
		reqs[i].ProgressIndex = int64(tokensPerReq)
	}

	// Fill remaining cache with a filler (10 free blocks)
	filler := makeRequest("filler", 40, 0, 50000) // 40 tokens = 10 blocks
	ok := kvc.AllocateKVBlocks(filler, 0, 40, []int64{})
	require.True(t, ok, "filler allocation should succeed")
	require.Equal(t, int64(0), kvc.countFreeBlocks(), "cache should be full")

	// For each request: decode-fail -> verify RequestMap preserved -> evict filler ->
	// retry decode -> verify RequestMap grew -> re-create filler
	for i := 0; i < numRequests; i++ {
		req := reqs[i]
		decodeIdx := int64(tokensPerReq) // decode token index

		// Check last block status to determine if decode will fail
		ids := kvc.RequestMap[req.ID]
		mapLenBefore := len(ids)
		lastBlk := kvc.Blocks[ids[len(ids)-1]]

		if int64(len(lastBlk.Tokens)) == blockSize {
			// Last block full, 0 free -> must fail
			ok := kvc.AllocateKVBlocks(req, decodeIdx, decodeIdx+1, []int64{})
			assert.False(t, ok, "decode for r%d should fail (cache full)", i)

			// RequestMap must be preserved (not deleted by rollback — #1061)
			assert.Len(t, kvc.RequestMap[req.ID], mapLenBefore,
				"RequestMap length must be unchanged after decode failure for r%d", i)
		}

		// Release filler to make room
		kvc.ReleaseKVBlocks(filler)
		assert.Greater(t, kvc.countFreeBlocks(), int64(0), "filler release should free blocks")
		assertFullConservation(t, kvc)

		// Retry decode — should succeed
		ok = kvc.AllocateKVBlocks(req, decodeIdx, decodeIdx+1, []int64{})
		assert.True(t, ok, "decode retry for r%d should succeed after filler eviction", i)
		assert.Len(t, kvc.RequestMap[req.ID], mapLenBefore+1,
			"RequestMap should grow by 1 after successful decode for r%d", i)
		assertFullConservation(t, kvc)

		// Re-create filler to re-fill cache for the next iteration
		fillerTokens := 40 - int(kvc.countFreeBlocks())*int(blockSize)
		if kvc.countFreeBlocks() > 0 {
			filler = makeRequest("filler", int(kvc.countFreeBlocks())*int(blockSize), 0, 50000+i*1000)
			ok = kvc.AllocateKVBlocks(filler, 0, int64(len(filler.InputTokens)), []int64{})
			require.True(t, ok, "filler re-creation should succeed (cycle %d, filling %d blocks)", i, fillerTokens)
		}
	}

	// Release all requests
	for i := 0; i < numRequests; i++ {
		if _, exists := kvc.RequestMap[reqs[i].ID]; exists {
			kvc.ReleaseKVBlocks(reqs[i])
		}
	}
	// Release filler if still held
	if _, exists := kvc.RequestMap[filler.ID]; exists {
		kvc.ReleaseKVBlocks(filler)
	}

	assertAllBlocksFree(t, kvc)
}

// TestStress_CachedBlockBudgetExhaustionUnderPressure simulates the scenario
// where cached blocks on the free list consume free-block budget. Multiple
// requests with shared prefix compete for limited blocks. Verifies that the
// pre-check correctly accounts for cached blocks and rejects cleanly (#1057).
func TestStress_CachedBlockBudgetExhaustionUnderPressure(t *testing.T) {
	// 8 blocks, blockSize=2
	kvc := NewKVCacheState(8, 2)

	// Step 1: Allocate and release a request to populate prefix cache.
	// 4 tokens = 2 blocks become cached on the free list.
	req1 := &sim.Request{ID: "prefix-seed", InputTokens: []int{1, 2, 3, 4}}
	ok := kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	require.True(t, ok)
	kvc.ReleaseKVBlocks(req1)
	// Now: 8 free blocks, 2 of which are cached prefix blocks with hashes

	// Step 2: Fill 5 blocks with fillers (8 free -> 3 free).
	// But we want to leave only 2 free blocks (the cached ones) + some pressure.
	// Fill 6 blocks -> leaves 2 free (both are the cached prefix blocks).
	fillers := make([]*sim.Request, 3)
	for i := 0; i < 3; i++ {
		fillers[i] = &sim.Request{
			ID:          fmt.Sprintf("filler%d", i),
			InputTokens: []int{100 + i*10, 101 + i*10, 102 + i*10, 103 + i*10},
		}
		ok := kvc.AllocateKVBlocks(fillers[i], 0, 4, []int64{})
		require.True(t, ok, "filler %d allocation should succeed", i)
	}
	// 3 fillers * 2 blocks = 6 used. 2 free blocks remain (the cached prefix blocks).
	require.Equal(t, int64(2), kvc.countFreeBlocks(), "should have 2 free blocks (cached prefix)")
	assertFullConservation(t, kvc)

	// Step 3: Try to allocate a new request with the same prefix plus new tokens.
	// Prefix [1,2,3,4] = 2 cached blocks (will be claimed from free list).
	// New tokens [5,6,7,8] = 2 more blocks needed from free list.
	// After claiming cached blocks: 0 free remain. Need 2 more -> should fail.
	req2 := &sim.Request{ID: "pressure-req", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)
	require.Len(t, cached, 2, "should find 2 cached prefix blocks")

	freeBefore := kvc.countFreeBlocks()
	ok = kvc.AllocateKVBlocks(req2, 4, 8, cached)

	// THEN: allocation fails, no state mutated
	assert.False(t, ok, "allocation should fail (cached blocks consume free budget)")
	assert.Equal(t, freeBefore, kvc.countFreeBlocks(),
		"free block count must be unchanged after pre-check rejection")
	assert.Empty(t, kvc.RequestMap["pressure-req"],
		"no RequestMap entry should exist after rejection")
	assertFullConservation(t, kvc)

	// Step 4: Verify that freeing a filler allows the allocation to succeed.
	kvc.ReleaseKVBlocks(fillers[0])
	require.Equal(t, int64(4), kvc.countFreeBlocks(), "should have 4 free after releasing filler")

	cached = kvc.GetCachedBlocks(req2.InputTokens)
	ok = kvc.AllocateKVBlocks(req2, 4, 8, cached)
	assert.True(t, ok, "allocation should succeed after freeing a filler")
	assertFullConservation(t, kvc)

	// Cleanup
	kvc.ReleaseKVBlocks(req2)
	for i := 1; i < 3; i++ {
		kvc.ReleaseKVBlocks(fillers[i])
	}
	assertAllBlocksFree(t, kvc)
}

// TestStress_BlockConservationThroughCompleteLifecycle is a comprehensive
// lifecycle test: allocate, decode, release, re-allocate with prefix cache hits,
// decode again, release again. After every operation, verifies INV-4 conservation.
func TestStress_BlockConservationThroughCompleteLifecycle(t *testing.T) {
	const totalBlocks, blockSize = int64(20), int64(4)
	const numRounds = 5
	const reqsPerRound = 3
	const inputTokensPerReq = 8  // 2 blocks per request
	const decodeStepsPerReq = 2  // 2 decode tokens per request

	kvc := NewKVCacheState(totalBlocks, blockSize)

	for round := 0; round < numRounds; round++ {
		t.Run(fmt.Sprintf("round_%d", round), func(t *testing.T) {
			reqs := make([]*sim.Request, reqsPerRound)

			// Phase A: Allocate prefill for 3 requests
			for i := 0; i < reqsPerRound; i++ {
				// Use consistent tokens so prefix cache can hit on subsequent rounds.
				// Each request i uses tokens [i*100..i*100+7] for input.
				reqs[i] = makeRequest(
					fmt.Sprintf("round%d-r%d", round, i),
					inputTokensPerReq, decodeStepsPerReq+2,
					i*100,
				)

				// Check for cached prefix blocks (from previous rounds)
				cached := kvc.GetCachedBlocks(reqs[i].InputTokens)
				startIdx := int64(len(cached)) * blockSize
				ok := kvc.AllocateKVBlocks(reqs[i], startIdx, int64(inputTokensPerReq), cached)
				require.True(t, ok, "round %d: prefill for r%d should succeed", round, i)
				reqs[i].ProgressIndex = int64(inputTokensPerReq)
				assertFullConservation(t, kvc)
			}

			// Phase B: Decode each request 2 times
			for step := 0; step < decodeStepsPerReq; step++ {
				for i := 0; i < reqsPerRound; i++ {
					req := reqs[i]
					decodeIdx := int64(inputTokensPerReq + step)
					ok := kvc.AllocateKVBlocks(req, decodeIdx, decodeIdx+1, []int64{})
					assert.True(t, ok, "round %d: decode step %d for r%d should succeed", round, step, i)
					assertFullConservation(t, kvc)
				}
			}

			// Phase C: Release all 3 requests
			for i := 0; i < reqsPerRound; i++ {
				kvc.ReleaseKVBlocks(reqs[i])
				assertFullConservation(t, kvc)
			}

			// Phase D: Verify conservation and free/used accounting
			assert.Equal(t, kvc.TotalBlocks, kvc.countFreeBlocks()+kvc.UsedBlocks(),
				"round %d: countFreeBlocks + UsedBlocks must equal TotalBlocks", round)
		})
	}

	// Final: all blocks must be free after all rounds
	assertAllBlocksFree(t, kvc)
}
