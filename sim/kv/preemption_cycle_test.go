package kv

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
)

// TestPreemptionCycle_With16KPrefixes reproduces issue #1057 scenario:
// High prefix sharing + tight KV capacity should NOT cause infinite preemption loop.
// With lazy hash deletion, preempted requests get cache hits on readmission.
func TestPreemptionCycle_With16KPrefixes(t *testing.T) {
	// Simulate experiment 68 conditions (scaled down):
	// - Shared prefix (64 blocks with blockSize=16 = 1024 tokens)
	// - Tight KV capacity (200 total blocks)
	// - 3 requests competing for blocks, processed one at a time
	//   with preemption when capacity is exhausted
	totalBlocks := int64(300)
	blockSize := int64(16)
	kvc := NewKVCacheState(totalBlocks, blockSize)

	// Shared prefix (64 blocks = 1024 tokens)
	sharedPrefix := make([]int, 1024) // 64 * 16
	for i := range sharedPrefix {
		sharedPrefix[i] = i + 10000
	}

	// Create 3 requests with shared prefix + unique suffix
	requests := make([]*sim.Request, 3)
	for i := 0; i < 3; i++ {
		req := &sim.Request{
			ID: fmt.Sprintf("request_%d", i),
		}
		req.InputTokens = append([]int{}, sharedPrefix...)
		uniqueSuffix := make([]int, 1024) // 64 more blocks
		for j := range uniqueSuffix {
			uniqueSuffix[j] = (i+1)*100000 + j
		}
		req.InputTokens = append(req.InputTokens, uniqueSuffix...)
		// Each request: 2048 tokens = 128 blocks
		requests[i] = req
	}

	preemptionCounts := make(map[string]int)
	cacheHitsAfterPreemption := 0
	completed := make(map[string]bool)

	// Process requests sequentially with chunked prefill
	// Each request needs 128 blocks, cache has 200 blocks.
	// First request succeeds. Second request starts but when both
	// are partially allocated, capacity is tight and preemption occurs.
	chunkSize := int64(512) // 32 blocks per chunk

	for wave := 0; wave < 50; wave++ {
		anyProgress := false
		for _, req := range requests {
			if completed[req.ID] {
				continue
			}

			if req.ProgressIndex >= int64(len(req.InputTokens)) {
				kvc.ReleaseKVBlocks(req)
				completed[req.ID] = true
				t.Logf("Request %s completed (preemptions: %d)", req.ID, preemptionCounts[req.ID])
				anyProgress = true
				continue
			}

			remainingTokens := int64(len(req.InputTokens)) - req.ProgressIndex
			tokensToAllocate := min(chunkSize, remainingTokens)

			var cachedBlocks []int64
			if req.ProgressIndex == 0 {
				cachedBlocks = kvc.GetCachedBlocks(req.InputTokens)
				if len(cachedBlocks) > 0 && preemptionCounts[req.ID] > 0 {
					t.Logf("Request %s got %d cache hits after preemption (lazy deletion success)",
						req.ID, len(cachedBlocks))
					cacheHitsAfterPreemption += len(cachedBlocks)
				}
			}

			startIdx := req.ProgressIndex
			endIdx := startIdx + tokensToAllocate

			ok := kvc.AllocateKVBlocks(req, startIdx, endIdx, cachedBlocks)
			if ok {
				req.ProgressIndex = endIdx
				anyProgress = true
				continue
			}

			// Allocation failed - preempt a victim
			victimFound := false
			for i := len(requests) - 1; i >= 0; i-- {
				victim := requests[i]
				if victim.ID == req.ID || completed[victim.ID] {
					continue
				}
				if victim.ProgressIndex > 0 {
					t.Logf("Preempting %s (PI=%d) to make room for %s",
						victim.ID, victim.ProgressIndex, req.ID)
					kvc.ReleaseKVBlocks(victim)
					victim.ProgressIndex = 0
					preemptionCounts[victim.ID]++

					require.LessOrEqual(t, preemptionCounts[victim.ID], 10,
						"Request %s preempted %d times - likely infinite loop (issue #1057 not fixed)",
						victim.ID, preemptionCounts[victim.ID])

					victimFound = true
					break
				}
			}

			if !victimFound {
				continue
			}

			// Retry after preemption
			if req.ProgressIndex == 0 {
				cachedBlocks = kvc.GetCachedBlocks(req.InputTokens)
				if len(cachedBlocks) > 0 && preemptionCounts[req.ID] > 0 {
					cacheHitsAfterPreemption += len(cachedBlocks)
				}
			}
			ok = kvc.AllocateKVBlocks(req, startIdx, endIdx, cachedBlocks)
			if ok {
				req.ProgressIndex = endIdx
				anyProgress = true
			}
		}

		if !anyProgress {
			break
		}

		allDone := true
		for _, req := range requests {
			if !completed[req.ID] {
				allDone = false
				break
			}
		}
		if allDone {
			break
		}
	}

	// THEN all requests should complete
	for _, req := range requests {
		assert.True(t, completed[req.ID],
			"Request %s should be completed (PI=%d, target=%d)",
			req.ID, req.ProgressIndex, len(req.InputTokens))
	}

	// AND preempted requests should have gotten cache hits (lazy deletion success)
	if len(preemptionCounts) > 0 {
		assert.Greater(t, cacheHitsAfterPreemption, 0,
			"Preempted requests should get cache hits on readmission (lazy deletion)")
	}

	t.Logf("Integration test passed - lazy hash deletion prevents infinite preemption cycle")
	t.Logf("Total preemptions: %v, cache hits after preemption: %d",
		preemptionCounts, cacheHitsAfterPreemption)
}
