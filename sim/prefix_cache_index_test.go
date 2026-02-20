package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPrefixCacheIndex_HierarchicalHashing_SharedPrefix verifies BC-4:
// Two requests sharing the first K blocks produce identical hashes for those blocks.
func TestPrefixCacheIndex_HierarchicalHashing_SharedPrefix(t *testing.T) {
	idx := NewPrefixCacheIndex(4, 100) // block size 4, capacity 100

	// GIVEN two token sequences sharing first 8 tokens (2 blocks) but different suffix
	tokensA := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}  // 3 blocks
	tokensB := []int{1, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97, 96} // 3 blocks, different block 3

	hashesA := idx.ComputeBlockHashes(tokensA)
	hashesB := idx.ComputeBlockHashes(tokensB)

	require.Len(t, hashesA, 3)
	require.Len(t, hashesB, 3)

	// THEN first 2 block hashes are identical (shared prefix)
	assert.Equal(t, hashesA[0], hashesB[0], "block 0 hashes must match")
	assert.Equal(t, hashesA[1], hashesB[1], "block 1 hashes must match")
	// THEN third block hash differs (different suffix)
	assert.NotEqual(t, hashesA[2], hashesB[2], "block 2 hashes must differ")
}

// TestPrefixCacheIndex_ShortPrefix_ZeroBlocks verifies BC-6:
// Requests shorter than one block produce no block hashes.
func TestPrefixCacheIndex_ShortPrefix_ZeroBlocks(t *testing.T) {
	idx := NewPrefixCacheIndex(16, 100) // block size 16

	// GIVEN 10 tokens (< 16 block size)
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	hashes := idx.ComputeBlockHashes(tokens)

	// THEN no block hashes produced
	assert.Len(t, hashes, 0)
}

// TestPrefixCacheIndex_MatchLength_ConsecutiveFromStart verifies match counting.
func TestPrefixCacheIndex_MatchLength_ConsecutiveFromStart(t *testing.T) {
	idx := NewPrefixCacheIndex(4, 100)

	// Record 3 blocks for instance "inst_0"
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	hashes := idx.ComputeBlockHashes(tokens)
	idx.RecordBlocks(hashes, "inst_0")

	// WHEN looking up a request with same 2-block prefix but different third block
	queryTokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97, 96}
	queryHashes := idx.ComputeBlockHashes(queryTokens)

	// THEN match length is 2 (first 2 consecutive blocks match)
	matched := idx.MatchLength(queryHashes, "inst_0")
	assert.Equal(t, 2, matched)

	// THEN unknown instance has 0 matches
	matched = idx.MatchLength(queryHashes, "inst_1")
	assert.Equal(t, 0, matched)
}

// TestPrefixCacheIndex_LRUEviction_BoundsCapacity verifies BC-10 (INV-7).
func TestPrefixCacheIndex_LRUEviction_BoundsCapacity(t *testing.T) {
	idx := NewPrefixCacheIndex(1, 3) // block size 1, capacity 3 per instance

	// Record 5 single-block hashes for "inst_0" (exceeds capacity of 3)
	for i := 0; i < 5; i++ {
		hashes := idx.ComputeBlockHashes([]int{i * 10}) // each is 1 block
		idx.RecordBlocks(hashes, "inst_0")
	}

	// THEN cache is bounded at capacity (3 blocks)
	assert.Equal(t, 3, idx.InstanceBlockCount("inst_0"))
	// THEN oldest blocks (tokens 0, 10) were evicted; newest (20, 30, 40) remain
	assert.Equal(t, 0, idx.MatchLength(idx.ComputeBlockHashes([]int{0}), "inst_0"))
	assert.Equal(t, 0, idx.MatchLength(idx.ComputeBlockHashes([]int{10}), "inst_0"))
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{20}), "inst_0"))
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{30}), "inst_0"))
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{40}), "inst_0"))
}

// TestPrefixCacheIndex_EmptyTokens_NoHashes verifies edge case.
func TestPrefixCacheIndex_EmptyTokens_NoHashes(t *testing.T) {
	idx := NewPrefixCacheIndex(16, 100)
	hashes := idx.ComputeBlockHashes([]int{})
	assert.Len(t, hashes, 0)
}

// TestPrefixCacheIndex_Deterministic verifies INV-3.
func TestPrefixCacheIndex_Deterministic(t *testing.T) {
	idx1 := NewPrefixCacheIndex(4, 100)
	idx2 := NewPrefixCacheIndex(4, 100)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	h1 := idx1.ComputeBlockHashes(tokens)
	h2 := idx2.ComputeBlockHashes(tokens)

	assert.Equal(t, h1, h2, "same inputs must produce same hashes")
}

// TestPrefixCacheIndex_RecordTouches_PreventEviction verifies LRU touch semantics.
func TestPrefixCacheIndex_RecordTouches_PreventEviction(t *testing.T) {
	idx := NewPrefixCacheIndex(1, 3) // capacity 3

	// Record blocks A, B, C for inst_0
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{1}), "inst_0") // A
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{2}), "inst_0") // B
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{3}), "inst_0") // C

	// Touch A again (re-record it)
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{1}), "inst_0") // A refreshed

	// Record D — should evict B (oldest untouched), not A
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{4}), "inst_0") // D

	// A should still be present (was refreshed)
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{1}), "inst_0"), "A should survive (touched)")
	// B should be evicted
	assert.Equal(t, 0, idx.MatchLength(idx.ComputeBlockHashes([]int{2}), "inst_0"), "B should be evicted")
	// C and D should be present
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{3}), "inst_0"), "C should survive")
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{4}), "inst_0"), "D should be present")
}
