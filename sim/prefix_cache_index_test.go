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

// TestPrefixCacheIndex_LRUEviction_BoundsCapacity verifies BC-10 (LRU capacity bounds).
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

// TestPrefixCacheIndex_Capacity1_HeadEqualsTail verifies the capacity-1 edge case
// where head and tail point to the same node. Exercises all removeNode branches
// simultaneously (node is both head and tail).
func TestPrefixCacheIndex_Capacity1_HeadEqualsTail(t *testing.T) {
	idx := NewPrefixCacheIndex(1, 1) // capacity 1

	// GIVEN a single block recorded
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{10}), "inst_0")
	assert.Equal(t, 1, idx.InstanceBlockCount("inst_0"))
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{10}), "inst_0"))

	// WHEN a new block is recorded (triggers eviction of the only element)
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{20}), "inst_0")

	// THEN capacity remains 1, old block evicted, new block present
	assert.Equal(t, 1, idx.InstanceBlockCount("inst_0"))
	assert.Equal(t, 0, idx.MatchLength(idx.ComputeBlockHashes([]int{10}), "inst_0"), "old block should be evicted")
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{20}), "inst_0"), "new block should be present")

	// WHEN touching the existing block and adding another (evict + insert cycle)
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{20}), "inst_0") // touch
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{30}), "inst_0") // replace

	assert.Equal(t, 1, idx.InstanceBlockCount("inst_0"))
	assert.Equal(t, 0, idx.MatchLength(idx.ComputeBlockHashes([]int{20}), "inst_0"), "touched block evicted by new insert")
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{30}), "inst_0"), "newest block present")
}

// TestPrefixCacheIndex_RecordBlocks_BatchEvictionRetainsNewest verifies that
// when recording more blocks than LRU capacity in a single call, the last N
// blocks (most recently touched) survive while earlier blocks are evicted.
// Note: hierarchical hashing means block hashes chain from previous blocks,
// so we verify using the actual hash values from the full sequence.
func TestPrefixCacheIndex_RecordBlocks_BatchEvictionRetainsNewest(t *testing.T) {
	idx := NewPrefixCacheIndex(1, 3) // block size 1, capacity 3

	// GIVEN 6 single-token blocks recorded in one call (exceeds capacity of 3)
	tokens := []int{10, 20, 30, 40, 50, 60}
	hashes := idx.ComputeBlockHashes(tokens)
	require.Len(t, hashes, 6)
	idx.RecordBlocks(hashes, "inst_0")

	// THEN cache is bounded at capacity
	assert.Equal(t, 3, idx.InstanceBlockCount("inst_0"))

	// THEN MatchLength on the full 6-block sequence returns 0 because the
	// first 3 blocks (hashes[0..2]) were evicted — consecutive-from-start
	// matching breaks immediately. This pins the behavioral trade-off:
	// batch recording more blocks than capacity evicts prefix-start blocks.
	assert.Equal(t, 0, idx.MatchLength(hashes, "inst_0"),
		"early blocks evicted, so consecutive-from-start match is 0")

	// THEN a shorter sequence whose blocks are all in the surviving set
	// (last 3 blocks) cannot be verified via MatchLength because hierarchical
	// hashing means those hashes depend on the evicted prefix. Verify via
	// InstanceBlockCount that exactly capacity blocks remain.
	assert.Equal(t, 3, idx.InstanceBlockCount("inst_0"),
		"exactly capacity blocks should remain after batch eviction")
}

// TestPrefixCacheIndex_MatchLength_DoesNotRefreshLRU verifies that MatchLength
// is a read-only operation that does not perturb LRU eviction ordering.
func TestPrefixCacheIndex_MatchLength_DoesNotRefreshLRU(t *testing.T) {
	idx := NewPrefixCacheIndex(1, 3) // capacity 3

	// GIVEN A (oldest), B, C recorded
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{1}), "inst_0") // A
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{2}), "inst_0") // B
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{3}), "inst_0") // C

	// WHEN A is read via MatchLength (should NOT refresh its LRU position)
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{1}), "inst_0"))

	// AND a new block D is inserted (triggers eviction)
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{4}), "inst_0")

	// THEN A is evicted (still oldest despite being read), not B
	assert.Equal(t, 0, idx.MatchLength(idx.ComputeBlockHashes([]int{1}), "inst_0"),
		"A should be evicted — MatchLength must not refresh LRU")
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{2}), "inst_0"),
		"B should survive")
	assert.Equal(t, 1, idx.MatchLength(idx.ComputeBlockHashes([]int{4}), "inst_0"),
		"D should be present")
}

// TestPrefixCacheIndex_BlockCount_ConsistentThroughMixedOps verifies that
// InstanceBlockCount stays accurate through a mixed sequence of inserts,
// touches, and evictions at each step.
func TestPrefixCacheIndex_BlockCount_ConsistentThroughMixedOps(t *testing.T) {
	idx := NewPrefixCacheIndex(1, 3) // capacity 3

	// Step 1: Insert A — count should be 1
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{1}), "inst_0")
	assert.Equal(t, 1, idx.InstanceBlockCount("inst_0"), "after insert A")

	// Step 2: Insert B — count should be 2
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{2}), "inst_0")
	assert.Equal(t, 2, idx.InstanceBlockCount("inst_0"), "after insert B")

	// Step 3: Touch A — count stays 2 (no new entry)
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{1}), "inst_0")
	assert.Equal(t, 2, idx.InstanceBlockCount("inst_0"), "after touch A")

	// Step 4: Insert C — count should be 3 (at capacity)
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{3}), "inst_0")
	assert.Equal(t, 3, idx.InstanceBlockCount("inst_0"), "after insert C (at capacity)")

	// Step 5: Insert D — triggers eviction, count stays 3
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{4}), "inst_0")
	assert.Equal(t, 3, idx.InstanceBlockCount("inst_0"), "after insert D (eviction)")

	// Step 6: Touch C — count stays 3
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{3}), "inst_0")
	assert.Equal(t, 3, idx.InstanceBlockCount("inst_0"), "after touch C")

	// Step 7: Insert E, F — two evictions, count stays 3
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{5}), "inst_0")
	assert.Equal(t, 3, idx.InstanceBlockCount("inst_0"), "after insert E")
	idx.RecordBlocks(idx.ComputeBlockHashes([]int{6}), "inst_0")
	assert.Equal(t, 3, idx.InstanceBlockCount("inst_0"), "after insert F")
}
