package kv

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// --- cpuTier unit tests (BC-4, BC-1 touch) ---

func TestCpuTier_Store_EvictsOldestWhenFull(t *testing.T) {
	// BC-4: GIVEN CPU tier with capacity 2
	cpu := newCpuTier(2, 4) // capacity=2, blockSize=4 tokens

	// WHEN we store 3 blocks
	cpu.store("hash-a", []int{1, 2, 3, 4})
	cpu.store("hash-b", []int{5, 6, 7, 8})
	cpu.store("hash-c", []int{9, 10, 11, 12})

	// THEN oldest (hash-a) is evicted, newest two remain
	assert.Nil(t, cpu.lookup("hash-a"), "oldest block should be evicted")
	assert.NotNil(t, cpu.lookup("hash-b"), "second block should survive")
	assert.NotNil(t, cpu.lookup("hash-c"), "newest block should survive")
	assert.Equal(t, int64(2), cpu.used)
	assert.Equal(t, int64(1), cpu.evictionCount)
}

func TestCpuTier_Touch_MovesToTail(t *testing.T) {
	// BC-1 (touch): GIVEN CPU tier with 2 blocks
	cpu := newCpuTier(2, 4)
	cpu.store("hash-a", []int{1, 2, 3, 4})
	cpu.store("hash-b", []int{5, 6, 7, 8})

	// WHEN we touch hash-a (refreshes recency)
	cpu.touch("hash-a")

	// AND store a third block (triggers eviction)
	cpu.store("hash-c", []int{9, 10, 11, 12})

	// THEN hash-b is evicted (it's now oldest), hash-a survives (was touched)
	assert.Nil(t, cpu.lookup("hash-b"), "untouched block should be evicted")
	assert.NotNil(t, cpu.lookup("hash-a"), "touched block should survive")
	assert.NotNil(t, cpu.lookup("hash-c"), "newest block should survive")
}

func TestCpuTier_Lookup_ReturnsNilForMissing(t *testing.T) {
	cpu := newCpuTier(10, 4)
	assert.Nil(t, cpu.lookup("nonexistent"))
}

func TestCpuTier_Store_DuplicateHashIsNoOp(t *testing.T) {
	cpu := newCpuTier(10, 2)
	cpu.store("h1", []int{1, 2})
	cpu.store("h1", []int{99, 99}) // duplicate — should not overwrite

	blk := cpu.lookup("h1")
	assert.NotNil(t, blk)
	assert.Equal(t, []int{1, 2}, blk.tokens, "original tokens preserved")
	assert.Equal(t, int64(1), cpu.used, "no duplicate storage")
}

func TestCpuTier_Touch_NoOpForMissing(t *testing.T) {
	cpu := newCpuTier(10, 2)
	cpu.store("h1", []int{1, 2})
	cpu.touch("nonexistent") // should not panic
	assert.Equal(t, int64(1), cpu.used)
}

func TestCpuTier_EvictHead_EmptyListIsNoOp(t *testing.T) {
	cpu := newCpuTier(10, 2)
	cpu.evictHead() // should not panic on empty list
	assert.Equal(t, int64(0), cpu.used)
}

func TestCpuTier_TokenSlicePoolRecycling(t *testing.T) {
	// Pre-allocated pool should be recycled on eviction
	cpu := newCpuTier(2, 4)
	assert.Equal(t, 2, len(cpu.freeTokenSlices), "pool should start with capacity slices")

	cpu.store("h1", []int{1, 2, 3, 4}) // consumes 1 slice
	assert.Equal(t, 1, len(cpu.freeTokenSlices))

	cpu.store("h2", []int{5, 6, 7, 8}) // consumes last slice
	assert.Equal(t, 0, len(cpu.freeTokenSlices))

	// store("h3") evicts h1 (returns slice to pool), then immediately consumes
	// that returned slice for h3. Net pool effect: 0 → 1 → 0.
	cpu.store("h3", []int{9, 10, 11, 12})
	assert.Equal(t, 0, len(cpu.freeTokenSlices), "evicted slice consumed by new block")

	// Verify content — h3 reused h1's pre-allocated slice via copy
	blk := cpu.lookup("h3")
	assert.NotNil(t, blk)
	assert.Equal(t, []int{9, 10, 11, 12}, blk.tokens)

	// Now evict h2 by storing h4 — h2's slice returns to pool, h4 consumes it
	cpu.store("h4", []int{13, 14, 15, 16})
	assert.Equal(t, 0, len(cpu.freeTokenSlices))

	// Release a block explicitly and verify pool grows
	cpu.evictHead() // evicts h3 (oldest)
	assert.Equal(t, 1, len(cpu.freeTokenSlices), "explicit eviction returns slice to pool")
}

func TestCpuTier_Validation_ZeroCapacity_Panics(t *testing.T) {
	assert.Panics(t, func() { newCpuTier(0, 4) })
}

func TestCpuTier_Validation_NegativeCapacity_Panics(t *testing.T) {
	assert.Panics(t, func() { newCpuTier(-1, 4) })
}

func TestCpuTier_Validation_ZeroBlockSize_Panics(t *testing.T) {
	assert.Panics(t, func() { newCpuTier(10, 0) })
}
