package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewKVStore_ZeroTotalBlocks_Panics(t *testing.T) {
	// BC-8: NewKVStore validates TotalKVBlocks > 0
	assert.PanicsWithValue(t,
		"KVStore: TotalKVBlocks must be > 0, got 0",
		func() {
			NewKVStore(NewKVCacheConfig(0, 16, 0, 0, 0, 0))
		})
}

func TestNewKVStore_ZeroBlockSize_Panics(t *testing.T) {
	// BC-8: NewKVStore validates BlockSizeTokens > 0
	assert.PanicsWithValue(t,
		"KVStore: BlockSizeTokens must be > 0, got 0",
		func() {
			NewKVStore(NewKVCacheConfig(100, 0, 0, 0, 0, 0))
		})
}

func TestNewKVStore_NegativeTotalBlocks_Panics(t *testing.T) {
	assert.Panics(t, func() {
		NewKVStore(NewKVCacheConfig(-1, 16, 0, 0, 0, 0))
	})
}

func TestNewKVStore_ValidConfig_SingleTier_Succeeds(t *testing.T) {
	// BC-8: Valid config produces a working KVStore
	store := NewKVStore(NewKVCacheConfig(100, 16, 0, 0, 0, 0))
	assert.Equal(t, int64(100), store.TotalCapacity())
	assert.Equal(t, int64(0), store.UsedBlocks())
}

func TestNewKVStore_ValidConfig_Tiered_Succeeds(t *testing.T) {
	store := NewKVStore(NewKVCacheConfig(100, 16, 50, 0.8, 1.0, 10))
	assert.Equal(t, int64(100), store.TotalCapacity())
}

func TestKVCacheState_SetClock_IsNoOp(t *testing.T) {
	// BC-5: Single-tier SetClock is a no-op (no observable effect)
	kv := NewKVCacheState(100, 16)
	kv.SetClock(1000)
	assert.Equal(t, int64(100), kv.TotalCapacity())
}

func TestKVStore_SetClock_InterfaceSatisfied(t *testing.T) {
	// BC-5: Both implementations satisfy KVStore interface including SetClock
	var store KVStore
	store = NewKVCacheState(100, 16)
	store.SetClock(0) // compiles and runs

	store = NewTieredKVCache(NewKVCacheState(100, 16), 50, 0.8, 1.0, 10)
	store.SetClock(500)
}

// setupTieredWithLatency creates a TieredKVCache and triggers natural offload+reload
// to accumulate pendingLatency without direct field access.
// Returns the tiered cache with non-zero pendingLatency.
func setupTieredWithLatency(t *testing.T) *TieredKVCache {
	t.Helper()
	// 5 GPU blocks, 2 tokens/block. Threshold 0.3 => offload when util > 30% (>1.5 blocks used).
	// bandwidth=1.0, baseLatency=10 => each reload adds 10 + ceil(2/1.0) = 12 ticks.
	gpu := NewKVCacheState(5, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.3, 1.0, 10)

	// Step 1: Allocate r1 (2 blocks) and r2 (2 blocks) => 4 used, 1 free. util=0.8.
	r1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	assert.True(t, tiered.AllocateKVBlocks(r1, 0, 4, []int64{}), "r1 alloc")
	r2 := &Request{ID: "r2", InputTokens: []int{10, 20, 30, 40}}
	assert.True(t, tiered.AllocateKVBlocks(r2, 0, 4, []int64{}), "r2 alloc")

	// Step 2: Release r1. util drops to 2/5=0.4 > 0.3 => maybeOffload triggers.
	// r1's 2 free blocks (with hashes) get offloaded to CPU.
	tiered.ReleaseKVBlocks(r1)
	// GPU: 2 used (r2), 3 free (all empty). CPU: 2 blocks with r1's hashes.

	// Step 3: Fill GPU free blocks to force next allocation to fail => triggers reload.
	r3 := &Request{ID: "r3", InputTokens: []int{50, 60}}
	assert.True(t, tiered.AllocateKVBlocks(r3, 0, 2, []int64{}), "r3 alloc")
	// GPU: 3 used, 2 free.
	r4 := &Request{ID: "r4", InputTokens: []int{70, 80}}
	assert.True(t, tiered.AllocateKVBlocks(r4, 0, 2, []int64{}), "r4 alloc")
	// GPU: 4 used, 1 free.
	r5 := &Request{ID: "r5", InputTokens: []int{90, 91}}
	assert.True(t, tiered.AllocateKVBlocks(r5, 0, 2, []int64{}), "r5 alloc")
	// GPU: 5 used, 0 free.

	// Step 4: Release r3 to free exactly 1 block. GPU: 4 used, 1 free.
	tiered.ReleaseKVBlocks(r3)
	// util=4/5=0.8 > 0.3 => maybeOffload triggers. But r3's freed block
	// has a hash, so it gets offloaded to CPU. GPU: 4 used, 1 free (empty).

	// Step 5: Try allocating with prefix matching r1 ([1,2,3,4]).
	// GPU has 0 cached blocks for r1's prefix (hashes are on CPU).
	// Needs 2 new blocks but only 1 free => GPU alloc fails.
	// tryReloadFromCPU runs: pops the 1 free GPU block, fills with CPU content.
	// pendingLatency += 12. Then tries second CPU block but no more free GPU blocks.
	// Retry: 1 cached block now, startIndex=2, needs 1 new block. But 0 free blocks remain.
	// Hmm, the reloaded block was re-appended to free list. Let me trace more carefully...
	// Actually after reload: the free block was popped, filled with r1's content, re-appended.
	// So free list still has 1 block (the reloaded one). But it's now hashed/cached.
	// GetCachedBlocks finds 1 match. newStart=2. Need tokens[2:6] = [3,4,5,6] = 2 blocks.
	// Wait, original r1 had [1,2,3,4]. req6 has [1,2,3,4,5,6]. The first CPU block has
	// hash for [1,2]. The second CPU block has hash for [1,2,3,4].
	// After reload of first CPU block: GPU has hash for [1,2]. GetCachedBlocks([1,2,3,4,5,6])
	// finds 1 cached block. newStart = 1*2 = 2. Need to alloc from 2 to 6 = 4 tokens = 2 blocks.
	// Only the reloaded block is free (but it's the cached one with hash [1,2] and InUse=false,
	// RefCount=0, in free list). popFreeBlock would pop it... but that would destroy the
	// cache entry. Hmm, this is getting complicated. Let's just check if latency accumulated.

	req6 := &Request{ID: "r6", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	tiered.AllocateKVBlocks(req6, 0, 6, []int64{}) // may or may not succeed

	latency := tiered.PendingTransferLatency()
	if latency == 0 {
		t.Fatal("setupTieredWithLatency: offload/reload path did not accumulate transfer latency")
	}
	return tiered
}

func TestTieredKVCache_PendingTransferLatency_PureQuery(t *testing.T) {
	// BC-8: PendingTransferLatency is a pure query (no side effects)
	tiered := setupTieredWithLatency(t)

	// WHEN PendingTransferLatency is called multiple times
	first := tiered.PendingTransferLatency()
	second := tiered.PendingTransferLatency()

	// THEN both calls return the same value (pure query, idempotent)
	assert.Equal(t, first, second, "PendingTransferLatency must be idempotent")
	// THEN value is positive (latency was accumulated)
	assert.Greater(t, first, int64(0), "PendingTransferLatency should be > 0")
}

func TestTieredKVCache_ConsumePendingTransferLatency_ClearsValue(t *testing.T) {
	// BC-8: ConsumePendingTransferLatency returns value and clears to 0
	tiered := setupTieredWithLatency(t)

	latencyBefore := tiered.PendingTransferLatency()

	// WHEN ConsumePendingTransferLatency is called
	consumed := tiered.ConsumePendingTransferLatency()

	// THEN it returns the accumulated value
	assert.Equal(t, latencyBefore, consumed, "Consume should return the accumulated value")

	// THEN subsequent query returns 0 (cleared)
	assert.Equal(t, int64(0), tiered.PendingTransferLatency(), "After consume, latency must be 0")

	// THEN second consume also returns 0
	assert.Equal(t, int64(0), tiered.ConsumePendingTransferLatency(), "Second consume must return 0")
}

func TestKVCacheState_ConsumePendingTransferLatency_AlwaysZero(t *testing.T) {
	kv := NewKVCacheState(100, 16)
	assert.Equal(t, int64(0), kv.ConsumePendingTransferLatency())
	assert.Equal(t, int64(0), kv.ConsumePendingTransferLatency()) // idempotent
}
