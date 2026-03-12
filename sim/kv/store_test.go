package kv

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
)

func TestNewKVCacheState_ZeroTotalBlocks_Panics(t *testing.T) {
	// BC-8: NewKVCacheState validates TotalKVBlocks > 0
	assert.PanicsWithValue(t,
		"NewKVCacheState: TotalKVBlocks must be > 0, got 0",
		func() {
			NewKVCacheState(0, 16)
		})
}

func TestNewKVCacheState_ZeroBlockSize_Panics(t *testing.T) {
	// BC-8: NewKVCacheState validates BlockSizeTokens > 0
	assert.PanicsWithValue(t,
		"NewKVCacheState: BlockSizeTokens must be > 0, got 0",
		func() {
			NewKVCacheState(100, 0)
		})
}

func TestNewKVCacheState_NegativeTotalBlocks_Panics(t *testing.T) {
	assert.Panics(t, func() {
		NewKVCacheState(-1, 16)
	})
}

func TestNewKVCacheState_ValidConfig_SingleTier_Succeeds(t *testing.T) {
	// BC-8: Valid config produces a working KVStore
	store := NewKVCacheState(100, 16)
	assert.Equal(t, int64(100), store.TotalCapacity())
	assert.Equal(t, int64(0), store.UsedBlocks())
}

func TestNewTieredKVCache_NilGPU_Panics(t *testing.T) {
	assert.PanicsWithValue(t,
		"NewTieredKVCache: gpu must not be nil",
		func() {
			NewTieredKVCache(nil, 10, 0.5, 1.0, 0)
		})
}

func TestNewTieredKVCache_ValidConfig_Succeeds(t *testing.T) {
	gpu := NewKVCacheState(100, 16)
	store := NewTieredKVCache(gpu, 50, 0.8, 1.0, 10)
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
	var store sim.KVStore
	store = NewKVCacheState(100, 16)
	store.SetClock(0) // compiles and runs

	store = NewTieredKVCache(NewKVCacheState(100, 16), 50, 0.8, 1.0, 10)
	store.SetClock(500)
}

func TestKVStore_MirrorToCPU_InterfaceSatisfied(t *testing.T) {
	// BC-8: Both implementations satisfy KVStore interface including MirrorToCPU
	var store sim.KVStore
	store = NewKVCacheState(100, 16)
	store.MirrorToCPU(nil) // no-op, no panic

	store = NewTieredKVCache(NewKVCacheState(100, 16), 50, 0.8, 1.0, 10)
	store.MirrorToCPU(nil) // tiered stub, nil batch is safe
}

// setupTieredWithLatency creates a TieredKVCache and triggers natural
// mirror+reload to accumulate pendingLatency without direct field access.
// Returns the tiered cache with non-zero pendingLatency.
//
// Flow: allocate prefix → mirror to CPU → release → fill GPU (evicts prefix)
// → release some fillers → request same prefix (triggers targeted reload).
func setupTieredWithLatency(t *testing.T) *TieredKVCache {
	t.Helper()
	// 6 GPU blocks, 2 tokens/block. bandwidth=1.0, baseLatency=10
	// => each reload adds 10 + ceil(2/1.0) = 12 ticks.
	gpu := NewKVCacheState(6, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 10)

	// Step 1: Allocate r1 (2 blocks) and mirror to CPU
	r1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	assert.True(t, tiered.AllocateKVBlocks(r1, 0, 4, []int64{}), "r1 alloc")
	tiered.MirrorToCPU([]*sim.Request{r1})

	// Step 2: Release r1 (blocks stay on GPU with hashes — BC-3)
	tiered.ReleaseKVBlocks(r1)

	// Step 3: Fill GPU completely to evict r1's hashes via popFreeBlock
	for i := 0; i < 6; i++ {
		f := &sim.Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 20, i*2 + 21}}
		assert.True(t, tiered.AllocateKVBlocks(f, 0, 2, []int64{}), fmt.Sprintf("f%d alloc", i))
	}
	// GPU: 6 used, 0 free. r1's hashes cleared by popFreeBlock.

	// Step 4: Release 1 filler → 5 used, 1 free.
	tiered.ReleaseKVBlocks(&sim.Request{ID: "f0"})

	// Step 5: Request same prefix [1,2,3,4] → need 2 blocks, 1 free → fails → reload.
	// Reload: 1 block from CPU (limited by maxReloads=1).
	// pendingLatency = 1 × (10 + ceil(2/1.0)) = 12.
	r2 := &sim.Request{ID: "r2", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(r2, 0, 4, []int64{}) // may not fully succeed

	latency := tiered.PendingTransferLatency()
	if latency == 0 {
		t.Fatal("setupTieredWithLatency: mirror+reload path did not accumulate transfer latency")
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
