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
			NewKVStore(SimConfig{TotalKVBlocks: 0, BlockSizeTokens: 16})
		})
}

func TestNewKVStore_ZeroBlockSize_Panics(t *testing.T) {
	// BC-8: NewKVStore validates BlockSizeTokens > 0
	assert.PanicsWithValue(t,
		"KVStore: BlockSizeTokens must be > 0, got 0",
		func() {
			NewKVStore(SimConfig{TotalKVBlocks: 100, BlockSizeTokens: 0})
		})
}

func TestNewKVStore_NegativeTotalBlocks_Panics(t *testing.T) {
	assert.Panics(t, func() {
		NewKVStore(SimConfig{TotalKVBlocks: -1, BlockSizeTokens: 16})
	})
}

func TestNewKVStore_ValidConfig_SingleTier_Succeeds(t *testing.T) {
	// BC-8: Valid config produces a working KVStore
	store := NewKVStore(SimConfig{TotalKVBlocks: 100, BlockSizeTokens: 16})
	assert.Equal(t, int64(100), store.TotalCapacity())
	assert.Equal(t, int64(0), store.UsedBlocks())
}

func TestNewKVStore_ValidConfig_Tiered_Succeeds(t *testing.T) {
	store := NewKVStore(SimConfig{
		TotalKVBlocks:         100,
		BlockSizeTokens:       16,
		KVCPUBlocks:           50,
		KVOffloadThreshold:    0.8,
		KVTransferBandwidth:   1.0,
		KVTransferBaseLatency: 10,
	})
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

func TestTieredKVCache_PendingTransferLatency_PureQuery(t *testing.T) {
	// BC-3: PendingTransferLatency is a pure query (no side effects)
	gpu := NewKVCacheState(100, 16)
	tiered := NewTieredKVCache(gpu, 50, 0.5, 1.0, 10)
	// Direct field access: acceptable in same-package unit test because triggering
	// natural latency accumulation (offload→reload) requires complex multi-step setup.
	// The test's purpose is verifying query vs. consume semantics, not accumulation logic.
	tiered.pendingLatency = 42

	first := tiered.PendingTransferLatency()
	second := tiered.PendingTransferLatency()
	assert.Equal(t, int64(42), first)
	assert.Equal(t, int64(42), second, "PendingTransferLatency must be idempotent (BC-3)")
}

func TestTieredKVCache_ConsumePendingTransferLatency_ClearsValue(t *testing.T) {
	// BC-4: ConsumePendingTransferLatency returns value and clears
	gpu := NewKVCacheState(100, 16)
	tiered := NewTieredKVCache(gpu, 50, 0.5, 1.0, 10)
	tiered.pendingLatency = 42 // same-package direct access (see comment in BC-3 test above)

	consumed := tiered.ConsumePendingTransferLatency()
	assert.Equal(t, int64(42), consumed)
	assert.Equal(t, int64(0), tiered.PendingTransferLatency(), "After consume, latency must be 0")
}

func TestKVCacheState_ConsumePendingTransferLatency_AlwaysZero(t *testing.T) {
	kv := NewKVCacheState(100, 16)
	assert.Equal(t, int64(0), kv.ConsumePendingTransferLatency())
	assert.Equal(t, int64(0), kv.ConsumePendingTransferLatency()) // idempotent
}
