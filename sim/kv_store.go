package sim

import "fmt"

// KVStore abstracts KV cache operations for the simulator.
// KVCacheState (single-tier GPU) and TieredKVCache (GPU+CPU) both implement this.
type KVStore interface {
	AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool
	GetCachedBlocks(tokens []int) []int64
	ReleaseKVBlocks(req *Request)
	BlockSize() int64
	UsedBlocks() int64
	TotalCapacity() int64
	CacheHitRate() float64
	PendingTransferLatency() int64
	KVThrashingRate() float64
	SetClock(clock int64) // Synchronize clock for time-dependent operations. No-op for single-tier.
}

// NewKVStore creates a KVStore from SimConfig.
// Returns *KVCacheState for single-tier (KVCPUBlocks <= 0, the default).
// Returns *TieredKVCache for tiered mode (KVCPUBlocks > 0).
func NewKVStore(cfg SimConfig) KVStore {
	if cfg.TotalKVBlocks <= 0 {
		panic(fmt.Sprintf("KVStore: TotalKVBlocks must be > 0, got %d", cfg.TotalKVBlocks))
	}
	if cfg.BlockSizeTokens <= 0 {
		panic(fmt.Sprintf("KVStore: BlockSizeTokens must be > 0, got %d", cfg.BlockSizeTokens))
	}
	gpu := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	if cfg.KVCPUBlocks <= 0 {
		return gpu
	}
	return NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold,
		cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
}
