package sim

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
	PendingTransferLatency() int64            // Pure query: returns accumulated transfer latency without clearing.
	ConsumePendingTransferLatency() int64     // Read and clear: returns accumulated transfer latency and resets to zero.
	KVThrashingRate() float64
	SetClock(clock int64) // Synchronize clock for time-dependent operations. No-op for single-tier.
}

// NewKVCacheStateFunc is a factory function for creating single-tier KVStore implementations.
// Set by sim/kv package's init() via registration. This breaks the import cycle between
// sim/ (which defines KVStore) and sim/kv/ (which implements it).
//
// Production callers should use kv.NewKVCacheState() directly.
// Test code in package sim uses this to avoid importing sim/kv (which would create a cycle).
var NewKVCacheStateFunc func(totalBlocks, blockSizeTokens int64) KVStore
