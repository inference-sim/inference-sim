package sim

// KVStore abstracts KV cache operations for the simulator.
// kv.KVCacheState (single-tier GPU) and kv.TieredKVCache (GPU+CPU) both implement this.
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
// Production callers should import sim/kv and use its constructors directly
// (see cluster.NewInstanceSimulator for the pattern).
// Test code in package sim uses this to avoid importing sim/kv (which would create a cycle).
// Test files in package sim_test use kv_import_test.go (blank import) to trigger registration.
var NewKVCacheStateFunc func(totalBlocks, blockSizeTokens int64) KVStore

// MustNewKVCacheState calls NewKVCacheStateFunc with a nil guard. Panics with an
// actionable message if the factory has not been registered (missing sim/kv import).
func MustNewKVCacheState(totalBlocks, blockSizeTokens int64) KVStore {
	if NewKVCacheStateFunc == nil {
		panic("NewKVCacheStateFunc not registered: import sim/kv to register it " +
			"(add: import _ \"github.com/inference-sim/inference-sim/sim/kv\")")
	}
	return NewKVCacheStateFunc(totalBlocks, blockSizeTokens)
}

// NewKVStoreFromConfig constructs the appropriate KVStore (single-tier or tiered) based on config.
// Registered by sim/kv package's init(). Used by test code in package sim.
var NewKVStoreFromConfig func(cfg KVCacheConfig) KVStore
