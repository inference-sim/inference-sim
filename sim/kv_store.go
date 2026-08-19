package sim

// KVStore abstracts KV cache operations for the simulator.
// kv.KVCacheState (single-tier GPU) and kv.TieredKVCache (GPU+CPU) both implement this.
type KVStore interface {
	AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool
	GetCachedBlocks(tokens []TokenID) []int64
	ReleaseKVBlocks(req *Request)
	BlockSize() int64
	UsedBlocks() int64
	TotalCapacity() int64
	CacheHitRate() float64
	PendingTransferLatency() int64            // Pure query: returns accumulated transfer latency without clearing.
	ConsumePendingTransferLatency() int64     // Read and clear: returns accumulated transfer latency and resets to zero.
	KVThrashingRate() float64
	SetClock(clock int64)            // Synchronize clock for time-dependent operations. No-op for single-tier.
	MirrorToCPU(batch []*Request)    // Copy newly-completed full blocks to CPU tier. No-op for single-tier.
}

// DeferrableKVStore is the optional capability a KVStore implements when it can
// defer a new prefill admission that needs KV blocks resident only on a secondary
// (disk/object-store) offload tier, and re-poll it at each scheduler step instead
// of blocking on the disk or recomputing immediately (H3 kv_deferral, #1591). Only
// kv.OffloadCache implements it; the single-tier and legacy-tiered stores do not,
// so batch formation type-asserts and the whole mechanism is inert (byte-identical,
// INV-6) when offload is off.
//
// The step-quantized re-poll is realized entirely through the existing step loop —
// there is NO new sim.Event. A deferred request stays in the WaitQ (so INV-1
// still_queued counts it and INV-8 keeps a StepEvent scheduled) and is re-examined
// each step until its secondary→CPU promotion lands, then admitted; a bounded fetch
// attempt guarantees it never defers forever (BC-T3).
type DeferrableKVStore interface {
	// PollDeferred advances every tracked deferred request's state machine by one
	// scheduler round (using completions applied by the preceding SetClock) and
	// returns the ids of requests that are STILL deferred this step. Called once per
	// step at the top of batch formation. Cost is O(deferred), not O(waitq).
	PollDeferred(now int64) []string
	// IsDeferred reports whether the request is currently set aside for a
	// secondary-tier fetch (map membership). Batch formation calls it after a failed
	// admission to tell a fresh deferral (skip, keep in WaitQ) apart from GPU
	// pressure (break).
	IsDeferred(id string) bool
	// ClearDeferred forgets a request that left the WaitQ by a non-admit path
	// (timeout, gateway eviction, drain-redirect) so its deferral state and any
	// in-flight promotion bookkeeping do not leak. Idempotent; a no-op for an
	// untracked id.
	ClearDeferred(id string)
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
// Registered by sim/kv package's init(). Used by test code in package sim that cannot
// import sim/kv directly (import cycle). Production code uses kv.NewKVStore() directly.
var NewKVStoreFromConfig func(cfg KVCacheConfig) KVStore

// MustNewKVStoreFromConfig calls NewKVStoreFromConfig with a nil guard. Panics with an
// actionable message if the factory has not been registered (missing sim/kv import).
func MustNewKVStoreFromConfig(cfg KVCacheConfig) KVStore {
	if NewKVStoreFromConfig == nil {
		panic("NewKVStoreFromConfig not registered: import sim/kv to register it " +
			"(add: import _ \"github.com/inference-sim/inference-sim/sim/kv\")")
	}
	return NewKVStoreFromConfig(cfg)
}
