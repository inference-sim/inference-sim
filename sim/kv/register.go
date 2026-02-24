// register.go wires sim/kv constructors into the sim package's registration
// variables (NewKVCacheStateFunc, NewKVStoreFromConfig). This init() runs when
// any package imports sim/kv, breaking the import cycle between sim/ (interface
// owner) and sim/kv/ (implementation). Production code imports sim/kv directly;
// test code in package sim uses kv_import_test.go for the blank import.
package kv

import "github.com/inference-sim/inference-sim/sim"

func init() {
	sim.NewKVCacheStateFunc = func(totalBlocks, blockSizeTokens int64) sim.KVStore {
		return NewKVCacheState(totalBlocks, blockSizeTokens)
	}
	sim.NewKVStoreFromConfig = NewKVStore
}

// NewKVStore creates a KVStore from KVCacheConfig.
// Returns *KVCacheState for single-tier (KVCPUBlocks <= 0, the default).
// Returns *TieredKVCache for tiered mode (KVCPUBlocks > 0).
func NewKVStore(cfg sim.KVCacheConfig) sim.KVStore {
	gpu := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	if cfg.KVCPUBlocks <= 0 {
		return gpu
	}
	return NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold,
		cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
}
