package kv

import "github.com/inference-sim/inference-sim/sim"

func init() {
	sim.NewKVCacheStateFunc = func(totalBlocks, blockSizeTokens int64) sim.KVStore {
		return NewKVCacheState(totalBlocks, blockSizeTokens)
	}
	sim.NewKVStoreFromConfig = func(cfg sim.KVCacheConfig) sim.KVStore {
		gpu := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
		if cfg.KVCPUBlocks <= 0 {
			return gpu
		}
		return NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold,
			cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
	}
}
