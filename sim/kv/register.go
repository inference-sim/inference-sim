package kv

import "github.com/inference-sim/inference-sim/sim"

func init() {
	sim.NewKVCacheStateFunc = func(totalBlocks, blockSizeTokens int64) sim.KVStore {
		return NewKVCacheState(totalBlocks, blockSizeTokens)
	}
}
