// register.go wires sim/kv constructors into the sim package's registration
// variables (NewKVCacheStateFunc, NewKVStoreFromConfig). This init() runs when
// any package imports sim/kv, breaking the import cycle between sim/ (interface
// owner) and sim/kv/ (implementation). Production code imports sim/kv directly;
// test code in package sim uses kv_import_test.go for the blank import.
package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

func init() {
	sim.NewKVCacheStateFunc = func(totalBlocks, blockSizeTokens int64) sim.KVStore {
		return NewKVCacheState(totalBlocks, blockSizeTokens)
	}
	sim.NewKVStoreFromConfig = NewKVStore
}

// NewKVStore creates a KVStore from KVCacheConfig. seed is the run seed; it is
// used only to derive the offload device model's jitter RNG partition (#1581) and
// is otherwise ignored (byte-identical behavior for every non-jitter run).
// Returns *KVCacheState for single-tier (KVCPUBlocks <= 0, the default).
// Returns *TieredKVCache for tiered mode (KVCPUBlocks > 0).
func NewKVStore(cfg sim.KVCacheConfig, seed int64) sim.KVStore {
	gpu := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	// #1590 (H1): the multi-tier offload chain (--kv-offload-config) and the legacy
	// single-CPU-tier path (--kv-cpu-blocks) are two distinct offload models; setting
	// both is ambiguous. Refuse loudly rather than silently pick one (R1/R22). The CLI
	// checks this first for a clean message; this is defense-in-depth.
	if cfg.Offload.IsEnabled() {
		if cfg.KVCPUBlocks > 0 {
			panic("NewKVStore: --kv-offload-config and --kv-cpu-blocks are mutually exclusive (distinct offload models); set only one")
		}
		// Dedicated, seed-derived RNG partition for the storage device model's
		// latency jitter (#1581 BC-D5/BC-D6). Built unconditionally when offload is
		// enabled (cheap); consumed only when a tier sets latency_jitter_stddev > 0.
		jitterRNG := sim.NewPartitionedRNG(sim.NewSimulationKey(seed)).ForSubsystem(sim.SubsystemKVOffload)
		return NewOffloadCache(gpu, cfg.Offload, WithOffloadRNG(jitterRNG))
	}
	if cfg.KVCPUBlocks <= 0 {
		return gpu
	}
	// Validate tiered-mode parameters at the KVCacheConfig level (R3).
	// Inf is caught implicitly: +Inf > 1 and -Inf < 0 in IEEE 754.
	if cfg.KVOffloadThreshold < 0 || cfg.KVOffloadThreshold > 1 || math.IsNaN(cfg.KVOffloadThreshold) {
		panic(fmt.Sprintf("NewKVStore: KVOffloadThreshold must be in [0,1] when KVCPUBlocks > 0, got %v", cfg.KVOffloadThreshold))
	}
	if cfg.KVTransferBandwidth <= 0 || math.IsNaN(cfg.KVTransferBandwidth) || math.IsInf(cfg.KVTransferBandwidth, 0) {
		panic(fmt.Sprintf("NewKVStore: KVTransferBandwidth must be finite and > 0 when KVCPUBlocks > 0, got %v", cfg.KVTransferBandwidth))
	}
	return NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold,
		cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
}
