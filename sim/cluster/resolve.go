package cluster

import "github.com/inference-sim/inference-sim/sim"

// PoolOverrides holds optional per-pool hardware overrides for PD disaggregation.
// Nil pointer / empty string means "use global config" for that field.
// Pointer types for TP, MaxModelLen, TotalKVBlocks because zero is a valid
// user value (R9: distinguish "not set" from "set to zero").
type PoolOverrides struct {
	TP             *int   // tensor parallelism (nil = use global)
	GPU            string // GPU type ("" = use global)
	LatencyBackend string // latency model backend ("" = use global)
	MaxModelLen    *int64 // max sequence length (nil = use global)
	TotalKVBlocks  *int64 // KV blocks (nil = use global; set by CLI after auto-calc)
}

// IsEmpty returns true when no overrides are set.
func (o PoolOverrides) IsEmpty() bool {
	return o.TP == nil && o.GPU == "" && o.LatencyBackend == "" &&
		o.MaxModelLen == nil && o.TotalKVBlocks == nil
}

// ResolvePoolConfig applies per-pool overrides to a global SimConfig.
// Returns a new SimConfig with overridden fields; the global config is not mutated.
//
// Safe to struct-copy SimConfig: ModelConfig/HardwareCalib are value types.
// LatencyCoeffs contains slices (BetaCoeffs/AlphaCoeffs) but the resolver never
// mutates them — shared backing arrays are safe (written once at CLI time).
func ResolvePoolConfig(global sim.SimConfig, overrides PoolOverrides) sim.SimConfig {
	resolved := global // struct copy

	if overrides.TP != nil {
		resolved.ModelHardwareConfig.TP = *overrides.TP
	}
	if overrides.GPU != "" {
		resolved.ModelHardwareConfig.GPU = overrides.GPU
	}
	if overrides.LatencyBackend != "" {
		resolved.ModelHardwareConfig.Backend = overrides.LatencyBackend
	}
	if overrides.MaxModelLen != nil {
		resolved.ModelHardwareConfig.MaxModelLen = *overrides.MaxModelLen
	}
	if overrides.TotalKVBlocks != nil {
		resolved.KVCacheConfig.TotalKVBlocks = *overrides.TotalKVBlocks
	}

	return resolved
}
