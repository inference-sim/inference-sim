package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// PoolOverrides holds optional per-pool hardware overrides for PD disaggregation.
// Nil pointer / empty string means "use global config" for that field.
// Pointer types for TP, MaxModelLen, TotalKVBlocks to distinguish "not set" (nil = use
// global) from an explicit value. CLI validates TP > 0 and MaxModelLen > 0 when set;
// TotalKVBlocks may be set by auto-calculation.
//
// Contract for library callers constructing PoolOverrides directly (bypassing CLI):
// - *TP must be > 0 when non-nil (the latency model factory enforces TP > 0; TP=0 will
//   panic at instance construction time for analytical backends)
// - *MaxModelLen must be > 0 when non-nil
type PoolOverrides struct {
	TP             *int   // tensor parallelism (nil = use global)
	GPU            string // GPU type ("" = use global)
	LatencyBackend string // latency model backend ("" = use global)
	MaxModelLen    *int64 // max sequence length (nil = use global)
	TotalKVBlocks  *int64 // KV blocks (nil = use global; set by CLI after auto-calc)
}

// Validate checks that non-nil pointer fields satisfy their constraints (R3).
// name is used in error messages (e.g., "prefill pool" or "decode pool").
// Library callers that construct PoolOverrides directly (bypassing CLI validation)
// should call Validate before passing overrides to DeploymentConfig.
func (o PoolOverrides) Validate(name string) error {
	if o.TP != nil && *o.TP <= 0 {
		return fmt.Errorf("%s: PoolOverrides.TP must be > 0 when set, got %d", name, *o.TP)
	}
	if o.MaxModelLen != nil && *o.MaxModelLen <= 0 {
		return fmt.Errorf("%s: PoolOverrides.MaxModelLen must be > 0 when set, got %d", name, *o.MaxModelLen)
	}
	if o.TotalKVBlocks != nil && *o.TotalKVBlocks <= 0 {
		return fmt.Errorf("%s: PoolOverrides.TotalKVBlocks must be > 0 when set, got %d", name, *o.TotalKVBlocks)
	}
	return nil
}

// IsEmpty returns true when no overrides are set.
func (o PoolOverrides) IsEmpty() bool {
	return o.TP == nil && o.GPU == "" && o.LatencyBackend == "" &&
		o.MaxModelLen == nil && o.TotalKVBlocks == nil
}

// ResolvePoolConfig applies per-pool overrides to a global SimConfig.
// Returns a new SimConfig with overridden fields; the global config is not mutated.
//
// Struct-copy safety: ModelConfig and HardwareCalib are pure value types (safe to copy).
// LatencyCoeffs contains slices (BetaCoeffs/AlphaCoeffs) that share backing arrays
// with the global config after copy. This is safe because: (1) the resolver never
// mutates slice elements, and (2) slices are written once at CLI time and never
// modified during simulation. If future code needs to mutate per-pool coefficients,
// deep-copy the slices here.
//
// Latency backend constraint: when using per-pool LatencyBackend overrides, all
// analytical backends (roofline, crossmodel, trained-roofline) share the same model
// architecture (HFConfig) and LatencyCoeffs. Mixing analytical and blackbox backends
// across pools is supported but note that LatencyCoeffs are global — they are only
// meaningful for the blackbox backend and are ignored by analytical backends.
func ResolvePoolConfig(global sim.SimConfig, overrides PoolOverrides) sim.SimConfig {
	resolved := global // struct copy

	if overrides.TP != nil {
		resolved.TP = *overrides.TP
	}
	if overrides.GPU != "" {
		resolved.GPU = overrides.GPU
	}
	if overrides.LatencyBackend != "" {
		resolved.Backend = overrides.LatencyBackend
	}
	if overrides.MaxModelLen != nil {
		resolved.MaxModelLen = *overrides.MaxModelLen
	}
	if overrides.TotalKVBlocks != nil {
		resolved.TotalKVBlocks = *overrides.TotalKVBlocks
	}

	return resolved
}
