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
// - *TP must be > 0 when non-nil
// - *MaxModelLen must be > 0 when non-nil
type PoolOverrides struct {
	TP            *int   // tensor parallelism (nil = use global)
	GPU           string // GPU type ("" = use global)
	MaxModelLen   *int64 // max sequence length (nil = use global)
	TotalKVBlocks *int64 // KV blocks (nil = use global; set by the CLI from the kernel)

	// LatencyModel prices this pool's steps (nil = use global). A disaggregated deployment
	// runs a different engine per role -- its own parallelism, token budget and graph mode
	// -- so each role is priced by the kernel for its own pool rather than by one copied to
	// every pool.
	LatencyModel sim.LatencyModel
	// MaxNumSeqs, MaxNumBatchedTokens and PrefixCachingDisabled are the pool's own engine
	// admission settings (nil = use global), for the same reason.
	MaxNumSeqs            *int64
	MaxNumBatchedTokens   *int64
	PrefixCachingDisabled *bool
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
	if o.MaxNumSeqs != nil && *o.MaxNumSeqs <= 0 {
		return fmt.Errorf("%s: PoolOverrides.MaxNumSeqs must be > 0 when set, got %d", name, *o.MaxNumSeqs)
	}
	if o.MaxNumBatchedTokens != nil && *o.MaxNumBatchedTokens <= 0 {
		return fmt.Errorf("%s: PoolOverrides.MaxNumBatchedTokens must be > 0 when set, got %d", name, *o.MaxNumBatchedTokens)
	}
	return nil
}

// IsEmpty returns true when no overrides are set.
func (o PoolOverrides) IsEmpty() bool {
	return o.TP == nil && o.GPU == "" &&
		o.MaxModelLen == nil && o.TotalKVBlocks == nil &&
		o.LatencyModel == nil && o.MaxNumSeqs == nil && o.MaxNumBatchedTokens == nil &&
		o.PrefixCachingDisabled == nil
}

// ResolvePoolConfig applies per-pool overrides to a global SimConfig.
// Returns a new SimConfig with overridden fields; the global config is not mutated.
//
// Struct-copy safety: ModelConfig is a pure value type (safe to copy).
// SLOPriorityOverrides is a map[string]int that shares its backing map across copies.
// Safe because NewSLOPriorityMap only reads the map (for range), never mutates. The
// per-pool LatencyModel is shared by reference: the kernel adapter is read-only once built.
func ResolvePoolConfig(global sim.SimConfig, overrides PoolOverrides) sim.SimConfig {
	resolved := global // struct copy

	if overrides.TP != nil {
		resolved.TP = *overrides.TP
	}
	if overrides.GPU != "" {
		resolved.GPU = overrides.GPU
	}
	if overrides.MaxModelLen != nil {
		resolved.MaxModelLen = *overrides.MaxModelLen
	}
	if overrides.TotalKVBlocks != nil {
		resolved.TotalKVBlocks = *overrides.TotalKVBlocks
	}
	if overrides.LatencyModel != nil {
		resolved.LatencyModelOverride = overrides.LatencyModel
	}
	if overrides.MaxNumSeqs != nil {
		resolved.MaxNumSeqs = *overrides.MaxNumSeqs
	}
	if overrides.MaxNumBatchedTokens != nil {
		resolved.MaxNumBatchedTokens = *overrides.MaxNumBatchedTokens
	}
	if overrides.PrefixCachingDisabled != nil {
		resolved.PrefixCachingDisabled = *overrides.PrefixCachingDisabled
	}

	return resolved
}
