package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
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

	// MoECommBackend is the MoE all-to-all backend for this pool ("" = use global),
	// mirroring vLLM's VLLM_ALL2ALL_BACKEND being a per-process environment variable
	// (#1548). It is per-pool because the production recipe is per-ROLE: DeepEP
	// high-throughput on the prefill engines (large batched dispatches) and DeepEP
	// low-latency on the decode engines (tiny latency-critical dispatches). A single
	// global mode cannot express that.
	//
	// Validated by the CLI against latency.IsValidMoECommBackend, and again by Validate below
	// for library callers that bypass the CLI — an unrecognized name is rejected, never
	// silently resolved (R1). The trained-physics constructor re-checks it as well, but only
	// on that backend: a pool resolving to roofline would otherwise ignore a bad value.
	MoECommBackend string
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
	// #1548: the CLI validates the per-role backend name before building the overrides, but a
	// library caller constructing PoolOverrides directly does not go through it. The
	// trained-physics constructor re-checks the name, so an invalid value cannot reach the
	// step-time model — but only on that backend: a pool resolving to roofline (a legal
	// combination whenever DP/EP is off) would silently ignore it. Check it here so the
	// failure is loud wherever it originates (R1).
	if o.MoECommBackend != "" && !latency.IsValidMoECommBackend(o.MoECommBackend) {
		return fmt.Errorf("%s: PoolOverrides.MoECommBackend %q is not a recognized vLLM MoE all-to-all "+
			"backend (valid: %v)", name, o.MoECommBackend, latency.ValidMoECommBackends)
	}
	return nil
}

// IsEmpty returns true when no overrides are set.
func (o PoolOverrides) IsEmpty() bool {
	return o.TP == nil && o.GPU == "" && o.LatencyBackend == "" &&
		o.MaxModelLen == nil && o.TotalKVBlocks == nil && o.MoECommBackend == ""
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
// SLOPriorityOverrides is a map[string]int that shares its backing map across copies.
// Safe for the same reason: NewSLOPriorityMap only reads the map (for range), never mutates.
//
// Latency backend constraint: when using per-pool LatencyBackend overrides, all
// backends (roofline, trained-physics) share the same model architecture (HFConfig)
// and LatencyCoeffs. LatencyCoeffs are global and used by trained-physics;
// roofline ignores them.
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
	if overrides.MoECommBackend != "" {
		resolved.MoECommBackend = overrides.MoECommBackend
	}
	if overrides.MaxModelLen != nil {
		resolved.MaxModelLen = *overrides.MaxModelLen
	}
	if overrides.TotalKVBlocks != nil {
		resolved.TotalKVBlocks = *overrides.TotalKVBlocks
	}

	return resolved
}
