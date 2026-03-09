package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// DeploymentConfig describes a cluster where all instances share identical
// hardware and model configuration. NumInstances must be >= 1.
type DeploymentConfig struct {
	sim.SimConfig // Embeds all instance-level config (horizon, seed, KV, batch, latency, policy)

	NumInstances int

	// Online routing pipeline configuration (PR4+)
	AdmissionPolicy       string  // "always-admit" (default) or "token-bucket"
	AdmissionLatency      int64   // microseconds, default 0
	RoutingLatency        int64   // microseconds, default 0
	TokenBucketCapacity   float64 // max tokens, default 10000
	TokenBucketRefillRate float64 // tokens/second, default 1000

	// Routing policy configuration (PR6, evolved in PR17)
	RoutingPolicy        string             // "round-robin" (default), "least-loaded", "weighted", "always-busiest"
	RoutingScorerConfigs []sim.ScorerConfig // for weighted routing scorer pipeline (nil = use defaults)

	// Decision trace configuration (PR13)
	TraceLevel      string // "none" (default), "decisions"
	CounterfactualK int    // number of counterfactual candidates, default 0

	// Snapshot staleness configuration (H3 experiment, unified in #463)
	// When > 0, all Prometheus-sourced signals (QueueDepth, BatchSize, KVUtilization)
	// use Periodic refresh with this interval (microseconds). 0 = Immediate (default).
	SnapshotRefreshInterval int64

	// Disaggregated serving configuration
	ServingMode              string // "mixed" (default) or "disaggregated"
	NumPrefillInstances      int    // Prefill pool size (required when disaggregated)
	NumDecodeInstances       int    // Decode pool size (required when disaggregated)
	KVTransferPerToken       int64  // Per-input-token transfer cost in microseconds
	DisagKVTransferBaseLat   int64  // Fixed base KV transfer cost in microseconds
	PrefillMaxScheduledTokens int64 // Override MaxScheduledTokens for prefill pool (0 = use base)
	DecodeMaxRunningReqs     int64  // Override MaxRunningReqs for decode pool (0 = use base)
	DecodeKVBlocks           int64  // Override TotalKVBlocks for decode pool (0 = use base)
}

// ToSimConfig returns the embedded SimConfig for per-instance construction.
// WorkloadConfig is an empty struct: cluster mode generates workload centrally
// and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return d.SimConfig
}

// IsDisaggregated returns true if serving mode is "disaggregated".
func (d DeploymentConfig) IsDisaggregated() bool {
	return d.ServingMode == "disaggregated"
}

// ValidateDisaggregated validates disaggregated-specific config fields.
// Returns nil for "mixed" or empty serving mode.
func (d DeploymentConfig) ValidateDisaggregated() error {
	if d.ServingMode == "" || d.ServingMode == "mixed" {
		return nil
	}
	if d.ServingMode != "disaggregated" {
		return fmt.Errorf("unknown serving mode %q; valid options: mixed, disaggregated", d.ServingMode)
	}
	if d.NumPrefillInstances < 1 {
		return fmt.Errorf("disaggregated mode requires NumPrefillInstances >= 1, got %d", d.NumPrefillInstances)
	}
	if d.NumDecodeInstances < 1 {
		return fmt.Errorf("disaggregated mode requires NumDecodeInstances >= 1, got %d", d.NumDecodeInstances)
	}
	if d.KVTransferPerToken < 0 {
		return fmt.Errorf("KVTransferPerToken must be >= 0, got %d", d.KVTransferPerToken)
	}
	if d.DisagKVTransferBaseLat < 0 {
		return fmt.Errorf("DisagKVTransferBaseLat must be >= 0, got %d", d.DisagKVTransferBaseLat)
	}
	// I1 fix: validate per-pool override values (R3)
	if d.PrefillMaxScheduledTokens < 0 {
		return fmt.Errorf("PrefillMaxScheduledTokens must be >= 0, got %d", d.PrefillMaxScheduledTokens)
	}
	if d.DecodeMaxRunningReqs < 0 {
		return fmt.Errorf("DecodeMaxRunningReqs must be >= 0, got %d", d.DecodeMaxRunningReqs)
	}
	if d.DecodeKVBlocks < 0 {
		return fmt.Errorf("DecodeKVBlocks must be >= 0, got %d", d.DecodeKVBlocks)
	}
	return nil
}

// ToPrefillSimConfig returns a SimConfig for prefill instances,
// applying PrefillMaxScheduledTokens override if set.
func (d DeploymentConfig) ToPrefillSimConfig() sim.SimConfig {
	cfg := d.SimConfig
	if d.PrefillMaxScheduledTokens > 0 {
		cfg.MaxScheduledTokens = d.PrefillMaxScheduledTokens
	}
	return cfg
}

// ToDecodeSimConfig returns a SimConfig for decode instances,
// applying DecodeMaxRunningReqs and DecodeKVBlocks overrides if set.
func (d DeploymentConfig) ToDecodeSimConfig() sim.SimConfig {
	cfg := d.SimConfig
	if d.DecodeMaxRunningReqs > 0 {
		cfg.MaxRunningReqs = d.DecodeMaxRunningReqs
	}
	if d.DecodeKVBlocks > 0 {
		cfg.TotalKVBlocks = d.DecodeKVBlocks
	}
	return cfg
}
