package cluster

import "github.com/inference-sim/inference-sim/sim"

// DeploymentConfig describes a cluster deployment with optional per-pool
// hardware overrides for PD disaggregation. NumInstances must be >= 1.
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

	// PD disaggregation configuration (PR1)
	// When both PrefillInstances and DecodeInstances are 0, disaggregation is disabled
	// and the pipeline is unchanged (BC-PD-1).
	PrefillInstances int    // Number of instances dedicated to prefill (0 = disabled)
	DecodeInstances  int    // Number of instances dedicated to decode (0 = disabled)
	PDDecider         string // Disaggregation decider: "" or "never" (default), "always", "prefix-threshold"
	PDPrefixThreshold int    // Non-cached token threshold for prefix-threshold decider (>= 0, default 512 from CLI)

	// PD KV transfer configuration (PR2)
	PDTransferBandwidthGBps float64 // Inter-instance KV transfer bandwidth in GB/s (default 25.0)
	PDTransferBaseLatencyMs float64 // Inter-instance KV transfer base latency in ms (default 0.05)
	PDKVBytesPerToken       int64   // KV cache bytes per token for transfer duration (default 512)

	// Per-pool routing scorer configuration (PR2)
	// When nil, both pools use the main RoutingScorerConfigs.
	PrefillScorerConfigs []sim.ScorerConfig // Scorer configs for prefill pool routing
	DecodeScorerConfigs  []sim.ScorerConfig // Scorer configs for decode pool routing

	// Per-pool hardware overrides (Phase 2, PR1)
	// Zero-valued PoolOverrides = use global config (BC-P2-1).
	PrefillOverrides PoolOverrides // Per-pool overrides for prefill instances
	DecodeOverrides  PoolOverrides // Per-pool overrides for decode instances
}

// ToSimConfig returns the embedded SimConfig for per-instance construction.
// WorkloadConfig is an empty struct: cluster mode generates workload centrally
// and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return d.SimConfig
}

// resolveConfigForRole returns a SimConfig resolved for the given pool role.
// Instances with no pool role (PoolRole zero value) receive the global config.
func (d DeploymentConfig) resolveConfigForRole(role PoolRole) sim.SimConfig {
	global := d.ToSimConfig()
	switch role {
	case PoolRolePrefill:
		return ResolvePoolConfig(global, d.PrefillOverrides)
	case PoolRoleDecode:
		return ResolvePoolConfig(global, d.DecodeOverrides)
	default:
		return global
	}
}
