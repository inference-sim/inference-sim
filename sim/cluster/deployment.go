package cluster

import "github.com/inference-sim/inference-sim/sim"

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

	// Phase 1A: Node pool infrastructure (optional — empty = backward-compatible mode).
	// When non-empty, activates PlacementManager for GPU inventory tracking.
	NodePools []NodePoolConfig

	// Phase 1A: Instance lifecycle configuration (loading delay, warm-up, drain policy).
	// Zero value is safe: no loading delay, no warm-up, WAIT drain policy.
	InstanceLifecycle InstanceLifecycleConfig

	// PD disaggregation configuration (PR1)
	// When both PrefillInstances and DecodeInstances are 0, disaggregation is disabled
	// and the pipeline is unchanged (BC-PD-1).
	PrefillInstances int    // Number of instances dedicated to prefill (0 = disabled)
	DecodeInstances  int    // Number of instances dedicated to decode (0 = disabled)
	PDDecider             string // Disaggregation decider: "" or "never" (default), "always", "prefix-threshold", "direct-to-decode"
	PDPrefixThreshold     int    // Non-cached token threshold for prefix-threshold decider (PR6)
	PDDirectDecodeThreshold int  // Input token threshold for direct-to-decode decider (>= 0, default 256 from CLI)

	// PD KV transfer configuration (PR2)
	PDTransferBandwidthGBps float64 // Inter-instance KV transfer bandwidth in GB/s (default 25.0)
	PDTransferBaseLatencyMs float64 // Inter-instance KV transfer base latency in ms (default 0.05)
	PDKVBytesPerToken       int64   // KV cache bytes per token for transfer duration (default 512)

	// Per-pool routing scorer configuration (PR2)
	// When nil, both pools use the main RoutingScorerConfigs.
	PrefillScorerConfigs []sim.ScorerConfig // Scorer configs for prefill pool routing
	DecodeScorerConfigs  []sim.ScorerConfig // Scorer configs for decode pool routing

	// Phase 1B-1a: tier-ordered admission shedding config (issue #809).
	// Zero value is safe: TierShedMinPriority=0 admits all tiers (same as AlwaysAdmit),
	// but callers should explicitly set 3 (Standard) for meaningful protection.
	TierShedThreshold   int `yaml:"tier_shed_threshold,omitempty"`
	TierShedMinPriority int `yaml:"tier_shed_min_priority,omitempty"`
}

// ToSimConfig returns the embedded SimConfig for per-instance construction.
// WorkloadConfig is an empty struct: cluster mode generates workload centrally
// and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return d.SimConfig
}
