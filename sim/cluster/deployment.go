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
	RoutingPolicy        string             // "round-robin" (default), "least-loaded", "weighted", "prefix-affinity"
	RoutingScorerConfigs []sim.ScorerConfig // for weighted routing scorer pipeline (nil = use defaults)

	// Decision trace configuration (PR13)
	TraceLevel      string // "none" (default), "decisions"
	CounterfactualK int    // number of counterfactual candidates, default 0

	// Snapshot staleness configuration (H3 experiment)
	SnapshotRefreshInterval int64 // microseconds, 0 = Immediate (default)
}

// ToSimConfig returns the embedded SimConfig for per-instance construction.
// WorkloadConfig is an empty struct: cluster mode generates workload centrally
// and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return d.SimConfig
}
