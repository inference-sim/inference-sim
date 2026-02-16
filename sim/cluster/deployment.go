package cluster

import "github.com/inference-sim/inference-sim/sim"

// DeploymentConfig describes a cluster where all instances share identical
// hardware and model configuration. NumInstances must be >= 1.
type DeploymentConfig struct {
	NumInstances              int
	Horizon                   int64
	Seed                      int64
	TotalKVBlocks             int64
	BlockSizeTokens           int64
	MaxRunningReqs            int64
	MaxScheduledTokens        int64
	LongPrefillTokenThreshold int64
	BetaCoeffs                []float64
	AlphaCoeffs               []float64
	ModelConfig               sim.ModelConfig
	HWConfig                  sim.HardwareCalib
	Model                     string
	GPU                       string
	TP                        int
	Roofline                  bool

	// Online routing pipeline configuration (PR4+)
	AdmissionPolicy       string  // "always-admit" (default) or "token-bucket"
	AdmissionLatency      int64   // microseconds, default 0
	RoutingLatency        int64   // microseconds, default 0
	TokenBucketCapacity   float64 // max tokens, default 10000
	TokenBucketRefillRate float64 // tokens/second, default 1000

	// Routing policy configuration (PR6)
	RoutingPolicy      string  // "round-robin" (default), "least-loaded", "weighted", "prefix-affinity"
	RoutingCacheWeight float64 // for weighted scoring, default 0.6
	RoutingLoadWeight  float64 // for weighted scoring, default 0.4

	// Priority and scheduler configuration (PR7)
	PriorityPolicy string // "constant" (default) or "slo-based"
	Scheduler      string // "fcfs" (default), "priority-fcfs", "sjf"
}

// ToSimConfig converts DeploymentConfig to SimConfig for per-instance construction.
// All instances receive the same config including Seed (identical RNG streams).
// GuideLLMConfig and TracesWorkloadFilePath are intentionally omitted:
// cluster mode generates workload centrally and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return sim.SimConfig{
		Horizon:                   d.Horizon,
		Seed:                      d.Seed,
		TotalKVBlocks:             d.TotalKVBlocks,
		BlockSizeTokens:           d.BlockSizeTokens,
		MaxRunningReqs:            d.MaxRunningReqs,
		MaxScheduledTokens:        d.MaxScheduledTokens,
		LongPrefillTokenThreshold: d.LongPrefillTokenThreshold,
		BetaCoeffs:                d.BetaCoeffs,
		AlphaCoeffs:               d.AlphaCoeffs,
		ModelConfig:               d.ModelConfig,
		HWConfig:                  d.HWConfig,
		Model:                     d.Model,
		GPU:                       d.GPU,
		TP:                        d.TP,
		Roofline:                  d.Roofline,
		PriorityPolicy:            d.PriorityPolicy,
		Scheduler:                 d.Scheduler,
	}
}
