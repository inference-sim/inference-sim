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
}
