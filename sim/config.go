package sim

// KVCacheConfig groups KV cache parameters for NewKVStore.
type KVCacheConfig struct {
	TotalKVBlocks         int64   // GPU tier capacity in blocks (must be > 0)
	BlockSizeTokens       int64   // tokens per block (must be > 0)
	KVCPUBlocks           int64   // CPU tier capacity (0 = single-tier, default)
	KVOffloadThreshold    float64 // GPU utilization threshold for offload (default 0.9)
	KVTransferBandwidth   float64 // blocks/tick transfer rate (default 100.0)
	KVTransferBaseLatency int64   // fixed cost per transfer (ticks, default 0)
}

// BatchConfig groups batch formation parameters.
type BatchConfig struct {
	MaxRunningReqs            int64 // max requests in RunningBatch
	MaxScheduledTokens        int64 // max total new tokens across all requests in RunningBatch
	LongPrefillTokenThreshold int64 // threshold for long prefill chunking
}

// LatencyCoeffs groups regression coefficients for the latency model.
type LatencyCoeffs struct {
	BetaCoeffs  []float64 // regression coefficients for step time (≥3 elements required)
	AlphaCoeffs []float64 // regression coefficients for queueing time (≥3 elements required)
}

// ModelHardwareConfig groups model identity and hardware specification.
type ModelHardwareConfig struct {
	ModelConfig ModelConfig   // HuggingFace model parameters (for roofline mode)
	HWConfig    HardwareCalib // GPU specifications (for roofline mode)
	Model       string        // model name (e.g., "meta-llama/llama-3.1-8b-instruct")
	GPU         string        // GPU type (e.g., "H100")
	TP          int           // tensor parallelism degree
	Roofline    bool          // true = analytical roofline mode, false = blackbox regression
}

// PolicyConfig groups scheduling and priority policy selection.
type PolicyConfig struct {
	PriorityPolicy string // "constant" (default) or "slo-based"
	Scheduler      string // "fcfs" (default), "priority-fcfs", "sjf", "reverse-priority"
}

// WorkloadConfig groups workload generation parameters.
// Both fields zero-valued means no workload generation (caller injects via InjectArrival).
// In cluster mode, these fields are typically zero-valued because workload is passed
// separately to NewClusterSimulator. They are populated only in single-instance mode.
type WorkloadConfig struct {
	GuideLLMConfig         *GuideLLMConfig // distribution-based workload (optional)
	TracesWorkloadFilePath string          // CSV trace file path (optional)
}
