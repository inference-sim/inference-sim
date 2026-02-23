package sim

// KVCacheConfig groups KV cache parameters for NewKVStore.
type KVCacheConfig struct {
	TotalKVBlocks         int64   // GPU tier capacity in blocks (must be > 0)
	BlockSizeTokens       int64   // tokens per block (must be > 0)
	KVCPUBlocks           int64   // CPU tier capacity (0 = single-tier, default)
	KVOffloadThreshold    float64 // GPU utilization threshold for offload (CLI default: 0.9, zero-value: 0)
	KVTransferBandwidth   float64 // blocks/tick transfer rate (CLI default: 100.0, zero-value: 0)
	KVTransferBaseLatency int64   // fixed cost per transfer (ticks, default 0)
}

// NewKVCacheConfig creates a KVCacheConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Parameter order matches struct field order.
func NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks int64,
	kvOffloadThreshold, kvTransferBandwidth float64,
	kvTransferBaseLatency int64) KVCacheConfig {
	return KVCacheConfig{
		TotalKVBlocks:         totalKVBlocks,
		BlockSizeTokens:       blockSizeTokens,
		KVCPUBlocks:           kvCPUBlocks,
		KVOffloadThreshold:    kvOffloadThreshold,
		KVTransferBandwidth:   kvTransferBandwidth,
		KVTransferBaseLatency: kvTransferBaseLatency,
	}
}

// BatchConfig groups batch formation parameters.
type BatchConfig struct {
	MaxRunningReqs            int64 // max requests in RunningBatch
	MaxScheduledTokens        int64 // max total new tokens across all requests in RunningBatch
	LongPrefillTokenThreshold int64 // threshold for long prefill chunking
}

// NewBatchConfig creates a BatchConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64) BatchConfig {
	return BatchConfig{
		MaxRunningReqs:            maxRunningReqs,
		MaxScheduledTokens:        maxScheduledTokens,
		LongPrefillTokenThreshold: longPrefillTokenThreshold,
	}
}

// LatencyCoeffs groups regression coefficients for the latency model.
type LatencyCoeffs struct {
	BetaCoeffs  []float64 // regression coefficients for step time (≥3 elements required)
	AlphaCoeffs []float64 // regression coefficients for queueing time (≥3 elements required)
}

// NewLatencyCoeffs creates a LatencyCoeffs with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewLatencyCoeffs(betaCoeffs, alphaCoeffs []float64) LatencyCoeffs {
	return LatencyCoeffs{
		BetaCoeffs:  betaCoeffs,
		AlphaCoeffs: alphaCoeffs,
	}
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

// NewModelHardwareConfig creates a ModelHardwareConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Parameter order matches struct field order.
func NewModelHardwareConfig(modelConfig ModelConfig, hwConfig HardwareCalib,
	model, gpu string, tp int, roofline bool) ModelHardwareConfig {
	return ModelHardwareConfig{
		ModelConfig: modelConfig,
		HWConfig:    hwConfig,
		Model:       model,
		GPU:         gpu,
		TP:          tp,
		Roofline:    roofline,
	}
}

// PolicyConfig groups scheduling and priority policy selection.
type PolicyConfig struct {
	PriorityPolicy string // "constant" (default) or "slo-based"
	Scheduler      string // "fcfs" (default), "priority-fcfs", "sjf", "reverse-priority"
}

// NewPolicyConfig creates a PolicyConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewPolicyConfig(priorityPolicy, scheduler string) PolicyConfig {
	return PolicyConfig{
		PriorityPolicy: priorityPolicy,
		Scheduler:      scheduler,
	}
}

// WorkloadConfig groups workload generation parameters.
// Both fields zero-valued means no workload generation (caller injects via InjectArrival).
// In cluster mode, these fields are typically zero-valued because workload is passed
// separately to NewClusterSimulator. They are populated only in single-instance mode.
type WorkloadConfig struct {
	GuideLLMConfig         *GuideLLMConfig // distribution-based workload (optional)
	TracesWorkloadFilePath string          // CSV trace file path (optional)
}

// NewWorkloadConfig creates a WorkloadConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewWorkloadConfig(guideLLMConfig *GuideLLMConfig, tracesWorkloadFilePath string) WorkloadConfig {
	return WorkloadConfig{
		GuideLLMConfig:         guideLLMConfig,
		TracesWorkloadFilePath: tracesWorkloadFilePath,
	}
}
