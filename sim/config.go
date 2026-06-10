package sim

import (
	"fmt"
	"math"
)

// KVCacheConfig groups KV cache parameters for KV store construction.
type KVCacheConfig struct {
	TotalKVBlocks         int64   // GPU tier capacity in blocks (must be > 0)
	BlockSizeTokens       int64   // tokens per block (must be > 0)
	KVCPUBlocks           int64   // CPU tier capacity (0 = single-tier, default)
	KVOffloadThreshold    float64 // DEPRECATED: Ignored in vLLM v1 mirror model. Was: GPU utilization threshold for offload. (CLI default: 0.9, zero-value: 0)
	KVTransferBandwidth   float64 // blocks/tick transfer rate (CLI default: 100.0, zero-value: 0)
	KVTransferBaseLatency int64   // fixed cost per transfer (ticks, default 0)
}

// NewKVCacheConfig creates a KVCacheConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Parameter order matches struct field order.
func NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks int64,
	kvOffloadThreshold, kvTransferBandwidth float64,
	kvTransferBaseLatency int64) KVCacheConfig {
	if totalKVBlocks <= 0 {
		panic(fmt.Sprintf("NewKVCacheConfig: TotalKVBlocks must be > 0, got %d", totalKVBlocks))
	}
	if blockSizeTokens <= 0 {
		panic(fmt.Sprintf("NewKVCacheConfig: BlockSizeTokens must be > 0, got %d", blockSizeTokens))
	}
	if kvCPUBlocks < 0 {
		panic(fmt.Sprintf("NewKVCacheConfig: KVCPUBlocks must be >= 0, got %d", kvCPUBlocks))
	}
	if kvCPUBlocks > 0 {
		// Note: KVOffloadThreshold is NOT validated here — it is deprecated and
		// ignored in the vLLM v1 mirror model. NewKVStore validates it for legacy
		// reasons, but the constructor should not tighten a deprecated contract.
		if kvTransferBandwidth <= 0 || math.IsNaN(kvTransferBandwidth) || math.IsInf(kvTransferBandwidth, 0) {
			panic(fmt.Sprintf("NewKVCacheConfig: KVTransferBandwidth must be finite and > 0 when KVCPUBlocks > 0, got %v", kvTransferBandwidth))
		}
		if kvTransferBaseLatency < 0 {
			panic(fmt.Sprintf("NewKVCacheConfig: KVTransferBaseLatency must be >= 0 when KVCPUBlocks > 0, got %d", kvTransferBaseLatency))
		}
	}
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
// Panics on invalid values: MaxRunningReqs and MaxScheduledTokens must be > 0,
// LongPrefillTokenThreshold must be >= 0 (0 means disabled).
func NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64) BatchConfig {
	if maxRunningReqs <= 0 {
		panic(fmt.Sprintf("NewBatchConfig: MaxRunningReqs must be > 0, got %d", maxRunningReqs))
	}
	if maxScheduledTokens <= 0 {
		panic(fmt.Sprintf("NewBatchConfig: MaxScheduledTokens must be > 0, got %d", maxScheduledTokens))
	}
	if longPrefillTokenThreshold < 0 {
		panic(fmt.Sprintf("NewBatchConfig: LongPrefillTokenThreshold must be >= 0, got %d", longPrefillTokenThreshold))
	}
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
	ModelConfig ModelConfig   // HuggingFace model parameters (for roofline and trained-physics modes)
	HWConfig    HardwareCalib // GPU specifications (for roofline and trained-physics modes)
	Model       string        // model name (e.g., "meta-llama/llama-3.1-8b-instruct")
	GPU         string        // GPU type (e.g., "H100")
	TP          int           // tensor parallelism degree

	// DP is the data-parallel degree (default 1). DP > 1 is only meaningful for
	// MoE models — vLLM treats non-MoE DP as N independent engines, which BLIS
	// already expresses via the router-replica mechanism — and only affects the
	// trained-physics latency backend.
	DP int
	// EnableExpertParallel mirrors vLLM --enable-expert-parallel. When true (and
	// the model is MoE), the flattened TP·DP MoE group is used as the
	// expert-parallel group. Only affects the trained-physics latency backend.
	EnableExpertParallel bool

	Backend     string // latency model backend: "" or "roofline" (default), "trained-physics"
	MaxModelLen int64  // max total sequence length (input + output); 0 = unlimited (mirrors vLLM --max-model-len)
}

// NewModelHardwareConfig creates a ModelHardwareConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Parameter order matches struct field order.
//
// Validation (library boundary → panic): MaxModelLen must be >= 0; DP must be >= 1;
// DP > 1 is rejected for dense models (NumLocalExperts <= 1) because dense data
// parallelism is the router-replica mechanism, not a latency divisor. DP > 1 is
// allowed for MoE models with either EnableExpertParallel setting.
//
// TP is intentionally NOT validated here: TP=0 is a meaningful "invalid config"
// vehicle that the latency-model factory (NewLatencyModel → trained-physics/roofline)
// rejects with an error, and several tests rely on that factory-validation seam.
// Every consumer of the TP-based helpers (EffectiveMoEGroupSize/EffectiveEP) goes
// through NewLatencyModel, so a zero-TP divisor cannot reach latency math.
func NewModelHardwareConfig(modelConfig ModelConfig, hwConfig HardwareCalib,
	model, gpu string, tp, dp int, enableExpertParallel bool,
	backend string, maxModelLen int64) ModelHardwareConfig {
	if maxModelLen < 0 {
		panic(fmt.Sprintf("NewModelHardwareConfig: MaxModelLen must be >= 0, got %d", maxModelLen))
	}
	if dp < 1 {
		panic(fmt.Sprintf("NewModelHardwareConfig: DP must be >= 1, got %d", dp))
	}
	if dp > 1 && modelConfig.NumLocalExperts <= 1 {
		panic(fmt.Sprintf("NewModelHardwareConfig: DP > 1 is only supported for MoE models "+
			"(NumLocalExperts > 1), got DP=%d with NumLocalExperts=%d. Dense data parallelism "+
			"is expressed via router replicas, not the latency model.", dp, modelConfig.NumLocalExperts))
	}
	return ModelHardwareConfig{
		ModelConfig:          modelConfig,
		HWConfig:             hwConfig,
		Model:                model,
		GPU:                  gpu,
		TP:                   tp,
		DP:                   dp,
		EnableExpertParallel: enableExpertParallel,
		Backend:              backend,
		MaxModelLen:          maxModelLen,
	}
}

// isMoE reports whether the model is a mixture-of-experts model. The threshold
// (NumLocalExperts > 1) matches the parsing layer (sim/latency/config.go) and
// ExtractKVCapacityParams: single-expert configs are dense-equivalent.
func (c ModelHardwareConfig) isMoE() bool {
	return c.ModelConfig.NumLocalExperts > 1
}

// EffectiveDP returns the data-parallel degree, clamped to a minimum of 1. The
// canonical constructor already rejects DP < 1, so this clamp only guards a
// zero-valued struct built directly (bypassing NewModelHardwareConfig, which R4
// discourages) — there it treats an unset DP as a single rank.
func (c ModelHardwareConfig) EffectiveDP() int {
	if c.DP < 1 {
		return 1
	}
	return c.DP
}

// EffectiveMoEGroupSize returns the size of the flattened MoE tensor-parallel
// group. For MoE models this is TP·DP (mirroring vLLM's flattened dp·pcp·tp MoE
// group; PCP is not modeled here and is assumed 1), used by both the EP-off and
// EP-on MoE paths. For dense models it is just TP. This is the sharding divisor
// for routed-expert weights/compute.
func (c ModelHardwareConfig) EffectiveMoEGroupSize() int {
	if c.isMoE() {
		return c.TP * c.EffectiveDP()
	}
	return c.TP
}

// EffectiveEP returns the expert-parallel group size: TP·DP when expert
// parallelism is enabled for an MoE model, else 1. This is an EP-mode
// predicate/size only — it is NOT the EP-off sharding divisor (that is
// EffectiveMoEGroupSize).
func (c ModelHardwareConfig) EffectiveEP() int {
	if c.EnableExpertParallel && c.isMoE() {
		return c.TP * c.EffectiveDP()
	}
	return 1
}

// PolicyConfig groups scheduling and preemption policy selection.
type PolicyConfig struct {
	Scheduler        string // "fcfs" (default), "priority-fcfs", "sjf", "reverse-priority"
	PreemptionPolicy string // "fcfs" (default) or "priority"
}

// NewPolicyConfig creates a PolicyConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
func NewPolicyConfig(scheduler, preemptionPolicy string) PolicyConfig {
	return PolicyConfig{
		Scheduler:        scheduler,
		PreemptionPolicy: preemptionPolicy,
	}
}

// WorkloadConfig is retained as an empty struct for SimConfig embedding compatibility.
// All workload generation now happens externally via workload.GenerateRequests().
type WorkloadConfig struct{}

// NewWorkloadConfig creates an empty WorkloadConfig.
func NewWorkloadConfig() WorkloadConfig {
	return WorkloadConfig{}
}
