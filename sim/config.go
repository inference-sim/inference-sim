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

	// MoECommBackend mirrors vLLM VLLM_ALL2ALL_BACKEND. It selects the MoE
	// dispatch/combine communication cost model used by the trained-physics
	// backend when DP > 1. Empty string resolves to the vLLM default
	// (allgather_reducescatter) at the latency-model factory. Only meaningful for
	// MoE models on the trained-physics backend.
	MoECommBackend string

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
	moeCommBackend, backend string, maxModelLen int64) ModelHardwareConfig {
	if maxModelLen < 0 {
		panic(fmt.Sprintf("NewModelHardwareConfig: MaxModelLen must be >= 0, got %d", maxModelLen))
	}
	if dp < 1 {
		panic(fmt.Sprintf("NewModelHardwareConfig: DP must be >= 1, got %d", dp))
	}
	if dp > 1 && !modelConfig.IsMoE() {
		panic(fmt.Sprintf("NewModelHardwareConfig: DP > 1 is only supported for MoE models "+
			"(NumLocalExperts >= %d), got DP=%d with NumLocalExperts=%d. Dense data parallelism "+
			"is expressed via router replicas, not the latency model.", MoEMinExperts, dp, modelConfig.NumLocalExperts))
	}
	return ModelHardwareConfig{
		ModelConfig:          modelConfig,
		HWConfig:             hwConfig,
		Model:                model,
		GPU:                  gpu,
		TP:                   tp,
		DP:                   dp,
		EnableExpertParallel: enableExpertParallel,
		MoECommBackend:       moeCommBackend,
		Backend:              backend,
		MaxModelLen:          maxModelLen,
	}
}

// isMoE reports whether the model is a mixture-of-experts model. It delegates to
// the canonical predicate ModelConfig.IsMoE (threshold MoEMinExperts); see that
// constant for the rationale and the documented vLLM divergence.
func (c ModelHardwareConfig) isMoE() bool {
	return c.ModelConfig.IsMoE()
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

// AdapterSpec declares one LoRA adapter in the pre-declared registry
// (contracts/config-schema.md, data-model.md "Adapter"). Rank is the single source
// of truth for both cold-load latency and HBM footprint. BaseModel is optional: when
// set it names the base model the adapter attaches to and workload references are
// checked against it; when empty, base-model matching is skipped (permissive).
type AdapterSpec struct {
	ID        string `yaml:"id"`
	Rank      int    `yaml:"rank"`
	BaseModel string `yaml:"base_model,omitempty"`
}

// StepOverheadTier holds the per-rank-tier compute-overhead coefficients (DT Eq.1:
// K6·A + K7, fitted per rank). K7 is the per-tier normalization denominator (> 0);
// K6 is the linear coefficient in active-batch adapter count (>= 0). Consumed by the
// per-step compute-overhead factor in a later PR; validated here so the config is
// well-formed at declaration time. Pointer fields (R9): a zero K6 is a meaningful
// user value (no overhead for that tier), distinct from an omitted coefficient.
type StepOverheadTier struct {
	K6 *float64 `yaml:"k6"`
	K7 *float64 `yaml:"k7"`
}

// LoRAConfig is the module-scoped sub-config for the LoRA control-plane subsystem
// (R16; contracts/config-schema.md). It is the 7th SimConfig sub-config. Every field
// is optional and the zero value is inert: with no adapters declared and no capacity
// set, the subsystem is a no-op and output is byte-identical to a pre-feature build
// (INV-6, SC-001).
//
// Pointer types are used where zero is a meaningful user value (R9): AdapterCapacity
// of 0 means "adapters forbidden" (distinct from unset/inert), and the cost
// coefficients distinguish an explicit 0 from an unset default supplied by
// defaults.yaml.
type LoRAConfig struct {
	// AdapterCapacity is the per-instance resident adapter slot count. nil/absent =>
	// subsystem inert (no-op default). A pointer to 0 with adapters declared is a
	// configuration error (adapters forbidden). Consumed by the resident set (PR2).
	AdapterCapacity *int `yaml:"adapter_capacity,omitempty"`

	// Cold-load cost shape (deltas onto the calibrated base). Consumed by PR3.
	LoadBaseLatencyUs    *float64 `yaml:"load_base_latency_us,omitempty"`    // >= 0
	LoadBandwidthBytesUs *float64 `yaml:"load_bandwidth_bytes_us,omitempty"` // > 0 (R11 divisor guard)

	// StepOverheadTiers maps adapter rank -> compute-overhead coefficients. Config-file
	// only (a scalar flag cannot express a per-rank map). Consumed by PR4.
	StepOverheadTiers map[int]StepOverheadTier `yaml:"step_overhead_tiers,omitempty"`

	// FootprintBytesPerRank derives per-adapter HBM footprint (linear first-cut).
	// Consumed by PR5.
	FootprintBytesPerRank *float64 `yaml:"footprint_bytes_per_rank,omitempty"` // > 0

	// Adapters is the pre-declared registry (id -> rank[, base model]). Empty => inert.
	Adapters []AdapterSpec `yaml:"adapters,omitempty"`

	// EvictionPolicy names the resident-adapter eviction policy at the cold-load gate
	// (B-4). Empty => lru (the no-op default, byte-identical to B-3). Resolution and
	// name validation live in sim/lora/eviction.New (cross-package registry); Validate
	// deliberately does NOT check the name here, to keep sim free of the eviction
	// import — the CLI fails fast against sim.ValidEvictionPolicyNames.
	EvictionPolicy string `yaml:"eviction_policy,omitempty"`
}

// HasAdapters reports whether any adapter is declared. When false the subsystem is
// inert regardless of the other fields (INV-6 no-op default).
func (c LoRAConfig) HasAdapters() bool {
	return len(c.Adapters) > 0
}

// Validate checks the LoRAConfig for internal consistency (R3 numeric guards, R11
// divisor guard). It is a pure query — the library boundary returns an error and the
// caller decides fatality (cmd/ -> logrus.Fatalf; sim/ factory -> panic). An empty
// config is valid and inert (INV-6).
//
// Note: cross-validation of workload adapter references against this registry
// (every referenced id must resolve, base model must match) lives in the workload
// layer where the registry meets the client/cohort model — see
// sim/workload.ValidateAdapterReferences.
func (c LoRAConfig) Validate() error {
	// Capacity: a pointer to a non-positive value is only an error when adapters are
	// actually declared (adapters present but zero/negative slots is unservable).
	if c.HasAdapters() && c.AdapterCapacity != nil && *c.AdapterCapacity <= 0 {
		return fmt.Errorf("LoRAConfig: adapter_capacity must be > 0 when adapters are declared, got %d", *c.AdapterCapacity)
	}

	// Adapter registry entries: unique non-empty ids, positive rank (R3).
	seen := make(map[string]struct{}, len(c.Adapters))
	for i, a := range c.Adapters {
		if a.ID == "" {
			return fmt.Errorf("LoRAConfig: adapter[%d] has empty id", i)
		}
		if a.Rank <= 0 {
			return fmt.Errorf("LoRAConfig: adapter %q rank must be > 0, got %d", a.ID, a.Rank)
		}
		if _, dup := seen[a.ID]; dup {
			return fmt.Errorf("LoRAConfig: duplicate adapter id %q", a.ID)
		}
		seen[a.ID] = struct{}{}
	}

	// Cost coefficients (validated whenever set, even absent adapters, so a malformed
	// coefficient is caught at declaration time). Non-finite values (NaN/±Inf) slip
	// past the ordering guards below — NaN comparisons are always false and +Inf
	// passes > 0 — and later poison the cost model's arithmetic, so reject them (R3/R20).
	if c.LoadBaseLatencyUs != nil && (math.IsNaN(*c.LoadBaseLatencyUs) || math.IsInf(*c.LoadBaseLatencyUs, 0) || *c.LoadBaseLatencyUs < 0) {
		return fmt.Errorf("LoRAConfig: load_base_latency_us must be finite and >= 0, got %v", *c.LoadBaseLatencyUs)
	}
	if c.LoadBandwidthBytesUs != nil && (math.IsNaN(*c.LoadBandwidthBytesUs) || math.IsInf(*c.LoadBandwidthBytesUs, 0) || *c.LoadBandwidthBytesUs <= 0) {
		return fmt.Errorf("LoRAConfig: load_bandwidth_bytes_us must be finite and > 0 (divisor guard), got %v", *c.LoadBandwidthBytesUs)
	}
	if c.FootprintBytesPerRank != nil && (math.IsNaN(*c.FootprintBytesPerRank) || math.IsInf(*c.FootprintBytesPerRank, 0) || *c.FootprintBytesPerRank <= 0) {
		return fmt.Errorf("LoRAConfig: footprint_bytes_per_rank must be finite and > 0, got %v", *c.FootprintBytesPerRank)
	}

	// Step-overhead tiers: rank key > 0, K6 >= 0, K7 > 0 (divisor).
	for rank, tier := range c.StepOverheadTiers {
		if rank <= 0 {
			return fmt.Errorf("LoRAConfig: step_overhead_tiers rank key must be > 0, got %d", rank)
		}
		if tier.K6 != nil && (math.IsNaN(*tier.K6) || math.IsInf(*tier.K6, 0) || *tier.K6 < 0) {
			return fmt.Errorf("LoRAConfig: step_overhead_tiers[%d].k6 must be finite and >= 0, got %v", rank, *tier.K6)
		}
		if tier.K7 == nil || math.IsNaN(*tier.K7) || math.IsInf(*tier.K7, 0) || *tier.K7 <= 0 {
			return fmt.Errorf("LoRAConfig: step_overhead_tiers[%d].k7 must be finite and > 0 (divisor guard)", rank)
		}
	}

	// Cost coefficients are REQUIRED once the subsystem is active (adapters declared
	// with a positive capacity): the cold-load gate consumes them via the cost model
	// (#1466), and NewSimulator hard-fails without them. Require them here so the CLI
	// catches the gap at its validation gate with a clear message rather than
	// surfacing a confusing constructor error later. The CLI backfills these from
	// defaults.yaml before validating (resolveLoRAConfig), so a config that reaches
	// here still missing them declared adapters with no cost source at all. Checked
	// last so a malformed-but-present coefficient reports its own specific error first.
	if c.HasAdapters() && c.AdapterCapacity != nil && *c.AdapterCapacity > 0 {
		if c.LoadBaseLatencyUs == nil || c.LoadBandwidthBytesUs == nil || c.FootprintBytesPerRank == nil {
			return fmt.Errorf("LoRAConfig: load_base_latency_us, load_bandwidth_bytes_us, and footprint_bytes_per_rank must all be set when adapters are declared with a positive capacity")
		}
	}

	return nil
}

// WorkloadConfig is retained as an empty struct for SimConfig embedding compatibility.
// All workload generation now happens externally via workload.GenerateRequests().
type WorkloadConfig struct{}

// NewWorkloadConfig creates an empty WorkloadConfig.
func NewWorkloadConfig() WorkloadConfig {
	return WorkloadConfig{}
}
