package cluster

import "github.com/inference-sim/inference-sim/sim"

// DefaultCacheSignalDelay is the default propagation delay for prefix cache
// signals in microseconds (50ms). Only affects precise-prefix-cache and
// no-hit-lru scorers; has no observable effect on other routing policies.
// Models aggregate signal staleness from production llm-d metrics polling.
// Set to 0 for oracle mode (live cache state).
const DefaultCacheSignalDelay int64 = 50_000

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

	// Cache signal propagation delay for precise prefix cache scoring (issue #919).
	// Only affects routing when precise-prefix-cache or no-hit-lru scorers are active;
	// has no observable effect on other routing policies (round-robin, least-loaded, etc.).
	// When > 0, those scorers query a periodically-refreshed stale snapshot of each
	// instance's KV cache block hash map instead of live state.
	// Models the asynchronous KV event propagation delay in production llm-d.
	// Default: DefaultCacheSignalDelay (50ms). Feeds into ObservabilityConfig.CacheBlocks.
	// 0 = oracle mode (scorers read live cache state with zero delay).
	// Units: microseconds of simulated time.
	CacheSignalDelay int64

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
	PDDecider         string // Disaggregation decider: "" or "never" (default), "always", "prefix-threshold"
	PDPrefixThreshold int    // Non-cached token threshold for prefix-threshold decider (PR6)

	// PD KV transfer configuration (PR2)
	PDTransferBandwidthGBps float64 // Inter-instance KV transfer bandwidth in GB/s (default 25.0)
	PDTransferBaseLatencyMs float64 // Inter-instance KV transfer base latency in ms (default 0.05)
	PDTransferContention    bool    // Enable fair-share bandwidth contention model (--pd-transfer-contention, INV-P2-2)

	// Per-pool routing scorer configuration (PR2)
	// When nil, both pools use the main RoutingScorerConfigs.
	PrefillScorerConfigs []sim.ScorerConfig // Scorer configs for prefill pool routing
	DecodeScorerConfigs  []sim.ScorerConfig // Scorer configs for decode pool routing

	// Per-pool hardware overrides
	// When empty (all nil/zero), all instances use the global SimConfig (BC-P2-1).
	PrefillOverrides PoolOverrides // Hardware overrides for prefill pool instances
	DecodeOverrides  PoolOverrides // Hardware overrides for decode pool instances

	// Phase 1C: Model autoscaler pipeline (issue #692).
	// Zero value is safe: ModelAutoscalerIntervalUs=0 disables the autoscaler entirely (INV-6).
	ModelAutoscalerIntervalUs float64                    `yaml:"model_autoscaler_interval_us,omitempty"` // tick interval in μs; 0 = autoscaler disabled
	HPAScrapeDelay                 DelaySpec                  `yaml:"hpa_scrape_delay,omitempty"`                    // HPA scrape lag: time from WVA metric emission to HPA acting; zero = same-tick actuation; Mean/Stddev in seconds
	ScaleUpStabilizationWindowUs   float64                    `yaml:"scale_up_stabilization_window_us,omitempty"`    // HPA scale-up stabilization window in μs; 0 = act on first signal (HPA default)
	ScaleDownStabilizationWindowUs float64                    `yaml:"scale_down_stabilization_window_us,omitempty"`  // HPA scale-down stabilization window in μs; 0 = no stabilization (pass immediately). Set to 300,000,000 (= 5 minutes) to match the Kubernetes HPA default.
	// AutoscalerAnalyzerConfig holds V2SaturationAnalyzer thresholds.
	// Zero values are safe: NewClusterSimulator applies WVA reference defaults
	// (KvCacheThreshold=0.8, ScaleUpThreshold=0.8, ScaleDownBoundary=0.4, AvgInputTokens=512).
	AutoscalerAnalyzerConfig  V2SaturationAnalyzerConfig `yaml:"autoscaler_analyzer,omitempty"`

	// Phase 1B-1a: tier-ordered admission shedding config (issue #809).
	// TierShedMinPriority=0 rejects sheddable tiers (priority < 0) under overload.
	// Set to 3 (Standard) for Standard-and-above protection, or -3 to admit all.
	TierShedThreshold   int `yaml:"tier_shed_threshold,omitempty"`
	TierShedMinPriority int `yaml:"tier_shed_min_priority,omitempty"`

	// GAIE-legacy admission thresholds (issue #1014). Only used when AdmissionPolicy = "gaie-legacy".
	GAIEQDThreshold float64 // queue depth threshold per instance (default 5)
	GAIEKVThreshold float64 // KV cache utilization threshold (default 0.8)

	// Phase 1B-2a: per-tenant fair-share budgets (issue #811).
	// Key: TenantID string. Value: fraction of total cluster capacity (0.0–1.0).
	// Zero value is safe: nil = no enforcement (all tenants unlimited).
	TenantBudgets map[string]float64 `yaml:"tenant_budgets,omitempty"`

	// Flow control configuration (issue #882, GIE parity).
	// When FlowControlEnabled is false (default), the gateway queue is bypassed
	// and requests flow directly from admission to routing (BC-1 pass-through).
	FlowControlEnabled              bool    `yaml:"flow_control_enabled,omitempty"`
	FlowControlDetector             string  `yaml:"flow_control_detector,omitempty"`               // "never" (default), "utilization", "concurrency"
	FlowControlDispatchOrder        string  `yaml:"flow_control_dispatch_order,omitempty"`          // "fifo" (default), "priority"
	FlowControlMaxQueueDepth        int     `yaml:"flow_control_max_queue_depth,omitempty"`         // 0 = unlimited
	FlowControlQueueDepthThreshold  float64 `yaml:"flow_control_queue_depth_threshold,omitempty"`   // for utilization detector
	FlowControlKVCacheUtilThreshold float64 `yaml:"flow_control_kv_cache_util_threshold,omitempty"` // for utilization detector
	FlowControlMaxConcurrency       int     `yaml:"flow_control_max_concurrency,omitempty"`         // for concurrency detector
	FlowControlPerBandCapacity      int     `yaml:"flow_control_per_band_capacity,omitempty"`       // 0 = unlimited; max requests per priority band
	FlowControlUsageLimitThreshold  float64 `yaml:"flow_control_usage_limit_threshold,omitempty"`   // per-band HoL blocking ceiling (1.0=no HoL, <1.0 gates lower bands earlier)

	// Issue #893: per-GPU-type hardware calibration for roofline and trained-physics backends.
	// Key: GPU type string (e.g., "A100", "H100"). Value: HardwareCalib for that GPU.
	// When non-nil and a pool's gpu_type is found in the map, the matched HardwareCalib
	// overrides simCfg.HWConfig at instance construction time (both sync and deferred paths),
	// ensuring pool-placed instances use the correct roofline hardware coefficients
	// (TFlopsPeak, BwPeakTBs) rather than the CLI --gpu calibration.
	// Zero value (nil) is safe: no override, backward-compatible with all existing callers.
	HWConfigByGPU map[string]sim.HardwareCalib `yaml:"hw_config_by_gpu,omitempty"`
}

// ToSimConfig returns the embedded SimConfig for per-instance construction.
// WorkloadConfig is an empty struct: cluster mode generates workload centrally
// and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return d.SimConfig
}

// EffectivePrefillTP returns the tensor parallelism degree used by the prefill pool.
// Used for KV transfer sizing in both NewClusterSimulator (upfront validation) and
// KVTransferStartedEvent.Execute (runtime). Note: resolveConfigForRole independently
// applies PrefillOverrides via ResolvePoolConfig.
func (d DeploymentConfig) EffectivePrefillTP() int {
	if d.PrefillOverrides.TP != nil {
		return *d.PrefillOverrides.TP
	}
	return d.TP
}

// resolveConfigForRole returns the SimConfig appropriate for an instance in the given pool role.
// For PoolRolePrefill: applies PrefillOverrides to the global SimConfig.
// For PoolRoleDecode: applies DecodeOverrides to the global SimConfig.
// For any other role (including 0/unset): returns the global SimConfig unchanged.
// The global SimConfig is never mutated.
func (d DeploymentConfig) resolveConfigForRole(role PoolRole) sim.SimConfig {
	switch role {
	case PoolRolePrefill:
		return ResolvePoolConfig(d.SimConfig, d.PrefillOverrides)
	case PoolRoleDecode:
		return ResolvePoolConfig(d.SimConfig, d.DecodeOverrides)
	default:
		return d.SimConfig
	}
}
