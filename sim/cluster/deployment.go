package cluster

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// DeploymentConfig describes a cluster where all instances share identical
// hardware and model configuration. NumInstances must be >= 1.
type DeploymentConfig struct {
	sim.SimConfig // Embeds all instance-level config (horizon, seed, KV, batch, latency, policy)

	NumInstances int

	// DPLockstepRanks is the data-parallel width of an MoE replica (#1909): each run of this
	// many consecutive instances is one replica whose ranks step in lockstep and share one
	// router endpoint, behind vLLM's internal DP load balancer. 0 or 1 = every instance is an
	// independent engine and its own endpoint. NumInstances and every pool count must be a
	// multiple of it.
	DPLockstepRanks int

	// DPLoadBalancing is how a lockstep replica's requests reach its ranks (#1909), as vLLM
	// v0.31.0's serve flags choose (vllm/engine/arg_utils.py:536-538, :1225-1242;
	// EngineCoreClient.make_async_mp_client, vllm/v1/engine/core_client.py:155-166):
	//   - DPLoadBalancingInternal ("" or "internal", vLLM's default) and
	//     DPLoadBalancingHybrid (--data-parallel-hybrid-lb): one endpoint per replica,
	//     whose front-end balances requests over its ranks (DPLBAsyncMPClient). Hybrid
	//     balances over one node's ranks, which is the whole replica when it fits a node.
	//   - DPLoadBalancingExternal (--data-parallel-external-lb, or
	//     --data-parallel-multi-port-external-lb): every rank serves its own port
	//     (DPAsyncMPClient, one client per rank), so the router sees and picks ranks.
	// Lockstep stepping is the same in every mode.
	DPLoadBalancing string

	// Decision trace configuration (PR13)
	TraceLevel      string // "none" (default), "decisions"
	CounterfactualK int    // routing trace: top-k scored endpoints per decision, 0 = off

	// Phase 1A: Node pool infrastructure (optional — empty = backward-compatible mode).
	// When non-empty, activates PlacementManager for GPU inventory tracking.
	NodePools []NodePoolConfig

	// Phase 1A: Instance lifecycle configuration (loading delay, warm-up, drain policy).
	// Zero value is safe: no loading delay, no warm-up, WAIT drain policy.
	InstanceLifecycle InstanceLifecycleConfig

	// PD disaggregation configuration (PR1)
	// When both PrefillInstances and DecodeInstances are 0, disaggregation is disabled
	// and the pipeline is unchanged (BC-PD-1).
	PrefillInstances int // Number of instances dedicated to prefill (0 = disabled)
	DecodeInstances  int // Number of instances dedicated to decode (0 = disabled)
	// SharedInstances (issue #1276, GAP-5) — number of instances serving both
	// prefill and decode (shared-role pods, llm-d "prefill-decode" / legacy "both"
	// role labels). When > 0, ValidatePoolTopology accepts
	// prefill+decode+shared <= total, and BuildPoolMembershipFromIndices
	// assigns PoolRolePrefillDecode to the last `shared` indices after
	// the prefill-only and decode-only ranges.
	SharedInstances int

	// E/P/D disaggregation configuration (GAP-4, issue #1264).
	// When EncodeInstances == 0 (default), the encode stage is disabled and the
	// pipeline is byte-identical to the pre-PR simulator (BC-EPD-1).
	EncodeInstances int // Number of instances dedicated to encoding multimodal input (0 = disabled)

	// PD KV transfer configuration (PR2).
	//
	// PDTransferTime prices moving one request's KV from a prefill instance to a decode
	// instance, in ticks, for the token capacity of the KV blocks being moved. The latency
	// backend owns the transfer price (blis-latency-kernel's PDTransferTime reads the fabric
	// and the KV geometry) and the simulator owns only when it happens. Required whenever
	// PD disaggregation is enabled; NewClusterSimulator refuses a PD deployment without it.
	PDTransferTime func(tokens int64, from, to InstanceID) int64

	// PDTransferContention makes concurrent handoffs share the links they cross (#1922,
	// INV-P2-2): each handoff's demand (PDTransferDemand) enters a kvtransfer.LinkArbiter, which
	// divides every link max-min fairly among the handoffs on it and re-rates them as handoffs
	// start, drain and finish. A handoff alone still completes at exactly PDTransferTime. False
	// (the default) prices every handoff as if alone, byte-identical to the behaviour before.
	PDTransferContention bool
	// PDTransferDemand is the demand behind PDTransferTime for the same handoff: payload bytes,
	// the floor, and the links crossed with their full rates (blis-schemas
	// kernel.PDTransferDemand). Required when PDTransferContention is set; the simulator invents
	// no rate, floor or link of its own.
	PDTransferDemand func(tokens int64, from, to InstanceID) kvtransfer.Flow

	// PDPrefillKVHold keeps a finished prefill sub-request's KV blocks allocated on its
	// prefill instance until the decode side has pulled them, as a NIXL prefill engine does:
	// request_finished returns delay_free_blocks and records a lease deadline instead of
	// freeing (vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:247-266 at
	// v0.31.0; push_scheduler.py:260-270 likewise). The blocks are freed when the pull
	// completes, at once when the decode side aborts the request before pulling, and at lease
	// expiry when the decode side vanishes (#1934, INV-PD-7). False frees them when the
	// prefill finishes, the behaviour before #1934.
	PDPrefillKVHold bool
	// PDKVLeaseUs is NIXL's kv_lease_duration in microseconds: how long a prefill engine keeps
	// a finished request's blocks for a decode side that neither pulls them nor renews the
	// lease. 0 takes vLLM's default of 30 s (nixl/base_scheduler.py:81-85).
	PDKVLeaseUs int64

	// Per-pool hardware overrides
	// When empty (all nil/zero), all instances use the global SimConfig (BC-P2-1).
	PrefillOverrides PoolOverrides // Hardware overrides for prefill pool instances
	DecodeOverrides  PoolOverrides // Hardware overrides for decode pool instances

	// Phase 1C: Model autoscaler pipeline (issue #692).
	// Zero value is safe: ModelAutoscalerIntervalUs=0 disables the autoscaler entirely (INV-6).
	ModelAutoscalerIntervalUs      float64   `yaml:"model_autoscaler_interval_us,omitempty"`       // tick interval in μs; 0 = autoscaler disabled
	HPAScrapeDelay                 DelaySpec `yaml:"hpa_scrape_delay,omitempty"`                   // HPA scrape lag: time from WVA metric emission to HPA acting; zero = same-tick actuation; Mean/Stddev in seconds
	ScaleUpStabilizationWindowUs   float64   `yaml:"scale_up_stabilization_window_us,omitempty"`   // HPA scale-up stabilization window in μs; 0 = act on first signal (HPA default)
	ScaleDownStabilizationWindowUs float64   `yaml:"scale_down_stabilization_window_us,omitempty"` // HPA scale-down stabilization window in μs; 0 = no stabilization (pass immediately). Set to 300,000,000 (= 5 minutes) to match the Kubernetes HPA default.
	// AutoscalerAnalyzerConfig holds V2SaturationAnalyzer thresholds.
	// Zero values are safe: NewClusterSimulator applies WVA reference defaults
	// (KvCacheThreshold=0.8, ScaleUpThreshold=0.8, ScaleDownBoundary=0.4, AvgInputTokens=512).
	AutoscalerAnalyzerConfig V2SaturationAnalyzerConfig `yaml:"autoscaler_analyzer,omitempty"`
}

// ToSimConfig returns the embedded SimConfig for per-instance construction.
// WorkloadConfig is an empty struct: cluster mode generates workload centrally
// and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return d.SimConfig
}

// resolveConfigForRole returns the SimConfig appropriate for an instance in the given pool role.
// For PoolRolePrefill: applies PrefillOverrides to the global SimConfig.
// For PoolRoleDecode: applies DecodeOverrides to the global SimConfig.
// For PoolRolePrefillDecode (shared-role): applies DecodeOverrides — "decode wins"
// precedence per issue #1276 D-2 (matches the spirit of llm-d's allowsNoLabel=true
// decode default; a shared pod is decode-capable in every deployment, and picking
// decode avoids oversizing TP for pods that also do prefill).
// For any other role (including 0/unset): returns the global SimConfig unchanged.
// The global SimConfig is never mutated.
func (d DeploymentConfig) resolveConfigForRole(role PoolRole) sim.SimConfig {
	switch role {
	case PoolRolePrefillDecode:
		return ResolvePoolConfig(d.SimConfig, d.DecodeOverrides)
	case PoolRolePrefill:
		return ResolvePoolConfig(d.SimConfig, d.PrefillOverrides)
	case PoolRoleDecode:
		return ResolvePoolConfig(d.SimConfig, d.DecodeOverrides)
	case PoolRoleEncode:
		// Encode pool uses the global SimConfig in this PR; per-pool
		// overrides are a follow-up (GAP-4 design doc D6).
		return d.SimConfig
	default:
		return d.SimConfig
	}
}
