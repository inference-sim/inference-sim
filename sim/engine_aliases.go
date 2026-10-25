// sim/engine_aliases.go: package sim's names for the engine, which lives in blis-vllm
// (compat/engine and the two leaf packages the host uses: compat/kvtransfer, the P/D link
// arbiter, and compat/prefix, the prompt-chain hash; #2021 E2.3d). It is the host's one
// compatibility seam onto the engine, and it leaves with the 0.29/0.31 compatibility engine
// at the re-baseline (#2027). Every name a caller of package sim used before the engine moved
// is kept here, so no caller changes: types are aliases, constants are re-declared, and
// functions and the constructor registration variables forward to the engine package.
package sim

import (
	"github.com/inference-sim/blis-vllm/compat/dplb"
	"github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kvtransfer"
	"github.com/inference-sim/blis-vllm/compat/nixl"
	hash "github.com/inference-sim/blis-vllm/compat/prefix"
)

type (
	// A data-parallel replica's load balancer, the rule that picks the rank a request reaching
	// the replica's endpoint runs on, and what it reads from each rank.
	DPBalancer   = dplb.Balancer
	DPRank       = dplb.Rank
	DPRankCounts = dplb.Counts
	// The NIXL handoff's rules: how a handoff that ended without a pull releases the prefill's
	// held KV, and what a completed pull does (blis-vllm compat/nixl).
	NIXLRelease      = nixl.Release
	NIXLOutcome      = nixl.Outcome
	NIXLPullComplete = nixl.PullComplete
	// The P/D link arbiter (host-side network contention, INV-P2-2) and its flows.
	LinkArbiter            = kvtransfer.LinkArbiter
	Link                   = kvtransfer.Link
	Flow                   = kvtransfer.Flow
	FlowID                 = kvtransfer.FlowID
	KernelPoolProvenance   = engine.KernelPoolProvenance
	RoutingRecord          = engine.RoutingRecord
	RuleFinding            = engine.RuleFinding
	AdapterCost            = engine.AdapterCost
	AdapterRegistry        = engine.AdapterRegistry
	AdapterSpec            = engine.AdapterSpec
	AgenticTag             = engine.AgenticTag
	Batch                  = engine.Batch
	BatchConfig            = engine.BatchConfig
	BatchContext           = engine.BatchContext
	BatchOption            = engine.BatchOption
	BatchResult            = engine.BatchResult
	Completion             = engine.Completion
	DPStepPricer           = engine.DPStepPricer
	DPWave                 = engine.DPWave
	RemoteKVAdmission      = engine.RemoteKVAdmission
	DeferrableKVStore      = engine.DeferrableKVStore
	EngineCore             = engine.EngineCore
	EngineMetrics          = engine.EngineMetrics
	FCFSScheduler          = engine.FCFSScheduler
	FrontendConfig         = engine.FrontendConfig
	InflightPrefill        = engine.InflightPrefill
	InstanceScheduler      = engine.InstanceScheduler
	KVCacheConfig          = engine.KVCacheConfig
	KVCacheOption          = engine.KVCacheOption
	KVEvent                = engine.KVEvent
	KVOffloadConfig        = engine.KVOffloadConfig
	KVOffloadTier          = engine.KVOffloadTier
	KVPrefixHit            = engine.KVPrefixHit
	KVSequenceState        = engine.KVSequenceState
	KVSpeculation          = engine.KVSpeculation
	KVStore                = engine.KVStore
	LatencyModel           = engine.LatencyModel
	LoRAConfig             = engine.LoRAConfig
	ModelConfig            = engine.ModelConfig
	ModelHardwareConfig    = engine.ModelHardwareConfig
	OffloadTierSharing     = engine.OffloadTierSharing
	PartitionedRNG         = engine.PartitionedRNG
	Phase                  = engine.Phase
	PolicyConfig           = engine.PolicyConfig
	PrefixLookup           = engine.PrefixLookup
	PriorityFCFSScheduler  = engine.PriorityFCFSScheduler
	ReloadReportingKVStore = engine.ReloadReportingKVStore
	RemoteKVReserver       = engine.RemoteKVReserver
	Renderer               = engine.Renderer
	Request                = engine.Request
	RequestBlockReporter   = engine.RequestBlockReporter
	RequestState           = engine.RequestState
	ResidentAdapterSet     = engine.ResidentAdapterSet
	SJFScheduler           = engine.SJFScheduler
	SLOPriorityMap         = engine.SLOPriorityMap
	ServerSeriesOutput     = engine.ServerSeriesOutput
	SimConfig              = engine.SimConfig
	SimulationKey          = engine.SimulationKey
	SkippedPolls           = engine.SkippedPolls
	SpeculativeConfig      = engine.SpeculativeConfig
	StepOverheadTier       = engine.StepOverheadTier
	ThrashCounts           = engine.ThrashCounts
	TokenID                = engine.TokenID
	VLLMBatchFormation     = engine.VLLMBatchFormation
	WaitQueue              = engine.WaitQueue
	WorkloadConfig         = engine.WorkloadConfig
)

const (
	// DPStatsIntervalUs is how often a replica's ranks publish their counts to its balancer.
	DPStatsIntervalUs = dplb.StatsIntervalUs
	// The DP load-balancing modes, as vLLM's serve flags choose them (internal, hybrid, external).
	DPLoadBalancingInternal = dplb.Internal
	DPLoadBalancingHybrid   = dplb.Hybrid
	DPLoadBalancingExternal = dplb.External
	// How a NIXL handoff ended without a pull, and the prefill release each applies.
	NIXLDecodeNeverReceived    = nixl.DecodeNeverReceived
	NIXLDecodeRejected         = nixl.DecodeRejected
	NIXLDecodeVanished         = nixl.DecodeVanished
	NIXLFreeNow                = nixl.FreeNow
	NIXLHoldUntilFirstExpiry   = nixl.HoldUntilFirstExpiry
	NIXLHoldUntilRenewedExpiry = nixl.HoldUntilRenewedExpiry
	EvidenceFromKernel         = engine.EvidenceFromKernel
	EvidenceFromProvenance     = engine.EvidenceFromProvenance
	AgenticCancelled           = engine.AgenticCancelled
	AgenticCompleted           = engine.AgenticCompleted
	AgenticContextOverflow     = engine.AgenticContextOverflow
	AgenticError               = engine.AgenticError
	DPSyncInterval             = engine.DPSyncInterval
	MaxSpeculativeTokens       = engine.MaxSpeculativeTokens
	MaxTokenID                 = engine.MaxTokenID
	PhaseGrace                 = engine.PhaseGrace
	PhaseProfile               = engine.PhaseProfile
	PhaseUnset                 = engine.PhaseUnset
	PhaseWarmup                = engine.PhaseWarmup
	PreemptionFCFS             = engine.PreemptionFCFS
	PreemptionPriority         = engine.PreemptionPriority
	StateCompleted             = engine.StateCompleted
	StateQueued                = engine.StateQueued
	StateRunning               = engine.StateRunning
	StateTimedOut              = engine.StateTimedOut
	StateWaitingForRemoteKVs   = engine.StateWaitingForRemoteKVs
	SubsystemKVOffload         = engine.SubsystemKVOffload
	SubsystemRouter            = engine.SubsystemRouter
	SubsystemWorkload          = engine.SubsystemWorkload
	SubsystemWorkloadGen       = engine.SubsystemWorkloadGen
)

var (
	AdapterRegistryConstructor    = engine.AdapterRegistryConstructor
	BuildAdapterCost              = engine.BuildAdapterCost
	BuildServerSeries             = engine.BuildServerSeries
	DefaultSLOPriorityMap         = engine.DefaultSLOPriorityMap
	GenerateRandomTokenIDs        = engine.GenerateRandomTokenIDs
	HasWorkOf                     = engine.HasWorkOf
	IsValidScheduler              = engine.IsValidScheduler
	KVStoreConstructor            = engine.KVStoreConstructor
	MustNewKVCacheState           = engine.MustNewKVCacheState
	MustNewKVStoreFromConfig      = engine.MustNewKVStoreFromConfig
	NewBatchConfig                = engine.NewBatchConfig
	NewBatchFormation             = engine.NewBatchFormation
	NewEngineCore                 = engine.NewEngineCore
	NewKVCacheConfig              = engine.NewKVCacheConfig
	NewModelHardwareConfig        = engine.NewModelHardwareConfig
	NewPartitionedRNG             = engine.NewPartitionedRNG
	NewPolicyConfig               = engine.NewPolicyConfig
	NewRenderer                   = engine.NewRenderer
	NewSLOPriorityMap             = engine.NewSLOPriorityMap
	NewScheduler                  = engine.NewScheduler
	NewSimulationKey              = engine.NewSimulationKey
	NewSpeculativeConfig          = engine.NewSpeculativeConfig
	NewWorkloadConfig             = engine.NewWorkloadConfig
	PreemptionPolicyNameTable     = engine.PreemptionPolicyNameTable
	SchedulerNameTable            = engine.SchedulerNameTable
	SetNewAdapterCostFunc         = engine.SetNewAdapterCostFunc
	SetNewAdapterRegistryFunc     = engine.SetNewAdapterRegistryFunc
	SetNewKVCacheStateFunc        = engine.SetNewKVCacheStateFunc
	SetNewKVStoreFromConfig       = engine.SetNewKVStoreFromConfig
	SetNewResidentAdapterSetFunc  = engine.SetNewResidentAdapterSetFunc
	ShareRenderer                 = engine.ShareRenderer
	ReplicaAPIServers             = engine.ReplicaAPIServers
	SubsystemInstance             = engine.SubsystemInstance
	WithAdapterOverhead           = engine.WithAdapterOverhead
	WithKVOffload                 = engine.WithKVOffload
	WithKVTransferTicksPerBlock   = engine.WithKVTransferTicksPerBlock
	WithPrefixCachingDisabled     = engine.WithPrefixCachingDisabled
	WithPrefixHit                 = engine.WithPrefixHit
	WithSequenceState             = engine.WithSequenceState
	NewVLLMBatchFormationForTest  = engine.NewVLLMBatchFormationForTest
	NewServerSeriesForTest        = engine.NewServerSeriesForTest
	KVCacheUsageForTest           = engine.KVCacheUsageForTest
	NewKVCacheStateFuncForTest    = engine.NewKVCacheStateFuncForTest
	ResidentAdapterSetConstructor = engine.ResidentAdapterSetConstructor
	// The NIXL handoff's lease on a prefill engine's held KV: the stated or default lease, and
	// when it expires with no decode side, or after the decode side vanished.
	NIXLLease             = nixl.Lease
	NIXLFirstExpiry       = nixl.FirstExpiry
	NIXLExpiryAfterVanish = nixl.ExpiryAfterVanish
	// The NIXL handoff's decisions the cluster asks: whether the decode pool can ever take a
	// prompt, which release applies to how a handoff ended and when it frees, and what a
	// completed pull does.
	NIXLDecodeAccepts  = nixl.DecodeAccepts
	NIXLReleaseOnDrop  = nixl.ReleaseOnDrop
	NIXLReleaseAt      = nixl.ReleaseAt
	NIXLOnPullComplete = nixl.OnPullComplete
	// NewDPBalancer is the balancer of a replica of n ranks.
	NewDPBalancer = dplb.New
	// ValidDPLoadBalancing reports whether a mode names a DP load-balancing mode ("" is internal),
	// and DPEachRankIsAnEndpoint whether, under a mode, every rank serves its own endpoint.
	ValidDPLoadBalancing   = dplb.ValidMode
	DPEachRankIsAnEndpoint = dplb.EachRankIsAnEndpoint
	// The P/D link arbiter's constructor and the alone-on-its-links transfer time.
	NewLinkArbiter = kvtransfer.NewLinkArbiter
	AloneTicks     = kvtransfer.AloneTicks
	// HashBlock chains a prompt block onto its prefix's hash: the router's prompt-chain hash,
	// the same one the engine's prefix cache keys blocks with (BC-3).
	HashBlock = hash.HashBlock
)
