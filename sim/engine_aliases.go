// sim/engine_aliases.go: package sim's names for the engine, which lives in sim/engine (#2021
// E2.3d). Every name a caller of package sim used before the engine moved is kept here, so no
// caller changes: types are aliases, constants are re-declared, and functions and the
// constructor registration variables forward to the engine package.
package sim

import "github.com/inference-sim/blis-vllm/compat/engine"

type (
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
	EvidenceFromKernel       = engine.EvidenceFromKernel
	EvidenceFromProvenance   = engine.EvidenceFromProvenance
	AgenticCancelled         = engine.AgenticCancelled
	AgenticCompleted         = engine.AgenticCompleted
	AgenticContextOverflow   = engine.AgenticContextOverflow
	AgenticError             = engine.AgenticError
	DPSyncInterval           = engine.DPSyncInterval
	MaxSpeculativeTokens     = engine.MaxSpeculativeTokens
	MaxTokenID               = engine.MaxTokenID
	PhaseGrace               = engine.PhaseGrace
	PhaseProfile             = engine.PhaseProfile
	PhaseUnset               = engine.PhaseUnset
	PhaseWarmup              = engine.PhaseWarmup
	PreemptionFCFS           = engine.PreemptionFCFS
	PreemptionPriority       = engine.PreemptionPriority
	StateCompleted           = engine.StateCompleted
	StateQueued              = engine.StateQueued
	StateRunning             = engine.StateRunning
	StateTimedOut            = engine.StateTimedOut
	StateWaitingForRemoteKVs = engine.StateWaitingForRemoteKVs
	SubsystemKVOffload       = engine.SubsystemKVOffload
	SubsystemRouter          = engine.SubsystemRouter
	SubsystemWorkload        = engine.SubsystemWorkload
	SubsystemWorkloadGen     = engine.SubsystemWorkloadGen
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
)
