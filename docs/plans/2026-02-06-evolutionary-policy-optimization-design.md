# BLIS Extensions for Evolutionary Policy Optimization

**Date:** 2026-02-06
**Status:** Draft
**Branch:** paperplanning
**Last Updated:** 2026-02-09

## Overview

This design extends BLIS from a single-replica LLM inference simulator to a multi-replica cluster simulator with pluggable control policies. The goal is to enable evolutionary optimization frameworks (OpenEvolve, GEPA) to discover routing, admission, scheduling, and auto-scaling policies through simulation-based fitness evaluation.

### Design Scope

**In scope:**
- Integration layer between BLIS and evolutionary frameworks
- Policy representation (parameterized, code, decision trees)
- BLIS extensions for multi-replica simulation and pluggable policies
- Auto-scaling module for experimentation with scaling policies (scale deployment configurations up/down)
- Workload generation supporting:
  - Multi-turn chat and conversation sessions
  - Different AI workload types (agentic, reasoning, completion, chat)
  - Multiple tenants with distinct behavior profiles
  - Multiple models with different deployment configurations
  - Time-varying load patterns for auto-scaling research
  - Adversarial/misbehaving users (bursty requests, oversized prompts, abuse patterns)
  - SLO classes for scheduling research

**Out of scope (for now):**
- Sim-to-real transfer validation
- Production deployment of evolved policies

### Research Thread Requirements

Different research threads emphasize different workload properties, but all benefit from the full workload model:

| Research Thread | Key Workload Properties |
|-----------------|------------------------|
| Routing | Diverse inference types, prefix patterns, cache-aware decisions |
| Auto-scaling | Time-varying load, multi-model, deployment config scaling |
| Admission | Good/bad actors, burst patterns, abuse scenarios, rate limiting |
| Scheduling | SLO classes, priority differentiation, preemption policies |

### Target Frameworks

**OpenEvolve:**
- LLM-guided evolutionary framework using MAP-Elites + quality-diversity search
- Island-based architecture with ring topology migration
- User provides: initial program, evaluator function (fitness callback), YAML config

**GEPA:**
- Reflective text evolution with Pareto-aware selection
- Requires `GEPAAdapter` interface: `Evaluate()` + `ExtractTracesForReflection()`
- Policies as dictionaries of text components
- Uses execution traces to guide LLM reflection and mutation

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Evolutionary Framework                            │
│              (OpenEvolve / GEPA / custom)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐          │
│  │   Policy    │  │   Fitness   │  │  Trace Collector    │          │
│  │  Candidates │  │  Evaluator  │  │  (for reflection)   │          │
│  └─────────────┘  └─────────────┘  └─────────────────────┘          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Policy + Workload
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   BLIS Cluster Simulator                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Auto-Scaler                                 │ │
│  │   Monitors load → Adjusts replica counts per deployment config  │ │
│  │   (periodic or event-triggered)                                 │ │
│  └────────────────────────────────┬───────────────────────────────┘ │
│                                   │ scale up/down                    │
│  ┌────────────────────────────────▼───────────────────────────────┐ │
│  │            Cluster Control Plane (Router)                       │ │
│  │   Admission → Priority/Scheduling → Routing                     │ │
│  └────────────────────────────────┬───────────────────────────────┘ │
│                                   │                                  │
│         ┌─────────────────────────┼─────────────────────────────┐   │
│         │                         │                             │   │
│         ▼                         ▼                             ▼   │
│  ┌─────────────┐  ┌─────────────────────────────┐  ┌─────────────┐  │
│  │  Model A    │  │         Model B             │  │  Model C    │  │
│  │  Config 1   │  │  ┌─────────┐  ┌─────────┐   │  │  Config 1   │  │
│  │ ┌─────────┐ │  │  │Prefill  │  │ Decode  │   │  │ ┌─────────┐ │  │
│  │ │Replica 0│ │  │  │  Pool   │  │  Pool   │   │  │ │Replica 0│ │  │
│  │ │Replica 1│ │  │  │ ┌─────┐ │  │ ┌─────┐ │   │  │ │Replica 1│ │  │
│  │ └─────────┘ │  │  │ │ P0  │ │  │ │ D0  │ │   │  │ └─────────┘ │  │
│  │ (monolithic)│  │  │ │ P1  │ │  │ │ D1  │ │   │  │ (monolithic)│  │
│  └─────────────┘  │  │ └─────┘ │  │ │ D2  │ │   │  └─────────────┘  │
│                   │  └─────────┘  │ └─────┘ │   │                   │
│                   │  (disaggregated P/D)    │   │                   │
│                   └─────────────────────────┘   │                   │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    Metrics + Traces
```

### Design Principles

- **Separation of concerns**: Cluster control plane, auto-scaler, and instance schedulers are distinct, pluggable components
- **Framework-agnostic**: BLIS exposes an evaluation API that any evolutionary framework can call
- **Policy-as-input**: Policies are loaded at simulation start, not compiled into BLIS
- **Architecture flexibility**: Support both monolithic and prefill-decode disaggregated deployments
- **Expressive interfaces**: Single flexible interfaces that can express filter/score/weight patterns internally
- **Deterministic execution**: Bit-for-bit reproducible simulations for reliable evolutionary optimization

---

## 2. Deployment Model

The deployment model captures the relationship between models, deployment configurations, and instances.

```go
type Model struct {
    ModelID     string
    Parameters  int64              // model size
    Configs     []DeploymentConfig
}

type DeploymentConfig struct {
    ConfigID        string
    ModelID         string
    Architecture    ArchitectureType   // MONOLITHIC | DISAGGREGATED_PD
    HardwareConfig  HardwareConfig

    // For monolithic
    ReplicaPool     *ReplicaPool

    // For disaggregated P/D
    PrefillPool     *ReplicaPool
    DecodePool      *ReplicaPool
}

type ArchitectureType string
const (
    MONOLITHIC        ArchitectureType = "monolithic"
    DISAGGREGATED_PD  ArchitectureType = "disaggregated_pd"
)

type ReplicaPool struct {
    PoolID      string
    PoolType    PoolType           // MONOLITHIC | PREFILL | DECODE
    Instances   []*Instance
    MinReplicas int
    MaxReplicas int
}

type PoolType string
const (
    POOL_MONOLITHIC PoolType = "monolithic"
    POOL_PREFILL    PoolType = "prefill"
    POOL_DECODE     PoolType = "decode"
)
```

### Scaling Unit

For auto-scaling purposes, the unit of scaling is a **deployment configuration**:
- A model can have multiple deployment configurations
- Each configuration can be scaled independently
- Scaling adjusts replica count within the configured min/max bounds

---

## 3. Cluster Control Plane (Router Layer)

The cluster control plane models the llm-d router as a pipeline of three pluggable policy stages.

```
Request Arrival
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AdmissionPolicy                               │
│  Input:  request, router_state, cluster_state                   │
│  Output: ADMIT | REJECT | DELAY(duration)                       │
└─────────────────────────────┬───────────────────────────────────┘
      │ (admitted requests)
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PriorityPolicy                                │
│  Input:  request, router_state, cluster_state                   │
│  Output: priority_score (float), scheduling_hints (map)         │
└─────────────────────────────┬───────────────────────────────────┘
      │ (prioritized requests)
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RoutingPolicy                                 │
│  Input:  request, priority_score, router_state, cluster_state   │
│  Output: RoutingDecision                                        │
│                                                                 │
│  Flexible interface supporting internal implementation of:      │
│  - Filtering (eliminate candidates)                             │
│  - Scoring (rank candidates)                                    │
│  - Flow control (rate limiting, backpressure)                   │
│  - Weight combination                                           │
└─────────────────────────────┴───────────────────────────────────┘
      │
      ▼
  Dispatch to Instance
```

### Router State

The router maintains state computed from request stream and decisions:

```go
type RouterState struct {
    // Per-tenant tracking
    PerTenant       map[TenantID]*TenantState

    // Global metrics
    Global          GlobalRouterMetrics

    // Historical windows
    Historical      HistoricalMetrics

    // KV state views (see below)
    ObservedKV      map[InstanceID]*ObservedKVState   // authoritative metrics
    ShadowKV        map[InstanceID]*ShadowKVModel     // predicted placement
}

type TenantState struct {
    RequestCount    int
    ActiveRequests  int
    RecentRate      float64
    SLOClass        string
    BurstDetected   bool
}

type GlobalRouterMetrics struct {
    InFlight        int
    Throughput      float64
    RequestRate     float64
}
```

### KV State: Observed vs Shadow

The router provides policies with two views of KV cache state:

**ObservedKV** (authoritative): Coarse metrics from engine backends, always accurate.

```go
type ObservedKVState struct {
    Utilization     map[KVTier]float64  // per-tier utilization
    BlocksFree      map[KVTier]int      // available blocks per tier
    CacheHitRate    float64             // recent hit rate
    OffloadRate     float64             // recent offload rate
    ReloadRate      float64             // recent reload rate
}
```

**ShadowKV** (predicted): Router's model of prefix placement based on routing decisions.

```go
type ShadowKVModel struct {
    PrefixHashes        map[Hash]KVLocation  // estimated cached prefixes
    EstimatedUtilization map[KVTier]float64  // predicted per-tier utilization
    EvictionQueue       LRU[Hash]            // predicted eviction order
}

type KVTier string
const (
    KV_GPU     KVTier = "gpu"
    KV_CPU     KVTier = "cpu"
    KV_STORAGE KVTier = "storage"
)

type KVLocation struct {
    Tier        KVTier
    LastAccess  float64
}
```

### Shadow Model Divergence Tracking

The shadow model is updated only from router-visible events:
- Routing decisions (prefix routed to instance)
- Reported evictions (from instance metrics)
- Reported transfers (offload/reload notifications)

**Divergence logging**: When actual cache behavior differs from shadow prediction (e.g., expected hit becomes miss), this is logged as a `shadow_divergence` anomaly. This enables:
- Debugging of prefix-aware routing policies
- Measuring shadow model accuracy
- GEPA reflection on routing decision quality

Policies decide how to use both ObservedKV and ShadowKV. The system does not enforce reconciliation but makes drift measurable.

### Routing Decision

```go
type RoutingDecision struct {
    TargetInstance  InstanceID

    // For disaggregated P/D
    PrefillInstance InstanceID  // optional, for P/D handoff
    DecodeInstance  InstanceID  // optional, for P/D handoff

    // Metadata for tracing
    Reason          string
    Scores          map[string]float64  // internal scoring breakdown
    FilteredOut     []InstanceID        // candidates eliminated

    // For counterfactual analysis
    TopKCandidates  []CandidateScore    // top-k alternatives considered
}

type CandidateScore struct {
    InstanceID      InstanceID
    Score           float64
    ScoreBreakdown  map[string]float64
}
```

### Instance State (Observable by Router)

```go
type InstanceSnapshot struct {
    InstanceID      InstanceID
    PoolType        PoolType
    QueueDepth      int
    BatchSize       int
    KVUtilization   map[KVTier]float64  // per-tier utilization
    InFlightCount   int
    RecentTTFT      float64
    RecentTPOT      float64
    CacheHitRate    float64

    // For routing policies
    AvailableCapacity int
    EstimatedWait     float64
}
```

---

## 4. Auto-Scaler

The auto-scaler adjusts replica counts per deployment configuration based on load signals.

```go
type AutoScaler struct {
    Policy          AutoScalePolicy
    TriggerMode     TriggerMode        // PERIODIC | EVENT_TRIGGERED
    PollInterval    float64            // for periodic mode (e.g., 30s)

    // State
    DeploymentStates map[ConfigID]*DeploymentScaleState
}

type TriggerMode string
const (
    TRIGGER_PERIODIC TriggerMode = "periodic"
    TRIGGER_EVENT    TriggerMode = "event"
)

type AutoScalePolicy interface {
    // Evaluate scaling decision for a deployment config
    Evaluate(ctx AutoScaleContext) ScaleDecision
}

type AutoScaleContext struct {
    ConfigID        string
    CurrentReplicas int
    MinReplicas     int
    MaxReplicas     int

    // Signals
    Metrics         DeploymentMetrics
    LoadHistory     []LoadSample
    PendingRequests int

    // Time
    Clock           float64
    TimeSinceLastScale float64
}

type DeploymentMetrics struct {
    AvgQueueDepth       float64
    AvgUtilization      float64
    RequestRate         float64
    SLOAttainment       map[string]float64
    AvgLatency          float64
    P99Latency          float64
}

type ScaleDecision struct {
    Action          ScaleAction
    TargetReplicas  int
    Reason          string
}

type ScaleAction string
const (
    SCALE_NONE  ScaleAction = "none"
    SCALE_UP    ScaleAction = "up"
    SCALE_DOWN  ScaleAction = "down"
)
```

### Scaling Actuation Model

Scaling is not instantaneous. The auto-scaler models real-world delays:

```go
type ScalingActuationConfig struct {
    // Time to provision new instance (spin up, allocate resources)
    ProvisioningDelay   Distribution

    // Time to load model weights into GPU memory
    ModelLoadTime       Distribution

    // Warmup profile: cold cache penalty over time
    WarmupProfile       WarmupProfile

    // Scale-down behavior
    DrainPolicy         DrainPolicy
}

type WarmupProfile struct {
    // How long until instance reaches steady-state performance
    WarmupDuration      float64

    // Performance penalty curve during warmup
    // e.g., "linear" (starts at PenaltyFactor, decays to 1.0)
    //       "exponential" (faster initial recovery)
    //       "step" (full penalty until WarmupDuration, then normal)
    CurveType           string

    // Initial performance penalty (1.5 = 50% slower during cold start)
    InitialPenaltyFactor float64

    // Cold cache effects
    InitialCacheHitRate float64  // starts at 0, builds up over time
}

type DrainPolicy struct {
    // How to handle in-flight requests on scale-down
    Mode                DrainMode

    // For DRAIN_WAIT: max time to wait for requests to complete
    DrainTimeout        float64
}

type DrainMode string
const (
    DRAIN_IMMEDIATE DrainMode = "immediate"  // kill in-flight (loses work)
    DRAIN_WAIT      DrainMode = "wait"       // wait for completion
    DRAIN_REDIRECT  DrainMode = "redirect"   // migrate to other instances
)
```

### Event Triggers

For event-triggered mode, the auto-scaler responds to:
- Queue depth threshold exceeded
- SLO attainment dropping below target
- Sustained high/low utilization
- Request rate changes

---

## 5. Instance-Level Scheduler Interface

Each instance runs a BLIS simulator with a pluggable scheduler controlling batch formation, preemption, and request ordering.

```go
type InstanceScheduler interface {
    // Called each step to form the next batch
    MakeBatch(ctx SchedulerContext) BatchDecision

    // Called when KV pressure requires preemption
    SelectPreemptionVictim(ctx SchedulerContext) *Request

    // Called on new request arrival from router
    OnRequestArrival(req *Request, ctx SchedulerContext)
}

type SchedulerContext struct {
    // Current state
    WaitQueue       []*Request
    RunningBatch    []*Request
    KVCacheState    KVCacheSnapshot
    Clock           float64

    // Instance info
    InstanceType    PoolType  // monolithic, prefill, or decode

    // Constraints
    MaxBatchSize    int
    MaxTokenBudget  int

    // Historical (for learned policies)
    RecentMetrics   MetricsWindow
}

type BatchDecision struct {
    RequestsToSchedule  []*Request
    PrefillChunkSizes   map[RequestID]int  // optional chunking hints
}
```

### Default Implementations

- `FCFSScheduler` - Current BLIS behavior (baseline)
- `PriorityScheduler` - Respects priority scores from router
- `SJFScheduler` - Shortest job first (by expected output length)

---

## 6. KV Cache Model

Tiered KV cache model supporting offloading and disaggregated architectures.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Instance KV Cache Model                       │
│                                                                  │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│  │     GPU     │ ←──→ │     CPU     │ ←──→ │   Storage   │      │
│  │   Blocks    │      │   Blocks    │      │   Blocks    │      │
│  │             │      │             │      │             │      │
│  │ capacity: N │      │ capacity: M │      │ capacity: K │      │
│  │ latency: L0 │      │ latency: L1 │      │ latency: L2 │      │
│  └─────────────┘      └─────────────┘      └─────────────┘      │
│         │                                                        │
│         │ (disaggregated P/D only)                              │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Inter-Instance KV Transfer                      ││
│  │         Prefill Instance ───RDMA───→ Decode Instance         ││
│  │                    (transfer latency: L_transfer)            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

```go
type KVCacheConfig struct {
    Tiers           []KVTierConfig
    EvictionPolicy  string             // "lru", "lfu", "priority"
    OffloadPolicy   string             // "eager", "lazy", "threshold"

    // For disaggregated P/D
    PDTransferConfig *PDTransferConfig
}

type KVTierConfig struct {
    Tier            KVTier
    Capacity        int                // blocks
    AccessLatency   float64            // ms
    TransferLatency map[KVTier]float64 // ms to transfer to/from other tiers
    Bandwidth       float64            // blocks per second (for transfers)
}

type PDTransferConfig struct {
    Enabled             bool
    TransferLatency     float64        // base latency ms for P→D transfer
    Bandwidth           float64        // blocks per second
    TransferGranularity TransferGranularity
    PipelineMode        bool           // can decode start before full transfer?
}

type TransferGranularity string
const (
    TRANSFER_BLOCK   TransferGranularity = "block"   // transfer whole blocks
    TRANSFER_CHUNK   TransferGranularity = "chunk"   // transfer in chunks
    TRANSFER_STREAM  TransferGranularity = "stream"  // streaming transfer
)

type KVCacheState struct {
    // Per-tier state
    TierState       map[KVTier]*TierState

    // Block tracking
    Blocks          map[BlockID]*KVBlock

    // Transfer queue (for offloading and P/D transfers)
    PendingTransfers []KVTransfer
}

type TierState struct {
    Capacity        int
    Used            int
    Utilization     float64
}

type KVBlock struct {
    BlockID         BlockID
    RequestID       RequestID
    PrefixHash      Hash
    Tier            KVTier
    LastAccess      float64
    Size            int

    // Ownership tracking (for P/D)
    OwnerInstance   InstanceID
    TransferState   BlockTransferState
}

type BlockTransferState string
const (
    BLOCK_LOCAL       BlockTransferState = "local"       // owned by this instance
    BLOCK_TRANSFERRING BlockTransferState = "transferring" // in transit
    BLOCK_TRANSFERRED BlockTransferState = "transferred"  // ownership moved
)

type KVTransfer struct {
    BlockID         BlockID
    FromTier        KVTier
    ToTier          KVTier
    FromInstance    InstanceID         // for inter-instance
    ToInstance      InstanceID         // for inter-instance
    StartTime       float64
    EstCompletion   float64
    TransferType    TransferType
}

type TransferType string
const (
    TRANSFER_OFFLOAD  TransferType = "offload"   // GPU → CPU/Storage
    TRANSFER_RELOAD   TransferType = "reload"    // CPU/Storage → GPU
    TRANSFER_PD       TransferType = "pd"        // Prefill → Decode instance
)
```

### Cache Hit Semantics

A "cache hit" has different latency implications depending on tier:
- **GPU-resident**: Fast (no transfer needed)
- **CPU-resident**: Medium (GPU←CPU transfer latency)
- **Storage-resident**: Slow (GPU←Storage transfer latency)

### P/D KV Ownership Contract

In disaggregated prefill-decode architecture:

1. **Single ownership model**: KV blocks are transferred, not duplicated. After transfer completes, the prefill instance no longer owns the blocks.

2. **Transfer semantics**:
   - Prefill instance produces KV blocks
   - Blocks transfer to decode instance (RDMA-style)
   - Decode instance takes ownership upon transfer completion
   - Prefill instance marks blocks as `BLOCK_TRANSFERRED`

3. **Pipelining** (optional, controlled by `PipelineMode`):
   - If enabled: decode can begin when sufficient blocks have arrived (streaming)
   - If disabled: decode waits for complete transfer

4. **Backpressure**: If decode instance's transfer queue grows beyond threshold, prefill instance may be throttled.

---

## 7. Workload Generation

Extended workload system supporting multi-tenant, multi-model, priority-aware, prefix-structured, and bursty traffic.

```go
type WorkloadSpec struct {
    Models        []ModelWorkloadSpec
    Tenants       []TenantSpec
    GlobalArrival ArrivalPattern
    Duration      float64

    // Time-varying properties (for auto-scaling research)
    LoadProfile   *LoadProfile

    // Correlation structure (optional)
    Correlations  *CorrelationConfig
}

type ModelWorkloadSpec struct {
    ModelID         string
    Weight          float64            // fraction of total traffic
    RequestTypes    []RequestTypeSpec
}

type RequestTypeSpec struct {
    Type            RequestType
    Weight          float64
    TokenDist       TokenDistribution

    // Type-specific properties
    MultiTurnConfig *MultiTurnConfig   // for chat
    AgenticConfig   *AgenticConfig     // for agentic
    ReasoningConfig *ReasoningConfig   // for reasoning
}

type RequestType string
const (
    REQUEST_COMPLETION RequestType = "completion"
    REQUEST_CHAT       RequestType = "chat"
    REQUEST_AGENTIC    RequestType = "agentic"
    REQUEST_REASONING  RequestType = "reasoning"
)

type MultiTurnConfig struct {
    AvgTurns        int
    TurnDistribution Distribution
    ContextGrowth   string             // "linear", "sublinear", "bounded"
}

type AgenticConfig struct {
    ToolCallProb    float64
    AvgToolCalls    int
    ToolLatency     Distribution       // external tool response time
}

type ReasoningConfig struct {
    ChainLength     Distribution       // reasoning steps
    TokensPerStep   Distribution
}

type TenantSpec struct {
    TenantID        string
    Weight          float64            // fraction of total traffic
    PriorityClass   string             // e.g., "realtime", "batch", "best-effort"
    SLO             SLOSpec
    BehaviorProfile BehaviorProfile
    TokenDist       TokenDistribution
    PrefixSpec      PrefixSpec
    ArrivalPattern  ArrivalPattern     // tenant-specific modulation

    // Tenant archetype (for correlation)
    Archetype       *TenantArchetype
}

type TenantArchetype struct {
    // Stable latent parameters that persist across requests
    TypicalPromptSize   string         // "small", "medium", "large", "elephant"
    PrefixReuseAffinity float64        // how likely to reuse same prefixes
    BurstPropensity     float64        // tendency toward bursty behavior
    SessionLength       Distribution   // typical session duration
}

type BehaviorProfile struct {
    Type            BehaviorType

    // For misbehaving tenants
    BurstConfig     *BurstConfig
    AbuseConfig     *AbuseConfig
}

type BehaviorType string
const (
    BEHAVIOR_NORMAL      BehaviorType = "normal"
    BEHAVIOR_BURSTY      BehaviorType = "bursty"
    BEHAVIOR_ABUSIVE     BehaviorType = "abusive"
    BEHAVIOR_ADVERSARIAL BehaviorType = "adversarial"
)

type BurstConfig struct {
    BurstProb       float64
    BurstSize       Distribution       // requests per burst
    BurstInterval   Distribution       // time between bursts
}

type AbuseConfig struct {
    OversizedPrompts    bool           // very long prompts
    RapidFire           bool           // many requests in quick succession
    ResourceExhaustion  bool           // attempts to exhaust KV cache
}

type SLOSpec struct {
    TTFTTarget    float64              // ms
    TPOTTarget    float64              // ms per token
    E2ETarget     float64              // ms, optional
}

type PrefixSpec struct {
    PrefixGroups     []PrefixGroup
    PrefixLength     Distribution
    ReuseProb        float64
}

type PrefixGroup struct {
    GroupID      string
    SystemPrompt string               // actual prefix content (or hash)
    Popularity   float64
}

type ArrivalPattern struct {
    Type    string                    // "poisson", "bursty", "periodic", "trace", "adversarial"
    Params  map[string]any
}

type LoadProfile struct {
    Type        string                // "constant", "step", "ramp", "diurnal", "spike"
    Segments    []LoadSegment
}

type LoadSegment struct {
    StartTime   float64
    EndTime     float64
    Multiplier  float64               // relative to base rate
}
```

### Correlation Structure (Optional)

```go
type CorrelationConfig struct {
    // Prefix reuse correlation within tenant
    IntraTenantPrefixCorrelation float64  // 0.0 = independent, 1.0 = always same prefix

    // Burst-size correlation (bursty tenants send larger prompts during bursts)
    BurstSizeCorrelation         float64

    // Session-level Markov evolution
    SessionModel                 *SessionMarkovModel

    // Closed-loop mode (optional): arrival rate depends on response time
    ClosedLoop                   *ClosedLoopConfig
}

type SessionMarkovModel struct {
    // Prompt/decode length evolves within session
    PromptGrowthRate    float64       // per-turn growth factor
    DecodeVariation     float64       // variance in decode length
}

type ClosedLoopConfig struct {
    Enabled             bool
    ConcurrencyPerTenant int          // fixed concurrent requests per tenant
    ThinkTime           Distribution  // time between response and next request
}
```

### Arrival Pattern Types

| Pattern | Description |
|---------|-------------|
| `poisson` | Standard memoryless arrivals |
| `bursty` | Pareto-distributed inter-arrival times |
| `periodic` | Regular intervals with jitter |
| `trace` | Replay from recorded timestamps |
| `adversarial` | Stress-test patterns (synchronized bursts, long prompts) |

### Load Profile Types (for Auto-scaling Research)

| Profile | Description |
|---------|-------------|
| `constant` | Steady load throughout |
| `step` | Discrete load level changes |
| `ramp` | Gradual increase/decrease |
| `diurnal` | Day/night pattern |
| `spike` | Sudden bursts with recovery |

---

## 8. Policy Representation

Three policy representation types, all loaded dynamically at simulation start.

```go
type PolicyBundle struct {
    AdmissionPolicy   PolicySpec
    PriorityPolicy    PolicySpec
    RoutingPolicy     PolicySpec
    InstanceScheduler PolicySpec
    AutoScalePolicy   PolicySpec

    // Metadata
    Generation        int
    ParentID          string
    Mutations         []string
}

type PolicySpec struct {
    Type       string    // "parameterized", "code", "decision_tree"
    Definition any
}
```

### Type 1: Parameterized Functions

Fixed structure with evolvable numeric parameters.

```go
type ParameterizedPolicy struct {
    Template   string
    Parameters map[string]float64
}

// Example:
// Template: "weighted_affinity_routing"
// Parameters: {
//   "cache_affinity_weight": 0.7,
//   "load_balance_weight": 0.2,
//   "queue_depth_weight": 0.1,
//   "utilization_threshold": 0.85
// }
```

### Type 2: Evolved Code

LLM-generated code implementing the policy interface.

```go
type CodePolicy struct {
    Source     string
    Entrypoint string
    Language   string    // "starlark", "wasm"
    Imports    []string  // restricted allowlist
}
```

**Sandbox Requirements**: Evolved code policies must execute in a sandboxed environment to ensure safety during large-scale evolutionary search.

| Requirement | Implementation |
|-------------|----------------|
| No infinite loops | Step/instruction budget per policy call |
| No memory blowup | Memory limit per execution |
| No external I/O | Sandbox forbids network, filesystem, syscalls |
| No time access | Policies cannot call time functions (breaks determinism) |
| No hidden state | Policies cannot access simulator internals |
| Deterministic | Same inputs must produce same outputs |

**Recommended approaches:**
- **Starlark** (Python-like, used by Bazel): Safe, deterministic, embeddable
- **WASM** (via TinyGo or AssemblyScript): Fast, sandboxed, portable

The sandbox enforces:
```go
type SandboxConfig struct {
    MaxInstructions     int64         // step budget per call
    MaxMemoryBytes      int64         // memory limit
    MaxExecutionTime    float64       // wall-clock timeout (fallback)
    AllowedImports      []string      // restricted module allowlist
}
```

### Type 3: Decision Trees

Interpretable symbolic rules.

```go
type DecisionTreePolicy struct {
    Root *TreeNode
}

type TreeNode struct {
    Condition  Predicate
    TrueChild  *TreeNode
    FalseChild *TreeNode
    Action     any
}
```

---

## 9. Fitness Evaluation

Flexible fitness system supporting single-objective, multi-objective, and SLO-based evaluation.

```go
type EvaluationResult struct {
    Metrics     RawMetrics
    Fitness     map[string]float64
    ParetoRank  int
    PolicyID    string
    Workload    string
    SimDuration float64
}

type RawMetrics struct {
    // Latency distributions (per priority class)
    TTFT        map[string]Distribution
    TPOT        map[string]Distribution
    E2E         map[string]Distribution

    // Throughput
    RequestsPerSec   float64
    TokensPerSec     float64

    // SLO attainment
    SLOAttainment    map[string]map[string]float64

    // Fairness
    TenantThroughput map[string]float64
    TenantLatency    map[string]float64
    JainFairnessIdx  float64

    // Efficiency
    CacheHitRate     map[KVTier]float64  // per-tier hit rates
    KVUtilization    map[KVTier][]float64
    KVTransferRate   float64             // offload/reload rate
    PreemptionRate   float64
    AdmissionRate    float64

    // Auto-scaling metrics
    ScaleEvents      int
    AvgReplicaCount  map[string]float64
    ScaleEfficiency  float64            // SLO attainment per replica

    // Pathology indicators
    PriorityInversions  int
    HeadOfLineBlocking  float64
    TailLatencyBlowup   float64
}
```

### Fitness Function Interface

```go
type FitnessFunction interface {
    Compute(metrics RawMetrics) map[string]float64
    Objectives() []ObjectiveSpec
}

type ObjectiveSpec struct {
    Name      string
    Direction string    // "maximize" or "minimize"
    Weight    float64
}
```

---

## 10. Trace Collection

Rich execution traces for evolutionary framework feedback.

```go
type SimulationTrace struct {
    Summary       EvaluationResult
    Events        []TraceEvent          // optional
    Decisions     DecisionTrace
    Anomalies     []Anomaly
}

type DecisionTrace struct {
    AdmissionDecisions  []AdmissionRecord
    PriorityDecisions   []PriorityRecord
    RoutingDecisions    []RoutingRecord
    BatchFormations     []BatchRecord
    PreemptionEvents    []PreemptionRecord
    ScaleDecisions      []ScaleRecord
    KVTransferEvents    []KVTransferRecord
}

type RoutingRecord struct {
    Timestamp       float64
    RequestID       string
    TenantID        string
    PriorityClass   string
    InputTokens     int
    PrefixHash      string
    InstanceStates  []InstanceSnapshot
    RouterState     RouterSnapshot
    ChosenInstance  InstanceID
    Reason          string
    ActualTTFT      float64
    CacheHit        bool
    CacheTier       KVTier              // which tier was hit

    // For counterfactual analysis (GEPA reflection)
    TopKCandidates  []CandidateScore    // alternatives considered
    ExpectedCacheHit bool               // what shadow model predicted
    Regret          float64             // post-hoc: how much better was best alternative?
}

type ScaleRecord struct {
    Timestamp       float64
    ConfigID        string
    Action          ScaleAction
    PreviousCount   int
    NewCount        int
    Reason          string
    Metrics         DeploymentMetrics

    // Actuation tracking
    ProvisioningDelay float64          // actual delay incurred
    WarmupEndTime     float64          // when instance reaches steady state
}

type KVTransferRecord struct {
    Timestamp       float64
    InstanceID      InstanceID
    BlockID         BlockID
    FromTier        KVTier
    ToTier          KVTier
    TransferLatency float64
    TransferType    TransferType
    Reason          string             // "offload", "reload", "pd_handoff"
}

type Anomaly struct {
    Type        string
    Timestamp   float64
    Description string
    Requests    []string
    Severity    float64
}
```

### Anomaly Types

| Type | Description |
|------|-------------|
| `priority_inversion` | Lower priority request scheduled before higher priority |
| `slo_violation` | Request missed SLO target |
| `cache_thrashing` | Rapid offload/reload cycles |
| `shadow_divergence` | Shadow KV prediction differed from actual (expected hit was miss, or vice versa) |
| `scale_oscillation` | Rapid scale up/down cycles |
| `pd_backpressure` | P/D transfer queue exceeded threshold |
| `preemption_storm` | Excessive preemptions in short window |

### Trace Verbosity Levels

| Level | Collected | Use Case |
|-------|-----------|----------|
| `minimal` | Summary metrics only | Fast fitness evaluation |
| `decisions` | All policy decisions + context | GEPA reflection, debugging |
| `full` | All events + decisions + anomalies | Deep analysis |

---

## 11. Trace Summarization

Intelligent summarization producing compact, actionable textual feedback for LLM-based evolution.

```go
type TraceSummary struct {
    Scores        map[string]float64
    TextFeedback  string              // target: 500-1500 tokens
}

type SummaryConfig struct {
    MaxTokens         int
    FocusAreas        []string
    IncludeExamples   int
    IncludeCounters   bool
}
```

### Example Summary Output

```
## Outcome
- Throughput: 847 tok/s (target: 1000) ❌
- P99 TTFT: 312ms (target: 200ms) ❌
- P99 TPOT: 42ms (target: 50ms) ✓
- SLO attainment: 71% realtime, 89% batch
- Fairness (Jain): 0.82
- Scale events: 3 (2 up, 1 down)

## Key Issues (3 detected)
1. ROUTING: 67% of cache misses came from prefix-unaware decisions
   - 142 requests routed away from instance with matching prefix
   - Pattern: load-balance weight dominated when queue_depth < 5
   - Shadow divergence rate: 23% (predictions often wrong)

2. ADMISSION: Bursty tenant "T3" caused 23% of realtime SLO misses
   - 15:42-15:47: T3 burst admitted 340 reqs, realtime TTFT spiked 3.2x

3. KV OFFLOAD: Excessive GPU→CPU transfers during burst
   - 847 offload events, 312 immediate reloads (thrashing)

## Counterfactual Analysis
- Routing regret: avg 12ms per decision (if best alternative chosen)
- 18 decisions where alternative would have hit cache (chosen missed)

## Worst Decisions (3 examples)
1. t=15:43.2 | Routed req_8821 (prefix=0xA3F) → instance 1
   Context: instance 0 had prefix cached (GPU), util=72%, queue=3
   Result: cache miss, TTFT=487ms (3x expected)
   Regret: instance 0 would have achieved TTFT=142ms

## Suggested Focus
- Increase cache_affinity_weight when prefix match exists
- Add tenant burst detection to admission policy
- Consider KV tier in routing decisions
```

### Summarization Strategies

| Strategy | Description |
|----------|-------------|
| `issue_focused` | Top N problems with examples |
| `decision_audit` | Sample decisions, annotate outcomes |
| `counterfactual` | "If X decided differently, Y improves" |
| `pattern_mining` | Frequent patterns correlated with poor outcomes |

---

## 12. Multi-Replica Simulation Engine

Cluster simulator orchestrating multiple instance simulators with a shared event loop.

```go
type ClusterSimulator struct {
    // Configuration
    Models          []Model
    Deployments     []DeploymentConfig
    Workload        WorkloadSpec

    // Policies
    Policy          PolicyBundle
    Router          *RouterState
    AdmissionPolicy AdmissionPolicy
    PriorityPolicy  PriorityPolicy
    RoutingPolicy   RoutingPolicy
    AutoScaler      *AutoScaler

    // Instances (organized by deployment)
    Instances       map[InstanceID]*InstanceSimulator

    // Simulation state
    EventQueue      *EventHeap
    Clock           float64

    // Determinism (see Section 13)
    RNG             *PartitionedRNG

    TraceConfig     TraceConfig
    Trace           SimulationTrace
}

type InstanceSimulator struct {
    ID              InstanceID
    DeploymentConfig string
    PoolType        PoolType
    Scheduler       InstanceScheduler
    WaitQueue       []*Request
    RunningBatch    *Batch
    KVCache         *KVCacheState
    Metrics         *MetricsCollector
}
```

### Event Types

```go
// Cluster-level
type RequestArrivalEvent struct { ... }
type RouteDecisionEvent struct { ... }
type InstanceStateUpdateEvent struct { ... }
type AutoScaleCheckEvent struct { ... }
type ScaleActionEvent struct { ... }
type InstanceReadyEvent struct { ... }  // after provisioning + warmup

// Instance-level
type InstanceStepEvent struct { ... }
type RequestCompletedEvent struct { ... }
type KVTransferEvent struct { ... }
type PDHandoffEvent struct { ... }  // for disaggregated P/D
```

### Event Ordering and Time Model

The simulation uses a single global event heap with strict ordering rules:

```go
type EventOrdering struct {
    // Primary: by timestamp
    // Secondary: by event type priority (for simultaneous events)
    // Tertiary: by event ID (deterministic tie-breaker)
    TypePriority map[EventType]int
}

// Event type priorities (lower = processed first at same timestamp)
var DefaultEventPriority = map[EventType]int{
    "RequestArrival":       1,  // arrivals first
    "RouteDecision":        2,  // then routing
    "InstanceStep":         3,  // then instance processing
    "RequestCompleted":     4,  // then completions
    "KVTransfer":           5,  // then transfers
    "AutoScaleCheck":       6,  // then scaling
    "ScaleAction":          7,
    "InstanceReady":        8,
}
```

**Time model:**
- Routing decisions are instantaneous (no simulated router CPU time)
- Instance steps are event-driven (triggered by completions, arrivals, or periodic ticks)
- Simultaneous events at same timestamp are ordered by type priority, then by deterministic event ID

### Simulation Loop

```go
func (c *ClusterSimulator) Run() EvaluationResult {
    c.initialize()
    c.generateArrivals()
    c.scheduleAutoScaleChecks()

    for c.EventQueue.Len() > 0 {
        event := heap.Pop(c.EventQueue)
        c.Clock = event.Timestamp()

        switch e := event.(type) {
        case *RequestArrivalEvent:
            c.handleArrival(e)  // admission → priority → routing
        case *RouteDecisionEvent:
            c.dispatchToInstance(e)
        case *InstanceStepEvent:
            c.Instances[e.InstanceID].Step(c.Clock)
        case *InstanceStateUpdateEvent:
            c.updateInstanceState(e.InstanceID)
        case *RequestCompletedEvent:
            c.recordCompletion(e)
        case *AutoScaleCheckEvent:
            c.evaluateAutoScale(e)
        case *ScaleActionEvent:
            c.executeScaleAction(e)
        case *InstanceReadyEvent:
            c.activateInstance(e)
        case *KVTransferEvent:
            c.handleKVTransfer(e)
        case *PDHandoffEvent:
            c.handlePDHandoff(e)
        }

        if c.Clock > c.Workload.Duration { break }
    }

    return c.computeResults()
}
```

### Framework Integration

**GEPA Adapter:**

```go
type BLISGEPAAdapter struct{}

func (a BLISGEPAAdapter) Evaluate(policy PolicyBundle, workload WorkloadSpec) (float64, SimulationTrace) {
    sim := NewClusterSimulator(policy, workload)
    result := sim.Run()
    return result.Fitness["primary"], result.Trace
}

func (a BLISGEPAAdapter) ExtractTracesForReflection(trace SimulationTrace) string {
    return SummarizeTrace(trace, defaultReflectionConfig)
}
```

**OpenEvolve Evaluator:**

```go
func BLISEvaluator(candidate PolicyBundle, config EvalConfig) map[string]float64 {
    sim := NewClusterSimulator(candidate, config.Workload)
    result := sim.Run()
    return result.Fitness
}
```

---

## 13. Behavioral Contracts

Contracts for BDD/TDD development. Full specifications to be refined during detailed design.

### Determinism Contract

Evolutionary optimization requires bit-for-bit reproducible simulations. Every run is uniquely determined by:

```go
type SimulationKey struct {
    PolicyID      string
    WorkloadSeed  int64
    SimSeed       int64
    JitterSeed    int64   // for timing noise, if any
}
```

**Determinism requirements:**

1. **Reproducibility**: Given the same `SimulationKey`, the simulation produces identical:
   - All metrics (bit-for-bit)
   - All traces (event-for-event)
   - All decisions

2. **Partitioned RNG**: A single RNG interface partitioned deterministically by subsystem:

```go
type PartitionedRNG struct {
    Master        *rand.Rand
    Subsystems    map[string]*rand.Rand  // "workload", "router", "instance_0", ...
}

func (p *PartitionedRNG) ForSubsystem(name string) *rand.Rand {
    if rng, ok := p.Subsystems[name]; ok {
        return rng
    }
    // Derive deterministically from master
    seed := p.Master.Int63() ^ hash(name)
    p.Subsystems[name] = rand.New(rand.NewSource(seed))
    return p.Subsystems[name]
}
```

3. **Tie-breaking rules**: All ties are resolved deterministically:
   - Routing ties (equal scores): lowest instance ID wins
   - Scheduler ties (equal priority): FIFO by arrival order
   - Simultaneous events: type priority, then event ID
   - KV eviction ties (equal LRU time): lowest block ID evicted

4. **No external state**: Policies cannot access:
   - Wall-clock time
   - External randomness
   - Filesystem / network
   - Global mutable state

### System-Wide Invariants

```
INVARIANT request_lifecycle:
  Every request reaches exactly one terminal state: COMPLETED | REJECTED | TIMED_OUT

INVARIANT causality:
  arrival_time ≤ admission_time ≤ routing_time ≤ enqueue_time ≤ completion_time

INVARIANT clock_monotonicity:
  Simulation clock never decreases

INVARIANT kv_cache_conservation:
  For each tier: allocated_blocks + free_blocks = total_blocks (per instance)

INVARIANT router_shadow_update:
  Shadow KV model updated on every routing decision

INVARIANT scale_bounds:
  For each deployment config: min_replicas ≤ current_replicas ≤ max_replicas

INVARIANT pd_handoff (disaggregated only):
  Decode cannot begin until prefill completes AND KV transfer completes
  (unless PipelineMode enabled, in which case: sufficient blocks transferred)

INVARIANT pd_ownership:
  After P→D transfer completes, source instance marks blocks as TRANSFERRED
  Blocks are never duplicated; single ownership at any time

INVARIANT no_double_counting:
  Each prefill token counted exactly once in throughput metrics
  Each decode token counted exactly once in TPOT calculation
  Preempted-and-restarted work is not double-counted

INVARIANT determinism:
  Same SimulationKey produces identical results across runs
```

### Policy Contracts

**AdmissionPolicy:**
- Input: request, router_state, cluster_state
- Output: ADMIT | REJECT | DELAY(duration)
- Must not modify inputs; decision recorded in trace

**PriorityPolicy:**
- Input: request, router_state, cluster_state
- Output: score (float64), hints (map)
- Deterministic for identical inputs

**RoutingPolicy:**
- Input: request, router_state, cluster_state
- Output: RoutingDecision with valid instance_id
- Updates router shadow KV model
- May implement filtering, scoring, flow control internally
- Must record top-k candidates for counterfactual analysis

**InstanceScheduler:**
- MakeBatch respects max_batch_size and max_token_budget
- SelectPreemptionVictim returns request from running_batch or nil
- No request loss: all requests eventually complete or timeout

**AutoScalePolicy:**
- Input: AutoScaleContext
- Output: ScaleDecision
- Respects min/max replica bounds
- Scale actions recorded in trace

### BDD Scenarios (Examples)

```gherkin
Scenario: Route to instance with cached prefix
  Given instance 0 has prefix "0xABC123" in GPU tier
  And instance 1 does not have prefix "0xABC123"
  When a request with prefix "0xABC123" arrives
  Then the routing policy selects instance 0

Scenario: Realtime requests scheduled before batch
  Given wait queue has batch request (arrival=100) and realtime request (arrival=101)
  When MakeBatch is called with priority-aware scheduler
  Then realtime request is scheduled first

Scenario: Auto-scale up on sustained high load
  Given deployment config "llama-70b-v1" has 2 replicas
  And queue depth exceeds threshold for 60 seconds
  When auto-scale policy evaluates
  Then scale decision is SCALE_UP

Scenario: Scale-up includes provisioning delay
  Given auto-scaler decides to scale up
  When ScaleActionEvent is processed
  Then new instance enters PROVISIONING state
  And instance becomes READY after ProvisioningDelay + ModelLoadTime
  And instance has degraded performance during WarmupDuration

Scenario: KV offload under memory pressure
  Given instance GPU KV utilization is 95%
  And CPU tier has available capacity
  When a new request requires KV allocation
  Then least-recently-used blocks are offloaded to CPU tier

Scenario: P/D handoff in disaggregated mode
  Given deployment uses disaggregated P/D architecture
  When prefill completes on prefill instance P0
  Then KV blocks transfer to decode instance D0
  And decode begins only after transfer completes
  And P0 marks transferred blocks as TRANSFERRED

Scenario: Deterministic replay
  Given a simulation with key (policy_1, seed_42, seed_99, seed_7)
  When the simulation runs twice
  Then both runs produce identical metrics and traces

Scenario: Shadow divergence tracking
  Given router's shadow model predicts prefix "0xABC" is on instance 0
  And actual cache state shows prefix was evicted
  When request with prefix "0xABC" is routed to instance 0
  Then a shadow_divergence anomaly is logged
  And routing record shows ExpectedCacheHit=true, CacheHit=false
```

---

## Summary

This design extends BLIS to support evolutionary policy optimization through:

1. **Multi-replica cluster simulation** with shared event loop and deterministic execution
2. **Flexible deployment model** supporting monolithic and disaggregated P/D architectures
3. **Pluggable policy pipeline** (admission → priority → routing) at cluster level
4. **Auto-scaling module** with realistic actuation latency (provisioning, warmup, drain)
5. **Pluggable instance schedulers** at instance level
6. **Tiered KV cache model** with GPU/CPU/Storage hierarchy, transfer latencies, and P/D ownership semantics
7. **Rich workload modeling** (multi-tenant, multi-model, priorities, prefixes, bursty arrivals, correlations)
8. **Sandboxed policy execution** for safe evolutionary search (Starlark/WASM)
9. **Flexible fitness evaluation** (single/multi-objective, SLO-based)
10. **Trace collection with counterfactual analysis** for GEPA reflection
11. **Framework-agnostic integration** supporting OpenEvolve and GEPA

Behavioral contracts—including determinism, KV ownership, and accounting invariants—provide the foundation for reliable evolutionary optimization and BDD/TDD development. Interfaces are designed to be expressive and flexible; implementation details will be refined during detailed design phases.
