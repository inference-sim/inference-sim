# Phase 1: Core Engine & Determinism - Implementation Plan

**Date:** 2026-02-09
**Status:** Draft
**Parent Design:** [Evolutionary Policy Optimization Design](./2026-02-06-evolutionary-policy-optimization-design.md)
**Branch:** TBD

## Overview

Phase 1 transforms BLIS from a single-instance simulator into a multi-replica cluster simulator with **guaranteed deterministic execution**. This is the foundation for all subsequent phases—without determinism, evolutionary optimization cannot reliably compare policy fitness.

### Goal

Multi-replica simulation engine with deterministic execution.

### Scope

- `ClusterSimulator` with event heap and simulation loop
- `InstanceSimulator` with basic request processing
- `DeploymentConfig` and `ReplicaPool` (monolithic only)
- `PartitionedRNG` and `SimulationKey` for determinism
- Event ordering rules and tie-breaking
- Basic `Request` lifecycle (arrival → enqueue → complete)
- **HFModelConfig**: HuggingFace config.json integration for model architecture
- **VLLMEngineConfig**: vLLM deployment parameters (TP, PP, DP, batch limits, KV blocks)

### Not In Scope

- Pluggable policies (admission, routing, priority) → Phase 2
- KV cache tiers (CPU, storage) → Phase 3
- Workload generation → Phase 4
- Auto-scaler → Phase 5
- P/D disaggregation → Phase 6

---

## Current State Analysis

### Existing Architecture (sim/ package)

| Component | File | Description |
|-----------|------|-------------|
| Event loop | `simulator.go` | Discrete-event simulation with min-heap |
| Events | `event.go` | `ArrivalEvent`, `QueuedEvent`, `StepEvent`, etc. |
| Requests | `request.go` | Request struct with state machine |
| KV Cache | `kvcache.go` | Block-based cache with LRU eviction |
| Batch | `batch.go` | Batch formation logic |
| Queue | `queue.go` | FIFO wait queue |
| Metrics | `metrics.go` | TTFT, TPOT, E2E collection |
| Model Config | `model_hardware_config.go` | `HFConfig`, `ModelConfig`, `HardwareCalib` |
| Defaults | `defaults.yaml` | Pre-trained coefficients, vLLM versions |

### Gap Analysis

| Capability | Current | Phase 1 Target |
|------------|---------|----------------|
| Instances | 1 | N (configurable) |
| Event ordering | Timestamp only | Timestamp → Type → EventID |
| RNG | Single, unpartitioned | Partitioned by subsystem |
| Deployment model | None | Model → Config → Pool → Instances |
| Model config | Flat `ModelConfig` | `HFModelConfig` with MoE support |
| Engine config | Scattered | `VLLMEngineConfig` struct |
| Determinism | Partial | Guaranteed bit-for-bit replay |

---

## PR Breakdown

### PR 1: Deployment Model & Configuration

**Goal:** Define deployment hierarchy, model configuration, and engine configuration.

### PR 2: ClusterSimulator & Event Orchestration

**Goal:** Create cluster-level simulation loop with deterministic event ordering.

### PR 3: PartitionedRNG & Deterministic Replay

**Goal:** Guarantee bit-for-bit reproducibility across simulation runs.

---

## Data Structures

### Type Definitions

```go
// Identity types
type InstanceID string
type ConfigID string
type ModelID string

// Pool and architecture types
type PoolType string      // "monolithic", "prefill", "decode"
type ArchitectureType string  // "monolithic", "disaggregated_pd"

// Event types with priority ordering
type EventType string
// Priority: RequestArrival(1) < RouteDecision(2) < InstanceStep(3) < RequestCompleted(4)
```

### HFModelConfig

Wraps HuggingFace config.json with computed properties for simulation.

**Fields:**
- Identity: `ModelID`, `ModelType`, `Architectures`
- Dimensions: `NumLayers`, `HiddenSize`, `IntermediateSize`, `NumAttentionHeads`, `NumKVHeads`, `VocabSize`, `MaxPositionEmbeddings`
- Precision: `TorchDtype`, `BytesPerParam`
- MoE: `*MoEConfig` (nil for dense models)
- Raw: `*sim.HFConfig` (preserves all HF fields)

**MoEConfig fields:**
- `NumLocalExperts` - Total experts per layer (e.g., 8 for Mixtral)
- `NumExpertsPerTok` - Active experts per token (e.g., 2)
- `ExpertIntermediateSize` - Expert MLP hidden size
- `NumSharedExperts` - Shared experts (DeepSeek-style, 0 if none)
- `SharedExpertIntermediateSize` - Shared expert MLP size

**Computed properties:**
- `IsMoE() bool` - True if MoE != nil && NumLocalExperts > 1
- `HeadDim() int` - HiddenSize / NumAttentionHeads
- `KVCacheBytesPerToken() int64` - 2 × NumLayers × NumKVHeads × HeadDim × BytesPerParam
- `TotalParameters() int64` - Full model parameter count
- `ActiveParametersPerToken() int64` - Parameters activated per token (differs for MoE)

### VLLMEngineConfig

Captures vLLM deployment parameters affecting simulation behavior.

**Fields:**
- Identity: `VLLMVersion`
- Parallelism: `TensorParallelSize`, `PipelineParallelSize`, `DataParallelSize`
- MoE: `*VLLMMoEConfig` (nil for dense models)
- Batch limits: `MaxNumSeqs`, `MaxNumBatchedTokens`
- KV cache: `BlockSize`, `GPUMemoryUtilization`, `TotalKVBlocks`, `SwapSpace`
- Scheduling: `EnableChunkedPrefill`, `MaxNumPartialPrefills`, `ChunkSize`
- Caching: `EnablePrefixCaching`

**VLLMMoEConfig fields:**
- `EnableAllToAll` - All-to-all communication for expert routing
- `UseFusedMoEKernel` - Use optimized MoE kernels
- `MoEKernelBackend` - "triton", "cuda", or "auto"

**Computed properties:**
- `TotalGPUs() int` - TP × PP × DP
- `EffectiveExpertParallelism() int` - DP × TP (implicit EP for MoE)
- `IsMoEDeployment() bool` - True if MoEConfig != nil
- `Validate() error` - Returns error if config is invalid

**Expert Parallelism Note:**
In vLLM, expert parallelism (EP) is not a separate parameter. It is implicit as DP × TP.
Example: With DP=2 and TP=4, experts are distributed across all 8 GPUs.

### DeploymentConfig

```go
type DeploymentConfig struct {
    ConfigID       ConfigID
    ModelID        ModelID
    Architecture   ArchitectureType      // MONOLITHIC only in Phase 1
    ModelConfig    *HFModelConfig        // HuggingFace model configuration
    EngineConfig   *VLLMEngineConfig     // vLLM engine parameters
    HardwareConfig *sim.HardwareCalib    // GPU specs (reuse existing type)
    AlphaCoeffs    []float64             // Queueing delay model coefficients
    BetaCoeffs     []float64             // Step time model coefficients
    ReplicaPool    *ReplicaPool
}
```

### ReplicaPool

```go
type ReplicaPool struct {
    PoolID      string
    PoolType    PoolType
    Instances   []*InstanceSimulator
    MinReplicas int
    MaxReplicas int
}
```

**Operations:** `AddInstance`, `RemoveInstance`, `GetInstance`, `Len`

### InstanceSimulator

Wraps existing sim package components (WaitQueue, Batch, KVCacheState) to represent a single vLLM instance.

**Fields:**
- Identity: `ID`, `DeploymentConfig`, `PoolType`
- State: `WaitQueue`, `RunningBatch`, `KVCache`, `Clock`, `Metrics`
- Config: `ModelConfig`, `EngineConfig`, `HardwareConfig`, `AlphaCoeffs`, `BetaCoeffs`

**Operations:**
- `EnqueueRequest(req *Request)`
- `WaitQueueDepth() int`
- `RunningBatchSize() int`
- `KVCacheUtilization() float64`
- `TotalKVBlocks() int`
- `Step(clock int64) (stepDuration int64, completedReqs []*Request)`

### ClusterSimulator

```go
type ClusterSimulator struct {
    // Configuration
    Models      []*Model
    Deployments map[ConfigID]*DeploymentConfig
    Instances   map[InstanceID]*InstanceSimulator

    // Simulation state
    EventQueue        *EventHeap
    Clock             int64
    Horizon           int64

    // Request tracking
    Requests          map[string]*Request
    PendingRequests   int
    CompletedRequests int

    // Determinism
    Key SimulationKey
    RNG *PartitionedRNG

    // Metrics
    Metrics *ClusterMetrics
}
```

**Operations:**
- `AddDeployment(config *DeploymentConfig) error`
- `GetInstance(id InstanceID) *InstanceSimulator`
- `ListInstances() []InstanceID`
- `ScheduleEvent(e Event)`
- `Run() *ClusterMetrics`

### SimulationKey

```go
type SimulationKey struct {
    PolicyID     string  // Identifies policy version (Phase 2+)
    WorkloadSeed int64   // Controls workload generation (Phase 4+)
    SimSeed      int64   // Controls simulation randomness
    JitterSeed   int64   // Controls timing noise (if any)
}
```

### PartitionedRNG

Provides isolated RNG streams per subsystem for deterministic simulation.

**Fields:** `subsystems map[string]*rand.Rand`

**Operations:**
- `ForSubsystem(name string) *rand.Rand`
- `ForInstance(id InstanceID) *rand.Rand`

**Subsystem constants:** `"workload"`, `"router"`, `"instance_{id}"`, `"scheduler_{id}"`

### Event Interface

```go
type Event interface {
    Timestamp() int64
    EventID() uint64
    Type() EventType
    Execute(sim *ClusterSimulator)
}
```

**Event types:** `RequestArrivalEvent`, `RouteDecisionEvent`, `InstanceStepEvent`, `RequestCompletedEvent`

### Request State Machine

Requests transition through states:
```
PENDING → ROUTED → QUEUED → RUNNING → COMPLETED
```

**Request fields** (additions to existing sim.Request):
- `TargetInstance InstanceID` - Assigned instance
- `State string` - Current lifecycle state
- `ArrivalTime`, `RouteTime`, `EnqueueTime`, `ScheduleTime`, `CompletionTime`, `FirstTokenTime`

### Metrics

**ClusterMetrics:**
- `CompletedRequests`, `TotalRequests`
- `TotalInputTokens`, `TotalOutputTokens`
- `SimDuration`
- `PerInstance map[InstanceID]*InstanceMetrics`

**InstanceMetrics:**
- `CompletedRequests`, `TotalInputTokens`, `TotalOutputTokens`
- `PeakWaitQueueDepth`, `PeakBatchSize`

---

## Behavioral Contracts

### BC-1: HFModelConfig Validity

```
INVARIANT model_config_valid:
  For all HFModelConfig m:
    m.NumLayers > 0
    m.HiddenSize > 0
    m.NumAttentionHeads > 0
    m.NumKVHeads > 0 AND m.NumKVHeads <= m.NumAttentionHeads
    m.HiddenSize % m.NumAttentionHeads == 0  (HeadDim is integer)
    m.BytesPerParam in {1, 2, 4}

  If m.IsMoE():
    m.MoE.NumLocalExperts > 1
    m.MoE.NumExpertsPerTok >= 1
    m.MoE.NumExpertsPerTok <= m.MoE.NumLocalExperts
```

### BC-2: VLLMEngineConfig Validity

```
INVARIANT engine_config_valid:
  For all VLLMEngineConfig e:
    e.TensorParallelSize >= 1
    e.PipelineParallelSize >= 1
    e.DataParallelSize >= 1
    e.MaxNumSeqs > 0
    e.MaxNumBatchedTokens > 0
    e.BlockSize > 0
    0 < e.GPUMemoryUtilization <= 1.0
    e.TotalKVBlocks > 0 (if explicitly set)
```

### BC-3: ReplicaPool Bounds

```
INVARIANT pool_bounds:
  For all ReplicaPool p:
    p.MinReplicas <= len(p.Instances) <= p.MaxReplicas

  AddInstance fails if len(Instances) == MaxReplicas
  RemoveInstance fails if len(Instances) == MinReplicas
```

### BC-4: Instance Isolation

```
INVARIANT instance_isolation:
  Each InstanceSimulator has independent:
    - WaitQueue
    - RunningBatch
    - KVCacheState
    - Metrics

  Modifying instance A does not affect instance B.
  No shared mutable state between instances.
```

### BC-5: Clock Monotonicity

```
INVARIANT clock_monotonicity:
  For all consecutive events e1, e2 processed by ClusterSimulator:
    e2.Timestamp() >= e1.Timestamp()

  The simulation Clock field never decreases.
  Processing an event with timestamp < Clock is a fatal error.
```

### BC-6: Event Ordering (Deterministic)

```
INVARIANT event_ordering:
  Events are processed in strict order:
    1. Lower timestamp first
    2. At same timestamp: lower EventType priority first
       (Arrival=1 < Route=2 < Step=3 < Completed=4)
    3. At same timestamp and type: lower EventID first

  This ordering is deterministic regardless of insertion order.
```

### BC-7: Request Lifecycle

```
INVARIANT request_lifecycle:
  Every request transitions through states in order:
    PENDING → ROUTED → QUEUED → RUNNING → COMPLETED

  No request is lost. All requests either:
    - Reach COMPLETED state, or
    - Remain in queue when simulation ends (horizon reached)

  State transitions are monotonic (no backward transitions).
```

### BC-8: Causality

```
INVARIANT causality:
  For every completed request r:
    r.ArrivalTime <= r.RouteTime <= r.EnqueueTime <= r.ScheduleTime <= r.CompletionTime
    r.FirstTokenTime >= r.ScheduleTime
    r.FirstTokenTime <= r.CompletionTime
```

### BC-9: Deterministic Replay

```
INVARIANT determinism:
  Given identical inputs (SimulationKey, DeploymentConfig, Request arrivals):
    Run1 = NewClusterSimulator(key, config).Run()
    Run2 = NewClusterSimulator(key, config).Run()

  Then:
    Run1.Metrics == Run2.Metrics (byte-for-byte identical)
    Run1.Clock == Run2.Clock
    For all requests r: Run1.r.* == Run2.r.* (all timestamps identical)
```

### BC-10: RNG Subsystem Isolation

```
INVARIANT rng_isolation:
  For PartitionedRNG with seed S:
    The sequence returned by ForSubsystem("A") is independent of:
      - Whether ForSubsystem("B") was called
      - When ForSubsystem("B") was called
      - How many values were drawn from subsystem "B"

  Subsystem seed derivation must be order-independent.
  Seed for subsystem X = f(master_seed, X) where f is deterministic.
```

### BC-11: No External State Dependency

```
INVARIANT no_external_state:
  Simulation results depend ONLY on:
    - SimulationKey
    - DeploymentConfig
    - Injected request arrivals

  Results do NOT depend on:
    - Wall-clock time
    - Goroutine scheduling
    - External randomness (only PartitionedRNG)
    - Filesystem or network state
```

### BC-12: Batch Size Limit

```
INVARIANT batch_size_limit:
  For all InstanceSimulator i:
    i.RunningBatchSize() <= i.EngineConfig.MaxNumSeqs

  Batch formation respects engine configuration limits.
```

### BC-13: KV Cache Conservation

```
INVARIANT kv_cache_conservation:
  For all InstanceSimulator i:
    i.KVCache.AllocatedBlocks + i.KVCache.FreeBlocks == i.TotalKVBlocks()

  KV blocks are neither created nor destroyed during simulation.
```

### BC-14: Event Queue Bounds

```
INVARIANT event_queue_bounded:
  Peak event queue size = O(active_requests × events_per_request)

  Each request generates bounded events:
    - 1 arrival
    - 1 route decision
    - O(output_tokens / batch_size) step events
    - 1 completion

  Queue does not grow unboundedly.
```

---

## Test Scenarios

### Feature: HuggingFace Model Configuration

```gherkin
Scenario: Load dense model config from config.json
  Given a HuggingFace config.json for "meta-llama/llama-3.1-8b-instruct"
  When the config is loaded
  Then ModelType == "llama"
  And NumLayers == 32, HiddenSize == 4096
  And NumAttentionHeads == 32, NumKVHeads == 8
  And IsMoE() == false

Scenario: Load MoE model config (Mixtral)
  Given a HuggingFace config.json for "mistralai/Mixtral-8x7B-Instruct-v0.1"
  When the config is loaded
  Then IsMoE() == true
  And MoE.NumLocalExperts == 8
  And MoE.NumExpertsPerTok == 2

Scenario: Compute KV cache bytes per token
  Given HFModelConfig with NumLayers=32, NumKVHeads=8, HiddenSize=4096, NumAttentionHeads=32, BytesPerParam=2
  Then KVCacheBytesPerToken() == 2 * 32 * 8 * 128 * 2 == 131072

Scenario: KV cache identical for dense and MoE with same attention config
  Given dense and MoE models with identical attention dimensions
  Then both have identical KVCacheBytesPerToken()

Scenario: Active parameters less than total for MoE
  Given MoE model with 8 experts, 2 active per token
  Then ActiveParametersPerToken() < TotalParameters()

Scenario: Reject invalid model config
  Given HFModelConfig with NumKVHeads > NumAttentionHeads
  When Validate() is called
  Then error is returned
```

### Feature: vLLM Engine Configuration

```gherkin
Scenario: Default engine config has sensible values
  When DefaultVLLMEngineConfig() is called
  Then TensorParallelSize == 1
  And MaxNumSeqs == 256
  And BlockSize == 16
  And EnableChunkedPrefill == true

Scenario: Total GPUs calculation
  Given VLLMEngineConfig with TP=2, PP=2, DP=4
  Then TotalGPUs() == 16

Scenario: Effective expert parallelism for MoE
  Given VLLMEngineConfig with TP=2, DP=4
  Then EffectiveExpertParallelism() == 8  # DP × TP

Scenario: Reject invalid engine config
  Given VLLMEngineConfig with TensorParallelSize=0
  When Validate() is called
  Then error is returned

Scenario: Reject invalid GPU memory utilization
  Given VLLMEngineConfig with GPUMemoryUtilization=1.5
  When Validate() is called
  Then error is returned
```

### Feature: Replica Pool Management

```gherkin
Scenario: Add instance within bounds
  Given ReplicaPool with min=1, max=3, current=2
  When AddInstance is called
  Then pool size == 3, no error

Scenario: Add instance exceeds max
  Given ReplicaPool with max=3, current=3
  When AddInstance is called
  Then error is returned, pool size == 3

Scenario: Remove instance within bounds
  Given ReplicaPool with min=1, current=2
  When RemoveInstance is called
  Then pool size == 1, no error

Scenario: Remove instance violates min
  Given ReplicaPool with min=1, current=1
  When RemoveInstance is called
  Then error is returned, pool size == 1
```

### Feature: Instance Isolation

```gherkin
Scenario: Enqueue affects only target instance
  Given deployment with instances [inst_0, inst_1]
  When request R1 is enqueued to inst_0
  Then inst_0.WaitQueueDepth() == 1
  And inst_1.WaitQueueDepth() == 0

Scenario: Step affects only target instance
  Given deployment with instances [inst_0, inst_1] both with queued requests
  When inst_0.Step() is called
  Then only inst_0 state changes
  And inst_1 state is unchanged
```

### Feature: Event Ordering

```gherkin
Scenario: Events processed in timestamp order
  Given events at timestamps [100, 50, 150]
  When simulation runs
  Then events processed in order [50, 100, 150]

Scenario: Same-timestamp events use type priority
  Given RequestArrivalEvent and InstanceStepEvent both at t=100
  When simulation runs
  Then Arrival (priority=1) processes before Step (priority=3)

Scenario: Same-timestamp same-type events use EventID
  Given two InstanceStepEvents at t=100 with EventIDs [5, 3]
  When simulation runs
  Then EventID=3 processes before EventID=5

Scenario: Event ordering deterministic regardless of insertion order
  Given events inserted in random order
  When simulation runs twice with same events
  Then processing order is identical both times
```

### Feature: Clock Behavior

```gherkin
Scenario: Clock advances monotonically
  Given simulation with events at various timestamps
  When simulation runs
  Then Clock never decreases between events

Scenario: Clock stops at horizon
  Given horizon=1000 and events at [500, 800, 1200]
  When simulation runs
  Then events at [500, 800] processed
  And event at 1200 not processed
```

### Feature: Request Lifecycle

```gherkin
Scenario: Request completes successfully
  Given cluster with 2 instances
  And request R1 arrives at t=0
  When simulation runs to completion
  Then R1.State == "COMPLETED"
  And R1.CompletionTime > R1.ArrivalTime

Scenario: Multiple requests complete
  Given requests [R1, R2, R3] arriving at [0, 10, 20]
  When simulation runs to completion
  Then all requests reach COMPLETED state
  And CompletedRequests == 3

Scenario: Request timestamps satisfy causality
  Given completed request R1
  Then R1.ArrivalTime <= R1.RouteTime <= R1.EnqueueTime
  And R1.EnqueueTime <= R1.ScheduleTime <= R1.CompletionTime
```

### Feature: Deterministic Replay

```gherkin
Scenario: Same seed produces identical results
  Given SimulationKey with SimSeed=42
  And identical deployment config
  And identical request arrivals
  When simulation runs twice
  Then both runs produce identical:
    - CompletedRequests
    - TotalInputTokens, TotalOutputTokens
    - Final Clock value
    - Per-request timestamps (TTFT, completion time, etc.)

Scenario: RNG subsystem isolation
  Given PartitionedRNG with seed=42
  When "router" subsystem generates sequence S1
  And new PartitionedRNG with seed=42 is created
  And "workload" subsystem generates 100 values (consuming RNG)
  And "router" subsystem generates sequence S2
  Then S1 == S2

Scenario: Simulation independent of wall-clock time
  Given identical SimulationKey and config
  When simulation runs at different wall-clock times
  Then results are identical
```

### Feature: Batch Size Limits

```gherkin
Scenario: Batch respects MaxNumSeqs
  Given EngineConfig with MaxNumSeqs=128
  And 200 requests in wait queue
  When batch is formed
  Then batch size <= 128
```

### Feature: KV Cache Conservation

```gherkin
Scenario: KV blocks conserved during simulation
  Given instance with TotalKVBlocks=29205
  When simulation runs with various requests
  Then at all times: allocated + free == 29205
```

---

## File Structure

```
sim/cluster/
├── types.go               # InstanceID, ConfigID, ModelID, PoolType, ArchitectureType, EventType
├── model_config.go        # HFModelConfig, MoEConfig, RopeScalingConfig
├── model_config_test.go   # BC-1 tests
├── engine_config.go       # VLLMEngineConfig, VLLMMoEConfig
├── engine_config_test.go  # BC-2 tests
├── deployment.go          # Model, DeploymentConfig, ReplicaPool
├── deployment_test.go     # BC-3 tests
├── instance.go            # InstanceSimulator
├── instance_test.go       # BC-4, BC-12, BC-13 tests
├── events.go              # Event interface, event types, priorities
├── event_heap.go          # Priority queue with deterministic ordering
├── event_heap_test.go     # BC-6 tests
├── simulator.go           # ClusterSimulator, Run loop
├── simulator_test.go      # BC-5, BC-7, BC-8, BC-14 tests
├── metrics.go             # ClusterMetrics, InstanceMetrics
├── rng.go                 # SimulationKey, PartitionedRNG
├── rng_test.go            # BC-10 tests
└── determinism_test.go    # BC-9, BC-11 comprehensive tests
```

---

## Dependencies

### Internal (reuse from sim/)
- `sim.HFConfig` - Raw HuggingFace config wrapper
- `sim.HardwareCalib` - GPU hardware specifications
- `sim.WaitQueue` - FIFO request queue
- `sim.Batch` - Batch formation
- `sim.KVCacheState` - Block-based KV cache
- `sim.Request` - Request struct (extend with new fields)
- `defaults.yaml` - Pre-trained coefficients

### External (standard library)
- `container/heap` - Heap interface for event queue
- `hash/fnv` - Deterministic hash for RNG seed derivation
- `encoding/json` - HuggingFace config parsing

---

## Integration Notes

### Extending sim.Request

Add fields to existing Request struct:
- `TargetInstance InstanceID`
- `State string` (lifecycle state)
- Additional timestamps as needed

### InstanceSimulator Composition

InstanceSimulator composes (not inherits) existing sim package components:
- Creates its own `WaitQueue`, `Batch`, `KVCacheState`
- Delegates step logic to existing implementations
- Wraps to add cluster-level concerns (instance ID, metrics)

### KV Cache Initialization

InstanceSimulator initializes KVCache with capacity from:
1. `EngineConfig.TotalKVBlocks` if explicitly set, OR
2. Computed from `ModelConfig` + `HardwareConfig` + `EngineConfig.GPUMemoryUtilization`

### Round-Robin Routing (Phase 1)

Phase 1 uses simple round-robin instance selection:
- Maintain index of last-selected instance
- Cycle through available instances
- Phase 2 replaces with pluggable RoutingPolicy

### Step Time Estimation (Unchanged)

Phase 1 **reuses the existing step time estimation logic unchanged**. The existing `sim` package provides two modes:

**1. Blackbox (Data-driven) mode:**
```
stepTime = β₀ + β₁ × TotalCacheMissTokens + β₂ × TotalDecodeTokens
```
- Uses pre-trained linear regression coefficients from `defaults.yaml`
- `BetaCoeffs` in DeploymentConfig provides these coefficients
- Requires pre-training for each (model, GPU, TP, vLLM version) combination

**2. Roofline (Analytical) mode:**
- Computes FLOPs for attention and MLP based on `HFModelConfig`
- Computes memory access bytes (weights, KV cache, activations)
- Applies roofline model: `time = max(compute_bound, memory_bound)`
- Adds per-layer overhead and TP communication overhead from `HardwareCalib`
- No pre-training required; works with any model via config.json

**3. Alpha model for queueing/processing overhead:**
```
queueingTime = α₀ + α₁ × inputLength
outputTokenProcessingTime = α₂
```
- `AlphaCoeffs` in DeploymentConfig provides these coefficients

**InstanceSimulator.Step() delegates to existing logic:**
- Calls existing `makeRunningBatch()` for batch formation
- Calls existing `getStepTime()` or `getStepTimeRoofline()` based on mode
- Phase 1 wraps this; does not modify the estimation algorithms

---

## Success Criteria

Phase 1 is complete when:

1. **Multi-replica works**
   - [ ] Simulation runs with N instances
   - [ ] Requests distributed across instances (round-robin)
   - [ ] Each instance has isolated state (BC-4)

2. **Determinism verified**
   - [ ] Same SimulationKey → identical results (BC-9)
   - [ ] RNG subsystem isolation verified (BC-10)
   - [ ] No external state dependency (BC-11)

3. **Event ordering explicit**
   - [ ] Timestamp → type → EventID ordering (BC-6)
   - [ ] Clock monotonicity (BC-5)

4. **All invariants hold**
   - [ ] All 14 behavioral contracts pass automated tests
   - [ ] Tests run in CI

5. **Metrics aggregate correctly**
   - [ ] ClusterMetrics sums InstanceMetrics
   - [ ] Per-request metrics tracked

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Existing simulator tightly coupled | Hard to wrap in InstanceSimulator | Refactor incrementally; composition over inheritance |
| Hidden global state in existing code | Breaks determinism (BC-9, BC-11) | Audit all `rand` usage; inject PartitionedRNG |
| RNG seed derivation order-dependent | Breaks BC-10 | Use hash-only derivation: `seed = master_seed XOR hash(subsystem_name)` |
| Test flakiness | False confidence | No wall-clock assertions; pure logic tests |

**Non-Risks (addressed by design):**
- **Floating-point non-determinism**: Go IEEE 754 ops are deterministic on same platform; cross-platform not required
- **Event heap performance**: Standard heap O(log n) sufficient; bounded by BC-14
- **Instance failure handling**: Out of scope; Phase 5 (auto-scaler)

---

## Open Questions (Resolved)

1. **Request struct modification**: Add `TargetInstance` and `State` to existing `sim.Request`.

2. **Instance step granularity**: Reuse existing step logic; return duration and completed requests.

3. **Metric aggregation**: Post-simulation computation for simplicity.

---

## Next Phase Preview

**Phase 2: Policy Interfaces & Defaults** will add:
- `AdmissionPolicy` interface + `AlwaysAdmit` default
- `PriorityPolicy` interface + `ConstantPriority` default
- `RoutingPolicy` interface + `RoundRobin` default
- `InstanceScheduler` interface + `FCFSScheduler` default
- `PolicyBundle` for grouping policies

Phase 1's round-robin routing will be replaced by pluggable `RoutingPolicy`.
