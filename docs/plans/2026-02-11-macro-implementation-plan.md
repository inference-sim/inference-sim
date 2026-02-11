# BLIS Evolutionary Policy Optimization: Macro-Level Implementation Plan

**Date:** 2026-02-11
**Status:** Draft
**Target:** Multi-replica cluster simulation with pluggable policies
**Based on:** [Design Document](2026-02-06-evolutionary-policy-optimization-design.md)

---

## A) Executive Summary

This plan transforms BLIS from a single-instance LLM inference simulator into a multi-replica cluster simulator with pluggable control policies.

**BLIS remains a standalone tool.** Users can:
1. Run cluster simulations directly via CLI for capacity planning
2. Use parameterized policies to experiment with routing/scheduling strategies
3. Analyze results via metrics and decision traces

**Optional framework integration.** Evolutionary frameworks (OpenEvolve, GEPA) can wrap BLIS as an evaluator—BLIS provides adapters to make this easier, but does not depend on these frameworks.

**Coefficient learning is out of scope.** BLIS consumes pre-trained coefficients (alpha/beta for latency estimation). A separate instrumentation effort collects real vLLM data to train these coefficients.

**Implementation:**
- 6 phases, 23 PRs
- Each PR is CLI-exercisable immediately after merge
- Estimated 10-11 weeks with 3-4 developers

---

## B) Repository Recon Summary

### B.1 Package Structure

| Package | SLOC | Responsibility |
|---------|------|----------------|
| `sim/` | 2051 | Core single-instance discrete-event simulator |
| `cmd/` | 373 | CLI interface (Cobra), configuration loading |
| `main.go` | 12 | Entry point |

### B.2 Core Data Structures

**Simulator** (`sim/simulator.go:72-113`):
- Event queue (min-heap), clock (`int64` ticks), wait queue, KV cache, running batch
- Single `*rand.Rand`, alpha/beta coefficients for latency estimation

**Request** (`sim/request.go:18-34`):
- Lifecycle: `queued → running → completed`
- Tracks `ProgressIndex`, TTFT, ITL, arrival/completion times

**Events** (`sim/event.go`):
- Interface: `Timestamp() int64`, `Execute(*Simulator)`
- Types: `ArrivalEvent`, `QueuedEvent`, `ScheduledEvent`, `StepEvent`, `RequestLeftEvent`, `PreemptionEvent`

**KVCache** (`sim/kvcache.go`):
- Block-based with LRU eviction (models vLLM's PagedAttention)
- Prefix caching via SHA256 hash matching

### B.3 Hardcoded Behaviors Requiring Extraction

| Behavior | Location | Target Interface |
|----------|----------|------------------|
| FIFO batch formation | `simulator.go:341-365` | `InstanceScheduler` |
| LIFO preemption | `simulator.go:248-277` | `InstanceScheduler` |
| Single-tier KV | `kvcache.go` | Tiered `KVCacheState` |
| All-admit | implicit | `AdmissionPolicy` |

### B.4 Extension Strategy

- **Composition over modification**: `ClusterSimulator` wraps `InstanceSimulator` instances
- **Interface extraction**: Pull hardcoded behaviors into pluggable interfaces
- **Backward compatibility**: `--num-instances 1` produces identical results to current

---

## C) High-Level Objectives + Non-Goals

### Objectives

1. **Multi-replica simulation** with shared clock and coordinated events
2. **Deterministic execution** — same seeds produce bit-for-bit identical results
3. **Pluggable policies** — admission, priority, routing, scheduling, auto-scaling
4. **Tiered KV cache** — GPU + CPU with offload/reload latency modeling
5. **P/D disaggregation** — separate prefill and decode pools
6. **Rich observability** — decision traces, counterfactual analysis
7. **Framework adapters** — optional conveniences for GEPA/OpenEvolve

### Non-Goals

- **Coefficient training** — BLIS consumes coefficients; training is a separate effort
- **Sim-to-real validation** — deferred to instrumentation workstream
- **Production deployment** — research/planning tool only
- **GPU execution** — CPU-only simulation
- **Arbitrary code execution** — no Starlark/WASM; frameworks handle code generation

### Compatibility Guarantees

- Existing `./simulation_worker run` commands unchanged
- Single-instance mode identical to current implementation
- Output JSON schema: additions only, no breaking changes

---

## D) Modeling Decisions and Simplifications

This section documents how BLIS models real systems and where we intentionally simplify.

### D.1 KV Cache Model

**Based on:** vLLM PagedAttention [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023]

| Real System | BLIS Model | Simplification |
|-------------|------------|----------------|
| Variable block sizes | Fixed `BlockSizeTokens` | Simpler allocation math |
| GPU memory fragmentation | Perfect packing | Overly optimistic; acceptable for relative comparisons |
| Async block operations | Synchronous within step | No intra-step concurrency |

**Prefix Caching:** Models vLLM's automatic prefix caching. Hash computed over token sequence; matching prefix reuses existing blocks.

### D.2 Latency Estimation

**Based on:** Empirical coefficients from instrumented vLLM deployments.

| Component | Model | Coefficients |
|-----------|-------|--------------|
| Queueing delay | `α₀ + α₁ * input_len` | Learned from trace data |
| Step time | `β₀ + β₁ * cache_miss_tokens + β₂ * decode_tokens` | Learned from busy-loop instrumentation |
| Roofline (alternative) | FLOPs / peak_throughput | Analytical, for new hardware |

**Limitation:** Coefficients are hardware/model/TP-specific. Generalization requires re-training.

### D.3 Multi-Tier KV Cache

**Based on:** vLLM CPU offloading and research systems like FlexGen [Sheng et al., 2023].

| Tier | Latency Model | Capacity Model |
|------|---------------|----------------|
| GPU | `access_latency = 0` | `gpu_blocks` parameter |
| CPU | `access_latency = transfer_time(blocks)` | `cpu_blocks` parameter |
| Storage | Deferred | — |

**Transfer model:** `time = base_latency + blocks * block_size / bandwidth`

### D.4 Prefill-Decode Disaggregation

**Based on:** DistServe [Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024] and Splitwise [Patel et al., 2024].

| Real System | BLIS Model | Simplification |
|-------------|------------|----------------|
| RDMA KV transfer | Latency + bandwidth model | No queue depth modeling |
| Speculative decode | Not modeled | Out of scope |
| Multi-stage pipeline | Two-stage (P → D) only | Sufficient for policy research |

**Ownership model:** Single ownership; blocks transfer from prefill to decode instance, not duplicated.

### D.5 Router and Scheduling

**Based on:** llm-d router architecture concepts.

| Component | BLIS Model | Notes |
|-----------|------------|-------|
| Admission | ADMIT / REJECT / DELAY | Rate limiting, tenant quotas |
| Priority | Numeric score | Enables SLO differentiation |
| Routing | Filter → Score → Select | Prefix-aware, load-aware |
| Instance scheduling | Batch formation + preemption | FCFS default, pluggable |

**Routing policy freedom:** Individual routing policy implementations are free to maintain their own internal state (e.g., predicted cache state for prefix-aware routing). The router provides observable metrics and instance snapshots; policies decide what additional tracking they need.

### D.6 Auto-Scaling

**Based on:** Kubernetes HPA patterns with LLM-specific considerations.

| Aspect | BLIS Model |
|--------|------------|
| Trigger | Periodic or event-based (queue depth, SLO) |
| Actuation | Provisioning delay + model load time + warmup |
| Scale-down | Drain policy (wait, immediate, redirect) |
| Cost modeling | Replica-seconds accumulated; configurable cost-per-replica |

**Simplification:** No predictive scaling; threshold-based default policy.

### D.7 Observable Signals (Policy Input Contract)

Policies operate under **architectural locality constraints**: they may only consume signals naturally available at their control point. This section enumerates what each policy type can observe.

**Admission Policy Observables:**
- Request: `InputTokens`, `TenantID`, `PrefixHash`, `SLOClass`
- Tenant state: `RequestCount`, `ActiveRequests`, `RecentRate`
- Global: `InFlightRequests`, `RequestRate`

**Priority Policy Observables:**
- Same as admission, plus admission decision

**Routing Policy Observables:**
- Request: as above, plus `PriorityScore`
- Per-instance: `QueueDepth`, `BatchSize`, `KVUtilization`, `CacheHitRate`, `RecentTTFT`, `RecentTPOT`, `EstimatedWaitTime`
- Global: aggregate metrics across instances

**Instance Scheduler Observables:**
- Local only: `WaitQueue`, `RunningBatch`, `KVCacheState`
- Request metadata: `PriorityScore`, `ArrivalTime`, `InputTokens`

**AutoScale Policy Observables:**
- Deployment: `CurrentReplicas`, `MinReplicas`, `MaxReplicas`
- Aggregate metrics: `AvgQueueDepth`, `AvgUtilization`, `RequestRate`, `SLOAttainment`
- History: `TimeSinceLastScale`, recent load samples

**Non-observables (internal state):**
- Exact KV block placement (only utilization % is visible)
- Other tenants' request contents
- Future arrivals
- Internal scheduler decisions on other instances

### D.8 Failure Mode Detection

The research agenda identifies specific failure modes that policies should suppress. BLIS detects these as **anomalies** in the decision trace.

| Failure Mode | Detection Method | Metric |
|--------------|------------------|--------|
| **Priority inversion** | Higher-priority request scheduled after lower-priority | `PriorityInversionCount` |
| **Head-of-line blocking** | Request waits while lower-priority requests in same queue complete | `HOLBlockingEvents` |
| **Scale oscillation** | UP followed by DOWN (or vice versa) within cooldown window | `ScaleOscillationCount` |
| **KV thrashing** | Offload immediately followed by reload for same block | `KVThrashingRate` |

These are detected during simulation and reported in `RawMetrics` and `DecisionTrace`.

---

## E) Determinism Guarantees

Evolutionary optimization requires bit-for-bit reproducible simulations. This section specifies how BLIS achieves determinism.

### E.1 Sources of Non-Determinism and Mitigations

| Source | Risk | Mitigation |
|--------|------|------------|
| **Go map iteration** | Non-deterministic order | Use slices with explicit sorting, or `OrderedMap` |
| **Goroutines** | Scheduling non-determinism | Single-threaded event loop; no goroutines in hot path |
| **`time.Now()`** | Wall-clock dependency | Forbidden in simulation; use simulated `Clock` only |
| **Floating-point ordering** | Accumulated error differences | Use `int64` ticks for time; careful operation ordering |
| **External I/O** | File system, network | Policies are pure functions; no I/O during simulation |
| **Uninitialized memory** | Random values | Go zero-initializes; explicit initialization in structs |

### E.2 Partitioned RNG Design

```go
// Each subsystem gets a deterministically-derived RNG
type PartitionedRNG struct {
    masterSeed int64
    subsystems map[string]*rand.Rand // lazily created
}

func (p *PartitionedRNG) ForSubsystem(name string) *rand.Rand {
    if rng, ok := p.subsystems[name]; ok {
        return rng
    }
    // Derive seed deterministically: masterSeed XOR hash(name)
    derivedSeed := p.masterSeed ^ int64(fnv1a(name))
    p.subsystems[name] = rand.New(rand.NewSource(derivedSeed))
    return p.subsystems[name]
}
```

**Subsystems:** `"workload"`, `"router"`, `"instance_0"`, `"instance_1"`, ...

**Isolation guarantee:** Changing draws in one subsystem does not affect others.

### E.3 Event Ordering Rules

```go
// Events are ordered by: (1) timestamp, (2) type priority, (3) event ID
func (a Event) Less(b Event) bool {
    if a.Timestamp() != b.Timestamp() {
        return a.Timestamp() < b.Timestamp()
    }
    if a.TypePriority() != b.TypePriority() {
        return a.TypePriority() < b.TypePriority()
    }
    return a.ID() < b.ID() // deterministic tie-breaker
}

// Type priorities (lower = processed first)
const (
    PriorityArrival    = 1
    PriorityAdmission  = 2
    PriorityRouting    = 3
    PriorityStep       = 4
    PriorityCompletion = 5
    PriorityScaleCheck = 6
)
```

### E.4 Tie-Breaking Rules

| Situation | Rule |
|-----------|------|
| Routing: equal scores | Select instance with lowest `InstanceID` (lexicographic) |
| Scheduling: equal priority | FIFO by `Request.ArrivalTime`, then `Request.ID` |
| Eviction: equal LRU time | Evict block with lowest `BlockID` |
| Preemption: equal candidates | Preempt request with highest `Request.ID` (LIFO) |

### E.5 Determinism Verification

```bash
# CI test: run 100 times, verify identical output
for i in {1..100}; do
  ./simulation_worker run --seed 42 --results-path /tmp/run_$i.json
done
md5sum /tmp/run_*.json | awk '{print $1}' | sort -u | wc -l
# Expected output: 1 (all identical)
```

---

## F) Policy Expressiveness

Without Starlark/WASM, users configure policies through **parameterized templates**.

### F.1 Policy Configuration Model

```yaml
# policies.yaml
admission:
  type: "token-bucket"
  params:
    bucket_size: 1000
    refill_rate: 100  # tokens/sec
    per_tenant: true

priority:
  type: "slo-based"
  params:
    realtime_score: 100
    batch_score: 10
    default_score: 50

routing:
  type: "weighted-scoring"
  params:
    cache_affinity_weight: 0.6
    load_balance_weight: 0.3
    queue_depth_weight: 0.1
    prefix_match_bonus: 0.5

scheduler:
  type: "priority-fcfs"
  params:
    respect_priority: true
    preemption_enabled: true
    preemption_policy: "lowest-priority"

autoscale:
  type: "threshold"
  params:
    scale_up_threshold: 0.8    # utilization
    scale_down_threshold: 0.3
    cooldown_seconds: 60
```

### F.2 Built-in Policy Templates

| Policy Type | Templates Available |
|-------------|---------------------|
| **Admission** | `always-admit`, `token-bucket`, `rate-limit`, `tenant-quota` |
| **Priority** | `constant`, `slo-based`, `tenant-priority`, `deadline-aware` |
| **Routing** | `round-robin`, `least-loaded`, `weighted-scoring`, `prefix-affinity` |
| **Scheduler** | `fcfs`, `priority-fcfs`, `sjf` (shortest job first) |
| **AutoScale** | `threshold`, `target-utilization`, `queue-depth` |

### F.3 Evolutionary Search Space

Frameworks like OpenEvolve explore the parameter space:

```python
# OpenEvolve candidate representation
candidate = {
    "routing.cache_affinity_weight": 0.7,
    "routing.load_balance_weight": 0.2,
    "routing.queue_depth_weight": 0.1,
    "admission.bucket_size": 500,
    "autoscale.scale_up_threshold": 0.75,
    # ... etc
}
```

BLIS evaluates candidates by loading parameters into built-in templates.

### F.4 Extensibility Path

For policies that can't be expressed as parameter combinations:
1. **Add new template** to BLIS (requires PR)
2. **External policy service** — BLIS calls out to user-provided HTTP endpoint (future)

---

## G) Concrete Interface Definitions

### G.1 Core Policy Interfaces

```go
// === Admission ===

type AdmissionDecision struct {
    Action AdmissionAction // ADMIT, REJECT, DELAY
    Delay  time.Duration   // if Action == DELAY
    Reason string          // for tracing
}

type AdmissionAction string

const (
    Admit  AdmissionAction = "admit"
    Reject AdmissionAction = "reject"
    Delay  AdmissionAction = "delay"
)

type AdmissionPolicy interface {
    Decide(req *Request, state *RouterState, clock int64) AdmissionDecision
}

// === Priority ===

type PriorityDecision struct {
    Score float64
    Hints map[string]any // optional scheduling hints
}

type PriorityPolicy interface {
    Compute(req *Request, state *RouterState, clock int64) PriorityDecision
}

// === Routing ===

type RoutingDecision struct {
    // For monolithic architecture
    TargetInstance InstanceID

    // For disaggregated P/D
    PrefillInstance InstanceID
    DecodeInstance  InstanceID

    // Observability
    Reason     string
    Scores     map[InstanceID]float64
    Candidates []CandidateScore // top-k for counterfactual
}

type CandidateScore struct {
    Instance  InstanceID
    Score     float64
    Breakdown map[string]float64
}

type RoutingPolicy interface {
    Route(req *Request, priority PriorityDecision, snapshots []InstanceSnapshot,
          state *RouterState, clock int64) RoutingDecision
}

// === Instance Scheduler ===

type BatchDecision struct {
    ToSchedule       []*Request
    ChunkSizes       map[RequestID]int // optional prefill chunking
    PreemptionVictim *Request          // nil if no preemption needed
}

type SchedulerContext struct {
    WaitQueue      []*Request
    RunningBatch   []*Request
    KVCache        *KVCacheSnapshot
    Clock          int64
    MaxBatchSize   int
    MaxTokenBudget int
}

type InstanceScheduler interface {
    MakeBatch(ctx SchedulerContext) BatchDecision
    OnRequestArrival(req *Request, ctx SchedulerContext)
}

// === Auto-Scale ===

type ScaleDecision struct {
    Action         ScaleAction // NONE, UP, DOWN
    TargetReplicas int
    Reason         string
}

type ScaleAction string

const (
    ScaleNone ScaleAction = "none"
    ScaleUp   ScaleAction = "up"
    ScaleDown ScaleAction = "down"
)

type AutoScaleContext struct {
    ConfigID           string
    CurrentReplicas    int
    MinReplicas        int
    MaxReplicas        int
    Metrics            DeploymentMetrics
    Clock              int64
    TimeSinceLastScale time.Duration
}

type AutoScalePolicy interface {
    Evaluate(ctx AutoScaleContext) ScaleDecision
}
```

### G.2 State Structures

```go
type RouterState struct {
    PerTenant  map[TenantID]*TenantState
    Global     GlobalMetrics
    Instances  map[InstanceID]*InstanceSnapshot
    Clock      int64
}

type TenantState struct {
    RequestCount   int
    ActiveRequests int
    RecentRate     float64 // requests/sec over sliding window
    SLOClass       string
}

type InstanceSnapshot struct {
    ID                InstanceID
    PoolType          PoolType // MONOLITHIC, PREFILL, DECODE
    QueueDepth        int
    BatchSize         int
    KVUtilization     float64
    InFlightRequests  int
    RecentTTFT        float64
    RecentTPOT        float64
    CacheHitRate      float64 // recent cache hit rate (observable)
    EstimatedWaitTime float64
}
```

### G.3 Evaluation Result

```go
type EvaluationResult struct {
    // Fitness scores for evolutionary optimization
    Fitness map[string]float64

    // Raw metrics for analysis
    Metrics *RawMetrics

    // Decision trace (verbosity-controlled)
    Trace *SimulationTrace

    // Summary for LLM reflection
    Summary *TraceSummary

    // Metadata
    PolicyID    string
    WorkloadID  string
    SimDuration int64
    WallTime    time.Duration
}

type RawMetrics struct {
    // Latency distributions (per SLO class)
    TTFT map[string]*Distribution
    TPOT map[string]*Distribution
    E2E  map[string]*Distribution

    // Throughput
    RequestsPerSec float64
    TokensPerSec   float64

    // SLO attainment
    SLOAttainment map[string]float64 // class -> fraction meeting target

    // Fairness
    JainFairnessIndex float64

    // Efficiency
    CacheHitRate    map[KVTier]float64
    PreemptionRate  float64
    OffloadRate     float64

    // Scale events and cost
    ScaleUpCount       int
    ScaleDownCount     int
    TotalReplicaSeconds float64 // for cost modeling
    ScaleOscillations  int      // UP→DOWN or DOWN→UP within cooldown

    // Failure mode detection
    PriorityInversions int     // higher-priority scheduled after lower
    HOLBlockingEvents  int     // head-of-line blocking detected
    KVThrashingRate    float64 // offload-then-reload rate
}
```

---

## H) Architectural Evolution

### Before → After

```
CURRENT                              TARGET
───────                              ──────
┌─────────────┐                      ┌────────────────────────────────────────┐
│  Simulator  │                      │          ClusterSimulator              │
│  (single)   │                      │                                        │
│             │                      │  PolicyBundle ────────────────────┐    │
│ - WaitQueue │                      │  RouterState                      │    │
│ - KVCache   │      ────────►       │  EventHeap (ordered)              │    │
│ - Batch     │                      │  PartitionedRNG                   │    │
│ - EventQ    │                      │  Clock (int64)                    │    │
│ - Clock     │                      │                                   │    │
│ - RNG       │                      │  ┌──────────┐  ┌──────────┐      │    │
└─────────────┘                      │  │Instance 0│  │Instance 1│ ...  │    │
                                     │  │-WaitQ    │  │-WaitQ    │      │    │
                                     │  │-KVCache  │  │-KVCache  │      │    │
                                     │  │-Scheduler│  │-Scheduler│◄─────┘    │
                                     │  └──────────┘  └──────────┘           │
                                     └────────────────────────────────────────┘
```

### Package Structure

```
sim/
├── simulator.go          # Existing (minimal changes)
├── request.go            # Existing (extended with TenantID, Priority)
├── event.go              # Existing (kept for single-instance compat)
├── kvcache.go            # Existing (extended with Tier field)
├── rng.go                # NEW: PartitionedRNG
├── cluster/
│   ├── cluster.go        # ClusterSimulator
│   ├── instance.go       # InstanceSimulator wrapper
│   ├── event.go          # Cluster event types
│   ├── deployment.go     # DeploymentConfig, ReplicaPool
│   ├── router_state.go   # RouterState, TenantState
│   └── metrics.go        # ClusterMetrics, RawMetrics
├── policy/
│   ├── admission.go      # AdmissionPolicy interface + templates
│   ├── priority.go       # PriorityPolicy interface + templates
│   ├── routing.go        # RoutingPolicy interface + templates
│   ├── scheduler.go      # InstanceScheduler interface + templates
│   ├── autoscale.go      # AutoScalePolicy interface + templates
│   └── bundle.go         # PolicyBundle, loading from YAML
├── kv/
│   ├── tiered.go         # TieredKVCache
│   └── transfer.go       # Offload/reload, P/D transfer
├── workload/
│   ├── spec.go           # WorkloadSpec, TenantSpec
│   ├── generator.go      # Workload generation
│   └── arrival.go        # ArrivalPattern implementations
├── trace/
│   ├── trace.go          # SimulationTrace, DecisionTrace
│   ├── record.go         # RoutingRecord, ScaleRecord, etc.
│   └── summary.go        # TraceSummary, summarization
└── adapter/
    ├── gepa.go           # BLISGEPAAdapter
    └── openevolve.go     # BLISEvaluator
```

---

## I) PR Plan (Restructured)

### Phase 1: Foundation (Sequential)

These PRs must be done in order—each depends on the previous.

---

#### PR 1: PartitionedRNG and SimulationKey

| Aspect | Details |
|--------|---------|
| **Title** | `feat(sim): Add PartitionedRNG for deterministic multi-subsystem simulation` |
| **Motivation** | Determinism requires isolated RNG per subsystem |
| **In Scope** | `PartitionedRNG`, `SimulationKey`, refactor `Simulator.randomNumberGenerator` |
| **Out of Scope** | Multi-replica, policies |
| **Files Changed** | New: `sim/rng.go` (~100 LOC). Modified: `sim/simulator.go` (~20 LOC) |
| **CLI** | `./simulation_worker run --model X --seed 42` (unchanged, proves compatibility) |
| **Tests** | Unit: subsystem isolation. Integration: determinism verification |
| **No Dead Code** | `PartitionedRNG` immediately used by `Simulator` |
| **LOC Estimate** | ~120 |

---

#### PR 2: InstanceSimulator Wrapper

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add InstanceSimulator wrapper` |
| **Motivation** | Composable unit for multi-replica |
| **In Scope** | `sim/cluster/` package, `InstanceSimulator`, `InstanceID` type |
| **Out of Scope** | `ClusterSimulator`, `DeploymentConfig`, policies |
| **Files Changed** | New: `sim/cluster/instance.go` (~150 LOC) |
| **CLI** | Same as PR 1 (internal refactor, external behavior identical) |
| **Tests** | `InstanceSimulator.Step()` produces identical results to `Simulator.Step()` |
| **No Dead Code** | Single-instance mode runs through wrapper |
| **LOC Estimate** | ~150 |

---

#### PR 3: DeploymentConfig and ReplicaPool

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add DeploymentConfig and ReplicaPool` |
| **Motivation** | Model deployment topology |
| **In Scope** | `DeploymentConfig`, `ReplicaPool`, `ArchitectureType`, `PoolType` |
| **Out of Scope** | `ClusterSimulator`, P/D disaggregation |
| **Files Changed** | New: `sim/cluster/deployment.go` (~100 LOC) |
| **CLI** | Config loading only; not yet exposed via flags |
| **Tests** | Unit: config validation, serialization |
| **No Dead Code** | Used by `ClusterSimulator` in PR 4 |
| **LOC Estimate** | ~100 |

---

#### PR 4: ClusterSimulator with Event Loop

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add ClusterSimulator with multi-instance event loop` |
| **Motivation** | Run N instances with shared clock |
| **In Scope** | `ClusterSimulator`, `EventHeap` with ordering, `--num-instances` flag, basic round-robin dispatch |
| **Out of Scope** | Policy interfaces (temporary hardcoded dispatch) |
| **Files Changed** | New: `sim/cluster/cluster.go` (~300 LOC), `sim/cluster/event.go` (~150 LOC). Modified: `cmd/root.go` |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --rate 20` |
| **Tests** | `--num-instances 1` identical to original; deterministic replay with N>1 |
| **No Dead Code** | `--num-instances` flag exercises all paths |
| **LOC Estimate** | ~500 |

---

### Phase 2: Policy Interfaces (Parallelizable)

After PR 4, these PRs can be developed **in parallel**.

---

#### PR 5: AdmissionPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add AdmissionPolicy with AlwaysAdmit and TokenBucket` |
| **In Scope** | `AdmissionPolicy` interface, `AlwaysAdmit`, `TokenBucket` templates |
| **Files Changed** | New: `sim/policy/admission.go` (~150 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --admission-policy token-bucket --admission-bucket-size 100` |
| **Parallel With** | PR 6, PR 7, PR 8 |
| **LOC Estimate** | ~150 |

---

#### PR 6: PriorityPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add PriorityPolicy with Constant and SLOBased` |
| **In Scope** | `PriorityPolicy` interface, `ConstantPriority`, `SLOBasedPriority` templates |
| **Files Changed** | New: `sim/policy/priority.go` (~120 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --priority-policy slo-based` |
| **Parallel With** | PR 5, PR 7, PR 8 |
| **LOC Estimate** | ~120 |

---

#### PR 7: RoutingPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RoutingPolicy with RoundRobin and WeightedScoring` |
| **In Scope** | `RoutingPolicy` interface, `InstanceSnapshot`, `RoundRobin`, `WeightedScoring` templates |
| **Files Changed** | New: `sim/policy/routing.go` (~200 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --routing-policy weighted --routing-cache-weight 0.6` |
| **Parallel With** | PR 5, PR 6, PR 8 |
| **LOC Estimate** | ~200 |

---

#### PR 8: InstanceScheduler Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add InstanceScheduler with FCFS and PriorityFCFS` |
| **In Scope** | `InstanceScheduler` interface, `SchedulerContext`, `FCFSScheduler`, `PriorityFCFSScheduler` |
| **Files Changed** | New: `sim/policy/scheduler.go` (~180 LOC). Modified: `sim/cluster/instance.go` (delegate to scheduler) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --scheduler priority-fcfs` |
| **Parallel With** | PR 5, PR 6, PR 7 |
| **LOC Estimate** | ~200 |

---

#### PR 9: RouterState and PolicyBundle

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RouterState and PolicyBundle configuration` |
| **Depends On** | PR 5, PR 6, PR 7, PR 8 |
| **In Scope** | `RouterState`, `TenantState`, `GlobalMetrics`, `PolicyBundle`, YAML loading |
| **Files Changed** | New: `sim/cluster/router_state.go` (~150 LOC), `sim/policy/bundle.go` (~100 LOC) |
| **CLI** | `./simulation_worker run --model X --policy-config policies.yaml` |
| **LOC Estimate** | ~250 |

---

### Phase 3: KV Cache & Workload (Parallelizable)

After PR 9, these PRs can proceed **in parallel**.

---

#### PR 10: KVTier Types and Configuration

| Aspect | Details |
|--------|---------|
| **Title** | `feat(kv): Add KVTier types and multi-tier configuration` |
| **In Scope** | `KVTier` enum, `KVTierConfig`, extend `KVBlock` with `Tier` field |
| **Files Changed** | New: `sim/kv/tiered.go` (~100 LOC). Modified: `sim/kvcache.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --kv-gpu-blocks 10000 --kv-cpu-blocks 50000` |
| **Parallel With** | PR 12, PR 13 |
| **LOC Estimate** | ~130 |

---

#### PR 11: KV Offload/Reload Mechanics

| Aspect | Details |
|--------|---------|
| **Title** | `feat(kv): Add offload/reload transfer mechanics` |
| **Depends On** | PR 10 |
| **In Scope** | `KVTransfer` event, offload trigger, reload on CPU hit, transfer latency |
| **Files Changed** | New: `sim/kv/transfer.go` (~200 LOC). Modified: `sim/cluster/event.go` |
| **CLI** | `./simulation_worker run --model X --kv-gpu-blocks 1000 --kv-cpu-blocks 10000 --kv-offload-threshold 0.9` |
| **LOC Estimate** | ~220 |

---

#### PR 12: WorkloadSpec and TenantSpec

| Aspect | Details |
|--------|---------|
| **Title** | `feat(workload): Add WorkloadSpec and TenantSpec` |
| **In Scope** | `WorkloadSpec`, `TenantSpec`, `SLOSpec`, `PrefixSpec`, `ArrivalPattern` types |
| **Files Changed** | New: `sim/workload/spec.go` (~150 LOC) |
| **CLI** | Config types only; generator in PR 13 |
| **Parallel With** | PR 10, PR 13 |
| **LOC Estimate** | ~150 |

---

#### PR 13: Workload Generator

| Aspect | Details |
|--------|---------|
| **Title** | `feat(workload): Add multi-tenant workload generator` |
| **Depends On** | PR 12 |
| **In Scope** | `WorkloadGenerator`, Poisson/bursty arrival, prefix reuse, `--workload-spec` flag |
| **Files Changed** | New: `sim/workload/generator.go` (~200 LOC), `sim/workload/arrival.go` (~100 LOC) |
| **CLI** | `./simulation_worker run --model X --workload-spec workload.yaml` |
| **LOC Estimate** | ~300 |

---

#### PR 14: RawMetrics and Fitness Evaluation

| Aspect | Details |
|--------|---------|
| **Title** | `feat(metrics): Add RawMetrics, SLO attainment, and FitnessFunction` |
| **In Scope** | `RawMetrics`, `Distribution`, `FitnessFunction`, `EvaluationResult` |
| **Files Changed** | New: `sim/cluster/metrics.go` (~200 LOC) |
| **CLI** | `./simulation_worker run --model X --workload-spec w.yaml --fitness-weights "throughput:0.5,p99_ttft:0.3"` |
| **Parallel With** | PR 10, PR 12 |
| **LOC Estimate** | ~200 |

---

### Phase 4: Advanced Features (Parallelizable)

After Phase 3, these three tracks can proceed **in parallel**.

---

**Track A: Auto-Scaling**

#### PR 15: AutoScaler Core

| Aspect | Details |
|--------|---------|
| **Title** | `feat(autoscaler): Add AutoScaler with ThresholdScaler` |
| **In Scope** | `AutoScaler`, `AutoScalePolicy`, `AutoScaleContext`, `ThresholdScaler` |
| **Files Changed** | New: `sim/policy/autoscale.go` (~200 LOC). Modified: `sim/cluster/cluster.go` |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --autoscaler-enabled --autoscaler-max 8` |
| **Parallel With** | PR 17, PR 19 |
| **LOC Estimate** | ~220 |

---

#### PR 16: Scaling Actuation Model

| Aspect | Details |
|--------|---------|
| **Title** | `feat(autoscaler): Add provisioning delays, warmup, and drain` |
| **Depends On** | PR 15 |
| **In Scope** | `WarmupProfile`, `DrainPolicy`, `InstanceState` lifecycle |
| **Files Changed** | Modified: `sim/policy/autoscale.go` (~150 LOC), `sim/cluster/instance.go` (~50 LOC) |
| **CLI** | `./simulation_worker run --model X --autoscaler-enabled --provisioning-delay 30s --warmup-duration 60s` |
| **LOC Estimate** | ~200 |

---

**Track B: P/D Disaggregation**

#### PR 17: P/D Architecture

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add disaggregated prefill-decode architecture` |
| **In Scope** | `DISAGGREGATED_PD` type, `PrefillPool`, `DecodePool`, `PDHandoffEvent`, routing changes |
| **Files Changed** | Modified: `sim/cluster/deployment.go` (~50 LOC), `sim/cluster/cluster.go` (~150 LOC), `sim/cluster/event.go` (~100 LOC) |
| **CLI** | `./simulation_worker run --model X --architecture pd --prefill-replicas 2 --decode-replicas 4` |
| **Parallel With** | PR 15, PR 19 |
| **LOC Estimate** | ~300 |

---

#### PR 18: KV Transfer for P/D

| Aspect | Details |
|--------|---------|
| **Title** | `feat(pd): Add KV transfer with ownership tracking` |
| **Depends On** | PR 17, PR 11 |
| **In Scope** | `PDTransferConfig`, `BlockTransferState`, ownership transfer |
| **Files Changed** | Modified: `sim/kv/transfer.go` (~150 LOC) |
| **CLI** | `./simulation_worker run --model X --architecture pd --pd-transfer-latency 1ms --pd-transfer-bandwidth 10GB/s` |
| **LOC Estimate** | ~150 |

---

**Track C: Observability**

#### PR 19: Decision Traces

| Aspect | Details |
|--------|---------|
| **Title** | `feat(trace): Add DecisionTrace with RoutingRecord` |
| **In Scope** | `SimulationTrace`, `DecisionTrace`, `RoutingRecord`, `TraceConfig`, `--trace-level` flag |
| **Files Changed** | New: `sim/trace/trace.go` (~100 LOC), `sim/trace/record.go` (~150 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --trace-level decisions` |
| **Parallel With** | PR 15, PR 17 |
| **LOC Estimate** | ~250 |

---

#### PR 20: Counterfactual Analysis and Trace Summary

| Aspect | Details |
|--------|---------|
| **Title** | `feat(trace): Add counterfactual analysis and trace summarization` |
| **Depends On** | PR 19 |
| **In Scope** | `TopKCandidates`, `Regret` calculation, `TraceSummary`, `--summarize-trace` flag |
| **Files Changed** | New: `sim/trace/summary.go` (~200 LOC). Modified: `sim/trace/record.go` (~50 LOC) |
| **CLI** | `./simulation_worker run --model X --trace-level decisions --counterfactual-k 5 --summarize-trace` |
| **LOC Estimate** | ~250 |

---

### Phase 5: Framework Adapters (Optional)

BLIS is fully functional without these. They provide convenience for framework integration.

---

#### PR 21: GEPA Adapter

| Aspect | Details |
|--------|---------|
| **Title** | `feat(adapter): Add GEPA adapter` |
| **In Scope** | `BLISGEPAAdapter`, `Evaluate()`, `ExtractTracesForReflection()`, `gepa-evaluate` command |
| **Files Changed** | New: `sim/adapter/gepa.go` (~150 LOC). Modified: `cmd/root.go` |
| **CLI** | `./simulation_worker gepa-evaluate --policy-config p.yaml --workload w.yaml` |
| **Parallel With** | PR 22 |
| **LOC Estimate** | ~180 |

---

#### PR 22: OpenEvolve Evaluator

| Aspect | Details |
|--------|---------|
| **Title** | `feat(adapter): Add OpenEvolve evaluator` |
| **In Scope** | `BLISEvaluator`, multi-objective fitness, feature extraction, `openevolve-evaluate` command |
| **Files Changed** | New: `sim/adapter/openevolve.go` (~150 LOC). Modified: `cmd/root.go` |
| **CLI** | `./simulation_worker openevolve-evaluate --config oe.yaml --candidate c.yaml` |
| **Parallel With** | PR 21 |
| **LOC Estimate** | ~180 |

---

### Phase 6: Validation

#### PR 23: Integration Tests and Examples

| Aspect | Details |
|--------|---------|
| **Title** | `test: Add comprehensive integration test suite and examples` |
| **In Scope** | Integration tests, sample configs, example policies, CI validation |
| **Files Changed** | New: `test/integration/` (~500 LOC), `examples/` (configs) |
| **CLI** | `go test ./test/integration/...` |
| **LOC Estimate** | ~500 |

---

## J) Dependency DAG

### PR Dependency Graph

```
PHASE 1: FOUNDATION (Sequential)
════════════════════════════════════════════════════════════════════════════════

  PR 1 ──────► PR 2 ──────► PR 3 ──────► PR 4
  (RNG)      (Instance)   (Deploy)    (Cluster)
                                          │
                                          ▼
PHASE 2: POLICY INTERFACES ═════════════════════════════════════════════════════
                                          │
              ┌───────────┬───────────┬───┴───────┐
              ▼           ▼           ▼           ▼
            PR 5        PR 6        PR 7        PR 8
          (Admit)    (Priority)   (Route)    (Sched)
              │           │           │           │
              └───────────┴─────┬─────┴───────────┘
                                ▼
                              PR 9
                           (Bundle)
                                │
                                ▼
PHASE 3: KV + WORKLOAD ═════════════════════════════════════════════════════════
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
         PR 10                PR 12                PR 14
       (KV Tier)           (Workload)           (Metrics)
           │                    │
           ▼                    ▼
         PR 11                PR 13
       (Transfer)          (Generator)
           │                    │                    │
           └────────────────────┴────────────────────┘
                                │
                                ▼
PHASE 4: ADVANCED FEATURES ═════════════════════════════════════════════════════
                                │
     ┌──────────────────────────┼──────────────────────────┐
     ▼                          ▼                          ▼
   PR 15                      PR 17                      PR 19
 (AutoScale)                  (P/D)                    (Traces)
     │                          │                          │
     ▼                          ▼                          ▼
   PR 16                      PR 18                      PR 20
 (Actuation)               (Transfer)                (Summary)
                                │
                                ▼
PHASE 5: ADAPTERS (Optional) ═══════════════════════════════════════════════════
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                  PR 21                   PR 22
                 (GEPA)               (OpenEvolve)
                    │                       │
                    └───────────┬───────────┘
                                ▼
PHASE 6: VALIDATION ════════════════════════════════════════════════════════════
                                │
                                ▼
                              PR 23
                            (Tests)
```

### Parallel Development Matrix

| Gate | Completed PRs | Unlocked for Parallel Development |
|------|---------------|-----------------------------------|
| **G1** | PR 4 | PR 5, PR 6, PR 7, PR 8 (4 parallel) |
| **G2** | PR 9 | PR 10+11, PR 12+13, PR 14 (3 parallel tracks) |
| **G3** | PR 11, PR 13, PR 14 | PR 15+16, PR 17+18, PR 19+20 (3 parallel tracks) |
| **G4** | PR 20 | PR 21, PR 22 (2 parallel) |

### Critical Merge Gates

| Gate | Verification | Failure Action |
|------|--------------|----------------|
| **After PR 4** | Determinism: 100 runs identical | Block until fixed |
| **After PR 9** | Policy interfaces frozen | No interface changes in later PRs |
| **After PR 14** | Fitness evaluation works E2E | Required for Phase 4 |
| **After PR 20** | All observability features working | Required for adapters |

### Timeline Estimate (3-4 developers)

```
Week 1-2:   Phase 1 (PR 1-4, sequential, 1 dev)

Week 3:     Phase 2 (PR 5-8, 4 devs parallel)
Week 4:     Phase 2 (PR 9, integrates PR 5-8)

Week 5-6:   Phase 3 (PR 10-14, 3 tracks parallel)

Week 7-9:   Phase 4 (PR 15-21, 3 tracks parallel)

Week 10:    Phase 5 (PR 22-23, 2 devs parallel)

Week 11:    Phase 6 (PR 24)

Total: ~11 weeks with 3-4 developers
```

---

## K) Validation Strategy

### K.1 Unit Test Coverage

| Component | Test Focus |
|-----------|------------|
| `PartitionedRNG` | Subsystem isolation, determinism |
| Policy interfaces | Default implementations, edge cases |
| `KVCacheState` | Conservation invariant, LRU ordering |
| `EventHeap` | Ordering rules, tie-breaking |
| Workload generator | Distribution correctness, prefix reuse |

### K.2 Integration Tests

| Test | Description |
|------|-------------|
| `TestSingleInstanceCompatibility` | `--num-instances 1` produces identical results to original |
| `TestDeterministicReplay` | 100 runs with same seed produce identical output |
| `TestPolicyPipeline` | Admission → Priority → Routing → Scheduling flow |
| `TestKVTierTransfer` | Offload/reload timing and conservation |
| `TestPDHandoff` | Prefill → Transfer → Decode lifecycle |
| `TestAutoScaling` | Scale up/down with warmup and drain |

### K.3 Behavioral Validation

| Invariant | Verification Method |
|-----------|---------------------|
| `request_lifecycle` | Track all requests; assert exactly one terminal state |
| `clock_monotonicity` | Assert `new_clock >= old_clock` after every event |
| `kv_conservation` | Assert `used + free = total` after every KV operation |
| `scale_bounds` | Assert `min <= current <= max` after every scale action |
| `determinism` | Diff outputs of multiple runs |

### K.4 Sim-to-Real Validation Framework

The research agenda treats sim-to-real transfer as a first-class question. BLIS supports this through structured validation:

**Validation Protocol:**
1. **Policy discovery** — Evolve policies in BLIS simulation
2. **Policy export** — Serialize top-k policies as configuration
3. **Real deployment** — Deploy policies in llm-d with vLLM backends
4. **Metrics collection** — Collect same metrics as BLIS (TTFT, TPOT, SLO attainment, etc.)
5. **Transfer analysis** — Compare sim vs. real rankings and performance gaps

**Transfer Quality Metrics:**
| Metric | Definition |
|--------|------------|
| **Ranking consistency** | Spearman correlation between sim and real policy rankings |
| **Absolute gap** | |sim_metric - real_metric| for each policy |
| **Relative ordering preservation** | % of pairwise comparisons where sim correctly predicts winner |

**BLIS Output for Validation:**
```json
{
  "policy_id": "routing_v42",
  "sim_fitness": {"throughput": 0.85, "p99_ttft": 0.72},
  "sim_metrics": { ... },
  "config_for_deployment": { ... }  // directly loadable by llm-d
}
```

**Coefficient Refinement Loop:**
When sim-to-real gap exceeds threshold:
1. Collect detailed traces from real deployment
2. Identify divergence sources (latency model, cache behavior, etc.)
3. Refine alpha/beta coefficients using real data
4. Re-validate transfer quality

---

## L) Design Bug Prevention Checklist

### L.1 Invariants

| Invariant | Enforcement |
|-----------|-------------|
| `request_lifecycle` | Assert on simulation end: all requests terminated |
| `clock_monotonicity` | Assert in event loop: `new >= old` |
| `kv_conservation` | Assert after every `Allocate`/`Release`: `used + free = total` |
| `scale_bounds` | Assert in `executeScaleAction`: `min <= target <= max` |
| `pd_ownership` | Assert: block has exactly one owner at any time |
| `determinism` | CI: run twice, diff outputs |

### L.2 Regression Surfaces

| Surface | Risk | Mitigation |
|---------|------|------------|
| Event ordering | Non-determinism | Explicit type priorities, event IDs, no map iteration |
| RNG isolation | Cross-contamination | Hash-based derivation, isolation tests |
| KV reference counting | Leaks | Conservation check after every op |
| Policy interfaces | Breaking changes | Interface freeze after PR 9 |
| Floating-point | Non-determinism | Use int64 for time; careful ordering |

### L.3 Failure Mode Prevention

| Failure | Prevention |
|---------|------------|
| Non-deterministic ties | Explicit rules documented in code |
| Scale oscillation | Cooldown period, hysteresis in `ThresholdScaler` |
| P/D deadlock | Transfer timeout, backpressure threshold |
| KV thrashing | Offload rate metric, alert if high |

---

## M) Performance Expectations

BLIS is a CPU-only discrete-event simulator. Performance should be fast enough for interactive use and batch evolutionary evaluation.

### Target Performance

| Workload | Target | Notes |
|----------|--------|-------|
| 1K requests, 1 instance | < 100ms | Interactive CLI use |
| 10K requests, 4 instances | < 1 second | Batch evaluation |
| 100K requests, 16 instances | < 10 seconds | Large-scale simulation |

### Performance Principles

1. **No allocation in hot path** — Pre-allocate event structs, reuse slices
2. **Efficient event heap** — Standard library `container/heap` is sufficient
3. **Avoid map iteration** — Use slices for ordered iteration
4. **Profile before optimizing** — Measure actual bottlenecks

### Benchmark Requirements

Each PR affecting the hot path must include benchmarks:

```go
func BenchmarkClusterSimulator_10K(b *testing.B) {
    // Setup: 10K requests, 4 instances
    for i := 0; i < b.N; i++ {
        sim.Run()
    }
}
```

**CI gate:** Performance regression >20% blocks merge.

---

## N) Summary

| Metric | Value |
|--------|-------|
| **Phases** | 6 |
| **Total PRs** | 23 |
| **Total LOC Estimate** | ~4,800 |
| **Max Parallel PRs** | 4 (Phase 2) |
| **Estimated Weeks** | 10-11 (with 3-4 developers) |

### Key Decisions

1. **BLIS is standalone** — framework adapters are optional conveniences
2. **No arbitrary code execution** — parameterized policy templates instead
3. **Determinism is foundational** — explicit rules for all non-deterministic sources
4. **vLLM-grounded modeling** — explicit citations and simplifications documented
5. **Routing policy freedom** — policies can maintain their own internal state; router provides observables, not mandated tracking
6. **Phases restructured** — based on dependency analysis, not design doc chapters

### Research Agenda Alignment

This plan directly supports the four llm-d inference control problems:

| Research Problem | BLIS Support | Key PRs |
|-----------------|--------------|---------|
| **Routing Policy Evolution** | `RoutingPolicy` interface, `InstanceSnapshot` observables, prefix-aware workloads, KV cache modeling | PR 7, 10-11, 12-13 |
| **Admission Control Evolution** | `AdmissionPolicy` interface, multi-tenant `TenantState`, fairness metrics, bursty arrival patterns | PR 5, 9, 12-14 |
| **Joint Priority Scheduling** | `PriorityPolicy` + `InstanceScheduler` separation, priority inversion detection, HOL blocking detection | PR 6, 8, 14 |
| **Autoscaling Evolution** | `AutoScalePolicy` interface, provisioning/warmup modeling, cost metrics, scale oscillation detection | PR 15-16, 14 |

**Architectural Locality:** Each policy interface respects control boundaries—routing sees only `RouterState` + `InstanceSnapshot`, instance schedulers see only local state, autoscalers see only aggregate metrics.

**Sim-to-Real Transfer:** BLIS outputs directly deployable policy configurations, with structured validation protocol (Section K.4) to measure transfer quality.

### References

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
2. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024
3. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024
4. Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," ICML 2023

### Next Steps

1. Review and approve this plan
2. Create Phase 1 micro-level implementation plan (PR 1-4)
3. Begin PR 1 (PartitionedRNG)
