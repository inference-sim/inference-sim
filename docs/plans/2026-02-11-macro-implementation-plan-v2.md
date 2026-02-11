# BLIS Evolutionary Policy Optimization: Macro-Level Implementation Plan (v2)

**Date:** 2026-02-11
**Revision:** v2.1 (incorporates Perplexity, Gemini, and GPT-4o review feedback)
**Status:** Draft
**Target:** Multi-replica cluster simulation with pluggable policies
**Based on:** [Design Document](2026-02-06-evolutionary-policy-optimization-design.md)

---

## Revision Notes (v2.1)

This revision incorporates feedback from external review (Perplexity, Gemini, GPT-4o):

**v2 changes (Perplexity):**
1. **Earlier research-ready checkpoint** — Metrics/fitness evaluation moved up to enable policy research after Phase 2
2. **Deferred tiered KV** — Single-tier KV sufficient for initial policy research; tiered offload/reload moves to Phase 4
3. **Pathological policy templates** — Added to each policy PR for baseline testing and anomaly detection validation
4. **Mock study checkpoint** — Added after PR 4 to validate interfaces before freeze
5. **Reordered for fastest research loop** — Research-ready in ~5 weeks vs ~6 weeks

**v2.1 changes (Gemini + GPT-4o):**
6. **Interface extension point** — Added `Extended` map to `InstanceSnapshot` for Phase 4+ observables; clarified "freeze" means no breaking changes
7. **Policy lifecycle clarification** — New section F.6 documents that policy instances persist and may maintain internal state
8. **Edge case workloads** — Expanded PR 12-13 scope to include bursty/diurnal arrivals and multi-tenant fairness scenarios
9. **Parallel trace collection** — Added note that real-world trace collection can begin during Phase 2 as separate workstream

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
- 6 phases, 24 PRs
- **Research-ready checkpoint after Phase 2** (~5 weeks)
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

### B.3 Existing Assets We Can Leverage

| Asset | Location | Reuse in Plan |
|-------|----------|---------------|
| Basic workload generation | `sim/workload_config.go` | Sufficient for Phase 2 research |
| Metrics collection | `sim/metrics.go` | Foundation for RawMetrics |
| KV cache with prefix caching | `sim/kvcache.go` | Single-tier sufficient initially |

### B.4 Hardcoded Behaviors Requiring Extraction

| Behavior | Location | Target Interface |
|----------|----------|------------------|
| FIFO batch formation | `simulator.go:341-365` | `InstanceScheduler` |
| LIFO preemption | `simulator.go:248-277` | `InstanceScheduler` |
| Single-tier KV | `kvcache.go` | Tiered `KVCacheState` (Phase 4) |
| All-admit | implicit | `AdmissionPolicy` |

### B.5 Extension Strategy

- **Composition over modification**: `ClusterSimulator` wraps `InstanceSimulator` instances
- **Interface extraction**: Pull hardcoded behaviors into pluggable interfaces
- **Backward compatibility**: `--num-instances 1` produces identical results to current

---

## C) High-Level Objectives + Non-Goals

### Objectives

1. **Multi-replica simulation** with shared clock and coordinated events
2. **Deterministic execution** — same seeds produce bit-for-bit identical results
3. **Pluggable policies** — admission, priority, routing, scheduling, auto-scaling
4. **Rich observability** — decision traces, counterfactual analysis
5. **Research-ready checkpoint** — enable policy experiments early (end of Phase 2)
6. **Tiered KV cache** — GPU + CPU with offload/reload latency modeling (Phase 4)
7. **P/D disaggregation** — separate prefill and decode pools (Phase 4)
8. **Framework adapters** — optional conveniences for GEPA/OpenEvolve

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

**Phased Implementation:**
- **Phase 2:** Single-tier KV (existing `sim/kvcache.go`) — sufficient for routing/scheduling research
- **Phase 4:** Multi-tier with offload/reload — fidelity enhancement

### D.2 Latency Estimation

**Based on:** Empirical coefficients from instrumented vLLM deployments.

| Component | Model | Coefficients |
|-----------|-------|--------------|
| Queueing delay | `α₀ + α₁ * input_len` | Learned from trace data |
| Step time | `β₀ + β₁ * cache_miss_tokens + β₂ * decode_tokens` | Learned from busy-loop instrumentation |
| Roofline (alternative) | FLOPs / peak_throughput | Analytical, for new hardware |

**Limitation:** Coefficients are hardware/model/TP-specific. Generalization requires re-training.

### D.3 Multi-Tier KV Cache (Phase 4)

**Based on:** vLLM CPU offloading and research systems like FlexGen [Sheng et al., 2023].

| Tier | Latency Model | Capacity Model |
|------|---------------|----------------|
| GPU | `access_latency = 0` | `gpu_blocks` parameter |
| CPU | `access_latency = transfer_time(blocks)` | `cpu_blocks` parameter |
| Storage | Deferred | — |

**Transfer model:** `time = base_latency + blocks * block_size / bandwidth`

### D.4 Prefill-Decode Disaggregation (Phase 4)

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
| **Admission** | `always-admit`, `token-bucket`, `rate-limit`, `tenant-quota`, `reject-all`* |
| **Priority** | `constant`, `slo-based`, `tenant-priority`, `deadline-aware`, `inverted-slo`* |
| **Routing** | `round-robin`, `least-loaded`, `weighted-scoring`, `prefix-affinity`, `always-busiest`* |
| **Scheduler** | `fcfs`, `priority-fcfs`, `sjf` (shortest job first), `reverse-priority`* |
| **AutoScale** | `threshold`, `target-utilization`, `queue-depth`, `oscillator`* |

*\* Pathological templates for baseline testing and anomaly detection validation*

### F.3 Pathological Templates (New in v2)

Each policy type includes at least one intentionally bad template to:
1. Validate anomaly detection metrics work correctly
2. Provide baselines for comparing evolved policies
3. Test failure mode detection

| Template | Purpose | Expected Anomalies |
|----------|---------|-------------------|
| `reject-all` | Admission baseline | 100% rejection rate |
| `inverted-slo` | Priority inversion testing | High `PriorityInversionCount` |
| `always-busiest` | Load imbalance testing | High `HOLBlockingEvents`, poor tail latency |
| `reverse-priority` | Scheduler fairness testing | High `PriorityInversionCount` |
| `oscillator` | Scale stability testing | High `ScaleOscillationCount` |

### F.4 Evolutionary Search Space

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

### F.5 Extensibility Path

For policies that can't be expressed as parameter combinations:
1. **Add new template** to BLIS (requires PR)
2. **External policy service** — BLIS calls out to user-provided HTTP endpoint (future)

### F.6 Policy Instance Lifecycle (New in v2.1)

Policy structs are **instantiated once per simulation** from the `PolicyBundle` and **persist for the entire run**. Internal state is allowed and expected for stateful policies.

**Determinism constraint:** State updates must depend only on method inputs (`clock`, `req`, `state`), never on wall-clock time (`time.Now()`) or external I/O.

**Example: TokenBucket with internal state**

```go
type TokenBucketPolicy struct {
    // Configuration (from YAML)
    bucketSize   int
    refillRate   float64  // tokens per tick

    // Internal state (persists across Decide() calls)
    tokens       float64
    lastRefillTs int64
}

func (t *TokenBucketPolicy) Decide(req *Request, state *RouterState, clock int64) AdmissionDecision {
    // Refill tokens based on simulated time elapsed (deterministic)
    elapsed := clock - t.lastRefillTs
    t.tokens = min(float64(t.bucketSize), t.tokens + float64(elapsed) * t.refillRate)
    t.lastRefillTs = clock

    // Consume token if available
    if t.tokens >= 1.0 {
        t.tokens -= 1.0
        return AdmissionDecision{Action: Admit, Reason: "token available"}
    }
    return AdmissionDecision{Action: Reject, Reason: "bucket empty"}
}
```

**Lifecycle guarantees:**
- Policy is instantiated once when `PolicyBundle` is loaded
- Same instance is used for all calls during the simulation
- State is NOT persisted across simulation runs (each run starts fresh)
- State is NOT shared across policy types (each policy has its own instance)

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
    // Core fields (frozen after PR 9 - no removals or type changes)
    ID                InstanceID
    PoolType          PoolType // MONOLITHIC, PREFILL, DECODE
    QueueDepth        int
    BatchSize         int
    KVUtilization     float64  // GPU tier utilization (0.0-1.0)
    InFlightRequests  int
    RecentTTFT        float64
    RecentTPOT        float64
    CacheHitRate      float64  // recent cache hit rate (observable)
    EstimatedWaitTime float64

    // Extension point for Phase 4+ observables (New in v2.1)
    // Policies should handle missing keys gracefully (use default or ignore).
    // Known keys added in Phase 4:
    //   "CPUKVUtilization"   - CPU tier utilization (0.0-1.0)
    //   "TransferQueueDepth" - pending offload/reload operations
    //   "PendingOffloads"    - blocks queued for GPU→CPU transfer
    //   "PendingReloads"     - blocks queued for CPU→GPU transfer
    Extended map[string]float64
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
    CacheHitRate    float64  // single value for Phase 2; per-tier in Phase 4
    PreemptionRate  float64

    // Scale events and cost
    ScaleUpCount       int
    ScaleDownCount     int
    TotalReplicaSeconds float64 // for cost modeling
    ScaleOscillations  int      // UP→DOWN or DOWN→UP within cooldown

    // Failure mode detection
    PriorityInversions int     // higher-priority scheduled after lower
    HOLBlockingEvents  int     // head-of-line blocking detected
}
```

---

## H) Architectural Evolution

### Before → After

```
CURRENT                              TARGET (Phase 2 Research-Ready)
───────                              ──────────────────────────────
┌─────────────┐                      ┌────────────────────────────────────────┐
│  Simulator  │                      │          ClusterSimulator              │
│  (single)   │                      │                                        │
│             │                      │  PolicyBundle ────────────────────┐    │
│ - WaitQueue │                      │  RouterState                      │    │
│ - KVCache   │      ────────►       │  EventHeap (ordered)              │    │
│ - Batch     │                      │  PartitionedRNG                   │    │
│ - EventQ    │                      │  Clock (int64)                    │    │
│ - Clock     │                      │  RawMetrics ◄──── Fitness eval    │    │
│ - RNG       │                      │                                   │    │
└─────────────┘                      │  ┌──────────┐  ┌──────────┐      │    │
                                     │  │Instance 0│  │Instance 1│ ...  │    │
                                     │  │-WaitQ    │  │-WaitQ    │      │    │
                                     │  │-KVCache  │  │-KVCache  │      │    │
                                     │  │-Scheduler│  │-Scheduler│◄─────┘    │
                                     │  └──────────┘  └──────────┘           │
                                     └────────────────────────────────────────┘

TARGET (Phase 4 Full Fidelity)
──────────────────────────────
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ClusterSimulator                                   │
│                                                                             │
│  PolicyBundle ─────────────────────────────────────────────────────────┐    │
│  RouterState                                                           │    │
│  EventHeap (ordered)                                                   │    │
│  PartitionedRNG                                                        │    │
│  AutoScaler ◄──── Scale decisions                                      │    │
│  DecisionTrace ◄──── Counterfactual analysis                           │    │
│                                                                        │    │
│  ┌─────────────────────────┐    ┌─────────────────────────┐           │    │
│  │     Prefill Pool        │    │      Decode Pool        │           │    │
│  │  ┌────────┐ ┌────────┐  │    │  ┌────────┐ ┌────────┐  │           │    │
│  │  │Inst P0 │ │Inst P1 │  │───►│  │Inst D0 │ │Inst D1 │  │           │    │
│  │  │TieredKV│ │TieredKV│  │ KV │  │TieredKV│ │TieredKV│  │           │    │
│  │  └────────┘ └────────┘  │xfer│  └────────┘ └────────┘  │           │    │
│  └─────────────────────────┘    └─────────────────────────┘           │    │
│                                                                        │    │
└────────────────────────────────────────────────────────────────────────┘    │
```

### Package Structure

```
sim/
├── simulator.go          # Existing (minimal changes)
├── request.go            # Existing (extended with TenantID, Priority)
├── event.go              # Existing (kept for single-instance compat)
├── kvcache.go            # Existing (single-tier, sufficient for Phase 2)
├── metrics.go            # Existing (foundation for RawMetrics)
├── rng.go                # NEW: PartitionedRNG
├── cluster/
│   ├── cluster.go        # ClusterSimulator
│   ├── instance.go       # InstanceSimulator wrapper
│   ├── event.go          # Cluster event types
│   ├── deployment.go     # DeploymentConfig, ReplicaPool
│   ├── router_state.go   # RouterState, TenantState
│   └── metrics.go        # ClusterMetrics, RawMetrics, FitnessFunction
├── policy/
│   ├── admission.go      # AdmissionPolicy interface + templates
│   ├── priority.go       # PriorityPolicy interface + templates
│   ├── routing.go        # RoutingPolicy interface + templates
│   ├── scheduler.go      # InstanceScheduler interface + templates
│   ├── autoscale.go      # AutoScalePolicy interface + templates
│   └── bundle.go         # PolicyBundle, loading from YAML
├── kv/                   # Phase 4
│   ├── tiered.go         # TieredKVCache
│   └── transfer.go       # Offload/reload, P/D transfer
├── workload/             # Phase 3 (optional enhancement)
│   ├── spec.go           # WorkloadSpec, TenantSpec
│   ├── generator.go      # Workload generation
│   └── arrival.go        # ArrivalPattern implementations
├── trace/                # Phase 4
│   ├── trace.go          # SimulationTrace, DecisionTrace
│   ├── record.go         # RoutingRecord, ScaleRecord, etc.
│   └── summary.go        # TraceSummary, summarization
└── adapter/              # Phase 5
    ├── gepa.go           # BLISGEPAAdapter
    └── openevolve.go     # BLISEvaluator
```

---

## I) PR Plan (Restructured for Research-First)

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
| **Benchmark** | `BenchmarkClusterSimulator_10K` added here |
| **No Dead Code** | `--num-instances` flag exercises all paths |
| **LOC Estimate** | ~500 |

**⚠️ CHECKPOINT: Mock Study**

After PR 4 merges, before starting Phase 2:
1. Write 2-3 hand-coded policies directly in `cluster_test.go`
2. Exercise against simple workloads (existing `--workload distribution`)
3. Document any missing observables or awkward patterns
4. Adjust interface designs in PR 5-8 based on findings

---

### Phase 2: Policy Interfaces + Metrics (Parallelizable after PR 4)

After PR 4 and mock study, these PRs can be developed **in parallel**.

---

#### PR 5: AdmissionPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add AdmissionPolicy with AlwaysAdmit, TokenBucket, and RejectAll` |
| **In Scope** | `AdmissionPolicy` interface, `AlwaysAdmit`, `TokenBucket`, `RejectAll`* templates |
| **Files Changed** | New: `sim/policy/admission.go` (~180 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --admission-policy token-bucket --admission-bucket-size 100` |
| **Parallel With** | PR 6, PR 7, PR 8, PR 10 |
| **LOC Estimate** | ~180 |

*\* Pathological template*

---

#### PR 6: PriorityPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add PriorityPolicy with Constant, SLOBased, and InvertedSLO` |
| **In Scope** | `PriorityPolicy` interface, `ConstantPriority`, `SLOBasedPriority`, `InvertedSLOPriority`* templates |
| **Files Changed** | New: `sim/policy/priority.go` (~150 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --priority-policy slo-based` |
| **Parallel With** | PR 5, PR 7, PR 8, PR 10 |
| **LOC Estimate** | ~150 |

*\* Pathological template*

---

#### PR 7: RoutingPolicy Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add RoutingPolicy with RoundRobin, WeightedScoring, and AlwaysBusiest` |
| **In Scope** | `RoutingPolicy` interface, `InstanceSnapshot`, `RoundRobin`, `WeightedScoring`, `AlwaysBusiest`* templates |
| **Files Changed** | New: `sim/policy/routing.go` (~230 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --routing-policy weighted --routing-cache-weight 0.6` |
| **Parallel With** | PR 5, PR 6, PR 8, PR 10 |
| **LOC Estimate** | ~230 |

*\* Pathological template*

---

#### PR 8: InstanceScheduler Interface

| Aspect | Details |
|--------|---------|
| **Title** | `feat(policy): Add InstanceScheduler with FCFS, PriorityFCFS, and ReversePriority` |
| **In Scope** | `InstanceScheduler` interface, `SchedulerContext`, `FCFSScheduler`, `PriorityFCFSScheduler`, `ReversePriorityScheduler`* |
| **Files Changed** | New: `sim/policy/scheduler.go` (~210 LOC). Modified: `sim/cluster/instance.go` (delegate to scheduler) |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --scheduler priority-fcfs` |
| **Parallel With** | PR 5, PR 6, PR 7, PR 10 |
| **LOC Estimate** | ~230 |

*\* Pathological template*

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

**⚠️ INTERFACE FREEZE after PR 9** — Policy interfaces are stable. "Freeze" means no breaking changes (no field removals, no type changes, no method signature changes). Adding new fields to structs or new keys to `Extended` maps is permitted in later phases.

---

#### PR 10: RawMetrics, Fitness Evaluation, and Anomaly Detection

| Aspect | Details |
|--------|---------|
| **Title** | `feat(metrics): Add RawMetrics, FitnessFunction, and anomaly detection` |
| **Parallel With** | PR 5, PR 6, PR 7, PR 8 (can start before PR 9) |
| **In Scope** | `RawMetrics`, `Distribution`, `FitnessFunction`, `EvaluationResult`, anomaly counters (`PriorityInversions`, `HOLBlockingEvents`, `ScaleOscillations`) |
| **Files Changed** | New: `sim/cluster/metrics.go` (~250 LOC). Modified: `sim/cluster/cluster.go` (~50 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --fitness-weights "throughput:0.5,p99_ttft:0.3"` |
| **LOC Estimate** | ~300 |

---

#### PR 11: Anomaly Detection Validation

| Aspect | Details |
|--------|---------|
| **Title** | `test(metrics): Add anomaly detection validation with pathological policies` |
| **Depends On** | PR 9, PR 10 |
| **In Scope** | Integration tests running pathological policies, asserting anomaly metrics are non-zero |
| **Files Changed** | New: `sim/cluster/anomaly_test.go` (~200 LOC) |
| **CLI** | `go test ./sim/cluster/... -run TestAnomalyDetection` |
| **LOC Estimate** | ~200 |

---

### 🎯 RESEARCH-READY CHECKPOINT (After PR 11)

At this point (~5 weeks), BLIS supports:
- ✅ Multi-instance cluster simulation
- ✅ All 4 policy interfaces (admission, priority, routing, scheduler)
- ✅ PolicyBundle with YAML configuration
- ✅ RawMetrics with fitness evaluation
- ✅ Anomaly detection validated with pathological baselines
- ✅ Existing workload generation (`--workload distribution`)
- ✅ Single-tier KV cache (existing `sim/kvcache.go`)

**You can begin policy research experiments here.**

---

### Phase 3: Enhanced Workloads (Optional, Parallel)

These PRs improve workload fidelity but are not required for initial research.

---

#### PR 12: WorkloadSpec and TenantSpec

| Aspect | Details |
|--------|---------|
| **Title** | `feat(workload): Add WorkloadSpec and TenantSpec` |
| **In Scope** | `WorkloadSpec`, `TenantSpec`, `SLOSpec`, `PrefixSpec`, `ArrivalPattern` types |
| **Files Changed** | New: `sim/workload/spec.go` (~180 LOC) |
| **CLI** | Config types only; generator in PR 13 |
| **Parallel With** | PR 14, PR 15, PR 17 |
| **LOC Estimate** | ~180 |

**Arrival patterns defined (v2.1 expansion):**
- `Poisson` — constant rate λ
- `Bursty` — Poisson with periodic 10x spikes (configurable spike interval and duration)
- `Diurnal` — sinusoidal rate variation (models day/night traffic patterns)

---

#### PR 13: Workload Generator

| Aspect | Details |
|--------|---------|
| **Title** | `feat(workload): Add multi-tenant workload generator with edge case scenarios` |
| **Depends On** | PR 12 |
| **In Scope** | `WorkloadGenerator`, all arrival patterns, prefix reuse, `--workload-spec` flag, edge case scenarios |
| **Files Changed** | New: `sim/workload/generator.go` (~250 LOC), `sim/workload/arrival.go` (~150 LOC), `sim/workload/scenarios.go` (~100 LOC) |
| **CLI** | `./simulation_worker run --model X --workload-spec workload.yaml` |
| **LOC Estimate** | ~500 |

**Built-in edge case scenarios (v2.1 expansion):**

| Scenario | Description | Tests |
|----------|-------------|-------|
| `bursty-traffic` | Poisson baseline with 10x spikes every 60s | Admission policy resilience |
| `diurnal-cycle` | 24-hour sinusoidal rate (10x peak-to-trough) | Autoscaler response |
| `unfair-tenants` | 90% low-priority, 10% high-priority requests | Multi-tenant fairness, priority inversion |
| `prefix-heavy` | 80% requests share common prefixes | Cache-aware routing effectiveness |
| `mixed-slo` | Equal mix of realtime/interactive/batch | Joint priority scheduling |

**Example scenario config:**
```yaml
scenario: "unfair-tenants"
tenants:
  - id: "low-priority-bulk"
    weight: 0.9
    slo_class: "batch"
    rate_fraction: 0.9
  - id: "high-priority-realtime"
    weight: 0.1
    slo_class: "realtime"
    rate_fraction: 0.1
```

---

### Phase 4: Advanced Features (Parallelizable)

After Research-Ready checkpoint, these three tracks can proceed **in parallel**.

---

**Track A: Auto-Scaling**

#### PR 14: AutoScaler Core

| Aspect | Details |
|--------|---------|
| **Title** | `feat(autoscaler): Add AutoScaler with ThresholdScaler and Oscillator` |
| **In Scope** | `AutoScaler`, `AutoScalePolicy`, `AutoScaleContext`, `ThresholdScaler`, `Oscillator`* |
| **Files Changed** | New: `sim/policy/autoscale.go` (~230 LOC). Modified: `sim/cluster/cluster.go` |
| **CLI** | `./simulation_worker run --model X --num-instances 2 --autoscaler-enabled --autoscaler-max 8` |
| **Parallel With** | PR 16, PR 18 |
| **LOC Estimate** | ~250 |

*\* Pathological template*

---

#### PR 15: Scaling Actuation Model

| Aspect | Details |
|--------|---------|
| **Title** | `feat(autoscaler): Add provisioning delays, warmup, and drain` |
| **Depends On** | PR 14 |
| **In Scope** | `WarmupProfile`, `DrainPolicy`, `InstanceState` lifecycle |
| **Files Changed** | Modified: `sim/policy/autoscale.go` (~150 LOC), `sim/cluster/instance.go` (~50 LOC) |
| **CLI** | `./simulation_worker run --model X --autoscaler-enabled --provisioning-delay 30s --warmup-duration 60s` |
| **LOC Estimate** | ~200 |

---

**Track B: Tiered KV + P/D Disaggregation**

#### PR 16: KVTier Types and Configuration

| Aspect | Details |
|--------|---------|
| **Title** | `feat(kv): Add KVTier types and multi-tier configuration` |
| **In Scope** | `KVTier` enum, `KVTierConfig`, extend `KVBlock` with `Tier` field |
| **Files Changed** | New: `sim/kv/tiered.go` (~100 LOC). Modified: `sim/kvcache.go` (~30 LOC) |
| **CLI** | `./simulation_worker run --model X --kv-gpu-blocks 10000 --kv-cpu-blocks 50000` |
| **Parallel With** | PR 14, PR 18 |
| **LOC Estimate** | ~130 |

---

#### PR 17: KV Offload/Reload Mechanics

| Aspect | Details |
|--------|---------|
| **Title** | `feat(kv): Add offload/reload transfer mechanics` |
| **Depends On** | PR 16 |
| **In Scope** | `KVTransfer` event, offload trigger, reload on CPU hit, transfer latency, `KVThrashingRate` metric |
| **Files Changed** | New: `sim/kv/transfer.go` (~200 LOC). Modified: `sim/cluster/event.go`, `sim/cluster/metrics.go` |
| **CLI** | `./simulation_worker run --model X --kv-gpu-blocks 1000 --kv-cpu-blocks 10000 --kv-offload-threshold 0.9` |
| **LOC Estimate** | ~220 |

---

#### PR 18: P/D Architecture

| Aspect | Details |
|--------|---------|
| **Title** | `feat(cluster): Add disaggregated prefill-decode architecture` |
| **Depends On** | PR 17 |
| **In Scope** | `DISAGGREGATED_PD` type, `PrefillPool`, `DecodePool`, `PDHandoffEvent`, routing changes |
| **Files Changed** | Modified: `sim/cluster/deployment.go` (~50 LOC), `sim/cluster/cluster.go` (~150 LOC), `sim/cluster/event.go` (~100 LOC) |
| **CLI** | `./simulation_worker run --model X --architecture pd --prefill-replicas 2 --decode-replicas 4` |
| **Parallel With** | PR 14, PR 16 |
| **LOC Estimate** | ~300 |

---

#### PR 19: KV Transfer for P/D

| Aspect | Details |
|--------|---------|
| **Title** | `feat(pd): Add KV transfer with ownership tracking` |
| **Depends On** | PR 18 |
| **In Scope** | `PDTransferConfig`, `BlockTransferState`, ownership transfer |
| **Files Changed** | Modified: `sim/kv/transfer.go` (~150 LOC) |
| **CLI** | `./simulation_worker run --model X --architecture pd --pd-transfer-latency 1ms --pd-transfer-bandwidth 10GB/s` |
| **LOC Estimate** | ~150 |

---

**Track C: Observability**

#### PR 20: Decision Traces

| Aspect | Details |
|--------|---------|
| **Title** | `feat(trace): Add DecisionTrace with RoutingRecord` |
| **In Scope** | `SimulationTrace`, `DecisionTrace`, `RoutingRecord`, `TraceConfig`, `--trace-level` flag |
| **Files Changed** | New: `sim/trace/trace.go` (~100 LOC), `sim/trace/record.go` (~150 LOC) |
| **CLI** | `./simulation_worker run --model X --num-instances 4 --trace-level decisions` |
| **Parallel With** | PR 14, PR 16 |
| **LOC Estimate** | ~250 |

---

#### PR 21: Counterfactual Analysis and Trace Summary

| Aspect | Details |
|--------|---------|
| **Title** | `feat(trace): Add counterfactual analysis and trace summarization` |
| **Depends On** | PR 20 |
| **In Scope** | `TopKCandidates`, `Regret` calculation, `TraceSummary`, `--summarize-trace` flag |
| **Files Changed** | New: `sim/trace/summary.go` (~200 LOC). Modified: `sim/trace/record.go` (~50 LOC) |
| **CLI** | `./simulation_worker run --model X --trace-level decisions --counterfactual-k 5 --summarize-trace` |
| **LOC Estimate** | ~250 |

---

### Phase 5: Framework Adapters (Optional)

BLIS is fully functional without these. They provide convenience for framework integration.

---

#### PR 22: GEPA Adapter

| Aspect | Details |
|--------|---------|
| **Title** | `feat(adapter): Add GEPA adapter` |
| **In Scope** | `BLISGEPAAdapter`, `Evaluate()`, `ExtractTracesForReflection()`, `gepa-evaluate` command |
| **Files Changed** | New: `sim/adapter/gepa.go` (~150 LOC). Modified: `cmd/root.go` |
| **CLI** | `./simulation_worker gepa-evaluate --policy-config p.yaml --workload w.yaml` |
| **Parallel With** | PR 23 |
| **LOC Estimate** | ~180 |

---

#### PR 23: OpenEvolve Evaluator

| Aspect | Details |
|--------|---------|
| **Title** | `feat(adapter): Add OpenEvolve evaluator` |
| **In Scope** | `BLISEvaluator`, multi-objective fitness, feature extraction, `openevolve-evaluate` command |
| **Files Changed** | New: `sim/adapter/openevolve.go` (~150 LOC). Modified: `cmd/root.go` |
| **CLI** | `./simulation_worker openevolve-evaluate --config oe.yaml --candidate c.yaml` |
| **Parallel With** | PR 22 |
| **LOC Estimate** | ~180 |

---

### Phase 6: Validation

#### PR 24: Integration Tests and Examples

| Aspect | Details |
|--------|---------|
| **Title** | `test: Add comprehensive integration test suite and examples` |
| **In Scope** | Integration tests, sample configs, example policies, CI validation |
| **Files Changed** | New: `test/integration/` (~500 LOC), `examples/` (configs) |
| **CLI** | `go test ./test/integration/...` |
| **LOC Estimate** | ~500 |

---

## J) Dependency DAG (Restructured)

### PR Dependency Graph

```
PHASE 1: FOUNDATION (Sequential)
════════════════════════════════════════════════════════════════════════════════

  PR 1 ──────► PR 2 ──────► PR 3 ──────► PR 4
  (RNG)      (Instance)   (Deploy)    (Cluster)
                                          │
                                          ▼
                              ⚠️ MOCK STUDY CHECKPOINT
                                          │
                                          ▼
PHASE 2: POLICY INTERFACES + METRICS ══════════════════════════════════════════
                                          │
              ┌───────────┬───────────┬───┴───────┬───────────┐
              ▼           ▼           ▼           ▼           ▼
            PR 5        PR 6        PR 7        PR 8        PR 10
          (Admit)    (Priority)   (Route)    (Sched)    (Metrics)
              │           │           │           │           │
              └───────────┴─────┬─────┴───────────┘           │
                                ▼                             │
                              PR 9 ◄──────────────────────────┘
                           (Bundle)
                                │
                                ▼
                    ⚠️ INTERFACE FREEZE
                                │
                                ▼
                             PR 11
                      (Anomaly Validation)
                                │
                                ▼
                    🎯 RESEARCH-READY CHECKPOINT
                                │
                                ▼
PHASE 3: ENHANCED WORKLOADS (Optional) ════════════════════════════════════════
                                │
                    ┌───────────┴───────────┐
                    ▼                       │
                  PR 12                     │
               (Workload)                   │
                    │                       │
                    ▼                       │
                  PR 13                     │
               (Generator)                  │
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
PHASE 4: ADVANCED FEATURES ════════════════════════════════════════════════════
                                │
     ┌──────────────────────────┼──────────────────────────┐
     ▼                          ▼                          ▼
   PR 14                      PR 16                      PR 20
 (AutoScale)                (KV Tier)                  (Traces)
     │                          │                          │
     ▼                          ▼                          ▼
   PR 15                      PR 17                      PR 21
 (Actuation)               (Transfer)                 (Summary)
                                │
                                ▼
                              PR 18
                              (P/D)
                                │
                                ▼
                              PR 19
                           (P/D Xfer)
                                │
                                ▼
PHASE 5: ADAPTERS (Optional) ══════════════════════════════════════════════════
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                  PR 22                   PR 23
                 (GEPA)               (OpenEvolve)
                    │                       │
                    └───────────┬───────────┘
                                ▼
PHASE 6: VALIDATION ═══════════════════════════════════════════════════════════
                                │
                                ▼
                              PR 24
                            (Tests)
```

### Parallel Development Matrix

| Gate | Completed PRs | Unlocked for Parallel Development |
|------|---------------|-----------------------------------|
| **G1** | PR 4 + Mock Study | PR 5, PR 6, PR 7, PR 8, PR 10 (5 parallel) |
| **G2** | PR 11 (Research-Ready) | PR 12, PR 14, PR 16, PR 20 (4 parallel tracks) |
| **G3** | PR 15, PR 19, PR 21 | PR 22, PR 23 (2 parallel) |

### Critical Checkpoints

| Checkpoint | Verification | Failure Action |
|------------|--------------|----------------|
| **Mock Study (after PR 4)** | Hand-coded policies work, no missing observables | Adjust interfaces before Phase 2 |
| **After PR 4** | Determinism: 100 runs identical | Block until fixed |
| **Interface Freeze (after PR 9)** | Policy interfaces frozen (no breaking changes) | Additive changes (new fields, new `Extended` keys) permitted |
| **Research-Ready (after PR 11)** | Pathological policies trigger anomalies | Required for research |
| **After PR 21** | All observability features working | Required for adapters |

### Timeline Estimate (3-4 developers)

```
Week 1-2:   Phase 1 (PR 1-4, sequential, 1 dev)
            + Mock Study (2-3 days)

Week 3-4:   Phase 2 (PR 5-10, 4-5 devs parallel)
Week 5:     Phase 2 (PR 9, PR 11, integrates)
            → 🎯 RESEARCH-READY

Week 6-7:   Phase 3 (PR 12-13, optional, 1 dev)
            Phase 4 Track A (PR 14-15, 1 dev)
            Phase 4 Track B (PR 16-19, 1 dev)
            Phase 4 Track C (PR 20-21, 1 dev)

Week 8-9:   Phase 4 continued

Week 10:    Phase 5 (PR 22-23, 2 devs parallel)

Week 11:    Phase 6 (PR 24)

Total: ~11 weeks with 3-4 developers
Research-ready: ~5 weeks
```

---

## K) Validation Strategy

### K.1 Unit Test Coverage

| Component | Test Focus |
|-----------|------------|
| `PartitionedRNG` | Subsystem isolation, determinism |
| Policy interfaces | Default implementations, edge cases |
| Pathological policies | Known-bad behavior produces expected anomalies |
| `KVCacheState` | Conservation invariant, LRU ordering |
| `EventHeap` | Ordering rules, tie-breaking |
| Workload generator | Distribution correctness, prefix reuse |

### K.2 Integration Tests

| Test | Description | Phase |
|------|-------------|-------|
| `TestSingleInstanceCompatibility` | `--num-instances 1` produces identical results to original | PR 4 |
| `TestDeterministicReplay` | 100 runs with same seed produce identical output | PR 4 |
| `TestPolicyPipeline` | Admission → Priority → Routing → Scheduling flow | PR 9 |
| `TestAnomalyDetection_PathologicalPolicies` | Pathological policies trigger expected anomalies | PR 11 |
| `TestKVTierTransfer` | Offload/reload timing and conservation | PR 17 |
| `TestPDHandoff` | Prefill → Transfer → Decode lifecycle | PR 19 |
| `TestAutoScaling` | Scale up/down with warmup and drain | PR 15 |

### K.3 Behavioral Validation

| Invariant | Verification Method |
|-----------|---------------------|
| `request_lifecycle` | Track all requests; assert exactly one terminal state |
| `clock_monotonicity` | Assert `new_clock >= old_clock` after every event |
| `kv_conservation` | Assert `used + free = total` after every KV operation |
| `scale_bounds` | Assert `min <= current <= max` after every scale action |
| `determinism` | Diff outputs of multiple runs |
| `anomaly_detection` | Pathological policies produce non-zero anomaly counts |

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

**Sanity Scenario (New in v2):**
Maintain one simple scenario that mirrors a real deployment you can instrument:
- 4 replicas, monolithic architecture
- Poisson arrivals at 80% utilization
- Single SLO class
- Measure: P50/P99 TTFT, throughput, cache hit rate

This scenario is used for:
1. Early coefficient sanity checks
2. Sim-to-real gap baseline
3. Regression detection when models change

**Parallel Trace Collection (New in v2.1):**
Real-world trace collection for coefficient refinement can begin **in parallel with Phase 2** as a separate workstream. This is not a blocking dependency for BLIS development.

Recommended parallel activities:
1. **During Phase 1-2:** Set up instrumented vLLM deployment with trace export
2. **During Phase 3-4:** Collect traces under sanity scenario workload
3. **After Research-Ready:** Use traces to refine alpha/beta coefficients

This parallelization allows sim-to-real validation to begin shortly after the research-ready checkpoint without delaying BLIS core development.

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
| `anomaly_detection` | Pathological policies produce non-zero counts |

### L.2 Regression Surfaces

| Surface | Risk | Mitigation |
|---------|------|------------|
| Event ordering | Non-determinism | Explicit type priorities, event IDs, no map iteration |
| RNG isolation | Cross-contamination | Hash-based derivation, isolation tests |
| KV reference counting | Leaks | Conservation check after every op |
| Policy interfaces | Breaking changes | Interface freeze after PR 9 |
| Floating-point | Non-determinism | Use int64 for time; careful ordering |
| Anomaly detection | False negatives | Pathological policy tests |

### L.3 Failure Mode Prevention

| Failure | Prevention |
|---------|------------|
| Non-deterministic ties | Explicit rules documented in code |
| Scale oscillation | Cooldown period, hysteresis in `ThresholdScaler` |
| P/D deadlock | Transfer timeout, backpressure threshold |
| KV thrashing | Offload rate metric, alert if high |
| Missing anomalies | Pathological policy test suite |

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

Benchmarks added in PR 4 (ClusterSimulator):

```go
func BenchmarkClusterSimulator_1K_1Instance(b *testing.B) {
    // Setup: 1K requests, 1 instance
    for i := 0; i < b.N; i++ {
        sim.Run()
    }
}

func BenchmarkClusterSimulator_10K_4Instances(b *testing.B) {
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
| **Total PRs** | 24 |
| **Total LOC Estimate** | ~5,400 |
| **Max Parallel PRs** | 5 (Phase 2) |
| **Research-Ready** | ~5 weeks |
| **Full Implementation** | ~11 weeks (with 3-4 developers) |

### Key Decisions

1. **BLIS is standalone** — framework adapters are optional conveniences
2. **No arbitrary code execution** — parameterized policy templates instead
3. **Determinism is foundational** — explicit rules for all non-deterministic sources
4. **vLLM-grounded modeling** — explicit citations and simplifications documented
5. **Routing policy freedom** — policies can maintain their own internal state; router provides observables, not mandated tracking
6. **Research-first ordering** — metrics and anomaly detection moved up; tiered KV deferred
7. **Pathological baselines** — every policy type has intentionally bad templates for testing
8. **Mock study before freeze** — validate interfaces with real experiments
9. **Interface extension point** — `Extended` map allows Phase 4 observables without breaking freeze (v2.1)
10. **Stateful policies supported** — policy instances persist; internal state is allowed and expected (v2.1)

### Changes from v1

| Change | Rationale |
|--------|-----------|
| PR 10 (Metrics) moved to Phase 2 | Enables research after Phase 2 |
| PR 11 (Anomaly Validation) added | Validates pathological templates work |
| Tiered KV (PR 16-17) moved to Phase 4 | Single-tier sufficient for initial research |
| P/D (PR 18-19) depends on tiered KV | Natural dependency |
| Pathological templates added to PR 5-8 | Baseline testing, anomaly validation |
| Mock study checkpoint added | De-risk interface freeze |
| Benchmarks added in PR 4 | Early performance visibility |

### Changes from v2 (v2.1)

| Change | Rationale |
|--------|-----------|
| `Extended` map added to `InstanceSnapshot` | Phase 4 observables without breaking interface freeze |
| Section F.6 (Policy Lifecycle) added | Clarifies stateful policies are supported |
| PR 12-13 scope expanded | Edge case workloads (bursty, diurnal, unfair tenants) |
| Parallel trace collection note added | Enables sim-to-real work during Phase 2 |
| Interface freeze clarified | "No breaking changes" vs "no changes at all" |

### Research Agenda Alignment

This plan directly supports the four llm-d inference control problems:

| Research Problem | BLIS Support | Key PRs |
|-----------------|--------------|---------|
| **Routing Policy Evolution** | `RoutingPolicy` interface, `InstanceSnapshot` observables, existing prefix caching | PR 7, PR 10-11 |
| **Admission Control Evolution** | `AdmissionPolicy` interface, multi-tenant `TenantState`, fairness metrics | PR 5, PR 9-11 |
| **Joint Priority Scheduling** | `PriorityPolicy` + `InstanceScheduler` separation, priority inversion detection, HOL blocking detection | PR 6, PR 8, PR 10-11 |
| **Autoscaling Evolution** | `AutoScalePolicy` interface, provisioning/warmup modeling, cost metrics, scale oscillation detection | PR 14-15 |

**Architectural Locality:** Each policy interface respects control boundaries—routing sees only `RouterState` + `InstanceSnapshot`, instance schedulers see only local state, autoscalers see only aggregate metrics.

**Sim-to-Real Transfer:** BLIS outputs directly deployable policy configurations, with structured validation protocol (Section K.4) to measure transfer quality.

### References

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
2. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024
3. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024
4. Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," ICML 2023

### Next Steps

1. Review and approve this plan (v2.1)
2. Create Phase 1 micro-level implementation plan (PR 1-4)
3. Begin PR 1 (PartitionedRNG)
4. (Parallel) Set up instrumented vLLM deployment for trace collection
