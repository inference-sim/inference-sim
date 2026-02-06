# BLIS Extensions for Evolutionary Policy Optimization

**Date:** 2026-02-06
**Status:** Draft
**Branch:** paperplanning

## Overview

This design extends BLIS from a single-replica LLM inference simulator to a multi-replica cluster simulator with pluggable control policies. The goal is to enable evolutionary optimization frameworks (OpenEvolve, GEPA) to discover routing, admission, and scheduling policies through simulation-based fitness evaluation.

### Design Scope

**In scope:**
- Integration layer between BLIS and evolutionary frameworks
- Policy representation (parameterized, code, decision trees)
- BLIS extensions for multi-replica simulation and pluggable policies

**Out of scope (for now):**
- Sim-to-real transfer validation
- Production deployment of evolved policies

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
┌─────────────────────────────────────────────────────────────┐
│                    Evolutionary Framework                    │
│              (OpenEvolve / GEPA / custom)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Policy    │  │   Fitness   │  │  Trace Collector    │  │
│  │  Candidates │  │  Evaluator  │  │  (for reflection)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ Policy + Workload
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   BLIS Cluster Simulator                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            Cluster Control Plane (Router)              │ │
│  │   Admission → Priority/Scheduling → Routing            │ │
│  └────────────────────────────┬───────────────────────────┘ │
│                               │                              │
│         ┌─────────────────────┼─────────────────────┐       │
│         ▼                     ▼                     ▼       │
│    ┌─────────┐           ┌─────────┐           ┌─────────┐  │
│    │Replica 0│           │Replica 1│           │Replica N│  │
│    │Scheduler│           │Scheduler│           │Scheduler│  │
│    └─────────┘           └─────────┘           └─────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    Metrics + Traces
```

### Design Principles

- **Separation of concerns**: Cluster control plane and instance schedulers are distinct, pluggable components
- **Framework-agnostic**: BLIS exposes an evaluation API that any evolutionary framework can call
- **Policy-as-input**: Policies are loaded at simulation start, not compiled into BLIS

---

## 2. Cluster Control Plane (Router Layer)

The cluster control plane models the llm-d router as a pipeline of three pluggable policy stages.

```
Request Arrival
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                   AdmissionPolicy                           │
│  Input:  request, router_state, replica_states[]            │
│  Output: ADMIT | REJECT | DELAY(duration)                   │
└─────────────────────────────┬───────────────────────────────┘
      │ (admitted requests)
      ▼
┌─────────────────────────────────────────────────────────────┐
│                   PriorityPolicy                            │
│  Input:  request, router_state, replica_states[]            │
│  Output: priority_score (float), scheduling_hints (map)     │
└─────────────────────────────┬───────────────────────────────┘
      │ (prioritized requests)
      ▼
┌─────────────────────────────────────────────────────────────┐
│                   RoutingPolicy                             │
│  Input:  request, priority_score, router_state,             │
│          replica_states[]                                   │
│  Output: replica_id                                         │
└─────────────────────────────┴───────────────────────────────┘
      │
      ▼
  Dispatch to Replica
```

### Router State

The router maintains state computed from request stream and decisions:

```go
RouterState {
    // Computed from request stream + decisions
    per_tenant:      map[TenantID] → {request_count, active, recent_rate}
    global:          {in_flight, throughput, request_rate}
    historical:      {latency_windows, admission_counts}

    // Shadow KV model per replica (approximate)
    replica_kv_shadow: map[ReplicaID] → {
        prefix_hashes: set[Hash]         // estimated cached prefixes
        estimated_utilization: float     // approximate KV fill level
        eviction_queue: LRU[Hash]        // predicted eviction order
    }
}
```

The router maintains an approximate shadow model of each replica's KV state based on:
- Requests it has routed to each replica
- Prefix content/hashes of those requests
- Knowledge of KV eviction policies (LRU, capacity limits)

This enables prefix-aware routing without real-time queries. The actual replica state (pulled via interface) reconciles drift periodically.

### Replica State (Observable by Router)

```go
ReplicaSnapshot {
    QueueDepth:     int
    BatchSize:      int
    KVUtilization:  float64    // percentage
    InFlightCount:  int
    RecentTTFT:     float64
    RecentTPOT:     float64
    CacheHitRate:   float64
}
```

---

## 3. Instance-Level Scheduler Interface

Each replica runs a BLIS simulator with a pluggable scheduler controlling batch formation, preemption, and request ordering.

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

## 4. Workload Generation

Extended workload system supporting multi-tenant, priority-aware, prefix-structured, and bursty traffic.

```go
type WorkloadSpec struct {
    Tenants       []TenantSpec
    GlobalArrival ArrivalPattern
    Duration      float64
}

type TenantSpec struct {
    TenantID       string
    Weight         float64            // fraction of total traffic
    PriorityClass  string             // e.g., "realtime", "batch", "best-effort"
    SLO            SLOSpec
    TokenDist      TokenDistribution
    PrefixSpec     PrefixSpec
    ArrivalPattern ArrivalPattern     // tenant-specific modulation
}

type SLOSpec struct {
    TTFTTarget    float64            // ms
    TPOTTarget    float64            // ms per token
    E2ETarget     float64            // ms, optional
}

type PrefixSpec struct {
    PrefixGroups     []PrefixGroup
    PrefixLength     Distribution
    ReuseProb        float64
}

type PrefixGroup struct {
    GroupID      string
    SystemPrompt string             // actual prefix content (or hash)
    Popularity   float64
}

type ArrivalPattern struct {
    Type    string                  // "poisson", "bursty", "periodic", "trace", "adversarial"
    Params  map[string]any
}
```

### Arrival Pattern Types

| Pattern | Description |
|---------|-------------|
| `poisson` | Standard memoryless arrivals |
| `bursty` | Pareto-distributed inter-arrival times |
| `periodic` | Regular intervals with jitter |
| `trace` | Replay from recorded timestamps |
| `adversarial` | Stress-test patterns (long prompts, synchronized bursts) |

---

## 5. Policy Representation

Three policy representation types, all loaded dynamically at simulation start.

```go
type PolicyBundle struct {
    AdmissionPolicy   PolicySpec
    PriorityPolicy    PolicySpec
    RoutingPolicy     PolicySpec
    InstanceScheduler PolicySpec

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

LLM-generated Go code implementing the policy interface.

```go
type CodePolicy struct {
    Source     string
    Entrypoint string
    Imports    []string    // restricted allowlist
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

## 6. Fitness Evaluation

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
    CacheHitRate     float64
    KVUtilization    []float64
    PreemptionRate   float64
    AdmissionRate    float64

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

## 7. Trace Collection

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
}

type RoutingRecord struct {
    Timestamp     float64
    RequestID     string
    TenantID      string
    PriorityClass string
    InputTokens   int
    PrefixHash    string
    ReplicaStates []ReplicaSnapshot
    RouterState   RouterSnapshot
    ChosenReplica int
    Reason        string
    ActualTTFT    float64
    CacheHit      bool
}

type Anomaly struct {
    Type        string
    Timestamp   float64
    Description string
    Requests    []string
    Severity    float64
}
```

### Trace Verbosity Levels

| Level | Collected | Use Case |
|-------|-----------|----------|
| `minimal` | Summary metrics only | Fast fitness evaluation |
| `decisions` | All policy decisions + context | GEPA reflection, debugging |
| `full` | All events + decisions + anomalies | Deep analysis |

---

## 8. Trace Summarization

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

## Key Issues (3 detected)
1. ROUTING: 67% of cache misses came from prefix-unaware decisions
   - 142 requests routed away from replica with matching prefix
   - Pattern: load-balance weight dominated when queue_depth < 5

2. ADMISSION: Bursty tenant "T3" caused 23% of realtime SLO misses
   - 15:42-15:47: T3 burst admitted 340 reqs, realtime TTFT spiked 3.2x

3. SCHEDULING: Priority inversions in replica 2
   - 18 batch requests scheduled before queued realtime requests

## Worst Decisions (3 examples)
1. t=15:43.2 | Routed req_8821 (prefix=0xA3F) → replica 1
   Context: replica 0 had prefix cached, util=72%, queue=3
   Result: cache miss, TTFT=487ms (3x expected)

## Suggested Focus
- Increase cache_affinity_weight when prefix match exists
- Add tenant burst detection to admission policy
```

### Summarization Strategies

| Strategy | Description |
|----------|-------------|
| `issue_focused` | Top N problems with examples |
| `decision_audit` | Sample decisions, annotate outcomes |
| `counterfactual` | "If X decided differently, Y improves" |
| `pattern_mining` | Frequent patterns correlated with poor outcomes |

---

## 9. Multi-Replica Simulation Engine

Cluster simulator orchestrating multiple replica simulators with a shared event loop.

```go
type ClusterSimulator struct {
    NumReplicas     int
    Policy          PolicyBundle
    Workload        WorkloadSpec

    Router          RouterState
    AdmissionPolicy AdmissionPolicy
    PriorityPolicy  PriorityPolicy
    RoutingPolicy   RoutingPolicy

    Replicas        []*ReplicaSimulator
    EventQueue      *EventHeap
    Clock           float64

    TraceConfig     TraceConfig
    Trace           SimulationTrace
}

type ReplicaSimulator struct {
    ID              int
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
type ReplicaStateUpdateEvent struct { ... }

// Replica-level
type ReplicaStepEvent struct { ... }
type RequestCompletedEvent struct { ... }
```

### Simulation Loop

```go
func (c *ClusterSimulator) Run() EvaluationResult {
    c.initialize()
    c.generateArrivals()

    for c.EventQueue.Len() > 0 {
        event := heap.Pop(c.EventQueue)
        c.Clock = event.Timestamp()

        switch e := event.(type) {
        case *RequestArrivalEvent:
            c.handleArrival(e)  // admission → priority → routing
        case *RouteDecisionEvent:
            c.Replicas[e.ReplicaID].EnqueueRequest(e.Request)
        case *ReplicaStepEvent:
            c.Replicas[e.ReplicaID].Step(c.Clock)
        case *ReplicaStateUpdateEvent:
            c.syncReplicaState(e.ReplicaID)
        case *RequestCompletedEvent:
            c.recordCompletion(e)
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

## 10. Behavioral Contracts

Contracts for BDD/TDD development. Full specifications to be refined during detailed design.

### System-Wide Invariants

```
INVARIANT request_lifecycle:
  Every request reaches exactly one terminal state: COMPLETED | REJECTED | TIMED_OUT

INVARIANT causality:
  arrival_time ≤ admission_time ≤ routing_time ≤ enqueue_time ≤ completion_time

INVARIANT clock_monotonicity:
  Simulation clock never decreases

INVARIANT kv_cache_conservation:
  allocated_blocks + free_blocks = total_blocks (per replica)

INVARIANT router_shadow_consistency:
  Shadow KV model updated on every routing decision
```

### Policy Contracts

**AdmissionPolicy:**
- Input: request, router_state, replica_states[]
- Output: ADMIT | REJECT | DELAY(duration)
- Must not modify inputs; decision recorded in trace

**PriorityPolicy:**
- Input: request, router_state, replica_states[]
- Output: score (float64), hints (map)
- Deterministic for identical inputs

**RoutingPolicy:**
- Input: request, router_state, replica_states[]
- Output: valid replica_id
- Updates router shadow KV model

**InstanceScheduler:**
- MakeBatch respects max_batch_size and max_token_budget
- SelectPreemptionVictim returns request from running_batch or nil
- No request loss: all requests eventually complete or timeout

### BDD Scenarios (Examples)

```gherkin
Scenario: Route to replica with cached prefix
  Given replica 0 has prefix "0xABC123" in KV cache
  And replica 1 does not have prefix "0xABC123"
  When a request with prefix "0xABC123" arrives
  Then the routing policy selects replica 0

Scenario: Realtime requests scheduled before batch
  Given wait queue has batch request (arrival=100) and realtime request (arrival=101)
  When MakeBatch is called with priority-aware scheduler
  Then realtime request is scheduled first
```

---

## Summary

This design extends BLIS to support evolutionary policy optimization through:

1. **Multi-replica cluster simulation** with shared event loop
2. **Pluggable policy pipeline** (admission → priority → routing) at cluster level
3. **Pluggable instance schedulers** at replica level
4. **Rich workload modeling** (multi-tenant, priorities, prefixes, bursty arrivals)
5. **Flexible fitness evaluation** (single/multi-objective, SLO-based)
6. **Trace collection and summarization** for evolutionary framework feedback
7. **Framework-agnostic integration** supporting OpenEvolve and GEPA

Behavioral contracts provide the foundation for BDD/TDD development.
