# PR4 Phases 2-5: Implementation Planning

## Phase 2: Component Interaction

### 2.1 Component Diagram

```
┌─────────────────────────── ClusterSimulator ──────────────────────────────┐
│                                                                           │
│  ┌──────────────────────── Control Plane ────────────────────────────┐   │
│  │                                                                    │   │
│  │  ClusterEventQueue (min-heap: timestamp, priority, seqID)         │   │
│  │    │                                                               │   │
│  │    ├─ ClusterArrivalEvent (prio=0)                                │   │
│  │    │    └──► AdmissionDecisionEvent (prio=1)                      │   │
│  │    │           └──► RoutingDecisionEvent (prio=2)                 │   │
│  │    │                   └──► instance.InjectRequestOnline()        │   │
│  │    │                                                               │   │
│  │  AdmissionPolicy ←── AlwaysAdmit | TokenBucket                   │   │
│  │  SnapshotProvider ←── CachedSnapshotProvider(ObservabilityConfig) │   │
│  │  roundRobinCounter int                                             │   │
│  │  admissionLatency, routingLatency int64 (µs)                      │   │
│  │                                                                    │   │
│  └────────────────────────────┬───────────────────────────────────────┘   │
│                               │ InjectRequestOnline(req, eventTime)       │
│                               ▼                                           │
│  ┌──────────────────────── Data Plane ────────────────────────────────┐   │
│  │                                                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │   │
│  │  │ Instance 0   │  │ Instance 1   │  │ Instance N-1 │  ...       │   │
│  │  │  EventQueue  │  │  EventQueue  │  │  EventQueue  │            │   │
│  │  │  WaitQueue   │  │  WaitQueue   │  │  WaitQueue   │            │   │
│  │  │  KVCache     │  │  KVCache     │  │  KVCache     │            │   │
│  │  │  RunningBatch│  │  RunningBatch│  │  RunningBatch│            │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘            │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  Main Loop:                                                               │
│    1. Find min(clusterQueue.peek, instanceQueues.peek)                    │
│    2. If cluster event <= instance event: pop & execute cluster event     │
│    3. Else: pop & execute instance event (delegate to instance)           │
│    4. Break when all queues empty or clock > horizon                      │
└───────────────────────────────────────────────────────────────────────────┘
```

**Data flow:** `Request → ClusterArrivalEvent → AdmissionDecisionEvent → RoutingDecisionEvent → InstanceSimulator.InjectRequestOnline()`

**Types crossing boundaries:**
- Control→Data: `*sim.Request` (passed by pointer; never mutated by control plane)
- Data→Control: `InstanceSnapshot` (value type, copied from instance observation methods)

### 2.2 API Contracts

**ClusterEvent interface** (`sim/cluster/cluster_event.go`)
```go
// ClusterEvent is processed by ClusterSimulator, separate from sim.Event
// which is processed by sim.Simulator. (Addresses deviation D4.)
type ClusterEvent interface {
    Timestamp() int64
    Priority() int               // 0=Arrival, 1=Admission, 2=Routing
    Execute(*ClusterSimulator)
}
```

**ClusterEventQueue** (`sim/cluster/cluster_event.go`)
```go
// ClusterEventQueue is a min-heap ordered by (Timestamp, Priority, seqID).
// seqID is a monotonic counter assigned at push time for deterministic FIFO
// tie-breaking within same (Timestamp, Priority).
type clusterEventEntry struct {
    event ClusterEvent
    seqID int64
}
type ClusterEventQueue []clusterEventEntry
// Implements heap.Interface with Less: timestamp < priority < seqID
```

**Concrete cluster events** (`sim/cluster/cluster_event.go`)
```go
type ClusterArrivalEvent struct {
    time    int64
    request *sim.Request
}
// Priority() = 0
// Execute(): pushes AdmissionDecisionEvent{time: e.time + cs.admissionLatency}

type AdmissionDecisionEvent struct {
    time    int64
    request *sim.Request
}
// Priority() = 1
// Execute(): calls cs.admissionPolicy.Admit(request, snapshots).
//   If admitted: pushes RoutingDecisionEvent{time: e.time + cs.routingLatency}
//   If rejected: increments cs.rejectedRequests counter

type RoutingDecisionEvent struct {
    time    int64
    request *sim.Request
}
// Priority() = 2
// Execute(): selects target = cs.instances[cs.roundRobinCounter % N],
//   calls target.InjectRequestOnline(request, e.time), increments counter
```

**InstanceSnapshot** (`sim/cluster/snapshot.go`)
```go
// InstanceSnapshot is an immutable value type (returned by value, no shared pointers).
// Fields limited to those observable in PR4. Extended in later PRs.
type InstanceSnapshot struct {
    ID            InstanceID
    Timestamp     int64    // clock time when captured
    QueueDepth    int      // len(WaitQ)
    BatchSize     int      // len(RunningBatch.Requests), 0 if nil
    KVUtilization float64  // UsedBlockCnt / TotalBlocks
    FreeKVBlocks  int64    // TotalBlocks - UsedBlockCnt
}
```

**SnapshotProvider interface** (`sim/cluster/snapshot.go`)
```go
type SnapshotProvider interface {
    Snapshot(id InstanceID, clock int64) InstanceSnapshot
    RefreshAll(clock int64)
}
```

**CachedSnapshotProvider** (`sim/cluster/snapshot.go`)
```go
type UpdateMode int
const (
    Immediate UpdateMode = iota // re-read every access
    Periodic                     // re-read when interval elapsed
    OnDemand                     // re-read only on RefreshAll()
)

type FieldConfig struct {
    Mode     UpdateMode
    Interval int64 // ticks, only for Periodic
}

type ObservabilityConfig struct {
    QueueDepth    FieldConfig // default: Immediate
    BatchSize     FieldConfig // default: Immediate
    KVUtilization FieldConfig // default: Immediate
    // FreeKVBlocks: always Immediate, not configurable
}

type CachedSnapshotProvider struct {
    instances map[InstanceID]*InstanceSimulator
    config    ObservabilityConfig
    cache     map[InstanceID]InstanceSnapshot
    lastRefresh map[InstanceID]fieldTimestamps // per-field last-refresh times
}
// Snapshot(id, clock): per-field check against config, re-read if stale
// RefreshAll(clock): force re-read all OnDemand fields
```

**AdmissionPolicy interface** (`sim/policy/admission.go`)
```go
type AdmissionPolicy interface {
    // Admit returns true if the request should proceed to routing.
    // snapshots provides current cluster state for informed decisions.
    Admit(req *sim.Request, snapshots map[cluster.InstanceID]cluster.InstanceSnapshot, clock int64) (admitted bool, reason string)
}

type AlwaysAdmit struct{}
// Admit() always returns (true, "")

type TokenBucket struct {
    capacity      float64 // max tokens
    refillRate    float64 // tokens per second (refilled based on clock µs)
    currentTokens float64
    lastRefill    int64   // last refill clock time (µs)
}
// Admit(): refill based on elapsed time, check currentTokens >= len(req.InputTokens),
//   deduct on admit, return (false, "insufficient tokens") on reject
```

**New observation methods on InstanceSimulator** (`sim/cluster/instance.go`)
```go
func (i *InstanceSimulator) QueueDepth() int         // delegates to i.sim.WaitQ.Len()
func (i *InstanceSimulator) BatchSize() int           // len(i.sim.RunningBatch.Requests), 0 if nil
func (i *InstanceSimulator) KVUtilization() float64   // float64(UsedBlockCnt) / float64(TotalBlocks)
func (i *InstanceSimulator) FreeKVBlocks() int64      // TotalBlocks - UsedBlockCnt
```

**New methods on base Simulator** (`sim/simulator.go`)
```go
// InjectArrivalAt schedules an ArrivalEvent at eventTime (not req.ArrivalTime).
// Metrics.Requests uses req.ArrivalTime for ArrivedAt (preserves original arrival).
func (sim *Simulator) InjectArrivalAt(req *Request, eventTime int64)
```

**New method on InstanceSimulator** (`sim/cluster/instance.go`)
```go
// InjectRequestOnline injects a request during the cluster event loop.
// Unlike InjectRequest, does NOT check hasRun (addresses D5).
func (i *InstanceSimulator) InjectRequestOnline(req *sim.Request, eventTime int64)
// Delegates to i.sim.InjectArrivalAt(req, eventTime)
```

**WaitQueue.Len()** (`sim/queue.go`)
```go
func (wq *WaitQueue) Len() int { return len(wq.queue) }
```

### 2.3 State Changes

**New mutable state on ClusterSimulator:**

| Field | Type | Owner | Lifecycle |
|-------|------|-------|-----------|
| `clusterEvents` | `ClusterEventQueue` | ClusterSimulator | Initialized empty in constructor; populated in Run() during request scheduling; drained during event loop |
| `nextSeqID` | `int64` | ClusterSimulator | Monotonic counter, incremented on each cluster event push; ensures deterministic FIFO ordering |
| `admissionPolicy` | `AdmissionPolicy` | ClusterSimulator | Set in constructor based on config; immutable during Run() |
| `snapshotProvider` | `SnapshotProvider` | ClusterSimulator | Set in constructor; snapshots refreshed during event loop |
| `admissionLatency` | `int64` | ClusterSimulator | Set in constructor from config; immutable during Run() |
| `routingLatency` | `int64` | ClusterSimulator | Set in constructor from config; immutable during Run() |
| `roundRobinCounter` | `int` | ClusterSimulator | Starts at 0; incremented by RoutingDecisionEvent.Execute() |
| `rejectedRequests` | `int` | ClusterSimulator | Starts at 0; incremented by AdmissionDecisionEvent when Admit() returns false |

**New mutable state on TokenBucket:**

| Field | Type | Owner | Lifecycle |
|-------|------|-------|-----------|
| `currentTokens` | `float64` | TokenBucket | Initialized to capacity; decremented on admit, refilled on each Admit() call |
| `lastRefill` | `int64` | TokenBucket | Initialized to 0; updated on each Admit() call |

**Modified state on DeploymentConfig:**

| Field | Type | Added |
|-------|------|-------|
| `AdmissionPolicy` | `string` | New: "always-admit" (default), "token-bucket" |
| `AdmissionLatency` | `int64` | New: microseconds, default 0 |
| `RoutingLatency` | `int64` | New: microseconds, default 0 |
| `TokenBucketCapacity` | `float64` | New: max token count, default 10000 |
| `TokenBucketRefillRate` | `float64` | New: tokens/second, default 1000 |

---

## Phase 3: Deviation Log

| # | Macro Plan Says | Micro Plan Does | Reason |
|---|-----------------|-----------------|--------|
| 1 | `sim/request.go` (~5 LOC TenantID) | **Deferred to PR5.** No TenantID addition in PR4. | No PR4 contract reads TenantID. AlwaysAdmit and TokenBucket don't use it. Adding it would be dead code. PR5 (PriorityPolicy) is the first consumer. |
| 2 | Files changed list does not include `sim/simulator.go` | **Add `InjectArrivalAt()` to `sim/simulator.go` (~8 LOC).** | `ArrivalEvent.time` field is unexported; only `sim` package can construct it. Needed for RoutingDecisionEvent to inject at routing time, not original arrival time. With zero latencies, times are equal (backward compat preserved). |
| 3 | Files changed list does not include `sim/queue.go` | **Add `Len()` to `sim/queue.go` (~3 LOC).** | `WaitQueue.queue` is unexported. `QueueDepth()` observation method needs public accessor. Already identified in Phase 0 as deviation D1. |
| 4 | `InstanceSnapshot` has 12+ fields (PoolType, InFlightRequests, RecentTTFT, etc.) | **PR4 defines 6 fields:** ID, Timestamp, QueueDepth, BatchSize, KVUtilization, FreeKVBlocks. | Only 4 observation methods are added in PR4. Defining unpopulated fields is dead code. Go struct extension (adding fields later) is backward compatible. |
| 5 | `ObservabilityConfig` has 6 field configs (includes CacheHitRate, RecentTTFT, RecentTPOT) | **PR4 defines 3 field configs:** QueueDepth, BatchSize, KVUtilization. FreeKVBlocks is always Immediate. | No observation methods for CacheHitRate, RecentTTFT, RecentTPOT in PR4. Config for unobservable fields is dead code. |
| 6 | `--policy-config` YAML parameterization in PR8 | **PR4 adds `--token-bucket-capacity` and `--token-bucket-refill-rate` flags.** | TokenBucket needs parameterization to be exercisable. Dedicated flags are sufficient for PR4; PR8 replaces with generic YAML config. |
| 7 | LOC estimate ~450 | **Estimated ~500 LOC** (150 cluster_event + 200 snapshot + 150 admission). Modifications ~50 LOC additional. | Adding `InjectArrivalAt`, `Len()`, and dedicated TokenBucket CLI flags slightly exceeds macro estimate. |

---

## Phase 4: Implementation Summary

### 4.1 Files to Modify

1. **`sim/cluster/cluster.go`** (~200 LOC restructure): Add ClusterEventQueue field, admissionPolicy, snapshotProvider, latency fields, roundRobinCounter, rejectedRequests. Replace pre-dispatch loop (lines 81-88) with ClusterArrivalEvent scheduling. Restructure shared-clock loop to check both cluster and instance queues. Update `NewClusterSimulator` constructor. Add `ScheduleClusterEvent` helper.
2. **`sim/cluster/instance.go`** (~35 LOC): Add 4 observation methods (`QueueDepth`, `BatchSize`, `KVUtilization`, `FreeKVBlocks`) and `InjectRequestOnline(req, eventTime)`.
3. **`sim/cluster/deployment.go`** (~10 LOC): Add `AdmissionPolicy`, `AdmissionLatency`, `RoutingLatency`, `TokenBucketCapacity`, `TokenBucketRefillRate` fields to `DeploymentConfig`.
4. **`sim/simulator.go`** (~8 LOC): Add `InjectArrivalAt(req, eventTime)` method.
5. **`sim/queue.go`** (~3 LOC): Add `WaitQueue.Len()` method.
6. **`cmd/root.go`** (~30 LOC): Add `--admission-policy`, `--admission-latency`, `--routing-latency`, `--token-bucket-capacity`, `--token-bucket-refill-rate` flags. Wire into `DeploymentConfig`.

### 4.2 New Files to Create

1. **`sim/cluster/cluster_event.go`** (~150 LOC): `ClusterEvent` interface, `ClusterEventQueue` (min-heap with seqID), `ClusterArrivalEvent`, `AdmissionDecisionEvent`, `RoutingDecisionEvent`. **Justification:** Separate interface from `sim.Event` (D4); cluster events execute on `*ClusterSimulator`, not `*Simulator`. All three event types are exercised by every cluster simulation run.
2. **`sim/cluster/snapshot.go`** (~200 LOC): `InstanceSnapshot` struct, `SnapshotProvider` interface, `CachedSnapshotProvider`, `UpdateMode` enum, `FieldConfig`, `ObservabilityConfig`, `DefaultObservabilityConfig()`. **Justification:** Snapshot is consumed by `AdmissionDecisionEvent` (passes snapshots to `Admit()`). CachedSnapshotProvider exercises the staleness model. All code reachable via event pipeline.
3. **`sim/policy/admission.go`** (~150 LOC): `AdmissionPolicy` interface, `AlwaysAdmit` struct, `TokenBucket` struct, `NewAdmissionPolicy(name, config)` factory. **Justification:** New `sim/policy/` package establishes policy namespace per macro plan. AlwaysAdmit is the default (exercised every run). TokenBucket exercisable via `--admission-policy token-bucket`.

### 4.3 Key Decisions

1. **Separate `ClusterEvent` interface (not extending `sim.Event`):** `sim.Event.Execute(*Simulator)` is coupled to single-instance Simulator. Cluster events need `Execute(*ClusterSimulator)`. A unified interface would require type assertions or generics. Separate interfaces keep the boundary clean. (D4)

2. **`InjectRequestOnline` bypasses `hasRun` guard via new method (not modifying `InjectRequest`):** Existing `InjectRequest` panics after `Run()` as a safety guard. Rather than removing this guard (which protects standalone mode), we add `InjectRequestOnline` specifically for cluster-mode injection during the event loop. The two methods have different semantics: `InjectRequest` is for pre-loop setup, `InjectRequestOnline` is for online routing.

3. **`InjectArrivalAt` on `Simulator` (not hacking `req.ArrivalTime`):** With non-zero latencies, the instance ArrivalEvent time differs from `req.ArrivalTime`. Mutating `req.ArrivalTime` would corrupt metrics (`ArrivedAt`, `RequestSchedulingDelays`). A new method keeps the original arrival time for metrics while scheduling the event at the routing decision time.

4. **Cluster events before instance events at same timestamp (BC-4) via `<=` comparison:** The main loop uses `clusterTime <= instanceTime` to prioritize cluster events. This naturally drains all cluster events at time T before any instance event at T, because: (a) processing a cluster event may add more cluster events at T (e.g., zero-latency chain), (b) the loop re-evaluates after each event, (c) cluster events only cease having priority when the queue is empty or moves past T.

5. **seqID for deterministic FIFO within same (timestamp, priority):** Multiple requests can arrive at the same timestamp. Without seqID, heap ordering among ClusterArrivalEvents at the same time is non-deterministic. seqID (monotonic counter assigned at push time) ensures FIFO ordering, making round-robin assignment match PR3's dispatch order.

### 4.4 No Dead Code Confirmation

- **All cluster event types** are exercised on every multi-instance run (ClusterArrival→Admission→Routing pipeline).
- **AlwaysAdmit** is the default policy, exercised by default on every multi-instance run.
- **TokenBucket** is exercisable via `--admission-policy token-bucket` with `--token-bucket-capacity` and `--token-bucket-refill-rate` flags.
- **CachedSnapshotProvider** is constructed by `ClusterSimulator` and consumed by `AdmissionDecisionEvent` on every admission check.
- **Observation methods** (`QueueDepth`, `BatchSize`, `KVUtilization`, `FreeKVBlocks`) are called by `CachedSnapshotProvider.Snapshot()` on every admission event.
- **`WaitQueue.Len()`** is called by `InstanceSimulator.QueueDepth()`.
- **`InjectArrivalAt`** is called by `InjectRequestOnline` which is called by `RoutingDecisionEvent.Execute()`.
- **`ObservabilityConfig` Periodic/OnDemand modes** are exercisable via test configuration (not CLI in PR4, but fully reachable via `CachedSnapshotProvider` constructor).
- **No TenantID** added (deferred to PR5 — avoids dead field).
- **No unpopulated InstanceSnapshot fields** (only observable fields defined).

---

## Phase 5: Exercisability Proof

### 5.1 CLI Exercisable Paths

| Codepath | CLI Command | Observable Behavior |
|----------|-------------|---------------------|
| Online routing pipeline with AlwaysAdmit (BC-1,2,3,4,11) | `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 2 --admission-policy always-admit` | Output matches PR3 exactly (same TTFT, E2E, ITL per request). Backward compat verified. |
| Online routing with non-zero latencies (BC-2) | `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 2 --admission-latency 100 --routing-latency 50` | All request metrics show latency offset: scheduling delays increase by 150µs vs zero-latency baseline. |
| TokenBucket admission (BC-10, EC-2) | `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 2 --admission-policy token-bucket --token-bucket-capacity 500 --token-bucket-refill-rate 100 --rate 10 --max-prompts 50` | Some requests rejected (reduced CompletedRequests vs always-admit). Rejected count logged/visible in output. |
| Single-instance backward compat | `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 1` | Single-instance path unchanged (uses `InstanceSimulator.Run()`, not cluster event loop). |
| Invalid admission policy (EC-1) | `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 2 --admission-policy invalid-name` | Panics with: `unknown admission policy "invalid-name"; valid policies: [always-admit, token-bucket]` |

### 5.2 Test-Only Paths

| Codepath | Test Name | Why CLI N/A |
|----------|-----------|-------------|
| ClusterEvent ordering (BC-4) | `TestClusterEventQueue_Ordering` | Internal heap ordering; not directly observable from CLI output, only from event execution order |
| InstanceSnapshot immutability (BC-5, NC-2) | `TestInstanceSnapshot_Immutability` | Value semantics verification; no CLI output for snapshot internals |
| CachedSnapshotProvider Periodic refresh (BC-6) | `TestCachedSnapshotProvider_PeriodicRefresh` | ObservabilityConfig with Periodic mode not exposed via CLI in PR4; requires programmatic configuration |
| CachedSnapshotProvider default config (BC-7) | `TestCachedSnapshotProvider_DefaultImmediate` | Defaults are implicit; test verifies all fields use Immediate mode |
| Observation methods accuracy (BC-8) | `TestInstanceSimulator_ObservationMethods` | Internal observation accuracy; CLI output shows aggregate metrics, not per-instance observations |
| AlwaysAdmit unit (BC-9) | `TestAlwaysAdmit` | Trivial policy; isolated unit test faster than full simulation |
| TokenBucket arithmetic (BC-10) | `TestTokenBucket_RefillAndDeduct` | Precise token arithmetic; CLI only shows aggregate effect (requests admitted/rejected) |
| No pre-dispatch (NC-1) | `TestRun_NoPredispatch` | Verifies instance event queues are empty before loop starts; internal invariant not in CLI output |
| Determinism (NC-3) | `TestClusterSimulator_Determinism` | Runs simulation K times, asserts bit-identical metrics; verifiable via CLI but test is more reliable |
