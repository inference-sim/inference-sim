# PR4 Micro-Plan: Cluster Control Plane + AdmissionPolicy

**PR Title:** `feat(cluster): Add cluster event infrastructure, SnapshotProvider, and AdmissionPolicy`

**Status:** Ready for Implementation

---

## PART 1: Human Review

### A) Executive Summary

PR4 introduces the **control plane / data plane separation** for cluster simulation, addressing the critical finding from the mock study that pre-dispatch routing breaks load-aware policies.

**Building block:** ClusterSimulator control plane with online routing pipeline
**Adjacent blocks:** InstanceSimulators (data plane), sim.Event system, DeploymentConfig
**Key change:** Requests are now routed at arrival time during the event loop, not batch-dispatched before it

**Deviations from macro plan:**
- D1 (CRITICAL): `WaitQueue.Len()` missing — added in PR4
- D5 (CRITICAL): `InjectRequest` panic guard — new `InjectRequestOnline` method added
- TenantID deferred to PR5 (no PR4 consumer)

**LOC estimate:** ~500 (vs macro's ~450 due to InjectArrivalAt + CLI flags)

---

### B) Behavioral Contracts

**16 contracts total (11 behavioral, 3 negative, 2 error)**

#### Online Routing Pipeline (BC-1 to BC-4)
- **BC-1:** ClusterArrivalEvent scheduling replaces pre-dispatch
- **BC-2:** Admission→routing pipeline with configurable latency
- **BC-3:** RoutingDecisionEvent injects into instance via `InjectRequestOnline`
- **BC-4:** Event ordering: cluster events before instance events at same timestamp

#### InstanceSnapshot and SnapshotProvider (BC-5 to BC-7)
- **BC-5:** Immutable value-type snapshots with Timestamp
- **BC-6:** CachedSnapshotProvider with Immediate/Periodic/OnDemand refresh
- **BC-7:** Default config uses all-Immediate mode

#### Observation Methods (BC-8)
- **BC-8:** QueueDepth, BatchSize, KVUtilization, FreeKVBlocks delegate to wrapped Simulator

#### AdmissionPolicy (BC-9, BC-10)
- **BC-9:** AlwaysAdmit (default) admits all requests
- **BC-10:** TokenBucket respects capacity/refill-rate, deterministic arithmetic

#### Backward Compatibility (BC-11)
- **BC-11:** Default config produces identical metrics to PR3

#### Negative Contracts (NC-1 to NC-3)
- **NC-1:** No pre-dispatch before event loop
- **NC-2:** No state leakage between snapshots
- **NC-3:** No determinism regression

#### Error Contracts (EC-1, EC-2)
- **EC-1:** Invalid policy name fails at construction
- **EC-2:** TokenBucket rejection recorded in cluster metrics

*Full contract details in `docs/plans/pr4-phase1-behavioral-contracts.md`*

---

### C) Component Interaction

```
┌─────────────────────────── ClusterSimulator ───────────────────────────┐
│  ┌──────────────────────── Control Plane ────────────────────────────┐ │
│  │  ClusterEventQueue (min-heap: timestamp, priority, seqID)         │ │
│  │    ├─ ClusterArrivalEvent (prio=0) → AdmissionDecisionEvent (1)  │ │
│  │    │                                  → RoutingDecisionEvent (2)  │ │
│  │  AdmissionPolicy ←── AlwaysAdmit | TokenBucket                    │ │
│  │  SnapshotProvider ←── CachedSnapshotProvider(ObservabilityConfig) │ │
│  └───────────────────────────────┬───────────────────────────────────┘ │
│                                  │ InjectRequestOnline()               │
│  ┌──────────────────────── Data Plane ───────────────────────────────┐ │
│  │  Instance 0         Instance 1         Instance N-1               │ │
│  │  [EventQ, WaitQ,    [EventQ, WaitQ,    [EventQ, WaitQ,            │ │
│  │   KVCache, Batch]    KVCache, Batch]    KVCache, Batch]           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

**Data flow:** Request → ClusterArrivalEvent → AdmissionDecisionEvent → RoutingDecisionEvent → Instance injection

**Types crossing boundaries:**
- Control→Data: `*sim.Request` (by pointer, never mutated)
- Data→Control: `InstanceSnapshot` (value type, copied)

---

### D) Deviation Log

| # | Macro Plan Says | Micro Plan Does | Reason |
|---|-----------------|-----------------|--------|
| 1 | TenantID in sim/request.go | Deferred to PR5 | No PR4 consumer; would be dead code |
| 2 | No sim/simulator.go changes | Add InjectArrivalAt (~8 LOC) | ArrivalEvent.time unexported |
| 3 | No sim/queue.go changes | Add Len() (~3 LOC) | D1 fix for QueueDepth |
| 4 | InstanceSnapshot 12+ fields | 6 fields only | Only observable fields; avoid dead fields |
| 5 | ObservabilityConfig 6 fields | 3 field configs | No CacheHitRate observation in PR4 |
| 6 | --policy-config YAML (PR8) | Dedicated TokenBucket flags | Sufficient for PR4 exercisability |
| 7 | ~450 LOC | ~500 LOC | Additional methods + CLI flags |

---

### E) Review Guide

1. **THE TRICKY PART:** The restructured `ClusterSimulator.Run()` event loop must correctly interleave cluster-level events with instance-level events, respecting BC-4 priority ordering. The zero-latency path must produce identical output to PR3.

2. **WHAT TO SCRUTINIZE:** BC-11 (backward compatibility) and BC-4 (event ordering). For BC-11, the golden equivalence test is the oracle. For BC-4, manually trace concurrent arrivals at the same timestamp. Also scrutinize `InjectRequestOnline` (D5 fix).

3. **WHAT'S SAFE TO SKIM:** AlwaysAdmit (2 lines), InstanceSnapshot struct definition, ObservabilityConfig fields, WaitQueue.Len() (1 line), CLI flag wiring.

4. **KNOWN DEBT:** (a) WaitQueue.queue unexported — Len() added. (b) RunningBatch can be nil. (c) CacheHitRate deferred. (d) TokenBucket precision adequate for PR4 scope.

---

## PART 2: Implementation Reference

### F) Implementation Summary

**Files to modify (6):**
1. `sim/cluster/cluster.go` (~200 LOC): Add ClusterEventQueue, restructure Run()
2. `sim/cluster/instance.go` (~35 LOC): Add observation methods + InjectRequestOnline
3. `sim/cluster/deployment.go` (~10 LOC): Add config fields
4. `sim/simulator.go` (~8 LOC): Add InjectArrivalAt
5. `sim/queue.go` (~3 LOC): Add Len()
6. `cmd/root.go` (~30 LOC): Add CLI flags

**Files to create (3):**
1. `sim/cluster/cluster_event.go` (~150 LOC): ClusterEvent interface + 3 event types
2. `sim/cluster/snapshot.go` (~200 LOC): InstanceSnapshot + SnapshotProvider
3. `sim/policy/admission.go` (~150 LOC): AdmissionPolicy + AlwaysAdmit + TokenBucket

**Key decisions:**
1. Separate ClusterEvent interface (not extending sim.Event) — clean boundary
2. InjectRequestOnline bypasses hasRun guard — preserves standalone safety
3. InjectArrivalAt keeps original arrival time for metrics
4. Cluster events before instance events via `<=` comparison
5. seqID for deterministic FIFO tie-breaking

---

### G) Exercisability Proof

**CLI Exercisable:**
| Codepath | CLI Command |
|----------|-------------|
| Default pipeline (BC-1,2,3,4,11) | `--num-instances 2 --admission-policy always-admit` |
| Non-zero latencies (BC-2) | `--admission-latency 100 --routing-latency 50` |
| TokenBucket (BC-10, EC-2) | `--admission-policy token-bucket --token-bucket-capacity 500` |
| Invalid policy (EC-1) | `--admission-policy invalid-name` → panic |

**Test-Only:**
| Codepath | Test Name |
|----------|-----------|
| Event ordering (BC-4) | TestClusterEventQueue_Ordering |
| Snapshot immutability (BC-5) | TestInstanceSnapshot_Immutability |
| Observation accuracy (BC-8) | TestInstanceSimulator_ObservationMethods |
| Determinism (NC-3) | TestClusterSimulator_Determinism |

---

### H) Test Strategy

| Contract | Test Type | Test File | Test Name |
|----------|-----------|-----------|-----------|
| BC-1, NC-1 | Integration | cluster_test.go | TestClusterSimulator_OnlineRouting_NoPreDispatch |
| BC-2 | Integration | cluster_event_test.go | TestClusterEvents_AdmissionToRoutingPipeline |
| BC-3 | Integration | cluster_event_test.go | TestRoutingDecisionEvent_InjectsIntoInstance |
| BC-4 | Unit | cluster_event_test.go | TestClusterEventQueue_Ordering |
| BC-5, NC-2 | Unit | snapshot_test.go | TestInstanceSnapshot_Immutability |
| BC-6 | Unit | snapshot_test.go | TestCachedSnapshotProvider_RefreshBehavior |
| BC-7 | Unit | snapshot_test.go | TestSnapshotProvider_DefaultConfig_AllImmediate |
| BC-8 | Unit | instance_test.go | TestInstanceSimulator_ObservationMethods |
| BC-9 | Unit | admission_test.go | TestAlwaysAdmit_AdmitsAll |
| BC-10 | Unit | admission_test.go | TestTokenBucket_AdmitAndReject |
| BC-11 | Integration | cluster_test.go | TestClusterSimulator_SingleInstance_GoldenEquivalence |
| NC-3 | Integration | cluster_test.go | TestClusterSimulator_MultiInstance_Determinism |
| EC-1 | Unit | admission_test.go | TestAdmissionPolicy_InvalidName_Panics |
| EC-2 | Integration | cluster_event_test.go | TestAdmissionDecisionEvent_Rejection |

**New test files:** `cluster_event_test.go`, `snapshot_test.go`, `sim/policy/admission_test.go`

---

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Large PR scope (~500 LOC) | Medium | Medium | Split into logical commits; structured review |
| Architectural pivot | Medium | High | BC-11 golden equivalence test as gatekeeper |
| Backward compatibility regression | Medium | High | Per-request TTFT/E2E comparison in tests |
| Determinism regression | Low | High | NC-3 runs 3x with full map comparison |
| Event ordering complexity | Medium | High | BC-4 table-driven unit tests |
| Nil RunningBatch panic | High | Medium | BC-8 explicit nil test case |
| InjectRequest panic guard (D5) | High | High | New InjectRequestOnline method |

---

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used
- [x] CLAUDE.md updates planned (sim/policy/, CLI flags, PR4 status)
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — all resolved

---

## APPENDIX: File-Level Details

### K.1 ClusterEvent Interface (`sim/cluster/cluster_event.go`)

```go
type ClusterEvent interface {
    Timestamp() int64
    Priority() int  // 0=Arrival, 1=Admission, 2=Routing
    Execute(*ClusterSimulator)
}

type clusterEventEntry struct {
    event ClusterEvent
    seqID int64
}

type ClusterEventQueue []clusterEventEntry
// heap.Interface: Less by (Timestamp, Priority, seqID)
```

### K.2 Concrete Event Types

```go
type ClusterArrivalEvent struct {
    time    int64
    request *sim.Request
}
// Execute: push AdmissionDecisionEvent{time: e.time + cs.admissionLatency}

type AdmissionDecisionEvent struct {
    time    int64
    request *sim.Request
}
// Execute: call cs.admissionPolicy.Admit(); if admitted, push RoutingDecisionEvent

type RoutingDecisionEvent struct {
    time    int64
    request *sim.Request
}
// Execute: target = cs.instances[cs.roundRobinCounter % N]; target.InjectRequestOnline()
```

### K.3 InstanceSnapshot (`sim/cluster/snapshot.go`)

```go
type InstanceSnapshot struct {
    ID            InstanceID
    Timestamp     int64
    QueueDepth    int
    BatchSize     int
    KVUtilization float64
    FreeKVBlocks  int64
}
```

### K.4 SnapshotProvider + CachedSnapshotProvider

```go
type UpdateMode int
const (
    Immediate UpdateMode = iota
    Periodic
    OnDemand
)

type FieldConfig struct {
    Mode     UpdateMode
    Interval int64
}

type ObservabilityConfig struct {
    QueueDepth    FieldConfig
    BatchSize     FieldConfig
    KVUtilization FieldConfig
}

type CachedSnapshotProvider struct {
    instances   map[InstanceID]*InstanceSimulator
    config      ObservabilityConfig
    cache       map[InstanceID]InstanceSnapshot
    lastRefresh map[InstanceID]fieldTimestamps
}
```

### K.5 AdmissionPolicy (`sim/policy/admission.go`)

```go
type AdmissionPolicy interface {
    Admit(req *sim.Request, snapshots map[cluster.InstanceID]cluster.InstanceSnapshot, clock int64) (admitted bool, reason string)
}

type AlwaysAdmit struct{}
// Admit() returns (true, "")

type TokenBucket struct {
    capacity      float64
    refillRate    float64
    currentTokens float64
    lastRefill    int64
}
// Admit(): refill based on elapsed time, check tokens >= cost
```

### K.6 New Methods

```go
// sim/cluster/instance.go
func (i *InstanceSimulator) QueueDepth() int
func (i *InstanceSimulator) BatchSize() int
func (i *InstanceSimulator) KVUtilization() float64
func (i *InstanceSimulator) FreeKVBlocks() int64
func (i *InstanceSimulator) InjectRequestOnline(req *sim.Request, eventTime int64)

// sim/simulator.go
func (sim *Simulator) InjectArrivalAt(req *Request, eventTime int64)

// sim/queue.go
func (wq *WaitQueue) Len() int { return len(wq.queue) }
```

### K.7 CLI Flags (`cmd/root.go`)

```go
--admission-policy string      // "always-admit" (default), "token-bucket"
--admission-latency int        // microseconds, default 0
--routing-latency int          // microseconds, default 0
--token-bucket-capacity float  // max tokens, default 10000
--token-bucket-refill-rate float // tokens/sec, default 1000
```

### K.8 DeploymentConfig Extensions (`sim/cluster/deployment.go`)

```go
type DeploymentConfig struct {
    // existing fields...
    AdmissionPolicy       string
    AdmissionLatency      int64
    RoutingLatency        int64
    TokenBucketCapacity   float64
    TokenBucketRefillRate float64
}
```

---

*Generated by PR4 planning team: recon-lead, contract-architect, impl-planner, test-analyst*
