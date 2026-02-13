# PR4 Phase 1: Behavioral Contracts

## Summary

16 contracts covering the PR4 scope: cluster control plane with online routing pipeline, InstanceSnapshot observability, SnapshotProvider caching, AdmissionPolicy (AlwaysAdmit + TokenBucket), configurable latency, and backward compatibility. Contracts are grounded in codebase recon deviations (D1, D4, D5) and the v2.3 macro plan.

**Coverage areas:**
- Online routing pipeline (4 contracts): ClusterArrivalEvent → AdmissionDecisionEvent → RoutingDecisionEvent → instance injection
- InstanceSnapshot and SnapshotProvider (3 contracts): immutable snapshots, timestamp, refresh behavior
- Observation methods (1 contract): QueueDepth, BatchSize, KVUtilization, FreeKVBlocks
- AdmissionPolicy (2 contracts): AlwaysAdmit, TokenBucket
- Backward compatibility (1 contract): default config matches PR3 output exactly
- Negative contracts (3 contracts): no pre-dispatch, no state leakage, no determinism regression
- Error contracts (2 contracts): invalid admission policy, TokenBucket rejection

---

## 1. Behavioral Contracts

### 1.1 Online Routing Pipeline

```
BC-1: ClusterArrivalEvent Scheduling
- GIVEN a ClusterSimulator with N >= 1 instances and a workload generating R requests
- WHEN Run() begins the event loop
- THEN each request MUST be scheduled as a ClusterArrivalEvent in the cluster-level
  event queue at its original arrival time, NOT pre-dispatched to instances before the
  event loop starts
- MECHANISM: Run() generates requests, then for each request schedules a
  ClusterArrivalEvent with timestamp = request.ArrivalTime into a cluster-level
  min-heap event queue. The current batch-dispatch loop (cluster.go:81-88) is replaced.
  (Addresses deviation D5: InjectRequest panic guard blocks online routing.)
```

```
BC-2: Admission-to-Routing Pipeline
- GIVEN a ClusterArrivalEvent executing at time T with admission-latency=La and
  routing-latency=Lr (both in microseconds, both >= 0)
- WHEN the ClusterArrivalEvent executes
- THEN it MUST schedule an AdmissionDecisionEvent at time T+La, and if the request
  is admitted, that event MUST schedule a RoutingDecisionEvent at time T+La+Lr
- MECHANISM: ClusterArrivalEvent.Execute() creates AdmissionDecisionEvent{time: T+La}.
  AdmissionDecisionEvent.Execute() calls AdmissionPolicy.Admit(), and on admit creates
  RoutingDecisionEvent{time: T+La+Lr}. Each event is pushed onto the cluster-level
  event queue. (Addresses deviation D4: separate ClusterEvent interface with
  Execute(*ClusterSimulator) avoids coupling to sim.Event's Execute(*Simulator).)
```

```
BC-3: RoutingDecisionEvent Instance Injection
- GIVEN a RoutingDecisionEvent executing for a request targeting instance I
- WHEN the RoutingDecisionEvent executes
- THEN the request MUST be injected into instance I's event queue as an ArrivalEvent
  with timestamp = RoutingDecisionEvent.time, and the request MUST appear in instance
  I's Metrics.Requests map
- MECHANISM: RoutingDecisionEvent.Execute() calls a new online injection method on
  InstanceSimulator (bypassing the hasRun panic guard from D5) which delegates to
  sim.InjectArrival(). The instance-level ArrivalEvent proceeds through the existing
  QueuedEvent → StepEvent pipeline unchanged.
```

```
BC-4: Cluster Event Ordering
- GIVEN cluster-level and instance-level events with the same timestamp T
- WHEN the shared-clock event loop selects the next event to process
- THEN cluster-level events MUST be processed before instance-level events at the
  same timestamp, and among cluster events at the same timestamp, ordering MUST be
  ClusterArrival (priority 0) < Admission (priority 1) < Routing (priority 2)
- MECHANISM: The cluster event loop maintains a cluster-level min-heap ordered by
  (timestamp, type_priority, event_id). At each tick, cluster events at time T are
  fully drained before any instance event at time T is processed. Type priorities
  follow macro plan E.3: ClusterArrival=0, Admission=1, Routing=2, instance events>=3.
```

### 1.2 InstanceSnapshot and SnapshotProvider

```
BC-5: InstanceSnapshot Immutability and Timestamp
- GIVEN an InstanceSimulator with ID="instance_0" at clock time T
- WHEN a snapshot is captured via SnapshotProvider.Snapshot(id, clock)
- THEN the returned InstanceSnapshot MUST be an immutable value type (struct, not
  pointer) with Timestamp == T, and subsequent state changes to the InstanceSimulator
  MUST NOT alter the previously returned snapshot's field values
- MECHANISM: InstanceSnapshot is a plain Go struct (value semantics). The snapshot
  capture method reads InstanceSimulator fields, copies values into a new
  InstanceSnapshot{Timestamp: clock, ...}, and returns it by value. No pointers or
  slices are shared with the source.
```

```
BC-6: CachedSnapshotProvider Refresh Behavior
- GIVEN a CachedSnapshotProvider with ObservabilityConfig where QueueDepth.Mode=Immediate
  and KVUtilization.Mode=Periodic with Interval=1000 ticks
- WHEN Snapshot(id, clock=500) is called, then the instance's queue depth changes, then
  Snapshot(id, clock=600) is called again
- THEN the second call MUST return the updated QueueDepth (Immediate mode re-reads on
  every access) but MUST return the same KVUtilization as the first call (Periodic mode:
  1000-tick interval not yet elapsed)
- MECHANISM: CachedSnapshotProvider maintains per-instance, per-field last-refresh
  timestamps. On Snapshot(), each field checks (clock - lastRefresh >= interval) for
  Periodic, always re-reads for Immediate, and only re-reads on explicit RefreshAll()
  for OnDemand.
```

```
BC-7: SnapshotProvider Default Configuration
- GIVEN no explicit ObservabilityConfig provided (default configuration)
- WHEN any SnapshotProvider method is called
- THEN all observable fields (QueueDepth, BatchSize, KVUtilization, FreeKVBlocks)
  MUST use Immediate update mode, meaning every Snapshot() call returns fresh values
- MECHANISM: Default ObservabilityConfig initializes all FieldConfig entries with
  Mode=Immediate. This ensures zero-surprise behavior for users who don't configure
  staleness — matching PR3's implicit behavior of always-current state.
```

### 1.3 Observation Methods

```
BC-8: InstanceSimulator Observation Methods
- GIVEN an InstanceSimulator wrapping a Simulator with WaitQ containing 3 requests,
  RunningBatch containing 5 requests, KVCache with TotalBlocks=100 and UsedBlockCnt=40
- WHEN QueueDepth(), BatchSize(), KVUtilization(), and FreeKVBlocks() are called
- THEN QueueDepth() MUST return 3 (len(WaitQ.queue)), BatchSize() MUST return 5
  (len(RunningBatch.Requests) or 0 if RunningBatch is nil per simulator.go:582),
  KVUtilization() MUST return 0.4 (UsedBlockCnt/TotalBlocks as float64),
  and FreeKVBlocks() MUST return 60 (TotalBlocks - UsedBlockCnt)
- MECHANISM: Each method delegates to the wrapped sim.Simulator's public fields.
  QueueDepth() returns len(sim.WaitQ.queue) (requires D1: adding Len() to WaitQueue).
  BatchSize() returns len(sim.RunningBatch.Requests) with nil guard for RunningBatch.
  KVUtilization() returns float64(sim.KVCache.UsedBlockCnt) / float64(sim.KVCache.TotalBlocks).
  FreeKVBlocks() returns sim.KVCache.TotalBlocks - sim.KVCache.UsedBlockCnt.
```

### 1.4 AdmissionPolicy

```
BC-9: AlwaysAdmit Policy
- GIVEN an AdmissionPolicy configured as "always-admit" and any request R
- WHEN Admit(request, snapshots) is called
- THEN it MUST return (admitted=true, reason="") for every request regardless of
  cluster state
- MECHANISM: AlwaysAdmit is a stateless struct implementing the AdmissionPolicy
  interface. Its Admit() method unconditionally returns true. This is the default
  policy matching PR3's implicit all-admit behavior.
```

```
BC-10: TokenBucket Admission Policy
- GIVEN an AdmissionPolicy configured as "token-bucket" with bucket_size=B and
  refill_rate=R tokens/second, and the bucket currently has T tokens remaining
- WHEN Admit(request, snapshots) is called for a request with input_token_count=C
- THEN if T >= C, the request MUST be admitted and the bucket MUST be decremented by C;
  if T < C, the request MUST be rejected
- MECHANISM: TokenBucket maintains internal state: current_tokens (float64),
  last_refill_time (int64). On each Admit() call, it first refills:
  elapsed = (clock - last_refill_time), tokens += elapsed * refill_rate / 1e6,
  capped at bucket_size. Then checks current_tokens >= request token cost.
  TokenBucket is deterministic (no RNG needed — pure arithmetic on clock).
```

### 1.5 Backward Compatibility

```
BC-11: Default Configuration Backward Compatibility
- GIVEN a ClusterSimulator with default configuration (admission-policy="always-admit",
  admission-latency=0, routing-latency=0, routing=round-robin) and the same seed,
  workload, and hardware config as a PR3 simulation
- WHEN Run() completes
- THEN the AggregatedMetrics MUST be identical to PR3's output for the same inputs:
  same CompletedRequests, same RequestTTFTs (per-request), same RequestE2Es
  (per-request), same RequestITLs (per-request), and same deterministic ordering
- MECHANISM: With zero latency and always-admit, ClusterArrivalEvent at time T
  immediately chains to AdmissionDecisionEvent at T then RoutingDecisionEvent at T.
  Cluster events at time T are processed before instance events at time T (BC-4),
  so the instance-level ArrivalEvent is injected at time T and processes identically
  to PR3's pre-dispatch. Round-robin assignment order matches PR3's i%N dispatch.
```

---

## 2. Negative Contracts

```
NC-1: No Pre-Dispatch of Requests Before Event Loop
- GIVEN a ClusterSimulator with any configuration
- WHEN Run() is called
- THEN requests MUST NOT be injected into instance event queues before the shared-clock
  event loop begins. All requests MUST enter instances only via the
  ClusterArrivalEvent → AdmissionDecisionEvent → RoutingDecisionEvent pipeline
  during the event loop
- MECHANISM: Run() replaces the current batch dispatch loop (cluster.go:81-88) with
  ClusterArrivalEvent scheduling. The only path from request generation to instance
  injection is through the cluster event pipeline. This is verified by checking that
  instance event queues are empty before the event loop starts (after request
  generation).
```

```
NC-2: No State Leakage Between Snapshots
- GIVEN two consecutive Snapshot() calls returning snap1 and snap2
- WHEN the underlying InstanceSimulator state changes between the two calls
- THEN snap1's field values MUST NOT be modified by either the state change or the
  snap2 capture. snap1 and snap2 MUST be independent value copies
- MECHANISM: InstanceSnapshot is a struct returned by value (BC-5). No internal
  pointers, slices, or maps are shared between the snapshot and the source
  InstanceSimulator. The Extended map (if present) is deep-copied during capture.
```

```
NC-3: No Determinism Regression
- GIVEN any ClusterSimulator configuration with a fixed seed
- WHEN Run() is executed K times (K >= 2) with identical inputs
- THEN all K runs MUST produce bit-identical AggregatedMetrics (same maps, same values,
  same ordering in slice fields)
- MECHANISM: All new cluster event types use deterministic tie-breaking (BC-4).
  AdmissionPolicy implementations are purely deterministic (no RNG in AlwaysAdmit;
  TokenBucket uses only clock arithmetic). RoutingDecisionEvent uses round-robin with
  deterministic counter. No goroutines, no map iteration for ordering decisions.
```

---

## 3. Error Handling Contracts

```
EC-1: Invalid Admission Policy Name
- GIVEN an admission-policy flag value that is not "always-admit" or "token-bucket"
- WHEN the ClusterSimulator is constructed
- THEN construction MUST fail with a clear error message listing the invalid policy name
  and the set of valid policy names, before any simulation begins
- MECHANISM: A policy registry (map[string]constructor) validates the policy name
  during ClusterSimulator construction. Unknown names produce a panic or error with
  format: "unknown admission policy %q; valid policies: [always-admit, token-bucket]".
```

```
EC-2: TokenBucket Rejection Handling
- GIVEN a TokenBucket admission policy that rejects a request (bucket tokens < request cost)
- WHEN the AdmissionDecisionEvent executes and Admit() returns false
- THEN the request MUST NOT be routed to any instance, MUST NOT appear in any instance's
  event queue or metrics, and the rejection MUST be recorded in cluster-level metrics
  (e.g., a rejected_requests counter or per-request admission status)
- MECHANISM: AdmissionDecisionEvent.Execute() checks the Admit() return value. On
  rejection, no RoutingDecisionEvent is scheduled. The request is logged as rejected
  in cluster-level metrics. The request object is not referenced by any instance.
```

---

## Contract-to-Component Matrix

| Contract | Component | Test Type | Deviation |
|----------|-----------|-----------|-----------|
| BC-1 | ClusterSimulator.Run(), ClusterArrivalEvent | Integration | D5 |
| BC-2 | ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent | Integration | D4 |
| BC-3 | RoutingDecisionEvent, InstanceSimulator | Integration | D5 |
| BC-4 | Cluster event queue, event ordering | Unit | D4 |
| BC-5 | InstanceSnapshot, SnapshotProvider | Unit | — |
| BC-6 | CachedSnapshotProvider, ObservabilityConfig | Unit | — |
| BC-7 | CachedSnapshotProvider defaults | Unit | — |
| BC-8 | InstanceSimulator observation methods | Unit | D1 |
| BC-9 | AlwaysAdmit | Unit | — |
| BC-10 | TokenBucket | Unit | — |
| BC-11 | ClusterSimulator end-to-end | Integration | D4, D5 |
| NC-1 | ClusterSimulator.Run() | Integration | D5 |
| NC-2 | InstanceSnapshot | Unit | — |
| NC-3 | ClusterSimulator determinism | Integration | — |
| EC-1 | Policy registry / ClusterSimulator construction | Unit | — |
| EC-2 | AdmissionDecisionEvent, TokenBucket, cluster metrics | Integration | — |
