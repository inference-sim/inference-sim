# Data Model: Deferred Queue for Batch and Background Requests

## Entities

### DeferredQueue (slice on ClusterSimulator)

A FIFO ordered slice of `*sim.Request` objects representing Batch/Background requests that have arrived but cannot yet be admitted because the cluster is busy.

**Ownership**: `ClusterSimulator.deferredQueue []*sim.Request`
**Initialization**: nil (Go zero value; safe for `len()`, `append()`, and slice truncation)
**Lifecycle**: Requests enter via `deferredQueue = append(deferredQueue, req)` in `AdmissionDecisionEvent.Execute()`. Requests leave via `promoteDeferred()` which injects them as `ClusterArrivalEvent`s and truncates the slice.

**Invariants**:
- A request in `deferredQueue` has `SLOClass == "batch"` or `SLOClass == "background"`.
- A request in `deferredQueue` has not been counted as rejected (`rejectedRequests` is not incremented).
- At simulation end, `len(deferredQueue)` equals `DeferredHorizonInterrupted` in `RawMetrics`.

---

### ClusterBusyState (derived, not stored)

A transient boolean derived by `isBusy()` — not stored as a field. True when any instance has non-zero combined effective load.

**Derivation**: `∃ inst ∈ instances where inst.State != InstanceStateTerminated: QueueDepth(inst) + BatchSize(inst) + inFlightRequests[inst.ID()] > 0`
**Empty cluster**: returns false (a cluster with no instances is idle — requests are admitted normally).

---

### DeferredHorizonInterrupted (metric field)

An integer count added to `RawMetrics` representing requests still in `deferredQueue` at simulation horizon.

**Field**: `RawMetrics.DeferredHorizonInterrupted int`
**Population**: `rawMetrics.DeferredHorizonInterrupted = cs.DeferredQueueLen()` in `cmd/root.go` and `cmd/replay.go` after `CollectRawMetrics()` returns.
**INV-1 role**: Extends conservation equation: `injected == completed + running + queued + shed + dropped + timed_out + deferred_horizon_interrupted`.

---

## State Transitions for a Batch/Background Request

```
Arrival (ClusterArrivalEvent)
  └─► AdmissionDecisionEvent fires
         │
         ├─ isBusy() == true  ──► deferredQueue.append(req)   [held, not rejected]
         │                         └─► isBusy() == false later ──► promoteDeferred()
         │                                                           └─► ClusterArrivalEvent re-injected
         │                                                                └─► ... (back to top)
         │
         ├─ isBusy() == false ──► admissionPolicy.Admit()
         │                        ├─ admitted=true  ──► RoutingDecisionEvent
         │                        └─ admitted=false ──► rejectedRequests++ (e.g., reject-all policy)
         │
         └─ Horizon reached while in deferredQueue ──► DeferredHorizonInterrupted = len(deferredQueue) [post-Run]
```

## Relationships

- `deferredQueue` is owned exclusively by `ClusterSimulator`; no other component reads or writes it.
- `isBusy()` reads `c.instances` (slice) and `c.inFlightRequests` (map) — both owned by `ClusterSimulator`.
- `promoteDeferred()` writes to `c.clusterEvents` (heap) — same ownership scope.
- `DeferredQueueLen()` is a read-only accessor; callers in `cmd/` may only call it after `Run()` returns (`hasRun` guard).
