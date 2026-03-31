# Contract: DeferredQueue

**Package**: `sim/cluster`
**File**: `sim/cluster/cluster.go`
**Type**: Internal field + methods on `ClusterSimulator`

## Behavioral Contract

### isBusy() bool

**Signature**: `func (c *ClusterSimulator) isBusy() bool`

**Preconditions**:
- May be called at any point during or after `Run()`.
- `c.instances` may be empty (zero-length slice) — must not panic.

**Postconditions**:
- Returns `true` iff `∃ inst ∈ c.instances where inst.State != InstanceStateTerminated: inst.QueueDepth() + inst.BatchSize() + c.inFlightRequests[string(inst.ID())] > 0`.
- Returns `false` when `c.instances` is empty (zero-length cluster is not busy).
- Pure read: no mutation of `c`, `c.instances`, or `c.inFlightRequests`.
- INV-9 compliant: does not read `req.OutputTokens` (receives no request parameter).

---

### promoteDeferred()

**Signature**: `func (c *ClusterSimulator) promoteDeferred()`

**Preconditions**:
- Called only when `len(c.deferredQueue) > 0 && !c.isBusy()`.
- Called only inside the `Run()` event loop.

**Postconditions**:
- All requests from `c.deferredQueue` are injected as `ClusterArrivalEvent{time: c.clock}` into `c.clusterEvents`.
- `c.deferredQueue` is truncated to length 0 (capacity preserved).
- `c.nextSeqID()` is called once per promoted request for deterministic event ordering.
- No mutation of the `*sim.Request` structs — requests are re-injected as-is.

**Side effects**: Writes to `c.clusterEvents` (heap push); clears `c.deferredQueue`.

**Re-deferral note**: With non-zero admission latency, standard traffic arriving in the `[clock, clock+admissionLatency]` window may make `isBusy()` return `true` before a promoted request reaches `AdmissionDecisionEvent`, causing it to be re-deferred. This is intentional (Decision 4 in `research.md`) but callers should be aware that `DeferredHorizonInterrupted` may be inflated under continuous light standard load.

---

### Pre-admission deferral intercept (in AdmissionDecisionEvent.Execute)

**Location**: `sim/cluster/cluster_event.go`, beginning of `AdmissionDecisionEvent.Execute()`

**Preconditions**:
- `e.request` is non-nil with `SLOClass` set.
- Fires BEFORE `buildRouterState()` and `admissionPolicy.Admit()`.

**Postconditions**:
- When `(e.request.SLOClass == "batch" || e.request.SLOClass == "background") && cs.isBusy()`:
  - `cs.deferredQueue = append(cs.deferredQueue, e.request)` — request is held.
  - Returns immediately (no admission decision, no rejection counter increment, no trace record).
- When `SLOClass` is neither `"batch"` nor `"background"`: intercept does not fire; normal admission proceeds.
- When `SLOClass` is `"batch"` or `"background"` but `!cs.isBusy()`: intercept does not fire; request admitted normally.
- INV-9 compliant: does not read `e.request.OutputTokens`.

---

### DeferredQueueLen() int

**Signature**: `func (c *ClusterSimulator) DeferredQueueLen() int`

**Preconditions**:
- Must be called after `Run()` completes (`c.hasRun == true`).
- Calling before `Run()` panics with a descriptive message.

**Postconditions**:
- Returns `len(c.deferredQueue)` — the count of requests still deferred at simulation end.
- Pure read: no mutation.

---

## Idle-Capacity Check in Run()

**Location**: End of the main event loop body in `ClusterSimulator.Run()`, after each event is processed.

**Contract**:
```
After processing each event:
  if len(c.deferredQueue) > 0 && !c.isBusy() {
      c.promoteDeferred()
  }
```

**Invariant preserved**: INV-8 (work-conserving) — when the cluster becomes idle and deferred requests are waiting, promotion fires at the very next event processing step, injecting them as `ClusterArrivalEvent`s. The deferred requests become active arrivals within the same simulation tick.

---

## INV-1 Extension

The deferred queue adds a new terminal category to request conservation:

```
injected == completed + still_running + still_queued + shed + dropped_unservable + timed_out + deferred_horizon_interrupted
```

Where `deferred_horizon_interrupted = DeferredQueueLen()` (requests still in deferredQueue when horizon is reached).

Verified by: `TestDeferredQueue_INV1_Conservation` invariant test.
