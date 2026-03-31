# Research: Deferred Queue for Batch and Background Requests

## Decision 1: Placement of the deferral intercept

**Decision**: The pre-admission intercept fires inside `AdmissionDecisionEvent.Execute()`, BEFORE calling `buildRouterState()` or `admissionPolicy.Admit()`.

**Rationale**: A deferred request is neither admitted nor rejected — it is held. Calling `buildRouterState()` is unnecessary (no routing decision is made), and calling `admissionPolicy.Admit()` is incorrect (would count as an admission decision and potentially emit a trace record). Intercepting first is the minimal path.

**Alternatives considered**:
- Inside `admissionPolicy.Admit()`: rejected — admission policies are stateless and cannot push to `deferredQueue` (INV-9, no side effects on state).
- A new `DeferralDecisionEvent` type: rejected — adds event-queue complexity for a simple state check; the deferral condition is a pure function of SLOClass and cluster busy-ness.

---

## Decision 2: Definition of "cluster busy"

**Decision**: `isBusy()` returns true when any instance has `QueueDepth() + BatchSize() + inFlightRequests[instID] > 0`. An instance pool of size zero is not busy (returns false).

**Rationale**: This three-component definition matches the definition already used for `RoutingSnapshot.EffectiveLoad()` in `sim/admission.go`. It accounts for (a) requests waiting in the WaitQ (`QueueDepth`), (b) requests in the current batch being stepped (`BatchSize`), and (c) requests dispatched but not yet completed (`inFlightRequests`). Using the same three components keeps the busy definition consistent with the existing load signal.

**Alternatives considered**:
- `inFlightRequests` only: rejected — doesn't capture requests still in the queue that haven't been dispatched yet.
- `EffectiveLoad()` on snapshots: rejected — snapshots may be stale (INV-7 tiered freshness); `isBusy()` must read live instance state for correctness.
- `QueueDepth() + BatchSize()` without `inFlightRequests`: rejected — misses dispatched-but-not-yet-confirmed requests during the dispatch-to-completion window.

---

## Decision 3: Promotion timing — after every event vs. after instance completions only

**Decision**: The idle-capacity check (`if len(c.deferredQueue) > 0 && !c.isBusy()`) is placed at the end of the main event loop body, firing after BOTH cluster events and instance events.

**Rationale**: Placing the check after every event ensures minimum latency from idle detection to promotion. Restricting to instance events only would miss the edge case where the last work finishes via a cluster-side event (e.g., a timeout or drop counted at the cluster level). The check is O(N_instances) which is negligible.

**Alternatives considered**:
- After instance events only: rejected — misses cluster-side completions.
- Polling at a fixed interval: rejected — introduces artificial latency and complicates determinism.
- Event-driven notification from instances: rejected — over-engineering; simple poll-after-event is sufficient.

---

## Decision 4: Promotion is atomic (all-at-once, not one-at-a-time)

**Decision**: `promoteDeferred()` injects ALL deferred requests as `ClusterArrivalEvent`s at `c.clock`, then truncates `deferredQueue` to length 0.

**Rationale**: Atomic promotion is simpler to reason about and avoids partial-promotion states. If only some requests were promoted and the cluster became busy again mid-promotion, the remaining requests would re-defer at their next admission decision — which is correct behavior. Atomic promotion is also consistent with how arrivals are injected: all in one pass.

**Alternatives considered**:
- One-at-a-time promotion: rejected — complicates the state machine; re-deferral handles the "cluster refills immediately" case naturally regardless.

---

## Decision 5: DeferredHorizonInterrupted wiring via accessor (not CollectRawMetrics parameter)

**Decision**: Add `DeferredQueueLen() int` accessor to `ClusterSimulator` (gated on `hasRun`), assign `rawMetrics.DeferredHorizonInterrupted = cs.DeferredQueueLen()` in `cmd/root.go` and `cmd/replay.go` after `CollectRawMetrics`. This extends the pattern established by `ShedByTier()` in Phase 1B-1a.

**Rationale**: Adding a new parameter to `CollectRawMetrics` would require updating all call sites (two in `cmd/`) anyway; the accessor pattern achieves the same result with a cleaner separation. `CollectRawMetrics` should not grow its parameter list for each new derived metric.

**Alternatives considered**:
- Add parameter to `CollectRawMetrics`: rejected — grows the function signature monotonically for each new metric; accessor pattern is established precedent.
- Store in `RawMetrics` inside `CollectRawMetrics` via a passed-in `*ClusterSimulator`: rejected — creates a dependency from metrics.go back to the cluster struct, breaking separation of concerns.

---

## Decision 6: No trace record for deferred requests

**Decision**: `AdmissionDecisionEvent.Execute()` does not emit a `trace.AdmissionRecord` when deferring a request.

**Rationale**: Deferred requests are in a transitional state — they will re-enter the admission path when promoted. Recording an admission decision (admitted=false) would misrepresent the outcome and confuse trace consumers. A future PR can add a dedicated `DeferralRecord` trace type if needed.

**Alternatives considered**:
- Emit `AdmissionRecord{Admitted: false, Reason: "deferred"}`: rejected — semantically incorrect; deferred ≠ rejected.

---

## R4 Check: ClusterSimulator struct literal construction sites

**Finding**: One construction site at `NewClusterSimulator()` in `sim/cluster/cluster.go:150`. No other literal construction sites found. The new `deferredQueue` field must be initialized there with `deferredQueue: nil` (or omitted — zero value for slice is nil, which is safe for `len()` and `append()`).

## INV-1 Extended Form

The deferred queue adds a new accounting category:

```
injected == completed + running + queued + shed + dropped + timed_out + deferred_horizon_interrupted
```

Where `deferred_horizon_interrupted = len(cs.deferredQueue)` at simulation end (after `Run()` returns, before `deferredQueue` is cleared).
