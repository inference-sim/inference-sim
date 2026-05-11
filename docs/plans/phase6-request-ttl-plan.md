# Phase 6: Request TTL in Gateway Queue — Implementation Plan

**Goal:** Add optional per-request TTL (time-to-live) to the gateway queue. When enabled, requests that sit in the queue longer than the TTL are expired and removed — matching GIE's `DefaultRequestTTL` behavior.

**Source:** [Issue #1193](https://github.com/inference-sim/inference-sim/issues/1193) (Phase 6 of [#899](https://github.com/inference-sim/inference-sim/issues/899) — GIE flow control parity).

**Closes:** `Fixes #1193`

**PR Size Tier:** Small/Medium (5-6 production files, 1 new event type, 1 new CLI flag, 1 new internal struct type)

## Behavioral Contracts

### BC-1: TTL disabled by default
- GIVEN `--request-ttl` is not set (default 0)
- WHEN any request is enqueued into the gateway queue
- THEN no TTL event is scheduled, requests wait indefinitely (same behavior as today)

### BC-2: TTL expiry removes queued request
- GIVEN `--request-ttl 5000` (5ms) and a request enqueued at tick 100
- WHEN the clock reaches tick 5100 and the request is still in the queue
- THEN the request is removed from the queue, `gatewayExpired` counter increments by 1, and INV-1 conservation still holds

### BC-3: Dispatched before TTL is a no-op
- GIVEN `--request-ttl 5000` and a request enqueued at tick 100
- WHEN the request is dispatched at tick 200 (before TTL fires at tick 5100)
- THEN the TTL event at tick 5100 is a no-op (request is no longer in the queue)

### BC-4: TTL expiry counts in ShedByTier
- GIVEN `--request-ttl 5000` and a sheddable request with `SLOClass="batch"` expires from the queue
- WHEN the TTL event fires
- THEN `shedByTier["batch"]` increments by 1

### BC-5: TTL expiry triggers dispatch attempt
- GIVEN `--request-ttl 5000` and the queue has both an expiring request and a waiting request
- WHEN the TTL event fires and removes the expired request
- THEN `tryDispatchFromGatewayQueue()` is called (freed slot may allow dispatch)

### BC-6: INV-1 conservation with TTL
- GIVEN `--request-ttl` > 0 and requests expire from the queue
- WHEN simulation ends
- THEN `injected_requests == completed + still_queued + still_running + dropped_unservable + timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed + gateway_queue_rejected + gateway_evicted + gateway_expired`

## Design Overview

**How it works:**

1. New config field `FlowControlRequestTTL` (int64, microseconds, 0=disabled) on `DeploymentConfig`.
2. New CLI flag `--request-ttl` (int, microseconds, default 0).
3. When TTL > 0 and a request is enqueued via `FlowControlAdmission.Admit()`, a `GatewayQueueTTLEvent` is scheduled at `enqueue_time + TTL`.
4. `GatewayQueue` gets a request-ID index (`map[string]requestLocation`) populated on Enqueue, removed on Dequeue/shed/TTL-expiry. This enables O(1) lookup for TTL removal.
5. `GatewayQueue.RemoveByRequestID(id)` removes a specific request. Returns the request (for shedByTier accounting) or nil if not found (already dispatched/shed — no-op).
6. New `GatewayQueueTTLEvent` (priority 6, after eviction at 5): looks up request in queue → if found, removes it, increments `gatewayExpired` + `shedByTier`, calls `tryDispatchFromGatewayQueue()`. If not found, no-op.
7. New `gatewayExpired` counter on `ClusterSimulator`, exposed via `GatewayExpired()` accessor. Added to INV-1 conservation equation.
8. `ProgressSnapshot` gets `GatewayExpired` field. `RawMetrics` gets `GatewayExpired` field.

**INV-1 bucket semantics:** `gateway_expired` is a **terminal bucket**, mutually exclusive with `gateway_queue_shed`, `gateway_queue_rejected`, and `gateway_evicted`. A request lands in exactly one terminal bucket. `shedByTier` is an **aggregate counter** across all removal types (admission rejections + queue shed + in-flight evictions + TTL expiry) — it does NOT appear in the INV-1 equation.

**Tenant tracking:** When a request expires from the gateway queue, it was never dispatched — `tenantTracker.OnStart()` was never called. Therefore `GatewayQueueTTLEvent.Execute()` must NOT call `tenantTracker.OnComplete()`.

**Terminal state:** TTL-expired requests never reach any instance. No `Request.State` transition is needed — the request is simply removed from the gateway queue and counted in `gateway_expired` + `shedByTier`.

**What NOT to do:**
- No per-request TTL field on `Request` — all requests in a simulation share the same TTL (matches GIE's controller-level default).
- No EDF ordering — that's Phase 7.
- No changes to instance-level `Deadline`/`TimeoutEvent` — that's a separate, existing mechanism.

## Tasks

### Task 1: Request-ID index on GatewayQueue (BC-3 enabler)

**Files:** modify `sim/cluster/gateway_queue.go`, test `sim/cluster/gateway_queue_test.go`

**Test:**
```go
func TestGatewayQueue_RemoveByRequestID(t *testing.T) {
    pm := sim.DefaultSLOPriorityMap()
    q := NewGatewayQueue("fifo", 0, pm)

    // Enqueue 3 requests.
    r1 := &sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1"}
    r2 := &sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1"}
    r3 := &sim.Request{ID: "r3", SLOClass: "batch", TenantID: "t2"}
    q.Enqueue(r1, 1)
    q.Enqueue(r2, 2)
    q.Enqueue(r3, 3)

    if q.Len() != 3 {
        t.Fatalf("expected 3, got %d", q.Len())
    }

    // Remove middle request.
    got := q.RemoveByRequestID("r2")
    if got != r2 {
        t.Fatalf("expected r2, got %v", got)
    }
    if q.Len() != 2 {
        t.Fatalf("expected 2, got %d", q.Len())
    }

    // Remove non-existent (already dispatched/shed) → no-op.
    got = q.RemoveByRequestID("r999")
    if got != nil {
        t.Fatalf("expected nil for missing ID, got %v", got)
    }
    if q.Len() != 2 {
        t.Fatalf("expected 2, got %d", q.Len())
    }

    // Remaining requests dequeue correctly.
    d1 := q.Dequeue()
    d2 := q.Dequeue()
    if d1.ID != "r1" || d2.ID != "r3" {
        t.Fatalf("expected r1, r3; got %s, %s", d1.ID, d2.ID)
    }
    if q.Len() != 0 {
        t.Fatalf("expected 0, got %d", q.Len())
    }
}
```

**Impl:**

Add to `GatewayQueue`:
- `requestIndex map[string]requestLocation` field (initialized in constructor).
- `requestLocation` struct: `{band *priorityBand, flow *flowQueue, idx int}` — but since flow entries shift on dequeue, store `band *priorityBand, tenantID string` and do a linear scan within the flow. Flow queues are short (per-tenant, per-priority), so this is fast.
- Actually, simpler: store just the flow key info and scan the flow. Flows are tiny (typically 1-10 entries per tenant-priority combo).

Concrete approach:
```go
type requestLocation struct {
    bandPriority int
    tenantID     string
}
```

- On `Enqueue`: `q.requestIndex[req.ID] = requestLocation{priority, tenantID}`.
- On `Dequeue`/`dequeueFromBand`: `delete(q.requestIndex, req.ID)`.
- On `removeEntryByIndex` (shed): `delete(q.requestIndex, entry.request.ID)`.
- New `RemoveByRequestID(id string) *sim.Request`: look up in index → find band → find flow → scan flow for matching ID → remove entry using slice shift (NOT `removeEntryByIndex` which panics on non-tail removal), update counters, delete from index. Return request or nil.

**Arbitrary-position removal:** The existing `removeEntryByIndex` panics if `idx != last` (it assumes shed victims are always flow tails). `RemoveByRequestID` must handle removal at any position. Use `append(flow.requests[:i], flow.requests[i+1:]...)` to shift elements, preserving seqID ordering. Do NOT call `removeEntryByIndex`.

Expiry counting is done by the caller (`GatewayQueueTTLEvent.Execute()` on `ClusterSimulator`), not inside `GatewayQueue`. `RemoveByRequestID` is a pure data operation.

**Verify:** `go test ./sim/cluster/... -run TestGatewayQueue_RemoveByRequestID`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): add request-ID index and RemoveByRequestID to GatewayQueue (BC-3)`

### Task 2: GatewayQueueTTLEvent + gatewayExpired counter (BC-2, BC-4, BC-5)

**Files:** modify `sim/cluster/cluster_event.go`, modify `sim/cluster/cluster.go`, test `sim/cluster/gateway_queue_ttl_test.go`

**Test:** Integration tests using `NewClusterSimulator` with flow control + TTL. Uses `newTestDeploymentConfig` helper with flow control fields set, and `newTestRequests` for request generation. The "always saturated" scenario uses `FlowControlDetector: "concurrency"` with `FlowControlMaxConcurrency: 1` and enough requests to saturate, while "never saturated" uses `FlowControlDetector: "never"`.

```go
func newFlowControlTTLConfig(numInstances int, ttlUs int64, detector string) DeploymentConfig {
    cfg := newTestDeploymentConfig(numInstances)
    cfg.FlowControlEnabled = true
    cfg.FlowControlRequestTTL = ttlUs
    cfg.FlowControlDetector = detector
    cfg.FlowControlDispatchOrder = "fifo"
    if detector == "concurrency" {
        cfg.FlowControlMaxConcurrency = 1
    }
    return cfg
}

func TestGatewayQueueTTL_ExpiresQueuedRequest(t *testing.T) {
    // Concurrency detector with max=1: first request fills capacity,
    // second request stays queued and hits TTL.
    cfg := newFlowControlTTLConfig(1, 5000, "concurrency")
    cfg.FlowControlMaxConcurrency = 1
    cfg.SimConfig.Horizon = 100_000
    reqs := []*sim.Request{
        {ID: "r1", ArrivalTime: 0, SLOClass: "standard",
            InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: sim.StateQueued},
        {ID: "r2", ArrivalTime: 100, SLOClass: "batch",
            InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: sim.StateQueued},
    }
    cs := NewClusterSimulator(cfg, reqs, nil)
    mustRun(t, cs)

    if cs.GatewayExpired() != 1 {
        t.Fatalf("expected gatewayExpired=1, got %d", cs.GatewayExpired())
    }
    // INV-1 conservation: r1 completed, r2 expired.
    // shedByTier should include the expired request.
    shed := cs.ShedByTier()
    if shed["batch"] != 1 {
        t.Fatalf("expected shedByTier[batch]=1, got %d", shed["batch"])
    }
}

func TestGatewayQueueTTL_NoOpWhenDispatched(t *testing.T) {
    // Never-saturated detector: all requests dispatch immediately.
    // TTL events fire later but find requests already dispatched → no-op.
    cfg := newFlowControlTTLConfig(1, 5000, "never")
    cfg.SimConfig.Horizon = 100_000
    reqs := newTestRequests(3)
    cs := NewClusterSimulator(cfg, reqs, nil)
    mustRun(t, cs)

    if cs.GatewayExpired() != 0 {
        t.Fatalf("expected gatewayExpired=0 (all dispatched before TTL), got %d", cs.GatewayExpired())
    }
}

func TestGatewayQueueTTL_DisabledByDefault(t *testing.T) {
    // TTL=0 (default): no TTL events, no expirations.
    cfg := newFlowControlTTLConfig(1, 0, "never")
    cfg.SimConfig.Horizon = 100_000
    reqs := newTestRequests(5)
    cs := NewClusterSimulator(cfg, reqs, nil)
    mustRun(t, cs)

    if cs.GatewayExpired() != 0 {
        t.Fatalf("expected gatewayExpired=0 (TTL disabled), got %d", cs.GatewayExpired())
    }
}
```

**Impl:**

1. Add `FlowControlRequestTTL int64` field to `DeploymentConfig` in `sim/cluster/deployment.go` (needed for test helper above).
2. Add `gatewayExpired int` and `requestTTL int64` fields to `ClusterSimulator` struct.
3. Wire `requestTTL` from `config.FlowControlRequestTTL` in `NewClusterSimulator`.
4. Add `GatewayExpired() int` accessor method.
4. New event type `GatewayQueueTTLEvent`:
   ```go
   type GatewayQueueTTLEvent struct {
       time      int64
       requestID string
   }
   func (e *GatewayQueueTTLEvent) Timestamp() int64 { return e.time }
   func (e *GatewayQueueTTLEvent) Priority() int     { return 6 }
   ```
   `Execute`: call `cs.gatewayQueue.RemoveByRequestID(e.requestID)` → if non-nil, increment `cs.gatewayExpired`, increment `cs.shedByTier[tier]`, call `cs.tryDispatchFromGatewayQueue()`.

5. In `FlowControlAdmission.Admit()`: after successful enqueue (outcome == Enqueued or ShedVictim), if TTL > 0, return the TTL so the caller can schedule the event. Or better: pass TTL into `FlowControlAdmission` at construction and have `Admit()` schedule the event by returning additional info.

   **Simplest approach:** Add a `TTLSchedule() (requestID string, fireAt int64)` method that returns what to schedule after Admit. The caller (AdmissionDecisionEvent) checks if non-zero and pushes the TTL event. This keeps event scheduling in the event layer (not in the admission policy).

   Actually even simpler: store TTL on `ClusterSimulator.requestTTL`. In `AdmissionDecisionEvent.Execute()`, after the flow control enqueue block, if `cs.requestTTL > 0` and outcome is Enqueued or ShedVictim (request was accepted into queue), schedule a `GatewayQueueTTLEvent` at `cs.clock + cs.requestTTL` for the request ID.

**Verify:** `go test ./sim/cluster/... -run TestGatewayQueueTTL`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): add GatewayQueueTTLEvent for request expiry in gateway queue (BC-2, BC-4, BC-5)`

### Task 3: Config, CLI flag, INV-1, observability (BC-1, BC-6)

**Files:** modify `sim/cluster/deployment.go`, modify `cmd/root.go`, modify `sim/progress_hook.go`, modify `sim/cluster/metrics.go`, modify `sim/cluster/cluster.go`

**Test:**
```go
func TestGatewayQueueTTL_INV1Conservation(t *testing.T) {
    // Run a full simulation with flow control + TTL + always-saturated detector.
    // Verify the INV-1 conservation equation holds with the new gateway_expired bucket.
}
```

**Impl:**

1. `cmd/root.go`: add `--request-ttl` flag (int, default 0, help: "Gateway queue request TTL in microseconds (0=disabled, default 0). Requires --flow-control."). Validate >= 0. Warn if set without `--flow-control`.
2. Wire `--request-ttl` value into `DeploymentConfig.FlowControlRequestTTL` (field added in Task 2).
3. `ProgressSnapshot`: add `GatewayExpired int` field.
5. `RawMetrics`: add `GatewayExpired int` field.
6. Wire `GatewayExpired` into snapshot building and metrics collection (same pattern as `GatewayEvicted`).
7. Update INV-1 verification: add `gateway_expired` to the conservation equation in output.
8. Update `invariants.md` documentation.

**Verify:** `go test ./... -count=1`
**Lint:** `golangci-lint run ./...`
**Commit:** `feat(admission): add --request-ttl flag and wire INV-1 + observability (BC-1, BC-6)`

### Task 4: Documentation

**Files:** modify `docs/contributing/standards/invariants.md`, modify `CLAUDE.md`

**Impl:**

1. Update INV-1 in `invariants.md`: add `+ gateway_expired` to the conservation equation and describe the bucket.
2. Update CLAUDE.md INV-1 working copy to match.
3. Update issue #899 tracking table (mark Phase 6 done — defer to after merge).

**Verify:** `go test ./... -count=1` (ensure no regressions)
**Lint:** `golangci-lint run ./...`
**Commit:** `docs(invariants): add gateway_expired bucket to INV-1 conservation equation`

## Sanity Checklist

- [ ] **R1 (silent continue):** No silent `continue` — TTL no-op is an explicit `return` with no counter increment.
- [ ] **R2 (determinism/INV-6):** TTL events use `(timestamp, priority, seqID)` ordering. No map iteration for output. `requestIndex` is only used for lookup, not iteration.
- [ ] **R3 (factory validation):** CLI validates `--request-ttl >= 0`. Config field is plain int64 (0=disabled, no ambiguity).
- [ ] **R4 (canonical constructor):** `ClusterSimulator` construction site updated to wire `requestTTL`.
- [ ] **R8 (exported mutable maps):** `requestIndex` is unexported on `GatewayQueue`.
- [ ] **INV-1:** Conservation equation extended with `gateway_expired` bucket. ShedByTier conservation: `sum(ShedByTier) == RejectedRequests + GatewayQueueShed + GatewayEvicted + GatewayExpired`.
- [ ] **INV-5 (causality):** TTL event fires at `enqueue_time + TTL` ≥ `enqueue_time`. No clock regression.
- [ ] **INV-9 (oracle boundary):** TTL expiry does not read `OutputTokens`.
- [ ] **Tenant tracking:** TTL expiry does NOT call `tenantTracker.OnComplete()` — request was never dispatched.
