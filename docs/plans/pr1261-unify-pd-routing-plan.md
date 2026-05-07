# PR #1261: Unify PD and Standard Routing into a Single Code Path (GAP-1)

**Goal:** Eliminate the `poolsConfigured()` fork at both dispatch sites by merging PD disaggregation routing into `RoutingDecisionEvent`. Delete `DisaggregationDecisionEvent`.

**Source:** Issue #1261 — https://github.com/inference-sim/inference-sim/issues/1261

**Closes:** #1261

**Tier:** Medium (2 files changed; behavioral refactor with event-type deletion).

---

## Executive Summary

Today two call sites — `AdmissionDecisionEvent.Execute` (`cluster_event.go:233`) and `ClusterSimulator.tryDispatchFromGatewayQueue` (`cluster.go:1332`) — each branch on `poolsConfigured()` and schedule **either** `DisaggregationDecisionEvent` **or** `RoutingDecisionEvent`. The disagg event duplicates warm-up tracking, trace recording, in-flight counting, and instance injection already present in the non-disagg event.

This PR collapses the fork: both call sites schedule `RoutingDecisionEvent` unconditionally at `time + routingLatency`, and `RoutingDecisionEvent.Execute` branches internally via two private helpers `executeStandardRouting` and `executeDisaggregatedRouting`. `DisaggregationDecisionEvent` is deleted.

Parity target: llm-d-inference-scheduler `disagg-profile-handler` @ `71e39f2`, where both disaggregated and non-disaggregated requests share one entry point with an internal branch.

---

## Behavioral Contracts

**BC-1 (Single-entry routing):** `RoutingDecisionEvent.Execute` is the sole entry point for post-admission routing. Regardless of whether pools are configured, it produces exactly one `trace.RoutingRecord` per request (disagg path also produces a `DisaggregationRecord`).

**BC-2 (Timing preservation):** For every request, the effective injection time (wall clock at which the selected decode/target instance receives `InjectRequestOnline`) is unchanged: `admission_time + routingLatency`. For disaggregated requests, `PrefillRoutingEvent` is scheduled at `admission_time + routingLatency`, unchanged.

**BC-3 (Decode-first ordering):** When `poolsConfigured()`, the decode pod is selected before the disaggregation decision (unchanged — preserved inside `executeDisaggregatedRouting`).

**BC-4 (Non-disaggregated pool targeting):** When `poolsConfigured()` but the disaggregation decider returns `Disaggregate=false`, the request is injected into the decode pool only (not the full instance set) — preserved.

**BC-5 (Observer & counters):** `notifyDisaggregationObserver`, `inFlightRequests[target]++`, `tenantTracker.OnStart`, and warm-up recording fire on both paths.

**BC-6 (Trace compatibility):** `len(trace.Routings)` is unchanged (one record per request). `len(trace.Disaggregations)` is unchanged (one record per request when pools configured). The `Clock` field on disagg-path `RoutingRecord` and `DisaggregationRecord` will now reflect `admission_time + routingLatency` (previously `admission_time`); this makes trace semantics uniform across paths and no existing test asserts the old value.

---

## Timing Adjustment

`DisaggregationDecisionEvent` was scheduled at `e.time` (no `+routingLatency`) and re-added latency at the injection/PrefillRouting sites. `RoutingDecisionEvent` is scheduled at `e.time + cs.routingLatency` and injects at `e.time`. After the merge, the relocated disagg body receives `time = admission_time + routingLatency` from the caller and must drop its internal `+ cs.routingLatency` additions:

- Non-disagg injection: `decodeInst.InjectRequestOnline(req, time)` (was `e.time + cs.routingLatency`)
- `PrefillRoutingEvent.time`: `time` (was `e.time + cs.routingLatency`)

Effective wall-time at which injection / prefill routing happens is unchanged.

---

## Tasks

### T1 — Add regression test for injection-time parity (TDD, fails at first)

File: `sim/cluster/disaggregation_test.go` (append).

Test: `TestPDRouting_UnifiedEntry_InjectionTimingPreserved`.
- Build two configs: one with pools, one without.
- In each, `RoutingLatency = 100_000` (100 ms).
- Arrival at `t=0`. Assert that the running-request's first-seen time (via `InstanceSimulator` or via observable `GatewayDispatchTime` / `AssignedInstance` population + events) is `0 + admissionLatency + routingLatency`, identical to pre-refactor.

Because this test asserts externally-observable behavior (injection time visible via first-step scheduling / trace Clock on `PrefillRoutingRecord`), it survives the refactor and fails neither before nor after when behavior is preserved. I'll write the assertion on the `PrefillRoutingRecord.Clock` (disagg path) and on the instance's first `StepEvent` timestamp (non-disagg path) using trace data.

Actual failing-first test: `TestPDRouting_NoMoreDisaggregationDecisionEventType` — asserts that the event type name no longer exists. This is trivially compile-time checked by deleting the type. Used as a marker test.

### T2 — Remove fork at `AdmissionDecisionEvent.Execute`

File: `sim/cluster/cluster_event.go` (lines 233–251). Replace fork with:
```go
heap.Push(&cs.clusterEvents, clusterEventEntry{
    event: &RoutingDecisionEvent{time: e.time + cs.routingLatency, request: e.request},
    seqID: cs.nextSeqID(),
})
```

### T3 — Remove fork at `tryDispatchFromGatewayQueue`

File: `sim/cluster/cluster.go` (lines 1332–1348). Same replacement.

### T4 — Extract executeStandardRouting + executeDisaggregatedRouting helpers

File: `sim/cluster/cluster.go` — append two private methods on `*ClusterSimulator`:
- `executeStandardRouting(req *sim.Request, time int64)` — body from current `RoutingDecisionEvent.Execute` (no timing change).
- `executeDisaggregatedRouting(req *sim.Request, time int64)` — body from `DisaggregationDecisionEvent.Execute`, with timing adjustments described above.

### T5 — Update RoutingDecisionEvent.Execute to dispatch

```go
func (e *RoutingDecisionEvent) Execute(cs *ClusterSimulator) {
    if cs.poolsConfigured() {
        cs.executeDisaggregatedRouting(e.request, e.time)
    } else {
        cs.executeStandardRouting(e.request, e.time)
    }
}
```

### T6 — Delete DisaggregationDecisionEvent

Remove lines 334–463 of `cluster_event.go`.

### T7 — Update event-priority comment

File: `sim/cluster/cluster_event.go` line 17: change
`// 0=Arrival, 1=Admission, 2=Routing, 3=Disaggregation, 4-6=PD, 8=ScalingTick, 9=ScaleActuation`
to drop `3=Disaggregation`.

### T8 — Refresh stale comments

Rename `DisaggregationDecisionEvent` → `RoutingDecisionEvent` in comments at:
- `sim/cluster/pd_events.go:16,183,220`
- `sim/cluster/parent_request.go:38`
- `sim/trace/record.go:48`
- `sim/cluster/pd_traces_test.go:84,175,298,313` (comments only; test assertions unchanged)
- `sim/cluster/disaggregation_test.go:592,659,1753,1787,1824` (comments only)

### T9 — Run tests + lint

```
go test ./sim/cluster/... -count=1
go test ./... -count=1
golangci-lint run ./...
```

### T10 — Commit, push, open PR, post review comment

---

## Sanity Checklist

- [x] INV-1 (conservation): unaffected — same counters incremented in both paths
- [x] INV-3 (clock monotonicity): unaffected — RoutingDecisionEvent always scheduled at `+routingLatency`
- [x] INV-6 (determinism): event priority order preserved — removing event priority 3 is safe because no concurrent event at priority 3 existed; only Disaggregation used it
- [x] INV-7 (signal freshness): unchanged — snapshot construction untouched
- [x] Warm-up tracking preserved on both paths
- [x] `notifyDisaggregationObserver` fires on both paths
- [x] `tenantTracker.OnStart` fires on both paths
- [x] R4 (canonical constructors): `RoutingDecisionEvent{}` struct literal site count unchanged; `DisaggregationDecisionEvent{}` deleted
- [x] `blis run` + `blis replay` affected equally (shared DES kernel); `blis observe` unaffected (HTTP client, no event pipeline)
