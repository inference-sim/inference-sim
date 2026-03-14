# Client Behavior Model: Request Timeouts + Completion-Driven Sessions — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add request timeouts and closed-loop session management so BLIS can simulate reasoning workloads that complete in bounded time, with realistic multi-turn arrival patterns.

**The problem today:** BLIS hangs indefinitely on reasoning workloads (#529) because the simulator has no client timeout mechanism — requests stuck in KV preemption spirals live forever. Multi-turn workloads pre-generate all rounds with fixed timing, ignoring that real clients wait for each response before sending the next request.

**What this PR adds:**
1. Request timeouts — every request has a deadline; if the simulator can't complete it in time, the request is cancelled, KV blocks are freed, and the timed-out count increments. This bounds preemption spirals.
2. Closed-loop sessions — multi-turn reasoning workloads generate only round-0 upfront; subsequent rounds are generated on-demand when the previous round completes, with realistic timing.
3. Session cancellation — when a round times out, the entire session is cancelled (no more rounds generated).
4. Event priority ordering — the per-instance event queue gains `(timestamp, priority, seqID)` ordering, making same-timestamp event processing deterministic (INV-6 improvement).

**Why this matters:** Enables capacity planning for reasoning workloads (the fastest-growing inference pattern). Without this, BLIS cannot model o1/DeepSeek-R1/QwQ-style workloads.

**Architecture:** New `TimeoutEvent` in `sim/event.go` with priority 5 (fires after step events at equal timestamps). `OnRequestDone` callback function on `Simulator` bridges to `SessionManager` in `sim/workload/session.go`. `GenerateWorkload()` in `sim/workload/generator.go` returns session blueprints alongside requests. Cluster mode routes follow-up rounds through the existing `ClusterArrivalEvent` pipeline. Per-instance `EventQueue` upgraded from timestamp-only to `(timestamp, priority, seqID)` ordering.

**Source:** Design doc: `docs/plans/2026-03-13-client-behavior-model-design.md`, GitHub issue #627

**Closes:** Fixes #627

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds client-side behavior modeling to BLIS: request timeouts (bounded request lifetime) and completion-driven sessions (closed-loop multi-turn). The core changes are:

1. **Event queue infrastructure** (sim/): `EventQueue` gains `(timestamp, priority, seqID)` ordering via a wrapper struct. All 5 existing event types get priority constants. New `TimeoutEvent` with priority 5.
2. **Request model** (sim/): New `StateTimedOut` terminal state, `Deadline` field on `Request`.
3. **Simulator timeout handling** (sim/): `TimeoutEvent` execution, `OnRequestDone` callback, `WaitQueue.Remove()`, timeout scheduling in `EnqueueRequest`.
4. **Session manager** (sim/workload/): New `SessionManager` type tracking active sessions, generating follow-up rounds on completion.
5. **Workload generation** (sim/workload/): New `GenerateWorkload()` producing session blueprints alongside requests. `closed_loop` opt-out field on `ClientSpec`.
6. **Cluster integration** (sim/cluster/): `OnRequestDone` callback wired to push follow-ups as `ClusterArrivalEvent`s. `TimedOutRequests` aggregated.
7. **CLI wiring** (cmd/): Switch to `GenerateWorkload`, create `SessionManager`, wire callbacks.
8. **Metrics** (sim/): `TimedOutRequests` counter, INV-1 5-term formula, `MetricsOutput` update.

Adjacent blocks: KV cache (ReleaseKVBlocks on timeout), batch formation (FormBatch sees post-timeout running batch), latency model (unmodified), routing (follow-up rounds routed through pipeline).

No deviations from design doc.

### B) Behavioral Contracts

**Positive contracts:**

**BC-1: Timeout cancels queued request.**
- GIVEN a request with Deadline = ArrivalTime + 5000, enqueued at tick 1000
- WHEN TimeoutEvent fires at tick 6000 and request is still in WaitQ
- THEN request state MUST be `timed_out`, Metrics.TimedOutRequests MUST increment by 1, WaitQ MUST NOT contain the request
- MECHANISM: TimeoutEvent.Execute checks state, sets StateTimedOut, calls WaitQ.Remove, increments counter

**BC-2: Timeout cancels running request.**
- GIVEN a running request in RunningBatch with Deadline = ArrivalTime + 10000
- WHEN TimeoutEvent fires at Deadline tick
- THEN request MUST NOT be in RunningBatch.Requests, KV blocks MUST be released (allocated+free=total), reqNumComputedTokens entry MUST be deleted, state MUST be `timed_out`. If RunningBatch becomes empty and WaitQ.Len() > 0 and stepEvent == nil, a StepEvent MUST be scheduled (INV-8).
- MECHANISM: TimeoutEvent builds new-slice excluding timed-out request, calls ReleaseKVBlocks, deletes computed-token map entry, checks work-conserving property

**BC-3: Timeout no-op for completed request.**
- GIVEN a request that completed at tick 5000 with Deadline = 10000
- WHEN TimeoutEvent fires at tick 10000
- THEN no state change, no metric increment, no KV release
- MECHANISM: TimeoutEvent checks req.State == StateCompleted, returns immediately

**BC-4: INV-1 conservation with timeouts.**
- GIVEN any workload with timeouts
- WHEN simulation ends
- THEN `completed + still_queued + still_running + dropped_unservable + timed_out == injected`
- MECHANISM: Every request reaches exactly one terminal bucket. InjectedRequests in SaveResults updated to 5-term sum.

**BC-5: Dropped-unservable gets no timeout event.**
- GIVEN a request rejected by EnqueueRequest guards
- WHEN the request is dropped
- THEN no TimeoutEvent is scheduled (timeout scheduling happens after guards pass)
- MECHANISM: Timeout scheduling at the end of EnqueueRequest, after all guards

**BC-6: Closed-loop round generation.**
- GIVEN a 3-round session, round 0 completes at tick T, ThinkTimeUs = D
- WHEN OnRequestDone callback fires for round 0
- THEN SessionManager generates round 1 with ArrivalTime = T + D
- MECHANISM: SessionManager.OnComplete looks up session, advances round counter, samples tokens, sets arrival time

**BC-7: Session cancellation on timeout.**
- GIVEN a 5-round session, round 2 times out
- WHEN OnRequestDone callback fires
- THEN SessionManager marks session cancelled, rounds 3-4 MUST NOT be generated
- MECHANISM: OnComplete checks req.State == StateTimedOut, sets session state to cancelled

**BC-8: Context accumulation in closed-loop.**
- GIVEN ContextGrowth = "accumulate", round 0 with input I₀ (100 tokens) and output O₀ (50 tokens)
- WHEN round 0 completes and round 1 is generated
- THEN round 1's InputTokens MUST begin with the 150 accumulated tokens (I₀ + O₀)
- MECHANISM: SessionManager appends round's input+output to contextTokens, prepends to next round

**BC-9: Cluster routing for follow-up rounds.**
- GIVEN cluster mode with 2 instances, round 0 completes on instance_0
- WHEN OnRequestDone callback fires
- THEN round 1 MUST enter ClusterArrivalEvent pipeline (admission → routing → instance injection), NOT be injected directly into instance_0
- MECHANISM: Cluster-mode callback pushes ClusterArrivalEvent into cluster event queue

**BC-10: Determinism.**
- GIVEN same seed and workload spec
- WHEN run twice with timeouts and sessions
- THEN byte-identical stdout output
- MECHANISM: Event priority+seqID ordering, per-session RNG from client RNG, counter-based session IDs

**BC-11: Backward compatibility.**
- GIVEN a non-session workload spec with no explicit timeout, all requests completing within 300s
- WHEN processed by new code
- THEN output MUST be identical to current code
- MECHANISM: Default 300s timeout → TimeoutEvent fires as no-op after completion. Priority ordering preserves existing event sequence (verified via golden dataset).

**BC-12: Completion wins at equal timestamps.**
- GIVEN a StepEvent and TimeoutEvent at the same tick for the same request
- WHEN events are processed
- THEN StepEvent fires first (priority 2 < 5), request completes normally, TimeoutEvent is no-op
- MECHANISM: EventQueue orders by (timestamp, priority, seqID); StepEvent priority=2, TimeoutEvent priority=5

**BC-13: INV-4 conservation after timeout.**
- GIVEN a running request with N KV blocks allocated
- WHEN it times out
- THEN `allocated_blocks + free_blocks == total_blocks` after ReleaseKVBlocks
- MECHANISM: ReleaseKVBlocks iterates RequestMap entries, decrements RefCount, returns to free list

**BC-14: Horizon-interrupted session.**
- GIVEN a 5-round session, rounds 0-2 complete, horizon reached before round 3 completes
- WHEN simulation ends
- THEN round 3 counted as still_queued or still_running in INV-1, rounds 4 never generated
- MECHANISM: Event loop stops at horizon; SessionManager horizon guard (BC-19) prevents phantom requests

**BC-15: Preempt-then-timeout is safe.**
- GIVEN a running request preempted (KV released, moved to WaitQ), then times out while queued
- WHEN TimeoutEvent fires
- THEN KV release is a no-op (zero blocks), request removed from WaitQ, counter increments. No double-free.
- MECHANISM: ReleaseKVBlocks on empty RequestMap entry is safe (Go delete on missing key, empty slice iteration)

**BC-16: Length-capped session continues.**
- GIVEN a session where round N is length-capped at MaxModelLen
- WHEN OnRequestDone callback fires
- THEN session generates round N+1 with accumulated context including truncated output
- MECHANISM: OnComplete sees StateCompleted (length-capped is a completion), generates next round

**BC-17: Dropped-unservable follow-up cancels session.**
- GIVEN a session where round N+1 is dropped by enqueue guards
- WHEN OnRequestDone callback fires from EnqueueRequest drop path
- THEN session marked cancelled
- MECHANISM: EnqueueRequest invokes callback for ALL dropped requests; SessionManager sees the drop and cancels

**BC-18: Queued-timeout work-conserving (defense-in-depth).**
- GIVEN queued request times out when running batch is empty, stepEvent == nil, WaitQ still has items after removal
- WHEN TimeoutEvent fires
- THEN a StepEvent MUST be scheduled at current tick
- MECHANISM: Timeout handler checks stepEvent == nil && WaitQ.Len() > 0 after removal

**BC-19: Follow-up rounds beyond horizon not generated.**
- GIVEN session where round N completes, computed round N+1 arrival > horizon
- WHEN OnRequestDone callback fires
- THEN round N+1 NOT generated, session marked horizon-interrupted
- MECHANISM: SessionManager.OnComplete checks arrival > horizon before generating

**Negative contracts:**

**NC-1: No silent data loss.** Every timed-out request is counted in exactly one INV-1 bucket. No request disappears from accounting.

**NC-2: No performance regression for non-session workloads.** The only hot-path change is the EventQueue comparator (3 integer comparisons vs 1). WaitQ.Remove is never called for non-timeout workloads. SessionManager is not instantiated.

**NC-3: No R21 violation.** RunningBatch removal in timeout handler uses new-slice construction, not in-place modification.

**Error handling contracts:**

**EC-1: Timeout = 0 on ClientSpec treated as default (300s).** Validated at ClientSpec.Validate() time using `*int64` pointer type (nil = use default).

**EC-2: Past-due timeout (deadline ≤ current tick at enqueue).** Request is immediately timed out before entering WaitQ. Input tokens counted. Counted as `timed_out` in INV-1.

### C) Component Interaction

```
Workload Generation                    Simulation Kernel                    Cluster
┌──────────────────┐                ┌──────────────────────┐          ┌──────────────┐
│ GenerateWorkload │─ requests ─────│ InjectArrival        │          │ ClusterSim   │
│                  │─ blueprints ──▶│ ├─ ArrivalEvent      │          │ ├─ Run()      │
│ SessionManager   │◀── callback ──│ ├─ QueuedEvent        │          │ ├─ callback ──│
│ ├─ OnComplete()  │               │ │  └─ EnqueueRequest  │          │ │  pushes     │
│ └─ blueprints    │               │ │     ├─ guards       │          │ │  ClusterArr │
└──────────────────┘               │ │     ├─ timeout sched│          │ │  Events     │
                                   │ │     └─ WaitQ.Enqueue│          │ └─ aggregate  │
                                   │ ├─ StepEvent (pri=2)  │          └──────────────┘
                                   │ │  └─ Step()          │
                                   │ │     ├─ scheduleBatch│
                                   │ │     ├─ executeBatch │
                                   │ │     ├─ completions ─── callback
                                   │ │     └─ scheduleNext │
                                   │ ├─ TimeoutEvent(pri=5)│
                                   │ │  └─ Execute()       │
                                   │ │     ├─ state check  │
                                   │ │     ├─ release KV   │
                                   │ │     ├─ remove req   │
                                   │ │     ├─ INV-8 check  │
                                   │ │     └─ callback ────── callback
                                   │ └─ OnRequestDone func │
                                   └──────────────────────┘
```

**State ownership:** Request.State, Request.Deadline owned by simulator. Session lifecycle state owned exclusively by SessionManager. KV blocks owned by KVStore. Event queue owned by Simulator.

**Extension friction:** Adding a new event type after this PR: 2 files (event definition + priority constant). Adding a new session strategy: 1 file (+ factory when 2nd impl exists).

### D) Deviation Log

No deviations from design document `docs/plans/2026-03-13-client-behavior-model-design.md`.

### E) Review Guide

**The tricky part:** The event priority ordering change touches the core DES kernel. The `eventEntry` wrapper struct that adds `priority` and `seqID` to every event must not break existing same-timestamp ordering. Verify that the golden dataset still produces identical output (BC-11).

**What to scrutinize:** (1) INV-1 conservation — the 5-term formula has broken twice before. Check every path: normal completion, timeout, dropped-unservable, past-due timeout, length-capped. (2) The timeout handler's new-slice construction for RunningBatch removal — must not use in-place modification (R21). (3) The callback invocation in `EnqueueRequest` for dropped requests — must be session-agnostic.

**What's safe to skim:** SessionManager logic is straightforward (map lookup, round counter, context append). The `GenerateWorkload` function is mostly a refactor of existing `GenerateRequests` with session blueprint extraction.

**Known debt:** R14 — timeout handler spans multiple concerns (state, KV, queue, metrics), matching existing `processCompletions` pattern. State/statistics separation deferred.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/workload/session.go` — SessionManager, SessionBlueprint, activeSession

**Files to modify:**
- `sim/event.go` — eventEntry wrapper, TimeoutEvent, Priority() on all event types
- `sim/simulator.go` — EventQueue with priority+seqID, OnRequestDone field, timeout scheduling in EnqueueRequest, callback invocations in processCompletions, work-conserving check in timeout handler
- `sim/request.go` — StateTimedOut, Deadline field
- `sim/queue.go` — WaitQueue.Remove()
- `sim/metrics.go` — TimedOutRequests counter, SaveResults update
- `sim/metrics_utils.go` — MetricsOutput TimedOutRequests field
- `sim/workload/spec.go` — Timeout and ClosedLoop fields on ClientSpec
- `sim/workload/generator.go` — GenerateWorkload(), closed-loop reasoning path
- `sim/cluster/cluster.go` — OnRequestDone wiring, TimedOutRequests aggregation, inFlightRequests delta
- `sim/cluster/metrics.go` — TimedOutRequests in RawMetrics and CollectRawMetrics
- `cmd/root.go` — Switch to GenerateWorkload, SessionManager creation, callback wiring

**Files to create (tests):**
- `sim/timeout_test.go` — TimeoutEvent unit tests (BC-1, BC-2, BC-3, BC-12, BC-13, BC-15, BC-18)
- `sim/workload/session_test.go` — SessionManager unit tests (BC-6, BC-7, BC-8, BC-16, BC-17, BC-19)

**Files to modify (tests):**
- `sim/simulator_test.go` — Conservation test updated for 5-term formula (BC-4), golden dataset regression (BC-11)
- `sim/cluster/cluster_test.go` — Cluster conservation updated (BC-4), cluster routing test (BC-9)

**Key decisions:** Event priority uses sequential integers 0-5. `*int64` pointer type for Timeout on ClientSpec (R9). Counter-based session IDs (not UUID). New-slice pattern for RunningBatch removal (R21).

### G) Task Breakdown

This plan is organized into 10 tasks. Each task is independently testable and builds on prior tasks.

---

### Task 1: Event Queue Priority Infrastructure

**Contracts:** BC-12 (completion wins at equal timestamps), BC-10 (determinism), BC-11 (backward compat)

**Files:**
- Modify: `sim/event.go` — add `Priority() int` to Event interface, implement on all 5 event types
- Modify: `sim/simulator.go` — eventEntry wrapper, EventQueue with (timestamp, priority, seqID)
- Test: `sim/simulator_test.go` — golden dataset regression (BC-11), priority ordering test (BC-12)

**Step 1: Add Priority() to Event interface and all implementations**

In `sim/event.go`, add `Priority() int` to the Event interface. Implement on all 5 existing types:
- ArrivalEvent: priority 0
- QueuedEvent: priority 1
- StepEvent: priority 2
- ScheduledEvent: priority 3
- RequestLeftEvent: priority 4

**Step 1b: Fix QueuedEvent phantom StepEvent after early-return paths**

In `sim/event.go`, modify `QueuedEvent.Execute` to check `WaitQ.Len() > 0` before scheduling a StepEvent:
```go
func (e *QueuedEvent) Execute(sim *Simulator) {
    logrus.Debugf("<< Queued: %s at %d ticks", e.Request.ID, e.time)
    sim.EnqueueRequest(e.Request)
    // Only schedule StepEvent if there is work to do.
    // EnqueueRequest may return early (past-due timeout, StateTimedOut guard,
    // dropped-unservable) leaving WaitQ empty. Without this check, a phantom
    // StepEvent fires with no work, inflating stepCount and adding empty-batch
    // entries to queue-depth time series.
    if sim.stepEvent == nil && sim.WaitQ.Len() > 0 {
        pbe := &StepEvent{time: e.time}
        sim.Schedule(pbe)
        sim.stepEvent = pbe
    }
}
```
This replaces the current unconditional `sim.stepEvent == nil` check with `sim.stepEvent == nil && sim.WaitQ.Len() > 0`, and correctly sets `sim.stepEvent` after scheduling (matching the pattern used in `scheduleNextStep` and the timeout handler's BC-18 guard). This prevents duplicate StepEvent scheduling from consecutive QueuedEvents and also fixes a pre-existing issue where dropped-unservable requests triggered phantom StepEvents.

**Step 2: Add eventEntry wrapper and upgrade EventQueue**

In `sim/simulator.go`, replace the raw `[]Event` EventQueue with a `[]eventEntry` where:
```go
type eventEntry struct {
    event    Event
    seqID    int64
}
```
Add a `seqCounter int64` to Simulator. Update `Schedule()` to wrap events in `eventEntry` with monotonic seqID. Update `EventQueue.Less` to compare `(timestamp, priority, seqID)`. Update `Push`/`Pop`/`ProcessNextEvent` to handle the wrapper.

**Step 3: Write priority ordering test (BC-12)**

Test that when a StepEvent and a future TimeoutEvent (Task 2) share a timestamp, StepEvent fires first. For now, test that same-timestamp events with different priorities fire in priority order using existing event types (e.g., ArrivalEvent at tick T fires before QueuedEvent at tick T).

**Step 4: Run golden dataset test to verify BC-11**

Run: `go test ./sim/... -run TestSimulator_GoldenDataset -v`
Expected: PASS — output identical to current. If FAIL (same-timestamp event pairs in golden data changed order), regenerate golden dataset (R12).

**Step 5: Run all tests, lint, commit**

Run: `go test ./sim/... && go test ./sim/cluster/... && golangci-lint run ./sim/...`
Commit: `feat(sim): add (timestamp, priority, seqID) event queue ordering (BC-12)`

---

### Task 2: Request Model + TimeoutEvent Type

**Contracts:** BC-1, BC-2, BC-3 (timeout mechanics), BC-5 (no timeout for dropped)

**Files:**
- Modify: `sim/request.go` — add StateTimedOut, Deadline field
- Modify: `sim/event.go` — add TimeoutEvent with Priority() = 5
- Modify: `sim/queue.go` — add WaitQueue.Remove()
- Test: `sim/timeout_test.go` — new file

**Step 1: Add StateTimedOut and Deadline to Request**

In `sim/request.go`:
```go
const (
    StateQueued    RequestState = "queued"
    StateRunning   RequestState = "running"
    StateCompleted RequestState = "completed"
    StateTimedOut  RequestState = "timed_out"
)
```
Add `Deadline int64` field to Request struct. Update ALL construction sites (R4): `sim/workload/generator.go:261`, `sim/workload/reasoning.go:68`, `sim/workload/replay.go:44`, `sim/test_helpers_test.go:38`, `sim/cluster/test_helpers_test.go:40`. Zero-value (0) means no deadline — backward compatible.

**Step 2: Add WaitQueue.Remove()**

In `sim/queue.go`:
```go
// Remove removes a specific request from the queue by pointer identity.
// O(n) linear scan — acceptable because timeouts are infrequent relative to step processing.
// Returns true if the request was found and removed, false otherwise.
func (wq *WaitQueue) Remove(req *Request) bool {
    for i, r := range wq.queue {
        if r == req {
            wq.queue = append(wq.queue[:i], wq.queue[i+1:]...)
            return true
        }
    }
    return false
}
```

**Step 3: Add TimeoutEvent**

In `sim/event.go`:
```go
// TimeoutEvent models client-side request cancellation at the deadline tick.
// Classification: mixed exogenous/endogenous (round-0 exogenous, follow-up endogenous).
// Priority 5: fires after all other event types at equal timestamps (BC-12).
type TimeoutEvent struct {
    time    int64
    Request *Request
}

func (e *TimeoutEvent) Timestamp() int64 { return e.time }
func (e *TimeoutEvent) Priority() int    { return 5 }

func (e *TimeoutEvent) Execute(sim *Simulator) {
    // No-op guard: request already completed or timed out (BC-3)
    if e.Request.State == StateCompleted || e.Request.State == StateTimedOut {
        return
    }
    wasRunning := e.Request.State == StateRunning
    e.Request.State = StateTimedOut
    sim.Metrics.TimedOutRequests++

    // Release KV blocks (safe for zero-block queued requests per BC-15)
    sim.KVCache.ReleaseKVBlocks(e.Request)

    // Clean up computed-token tracking for ALL timed-out requests (prevents memory leak).
    // A preempted-then-queued request still has a reqNumComputedTokens entry from
    // its prior running phase. Must clean up in both branches, not just wasRunning.
    delete(sim.reqNumComputedTokens, e.Request.ID)

    if wasRunning {
        // New-slice construction (R21): build excluding timed-out request
        remaining := make([]*Request, 0, len(sim.RunningBatch.Requests)-1)
        for _, r := range sim.RunningBatch.Requests {
            if r != e.Request {
                remaining = append(remaining, r)
            }
        }
        sim.RunningBatch.Requests = remaining
    } else {
        sim.WaitQ.Remove(e.Request)
    }

    // INV-8 work-conserving: if running batch is now empty but WaitQ has work,
    // schedule a StepEvent (defense-in-depth, BC-18)
    if (sim.RunningBatch == nil || len(sim.RunningBatch.Requests) == 0) &&
        sim.stepEvent == nil && sim.WaitQ.Len() > 0 {
        pbe := StepEvent{time: e.time}
        sim.Schedule(&pbe)
        sim.stepEvent = &pbe
    }

    // Invoke completion callback for session management
    if sim.OnRequestDone != nil {
        for _, next := range sim.OnRequestDone(e.Request, e.time) {
            sim.InjectArrival(next)
        }
    }
}
```

**Step 4: Write timeout tests**

In `sim/timeout_test.go`, write tests for BC-1 (queued timeout), BC-2 (running timeout), BC-3 (no-op), BC-15 (preempt-then-timeout). Use the existing test helper pattern: create simulator with known config, inject requests, manually fire events.

**Step 5: Run tests, lint, commit**

Run: `go test ./sim/... -run TestTimeout -v && golangci-lint run ./sim/...`
Commit: `feat(sim): add TimeoutEvent, StateTimedOut, WaitQueue.Remove (BC-1, BC-2, BC-3, BC-5, BC-15)`

---

### Task 3: Timeout Scheduling + Metrics

**Contracts:** BC-4 (INV-1 conservation), BC-5 (no timeout for dropped), EC-2 (past-due guard)

**Files:**
- Modify: `sim/simulator.go` — add OnRequestDone field, timeout scheduling in EnqueueRequest, past-due guard
- Modify: `sim/metrics.go` — TimedOutRequests counter, SaveResults INV-1 update
- Modify: `sim/metrics_utils.go` — MetricsOutput TimedOutRequests field
- Test: `sim/simulator_test.go` — conservation tests updated

**Step 1: Add OnRequestDone and TimedOutRequests**

In `sim/simulator.go`, add to Simulator struct:
```go
OnRequestDone func(req *Request, tick int64) []*Request
```

In `sim/metrics.go`, add to Metrics struct:
```go
TimedOutRequests int // Requests cancelled by client timeout
```

In `sim/metrics.go` SaveResults (around line 74), update InjectedRequests:
```go
InjectedRequests: m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable + m.TimedOutRequests,
```

In `sim/metrics_utils.go` MetricsOutput, add:
```go
TimedOutRequests int `json:"timed_out_requests,omitempty"`
```

**Step 2: Add timeout scheduling to EnqueueRequest**

Restructure the end of `EnqueueRequest` (after guards pass, before WaitQ.Enqueue). The new order matches the design doc Section 4.2: (1) count input tokens, (2) check past-due, (3) enqueue + schedule or (4) immediate timeout:

```go
// Input tokens counted BEFORE past-due check (request was received)
sim.Metrics.TotalInputTokens += len(r.InputTokens)

// Past-due guard (EC-2): check BEFORE enqueue to avoid enqueue-then-remove
if r.Deadline > 0 && r.Deadline <= sim.Clock {
    r.State = StateTimedOut
    sim.Metrics.TimedOutRequests++
    if sim.OnRequestDone != nil {
        for _, next := range sim.OnRequestDone(r, sim.Clock) {
            sim.InjectArrival(next)
        }
    }
    return
}

sim.WaitQ.Enqueue(r)

// Schedule timeout event (after all guards + enqueue — BC-5)
if r.Deadline > 0 && r.Deadline <= sim.Horizon {
    // Skip scheduling when deadline > horizon (perf: avoids orphaned events)
    sim.Schedule(&TimeoutEvent{time: r.Deadline, Request: r})
}
```

Also add a `StateTimedOut` guard at the TOP of `EnqueueRequest` to handle the pre-QueuedEvent timeout race (CRITICAL fix):
```go
// Guard -1: Already timed out (race: TimeoutEvent fired before QueuedEvent)
if r.State == StateTimedOut {
    // Request was timed out between ArrivalEvent and QueuedEvent.
    // Don't enqueue — it's already counted as timed_out.
    // TotalInputTokens is NOT counted for this path: the request timed out
    // during the queueing delay (alpha overhead) before the server processed
    // the input. Analogous to a client disconnect before HTTP body received.
    // INV-1 holds: request is counted in timed_out bucket.
    return
}
```

**Pre-QueuedEvent timeout: third path in TimeoutEvent.Execute.** When the TimeoutEvent fires for a request that hasn't been enqueued yet (between ArrivalEvent and QueuedEvent), the request is NOT in WaitQ or RunningBatch. The timeout handler:
- Sets `StateTimedOut` (correct)
- Calls `WaitQ.Remove` → returns false (correct, request not in queue)
- Calls `ReleaseKVBlocks` → no-op (no blocks allocated)
- `reqNumComputedTokens` delete → no-op (no entry)
- Callback fires → SessionManager cancels session if applicable
The request remains in `Metrics.Requests` (from InjectArrival) — this is intentional. Timed-out requests appear in per-request JSON output with their arrival time and timed-out status. They are NOT deleted from `Metrics.Requests` (unlike dropped-unservable requests which call `delete(sim.Metrics.Requests, r.ID)`). Counted as `timed_out` in INV-1. `TotalInputTokens` not counted (intentional — see guard -1 above).

**Note on `Metrics.Requests` lifecycle:** For dropped-unservable requests, `EnqueueRequest` deletes from `Metrics.Requests`. For timed-out requests (all paths: normal timeout, pre-QueuedEvent race, past-due), the entry is RETAINED so per-request output includes them. This asymmetry is correct: dropped requests were server-rejected (never entered the system), while timed-out requests were client-cancelled (they entered but didn't finish).

**Step 3: Add callback invocations in processCompletions**

In `processCompletions`, after `sim.recordRequestCompletion(req)` for both normal and length-capped paths, add:
```go
if sim.OnRequestDone != nil {
    for _, next := range sim.OnRequestDone(req, now+currStepAdvance) {
        sim.InjectArrival(next)
    }
}
```

Also add callback for dropped-unservable requests in `EnqueueRequest` — in each guard's drop path (Guards 0, 1, 2), before `return`, add:
```go
if sim.OnRequestDone != nil {
    // Iterate return value (R1: don't silently discard).
    // SessionManager returns nil for drops (cancels session), but
    // future callbacks might return follow-ups.
    for _, next := range sim.OnRequestDone(r, sim.Clock) {
        sim.InjectArrival(next)
    }
}
```

**Step 4: Update conservation tests**

Update `TestSimulator_RequestConservation_FiniteHorizon_ThreeTermEquation` to use 5-term formula:
```go
assert.Equal(t, injected, completed+stillQueued+stillRunning+droppedUnservable+timedOut)
```

Write new test `TestSimulator_Conservation_WithTimeout` that injects requests with short deadlines, verifies 5-term conservation.

**Step 5: Run all tests, lint, commit**

Run: `go test ./sim/... && golangci-lint run ./sim/...`
Commit: `feat(sim): timeout scheduling in EnqueueRequest, INV-1 5-term conservation (BC-4, BC-5, EC-2)`

---

### Task 4: ClientSpec Timeout + ClosedLoop Fields

**Contracts:** EC-1 (timeout validation), BC-11 (backward compat)

**Files:**
- Modify: `sim/workload/spec.go` — Timeout *int64, ClosedLoop *bool on ClientSpec
- Test: `sim/workload/generator_test.go` — validation tests

**Step 1: Add fields to ClientSpec**

In `sim/workload/spec.go`, add to ClientSpec:
```go
Timeout   *int64 `yaml:"timeout,omitempty"`    // Per-request timeout in µs. nil = default (300s). 0 = no timeout.
ClosedLoop *bool  `yaml:"closed_loop,omitempty"` // nil = default (true for reasoning). false = open-loop (all rounds pre-generated).
```

**Step 2: Add Deadline computation to workload generation**

In `sim/workload/generator.go`, add a helper:
```go
const DefaultTimeoutUs = 300_000_000 // 300s default timeout

func computeDeadline(arrivalTime int64, clientTimeout *int64) int64 {
    if clientTimeout == nil {
        return arrivalTime + DefaultTimeoutUs
    }
    if *clientTimeout == 0 {
        return 0 // no timeout
    }
    return arrivalTime + *clientTimeout
}
```

Also add validation in `ClientSpec.Validate()` (in spec.go's `validateClient` function):
```go
// R3: Validate timeout if specified
if c.Timeout != nil && *c.Timeout < 0 {
    return fmt.Errorf("client %q: timeout must be non-negative, got %d", c.ID, *c.Timeout)
}
```

Set `req.Deadline = computeDeadline(req.ArrivalTime, client.Timeout)` for every request construction site in generator.go (line 261 standard path). Also set Deadline on requests returned by `GenerateReasoningRequests` — after the call returns, loop over the results and set `req.Deadline = computeDeadline(req.ArrivalTime, client.Timeout)` for each. This covers both SingleSession (line 135) and multi-session (line 183) reasoning paths. The reasoning.go construction site (line 68) does NOT set Deadline — it's set by the caller in generator.go after return.

**Step 3: Add isClosedLoop helper**

```go
func isClosedLoop(client *ClientSpec) bool {
    if client.ClosedLoop != nil {
        return *client.ClosedLoop
    }
    // Default: true for reasoning/multi-turn clients
    return client.Reasoning != nil && client.Reasoning.MultiTurn != nil
}
```

**Step 4: Write validation tests**

Test that nil timeout → 300s default, explicit 0 → no deadline, explicit value → correct deadline. Test that ClosedLoop defaults to true for reasoning, false for non-reasoning.

**Step 5: Run tests, lint, commit**

Run: `go test ./sim/workload/... && golangci-lint run ./sim/workload/...`
Commit: `feat(workload): add Timeout and ClosedLoop fields to ClientSpec (EC-1)`

---

### Task 5: SessionManager

**Contracts:** BC-6 (round generation), BC-7 (session cancellation), BC-8 (context accumulation), BC-16 (length-capped continues), BC-17 (dropped cancels), BC-19 (horizon guard)

**Files:**
- Create: `sim/workload/session.go`
- Test: `sim/workload/session_test.go`

**Step 1: Define types**

In `sim/workload/session.go`:
```go
package workload

import (
    "math/rand"
    "github.com/inference-sim/inference-sim/sim"
)

type sessionState string
const (
    sessionActive           sessionState = "active"
    sessionCompleted        sessionState = "completed"
    sessionCancelled        sessionState = "cancelled"
    sessionHorizonInterrupted sessionState = "horizon_interrupted"
)

type SessionBlueprint struct {
    SessionID     string
    ClientID      string
    MaxRounds     int
    ContextGrowth string
    ThinkTimeUs   int64
    Timeout       *int64
    Horizon       int64
    InputSampler  LengthSampler
    OutputSampler LengthSampler
    RNG           *rand.Rand
    Prefix        []int
    TenantID      string
    SLOClass      string
    Model         string
}

type activeSession struct {
    blueprint     *SessionBlueprint
    currentRound  int
    contextTokens []int
    state         sessionState
}

type SessionManager struct {
    sessions map[string]*activeSession
    idCounter int64
}
```

**Step 2: Implement NewSessionManager and OnComplete**

```go
func NewSessionManager(blueprints []SessionBlueprint) *SessionManager {
    sm := &SessionManager{sessions: make(map[string]*activeSession, len(blueprints))}
    for i := range blueprints {
        bp := &blueprints[i]
        if bp.MaxRounds < 1 {
            panic(fmt.Sprintf("NewSessionManager: session %s has MaxRounds=%d, must be >= 1", bp.SessionID, bp.MaxRounds))
        }
        sm.sessions[bp.SessionID] = &activeSession{
            blueprint: bp, state: sessionActive,
        }
    }
    return sm
}

func (sm *SessionManager) OnComplete(req *sim.Request, tick int64) []*sim.Request {
    sess, ok := sm.sessions[req.SessionID]
    if !ok { return nil } // non-session request
    if sess.state != sessionActive { return nil } // already terminal

    // Session cancellation on timeout (BC-7)
    if req.State == sim.StateTimedOut {
        sess.state = sessionCancelled
        return nil
    }

    // Dropped-unservable follow-up cancels session (BC-17)
    // (callback invoked from EnqueueRequest drop path — req never completed)
    if req.State == sim.StateQueued {
        // Request was dropped before entering queue (state still queued from construction)
        sess.state = sessionCancelled
        return nil
    }

    // Length-capped: continues session (BC-16) — state is StateCompleted

    // Final round check
    if sess.currentRound >= sess.blueprint.MaxRounds-1 {
        sess.state = sessionCompleted
        return nil
    }

    bp := sess.blueprint

    // Horizon guard (BC-19)
    arrivalTime := tick + bp.ThinkTimeUs
    if arrivalTime > bp.Horizon {
        sess.state = sessionHorizonInterrupted
        return nil
    }

    // Generate round N+1
    inputLen := bp.InputSampler.Sample(bp.RNG)
    outputLen := bp.OutputSampler.Sample(bp.RNG)
    newInputTokens := sim.GenerateRandomTokenIDs(bp.RNG, inputLen)
    outputTokens := sim.GenerateRandomTokenIDs(bp.RNG, outputLen)

    // Context accumulation (BC-8)
    // Use ACTUAL generated output count, not oracle OutputTokens.
    // For length-capped requests, ProgressIndex - len(InputTokens) gives actual output count.
    // For normal completions, this equals len(OutputTokens).
    actualOutputLen := int(req.ProgressIndex) - len(req.InputTokens)
    if actualOutputLen < 0 { actualOutputLen = 0 }

    var inputTokens []int
    if bp.ContextGrowth == "accumulate" {
        // Accumulate: prepend prior context + this round's actual input/output
        sess.contextTokens = append(sess.contextTokens, req.InputTokens...)
        if actualOutputLen > 0 && len(req.OutputTokens) > 0 {
            // Use actual output count, not full oracle OutputTokens (BC-16: truncated)
            outTokens := req.OutputTokens
            if actualOutputLen < len(outTokens) {
                outTokens = outTokens[:actualOutputLen]
            }
            sess.contextTokens = append(sess.contextTokens, outTokens...)
        }
        inputTokens = append(append([]int{}, sess.contextTokens...), newInputTokens...)
    } else {
        inputTokens = newInputTokens
    }

    // Prepend prefix
    if len(bp.Prefix) > 0 {
        inputTokens = append(append([]int{}, bp.Prefix...), inputTokens...)
    }

    sess.currentRound++
    sm.idCounter++
    nextReq := &sim.Request{
        ID:           fmt.Sprintf("session_%s_round_%d", bp.SessionID, sess.currentRound),
        ArrivalTime:  arrivalTime,
        InputTokens:  inputTokens,
        OutputTokens: outputTokens,
        MaxOutputLen: len(outputTokens),
        State:        sim.StateQueued,
        Deadline:     computeDeadline(arrivalTime, bp.Timeout),
        TenantID:     bp.TenantID,
        SLOClass:     bp.SLOClass,
        Model:        bp.Model,
        SessionID:    bp.SessionID,
        RoundIndex:   sess.currentRound,
    }
    return []*sim.Request{nextReq}
}
```

**Step 3: Write comprehensive tests**

In `sim/workload/session_test.go`, write table-driven tests for:
- BC-6: round generation with correct arrival time
- BC-7: session cancellation on timeout
- BC-8: context accumulation
- BC-16: length-capped continues
- BC-17: dropped cancels (pass request with State=StateQueued)
- BC-19: horizon guard

**Step 4: Run tests, lint, commit**

Run: `go test ./sim/workload/... -run TestSession -v && golangci-lint run ./sim/workload/...`
Commit: `feat(workload): add SessionManager with closed-loop round generation (BC-6, BC-7, BC-8, BC-16, BC-17, BC-19)`

---

### Task 6: GenerateWorkload + Closed-Loop Reasoning Path

**Contracts:** BC-6, BC-10, BC-11

**Files:**
- Modify: `sim/workload/generator.go` — GenerateWorkload(), closed-loop reasoning modification
- Test: `sim/workload/generator_test.go`

**Step 1: Define GeneratedWorkload and GenerateWorkload**

```go
type GeneratedWorkload struct {
    Requests []*sim.Request
    Sessions []SessionBlueprint
}

func GenerateWorkload(spec *WorkloadSpec, horizon int64, maxRequests int64) (*GeneratedWorkload, error) {
    // ... (same logic as GenerateRequests, but with session blueprint extraction)
}
```

**Step 2: Modify reasoning path for closed-loop**

In the reasoning code paths, when `isClosedLoop(client)` is true:
- **SingleSession:** Generate only round-0, create SessionBlueprint
- **Multi-session:** Generate only round-0 per session, create SessionBlueprint per session

When `isClosedLoop(client)` is false (opt-out): preserve current behavior (all rounds pre-generated).

**Step 3: Update GenerateRequests to delegate**

```go
func GenerateRequests(spec *WorkloadSpec, horizon int64, maxRequests int64) ([]*sim.Request, error) {
    wl, err := GenerateWorkload(spec, horizon, maxRequests)
    if err != nil { return nil, err }
    return wl.Requests, nil
}
```

**Step 4: Write tests**

Test that closed-loop reasoning generates only round-0 + blueprints. Test that `closed_loop: false` preserves all-rounds generation. Test determinism (BC-10).

**Step 5: Run tests, lint, commit**

Run: `go test ./sim/workload/... && golangci-lint run ./sim/workload/...`
Commit: `feat(workload): add GenerateWorkload with closed-loop session blueprints (BC-6, BC-10)`

---

### Task 7: Cluster Integration

**Contracts:** BC-9 (cluster routing for follow-ups), BC-4 (cluster conservation)

**Files:**
- Modify: `sim/cluster/cluster.go` — OnRequestDone wiring, TimedOutRequests aggregation, inFlightRequests delta
- Modify: `sim/cluster/metrics.go` — TimedOutRequests in RawMetrics
- Test: `sim/cluster/cluster_test.go`

**Step 1: Wire OnRequestDone callback in ClusterSimulator**

The SessionManager wiring happens in `cmd/root.go` (NOT in sim/cluster/) to preserve the dependency direction `cmd/ → sim/cluster/ → sim/`. The cluster accepts a generic callback function, not a SessionManager reference:

Add `onRequestDone func(*sim.Request, int64) []*sim.Request` parameter to `NewClusterSimulator`. The cluster wires it onto each instance's Simulator, wrapping it to route follow-ups through the cluster pipeline:

```go
// In NewClusterSimulator, if onRequestDone is non-nil:
for _, inst := range instances {
    inst := inst // capture for closure
    inst.sim.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
        nextReqs := onRequestDone(req, tick)
        for _, next := range nextReqs {
            heap.Push(&cs.clusterEvents, clusterEventEntry{
                event: &ClusterArrivalEvent{time: next.ArrivalTime, request: next},
                seqID: cs.nextSeqID(),
            })
        }
        return nil // don't inject locally — route through cluster pipeline
    }
}
```

The `cmd/root.go` creates the SessionManager and passes `sessionMgr.OnComplete` as the callback:
```go
var onRequestDone func(*sim.Request, int64) []*sim.Request
if sessionMgr != nil {
    onRequestDone = sessionMgr.OnComplete
}
cs := cluster.NewClusterSimulator(config, wl.Requests, onRequestDone)
```

This keeps `sim/cluster/` importing only `sim/` (no `sim/workload/` dependency).

**Step 2: Update inFlightRequests delta for timedOut**

In the cluster event loop, update delta computation:
```go
timedOutAfter := inst.Metrics().TimedOutRequests
delta := (completedAfter - completedBefore) + (droppedAfter - droppedBefore) + (timedOutAfter - timedOutBefore)
```

**Step 3: Update aggregateMetrics**

Add `merged.TimedOutRequests += m.TimedOutRequests` to the aggregation loop.

**Step 4: Update RawMetrics and CollectRawMetrics**

In `sim/cluster/metrics.go`, add `TimedOutRequests int` to RawMetrics and collect it.

**Step 5: Update all NewClusterSimulator call sites (R4)**

The new `onRequestDone` parameter is added to `NewClusterSimulator`. ALL callers must be updated:
- `cmd/root.go` — pass `sessionMgr.OnComplete` (Task 8)
- `sim/cluster/cluster_test.go` — pass `nil` (multiple tests)
- `sim/cluster/inflight_requests_test.go` — pass `nil`
- `sim/cluster/prefix_routing_test.go` — pass `nil`
- `sim/cluster/cluster_trace_test.go` — pass `nil`
- `sim/cluster/evaluation_test.go` — pass `nil`
- `sim/cluster/metrics_test.go` — pass `nil` (2 calls)
- `sim/cluster/metrics_substrate_test.go` — pass `nil` (4 calls)
- `sim/cluster/cluster_warnings_test.go` — pass `nil` (5 calls)
- `sim/cluster/cluster_event_test.go` — pass `nil` (3 calls)

All test callers pass `nil` → no callback → non-session behavior preserved. Total: ~73 test call sites across these files, all mechanical `, nil` additions.

**Step 6: Write cluster conservation test**

Test that cluster mode with timeouts maintains 5-term INV-1 conservation. Test that follow-up rounds go through routing (BC-9). Verify the pre-QueuedEvent timeout race in cluster mode: inFlightRequests delta correctly accounts for timed-out requests.

**Step 6: Run tests, lint, commit**

Run: `go test ./sim/cluster/... && golangci-lint run ./sim/cluster/...`
Commit: `feat(cluster): wire timeout + session manager, cluster routing for follow-ups (BC-9, BC-4)`

---

### Task 8: CLI Wiring

**Contracts:** BC-11 (backward compat for CLI workloads)

**Files:**
- Modify: `cmd/root.go` — Switch to GenerateWorkload, create SessionManager, wire to cluster

**Step 1: Switch to GenerateWorkload**

Replace `workload.GenerateRequests(spec, simulationHorizon, maxRequests)` with `workload.GenerateWorkload(...)`. Extract requests and sessions.

**Step 2: Create SessionManager if sessions exist**

```go
var sessionMgr *workload.SessionManager
if wl.Sessions != nil && len(wl.Sessions) > 0 {
    sessionMgr = workload.NewSessionManager(wl.Sessions)
}
```

**Step 3: Pass callback to ClusterSimulator**

```go
var onRequestDone func(*sim.Request, int64) []*sim.Request
if sessionMgr != nil {
    onRequestDone = sessionMgr.OnComplete
}
cs := cluster.NewClusterSimulator(config, wl.Requests, onRequestDone)
```

This passes the callback function, not the SessionManager — preserving the `cmd/ → sim/cluster/ → sim/` dependency direction.

**Step 4: Verify single-instance mode is handled by cluster path**

`cmd/root.go` uses a unified cluster path for ALL values of `numInstances` (including 1). The `onRequestDone` callback passed to `NewClusterSimulator` in Step 3 handles both single-instance and multi-instance modes. No separate `sim.OnRequestDone` wiring is needed — the cluster wraps it for each instance. No code change in this step.

**Step 5: Run end-to-end test, commit**

Run: `go build -o blis main.go && go test ./cmd/... && go test ./sim/... -run TestSimulator_GoldenDataset`
Commit: `feat(cmd): wire GenerateWorkload + SessionManager in CLI (BC-11)`

---

### Task 9: Integration Tests — Conservation + #529 Regression

**Contracts:** BC-4 (conservation), BC-13 (INV-4), BC-14 (horizon-interrupted), BC-18 (work-conserving)

**Files:**
- Modify: `sim/simulator_test.go` — 5-term conservation, timeout+KV conservation
- Modify: `sim/cluster/cluster_test.go` — cluster conservation with timeouts

**Step 1: Write 5-term conservation test with timeouts**

Create requests where some have short deadlines (will timeout) and verify:
`completed + still_queued + still_running + dropped_unservable + timed_out == injected`

**Step 2: Write INV-4 test after running-request timeout (BC-13)**

Verify `allocated_blocks + free_blocks == total_blocks` after a running request times out.

**Step 3: Write horizon-interrupted session test (BC-14)**

Create a session with rounds that would extend past horizon. Verify session stops at horizon, remaining rounds not generated.

**Step 4: Run all tests, lint, commit**

Run: `go test ./sim/... ./sim/cluster/... ./cmd/... && golangci-lint run ./...`
Commit: `test(sim): integration tests for timeout conservation (BC-4, BC-13, BC-14, BC-18)`

---

### Task 10: Documentation + CLAUDE.md Update

**Contracts:** N/A — documentation only

**Files:**
- Modify: `CLAUDE.md` — update File Organization tree, add event.go description update, INV-1 formula, new files
- Modify: `docs/contributing/standards/invariants.md` — INV-1 5-term formula, INV-10, INV-11

**Step 1: Update CLAUDE.md**

- File Organization: add `sim/workload/session.go` entry
- Update `event.go` description to mention TimeoutEvent
- Update `request.go` description to mention StateTimedOut, Deadline
- Update INV-1 formula to 5-term
- Add INV-10 and INV-11

**Step 2: Update invariants.md**

Add INV-10 (Session causality) and INV-11 (Session completeness) to the canonical invariants list. Update INV-1 formula.

**Step 3: Commit**

Commit: `docs: update CLAUDE.md and invariants.md for client behavior model`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | TestTimeout_QueuedRequest_TimesOut |
| BC-2 | Task 2 | Unit | TestTimeout_RunningRequest_TimesOut |
| BC-3 | Task 2 | Unit | TestTimeout_CompletedRequest_NoOp |
| BC-4 | Task 9 | Integration | TestConservation_WithTimeouts_FiveTermFormula |
| BC-5 | Task 3 | Unit | TestTimeout_DroppedRequest_NoTimeoutEvent |
| BC-6 | Task 5 | Unit | TestSession_RoundGeneration_CorrectArrivalTime |
| BC-7 | Task 5 | Unit | TestSession_TimeoutCancels_NoMoreRounds |
| BC-8 | Task 5 | Unit | TestSession_ContextAccumulation_PrependsPriorTokens |
| BC-9 | Task 7 | Integration | TestCluster_FollowUpRouted_NotHardcoded |
| BC-10 | Task 6 | Unit | TestGenerateWorkload_Deterministic_SameSeed |
| BC-11 | Task 1 | Golden | TestSimulator_GoldenDataset (existing, must still pass) |
| BC-12 | Task 1 | Unit | TestEventQueue_SameTimestamp_PriorityOrder |
| BC-13 | Task 9 | Integration | TestTimeout_RunningRequest_KVConservation |
| BC-14 | Task 9 | Integration | TestSession_HorizonInterrupted_RoundsCounted |
| BC-15 | Task 2 | Unit | TestTimeout_PreemptThenTimeout_SafeNoOp |
| BC-16 | Task 5 | Unit | TestSession_LengthCapped_ContinuesSession |
| BC-17 | Task 5 | Unit | TestSession_DroppedFollowUp_CancelsSession |
| BC-18 | Task 2 | Unit | TestTimeout_EmptyBatch_SchedulesStepEvent |
| BC-19 | Task 5 | Unit | TestSession_BeyondHorizon_NotGenerated |
| NC-1 | Task 9 | Invariant | (covered by BC-4 conservation test) |
| NC-2 | Task 1 | Golden | (covered by BC-11 golden dataset) |
| NC-3 | Task 2 | Unit | (covered by BC-2 new-slice construction) |
| EC-1 | Task 4 | Unit | TestClientSpec_TimeoutValidation |
| EC-2 | Task 3 | Unit | TestEnqueue_PastDueTimeout_ImmediateTimeout |

**Golden dataset:** Task 1 verifies golden dataset still passes. If the priority ordering change affects output, regenerate golden values in Task 1 (R12).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Priority ordering breaks golden dataset | Medium | Medium | Verify in Task 1; regenerate if needed (R12) | 1 |
| INV-1 5-term conservation wrong | Medium | High | Explicit test in Task 9; INV-1 broke twice before | 3, 9 |
| RunningBatch in-place modification (R21) | Low | High | New-slice pattern enforced in TimeoutEvent.Execute | 2 |
| WaitQ.Remove not finding request | Low | Medium | Return bool; log warning if not found | 2 |
| Session RNG non-determinism | Low | Medium | Per-session RNG from client RNG; tested in Task 6 | 5, 6 |
| Callback invoked during processCompletions iteration | Low | High | Callback only calls InjectArrival (modifies eventQueue, not RunningBatch) | 3 |
| Past-due timeout double-counting | Low | High | Input tokens counted before guard; verified in Task 3 | 3 |
| Orphaned TimeoutEvents accumulate in heap | Medium | Low | Every completed request leaves one orphaned TimeoutEvent (no-op on fire). `deadline > horizon` optimization skips scheduling (Task 3). For <10K requests: negligible. For >100K: heap grows by O(N), operations slow by constant factor O(log(2N)) vs O(log(N)). **Follow-up optimization:** lazy deletion (cancelled flag) or timeout cancellation on completion. Accepted technical debt — not in this PR scope. | 3 |
| Pre-QueuedEvent timeout race: TotalInputTokens not counted | Low | Low | Intentional: request timed out during alpha delay before server processed input. Analogous to client disconnect before HTTP body received. INV-1 holds (counted as timed_out). Documented in Task 3 guard -1 comment. | 2, 3 |
| NewClusterSimulator signature change breaks test callers | Medium | Medium | All ~73 test call sites pass `nil` for callback. Complete list in Task 7 Step 5. | 7 |
| QueuedEvent phantom StepEvent after early return | Medium | Low | Pre-existing issue amplified by past-due/guard-1 paths. Fixed in Task 1 Step 1b: add `WaitQ.Len() > 0` check to QueuedEvent.Execute | 1 |

### J) Sanity Checklist

- [x] No unnecessary abstractions (no session strategy factory — R13 deviation justified)
- [x] No feature creep (closed_loop opt-out is minimal)
- [x] No unexercised flags (Deadline used by TimeoutEvent, ClosedLoop used by generator)
- [x] No partial implementations (every component has tests)
- [x] Breaking change documented (open-loop → closed-loop with opt-out)
- [x] No hidden global state (SessionManager owned by caller, not global)
- [x] All code passes golangci-lint
- [x] Shared test helpers used (testGenerateRequests, injectRequests)
- [x] CLAUDE.md updated (Task 10)
- [x] INV-1 formula in invariants.md updated (Task 10)
- [x] Deviation log reviewed — no deviations
- [x] Each task produces testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5→6→7→8→9→10)
- [x] All 19 BCs mapped to tasks
- [x] Golden dataset verified (Task 1)
- [x] Construction site audit: Request struct has ~80+ construction sites in tests; only production sites need Deadline field (zero-value is backward compat for tests)
- [x] R1: No silent continue/return — all paths increment counters or invoke callbacks
- [x] R4: Construction sites audited — 4 production sites + 2 test helpers
- [x] R9: *int64 for Timeout on ClientSpec
- [x] R10: KnownFields(true) — new YAML fields are optional (omitempty)
- [x] R21: New-slice pattern for RunningBatch removal in TimeoutEvent

<function_calls>
<invoke name="AskUserQuestion">
<parameter name="questions">[{"question": "Part 1 (Design Validation) is written with 19 BCs + 3 NCs + 2 ECs. The task breakdown will be ~2000+ lines with complete code. How should we proceed?", "header": "Plan scope", "options": [{"label": "Continue writing full plan now", "description": "Write all 10 tasks with complete code, tests, and commands. Large output but gives you the complete plan to review."}, {"label": "Review Part 1 first", "description": "You review the BCs and component interaction before I invest in the full task breakdown. Catches design-level issues early."}, {"label": "Write plan, then convergence review", "description": "I write the full plan, then run /convergence-review pr-plan to validate it before your review."}], "multiSelect": false}]