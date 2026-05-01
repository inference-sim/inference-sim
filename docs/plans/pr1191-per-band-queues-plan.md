# Per-Priority-Band Queues + FlowControlAdmission Policy — Implementation Plan

**Goal:** Replace the single-heap gateway queue with a per-priority-band, per-flow queue structure, and add a `FlowControlAdmission` policy that merges admission and queuing into one step — matching GIE's flow control architecture.

**The problem today:** BLIS runs admission first (via `AdmissionPolicy.Admit()`), then separately enqueues in the gateway queue as a second step wired in `cluster_event.go`. In llm-d, when flow control is enabled the flow controller IS the admission controller — `Admit()` calls `EnqueueAndWait()` internally. BLIS's current two-step split cannot correctly model GIE's per-band queue capacity limits (where enqueue outcome determines admission) and makes the dispatch ordering obsolete (the `--dispatch-order` flag controls a single heap but GIE always uses strict priority across bands).

**What this PR adds:**

1. Per-priority-band, per-flow queue data structure (`FlowKey`, `priorityBand`, `flowQueue`) replacing the single heap in `gateway_queue.go`
2. `FlowControlAdmission` policy in `sim/cluster/` that owns the queue internally and returns enqueue outcome — selected when `--flow-control` is enabled
3. Simplified wiring in `cluster_event.go` that replaces the inline flow-control block with a smaller `cs.flowControlAdmission != nil` check
4. New `--per-band-capacity` CLI flag for per-band queue depth limits

**Why this matters:** Brings BLIS's request pipeline to architectural parity with llm-d/GIE Phase 2, enabling admission control algorithms developed in BLIS to transfer cleanly to production.

**Architecture:** New types in `sim/cluster/gateway_queue.go` (per-band queue), new policy in `sim/cluster/flow_control_admission.go` (`FlowControlAdmission` — lives in `sim/cluster/` because it needs `GatewayQueue`; implements `sim.AdmissionPolicy`). Wiring in `cluster_event.go` replaces the inline `if cs.flowControlEnabled { ... }` block (lines 178-208) with a smaller check using the typed policy reference. The existing `SaturationDetector` interface is reused without changes.

**Source:** [Issue #1191](https://github.com/inference-sim/inference-sim/issues/1191), part of [#899](https://github.com/inference-sim/inference-sim/issues/899) (GIE flow control parity — Phase 2). Requires #1190 (Phase 1 — merged as PR #1202).

**Closes:** `Fixes #1191`

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** Gateway queue data structure (`sim/cluster/gateway_queue.go`) and flow control admission policy (`sim/cluster/flow_control_admission.go`).
2. **Adjacent blocks:** `ClusterSimulator` (owns the queue and admission policy), `cluster_event.go` (wiring), `SaturationDetector` (dispatch gating), `SLOPriorityMap` (priority mapping), `TenantBudgetAdmission` (decorator — wraps `FlowControlAdmission`; budget check happens BEFORE enqueue, matching llm-d where over-budget sheddable requests are never enqueued).
3. **Invariants touched:** INV-1 (request conservation — same buckets, simplified wiring), INV-5 (causality — gateway enqueue/dispatch timestamps), INV-6 (determinism — per-band flow iteration must use sorted keys, not map order), INV-9 (oracle knowledge boundary — `FlowControlAdmission.Admit()` must NOT read `Request.OutputTokens`).
4. **Construction site audit:**
   - `GatewayQueue` — constructed in `cluster.go:407` (`NewGatewayQueue(...)`) when flow control is enabled. This PR keeps the construction in `NewClusterSimulator` (via `FlowControlAdmission`).
   - `AdmissionPolicy` — constructed in `cluster.go:361-396` via factory logic. This PR adds the `FlowControlAdmission` path.
   - `ClusterSimulator` — single construction site in `NewClusterSimulator`.

---

## Part 1: Design Validation

### A) Executive Summary

This PR replaces the flat gateway queue heap with a hierarchical per-priority-band, per-flow queue structure. Each unique `(TenantID, Priority)` pair gets its own FIFO queue. Priority bands are iterated highest-first during dispatch. When `--flow-control` is enabled, the new `FlowControlAdmission` policy replaces the current legacy admission policy — admission IS the queue, matching llm-d's architecture. When `--flow-control` is disabled, behavior is identical to today (BC-1 backward compatibility). The existing `--dispatch-order` flag becomes a no-op when flow control is enabled (per-band queues are always priority-ordered across bands; a `logrus.Warnf` is emitted when both flags are set). No new packages or interfaces are introduced — `FlowControlAdmission` implements the existing `sim.AdmissionPolicy` interface.

### B) Behavioral Contracts

**Positive contracts (what MUST happen):**

BC-1: Backward Compatibility
- GIVEN `--flow-control` is not enabled (default)
- WHEN a request arrives
- THEN the existing admission policy (AlwaysAdmit/TokenBucket/TierShed/GAIELegacy) handles admission with no queue involvement, and behavior is byte-identical to the previous release

BC-2: Flow Key Mapping
- GIVEN `--flow-control` is enabled and a request arrives with TenantID="tenant-A" and SLOClass="standard" (priority 3)
- WHEN the request is enqueued
- THEN it is placed in the flow queue keyed by `("tenant-A", 3)` within priority band 3

BC-3: Default Flow for Empty TenantID
- GIVEN `--flow-control` is enabled and a request arrives with TenantID="" and SLOClass="critical" (priority 4)
- WHEN the request is enqueued
- THEN it is placed in the flow queue keyed by `("default", 4)` within priority band 4

BC-4: Priority Band Dispatch Order
- GIVEN bands `{4: [A:4], 3: [B:3, C:3], -1: [D:-1]}` with requests enqueued
- WHEN `Dequeue()` is called
- THEN returns the head of the highest non-empty band (band 4 first), skipping empty bands, returning nil only when ALL bands are empty

BC-5: Global-Strict Fairness Within Band
- GIVEN band 3 has flows `[B:3 with 2 reqs at seqID=10,20]` and `[C:3 with 1 req at seqID=5]`
- WHEN `Dequeue()` picks from band 3
- THEN returns C:3's request (seqID=5, earliest enqueue across all flows in the band)

BC-6: Per-Band Capacity Enforcement
- GIVEN `--per-band-capacity 10` and band 3 already has 10 requests (all non-sheddable)
- WHEN a new standard(priority=3) request arrives
- THEN the request is rejected at the band level (no sheddable displacement available within the band)

BC-7: FlowControlAdmission as Admission Policy
- GIVEN `--flow-control` is enabled
- WHEN a request arrives at `AdmissionDecisionEvent.Execute`
- THEN `cs.admissionPolicy.Admit()` is called (which is `FlowControlAdmission.Admit()`), and the handler checks `cs.flowControlAdmission != nil` to handle queue-level outcomes (shed victim accounting, dispatch)

BC-8: Saturation Gating Preserved
- GIVEN `--flow-control` is enabled and cluster saturation >= 1.0
- WHEN `FlowControlAdmission.Admit()` enqueues a request and the handler calls `tryDispatchFromGatewayQueue()`
- THEN the request is held in the queue (saturation gate blocks dispatch) and dispatched later when saturation drops below 1.0

BC-9: Completion-Triggered Dispatch Preserved
- GIVEN `--flow-control` is enabled and requests are held in the queue
- WHEN a request completes on an instance
- THEN `tryDispatchFromGatewayQueue()` is called, dequeuing from the highest-priority non-empty band

BC-10: INV-1 Conservation
- GIVEN any simulation run with `--flow-control` enabled
- WHEN the simulation completes
- THEN `num_requests == injected_requests + rejected_requests` and `injected_requests == completed + still_queued + still_running + dropped_unservable + timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed + gateway_queue_rejected`

**Negative contracts (what MUST NOT happen):**

BC-11: Non-Sheddable Never Evicted in Per-Band Check
- GIVEN per-band capacity is reached for band 3 (all non-sheddable)
- WHEN a new priority-3 request arrives
- THEN the request is rejected (not enqueued), and no existing request is evicted

BC-12: Global Capacity Uses Cross-Band Shedding
- GIVEN global `maxDepth` is reached
- WHEN a new request arrives (regardless of band)
- THEN the existing cross-band shedding logic applies: find the lowest-priority sheddable entry across ALL bands and evict it if the incoming request has higher priority (preserving the Phase 1 criticality protection behavior for the global limit)

**Capacity check order:** Per-band capacity is checked FIRST within the target band. If that passes, global capacity is checked SECOND with cross-band shedding. BC-11 applies only to per-band capacity. BC-12 applies to global capacity.

### C) Component Interaction

```
Request Arrival
    |
    v
ClusterArrivalEvent --> AdmissionDecisionEvent
                              |
                    cs.admissionPolicy.Admit()
                              |
              +---------------+---------------+
              |                               |
        [Legacy path]              [FlowControl path]
        (AlwaysAdmit, etc)        (FlowControlAdmission)
              |                          |
              |                   Enqueue into per-band queue
              |                   Return (true, reason)
              |                          |
              |               Handler checks flowControlAdmission:
              |               - Read LastOutcome/LastShedVictim
              |               - Shed victim accounting
              |               - tryDispatchFromGatewayQueue()
              |                          |
              v                          v
    RoutingDecisionEvent <---------------+
              |
              v
         Instance
```

**Data ownership:**
- `FlowControlAdmission` owns the `GatewayQueue` internally; exposes `Queue()` accessor for completion-triggered dispatch and metrics
- `ClusterSimulator` owns the `AdmissionPolicy` (which may be `FlowControlAdmission` or wrapped in `TenantBudgetAdmission`)
- `ClusterSimulator` stores a typed `*FlowControlAdmission` reference (`cs.flowControlAdmission`) for completion-triggered dispatch and outcome inspection

**INV-1 bucket assignment for FlowControlAdmission:**
- `FlowControlAdmission.Admit()` always returns `admitted=true`. Enqueue outcomes (Rejected, ShedVictim, Enqueued) are exposed via `LastOutcome()`/`LastShedVictim()`.
- The handler NEVER increments `cs.rejectedRequests` for flow-control outcomes — that counter is for admission-level rejections only.
- Queue rejections are tracked by `gatewayQueue.RejectedCount()` (existing INV-1 bucket `gateway_queue_rejected`).
- Shed victims are tracked by `gatewayQueue.ShedCount()` (existing INV-1 bucket `gateway_queue_shed`).
- This matches the current wiring where gateway queue outcomes bypass `cs.rejectedRequests` (cluster_event.go:186-188).

**TenantBudgetAdmission wrapping order:**
`TenantBudgetAdmission` wraps `FlowControlAdmission`. Budget check happens BEFORE enqueue. Over-budget sheddable requests are rejected at the `TenantBudgetAdmission` level and never reach the queue. This is correct: in llm-d, admission filtering (budget, rate limits) precedes the flow controller.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| `GatewayQueue.Remove(requestID)` for TTL | Not implemented | DEFERRAL: Issue says "for TTL (Phase 4)" — explicitly out of scope. |
| `GatewayQueue.LenByBand(priority)` | Implemented | ADDITION: Useful for metrics and testing, mentioned in issue. |
| `--dispatch-order` flag controls heap ordering | Becomes no-op with `logrus.Warnf` when `--flow-control` is enabled | CLARIFICATION: Per-band queues always use priority across bands. We keep the flag for backward compat but warn. |
| Issue says ~400-500 lines | Slightly less due to reusing existing infrastructure | SIMPLIFICATION: Reuse existing `SaturationDetector` and `SLOPriorityMap` without modification. |
| Source says "no separate flow-control branch" | Handler still has `if cs.flowControlAdmission != nil` check | CLARIFICATION: The inline 30-line flow-control block is replaced with a ~15-line check for outcomes + dispatch. A branch remains but is significantly smaller and more focused. |
| Source does not mention `--per-band-capacity` CLI flag | Micro plan adds `--per-band-capacity` flag and `FlowControlPerBandCapacity` config field | ADDITION: Enables per-band queue capacity enforcement described in BC-6 and BC-11. Issue #1191 specifies per-band capacity as a feature; the flag is the user-facing interface. |

### E) Review Guide

**Tricky part:** `FlowControlAdmission.Admit()` always returns `admitted=true` — even when the queue rejects. This is intentional: queue rejections are a separate INV-1 bucket from admission rejections. The handler reads `LastOutcome()` to determine what happened. Scrutinize the INV-1 accounting path in the handler to ensure no double-counting.

**What to scrutinize:** Per-band enqueue/dequeue correctness (especially per-band capacity vs global capacity interaction), shed victim accounting via `LastShedVictim()`, and the handler's outcome-based branching.

**Safe to skim:** CLI flag registration (mechanical), `FlowKey`/`priorityBand`/`flowQueue` type definitions (straightforward data structures).

**Known debt:** Round-robin within-band fairness deferred to Phase 3 (#TBD). Default is global-strict (best head wins). `Dequeue()` in FIFO mode is O(B*F) where B = bands, F = flows per band — acceptable for current BLIS scale (B <= 10, F <= 100 typical).

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to create or modify:
- `sim/cluster/gateway_queue.go` — Replace heap with per-band, per-flow queue structure
- `sim/cluster/gateway_queue_test.go` — New tests for per-band behavior + backward compat
- `sim/cluster/flow_control_admission.go` — New `FlowControlAdmission` policy (in `sim/cluster/` to access `GatewayQueue`)
- `sim/cluster/flow_control_admission_test.go` — Tests for the policy
- `sim/cluster/cluster.go` — Wire `FlowControlAdmission` when `--flow-control` is enabled
- `sim/cluster/cluster_event.go` — Replace inline flow-control block with outcome-based check
- `sim/cluster/deployment.go` — Add `FlowControlPerBandCapacity` field
- `cmd/root.go` — Add `--per-band-capacity` flag, add `--dispatch-order` no-op warning
- `cmd/replay.go` — Wire `--per-band-capacity` flag for replay
- `docs/guide/admission.md` — Document flow control admission and per-band queues
- `CLAUDE.md` — Update Recent Changes section

Key decisions:
- `FlowControlAdmission` lives in `sim/cluster/` (not `sim/`) to avoid import cycle with `GatewayQueue`
- `Admit()` always returns `admitted=true`; enqueue outcomes exposed via `LastOutcome()`/`LastShedVictim()`
- `sim/bundle.go` NOT modified — `FlowControlAdmission` is wired directly when `--flow-control` is enabled, not via the admission policy factory
- The policy owns a `seqCounter int64` for deterministic enqueue ordering (separate from `cs.nextSeqID()` event queue namespace)
- `TenantBudgetAdmission` wraps `FlowControlAdmission` (budget check before enqueue)

No dead code — the old heap-based `GatewayQueue` internals are fully replaced.

### G) Task Breakdown

#### Task 1: Per-Band Queue Data Structure (BC-2, BC-3, BC-4, BC-5)

**Files:** modify `sim/cluster/gateway_queue.go`, test `sim/cluster/gateway_queue_test.go`

**Test (write first — must fail):**

```go
func TestGatewayQueue_PerBand_FlowKeyMapping(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	r1 := &sim.Request{ID: "r1", TenantID: "tenant-A", SLOClass: "standard"} // priority 3
	r2 := &sim.Request{ID: "r2", TenantID: "tenant-B", SLOClass: "standard"} // priority 3
	r3 := &sim.Request{ID: "r3", TenantID: "", SLOClass: "critical"}         // priority 4, default tenant

	q.Enqueue(r1, 1)
	q.Enqueue(r2, 2)
	q.Enqueue(r3, 3)

	// Band 4 (critical) dispatched first
	got := q.Dequeue()
	if got.ID != "r3" {
		t.Errorf("expected r3 (critical band 4), got %s", got.ID)
	}
	// Band 3: r1 and r2 in separate flows. Global-strict: earliest seqID wins.
	got = q.Dequeue()
	if got.ID != "r1" {
		t.Errorf("expected r1 (earliest in band 3), got %s", got.ID)
	}
	got = q.Dequeue()
	if got.ID != "r2" {
		t.Errorf("expected r2, got %s", got.ID)
	}
}

func TestGatewayQueue_PerBand_DefaultTenantID(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "r1", TenantID: "", SLOClass: "standard"}, 1)
	if q.LenByBand(sim.DefaultSLOPriorityMap().Priority("standard")) != 1 {
		t.Errorf("expected 1 request in standard band")
	}
}

func TestGatewayQueue_PerBand_DefaultTenantIDCollision(t *testing.T) {
	// Empty TenantID and explicit "default" TenantID map to the same flow.
	// This is by design: "" → "default" normalization means they share a queue.
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "r1", TenantID: "", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", TenantID: "default", SLOClass: "standard"}, 2)

	// Both in same flow → FIFO order by seqID
	got1 := q.Dequeue()
	got2 := q.Dequeue()
	if got1.ID != "r1" || got2.ID != "r2" {
		t.Errorf("got %s, %s; want r1, r2 (same flow, FIFO by seqID)", got1.ID, got2.ID)
	}
}

func TestGatewayQueue_PerBand_DispatchOrder(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "d", SLOClass: "batch"}, 1)       // band -1
	q.Enqueue(&sim.Request{ID: "b", SLOClass: "standard"}, 2)    // band 3
	q.Enqueue(&sim.Request{ID: "a", SLOClass: "critical"}, 3)    // band 4
	q.Enqueue(&sim.Request{ID: "c", SLOClass: "standard"}, 4)    // band 3

	// Dispatch: band 4 first, then band 3 (FIFO within), then band -1
	expected := []string{"a", "b", "c", "d"}
	for i, exp := range expected {
		got := q.Dequeue()
		if got.ID != exp {
			t.Errorf("position %d: got %s, want %s", i, got.ID, exp)
		}
	}
	if q.Dequeue() != nil {
		t.Error("expected nil from empty queue")
	}
}

func TestGatewayQueue_PerBand_GlobalStrictFairness(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "b1", TenantID: "B", SLOClass: "standard"}, 10)
	q.Enqueue(&sim.Request{ID: "b2", TenantID: "B", SLOClass: "standard"}, 20)
	q.Enqueue(&sim.Request{ID: "c1", TenantID: "C", SLOClass: "standard"}, 5) // earlier seqID

	// Global-strict: pick earliest head across flows -> c1 (seqID=5)
	got := q.Dequeue()
	if got.ID != "c1" {
		t.Errorf("expected c1 (earliest head), got %s", got.ID)
	}
	got = q.Dequeue()
	if got.ID != "b1" {
		t.Errorf("expected b1, got %s", got.ID)
	}
	got = q.Dequeue()
	if got.ID != "b2" {
		t.Errorf("expected b2, got %s", got.ID)
	}
}

func TestGatewayQueue_PerBand_LenByBand(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 0, pm)
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "critical"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2)
	q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)
	q.Enqueue(&sim.Request{ID: "r4", SLOClass: "background"}, 4)

	if q.LenByBand(pm.Priority("critical")) != 1 { t.Errorf("critical band wrong") }
	if q.LenByBand(pm.Priority("standard")) != 2 { t.Errorf("standard band wrong") }
	if q.LenByBand(pm.Priority("background")) != 1 { t.Errorf("background band wrong") }
	if q.LenByBand(99) != 0 { t.Errorf("nonexistent band should be 0") }
	if q.Len() != 4 { t.Errorf("total: got %d", q.Len()) }
}

func TestGatewayQueue_PerBand_CustomSLOPriorities(t *testing.T) {
	// Custom: batch promoted to priority 0 (non-sheddable)
	pm := sim.NewSLOPriorityMap(map[string]int{"batch": 0})
	q := NewGatewayQueue("priority", 0, pm)
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "batch"}, 1)     // band 0 (custom)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2)  // band 3

	got := q.Dequeue()
	if got.ID != "r2" { t.Errorf("expected standard (band 3) first, got %s", got.ID) }
	got = q.Dequeue()
	if got.ID != "r1" { t.Errorf("expected batch (band 0) second, got %s", got.ID) }
}
```

**Impl:** Replace the heap-based `GatewayQueue` internals with per-band, per-flow structure:

```go
type FlowKey struct {
	TenantID string // empty -> "default"
	Priority int
}

type flowEntry struct {
	request *sim.Request
	seqID   int64
}

type flowQueue struct {
	key      FlowKey
	requests []flowEntry // FIFO order
}

type priorityBand struct {
	priority int
	flows    map[string]*flowQueue // tenantID -> per-flow queue
	totalLen int
}
```

`GatewayQueue` holds `bands []*priorityBand` sorted descending by priority (maintained via binary search insertion on band creation — NOT `sort.Slice` on every enqueue). `Enqueue` finds/creates band and flow. `Dequeue` iterates bands highest-first; within each band, picks flow with earliest-seqID head (global-strict fairness). FIFO mode scans ALL bands for globally-earliest seqID. Complexity: `Dequeue()` is O(B*F) where B = distinct priority values and F = max flows per band. Acceptable for BLIS scale (B <= 10, F <= 100).

**INV-6 determinism requirement:** The `flows` field in `priorityBand` is `map[string]*flowQueue`. Go map iteration is non-deterministic. All code that iterates `flows` for shedding victim selection or any operation where iteration order affects the outcome MUST sort map keys first (R2). Helper: `sortedKeys(m map[string]*flowQueue) []string`. `Dequeue()` within-band is safe because it selects by `min(seqID)` across flows — the result is unique regardless of iteration order. Shedding is NOT safe without sorted iteration because multiple entries may tie on sheddability criteria.

**Verify:** `go test ./sim/cluster/... -run TestGatewayQueue_PerBand -v`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): per-band per-flow queue data structure (BC-2, BC-3, BC-4, BC-5)`

---

#### Task 2: Per-Band Capacity and Shedding (BC-6, BC-11, BC-12)

**Files:** modify `sim/cluster/gateway_queue.go`, test `sim/cluster/gateway_queue_test.go`

**Test (write first — must fail):**

```go
func TestGatewayQueue_PerBand_BandCapacityRejection(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetPerBandCapacity(2)

	// Fill band 3 with 2 non-sheddable requests
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2)

	// Third standard -> rejected (band full, no sheddable to evict within band)
	outcome, victim := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)
	if outcome != Rejected { t.Errorf("expected Rejected, got %v", outcome) }
	if victim != nil { t.Error("expected nil victim") }

	// Different band still accepts
	outcome, _ = q.Enqueue(&sim.Request{ID: "r4", SLOClass: "batch"}, 4) // band -1
	if outcome != Enqueued { t.Errorf("expected Enqueued in different band, got %v", outcome) }
}

func TestGatewayQueue_PerBand_BandCapacitySamePriorityEviction(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetPerBandCapacity(2)

	// Fill band -1 with 2 sheddable same-priority requests
	q.Enqueue(&sim.Request{ID: "r1", TenantID: "A", SLOClass: "batch"}, 1) // band -1
	q.Enqueue(&sim.Request{ID: "r2", TenantID: "B", SLOClass: "batch"}, 2) // band -1

	// Band -1 full. Incoming batch with later seqID -> rejected (cannot displace same-priority)
	outcome, _ := q.Enqueue(&sim.Request{ID: "r3", TenantID: "C", SLOClass: "batch"}, 3)
	if outcome != Rejected { t.Errorf("later seqID same-priority should be rejected, got %v", outcome) }
}

func TestGatewayQueue_PerBand_GlobalMaxDepthStillWorks(t *testing.T) {
	// Backward compat: maxDepth=2, no per-band capacity
	q := NewGatewayQueue("priority", 2, nil)
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "critical"}, 2)

	// Queue full. Sheddable -> rejected (no sheddable victims)
	outcome, _ := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "sheddable"}, 3)
	if outcome != Rejected { t.Errorf("expected Rejected, got %v", outcome) }

	if q.Len() != 2 { t.Errorf("expected 2, got %d", q.Len()) }
}

func TestGatewayQueue_PerBand_CapacityCheckOrder(t *testing.T) {
	// Per-band capacity=5, global maxDepth=100
	// Band fills up before global -> per-band rejection
	q := NewGatewayQueue("priority", 100, nil)
	q.SetPerBandCapacity(2)

	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2)

	// Per-band cap reached (2), global not (2 < 100)
	outcome, _ := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)
	if outcome != Rejected { t.Errorf("per-band cap should reject, got %v", outcome) }
}
```

**Impl:** Add `maxBandCapacity int` field to `GatewayQueue` with `SetPerBandCapacity(n int)` setter. Capacity check pseudocode in `Enqueue`:

```go
func (q *GatewayQueue) Enqueue(req *sim.Request, seqID int64) (EnqueueOutcome, *sim.Request) {
	priority := q.priorityMap.Priority(req.SLOClass)
	band := q.findOrCreateBand(priority)
	tenantID := req.TenantID
	if tenantID == "" {
		tenantID = "default"
	}

	// Step 1: Per-band capacity check (within-band shedding only)
	if q.maxBandCapacity > 0 && band.totalLen >= q.maxBandCapacity {
		// Find lowest-priority sheddable entry WITHIN this band only.
		// Iterate flows in sorted tenantID order (determinism — INV-6).
		var victim *flowEntry
		var victimFlow *flowQueue
		sortedTenants := sortedKeys(band.flows) // deterministic iteration
		for _, tid := range sortedTenants {
			flow := band.flows[tid]
			for i := len(flow.requests) - 1; i >= 0; i-- {
				entry := &flow.requests[i]
				if q.priorityMap.IsSheddable(entry.request.SLOClass) &&
					(victim == nil || entry.seqID > victim.seqID) { // oldest sheddable = latest seqID
					victim = entry
					victimFlow = flow
				}
			}
		}
		if victim == nil {
			q.rejectedCount++
			return Rejected, nil
		}
		// Remove victim from its flow, decrement band.totalLen
		evicted := victim.request
		q.removeEntry(victimFlow, band, victim)
		q.shedCount++
		// Fall through to enqueue below
		_ = evicted // returned as shed victim
	}

	// Step 2: Global capacity check (cross-band shedding)
	if q.maxDepth > 0 && q.totalLen >= q.maxDepth {
		// Find lowest-priority sheddable across ALL bands.
		// Same deterministic iteration: sorted bands, sorted tenants within each band.
		// ... (same pattern as existing criticality shedding, adapted for per-band structure)
	}

	// Step 3: Enqueue into target band/flow
	flow := band.findOrCreateFlow(tenantID)
	flow.requests = append(flow.requests, flowEntry{request: req, seqID: seqID})
	band.totalLen++
	q.totalLen++
	return Enqueued, nil
}
```

Key implementation detail: all map iterations in shedding paths MUST use sorted keys (R2, INV-6) to ensure deterministic victim selection.

**Verify:** `go test ./sim/cluster/... -run TestGatewayQueue_PerBand -v`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): per-band capacity enforcement with criticality shedding (BC-6, BC-11, BC-12)`

---

#### Task 3: Backward Compatibility — All Existing Tests Pass (BC-1)

**Files:** modify `sim/cluster/gateway_queue.go` if needed

**Test:** All existing `TestGatewayQueue_*` tests must pass unchanged:

```bash
go test ./sim/cluster/... -run TestGatewayQueue -v
```

The refactored `GatewayQueue` uses per-band structure internally but exposes the same API: `NewGatewayQueue(dispatchOrder, maxDepth, priorityMap)`, `Enqueue(req, seqID)`, `Dequeue()`, `Len()`, `ShedCount()`, `RejectedCount()`.

FIFO mode: when `dispatchOrder="fifo"`, `Dequeue()` scans all bands for the globally-earliest seqID. This is O(B*F) but matches existing FIFO behavior.

**Verify:** `go test ./sim/cluster/... -run TestGatewayQueue -v` — all 11 existing tests pass
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `test(cluster): verify backward compatibility of per-band queue (BC-1)`

---

#### Task 4: FlowControlAdmission Policy (BC-7, BC-8)

**Files:** create `sim/cluster/flow_control_admission.go`, create `sim/cluster/flow_control_admission_test.go`

**Test (write first — must fail):**

```go
func TestFlowControlAdmission_Enqueue(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 0, pm)
	fc := NewFlowControlAdmission(q, pm)

	req := &sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1"}
	state := &sim.RouterState{Clock: 100}

	admitted, _ := fc.Admit(req, state)
	if !admitted {
		t.Error("FlowControlAdmission.Admit must always return true")
	}
	if fc.LastOutcome() != Enqueued {
		t.Errorf("expected Enqueued, got %v", fc.LastOutcome())
	}
	if q.Len() != 1 {
		t.Errorf("expected 1 in queue, got %d", q.Len())
	}
	if req.GatewayEnqueueTime != 100 {
		t.Errorf("expected enqueue time 100, got %d", req.GatewayEnqueueTime)
	}
}

func TestFlowControlAdmission_QueueRejection(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 1, pm) // maxDepth=1
	fc := NewFlowControlAdmission(q, pm)

	// Fill queue
	fc.Admit(&sim.Request{ID: "r1", SLOClass: "standard"}, &sim.RouterState{Clock: 100})
	// Second request -> queue-rejected
	r2 := &sim.Request{ID: "r2", SLOClass: "standard"}
	admitted, reason := fc.Admit(r2, &sim.RouterState{Clock: 200})

	if !admitted {
		t.Error("FlowControlAdmission.Admit must always return true (even for queue rejection)")
	}
	if fc.LastOutcome() != Rejected {
		t.Errorf("expected Rejected outcome, got %v", fc.LastOutcome())
	}
	if reason != "flow-control-queue-rejected" {
		t.Errorf("expected queue-rejected reason, got %q", reason)
	}
	if r2.GatewayEnqueueTime != 0 {
		t.Error("rejected request should have GatewayEnqueueTime cleared to 0")
	}
}

func TestFlowControlAdmission_ShedVictim(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 1, pm) // maxDepth=1
	fc := NewFlowControlAdmission(q, pm)

	// Fill with sheddable
	fc.Admit(&sim.Request{ID: "r1", SLOClass: "batch"}, &sim.RouterState{Clock: 100})
	// Higher priority -> shed victim
	fc.Admit(&sim.Request{ID: "r2", SLOClass: "standard"}, &sim.RouterState{Clock: 200})

	if fc.LastOutcome() != ShedVictim {
		t.Errorf("expected ShedVictim, got %v", fc.LastOutcome())
	}
	victim := fc.LastShedVictim()
	if victim == nil || victim.ID != "r1" {
		t.Errorf("expected victim r1, got %v", victim)
	}
	if victim.GatewayEnqueueTime != 0 {
		t.Error("shed victim should have GatewayEnqueueTime cleared")
	}
}
```

**Impl:**

```go
// FlowControlAdmission implements sim.AdmissionPolicy by delegating to a
// per-band gateway queue. When flow control is enabled, this policy replaces
// the legacy admission policy — the queue IS the admission decision.
// Matches llm-d's FlowControlAdmissionController.
//
// Admit() always returns admitted=true. Enqueue outcomes are exposed via
// LastOutcome() and LastShedVictim() for the handler to read.
// This is because queue rejections are a separate INV-1 bucket from
// admission rejections — they must NOT increment cs.rejectedRequests.
type FlowControlAdmission struct {
	queue       *GatewayQueue
	priorityMap *sim.SLOPriorityMap
	seqCounter  int64
	lastOutcome EnqueueOutcome
	lastVictim  *sim.Request
}

func NewFlowControlAdmission(queue *GatewayQueue, priorityMap *sim.SLOPriorityMap) *FlowControlAdmission {
	if queue == nil { panic("FlowControlAdmission: queue must not be nil") }
	if priorityMap == nil { priorityMap = sim.DefaultSLOPriorityMap() }
	return &FlowControlAdmission{queue: queue, priorityMap: priorityMap}
}

func (fc *FlowControlAdmission) Admit(req *sim.Request, state *sim.RouterState) (bool, string) {
	fc.lastVictim = nil
	req.GatewayEnqueueTime = state.Clock
	fc.seqCounter++
	outcome, victim := fc.queue.Enqueue(req, fc.seqCounter)
	fc.lastOutcome = outcome

	switch outcome {
	case Rejected:
		req.GatewayEnqueueTime = 0
		return true, "flow-control-queue-rejected"
	case ShedVictim:
		if victim != nil {
			victim.GatewayEnqueueTime = 0
			fc.lastVictim = victim
		}
		return true, "flow-control-enqueued"
	case Enqueued:
		return true, "flow-control-enqueued"
	default:
		panic(fmt.Sprintf("FlowControlAdmission: unhandled EnqueueOutcome %d", outcome))
	}
}

func (fc *FlowControlAdmission) Queue() *GatewayQueue { return fc.queue }
func (fc *FlowControlAdmission) LastOutcome() EnqueueOutcome { return fc.lastOutcome }
func (fc *FlowControlAdmission) LastShedVictim() *sim.Request { return fc.lastVictim }
```

**Verify:** `go test ./sim/cluster/... -run TestFlowControlAdmission -v`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): FlowControlAdmission policy — queue IS admission (BC-7, BC-8)`

---

#### Task 5: Wire FlowControlAdmission into ClusterSimulator (BC-7, BC-9, BC-10)

**Files:** modify `sim/cluster/cluster.go`, `sim/cluster/cluster_event.go`, `sim/cluster/deployment.go`

**Test:** All existing integration tests must pass: `go test ./sim/cluster/... -count=1`

**Impl:**

In `deployment.go`, add to the flow control config block:
```go
FlowControlPerBandCapacity int `yaml:"flow_control_per_band_capacity,omitempty"` // 0 = unlimited
```

In `cluster.go`, add field to `ClusterSimulator`:
```go
flowControlAdmission *FlowControlAdmission // typed ref for outcome inspection + completion dispatch; nil when flow control disabled
```

In `NewClusterSimulator`, replace the flow control initialization block (lines 399-417) with:
```go
if config.FlowControlEnabled {
	dispatchOrder := config.FlowControlDispatchOrder
	if dispatchOrder == "" {
		dispatchOrder = "fifo"
	}
	gq := NewGatewayQueue(dispatchOrder, config.FlowControlMaxQueueDepth, cs.priorityMap)
	if config.FlowControlPerBandCapacity > 0 {
		gq.SetPerBandCapacity(config.FlowControlPerBandCapacity)
	}
	cs.gatewayQueue = gq // keep pointer for accessors (GatewayQueueDepth etc.)
	cs.flowControlEnabled = true
	cs.saturationDetector = sim.NewSaturationDetector(...)
	fcAdmission := NewFlowControlAdmission(gq, cs.priorityMap)
	cs.flowControlAdmission = fcAdmission
	cs.admissionPolicy = fcAdmission // replaces legacy policy
}
```

After admission policy setup, if tenant budgets are configured, wrap as before:
```go
if cs.tenantTracker != nil {
	cs.admissionPolicy = sim.NewTenantBudgetAdmission(cs.admissionPolicy, cs.tenantTracker, cs.priorityMap)
}
```

In `cluster_event.go`, replace the `if cs.flowControlEnabled { ... }` block in `AdmissionDecisionEvent.Execute` (lines 178-208) with:
```go
if cs.flowControlAdmission != nil {
	// FlowControlAdmission.Admit() always returns admitted=true.
	// Handle queue-level outcomes.
	switch cs.flowControlAdmission.LastOutcome() {
	case Rejected:
		// Queue-rejected. NOT an admission rejection.
		// Accounted via gatewayQueue.RejectedCount() (INV-1: gateway_queue_rejected).
		return
	case ShedVictim:
		victim := cs.flowControlAdmission.LastShedVictim()
		if victim != nil {
			tier := victim.SLOClass
			if tier == "" { tier = "standard" }
			cs.shedByTier[tier]++
		}
	case Enqueued:
		// nothing extra
	}
	cs.tryDispatchFromGatewayQueue()
	return
}
```

For completion-triggered dispatch: keep the existing `if c.flowControlEnabled` loop in `cluster.go` that calls `tryDispatchFromGatewayQueue()` — it works unchanged because `cs.gatewayQueue` still points to the same queue.

**Verify:** `go test ./sim/cluster/... -count=1`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): wire FlowControlAdmission + simplify AdmissionDecisionEvent (BC-7, BC-9, BC-10)`

---

#### Task 6: CLI Flag + Documentation (BC-6)

**Files:** modify `cmd/root.go`, `cmd/replay.go`, `docs/guide/admission.md`, `CLAUDE.md`

**Test:** `go test ./cmd/... -count=1` — existing flag tests pass.

**Impl:**

In `cmd/root.go`:
- Add `var flowControlPerBandCapacity int`
- Register flag: `cmd.Flags().IntVar(&flowControlPerBandCapacity, "per-band-capacity", 0, "Max requests per priority band when --flow-control is enabled (0=unlimited)")`
- Validate: if `< 0`, `logrus.Fatalf`
- Wire to `DeploymentConfig.FlowControlPerBandCapacity`
- Add warning: if `flowControlEnabled && flowControlDispatchOrder != "fifo"`, emit `logrus.Warnf("--dispatch-order is ignored when --flow-control is enabled (per-band queues always use priority dispatch)")`

In `cmd/replay.go`:
- Add same `var flowControlPerBandCapacity int` declaration
- Register same flag: `replayCmd.Flags().IntVar(&flowControlPerBandCapacity, "per-band-capacity", 0, "Max requests per priority band when --flow-control is enabled (0=unlimited)")`
- Wire to `DeploymentConfig.FlowControlPerBandCapacity` in the replay config builder (near existing `FlowControlMaxQueueDepth` wiring, around line 236)
- Add same `--dispatch-order` no-op warning

In `cmd/simconfig_shared_test.go`, add `"per-band-capacity"` to the flow control flag test list. In `cmd/replay_test.go`, add the same.

In `CLAUDE.md`, add to Recent Changes (top of list):
```
- Per-priority-band queues + FlowControlAdmission policy (#1191): `GatewayQueue` replaced with per-band, per-flow queue structure keyed by (TenantID, Priority). `FlowControlAdmission` policy merges admission and queuing when `--flow-control` is enabled, matching llm-d architectural parity. Per-band capacity enforcement via `--per-band-capacity` flag (0=unlimited). `--dispatch-order` becomes a no-op when flow control is enabled. Closes #1191, part of #899.
```

In `docs/guide/admission.md`, add after "## GAIE-Legacy Admission":

```markdown
## Flow Control Admission

When `--flow-control` is enabled, the `FlowControlAdmission` policy replaces the configured
admission policy. In this mode, admission and queuing are a single step — the queue IS the
admission decision, matching llm-d's `FlowControlAdmissionController`.

### How It Works

1. Incoming request → enqueue into per-priority-band, per-flow queue
2. Each unique (TenantID, Priority) pair gets its own FIFO queue within a priority band
3. Dispatch iterates bands highest-priority first (strict priority across bands)
4. Within a band, the request with the earliest arrival (lowest sequence ID) is dispatched first (global-strict fairness)
5. Saturation gating: dispatch only when cluster saturation < 1.0
6. Completion-triggered dispatch: each completion frees capacity and tries to dispatch from the queue

### Per-Band Capacity

| Flag | Description | Default |
|------|-------------|---------|
| `--per-band-capacity` | Max requests per priority band (0=unlimited) | 0 |

When a band reaches its capacity limit, incoming requests for that band are rejected unless a sheddable
entry within the band can be evicted. The global `--max-gateway-queue-depth` limit applies across all bands.

### Example

\```bash
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8 \
  --per-band-capacity 100 --max-gateway-queue-depth 500
\```

### Comparison with Legacy Admission

| Aspect | Legacy (AlwaysAdmit, TierShed, etc.) | FlowControlAdmission |
|--------|--------------------------------------|---------------------|
| Admission | Separate from queuing | Queue IS admission |
| Queue structure | Single heap | Per-priority-band, per-flow |
| Dispatch order | `--dispatch-order` (fifo/priority) | Always priority across bands |
| Capacity | Global only | Per-band + global |
```

In `CLAUDE.md`, add to Recent Changes.

**Verify:** `go test ./cmd/... -count=1`
**Lint:** `golangci-lint run ./cmd/...`
**Commit:** `feat(cmd): --per-band-capacity flag + flow control documentation (BC-6)`

---

#### Task 7: End-to-End Integration Test (BC-10)

**Files:** add test in `sim/cluster/flow_control_admission_test.go`

**Test (write first — must fail):**

```go
func TestFlowControlAdmission_INV1_Conservation(t *testing.T) {
	// Build config with flow control + per-band capacity.
	config := newTestDeploymentConfig(2) // 2 instances
	config.FlowControlEnabled = true
	config.FlowControlDetector = "utilization"
	config.FlowControlDispatchOrder = "priority"
	config.FlowControlMaxQueueDepth = 5
	config.FlowControlPerBandCapacity = 3
	config.FlowControlQueueDepthThreshold = 3
	config.FlowControlKVCacheUtilThreshold = 0.8

	// Inject 20 mixed-SLO requests across 4 priority bands.
	requests := make([]*sim.Request, 20)
	sloClasses := []string{"critical", "standard", "batch", "sheddable", "background"}
	for i := 0; i < 20; i++ {
		requests[i] = &sim.Request{
			ID:            fmt.Sprintf("r%d", i),
			TenantID:      fmt.Sprintf("tenant-%d", i%3),
			SLOClass:      sloClasses[i%len(sloClasses)],
			ArrivalTime:   int64(i * 100_000), // 100ms apart
			PromptTokens:  100,
			OutputTokens:  50,
			MaxOutputLen:  200,
		}
	}

	cs := NewClusterSimulator(config, requests, nil)
	if err := cs.Run(); err != nil {
		t.Fatalf("simulation failed: %v", err)
	}

	results := cs.Results()

	// INV-1 full pipeline: num_requests == injected + rejected
	numRequests := len(requests)
	injected := results.InjectedRequests
	rejected := results.RejectedRequests
	if numRequests != injected+rejected {
		t.Errorf("INV-1 pipeline: num_requests=%d != injected=%d + rejected=%d",
			numRequests, injected, rejected)
	}

	// INV-1 conservation: injected == completed + still_queued + still_running +
	//   dropped + timed_out + routing_rejections + gw_queue_depth + gw_queue_shed + gw_queue_rejected
	completed := results.CompletedRequests
	queued := results.StillQueued
	running := results.StillRunning
	dropped := results.DroppedUnservable
	timedOut := results.TimedOut
	routingRej := results.RoutingRejections
	gwDepth := cs.GatewayQueueDepth()
	gwShed := cs.GatewayQueueShed()
	gwRejected := cs.GatewayQueueRejected()

	conserved := completed + queued + running + dropped + timedOut + routingRej + gwDepth + gwShed + gwRejected
	if injected != conserved {
		t.Errorf("INV-1 conservation: injected=%d != completed=%d + queued=%d + running=%d + "+
			"dropped=%d + timedOut=%d + routingRej=%d + gwDepth=%d + gwShed=%d + gwRejected=%d (sum=%d)",
			injected, completed, queued, running, dropped, timedOut, routingRej,
			gwDepth, gwShed, gwRejected, conserved)
	}

	// Verify per-band capacity caused some rejections or shedding
	// (with maxDepth=5, perBandCapacity=3, 20 requests — queue MUST overflow)
	if gwShed+gwRejected == 0 {
		t.Error("expected some gateway queue shedding or rejection with tight capacity limits")
	}
}
```

**Impl:** No additional production code — this test exercises the full wiring from Tasks 1-6. If the test fails, it means the wiring or accounting is broken. Fix by tracing the INV-1 buckets.

This is the ultimate behavioral test that the new wiring preserves INV-1.

**Verify:** `go test ./sim/cluster/... -run TestFlowControlAdmission_INV1 -v`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `test(cluster): INV-1 conservation for FlowControlAdmission (BC-10)`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 3 | Regression | All existing TestGatewayQueue_* tests (11 tests) |
| BC-2 | Task 1 | Unit | TestGatewayQueue_PerBand_FlowKeyMapping |
| BC-3 | Task 1 | Unit | TestGatewayQueue_PerBand_DefaultTenantID |
| BC-4 | Task 1 | Unit | TestGatewayQueue_PerBand_DispatchOrder |
| BC-5 | Task 1 | Unit | TestGatewayQueue_PerBand_GlobalStrictFairness |
| BC-6, BC-11 | Task 2 | Unit | TestGatewayQueue_PerBand_BandCapacityRejection |
| BC-12 | Task 2 | Unit | TestGatewayQueue_PerBand_GlobalMaxDepthStillWorks |
| BC-7 | Task 4 | Unit | TestFlowControlAdmission_Enqueue |
| BC-7 | Task 4 | Unit | TestFlowControlAdmission_QueueRejection |
| BC-7 | Task 4 | Unit | TestFlowControlAdmission_ShedVictim |
| BC-8, BC-9 | Task 5 | Integration | Existing cluster tests with flow control |
| BC-10 | Task 7 | Integration | TestFlowControlAdmission_INV1_Conservation |
| Custom priorities | Task 1 | Unit | TestGatewayQueue_PerBand_CustomSLOPriorities |
| BC-3 (collision) | Task 1 | Unit | TestGatewayQueue_PerBand_DefaultTenantIDCollision |

Key invariants verified:
- **INV-1** (request conservation): Task 7 integration test
- **INV-5** (causality): GatewayEnqueueTime set on Admit(), GatewayDispatchTime set on dispatch
- **INV-6** (determinism): seqCounter provides deterministic ordering (separate namespace from event seqIDs)

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| INV-1 conservation broken by Admit() always returning true | Medium | High | Explicit accounting in handler; dedicated integration test | Task 5, 7 |
| Circular dependency (sim/ <-> sim/cluster/) | N/A | N/A | FlowControlAdmission in sim/cluster/ (resolved at design time) | Task 4 |
| FIFO mode regression | Low | Medium | All 11 existing tests preserved | Task 3 |
| Per-band + global capacity interaction confusion | Medium | Medium | Explicit capacity check order documented; dedicated tests | Task 2 |
| TenantBudget wrapping order breaks flow control | Low | High | Wrapping order documented; budget checks before enqueue | Task 5 |
| O(B*F) dequeue complexity at scale | Low | Low | Acceptable for BLIS scale; documented for future optimization | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `FlowControlAdmission` implements existing `AdmissionPolicy` interface
- [x] No feature creep — round-robin fairness explicitly deferred to Phase 3
- [x] No unexercised flags — `--per-band-capacity` tested via Task 2 and Task 7
- [x] No partial implementations — all 4 behaviors from the issue are covered
- [x] No breaking changes — BC-1 ensures backward compatibility
- [x] No hidden global state — all state owned by `FlowControlAdmission` or `GatewayQueue`
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md updated (Task 6)
- [x] No stale references in CLAUDE.md
- [x] INV-9 boundary: `FlowControlAdmission.Admit()` reads only `SLOClass`, `TenantID`, and `GatewayEnqueueTime` — does NOT read `OutputTokens`
- [x] Documentation DRY — admission.md is the canonical source for admission docs
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1->2->3->4->5->6->7)
- [x] All contracts mapped to specific tasks
- [x] Construction site audit completed (Phase 0)
- [x] sim/bundle.go NOT modified (FlowControlAdmission not registered in factory)

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — all enqueue outcomes handled with explicit switch
- [x] R2: Map keys — per-band iteration uses sorted slice, not map iteration. Shedding paths MUST sort `flows` map keys before iteration (INV-6 determinism)
- [x] R3: `--per-band-capacity` validated (>= 0)
- [x] R4: Construction sites audited (Phase 0)
- [x] R5: No mid-loop allocation rollback needed
- [x] R6: No `logrus.Fatalf` in `sim/` packages
- [x] R7: Integration test (Task 7) validates conservation invariant
- [x] R8: No exported mutable maps — `flows` map is unexported within `priorityBand`
- [x] R9: N/A (`FlowControlPerBandCapacity` is `int`, zero unambiguously means unlimited; no pointer type needed)
- [x] R10: YAML strict parsing inherited
- [x] R11: No division by runtime-derived denominators
- [x] R12: No golden dataset changes
- [x] R13: No new interfaces — uses existing `AdmissionPolicy`
- [x] R14: No method spans multiple module responsibilities
- [x] R15: No stale PR references
- [x] R16: Config grouped in `DeploymentConfig` flow control section
- [x] R17: N/A (no new routing scorer signals)
- [x] R18: N/A (no defaults.yaml interaction)
- [x] R19: No unbounded retry loops
- [x] R20: Empty queue/band handled (return nil/0)
- [x] R21: No range over shrinking slices
- [x] R22: N/A
- [x] R23: N/A
