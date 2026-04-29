# Phase 1: Criticality-Based Shedding + TenantBudget Admission Policy

**Goal:** Fix the gateway queue to never evict protected requests (priority >= 0), and move tenant budget enforcement from inline event handler code into a proper `AdmissionPolicy` implementation.

**The problem today:**
1. `GatewayQueue.Enqueue()` evicts the lowest-priority queued request on overflow regardless of criticality. A `critical:4` request can be evicted. GIE guarantees only `priority < 0` requests are eviction candidates.
2. Tenant budget enforcement is hardcoded inline in `cluster_event.go`'s `AdmissionDecisionEvent.Execute()` as a special `if` block, bypassing the `AdmissionPolicy` interface. All admit/reject decisions should go through `AdmissionPolicy.Admit()`.

**What this PR adds:**
1. Criticality-aware shedding: only `priority < 0` entries are eviction candidates in the gateway queue
2. Rejection semantics: when queue is full and no sheddable candidate exists (or incoming has lower priority than all sheddable candidates), the incoming request is rejected
3. `EnqueueOutcome` type with three states: `Enqueued`, `ShedVictim`, `Rejected`
4. `TenantBudgetAdmission` policy that wraps any inner `AdmissionPolicy` and applies tenant budget enforcement through the standard `Admit()` interface

**Why this matters:** Algorithms developed in BLIS must transfer cleanly to llm-d. Both fixes align BLIS with GIE's admission contracts — criticality protection and unified admission interface.

**Architecture:** Changes span three packages: `sim/` (new types and policy in `admission.go`), `sim/cluster/` (shedding fix in `gateway_queue.go`, wiring cleanup in `cluster_event.go` and `cluster.go`), and `cmd/` (metric output wiring in `root.go`). The `TenantBudgetAdmission` uses the decorator pattern — it wraps an inner `AdmissionPolicy`, delegates `Admit()` first, then applies budget checks. The `tenantTracker` field stays on `ClusterSimulator` (needed by 6+ `OnStart`/`OnComplete` call sites in routing and completion events); only the inline admission check is removed from `cluster_event.go`. No new packages or interfaces needed.

**Source:** #1190 (Phase 1 of #899 — GIE flow control parity)
**Closes:** `Fixes #1190, fixes #1020`

**Behavioral Contracts:** See Part 1, Section B below.

---

## Phase 0: Component Context

1. **Building block:** Gateway queue shedding logic (`sim/cluster/gateway_queue.go`) and admission policy composition (`sim/admission.go`)
2. **Adjacent blocks:** `cluster_event.go` (event pipeline that calls both), `cluster.go` (initialization and dispatch), `metrics.go` (counters)
3. **Invariants touched:** INV-1 (request conservation — new `gw_rejected` bucket)
4. **Construction Site Audit:**
   - `GatewayQueue` struct (`gateway_queue.go:55`): constructed in `cluster.go:401` (`NewGatewayQueue(...)`) — single site
   - `TenantTracker` struct (`tenant.go:12`): constructed in `cluster.go:388` (`NewTenantTracker(...)`) — single site
   - `ClusterSimulator` struct (`cluster.go:40`): constructed in `cluster.go:121` (`NewClusterSimulator(...)`) — single site. Field `tenantTracker` at line 68 stays (needed by 6+ OnStart/OnComplete call sites); inline admission check removed from cluster_event.go.

---

## Part 1: Design Validation

### A) Executive Summary

This PR makes two correctness fixes to align BLIS with GIE/llm-d admission semantics:

1. **Criticality shedding fix** (~30 lines in `gateway_queue.go`): Restrict eviction candidates to `priority < 0` entries. When the queue is full and no sheddable candidate exists, reject the incoming request instead of evicting a protected one. Change `Enqueue` return type from `shed bool` to `(EnqueueOutcome, *Request)` with three states.

2. **TenantBudget admission policy** (~80 lines in `admission.go`): Create `TenantBudgetAdmission` implementing `AdmissionPolicy` using the decorator pattern — wraps an inner policy, delegates `Admit()` first, then checks tenant budget. Remove inline tenant budget check from `cluster_event.go`. The `tenantTracker` field stays on `ClusterSimulator` (needed by 6+ `OnStart`/`OnComplete` call sites).

Adjacent components: `cluster_event.go` (update `Enqueue` call site to handle new return type), `cluster.go` (wire `TenantBudgetAdmission` wrapper), `metrics.go` (add `GatewayQueueRejected` counter).

### B) Behavioral Contracts

#### Positive contracts (what MUST happen)

```
BC-1: Sheddable eviction
- GIVEN gateway queue at maxDepth with entries including at least one with priority < 0
- WHEN a new request with higher priority than the lowest sheddable entry arrives
- THEN the lowest-priority entry with priority < 0 is evicted (ShedVictim outcome)
- MECHANISM: Enqueue scans for min-priority entry among priority < 0 only, then checks incoming > victim priority before displacing

BC-2: Rejection when no sheddable victim or incoming not higher priority
- GIVEN gateway queue at maxDepth
- WHEN a new request arrives AND (no queued entry has priority < 0, OR the incoming request has lower/equal priority to the lowest sheddable entry)
- THEN the incoming request is rejected (Rejected outcome), no queued entry is evicted
- MECHANISM: Enqueue finds no sheddable candidate OR fails priority comparison, returns Rejected

BC-3: Normal enqueue below capacity
- GIVEN gateway queue below maxDepth (or maxDepth=0, unlimited)
- WHEN any request arrives
- THEN the request is enqueued (Enqueued outcome), no eviction
- MECHANISM: heap.Push without capacity check

BC-4: EnqueueOutcome return type
- GIVEN any call to Enqueue
- WHEN the method returns
- THEN the outcome is exactly one of: Enqueued (no victim), ShedVictim (victim returned as second value), Rejected (incoming request not enqueued)
- MECHANISM: (EnqueueOutcome, *Request) return type

BC-5: TenantBudget via AdmissionPolicy
- GIVEN TenantBudgetAdmission wrapping an inner AdmissionPolicy
- WHEN Admit(req, state) is called
- THEN the inner policy's Admit() is called first; if the inner policy rejects, the rejection stands; if the inner policy admits, the tenant budget is checked
- MECHANISM: Decorator pattern — delegate then check

BC-6: TenantBudget protects non-sheddable
- GIVEN a request with priority >= 0 (non-sheddable) and its tenant is over budget
- WHEN TenantBudgetAdmission.Admit() is called
- THEN the request is admitted (budget enforcement does not apply to non-sheddable requests)
- MECHANISM: IsSheddable(req.SLOClass) check before budget rejection

BC-7: TenantBudget rejects sheddable over-budget
- GIVEN a request with priority < 0 (sheddable) and its tenant is over budget
- WHEN TenantBudgetAdmission.Admit() is called
- THEN the request is rejected with reason "tenant-budget-shed"
- MECHANISM: IsOverBudget(tenantID) && IsSheddable(class)
```

#### Negative contracts (what MUST NOT happen)

```
BC-8: Non-sheddable never evicted from queue
- GIVEN any request with priority >= 0 in the gateway queue
- WHEN the queue is at capacity and a new request arrives
- THEN the priority >= 0 request is never selected as eviction victim
- MECHANISM: Eviction candidate scan filters to priority < 0 only

BC-9: No inline admission logic in event handlers
- GIVEN the cluster event pipeline
- WHEN AdmissionDecisionEvent.Execute() runs
- THEN all admit/reject decisions go through AdmissionPolicy.Admit() — no inline tenantTracker.IsOverBudget check
- MECHANISM: Inline tenant budget block (cluster_event.go:160-181) removed; TenantBudgetAdmission wraps the policy. tenantTracker field stays on ClusterSimulator for OnStart/OnComplete routing callbacks (not admission decisions).
```

#### Conservation contract

```
BC-10: INV-1 conservation with gw_rejected
- GIVEN any simulation run with flow control enabled
- WHEN the simulation completes
- THEN injected == completed + queued + running + dropped + timed_out + routing_rejections + gw_depth + gw_shed + gw_rejected
- MECHANISM: GatewayQueueRejected counter added; rejected requests counted in conservation formula
```

### C) Component Interaction

```
Request → ClusterArrivalEvent
           → AdmissionDecisionEvent
               → AdmissionPolicy.Admit(req, state)
                   ├── [inner policy: AlwaysAdmit/TierShed/GAIELegacy]
                   └── [if TenantBudgets configured: TenantBudgetAdmission wraps inner]
               → if rejected: increment rejectedRequests, return
               → if flow control enabled:
                   → GatewayQueue.Enqueue(req, seqID) → (EnqueueOutcome, *Request)
                       ├── Enqueued: proceed to dispatch
                       ├── ShedVictim: log victim, proceed to dispatch
                       └── Rejected: increment rejectedRequests, return
                   → tryDispatchFromGatewayQueue()
               → else: schedule RoutingDecisionEvent
```

State ownership:
- `GatewayQueue` owns: heap entries, shedCount, rejectedCount
- `TenantBudgetAdmission` owns: reference to TenantTracker + inner policy + priorityMap
- `ClusterSimulator` owns: rejectedRequests counter, shedByTier map, tenantTracker (kept for OnStart/OnComplete; admission check moved to TenantBudgetAdmission)

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue proposes `EnqueueOutcome` as a standalone type | Defined in `gateway_queue.go` (same package as `GatewayQueue`) | SIMPLIFICATION — no need for a separate file; the type is only used by `GatewayQueue` |
| Issue mentions removing `tenantTracker` field from `ClusterSimulator` | Keeping the field; only removing inline admission check from cluster_event.go | CORRECTION — The `tenantTracker` field is used by 6+ call sites across `cluster_event.go`, `pd_events.go`, and `cluster.go` for `OnStart`/`OnComplete` routing and completion callbacks. These are NOT admission decisions. Removing the field would require restructuring all event handlers. Instead, only the inline `IsOverBudget` admission check (cluster_event.go:160-181) is removed; `TenantBudgetAdmission` gets a reference to the same `TenantTracker` instance for admission decisions. |
| Issue says "Add rejectedCount counter, extend INV-1" | Add both `GatewayQueueRejected` metric field on `RawMetrics` AND `rejectedCount` on `GatewayQueue` | CLARIFICATION — two counters needed: one on the queue (data structure) and one on metrics (output) |
| Per-tier tracking gap for gateway-shed requests | Not fixed in this PR | DEFERRAL — known accounting gap (gateway-shed requests not tracked per-tier). Separate issue. |
| Existing test expects critical:4 displaces standard:3 at full queue | Test updated: non-sheddable entries cannot displace each other | CORRECTION — old behavior allowed any displacement; new behavior only allows sheddable victims |
| GIE enqueue and eviction are separate operations (EPP enqueues, SheddableFilter evicts independently) | BLIS performs atomic enqueue+evict in a single `Enqueue` call | DES SIMPLIFICATION — In BLIS's single-threaded DES, there is no concurrency between enqueue and eviction. The atomic operation produces identical observable behavior (same victim selected, same outcome) while being simpler to reason about. This is a deliberate DES modeling simplification, not a GIE parity gap. |
| Existing `shedCount` counted both evicted victims and rejected incoming requests | Split into `shedCount` (evicted victims only) and `rejectedCount` (rejected incoming only) | SEMANTIC NARROWING — The old `shedCount` conflated two distinct outcomes. Splitting them aligns with GIE's separate accounting for shed vs rejected and enables the new `gw_rejected` INV-1 bucket. Old `ShedCount()` accessor retains its name but now only counts evictions. |

### E) Review Guide

**Tricky part:** The `Enqueue` method change is the core logic — verify the eviction candidate scan correctly filters to `priority < 0` only, that the priority comparison guard prevents lower-priority incoming from evicting higher-priority sheddable entries, and that the three outcomes are exhaustive. Pay attention to the edge case where the incoming request itself is sheddable but has lower priority than all sheddable queued entries (should be rejected, not displace them).

**Scrutinize:** The `TenantBudgetAdmission` decorator — verify it delegates to the inner policy first and only checks budget on admitted requests. Verify `OnStart`/`OnComplete` wiring still works after the field move.

**Safe to skim:** Test helper functions, metrics plumbing, documentation updates.

**Known debt:** Per-tier tracking for gateway-shed requests is not addressed (separate issue). Trace gap: requests admitted but subsequently gateway-rejected have `Admitted:true` in trace with no corresponding rejection trace event (pre-existing gap, now affects a broader class including non-sheddable requests). Equal-priority sheddable displacement test deferred (covered by priority comparison guard logic but no dedicated test).

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action | Purpose |
|------|--------|---------|
| `sim/cluster/gateway_queue.go` | Modify | Add `EnqueueOutcome` type, fix `Enqueue` to use criticality check, add `rejectedCount` |
| `sim/cluster/gateway_queue_test.go` | Modify | Update shedding tests for new semantics, add criticality protection tests |
| `sim/admission.go` | Modify | Add `TenantBudgetAdmission` type |
| `sim/admission_test.go` | Modify | Add tests for `TenantBudgetAdmission` |
| `sim/cluster/cluster_event.go` | Modify | Handle `EnqueueOutcome`, remove inline tenant budget check |
| `sim/cluster/cluster.go` | Modify | Wire `TenantBudgetAdmission`, add `GatewayQueueRejected()` accessor |
| `sim/cluster/cluster_tenant_test.go` | Modify | Update tenant tests for new policy-based enforcement |
| `sim/cluster/metrics.go` | Modify | Add `GatewayQueueRejected` field |
| `cmd/root.go` | Modify | Wire `GatewayQueueRejected` into metric output and anomaly condition |
| `sim/bundle.go` | No change | No new named policy (TenantBudget is auto-composed, not a CLI choice) |
| `docs/contributing/standards/invariants.md` | Modify | Update INV-1 formula with `gw_rejected` |
| `CLAUDE.md` | Modify | Update INV-1 formula, add recent changes entry |

Key decisions:
- `EnqueueOutcome` lives in `sim/cluster/` (same package as `GatewayQueue`)
- `TenantBudgetAdmission` lives in `sim/` (same package as other admission policies)
- No new CLI flags — tenant budgets remain YAML-only via `PolicyBundle.TenantBudgets`
- `TenantTracker` struct stays in `sim/cluster/tenant.go` unchanged — field stays on `ClusterSimulator` (needed by `OnStart`/`OnComplete` call sites in routing and completion events). `TenantBudgetAdmission` gets a reference to the same instance for admission decisions. Only the inline `IsOverBudget` check in `cluster_event.go` is removed.

### G) Task Breakdown

#### Task 1: Add `EnqueueOutcome` type and fix criticality check in `Enqueue` (BC-1, BC-2, BC-3, BC-4, BC-8)

**Files:** modify `sim/cluster/gateway_queue.go`, test `sim/cluster/gateway_queue_test.go`

**Step 1 — Write failing tests:**

Add to `gateway_queue_test.go`:

```go
func TestGatewayQueue_CriticalityProtection_NonSheddableNeverEvicted(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	r1 := &sim.Request{ID: "r1", SLOClass: "standard"}  // priority=3
	r2 := &sim.Request{ID: "r2", SLOClass: "critical"}   // priority=4

	outcome1, victim1 := q.Enqueue(r1, 1)
	if outcome1 != Enqueued {
		t.Errorf("r1 should be enqueued, got %v", outcome1)
	}
	if victim1 != nil {
		t.Error("no victim expected for r1")
	}

	outcome2, victim2 := q.Enqueue(r2, 2)
	if outcome2 != Enqueued {
		t.Errorf("r2 should be enqueued, got %v", outcome2)
	}
	if victim2 != nil {
		t.Error("no victim expected for r2")
	}

	// Queue full with only non-sheddable entries. New request should be rejected.
	r3 := &sim.Request{ID: "r3", SLOClass: "critical"}
	outcome3, victim3 := q.Enqueue(r3, 3)
	if outcome3 != Rejected {
		t.Errorf("r3 should be rejected — no sheddable victim available, got %v", outcome3)
	}
	if victim3 != nil {
		t.Error("no victim expected when rejected")
	}

	// Queue depth unchanged
	if q.Len() != 2 {
		t.Errorf("queue should still have 2 entries, got %d", q.Len())
	}
	if q.RejectedCount() != 1 {
		t.Errorf("expected 1 rejection, got %d", q.RejectedCount())
	}
}

func TestGatewayQueue_CriticalityProtection_SheddableEvicted(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	r1 := &sim.Request{ID: "r1", SLOClass: "batch"}      // priority=-1 (sheddable)
	r2 := &sim.Request{ID: "r2", SLOClass: "standard"}    // priority=3

	q.Enqueue(r1, 1)
	q.Enqueue(r2, 2)

	// Queue full. New standard request should evict the sheddable batch entry.
	r3 := &sim.Request{ID: "r3", SLOClass: "standard"}
	outcome, victim := q.Enqueue(r3, 3)
	if outcome != ShedVictim {
		t.Errorf("should shed a victim, got %v", outcome)
	}
	if victim == nil || victim.ID != "r1" {
		t.Errorf("batch request (r1) should be the victim, got %v", victim)
	}

	if q.Len() != 2 {
		t.Errorf("queue depth should be unchanged at 2, got %d", q.Len())
	}
	if q.ShedCount() != 1 {
		t.Errorf("expected 1 shed, got %d", q.ShedCount())
	}
}

func TestGatewayQueue_CriticalityProtection_SheddableIncomingRejectedWhenNoLowerVictim(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	r1 := &sim.Request{ID: "r1", SLOClass: "standard"} // priority=3
	r2 := &sim.Request{ID: "r2", SLOClass: "critical"}  // priority=4

	q.Enqueue(r1, 1)
	q.Enqueue(r2, 2)

	// Queue full with non-sheddable only. Even a sheddable incoming request is rejected.
	r3 := &sim.Request{ID: "r3", SLOClass: "batch"} // priority=-1
	outcome, victim := q.Enqueue(r3, 3)
	if outcome != Rejected {
		t.Errorf("batch request should be rejected — no sheddable victim in queue, got %v", outcome)
	}
	if victim != nil {
		t.Error("no victim expected")
	}
	if q.RejectedCount() != 1 {
		t.Errorf("expected 1 rejection, got %d", q.RejectedCount())
	}
}

func TestGatewayQueue_CriticalityProtection_LowerSheddableDoesNotEvictHigherSheddable(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	r1 := &sim.Request{ID: "r1", SLOClass: "batch"}      // priority=-1
	r2 := &sim.Request{ID: "r2", SLOClass: "standard"}    // priority=3

	q.Enqueue(r1, 1)
	q.Enqueue(r2, 2)

	// Queue full. Incoming background (-3) is lower than queued batch (-1). Should be rejected.
	r3 := &sim.Request{ID: "r3", SLOClass: "background"} // priority=-3
	outcome, victim := q.Enqueue(r3, 3)
	if outcome != Rejected {
		t.Errorf("lower-priority sheddable should be rejected, not evict higher sheddable, got %v", outcome)
	}
	if victim != nil {
		t.Error("no victim expected")
	}
	if q.RejectedCount() != 1 {
		t.Errorf("expected 1 rejection, got %d", q.RejectedCount())
	}

	// Verify the batch entry is still in the queue
	if q.Len() != 2 {
		t.Errorf("queue should still have 2 entries, got %d", q.Len())
	}
}
```

**Step 2 — Run test to verify failure:**
```bash
cd /Users/toslali/.../inference-sim/.worktrees/pr-phase1-parity-fixes
go test ./sim/cluster/ -run TestGatewayQueue_CriticalityProtection -v
# Expected: compilation error (EnqueueOutcome, ShedVictim, Rejected, RejectedCount not defined)
```

**Step 3 — Implement:**

In `gateway_queue.go`, add `EnqueueOutcome` type and fix `Enqueue`:

```go
// EnqueueOutcome represents the result of enqueuing a request.
type EnqueueOutcome int

const (
	Enqueued   EnqueueOutcome = iota // request accepted into queue
	ShedVictim                       // request accepted, a sheddable victim was evicted
	Rejected                         // queue full, no sheddable victim — incoming request not enqueued
)
```

Replace the `Enqueue` method:

```go
// Enqueue adds a request to the gateway queue.
// When the queue is at capacity, only sheddable (priority < 0) entries are eviction candidates.
// If no sheddable candidate exists, the incoming request is rejected.
// Returns the outcome and the evicted victim (non-nil only for ShedVictim).
func (q *GatewayQueue) Enqueue(req *sim.Request, seqID int64) (EnqueueOutcome, *sim.Request) {
	priority := q.priorityMap.Priority(req.SLOClass)
	entry := gatewayQueueEntry{request: req, priority: priority, seqID: seqID}

	if q.maxDepth > 0 && q.heap.Len() >= q.maxDepth {
		// Find the lowest-priority sheddable entry (priority < 0 only).
		minIdx := -1
		for i := 0; i < q.heap.Len(); i++ {
			if q.heap.entries[i].priority >= 0 {
				continue // non-sheddable — skip
			}
			if minIdx == -1 ||
				q.heap.entries[i].priority < q.heap.entries[minIdx].priority ||
				(q.heap.entries[i].priority == q.heap.entries[minIdx].priority &&
					q.heap.entries[i].seqID > q.heap.entries[minIdx].seqID) {
				minIdx = i
			}
		}

		if minIdx == -1 {
			// No sheddable candidate — reject the incoming request.
			q.rejectedCount++
			return Rejected, nil
		}

		// Only displace if incoming has higher priority (or same priority + earlier arrival).
		// Otherwise reject the incoming request — don't evict a higher-priority victim.
		minEntry := q.heap.entries[minIdx]
		if priority > minEntry.priority || (priority == minEntry.priority && seqID < minEntry.seqID) {
			victim := minEntry.request
			heap.Remove(&q.heap, minIdx)
			q.shedCount++
			heap.Push(&q.heap, entry)
			return ShedVictim, victim
		}

		// Incoming has lower/equal priority — reject it.
		q.rejectedCount++
		return Rejected, nil
	}

	heap.Push(&q.heap, entry)
	return Enqueued, nil
}
```

Add `rejectedCount` field and accessor:

```go
// In GatewayQueue struct, add:
rejectedCount int // number of requests rejected (queue full, no sheddable victim)

// Add method:
func (q *GatewayQueue) RejectedCount() int {
	return q.rejectedCount
}
```

**Step 4 — Run test to verify pass:**
```bash
go test ./sim/cluster/ -run TestGatewayQueue_CriticalityProtection -v
# Expected: all 3 tests pass
```

**Step 5 — Fix existing tests:**

Update `TestGatewayQueue_CapacityShed_LowestPriority` to use new return type and updated semantics:

```go
func TestGatewayQueue_CapacityShed_LowestPriority(t *testing.T) {
	q := NewGatewayQueue("priority", 2, nil) // max 2
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "critical"}, 2)

	// Queue full with non-sheddable only. Sheddable request → rejected (no sheddable victim in queue).
	outcome3, victim3 := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "sheddable"}, 3)
	if outcome3 != Rejected {
		t.Errorf("expected Rejected for sheddable at full non-sheddable queue, got %v", outcome3)
	}
	if victim3 != nil {
		t.Error("expected nil victim for Rejected outcome")
	}
	if q.RejectedCount() != 1 {
		t.Errorf("rejected count should be 1, got %d", q.RejectedCount())
	}

	// Another critical → also rejected (non-sheddable cannot displace non-sheddable).
	outcome4, victim4 := q.Enqueue(&sim.Request{ID: "r4", SLOClass: "critical"}, 4)
	if outcome4 != Rejected {
		t.Errorf("expected Rejected — non-sheddable cannot displace non-sheddable, got %v", outcome4)
	}
	if victim4 != nil {
		t.Error("expected nil victim")
	}
	if q.RejectedCount() != 2 {
		t.Errorf("rejected count should be 2, got %d", q.RejectedCount())
	}

	// Dequeue: r2 (critical, higher priority) then r1 (standard) — original entries unchanged.
	got1 := q.Dequeue()
	got2 := q.Dequeue()
	if got1.ID != "r2" || got2.ID != "r1" {
		t.Errorf("got %s, %s; want r2, r1", got1.ID, got2.ID)
	}
}
```

Also update `TestGatewayQueue_FIFO_DequeueOrder`, `TestGatewayQueue_Priority_DequeueOrder`, and `TestGatewayQueue_Priority_SamePriority_FIFO` — their `q.Enqueue(...)` calls need to accept the new `(EnqueueOutcome, *sim.Request)` return type. Since these tests don't check the return value (unlimited queue), assign to `_, _`:

```go
// In each test, change:
//   q.Enqueue(&sim.Request{...}, N)
// To:
//   _, _ = q.Enqueue(&sim.Request{...}, N)
```

```bash
go test ./sim/cluster/ -run TestGatewayQueue -v
# Expected: all gateway queue tests pass
```

**Step 6 — Lint:**
```bash
golangci-lint run ./sim/cluster/...
```

**Commit:** `fix(gateway-queue): criticality-based shedding — only evict priority < 0 (BC-1, BC-2, BC-3, BC-4, BC-8)`

---

#### Task 2: Add `GatewayQueueRejected` metric and update INV-1 (BC-10)

**Files:** modify `sim/cluster/metrics.go`, modify `sim/cluster/cluster.go`, modify `cmd/root.go`, test `sim/cluster/cluster_test.go`

**Accounting note:** `ShedVictim` is covered by the existing `gw_shed` bucket (incremented by `q.shedCount`). `Rejected` is covered by the new `gw_rejected` bucket (incremented by `q.rejectedCount`). Neither flows through `cs.rejectedRequests` — they are separate INV-1 buckets within `injected_requests`, not part of the admission-rejection pipeline count.

**Step 1 — Write failing test:**

Add to `cluster_test.go` (find the existing `TestClusterSimulator_FlowControl_Conservation` test):

Update the INV-1 conservation formula to include `gwRejected`:

```go
// In the conservation assertion, add gwRejected:
gwRejected := cs.GatewayQueueRejected()
accounted := completed + queued + running + dropped + timedout + routingRejections + gwDepth + gwShed + gwRejected
```

**Step 2 — Run test to verify failure:**
```bash
go test ./sim/cluster/ -run TestClusterSimulator_FlowControl_Conservation -v
# Expected: compilation error (GatewayQueueRejected not defined)
```

**Step 3 — Implement:**

In `metrics.go`, add field to `RawMetrics` struct (after `GatewayQueueShed` at line 107):
```go
GatewayQueueRejected int // Requests rejected from gateway queue (no sheddable victim)
```

Also update the INV-1 comment on line 105:
```go
// INV-1 extended: injected == completed + running + queued + shed + dropped + timed_out + gw_depth + gw_shed + gw_rejected
```

In `cluster.go`, add accessor:
```go
func (c *ClusterSimulator) GatewayQueueRejected() int {
	if c.gatewayQueue == nil {
		return 0
	}
	return c.gatewayQueue.RejectedCount()
}
```

Wire into metrics collection (find `collectMetrics` or equivalent, near where `GatewayQueueShed` is collected).

In `cluster_event.go`, update the `Enqueue` call site to handle the new return type. Note that `e.request.GatewayEnqueueTime = cs.clock` is already set on the line before the Enqueue call (existing code). The full context around the Enqueue call:

```go
// Replace the existing block:
//   shed := cs.gatewayQueue.Enqueue(e.request, cs.nextSeqID())
//   if shed { ... }
// With (keeping the existing GatewayEnqueueTime = cs.clock line above):
outcome, victim := cs.gatewayQueue.Enqueue(e.request, cs.nextSeqID())
switch outcome {
case Rejected:
	e.request.GatewayEnqueueTime = 0 // not enqueued — clear timestamp
	// INV-1 accounting: rejected requests flow into gw_rejected bucket via
	// gatewayQueue.rejectedCount → cs.GatewayQueueRejected() → rawMetrics.GatewayQueueRejected.
	// They are NOT added to cs.rejectedRequests (that counter is for admission rejections only).
	return
case ShedVictim:
	if victim != nil {
		victim.GatewayEnqueueTime = 0 // evicted — clear stale timestamp
	}
	// INV-1 accounting: victim flows into gw_shed bucket via
	// gatewayQueue.shedCount → cs.GatewayQueueShed() → rawMetrics.GatewayQueueShed.
	break
case Enqueued:
	break
}
cs.tryDispatchFromGatewayQueue()
```

In `cmd/root.go`, add `GatewayQueueRejected` wiring parallel to existing `GatewayQueueShed`:

```go
// After line 1604 (rawMetrics.GatewayQueueShed = ...):
rawMetrics.GatewayQueueRejected = cs.GatewayQueueRejected()

// In the anomaly condition (line 1634), add:
// || rawMetrics.GatewayQueueRejected > 0

// In the print block (after GatewayQueueShed print), add:
if rawMetrics.GatewayQueueRejected > 0 {
    fmt.Printf("Gateway Queue Rejected: %d\n", rawMetrics.GatewayQueueRejected)
}
```

**Step 4 — Run test:**
```bash
go test ./sim/cluster/ -run TestClusterSimulator_FlowControl -v
```

**Step 5 — Lint:**
```bash
golangci-lint run ./sim/cluster/...
```

**Commit:** `feat(metrics): add GatewayQueueRejected counter, update INV-1 conservation (BC-10)`

---

#### Task 3: Create `TenantBudgetAdmission` policy (BC-5, BC-6, BC-7)

**Files:** modify `sim/admission.go`, test `sim/admission_test.go`

**Step 1 — Write failing tests:**

```go
// stubTracker is a test double implementing TenantBudgetTracker.
// Avoids importing sim/cluster/ from sim/ test files.
type stubTracker struct{ overBudget bool }
func (s *stubTracker) IsOverBudget(string) bool { return s.overBudget }

func TestTenantBudgetAdmission_DelegatesToInnerPolicy(t *testing.T) {
	inner := &RejectAll{}
	tracker := &stubTracker{overBudget: false}
	pm := DefaultSLOPriorityMap()
	policy := NewTenantBudgetAdmission(inner, tracker, pm)

	req := &Request{ID: "r1", TenantID: "t1", SLOClass: "standard"}
	admitted, reason := policy.Admit(req, &RouterState{})
	// Inner policy rejects — budget check never runs
	if admitted {
		t.Error("inner policy rejection should stand")
	}
	if reason == "" {
		t.Error("should have rejection reason from inner policy")
	}
}

func TestTenantBudgetAdmission_SheddableOverBudgetRejected(t *testing.T) {
	inner := &AlwaysAdmit{}
	tracker := &stubTracker{overBudget: true}
	pm := DefaultSLOPriorityMap()
	policy := NewTenantBudgetAdmission(inner, tracker, pm)

	req := &Request{ID: "r1", TenantID: "t1", SLOClass: "batch"} // priority=-1, sheddable
	admitted, reason := policy.Admit(req, &RouterState{})
	if admitted {
		t.Error("sheddable over-budget request should be rejected")
	}
	if reason != "tenant-budget-shed" {
		t.Errorf("expected reason 'tenant-budget-shed', got %q", reason)
	}
}

func TestTenantBudgetAdmission_NonSheddableOverBudgetAdmitted(t *testing.T) {
	inner := &AlwaysAdmit{}
	tracker := &stubTracker{overBudget: true}
	pm := DefaultSLOPriorityMap()
	policy := NewTenantBudgetAdmission(inner, tracker, pm)

	req := &Request{ID: "r1", TenantID: "t1", SLOClass: "standard"} // priority=3, non-sheddable
	admitted, _ := policy.Admit(req, &RouterState{})
	if !admitted {
		t.Error("non-sheddable request should be admitted even when over budget")
	}
}
```

**Step 2 — Run test to verify failure:**
```bash
go test ./sim/... -run TestTenantBudgetAdmission -v
# Expected: compilation error (NewTenantBudgetAdmission not defined)
```

**Step 3 — Implement:**

In `sim/admission.go`:

```go
// TenantBudgetAdmission wraps an inner AdmissionPolicy and applies per-tenant
// budget enforcement after the inner policy admits the request.
// Only sheddable requests (priority < 0) are rejected when over budget.
// Non-sheddable requests always pass the budget check.
type TenantBudgetAdmission struct {
	inner       AdmissionPolicy
	tracker     TenantBudgetTracker
	priorityMap *SLOPriorityMap
}

// TenantBudgetTracker is the interface needed by TenantBudgetAdmission.
// Implemented by cluster.TenantTracker.
type TenantBudgetTracker interface {
	IsOverBudget(tenantID string) bool
}

func NewTenantBudgetAdmission(inner AdmissionPolicy, tracker TenantBudgetTracker, pm *SLOPriorityMap) *TenantBudgetAdmission {
	if inner == nil {
		panic("TenantBudgetAdmission: inner policy must not be nil")
	}
	if tracker == nil {
		panic("TenantBudgetAdmission: tracker must not be nil")
	}
	if pm == nil {
		pm = DefaultSLOPriorityMap()
	}
	return &TenantBudgetAdmission{inner: inner, tracker: tracker, priorityMap: pm}
}

func (t *TenantBudgetAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	admitted, reason := t.inner.Admit(req, state)
	if !admitted {
		return false, reason
	}
	if t.tracker.IsOverBudget(req.TenantID) && t.priorityMap.IsSheddable(req.SLOClass) {
		return false, "tenant-budget-shed"
	}
	return true, ""
}
```

Note: `TenantBudgetTracker` is a minimal interface so `sim/admission.go` doesn't import `sim/cluster/`. The existing `cluster.TenantTracker` already has `IsOverBudget(string) bool` and satisfies this interface.

**Step 4 — Run test:**
```bash
go test ./sim/... -run TestTenantBudgetAdmission -v
```

**Step 5 — Lint:**
```bash
golangci-lint run ./sim/...
```

**Commit:** `feat(admission): TenantBudgetAdmission decorator policy (BC-5, BC-6, BC-7)`

---

#### Task 4: Wire `TenantBudgetAdmission` in cluster, remove inline check (BC-9)

**Files:** modify `sim/cluster/cluster.go`, modify `sim/cluster/cluster_event.go`, test `sim/cluster/cluster_tenant_test.go`

**Step 1 — Write failing test:**

Update `TestTenantAdmission_INV1_BudgetShedConservation` in `cluster_tenant_test.go` to verify tenant budget rejection still works after the refactor. The test should pass with the same behavior — sheddable over-budget requests rejected, non-sheddable admitted.

**Step 2 — Run test to verify it passes before refactor** (baseline):
```bash
go test ./sim/cluster/ -run TestTenantAdmission -v
# Expected: passes (existing behavior)
```

**Step 3 — Implement refactor:**

In `cluster.go`:
1. Keep `tenantTracker *TenantTracker` field on `ClusterSimulator` (needed by 6+ `OnStart`/`OnComplete` call sites in `cluster_event.go`, `pd_events.go`, `cluster.go`)
2. When `config.TenantBudgets != nil`, wrap the admission policy with `TenantBudgetAdmission` using the same tracker instance:
```go
// In NewClusterSimulator, after creating tenantTracker (existing code at ~line 391):
if config.TenantBudgets != nil {
	totalCapacity := config.NumInstances * int(config.MaxRunningReqs)
	cs.tenantTracker = NewTenantTracker(config.TenantBudgets, totalCapacity)
	// Wrap admission policy with tenant budget decorator
	admissionPolicy = sim.NewTenantBudgetAdmission(admissionPolicy, cs.tenantTracker, cs.priorityMap)
	cs.admissionPolicy = admissionPolicy // reassign — struct was already constructed with unwrapped policy
}
```
3. All existing `cs.tenantTracker.OnStart()` and `cs.tenantTracker.OnComplete()` calls remain unchanged.

In `cluster_event.go`:
1. Remove the inline tenant budget check block (lines 160-181) — this is the `if cs.tenantTracker != nil && cs.tenantTracker.IsOverBudget(...)` block
2. The `AdmissionPolicy.Admit()` call now handles tenant budget via the `TenantBudgetAdmission` decorator
3. All other `cs.tenantTracker` references (OnStart/OnComplete) remain unchanged

**Step 4 — Run test:**
```bash
go test ./sim/cluster/ -run TestTenantAdmission -v
# Expected: all tenant tests pass with same behavior
```

Also run full test suite to catch any breakage:
```bash
go test ./...
```

**Step 5 — Lint:**
```bash
golangci-lint run ./sim/cluster/...
```

**Commit:** `refactor(cluster): wire TenantBudgetAdmission, remove inline tenant check (BC-9)`

---

#### Task 5: Update documentation (BC-10)

**Files:** modify `docs/contributing/standards/invariants.md`, modify `CLAUDE.md`

**Step 1 — Update INV-1 formula:**

In `docs/contributing/standards/invariants.md`, update the INV-1 formula to include `gateway_queue_rejected`:
```
injected_requests == completed_requests + still_queued + still_running + dropped_unservable +
    timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed + gateway_queue_rejected
```

In `CLAUDE.md`, update the INV-1 line (line ~164) with the same formula.

**Step 2 — Add recent changes entry to CLAUDE.md:**

Add under `## Recent Changes`:
```
- Criticality-based gateway queue shedding (#1190): `GatewayQueue.Enqueue()` now only evicts sheddable requests (`priority < 0`) on overflow. Non-sheddable requests (priority >= 0) are never evicted. When queue is full with no sheddable candidate, incoming request is rejected (`EnqueueOutcome.Rejected`). New `GatewayQueueRejected` counter added to INV-1. `TenantBudgetAdmission` policy wraps any inner `AdmissionPolicy` with per-tenant budget enforcement (decorator pattern). Inline tenant budget check removed from `cluster_event.go`. Closes #1020.
```

**Step 3 — Lint and verify:**
```bash
go test ./...
golangci-lint run ./...
```

**Commit:** `docs: update INV-1 formula with gw_rejected, add recent changes entry (BC-10)`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestGatewayQueue_CriticalityProtection_SheddableEvicted` |
| BC-2 | Task 1 | Unit | `TestGatewayQueue_CriticalityProtection_NonSheddableNeverEvicted`, `TestGatewayQueue_CriticalityProtection_LowerSheddableDoesNotEvictHigherSheddable` |
| BC-3 | Task 1 | Unit | `TestGatewayQueue_CriticalityProtection_NonSheddableNeverEvicted` |
| BC-4 | Task 1 | Unit | All three new tests verify outcome types |
| BC-5 | Task 3 | Unit | `TestTenantBudgetAdmission_DelegatesToInnerPolicy` |
| BC-6 | Task 3 | Unit | `TestTenantBudgetAdmission_NonSheddableOverBudgetAdmitted` |
| BC-7 | Task 3 | Unit | `TestTenantBudgetAdmission_SheddableOverBudgetRejected` |
| BC-8 | Task 1 | Unit | `TestGatewayQueue_CriticalityProtection_NonSheddableNeverEvicted` |
| BC-9 | Task 4 | Integration | `TestTenantAdmission_INV1_BudgetShedConservation` (existing, still passes) |
| BC-10 | Task 2 | Integration | `TestClusterSimulator_FlowControl_Conservation` (updated formula) |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Existing tests break due to new `Enqueue` return type | High | Low | Update all call sites (only 2: `cluster_event.go` and tests) | Task 1, 2 |
| `TenantTracker.OnStart`/`OnComplete` broken after refactor | Low | High | Field stays on ClusterSimulator — no changes to OnStart/OnComplete call sites | Task 4 |
| Import cycle: `sim/admission.go` importing `sim/cluster/` for `TenantTracker` | Medium | High | Use `TenantBudgetTracker` interface in `sim/` instead of concrete type; tests use stub | Task 3 |
| Conservation (INV-1) broken | Low | High | Explicit test in Task 2 | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `TenantBudgetTracker` interface is minimal (1 method), needed to avoid import cycle
- [x] No feature creep — no new CLI flags, no new metrics beyond `gw_rejected`
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates — `Enqueue` return type changes, all call sites updated; `tenantTracker` field kept (no breakage to 6+ OnStart/OnComplete sites)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md updated: INV-1 formula, recent changes entry
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: invariants.md canonical source updated, CLAUDE.md working copy updated
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered: Task 1 → Task 2 → Task 3 → Task 4 → Task 5
- [x] All contracts mapped to tasks
- [x] Construction site audit completed — `GatewayQueue` (1 site), `TenantTracker` (1 site), `ClusterSimulator` (1 site)

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — all paths have explicit outcomes or counter increments
- [x] R3: No new numeric parameters (no new CLI flags)
- [x] R4: Construction sites audited — `GatewayQueue` built in `cluster.go:403`, `TenantTracker` in `cluster.go:391`
- [x] R6: No `logrus.Fatalf` in `sim/` — using `panic` for constructor validation (existing pattern)
- [x] R7: INV-1 conservation test serves as invariant test
- [x] R8: No exported mutable maps
- [x] R11: No division in new code
- [x] R13: `TenantBudgetTracker` interface has 1 method, used by 1 implementation — acceptable because it's needed to avoid import cycle, not for polymorphism
- [x] R14: No method spans multiple concerns
- [x] R21: No range over shrinking slices

