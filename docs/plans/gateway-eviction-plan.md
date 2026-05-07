# Gateway In-Flight Eviction Implementation Plan

**Goal:** When flow control is enabled and the system is saturated, evict lowest-priority sheddable requests that are already running on instances, freeing capacity for higher-priority waiting requests — matching GIE's eviction architecture.

**The problem today:** Once a request is dispatched from the gateway queue to an instance, it cannot be recalled. If the system saturates while low-priority (sheddable) requests are running, higher-priority requests must wait even though evicting sheddable in-flight requests would free capacity immediately.

**What this PR adds:**

1. An eviction tracking heap that monitors dispatched sheddable requests, ordered by (priority ASC, dispatch_time DESC)
2. An eviction trigger in the dispatch cycle: when saturation >= 1.0 and higher-priority requests are waiting, evict N sheddable in-flight requests
3. A `GatewayEvictionEvent` that terminates evicted requests on their instance, frees KV blocks, and counts them as `gw_evicted`
4. INV-1 extension with the `gw_evicted` counter in metrics and output

**Why this matters:** This completes Phase 4 of GIE flow control parity (#899), bringing BLIS's gateway-level behavior in line with llm-d's eviction system where sheddable in-flight requests can be aborted to make room for higher-priority traffic.

**Architecture:** New `EvictionTracker` data structure in `sim/cluster/` (min-heap). New `GatewayEvictionEvent` in `sim/cluster/cluster_event.go`. Wired into the existing `tryDispatchFromGatewayQueue` path: before dispatching, if saturated and a non-sheddable request is waiting, evict one sheddable in-flight request. Eviction terminates the request on the instance (removes from running batch, frees KV), records it as a new terminal state.

**Source:** [Issue #1228](https://github.com/inference-sim/inference-sim/issues/1228), part of [#899](https://github.com/inference-sim/inference-sim/issues/899) Phase 4.

**Closes:** `Fixes #1228`

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** Gateway-level eviction — a new flow control capability in `sim/cluster/`.
2. **Adjacent blocks:** `FlowControlAdmission` (owns the gateway queue), `GatewayQueue` (tracks queued requests), `ClusterSimulator` (owns the event loop and dispatch), `SaturationDetector` (determines saturation level), `InstanceSimulator` (where evicted requests run).
3. **Invariants touched:** INV-1 (request conservation — new `gw_evicted` bucket), INV-5 (causality — eviction time must be >= dispatch time).
4. **Construction site audit:**
   - `RawMetrics` struct in `sim/cluster/metrics.go` — one construction site in `CollectRawMetrics()`. Adding `GatewayEvicted int` field.
   - `ProgressSnapshot` in `sim/progress_hook.go` — one construction site in `ClusterSimulator.emitProgress()`. Adding `GatewayEvicted int` field.
   - `ClusterSimulator` in `sim/cluster/cluster.go` — one construction site in `NewClusterSimulator()`. Adding `evictionTracker *EvictionTracker` field.

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds gateway-level in-flight eviction for sheddable requests when the system is saturated. When flow control is enabled and a higher-priority request is blocked from dispatch because saturation >= 1.0, the eviction trigger pops the most-evictable sheddable in-flight request (lowest priority, newest dispatch time) and terminates it on its instance. This frees one slot of capacity, allowing the higher-priority request to dispatch.

The design is minimal: a tracking heap, a trigger wired into the dispatch path, and an event that terminates the request. No new CLI flags beyond what `--flow-control` already provides — eviction is always active when flow control is enabled (matching GIE where eviction is a core part of the flow controller, not an optional add-on).

### B) Behavioral Contracts

**Positive contracts (what MUST happen):**

```
BC-1: Eviction of sheddable in-flight requests when saturated
- GIVEN flow control enabled AND saturation >= 1.0
- AND a sheddable request (priority < 0) is dispatched and running on an instance
- AND a non-sheddable request (priority >= 0) is waiting in the gateway queue
- WHEN the dispatch cycle runs (tryDispatchFromGatewayQueue)
- THEN the most-evictable sheddable in-flight request is terminated, its KV freed, gw_evicted incremented
```

```
BC-2: Eviction ordering — lowest priority first, newest dispatch time first
- GIVEN multiple sheddable requests are in-flight with different priorities or dispatch times
- WHEN eviction selects a victim
- THEN it picks the request with the lowest priority; among equal priority, the one with the newest (latest) dispatch time
```

```
BC-3: Tracking starts at routing (not gateway dispatch), ends at completion
- GIVEN flow control enabled and a sheddable request is routed to an instance
- WHEN RoutingDecisionEvent executes (request injected into instance, AssignedInstance set)
- THEN it is added to the eviction tracker with its target instance ID
- AND when it completes normally, it is removed from the eviction tracker
- NOTE: tracking starts at routing, not gateway dispatch, because between dispatch
  and routing the request is in the event pipeline (not yet on any instance)
```

```
BC-4: INV-1 conservation with gw_evicted
- GIVEN any simulation run with flow control and eviction
- WHEN the simulation ends
- THEN injected == completed + running + queued + routing_rejections + dropped + timed_out + gw_depth + gw_shed + gw_rejected + gw_evicted
```

```
BC-5: Evicted request is terminated (not requeued)
- GIVEN a request is selected for gateway eviction
- WHEN the GatewayEvictionEvent executes
- THEN the request is removed from the instance's running batch and wait queue, its KV blocks freed, and it is counted as gw_evicted (terminal state)
```

**Negative contracts (what MUST NOT happen):**

```
BC-6: Non-sheddable requests never evicted
- GIVEN flow control enabled and saturation >= 1.0
- AND only non-sheddable requests (priority >= 0) are in-flight
- WHEN the eviction trigger fires
- THEN no eviction occurs — non-sheddable requests are protected
```

```
BC-7: No eviction when flow control disabled
- GIVEN flow control is NOT enabled (default mode)
- WHEN any request is dispatched or completed
- THEN no eviction tracking or eviction occurs
```

```
BC-8: No eviction when not saturated
- GIVEN flow control enabled AND saturation < 1.0
- WHEN tryDispatchFromGatewayQueue runs
- THEN no eviction occurs (normal dispatch proceeds instead)
```

### C) Component Interaction

```
                   ┌─────────────────────────────────────────────┐
                   │           ClusterSimulator                   │
                   │                                             │
                   │  ┌────────────────────┐                    │
                   │  │ FlowControlAdmission│◀── Admit()         │
                   │  │  └─ GatewayQueue    │                    │
                   │  └────────────────────┘                    │
                   │            │                                │
                   │     Enqueue/Dequeue                         │
                   │            ▼                                │
                   │  ┌────────────────────┐                    │
                   │  │tryDispatchFromGW()  │ ◀── on completion  │
                   │  │  1. Check saturation│       or enqueue   │
                   │  │  2. IF sat>=1.0 AND │                    │
                   │  │     waiting non-shed:│                    │
                   │  │     → tryEvictOne() │                    │
                   │  │  3. DequeueGated()  │                    │
                   │  └────────┬───────────┘                    │
                   │           │                                │
                   │     dispatch / evict                        │
                   │           ▼                                │
                   │  ┌────────────────────┐   ┌─────────────┐ │
                   │  │  EvictionTracker    │   │ Instance[i] │ │
                   │  │  (min-heap of       │   │  RunningBatch│
                   │  │   routed sheddable  │──▶│  WaitQ       │
                   │  │   reqs + instanceID)│   │  KVStore     │
                   │  └────────────────────┘   └─────────────┘ │
                   └─────────────────────────────────────────────┘

Data flow:
- On routing (executeStandardRouting): tracker.Track(req, instanceID) — adds sheddable req to heap
- On terminal state (OnRequestDone callback): tracker.Untrack(req.ID) — removes from heap
- On eviction trigger: tracker.Pop() → returns (req, instanceID) → schedule GatewayEvictionEvent
- GatewayEvictionEvent: removes req from instance, frees KV, increments gw_evicted

KEY DESIGN DECISIONS:
1. Tracking starts at ROUTING (not gateway dispatch) because between gateway dispatch
   and routing, the request is in the event pipeline — it hasn't reached any instance yet.
2. Untracking uses the OnRequestDone callback (fires on completion, timeout, drop) so all
   terminal paths are covered without scattering Untrack calls.
3. EvictionTracker stores the target instance ID alongside each request so Pop() returns
   (req, instanceID) directly — no O(n) scan needed.
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue says "Eviction trigger — when FlowControlAdmission detects saturation" | Trigger is in `tryDispatchFromGatewayQueue`, not in FlowControlAdmission.Admit() | CLARIFICATION — dispatch is the right place because saturation is already computed there; GIE also triggers from the dispatch/processor cycle, not from admission |
| Issue says "~150-200 lines" | Actual estimate is ~200-250 lines (including tests) | CLARIFICATION — tests add volume but the production code is ~150 lines |
| GIE tracks ALL dispatched requests in `allInFlight` map, plus a filtered heap for evictable ones | BLIS only tracks sheddable requests in the heap (no separate allInFlight map) | SIMPLIFICATION — GIE needs the allInFlight map for its thread-safe removal by requestID on completion; BLIS is single-threaded DES so the heap with a map[requestID]→heapIndex suffices |
| Issue says "may need flag to enable/disable in-flight eviction" | No new flag — eviction is always active when `--flow-control` is active | SIMPLIFICATION — matches GIE where eviction is integral to flow control, not optional. If needed later, a disable flag is a one-line addition. |

### E) Review Guide

**Tricky part:** The eviction trigger placement in `tryDispatchFromGatewayQueue`. The trigger must fire BEFORE the dispatch attempt, because eviction is meant to free capacity so dispatch can succeed. The condition is: `saturation >= 1.0 AND gateway queue has a non-sheddable head-of-line request AND eviction tracker has evictable entries`.

**Scrutinize:** INV-1 conservation — every evicted request must be counted exactly once in `gw_evicted`. No double-counting if the request was also in the gateway queue (it shouldn't be — tracking starts AFTER dispatch from queue).

**Safe to skim:** The heap implementation is standard Go `container/heap` min-heap pattern.

**Known debt:** GIE has pluggable ordering and filter policies (interfaces). BLIS hardcodes the single GIE-parity ordering (priority ASC, dispatch_time DESC) and filter (sheddable only). This is fine — BLIS is the only consumer and can add interfaces if a second policy appears.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| Action | File | Purpose |
|--------|------|---------|
| Create | `sim/cluster/eviction_tracker.go` | EvictionTracker min-heap + Track/Untrack/Pop/Len |
| Create | `sim/cluster/eviction_tracker_test.go` | Unit tests for tracker |
| Modify | `sim/cluster/cluster_event.go` | Add `GatewayEvictionEvent` |
| Modify | `sim/cluster/cluster.go` | Add `evictionTracker` field, wire Track on dispatch, Untrack on completion, tryEvictOne trigger |
| Modify | `sim/cluster/metrics.go` | Add `GatewayEvicted` to `RawMetrics` |
| Modify | `sim/progress_hook.go` | Add `GatewayEvicted` to `ProgressSnapshot` |
| Modify | `cmd/root.go` | Populate `GatewayEvicted` from cluster, display in output |
| Modify | `cmd/replay.go` | Same as root.go for replay command |
| Modify | `docs/guide/admission.md` | Document eviction behavior in Flow Control section |

No dead code. No unused interfaces. No feature flags.

### G) Task Breakdown

#### Task 1: EvictionTracker data structure (BC-2, BC-3, BC-6)

**Files:** create `sim/cluster/eviction_tracker.go`, create `sim/cluster/eviction_tracker_test.go`

**Test (write first, must fail):**

```go
// sim/cluster/eviction_tracker_test.go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestEvictionTracker_TrackAndPop(t *testing.T) {
	tracker := NewEvictionTracker()

	// Track 3 sheddable requests with different priorities and dispatch times
	r1 := &sim.Request{ID: "r1", SLOClass: "sheddable", GatewayDispatchTime: 100}
	r2 := &sim.Request{ID: "r2", SLOClass: "background", GatewayDispatchTime: 200}
	r3 := &sim.Request{ID: "r3", SLOClass: "batch", GatewayDispatchTime: 150}

	pm := sim.DefaultSLOPriorityMap()
	tracker.Track(r1, pm) // priority -2
	tracker.Track(r2, pm) // priority -3 (lowest — evict first)
	tracker.Track(r3, pm) // priority -1

	if tracker.Len() != 3 {
		t.Fatalf("expected Len()=3, got %d", tracker.Len())
	}

	// Pop should return lowest priority first (background=-3)
	victim := tracker.Pop()
	if victim == nil || victim.ID != "r2" {
		t.Fatalf("expected r2 (background, priority -3), got %v", victim)
	}

	// Next: sheddable=-2
	victim = tracker.Pop()
	if victim == nil || victim.ID != "r1" {
		t.Fatalf("expected r1 (sheddable, priority -2), got %v", victim)
	}

	if tracker.Len() != 1 {
		t.Fatalf("expected Len()=1, got %d", tracker.Len())
	}
}

func TestEvictionTracker_SamePriority_NewestDispatchFirst(t *testing.T) {
	tracker := NewEvictionTracker()
	pm := sim.DefaultSLOPriorityMap()

	// Two sheddable requests (both priority -2), different dispatch times
	r1 := &sim.Request{ID: "r1", SLOClass: "sheddable", GatewayDispatchTime: 100}
	r2 := &sim.Request{ID: "r2", SLOClass: "sheddable", GatewayDispatchTime: 200}

	tracker.Track(r1, pm)
	tracker.Track(r2, pm)

	// Same priority → newest dispatch time evicted first (r2 dispatched later)
	victim := tracker.Pop()
	if victim == nil || victim.ID != "r2" {
		t.Fatalf("expected r2 (newest dispatch time 200), got %v", victim)
	}
}

func TestEvictionTracker_Untrack(t *testing.T) {
	tracker := NewEvictionTracker()
	pm := sim.DefaultSLOPriorityMap()

	r1 := &sim.Request{ID: "r1", SLOClass: "sheddable", GatewayDispatchTime: 100}
	r2 := &sim.Request{ID: "r2", SLOClass: "sheddable", GatewayDispatchTime: 200}

	tracker.Track(r1, pm)
	tracker.Track(r2, pm)

	// Untrack r2 (completed normally)
	tracker.Untrack("r2")

	if tracker.Len() != 1 {
		t.Fatalf("expected Len()=1 after untrack, got %d", tracker.Len())
	}

	victim := tracker.Pop()
	if victim == nil || victim.ID != "r1" {
		t.Fatalf("expected r1 after r2 untracked, got %v", victim)
	}
}

func TestEvictionTracker_NonSheddableNotTracked(t *testing.T) {
	tracker := NewEvictionTracker()
	pm := sim.DefaultSLOPriorityMap()

	// Non-sheddable requests should not be tracked
	r1 := &sim.Request{ID: "r1", SLOClass: "critical", GatewayDispatchTime: 100}
	r2 := &sim.Request{ID: "r2", SLOClass: "standard", GatewayDispatchTime: 200}

	tracker.Track(r1, pm)
	tracker.Track(r2, pm)

	if tracker.Len() != 0 {
		t.Fatalf("expected Len()=0 for non-sheddable requests, got %d", tracker.Len())
	}
}

func TestEvictionTracker_PopEmpty(t *testing.T) {
	tracker := NewEvictionTracker()
	victim := tracker.Pop()
	if victim != nil {
		t.Fatalf("expected nil from empty tracker, got %v", victim)
	}
}
```

**Implementation:**

```go
// sim/cluster/eviction_tracker.go
package cluster

import (
	"container/heap"

	"github.com/inference-sim/inference-sim/sim"
)

// evictionEntry represents a routed sheddable request eligible for eviction.
type evictionEntry struct {
	req          *sim.Request
	priority     int    // from SLOPriorityMap (negative for sheddable)
	dispatchTime int64  // GatewayDispatchTime
	instanceID   string // target instance (set at routing time)
	index        int    // heap index
}

// evictionHeap implements heap.Interface for eviction ordering:
// lowest priority first, newest dispatch time first (tie-breaker).
type evictionHeap []*evictionEntry

func (h evictionHeap) Len() int { return len(h) }

func (h evictionHeap) Less(i, j int) bool {
	if h[i].priority != h[j].priority {
		return h[i].priority < h[j].priority
	}
	// Same priority: newest dispatch time evicted first (higher time = more evictable)
	return h[i].dispatchTime > h[j].dispatchTime
}

func (h evictionHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *evictionHeap) Push(x interface{}) {
	entry := x.(*evictionEntry)
	entry.index = len(*h)
	*h = append(*h, entry)
}

func (h *evictionHeap) Pop() interface{} {
	old := *h
	n := len(old)
	entry := old[n-1]
	old[n-1] = nil
	entry.index = -1
	*h = old[:n-1]
	return entry
}

// EvictionTracker tracks dispatched sheddable requests for in-flight eviction.
// Only sheddable requests (priority < 0) are tracked.
// Single-threaded (DES event loop only) — no synchronization needed.
type EvictionTracker struct {
	h       evictionHeap
	byID    map[string]*evictionEntry // requestID → entry for O(log n) removal
}

// NewEvictionTracker creates an empty eviction tracker.
func NewEvictionTracker() *EvictionTracker {
	return &EvictionTracker{
		byID: make(map[string]*evictionEntry),
	}
}

// Track adds a routed request to the eviction tracker.
// Only sheddable requests (priority < 0) are tracked; non-sheddable requests are ignored.
// Called from RoutingDecisionEvent after the request is injected into its target instance.
func (et *EvictionTracker) Track(req *sim.Request, instanceID string, pm *sim.SLOPriorityMap) {
	if !pm.IsSheddable(req.SLOClass) {
		return
	}
	entry := &evictionEntry{
		req:          req,
		priority:     pm.Priority(req.SLOClass),
		dispatchTime: req.GatewayDispatchTime,
		instanceID:   instanceID,
	}
	heap.Push(&et.h, entry)
	et.byID[req.ID] = entry
}

// Untrack removes a request from the eviction tracker (e.g., on normal completion).
// No-op if the request is not tracked.
func (et *EvictionTracker) Untrack(requestID string) {
	entry, ok := et.byID[requestID]
	if !ok {
		return
	}
	heap.Remove(&et.h, entry.index)
	delete(et.byID, requestID)
}

// Pop removes and returns the most-evictable request and its instance ID.
// Returns (nil, "") if the tracker is empty.
func (et *EvictionTracker) Pop() (*sim.Request, string) {
	if et.h.Len() == 0 {
		return nil, ""
	}
	entry := heap.Pop(&et.h).(*evictionEntry)
	delete(et.byID, entry.req.ID)
	return entry.req, entry.instanceID
}

// Len returns the number of tracked evictable requests.
func (et *EvictionTracker) Len() int {
	return et.h.Len()
}
```

**Verify:** `go test ./sim/cluster/... -run TestEvictionTracker -v`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): add EvictionTracker min-heap for in-flight eviction (BC-2, BC-3, BC-6)`

---

#### Task 2: InstanceSimulator.EvictRequest method (BC-5)

**Files:** modify the instance simulator (the file containing `InstanceSimulator` — likely `sim/cluster/instance.go` or the inner `sim/simulator.go`)

**Test (write first, must fail):**

```go
func TestInstanceSimulator_EvictRequest_RemovesFromWaitQ(t *testing.T) {
	// Setup: create an instance, inject a request, verify it's in WaitQ
	// Then call EvictRequest — verify request removed, KV freed
	inst := newTestInstance(t, testInstanceConfig())
	req := &sim.Request{ID: "evict-me", InputTokens: 10, OutputTokens: 100}
	inst.InjectRequestOnline(req, 1000)

	// Request should be in WaitQ (not yet scheduled into running batch)
	if inst.QueueDepth() == 0 {
		t.Fatal("expected request in WaitQ")
	}

	freed := inst.EvictRequest(req)
	if !freed {
		t.Fatal("expected EvictRequest to return true")
	}
	if inst.QueueDepth() != 0 {
		t.Fatal("expected WaitQ empty after eviction")
	}
}

func TestInstanceSimulator_EvictRequest_NotFound(t *testing.T) {
	inst := newTestInstance(t, testInstanceConfig())
	req := &sim.Request{ID: "ghost"}
	freed := inst.EvictRequest(req)
	if freed {
		t.Fatal("expected EvictRequest to return false for unknown request")
	}
}
```

**Implementation:** Add `EvictRequest(*Request) bool` method to the instance simulator:
- Search WaitQ for the request by ID — if found, remove it and free any allocated KV blocks
- If not in WaitQ, search RunningBatch — if found, remove it and free KV blocks
- Return true if found and removed, false otherwise (idempotent/safe for already-completed requests)

**Verify:** `go test ./sim/... -run TestInstanceSimulator_EvictRequest -v`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `feat(sim): add EvictRequest for gateway-level eviction (BC-5)`

---

#### Task 3: GatewayEvictionEvent (BC-5)

**Files:** modify `sim/cluster/cluster_event.go`, modify `sim/cluster/cluster.go` (add `gatewayEvicted int` field)

**Test (write first, must fail):**

```go
func TestGatewayEvictionEvent_TerminatesRequest(t *testing.T) {
	// Setup: minimal cluster with one instance and flow control
	cfg := testClusterConfig(t, 1)
	cfg.FlowControlEnabled = true
	cs := newTestCluster(t, cfg)

	req := &sim.Request{
		ID:                  "evict-me",
		SLOClass:            "sheddable",
		InputTokens:         10,
		OutputTokens:        100,
		GatewayDispatchTime: 1000,
	}

	instID := string(cs.instances[0].ID())
	cs.instances[0].InjectRequestOnline(req, 1000)
	cs.inFlightRequests[instID]++

	event := &GatewayEvictionEvent{
		time:           2000,
		request:        req,
		targetInstance: instID,
	}
	event.Execute(cs)

	if cs.gatewayEvicted != 1 {
		t.Fatalf("expected gatewayEvicted=1, got %d", cs.gatewayEvicted)
	}
	if cs.inFlightRequests[instID] != 0 {
		t.Fatalf("expected inFlightRequests=0, got %d", cs.inFlightRequests[instID])
	}
}
```

**Implementation:** Add to `sim/cluster/cluster_event.go`:

```go
// GatewayEvictionEvent terminates a dispatched sheddable request on its instance.
// This is a gateway-level eviction (distinct from instance-level KV preemption).
// The request is terminated — not requeued. This is a terminal state.
// Priority 5: processed after routing decisions at the same timestamp.
type GatewayEvictionEvent struct {
	time           int64
	request        *sim.Request
	targetInstance string
}

func (e *GatewayEvictionEvent) Timestamp() int64 { return e.time }
func (e *GatewayEvictionEvent) Priority() int     { return 5 }

// Execute removes the evicted request from its target instance and frees resources.
func (e *GatewayEvictionEvent) Execute(cs *ClusterSimulator) {
	logrus.Debugf("[cluster] gateway eviction: req %s evicted from instance %s at tick %d",
		e.request.ID, e.targetInstance, e.time)

	// Remove request from instance (running batch or wait queue) and free KV.
	// EvictRequest returns false if request already completed (same-tick race) — in that
	// case, skip all accounting (request was already counted as completed, not evicted).
	var evicted bool
	for _, inst := range cs.instances {
		if string(inst.ID()) == e.targetInstance {
			evicted = inst.EvictRequest(e.request)
			break
		}
	}
	if !evicted {
		return
	}

	// Decrement in-flight count.
	cs.inFlightRequests[e.targetInstance]--
	if cs.inFlightRequests[e.targetInstance] < 0 {
		cs.inFlightRequests[e.targetInstance] = 0
	}

	// Track tenant completion for fair-share.
	if cs.tenantTracker != nil {
		cs.tenantTracker.OnComplete(e.request.TenantID)
	}

	// Increment gateway evicted counter (INV-1).
	cs.gatewayEvicted++

	// Re-trigger dispatch: freed capacity may allow the waiting request to dispatch now.
	cs.tryDispatchFromGatewayQueue()
}
```

Also add the `gatewayEvicted` field to `ClusterSimulator` and `EvictRequest` method to `InstanceSimulator`. The `EvictRequest` method needs to be added to the instance simulator — it removes the request from running batch + WaitQ and frees its KV blocks.

**Verify:** `go test ./sim/cluster/... -run TestGatewayEvictionEvent -v`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): add GatewayEvictionEvent for in-flight termination (BC-5)`

---

#### Task 4: Wire eviction tracking into routing and completion (BC-3, BC-7)

**Files:** modify `sim/cluster/cluster.go`, modify `sim/cluster/cluster_event.go`

**Test:**

```go
func TestEvictionTracking_OnRoutingAndCompletion(t *testing.T) {
	// Setup: cluster with flow control, inject sheddable and critical requests.
	// Verify: sheddable requests tracked after routing, untracked on completion.
	// Verify: critical requests NOT tracked.
	cfg := testClusterConfig(t, 1)
	cfg.FlowControlEnabled = true
	cs := newTestCluster(t, cfg)

	// Simulate: route a sheddable request → tracker.Len() == 1
	sheddableReq := &sim.Request{ID: "s1", SLOClass: "sheddable", InputTokens: 10, OutputTokens: 50}
	// ... (inject through routing event, check tracker has entry)

	// Simulate: route a critical request → tracker.Len() still 1
	criticalReq := &sim.Request{ID: "c1", SLOClass: "critical", InputTokens: 10, OutputTokens: 50}
	// ... (inject through routing event, check tracker unchanged)

	// Simulate: sheddable completes → tracker.Len() == 0
}
```

**Implementation:** In `ClusterSimulator`:

1. Add fields: `evictionTracker *EvictionTracker`, `priorityMap *sim.SLOPriorityMap`
2. In `NewClusterSimulator`: `if flowControlEnabled { c.evictionTracker = NewEvictionTracker() }`
3. In `RoutingDecisionEvent.Execute()`: after `inst.InjectRequestOnline(req, time)`, add:
   ```go
   if cs.evictionTracker != nil {
       cs.evictionTracker.Track(e.request, decision.TargetInstance, cs.priorityMap)
   }
   ```
4. In the completion handler (where `inFlightRequests` is decremented): add:
   ```go
   if c.evictionTracker != nil {
       c.evictionTracker.Untrack(req.ID)
   }
   ```
   This must be called for every terminal path: completion, timeout, drop.

**Verify:** `go test ./sim/cluster/... -run TestEvictionTracking -v`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): wire eviction tracking into routing/completion (BC-3, BC-7)`

---

#### Task 4: Eviction trigger in dispatch cycle (BC-1, BC-6, BC-8)

**Files:** modify `sim/cluster/cluster.go`

**Test:**

```go
func TestGatewayEviction_EndToEnd(t *testing.T) {
	// Full integration: flow control enabled, saturated system, sheddable running,
	// non-sheddable waiting → eviction occurs.
	//
	// Setup: 1 instance, max-concurrency=1, flow control enabled.
	// 1. Dispatch a sheddable request (fills capacity, saturation=1.0)
	// 2. Arrive a critical request (enters gateway queue, cannot dispatch — saturated)
	// 3. On next dispatch attempt, eviction trigger fires:
	//    - sheddable request evicted
	//    - critical request dispatched
	// Verify: gw_evicted=1, critical request completes, INV-1 holds.
}

func TestGatewayEviction_NoEvictionWhenNotSaturated(t *testing.T) {
	// Saturation < 1.0 → no eviction even if sheddable in-flight.
}

func TestGatewayEviction_NoEvictionWhenOnlyNonSheddable(t *testing.T) {
	// All in-flight are non-sheddable → no eviction even if saturated.
}
```

**Implementation:** In `tryDispatchFromGatewayQueue`, insert the eviction logic BEFORE the `DequeueGated` call:

```go
func (c *ClusterSimulator) tryDispatchFromGatewayQueue() bool {
	if c.gatewayQueue == nil || c.gatewayQueue.Len() == 0 {
		return false
	}
	state := buildRouterState(c, nil)
	sat := c.saturationDetector.Saturation(state)
	if math.IsNaN(sat) || math.IsInf(sat, 0) {
		panic(...)
	}

	// Eviction trigger: if saturated and evictable requests exist,
	// try to evict one to free capacity for the waiting request.
	if sat >= 1.0 && c.evictionTracker != nil && c.evictionTracker.Len() > 0 {
		// Only evict if a non-sheddable request is waiting (don't evict for sheddable-for-sheddable)
		if c.gatewayQueue.HasNonSheddableWaiting(c.priorityMap) {
			c.tryEvictOne()
		}
	}

	req := c.gatewayQueue.DequeueGated(sat)
	// ... (rest unchanged)
}

// tryEvictOne pops the most-evictable request and schedules its termination.
func (c *ClusterSimulator) tryEvictOne() {
	victim := c.evictionTracker.Pop()
	if victim == nil {
		return
	}
	// Find which instance the victim is running on
	targetInstance := c.findInstanceForRequest(victim.ID)
	if targetInstance == "" {
		return // request already completed between check and eviction (edge case)
	}
	heap.Push(&c.clusterEvents, clusterEventEntry{
		event: &GatewayEvictionEvent{
			time:           c.clock,
			request:        victim,
			targetInstance: targetInstance,
		},
		seqID: c.nextSeqID(),
	})
}
```

Also add `HasNonSheddableWaiting` method to `GatewayQueue` and `findInstanceForRequest` helper to `ClusterSimulator`.

**Verify:** `go test ./sim/cluster/... -run TestGatewayEviction -v`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `feat(cluster): eviction trigger in dispatch cycle (BC-1, BC-6, BC-8)`

---

#### Task 5: INV-1 extension — metrics and output (BC-4)

**Files:** modify `sim/cluster/metrics.go`, `sim/progress_hook.go`, `cmd/root.go`, `cmd/replay.go`

**Test:**

```go
func TestINV1_WithGatewayEviction(t *testing.T) {
	// Run a simulation that produces gateway evictions.
	// Verify: injected == completed + running + queued + rejected + dropped +
	//         timed_out + gw_depth + gw_shed + gw_rejected + gw_evicted
}
```

**Implementation:**

1. `sim/cluster/metrics.go`: Add `GatewayEvicted int` to `RawMetrics`
2. `sim/progress_hook.go`: Add `GatewayEvicted int` to `ProgressSnapshot`
3. `sim/cluster/cluster.go`: Add `GatewayEvicted()` accessor method, populate in `emitProgress()`
4. `cmd/root.go`: Set `rawMetrics.GatewayEvicted = cs.GatewayEvicted()`, display in anomaly output
5. `cmd/replay.go`: Same as root.go
6. Update INV-1 comment in `metrics.go`

**Verify:** `go test ./... -run TestINV1 -v`
**Lint:** `golangci-lint run ./...`
**Commit:** `feat(cluster): INV-1 extension with gw_evicted counter (BC-4)`

---

#### Task 6: InstanceSimulator.EvictRequest method (BC-5)

**Files:** modify `sim/simulator.go` (or wherever InstanceSimulator lives)

**Test:**

```go
func TestInstanceSimulator_EvictRequest(t *testing.T) {
	// Inject a request into an instance, then evict it.
	// Verify: request removed from running batch or wait queue, KV freed.
}
```

**Implementation:** Add `EvictRequest(*Request)` method to `InstanceSimulator` that:
1. Searches WaitQ for the request — if found, removes it
2. If not in WaitQ, searches RunningBatch — if found, removes it and frees KV blocks
3. Returns silently if request not found (may have completed between trigger and event execution)

**Verify:** `go test ./sim/... -run TestInstanceSimulator_EvictRequest -v`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `feat(sim): add EvictRequest for gateway-level eviction (BC-5)`

---

#### Task 7: Documentation (user-facing)

**Files:** modify `docs/guide/admission.md`, update CLAUDE.md if needed

**Implementation:** Add an "In-Flight Eviction" subsection to the Flow Control Admission section in `docs/guide/admission.md`:

```markdown
### In-Flight Eviction

When flow control is enabled and the system is saturated, BLIS can evict sheddable
requests that are already running on instances to free capacity for higher-priority
waiting requests. This matches GIE's eviction system.

**How it works:**

1. When saturation >= 1.0 and a non-sheddable request is waiting in the gateway queue
2. The eviction trigger selects the most-evictable in-flight request:
   - Lowest priority first (background=-3 before sheddable=-2 before batch=-1)
   - Among equal priority: newest dispatch time first (minimizes wasted compute)
3. The selected request is terminated on its instance (KV freed, resources released)
4. The freed capacity allows the waiting higher-priority request to dispatch

**Key properties:**

- Only sheddable requests (priority < 0) can be evicted — critical and standard are always protected
- Eviction is automatic when `--flow-control` is active — no additional flag needed
- Evicted requests are counted in `Gateway Evicted` output and are part of INV-1 conservation
- This is a terminal state — evicted requests are not requeued
```

**Verify:** `mkdocs build` (if available) or manual review
**Lint:** N/A (markdown)
**Commit:** `docs(guide): document in-flight eviction in flow control section`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 4 | Integration | TestGatewayEviction_EndToEnd |
| BC-2 | Task 1 | Unit | TestEvictionTracker_TrackAndPop, TestEvictionTracker_SamePriority_NewestDispatchFirst |
| BC-3 | Task 1, 3 | Unit + Integration | TestEvictionTracker_Untrack, TestEvictionTracking_OnDispatchAndCompletion |
| BC-4 | Task 5 | Integration | TestINV1_WithGatewayEviction |
| BC-5 | Task 2, 6 | Unit + Integration | TestGatewayEvictionEvent_TerminatesRequest, TestInstanceSimulator_EvictRequest |
| BC-6 | Task 1, 4 | Unit + Integration | TestEvictionTracker_NonSheddableNotTracked, TestGatewayEviction_NoEvictionWhenOnlyNonSheddable |
| BC-7 | Task 3 | Integration | TestEvictionTracking_OnDispatchAndCompletion (flow control disabled case) |
| BC-8 | Task 4 | Integration | TestGatewayEviction_NoEvictionWhenNotSaturated |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Eviction of a request that already completed (race between completion event and eviction event) | Low | Low | `EvictRequest` is a no-op if request not found; `findInstanceForRequest` returns "" if not found | Task 4, 6 |
| INV-1 double-counting: evicted request also counted as completed | Medium | High | Eviction removes from instance BEFORE completion can fire. Clear test in Task 5. | Task 5 |
| Heap corruption if Untrack called with wrong ID | Low | High | Map lookup returns false → no-op. Tested in Task 1. | Task 1 |
| Infinite eviction loop if eviction doesn't actually reduce saturation | Low | Medium | Only one eviction per dispatch attempt. After eviction, saturation re-computed on next call. | Task 4 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — no pluggable ordering/filter interfaces, hardcoded GIE behavior.
- [x] No feature creep — only in-flight eviction, no TTL, no EDF.
- [x] No unexercised flags — no new CLI flags.
- [x] No partial implementations — all contracts fully implemented.
- [x] No breaking changes.
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint.
- [x] Shared test helpers used from existing test package.
- [x] CLAUDE.md updated — add `gw_evicted` to INV-1 formula.
- [x] No stale references.
- [x] Documentation DRY — admission.md is the canonical source for eviction docs.
- [x] Deviation log reviewed — all deviations justified.
- [x] Each task produces working, testable code.
- [x] Task dependencies correctly ordered (1 → 2 → 3 → 4 → 5 → 6 → 7).
- [x] All contracts mapped to specific tasks.
- [x] Construction site audit completed — RawMetrics, ProgressSnapshot, ClusterSimulator.

**Antipattern rules:**
- [x] R1: No silent continue/return dropping data — EvictRequest no-op is intentional and logged.
- [x] R2: No map iteration for ordered output — heap ordering is deterministic.
- [x] R3: No new numeric CLI parameters (no new flags).
- [x] R4: All struct construction sites audited (3 sites listed in Phase 0).
- [x] R5: No resource allocation loops.
- [x] R6: No logrus.Fatalf in sim/ — using logrus.Debugf only.
- [x] R7: Integration tests validate INV-1 invariant alongside golden-style checks.
- [x] R8: No exported mutable maps.
- [x] R9: No new YAML fields.
- [x] R10: No new YAML parsing.
- [x] R11: No division by runtime-derived denominators.
- [x] R12: No golden dataset changes.
- [x] R13: No new interfaces (EvictionTracker is a concrete type).
- [x] R14: No method spans multiple module responsibilities.
- [x] R19: No unbounded retry loops — single eviction per dispatch attempt.
- [x] R21: No range over slices that shrink during iteration.

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/eviction_tracker.go`

- **Purpose:** Min-heap tracking dispatched sheddable requests for in-flight eviction.
- **Complete implementation:** See Task 1 above.
- **Key notes:**
  - **Ordering:** priority ASC (lowest evicted first), dispatch_time DESC (newest evicted first within same priority). This matches GIE's `PriorityThenTimeOrdering`.
  - **Filter:** Only `IsSheddable()` requests enter the heap (checked in `Track()`).
  - **State mutation:** `Track` adds to heap + map. `Untrack` removes from both. `Pop` removes from both.
  - **Error handling:** `Untrack` of unknown ID is a no-op. `Pop` on empty returns nil.

### File: `sim/cluster/cluster_event.go`

- **Purpose:** Add `GatewayEvictionEvent` — terminates an evicted request on its instance.
- **Event ordering:** Priority 5 (after routing at 2, before scaling at 10). Ensures eviction doesn't interfere with routing decisions at the same tick.
- **State mutation:** Decrements `inFlightRequests`, increments `gatewayEvicted`, calls `inst.EvictRequest()`.

### File: `sim/cluster/cluster.go`

- **Purpose:** Wire eviction tracker into dispatch/completion, add trigger logic.
- **New fields:** `evictionTracker *EvictionTracker`, `gatewayEvicted int`, `priorityMap *sim.SLOPriorityMap` (reference for IsSheddable checks).
- **Key integration points:**
  - `tryDispatchFromGatewayQueue()` — eviction trigger before DequeueGated
  - Completion handler — `evictionTracker.Untrack(req.ID)`
  - `tryDispatchFromGatewayQueue()` after dispatch — `evictionTracker.Track(req, pm)`

### File: `sim/cluster/gateway_queue.go`

- **Purpose:** Add `HasNonSheddableWaiting()` helper method.
- **Implementation:** Iterate bands in priority order; return true if any band with priority >= 0 has entries.

### File: `sim/simulator.go` (or equivalent)

- **Purpose:** Add `EvictRequest(*Request)` to InstanceSimulator.
- **Implementation:** Search WaitQ then RunningBatch for the request by ID. Remove and free KV blocks. No-op if not found.

### File: `sim/cluster/metrics.go`

- **Purpose:** Add `GatewayEvicted int` field to `RawMetrics`.
- **INV-1 comment update:** `injected == completed + running + queued + routing_rejections + dropped + timed_out + gw_depth + gw_shed + gw_rejected + gw_evicted`

### File: `sim/progress_hook.go`

- **Purpose:** Add `GatewayEvicted int` to `ProgressSnapshot` for progress reporting.

### File: `cmd/root.go` and `cmd/replay.go`

- **Purpose:** Populate `rawMetrics.GatewayEvicted`, display in anomaly output section.
- **Pattern:** Same as existing `GatewayQueueShed` / `GatewayQueueRejected` display logic.

### File: `docs/guide/admission.md`

- **Purpose:** Document in-flight eviction behavior for users.
- **Location:** New subsection under "Flow Control Admission" → "In-Flight Eviction".
