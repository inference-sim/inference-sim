# PR 1012: Remove Deferred-Queue Pre-Admission Intercept — Implementation Plan

**Goal:** Remove the pre-admission intercept in `AdmissionDecisionEvent.Execute()` that unconditionally defers batch/background requests before they reach `AdmissionPolicy.Admit()`, so that `always-admit` truly admits all requests and `tier-shed` properly evaluates batch/background under its priority logic.

**The problem today:** The intercept at `cluster_event.go:141` fires BEFORE `AdmissionPolicy.Admit()`, silently deferring batch/background requests when the cluster is busy. This makes `always-admit` dishonest (it doesn't truly admit all traffic), serializes batch requests causing 6-100x TTFT inflation (#965), and bypasses the admission policy contract. llm-d has no deferral concept at admission — the `LegacyAdmissionController` is a binary admit/reject.

**What this PR adds:**
1. Removes the pre-admission intercept block so batch/background requests flow through `Admit()` normally
2. Removes the batch/background bypass in `TierShedAdmission.Admit()` so tier-shed properly evaluates them
3. Removes the `DeferredHorizonInterrupted` metric field and all CLI wiring (no requests enter the deferred queue anymore)
4. Simplifies INV-1 conservation formula by dropping the `deferred_horizon_interrupted` term
5. Removes the post-event promotion check in the main event loop (queue is no longer fed)
6. Rewrites deferred queue tests to verify the new behavior: batch/background flow through admission normally

**Why this matters:** This is Part A of the GAIE-parity admission overhaul (#1011). It decouples admission from deferral, making the admission policy the single authority on admit/reject decisions. The deferred queue infrastructure stays for reuse in #899 (scheduling-tier deferral).

**Architecture:** Changes span 3 layers: DES event handling (`sim/cluster/cluster_event.go`), cluster event loop (`sim/cluster/cluster.go`), admission policy (`sim/admission.go`), metrics (`sim/cluster/metrics.go`), and CLI output wiring (`cmd/root.go`, `cmd/replay.go`). Tests in `sim/cluster/cluster_deferred_test.go` and `sim/cluster/cluster_test.go`.

**Source:** [Issue #1012](https://github.com/inference-sim/inference-sim/issues/1012), tracking issue [#1011](https://github.com/inference-sim/inference-sim/issues/1011)

**Closes:** `Fixes #1012`

---

## Phase 0: Component Context

1. **Building block:** Admission tier (`AdmissionDecisionEvent` + `AdmissionPolicy` interface)
2. **Adjacent blocks:** Cluster event loop (calls admission), CLI output (consumes `RawMetrics`), deferred queue infrastructure (stays untouched)
3. **Invariants touched:** INV-1 (conservation formula simplifies), INV-8 (work-conserving improves — no silent deferral)
4. **Construction site audit:**
   - `RawMetrics` struct: constructed in `sim/cluster/metrics.go:CollectRawMetrics()` (canonical). Fields populated in `cmd/root.go` and `cmd/replay.go` post-collection. Removing `DeferredHorizonInterrupted` field requires updating both CLI files.

---

## Part 1: Design Validation

### A) Executive Summary

This PR removes a pre-admission intercept that unconditionally defers batch/background requests before they reach `AdmissionPolicy.Admit()`. The intercept was added in Phase 1B-1b but creates a dishonest admission contract — `always-admit` doesn't truly admit all traffic, and `tier-shed` never sees batch/background requests.

After this PR: all requests flow through `Admit()` regardless of SLO class. `always-admit` admits everything. `tier-shed` evaluates batch/background against its priority thresholds like any other tier. The deferred queue infrastructure (`deferredQueue`, `promoteDeferred()`, `DeferredQueueLen()`, `isBusy()`) stays for reuse in #899.

This is a removal PR — it deletes code and simplifies behavior. No new types, interfaces, or CLI flags.

### B) Behavioral Contracts

**Positive contracts (what MUST happen):**

BC-1: Always-admit truly admits all requests
- GIVEN a cluster with `always-admit` admission policy (the default)
- WHEN batch or background requests arrive while the cluster is busy
- THEN they are admitted and routed normally (not deferred), and they complete successfully

BC-2: Tier-shed evaluates batch/background under overload
- GIVEN a cluster with `tier-shed` admission policy and `MinAdmitPriority=2` (which excludes batch=1 and background=0)
- WHEN the cluster is overloaded and batch/background requests arrive
- THEN they are rejected by the admission policy (not silently deferred)

BC-3: INV-1 conservation holds without deferred term
- GIVEN any cluster simulation run
- WHEN simulation completes
- THEN `injected == completed + queued + running + dropped + timed_out + routing_rejected + gw_depth + gw_shed` (no `deferred_horizon_interrupted` term)

**Negative contracts (what MUST NOT happen):**

BC-4: No pre-admission intercept exists
- GIVEN `AdmissionDecisionEvent.Execute()` receives a batch or background request
- WHEN the cluster is busy
- THEN the request still reaches `AdmissionPolicy.Admit()` (no early return to deferred queue)

BC-5: Deferred queue infrastructure preserved
- GIVEN this PR is merged
- WHEN checking the codebase
- THEN `deferredQueue` field, `promoteDeferred()`, `DeferredQueueLen()`, and `isBusy()` still exist (for #899)

### C) Component Interaction

```
Request Arrival
    │
    ▼
AdmissionDecisionEvent.Execute()
    │
    │  BEFORE this PR:                    AFTER this PR:
    │  ┌─────────────────────┐            ┌─────────────────────┐
    │  │ if batch/bg && busy │──► defer   │ (intercept removed) │
    │  └────────┬────────────┘            └────────┬────────────┘
    │           │                                   │
    ▼           ▼                                   ▼
buildRouterState()                      buildRouterState()
    │                                       │
    ▼                                       ▼
AdmissionPolicy.Admit()                AdmissionPolicy.Admit()
    │                                       │
    ├─ admitted ──► RoutingDecision     ├─ admitted ──► RoutingDecision
    └─ rejected ──► counter             └─ rejected ──► counter
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue says "Remove `sim/cluster/cluster.go:709-710`" (horizon warning about deferred requests) | Remove the entire `else if len(c.deferredQueue) > 0` block at line 709-711 | CLARIFICATION: the issue refers to lines 709-710 but the block spans 709-711 including the log message |
| No clarifications needed for other items | All other items match the codebase exactly | — |

### E) Review Guide

**Tricky part:** The `TierShedAdmission` bypass removal (BC-2). After removing the `if batch/background { return true }` block, tier-shed will reject low-priority tiers under overload. The test must verify this with a concrete overload scenario.

**What to scrutinize:** The INV-1 conservation formula changes — every test that checks conservation must be updated to remove the `DeferredQueueLen()` term.

**Safe to skim:** CLI wiring removals in `cmd/root.go` and `cmd/replay.go` — these are mechanical deletions of print statements.

**Known debt:** Deferred queue infrastructure stays as dead code until #899 reuses it. This is intentional per the issue.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action | Purpose |
|------|--------|---------|
| `sim/cluster/cluster_event.go` | Modify | Remove pre-admission intercept block (lines 137-149) and stale comment |
| `sim/cluster/cluster.go` | Modify | Remove post-event promotion check (lines 612-616) and horizon warning (lines 709-711) |
| `sim/admission.go` | Modify | Remove batch/background bypass in `TierShedAdmission.Admit()` (lines 121-124) and update comment |
| `sim/cluster/metrics.go` | Modify | Remove `DeferredHorizonInterrupted` field and update INV-1 comment |
| `cmd/root.go` | Modify | Remove 3 `DeferredHorizonInterrupted` references |
| `cmd/replay.go` | Modify | Remove 3 `DeferredHorizonInterrupted` references |
| `sim/cluster/cluster_test.go` | Modify | Remove `DeferredQueueLen()` from INV-1 conservation formula in test |
| `sim/cluster/cluster_deferred_test.go` | Rewrite | Replace deferral-asserting tests with admission-flow tests |
| `sim/admission_tier_test.go` | Modify | Remove/rewrite `TestTierShedAdmission_BatchAndBackgroundAlwaysAdmitted` (asserts old bypass behavior) |
| `sim/cluster/cluster_tier_test.go` | Modify | Rewrite `TestTierShed_BatchBackgroundNeverShed` (asserts old bypass behavior) |
| `docs/contributing/standards/invariants.md` | Modify | Update INV-1 to remove `deferred_horizon_interrupted` term |
| `CLAUDE.md` | Modify | Update INV-1 formula in Key Invariants section |

### G) Task Breakdown

---

#### Task 1: Remove pre-admission intercept and post-event promotion (BC-1, BC-4)

**What this does:** Removes the intercept block in `AdmissionDecisionEvent.Execute()` that defers batch/background requests before admission, and removes the post-event promotion check that feeds off this intercept.

**Files:** modify `sim/cluster/cluster_event.go`, modify `sim/cluster/cluster.go`

**Step 1 — Impl (remove intercept in cluster_event.go):**

In `sim/cluster/cluster_event.go`, delete the intercept code block at lines 137-149 (the comment and the `if` block through `return`). Also update the function's doc comment at lines 128-135 — remove the reference to deferring batch/background requests. The new doc comment should say:

```go
// Execute processes the admission decision for an incoming request.
// Checks admission policy with full RouterState (BC-8: includes snapshots).
// If admitted, schedules a RoutingDecisionEvent.
// If rejected, increments cs.rejectedRequests counter (EC-2).
```

**Step 2 — Impl (remove post-event promotion in cluster.go):**

In `sim/cluster/cluster.go`, delete the promotion check block at lines 612-616:
```go
		// Phase 1B-1b: after each event, promote deferred Batch/Background requests
		// if the cluster has become idle. INV-8: ensures no stall while deferred work waits.
		if len(c.deferredQueue) > 0 && !c.isBusy() {
			c.promoteDeferred()
		}
```

**Step 3 — Impl (remove horizon warning in cluster.go):**

In `sim/cluster/cluster.go`, delete the `else if` block at lines 709-711:
```go
		} else if len(c.deferredQueue) > 0 {
			logrus.Warnf("[cluster] no requests completed — %d batch/background requests remain deferred at horizon (cluster never became idle; mix in standard/critical traffic to trigger promotion)", len(c.deferredQueue))
		}
```

**Verify:** `go build ./sim/cluster/...`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `refactor(cluster): remove pre-admission intercept and promotion check (BC-1, BC-4)`

---

#### Task 2: Remove batch/background bypass in TierShedAdmission (BC-2)

**What this does:** Removes the `if class == "batch" || class == "background" { return true, "" }` bypass so tier-shed evaluates these classes under its priority logic like any other tier. Also updates tests that assert the old bypass behavior.

**Files:** modify `sim/admission.go`, modify `sim/admission_tier_test.go`, modify `sim/cluster/cluster_tier_test.go`

**Step 1 — Impl (admission.go):**

In `sim/admission.go`, update the struct doc comment at line 93 — remove:
```go
// Batch and Background always pass through (deferred queue PR handles them).
```

Delete the `Admit` method doc comment line 117 (`// Batch and Background classes always return admitted=true.`), and delete the bypass code block at lines 121-124:
```go
	// Batch/Background bypass tier-shed (deferred queue handles them in PR-2).
	if class == "batch" || class == "background" {
		return true, ""
	}
```

The updated `Admit` doc comment should be:
```go
// Admit rejects requests whose tier priority is below MinAdmitPriority when the
// cluster is overloaded (max effective load across instances > OverloadThreshold).
// Empty Snapshots (no instances) also returns admitted=true (safe default).
```

**Step 2 — Impl (admission_tier_test.go):**

In `sim/admission_tier_test.go`, rewrite `TestTierShedAdmission_BatchAndBackgroundAlwaysAdmitted` (line 122) to assert the NEW behavior — batch/background ARE rejected under overload when their priority is below `MinAdmitPriority`:

```go
// T011 — Batch and Background are rejected under overload when below MinAdmitPriority.
func TestTierShedAdmission_BatchAndBackgroundRejectedUnderOverload(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 0, MinAdmitPriority: 3}
	state := makeOverloadedState(9999)

	for _, class := range []string{"batch", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, _ := policy.Admit(req, state)
		if admitted {
			t.Errorf("class=%q (priority < MinAdmitPriority=3) should be rejected under overload, got admitted", class)
		}
	}
}
```

**Step 3 — Impl (cluster_tier_test.go):**

In `sim/cluster/cluster_tier_test.go`, rewrite `TestTierShed_BatchBackgroundNeverShed` (line 119) to assert the NEW behavior — batch/background ARE shed under overload:

```go
// Batch and Background ARE shed by tier-shed when below MinAdmitPriority.
func TestTierShed_BatchBackgroundShedUnderOverload(t *testing.T) {
	const n = 40
	var requests []*sim.Request
	for _, class := range []string{"batch", "background"} {
		for i := 0; i < n; i++ {
			requests = append(requests, &sim.Request{
				ID:           fmt.Sprintf("req_%s_%d", class, i),
				ArrivalTime:  int64(i) * 5,
				SLOClass:     class,
				InputTokens:  make([]int, 100),
				OutputTokens: make([]int, 50),
				State:        sim.StateQueued,
			})
		}
	}

	cfg := newTierShedConfig(0, 3) // MinAdmitPriority=3 rejects batch(1) and background(0)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	// With MinAdmitPriority=3, batch(priority=1) and background(priority=0) should be rejected
	if cs.RejectedRequests() == 0 {
		t.Errorf("batch/background should be rejected under overload with MinAdmitPriority=3, got RejectedRequests=0")
	}
}
```

**Verify:** `go build ./sim/...`
**Lint:** `golangci-lint run ./sim/... ./sim/cluster/...`
**Commit:** `refactor(admission): remove batch/background bypass in tier-shed (BC-2)`

---

#### Task 3: Remove DeferredHorizonInterrupted metric and CLI wiring (BC-3)

**What this does:** Removes the `DeferredHorizonInterrupted` field from `RawMetrics` and all CLI code that populates/prints it.

**Files:** modify `sim/cluster/metrics.go`, modify `cmd/root.go`, modify `cmd/replay.go`

**Step 1 — Impl (metrics.go):**

In `sim/cluster/metrics.go`, delete line 106:
```go
	DeferredHorizonInterrupted int           // Batch/Background requests still deferred at horizon (Phase 1B-1b)
```

Update the INV-1 comment on line 105 from:
```go
	// INV-1 extended: injected == completed + running + queued + shed + dropped + timed_out + deferred_horizon_interrupted + gw_depth + gw_shed
```
to:
```go
	// INV-1 extended: injected == completed + running + queued + shed + dropped + timed_out + gw_depth + gw_shed
```

**Step 2 — Impl (cmd/root.go):**

Remove 3 lines referencing `DeferredHorizonInterrupted`:

1. Line 1629: `rawMetrics.DeferredHorizonInterrupted = cs.DeferredQueueLen()` — delete entire line
2. Line 1661: Remove `|| rawMetrics.DeferredHorizonInterrupted > 0` from the anomaly check condition
3. Lines 1679-1680: Delete the print block:
```go
			if rawMetrics.DeferredHorizonInterrupted > 0 {
				fmt.Printf("Deferred (horizon-interrupted): %d\n", rawMetrics.DeferredHorizonInterrupted)
			}
```

**Step 3 — Impl (cmd/replay.go):**

Remove 3 lines referencing `DeferredHorizonInterrupted`:

1. Line 247: `rawMetrics.DeferredHorizonInterrupted = cs.DeferredQueueLen()` — delete entire line
2. Line 250: Remove `|| rawMetrics.DeferredHorizonInterrupted > 0` from the anomaly check condition
3. Lines 268-269: Delete the print block:
```go
			if rawMetrics.DeferredHorizonInterrupted > 0 {
				fmt.Printf("Deferred (horizon-interrupted): %d\n", rawMetrics.DeferredHorizonInterrupted)
			}
```

**Verify:** `go build ./...`
**Lint:** `golangci-lint run ./cmd/... ./sim/cluster/...`
**Commit:** `refactor(metrics): remove DeferredHorizonInterrupted field and CLI wiring (BC-3)`

---

#### Task 4: Update INV-1 conservation in tests (BC-3)

**What this does:** Removes `DeferredQueueLen()` from the INV-1 conservation formula in `cluster_test.go`.

**Files:** modify `sim/cluster/cluster_test.go`

**Step 1 — Impl:**

At line 1793, update the INV-1 comment from:
```go
	// INV-1: injected == completed + queued + running + dropped + timedout + deferred + routingRejections + gwDepth + gwShed
```
to:
```go
	// INV-1: injected == completed + queued + running + dropped + timedout + routingRejections + gwDepth + gwShed
```

At line 1795, remove `cs.DeferredQueueLen() +` from the accounted sum:
```go
	accounted := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable + m.TimedOutRequests + cs.RoutingRejections() + gwDepth + gwShed
```

At lines 1797-1800, remove `deferred=%d` and `cs.DeferredQueueLen()` from the error message:
```go
		if injected != accounted {
			t.Errorf("INV-1: injected=%d != accounted=%d (completed=%d queued=%d running=%d dropped=%d timedout=%d routingRejections=%d gwDepth=%d gwShed=%d)",
				injected, accounted,
				m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable,
				m.TimedOutRequests, cs.RoutingRejections(), gwDepth, gwShed)
		}
```

Also update the comment at line ~2455:
```go
// NOTE: assertINV1Conservation checks 5 of 9 INV-1 terms; the 4 missing terms
// (DeferredHorizonInterrupted, RoutingRejections, GatewayQueueDepth, GatewayQueueShed)
```
Change to:
```go
// NOTE: assertINV1Conservation checks 5 of 8 INV-1 terms; the 3 missing terms
// (RoutingRejections, GatewayQueueDepth, GatewayQueueShed)
```

**Verify:** `go test ./sim/cluster/... -run TestClusterSimulator_FlowControl_INV1`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `test(cluster): update INV-1 conservation formula — remove deferred term (BC-3)`

---

#### Task 5: Rewrite deferred queue tests (BC-1, BC-2, BC-3, BC-4, BC-5)

**What this does:** Replaces the old deferral-asserting tests with tests that verify the new behavior: batch/background flow through admission normally, tier-shed rejects them under overload, conservation holds without deferred term, and deferred queue infrastructure still exists.

**Files:** rewrite `sim/cluster/cluster_deferred_test.go`

**Step 1 — Write complete replacement test file:**

```go
package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// newBatchTestRequests creates n requests with the given SLOClass,
// arriving every 10us starting at t=0, with 50 input tokens and 20 output tokens.
func newBatchTestRequests(n int, sloClass string) []*sim.Request {
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("req_%s_%d", sloClass, i),
			ArrivalTime:  int64(i) * 10,
			SLOClass:     sloClass,
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 20),
			State:        sim.StateQueued,
		}
	}
	return reqs
}

// TestAlwaysAdmit_BatchNotDeferred (BC-1, BC-4) verifies that batch and background
// requests are admitted normally under always-admit (the default), even when the
// cluster is busy. They must NOT be silently deferred.
//
// Setup: 30 standard requests (keep cluster busy) + 1 batch/background request.
// Before this PR, the batch/background request would be deferred. After this PR,
// it flows through Admit() and is admitted.
func TestAlwaysAdmit_BatchNotDeferred(t *testing.T) {
	for _, sloClass := range []string{"batch", "background"} {
		t.Run(sloClass, func(t *testing.T) {
			var requests []*sim.Request
			for i := 0; i < 30; i++ {
				requests = append(requests, &sim.Request{
					ID:           fmt.Sprintf("std_%d", i),
					ArrivalTime:  int64(i) * 5,
					SLOClass:     "standard",
					InputTokens:  make([]int, 100),
					OutputTokens: make([]int, 50),
					State:        sim.StateQueued,
				})
			}
			requests = append(requests, &sim.Request{
				ID:           sloClass + "_0",
				ArrivalTime:  50,
				SLOClass:     sloClass,
				InputTokens:  make([]int, 50),
				OutputTokens: make([]int, 20),
				State:        sim.StateQueued,
			})

			cfg := newTestDeploymentConfig(1)
			cs := NewClusterSimulator(cfg, requests, nil)
			mustRun(t, cs)

			// BC-1: batch/background must not be rejected
			if cs.RejectedRequests() > 0 {
				t.Errorf("%s request should be admitted (always-admit), got RejectedRequests=%d", sloClass, cs.RejectedRequests())
			}
			// BC-4: deferred queue must be empty — no pre-admission intercept
			if cs.DeferredQueueLen() != 0 {
				t.Errorf("%s request should NOT be deferred (intercept removed), got DeferredQueueLen=%d", sloClass, cs.DeferredQueueLen())
			}
		})
	}
}

// TestTierShed_RejectsBatchUnderOverload (BC-2) verifies that tier-shed rejects
// batch/background requests when the cluster is overloaded and their priority is
// below MinAdmitPriority.
//
// Setup: tier-shed with MinAdmitPriority=2 (rejects batch=1, background=0).
// Dense standard traffic to create overload. Batch request arrives under overload.
func TestTierShed_RejectsBatchUnderOverload(t *testing.T) {
	for _, tc := range []struct {
		sloClass string
		priority int // batch=1, background=0
	}{
		{"batch", 1},
		{"background", 0},
	} {
		t.Run(tc.sloClass, func(t *testing.T) {
			var requests []*sim.Request
			// Dense standard traffic to trigger overload
			for i := 0; i < 50; i++ {
				requests = append(requests, &sim.Request{
					ID:           fmt.Sprintf("std_%d", i),
					ArrivalTime:  int64(i) * 2,
					SLOClass:     "standard",
					InputTokens:  make([]int, 200),
					OutputTokens: make([]int, 100),
					State:        sim.StateQueued,
				})
			}
			// Low-priority request arrives while cluster is saturated
			requests = append(requests, &sim.Request{
				ID:           tc.sloClass + "_0",
				ArrivalTime:  10,
				SLOClass:     tc.sloClass,
				InputTokens:  make([]int, 50),
				OutputTokens: make([]int, 20),
				State:        sim.StateQueued,
			})

			cfg := newTestDeploymentConfig(1)
			cfg.AdmissionPolicy = "tier-shed"
			cfg.TierShedThreshold = 1 // any load triggers overload
			cfg.TierShedMinPriority = 2  // rejects batch(1) and background(0)
			cs := NewClusterSimulator(cfg, requests, nil)
			mustRun(t, cs)

			// BC-2: tier-shed must reject the low-priority request under overload
			if cs.RejectedRequests() == 0 {
				t.Errorf("tier-shed should reject %s (priority=%d < min=2) under overload, got RejectedRequests=0", tc.sloClass, tc.priority)
			}
			// BC-4: deferred queue must be empty — no pre-admission intercept
			if cs.DeferredQueueLen() != 0 {
				t.Errorf("deferred queue should be empty (intercept removed), got DeferredQueueLen=%d", cs.DeferredQueueLen())
			}
		})
	}
}

// TestINV1_NoDeferredTerm (BC-3) verifies INV-1 conservation holds without the
// deferred_horizon_interrupted term. Uses mixed SLO-class traffic.
func TestINV1_NoDeferredTerm(t *testing.T) {
	var requests []*sim.Request
	// Mix of standard and batch traffic
	for i := 0; i < 10; i++ {
		requests = append(requests, &sim.Request{
			ID:           fmt.Sprintf("std_%d", i),
			ArrivalTime:  int64(i) * 20,
			SLOClass:     "standard",
			InputTokens:  make([]int, 80),
			OutputTokens: make([]int, 40),
			State:        sim.StateQueued,
		})
	}
	for i := 0; i < 5; i++ {
		requests = append(requests, &sim.Request{
			ID:           fmt.Sprintf("batch_%d", i),
			ArrivalTime:  int64(i) * 20,
			SLOClass:     "batch",
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 25),
			State:        sim.StateQueued,
		})
	}

	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	// BC-3: INV-1 without deferred term
	injected := len(requests) - cs.RejectedRequests()
	accounted := m.CompletedRequests + m.StillQueued + m.StillRunning +
		m.DroppedUnservable + m.TimedOutRequests + cs.RoutingRejections()
	if injected != accounted {
		t.Errorf("INV-1: injected=%d != accounted=%d (completed=%d queued=%d running=%d dropped=%d timedout=%d routingRejected=%d)",
			injected, accounted,
			m.CompletedRequests, m.StillQueued, m.StillRunning,
			m.DroppedUnservable, m.TimedOutRequests, cs.RoutingRejections())
	}
	// BC-4: deferred queue must be empty
	if cs.DeferredQueueLen() != 0 {
		t.Errorf("deferred queue should be empty (intercept removed), got DeferredQueueLen=%d", cs.DeferredQueueLen())
	}
}

// TestDeferredQueueInfraExists (BC-5) verifies the deferred queue infrastructure
// is still callable (preserved for #899).
func TestDeferredQueueInfraExists(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	requests := newBatchTestRequests(1, "batch")
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	// BC-5: DeferredQueueLen() must still be callable
	dql := cs.DeferredQueueLen()
	if dql != 0 {
		t.Errorf("expected DeferredQueueLen=0 (nothing feeds the queue now), got %d", dql)
	}
}

// TestBatchRequestsNotSerialized verifies that batch requests are NOT serialized
// after the intercept removal — they flow through admission like standard requests.
// This is a regression guard for issue #965.
func TestBatchRequestsNotSerialized(t *testing.T) {
	requests := newBatchTestRequests(10, "batch")
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 10 {
		t.Fatalf("completed %d requests, want 10", m.CompletedRequests)
	}

	// After intercept removal, batch requests should have similar TTFT to standard
	// (not serialized). Using the same 15ms bound from the old test.
	ttftMeanMs := float64(m.TTFTSum) / float64(m.CompletedRequests) / 1000.0
	const boundMs = 15.0
	if ttftMeanMs >= boundMs {
		t.Errorf("mean TTFT %.2fms >= bound %.1fms: batch requests are being serialized (regression: #965)", ttftMeanMs, boundMs)
	}
}
```

**Verify:** `go test ./sim/cluster/... -run "TestAlwaysAdmit_BatchNotDeferred|TestTierShed_RejectsBatchUnderOverload|TestINV1_NoDeferredTerm|TestDeferredQueueInfraExists|TestBatchRequestsNotSerialized"`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `test(cluster): rewrite deferred queue tests for intercept removal (BC-1..BC-5)`

---

#### Task 6: Update documentation — invariants.md and CLAUDE.md (BC-3)

**What this does:** Updates the canonical INV-1 definition in `invariants.md` and the CLAUDE.md working copy to remove `deferred_horizon_interrupted` from the conservation formula.

**Files:** modify `docs/contributing/standards/invariants.md`, modify `CLAUDE.md`

**Step 1 — Impl (invariants.md):**

Update the cluster-level extension paragraph (line 11) — change:
```
`injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out + deferred_horizon_interrupted + routing_rejections + gateway_queue_depth + gateway_queue_shed`
```
to:
```
`injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed`
```

Remove the sentence: `` `deferred_horizon_interrupted` counts Batch/Background requests still parked in the deferred queue when the simulation horizon is reached. ``

Update the Verification line (line 15) — remove `and `sim/cluster/cluster_deferred_test.go`` from the conservation tests reference (the file still exists but the conservation test is now `TestINV1_NoDeferredTerm`). Remove the sentence about `DeferredHorizonInterrupted` being in `RawMetrics`.

**Step 2 — Impl (CLAUDE.md):**

Update the INV-1 line to match invariants.md:
```
- **INV-1 Request conservation**: `injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed` at simulation end. Full pipeline: `num_requests == injected_requests + rejected_requests`.
```

**Verify:** Confirm no remaining references to `deferred_horizon_interrupted` in the modified files.
**Commit:** `docs(invariants): update INV-1 formula — remove deferred_horizon_interrupted term (BC-3)`

---

#### Task 7: Run full test suite and verify (all contracts)

**What this does:** Final verification that all tests pass, lint is clean, and the build succeeds.

**Steps:**
1. `go build ./...`
2. `go test ./... -count=1`
3. `golangci-lint run ./...`

**Expected:** All pass. No lint issues. No test failures.

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 5 | Integration | `TestAlwaysAdmit_BatchNotDeferred` |
| BC-2 | Task 2, 5 | Unit + Integration | `TestTierShedAdmission_BatchAndBackgroundRejectedUnderOverload`, `TestTierShed_BatchBackgroundShedUnderOverload`, `TestTierShed_RejectsBatchUnderOverload` |
| BC-3 | Task 4, 5 | Integration | `TestINV1_NoDeferredTerm`, `TestClusterSimulator_FlowControl_INV1` |
| BC-4 | Task 5 | Integration | Verified as assertions within BC-1, BC-2, BC-3 tests (DeferredQueueLen==0) |
| BC-5 | Task 5 | Unit | `TestDeferredQueueInfraExists` |
| #965 reg | Task 5 | Integration | `TestBatchRequestsNotSerialized` |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Removing intercept breaks conservation | Low | High | `TestINV1_NoDeferredTerm` + `TestClusterSimulator_FlowControl_INV1` verify conservation | Task 4, 5 |
| Tier-shed bypass removal causes unexpected rejections in default config | Low | Medium | Default is `always-admit`, not `tier-shed`; test verifies tier-shed behavior explicitly | Task 2, 5 |
| Stale reference to `DeferredHorizonInterrupted` somewhere | Low | Low | Grep for all references before committing | Task 3, 7 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — this is pure removal
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (`newTestDeploymentConfig`, `mustRun`)
- [x] CLAUDE.md updated (INV-1 formula)
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: invariants.md (canonical) and CLAUDE.md (working copy) both updated
- [x] Deviation log reviewed — one clarification documented
- [x] Each task produces working, testable code
- [x] Task dependencies are correctly ordered (1→2→3→4→5→6→7)
- [x] All contracts mapped to specific tasks
- [x] Construction site audit completed — `RawMetrics` field removal covered in Task 3

**Antipattern rules:**
- [x] R1: No silent continue/return — we're removing the silent return
- [x] R4: `RawMetrics` construction site — field removed from struct + both CLI files
- [x] R15: Stale PR references resolved (Phase 1B-1b references removed from modified code)
- Other rules (R2, R3, R5-R14, R16-R23) not applicable — this PR removes code, adds no new types/interfaces/flags

---

## Appendix: File-Level Implementation Details

**File: `sim/cluster/cluster_event.go`**
- Purpose: Remove pre-admission intercept (lines 128-149)
- State mutation: None removed (the intercept mutated `cs.deferredQueue`)
- Error handling: No change

**File: `sim/cluster/cluster.go`**
- Purpose: Remove post-event promotion check (lines 612-616) and horizon warning (lines 709-711)
- State mutation: Promotion check called `promoteDeferred()` which mutated `deferredQueue` — no longer called from this path
- The `promoteDeferred()` method itself stays (for #899)

**File: `sim/admission.go`**
- Purpose: Remove batch/background bypass in `TierShedAdmission.Admit()` (lines 117-124)
- After removal, batch/background flow through the normal priority evaluation path

**File: `sim/cluster/metrics.go`**
- Purpose: Remove `DeferredHorizonInterrupted` field and update INV-1 comment

**File: `cmd/root.go`**
- Purpose: Remove 3 references to `DeferredHorizonInterrupted` (assignment, condition, print)

**File: `cmd/replay.go`**
- Purpose: Remove 3 references to `DeferredHorizonInterrupted` (assignment, condition, print)

**File: `sim/cluster/cluster_test.go`**
- Purpose: Remove `DeferredQueueLen()` from INV-1 conservation formula

**File: `sim/cluster/cluster_deferred_test.go`**
- Purpose: Complete rewrite with 5 tests covering BC-1 through BC-5 + #965 regression guard

**File: `docs/contributing/standards/invariants.md`**
- Purpose: Update canonical INV-1 definition

**File: `CLAUDE.md`**
- Purpose: Update INV-1 working copy to match invariants.md
