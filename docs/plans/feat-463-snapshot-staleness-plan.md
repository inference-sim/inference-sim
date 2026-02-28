# feat(snapshot): Unify Routing Signal Staleness Model — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make all Prometheus-sourced routing signals (QueueDepth, BatchSize, KVUtilization) share the same configurable staleness, and model the InFlightRequests signal with a realistic dispatch-to-response lifecycle window instead of a microsecond-scale counter.

**The problem today:** BLIS models QueueDepth and BatchSize as instantaneous reads, but in real vLLM deployments all three metrics (including KVUtilization) are scraped from the same Prometheus endpoint at the same interval (2-15s). Additionally, PendingRequests tracks a microsecond-scale window (routed→enqueued), but in real HTTP routers (llm-d, Envoy), in-flight request tracking spans the full dispatch-to-response window (milliseconds to seconds) — a 3-4 order-of-magnitude fidelity gap.

**What this PR adds:**
1. **Unified Prometheus staleness** — when `--snapshot-refresh-interval` is set, QueueDepth and BatchSize use the same Periodic refresh as KVUtilization, matching the real-world single-endpoint scrape model.
2. **InFlightRequests signal** — replaces PendingRequests with a realistic lifecycle: increment at dispatch (routing decision), decrement at response (request completion), not at queue absorption. The `EffectiveLoad()` formula is updated accordingly.
3. **INV-7 update** — the signal freshness hierarchy documentation is updated to reflect the new uniform staleness model and InFlightRequests semantics.

**Why this matters:** The simulator's routing fidelity depends on how accurately it models what real load balancers see. Experiments H29 and H4 showed that signal staleness and load tracking semantics significantly affect routing quality. This PR closes the gap between BLIS's idealized model and real inference serving platforms (llm-d, Envoy, KServe).

**Architecture:** Phase 1 changes `newObservabilityConfig()` in `sim/cluster/snapshot.go` to apply the refresh interval uniformly. Phase 2 renames `PendingRequests` → `InFlightRequests` across `sim/routing.go`, `sim/cluster/cluster.go`, `sim/cluster/cluster_event.go`, updates the decrement trigger from `QueuedEvent` to request completion, and adjusts `EffectiveLoad()`. Documentation updates to INV-7, CLAUDE.md, and scorer R17 comments accompany the code changes.

**Source:** GitHub issue #463

**Closes:** Fixes #463

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR makes two targeted fidelity improvements to BLIS's routing signal model:

1. **Uniform Prometheus staleness (Phase 1):** The `newObservabilityConfig()` factory currently applies `Periodic` mode only to KVUtilization when a refresh interval is set, leaving QueueDepth and BatchSize as `Immediate`. In real vLLM, all three are Prometheus gauges from the same `/metrics` endpoint with the same scrape interval. The fix: when `SnapshotRefreshInterval > 0`, all three fields use `Periodic`.

2. **Realistic in-flight tracking (Phase 2):** `PendingRequests` currently tracks a microsecond-scale window (routed → queue absorbed via `QueuedEvent`). Real HTTP routers track in-flight requests from dispatch to response receipt. The fix: rename to `InFlightRequests`, change the decrement trigger from `QueuedEvent` to request completion (via `Metrics.Requests` map population), and update `EffectiveLoad()`.

**Adjacent blocks:** `CachedSnapshotProvider` (snapshot.go), `ClusterSimulator` event loop (cluster.go, cluster_event.go), `RoutingSnapshot`/`EffectiveLoad()` (routing.go), all scorers reading `EffectiveLoad()` (routing_scorers.go), counterfactual scoring (counterfactual.go), pending_requests tests, INV-7 documentation.

**No deviations from issue #463** — Phase 3 (re-validation of H29/H4) is explicitly deferred per the issue's phased plan.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Uniform Prometheus Staleness
- GIVEN `SnapshotRefreshInterval > 0` (e.g., 100μs)
- WHEN `newObservabilityConfig(100)` is called
- THEN the returned config has QueueDepth, BatchSize, AND KVUtilization all set to Periodic mode with the same interval
- MECHANISM: `newObservabilityConfig` sets all three `FieldConfig` entries to `{Mode: Periodic, Interval: refreshInterval}` when interval > 0

BC-2: Zero Interval Preserves Immediate Behavior
- GIVEN `SnapshotRefreshInterval == 0`
- WHEN `newObservabilityConfig(0)` is called
- THEN all three fields use Immediate mode (backward-compatible, identical to current behavior)
- MECHANISM: The `if refreshInterval <= 0` guard returns `DefaultObservabilityConfig()` unchanged

BC-3: InFlightRequests Tracks Dispatch-to-Completion Window
- GIVEN a cluster with routing latency and multiple requests
- WHEN requests are routed and subsequently complete (or are dropped as unservable)
- THEN `InFlightRequests` for each instance is incremented at routing decision time and decremented when the request completes OR is dropped (not when it enters the queue)
- MECHANISM: Increment in `RoutingDecisionEvent.Execute()`, decrement in the event loop based on `CompletedRequests + DroppedUnservable` counter delta after each instance event

BC-4: InFlightRequests Accounting Invariant Post-Simulation
- GIVEN a simulation that runs (some or all requests may complete before horizon)
- WHEN the simulation ends
- THEN for each instance, `inFlightRequests[inst] == StillQueued[inst] + StillRunning[inst]` (requests that haven't completed remain in-flight; completed and dropped requests have been decremented)
- MECHANISM: Post-simulation check compares `inFlightRequests` against `StillQueued + StillRunning` per instance. A mismatch (not just non-zero) indicates a bookkeeping bug. When all requests complete, both sides are 0.

BC-5: EffectiveLoad Includes InFlightRequests
- GIVEN a RoutingSnapshot with QueueDepth=2, BatchSize=1, InFlightRequests=3
- WHEN `EffectiveLoad()` is called
- THEN the result is 6 (2 + 1 + 3)
- MECHANISM: `EffectiveLoad()` formula uses renamed `InFlightRequests` field

BC-7: DroppedUnservable Requests Decrement InFlightRequests
- GIVEN a cluster where a request is routed to an instance with insufficient KV capacity
- WHEN the request is dropped as unservable (DroppedUnservable incremented in EnqueueRequest)
- THEN `InFlightRequests` for that instance is decremented (the request exits the in-flight window)
- MECHANISM: Decrement logic tracks `DroppedUnservable` delta alongside `CompletedRequests` delta

BC-8: EffectiveLoad Double-Counting is Intentional
- GIVEN the formula `EffectiveLoad() = QueueDepth + BatchSize + InFlightRequests`
- WHEN InFlightRequests tracks dispatch-to-completion (overlapping with QueueDepth and BatchSize)
- THEN EffectiveLoad ≈ 2×(QD+BS) + pipeline_requests, which preserves relative ranking for `scoreQueueDepth` (min-max normalization is scale-invariant) and `LeastLoaded` (argmin unaffected by monotone inflation)
- MECHANISM: The double-counting combines stale Prometheus signals (QD/BS) with the synchronous gateway signal (InFlightRequests), providing both historical and real-time load views. Real load balancers (llm-d Endpoint Picker) similarly overlap in-flight HTTP connections with Prometheus-scraped metrics. **Note:** `scoreLoadBalance` (`1/(1+x)`) is a nonlinear transform — ratios shift slightly under doubled inputs (e.g., 1.83→1.90), though the argmax winner is robust because multi-scorer weighting dominates. Scorer weight retuning deferred to Phase 3 (H29/H4 re-validation).

BC-6: INV-7 Documentation Reflects New Model
- GIVEN the updated code
- WHEN a contributor reads `docs/contributing/standards/invariants.md` INV-7
- THEN the signal freshness table shows QueueDepth and BatchSize as "Periodic (when interval > 0)" and InFlightRequests (not PendingRequests) as the gateway signal
- MECHANISM: Documentation update in INV-7 section

**Negative Contracts:**

NC-1a: Phase 1 (Staleness Config) Backward Compatible at Default Settings
- GIVEN `SnapshotRefreshInterval == 0` (the default)
- WHEN `newObservabilityConfig(0)` is called
- THEN all fields use Immediate mode, identical to pre-PR behavior
- MECHANISM: `newObservabilityConfig(0)` returns `DefaultObservabilityConfig()` unchanged

NC-1b: Phase 2 (InFlightRequests) Changes Routing Behavior — Intentional
- GIVEN any cluster simulation (any `SnapshotRefreshInterval`)
- WHEN simulation runs with load-aware routing (least-loaded, weighted, always-busiest)
- THEN routing decisions WILL differ from pre-PR behavior because `InFlightRequests` holds higher values (dispatch-to-completion vs dispatch-to-queue window), changing `EffectiveLoad()` inputs
- MECHANISM: This is an intentional fidelity improvement, not a regression. Golden dataset MUST be regenerated (R12). RoundRobin routing is unaffected (does not read EffectiveLoad).

NC-2: InFlightRequests Never Goes Negative
- GIVEN any simulation
- WHEN events are processed
- THEN `inFlightRequests[instID]` is never decremented below 0
- MECHANISM: Guard `if c.inFlightRequests[instID] > 0` before decrement (existing pattern)

**Error Handling:**

EC-1: Negative Refresh Interval Treated as Zero
- GIVEN `SnapshotRefreshInterval < 0`
- WHEN `newObservabilityConfig(-5)` is called
- THEN returns `DefaultObservabilityConfig()` (all Immediate)
- MECHANISM: Existing `if refreshInterval <= 0` guard

### C) Component Interaction

```
Request Arrival → Admission → Routing Decision → Instance Injection → ... → Completion
                                 │                      │                        │
                                 │ ++inFlightRequests    │                        │ --inFlightRequests
                                 │                      │                        │
                                 ▼                      ▼                        ▼
                          [snapshot.go]           [cluster.go]             [cluster.go]
                     CachedSnapshotProvider     pendingRequests→       event loop detects
                     QueueDepth: Periodic*      inFlightRequests       completion, decrements
                     BatchSize:  Periodic*
                     KVUtil:     Periodic*
                     (* when interval > 0; Immediate when interval == 0)
```

**API changes:**
- `RoutingSnapshot.PendingRequests` → `RoutingSnapshot.InFlightRequests` (rename)
- `EffectiveLoad()` unchanged signature, uses renamed field
- `newObservabilityConfig()` unchanged signature, changed behavior (applies interval to all fields)
- `ClusterSimulator.pendingRequests` → `ClusterSimulator.inFlightRequests` (unexported, rename)

**State lifecycle for InFlightRequests:**
- Created: `make(map[string]int)` in `NewClusterSimulator`
- Incremented: `RoutingDecisionEvent.Execute()` (same as before)
- Decremented: Event loop when processing a completion event (NEW — was `QueuedEvent` before)
- Destroyed: End of simulation

**Extension friction:** 1 file to add a new field to `RoutingSnapshot` (already exists, just renamed). Low friction.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| Phase 2 renames `PendingRequests` → `InFlightRequests` and changes decrement to response completion | Same | Direct implementation |
| Phase 2 updates `EffectiveLoad()` formula | Formula unchanged — still `QueueDepth + BatchSize + InFlightRequests` | SIMPLIFICATION: The formula doesn't change, only the field name. The *semantics* change (wider window) but the *formula* is the same sum. |
| Phase 3: Re-validate H29 and H4 | Deferred | DEFERRAL: Issue #463 explicitly describes this as a separate phase. Not in scope for this PR. |
| Issue mentions updating `sim/routing_scorers.go` | Only R17 doc comments update | SIMPLIFICATION: Scorer function bodies don't reference `PendingRequests` directly — they call `EffectiveLoad()`. Only the R17 freshness doc comments need updating. |
| Issue Phase 2 specifies streaming (decrement at TTFT) vs non-streaming (decrement at E2E) | Plan always decrements at E2E completion | SIMPLIFICATION: BLIS does not currently model streaming vs non-streaming response modes. All requests complete when all output tokens are generated. Adding streaming-aware decrement would require a new request metadata field and event type — deferred to a future PR if streaming mode is added to the simulator. |
| (Not in issue) | CandidateScore JSON field name changes from `PendingRequests` to `InFlightRequests` | ADDITION: Breaking change to trace schema. No explicit JSON tags exist — Go uses field name. Downstream trace parsers must update. Acceptable for pre-1.0 project. |
| (Not in issue) | CacheHitRate and FreeKVBlocks inherit KVUtilization's refresh timing | ADDITION: These piggyback on the KVUtilization `shouldRefresh` check (snapshot.go:109-113). When KVUtil becomes Periodic, they also become Periodic. This is correct (same Prometheus endpoint) but worth documenting. |
| (Not in issue) | EffectiveLoad double-counts under new semantics | ADDITION: `EffectiveLoad() ≈ 2×(QD+BS)` because InFlightRequests overlaps with QD/BS. This is intentional — preserves relative ranking, combines stale+fresh signals. See BC-8. Scorer weight retuning deferred to Phase 3. |

### E) Review Guide

**The tricky part:** The InFlightRequests decrement trigger change. Currently `QueuedEvent` decrements `pendingRequests`. We need to find the completion signal in the cluster event loop. The instance-level `Simulator` marks requests as `StateCompleted` and records them in `Metrics.Requests`. The cluster event loop processes instance events via `ProcessNextEvent()` — we need to detect completion there. The approach: check `Metrics.CompletedRequests` delta after each instance event, or listen for a specific completion event type.

**What to scrutinize:** BC-3 (the decrement trigger change) and BC-4 (drains to zero). The rest is mechanical renaming and config changes.

**What's safe to skim:** BC-1/BC-2 (config factory — 2 lines), BC-6 (documentation), NC-1 (backward compat — follows from BC-2).

**Known debt:** The H29 and H4 experiments may need re-validation under the new staleness model (Phase 3, deferred to future PR).

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/cluster/snapshot.go` — `newObservabilityConfig`: apply interval to all 3 fields (BC-1, BC-2)
- `sim/routing.go` — Rename `PendingRequests` → `InFlightRequests` in `RoutingSnapshot`, update `EffectiveLoad()` comment (BC-5)
- `sim/cluster/cluster.go` — Rename `pendingRequests` → `inFlightRequests`, change decrement from QueuedEvent to completion detection (BC-3, BC-4)
- `sim/cluster/cluster_event.go` — Rename `pendingRequests` → `inFlightRequests` in `buildRouterState` and `RoutingDecisionEvent.Execute` (BC-3)
- `sim/routing_scorers.go` — Update R17 doc comments to reflect unified staleness (BC-6)
- `sim/routing_prefix_scorer.go` — Update R17 doc comment
- `sim/cluster/deployment.go` — Update `SnapshotRefreshInterval` comment to note it now applies to all fields
- `sim/cluster/snapshot_test.go` — Add/update tests for unified staleness
- `sim/cluster/pending_requests_test.go` — Rename to `inflight_requests_test.go`, update to test completion-based decrement
- `docs/contributing/standards/invariants.md` — Update INV-7 table (BC-6)
- `CLAUDE.md` — Update INV-7 description and SnapshotRefreshInterval description
- `sim/cluster/counterfactual.go` — Update any PendingRequests references
- `sim/trace/record.go` — Rename PendingRequests field in CandidateScore

**Key decisions:**
- Completion detection: Use a `CompletedRequests` counter snapshot before/after each instance event. When `Metrics.CompletedRequests` increases, the delta equals the number of completions — decrement `inFlightRequests` by that amount. This is simpler than detecting specific event types and matches the existing pattern of observing instance metrics.
- No new CLI flags needed — `--snapshot-refresh-interval` now applies to all fields (not just KV).

**Confirmation:** No dead code. All paths exercisable via existing test patterns plus new tests.

### G) Task Breakdown

---

#### Task 1: Unify Snapshot Staleness Configuration (Phase 1)

**Contracts Implemented:** BC-1, BC-2, EC-1

**Files:**
- Modify: `sim/cluster/snapshot.go:39-48`
- Modify: `sim/cluster/snapshot_test.go`
- Modify: `sim/cluster/deployment.go:28` (comment update)

**Step 1: Write failing test for BC-1 (unified staleness)**

Context: We need to verify that when a refresh interval is set, ALL three fields become Periodic, not just KVUtilization.

In `sim/cluster/snapshot_test.go`, add:
```go
// TestNewObservabilityConfig_NonZeroInterval_AllFieldsPeriodic verifies BC-1:
// GIVEN a non-zero refresh interval
// WHEN newObservabilityConfig is called
// THEN all three fields (QueueDepth, BatchSize, KVUtilization) use Periodic mode
// with the same interval.
func TestNewObservabilityConfig_NonZeroInterval_AllFieldsPeriodic(t *testing.T) {
	config := newObservabilityConfig(5000) // 5ms

	fields := []struct {
		name string
		fc   FieldConfig
	}{
		{"QueueDepth", config.QueueDepth},
		{"BatchSize", config.BatchSize},
		{"KVUtilization", config.KVUtilization},
	}
	for _, f := range fields {
		t.Run(f.name, func(t *testing.T) {
			if f.fc.Mode != Periodic {
				t.Errorf("Mode = %d, want Periodic (%d)", f.fc.Mode, Periodic)
			}
			if f.fc.Interval != 5000 {
				t.Errorf("Interval = %d, want 5000", f.fc.Interval)
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestNewObservabilityConfig_NonZeroInterval -v`
Expected: FAIL — QueueDepth and BatchSize have Mode=Immediate, not Periodic

**Step 3: Implement unified staleness in newObservabilityConfig**

In `sim/cluster/snapshot.go`, replace the non-zero interval branch:

```go
// newObservabilityConfig creates an ObservabilityConfig based on the refresh interval.
// If interval is 0, all fields use Immediate mode (default behavior).
// If interval > 0, all fields use Periodic mode with the given interval,
// matching real vLLM Prometheus endpoint behavior where all gauges share the
// same scrape interval.
func newObservabilityConfig(refreshInterval int64) ObservabilityConfig {
	if refreshInterval <= 0 {
		return DefaultObservabilityConfig()
	}
	periodic := FieldConfig{Mode: Periodic, Interval: refreshInterval}
	return ObservabilityConfig{
		QueueDepth:    periodic,
		BatchSize:     periodic,
		KVUtilization: periodic,
	}
}
```

Update the `SnapshotRefreshInterval` comment in `sim/cluster/deployment.go`:
```go
	// Snapshot staleness configuration (H3 experiment, unified in #463)
	// When > 0, all Prometheus-sourced signals (QueueDepth, BatchSize, KVUtilization)
	// use Periodic refresh with this interval (microseconds). 0 = Immediate (default).
	SnapshotRefreshInterval int64
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestNewObservabilityConfig_NonZeroInterval -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/snapshot.go sim/cluster/snapshot_test.go sim/cluster/deployment.go
git commit -m "feat(snapshot): unify QueueDepth/BatchSize staleness with KVUtilization (BC-1, BC-2)

- Apply refresh interval to all three Prometheus-sourced signals uniformly
- When --snapshot-refresh-interval > 0, QueueDepth and BatchSize now use
  Periodic mode (matching real vLLM single-endpoint scrape behavior)
- Zero interval preserves existing Immediate behavior (backward compatible)

Refs: #463

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Rename PendingRequests → InFlightRequests in RoutingSnapshot

**Contracts Implemented:** BC-5

**Files:**
- Modify: `sim/routing.go` (RoutingSnapshot struct, EffectiveLoad, comments)
- Modify: `sim/cluster/cluster_event.go` (buildRouterState)
- Modify: `sim/cluster/cluster.go` (pendingRequests map → inFlightRequests)
- Modify: `sim/cluster/counterfactual.go` (PendingRequests references)
- Modify: `sim/trace/record.go` (CandidateScore.PendingRequests)
- Modify: `sim/routing_test.go` (PendingRequests → InFlightRequests in test struct literals)
- Modify: `sim/routing_scorers_test.go` (PendingRequests → InFlightRequests)
- Modify: `sim/routing_constructors_test.go` (PendingRequests → InFlightRequests)
- Modify: `sim/cluster/counterfactual_test.go` (PendingRequests → InFlightRequests)
- Modify: `sim/cluster/pending_requests_test.go` (PendingRequests → InFlightRequests)

**Step 1: Write failing test for BC-5 (EffectiveLoad with renamed field)**

Context: We need to ensure EffectiveLoad still works correctly with the renamed field.

In `sim/routing_test.go` (find an appropriate location or create a new test):
```go
// TestRoutingSnapshot_EffectiveLoad_IncludesInFlightRequests verifies BC-5:
// GIVEN a RoutingSnapshot with QueueDepth=2, BatchSize=1, InFlightRequests=3
// WHEN EffectiveLoad() is called
// THEN the result is 6
func TestRoutingSnapshot_EffectiveLoad_IncludesInFlightRequests(t *testing.T) {
	snap := RoutingSnapshot{
		ID:               "test",
		QueueDepth:       2,
		BatchSize:        1,
		InFlightRequests: 3,
	}
	if got := snap.EffectiveLoad(); got != 6 {
		t.Errorf("EffectiveLoad() = %d, want 6", got)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestRoutingSnapshot_EffectiveLoad_IncludesInFlightRequests -v`
Expected: FAIL — compilation error, `InFlightRequests` field does not exist

**Step 3: Rename PendingRequests → InFlightRequests across all files**

In `sim/routing.go`:
```go
type RoutingSnapshot struct {
	ID               string
	QueueDepth       int
	BatchSize        int
	KVUtilization    float64
	FreeKVBlocks     int64
	CacheHitRate     float64
	InFlightRequests int // Requests dispatched to this instance but not yet completed
}

// EffectiveLoad returns the total effective load on this instance:
// QueueDepth + BatchSize + InFlightRequests.
// Used by routing policies and counterfactual scoring for consistent load calculations.
func (s RoutingSnapshot) EffectiveLoad() int {
	return s.QueueDepth + s.BatchSize + s.InFlightRequests
}
```

In `sim/cluster/cluster.go`, rename the map field:
```go
	inFlightRequests       map[string]int // instance ID → dispatched-but-not-completed count (#463)
```
And all references: `pendingRequests` → `inFlightRequests` (in `NewClusterSimulator` init, event loop decrement, post-simulation check).

In `sim/cluster/cluster_event.go`, update `buildRouterState`:
```go
	snap.InFlightRequests = cs.inFlightRequests[string(inst.ID())]
```

And `RoutingDecisionEvent.Execute`:
```go
	cs.inFlightRequests[decision.TargetInstance]++
```

In `sim/cluster/counterfactual.go`, update ALL `PendingRequests` references to `InFlightRequests`, including the doc comment at line 29: `-(QueueDepth + BatchSize + InFlightRequests)`.

In `sim/trace/record.go`, rename `PendingRequests` → `InFlightRequests` in `CandidateScore`. Note: this changes the implicit JSON key from `"PendingRequests"` to `"InFlightRequests"` (no explicit JSON tags exist). This is a trace schema breaking change — documented in Deviation Log.

In `sim/routing.go`, also update doc comments on `LeastLoaded` (lines 102-104) and `AlwaysBusiest` (line 200) which reference `PendingRequests` in their type-level comments.

In `sim/cluster/snapshot.go`, update the comment at line 92-93 from `PendingRequests` to `InFlightRequests` and update the semantics: "dispatched but not yet completed" (not "not yet in queue").

Rename the test file as part of this task (before Task 3 adds new tests):
```bash
git mv sim/cluster/pending_requests_test.go sim/cluster/inflight_requests_test.go
```
Update all references in the renamed file: `pendingRequests` → `inFlightRequests`, `PendingRequests` → `InFlightRequests`, test function names, comments.

**Behavioral rewrites required (not just renames):**

**CausalDecrement** — The old test verified decrement on QueuedEvent. Remove it entirely in Task 2 (it tests old semantics). The replacement test `TestClusterSimulator_InFlightRequests_CompletionBasedDecrement` proving the NEW causal property is added in **Task 3 Step 3c** (after the completion-based decrement is implemented, so the test can verify the new behavior):

The following test code is documented here for reference but is IMPLEMENTED in Task 3 Step 3b (after the completion-based decrement logic is in place):

```go
// TestClusterSimulator_InFlightRequests_CompletionBasedDecrement verifies BC-3:
// GIVEN requests routed with routing latency
// WHEN a request enters the queue (QueuedEvent fires)
// THEN InFlightRequests does NOT decrement (stays elevated)
// AND InFlightRequests decrements only when the request completes
func TestClusterSimulator_InFlightRequests_CompletionBasedDecrement(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:    1,
		RoutingLatency:  100,
		TraceLevel:      "decisions",
		CounterfactualK: 1,
	}
	reqs := make([]*sim.Request, 4)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("causal_req_%d", i),
			ArrivalTime:  0,
			InputTokens:  make([]int, 16),
			OutputTokens: make([]int, 8),
			State:        sim.StateQueued,
		}
	}
	cs := NewClusterSimulator(config, reqs)
	mustRun(t, cs)

	// Key assertion: at least one routing decision sees InFlightRequests > QueueDepth + BatchSize
	// This proves the in-flight window extends beyond queue absorption
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	foundElevated := false
	for _, r := range tr.Routings {
		for _, c := range r.Candidates {
			if c.InFlightRequests > c.QueueDepth+c.BatchSize {
				foundElevated = true
				break
			}
		}
	}
	if !foundElevated {
		t.Error("expected at least one routing decision where InFlightRequests > QueueDepth + BatchSize, " +
			"proving the in-flight window extends beyond queue absorption")
	}

	// Post-simulation: must drain per BC-4
	for instID, inflight := range cs.inFlightRequests {
		if inflight != 0 {
			t.Errorf("instance %s: inFlightRequests = %d, want 0", instID, inflight)
		}
	}
}
```

**VisibleInRoutingState and CounterfactualIncludesPending** — Under the new completion-based semantics, `InFlightRequests > 0` is trivially true for any routed request not yet completed. The old assertion (`PendingRequests > 0` during routing) becomes a tautology. Update these tests to assert the stronger property: `InFlightRequests > QueueDepth + BatchSize` for at least one routing decision, proving the dispatch-to-completion window is wider than the dispatch-to-queue window. The `TestComputeCounterfactual_IncludesPendingRequests` unit test only needs a mechanical rename (it tests snapshot field propagation, not lifecycle semantics).

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestRoutingSnapshot_EffectiveLoad_IncludesInFlightRequests -v`
Expected: PASS

Then run full test suite to catch any compilation errors from the rename:
Run: `go test ./...`
Expected: PASS (some test files may need updating — see Step 3 continuation)

**Step 5: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/routing.go sim/routing_test.go sim/routing_scorers_test.go sim/routing_constructors_test.go \
  sim/cluster/cluster_event.go sim/cluster/cluster.go sim/cluster/counterfactual.go \
  sim/cluster/counterfactual_test.go sim/cluster/snapshot.go sim/trace/record.go \
  sim/cluster/inflight_requests_test.go
git commit -m "refactor(routing): rename PendingRequests → InFlightRequests (BC-5)

- Rename RoutingSnapshot.PendingRequests → InFlightRequests
- Rename ClusterSimulator.pendingRequests → inFlightRequests
- Rename CandidateScore.PendingRequests → InFlightRequests
- Update EffectiveLoad() comment
- All references updated across sim/, sim/cluster/, sim/trace/

Refs: #463

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Change InFlightRequests Decrement to Completion-Based

**Contracts Implemented:** BC-3, BC-4, BC-7, NC-2

**Files:**
- Modify: `sim/cluster/cluster.go` (event loop decrement logic)
- Modify: `sim/cluster/inflight_requests_test.go` (already renamed in Task 2)

**Step 1: Write failing test for BC-3 (completion-based decrement)**

Context: The key behavioral difference: InFlightRequests should remain elevated while a request is queued and running, not just while it's "pending" in the routing pipeline. We test that InFlightRequests is still > 0 when QueuedEvent fires (it should NOT decrement on queue absorption anymore).

**Note:** Add `"fmt"` to the import block of `inflight_requests_test.go` — needed by the BC-7 and BC-3 tests below that use `fmt.Sprintf`.

In `sim/cluster/inflight_requests_test.go` (renamed from pending_requests_test.go):

```go
// TestClusterSimulator_InFlightRequests_DrainsToZeroAfterCompletion verifies BC-4:
// GIVEN a cluster with requests that all complete before horizon
// WHEN simulation ends
// THEN all inFlightRequests values are 0
func TestClusterSimulator_InFlightRequests_DrainsToZeroAfterCompletion(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:         2,
		RoutingPolicy:        "weighted",
		RoutingScorerConfigs: sim.DefaultScorerConfigs(),
	}
	requests := testGenerateRequests(42, 10000000, 2.0/1e6, 6,
		0, 16, 0, 16, 16, 8, 0, 8, 8)
	cs := NewClusterSimulator(config, requests)

	mustRun(t, cs)

	for instID, inflight := range cs.inFlightRequests {
		if inflight != 0 {
			t.Errorf("instance %s: inFlightRequests = %d after completion, want 0", instID, inflight)
		}
	}

	m := cs.AggregatedMetrics()
	if m.CompletedRequests == 0 {
		t.Error("no requests completed — test setup issue")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_InFlightRequests_DrainsToZeroAfterCompletion -v`
Expected: May pass or fail depending on whether requests complete — this is a drain-to-zero test. If it passes, that's fine; the real behavioral test is that the decrement trigger changed.

**Step 3: Implement completion-based decrement**

In `sim/cluster/cluster.go`, replace the `QueuedEvent`-based decrement with completion-based decrement in the instance event processing section:

```go
		} else {
			c.clock = instanceTime
			if c.clock > c.config.Horizon {
				break
			}
			inst := c.instances[instanceIdx]
			instID := string(inst.ID())

			// Snapshot counters BEFORE processing the event
			completedBefore := inst.Metrics().CompletedRequests
			droppedBefore := inst.Metrics().DroppedUnservable

			ev := inst.ProcessNextEvent()
			_ = ev // Event type no longer used for decrement

			// Completion-based decrement (#463, BC-3, BC-7): InFlightRequests tracks the full
			// dispatch-to-completion window. Decrement by the number of newly completed OR
			// dropped-unservable requests. DroppedUnservable requests never reach CompletedRequests
			// but still exit the in-flight window (they were rejected during EnqueueRequest).
			completedAfter := inst.Metrics().CompletedRequests
			droppedAfter := inst.Metrics().DroppedUnservable
			delta := (completedAfter - completedBefore) + (droppedAfter - droppedBefore)
			if delta > 0 {
				c.inFlightRequests[instID] -= delta
				if c.inFlightRequests[instID] < 0 {
					logrus.Warnf("inFlightRequests[%s] went negative (%d) — bookkeeping bug", instID, c.inFlightRequests[instID])
					c.inFlightRequests[instID] = 0
				}
			}
		}
```

Remove the old `QueuedEvent` type-switch decrement logic entirely.

Also update the post-simulation invariant check. **IMPORTANT: Move this check AFTER the Finalize loop** (currently at cluster.go:184-187). `StillQueued` and `StillRunning` are only populated by `Finalize()` — reading them before Finalize returns zero for both fields, making the check degenerate to `inflight != 0`. The check must execute between `c.aggregatedMetrics = c.aggregateMetrics()` (line 188) and the diagnostic warnings (line 190):
```go
	// Post-simulation invariant: inFlightRequests should match StillQueued + StillRunning
	for _, inst := range c.instances {
		instID := string(inst.ID())
		inflight := c.inFlightRequests[instID]
		m := inst.Metrics()
		expectedInFlight := m.StillQueued + m.StillRunning
		if inflight != expectedInFlight {
			logrus.Warnf("post-simulation: inFlightRequests[%s] = %d, expected %d (StillQueued=%d + StillRunning=%d) — bookkeeping bug",
				instID, inflight, expectedInFlight, m.StillQueued, m.StillRunning)
		}
	}
```

**Step 3b: Write BC-7 test (DroppedUnservable decrements InFlightRequests)**

Context: When a request is dropped as unservable (KV blocks too small), InFlightRequests must still decrement.

```go
// TestClusterSimulator_InFlightRequests_DroppedUnservable_Decrements verifies BC-7:
// GIVEN a cluster where TotalKVBlocks is too small for some requests
// WHEN requests are routed and some are dropped as unservable
// THEN inFlightRequests drains correctly (accounting for drops)
func TestClusterSimulator_InFlightRequests_DroppedUnservable_Decrements(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(5, 16, 0, 0, 0, 0), // Very small — will force drops
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances: 1,
	}
	// Requests with large input that exceeds KV capacity
	reqs := make([]*sim.Request, 3)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("drop_req_%d", i),
			ArrivalTime:  int64(i * 1000),
			InputTokens:  make([]int, 200), // 200 tokens / 16 block_size = 13 blocks > 5 total
			OutputTokens: make([]int, 8),
			State:        sim.StateQueued,
		}
	}
	cs := NewClusterSimulator(config, reqs)
	mustRun(t, cs)

	// All requests should be dropped as unservable
	m := cs.AggregatedMetrics()
	if m.DroppedUnservable == 0 {
		t.Fatal("expected DroppedUnservable > 0 — test setup issue (KV blocks too large)")
	}

	// InFlightRequests must still drain to match expected
	for _, inst := range cs.Instances() {
		instID := string(inst.ID())
		inflight := cs.inFlightRequests[instID]
		im := inst.Metrics()
		expected := im.StillQueued + im.StillRunning
		if inflight != expected {
			t.Errorf("instance %s: inFlightRequests=%d, expected %d (StillQueued=%d + StillRunning=%d)",
				instID, inflight, expected, im.StillQueued, im.StillRunning)
		}
	}
}
```

**Step 3c: Add CompletionBasedDecrement test (BC-3)**

Add `TestClusterSimulator_InFlightRequests_CompletionBasedDecrement` from the code block in Task 2 (documented there for reference, implemented here). This test verifies the core behavioral change: InFlightRequests stays elevated after queue absorption and decrements only on completion.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_InFlightRequests -v`
Expected: PASS

Run full test suite: `go test ./...`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/inflight_requests_test.go
git commit -m "feat(cluster): change InFlightRequests to completion-based decrement (BC-3, BC-4)

- InFlightRequests now tracks dispatch-to-completion window (not dispatch-to-queue)
- Decrement based on CompletedRequests counter delta after each instance event
- Matches real HTTP router in-flight tracking (dispatched→response received)
- Remove QueuedEvent-based decrement (old microsecond-scale window)

Refs: #463

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Update Documentation (INV-7, CLAUDE.md, Scorer Comments)

**Contracts Implemented:** BC-6

**Files:**
- Modify: `docs/contributing/standards/invariants.md` (INV-7 table)
- Modify: `CLAUDE.md` (INV-7 description, snapshot refresh description)
- Modify: `sim/routing_scorers.go` (R17 doc comments)
- Modify: `sim/routing_prefix_scorer.go` (R17 doc comment)
- Modify: `docs/guide/routing.md` (PendingRequests → InFlightRequests, update semantics description)
- Modify: `docs/concepts/architecture.md` (4 PendingRequests references + decrement semantics)
- Modify: `docs/concepts/glossary.md` (PendingRequests definition → InFlightRequests)
- Modify: `docs/contributing/extension-recipes.md` (EffectiveLoad description)
- Modify: `docs/contributing/standards/rules.md` (R17 freshness hierarchy)
- Modify: `examples/routing-comparison.sh` (PendingRequests comment → InFlightRequests)
- Modify: `.claude/skills/hypothesis-experiment/review-prompts.md` (INV-7 PendingRequests reference)
- Modify: `docs/concepts/diagrams/cluster-data-flow.excalidraw` (PendingRequests text in diagram JSON)
- Modify: `cmd/root.go:780` (update `--snapshot-refresh-interval` help text from "KV utilization snapshot refresh" to "Prometheus snapshot refresh interval for all instance metrics")

For each documentation file: grep for `PendingRequests`, replace with `InFlightRequests`, and update any semantic descriptions of "routed but not yet in queue" to "dispatched but not yet completed".

**Step 1: Update INV-7 signal freshness table**

In `docs/contributing/standards/invariants.md`, replace the INV-7 table:

```markdown
## INV-7: Signal Freshness Hierarchy

**Statement:** Routing snapshot signals have tiered freshness due to DES event ordering and configurable staleness.

| Signal | Owner | Freshness (interval=0) | Freshness (interval>0) | Updated By |
|--------|-------|------------------------|------------------------|------------|
| InFlightRequests | Cluster | Synchronous | Synchronous | `RoutingDecisionEvent.Execute()` (increment), completion detection (decrement) |
| QueueDepth | Instance | Immediate | Periodic | `QueuedEvent.Execute()` |
| BatchSize | Instance | Immediate | Periodic | `StepEvent.Execute()` |
| KVUtilization | Instance | Immediate | Periodic | `FormBatch()` → `AllocateKVBlocks()` |
| CacheHitRate | Instance | Immediate | Periodic | `FormBatch()` |

**Design implication:** When `--snapshot-refresh-interval > 0`, all Prometheus-sourced signals (QueueDepth, BatchSize, KVUtilization) share the same scrape interval — matching real vLLM deployments where all three are exposed via the same `/metrics` endpoint. `InFlightRequests` remains synchronous (gateway-local counter, not Prometheus-sourced).

`EffectiveLoad()` = `QueueDepth + BatchSize + InFlightRequests`. The synchronous `InFlightRequests` term compensates for Periodic staleness in the other two terms.
```

**Step 2: Update CLAUDE.md INV-7 description**

Replace the INV-7 line:
```markdown
- **INV-7 Signal freshness**: Routing snapshot signals have tiered freshness — InFlightRequests (synchronous) vs QueueDepth/BatchSize/KVUtilization (Periodic when `--snapshot-refresh-interval > 0`, Immediate when 0). See `docs/contributing/standards/invariants.md` for the full hierarchy.
```

Update the `--snapshot-refresh-interval` entry in the CLI flags section:
```markdown
--snapshot-refresh-interval  # Prometheus scrape interval (μs) for all instance metrics (QueueDepth, BatchSize, KVUtilization). 0 = Immediate.
```

**Step 3: Update R17 doc comments in scorers**

In `sim/routing_scorers.go`, update `scoreQueueDepth`:
```go
// Signal freshness (R17, INV-7):
//
//	Reads: EffectiveLoad() = QueueDepth (Periodic when interval>0, else Immediate) +
//	       BatchSize (same) + InFlightRequests (synchronous).
//	The synchronous InFlightRequests term compensates for Periodic staleness.
```

Update `scoreKVUtilization`:
```go
// Signal freshness (R17, INV-7):
//
//	Reads: KVUtilization (Periodic when interval>0, else Immediate).
//	WARNING: At high request rates with large intervals, this signal can be significantly stale.
//	Pair with a load-aware scorer (e.g., queue-depth) for robust routing.
```

Update `scoreLoadBalance` similarly.

In `sim/routing_prefix_scorer.go`, update the R17 comment if it references PendingRequests.

**Step 4: Run full test suite**

Run: `go test ./...`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add docs/contributing/standards/invariants.md CLAUDE.md sim/routing_scorers.go sim/routing_prefix_scorer.go \
  docs/guide/routing.md docs/concepts/architecture.md docs/concepts/glossary.md \
  docs/contributing/extension-recipes.md docs/contributing/standards/rules.md \
  examples/routing-comparison.sh .claude/skills/hypothesis-experiment/review-prompts.md \
  docs/concepts/diagrams/cluster-data-flow.excalidraw cmd/root.go
git commit -m "docs: update INV-7 for unified staleness and InFlightRequests (BC-6)

- INV-7 table now shows per-interval freshness for all signals
- PendingRequests → InFlightRequests throughout documentation
- R17 scorer comments updated for new staleness model
- CLAUDE.md updated for --snapshot-refresh-interval description

Refs: #463

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Golden Dataset Regeneration and Integration Verification

**Contracts Implemented:** NC-1b (intentional behavioral change), R12

**Files:**
- Modify: `testdata/goldendataset.json` (WILL change — InFlightRequests widens the load window, changing routing decisions for load-aware policies)

**Step 1: Run golden dataset tests — expect failure**

Context: NC-1b acknowledges that Phase 2 (completion-based decrement) changes `EffectiveLoad()` values during routing, which changes routing decisions for load-aware policies. The golden dataset WILL need regeneration (R12).

Run: `go test ./sim/... -run TestGolden -v`
Expected: FAIL — golden values no longer match due to changed routing behavior

**Step 2: Regenerate golden dataset**

Run the regeneration command (check test code for the exact flag):
```bash
go test ./sim/internal/testutil/... -run TestRegenerate -update -v
# OR the appropriate regeneration command per the test infrastructure
```

**Step 3: Verify invariant tests still pass alongside new golden values**

Run: `go test ./... -count=1`
Expected: All packages PASS — invariant tests (conservation, causality, determinism) must pass with the new golden values. If invariant tests fail, the behavioral change has a bug.

**Step 4: Run full lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit golden dataset regeneration**

```bash
git add testdata/goldendataset.json
git commit -m "test: regenerate golden dataset for InFlightRequests semantics (R12, NC-1b)

- Golden values change because InFlightRequests now tracks dispatch-to-completion
  window, increasing EffectiveLoad() during routing → different routing decisions
- All invariant tests (conservation, causality, determinism) verified passing
- This is an intentional fidelity improvement, not a regression

Refs: #463

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|---|---|---|---|
| BC-1 | Task 1 | Unit | `TestNewObservabilityConfig_NonZeroInterval_AllFieldsPeriodic` |
| BC-2 | Task 1 | Unit | `TestSnapshotProvider_DefaultConfig_AllImmediate` (existing, unchanged) |
| BC-3 | Task 3 | Integration | `TestClusterSimulator_InFlightRequests_CompletionBasedDecrement` (Step 3c) |
| BC-4 | Task 3 | Integration | `TestClusterSimulator_InFlightRequests_DrainsToZeroAfterCompletion` + post-simulation invariant check |
| BC-5 | Task 2 | Unit | `TestRoutingSnapshot_EffectiveLoad_IncludesInFlightRequests` |
| BC-6 | Task 4 | Manual | Documentation review |
| BC-7 | Task 3 | Integration | `TestClusterSimulator_InFlightRequests_DroppedUnservable_Decrements` |
| BC-8 | Task 4 | Docs | EffectiveLoad double-counting documented in INV-7 design implication |
| NC-1a | Task 1 | Unit | `TestNewObservabilityConfig` tests verify zero-interval backward compat |
| NC-1b | Task 5 | Golden | Golden dataset regenerated (R12) — intentional behavioral change |
| NC-2 | Task 3 | Integration | Guard in decrement logic + post-simulation invariant check |
| EC-1 | Task 1 | Unit | `TestNewObservabilityConfig_NonZeroInterval_AllFieldsPeriodic` covers positive; existing `DefaultConfig` test covers zero/negative |

**Golden dataset:** WILL change (NC-1b). The wider InFlightRequests window changes EffectiveLoad() values during routing, which changes routing decisions for load-aware policies. Regeneration per R12 is required. Invariant tests must pass with new golden values to verify the change is a fidelity improvement, not a regression.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|---|---|---|---|---|
| Completion counter delta misses concurrent completions | Low | Medium | `CompletedRequests` is a simple counter incremented atomically per completion. Delta approach is correct for single-threaded DES. | Task 3 |
| InFlightRequests doesn't drain to zero at horizon | Medium | High | Post-simulation invariant check (existing pattern). Requests not completed by horizon remain in-flight — this is expected behavior, not a bug. Warn only if requests DID complete but counter is non-zero. | Task 3 |
| Golden dataset changes (expected) | Certain | Low | Golden dataset WILL change due to wider InFlightRequests window. Regenerate per R12. Verify invariant tests pass with new values. | Task 5 |
| DroppedUnservable leaks InFlightRequests | Medium | High | Track DroppedUnservable delta alongside CompletedRequests delta. Test with low KV blocks. | Task 3 |
| EffectiveLoad magnitude change may need scorer weight retuning | Medium | Medium | EffectiveLoad ≈ 2×(QD+BS) due to double-counting (BC-8). Relative ranking preserved for min-max normalization. Phase 3 re-validation (H29/H4) should test scorer sensitivity. | Deferred (Phase 3) |
| Rename breaks downstream references (hypotheses, docs) | Low | Low | Grep for all PendingRequests references. Hypothesis FINDINGS.md files are historical records — don't rename in them. Only update living documentation. | Task 2, 4 |
| CandidateScore JSON field name change | Certain | Low | Trace schema breaking change — `PendingRequests` JSON key → `InFlightRequests`. No explicit JSON tags exist. Acceptable for pre-1.0 project. | Task 2 |

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (Phase 3 deferred)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates (rename is documented)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (testGenerateRequests, mustRun)
- [x] CLAUDE.md updated (Task 4)
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: INV-7 canonical source updated, CLAUDE.md working copy updated
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1 → 2 → 3 → 4 → 5)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration documented (WILL be needed per NC-1b — Task 5)
- [x] Construction site audit: `RoutingSnapshot` literal construction sites — `NewRoutingSnapshot()` is canonical constructor, plus test-only struct literals in snapshot_test.go, pending_requests_test.go, counterfactual_test.go
- [x] R1: No silent continue/return
- [x] R2: No new map iteration for ordered output
- [x] R3: No new CLI flags (reusing existing --snapshot-refresh-interval)
- [x] R4: RoutingSnapshot rename — all construction sites audited
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: Golden tests already have invariant companions
- [x] R14: No methods spanning multiple concerns
- [x] R17: Signal freshness docs updated for all scorers

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/snapshot.go`

**Purpose:** Change `newObservabilityConfig` to apply refresh interval uniformly.

**Changes:** Lines 39-48 only. Replace the three-field return with a single `periodic` variable applied to all fields.

### File: `sim/routing.go`

**Purpose:** Rename `PendingRequests` → `InFlightRequests` in the `RoutingSnapshot` struct and `EffectiveLoad()` method.

**Construction sites for RoutingSnapshot:**
- `sim/routing.go:32-36` — `NewRoutingSnapshot()` canonical constructor (no PendingRequests field set)
- `sim/cluster/snapshot.go:124` — `RefreshAll()` uses `NewRoutingSnapshot()`
- `sim/cluster/snapshot_test.go` — test struct literals
- `sim/cluster/pending_requests_test.go` — test struct literals
- `sim/cluster/counterfactual.go` — reads PendingRequests for candidate scoring
- `sim/trace/record.go` — `CandidateScore.PendingRequests`

### File: `sim/cluster/cluster.go`

**Purpose:** Rename `pendingRequests` → `inFlightRequests`, change decrement from QueuedEvent-based to completion-based.

**Key behavioral change:** The decrement logic moves from type-switching on `*sim.QueuedEvent` to computing `CompletedRequests` delta. This widens the InFlightRequests window from microseconds (route→queue) to the full request lifecycle (route→complete).

### File: `sim/cluster/cluster_event.go`

**Purpose:** Rename `pendingRequests` → `inFlightRequests` in `buildRouterState()` and `RoutingDecisionEvent.Execute()`.

### File: `sim/trace/record.go`

**Purpose:** Rename `PendingRequests` → `InFlightRequests` in `CandidateScore` struct. No explicit JSON tags exist — Go's default marshaling uses the field name as JSON key. Renaming changes the JSON output key (trace schema breaking change, documented in Deviation Log).

### File: `docs/contributing/standards/invariants.md`

**Purpose:** Update INV-7 signal freshness table to show per-interval freshness tiers and InFlightRequests.
