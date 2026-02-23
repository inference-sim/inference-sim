# Fix #373: KV Livelock Circuit Breaker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent the simulator from livelocking when a request's input tokens require more KV cache blocks than the total cache capacity.

**The problem today:** When a request needs more KV blocks than exist (e.g., 57 blocks for a 912-token input but only 50 total blocks), the simulator enters an infinite loop: batch formation fails to allocate → request stays in queue → work-conserving invariant schedules another step → repeat forever. The simulator hangs at 100% CPU with no progress. In real vLLM, such requests are rejected before entering the engine.

**What this PR adds:**
1. **Early rejection in EnqueueRequest** — before a request enters the wait queue, check if its input tokens require more KV blocks than total cache capacity. If so, log a warning, increment a counter, and discard the request (mirroring vLLM's pre-engine validation).
2. **DroppedUnservable metric** — a new counter in `Metrics` and `MetricsOutput` tracking how many requests were dropped because they exceeded KV capacity. Surfaced in JSON output and CLI anomaly counters.
3. **Conservation invariant update** — `InjectedRequests` in JSON output includes dropped requests: `injected = completed + queued + running + dropped_unservable`. The full pipeline: `num_requests = injected + rejected`.

**Why this matters:** This is a correctness bug (R19 violation) that causes the simulator to hang indefinitely on certain valid configurations. Users running capacity planning sweeps with varied KV block sizes can hit this silently.

**Architecture:** The fix touches `sim/simulator.go` (EnqueueRequest check), `sim/metrics.go` (new counter field), `sim/metrics_utils.go` (JSON output field), `sim/cluster/cluster.go` (aggregation), `sim/cluster/metrics.go` (RawMetrics field), and `cmd/root.go` (CLI output). All changes are within existing module boundaries — no new interfaces, types, or packages.

**Source:** GitHub issue #373

**Closes:** Fixes #373

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds a circuit breaker to prevent KV cache livelock. When a request arrives whose input tokens require more KV blocks than the total cache capacity, the request is dropped at enqueue time — before it enters the wait queue. This mirrors real vLLM behavior where oversized requests are rejected before reaching the engine.

The fix is in the request injection path (`EnqueueRequest`), which is the entry point for all requests in both single-instance and cluster modes. A new `DroppedUnservable` counter in `Metrics` tracks dropped requests for observability and conservation accounting.

Adjacent components: KVStore (queried for capacity), Metrics (new counter), MetricsOutput (new JSON field), cluster aggregation (sums counter across instances), CLI output (prints counter in anomaly section).

Deviations from the issue description documented in Section D below (fix location changed from FormBatch to EnqueueRequest; per-request check instead of workload-level).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Oversized request dropped at enqueue
- GIVEN a request whose input tokens require `ceil(len(InputTokens) / BlockSize)` blocks exceeding `TotalCapacity()`
- WHEN `EnqueueRequest` is called
- THEN the request MUST NOT appear in the wait queue AND `DroppedUnservable` counter MUST increment by 1
- MECHANISM: Block count check in `EnqueueRequest` before `WaitQ.Enqueue()`

BC-2: Normal requests unaffected
- GIVEN a request whose input tokens fit within KV cache capacity
- WHEN `EnqueueRequest` is called
- THEN the request MUST appear in the wait queue AND `DroppedUnservable` MUST remain unchanged
- MECHANISM: Block count check passes, existing enqueue path runs

BC-3: Simulation terminates with oversized requests
- GIVEN a workload where some requests exceed KV capacity
- WHEN the simulation runs to completion
- THEN the simulation MUST terminate (no hang) AND `DroppedUnservable` MUST equal the count of oversized requests
- MECHANISM: Oversized requests never enter WaitQ, so livelock cycle cannot form

BC-4: DroppedUnservable appears in JSON output
- GIVEN a simulation with dropped requests
- WHEN metrics are saved
- THEN the JSON output MUST include `"dropped_unservable": N` where N > 0

BC-5: Conservation with dropped requests
- GIVEN a simulation where D requests are dropped
- WHEN metrics are finalized
- THEN `InjectedRequests` in JSON output MUST equal `completed + still_queued + still_running + dropped_unservable`
- MECHANISM: `InjectedRequests` formula updated in `SaveResults()` to include `DroppedUnservable`

**Negative Contracts:**

BC-6: No livelock on oversized requests
- GIVEN a configuration where all requests exceed KV capacity
- WHEN the simulation runs
- THEN it MUST terminate within the horizon (no infinite loop)

BC-7: Dropped requests removed from per-request tracking
- GIVEN an oversized request that is dropped
- WHEN metrics are saved
- THEN the dropped request MUST NOT appear in the per-request `requests` array in JSON output
- MECHANISM: `delete(sim.Metrics.Requests, r.ID)` after drop

**Error Handling Contracts:**

BC-8: Warning logged for each dropped request
- GIVEN an oversized request
- WHEN it is dropped at enqueue
- THEN a logrus warning MUST be emitted with the request ID, blocks needed, and total capacity

### C) Component Interaction

```
Request arrives → ArrivalEvent → QueuedEvent → EnqueueRequest()
                                                    │
                                              ┌─────┴─────┐
                                              │ blocks >   │
                                              │ capacity?  │
                                              └─────┬─────┘
                                               yes/    \no
                                              /          \
                                   log warning      WaitQ.Enqueue()
                                   increment        TotalInputTokens++
                                   DroppedUnservable
                                   delete Requests[ID]
                                   return
```

**API Contracts:**
- `EnqueueRequest(r *Request)` — no signature change; behavior change: may now decline to enqueue
- `Metrics.DroppedUnservable int` — new field, zero-value safe (`int` to match request counter pattern: CompletedRequests, StillQueued, StillRunning)
- `MetricsOutput.DroppedUnservable int` — new JSON field `"dropped_unservable"` (no omitempty, `int` to match request counter pattern)
- `RawMetrics.DroppedUnservable int` — new field for cluster-level aggregation

**State Changes:**
- `Metrics.DroppedUnservable` — owned by Simulator, incremented in `EnqueueRequest`, aggregated in `ClusterSimulator.aggregateMetrics()`
- `Metrics.Requests` — entries deleted for dropped requests (owned by Simulator)

**Cluster pending-request interaction:** In cluster mode, `pendingRequests[instID]++` happens at routing time. The decrement happens when the instance processes the request (QueuedEvent fires → EnqueueRequest runs). Whether the request is enqueued or dropped, it is no longer "pending at the control plane." The decrement is correct for both paths — no special handling needed.

**Why input-only check is sufficient:** The check computes `ceil(len(InputTokens) / BlockSize)`. During decode, output tokens also consume blocks, but the existing `preemptForTokens` circuit breaker (`batch_formation.go:148-153`) already handles the case where a running request's decode allocation fails with an empty batch. The livelock only occurs in Phase 2 (new request dequeue from WaitQ), where the initial prefill allocation is attempted. If the full input can't fit, the request can never start prefill.

**Extension Friction:**
- Adding one field to `Metrics`: 5 files (metrics.go, metrics_utils.go, cluster.go, cluster/metrics.go, cmd/root.go). This is the existing pattern for all metric counters (e.g., `PreemptionCount` touches the same 5 files).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Add pre-check in FormBatch or NewSimulator" | Check in EnqueueRequest instead | CORRECTION: User clarified that vLLM rejects before engine entry. EnqueueRequest is the earliest point with KVCache access. FormBatch defense-in-depth unnecessary since token counts are immutable after creation. |
| "Compare total_kv_blocks against max possible input tokens in workload" | Compare per-request, not against workload max | SIMPLIFICATION: Per-request check is more precise and handles heterogeneous workloads. Workload-level check would over-reject. |

### E) Review Guide

**The tricky part:** The conservation accounting. Dropped requests are registered in `Metrics.Requests` by `InjectArrival()` before `EnqueueRequest()` runs. We must `delete` them from `Metrics.Requests` to avoid phantom entries in JSON output. Verify BC-7 test covers this.

**What to scrutinize:** BC-5 conservation test — does the arithmetic actually close? The formula changes from `injected = completed + queued + running` to needing `dropped_unservable` for the full pipeline count.

**What's safe to skim:** MetricsOutput field addition, cluster aggregation, CLI print — all mechanical.

**Known debt:** None introduced. The existing `preemptForTokens` Phase 1 circuit breaker (batch_formation.go:148-153) is a separate defense for a different scenario (continuing requests in a running batch).

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/metrics.go` — add `DroppedUnservable` field to `Metrics`
- `sim/metrics_utils.go` — add `DroppedUnservable` to `MetricsOutput` JSON struct
- `sim/simulator.go` — add capacity check in `EnqueueRequest()`
- `sim/cluster/cluster.go` — aggregate `DroppedUnservable` across instances
- `sim/cluster/metrics.go` — add `DroppedUnservable` to `RawMetrics`, pass to CLI
- `cmd/root.go` — print `DroppedUnservable` in anomaly counters section

**Files to create:**
- None

**Test files to modify:**
- `sim/simulator_test.go` — add tests for BC-1, BC-2, BC-3, BC-5, BC-6, BC-7, BC-8
- `sim/cluster/metrics_test.go` — add test for aggregation

**Key decisions:**
- Check in `EnqueueRequest()` not `FormBatch()` — mirrors vLLM pre-engine rejection
- Per-request check, not workload-level — handles heterogeneous request sizes
- Delete from `Metrics.Requests` — prevents phantom entries in JSON output

### G) Task Breakdown

---

### Task 1: Add DroppedUnservable to Metrics and MetricsOutput

**Contracts Implemented:** BC-4 (partial — field exists but not yet populated)

**Files:**
- Modify: `sim/metrics.go:18-30` (Metrics struct)
- Modify: `sim/metrics.go:63-78` (SaveResults — MetricsOutput population)
- Modify: `sim/metrics_utils.go:49-76` (MetricsOutput struct)
- Modify: `sim/cluster/cluster.go:294-341` (aggregateMetrics)
- Modify: `sim/cluster/metrics.go:88-109` (RawMetrics struct)
- Modify: `sim/cluster/metrics.go:117-120` (CollectRawMetrics)
- Modify: `cmd/root.go:508-514` (CLI anomaly counters)

**Step 1: Add DroppedUnservable field to Metrics struct**

In `sim/metrics.go`, add to Metrics struct after `StillRunning`:
```go
DroppedUnservable int // Requests dropped because input tokens exceed KV cache capacity (R19)
```

**Step 2: Add DroppedUnservable to MetricsOutput struct**

In `sim/metrics_utils.go`, add to MetricsOutput struct after `PreemptionCount`:
```go
DroppedUnservable int `json:"dropped_unservable"`
```

**Step 3: Populate DroppedUnservable in SaveResults and update InjectedRequests**

In `sim/metrics.go` `SaveResults()`, after the line `PreemptionCount: m.PreemptionCount,`:
```go
DroppedUnservable: m.DroppedUnservable,
```

Also update the `InjectedRequests` formula (line ~72) to include dropped requests:
```go
InjectedRequests: m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable,
```

This ensures `InjectedRequests` counts ALL requests that entered `EnqueueRequest`, whether they were enqueued or dropped. The full pipeline conservation is: `num_requests = injected + rejected` (rejected at cluster admission level).

**Step 4: Aggregate DroppedUnservable in cluster**

In `sim/cluster/cluster.go` `aggregateMetrics()`, after `merged.KVAllocationFailures += m.KVAllocationFailures`:
```go
merged.DroppedUnservable += m.DroppedUnservable
```

**Step 5: Add to RawMetrics and CollectRawMetrics**

In `sim/cluster/metrics.go`, add to `RawMetrics` struct after `RejectedRequests`:
```go
DroppedUnservable int
```

In `CollectRawMetrics()`, after `RejectedRequests: rejectedRequests,`:
```go
DroppedUnservable: aggregated.DroppedUnservable,
```

**Step 6: Update CLI anomaly counters**

In `cmd/root.go`, update the anomaly counters condition and output:
```go
if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.DroppedUnservable > 0 {
    fmt.Println("=== Anomaly Counters ===")
    fmt.Printf("Priority Inversions: %d\n", rawMetrics.PriorityInversions)
    fmt.Printf("HOL Blocking Events: %d\n", rawMetrics.HOLBlockingEvents)
    fmt.Printf("Rejected Requests: %d\n", rawMetrics.RejectedRequests)
    fmt.Printf("Dropped Unservable: %d\n", rawMetrics.DroppedUnservable)
}
```

**Step 7: Run build to verify**

Run: `go build ./...`
Expected: PASS (no compilation errors)

**Step 8: Run lint check**

Run: `golangci-lint run ./sim/... ./cmd/...`
Expected: No new issues

**Step 9: Commit**

```bash
git add sim/metrics.go sim/metrics_utils.go sim/cluster/cluster.go sim/cluster/metrics.go cmd/root.go
git commit -m "fix(sim): add DroppedUnservable metric field (#373)

- Add DroppedUnservable counter to Metrics struct
- Add dropped_unservable to MetricsOutput JSON
- Aggregate across instances in ClusterSimulator
- Add to RawMetrics and CLI anomaly counters

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Implement EnqueueRequest capacity check

**Contracts Implemented:** BC-1, BC-2, BC-7, BC-8

**Files:**
- Modify: `sim/simulator.go:303-307` (EnqueueRequest)

**Step 1: Write failing test for BC-1 (oversized request dropped)**

In `sim/simulator_test.go`, add:

```go
func TestEnqueueRequest_OversizedInput_DroppedNotEnqueued(t *testing.T) {
	// GIVEN a simulator with 10 KV blocks of 16 tokens each (160 token capacity)
	cfg := SimConfig{
		Horizon: 1_000_000,
		Seed:    42,
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   10,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     256,
			MaxScheduledTokens: 2048,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{1000, 1, 1},
			AlphaCoeffs: []float64{0, 0, 0},
		},
	}
	sim, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND a request with 200 input tokens (needs ceil(200/16) = 13 blocks > 10 total)
	oversized := &Request{
		ID:          "oversized_req",
		InputTokens: make([]int, 200),
		State:       StateQueued,
	}
	// Register it in Metrics.Requests (simulating InjectArrival behavior)
	sim.Metrics.Requests[oversized.ID] = NewRequestMetrics(oversized, 0)

	// WHEN we try to enqueue it
	sim.EnqueueRequest(oversized)

	// THEN it must NOT be in the wait queue
	if sim.WaitQ.Len() != 0 {
		t.Errorf("WaitQ.Len() = %d, want 0 (oversized request should not be enqueued)", sim.WaitQ.Len())
	}

	// AND DroppedUnservable must be incremented
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}

	// AND the request must be removed from per-request tracking (BC-7)
	if _, exists := sim.Metrics.Requests[oversized.ID]; exists {
		t.Error("dropped request should be removed from Metrics.Requests")
	}

	// AND TotalInputTokens must NOT include the dropped request's tokens
	if sim.Metrics.TotalInputTokens != 0 {
		t.Errorf("TotalInputTokens = %d, want 0 (dropped request tokens not counted)", sim.Metrics.TotalInputTokens)
	}
}
```

**Step 2: Write test for BC-2 (normal request unaffected)**

In `sim/simulator_test.go`, add:

```go
func TestEnqueueRequest_NormalInput_Enqueued(t *testing.T) {
	// GIVEN a simulator with 100 KV blocks of 16 tokens each
	cfg := SimConfig{
		Horizon: 1_000_000,
		Seed:    42,
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   100,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     256,
			MaxScheduledTokens: 2048,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{1000, 1, 1},
			AlphaCoeffs: []float64{0, 0, 0},
		},
	}
	sim, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND a request that fits (100 tokens needs ceil(100/16) = 7 blocks <= 100 total)
	normal := &Request{
		ID:          "normal_req",
		InputTokens: make([]int, 100),
		State:       StateQueued,
	}

	// WHEN we enqueue it
	sim.EnqueueRequest(normal)

	// THEN it must be in the wait queue
	if sim.WaitQ.Len() != 1 {
		t.Errorf("WaitQ.Len() = %d, want 1", sim.WaitQ.Len())
	}

	// AND DroppedUnservable must remain 0
	if sim.Metrics.DroppedUnservable != 0 {
		t.Errorf("DroppedUnservable = %d, want 0", sim.Metrics.DroppedUnservable)
	}

	// AND TotalInputTokens must include the request's tokens
	if sim.Metrics.TotalInputTokens != 100 {
		t.Errorf("TotalInputTokens = %d, want 100", sim.Metrics.TotalInputTokens)
	}
}
```

**Step 3: Run tests to verify they fail**

Run: `go test ./sim/... -run "TestEnqueueRequest_OversizedInput|TestEnqueueRequest_NormalInput" -v`
Expected: FAIL (EnqueueRequest doesn't check capacity yet)

**Step 4: Implement the capacity check**

In `sim/simulator.go`, replace `EnqueueRequest`:

```go
// EnqueueRequest adds a newly arrived request to the waiting queue.
// Requests whose input tokens require more KV blocks than the total cache
// capacity are dropped with a warning (R19: livelock protection). This mirrors
// real vLLM behavior where oversized requests are rejected before entering
// the engine.
func (sim *Simulator) EnqueueRequest(r *Request) {
	blocksNeeded := (int64(len(r.InputTokens)) + sim.KVCache.BlockSize() - 1) / sim.KVCache.BlockSize()
	if blocksNeeded > sim.KVCache.TotalCapacity() {
		logrus.Warnf("dropping request %s: input requires %d KV blocks but cache has only %d total",
			r.ID, blocksNeeded, sim.KVCache.TotalCapacity())
		sim.Metrics.DroppedUnservable++
		delete(sim.Metrics.Requests, r.ID)
		return
	}
	sim.WaitQ.Enqueue(r)
	sim.Metrics.TotalInputTokens += len(r.InputTokens)
}
```

**Step 5: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestEnqueueRequest_OversizedInput|TestEnqueueRequest_NormalInput" -v`
Expected: PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/simulator.go sim/simulator_test.go
git commit -m "fix(sim): drop oversized requests in EnqueueRequest (BC-1, BC-2, BC-7, BC-8)

Requests whose input tokens require more KV blocks than total cache
capacity are now dropped before entering the wait queue, preventing
the livelock described in #373. Mirrors vLLM pre-engine validation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Integration test — simulation terminates with oversized requests

**Contracts Implemented:** BC-3, BC-5, BC-6

**Files:**
- Modify: `sim/simulator_test.go`

**Step 1: Write test for BC-3/BC-6 (simulation terminates, no livelock)**

```go
func TestSimulator_OversizedRequests_TerminatesNoLivelock(t *testing.T) {
	// GIVEN a simulator with very small KV cache (50 blocks × 16 tokens = 800 tokens)
	// This is the exact reproduction case from issue #373
	cfg := SimConfig{
		Horizon: 10_000_000,
		Seed:    42,
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   50,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     256,
			MaxScheduledTokens: 2048,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{6910, 17.67, 2.84},
			AlphaCoeffs: []float64{0, 0, 0},
		},
	}
	sim, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND a mix of requests: some fit, some don't
	// Request 0: 900 tokens → ceil(900/16) = 57 blocks > 50 → dropped
	oversized := &Request{
		ID:           "request_oversized",
		InputTokens:  make([]int, 900),
		OutputTokens: make([]int, 10),
		ArrivalTime:  100_000,
		State:        StateQueued,
	}
	// Request 1: 100 tokens → ceil(100/16) = 7 blocks <= 50 → fits
	normal := &Request{
		ID:           "request_normal",
		InputTokens:  make([]int, 100),
		OutputTokens: make([]int, 10),
		ArrivalTime:  200_000,
		State:        StateQueued,
	}

	sim.InjectArrival(oversized)
	sim.InjectArrival(normal)

	// WHEN we run the simulation
	sim.Run()

	// THEN it must terminate (reaching this line proves no livelock — BC-6)
	sim.Finalize()

	// AND the oversized request must be dropped
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}

	// AND the normal request must complete
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests = %d, want 1", sim.Metrics.CompletedRequests)
	}

	// AND conservation must hold (BC-5):
	// completed + still_queued + still_running + dropped = total injected into EnqueueRequest
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 2 {
		t.Errorf("conservation: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, want 2",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning,
			sim.Metrics.DroppedUnservable, total)
	}
}
```

**Step 2: Write test for BC-6 (all oversized — simulation still terminates)**

```go
func TestSimulator_AllOversized_TerminatesEmpty(t *testing.T) {
	// GIVEN a simulator with tiny KV cache
	cfg := SimConfig{
		Horizon: 10_000_000,
		Seed:    42,
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   5,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     256,
			MaxScheduledTokens: 2048,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{1000, 1, 1},
			AlphaCoeffs: []float64{0, 0, 0},
		},
	}
	sim, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND all requests are oversized (200 tokens → 13 blocks > 5 total)
	for i := 0; i < 5; i++ {
		req := &Request{
			ID:           fmt.Sprintf("request_%d", i),
			InputTokens:  make([]int, 200),
			OutputTokens: make([]int, 10),
			ArrivalTime:  int64(i) * 100_000,
			State:        StateQueued,
		}
		sim.InjectArrival(req)
	}

	// WHEN we run the simulation
	sim.Run()

	// THEN it must terminate (no livelock)
	// AND all requests must be dropped
	if sim.Metrics.DroppedUnservable != 5 {
		t.Errorf("DroppedUnservable = %d, want 5", sim.Metrics.DroppedUnservable)
	}

	// AND no requests completed
	if sim.Metrics.CompletedRequests != 0 {
		t.Errorf("CompletedRequests = %d, want 0", sim.Metrics.CompletedRequests)
	}
}
```

**Step 3: Run tests**

Run: `go test ./sim/... -run "TestSimulator_OversizedRequests_Terminates|TestSimulator_AllOversized" -v`
Expected: PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/simulator_test.go
git commit -m "test(sim): integration tests for KV livelock fix (BC-3, BC-5, BC-6)

- TestSimulator_OversizedRequests_TerminatesNoLivelock: mixed workload
  with oversized and normal requests; verifies termination and conservation
- TestSimulator_AllOversized_TerminatesEmpty: all requests oversized;
  verifies simulation terminates with zero completions

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Cluster-level metrics test and JSON output test

**Contracts Implemented:** BC-4

**Files:**
- Modify: `sim/cluster/metrics_test.go`
- Modify: `sim/metrics_test.go` (or create test for JSON output)

**Step 1: Write test for cluster aggregation**

In `sim/cluster/metrics_test.go`, add:

```go
func TestCollectRawMetrics_DroppedUnservable(t *testing.T) {
	// GIVEN aggregated metrics with dropped requests
	m := sim.NewMetrics()
	m.DroppedUnservable = 3

	// WHEN collecting raw metrics
	raw := CollectRawMetrics(m, nil, 0, "")

	// THEN DroppedUnservable is captured
	if raw.DroppedUnservable != 3 {
		t.Errorf("DroppedUnservable: got %d, want 3", raw.DroppedUnservable)
	}
}
```

**Step 2: Write test for JSON output (BC-4)**

In `sim/metrics_test.go`, add:

```go
func TestSaveResults_DroppedUnservable_InJSON(t *testing.T) {
	// GIVEN metrics with dropped requests
	m := NewMetrics()
	m.DroppedUnservable = 2
	m.SimEndedTime = 1_000_000

	// WHEN saving results to a temp file
	tmpFile := filepath.Join(t.TempDir(), "test_output.json")
	m.SaveResults("test", 10_000_000, 100, tmpFile)

	// THEN the JSON file must contain dropped_unservable
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("reading output: %v", err)
	}

	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("parsing JSON: %v", err)
	}

	if output.DroppedUnservable != 2 {
		t.Errorf("DroppedUnservable in JSON = %d, want 2", output.DroppedUnservable)
	}

	// AND InjectedRequests must include dropped requests (BC-5)
	if output.InjectedRequests != 2 {
		t.Errorf("InjectedRequests = %d, want 2 (should include dropped)", output.InjectedRequests)
	}
}
```

**Step 3: Run tests**

Run: `go test ./sim/... ./sim/cluster/... -run "TestCollectRawMetrics_DroppedUnservable|TestSaveResults_DroppedUnservable" -v`
Expected: PASS

**Step 4: Run full test suite**

Run: `go test ./...`
Expected: ALL PASS (no regressions)

**Step 5: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/metrics_test.go sim/cluster/metrics_test.go
git commit -m "test(sim): metrics tests for DroppedUnservable (BC-4)

- Cluster aggregation test
- JSON output inclusion test

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Golden dataset regression check and final verification

**Contracts Implemented:** (verification only)

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 2: Run lint**

Run: `golangci-lint run ./...`
Expected: No issues

**Step 3: Verify golden dataset is unaffected**

Run: `go test ./sim/... -run Golden -v`
Expected: PASS (golden dataset uses default configs where all requests fit in KV cache)

**Step 4: Commit plan file**

```bash
git add docs/plans/fix-373-kv-livelock-plan.md
git commit -m "docs: add implementation plan for KV livelock fix (#373)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | TestEnqueueRequest_OversizedInput_DroppedNotEnqueued |
| BC-2 | Task 2 | Unit | TestEnqueueRequest_NormalInput_Enqueued |
| BC-3 | Task 3 | Integration | TestSimulator_OversizedRequests_TerminatesNoLivelock |
| BC-4 | Task 4 | Unit | TestSaveResults_DroppedUnservable_InJSON |
| BC-5 | Task 3 | Invariant | (conservation check in TestSimulator_OversizedRequests_TerminatesNoLivelock) |
| BC-6 | Task 3 | Integration | TestSimulator_AllOversized_TerminatesEmpty |
| BC-7 | Task 2 | Unit | (assertion in TestEnqueueRequest_OversizedInput_DroppedNotEnqueued) |
| BC-8 | Task 2 | Unit | (logrus warning — verified by code review, not test) |

Golden dataset: No update needed — default test configs don't trigger the bug.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Conservation invariant broken by dropped requests | Medium | High | Explicit conservation test (BC-5) in Task 3 | Task 3 |
| False positive drops (request fits but check says no) | Low | High | Check uses `ceil()` arithmetic matching `AllocateKVBlocks` | Task 2 |
| Existing tests break from new field | Low | Low | Zero-value field; no NewMetrics() constructor change needed | Task 1 |
| Dropped request still appears in JSON | Medium | Medium | Explicit `delete(Metrics.Requests, r.ID)` + BC-7 test | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific:**
- [x] No unnecessary abstractions (no new interfaces or types)
- [x] No feature creep (only the livelock fix + observability)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes (zero-value safe new field)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md: update INV-1 full pipeline formula to include `dropped_unservable`
- [x] docs/standards/invariants.md: update INV-1 to include `dropped_unservable` in full pipeline
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1 → 2 → 3 → 4 → 5)
- [x] All contracts mapped to tasks
- [x] Golden dataset unaffected
- [x] Construction sites: `NewMetrics()` uses zero-values for int fields — no change needed (R4)

**Antipattern rules:**
- [x] R1: Drop path increments counter + logs warning (not silent)
- [x] R4: `Metrics` has canonical `NewMetrics()` constructor; zero-value is correct for new int field
- [x] R6: No `logrus.Fatalf` in `sim/` — uses `logrus.Warnf`
- [x] R7: Conservation invariant test (BC-5) alongside integration tests
- [x] R14: `EnqueueRequest` is a single-concern method (enqueue gate)
- [x] R19: This PR IS the R19 fix

---

## Appendix: File-Level Implementation Details

### File: `sim/metrics.go`

**Purpose:** Add `DroppedUnservable` counter to Metrics struct and MetricsOutput population.

Add field after `StillRunning` (line ~30):
```go
DroppedUnservable int // Requests dropped because input tokens exceed KV cache capacity (R19)
```

In `SaveResults()`, add to MetricsOutput literal after `PreemptionCount`:
```go
DroppedUnservable: m.DroppedUnservable,
```

Update `InjectedRequests` formula (line ~72):
```go
InjectedRequests: m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable,
```

### File: `sim/metrics_utils.go`

**Purpose:** Add `DroppedUnservable` to JSON output struct.

Add field to `MetricsOutput` after `PreemptionCount`:
```go
DroppedUnservable int `json:"dropped_unservable"`
```

### File: `sim/simulator.go`

**Purpose:** Add capacity check to `EnqueueRequest`.

Replace lines 303-307:
```go
// EnqueueRequest adds a newly arrived request to the waiting queue.
// Requests whose input tokens require more KV blocks than the total cache
// capacity are dropped with a warning (R19: livelock protection). This mirrors
// real vLLM behavior where oversized requests are rejected before entering
// the engine.
func (sim *Simulator) EnqueueRequest(r *Request) {
	blocksNeeded := (int64(len(r.InputTokens)) + sim.KVCache.BlockSize() - 1) / sim.KVCache.BlockSize()
	if blocksNeeded > sim.KVCache.TotalCapacity() {
		logrus.Warnf("dropping request %s: input requires %d KV blocks but cache has only %d total",
			r.ID, blocksNeeded, sim.KVCache.TotalCapacity())
		sim.Metrics.DroppedUnservable++
		delete(sim.Metrics.Requests, r.ID)
		return
	}
	sim.WaitQ.Enqueue(r)
	sim.Metrics.TotalInputTokens += len(r.InputTokens)
}
```

### File: `sim/cluster/cluster.go`

**Purpose:** Aggregate DroppedUnservable across instances.

In `aggregateMetrics()`, add after line `merged.KVAllocationFailures += m.KVAllocationFailures`:
```go
merged.DroppedUnservable += m.DroppedUnservable
```

### File: `sim/cluster/metrics.go`

**Purpose:** Surface DroppedUnservable in RawMetrics.

Add to `RawMetrics` struct after `RejectedRequests`:
```go
DroppedUnservable int
```

In `CollectRawMetrics()`, update the `raw` literal initialization:
```go
raw := &RawMetrics{
    RejectedRequests:  rejectedRequests,
    DroppedUnservable: aggregated.DroppedUnservable,
}
```

### File: `cmd/root.go`

**Purpose:** Print DroppedUnservable in CLI anomaly counters.

Update the anomaly counters condition (line ~509) and output block to include `DroppedUnservable`.
