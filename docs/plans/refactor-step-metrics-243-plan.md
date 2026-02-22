# Separate State Mutation from Metrics Recording in Step() Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix a DES anti-pattern where a statistics field (`RequestRate`) feeds back into state evolution, and cleanly separate metric recording from state mutations in the Step() method.

**The problem today:** The `Metrics.RequestRate` field is classified as a statistic but is read back as input to workload generation — a violation of the DES principle that statistics must be derived from state, never drive it. Additionally, Step() interleaves ~28 metric writes across 5 phases with no structural separation, making it hard to audit which parts advance the simulation vs. record observations.

**What this PR adds:**
1. **RequestRate relocated** — moves the request rate from the `Metrics` struct (statistics) to the `Simulator` struct (state), eliminating the statistics→state feedback loop
2. **Observational metrics extracted** — factors queue depth snapshots, KV usage tracking, and per-request completion metrics into dedicated helper methods, giving Step() a clear phase structure
3. **CacheHitRate documented** — documents the `CacheHitRate` field in Metrics as an intentional observability signal flowing into `RoutingSnapshot`

**Why this matters:** Enforces the DES foundation (Banks et al.) that state variables evolve the system while statistics are derived observations. Prepares Step() for the BatchFormation extraction (#242) by reducing interleaved concerns.

**Architecture:** Pure refactor within `sim/` package. The `requestRate` field moves from `Metrics` to `Simulator` with a `SetRequestRate()` method for cluster mode access. Three helper methods (`recordQueueSnapshots`, `recordKVUsageMetrics`, `recordRequestCompletion`) are extracted from Step(), each handling a distinct observation concern.

**Source:** GitHub issue #243

**Closes:** Fixes #243

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a DES anti-pattern and adds structural clarity to the simulation's core Step() method. The `Metrics.RequestRate` field is used by workload generation to compute inter-arrival times — a statistics→state feedback loop that violates discrete-event simulation principles. The fix moves `RequestRate` to the Simulator struct where it belongs as configuration-derived state.

The second change extracts observational metric recording from Step() into helper methods, creating clear phase boundaries: schedule → observe queue state → execute → observe KV state → complete requests + record completion metrics → schedule next step.

No output or behavioral changes. All existing invariants (INV-1 through INV-8) preserved. Golden dataset unaffected.

Adjacent blocks: `sim/cluster/instance.go` (calls `SetRequestRate`), `sim/workload_config.go` (reads rate for arrival generation).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: RequestRate Relocated
- GIVEN a Simulator constructed with `GuideLLMConfig.Rate = R`
- WHEN workload generation runs
- THEN the inter-arrival interval is `1/R` ticks, identical to pre-refactor behavior
- MECHANISM: `requestRate` field on Simulator replaces `Metrics.RequestRate`; all consumers updated

BC-2: Output Determinism Preserved (INV-6)
- GIVEN the golden dataset test configuration
- WHEN the simulation runs with the same seed
- THEN the output is byte-identical to the pre-refactor golden dataset
- MECHANISM: No behavioral change — only field location and method extraction

BC-3: Step Observation Separation
- GIVEN Step() processes a batch with completions
- WHEN the step completes
- THEN queue depth snapshots, KV usage metrics, and per-request completion metrics are recorded with identical values as pre-refactor
- MECHANISM: Helper methods `recordQueueSnapshots`, `recordKVUsageMetrics`, `recordRequestCompletion` called from Step() at the same points

**Negative Contracts:**

BC-4: No Invariant Regression
- GIVEN any simulation configuration
- WHEN the simulation runs to completion
- THEN INV-1 (request conservation), INV-3 (clock monotonicity), INV-4 (KV block conservation), INV-5 (causality), and INV-8 (work-conserving) all hold
- MECHANISM: Refactoring extracts code into helpers without changing execution order or logic

BC-5: CacheHitRate Documented
- GIVEN a developer reading the Metrics struct definition
- WHEN they encounter the CacheHitRate field
- THEN the documentation clearly states it is a read-only observability signal set at finalization, not a state feedback loop
- MECHANISM: Inline documentation comment on the field

### C) Component Interaction

```
SimConfig.GuideLLMConfig.Rate
          │
          ▼
   ┌──────────────┐     ┌────────────────────┐
   │  Simulator    │────▶│ workload_config.go  │
   │  .requestRate │     │ generateWorkload*() │
   └──────┬───────┘     └────────────────────┘
          │
          │  SetRequestRate()
          ▼
   ┌──────────────────┐
   │ InstanceSimulator │  (cluster mode)
   │ .SetRequestRate() │
   └──────────────────┘
```

**State changes:** `requestRate` moves from `Metrics` (statistics struct) to `Simulator` (simulation state). No new mutable state introduced.

**Extension friction:** Adding a new metric requires 1 file (helpers in simulator.go). No change from current friction.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Move RequestRate to SimConfig or Simulator field" | Moves to Simulator field (not SimConfig) | SIMPLIFICATION: SimConfig is a value type passed to constructor; requestRate is set conditionally in constructor logic (CSV mode = 0, GuideLLM mode = Rate). A Simulator field with SetRequestRate() setter is cleaner than conditional SimConfig mutation. |
| "Extract metric recording into separate recordStepMetrics() or observer callback" | Extracts into 3 focused helpers instead of one monolithic method | ADDITION: Three helpers (queue snapshots, KV metrics, request completion) provide better separation than a single `recordStepMetrics()` that would still be ~40 lines. |
| "Structure Step() as two phases: (a) advance state, (b) record observations" | Structures as 4 labeled phases with interleaved recording where necessary | CORRECTION: A strict two-phase separation is impractical — TTFT recording in the execution loop must happen during iteration (separating into a second pass would double the loop). Instead, clear phase labels + extracted helpers achieve the design intent. |
| "Document CacheHitRate in RoutingSnapshot" | Documents in Metrics struct comment | SIMPLIFICATION: CacheHitRate is set in cluster/instance.go Finalize(), not in Step(). RoutingSnapshot doesn't currently read CacheHitRate — it reads CacheHitRate from KVStore methods. A Metrics struct comment suffices. |

### E) Review Guide

**The tricky part:** The `workload_config_test.go` tests construct `Simulator{}` struct literals that directly set `Metrics.RequestRate`. After moving the field, these must use `requestRate` (unexported) — valid because the tests are in package `sim`.

**What to scrutinize:** BC-1 — verify that `requestRate` is set in all the same code paths as `Metrics.RequestRate` was (NewSimulator CSV path, NewSimulator GuideLLM path, cluster SetRequestRate).

**What's safe to skim:** The three helper method extractions (BC-3) — these are mechanical moves with no logic changes.

**Known debt:** `Metrics.PeakKVBlocksUsed` uses a self-read pattern (running max). This is a statistics→statistics accumulation, not a state feedback loop, so it's acceptable. Filed for awareness, not action.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/simulator.go` — add `requestRate` field, `SetRequestRate()`, extract 3 helpers, restructure Step()
- `sim/metrics.go` — remove `RequestRate` field, add CacheHitRate documentation
- `sim/workload_config.go` — use `sim.requestRate` instead of `sim.Metrics.RequestRate`
- `sim/workload_config_test.go` — update test struct literals
- `sim/cluster/instance.go` — delegate `SetRequestRate` to `sim.Simulator.SetRequestRate()`

**Key decisions:**
- `requestRate` is unexported (private) — access only via `SetRequestRate()` from cluster package
- Three focused helpers instead of one monolithic `recordStepMetrics()`
- TTFT recording stays in execution loop (correctly co-located with state transition)

**Confirmation:** No dead code, no new types, no new interfaces, all paths exercisable.

### G) Task Breakdown

---

### Task 1: Move RequestRate from Metrics to Simulator

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `sim/simulator.go` (add field, setter, update constructor)
- Modify: `sim/metrics.go` (remove field)
- Modify: `sim/workload_config.go` (update reads)
- Modify: `sim/workload_config_test.go` (update struct literals)
- Modify: `sim/cluster/instance.go` (delegate to new setter)
- Modify: `sim/cluster/cluster.go` (remove aggregation of RequestRate)
- Modify: `sim/cluster/cluster_test.go` (remove RequestRate assertion)

**Step 1: Update workload_config_test.go to use new field**

Context: The existing tests construct `Simulator{}` with `Metrics.RequestRate`. We update them to use the new `requestRate` field before moving it, so we can verify the test intent is preserved.

In `sim/workload_config_test.go`, replace entire file:
```go
package sim

import (
	"testing"
)

func TestGenerateWorkloadDistribution_ZeroRate_Panics(t *testing.T) {
	// GIVEN a simulator with requestRate = 0
	// WHEN generateWorkloadDistribution is called
	// THEN it panics instead of entering an infinite loop (#202)
	sim := &Simulator{
		requestRate: 0,
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for zero requestRate, but did not panic")
		}
	}()
	sim.generateWorkloadDistribution()
}

func TestGenerateWorkloadDistribution_NegativeRate_Panics(t *testing.T) {
	// GIVEN a simulator with requestRate = -1
	// WHEN generateWorkloadDistribution is called
	// THEN it panics
	sim := &Simulator{
		requestRate: -1,
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for negative requestRate, but did not panic")
		}
	}()
	sim.generateWorkloadDistribution()
}
```

**Step 2: Run test to verify it fails (compilation)**

Run: `go test ./sim/... -run TestGenerateWorkloadDistribution -v 2>&1 | head -20`
Expected: FAIL — `requestRate` field does not exist yet on Simulator

**Step 3: Implement the field move**

In `sim/simulator.go`, add field to Simulator struct (after `latencyModel`):
```go
	latencyModel           LatencyModel
	requestRate            float64 // arrival rate for workload generation (moved from Metrics — DES state/statistics separation, #243)
```

In `sim/simulator.go` NewSimulator, replace:
```go
	if cfg.TracesWorkloadFilePath != "" && cfg.GuideLLMConfig == nil {
		s.Metrics.RequestRate = 0.0
```
with:
```go
	if cfg.TracesWorkloadFilePath != "" && cfg.GuideLLMConfig == nil {
		s.requestRate = 0.0
```

And replace:
```go
	} else if cfg.GuideLLMConfig != nil {
		s.Metrics.RequestRate = cfg.GuideLLMConfig.Rate
```
with:
```go
	} else if cfg.GuideLLMConfig != nil {
		s.requestRate = cfg.GuideLLMConfig.Rate
```

Add setter method to `sim/simulator.go` (after SimHorizon):
```go
// SetRequestRate sets the arrival rate for workload generation.
// Used by cluster mode to propagate the per-instance rate.
// Precondition: rate >= 0. Callers are responsible for validation (R3).
func (sim *Simulator) SetRequestRate(rate float64) { sim.requestRate = rate }
```

In `sim/workload_config.go`, replace line 116:
```go
	if sim.requestRate <= 0 {
```
and replace line 162:
```go
		currentTime += int64(1 / sim.requestRate)
```

In `sim/metrics.go`, remove the `RequestRate` field from the Metrics struct:
```go
	RequestRate       float64 // Incoming request rate   ← DELETE THIS LINE
```

In `sim/cluster/instance.go`, replace SetRequestRate:
```go
// SetRequestRate sets the request rate on the underlying simulator.
func (i *InstanceSimulator) SetRequestRate(rate float64) {
	i.sim.SetRequestRate(rate)
}
```

In `sim/cluster/cluster.go`, in `aggregateMetrics()`, remove lines:
```go
	if c.workload != nil {
		merged.RequestRate = c.workload.Rate
	}
```
(RequestRate is no longer in Metrics — the rate is stored on each Simulator instance.)

In `sim/cluster/cluster_test.go`, remove the RequestRate assertion (lines ~464-466):
```go
	if agg.RequestRate != workload.Rate {
		t.Errorf("aggregated RequestRate: got %v, want %v", agg.RequestRate, workload.Rate)
	}
```
(RequestRate is no longer part of aggregated Metrics.)

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run TestGenerateWorkloadDistribution -v`
Expected: PASS — both zero and negative rate tests pass

Run: `go test ./sim/... -v 2>&1 | tail -20`
Expected: PASS — all sim tests pass including golden dataset

Run: `go test ./sim/cluster/... -v 2>&1 | tail -20`
Expected: PASS — all cluster tests pass

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/... ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/metrics.go sim/workload_config.go sim/workload_config_test.go sim/cluster/instance.go sim/cluster/cluster.go sim/cluster/cluster_test.go
git commit -m "refactor(sim): move RequestRate from Metrics to Simulator (BC-1, BC-2)

Move requestRate from the Metrics statistics struct to the Simulator
struct, fixing a DES anti-pattern where a statistic fed back into
state evolution (workload generation read Metrics.RequestRate to
compute inter-arrival intervals).

- Add requestRate field to Simulator
- Add SetRequestRate() setter for cluster mode
- Update workload_config.go to use sim.requestRate
- Update workload_config_test.go to use new field
- Delegate cluster/instance.go SetRequestRate to Simulator

No behavioral change — golden dataset output identical.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Extract per-step observational metrics from Step()

**Contracts Implemented:** BC-3, BC-4

**Files:**
- Modify: `sim/simulator.go` (extract helpers, restructure Step())

**Step 1: Verify golden dataset baseline**

Run: `go test ./sim/... -run TestSimulator_GoldenDataset -v`
Expected: PASS — establish baseline before restructuring

**Step 2: Extract helper methods and restructure Step()**

Context: We extract three helper methods from Step() and add phase labels. The execution loop TTFT recording stays in place (correctly co-located with state). The completion loop is split into state transitions followed by a recordRequestCompletion() call.

Add three helper methods to `sim/simulator.go` (before Step):

```go
// recordQueueSnapshots records the wait queue and running batch sizes at this step.
// Called after batch formation, before execution.
func (sim *Simulator) recordQueueSnapshots() {
	sim.Metrics.NumWaitQRequests = append(sim.Metrics.NumWaitQRequests, sim.WaitQ.Len())
	sim.Metrics.NumRunningBatchRequests = append(sim.Metrics.NumRunningBatchRequests, len(sim.RunningBatch.Requests))
}

// recordKVUsageMetrics records peak and time-weighted KV block usage.
// Called after execution, before completion processing.
func (sim *Simulator) recordKVUsageMetrics(stepDuration int64) {
	used := sim.KVCache.UsedBlocks()
	if used > sim.Metrics.PeakKVBlocksUsed {
		sim.Metrics.PeakKVBlocksUsed = used
	}
	sim.Metrics.KVBlocksUsed += float64(used) * float64(stepDuration)
}

// recordRequestCompletion records per-request metrics for a completed request.
// Called after state transitions (req.State, req.ITL, req.FinishedStepIdx)
// and KV cleanup are done.
func (sim *Simulator) recordRequestCompletion(req *Request) {
	sim.Metrics.CompletedRequests++

	var itlSum int64
	for _, v := range req.ITL {
		itlSum += v
	}
	lat := req.FirstTokenTime + itlSum
	sim.Metrics.RequestE2Es[req.ID] = float64(lat)
	logrus.Debugf("Finished req: ID: %s at time: %d", req.ID, lat+req.ArrivalTime)
	if len(req.OutputTokens) > 0 {
		reqTotalOutput := lat - req.FirstTokenTime
		sim.Metrics.RequestITLs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.OutputTokens)-1, 1))
	} else {
		sim.Metrics.RequestITLs[req.ID] = 0
	}
	sim.Metrics.RequestStepCounters = append(sim.Metrics.RequestStepCounters, req.FinishedStepIdx-req.ScheduledStepIdx)
	sim.Metrics.RequestCompletionTimes[req.ID] = float64(lat + req.ArrivalTime)
	sim.Metrics.AllITLs = append(sim.Metrics.AllITLs, req.ITL...)
}
```

Replace the Step() method body with the restructured version:

```go
// Step simulates a single vllm step(): batch scheduling, model execution, and completion.
// Phases: (1) schedule batch, (2) execute prefill/decode, (3) process completions, (4) schedule next step.
// Observational metrics are recorded via helper methods at phase boundaries.
func (sim *Simulator) Step(now int64) {

	// === Phase 1: Schedule batch ===
	sim.stepCount += 1

	// Synchronize KV cache clock for thrashing detection (no-op for single-tier KVCacheState)
	sim.KVCache.SetClock(now)

	// Assign priorities to queued requests and order queue per scheduler policy
	for _, req := range sim.WaitQ.Items() {
		req.Priority = sim.priorityPolicy.Compute(req, now)
	}
	sim.WaitQ.Reorder(func(reqs []*Request) {
		sim.scheduler.OrderQueue(reqs, now)
	})

	// Subprocess: fill running batch from wait queue, similar to vLLM's scheduler.schedule()
	sim.makeRunningBatch(now)

	// Record queue depth observations after batch formation
	sim.recordQueueSnapshots()

	// === Phase 2: Execute batch (prefill + decode) ===

	// Estimate step time via LatencyModel (blackbox or roofline, selected at construction)
	currStepAdvance := sim.latencyModel.StepTime(sim.RunningBatch.Requests)

	// Add transfer latency from CPU→GPU reloads (0 for single-tier)
	currStepAdvance += sim.KVCache.ConsumePendingTransferLatency()

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex < Len64(req.InputTokens) {
			req.ProgressIndex = sim.reqNumComputedTokens[req.ID]
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else {
			// this request goes through decode phase in this batch
			req.ProgressIndex++
			sim.Metrics.TotalOutputTokens++
			req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
		}
		if req.ProgressIndex == Len64(req.InputTokens) { // prefill complete, first token is generated
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance + sim.latencyModel.OutputTokenProcessingTime() - req.ArrivalTime
			sim.Metrics.TTFTSum += req.FirstTokenTime // in microsec
			sim.Metrics.RequestTTFTs[req.ID] = float64(req.FirstTokenTime)
		}
	}

	// Record KV cache usage observations after execution
	sim.recordKVUsageMetrics(currStepAdvance)

	// === Phase 3: Process completions ===

	// IMPORTANT: This completion loop MUST run as a separate pass after the
	// prefill/decode execution loop above. For zero-output-token requests,
	// both "prefill completed" and "request completed" conditions are true
	// in the same step. The two-pass design ensures prefill metrics (TTFT)
	// are recorded before completion metrics (E2E). If these loops were ever
	// consolidated into a single pass, both branches would fire for the
	// same request in the same step.
	remaining := []*Request{}
	for _, req := range sim.RunningBatch.Requests {
		// in cases where there are 0 output tokens, set it to 1 manually to avoid errors
		if req.ProgressIndex == Len64(req.InputTokens)+max(Len64(req.OutputTokens), 1)-1 {
			// State transitions
			req.State = StateCompleted
			req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
			if len(req.OutputTokens) > 0 {
				ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
				if !ok {
					logrus.Errorf("[tick %07d] KV allocation failed for completing request %s (request will still complete) — this indicates a cache accounting bug", now, req.ID)
					sim.Metrics.KVAllocationFailures++
				}
			}
			// ReleaseKVBlocks is safe even when the final-token allocation failed:
			// AllocateKVBlocks only modifies RequestMap on success, so Release
			// frees exactly the blocks from prior successful allocations.
			sim.KVCache.ReleaseKVBlocks(req)
			req.FinishedStepIdx = sim.stepCount
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance,
				Request: req,
			})

			// Record completion metrics
			sim.recordRequestCompletion(req)
		} else {
			remaining = append(remaining, req)
		}
	}

	// === Phase 4: Schedule next step ===
	if len(remaining) > 0 {
		sim.RunningBatch.Requests = remaining
		// estimate queue overhead from LR (sim.features)
		//
		pbe := StepEvent{time: now + currStepAdvance}
		sim.Schedule(&pbe)
		sim.stepEvent = &pbe
	} else {
		sim.RunningBatch = nil
		sim.stepEvent = nil
		// Work-conserving: if WaitQ has pending requests, immediately
		// schedule a new step to form the next batch. Without this,
		// queued requests are stranded until the next arrival event
		// triggers a QueuedEvent — violating the work-conserving
		// property that real vLLM maintains.
		if sim.WaitQ.Len() > 0 {
			pbe := StepEvent{time: now + currStepAdvance}
			sim.Schedule(&pbe)
			sim.stepEvent = &pbe
		}
	}
}
```

**Step 3: Run all tests to verify behavioral preservation**

Run: `go test ./sim/... -v 2>&1 | tail -20`
Expected: PASS — all tests including golden dataset

Run: `go test ./sim/cluster/... -v 2>&1 | tail -20`
Expected: PASS — all cluster tests

Run: `go test ./sim/... -run TestSimulator_Determinism -v`
Expected: PASS — determinism preserved (INV-6)

Run: `go test ./sim/... -run TestSimulator_RequestConservation -v`
Expected: PASS — request conservation preserved (INV-1)

Run: `go test ./sim/... -run TestWorkConserving -v`
Expected: PASS — work-conserving preserved (INV-8)

**Step 4: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/simulator.go
git commit -m "refactor(sim): extract per-step observational metrics from Step() (BC-3, BC-4)

Extract three helper methods from Step() to separate metric recording
from state mutation:
- recordQueueSnapshots(): WaitQ/RunningBatch depth after batch formation
- recordKVUsageMetrics(): peak and time-weighted KV block usage
- recordRequestCompletion(): per-request E2E, ITL, step counters

Add phase labels to Step() for structural clarity:
Phase 1 (schedule batch), Phase 2 (execute), Phase 3 (completions),
Phase 4 (schedule next step).

TTFT recording stays in execution loop (correctly co-located with
the prefill-complete state transition).

No behavioral change — all invariants and golden dataset preserved.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Document CacheHitRate and update CLAUDE.md

**Contracts Implemented:** BC-5 (CacheHitRate documented)

**Files:**
- Modify: `sim/metrics.go` (add documentation comment)
- Modify: `CLAUDE.md` (update simulator.go description)

**Step 1: Add documentation to CacheHitRate in Metrics**

In `sim/metrics.go`, replace:
```go
	CacheHitRate         float64 // Cumulative cache hit rate at finalization (PR12)
```
with:
```go
	CacheHitRate         float64 // Cumulative cache hit rate at finalization (PR12). Intentional observability signal: set by cluster/instance.go Finalize() from KVStore.CacheHitRate(). Read-only statistic — does not feed back into state evolution.
```

**Step 2: Update CLAUDE.md simulator.go description**

In `CLAUDE.md`, in the `sim/simulator.go` bullet, append after "observation methods" text:

Replace:
```
- **simulator.go**: `SimConfig` struct, `NewSimulator(SimConfig) (*Simulator, error)` constructor, `Simulator` struct and event loop (`Run()`), batch formation (`makeRunningBatch`), step execution, observation methods (`QueueDepth()`, `BatchSize()`, `CurrentClock()`, `SimHorizon()`)
```
with:
```
- **simulator.go**: `SimConfig` struct, `NewSimulator(SimConfig) (*Simulator, error)` constructor, `Simulator` struct and event loop (`Run()`), batch formation (`makeRunningBatch`), step execution with phased metric recording (`recordQueueSnapshots`, `recordKVUsageMetrics`, `recordRequestCompletion`), observation methods (`QueueDepth()`, `BatchSize()`, `CurrentClock()`, `SimHorizon()`)
```

**Step 3: Build verification**

Run: `go build ./...`
Expected: Success

**Step 4: Commit**

```bash
git add sim/metrics.go CLAUDE.md
git commit -m "docs(sim): document CacheHitRate as intentional observability signal (BC-5)

- Add documentation to CacheHitRate field clarifying it is a read-only
  statistic set at finalization, not a state feedback loop
- Update CLAUDE.md simulator.go description with new helper methods

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 1 | Unit | TestGenerateWorkloadDistribution_ZeroRate_Panics (updated) |
| BC-1 | Task 1 | Unit | TestGenerateWorkloadDistribution_NegativeRate_Panics (updated) |
| BC-2 | Task 1 | Golden | TestSimulator_GoldenDataset (existing, unchanged) |
| BC-2 | Task 1 | Invariant | TestSimulator_Determinism_ByteIdenticalJSON (existing) |
| BC-3 | Task 2 | Golden | TestSimulator_GoldenDataset (existing, unchanged) |
| BC-4 | Task 2 | Invariant | TestSimulator_RequestConservation_* (existing) |
| BC-4 | Task 2 | Invariant | TestSimulator_Causality_FullChain (existing) |
| BC-4 | Task 2 | Invariant | TestSimulator_ClockMonotonicity (existing) |
| BC-4 | Task 2 | Invariant | TestSimulator_KVBlockConservation (existing) |
| BC-4 | Task 2 | Invariant | TestWorkConserving_StepRestartsWhenWaitQNonEmpty (existing) |

**Golden dataset update:** Not needed — this is a pure refactor with no output changes.

**New tests:** None required. The existing test suite (golden + invariant + determinism) fully covers all behavioral contracts. The workload_config tests are updated (not new) to use the relocated field.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Missed consumer of Metrics.RequestRate | Low | High (compilation fails) | Comprehensive grep found 6 source sites + 3 test sites (incl. cluster.go aggregation + cluster_test.go assertion) | Task 1 |
| recordRequestCompletion changes execution order | Low | High (wrong metrics) | Helper called at exact same point as original inline code; golden dataset verifies | Task 2 |
| Cluster SetRequestRate breaks | Low | Medium | Cluster tests cover this path directly | Task 1 |

### E) Review Guide (repeated for visibility)

**THE TRICKY PART:** The `recordRequestCompletion` extraction must preserve the exact computation order for E2E latency (`lat = FirstTokenTime + sum(ITL)`) and ensure the ITL append (`req.ITL = append(...)`) and `req.FinishedStepIdx = sim.stepCount` are set before the helper is called. Note: `CompletedRequests++` moves after `RequestLeftEvent` scheduling (was before) — this is safe because neither reads the counter during Step().

**WHAT TO SCRUTINIZE:** Task 1 — verify all 6 production consumers and 3 test consumers of `Metrics.RequestRate` are updated (simulator.go ×2, workload_config.go ×2, cluster/instance.go ×1, cluster/cluster.go ×1, workload_config_test.go ×2, cluster_test.go ×1).

**WHAT'S SAFE TO SKIM:** The three helper method bodies — they're copy-paste from Step() with no logic changes.

**KNOWN DEBT:** `Metrics.PeakKVBlocksUsed` self-read pattern (running max) is a statistics→statistics accumulation, not a state feedback. Acceptable per DES principles but noted.

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — 3 focused helpers, no new types or interfaces
- [x] No feature creep — strictly #243 scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes — pure internal refactor
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing shared test package
- [x] CLAUDE.md updated (Task 3)
- [x] No stale references
- [x] Deviation log reviewed — all 4 deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 → Task 2 → Task 3)
- [x] All contracts mapped to tasks
- [x] Golden dataset regeneration not needed
- [x] Construction site audit: Metrics struct has one canonical constructor (NewMetrics); Simulator has one constructor (NewSimulator) + 2 test struct literals in workload_config_test.go — all covered
- [x] Not part of active macro plan

**Antipattern rules:**
- [x] R1: No silent continue/return dropping data
- [x] R2: No new map iteration for ordered output
- [x] R3: No new CLI flags
- [x] R4: Construction sites audited (Metrics field removed, Simulator field added)
- [x] R5: No new resource allocation loops
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: Existing invariant tests cover all contracts
- [x] R8: No new exported maps
- [x] R9: No new YAML fields
- [x] R10: No new YAML parsing
- [x] R11: No new division operations
- [x] R12: Golden dataset unchanged
- [x] R13: No new interfaces
- [x] R14: Helper methods are single-concern (R14 improvement)
- [x] R15: No stale PR references
- [x] R16: N/A (no new config params)
- [x] R17: N/A (no new routing signals)

---

## Appendix: File-Level Implementation Details

### File: `sim/simulator.go`

**Purpose:** Core simulator — add `requestRate` field, `SetRequestRate()` setter, three metric helper methods, restructured Step().

**Changes:**

1. Add to Simulator struct:
```go
requestRate float64 // arrival rate for workload generation (moved from Metrics — DES state/statistics separation, #243)
```

2. In NewSimulator, replace `s.Metrics.RequestRate` with `s.requestRate` (2 sites: lines 203, 208).

3. Add setter method:
```go
// SetRequestRate sets the arrival rate for workload generation.
// Used by cluster mode to propagate the per-instance rate.
// Precondition: rate >= 0. Callers are responsible for validation (R3).
func (sim *Simulator) SetRequestRate(rate float64) { sim.requestRate = rate }
```

4. Add three helper methods (see Task 2 Step 2 for complete code):
- `recordQueueSnapshots()` — 3 lines
- `recordKVUsageMetrics(stepDuration int64)` — 6 lines
- `recordRequestCompletion(req *Request)` — 18 lines

5. Restructure Step() with phase labels and helper calls (see Task 2 Step 2 for complete code).

### File: `sim/metrics.go`

**Purpose:** Remove `RequestRate` field, document `CacheHitRate`.

**Changes:**
1. Delete line: `RequestRate float64 // Incoming request rate`
2. Update `CacheHitRate` comment to document it as intentional observability signal.

### File: `sim/workload_config.go`

**Purpose:** Use `sim.requestRate` instead of `sim.Metrics.RequestRate`.

**Changes:**
1. Line 116: `sim.Metrics.RequestRate` → `sim.requestRate`
2. Line 162: `sim.Metrics.RequestRate` → `sim.requestRate`

### File: `sim/workload_config_test.go`

**Purpose:** Update test struct literals for moved field.

**Changes:** Replace `Metrics: &Metrics{RequestRate: N}` with `requestRate: N` (2 sites).

### File: `sim/cluster/instance.go`

**Purpose:** Delegate SetRequestRate to Simulator method.

**Changes:**
1. Update comment: "sets the request rate on the underlying simulator"
2. Replace body: `i.sim.Metrics.RequestRate = rate` → `i.sim.SetRequestRate(rate)`

### File: `sim/cluster/cluster.go`

**Purpose:** Remove RequestRate aggregation from aggregateMetrics().

**Changes:**
1. Delete lines 340-342: `if c.workload != nil { merged.RequestRate = c.workload.Rate }`
(RequestRate is no longer in Metrics — the rate is stored on each Simulator instance.)

### File: `sim/cluster/cluster_test.go`

**Purpose:** Remove RequestRate assertion from aggregation test.

**Changes:**
1. Delete lines ~464-466: `if agg.RequestRate != workload.Rate { ... }`
(RequestRate is no longer part of aggregated Metrics.)
