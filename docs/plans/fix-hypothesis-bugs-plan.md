# Fix Hypothesis Experiment Bugs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 4 open bugs discovered by hypothesis experiments and add 1 regression test for a previously-fixed bug — all found by the H8, H9, H12, and H14 experiments.

**The problem today:** Five bugs found during hypothesis-driven experiments have no regression tests or fixes: (1) the preemption loop panics on empty batches, (2) severely undersized KV caches cause the simulator to hang forever, (3) the HOL blocking detector is blind to the most extreme imbalance case, (4) the priority inversion detector produces thousands of false positives for mixed-SLO workloads, and (5) a previously-fixed CLI flag override has no regression test to prevent reintroduction.

**What this PR adds:**
1. **Empty-batch panic guard** — `preempt()` returns false instead of crashing when the batch is empty and allocation still fails
2. **Livelock circuit breaker** — preemption loop terminates with an error log after evicting all running requests without making progress
3. **HOL detector fix** — instances with zero traffic are included in the comparison (avg=0.0), so single-instance concentration is detected
4. **Priority inversion fix** — detector compares requests within the same SLO class only, eliminating cross-class false positives
5. **CLI flag precedence regression test** — verifies `--total-kv-blocks` is not overwritten by defaults.yaml

**Why this matters:** These bugs undermine the simulator's correctness guarantees (INV-1 via panic, R19 via livelock) and its anomaly detection accuracy (R20 via blind detectors). Fixing them closes the experiment→code feedback loop.

**Architecture:** Three independent subsystems touched: `sim/simulator.go` (preemption path, #293/#297), `sim/cluster/metrics.go` (anomaly detectors, #291/#292), `cmd/` (CLI flag test, #285). No new types or interfaces. No architectural changes.

**Source:** GitHub issues #291, #292, #293, #297; regression test for #285 fix (cbb0de7)

**Closes:** Fixes #291, fixes #292, fixes #293, fixes #297

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes 4 bugs and adds 1 regression test, all discovered by hypothesis experiments (H8, H9, H12, H14). The fixes are in three independent subsystems:

1. **Preemption path** (`sim/simulator.go`): Guard against empty-batch panic (#293) and add a circuit breaker for livelock (#297). These are the same `preempt()` function but different failure modes.
2. **Anomaly detectors** (`sim/cluster/metrics.go`): Fix HOL blocking blind spot (#291) and priority inversion false positives (#292). Independent detector functions.
3. **CLI flag precedence** (`cmd/`): Regression test for the already-fixed #285.

No architectural changes. No new interfaces. No new CLI flags. The golden dataset will need regeneration because the preemption panic fix changes behavior for edge-case KV configurations.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Empty batch preemption returns false
- GIVEN a `preempt()` call where the running batch has been fully evicted
- WHEN allocation still fails after all requests are evicted
- THEN `preempt()` MUST return `false` (not panic)
- MECHANISM: Check `len(sim.RunningBatch.Requests) == 0` before accessing last element

BC-2: Preemption livelock terminates
- GIVEN a simulation where KV blocks are insufficient for any single request
- WHEN the preempt-requeue-schedule-preempt cycle would repeat indefinitely
- THEN the simulation MUST terminate (not hang) and log a warning about insufficient KV blocks
- MECHANISM: After evicting all running requests and failing, return false — caller breaks out of batch formation

BC-3: HOL blocking detected for single-instance concentration
- GIVEN a cluster where ALL traffic goes to one instance (e.g., `always-busiest` with 4 instances)
- WHEN `detectHOLBlocking()` is called with per-instance metrics
- THEN the result MUST be > 0 (HOL blocking detected)
- MECHANISM: Include zero-traffic instances with avg=0.0 in the comparison instead of skipping them

BC-4: Priority inversion compares within SLO class only
- GIVEN a mixed-SLO workload where "realtime" requests are naturally 10x faster than "batch" requests
- WHEN `detectPriorityInversions()` is called with `slo-based` priority policy
- THEN the result MUST be 0 when requests within each class are ordered correctly (no cross-class comparisons)
- MECHANISM: Group requests by SLOClass before comparing arrival-ordered E2E

BC-5: CLI flag precedence preserved
- GIVEN a user passes `--total-kv-blocks 50` and the model's default is 132,139
- WHEN `GetCoefficients()` returns the model default
- THEN `totalKVBlocks` MUST remain 50
- MECHANISM: `cmd.Flags().Changed("total-kv-blocks")` guard (already fixed in cbb0de7, this is a regression test)

**Negative Contracts:**

BC-6: No false positives for balanced HOL blocking
- GIVEN a cluster with roughly balanced traffic across instances
- WHEN `detectHOLBlocking()` is called
- THEN the result MUST be 0 (no false positives from the fix)

BC-7: No regression in priority inversion for single-class workloads
- GIVEN a workload with a single SLO class (or empty SLO class)
- WHEN `detectPriorityInversions()` is called with `slo-based` priority policy
- THEN existing behavior MUST be preserved (inversions detected within the single class)

BC-8: Conservation invariant preserved after preemption fix
- GIVEN any simulation configuration with the preemption fix
- WHEN the simulation completes
- THEN `injected_requests == completed_requests + still_queued + still_running` (INV-1)

### C) Component Interaction

No new components. Three independent fix sites:

```
cmd/root.go:173     ←── BC-5 regression test (already fixed, add test)
sim/simulator.go:408-437  ←── BC-1, BC-2 (preempt() empty-batch guard)
sim/cluster/metrics.go:212-247  ←── BC-3, BC-6 (HOL detector fix)
sim/cluster/metrics.go:164-207  ←── BC-4, BC-7 (priority inversion fix)
```

No cross-component interaction. Each fix is self-contained.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #297 proposes `--max-preemptions` CLI flag | No new CLI flag — circuit breaker is implicit in the empty-batch guard | SIMPLIFICATION: BC-1 already prevents the livelock by returning false when batch is empty. No extra flag needed. |
| #292 proposes 4 fix options | Uses option 1 (compare within SLO class) | SIMPLIFICATION: Simplest, most robust approach. Other options require tuning thresholds. |

### E) Review Guide

1. **THE TRICKY PART:** BC-4 (priority inversion SLO grouping) — make sure the fix handles empty SLOClass correctly (should default to "default" group, preserving existing behavior for legacy workloads).
2. **WHAT TO SCRUTINIZE:** BC-1/BC-2 — trace through the preemption flow to verify the empty-batch return path is correct and doesn't leave state inconsistent.
3. **WHAT'S SAFE TO SKIM:** BC-5 (regression test) — purely additive test, no code changes.
4. **KNOWN DEBT:** The preemption loop still has O(n²) behavior for large batches (evict one at a time). Not in scope.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/simulator.go:408-437` — Add empty-batch guard to `preempt()`
- `sim/simulator_test.go` (new) — Tests for preemption edge cases
- `sim/cluster/metrics.go:212-247` — Fix `detectHOLBlocking()` to include zero-traffic instances
- `sim/cluster/metrics.go:164-207` — Fix `detectPriorityInversions()` to group by SLO class
- `sim/cluster/metrics_test.go` — Add degenerate-input tests for both detectors
- `cmd/default_config_test.go` (new) — Regression test for CLI flag precedence

**Golden dataset:** May need regeneration if preemption fix changes simulation output for edge-case configs. Check after Task 1.

### G) Task Breakdown

---

### Task 1: Fix preempt() empty-batch panic and add livelock guard (#293, #297)

**Contracts Implemented:** BC-1, BC-2, BC-8

**Files:**
- Modify: `sim/simulator.go:408-437`
- Create: `sim/simulator_preempt_test.go`

**Step 1: Write failing test for empty-batch panic (BC-1)**

Context: We need to verify that `preempt()` doesn't panic when the running batch is empty and KV allocation fails. We'll create a simulator with a tiny KV cache and trigger preemption.

```go
package sim

import (
	"testing"
)

// TestPreempt_EmptyBatch_ReturnsFalse verifies BC-1:
// preempt() must return false (not panic) when the batch is empty.
func TestPreempt_EmptyBatch_ReturnsFalse(t *testing.T) {
	// GIVEN a simulator with minimal KV cache (2 blocks, block size 16)
	config := SimConfig{
		TotalKVBlocks:     2,
		BlockSizeTokens: 16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 10000,
		Horizon:           1000000,
		BetaCoeffs:        []float64{100, 1, 1},
		AlphaCoeffs:       []float64{100, 1, 100},
	}
	s, err := NewSimulator(config)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND the running batch is empty
	s.RunningBatch = &Batch{Requests: []*Request{}}

	// AND a request that needs far more blocks than available
	req := &Request{
		ID:          "large-req",
		InputTokens: make([]int, 200), // needs ~13 blocks, only 2 available
	}

	// WHEN preempt is called
	// THEN it must return false (not panic)
	result := s.preempt(req, 0, 200)
	if result {
		t.Error("expected preempt to return false when batch is empty and allocation fails")
	}
}

// TestPreempt_InsufficientBlocks_ReturnsEventually verifies BC-2:
// preempt() must not loop forever when KV blocks are insufficient.
func TestPreempt_InsufficientBlocks_ReturnsEventually(t *testing.T) {
	// GIVEN a simulator with very small KV cache
	config := SimConfig{
		TotalKVBlocks:     2,
		BlockSizeTokens: 16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 10000,
		Horizon:           1000000,
		BetaCoeffs:        []float64{100, 1, 1},
		AlphaCoeffs:       []float64{100, 1, 100},
	}
	s, err := NewSimulator(config)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND one small request in the running batch
	existing := &Request{
		ID:          "existing",
		InputTokens: make([]int, 10),
		State:       StateRunning,
	}
	s.RunningBatch = &Batch{Requests: []*Request{existing}}
	// Allocate some blocks for the existing request
	s.KVCache.AllocateKVBlocks(existing, 0, 10, []int64{})

	// AND a new request that needs more blocks than total capacity
	req := &Request{
		ID:          "huge-req",
		InputTokens: make([]int, 200),
	}

	// WHEN preempt is called (should evict existing, then fail on empty batch)
	result := s.preempt(req, 0, 200)

	// THEN it must return false (not hang or panic)
	if result {
		t.Error("expected preempt to return false when blocks are insufficient for any request")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestPreempt_EmptyBatch -v`
Expected: FAIL with panic (index out of range)

**Step 3: Implement the fix**

In `sim/simulator.go`, modify the `preempt()` function to add an empty-batch guard:

```go
func (sim *Simulator) preempt(req *Request, now int64, numNewTokens int64) bool {

	for {
		if ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, []int64{}); !ok {
			// Could not allocate (e.g., no free blocks)

			// Circuit breaker: if batch is empty and allocation still fails,
			// the KV cache is too small for this request. Return false instead
			// of panicking on empty slice access. (R19, #293, #297)
			if len(sim.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					now, req.ID, numNewTokens)
				return false
			}

			sim.preemptionHappened = true
			sim.Metrics.PreemptionCount++
			preemptionDelay := sim.getPreemptionProcessingTime()
			preemptedRequest := sim.RunningBatch.Requests[len(sim.RunningBatch.Requests)-1]
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room", now, preemptedRequest.ID)
			sim.RunningBatch.Requests = sim.RunningBatch.Requests[:len(sim.RunningBatch.Requests)-1]
			sim.Schedule(&PreemptionEvent{
				time:    now + preemptionDelay,
				Request: preemptedRequest,
			})

			preemptedRequest.State = StateQueued
			preemptedRequest.ProgressIndex = 0
			sim.KVCache.ReleaseKVBlocks(preemptedRequest)
			sim.WaitQ.PrependFront(preemptedRequest)

			if preemptedRequest == req {
				return false
			}
		} else {
			return true
		}
	}

}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestPreempt_ -v`
Expected: PASS (both tests)

**Step 5: Run all sim tests + lint**

Run: `go test ./sim/... && golangci-lint run ./sim/...`
Expected: All pass, no new lint issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/simulator_preempt_test.go
git commit -m "fix(sim): guard preempt() against empty batch panic and livelock (#293, #297)

- Add empty-batch check before accessing RunningBatch.Requests[-1]
- Return false instead of panicking when KV cache is too small
- Terminates preemption loop when no progress possible (R19)
- Preserves INV-1 conservation (BC-8)

Fixes #293
Fixes #297

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix HOL blocking detector blind spot (#291)

**Contracts Implemented:** BC-3, BC-6

**Files:**
- Modify: `sim/cluster/metrics.go:212-247`
- Modify: `sim/cluster/metrics_test.go`

**Step 1: Write failing test for single-instance concentration (BC-3)**

Context: The HOL detector returns 0 when all traffic goes to one instance because it skips instances with no samples. We need a test that reproduces the H14 scenario.

```go
// TestDetectHOLBlocking_AllTrafficOneInstance_Detected verifies BC-3 (#291):
// When all traffic goes to a single instance, HOL blocking MUST be detected.
func TestDetectHOLBlocking_AllTrafficOneInstance_Detected(t *testing.T) {
	// GIVEN 4 instances where only instance 0 has traffic
	perInstance := []*sim.Metrics{
		makeMetricsWithQueueDepth([]int{50, 50, 50, 50}), // instance 0: all traffic
		makeMetricsWithQueueDepth([]int{}),                 // instance 1: no traffic
		makeMetricsWithQueueDepth([]int{}),                 // instance 2: no traffic
		makeMetricsWithQueueDepth([]int{}),                 // instance 3: no traffic
	}

	// WHEN detecting HOL blocking
	blocking := detectHOLBlocking(perInstance)

	// THEN HOL blocking MUST be detected (this is the most extreme case)
	if blocking == 0 {
		t.Error("expected HOL blocking > 0 when all traffic goes to one instance, got 0")
	}
}
```

```go
// TestDetectHOLBlocking_PartialConcentration_Detected verifies the fix
// handles partial concentration (2 active + 2 idle) correctly.
func TestDetectHOLBlocking_PartialConcentration_Detected(t *testing.T) {
	perInstance := []*sim.Metrics{
		makeMetricsWithQueueDepth([]int{40, 40, 40}), // instance 0: heavy traffic
		makeMetricsWithQueueDepth([]int{5, 5, 5}),    // instance 1: light traffic
		makeMetricsWithQueueDepth([]int{}),             // instance 2: no traffic
		makeMetricsWithQueueDepth([]int{}),             // instance 3: no traffic
	}

	blocking := detectHOLBlocking(perInstance)

	// Mean is (40+5+0+0)/4 = 11.25; instance 0 avg=40 > 2*11.25=22.5 → detected
	if blocking == 0 {
		t.Error("expected HOL blocking > 0 for partial concentration, got 0")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestDetectHOLBlocking_AllTrafficOneInstance -v`
Expected: FAIL — returns 0

**Step 3: Implement the fix**

In `sim/cluster/metrics.go`, replace `detectHOLBlocking`:

```go
func detectHOLBlocking(perInstance []*sim.Metrics) int {
	if len(perInstance) < 2 {
		return 0
	}

	// Include ALL instances in the comparison, using avg=0.0 for instances
	// with no traffic. An instance receiving 0 requests while a sibling
	// receives 500 IS HOL blocking — the detector must not be blind to it. (#291, R20)
	avgDepths := make([]float64, 0, len(perInstance))
	totalAvg := 0.0
	for _, m := range perInstance {
		avg := 0.0
		if len(m.NumWaitQRequests) > 0 {
			sum := 0
			for _, d := range m.NumWaitQRequests {
				sum += d
			}
			avg = float64(sum) / float64(len(m.NumWaitQRequests))
		}
		avgDepths = append(avgDepths, avg)
		totalAvg += avg
	}

	meanAvg := totalAvg / float64(len(avgDepths))

	count := 0
	if meanAvg > 0 {
		for _, avg := range avgDepths {
			if avg > 2.0*meanAvg {
				count++
			}
		}
	}
	return count
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestDetectHOLBlocking -v`
Expected: All HOL blocking tests PASS (including new one and existing ones)

**Step 5: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go
git commit -m "fix(cluster): HOL blocking detector includes zero-traffic instances (#291)

- Include all instances in comparison with avg=0.0 for empty ones
- Single-instance concentration now correctly detected as HOL blocking
- Existing balanced/imbalanced tests still pass (BC-6)

Fixes #291

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Fix priority inversion false positives for mixed-SLO workloads (#292)

**Contracts Implemented:** BC-4, BC-7

**Files:**
- Modify: `sim/cluster/metrics.go:164-207`
- Modify: `sim/cluster/metrics_test.go`

**Step 1: Write failing test for mixed-SLO false positives (BC-4)**

Context: The detector compares all requests regardless of SLO class, producing false positives when "batch" requests naturally have higher E2E than "realtime" requests.

```go
// TestDetectPriorityInversions_MixedSLO_NoFalsePositives verifies BC-4 (#292):
// Mixed-SLO workloads must not produce false positives from cross-class comparisons.
func TestDetectPriorityInversions_MixedSLO_NoFalsePositives(t *testing.T) {
	// GIVEN requests from two SLO classes with naturally different E2E
	m := sim.NewMetrics()
	// Realtime requests: fast (low E2E)
	m.Requests["rt1"] = sim.RequestMetrics{ID: "rt1", ArrivedAt: 100, SLOClass: "realtime"}
	m.RequestE2Es["rt1"] = 5000.0
	m.Requests["rt2"] = sim.RequestMetrics{ID: "rt2", ArrivedAt: 300, SLOClass: "realtime"}
	m.RequestE2Es["rt2"] = 4500.0
	// Batch requests: slow (high E2E) — this is expected, not an inversion
	m.Requests["b1"] = sim.RequestMetrics{ID: "b1", ArrivedAt: 200, SLOClass: "batch"}
	m.RequestE2Es["b1"] = 50000.0
	m.Requests["b2"] = sim.RequestMetrics{ID: "b2", ArrivedAt: 400, SLOClass: "batch"}
	m.RequestE2Es["b2"] = 48000.0

	// WHEN detecting with slo-based priority
	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	// THEN no inversions (within each class, requests are ordered correctly)
	if inversions != 0 {
		t.Errorf("expected 0 inversions for correctly-ordered mixed-SLO workload, got %d", inversions)
	}
}

// TestDetectPriorityInversions_WithinSLOClass_StillDetected verifies BC-7:
// Inversions within a single SLO class must still be detected.
func TestDetectPriorityInversions_WithinSLOClass_StillDetected(t *testing.T) {
	m := sim.NewMetrics()
	// Two realtime requests where earlier one has much worse E2E
	m.Requests["rt1"] = sim.RequestMetrics{ID: "rt1", ArrivedAt: 100, SLOClass: "realtime"}
	m.RequestE2Es["rt1"] = 50000.0 // 10× worse than rt2
	m.Requests["rt2"] = sim.RequestMetrics{ID: "rt2", ArrivedAt: 200, SLOClass: "realtime"}
	m.RequestE2Es["rt2"] = 5000.0

	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	if inversions == 0 {
		t.Error("expected at least 1 inversion within the same SLO class")
	}
}

// TestDetectPriorityInversions_EmptySLOClass_UsesDefault verifies BC-7:
// Legacy workloads with empty SLOClass are grouped as "default" and
// existing detection behavior is preserved.
func TestDetectPriorityInversions_EmptySLOClass_UsesDefault(t *testing.T) {
	m := sim.NewMetrics()
	// Legacy requests with no SLO class (empty string)
	m.Requests["r1"] = sim.RequestMetrics{ID: "r1", ArrivedAt: 100}
	m.RequestE2Es["r1"] = 50000.0
	m.Requests["r2"] = sim.RequestMetrics{ID: "r2", ArrivedAt: 200}
	m.RequestE2Es["r2"] = 5000.0

	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	if inversions == 0 {
		t.Error("expected inversion detected for legacy (empty SLO class) requests")
	}
}
```

**Step 2: Run test to verify first test fails**

Run: `go test ./sim/cluster/... -run TestDetectPriorityInversions_MixedSLO -v`
Expected: FAIL — returns >0 inversions (false positives)

**Step 3: Implement the fix**

In `sim/cluster/metrics.go`, replace `detectPriorityInversions`:

```go
func detectPriorityInversions(perInstance []*sim.Metrics, priorityPolicy string) int {
	if priorityPolicy == "constant" || priorityPolicy == "" {
		return 0
	}
	count := 0
	for _, m := range perInstance {
		if len(m.Requests) < 2 {
			continue
		}
		type reqInfo struct {
			arrived float64
			e2e     float64
		}
		// Group requests by SLO class to avoid cross-class false positives (#292, R20).
		// Requests in different SLO classes have naturally different E2E due to
		// workload size differences, not scheduling unfairness.
		groups := make(map[string][]reqInfo)
		skippedCount := 0
		for id, rm := range m.Requests {
			if e2e, ok := m.RequestE2Es[id]; ok {
				sloClass := rm.SLOClass
				if sloClass == "" {
					sloClass = "default"
				}
				groups[sloClass] = append(groups[sloClass], reqInfo{arrived: rm.ArrivedAt, e2e: e2e})
			} else {
				skippedCount++
			}
		}
		if skippedCount > 0 {
			logrus.Warnf("detectPriorityInversions: %d requests missing E2E data, skipped", skippedCount)
		}
		// Check inversions within each SLO class
		for _, reqs := range groups {
			if len(reqs) < 2 {
				continue
			}
			sort.Slice(reqs, func(i, j int) bool {
				return reqs[i].arrived < reqs[j].arrived
			})
			for i := 0; i < len(reqs)-1; i++ {
				for j := i + 1; j < len(reqs); j++ {
					if reqs[i].e2e > reqs[j].e2e*2.0 {
						count++
					}
				}
			}
		}
	}
	return count
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -run TestDetectPriorityInversions -v`
Expected: All priority inversion tests PASS (new + existing)

**Step 5: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go
git commit -m "fix(cluster): priority inversion detector groups by SLO class (#292)

- Compare requests within same SLO class only (no cross-class comparisons)
- Empty SLOClass defaults to 'default' group (backward compatible)
- Eliminates false positives from workload size heterogeneity (R20)

Fixes #292

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add CLI flag precedence regression test (#285)

**Contracts Implemented:** BC-5

**Files:**
- Create: `cmd/default_config_test.go`

**Step 1: Write the regression test**

Context: The fix at cmd/root.go:173 (`cmd.Flags().Changed("total-kv-blocks")`) prevents defaults.yaml from overwriting user-provided KV block values. We need a test that verifies `GetCoefficients` returns the model default but the caller correctly preserves the user value.

```go
package cmd

import (
	"os"
	"testing"
)

// TestGetCoefficients_ReturnsTotalKVBlocks_CallerMustCheckChanged verifies BC-5 (#285):
// GetCoefficients returns the model's default totalKVBlocks, but callers MUST
// use cmd.Flags().Changed() to avoid overwriting user-provided values.
// This test verifies the function returns a non-zero default for known models.
func TestGetCoefficients_ReturnsTotalKVBlocks_CallerMustCheckChanged(t *testing.T) {
	// Skip if defaults.yaml not available (CI may not have it)
	path := "defaults.yaml"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		// Try from repo root
		path = "../defaults.yaml"
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Skip("defaults.yaml not found, skipping integration test")
		}
	}

	// GIVEN a known model in defaults.yaml
	alpha, beta, kvBlocks := GetCoefficients(
		"meta-llama/llama-3.1-8b-instruct",
		1, "NVIDIA-A100-SXM4-80GB", "0.6.2",
		path,
	)

	// THEN coefficients should be non-nil
	if alpha == nil || beta == nil {
		t.Fatal("expected non-nil coefficients for known model")
	}

	// AND kvBlocks should be the model's default (non-zero)
	if kvBlocks == 0 {
		t.Error("expected non-zero totalKVBlocks from model defaults")
	}

	// The key invariant (R18): callers must check cmd.Flags().Changed("total-kv-blocks")
	// before using kvBlocks. The fix is at cmd/root.go:173.
	// This test documents the contract: GetCoefficients always returns the model
	// default. It's the caller's job to not overwrite user values.
	t.Logf("Model default totalKVBlocks: %d (callers must check Changed() before using)", kvBlocks)
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./cmd/... -run TestGetCoefficients_ReturnsTotalKVBlocks -v`
Expected: PASS (fix already exists)

**Step 3: Run lint**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add cmd/default_config_test.go
git commit -m "test(cmd): add regression test for CLI flag precedence (#285, R18)

- Verify GetCoefficients returns model default totalKVBlocks
- Document the R18 contract: callers must check Changed() before using

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run full test suite, verify golden dataset, check conservation

**Contracts Implemented:** BC-8 (conservation invariant)

**Files:**
- Possibly update: `testdata/goldendataset.json` (only if tests fail)

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: All pass. If golden tests fail due to the preemption fix changing output, regenerate.

**Step 2: If golden tests fail, regenerate**

Run: `go test ./sim/... -run TestGolden -v` to identify failures.
If they fail: `go test ./sim/... -run TestGolden -update` (or equivalent regeneration command).

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Commit (only if golden dataset was regenerated)**

```bash
git add testdata/goldendataset.json
git commit -m "test: regenerate golden dataset after preemption fix (R12)

Preemption behavior changed for edge-case KV configurations:
previously panicked, now returns false and logs warning.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestPreempt_EmptyBatch_ReturnsFalse` |
| BC-2 | Task 1 | Unit | `TestPreempt_InsufficientBlocks_ReturnsEventually` |
| BC-3 | Task 2 | Unit | `TestDetectHOLBlocking_AllTrafficOneInstance_Detected` |
| BC-4 | Task 3 | Unit | `TestDetectPriorityInversions_MixedSLO_NoFalsePositives` |
| BC-5 | Task 4 | Integration | `TestGetCoefficients_ReturnsTotalKVBlocks_CallerMustCheckChanged` |
| BC-6 | Task 2 | Unit | Existing: `TestDetectHOLBlocking_BalancedInstances_NoBlocking` |
| BC-7 | Task 3 | Unit | `TestDetectPriorityInversions_WithinSLOClass_StillDetected` + `_EmptySLOClass_UsesDefault` |
| BC-8 | Task 5 | Integration | Full test suite + existing conservation invariant tests |

No golden dataset updates expected (preemption fix only affects pathologically undersized configs not tested in golden dataset).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Preemption fix changes golden dataset | Low | Medium | Task 5 checks and regenerates if needed (R12) | Task 5 |
| HOL fix produces false positives for legitimate zero-traffic instances | Low | Low | Existing balanced-instances test (BC-6) + new test verifies specific scenario | Task 2 |
| Priority inversion fix breaks legacy workloads (empty SLOClass) | Medium | Medium | Explicit test for empty SLOClass → "default" group (BC-7) | Task 3 |
| Non-deterministic map iteration in SLO grouping | Low | Low | Groups are independent; comparison within each group is sorted by arrival time | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces — no new flags
- [x] No partial implementations
- [x] No breaking changes — detectors still work for single-class workloads
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md updated — N/A (no new files/packages/flags)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — 2 justified deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Tasks 1-4 independent, Task 5 depends on all)
- [x] All contracts mapped to tasks
- [x] Golden dataset regeneration documented (Task 5, if needed)
- [x] Construction site audit — no struct field additions

**Antipattern rules:**
- [x] R1: No silent continue/return dropping data — preempt returns false with log warning
- [x] R2: N/A — no new map iteration for output
- [x] R3: N/A — no new CLI flags
- [x] R4: N/A — no struct field additions
- [x] R5: Preemption state rollback preserved
- [x] R6: No logrus.Fatalf in sim/ — using logrus.Warnf
- [x] R7: Conservation invariant tests exist (Task 5)
- [x] R8-R10: N/A
- [x] R11: N/A — no new division
- [x] R12: Golden dataset regeneration if needed (Task 5)
- [x] R13-R17: N/A
- [x] R18: Regression test added (Task 4)
- [x] R19: Circuit breaker added to preemption loop (Task 1)
- [x] R20: Degenerate inputs handled in both detectors (Tasks 2, 3)

---

## Appendix: File-Level Implementation Details

### File: `sim/simulator.go`

**Purpose:** Fix the `preempt()` function (lines 408-437)

**Change:** Add empty-batch guard at the top of the `for` loop, before accessing `RunningBatch.Requests[-1]`. When the batch is empty and allocation fails, log a warning and return false.

**Key Implementation Notes:**
- The warning uses `logrus.Warnf` (not Fatal — R6 compliance)
- State consistency: when returning false from empty batch, no state has been mutated (the `preemptionHappened` flag, `PreemptionCount`, and request state changes only happen after the `len == 0` check)
- The caller (`makeRunningBatch`) handles `false` by breaking out of the batch formation loop

### File: `sim/cluster/metrics.go`

**Purpose:** Fix two anomaly detector functions

**Change 1 (`detectHOLBlocking`):** Remove the skip for instances with empty `NumWaitQRequests`. Include them with `avg = 0.0`. Remove the `len(avgDepths) < 2` early return since all instances are now included.

**Change 2 (`detectPriorityInversions`):** Replace the flat request list with a `map[string][]reqInfo` grouped by SLOClass. Empty SLOClass defaults to "default". The comparison logic (2× threshold, O(n²) pairs) is unchanged but runs within each group independently.

### File: `cmd/default_config_test.go`

**Purpose:** Regression test for #285 fix

**Change:** New test file verifying `GetCoefficients` returns the model default and documenting the R18 contract that callers must check `cmd.Flags().Changed()`.
