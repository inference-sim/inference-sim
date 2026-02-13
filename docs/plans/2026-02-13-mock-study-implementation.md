# Mock Study Implementation Plan

> **Note (v2.3):** This plan was executed before the v2.3 macro plan restructuring. References to "PR 6" for InstanceSnapshot now correspond to PR 4 in the v2.3 plan. See `2026-02-11-macro-implementation-plan-v2.md` for current PR mapping.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write 4 hand-coded routing policies in test to validate the ClusterSimulator API, identify missing observables, and produce findings for Phase 2 interface design.

**Architecture:** A single test file `sim/cluster/mock_study_test.go` containing a `runClusterWithPolicy` helper that replicates `ClusterSimulator.Run()` but accepts a routing callback. Four policies (round-robin, least-loaded, random, KV-aware) exercise the API and reveal gaps. A `WaitQueue.Len()` method is added to `sim/queue.go` since it's immediately needed and trivially missing.

**Tech Stack:** Go testing, existing `sim` and `sim/cluster` packages.

---

### Task 1: Add WaitQueue.Len() method

The mock study will immediately need queue depth. `WaitQueue` has no `Len()` method — its `queue` field is unexported, making it inaccessible from `package cluster`. This is a prerequisite.

**Files:**
- Modify: `sim/queue.go`
- Test: `sim/simulator_test.go` (verify indirectly via existing tests, no new test needed — Len() is trivial)

**Step 1: Add Len() to WaitQueue**

In `sim/queue.go`, add after the `Enqueue` method:

```go
// Len returns the number of requests in the wait queue.
func (wq *WaitQueue) Len() int {
	return len(wq.queue)
}
```

**Step 2: Verify tests still pass**

Run: `go test ./sim/... -count=1`
Expected: All existing tests PASS (no behavioral change)

**Step 3: Commit**

```bash
git add sim/queue.go
git commit -m "feat(sim): Add WaitQueue.Len() for queue depth observability"
```

---

### Task 2: Write the test helper and round-robin baseline

**Files:**
- Create: `sim/cluster/mock_study_test.go`

**Step 1: Write the test helper and round-robin policy**

Create `sim/cluster/mock_study_test.go` with:

```go
package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// routingPolicy decides which instance receives a request.
// Returns the index into the instances slice.
type routingPolicy func(req *sim.Request, instances []*InstanceSimulator, clock int64) int

// runClusterWithPolicy replicates ClusterSimulator.Run() but uses a custom
// routing policy instead of hardcoded round-robin.
// This is the core mock study harness.
func runClusterWithPolicy(
	config DeploymentConfig,
	workload *sim.GuideLLMConfig,
	policy routingPolicy,
) *sim.Metrics {
	// 1. Create instances (same as NewClusterSimulator)
	instances := make([]*InstanceSimulator, config.NumInstances)
	for idx := range instances {
		instances[idx] = NewInstanceSimulatorWithoutWorkload(
			InstanceID(fmt.Sprintf("instance_%d", idx)),
			config.Horizon,
			config.Seed,
			config.TotalKVBlocks,
			config.BlockSizeTokens,
			config.MaxRunningReqs,
			config.MaxScheduledTokens,
			config.LongPrefillTokenThreshold,
			config.BetaCoeffs,
			config.AlphaCoeffs,
			config.ModelConfig,
			config.HWConfig,
			config.Model,
			config.GPU,
			config.TP,
			config.Roofline,
		)
	}

	// 2. Generate requests (reuse ClusterSimulator's generation for parity)
	tmpCS := NewClusterSimulator(config, workload, "")
	requests := tmpCS.generateRequests()

	// 3. Dispatch via policy
	for _, req := range requests {
		idx := policy(req, instances, req.ArrivalTime)
		instances[idx].InjectRequest(req)
	}
	for _, inst := range instances {
		inst.SetRequestRate(workload.Rate)
	}

	// 4. Shared-clock event loop (identical to ClusterSimulator.Run)
	for {
		earliestTime := int64(math.MaxInt64)
		earliestIdx := -1
		for idx, inst := range instances {
			if inst.HasPendingEvents() {
				t := inst.PeekNextEventTime()
				if t < earliestTime {
					earliestTime = t
					earliestIdx = idx
				}
			}
		}
		if earliestIdx == -1 {
			break
		}
		if earliestTime > config.Horizon {
			break
		}
		instances[earliestIdx].ProcessNextEvent()
	}

	// 5. Finalize and aggregate
	for _, inst := range instances {
		inst.Finalize()
	}
	return aggregateInstanceMetrics(instances, workload)
}

// aggregateInstanceMetrics merges metrics from all instances.
// Same logic as ClusterSimulator.aggregateMetrics().
func aggregateInstanceMetrics(instances []*InstanceSimulator, workload *sim.GuideLLMConfig) *sim.Metrics {
	merged := sim.NewMetrics()
	for _, inst := range instances {
		m := inst.Metrics()
		merged.CompletedRequests += m.CompletedRequests
		merged.TotalInputTokens += m.TotalInputTokens
		merged.TotalOutputTokens += m.TotalOutputTokens
		merged.TTFTSum += m.TTFTSum
		merged.ITLSum += m.ITLSum
		if m.SimEndedTime > merged.SimEndedTime {
			merged.SimEndedTime = m.SimEndedTime
		}
		merged.KVBlocksUsed += m.KVBlocksUsed
		if m.PeakKVBlocksUsed > merged.PeakKVBlocksUsed {
			merged.PeakKVBlocksUsed = m.PeakKVBlocksUsed
		}
		for k, v := range m.RequestTTFTs {
			merged.RequestTTFTs[k] = v
		}
		for k, v := range m.RequestE2Es {
			merged.RequestE2Es[k] = v
		}
		for k, v := range m.RequestITLs {
			merged.RequestITLs[k] = v
		}
		for k, v := range m.RequestSchedulingDelays {
			merged.RequestSchedulingDelays[k] = v
		}
		for k, v := range m.RequestCompletionTimes {
			merged.RequestCompletionTimes[k] = v
		}
		for k, v := range m.Requests {
			merged.Requests[k] = v
		}
		merged.AllITLs = append(merged.AllITLs, m.AllITLs...)
		merged.RequestStepCounters = append(merged.RequestStepCounters, m.RequestStepCounters...)
	}
	if workload != nil {
		merged.RequestRate = workload.Rate
	}
	return merged
}

// === Routing Policies ===

func roundRobinPolicy() routingPolicy {
	counter := 0
	return func(_ *sim.Request, instances []*InstanceSimulator, _ int64) int {
		idx := counter % len(instances)
		counter++
		return idx
	}
}
```

**Step 2: Write the baseline validation test**

Add to the same file:

```go
// TestMockStudy_RoundRobin_MatchesClusterSimulator validates the test harness
// by verifying round-robin policy produces identical metrics to ClusterSimulator.Run().
func TestMockStudy_RoundRobin_MatchesClusterSimulator(t *testing.T) {
	config := newTestDeploymentConfig(2)
	workload := newTestWorkload(50)

	// Reference: ClusterSimulator
	cs := NewClusterSimulator(config, workload, "")
	cs.Run()
	ref := cs.AggregatedMetrics()

	// Mock study harness
	got := runClusterWithPolicy(config, workload, roundRobinPolicy())

	// Exact match on integer metrics
	if got.CompletedRequests != ref.CompletedRequests {
		t.Errorf("CompletedRequests: got %d, want %d", got.CompletedRequests, ref.CompletedRequests)
	}
	if got.TotalInputTokens != ref.TotalInputTokens {
		t.Errorf("TotalInputTokens: got %d, want %d", got.TotalInputTokens, ref.TotalInputTokens)
	}
	if got.TotalOutputTokens != ref.TotalOutputTokens {
		t.Errorf("TotalOutputTokens: got %d, want %d", got.TotalOutputTokens, ref.TotalOutputTokens)
	}
	if got.SimEndedTime != ref.SimEndedTime {
		t.Errorf("SimEndedTime: got %d, want %d", got.SimEndedTime, ref.SimEndedTime)
	}
	if got.TTFTSum != ref.TTFTSum {
		t.Errorf("TTFTSum: got %d, want %d", got.TTFTSum, ref.TTFTSum)
	}
	if got.ITLSum != ref.ITLSum {
		t.Errorf("ITLSum: got %d, want %d", got.ITLSum, ref.ITLSum)
	}
}
```

**Step 3: Run test to verify harness parity**

Run: `go test ./sim/cluster/... -run TestMockStudy_RoundRobin -v -count=1`
Expected: PASS — metrics match exactly

**Step 4: Commit**

```bash
git add sim/cluster/mock_study_test.go
git commit -m "test(cluster): Add mock study harness with round-robin baseline"
```

---

### Task 3: Add least-loaded and random policies

**Files:**
- Modify: `sim/cluster/mock_study_test.go`

**Step 1: Add the three remaining policies**

Append to `mock_study_test.go`:

```go
// leastLoadedPolicy routes to the instance with fewest queued+running requests.
// OBSERVABLE GAP: Must reach through inst.sim to access WaitQ and RunningBatch.
func leastLoadedPolicy() routingPolicy {
	return func(_ *sim.Request, instances []*InstanceSimulator, _ int64) int {
		bestIdx := 0
		bestLoad := math.MaxInt64
		for idx, inst := range instances {
			// GAP: WaitQ.Len() requires the method we added in Task 1.
			// Without it, queue depth is completely invisible from outside package sim.
			queueDepth := inst.sim.WaitQ.Len()
			// GAP: RunningBatch.Requests is exported, but only accessible via inst.sim
			// (unexported field on InstanceSimulator). InstanceSimulator has no BatchSize() accessor.
			batchSize := len(inst.sim.RunningBatch.Requests)
			load := queueDepth + batchSize
			if load < bestLoad {
				bestLoad = load
				bestIdx = idx
			}
		}
		return bestIdx
	}
}

// randomPolicy routes randomly using a seeded RNG.
func randomPolicy(seed int64) routingPolicy {
	rng := rand.New(rand.NewSource(seed))
	return func(_ *sim.Request, instances []*InstanceSimulator, _ int64) int {
		return rng.Intn(len(instances))
	}
}

// kvAwarePolicy routes to the instance with the most free KV blocks.
// OBSERVABLE GAP: Must reach through inst.sim to access KVCache state.
func kvAwarePolicy() routingPolicy {
	return func(_ *sim.Request, instances []*InstanceSimulator, _ int64) int {
		bestIdx := 0
		bestFree := int64(-1)
		for idx, inst := range instances {
			// GAP: KVCache is exported on Simulator, but only reachable via inst.sim.
			// InstanceSimulator has no KV utilization accessor.
			freeBlocks := inst.sim.KVCache.TotalBlocks - inst.sim.KVCache.UsedBlockCnt
			if freeBlocks > bestFree {
				bestFree = freeBlocks
				bestIdx = idx
			}
		}
		return bestIdx
	}
}
```

**Step 2: Verify compilation**

Run: `go build ./sim/cluster/...`
Expected: BUILD SUCCESS (no errors)

**Step 3: Commit**

```bash
git add sim/cluster/mock_study_test.go
git commit -m "test(cluster): Add least-loaded, random, and KV-aware routing policies"
```

---

### Task 4: Add comparison test and observable gap test

**Files:**
- Modify: `sim/cluster/mock_study_test.go`

**Step 1: Add the comparison test**

Append to `mock_study_test.go`:

```go
// policyResult captures aggregated metrics from a policy run for comparison.
type policyResult struct {
	name              string
	completedRequests int
	meanTTFT          float64
	p99TTFT           float64
	meanE2E           float64
	throughput        float64 // completed requests / sim duration (seconds)
}

func computePolicyResult(name string, m *sim.Metrics) policyResult {
	pr := policyResult{
		name:              name,
		completedRequests: m.CompletedRequests,
	}
	if m.CompletedRequests > 0 && m.SimEndedTime > 0 {
		pr.throughput = float64(m.CompletedRequests) / (float64(m.SimEndedTime) / 1e6)
	}

	// Mean TTFT
	if len(m.RequestTTFTs) > 0 {
		sum := 0.0
		for _, v := range m.RequestTTFTs {
			sum += v
		}
		pr.meanTTFT = sum / float64(len(m.RequestTTFTs)) / 1e3 // to ms
	}

	// P99 TTFT
	if len(m.RequestTTFTs) > 0 {
		vals := make([]float64, 0, len(m.RequestTTFTs))
		for _, v := range m.RequestTTFTs {
			vals = append(vals, v)
		}
		sort.Float64s(vals)
		idx := int(float64(len(vals)) * 0.99)
		if idx >= len(vals) {
			idx = len(vals) - 1
		}
		pr.p99TTFT = vals[idx] / 1e3
	}

	// Mean E2E
	if len(m.RequestE2Es) > 0 {
		sum := 0.0
		for _, v := range m.RequestE2Es {
			sum += v
		}
		pr.meanE2E = sum / float64(len(m.RequestE2Es)) / 1e3
	}

	return pr
}

// TestMockStudy_RoutingPolicies_CompareMetrics runs all 4 policies and logs comparison.
func TestMockStudy_RoutingPolicies_CompareMetrics(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	policies := []struct {
		name   string
		policy routingPolicy
	}{
		{"round-robin", roundRobinPolicy()},
		{"least-loaded", leastLoadedPolicy()},
		{"random", randomPolicy(42)},
		{"kv-aware", kvAwarePolicy()},
	}

	results := make([]policyResult, 0, len(policies))
	for _, p := range policies {
		m := runClusterWithPolicy(config, workload, p.policy)
		results = append(results, computePolicyResult(p.name, m))
	}

	// Log comparison table
	t.Logf("\n%-15s | %10s | %10s | %10s | %10s | %10s",
		"Policy", "Completed", "MeanTTFT", "P99TTFT", "MeanE2E", "Throughput")
	t.Logf("%-15s-+-%10s-+-%10s-+-%10s-+-%10s-+-%10s",
		"---------------", "----------", "----------", "----------", "----------", "----------")
	for _, r := range results {
		t.Logf("%-15s | %10d | %8.2fms | %8.2fms | %8.2fms | %8.2f/s",
			r.name, r.completedRequests, r.meanTTFT, r.p99TTFT, r.meanE2E, r.throughput)
	}

	// Sanity: all policies should complete all requests with generous resources
	for _, r := range results {
		if r.completedRequests != 100 {
			t.Errorf("policy %s: completed %d requests, want 100", r.name, r.completedRequests)
		}
	}
}
```

**Step 2: Add the observable gaps test**

Append to `mock_study_test.go`:

```go
// TestMockStudy_ObservableGaps documents which fields the routing policies
// needed to access via unexported inst.sim reach-through.
// This test serves as the findings generator for the mock study.
func TestMockStudy_ObservableGaps(t *testing.T) {
	config := newTestDeploymentConfig(2)
	workload := newTestWorkload(10)

	// Create instances the same way the harness does
	instances := make([]*InstanceSimulator, config.NumInstances)
	for idx := range instances {
		instances[idx] = NewInstanceSimulatorWithoutWorkload(
			InstanceID(fmt.Sprintf("instance_%d", idx)),
			config.Horizon, config.Seed,
			config.TotalKVBlocks, config.BlockSizeTokens,
			config.MaxRunningReqs, config.MaxScheduledTokens,
			config.LongPrefillTokenThreshold,
			config.BetaCoeffs, config.AlphaCoeffs,
			config.ModelConfig, config.HWConfig,
			config.Model, config.GPU, config.TP, config.Roofline,
		)
	}

	// Document what CANNOT be accessed via InstanceSimulator's public API:
	t.Log("=== OBSERVABLE GAPS (fields accessed via inst.sim) ===")

	inst := instances[0]

	// 1. Queue depth
	_ = inst.sim.WaitQ.Len()
	t.Log("GAP 1: WaitQ.Len() — required inst.sim.WaitQ reach-through")
	t.Log("  → InstanceSimulator needs: QueueDepth() int")

	// 2. Running batch size
	_ = len(inst.sim.RunningBatch.Requests)
	t.Log("GAP 2: RunningBatch size — required inst.sim.RunningBatch.Requests reach-through")
	t.Log("  → InstanceSimulator needs: BatchSize() int")

	// 3. KV cache utilization
	_ = inst.sim.KVCache.TotalBlocks
	_ = inst.sim.KVCache.UsedBlockCnt
	t.Log("GAP 3: KV utilization — required inst.sim.KVCache.{TotalBlocks,UsedBlockCnt} reach-through")
	t.Log("  → InstanceSimulator needs: KVUtilization() float64")

	// 4. KV cache prefix hash state (for affinity routing)
	_ = inst.sim.KVCache.HashToBlock
	t.Log("GAP 4: Prefix cache state — required inst.sim.KVCache.HashToBlock reach-through")
	t.Log("  → InstanceSnapshot needs: CacheHitRate float64 or HasPrefix(hash) bool")

	// 5. In-flight request count (running requests)
	_ = inst.sim.RunningBatch.Requests
	t.Log("GAP 5: In-flight request details — required inst.sim.RunningBatch.Requests reach-through")
	t.Log("  → InstanceSimulator needs: InFlightRequests() int")

	t.Log("")
	t.Log("=== AWKWARD PATTERNS ===")
	t.Log("1. InstanceSimulator.sim is unexported — policies in package cluster can reach through,")
	t.Log("   but policies in a separate sim/policy/ package CANNOT. This makes the current API")
	t.Log("   insufficient for the planned package layout.")
	t.Log("2. WaitQueue had no Len() method at all — added in Task 1 as prerequisite.")
	t.Log("3. No snapshot/point-in-time read — policies see mutable state that may change")
	t.Log("   between observation and routing decision. InstanceSnapshot (plan) solves this.")
	t.Log("4. Metrics() returns live pointer — stale reads possible if not synchronized.")
	t.Log("   Single-threaded sim makes this safe today, but the API doesn't express this guarantee.")

	t.Log("")
	t.Log("=== RECOMMENDED InstanceSnapshot FIELDS (for PR 6) ===")
	t.Log("  ID              InstanceID")
	t.Log("  QueueDepth      int        // WaitQ.Len()")
	t.Log("  BatchSize       int        // len(RunningBatch.Requests)")
	t.Log("  KVUtilization   float64    // UsedBlockCnt / TotalBlocks")
	t.Log("  InFlightRequests int       // len(RunningBatch.Requests) -- same as BatchSize for now")
	t.Log("  FreeKVBlocks    int64      // TotalBlocks - UsedBlockCnt")
	t.Log("  CacheHitRate    float64    // requires tracking (not currently computed)")
}
```

**Step 3: Run all mock study tests**

Run: `go test ./sim/cluster/... -run TestMockStudy -v -count=1`
Expected: All 3 tests PASS. Comparison table and gaps logged.

**Step 4: Commit**

```bash
git add sim/cluster/mock_study_test.go
git commit -m "test(cluster): Add policy comparison and observable gap tests"
```

---

### Task 5: Run tests, capture output, write findings

**Files:**
- Create: `docs/plans/2026-02-13-mock-study-findings.md`

**Step 1: Run mock study tests with verbose output**

Run: `go test ./sim/cluster/... -run TestMockStudy -v -count=1 2>&1`
Capture the output for the findings document.

**Step 2: Run full test suite to confirm no regressions**

Run: `go test ./... -count=1`
Expected: All tests PASS

**Step 3: Write findings document**

Create `docs/plans/2026-02-13-mock-study-findings.md` with the following structure (fill in actual values from test output):

```markdown
# Mock Study Findings

**Date:** 2026-02-13
**Status:** Complete
**Context:** Post-PR3 checkpoint per macro implementation plan v2.2

## 1. Observable Gaps

| # | Observable Needed | Current Access Path | Proposed API |
|---|---|---|---|
| 1 | Queue depth | `inst.sim.WaitQ.Len()` | `InstanceSimulator.QueueDepth() int` or `InstanceSnapshot.QueueDepth` |
| 2 | Running batch size | `len(inst.sim.RunningBatch.Requests)` | `InstanceSimulator.BatchSize() int` or `InstanceSnapshot.BatchSize` |
| 3 | KV utilization | `inst.sim.KVCache.{TotalBlocks,UsedBlockCnt}` | `InstanceSnapshot.KVUtilization float64` |
| 4 | Prefix cache state | `inst.sim.KVCache.HashToBlock` | `InstanceSnapshot.CacheHitRate float64` |
| 5 | In-flight request count | `len(inst.sim.RunningBatch.Requests)` | `InstanceSnapshot.InFlightRequests int` |

## 2. Awkward Patterns

1. **Package boundary problem:** `InstanceSimulator.sim` is unexported. Same-package test code can reach through, but `sim/policy/` package code cannot. The InstanceSnapshot abstraction (planned for PR 6) is essential — not optional.
2. **Missing WaitQueue.Len():** Had to add as prerequisite — the simplest observable was missing.
3. **No snapshot semantics:** Policies read mutable state. InstanceSnapshot provides point-in-time consistency.
4. **No cache hit rate tracking:** KV cache tracks block usage but not hit/miss rates. Needed for prefix-affinity routing.

## 3. Metric Comparison

[Table from test output — fill in after running]

## 4. Interface Adjustment Recommendations

### For PR 6 (RoutingPolicy + InstanceSnapshot)
- `InstanceSnapshot` must include at minimum: `QueueDepth`, `BatchSize`, `KVUtilization`, `FreeKVBlocks`, `InFlightRequests`
- Add `CacheHitRate` field (requires new tracking in KVCacheState)
- `InstanceSnapshot` should be constructed by ClusterSimulator before each routing decision (not by policy)

### For PR 4 (AdmissionPolicy)
- AdmissionPolicy needs cluster-wide load: sum of QueueDepth across instances
- RouterState.Global should include aggregate queue depth

### General
- Add `WaitQueue.Len()` to sim package (done in this mock study)
- Consider adding `InstanceSimulator.Snapshot() InstanceSnapshot` method in PR 6
```

**Step 4: Run linter**

Run: `golangci-lint run ./...`
Expected: No new warnings

**Step 5: Commit findings and all changes**

```bash
git add docs/plans/2026-02-13-mock-study-findings.md
git commit -m "docs: Add mock study findings with observable gaps and recommendations"
```

---

## Summary

| Task | Description | Est. |
|------|-------------|------|
| 1 | Add `WaitQueue.Len()` | 2 min |
| 2 | Test helper + round-robin baseline | 5 min |
| 3 | Least-loaded, random, KV-aware policies | 3 min |
| 4 | Comparison test + observable gap test | 5 min |
| 5 | Run tests, write findings document | 5 min |

**Total: ~20 minutes, 5 commits**
