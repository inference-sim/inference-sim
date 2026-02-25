# Promote Hypothesis Findings to Go Tests (#378, #379) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CI regression tests that codify two deterministic findings from hypothesis experiments H25 and H26 — protecting the event pipeline's causal ordering and full-stack conservation invariant.

**The problem today:** Hypothesis experiments H25 (full-stack conservation) and H26 (admission latency causal ordering) proved deterministic properties of the simulator via shell scripts and Python analyzers. These findings live only in `hypotheses/` directories — they aren't enforced by CI. A regression could silently break these properties without any test failing.

**What this PR adds:**
1. **Admission latency causal ordering test** (#378) — verifies that configuring `--admission-latency L` produces an exact additive offset of L/1000 ms in both TTFT and E2E under low load with constant tokens. Regression protection for the cluster event pipeline's timestamp chain.
2. **Full-stack conservation test** (#379) — verifies INV-1 holds when all policy modules run simultaneously (weighted routing + token-bucket admission + priority-FCFS scheduling), including under moderate preemption pressure with constrained KV blocks.

**Why this matters:** These tests promote empirical findings into contractual guarantees enforced by CI. They protect critical simulator properties (event pipeline ordering, request conservation across the full policy stack) that would otherwise only be checked by manual hypothesis reruns.

**Architecture:** Both tests are added to `sim/cluster/cluster_test.go`, following the existing pattern of cluster-level invariant tests (e.g., `TestClusterSimulator_OverloadConservation`, `TestClusterSimulator_Conservation_PolicyMatrix`). No new types, interfaces, or packages — purely new test functions using existing `DeploymentConfig` and `ClusterSimulator` APIs.

**Source:** GitHub issues #378, #379 (promotion label)

**Closes:** Fixes #378, fixes #379

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds two test functions to `sim/cluster/cluster_test.go`:

1. **`TestClusterSimulator_AdmissionLatency_ExactOffset`** — runs the cluster with `AdmissionLatency=0` and `AdmissionLatency=10000` (10ms), asserts the mean TTFT and E2E deltas are exactly 10.0ms (within 0.1ms tolerance). Uses constant tokens and low rate to eliminate queueing confounds. Promoted from H26 experiment (#372).

2. **`TestClusterSimulator_FullStackConservation`** — runs the full policy stack (weighted routing with prefix-affinity/queue-depth/kv-utilization, token-bucket admission, priority-FCFS scheduling) across two configs: (a) always-admit with ample KV (happy path), (b) always-admit with constrained KV to force preemptions (stress path). Asserts INV-1 conservation and pipeline conservation in both cases. Promoted from H25 experiment (#372).

No production code changes. No new types or interfaces. Fits alongside existing cluster invariant tests.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Admission Latency Exact Offset
- GIVEN a 4-instance cluster with constant input (128) and output (32) tokens, low rate (10 req/s), seed=42
- WHEN run with `AdmissionLatency=0` (baseline) and `AdmissionLatency=10000` (10ms)
- THEN the mean TTFT delta MUST be within 0.1ms of 10.0ms AND the mean E2E delta MUST be within 0.1ms of 10.0ms
- MECHANISM: `ClusterArrivalEvent.Execute()` adds `admissionLatency` to the event timestamp at `cluster_event.go:89`. Since `ArrivalTime` is set at generation (never modified), all metrics computed as `now - ArrivalTime` include the admission delay as an exact additive offset.

BC-2: Admission Latency Linearity
- GIVEN the same setup as BC-1 but with a third config: `AdmissionLatency=50000` (50ms)
- WHEN all three configs run
- THEN the E2E delta ratio (50ms config / 10ms config) MUST equal 5.0 (within 0.01 tolerance)
- MECHANISM: The admission latency is a simple timestamp offset — no interaction effects, no overhead.

BC-3: Full-Stack Conservation (Happy Path)
- GIVEN 4 instances, 100 requests at rate=2000/s, weighted routing (prefix-affinity:3,queue-depth:2,kv-utilization:2), always-admit, priority-FCFS scheduling, ample KV blocks, infinite horizon, seed=42
- WHEN the simulation completes
- THEN `completed + still_queued + still_running == len(Requests)` (INV-1 map-based conservation) AND `completed == 100` (all requests finish under infinite horizon with ample resources)
- MECHANISM: Request lifecycle tracking is correct through all policy modules — every request is either completed, queued, or running at sim end.

BC-4: Full-Stack Conservation Under Preemption
- GIVEN the same setup as BC-3 but with `TotalKVBlocks=150, BlockSizeTokens=16` (constrained KV to trigger preemptions — at rate=2000/s, batch formation schedules ~12 prefill requests per step, each needing ~10 blocks = ~120 blocks, plus decode requests accumulating blocks, exceeding 150 capacity)
- WHEN the simulation completes
- THEN `completed + still_queued + still_running == len(Requests)` (INV-1 conservation) AND `PreemptionCount > 0` (the test actually exercises preemptions) AND `DroppedUnservable == 0` (no requests dropped — max single request = ceil((32+256)/16) = 18 blocks ≤ 150)
- MECHANISM: Preemptions re-queue requests but don't drop them; conservation holds through the preempt-requeue cycle.

BC-5: Full-Stack Pipeline Conservation (Token-Bucket)
- GIVEN 4 instances, 100 requests at rate=2000/s, token-bucket admission (cap=500, refill=300), weighted routing, priority-FCFS, seed=42
- WHEN the simulation completes
- THEN `len(Requests) + rejected == 100` (pipeline conservation: injected + rejected == total generated) AND `completed + still_queued + still_running == len(Requests)` (INV-1)
- MECHANISM: Token-bucket rejects requests that exceed the token budget; rejected requests are counted in `rejectedRequests` but not in the Requests map.

**Negative Contracts:**

NC-1: No Panics Under Combined Load
- GIVEN any of the full-stack configurations (BC-3, BC-4, BC-5)
- WHEN the simulation runs
- THEN no panics occur (test does not recover from panic — Go test framework catches panics as failures)

### C) Component Interaction

```
No new components. Tests exercise existing components:

cluster_test.go (new tests)
    ├── Uses: DeploymentConfig (deployment.go)
    ├── Uses: ClusterSimulator.Run() (cluster.go)
    ├── Uses: ClusterSimulator.AggregatedMetrics() (cluster.go)
    ├── Uses: ClusterSimulator.RejectedRequests() (cluster.go)
    ├── Uses: ClusterSimulator.PerInstanceMetrics() (cluster.go)
    └── Uses: sim.Metrics (metrics.go)
```

No new state, no new interfaces, no API changes.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #378: "assert abs(E2E_mean_with_latency - E2E_mean_baseline - 10.0) < 0.1" | Also asserts TTFT delta and linearity (50ms/10ms = 5.0) | ADDITION: linearity check provides stronger regression protection per H26 FINDINGS |
| #379: "Tests at least 2 configs: always-admit and constrained KV" | Tests 3 configs: always-admit (happy), constrained KV (preemption), token-bucket (rejection) | ADDITION: token-bucket config exercises pipeline conservation (injected + rejected == total) |
| #379: "priority-FCFS" | Uses priority-FCFS with slo-based priority | CORRECTION: priority-FCFS requires a priority policy; slo-based is the standard pairing |
| #379: "full stack with tiered KV + tracing" | Omits tiered KV and tracing | SIMPLIFICATION: Tiered KV (cpu-blocks, offload-threshold) and tracing (trace-level, counterfactual-k) are orthogonal to conservation. Conservation is tested without them to isolate the routing+admission+scheduling stack. Tiered KV is already tested by `TestTieredKVCache_*` in sim/kv/. |
| #379: "multi-turn chat workload (5 rounds)" | Uses single-turn workload | SIMPLIFICATION: Multi-turn with context accumulation risks #349 cascading preemptions under constrained KV. Single-turn workload exercises the same conservation path without non-termination risk. |
| #379: "500 requests at 2000 req/s" | Uses 100 requests at 2000 req/s | SIMPLIFICATION: 100 requests is sufficient for conservation verification. Conservation is a structural invariant — it doesn't require large N. |
| #379 H25 Config C: "conservation under constrained KV" | BC-4 is a NEW test, not a direct promotion | ADDITION: H25 Config C (100 blocks) did NOT terminate due to #349 cascading preemptions and produced no conservation result. BC-4 designs a new constrained-KV configuration (150 blocks, single-turn) to fill this gap. |

### E) Review Guide

**The tricky part:** BC-1/BC-2 (admission latency) depend on computing mean TTFT/E2E from per-request maps. The `RequestTTFTs` and `RequestE2Es` map values are in ticks (microseconds). The assertion must convert to ms (divide by 1000) to compare against the expected 10.0ms delta.

**What to scrutinize:** (1) The tolerance values (0.1ms for BC-1, 0.01 for BC-2 ratio). These must be tight enough to catch regressions but loose enough to handle floating-point arithmetic. H26 showed exact 4+ decimal place agreement, so 0.1ms is generous. (2) The constrained-KV config (150 blocks per instance, 16 tokens/block) at rate=2000/s. At this rate, batch formation saturates with ~12 prefill requests per step, each needing ~10 blocks. Decode requests accumulate blocks across steps. Total KV pressure exceeds 150 blocks, triggering preemptions.

**What's safe to skim:** The full-stack conservation test (BC-3/BC-4/BC-5) follows the exact same pattern as `TestClusterSimulator_OverloadConservation` and `TestClusterSimulator_Conservation_PolicyMatrix`.

**Known debt:** None introduced.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files:**
- Modify: `sim/cluster/cluster_test.go` (add 2 new test functions + 1 helper)

**Key decisions:**
- Both tests in the same file as existing cluster invariant tests (no new test file)
- Helper function `meanMapValues(m map[string]float64) float64` to compute mean from per-request maps
- Constant tokens (no stddev) to eliminate variance — this is the key insight from H26

### G) Task Breakdown

---

### Task 1: Add admission latency causal ordering test (BC-1, BC-2)

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `sim/cluster/cluster_test.go`

**Step 1: Write the test**

Context: This test promotes H26's finding that admission latency creates an exact additive offset in TTFT and E2E. We use constant tokens and low rate to eliminate all confounds.

```go
// meanMapValues computes the arithmetic mean of all values in a map.
// Panics on empty map (test infrastructure — should never be empty).
func meanMapValues(m map[string]float64) float64 {
	if len(m) == 0 {
		panic("meanMapValues: empty map")
	}
	sum := 0.0
	for _, v := range m {
		sum += v
	}
	return sum / float64(len(m))
}

// TestClusterSimulator_AdmissionLatency_ExactOffset verifies that admission
// latency creates an exact additive offset in TTFT and E2E
// (promoted from H26 experiment, PR #372, issue #378):
// GIVEN constant token lengths, low rate (no queuing), and deterministic seed
// WHEN the cluster runs with AdmissionLatency=0, 10000 (10ms), and 50000 (50ms)
// THEN TTFT and E2E deltas MUST match the admission latency exactly (within 0.1ms)
// AND the linearity ratio (50ms/10ms) MUST equal 5.0 (within 0.01).
func TestClusterSimulator_AdmissionLatency_ExactOffset(t *testing.T) {
	const (
		numRequests  = 50
		numInstances = 4
		rateReqPerS  = 10.0
		inputTokens  = 128
		outputTokens = 32
	)

	// Constant tokens (zero stddev) eliminates variance.
	mkWorkload := func() *sim.GuideLLMConfig {
		return &sim.GuideLLMConfig{
			Rate:               rateReqPerS / 1e6,
			NumRequests:        numRequests,
			PrefixTokens:       0,
			PromptTokens:       inputTokens,
			PromptTokensStdDev: 0,
			PromptTokensMin:    inputTokens,
			PromptTokensMax:    inputTokens,
			OutputTokens:       outputTokens,
			OutputTokensStdDev: 0,
			OutputTokensMin:    outputTokens,
			OutputTokensMax:    outputTokens,
		}
	}

	runWithLatency := func(latencyUS int64) *sim.Metrics {
		config := newTestDeploymentConfig(numInstances)
		config.RoutingPolicy = "least-loaded"
		config.AdmissionLatency = latencyUS
		cs := NewClusterSimulator(config, mkWorkload(), "")
		mustRun(t, cs)
		return cs.AggregatedMetrics()
	}

	mA := runWithLatency(0)      // baseline
	mB := runWithLatency(10000)  // 10ms
	mC := runWithLatency(50000)  // 50ms

	// Compute mean TTFT and E2E (in ticks/microseconds), convert to ms
	ttftA := meanMapValues(mA.RequestTTFTs) / 1000.0
	ttftB := meanMapValues(mB.RequestTTFTs) / 1000.0
	ttftC := meanMapValues(mC.RequestTTFTs) / 1000.0

	e2eA := meanMapValues(mA.RequestE2Es) / 1000.0
	e2eB := meanMapValues(mB.RequestE2Es) / 1000.0
	e2eC := meanMapValues(mC.RequestE2Es) / 1000.0

	// BC-1: TTFT and E2E deltas must match admission latency (within 0.1ms)
	const tol = 0.1 // ms

	ttftDeltaB := ttftB - ttftA
	e2eDeltaB := e2eB - e2eA
	if math.Abs(ttftDeltaB-10.0) > tol {
		t.Errorf("BC-1 TTFT delta (10ms latency): got %.4f ms, want 10.0 ± %.1f ms", ttftDeltaB, tol)
	}
	if math.Abs(e2eDeltaB-10.0) > tol {
		t.Errorf("BC-1 E2E delta (10ms latency): got %.4f ms, want 10.0 ± %.1f ms", e2eDeltaB, tol)
	}

	ttftDeltaC := ttftC - ttftA
	e2eDeltaC := e2eC - e2eA
	if math.Abs(ttftDeltaC-50.0) > tol {
		t.Errorf("BC-1 TTFT delta (50ms latency): got %.4f ms, want 50.0 ± %.1f ms", ttftDeltaC, tol)
	}
	if math.Abs(e2eDeltaC-50.0) > tol {
		t.Errorf("BC-1 E2E delta (50ms latency): got %.4f ms, want 50.0 ± %.1f ms", e2eDeltaC, tol)
	}

	// BC-2: Linearity check — 50ms/10ms ratio must be 5.0
	if e2eDeltaB > 0 {
		ratio := e2eDeltaC / e2eDeltaB
		if math.Abs(ratio-5.0) > 0.01 {
			t.Errorf("BC-2 linearity: E2E delta ratio (50ms/10ms) = %.4f, want 5.0 ± 0.01", ratio)
		}
	} else {
		t.Error("BC-2: E2E delta for 10ms config is <= 0, cannot check linearity")
	}

	// Sanity: all requests completed in all configs
	if mA.CompletedRequests != numRequests {
		t.Errorf("baseline: completed %d, want %d", mA.CompletedRequests, numRequests)
	}
	if mB.CompletedRequests != numRequests {
		t.Errorf("10ms config: completed %d, want %d", mB.CompletedRequests, numRequests)
	}
	if mC.CompletedRequests != numRequests {
		t.Errorf("50ms config: completed %d, want %d", mC.CompletedRequests, numRequests)
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_AdmissionLatency_ExactOffset -v`
Expected: PASS (this is a promotion of a confirmed finding — the test codifies existing behavior)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "test(cluster): promote H26 admission latency finding to Go test (BC-1, BC-2)

- Add TestClusterSimulator_AdmissionLatency_ExactOffset
- Verifies admission latency creates exact additive offset in TTFT/E2E
- Linearity check: 50ms/10ms ratio = 5.0
- Fixes #378

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add full-stack conservation test (BC-3, BC-4, BC-5, NC-1)

**Contracts Implemented:** BC-3, BC-4, BC-5, NC-1

**Files:**
- Modify: `sim/cluster/cluster_test.go`

**Step 1: Write the test**

Context: This test promotes H25's finding that INV-1 conservation holds across the full policy stack. We test three configurations: happy path (ample resources), stress path (constrained KV forcing preemptions), and token-bucket path (admission rejections).

```go
// TestClusterSimulator_FullStackConservation verifies INV-1 conservation
// across the full policy stack: weighted routing + admission control +
// priority scheduling (promoted from H25 experiment, PR #372, issue #379):
// GIVEN weighted routing (prefix-affinity:3,queue-depth:2,kv-utilization:2),
//   priority-FCFS scheduling, and multiple admission/KV configurations
// WHEN the simulation completes
// THEN conservation holds: completed + still_queued + still_running == len(Requests)
// AND preemptions are triggered in the constrained-KV config (stress path exercised)
// AND pipeline conservation holds for token-bucket: len(Requests) + rejected == total.
func TestClusterSimulator_FullStackConservation(t *testing.T) {
	const (
		numRequests  = 100
		numInstances = 4
		rateReqPerS  = 2000.0 // High rate to saturate batch and create KV pressure
	)

	mkWorkload := func() *sim.GuideLLMConfig {
		return &sim.GuideLLMConfig{
			Rate:               rateReqPerS / 1e6,
			NumRequests:        numRequests,
			PrefixTokens:       32,
			PromptTokens:       128,
			PromptTokensStdDev: 32,
			PromptTokensMin:    32,
			PromptTokensMax:    256,
			OutputTokens:       64,
			OutputTokensStdDev: 16,
			OutputTokensMin:    16,
			OutputTokensMax:    128,
		}
	}

	mkFullStackConfig := func() DeploymentConfig {
		config := newTestDeploymentConfig(numInstances)
		config.RoutingPolicy = "weighted"
		config.RoutingScorerConfigs = sim.DefaultScorerConfigs()
		config.Scheduler = "priority-fcfs"
		config.PriorityPolicy = "slo-based"
		config.AdmissionPolicy = "always-admit"
		return config
	}

	t.Run("always-admit/ample-kv", func(t *testing.T) {
		// BC-3: Happy path — all modules active, ample resources
		config := mkFullStackConfig()
		cs := NewClusterSimulator(config, mkWorkload(), "")
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()
		injected := len(agg.Requests)

		// INV-1 conservation (map-based three-term)
		conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
		if conservation != injected {
			t.Errorf("INV-1: completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
				agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
		}

		// All requests complete under infinite horizon with ample resources
		if agg.CompletedRequests != numRequests {
			t.Errorf("expected all %d requests to complete, got %d", numRequests, agg.CompletedRequests)
		}

		// No requests dropped as unservable (ample KV)
		if agg.DroppedUnservable != 0 {
			t.Errorf("expected 0 DroppedUnservable with ample KV, got %d", agg.DroppedUnservable)
		}
	})

	t.Run("always-admit/constrained-kv", func(t *testing.T) {
		// BC-4: Stress path — constrained KV blocks force preemptions
		config := mkFullStackConfig()
		config.TotalKVBlocks = 150
		config.BlockSizeTokens = 16
		cs := NewClusterSimulator(config, mkWorkload(), "")
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()
		injected := len(agg.Requests)

		// INV-1 conservation (map-based three-term)
		conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
		if conservation != injected {
			t.Errorf("INV-1: completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
				agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
		}

		// Verify stress path is actually exercised: preemptions must occur
		if agg.PreemptionCount == 0 {
			t.Error("expected preemptions with constrained KV (150 blocks) at rate=2000, got 0 — test is not exercising the stress path")
		}

		// Verify no requests dropped as unservable (max single request = ceil((32+256)/16) = 18 blocks ≤ 150)
		if agg.DroppedUnservable != 0 {
			t.Errorf("expected 0 DroppedUnservable with 150 blocks (max request needs 18), got %d", agg.DroppedUnservable)
		}
	})

	t.Run("token-bucket", func(t *testing.T) {
		// BC-5: Pipeline conservation with admission rejections
		config := mkFullStackConfig()
		config.AdmissionPolicy = "token-bucket"
		config.TokenBucketCapacity = 500
		config.TokenBucketRefillRate = 300
		cs := NewClusterSimulator(config, mkWorkload(), "")
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()
		injected := len(agg.Requests)
		rejected := cs.RejectedRequests()

		// INV-1 conservation (map-based three-term)
		conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
		if conservation != injected {
			t.Errorf("INV-1: completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
				agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
		}

		// Pipeline conservation: injected + rejected == total generated
		if injected+rejected != numRequests {
			t.Errorf("pipeline conservation: injected(%d) + rejected(%d) = %d, want %d",
				injected, rejected, injected+rejected, numRequests)
		}

		// Sanity: token-bucket should reject some requests (not all admitted)
		if rejected == 0 {
			t.Error("expected some rejections with token-bucket(cap=500,refill=300) at rate=2000, got 0")
		}
	})
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_FullStackConservation -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `go test ./sim/cluster/... -v`
Expected: All tests PASS (no regressions)

**Step 4: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "test(cluster): promote H25 full-stack conservation to Go test (BC-3, BC-4, BC-5)

- Add TestClusterSimulator_FullStackConservation
- Three configs: ample KV (happy), constrained KV (preemption), token-bucket (rejection)
- Verifies INV-1 conservation across full policy stack
- Verifies pipeline conservation (injected + rejected == total)
- Fixes #379

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1     | Task 1 | Invariant | TestClusterSimulator_AdmissionLatency_ExactOffset (TTFT/E2E delta) |
| BC-2     | Task 1 | Invariant | TestClusterSimulator_AdmissionLatency_ExactOffset (linearity) |
| BC-3     | Task 2 | Invariant | TestClusterSimulator_FullStackConservation/always-admit/ample-kv |
| BC-4     | Task 2 | Invariant | TestClusterSimulator_FullStackConservation/always-admit/constrained-kv |
| BC-5     | Task 2 | Invariant | TestClusterSimulator_FullStackConservation/token-bucket |
| NC-1     | Task 2 | Invariant | All three subtests (panic = test failure) |

No golden dataset changes. No shared test infrastructure changes. Both tests are pure invariant tests (no golden values).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Constrained KV config causes non-termination (#349) | Low | High | Using 150 blocks with rate=2000/s (moderate pressure). Max single request = ceil((32+256)/16) = 18 blocks ≤ 150, so no drops. Non-multi-turn workload avoids context accumulation that caused #349. |
| Floating-point mean computation introduces rounding | Low | Low | 0.1ms tolerance is 1000x the expected precision (H26 showed 4+ decimal place match). |
| Token-bucket rejects all requests | Low | Medium | Rate=2000, cap=500, refill=300 — matches H25 Config A parameters. At 2000/s with ~160 tokens/req, token demand (320,000 tokens/s) far exceeds supply (300/s), so ~3-4 requests admitted from initial burst, rest rejected. Conservation check exercises pipeline with non-trivial rejection count. |

**Known debt:** None.

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions (only 1 small helper: `meanMapValues`)
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (newTestDeploymentConfig, mustRun)
- [x] CLAUDE.md — no changes needed (no new files/packages/flags)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY — no canonical sources modified
- [x] No deviations unresolved
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 adds helper, Task 2 can use it if needed but doesn't depend on it)
- [x] All contracts mapped to tasks
- [x] No golden dataset changes needed
- [x] No construction site audit needed (no struct fields added)
- [x] Not part of a macro plan

**Antipattern rules:**
- [x] R1: No silent data loss — tests don't drop data
- [x] R2: Map iteration for mean uses summation (order-independent) — no sorting needed
- [x] R4: No struct fields added
- [x] R6: No Fatalf in sim/ (tests only)
- [x] R7: Both tests are invariant tests (no golden values)

---

## Appendix: File-Level Implementation Details

**File: `sim/cluster/cluster_test.go`**

**Purpose:** Add two promoted hypothesis tests + one helper function.

**Changes:**
1. Add `meanMapValues` helper after the existing `sortedRequestMetrics` function (~line 1191)
2. Add `TestClusterSimulator_AdmissionLatency_ExactOffset` after the existing `TestClusterSimulator_SchedulerLiveness` function
3. Add `TestClusterSimulator_FullStackConservation` after the admission latency test

**Key notes:**
- `RequestTTFTs` and `RequestE2Es` values are in ticks (microseconds) — divide by 1000 for ms
- `newTestDeploymentConfig` sets `KVCacheConfig` with 10000 blocks and 16-token block size — sufficient for the happy path. The constrained-KV subtest overrides to 150 blocks at rate=2000/s (batch saturation creates KV pressure: ~12 prefill requests per step × ~10 blocks + accumulated decode blocks > 150 capacity). Max single request = ceil((32+256)/16) = 18 blocks ≤ 150, so no requests are dropped as unservable.
- `sim.DefaultScorerConfigs()` returns `[{prefix-affinity, 3}, {queue-depth, 2}, {kv-utilization, 2}]` — the llm-d parity profile.
- `meanMapValues` panics on empty map because it's test infrastructure — an empty map in this context means the simulation produced zero results, which is a test setup failure, not a graceful degradation scenario.
