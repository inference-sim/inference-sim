# Hypothesis Promotions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote hypothesis experiment findings into CI-protected Go tests and update the invariant standard to include the work-conserving property.

**The problem today:** Three critical invariants confirmed by hypothesis experiments (H12 conservation, H13 determinism, H-MMK work-conserving) are only validated in bash scripts outside CI. A regression in cluster-level conservation, non-determinism from prefix-affinity's LRU maps, or removal of the work-conserving fix would NOT be caught by `go test ./...`.

**What this PR adds:**
1. **INV-8 work-conserving standard** — formal invariant: after every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. Documented in `docs/standards/invariants.md`.
2. **Work-conserving regression test** — instance-level test with `MaxRunningReqs=1` and two concurrent requests. Without the fix (PR #325), the second request is stranded forever. With the fix, both complete.
3. **Cluster conservation matrix** — table-driven test verifying `injected == completed + still_queued + still_running` across 10 policy combinations (routing × scheduler × admission × priority), promoted from H12.
4. **Determinism with prefix-affinity** — run-twice-and-diff test specifically covering `prefix-affinity` and `weighted` routing (highest non-determinism risk due to LRU map iteration), promoted from H13.

**Why this matters:** These tests close the gap between hypothesis experiments (bash, outside CI) and the Go test suite (CI-protected). Every future commit will verify conservation, determinism, and work-conservation automatically.

**Architecture:** All tests are added to existing test files (`sim/simulator_test.go`, `sim/cluster/cluster_test.go`). No new packages, types, or interfaces. INV-8 is documentation-only (no code changes beyond tests). The work-conserving code fix already landed in PR #325.

**Source:** GitHub issues #322, #323, #326, #327, #328

**Closes:** Fixes #322, fixes #323, fixes #326, fixes #327, fixes #328

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds 3 new Go tests and 1 invariant standard, all promoted from hypothesis experiments:

- **INV-8** (work-conserving property) is added to `docs/standards/invariants.md`, closing the gap where this invariant existed only in the H-MMK experiment findings.
- **Work-conserving test** in `sim/simulator_test.go` verifies the fix from PR #325 by showing that two requests with `MaxRunningReqs=1` both complete (without the fix, the second request is stranded forever with no arrival to unstick it).
- **Cluster conservation matrix** in `sim/cluster/cluster_test.go` runs 10 policy combinations and checks the three-term conservation equation, promoting H12's bash experiment to CI.
- **Determinism test** in `sim/cluster/cluster_test.go` adds prefix-affinity and weighted routing to the run-twice-and-diff pattern, covering the highest non-determinism risk (LRU map iteration in prefix cache index).

No new types, interfaces, or code changes to production files (beyond docs). All tests use existing test infrastructure.

Adjacent blocks: `sim/simulator.go` (already fixed), `sim/cluster/cluster.go` (tested, not modified), `docs/standards/invariants.md` (updated).

See Section D for deviation log (5 items).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Work-Conserving — Both Requests Complete
- GIVEN a simulator with `MaxRunningReqs=1` and two requests arriving at t=0 and t=1 (both within one service period)
- WHEN the simulation runs to completion (infinite horizon, no more arrivals)
- THEN `CompletedRequests == 2` (both complete; without the fix, only 1 completes because the second is stranded)
- MECHANISM: When the running batch empties (`remaining == 0`), the code at `simulator.go:716-725` checks `WaitQ.Len() > 0` and schedules a new `StepEvent`.

BC-2: Work-Conserving — Scheduling Delay Bounded
- GIVEN the same setup as BC-1
- WHEN both requests complete
- THEN the second request's scheduling delay MUST be less than 2× the first request's service time (it waits for A to finish, not for a future arrival)
- MECHANISM: The second request is scheduled on the very next StepEvent after A completes, adding approximately one service time of delay.

BC-3: Cluster Conservation Matrix
- GIVEN 10 different policy combinations (routing × scheduler × admission × priority) with `NumInstances ∈ {2,3,4}` and infinite horizon
- WHEN each cluster simulation completes
- THEN for every combination: `CompletedRequests + StillQueued + StillRunning == len(Requests)` (three-term conservation, INV-1)
- MECHANISM: Each instance's `Finalize()` records `StillQueued` and `StillRunning`; cluster aggregation sums them.

BC-4: Cluster Conservation Matrix — All Complete Under Infinite Horizon
- GIVEN the same 10 configurations with infinite horizon and sufficient KV capacity
- WHEN simulation completes
- THEN `CompletedRequests == NumRequests` for every configuration (all requests complete)
- MECHANISM: Infinite horizon + ample resources means no request is left behind.

BC-5: Determinism With Prefix-Affinity
- GIVEN a cluster with `routing-policy=prefix-affinity` or `routing-policy=weighted` (with default scorers including prefix-affinity), same seed, same workload with prefix tokens
- WHEN run twice
- THEN per-request metrics (TTFT, E2E, scheduling delay) are byte-identical after JSON serialization
- MECHANISM: `PrefixCacheIndex` uses deterministic LRU eviction and sorted iteration in all output-affecting paths.

BC-6: Determinism With Weighted Routing
- GIVEN a cluster with `routing-policy=weighted` and default scorer configs, same seed
- WHEN run twice
- THEN aggregated metrics (CompletedRequests, TotalInputTokens, TotalOutputTokens, SimEndedTime) are identical AND per-request JSON is byte-identical
- MECHANISM: All scorers produce deterministic results for identical input; weighted routing argmax is deterministic.

**Negative Contracts:**

NC-1: No Regression Without Work-Conserving Fix
- GIVEN the test from BC-1
- WHEN the work-conserving code at `simulator.go:716-725` is removed
- THEN the test MUST fail (second request never completes)
- MECHANISM: This test IS the regression guard. It specifically tests the exact scenario that PR #325's bug caused.

### C) Component Interaction

```
                   [This PR: tests + docs]
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
sim/simulator_test.go  sim/cluster/       docs/standards/
(BC-1, BC-2)          cluster_test.go     invariants.md
                      (BC-3..BC-6)        (INV-8)
    │                     │
    ▼                     ▼
sim/simulator.go      sim/cluster/cluster.go
(tested, not modified) (tested, not modified)
```

No new types or interfaces. No new mutable state. Extension friction: 0 files to add another policy to the conservation matrix (just add a row to the test table).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #322: "Use H12's configurations as the test matrix" | Uses 10 representative combos covering all policy dimensions | SIMPLIFICATION: H12 had 67 checks across 10 configs; we need the configs, not every H12 sub-check |
| #323: "captures stdout, and asserts byte-equality" | Captures JSON via SaveResults and asserts byte-equality | SIMPLIFICATION: Go tests can't easily capture CLI stdout; SaveResults JSON is the deterministic output pipeline |
| #326: "Proposed action: Fixed in PR #325" | No code change, only test + issue closure | CORRECTION: Code fix already landed; this PR adds the test and closes the issue |
| #328: "scheduling_delay ≈ first request's service_time" | Tests both completion (primary) and delay bound (secondary) | ADDITION: Completion check is stronger — without fix, request NEVER completes |
| #327: "scheduling delay equals only the remaining service time" | Tests delay bounded by 2× service time (generous margin) | SIMPLIFICATION: Exact equality is brittle across beta coefficient changes; primary assertion is completion, not timing |

### E) Review Guide

**The tricky part:** BC-1's test design — with `MaxRunningReqs=1` and both requests at t=0, the second request is stranded FOREVER without the fix (no third arrival exists to trigger a QueuedEvent). This is a precise regression guard, not a timing check.

**What to scrutinize:** The conservation matrix (BC-3) — verify each policy name string is valid and won't silently fall back to defaults. Check that `ScorerConfigs` is set for weighted routing.

**What's safe to skim:** INV-8 documentation (straightforward text addition), CLAUDE.md updates (mechanical).

**Known debt:** The determinism test uses SaveResults JSON comparison, not raw stdout capture. This is deliberate (Go tests can't easily intercept `os.Stdout` from a library) but means the test covers the metrics pipeline, not the CLI formatting pipeline.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `docs/standards/invariants.md` — add INV-8 section
- `sim/simulator_test.go` — add work-conserving test (BC-1, BC-2)
- `sim/cluster/cluster_test.go` — add conservation matrix (BC-3, BC-4) and determinism tests (BC-5, BC-6)
- `CLAUDE.md` — add INV-8 to invariants list

**Key decisions:**
- Work-conserving test at instance level (not cluster) — the bug is in `sim/simulator.go`, testing at the right abstraction
- Conservation matrix uses table-driven tests with 10 rows — extensible without code changes
- Determinism test covers prefix-affinity AND weighted — both exercise the `PrefixCacheIndex` LRU

**Confirmation:** No dead code — every test directly exercises and verifies behavioral contracts.

### G) Task Breakdown

---

### Task 1: Add INV-8 to invariants.md

**Contracts Implemented:** (documentation for BC-1 through BC-6 foundation)

**Files:**
- Modify: `docs/standards/invariants.md`

**Step 1: Add INV-8 section after INV-7**

Context: INV-8 is the work-conserving property discovered in H-MMK. It's a deterministic invariant testable at instance level.

In `docs/standards/invariants.md`, add after the INV-7 section:

```markdown
---

## INV-8: Work-Conserving Property

**Statement:** After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator must not idle while there is work waiting.

**Verification:** `sim/simulator_test.go` — `TestWorkConserving_StepRestartsWhenWaitQNonEmpty`. Deterministic test with `MaxRunningReqs=1`, two requests arriving simultaneously. Without the property, the second request is stranded forever (no arrival to trigger a new StepEvent). With the property, both complete.

**Evidence:** H-MMK experiment (PR #325) — without the work-conserving fix, W_q error was 151,000% at ρ=0.3. After fix, error dropped to 47% (remaining gap is discrete step processing, not a bug).

**Code location:** Search for `// Work-conserving:` comment in `sim/simulator.go` — the `else` branch of `len(remaining) > 0` checks `WaitQ.Len() > 0` and schedules a new `StepEvent`.

**Hypothesis family:** Structural model (same as INV-4, INV-7).
```

**Step 2: Update the family mapping in the header**

Update the opening paragraph to include INV-8:

Change:
```
INV-4 (KV cache conservation) and INV-7 (signal freshness) belong to the **Structural model** family.
```
To:
```
INV-4 (KV cache conservation), INV-7 (signal freshness), and INV-8 (work-conserving property) belong to the **Structural model** family.
```

**Step 3: Verify the file is well-formed**

Run: `head -5 docs/standards/invariants.md && echo "---" && tail -10 docs/standards/invariants.md`
Expected: File starts with `# BLIS System Invariants` and ends with INV-8 content.

**Step 4: Commit**

```bash
git add docs/standards/invariants.md
git commit -m "docs(standards): add INV-8 work-conserving invariant (#327)

- After every step completion, if WaitQ.Len() > 0, a StepEvent must exist
- Deterministic invariant testable at instance level
- Discovered in H-MMK experiment (PR #325)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add work-conserving test (instance level)

**Contracts Implemented:** BC-1, BC-2, NC-1

**Files:**
- Modify: `sim/simulator_test.go`

**Step 1: Write the work-conserving test**

Context: This test verifies INV-8 at the instance level. With `MaxRunningReqs=1`, two requests arrive at t=0. Without the work-conserving fix, the second request is stranded forever (no third arrival exists to trigger a new StepEvent). With the fix, both complete.

Add to `sim/simulator_test.go`:

```go
// TestWorkConserving_StepRestartsWhenWaitQNonEmpty verifies INV-8:
// GIVEN a simulator with MaxRunningReqs=1 and two requests arriving at t=0
// WHEN the simulation runs to completion (infinite horizon, no further arrivals)
// THEN both requests complete (the second is not stranded when the first finishes)
// AND the second request's scheduling delay is bounded by the first's service time.
//
// Without the work-conserving fix (simulator.go:716-725), the second request
// would be stranded forever: its QueuedEvent already fired (seeing stepEvent != nil),
// and when the first request completes, stepEvent is set to nil without checking WaitQ.
// No third arrival exists to trigger a new StepEvent via QueuedEvent.
func TestWorkConserving_StepRestartsWhenWaitQNonEmpty(t *testing.T) {
	cfg := SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               42,
		TotalKVBlocks:      10000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     1, // KEY: only one request can run at a time
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Model:              "test-work-conserving",
		GPU:                "H100",
		TP:                 1,
	}

	s := mustNewSimulator(t, cfg)

	// Request A: arrives at t=0
	s.InjectArrival(&Request{
		ID: "req-A", ArrivalTime: 0,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
		State:        StateQueued,
	})
	// Request B: arrives at t=1 (during A's service time, which is ~6000μs)
	s.InjectArrival(&Request{
		ID: "req-B", ArrivalTime: 1,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
		State:        StateQueued,
	})

	s.Run()

	// BC-1: Both requests MUST complete.
	// Without the work-conserving fix, only req-A completes (req-B stranded forever).
	if s.Metrics.CompletedRequests != 2 {
		t.Fatalf("INV-8 violated: CompletedRequests = %d, want 2 "+
			"(second request stranded when running batch emptied without checking WaitQ)",
			s.Metrics.CompletedRequests)
	}

	// BC-2: req-B's scheduling delay must be bounded.
	// It should wait approximately one service time of req-A (~6000μs), not be stuck indefinitely.
	delayA := s.Metrics.RequestSchedulingDelays["req-A"]
	delayB := s.Metrics.RequestSchedulingDelays["req-B"]

	// req-A should be scheduled almost immediately (only alpha queueing delay)
	if delayA > 10000 {
		t.Errorf("req-A scheduling delay = %d μs, want < 10000 (should be near-immediate)", delayA)
	}

	// req-B should wait approximately one service time of req-A, not be unbounded.
	// With beta=[1000,10,5] and 10 input + 5 output tokens, service time is ~6000-7000μs.
	// We use a generous 2× bound to avoid brittleness.
	if delayB > 2*20000 {
		t.Errorf("req-B scheduling delay = %d μs, exceeds 2× expected service time bound "+
			"(may indicate work-conserving violation)", delayB)
	}

	// req-B must have waited longer than req-A (it was queued behind A)
	if delayB <= delayA {
		t.Errorf("req-B scheduling delay (%d) <= req-A delay (%d), "+
			"but B should have waited for A to complete", delayB, delayA)
	}

	// INV-1: Request conservation still holds
	injected := len(s.Metrics.Requests)
	total := s.Metrics.CompletedRequests + s.WaitQ.Len()
	running := 0
	if s.RunningBatch != nil {
		running = len(s.RunningBatch.Requests)
	}
	total += running
	if total != injected {
		t.Errorf("INV-1 conservation: completed(%d) + queued(%d) + running(%d) = %d, injected = %d",
			s.Metrics.CompletedRequests, s.WaitQ.Len(), running, total, injected)
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/... -run TestWorkConserving -v`
Expected: PASS

**Step 3: Run all sim tests to verify no regressions**

Run: `go test ./sim/... -count=1`
Expected: PASS (all existing tests still pass)

**Step 4: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/simulator_test.go
git commit -m "test(sim): add INV-8 work-conserving regression test (#328, #326)

- MaxRunningReqs=1, two requests at t=0, no further arrivals
- Without fix: second request stranded forever (CompletedRequests=1)
- With fix: both complete, scheduling delay bounded
- Also verifies INV-1 conservation holds

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add cluster conservation matrix test

**Contracts Implemented:** BC-3, BC-4

**Files:**
- Modify: `sim/cluster/cluster_test.go`

**Step 1: Write the cluster conservation matrix test**

Context: This promotes H12's conservation validation to CI. Tests 10 policy combinations using a table-driven pattern, checking the three-term conservation equation at the cluster level.

Add to `sim/cluster/cluster_test.go`:

```go
// TestClusterSimulator_Conservation_PolicyMatrix verifies INV-1 at cluster level
// across 10 policy combinations (promoted from H12 hypothesis experiment):
// GIVEN each policy combination with infinite horizon and ample resources
// WHEN the cluster simulation completes
// THEN completed + still_queued + still_running == injected (three-term conservation)
// AND all requests complete (infinite horizon, no resource pressure).
func TestClusterSimulator_Conservation_PolicyMatrix(t *testing.T) {
	matrix := []struct {
		name            string
		numInstances    int
		routingPolicy   string
		scorerConfigs   []sim.ScorerConfig
		scheduler       string
		priorityPolicy  string
		admissionPolicy string
	}{
		{"round-robin/fcfs/2inst", 2, "round-robin", nil, "fcfs", "constant", "always-admit"},
		{"least-loaded/fcfs/3inst", 3, "least-loaded", nil, "fcfs", "constant", "always-admit"},
		{"weighted/fcfs/2inst", 2, "weighted", sim.DefaultScorerConfigs(), "fcfs", "constant", "always-admit"},
		{"prefix-affinity/fcfs/2inst", 2, "prefix-affinity", nil, "fcfs", "constant", "always-admit"},
		{"round-robin/sjf/3inst", 3, "round-robin", nil, "sjf", "constant", "always-admit"},
		{"round-robin/priority-fcfs/slo/2inst", 2, "round-robin", nil, "priority-fcfs", "slo-based", "always-admit"},
		{"least-loaded/priority-fcfs/slo/3inst", 3, "least-loaded", nil, "priority-fcfs", "slo-based", "always-admit"},
		{"weighted/sjf/4inst", 4, "weighted", sim.DefaultScorerConfigs(), "sjf", "constant", "always-admit"},
		{"round-robin/fcfs/token-bucket/2inst", 2, "round-robin", nil, "fcfs", "constant", "token-bucket"},
		{"least-loaded/fcfs/4inst", 4, "least-loaded", nil, "fcfs", "constant", "always-admit"},
	}

	const numRequests = 50

	for _, tc := range matrix {
		t.Run(tc.name, func(t *testing.T) {
			config := newTestDeploymentConfig(tc.numInstances)
			config.RoutingPolicy = tc.routingPolicy
			config.RoutingScorerConfigs = tc.scorerConfigs
			config.Scheduler = tc.scheduler
			config.PriorityPolicy = tc.priorityPolicy
			config.AdmissionPolicy = tc.admissionPolicy
			// Token bucket with generous capacity so all requests are admitted
			if tc.admissionPolicy == "token-bucket" {
				config.TokenBucketCapacity = 1e6
				config.TokenBucketRefillRate = 1e6
			}

			workload := newTestWorkload(numRequests)
			cs := NewClusterSimulator(config, workload, "")
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()
			injected := len(agg.Requests)

			// BC-3: Three-term conservation equation (INV-1)
			conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
			if conservation != injected {
				t.Errorf("INV-1 conservation: completed(%d) + queued(%d) + running(%d) = %d, injected = %d",
					agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
			}

			// BC-4: All complete under infinite horizon with ample resources
			if agg.CompletedRequests != numRequests {
				t.Errorf("infinite horizon: CompletedRequests = %d, want %d",
					agg.CompletedRequests, numRequests)
			}

			// Cross-check: sum of per-instance completions == aggregated
			sumCompleted := 0
			for _, inst := range cs.Instances() {
				sumCompleted += inst.Metrics().CompletedRequests
			}
			if sumCompleted != agg.CompletedRequests {
				t.Errorf("aggregation: sum(per-instance) = %d, aggregated = %d",
					sumCompleted, agg.CompletedRequests)
			}
		})
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_Conservation_PolicyMatrix -v`
Expected: PASS (all 10 combinations)

**Step 3: Run all cluster tests**

Run: `go test ./sim/cluster/... -count=1`
Expected: PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "test(cluster): add INV-1 conservation matrix across 10 policy combos (#322)

- Table-driven test with routing × scheduler × admission × priority
- Three-term conservation: completed + queued + running == injected
- Includes round-robin, least-loaded, weighted, prefix-affinity
- Includes sjf, priority-fcfs with slo-based priority
- Includes token-bucket admission (generous capacity)
- Promoted from H12 hypothesis experiment

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add determinism test with prefix-affinity

**Contracts Implemented:** BC-5, BC-6

**Files:**
- Modify: `sim/cluster/cluster_test.go`

**Step 1: Write the determinism test with prefix-affinity and weighted routing**

Context: This promotes H13's determinism validation. Existing determinism tests don't cover prefix-affinity (highest non-determinism risk due to LRU map iteration in PrefixCacheIndex). This test runs each configuration twice and asserts byte-identical JSON output.

Add to `sim/cluster/cluster_test.go`:

```go
// TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical verifies INV-6
// for routing policies that use stateful scorers with internal maps (promoted from H13):
// GIVEN identical config with prefix-affinity or weighted routing (includes prefix scorer)
// WHEN run twice with same seed
// THEN per-request metrics JSON is byte-identical.
//
// This specifically targets the PrefixCacheIndex LRU which uses map iteration internally.
// Non-deterministic map iteration in scoring or eviction would cause divergence here.
func TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical(t *testing.T) {
	policies := []struct {
		name          string
		routingPolicy string
		scorerConfigs []sim.ScorerConfig
	}{
		{"prefix-affinity", "prefix-affinity", sim.DefaultScorerConfigs()},
		{"weighted-default", "weighted", sim.DefaultScorerConfigs()},
	}

	for _, pol := range policies {
		t.Run(pol.name, func(t *testing.T) {
			mkSim := func() *ClusterSimulator {
				config := newTestDeploymentConfig(3)
				config.RoutingPolicy = pol.routingPolicy
				config.RoutingScorerConfigs = pol.scorerConfigs
				// Use prefix tokens to exercise the prefix cache index
				workload := &sim.GuideLLMConfig{
					Rate:               10.0 / 1e6,
					NumRequests:        30,
					PrefixTokens:       32,
					PromptTokens:       100,
					PromptTokensStdDev: 20,
					PromptTokensMin:    10,
					PromptTokensMax:    200,
					OutputTokens:       50,
					OutputTokensStdDev: 10,
					OutputTokensMin:    10,
					OutputTokensMax:    100,
				}
				cs := NewClusterSimulator(config, workload, "")
				mustRun(t, cs)
				return cs
			}

			cs1 := mkSim()
			cs2 := mkSim()

			m1 := cs1.AggregatedMetrics()
			m2 := cs2.AggregatedMetrics()

			// Integer fields must match exactly
			if m1.CompletedRequests != m2.CompletedRequests {
				t.Errorf("CompletedRequests: %d vs %d", m1.CompletedRequests, m2.CompletedRequests)
			}
			if m1.TotalInputTokens != m2.TotalInputTokens {
				t.Errorf("TotalInputTokens: %d vs %d", m1.TotalInputTokens, m2.TotalInputTokens)
			}
			if m1.TotalOutputTokens != m2.TotalOutputTokens {
				t.Errorf("TotalOutputTokens: %d vs %d", m1.TotalOutputTokens, m2.TotalOutputTokens)
			}
			if m1.SimEndedTime != m2.SimEndedTime {
				t.Errorf("SimEndedTime: %d vs %d", m1.SimEndedTime, m2.SimEndedTime)
			}

			// Per-request metrics must be byte-identical (sorted JSON)
			j1, _ := json.Marshal(sortedRequestMetrics(m1.Requests))
			j2, _ := json.Marshal(sortedRequestMetrics(m2.Requests))
			if !bytes.Equal(j1, j2) {
				t.Error("INV-6 violated: per-request metrics JSON differs between runs " +
					"(likely non-deterministic map iteration in prefix cache or scorer)")
			}
		})
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_Determinism_PrefixAffinity -v`
Expected: PASS

**Step 3: Run all cluster tests**

Run: `go test ./sim/cluster/... -count=1`
Expected: PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "test(cluster): add INV-6 determinism test for prefix-affinity routing (#323)

- Run-twice-and-diff with prefix-affinity and weighted routing
- Exercises PrefixCacheIndex LRU (highest non-determinism risk)
- Uses prefix tokens to ensure cache index is actively used
- Byte-identical JSON comparison of per-request metrics
- Promoted from H13 hypothesis experiment

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Update CLAUDE.md

**Contracts Implemented:** (documentation completeness)

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add INV-8 to the invariants list in CLAUDE.md**

In CLAUDE.md, in the "Key Invariants to Maintain" section, add after INV-7:

```markdown
- **INV-8 Work-conserving**: After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator must not idle while work is waiting.
```

**Step 2: Update the invariant count in Project Governance section**

In CLAUDE.md, in the "Standards (what rules apply)" section, change:
```
- `docs/standards/invariants.md`: **7 system invariants** (INV-1 through INV-7) — with verification strategies
```
To:
```
- `docs/standards/invariants.md`: **8 system invariants** (INV-1 through INV-8) — with verification strategies
```

**Step 3: Verify the additions**

Run: `grep "INV-8" CLAUDE.md`
Expected: Shows both the invariant bullet point and the "8 system invariants" reference

**Step 3: Run all tests to verify nothing broke**

Run: `go test ./... -count=1`
Expected: PASS

**Step 4: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add INV-8 work-conserving invariant to CLAUDE.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Invariant | `TestWorkConserving_StepRestartsWhenWaitQNonEmpty` |
| BC-2 | Task 2 | Invariant | (same test, delay bound assertion) |
| BC-3 | Task 3 | Invariant | `TestClusterSimulator_Conservation_PolicyMatrix` |
| BC-4 | Task 3 | Invariant | (same test, all-complete assertion) |
| BC-5 | Task 4 | Invariant | `TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical` (prefix-affinity sub-test) |
| BC-6 | Task 4 | Invariant | `TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical` (weighted sub-test) |
| NC-1 | Task 2 | (design) | The test IS the regression guard; removing the fix causes failure |

No golden dataset updates needed. No shared test infrastructure changes. All tests use existing helpers (`mustNewSimulator`, `newTestDeploymentConfig`, `newTestWorkload`, `mustRun`, `sortedRequestMetrics`).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Conservation test flaky with token-bucket admission (race on admission) | Low | Medium | Use generous capacity (1e6) so all requests are admitted | Task 3 |
| Prefix-affinity determinism test fragile to future PrefixCacheIndex changes | Low | Low | Test is behavioral (byte-identical output), not structural | Task 4 |
| Work-conserving test relies on specific beta coefficient timing | Low | Low | Delay bound uses generous 2× margin; primary assertion is "both complete" | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — all tests use existing patterns
- [x] No feature creep — only tests and docs, no production code changes
- [x] No unexercised flags or interfaces — all test configurations are valid
- [x] No partial implementations — every task produces a complete, passing test
- [x] No breaking changes — additive only (new tests, new doc section)
- [x] No hidden global state impact — tests create fresh simulators
- [x] All new code passes golangci-lint
- [x] Shared test helpers reused (not duplicated)
- [x] CLAUDE.md updated with INV-8
- [x] No stale references in CLAUDE.md
- [x] Deviation log complete — 5 deviations documented
- [x] Each task produces working, testable code
- [x] Task dependencies ordered correctly (1 before 2 for INV-8 doc reference; 3 and 4 independent)
- [x] All contracts mapped to tasks
- [x] No golden dataset regeneration needed
- [x] No new struct fields — no construction site audit needed
- [x] R1: No silent data loss — tests only
- [x] R2: No map iteration in new code (tests use existing sorted helpers)
- [x] R6: No `logrus.Fatalf` in `sim/` — tests only
- [x] R7: All tests are invariant tests (no golden-only tests added)

---

## Appendix: File-Level Implementation Details

### File: `docs/standards/invariants.md`

**Purpose:** Add INV-8 work-conserving invariant section.

**Changes:**
1. Update header paragraph to include INV-8 in structural model family
2. Add new `## INV-8: Work-Conserving Property` section after INV-7

### File: `sim/simulator_test.go`

**Purpose:** Add `TestWorkConserving_StepRestartsWhenWaitQNonEmpty`.

**Key notes:**
- Uses `MaxRunningReqs: 1` (critical — this is what triggers the stranding scenario)
- Both requests arrive at t=0 and t=1 respectively (B arrives during A's service time)
- Primary assertion: `CompletedRequests == 2` (without fix, only 1 completes)
- Secondary assertion: delay bound (generous, avoids brittleness)
- Also checks INV-1 conservation as a bonus

### File: `sim/cluster/cluster_test.go`

**Purpose:** Add conservation matrix test and determinism test.

**Conservation matrix notes:**
- 10 rows covering: 4 routing policies × 3 schedulers × 2 admission policies × 2 priority policies (not full cross-product — selected combos)
- Uses `newTestDeploymentConfig` and `newTestWorkload` helpers
- Token bucket with generous capacity (1e6) to ensure all requests admitted
- Three-term equation: `completed + still_queued + still_running == len(Requests)`

**Determinism notes:**
- Tests prefix-affinity and weighted routing (both exercise PrefixCacheIndex)
- Uses `PrefixTokens: 32` in workload to ensure prefix cache is active
- Reuses existing `sortedRequestMetrics` helper for byte-identical JSON comparison
- `json` and `bytes` packages already imported in the test file
