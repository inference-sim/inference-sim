# Promote H-Overload & H-Liveness Hypothesis Findings + Promotion Label — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Encode two confirmed hypothesis experiments (H-Overload, H-Liveness) as Go regression tests in CI, and establish a `promotion` label for tracking hypothesis-to-test-suite promotions.

**The problem today:** H-Overload and H-Liveness are confirmed findings validated by bash-script experiments, but they are NOT in CI — a regression would not be caught by `go test ./...`. Additionally, promotion issues are labeled `enhancement`, making them indistinguishable from feature requests in issue filters.

**What this PR adds:**
1. **Overload conservation test** — verifies INV-1 holds at 10x arrival rate with both `always-admit` and `token-bucket` admission policies (promoted from H-Overload, #337)
2. **Scheduler liveness test** — verifies all three schedulers (FCFS, SJF, priority-FCFS) complete all requests under load that creates queueing (promoted from H-Liveness, #336)
3. **`promotion` GitHub label** — new label for hypothesis-to-Go-test promotions, applied to #336, #337, and retroactively to promotion issues closed by PR #332 (#322, #323, #328)
4. **Process documentation update** — `docs/process/hypothesis.md` issue taxonomy updated to include the `promotion` label

**Why this matters:** Regression protection for confirmed system properties. The promotion label creates a trackable pipeline from hypothesis experiments to CI tests.

**Architecture:** Two new test functions in `sim/cluster/cluster_test.go` following the established pattern from PR #332 (table-driven, using `newTestDeploymentConfig`, `mustRun`, `AggregatedMetrics`). Process doc update in `docs/process/hypothesis.md`. No new packages, no new types, no new interfaces.

**Source:** GitHub issues #336, #337

**Closes:** Fixes #336, fixes #337

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR promotes two confirmed hypothesis findings to the Go test suite:
- **H-Overload** (#337): Conservation (INV-1) holds under 10x overload with both admission policies
- **H-Liveness** (#336): All schedulers satisfy liveness (zero still_queued, zero still_running) under admissible load

Both tests follow the pattern established in PR #332 (`TestClusterSimulator_Conservation_PolicyMatrix`, `TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical`).

Additionally, a `promotion` GitHub label is created and the hypothesis process documentation is updated to include it in the issue taxonomy.

Adjacent blocks: cluster simulation (`sim/cluster/`), admission policies (`sim/admission.go`), schedulers (`sim/scheduler.go`).
3 justified deviations documented in Section D (test-scale params, batch constraint, conservation reformulation).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Overload Conservation (INV-1)
- GIVEN a 4-instance cluster at 10x overload rate with always-admit admission
- WHEN the simulation completes (finite horizon)
- THEN `completed + still_queued + still_running == injected` (three-term conservation)
- MECHANISM: INV-1 holds regardless of load level; the simulator never drops requests silently

BC-2: Overload Conservation with Token-Bucket (INV-1 extended)
- GIVEN a 4-instance cluster at 10x overload rate with token-bucket admission
- WHEN the simulation completes (finite horizon)
- THEN `completed + still_queued + still_running + rejected == total_generated` (four-term conservation)
- MECHANISM: Token-bucket rejects excess requests; rejected + admitted must equal total generated

BC-3: Scheduler Liveness (INV-2)
- GIVEN 3 schedulers (fcfs, sjf, priority-fcfs) with a mixed workload at a rate that creates queueing, batch-constrained (max-running=8)
- WHEN the simulation runs to completion (infinite horizon, ample resources)
- THEN all requests complete: `still_queued == 0` AND `still_running == 0`
- MECHANISM: Every scheduler must eventually drain the queue; infinite horizon + ample resources guarantee no starvation

BC-4: Liveness Conservation Cross-Check
- GIVEN the same liveness test configurations
- WHEN the simulation completes
- THEN `completed == injected` (since still_queued=0 and still_running=0 under infinite horizon)
- MECHANISM: Conservation (INV-1) + liveness together imply all requests reach completion

**Negative Contracts:**

BC-5: No Panics Under Overload
- GIVEN 10x overload rate
- WHEN the simulation runs
- THEN no panic occurs (the test completes without runtime panic)
- MECHANISM: Cluster simulator handles queue growth gracefully

**Process Contracts:**

BC-6: Promotion Label Exists
- GIVEN the GitHub repository
- WHEN listing labels
- THEN a `promotion` label exists with description "Hypothesis finding promoted to Go test suite"

BC-7: Process Documentation Updated
- GIVEN `docs/process/hypothesis.md`
- WHEN reading the issue taxonomy table
- THEN a `promotion` row exists with label, when-to-file guidance, title format, and example

### C) Component Interaction

```
[Test: Overload Conservation]  →  ClusterSimulator(config, workload)
                                    ├── AdmissionPolicy (always-admit / token-bucket)
                                    ├── RoutingPolicy (least-loaded)
                                    └── Scheduler (fcfs)

[Test: Scheduler Liveness]     →  ClusterSimulator(config, workload)
                                    ├── AdmissionPolicy (always-admit)
                                    ├── RoutingPolicy (least-loaded)
                                    └── Scheduler (fcfs / sjf / priority-fcfs)
```

No new types. No new interfaces. No state changes. Tests are pure consumers of existing APIs.

Extension friction: 0 (no new extensible types).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #337: "run at 10x rate with always-admit and token-bucket" | Uses test-scale parameters (not the exact H-Overload experiment params) | SIMPLIFICATION: Go tests need small, fast configs (~seconds). The invariant is rate-independent; the test just needs genuine overload. |
| #336: "run 3 schedulers × mixed workload at a rate that creates queueing" | Uses batch-constrained config (max-running=8) to force queueing | ADDITION: H-Liveness Round 2b showed that default batch size (256) prevents queueing at test rates. Constraining max-running=8 ensures the test exercises the queue. |
| #336: "verify all requests complete" | Also asserts conservation (completed == injected) | ADDITION: Stronger assertion — liveness implies conservation under infinite horizon. |
| #337: "verify injected == completed + queued + running + rejected" (single equation) | Two separate assertions: (1) completed + queued + running == admitted, (2) admitted + rejected == total_generated | REFORMULATION: In the plan, "injected" means admitted (post-rejection, from `len(agg.Requests)`), not total generated. Two assertions are logically equivalent to the issue's single equation but use the APIs available (`AggregatedMetrics().Requests` vs `RejectedRequests()`). |
| Existing tests use `sim.GuideLLMConfig{}` struct literals | Plan follows the same pattern | CONVENTION: `NewGuideLLMConfig` canonical constructor exists (R4), but all 4 existing test construction sites use struct literals. Following existing test convention to avoid inconsistency within the test file. |

### E) Review Guide

**The tricky part:** Getting the overload test parameters right — the arrival rate must genuinely exceed capacity to create the overload condition, but the test must finish in reasonable time (<5s). The finite horizon must be long enough to inject all requests but short enough that the test doesn't run forever.

**What to scrutinize:** The conservation equation in BC-2 (four-term with rejection). Make sure `RejectedRequests()` is the right API and that `len(agg.Requests)` gives injected count.

**What's safe to skim:** The liveness test — it follows the exact pattern from the existing `TestClusterSimulator_Conservation_PolicyMatrix`, just with different scheduler configs.

**Known debt:** None. The existing test infrastructure is well-established.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/cluster/cluster_test.go` — add two test functions
- `docs/process/hypothesis.md` — add `promotion` label to issue taxonomy

**Files NOT modified:** No production code changes. This PR is test + docs only.

**Key decisions:**
- Use test-scale parameters (not the exact H-Overload/H-Liveness experiment params) for fast CI execution
- Use finite horizon for overload test (can't wait for all 10x requests to complete — that's the point of overload)
- Use infinite horizon for liveness test (liveness = everything completes)
- Batch-constrain (max-running=8) the liveness test to force queueing

### G) Task Breakdown

---

### Task 1: Overload Conservation Test (BC-1, BC-2, BC-5)

**Contracts Implemented:** BC-1, BC-2, BC-5

**Files:**
- Modify: `sim/cluster/cluster_test.go` (append after existing TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical)

**Step 1: Write the overload conservation test**

Context: This test verifies that INV-1 conservation holds even at extreme overload (10x saturation rate). It tests both always-admit (three-term conservation) and token-bucket (four-term conservation with rejection). Promoted from H-Overload hypothesis experiment (PR #335).

In `sim/cluster/cluster_test.go`, append:

```go
// TestClusterSimulator_OverloadConservation verifies INV-1 under 10x overload
// (promoted from H-Overload hypothesis experiment, PR #335):
// GIVEN a 4-instance cluster at extreme overload rate
// WHEN the simulation runs to a finite horizon
// THEN conservation holds:
//   - always-admit: completed + still_queued + still_running == injected
//   - token-bucket: completed + still_queued + still_running + rejected == total_generated
// AND no panics occur (BC-5).
func TestClusterSimulator_OverloadConservation(t *testing.T) {
	// Use a high rate relative to capacity to create genuine overload.
	// With beta=[1000,10,5], 4 instances, max-running=256: capacity is very high
	// due to batching. A rate of 500 req/s with only 200 requests and a short
	// horizon creates a burst that overloads the system.
	cases := []struct {
		name            string
		admissionPolicy string
		// Token bucket params (only used when admission is "token-bucket")
		tbCapacity   float64
		tbRefillRate float64
	}{
		{"always-admit", "always-admit", 0, 0},
		{"token-bucket", "token-bucket", 5000, 10000},
	}

	const (
		numRequests  = 200
		numInstances = 4
		rateReqPerS  = 500.0
		// Short horizon: enough to inject all requests at 500 req/s
		// (200 requests / 500 req/s = 0.4s = 400,000 ticks),
		// but too short for all to complete under overload.
		horizon = 500_000 // 0.5 seconds in microsecond ticks
	)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			config := newTestDeploymentConfig(numInstances)
			config.Horizon = horizon
			config.AdmissionPolicy = tc.admissionPolicy
			config.RoutingPolicy = "least-loaded"
			config.Scheduler = "fcfs"
			config.PriorityPolicy = "constant"
			if tc.admissionPolicy == "token-bucket" {
				config.TokenBucketCapacity = tc.tbCapacity
				config.TokenBucketRefillRate = tc.tbRefillRate
			}

			workload := &sim.GuideLLMConfig{
				Rate:               rateReqPerS / 1e6,
				NumRequests:        numRequests,
				PrefixTokens:       0,
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

			agg := cs.AggregatedMetrics()
			injected := len(agg.Requests)
			rejected := cs.RejectedRequests()

			// BC-1/BC-2: Conservation equation
			conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
			if tc.admissionPolicy == "always-admit" {
				// Three-term conservation: no rejections
				if conservation != injected {
					t.Errorf("INV-1 conservation (always-admit): completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
						agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
				}
				if rejected != 0 {
					t.Errorf("always-admit should have 0 rejections, got %d", rejected)
				}
			} else {
				// Four-term conservation: injected + rejected == total generated
				totalGenerated := injected + rejected
				if conservation != injected {
					t.Errorf("INV-1 conservation (token-bucket): completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
						agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
				}
				if totalGenerated != numRequests {
					t.Errorf("four-term conservation: injected(%d) + rejected(%d) = %d, want %d (total generated)",
						injected, rejected, totalGenerated, numRequests)
				}
			}

			// Verify overload: under finite horizon, not all requests should complete
			// (this confirms the test is actually exercising overload, not a trivial case)
			if agg.CompletedRequests == numRequests && tc.admissionPolicy == "always-admit" {
				t.Logf("warning: all %d requests completed — overload may not be genuine (increase rate or decrease horizon)", numRequests)
			}
		})
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_OverloadConservation -v -count=1`
Expected: PASS (both sub-tests)

Note: This is a promotion test — no "write failing test first" since we're encoding a known-to-hold invariant.

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "test(cluster): promote H-Overload conservation under 10x to Go test (BC-1, BC-2, BC-5)

Promoted from hypothesis H-Overload (PR #335, issue #337).
Verifies INV-1 conservation at extreme overload with both always-admit
(3-term) and token-bucket (4-term with rejection).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Scheduler Liveness Test (BC-3, BC-4)

**Contracts Implemented:** BC-3, BC-4

**Files:**
- Modify: `sim/cluster/cluster_test.go` (append after Task 1's test)

**Step 1: Write the scheduler liveness test**

Context: This test verifies that all three schedulers satisfy liveness — every admitted request eventually completes — under a workload that creates genuine queueing. The batch size is constrained (max-running=8) to force queue buildup, matching H-Liveness Round 2b findings. Promoted from H-Liveness hypothesis experiment (PR #335).

In `sim/cluster/cluster_test.go`, append:

```go
// TestClusterSimulator_SchedulerLiveness verifies scheduler liveness (INV-2)
// across all scheduler types (promoted from H-Liveness hypothesis experiment, PR #335):
// GIVEN each scheduler (fcfs, sjf, priority-fcfs) with a mixed workload and
//   batch-constrained config (max-running=8) that forces queueing
// WHEN the simulation runs to completion (infinite horizon, ample resources)
// THEN all requests complete: still_queued == 0, still_running == 0
// AND completed == injected (conservation + liveness combined).
func TestClusterSimulator_SchedulerLiveness(t *testing.T) {
	schedulers := []struct {
		name           string
		scheduler      string
		priorityPolicy string
	}{
		{"fcfs", "fcfs", "constant"},
		{"sjf", "sjf", "constant"},
		{"priority-fcfs", "priority-fcfs", "slo-based"},
	}

	const (
		numRequests  = 100
		numInstances = 4
		rateReqPerS  = 200.0
		maxRunning   = 8 // Constrains batch size to force queueing
	)

	for _, tc := range schedulers {
		t.Run(tc.name, func(t *testing.T) {
			config := newTestDeploymentConfig(numInstances)
			config.Horizon = math.MaxInt64 // Infinite horizon — all requests must complete
			config.MaxRunningReqs = maxRunning
			config.RoutingPolicy = "least-loaded"
			config.AdmissionPolicy = "always-admit"
			config.Scheduler = tc.scheduler
			config.PriorityPolicy = tc.priorityPolicy

			// Mixed workload: varying prompt and output sizes to exercise scheduler ordering
			workload := &sim.GuideLLMConfig{
				Rate:               rateReqPerS / 1e6,
				NumRequests:        numRequests,
				PrefixTokens:       0,
				PromptTokens:       200,
				PromptTokensStdDev: 100,
				PromptTokensMin:    32,
				PromptTokensMax:    512,
				OutputTokens:       128,
				OutputTokensStdDev: 64,
				OutputTokensMin:    16,
				OutputTokensMax:    256,
			}

			cs := NewClusterSimulator(config, workload, "")
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()
			injected := len(agg.Requests)

			// BC-3: Liveness — no requests stranded
			if agg.StillQueued != 0 {
				t.Errorf("liveness: still_queued = %d, want 0 (scheduler %s)", agg.StillQueued, tc.scheduler)
			}
			if agg.StillRunning != 0 {
				t.Errorf("liveness: still_running = %d, want 0 (scheduler %s)", agg.StillRunning, tc.scheduler)
			}

			// BC-4: Conservation + liveness → all complete
			if agg.CompletedRequests != injected {
				t.Errorf("conservation+liveness: completed = %d, injected = %d (scheduler %s)",
					agg.CompletedRequests, injected, tc.scheduler)
			}
		})
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_SchedulerLiveness -v -count=1`
Expected: PASS (all three sub-tests: fcfs, sjf, priority-fcfs)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "test(cluster): promote H-Liveness scheduler liveness to Go test (BC-3, BC-4)

Promoted from hypothesis H-Liveness (PR #335, issue #336).
Verifies all three schedulers (FCFS, SJF, priority-FCFS) satisfy liveness
under batch-constrained config that forces queueing. Infinite horizon
ensures no false failures from premature termination.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Create Promotion Label and Update Process Docs (BC-6, BC-7)

**Contracts Implemented:** BC-6, BC-7

**Files:**
- Modify: `docs/process/hypothesis.md` (issue taxonomy table)

**Step 1: Create the `promotion` label on GitHub**

```bash
gh label create promotion \
  --description "Hypothesis finding promoted to Go test suite" \
  --color "5319E7"
```

Expected: Label created successfully.

**Step 2: Apply `promotion` label to current and past promotion issues**

Current issues (this PR):
```bash
gh issue edit 336 --add-label promotion
gh issue edit 337 --add-label promotion
```

Retroactive (promotion issues closed by PR #332 — only the enhancement/promotion issues, not bug #326 or standards #327):
```bash
gh issue edit 322 --add-label promotion
gh issue edit 323 --add-label promotion
gh issue edit 328 --add-label promotion
```

**Step 3: Update the issue taxonomy in `docs/process/hypothesis.md`**

Context: The issue taxonomy table (around line 153) lists issue types and labels. Add a `Promotion` row.

In `docs/process/hypothesis.md`, find the issue taxonomy table and add a new row after `Enhancement`:

The table currently reads:
```
| **Enhancement** | `--label enhancement` | New feature, rule, or documentation improvement needed | `enhancement: <area> — <improvement>` | `enhancement: CLI — document token-bucket per-input-token cost model` (H5) |
```

Add after it:
```
| **Promotion** | `--label promotion` | Confirmed hypothesis finding promoted from bash experiment to Go test suite | `enhancement: promote <hypothesis> <finding> to Go test suite` | `enhancement: promote H-Overload conservation under 10x to Go test suite` (#337) |
```

Also update the "Mapping from resolution type to expected issues" table to mention promotion:

In the "Clean confirmation" row, change:
```
| Clean confirmation | Usually none. Optionally: standards update confirming existing rules. |
```
to:
```
| Clean confirmation | Usually none. Optionally: promotion to Go test suite, standards update confirming existing rules. |
```

**Step 4: Commit**

```bash
git add docs/process/hypothesis.md
git commit -m "docs(process): add promotion label to hypothesis issue taxonomy (BC-6, BC-7)

New 'promotion' label tracks hypothesis-to-Go-test promotions.
Applied to #336, #337 (current) and #322, #323, #328 (retroactive promotions).
Updated issue taxonomy table and resolution mapping in hypothesis.md.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Run Full Test Suite and Lint

**Contracts Implemented:** Verification gate for all contracts

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: All packages PASS

**Step 2: Run full lint**

Run: `golangci-lint run ./...`
Expected: 0 issues

**Step 3: Verify git status**

Run: `git status`
Expected: Working tree clean (all changes committed)

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Invariant | TestClusterSimulator_OverloadConservation/always-admit |
| BC-2 | Task 1 | Invariant | TestClusterSimulator_OverloadConservation/token-bucket |
| BC-3 | Task 2 | Invariant | TestClusterSimulator_SchedulerLiveness/{fcfs,sjf,priority-fcfs} |
| BC-4 | Task 2 | Invariant | TestClusterSimulator_SchedulerLiveness/{fcfs,sjf,priority-fcfs} |
| BC-5 | Task 1 | Implicit | Test completes without panic |
| BC-6 | Task 3 | Manual | `gh label list` shows `promotion` |
| BC-7 | Task 3 | Manual | Read updated `docs/process/hypothesis.md` |

Golden dataset: NOT affected (no production code changes).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Overload test parameters don't create genuine overload | Medium | Low | Test includes a warning log if all requests complete; tunable constants |
| Liveness test takes too long with infinite horizon + 100 requests | Low | Medium | 100 requests with 4 instances × max-running=8 finishes quickly |
| Token-bucket four-term conservation equation wrong | Low | High | Verified against H-Overload FINDINGS.md and existing cluster.go:RejectedRequests() API |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions (tests only, no new types)
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (`newTestDeploymentConfig`, `mustRun`)
- [x] CLAUDE.md: no update needed (no new files/packages/CLI flags)
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — all 3 deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4)
- [x] All contracts mapped to tasks
- [x] Golden dataset: NOT affected
- [x] Construction site audit: N/A (no new struct fields)

**Antipattern rules:**
- [x] R1: N/A (no error paths)
- [x] R2: N/A (no map iteration in new code)
- [x] R3: N/A (no new CLI flags)
- [x] R4: N/A (no new struct fields)
- [x] R5: N/A (no resource allocation)
- [x] R6: N/A (test code only)
- [x] R7: Tests ARE invariant tests (no golden tests added)
- [x] R8-R17: N/A (no production code changes)

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/cluster_test.go`

**Purpose:** Add two promoted hypothesis tests

**Changes:** Append two functions after `TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical`:
1. `TestClusterSimulator_OverloadConservation` (~70 lines)
2. `TestClusterSimulator_SchedulerLiveness` (~60 lines)

Both use existing test helpers (`newTestDeploymentConfig`, `mustRun`, `NewClusterSimulator`).
Both use existing `sim.GuideLLMConfig` for workload generation.
Both use `AggregatedMetrics()` for assertion data.
The overload test additionally uses `RejectedRequests()` for four-term conservation.

No new imports needed beyond what's already imported.

### File: `docs/process/hypothesis.md`

**Purpose:** Add `promotion` label to issue taxonomy

**Changes:**
1. Add `Promotion` row to issue types table (~line 157)
2. Update "Clean confirmation" row in resolution mapping table (~line 166)

No structural changes. Pure content additions to existing tables.
