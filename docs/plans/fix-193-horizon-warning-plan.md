# Horizon Too Small Warning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Warn users when the simulation horizon is too small for any requests to complete, or when all requests are rejected — two silent-failure scenarios that produce empty output with no explanation.

**The problem today:** When `horizon < admissionLatency + routingLatency`, all requests are silently dropped because their pipeline events exceed the horizon timestamp. The event loop breaks before processing them, producing 0 completed requests with no error or warning. Similarly, when all requests are rejected by admission (e.g., `RejectAll` policy), the user sees no output because `SaveResults` guards metrics output behind `CompletedRequests > 0`. In both cases, the user gets silence instead of an actionable diagnostic.

**What this PR adds:**
1. **Startup horizon check** — a warning at simulation start if the configured horizon is shorter than the admission + routing pipeline latency, telling the user no requests can complete
2. **Post-simulation all-rejected warning** — a warning after the event loop if all generated requests were rejected and none completed, telling the user how many were rejected and by what policy
3. **Post-simulation zero-completion warning** — a warning if no requests completed at all (catches the horizon-too-small case and any other silent-drop scenario)

**Why this matters:** Silent failures waste user time. Capacity planning requires clear feedback when configuration is misconfigured. These warnings catch the two most common misconfiguration patterns.

**Architecture:** All changes in `sim/cluster/cluster.go` — add warnings in `NewClusterSimulator()` (startup check) and at the end of `Run()` (post-simulation checks). No new types, interfaces, or CLI flags. Warnings use `logrus.Warnf` at the CLI boundary-adjacent cluster level.

**Source:** GitHub issue #193

**Closes:** Fixes #193

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds three diagnostic warnings to the cluster simulator to surface two silent-failure scenarios: (1) horizon too short for the admission+routing pipeline, and (2) all requests rejected with no completions. The warnings are pure logging — no behavioral changes to the simulation itself. They integrate at the cluster level (`sim/cluster/cluster.go`) in the constructor and post-simulation finalization. No adjacent components are modified. No deviations from the issue description.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Startup Horizon Warning
- GIVEN a cluster simulator configured with `horizon < admissionLatency + routingLatency`
- WHEN the cluster simulator is constructed
- THEN a warning message MUST be logged containing the horizon value and pipeline latency sum
- MECHANISM: Check in `NewClusterSimulator` after latencies are resolved

BC-2: All-Rejected Warning
- GIVEN a simulation that completes with `rejectedRequests > 0` and `CompletedRequests == 0`
- WHEN the Run() method finishes
- THEN a warning message MUST be logged stating the count of rejected requests
- MECHANISM: Post-loop check in `Run()` after `aggregateMetrics()`

BC-3: Zero-Completion Warning
- GIVEN a simulation that completes with `CompletedRequests == 0` and `rejectedRequests == 0`
- WHEN the Run() method finishes
- THEN a warning message MUST be logged indicating no requests completed
- MECHANISM: Post-loop check in `Run()` after `aggregateMetrics()`

**Negative Contracts:**

NC-1: No Warning on Normal Operation
- GIVEN a simulation where `CompletedRequests > 0`
- WHEN the Run() method finishes
- THEN no zero-completion or all-rejected warnings MUST be logged
- MECHANISM: Guards on `CompletedRequests == 0`

NC-2: No False Startup Warning
- GIVEN a cluster simulator where `horizon >= admissionLatency + routingLatency`
- WHEN the cluster simulator is constructed
- THEN no horizon warning MUST be logged
- MECHANISM: Strict `<` comparison (not `<=`)

NC-3: No Behavioral Change
- GIVEN any simulation configuration
- WHEN warnings are logged
- THEN the simulation output (metrics, traces, JSON) MUST be byte-identical to what it would produce without this PR
- MECHANISM: Warnings are additive logging only — no control flow changes

### C) Component Interaction

```
cmd/root.go → NewClusterSimulator() → [BC-1: startup warning check]
                                     ↓
                                   Run()
                                     ↓
                              [event loop — unchanged]
                                     ↓
                              aggregateMetrics()
                                     ↓
                              [BC-2, BC-3: post-sim warning checks]
```

No new types, interfaces, or state. Warnings read existing fields (`admissionLatency`, `routingLatency`, `rejectedRequests`, `aggregatedMetrics.CompletedRequests`).

**Extension friction:** 0 files to add another warning of the same kind — just add another `if` block in `Run()`.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Suggests `logrus.Warnf("[cluster] horizon...")` format | Uses same format | No deviation |
| Suggests checking in constructor lines 44-89 | Checks after constructor returns full struct | CORRECTION: Need access to resolved `admissionLatency` and `routingLatency` fields, which are set during construction. The check goes at the end of the constructor, after the struct is built. |
| Two scenarios only | Adds BC-3 for zero-completion without rejection | ADDITION: Catches edge cases beyond the two described (e.g., horizon > pipeline but still too short for any request to finish) |

### E) Review Guide

1. **THE TRICKY PART:** The startup horizon check must use the correct fields (`admissionLatency` and `routingLatency` from the `ClusterSimulator`, not the `DeploymentConfig`). Both should be the same, but the struct fields are what the event pipeline actually uses.
2. **WHAT TO SCRUTINIZE:** BC-1 comparison — `horizon < admissionLatency + routingLatency` must be strict `<`, not `<=`. When equal, the first request's routing event lands exactly at the horizon boundary, and the event loop uses `> horizon` (not `>=`) for the break condition, so it would still be processed.
3. **WHAT'S SAFE TO SKIM:** The test infrastructure — it's straightforward log capture.
4. **KNOWN DEBT:** None.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files:**
- Modify: `sim/cluster/cluster.go` — add startup warning in constructor, post-sim warnings in `Run()`
- Create: `sim/cluster/cluster_warnings_test.go` — behavioral tests for all 3 contracts + 2 negative contracts

**Key decisions:**
- Warnings are `logrus.Warnf`, not `logrus.Fatalf` — misconfiguration should inform, not crash
- Post-sim warnings check after `aggregateMetrics()` so `CompletedRequests` is finalized
- No dead code — every warning is exercised by tests

### G) Task Breakdown

---

### Task 1: Startup Horizon Warning (BC-1, NC-2)

**Contracts Implemented:** BC-1, NC-2

**Files:**
- Modify: `sim/cluster/cluster.go`
- Create: `sim/cluster/cluster_warnings_test.go`

**Step 1: Write failing test for BC-1**

Context: We need to verify that constructing a ClusterSimulator with horizon < pipeline latency produces a warning. We'll capture log output to assert.

```go
package cluster

import (
	"bytes"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// captureLogOutput runs fn and returns the log output as a string.
func captureLogOutput(fn func()) string {
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	origLevel := logrus.GetLevel()
	logrus.SetLevel(logrus.WarnLevel)
	defer func() {
		logrus.SetOutput(nil) // reset to default (stderr)
		logrus.SetLevel(origLevel)
	}()
	fn()
	return buf.String()
}

func TestClusterSimulator_HorizonTooSmall_WarnsAtStartup(t *testing.T) {
	// GIVEN horizon (100) < admissionLatency (200) + routingLatency (300) = 500
	config := DeploymentConfig{
		NumInstances:      1,
		Horizon:           100,
		Seed:              42,
		TotalKVBlocks:     100,
		BlockSizeTokens:   16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 1000,
		AdmissionPolicy:   "always-admit",
		RoutingPolicy:     "round-robin",
		PriorityPolicy:    "constant",
		Scheduler:         "fcfs",
		AdmissionLatency:  200,
		RoutingLatency:    300,
	}
	workload := &sim.GuideLLMConfig{Rate: 1.0, MaxPrompts: 10, PromptTokens: 10, OutputTokens: 10,
		PromptTokensMin: 1, PromptTokensMax: 20, OutputTokensMin: 1, OutputTokensMax: 20}

	// WHEN the cluster simulator is constructed
	output := captureLogOutput(func() {
		NewClusterSimulator(config, workload, "")
	})

	// THEN a warning about horizon being too small MUST be logged
	if !strings.Contains(output, "horizon") || !strings.Contains(output, "pipeline latency") {
		t.Errorf("expected warning about horizon < pipeline latency, got: %q", output)
	}
}

func TestClusterSimulator_HorizonSufficient_NoWarning(t *testing.T) {
	// GIVEN horizon (10000) >= admissionLatency (200) + routingLatency (300) = 500
	config := DeploymentConfig{
		NumInstances:      1,
		Horizon:           10000,
		Seed:              42,
		TotalKVBlocks:     100,
		BlockSizeTokens:   16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 1000,
		AdmissionPolicy:   "always-admit",
		RoutingPolicy:     "round-robin",
		PriorityPolicy:    "constant",
		Scheduler:         "fcfs",
		AdmissionLatency:  200,
		RoutingLatency:    300,
	}
	workload := &sim.GuideLLMConfig{Rate: 1.0, MaxPrompts: 10, PromptTokens: 10, OutputTokens: 10,
		PromptTokensMin: 1, PromptTokensMax: 20, OutputTokensMin: 1, OutputTokensMax: 20}

	// WHEN the cluster simulator is constructed
	output := captureLogOutput(func() {
		NewClusterSimulator(config, workload, "")
	})

	// THEN no horizon warning MUST be logged
	if strings.Contains(output, "horizon") && strings.Contains(output, "pipeline latency") {
		t.Errorf("unexpected horizon warning for sufficient horizon, got: %q", output)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_Horizon -v`
Expected: FAIL (no warning produced yet)

**Step 3: Implement startup warning**

Context: Add the horizon check at the end of `NewClusterSimulator`, after the struct is fully constructed.

In `sim/cluster/cluster.go`, add after the `return &ClusterSimulator{...}` block (change to assign to variable, add check, then return):

```go
	cs := &ClusterSimulator{
		// ... existing fields unchanged ...
	}

	// Startup warning: horizon too small for pipeline (BC-1)
	pipelineLatency := cs.admissionLatency + cs.routingLatency
	if cs.config.Horizon > 0 && cs.config.Horizon < pipelineLatency {
		logrus.Warnf("[cluster] horizon (%d) < pipeline latency (%d); no requests can complete — increase --horizon or reduce admission/routing latency",
			cs.config.Horizon, pipelineLatency)
	}

	return cs
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_Horizon -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/cluster_warnings_test.go
git commit -m "fix(cluster): warn when horizon < pipeline latency (BC-1, NC-2)

- Add startup check in NewClusterSimulator
- Warn when horizon < admissionLatency + routingLatency
- Add tests for warning presence and absence

Fixes #193 (partial)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Post-Simulation Warnings (BC-2, BC-3, NC-1)

**Contracts Implemented:** BC-2, BC-3, NC-1

**Files:**
- Modify: `sim/cluster/cluster.go`
- Modify: `sim/cluster/cluster_warnings_test.go`

**Step 1: Write failing tests for BC-2 and BC-3**

Context: We need to test post-simulation warnings. BC-2 requires all requests rejected (use `reject-all` policy). BC-3 requires zero completions without rejections (use very short horizon). NC-1 verifies no spurious warnings on normal operation.

Add to `sim/cluster/cluster_warnings_test.go`:

```go
func TestClusterSimulator_AllRejected_WarnsAfterRun(t *testing.T) {
	// GIVEN a cluster with reject-all admission policy
	config := DeploymentConfig{
		NumInstances:      1,
		Horizon:           100000,
		Seed:              42,
		TotalKVBlocks:     100,
		BlockSizeTokens:   16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 1000,
		AdmissionPolicy:   "reject-all",
		RoutingPolicy:     "round-robin",
		PriorityPolicy:    "constant",
		Scheduler:         "fcfs",
	}
	workload := &sim.GuideLLMConfig{Rate: 1.0, MaxPrompts: 5, PromptTokens: 10, OutputTokens: 10,
		PromptTokensMin: 1, PromptTokensMax: 20, OutputTokensMin: 1, OutputTokensMax: 20}

	cs := NewClusterSimulator(config, workload, "")

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		cs.Run()
	})

	// THEN a warning about all requests being rejected MUST be logged
	if !strings.Contains(output, "rejected") {
		t.Errorf("expected all-rejected warning, got: %q", output)
	}
}

func TestClusterSimulator_ZeroCompletions_WarnsAfterRun(t *testing.T) {
	// GIVEN a cluster with horizon too short for any request to finish
	// (horizon > pipeline latency but too short for actual processing)
	config := DeploymentConfig{
		NumInstances:      1,
		Horizon:           1, // 1 tick — admits but can't finish
		Seed:              42,
		TotalKVBlocks:     100,
		BlockSizeTokens:   16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 1000,
		AdmissionPolicy:   "always-admit",
		RoutingPolicy:     "round-robin",
		PriorityPolicy:    "constant",
		Scheduler:         "fcfs",
	}
	workload := &sim.GuideLLMConfig{Rate: 1.0, MaxPrompts: 5, PromptTokens: 10, OutputTokens: 10,
		PromptTokensMin: 1, PromptTokensMax: 20, OutputTokensMin: 1, OutputTokensMax: 20}

	cs := NewClusterSimulator(config, workload, "")

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		cs.Run()
	})

	// THEN a warning about no completed requests MUST be logged
	if !strings.Contains(output, "no requests completed") {
		t.Errorf("expected zero-completion warning, got: %q", output)
	}
}

func TestClusterSimulator_NormalOperation_NoPostSimWarning(t *testing.T) {
	// GIVEN a properly configured cluster that will complete requests
	config := DeploymentConfig{
		NumInstances:      1,
		Horizon:           1000000,
		Seed:              42,
		TotalKVBlocks:     100,
		BlockSizeTokens:   16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 1000,
		AdmissionPolicy:   "always-admit",
		RoutingPolicy:     "round-robin",
		PriorityPolicy:    "constant",
		Scheduler:         "fcfs",
	}
	workload := &sim.GuideLLMConfig{Rate: 0.001, MaxPrompts: 2, PromptTokens: 10, OutputTokens: 5,
		PromptTokensMin: 5, PromptTokensMax: 15, OutputTokensMin: 3, OutputTokensMax: 8}

	cs := NewClusterSimulator(config, workload, "")

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		cs.Run()
	})

	// THEN no rejection or zero-completion warnings MUST be logged
	if strings.Contains(output, "rejected") || strings.Contains(output, "no requests completed") {
		t.Errorf("unexpected warning during normal operation, got: %q", output)
	}
}
```

**Step 2: Run test to verify they fail**

Run: `go test ./sim/cluster/... -run "TestClusterSimulator_AllRejected|TestClusterSimulator_ZeroCompletions|TestClusterSimulator_NormalOperation_NoPostSim" -v`
Expected: FAIL for BC-2 and BC-3 tests (no warning produced); PASS for NC-1 (no warning expected)

**Step 3: Implement post-simulation warnings**

Context: Add checks at the end of `Run()`, after `aggregateMetrics()` but before the method returns.

In `sim/cluster/cluster.go`, add after line `c.aggregatedMetrics = c.aggregateMetrics()`:

```go
	// Post-simulation diagnostic warnings (BC-2, BC-3)
	if c.aggregatedMetrics.CompletedRequests == 0 {
		if c.rejectedRequests > 0 {
			logrus.Warnf("[cluster] all %d requests rejected by admission policy %q — no requests completed",
				c.rejectedRequests, c.config.AdmissionPolicy)
		} else {
			logrus.Warnf("[cluster] no requests completed — horizon may be too short or workload too small")
		}
	}
```

**Step 4: Run test to verify they pass**

Run: `go test ./sim/cluster/... -run "TestClusterSimulator_AllRejected|TestClusterSimulator_ZeroCompletions|TestClusterSimulator_NormalOperation_NoPostSim" -v`
Expected: PASS

**Step 5: Run full test suite and lint**

Run: `go test ./... -count=1 && golangci-lint run ./...`
Expected: All tests pass, 0 lint issues

**Step 6: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/cluster_warnings_test.go
git commit -m "fix(cluster): warn on all-rejected or zero-completion simulations (BC-2, BC-3, NC-1)

- Add post-simulation warning when all requests rejected by admission
- Add post-simulation warning when no requests completed (any cause)
- Normal operation produces no spurious warnings

Fixes #193

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestClusterSimulator_HorizonTooSmall_WarnsAtStartup |
| NC-2 | Task 1 | Unit | TestClusterSimulator_HorizonSufficient_NoWarning |
| BC-2 | Task 2 | Integration | TestClusterSimulator_AllRejected_WarnsAfterRun |
| BC-3 | Task 2 | Integration | TestClusterSimulator_ZeroCompletions_WarnsAfterRun |
| NC-1 | Task 2 | Integration | TestClusterSimulator_NormalOperation_NoPostSimWarning |

**Golden dataset:** No changes — this PR adds warnings only, no metric or output format changes. NC-3 is verified by the existing golden dataset tests continuing to pass unchanged.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Log capture interferes with test parallelism | Low | Medium | Use `captureLogOutput` helper that saves/restores log settings | Task 1 |
| `aggregatedMetrics` nil when Run() not called | Low | Low | Post-sim warnings are in Run(), after aggregateMetrics() | Task 2 |
| Warning text changes break tests | Low | Low | Tests check for key substrings ("horizon", "rejected"), not exact messages | Both |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — just `if` checks and `logrus.Warnf`
- [x] No feature creep beyond PR scope — warnings only, no behavioral changes
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes — additive logging only
- [x] No hidden global state impact — logrus is already used throughout
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: `captureLogOutput` is local to this test file (not shared — too simple to extract)
- [x] CLAUDE.md: no update needed (no new files, packages, or CLI flags)
- [x] Deviation log reviewed — one CORRECTION (check placement) and one ADDITION (BC-3) documented
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 2 depends on Task 1 for test helper)
- [x] All contracts mapped to tasks
- [x] Golden dataset: no changes needed
- [x] Construction site audit: no new struct fields added
- [x] No new CLI flags
- [x] No error paths with silent `continue`
- [x] No map iteration affecting output
- [x] Library code: warnings use `logrus.Warnf` (non-fatal) — acceptable in `sim/cluster/` per convention (cluster is CLI-adjacent)
- [x] No resource allocation loops
- [x] No exported mutable maps
- [x] No YAML config changes
- [x] No division operations
- [x] Grepped for "193" references — none in codebase

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/cluster.go`

**Purpose:** Add startup and post-simulation diagnostic warnings.

**Change 1 (constructor):** Replace `return &ClusterSimulator{...}` with:

```go
	cs := &ClusterSimulator{
		// ... all existing fields unchanged ...
	}

	// Startup warning: horizon too small for pipeline (BC-1)
	pipelineLatency := cs.admissionLatency + cs.routingLatency
	if cs.config.Horizon > 0 && cs.config.Horizon < pipelineLatency {
		logrus.Warnf("[cluster] horizon (%d) < pipeline latency (%d); no requests can complete — increase --horizon or reduce admission/routing latency",
			cs.config.Horizon, pipelineLatency)
	}

	return cs
```

**Change 2 (end of Run):** After `c.aggregatedMetrics = c.aggregateMetrics()`, add:

```go
	// Post-simulation diagnostic warnings (BC-2, BC-3)
	if c.aggregatedMetrics.CompletedRequests == 0 {
		if c.rejectedRequests > 0 {
			logrus.Warnf("[cluster] all %d requests rejected by admission policy %q — no requests completed",
				c.rejectedRequests, c.config.AdmissionPolicy)
		} else {
			logrus.Warnf("[cluster] no requests completed — horizon may be too short or workload too small")
		}
	}
```

### File: `sim/cluster/cluster_warnings_test.go`

**Purpose:** Behavioral tests for all warning contracts.

**Complete implementation:** See Task 1 and Task 2 Step 1 above for full test code.

**Key notes:**
- `captureLogOutput` helper captures logrus output by redirecting to a buffer
- Tests check substring matches ("horizon", "rejected", "no requests completed") for resilience
- NC-1 and NC-2 are negative tests that verify absence of warnings
