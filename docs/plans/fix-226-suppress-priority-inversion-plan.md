# Fix #226: Suppress Priority Inversion Counter for Constant Priority — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Suppress misleading priority inversion anomaly counter when no priority differentiation is configured.

**The problem today:** Running the default configuration (`--priority-policy constant --scheduler fcfs`) reports 777 priority inversions for 100 requests. This is misleading because, with constant priority (all requests have priority 1.0), E2E differences come entirely from workload variance (different token counts), not scheduling unfairness. Users see an alarming anomaly counter that represents normal behavior.

**What this PR adds:**
1. Priority-aware anomaly detection — the priority inversion counter is suppressed (returns 0) when the priority policy is "constant", since there are no priorities to invert.
2. The `CollectRawMetrics` function accepts the priority policy name, threading it through to `detectPriorityInversions`.

**Why this matters:** False-positive anomaly counters erode user trust in the simulator's diagnostics. Suppressing the counter for constant priority makes the output actionable — when inversions are reported, they represent real scheduling unfairness.

**Architecture:** The change is localized to `sim/cluster/metrics.go` (add `priorityPolicy` parameter to `CollectRawMetrics` and suppress logic in `detectPriorityInversions`) and its call site in `cmd/root.go` (pass the priority policy string). No new types, interfaces, or packages.

**Source:** GitHub issue #226

**Closes:** Fixes #226

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a false-positive anomaly counter. The priority inversion detector (`detectPriorityInversions`) performs O(n²) pairwise E2E comparisons on all requests, flagging when an earlier-arriving request takes >2× longer than a later one. With constant priority and FCFS scheduling, this measures workload heterogeneity (different token counts), not scheduling unfairness.

The fix adds the priority policy name as a parameter to `CollectRawMetrics`. When the policy is "constant", priority inversion detection is skipped (returns 0). All other anomaly counters (HOL blocking, rejected requests) are unaffected.

Adjacent blocks: `cmd/root.go` (call site), `sim/cluster/metrics.go` (implementation), `sim/cluster/evaluation.go` (wraps `CollectRawMetrics` for trace evaluation).

No deviations from source.

### B) Behavioral Contracts

**BC-1: Constant priority suppresses inversion counter**
- GIVEN per-instance metrics with requests that would normally trigger priority inversions
- WHEN `CollectRawMetrics` is called with `priorityPolicy = "constant"`
- THEN `PriorityInversions` is 0

**BC-2: Non-constant priority preserves inversion detection**
- GIVEN per-instance metrics with requests that trigger priority inversions (earlier request has >2× E2E of later request)
- WHEN `CollectRawMetrics` is called with `priorityPolicy = "slo-based"`
- THEN `PriorityInversions` is > 0 (same behavior as before)

**BC-3: Other anomaly counters unaffected**
- GIVEN the same inputs
- WHEN `CollectRawMetrics` is called with any priority policy
- THEN `HOLBlockingEvents` and `RejectedRequests` are computed identically regardless of priority policy

### C) Component Interaction

```
cmd/root.go (knows priorityPolicy string)
    ↓ passes to
cluster.CollectRawMetrics(aggregated, perInstance, rejectedRequests, priorityPolicy)
    ↓ passes to
detectPriorityInversions(perInstance, priorityPolicy)
    ↓ returns 0 if priorityPolicy == "constant"
```

Also: `cluster/evaluation.go` calls `CollectRawMetrics` — needs the same parameter update.

### D) Deviation Log

None. Source is issue #226, which describes this exact fix.

### E) Review Guide

Focus on: (1) all call sites of `CollectRawMetrics` are updated, (2) test covers both "constant" suppression and "slo-based" pass-through, (3) no other anomaly counters are affected.

---

## Part 2: Implementation

### F) Implementation Overview

Two files modified: `sim/cluster/metrics.go` (add parameter, guard logic) and `cmd/root.go` (pass priority policy). One file potentially modified: `sim/cluster/evaluation.go` (if it calls `CollectRawMetrics`).

### G) Task Breakdown

#### Task 1: Add priorityPolicy parameter to CollectRawMetrics and suppress detection

**Behavioral contracts:** BC-1, BC-2, BC-3

**Step 1: Write failing tests**

Add two tests to `sim/cluster/metrics_test.go`:

```go
// TestCollectRawMetrics_ConstantPriority_SuppressesInversions verifies BC-1.
func TestCollectRawMetrics_ConstantPriority_SuppressesInversions(t *testing.T) {
	// GIVEN per-instance metrics with requests that would trigger inversions
	m := sim.NewMetrics()
	m.Requests["early"] = sim.RequestMetrics{ID: "early", ArrivedAt: 100}
	m.RequestE2Es["early"] = 50000.0 // 10× slower than "late"
	m.Requests["late"] = sim.RequestMetrics{ID: "late", ArrivedAt: 200}
	m.RequestE2Es["late"] = 5000.0

	aggregated := sim.NewMetrics()
	aggregated.CompletedRequests = 2
	aggregated.SimEndedTime = 1_000_000

	// WHEN collecting with constant priority policy
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m}, 0, "constant")

	// THEN priority inversions should be suppressed
	if raw.PriorityInversions != 0 {
		t.Errorf("expected 0 priority inversions with constant policy, got %d", raw.PriorityInversions)
	}
}

// TestCollectRawMetrics_SLOBasedPriority_DetectsInversions verifies BC-2.
func TestCollectRawMetrics_SLOBasedPriority_DetectsInversions(t *testing.T) {
	// GIVEN per-instance metrics with requests that would trigger inversions
	m := sim.NewMetrics()
	m.Requests["early"] = sim.RequestMetrics{ID: "early", ArrivedAt: 100}
	m.RequestE2Es["early"] = 50000.0
	m.Requests["late"] = sim.RequestMetrics{ID: "late", ArrivedAt: 200}
	m.RequestE2Es["late"] = 5000.0

	aggregated := sim.NewMetrics()
	aggregated.CompletedRequests = 2
	aggregated.SimEndedTime = 1_000_000

	// WHEN collecting with slo-based priority policy
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m}, 0, "slo-based")

	// THEN priority inversions should be detected
	if raw.PriorityInversions == 0 {
		t.Error("expected priority inversions > 0 with slo-based policy")
	}
}
```

**Verify tests fail:**
```bash
go test ./sim/cluster/... -run "TestCollectRawMetrics_(ConstantPriority|SLOBasedPriority)" -count=1
```
Expected: compilation error (wrong number of arguments to `CollectRawMetrics`).

**Step 2: Update CollectRawMetrics signature and implementation**

In `sim/cluster/metrics.go`:

1. Change `CollectRawMetrics` signature to accept `priorityPolicy string`:
```go
func CollectRawMetrics(aggregated *sim.Metrics, perInstance []*sim.Metrics, rejectedRequests int, priorityPolicy string) *RawMetrics {
```

2. Pass `priorityPolicy` to `detectPriorityInversions`:
```go
raw.PriorityInversions = detectPriorityInversions(perInstance, priorityPolicy)
```

3. Update `detectPriorityInversions` to accept and check policy:
```go
func detectPriorityInversions(perInstance []*sim.Metrics, priorityPolicy string) int {
	if priorityPolicy == "constant" {
		return 0
	}
	// ... rest unchanged
}
```

**Step 3: Update all call sites**

Find all callers of `CollectRawMetrics`:
- `cmd/root.go`: pass `priorityPolicy` (already in scope as the `priorityPolicy` variable)
- `sim/cluster/evaluation.go`: check if it calls `CollectRawMetrics` and update
- `sim/cluster/metrics_test.go`: update existing test calls to pass `""` or `"constant"` as appropriate

For `cmd/root.go`, the call becomes:
```go
rawMetrics := cluster.CollectRawMetrics(
    cs.AggregatedMetrics(),
    cs.PerInstanceMetrics(),
    cs.RejectedRequests(),
    priorityPolicy,
)
```

For existing tests that pass `nil` for `perInstance` (no anomaly detection), pass `""` as policy — the guard in `detectPriorityInversions` won't even be reached because `perInstance` is `nil`.

**Verify tests pass:**
```bash
go test ./sim/cluster/... -count=1
go test ./cmd/... -count=1
```

**Step 4: Lint**
```bash
golangci-lint run ./sim/cluster/... ./cmd/...
```

### H) Test Strategy

- **BC-1 test:** Construct per-instance metrics with a known inversion (10× E2E difference), call with "constant" → assert 0 inversions
- **BC-2 test:** Same metrics, call with "slo-based" → assert > 0 inversions
- **BC-3:** Covered implicitly — HOL blocking and rejected request tests don't pass priority policy and their behavior is unchanged
- **Existing tests:** All existing `detectPriorityInversions` and `detectHOLBlocking` tests continue to pass

### I) Risk Analysis

- **Low risk:** Single parameter addition to one function, localized guard clause
- **Call site audit:** Must find ALL callers of `CollectRawMetrics` — grep for it across the codebase
- **Backward compatibility:** Existing behavior preserved for all non-"constant" policies

---

## Part 3: Sanity Checklist

### J) Sanity Checklist

- [x] No new exported mutable maps
- [x] No YAML config changes (no zero-value ambiguity)
- [x] No new division operations
- [x] No new `logrus.Fatalf` in library code
- [x] No new `continue` or early `return` that drops data
- [x] Construction site audit: `CollectRawMetrics` call sites identified (cmd/root.go, evaluation.go, metrics_test.go)
- [x] Tests are behavioral (GIVEN/WHEN/THEN), not structural
- [x] No new interfaces or interface changes

---

## Appendix: File-Level Details

### sim/cluster/metrics.go
- `CollectRawMetrics`: Add `priorityPolicy string` parameter
- `detectPriorityInversions`: Add `priorityPolicy string` parameter, return 0 when "constant"

### cmd/root.go
- Pass `priorityPolicy` to `cluster.CollectRawMetrics()`

### sim/cluster/evaluation.go
- Update call to `CollectRawMetrics` if present (pass deployment config's priority policy)

### sim/cluster/metrics_test.go
- Add `TestCollectRawMetrics_ConstantPriority_SuppressesInversions` (BC-1)
- Add `TestCollectRawMetrics_SLOBasedPriority_DetectsInversions` (BC-2)
- Update existing `CollectRawMetrics` calls to include priority policy parameter
