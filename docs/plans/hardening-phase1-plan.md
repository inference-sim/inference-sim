# Phase 1: Structural Helpers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add four foundational helper functions that centralize repeated patterns, making downstream bug fixes simpler and preventing recurrence of known antipatterns.

**The problem today:** The codebase has three copies of the load calculation formula across routing policies (DRY violation that caused #175), three separate `RequestMetrics{}` construction sites that can silently miss new fields (#189), non-deterministic float accumulation in `ComputeFitness`/`JainFairnessIndex` that violates the determinism invariant (#195), and three missing fields in `aggregateMetrics` that produce zeros in cluster-level output (#191).

**What this PR adds:**
1. `EffectiveLoad()` method on `RoutingSnapshot` — one canonical load formula (`QueueDepth + BatchSize + PendingRequests`) replacing 4 inline calculations across routing policies and counterfactual scoring
2. `NewRequestMetrics()` factory function — a single constructor for `RequestMetrics` that ensures all fields (including `SLOClass`, `TenantID`, `HandledBy`) are propagated at every injection site
3. `sortedKeys()` helper — deterministic map iteration for float accumulation in `ComputeFitness`, `JainFairnessIndex`, and `mapValues`
4. Complete `aggregateMetrics()` field coverage — adds `PreemptionCount` (summed), `CacheHitRate` (averaged), `KVThrashingRate` (averaged) to cluster-level metrics aggregation

**Why this matters:** Phase 2 bug fixes depend on these helpers — `EffectiveLoad()` *is* the structural fix for routing inconsistency, and `NewRequestMetrics()` *is* the fix for CSV field propagation. Phase 4 invariant tests need deterministic output to work. These helpers are the foundation for the entire hardening PR.

**Architecture:** All changes are in existing files (`sim/routing.go`, `sim/metrics_utils.go`, `sim/simulator.go`, `sim/workload_config.go`, `sim/cluster/metrics.go`, `sim/cluster/cluster.go`, `sim/cluster/counterfactual.go`). No new packages, no new interfaces, no architectural changes. Pure refactoring with two behavioral fixes (#189 CSV field propagation, #191 aggregation completeness).

**Source:** GitHub issue #208 (Phase 1 of #214), design doc `docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md` Phase 1

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds four small helper functions that centralize repeated patterns in routing, metrics construction, float accumulation, and metrics aggregation. It replaces inline load calculations with a method, inline `RequestMetrics{}` construction with a factory, unsorted map iteration with sorted key iteration, and adds three missing fields to cluster-level metrics aggregation.

Where it fits: Phase 1 of 6 in the hardening PR (#214). Phase 2 bug fixes depend on these helpers. No dependencies on prior unmerged work — builds directly on current `main`.

Adjacent blocks: `sim/routing.go` (routing policies), `sim/simulator.go` (request injection), `sim/workload_config.go` (CSV loading), `sim/cluster/metrics.go` (fitness computation), `sim/cluster/cluster.go` (aggregation), `sim/cluster/counterfactual.go` (counterfactual scoring).

DEVIATION: The design doc says `LeastLoaded` and `AlwaysBusiest` use `QueueDepth + BatchSize` only, but the code already includes `PendingRequests` (#175 was already fixed). See Deviation Log (Section D).

### B) Behavioral Contracts

**Positive Contracts:**

**BC-1: EffectiveLoad formula consistency**
- GIVEN any `RoutingSnapshot` with `QueueDepth=5`, `BatchSize=3`, `PendingRequests=2`
- WHEN `EffectiveLoad()` is called
- THEN it MUST return `10` (`5 + 3 + 2`)
- MECHANISM: Value-receiver method on `RoutingSnapshot` in `sim/routing.go`

**BC-2: Routing policies use EffectiveLoad**
- GIVEN `LeastLoaded`, `WeightedScoring`, and `AlwaysBusiest` routing policies
- WHEN routing with snapshots containing non-zero `PendingRequests`
- THEN all three MUST use `EffectiveLoad()` for their load calculations (no inline formulas)
- MECHANISM: Refactor all 3 routing policies + counterfactual scoring to call `snap.EffectiveLoad()`

**BC-3: NewRequestMetrics field propagation**
- GIVEN a `Request` with `SLOClass="realtime"`, `TenantID="tenant_1"`, `AssignedInstance="instance_0"`
- WHEN `NewRequestMetrics(req, arrivedAt)` is called
- THEN the returned `RequestMetrics` MUST have `SLOClass="realtime"`, `TenantID="tenant_1"`, `HandledBy="instance_0"`, correct `NumPrefillTokens`, correct `NumDecodeTokens`, and matching `ID`/`ArrivedAt`
- MECHANISM: Canonical constructor in `sim/metrics_utils.go`

**BC-4: CSV path uses canonical injection**
- GIVEN a CSV workload file loaded via `generateWorkloadFromCSV`
- WHEN requests are processed
- THEN each request MUST be registered via `InjectArrival()` (not inline `RequestMetrics{}` construction)
- MECHANISM: Refactor `generateWorkloadFromCSV` to construct `Request` then call `sim.InjectArrival(req)`

**BC-5: Deterministic fitness score**
- GIVEN identical `RawMetrics` and weight map
- WHEN `ComputeFitness` is called N times
- THEN all N calls MUST return identical `Score` values
- MECHANISM: `sortedKeys()` helper ensures deterministic map iteration before float accumulation

**BC-6: Deterministic fairness index**
- GIVEN identical throughput map
- WHEN `JainFairnessIndex` is called N times
- THEN all N calls MUST return identical values
- MECHANISM: `sortedKeys()` helper for deterministic iteration over throughput map

**BC-7: aggregateMetrics field completeness**
- GIVEN a cluster with 2 instances where instance_0 has `PreemptionCount=3, CacheHitRate=0.8, KVThrashingRate=0.1` and instance_1 has `PreemptionCount=5, CacheHitRate=0.6, KVThrashingRate=0.3`
- WHEN `aggregateMetrics()` runs after finalization
- THEN aggregated `PreemptionCount` MUST be `8` (summed), `KVAllocationFailures` MUST be summed, `CacheHitRate` MUST be `0.7` (averaged), `KVThrashingRate` MUST be `0.2` (averaged)
- MECHANISM: Add sum/average logic to `aggregateMetrics()` in `sim/cluster/cluster.go`

**Negative Contracts:**

**NC-1: No routing behavioral regression**
- GIVEN all existing routing policy tests
- WHEN the `EffectiveLoad()` refactoring is applied
- THEN all existing tests MUST pass without modification
- MECHANISM: Pure refactoring — same formula, extracted to method

**NC-2: No inline RequestMetrics construction in production code**
- GIVEN the codebase after this PR
- WHEN searching for `RequestMetrics{` in `sim/` non-test Go files
- THEN zero matches MUST be found (all production sites use `NewRequestMetrics`)
- MECHANISM: Replace all 3 production construction sites; test files may still use inline construction

### C) Component Interaction

```
sim/routing.go          sim/cluster/counterfactual.go
  LeastLoaded ─────┐       computeCounterfactual ──┐
  WeightedScoring ──┼── EffectiveLoad() ────────────┘
  AlwaysBusiest ───┘    (method on RoutingSnapshot)

sim/simulator.go        sim/workload_config.go
  InjectArrival ────┐       generateWorkloadFromCSV ──┐
  InjectArrivalAt ──┼── NewRequestMetrics() ──────────┘
                    │   (factory in metrics_utils.go)
                    └── (CSV path now calls InjectArrival directly)

sim/cluster/metrics.go
  ComputeFitness ───┐
  JainFairnessIndex ┼── sortedKeys()
  mapValues ────────┘   (helper in metrics.go)

sim/cluster/cluster.go
  aggregateMetrics() ── adds PreemptionCount, CacheHitRate, KVThrashingRate
```

**API Contracts:**
- `EffectiveLoad() int` — pure, no side effects, returns sum of 3 int fields
- `NewRequestMetrics(req *Request, arrivedAt float64) RequestMetrics` — pure constructor, no side effects
- `sortedKeys(m map[string]float64) []string` — pure, returns sorted copy of keys

**State Changes:** None. All helpers are pure functions/methods. `aggregateMetrics` writes to existing fields on the `Metrics` struct.

**Extension Friction:** Adding a new field to `RequestMetrics` requires changing 1 file (the `NewRequestMetrics` constructor) instead of 3. Adding a new metric to fitness computation automatically gets deterministic iteration.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| `EffectiveLoad()` returns `int64` (design doc line 79) | Returns `int` | CORRECTION: `QueueDepth`, `BatchSize`, `PendingRequests` are all `int` fields. Returning `int64` would require explicit casts at every call site with no benefit. |
| `LeastLoaded` and `AlwaysBusiest` use `QueueDepth + BatchSize` (design doc lines 85, 87) | All 3 policies already use `QueueDepth + BatchSize + PendingRequests` | CORRECTION: #175 was already fixed (issue is closed). The `EffectiveLoad()` refactor is a pure DRY consolidation, not a behavior change. |
| Design doc Phase 1a lists 3 inline calculations | Plan updates 4 sites (3 routing + 1 counterfactual) | ADDITION: `computeCounterfactual()` in `sim/cluster/counterfactual.go:51` also uses the inline formula. Issue #175 mentions it explicitly. |
| `NewRequestMetrics` omits `HandledBy` (design doc lines 100-108) | Includes `HandledBy: req.AssignedInstance` | CORRECTION: Existing `InjectArrival`/`InjectArrivalAt` already propagate `HandledBy` from `req.AssignedInstance` (#181). Omitting it from the canonical constructor would regress existing field propagation. |
| Design doc shows `sortedKeys` applied to `mapValues` | Plan applies it to `mapValues` for principle consistency | SIMPLIFICATION: `mapValues` output is always sorted downstream by `NewDistribution`, so sorting keys first is belt-and-suspenders. Included for consistency with the determinism principle. |

### E) Review Guide

**The tricky part:** The CSV path refactoring (BC-4). `generateWorkloadFromCSV` currently uses `arrivalFloat` (raw CSV float) for `ArrivedAt`, while `InjectArrival` uses `float64(req.ArrivalTime)/1e6` which round-trips through `int64`. This produces a tiny floating-point difference (< 1 microsecond). This is actually more correct — it uses the canonical tick-based time — but could cause golden dataset value changes if CSV-based tests exist.

**What to scrutinize:** BC-7 (aggregateMetrics). Verify the averaging logic handles N=0 instances (division by zero) and N=1 instance (average = value).

**What's safe to skim:** BC-1, BC-2, NC-1 — pure mechanical extraction of existing formulas into a method. BC-5, BC-6 — straightforward sorted iteration.

**Known debt:**
- Test files still construct `RequestMetrics{}` inline (8+ sites in test code). This is intentional — tests construct partial structs for specific scenarios and shouldn't be forced through the factory.
- `sim/workload_config.go` and `sim/metrics_utils.go` contain pre-existing `logrus.Fatalf` calls that violate the library boundary rule. This is out of scope — tracked as Phase 6e in the design doc.
- `ComputePerSLODistributions` and `detectPriorityInversions` iterate maps non-deterministically but are safe: `ComputePerSLODistributions` feeds `NewDistribution` which sorts internally; `detectPriorityInversions` sorts its accumulated slice; `SLOAttainment` uses integer addition (associative). These do not need `sortedKeys`.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/routing.go` — add `EffectiveLoad()` method, update 3 routing policies
- `sim/metrics_utils.go` — add `NewRequestMetrics()` factory
- `sim/simulator.go` — update `InjectArrival()` and `InjectArrivalAt()` to use factory
- `sim/workload_config.go` — refactor `generateWorkloadFromCSV()` to call `InjectArrival`
- `sim/cluster/metrics.go` — add `sortedKeys()` helper, apply to 3 functions
- `sim/cluster/cluster.go` — add 3 missing fields to `aggregateMetrics()`
- `sim/cluster/counterfactual.go` — update `computeCounterfactual()` to use `EffectiveLoad()`

**Files to create:**
- None

**Test files to modify:**
- `sim/metrics_utils_test.go` — add `TestNewRequestMetrics_PropagatesAllFields`
- `sim/cluster/cluster_test.go` — add `TestAggregateMetrics_IncludesKVCacheFields`

**Key decisions:**
- `EffectiveLoad()` returns `int` (not `int64`) to match field types
- `NewRequestMetrics` includes `HandledBy` from `req.AssignedInstance`
- `sortedKeys` is unexported (only used within `cluster` package)
- Test files keep inline `RequestMetrics{}` construction (intentional partial structs)

**Confirmation:** No dead code. All helpers are called by refactored callers in the same PR. All paths exercisable via existing CLI and tests.

### G) Task Breakdown

---

#### Task 1: EffectiveLoad() method and routing policy refactor

**Contracts Implemented:** BC-1, BC-2, NC-1

**Files:**
- Modify: `sim/routing.go` (add method + update 3 policies)
- Modify: `sim/cluster/counterfactual.go:51` (update load-based fallback)

**Step 1: Add EffectiveLoad() method and refactor routing policies**

Context: Adding a value-receiver method on `RoutingSnapshot` that centralizes the load formula, then refactoring all 4 inline calculations to use it.

In `sim/routing.go`, add after the `RoutingSnapshot` struct (after line 18):

```go
// EffectiveLoad returns the total effective load on this instance:
// QueueDepth + BatchSize + PendingRequests.
// Used by routing policies and counterfactual scoring for consistent load calculations.
func (s RoutingSnapshot) EffectiveLoad() int {
	return s.QueueDepth + s.BatchSize + s.PendingRequests
}
```

Then update `LeastLoaded.Route()` (lines 72-76):
```go
	minLoad := snapshots[0].EffectiveLoad()
	target := snapshots[0]

	for i := 1; i < len(snapshots); i++ {
		load := snapshots[i].EffectiveLoad()
```

Update `WeightedScoring.Route()` (line 142):
```go
		effectiveLoad := snap.EffectiveLoad()
```

Update `AlwaysBusiest.Route()` (lines 218-222):
```go
	maxLoad := snapshots[0].EffectiveLoad()
	target := snapshots[0]

	for i := 1; i < len(snapshots); i++ {
		load := snapshots[i].EffectiveLoad()
```

In `sim/cluster/counterfactual.go`, update line 51:
```go
			s = -float64(snap.EffectiveLoad())
```

**Step 2: Run existing tests to verify no regression**

Run: `go test ./sim/... -run "TestLeastLoaded\|TestWeightedScoring\|TestAlwaysBusiest\|TestPrefixAffinity\|TestRoundRobin" -v`
Expected: All PASS (NC-1: pure refactoring, no behavioral change)

Run: `go test ./sim/cluster/... -run "TestComputeCounterfactual" -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `go test ./...`
Expected: All PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/... ./sim/cluster/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/routing.go sim/cluster/counterfactual.go
git commit -m "refactor(sim): add EffectiveLoad() method on RoutingSnapshot (BC-1, BC-2)

- Centralize QueueDepth + BatchSize + PendingRequests formula
- Refactor LeastLoaded, WeightedScoring, AlwaysBusiest to use EffectiveLoad()
- Update computeCounterfactual() load-based fallback
- Fixes structural aspect of #175

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: NewRequestMetrics() factory function

**Contracts Implemented:** BC-3, NC-2

**Files:**
- Modify: `sim/metrics_utils.go` (add factory)
- Modify: `sim/simulator.go:255-282` (update InjectArrival, InjectArrivalAt)
- Test: `sim/metrics_utils_test.go` (add factory test)

**Step 1: Write failing test for NewRequestMetrics**

Context: Verify the factory propagates all fields including SLOClass, TenantID, and HandledBy (the fields missing from the CSV path in #189).

In `sim/metrics_utils_test.go` (create if needed, or append):

```go
package sim

import "testing"

func TestNewRequestMetrics_PropagatesAllFields(t *testing.T) {
	// GIVEN a request with all metadata fields populated
	req := &Request{
		ID:               "test_req_1",
		ArrivalTime:      2000000, // 2 seconds in ticks
		InputTokens:      make([]int, 128),
		OutputTokens:     make([]int, 64),
		SLOClass:         "realtime",
		TenantID:         "tenant_alpha",
		AssignedInstance:  "instance_3",
	}
	arrivedAt := float64(req.ArrivalTime) / 1e6

	// WHEN NewRequestMetrics is called
	rm := NewRequestMetrics(req, arrivedAt)

	// THEN all fields MUST be propagated
	if rm.ID != "test_req_1" {
		t.Errorf("ID: got %q, want %q", rm.ID, "test_req_1")
	}
	if rm.ArrivedAt != 2.0 {
		t.Errorf("ArrivedAt: got %f, want 2.0", rm.ArrivedAt)
	}
	if rm.NumPrefillTokens != 128 {
		t.Errorf("NumPrefillTokens: got %d, want 128", rm.NumPrefillTokens)
	}
	if rm.NumDecodeTokens != 64 {
		t.Errorf("NumDecodeTokens: got %d, want 64", rm.NumDecodeTokens)
	}
	if rm.SLOClass != "realtime" {
		t.Errorf("SLOClass: got %q, want %q", rm.SLOClass, "realtime")
	}
	if rm.TenantID != "tenant_alpha" {
		t.Errorf("TenantID: got %q, want %q", rm.TenantID, "tenant_alpha")
	}
	if rm.HandledBy != "instance_3" {
		t.Errorf("HandledBy: got %q, want %q", rm.HandledBy, "instance_3")
	}
}

func TestNewRequestMetrics_ZeroValueFields_OmittedInJSON(t *testing.T) {
	// GIVEN a request with empty metadata (typical CSV trace)
	req := &Request{
		ID:          "csv_req_1",
		ArrivalTime: 1000000,
		InputTokens: make([]int, 10),
		OutputTokens: make([]int, 5),
	}

	// WHEN NewRequestMetrics is called
	rm := NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)

	// THEN metadata fields MUST be empty strings (will be omitted in JSON via omitempty)
	if rm.SLOClass != "" {
		t.Errorf("SLOClass: got %q, want empty", rm.SLOClass)
	}
	if rm.TenantID != "" {
		t.Errorf("TenantID: got %q, want empty", rm.TenantID)
	}
	if rm.HandledBy != "" {
		t.Errorf("HandledBy: got %q, want empty", rm.HandledBy)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run "TestNewRequestMetrics" -v`
Expected: FAIL — `NewRequestMetrics` not defined

**Step 3: Implement NewRequestMetrics and update InjectArrival/InjectArrivalAt**

In `sim/metrics_utils.go`, add after the `RequestMetrics` struct (after line 36):

```go
// NewRequestMetrics creates a RequestMetrics from a Request and its arrival time.
// This is the canonical constructor — all production code MUST use this instead of
// inline RequestMetrics{} literals. Test code may still use literals for partial construction.
func NewRequestMetrics(req *Request, arrivedAt float64) RequestMetrics {
	return RequestMetrics{
		ID:               req.ID,
		ArrivedAt:        arrivedAt,
		NumPrefillTokens: len(req.InputTokens),
		NumDecodeTokens:  len(req.OutputTokens),
		SLOClass:         req.SLOClass,
		TenantID:         req.TenantID,
		HandledBy:        req.AssignedInstance,
	}
}
```

In `sim/simulator.go`, update `InjectArrival` (lines 257-265):

```go
func (sim *Simulator) InjectArrival(req *Request) {
	sim.Schedule(&ArrivalEvent{time: req.ArrivalTime, Request: req})
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}
```

Update `InjectArrivalAt` (lines 273-281):

```go
func (sim *Simulator) InjectArrivalAt(req *Request, eventTime int64) {
	sim.Schedule(&ArrivalEvent{time: eventTime, Request: req})
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run "TestNewRequestMetrics" -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `go test ./...`
Expected: All PASS

**Step 6: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/metrics_utils.go sim/metrics_utils_test.go sim/simulator.go
git commit -m "refactor(sim): add NewRequestMetrics() canonical constructor (BC-3)

- Replace inline RequestMetrics{} in InjectArrival and InjectArrivalAt
- Factory propagates SLOClass, TenantID, HandledBy from Request
- Add behavioral tests for field propagation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Refactor CSV path to use InjectArrival

**Contracts Implemented:** BC-4, NC-2

**Files:**
- Modify: `sim/workload_config.go:16-97` (refactor generateWorkloadFromCSV)

**Step 1: Refactor generateWorkloadFromCSV to use InjectArrival**

Context: The CSV path currently schedules the ArrivalEvent and constructs RequestMetrics inline, missing SLOClass/TenantID/HandledBy. Refactoring to call `InjectArrival` consolidates both operations and ensures field propagation.

In `sim/workload_config.go`, replace lines 82-95 (from `// 4. Push to schedule and metrics` to `reqIdx++`):

```go
		// 4. Inject via canonical path (handles both event scheduling and metrics registration)
		sim.InjectArrival(req)

		reqIdx++
```

This removes:
- The manual `sim.Schedule(&ArrivalEvent{...})` call (now inside InjectArrival)
- The inline `sim.Metrics.Requests[reqID] = RequestMetrics{...}` (now inside InjectArrival via NewRequestMetrics)

**Note on ArrivedAt precision:** `InjectArrival` computes `ArrivedAt` as `float64(req.ArrivalTime)/1e6` which round-trips through `int64(arrivalFloat * 1e6)`. This may differ from `arrivalFloat` by < 1 microsecond due to floating-point rounding. This is the canonical representation used everywhere else and is more correct.

**Step 2: Run existing CSV-based tests**

Run: `go test ./sim/... -v`
Expected: All PASS

Run: `go test ./... -count=1`
Expected: All PASS (including golden dataset tests which may use CSV traces)

**Step 3: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/workload_config.go
git commit -m "fix(sim): CSV workload path uses InjectArrival for field propagation (BC-4)

- Refactor generateWorkloadFromCSV to call InjectArrival instead of
  inline ArrivalEvent scheduling + RequestMetrics construction
- Fixes #189: CSV path now propagates SLOClass/TenantID/HandledBy
- All production RequestMetrics construction now goes through NewRequestMetrics

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: sortedKeys() helper and deterministic iteration

**Contracts Implemented:** BC-5, BC-6

**Files:**
- Modify: `sim/cluster/metrics.go` (add helper, update 3 functions)

**Step 1: Add sortedKeys helper and apply to ComputeFitness, JainFairnessIndex, mapValues**

Context: Go map iteration is non-deterministic. Float accumulation in different orders produces different results due to non-associativity. The `sortedKeys` helper ensures deterministic iteration order.

In `sim/cluster/metrics.go`, add after the imports (after line 12):

```go
// sortedKeys returns the keys of a map[string]float64 in sorted order.
// Used to ensure deterministic float accumulation across map iterations.
func sortedKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
```

Update `ComputeFitness` (lines 355-363) to use sorted iteration:

```go
func ComputeFitness(metrics *RawMetrics, weights map[string]float64) *FitnessResult {
	result := &FitnessResult{
		Components: make(map[string]float64, len(weights)),
	}

	for _, key := range sortedKeys(weights) {
		weight := weights[key]
		value, ok := extractMetric(metrics, key)
		if !ok {
			logrus.Warnf("ComputeFitness: unknown metric key %q, ignoring", key)
			continue
		}
		result.Components[key] = value
		result.Score += value * weight
	}

	return result
}
```

Update `JainFairnessIndex` (lines 304-319) to use sorted iteration:

```go
func JainFairnessIndex(throughputs map[string]float64) float64 {
	n := float64(len(throughputs))
	if n == 0 {
		return 0
	}
	sumX := 0.0
	sumX2 := 0.0
	for _, k := range sortedKeys(throughputs) {
		x := throughputs[k]
		sumX += x
		sumX2 += x * x
	}
	if sumX2 == 0 {
		return 0
	}
	return (sumX * sumX) / (n * sumX2)
}
```

Update `mapValues` (lines 322-328) to use sorted iteration:

```go
func mapValues(m map[string]float64) []float64 {
	vals := make([]float64, 0, len(m))
	for _, k := range sortedKeys(m) {
		vals = append(vals, m[k])
	}
	return vals
}
```

**Step 2: Run existing tests to verify no regression**

Run: `go test ./sim/cluster/... -run "TestComputeFitness\|TestJainFairness\|TestCollectRawMetrics" -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `go test ./...`
Expected: All PASS

**Step 4: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/cluster/metrics.go
git commit -m "fix(cluster): deterministic float accumulation via sortedKeys (BC-5, BC-6)

- Add sortedKeys() helper for deterministic map iteration
- Apply to ComputeFitness, JainFairnessIndex, and mapValues
- Fixes #195: same inputs now produce identical scores across runs

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: Complete aggregateMetrics field coverage

**Contracts Implemented:** BC-7

**Files:**
- Modify: `sim/cluster/cluster.go:266-306` (add 3 fields to aggregateMetrics)
- Test: `sim/cluster/cluster_test.go` (add aggregation test)

**Step 1: Write failing test for aggregateMetrics field coverage**

Context: Verify that PreemptionCount is summed and CacheHitRate/KVThrashingRate are averaged across instances in the aggregated metrics.

In `sim/cluster/cluster_test.go` (append to existing file):

```go
func TestAggregateMetrics_IncludesKVCacheFields(t *testing.T) {
	// GIVEN a cluster simulation with 2 instances
	cfg := newTestDeploymentConfig(2)
	cs := NewClusterSimulator(cfg, newTestWorkload(10), "")
	cs.Run()

	agg := cs.AggregatedMetrics()
	perInst := cs.PerInstanceMetrics()

	// THEN PreemptionCount MUST be the sum of per-instance counts
	expectedPreemption := int64(0)
	for _, m := range perInst {
		expectedPreemption += m.PreemptionCount
	}
	if agg.PreemptionCount != expectedPreemption {
		t.Errorf("PreemptionCount: got %d, want %d (sum of per-instance)", agg.PreemptionCount, expectedPreemption)
	}

	// THEN KVAllocationFailures MUST be the sum of per-instance counts
	expectedKVFailures := int64(0)
	for _, m := range perInst {
		expectedKVFailures += m.KVAllocationFailures
	}
	if agg.KVAllocationFailures != expectedKVFailures {
		t.Errorf("KVAllocationFailures: got %d, want %d (sum of per-instance)", agg.KVAllocationFailures, expectedKVFailures)
	}

	// THEN CacheHitRate MUST be the average of per-instance rates
	expectedCacheHit := 0.0
	for _, m := range perInst {
		expectedCacheHit += m.CacheHitRate
	}
	expectedCacheHit /= float64(len(perInst))
	if math.Abs(agg.CacheHitRate-expectedCacheHit) > 1e-9 {
		t.Errorf("CacheHitRate: got %f, want %f (average of per-instance)", agg.CacheHitRate, expectedCacheHit)
	}

	// THEN KVThrashingRate MUST be the average of per-instance rates
	expectedThrashing := 0.0
	for _, m := range perInst {
		expectedThrashing += m.KVThrashingRate
	}
	expectedThrashing /= float64(len(perInst))
	if math.Abs(agg.KVThrashingRate-expectedThrashing) > 1e-9 {
		t.Errorf("KVThrashingRate: got %f, want %f (average of per-instance)", agg.KVThrashingRate, expectedThrashing)
	}
}

func TestAggregateMetrics_SingleInstance_AverageEqualsSelf(t *testing.T) {
	// GIVEN a cluster with exactly 1 instance (edge case: average = self)
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, newTestWorkload(5), "")
	cs.Run()

	agg := cs.AggregatedMetrics()
	perInst := cs.PerInstanceMetrics()

	// THEN for a single instance, aggregated values MUST equal the instance values
	if agg.PreemptionCount != perInst[0].PreemptionCount {
		t.Errorf("PreemptionCount: got %d, want %d (single instance)", agg.PreemptionCount, perInst[0].PreemptionCount)
	}
	if math.Abs(agg.CacheHitRate-perInst[0].CacheHitRate) > 1e-9 {
		t.Errorf("CacheHitRate: got %f, want %f (single instance)", agg.CacheHitRate, perInst[0].CacheHitRate)
	}
	if math.Abs(agg.KVThrashingRate-perInst[0].KVThrashingRate) > 1e-9 {
		t.Errorf("KVThrashingRate: got %f, want %f (single instance)", agg.KVThrashingRate, perInst[0].KVThrashingRate)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run "TestAggregateMetrics_IncludesKVCacheFields" -v`
Expected: FAIL — PreemptionCount/CacheHitRate/KVThrashingRate are zero in aggregated metrics

**Step 3: Add missing fields to aggregateMetrics**

In `sim/cluster/cluster.go`, inside the `aggregateMetrics()` method, add after line 300 (`merged.RequestStepCounters = append(...)`):

```go
		merged.PreemptionCount += m.PreemptionCount
```

Also inside the same loop, add `KVAllocationFailures` (same class of missing field, summed counter) and accumulation for CacheHitRate and KVThrashingRate (after `merged.PreemptionCount += m.PreemptionCount`):

```go
		merged.KVAllocationFailures += m.KVAllocationFailures
```

Then add:

```go
		merged.CacheHitRate += m.CacheHitRate
		merged.KVThrashingRate += m.KVThrashingRate
```

After the loop (after line 301, before the `if c.workload != nil` check), compute averages:

```go
	if n := len(c.instances); n > 0 {
		merged.CacheHitRate /= float64(n)
		merged.KVThrashingRate /= float64(n)
	}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run "TestAggregateMetrics_IncludesKVCacheFields" -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `go test ./...`
Expected: All PASS

**Step 6: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/cluster_test.go
git commit -m "fix(cluster): complete aggregateMetrics with PreemptionCount/CacheHitRate/KVThrashingRate (BC-7)

- Sum PreemptionCount across instances
- Average CacheHitRate and KVThrashingRate across instances
- Fixes #191: aggregated metrics no longer report zeros for KV cache fields

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1, BC-2 | Task 1 | Regression | Existing `TestLeastLoaded_*`, `TestWeightedScoring_*`, `TestAlwaysBusiest_*` |
| BC-2 | Task 1 | Regression | Existing `TestComputeCounterfactual_*` |
| BC-3 | Task 2 | Unit | `TestNewRequestMetrics_PropagatesAllFields` |
| BC-3 | Task 2 | Unit | `TestNewRequestMetrics_ZeroValueFields_OmittedInJSON` |
| BC-4 | Task 3 | Regression | Existing golden dataset tests (CSV traces) |
| BC-5, BC-6 | Task 4 | Regression | Existing `TestComputeFitness_*`, `TestJainFairness_*` |
| BC-7 | Task 5 | Unit | `TestAggregateMetrics_IncludesKVCacheFields` |
| BC-7 | Task 5 | Unit | `TestAggregateMetrics_SingleInstance_AverageEqualsSelf` |
| NC-1 | Task 1 | Regression | All existing routing tests |

**Golden dataset update strategy:** Not needed. This PR does not change simulation output values — `EffectiveLoad` is a pure refactoring (same formula extracted), `sortedKeys` fixes non-determinism but the golden tests use specific models/seeds that produce deterministic output already, and `aggregateMetrics` fields were previously zero (so golden comparisons that don't check these fields are unaffected).

**Invariant test requirement:** No new invariant tests in this PR. The invariant tests (conservation, causality, determinism) are Phase 4 scope (#211). The `sortedKeys` fix enables the Phase 4 determinism test to succeed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| CSV ArrivedAt precision change | Low | Low | Round-trip through int64 ticks is the canonical representation; < 1μs difference | Task 3 |
| aggregateMetrics test needs workload config | Medium | Low | Test uses minimal config with distribution-based workload | Task 5 |
| sortedKeys performance overhead | Low | Low | O(n log n) on small maps (2-8 keys typical); negligible | Task 4 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — all helpers have 3+ callers
- [x] No feature creep — strictly Phase 1 scope from design doc
- [x] No unexercised flags or interfaces
- [x] No partial implementations — each helper is fully used in the same PR
- [x] No breaking changes — pure refactoring + additive field coverage
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: uses existing test infrastructure (no new shared helpers needed)
- [x] CLAUDE.md: no updates needed (no new files/packages/CLI flags)
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — all 4 deviations documented and justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 2 before Task 3; Task 1/4/5 independent)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration: not needed
- [x] Construction site audit: `RequestMetrics{}` has 3 production sites → all updated by Tasks 2-3
- [x] No new CLI flags
- [x] No silent `continue` that drops data (no error paths added)
- [x] `sortedKeys` ensures no unsorted map iteration for float accumulation
- [x] No `logrus.Fatalf` in new library code
- [x] No resource allocation loops (no rollback needed)

---

## Appendix: File-Level Implementation Details

### File: `sim/routing.go`

**Purpose:** Add `EffectiveLoad()` method and refactor 3 routing policies.

**Changes:**
- Add `EffectiveLoad() int` method on `RoutingSnapshot` (after line 18)
- `LeastLoaded.Route()`: replace `snapshots[i].QueueDepth + snapshots[i].BatchSize + snapshots[i].PendingRequests` with `snapshots[i].EffectiveLoad()` (lines 72, 76)
- `WeightedScoring.Route()`: replace `snap.QueueDepth + snap.BatchSize + snap.PendingRequests` with `snap.EffectiveLoad()` (line 142)
- `AlwaysBusiest.Route()`: replace `snapshots[i].QueueDepth + snapshots[i].BatchSize + snapshots[i].PendingRequests` with `snapshots[i].EffectiveLoad()` (lines 218, 222)

**Behavioral notes:**
- `EffectiveLoad()` returns `int` (not `int64`) because all 3 contributing fields are `int`
- Value receiver (not pointer) because `RoutingSnapshot` is a small value type
- No change to routing behavior — all 3 policies already use `QueueDepth + BatchSize + PendingRequests`

### File: `sim/metrics_utils.go`

**Purpose:** Add `NewRequestMetrics()` canonical constructor.

**Changes:**
- Add `NewRequestMetrics(req *Request, arrivedAt float64) RequestMetrics` after the `RequestMetrics` struct

**Behavioral notes:**
- Maps `req.AssignedInstance` → `HandledBy` (the field names differ between Request and RequestMetrics)
- Uses `len(req.InputTokens)` and `len(req.OutputTokens)` for token counts (matching existing inline behavior)

### File: `sim/simulator.go`

**Purpose:** Update `InjectArrival` and `InjectArrivalAt` to use `NewRequestMetrics`.

**Changes:**
- `InjectArrival` (line 257): replace 8-line `RequestMetrics{}` literal with `NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)`
- `InjectArrivalAt` (line 273): same replacement

### File: `sim/workload_config.go`

**Purpose:** Refactor `generateWorkloadFromCSV` to call `InjectArrival` instead of inline event scheduling + metrics construction.

**Changes:**
- Remove lines 83-93 (manual `Schedule` + inline `RequestMetrics{}`)
- Replace with single `sim.InjectArrival(req)` call

**Behavioral notes:**
- `ArrivedAt` will now be `float64(int64(arrivalFloat * 1e6)) / 1e6` instead of `arrivalFloat` directly. This round-trips through the canonical int64 tick representation. Precision difference is < 1 microsecond.

### File: `sim/cluster/metrics.go`

**Purpose:** Add `sortedKeys()` helper and apply to `ComputeFitness`, `JainFairnessIndex`, `mapValues`.

**Changes:**
- Add `sortedKeys(m map[string]float64) []string` helper (unexported)
- `ComputeFitness`: iterate `sortedKeys(weights)` instead of `range weights`
- `JainFairnessIndex`: iterate `sortedKeys(throughputs)` instead of `range throughputs`
- `mapValues`: iterate `sortedKeys(m)` instead of `range m`

### File: `sim/cluster/cluster.go`

**Purpose:** Add 3 missing fields to `aggregateMetrics()`.

**Changes:**
- Inside the existing instance loop: add `merged.PreemptionCount += m.PreemptionCount`, `merged.KVAllocationFailures += m.KVAllocationFailures`, `merged.CacheHitRate += m.CacheHitRate`, `merged.KVThrashingRate += m.KVThrashingRate`
- After the loop: divide CacheHitRate and KVThrashingRate by instance count to compute averages

### File: `sim/cluster/counterfactual.go`

**Purpose:** Update load-based fallback scoring to use `EffectiveLoad()`.

**Changes:**
- Line 51: replace `snap.QueueDepth + snap.BatchSize + snap.PendingRequests` with `snap.EffectiveLoad()`
