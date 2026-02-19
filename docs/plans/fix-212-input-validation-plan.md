# Phase 5: Input Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent invalid numeric inputs from causing panics or wrong results by adding validation guards to fitness weight parsing and percentile calculation.

**The problem today:** Two input validation gaps exist. First, `ParseFitnessWeights` accepts NaN, Inf, and negative weight values without error — these propagate silently through fitness computation, producing meaningless scores. Second, `CalculatePercentile` panics on empty input (index out of bounds at `data[-1]`) — this is reachable when all completed requests have zero output tokens, making `AllITLs` empty.

**What this PR adds:**
1. **NaN/Inf/negative weight rejection** — `ParseFitnessWeights` returns an error for invalid weight values, with a clear message like `"invalid weight value for 'throughput': NaN"`
2. **Empty-input percentile guard** — `CalculatePercentile` returns 0 for empty input (matching the existing cluster-side `percentile()` function pattern), preventing index-out-of-bounds panics

**Why this matters:** These are the last two items in Phase 5 of the hardening plan. Fixing them enforces Antipattern Rule 3 (validate all numeric CLI flags) at the library level, protecting all current and future callers — including PR15 framework adapters that will embed `sim/` as a library.

**Architecture:** Both changes are localized. 5b modifies `ParseFitnessWeights` in `sim/cluster/metrics.go` (add guard after `strconv.ParseFloat`). 5c modifies `CalculatePercentile` in `sim/metrics_utils.go` (add early return). No new types, no interface changes, no new files.

**Source:** GitHub issue #212 (Phase 5 of hardening design doc `docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md`)

**Closes:** Fixes #212

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds two input validation guards:
1. `ParseFitnessWeights` rejects NaN, Inf, and negative weight values (returns error)
2. `CalculatePercentile` returns 0 for empty input (prevents panic)

Both are in the `sim/` package tree (library code), so they return errors — no `logrus.Fatalf`. The CLI boundary in `cmd/root.go` already converts `ParseFitnessWeights` errors to fatal messages; no CLI changes needed.

Sub-task 5a (Rate>0, #202) is already closed — only 5b and 5c remain.

No deviations from the design doc.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: NaN weight rejection
- GIVEN a fitness weight string containing a NaN value (e.g., `"throughput:NaN"`)
- WHEN `ParseFitnessWeights` is called
- THEN it MUST return a non-nil error whose message contains the key name and "NaN"
- MECHANISM: `math.IsNaN(val)` check after `strconv.ParseFloat`

BC-2: Inf weight rejection
- GIVEN a fitness weight string containing +Inf or -Inf (e.g., `"throughput:Inf"`, `"throughput:-Inf"`)
- WHEN `ParseFitnessWeights` is called
- THEN it MUST return a non-nil error whose message contains the key name
- MECHANISM: `math.IsInf(val, 0)` check after `strconv.ParseFloat`

BC-3: Negative weight rejection
- GIVEN a fitness weight string containing a negative value (e.g., `"throughput:-0.5"`)
- WHEN `ParseFitnessWeights` is called
- THEN it MUST return a non-nil error whose message contains the key name
- MECHANISM: `val < 0` check after `strconv.ParseFloat`

BC-4: Zero weight acceptance
- GIVEN a fitness weight string containing zero (e.g., `"throughput:0"`)
- WHEN `ParseFitnessWeights` is called
- THEN it MUST return the weight map with the zero value (no error)
- MECHANISM: Zero is a valid weight (disables a metric), so the guard is `val < 0`, not `val <= 0`

BC-5: Valid weights still accepted
- GIVEN a fitness weight string with valid positive values (e.g., `"throughput:0.5,p99_ttft:0.3"`)
- WHEN `ParseFitnessWeights` is called
- THEN it MUST return the correct weight map (no error), identical to current behavior
- MECHANISM: Guard only triggers for invalid values; existing parsing logic unchanged

BC-6: Empty percentile returns zero
- GIVEN an empty data slice (`[]T{}`)
- WHEN `CalculatePercentile` is called with any percentile value
- THEN it MUST return 0.0 (not panic)
- MECHANISM: `len(data) == 0` early return before rank computation

BC-7: Non-empty percentile unchanged
- GIVEN a non-empty sorted data slice
- WHEN `CalculatePercentile` is called
- THEN it MUST return the same value as the current implementation
- MECHANISM: Early return only activates for empty input; all other paths unchanged

**Negative Contracts:**

BC-8: No panic on empty percentile
- GIVEN an empty data slice
- WHEN `CalculatePercentile` is called
- THEN it MUST NOT panic (index out of bounds)
- MECHANISM: Early return guard before any array indexing

BC-9: No silent NaN propagation
- GIVEN NaN/Inf weights passed via `--fitness-weights`
- WHEN the CLI parses them
- THEN invalid values MUST NOT silently propagate to `ComputeFitness`
- MECHANISM: `ParseFitnessWeights` returns error; CLI converts to `logrus.Fatalf`

### C) Component Interaction

```
cmd/root.go
  └── calls cluster.ParseFitnessWeights(fitnessWeights)  [BC-1..5, BC-9]
        └── returns error for NaN/Inf/negative
        └── cmd/root.go: logrus.Fatalf on error (existing)

sim/metrics.go (SaveResults)
  └── calls CalculatePercentile(sortedData, p)  [BC-6..8]
        └── returns 0 for empty input
        └── existing callers unaffected (non-empty data)
```

No new types, interfaces, or state. No API contract changes (both functions already return error / float64). The `ParseFitnessWeights` error message format changes (new validation errors), but error message content is not a stability contract.

Extension friction: 0 files. These are leaf changes with no ripple effects.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Phase 5a: Rate>0 (#202) | Omitted | SCOPE_CHANGE: Already fixed in commit `69bdfed`, issue #202 closed |
| Design doc shows `CalculatePercentile(data []float64, p float64)` | Actual signature is generic: `CalculatePercentile[T IntOrFloat64](data []T, p float64)` | CORRECTION: Design doc used simplified signature; guard works identically for generic version |

### E) Review Guide

1. **THE TRICKY PART:** BC-4 (zero weight acceptance). The guard must be `val < 0`, not `val <= 0`. Zero is a legitimate weight that means "exclude this metric from fitness." Getting this wrong would break existing valid configurations.
2. **WHAT TO SCRUTINIZE:** The NaN/Inf check in BC-1/BC-2. Note that `strconv.ParseFloat` already parses `"NaN"` and `"Inf"` as valid float64 values — they don't cause a parse error. This is why a separate post-parse check is needed.
3. **WHAT'S SAFE TO SKIM:** BC-6/BC-7 (percentile guard) — this is a one-line early return matching an existing pattern in the cluster-side `percentile()` function.
4. **KNOWN DEBT:** None introduced. The existing test comment at `sim/metrics_test.go:176` documents this exact issue.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/cluster/metrics.go:448` — Add NaN/Inf/negative guard after `strconv.ParseFloat` in `ParseFitnessWeights`
- `sim/cluster/metrics_test.go` — Add test cases for NaN, Inf, negative, and zero weights
- `sim/metrics_utils.go:84` — Add empty-input early return to `CalculatePercentile`
- `sim/metrics_utils_test.go` (create) — Add test for empty-input percentile

**Key decisions:**
- Zero is valid (disabled metric), negative is invalid → `val < 0` not `val <= 0`
- Error message format: `"invalid weight for %q: must be a finite non-negative number, got %v"` — clear, actionable
- `CalculatePercentile` returns `0.0` for empty (not `NaN` or error) — matches `CalculateMean` and cluster-side `percentile()` patterns

**Confirmation:** No dead code. Both changes are immediately exercised by new tests and existing callers.

### G) Task Breakdown

---

### Task 1: NaN/Inf/negative fitness weight validation

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-4, BC-5, BC-9

**Files:**
- Modify: `sim/cluster/metrics.go:448` (add guard after ParseFloat)
- Modify: `sim/cluster/metrics_test.go` (add test cases)

**Step 1: Write failing tests for invalid weight values**

Context: We need tests that verify `ParseFitnessWeights` rejects NaN, +Inf, -Inf, and negative values while still accepting zero and positive values. These are table-driven to cover all cases concisely.

In `sim/cluster/metrics_test.go`, add after the existing `TestParseFitnessWeights_InvalidFormat` function:

```go
// TestParseFitnessWeights_InvalidValues_ReturnsError verifies BC-1, BC-2, BC-3.
func TestParseFitnessWeights_InvalidValues_ReturnsError(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"NaN value", "throughput:NaN"},
		{"positive Inf", "throughput:Inf"},
		{"negative Inf", "throughput:-Inf"},
		{"explicit +Inf", "throughput:+Inf"},
		{"negative weight", "throughput:-0.5"},
		{"negative one", "p99_ttft:-1"},
		{"NaN after valid", "throughput:0.5,p99_ttft:NaN"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseFitnessWeights(tt.input)
			if err == nil {
				t.Errorf("expected error for input %q, got nil", tt.input)
			}
		})
	}
}

// TestParseFitnessWeights_ZeroWeight_Accepted verifies BC-4.
func TestParseFitnessWeights_ZeroWeight_Accepted(t *testing.T) {
	weights, err := ParseFitnessWeights("throughput:0,p99_ttft:0.3")
	if err != nil {
		t.Fatalf("zero weight should be accepted, got error: %v", err)
	}
	if weights["throughput"] != 0.0 {
		t.Errorf("throughput: got %f, want 0.0", weights["throughput"])
	}
	if weights["p99_ttft"] != 0.3 {
		t.Errorf("p99_ttft: got %f, want 0.3", weights["p99_ttft"])
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/cluster/... -run "TestParseFitnessWeights_InvalidValues|TestParseFitnessWeights_ZeroWeight" -v`
Expected: `TestParseFitnessWeights_InvalidValues` FAILS (NaN/Inf/negative currently accepted). `TestParseFitnessWeights_ZeroWeight` PASSES (zero is already accepted).

**Step 3: Implement the validation guard**

Context: Add a check after `strconv.ParseFloat` succeeds but before storing the value in the map. Must check for NaN, Inf (both signs), and negative values.

In `sim/cluster/metrics.go`, replace the existing block at lines 446-449:

```go
		if err != nil {
			return nil, fmt.Errorf("invalid fitness weight value for %q: %w", key, err)
		}
		weights[key] = val
```

With:

```go
		if err != nil {
			return nil, fmt.Errorf("invalid fitness weight value for %q: %w", key, err)
		}
		if math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
			return nil, fmt.Errorf("invalid weight for %q: must be a finite non-negative number, got %v", key, val)
		}
		weights[key] = val
```

Ensure `"math"` is in the import block (it should already be imported for other uses in this file).

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -run "TestParseFitnessWeights" -v`
Expected: ALL ParseFitnessWeights tests PASS (including existing ValidInput, Empty, InvalidFormat)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go
git commit -m "fix(cluster): reject NaN/Inf/negative fitness weights (BC-1..5, BC-9)

- Add validation guard in ParseFitnessWeights after strconv.ParseFloat
- NaN, Inf, -Inf, and negative values now return descriptive errors
- Zero weights remain valid (disables a metric in fitness computation)
- Table-driven tests for all invalid value categories

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: CalculatePercentile empty-input guard

**Contracts Implemented:** BC-6, BC-7, BC-8

**Files:**
- Modify: `sim/metrics_utils.go:84` (add early return)
- Create: `sim/metrics_utils_test.go` (add test — file may not exist yet for percentile tests)

**Step 1: Write failing test for empty input**

Context: `CalculatePercentile` is a generic function accepting `[]T` where `T` is `int | int64 | float64`. We test with `[]float64{}` to match the most common call pattern (SaveResults uses both `[]float64` and `[]int64`). We also add a test with `[]int64{}` to verify the generic guard works for both types.

First, check if `sim/metrics_utils_test.go` exists. If not, create it. Add:

```go
package sim

import "testing"

// TestCalculatePercentile_EmptyInput_ReturnsZero verifies BC-6, BC-8.
func TestCalculatePercentile_EmptyInput_ReturnsZero(t *testing.T) {
	// GIVEN empty float64 slice
	// WHEN CalculatePercentile is called
	result := CalculatePercentile([]float64{}, 99)
	// THEN it returns 0 (not panic)
	if result != 0.0 {
		t.Errorf("expected 0.0 for empty input, got %f", result)
	}

	// Also verify with int64 (generic constraint covers both)
	resultInt := CalculatePercentile([]int64{}, 50)
	if resultInt != 0.0 {
		t.Errorf("expected 0.0 for empty int64 input, got %f", resultInt)
	}
}

// TestCalculatePercentile_SingleElement_ReturnsScaled verifies BC-7.
func TestCalculatePercentile_SingleElement_ReturnsScaled(t *testing.T) {
	// GIVEN a single-element slice
	// WHEN CalculatePercentile is called
	result := CalculatePercentile([]float64{1000.0}, 99)
	// THEN it returns the element divided by 1000 (ms conversion)
	if result != 1.0 {
		t.Errorf("expected 1.0 for single element 1000.0, got %f", result)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run "TestCalculatePercentile_EmptyInput" -v`
Expected: FAIL with panic: `runtime error: index out of range [-1]`

**Step 3: Implement the empty-input guard**

Context: Add an early return at the top of `CalculatePercentile`, before the rank computation that produces a negative index when `n=0`.

In `sim/metrics_utils.go`, replace lines 84-85:

```go
func CalculatePercentile[T IntOrFloat64](data []T, p float64) float64 {
	n := len(data)
```

With:

```go
func CalculatePercentile[T IntOrFloat64](data []T, p float64) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestCalculatePercentile" -v`
Expected: ALL CalculatePercentile tests PASS

**Step 5: Run full test suite to verify no regressions**

Run: `go test ./... -count=1`
Expected: ALL PASS (the guard only activates for empty input; existing non-empty callers unaffected)

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 7: Commit with contract reference**

```bash
git add sim/metrics_utils.go sim/metrics_utils_test.go
git commit -m "fix(sim): guard CalculatePercentile against empty input (BC-6..8)

- Return 0 for empty data slice (matches CalculateMean and cluster percentile patterns)
- Prevents index-out-of-bounds panic when AllITLs is empty
- Add tests for empty float64, empty int64, and single-element inputs

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestParseFitnessWeights_InvalidValues_ReturnsError (NaN case) |
| BC-2 | Task 1 | Unit | TestParseFitnessWeights_InvalidValues_ReturnsError (Inf cases) |
| BC-3 | Task 1 | Unit | TestParseFitnessWeights_InvalidValues_ReturnsError (negative cases) |
| BC-4 | Task 1 | Unit | TestParseFitnessWeights_ZeroWeight_Accepted |
| BC-5 | Task 1 | Unit | TestParseFitnessWeights_ValidInput (existing, unchanged) |
| BC-6 | Task 2 | Unit | TestCalculatePercentile_EmptyInput_ReturnsZero |
| BC-7 | Task 2 | Unit | TestCalculatePercentile_SingleElement_ReturnsScaled |
| BC-8 | Task 2 | Unit | TestCalculatePercentile_EmptyInput_ReturnsZero (no-panic = implicit) |
| BC-9 | Task 1 | Unit | TestParseFitnessWeights_InvalidValues_ReturnsError (NaN propagation blocked at parse level) |

No golden dataset changes needed. No shared test infrastructure needed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Zero weight rejected by mistake | Low | High (breaks valid configs) | BC-4 test explicitly verifies zero acceptance; guard uses `val < 0` not `val <= 0` | Task 1 |
| `strconv.ParseFloat` already rejects NaN/Inf | Low | None (would make guard redundant, not harmful) | Verified: `strconv.ParseFloat("NaN", 64)` returns `(NaN, nil)` — the guard IS needed | Task 1 |
| Empty percentile guard changes existing behavior | None | N/A | Guard only activates for `len(data) == 0`; all existing callers pass non-empty slices (guarded by `CompletedRequests > 0`) | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions.
- [x] No feature creep beyond PR scope.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes without explicit contract updates.
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint.
- [x] Shared test helpers used from existing shared test package (not duplicated locally).
- [x] CLAUDE.md update not needed (no new files, packages, CLI flags, or file organization changes).
- [x] No stale references left in CLAUDE.md.
- [x] Deviation log reviewed — no unresolved deviations.
- [x] Each task produces working, testable code (no scaffolding).
- [x] Task dependencies are correctly ordered (Task 1 and Task 2 are independent).
- [x] All contracts are mapped to specific tasks.
- [x] Golden dataset regeneration not needed (no output format or metric changes).
- [x] Construction site audit: no struct fields added.
- [x] No new CLI flags.
- [x] Every error path returns error (no silent continue).
- [x] No map iteration feeds float accumulation (ParseFitnessWeights iterates pairs, not accumulating floats).
- [x] Library code (sim/, sim/cluster/) never calls logrus.Fatalf — errors returned to callers.
- [x] No resource allocation loops (no rollback needed).
- [x] No exported mutable maps.
- [x] No YAML config structs modified.
- [x] No YAML loading changes.
- [x] No division operations added.
- [x] No new interfaces.
- [x] No methods spanning multiple concerns.
- [x] No configuration parameters added.
- [x] Grepped for "planned for PR 5" / "Phase 5" references — the design doc Phase 5 section itself is the reference; no stale TODO comments in code.
- [x] Macro plan update not needed (this is a hardening sub-issue, not a macro plan PR).

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/metrics.go`

**Purpose:** Add NaN/Inf/negative validation to `ParseFitnessWeights`.

**Change:** Insert 3 lines after the existing `strconv.ParseFloat` error check (line 448), before `weights[key] = val` (line 449):

```go
if math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
    return nil, fmt.Errorf("invalid weight for %q: must be a finite non-negative number, got %v", key, val)
}
```

**Key notes:**
- `math.IsInf(val, 0)` checks both +Inf and -Inf (second arg 0 means either sign)
- `val < 0` rejects negative but NOT zero (zero is valid per BC-4)
- `math` is already imported in this file (used by `percentile()` and `JainFairnessIndex()`)
- Error message includes the key name so the user knows which weight is invalid

### File: `sim/cluster/metrics_test.go`

**Purpose:** Add test cases for NaN/Inf/negative/zero weight validation.

**Change:** Add two new test functions after `TestParseFitnessWeights_InvalidFormat` (line 274).

### File: `sim/metrics_utils.go`

**Purpose:** Guard `CalculatePercentile` against empty input.

**Change:** Insert 3 lines after `n := len(data)` (line 85):

```go
if n == 0 {
    return 0
}
```

**Key notes:**
- Returns `0`, not `0.0` — Go infers the float64 return type
- Matches the pattern used by `CalculateMean` (line 106-108) and cluster-side `percentile()` (line 65-67)
- The `/1000` ms conversion is irrelevant for empty input (0/1000 = 0)

### File: `sim/metrics_utils_test.go`

**Purpose:** Test `CalculatePercentile` edge cases (empty input, single element).

**Change:** Create new file if it doesn't exist, or append to existing file. Tests verify the empty guard and confirm existing behavior for non-empty input.
