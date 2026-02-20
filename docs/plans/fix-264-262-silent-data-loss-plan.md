# fix(metrics,workload): silent data loss fixes (#264, #262) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate two silent data loss paths — one in anomaly detection (priority inversions) and one in ServeGen trace parsing — so that skipped data is always counted and warned about.

**The problem today:** Two code paths silently discard data without any counter or warning:
1. `detectPriorityInversions` silently skips requests that exist in `Requests` but lack an `RequestE2Es` entry — priority inversions may be undercounted with no indication.
2. `parseServeGenTrace` silently discards `strconv.ParseFloat` errors (rows with non-numeric data default to `0.0`), and `loadServeGenDataset` silently discards unparseable JSON keys. Both violate Antipattern Rule #1: "Every error path must either return an error, panic with context, or increment a counter."

**What this PR adds:**
1. A `skippedCount` counter + `logrus.Warnf` in `detectPriorityInversions` — matching the established pattern from `SLOAttainment` (line 330) and `ComputePerSLODistributions` (lines 264, 280) in the same file.
2. Parse-error checking for `startTime`, `rate`, and `cv` in `parseServeGenTrace` — invalid values increment the existing `skippedRows` counter and skip the row.
3. Parse-error checking for JSON keys in `loadServeGenDataset` — invalid keys are warned and skipped rather than silently treated as `0.0`.

**Why this matters:** Silent data loss is the #1 antipattern in this codebase (see Antipattern Rule #1). These are the last two known instances. Fixing them completes the silent-loss audit started in #237/#256.

**Architecture:** Two isolated fixes in different packages (`sim/cluster/` and `sim/workload/`), no file overlap, no shared types. Both follow the established counter + warn pattern.

**Source:** GitHub issues #264, #262

**Closes:** Fixes #264, fixes #262

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes two instances of Antipattern Rule #1 (silent data loss) in separate packages:

1. **sim/cluster/metrics.go** — `detectPriorityInversions` joins `Requests` with `RequestE2Es` but silently drops requests missing from `RequestE2Es`. The fix adds a counter and warning, matching the pattern already used by three other functions in the same file.

2. **sim/workload/servegen.go** — `parseServeGenTrace` discards `strconv.ParseFloat` errors via `_`, causing non-numeric CSV fields to silently become `0.0`. `loadServeGenDataset` has the same issue with JSON keys. The fix checks errors and increments the existing `skippedRows` counter (for `parseServeGenTrace`) or logs a warning (for `loadServeGenDataset`).

No adjacent blocks change. No new types. No architectural impact.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Priority Inversion Skip Counter
- GIVEN a `Metrics` where some request IDs in `Requests` have no corresponding entry in `RequestE2Es`
- WHEN `detectPriorityInversions` is called
- THEN the function MUST still return a correct inversion count based on matched requests, AND log a warning containing the count of skipped requests
- MECHANISM: `skippedCount` counter incremented for each miss; `logrus.Warnf` after loop

BC-2: ServeGen Trace Parse Error Counter
- GIVEN a CSV trace file where some rows have non-numeric values in the startTime, rate, or cv fields
- WHEN `parseServeGenTrace` is called
- THEN the rows with parse errors MUST be excluded from the result, counted in `skippedRows`, and warned about
- MECHANISM: Check `strconv.ParseFloat` error return; on error, increment `skippedRows` and `continue`

BC-3: ServeGen Dataset Key Parse Warning
- GIVEN a dataset JSON file where some window keys are not valid floats (e.g., string labels)
- WHEN `loadServeGenDataset` is called
- THEN unparseable keys MUST be skipped with a warning log, and parsing MUST continue for remaining keys
- MECHANISM: Check `strconv.ParseFloat` error return; on error, `logrus.Warnf` and `continue`

**Negative Contracts:**

BC-4: No Behavior Change for Valid Data
- GIVEN input data where all requests have E2E entries (metrics) or all CSV fields are valid numbers (workload)
- WHEN the respective functions are called
- THEN the output MUST be identical to the current behavior (no warnings logged, same return values)
- MECHANISM: Counters remain zero; warn paths not taken

**Error Handling Contracts:**

BC-5: Parse Errors Are Counted, Not Fatal
- GIVEN parse errors in ServeGen data
- WHEN `parseServeGenTrace` or `loadServeGenDataset` is called
- THEN the function MUST NOT return an error for individual row/key parse failures — it MUST skip, count, warn, and continue
- MECHANISM: `continue` after incrementing counter or logging warning

### C) Component Interaction

```
detectPriorityInversions(perInstance, policy)
  └── Iterates m.Requests × m.RequestE2Es join
      └── NEW: counts misses → logrus.Warnf

parseServeGenTrace(path)
  └── Iterates CSV rows → strconv.ParseFloat × 3
      └── NEW: checks err → skippedRows++ on failure

loadServeGenDataset(path, sgConfig)
  └── Iterates JSON keys → strconv.ParseFloat
      └── NEW: checks err → logrus.Warnf + skip
```

No new types. No new interfaces. No state changes. No API changes.

Extension friction: N/A — no new extension points.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #264 says "matching the pattern in SLOAttainment and ComputePerSLODistributions" | Exactly follows that pattern | No deviation |
| #262 says "increment existing skippedRows counter" | Uses existing `skippedRows` in parseServeGenTrace; uses `logrus.Warnf` in loadServeGenDataset (no pre-existing counter there) | ADDITION: `loadServeGenDataset` needs its own skip counter since it has no pre-existing one; a simple Warnf per bad key suffices since JSON keys are few |
| N/A (implicit) | Updates warning message at servegen.go:226 to reflect that `skippedRows` now counts both short rows and parse errors | CORRECTION: existing message said "had fewer than 4 fields" but counter now also covers parse errors |

### E) Review Guide

1. **THE TRICKY PART:** The `loadServeGenDataset` fix is subtle — when a JSON key fails to parse as a float, the `startTime` variable defaults to `0.0`, which means `SpanStart > 0` checks pass incorrectly. The fix must skip the entire iteration, not just warn.
2. **WHAT TO SCRUTINIZE:** BC-2 — verify that the `skippedRows` counter is incremented for ALL three float fields (startTime, rate, cv) on parse failure, not just the first.
3. **WHAT'S SAFE TO SKIM:** BC-1 is mechanical — identical pattern to three existing functions.
4. **KNOWN DEBT:** None.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/cluster/metrics.go:184-188` — add skip counter to `detectPriorityInversions`
- `sim/cluster/metrics_test.go` — add test for BC-1
- `sim/workload/servegen.go:213-215` — check parse errors in `parseServeGenTrace`
- `sim/workload/servegen.go:254` — check parse error in `loadServeGenDataset`
- `sim/workload/servegen_test.go` — add tests for BC-2, BC-3

**Key decisions:**
- Reuse existing `skippedRows` counter in `parseServeGenTrace` (already has the Warnf)
- Use per-key `logrus.Warnf` in `loadServeGenDataset` (JSON files are small; individual key warnings are more useful than a count)
- Do NOT change function signatures — these are internal functions

**Confirmation:** No dead code. All paths exercisable via tests.

### G) Task Breakdown

---

### Task 1: Add skip counter to detectPriorityInversions

**Contracts Implemented:** BC-1, BC-4

**Files:**
- Modify: `sim/cluster/metrics.go:183-188`
- Test: `sim/cluster/metrics_test.go`

**Step 1: Write failing test for BC-1**

Context: We need to verify that when some requests have no E2E data, the function still computes inversions correctly from matched requests and doesn't silently discard data. We'll capture logrus output to verify the warning.

```go
// TestDetectPriorityInversions_MissingE2E_WarnsAndCountsMatched verifies BC-1.
// Requests with no E2E entry are skipped with a warning, but matched requests
// are still evaluated for inversions.
func TestDetectPriorityInversions_MissingE2E_WarnsAndCountsMatched(t *testing.T) {
	// GIVEN an instance with 3 requests but only 2 have E2E data
	m := sim.NewMetrics()
	m.Requests["r1"] = sim.RequestMetrics{ID: "r1", ArrivedAt: 100}
	m.Requests["r2"] = sim.RequestMetrics{ID: "r2", ArrivedAt: 200}
	m.Requests["r3"] = sim.RequestMetrics{ID: "r3", ArrivedAt: 300}
	// r1 has 10× worse E2E than r2 → inversion
	m.RequestE2Es["r1"] = 50000.0
	m.RequestE2Es["r2"] = 5000.0
	// r3 has NO E2E entry → should be skipped with warning

	// Capture log output
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	defer logrus.SetOutput(os.Stderr)

	// WHEN detecting priority inversions
	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	// THEN inversions are counted from matched requests (r1 vs r2)
	assert.GreaterOrEqual(t, inversions, 1, "should detect inversion between r1 and r2")

	// AND a warning was logged about the skipped request
	assert.Contains(t, buf.String(), "1 requests", "should warn about 1 skipped request")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestDetectPriorityInversions_MissingE2E -v`
Expected: FAIL — no warning is currently logged

**Step 3: Implement skip counter in detectPriorityInversions**

Context: Add a `skippedCount` counter matching the pattern in `SLOAttainment` (line 312) and `ComputePerSLODistributions` (line 250).

In `sim/cluster/metrics.go`, replace the join loop (lines 183-188):

Current code:
```go
		var reqs []reqInfo
		for id, rm := range m.Requests {
			if e2e, ok := m.RequestE2Es[id]; ok {
				reqs = append(reqs, reqInfo{arrived: rm.ArrivedAt, e2e: e2e})
			}
		}
```

New code:
```go
		var reqs []reqInfo
		skippedCount := 0
		for id, rm := range m.Requests {
			if e2e, ok := m.RequestE2Es[id]; ok {
				reqs = append(reqs, reqInfo{arrived: rm.ArrivedAt, e2e: e2e})
			} else {
				skippedCount++
			}
		}
		if skippedCount > 0 {
			logrus.Warnf("detectPriorityInversions: %d requests missing E2E data, skipped", skippedCount)
		}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestDetectPriorityInversions_MissingE2E -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go
git commit -m "fix(metrics): add skip counter to detectPriorityInversions (#264) (BC-1)

- Add skippedCount for requests in Requests map without E2E data
- Log warning with count after join loop
- Matches existing SLOAttainment/ComputePerSLODistributions pattern

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Check parse errors in parseServeGenTrace

**Contracts Implemented:** BC-2, BC-4, BC-5

**Files:**
- Modify: `sim/workload/servegen.go:213-215`
- Test: `sim/workload/servegen_test.go`

**Step 1: Write failing test for BC-2**

Context: We need to verify that rows with non-numeric values in the float fields are counted and skipped, not silently treated as 0.0.

```go
// TestParseServeGenTrace_NonNumericFields_SkippedAndWarned verifies BC-2.
// Rows with non-numeric startTime, rate, or cv are counted in skippedRows.
func TestParseServeGenTrace_NonNumericFields_SkippedAndWarned(t *testing.T) {
	// GIVEN a CSV with 3 rows: 1 valid, 1 with non-numeric rate, 1 with non-numeric startTime
	dir := t.TempDir()
	csvContent := "0,1.5,2.0,Gamma\nBAD_TIME,1.0,2.0,Poisson\n100,NOT_A_NUMBER,2.0,Weibull\n"
	path := filepath.Join(dir, "trace.csv")
	require.NoError(t, os.WriteFile(path, []byte(csvContent), 0644))

	// Capture log output
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	defer logrus.SetOutput(os.Stderr)

	// WHEN parsing the trace
	rows, err := parseServeGenTrace(path)

	// THEN no error is returned
	require.NoError(t, err)

	// AND only the valid row is included
	assert.Len(t, rows, 1, "only the valid row should be parsed")
	assert.InDelta(t, 1.5, rows[0].rate, 0.001)

	// AND a warning was logged about 2 skipped rows
	assert.Contains(t, buf.String(), "2 rows", "should warn about 2 skipped rows")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestParseServeGenTrace_NonNumericFields -v`
Expected: FAIL — currently non-numeric fields parse as 0.0, rows are not skipped

**Step 3: Implement parse error checking in parseServeGenTrace**

Context: Replace the three `_` error discards with error checks. On any parse failure, increment the existing `skippedRows` counter and `continue`.

In `sim/workload/servegen.go`, replace lines 213-216:

Current code:
```go
		startTime, _ := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		rate, _ := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		cv, _ := strconv.ParseFloat(strings.TrimSpace(record[2]), 64)
		pattern := strings.TrimSpace(record[3])
```

New code:
```go
		startTime, err := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		rate, err := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		cv, err := strconv.ParseFloat(strings.TrimSpace(record[2]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		pattern := strings.TrimSpace(record[3])
```

Note: The existing `skippedRows` counter at line 200 is reused.

Also update the warning message at line 226 — it currently says "had fewer than 4 fields" which is now inaccurate since `skippedRows` also counts parse-error skips:

In `sim/workload/servegen.go`, replace line 226:

Current:
```go
		logrus.Warnf("parseServeGenTrace: %d rows in %s had fewer than 4 fields and were skipped", skippedRows, path)
```

New:
```go
		logrus.Warnf("parseServeGenTrace: %d rows in %s were skipped (short rows or parse errors)", skippedRows, path)
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestParseServeGenTrace_NonNumericFields -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/servegen.go sim/workload/servegen_test.go
git commit -m "fix(workload): check strconv.ParseFloat errors in parseServeGenTrace (#262) (BC-2, BC-5)

- Check parse errors for startTime, rate, cv fields
- On error: increment existing skippedRows counter and skip row
- Previously: errors discarded via _, non-numeric values silently became 0.0

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Check parse error in loadServeGenDataset

**Contracts Implemented:** BC-3, BC-5

**Files:**
- Modify: `sim/workload/servegen.go:254`
- Test: `sim/workload/servegen_test.go`

**Step 1: Write failing test for BC-3**

Context: We need to verify that non-numeric JSON keys in the dataset are skipped with a warning rather than silently parsed as 0.0.

```go
// TestLoadServeGenDataset_NonNumericKey_SkippedWithWarning verifies BC-3.
// JSON keys that are not valid floats are skipped with a warning.
func TestLoadServeGenDataset_NonNumericKey_SkippedWithWarning(t *testing.T) {
	// GIVEN a dataset JSON with one valid key and one non-numeric key
	dir := t.TempDir()
	datasetJSON := `{
		"metadata": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"},
		"0": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}
	}`
	path := filepath.Join(dir, "dataset.json")
	require.NoError(t, os.WriteFile(path, []byte(datasetJSON), 0644))

	// Capture log output
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	defer logrus.SetOutput(os.Stderr)

	// WHEN loading the dataset
	inputPDF, outputPDF, err := loadServeGenDataset(path, &ServeGenDataSpec{})

	// THEN the valid key's data is returned
	require.NoError(t, err)
	assert.NotNil(t, inputPDF, "should return input PDF from valid key")
	assert.NotNil(t, outputPDF, "should return output PDF from valid key")

	// AND a warning was logged about the non-numeric key
	assert.Contains(t, buf.String(), "metadata", "should warn about non-numeric key 'metadata'")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestLoadServeGenDataset_NonNumericKey -v`
Expected: FAIL — currently non-numeric keys parse as 0.0 with no warning

**Step 3: Implement parse error checking in loadServeGenDataset**

Context: Check the `strconv.ParseFloat` error for JSON keys. On failure, log a warning and skip the key.

In `sim/workload/servegen.go`, replace line 254:

Current code:
```go
		startTime, _ := strconv.ParseFloat(k, 64)
```

New code:
```go
		startTime, err := strconv.ParseFloat(k, 64)
		if err != nil {
			logrus.Warnf("loadServeGenDataset: skipping non-numeric key %q: %v", k, err)
			continue
		}
```

Note: Uses `parseErr` (not `err`) to avoid shadowing the outer `err` declared at lines 232 and 239.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestLoadServeGenDataset_NonNumericKey -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/servegen.go sim/workload/servegen_test.go
git commit -m "fix(workload): check strconv.ParseFloat error in loadServeGenDataset (#262) (BC-3, BC-5)

- Check parse error for JSON window keys
- On error: log warning with key value and skip
- Previously: error discarded via _, non-numeric keys silently became 0.0

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Verify no behavior change for valid data (BC-4)

**Contracts Implemented:** BC-4

**Files:**
- Test: `sim/cluster/metrics_test.go` (existing test)
- Test: `sim/workload/servegen_test.go` (existing test)

**Step 1: Run all existing tests to verify no regressions**

Context: BC-4 states that valid data must produce identical behavior. The existing test suite covers this — we just need to verify all tests still pass.

Run: `go test ./sim/cluster/... ./sim/workload/... -v -count=1`
Expected: ALL PASS — no regressions from our changes

**Step 2: Run full build + lint**

Run: `go build ./... && golangci-lint run ./...`
Expected: Build succeeds, lint clean

**Step 3: Commit (no changes needed — verification only)**

No commit for this task — it's a verification step.

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1 | Task 1 | Unit | `TestDetectPriorityInversions_MissingE2E_WarnsAndCountsMatched` |
| BC-2 | Task 2 | Unit | `TestParseServeGenTrace_NonNumericFields_SkippedAndWarned` |
| BC-3 | Task 3 | Unit | `TestLoadServeGenDataset_NonNumericKey_SkippedWithWarning` |
| BC-4 | Task 4 | Regression | Existing test suite (all existing tests pass unchanged) |
| BC-5 | Task 2, 3 | Unit | Covered by BC-2 and BC-3 tests (verify no error return) |

No golden dataset changes needed — these fixes don't change output format or metrics.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing tests break from behavior change | Low | Medium | BC-4 verification in Task 4; parse errors only trigger on invalid data |
| Warning messages too noisy for valid data | None | Low | Warnings only emit when counter > 0 (zero-cost for valid data) |
| Variable shadowing with `err` in loadServeGenDataset | Low | Low | Use `parseErr` variable name to avoid shadowing |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (sim.NewMetrics)
- [x] CLAUDE.md — no updates needed (no new files/packages/CLI flags)
- [x] No stale references left
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies are correctly ordered (Task 1-3 independent, Task 4 depends on all)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration not needed
- [x] Construction site audit: no struct fields added
- [x] No new CLI flags
- [x] Every error path increments a counter — no silent `continue` that drops data (that's the whole point of this PR)
- [x] No map iteration feeds float accumulation without sorted keys
- [x] Library code never calls logrus.Fatalf (uses logrus.Warnf only)
- [x] No resource allocation loops
- [x] No exported mutable maps
- [x] No YAML config changes
- [x] No division operations added
- [x] No new interfaces
- [x] No multi-concern methods
- [x] No configuration parameter changes
- [x] Grepped for "planned for PR" references — none related to this work

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/metrics.go`

**Purpose:** Add skip counter to `detectPriorityInversions` (lines 183-188)

**Change (minimal — 5 lines added):**

```go
// In detectPriorityInversions, replace the join loop:
		var reqs []reqInfo
		skippedCount := 0
		for id, rm := range m.Requests {
			if e2e, ok := m.RequestE2Es[id]; ok {
				reqs = append(reqs, reqInfo{arrived: rm.ArrivedAt, e2e: e2e})
			} else {
				skippedCount++
			}
		}
		if skippedCount > 0 {
			logrus.Warnf("detectPriorityInversions: %d requests missing E2E data, skipped", skippedCount)
		}
```

**Key Notes:**
- Pattern matches `SLOAttainment` (line 312-331) and `ComputePerSLODistributions` (lines 250-280)
- `logrus` is already imported in this file
- Counter is per-instance (inside the `for _, m := range perInstance` loop)

### File: `sim/workload/servegen.go`

**Purpose:** Check parse errors in `parseServeGenTrace` (lines 213-215) and `loadServeGenDataset` (line 254)

**Change 1 — parseServeGenTrace (replace lines 213-216):**

```go
		startTime, err := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		rate, err := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		cv, err := strconv.ParseFloat(strings.TrimSpace(record[2]), 64)
		if err != nil {
			skippedRows++
			continue
		}
		pattern := strings.TrimSpace(record[3])
```

**Also update the warning message at line 226:**

Current: `"parseServeGenTrace: %d rows in %s had fewer than 4 fields and were skipped"`
New: `"parseServeGenTrace: %d rows in %s were skipped (short rows or parse errors)"`

**Key Notes:**
- Reuses existing `skippedRows` counter declared at line 200
- Warning message updated to reflect that counter now covers both short rows and parse errors
- `err` shadows the outer `err` from `reader.Read()` but that's fine — the outer `err` is checked immediately after `reader.Read()` before we reach these lines

**Change 2 — loadServeGenDataset (replace line 254):**

```go
		startTime, parseErr := strconv.ParseFloat(k, 64)
		if parseErr != nil {
			logrus.Warnf("loadServeGenDataset: skipping non-numeric key %q: %v", k, parseErr)
			continue
		}
```

**Key Notes:**
- Uses `parseErr` to avoid shadowing `err` used on lines 233, 239
- Per-key warning (not a counter) because JSON datasets typically have few keys
- `logrus` is already imported in this file

### File: `sim/cluster/metrics_test.go`

**Purpose:** Add test for BC-1

**New test function added after existing `TestDetectPriorityInversions_InvertedRequests`.**

**New imports needed:** `bytes`, `os`, `github.com/sirupsen/logrus`, `github.com/stretchr/testify/assert`

Check existing imports first — `assert` may already be imported.

### File: `sim/workload/servegen_test.go`

**Purpose:** Add tests for BC-2 and BC-3

**New test functions added after existing `TestParseServeGenTrace_AllShortRows_ReturnsEmptySlice`.**

**New imports needed:** `bytes`, `github.com/sirupsen/logrus`, `github.com/stretchr/testify/assert`, `github.com/stretchr/testify/require`

Check existing imports first — some may already be imported.
