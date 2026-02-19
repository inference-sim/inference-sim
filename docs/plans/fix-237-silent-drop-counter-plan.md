# fix(metrics): Add Counter for Silently Dropped Requests — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make dropped requests and skipped CSV rows visible by adding counters and warnings instead of silently continuing.

**The problem today:** `ComputePerSLODistributions` in `sim/cluster/metrics.go` silently skips requests that appear in `RequestTTFTs`/`RequestE2Es` but are missing from the `Requests` map — a bare `continue` with no counter and no log. The same file's `SLOAttainment` function handles the identical situation correctly with a `droppedCount` counter and `logrus.Warnf`. Similarly, `parseServeGenTrace` in `sim/workload/servegen.go` silently skips CSV rows with fewer than 4 fields. These silent drops violate Antipattern Rule #1 (no silent data loss).

**What this PR adds:**
1. **Dropped-request counter in per-SLO metrics** — when `ComputePerSLODistributions` encounters a request ID in the TTFT/E2E maps that's missing from the Requests map, it counts the occurrence and logs a warning with the total count after each loop
2. **Skipped-row counter in ServeGen trace parsing** — when `parseServeGenTrace` encounters a CSV row with fewer than 4 fields, it counts the occurrence and logs a warning with the total count after parsing completes

**Why this matters:** Silent data loss is the #1 source of correctness bugs in this codebase (see issue #183 where a silently-dropped request was encoded into golden test data for months). Consistent warning visibility across all metrics code prevents future silent-drop bugs from hiding.

**Architecture:** Pure behavioral fix — no new types, no interface changes, no new files. Adds `droppedCount` counters to two loops in `sim/cluster/metrics.go` and a `skippedRows` counter to one loop in `sim/workload/servegen.go`, following the existing pattern in `SLOAttainment` (same file, line 302).

**Source:** GitHub issue #237

**Closes:** Fixes #237

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds counters and `logrus.Warnf` calls to three silent `continue` sites:
- Two loops in `ComputePerSLODistributions` (TTFT and E2E) that skip requests missing from the Requests map
- One loop in `parseServeGenTrace` that skips CSV rows with < 4 fields

The fix mirrors the existing `SLOAttainment` pattern (line 302-322 of the same file) where `droppedCount` is incremented and a warning is logged after the loop. No new types, interfaces, or behavioral changes — dropped items are still dropped, but now they're visible.

No deviations from issue #237.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: TTFT Dropped Request Warning
- GIVEN a `Metrics` object with request IDs in `RequestTTFTs` that are missing from `Requests`
- WHEN `ComputePerSLODistributions` is called
- THEN it MUST log a warning containing the count of dropped requests AND still compute correct distributions for the requests that are present
- MECHANISM: `droppedCount` counter incremented on `!ok`, `logrus.Warnf` after loop if count > 0

BC-2: E2E Dropped Request Warning
- GIVEN a `Metrics` object with request IDs in `RequestE2Es` that are missing from `Requests`
- WHEN `ComputePerSLODistributions` is called
- THEN it MUST log a warning containing the count of dropped requests AND still compute correct distributions for the requests that are present
- MECHANISM: Separate `droppedCount` for E2E loop (independent of TTFT loop)

BC-3: ServeGen Skipped Row Warning
- GIVEN a ServeGen trace CSV file containing rows with fewer than 4 fields
- WHEN `parseServeGenTrace` is called
- THEN it MUST log a warning containing the count of skipped rows AND still parse all valid rows correctly
- MECHANISM: `skippedRows` counter incremented on `len(record) < 4`, `logrus.Warnf` after loop if count > 0

**Negative Contracts:**

BC-4: No Warning When All Requests Present
- GIVEN a `Metrics` object where every request ID in `RequestTTFTs` and `RequestE2Es` also exists in `Requests`
- WHEN `ComputePerSLODistributions` is called
- THEN it MUST NOT log any warning
- MECHANISM: `droppedCount` stays 0, guard `if droppedCount > 0` prevents spurious logging

BC-5: No Warning When All CSV Rows Valid
- GIVEN a ServeGen trace CSV where every row has >= 4 fields
- WHEN `parseServeGenTrace` is called
- THEN it MUST NOT log any warning
- MECHANISM: `skippedRows` stays 0, guard prevents spurious logging

BC-6: Existing Behavior Unchanged
- GIVEN existing callers of `ComputePerSLODistributions` and `parseServeGenTrace`
- WHEN these functions are called with well-formed data
- THEN return values MUST be byte-identical to pre-change behavior
- MECHANISM: Only adding counter + log; no changes to return values, types, or control flow

### C) Component Interaction

```
ComputePerSLODistributions(aggregated *sim.Metrics) → map[string]*SLOMetrics
  ├── Reads: aggregated.RequestTTFTs, aggregated.RequestE2Es, aggregated.Requests
  ├── Outputs: per-SLO-class distributions (unchanged)
  └── Side effect: logrus.Warnf to stderr if dropped > 0 (NEW)

parseServeGenTrace(path string) → ([]serveGenTraceRow, error)
  ├── Reads: CSV file at path
  ├── Outputs: parsed rows (unchanged)
  └── Side effect: logrus.Warnf to stderr if skipped > 0 (NEW)
```

No new state. No new types. No API changes. Warnings go to stderr via logrus (correct per output channel separation — diagnostic messages use logrus, not fmt).

### D) Deviation Log

No deviations from issue #237.

The issue mentions `sim/workload/servegen.go:207` — the actual line is 206 (`if len(record) < 4`). This is a minor line-number drift, not a semantic deviation.

### E) Review Guide

- **The tricky part:** Nothing subtle — this is a mechanical pattern application. The only thing to verify is that the two `droppedCount` variables (TTFT and E2E) are independent (not accidentally shared).
- **What to scrutinize:** Verify the warning messages are distinct enough to tell TTFT drops from E2E drops.
- **What's safe to skim:** The test scaffolding — it follows existing patterns exactly.
- **Known debt:** None introduced. This PR reduces existing debt (the silent `continue` antipattern).

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/cluster/metrics.go:250-271` — add `droppedCount` to both loops in `ComputePerSLODistributions`
- `sim/workload/servegen.go:206-208` — add `skippedRows` counter in `parseServeGenTrace`

**Files to add tests to:**
- `sim/cluster/metrics_slo_test.go` — new test for dropped request warning behavior
- `sim/workload/servegen_test.go` — new test for skipped row warning behavior

**Key decisions:**
- Use `logrus.Warnf` (not `logrus.Errorf`) — dropped requests are anomalous but not fatal, matching `SLOAttainment` precedent
- Separate counters for TTFT and E2E loops — they can have different drop counts since the maps are independent
- No new import needed in `metrics.go` — `logrus` is already imported

### G) Task Breakdown

---

### Task 1: Add Dropped-Request Counter to ComputePerSLODistributions

**Contracts Implemented:** BC-1, BC-2, BC-4, BC-6

**Files:**
- Modify: `sim/cluster/metrics.go:250-271`
- Test: `sim/cluster/metrics_slo_test.go`

**Step 1: Write failing test for dropped-request warning behavior**

Context: We test that `ComputePerSLODistributions` still computes correct distributions when some request IDs are missing from the Requests map, and that those present requests are not affected.

```go
func TestComputePerSLODistributions_MissingRequests_StillComputesPresent(t *testing.T) {
	// GIVEN metrics with 10 TTFT entries and 10 E2E entries,
	// but only 7 have corresponding entries in the Requests map
	m := sim.NewMetrics()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.RequestTTFTs[id] = float64(100 + i)
		m.RequestE2Es[id] = float64(500 + i)
	}
	// Only 7 out of 10 have Requests entries
	for i := 0; i < 7; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "batch"}
	}

	// WHEN computing per-SLO distributions
	result := ComputePerSLODistributions(m)

	// THEN distributions are computed for the 7 present requests
	if result["batch"] == nil {
		t.Fatal("expected batch class in result")
	}
	if result["batch"].TTFT.Count != 7 {
		t.Errorf("TTFT count = %d, want 7 (only present requests)", result["batch"].TTFT.Count)
	}
	if result["batch"].E2E.Count != 7 {
		t.Errorf("E2E count = %d, want 7 (only present requests)", result["batch"].E2E.Count)
	}
}
```

**Step 2: Run test to verify it passes (baseline — this test validates existing correct behavior)**

Run: `go test ./sim/cluster/... -run TestComputePerSLODistributions_MissingRequests -v`
Expected: PASS (the existing logic already skips missing requests correctly; we're confirming the behavioral baseline before adding the counter)

**Step 3: Implement counters in ComputePerSLODistributions**

Context: Add `droppedCount` counters to both loops, following the `SLOAttainment` pattern at lines 302-322.

In `sim/cluster/metrics.go`, replace the TTFT loop (lines 250-260):

```go
	droppedTTFT := 0
	for reqID, ttft := range aggregated.RequestTTFTs {
		req, ok := aggregated.Requests[reqID]
		if !ok {
			droppedTTFT++
			continue
		}
		sloClass := req.SLOClass
		if sloClass == "" {
			sloClass = "default"
		}
		ttftByClass[sloClass] = append(ttftByClass[sloClass], ttft)
	}
	if droppedTTFT > 0 {
		logrus.Warnf("ComputePerSLODistributions: %d requests in RequestTTFTs missing from Requests map", droppedTTFT)
	}
```

Replace the E2E loop (lines 261-271):

```go
	droppedE2E := 0
	for reqID, e2e := range aggregated.RequestE2Es {
		req, ok := aggregated.Requests[reqID]
		if !ok {
			droppedE2E++
			continue
		}
		sloClass := req.SLOClass
		if sloClass == "" {
			sloClass = "default"
		}
		e2eByClass[sloClass] = append(e2eByClass[sloClass], e2e)
	}
	if droppedE2E > 0 {
		logrus.Warnf("ComputePerSLODistributions: %d requests in RequestE2Es missing from Requests map", droppedE2E)
	}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestComputePerSLODistributions -v`
Expected: PASS (all ComputePerSLODistributions tests pass, including the new one and the existing `_SegregatesCorrectly` test)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_slo_test.go
git commit -m "fix(metrics): add dropped-request counter to ComputePerSLODistributions (BC-1, BC-2, BC-4)

- Add droppedTTFT counter to TTFT loop with logrus.Warnf
- Add droppedE2E counter to E2E loop with logrus.Warnf
- Mirrors existing SLOAttainment pattern (same file, line 302)
- Add test confirming distributions computed correctly for present requests

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add Skipped-Row Counter to parseServeGenTrace

**Contracts Implemented:** BC-3, BC-5

**Files:**
- Modify: `sim/workload/servegen.go` (imports + `parseServeGenTrace` function at lines 189-222)
- Test: `sim/workload/servegen_test.go`

**Step 1: Write failing test for skipped-row behavior**

Context: We test that `parseServeGenTrace` correctly handles a CSV file where all rows have fewer than 4 fields. Go's `csv.Reader` defaults to `FieldsPerRecord = 0` (set from the first row), so the `len(record) < 4` guard only fires when the first row itself has < 4 fields (mixed-length rows trigger `csv.ErrFieldCount` before reaching our guard). We use a consistently short file to exercise this path.

```go
func TestParseServeGenTrace_AllShortRows_ReturnsEmptySlice(t *testing.T) {
	// GIVEN a CSV file where all rows have fewer than 4 fields
	dir := t.TempDir()
	csvContent := "short,row\nonly,two\n"
	path := filepath.Join(dir, "trace.csv")
	if err := os.WriteFile(path, []byte(csvContent), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN parsing the trace
	rows, err := parseServeGenTrace(path)

	// THEN no error is returned but the result is empty (all rows skipped)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rows) != 0 {
		t.Errorf("got %d rows, want 0 (all rows should be skipped)", len(rows))
	}
}
```

**Step 2: Run test to verify it passes (baseline)**

Run: `go test ./sim/workload/... -run TestParseServeGenTrace_AllShortRows -v`
Expected: PASS (existing logic already skips short rows correctly)

**Step 3: Implement counter in parseServeGenTrace**

Context: Add `skippedRows` counter, following the same pattern as the metrics counter.

In `sim/workload/servegen.go`, first add `logrus` to the imports:

```go
import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
)
```

Then replace the loop in `parseServeGenTrace` (lines 198-221):

```go
	reader := csv.NewReader(file)
	var rows []serveGenTraceRow
	skippedRows := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading trace CSV: %w", err)
		}
		if len(record) < 4 {
			skippedRows++
			continue
		}
		startTime, _ := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		rate, _ := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		cv, _ := strconv.ParseFloat(strings.TrimSpace(record[2]), 64)
		pattern := strings.TrimSpace(record[3])

		rows = append(rows, serveGenTraceRow{
			startTimeSec: startTime,
			rate:         rate,
			cv:           cv,
			pattern:      pattern,
		})
	}
	if skippedRows > 0 {
		logrus.Warnf("parseServeGenTrace: %d rows in %s had fewer than 4 fields and were skipped", skippedRows, path)
	}
	return rows, nil
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestParseServeGenTrace -v`
Expected: PASS

Also run all workload tests to verify no regressions:
Run: `go test ./sim/workload/... -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/servegen.go sim/workload/servegen_test.go
git commit -m "fix(workload): add skipped-row counter to parseServeGenTrace (BC-3, BC-5)

- Add skippedRows counter for CSV rows with < 4 fields
- Log warning with count and file path after loop
- Add test confirming short rows are skipped without error

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Full Verification and Documentation

**Contracts Implemented:** BC-6 (full regression check)

**Files:**
- No code changes — verification only

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: All packages PASS

**Step 2: Run full lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 3: Build verification**

Run: `go build ./...`
Expected: Build succeeds

**Step 4: Commit — no commit needed (verification-only task)**

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1     | Task 1 | Unit | `TestComputePerSLODistributions_MissingRequests_StillComputesPresent` |
| BC-2     | Task 1 | Unit | (Same test — covers both TTFT and E2E loops) |
| BC-3     | Task 2 | Unit | `TestParseServeGenTrace_AllShortRows_ReturnsEmptySlice` |
| BC-4     | Task 1 | Unit | `TestComputePerSLODistributions_SegregatesCorrectly` (existing — covers no-drop case) |
| BC-5     | Task 2 | Unit | `TestServeGenDataLoading_SyntheticDataset_ProducesClients` (existing — covers no-skip case) |
| BC-6     | Task 3 | Regression | Full test suite `go test ./...` |

No golden dataset changes needed — this fix adds warnings to stderr only, no changes to return values.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| logrus import missing in servegen.go | Low | Low | Explicitly add to imports in Task 2 |
| Warning message spam in normal operation | Low | Low | Guard with `if count > 0` — no warning when data is clean (BC-4, BC-5) |
| csv.Reader field count interaction | Medium | Low | Test uses consistently short rows; documented in Task 2 that mixed-length rows trigger csv.ErrFieldCount before reaching our guard |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — pure counter + log addition
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (sim.NewMetrics() from existing package)
- [x] CLAUDE.md — no update needed (no new files, packages, or CLI flags)
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — no deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration — not needed
- [x] Construction site audit — no new struct fields added
- [x] No new CLI flags
- [x] Every error path has counter or log — that's the entire point of this PR
- [x] No map iteration feeds float accumulation without sorted keys — N/A
- [x] Library code never calls logrus.Fatalf — uses logrus.Warnf only
- [x] No resource allocation loops
- [x] No exported mutable maps
- [x] No YAML config changes
- [x] No division operations added
- [x] No new interfaces
- [x] No methods spanning multiple concerns
- [x] No config struct changes
- [x] Grepped for "PR 237" / "#237" references — none in codebase

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/metrics.go`

**Purpose:** Add `droppedTTFT` and `droppedE2E` counters to the two loops in `ComputePerSLODistributions`.

**Changes (lines 250-271):**

The TTFT loop currently reads:
```go
for reqID, ttft := range aggregated.RequestTTFTs {
    req, ok := aggregated.Requests[reqID]
    if !ok {
        continue
    }
```

Replace with:
```go
droppedTTFT := 0
for reqID, ttft := range aggregated.RequestTTFTs {
    req, ok := aggregated.Requests[reqID]
    if !ok {
        droppedTTFT++
        continue
    }
```

After the TTFT loop, add:
```go
if droppedTTFT > 0 {
    logrus.Warnf("ComputePerSLODistributions: %d requests in RequestTTFTs missing from Requests map", droppedTTFT)
}
```

The E2E loop gets the identical treatment with `droppedE2E`.

**Key notes:**
- `logrus` is already imported (used by `SLOAttainment` in the same file)
- No changes to return type or return values
- Pattern matches `SLOAttainment` (line 302-322) exactly

### File: `sim/workload/servegen.go`

**Purpose:** Add `skippedRows` counter to `parseServeGenTrace` for rows with < 4 fields.

**Changes:**

1. Add `logrus` to imports (new import for this file)
2. Add `skippedRows := 0` before the parsing loop
3. Change `continue` to `skippedRows++; continue`
4. Add warning after loop

**Key notes:**
- This file currently does NOT import logrus — must add it
- The `csv.Reader` with default `FieldsPerRecord = 0` means rows with different field counts from the first row trigger `csv.ErrFieldCount` before reaching our guard. The `len(record) < 4` guard fires when ALL rows are consistently short (first row sets the expected count). The warning message includes the file path for debugging.
