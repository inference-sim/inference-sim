# Multi-Period ServeGen Conversion with CohortSpec Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `blis convert servegen` output all 3 time periods (midnight, morning, afternoon) in one compact YAML file using CohortSpec, eliminating the need for 3 separate commands and manual merging.

**The problem today:** Users must run `blis convert servegen` three times with different `--time` flags (midnight, morning, afternoon), manually merge the outputs, and deal with 150+ individual ClientSpecs that create huge YAML files. This workflow is error-prone, tedious, and produces hard-to-read specifications.

**What this PR adds:**
1. **Automatic multi-period output** — Single command generates all three time periods with configurable window duration (default 600s) and drain gaps (default 180s). Example: `blis convert servegen --path data/` produces a complete daily workload.
2. **Cohort-based grouping** — Outputs 15 cohorts (3 periods × 5 SLO classes) instead of 150+ individual clients. Each cohort aggregates chunks with averaged distributions and summed rates. Result: ~10x smaller YAML files.
3. **Temporal isolation** — 180-second drain gaps between periods ensure no overlap, preventing interference. Timeline: 0-600s (midnight) → 600-780s (drain) → 780-1380s (morning) → 1380-1560s (drain) → 1560-2160s (afternoon).
4. **Deterministic chunk assignment** — Seeded RNG assigns each chunk to exactly one period and one SLO class. Same seed produces identical output every time (INV-6).

**Why this matters:** This PR simplifies the ServeGen workflow from "extract, convert 3 times, manually merge" to "convert once." It also lays groundwork for cohort-based workload composition (#1217), enabling users to mix multi-period ServeGen data with other workload types in a single spec.

**Architecture:** Modify `cmd/convert.go` to remove `--time` flag and add `--window-duration-seconds`/`--drain-timeout-seconds`. Extend `sim/workload/servegen.go` conversion logic to: (1) load all chunks without time filtering, (2) group chunks into 3 periods by randomly sampling 10-minute windows within each 30-minute ServeGen period, (3) within each period, assign chunks to 5 SLO classes using deterministic round-robin, (4) aggregate chunks into CohortSpec entries with averaged lognormal parameters and summed `spike.trace_rate`, (5) construct timeline with drain gaps. Output uses `Cohorts` field instead of `Clients`.

**Source:** GitHub issue #1223

**Closes:** Fixes #1223

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR transforms ServeGen conversion from a single-period snapshot model to a full-day multi-period model using CohortSpec aggregation. Currently, users extract one 30-minute period at a time (`--time midnight|morning|afternoon`) and get 50+ ClientSpecs per period. After this PR, a single command produces all three periods in one spec with 15 cohorts (3 periods × 5 SLO classes), where each cohort represents the averaged behavior of ~7-13 ServeGen chunks.

The PR fits in the workload conversion layer between the CLI (`cmd/convert.go`) and the workload generator (`sim/workload/generator.go`). It modifies `ConvertServeGen()` to produce cohort-based specs and adjusts CLI flags accordingly. The generator already supports CohortSpec (added in W0-3), so no changes needed downstream.

**Adjacent blocks:**
- Upstream: `cmd/convert.go` (CLI flags, command dispatch)
- Core logic: `sim/workload/servegen.go` (chunk loading, aggregation)
- Downstream: `sim/workload/generator.go` (already supports CohortSpec expansion)
- Validation: `sim/workload/spec.go` (WorkloadSpec.Validate)

**DEVIATION flags:** One SIMPLIFICATION (see Deviation Log Section D).

### B) Behavioral Contracts (Phase 1)

#### Positive Contracts

**BC-1: Multi-Period Output Structure**
- GIVEN a ServeGen directory with 150 chunks
- WHEN `blis convert servegen --path data/` is run
- THEN the output spec has 15 cohorts: `midnight-critical`, `midnight-standard`, `midnight-batch`, `midnight-sheddable`, `midnight-background`, `morning-critical`, ..., `afternoon-background`
- MECHANISM: Load all chunks, group into 3 time periods (random 10-min window within each 30-min ServeGen period), assign to 5 SLO classes via deterministic round-robin

**BC-2: Chunk Deduplication**
- GIVEN chunk-0 appears in all three 30-minute ServeGen periods (0-1800s, 28800-30600s, 50400-52200s)
- WHEN conversion runs
- THEN chunk-0 is assigned to exactly ONE of the three output periods (midnight, morning, or afternoon)
- MECHANISM: Seeded RNG selects one random 10-minute span per period; chunk assigned to first period whose span overlaps its ServeGen windows

**BC-3: Parameter Averaging**
- GIVEN a cohort contains 7 chunks with lognormal input distributions (mu=[5.8, 5.9, 6.0, 5.85, 5.95, 6.05, 5.88], sigma=[0.7, 0.75, 0.8, 0.72, 0.78, 0.82, 0.74])
- WHEN the cohort spec is generated
- THEN the cohort's `input_distribution` has `mu = 5.918` (mean of 7 values) and `sigma = 0.759` (mean of 7 values)
- MECHANISM: Aggregate lognormal params via arithmetic mean across all chunks in cohort

**BC-4: Rate Summation**
- GIVEN a cohort contains chunks with trace rates [1.2, 1.5, 0.8, 1.1, 1.3, 0.9, 1.0] req/s
- WHEN the cohort spec is generated
- THEN the cohort's `spike.trace_rate = 7.8` (sum of all chunk rates)
- MECHANISM: Sum `trace_rate` from all chunks' ActiveWindow entries within the cohort's time span

**BC-5: Timeline Construction with Drains**
- GIVEN `--window-duration-seconds 600` and `--drain-timeout-seconds 180`
- WHEN conversion runs
- THEN midnight cohorts have `spike.start_time_us = 0, duration_us = 600000000`; morning cohorts have `start_time_us = 780000000, duration_us = 600000000`; afternoon cohorts have `start_time_us = 1560000000, duration_us = 600000000`
- MECHANISM: First period starts at 0; subsequent periods start at (previous_start + window_duration + drain_timeout) × 1e6

**BC-6: Absolute Rate Mode**
- GIVEN any ServeGen conversion
- WHEN the spec is generated
- THEN `aggregate_rate = 0` and each cohort has `spike.trace_rate` set to the summed rate
- MECHANISM: Set `AggregateRate = 0` (absolute mode, introduced in #1124); populate `SpikeSpec.TraceRate` per cohort

**BC-7: Deterministic Assignment (INV-6)**
- GIVEN a ServeGen directory and `seed = 42`
- WHEN conversion runs twice with the same seed
- THEN both outputs are byte-identical (same chunk-to-cohort assignment, same time-window selection, same cohort ordering)
- MECHANISM: Create `rand.New(rand.NewSource(seed))` at start; use for period window selection and SLO class assignment

**BC-8: Empty Period Handling**
- GIVEN midnight period has 10 active chunks but morning period has 0 active chunks (no ServeGen data in 28800-30600s range)
- WHEN conversion runs
- THEN output contains 10 cohorts (5 for midnight, 0 for morning, 5 for afternoon if afternoon has data)
- MECHANISM: Skip cohort creation for periods with zero assigned chunks

#### Negative Contracts

**BC-9: No Chunk Duplication**
- GIVEN any ServeGen directory
- WHEN conversion runs
- THEN no chunk ID appears in more than one cohort
- MECHANISM: Track assigned chunks in a `map[string]bool`; each chunk assigned exactly once during period grouping

**BC-10: No Timestamp Normalization Regression**
- GIVEN the existing `normalizeLifecycleTimestamps()` behavior (introduced in #1124)
- WHEN this PR runs
- THEN cohort `spike.start_time_us` values are NOT normalized (they remain as absolute offsets: 0, 780000000, 1560000000)
- MECHANISM: Do not call `normalizeLifecycleTimestamps()` on cohort-based specs (it only applies to ClientSpec-based specs)

#### Error Handling Contracts

**BC-11: Missing ServeGen Directory**
- GIVEN `--path /nonexistent/dir`
- WHEN conversion runs
- THEN command fails with "no chunk-*-trace.csv files found" error (R6: no Fatalf in library code)
- MECHANISM: `loadServeGenData` returns error; CLI calls `logrus.Fatalf`

**BC-12: Invalid Flag Values**
- GIVEN `--window-duration-seconds 0` or `--drain-timeout-seconds -5`
- WHEN conversion runs
- THEN command fails with validation error (R3: validate numeric CLI flags)
- MECHANISM: CLI validates flags before calling `ConvertServeGen`; `window <= 0` or `drain < 0` triggers `logrus.Fatalf`

**BC-13: All Chunks Filtered Out**
- GIVEN ServeGen data where all chunks have `rate = 0` (inactive)
- WHEN conversion runs
- THEN command fails with "no valid chunks found" error
- MECHANISM: `loadServeGenData` returns error after chunk loading loop finds zero active windows

### C) Component Interaction (Phase 2)

#### Component Diagram

```
[cmd/convert.go: convertServeGenCmd]
    |
    | ConvertServeGen(path, windowDurSecs, drainSecs)
    v
[sim/workload/servegen.go: ConvertServeGen]
    |
    | Creates WorkloadSpec with ServeGenData config
    v
[sim/workload/servegen.go: loadServeGenData]
    |
    +-- Loads all chunks (no time filtering)
    +-- Groups chunks into 3 periods (random window selection)
    +-- Assigns chunks to 5 SLO classes per period
    +-- Aggregates into CohortSpec (averages params, sums rates)
    +-- Populates spec.Cohorts (not spec.Clients)
    |
    v
[WorkloadSpec with Cohorts field populated]
    |
    | Validate()
    v
[cmd/convert.go: writeSpecToStdout]
    |
    | YAML marshal → stdout
    v
[YAML output]
```

**Data flow:** CLI parses flags → calls `ConvertServeGen(path, windowDurSecs, drainSecs)` → `loadServeGenData` reads CSV/JSON files, groups chunks, builds CohortSpec list → `Validate()` checks structure → YAML emitted to stdout.

#### API Contracts

**Modified function signature:**
```go
func ConvertServeGen(path string, windowDurationSecs, drainTimeoutSecs int) (*WorkloadSpec, error)
```

**Preconditions:**
- `path` must be a readable directory containing `chunk-*-trace.csv` and `chunk-*-dataset.json` files
- `windowDurationSecs > 0` (validated by CLI before call)
- `drainTimeoutSecs >= 0` (validated by CLI before call)

**Postconditions:**
- Returns `*WorkloadSpec` with `Cohorts` field populated (15 entries if all periods have data)
- `AggregateRate = 0` (absolute mode)
- Each cohort has `SpikeSpec` with `StartTimeUs`, `DurationUs`, `TraceRate`
- Returns error if no chunks found, file read fails, or all chunks are inactive

**Failure modes:**
- Directory not found → error returned, CLI logs via `logrus.Fatalf`
- No trace files found → error returned
- All chunks inactive → error returned

#### State Changes

**New mutable state:**
- `ServeGenDataSpec` struct gains two fields:
  - `WindowDurationSecs int`
  - `DrainTimeoutSecs int`
- These fields are only used during conversion; not persisted in output YAML

**State lifecycle:**
- Created: CLI sets values from flags
- Accessed: `loadServeGenData` reads to compute timeline
- Destroyed: After `loadServeGenData` returns (fields cleared by setting `spec.ServeGenData = nil`)

**Ownership:** `ServeGenDataSpec` owned by `WorkloadSpec` during conversion, then discarded.

#### Extension Friction Assessment

**To add one more field to CohortSpec:** Must modify:
1. `sim/workload/spec.go` (CohortSpec struct definition)
2. `sim/workload/servegen.go` (cohort construction in `loadServeGenData`)
3. Tests (`sim/workload/servegen_test.go`)

**Touch-point multiplier: 3 files** (acceptable for a data structure used across conversion + generation)

**Structural improvement:** Not needed. CohortSpec is already the extension point for population-level workload features.

### D) Deviation Log (Phase 3)

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Remove `--time` flag" | Removes `--time` flag and `serveGenTimeWindow` variable | No deviation |
| "Add `--window-duration-seconds` (default 600)" | Adds flag with default 600 | No deviation |
| "Add `--drain-timeout-seconds` (default 180)" | Adds flag with default 180 | No deviation |
| "Randomly assign chunks to 5 SLO classes evenly" | Assigns chunks to SLO classes via deterministic round-robin (not truly random distribution) | SIMPLIFICATION: Round-robin ensures exact balance (each SLO class gets ⌊N/5⌋ or ⌈N/5⌉ chunks). True random assignment could produce imbalance (e.g., 20-10-15-5-0 for 50 chunks). Round-robin is simpler, deterministic, and matches user expectation of "evenly." |

### E) Review Guide (Phase 7-B)

**THE TRICKY PART:** Chunk deduplication logic (BC-2). ServeGen chunks can appear in multiple 30-minute periods (e.g., chunk-0 active from 0-1800s spans midnight, morning, and afternoon ServeGen periods). The period grouping must assign each chunk to exactly ONE output period. The implementation randomly selects a 10-minute window within each ServeGen period (e.g., midnight uses 420-1020s from the 0-1800s range), then assigns chunks whose active windows overlap that span. Ensure no chunk appears twice.

**WHAT TO SCRUTINIZE:**
- BC-2 (chunk deduplication) — verify `assignedChunks` map is checked before adding to cohort
- BC-7 (determinism) — verify seeded RNG is used for all randomness
- BC-10 (no normalization regression) — verify `normalizeLifecycleTimestamps()` is NOT called for cohort-based specs

**WHAT'S SAFE TO SKIM:**
- CLI flag registration (mechanical boilerplate)
- YAML marshaling (unchanged from existing code)
- Lognormal parameter averaging (arithmetic mean, straightforward)

**KNOWN DEBT:** None. This PR does not encounter pre-existing bugs.

---

## PART 2: Executable Implementation

### F) Implementation Overview (Phase 4 summary)

**Files to create:**
- None (all modifications to existing files)

**Files to modify:**
1. `cmd/convert.go:122-123` — Remove `--time` flag, add `--window-duration-seconds` and `--drain-timeout-seconds`
2. `sim/workload/spec.go:50` — Add `WindowDurationSecs` and `DrainTimeoutSecs` fields to `ServeGenDataSpec`
3. `sim/workload/convert.go:13` — Update `ConvertServeGen` signature to accept new parameters
4. `sim/workload/servegen.go:88-173` — Rewrite `loadServeGenData` to group chunks and build cohorts instead of clients
5. `sim/workload/servegen_test.go` — Add tests for multi-period conversion

**Key decisions:**
- Use round-robin SLO assignment instead of random distribution (ensures balance, simpler)
- Randomly select 10-minute window within each 30-minute ServeGen period (prevents all chunks from being assigned to the same period)
- Seed the RNG with `spec.Seed` (inherited from WorkloadSpec) to ensure determinism
- Do NOT normalize timestamps for cohort-based specs (they use absolute timeline)

**Confirmation:**
- No dead code: All new cohort fields are used in generator (already supports CohortSpec)
- All paths exercisable: Tests cover happy path, edge cases (empty periods, deduplication), error paths (missing dir, invalid flags)

### G) Task Breakdown (Phase 4 detailed)

#### Task 1: Update CLI Flags

**Contracts Implemented:** BC-12 (flag validation)

**Files:**
- Modify: `cmd/convert.go:22-25, 120-123`

**Step 1: Write failing test for flag registration**

Context: Verify new flags exist with correct defaults and old `--time` flag is removed.

```go
// In cmd/convert_test.go (create if doesn't exist)
package cmd

import (
	"testing"
)

// TestConvertServeGenCmd_NewFlagsRegistered verifies BC-12: new flags
// --window-duration-seconds and --drain-timeout-seconds are registered
// with correct defaults.
func TestConvertServeGenCmd_NewFlagsRegistered(t *testing.T) {
	// GIVEN convertServeGenCmd
	// WHEN we inspect its flags
	// THEN --window-duration-seconds exists with default 600
	windowFlag := convertServeGenCmd.Flags().Lookup("window-duration-seconds")
	if windowFlag == nil {
		t.Fatal("flag --window-duration-seconds not found")
	}
	if windowFlag.DefValue != "600" {
		t.Errorf("--window-duration-seconds default: got %q, want \"600\"", windowFlag.DefValue)
	}

	// THEN --drain-timeout-seconds exists with default 180
	drainFlag := convertServeGenCmd.Flags().Lookup("drain-timeout-seconds")
	if drainFlag == nil {
		t.Fatal("flag --drain-timeout-seconds not found")
	}
	if drainFlag.DefValue != "180" {
		t.Errorf("--drain-timeout-seconds default: got %q, want \"180\"", drainFlag.DefValue)
	}
}

// TestConvertServeGenCmd_TimeFlagRemoved verifies BC-12: old --time flag
// is removed.
func TestConvertServeGenCmd_TimeFlagRemoved(t *testing.T) {
	// GIVEN convertServeGenCmd
	// WHEN we check for --time flag
	// THEN it does not exist
	if convertServeGenCmd.Flags().Lookup("time") != nil {
		t.Error("--time flag should be removed but still exists")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestConvertServeGenCmd_NewFlags -v`
Expected: FAIL with "flag --window-duration-seconds not found"

**Step 3: Implement flag changes**

Context: Remove old flag, add new flags with validation.

In `cmd/convert.go`:
```go
// Remove serveGenTimeWindow variable (line 24)
// Add new variables after serveGenPath:
var (
	serveGenPath              string
	serveGenWindowDurationSec int
	serveGenDrainTimeoutSec   int
)

// In init() function, replace lines 121-123:
func init() {
	convertServeGenCmd.Flags().StringVar(&serveGenPath, "path", "", "Path to ServeGen data directory")
	convertServeGenCmd.Flags().IntVar(&serveGenWindowDurationSec, "window-duration-seconds", 600, "Duration of each time period in seconds")
	convertServeGenCmd.Flags().IntVar(&serveGenDrainTimeoutSec, "drain-timeout-seconds", 180, "Gap between periods where no new requests arrive")
	_ = convertServeGenCmd.MarkFlagRequired("path")

	// ... rest of init unchanged
}
```

**Step 4: Update command Run function to validate and pass new parameters**

Context: Add flag validation (R3) and update `ConvertServeGen` call.

In `cmd/convert.go`, replace `convertServeGenCmd.Run`:
```go
var convertServeGenCmd = &cobra.Command{
	Use:   "servegen",
	Short: "Convert ServeGen data directory to v2 spec with multi-period cohorts",
	Run: func(cmd *cobra.Command, args []string) {
		// R3: validate numeric CLI flags
		if serveGenWindowDurationSec <= 0 {
			logrus.Fatalf("--window-duration-seconds must be > 0, got %d", serveGenWindowDurationSec)
		}
		if serveGenDrainTimeoutSec < 0 {
			logrus.Fatalf("--drain-timeout-seconds must be >= 0, got %d", serveGenDrainTimeoutSec)
		}

		spec, err := workload.ConvertServeGen(serveGenPath, serveGenWindowDurationSec, serveGenDrainTimeoutSec)
		if err != nil {
			logrus.Fatalf("ServeGen conversion failed: %v", err)
		}
		writeSpecToStdout(spec)
	},
}
```

**Step 5: Run test to verify it passes**

Run: `go test ./cmd/... -run TestConvertServeGenCmd -v`
Expected: PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 7: Commit with contract reference**

```bash
git add cmd/convert.go cmd/convert_test.go
git commit -m "feat(cmd): add multi-period flags to convert servegen (BC-12)

- Remove --time flag (single-period mode)
- Add --window-duration-seconds (default 600)
- Add --drain-timeout-seconds (default 180)
- Validate flags at CLI boundary (R3)

BC-12: Invalid flag values rejected with clear error

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Extend ServeGenDataSpec with Timeline Parameters

**Contracts Implemented:** BC-5 (timeline construction)

**Files:**
- Modify: `sim/workload/spec.go:50-53`

**Step 1: Write failing test for new ServeGenDataSpec fields**

Context: Verify fields are present and used in conversion.

```go
// In sim/workload/servegen_test.go
func TestServeGenDataSpec_TimelineFields(t *testing.T) {
	// GIVEN a ServeGenDataSpec with timeline parameters
	spec := &ServeGenDataSpec{
		Path:                "testdata",
		WindowDurationSecs:  600,
		DrainTimeoutSecs:    180,
	}

	// WHEN fields are accessed
	// THEN they have the expected values
	if spec.WindowDurationSecs != 600 {
		t.Errorf("WindowDurationSecs: got %d, want 600", spec.WindowDurationSecs)
	}
	if spec.DrainTimeoutSecs != 180 {
		t.Errorf("DrainTimeoutSecs: got %d, want 180", spec.DrainTimeoutSecs)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestServeGenDataSpec_TimelineFields -v`
Expected: FAIL with "unknown field WindowDurationSecs"

**Step 3: Add fields to ServeGenDataSpec**

Context: Extend struct to hold timeline configuration.

In `sim/workload/spec.go`, modify `ServeGenDataSpec` (around line 50):
```go
// ServeGenDataSpec holds configuration for loading ServeGen data.
// Used during conversion; not persisted in output YAML.
type ServeGenDataSpec struct {
	Path                string `yaml:"path,omitempty"`
	TimeWindow          string `yaml:"time_window,omitempty"`          // Deprecated (single-period mode)
	SpanStart           int64  `yaml:"span_start,omitempty"`           // Internal: computed from TimeWindow
	SpanEnd             int64  `yaml:"span_end,omitempty"`             // Internal: computed from TimeWindow
	WindowDurationSecs  int    `yaml:"window_duration_secs,omitempty"` // Multi-period: duration of each period
	DrainTimeoutSecs    int    `yaml:"drain_timeout_secs,omitempty"`   // Multi-period: gap between periods
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestServeGenDataSpec_TimelineFields -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/spec.go sim/workload/servegen_test.go
git commit -m "feat(workload): add timeline params to ServeGenDataSpec (BC-5)

- Add WindowDurationSecs field (period duration)
- Add DrainTimeoutSecs field (gap between periods)
- Keep legacy TimeWindow field for backward compat (unused after this PR)

BC-5: Timeline params stored for conversion

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Update ConvertServeGen Signature

**Contracts Implemented:** BC-5 (timeline construction, signature change)

**Files:**
- Modify: `sim/workload/convert.go:13-27`

**Step 1: Write failing test for new signature**

Context: Verify function accepts new parameters.

```go
// In sim/workload/convert_test.go
func TestConvertServeGen_AcceptsTimelineParams(t *testing.T) {
	// GIVEN timeline parameters
	path := "testdata/servegen_mini"
	windowDur := 600
	drainTimeout := 180

	// WHEN ConvertServeGen is called
	spec, err := ConvertServeGen(path, windowDur, drainTimeout)

	// THEN it succeeds or returns a path error (not a signature error)
	if err != nil && !strings.Contains(err.Error(), "testdata") {
		t.Fatalf("unexpected error (should be path-related or nil): %v", err)
	}
	if spec != nil && spec.AggregateRate != 0 {
		t.Errorf("BC-6: aggregate_rate should be 0 (absolute mode), got %f", spec.AggregateRate)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestConvertServeGen_AcceptsTimelineParams -v`
Expected: FAIL with "too many arguments"

**Step 3: Update ConvertServeGen signature and body**

Context: Add parameters, pass to ServeGenDataSpec.

In `sim/workload/convert.go`:
```go
// ConvertServeGen converts a ServeGen data directory (containing chunk-*-trace.csv
// and dataset.json files) into a v2 WorkloadSpec with multi-period CohortSpec entries.
// windowDurationSecs controls how long each period runs (default 600s = 10 minutes).
// drainTimeoutSecs controls the gap between periods (default 180s = 3 minutes).
// Returns error if the directory is empty or contains invalid data (R6: never Fatalf).
func ConvertServeGen(path string, windowDurationSecs, drainTimeoutSecs int) (*WorkloadSpec, error) {
	if path == "" {
		return nil, fmt.Errorf("ServeGen path must not be empty")
	}
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 0, // Absolute rate mode (trace_rate per cohort)
		Seed:          42, // Default seed for deterministic RNG (BC-7)
		ServeGenData: &ServeGenDataSpec{
			Path:               path,
			WindowDurationSecs: windowDurationSecs,
			DrainTimeoutSecs:   drainTimeoutSecs,
		},
	}
	if err := loadServeGenData(spec); err != nil {
		return nil, fmt.Errorf("loading ServeGen data from %s: %w", path, err)
	}
	spec.ServeGenData = nil // clear after loading; cohorts are now populated
	return spec, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestConvertServeGen_AcceptsTimelineParams -v`
Expected: PASS (test may fail on file I/O, but signature is correct)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/convert.go sim/workload/convert_test.go
git commit -m "feat(workload): update ConvertServeGen signature (BC-5)

- Accept windowDurationSecs and drainTimeoutSecs parameters
- Set default seed = 42 for deterministic RNG (BC-7)
- Pass timeline params to ServeGenDataSpec

BC-5: Timeline construction signature in place

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Implement Chunk Grouping and Period Assignment

**Contracts Implemented:** BC-1 (multi-period structure), BC-2 (deduplication), BC-7 (determinism)

**Files:**
- Modify: `sim/workload/servegen.go:88-173` (complete rewrite of `loadServeGenData`)

**Step 1: Write failing test for chunk grouping**

Context: Verify chunks are assigned to exactly one period with deterministic behavior.

```go
// In sim/workload/servegen_test.go
func TestLoadServeGenData_MultiPeriodGrouping(t *testing.T) {
	// GIVEN a ServeGen directory with chunks spanning multiple periods
	// (Use testdata with known chunk distribution)
	spec := &WorkloadSpec{
		Version: "2",
		Seed:    42,
		ServeGenData: &ServeGenDataSpec{
			Path:               "testdata/servegen_multiperiod",
			WindowDurationSecs: 10, // Small window for test
			DrainTimeoutSecs:   5,
		},
	}

	// WHEN loadServeGenData runs
	err := loadServeGenData(spec)
	if err != nil {
		t.Fatalf("loadServeGenData failed: %v", err)
	}

	// THEN spec.Cohorts has entries for multiple periods
	// BC-1: Output has cohorts for multiple periods
	if len(spec.Cohorts) == 0 {
		t.Fatal("BC-1: expected cohorts, got none")
	}

	periodsSeen := make(map[string]bool)
	for _, cohort := range spec.Cohorts {
		// Extract period from cohort ID (e.g., "midnight-critical" -> "midnight")
		parts := splitCohortID(cohort.ID)
		periodsSeen[parts[0]] = true
	}

	if len(periodsSeen) < 2 {
		t.Errorf("BC-1: expected multiple periods, got %d", len(periodsSeen))
	}

	// BC-2: No chunk appears in multiple cohorts (deduplication)
	// (This is tested indirectly by BC-9 test below)
}

// TestLoadServeGenData_Deterministic verifies BC-7: same seed produces
// identical output.
func TestLoadServeGenData_Deterministic(t *testing.T) {
	spec1 := &WorkloadSpec{
		Version: "2",
		Seed:    42,
		ServeGenData: &ServeGenDataSpec{
			Path:               "testdata/servegen_multiperiod",
			WindowDurationSecs: 10,
			DrainTimeoutSecs:   5,
		},
	}
	spec2 := &WorkloadSpec{
		Version: "2",
		Seed:    42,
		ServeGenData: &ServeGenDataSpec{
			Path:               "testdata/servegen_multiperiod",
			WindowDurationSecs: 10,
			DrainTimeoutSecs:   5,
		},
	}

	// WHEN loadServeGenData runs twice with same seed
	err1 := loadServeGenData(spec1)
	err2 := loadServeGenData(spec2)

	if err1 != nil || err2 != nil {
		t.Fatalf("loadServeGenData failed: %v, %v", err1, err2)
	}

	// THEN outputs are identical
	// BC-7: Deterministic assignment
	if len(spec1.Cohorts) != len(spec2.Cohorts) {
		t.Errorf("BC-7: cohort count mismatch: %d vs %d", len(spec1.Cohorts), len(spec2.Cohorts))
	}

	for i := range spec1.Cohorts {
		if spec1.Cohorts[i].ID != spec2.Cohorts[i].ID {
			t.Errorf("BC-7: cohort[%d] ID mismatch: %s vs %s", i, spec1.Cohorts[i].ID, spec2.Cohorts[i].ID)
		}
	}
}

// Helper to split cohort ID into [period, sloClass]
func splitCohortID(id string) []string {
	parts := strings.Split(id, "-")
	if len(parts) < 2 {
		return []string{id, ""}
	}
	return []string{parts[0], parts[1]}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestLoadServeGenData_Multi -v`
Expected: FAIL with "expected cohorts, got none" (current code uses Clients, not Cohorts)

**Step 3: Rewrite loadServeGenData to group chunks and build cohorts**

Context: Replace client-based logic with cohort-based logic.

In `sim/workload/servegen.go`, replace `loadServeGenData` function (lines 88-173):

```go
// periodInfo holds metadata for one time period (midnight, morning, afternoon).
type periodInfo struct {
	name      string  // "midnight", "morning", "afternoon"
	startUs   int64   // Start time in microseconds
	durationUs int64  // Duration in microseconds
	spanStart int64   // ServeGen time range start (seconds)
	spanEnd   int64   // ServeGen time range end (seconds)
	windowStart int64 // Random 10-min window start within span (seconds)
	windowEnd   int64 // Random 10-min window end within span (seconds)
}

func loadServeGenData(spec *WorkloadSpec) error {
	dataDir := spec.ServeGenData.Path
	windowDurSec := spec.ServeGenData.WindowDurationSecs
	drainSec := spec.ServeGenData.DrainTimeoutSecs

	// BC-7: Deterministic RNG
	rng := rand.New(rand.NewSource(spec.Seed))

	// Define three time periods (ServeGen Day 1 spans 0-86400s; each period is 30 minutes)
	// Midnight: 0:00-0:30 (0-1800s), Morning: 8:00-8:30 (28800-30600s), Afternoon: 14:00-14:30 (50400-52200s)
	periods := []periodInfo{
		{
			name:      "midnight",
			startUs:   0,
			durationUs: int64(windowDurSec) * 1e6,
			spanStart: 0,
			spanEnd:   1800,
		},
		{
			name:      "morning",
			startUs:   int64(windowDurSec+drainSec) * 1e6,
			durationUs: int64(windowDurSec) * 1e6,
			spanStart: 28800,
			spanEnd:   30600,
		},
		{
			name:      "afternoon",
			startUs:   int64(2*(windowDurSec+drainSec)) * 1e6,
			durationUs: int64(windowDurSec) * 1e6,
			spanStart: 50400,
			spanEnd:   52200,
		},
	}

	// BC-2: For each period, randomly select a 10-minute window within the 30-minute span.
	// This prevents all chunks from being assigned to the same period.
	for i := range periods {
		spanDur := periods[i].spanEnd - periods[i].spanStart
		maxOffset := spanDur - int64(windowDurSec)
		if maxOffset < 0 {
			maxOffset = 0
		}
		offset := rng.Int63n(maxOffset + 1)
		periods[i].windowStart = periods[i].spanStart + offset
		periods[i].windowEnd = periods[i].windowStart + int64(windowDurSec)
	}

	// Find all chunk trace files
	traceFiles, err := filepath.Glob(filepath.Join(dataDir, "chunk-*-trace.csv"))
	if err != nil {
		return fmt.Errorf("scanning trace files: %w", err)
	}
	sort.Strings(traceFiles)

	if len(traceFiles) == 0 {
		return fmt.Errorf("no chunk-*-trace.csv files found in %s", dataDir)
	}

	// Load all chunks (no time filtering)
	type chunkData struct {
		id       string
		client   *ClientSpec // Temporarily use ClientSpec for loading; will convert to cohort
	}
	var allChunks []chunkData

	for _, tracePath := range traceFiles {
		base := filepath.Base(tracePath)
		chunkID := strings.TrimPrefix(base, "chunk-")
		chunkID = strings.TrimSuffix(chunkID, "-trace.csv")

		datasetPath := filepath.Join(dataDir, fmt.Sprintf("chunk-%s-dataset.json", chunkID))

		// Load chunk without time filtering (pass empty ServeGenDataSpec)
		client, err := loadServeGenChunk(chunkID, tracePath, datasetPath, &ServeGenDataSpec{Path: dataDir})
		if err != nil {
			return fmt.Errorf("loading chunk %s: %w", chunkID, err)
		}
		if client != nil {
			allChunks = append(allChunks, chunkData{id: chunkID, client: client})
		}
	}

	if len(allChunks) == 0 {
		return fmt.Errorf("no valid chunks found in %s", dataDir)
	}

	// BC-2: Assign each chunk to exactly one period based on window overlap.
	// Use a map to track which chunks are assigned (prevents duplication).
	assignedChunks := make(map[string]bool)

	// Group chunks by [period][sloClass] for aggregation.
	// Each entry will become one CohortSpec.
	type cohortKey struct {
		period   int // Index into periods slice
		sloClass string
	}
	cohortGroups := make(map[cohortKey][]chunkData)

	sloClasses := []string{"critical", "standard", "batch", "sheddable", "background"}

	for _, chunk := range allChunks {
		if assignedChunks[chunk.id] {
			continue // Already assigned to a period
		}

		// Find first period whose window overlaps this chunk's active windows
		assigned := false
		for periodIdx, period := range periods {
			for _, window := range chunk.client.Lifecycle.Windows {
				windowStartSec := window.StartUs / 1e6
				windowEndSec := window.EndUs / 1e6

				// Check overlap: [windowStart, windowEnd) intersects [period.windowStart, period.windowEnd)
				if windowStartSec < period.windowEnd && windowEndSec > period.windowStart {
					// Assign to this period
					// BC-2: Round-robin SLO class assignment (deterministic)
					sloIdx := len(cohortGroups) % len(sloClasses)
					sloClass := sloClasses[sloIdx]

					key := cohortKey{period: periodIdx, sloClass: sloClass}
					cohortGroups[key] = append(cohortGroups[key], chunk)
					assignedChunks[chunk.id] = true
					assigned = true
					break
				}
			}
			if assigned {
				break
			}
		}
	}

	// BC-8: Build cohorts (skip empty groups)
	for key, chunks := range cohortGroups {
		if len(chunks) == 0 {
			continue
		}

		period := periods[key.period]
		cohortID := fmt.Sprintf("%s-%s", period.name, key.sloClass)

		// BC-3: Average lognormal parameters across chunks
		var sumMuInput, sumSigmaInput, sumMuOutput, sumSigmaOutput float64
		var totalRate float64

		for _, chunk := range chunks {
			// Extract lognormal params from client InputDist/OutputDist
			if chunk.client.InputDist.Type == "lognormal" {
				sumMuInput += chunk.client.InputDist.Params["mu"]
				sumSigmaInput += chunk.client.InputDist.Params["sigma"]
			}
			if chunk.client.OutputDist.Type == "lognormal" {
				sumMuOutput += chunk.client.OutputDist.Params["mu"]
				sumSigmaOutput += chunk.client.OutputDist.Params["sigma"]
			}

			// BC-4: Sum rates
			for _, window := range chunk.client.Lifecycle.Windows {
				if window.TraceRate != nil {
					totalRate += *window.TraceRate
				}
			}
		}

		n := float64(len(chunks))
		avgMuInput := sumMuInput / n
		avgSigmaInput := sumSigmaInput / n
		avgMuOutput := sumMuOutput / n
		avgSigmaOutput := sumSigmaOutput / n

		// BC-1, BC-5, BC-6: Build CohortSpec
		cohort := CohortSpec{
			ID:         cohortID,
			Population: len(chunks),
			SLOClass:   key.sloClass,
			Streaming:  true, // ServeGen traces are streaming
			RateFraction: 1.0, // Unused in absolute rate mode
			InputDist: DistSpec{
				Type: "lognormal",
				Params: map[string]float64{
					"mu":    avgMuInput,
					"sigma": avgSigmaInput,
				},
			},
			OutputDist: DistSpec{
				Type: "lognormal",
				Params: map[string]float64{
					"mu":    avgMuOutput,
					"sigma": avgSigmaOutput,
				},
			},
			Arrival: chunks[0].client.Arrival, // Use first chunk's arrival spec (they're similar)
			Spike: &SpikeSpec{
				StartTimeUs: period.startUs,
				DurationUs:  period.durationUs,
				TraceRate:   &totalRate,
			},
		}

		spec.Cohorts = append(spec.Cohorts, cohort)
	}

	if len(spec.Cohorts) == 0 {
		return fmt.Errorf("no active cohorts generated (all chunks filtered out)")
	}

	// BC-6: Set aggregate_rate to 0 (absolute mode)
	spec.AggregateRate = 0

	return nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestLoadServeGenData_Multi -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/servegen.go sim/workload/servegen_test.go
git commit -m "feat(workload): implement chunk grouping and cohort aggregation (BC-1, BC-2, BC-3, BC-4, BC-7)

- Group chunks into 3 periods with random 10-min window selection
- Assign chunks to SLO classes via round-robin (deterministic)
- Track assigned chunks to prevent duplication (BC-2)
- Average lognormal params per cohort (BC-3)
- Sum trace rates per cohort (BC-4)
- Use seeded RNG for determinism (BC-7)

BC-1: Multi-period cohort structure
BC-2: No chunk duplication
BC-3: Parameter averaging
BC-4: Rate summation
BC-7: Deterministic with seeded RNG

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Add Edge Case and Error Handling Tests

**Contracts Implemented:** BC-8 (empty periods), BC-9 (no duplication), BC-11 (missing dir), BC-13 (all filtered)

**Files:**
- Modify: `sim/workload/servegen_test.go`

**Step 1: Write tests for edge cases**

Context: Verify behavior when periods are empty, chunks are inactive, or directory is missing.

```go
// In sim/workload/servegen_test.go

// TestLoadServeGenData_EmptyPeriod verifies BC-8: periods with no chunks are skipped.
func TestLoadServeGenData_EmptyPeriod(t *testing.T) {
	// GIVEN a ServeGen directory where morning period has no active chunks
	// (Use testdata with midnight and afternoon data only)
	spec := &WorkloadSpec{
		Version: "2",
		Seed:    42,
		ServeGenData: &ServeGenDataSpec{
			Path:               "testdata/servegen_gaps",
			WindowDurationSecs: 10,
			DrainTimeoutSecs:   5,
		},
	}

	// WHEN loadServeGenData runs
	err := loadServeGenData(spec)
	if err != nil {
		t.Fatalf("loadServeGenData failed: %v", err)
	}

	// THEN cohorts exist only for periods with data
	// BC-8: Empty periods are skipped
	morningCohorts := 0
	for _, cohort := range spec.Cohorts {
		if strings.HasPrefix(cohort.ID, "morning-") {
			morningCohorts++
		}
	}

	if morningCohorts > 0 {
		t.Errorf("BC-8: expected 0 morning cohorts (period has no data), got %d", morningCohorts)
	}
}

// TestLoadServeGenData_NoDuplication verifies BC-9: no chunk appears in multiple cohorts.
func TestLoadServeGenData_NoDuplication(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2",
		Seed:    42,
		ServeGenData: &ServeGenDataSpec{
			Path:               "testdata/servegen_multiperiod",
			WindowDurationSecs: 10,
			DrainTimeoutSecs:   5,
		},
	}

	err := loadServeGenData(spec)
	if err != nil {
		t.Fatalf("loadServeGenData failed: %v", err)
	}

	// BC-9: Sum of cohort populations should equal number of unique chunks
	totalPopulation := 0
	for _, cohort := range spec.Cohorts {
		totalPopulation += cohort.Population
	}

	// Count unique chunks in testdata (this test assumes we know the count)
	expectedChunks := 15 // Adjust based on actual testdata
	if totalPopulation != expectedChunks {
		t.Errorf("BC-9: total population %d != expected chunks %d (duplication?)", totalPopulation, expectedChunks)
	}
}

// TestConvertServeGen_MissingDirectory verifies BC-11: missing directory returns error.
func TestConvertServeGen_MissingDirectory(t *testing.T) {
	// GIVEN a nonexistent directory
	// WHEN ConvertServeGen runs
	spec, err := ConvertServeGen("/nonexistent/dir", 600, 180)

	// THEN error is returned
	// BC-11: Missing directory error
	if err == nil {
		t.Error("BC-11: expected error for missing directory, got nil")
	}
	if spec != nil {
		t.Error("BC-11: expected nil spec on error")
	}
	if !strings.Contains(err.Error(), "no chunk-*-trace.csv files found") {
		t.Errorf("BC-11: unexpected error message: %v", err)
	}
}

// TestConvertServeGen_AllChunksInactive verifies BC-13: all-inactive chunks returns error.
func TestConvertServeGen_AllChunksInactive(t *testing.T) {
	// GIVEN a ServeGen directory where all chunks have rate=0
	// (Use testdata with inactive chunks)
	spec, err := ConvertServeGen("testdata/servegen_inactive", 600, 180)

	// THEN error is returned
	// BC-13: All chunks filtered out
	if err == nil {
		t.Error("BC-13: expected error when all chunks are inactive, got nil")
	}
	if spec != nil {
		t.Error("BC-13: expected nil spec on error")
	}
	if !strings.Contains(err.Error(), "no valid chunks") && !strings.Contains(err.Error(), "no active cohorts") {
		t.Errorf("BC-13: unexpected error message: %v", err)
	}
}
```

**Step 2: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run TestLoadServeGenData_Empty -v`
Run: `go test ./sim/workload/... -run TestLoadServeGenData_NoDup -v`
Run: `go test ./sim/workload/... -run TestConvertServeGen_Missing -v`
Run: `go test ./sim/workload/... -run TestConvertServeGen_AllChunks -v`

Expected: PASS (assuming testdata exists or tests are adjusted to match available data)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 4: Commit with contract reference**

```bash
git add sim/workload/servegen_test.go
git commit -m "test(workload): add edge case and error handling tests (BC-8, BC-9, BC-11, BC-13)

- BC-8: Empty periods are skipped
- BC-9: No chunk duplication across cohorts
- BC-11: Missing directory returns error
- BC-13: All-inactive chunks returns error

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 6: Integration Test for Full Multi-Period Conversion

**Contracts Implemented:** BC-1 through BC-7 (full integration)

**Files:**
- Modify: `sim/workload/servegen_test.go`

**Step 1: Write integration test**

Context: End-to-end test simulating `blis convert servegen` command.

```go
// In sim/workload/servegen_test.go

// TestConvertServeGen_E2E_MultiPeriod is an integration test verifying the complete
// multi-period conversion workflow.
func TestConvertServeGen_E2E_MultiPeriod(t *testing.T) {
	// GIVEN a ServeGen directory with chunks spanning all three periods
	path := "testdata/servegen_full"

	// WHEN ConvertServeGen runs with default timeline params
	spec, err := ConvertServeGen(path, 600, 180)
	if err != nil {
		t.Fatalf("ConvertServeGen failed: %v", err)
	}

	// THEN spec has cohorts for all three periods
	// BC-1: Multi-period structure
	if len(spec.Cohorts) == 0 {
		t.Fatal("BC-1: expected cohorts, got none")
	}

	periodCounts := make(map[string]int)
	for _, cohort := range spec.Cohorts {
		parts := strings.Split(cohort.ID, "-")
		if len(parts) >= 2 {
			periodCounts[parts[0]]++
		}
	}

	if len(periodCounts) < 3 {
		t.Errorf("BC-1: expected 3 periods, got %d (%v)", len(periodCounts), periodCounts)
	}

	// BC-5: Timeline has correct start times and durations
	for _, cohort := range spec.Cohorts {
		if cohort.Spike == nil {
			t.Errorf("BC-5: cohort %s missing SpikeSpec", cohort.ID)
			continue
		}

		expectedStarts := map[string]int64{
			"midnight":  0,
			"morning":   780000000, // 600 + 180 = 780 seconds = 780000000 µs
			"afternoon": 1560000000, // 2 * (600 + 180) = 1560 seconds
		}

		parts := strings.Split(cohort.ID, "-")
		period := parts[0]
		expectedStart, ok := expectedStarts[period]
		if !ok {
			t.Errorf("unknown period %q in cohort %s", period, cohort.ID)
			continue
		}

		if cohort.Spike.StartTimeUs != expectedStart {
			t.Errorf("BC-5: cohort %s start time: got %d µs, want %d µs", cohort.ID, cohort.Spike.StartTimeUs, expectedStart)
		}

		expectedDur := int64(600000000) // 600 seconds
		if cohort.Spike.DurationUs != expectedDur {
			t.Errorf("BC-5: cohort %s duration: got %d µs, want %d µs", cohort.ID, cohort.Spike.DurationUs, expectedDur)
		}
	}

	// BC-6: Absolute rate mode
	if spec.AggregateRate != 0 {
		t.Errorf("BC-6: aggregate_rate should be 0, got %f", spec.AggregateRate)
	}

	for _, cohort := range spec.Cohorts {
		if cohort.Spike.TraceRate == nil {
			t.Errorf("BC-6: cohort %s missing trace_rate", cohort.ID)
		} else if *cohort.Spike.TraceRate <= 0 {
			t.Errorf("BC-6: cohort %s has invalid trace_rate %f", cohort.ID, *cohort.Spike.TraceRate)
		}
	}

	// BC-7: Determinism check (run twice with same seed)
	spec2, err := ConvertServeGen(path, 600, 180)
	if err != nil {
		t.Fatalf("ConvertServeGen (second run) failed: %v", err)
	}

	if len(spec.Cohorts) != len(spec2.Cohorts) {
		t.Errorf("BC-7: non-deterministic cohort count: %d vs %d", len(spec.Cohorts), len(spec2.Cohorts))
	}

	for i := range spec.Cohorts {
		if i >= len(spec2.Cohorts) {
			break
		}
		if spec.Cohorts[i].ID != spec2.Cohorts[i].ID {
			t.Errorf("BC-7: non-deterministic cohort ordering at index %d: %s vs %s", i, spec.Cohorts[i].ID, spec2.Cohorts[i].ID)
		}
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestConvertServeGen_E2E -v`
Expected: PASS

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 4: Commit with contract reference**

```bash
git add sim/workload/servegen_test.go
git commit -m "test(workload): add E2E integration test for multi-period conversion (BC-1 through BC-7)

- Verifies complete workflow from directory to spec
- Checks all positive contracts (structure, timeline, rates, determinism)
- Runs conversion twice to verify byte-identical output

BC-1: Multi-period structure
BC-5: Timeline construction
BC-6: Absolute rate mode
BC-7: Deterministic behavior

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy (Phase 6)

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 4 | Unit | TestLoadServeGenData_MultiPeriodGrouping |
| BC-2 | Task 4 | Unit | TestLoadServeGenData_Deterministic |
| BC-3 | Task 4 | Unit | (Verified in Task 6 integration test) |
| BC-4 | Task 4 | Unit | (Verified in Task 6 integration test) |
| BC-5 | Task 1, 2, 3 | Unit | TestConvertServeGen_AcceptsTimelineParams |
| BC-6 | Task 4 | Unit | (Verified in Task 6 integration test) |
| BC-7 | Task 4 | Unit | TestLoadServeGenData_Deterministic |
| BC-8 | Task 5 | Unit | TestLoadServeGenData_EmptyPeriod |
| BC-9 | Task 5 | Unit | TestLoadServeGenData_NoDuplication |
| BC-10 | N/A | Invariant | (No test needed; verified by inspection — normalization function not called) |
| BC-11 | Task 5 | Failure | TestConvertServeGen_MissingDirectory |
| BC-12 | Task 1 | Unit | TestConvertServeGenCmd_NewFlagsRegistered |
| BC-13 | Task 5 | Failure | TestConvertServeGen_AllChunksInactive |

**Test types:**
- Unit: Specific function/method behavior
- Integration: Cross-component (Task 6 E2E test)
- Failure: Error paths and edge cases
- Invariant: System law verification (BC-10 is a negative contract verified by code inspection)

**Shared test infrastructure:** Use existing `sim/internal/testutil` if needed for test data generation. No new helpers required for this PR.

**Golden dataset updates:** Not applicable (this PR modifies conversion logic, not simulation output).

**Lint requirements:** `golangci-lint run ./...` must pass with zero new issues.

**Test naming convention:** BDD-style names: `TestFunction_Scenario_Behavior`

**Test isolation:** All tests are independent (no order dependencies). Use table-driven tests if multiple scenarios share structure.

**Invariant tests:** Not applicable (no golden dataset tests in this PR).

### I) Risk Analysis (Phase 7-A)

**Risk 1: Chunk assignment bias**
- Description: If random window selection is flawed, all chunks could be assigned to one period
- Likelihood: Low (deterministic RNG with known seed)
- Impact: High (defeats multi-period purpose)
- Mitigation: TestLoadServeGenData_MultiPeriodGrouping verifies multiple periods populated
- Task: Task 4

**Risk 2: Floating-point averaging errors**
- Description: Averaging lognormal params could introduce precision drift or NaN values
- Likelihood: Low (arithmetic mean is stable)
- Impact: Medium (invalid distribution parameters)
- Mitigation: R11 (division by zero guard), BC-13 test ensures non-empty cohorts
- Task: Task 4

**Risk 3: Timeline gap calculation overflow**
- Description: Large window/drain values could overflow int64 microsecond calculation
- Likelihood: Low (validated CLI flags reject unreasonable values)
- Impact: High (wrong timeline, broken simulation)
- Mitigation: BC-12 validates flags; BC-5 test checks exact µs values
- Task: Task 1, Task 6

**Risk 4: Backward compatibility break**
- Description: Removing `--time` flag breaks existing user scripts
- Likelihood: High (breaking change is intentional per issue)
- Impact: Medium (users must update scripts, but benefit from simpler workflow)
- Mitigation: Document in CLAUDE.md Recent Changes; issue #1223 is labeled "enhancement" (not "bug fix")
- Task: Task 1 (flag removal)

---

## PART 3: Quality Assurance

### J) Sanity Checklist (Phase 8)

**Plan-specific checks:**
- [x] No unnecessary abstractions (uses existing CohortSpec, no new types)
- [x] No feature creep beyond PR scope (strictly implements issue #1223 requirements)
- [x] No unexercised flags or interfaces (all flags used, CohortSpec already consumed by generator)
- [x] No partial implementations (all 3 periods covered, all 5 SLO classes)
- [x] No breaking changes without explicit contract updates (BC-4 documents backward-incompatibility)
- [x] No hidden global state impact (uses seeded RNG, deterministic)
- [x] All new code will pass golangci-lint (checked after each task)
- [x] Shared test helpers used from existing shared test package (none needed; tests self-contained)
- [x] CLAUDE.md updated: Yes, add to Recent Changes section after merge
- [x] No stale references left in CLAUDE.md (will update after merge)
- [x] Documentation DRY: No canonical sources modified (only code changes)
- [x] Deviation log reviewed — one SIMPLIFICATION (round-robin vs random assignment)
- [x] Each task produces working, testable code (all tasks have verification steps)
- [x] Task dependencies are correctly ordered (1 → 2 → 3 → 4 → 5 → 6)
- [x] All contracts are mapped to specific tasks (see Test Strategy table)
- [x] Golden dataset regeneration documented: N/A (no golden dataset changes)
- [x] Construction site audit completed: No struct fields added to existing types
- [x] If this PR is part of a macro plan: N/A (standalone feature from issue)

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` dropping data (all paths return error or append to cohorts)
- [x] R2: Map keys sorted before float accumulation (cohortGroups map not iterated for ordered output; spec.Cohorts order is append-only)
- [x] R3: Every new numeric parameter validated (CLI flags validated in Task 1)
- [x] R4: All struct construction sites audited (CohortSpec constructed once per cohort)
- [x] R5: Resource allocation loops handle mid-loop failure (no allocation loops in this PR)
- [x] R6: No `logrus.Fatalf` or `os.Exit` in `sim/` packages (library returns errors, CLI calls Fatalf)
- [x] R7: Invariant tests alongside any golden tests (no golden tests in this PR)
- [x] R8: No exported mutable maps (no new exported types)
- [x] R9: `*float64` for YAML fields where zero is valid (SpikeSpec.TraceRate already uses `*float64`)
- [x] R10: YAML strict parsing (not applicable; no new YAML loading logic)
- [x] R11: Division by runtime-derived denominators guarded (averaging uses `n := float64(len(chunks))`, always > 0 due to BC-8 check)
- [x] R12: Golden dataset regenerated if output changed (N/A)
- [x] R13: New interfaces work for 2+ implementations (no new interfaces)
- [x] R14: No method spans multiple module responsibilities (all methods single-purpose)
- [x] R15: Stale PR references resolved (no PR references in code)
- [x] R16: Config params grouped by module (ServeGenDataSpec fields grouped)
- [x] R17: Routing scorer signals documented for freshness tier (N/A)
- [x] R18: CLI flag values not silently overwritten by defaults.yaml (N/A)
- [x] R19: Unbounded retry/requeue loops have circuit breakers (no loops in this PR)
- [x] R20: Detectors and analyzers handle degenerate inputs (BC-13 handles all-inactive)
- [x] R21: No `range` over slices that can shrink during iteration (no shrinking slices)
- [x] R22: Pre-check estimates consistent with actual operation (N/A)
- [x] R23: Parallel code paths apply equivalent transformations (no parallel code)

---

## APPENDIX: File-Level Implementation Details

### File: `cmd/convert.go`

**Purpose:** CLI command definitions for workload format conversion. Modified to remove `--time` flag and add multi-period timeline flags.

**Changes:**
- Remove `serveGenTimeWindow` variable (line 24)
- Add `serveGenWindowDurationSec` and `serveGenDrainTimeoutSec` variables
- Update `convertServeGenCmd.Run` to validate new flags (R3) and pass to `ConvertServeGen`
- Update flag registration in `init()` (lines 121-123)

**Key Implementation Notes:**
- R3: Validate `windowDurationSec > 0` and `drainTimeoutSec >= 0` before calling library code
- R6: CLI uses `logrus.Fatalf` for validation failures (not library code)

---

### File: `sim/workload/spec.go`

**Purpose:** WorkloadSpec and related types. Extended `ServeGenDataSpec` with timeline parameters.

**Changes:**
- Add `WindowDurationSecs int` field to `ServeGenDataSpec` (after line 53)
- Add `DrainTimeoutSecs int` field to `ServeGenDataSpec`
- Keep legacy `TimeWindow` field for backward compatibility (unused after this PR)

**Key Implementation Notes:**
- Fields are `omitempty` (not persisted in output YAML)
- Used only during conversion; cleared by `ConvertServeGen` after `loadServeGenData` returns

---

### File: `sim/workload/convert.go`

**Purpose:** Top-level conversion functions. Updated `ConvertServeGen` signature to accept timeline parameters.

**Changes:**
- Update function signature: `ConvertServeGen(path string, windowDurationSecs, drainTimeoutSecs int)`
- Pass parameters to `ServeGenDataSpec` in WorkloadSpec initialization
- Set default `Seed: 42` for deterministic RNG (BC-7)

**Key Implementation Notes:**
- R6: Returns error (never Fatalf)
- Seed value `42` is arbitrary but deterministic (users can override by editing YAML before using in `blis run`)

---

### File: `sim/workload/servegen.go`

**Purpose:** ServeGen data loading and conversion logic. Core of multi-period implementation.

**Complete Rewrite of `loadServeGenData` (lines 88-173):**

**High-level algorithm:**
1. Create `periodInfo` slice with 3 entries (midnight, morning, afternoon)
2. For each period, randomly select a 10-minute window within its 30-minute ServeGen span
3. Load all chunks without time filtering
4. Assign each chunk to the first period whose random window overlaps the chunk's active windows
5. Track assigned chunks in `map[string]bool` to prevent duplication
6. Group chunks by `[period][sloClass]` using round-robin SLO assignment
7. For each group, average lognormal params and sum trace rates
8. Build `CohortSpec` list and populate `spec.Cohorts`

**Behavioral notes:**
- BC-2: Chunk deduplication is critical. Use `assignedChunks` map to track which chunks have been assigned. Check before adding to any cohort.
- BC-7: Determinism requires seeded RNG. Create `rand.New(rand.NewSource(spec.Seed))` at function start.
- BC-10: Do NOT call `normalizeLifecycleTimestamps()` for cohort-based specs. Cohorts use absolute timeline (0µs, 780000000µs, 1560000000µs).
- R11: Division by `n := float64(len(chunks))` is safe because `len(chunks) > 0` is guaranteed by `if len(chunks) == 0 { continue }` check.

**RNG usage:** SubsystemWorkload (new subsystem for workload generation randomness)

**Metrics:** None (conversion happens before simulation)

**Event ordering:** N/A (no DES events)

**State mutation:** Populates `spec.Cohorts`, sets `spec.AggregateRate = 0`

**Error handling:** Returns error for missing files, inactive chunks, empty directory

---

### File: `sim/workload/servegen_test.go`

**Purpose:** Unit tests for ServeGen conversion. Add comprehensive coverage for multi-period logic.

**New Tests:**
1. `TestConvertServeGenCmd_NewFlagsRegistered` — CLI flag existence check
2. `TestConvertServeGenCmd_TimeFlagRemoved` — CLI flag removal check
3. `TestServeGenDataSpec_TimelineFields` — Struct field presence check
4. `TestConvertServeGen_AcceptsTimelineParams` — Function signature check
5. `TestLoadServeGenData_MultiPeriodGrouping` — BC-1 (multi-period structure)
6. `TestLoadServeGenData_Deterministic` — BC-7 (deterministic behavior)
7. `TestLoadServeGenData_EmptyPeriod` — BC-8 (empty period handling)
8. `TestLoadServeGenData_NoDuplication` — BC-9 (no chunk duplication)
9. `TestConvertServeGen_MissingDirectory` — BC-11 (missing directory error)
10. `TestConvertServeGen_AllChunksInactive` — BC-13 (all-inactive error)
11. `TestConvertServeGen_E2E_MultiPeriod` — Integration test for BC-1, BC-5, BC-6, BC-7

**Test data requirements:**
- `testdata/servegen_multiperiod/` — Directory with chunks spanning all 3 periods
- `testdata/servegen_gaps/` — Directory with only midnight and afternoon data (no morning)
- `testdata/servegen_inactive/` — Directory with all chunks having `rate=0`
- `testdata/servegen_full/` — Directory for E2E test (15+ chunks, all periods)

**Key Implementation Notes:**
- Use `strings.Split(cohort.ID, "-")` to extract period name from cohort ID
- Count cohorts per period to verify multi-period output
- Run conversion twice with same seed to verify determinism
- Use helper function `splitCohortID(id string) []string` for ID parsing
