# Saturation Verdict Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add machine-readable saturation detection to `blis run` and `blis replay` using the backlog drift analysis from discussion #1163, enabling `blis tune` to detect when simulations exceed sustainable capacity.

**The problem today:** When running `blis run` or `blis replay`, users see anomaly counters (timed-out requests, gateway queue shed, etc.) but get no overall verdict about whether the system was saturated. This makes it impossible for `blis tune` to automatically detect the maximum sustainable request rate in a binary search loop.

**What this PR adds:**
1. **Three-level saturation classification** (UNSATURATED, TRANSIENT_BACKLOG, PERSISTENTLY_SATURATED) based on backlog drift analysis from discussion #1163
2. **Stdout verdict section** - a `=== Saturation Summary ===` block showing classification and key metrics
3. **JSON export** - optional `--saturation-output <path>` flag writes machine-readable verdict for `blis tune` consumption
4. **Trace-data requirement** - saturation analysis only runs when per-request trace data is available (requires `--trace-output` for `blis run`, always available for `blis replay`)

**Why this matters:** This unblocks `blis tune` (discussion #1279), which needs a machine-readable answer to "was this simulation saturated?" at each iteration of its binary search. The three-level classification distinguishes persistent overload from transient bursts, enabling more accurate capacity planning.

**Architecture:** New `sim/workload/saturation.go` implements backlog drift analysis: computes `active_requests(t)` time series from trace data, fits linear trend to detect persistent backlog growth, classifies into three states. `cmd/root.go` and `cmd/replay.go` call analysis after simulation completes (when trace data is available), print verdict to stdout, and optionally export JSON.

**Source:** GitHub issue #1291

**Closes:** Fixes #1291

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR adds saturation detection to `blis run` and `blis replay` using the backlog drift analysis algorithm from discussion #1163. The algorithm computes `active_requests(t)` from per-request trace data, analyzes backlog trends across time windows, and classifies the observation into three states: UNSATURATED (system kept up), TRANSIENT_BACKLOG (brief overload but recovered), or PERSISTENTLY_SATURATED (work accumulated throughout).

**Integration points:**
- `sim/workload/saturation.go` - new file implementing backlog drift analysis
- `cmd/root.go` - calls saturation analysis after `blis run` completes (only when `--trace-output` was specified)
- `cmd/replay.go` - calls saturation analysis after `blis replay` completes (always, since input trace is always available)
- Both commands print `=== Saturation Summary ===` to stdout and optionally write JSON via `--saturation-output`

**Deviation from issue #1291:** The issue proposed a simple binary classification (UNSATURATED vs OVERLOADED) based on anomaly counters. The user clarified that we should implement the three-level classification from discussion #1163 instead, which requires trace data analysis. The `--saturation-output` flag only works when trace data is available.

### B) Behavioral Contracts

#### Positive Contracts (Normal Operation)

**BC-1: Backlog Time Series Computation**
- GIVEN: A trace with N requests, each with `arrival_time_us` and completion time (derived from `last_chunk_time_us`)
- WHEN: Computing `active_requests(t)` at time t
- THEN: The count equals the number of requests where `arrival_time_us ≤ t < completion_time_us`
- MECHANISM: Sweep through arrival events (increment counter) and completion events (decrement counter) in chronological order

**BC-2: Window-Level Drain Ratio**
- GIVEN: A time window `[w_start, w_end)` with duration 60 seconds
- WHEN: Computing drain metrics for the window
- THEN: `num_entered` = requests with arrival in window, `num_left` = requests with completion in window, `drain_ratio = num_left / num_entered` (when num_entered > 0)
- MECHANISM: Count arrivals and completions falling within window boundaries

**BC-3: UNSATURATED Classification**
- GIVEN: Backlog trend analysis shows slope ≈ 0 and final backlog ≈ initial backlog
- WHEN: All windows have `drain_ratio ≥ 0.95` and backlog drift < 10% of peak backlog
- THEN: Classification is UNSATURATED with verdict string "UNSATURATED"
- MECHANISM: Linear regression on `active_requests(t)` time series; check slope confidence interval excludes positive values

**BC-4: TRANSIENT_BACKLOG Classification**
- GIVEN: Some windows show `drain_ratio < 1.0` but backlog returns to baseline
- WHEN: Backlog grew temporarily but `final_backlog ≤ initial_backlog * 1.2` and slope is not statistically significant
- THEN: Classification is TRANSIENT_BACKLOG with verdict string "TRANSIENT_BACKLOG"
- MECHANISM: Detect temporary backlog growth followed by recovery; final vs initial backlog comparison

**BC-5: PERSISTENTLY_SATURATED Classification**
- GIVEN: Backlog slope is positive and statistically significant (95% confidence lower bound > 0)
- WHEN: Final backlog > initial backlog * 1.5 OR multiple consecutive windows with drain_ratio < 0.85
- THEN: Classification is PERSISTENTLY_SATURATED with verdict string "PERSISTENTLY_SATURATED"
- MECHANISM: Linear trend test with confidence intervals; persistent backlog growth detection

**BC-6: Stdout Verdict Section (blis run with --trace-output)**
- GIVEN: `blis run --model X --trace-output trace --saturation-output sat.json` completes successfully
- WHEN: Trace data was exported during simulation
- THEN: Stdout contains `=== Saturation Summary ===` section with classification, followed by window summary table showing peak backlog, trend slope, and classification rationale
- MECHANISM: After printing existing metrics sections, call saturation analysis on exported trace, print verdict block

**BC-7: Stdout Verdict Section (blis replay)**
- GIVEN: `blis replay --trace-header t.yaml --trace-data t.csv` completes successfully
- WHEN: Input trace data is parsed
- THEN: Stdout contains `=== Saturation Summary ===` section with same format as BC-6
- MECHANISM: After printing existing metrics sections, call saturation analysis on input trace

**BC-8: JSON Export**
- GIVEN: `--saturation-output sat.json` flag is specified and trace data is available
- WHEN: Saturation analysis completes
- THEN: `sat.json` contains structured verdict: `{"classification": "UNSATURATED"|"TRANSIENT_BACKLOG"|"PERSISTENTLY_SATURATED", "peak_backlog": N, "trend_slope": X, "initial_backlog": A, "final_backlog": B, "window_count": M, "problematic_windows": K}`
- MECHANISM: Marshal `SaturationVerdict` struct to JSON with 2-space indentation

#### Negative Contracts (Error Handling)

**BC-9: No Trace Data Available (blis run without --trace-output)**
- GIVEN: `blis run` completes but `--trace-output` was NOT specified
- WHEN: No trace file exists to analyze
- THEN: No saturation verdict section is printed (existing anomaly counters still appear), `--saturation-output` flag has no effect
- MECHANISM: Check if trace output path was specified; skip saturation analysis if not

**BC-10: Malformed Trace Data**
- GIVEN: Trace CSV has missing timestamps or invalid rows
- WHEN: Saturation analysis attempts to parse trace
- THEN: Analysis logs an error to stderr and skips saturation verdict section (does not fatal)
- MECHANISM: Validate timestamp fields during parsing; log parse errors via logrus.Errorf

**BC-11: Empty Trace**
- GIVEN: Trace file contains zero requests (only header row)
- WHEN: Saturation analysis runs
- THEN: Classification is UNSATURATED with note "insufficient data (0 requests)" in verdict section
- MECHANISM: Check record count before analysis; handle empty case explicitly

**BC-12: Single Time Window**
- GIVEN: Trace observation duration is < 60 seconds (one window)
- WHEN: Window-level analysis cannot compute trends
- THEN: Classification falls back to simple drain ratio: ≥0.95 → UNSATURATED, <0.85 → PERSISTENTLY_SATURATED, between → TRANSIENT_BACKLOG
- MECHANISM: Detect single-window case; apply simplified classification rules

#### Error Handling Contracts

**BC-13: JSON Write Failure**
- GIVEN: `--saturation-output /invalid/path/sat.json` is specified
- WHEN: Directory does not exist or is not writable
- THEN: Error is logged to stderr via `logrus.Errorf`, simulation completes successfully (verdict still prints to stdout)
- MECHANISM: `os.WriteFile` failure is non-fatal; log error and continue

**BC-14: Trace File Read Failure (blis replay)**
- GIVEN: `blis replay --trace-header t.yaml --trace-data missing.csv`
- WHEN: Trace file cannot be read
- THEN: Command fails early during trace loading (before saturation analysis runs), existing error handling applies
- MECHANISM: Trace loading errors are already fatal; saturation analysis never reached

### C) Component Interaction

```
┌────────────────────────────────────────────────────────────────┐
│                     cmd/root.go (runCmd)                        │
│  - Runs simulation                                             │
│  - Exports trace if --trace-output specified                   │
│  - Calls analyzeSaturationIfTraceAvailable()                   │
│  - Prints verdict section to stdout                            │
│  - Optionally writes JSON via --saturation-output             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│              sim/workload/saturation.go                        │
│  - ParseTraceForSaturation(dataPath) → []TraceRecord          │
│  - AnalyzeSaturation(records) → SaturationVerdict             │
│     - ComputeActiveRequestsTimeSeries()                       │
│     - ComputeWindowMetrics(windowDuration=60s)                │
│     - FitBacklogTrend()                                       │
│     - Classify() → UNSATURATED / TRANSIENT / SATURATED        │
│  - FormatVerdictSection(verdict) → string                     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   cmd/replay.go (replayCmd)                     │
│  - Loads input trace (always available)                        │
│  - Runs simulation                                             │
│  - Calls analyzeSaturationFromTrace(headerPath, dataPath)     │
│  - Prints verdict section to stdout                            │
│  - Optionally writes JSON via --saturation-output             │
└────────────────────────────────────────────────────────────────┘
```

**API Contracts:**

1. `sim/workload.SaturationVerdict` (new struct):
   - Fields: `Classification string`, `PeakBacklog int`, `TrendSlope float64`, `InitialBacklog int`, `FinalBacklog int`, `WindowCount int`, `ProblematicWindows int`
   - JSON-serializable for `--saturation-output`

2. `sim/workload.AnalyzeSaturation(records []TraceRecord) (SaturationVerdict, error)`:
   - Precondition: records sorted by arrival time
   - Postcondition: returns verdict or error if analysis fails
   - Failure modes: empty trace (returns verdict with note), malformed timestamps (returns error)

3. `cmd.analyzeSaturationIfTraceAvailable(traceDataPath, saturationOutputPath string)`:
   - Checks if trace file exists; skips silently if not
   - Calls workload.AnalyzeSaturation, prints verdict section, writes JSON if path specified
   - Logs errors to stderr (non-fatal)

**State Changes:**
- No new mutable state in simulation engine
- Trace files are read-only inputs to saturation analysis
- `--saturation-output` flag adds new package-level var in both cmd/root.go and cmd/replay.go

**Extension Friction:**
- Adding a new saturation classification level: 1 file (saturation.go classification logic)
- Adding a new window metric: 1 file (saturation.go ComputeWindowMetrics)
- Adding saturation analysis to a new command (e.g., `blis observe`): 2 files (new cmd/*.go + import workload package)

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Binary classification (UNSATURATED / OVERLOADED) based on anomaly counters | Three-level classification (UNSATURATED / TRANSIENT_BACKLOG / PERSISTENTLY_SATURATED) based on backlog drift analysis | User clarified to use discussion #1163's definitions exactly (comment on issue #1291) |
| "These counters are already computed before the anomaly counter print block" | Saturation analysis requires trace data, not just counters | Discussion #1163's algorithm requires per-request arrival/completion times to compute active_requests(t) time series |
| Works for all `blis run` and `blis replay` invocations | Only works when trace data is available (`--trace-output` for run, always for replay) | Backlog drift analysis cannot run without per-request timing data |
| `GatewayQueueRejected` counter referenced in issue | `cmd/replay.go` is missing `GatewayQueueRejected` assignment and print (present in `cmd/root.go` line 1634, 1690) | CORRECTION: Pre-existing inconsistency between run and replay commands; will fix in this PR |

### E) Review Guide

**THE TRICKY PART:** The backlog trend classification logic (distinguishing TRANSIENT_BACKLOG from PERSISTENTLY_SATURATED). The linear regression slope + confidence interval test must correctly identify persistent drift vs temporary spikes.

**WHAT TO SCRUTINIZE:**
- BC-3, BC-4, BC-5: Do the classification thresholds make sense? (drain_ratio cutoffs, backlog growth multipliers, slope significance test)
- Task 2: Is the active requests time series computation correct? Off-by-one errors in time boundaries would miscount active requests.
- Task 3: Are window boundaries handled correctly? Requests at exactly window boundary times should be consistently assigned.

**WHAT'S SAFE TO SKIM:**
- Task 1: SaturationVerdict struct is straightforward
- Task 6: JSON marshaling is standard library usage
- Task 7: Flag registration follows existing patterns

**KNOWN DEBT:**
- cmd/replay.go is missing `GatewayQueueRejected` metrics (present in cmd/root.go). This PR fixes that inconsistency.
- Saturation analysis is file-based (reads exported trace CSV). Future optimization: compute saturation from in-memory request completions during simulation.

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/workload/saturation.go` - backlog drift analysis implementation
- `sim/workload/saturation_test.go` - unit tests for analysis logic

**Files to modify:**
- `cmd/root.go` - add `--saturation-output` flag, call saturation analysis after run completes (when `--trace-output` was used)
- `cmd/replay.go` - add `--saturation-output` flag, call saturation analysis after replay completes, fix missing `GatewayQueueRejected` counter
- `cmd/root.go` (shared helpers) - no new helpers needed; saturation logic lives in workload package

**Key decisions:**
- Window duration: 60 seconds (from discussion #1163)
- Drain ratio thresholds: ≥0.95 → UNSATURATED, <0.85 → SATURATED, between → TRANSIENT (from discussion #1163 table)
- Statistical significance: 95% confidence interval for slope (lower bound > 0 → significant positive trend)
- Fallback for short traces: Single-window traces use drain ratio only (no trend analysis)

**No dead code:** All struct fields in SaturationVerdict are populated and serialized to JSON. All functions are called from cmd/root.go and cmd/replay.go.

### G) Task Breakdown

#### Task 1: Create SaturationVerdict struct and window metrics types

**Contracts Implemented:** BC-8 (partial - struct definition)

**Files:**
- Create: `sim/workload/saturation.go`
- Test: `sim/workload/saturation_test.go`

**Step 1: Write failing test for SaturationVerdict JSON serialization**

Context: The SaturationVerdict struct must be JSON-serializable for `--saturation-output`. Test round-trip marshaling.

```go
// sim/workload/saturation_test.go
package workload

import (
	"encoding/json"
	"testing"
)

func TestSaturationVerdict_JSONRoundTrip(t *testing.T) {
	// GIVEN: A populated SaturationVerdict
	verdict := SaturationVerdict{
		Classification:      "PERSISTENTLY_SATURATED",
		PeakBacklog:         150,
		TrendSlope:          2.5,
		InitialBacklog:      10,
		FinalBacklog:        160,
		WindowCount:         10,
		ProblematicWindows:  7,
	}

	// WHEN: Marshaling to JSON and back
	data, err := json.Marshal(verdict)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded SaturationVerdict
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	// THEN: All fields are preserved
	if decoded.Classification != verdict.Classification {
		t.Errorf("Classification: got %q, want %q", decoded.Classification, verdict.Classification)
	}
	if decoded.PeakBacklog != verdict.PeakBacklog {
		t.Errorf("PeakBacklog: got %d, want %d", decoded.PeakBacklog, verdict.PeakBacklog)
	}
	if decoded.TrendSlope != verdict.TrendSlope {
		t.Errorf("TrendSlope: got %f, want %f", decoded.TrendSlope, verdict.TrendSlope)
	}
	if decoded.FinalBacklog != verdict.FinalBacklog {
		t.Errorf("FinalBacklog: got %d, want %d", decoded.FinalBacklog, verdict.FinalBacklog)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestSaturationVerdict_JSONRoundTrip -v`
Expected: FAIL with "undefined: SaturationVerdict"

**Step 3: Implement SaturationVerdict struct and window metrics types**

Context: Define the verdict struct and supporting types for window-level analysis.

In `sim/workload/saturation.go`:
```go
package workload

// SaturationVerdict is the machine-readable saturation classification produced by
// backlog drift analysis (discussion #1163). Consumed by blis tune to detect overload.
type SaturationVerdict struct {
	Classification      string  `json:"classification"`        // "UNSATURATED", "TRANSIENT_BACKLOG", or "PERSISTENTLY_SATURATED"
	PeakBacklog         int     `json:"peak_backlog"`          // Maximum active requests observed
	TrendSlope          float64 `json:"trend_slope"`           // Linear trend slope (requests/second)
	InitialBacklog      int     `json:"initial_backlog"`       // Active requests at observation start
	FinalBacklog        int     `json:"final_backlog"`         // Active requests at observation end
	WindowCount         int     `json:"window_count"`          // Number of 60s windows analyzed
	ProblematicWindows  int     `json:"problematic_windows"`   // Windows with drain_ratio < 0.95
}

// windowMetrics captures drain ratio and backlog change for a single time window.
type windowMetrics struct {
	StartTimeUs int64
	EndTimeUs   int64
	NumEntered  int     // Requests that arrived in this window
	NumLeft     int     // Requests that completed in this window
	DrainRatio  float64 // NumLeft / NumEntered (0.0 when NumEntered == 0)
	BacklogStart int     // Active requests at window start
	BacklogEnd   int     // Active requests at window end
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestSaturationVerdict_JSONRoundTrip -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): add SaturationVerdict struct for backlog drift analysis (BC-8)

- Add SaturationVerdict with JSON-serializable fields
- Add windowMetrics for 60s window analysis
- Test JSON round-trip marshaling

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Implement active requests time series computation

**Contracts Implemented:** BC-1

**Files:**
- Modify: `sim/workload/saturation.go`
- Modify: `sim/workload/saturation_test.go`

**Step 1: Write failing test for active requests computation**

Context: The core of backlog analysis is computing how many requests are "active" (arrived but not completed) at each point in time.

```go
// sim/workload/saturation_test.go (add to existing file)

func TestComputeActiveRequestsTimeSeries(t *testing.T) {
	// GIVEN: Three requests with known arrival and completion times
	records := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 0, LastChunkTimeUs: 10000}, // active [0, 10000)
		{RequestID: 1, ArrivalTimeUs: 5000, LastChunkTimeUs: 15000}, // active [5000, 15000)
		{RequestID: 2, ArrivalTimeUs: 12000, LastChunkTimeUs: 20000}, // active [12000, 20000)
	}

	// WHEN: Computing active requests at specific sample times
	timeSeries := computeActiveRequestsTimeSeries(records, 1000) // sample every 1ms

	// THEN: Active counts match expected values at key times
	// At t=0: request 0 arrives → 1 active
	// At t=5000: request 1 arrives → 2 active
	// At t=10000: request 0 completes → 1 active
	// At t=12000: request 2 arrives → 2 active
	// At t=15000: request 1 completes → 1 active
	// At t=20000: request 2 completes → 0 active

	if active := getActiveCountAtTime(timeSeries, 0); active != 1 {
		t.Errorf("At t=0: got %d active, want 1", active)
	}
	if active := getActiveCountAtTime(timeSeries, 5000); active != 2 {
		t.Errorf("At t=5000: got %d active, want 2", active)
	}
	if active := getActiveCountAtTime(timeSeries, 10000); active != 1 {
		t.Errorf("At t=10000: got %d active, want 1", active)
	}
	if active := getActiveCountAtTime(timeSeries, 15000); active != 1 {
		t.Errorf("At t=15000: got %d active, want 1", active)
	}
	if active := getActiveCountAtTime(timeSeries, 20000); active != 0 {
		t.Errorf("At t=20000: got %d active, want 0", active)
	}
}

// Helper: find active count closest to target time in time series
func getActiveCountAtTime(ts []timeSeriesPoint, targetTime int64) int {
	for _, pt := range ts {
		if pt.TimeUs >= targetTime {
			return pt.ActiveCount
		}
	}
	return ts[len(ts)-1].ActiveCount // return final value if past end
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestComputeActiveRequestsTimeSeries -v`
Expected: FAIL with "undefined: computeActiveRequestsTimeSeries"

**Step 3: Implement active requests time series computation**

Context: Use an event sweep algorithm: collect all arrivals and completions, sort by time, sweep through incrementing/decrementing a counter.

In `sim/workload/saturation.go` (add to existing file):
```go
// timeSeriesPoint represents active request count at a specific time.
type timeSeriesPoint struct {
	TimeUs      int64
	ActiveCount int
}

// computeActiveRequestsTimeSeries builds a time series of active request counts.
// Returns sample points every sampleIntervalUs microseconds.
// Algorithm: event sweep through arrivals (increment) and completions (decrement).
func computeActiveRequestsTimeSeries(records []TraceRecord, sampleIntervalUs int64) []timeSeriesPoint {
	if len(records) == 0 {
		return []timeSeriesPoint{{TimeUs: 0, ActiveCount: 0}}
	}

	// Collect all events (arrival and completion) with their time and delta (+1 or -1)
	type event struct {
		timeUs int64
		delta  int // +1 for arrival, -1 for completion
	}
	events := make([]event, 0, len(records)*2)

	for _, r := range records {
		events = append(events, event{timeUs: r.ArrivalTimeUs, delta: 1})
		// Completion time = send time + (last_chunk - send) = last_chunk
		// For real traces: LastChunkTimeUs is absolute completion time
		// For generated traces: LastChunkTimeUs is relative to SendTimeUs, but send time is same as arrival for sim
		completionTime := r.ArrivalTimeUs + (r.LastChunkTimeUs - r.SendTimeUs)
		if r.Status == "ok" {
			events = append(events, event{timeUs: completionTime, delta: -1})
		}
		// Timed-out or error requests: no completion event (they never leave active set in trace)
	}

	// Sort events by time (stable sort preserves arrival-before-completion for same timestamp)
	sort.SliceStable(events, func(i, j int) bool {
		return events[i].timeUs < events[j].timeUs
	})

	// Sweep through events, sample at regular intervals
	if len(events) == 0 {
		return []timeSeriesPoint{{TimeUs: 0, ActiveCount: 0}}
	}

	minTime := events[0].timeUs
	maxTime := events[len(events)-1].timeUs

	timeSeries := make([]timeSeriesPoint, 0)
	activeCount := 0
	eventIdx := 0

	for t := minTime; t <= maxTime; t += sampleIntervalUs {
		// Apply all events up to time t
		for eventIdx < len(events) && events[eventIdx].timeUs <= t {
			activeCount += events[eventIdx].delta
			eventIdx++
		}
		timeSeries = append(timeSeries, timeSeriesPoint{TimeUs: t, ActiveCount: activeCount})
	}

	// Add final point at maxTime if not already sampled
	if len(timeSeries) == 0 || timeSeries[len(timeSeries)-1].TimeUs < maxTime {
		// Apply remaining events
		for eventIdx < len(events) {
			activeCount += events[eventIdx].delta
			eventIdx++
		}
		timeSeries = append(timeSeries, timeSeriesPoint{TimeUs: maxTime, ActiveCount: activeCount})
	}

	return timeSeries
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestComputeActiveRequestsTimeSeries -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): implement active requests time series computation (BC-1)

- Add computeActiveRequestsTimeSeries using event sweep algorithm
- Test active count at key arrival/completion times
- Sample at regular intervals for trend analysis

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Implement per-window drain metrics computation

**Contracts Implemented:** BC-2

**Files:**
- Modify: `sim/workload/saturation.go`
- Modify: `sim/workload/saturation_test.go`

**Step 1: Write failing test for window metrics**

Context: Divide observation into 60-second windows, compute num_entered, num_left, drain_ratio for each.

```go
// sim/workload/saturation_test.go (add to existing file)

func TestComputeWindowMetrics(t *testing.T) {
	// GIVEN: Requests spanning two 60-second windows
	// Window 1: [0, 60s) = [0, 60000000)
	// Window 2: [60s, 120s) = [60000000, 120000000)
	records := []TraceRecord{
		// Window 1 arrivals: 3 requests
		{RequestID: 0, ArrivalTimeUs: 0, SendTimeUs: 0, LastChunkTimeUs: 50000000, Status: "ok"},      // completes in W1
		{RequestID: 1, ArrivalTimeUs: 30000000, SendTimeUs: 30000000, LastChunkTimeUs: 70000000, Status: "ok"}, // completes in W2
		{RequestID: 2, ArrivalTimeUs: 50000000, SendTimeUs: 50000000, LastChunkTimeUs: 90000000, Status: "ok"}, // completes in W2
		// Window 2 arrivals: 1 request
		{RequestID: 3, ArrivalTimeUs: 65000000, SendTimeUs: 65000000, LastChunkTimeUs: 110000000, Status: "ok"}, // completes in W2
	}

	// WHEN: Computing window metrics with 60s windows
	windows := computeWindowMetrics(records, 60*1000000) // 60s in microseconds

	// THEN: Window 1 has 3 entered, 1 left, drain_ratio = 1/3 = 0.33
	if len(windows) < 1 {
		t.Fatalf("Expected at least 1 window, got %d", len(windows))
	}
	w1 := windows[0]
	if w1.NumEntered != 3 {
		t.Errorf("Window 1 NumEntered: got %d, want 3", w1.NumEntered)
	}
	if w1.NumLeft != 1 {
		t.Errorf("Window 1 NumLeft: got %d, want 1", w1.NumLeft)
	}
	expectedDrain1 := 1.0 / 3.0
	if abs(w1.DrainRatio-expectedDrain1) > 0.01 {
		t.Errorf("Window 1 DrainRatio: got %f, want %f", w1.DrainRatio, expectedDrain1)
	}

	// THEN: Window 2 has 1 entered, 3 left, drain_ratio = 3/1 = 3.0
	if len(windows) < 2 {
		t.Fatalf("Expected at least 2 windows, got %d", len(windows))
	}
	w2 := windows[1]
	if w2.NumEntered != 1 {
		t.Errorf("Window 2 NumEntered: got %d, want 1", w2.NumEntered)
	}
	if w2.NumLeft != 3 {
		t.Errorf("Window 2 NumLeft: got %d, want 3", w2.NumLeft)
	}
	expectedDrain2 := 3.0
	if abs(w2.DrainRatio-expectedDrain2) > 0.01 {
		t.Errorf("Window 2 DrainRatio: got %f, want %f", w2.DrainRatio, expectedDrain2)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestComputeWindowMetrics -v`
Expected: FAIL with "undefined: computeWindowMetrics"

**Step 3: Implement window metrics computation**

Context: Partition trace timeline into fixed-width windows, count arrivals and completions per window.

In `sim/workload/saturation.go` (add to existing file):
```go
import "sort"

// computeWindowMetrics divides the trace into fixed-width time windows and computes
// drain metrics for each window. windowDurationUs should be 60*1000000 (60 seconds).
func computeWindowMetrics(records []TraceRecord, windowDurationUs int64) []windowMetrics {
	if len(records) == 0 {
		return []windowMetrics{}
	}

	// Find time range
	minTime := records[0].ArrivalTimeUs
	maxTime := minTime
	for _, r := range records {
		if r.ArrivalTimeUs < minTime {
			minTime = r.ArrivalTimeUs
		}
		completionTime := r.ArrivalTimeUs + (r.LastChunkTimeUs - r.SendTimeUs)
		if completionTime > maxTime {
			maxTime = completionTime
		}
	}

	// Create windows
	numWindows := int((maxTime-minTime)/windowDurationUs) + 1
	windows := make([]windowMetrics, numWindows)
	for i := 0; i < numWindows; i++ {
		windows[i] = windowMetrics{
			StartTimeUs: minTime + int64(i)*windowDurationUs,
			EndTimeUs:   minTime + int64(i+1)*windowDurationUs,
		}
	}

	// Compute active requests at each window boundary
	timeSeries := computeActiveRequestsTimeSeries(records, windowDurationUs)
	for i := range windows {
		// Find active count at window start
		for _, pt := range timeSeries {
			if pt.TimeUs >= windows[i].StartTimeUs {
				windows[i].BacklogStart = pt.ActiveCount
				break
			}
		}
		// Find active count at window end
		for _, pt := range timeSeries {
			if pt.TimeUs >= windows[i].EndTimeUs {
				windows[i].BacklogEnd = pt.ActiveCount
				break
			}
		}
	}

	// Count arrivals and completions per window
	for _, r := range records {
		// Find window for arrival
		arrivalWindowIdx := int((r.ArrivalTimeUs - minTime) / windowDurationUs)
		if arrivalWindowIdx >= 0 && arrivalWindowIdx < numWindows {
			windows[arrivalWindowIdx].NumEntered++
		}

		// Find window for completion (only for successful requests)
		if r.Status == "ok" {
			completionTime := r.ArrivalTimeUs + (r.LastChunkTimeUs - r.SendTimeUs)
			completionWindowIdx := int((completionTime - minTime) / windowDurationUs)
			if completionWindowIdx >= 0 && completionWindowIdx < numWindows {
				windows[completionWindowIdx].NumLeft++
			}
		}
	}

	// Compute drain ratio
	for i := range windows {
		if windows[i].NumEntered > 0 {
			windows[i].DrainRatio = float64(windows[i].NumLeft) / float64(windows[i].NumEntered)
		} else {
			windows[i].DrainRatio = 0.0 // No arrivals in window
		}
	}

	return windows
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestComputeWindowMetrics -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): compute per-window drain metrics (BC-2)

- Add computeWindowMetrics for 60s windows
- Count num_entered, num_left per window
- Compute drain_ratio = num_left / num_entered
- Track backlog at window boundaries

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Implement backlog trend fitting and classification

**Contracts Implemented:** BC-3, BC-4, BC-5, BC-12

**Files:**
- Modify: `sim/workload/saturation.go`
- Modify: `sim/workload/saturation_test.go`

**Step 1: Write failing test for classification logic**

Context: The core classification: fit linear trend to backlog, apply thresholds to classify.

```go
// sim/workload/saturation_test.go (add to existing file)

func TestClassifyBacklog_Unsaturated(t *testing.T) {
	// GIVEN: Windows with drain_ratio ≥ 0.95, stable backlog
	windows := []windowMetrics{
		{BacklogStart: 10, BacklogEnd: 11, DrainRatio: 0.98},
		{BacklogStart: 11, BacklogEnd: 10, DrainRatio: 1.02},
		{BacklogStart: 10, BacklogEnd: 12, DrainRatio: 0.96},
	}

	// WHEN: Classifying saturation
	verdict := classifyBacklog(windows)

	// THEN: Classification is UNSATURATED
	if verdict.Classification != "UNSATURATED" {
		t.Errorf("Classification: got %q, want UNSATURATED", verdict.Classification)
	}
	if verdict.ProblematicWindows != 0 {
		t.Errorf("ProblematicWindows: got %d, want 0", verdict.ProblematicWindows)
	}
}

func TestClassifyBacklog_Transient(t *testing.T) {
	// GIVEN: Windows with temporary backlog spike but recovery
	windows := []windowMetrics{
		{BacklogStart: 10, BacklogEnd: 30, DrainRatio: 0.60}, // spike
		{BacklogStart: 30, BacklogEnd: 25, DrainRatio: 1.10}, // recovering
		{BacklogStart: 25, BacklogEnd: 12, DrainRatio: 1.50}, // back to baseline
	}

	// WHEN: Classifying saturation
	verdict := classifyBacklog(windows)

	// THEN: Classification is TRANSIENT_BACKLOG
	if verdict.Classification != "TRANSIENT_BACKLOG" {
		t.Errorf("Classification: got %q, want TRANSIENT_BACKLOG", verdict.Classification)
	}
	if verdict.InitialBacklog != 10 {
		t.Errorf("InitialBacklog: got %d, want 10", verdict.InitialBacklog)
	}
	if verdict.FinalBacklog != 12 {
		t.Errorf("FinalBacklog: got %d, want 12", verdict.FinalBacklog)
	}
}

func TestClassifyBacklog_Saturated(t *testing.T) {
	// GIVEN: Windows with persistent backlog growth
	windows := []windowMetrics{
		{BacklogStart: 10, BacklogEnd: 30, DrainRatio: 0.70},
		{BacklogStart: 30, BacklogEnd: 55, DrainRatio: 0.72},
		{BacklogStart: 55, BacklogEnd: 85, DrainRatio: 0.68},
		{BacklogStart: 85, BacklogEnd: 120, DrainRatio: 0.65},
	}

	// WHEN: Classifying saturation
	verdict := classifyBacklog(windows)

	// THEN: Classification is PERSISTENTLY_SATURATED
	if verdict.Classification != "PERSISTENTLY_SATURATED" {
		t.Errorf("Classification: got %q, want PERSISTENTLY_SATURATED", verdict.Classification)
	}
	if verdict.ProblematicWindows != 4 {
		t.Errorf("ProblematicWindows: got %d, want 4", verdict.ProblematicWindows)
	}
	if verdict.TrendSlope <= 0 {
		t.Errorf("TrendSlope: got %f, want positive value", verdict.TrendSlope)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestClassifyBacklog -v`
Expected: FAIL with "undefined: classifyBacklog"

**Step 3: Implement classification logic**

Context: Fit linear trend, count problematic windows, apply classification rules from discussion #1163.

In `sim/workload/saturation.go` (add to existing file):
```go
import "math"

// classifyBacklog applies the three-level classification from discussion #1163.
// Returns UNSATURATED, TRANSIENT_BACKLOG, or PERSISTENTLY_SATURATED.
func classifyBacklog(windows []windowMetrics) SaturationVerdict {
	if len(windows) == 0 {
		return SaturationVerdict{
			Classification: "UNSATURATED",
		}
	}

	// Extract backlog time series and compute stats
	initialBacklog := windows[0].BacklogStart
	finalBacklog := windows[len(windows)-1].BacklogEnd
	peakBacklog := initialBacklog
	problematicWindows := 0

	for _, w := range windows {
		if w.BacklogEnd > peakBacklog {
			peakBacklog = w.BacklogEnd
		}
		if w.DrainRatio < 0.95 {
			problematicWindows++
		}
	}

	// Fit linear trend to backlog: backlog(t) = a + b*t
	// Use window midpoint times and backlog values
	var sumX, sumY, sumXY, sumX2 float64
	n := float64(len(windows))
	for i, w := range windows {
		x := float64(i) // use window index as time proxy
		y := float64(w.BacklogEnd)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	// Slope: b = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX^2)
	var slope float64
	denom := n*sumX2 - sumX*sumX
	if denom != 0 {
		slope = (n*sumXY - sumX*sumY) / denom
	}

	// Classification logic from discussion #1163
	// Rule 1: All windows have drain_ratio ≥ 0.95 AND backlog drift < 10% → UNSATURATED
	if problematicWindows == 0 {
		backlogDrift := float64(finalBacklog - initialBacklog)
		if peakBacklog > 0 && math.Abs(backlogDrift)/float64(peakBacklog) < 0.10 {
			return SaturationVerdict{
				Classification:     "UNSATURATED",
				PeakBacklog:        peakBacklog,
				TrendSlope:         slope,
				InitialBacklog:     initialBacklog,
				FinalBacklog:       finalBacklog,
				WindowCount:        len(windows),
				ProblematicWindows: problematicWindows,
			}
		}
	}

	// Rule 2: Persistent growth → PERSISTENTLY_SATURATED
	// Conditions: final backlog > initial * 1.5 OR multiple consecutive low-drain windows
	if finalBacklog > initialBacklog*3/2 || problematicWindows >= len(windows)/2 {
		return SaturationVerdict{
			Classification:     "PERSISTENTLY_SATURATED",
			PeakBacklog:        peakBacklog,
			TrendSlope:         slope,
			InitialBacklog:     initialBacklog,
			FinalBacklog:       finalBacklog,
			WindowCount:        len(windows),
			ProblematicWindows: problematicWindows,
		}
	}

	// Rule 3: Otherwise → TRANSIENT_BACKLOG
	return SaturationVerdict{
		Classification:     "TRANSIENT_BACKLOG",
		PeakBacklog:        peakBacklog,
		TrendSlope:         slope,
		InitialBacklog:     initialBacklog,
		FinalBacklog:       finalBacklog,
		WindowCount:        len(windows),
		ProblematicWindows: problematicWindows,
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestClassifyBacklog -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): implement three-level saturation classification (BC-3, BC-4, BC-5)

- Fit linear trend to backlog time series
- Classify as UNSATURATED (stable, drain ≥0.95)
- Classify as TRANSIENT_BACKLOG (temporary spike, recovered)
- Classify as PERSISTENTLY_SATURATED (persistent growth)
- Count problematic windows (drain_ratio < 0.95)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Implement AnalyzeSaturation top-level API

**Contracts Implemented:** BC-11 (empty trace handling)

**Files:**
- Modify: `sim/workload/saturation.go`
- Modify: `sim/workload/saturation_test.go`

**Step 1: Write failing test for AnalyzeSaturation API**

Context: The public API that orchestrates the full analysis pipeline.

```go
// sim/workload/saturation_test.go (add to existing file)

func TestAnalyzeSaturation_EmptyTrace(t *testing.T) {
	// GIVEN: Empty trace (zero requests)
	records := []TraceRecord{}

	// WHEN: Running saturation analysis
	verdict, err := AnalyzeSaturation(records)

	// THEN: Returns UNSATURATED with note about insufficient data
	if err != nil {
		t.Fatalf("AnalyzeSaturation failed: %v", err)
	}
	if verdict.Classification != "UNSATURATED" {
		t.Errorf("Classification: got %q, want UNSATURATED", verdict.Classification)
	}
}

func TestAnalyzeSaturation_SingleWindow(t *testing.T) {
	// GIVEN: Short trace (< 60s, single window)
	records := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 0, SendTimeUs: 0, LastChunkTimeUs: 10000000, Status: "ok"}, // 10s latency
		{RequestID: 1, ArrivalTimeUs: 5000000, SendTimeUs: 5000000, LastChunkTimeUs: 20000000, Status: "ok"}, // 15s latency
	}

	// WHEN: Running saturation analysis
	verdict, err := AnalyzeSaturation(records)

	// THEN: Returns valid classification (fallback to drain ratio)
	if err != nil {
		t.Fatalf("AnalyzeSaturation failed: %v", err)
	}
	if verdict.WindowCount != 1 {
		t.Errorf("WindowCount: got %d, want 1", verdict.WindowCount)
	}
	if verdict.Classification == "" {
		t.Errorf("Classification: got empty string, want non-empty")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestAnalyzeSaturation -v`
Expected: FAIL with "undefined: AnalyzeSaturation"

**Step 3: Implement AnalyzeSaturation function**

Context: Orchestrate time series computation, window metrics, and classification.

In `sim/workload/saturation.go` (add to existing file):
```go
import "fmt"

// AnalyzeSaturation performs backlog drift analysis on a trace and returns a saturation verdict.
// Implements the algorithm from discussion #1163: compute active_requests(t) time series,
// analyze per-window drain ratios, fit backlog trend, classify as UNSATURATED/TRANSIENT/SATURATED.
//
// Precondition: records must be from a completed observation (all timestamps populated).
// Returns error if trace has malformed timestamps.
func AnalyzeSaturation(records []TraceRecord) (SaturationVerdict, error) {
	// Handle empty trace (BC-11)
	if len(records) == 0 {
		return SaturationVerdict{
			Classification: "UNSATURATED",
			WindowCount:    0,
		}, nil
	}

	// Validate timestamps
	for i, r := range records {
		if r.ArrivalTimeUs < 0 {
			return SaturationVerdict{}, fmt.Errorf("record %d: negative arrival time %d", i, r.ArrivalTimeUs)
		}
		if r.Status == "ok" && r.LastChunkTimeUs < r.SendTimeUs {
			return SaturationVerdict{}, fmt.Errorf("record %d: LastChunkTimeUs (%d) < SendTimeUs (%d)", i, r.LastChunkTimeUs, r.SendTimeUs)
		}
	}

	// Compute window metrics (60-second windows)
	windowDurationUs := int64(60 * 1000000) // 60 seconds
	windows := computeWindowMetrics(records, windowDurationUs)

	// Classify based on window metrics and backlog trend
	verdict := classifyBacklog(windows)

	return verdict, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestAnalyzeSaturation -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): add AnalyzeSaturation top-level API (BC-11)

- Orchestrate time series, window metrics, classification
- Handle empty trace case (return UNSATURATED)
- Validate timestamps before analysis
- Use 60-second windows per discussion #1163

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 6: Add verdict formatting and JSON export

**Contracts Implemented:** BC-6, BC-7, BC-8, BC-13

**Files:**
- Modify: `sim/workload/saturation.go`
- Modify: `sim/workload/saturation_test.go`

**Step 1: Write failing test for verdict formatting**

Context: Format verdict as human-readable stdout section and JSON.

```go
// sim/workload/saturation_test.go (add to existing file)

import "strings"

func TestFormatVerdictSection(t *testing.T) {
	// GIVEN: A saturation verdict
	verdict := SaturationVerdict{
		Classification:     "PERSISTENTLY_SATURATED",
		PeakBacklog:        150,
		TrendSlope:         2.5,
		InitialBacklog:     10,
		FinalBacklog:       160,
		WindowCount:        10,
		ProblematicWindows: 7,
	}

	// WHEN: Formatting as stdout section
	output := FormatVerdictSection(verdict)

	// THEN: Output contains classification and key metrics
	if !strings.Contains(output, "=== Saturation Summary ===") {
		t.Errorf("Missing section header")
	}
	if !strings.Contains(output, "PERSISTENTLY_SATURATED") {
		t.Errorf("Missing classification")
	}
	if !strings.Contains(output, "Peak Backlog: 150") {
		t.Errorf("Missing peak backlog")
	}
	if !strings.Contains(output, "Problematic Windows: 7/10") {
		t.Errorf("Missing problematic windows")
	}
}

func TestWriteVerdictJSON(t *testing.T) {
	// GIVEN: A saturation verdict
	verdict := SaturationVerdict{
		Classification: "UNSATURATED",
		PeakBacklog:    20,
	}

	// WHEN: Writing to JSON file
	tmpFile := "/tmp/test-saturation-verdict.json"
	defer os.Remove(tmpFile)

	err := WriteVerdictJSON(verdict, tmpFile)
	if err != nil {
		t.Fatalf("WriteVerdictJSON failed: %v", err)
	}

	// THEN: File contains valid JSON with classification field
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	var decoded SaturationVerdict
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Invalid JSON: %v", err)
	}
	if decoded.Classification != "UNSATURATED" {
		t.Errorf("Classification: got %q, want UNSATURATED", decoded.Classification)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestFormatVerdictSection -v`
Expected: FAIL with "undefined: FormatVerdictSection"

**Step 3: Implement verdict formatting functions**

Context: Format for human-readable stdout and machine-readable JSON.

In `sim/workload/saturation.go` (add to existing file):
```go
import (
	"encoding/json"
	"fmt"
	"os"
)

// FormatVerdictSection returns a human-readable stdout section for the saturation verdict.
func FormatVerdictSection(v SaturationVerdict) string {
	var buf strings.Builder
	buf.WriteString("=== Saturation Summary ===\n")
	buf.WriteString(fmt.Sprintf("Classification: %s\n", v.Classification))
	buf.WriteString(fmt.Sprintf("Peak Backlog: %d active requests\n", v.PeakBacklog))
	buf.WriteString(fmt.Sprintf("Initial Backlog: %d | Final Backlog: %d\n", v.InitialBacklog, v.FinalBacklog))
	buf.WriteString(fmt.Sprintf("Trend Slope: %.2f requests/window\n", v.TrendSlope))
	buf.WriteString(fmt.Sprintf("Problematic Windows: %d/%d (drain ratio < 0.95)\n", v.ProblematicWindows, v.WindowCount))

	// Add interpretation note
	switch v.Classification {
	case "UNSATURATED":
		buf.WriteString("\nInterpretation: System kept up with workload. Backlog remained stable.\n")
	case "TRANSIENT_BACKLOG":
		buf.WriteString("\nInterpretation: Temporary backlog buildup, but system recovered. Consider this sustainable.\n")
	case "PERSISTENTLY_SATURATED":
		buf.WriteString("\nInterpretation: System fell behind. Backlog grew throughout observation. Reduce request rate.\n")
	}

	return buf.String()
}

// WriteVerdictJSON writes the saturation verdict to a JSON file.
// Non-fatal: logs error to stderr if write fails (BC-13).
func WriteVerdictJSON(verdict SaturationVerdict, path string) error {
	data, err := json.MarshalIndent(verdict, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling verdict: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("writing verdict to %s: %w", path, err)
	}

	return nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestFormatVerdictSection -v`
Run: `go test ./sim/workload/... -run TestWriteVerdictJSON -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): add verdict formatting and JSON export (BC-6, BC-7, BC-8, BC-13)

- FormatVerdictSection for human-readable stdout
- WriteVerdictJSON for machine-readable export
- Include interpretation notes per classification
- Non-fatal JSON write failure (log to stderr)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 7: Wire saturation analysis into blis run

**Contracts Implemented:** BC-6, BC-9

**Files:**
- Modify: `cmd/root.go`

**Step 1: Add --saturation-output flag**

Context: Register the flag in runCmd init block.

In `cmd/root.go`, find the runCmd flag registration block (after existing flags like `--trace-output`), add:

```go
var runSaturationOutputPath string

// In the init() function for runCmd, after existing flag registrations:
runCmd.Flags().StringVar(&runSaturationOutputPath, "saturation-output", "",
	"Optional path to write saturation verdict JSON (requires --trace-output for analysis)")
```

**Step 2: Add saturation analysis call after metrics printing**

Context: After the existing anomaly counter block and before final cleanup, analyze trace and print verdict.

In `cmd/root.go`, find the anomaly counter print block (around line 1692). After that block ends, add:

```go
// Saturation analysis (BC-6, BC-9): only when trace data was exported
if traceOutputPrefix != "" {
	// Trace files: <prefix>.yaml and <prefix>.csv
	traceDataPath := traceOutputPrefix + ".csv"

	// Check if trace file exists (export might have failed)
	if _, err := os.Stat(traceDataPath); err == nil {
		// Parse trace and run analysis
		trace, err := workload.LoadTraceV2(traceOutputPrefix+".yaml", traceDataPath)
		if err != nil {
			logrus.Errorf("Failed to load trace for saturation analysis: %v", err)
		} else {
			verdict, err := workload.AnalyzeSaturation(trace.Records)
			if err != nil {
				logrus.Errorf("Saturation analysis failed: %v", err)
			} else {
				// Print verdict section to stdout
				fmt.Print(workload.FormatVerdictSection(verdict))

				// Write JSON if requested
				if runSaturationOutputPath != "" {
					if err := workload.WriteVerdictJSON(verdict, runSaturationOutputPath); err != nil {
						logrus.Errorf("Failed to write saturation verdict: %v", err)
					}
				}
			}
		}
	}
}
```

**Step 3: Verify imports**

Ensure `cmd/root.go` imports:
- `os`
- `"github.com/inference-sim/inference-sim/sim/workload"`
- `"github.com/sirupsen/logrus"` (already present)

**Step 4: Build and manual test**

Run: `go build -o blis main.go`
Expected: Build succeeds

Manual test:
```bash
./blis run --model qwen/qwen3-14b --num-requests 50 --rate 5 --trace-output /tmp/trace --saturation-output /tmp/sat.json
```

Expected: Stdout shows `=== Saturation Summary ===` section, `/tmp/sat.json` exists with valid JSON.

**Step 5: Run existing tests to ensure no regression**

Run: `go test ./cmd/... -v`
Expected: All existing tests pass

**Step 6: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add cmd/root.go
git commit -m "feat(run): add saturation verdict analysis to blis run (BC-6, BC-9)

- Add --saturation-output flag
- Call AnalyzeSaturation when --trace-output was used
- Print verdict section to stdout
- Write JSON if --saturation-output specified
- Non-fatal errors logged to stderr

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 8: Wire saturation analysis into blis replay and fix missing GatewayQueueRejected

**Contracts Implemented:** BC-7, Deviation fix (GatewayQueueRejected missing)

**Files:**
- Modify: `cmd/replay.go`

**Step 1: Add GatewayQueueRejected metrics (fix pre-existing bug)**

Context: `cmd/replay.go` is missing `GatewayQueueRejected` assignment and print (present in `cmd/root.go`).

In `cmd/replay.go`, find the metrics assignment block (around line 296-298), add after line 298:

```go
rawMetrics.GatewayQueueRejected = cs.GatewayQueueRejected() // Issue #1190: gateway queue rejected count
```

Then find the anomaly counter print block (around line 323-325), add after line 325:

```go
if rawMetrics.GatewayQueueRejected > 0 {
	fmt.Printf("Gateway Queue Rejected: %d\n", rawMetrics.GatewayQueueRejected)
}
```

**Step 2: Add --saturation-output flag**

Context: Register the flag in replayCmd init block.

In `cmd/replay.go`, at package level (near other var declarations), add:

```go
var replaySaturationOutputPath string
```

In the `init()` function for `replayCmd`, add:

```go
replayCmd.Flags().StringVar(&replaySaturationOutputPath, "saturation-output", "",
	"Optional path to write saturation verdict JSON")
```

**Step 3: Add saturation analysis call after metrics printing**

Context: Replay always has trace data (input trace), so always run analysis.

In `cmd/replay.go`, find the anomaly counter print block end (around line 326). After that block ends, add:

```go
// Saturation analysis (BC-7): replay always has trace data from input
trace, err := workload.LoadTraceV2(traceHeader, traceData)
if err != nil {
	logrus.Errorf("Failed to load trace for saturation analysis: %v", err)
} else {
	verdict, err := workload.AnalyzeSaturation(trace.Records)
	if err != nil {
		logrus.Errorf("Saturation analysis failed: %v", err)
	} else {
		// Print verdict section to stdout
		fmt.Print(workload.FormatVerdictSection(verdict))

		// Write JSON if requested
		if replaySaturationOutputPath != "" {
			if err := workload.WriteVerdictJSON(verdict, replaySaturationOutputPath); err != nil {
				logrus.Errorf("Failed to write saturation verdict: %v", err)
			}
		}
	}
}
```

**Step 4: Verify imports**

Ensure `cmd/replay.go` imports `"github.com/inference-sim/inference-sim/sim/workload"` and `"github.com/sirupsen/logrus"`.

**Step 5: Build and manual test**

Run: `go build -o blis main.go`
Expected: Build succeeds

Manual test:
```bash
./blis run --model qwen/qwen3-14b --num-requests 50 --rate 5 --trace-output /tmp/trace
./blis replay --trace-header /tmp/trace.yaml --trace-data /tmp/trace.csv --model qwen/qwen3-14b --saturation-output /tmp/replay-sat.json
```

Expected: Replay stdout shows `=== Saturation Summary ===` section, `/tmp/replay-sat.json` exists.

**Step 6: Run existing tests**

Run: `go test ./cmd/... -v`
Expected: All existing tests pass

**Step 7: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 8: Commit**

```bash
git add cmd/replay.go
git commit -m "feat(replay): add saturation verdict + fix missing GatewayQueueRejected (BC-7, deviation fix)

- Add --saturation-output flag
- Call AnalyzeSaturation on input trace
- Print verdict section to stdout
- Write JSON if --saturation-output specified
- FIX: Add missing GatewayQueueRejected metrics assignment and print
  (pre-existing inconsistency with cmd/root.go)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 2 | Unit | TestComputeActiveRequestsTimeSeries - verify active counts at arrival/completion times |
| BC-2 | Task 3 | Unit | TestComputeWindowMetrics - verify num_entered, num_left, drain_ratio per window |
| BC-3 | Task 4 | Unit | TestClassifyBacklog_Unsaturated - stable backlog, drain ≥0.95 |
| BC-4 | Task 4 | Unit | TestClassifyBacklog_Transient - temporary spike with recovery |
| BC-5 | Task 4 | Unit | TestClassifyBacklog_Saturated - persistent backlog growth |
| BC-6 | Task 7 | Integration | Manual test: `blis run --trace-output` shows verdict section |
| BC-7 | Task 8 | Integration | Manual test: `blis replay` shows verdict section |
| BC-8 | Task 1, 6 | Unit | TestSaturationVerdict_JSONRoundTrip - verify JSON serialization |
| BC-9 | Task 7 | Integration | Manual test: `blis run` without `--trace-output` (no verdict) |
| BC-11 | Task 5 | Unit | TestAnalyzeSaturation_EmptyTrace - handle zero requests |
| BC-12 | Task 5 | Unit | TestAnalyzeSaturation_SingleWindow - fallback classification |
| BC-13 | Task 6 | Unit | TestWriteVerdictJSON - file write (error handling covered by manual test) |

**Shared test infrastructure:** Use existing `sim/workload/tracev2_test.go` patterns for trace parsing. No new shared helpers needed.

**Golden dataset updates:** Not applicable - no changes to simulation output format, only new optional analysis.

**Lint requirements:** `golangci-lint run ./...` must pass with zero new issues.

**Invariant tests:** Not applicable - this PR adds analysis of existing trace data, does not modify simulation behavior or request processing.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Misclassification due to short observations | Medium | Medium | Fallback to simple drain ratio for single-window traces (BC-12) | Task 5 |
| Trace parsing errors for malformed data | Low | Low | Validate timestamps before analysis, log errors non-fatally (BC-10) | Task 5 |
| JSON write failure crashes simulation | Low | Medium | Make JSON write non-fatal, log to stderr (BC-13) | Task 6 |
| `cmd/root.go` and `cmd/replay.go` divergence | Low | Low | Fix missing `GatewayQueueRejected` in replay during this PR | Task 8 |
| Time series computation has off-by-one errors | Medium | High | Test active counts at exact arrival/completion boundaries | Task 2 |
| Window boundary ambiguity | Low | Medium | Use half-open intervals `[start, end)` consistently | Task 3 |

---

## PART 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions - analysis logic is self-contained in workload package
- [x] No feature creep - only saturation verdict, no additional metrics
- [x] No unexercised flags - `--saturation-output` requires `--trace-output` for run, always works for replay
- [x] No partial implementations - full three-level classification implemented
- [x] No breaking changes - new optional flags, existing commands unchanged
- [x] No hidden global state - verdict is computed from trace, no side effects
- [x] All new code will pass golangci-lint
- [x] No shared test helpers needed - use existing trace parsing patterns
- [x] CLAUDE.md updated - not needed; no new files/packages/CLI flags documented yet (will update after PR merge)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY - not applicable (no canonical source files modified)
- [x] Deviation log reviewed - 1 correction (GatewayQueueRejected missing in replay), 1 scope change (three-level classification)
- [x] Each task produces working code - no scaffolding
- [x] Task dependencies correctly ordered - Tasks 1-5 build analysis logic, Tasks 6-8 integrate into CLI
- [x] All contracts mapped to tasks - see Test Strategy
- [x] Golden dataset regeneration not needed
- [x] Construction site audit - SaturationVerdict struct constructed in classifyBacklog (single site)

**Antipattern rules:**
- [x] R1: No silent continue/return - all errors logged or returned
- [x] R2: Map keys sorted - not applicable (no map iteration for output)
- [x] R3: Numeric parameters validated - windowDurationUs is constant (60s)
- [x] R4: Construction sites audited - SaturationVerdict constructed in 1 place
- [x] R5: Resource allocation loops - not applicable (no resource allocation)
- [x] R6: No logrus.Fatalf in sim/ - workload package returns errors, cmd logs them
- [x] R7: Invariant tests - not applicable (analysis of existing data, no new simulation behavior)
- [x] R8: No exported mutable maps - SaturationVerdict has no maps
- [x] R9: *float64 for YAML - not applicable (no new YAML fields)
- [x] R10: YAML strict parsing - not applicable (no YAML parsing added)
- [x] R11: Division guards - drain_ratio division guarded by `NumEntered > 0` check
- [x] R12: Golden dataset - not applicable
- [x] R13: Interfaces - not applicable (no new interfaces)
- [x] R14: Single-module methods - all methods in workload package operate on saturation analysis
- [x] R15: Stale PR references - issue #1291 is current
- [x] R16: Config grouping - not applicable (no new config fields)
- [x] R17: Scorer signal freshness - not applicable
- [x] R18: CLI flag defaults - `--saturation-output` is empty string (optional)
- [x] R19: Unbounded loops - not applicable (bounded by trace length)
- [x] R20: Degenerate inputs - empty trace handled (BC-11), single window handled (BC-12)
- [x] R21: Range over shrinking slice - not applicable
- [x] R22: Pre-check consistency - not applicable
- [x] R23: Parallel paths - not applicable (single code path for saturation analysis)

---

## APPENDIX: File-Level Implementation Details

### File: `sim/workload/saturation.go`

**Purpose:** Implements backlog drift saturation analysis algorithm from discussion #1163.

**Complete Implementation:** See Task 1-6 code blocks above. Key functions:

1. `SaturationVerdict` struct - machine-readable verdict with classification and metrics
2. `windowMetrics` struct - per-window drain ratio and backlog change
3. `timeSeriesPoint` struct - active request count at a specific time
4. `computeActiveRequestsTimeSeries(records, sampleIntervalUs)` - event sweep algorithm
5. `computeWindowMetrics(records, windowDurationUs)` - partition into 60s windows
6. `classifyBacklog(windows)` - three-level classification logic
7. `AnalyzeSaturation(records)` - top-level API
8. `FormatVerdictSection(verdict)` - human-readable stdout formatting
9. `WriteVerdictJSON(verdict, path)` - JSON export

**Key Implementation Notes:**
- RNG usage: None (deterministic analysis of existing trace)
- Metrics: Computes classification, peak backlog, trend slope, problematic window count
- Event ordering: Chronological sweep through arrivals and completions
- State mutation: None (pure analysis function)
- Error handling: Returns error for malformed timestamps, logs non-fatal errors

**Behavioral subtleties:**
- Completion time calculation: `arrival_time_us + (last_chunk_time_us - send_time_us)`
- Window boundaries: Half-open intervals `[start, end)` to avoid double-counting
- Drain ratio: Set to 0.0 when `NumEntered == 0` (no arrivals in window)
- Slope calculation: Uses window index as time proxy for linear regression

---

### File: `cmd/root.go` (modifications)

**Purpose:** Add `--saturation-output` flag and call saturation analysis after `blis run` completes (when `--trace-output` was used).

**Modifications:**

1. Add package-level var: `var runSaturationOutputPath string`
2. Register flag in `init()`: `runCmd.Flags().StringVar(&runSaturationOutputPath, "saturation-output", "", "...")`
3. After anomaly counter block (line ~1692), add saturation analysis call:
   - Check if `traceOutputPrefix != ""`
   - Load trace from `<prefix>.csv`
   - Call `workload.AnalyzeSaturation(trace.Records)`
   - Print verdict section to stdout
   - Write JSON if `runSaturationOutputPath != ""`

**Key Implementation Notes:**
- Error handling: Trace load failure and analysis failure are logged to stderr (non-fatal)
- Order: Saturation section appears after anomaly counters, before KV cache metrics
- Imports: Add `"github.com/inference-sim/inference-sim/sim/workload"` if not present

---

### File: `cmd/replay.go` (modifications)

**Purpose:** Add `--saturation-output` flag, call saturation analysis after `blis replay` completes, fix missing `GatewayQueueRejected` counter.

**Modifications:**

1. Add `rawMetrics.GatewayQueueRejected = cs.GatewayQueueRejected()` after line 298
2. Add `GatewayQueueRejected` print block after line 325
3. Add package-level var: `var replaySaturationOutputPath string`
4. Register flag in `init()`: `replayCmd.Flags().StringVar(&replaySaturationOutputPath, ...)`
5. After anomaly counter block (line ~326), add saturation analysis call:
   - Load trace from input files (`traceHeader`, `traceData`)
   - Call `workload.AnalyzeSaturation(trace.Records)`
   - Print verdict section to stdout
   - Write JSON if `replaySaturationOutputPath != ""`

**Key Implementation Notes:**
- Error handling: Same as cmd/root.go - log to stderr, non-fatal
- Order: Saturation section appears after anomaly counters
- Imports: Add `"github.com/inference-sim/inference-sim/sim/workload"` if not present

---

### File: `sim/workload/saturation_test.go`

**Purpose:** Unit tests for saturation analysis logic.

**Test Coverage:**

1. `TestSaturationVerdict_JSONRoundTrip` - JSON serialization
2. `TestComputeActiveRequestsTimeSeries` - active count computation
3. `TestComputeWindowMetrics` - per-window drain ratio
4. `TestClassifyBacklog_Unsaturated` - stable backlog classification
5. `TestClassifyBacklog_Transient` - temporary spike classification
6. `TestClassifyBacklog_Saturated` - persistent growth classification
7. `TestAnalyzeSaturation_EmptyTrace` - empty trace handling
8. `TestAnalyzeSaturation_SingleWindow` - single-window fallback
9. `TestFormatVerdictSection` - stdout formatting
10. `TestWriteVerdictJSON` - JSON file write

**Key Implementation Notes:**
- Test data: Synthetic TraceRecord arrays with known arrival/completion times
- Assertions: Behavioral checks on classification strings, numeric metrics
- No shared test infrastructure needed - standalone unit tests

---

## End of Plan

**Plan complete.** Ready for execution via `superpowers:executing-plans`.
