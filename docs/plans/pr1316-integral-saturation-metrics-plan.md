# Integral-Based Window Metrics for Burst-Robust Saturation Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace point-sampled boundary metrics with time-weighted integral metrics to detect bursty workload saturation that occurs within measurement windows

**The problem today:** The backlog-drift saturation analyzer samples `ActiveEnd` — the in-flight request count at the exact window boundary instant. This point-sampling misses bursty workloads where congestion happens *within* a window but drains before the boundary. A 15-second burst within a 60-second window that drains before the boundary produces `ActiveEnd = 0`, making the burst invisible to the classifier.

**What this PR adds:**
1. **Time-weighted average in-flight count (`MeanInFlight`)** — Computes the integral of active requests over each window divided by window duration. A burst that lasts 10 seconds within a 60-second window contributes its proportional weight to the average.
2. **True peak within window (`PeakInFlight`)** — Tracks the maximum in-flight count at any instant within the window, not just at boundaries. Catches intra-window peaks that point-sampling misses.
3. **Burst-robust regression** — The slope regression now fits time-averaged load per window instead of instantaneous boundary samples, making it immune to burst timing and window alignment effects.

**Why this matters:** Real workloads are bursty. Point-sampling produces phantom patterns where the same workload gets different classifications depending on when you start observing. Integral metrics provide stable, physics-accurate saturation detection that matches what users actually experience.

**Architecture:** Modifies `computeWindowMetrics` in `sim/workload/saturation.go` to compute integral and peak metrics using event transitions. Adds two fields to `WindowMetrics` struct. Updates regression and classification logic to use `MeanInFlight` instead of `ActiveEnd`. All changes are internal to the saturation analyzer — no CLI changes, no new flags.

**Source:** GitHub issue #1316

**Closes:** Fixes #1316

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR enhances the backlog-drift saturation analyzer (introduced in PR #1310) by replacing point-sampled boundary metrics with time-weighted integral metrics. Currently, the analyzer samples `ActiveEnd` — the in-flight count at window boundaries — which misses bursts that occur within windows but drain before boundaries. This creates unstable classifications that depend on observation timing.

The PR computes two new per-window metrics:
- `MeanInFlight`: time-weighted average of active requests (integral / duration)
- `PeakInFlight`: true maximum active count within the window

These metrics feed into the existing regression and peak/mean classification logic, making saturation detection burst-robust and deterministic regardless of window alignment.

**Where it fits:** Extends the saturation analyzer in `sim/workload/` (post-hoc analysis, not part of the DES engine). Depends on the `RequestInterval` and `WindowMetrics` types introduced in PR #1310. The `AnalyzeBacklogDrift` orchestration function remains unchanged — only the internal metric computation and usage change.

**Adjacent blocks:** Used by `blis run`, `blis replay`, and `blis calibrate` commands via the `--saturation-output` flag. Produces `BacklogDriftReport` JSON files consumed by capacity planning tools.

**No deviations flagged:** Issue #1316 is unambiguous and matches the current codebase structure. PR #1310 has merged, providing the foundation this PR builds on.

### B) Behavioral Contracts (Phase 1)

#### Positive Contracts

**BC-1: Time-Weighted Mean Correctness**
- GIVEN a window [startUs, endUs) with known request arrivals and departures
- WHEN computing `MeanInFlight`
- THEN `MeanInFlight` MUST equal the time-weighted integral of active requests divided by (endUs - startUs)
- MECHANISM: Walk sorted transition events (arrivals +1, departures -1), accumulate area = count × Δt for each segment

**BC-2: Peak Within Window**
- GIVEN a window [startUs, endUs) with requests active at various times
- WHEN computing `PeakInFlight`
- THEN `PeakInFlight` MUST be >= `max(ActiveStart, ActiveEnd)`
- MECHANISM: Track max count while walking transition events

**BC-3: Steady-State Equivalence**
- GIVEN a window where all requests are active for the entire duration (constant count C)
- WHEN computing integral metrics
- THEN `MeanInFlight` MUST equal C and `PeakInFlight` MUST equal C
- MECHANISM: Constant count produces area = C × duration, so mean = area / duration = C

**BC-4: Burst Detection**
- GIVEN a burst that starts and ends within a single window
- WHEN `ActiveStart == 0` and `ActiveEnd == 0` (burst invisible to boundary sampling)
- THEN `PeakInFlight > 0` and `MeanInFlight > 0`
- MECHANISM: Transition events within the window update peak and contribute to integral

**BC-5: Regression Uses Time-Averaged Load**
- GIVEN windows with computed `MeanInFlight` values
- WHEN fitting slope regression
- THEN samples MUST use `int(math.Round(w.MeanInFlight))` instead of `w.ActiveEnd`
- MECHANISM: Direct substitution in `AnalyzeBacklogDrift` sample preparation loop

**BC-6: Peak/Mean Uses Intra-Window Peak**
- GIVEN windows with computed `PeakInFlight` values
- WHEN computing peak-to-mean ratio for transient classification
- THEN MUST use `max(w.PeakInFlight)` across all windows, not `max(w.ActiveEnd)`
- MECHANISM: Replace ActiveEnd with PeakInFlight in classification logic

**BC-7: Backward Compatibility (JSON Output)**
- GIVEN existing tools parsing `BacklogDriftReport` JSON
- WHEN adding `MeanInFlight` and `PeakInFlight` fields to `WindowMetrics`
- THEN existing fields MUST remain unchanged and parsers MUST NOT break
- MECHANISM: Additive changes only — no field removal or type changes

#### Negative Contracts

**BC-8: No Arithmetic Overflow**
- GIVEN realistic window durations and request counts
- WHEN computing integral (area += count × Δt)
- THEN MUST NOT overflow int64 or produce NaN/Inf
- MECHANISM: Use int64 for timestamps and intermediate products; time deltas are < 7 days per existing guard

**BC-9: No Event Leakage**
- GIVEN events occurring outside [startUs, endUs)
- WHEN collecting transition events
- THEN events outside the window MUST NOT affect MeanInFlight or PeakInFlight
- MECHANISM: Filter: `startUs <= event.time < endUs` for arrivals/departures

#### Error Handling Contracts

**BC-10: Empty Window Behavior**
- GIVEN a window with no arrivals or departures (but may have ActiveStart carryover)
- WHEN computing metrics
- THEN `MeanInFlight` MUST equal `ActiveStart` (constant throughout) and `PeakInFlight` MUST equal `ActiveStart`
- MECHANISM: If no events, area = ActiveStart × (endUs - startUs), peak = ActiveStart

**BC-11: Zero-Duration Window Degenerate Case**
- GIVEN a window with startUs == endUs (degenerate, should not occur in practice)
- WHEN computing metrics
- THEN MUST NOT divide by zero; MeanInFlight can default to 0 or NaN per existing NaN-handling policy
- MECHANISM: Guard: if duration == 0, skip integral computation (existing code already prevents this in computeWindowMetrics caller)

### C) Component Interaction (Phase 2)

#### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ AnalyzeBacklogDrift (orchestrator, unchanged)                   │
│   - Calls RequestsToIntervals (eligibility filter)              │
│   - Calls computeWindowMetrics (MODIFIED)                       │
│   - Prepares regression samples (MODIFIED: uses MeanInFlight)   │
│   - Calls fitSlopeRegression (unchanged)                        │
│   - Calls classifyBacklogDrift (MODIFIED: uses PeakInFlight)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ computeWindowMetrics (internal function, MODIFIED)              │
│   Input: []RequestInterval, windowSizeUs, totalDurationUs       │
│   Output: []WindowMetrics (with new MeanInFlight, PeakInFlight) │
│   Computes per-window:                                           │
│     - Existing: NumEntered, NumLeft, ActiveStart, ActiveEnd      │
│     - NEW: MeanInFlight (time-weighted avg), PeakInFlight (max)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ WindowMetrics struct (MODIFIED: +2 fields)                      │
│   - Existing fields: Start, End, NumEntered, NumLeft, etc.      │
│   - NEW: MeanInFlight float64, PeakInFlight int                 │
└─────────────────────────────────────────────────────────────────┘
```

#### API Contracts

1. **computeWindowMetrics signature (unchanged)**
   - `func computeWindowMetrics(intervals []RequestInterval, windowSizeUs, totalDurationUs int64) []WindowMetrics`
   - Precondition: `len(intervals) > 0`, `windowSizeUs > 0`, `totalDurationUs > 0`
   - Postcondition: Returns one WindowMetrics per complete window; each has MeanInFlight and PeakInFlight computed
   - Failure mode: Returns `[]WindowMetrics{}` if totalDurationUs > 7 days (existing guard)

2. **WindowMetrics struct (additive change)**
   - Existing fields preserved
   - New fields: `MeanInFlight float64`, `PeakInFlight int`
   - Both fields always populated (never left at zero-value unintentionally)

3. **BacklogDriftReport struct (unchanged at top level)**
   - `Windows []WindowMetrics` now includes new fields per window
   - JSON marshaling: new fields appear in output (backward-compatible addition)

#### State Changes

- **New state:** Each `WindowMetrics` now owns `MeanInFlight` and `PeakInFlight` in addition to existing fields
- **State lifecycle:** Computed once per window in `computeWindowMetrics`, never mutated afterward
- **Accessed by:** `AnalyzeBacklogDrift` (for regression samples), `classifyBacklogDrift` (for peak/mean ratio), JSON serialization

#### Extension Friction Assessment

- **Files changed:** 2 (saturation.go, saturation_test.go)
- **To add one more window metric:** Modify `computeWindowMetrics` (1 location), update `WindowMetrics` struct (1 location), write tests (1 file). **Friction: Low (3 touches)**
- **To change regression input:** Modify sample preparation loop in `AnalyzeBacklogDrift` (1 location). **Friction: Low (1 touch)**

This is acceptable. The saturation analyzer is a self-contained module with clear extension points.

### D) Deviation Log (Phase 3)

No deviations from source document. Issue #1316 provides:
- Complete algorithm (step-by-step event-walking integral computation)
- Exact struct changes (`MeanInFlight float64`, `PeakInFlight int`)
- Usage updates (regression uses MeanInFlight, classification uses PeakInFlight)
- Test cases (4 behavioral scenarios with expected values)

All guidance is directly implementable without interpretation.

### E) Review Guide (Phase 7-B)

#### The Tricky Part

The integral computation must correctly handle:
1. **Initial segment:** Area from `startUs` to first event uses `ActiveStart` (carried from previous window)
2. **Final segment:** Area from last event to `endUs` uses final `currentCount`
3. **Event ordering:** Arrivals and departures at the same timestamp (stable sort: arrivals before departures)

The most subtle invariant: `MeanInFlight` must equal `ActiveStart` for empty windows (no events). This requires the final segment logic to correctly handle "no events" case.

#### What to Scrutinize

1. **BC-1 (integral correctness):** Verify test case with known geometry (e.g., 1 request active for full window → MeanInFlight == 1.0)
2. **BC-4 (burst detection):** Verify sub-window burst produces PeakInFlight > 0 even when ActiveStart == ActiveEnd == 0
3. **BC-10 (empty window):** Verify window with no arrivals/departures computes MeanInFlight == ActiveStart

#### What's Safe to Skim

1. Type definitions (straightforward field additions)
2. Regression sample preparation (one-line change: ActiveEnd → MeanInFlight)
3. Test table setup (repetitive struct initialization)

#### Known Debt

None. This PR is self-contained. The only pre-existing issue is that PR #1310's fix for `simEndUs + 1` boundary handling (issue #1298) is already merged — no conflict.

---

## PART 2: Executable Implementation

### F) Implementation Overview (Phase 4 summary)

**Files to modify:**
- `sim/workload/saturation.go`: Add fields to `WindowMetrics`, rewrite integral computation in `computeWindowMetrics`, update regression sample prep and classification to use new fields
- `sim/workload/saturation_test.go`: Add 5 new tests for integral computation (BC-1, BC-2, BC-3, BC-4, BC-10), update existing end-to-end tests to verify new fields populated

**Key decisions:**
1. Event-transition algorithm: Sort arrivals/departures, walk with running count, accumulate area
2. Stable sort for same-timestamp events: arrivals before departures (existing Go sort is stable)
3. Reuse existing `RequestInterval` struct — no new types needed
4. No changes to function signatures (internal implementation only)

**Dead code confirmation:** All new fields (`MeanInFlight`, `PeakInFlight`) are used:
- `MeanInFlight`: Used in regression sample preparation (line ~424 in saturation.go)
- `PeakInFlight`: Used in peak-to-mean ratio calculation (line ~433 in saturation.go)

Both fields are exercised by tests and visible in JSON output.

### G) Task Breakdown (Phase 4 detailed)

### Task 1: Add MeanInFlight and PeakInFlight Fields to WindowMetrics

**Contracts Implemented:** BC-7 (backward compatibility)

**Files:**
- Modify: `sim/workload/saturation.go:94-104` (WindowMetrics struct)

**Step 1: Write failing test for new fields presence**

Context: Verify that `WindowMetrics` includes the new fields and they can be set/read.

```go
// In sim/workload/saturation_test.go, add after TestWriteBacklogDriftReportJSON_SanitizesNaN (line ~585)

func TestWindowMetrics_NewFields_Accessible(t *testing.T) {
	// GIVEN a WindowMetrics struct
	// WHEN setting MeanInFlight and PeakInFlight
	// THEN fields are accessible and retain their values
	w := WindowMetrics{
		StartUs:      0,
		EndUs:        60_000_000,
		NumEntered:   10,
		NumLeft:      8,
		ActiveStart:  5,
		ActiveEnd:    7,
		DeltaBacklog: 2,
		DrainRatio:   0.8,
		MeanInFlight: 6.5,
		PeakInFlight: 12,
	}

	if w.MeanInFlight != 6.5 {
		t.Errorf("MeanInFlight: expected 6.5, got %f", w.MeanInFlight)
	}
	if w.PeakInFlight != 12 {
		t.Errorf("PeakInFlight: expected 12, got %d", w.PeakInFlight)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestWindowMetrics_NewFields -v`
Expected: Compilation error: "unknown field MeanInFlight in struct literal"

**Step 3: Add new fields to WindowMetrics struct**

Context: Extend WindowMetrics with integral-based metrics.

In `sim/workload/saturation.go`, modify `WindowMetrics` struct (line ~94):

```go
// WindowMetrics captures per-window saturation metrics (BC-1).
type WindowMetrics struct {
	StartUs      int64   // Window start timestamp (µs)
	EndUs        int64   // Window end timestamp (µs)
	NumEntered   int     // Requests with arrival in [start, end)
	NumLeft      int     // Requests with completion in [start, end)
	ActiveStart  int     // Active requests at window start
	ActiveEnd    int     // Active requests at window end
	DeltaBacklog int     // ActiveEnd - ActiveStart (change in backlog over window, BC-1)
	DrainRatio   float64 // NumLeft / NumEntered (NaN if NumEntered==0)
	MeanInFlight float64 // Time-weighted average in-flight count over the window (BC-1)
	PeakInFlight int     // Maximum in-flight count at any instant within the window (BC-2)
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestWindowMetrics_NewFields -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): add MeanInFlight and PeakInFlight to WindowMetrics (BC-7)

- Add MeanInFlight float64 field for time-weighted average in-flight count
- Add PeakInFlight int field for true intra-window peak
- Backward-compatible addition to WindowMetrics struct (BC-7)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Implement Integral Computation in computeWindowMetrics

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-8, BC-9, BC-10

**Files:**
- Modify: `sim/workload/saturation.go:172-235` (computeWindowMetrics function)
- Test: `sim/workload/saturation_test.go`

**Step 1: Write failing test for constant-load case (BC-3)**

Context: Verify integral correctness for the simplest case: constant load throughout window.

```go
// In sim/workload/saturation_test.go, add after TestWindowMetrics_NewFields_Accessible

func TestComputeWindowMetrics_ConstantLoad_IntegralCorrect(t *testing.T) {
	// GIVEN 10 requests active for the entire window [0, 60s)
	// WHEN computing window metrics
	// THEN MeanInFlight == 10.0 and PeakInFlight == 10 (BC-3)
	intervals := []RequestInterval{
		{ArrivalUs: -10_000, CompletionUs: 70_000_000},   // Request 1: active throughout
		{ArrivalUs: -9_000, CompletionUs: 70_000_000},    // Request 2: active throughout
		{ArrivalUs: -8_000, CompletionUs: 70_000_000},    // Request 3
		{ArrivalUs: -7_000, CompletionUs: 70_000_000},    // Request 4
		{ArrivalUs: -6_000, CompletionUs: 70_000_000},    // Request 5
		{ArrivalUs: -5_000, CompletionUs: 70_000_000},    // Request 6
		{ArrivalUs: -4_000, CompletionUs: 70_000_000},    // Request 7
		{ArrivalUs: -3_000, CompletionUs: 70_000_000},    // Request 8
		{ArrivalUs: -2_000, CompletionUs: 70_000_000},    // Request 9
		{ArrivalUs: -1_000, CompletionUs: 70_000_000},    // Request 10
	}
	windowSizeUs := int64(60_000_000) // 60 seconds
	totalDurationUs := int64(60_000_000)

	windows := computeWindowMetrics(intervals, windowSizeUs, totalDurationUs)

	if len(windows) != 1 {
		t.Fatalf("Expected 1 window, got %d", len(windows))
	}

	w := windows[0]
	const expectedMean = 10.0
	const expectedPeak = 10

	if math.Abs(w.MeanInFlight-expectedMean) > 0.01 {
		t.Errorf("MeanInFlight: expected %.2f, got %.2f", expectedMean, w.MeanInFlight)
	}
	if w.PeakInFlight != expectedPeak {
		t.Errorf("PeakInFlight: expected %d, got %d", expectedPeak, w.PeakInFlight)
	}
	// Also verify continuity with existing fields
	if w.ActiveStart != 10 || w.ActiveEnd != 10 {
		t.Errorf("ActiveStart/ActiveEnd should be 10, got %d/%d", w.ActiveStart, w.ActiveEnd)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestComputeWindowMetrics_ConstantLoad -v`
Expected: FAIL (MeanInFlight == 0.0, PeakInFlight == 0 because not yet implemented)

**Step 3: Implement integral computation algorithm**

Context: Rewrite `computeWindowMetrics` to compute MeanInFlight and PeakInFlight using event-transition algorithm per issue #1316.

In `sim/workload/saturation.go`, replace the `computeWindowMetrics` function (lines 172-235):

```go
// computeWindowMetrics computes per-window saturation metrics per BC-1.
// Returns one WindowMetrics entry per complete window of size windowSizeUs.
// Computes DeltaBacklog = ActiveEnd - ActiveStart (change in backlog over window).
// Computes MeanInFlight (time-weighted average) and PeakInFlight (true peak within window).
func computeWindowMetrics(intervals []RequestInterval, windowSizeUs, totalDurationUs int64) []WindowMetrics {
	if len(intervals) == 0 || totalDurationUs <= 0 || windowSizeUs <= 0 {
		return []WindowMetrics{}
	}

	// Guard against unreasonably large durations (e.g., MaxInt64 when no workload specified)
	// Cap at 7 days = 604800 seconds = 604800000000 microseconds
	const maxReasonableDurationUs int64 = 604800 * 1e6
	if totalDurationUs > maxReasonableDurationUs {
		logrus.Warnf("Saturation analysis: totalDurationUs (%d) exceeds reasonable limit (%d us = 7 days), skipping window metrics", totalDurationUs, maxReasonableDurationUs)
		return []WindowMetrics{}
	}

	numWindows := int((totalDurationUs + windowSizeUs - 1) / windowSizeUs) // Ceiling division
	windows := make([]WindowMetrics, numWindows)

	for i := 0; i < numWindows; i++ {
		startUs := int64(i) * windowSizeUs
		endUs := startUs + windowSizeUs
		if endUs > totalDurationUs {
			endUs = totalDurationUs
		}

		w := WindowMetrics{
			StartUs: startUs,
			EndUs:   endUs,
		}

		// Compute existing metrics by scanning all intervals
		for _, iv := range intervals {
			// NumEntered: arrival in [startUs, endUs)
			if iv.ArrivalUs >= startUs && iv.ArrivalUs < endUs {
				w.NumEntered++
			}
			// NumLeft: completion in [startUs, endUs)
			if iv.CompletionUs >= startUs && iv.CompletionUs < endUs {
				w.NumLeft++
			}
			// ActiveStart: interval contains startUs (arrival <= startUs < completion)
			if iv.ArrivalUs <= startUs && startUs < iv.CompletionUs {
				w.ActiveStart++
			}
			// ActiveEnd: interval contains endUs
			if iv.ArrivalUs <= endUs && endUs < iv.CompletionUs {
				w.ActiveEnd++
			}
		}

		// Compute integral-based metrics (BC-1, BC-2)
		// Step 1: Collect transition events within [startUs, endUs)
		type event struct {
			timeUs int64
			delta  int // +1 for arrival, -1 for departure
		}
		events := make([]event, 0, len(intervals)*2)
		for _, iv := range intervals {
			// Arrival event
			if iv.ArrivalUs >= startUs && iv.ArrivalUs < endUs {
				events = append(events, event{timeUs: iv.ArrivalUs, delta: +1})
			}
			// Departure event
			if iv.CompletionUs >= startUs && iv.CompletionUs < endUs {
				events = append(events, event{timeUs: iv.CompletionUs, delta: -1})
			}
		}

		// Step 2: Sort events by time (Go's sort.Slice is stable, so arrivals before departures at same time)
		sort.Slice(events, func(i, j int) bool {
			if events[i].timeUs != events[j].timeUs {
				return events[i].timeUs < events[j].timeUs
			}
			// Stable sort ensures original order preserved for same timestamp
			// If needed, explicit tiebreaker: arrivals (+1) before departures (-1)
			return events[i].delta > events[j].delta
		})

		// Step 3: Walk events to compute integral and peak
		currentCount := w.ActiveStart // Requests active at window start
		area := int64(0)
		peakCount := currentCount
		prevTimeUs := startUs

		for _, ev := range events {
			// Accumulate area of rectangle [prevTimeUs, ev.timeUs) at height currentCount
			deltaTime := ev.timeUs - prevTimeUs
			area += int64(currentCount) * deltaTime
			if currentCount > peakCount {
				peakCount = currentCount
			}

			// Update count
			currentCount += ev.delta
			if currentCount > peakCount {
				peakCount = currentCount
			}

			prevTimeUs = ev.timeUs
		}

		// Final segment [lastEvent, endUs) at height currentCount
		deltaTime := endUs - prevTimeUs
		area += int64(currentCount) * deltaTime
		if currentCount > peakCount {
			peakCount = currentCount
		}

		// Step 4: Compute derived metrics
		windowDuration := endUs - startUs
		if windowDuration > 0 {
			w.MeanInFlight = float64(area) / float64(windowDuration)
		} else {
			// Degenerate case (should not occur): zero-duration window
			w.MeanInFlight = 0.0
		}
		w.PeakInFlight = peakCount

		// DeltaBacklog and DrainRatio (unchanged)
		w.DeltaBacklog = w.ActiveEnd - w.ActiveStart
		if w.NumEntered > 0 {
			w.DrainRatio = float64(w.NumLeft) / float64(w.NumEntered)
		} else {
			w.DrainRatio = math.NaN() // Undefined when no arrivals
		}

		windows[i] = w
	}

	return windows
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestComputeWindowMetrics_ConstantLoad -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): compute MeanInFlight and PeakInFlight via event transitions (BC-1, BC-2, BC-3)

- Implement event-transition algorithm for integral computation
- Walk sorted arrivals/departures, accumulate area = count × Δt per segment
- Track peak across all segments
- Verify constant-load case: 10 requests → MeanInFlight=10.0, PeakInFlight=10

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Test Burst Detection (Sub-Window Burst)

**Contracts Implemented:** BC-4

**Files:**
- Test: `sim/workload/saturation_test.go`

**Step 1: Write failing test for sub-window burst (BC-4)**

Context: Verify that a burst occurring entirely within a window is detected even when ActiveStart == ActiveEnd == 0.

```go
// In sim/workload/saturation_test.go, add after TestComputeWindowMetrics_ConstantLoad_IntegralCorrect

func TestComputeWindowMetrics_SubWindowBurst_Detected(t *testing.T) {
	// GIVEN 100 requests that arrive at t=20s and complete at t=40s
	// WHEN window is [0, 60s)
	// THEN ActiveStart=0, ActiveEnd=0 (burst invisible to boundaries)
	//      BUT MeanInFlight ≈ 33.3 and PeakInFlight=100 (BC-4)
	intervals := make([]RequestInterval, 100)
	for i := 0; i < 100; i++ {
		intervals[i] = RequestInterval{
			ArrivalUs:    20_000_000, // All arrive at t=20s
			CompletionUs: 40_000_000, // All complete at t=40s
		}
	}
	windowSizeUs := int64(60_000_000) // 60 seconds
	totalDurationUs := int64(60_000_000)

	windows := computeWindowMetrics(intervals, windowSizeUs, totalDurationUs)

	if len(windows) != 1 {
		t.Fatalf("Expected 1 window, got %d", len(windows))
	}

	w := windows[0]

	// Verify boundary counts are zero (burst not visible at boundaries)
	if w.ActiveStart != 0 || w.ActiveEnd != 0 {
		t.Errorf("ActiveStart/ActiveEnd should be 0 (boundary-invisible burst), got %d/%d", w.ActiveStart, w.ActiveEnd)
	}

	// Verify burst IS visible via integral metrics
	const expectedPeak = 100
	expectedMean := 100.0 * 20.0 / 60.0 // 100 requests × 20s / 60s ≈ 33.33

	if w.PeakInFlight != expectedPeak {
		t.Errorf("PeakInFlight: expected %d, got %d", expectedPeak, w.PeakInFlight)
	}
	if math.Abs(w.MeanInFlight-expectedMean) > 0.1 {
		t.Errorf("MeanInFlight: expected %.2f, got %.2f", expectedMean, w.MeanInFlight)
	}
}
```

**Step 2: Run test to verify it fails (if not already passing from Task 2)**

Run: `go test ./sim/workload/... -run TestComputeWindowMetrics_SubWindowBurst -v`
Expected: PASS (implementation from Task 2 should handle this)

**Step 3: No implementation needed (covered by Task 2)**

The event-transition algorithm from Task 2 already handles this case.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestComputeWindowMetrics_SubWindowBurst -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/saturation_test.go
git commit -m "test(workload): verify sub-window burst detection (BC-4)

- Test burst that occurs entirely within window (t=20-40s in 0-60s window)
- Verify ActiveStart=0, ActiveEnd=0 (boundary-invisible)
- Verify PeakInFlight=100, MeanInFlight≈33.3 (burst detected via integral)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Test Empty Window Case (No Events)

**Contracts Implemented:** BC-10

**Files:**
- Test: `sim/workload/saturation_test.go`

**Step 1: Write failing test for empty window with ActiveStart carryover**

Context: Verify that a window with no arrivals or departures correctly computes MeanInFlight == ActiveStart.

```go
// In sim/workload/saturation_test.go, add after TestComputeWindowMetrics_SubWindowBurst_Detected

func TestComputeWindowMetrics_EmptyWindow_UsesActiveStart(t *testing.T) {
	// GIVEN 5 requests active at window start, no arrivals or departures within window
	// WHEN computing window metrics
	// THEN MeanInFlight == 5.0 and PeakInFlight == 5 (BC-10)
	intervals := []RequestInterval{
		// All 5 requests: arrive before window, complete after window
		{ArrivalUs: -10_000, CompletionUs: 70_000_000},
		{ArrivalUs: -9_000, CompletionUs: 70_000_000},
		{ArrivalUs: -8_000, CompletionUs: 70_000_000},
		{ArrivalUs: -7_000, CompletionUs: 70_000_000},
		{ArrivalUs: -6_000, CompletionUs: 70_000_000},
	}
	windowSizeUs := int64(60_000_000) // 60 seconds
	totalDurationUs := int64(60_000_000)

	windows := computeWindowMetrics(intervals, windowSizeUs, totalDurationUs)

	if len(windows) != 1 {
		t.Fatalf("Expected 1 window, got %d", len(windows))
	}

	w := windows[0]

	// Verify no events within window
	if w.NumEntered != 0 || w.NumLeft != 0 {
		t.Errorf("NumEntered/NumLeft should be 0 (no events in window), got %d/%d", w.NumEntered, w.NumLeft)
	}

	// Verify integral metrics use ActiveStart
	const expectedMean = 5.0
	const expectedPeak = 5

	if math.Abs(w.MeanInFlight-expectedMean) > 0.01 {
		t.Errorf("MeanInFlight: expected %.2f, got %.2f (should equal ActiveStart)", expectedMean, w.MeanInFlight)
	}
	if w.PeakInFlight != expectedPeak {
		t.Errorf("PeakInFlight: expected %d, got %d (should equal ActiveStart)", expectedPeak, w.PeakInFlight)
	}
	if w.ActiveStart != 5 {
		t.Errorf("ActiveStart: expected 5, got %d", w.ActiveStart)
	}
}
```

**Step 2: Run test to verify it fails (if not already passing)**

Run: `go test ./sim/workload/... -run TestComputeWindowMetrics_EmptyWindow -v`
Expected: PASS (Task 2 implementation handles this via "no events → area = ActiveStart × duration")

**Step 3: No implementation needed (covered by Task 2)**

The final segment logic in Task 2 handles this: if no events, prevTimeUs == startUs and currentCount == ActiveStart, so area = ActiveStart × duration.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestComputeWindowMetrics_EmptyWindow -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/saturation_test.go
git commit -m "test(workload): verify empty window uses ActiveStart for integral (BC-10)

- Test window with no arrivals or departures (5 requests active throughout)
- Verify MeanInFlight == ActiveStart (constant load, no events)
- Verify PeakInFlight == ActiveStart

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Update Regression to Use MeanInFlight

**Contracts Implemented:** BC-5

**Files:**
- Modify: `sim/workload/saturation.go:416-425` (sample preparation in AnalyzeBacklogDrift)
- Test: `sim/workload/saturation_test.go`

**Step 1: Write failing test for regression using MeanInFlight**

Context: Verify that slope regression uses MeanInFlight instead of ActiveEnd.

```go
// In sim/workload/saturation_test.go, add after TestComputeWindowMetrics_EmptyWindow_UsesActiveStart

func TestAnalyzeBacklogDrift_RegressionUsesMeanInFlight(t *testing.T) {
	// GIVEN a workload with growing backlog but where ActiveEnd samples don't reflect true load
	// WHEN analyzing with sufficient windows
	// THEN regression MUST use MeanInFlight, not ActiveEnd (BC-5)
	//
	// Strategy: Create scenario where MeanInFlight shows clear growth but ActiveEnd is noisy
	// If regression uses MeanInFlight → stable positive slope
	// If regression uses ActiveEnd → slope may be zero or negative

	cfg := NewBacklogDriftConfig(10*time.Second, 3, 2.0, 0.2, 0.95)

	// Create 40-second workload: arrivals at 20 req/s, completions staggered to create growing backlog
	var requests []*sim.Request
	for i := 0; i < 200; i++ {
		arrivalUs := int64(i) * 50_000 // 20 req/s
		completionUs := arrivalUs + 100_000 + int64(i)*10_000 // Growing service time

		var state sim.RequestState
		var ttftSet bool
		var itl []int64

		if completionUs <= 40_000_000 {
			state = sim.StateCompleted
			ttftSet = true
			itl = []int64{50_000}
		} else {
			state = sim.StateRunning
			ttftSet = false
			itl = nil
		}

		requests = append(requests, &sim.Request{
			ID:             fmt.Sprintf("req_%d", i),
			ArrivalTime:    arrivalUs,
			FirstTokenTime: 50_000,
			TTFTSet:        ttftSet,
			ITL:            itl,
			State:          state,
			InputTokens:    []int{0},
			OutputTokens:   []int{0},
		})
	}

	simEndUs := int64(40_000_000)
	report := AnalyzeBacklogDrift(requests, simEndUs, cfg)

	// Verify MeanInFlight was populated (proves it's being computed)
	if report.MeanInFlight == 0 {
		t.Fatal("MeanInFlight is 0 — not populated correctly")
	}

	// Verify slope is positive (growing backlog)
	if report.Slope <= 0 {
		t.Errorf("Expected positive slope (growing backlog), got %.6e", report.Slope)
	}

	t.Logf("Slope: %.6e (CI: [%.6e, %.6e])", report.Slope, report.SlopeLower, report.SlopeUpper)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestAnalyzeBacklogDrift_RegressionUsesMeanInFlight -v`
Expected: FAIL or unexpected behavior (regression still uses ActiveEnd)

**Step 3: Update regression sample preparation to use MeanInFlight**

Context: Modify `AnalyzeBacklogDrift` to use `w.MeanInFlight` instead of `w.ActiveEnd` for regression samples.

In `sim/workload/saturation.go`, find the sample preparation loop (around line 417-425):

```go
// Step 3: Prepare time series samples for regression
samples := make([]struct {
	timeUs int64
	count  int
}, len(windows))
for i, w := range windows {
	// Use window midpoint as time coordinate
	samples[i].timeUs = (w.StartUs + w.EndUs) / 2
	// BC-5: Use time-averaged load (MeanInFlight) instead of boundary sample (ActiveEnd)
	samples[i].count = int(math.Round(w.MeanInFlight))
}
```

Replace the line:
```go
samples[i].count = w.ActiveEnd
```
with:
```go
samples[i].count = int(math.Round(w.MeanInFlight))
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestAnalyzeBacklogDrift_RegressionUsesMeanInFlight -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): use MeanInFlight for slope regression (BC-5)

- Replace ActiveEnd with MeanInFlight in regression sample preparation
- Slope now fits time-averaged load per window (burst-robust)
- Test verifies positive slope for growing backlog scenario

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Update Peak/Mean Classification to Use PeakInFlight

**Contracts Implemented:** BC-6

**Files:**
- Modify: `sim/workload/saturation.go:430-441` (summary statistics in AnalyzeBacklogDrift)
- Test: `sim/workload/saturation_test.go`

**Step 1: Write failing test for peak/mean using PeakInFlight**

Context: Verify that TRANSIENT_BACKLOG detection uses PeakInFlight, not max(ActiveEnd).

```go
// In sim/workload/saturation_test.go, add after TestAnalyzeBacklogDrift_RegressionUsesMeanInFlight

func TestAnalyzeBacklogDrift_PeakMeanUsesPeakInFlight(t *testing.T) {
	// GIVEN a workload with a sharp intra-window peak
	// WHEN peak occurs between window boundaries (not at boundaries)
	// THEN classification MUST use PeakInFlight, not max(ActiveEnd) (BC-6)
	//
	// Strategy: Create burst in middle of window 2
	// If using PeakInFlight → TRANSIENT_BACKLOG (high peak/mean ratio)
	// If using max(ActiveEnd) → UNSATURATED (boundary samples miss peak)

	cfg := NewBacklogDriftConfig(10*time.Second, 3, 2.0, 0.2, 0.95)

	// Windows: [0-10s], [10-20s], [20-30s]
	// Burst: 60 requests arrive at t=15s, complete at t=16s (intra-window peak in window 2)
	var requests []*sim.Request

	// Window 1: 5 requests active throughout
	for i := 0; i < 5; i++ {
		requests = append(requests, &sim.Request{
			ID:             fmt.Sprintf("w1_%d", i),
			ArrivalTime:    int64(i * 1_000_000), // 0-5s
			FirstTokenTime: 5_000_000,
			TTFTSet:        true,
			ITL:            []int64{25_000_000}, // Complete at 30-35s (window 3)
			State:          sim.StateCompleted,
			InputTokens:    []int{0},
			OutputTokens:   []int{0},
		})
	}

	// Window 2: Sharp burst at t=15s (mid-window)
	for i := 0; i < 60; i++ {
		requests = append(requests, &sim.Request{
			ID:             fmt.Sprintf("burst_%d", i),
			ArrivalTime:    15_000_000,                     // All at t=15s
			FirstTokenTime: 500_000,                        // 0.5s TTFT
			TTFTSet:        true,
			ITL:            []int64{500_000},               // Complete at t=16s
			State:          sim.StateCompleted,
			InputTokens:    []int{0},
			OutputTokens:   []int{0},
		})
	}

	// Window 3: Same 5 requests still active
	// (No new requests needed — carryover from window 1)

	simEndUs := int64(30_000_000) // 30 seconds (3 windows)
	report := AnalyzeBacklogDrift(requests, simEndUs, cfg)

	// At t=10s (boundary): 5 active
	// At t=15s (mid-window 2): 5 + 60 = 65 active (PEAK)
	// At t=16s: back to 5 active
	// At t=20s (boundary): 5 active
	// At t=30s (boundary): 0 active (all completed)

	// Expected: PeakInFlight = 65 (from window 2)
	//           MeanInFlight across windows ≈ (5 + (5+60*0.1) + 0) / 3 ≈ 3.7
	//           Peak/Mean ≈ 65 / 3.7 ≈ 17.6 >> 2.0 → TRANSIENT_BACKLOG

	if report.PeakInFlight < 60 {
		t.Errorf("PeakInFlight: expected >= 60 (burst peak), got %d", report.PeakInFlight)
	}

	if report.Classification != "TRANSIENT_BACKLOG" {
		peakRatio := 0.0
		if report.MeanInFlight > 0 {
			peakRatio = float64(report.PeakInFlight) / report.MeanInFlight
		}
		t.Errorf("Expected TRANSIENT_BACKLOG (high intra-window peak), got %s\nPeak=%d, Mean=%.2f, Ratio=%.2f\nNote: %s",
			report.Classification, report.PeakInFlight, report.MeanInFlight, peakRatio, report.Note)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestAnalyzeBacklogDrift_PeakMeanUsesPeakInFlight -v`
Expected: FAIL (classification may be UNSATURATED if using max(ActiveEnd) instead of PeakInFlight)

**Step 3: Update peak/mean computation to use PeakInFlight**

Context: Modify `AnalyzeBacklogDrift` to compute peak across `w.PeakInFlight` instead of `w.ActiveEnd`.

In `sim/workload/saturation.go`, find the summary statistics computation (around line 430-441):

```go
// Step 5: Compute summary statistics
initialBacklog := windows[0].ActiveStart
finalBacklog := windows[len(windows)-1].ActiveEnd
peakInFlight := 0
var sumInFlight float64
for _, w := range windows {
	// BC-6: Use true intra-window peak, not boundary sample
	if w.PeakInFlight > peakInFlight {
		peakInFlight = w.PeakInFlight
	}
	// BC-6: Use time-weighted mean for overall average
	sumInFlight += w.MeanInFlight
}
meanInFlight := sumInFlight / float64(len(windows))
```

Replace the lines:
```go
peakInFlight := 0
sumInFlight := 0
for _, w := range windows {
	if w.ActiveEnd > peakInFlight {
		peakInFlight = w.ActiveEnd
	}
	sumInFlight += w.ActiveEnd
}
meanInFlight := float64(sumInFlight) / float64(len(windows))
```

with the code block above.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestAnalyzeBacklogDrift_PeakMeanUsesPeakInFlight -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/saturation.go sim/workload/saturation_test.go
git commit -m "feat(workload): use PeakInFlight for transient classification (BC-6)

- Replace max(ActiveEnd) with max(PeakInFlight) for peak detection
- Use mean(MeanInFlight) for overall average (weighted by window duration)
- Test verifies TRANSIENT_BACKLOG for intra-window burst (peak at t=15s in 10-20s window)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Update Existing Tests to Verify New Fields Populated

**Contracts Implemented:** BC-1, BC-2 (verification), BC-7 (backward compatibility)

**Files:**
- Modify: `sim/workload/saturation_test.go` (existing end-to-end tests)

**Step 1: Write test verifier for new fields in existing tests**

Context: Extend existing end-to-end tests to assert that MeanInFlight and PeakInFlight are populated and satisfy basic invariants.

```go
// In sim/workload/saturation_test.go, add helper function before TestAnalyzeBacklogDrift_InsufficientData (around line 360)

// verifyIntegralMetricsPopulated checks that MeanInFlight and PeakInFlight are computed
// and satisfy basic invariants (BC-1, BC-2).
func verifyIntegralMetricsPopulated(t *testing.T, report BacklogDriftReport) {
	t.Helper()

	for i, w := range report.Windows {
		// BC-2: PeakInFlight >= max(ActiveStart, ActiveEnd)
		minBoundary := w.ActiveStart
		if w.ActiveEnd > minBoundary {
			minBoundary = w.ActiveEnd
		}
		if w.PeakInFlight < minBoundary {
			t.Errorf("Window %d: PeakInFlight (%d) < max(ActiveStart=%d, ActiveEnd=%d) — violates BC-2",
				i, w.PeakInFlight, w.ActiveStart, w.ActiveEnd)
		}

		// MeanInFlight should be non-negative (negative would indicate a bug)
		if w.MeanInFlight < 0 {
			t.Errorf("Window %d: MeanInFlight (%.2f) < 0 — invalid", i, w.MeanInFlight)
		}

		// Peak should be >= Mean (for non-zero windows)
		if w.MeanInFlight > 0 && float64(w.PeakInFlight) < w.MeanInFlight {
			t.Errorf("Window %d: PeakInFlight (%d) < MeanInFlight (%.2f) — violates peak definition",
				i, w.PeakInFlight, w.MeanInFlight)
		}
	}
}
```

**Step 2: Run test to verify helper compiles**

Run: `go test ./sim/workload/... -run TestAnalyzeBacklogDrift_InsufficientData -v`
Expected: PASS (no changes to behavior, just added helper)

**Step 3: Add verifier calls to existing end-to-end tests**

Context: Modify `TestAnalyzeBacklogDrift_EndToEnd_PERSISTENTLY_SATURATED` and `TestAnalyzeBacklogDrift_EndToEnd_UNSATURATED` to call the verifier.

In `sim/workload/saturation_test.go`, find `TestAnalyzeBacklogDrift_EndToEnd_PERSISTENTLY_SATURATED` (around line 415):

Add after the classification assertion (line ~431):
```go
// Verify new integral metrics are populated and satisfy invariants
verifyIntegralMetricsPopulated(t, report)
```

Similarly, find `TestAnalyzeBacklogDrift_EndToEnd_UNSATURATED` (around line 447):

Add after the classification assertion (line ~469):
```go
// Verify new integral metrics are populated and satisfy invariants
verifyIntegralMetricsPopulated(t, report)
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run TestAnalyzeBacklogDrift_EndToEnd -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/saturation_test.go
git commit -m "test(workload): verify integral metrics in end-to-end tests (BC-1, BC-2)

- Add verifyIntegralMetricsPopulated helper (checks PeakInFlight >= boundaries)
- Extend PERSISTENTLY_SATURATED and UNSATURATED tests to verify new fields
- Ensure backward compatibility (BC-7) — existing tests still pass

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Run Full Test Suite and Verify No Regressions

**Contracts Implemented:** All contracts (integration verification)

**Files:**
- No code changes (verification only)

**Step 1: Run complete test suite**

Run: `go test ./sim/workload/... -v -count=1`
Expected: All tests PASS

**Step 2: Run long-running tests (if not in short mode)**

Run: `go test ./sim/workload/... -v -count=1 -timeout 5m`
Expected: All tests PASS (including TestSaturationProgression_RealWorkloads)

**Step 3: Run lint on entire package**

Run: `golangci-lint run ./sim/workload/...`
Expected: Zero new issues

**Step 4: Verify build succeeds**

Run: `go build ./...`
Expected: Build succeeds with no errors

**Step 5: Check git status**

Run: `git status`
Expected: No uncommitted changes (all work committed in previous tasks)

**Step 6: Create summary (no commit needed)**

No commit — this task is verification only. All code committed in Tasks 1-7.

---

### H) Test Strategy (Phase 6)

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 (integral correctness) | Task 2 | Unit | `TestComputeWindowMetrics_ConstantLoad_IntegralCorrect` — 10 requests active throughout → MeanInFlight=10.0 |
| BC-2 (peak >= boundaries) | Task 7 | Invariant | `verifyIntegralMetricsPopulated` — checks PeakInFlight >= max(ActiveStart, ActiveEnd) for all windows |
| BC-3 (steady-state equivalence) | Task 2 | Unit | `TestComputeWindowMetrics_ConstantLoad_IntegralCorrect` — constant count C → MeanInFlight=C, PeakInFlight=C |
| BC-4 (burst detection) | Task 3 | Unit | `TestComputeWindowMetrics_SubWindowBurst_Detected` — burst at t=20-40s in 0-60s window → PeakInFlight=100, MeanInFlight≈33.3 |
| BC-5 (regression uses MeanInFlight) | Task 5 | Integration | `TestAnalyzeBacklogDrift_RegressionUsesMeanInFlight` — growing backlog → positive slope using MeanInFlight |
| BC-6 (peak/mean uses PeakInFlight) | Task 6 | Integration | `TestAnalyzeBacklogDrift_PeakMeanUsesPeakInFlight` — intra-window peak → TRANSIENT_BACKLOG via PeakInFlight |
| BC-7 (backward compatibility) | Task 1, 7 | Regression | Existing end-to-end tests still pass after struct changes |
| BC-8 (no overflow) | All | Implicit | Existing guard: totalDurationUs capped at 7 days; int64 timestamps sufficient for all realistic workloads |
| BC-9 (no event leakage) | Task 2 | Unit | Event filtering in `computeWindowMetrics`: `startUs <= time < endUs` |
| BC-10 (empty window) | Task 4 | Unit | `TestComputeWindowMetrics_EmptyWindow_UsesActiveStart` — no events → MeanInFlight=ActiveStart |
| BC-11 (zero-duration guard) | N/A | Implicit | Existing code: computeWindowMetrics caller ensures windowSizeUs > 0 |

#### Additional Requirements

1. **Shared test infrastructure:** Uses existing `RequestInterval`, `WindowMetrics`, `BacklogDriftReport` types from PR #1310. No new test helpers needed beyond `verifyIntegralMetricsPopulated`.

2. **Golden dataset updates:** Not applicable — saturation analyzer produces separate JSON output (`--saturation-output`), not part of stdout. No golden dataset changes needed.

3. **Lint requirements:** `golangci-lint run ./sim/workload/...` must pass with zero new issues. Verified in Task 8.

4. **Test naming convention:** All new tests follow `TestType_Scenario_Behavior` pattern (e.g., `TestComputeWindowMetrics_SubWindowBurst_Detected`).

5. **Test isolation:** All tests are independently runnable. Use `RequestInterval` slices directly (no shared global state).

6. **Invariant tests alongside golden tests:** No golden tests in this PR. All tests verify behavioral contracts (time-weighted averages, peak detection) derived from mathematical definitions, not from "running the code and capturing output."

---

## PART 3: Quality Assurance

### J) Sanity Checklist (Phase 8)

**Plan-specific checks:**
- [x] No unnecessary abstractions — reuses existing `RequestInterval` struct and event-walking pattern
- [x] No feature creep — strictly implements issue #1316 algorithm, no additional metrics
- [x] No unexercised flags or interfaces — no new CLI flags, no new public interfaces
- [x] No partial implementations — both `MeanInFlight` and `PeakInFlight` fully implemented and used
- [x] No breaking changes — additive struct fields only (BC-7)
- [x] No hidden global state impact — all computation is pure (no side effects)
- [x] All new code will pass golangci-lint — verified in each task
- [x] Shared test helpers used — `verifyIntegralMetricsPopulated` added, no duplication
- [x] CLAUDE.md updated — Not needed (no new CLI flags, no file organization changes, saturation analyzer already documented)
- [x] No stale references in CLAUDE.md — N/A
- [x] Documentation DRY — Not applicable (no changes to standards docs)
- [x] Deviation log reviewed — Zero deviations (see Section D)
- [x] Each task produces working, testable code — verified per task
- [x] Task dependencies correctly ordered — Task 2 (implementation) before Tasks 3-4 (tests), Task 5-6 (usage) after Task 2
- [x] All contracts mapped to tasks — see Test Strategy (Section H)
- [x] Golden dataset regeneration — Not applicable
- [x] Construction site audit — No new struct fields added to existing structs constructed elsewhere; `WindowMetrics` is constructed only in `computeWindowMetrics`
- [x] Macro plan status update — Not applicable (no macro plan)

**Antipattern rules:**
- [x] R1: No silent continue/return — No new error paths
- [x] R2: Map keys sorted — No map iteration over floats; `events` slice is sorted explicitly
- [x] R3: Numeric parameter validation — No new numeric parameters (reuses existing `windowSizeUs`, `totalDurationUs`)
- [x] R4: Construction site audit — `WindowMetrics` constructed in one place (`computeWindowMetrics`)
- [x] R5: Resource allocation rollback — No resource allocation loops
- [x] R6: No logrus.Fatalf in sim/ — No new `logrus.Fatalf` calls
- [x] R7: Invariant tests alongside golden — No golden tests in this PR
- [x] R8: No exported mutable maps — No new maps
- [x] R9: *float64 for YAML — No new YAML fields
- [x] R10: YAML strict parsing — No YAML parsing changes
- [x] R11: Division guards — Division by `windowDuration` guarded: `if windowDuration > 0`
- [x] R12: Golden dataset regeneration — Not applicable
- [x] R13: Interfaces work for 2+ implementations — No new interfaces
- [x] R14: No multi-module methods — All changes within `sim/workload/` module
- [x] R15: Stale PR references — No PR references in comments
- [x] R16: Config params grouped — No new config params
- [x] R17: Routing scorer signals — Not applicable (no routing changes)
- [x] R18: CLI flag defaults — No CLI flag changes
- [x] R19: Unbounded retry loops — No retry loops
- [x] R20: Degenerate inputs — Empty window case tested (BC-10), zero-duration guarded (BC-11)
- [x] R21: No range over shrinking slices — No slices modified during iteration
- [x] R22: Pre-check consistency — No pre-checks
- [x] R23: Parallel transformations — No parallel code paths

---

## APPENDIX: File-Level Implementation Details

### File: `sim/workload/saturation.go`

**Purpose:** Backlog-drift saturation analyzer with integral-based burst-robust metrics.

**Complete Implementation:**

See Task 2, Step 3 for the complete `computeWindowMetrics` function implementation (lines 172-235 replacement).

See Task 1, Step 3 for the `WindowMetrics` struct definition (lines 94-104 replacement).

See Task 5, Step 3 for the regression sample preparation change (line ~424).

See Task 6, Step 3 for the peak/mean statistics computation change (lines ~430-441).

**Key Implementation Notes:**

- **RNG usage:** None (deterministic computation based on input intervals)
- **Metrics:** `MeanInFlight` and `PeakInFlight` added to `WindowMetrics` struct, populated in `computeWindowMetrics`, used in `AnalyzeBacklogDrift`
- **Event ordering:** Events sorted by timestamp; stable sort ensures arrivals before departures at same time
- **State mutation:** None — pure function (takes intervals, returns metrics)
- **Error handling:** Returns empty slice if `totalDurationUs > 7 days` (existing guard), handles zero-duration window (guard: `if windowDuration > 0`)

**Behavioral subtleties:**
- Initial segment (before first event) uses `ActiveStart` as count (carried from previous window or zero for first window)
- Final segment (after last event to `endUs`) uses `currentCount` (final count after all events processed)
- Empty window (no events within [startUs, endUs)): `prevTimeUs == startUs`, `currentCount == ActiveStart`, so `area = ActiveStart × (endUs - startUs)` → `MeanInFlight = ActiveStart`
- Peak tracked across all segments, including before first event and after last event (so peak is never less than `ActiveStart` or final `currentCount`)

### File: `sim/workload/saturation_test.go`

**Purpose:** Behavioral tests for saturation analyzer integral metrics.

**Complete Implementation:**

See Task 1, Step 1 for `TestWindowMetrics_NewFields_Accessible` (field accessibility test).

See Task 2, Step 1 for `TestComputeWindowMetrics_ConstantLoad_IntegralCorrect` (BC-3: steady-state equivalence).

See Task 3, Step 1 for `TestComputeWindowMetrics_SubWindowBurst_Detected` (BC-4: sub-window burst).

See Task 4, Step 1 for `TestComputeWindowMetrics_EmptyWindow_UsesActiveStart` (BC-10: empty window).

See Task 5, Step 1 for `TestAnalyzeBacklogDrift_RegressionUsesMeanInFlight` (BC-5: regression input).

See Task 6, Step 1 for `TestAnalyzeBacklogDrift_PeakMeanUsesPeakInFlight` (BC-6: peak/mean classification).

See Task 7, Step 1 for `verifyIntegralMetricsPopulated` helper function (BC-2: peak >= boundaries invariant).

**Key Implementation Notes:**

- All tests are table-driven or use explicit `RequestInterval` slices for reproducibility
- Tests verify observable behavior (MeanInFlight values, PeakInFlight values, classification outcomes), not internal implementation details
- Existing tests extended with `verifyIntegralMetricsPopulated` call to ensure backward compatibility

---

**Plan complete. Ready for execution via `superpowers:executing-plans` or `superpowers:subagent-driven-development`.**
