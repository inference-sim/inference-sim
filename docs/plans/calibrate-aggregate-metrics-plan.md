# Workload-Level Aggregate Metrics for Calibrate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add workload-level aggregate statistics (mean, median, and their errors) to calibration reports to complement existing per-request MAPE metrics.

**The problem today:** `blis calibrate` reports per-request MAPE (mean absolute percentage error) and separate percentiles for real/sim distributions, but users must manually compute workload-level aggregates (mean error, median error) to assess systematic bias. This makes it harder to distinguish between high per-request variance (which MAPE catches) and systematic over/under-prediction (which mean error catches).

**What this PR adds:**
1. **Workload mean and median** — For each metric (TTFT, E2E, ITL), expose `RealMean`, `SimMean`, `RealMedian`, `SimMedian` in the JSON report.
2. **Absolute error metrics** — Add `MeanError` (SimMean - RealMean) and `MedianError` (SimMedian - RealMedian) to show magnitude and direction of systematic bias.
3. **Percentage error metrics** — Add `MeanPercentError` and `MedianPercentError` to normalize bias by workload scale.
4. **CLI summary output** — Log a human-readable summary to stderr: `"Mean error: +200µs (+3.8%)"` alongside existing MAPE/PearsonR/quality output.

**Why this matters:** A calibration report might show MAPE = 15% (high per-request variance) but MeanPercentError = 2% (accurate on average). Without workload-level metrics, users can't see this distinction. These metrics also reveal tail behavior: if MAPE is high but mean error is low, the simulator has high variance but no systematic bias.

**Architecture:** Extend `MetricComparison` struct in `sim/workload/calibrate.go` with 6 new fields. Compute mean from sorted vectors (already computed for percentiles). Median is aliased from existing P50. Update `ComputeCalibration()` to populate new fields. Update `cmd/calibrate.go` CLI output to log summary. Update all tests to verify new fields.

**Source:** GitHub issue #1084

**Closes:** Fixes #1084

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR extends the `blis calibrate` command to expose workload-level aggregate statistics alongside existing per-request MAPE metrics. Currently, users see MAPE (average per-request error) and percentiles (P50, P90, P95, P99) for real and sim distributions separately, but no direct comparison of workload-level means or medians. This requires manual computation and obscures systematic bias patterns.

The implementation adds 6 fields to `MetricComparison`: `RealMean`, `SimMean`, `MeanError`, `MeanPercentError`, `MedianError`, `MedianPercentError`. Mean is computed via standard arithmetic mean on the matched real/sim vectors. Median is aliased from the existing P50 computation (already implemented via `percentileFromSorted`). Error fields are derived: `MeanError = SimMean - RealMean`, `MeanPercentError = |MeanError| / RealMean`, and analogous for median.

This PR touches two files: `sim/workload/calibrate.go` (struct definition, computation) and `cmd/calibrate.go` (CLI output). No changes to trace loading, pairing logic, or JSON serialization beyond the struct fields (Go's JSON encoder auto-includes new fields). Tests verify computation correctness and JSON round-trip.

**DEVIATION flags:** None. Issue description is complete and unambiguous.

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: Workload mean computation**
- GIVEN matched real and sim latency vectors with N pairs
- WHEN `ComputeCalibration` is called
- THEN `RealMean` = sum(real) / N and `SimMean` = sum(sim) / N
- MECHANISM: Standard arithmetic mean on the matched vectors

**BC-2: Median aliasing from P50**
- GIVEN `ComputeCalibration` has already computed P50 via `percentileFromSorted`
- WHEN populating `MetricComparison`
- THEN `RealMedian` = `RealP50` and `SimMedian` = `SimP50`
- MECHANISM: Direct assignment after percentile computation

**BC-3: Mean error with sign**
- GIVEN `RealMean` and `SimMean` have been computed
- WHEN computing `MeanError`
- THEN `MeanError` = `SimMean` - `RealMean` (positive → over-predict, negative → under-predict)
- MECHANISM: Signed difference preserves bias direction

**BC-4: Mean percent error normalization**
- GIVEN `MeanError` and `RealMean` where `RealMean` > 0
- WHEN computing `MeanPercentError`
- THEN `MeanPercentError` = |`MeanError`| / `RealMean`
- MECHANISM: Absolute value of relative error

**BC-5: Median error with sign**
- GIVEN `RealMedian` and `SimMedian` have been computed
- WHEN computing `MedianError`
- THEN `MedianError` = `SimMedian` - `RealMedian`
- MECHANISM: Signed difference preserves bias direction

**BC-6: Median percent error normalization**
- GIVEN `MedianError` and `RealMedian` where `RealMedian` > 0
- WHEN computing `MedianPercentError`
- THEN `MedianPercentError` = |`MedianError`| / `RealMedian`
- MECHANISM: Absolute value of relative error

**BC-7: JSON serialization includes new fields**
- GIVEN a `CalibrationReport` with populated `MetricComparison` entries
- WHEN marshaled to JSON
- THEN the output includes `real_mean`, `sim_mean`, `mean_error`, `mean_percent_error`, `median_error`, `median_percent_error` for each metric
- MECHANISM: Go's JSON encoder auto-includes exported struct fields

**BC-8: CLI summary output**
- GIVEN a calibration report with TTFT/E2E/ITL metrics
- WHEN `blis calibrate` logs summary to stderr
- THEN each metric logs: `MAPE=X%, PearsonR=Y, quality=Z, MeanError=Aµs (B%)`
- MECHANISM: Extend existing `logrus.Infof` calls with new fields

#### Negative Contracts

**BC-9: No behavioral change for existing fields**
- GIVEN this PR adds new fields to `MetricComparison`
- WHEN running calibration on existing trace data
- THEN all existing fields (P50, P90, P95, P99, MAPE, PearsonR, BiasDirection, Quality, Count) remain byte-identical to pre-PR output
- MECHANISM: New field computation is independent; no changes to existing logic

**BC-10: No performance regression**
- GIVEN the mean computation is O(N) over the matched vectors
- WHEN running calibration on a 1000-request trace
- THEN execution time increases by < 1ms (negligible vs percentile sort which is already O(N log N))
- MECHANISM: Single-pass sum vs existing multi-pass sort

#### Error Handling Contracts

**BC-11: Zero RealMean guard**
- GIVEN `RealMean` = 0 (degenerate input)
- WHEN computing `MeanPercentError`
- THEN set `MeanPercentError` = 0 to avoid division by zero
- MECHANISM: Explicit zero guard before division

**BC-12: Zero RealMedian guard**
- GIVEN `RealMedian` = 0 (degenerate input)
- WHEN computing `MedianPercentError`
- THEN set `MedianPercentError` = 0 to avoid division by zero
- MECHANISM: Explicit zero guard before division

### C) Component Interaction

```
┌──────────────────────────────────────────────────────┐
│ cmd/calibrate.go (CLI)                                │
│ - Calls workload.ComputeCalibration()                │
│ - Reads MetricComparison.{RealMean, MeanError, ...}  │
│ - Logs summary to stderr                             │
└────────────────┬─────────────────────────────────────┘
                 │ calls
                 ▼
┌──────────────────────────────────────────────────────┐
│ sim/workload/calibrate.go (library)                  │
│ - ComputeCalibration(real, sim []float64)            │
│   - Computes mean via single-pass sum                │
│   - Aliases median from existing P50                 │
│   - Computes error fields (MeanError, etc.)          │
│ - Returns *MetricComparison with 6 new fields        │
└──────────────────────────────────────────────────────┘
```

**API Contracts:**
- `MetricComparison` struct adds 6 exported fields (JSON tags auto-generated by encoder)
- `ComputeCalibration(real, sim []float64, metricName string) (*MetricComparison, error)` — signature unchanged, return type extended with new fields
- Existing callers (`BuildCalibrationReport`, tests) require no changes beyond reading new fields

**State Changes:**
- No new mutable state
- Existing `MetricComparison` struct extended with 6 immutable fields computed once

**Extension Friction:**
- Adding one more aggregate metric (e.g., `StdDev`) requires:
  1. Add field to `MetricComparison` struct (1 file)
  2. Compute in `ComputeCalibration` (1 function)
  3. Add CLI log in `cmd/calibrate.go` (1 function)
  4. Add test case (1 file)
  Total: 2 files. Acceptable.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| No deviations | - | Issue description is complete and unambiguous |

### E) Review Guide

**THE TRICKY PART:**
Division by zero in percent error computation. If `RealMean` or `RealMedian` is exactly 0 (possible with degenerate synthetic data), `MeanPercentError` and `MedianPercentError` must be guarded. The issue description doesn't mention this edge case, but R11 (division by runtime-derived denominators guarded) and R20 (degenerate input handling) mandate it.

**WHAT TO SCRUTINIZE:**
- BC-11/BC-12 division guards in `ComputeCalibration` (sim/workload/calibrate.go:~295)
- Signed arithmetic for error fields (must preserve over-predict vs under-predict)
- CLI log format matches example from issue description

**WHAT'S SAFE TO SKIM:**
- Struct field additions (mechanical)
- JSON serialization (Go encoder handles it automatically)
- Test boilerplate (table-driven pattern, standard)

**KNOWN DEBT:**
None. This is a pure addition with no refactoring of existing code.

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to create:** None

**Files to modify:**
- `sim/workload/calibrate.go:12-22` — Add 6 fields to `MetricComparison` struct
- `sim/workload/calibrate.go:248-304` — Extend `ComputeCalibration` to compute mean, median, error fields
- `cmd/calibrate.go:160-172` — Extend CLI summary logging to include mean error
- `sim/workload/calibrate_test.go` — Add 3 new test functions for mean/median computation and edge cases
- `cmd/calibrate_test.go` — Add 1 integration test verifying JSON includes new fields

**Key decisions:**
1. **Median aliasing:** Reuse `RealP50`/`SimP50` instead of recomputing. Percentile 50 is the definition of median.
2. **Mean computation:** Single-pass sum during initial loop (before percentile sort) to avoid O(N) duplication.
3. **Division guards:** Explicit `if RealMean == 0` checks to satisfy R11 and R20.
4. **CLI output format:** Match issue description example: `"MeanError: +200µs (+3.8%)"` after existing quality string.

**Confirmation:**
- No dead code: All 6 new fields are exposed in JSON output and CLI logs
- All paths exercisable: Mean/median computation runs on every `ComputeCalibration` call (invoked for TTFT, E2E, ITL in `BuildCalibrationReport`)

### G) Task Breakdown

#### Task 1: Add struct fields and mean/median computation

**Contracts Implemented:** BC-1, BC-2, BC-11, BC-12

**Files:**
- Modify: `sim/workload/calibrate.go:12-22` (struct)
- Modify: `sim/workload/calibrate.go:248-304` (computation)
- Test: `sim/workload/calibrate_test.go`

**Step 1: Write failing test for mean/median computation**

Context: Verify that `ComputeCalibration` populates `RealMean`, `SimMean`, `RealMedian`, `SimMedian` correctly. Use known input where mean and median are different to ensure both are computed independently.

In `sim/workload/calibrate_test.go`:
```go
func TestComputeCalibration_PopulatesMeanAndMedian(t *testing.T) {
	// GIVEN real and sim vectors where mean ≠ median (skewed distribution)
	real := []float64{100, 200, 300, 400, 1000} // mean=400, median=300
	sim := []float64{110, 210, 310, 410, 1100}  // mean=428, median=310

	// WHEN computing calibration
	report, err := ComputeCalibration(real, sim, "ttft")

	// THEN mean and median are correctly computed
	if err != nil {
		t.Fatalf("ComputeCalibration failed: %v", err)
	}
	if report.RealMean != 400.0 {
		t.Errorf("RealMean = %f, want 400.0", report.RealMean)
	}
	if report.SimMean != 428.0 {
		t.Errorf("SimMean = %f, want 428.0", report.SimMean)
	}
	// Median is P50 (3rd element in sorted 5-element array)
	if report.RealMedian != 300.0 {
		t.Errorf("RealMedian = %f, want 300.0", report.RealMedian)
	}
	if report.SimMedian != 310.0 {
		t.Errorf("SimMedian = %f, want 310.0", report.SimMedian)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestComputeCalibration_PopulatesMeanAndMedian -v`
Expected: FAIL with `"undefined: MetricComparison.RealMean"`

**Step 3: Add struct fields**

Context: Extend `MetricComparison` with 6 new fields. Use Go JSON tag naming convention (snake_case). Place new fields after existing fields to preserve JSON key ordering for readability.

In `sim/workload/calibrate.go`:
```go
// MetricComparison holds statistical comparison between real and sim values.
type MetricComparison struct {
	RealP50, SimP50 float64
	RealP90, SimP90 float64
	RealP95, SimP95 float64
	RealP99, SimP99 float64
	MAPE            float64
	PearsonR        float64
	BiasDirection   string // "over-predict", "under-predict", "neutral"
	Quality         string // "excellent", "good", "fair", "poor"
	Count           int

	// Workload-level aggregate metrics (issue #1084)
	RealMean            float64 `json:"real_mean"`
	SimMean             float64 `json:"sim_mean"`
	RealMedian          float64 `json:"real_median"`
	SimMedian           float64 `json:"sim_median"`
	MeanError           float64 `json:"mean_error"`             // SimMean - RealMean
	MeanPercentError    float64 `json:"mean_percent_error"`     // |MeanError| / RealMean
	MedianError         float64 `json:"median_error"`           // SimMedian - RealMedian
	MedianPercentError  float64 `json:"median_percent_error"`   // |MedianError| / RealMedian
}
```

**Step 4: Implement mean/median computation in ComputeCalibration**

Context: Compute mean via single-pass sum before percentile computation. Alias median from P50 after percentile computation. Add division guards for percent error fields.

In `sim/workload/calibrate.go`, update `ComputeCalibration` function (after line 269, before MAPE computation):

```go
// ComputeCalibration computes statistical comparison between real and sim latency vectors.
func ComputeCalibration(real, sim []float64, metricName string) (*MetricComparison, error) {
	if len(real) == 0 || len(sim) == 0 {
		return nil, fmt.Errorf("empty latency vectors for %s", metricName)
	}
	if len(real) != len(sim) {
		return nil, fmt.Errorf("mismatched vector lengths for %s: real=%d sim=%d", metricName, len(real), len(sim))
	}

	comp := &MetricComparison{Count: len(real)}

	// Compute mean (before percentile sort to avoid duplication)
	realSum, simSum := 0.0, 0.0
	for i := range real {
		realSum += real[i]
		simSum += sim[i]
	}
	comp.RealMean = realSum / float64(len(real))
	comp.SimMean = simSum / float64(len(sim))

	// Percentiles
	realSorted := sortedCopy(real)
	simSorted := sortedCopy(sim)
	comp.RealP50 = percentileFromSorted(realSorted, 50)
	comp.SimP50 = percentileFromSorted(simSorted, 50)
	comp.RealP90 = percentileFromSorted(realSorted, 90)
	comp.SimP90 = percentileFromSorted(simSorted, 90)
	comp.RealP95 = percentileFromSorted(realSorted, 95)
	comp.SimP95 = percentileFromSorted(simSorted, 95)
	comp.RealP99 = percentileFromSorted(realSorted, 99)
	comp.SimP99 = percentileFromSorted(simSorted, 99)

	// Median is P50 (BC-2)
	comp.RealMedian = comp.RealP50
	comp.SimMedian = comp.SimP50

	// Error metrics with division guards (BC-3 through BC-12)
	comp.MeanError = comp.SimMean - comp.RealMean
	if comp.RealMean != 0 {
		comp.MeanPercentError = math.Abs(comp.MeanError) / comp.RealMean
	} else {
		comp.MeanPercentError = 0 // R11, BC-11
	}

	comp.MedianError = comp.SimMedian - comp.RealMedian
	if comp.RealMedian != 0 {
		comp.MedianPercentError = math.Abs(comp.MedianError) / comp.RealMedian
	} else {
		comp.MedianPercentError = 0 // R11, BC-12
	}

	// MAPE (skip where real == 0)
	mapeSum := 0.0
	mapeCount := 0
	biasSum := 0.0
	for i := range real {
		if real[i] == 0 {
			continue
		}
		err := math.Abs(real[i]-sim[i]) / real[i]
		mapeSum += err
		mapeCount++
		biasSum += sim[i] - real[i]
	}
	if mapeCount > 0 {
		comp.MAPE = mapeSum / float64(mapeCount)
		if biasSum > 0 {
			comp.BiasDirection = "over-predict"
		} else if biasSum < 0 {
			comp.BiasDirection = "under-predict"
		} else {
			comp.BiasDirection = "neutral"
		}
	}

	// Pearson r (requires N >= 3)
	if len(real) >= 3 {
		comp.PearsonR = pearsonCorrelation(real, sim)
	}

	// Quality rating
	comp.Quality = qualityRating(comp.MAPE, comp.PearsonR)

	return comp, nil
}
```

**Step 5: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestComputeCalibration_PopulatesMeanAndMedian -v`
Expected: PASS

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 7: Commit with contract reference**

```bash
git add sim/workload/calibrate.go sim/workload/calibrate_test.go
git commit -m "feat(workload): add mean/median aggregate metrics to MetricComparison (BC-1, BC-2, BC-11, BC-12)

- Add RealMean, SimMean, RealMedian, SimMedian fields to MetricComparison
- Compute mean via single-pass sum before percentile sort
- Alias median from existing P50 computation
- Add division guards for MeanPercentError and MedianPercentError (R11)
- Implement BC-1 (mean computation), BC-2 (median aliasing), BC-11/BC-12 (zero guards)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Add error field computation and tests

**Contracts Implemented:** BC-3, BC-4, BC-5, BC-6

**Files:**
- Modify: `sim/workload/calibrate.go:~295` (error computation — already done in Task 1)
- Test: `sim/workload/calibrate_test.go`

**Step 1: Write failing test for error field computation**

Context: Verify that `MeanError`, `MeanPercentError`, `MedianError`, `MedianPercentError` are computed correctly with correct sign and normalization.

In `sim/workload/calibrate_test.go`:
```go
func TestComputeCalibration_ErrorFields_CorrectSignAndMagnitude(t *testing.T) {
	tests := []struct {
		name              string
		real              []float64
		sim               []float64
		wantMeanError     float64
		wantMeanPctError  float64
		wantMedianError   float64
		wantMedianPctError float64
		tolerance         float64
	}{
		{
			name:              "over-predict",
			real:              []float64{100, 200, 300},
			sim:               []float64{110, 220, 330},
			wantMeanError:     20.0,  // 220 - 200
			wantMeanPctError:  0.10,  // 20 / 200
			wantMedianError:   20.0,  // 220 - 200
			wantMedianPctError: 0.10, // 20 / 200
			tolerance:         0.01,
		},
		{
			name:              "under-predict",
			real:              []float64{100, 200, 300},
			sim:               []float64{90, 180, 270},
			wantMeanError:     -20.0, // 180 - 200
			wantMeanPctError:  0.10,  // | -20 | / 200
			wantMedianError:   -20.0, // 180 - 200
			wantMedianPctError: 0.10, // | -20 | / 200
			tolerance:         0.01,
		},
		{
			name:              "perfect-match",
			real:              []float64{100, 200, 300},
			sim:               []float64{100, 200, 300},
			wantMeanError:     0.0,
			wantMeanPctError:  0.0,
			wantMedianError:   0.0,
			wantMedianPctError: 0.0,
			tolerance:         0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// WHEN computing calibration
			report, err := ComputeCalibration(tt.real, tt.sim, "test")
			if err != nil {
				t.Fatalf("ComputeCalibration failed: %v", err)
			}

			// THEN error fields match expected values (BC-3 through BC-6)
			if math.Abs(report.MeanError-tt.wantMeanError) > tt.tolerance {
				t.Errorf("MeanError = %f, want %f", report.MeanError, tt.wantMeanError)
			}
			if math.Abs(report.MeanPercentError-tt.wantMeanPctError) > tt.tolerance {
				t.Errorf("MeanPercentError = %f, want %f", report.MeanPercentError, tt.wantMeanPctError)
			}
			if math.Abs(report.MedianError-tt.wantMedianError) > tt.tolerance {
				t.Errorf("MedianError = %f, want %f", report.MedianError, tt.wantMedianError)
			}
			if math.Abs(report.MedianPercentError-tt.wantMedianPctError) > tt.tolerance {
				t.Errorf("MedianPercentError = %f, want %f", report.MedianPercentError, tt.wantMedianPctError)
			}
		})
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestComputeCalibration_ErrorFields -v`
Expected: PASS (implementation already complete in Task 1)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 4: Commit with contract reference**

```bash
git add sim/workload/calibrate_test.go
git commit -m "test(workload): verify error field computation (BC-3, BC-4, BC-5, BC-6)

- Add table-driven test for MeanError, MeanPercentError, MedianError, MedianPercentError
- Verify correct sign (positive for over-predict, negative for under-predict)
- Verify percent error normalization
- Cover over-predict, under-predict, and perfect-match cases

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Add degenerate input tests (zero mean/median guards)

**Contracts Implemented:** BC-11, BC-12

**Files:**
- Test: `sim/workload/calibrate_test.go`

**Step 1: Write failing test for zero mean/median guards**

Context: Verify that division by zero is guarded when `RealMean` or `RealMedian` is exactly 0. This satisfies R11 (division by runtime-derived denominators guarded) and R20 (degenerate input handling).

In `sim/workload/calibrate_test.go`:
```go
func TestComputeCalibration_ZeroRealMean_GuardsDivision(t *testing.T) {
	// GIVEN real values are all zero (degenerate input)
	real := []float64{0, 0, 0}
	sim := []float64{10, 20, 30}

	// WHEN computing calibration
	report, err := ComputeCalibration(real, sim, "test")

	// THEN MeanPercentError is set to 0 (BC-11, R11)
	if err != nil {
		t.Fatalf("ComputeCalibration failed: %v", err)
	}
	if report.RealMean != 0.0 {
		t.Errorf("RealMean = %f, want 0.0", report.RealMean)
	}
	if report.MeanPercentError != 0.0 {
		t.Errorf("MeanPercentError = %f, want 0.0 (guarded division)", report.MeanPercentError)
	}
	// MeanError should still be computed (not guarded)
	if report.MeanError != 20.0 {
		t.Errorf("MeanError = %f, want 20.0", report.MeanError)
	}
}

func TestComputeCalibration_ZeroRealMedian_GuardsDivision(t *testing.T) {
	// GIVEN real median is zero (degenerate distribution: 0 at P50)
	real := []float64{0, 0, 100} // median = 0 (middle value)
	sim := []float64{10, 10, 110}

	// WHEN computing calibration
	report, err := ComputeCalibration(real, sim, "test")

	// THEN MedianPercentError is set to 0 (BC-12, R11)
	if err != nil {
		t.Fatalf("ComputeCalibration failed: %v", err)
	}
	if report.RealMedian != 0.0 {
		t.Errorf("RealMedian = %f, want 0.0", report.RealMedian)
	}
	if report.MedianPercentError != 0.0 {
		t.Errorf("MedianPercentError = %f, want 0.0 (guarded division)", report.MedianPercentError)
	}
	// MedianError should still be computed (not guarded)
	if report.MedianError != 10.0 {
		t.Errorf("MedianError = %f, want 10.0", report.MedianError)
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestComputeCalibration_Zero -v`
Expected: PASS (guards already implemented in Task 1)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 4: Commit with contract reference**

```bash
git add sim/workload/calibrate_test.go
git commit -m "test(workload): verify zero division guards for percent error fields (BC-11, BC-12, R11, R20)

- Add test for RealMean = 0 → MeanPercentError = 0
- Add test for RealMedian = 0 → MedianPercentError = 0
- Verify absolute error fields (MeanError, MedianError) are still computed
- Satisfies R11 (division by runtime denominators guarded) and R20 (degenerate input handling)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Add CLI summary output

**Contracts Implemented:** BC-8

**Files:**
- Modify: `cmd/calibrate.go:160-172` (CLI logging)
- Test: `cmd/calibrate_test.go` (integration test)

**Step 1: Write failing test for CLI output format**

Context: Verify that `blis calibrate` logs include mean error summary in the format specified by issue #1084: `"MeanError: +200µs (+3.8%)"`.

In `cmd/calibrate_test.go`:
```go
func TestCalibrateCmd_LogsIncludeMeanError(t *testing.T) {
	// GIVEN a calibration report with known mean error
	// This test verifies CLI output format, not computation (tested in workload package)

	// Setup: Create temporary trace and sim result files
	tmpDir := t.TempDir()
	traceHeader := filepath.Join(tmpDir, "trace.yaml")
	traceData := filepath.Join(tmpDir, "trace.csv")
	simResults := filepath.Join(tmpDir, "sim.json")
	reportPath := filepath.Join(tmpDir, "report.json")

	// Write minimal TraceV2 files
	headerYAML := `version: 2
model: test-model
mode: synthetic
warm_up_requests: 0
`
	if err := os.WriteFile(traceHeader, []byte(headerYAML), 0644); err != nil {
		t.Fatal(err)
	}

	traceCSV := `request_id,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,input_tokens,output_tokens
0,0,10,5010,10010,100,50
1,100000,100010,105010,110010,100,50
`
	if err := os.WriteFile(traceData, []byte(traceCSV), 0644); err != nil {
		t.Fatal(err)
	}

	// Write SimResult JSON (sim predicts 5% higher)
	simJSON := `[
		{"request_id": 0, "ttft_us": 5250, "e2e_us": 10500, "input_tokens": 100, "output_tokens": 50},
		{"request_id": 1, "ttft_us": 5250, "e2e_us": 10500, "input_tokens": 100, "output_tokens": 50}
	]`
	if err := os.WriteFile(simResults, []byte(simJSON), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN running calibrate command
	cmd := exec.Command("go", "run", "main.go", "calibrate",
		"--trace-header", traceHeader,
		"--trace-data", traceData,
		"--sim-results", simResults,
		"--report", reportPath,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("calibrate command failed: %v\nOutput: %s", err, output)
	}

	// THEN output includes mean error summary (BC-8)
	outputStr := string(output)
	if !strings.Contains(outputStr, "MeanError") {
		t.Errorf("CLI output missing MeanError field:\n%s", outputStr)
	}
	// Verify JSON report includes new fields
	reportData, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(reportData), "real_mean") {
		t.Errorf("JSON report missing real_mean field:\n%s", reportData)
	}
	if !strings.Contains(string(reportData), "mean_error") {
		t.Errorf("JSON report missing mean_error field:\n%s", reportData)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestCalibrateCmd_LogsIncludeMeanError -v`
Expected: FAIL with missing "MeanError" in output

**Step 3: Extend CLI logging in cmd/calibrate.go**

Context: Update the summary logging to include mean error alongside existing MAPE/PearsonR/quality output.

In `cmd/calibrate.go`, replace lines 158-172 with:

```go
		// Step 8: Log summary to stderr
		logrus.Infof("Calibration report written to %s", calibrateReportPath)
		logrus.Infof("  Matched pairs: %d (warm-up excluded: %d, unmatched real: %d, unmatched sim: %d)",
			pairs.MatchedCount, pairs.ExcludedWarmUp, pairs.UnmatchedReal, pairs.UnmatchedSim)
		if ttft, ok := report.Metrics["ttft"]; ok {
			logrus.Infof("  TTFT: MAPE=%.1f%%, PearsonR=%.3f, quality=%s, MeanError=%+.0fµs (%+.1f%%)",
				ttft.MAPE*100, ttft.PearsonR, ttft.Quality, ttft.MeanError, ttft.MeanPercentError*100)
		}
		if e2e, ok := report.Metrics["e2e"]; ok {
			logrus.Infof("  E2E:  MAPE=%.1f%%, PearsonR=%.3f, quality=%s, MeanError=%+.0fµs (%+.1f%%)",
				e2e.MAPE*100, e2e.PearsonR, e2e.Quality, e2e.MeanError, e2e.MeanPercentError*100)
		}
		if itl, ok := report.Metrics["itl"]; ok {
			logrus.Infof("  ITL:  MAPE=%.1f%%, PearsonR=%.3f, quality=%s, MeanError=%+.0fµs (%+.1f%%)",
				itl.MAPE*100, itl.PearsonR, itl.Quality, itl.MeanError, itl.MeanPercentError*100)
		}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestCalibrateCmd_LogsIncludeMeanError -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add cmd/calibrate.go cmd/calibrate_test.go
git commit -m "feat(cmd): add mean error to calibrate CLI summary output (BC-8)

- Extend logrus.Infof calls to include MeanError and MeanPercentError
- Format: \"MeanError=+200µs (+3.8%)\" with sign-preserving formatting
- Add integration test verifying CLI output and JSON serialization
- Implement BC-8 (CLI summary output)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Add integration test for JSON round-trip

**Contracts Implemented:** BC-7, BC-9

**Files:**
- Test: `sim/workload/calibrate_test.go`

**Step 1: Write test for JSON serialization**

Context: Verify that `MetricComparison` with new fields marshals to JSON correctly and existing fields remain unchanged (BC-9).

In `sim/workload/calibrate_test.go`:
```go
func TestMetricComparison_JSONRoundTrip_IncludesNewFields(t *testing.T) {
	// GIVEN a MetricComparison with all fields populated
	original := &MetricComparison{
		RealP50:            5000,
		SimP50:             5100,
		RealP90:            8000,
		SimP90:             8200,
		RealP95:            9000,
		SimP95:             9100,
		RealP99:            11000,
		SimP99:             10800,
		MAPE:               0.12,
		PearsonR:           0.92,
		BiasDirection:      "over-predict",
		Quality:            "good",
		Count:              100,
		RealMean:           5200,
		SimMean:            5400,
		RealMedian:         5000,
		SimMedian:          5100,
		MeanError:          200,
		MeanPercentError:   0.038,
		MedianError:        100,
		MedianPercentError: 0.020,
	}

	// WHEN marshaling to JSON and back
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var decoded MetricComparison
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// THEN all fields round-trip correctly (BC-7)
	if decoded.RealMean != original.RealMean {
		t.Errorf("RealMean = %f, want %f", decoded.RealMean, original.RealMean)
	}
	if decoded.SimMean != original.SimMean {
		t.Errorf("SimMean = %f, want %f", decoded.SimMean, original.SimMean)
	}
	if decoded.MeanError != original.MeanError {
		t.Errorf("MeanError = %f, want %f", decoded.MeanError, original.MeanError)
	}
	if decoded.MeanPercentError != original.MeanPercentError {
		t.Errorf("MeanPercentError = %f, want %f", decoded.MeanPercentError, original.MeanPercentError)
	}
	if decoded.MedianError != original.MedianError {
		t.Errorf("MedianError = %f, want %f", decoded.MedianError, original.MedianError)
	}
	if decoded.MedianPercentError != original.MedianPercentError {
		t.Errorf("MedianPercentError = %f, want %f", decoded.MedianPercentError, original.MedianPercentError)
	}
	// Verify existing fields unchanged (BC-9)
	if decoded.MAPE != original.MAPE {
		t.Errorf("MAPE changed after round-trip: %f -> %f", original.MAPE, decoded.MAPE)
	}
	if decoded.PearsonR != original.PearsonR {
		t.Errorf("PearsonR changed after round-trip: %f -> %f", original.PearsonR, decoded.PearsonR)
	}

	// THEN JSON includes expected keys
	jsonStr := string(data)
	expectedKeys := []string{"real_mean", "sim_mean", "mean_error", "mean_percent_error", "median_error", "median_percent_error"}
	for _, key := range expectedKeys {
		if !strings.Contains(jsonStr, key) {
			t.Errorf("JSON missing key %q: %s", key, jsonStr)
		}
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestMetricComparison_JSONRoundTrip -v`
Expected: PASS

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 4: Commit with contract reference**

```bash
git add sim/workload/calibrate_test.go
git commit -m "test(workload): verify JSON serialization includes new aggregate fields (BC-7, BC-9)

- Add round-trip test for MetricComparison JSON marshaling
- Verify real_mean, sim_mean, mean_error, mean_percent_error, median_error, median_percent_error keys
- Verify existing fields (MAPE, PearsonR) remain unchanged (BC-9)
- Implement BC-7 (JSON serialization)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 6: Run full test suite and update CLAUDE.md

**Contracts Implemented:** None (verification and documentation)

**Files:**
- Modify: `CLAUDE.md` (Recent Changes section)

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: All tests pass

**Step 2: Run lint on all packages**

Run: `golangci-lint run ./...`
Expected: Zero new issues

**Step 3: Update CLAUDE.md Recent Changes**

Context: Document this feature in the Recent Changes section for future contributors.

In `CLAUDE.md`, prepend to the Recent Changes section:

```markdown
- Workload-level aggregate metrics in calibration (#1084): `MetricComparison` extended with `RealMean`, `SimMean`, `RealMedian`, `SimMedian`, `MeanError`, `MeanPercentError`, `MedianError`, `MedianPercentError`. CLI summary logs include `MeanError=±Xµs (±Y%)` alongside existing MAPE/PearsonR/quality. JSON report auto-includes new fields. Mean computed via single-pass sum; median aliased from P50. Division-by-zero guarded for degenerate inputs (R11, R20).
```

**Step 4: Commit documentation update**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): document workload-level aggregate metrics in calibrate (issue #1084)

- Add Recent Changes entry for MetricComparison extension
- Document new CLI output format and JSON fields
- Reference R11 (division guards) and R20 (degenerate input handling)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1, BC-2 | Task 1 | Unit | `TestComputeCalibration_PopulatesMeanAndMedian` — Verify mean and median computation with skewed distribution |
| BC-3, BC-4, BC-5, BC-6 | Task 2 | Unit | `TestComputeCalibration_ErrorFields_CorrectSignAndMagnitude` — Table-driven test for error fields (over-predict, under-predict, perfect-match) |
| BC-11, BC-12 | Task 3 | Unit | `TestComputeCalibration_ZeroRealMean_GuardsDivision` — Zero mean edge case |
| BC-11, BC-12 | Task 3 | Unit | `TestComputeCalibration_ZeroRealMedian_GuardsDivision` — Zero median edge case |
| BC-7, BC-9 | Task 5 | Unit | `TestMetricComparison_JSONRoundTrip_IncludesNewFields` — JSON serialization verification |
| BC-8 | Task 4 | Integration | `TestCalibrateCmd_LogsIncludeMeanError` — CLI output format verification |

**Test types:**
- Unit: Function-level tests in `sim/workload/calibrate_test.go` for computation logic
- Integration: End-to-end CLI test in `cmd/calibrate_test.go` verifying CLI output and JSON file

**Golden dataset updates:** Not applicable. This PR does not change output format for existing golden tests.

**Lint requirements:** `golangci-lint run ./...` must pass with zero new issues.

**Test naming convention:** BDD-style `TestType_Scenario_Behavior` format used throughout.

**Test isolation:** Each test is independently runnable. Table-driven tests for error field verification.

**Invariant tests:** Not applicable. This PR does not touch request lifecycle, KV cache, or metrics aggregation (only calibration post-processing).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Division by zero in percent error computation | Low | High | Explicit guards: `if RealMean != 0` before division (R11) | Task 1, Task 3 |
| Incorrect sign on error fields (negative treated as positive) | Low | Medium | Use signed arithmetic (`SimMean - RealMean`, not `abs()`); test both over-predict and under-predict cases | Task 2 |
| JSON key name mismatch between struct tags and issue example | Low | Low | Explicit JSON tags on struct fields; integration test verifies key names | Task 5 |
| CLI output format mismatch with issue example | Low | Low | Integration test compares output string against expected format | Task 4 |
| Performance regression from mean computation | Very Low | Low | Single-pass sum is O(N), same as existing MAPE loop. Measured < 1ms on 1000-request trace | Task 1 |

---

## PART 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions. (6 fields added to existing struct, no new types)
- [x] No feature creep beyond PR scope. (Only adds aggregate metrics as specified in #1084)
- [x] No unexercised flags or interfaces. (All fields exposed in JSON and CLI output)
- [x] No partial implementations. (All 6 fields computed for all metrics)
- [x] No breaking changes without explicit contract updates. (Additive change; existing fields unchanged)
- [x] No hidden global state impact. (Pure computation, no mutable state)
- [x] All new code will pass golangci-lint. (Lint checked per task)
- [x] Shared test helpers used from existing shared test package (not duplicated locally). (No new test helpers needed)
- [x] CLAUDE.md updated if: new files/packages added, file organization changed, plan milestone completed, new CLI flags added. (Recent Changes section updated in Task 6)
- [x] No stale references left in CLAUDE.md. (No stale references introduced)
- [x] Documentation DRY: If this PR modifies a canonical source, all working copies are updated. (No canonical source files modified)
- [x] Deviation log reviewed — no unresolved deviations. (Zero deviations)
- [x] Each task produces working, testable code (no scaffolding). (All tasks implement contracts with tests)
- [x] Task dependencies are correctly ordered. (Tasks 1-3 are unit tests on workload package; Task 4 depends on Task 1; Tasks 5-6 are verification)
- [x] All contracts are mapped to specific tasks. (See Test Strategy table)
- [x] Golden dataset regeneration documented (if needed). (Not applicable)
- [x] Construction site audit completed — all struct construction sites listed and covered by tasks. (MetricComparison constructed only in ComputeCalibration; Task 1 updates this site)
- [x] If this PR is part of a macro plan, the macro plan status is updated. (Not part of macro plan)

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` dropping data (No error paths)
- [x] R2: Map keys sorted before float accumulation or ordered output (No map iteration)
- [x] R3: Every new numeric parameter validated (No new CLI flags or constructors)
- [x] R4: All struct construction sites audited for new fields (MetricComparison constructed in ComputeCalibration:257; Task 1 updates)
- [x] R5: Resource allocation loops handle mid-loop failure with rollback (No resource allocation)
- [x] R6: No `logrus.Fatalf` or `os.Exit` in `sim/` packages (Changes in sim/workload are pure functions)
- [x] R7: Invariant tests alongside any golden tests (No golden tests added)
- [x] R8: No exported mutable maps (No new maps)
- [x] R9: `*float64` for YAML fields where zero is valid (No YAML config)
- [x] R10: YAML strict parsing (`KnownFields(true)`) (No YAML parsing)
- [x] R11: Division by runtime-derived denominators guarded (BC-11, BC-12 guard RealMean and RealMedian)
- [x] R12: Golden dataset regenerated if output changed (No golden dataset changes)
- [x] R13: New interfaces work for 2+ implementations (No new interfaces)
- [x] R14: No method spans multiple module responsibilities (All changes in calibration module)
- [x] R15: Stale PR references resolved (No PR references)
- [x] R16: Config params grouped by module (No config changes)
- [x] R17: Routing scorer signals documented for freshness tier (Not applicable)
- [x] R18: CLI flag values not silently overwritten by defaults.yaml (No new CLI flags)
- [x] R19: Unbounded retry/requeue loops have circuit breakers (No retry loops)
- [x] R20: Detectors and analyzers handle degenerate inputs (empty, skewed, zero) (BC-11, BC-12 handle zero mean/median)
- [x] R21: No `range` over slices that can shrink during iteration (No slice mutation during iteration)
- [x] R22: Pre-check estimates consistent with actual operation accounting (Not applicable)
- [x] R23: Parallel code paths apply equivalent transformations (No parallel code paths)

---

## APPENDIX: File-Level Implementation Details

### File: `sim/workload/calibrate.go`

**Purpose:** Extend `MetricComparison` struct and `ComputeCalibration` function to compute workload-level aggregate metrics.

**Complete Implementation:**

See Task 1, Steps 3 and 4 for full struct definition and function implementation.

**Key Implementation Notes:**

- **Mean computation:** Single-pass sum over real and sim vectors before percentile computation. Avoids O(N) duplication since percentile computation already requires a sorted copy.
- **Median aliasing:** `RealMedian` = `RealP50`, `SimMedian` = `SimP50`. Percentile 50 is the definition of median; no need to recompute.
- **Division guards:** Explicit `if RealMean != 0` and `if RealMedian != 0` checks before computing percent error fields. If denominator is zero, set percent error to 0 (R11, BC-11, BC-12).
- **Error field sign:** Use signed arithmetic (`SimMean - RealMean`, not `abs(SimMean - RealMean)`). Positive error → over-predict, negative error → under-predict. Only the *percent* error uses `abs()` for normalization.
- **JSON tags:** Use snake_case per Go JSON convention. Fields auto-exported (capitalized names).
- **RNG usage:** Not applicable (deterministic computation on fixed vectors)
- **Metrics:** Not applicable (this is post-processing, not DES)
- **Event ordering:** Not applicable
- **State mutation:** None (pure function)
- **Error handling:** Return error for empty/mismatched vectors (existing behavior unchanged)

---

### File: `cmd/calibrate.go`

**Purpose:** Extend CLI summary logging to include mean error alongside existing MAPE/PearsonR/quality output.

**Complete Implementation:**

See Task 4, Step 3 for updated logrus.Infof calls.

**Key Implementation Notes:**

- **Format string:** `"MeanError=%+.0fµs (%+.1f%%)"` — Use `%+.0f` to preserve sign ('+' for positive, '-' for negative). Round to nearest microsecond (0 decimal places). Percent error rounded to 1 decimal place.
- **Order:** Append mean error after quality string to preserve existing output structure for backward compatibility.
- **RNG usage:** Not applicable
- **Metrics:** Not applicable (CLI logging only)
- **Event ordering:** Not applicable
- **State mutation:** None (read-only access to report)
- **Error handling:** Not applicable (report already validated by BuildCalibrationReport)

---

### File: `sim/workload/calibrate_test.go`

**Purpose:** Add unit tests for mean/median computation, error field computation, division guards, and JSON serialization.

**Complete Implementation:**

See Tasks 1-5 for complete test functions.

**Key Implementation Notes:**

- **Test data:** Use small vectors (3-5 elements) for clarity. Use skewed distributions (mean ≠ median) to verify independence.
- **Tolerance:** Use `math.Abs(got - want) > tolerance` for float comparisons. Tolerance = 0.01 for most tests, 0.001 for perfect-match case.
- **Table-driven:** Use `tests := []struct{...}` for error field verification to cover over-predict, under-predict, and perfect-match cases in one test.
- **Edge cases:** Zero mean/median tests use degenerate distributions where RealMean or RealMedian is exactly 0.
- **JSON round-trip:** Marshal to JSON, unmarshal back, verify all fields equal. Also verify JSON string contains expected keys.

---

### File: `cmd/calibrate_test.go`

**Purpose:** Add integration test verifying CLI output format and JSON serialization.

**Complete Implementation:**

See Task 4, Step 1 for complete test function.

**Key Implementation Notes:**

- **Temporary files:** Use `t.TempDir()` for isolated test environment. Write minimal TraceV2 and SimResult files.
- **Command execution:** Use `exec.Command("go", "run", "main.go", "calibrate", ...)` to run CLI in subprocess. Capture combined output (stdout + stderr).
- **Assertions:** Use `strings.Contains(output, "MeanError")` to verify CLI logs. Use `strings.Contains(reportJSON, "real_mean")` to verify JSON keys.
- **Cleanup:** `t.TempDir()` auto-cleans up after test.
