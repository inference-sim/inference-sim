# Support Absolute Rate Mode for CohortSpec with Spike TraceRate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable cohorts with spike windows to specify per-cohort arrival rates in absolute rate mode, allowing multi-period workloads with non-overlapping temporal windows.

**The problem today:** Multi-period ServeGen workloads with non-overlapping temporal windows (midnight, morning, afternoon) break when using cohorts in proportional mode. The `spikeWindow()` function creates `ActiveWindow` without a `TraceRate` field, causing `computeProportionalRate()` to use the global `aggregate_rate` even when only a subset of clients are active. This inflates arrival rates by 2-3x during sparse periods. For example, with `aggregate_rate: 85.3` and only 35 active clients (rate_fractions summing to 0.32), each client gets `85.3 × (rate_fraction / 0.32)` instead of the expected proportional share of 34.2 req/s.

**What this PR adds:**
1. **Cohort-level spike rates** — `SpikeSpec.TraceRate` field allows specifying per-cohort arrival rate (e.g., `trace_rate: 7.6` for a 7-member cohort → 1.086 req/s per client)
2. **Propagation path** — `SpikeSpec.TraceRate` → `ActiveWindow.TraceRate` → used by absolute mode (`aggregate_rate: 0`)
3. **Per-client division** — During cohort expansion, `trace_rate` is divided by `population` so each client gets its share
4. **Relaxed validation** — Cohorts are now allowed in absolute mode if their spike windows provide `trace_rate`

**Why this matters:** This enables issue #1217 (multi-period ServeGen workloads) and issue #1223 (multi-period CohortSpec generation). Disjoint temporal windows have disjoint aggregate rates, so "proportional to a global aggregate" doesn't make semantic sense. Absolute mode correctly says "each period has its own rate budget."

**Architecture:** Extends existing workload spec types (`sim/workload/spec.go`, `sim/workload/cohort.go`). No new interfaces or types. Adds optional `TraceRate *float64` field to `SpikeSpec`, propagates through `spikeWindow()` to `ActiveWindow`, divides by population during `ExpandCohorts()`, and updates validation to allow cohorts in absolute mode when spike windows provide rates.

**Source:** GitHub issue #1225

**Closes:** Fixes #1225

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR enables cohorts with spike windows to work in absolute rate mode by adding per-cohort `trace_rate` specification. Today, cohorts require proportional mode (`aggregate_rate > 0`), which breaks for non-overlapping temporal windows because inactive clients still contribute to the denominator when computing proportional rates. This PR adds `SpikeSpec.TraceRate`, propagates it through `spikeWindow()` to `ActiveWindow.TraceRate`, divides by population during cohort expansion, and relaxes validation to allow cohorts in absolute mode.

Adjacent blocks: `sim/workload/generator.go` (consumes `ActiveWindow.TraceRate` in absolute mode at line 951), workload YAML parsing, ServeGen converter output.

**Deviation flags:**
- None. The issue description accurately reflects the current codebase state.

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: Spike TraceRate Propagation**
- GIVEN a `CohortSpec` with `Spike.TraceRate = 7.6` and `Population = 7`
- WHEN `ExpandCohorts()` processes the cohort
- THEN each expanded client's spike window MUST have `TraceRate = 1.086` (7.6 / 7)
- MECHANISM: `spikeWindow()` copies `SpikeSpec.TraceRate` to `ActiveWindow.TraceRate`; cohort expansion divides by population

**BC-2: Absolute Mode Rate Usage**
- GIVEN a workload with `aggregate_rate: 0` and a client with spike window `TraceRate = 1.086`
- WHEN the generator computes arrival rate for that window
- THEN `computeProportionalRate()` MUST return `1.086` directly (not scaled)
- MECHANISM: generator.go:951 returns `traceRate` when `aggregateRate == 0 && window.TraceRate != nil`

**BC-3: YAML Parsing with TraceRate**
- GIVEN a YAML cohort spec with `spike: { start_time_us: 0, duration_us: 600000000, trace_rate: 7.6 }`
- WHEN the YAML is unmarshaled into `CohortSpec`
- THEN `Spike.TraceRate` MUST be `*float64(7.6)` (not nil)
- MECHANISM: yaml struct tag `trace_rate,omitempty` allows optional field

**BC-4: Backward Compatibility**
- GIVEN a cohort spec without `spike.trace_rate` in proportional mode
- WHEN the workload is validated and expanded
- THEN behavior MUST be identical to before this PR (no `TraceRate` propagated, proportional allocation used)
- MECHANISM: `TraceRate *float64` defaults to `nil`; existing code paths unchanged

#### Negative Contracts

**BC-5: Cohort Without TraceRate in Absolute Mode Rejected**
- GIVEN a workload with `aggregate_rate: 0` and a cohort with spike but no `trace_rate`
- WHEN `Validate()` is called
- THEN it MUST return an error: `"aggregate_rate is 0 (absolute rate mode) but cohort %d has spike without trace_rate"`
- MECHANISM: Validation iterates cohorts, checks `Spike != nil && Spike.TraceRate == nil`

**BC-6: Diurnal and Drain in Absolute Mode**
- GIVEN a workload with `aggregate_rate: 0` and a cohort with `Diurnal` or `Drain`
- WHEN `Validate()` is called
- THEN it MUST pass validation (these patterns don't yet support per-window rates, but aren't blocked)
- MECHANISM: Validation only requires `trace_rate` for spike patterns

#### Error Handling Contracts

**BC-7: Nil TraceRate Pointer Safety**
- GIVEN a `SpikeSpec` with `TraceRate = nil`
- WHEN `spikeWindow()` is called
- THEN the returned `ActiveWindow.TraceRate` MUST be `nil` (no panic)
- MECHANISM: Direct field copy; Go handles nil pointer assignment

**BC-8: Division by Population Safety**
- GIVEN a cohort with `Population = 7` and spike `TraceRate = 7.6`
- WHEN cohort expansion divides `*TraceRate / float64(Population)`
- THEN the division MUST produce `1.086` without panic
- MECHANISM: `Population` validated > 0 during `Validate()`; safe to use as denominator

### C) Component Interaction

```
┌──────────────────┐
│ WorkloadSpec     │ (YAML input)
│ - Cohorts        │
│   - Spike        │
│     - TraceRate  │ ← NEW FIELD
└────────┬─────────┘
         │ ExpandCohorts()
         ▼
┌──────────────────┐
│ ClientSpec       │ (per-client)
│ - Lifecycle      │
│   - Windows      │
│     - TraceRate  │ ← PROPAGATED
└────────┬─────────┘
         │ GenerateRequests()
         ▼
┌──────────────────┐
│ generator.go     │
│ - computeProp... │ (uses TraceRate in absolute mode)
└──────────────────┘
```

**API Contracts:**
- `SpikeSpec.TraceRate *float64` (optional): Per-cohort rate for absolute mode; nil means proportional
- `spikeWindow(s *SpikeSpec) ActiveWindow`: Propagates `TraceRate` from spec to window
- `ExpandCohorts()`: Divides cohort-level `TraceRate` by `Population` for per-client windows
- `Validate()`: Rejects cohorts in absolute mode if spike lacks `trace_rate`

**State Changes:**
- None. This PR extends existing state with an optional field; no new mutable state introduced.

**Extension Friction:**
- Files to change: 2 (`spec.go`, `cohort.go`)
- To add one more lifecycle pattern with per-window rates: 3 lines (add field to pattern spec, propagate in pattern→window function, update validation)
- Touch-point multiplier: 1.5 (acceptable for feature completion)

### D) Deviation Log

No deviations from source document. Issue #1225 accurately describes the current code state and proposed changes.

### E) Review Guide

**THE TRICKY PART:** The per-client rate division logic in cohort expansion. Verify that `*window.TraceRate / float64(cohort.Population)` is applied BEFORE appending the window to the client, not after. The window must be copied/modified, not shared across clients.

**WHAT TO SCRUTINIZE:**
1. BC-1: Does each expanded client get `cohort_rate / population`? (Test with 7-member cohort, 7.6 rate)
2. BC-5: Validation correctly rejects cohorts in absolute mode without spike `trace_rate`?
3. Backward compat: Existing cohort specs (without `trace_rate`) still work in proportional mode?

**WHAT'S SAFE TO SKIM:**
- Struct field addition (standard Go YAML pattern)
- `spikeWindow()` field copy (mechanical propagation)

**KNOWN DEBT:**
- Diurnal and Drain patterns don't yet support per-window rates (not blocked, just incomplete)
- Pre-existing issue: `RateFraction` at client level vs cohort level is confusing (not addressed here)

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- None

**Files to modify:**
- `sim/workload/spec.go:92-95` — Add `TraceRate *float64` field to `SpikeSpec`
- `sim/workload/spec.go:287-289` — Update validation to allow cohorts in absolute mode with spike `trace_rate`
- `sim/workload/cohort.go:122-127` — Propagate `TraceRate` in `spikeWindow()`
- `sim/workload/cohort.go:57-58` — Divide cohort-level `trace_rate` by population during expansion
- `sim/workload/cohort_test.go` — Add tests for BC-1, BC-2, BC-3, BC-5

**Key decisions:**
- Use `*float64` for `TraceRate` to distinguish "not set" (nil) from "set to zero" (R9)
- Division happens during cohort expansion (not in `spikeWindow()`) to keep `spikeWindow()` stateless
- Validation allows Diurnal/Drain in absolute mode (not yet supported, but not blocked)

**Confirmation:**
- No dead code: `TraceRate` field used by generator.go:951 (existing code)
- All paths exercisable: Tests cover both nil and non-nil `TraceRate`, both proportional and absolute mode

### G) Task Breakdown

#### Task 1: Add TraceRate Field to SpikeSpec

**Contracts Implemented:** BC-3, BC-7

**Files:**
- Modify: `sim/workload/spec.go:92-95`
- Test: `sim/workload/cohort_test.go`

**Step 1: Write failing test for BC-3 (YAML parsing)**

Context: Verify that `trace_rate` field in YAML correctly unmarshals into `SpikeSpec.TraceRate` as a non-nil pointer.

```go
func TestSpikeSpec_YAMLParsing_WithTraceRate_ParsesCorrectly(t *testing.T) {
	yamlStr := `
spike:
  start_time_us: 300000000
  duration_us: 600000000
  trace_rate: 7.6
`
	var result struct {
		Spike *SpikeSpec `yaml:"spike"`
	}
	err := yaml.Unmarshal([]byte(yamlStr), &result)
	if err != nil {
		t.Fatalf("failed to parse YAML: %v", err)
	}
	if result.Spike == nil {
		t.Fatal("Spike is nil")
	}
	if result.Spike.TraceRate == nil {
		t.Fatal("TraceRate is nil; expected non-nil pointer")
	}
	if *result.Spike.TraceRate != 7.6 {
		t.Errorf("TraceRate = %v; expected 7.6", *result.Spike.TraceRate)
	}
}

func TestSpikeSpec_YAMLParsing_WithoutTraceRate_IsNil(t *testing.T) {
	yamlStr := `
spike:
  start_time_us: 300000000
  duration_us: 600000000
`
	var result struct {
		Spike *SpikeSpec `yaml:"spike"`
	}
	err := yaml.Unmarshal([]byte(yamlStr), &result)
	if err != nil {
		t.Fatalf("failed to parse YAML: %v", err)
	}
	if result.Spike == nil {
		t.Fatal("Spike is nil")
	}
	if result.Spike.TraceRate != nil {
		t.Errorf("TraceRate = %v; expected nil (omitempty)", *result.Spike.TraceRate)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestSpikeSpec_YAMLParsing -v`
Expected: FAIL with "unknown field trace_rate" or "TraceRate is nil"

**Step 3: Implement minimal code to satisfy contract**

Context: Add `TraceRate *float64` field to `SpikeSpec` struct with yaml tag `trace_rate,omitempty`.

In `sim/workload/spec.go:92-95`:
```go
// SpikeSpec configures a traffic spike as a lifecycle window.
// Clients are active during [StartTimeUs, StartTimeUs+DurationUs).
type SpikeSpec struct {
	StartTimeUs int64    `yaml:"start_time_us"`
	DurationUs  int64    `yaml:"duration_us"`
	TraceRate   *float64 `yaml:"trace_rate,omitempty"` // Cohort-level rate for absolute mode
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestSpikeSpec_YAMLParsing -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/spec.go sim/workload/cohort_test.go
git commit -m "feat(workload): add TraceRate field to SpikeSpec (BC-3, BC-7)

- Add TraceRate *float64 to SpikeSpec for cohort-level rates
- YAML tag: trace_rate,omitempty (nil means not set)
- Tests verify parsing with and without trace_rate field

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Propagate TraceRate in spikeWindow()

**Contracts Implemented:** BC-7

**Files:**
- Modify: `sim/workload/cohort.go:122-127`
- Test: `sim/workload/cohort_test.go`

**Step 1: Write failing test for TraceRate propagation**

Context: Verify that `spikeWindow()` copies `TraceRate` from `SpikeSpec` to `ActiveWindow`.

```go
func TestSpikeWindow_WithTraceRate_PropagatesField(t *testing.T) {
	rate := 7.6
	spec := &SpikeSpec{
		StartTimeUs: 300000000,
		DurationUs:  600000000,
		TraceRate:   &rate,
	}
	window := spikeWindow(spec)
	if window.TraceRate == nil {
		t.Fatal("ActiveWindow.TraceRate is nil; expected propagated value")
	}
	if *window.TraceRate != 7.6 {
		t.Errorf("ActiveWindow.TraceRate = %v; expected 7.6", *window.TraceRate)
	}
	if window.StartUs != 300000000 {
		t.Errorf("StartUs = %v; expected 300000000", window.StartUs)
	}
	if window.EndUs != 900000000 {
		t.Errorf("EndUs = %v; expected 900000000", window.EndUs)
	}
}

func TestSpikeWindow_WithoutTraceRate_LeavesNil(t *testing.T) {
	spec := &SpikeSpec{
		StartTimeUs: 300000000,
		DurationUs:  600000000,
		TraceRate:   nil,
	}
	window := spikeWindow(spec)
	if window.TraceRate != nil {
		t.Errorf("ActiveWindow.TraceRate = %v; expected nil", *window.TraceRate)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestSpikeWindow -v`
Expected: FAIL with "ActiveWindow.TraceRate is nil"

**Step 3: Implement minimal code to satisfy contract**

Context: Update `spikeWindow()` to copy `TraceRate` field from `SpikeSpec` to `ActiveWindow`.

In `sim/workload/cohort.go:122-127`:
```go
// spikeWindow creates a single lifecycle window for a traffic spike.
func spikeWindow(s *SpikeSpec) ActiveWindow {
	return ActiveWindow{
		StartUs:   s.StartTimeUs,
		EndUs:     s.StartTimeUs + s.DurationUs,
		TraceRate: s.TraceRate, // Propagate cohort-level rate
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestSpikeWindow -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/cohort.go sim/workload/cohort_test.go
git commit -m "feat(workload): propagate TraceRate in spikeWindow() (BC-7)

- Copy SpikeSpec.TraceRate to ActiveWindow.TraceRate
- Nil safety: nil pointer assignment is safe in Go
- Tests verify both nil and non-nil cases

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Divide TraceRate by Population During Cohort Expansion

**Contracts Implemented:** BC-1, BC-8

**Files:**
- Modify: `sim/workload/cohort.go:57-58`
- Test: `sim/workload/cohort_test.go`

**Step 1: Write failing test for per-client rate division**

Context: Verify that cohort expansion divides `TraceRate` by population so each client gets its proportional share.

```go
func TestExpandCohorts_SpikeTraceRate_DividedByPopulation(t *testing.T) {
	rate := 7.6
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 0, // Absolute mode
		Cohorts: []CohortSpec{
			{
				ID:           "midnight-critical",
				Population:   7,
				RateFraction: 0.089,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Spike:        &SpikeSpec{StartTimeUs: 300000000, DurationUs: 600000000, TraceRate: &rate},
			},
		},
	}
	expanded := ExpandCohorts(spec, 12345)
	if len(expanded) != 7 {
		t.Fatalf("expected 7 clients; got %d", len(expanded))
	}
	// Each client should get 7.6 / 7 = 1.086 (approximately)
	expectedRate := 7.6 / 7.0
	for i, client := range expanded {
		if client.Lifecycle == nil || len(client.Lifecycle.Windows) == 0 {
			t.Fatalf("client %d has no lifecycle windows", i)
		}
		window := client.Lifecycle.Windows[0]
		if window.TraceRate == nil {
			t.Fatalf("client %d window TraceRate is nil", i)
		}
		actualRate := *window.TraceRate
		// Allow floating point tolerance
		if actualRate < 1.085 || actualRate > 1.087 {
			t.Errorf("client %d: TraceRate = %v; expected ~1.086 (7.6/7)", i, actualRate)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestExpandCohorts_SpikeTraceRate -v`
Expected: FAIL with "client 0: TraceRate = 7.6; expected ~1.086"

**Step 3: Implement minimal code to satisfy contract**

Context: During cohort expansion, after creating the spike window, divide its `TraceRate` by population.

In `sim/workload/cohort.go:57-58`:
```go
		// Build lifecycle windows from cohort patterns
		var windows []ActiveWindow
		if cohort.Diurnal != nil {
			windows = append(windows, diurnalWindows(cohort.Diurnal, cohortRNG)...)
		}
		if cohort.Spike != nil {
			window := spikeWindow(cohort.Spike)

			// Divide cohort-level trace_rate by population for per-client rate
			if window.TraceRate != nil {
				perClientRate := *window.TraceRate / float64(cohort.Population)
				window.TraceRate = &perClientRate
			}

			windows = append(windows, window)
		}
		if cohort.Drain != nil {
			windows = append(windows, drainWindows(cohort.Drain)...)
		}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestExpandCohorts_SpikeTraceRate -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/cohort.go sim/workload/cohort_test.go
git commit -m "feat(workload): divide spike TraceRate by population (BC-1, BC-8)

- During cohort expansion, each client gets cohort_rate / population
- Example: 7-member cohort with trace_rate=7.6 → each client gets 1.086
- Division is safe: Population validated > 0 during Validate()

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Update Validation to Allow Cohorts in Absolute Mode

**Contracts Implemented:** BC-5, BC-6

**Files:**
- Modify: `sim/workload/spec.go:287-289`
- Test: `sim/workload/cohort_test.go`

**Step 1: Write failing test for validation**

Context: Verify that cohorts with spike `trace_rate` pass validation in absolute mode, and cohorts without `trace_rate` are rejected.

```go
func TestValidation_AbsoluteMode_CohortWithSpikeTraceRate_Passes(t *testing.T) {
	rate := 7.6
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 0, // Absolute mode
		Cohorts: []CohortSpec{
			{
				ID:           "midnight-critical",
				Population:   7,
				RateFraction: 0.089,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Spike:        &SpikeSpec{StartTimeUs: 300000000, DurationUs: 600000000, TraceRate: &rate},
			},
		},
	}
	err := spec.Validate()
	if err != nil {
		t.Errorf("expected validation to pass; got error: %v", err)
	}
}

func TestValidation_AbsoluteMode_CohortWithoutSpikeTraceRate_Fails(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 0, // Absolute mode
		Cohorts: []CohortSpec{
			{
				ID:           "test",
				Population:   7,
				RateFraction: 0.089,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Spike:        &SpikeSpec{StartTimeUs: 300000000, DurationUs: 600000000}, // No TraceRate
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected validation to fail for cohort without spike trace_rate in absolute mode")
	}
	if !strings.Contains(err.Error(), "spike without trace_rate") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestValidation_AbsoluteMode_CohortWithDiurnal_Passes(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 0, // Absolute mode
		Cohorts: []CohortSpec{
			{
				ID:           "test",
				Population:   5,
				RateFraction: 0.089,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Diurnal:      &DiurnalSpec{PeakHour: 14, PeakToTroughRatio: 2.0}, // No trace_rate yet
			},
		},
	}
	err := spec.Validate()
	if err != nil {
		t.Errorf("expected validation to pass for diurnal (not yet supported, but not blocked); got error: %v", err)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestValidation_AbsoluteMode_Cohort -v`
Expected: FAIL with "expected validation to pass" and "expected validation to fail"

**Step 3: Implement minimal code to satisfy contract**

Context: Replace the blanket "cohorts not supported in absolute mode" error with targeted validation that checks for spike `trace_rate`.

In `sim/workload/spec.go:287-289`:
```go
		}
		// Cohorts in absolute mode require spike.trace_rate
		if len(s.Cohorts) > 0 {
			for i, cohort := range s.Cohorts {
				if cohort.Spike != nil && cohort.Spike.TraceRate == nil {
					return fmt.Errorf("aggregate_rate is 0 (absolute rate mode) but cohort %d has spike without trace_rate", i)
				}
				// Diurnal/Drain patterns in absolute mode: not yet supported, but not blocked
			}
		}
	} else {
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestValidation_AbsoluteMode_Cohort -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/spec.go sim/workload/cohort_test.go
git commit -m "feat(workload): allow cohorts in absolute mode with spike trace_rate (BC-5, BC-6)

- Validation requires spike.trace_rate when aggregate_rate=0
- Diurnal/Drain patterns not blocked (not yet supported, but not rejected)
- Tests verify rejection without trace_rate, acceptance with it

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Integration Test - End-to-End Absolute Mode Flow

**Contracts Implemented:** BC-2, BC-4

**Files:**
- Test: `sim/workload/generator_test.go`

**Step 1: Write integration test for absolute mode**

Context: Verify that the full pipeline (YAML → expand → generate) produces correct arrival rates in absolute mode.

```go
func TestGenerateRequests_CohortAbsoluteMode_UsesTraceRate(t *testing.T) {
	rate := 7.6
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 0, // Absolute mode
		Cohorts: []CohortSpec{
			{
				ID:           "midnight-critical",
				Population:   7,
				RateFraction: 0.089,
				Arrival:      ArrivalSpec{Process: "constant"}, // Deterministic for testing
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Spike:        &SpikeSpec{StartTimeUs: 0, DurationUs: 10000000, TraceRate: &rate}, // 10 seconds
			},
		},
	}
	// Expand cohorts
	expanded := ExpandCohorts(spec, 12345)
	if len(expanded) != 7 {
		t.Fatalf("expected 7 clients; got %d", len(expanded))
	}
	// Generate requests for the first client
	spec.Clients = []ClientSpec{expanded[0]}
	requests, err := GenerateRequests(spec, 10000000, 67890) // 10 second horizon
	if err != nil {
		t.Fatalf("GenerateRequests failed: %v", err)
	}
	// Each client should generate ~1.086 req/s * 10s = ~10.86 requests
	// With constant arrival, expect exactly 10 or 11 requests
	if len(requests) < 10 || len(requests) > 11 {
		t.Errorf("expected ~10-11 requests (1.086 req/s * 10s); got %d", len(requests))
	}
}

func TestGenerateRequests_CohortProportionalMode_BackwardCompatible(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0, // Proportional mode
		Cohorts: []CohortSpec{
			{
				ID:           "test",
				Population:   5,
				RateFraction: 0.5, // 5 req/s shared among 5 clients = 1 req/s each
				Arrival:      ArrivalSpec{Process: "constant"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Spike:        &SpikeSpec{StartTimeUs: 0, DurationUs: 10000000}, // No TraceRate
			},
		},
	}
	expanded := ExpandCohorts(spec, 12345)
	if len(expanded) != 5 {
		t.Fatalf("expected 5 clients; got %d", len(expanded))
	}
	// Generate requests for the first client
	spec.Clients = []ClientSpec{expanded[0]}
	requests, err := GenerateRequests(spec, 10000000, 67890)
	if err != nil {
		t.Fatalf("GenerateRequests failed: %v", err)
	}
	// Each client: 10.0 * (0.5 / 5) / 1.0 = 1 req/s * 10s = ~10 requests
	if len(requests) < 9 || len(requests) > 11 {
		t.Errorf("expected ~10 requests (1 req/s * 10s); got %d", len(requests))
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestGenerateRequests_Cohort -v`
Expected: PASS (implementation already complete from previous tasks)

**Step 3: Lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 4: Commit with contract reference**

```bash
git add sim/workload/generator_test.go
git commit -m "test(workload): end-to-end cohort absolute mode flow (BC-2, BC-4)

- Integration test verifies full pipeline: YAML → expand → generate
- Absolute mode: 7-member cohort with trace_rate=7.6 → 10-11 requests in 10s
- Backward compat: proportional mode cohort without trace_rate still works

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1     | Task 3 | Unit | TestExpandCohorts_SpikeTraceRate_DividedByPopulation |
| BC-2     | Task 5 | Integration | TestGenerateRequests_CohortAbsoluteMode_UsesTraceRate |
| BC-3     | Task 1 | Unit | TestSpikeSpec_YAMLParsing_WithTraceRate_ParsesCorrectly |
| BC-4     | Task 5 | Integration | TestGenerateRequests_CohortProportionalMode_BackwardCompatible |
| BC-5     | Task 4 | Unit | TestValidation_AbsoluteMode_CohortWithoutSpikeTraceRate_Fails |
| BC-6     | Task 4 | Unit | TestValidation_AbsoluteMode_CohortWithDiurnal_Passes |
| BC-7     | Task 1, 2 | Unit | TestSpikeSpec_YAMLParsing_WithoutTraceRate_IsNil, TestSpikeWindow_WithoutTraceRate_LeavesNil |
| BC-8     | Task 3 | Unit | TestExpandCohorts_SpikeTraceRate_DividedByPopulation (implicitly tests division safety) |

**Shared test infrastructure:** None needed; uses existing `workload` package test utilities.

**Golden dataset updates:** Not applicable (no golden datasets modified).

**Lint requirements:** `golangci-lint run ./sim/workload/...` must pass with zero new issues.

**Test naming convention:** `TestComponent_Scenario_Behavior` (BDD-style).

**Test isolation:** All tests are independently runnable with table-driven patterns where applicable.

**Invariant tests:** Not applicable (no system invariants modified; this is a workload generation change).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|------------|--------|------------|------|
| Division by zero if Population=0 | Low | High (panic) | Population validated > 0 during Validate() (pre-existing) | Task 3 |
| Floating point precision issues in rate division | Low | Low | Use tolerance in integration test (~1.085-1.087) | Task 5 |
| Backward compatibility breakage for existing cohort specs | Medium | High | Tests verify nil TraceRate behavior; field is optional (omitempty) | Task 1, 4, 5 |
| Validation allows Diurnal/Drain in absolute mode but they don't work | Low | Medium | Comment in validation explains "not yet supported, but not blocked" | Task 4 |

---

## PART 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — Uses existing `*float64` pattern (R9)
- [x] No feature creep — Strictly implements issue #1225 scope
- [x] No unexercised flags — All code paths covered by tests
- [x] No partial implementations — Spike pattern fully supported; Diurnal/Drain explicitly not blocked
- [x] No breaking changes — Backward compatible (optional field, validation relaxed)
- [x] No hidden global state impact — Stateless transformation
- [x] All new code will pass golangci-lint — No linter-prone patterns introduced
- [x] Shared test helpers — Not needed (simple unit tests)
- [x] CLAUDE.md updated — Will update in final commit (add to Recent Changes section)
- [x] No stale references in CLAUDE.md — No references to remove
- [x] Documentation DRY — No canonical sources modified
- [x] Deviation log reviewed — No deviations
- [x] Each task produces working, testable code — Every task ends with passing tests
- [x] Task dependencies correctly ordered — Linear dependency: 1→2→3→4→5
- [x] All contracts mapped to tasks — See Test Strategy section
- [x] Golden dataset regeneration documented — Not applicable
- [x] Construction site audit — Not applicable (no struct fields added to existing constructors)
- [x] Macro plan status update — Issue #1225 is standalone; no macro plan dependency

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — Not applicable (no loops with early exit)
- [x] R2: Map keys sorted — Not applicable (no map iteration)
- [x] R3: Numeric parameter validation — `TraceRate` validated during Validate() (must be non-nil for spike in absolute mode)
- [x] R4: Construction site audit — Not applicable (no new struct construction sites)
- [x] R5: Resource allocation rollback — Not applicable (no multi-step resource allocation)
- [x] R6: No `logrus.Fatalf` in `sim/` — Not applicable (no error handling added)
- [x] R7: Invariant tests alongside golden — Not applicable (no golden tests added)
- [x] R8: No exported mutable maps — Not applicable (no maps added)
- [x] R9: `*float64` for YAML fields — ✅ Used `*float64` for `TraceRate`
- [x] R10: YAML strict parsing — Not applicable (existing YAML parsing unchanged)
- [x] R11: Division guards — ✅ Population validated > 0 (pre-existing); safe denominator
- [x] R12: Golden dataset regenerated — Not applicable
- [x] R13: Interfaces work for 2+ implementations — Not applicable (no interfaces added)
- [x] R14: No method spans multiple modules — Not applicable (no new methods)
- [x] R15: Stale PR references — Not applicable (no references to resolve)
- [x] R16: Config params grouped — Not applicable (no config changes)
- [x] R17: Routing scorer signals — Not applicable (no routing changes)
- [x] R18: CLI flag overrides — Not applicable (no CLI flags)
- [x] R19: Unbounded retry loops — Not applicable (no retry logic)
- [x] R20: Degenerate input handling — Validation checks nil `TraceRate` in absolute mode
- [x] R21: No `range` over shrinking slices — Not applicable (no slice mutation during iteration)
- [x] R22: Pre-check consistency — Not applicable (no pre-checks)
- [x] R23: Parallel path equivalence — Not applicable (no parallel code paths)

---

## APPENDIX: File-Level Implementation Details

### File: `sim/workload/spec.go`

**Purpose:** Define workload spec types with YAML unmarshaling.

**Changes:**

1. **Lines 92-95 (SpikeSpec):**

```go
// SpikeSpec configures a traffic spike as a lifecycle window.
// Clients are active during [StartTimeUs, StartTimeUs+DurationUs).
type SpikeSpec struct {
	StartTimeUs int64    `yaml:"start_time_us"`
	DurationUs  int64    `yaml:"duration_us"`
	TraceRate   *float64 `yaml:"trace_rate,omitempty"` // Cohort-level rate for absolute mode (req/s)
}
```

**Rationale:** `*float64` allows distinguishing "not set" (nil) from "set to zero" (R9). The `omitempty` tag allows backward compatibility (existing specs without `trace_rate` parse correctly).

2. **Lines 287-293 (Validation for absolute mode cohorts):**

```go
		}
		// Cohorts in absolute mode require spike.trace_rate
		if len(s.Cohorts) > 0 {
			for i, cohort := range s.Cohorts {
				if cohort.Spike != nil && cohort.Spike.TraceRate == nil {
					return fmt.Errorf("aggregate_rate is 0 (absolute rate mode) but cohort %d has spike without trace_rate", i)
				}
				// Diurnal/Drain patterns in absolute mode: not yet supported, but not blocked
			}
		}
	} else {
		// Normal proportional mode: aggregate_rate must be positive
```

**Rationale:** Replaces blanket "cohorts not supported in absolute mode" error. Allows cohorts with spike `trace_rate` while rejecting cohorts without it. Diurnal/Drain are allowed (validation doesn't block future work).

---

### File: `sim/workload/cohort.go`

**Purpose:** Cohort expansion logic (population → individual clients).

**Changes:**

1. **Lines 122-127 (spikeWindow):**

```go
// spikeWindow creates a single lifecycle window for a traffic spike.
func spikeWindow(s *SpikeSpec) ActiveWindow {
	return ActiveWindow{
		StartUs:   s.StartTimeUs,
		EndUs:     s.StartTimeUs + s.DurationUs,
		TraceRate: s.TraceRate, // Propagate cohort-level rate to window
	}
}
```

**Rationale:** Mechanical propagation of optional field. Nil safety is guaranteed by Go semantics (nil pointer assignment is safe).

2. **Lines 57-68 (cohort expansion with per-client division):**

```go
		// Build lifecycle windows from cohort patterns
		var windows []ActiveWindow
		if cohort.Diurnal != nil {
			windows = append(windows, diurnalWindows(cohort.Diurnal, cohortRNG)...)
		}
		if cohort.Spike != nil {
			window := spikeWindow(cohort.Spike)

			// Divide cohort-level trace_rate by population for per-client rate
			if window.TraceRate != nil {
				perClientRate := *window.TraceRate / float64(cohort.Population)
				window.TraceRate = &perClientRate
			}

			windows = append(windows, window)
		}
		if cohort.Drain != nil {
			windows = append(windows, drainWindows(cohort.Drain)...)
		}
```

**Rationale:** Division by population ensures each expanded client gets `cohort_rate / N`. Division is safe because `Population` is validated > 0 during `Validate()` (pre-existing check at spec.go:213). The window is copied before modification, so each client gets an independent `TraceRate` pointer.

**Behavioral notes:**
- Division only happens when `TraceRate != nil` (spike windows without `trace_rate` are unmodified)
- Result is stored in a new pointer (`&perClientRate`), not the original cohort-level pointer
- Consistent with existing `RateFraction` division at cohort.go:26

---

### File: `sim/workload/generator.go`

**Purpose:** Generate requests from workload spec (existing code, no changes).

**Relevant existing behavior (lines 947-953):**

```go
// Absolute rate mode: when aggregate_rate is 0, use trace_rate directly.
// This signals "use per-window rates verbatim, don't scale". Useful for
// workloads with time-varying aggregate load that cannot be represented
// by a single scalar aggregate_rate.
if aggregateRate == 0 && window.TraceRate != nil {
	return traceRate
}
```

**Why no changes needed:** The generator already correctly handles `ActiveWindow.TraceRate` in absolute mode. This PR simply ensures that spike windows can populate that field.

---

