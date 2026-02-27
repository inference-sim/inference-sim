package workload

import (
	"reflect"
	"strings"
	"testing"
)

func TestCohortValidation_ZeroPopulation_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Cohorts: []CohortSpec{
			{
				ID: "test", Population: 0, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for zero population cohort")
	}
	if !strings.Contains(err.Error(), "population must be positive") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCohortValidation_PeakToTroughLessThanOne_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Cohorts: []CohortSpec{
			{
				ID: "test", Population: 5, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Diurnal: &DiurnalSpec{PeakHour: 14, PeakToTroughRatio: 0.5},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for peak_to_trough_ratio < 1.0")
	}
	if !strings.Contains(err.Error(), "peak_to_trough_ratio must be") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCohortValidation_ValidCohort_NoError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Cohorts: []CohortSpec{
			{
				ID: "valid", Population: 10, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Diurnal: &DiurnalSpec{PeakHour: 14, PeakToTroughRatio: 4.0},
			},
		},
	}
	err := spec.Validate()
	if err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestCohortValidation_DrainZeroRamp_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Cohorts: []CohortSpec{
			{
				ID: "test", Population: 5, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Drain: &DrainSpec{StartTimeUs: 1000000, RampDurationUs: 0},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for zero drain ramp duration")
	}
}

// --- Expansion tests ---

func TestExpandCohorts_Determinism_SameSeedSameOutput(t *testing.T) {
	// GIVEN a cohort spec with seed=42
	cohorts := []CohortSpec{
		{
			ID: "workers", Population: 5, RateFraction: 1.0,
			TenantID: "t1", SLOClass: "standard",
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		},
	}

	// WHEN expanded twice with the same seed
	result1 := ExpandCohorts(cohorts, 42)
	result2 := ExpandCohorts(cohorts, 42)

	// THEN both expansions produce byte-identical client lists
	if !reflect.DeepEqual(result1, result2) {
		t.Fatal("cohort expansion is not deterministic: two expansions with same seed differ")
	}
	if len(result1) != 5 {
		t.Errorf("expected 5 expanded clients, got %d", len(result1))
	}
}

func TestExpandCohorts_ClientIDsUnique(t *testing.T) {
	cohorts := []CohortSpec{
		{
			ID: "group-a", Population: 3, RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		},
	}
	result := ExpandCohorts(cohorts, 42)
	ids := make(map[string]bool)
	for _, c := range result {
		if ids[c.ID] {
			t.Errorf("duplicate client ID: %s", c.ID)
		}
		ids[c.ID] = true
	}
}

func TestExpandCohorts_EmptyList_ReturnsEmpty(t *testing.T) {
	result := ExpandCohorts(nil, 42)
	if len(result) != 0 {
		t.Errorf("expected 0 clients, got %d", len(result))
	}
}

func TestExpandCohorts_FieldInheritance(t *testing.T) {
	cohorts := []CohortSpec{
		{
			ID: "inherited", Population: 2, RateFraction: 1.0,
			TenantID: "tenant-x", SLOClass: "critical", Model: "model-y",
			Arrival:   ArrivalSpec{Process: "constant"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			Streaming: true,
		},
	}
	result := ExpandCohorts(cohorts, 42)
	for _, c := range result {
		if c.TenantID != "tenant-x" {
			t.Errorf("client %s: TenantID = %q, want %q", c.ID, c.TenantID, "tenant-x")
		}
		if c.SLOClass != "critical" {
			t.Errorf("client %s: SLOClass = %q, want %q", c.ID, c.SLOClass, "critical")
		}
		if c.Model != "model-y" {
			t.Errorf("client %s: Model = %q, want %q", c.ID, c.Model, "model-y")
		}
		if !c.Streaming {
			t.Errorf("client %s: Streaming = false, want true", c.ID)
		}
	}
}

func TestExpandCohorts_DiurnalPattern_ProducesLifecycleWindows(t *testing.T) {
	cohorts := []CohortSpec{
		{
			ID: "diurnal", Population: 1, RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			Diurnal: &DiurnalSpec{PeakHour: 14, PeakToTroughRatio: 4.0},
		},
	}
	result := ExpandCohorts(cohorts, 42)
	if len(result) != 1 {
		t.Fatalf("expected 1 client, got %d", len(result))
	}
	if result[0].Lifecycle == nil {
		t.Fatal("expected lifecycle windows for diurnal cohort")
	}
	if len(result[0].Lifecycle.Windows) != 24 {
		t.Errorf("expected 24 hourly windows, got %d", len(result[0].Lifecycle.Windows))
	}
	// Verify peak hour window is larger than trough hour window
	peakWindow := result[0].Lifecycle.Windows[14]
	troughWindow := result[0].Lifecycle.Windows[2]
	peakDuration := peakWindow.EndUs - peakWindow.StartUs
	troughDuration := troughWindow.EndUs - troughWindow.StartUs
	if peakDuration <= troughDuration {
		t.Errorf("peak window duration (%d) should be > trough window duration (%d)", peakDuration, troughDuration)
	}
}

func TestExpandCohorts_SpikePattern_ProducesLifecycleWindow(t *testing.T) {
	cohorts := []CohortSpec{
		{
			ID: "spiked", Population: 1, RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			Spike: &SpikeSpec{StartTimeUs: 1_000_000, DurationUs: 500_000},
		},
	}
	result := ExpandCohorts(cohorts, 42)
	if result[0].Lifecycle == nil {
		t.Fatal("expected lifecycle for spike cohort")
	}
	// Should have a window covering [1_000_000, 1_500_000]
	found := false
	for _, w := range result[0].Lifecycle.Windows {
		if w.StartUs == 1_000_000 && w.EndUs == 1_500_000 {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected spike window [1000000, 1500000] not found in lifecycle")
	}
}

func TestExpandCohorts_DrainPattern_ProducesRampDownWindows(t *testing.T) {
	cohorts := []CohortSpec{
		{
			ID: "draining", Population: 1, RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			Drain: &DrainSpec{StartTimeUs: 1_000_000, RampDurationUs: 1_000_000},
		},
	}
	result := ExpandCohorts(cohorts, 42)
	if result[0].Lifecycle == nil {
		t.Fatal("expected lifecycle for drain cohort")
	}
	// Should have pre-drain window + 10 ramp segments = 11 windows
	if len(result[0].Lifecycle.Windows) != 11 {
		t.Errorf("expected 11 windows (1 pre-drain + 10 ramp segments), got %d", len(result[0].Lifecycle.Windows))
	}
	// Last ramp segment should be very short (approaching zero)
	lastWindow := result[0].Lifecycle.Windows[len(result[0].Lifecycle.Windows)-1]
	lastDuration := lastWindow.EndUs - lastWindow.StartUs
	firstRampWindow := result[0].Lifecycle.Windows[1]
	firstRampDuration := firstRampWindow.EndUs - firstRampWindow.StartUs
	if lastDuration >= firstRampDuration {
		t.Errorf("last ramp segment (%d) should be shorter than first (%d)", lastDuration, firstRampDuration)
	}
}

func TestGenerateRequests_WithCohorts_MergesWithExplicitClients(t *testing.T) {
	// GIVEN a spec with 1 explicit client + 1 cohort (population=3)
	spec := &WorkloadSpec{
		Version:       "2",
		Seed:          42,
		AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID: "explicit", RateFraction: 0.5,
				Arrival:   ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			},
		},
		Cohorts: []CohortSpec{
			{
				ID: "cohort", Population: 3, RateFraction: 0.5,
				Arrival:   ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			},
		},
	}

	// WHEN generating requests over a short horizon
	requests, err := GenerateRequests(spec, 10_000_000, 50)
	if err != nil {
		t.Fatalf("GenerateRequests failed: %v", err)
	}

	// THEN requests are generated (merged from explicit + cohort clients)
	if len(requests) == 0 {
		t.Fatal("expected non-empty request list")
	}
	// Requests must be sorted by arrival time (INV-5 causality)
	for i := 1; i < len(requests); i++ {
		if requests[i].ArrivalTime < requests[i-1].ArrivalTime {
			t.Errorf("requests not sorted: request[%d].ArrivalTime=%d < request[%d].ArrivalTime=%d",
				i, requests[i].ArrivalTime, i-1, requests[i-1].ArrivalTime)
		}
	}
}

func TestExpandCohorts_IndependentRNG_ReorderingDoesNotChange(t *testing.T) {
	cohortA := CohortSpec{
		ID: "a", Population: 3, RateFraction: 0.5,
		Arrival:   ArrivalSpec{Process: "poisson"},
		InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
		OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
	}
	cohortB := CohortSpec{
		ID: "b", Population: 2, RateFraction: 0.5,
		Arrival:   ArrivalSpec{Process: "constant"},
		InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 200, "std_dev": 20, "min": 1, "max": 400}},
		OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
	}

	// Expand in order [A, B]
	resultAB := ExpandCohorts([]CohortSpec{cohortA, cohortB}, 42)
	// Expand individually
	resultA := ExpandCohorts([]CohortSpec{cohortA}, 42)

	// Cohort A's expansion should be identical regardless of B's presence
	// (same index 0, same seed)
	for i := 0; i < 3; i++ {
		if resultAB[i].ID != resultA[i].ID {
			t.Errorf("client[%d]: ID differs between AB and A-only expansion", i)
		}
	}
}
