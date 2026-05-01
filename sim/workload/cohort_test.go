package workload

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

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
	expanded := ExpandCohorts(spec.Cohorts, 12345)
	if len(expanded) != 7 {
		t.Fatalf("expected 7 clients; got %d", len(expanded))
	}
	// Each client should get 7.6 / 7 = 1.086 (approximately)
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

// --- New-field tests (BC-1 through BC-7) ---

// BC-4: YAML round-trip with strict decoder preserves reasoning.multi_turn fields.
func TestCohortSpec_YAMLRoundTrip_Reasoning(t *testing.T) {
	input := `
version: "2"
aggregate_rate: 1.0
cohorts:
  - id: thinkers
    population: 1
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 100
        std_dev: 10
        min: 1
        max: 200
    output_distribution:
      type: gaussian
      params:
        mean: 50
        std_dev: 5
        min: 1
        max: 100
    reasoning:
      reason_ratio_distribution:
        type: constant
        params:
          value: 50
      multi_turn:
        max_rounds: 5
        think_time_us: 2000000
`
	var spec WorkloadSpec
	decoder := yaml.NewDecoder(strings.NewReader(input))
	decoder.KnownFields(true)
	if err := decoder.Decode(&spec); err != nil {
		t.Fatalf("YAML decode failed: %v", err)
	}
	if len(spec.Cohorts) != 1 {
		t.Fatalf("expected 1 cohort, got %d", len(spec.Cohorts))
	}
	c := spec.Cohorts[0]
	if c.Reasoning == nil {
		t.Fatal("Reasoning is nil after YAML decode")
	}
	if c.Reasoning.MultiTurn == nil {
		t.Fatal("Reasoning.MultiTurn is nil after YAML decode")
	}
	if c.Reasoning.MultiTurn.MaxRounds != 5 {
		t.Errorf("MaxRounds = %d, want 5", c.Reasoning.MultiTurn.MaxRounds)
	}
	if c.Reasoning.MultiTurn.ThinkTimeUs != 2_000_000 {
		t.Errorf("ThinkTimeUs = %d, want 2000000", c.Reasoning.MultiTurn.ThinkTimeUs)
	}
}

// BC-1 + BC-5: All 6 new fields propagate to every expanded ClientSpec.
func TestExpandCohorts_NewFieldsPropagate(t *testing.T) {
	closedLoopFalse := false
	timeoutVal := int64(60_000_000)
	cohorts := []CohortSpec{
		{
			ID: "agents", Population: 3, RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			PrefixLength: 64,
			Reasoning: &ReasoningSpec{
				MultiTurn: &MultiTurnSpec{MaxRounds: 4, ThinkTimeUs: 1_000_000},
			},
			ClosedLoop: &closedLoopFalse,
			Timeout:    &timeoutVal,
			Network:    &NetworkSpec{RTTMs: 2.5},
			Multimodal: &MultimodalSpec{
				TextDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 128}},
			},
		},
	}

	result := ExpandCohorts(cohorts, 42)
	if len(result) != 3 {
		t.Fatalf("expected 3 expanded clients, got %d", len(result))
	}

	for _, c := range result {
		if c.PrefixLength != 64 {
			t.Errorf("client %s: PrefixLength = %d, want 64", c.ID, c.PrefixLength)
		}
		if c.Reasoning == nil || c.Reasoning.MultiTurn == nil || c.Reasoning.MultiTurn.MaxRounds != 4 {
			t.Errorf("client %s: Reasoning.MultiTurn.MaxRounds not propagated", c.ID)
		}
		if c.ClosedLoop == nil || *c.ClosedLoop != false {
			t.Errorf("client %s: ClosedLoop not propagated as false", c.ID)
		}
		if c.Timeout == nil || *c.Timeout != 60_000_000 {
			t.Errorf("client %s: Timeout not propagated", c.ID)
		}
		if c.Network == nil || c.Network.RTTMs != 2.5 {
			t.Errorf("client %s: Network not propagated", c.ID)
		}
		if c.Multimodal == nil {
			t.Errorf("client %s: Multimodal is nil", c.ID)
		}
	}
}

// BC-6: nil/zero new fields leave expanded clients without accidental injection.
func TestExpandCohorts_NilNewFields_NoChange(t *testing.T) {
	cohorts := []CohortSpec{
		{
			ID: "simple", Population: 2, RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "constant"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			// PrefixLength, Reasoning, ClosedLoop, Timeout, Network, Multimodal all zero/nil
		},
	}

	result := ExpandCohorts(cohorts, 0)
	for _, c := range result {
		if c.PrefixLength != 0 {
			t.Errorf("client %s: PrefixLength = %d, want 0", c.ID, c.PrefixLength)
		}
		if c.Reasoning != nil {
			t.Errorf("client %s: Reasoning should be nil", c.ID)
		}
		if c.ClosedLoop != nil {
			t.Errorf("client %s: ClosedLoop should be nil", c.ID)
		}
		if c.Timeout != nil {
			t.Errorf("client %s: Timeout should be nil", c.ID)
		}
		if c.Network != nil {
			t.Errorf("client %s: Network should be nil", c.ID)
		}
		if c.Multimodal != nil {
			t.Errorf("client %s: Multimodal should be nil", c.ID)
		}
	}
}

// BC-2: reasoning.multi_turn.max_rounds < 1 → validation error.
func TestCohortValidation_InvalidMaxRounds_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Cohorts: []CohortSpec{
			{
				ID: "bad-rounds", Population: 1, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Reasoning: &ReasoningSpec{
					MultiTurn: &MultiTurnSpec{MaxRounds: 0, ThinkTimeUs: 1_000_000},
				},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for max_rounds=0")
	}
	if !strings.Contains(err.Error(), "max_rounds must be >= 1") {
		t.Errorf("unexpected error message: %v", err)
	}
}

// BC-3: timeout < 0 → validation error.
func TestCohortValidation_NegativeTimeout_ReturnsError(t *testing.T) {
	negTimeout := int64(-1)
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Cohorts: []CohortSpec{
			{
				ID: "bad-timeout", Population: 1, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Timeout: &negTimeout,
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for negative timeout")
	}
	if !strings.Contains(err.Error(), "timeout must be non-negative") {
		t.Errorf("unexpected error message: %v", err)
	}
}

// BC-7: prefix_length < 0 → validation error.
func TestCohortValidation_NegativePrefixLength_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Cohorts: []CohortSpec{
			{
				ID: "bad-prefix", Population: 1, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				PrefixLength: -1,
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for negative prefix_length")
	}
	if !strings.Contains(err.Error(), "prefix_length must be non-negative") {
		t.Errorf("unexpected error message: %v", err)
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

// --- Parity drift guard ---

// TestCohortClientSpecFieldParity ensures CohortSpec carries every field present
// on ClientSpec except the two intentionally excluded ones: ID (generated by
// ExpandCohorts per member) and Lifecycle (synthesized from Diurnal/Spike/Drain).
// If a new field is added to ClientSpec without mirroring it in CohortSpec,
// this test fails immediately, preventing the silent-drop bug class.
func TestCohortClientSpecFieldParity(t *testing.T) {
	cohortType := reflect.TypeOf(CohortSpec{})
	clientType := reflect.TypeOf(ClientSpec{})

	cohortFields := make(map[string]bool)
	for i := 0; i < cohortType.NumField(); i++ {
		cohortFields[cohortType.Field(i).Name] = true
	}

	// Fields on ClientSpec that are intentionally absent from CohortSpec.
	// ID: generated per member by ExpandCohorts.
	// Lifecycle: synthesized from Diurnal/Spike/Drain by ExpandCohorts.
	// Concurrency: client-level closed-loop mode; cohorts are rate-based.
	// ThinkTimeUs: paired with Concurrency; not applicable to cohorts.
	// CustomSamplerFactory: programmatic-only field, not exposed in YAML.
	excluded := map[string]bool{
		"ID":                   true,
		"Lifecycle":            true,
		"Concurrency":         true,
		"ThinkTimeUs":         true,
		"CustomSamplerFactory": true,
	}

	for i := 0; i < clientType.NumField(); i++ {
		name := clientType.Field(i).Name
		if excluded[name] {
			continue
		}
		if !cohortFields[name] {
			t.Errorf("ClientSpec.%s has no corresponding field in CohortSpec; add it or update the excluded list", name)
		}
	}
}

// --- End-to-end test: cohort with multi-turn reasoning produces multi-round requests ---

// TestGenerateRequests_CohortWithMultiTurn_ProducesMultiRoundRequests verifies
// the core purpose of this PR: a cohort with reasoning.multi_turn set produces
// requests with RoundIndex > 0 through the full GenerateRequests pipeline.
func TestGenerateRequests_CohortWithMultiTurn_ProducesMultiRoundRequests(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		Seed:          42,
		AggregateRate: 1.0,
		Cohorts: []CohortSpec{
			{
				ID: "agents", Population: 1, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Reasoning: &ReasoningSpec{
					ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}}, // 50% (0-100 scale; sampler does /100)
					MultiTurn:       &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 100_000, ContextGrowth: "accumulate"},
				},
			},
		},
	}

	// Use a long enough horizon to fit 3 rounds (3 × 100ms think time + generation time).
	requests, err := GenerateRequests(spec, 10_000_000_000, 0) // 10 seconds
	if err != nil {
		t.Fatalf("GenerateRequests failed: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from multi-turn cohort, got none")
	}

	// At least one request must have RoundIndex > 0 (i.e., a follow-up round).
	hasLaterRound := false
	for _, r := range requests {
		if r.RoundIndex > 0 {
			hasLaterRound = true
			break
		}
	}
	if !hasLaterRound {
		t.Errorf("no request with RoundIndex > 0 found; cohort multi-turn not reaching GenerateRequests reasoning path")
	}

	// INV-10: within each session, each round must arrive at least ThinkTimeUs
	// after the previous round (GenerateReasoningRequests advances time by
	// ThinkTimeUs + estimated output length per round).
	const thinkTimeUs = 100_000
	type sessionState struct {
		lastArrival    int64
		lastRound      int
		lastInputLen   int
	}
	sessions := make(map[string]*sessionState)
	for _, r := range requests {
		if r.SessionID == "" {
			continue
		}
		s, ok := sessions[r.SessionID]
		if !ok {
			sessions[r.SessionID] = &sessionState{
				lastArrival:  r.ArrivalTime,
				lastRound:    r.RoundIndex,
				lastInputLen: len(r.InputTokens),
			}
			continue
		}
		gap := r.ArrivalTime - s.lastArrival
		if gap < thinkTimeUs {
			t.Errorf("session %s: round %d→%d arrival gap %d µs < ThinkTimeUs %d (INV-10 violated)",
				r.SessionID, s.lastRound, r.RoundIndex, gap, thinkTimeUs)
		}
		// With context_growth: accumulate, each round's InputTokens must grow.
		if len(r.InputTokens) <= s.lastInputLen {
			t.Errorf("session %s: round %d input len %d <= round %d input len %d (context not accumulating)",
				r.SessionID, r.RoundIndex, len(r.InputTokens), s.lastRound, s.lastInputLen)
		}
		s.lastArrival = r.ArrivalTime
		s.lastRound = r.RoundIndex
		s.lastInputLen = len(r.InputTokens)
	}
}

func TestValidation_AbsoluteMode_CohortWithNegativeTraceRate_Fails(t *testing.T) {
	rate := -5.0
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
				Spike:        &SpikeSpec{StartTimeUs: 300000000, DurationUs: 600000000, TraceRate: &rate},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected validation to fail for negative trace_rate")
	}
	if !strings.Contains(err.Error(), "must be positive") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestValidation_AbsoluteMode_CohortWithNaNTraceRate_Fails(t *testing.T) {
	rate := math.NaN()
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
				Spike:        &SpikeSpec{StartTimeUs: 300000000, DurationUs: 600000000, TraceRate: &rate},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected validation to fail for NaN trace_rate")
	}
	if !strings.Contains(err.Error(), "must be a finite number") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestValidation_AbsoluteMode_CohortWithInfTraceRate_Fails(t *testing.T) {
	rate := math.Inf(1)
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
				Spike:        &SpikeSpec{StartTimeUs: 300000000, DurationUs: 600000000, TraceRate: &rate},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected validation to fail for Inf trace_rate")
	}
	if !strings.Contains(err.Error(), "must be a finite number") {
		t.Errorf("unexpected error message: %v", err)
	}
}
