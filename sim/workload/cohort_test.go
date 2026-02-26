package workload

import (
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
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "stdev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "stdev": 5, "min": 1, "max": 100}},
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
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "stdev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "stdev": 5, "min": 1, "max": 100}},
				Diurnal: &DiurnalSpec{PeakHour: 14, TroughHour: 2, PeakToTroughRatio: 0.5},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for peak_to_trough_ratio < 1.0")
	}
	if !strings.Contains(err.Error(), "peak_to_trough_ratio must be >= 1.0") {
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
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "stdev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "stdev": 5, "min": 1, "max": 100}},
				Diurnal: &DiurnalSpec{PeakHour: 14, TroughHour: 2, PeakToTroughRatio: 4.0},
			},
		},
	}
	err := spec.Validate()
	if err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestCohortValidation_SpikeInvalidMultiplier_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Cohorts: []CohortSpec{
			{
				ID: "test", Population: 5, RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "stdev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "stdev": 5, "min": 1, "max": 100}},
				Spike: &SpikeSpec{StartTimeUs: 1000000, DurationUs: 100000, Multiplier: -1.0},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for negative spike multiplier")
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
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "stdev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "stdev": 5, "min": 1, "max": 100}},
				Drain: &DrainSpec{StartTimeUs: 1000000, RampDurationUs: 0},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for zero drain ramp duration")
	}
}
