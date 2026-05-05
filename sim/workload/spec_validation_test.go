package workload

import (
	"strings"
	"testing"
)

func TestWorkloadSpec_Validate_AbsoluteRateMode(t *testing.T) {
	// Helper to create valid base spec
	validBaseSpec := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version:       "2",
			AggregateRate: 0,
			Clients: []ClientSpec{
				{
					ID:           "client-1",
					Concurrency:  0,
					RateFraction: 1.0,
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					Arrival:      ArrivalSpec{Process: "poisson"},
				},
			},
		}
	}

	t.Run("valid absolute rate mode", func(t *testing.T) {
		traceRate := 5.0
		spec := validBaseSpec()
		spec.Clients[0].Lifecycle = &LifecycleSpec{
			Windows: []ActiveWindow{
				{StartUs: 0, EndUs: 1000000, TraceRate: &traceRate},
			},
		}
		if err := spec.Validate(); err != nil {
			t.Errorf("expected valid, got error: %v", err)
		}
	})

	t.Run("absolute mode with missing lifecycle", func(t *testing.T) {
		spec := validBaseSpec()
		spec.Clients[0].Lifecycle = nil
		err := spec.Validate()
		if err == nil {
			t.Fatal("expected error for missing lifecycle, got nil")
		}
		if !strings.Contains(err.Error(), "no lifecycle windows") {
			t.Errorf("expected 'no lifecycle windows' error, got: %v", err)
		}
	})

	t.Run("absolute mode with missing trace_rate", func(t *testing.T) {
		spec := validBaseSpec()
		spec.Clients[0].Lifecycle = &LifecycleSpec{
			Windows: []ActiveWindow{
				{StartUs: 0, EndUs: 1000000, TraceRate: nil},
			},
		}
		err := spec.Validate()
		if err == nil {
			t.Fatal("expected error for missing trace_rate, got nil")
		}
		if !strings.Contains(err.Error(), "no trace_rate") {
			t.Errorf("expected 'no trace_rate' error, got: %v", err)
		}
	})

	t.Run("absolute mode with cohorts", func(t *testing.T) {
		traceRate := 5.0
		spec := validBaseSpec()
		spec.Clients[0].Lifecycle = &LifecycleSpec{
			Windows: []ActiveWindow{
				{StartUs: 0, EndUs: 1000000, TraceRate: &traceRate},
			},
		}
		spec.Cohorts = []CohortSpec{
			{
				Population:   10,
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			},
		}
		err := spec.Validate()
		if err == nil {
			t.Fatal("expected error for cohorts in absolute mode, got nil")
		}
		if !strings.Contains(err.Error(), "cohorts are present") {
			t.Errorf("expected 'cohorts are present' error, got: %v", err)
		}
	})

	t.Run("concurrency clients allowed with aggregate_rate=0", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 0,
			Clients: []ClientSpec{
				{
					ID:          "client-1",
					Concurrency: 10,
					InputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					Arrival:     ArrivalSpec{Process: "poisson"},
				},
			},
		}
		// Should be valid - aggregate_rate=0 validation only applies to rate-based clients
		if err := spec.Validate(); err != nil {
			t.Errorf("expected valid for concurrency clients, got error: %v", err)
		}
	})
}
