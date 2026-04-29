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

func TestWorkloadSpec_Validate_SLOClassConsistency(t *testing.T) {
	// Helper to create minimal valid spec
	validBaseSpec := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version:       "2",
			AggregateRate: 10.0,
			Clients: []ClientSpec{
				{
					ID:           "client-1",
					RateFraction: 1.0,
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				},
			},
		}
	}

	t.Run("BC-1: all-empty slo_class validates", func(t *testing.T) {
		// GIVEN all clients and cohorts have empty slo_class
		spec := validBaseSpec()
		spec.Clients[0].SLOClass = ""
		spec.Cohorts = []CohortSpec{
			{
				ID:           "cohort-1",
				Population:   5,
				RateFraction: 1.0,
				SLOClass:     "",
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			},
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it returns nil
		if err != nil {
			t.Errorf("expected valid, got error: %v", err)
		}
	})

	t.Run("BC-2: all-explicit slo_class validates", func(t *testing.T) {
		// GIVEN all clients and cohorts have non-empty slo_class
		spec := validBaseSpec()
		spec.Clients[0].SLOClass = "critical"
		spec.Cohorts = []CohortSpec{
			{
				ID:           "cohort-1",
				Population:   5,
				RateFraction: 1.0,
				SLOClass:     "standard",
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			},
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it returns nil
		if err != nil {
			t.Errorf("expected valid, got error: %v", err)
		}
	})

	t.Run("BC-3: zero cohorts with explicit clients validates", func(t *testing.T) {
		// GIVEN zero cohorts and all clients have explicit slo_class
		spec := validBaseSpec()
		spec.Clients[0].SLOClass = "critical"
		spec.Cohorts = []CohortSpec{} // empty cohort list
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it returns nil (empty cohorts not treated as "empty SLO class")
		if err != nil {
			t.Errorf("expected valid, got error: %v", err)
		}
	})

	t.Run("BC-4: mixed clients rejected with diagnostic error", func(t *testing.T) {
		// GIVEN one client with explicit slo_class and one with empty
		spec := validBaseSpec()
		spec.Clients[0].SLOClass = "critical"
		spec.Clients = append(spec.Clients, ClientSpec{
			ID:           "client-2",
			RateFraction: 1.0,
			SLOClass:     "", // empty
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
		})
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it returns error with diagnostic details
		if err == nil {
			t.Fatal("expected error for mixed slo_class, got nil")
		}
		errMsg := err.Error()
		if !strings.Contains(errMsg, "mixed slo_class specification") {
			t.Errorf("error missing 'mixed slo_class specification' phrase: %v", errMsg)
		}
		if !strings.Contains(errMsg, "1 have explicit values") {
			t.Errorf("error missing count of explicit clients: %v", errMsg)
		}
		if !strings.Contains(errMsg, "1 are empty") {
			t.Errorf("error missing count of empty clients: %v", errMsg)
		}
		if !strings.Contains(errMsg, "clients[0]") {
			t.Errorf("error missing identifier for explicit client: %v", errMsg)
		}
		if !strings.Contains(errMsg, "clients[1]") {
			t.Errorf("error missing identifier for empty client: %v", errMsg)
		}
	})

	t.Run("BC-4: mixed cohorts rejected with diagnostic error", func(t *testing.T) {
		// GIVEN all clients empty, one cohort explicit and one empty
		spec := validBaseSpec()
		spec.Clients[0].SLOClass = ""
		spec.Cohorts = []CohortSpec{
			{
				ID:           "cohort-1",
				Population:   5,
				RateFraction: 1.0,
				SLOClass:     "standard", // explicit
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			},
			{
				ID:           "cohort-2",
				Population:   5,
				RateFraction: 1.0,
				SLOClass:     "", // empty
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			},
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it returns error with diagnostic details including cohort identifiers
		if err == nil {
			t.Fatal("expected error for mixed cohort slo_class, got nil")
		}
		errMsg := err.Error()
		if !strings.Contains(errMsg, "mixed slo_class specification") {
			t.Errorf("error missing 'mixed slo_class specification' phrase: %v", errMsg)
		}
		if !strings.Contains(errMsg, "cohorts[0]") {
			t.Errorf("error missing identifier for explicit cohort: %v", errMsg)
		}
		if !strings.Contains(errMsg, "cohorts[1]") {
			t.Errorf("error missing identifier for empty cohort: %v", errMsg)
		}
	})

	t.Run("BC-4: mixed clients and cohorts rejected", func(t *testing.T) {
		// GIVEN one client explicit, one cohort empty
		spec := validBaseSpec()
		spec.Clients[0].SLOClass = "critical"
		spec.Cohorts = []CohortSpec{
			{
				ID:           "cohort-1",
				Population:   5,
				RateFraction: 1.0,
				SLOClass:     "", // empty
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			},
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it returns error mentioning both clients and cohorts
		if err == nil {
			t.Fatal("expected error for mixed clients+cohorts, got nil")
		}
		errMsg := err.Error()
		if !strings.Contains(errMsg, "clients[0]") {
			t.Errorf("error missing client identifier: %v", errMsg)
		}
		if !strings.Contains(errMsg, "cohorts[0]") {
			t.Errorf("error missing cohort identifier: %v", errMsg)
		}
	})

	t.Run("BC-5: InferencePerf expansion exempt from SLO class check", func(t *testing.T) {
		// GIVEN a spec with InferencePerf set (which will auto-generate clients with explicit SLOClass)
		// AND user-authored cohorts with empty SLOClass
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 10.0,
			Cohorts: []CohortSpec{
				{
					ID:           "cohort-1",
					Population:   5,
					RateFraction: 1.0,
					SLOClass:     "", // empty - would normally conflict with InferencePerf's "standard"
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				},
			},
			InferencePerf: &InferencePerfSpec{
				// InferencePerf expansion will create clients with SLOClass="standard"
				// But since InferencePerf is non-nil, the guard should skip checking spec.Clients
			},
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it should NOT return mixed specification error (InferencePerf clients are exempt)
		// (it may fail with other errors like missing InferencePerf fields, which is fine)
		if err != nil && strings.Contains(err.Error(), "mixed slo_class specification") {
			t.Errorf("BC-5 violation: InferencePerf should be exempt from SLO class check, but got mixed spec error: %v", err)
		}
	})

	t.Run("BC-5: ServeGen expansion exempt from SLO class check", func(t *testing.T) {
		// GIVEN a spec with ServeGenData set (which will auto-generate clients with explicit SLOClass)
		// AND user-authored cohorts with empty SLOClass
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 10.0,
			Cohorts: []CohortSpec{
				{
					ID:           "cohort-1",
					Population:   5,
					RateFraction: 1.0,
					SLOClass:     "", // empty - would normally conflict with ServeGen's "standard"
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				},
			},
			ServeGenData: &ServeGenDataSpec{
				// ServeGen expansion will create clients with SLOClass="standard"
				// But since ServeGenData is non-nil, the guard should skip checking spec.Clients
				Path: "dummy",
			},
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it should NOT return mixed specification error (ServeGen clients are exempt)
		// (it may fail with other errors like missing ServeGen files, which is fine)
		if err != nil && strings.Contains(err.Error(), "mixed slo_class specification") {
			t.Errorf("BC-5 violation: ServeGen should be exempt from SLO class check, but got mixed spec error: %v", err)
		}
	})

	t.Run("BC-8: pre-existing validation error takes precedence", func(t *testing.T) {
		// GIVEN a spec with BOTH a pre-existing error (invalid slo_class value)
		// AND a mixed specification (one explicit, one empty)
		spec := validBaseSpec()
		spec.Clients[0].SLOClass = "invalid_tier" // invalid value - will fail during validateClient
		spec.Clients = append(spec.Clients, ClientSpec{
			ID:           "client-2",
			RateFraction: 1.0,
			SLOClass:     "", // empty - creates mixed situation
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
		})
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it MUST return the pre-existing error (invalid slo_class), not the mixed spec error
		if err == nil {
			t.Fatal("expected validation error, got nil")
		}
		errMsg := err.Error()
		if !strings.Contains(errMsg, "unknown slo_class") {
			t.Errorf("BC-8 violation: expected pre-existing error (unknown slo_class) to fire first, got: %v", errMsg)
		}
		if strings.Contains(errMsg, "mixed slo_class specification") {
			t.Errorf("BC-8 violation: mixed spec error should not appear when pre-existing error exists, got: %v", errMsg)
		}
	})
}
