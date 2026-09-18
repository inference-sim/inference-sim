package workload

import (
	"strconv"
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

	t.Run("absolute mode with cohorts without spike trace_rate", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 0,
			Cohorts: []CohortSpec{
				{
					ID:           "cohort-1",
					Population:   10,
					RateFraction: 1.0,
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					Spike:        &SpikeSpec{StartTimeUs: 0, DurationUs: 1000000}, // No TraceRate
				},
			},
		}
		err := spec.Validate()
		if err == nil {
			t.Fatal("expected error for cohort spike without trace_rate in absolute mode, got nil")
		}
		if !strings.Contains(err.Error(), "spike without trace_rate") {
			t.Errorf("expected 'spike without trace_rate' error, got: %v", err)
		}
	})

	t.Run("absolute mode with cohorts with spike trace_rate", func(t *testing.T) {
		traceRate := 10.0
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 0,
			Cohorts: []CohortSpec{
				{
					ID:           "cohort-1",
					Population:   5,
					RateFraction: 0.5,
					Arrival:      ArrivalSpec{Process: "poisson"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					Spike:        &SpikeSpec{StartTimeUs: 0, DurationUs: 1000000, TraceRate: &traceRate},
				},
			},
		}
		err := spec.Validate()
		if err != nil {
			t.Errorf("expected valid for cohort with spike trace_rate in absolute mode, got error: %v", err)
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

	t.Run("all-empty slo_class validates", func(t *testing.T) {
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

	t.Run("all-explicit slo_class validates", func(t *testing.T) {
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

	t.Run("zero cohorts with explicit clients validates", func(t *testing.T) {
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

	t.Run("mixed clients rejected with diagnostic error", func(t *testing.T) {
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

	t.Run("mixed cohorts rejected with diagnostic error", func(t *testing.T) {
		// GIVEN no clients (empty slice), one cohort explicit and one empty
		spec := validBaseSpec()
		spec.Clients = []ClientSpec{} // no clients to avoid count confusion
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
		if !strings.Contains(errMsg, "1 have explicit values") {
			t.Errorf("error missing count of explicit cohorts: %v", errMsg)
		}
		if !strings.Contains(errMsg, "1 are empty") {
			t.Errorf("error missing count of empty cohorts: %v", errMsg)
		}
		if !strings.Contains(errMsg, "cohorts[0]") {
			t.Errorf("error missing identifier for explicit cohort: %v", errMsg)
		}
		if !strings.Contains(errMsg, "cohorts[1]") {
			t.Errorf("error missing identifier for empty cohort: %v", errMsg)
		}
	})

	t.Run("mixed clients and cohorts rejected", func(t *testing.T) {
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

	t.Run("error message truncates to 3 examples per category", func(t *testing.T) {
		// GIVEN 5 clients with explicit SLOClass and 4 cohorts with empty SLOClass
		spec := validBaseSpec()
		spec.Clients[0].SLOClass = "critical"
		// Add 4 more clients with explicit SLOClass
		for i := 1; i < 5; i++ {
			spec.Clients = append(spec.Clients, ClientSpec{
				ID:           "client-" + strconv.Itoa(i+1),
				RateFraction: 1.0,
				SLOClass:     "standard",
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			})
		}
		// Add 4 cohorts with empty SLOClass
		for i := 0; i < 4; i++ {
			spec.Cohorts = append(spec.Cohorts, CohortSpec{
				ID:           "cohort-" + strconv.Itoa(i),
				Population:   5,
				RateFraction: 1.0,
				SLOClass:     "", // empty
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			})
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN error shows at most 3 examples per category
		if err == nil {
			t.Fatal("expected error for mixed slo_class, got nil")
		}
		errMsg := err.Error()
		// Should show 5 explicit and 4 empty in counts
		if !strings.Contains(errMsg, "5 have explicit values") {
			t.Errorf("error should show count of 5 explicit: %v", errMsg)
		}
		if !strings.Contains(errMsg, "4 are empty") {
			t.Errorf("error should show count of 4 empty: %v", errMsg)
		}
		// Should show first 3 explicit examples: clients[0], clients[1], clients[2]
		if !strings.Contains(errMsg, "clients[0]") {
			t.Errorf("error should include clients[0] in explicit examples: %v", errMsg)
		}
		if !strings.Contains(errMsg, "clients[1]") {
			t.Errorf("error should include clients[1] in explicit examples: %v", errMsg)
		}
		if !strings.Contains(errMsg, "clients[2]") {
			t.Errorf("error should include clients[2] in explicit examples: %v", errMsg)
		}
		// Should NOT show clients[3] or clients[4] (truncated after 3)
		if strings.Contains(errMsg, "clients[3]") {
			t.Errorf("error should truncate explicit examples after 3, but found clients[3]: %v", errMsg)
		}
		if strings.Contains(errMsg, "clients[4]") {
			t.Errorf("error should truncate explicit examples after 3, but found clients[4]: %v", errMsg)
		}
		// Should show first 3 empty examples: cohorts[0], cohorts[1], cohorts[2]
		if !strings.Contains(errMsg, "cohorts[0]") {
			t.Errorf("error should include cohorts[0] in empty examples: %v", errMsg)
		}
		if !strings.Contains(errMsg, "cohorts[1]") {
			t.Errorf("error should include cohorts[1] in empty examples: %v", errMsg)
		}
		if !strings.Contains(errMsg, "cohorts[2]") {
			t.Errorf("error should include cohorts[2] in empty examples: %v", errMsg)
		}
		// Should NOT show cohorts[3] (truncated after 3)
		if strings.Contains(errMsg, "cohorts[3]") {
			t.Errorf("error should truncate empty examples after 3, but found cohorts[3]: %v", errMsg)
		}
	})

	t.Run("InferencePerf expansion exempt from SLO class check", func(t *testing.T) {
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
			t.Errorf("expected no mixed slo_class error when InferencePerf is set; got: %v", err)
		}
	})

	t.Run("ServeGen expansion exempt from SLO class check", func(t *testing.T) {
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
			t.Errorf("expected no mixed slo_class error when ServeGen is set; got: %v", err)
		}
	})

	t.Run("InferencePerf set does not exempt cohorts from SLO consistency check", func(t *testing.T) {
		// GIVEN a spec with InferencePerf set (clients exempted by guard)
		// AND user-authored cohorts with mixed SLOClass (one explicit, one empty)
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 10.0,
			Cohorts: []CohortSpec{
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
			},
			InferencePerf: &InferencePerfSpec{},
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it MUST return mixed specification error (cohorts are ALWAYS checked)
		// This documents the asymmetry: clients guarded by expansion check, cohorts always scanned
		if err == nil {
			t.Fatal("expected error for mixed cohort slo_class even with InferencePerf set, got nil")
		}
		if !strings.Contains(err.Error(), "mixed slo_class specification") {
			t.Errorf("expected 'mixed slo_class specification' error, got: %v", err)
		}
	})

	t.Run("ServeGen set does not exempt cohorts from SLO consistency check", func(t *testing.T) {
		// GIVEN a spec with ServeGenData set (clients exempted by guard)
		// AND user-authored cohorts with mixed SLOClass (one explicit, one empty)
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 10.0,
			Cohorts: []CohortSpec{
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
			},
			ServeGenData: &ServeGenDataSpec{
				Path: "dummy",
			},
		}
		// WHEN Validate is called
		err := spec.Validate()
		// THEN it MUST return mixed specification error (cohorts are ALWAYS checked)
		// This documents the asymmetry: clients guarded by expansion check, cohorts always scanned
		if err == nil {
			t.Fatal("expected error for mixed cohort slo_class even with ServeGenData set, got nil")
		}
		if !strings.Contains(err.Error(), "mixed slo_class specification") {
			t.Errorf("expected 'mixed slo_class specification' error, got: %v", err)
		}
	})

	t.Run("pre-existing validation error takes precedence", func(t *testing.T) {
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
			t.Errorf("expected pre-existing error to fire first, not mixed spec error; got: %v", errMsg)
		}
		if strings.Contains(errMsg, "mixed slo_class specification") {
			t.Errorf("expected pre-existing error to fire first, not mixed spec error; got: %v", errMsg)
		}
	})
}
