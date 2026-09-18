package trace

import (
	"testing"
)

func TestSimulationTrace_RecordAdmission_AppendsRecord(t *testing.T) {
	// GIVEN a trace configured for decisions
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions, CounterfactualK: 0})

	// WHEN an admission record is recorded
	st.RecordAdmission(AdmissionRecord{
		RequestID: "req_1",
		Clock:     1000,
		Admitted:  true,
		Reason:    "always-admit",
	})

	// THEN the trace contains one admission record with correct data
	if len(st.Admissions) != 1 {
		t.Fatalf("expected 1 admission, got %d", len(st.Admissions))
	}
	if st.Admissions[0].RequestID != "req_1" {
		t.Errorf("expected request ID req_1, got %s", st.Admissions[0].RequestID)
	}
	if !st.Admissions[0].Admitted {
		t.Error("expected admitted=true")
	}
}

func TestSimulationTrace_RecordRouting_AppendsRecord(t *testing.T) {
	// GIVEN a trace configured for decisions
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions, CounterfactualK: 0})

	// WHEN a routing record is recorded
	st.RecordRouting(RoutingRecord{
		RequestID:      "req_1",
		Clock:          2000,
		ChosenInstance: "instance_0",
		Reason:         "least-loaded (load=0)",
		Scores:         nil,
	})

	// THEN the trace contains one routing record with correct data
	if len(st.Routings) != 1 {
		t.Fatalf("expected 1 routing, got %d", len(st.Routings))
	}
	if st.Routings[0].ChosenInstance != "instance_0" {
		t.Errorf("expected instance_0, got %s", st.Routings[0].ChosenInstance)
	}
}

func TestSimulationTrace_MultipleRecords_PreservesOrder(t *testing.T) {
	// GIVEN a trace
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})

	// WHEN multiple records are added
	st.RecordAdmission(AdmissionRecord{RequestID: "req_1", Clock: 100, Admitted: true, Reason: "ok"})
	st.RecordAdmission(AdmissionRecord{RequestID: "req_2", Clock: 200, Admitted: false, Reason: "rejected"})
	st.RecordRouting(RoutingRecord{RequestID: "req_1", Clock: 150, ChosenInstance: "i_0", Reason: "rr"})

	// THEN order is preserved
	if len(st.Admissions) != 2 {
		t.Fatalf("expected 2 admissions, got %d", len(st.Admissions))
	}
	if st.Admissions[0].RequestID != "req_1" || st.Admissions[1].RequestID != "req_2" {
		t.Error("admission order not preserved")
	}
	if len(st.Routings) != 1 || st.Routings[0].RequestID != "req_1" {
		t.Error("routing record mismatch")
	}
}

func TestIsValidTraceLevel_ValidLevels(t *testing.T) {
	tests := []struct {
		level string
		valid bool
	}{
		{"none", true},
		{"decisions", true},
		{"", true}, // empty defaults to none
		{"detailed", false},
		{"foobar", false},
		{"NONE", false}, // case-sensitive
	}
	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			if got := IsValidTraceLevel(tt.level); got != tt.valid {
				t.Errorf("IsValidTraceLevel(%q) = %v, want %v", tt.level, got, tt.valid)
			}
		})
	}
}
