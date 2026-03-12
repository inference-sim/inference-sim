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

func TestSimulationTrace_NewRecordTypes_Initialized(t *testing.T) {
	// GIVEN a new SimulationTrace
	tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})

	// WHEN checked immediately after creation
	// THEN all PD-specific slices are non-nil and empty (not nil)
	if tr.Disaggregations == nil {
		t.Error("Disaggregations slice is nil, want empty non-nil slice")
	}
	if tr.PrefillRoutings == nil {
		t.Error("PrefillRoutings slice is nil, want empty non-nil slice")
	}
	if tr.DecodeRoutings == nil {
		t.Error("DecodeRoutings slice is nil, want empty non-nil slice")
	}
	if tr.KVTransfers == nil {
		t.Error("KVTransfers slice is nil, want empty non-nil slice")
	}
	if len(tr.Disaggregations) != 0 || len(tr.PrefillRoutings) != 0 ||
		len(tr.DecodeRoutings) != 0 || len(tr.KVTransfers) != 0 {
		t.Error("expected all PD slices empty after construction")
	}
}

func TestSimulationTrace_RecordDisaggregation(t *testing.T) {
	tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	tr.RecordDisaggregation(DisaggregationRecord{RequestID: "req_0", Clock: 100, Disaggregate: true})
	tr.RecordDisaggregation(DisaggregationRecord{RequestID: "req_1", Clock: 200, Disaggregate: false})

	if len(tr.Disaggregations) != 2 {
		t.Fatalf("expected 2 disaggregation records, got %d", len(tr.Disaggregations))
	}
	if !tr.Disaggregations[0].Disaggregate {
		t.Error("first record: Disaggregate should be true")
	}
	if tr.Disaggregations[1].Disaggregate {
		t.Error("second record: Disaggregate should be false")
	}
}

func TestSimulationTrace_RecordPrefillRouting(t *testing.T) {
	tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	tr.RecordPrefillRouting(PrefillRoutingRecord{
		ParentRequestID: "req_0",
		Clock:           150,
		ChosenInstance:  "instance_0",
		Scores:          map[string]float64{"instance_0": 0.9, "instance_1": 0.7},
		Regret:          0.2,
	})

	if len(tr.PrefillRoutings) != 1 {
		t.Fatalf("expected 1 prefill routing record, got %d", len(tr.PrefillRoutings))
	}
	r := tr.PrefillRoutings[0]
	if r.ChosenInstance != "instance_0" {
		t.Errorf("ChosenInstance = %q, want %q", r.ChosenInstance, "instance_0")
	}
	if r.Regret != 0.2 {
		t.Errorf("Regret = %f, want 0.2", r.Regret)
	}
}

func TestSimulationTrace_RecordKVTransfer(t *testing.T) {
	tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	tr.RecordKVTransfer(KVTransferRecord{
		ParentRequestID:   "req_0",
		TransferStartTime: 500,
		TransferDuration:  42,
		NumKVBlocks:       7,
		PrefillInstanceID: "instance_0",
		DecodeInstanceID:  "instance_2",
	})

	if len(tr.KVTransfers) != 1 {
		t.Fatalf("expected 1 KV transfer record, got %d", len(tr.KVTransfers))
	}
	r := tr.KVTransfers[0]
	if r.TransferDuration != 42 {
		t.Errorf("TransferDuration = %d, want 42", r.TransferDuration)
	}
	if r.NumKVBlocks != 7 {
		t.Errorf("NumKVBlocks = %d, want 7", r.NumKVBlocks)
	}
	if r.PrefillInstanceID != "instance_0" || r.DecodeInstanceID != "instance_2" {
		t.Errorf("instance IDs = (%q, %q), want (instance_0, instance_2)",
			r.PrefillInstanceID, r.DecodeInstanceID)
	}
}

func TestSimulationTrace_RecordDecodeRouting(t *testing.T) {
	tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	tr.RecordDecodeRouting(DecodeRoutingRecord{
		ParentRequestID: "req_0",
		Clock:           600,
		ChosenInstance:  "instance_2",
		Candidates: []CandidateScore{
			{InstanceID: "instance_2", Score: 0.8},
			{InstanceID: "instance_3", Score: 0.6},
		},
		Regret: 0.0,
	})

	if len(tr.DecodeRoutings) != 1 {
		t.Fatalf("expected 1 decode routing record, got %d", len(tr.DecodeRoutings))
	}
	r := tr.DecodeRoutings[0]
	if r.ChosenInstance != "instance_2" {
		t.Errorf("ChosenInstance = %q, want %q", r.ChosenInstance, "instance_2")
	}
	if len(r.Candidates) != 2 {
		t.Errorf("Candidates len = %d, want 2", len(r.Candidates))
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
