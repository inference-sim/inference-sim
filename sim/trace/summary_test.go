package trace

import "testing"

func TestSummarize_EmptyTrace_ZeroValues(t *testing.T) {
	// GIVEN an empty trace
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})

	// WHEN summarized
	summary := Summarize(st)

	// THEN all counts are zero
	if summary.TotalDecisions != 0 {
		t.Errorf("expected 0 total decisions, got %d", summary.TotalDecisions)
	}
	if summary.AdmittedCount != 0 || summary.RejectedCount != 0 {
		t.Error("expected 0 admitted and rejected")
	}
	if summary.UniqueTargets != 0 {
		t.Errorf("expected 0 unique targets, got %d", summary.UniqueTargets)
	}
	if summary.MeanRegret != 0 || summary.MaxRegret != 0 {
		t.Error("expected 0 regret values")
	}
	if len(summary.TargetDistribution) != 0 {
		t.Error("expected empty target distribution")
	}
}

func TestSummarize_PopulatedTrace_CorrectCounts(t *testing.T) {
	// GIVEN a trace with mixed admission and routing records
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordAdmission(AdmissionRecord{RequestID: "r1", Admitted: true, Reason: "ok"})
	st.RecordAdmission(AdmissionRecord{RequestID: "r2", Admitted: false, Reason: "rejected"})
	st.RecordAdmission(AdmissionRecord{RequestID: "r3", Admitted: true, Reason: "ok"})
	st.RecordRouting(RoutingRecord{RequestID: "r1", ChosenInstance: "i_0", Regret: 0.1})
	st.RecordRouting(RoutingRecord{RequestID: "r3", ChosenInstance: "i_1", Regret: 0.3})

	// WHEN summarized
	summary := Summarize(st)

	// THEN counts match
	if summary.TotalDecisions != 3 {
		t.Errorf("expected 3 total decisions, got %d", summary.TotalDecisions)
	}
	if summary.AdmittedCount != 2 {
		t.Errorf("expected 2 admitted, got %d", summary.AdmittedCount)
	}
	if summary.RejectedCount != 1 {
		t.Errorf("expected 1 rejected, got %d", summary.RejectedCount)
	}
	if summary.UniqueTargets != 2 {
		t.Errorf("expected 2 unique targets, got %d", summary.UniqueTargets)
	}
}

func TestSummarize_RegretStatistics_CorrectMeanAndMax(t *testing.T) {
	// GIVEN routing records with known regrets
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordRouting(RoutingRecord{RequestID: "r1", ChosenInstance: "i_0", Regret: 0.1})
	st.RecordRouting(RoutingRecord{RequestID: "r2", ChosenInstance: "i_0", Regret: 0.5})
	st.RecordRouting(RoutingRecord{RequestID: "r3", ChosenInstance: "i_1", Regret: 0.2})

	// WHEN summarized
	summary := Summarize(st)

	// THEN mean regret = (0.1 + 0.5 + 0.2) / 3 ≈ 0.2667
	expectedMean := (0.1 + 0.5 + 0.2) / 3.0
	if summary.MeanRegret < expectedMean-0.001 || summary.MeanRegret > expectedMean+0.001 {
		t.Errorf("expected mean regret ~%.4f, got %.4f", expectedMean, summary.MeanRegret)
	}

	// THEN max regret = 0.5
	if summary.MaxRegret != 0.5 {
		t.Errorf("expected max regret 0.5, got %.4f", summary.MaxRegret)
	}
}

func TestSummarize_TargetDistribution_CountsPerInstance(t *testing.T) {
	// GIVEN routing to same instance multiple times
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordRouting(RoutingRecord{RequestID: "r1", ChosenInstance: "i_0"})
	st.RecordRouting(RoutingRecord{RequestID: "r2", ChosenInstance: "i_0"})
	st.RecordRouting(RoutingRecord{RequestID: "r3", ChosenInstance: "i_1"})

	// WHEN summarized
	summary := Summarize(st)

	// THEN target distribution reflects counts
	if summary.TargetDistribution["i_0"] != 2 {
		t.Errorf("expected i_0 count 2, got %d", summary.TargetDistribution["i_0"])
	}
	if summary.TargetDistribution["i_1"] != 1 {
		t.Errorf("expected i_1 count 1, got %d", summary.TargetDistribution["i_1"])
	}
}
