package trace

import "testing"

func TestSummarize_NilTrace_ReturnsZeroValueSummary(t *testing.T) {
	// GIVEN a nil trace (e.g., tracing was disabled)
	// WHEN summarized
	summary := Summarize(nil)

	// THEN returns non-nil summary with all zero values
	if summary == nil {
		t.Fatal("expected non-nil summary for nil input")
	}
	if summary.TotalDecisions != 0 {
		t.Errorf("expected 0 total decisions, got %d", summary.TotalDecisions)
	}
	if len(summary.TargetDistribution) != 0 {
		t.Error("expected empty target distribution")
	}
}

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

func TestSummarize_PDFields_DisaggregationCounting(t *testing.T) {
	// GIVEN a trace with 2 disaggregate=true and 1 disaggregate=false decision
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordDisaggregation(DisaggregationRecord{RequestID: "r1", Clock: 100, Disaggregate: true})
	st.RecordDisaggregation(DisaggregationRecord{RequestID: "r2", Clock: 200, Disaggregate: true})
	st.RecordDisaggregation(DisaggregationRecord{RequestID: "r3", Clock: 300, Disaggregate: false})

	// WHEN summarized
	summary := Summarize(st)

	// THEN counts are exact
	if summary.DisaggregationCount != 3 {
		t.Errorf("DisaggregationCount: expected 3 (all decisions), got %d", summary.DisaggregationCount)
	}
	if summary.DisaggregatedCount != 2 {
		t.Errorf("DisaggregatedCount: expected 2 (true only), got %d", summary.DisaggregatedCount)
	}
	// No KV transfers recorded — KVTransferCount and MeanTransferDuration must be zero
	if summary.KVTransferCount != 0 {
		t.Errorf("KVTransferCount: expected 0, got %d", summary.KVTransferCount)
	}
	if summary.MeanTransferDuration != 0 {
		t.Errorf("MeanTransferDuration: expected 0 when KVTransferCount==0, got %f", summary.MeanTransferDuration)
	}
}

func TestSummarize_PDFields_KVTransferMean(t *testing.T) {
	// GIVEN KV transfers with known durations: 1000, 2000, 3000 µs
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordKVTransfer(KVTransferRecord{ParentRequestID: "r1", TransferStartTime: 100, TransferDuration: 1000, NumKVBlocks: 4, PrefillInstanceID: "p0", DecodeInstanceID: "d0"})
	st.RecordKVTransfer(KVTransferRecord{ParentRequestID: "r2", TransferStartTime: 200, TransferDuration: 2000, NumKVBlocks: 8, PrefillInstanceID: "p0", DecodeInstanceID: "d0"})
	st.RecordKVTransfer(KVTransferRecord{ParentRequestID: "r3", TransferStartTime: 300, TransferDuration: 3000, NumKVBlocks: 6, PrefillInstanceID: "p1", DecodeInstanceID: "d0"})

	// WHEN summarized
	summary := Summarize(st)

	// THEN KVTransferCount == 3 and MeanTransferDuration == (1000+2000+3000)/3 == 2000
	if summary.KVTransferCount != 3 {
		t.Errorf("KVTransferCount: expected 3, got %d", summary.KVTransferCount)
	}
	const wantMean = 2000.0
	if summary.MeanTransferDuration < wantMean-0.001 || summary.MeanTransferDuration > wantMean+0.001 {
		t.Errorf("MeanTransferDuration: expected %.1f, got %.6f", wantMean, summary.MeanTransferDuration)
	}
}

func TestSummarize_PDFields_EmptyPDRecords_ZeroValues(t *testing.T) {
	// GIVEN a trace with only standard admission/routing records (no PD records)
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordAdmission(AdmissionRecord{RequestID: "r1", Admitted: true})
	st.RecordRouting(RoutingRecord{RequestID: "r1", ChosenInstance: "i_0"})

	// WHEN summarized
	summary := Summarize(st)

	// THEN all PD summary fields are zero (disaggregation not active)
	if summary.DisaggregationCount != 0 {
		t.Errorf("DisaggregationCount: expected 0, got %d", summary.DisaggregationCount)
	}
	if summary.DisaggregatedCount != 0 {
		t.Errorf("DisaggregatedCount: expected 0, got %d", summary.DisaggregatedCount)
	}
	if summary.KVTransferCount != 0 {
		t.Errorf("KVTransferCount: expected 0, got %d", summary.KVTransferCount)
	}
	if summary.MeanTransferDuration != 0 {
		t.Errorf("MeanTransferDuration: expected 0, got %f", summary.MeanTransferDuration)
	}
}
