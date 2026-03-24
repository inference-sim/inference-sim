package trace

// TraceSummary aggregates statistics from a SimulationTrace.
type TraceSummary struct {
	TotalDecisions     int
	AdmittedCount      int
	RejectedCount      int
	MeanRegret         float64
	MaxRegret          float64
	UniqueTargets      int
	TargetDistribution map[string]int // instance ID → count of requests routed via standard routing only (not PD pool routing)

	// PD disaggregation summary (zero when disaggregation is not configured)
	DisaggregationCount  int     // number of disaggregation decisions recorded
	DisaggregatedCount   int     // number of requests routed to prefill pool (Disaggregate=true)
	KVTransferCount      int     // number of completed KV transfers
	MeanTransferDuration float64 // mean KV transfer duration in microseconds
}

// Summarize computes aggregate statistics from a SimulationTrace.
// Safe for nil or empty traces (returns zero-value fields).
func Summarize(st *SimulationTrace) *TraceSummary {
	summary := &TraceSummary{
		TargetDistribution: make(map[string]int),
	}
	if st == nil {
		return summary
	}

	summary.TotalDecisions = len(st.Admissions)
	for _, a := range st.Admissions {
		if a.Admitted {
			summary.AdmittedCount++
		} else {
			summary.RejectedCount++
		}
	}

	if len(st.Routings) > 0 {
		totalRegret := 0.0
		for _, r := range st.Routings {
			summary.TargetDistribution[r.ChosenInstance]++
			totalRegret += r.Regret
			if r.Regret > summary.MaxRegret {
				summary.MaxRegret = r.Regret
			}
		}
		summary.MeanRegret = totalRegret / float64(len(st.Routings))
	}

	summary.UniqueTargets = len(summary.TargetDistribution)

	// PD disaggregation summary
	summary.DisaggregationCount = len(st.Disaggregations)
	for _, d := range st.Disaggregations {
		if d.Disaggregate {
			summary.DisaggregatedCount++
		}
	}

	summary.KVTransferCount = len(st.KVTransfers)
	if len(st.KVTransfers) > 0 {
		totalDuration := int64(0)
		for _, kv := range st.KVTransfers {
			totalDuration += kv.TransferDuration
		}
		summary.MeanTransferDuration = float64(totalDuration) / float64(len(st.KVTransfers))
	}

	return summary
}
