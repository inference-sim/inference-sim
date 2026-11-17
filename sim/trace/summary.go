package trace

// TraceSummary aggregates statistics from a SimulationTrace.
type TraceSummary struct {
	TotalDecisions     int
	AdmittedCount      int
	RejectedCount      int
	MeanRegret         float64
	MaxRegret          float64
	UniqueTargets      int
	TargetDistribution map[string]int // instance ID → count of routing picks in Routings: the instance that serves the request, and in a disaggregated run the decode pick (the prefill pick is PrefillDistribution)

	// PrefillDistribution is instance ID → count of prefill-pool picks (PrefillRoutings); empty
	// when no request was disaggregated. Its counts sum to the number of PrefillRoutings, which
	// equals DisaggregatedCount unless a request found no routable prefill instance.
	PrefillDistribution map[string]int

	// PD disaggregation summary (zero when disaggregation is not configured)
	DisaggregationCount  int     // number of disaggregation decisions recorded (true and false combined)
	DisaggregatedCount   int     // number of requests for which disaggregation was decided (Disaggregate=true); prefill routing happens in a subsequent event
	KVTransferCount      int     // number of KV transfers that completed with successful decode KV allocation
	MeanTransferDuration float64 // mean KV transfer duration in microseconds; zero when KVTransferCount == 0
}

// Summarize computes aggregate statistics from a SimulationTrace.
// Safe for nil or empty traces (returns zero-value fields).
func Summarize(st *SimulationTrace) *TraceSummary {
	summary := &TraceSummary{
		TargetDistribution:  make(map[string]int),
		PrefillDistribution: make(map[string]int),
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

	for _, r := range st.PrefillRoutings {
		summary.PrefillDistribution[r.ChosenInstance]++
	}

	// UniqueTargets counts distinct instances in TargetDistribution (the routing picks, not the prefill picks).
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
		// Accumulate in float64 to avoid int64 overflow for large simulations with many
		// long-duration transfers (int64 max ~9.22×10^18 µs; float64 exact up to ~9×10^15 µs).
		totalDuration := 0.0
		for _, kv := range st.KVTransfers {
			totalDuration += float64(kv.TransferDuration)
		}
		summary.MeanTransferDuration = totalDuration / float64(len(st.KVTransfers))
	}

	return summary
}
