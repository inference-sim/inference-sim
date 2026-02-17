package workload

import (
	"fmt"
	"math"
	"sort"
)

// MetricComparison holds statistical comparison between real and sim values.
type MetricComparison struct {
	RealP50, SimP50 float64
	RealP90, SimP90 float64
	RealP95, SimP95 float64
	RealP99, SimP99 float64
	MAPE            float64
	PearsonR        float64
	BiasDirection   string  // "over-predict", "under-predict", "neutral"
	Quality         string  // "excellent", "good", "fair", "poor"
	Count           int
}

// CalibrationReport holds the complete calibration result.
type CalibrationReport struct {
	TraceInfo struct {
		NumRequests     int    `json:"num_requests"`
		WarmUpExcluded  int    `json:"warm_up_excluded"`
		MatchedPairs    int    `json:"matched_pairs"`
		TokenMismatches int    `json:"token_mismatches"`
		Duration        string `json:"duration,omitempty"`
	} `json:"trace_info"`
	Metrics          map[string]*MetricComparison `json:"metrics"`
	ConfigMatch      ConfigMatchInfo              `json:"config_match"`
	KnownLimitations []string                     `json:"known_limitations"`
}

// ConfigMatchInfo documents which sim params matched the trace header.
type ConfigMatchInfo struct {
	Matched   []string `json:"matched,omitempty"`
	Defaulted []string `json:"defaulted,omitempty"`
}

// CalibrationConfig holds normalization parameters.
type CalibrationConfig struct {
	WarmUpRequests int
	NetworkRTTUs   int64
	BandwidthMbps  float64
}

// SimResult holds per-request sim output for calibration matching.
// All latencies are server-side (no network), in microseconds.
type SimResult struct {
	RequestID    int
	TTFT         float64 // Server-side: FirstTokenTime - ArrivalTime (µs)
	E2E          float64 // Server-side: CompletionTime - ArrivalTime (µs)
	InputTokens  int
	OutputTokens int
}

// LatencyPair holds matched real-vs-sim latency vectors.
type LatencyPair struct {
	Real []float64
	Sim  []float64
}

// CalibrationPairs holds matched, normalized real-vs-sim latency vectors.
type CalibrationPairs struct {
	TTFT               LatencyPair
	E2E                LatencyPair
	TokenMismatchCount int
	ExcludedWarmUp     int
	MatchedCount       int
	UnmatchedReal      int
	UnmatchedSim       int
}

// PrepareCalibrationPairs matches real trace records with sim results,
// applies network normalization, excludes warm-up, and detects token mismatches.
func PrepareCalibrationPairs(
	realRecords []TraceRecord,
	simResults []SimResult,
	config *CalibrationConfig,
) (*CalibrationPairs, error) {
	if config == nil {
		config = &CalibrationConfig{}
	}

	// Index sim results by RequestID
	simByID := make(map[int]SimResult, len(simResults))
	for _, sr := range simResults {
		simByID[sr.RequestID] = sr
	}

	pairs := &CalibrationPairs{}
	matchedSimIDs := make(map[int]bool)

	for _, rec := range realRecords {
		// Skip warm-up
		if rec.RequestID < config.WarmUpRequests {
			pairs.ExcludedWarmUp++
			continue
		}

		sr, ok := simByID[rec.RequestID]
		if !ok {
			pairs.UnmatchedReal++
			continue
		}
		matchedSimIDs[rec.RequestID] = true
		pairs.MatchedCount++

		// Check token count mismatch
		if rec.InputTokens != sr.InputTokens || rec.OutputTokens != sr.OutputTokens {
			pairs.TokenMismatchCount++
		}

		// Compute real latencies (client-side)
		realTTFT := float64(rec.FirstChunkTimeUs - rec.SendTimeUs)
		realE2E := float64(rec.LastChunkTimeUs - rec.SendTimeUs)

		// Guard against negative latencies (clock skew or data corruption)
		if realTTFT < 0 || realE2E < 0 {
			pairs.UnmatchedReal++ // treat as unmatched rather than corrupt the data
			continue
		}

		// Compute sim client-perspective latencies (server-side + network)
		// Reuse network.go helpers for bandwidth delay computation
		networkAdjust := float64(config.NetworkRTTUs)
		uploadDelay := computeUploadDelay(config.BandwidthMbps, sr.InputTokens)
		downloadDelay := computeDownloadDelay(config.BandwidthMbps, sr.OutputTokens)
		simTTFT := sr.TTFT + networkAdjust + uploadDelay
		simE2E := sr.E2E + networkAdjust + uploadDelay + downloadDelay

		pairs.TTFT.Real = append(pairs.TTFT.Real, realTTFT)
		pairs.TTFT.Sim = append(pairs.TTFT.Sim, simTTFT)
		pairs.E2E.Real = append(pairs.E2E.Real, realE2E)
		pairs.E2E.Sim = append(pairs.E2E.Sim, simE2E)
	}

	// Count unmatched sim results
	for _, sr := range simResults {
		if !matchedSimIDs[sr.RequestID] {
			pairs.UnmatchedSim++
		}
	}

	return pairs, nil
}

// ComputeCalibration computes statistical comparison between real and sim latency vectors.
func ComputeCalibration(real, sim []float64, metricName string) (*MetricComparison, error) {
	if len(real) == 0 || len(sim) == 0 {
		return nil, fmt.Errorf("empty latency vectors for %s", metricName)
	}
	if len(real) != len(sim) {
		return nil, fmt.Errorf("mismatched vector lengths for %s: real=%d sim=%d", metricName, len(real), len(sim))
	}

	comp := &MetricComparison{Count: len(real)}

	// Percentiles
	realSorted := sortedCopy(real)
	simSorted := sortedCopy(sim)
	comp.RealP50 = percentileFromSorted(realSorted, 50)
	comp.SimP50 = percentileFromSorted(simSorted, 50)
	comp.RealP90 = percentileFromSorted(realSorted, 90)
	comp.SimP90 = percentileFromSorted(simSorted, 90)
	comp.RealP95 = percentileFromSorted(realSorted, 95)
	comp.SimP95 = percentileFromSorted(simSorted, 95)
	comp.RealP99 = percentileFromSorted(realSorted, 99)
	comp.SimP99 = percentileFromSorted(simSorted, 99)

	// MAPE (skip where real == 0)
	mapeSum := 0.0
	mapeCount := 0
	biasSum := 0.0
	for i := range real {
		if real[i] == 0 {
			continue
		}
		err := math.Abs(real[i]-sim[i]) / real[i]
		mapeSum += err
		mapeCount++
		biasSum += sim[i] - real[i]
	}
	if mapeCount > 0 {
		comp.MAPE = mapeSum / float64(mapeCount)
		if biasSum > 0 {
			comp.BiasDirection = "over-predict"
		} else if biasSum < 0 {
			comp.BiasDirection = "under-predict"
		} else {
			comp.BiasDirection = "neutral"
		}
	}

	// Pearson r (requires N >= 3)
	if len(real) >= 3 {
		comp.PearsonR = pearsonCorrelation(real, sim)
	}

	// Quality rating
	comp.Quality = qualityRating(comp.MAPE, comp.PearsonR)

	return comp, nil
}

// BuildCalibrationReport creates a complete calibration report from pairs.
func BuildCalibrationReport(pairs *CalibrationPairs, configMatch *ConfigMatchInfo) (*CalibrationReport, error) {
	report := &CalibrationReport{
		Metrics:     make(map[string]*MetricComparison),
		ConfigMatch: *configMatch,
		KnownLimitations: []string{
			"BLIS models discrete batch steps. Real servers use iteration-level continuous batching. This may cause systematic TTFT prediction error under high load.",
			"Sim constructs synthetic prefix token IDs. Prefix cache hit rates may differ from real server, especially after evictions.",
			"If the real server uses speculative decoding, actual token generation patterns differ from sim's sequential model.",
		},
	}
	report.TraceInfo.MatchedPairs = pairs.MatchedCount
	report.TraceInfo.WarmUpExcluded = pairs.ExcludedWarmUp
	report.TraceInfo.TokenMismatches = pairs.TokenMismatchCount
	report.TraceInfo.NumRequests = pairs.MatchedCount + pairs.ExcludedWarmUp + pairs.UnmatchedReal

	if len(pairs.TTFT.Real) > 0 {
		ttft, err := ComputeCalibration(pairs.TTFT.Real, pairs.TTFT.Sim, "ttft")
		if err != nil {
			return nil, err
		}
		report.Metrics["ttft"] = ttft
	}
	if len(pairs.E2E.Real) > 0 {
		e2e, err := ComputeCalibration(pairs.E2E.Real, pairs.E2E.Sim, "e2e")
		if err != nil {
			return nil, err
		}
		report.Metrics["e2e"] = e2e
	}
	return report, nil
}

// --- Helper functions ---

func sortedCopy(vals []float64) []float64 {
	s := make([]float64, len(vals))
	copy(s, vals)
	sort.Float64s(s)
	return s
}

func percentileFromSorted(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := p / 100.0 * float64(len(sorted)-1)
	lower := int(math.Floor(rank))
	upper := int(math.Ceil(rank))
	if lower == upper {
		return sorted[lower]
	}
	frac := rank - float64(lower)
	return sorted[lower] + frac*(sorted[upper]-sorted[lower])
}

func pearsonCorrelation(x, y []float64) float64 {
	n := float64(len(x))
	sumX, sumY, sumXY, sumX2, sumY2 := 0.0, 0.0, 0.0, 0.0, 0.0
	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	num := n*sumXY - sumX*sumY
	den := math.Sqrt((n*sumX2 - sumX*sumX) * (n*sumY2 - sumY*sumY))
	if den == 0 {
		return 0
	}
	return num / den
}

func qualityRating(mape, pearsonR float64) string {
	if mape < 0.10 && pearsonR > 0.95 {
		return "excellent"
	}
	if mape < 0.20 && pearsonR > 0.85 {
		return "good"
	}
	if mape < 0.35 && pearsonR > 0.70 {
		return "fair"
	}
	return "poor"
}
