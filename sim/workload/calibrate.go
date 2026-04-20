package workload

import (
	"fmt"
	"math"
	"sort"

	"github.com/sirupsen/logrus"
)

// WorkloadAggregates describes the latency distribution shape across the entire workload.
type WorkloadAggregates struct {
	RealMean           float64 `json:"real_mean"`
	SimMean            float64 `json:"sim_mean"`
	MeanError          float64 `json:"mean_error"`           // SimMean - RealMean
	MeanPercentError   float64 `json:"mean_percent_error"`   // |MeanError| / RealMean
	RealMedian         float64 `json:"real_median"`
	SimMedian          float64 `json:"sim_median"`
	MedianError        float64 `json:"median_error"`         // SimMedian - RealMedian
	MedianPercentError float64 `json:"median_percent_error"` // |MedianError| / RealMedian
	RealP50            float64 `json:"real_p50"`
	SimP50             float64 `json:"sim_p50"`
	RealP90            float64 `json:"real_p90"`
	SimP90             float64 `json:"sim_p90"`
	RealP95            float64 `json:"real_p95"`
	SimP95             float64 `json:"sim_p95"`
	RealP99            float64 `json:"real_p99"`
	SimP99             float64 `json:"sim_p99"`
}

// PredictionQuality describes how accurately the simulator predicts each individual request.
type PredictionQuality struct {
	MAPE          float64 `json:"mape"`
	PearsonR      float64 `json:"pearson_r"`
	BiasDirection string  `json:"bias_direction"` // "over-predict", "under-predict", "neutral"
	Quality       string  `json:"quality"`        // "excellent", "good", "fair", "poor"
}

// MetricComparison holds statistical comparison between real and sim values.
// Organized into workload-level aggregates and request-level prediction quality.
type MetricComparison struct {
	WorkloadLevel WorkloadAggregates `json:"workload_level"`
	RequestLevel  PredictionQuality  `json:"request_level"`
	NumRequests   int                `json:"num_requests"` // Number of matched request pairs in this comparison
}

// CalibrationReport holds the complete calibration result.
type CalibrationReport struct {
	TraceInfo struct {
		NumRequests     int    `json:"num_requests"`
		WarmUpExcluded  int    `json:"warm_up_excluded"`
		MatchedPairs    int    `json:"matched_pairs"`
		TokenMismatches int    `json:"token_mismatches"`
		ITLDropped      int    `json:"itl_dropped,omitempty"` // Requests dropped from ITL due to clock skew
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
// TTFT and E2E are server-side latencies in microseconds (simulation ticks).
type SimResult struct {
	RequestID    int     `json:"request_id"`
	TTFT         float64 `json:"ttft_us"` // server-side TTFT in microseconds
	E2E          float64 `json:"e2e_us"`  // server-side E2E in microseconds
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
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
	ITL                LatencyPair
	TokenMismatchCount int
	ExcludedWarmUp     int
	MatchedCount       int
	UnmatchedReal      int
	UnmatchedSim       int
	ITLDropped         int // Requests dropped from ITL due to clock skew (all negative deltas)
}

// PrepareCalibrationPairs matches real trace records with sim results,
// applies network normalization, excludes warm-up, and detects token mismatches.
// Returns the pairs and a simByID map for reuse by callers (e.g., PrepareCalibrationPairsWithITL).
func PrepareCalibrationPairs(
	realRecords []TraceRecord,
	simResults []SimResult,
	config *CalibrationConfig,
) (*CalibrationPairs, map[int]SimResult, error) {
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

	return pairs, simByID, nil
}

// PrepareCalibrationPairsWithITL extends PrepareCalibrationPairs with ITL data.
// ITL is computed as per-request mean inter-chunk latency (microseconds).
// First chunk delta is TTFT; subsequent deltas are ITL.
func PrepareCalibrationPairsWithITL(
	realRecords []TraceRecord,
	simResults []SimResult,
	itlRecords []ITLRecord,
	config *CalibrationConfig,
) (*CalibrationPairs, error) {
	// Start with standard pairs (reuse simByID map to avoid O(N) duplication)
	pairs, simByID, err := PrepareCalibrationPairs(realRecords, simResults, config)
	if err != nil {
		return nil, err
	}

	// Group ITL records by request ID
	itlByRequest := make(map[int][]ITLRecord)
	for _, rec := range itlRecords {
		itlByRequest[rec.RequestID] = append(itlByRequest[rec.RequestID], rec)
	}

	if config == nil {
		config = &CalibrationConfig{}
	}

	// Compute per-request ITL
	for _, rec := range realRecords {
		// Skip warm-up
		if rec.RequestID < config.WarmUpRequests {
			continue
		}

		sr, ok := simByID[rec.RequestID]
		if !ok {
			continue
		}

		chunks, ok := itlByRequest[rec.RequestID]
		if !ok || len(chunks) < 2 {
			continue // No ITL data for this request
		}

		// Sort chunks by index (defensive)
		sortITLRecords(chunks)

		// Compute real ITL: mean of chunk-to-chunk deltas (skip first, which is TTFT)
		var realITLSum float64
		realITLCount := 0
		for i := 1; i < len(chunks); i++ {
			delta := float64(chunks[i].TimestampUs - chunks[i-1].TimestampUs)
			if delta < 0 {
				// Clock skew or corrupt data — skip this delta
				continue
			}
			realITLSum += delta
			realITLCount++
		}
		if realITLCount == 0 {
			// All deltas were negative (clock skew) — drop this request from ITL (R1)
			logrus.Warnf("calibrate: request %d ITL dropped (all %d deltas negative, likely clock skew)", rec.RequestID, len(chunks)-1)
			pairs.ITLDropped++
			continue
		}
		realITL := realITLSum / float64(realITLCount)

		// Compute sim ITL: (E2E - TTFT) / OutputTokens
		// This approximates mean ITL assuming uniform token generation
		simITL := 0.0
		if sr.OutputTokens > 1 {
			simITL = (sr.E2E - sr.TTFT) / float64(sr.OutputTokens-1)
		}

		pairs.ITL.Real = append(pairs.ITL.Real, realITL)
		pairs.ITL.Sim = append(pairs.ITL.Sim, simITL)
	}

	return pairs, nil
}

func sortITLRecords(records []ITLRecord) {
	// Simple insertion sort (small N)
	for i := 1; i < len(records); i++ {
		key := records[i]
		j := i - 1
		for j >= 0 && records[j].ChunkIndex > key.ChunkIndex {
			records[j+1] = records[j]
			j--
		}
		records[j+1] = key
	}
}

// ComputeCalibration computes statistical comparison between real and sim latency vectors.
func ComputeCalibration(real, sim []float64, metricName string) (*MetricComparison, error) {
	if len(real) == 0 || len(sim) == 0 {
		return nil, fmt.Errorf("empty latency vectors for %s", metricName)
	}
	if len(real) != len(sim) {
		return nil, fmt.Errorf("mismatched vector lengths for %s: real=%d sim=%d", metricName, len(real), len(sim))
	}

	comp := &MetricComparison{NumRequests: len(real)}

	// Mean (single-pass sum before percentile sort)
	realSum, simSum := 0.0, 0.0
	for i := range real {
		realSum += real[i]
		simSum += sim[i]
	}
	n := float64(len(real))
	comp.WorkloadLevel.RealMean = realSum / n
	comp.WorkloadLevel.SimMean = simSum / n

	// Percentiles
	realSorted := sortedCopy(real)
	simSorted := sortedCopy(sim)
	comp.WorkloadLevel.RealP50 = percentileFromSorted(realSorted, 50)
	comp.WorkloadLevel.SimP50 = percentileFromSorted(simSorted, 50)
	comp.WorkloadLevel.RealP90 = percentileFromSorted(realSorted, 90)
	comp.WorkloadLevel.SimP90 = percentileFromSorted(simSorted, 90)
	comp.WorkloadLevel.RealP95 = percentileFromSorted(realSorted, 95)
	comp.WorkloadLevel.SimP95 = percentileFromSorted(simSorted, 95)
	comp.WorkloadLevel.RealP99 = percentileFromSorted(realSorted, 99)
	comp.WorkloadLevel.SimP99 = percentileFromSorted(simSorted, 99)

	// Median aliases P50
	comp.WorkloadLevel.RealMedian = comp.WorkloadLevel.RealP50
	comp.WorkloadLevel.SimMedian = comp.WorkloadLevel.SimP50

	// Mean error and percent error (with division guards, R11)
	comp.WorkloadLevel.MeanError = comp.WorkloadLevel.SimMean - comp.WorkloadLevel.RealMean
	if comp.WorkloadLevel.RealMean != 0 {
		comp.WorkloadLevel.MeanPercentError = math.Abs(comp.WorkloadLevel.MeanError) / comp.WorkloadLevel.RealMean
	}

	// Median error and percent error (with division guards, R11)
	comp.WorkloadLevel.MedianError = comp.WorkloadLevel.SimMedian - comp.WorkloadLevel.RealMedian
	if comp.WorkloadLevel.RealMedian != 0 {
		comp.WorkloadLevel.MedianPercentError = math.Abs(comp.WorkloadLevel.MedianError) / comp.WorkloadLevel.RealMedian
	}

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
		comp.RequestLevel.MAPE = mapeSum / float64(mapeCount)
		if biasSum > 0 {
			comp.RequestLevel.BiasDirection = "over-predict"
		} else if biasSum < 0 {
			comp.RequestLevel.BiasDirection = "under-predict"
		} else {
			comp.RequestLevel.BiasDirection = "neutral"
		}
	}

	// Pearson r (requires N >= 3)
	if len(real) >= 3 {
		comp.RequestLevel.PearsonR = pearsonCorrelation(real, sim)
	}

	// Quality rating
	comp.RequestLevel.Quality = qualityRating(comp.RequestLevel.MAPE, comp.RequestLevel.PearsonR)

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
	report.TraceInfo.ITLDropped = pairs.ITLDropped
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
	if len(pairs.ITL.Real) > 0 {
		itl, err := ComputeCalibration(pairs.ITL.Real, pairs.ITL.Sim, "itl")
		if err != nil {
			return nil, err
		}
		report.Metrics["itl"] = itl
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
