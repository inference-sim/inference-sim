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
	P50Error           float64 `json:"p50_error"`           // SimP50 - RealP50
	P50PercentError    float64 `json:"p50_percent_error"`   // |P50Error| / RealP50
	RealP90            float64 `json:"real_p90"`
	SimP90             float64 `json:"sim_p90"`
	P90Error           float64 `json:"p90_error"`           // SimP90 - RealP90
	P90PercentError    float64 `json:"p90_percent_error"`   // |P90Error| / RealP90
	RealP95            float64 `json:"real_p95"`
	SimP95             float64 `json:"sim_p95"`
	P95Error           float64 `json:"p95_error"`           // SimP95 - RealP95
	P95PercentError    float64 `json:"p95_percent_error"`   // |P95Error| / RealP95
	RealP99            float64 `json:"real_p99"`
	SimP99             float64 `json:"sim_p99"`
	P99Error           float64 `json:"p99_error"`           // SimP99 - RealP99
	P99PercentError    float64 `json:"p99_percent_error"`   // |P99Error| / RealP99
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
	Count         int                `json:"count"`
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
	// Goodput holds per-class observed-vs-simulated goodput comparison when
	// goodput targets were configured (issue #1413, BC-9). Omitted when targets
	// are absent so old consumers see the legacy report shape unchanged.
	Goodput *GoodputComparisonReport `json:"goodput,omitempty"`
	// HitRate holds the real-vs-sim KV cache hit-rate comparison (#1583, BC-6).
	// Populated when the trace header carries an observed hit-rate AND a sim
	// MetricsOutput with cache_hit_rate is supplied (--sim-metrics); omitted
	// otherwise so the legacy report shape is unchanged.
	HitRate *HitRateComparison `json:"hit_rate,omitempty"`
	// TTFTTolerance records the TTFT-MAPE tolerance verdict (#1583, BC-7) in the
	// report so automation consumers see it, not only the stderr log. Populated
	// whenever a TTFT metric exists; omitted otherwise.
	TTFTTolerance *ToleranceVerdict `json:"ttft_tolerance,omitempty"`
	// Throughput holds the real-vs-simulated aggregate throughput comparison (#1647):
	// output-token throughput and request-completion rate over the same request_id-matched
	// set used for latency. Populated whenever a positive real AND sim makespan and non-zero
	// matched output tokens make it derivable from the already-required inputs (the
	// present-when-derivable precedent of TTFTTolerance, not the flag-gated Goodput/HitRate);
	// omitted otherwise so no Inf/NaN value is ever written. NOTE: this is a calibrate report
	// (file artifact) shape addition, distinct from INV-6 sim-stdout byte-identity.
	Throughput *ThroughputComparison `json:"throughput,omitempty"`
}

// ToleranceVerdict is a compact machine-readable pass/fail against a MAPE threshold
// (#1583). MAPE and Threshold are fractions (not percentages).
type ToleranceVerdict struct {
	MAPE      float64 `json:"mape"`
	Threshold float64 `json:"threshold"`
	Within    bool    `json:"within"`
}

// HitRateComparison holds the real-vs-simulated KV cache hit-rate comparison (#1583).
// Real is the observed hit-rate scraped by `blis observe --scrape-kv-metrics` (trace
// header); Sim is the simulator's aggregate cache_hit_rate (from `blis replay
// --metrics-path`). The error is reported in percentage points, and Within tests it
// against the configured tolerance (default 5 pp).
type HitRateComparison struct {
	RealHitRate float64 `json:"real_hit_rate"`
	SimHitRate  float64 `json:"sim_hit_rate"`
	AbsErrorPP  float64 `json:"abs_error_pp"` // |sim − real| × 100 (percentage points)
	TolerancePP float64 `json:"tolerance_pp"`
	Within      bool    `json:"within"`
	// Source echoes the observation family ("tiered" or "gpu-prefix-cache-fallback")
	// so a weaker GPU-only comparison is never mistaken for a tiered one.
	Source string `json:"source"`
}

// ComputeHitRateComparison builds a HitRateComparison from a real (observed) and a
// simulated hit-rate, both fractions in [0,1]. tolerancePP is the absolute-error band
// in percentage points; Within is true when |sim − real| × 100 ≤ tolerancePP.
func ComputeHitRateComparison(realHitRate, simHitRate, tolerancePP float64, source string) *HitRateComparison {
	absErrPP := math.Abs(simHitRate-realHitRate) * 100
	// A tiny epsilon keeps the band inclusive at its intended boundary despite
	// float representation (e.g. |0.75−0.70|×100 = 5.000000000000004, which would
	// otherwise fail a "≤ 5 pp" band spuriously).
	const boundaryEps = 1e-9
	return &HitRateComparison{
		RealHitRate: realHitRate,
		SimHitRate:  simHitRate,
		AbsErrorPP:  absErrPP,
		TolerancePP: tolerancePP,
		Within:      absErrPP <= tolerancePP+boundaryEps,
		Source:      source,
	}
}

// ThroughputComparison holds the real-vs-simulated aggregate throughput comparison (#1647).
// Both endpoints live in the CLIENT frame: the real makespan runs from the earliest client
// SendTimeUs to the latest client LastChunkTimeUs; the sim makespan runs from the same earliest
// SendTimeUs to the latest (SendTimeUs + client-frame sim E2E), where the sim E2E is normalized
// with the identical network shift PrepareCalibrationPairs applies to the latency comparison
// (sr.E2E + NetworkRTTUs + upload + download). Throughput = matched output tokens / runtime and
// matched requests / runtime. The per-GPU fields divide output-token throughput by a caller-supplied
// GPU count; since one count divides both sides, per-GPU PercentError equals the raw PercentError.
//
// Validity boundary: the reconstructed sim makespan (SendTimeUs + simE2E) is a physical sim
// timeline only for FIXED-mode replay (the standard observe→replay→calibrate path, arrivals pinned
// from the trace). Under closed-loop / concurrent-session replay the sim regenerates the arrival
// schedule (round N+1 depends on the sim's completion of round N, INV-10), so the real SendTimeUs
// schedule is not the sim's arrival schedule — treat the number as an open-loop metric there.
type ThroughputComparison struct {
	MatchedRequests  int     `json:"matched_requests"`
	RealRuntimeSec   float64 `json:"real_runtime_sec"`
	SimRuntimeSec    float64 `json:"sim_runtime_sec"`
	RealOutputTokens int     `json:"real_output_tokens"`
	SimOutputTokens  int     `json:"sim_output_tokens"`

	RealOutputTokensPerSec         float64 `json:"real_output_tokens_per_sec"`
	SimOutputTokensPerSec          float64 `json:"sim_output_tokens_per_sec"`
	OutputTokensPerSecError        float64 `json:"output_tokens_per_sec_error"`         // sim − real
	OutputTokensPerSecPercentError float64 `json:"output_tokens_per_sec_percent_error"` // |error| / real

	RealRequestsPerSec         float64 `json:"real_requests_per_sec"`
	SimRequestsPerSec          float64 `json:"sim_requests_per_sec"`
	RequestsPerSecError        float64 `json:"requests_per_sec_error"`         // sim − real
	RequestsPerSecPercentError float64 `json:"requests_per_sec_percent_error"` // |error| / real

	// Per-GPU normalization — set only when numGPUs > 0 (pointers so omitempty drops them otherwise).
	NumGPUs                      *int     `json:"num_gpus,omitempty"`
	RealOutputTokensPerSecPerGPU *float64 `json:"real_output_tokens_per_sec_per_gpu,omitempty"`
	SimOutputTokensPerSecPerGPU  *float64 `json:"sim_output_tokens_per_sec_per_gpu,omitempty"`

	// Within-tolerance verdict on raw output-token throughput — set only when tolerancePct > 0.
	TolerancePct *float64 `json:"tolerance_pct,omitempty"`
	Within       *bool    `json:"within,omitempty"`
}

// ComputeThroughputComparison compares real vs simulated aggregate throughput over the
// request_id-matched set (#1647). Both endpoints live in the CLIENT frame: real end =
// LastChunkTimeUs (already client-side); sim end = SendTimeUs + client-frame sim E2E, where
// client-frame sim E2E = sr.E2E + config.NetworkRTTUs + upload + download — identical to the sim
// latency PrepareCalibrationPairs uses (so throughput stays consistent with the report's e2e block
// and is not biased by the server/client frame difference). numGPUs > 0 emits the per-GPU
// normalization; tolerancePct > 0 emits the within-tolerance verdict on raw output-token throughput.
//
// Returns nil (no fabricated comparison, R1) when the matched set is empty, either makespan is
// non-positive, or matched output tokens are zero — so the report key is omitted and no Inf/NaN
// value can reach json.MarshalIndent (which errors on non-finite floats and would discard the
// entire calibration report).
func ComputeThroughputComparison(
	realRecords []TraceRecord,
	simByID map[int]SimResult,
	matchedReqIDs map[int]bool,
	config *CalibrationConfig,
	numGPUs int,
	tolerancePct float64,
) *ThroughputComparison {
	if config == nil {
		config = &CalibrationConfig{}
	}

	var (
		minSend    int64 = -1
		maxRealEnd int64
		maxSimEnd  int64
		realOutTok int
		simOutTok  int
		matched    int
	)

	for _, rec := range realRecords {
		if !matchedReqIDs[rec.RequestID] {
			continue
		}
		sr, ok := simByID[rec.RequestID]
		if !ok {
			continue
		}
		// Only successfully-completed requests contribute to throughput (matches the
		// goodput numerator, goodput_compare.go): a failed/timed-out real record can
		// carry partial OutputTokens with a LastChunkTimeUs at the failure instant,
		// which would bias the compared quantity. Records with an unset Status (only
		// hand-built fixtures; real observe/run traces always populate it) are excluded.
		if rec.Status != "ok" {
			continue
		}
		// Same validity guards as the latency leg (calibrate.go real-latency guard):
		// a non-positive real makespan or corrupt (negative) sim E2E is excluded rather
		// than allowed to distort the aggregate window.
		if rec.SendTimeUs < 0 || rec.LastChunkTimeUs <= rec.SendTimeUs || sr.E2E < 0 {
			continue
		}

		// Client-frame sim completion time: server-side sim E2E shifted into the client
		// frame with the identical normalization the latency comparison applies.
		clientSimE2E := sr.E2E +
			float64(config.NetworkRTTUs) +
			computeUploadDelay(config.BandwidthMbps, sr.InputTokens) +
			computeDownloadDelay(config.BandwidthMbps, sr.OutputTokens)
		simEndUs := rec.SendTimeUs + int64(clientSimE2E)

		if minSend == -1 || rec.SendTimeUs < minSend {
			minSend = rec.SendTimeUs
		}
		if rec.LastChunkTimeUs > maxRealEnd {
			maxRealEnd = rec.LastChunkTimeUs
		}
		if simEndUs > maxSimEnd {
			maxSimEnd = simEndUs
		}
		realOutTok += rec.OutputTokens
		simOutTok += sr.OutputTokens
		matched++
	}

	if matched == 0 || minSend < 0 {
		return nil
	}
	realRuntimeSec := float64(maxRealEnd-minSend) / 1e6
	simRuntimeSec := float64(maxSimEnd-minSend) / 1e6
	// Guard against non-derivable makespans and empty token numerator (R11, I3): return nil
	// rather than emit an Inf/NaN throughput that would fail JSON marshaling and discard the
	// whole report.
	if realRuntimeSec <= 0 || simRuntimeSec <= 0 || realOutTok == 0 {
		return nil
	}

	tc := &ThroughputComparison{
		MatchedRequests:        matched,
		RealRuntimeSec:         realRuntimeSec,
		SimRuntimeSec:          simRuntimeSec,
		RealOutputTokens:       realOutTok,
		SimOutputTokens:        simOutTok,
		RealOutputTokensPerSec: float64(realOutTok) / realRuntimeSec,
		SimOutputTokensPerSec:  float64(simOutTok) / simRuntimeSec,
		RealRequestsPerSec:     float64(matched) / realRuntimeSec,
		SimRequestsPerSec:      float64(matched) / simRuntimeSec,
	}
	tc.OutputTokensPerSecError = tc.SimOutputTokensPerSec - tc.RealOutputTokensPerSec
	if tc.RealOutputTokensPerSec > 0 {
		tc.OutputTokensPerSecPercentError = math.Abs(tc.OutputTokensPerSecError) / tc.RealOutputTokensPerSec
	}
	tc.RequestsPerSecError = tc.SimRequestsPerSec - tc.RealRequestsPerSec
	if tc.RealRequestsPerSec > 0 {
		tc.RequestsPerSecPercentError = math.Abs(tc.RequestsPerSecError) / tc.RealRequestsPerSec
	}

	// Per-GPU normalization (numGPUs > 0). One GPU count divides both sides, so the per-GPU
	// PercentError is identical to the raw PercentError — reporting/comparability value only.
	if numGPUs > 0 {
		n := numGPUs
		realPerGPU := tc.RealOutputTokensPerSec / float64(numGPUs)
		simPerGPU := tc.SimOutputTokensPerSec / float64(numGPUs)
		tc.NumGPUs = &n
		tc.RealOutputTokensPerSecPerGPU = &realPerGPU
		tc.SimOutputTokensPerSecPerGPU = &simPerGPU
	}

	// Within-tolerance verdict on raw output-token throughput (tolerancePct > 0).
	if tolerancePct > 0 {
		tol := tolerancePct
		const boundaryEps = 1e-9
		within := tc.OutputTokensPerSecPercentError*100 <= tolerancePct+boundaryEps
		tc.TolerancePct = &tol
		tc.Within = &within
	}

	return tc
}

// GoodputComparisonReport summarizes observed-vs-simulated goodput per SLO class.
// All ratios are in [0, 1]; absolute counts/RPS use the natural scale.
type GoodputComparisonReport struct {
	Targets  map[string]SLODimTargets         `json:"targets"`
	PerClass map[string]GoodputClassComparison `json:"per_class"`
	// SkippedITL is true when --slo-itl was configured but ITL data was
	// missing on either side (real or sim); the ITL row is omitted with a
	// stderr warning.
	SkippedITL bool `json:"skipped_itl,omitempty"`
}

// GoodputClassComparison holds per-class observed/simulated goodput numbers.
type GoodputClassComparison struct {
	Count                   int     `json:"count"`
	RealSLOAttainment       float64 `json:"real_slo_attainment"`
	SimSLOAttainment        float64 `json:"sim_slo_attainment"`
	RealGoodputRPS          float64 `json:"real_goodput_rps"`
	SimGoodputRPS           float64 `json:"sim_goodput_rps"`
	RealAttainmentByDim     map[string]float64 `json:"real_attainment_by_dim,omitempty"`
	SimAttainmentByDim      map[string]float64 `json:"sim_attainment_by_dim,omitempty"`
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
// SLOClass, Model, and ITLMeanUs are optional — omitted from JSON when zero/empty
// so existing consumers that do not set these fields are unaffected (backward-compatible).
type SimResult struct {
	RequestID    int     `json:"request_id"`
	TTFT         float64 `json:"ttft_us"` // server-side TTFT in microseconds
	E2E          float64 `json:"e2e_us"`  // server-side E2E in microseconds
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	SLOClass     string  `json:"slo_class,omitempty"` // SLO tier (e.g., "standard", "batch"); empty if not set
	Model        string  `json:"model,omitempty"`     // model tag; empty if not set
	ITLMeanUs    float64 `json:"itl_mean_us,omitempty"` // mean ITL in microseconds; 0 if not recorded
}

// LatencyPair holds matched real-vs-sim latency vectors.
type LatencyPair struct {
	Real []float64
	Sim  []float64
}

// BreakdownPairs holds matched real-vs-sim latency vectors for a single breakdown dimension
// (e.g., one SLO class or one model tag). Tracks TTFT and E2E separately.
type BreakdownPairs struct {
	TTFT LatencyPair
	E2E  LatencyPair
}

// CalibrationPairs holds matched, normalized real-vs-sim latency vectors.
type CalibrationPairs struct {
	TTFT               LatencyPair
	E2E                LatencyPair
	ITL                LatencyPair
	BySLO              map[string]*BreakdownPairs // keyed by SLOClass; only populated when SLOClass is non-empty
	ByModel            map[string]*BreakdownPairs // keyed by Model; only populated when Model is non-empty
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

	pairs := &CalibrationPairs{
		BySLO:   make(map[string]*BreakdownPairs),
		ByModel: make(map[string]*BreakdownPairs),
	}
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
		// Use ServerInputTokens when available (handles prefix caching correctly)
		realInputTokens := rec.InputTokens
		if rec.ServerInputTokens > 0 {
			realInputTokens = rec.ServerInputTokens
		}

		if realInputTokens != sr.InputTokens || rec.OutputTokens != sr.OutputTokens {
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

		// Per-SLO breakdown (only when SLOClass is set)
		if sr.SLOClass != "" {
			bp, ok := pairs.BySLO[sr.SLOClass]
			if !ok {
				bp = &BreakdownPairs{}
				pairs.BySLO[sr.SLOClass] = bp
			}
			bp.TTFT.Real = append(bp.TTFT.Real, realTTFT)
			bp.TTFT.Sim = append(bp.TTFT.Sim, simTTFT)
			bp.E2E.Real = append(bp.E2E.Real, realE2E)
			bp.E2E.Sim = append(bp.E2E.Sim, simE2E)
		}
		// Per-model breakdown (only when Model is set)
		if sr.Model != "" {
			bp, ok := pairs.ByModel[sr.Model]
			if !ok {
				bp = &BreakdownPairs{}
				pairs.ByModel[sr.Model] = bp
			}
			bp.TTFT.Real = append(bp.TTFT.Real, realTTFT)
			bp.TTFT.Sim = append(bp.TTFT.Sim, simTTFT)
			bp.E2E.Real = append(bp.E2E.Real, realE2E)
			bp.E2E.Sim = append(bp.E2E.Sim, simE2E)
		}
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

	comp := &MetricComparison{Count: len(real)}

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
	if comp.WorkloadLevel.RealMean > 0 {
		comp.WorkloadLevel.MeanPercentError = math.Abs(comp.WorkloadLevel.MeanError) / comp.WorkloadLevel.RealMean
	}

	// Median error and percent error (with division guards, R11)
	comp.WorkloadLevel.MedianError = comp.WorkloadLevel.SimMedian - comp.WorkloadLevel.RealMedian
	if comp.WorkloadLevel.RealMedian > 0 {
		comp.WorkloadLevel.MedianPercentError = math.Abs(comp.WorkloadLevel.MedianError) / comp.WorkloadLevel.RealMedian
	}

	// P50 error and percent error (with division guards, R11)
	comp.WorkloadLevel.P50Error = comp.WorkloadLevel.SimP50 - comp.WorkloadLevel.RealP50
	if comp.WorkloadLevel.RealP50 > 0 {
		comp.WorkloadLevel.P50PercentError = math.Abs(comp.WorkloadLevel.P50Error) / comp.WorkloadLevel.RealP50
	}

	// P90 error and percent error (with division guards, R11)
	comp.WorkloadLevel.P90Error = comp.WorkloadLevel.SimP90 - comp.WorkloadLevel.RealP90
	if comp.WorkloadLevel.RealP90 > 0 {
		comp.WorkloadLevel.P90PercentError = math.Abs(comp.WorkloadLevel.P90Error) / comp.WorkloadLevel.RealP90
	}

	// P95 error and percent error (with division guards, R11)
	comp.WorkloadLevel.P95Error = comp.WorkloadLevel.SimP95 - comp.WorkloadLevel.RealP95
	if comp.WorkloadLevel.RealP95 > 0 {
		comp.WorkloadLevel.P95PercentError = math.Abs(comp.WorkloadLevel.P95Error) / comp.WorkloadLevel.RealP95
	}

	// P99 error and percent error (with division guards, R11)
	comp.WorkloadLevel.P99Error = comp.WorkloadLevel.SimP99 - comp.WorkloadLevel.RealP99
	if comp.WorkloadLevel.RealP99 > 0 {
		comp.WorkloadLevel.P99PercentError = math.Abs(comp.WorkloadLevel.P99Error) / comp.WorkloadLevel.RealP99
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
			"Token mismatch detection uses ServerInputTokens (server-reported prompt_tokens) when available. Non-zero token_mismatches on observe-generated prefix-cached traces typically reflect KV-block granularity rounding (simulator accounts tokens in multiples of BlockSizeTokens; server reports raw count). Expected variance is at most BlockSizeTokens tokens per request. This is expected behavior, not data corruption.",
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

// MapePct computes mean absolute percentage error between real and sim slices.
// Panics if len(real) != len(sim) — slices must be parallel vectors.
// Pairs where real==0, NaN, or Inf are skipped. Pairs where the computed
// absolute error is NaN or Inf (e.g., sim is NaN or Inf) are also skipped.
// Returns 0 if no valid pairs. Returns a fraction (not a percentage) — multiply by 100 for display.
func MapePct(real, sim []float64) float64 {
	if len(real) != len(sim) {
		panic(fmt.Sprintf("MapePct: real and sim slices must have equal length, got real=%d sim=%d", len(real), len(sim)))
	}
	var sum float64
	count := 0
	for i := range real {
		if real[i] == 0 || math.IsNaN(real[i]) || math.IsInf(real[i], 0) {
			continue
		}
		err := math.Abs(real[i]-sim[i]) / real[i]
		if math.IsNaN(err) || math.IsInf(err, 0) {
			continue
		}
		sum += err
		count++
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
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
