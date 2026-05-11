package workload

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/stat/distuv"
)

// ComputeSimEndUs computes the simulation end time from request completion times.
// Returns max(all completion times, horizon) where horizon is used as a floor if > 0.
// This is the canonical simEndUs calculation used by run, replay, and calibrate commands.
func ComputeSimEndUs(requests []*sim.Request, horizon int64) int64 {
	simEndUs := int64(0)
	for _, req := range requests {
		completionUs := req.ArrivalTime
		if req.TTFTSet {
			completionUs += req.FirstTokenTime
			// Only sum ITL if request has valid TTFT (prevents malformed data from inflating simEndUs)
			for _, itl := range req.ITL {
				completionUs += itl
			}
		}
		if completionUs > simEndUs {
			simEndUs = completionUs
		}
	}
	// Use horizon as floor if explicitly set and larger
	if horizon > 0 && horizon < math.MaxInt64 && horizon > simEndUs {
		simEndUs = horizon
	}
	return simEndUs
}

// RequestInterval represents a request's active period [ArrivalUs, CompletionUs).
// Used as input to backlog-drift saturation analysis (BC-2).
type RequestInterval struct {
	ArrivalUs    int64
	CompletionUs int64
}

// MetricsMode selects which metrics to use for saturation classification.
type MetricsMode string

const (
	// MetricsModeIntegral uses MeanInFlight and PeakInFlight (time-weighted, burst-robust).
	// This is the recommended mode for production use (issue #1316).
	MetricsModeIntegral MetricsMode = "integral"

	// MetricsModeBoundary uses ActiveEnd for both regression and peak detection (legacy).
	// Provided for backward compatibility and comparative analysis.
	MetricsModeBoundary MetricsMode = "boundary"
)

// BacklogDriftConfig configures the backlog-drift saturation analyzer.
type BacklogDriftConfig struct {
	WindowSize      time.Duration // Window width for sampling and per-window metrics (BC-1)
	MinWindows      int           // Minimum complete windows required for classification (BC-7)
	PeakRatio       float64       // Peak-to-mean threshold for TRANSIENT_BACKLOG detection (BC-6)
	PeakRatioBand   float64       // Confidence band around PeakRatio (±band creates borderline zone)
	ConfidenceCI    float64       // Confidence level for slope significance test (BC-3)
	MetricsMode     MetricsMode   // Which metrics to use: "integral" (default) or "boundary" (legacy)
}

// NewBacklogDriftConfig creates a BacklogDriftConfig with validation (BC-10, BC-14, R3).
// Panics if any parameter is invalid (NaN, Inf, out of range).
// MetricsMode defaults to MetricsModeIntegral if empty.
func NewBacklogDriftConfig(windowSize time.Duration, minWindows int, peakRatio, peakRatioBand, confidenceCI float64) BacklogDriftConfig {
	if windowSize <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: WindowSize must be > 0, got %v", windowSize))
	}
	if minWindows <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: MinWindows must be > 0, got %d", minWindows))
	}
	if peakRatio <= 0 || math.IsNaN(peakRatio) || math.IsInf(peakRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: PeakRatio must be a finite value > 0, got %f", peakRatio))
	}
	if peakRatioBand < 0 || math.IsNaN(peakRatioBand) || math.IsInf(peakRatioBand, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: PeakRatioBand must be >= 0, got %f", peakRatioBand))
	}
	if confidenceCI <= 0 || confidenceCI >= 1 || math.IsNaN(confidenceCI) || math.IsInf(confidenceCI, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: ConfidenceCI must be in (0, 1), got %f", confidenceCI))
	}
	return BacklogDriftConfig{
		WindowSize:     windowSize,
		MinWindows:     minWindows,
		PeakRatio:      peakRatio,
		PeakRatioBand:  peakRatioBand,
		ConfidenceCI:   confidenceCI,
		MetricsMode:    MetricsModeIntegral, // Default to new algorithm
	}
}

// DefaultBacklogDriftConfig returns the default configuration per issue #1298.
func DefaultBacklogDriftConfig() BacklogDriftConfig {
	return BacklogDriftConfig{
		WindowSize:     60 * time.Second,
		MinWindows:     5,
		PeakRatio:      2.0,
		PeakRatioBand:  0.2, // ±10% confidence band around threshold
		ConfidenceCI:   0.95,
		MetricsMode:    MetricsModeIntegral, // Default to new algorithm
	}
}

// WindowMetrics captures per-window saturation metrics (BC-1).
type WindowMetrics struct {
	StartUs      int64   // Window start timestamp (µs)
	EndUs        int64   // Window end timestamp (µs)
	NumEntered   int     // Requests with arrival in [start, end)
	NumLeft      int     // Requests with completion in [start, end)
	ActiveStart  int     // Active requests at window start
	ActiveEnd    int     // Active requests at window end
	DeltaBacklog int     // ActiveEnd - ActiveStart (change in backlog over window, BC-1)
	DrainRatio   float64 // NumLeft / NumEntered (NaN if NumEntered==0)

	// MeanInFlight is the time-weighted average in-flight request count over the window.
	// Unlike ActiveEnd (a boundary sample), MeanInFlight captures burst behavior within
	// the window via integral computation. Use this for robust saturation metrics.
	MeanInFlight float64

	// PeakInFlight is the maximum in-flight request count at any instant within the window.
	// Unlike max(ActiveEnd) across windows (which only samples boundaries), PeakInFlight
	// detects transient bursts that occur between window boundaries. Used for TRANSIENT_BACKLOG classification.
	PeakInFlight int
}

// BacklogDriftReport contains saturation classification results (BC-4, BC-5, BC-6, BC-7).
// CLI uses AnalyzeBacklogDrift() directly for saturation analysis.
type BacklogDriftReport struct {
	Classification string          `json:"classification"` // "UNSATURATED", "TRANSIENT_BACKLOG", "PERSISTENTLY_SATURATED"
	Slope          float64         `json:"slope"`          // Linear regression slope (req/µs)
	SlopeLower     float64         `json:"slope_lower"`    // Slope CI lower bound
	SlopeUpper     float64         `json:"slope_upper"`    // Slope CI upper bound
	InitialBacklog int             `json:"initial_backlog"`
	FinalBacklog   int             `json:"final_backlog"`
	PeakInFlight   int             `json:"peak_in_flight"`
	MeanInFlight   float64         `json:"mean_in_flight"`
	Windows        []WindowMetrics `json:"windows"`
	Note           string          `json:"note,omitempty"`           // Explanation (e.g., "observation too short")
	Recommendation string          `json:"recommendation,omitempty"` // User-facing guidance
}

// RequestsToIntervals converts sim.Request slices to RequestInterval slices for
// saturation analysis. Applies eligibility filter per BC-2:
//   - Exclude: TTFTSet==false AND (State==StateTimedOut OR State==StateQueued) (never executed)
//   - Include with simEndUs: TTFTSet==false AND State==StateRunning (horizon-truncated)
//   - Include with computed time: TTFTSet==true (completed)
//
// Logs included and excluded counts per BC-13 (R1: no silent discard).
func RequestsToIntervals(requests []*sim.Request, simEndUs int64) []RequestInterval {
	if len(requests) == 0 {
		return []RequestInterval{} // BC-15: degenerate input
	}

	intervals := make([]RequestInterval, 0, len(requests))
	excluded := 0

	for _, req := range requests {
		if !req.TTFTSet {
			// No valid TTFT — check state
			if req.State == sim.StateTimedOut || req.State == sim.StateQueued {
				// Case 1: Timed out before generating any output, or still queued at horizon — exclude
				excluded++
				continue
			}
			// Case 2: Horizon-truncated (StateRunning without TTFT) — still in-flight at horizon.
			// Use simEndUs+1 so half-open interval [arrival, simEndUs+1) is active at simEndUs.
			// This ensures these requests are correctly counted in ActiveEnd of the last window.
			intervals = append(intervals, RequestInterval{
				ArrivalUs:    req.ArrivalTime,
				CompletionUs: simEndUs + 1,
			})
		} else {
			// Case 3: Completed (TTFTSet==true) — compute completion time
			// Completion = FirstTokenTime (absolute) + Σ(inter-token latencies)
			// Note: FirstTokenTime is an absolute timestamp, not a duration from arrival
			completionUs := req.FirstTokenTime
			for _, itl := range req.ITL {
				completionUs += itl
			}
			intervals = append(intervals, RequestInterval{
				ArrivalUs:    req.ArrivalTime,
				CompletionUs: completionUs,
			})
		}
	}

	// BC-13: Log counts (R1 no silent discard)
	logrus.Infof("Saturation analysis: %d requests eligible, %d excluded (timed out or still queued)", len(intervals), excluded)

	return intervals
}

// computeWindowMetrics computes per-window saturation metrics per BC-1.
// Returns one WindowMetrics entry per complete window of size windowSizeUs.
// Computes DeltaBacklog = ActiveEnd - ActiveStart (change in backlog over window).
func computeWindowMetrics(intervals []RequestInterval, windowSizeUs, totalDurationUs int64) []WindowMetrics {
	if len(intervals) == 0 || totalDurationUs <= 0 || windowSizeUs <= 0 {
		return []WindowMetrics{}
	}

	// Guard against unreasonably large durations (e.g., MaxInt64 when no workload specified)
	// Cap at 7 days = 604800 seconds = 604800000000 microseconds
	const maxReasonableDurationUs int64 = 604800 * 1e6
	if totalDurationUs > maxReasonableDurationUs {
		logrus.Warnf("Saturation analysis: totalDurationUs (%d) exceeds reasonable limit (%d us = 7 days), skipping window metrics", totalDurationUs, maxReasonableDurationUs)
		return []WindowMetrics{}
	}

	numWindows := int((totalDurationUs + windowSizeUs - 1) / windowSizeUs) // Ceiling division
	windows := make([]WindowMetrics, numWindows)

	for i := 0; i < numWindows; i++ {
		startUs := int64(i) * windowSizeUs
		endUs := startUs + windowSizeUs
		if endUs > totalDurationUs {
			endUs = totalDurationUs
		}

		w := WindowMetrics{
			StartUs: startUs,
			EndUs:   endUs,
		}

		// Compute existing metrics by scanning all intervals
		for _, iv := range intervals {
			// NumEntered: arrival in [startUs, endUs)
			if iv.ArrivalUs >= startUs && iv.ArrivalUs < endUs {
				w.NumEntered++
			}
			// NumLeft: completion in [startUs, endUs)
			if iv.CompletionUs >= startUs && iv.CompletionUs < endUs {
				w.NumLeft++
			}
			// ActiveStart: interval contains startUs (arrival <= startUs < completion)
			if iv.ArrivalUs <= startUs && startUs < iv.CompletionUs {
				w.ActiveStart++
			}
			// ActiveEnd: interval contains endUs
			if iv.ArrivalUs <= endUs && endUs < iv.CompletionUs {
				w.ActiveEnd++
			}
		}

		// Compute integral-based metrics (BC-1, BC-2)
		// Step 1: Collect transition events within [startUs, endUs)
		type event struct {
			timeUs int64
			delta  int // +1 for arrival, -1 for departure
		}
		events := make([]event, 0, len(intervals)*2)
		for _, iv := range intervals {
			// Arrival event
			if iv.ArrivalUs >= startUs && iv.ArrivalUs < endUs {
				events = append(events, event{timeUs: iv.ArrivalUs, delta: +1})
			}
			// Departure event
			if iv.CompletionUs >= startUs && iv.CompletionUs < endUs {
				events = append(events, event{timeUs: iv.CompletionUs, delta: -1})
			}
		}

		// Step 2: Sort events by time (Go's sort.Slice is stable, so arrivals before departures at same time)
		sort.Slice(events, func(i, j int) bool {
			if events[i].timeUs != events[j].timeUs {
				return events[i].timeUs < events[j].timeUs
			}
			// Stable sort ensures original order preserved for same timestamp
			// If needed, explicit tiebreaker: arrivals (+1) before departures (-1)
			return events[i].delta > events[j].delta
		})

		// Step 3: Walk events to compute integral and peak
		currentCount := w.ActiveStart // Requests active at window start
		area := int64(0)
		peakCount := currentCount
		prevTimeUs := startUs

		for _, ev := range events {
			// Accumulate area of rectangle [prevTimeUs, ev.timeUs) at height currentCount
			deltaTime := ev.timeUs - prevTimeUs
			area += int64(currentCount) * deltaTime
			if currentCount > peakCount {
				peakCount = currentCount
			}

			// Update count
			currentCount += ev.delta
			if currentCount > peakCount {
				peakCount = currentCount
			}

			prevTimeUs = ev.timeUs
		}

		// Final segment [lastEvent, endUs) at height currentCount
		deltaTime := endUs - prevTimeUs
		area += int64(currentCount) * deltaTime
		if currentCount > peakCount {
			peakCount = currentCount
		}

		// Step 4: Compute derived metrics
		windowDuration := endUs - startUs
		if windowDuration > 0 {
			w.MeanInFlight = float64(area) / float64(windowDuration)
		} else {
			// Degenerate case (should not occur): zero-duration window
			w.MeanInFlight = 0.0
		}
		w.PeakInFlight = peakCount

		// DeltaBacklog and DrainRatio
		w.DeltaBacklog = w.ActiveEnd - w.ActiveStart
		if w.NumEntered > 0 {
			w.DrainRatio = float64(w.NumLeft) / float64(w.NumEntered)
		} else {
			w.DrainRatio = math.NaN() // Undefined when no arrivals
		}

		windows[i] = w
	}

	return windows
}

// fitSlopeRegression fits linear regression active(t) = a + b*t with time normalization (BC-3).
// Returns slope (requests per µs), and (lower, upper) confidence interval bounds.
// Time is normalized to [0,1] for numerical stability, then slope is un-normalized.
func fitSlopeRegression(samples []struct {
	timeUs int64
	count  int
}, totalDurationUs int64, confidenceCI float64) (slope, lower, upper float64) {
	if len(samples) == 0 || totalDurationUs <= 0 {
		return 0, 0, 0
	}

	n := float64(len(samples))

	// Normalize time to [0, 1] for numerical stability
	var sumT, sumY, sumTY, sumTT float64
	for _, s := range samples {
		t := float64(s.timeUs) / float64(totalDurationUs) // Normalized time in [0, 1]
		y := float64(s.count)
		sumT += t
		sumY += y
		sumTY += t * y
		sumTT += t * t
	}

	// Ordinary least squares: slope = (n*sumTY - sumT*sumY) / (n*sumTT - sumT^2)
	// intercept = (sumY - slope*sumT) / n
	denominator := n*sumTT - sumT*sumT
	if math.Abs(denominator) < 1e-12 {
		// Degenerate case: all times identical
		return 0, 0, 0
	}

	slopeNorm := (n*sumTY - sumT*sumY) / denominator
	intercept := (sumY - slopeNorm*sumT) / n

	// Compute residual sum of squares for standard error
	var rss float64
	for _, s := range samples {
		t := float64(s.timeUs) / float64(totalDurationUs)
		y := float64(s.count)
		predicted := intercept + slopeNorm*t
		residual := y - predicted
		rss += residual * residual
	}

	// Standard error of slope
	if n <= 2 {
		// Insufficient degrees of freedom
		return slopeNorm / float64(totalDurationUs), 0, 0
	}
	mse := rss / (n - 2)
	seSlope := math.Sqrt(mse / (n*sumTT - sumT*sumT))

	// t-distribution critical value for confidence interval
	// Use exact t-distribution quantile based on degrees of freedom (n-2)
	// and confidence level (two-tailed, so use (1 + confidenceCI) / 2 for upper tail)
	df := n - 2 // degrees of freedom
	tDist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: float64(df)}
	upperTailProb := (1 + confidenceCI) / 2 // e.g., 0.95 → 0.975 for two-tailed test
	tCritical := tDist.Quantile(upperTailProb)

	// Confidence interval on normalized slope
	marginNorm := tCritical * seSlope

	// Un-normalize slope: slope_original = slope_normalized / totalDurationUs
	slopeOrig := slopeNorm / float64(totalDurationUs)
	marginOrig := marginNorm / float64(totalDurationUs)

	return slopeOrig, slopeOrig - marginOrig, slopeOrig + marginOrig
}

// classifyBacklogDrift determines saturation classification per BC-4/5/6/7.
// Returns (classification, note, recommendation).
func classifyBacklogDrift(slope, slopeLower, slopeUpper float64,
	initialBacklog, finalBacklog, peakInFlight int, meanInFlight float64,
	cfg BacklogDriftConfig) (classification, note, recommendation string) {

	// BC-5: PERSISTENTLY_SATURATED — slope CI excludes zero (lower > 0)
	if slopeLower > 0 {
		classification = "PERSISTENTLY_SATURATED"
		note = fmt.Sprintf("Backlog grew persistently (slope=%.2e req/µs, CI=[%.2e, %.2e] excludes zero). "+
			"Initial=%d, Final=%d, Peak=%d.",
			slope, slopeLower, slopeUpper, initialBacklog, finalBacklog, peakInFlight)
		recommendation = "System is overloaded. Add capacity, reduce load, or increase request timeouts."
		return
	}

	// Guard against zero mean (prevents NaN in user-facing note)
	if meanInFlight == 0 {
		classification = "UNSATURATED"
		note = "No active requests observed in any window."
		recommendation = "System handled load without saturation. Current capacity is adequate."
		return
	}

	// BC-6: TRANSIENT_BACKLOG — slope CI includes zero but peak exceeds threshold
	peakRatio := float64(peakInFlight) / meanInFlight

	// Confidence band logic: Use tiebreaker when ratio is in borderline zone
	lowerBound := cfg.PeakRatio - cfg.PeakRatioBand
	upperBound := cfg.PeakRatio + cfg.PeakRatioBand

	if peakRatio > upperBound {
		// Clearly above threshold → TRANSIENT_BACKLOG
		classification = "TRANSIENT_BACKLOG"
		note = fmt.Sprintf("Backlog did not grow on average (slope CI includes zero), but peak in-flight (%d) "+
			"exceeded %.1f× mean (%.1f). Ratio=%.2f > %.2f.",
			peakInFlight, cfg.PeakRatio, meanInFlight, peakRatio, cfg.PeakRatio)
		recommendation = "System experienced transient congestion. Consider increasing burst capacity or smoothing load."
		return
	} else if peakRatio >= lowerBound && peakRatio <= upperBound {
		// Borderline zone [threshold-band, threshold+band] → use slope as tiebreaker
		if slope > 0 {
			// Positive slope (getting worse) → classify as TRANSIENT
			classification = "TRANSIENT_BACKLOG"
			note = fmt.Sprintf("Peak/mean ratio (%.2f) is borderline (threshold %.2f ± %.2f band). "+
				"Using positive slope (%.2e) as tiebreaker → TRANSIENT_BACKLOG.",
				peakRatio, cfg.PeakRatio, cfg.PeakRatioBand, slope)
			recommendation = "System is experiencing borderline congestion with worsening trend. Monitor closely."
			return
		} else {
			// Negative or zero slope (stable/recovering) → classify as UNSATURATED
			classification = "UNSATURATED"
			note = fmt.Sprintf("Peak/mean ratio (%.2f) is borderline (threshold %.2f ± %.2f band). "+
				"Using non-positive slope (%.2e) as tiebreaker → UNSATURATED.",
				peakRatio, cfg.PeakRatio, cfg.PeakRatioBand, slope)
			recommendation = "System handled load adequately. Peak was borderline but backlog is not growing."
			return
		}
	}

	// BC-4: UNSATURATED — slope CI excludes positive or includes zero with low peak
	classification = "UNSATURATED"
	if slopeUpper < 0 {
		note = fmt.Sprintf("Backlog decreased (slope=%.2e req/µs, CI=[%.2e, %.2e] excludes zero). "+
			"Initial=%d, Final=%d, Peak=%d.",
			slope, slopeLower, slopeUpper, initialBacklog, finalBacklog, peakInFlight)
	} else {
		note = fmt.Sprintf("Backlog remained stable (slope=%.2e req/µs, CI=[%.2e, %.2e] includes zero). "+
			"Peak/mean ratio=%.2f <= %.2f.",
			slope, slopeLower, slopeUpper, peakRatio, cfg.PeakRatio)
	}
	recommendation = "System handled load without saturation. Current capacity is adequate."
	return
}

// AnalyzeBacklogDrift performs end-to-end backlog-drift saturation analysis (BC-8).
// Orchestrates: eligibility filter → window metrics → regression → classification.
// Returns UNSATURATED with note if observation has fewer than MinWindows complete windows (BC-7).
func AnalyzeBacklogDrift(requests []*sim.Request, simEndUs int64, cfg BacklogDriftConfig) BacklogDriftReport {
	// Step 1: Filter eligible requests (BC-2, BC-13)
	intervals := RequestsToIntervals(requests, simEndUs)

	// BC-15: Early exit if all requests were excluded
	if len(intervals) == 0 {
		return BacklogDriftReport{
			Classification: "UNSATURATED",
			Note:           "no eligible requests for saturation analysis",
			Recommendation: "Check that requests completed successfully and were not all timed out before first token.",
		}
	}

	// Step 2: Compute per-window metrics (BC-1)
	windowSizeUs := int64(cfg.WindowSize / time.Microsecond)
	windows := computeWindowMetrics(intervals, windowSizeUs, simEndUs)

	// BC-7: Early exit if insufficient windows
	if len(windows) < cfg.MinWindows {
		minDurationSec := float64(cfg.MinWindows) * cfg.WindowSize.Seconds()
		return BacklogDriftReport{
			Classification: "UNSATURATED",
			Note: fmt.Sprintf("Observation too short: %d windows < MinWindows=%d. Cannot reliably classify saturation. "+
				"Conservatively marking as UNSATURATED.",
				len(windows), cfg.MinWindows),
			Recommendation: fmt.Sprintf("Run longer observations (at least %.0f seconds) for reliable classification.", minDurationSec),
			Windows:        windows,
		}
	}

	// Step 3: Prepare time series samples for regression
	samples := make([]struct {
		timeUs int64
		count  int
	}, len(windows))
	for i, w := range windows {
		// Use window midpoint as time coordinate
		samples[i].timeUs = (w.StartUs + w.EndUs) / 2
		// Select regression metric based on configured mode
		if cfg.MetricsMode == MetricsModeBoundary {
			// Legacy mode: Use boundary sample (ActiveEnd)
			samples[i].count = w.ActiveEnd
		} else {
			// Integral mode (default): Use time-averaged load (MeanInFlight)
			// BC-5: Captures true load including sub-window bursts
			samples[i].count = int(math.Round(w.MeanInFlight))
		}
	}

	// Step 4: Fit linear regression (BC-3)
	slope, slopeLower, slopeUpper := fitSlopeRegression(samples, simEndUs, cfg.ConfidenceCI)

	// Step 5: Compute summary statistics
	initialBacklog := windows[0].ActiveStart
	finalBacklog := windows[len(windows)-1].ActiveEnd
	peakInFlight := 0
	var sumInFlight float64

	for _, w := range windows {
		// Select peak/mean metrics based on configured mode
		if cfg.MetricsMode == MetricsModeBoundary {
			// Legacy mode: Use boundary samples
			if w.ActiveEnd > peakInFlight {
				peakInFlight = w.ActiveEnd
			}
			sumInFlight += float64(w.ActiveEnd)
		} else {
			// Integral mode (default): Use integral-based metrics
			// BC-6: Captures true intra-window peak and time-weighted mean
			if w.PeakInFlight > peakInFlight {
				peakInFlight = w.PeakInFlight
			}
			sumInFlight += w.MeanInFlight
		}
	}
	meanInFlight := sumInFlight / float64(len(windows))

	// Step 6: Classify saturation state (BC-4/5/6)
	classification, note, recommendation := classifyBacklogDrift(
		slope, slopeLower, slopeUpper,
		initialBacklog, finalBacklog, peakInFlight, meanInFlight,
		cfg,
	)

	return BacklogDriftReport{
		Classification: classification,
		Slope:          slope,
		SlopeLower:     slopeLower,
		SlopeUpper:     slopeUpper,
		InitialBacklog: initialBacklog,
		FinalBacklog:   finalBacklog,
		PeakInFlight:   peakInFlight,
		MeanInFlight:   meanInFlight,
		Windows:        windows,
		Note:           note,
		Recommendation: recommendation,
	}
}

// sanitizeReport replaces NaN values with 0 (JSON doesn't support NaN).
func sanitizeReport(report BacklogDriftReport) BacklogDriftReport {
	if math.IsNaN(report.Slope) {
		report.Slope = 0
	}
	if math.IsNaN(report.SlopeLower) {
		report.SlopeLower = 0
	}
	if math.IsNaN(report.SlopeUpper) {
		report.SlopeUpper = 0
	}
	if math.IsNaN(report.MeanInFlight) {
		report.MeanInFlight = 0
	}

	// Sanitize per-window DrainRatio values
	for i := range report.Windows {
		if math.IsNaN(report.Windows[i].DrainRatio) {
			report.Windows[i].DrainRatio = 0
		}
	}

	return report
}

// WriteBacklogDriftReportJSON writes a BacklogDriftReport to a JSON file (BC-9).
// Returns error if writing fails.
// NaN values are sanitized to 0 before marshaling (JSON doesn't support NaN).
func WriteBacklogDriftReportJSON(path string, report BacklogDriftReport) error {
	// Sanitize NaN values (JSON doesn't support NaN)
	report = sanitizeReport(report)

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal BacklogDriftReport to JSON: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON to %s: %w", path, err)
	}
	return nil
}

// ReadBacklogDriftReportJSON reads a BacklogDriftReport from a JSON file (BC-9).
// Returns error if reading or unmarshaling fails.
func ReadBacklogDriftReportJSON(path string) (BacklogDriftReport, error) {
	var report BacklogDriftReport
	data, err := os.ReadFile(path)
	if err != nil {
		return report, fmt.Errorf("failed to read JSON from %s: %w", path, err)
	}
	if err := json.Unmarshal(data, &report); err != nil {
		return report, fmt.Errorf("failed to unmarshal JSON from %s: %w", path, err)
	}
	return report, nil
}
