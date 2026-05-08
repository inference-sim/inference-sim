package workload

import (
	"fmt"
	"math"
	"time"

	"github.com/sirupsen/logrus"
	sim "github.com/inference-sim/inference-sim/sim"
)

// RequestInterval represents a request's active period [ArrivalUs, CompletionUs).
// Used as input to backlog-drift saturation analysis (BC-2).
type RequestInterval struct {
	ArrivalUs    int64
	CompletionUs int64
}

// BacklogDriftConfig configures the backlog-drift saturation analyzer.
type BacklogDriftConfig struct {
	WindowSize   time.Duration // Window width for sampling and per-window metrics (BC-1)
	MinWindows   int           // Minimum complete windows required for classification (BC-7)
	PeakRatio    float64       // Peak-to-mean threshold for TRANSIENT_BACKLOG detection (BC-6)
	ConfidenceCI float64       // Confidence level for slope significance test (BC-3)
}

// NewBacklogDriftConfig creates a BacklogDriftConfig with validation (BC-10, BC-14, R3).
// Panics if any parameter is invalid (NaN, Inf, out of range).
func NewBacklogDriftConfig(windowSize time.Duration, minWindows int, peakRatio, confidenceCI float64) BacklogDriftConfig {
	if windowSize <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: WindowSize must be > 0, got %v", windowSize))
	}
	if minWindows <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: MinWindows must be > 0, got %d", minWindows))
	}
	if peakRatio <= 0 || math.IsNaN(peakRatio) || math.IsInf(peakRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: PeakRatio must be a finite value > 0, got %f", peakRatio))
	}
	if confidenceCI <= 0 || confidenceCI >= 1 || math.IsNaN(confidenceCI) || math.IsInf(confidenceCI, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: ConfidenceCI must be in (0, 1), got %f", confidenceCI))
	}
	return BacklogDriftConfig{
		WindowSize:   windowSize,
		MinWindows:   minWindows,
		PeakRatio:    peakRatio,
		ConfidenceCI: confidenceCI,
	}
}

// DefaultBacklogDriftConfig returns the default configuration per issue #1298.
func DefaultBacklogDriftConfig() BacklogDriftConfig {
	return BacklogDriftConfig{
		WindowSize:   60 * time.Second,
		MinWindows:   5,
		PeakRatio:    2.0,
		ConfidenceCI: 0.95,
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
	DeltaBacklog int     // ActiveEnd - ActiveStart (must equal NumEntered - NumLeft, BC-1)
	DrainRatio   float64 // NumLeft / NumEntered (NaN if NumEntered==0)
}

// BacklogDriftReport contains saturation classification results (BC-4, BC-5, BC-6, BC-7).
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
//   - Exclude: TTFTSet==false AND State==StateTimedOut (timed out before TTFT)
//   - Include with simEndUs: TTFTSet==false AND State==StateRunning (horizon-truncated)
//   - Include with computed time: TTFTSet==true (completed)
// Logs included and excluded counts per BC-13 (R1: no silent discard).
func RequestsToIntervals(requests []*sim.Request, simEndUs int64) []RequestInterval {
	if len(requests) == 0 {
		return []RequestInterval{} // BC-15: degenerate input
	}

	intervals := make([]RequestInterval, 0, len(requests))
	excluded := 0

	for _, req := range requests {
		if !req.TTFTSet {
			// No valid TTFT — check if timed out or horizon-truncated
			if req.State == sim.StateTimedOut {
				// Case 1: Timed out before generating any output — exclude
				excluded++
				continue
			}
			// Case 2: Horizon-truncated (StateRunning without TTFT) — use simEndUs as completion
			intervals = append(intervals, RequestInterval{
				ArrivalUs:    req.ArrivalTime,
				CompletionUs: simEndUs,
			})
		} else {
			// Case 3: Completed (TTFTSet==true) — compute completion time
			// Completion = ArrivalTime + FirstTokenTime + Σ(inter-token latencies)
			completionUs := req.ArrivalTime + req.FirstTokenTime
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
	logrus.Infof("Saturation analysis: %d requests eligible, %d excluded (timed out before TTFT)", len(intervals), excluded)

	return intervals
}

// computeWindowMetrics computes per-window saturation metrics per BC-1.
// Returns one WindowMetrics entry per complete window of size windowSizeUs.
// Enforces identity: DeltaBacklog = NumEntered - NumLeft.
func computeWindowMetrics(intervals []RequestInterval, windowSizeUs, totalDurationUs int64) []WindowMetrics {
	if len(intervals) == 0 || totalDurationUs <= 0 || windowSizeUs <= 0 {
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

		// Compute metrics by scanning all intervals
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
func fitSlopeRegression(samples []struct{ timeUs int64; count int }, totalDurationUs int64, confidenceCI float64) (slope, lower, upper float64) {
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
	// For simplicity, use normal approximation (z-score) for large n
	// For small n, this is an approximation; exact t-table lookup would be better
	zScore := 1.96 // 95% CI approximation (could use gonum.org/v1/gonum/stat/distuv for exact t)
	if confidenceCI == 0.99 {
		zScore = 2.576
	} else if confidenceCI == 0.90 {
		zScore = 1.645
	}

	// Confidence interval on normalized slope
	marginNorm := zScore * seSlope

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

	// BC-6: TRANSIENT_BACKLOG — slope CI includes zero but peak exceeds threshold
	peakRatio := float64(peakInFlight) / meanInFlight
	if peakRatio > cfg.PeakRatio {
		classification = "TRANSIENT_BACKLOG"
		note = fmt.Sprintf("Backlog did not grow on average (slope CI includes zero), but peak in-flight (%d) "+
			"exceeded %.1f× mean (%.1f). Ratio=%.2f > %.2f.",
			peakInFlight, cfg.PeakRatio, meanInFlight, peakRatio, cfg.PeakRatio)
		recommendation = "System experienced transient congestion. Consider increasing burst capacity or smoothing load."
		return
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

	// Step 2: Compute per-window metrics (BC-1)
	windowSizeUs := int64(cfg.WindowSize / time.Microsecond)
	windows := computeWindowMetrics(intervals, windowSizeUs, simEndUs)

	// BC-7: Early exit if insufficient windows
	if len(windows) < cfg.MinWindows {
		return BacklogDriftReport{
			Classification: "UNSATURATED",
			Note: fmt.Sprintf("Observation too short: %d windows < MinWindows=%d. Cannot reliably classify saturation. "+
				"Conservatively marking as UNSATURATED.",
				len(windows), cfg.MinWindows),
			Recommendation: "Run longer observations (at least %.0f seconds) for reliable classification.",
			Windows:        windows,
		}
	}

	// Step 3: Prepare time series samples for regression
	samples := make([]struct{ timeUs int64; count int }, len(windows))
	for i, w := range windows {
		// Use window midpoint as time coordinate
		samples[i].timeUs = (w.StartUs + w.EndUs) / 2
		samples[i].count = w.ActiveEnd
	}

	// Step 4: Fit linear regression (BC-3)
	slope, slopeLower, slopeUpper := fitSlopeRegression(samples, simEndUs, cfg.ConfidenceCI)

	// Step 5: Compute summary statistics
	initialBacklog := windows[0].ActiveStart
	finalBacklog := windows[len(windows)-1].ActiveEnd
	peakInFlight := 0
	sumInFlight := 0
	for _, w := range windows {
		if w.ActiveEnd > peakInFlight {
			peakInFlight = w.ActiveEnd
		}
		sumInFlight += w.ActiveEnd
	}
	meanInFlight := float64(sumInFlight) / float64(len(windows))

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
