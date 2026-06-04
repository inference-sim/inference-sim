package workload

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
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

// BacklogDriftConfig configures the backlog-drift saturation analyzer.
type BacklogDriftConfig struct {
	WindowSize      time.Duration // Window width for sampling and per-window metrics (BC-1)
	MinWindows      int           // Minimum complete windows required for classification (BC-7)
	PeakRatio       float64       // Peak-to-mean threshold for TRANSIENT_BACKLOG detection (BC-6)
	PeakRatioBand   float64       // Confidence band around PeakRatio (±band creates borderline zone)
	ConfidenceCI    float64       // Confidence level for slope significance test (BC-3)

	// Drain-ratio classifier knobs (#1392). Used only by drainRatioClassifier; ignored
	// by slopeBasedClassifier. Validation in NewBacklogDriftConfig enforces the relation
	// SaturatedDrainRatio <= TransientDrainRatio so classification regions don't overlap.
	WarmupWindows       int     // Inject windows skipped at the start (engine ramp-up)
	TailWindows         int     // Inject windows skipped at the end (rate ramp-down boundary)
	SaturatedDrainRatio float64 // Mean DrainRatio < this → PERSISTENTLY_SATURATED
	TransientDrainRatio float64 // Mean DrainRatio < this → TRANSIENT_BACKLOG
}

// NewBacklogDriftConfig creates a BacklogDriftConfig with validation (BC-10, BC-14, R3).
// Panics if any parameter is invalid (NaN, Inf, out of range).
//
// warmupWindows must be >= 0. saturatedDrainRatio and transientDrainRatio must each be
// in (0, 1]; saturatedDrainRatio <= transientDrainRatio so PERSISTENTLY_SATURATED and
// TRANSIENT_BACKLOG regions form a contiguous partition of [0, 1].
func NewBacklogDriftConfig(
	windowSize time.Duration,
	minWindows int,
	peakRatio, peakRatioBand, confidenceCI float64,
	warmupWindows, tailWindows int,
	saturatedDrainRatio, transientDrainRatio float64,
) BacklogDriftConfig {
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
	if warmupWindows < 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: WarmupWindows must be >= 0, got %d", warmupWindows))
	}
	if tailWindows < 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: TailWindows must be >= 0, got %d", tailWindows))
	}
	if saturatedDrainRatio <= 0 || saturatedDrainRatio > 1 || math.IsNaN(saturatedDrainRatio) || math.IsInf(saturatedDrainRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: SaturatedDrainRatio must be in (0, 1], got %f", saturatedDrainRatio))
	}
	if transientDrainRatio <= 0 || transientDrainRatio > 1 || math.IsNaN(transientDrainRatio) || math.IsInf(transientDrainRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: TransientDrainRatio must be in (0, 1], got %f", transientDrainRatio))
	}
	if saturatedDrainRatio > transientDrainRatio {
		panic(fmt.Sprintf("BacklogDriftConfig: SaturatedDrainRatio (%f) must be <= TransientDrainRatio (%f); regions would overlap", saturatedDrainRatio, transientDrainRatio))
	}
	return BacklogDriftConfig{
		WindowSize:          windowSize,
		MinWindows:          minWindows,
		PeakRatio:           peakRatio,
		PeakRatioBand:       peakRatioBand,
		ConfidenceCI:        confidenceCI,
		WarmupWindows:       warmupWindows,
		TailWindows:         tailWindows,
		SaturatedDrainRatio: saturatedDrainRatio,
		TransientDrainRatio: transientDrainRatio,
	}
}

// DefaultBacklogDriftConfig returns the default configuration per issues #1298, #1392.
//
// WarmupWindows=2 and TailWindows=1 are empirical defaults from a Llama-3.1-70B
// reference experiment (rate=80, num_requests=6000):
//   - Window 1 was a clear engine ramp-up (DrainRatio ≈ 0.6) before steady state.
//   - The window where inject ends mid-bucket has artificially low NumEntered
//     and full NumLeft (engine continues draining backlog), pushing DrainRatio > 1
//     and biasing the steady-state mean upward toward "unsaturated".
//
// Routes through NewBacklogDriftConfig so the defaults are self-validating; if a
// future change introduces an inter-field invariant, this function will panic at
// init time rather than silently producing an inconsistent default config.
func DefaultBacklogDriftConfig() BacklogDriftConfig {
	return NewBacklogDriftConfig(
		60*time.Second, // WindowSize
		5,              // MinWindows
		2.0,            // PeakRatio
		0.2,            // PeakRatioBand (absolute, ≈ 10% of PeakRatio)
		0.95,           // ConfidenceCI
		2,              // WarmupWindows
		1,              // TailWindows
		0.95,           // SaturatedDrainRatio: mean DrainRatio < this → PERSISTENTLY_SATURATED
		0.98,           // TransientDrainRatio: mean DrainRatio < this → TRANSIENT_BACKLOG
	)
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
}

// BacklogDriftReport contains saturation classification results (BC-4, BC-5, BC-6, BC-7).
// Returned by both AnalyzeBacklogDriftWithClassifier (preferred, classifier-aware) and
// AnalyzeBacklogDrift (backward-compat shim defaulting to slope-based).
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
// saturation analysis. Applies eligibility filter per BC-2 (#1389):
//   - Exclude: TTFTSet==false AND State==StateTimedOut (never executed and gave up)
//   - Include with simEndUs+1: TTFTSet==false AND State IN (StateRunning, StateQueued)
//     (still in-flight at horizon — for backlog analysis, what matters is that the
//     request arrived and never finished, regardless of whether it was scheduled)
//   - Include with computed time: TTFTSet==true (completed normally OR timed out
//     mid-generation — the latter is rare since BLIS deadlines fire at req.Deadline,
//     not mid-step, but its computed completion is still meaningful for backlog
//     accounting because the request occupied the engine for ITL duration)
//
// Including queued-at-horizon requests is essential for backlog analysis (#1389):
// they are precisely the evidence of saturation. Excluding them silently drops
// the proof that the engine couldn't keep up with arrivals.
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
			if req.State == sim.StateTimedOut {
				// Case 1: Timed out before generating any output — exclude
				// (the request gave up, so it's not in-flight at horizon)
				excluded++
				continue
			}
			// Case 2: Still in-flight at horizon (StateRunning or StateQueued).
			// Use simEndUs+1 so half-open interval [arrival, simEndUs+1) is active at simEndUs.
			// This ensures these requests are correctly counted in ActiveEnd of the last window.
			intervals = append(intervals, RequestInterval{
				ArrivalUs:    req.ArrivalTime,
				CompletionUs: simEndUs + 1,
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
	logrus.Infof("Saturation analysis: %d requests eligible, %d excluded (timed out)", len(intervals), excluded)

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

// NewBacklogClassifier returns a BacklogClassifier by name (#1391).
//
// Names: "slope-based" (historical CI-on-slope predicate), "drain-ratio" (new default,
// queueing-theory ρ from per-window NumLeft/NumEntered). The empty string maps to the
// current default — drain-ratio.
//
// Panics on unknown name. CLI is expected to validate via sim.IsValidBacklogClassifier
// upstream and emit a friendly logrus.Fatalf — this panic is a backstop that should
// only trigger from misuse in library code.
func NewBacklogClassifier(name string) BacklogClassifier {
	switch name {
	case "", "drain-ratio":
		return drainRatioClassifier{}
	case "slope-based":
		return slopeBasedClassifier{}
	default:
		panic(fmt.Sprintf("unknown backlog classifier %q", name))
	}
}

// SlopeStats bundles the slope regression output for classifiers that want it.
// Computed once in AnalyzeBacklogDriftWithClassifier and passed to Classify.
// Reserved for additive extension — adding fields does not break existing implementations.
type SlopeStats struct {
	Slope      float64 // req/µs (slope of ActiveEnd over inject-phase window-midpoint times)
	SlopeLower float64 // CI lower bound at cfg.ConfidenceCI
	SlopeUpper float64 // CI upper bound at cfg.ConfidenceCI
}

// BacklogClassifier produces a saturation classification from per-window backlog
// metrics and slope statistics. Implementations may use any combination of inputs.
//
// Registered via NewBacklogClassifier(name); validate names with IsValidBacklogClassifier.
// Used by AnalyzeBacklogDriftWithClassifier to decide the final classification string.
type BacklogClassifier interface {
	Classify(
		windows []WindowMetrics,
		slope SlopeStats,
		initialBacklog, finalBacklog, peakInFlight int,
		meanInFlight float64,
		cfg BacklogDriftConfig,
	) (classification, note, recommendation string)
}

// slopeBasedClassifier preserves the historical classification logic (BC-4/5/6/7).
// Decides among UNSATURATED / TRANSIENT_BACKLOG / PERSISTENTLY_SATURATED based on
// the slope CI of ActiveEnd over inject-phase windows and the peak/mean ratio.
type slopeBasedClassifier struct{}

// Classify implements BacklogClassifier. See classifyBacklogDriftSlopeBased docstring.
//
// Practical-significance guard: with many windows the OLS CI shrinks proportionally
// to 1/√N, so even floating-point noise can produce a "statistically significant"
// positive slope. We require the projected drift over the observed duration to
// exceed a noise floor before firing PERSISTENTLY_SATURATED.
//
// The noise floor scales as max(1, sqrt(durationSec)) — heuristically motivated by
// random-walk variance in queueing systems (cumulative drift from Poisson
// fluctuation grows ~ sqrt(t)). This filters false-positive PERSISTENTLY_SATURATED
// at very low offered rates where finite-budget BLIS sims have long inject
// durations (e.g., rate=2 with num_requests=6000 → inject ≈ 3000s; even tiny per-µs
// slopes accumulate to >1 request). Genuine saturation produces drift orders of
// magnitude above this floor.
func (slopeBasedClassifier) Classify(
	windows []WindowMetrics,
	slope SlopeStats,
	initialBacklog, finalBacklog, peakInFlight int,
	meanInFlight float64,
	cfg BacklogDriftConfig,
) (classification, note, recommendation string) {
	// Practical-significance guard. If the slope's CI excludes zero but the projected
	// total drift is below the noise floor, suppress the PERSISTENTLY_SATURATED gate by
	// passing the equivalent of a slope_lower<=0 sentinel. The peak/mean ratio test
	// still applies.
	effectiveSlopeLower := slope.SlopeLower
	if effectiveSlopeLower > 0 && len(windows) > 0 {
		observedDurationUs := windows[len(windows)-1].EndUs - windows[0].StartUs
		projectedDrift := slope.Slope * float64(observedDurationUs)
		// Noise floor scales with sqrt of observation duration — see docstring above.
		durationSec := float64(observedDurationUs) / 1e6
		driftThreshold := math.Max(1.0, math.Sqrt(durationSec))
		if projectedDrift < driftThreshold {
			effectiveSlopeLower = 0
		}
	}
	return classifyBacklogDriftSlopeBased(
		slope.Slope, effectiveSlopeLower, slope.SlopeUpper,
		initialBacklog, finalBacklog, peakInFlight, meanInFlight, cfg,
	)
}

// drainRatioClassifier classifies saturation by averaging per-window DrainRatio over
// steady-state inject windows (#1392). DrainRatio = NumLeft / NumEntered is a direct
// measurement of μ/λ per window; its inverse is utilization ρ.
//
// Logic:
//  1. Identify "inject windows" via the last_arrival_window predicate — every window
//     up to and including the highest-indexed window with NumEntered > 0. This
//     classifies interior gaps (closed-loop / bursty workloads with intermittent
//     arrivals) as inject-phase, only the trailing all-zero tail as drain.
//  2. Discard the first cfg.WarmupWindows inject windows (engine ramp-up).
//  3. Compute mean DrainRatio over remaining steady-state inject windows.
//  4. Threshold against cfg.SaturatedDrainRatio (PERSISTENTLY_SATURATED) and
//     cfg.TransientDrainRatio (TRANSIENT_BACKLOG); else UNSATURATED.
//
// Robust to NaN/Inf DrainRatio values from windows with NumEntered==0 (skipped).
type drainRatioClassifier struct{}

// Classify implements BacklogClassifier for the drain-ratio policy.
func (drainRatioClassifier) Classify(
	windows []WindowMetrics,
	_ SlopeStats, // slope unused — drain-ratio decides from per-window NumLeft/NumEntered
	_, _, _ int, // initialBacklog, finalBacklog, peakInFlight unused
	_ float64, // meanInFlight unused
	cfg BacklogDriftConfig,
) (classification, note, recommendation string) {
	// Step 1: identify inject windows via the last_arrival_window predicate.
	// Find the highest index with NumEntered > 0; include windows up to and including it.
	// Interior windows with NumEntered==0 (closed-loop think-time gaps) remain in the
	// inject phase — only trailing all-zero windows are excluded as drain.
	lastInjectIdx := -1
	for i, w := range windows {
		if w.NumEntered > 0 {
			lastInjectIdx = i
		}
	}
	if lastInjectIdx < 0 {
		return "UNSATURATED",
			"No arrivals observed in any window.",
			"Check workload spec — synthetic workloads should always emit arrivals."
	}
	injectWindows := windows[:lastInjectIdx+1]

	// Step 2: discard warmup windows (engine ramp-up) at the start AND tail windows
	// at the end. The tail filter handles the boundary case where inject ends mid-window:
	// that window has reduced NumEntered (only partial inject) but full NumLeft (engine
	// continues draining backlog from prior windows), so DrainRatio exceeds 1 and biases
	// the mean upward. Symmetric warmup/tail trim isolates the true steady state.
	//
	// Note: AnalyzeBacklogDriftWithClassifier independently applies the same warmup/tail
	// trim to its slope-regression samples. The duplication is intentional: each classifier
	// owns the per-window granularity it needs (drain-ratio averages over windows; the
	// orchestrator regresses on ActiveEnd of windows). A future classifier that wants
	// different windowing can override locally without coordinating with the orchestrator.
	warmup := cfg.WarmupWindows
	if warmup < 0 {
		warmup = 0
	}
	tail := cfg.TailWindows
	if tail < 0 {
		tail = 0
	}
	if warmup+tail >= len(injectWindows) {
		return "UNSATURATED",
			fmt.Sprintf("Insufficient steady-state inject windows for analysis (have %d, need > %d).",
				len(injectWindows), warmup+tail),
			"Increase --num-requests, --rate, or decrease --saturation-warmup-windows / --saturation-tail-windows."
	}
	steady := injectWindows[warmup : len(injectWindows)-tail]

	// Step 3: mean DrainRatio over steady-state inject windows.
	// NaN/Inf entries can occur for interior NumEntered==0 windows (closed-loop
	// think-time gaps) that survive the inject-phase predicate; track and surface
	// the skip count per R1 (no silent discard).
	var sum float64
	var n, skipped int
	for _, w := range steady {
		if math.IsNaN(w.DrainRatio) || math.IsInf(w.DrainRatio, 0) {
			skipped++
			continue
		}
		sum += w.DrainRatio
		n++
	}
	if n == 0 {
		return "UNSATURATED",
			fmt.Sprintf("No usable steady-state inject windows (all %d DrainRatio NaN/Inf).", skipped),
			"Check workload spec — every inject window should have NumEntered > 0."
	}
	mean := sum / float64(n)

	// Step 4: classify by thresholds. ρ ≈ 1/DrainRatio is the queueing-theory utilization.
	// Defensive: mean is sum-of-non-negative DrainRatios over positive count; mean > 0
	// unless every steady window had NumLeft==0 (engine fully stalled).
	rhoEstimate := math.Inf(1)
	if mean > 0 {
		rhoEstimate = 1.0 / mean
	}
	note = fmt.Sprintf(
		"Steady-state mean DrainRatio = %.3f over %d windows (warmup=%d skipped, total inject windows=%d). "+
			"Estimated utilization ρ ≈ %.3f.",
		mean, n, warmup, len(injectWindows), rhoEstimate)
	if skipped > 0 {
		note += fmt.Sprintf(" (skipped %d windows with NaN/Inf DrainRatio)", skipped)
	}

	switch {
	case mean < cfg.SaturatedDrainRatio:
		return "PERSISTENTLY_SATURATED", note,
			fmt.Sprintf("System is overloaded (ρ ≈ %.2f). Add capacity or reduce load.", rhoEstimate)
	case mean < cfg.TransientDrainRatio:
		return "TRANSIENT_BACKLOG", note,
			"Borderline congestion. Monitor closely; consider provisioning headroom."
	default:
		return "UNSATURATED", note,
			"System handled load with healthy headroom."
	}
}

// classifyBacklogDriftSlopeBased determines saturation classification per BC-4/5/6/7.
// Returns (classification, note, recommendation). Internal helper used by slopeBasedClassifier.
func classifyBacklogDriftSlopeBased(slope, slopeLower, slopeUpper float64,
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

// AnalyzeBacklogDrift is a backward-compatible shim that uses the slope-based classifier.
//
// Deprecated: New code should call AnalyzeBacklogDriftWithClassifier with an explicit
// classifier from NewBacklogClassifier. This shim is preserved for callers that haven't
// yet been migrated to choose a classifier; the slope-based default does NOT match the
// new CLI default (drain-ratio after #1392). Calling this function will silently produce
// slope-based results regardless of the user's configured `--saturation-classifier`.
func AnalyzeBacklogDrift(requests []*sim.Request, simEndUs int64, cfg BacklogDriftConfig) BacklogDriftReport {
	return AnalyzeBacklogDriftWithClassifier(requests, simEndUs, cfg, slopeBasedClassifier{})
}

// AnalyzeBacklogDriftWithClassifier performs end-to-end backlog-drift saturation analysis (BC-8).
// Orchestrates: eligibility filter → window metrics → regression → classification.
// Returns UNSATURATED with note if observation has fewer than MinWindows complete windows (BC-7).
//
// The classifier parameter selects the classification policy. Pass slopeBasedClassifier{}
// for historical behavior, drainRatioClassifier{} for the new default, or any custom
// BacklogClassifier implementation. A nil classifier defaults to slopeBasedClassifier{}.
func AnalyzeBacklogDriftWithClassifier(requests []*sim.Request, simEndUs int64, cfg BacklogDriftConfig, classifier BacklogClassifier) BacklogDriftReport {
	if classifier == nil {
		classifier = slopeBasedClassifier{}
	}
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

	// Relativize time domain: shift intervals and simEndUs so windowing starts at 0.
	// This handles absolute Unix timestamps from observe-derived traces (#1405) while
	// being a no-op when the DES clock already starts at 0 (blis run).
	minArrival := intervals[0].ArrivalUs
	for _, iv := range intervals {
		if iv.ArrivalUs < minArrival {
			minArrival = iv.ArrivalUs
		}
	}
	if minArrival > 0 {
		for i := range intervals {
			intervals[i].ArrivalUs -= minArrival
			intervals[i].CompletionUs -= minArrival
		}
		simEndUs -= minArrival
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

	// Step 3: Prepare time series samples for regression — restrict to STEADY-STATE
	// inject windows (#1390). Saturation is observable only while load is being offered
	// AND after the engine has reached steady state; warmup ramp-up windows (queue
	// rising from 0 to steady-state) and the boundary tail window (where inject ends
	// mid-bucket) bias the slope estimate. Symmetric warmup/tail trim isolates the
	// regime where both slope-based and drain-ratio classifiers are well-defined.
	//
	// last_arrival_window predicate: include every window up to and including the
	// highest-indexed window with NumEntered > 0. Robust to closed-loop and bursty
	// workloads where interior inject windows can have NumEntered==0 (think-time gaps).
	lastInjectIdx := -1
	for i, w := range windows {
		if w.NumEntered > 0 {
			lastInjectIdx = i
		}
	}
	// Fall back to all windows if no arrivals at all (defensive — RequestsToIntervals
	// would have produced an empty intervals slice and we'd have early-returned by now,
	// but the inject-phase predicate must remain well-defined).
	injectWindows := windows
	if lastInjectIdx >= 0 {
		injectWindows = windows[:lastInjectIdx+1]
	}

	// Trim warmup/tail symmetrically. If insufficient steady windows remain, fall
	// back to all inject windows AND log a warning so the slope estimate's bias is
	// observable — drainRatioClassifier returns UNSATURATED in this regime via its
	// own short-data check, so symmetry of behavior across classifiers requires the
	// orchestrator to surface this case explicitly. (Pure orchestrator-level diagnostic;
	// classifier itself can still apply its own logic on the broader slice.)
	steadySamples := injectWindows
	warmup := cfg.WarmupWindows
	if warmup < 0 {
		warmup = 0
	}
	tail := cfg.TailWindows
	if tail < 0 {
		tail = 0
	}
	if warmup+tail < len(injectWindows) {
		steadySamples = injectWindows[warmup : len(injectWindows)-tail]
	} else {
		logrus.Warnf("Saturation analysis: warmup(%d) + tail(%d) >= inject windows(%d); slope regression uses unfiltered inject phase, may be biased by ramp-up/boundary effects",
			warmup, tail, len(injectWindows))
	}

	samples := make([]struct {
		timeUs int64
		count  int
	}, len(steadySamples))
	for i, w := range steadySamples {
		// Use window midpoint as time coordinate
		samples[i].timeUs = (w.StartUs + w.EndUs) / 2
		samples[i].count = w.ActiveEnd
	}

	// Step 4: Fit linear regression on steady-state inject windows (BC-3, #1390)
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

	// Step 6: Classify saturation state via the supplied classifier (BC-3 / #1391)
	classification, note, recommendation := classifier.Classify(
		windows,
		SlopeStats{Slope: slope, SlopeLower: slopeLower, SlopeUpper: slopeUpper},
		initialBacklog, finalBacklog, peakInFlight, meanInFlight, cfg,
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
