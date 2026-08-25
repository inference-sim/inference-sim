// sim/saturation/peak_rate.go
package saturation

import "math"

// PeakRateDetector classifies saturation from the growth of the backlog's
// high-water mark, needing no latency target and no estimate of server capacity.
//
// # The statistic
//
// Backlog in a queue is a random walk reflected at zero (Lindley:
// W <- max(0, W + service - interarrival)). The reflection is what makes the
// regimes distinguishable, and it licenses a statistic that estimates nothing
// about the server:
//
//	R_t = Peak_t / t     (running max of in-flight requests over elapsed time)
//
//	  positive drift (rho > 1)  =>  R_t -> a positive CONSTANT
//	  zero drift     (rho = 1)  =>  R_t -> 0  as 1/sqrt(t)   (Peak grows as sqrt(t))
//	  negative drift (rho < 1)  =>  R_t -> 0  as 1/t         (Peak is a finite r.v.)
//
// So an overloaded server holds R_t near a constant while a healthy one lets it
// decay, and the detector fires when R_t HOLDS above a threshold.
//
// # Why this is worth having alongside the other detectors
//
// It needs no absolute latency target. `threshold` compares mean E2E against a
// millisecond figure that must be re-tuned for every model and GPU; this reads a
// shape instead.
//
// It is also the natural foil to `backlog-drift`, and the comparison explains why
// that detector is weakest exactly where it matters most. backlog-drift fits a
// STRAIGHT LINE to backlog growth, but at rho ~= 1 backlog grows like sqrt(t), so a
// linear fit tends toward ZERO slope and reports STABLE at criticality -- the worst
// possible failure direction. R_t has no such degeneracy at the boundary.
//
// # Horizon dependence (load-bearing, not hygiene)
//
// The result above is asymptotic while every run is finite, so how well R_t
// separates healthy from overloaded depends on how long you watch. Measured on a
// frozen apparatus (Llama-3.1-8B-Instruct / H100 / TP=1), separation between
// sub- and super-capacity traffic:
//
//	n=500   2.3x      n=2000  4.7x      n=8000  14.6x
//
// The mechanism is exactly the one above: at 0.5x nominal load the peak goes
// 66 -> 71 over the run (it has levelled off, so R_t decays 13x), while at 1.5x it
// goes 329 -> 2743 (still climbing, so R_t barely moves). A short run cannot tell
// these apart because the healthy server's peak has not levelled off yet.
//
// MinObservations exists for this reason and is part of the algorithm, not input
// validation: below it the detector reports STABLE rather than guessing from a
// transient. An early R_t is large by construction (small t), so reading it too
// soon reports saturation on every run's opening moments.
//
// # State
//
// Four scalars: the running peak, in-flight count, first/last timestamps, and the
// consecutive-breach counter. No per-event or per-request retention, so memory is
// O(1) in the trace length -- unlike composite and threshold, which retain every
// event.
type PeakRateDetector struct {
	cfg peakRateConfig

	// Streaming state. Populated by Observe, read by Detect, cleared by Reset.
	// This is a causal computation: it consumes events in order and never looks
	// ahead.
	peak         int64 // running maximum of inFlight
	inFlight     int64 // arrivals - completions seen so far
	observations int64 // events folded in
	firstTsUs    int64
	lastTsUs     int64
	haveFirst    bool

	// consecutive counts successive breaches of the threshold. It is advanced in
	// Observe rather than Detect so that Detect stays a pure query: a verdict must
	// depend on the event stream, never on how many times a caller asked for it.
	consecutive int
}

// peakRateConfig holds the detector's resolved, validated parameters.
type peakRateConfig struct {
	// Threshold is the false-alarm calibration knob: fire when R_t exceeds it.
	// A LARGER value fires less. Its units are backlog per second, so it is
	// calibrated per deployment -- see the package docs on calibration.
	Threshold float64

	// MinObservations gates the verdict until enough of the run has been seen for
	// R_t to be meaningful (see the horizon discussion on PeakRateDetector).
	MinObservations int

	// ConsecutiveK is the number of successive breaches required before firing --
	// the anti-flapping lever that keeps a momentary excursion from flipping the
	// verdict.
	ConsecutiveK int

	// OverloadMultiple separates the two saturated levels: R_t above
	// OverloadMultiple*Threshold is OVERLOADED, between the two is BACKLOGGED.
	// Must be >= 1, or the BACKLOGGED band would be unsatisfiable.
	OverloadMultiple float64
}

// Default parameters, as validated by the optimization campaign that selected this
// statistic (5 seeds x 11 load rungs, false-alarm-calibrated first). The threshold
// is the calibrated operating point on that apparatus; MinObservations, ConsecutiveK
// and OverloadMultiple are the campaign's frozen factor levels.
const (
	defaultPeakRateThreshold        = 0.5
	defaultPeakRateMinObservations  = 20
	defaultPeakRateConsecutiveK     = 3
	defaultPeakRateOverloadMultiple = 3.0
)

// NewPeakRateDetector creates a peak-rate detector with the campaign-validated
// defaults.
func NewPeakRateDetector() Detector {
	return newPeakRateDetector(peakRateConfig{
		Threshold:        defaultPeakRateThreshold,
		MinObservations:  defaultPeakRateMinObservations,
		ConsecutiveK:     defaultPeakRateConsecutiveK,
		OverloadMultiple: defaultPeakRateOverloadMultiple,
	})
}

// newPeakRateDetector is the canonical constructor (R4): every construction path
// routes through it. Values are validated by resolvePeakRateConfig before arriving
// here; the clamps below are the in-process safety net for direct callers, so the
// banding can never be driven by a nonsense parameter.
func newPeakRateDetector(cfg peakRateConfig) Detector {
	// The floor matches the YAML resolver's (minCalibrationKnob), so the two layers
	// agree: a subnormal threshold underflows the score denominator and would
	// decouple Level from Score.
	if math.IsNaN(cfg.Threshold) || math.IsInf(cfg.Threshold, 0) || cfg.Threshold < minCalibrationKnob {
		cfg.Threshold = defaultPeakRateThreshold
	}
	if cfg.MinObservations <= 0 {
		cfg.MinObservations = defaultPeakRateMinObservations
	}
	if cfg.ConsecutiveK <= 0 {
		cfg.ConsecutiveK = defaultPeakRateConsecutiveK
	}
	if cfg.OverloadMultiple < 1 || math.IsNaN(cfg.OverloadMultiple) || math.IsInf(cfg.OverloadMultiple, 0) {
		cfg.OverloadMultiple = defaultPeakRateOverloadMultiple
	}
	return &PeakRateDetector{cfg: cfg}
}

func (p *PeakRateDetector) Name() string { return "peak-rate" }

// Observe folds one event into the running state and advances the breach counter.
//
// Both event types matter: arrivals raise the backlog (and so possibly the peak),
// completions lower it. An unknown event type is ignored rather than mis-counted.
func (p *PeakRateDetector) Observe(event Event) {
	// Ignore an unrecognized event type BEFORE touching any state. Advancing the
	// elapsed span for an event that does not change the backlog would shrink the
	// statistic while leaving the breach streak frozen, so the verdict would
	// contradict its own reported statistic (R1: no silent inconsistency).
	if event.Type != Arrival && event.Type != Completion {
		return
	}

	// Track the observed span SYMMETRICALLY: the minimum and maximum timestamp
	// seen, not "the first one" and the maximum. buildSortedEvents delivers events
	// in nondecreasing time, but Observe is a public interface method, and an
	// out-of-order first event would otherwise pin elapsed at zero forever --
	// making ready() permanently false and reporting STABLE on an unboundedly
	// growing backlog. That is a silent failure, not a degradation.
	if !p.haveFirst {
		p.firstTsUs, p.lastTsUs = event.Timestamp, event.Timestamp
		p.haveFirst = true
	}
	if event.Timestamp < p.firstTsUs {
		p.firstTsUs = event.Timestamp
	}
	if event.Timestamp > p.lastTsUs {
		p.lastTsUs = event.Timestamp
	}

	switch event.Type {
	case Arrival:
		p.inFlight++
	case Completion:
		// Guarded: a completion without its arrival would otherwise drive the
		// backlog negative. buildSortedEvents pairs them, but a direct caller
		// need not.
		if p.inFlight > 0 {
			p.inFlight--
		}
	}
	p.observations++

	if p.inFlight > p.peak {
		p.peak = p.inFlight
	}

	// Advance the breach streak here, not in Detect, so the verdict is a function
	// of the event stream alone.
	if p.ready() && p.statistic() > p.cfg.Threshold {
		p.consecutive++
	} else {
		p.consecutive = 0
	}
}

// Detect reports the current verdict. It is a pure query: calling it repeatedly
// without an intervening Observe returns the same Result.
func (p *PeakRateDetector) Detect() Result {
	stat := p.statistic()
	signals := map[string]float64{
		"peak_rate":         stat,
		"threshold":         p.cfg.Threshold,
		"peak_backlog":      float64(p.peak),
		"in_flight":         float64(p.inFlight),
		"elapsed_sec":       p.elapsedSec(),
		"observations":      float64(p.observations),
		"consecutive":       float64(p.consecutive),
		"overload_multiple": p.cfg.OverloadMultiple,
	}

	// Not enough of the run seen yet: report STABLE rather than guessing from the
	// opening transient, where R_t is large by construction (R20 -- degenerate
	// input is STABLE, never a panic).
	if !p.ready() {
		return Result{Level: Stable, Score: 0, Confidence: 0, Signals: signals}
	}

	// The OVERLOADED boundary, hoisted once so the band switch below and the score
	// denominator provably use the SAME value: if they diverged, Score reaching its
	// 1.0 cap would stop coinciding with the OVERLOADED band.
	overloadAt := p.cfg.OverloadMultiple * p.cfg.Threshold

	// Score is the normalized magnitude, capped at 1.0. The cap is reached exactly
	// when the statistic clears overloadAt -- the same `>` comparison the band uses,
	// so Score == 1.0 and Level == OVERLOADED cannot disagree at the boundary point
	// itself (Level additionally requires the breach streak, so a capped Score with
	// a STABLE level means "magnitude reached, debounce not yet satisfied").
	score := 0.0
	switch {
	case stat > overloadAt:
		score = 1.0
	case overloadAt > 0:
		score = math.Max(0.0, stat) / overloadAt
	}

	level := Stable
	if p.consecutive >= p.cfg.ConsecutiveK {
		level = Backlogged
		if stat > overloadAt {
			level = Overloaded
		}
	}

	// Confidence reuses composite's ramp so the streaming detectors agree.
	confidence := math.Min(1.0, float64(p.observations)/20.0)

	return Result{Level: level, Score: score, Confidence: confidence, Signals: signals}
}

// Reset returns the detector to its initial state so it can be reused across
// replay legs. The configuration survives; only accumulated state is cleared.
func (p *PeakRateDetector) Reset() {
	p.peak = 0
	p.inFlight = 0
	p.observations = 0
	p.firstTsUs = 0
	p.lastTsUs = 0
	p.haveFirst = false
	// Belt-and-braces: Observe already zeroes the streak on its first call after a
	// reset (the peak is 0, so ready() is false and the else-branch fires), which
	// makes this line unobservable from outside and therefore untestable
	// behaviorally. It is kept so "Reset clears all accumulated state" holds as a
	// property of Reset itself rather than depending on Observe's internals.
	p.consecutive = 0
}

// ready reports whether enough of the run has been observed for R_t to carry
// information.
//
// The observation count is the load-bearing condition: it keeps the detector quiet
// through the opening transient. The elapsed-span condition is defence in depth --
// statistic() independently returns 0 for a non-positive span, and a zero statistic
// cannot exceed a threshold (validation floors it at minCalibrationKnob > 0), so
// removing it here changes no verdict. It is kept so that "is a verdict meaningful
// yet?" is answered in one place rather than depending on statistic()'s guard, and
// so a future threshold of exactly 0 could not turn an undefined ratio into a
// breach (R11).
func (p *PeakRateDetector) ready() bool {
	return p.observations >= int64(p.cfg.MinObservations) && p.elapsedSec() > 0
}

// statistic is R_t = Peak_t / t, in units of backlog per second. Returns 0 when
// no time has elapsed, so a burst sharing one timestamp cannot divide by zero.
func (p *PeakRateDetector) statistic() float64 {
	elapsed := p.elapsedSec()
	if elapsed <= 0 {
		return 0
	}
	return float64(p.peak) / elapsed
}

// elapsedSec is the observed span in seconds.
func (p *PeakRateDetector) elapsedSec() float64 {
	if !p.haveFirst || p.lastTsUs <= p.firstTsUs {
		return 0
	}
	return float64(p.lastTsUs-p.firstTsUs) / 1e6
}
