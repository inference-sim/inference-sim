// sim/saturation/live.go
package saturation

// LiveDetector is a saturation detector driven incrementally during a simulation
// run. Unlike the batch Detector.Classify path (one label from the full completed
// set at the end), a LiveDetector observes arrival/completion events AS THEY OCCUR
// and can emit a cumulative saturation label at any interval boundary.
//
// This is the pluggable seam for live detection: composite and backlog-drift are
// the first two implementations, and future live detectors satisfy this interface
// to drop into the same event-loop wiring (LiveTimeline + sim.LiveSaturationObserver).
//
// Implementations differ in HOW they honor the contract:
//   - composite streams genuinely: Observe accumulates, LabelAt reads the running state.
//   - backlog-drift buffers events and re-runs its windowed regression at each LabelAt
//     (it is a retrospective analyzer with no meaningful single-event streaming state).
//
// Both present the same interface, so the driver and callers treat them uniformly.
type LiveDetector interface {
	// Name returns the detector's identifier (matches the --post-hoc-detector value).
	Name() string

	// Observe records one arrival or completion event. Events arrive in
	// non-decreasing Timestamp order (simulation-clock order).
	Observe(event Event)

	// LabelAt returns the cumulative saturation verdict for all events observed so
	// far, stamped at clockUs. arrivals and completions are the running counts the
	// driver maintains (passed in so every detector reports consistent counts without
	// each re-deriving them). The returned point's Label is computed via LabelFromResult.
	LabelAt(clockUs int64, arrivals, completions int, cfg TimelineConfig) TimelinePoint

	// Reset clears accumulated state for reuse.
	Reset()
}

// LiveTimeline drives a LiveDetector over the course of a run: it feeds observed
// events into the detector, maintains the cumulative arrival/completion counts, and
// collects one TimelinePoint per interval boundary. It is the concrete type wired
// into the simulator via the sim.LiveSaturationObserver seam.
//
// Single-goroutine use only (the DES event loop is single-threaded); no locking.
type LiveTimeline struct {
	det         LiveDetector
	cfg         TimelineConfig
	points      []TimelinePoint
	arrivals    int
	completions int
}

// NewLiveTimeline wraps a LiveDetector with the given config.
func NewLiveTimeline(det LiveDetector, cfg TimelineConfig) *LiveTimeline {
	return &LiveTimeline{det: det, cfg: cfg}
}

// ObserveArrival records a request arrival at clockUs (µs). The id lets a detector
// that reconstructs per-request intervals (backlog-drift) pair arrivals with their
// completions; streaming detectors (composite, threshold) ignore it.
func (t *LiveTimeline) ObserveArrival(id string, clockUs int64) {
	t.arrivals++
	t.det.Observe(Event{Type: Arrival, RequestID: id, Timestamp: clockUs})
}

// ObserveCompletion records a request completion at clockUs (µs) with its end-to-end
// latency in milliseconds (the detectors' latency signal is in ms).
func (t *LiveTimeline) ObserveCompletion(id string, clockUs int64, e2eMs float64) {
	t.completions++
	t.det.Observe(Event{Type: Completion, RequestID: id, Timestamp: clockUs, LatencyMs: e2eMs})
}

// EmitPoint appends the detector's cumulative verdict at clockUs to the timeline.
// Called by the driver at each interval boundary and once at the final clock.
func (t *LiveTimeline) EmitPoint(clockUs int64) {
	t.points = append(t.points, t.det.LabelAt(clockUs, t.arrivals, t.completions, t.cfg))
}

// Points returns the collected timeline. Safe to call after the run completes.
func (t *LiveTimeline) Points() []TimelinePoint {
	return t.points
}
