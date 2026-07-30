// sim/live_saturation.go
package sim

// LiveSaturationObserver is the sim-side seam for live saturation detection. It is
// defined in sim/ (not sim/saturation/) with primitive-only method signatures so the
// event loop can drive a detector during the run without importing sim/saturation
// (which imports sim — the reverse dependency would cycle). This mirrors
// BatchClassifier, the same pattern for the end-of-run path.
//
// The concrete implementation is *saturation.LiveTimeline, wired in from cmd/.
//
// Contract: implementations MUST be read-only with respect to simulation state —
// they observe events and accumulate their own state, but never enqueue simulation
// events or mutate requests. A synchronous, side-effect-free observer cannot change
// event ordering, so it cannot change deterministic stdout (INV-6), exactly like
// ProgressHook.
type LiveSaturationObserver interface {
	// ObserveArrival records a request arrival at clockUs (µs).
	ObserveArrival(id string, clockUs int64)
	// ObserveCompletion records a request completion at clockUs (µs) with its
	// end-to-end latency in milliseconds.
	ObserveCompletion(id string, clockUs int64, e2eMs float64)
	// EmitPoint records a cumulative saturation label stamped at clockUs. Called by
	// the driving loop at each interval boundary and once at the final clock.
	EmitPoint(clockUs int64)
}
