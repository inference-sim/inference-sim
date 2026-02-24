package sim

// LatencyModel estimates execution times for the DES step loop.
// Two implementations exist in sim/latency/: BlackboxLatencyModel (alpha/beta regression)
// and RooflineLatencyModel (analytical FLOPs/bandwidth).
// All time estimates are in microseconds (ticks).
type LatencyModel interface {
	// StepTime estimates the duration of one batch step given the running batch.
	// Precondition: each request in batch has NumNewTokens set by BatchFormation.FormBatch().
	StepTime(batch []*Request) int64

	// QueueingTime estimates the arrival-to-queue delay for a request.
	QueueingTime(req *Request) int64

	// OutputTokenProcessingTime estimates per-token post-processing time.
	OutputTokenProcessingTime() int64

	// SchedulingProcessingTime estimates scheduling overhead per request.
	SchedulingProcessingTime() int64

	// PreemptionProcessingTime estimates preemption overhead per eviction.
	PreemptionProcessingTime() int64
}

// NewLatencyModelFunc is a factory function for creating LatencyModel implementations.
// Set by sim/latency package's init() via registration. This breaks the import cycle
// between sim/ (which defines LatencyModel) and sim/latency/ (which implements it).
//
// Production callers should import sim/latency and use latency.NewLatencyModel() directly.
// Test code in package sim uses MustNewLatencyModel to avoid importing sim/latency.
var NewLatencyModelFunc func(coeffs LatencyCoeffs, hw ModelHardwareConfig) (LatencyModel, error)

// MustNewLatencyModel calls NewLatencyModelFunc with a nil guard. Panics with an
// actionable message if the factory has not been registered (missing sim/latency import).
func MustNewLatencyModel(coeffs LatencyCoeffs, hw ModelHardwareConfig) (LatencyModel, error) {
	if NewLatencyModelFunc == nil {
		panic("NewLatencyModelFunc not registered: import sim/latency to register it " +
			"(add: import _ \"github.com/inference-sim/inference-sim/sim/latency\")")
	}
	return NewLatencyModelFunc(coeffs, hw)
}
