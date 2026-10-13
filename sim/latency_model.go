package sim

// LatencyModel estimates execution times for the DES step loop.
// The production implementation is the blis-latency-kernel adapter in sim/kernelmodel; the
// caller builds it and supplies it through SimConfig.LatencyModel.
// All time estimates are in microseconds (ticks).
type LatencyModel interface {
	// StepTime estimates the duration of one batch step given the running batch.
	// Precondition: each request in batch has NumNewTokens set by BatchFormation.FormBatch().
	// Postcondition: return value >= 1 for all inputs (including empty batch).
	// A return value of 0 would stall the simulation clock, violating INV-3 (clock monotonicity).
	StepTime(batch []*Request) int64

	// QueueingTime estimates the arrival-to-queue delay for a request.
	QueueingTime(req *Request) int64

	// OutputTokenProcessingTime estimates per-token post-processing time.
	OutputTokenProcessingTime() int64

	// PostDecodeFixedOverhead estimates the fixed per-request post-decode overhead (µs).
	// This is the constant overhead at request completion (e.g., response setup, final API
	// processing) that is NOT per-token; a model that charges none returns 0. Used by
	// recordRequestCompletion to add to E2E without affecting TTFT.
	PostDecodeFixedOverhead() int64
}
