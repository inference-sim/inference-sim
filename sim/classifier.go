// sim/classifier.go
package sim

// BatchClassifier performs post-hoc classification on completed request metrics.
// This interface is defined in sim/ (not sim/saturation/) to avoid import cycles
// while maintaining compile-time type safety for the SaveResults integration.
type BatchClassifier interface {
	// Classify analyzes completed requests and returns a classification result.
	// totalArrivals includes all injected requests (completed + timed out + dropped).
	// The result is serialized as interface{} to avoid exposing saturation-specific
	// types in sim/ (would create reverse dependency). Callers should treat this
	// as opaque JSON payload.
	Classify(requests []RequestMetrics, totalArrivals int) interface{}
}
