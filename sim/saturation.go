package sim

import (
	"fmt"
	"math"
)

// SaturationDetector evaluates cluster-level saturation for flow control dispatch gating.
// Returns a continuous float [0.0, 1.0+]. Flow control holds requests when saturation >= 1.0.
type SaturationDetector interface {
	Saturation(state *RouterState) float64
}

// NeverSaturated always returns 0.0 — turns the gateway queue into a zero-latency pass-through.
// This is the AlwaysAdmit equivalent for flow control.
type NeverSaturated struct{}

func (n *NeverSaturated) Saturation(_ *RouterState) float64 { return 0.0 }

// UtilizationDetector computes saturation as avg(max(QueueDepth/queueThreshold, KVUtil/kvThreshold))
// across all instances. Mirrors GIE's utilization detector.
type UtilizationDetector struct {
	queueDepthThreshold  float64
	kvCacheUtilThreshold float64
}

// NewUtilizationDetector creates a UtilizationDetector with the given thresholds.
// Panics if either threshold is <= 0 (R3).
func NewUtilizationDetector(queueDepthThreshold, kvCacheUtilThreshold float64) *UtilizationDetector {
	if queueDepthThreshold <= 0 || math.IsNaN(queueDepthThreshold) || math.IsInf(queueDepthThreshold, 0) {
		panic(fmt.Sprintf("UtilizationDetector: queueDepthThreshold must be a finite value > 0, got %f", queueDepthThreshold))
	}
	if kvCacheUtilThreshold <= 0 || math.IsNaN(kvCacheUtilThreshold) || math.IsInf(kvCacheUtilThreshold, 0) {
		panic(fmt.Sprintf("UtilizationDetector: kvCacheUtilThreshold must be a finite value > 0, got %f", kvCacheUtilThreshold))
	}
	return &UtilizationDetector{
		queueDepthThreshold:  queueDepthThreshold,
		kvCacheUtilThreshold: kvCacheUtilThreshold,
	}
}

func (u *UtilizationDetector) Saturation(state *RouterState) float64 {
	if len(state.Snapshots) == 0 {
		return 1.0 // no instances → assume saturated (EC-2)
	}
	sum := 0.0
	for _, snap := range state.Snapshots {
		qScore := float64(snap.QueueDepth) / u.queueDepthThreshold
		kvScore := snap.KVUtilization / u.kvCacheUtilThreshold
		sum += math.Max(qScore, kvScore)
	}
	return sum / float64(len(state.Snapshots))
}

// ConcurrencyDetector computes saturation as totalInFlight / (numInstances × maxConcurrency).
// Mirrors GIE's concurrency detector.
type ConcurrencyDetector struct {
	maxConcurrency int
}

// NewConcurrencyDetector creates a ConcurrencyDetector with the given max concurrency per instance.
// Panics if maxConcurrency <= 0 (R3).
func NewConcurrencyDetector(maxConcurrency int) *ConcurrencyDetector {
	if maxConcurrency <= 0 {
		panic(fmt.Sprintf("ConcurrencyDetector: maxConcurrency must be > 0, got %d", maxConcurrency))
	}
	return &ConcurrencyDetector{maxConcurrency: maxConcurrency}
}

func (c *ConcurrencyDetector) Saturation(state *RouterState) float64 {
	if len(state.Snapshots) == 0 {
		return 1.0 // no instances → assume saturated (EC-2)
	}
	totalInFlight := 0
	for _, snap := range state.Snapshots {
		totalInFlight += snap.InFlightRequests
	}
	return float64(totalInFlight) / float64(len(state.Snapshots)*c.maxConcurrency)
}

// NewSaturationDetector creates a SaturationDetector by name.
// Valid names: "", "never", "utilization", "concurrency".
// Panics on unknown name.
func NewSaturationDetector(name string, queueDepthThreshold, kvCacheUtilThreshold float64, maxConcurrency int) SaturationDetector {
	switch name {
	case "", "never":
		return &NeverSaturated{}
	case "utilization":
		return NewUtilizationDetector(queueDepthThreshold, kvCacheUtilThreshold)
	case "concurrency":
		return NewConcurrencyDetector(maxConcurrency)
	default:
		panic(fmt.Sprintf("unknown saturation detector %q", name))
	}
}
