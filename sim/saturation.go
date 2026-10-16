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
		return 1.0 // no routable instances — hold all requests until an instance becomes available
	}
	sum := 0.0
	for _, snap := range state.Snapshots {
		qScore := float64(snap.QueueDepth) / u.queueDepthThreshold
		kvScore := snap.KVUtilization / u.kvCacheUtilThreshold
		sum += math.Max(qScore, kvScore)
	}
	return sum / float64(len(state.Snapshots))
}

// ConcurrencyMode selects the concurrency detector's accounting unit.
// Reference: llm-d-router main@981acb8a and v0.10.0
// pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency/config.go:82-92.
type ConcurrencyMode string

const (
	ConcurrencyModeRequests ConcurrencyMode = "requests"
	ConcurrencyModeTokens   ConcurrencyMode = "tokens"
	ConcurrencyModeHybrid   ConcurrencyMode = "hybrid"
)

// ConcurrencyDetectorConfig is the llm-d concurrency-detector parameter set.
// Defaults (config.go:94-103): maxConcurrency 100, headroom 0, mode requests,
// maxTokenConcurrency 1000000.
type ConcurrencyDetectorConfig struct {
	Mode                ConcurrencyMode
	MaxConcurrency      int64
	MaxTokenConcurrency int64
	Headroom            float64
	// InFlightLoadProducerName names the inflight-load-producer whose token count
	// the detector reads (empty = the producer's default name, "inflight-load-producer").
	InFlightLoadProducerName string

	// The following follow llm-d main (981acb8a) and are all false on v0.10.0:
	// FailOpen returns every candidate when the filter drops them all
	// (main:…/concurrency/config.go:81-89, detector.go:275-281; default true);
	// CountIncomingTokens adds the request's own uncached tokens to the token
	// check in tokens/hybrid mode, an idle endpoint always passing
	// (detector.go:259-297); DispatchReservations counts requests dispatched by
	// flow control but not yet routed in the request saturation
	// (detector.go:139-160, :210-228).
	FailOpen             bool
	CountIncomingTokens  bool
	DispatchReservations bool
}

// DefaultConcurrencyDetectorConfig returns llm-d's defaults (config.go:94-103).
func DefaultConcurrencyDetectorConfig() ConcurrencyDetectorConfig {
	return ConcurrencyDetectorConfig{
		Mode:                ConcurrencyModeRequests,
		MaxConcurrency:      100,
		MaxTokenConcurrency: 1000000,
	}
}

// Validate mirrors validateConfig (config.go:156-179).
func (c ConcurrencyDetectorConfig) Validate() error {
	if c.MaxConcurrency <= 0 {
		return fmt.Errorf("maxConcurrency must be strictly positive, got %d", c.MaxConcurrency)
	}
	if c.Headroom < 0 || math.IsNaN(c.Headroom) || math.IsInf(c.Headroom, 0) {
		return fmt.Errorf("headroom must be a finite non-negative value, got %f", c.Headroom)
	}
	if c.MaxTokenConcurrency <= 0 {
		return fmt.Errorf("maxTokenConcurrency must be strictly positive, got %d", c.MaxTokenConcurrency)
	}
	switch c.Mode {
	case ConcurrencyModeRequests, ConcurrencyModeTokens, ConcurrencyModeHybrid:
	default:
		return fmt.Errorf("unsupported concurrencyMode: %q", c.Mode)
	}
	return nil
}

// ConcurrencyDetector mirrors llm-d's concurrency-detector: a pool saturation
// signal for flow control plus a per-endpoint scheduling filter.
// Reference: llm-d-router main@981acb8a and v0.10.0
// pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency/detector.go.
//
// In-flight requests are RoutingSnapshot.InFlightRequests (dispatch to completion,
// the inflight-load-producer's request counter, producer.go:430 and :545); in-flight
// tokens are RoutingSnapshot.InFlightTokens.
type ConcurrencyDetector struct {
	cfg          ConcurrencyDetectorConfig
	reservations map[string]bool
}

// NewConcurrencyDetector creates a requests-mode ConcurrencyDetector with zero headroom.
// Panics if maxConcurrency <= 0 (R3).
func NewConcurrencyDetector(maxConcurrency int) *ConcurrencyDetector {
	if maxConcurrency <= 0 {
		panic(fmt.Sprintf("ConcurrencyDetector: maxConcurrency must be > 0, got %d", maxConcurrency))
	}
	cfg := DefaultConcurrencyDetectorConfig()
	cfg.MaxConcurrency = int64(maxConcurrency)
	return &ConcurrencyDetector{cfg: cfg, reservations: make(map[string]bool)}
}

// NewConcurrencyDetectorFromConfig creates a detector from a full llm-d parameter set.
// Panics on an invalid config (R3); callers validate first.
func NewConcurrencyDetectorFromConfig(cfg ConcurrencyDetectorConfig) *ConcurrencyDetector {
	if err := cfg.Validate(); err != nil {
		panic(fmt.Sprintf("ConcurrencyDetector: %v", err))
	}
	return &ConcurrencyDetector{cfg: cfg, reservations: make(map[string]bool)}
}

// ReserveDispatch counts a request dispatched by flow control until it is routed
// (main:…/concurrency/detector.go:139-150). No-op unless DispatchReservations.
func (c *ConcurrencyDetector) ReserveDispatch(reqID string) {
	if c.cfg.DispatchReservations && reqID != "" {
		c.reservations[reqID] = true
	}
}

// ReleaseDispatch ends a reservation; unknown IDs are no-ops (detector.go:152-162).
func (c *ConcurrencyDetector) ReleaseDispatch(reqID string) {
	delete(c.reservations, reqID)
}

// DispatchReservationTracker is implemented by saturation detectors that count
// dispatched-but-unrouted requests.
type DispatchReservationTracker interface {
	ReserveDispatch(reqID string)
	ReleaseDispatch(reqID string)
}

// Config returns the detector's parameters.
func (c *ConcurrencyDetector) Config() ConcurrencyDetectorConfig { return c.cfg }

// Saturation mirrors detector.go:132-173: aggregate inflight/capacity in requests
// and tokens modes; the per-endpoint max(requestRatio, tokenRatio) averaged across
// endpoints in hybrid mode. No endpoints ⇒ 1.0.
func (c *ConcurrencyDetector) Saturation(state *RouterState) float64 {
	if len(state.Snapshots) == 0 {
		return 1.0 // no routable instances — hold all requests until an instance becomes available
	}
	var reqInflight, reqCapacity, tokInflight, tokCapacity int64
	hybridSum := 0.0
	for _, snap := range state.Snapshots {
		reqCapacity += c.cfg.MaxConcurrency
		tokCapacity += c.cfg.MaxTokenConcurrency
		reqInflight += int64(snap.InFlightRequests)
		tokInflight += snap.InFlightTokens
		hybridSum += math.Max(
			concurrencyRatio(int64(snap.InFlightRequests), c.cfg.MaxConcurrency),
			concurrencyRatio(snap.InFlightTokens, c.cfg.MaxTokenConcurrency),
		)
	}
	pending := int64(len(c.reservations))
	switch c.cfg.Mode {
	case ConcurrencyModeTokens:
		return concurrencyRatio(tokInflight, tokCapacity)
	case ConcurrencyModeHybrid:
		if pending > 0 {
			return math.Max(hybridSum/float64(len(state.Snapshots)), concurrencyRatio(reqInflight+pending, reqCapacity))
		}
		return hybridSum / float64(len(state.Snapshots))
	default:
		return concurrencyRatio(reqInflight+pending, reqCapacity)
	}
}

// concurrencyRatio fails closed (1.0) on zero capacity (detector.go:175-181).
func concurrencyRatio(inflight, capacity int64) float64 {
	if capacity == 0 {
		return 1.0
	}
	return float64(inflight) / float64(capacity)
}

// Filter drops endpoints at or above the relaxed limit capacity×(1+headroom),
// truncated to int64 (main:…/concurrency/detector.go:257-258). An all-dropped
// result is returned empty unless FailOpen, and an empty set fails the request
// (pkg/epp/scheduling/scheduler_profile.go:134-142).
func (c *ConcurrencyDetector) Filter(snapshots []RoutingSnapshot) []RoutingSnapshot {
	return c.filterWithIncoming(snapshots, nil)
}

// filterWithIncoming is Filter with the request's own uncached tokens per
// endpoint (used in tokens/hybrid mode when CountIncomingTokens).
func (c *ConcurrencyDetector) filterWithIncoming(snapshots []RoutingSnapshot, incoming func(id string) int64) []RoutingSnapshot {
	reqLimit := int64(float64(c.cfg.MaxConcurrency) * (1.0 + c.cfg.Headroom))
	tokLimit := int64(float64(c.cfg.MaxTokenConcurrency) * (1.0 + c.cfg.Headroom))
	countIncoming := c.cfg.CountIncomingTokens && incoming != nil &&
		(c.cfg.Mode == ConcurrencyModeTokens || c.cfg.Mode == ConcurrencyModeHybrid)
	filtered := make([]RoutingSnapshot, 0, len(snapshots))
	for _, snap := range snapshots {
		var in int64
		if countIncoming {
			if v := incoming(snap.ID); v > 0 {
				in = v
			}
		}
		if c.admits(int64(snap.InFlightRequests), snap.InFlightTokens, in, reqLimit, tokLimit) {
			filtered = append(filtered, snap)
		}
	}
	if len(filtered) == 0 && c.cfg.FailOpen {
		return append(filtered, snapshots...)
	}
	return filtered
}

// admits mirrors main:…/concurrency/detector.go:288-298; with incoming = 0 it is
// v0.10.0's check (v0.10.0:…/detector.go:211-221).
func (c *ConcurrencyDetector) admits(requests, tokens, incoming, reqLimit, tokLimit int64) bool {
	tokensFit := tokens < tokLimit && (tokens == 0 || tokens+incoming <= tokLimit)
	switch c.cfg.Mode {
	case ConcurrencyModeTokens:
		return tokensFit
	case ConcurrencyModeHybrid:
		return requests < reqLimit && tokensFit
	default:
		return requests < reqLimit
	}
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
