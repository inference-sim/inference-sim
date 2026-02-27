package sim

import (
	"fmt"
	"math"
)

// AdmissionPolicy decides whether a request is admitted for processing.
// Used by ClusterSimulator's online routing pipeline to gate incoming requests.
// Receives *RouterState with cluster-wide snapshots and clock.
type AdmissionPolicy interface {
	Admit(req *Request, state *RouterState) (admitted bool, reason string)
}

// AlwaysAdmit admits all requests unconditionally.
type AlwaysAdmit struct{}

func (a *AlwaysAdmit) Admit(_ *Request, _ *RouterState) (bool, string) {
	return true, ""
}

// TokenBucket implements rate-limiting admission control.
type TokenBucket struct {
	capacity      float64
	refillRate    float64 // tokens per second
	currentTokens float64
	lastRefill    int64 // last refill clock time in microseconds
}

// NewTokenBucket creates a TokenBucket with the given capacity and refill rate.
// Panics if capacity or refillRate is <= 0, NaN, or Inf (R3: validate at construction).
func NewTokenBucket(capacity, refillRate float64) *TokenBucket {
	if capacity <= 0 || math.IsNaN(capacity) || math.IsInf(capacity, 0) {
		panic(fmt.Sprintf("NewTokenBucket: capacity must be a finite value > 0, got %v", capacity))
	}
	if refillRate <= 0 || math.IsNaN(refillRate) || math.IsInf(refillRate, 0) {
		panic(fmt.Sprintf("NewTokenBucket: refillRate must be a finite value > 0, got %v", refillRate))
	}
	return &TokenBucket{
		capacity:      capacity,
		refillRate:    refillRate,
		currentTokens: capacity,
	}
}

// Admit checks whether the request can be admitted given current token availability.
func (tb *TokenBucket) Admit(req *Request, state *RouterState) (bool, string) {
	clock := state.Clock
	elapsed := clock - tb.lastRefill
	if elapsed > 0 {
		refill := float64(elapsed) * tb.refillRate / 1e6
		tb.currentTokens = min(tb.capacity, tb.currentTokens+refill)
		tb.lastRefill = clock
	}
	cost := float64(len(req.InputTokens))
	if tb.currentTokens >= cost {
		tb.currentTokens -= cost
		return true, ""
	}
	return false, "insufficient tokens"
}

// SLOGatedAdmission selectively rejects sheddable requests during load spikes.
// When average EffectiveLoad across instances exceeds LoadThreshold, sheddable
// requests are rejected. Critical and standard requests are always admitted.
// This is a non-zero-sum mechanism: reducing queue depth during bursts benefits
// ALL tiers by shortening scheduling delay for everyone.
type SLOGatedAdmission struct {
	LoadThreshold float64 // average EffectiveLoad above which sheddable is rejected
}

// SLOGatedAdmissionConfig holds tunable parameters for slo-gated admission.
var SLOGatedAdmissionConfig = struct {
	LoadThreshold float64
}{
	LoadThreshold: 8.0,
}

func (s *SLOGatedAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	// Critical and standard always admitted
	if req.SLOClass == "critical" || req.SLOClass == "standard" || req.SLOClass == "" {
		return true, ""
	}

	// Compute average effective load across instances
	if len(state.Snapshots) == 0 {
		return true, ""
	}
	totalLoad := 0
	for _, snap := range state.Snapshots {
		totalLoad += snap.EffectiveLoad()
	}
	avgLoad := float64(totalLoad) / float64(len(state.Snapshots))

	if avgLoad > s.LoadThreshold {
		return false, fmt.Sprintf("slo-gated: sheddable rejected (avg_load=%.1f > threshold=%.1f)", avgLoad, s.LoadThreshold)
	}
	return true, ""
}

// RejectAll rejects all requests unconditionally (pathological template for testing).
type RejectAll struct{}

func (r *RejectAll) Admit(_ *Request, _ *RouterState) (bool, string) {
	return false, "reject-all"
}

// NewAdmissionPolicy creates an admission policy by name.
// Valid names are defined in ValidAdmissionPolicies (bundle.go).
// An empty string defaults to AlwaysAdmit (for CLI flag default compatibility).
// For token-bucket, capacity and refillRate configure the bucket.
// Panics on unrecognized names.
func NewAdmissionPolicy(name string, capacity, refillRate float64) AdmissionPolicy {
	if !IsValidAdmissionPolicy(name) {
		panic(fmt.Sprintf("unknown admission policy %q", name))
	}
	switch name {
	case "", "always-admit":
		return &AlwaysAdmit{}
	case "token-bucket":
		return NewTokenBucket(capacity, refillRate)
	case "reject-all":
		return &RejectAll{}
	case "slo-gated":
		return &SLOGatedAdmission{LoadThreshold: SLOGatedAdmissionConfig.LoadThreshold}
	default:
		panic(fmt.Sprintf("unhandled admission policy %q", name))
	}
}
