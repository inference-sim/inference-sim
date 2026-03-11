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

// SLOGatedAdmission admits protected SLO classes unconditionally,
// but rejects non-protected requests when total queue depth exceeds a threshold.
// This provides non-zero-sum load management: rejected sheddable requests
// free compute for admitted requests of all classes.
type SLOGatedAdmission struct {
	protectedClasses map[string]bool // SLO classes always admitted (unexported per R8)
	queueThreshold   int             // total queue depth above which non-protected is rejected
}

// NewSLOGatedAdmission creates an SLOGatedAdmission policy.
// Panics if queueThreshold <= 0 (R3: validate at construction).
func NewSLOGatedAdmission(protectedClasses []string, queueThreshold int) *SLOGatedAdmission {
	if queueThreshold <= 0 {
		panic(fmt.Sprintf("NewSLOGatedAdmission: queueThreshold must be > 0, got %d", queueThreshold))
	}
	protected := make(map[string]bool, len(protectedClasses))
	for _, cls := range protectedClasses {
		protected[cls] = true
	}
	return &SLOGatedAdmission{
		protectedClasses: protected,
		queueThreshold:   queueThreshold,
	}
}

// Admit checks whether the request should be admitted.
// Protected SLO classes are always admitted. Non-protected classes are rejected
// when the total queue depth across all instances exceeds the threshold.
func (s *SLOGatedAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	if s.protectedClasses[req.SLOClass] {
		return true, ""
	}
	totalQueueDepth := 0
	for _, snap := range state.Snapshots {
		totalQueueDepth += snap.QueueDepth
	}
	if totalQueueDepth > s.queueThreshold {
		return false, fmt.Sprintf("slo-gated: %s rejected (queue %d > threshold %d)", req.SLOClass, totalQueueDepth, s.queueThreshold)
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
		// capacity parameter reused as queue threshold for slo-gated
		return NewSLOGatedAdmission([]string{"critical", "standard"}, int(capacity))
	default:
		panic(fmt.Sprintf("unhandled admission policy %q", name))
	}
}
