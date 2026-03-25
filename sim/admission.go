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

// RejectAll rejects all requests unconditionally (pathological template for testing).
type RejectAll struct{}

func (r *RejectAll) Admit(_ *Request, _ *RouterState) (bool, string) {
	return false, "reject-all"
}

// SLOTierPriority maps an SLOClass string to an integer priority.
// Higher = more important. Background=0 … Critical=4.
// Empty or unknown string maps to Standard (3) for backward compatibility.
// Exported so sim/cluster/ can call it without a circular import.
func SLOTierPriority(class string) int {
	switch class {
	case "critical":
		return 4
	case "standard":
		return 3
	case "sheddable":
		return 2
	case "batch":
		return 1
	case "background":
		return 0
	default:
		return 3 // empty or unknown → Standard
	}
}

// TierShedAdmission sheds lower-priority requests under overload.
// Stateless: all decisions computed from RouterState at call time.
// Batch and Background always pass through (deferred queue PR handles them).
// Use NewTierShedAdmission to construct with validated parameters.
type TierShedAdmission struct {
	OverloadThreshold int // max per-instance effective load before shedding; 0 = any load triggers
	MinAdmitPriority  int // minimum tier priority admitted under overload; 0 = admit all (footgun)
}

// NewTierShedAdmission creates a TierShedAdmission with validated parameters.
// Panics if overloadThreshold < 0 or minAdmitPriority is outside [0, 4] (R3).
func NewTierShedAdmission(overloadThreshold, minAdmitPriority int) *TierShedAdmission {
	if overloadThreshold < 0 {
		panic(fmt.Sprintf("NewTierShedAdmission: overloadThreshold must be >= 0, got %d", overloadThreshold))
	}
	if minAdmitPriority < 0 || minAdmitPriority > 4 {
		panic(fmt.Sprintf("NewTierShedAdmission: minAdmitPriority must be in [0,4], got %d", minAdmitPriority))
	}
	return &TierShedAdmission{
		OverloadThreshold: overloadThreshold,
		MinAdmitPriority:  minAdmitPriority,
	}
}

// Admit rejects requests whose tier priority is below MinAdmitPriority when the
// cluster is overloaded (max effective load across instances > OverloadThreshold).
// Batch and Background classes always return admitted=true.
// Empty Snapshots (no instances) also returns admitted=true (safe default).
func (t *TierShedAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	class := req.SLOClass
	// Batch/Background bypass tier-shed (deferred queue handles them in PR-2).
	if class == "batch" || class == "background" {
		return true, ""
	}
	// Compute max effective load across all instance snapshots.
	maxLoad := 0
	for _, snap := range state.Snapshots {
		if l := snap.EffectiveLoad(); l > maxLoad {
			maxLoad = l
		}
	}
	if maxLoad <= t.OverloadThreshold {
		return true, "" // under threshold: admit all
	}
	// Under overload: reject tiers below MinAdmitPriority.
	priority := SLOTierPriority(class)
	if priority < t.MinAdmitPriority {
		return false, fmt.Sprintf("tier-shed: class=%s priority=%d < min=%d load=%d",
			class, priority, t.MinAdmitPriority, maxLoad)
	}
	return true, ""
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
	default:
		panic(fmt.Sprintf("unhandled admission policy %q", name))
	}
}
