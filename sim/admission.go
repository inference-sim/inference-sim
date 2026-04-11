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

// SLOPriorityMap maps SLOClass strings to integer priorities.
// Higher = more important. Negative = sheddable (matches GAIE's IsSheddable contract).
// Unexported map field prevents external mutation (R8).
type SLOPriorityMap struct {
	priorities map[string]int
	defaultPri int
}

// DefaultSLOPriorityMap returns the GAIE-compatible default priority mapping.
// critical=4, standard=3, batch=-1, sheddable=-2, background=-3.
// Empty or unknown class → 3 (Standard).
func DefaultSLOPriorityMap() *SLOPriorityMap {
	return &SLOPriorityMap{
		priorities: map[string]int{
			"critical":   4,
			"standard":   3,
			"batch":      -1,
			"sheddable":  -2,
			"background": -3,
		},
		defaultPri: 3,
	}
}

// NewSLOPriorityMap creates a priority map with defaults, then applies overrides.
// Nil or empty overrides → pure defaults. Override keys replace defaults for those classes.
func NewSLOPriorityMap(overrides map[string]int) *SLOPriorityMap {
	m := DefaultSLOPriorityMap()
	for k, v := range overrides {
		m.priorities[k] = v
	}
	return m
}

// Priority returns the integer priority for the given SLO class.
// Unknown or empty class → default priority (3 = Standard).
func (m *SLOPriorityMap) Priority(class string) int {
	if p, ok := m.priorities[class]; ok {
		return p
	}
	return m.defaultPri
}

// IsSheddable returns true iff the class has priority < 0.
// Matches GAIE's util/request/sheddable.go:21 contract.
func (m *SLOPriorityMap) IsSheddable(class string) bool {
	return m.Priority(class) < 0
}

// SLOTierPriority maps an SLOClass string to an integer priority using GAIE-compatible defaults.
// Deprecated: use SLOPriorityMap.Priority() for configurable priorities.
// Kept for backward compatibility — delegates to DefaultSLOPriorityMap().
// Note: return values changed from old [0,4] scale to GAIE scale (negative = sheddable).
// Not value-compatible with pre-#1013 code — use IsSheddable() for shedding decisions.
func SLOTierPriority(class string) int {
	return DefaultSLOPriorityMap().Priority(class)
}

// TierShedAdmission sheds lower-priority requests under overload.
// Stateless: all decisions computed from RouterState at call time.
// Use NewTierShedAdmission to construct with validated parameters.
type TierShedAdmission struct {
	OverloadThreshold int             // max per-instance effective load before shedding; 0 = any load triggers
	MinAdmitPriority  int             // minimum tier priority admitted under overload
	PriorityMap       *SLOPriorityMap // configurable priority mapping (nil-safe: defaults used)
}

// NewTierShedAdmission creates a TierShedAdmission with validated parameters and a priority map.
// Panics if overloadThreshold < 0. minAdmitPriority is unbounded (GAIE priorities are
// arbitrary integers with no range constraint; only the sign matters for IsSheddable).
// If priorityMap is nil, DefaultSLOPriorityMap() is used.
func NewTierShedAdmission(overloadThreshold, minAdmitPriority int, priorityMap *SLOPriorityMap) *TierShedAdmission {
	if overloadThreshold < 0 {
		panic(fmt.Sprintf("NewTierShedAdmission: overloadThreshold must be >= 0, got %d", overloadThreshold))
	}
	if priorityMap == nil {
		priorityMap = DefaultSLOPriorityMap()
	}
	return &TierShedAdmission{
		OverloadThreshold: overloadThreshold,
		MinAdmitPriority:  minAdmitPriority,
		PriorityMap:       priorityMap,
	}
}

// Admit rejects requests whose tier priority is below MinAdmitPriority when the
// cluster is overloaded (max effective load across instances > OverloadThreshold).
// Empty Snapshots (no instances) also returns admitted=true (safe default).
func (t *TierShedAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	class := req.SLOClass
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
	priority := t.PriorityMap.Priority(class)
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
