package policy

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// AdmissionPolicy decides whether a request should be admitted to the cluster.
// This interface matches cluster.AdmissionPolicy for duck-typing compatibility.
type AdmissionPolicy interface {
	Admit(req *sim.Request, clock int64) (admitted bool, reason string)
}

// AlwaysAdmit admits all requests unconditionally.
type AlwaysAdmit struct{}

func (a *AlwaysAdmit) Admit(_ *sim.Request, _ int64) (bool, string) {
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
func NewTokenBucket(capacity, refillRate float64) *TokenBucket {
	return &TokenBucket{
		capacity:      capacity,
		refillRate:    refillRate,
		currentTokens: capacity,
	}
}

// Admit checks whether the request can be admitted given current token availability.
func (tb *TokenBucket) Admit(req *sim.Request, clock int64) (bool, string) {
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

// NewAdmissionPolicy creates an admission policy by name.
// Valid names: "always-admit", "token-bucket".
// For token-bucket, capacity and refillRate configure the bucket.
func NewAdmissionPolicy(name string, capacity, refillRate float64) AdmissionPolicy {
	switch name {
	case "always-admit":
		return &AlwaysAdmit{}
	case "token-bucket":
		return NewTokenBucket(capacity, refillRate)
	default:
		panic(fmt.Sprintf("unknown admission policy %q; valid policies: [always-admit, token-bucket]", name))
	}
}
