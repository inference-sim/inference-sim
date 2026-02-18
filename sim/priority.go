package sim

import "fmt"

// PriorityPolicy computes a priority score for a request.
// Higher scores indicate higher priority (scheduled first by priority-aware schedulers).
// Implementations MUST NOT modify the request — only the return value is used.
type PriorityPolicy interface {
	Compute(req *Request, clock int64) float64
}

// ConstantPriority assigns a fixed priority score to all requests.
type ConstantPriority struct {
	Score float64
}

func (c *ConstantPriority) Compute(_ *Request, _ int64) float64 {
	return c.Score
}

// SLOBasedPriority computes priority based on request age (time waiting).
// Older requests receive higher priority to reduce SLO violation risk.
// Formula: BaseScore + AgeWeight * float64(clock - req.ArrivalTime)
//
// With default AgeWeight=1e-6, a request waiting 1 second (1e6 ticks) gets +1.0 priority.
// Full SLO class integration (using TenantState) is deferred to a future PR.
type SLOBasedPriority struct {
	BaseScore float64
	AgeWeight float64
}

func (s *SLOBasedPriority) Compute(req *Request, clock int64) float64 {
	age := float64(clock - req.ArrivalTime)
	return s.BaseScore + s.AgeWeight*age
}

// InvertedSLO computes priority inversely to request age (pathological template).
// Newer requests get higher priority, starving older ones — the opposite of SLOBasedPriority.
// Formula: BaseScore - AgeWeight * float64(clock - req.ArrivalTime)
type InvertedSLO struct {
	BaseScore float64
	AgeWeight float64
}

func (s *InvertedSLO) Compute(req *Request, clock int64) float64 {
	age := float64(clock - req.ArrivalTime)
	return s.BaseScore - s.AgeWeight*age
}

// NewPriorityPolicy creates a PriorityPolicy by name.
// Valid names are defined in validPriorityPolicies (bundle.go).
// Empty string defaults to ConstantPriority (for CLI flag default compatibility).
// Panics on unrecognized names.
func NewPriorityPolicy(name string) PriorityPolicy {
	if !IsValidPriorityPolicy(name) {
		panic(fmt.Sprintf("unknown priority policy %q", name))
	}
	switch name {
	case "", "constant":
		return &ConstantPriority{Score: 0.0}
	case "slo-based":
		return &SLOBasedPriority{BaseScore: 0.0, AgeWeight: 1e-6}
	case "inverted-slo":
		return &InvertedSLO{BaseScore: 0.0, AgeWeight: 1e-6}
	default:
		panic(fmt.Sprintf("unhandled priority policy %q", name))
	}
}
