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
// Full SLO class integration (using TenantState) is planned for PR8.
type SLOBasedPriority struct {
	BaseScore float64
	AgeWeight float64
}

func (s *SLOBasedPriority) Compute(req *Request, clock int64) float64 {
	age := float64(clock - req.ArrivalTime)
	return s.BaseScore + s.AgeWeight*age
}

// NewPriorityPolicy creates a PriorityPolicy by name.
// Valid names: "constant" (default), "slo-based".
// Empty string defaults to ConstantPriority (for CLI flag default compatibility).
// Panics on unrecognized names.
func NewPriorityPolicy(name string) PriorityPolicy {
	switch name {
	case "", "constant":
		return &ConstantPriority{Score: 0.0}
	case "slo-based":
		return &SLOBasedPriority{BaseScore: 0.0, AgeWeight: 1e-6}
	default:
		panic(fmt.Sprintf("unknown priority policy %q", name))
	}
}
