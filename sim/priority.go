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
// Per-request SLO metadata is available on Request.SLOClass but not yet used by this scorer.
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

// SLOClassPriority assigns different base priority scores based on SLO class.
// Critical requests get the highest base score (scheduled first), batch/background
// get the lowest (scheduled last). Age-weighted anti-starvation ensures no class
// is permanently starved: after sufficient waiting, lower-priority requests catch up.
//
// This enables routing-scheduling co-optimization: the router sends critical requests
// to low-load instances (via qd-heavy profile) and batch requests to cache-warm instances
// (via pa-heavy profile). When both types arrive at the same instance, the scheduler
// uses SLO-class priority to schedule critical first.
//
// Base scores: critical=10, standard=5, sheddable=3, batch=1, background=0.
// With AgeWeight=1e-6, a batch request (base=1) catches a fresh critical request (base=10)
// after 9 seconds of waiting — well within even critical SLO budgets.
type SLOClassPriority struct {
	BaseScores map[string]float64 // SLO class → base priority (unexported map, R8)
	AgeWeight  float64
}

func (s *SLOClassPriority) Compute(req *Request, clock int64) float64 {
	base := s.BaseScores[req.SLOClass] // returns 0.0 for unknown classes (map zero-value)
	age := float64(clock - req.ArrivalTime)
	return base + s.AgeWeight*age
}

// defaultSLOClassBaseScores returns the default base priority scores per SLO class.
// Unexported to prevent mutation (R8).
func defaultSLOClassBaseScores() map[string]float64 {
	return map[string]float64{
		"critical":   10.0,
		"standard":   5.0,
		"sheddable":  3.0,
		"batch":      1.0,
		"background": 0.0,
	}
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
	case "slo-class":
		// AgeWeight=1e-5: batch (base=1) catches fresh critical (base=10) after 0.9s.
		// This is fast enough to prevent starvation while maintaining meaningful
		// SLO differentiation within typical request lifetimes.
		return &SLOClassPriority{BaseScores: defaultSLOClassBaseScores(), AgeWeight: 1e-5}
	default:
		panic(fmt.Sprintf("unhandled priority policy %q", name))
	}
}
