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

// StaticClassWeight assigns a fixed priority based on request SLO class.
// Requests whose SLOClass is not in ClassWeights receive DefaultWeight.
// This policy is time-independent — clock is ignored.
type StaticClassWeight struct {
	ClassWeights  map[string]float64
	DefaultWeight float64
}

func (s *StaticClassWeight) Compute(req *Request, _ int64) float64 {
	if w, ok := s.ClassWeights[req.SLOClass]; ok {
		return w
	}
	return s.DefaultWeight
}

// DeadlineAwarePriority computes urgency from per-SLO-class TTFT deadlines.
// Formula: urgency = classWeight(SLOClass) / max(epsilon, 1.0 - elapsed / deadline(SLOClass))
// As a request approaches its deadline, urgency grows toward weight/epsilon.
// Past the deadline, urgency is capped at weight/epsilon (no division by zero or negative).
type DeadlineAwarePriority struct {
	ClassWeights    map[string]float64
	Deadlines       map[string]int64
	Epsilon         float64
	DefaultWeight   float64
	DefaultDeadline int64
}

func (d *DeadlineAwarePriority) Compute(req *Request, clock int64) float64 {
	weight := d.DefaultWeight
	if w, ok := d.ClassWeights[req.SLOClass]; ok {
		weight = w
	}
	deadline := d.DefaultDeadline
	if dl, ok := d.Deadlines[req.SLOClass]; ok {
		deadline = dl
	}
	elapsed := float64(clock - req.ArrivalTime)
	fraction := elapsed / float64(deadline)
	denominator := 1.0 - fraction
	if denominator < d.Epsilon {
		denominator = d.Epsilon
	}
	return weight / denominator
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
	case "static-class-weight":
		return &StaticClassWeight{
			ClassWeights:  map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
			DefaultWeight: 0.0,
		}
	case "deadline-aware":
		return &DeadlineAwarePriority{
			ClassWeights:    map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
			Deadlines:       map[string]int64{"critical": 100_000, "standard": 500_000, "sheddable": 2_000_000},
			Epsilon:         0.01,
			DefaultWeight:   0.0,
			DefaultDeadline: 500_000,
		}
	default:
		panic(fmt.Sprintf("unhandled priority policy %q", name))
	}
}
