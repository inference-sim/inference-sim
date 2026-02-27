package sim

import (
	"fmt"
	"math"
)

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

// SLOTieredPriority assigns priority based on SLO class with piecewise-linear urgency escalation.
//
// Formula: base[class] + max(0, AgeWeight * (age - threshold[class]))
//
// Base scores provide immediate separation at enqueue time. The threshold gives
// lower-priority classes a grace period before urgency activates, preventing them
// from competing with critical requests during the grace window. After the threshold,
// urgency escalates at the shared AgeWeight rate for all classes.
//
// Empty SLOClass maps to "standard" (safe default for legacy workloads).
type SLOTieredPriority struct {
	BaseCritical       float64 // base score for "critical" (highest)
	BaseStandard       float64 // base score for "standard" / empty
	BaseSheddable      float64 // base score for "sheddable", "batch", "background" (lowest)
	AgeWeight          float64 // shared age escalation rate (ticks⁻¹)
	ThresholdStandard  int64   // age (μs) before standard urgency activates
	ThresholdSheddable int64   // age (μs) before sheddable urgency activates
}

func (s *SLOTieredPriority) Compute(req *Request, clock int64) float64 {
	age := float64(clock - req.ArrivalTime)
	var base float64
	var threshold int64
	switch req.SLOClass {
	case "critical":
		base = s.BaseCritical
		threshold = 0 // critical escalates immediately
	case "", "standard":
		base = s.BaseStandard
		threshold = s.ThresholdStandard
	default: // "sheddable", "batch", "background"
		base = s.BaseSheddable
		threshold = s.ThresholdSheddable
	}
	urgency := math.Max(0, s.AgeWeight*(age-float64(threshold)))
	return base + urgency
}

// SLOTieredPriorityConfig holds tunable parameters for the slo-tiered priority policy.
// Set via CLI flags; zero values use defaults.
var SLOTieredPriorityConfig = struct {
	BaseCritical       float64
	BaseStandard       float64
	BaseSheddable      float64
	AgeWeight          float64
	ThresholdStandard  int64
	ThresholdSheddable int64
}{
	BaseCritical:       10.0,
	BaseStandard:       5.0,
	BaseSheddable:      1.0,
	AgeWeight:          1e-5,
	ThresholdStandard:  100000,
	ThresholdSheddable: 200000,
}

// LoadAdaptivePriority scales the SLO priority gap based on current queue depth.
// At sub-saturation (queueDepth <= LowLoadThreshold), the gap compresses toward zero
// (approaching FCFS, which H23 proved optimal at low load). At high load
// (queueDepth >= HighLoadThreshold), the gap expands to full SLO-tiered levels.
// Between thresholds, the gap interpolates linearly.
//
// This eliminates unnecessary sheddable degradation during inter-burst periods
// while maintaining full SLO differentiation during bursts.
//
// Call SetQueueDepth() before each Compute() loop to update the load signal.
type LoadAdaptivePriority struct {
	// Full-gap SLO priority (used when load_factor = 1.0)
	BaseCritical       float64
	BaseStandard       float64
	BaseSheddable      float64
	AgeWeight          float64
	ThresholdStandard  int64
	ThresholdSheddable int64
	// Load-regime thresholds
	LowLoadThreshold  int // queue depth below which gap = 0 (FCFS)
	HighLoadThreshold int // queue depth above which gap = full
	// Current load state (updated via SetQueueDepth)
	queueDepth int
}

// SetQueueDepth updates the observed queue depth for load-adaptive gap scaling.
// Called by the Simulator before each priority computation loop.
func (l *LoadAdaptivePriority) SetQueueDepth(depth int) {
	l.queueDepth = depth
}

func (l *LoadAdaptivePriority) Compute(req *Request, clock int64) float64 {
	// Compute load factor: 0 at sub-saturation, 1 at near-saturation
	loadFactor := 0.0
	if l.HighLoadThreshold > l.LowLoadThreshold {
		loadFactor = float64(l.queueDepth-l.LowLoadThreshold) / float64(l.HighLoadThreshold-l.LowLoadThreshold)
		if loadFactor < 0 {
			loadFactor = 0
		}
		if loadFactor > 1 {
			loadFactor = 1
		}
	} else if l.queueDepth > 0 {
		loadFactor = 1.0 // degenerate case: always full gap
	}

	// Scale base scores: at loadFactor=0, all get BaseSheddable (FCFS-like)
	// at loadFactor=1, full spread
	age := float64(clock - req.ArrivalTime)
	var fullBase float64
	var threshold int64
	switch req.SLOClass {
	case "critical":
		fullBase = l.BaseSheddable + loadFactor*(l.BaseCritical-l.BaseSheddable)
		threshold = 0
	case "", "standard":
		fullBase = l.BaseSheddable + loadFactor*(l.BaseStandard-l.BaseSheddable)
		threshold = l.ThresholdStandard
	default: // "sheddable", "batch", "background"
		fullBase = l.BaseSheddable
		threshold = l.ThresholdSheddable
	}
	urgency := math.Max(0, l.AgeWeight*(age-float64(threshold)))
	return fullBase + urgency
}

// LoadAdaptivePriorityConfig holds tunable parameters for the load-adaptive priority policy.
var LoadAdaptivePriorityConfig = struct {
	BaseCritical       float64
	BaseStandard       float64
	BaseSheddable      float64
	AgeWeight          float64
	ThresholdStandard  int64
	ThresholdSheddable int64
	LowLoadThreshold   int
	HighLoadThreshold  int
}{
	BaseCritical:       10.0,
	BaseStandard:       5.0,
	BaseSheddable:      1.0,
	AgeWeight:          1e-5,
	ThresholdStandard:  100000,
	ThresholdSheddable: 200000,
	LowLoadThreshold:   2,
	HighLoadThreshold:  15,
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
	case "slo-tiered":
		cfg := SLOTieredPriorityConfig
		return &SLOTieredPriority{
			BaseCritical:       cfg.BaseCritical,
			BaseStandard:       cfg.BaseStandard,
			BaseSheddable:      cfg.BaseSheddable,
			AgeWeight:          cfg.AgeWeight,
			ThresholdStandard:  cfg.ThresholdStandard,
			ThresholdSheddable: cfg.ThresholdSheddable,
		}
	case "load-adaptive":
		cfg := LoadAdaptivePriorityConfig
		return &LoadAdaptivePriority{
			BaseCritical:       cfg.BaseCritical,
			BaseStandard:       cfg.BaseStandard,
			BaseSheddable:      cfg.BaseSheddable,
			AgeWeight:          cfg.AgeWeight,
			ThresholdStandard:  cfg.ThresholdStandard,
			ThresholdSheddable: cfg.ThresholdSheddable,
			LowLoadThreshold:   cfg.LowLoadThreshold,
			HighLoadThreshold:  cfg.HighLoadThreshold,
		}
	default:
		panic(fmt.Sprintf("unhandled priority policy %q", name))
	}
}
