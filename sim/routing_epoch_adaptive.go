package sim

import (
	"fmt"
	"math"
)

// EpochAdaptiveConfig holds tunable parameters for epoch-based online weight adaptation.
// All parameters designed for Bayesian optimization.
type EpochAdaptiveConfig struct {
	// EpochSize: adapt weights every N requests
	EpochSize int // default: 100, range: [20, 500]

	// Rejection rate thresholds for adaptation
	HighThreshold float64 // default: 0.10 (10% rejection → increase QD)
	LowThreshold  float64 // default: 0.02 (2% rejection → increase PA)

	// Weight adjustment magnitude per epoch
	StepSize float64 // default: 0.5, range: [0.1, 2.0]

	// Weight bounds
	PAMin float64 // default: 1.0
	PAMax float64 // default: 5.0
	QDMin float64 // default: 2.0
	QDMax float64 // default: 5.0

	// Safety rule: maximum PA:QD ratio to prevent cascade failures (iter 14 finding)
	MaxPAQDRatio float64 // default: 1.33

	// Initial weights
	InitialPA float64 // default: 3.0
	InitialQD float64 // default: 2.0
}

// DefaultEpochAdaptiveConfig returns starting parameters for optimization.
func DefaultEpochAdaptiveConfig() EpochAdaptiveConfig {
	return EpochAdaptiveConfig{
		EpochSize:     100,
		HighThreshold: 0.10,
		LowThreshold:  0.02,
		StepSize:      0.5,
		PAMin:         1.0,
		PAMax:         5.0,
		QDMin:         2.0,
		QDMax:         5.0,
		MaxPAQDRatio:  1.33,
		InitialPA:     3.0,
		InitialQD:     2.0,
	}
}

// EpochAdaptiveScoring routes using WeightedScoring with continuously adapted
// PA:QD weights based on the observed admission rejection rate.
//
// The adaptation loop:
//  1. Every EpochSize requests, compute rejection rate = epochRejects / epochCounter
//  2. If rejection rate > HighThreshold: system overloaded → decrease PA, increase QD
//  3. If rejection rate < LowThreshold: system has headroom → increase PA, decrease QD
//  4. Enforce PA:QD ≤ MaxPAQDRatio safety rule (prevents iter-14's cascade failure)
//  5. Rebuild the scorer pipeline with new weights
//
// The admission controller calls RecordRejection() for each rejected request,
// providing the free real-time learning signal.
//
// Convergence: at equilibrium, rejection rate stabilizes between thresholds.
// This is a bang-bang controller with hysteresis — convergent when StepSize
// is small relative to the system's response sensitivity.
type EpochAdaptiveScoring struct {
	currentPA float64
	currentQD float64
	scorer    *WeightedScoring

	// Epoch tracking
	epochCounter int
	epochRejects int

	// Shared state
	prefixIdx *PrefixCacheIndex

	config EpochAdaptiveConfig
}

// RejectionObserver is an optional interface that routing policies can implement
// to receive admission rejection notifications. Used by EpochAdaptiveScoring
// for online weight adaptation. The cluster event loop checks for this interface
// via type assertion after each rejection (R13: multi-impl — EpochAdaptiveScoring
// implements it, all other policies don't need to).
type RejectionObserver interface {
	RecordRejection()
}

// RecordRejection implements RejectionObserver.
// Called by the cluster event loop when a request is rejected by the admission controller.
// This is the online learning signal that drives weight adaptation.
func (e *EpochAdaptiveScoring) RecordRejection() {
	e.epochRejects++
}

// CurrentWeights returns the current PA and QD weights (for logging/monitoring).
func (e *EpochAdaptiveScoring) CurrentWeights() (pa, qd float64) {
	return e.currentPA, e.currentQD
}

// Route implements RoutingPolicy for EpochAdaptiveScoring.
func (e *EpochAdaptiveScoring) Route(req *Request, state *RouterState) RoutingDecision {
	e.epochCounter++

	// Check if epoch is complete
	if e.epochCounter >= e.config.EpochSize {
		e.adaptWeights()
	}

	// Delegate to current scorer
	decision := e.scorer.Route(req, state)
	decision.Reason = fmt.Sprintf("epoch-adaptive[pa=%.1f,qd=%.1f,rej=%d/%d] %s",
		e.currentPA, e.currentQD, e.epochRejects, e.epochCounter, decision.Reason)
	return decision
}

// adaptWeights adjusts PA:QD weights using DUAL-SIGNAL disambiguation.
//
// Fix for Opus review (iter 15): a single rejection rate signal is ambiguous —
// both too-much-PA and too-much-QD can cause high rejection. The fix uses
// TWO independent signals:
//   - Rejection rate (from admission) → controls QD weight (unambiguous: rejection = overload → more QD)
//   - Cache hit rate (from routing PA scores) → controls PA weight (unambiguous: cache available → more PA)
//
// Uses multiplicative weights (GPT-4o review) for smooth proportional response
// instead of bang-bang step control which causes limit-cycle oscillation.
func (e *EpochAdaptiveScoring) adaptWeights() {
	if e.epochCounter == 0 {
		return
	}

	rejectionRate := float64(e.epochRejects) / float64(e.epochCounter)

	// SIGNAL 1: Rejection rate → QD adjustment (unambiguous direction)
	// High rejection → always increase QD (more load spreading needed)
	// Low rejection → decrease QD (less spreading, free up weight budget for PA)
	alpha := e.config.StepSize / 10.0 // multiplicative learning rate
	if rejectionRate > e.config.HighThreshold {
		// Overloaded: increase QD proportionally to rejection severity
		boost := 1.0 + alpha*(rejectionRate-e.config.HighThreshold)*10
		e.currentQD = math.Min(e.config.QDMax, e.currentQD*boost)
	} else if rejectionRate < e.config.LowThreshold {
		// Headroom: decrease QD slightly (allow more room for PA)
		decay := 1.0 - alpha*0.5
		e.currentQD = math.Max(e.config.QDMin, e.currentQD*decay)
	}

	// SIGNAL 2: Cache availability → PA adjustment
	// If the PA scorer is producing differentiated scores (some instances have cache
	// hits while others don't), PA weight should be high. If all scores are similar
	// (no cache benefit), PA weight should be low.
	// Heuristic: if rejection is low AND we have headroom, increase PA (cache available).
	// If rejection is high, decrease PA (avoid concentration under load).
	if rejectionRate < e.config.LowThreshold {
		// Low rejection = headroom → safe to increase cache exploitation
		e.currentPA = math.Min(e.config.PAMax, e.currentPA*(1.0+alpha*0.5))
	} else if rejectionRate > e.config.HighThreshold {
		// High rejection → reduce cache concentration to avoid hot spots
		decay := 1.0 - alpha*(rejectionRate-e.config.HighThreshold)*5
		if decay < 0.8 {
			decay = 0.8 // don't drop PA too fast
		}
		e.currentPA = math.Max(e.config.PAMin, e.currentPA*decay)
	}

	// Enforce safety rule: PA:QD ≤ MaxPAQDRatio (R11: guard division)
	if e.currentQD > 0 && e.currentPA/e.currentQD > e.config.MaxPAQDRatio {
		e.currentQD = e.currentPA / e.config.MaxPAQDRatio
		if e.currentQD > e.config.QDMax {
			e.currentQD = e.config.QDMax
			e.currentPA = e.currentQD * e.config.MaxPAQDRatio
		}
	}

	// Rebuild scorer pipeline with new weights
	e.rebuildScorer()

	// Reset epoch counters
	e.epochCounter = 0
	e.epochRejects = 0
}

// rebuildScorer creates a new WeightedScoring with the current PA:QD weights.
func (e *EpochAdaptiveScoring) rebuildScorer() {
	configs := []ScorerConfig{
		{Name: "prefix-affinity", Weight: e.currentPA},
		{Name: "queue-depth", Weight: e.currentQD},
	}

	scorers := make([]scorerFunc, len(configs))
	var observers []observerFunc

	for i, cfg := range configs {
		if cfg.Name == "prefix-affinity" {
			scorer, obs := newPrefixAffinityScorerWithIndex(e.prefixIdx)
			scorers[i] = scorer
			if obs != nil {
				observers = append(observers, obs)
			}
		} else {
			scorer, obs := newScorerWithObserver(cfg.Name, e.prefixIdx.BlockSize())
			scorers[i] = scorer
			if obs != nil {
				observers = append(observers, obs)
			}
		}
	}

	e.scorer = &WeightedScoring{
		scorers:   scorers,
		weights:   normalizeScorerWeights(configs),
		observers: observers,
	}
}
