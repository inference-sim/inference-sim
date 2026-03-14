package cluster

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// InterferenceLatencyModel wraps a LatencyModel to apply a multiplicative slowdown
// when prefill and decode phases co-locate in the same batch. This enables break-even
// analysis between disaggregation transfer cost and co-location interference cost.
//
// Extension type: tier composition (wraps LatencyModel).
//
// Multiplier formula: 1.0 + factor * (minority_count / total_count)
// where minority_count is the count of requests in the less-common phase.
// A factor of 1.0 at a 50/50 split produces at most a 1.5× step time multiplier (50% slowdown).
// A factor of 0.5 at a 50/50 split produces a 1.25× multiplier (25% slowdown).
//
// In PD disaggregated mode, prefill-only and decode-only pools always have phase-pure
// batches (INV-PD-2: Pool Exclusivity), so their multiplier is always 1.0 (BC-P2-10:
// no-op). The interference factors only take effect in non-disaggregated deployments
// where prefill and decode requests share the same instance.
//
// Behavioral guarantees:
//   - BC-P2-9:  factors=0 → step time identical to inner model
//   - BC-P2-10: phase-pure batch → multiplier=1.0
//   - BC-P2-11/INV-P2-3: multiplier >= 1.0 always
//   - BC-P2-12: LastAppliedMultiplier() records per-call multiplier
type InterferenceLatencyModel struct {
	inner               sim.LatencyModel
	prefillInterference float64 // slowdown for prefill-dominant batches (minority is decode)
	decodeInterference  float64 // slowdown for decode-dominant batches (minority is prefill)
	lastMultiplier      float64
}

// MaxInterferenceFactor is the upper bound for interference factors (R3: numeric parameter upper bound).
// Factor=100 at a 50/50 split produces exactly 51× slowdown (1.0 + 100×0.5 = 51.0). Values above this
// would cause float64 overflow in StepTime for any realistic inner model step time (> 10^18 µs).
const MaxInterferenceFactor = 100.0

// NewInterferenceLatencyModel creates an interference wrapper around the given LatencyModel.
// prefillFactor is the interference factor when prefill is the majority phase.
// decodeFactor is the interference factor when decode is the majority phase.
// Both factors must be in [0, MaxInterferenceFactor] and finite (R3).
func NewInterferenceLatencyModel(inner sim.LatencyModel, prefillFactor, decodeFactor float64) (*InterferenceLatencyModel, error) {
	if inner == nil {
		return nil, fmt.Errorf("NewInterferenceLatencyModel: inner must not be nil")
	}
	if prefillFactor < 0 || math.IsNaN(prefillFactor) || math.IsInf(prefillFactor, 0) || prefillFactor > MaxInterferenceFactor {
		return nil, fmt.Errorf("NewInterferenceLatencyModel: prefillFactor must be a finite number in [0, %.0f], got %f", MaxInterferenceFactor, prefillFactor)
	}
	if decodeFactor < 0 || math.IsNaN(decodeFactor) || math.IsInf(decodeFactor, 0) || decodeFactor > MaxInterferenceFactor {
		return nil, fmt.Errorf("NewInterferenceLatencyModel: decodeFactor must be a finite number in [0, %.0f], got %f", MaxInterferenceFactor, decodeFactor)
	}
	return &InterferenceLatencyModel{
		inner:               inner,
		prefillInterference: prefillFactor,
		decodeInterference:  decodeFactor,
		lastMultiplier:      1.0,
	}, nil
}

// StepTime applies the interference multiplier to the inner model's step time.
// Classifies each request as prefill (ProgressIndex < len(InputTokens)) or decode,
// then applies: multiplier = 1.0 + factor * (minority_count / total_count).
func (m *InterferenceLatencyModel) StepTime(batch []*sim.Request) int64 {
	baseTime := m.inner.StepTime(batch)

	multiplier := m.computeMultiplier(batch)
	m.lastMultiplier = multiplier

	result := int64(math.Round(float64(baseTime) * multiplier))
	if result < 1 {
		result = 1
	}
	return result
}

// computeMultiplier determines the interference multiplier from batch composition.
func (m *InterferenceLatencyModel) computeMultiplier(batch []*sim.Request) float64 {
	total := len(batch)
	if total == 0 {
		return 1.0
	}

	prefillCount := 0
	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			prefillCount++
		}
	}
	decodeCount := total - prefillCount

	minorityCount := min(prefillCount, decodeCount)
	if minorityCount == 0 {
		return 1.0
	}

	var factor float64
	switch {
	case prefillCount > decodeCount:
		factor = m.prefillInterference
	case decodeCount > prefillCount:
		factor = m.decodeInterference
	default:
		// Equal split: use the larger factor (conservative — worst-case interference
		// applies when neither phase dominates and both experience the other at 50%).
		factor = max(m.prefillInterference, m.decodeInterference)
	}

	return 1.0 + factor*(float64(minorityCount)/float64(total))
}

// QueueingTime delegates to inner model (interference does not affect queueing).
func (m *InterferenceLatencyModel) QueueingTime(req *sim.Request) int64 {
	return m.inner.QueueingTime(req)
}

// OutputTokenProcessingTime delegates to inner model.
func (m *InterferenceLatencyModel) OutputTokenProcessingTime() int64 {
	return m.inner.OutputTokenProcessingTime()
}

// PostDecodeFixedOverhead delegates to inner model.
func (m *InterferenceLatencyModel) PostDecodeFixedOverhead() int64 {
	return m.inner.PostDecodeFixedOverhead()
}

// LastAppliedMultiplier returns the multiplier applied in the most recent StepTime call.
// Returns 1.0 before any StepTime call (BC-P2-12).
func (m *InterferenceLatencyModel) LastAppliedMultiplier() float64 {
	return m.lastMultiplier
}
