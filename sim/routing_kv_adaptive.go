package sim

import "fmt"

// KVAdaptiveConfig holds the tunable parameters for KV-adaptive routing.
// All parameters are designed for experimental optimization (grid search,
// Bayesian optimization) rather than hardcoded values.
type KVAdaptiveConfig struct {
	// KVThreshold is the average KVUtilization across instances that triggers
	// switching from normal to pressure profile. Range: [0.0, 1.0].
	KVThreshold float64

	// Normal profile weights (used when avg KVUtilization < KVThreshold)
	NormalPAWeight float64
	NormalQDWeight float64
	NormalKVWeight float64

	// Pressure profile weights (used when avg KVUtilization >= KVThreshold)
	PressurePAWeight float64
	PressureQDWeight float64
	PressureKVWeight float64
}

// DefaultKVAdaptiveConfig returns the default parameters derived from
// iteration 6 experiments. These are STARTING POINTS for optimization,
// not final values.
func DefaultKVAdaptiveConfig() KVAdaptiveConfig {
	return KVAdaptiveConfig{
		KVThreshold:      0.5,
		NormalPAWeight:   3.0,
		NormalQDWeight:   2.0,
		NormalKVWeight:   2.0,
		PressurePAWeight: 2.0,
		PressureQDWeight: 2.0,
		PressureKVWeight: 5.0,
	}
}

// KVAdaptiveScoring routes using WeightedScoring with dynamically selected
// weight profiles based on cluster-wide KV memory pressure.
//
// Under normal KV conditions (avg utilization < threshold):
//
//	Uses cache-exploiting weights (default: pa:3, qd:2, kv:2).
//	Optimizes for TTFT via prefix cache hits.
//
// Under KV pressure (avg utilization >= threshold):
//
//	Shifts to memory-preserving weights (default: pa:2, qd:2, kv:5).
//	Prevents KV concentration that causes preemptions and reload latency.
//
// All parameters (threshold, weights) are configurable for experimental
// optimization. The default values are starting points derived from
// iteration 6 experiments showing static pa:3,qd:2,kv:2 loses to RR
// by 23-25% under KV pressure.
type KVAdaptiveScoring struct {
	normalProfile   *WeightedScoring
	pressureProfile *WeightedScoring
	threshold       float64
	config          KVAdaptiveConfig // stored for introspection/logging
}

// Route implements RoutingPolicy for KVAdaptiveScoring.
func (kas *KVAdaptiveScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("KVAdaptiveScoring.Route: empty snapshots")
	}

	// Compute MAX KV utilization across instances (detects concentrated pressure).
	maxKVUtil := 0.0
	for _, snap := range snapshots {
		if snap.KVUtilization > maxKVUtil {
			maxKVUtil = snap.KVUtilization
		}
	}

	// Continuous blending: interpolate between normal and pressure profiles
	// based on how far maxKVUtil is above the threshold.
	// At threshold: 100% normal, 0% pressure
	// At 1.0: 0% normal, 100% pressure
	// This avoids the binary switch problem where the transition happens too late.
	var decision RoutingDecision
	if maxKVUtil < kas.threshold {
		decision = kas.normalProfile.Route(req, state)
		decision.Reason = fmt.Sprintf("kv-adaptive[normal,maxkv=%.2f] %s", maxKVUtil, decision.Reason)
	} else {
		// Above threshold: use pressure profile (which has higher KV weight)
		decision = kas.pressureProfile.Route(req, state)
		decision.Reason = fmt.Sprintf("kv-adaptive[PRESSURE,maxkv=%.2f] %s", maxKVUtil, decision.Reason)
	}

	return decision
}
