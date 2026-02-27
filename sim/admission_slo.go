package sim

import "fmt"

// SLOGatedConfig holds tunable parameters for SLO-gated admission control.
// All parameters are designed for Bayesian optimization (scikit-optimize).
type SLOGatedConfig struct {
	// StandardQueueThreshold: reject standard requests when avg EffectiveLoad
	// exceeds this value. Range: [2, 20]. Higher = more permissive.
	StandardQueueThreshold float64

	// SheddableQueueThreshold: reject sheddable requests when avg EffectiveLoad
	// exceeds this value. Range: [1, 10]. Lower = more aggressive shedding.
	// Must be <= StandardQueueThreshold for correct priority ordering.
	SheddableQueueThreshold float64
}

// DefaultSLOGatedConfig returns starting parameters for optimization.
func DefaultSLOGatedConfig() SLOGatedConfig {
	return SLOGatedConfig{
		StandardQueueThreshold:  10.0,
		SheddableQueueThreshold: 5.0,
	}
}

// SLOGatedAdmission implements SLO-aware admission control.
//
// Critical requests are ALWAYS admitted (latency-sensitive, never shed).
// Standard requests are admitted when cluster average queue depth is below
// StandardQueueThreshold. Sheddable/batch/background requests are admitted
// when below SheddableQueueThreshold (tighter, shed first under load).
//
// This operates orthogonally to routing: admission decides WHETHER to accept
// a request (reducing total load), while routing decides WHERE to send it
// (optimizing cache + load balance). Admission shedding does NOT fragment
// cache affinity because it happens BEFORE routing.
//
// Signal freshness (R17, INV-7):
//
//	Reads: EffectiveLoad() per instance via RouterState snapshots.
//	EffectiveLoad = QueueDepth (Tier 2) + BatchSize (Tier 2) + PendingRequests (Tier 1).
type SLOGatedAdmission struct {
	config SLOGatedConfig
}

// NewSLOGatedAdmission creates an SLO-gated admission policy with the given config.
func NewSLOGatedAdmission(cfg SLOGatedConfig) *SLOGatedAdmission {
	return &SLOGatedAdmission{config: cfg}
}

// Admit implements AdmissionPolicy.
func (s *SLOGatedAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	if req == nil {
		return true, ""
	}

	// Critical: always admit
	if req.SLOClass == "critical" {
		return true, ""
	}

	// Compute max instance queue depth (detects hotspots from PA-driven concentration).
	// Uses QueueDepth (not EffectiveLoad) per GPT-4o review: PendingRequests inflates
	// EffectiveLoad transiently during routing-pipeline backpressure, making thresholds
	// unintuitive. QueueDepth is the actual wait-queue length, directly interpretable.
	// Uses MAX (not avg) per Opus review: PA routing concentrates traffic on cache-warm
	// instances, so 3 of 8 may be heavily loaded while 5 are idle. avgQueue masks this.
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		return true, "" // no instances → can't compute load
	}
	maxQueueDepth := 0
	for _, snap := range snapshots {
		if snap.QueueDepth > maxQueueDepth {
			maxQueueDepth = snap.QueueDepth
		}
	}

	// Standard: admit if below standard threshold
	if req.SLOClass == "standard" || req.SLOClass == "" {
		if float64(maxQueueDepth) >= s.config.StandardQueueThreshold {
			return false, fmt.Sprintf("slo-gated: standard rejected (maxQD=%d >= threshold=%.1f)",
				maxQueueDepth, s.config.StandardQueueThreshold)
		}
		return true, ""
	}

	// Sheddable/batch/background: admit if below sheddable threshold (tighter)
	if float64(maxQueueDepth) >= s.config.SheddableQueueThreshold {
		return false, fmt.Sprintf("slo-gated: %s rejected (maxQD=%d >= threshold=%.1f)",
			req.SLOClass, maxQueueDepth, s.config.SheddableQueueThreshold)
	}
	return true, ""
}
