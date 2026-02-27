package sim

import (
	"fmt"
	"math"
)

// PredictiveSLOConfig holds tunable parameters for predictive TTFT-budget admission.
// All parameters are designed for Bayesian optimization.
type PredictiveSLOConfig struct {
	// Per-SLO-class TTFT budgets in microseconds (ticks).
	BudgetCritical  float64 // default: 200000 (200ms)
	BudgetStandard  float64 // default: 500000 (500ms)
	BudgetSheddable float64 // default: 300000 (300ms)

	// Headroom multiplier on budget. 1.0 = exact, 1.5 = 50% tolerance.
	Headroom float64 // default: 1.0, range: [0.5, 3.0]

	// Average step time for queue wait estimation (microseconds).
	AvgStepTime float64 // default: 7000 (7ms)

	// Beta coefficients for prefill time estimation.
	Beta0 float64 // step overhead (default: 6910.42)
	Beta1 float64 // per-cache-miss-token cost (default: 17.67)
}

// DefaultPredictiveSLOConfig returns starting parameters for optimization.
func DefaultPredictiveSLOConfig() PredictiveSLOConfig {
	return PredictiveSLOConfig{
		BudgetCritical:  200000,
		BudgetStandard:  500000,
		BudgetSheddable: 300000,
		Headroom:        1.0,
		AvgStepTime:     7000,
		Beta0:           6910.42,
		Beta1:           17.67,
	}
}

// PredictiveSLOAdmission implements physics-informed admission control.
//
// Instead of a crude queue-depth threshold (SLOGatedAdmission), this policy
// estimates each request's BEST-CASE TTFT across all instances using the
// latency model's beta coefficients and the PrefixCacheIndex's cache state.
// It then admits only if the estimated TTFT is within the request's SLO budget.
//
// This is per-request, cache-aware, and SLO-budget-aware:
//   - A sheddable request with a full cache hit (7ms prefill) is admitted even
//     under high load, because its estimated TTFT is low.
//   - A sheddable request with zero cache (82ms prefill) is rejected under
//     moderate load, because it would miss its SLO budget anyway.
//
// Critical requests are ALWAYS admitted regardless of estimated TTFT.
//
// Signal freshness (R17, INV-7):
//
//	Reads: QueueDepth (Tier 2: Immediate mode), PrefixCacheIndex (Tier 1: synchronous).
//	Beta coefficients captured at construction (constant).
type PredictiveSLOAdmission struct {
	config    PredictiveSLOConfig
	prefixIdx *PrefixCacheIndex
}

// NewPredictiveSLOAdmission creates a predictive SLO admission policy.
// The PrefixCacheIndex must be shared with the routing layer's PA scorer
// so both see the same cache state.
func NewPredictiveSLOAdmission(cfg PredictiveSLOConfig, prefixIdx *PrefixCacheIndex) *PredictiveSLOAdmission {
	return &PredictiveSLOAdmission{
		config:    cfg,
		prefixIdx: prefixIdx,
	}
}

// Admit implements AdmissionPolicy.
func (p *PredictiveSLOAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	if req == nil {
		return true, ""
	}

	// Critical: ALWAYS admit
	if req.SLOClass == "critical" {
		return true, "critical-always-admit"
	}

	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		return true, ""
	}

	// Compute block hashes for cache match estimation
	totalTokens := len(req.InputTokens)
	var blockHashes []string
	if p.prefixIdx != nil && totalTokens > 0 {
		blockHashes = p.prefixIdx.ComputeBlockHashes(req.InputTokens)
	}

	// Cold-start bypass: if this prefix has NEVER been seen by the cache index
	// (no instance has any match), admit unconditionally to allow cache warming.
	// This prevents the starvation problem where new prefix groups are permanently
	// rejected because they always have zero cache match → high estimated TTFT.
	anyMatch := false
	if p.prefixIdx != nil && len(blockHashes) > 0 {
		for _, snap := range snapshots {
			if p.prefixIdx.MatchLength(blockHashes, snap.ID) > 0 {
				anyMatch = true
				break
			}
		}
	}
	if !anyMatch {
		return true, "cold-start-bypass"
	}

	// Find BEST-CASE estimated TTFT across all instances.
	// Uses continuous-batching-aware queue wait estimation:
	//   queueWait = ceil(QueueDepth / batchCapacity) × AvgStepTime
	// where batchCapacity = max(1, maxBatchSlots - BatchSize).
	// This accounts for multiple requests being scheduled per step.
	bestTTFT := math.MaxFloat64
	for _, snap := range snapshots {
		// Batch capacity: how many new requests can be pulled from queue per step
		// Approximate using BatchSize as current running count.
		// If BatchSize is low, many slots available → short wait.
		batchCapacity := 8.0 // conservative default: 8 requests per step
		if snap.BatchSize > 0 {
			// More requests running → fewer slots for new ones
			batchCapacity = math.Max(1.0, batchCapacity-float64(snap.BatchSize))
		}

		// Queue wait: how many step cycles until this request gets scheduled
		stepCycles := math.Ceil(float64(snap.QueueDepth) / batchCapacity)
		queueWait := stepCycles * p.config.AvgStepTime

		// Prefill time: depends on cache match at this instance
		cacheMissTokens := totalTokens
		if p.prefixIdx != nil && len(blockHashes) > 0 {
			cacheMatch := p.prefixIdx.MatchLength(blockHashes, snap.ID)
			cacheMissTokens = totalTokens - cacheMatch*p.prefixIdx.BlockSize()
			if cacheMissTokens < 0 {
				cacheMissTokens = 0
			}
		}
		prefillTime := p.config.Beta0 + p.config.Beta1*float64(cacheMissTokens)

		estimatedTTFT := queueWait + prefillTime
		if estimatedTTFT < bestTTFT {
			bestTTFT = estimatedTTFT
		}
	}

	// Get SLO budget for this request's class
	budget := p.budgetFor(req.SLOClass)

	// R11: guard against zero/NaN budget
	if budget <= 0 || math.IsNaN(budget) || math.IsInf(budget, 0) {
		return true, "budget-undefined"
	}

	// Admission decision
	threshold := budget * p.config.Headroom
	if bestTTFT <= threshold {
		return true, fmt.Sprintf("predictive-admit[%s](est=%.0fμs<=%.0fμs)",
			req.SLOClass, bestTTFT, threshold)
	}

	return false, fmt.Sprintf("predictive-reject[%s](est=%.0fμs>%.0fμs)",
		req.SLOClass, bestTTFT, threshold)
}

// budgetFor returns the TTFT budget (μs) for a given SLO class.
func (p *PredictiveSLOAdmission) budgetFor(sloClass string) float64 {
	switch sloClass {
	case "critical":
		return p.config.BudgetCritical
	case "standard", "":
		return p.config.BudgetStandard
	case "sheddable":
		return p.config.BudgetSheddable
	case "batch":
		return 5000000 // 5 seconds
	case "background":
		return math.MaxFloat64 // no budget
	default:
		return p.config.BudgetStandard
	}
}
