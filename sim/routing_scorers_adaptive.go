package sim

import (
	"math"
)

// newCostBenefitScorer creates a scorer that internalizes the cache-vs-load tradeoff.
//
// Unlike the independent PA and QD scorers, this scorer computes a NONLINEAR function
// of both cache match AND queue depth:
//
//	score = cache_saving / (cache_saving + queue_delay)
//
// Where:
//   - cache_saving = estimated prefill time reduction from prefix cache hit (in ticks)
//   - queue_delay = estimated additional queueing cost on this instance vs least-loaded
//
// This naturally adapts to system load: at low load (queue_delay ≈ 0), the score is ~1.0
// for any cache hit (exploit freely). At high load (queue_delay >> cache_saving), the
// score approaches 0 (ignore cache, distribute). The CROSSOVER POINT depends on the
// actual magnitudes, not on a static weight ratio.
//
// The beta coefficients are used to estimate prefill cost savings from cache hits.
// With beta3 (quadratic attention), cache savings are much larger for long prefixes,
// making this scorer dramatically more effective than the linear PA scorer.
//
// Signal freshness (R17, INV-7):
//
//	Reads: PrefixCacheIndex (Tier 1: synchronous), EffectiveLoad (Tier 1+2 composite).
//	Uses beta coefficients captured at construction (constant).
func newCostBenefitScorer(prefixIdx *PrefixCacheIndex, betaCoeffs []float64) scorerFunc {
	beta1 := 0.0
	if len(betaCoeffs) > 1 {
		beta1 = betaCoeffs[1]
	}
	beta3 := 0.0
	if len(betaCoeffs) > 3 {
		beta3 = betaCoeffs[3]
	}

	return func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))

		if prefixIdx == nil || req == nil || len(req.InputTokens) == 0 {
			// No cache info → score 0.5 for all (neutral)
			for _, snap := range snapshots {
				scores[snap.ID] = 0.5
			}
			return scores
		}

		hashes := prefixIdx.ComputeBlockHashes(req.InputTokens)
		totalBlocks := len(hashes)
		if totalBlocks == 0 {
			for _, snap := range snapshots {
				scores[snap.ID] = 0.5
			}
			return scores
		}

		blockSize := prefixIdx.BlockSize()
		totalTokens := len(req.InputTokens)

		// Find least-loaded instance for queue delay baseline
		minLoad := math.MaxInt
		for _, snap := range snapshots {
			if snap.EffectiveLoad() < minLoad {
				minLoad = snap.EffectiveLoad()
			}
		}

		// Estimate average service time (for queue delay estimation)
		// StepTime ≈ beta0 + beta1*totalTokens + beta3*totalTokens^2
		beta0 := 0.0
		if len(betaCoeffs) > 0 {
			beta0 = betaCoeffs[0]
		}
		beta2 := 0.0
		if len(betaCoeffs) > 2 {
			beta2 = betaCoeffs[2]
		}
		avgOutputTokens := 64.0 // reasonable estimate for mixed workloads
		svcTime := beta0 + beta1*float64(totalTokens) + beta3*float64(totalTokens)*float64(totalTokens) +
			avgOutputTokens*(beta0+beta2)

		for _, snap := range snapshots {
			matched := prefixIdx.MatchLength(hashes, snap.ID)
			cachedTokens := matched * blockSize
			newTokens := totalTokens - cachedTokens
			if newTokens < 0 {
				newTokens = 0
			}

			// Estimate cache saving: prefill cost with all tokens - prefill cost with only new tokens
			missTime := beta1*float64(totalTokens) + beta3*float64(totalTokens)*float64(totalTokens)
			hitTime := beta1*float64(newTokens) + beta3*float64(newTokens)*float64(totalTokens)
			cacheSaving := missTime - hitTime
			if cacheSaving < 0 {
				cacheSaving = 0
			}

			// Estimate queue delay: extra load relative to least-loaded × service time
			extraLoad := float64(snap.EffectiveLoad() - minLoad)
			queueDelay := extraLoad * svcTime

			// Cost-benefit ratio: cache saving / (cache saving + queue delay)
			denominator := cacheSaving + queueDelay
			if denominator > 0 {
				scores[snap.ID] = cacheSaving / denominator
			} else {
				scores[snap.ID] = 0.5 // neutral when no cache and no load difference
			}
		}

		return scores
	}
}

// newSLOHeadroomScorer creates a scorer that evaluates each instance based on
// whether the request can meet its SLO budget given the instance's current load.
//
// For each instance, it estimates the expected TTFT and compares to the SLO budget:
//
//	score = clamp(1 - estimated_ttft / slo_budget, 0, 1)
//
// Instances where TTFT would exceed the budget get score 0. Instances with plenty
// of headroom get score close to 1. This scorer is fundamentally REQUEST-DEPENDENT:
// the same instance gets different scores for critical vs batch requests because
// their budgets differ.
//
// This is the scorer that makes dynamic weighting structurally necessary — its
// output depends on request metadata (SLO class), not just system state.
// A static weight on this scorer applies uniformly, but different SLO classes
// need different weights: critical requests want this scorer weighted heavily
// (penalize any risk of SLO miss), batch requests want it weighted lightly
// (they have large budgets, focus on cache instead).
//
// Signal freshness (R17, INV-7):
//
//	Reads: EffectiveLoad (Tier 1+2 composite). SLO budgets captured at construction.
func newSLOHeadroomScorer(betaCoeffs []float64) scorerFunc {
	// SLO TTFT budgets in microseconds (ticks)
	sloBudgets := map[string]float64{
		"critical":   50000,   // 50ms TTFT budget
		"standard":   200000,  // 200ms
		"sheddable":  500000,  // 500ms
		"batch":      5000000, // 5s
		"background": math.MaxFloat64,
	}

	beta0 := 0.0
	if len(betaCoeffs) > 0 {
		beta0 = betaCoeffs[0]
	}
	beta1 := 0.0
	if len(betaCoeffs) > 1 {
		beta1 = betaCoeffs[1]
	}
	beta3 := 0.0
	if len(betaCoeffs) > 3 {
		beta3 = betaCoeffs[3]
	}

	return func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))

		sloClass := ""
		inputLen := 0
		if req != nil {
			sloClass = req.SLOClass
			inputLen = len(req.InputTokens)
		}

		budget, ok := sloBudgets[sloClass]
		if !ok {
			budget = sloBudgets["standard"] // default
		}

		// Estimate prefill time for this request
		prefillTime := beta0 + beta1*float64(inputLen) + beta3*float64(inputLen)*float64(inputLen)

		for _, snap := range snapshots {
			// Estimated TTFT = queue_delay + prefill_time
			// queue_delay ≈ effective_load × avg_step_time
			avgStepTime := beta0 + beta1*256 + beta3*256*256 // rough average
			estimatedTTFT := float64(snap.EffectiveLoad())*avgStepTime + prefillTime

			// Headroom: how much budget remains
			headroom := 1.0 - estimatedTTFT/budget
			if headroom < 0 {
				headroom = 0
			}
			if headroom > 1 {
				headroom = 1
			}
			scores[snap.ID] = headroom
		}

		return scores
	}
}
