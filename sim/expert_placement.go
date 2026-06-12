package sim

// ExpertPlacement maps a step's routed-token population onto per-GPU MoE cost,
// returning the load of the BUSIEST GPU in the flattened MoE group. A collective
// runs at its slowest participant, so step time is "max over GPUs" — returning
// the busiest GPU lets that physics emerge automatically once a future strategy
// (imbalanceFactor >= 1: EPLB, skewed routing, redundant experts) introduces
// load skew. It is a single-method, pure-query contract (R13/R14): it observes
// the routed-token population and parallelism degrees and computes a cost; it
// owns no state and mutates nothing.
//
// Lives in sim core, not sim/latency/: expert placement is a model-deployment
// concept (same domain as ModelConfig and the parallelism degrees) that future
// consumers — scheduler, KV-capacity — must reach without importing latency/,
// which would invert the dependency direction cmd/ -> sim/cluster/ -> sim/. The
// latency model already imports sim, so it consumes sim.ExpertPlacement for free.
//
// Parameters:
//   - globalTokens: routed tokens summed over the whole step (all DP ranks).
//   - kEff:         effective experts activated per token (top_k, possibly fractional).
//   - numExperts:   total routed experts in the layer.
//   - moeGroupSize: the flattened MoE group the experts are sharded over (TP·DP
//     for MoE, mirroring vLLM's flattened dp·pcp·tp group; 1 = single-GPU MoE).
//   - dp:           data-parallel degree. Each DP rank owns a disjoint ~globalTokens/dp
//     slice of the sequence tokens, so per-rank communication volume scales by 1/dp.
type ExpertPlacement interface {
	Resolve(globalTokens, kEff float64, numExperts, moeGroupSize, dp int) ExpertLoad
}

// ExpertLoad is the per-GPU MoE cost of the busiest GPU for one step, in the
// raw units the latency model's basis functions expect (coefficients are applied
// by the consumer, not here).
type ExpertLoad struct {
	// PerGPUComputeTokens is the token·activation count the max-loaded GPU computes
	// in the MoE FFN (drives compute-bound FLOPs).
	PerGPUComputeTokens float64
	// PerGPUExpertCount is the full-expert-equivalent count resident on the GPU,
	// driving routed-expert weight bytes. EP-mode-agnostic: numExperts/moeGroupSize
	// equals both EP-off tensor-sharded full-expert-equivalent bytes and EP-on whole
	// experts owned per GPU (see issue #1418 / design §4 vLLM proofs).
	PerGPUExpertCount float64
	// PerGPUCommTokens is the dispatch+combine all-to-all volume (token·top_k) the
	// busiest source GPU moves. Zero when moeGroupSize == 1 (nothing to exchange).
	PerGPUCommTokens float64
}

// BalancedPlacement is the default ExpertPlacement: it assumes routed tokens and
// experts are spread perfectly evenly across the MoE group (imbalanceFactor == 1).
// At the saturation operating point this latency model targets, the balanced-load
// assumption is exact, so the busiest GPU carries exactly the average share.
//
//	PerGPUComputeTokens = globalTokens · kEff / moeGroupSize
//	PerGPUExpertCount   = numExperts / moeGroupSize
//	PerGPUCommTokens    = (globalTokens / dp) · kEff · (moeGroupSize-1)/moeGroupSize · 2
//
// PerGPUCommTokens is divided by dp because the term is a per-rank latency paid by
// the busiest source GPU, and each DP rank owns only ~globalTokens/dp sequence
// tokens — not an aggregate cluster byte count. The (moeGroupSize-1)/moeGroupSize·2
// factor is the standard dispatch+combine all-to-all volume over the flattened
// group: a GPU sends to the other moeGroupSize-1 members on dispatch and receives
// back on combine.
type BalancedPlacement struct{}

// Resolve computes the balanced per-GPU MoE load. moeGroupSize and dp are assumed
// >= 1 (guaranteed by ModelHardwareConfig.EffectiveMoEGroupSize / EffectiveDP,
// which clamp to a minimum of 1); the formulas are well-defined and the comm term
// is exactly 0 at the degenerate moeGroupSize == 1.
func (BalancedPlacement) Resolve(globalTokens, kEff float64, numExperts, moeGroupSize, dp int) ExpertLoad {
	group := float64(moeGroupSize)
	return ExpertLoad{
		PerGPUComputeTokens: globalTokens * kEff / group,
		PerGPUExpertCount:   float64(numExperts) / group,
		PerGPUCommTokens:    (globalTokens / float64(dp)) * kEff * (group - 1) / group * 2,
	}
}
