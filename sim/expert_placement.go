package sim

import "fmt"

// ExpertPlacement maps a step's routed-token population onto per-GPU MoE cost,
// returning the load of the BUSIEST GPU in the flattened MoE group. A collective
// runs at its slowest participant, so step time is "max over GPUs" — returning
// the busiest GPU lets that physics emerge automatically once a future load-skew
// strategy (EPLB, skewed routing, redundant experts) replaces BalancedPlacement.
// It is a single-method, pure-query contract (R13/R14): it observes
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
//   - moeGroupSize: the flattened MoE group the experts are sharded over. BLIS
//     assumes PCP=1, so this is TP·DP for MoE (vLLM's flattened group is dp·pcp·tp;
//     see ModelHardwareConfig.EffectiveMoEGroupSize); 1 = single-GPU MoE.
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
// experts are spread perfectly evenly across the MoE group (no load skew). At the
// saturation operating point this latency model targets, the balanced-load
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
//
// PerGPUCommTokens models the DP>1 dispatch/combine path only. It deliberately
// carries a kEff factor (top-k routed-token volume per source rank) as the
// balanced-load approximation of that all-to-all; the consumer scales it by
// hidden·bytes-per-param and must NOT re-multiply by kEff (design §5). It is NOT
// the DP=1 collective: at DP=1 vLLM runs a tensor-parallel all-reduce on the dense
// output hidden states ([tokens, hidden], no top-k factor) — a different volume the
// trained-physics model charges under a separate reduction term. The two are not
// numerically equal and must not be conflated.
type BalancedPlacement struct{}

// Resolve computes the balanced per-GPU MoE load. moeGroupSize and dp must be
// >= 1 — production callers get this for free from ModelHardwareConfig.
// EffectiveMoEGroupSize / EffectiveDP, which clamp to a minimum of 1. A direct
// caller that passes 0 (or negative) would otherwise silently emit +Inf/NaN loads
// into downstream step-time math, so this panics at the library boundary instead
// (R1: no silent bad output). With both >= 1 the formulas are well-defined and the
// comm term is exactly 0 at the degenerate moeGroupSize == 1.
func (BalancedPlacement) Resolve(globalTokens, kEff float64, numExperts, moeGroupSize, dp int) ExpertLoad {
	if moeGroupSize < 1 {
		panic(fmt.Sprintf("BalancedPlacement.Resolve: moeGroupSize must be >= 1, got %d", moeGroupSize))
	}
	if dp < 1 {
		panic(fmt.Sprintf("BalancedPlacement.Resolve: dp must be >= 1, got %d", dp))
	}
	group := float64(moeGroupSize)
	return ExpertLoad{
		PerGPUComputeTokens: globalTokens * kEff / group,
		PerGPUExpertCount:   float64(numExperts) / group,
		PerGPUCommTokens:    (globalTokens / float64(dp)) * kEff * (group - 1) / group * 2,
	}
}
