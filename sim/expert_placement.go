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
//   - moeGroupSize: the group the requested cost is divided across (see below).
//   - dp:           data-parallel degree. Each DP rank owns a disjoint ~globalTokens/dp
//     slice of the sequence tokens, so per-rank communication volume scales by 1/dp.
//
// moeGroupSize is NOT one fixed quantity per model. Since #1548 the caller passes a
// different group depending on which field of the result it consumes, because the groups
// genuinely differ under expert parallelism (ModelHardwareConfig.EffectiveMoEGroupSize vs
// EffectiveExpertShardGroupSize). BLIS assumes PCP=1, so the compute group is TP·DP for MoE
// (vLLM's flattened group is dp·pcp·tp); 1 = single-GPU MoE. Concretely, the trained-physics
// consumer passes: the flattened TP·DP group for PerGPUComputeTokens; the expert-OWNING group
// (the EP group when expert parallelism is on, else TP·DP), clamped to the routed-expert
// count, for PerGPUExpertCount; and that same expert-owning group UNclamped for
// PerGPUCommTokens.
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
	// driving routed-expert weight bytes: numExperts/moeGroupSize.
	//
	// The FORMULA is EP-mode-agnostic — it expresses both EP-off tensor-sharded
	// full-expert-equivalent bytes and EP-on whole experts owned per GPU (issue #1418 /
	// design §4 vLLM proofs) — but the GROUP the caller passes is not. Since #1548 the
	// two modes can shard over different groups (a DP-as-placement replica's own TP·DP
	// understates its logical EP group), so the caller supplies the expert-OWNING group
	// here, not the compute group.
	//
	// Two bounds differ by mode and are the caller's responsibility, not this
	// function's: under EP-on the group is clamped to numExperts (a loaded rank holds one
	// WHOLE expert, so a sub-1 result would model memory that does not exist), while
	// under EP-off a result below 1 is CORRECT — a rank holds a fraction of every
	// expert. See latency.ClampExpertShardToExpertCount.
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
// PerGPUCommTokens models the dispatch/combine path — reached at DP>1 and, since #1548,
// whenever expert parallelism is on (EP-on owns whole experts per rank, so tokens must be
// routed to their owner even at DP=1). It deliberately carries a kEff factor (top-k
// routed-token volume per source rank) as the balanced-load approximation of that
// all-to-all; the consumer scales it by hidden·bytes-per-param and must NOT re-multiply by
// kEff (design §5).
//
// It is NOT the EP-OFF DP=1 collective: there vLLM runs a tensor-parallel all-reduce on the
// dense output hidden states ([tokens, hidden], no top-k factor) — a different volume the
// trained-physics model charges under a separate reduction term. The two are not numerically
// equal and must not be conflated. The group passed here is the expert-owning group
// UNCLAMPED by the expert count: the all-to-all genuinely spans every rank in the group,
// however few experts they hold.
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
