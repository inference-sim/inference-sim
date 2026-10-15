package sim

// ModelConfig carries the model facts the simulator itself acts on: the routed-expert
// geometry, which decides whether a deployment is mixture-of-experts. Everything a step's
// price depends on -- layer shapes, precisions, attention kind -- is read by the latency
// backend (blis-latency-kernel) from the model graph in blis-catalog; the simulator never
// sees it, so it is not carried here. The CLI fills these counts from the same graph.
type ModelConfig struct {
	NumLocalExperts  int // routed experts per MoE layer; 0 = dense model
	NumExpertsPerTok int // experts activated per token; 0 = dense model
}

// MoEMinExperts is the minimum NumLocalExperts for a model to be treated as MoE.
// It is the single source of truth for the MoE-vs-dense boundary across BLIS.
//
// Single-expert configs (NumLocalExperts == 1) are dense-equivalent in BLIS. This is an
// intentional, documented divergence from vLLM, whose is_moe is get_num_experts() > 0; on
// every real model the two thresholds agree — no real model has exactly one routed expert.
const MoEMinExperts = 2

// IsMoE reports whether the model is a mixture-of-experts model
// (NumLocalExperts >= MoEMinExperts). See MoEMinExperts for the threshold rationale
// and the vLLM divergence note. This is the canonical MoE-detection predicate:
// prefer it over inline NumLocalExperts comparisons at detection sites.
func (mc ModelConfig) IsMoE() bool {
	return mc.NumLocalExperts >= MoEMinExperts
}
