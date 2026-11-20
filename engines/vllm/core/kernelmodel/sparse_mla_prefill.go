// Original to blis-vllm.

package kernelmodel

import (
	"strconv"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	sim "github.com/inference-sim/blis-vllm/core/runconfig"
)

// sparseMLAPrefillSubstitution names where the pool's sparse-MLA prefill is priced from a family it
// does not hold in every step. vLLM runs a prefill whose longest sequence (computed plus scheduled
// tokens) is at most the layer's top-k (index_topk, 2048 for GLM-5.3-Flash) as dense multi-head
// attention, not as forward_mqa, unless sparse_mla_force_mqa or HiSparse is set
// (_use_dense_mha_prefill, vllm/model_executor/layers/attention/sparse_mla_attention.py:136-144 and
// :408-413; _use_sparse_mha, mla_attention.py:1215-1240; FlashInferMLASparseSM90Impl plans only the decode
// prefix when a prefill takes that route, flashinfer_mla_sparse_sm90.py:535-545, all at 4ba787ec).
// The masked-MHA alternative is SM100 only, so on an SM90 part the split is exactly the top-k.
//
// A plan that states the sparse-MLA kernel for the prefill primitive prices every prefill step from the
// family keyed to that kernel (VAL-8), which the registry fits on prefills whose prefix plus new tokens
// exceed the top-k: at or below it the form over-prices the measured module (signed 1.171 over the
// registry's 84 such rows). The pool says so. The decode primitive has no such split: forward_mqa
// serves every decode.
func sparseMLAPrefillSubstitution(g *model.Graph, p plan.PoolPlan) *sim.Substitution {
	stated := false
	for _, l := range p.Layers {
		if l.Primitive == plan.PrimAttentionPrefill && l.KernelImpl.Variant == "BatchMLAPagedAttentionWrapper[fa3]" {
			stated = true
			break
		}
	}
	if !stated || g == nil {
		return nil
	}
	topk := 0
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if n.AttentionKind == model.AttentionSparseMLA && n.ReadsEngineKV() {
				topk = max(topk, n.IndexTopK)
			}
		}
	}
	if topk <= 0 {
		return nil
	}
	n := strconv.Itoa(topk)
	return &sim.Substitution{Setting: "sparse_mla_prefill_path",
		Stated: "unset (sparse_mla_force_mqa and HiSparse are not stated)",
		Priced: "the sparse-MLA kernel's family for every prefill step",
		Reason: "vLLM runs a prefill whose longest sequence is at most " + n + " tokens (the top-k) as dense multi-head " +
			"attention, not as the sparse-MLA kernel (sparse_mla_attention.py _use_dense_mha_prefill at 4ba787ec); the pool " +
			"prices every prefill step from the family fitted above that length, which over-prices a short prefill " +
			"(signed 1.171 over the registry's rows at or below it). Decode is always the sparse-MLA kernel"}
}
