// Original to blis-vllm.

package lowering

import (
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// routedExpertFormat names the format of the stack's routed experts for the backend census below:
// "fp8_block", "bf16", "nvfp4", or the node's own dtype word for any other. It is "" when the stack
// has no routed experts, and "mixed" when two routed nodes differ.
func routedExpertFormat(g *model.Graph) string {
	found := ""
	for _, lk := range stackKinds(g.LayerKinds, g.Stack) {
		for _, n := range lk.Nodes {
			if n.Op != model.OpGroupedGEMM || n.TopK <= 0 || n.IsSharedExpert() {
				continue
			}
			d := n.WeightDType
			if d == "" {
				d = g.Global.WeightDType
			}
			word := string(d)
			switch {
			case d == model.DTypeFP8 && (g.Global.WeightScale == nil || g.Global.WeightScale.Elements != 32):
				word = "fp8_block"
			case d == model.DTypeBF16 || d == model.DTypeFP16:
				word = "bf16"
			}
			if found != "" && found != word {
				return "mixed"
			}
			found = word
		}
	}
	return found
}

// modularMoEBackends are the stated moe_backend words whose experts classes are all modular for the
// block-FP8, BF16 and NVFP4 weights routingMode resolves: Triton, DeepGEMM, vLLM's CUTLASS,
// FlashInfer's CUTLASS, Marlin and Humming
// (fused_moe/oracle/fp8.py backend_to_kernel_cls, :147-262@4ba787ec). "triton" is not in the set
// for MXFP4, where it selects the monolithic OpenAI Triton experts (oracle/mxfp4.py:247-253).
var modularMoEBackends = map[string]bool{
	"triton": true, "deep_gemm": true, "cutlass": true, "flashinfer_cutlass": true,
	"marlin": true, "humming": true,
}

// routingMode says where the routed MoE layers' expert selection runs, as the experts class vLLM
// v0.33.0 resolves decides it: MoERunner._apply_quant_method passes router_logits into a monolithic
// experts kernel and runs router.select_experts as its own launch for a modular one
// (fused_moe/runner/moe_runner.py:620-643@4ba787ec). The reason is the note to record when the
// answer rests on an assumption or is not resolved; it is "" when the resolution is certain.
//
//   - Hopper (SM90): the TRT-LLM and mxint4 experts require a Blackwell-family part
//     (trtllm_fp8_moe.py:165-172, trtllm_bf16_moe.py:95-102, trtllm_nvfp4_moe.py:204-211,
//     trtllm_mxint4_moe.py:45-55), so block-FP8 (Triton at expert-parallel width 1, FlashInfer's
//     CUTLASS above it, oracle/fp8.py:101-130), BF16 and NVFP4 experts, and an INT4 checkpoint's
//     Marlin experts, are all modular: separate.
//   - Blackwell (SM100) with block-FP8, BF16 or NVFP4 experts and the engine's own choice or
//     flashinfer_trtllm: the TRT-LLM class is first in each oracle's order, and its monolithic
//     variant is preferred over the modular one when it supports the model's routing method
//     (backend_to_kernel_cls returns [Monolithic, Modular]; modular_kernel.py:600-612). The
//     monolithic kernels support DeepSeekV3, Renormalize, Sigmoid and MiniMax2 routing
//     (trtllm_fp8_moe.py:408-420, trtllm_bf16_moe.py:263-271, trtllm_nvfp4_moe.py:500-508), the
//     routing every catalog MoE family uses; that is stated as an assumption: fused.
//   - A stated modular backend: separate.
//   - Anything else (MXFP4, MXFP8, INT4 on Blackwell, a part of another family, a stated backend
//     outside both sets): not stated, with the reason.
func routingMode(g *model.Graph, c *lowerContext, stated string) (plan.MoERouting, string) {
	format := routedExpertFormat(g)
	if format == "" {
		return plan.RoutingNotStated, ""
	}
	backend := strings.ReplaceAll(strings.ToLower(stated), "-", "_")
	if backend == "auto" {
		backend = ""
	}
	engineChoice := backend == "" || backend == "flashinfer_trtllm"
	resolvable := format == "fp8_block" || format == "bf16" || format == "nvfp4"
	switch {
	case c.family == 9 && backend != "flashinfer_trtllm" && (backend == "" || modularMoEBackends[backend]) &&
		(resolvable || format == string(model.DTypeINT4)):
		return plan.RoutingSeparate, ""
	case c.family == 10 && resolvable && modularMoEBackends[backend]:
		return plan.RoutingSeparate, ""
	case c.family == 10 && resolvable && engineChoice:
		return plan.RoutingFused, "the TRT-LLM monolithic experts are first in the oracle's order and take " +
			"router logits; the model's routing method is taken to be one they support " +
			"(DeepSeekV3, Renormalize, Sigmoid or MiniMax2), as every catalog MoE family's is"
	}
	return plan.RoutingNotStated, "the experts class vLLM resolves for " + format + " routed experts" +
		" (moe_backend " + strconvQuote(stated) + ") is not resolved here, so no selection launch is counted"
}

func strconvQuote(s string) string { return `"` + s + `"` }
