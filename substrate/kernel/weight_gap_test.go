package latencykernel

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// The resident weights a rank holds, as the kernel plans them from a catalog graph, against the
// bytes vLLM v0.33.0 allocates for the checkpoint at the same width (inference-sim/blis-substrate
// #254, #255, #107). The expected side is computed here from the checkpoint's shapes and stored
// widths -- its config.json and safetensors headers at the catalog's pinned revision -- and the
// rule vLLM applies to each module: which modules its quantization method builds unquantized,
// which it splits over the tensor-parallel ranks and which it replicates.
//
// Left out on both sides, each named: the MoE router (a node of its own with #228, removed from
// the graph here so the comparison does not depend on that change); the multimodal tower, which
// these graphs do not hold (#107 item 2); the expert biases gpt-oss stores, which the schema has
// no field for (0.074 GiB). The part is a B200, on which the
// kernel adds no expert padding (vLLM's MXFP4 round-up to 256 hidden and 128 intermediate,
// fused_moe/oracle/mxfp4.py:837-879, is engine lowering and is not counted here either).

const gib = 1 << 30

// ckptModule is one module of a checkpoint as vLLM builds it on one rank.
type ckptModule struct {
	params     float64 // elements of its weight, all layers
	bytesPer   float64 // resident bytes per element, scales included
	replicated bool    // whole on every rank (otherwise split over the tensor-parallel ranks)
}

func perRank(mods []ckptModule, tp int) float64 {
	var b float64
	for _, m := range mods {
		if m.replicated {
			b += m.params * m.bytesPer
		} else {
			b += m.params * m.bytesPer / float64(tp)
		}
	}
	return b
}

const (
	bf16 = 2.0
	fp8  = 1.0
	// fp8block is block-quantized FP8 with its float32 scale per 128 x 128 block (weight_scale_inv
	// F32, quantization_config.weight_block_size [128, 128]); the graph states the scale since #239.
	fp8block = fp8 + 4.0/(128*128)
	nvfp4    = 0.5 + 1.0/16 // U8 pairs and an F8_E4M3 scale per 16
	mxfp4    = 0.5 + 1.0/32 // U8 pairs and a U8 (E8M0) scale per 32
)

// gptOSS120B: openai/gpt-oss-120b@b5c939de. quant_method mxfp4 reaches the routed experts only
// (modules_to_not_convert names self_attn, the router, the embeddings and lm_head; Mxfp4Config
// gives every LinearBase UnquantizedLinearMethod), so attention and the head are BF16.
func gptOSS120B() []ckptModule {
	const L, H, E, I, V = 36, 2880, 128, 2880, 201088
	q, kv := 64*64, 8*64
	return []ckptModule{
		{L * float64(H*(q+2*kv)+q*H), bf16, false}, // qkv_proj and o_proj
		{L * E * 3 * float64(I) * H, mxfp4, false}, // experts gate_up [5760, 2880] and down
		{V * H, bf16, false}, {V * H, bf16, false}, // lm_head, embed_tokens (vocab-parallel)
	}
}

// kimiK25NVFP4: nvidia/Kimi-K2.5-NVFP4@0fd0a5e6. ModelOpt NVFP4 on the dense MLP, the routed and
// shared experts; the latent attention and the head are BF16 (excluded). The fused A-projection
// [q_lora + kv_lora + rope, hidden] is replicated (DeepSeekV2FusedQkvAProjLinear, disable_tp,
// deepseek_v2.py:953-968@4ba787ec); q_b, kv_b and o split.
func kimiK25NVFP4() []ckptModule {
	const L, H, V = 61, 7168, 163840
	const qLora, kvLora, rope, nope, vDim, heads = 1536, 512, 64, 128, 128, 64
	const E, I, dense = 384, 2048, 18432
	return []ckptModule{
		{L * float64(H*(qLora+kvLora+rope)), bf16, true},
		{L * float64(qLora*heads*(nope+rope)+kvLora*heads*(nope+vDim)+heads*vDim*H), bf16, false},
		{3 * float64(dense) * H, nvfp4, false},            // layer 0's MLP
		{60 * (E + 1) * 3 * float64(I) * H, nvfp4, false}, // routed experts and the shared one
		{V * H, bf16, false}, {V * H, bf16, false},
	}
}

// miniMaxM25NVFP4: nvidia/MiniMax-M2.5-NVFP4@b6220d65. NVFP4 routed experts; the attention is
// stored F32 and built at the model dtype (BF16), the head and embedding BF16.
func miniMaxM25NVFP4() []ckptModule {
	const L, H, E, I, V = 62, 3072, 256, 1536, 200064
	q, kv := 48*128, 8*128
	return []ckptModule{
		{L * float64(H*(q+2*kv)+q*H), bf16, false},
		{L * E * 3 * float64(I) * H, nvfp4, false},
		{V * H, bf16, false}, {V * H, bf16, false},
	}
}

// miniMaxM25: MiniMaxAI/MiniMax-M2.5@f710177d. FP8 block-quantized attention and experts, each
// with its float32 block scale;
// the head is BF16 in the header ([200064, 3072]), and Fp8Config builds ParallelLMHead
// unquantized (quantization/fp8.py:177-209@4ba787ec).
func miniMaxM25() []ckptModule {
	const L, H, E, I, V = 62, 3072, 256, 1536, 200064
	q, kv := 48*128, 8*128
	return []ckptModule{
		{L * float64(H*(q+2*kv)+q*H), fp8block, false},
		{L * E * 3 * float64(I) * H, fp8block, false},
		{V * H, bf16, false}, {V * H, bf16, false},
	}
}

// withoutRouterNodes is g without the router GEMMs the MoE change (#228) states.
func withoutRouterNodes(g *model.Graph) *model.Graph {
	c := *g
	c.LayerKinds = append([]model.LayerKind(nil), g.LayerKinds...)
	for i := range c.LayerKinds {
		var kept []model.Node
		for _, n := range g.LayerKinds[i].Nodes {
			if n.Role != "router" {
				kept = append(kept, n)
			}
		}
		c.LayerKinds[i].Nodes, c.LayerKinds[i].Edges = kept, nil
	}
	return &c
}

func TestPlannedResidentWeightsPerRankAreTheCheckpointsBytes(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		model  string
		served model.DType // what the lowering hands over for the deployment's quantization
		tps    []int
		ckpt   []ckptModule
	}{
		{"gpt-oss-120b", model.DTypeMXFP4, []int{1, 2, 4, 8}, gptOSS120B()},
		{"kimi-k2.5-nvfp4", model.DTypeNVFP4, []int{4, 8}, kimiK25NVFP4()},
		{"minimax-m2.5-nvfp4", model.DTypeNVFP4, []int{1, 2, 4, 8}, miniMaxM25NVFP4()},
		{"minimax-m2.5", model.DTypeFP8, []int{2, 4, 8}, miniMaxM25()},
	} {
		for _, tp := range tc.tps {
			t.Run(fmt.Sprintf("%s/tp%d", tc.model, tp), func(t *testing.T) {
				in := planCase(t, d, tc.model, "b200", 1, fmt.Sprintf("tp: %d, pp: 1, dp: 1", tp),
					func(l *plan.Lowered) { l.Plan.WeightFormat = string(tc.served) })
				in.Model = withoutRouterNodes(in.Model)
				got := float64(mustKernel(t, in).FixedBytes().Weights)
				want := perRank(tc.ckpt, tp)
				if math.Abs(got-want) > 1e-6*want {
					t.Errorf("a rank holds %.4f GiB of weights as planned; vLLM allocates %.4f GiB "+
						"for this checkpoint (off by %+.4f GiB)", got/gib, want/gib, (got-want)/gib)
				}
			})
		}
	}
}
