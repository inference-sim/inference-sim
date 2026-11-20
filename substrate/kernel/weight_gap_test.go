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
// the graph here so the comparison does not depend on that change, and its 36 x 128 bias
// elements); the multimodal tower, which these graphs do not hold (#107 item 2). The MXFP4 experts are allocated at
// vLLM's padded shapes (#273): hidden up to a multiple of 256 and each rank's intermediate up to
// 128 (fused_moe/oracle/mxfp4.py:837-879@v0.33.0), on Hopper and Blackwell alike.

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
func gptOSS120B(tp int) []ckptModule {
	const L, H, E, I, V = 36, 2880, 128, 2880, 201088
	q, kv := 64*64, 8*64
	up := func(x, m int) int { return (x + m - 1) / m * m }
	// w13 [E, 2 x I/tp, H] and w2 [E, H, I/tp] on a rank, at the padded widths; stated as the
	// whole layer (x tp) so perRank's split gives one rank's share.
	expertParams := float64(E*3*up(I/tp, 128)*up(H, 256)) * float64(tp)
	return []ckptModule{
		{L * float64(H*(q+2*kv)+q*H), bf16, false}, // qkv_proj and o_proj
		{L * expertParams, mxfp4, false},           // experts gate_up and down
		{V * H, bf16, false}, {V * H, bf16, false}, // lm_head, embed_tokens (vocab-parallel)
		// The biases the checkpoint stores (#281), at the unpadded shapes it stores them in: the
		// attention projections' (qkv split; o_proj row-parallel, whole on every rank, kept
		// split by the graph for its 0.2 MB), the experts' gate_up (split) and down (whole).
		{L * float64(q+2*kv), bf16, false}, {L * float64(H), bf16, false},
		{L * float64(E*2*I), bf16, false}, {L * float64(E*H), bf16, true},
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

// fixed is a checkpoint whose modules do not depend on the tensor-parallel width.
func fixed(m []ckptModule) func(int) []ckptModule { return func(int) []ckptModule { return m } }

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
		chip   string
		tps    []int
		ckpt   func(tp int) []ckptModule
	}{
		{"gpt-oss-120b", model.DTypeMXFP4, "b200", []int{1, 2, 4, 8}, gptOSS120B},
		{"gpt-oss-120b", model.DTypeMXFP4, "h200", []int{1, 2, 4, 8}, gptOSS120B},
		{"kimi-k2.5-nvfp4", model.DTypeNVFP4, "b200", []int{4, 8}, fixed(kimiK25NVFP4())},
		{"minimax-m2.5-nvfp4", model.DTypeNVFP4, "b200", []int{1, 2, 4, 8}, fixed(miniMaxM25NVFP4())},
		{"minimax-m2.5", model.DTypeFP8, "b200", []int{2, 4, 8}, fixed(miniMaxM25())},
	} {
		for _, tp := range tc.tps {
			t.Run(fmt.Sprintf("%s/%s/tp%d", tc.model, tc.chip, tp), func(t *testing.T) {
				in := planCase(t, d, tc.model, tc.chip, 1, fmt.Sprintf("tp: %d, pp: 1, dp: 1", tp),
					func(l *plan.Lowered) { l.Plan.WeightFormat = string(tc.served) })
				in.Model = withoutRouterNodes(in.Model)
				got := float64(mustKernel(t, in).FixedBytes().Weights)
				want := perRank(tc.ckpt(tp), tp)
				if math.Abs(got-want) > 1e-6*want {
					t.Errorf("a rank holds %.4f GiB of weights as planned; vLLM allocates %.4f GiB "+
						"for this checkpoint (off by %+.4f GiB)", got/gib, want/gib, (got-want)/gib)
				}
			})
		}
	}
}

// vLLM builds ParallelLMHead unquantized for an fp8 checkpoint (Fp8Config.get_quant_method
// returns None for it, quantization/fp8.py:177-209@4ba787ec), and the GLM DSA checkpoints'
// headers hold lm_head.weight BF16. A rank must therefore hold the head at two bytes a
// weight, one byte more than the same head at the global fp8 width would be.
func TestFP8GlmHeadIsPlannedAtTheCheckpointsBF16Width(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"glm-5", "glm-5.1-fp8", "glm-5.2-fp8", "glm-5.3"} {
		for _, tp := range []int{4, 8} {
			t.Run(fmt.Sprintf("%s/tp%d", name, tp), func(t *testing.T) {
				in := planCase(t, d, name, "b200", 1, fmt.Sprintf("tp: %d, pp: 1, dp: 1", tp),
					func(l *plan.Lowered) { l.Plan.WeightFormat = string(model.DTypeFP8) })
				asPlanned := float64(mustKernel(t, in).FixedBytes().Weights)
				at := -1
				for i, n := range in.Model.Head {
					if n.Role == "lm_head" {
						at = i
					}
				}
				if at < 0 {
					t.Fatal("the graph states no lm_head")
				}
				h := in.Model.Head[at]
				// The same graph with the head at the global fp8 width.
				g := *in.Model
				g.Head = append([]model.Node(nil), in.Model.Head...)
				g.Head[at].WeightDType = model.DTypeFP8
				atFP8 := in
				atFP8.Model = &g
				fp8Planned := float64(mustKernel(t, atFP8).FixedBytes().Weights)
				// One extra byte per head weight, split over the tensor-parallel group.
				want := float64(h.N) * float64(h.K) / float64(tp)
				if got := asPlanned - fp8Planned; math.Abs(got-want) > 1e-3*want {
					t.Errorf("the head holds %+.4f GiB a rank beyond its fp8 width; BF16 is %+.4f GiB",
						got/gib, want/gib)
				}
			})
		}
	}
}

// A checkpoint's bias is resident beside its weights. gpt-oss-120b stores a bias on the attention
// projections, the router and every routed expert's gate_up and down projections (bf16); a rank
// holds them, so the planned weights are the same graph without the biases plus their bytes.
func TestGptOssBiasesAreResidentWithTheWeights(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, tp := range []int{1, 2, 4, 8} {
		t.Run(fmt.Sprintf("tp%d", tp), func(t *testing.T) {
			in := planCase(t, d, "gpt-oss-120b", "b200", 1, fmt.Sprintf("tp: %d, pp: 1, dp: 1", tp),
				func(l *plan.Lowered) { l.Plan.WeightFormat = string(model.DTypeMXFP4) })
			withBias := float64(mustKernel(t, in).FixedBytes().Weights)
			g := *in.Model
			g.LayerKinds = append([]model.LayerKind(nil), in.Model.LayerKinds...)
			var want float64
			count := map[string]int{}
			for _, id := range in.Model.Stack.Expand() {
				count[id]++
			}
			for i, k := range g.LayerKinds {
				g.LayerKinds[i].Nodes = append([]model.Node(nil), k.Nodes...)
				for j, n := range g.LayerKinds[i].Nodes {
					if n.BiasDType == "" {
						continue
					}
					elems := float64(n.N)
					if n.Op == model.OpGroupedGEMM {
						// gate and up split over the ranks; the down projection's bias is whole on each.
						elems = float64(n.Experts) * (float64(2*n.N)/float64(tp) + float64(n.K))
					} else if n.Role != "router" && n.TP != model.TPRow {
						elems /= float64(tp)
					}
					want += float64(count[k.ID]) * elems * 2 // bf16
					g.LayerKinds[i].Nodes[j].BiasDType = ""
				}
			}
			if want == 0 {
				t.Fatal("the graph states no bias")
			}
			noBias := in
			noBias.Model = &g
			without := float64(mustKernel(t, noBias).FixedBytes().Weights)
			if got := withBias - without; math.Abs(got-want) > 0.02*want {
				t.Errorf("biases add %.4f GiB a rank; the checkpoint's are %.4f GiB", got/gib, want/gib)
			}
		})
	}
}
