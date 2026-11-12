package price

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// A Plan is a model graph collapsed for repeated evaluation.
//
// Step time is the hot path: a discrete-event simulator calls it once per simulated step,
// which is millions of times over a run. So the graph is flattened once, at construction,
// into the smallest form that still prices correctly — and after that a step costs
// arithmetic over a short slice with no map lookups, no string comparisons and no
// allocation.
//
// Three things make that possible.
//
// The stack collapses to a per-kind layer COUNT rather than a layer list. A 108-layer
// hybrid has three distinct kinds, so a step walks three entries and multiplies, not 108.
// Across the twenty-four models in the catalog that is 1436 layers against 38 distinct
// kinds: a 37x reduction in per-step work, and it is why the graph's compression is a
// storage choice rather than a speed one.
//
// Conditional nodes are resolved out. Whether a collective runs depends on the layout,
// which is fixed at construction, so a plan contains only the nodes that will actually
// execute. No condition is evaluated per step.
//
// Shape-dependent work is separated from shape-independent work. Weight bytes and expert
// counts do not vary with the batch, so they are summed once into scalars. Only the terms
// that depend on token counts are recomputed.
type Plan struct {
	// Layers holds one entry per distinct layer kind, with how many times it appears.
	Layers []PlannedLayer
	// Head is the final norm and language-model head, priced once per step rather than
	// per layer.
	Head PlannedLayer
	// TotalLayers is the expanded layer count, the base of PiecewiseSegments. Not the
	// divisor for one layer's share of the KV cache, which is the count of layers whose cache
	// is inferred (the kernel's kvLayers).
	TotalLayers int

	// TotalKernels is the launch count for one step: every layer's kernels times its
	// multiplicity, plus the head's and the embedding's (the drafter's are counted in Draft).
	// The per-kernel dispatch cost multiplies it, and that product dominates a
	// single-request decode step; eager launches multiply it too.
	TotalKernels int
	// PrefillOnlyKernels is how many of TotalKernels belong to nodes the graph marks
	// phase: prefill, which a step with no prefill row does not launch (see
	// PlannedLayer.PrefillOnlyKernels).
	PrefillOnlyKernels int

	// Embedding is the graph's embedding nodes -- the vocabulary lookup and the reduction
	// that completes it -- planned as one entry run once per step. Its Count is zero when
	// the graph states none, and the consumer then infers the table (see the kernel's
	// computeFixedBytes).
	Embedding PlannedLayer

	// Draft is the speculator, planned the same way as the target, or nil when the graph
	// states none.
	Draft *DraftPlan
}

// DraftPlan is a speculator's work, planned per part. How often each part runs is the
// drafting mode's to say, which the consumer reads from Drafting and NumSpec.
type DraftPlan struct {
	Method   string
	NumSpec  int
	Drafting model.Drafting
	// Layers is the draft stack, one entry per distinct layer kind like a target plan.
	Layers []PlannedLayer
	// Context is the work that runs once per step over the target's scheduled tokens.
	Context []PlannedLayer
	// Embedding and Head run before and after each pass of Layers, over the same rows;
	// SequentialHead once per draft position over one row per request.
	Embedding, Head, SequentialHead PlannedLayer
	// Kernels is one pass's launches through Embedding, Layers and Head; ContextKernels
	// and SequentialKernels the other two parts'.
	Kernels, ContextKernels, SequentialKernels int
}

// PlannedGEMM is one dense GEMM's shape and the per-token work it contributes, so a
// shape-aware efficiency can be weighted by the work each shape carries.
type PlannedGEMM struct {
	N, K          int
	FLOPsPerToken float64
	// ShardN is true when tensor parallelism splits the OUTPUT width rather than the
	// reduction, which is what decides the shape a rank actually runs.
	//
	// Derived from the graph rather than assumed, because the two cases shard opposite
	// dimensions and a shape-aware efficiency reads both. A column-parallel GEMM takes
	// the full hidden state and splits its output, so its K equals hidden_size; a
	// row-parallel GEMM consumes a sharded activation and reduces to the full hidden
	// state, so its N equals hidden_size. On minimax-m3 at hidden 6144 that classifies
	// qkv_proj (k=6144) and mlp_gate_up (k=6144) as column-parallel and o_proj (n=6144)
	// and mlp_down (n=6144) as row-parallel, which is what those projections are.
	//
	// Sharding the wrong axis is not a small error: mlp_gate_up is the widest GEMM in
	// the layer, and dividing its K by tp would price a 6144-deep reduction as 768-deep.
	ShardN bool
	// Replicated is a GEMM every rank runs whole (tp: replicated): neither dimension
	// splits, and each rank holds the whole weight. DeepSeek-V4.1's fused q/kv
	// A-projection, compressor, indexer projections, router and mHC are such, and dividing
	// them by tp priced a tp-th of the work.
	Replicated bool
	// TokenSharded is under_sequence_parallel: token_sharded. With the model's own
	// sequence parallelism on, the GEMM runs on ceil(tokens / tp) rows with its whole
	// weight; with it off, the layout above applies over every token.
	TokenSharded bool
	// WeightBytes is the whole GEMM's resident bytes, scales included, before any split:
	// zero on an alias, whose weights are resident where they live. WeightReadBytes is
	// what a step reads, which on an alias is its target's.
	WeightBytes, WeightReadBytes float64
	// MHCFn is a hyper-connection mixing GEMM (role ^hc_.*fn, DeepSeek-V4.1's hc_attn_fn and
	// hc_ffn_fn), which a measured mHC form covers (see PlannedMHC).
	MHCFn bool
	// Format is the weight format the GEMM computes from, which selects its compute rate.
	Format Format
	// ShardedAtDecode is a replicated GEMM whose output columns the engine splits over the
	// tensor-parallel ranks for a small step (blis-schemas Node.ShardedAtDecode), nil for
	// every other. It is never fused with a neighbour.
	ShardedAtDecode *model.ShardedAtDecode
}

// Format is the weight format a GEMM computes from: the dtype and, for a block-scaled
// format, how many elements share one scale. The two together decide which kernel the
// engine selects: an FP8 weight with one E8M0 scale per 32 elements is MXFP8, which SM90
// serves with a weight-only Marlin kernel at the BF16 rate, where an FP8 weight with
// 128 x 128 block scales runs natively at the FP8 rate.
type Format struct {
	DType         model.DType
	ScaleElements int
}

// MX reports whether the format carries an OCP microscaling scale, one per 32 elements.
func (f Format) MX() bool { return f.ScaleElements == 32 }

// PlanOptions is what planning needs beyond the graph and the layout's emit decisions.
type PlanOptions struct {
	// WeightBytesPerParam is the resident bytes of one parameter of a GEMM or GroupedGEMM
	// node, scales included; an alias's is its target's.
	WeightBytesPerParam func(model.Node) float64
	// Format is the format a GEMM or GroupedGEMM node computes from.
	Format func(model.Node) Format
	// EntryBytes is a cache's record size under the cache dtype that runs. Nil means the
	// stated bytes_per_entry.
	EntryBytes func(model.Cache) int
	// StateBytes is the recurrent state's element width.
	StateBytes float64
}

// PlannedLayer is one layer kind with its multiplicity and its pre-summed work.
type PlannedLayer struct {
	ID    string
	Count int

	// DenseFLOPsPerToken is the projection and dense-MLP work for one token, summed over
	// every GEMM in the layer. A step multiplies it by the token count.
	DenseFLOPsPerToken float64
	// DenseGEMMs is every dense GEMM's shape, kept alongside the FLOPs sum because a
	// GEMM's efficiency depends on its own n and k and not only on the step's token
	// count. Measured on AISimulate's vLLM fp8 sweep for H200 at a fixed m of 1024, the
	// median efficiency rises from 0.007 at k=32 to 0.476 at k=51200 -- so a ramp in m
	// alone misprices a narrow GEMM by orders of magnitude, and tensor parallelism is
	// what makes projections narrow. The pre-summed scalar above cannot express that,
	// which is why the shapes survive planning.
	//
	// Sharding is NOT applied here: it depends on tp, which planning does not know.
	// The consumer divides N or K as the parallelism dictates.
	DenseGEMMs []PlannedGEMM
	// DenseWeightBytes is the parameter bytes those GEMMs read, independent of batch.
	DenseWeightBytes float64

	// ExpertFLOPsPerTokenPerExpert is the grouped-GEMM work for one token routed to one
	// expert. A step multiplies by tokens and by top_k.
	ExpertFLOPsPerTokenPerExpert float64
	// ExpertWeightBytesPerExpert is one expert's parameter bytes. At decode every local
	// expert is touched, so a step multiplies by the local expert count.
	ExpertWeightBytesPerExpert float64
	// SharedExpertFLOPsPerToken is dense work every token pays, outside the routing.
	SharedExpertFLOPsPerToken float64
	SharedExpertWeightBytes   float64
	TopK                      int

	// Attention, when the layer has it. Zero heads means it does not.
	//
	// These name the layer's PRIMARY attention: the one that reads the KV cache the
	// engine sizes. A layer may run more than one attention kernel -- a block-sparse
	// layer scores which blocks to read before reading them -- and every one of them is
	// in Attentions. The singular fields are the primary entry, kept because the KV
	// geometry and the prefill term are properties of that one attention rather than of
	// the set.
	AttnQHeads, AttnKVHeads, AttnHeadDim int
	AttnKind                             model.AttentionKind
	AttnWindow                           int

	// AttnIndexTopK and AttnCompressRatio bound a sparse-MLA layer's KV read.
	//
	// They are carried because for this kind the byte count is NOT the context. The
	// kernel reads min(context, topk) tokens at full resolution plus, where the
	// architecture compresses the remainder rather than discarding it,
	// (context - topk) / ratio more. A layer selecting by a sliding window states a
	// window and no topk, in which case the window IS the selected count.
	//
	// Without these the only available byte count is the full context, which on
	// DeepSeek-V4-Pro overstates a 1M-token decode read by 128x. Measured on h200, such a
	// step costs 30.6us against 13.5us at 16K context; the full-context reading cannot
	// produce that curve at any rate below datasheet peak.
	AttnIndexTopK     int
	AttnCompressRatio int

	// AttnKVLoRARank is a latent layer's compressed KV width, zero for any other kind.
	// It is carried for the decode-context combine, whose OUTPUT crosses at this width
	// while its query crosses at the full head width: MLADCPManager is built with
	// query_head_dim = kv_lora_rank + qk_rope_head_dim and output_head_dim = kv_lora_rank
	// (vllm/model_executor/layers/attention/mla_attention.py:697-708 at v0.31.0).
	AttnKVLoRARank int

	// Attentions is every attention kernel this layer launches, in graph order.
	//
	// A layer with one attention has one entry and the singular fields repeat it. The
	// distinction matters because a secondary attention is NOT a smaller copy of the
	// primary: a block-sparse layer's indexer scans the whole context to rank blocks
	// while the attention it feeds reads only the chosen ones, so the two have different
	// head geometry AND different bounds. Collapsing them charges the step once and
	// drops the other -- on MiniMax-M3 at 8K context the dropped indexer read is 0.94x
	// the attention read it selects for, and 15x at 131K.
	Attentions []PlannedAttention

	// Recurrent state, when the layer has it.
	RecurrentKind       model.RecurrentKind
	RecurrentStateBytes float64
	// RecurrentHeads, RecurrentStateSize and RecurrentHeadDim are the mixer's geometry,
	// which a prefill priced without a measured form is computed from.
	RecurrentHeads, RecurrentStateSize, RecurrentHeadDim int

	// ElementwiseBytesPerToken is normalization and activation traffic per token.
	ElementwiseBytesPerToken float64

	// Collectives that survived resolution, with the bytes each moves per token.
	Collectives []PlannedCollective

	// Kernels is how many kernel launches this layer makes.
	//
	// Counted from the graph's surviving nodes rather than assumed, because the launch
	// cost is per kernel and a step's total is the count times a per-launch constant. It
	// is a lower bound: the graph records the primitives, and a real layer also launches
	// the epilogues, the router, and the permute and unpermute around a grouped GEMM that
	// the primitives imply. So the count is the primitive count, and the per-launch
	// constant it multiplies is calibrated against measurement with that in mind.
	Kernels int
	// PrefillOnlyKernels is how many of Kernels come from nodes whose context traffic the
	// graph marks phase: prefill. The engine runs those only when the batch has prefills --
	// DeepSeek-V4.1's KV dequantize-and-gather (vllm/models/deepseek_v41/nvidia/
	// flashmla.py:143-165, 305, 317 at v0.31.0) and its indexer's K gather
	// (vllm/model_executor/layers/sparse_attn_indexer.py:416-417, 481) -- so a decode-only
	// step launches Kernels minus these.
	PrefillOnlyKernels int

	// RoutedFormat is the routed experts' compute format, Experts their count, and
	// RoutedTokenSharded whether they see a token shard under the model's sequence
	// parallelism (only the rows divide; the experts stay as expert parallelism splits
	// them).
	RoutedFormat       Format
	Experts            int
	RoutedTokenSharded bool
	// ExpertInner is the routed experts' inner width, n, which a kernel that pads each
	// tensor-parallel slice of it pads.
	ExpertInner int
	// SharedFormat is the shared experts' compute format. SharedTokenSharded is whether
	// the model's sequence parallelism runs them on a token shard with their whole weight,
	// as DeepSeek-V4.1 does; otherwise they are tensor-parallel sharded like a projection.
	SharedFormat       Format
	SharedTokenSharded bool

	// ElementwiseReplicatedBytesPerToken is the traffic of nodes every rank runs whole
	// (tp: replicated), and ElementwiseTokenShardedBytesPerToken that of replicated nodes
	// that see a token shard under the model's sequence parallelism. The rest, in
	// ElementwiseBytesPerToken, divides by tp as it always did.
	ElementwiseReplicatedBytesPerToken   float64
	ElementwiseTokenShardedBytesPerToken float64

	// Context is the context-proportional traffic of this layer's Elementwise nodes.
	Context []PlannedContext
	// Gathers are the table lookups this layer runs.
	Gathers []PlannedGather

	// MHC are the layer's manifold-constrained hyper-connection nodes (Elementwise nodes
	// whose role starts mhc_), kept apart from the elementwise sums they are also in so a
	// measured mHC form can replace their bytes.
	MHC []PlannedMHC

	// AttnReplicated is a primary attention every rank runs over every head.
	AttnReplicated bool
	// AttnCaches are the caches the primary attention states, empty where the graph leaves
	// them to inference.
	AttnCaches []PlannedCache
}

// PlannedMHC is one mHC node. Pre and Post are the components it runs, from its role: pre
// where the role ends _pre, post where it contains post (mhc_attn_post_pre runs both, the
// previous site's post fused with this one's pre). Bytes is what the node adds to its
// elementwise sum per token, and Replicated and TokenSharded say which sum.
type PlannedMHC struct {
	Pre, Post                bool
	Bytes                    float64
	Replicated, TokenSharded bool
}

// mhcFn matches a hyper-connection mixing GEMM's role.
var mhcFn = regexp.MustCompile(`^hc_.*fn`)

// PlannedCache is one cache an attention node names, with its record resolved.
type PlannedCache struct {
	model.Cache
	// EntryBytes is the record under the cache dtype that runs.
	EntryBytes int
}

// PlannedContext is one Elementwise node's context-proportional traffic.
type PlannedContext struct {
	// BytesPerEntry is the traffic per context entry, the named cache's record included.
	BytesPerEntry float64
	// TokensPerEntry is how many context tokens one entry covers.
	TokensPerEntry int
	Per            model.ContextPer
	PrefillOnly    bool
	// MaxModelLenExtent is a decode step whose work spans max_model_len, not the context.
	MaxModelLenExtent bool
	// Replicated is work every rank does whole.
	Replicated bool
}

// PlannedGather is one table lookup.
type PlannedGather struct {
	Rows             int64
	RowBytes         int
	RowsPerToken     int
	OutBytesPerToken int
	Residency        model.Residency
	Control          model.ResidencyControl
	Shard            model.GatherShard
	// Aliased reads another node's table, so it adds traffic but no residency.
	Aliased      bool
	TokenSharded bool
	Role         string
}

// PlannedAttention is one attention kernel a layer launches.
//
// Window is the per-token read bound in tokens, zero when the read tracks context. A
// secondary attention with no window is the expensive case rather than the cheap one:
// it is the kernel that scans everything.
type PlannedAttention struct {
	QHeads, KVHeads, HeadDim int
	Kind                     model.AttentionKind
	Window                   int
	// CompressRatio is how many tokens share one cached state, zero or one where every
	// token has its own. A pooled indexer caches one state per index_kpool tokens, so
	// its scan covers the span's STATES rather than its tokens. Orthogonal to Window:
	// the window bounds how far back the scan looks, the ratio how many states that
	// span holds.
	CompressRatio int
	// HoldsKV is the graph's holds_engine_kv: true for the attention that reads the KV
	// cache the engine sizes, false for a block-index indexer's scorer, which keeps a
	// narrower cache of its own. A block-index scorer's cache is sized by the model's own
	// index projections rather than by num_key_value_heads, so it must not contribute to
	// the geometry that sizes pages. The graph states it; no label decides it.
	HoldsKV bool
	// Role is the graph's label for what this kernel does, e.g. "block_index_scores". It
	// names the kernel only as the key of its measured coefficients
	// (attention_decode_rate_<role>); nothing is decided by it.
	Role string
	// Replicated is a kernel every rank runs over every head (tp: replicated), as
	// DeepSeek-V4.1's indexer is.
	Replicated bool
	// Caches are the caches this kernel states.
	Caches []PlannedCache
}

// findCache returns the cache named name that one of these nodes states.
func findCache(nodes []model.Node, name string) (model.Cache, bool) {
	for _, n := range nodes {
		for _, c := range n.Caches {
			if c.Name == name {
				return c, true
			}
		}
	}
	return model.Cache{}, false
}

// cacheRecord is the record size of the named cache under the cache dtype that runs, or
// zero where no node of the layer names it (validation refuses that).
func cacheRecord(nodes []model.Node, name string, o PlanOptions) int {
	if c, ok := findCache(nodes, name); ok {
		return o.EntryBytes(c)
	}
	return 0
}

// GroupAxis names the parallelism group a collective runs across.
//
// It exists because the primitive alone does not identify a collective's cost. A
// tensor-parallel all-gather and a decode-context-parallel one are the same OP at two
// different widths, and a collective's floor grows with its group: over the nine parts
// this registry carries at both 4 and 8 ranks, the 8-rank floor is 1.35x to 1.59x the
// 4-rank figure for an all-gather and 1.35x to 1.60x for a reduce-scatter, reaching
// 1.02x to 1.89x for an all-reduce. So a consumer that keyed coefficients by op alone
// would price one of two groups against the other's width. The group is what picks the
// width, so it travels with the collective rather than being inferred from the op.
//
// The growth is not uniform across primitives, which is itself the reason to carry the
// axis rather than a single correction factor: an all-to-all's floor barely moves with
// width (1.00x to 1.16x in fp16), since a shuffle's setup does not grow the way a ring's
// does.
type GroupAxis uint8

const (
	// GroupTP is the tensor-parallel group: every ring-shaped collective a model graph
	// emits reduces or gathers across it.
	GroupTP GroupAxis = iota
	// GroupExpert is the expert-parallel group, which a routed MoE dispatch spans.
	GroupExpert
	// GroupDCP is the decode-context-parallel group. Nothing in a model graph names it:
	// DCP is a layout choice, so its collectives are added by the pricer rather than
	// planned, and this axis exists to give them their own width.
	GroupDCP
	// GroupPCP is the prefill-context-parallel group, which gathers the KV a split
	// prefill wrote. Separate from GroupDCP because the two widths are independent and
	// the engine builds them as separate process groups.
	GroupPCP
)

// PlannedCollective is one surviving collective and what it moves.
type PlannedCollective struct {
	Op model.Op
	// Group is the parallelism axis this collective spans, which selects its width.
	Group GroupAxis
	// BytesPerToken is the payload one token contributes. For a routed dispatch this is
	// multiplied by top_k at pricing time; for a ring-shaped one it is not, which is the
	// distinction the backend decides.
	BytesPerToken float64
	RoutedByTopK  bool
}

// Emitter decides whether a conditional node survives. The resolver implements it; the
// plan takes it as a parameter so this package does not depend on the resolver.
type Emitter interface {
	Emits(model.EmitCondition) bool
	Recognizes(model.EmitCondition) bool
}

// ActivationBytes is the width an activation crosses a collective or a normalization at:
// the model's compute dtype, bf16, whatever format the weights are served in. vLLM's
// quantized linears quantize their input transiently and return out_dtype=x.dtype
// (vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py:142-157 and
// cutlass.py:147,153 at v0.31.0), so the hidden state between projections, the reductions
// over it and the norms that read it are 16-bit on an fp8, int8 or fp4 deployment as on a
// bf16 one.
//
// The MoE combine is at this width too: the experts' output is bf16. The MoE DISPATCH is
// priced at it as well, and on a quantized deployment that is an upper bound, not the
// engine's width. The default allgather_reducescatter backend (vllm/config/parallel.py:202)
// quantizes the hidden state before gathering it -- moe_kernel_quantize_input in
// _quantize_and_setup_dispatch, vllm/model_executor/layers/fused_moe/prepare_finalize/
// naive_dp_ep.py:16-51 -- and gathers per-token or per-block scales beside it (a static
// scalar scale is not gathered, :46-50), unless the experts kernel
// quantizes its own input (expects_unquantized_inputs, modular_kernel.py:505-512). Which
// experts kernel serves a layer is chosen per quantization method and platform, and this
// kernel does not model that choice, so it cannot tell a 1-byte (fp8) or half-byte (fp4)
// dispatch from a 2-byte one. COVERAGE LIMIT: an fp8 w8a8 deployment whose experts kernel
// takes quantized input has its dispatch charged about twice its bytes; its combine is
// charged right.
//
// An earlier form sized all three at the served weight width, so an fp8 deployment's
// collectives and norms were charged half their bytes.
const ActivationBytes = 2.0

// PiecewiseSegments is how many graph segments a PIECEWISE capture splits a step into: the
// split points plus one. vLLM splits at every op in CompilationConfig._attention_ops
// (vllm/config/compilation.py:764-782 at v0.31.0) -- each layer's attention or recurrent
// mixer, and on a sparse-MLA layer its indexer as well (vllm::sparse_attn_indexer), which
// the graph states as a second, cache-less attention node.
func (p *Plan) PiecewiseSegments() int {
	return PiecewiseSegmentsOf(p.TotalLayers, p.Layers)
}

// PiecewiseSegmentsOf is PiecewiseSegments over any share of the stack: totalLayers decoder
// layers, of which layers are the kinds with their counts. A pipeline stage is its own
// compiled graph, split at its own layers' attention ops (each stage's worker compiles
// and captures the layers it holds, vllm/model_executor/models/utils.py:895-919@v0.32.0
// make_layers), so a stage's segment count is this over its layers.
func PiecewiseSegmentsOf(totalLayers int, layers []PlannedLayer) int {
	splits := totalLayers
	for _, l := range layers {
		if l.AttnKind != model.AttentionSparseMLA {
			continue
		}
		for _, a := range l.Attentions {
			if !a.HoldsKV {
				splits += l.Count
				break
			}
		}
	}
	return splits + 1
}

// BuildPlan flattens a graph under a layout, with every weight at dtypeBytes unless a
// node overrides its dtype, and recurrent state at stateDtypeBytes. It is BuildPlanWith
// for a caller that needs nothing finer.
func BuildPlan(g *model.Graph, em Emitter, dtypeBytes, stateDtypeBytes float64) (*Plan, error) {
	if g == nil {
		return nil, fmt.Errorf("no graph to plan")
	}
	return BuildPlanWith(g, em, PlanOptions{
		WeightBytesPerParam: func(n model.Node) float64 { return weightBytes(n, dtypeBytes) },
		Format: func(n model.Node) Format {
			if n.WeightDType != "" {
				return Format{DType: n.WeightDType}
			}
			return Format{DType: g.Global.WeightDType}
		},
		StateBytes: stateDtypeBytes,
	})
}

// BuildPlanWith flattens a graph under a layout.
//
// Any condition the emitter does not recognize is an error rather than a dropped node:
// silently omitting a collective would remove a cost with nothing reporting it, and the
// resulting step time would look plausible.
func BuildPlanWith(g *model.Graph, em Emitter, o PlanOptions) (*Plan, error) {
	if g == nil {
		return nil, fmt.Errorf("no graph to plan")
	}
	if o.EntryBytes == nil {
		o.EntryBytes = func(c model.Cache) int { return c.BytesPerEntry }
	}
	byID := make(map[string]model.LayerKind, len(g.LayerKinds))
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
	}
	hidden := g.Global.HiddenSize

	p := &Plan{TotalLayers: g.Stack.Layers()}
	layers, kernels, err := planStack(g.Stack, byID, em, o, hidden)
	if err != nil {
		return nil, err
	}
	p.Layers, p.TotalKernels = layers, kernels

	head, err := planNodes(g.Head, em, o, hidden)
	if err != nil {
		return nil, fmt.Errorf("head: %w", err)
	}
	head.ID = "head"
	head.Count = 1
	p.Head = head
	p.TotalKernels += head.Kernels

	if len(g.Embedding) > 0 {
		emb, err := planNodes(g.Embedding, em, o, hidden)
		if err != nil {
			return nil, fmt.Errorf("embedding: %w", err)
		}
		emb.ID, emb.Count = "embedding", 1
		p.Embedding = emb
		p.TotalKernels += emb.Kernels
	}

	p.PrefillOnlyKernels = p.Head.PrefillOnlyKernels + p.Embedding.PrefillOnlyKernels
	for _, l := range p.Layers {
		p.PrefillOnlyKernels += l.PrefillOnlyKernels * l.Count
	}

	if sp := g.Speculator; sp != nil {
		d := &DraftPlan{Method: sp.Method, NumSpec: sp.NumSpec, Drafting: sp.Drafting}
		if d.Layers, d.Kernels, err = planStack(sp.Stack, byID, em, o, hidden); err != nil {
			return nil, fmt.Errorf("speculator: %w", err)
		}
		if sp.ContextStack != nil {
			if d.Context, d.ContextKernels, err = planStack(*sp.ContextStack, byID, em, o,
				hidden); err != nil {
				return nil, fmt.Errorf("speculator context stack: %w", err)
			}
		}
		for _, part := range []struct {
			nodes []model.Node
			into  *PlannedLayer
			id    string
		}{
			{sp.Embedding, &d.Embedding, "draft_embedding"},
			{sp.Head, &d.Head, "draft_head"},
			{sp.SequentialHead, &d.SequentialHead, "draft_sequential_head"},
		} {
			if len(part.nodes) == 0 {
				continue
			}
			pl, err := planNodes(part.nodes, em, o, hidden)
			if err != nil {
				return nil, fmt.Errorf("speculator %s: %w", part.id, err)
			}
			pl.ID, pl.Count = part.id, 1
			*part.into = pl
		}
		d.Kernels += d.Embedding.Kernels + d.Head.Kernels
		d.SequentialKernels = d.SequentialHead.Kernels
		p.Draft = d
	}
	return p, nil
}

// planStack plans one stack, one entry per distinct layer kind in first-use order, and
// returns its launch count over the expanded stack.
func planStack(st model.Stack, byID map[string]model.LayerKind, em Emitter, o PlanOptions,
	hidden int) ([]PlannedLayer, int, error) {
	counts := map[string]int{}
	order := []string{}
	for _, id := range st.Expand() {
		if _, seen := counts[id]; !seen {
			order = append(order, id)
		}
		counts[id]++
	}
	var out []PlannedLayer
	kernels := 0
	for _, id := range order {
		lk, ok := byID[id]
		if !ok {
			return nil, 0, fmt.Errorf("the stack names layer kind %q, which the graph does not declare", id)
		}
		pl, err := planNodes(lk.Nodes, em, o, hidden)
		if err != nil {
			return nil, 0, fmt.Errorf("layer kind %q: %w", id, err)
		}
		pl.ID = id
		pl.Count = counts[id]
		kernels += pl.Kernels * pl.Count
		out = append(out, pl)
	}
	return out, kernels, nil
}

// kernelsPerPrimitive is how many launches one graph primitive implies.
//
// The graph records primitives, not kernels. A GEMM is one kernel; a grouped GEMM is
// several, because the tokens must be permuted into expert order, the experts run, and the
// result unpermuted. Attention on a paged cache launches its own reshape alongside the
// attention kernel itself.
//
// These multipliers are the one place a launch count is asserted rather than counted, and
// they are conservative: a real implementation may fuse some of them. Under-counting makes
// the launch term optimistic, which the calibrated per-launch constant then partly absorbs
// — so the split between count and constant is less certain than their product.
var kernelsPerPrimitive = map[model.Op]int{
	model.OpGEMM:            1,
	model.OpGroupedGEMM:     4, // permute, group offsets, the grouped GEMM, unpermute
	model.OpAttention:       2, // the attention kernel and its cache reshape
	model.OpElementwise:     1,
	model.OpRecurrentUpdate: 2,
	model.OpAllReduce:       1,
	model.OpAllGather:       1,
	model.OpReduceScatter:   1,
	model.OpAll2All:         2, // dispatch and combine are separate launches
}

// weightBytes is the width this node's parameters are stored at: its own override where
// it states one, the checkpoint's otherwise. Mixed-precision MoE is the case that needs
// it -- DeepSeek-V4-Pro stores routed experts at fp4 beside fp8 everywhere else, and
// pricing them at the global width doubles 720 GiB of experts to 1,441 GiB.
func weightBytes(n model.Node, global float64) float64 {
	if n.WeightDType != "" {
		if b := n.WeightDType.Bytes(); b > 0 {
			return b
		}
	}
	return global
}

func planNodes(nodes []model.Node, em Emitter, o PlanOptions,
	hidden int) (PlannedLayer, error) {
	var pl PlannedLayer
	// routedSet and sharedSet record that an expert format was taken, so a second node
	// stating another one is refused rather than silently overwriting it.
	routedSet, sharedSet := false, false
	setFormat := func(into *Format, set *bool, f Format, what string) error {
		if *set && *into != f {
			return fmt.Errorf("two %s nodes in one layer compute from different formats "+
				"(%s and %s), which one expert term cannot price", what, into.DType, f.DType)
		}
		*into, *set = f, true
		return nil
	}
	// fusable is the index in DenseGEMMs of the GEMM emitted by the IMMEDIATELY preceding
	// node, or -1 when the previous node was anything else. Fusion needs adjacency in the
	// node sequence, not in DenseGEMMs: minimax-m3's layer-0 has qkv_proj and mlp_gate_up
	// both column-parallel on the same K with the attention op between them, and vLLM
	// cannot fuse across that.
	fusable := -1
	// dispatchWidth is the width a token crosses the MoE dispatch and combine at: the routed
	// experts' input, which is latent_size where the layer's routed GroupedGEMM states one.
	// A latent MoE projects to that width before the experts run, and FusedMoE is built at
	// it, so its all-to-all carries latent rows: Nemotron-H passes the projection as
	// routed_input_transform and builds FusedMoE at moe_latent_size
	// (vllm/model_executor/models/nemotron_h.py:153-156, 230-250@v0.32.0; applied before
	// dispatch, fused_moe/runner/moe_runner.py:714-722), and Muse Spark pools to
	// moe_hidden_size before its experts (MuseSpark1xMoE, vllm/models/muse_spark/nvidia/
	// model.py:450-457, 597-601, 922-928@0f6b2619). Read off every node first, so the order
	// the graph states the All2All and the GroupedGEMM in does not matter.
	dispatchWidth := hidden
	for _, n := range nodes {
		if !n.IsSharedExpert() && n.Op == model.OpGroupedGEMM &&
			n.LatentSize > 0 && (n.Emit == model.EmitAlways || em.Emits(n.Emit)) {
			dispatchWidth = n.LatentSize
		}
	}
	for _, n := range nodes {
		if n.Emit != model.EmitAlways {
			if !em.Recognizes(n.Emit) {
				return pl, fmt.Errorf(
					"node %s carries condition %q, which this resolver cannot evaluate; "+
						"pricing it would either drop a cost or invent one",
					n.Op, n.Emit)
			}
			if !em.Emits(n.Emit) {
				continue
			}
		}
		// Every surviving node launches at least one kernel. Counted here, once, so a
		// conditional node that was resolved away costs no launch either.
		launches, ok := kernelsPerPrimitive[n.Op]
		if !ok {
			launches = 1
		}
		pl.Kernels += launches
		if n.Context != nil && n.Context.Phase == model.PhasePrefill {
			pl.PrefillOnlyKernels += launches
		}
		// Any node that is not a dense GEMM breaks a fusion chain: the engine fuses
		// projections of one hidden state, and an op between them consumes that state.
		// Set before the switch so every other case resets it without having to say so.
		wasFusable := fusable
		fusable = -1

		tokenSharded := n.UnderSequenceParallel == model.SPTokenSharded
		switch n.Op {
		case model.OpGEMM:
			// Two FLOPs per multiply-accumulate, per output element, per token.
			f := 2 * float64(n.N) * float64(n.K)
			pl.DenseFLOPsPerToken += f
			// The split is the graph's where it states one (tp, blis-schemas v0.3.0), and
			// inferred otherwise. hidden == 0 means the caller did not supply it; then the
			// inference does not fire and the consumer shards K, which is the
			// pre-existing behaviour.
			shardN := hidden > 0 && n.K == hidden && n.N != hidden
			switch n.TP {
			case model.TPColumn:
				shardN = true
			case model.TPRow, model.TPReplicated:
				shardN = false
			}
			params := float64(n.N) * float64(n.K)
			read := params * o.WeightBytesPerParam(n)
			resident := read
			if n.Aliases != "" {
				resident = 0
			}
			g := PlannedGEMM{
				N: n.N, K: n.K, FLOPsPerToken: f, ShardN: shardN,
				Replicated: n.TP == model.TPReplicated, TokenSharded: tokenSharded,
				WeightBytes: resident, WeightReadBytes: read, Format: o.Format(n),
				MHCFn: mhcFn.MatchString(n.Role), ShardedAtDecode: n.ShardedAtDecode,
			}
			// Fuse into the previous GEMM when the engine would. vLLM concatenates
			// column-parallel projections that share an input ALONG THE OUTPUT
			// DIMENSION -- QKVParallelLinear and MergedColumnParallelLinear, both in
			// vllm/model_executor/layers/linear.py -- so three projections of the same
			// hidden state are one matmul with the output widths summed, not three
			// matmuls.
			//
			// The condition is CONSECUTIVE, column-parallel, and the same K. Consecutive
			// matters: minimax-m3's layer-1 has qkv_proj (n=9216) immediately followed by
			// index_qk_proj (n=640) on the same k=6144, which vLLM fuses; its layer-0 has
			// qkv_proj and mlp_gate_up both column-parallel on k=6144 but separated by
			// the attention op, which it cannot. A fused pair must also split alike and
			// compute from one format, since one matmul has one layout and one kernel.
			//
			// This matters because efficiency is strongly non-linear in output width. On
			// AISimulate's vLLM fp8 sweep for H200 at 256 rows, measured efficiency is
			// 0.0008 at n=16 and 0.0138 at n=256, so a 640-wide projection priced as its
			// own kernel costs far more than the same work folded into a 9216-wide one --
			// and sharded over 8 ranks that 640 becomes 80.
			merged := false
			if wasFusable >= 0 {
				prev := &pl.DenseGEMMs[wasFusable]
				if prev.ShardN && g.ShardN && prev.K == g.K && prev.Format == g.Format &&
					prev.Replicated == g.Replicated && prev.TokenSharded == g.TokenSharded {
					prev.N += g.N
					prev.FLOPsPerToken += g.FLOPsPerToken
					prev.WeightBytes += g.WeightBytes
					prev.WeightReadBytes += g.WeightReadBytes
					fusable = wasFusable
					merged = true
				}
			}
			if !merged {
				pl.DenseGEMMs = append(pl.DenseGEMMs, g)
				fusable = len(pl.DenseGEMMs) - 1
			}
			// Weight bytes are per node whether or not the launches fuse: fusing changes
			// how many kernels read the parameters, not how many parameters there are.
			pl.DenseWeightBytes += resident
		case model.OpGroupedGEMM:
			ew := o.WeightBytesPerParam(n)
			if n.IsSharedExpert() {
				// A shared expert stated as its own node -- the only way the schema can give
				// it a dtype of its own, as DeepSeek-V4.1's MXFP8 shared expert beside its
				// MXFP4 routed ones and Nemotron-3-Ultra's FP8 one beside NVFP4 need. It is
				// dense: every token pays it once. It neither scales with nor sets top_k,
				// and it is not one of the routed experts a rank holds. Its input is the
				// hidden state: a latent projection runs before the ROUTED experts only
				// (Nemotron-H passes it as routed_input_transform, vllm/model_executor/
				// models/nemotron_h.py), so latent_size does not narrow it.
				shared := 3 * float64(n.N) * float64(n.K) * float64(max(n.Experts, 1))
				pl.SharedExpertFLOPsPerToken += 2 * shared
				pl.SharedExpertWeightBytes += shared * ew
				if err := setFormat(&pl.SharedFormat, &sharedSet, o.Format(n),
					"shared-expert"); err != nil {
					return pl, err
				}
				pl.SharedTokenSharded = tokenSharded
				break
			}
			// A gated expert holds three matrices; the node's N is the inner width. The
			// reduction is latent_size where the graph states one: a projection to that
			// narrower space runs before the routed experts, so K for their matmuls is the
			// latent width, not the hidden size (blis-schemas Node.LatentSize).
			//
			// COVERAGE LIMIT: every expert is priced as gated. A non-gated expert (relu2,
			// as Nemotron-H's are) holds two matrices, and blis-schemas v0.3.0 has no field
			// that says which an expert is, so such a model's experts are over-stated by
			// half.
			k := n.K
			if n.LatentSize > 0 {
				k = n.LatentSize
			}
			perExpert := 3 * float64(n.N) * float64(k)
			pl.ExpertFLOPsPerTokenPerExpert += 2 * perExpert
			// Only the WEIGHT terms take a node's dtype override: activations and
			// collectives cross at the served width whatever the weights are stored as.
			pl.ExpertWeightBytesPerExpert += perExpert * ew
			pl.TopK = max(pl.TopK, n.TopK)
			pl.Experts = max(pl.Experts, n.Experts)
			pl.ExpertInner = max(pl.ExpertInner, n.N)
			if err := setFormat(&pl.RoutedFormat, &routedSet, o.Format(n),
				"routed-expert"); err != nil {
				return pl, err
			}
			pl.RoutedTokenSharded = tokenSharded
			if n.SharedExperts > 0 {
				inner := n.N
				if n.SharedIntermediateSize > 0 {
					inner = n.SharedIntermediateSize
				}
				shared := 3 * float64(inner) * float64(n.K) * float64(n.SharedExperts)
				pl.SharedExpertFLOPsPerToken += 2 * shared
				pl.SharedExpertWeightBytes += shared * ew
				if err := setFormat(&pl.SharedFormat, &sharedSet, o.Format(n),
					"shared-expert"); err != nil {
					return pl, err
				}
				pl.SharedTokenSharded = tokenSharded
			}
		case model.OpAttention:
			caches := make([]PlannedCache, 0, len(n.Caches))
			for _, c := range n.Caches {
				caches = append(caches, PlannedCache{Cache: c, EntryBytes: o.EntryBytes(c)})
			}
			pl.Attentions = append(pl.Attentions, PlannedAttention{
				QHeads: n.NumQHeads, KVHeads: n.NumKVHeads, HeadDim: n.HeadDim,
				Kind: n.AttentionKind, Window: n.Window, Role: n.Role, HoldsKV: n.ReadsEngineKV(),
				CompressRatio: n.CompressRatio, Replicated: n.TP == model.TPReplicated,
				Caches: caches,
			})
			// The primary is the first attention that reads the engine's KV cache, not
			// the last node seen. An earlier assignment here overwrote, so on a layer
			// whose indexer precedes its attention the primary was whichever came last.
			if pl.AttnQHeads == 0 && n.ReadsEngineKV() {
				pl.AttnQHeads, pl.AttnKVHeads, pl.AttnHeadDim =
					n.NumQHeads, n.NumKVHeads, n.HeadDim
				pl.AttnKind, pl.AttnWindow = n.AttentionKind, n.Window
				pl.AttnIndexTopK, pl.AttnCompressRatio = n.IndexTopK, n.CompressRatio
				pl.AttnKVLoRARank = n.KVLoRARank
				pl.AttnReplicated = n.TP == model.TPReplicated
				pl.AttnCaches = caches
			}
		case model.OpRecurrentUpdate:
			pl.RecurrentKind = n.RecurrentKind
			pl.RecurrentStateBytes += recurrentBytes(n, o.StateBytes)
			pl.RecurrentHeads = max(pl.RecurrentHeads, n.NumHeads)
			pl.RecurrentStateSize = max(pl.RecurrentStateSize, n.StateSize)
			pl.RecurrentHeadDim = max(pl.RecurrentHeadDim, n.HeadDim)
		case model.OpElementwise:
			b := float64(n.BytesPerToken)
			if b == 0 && n.Context == nil {
				// A normalization reads and writes the hidden state, at the activation
				// width.
				b = 2 * float64(hidden) * ActivationBytes
			}
			// A cache insert writes one record each time it runs, which the node names
			// rather than restates (writes_cache): its BytesPerToken is the traffic beyond
			// the record.
			if n.WritesCache != "" {
				b += float64(cacheRecord(nodes, n.WritesCache, o))
			}
			// A node that runs once per compressed entry runs once every PerTokens tokens.
			if n.PerTokens > 1 {
				b /= float64(n.PerTokens)
			}
			if strings.HasPrefix(n.Role, "mhc_") {
				pl.MHC = append(pl.MHC, PlannedMHC{
					Pre: strings.HasSuffix(n.Role, "_pre"), Post: strings.Contains(n.Role, "post"),
					Bytes: b, Replicated: n.TP == model.TPReplicated, TokenSharded: tokenSharded,
				})
			}
			switch {
			case n.TP == model.TPReplicated && tokenSharded:
				pl.ElementwiseTokenShardedBytesPerToken += b
			case n.TP == model.TPReplicated:
				pl.ElementwiseReplicatedBytesPerToken += b
			default:
				pl.ElementwiseBytesPerToken += b
			}
			if c := n.Context; c != nil {
				pc := PlannedContext{
					BytesPerEntry: float64(c.BytesPerEntry), TokensPerEntry: c.TokensPerEntry,
					Per: c.Per, PrefillOnly: c.Phase == model.PhasePrefill,
					MaxModelLenExtent: c.DecodeExtent == model.ExtentMaxModelLen,
					Replicated:        n.TP == model.TPReplicated,
				}
				if c.ReadsCache != "" {
					if rc, ok := findCache(nodes, c.ReadsCache); ok {
						pc.BytesPerEntry += float64(o.EntryBytes(rc))
						pc.TokensPerEntry = rc.EntryTokens()
					}
				}
				pc.TokensPerEntry = max(pc.TokensPerEntry, 1)
				pl.Context = append(pl.Context, pc)
			}
		case model.OpGather:
			pl.Gathers = append(pl.Gathers, PlannedGather{
				Rows: n.Rows, RowBytes: n.RowBytes, RowsPerToken: n.RowsPerToken,
				OutBytesPerToken: n.OutBytesPerToken, Residency: n.Residency.Effective(),
				Control: n.ResidencyControl, Shard: n.Shard.Effective(),
				Aliased: n.Aliases != "", TokenSharded: tokenSharded, Role: n.Role,
			})
		case model.OpAllReduce, model.OpAllGather, model.OpReduceScatter:
			// The collective's full tensor per token: what the graph states where it is
			// not one hidden-width activation (bytes_per_token, blis-schemas v0.3.0) --
			// DeepSeek-V4.1's engram row all-gather moves 12,288 B a token against a
			// 10,240 B hidden state -- and that activation otherwise.
			b := float64(hidden) * ActivationBytes
			if n.BytesPerToken > 0 {
				b = float64(n.BytesPerToken)
			}
			pl.Collectives = append(pl.Collectives, PlannedCollective{
				Op: n.Op, Group: GroupTP, BytesPerToken: b,
			})
		case model.OpAll2All:
			pl.Collectives = append(pl.Collectives, PlannedCollective{
				Op: n.Op, Group: GroupExpert, BytesPerToken: float64(dispatchWidth) * ActivationBytes,
				// Whether top_k multiplies this is a backend property, set by the caller
				// that knows the resolved backend.
				RoutedByTopK: true,
			})
		}
	}
	// A layer that runs attention must have a primary: the attention that reads the cache
	// the engine sizes, which the graph marks with holds_engine_kv. With none, the
	// layer's attention, its caches and its KV would all go unpriced, so it is refused.
	if len(pl.Attentions) > 0 && pl.AttnQHeads == 0 {
		return pl, fmt.Errorf("the layer runs attention but no attention node states " +
			"holds_engine_kv true, so none reads the engine's cache")
	}
	return pl, nil
}

func recurrentBytes(n model.Node, stateBytes float64) float64 {
	switch n.RecurrentKind {
	case model.RecurrentMamba2:
		// A convolutional state and a temporal state, sized independently.
		convDim := n.IntermediateSize + 2*n.NumGroups*n.StateSize
		conv := float64(convDim) * float64(maxInt(n.ConvKernel-1, 1))
		temporal := float64(n.NumHeads) * float64(n.HeadDim) * float64(n.StateSize)
		if n.HeadDim == 0 {
			// Where the graph states no per-head width, the state is square in its size.
			temporal = float64(n.NumHeads) * float64(n.StateSize) * float64(n.StateSize)
		}
		return (conv + temporal) * stateBytes
	case model.RecurrentKDA, model.RecurrentGDN:
		// A matrix-valued state per head.
		return float64(n.NumHeads) * float64(n.StateSize) * float64(n.StateSize) * stateBytes
	}
	return 0
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
