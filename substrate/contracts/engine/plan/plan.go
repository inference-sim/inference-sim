// Package plan holds the boundary between an engine and the cost model (design §3.3, §6.3):
// the PoolPlan an engine hands over once per pool, the StepShape it forms each step, and the
// Transfer it states for each KV move (as the Move of the engine.Transfer it reports). The
// engine resolves its configuration, lowers the model and sizes its KV budget; the cost model
// prices what it is given and knows no engine.
//
// Three rules shape every type here, and each is a test.
//
//   - No field takes its values from an engine's vocabulary. A field holds a size, a count, a
//     catalog name or a word from a fixed list in contracts, never an engine's setting name or
//     default (X2).
//   - A zero value means "absent" or the field's documented neutral value (stage 0, one step in
//     flight), never an engine's default (X2, SUB-6).
//   - Every field changes some price, or is listed as not yet priced with an issue (SUB-11).
//
// Sizes carry their units and their basis as types (SUB-16, package units): a figure states
// whether it is per rank, per node or for the group, and the compiler refuses to mix them.
package plan

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/units"
)

// PoolPlan is fixed for the life of one engine pool.
type PoolPlan struct {
	// Hardware is the catalog chip's name, and Fabric the catalog network's name; Fabric is ""
	// on one node.
	Hardware string
	Fabric   string
	// Layout is the parallel widths and where ranks sit.
	Layout Layout
	// Layers has one entry per (layer, primitive) the engine runs.
	Layers []LoweredLayer
	// KV describes each cache group; KVBudget is the bytes per rank available for KV, one value
	// per pipeline stage.
	KV       []KVGroup
	KVBudget []units.PerRank[units.Bytes]
	// Tiers are the KV tiers below GPU memory, as the engine resolved them.
	Tiers []KVTier
	// Graphs says which launch shapes the engine captures.
	Graphs GraphPolicy
	// Drafts is speculative decoding; DraftLayers are the draft model's own layers, lowered like
	// the target's.
	Drafts      DraftPolicy
	DraftLayers []LoweredLayer
	// Host is per-request and per-token host work.
	Host HostPlan
	// Weights are the resident tensors per rank after weight post-processing; Buffers are the
	// buffers and workspaces the engine chooses, sized as its upstream sizes them. Together
	// with the cost model's activations and graph pools they make the memory breakdown close
	// (SUB-14).
	Weights []ResidentTensor
	Buffers []Buffer
	// InFlight is the most steps in flight at once; 0 means 1, the value without asynchronous
	// or overlap scheduling.
	InFlight int
}

// Implementations are the distinct kernel implementations the pool runs: those of Layers and of
// DraftLayers, each once, sorted by library, variant, code hash and toolkit: the pool-wide list a
// Result records. The cost model matches a coefficient against the implementations of the
// primitive it prices, which ImplementationsByPrimitive gives, or against the operations that
// consume it, which Operations gives.
func (p PoolPlan) Implementations() []KernelImpl {
	var all []KernelImpl
	for _, ls := range [][]LoweredLayer{p.Layers, p.DraftLayers} {
		for _, l := range ls {
			all = append(all, l.KernelImpl)
		}
	}
	return distinctSorted(all)
}

// distinctSorted returns each implementation once, sorted by library, variant, code hash and
// toolkit.
func distinctSorted(in []KernelImpl) []KernelImpl {
	seen := map[KernelImpl]bool{}
	var out []KernelImpl
	for _, k := range in {
		if !seen[k] {
			seen[k] = true
			out = append(out, k)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Library != b.Library {
			return a.Library < b.Library
		}
		if a.Variant != b.Variant {
			return a.Variant < b.Variant
		}
		if a.CodeHash != b.CodeHash {
			return a.CodeHash < b.CodeHash
		}
		return a.Toolkit < b.Toolkit
	})
	return out
}

// ImplementationsByPrimitive are the distinct kernel implementations the pool runs for each
// lowered primitive, from Layers and DraftLayers, each list sorted as Implementations is. It is
// what the cost model binds a coefficient to (design VAL-8): a coefficient that prices a
// primitive is matched against the implementations of that primitive, not against everything
// the pool runs. A caller building a kernel for this pool passes it as
// latencykernel.Inputs.Implementations.
func (p PoolPlan) ImplementationsByPrimitive() map[Primitive][]KernelImpl {
	return ImplementationsOf(p.Operations())
}

// ImplementationsOf pools operations by primitive: the distinct implementations of each,
// sorted as Implementations is.
func ImplementationsOf(ops []Operation) map[Primitive][]KernelImpl {
	out := map[Primitive][]KernelImpl{}
	for _, op := range ops {
		out[op.Primitive] = append(out[op.Primitive], op.Impl)
	}
	for prim, impls := range out {
		out[prim] = distinctSorted(impls)
	}
	return out
}

// Operation is one lowered operation and the kernel implementation that executes it: which
// layer of which graph (the target's, or the draft model's), which node of it (Scorer), which
// primitive, in which stored format, which kernel. It is the identity the cost model keeps when
// it binds a coefficient to the kernel that consumes it (design VAL-8, 6.2), so a coefficient
// read only by the sliding-window layers is matched against the kernel those layers run, one
// read only by the BF16 GEMMs against the kernel that runs those, and one read only by a
// layer's block-index scorer against the scorer's, and not against every kernel that runs the
// primitive.
type Operation struct {
	// Draft is true for an entry of DraftLayers, whose Layer indexes the draft model's own
	// stack; false for an entry of Layers, whose Layer indexes the target graph's.
	Draft bool
	Layer int
	// Scorer is the node of the layer (LoweredLayer.Scorer); false is the primary.
	Scorer    bool
	Primitive Primitive
	// Format is the stored format of the weight or the cache the operation reads
	// (PrimitiveShape.Format); the zero value is not stated.
	Format units.QuantFormat
	// Dtype is the element type of the cache the operation reads (PrimitiveShape.Dtype).
	Dtype Dtype
	Impl  KernelImpl
}

// Operations lists every lowered operation of the pool, the target's then the draft model's, in
// plan order. A caller building a kernel for this pool passes it as
// latencykernel.Inputs.Operations.
func (p PoolPlan) Operations() []Operation {
	var out []Operation
	for _, l := range p.Layers {
		out = append(out, Operation{Layer: l.Layer, Scorer: l.Scorer, Primitive: l.Primitive, Format: l.Shape.Format, Dtype: l.Shape.Dtype, Impl: l.KernelImpl})
	}
	for _, l := range p.DraftLayers {
		out = append(out, Operation{Draft: true, Layer: l.Layer, Scorer: l.Scorer, Primitive: l.Primitive, Format: l.Shape.Format, Dtype: l.Shape.Dtype, Impl: l.KernelImpl})
	}
	return out
}

// Layout is the parallel widths of a pool and the placement of its ranks. A width of 0 is
// read as 1: a layout that shards along no axis has one rank along it.
type Layout struct {
	Tensor, Pipeline, Data, Expert, Context int
	// RanksPerNode and Nodes place the pool's ranks: Nodes nodes of RanksPerNode ranks. A pool
	// that fills one node states Nodes 1.
	RanksPerNode int
	Nodes        int
}

// Width returns a width with its floor applied.
func width(w int) int {
	if w < 1 {
		return 1
	}
	return w
}

// Ranks is the number of ranks in the pool: tensor x pipeline x data. Expert and context
// parallelism reuse these ranks and add none (design §9.2).
func (l Layout) Ranks() int { return width(l.Tensor) * width(l.Pipeline) * width(l.Data) }

// StageRanks is the ranks in one pipeline stage.
func (l Layout) StageRanks() int { return width(l.Tensor) * width(l.Data) }

// Stages is the pipeline depth.
func (l Layout) Stages() int { return width(l.Pipeline) }

// Primitive is a word from a fixed list: what a lowered layer's kernel does. The list is in
// this package, not in an engine; an engine maps its own operation names onto it in its
// lowering.
type Primitive string

const (
	PrimAttentionPrefill Primitive = "attention.prefill"
	PrimAttentionDecode  Primitive = "attention.decode"
	PrimGEMM             Primitive = "gemm"
	PrimMoEGroupedGEMM   Primitive = "moe.grouped_gemm"
	PrimRecurrentUpdate  Primitive = "recurrent.update"
	PrimElementwise      Primitive = "elementwise"
	PrimGather           Primitive = "gather"
	PrimAllReduce        Primitive = "all_reduce"
	PrimAllGather        Primitive = "all_gather"
	PrimReduceScatter    Primitive = "reduce_scatter"
	PrimAllToAll         Primitive = "all2all"
)

var primitives = map[Primitive]bool{
	PrimAttentionPrefill: true, PrimAttentionDecode: true, PrimGEMM: true, PrimMoEGroupedGEMM: true,
	PrimRecurrentUpdate: true, PrimElementwise: true, PrimGather: true, PrimAllReduce: true,
	PrimAllGather: true, PrimReduceScatter: true, PrimAllToAll: true,
}

// Valid reports whether p is in the fixed list.
func (p Primitive) Valid() bool { return primitives[p] }

// IsCollective reports whether the primitive is a collective across ranks.
func (p Primitive) IsCollective() bool {
	switch p {
	case PrimAllReduce, PrimAllGather, PrimReduceScatter, PrimAllToAll:
		return true
	}
	return false
}

// LoweredLayer is one primitive of one layer, as the engine lowered it.
type LoweredLayer struct {
	// Layer is the index in the model graph, and Stage the pipeline stage that runs it.
	Layer int
	Stage int
	// Primitive is from the fixed list above.
	Primitive Primitive
	// Scorer is true when this entry executes the layer's block-index scorer, a node of its own
	// beside a sparse layer's primary attention, and false for the layer's primary node. A layer
	// that runs both has two entries for the primitive. The coefficient registry prices the two
	// apart (attention_decode_*_block_index_scores), so the cost model binds a coefficient to
	// the kernel of the node that reads it.
	Scorer bool
	// KernelImpl is what executes it; the coefficient registry looks coefficients up by this.
	KernelImpl KernelImpl
	// Shape is the primitive's geometry.
	Shape PrimitiveShape
	// Collective is set exactly when Primitive is a collective.
	Collective *Collective
}

// KernelImpl names the exact kernel that runs a primitive. The coefficient registry's lookup
// key is its CodeHash, not a package version (design §6.4): a version bump that leaves a kernel
// unchanged needs no substrate change.
type KernelImpl struct {
	// Library names the library, such as a named attention library. A plan requires only that it
	// is stated. The coefficient lookup matches Library, Variant and CodeHash together (VAL-8), so
	// a library name outside the registry's matches no keyed entry.
	Library string
	// Variant is the library's own exported kernel name; an engine's backend names map to it in
	// the engine's lowering.
	Variant string
	// CodeHash is the hash of the kernel's source and target GPU architecture, or of a binary
	// library's file and architecture. Empty means the engine could not identify the build; the
	// lookup then matches no coefficient that names an implementation.
	CodeHash string
	// Toolkit is the compiler or CUDA toolkit version the kernel was built with. A toolkit
	// change can change a kernel's speed without changing its source; the landing gate shows
	// whether the old value still holds.
	Toolkit string
}

// Dtype is a number format, in the catalog's words: the closed list of the catalog's dtypes, restated
// here because the plan imports no document type. A test in the kernel holds the two lists equal.
// The zero value is not stated.
type Dtype string

const (
	DtypeBF16  Dtype = "bf16"
	DtypeFP16  Dtype = "fp16"
	DtypeFP32  Dtype = "fp32"
	DtypeFP8   Dtype = "fp8"
	DtypeINT8  Dtype = "int8"
	DtypeINT4  Dtype = "int4"
	DtypeNVFP4 Dtype = "nvfp4"
	DtypeMXFP4 Dtype = "mxfp4"
)

// Dtypes lists the words, in the catalog's order.
func Dtypes() []Dtype {
	return []Dtype{DtypeBF16, DtypeFP16, DtypeFP8, DtypeNVFP4, DtypeMXFP4, DtypeINT8, DtypeINT4, DtypeFP32}
}

// Bits is the size of one element's payload, 0 for a word that is not a dtype.
func (d Dtype) Bits() int {
	switch d {
	case DtypeFP32:
		return 32
	case DtypeBF16, DtypeFP16:
		return 16
	case DtypeFP8, DtypeINT8:
		return 8
	case DtypeINT4, DtypeNVFP4, DtypeMXFP4:
		return 4
	}
	return 0
}

// Valid reports whether d is one of the words.
func (d Dtype) Valid() bool { return d.Bits() > 0 }

// PrimitiveShape is the geometry of a lowered primitive. Which fields apply depends on the
// primitive; the others are zero. Every figure is per rank unless its name says otherwise.
type PrimitiveShape struct {
	// M, N, K are a matrix multiply's dimensions: M rows per token of the step (the step's
	// token count multiplies it), N and K the weight's. A grouped multiply states Experts.
	M, N, K int
	Experts int
	// Heads, KVHeads and HeadDim are an attention primitive's geometry on this rank.
	Heads, KVHeads, HeadDim int
	// Window bounds context-dependent work in tokens; 0 is unbounded.
	Window int
	// Format is the stored format of the weight or the cache this primitive reads: its size, in
	// bits and scale blocks. It does not say which number format those bits hold, so BF16 and FP16
	// are both {PayloadBits 16}, and FP8 and INT8 both 8 bits.
	Format units.QuantFormat
	// Dtype is the element type of the cache an attention primitive reads, one of the catalog's
	// dtypes (model.DType: bf16, fp16, fp8, int8, ...); the zero value is not stated. Only an
	// attention primitive states it, and a stated Format must be the size of the type (a bf16
	// cache is 16 bits). The cost model prices a coefficient measured on one element type only for
	// that type, which Format cannot tell.
	Dtype Dtype
}

// Collective is a collective's group, message and whether it leaves its node.
type Collective struct {
	// Group is the number of ranks in the group.
	Group int
	// Bytes is the message each rank contributes.
	Bytes units.PerRank[units.Bytes]
	// CrossesNodes is true when the group spans more than one node.
	CrossesNodes bool
}

// KVGroup is one KV-cache group.
type KVGroup struct {
	// BytesPerPage and TokensPerPage describe a page; a sliding-window group states Window.
	BytesPerPage  units.Bytes
	TokensPerPage units.Tokens
	// Window is the sliding-window length in tokens; 0 is full attention.
	Window units.Tokens
}

// KVTier is a KV tier below GPU memory.
type KVTier struct {
	// Device is the catalog device's name, such as host DRAM or local NVMe.
	Device string
	// Bytes is the capacity per rank, as the engine sized it.
	Bytes units.PerRank[units.Bytes]
	// Shared is true when one copy serves the node rather than one per rank. A shared figure
	// counts once per node (SUB-14).
	Shared bool
}

// GraphPolicy says which launch shapes the engine captures.
type GraphPolicy struct {
	// Mode is how captured graphs are keyed: by token count, by batch size, or none.
	Mode GraphMode
	// Buckets are the captured sizes, in tokens or in requests according to Mode.
	Buckets []int
}

// GraphMode is how captured graphs are keyed.
type GraphMode string

const (
	GraphNone      GraphMode = ""
	GraphByTokens  GraphMode = "tokens"
	GraphByBatch   GraphMode = "batch"
	GraphPiecewise GraphMode = "piecewise"
)

// Valid reports whether m is a known mode.
func (m GraphMode) Valid() bool {
	switch m {
	case GraphNone, GraphByTokens, GraphByBatch, GraphPiecewise:
		return true
	}
	return false
}

// DraftPolicy is speculative decoding. A zero value is no speculation.
type DraftPolicy struct {
	// Chain is the draft length of a chain; Tree is top-k and steps of a tree, both 0 for none.
	Chain     int
	TreeTopK  int
	TreeSteps int
	// FirstPassRows and PassRequests state how an autoregressive drafter's passes run: which
	// rows its first pass processes, and for which requests the head and the later passes run.
	// They are the engine's execution decisions, resolved by the engine and stated here in
	// neutral words, so the cost model prices what it is told and resolves nothing (SUB-6).
	// Empty means not stated; a pool that speculates through an autoregressive drafter and
	// leaves either empty is refused by the cost model, and a parallel-block drafter, whose
	// target-wide work is its context stack, needs neither.
	FirstPassRows FirstPassRows
	PassRequests  PassRequests
}

// FirstPassRows is which rows an autoregressive drafter's first pass processes. Every pass
// after the first processes one row per drafting request.
type FirstPassRows string

// The words FirstPassRows takes. Empty is not stated.
const (
	// FirstPassPerRequest: the first pass processes one row per request, like every later pass.
	FirstPassPerRequest FirstPassRows = "per_request"
	// FirstPassScheduled: the first pass processes every token the target scheduled this step on
	// this rank, prefill tokens included, because the drafter writes its own KV for each.
	FirstPassScheduled FirstPassRows = "scheduled"
)

// Valid reports whether r is a stated word.
func (r FirstPassRows) Valid() bool { return r == FirstPassPerRequest || r == FirstPassScheduled }

// PassRequests is for which requests an autoregressive drafter's head and later passes run.
type PassRequests string

// The words PassRequests takes. Empty is not stated.
const (
	// PassRequestsSampling: the head and the later passes run for the requests that complete
	// their prompt this step.
	PassRequestsSampling PassRequests = "sampling"
	// PassRequestsAll: the head and the later passes run for every request of the step, a
	// request still in the middle of its prompt included.
	PassRequestsAll PassRequests = "all"
)

// Valid reports whether r is a stated word.
func (r PassRequests) Valid() bool { return r == PassRequestsSampling || r == PassRequestsAll }

// Enabled reports whether any speculation is configured.
func (d DraftPolicy) Enabled() bool { return d.Chain > 0 || d.TreeTopK > 0 || d.TreeSteps > 0 }

// HostPlan is per-request and per-token host work the engine runs: admission, sampling,
// detokenizing. The cost model prices it from host-overhead coefficients; the plan says which
// lane applies. The zero value is the default lane.
type HostPlan struct {
	// NativeFrontend is true when request admission and detokenizing run in a compiled
	// frontend, whose per-token cost differs from an interpreted one's.
	NativeFrontend bool
}

// ResidentTensor is one tensor resident on each rank after weight post-processing.
type ResidentTensor struct {
	// Name identifies the tensor in the model graph.
	Name string
	// Elements is the element count on this rank, and Format the stored format.
	Elements int64
	Format   units.QuantFormat
	// Where says whether the tensor is in GPU memory or in host memory, and Shared whether the
	// engines on a node share one copy of a host-resident tensor.
	Where  Residence
	Shared bool
}

// Residence is where a resident tensor lives.
type Residence string

const (
	ResidentGPU  Residence = ""
	ResidentHost Residence = "host"
)

// Valid reports whether r is a known residence.
func (r Residence) Valid() bool { return r == ResidentGPU || r == ResidentHost }

// Bytes returns the bytes the tensor occupies on one rank.
func (t ResidentTensor) Bytes() (units.PerRank[units.Bytes], error) {
	b, err := t.Format.StoredBytes(t.Elements)
	if err != nil {
		return units.PerRank[units.Bytes]{}, fmt.Errorf("tensor %q: %w", t.Name, err)
	}
	return units.Rank(b), nil
}

// Buffer is a buffer or workspace an engine chooses, sized as its upstream sizes it.
type Buffer struct {
	// Name says what the buffer is, from the engine's own function that sizes it.
	Name string
	// Bytes is its size, per rank or per node according to Basis.
	Bytes units.Bytes
	Basis Basis
	// Host is true for a pinned host buffer.
	Host bool
}

// Basis says what a sized figure is per.
type Basis string

const (
	BasisRank  Basis = "rank"
	BasisNode  Basis = "node"
	BasisGroup Basis = "group"
)

// Valid reports whether b is a known basis.
func (b Basis) Valid() bool { return b == BasisRank || b == BasisNode || b == BasisGroup }

// StepShape is one engine step on the data-parallel ranks that run in lockstep, already padded
// by the engine. The cost model prices the padded shape; the engine decides the padding.
type StepShape struct {
	Mode StepMode
	// Ranks has one entry per data-parallel rank in the lockstep group.
	Ranks []RankStep
	// Padded has one entry per rank: the shape actually launched.
	Padded []PaddedShape
	// Drafts is the tokens drafted and verified this step.
	Drafts DraftShape
	// Stage is the pipeline stage; 0 without pipeline parallelism.
	Stage int
	// DecodeQueryLen is the most tokens a request may schedule and still be priced as a decode:
	// the engine's own decode threshold, 1 plus the draft length under speculation and 1
	// without. A request that schedules more is priced as a prefill chunk. The engine states it
	// because only the engine knows which of its requests its attention kernels treat as a
	// decode, and the cost model resolves nothing (SUB-6). The zero value means 1.
	DecodeQueryLen units.Tokens
}

// DecodeWidth returns DecodeQueryLen with its floor applied: a step has no request that
// decodes in fewer than one token, so zero reads as one.
func (s StepShape) DecodeWidth() units.Tokens {
	if s.DecodeQueryLen < 1 {
		return 1
	}
	return s.DecodeQueryLen
}

// StepMode is what one step contains.
type StepMode string

const (
	StepMixed      StepMode = "mixed"
	StepExtendOnly StepMode = "extend_only"
	StepDecodeOnly StepMode = "decode_only"
	StepVerify     StepMode = "verify"
	// StepPrebuilt is a decode merge of a batch whose KV arrived from a prefill pool: the step
	// runs no forward pass and is priced at no compute, but still has scheduler and host cost.
	StepPrebuilt StepMode = "prebuilt"
	StepIdle     StepMode = "idle"
)

// Valid reports whether m is a known mode.
func (m StepMode) Valid() bool {
	switch m {
	case StepMixed, StepExtendOnly, StepDecodeOnly, StepVerify, StepPrebuilt, StepIdle:
		return true
	}
	return false
}

// RankStep is one data-parallel rank's requests in a step.
type RankStep struct {
	Requests []RequestStep
}

// RequestStep is one request's contribution to a step.
type RequestStep struct {
	// Scheduled is tokens scheduled this step. Under speculation it is the draft length plus
	// one, so a value of one does not imply a decode.
	Scheduled units.Tokens
	// Computed is tokens already computed; Prompt is the prompt length. Computed against Prompt
	// separates a chunked-prefill tail from a decode.
	Computed units.Tokens
	Prompt   units.Tokens
	// Cached is the prefix-cache hit, already subtracted from Scheduled.
	Cached units.Tokens
}

// PaddedShape is the shape one rank actually launches.
type PaddedShape struct {
	// Tokens is the launched token count after graph-bucket and data-parallel padding.
	Tokens units.Tokens
	// Launch says how the step launches; the three cost differently, so the engine states which.
	Launch Launch
}

// Launch is how a step launches.
type Launch string

const (
	LaunchEager     Launch = ""
	LaunchGraph     Launch = "graph"
	LaunchPiecewise Launch = "piecewise"
)

// Valid reports whether l is a known launch.
func (l Launch) Valid() bool { return l == LaunchEager || l == LaunchGraph || l == LaunchPiecewise }

// DraftShape is the speculative tokens in a step. Zero is none.
type DraftShape struct {
	Drafted  units.Tokens
	Verified units.Tokens
	// Passes is the drafter's launch on each data-parallel rank, in the order of StepShape.Ranks.
	// It is empty when the step runs no drafter pass. A drafter coordinates its own padding across
	// the lockstep group, apart from the target's, so a rank's draft pass can be wider than its own
	// draft rows, and the engine states how wide (the target's padding is PaddedShape).
	Passes []DraftPass
}

// DraftPass is one rank's drafter launch. The zero value is no drafter pass.
type DraftPass struct {
	// Padded is the token count the drafter launches after its graph bucket and the group's
	// agreed width, at least the rank's own draft rows.
	Padded units.Tokens
	// Launch says how the drafter launches: a captured graph, a piecewise replay or eager.
	Launch Launch
}

// Transfer is one KV move, with enough for the cost model to say what each link must carry.
// The engine states it, as the Move of the engine.Transfer it reports; the simulator hands it
// to Environment.TransferDemand unchanged.
//
// For a move to or from a tier, both ends are on the node whose memory or storage holds the
// tier. For a move between pools, each end is in its own pool's numbering: the engine states its
// own end, and the far end is the peer the request named (engine.Request.Peer).
//
// A move between pools may carry no tokens (Tokens 0): a remote prefill can find all of its KV
// already here. Its demand is zero bytes, with no link and no floor. A move to or from a tier
// always carries tokens; Validate refuses an empty one.
//
// Two limits, both deferred to M3. A Transfer has one From and one To, so a pool that spans
// nodes cannot say that every node's workers move their own shard at once
// (inference-sim/blis-substrate#48): today the engine states the node that holds the tier (the
// pool's first node when each node holds one), and the cost model prices that one node's link.
// And a Transfer has one Tokens count, so the blocks each KV cache group moves, which differ
// in a hybrid model, are not stated (inference-sim/blis-substrate#49): they are lost at this
// boundary, and the demand is for Tokens at the pool's page size.
type Transfer struct {
	// Tokens moved, and the node and rack at each end: where the bytes leave and where they land.
	Tokens units.Tokens
	From   Endpoint
	To     Endpoint
	// Kind says which tier or peer the bytes move to or from.
	Kind TransferKind
	// Tier names the catalog device for an offload move; "" for a move between pools.
	Tier string
}

// Endpoint names where a transfer starts or ends, by node and rack. A transfer within one
// node uses no NIC (SUB-17).
type Endpoint struct {
	Node int
	Rack int
}

// TransferKind says what a move is.
type TransferKind string

const (
	TransferToTier   TransferKind = "to_tier"
	TransferFromTier TransferKind = "from_tier"
	TransferPeer     TransferKind = "peer"
)

// Valid reports whether k is a known kind.
func (k TransferKind) Valid() bool {
	return k == TransferToTier || k == TransferFromTier || k == TransferPeer
}
