// Package plan holds the boundary between an engine and the cost model (design §3.3, §6.3):
// the PoolPlan an engine hands over once per pool, the StepShape it forms each step, and the
// Transfer it returns for each KV move. The engine resolves its configuration, lowers the
// model and sizes its KV budget; the cost model prices what it is given and knows no engine.
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
	// Library names the library, such as a named attention library. The fixed list of library
	// names is added with the exact kernel-implementation lookup (VAL-8); until then a plan
	// requires only that it is stated.
	Library string
	// Variant is the library's own exported kernel name; an engine's backend names map to it in
	// the engine's lowering.
	Variant string
	// CodeHash is the hash of the kernel's source and target GPU architecture, or of a binary
	// library's file and architecture.
	CodeHash string
	// Toolkit is the compiler or CUDA toolkit version the kernel was built with. A toolkit
	// change can change a kernel's speed without changing its source; the landing gate shows
	// whether the old value still holds.
	Toolkit string
}

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
	// Format is the stored format of the weight or the cache this primitive reads.
	Format units.QuantFormat
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
}

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
}

// Transfer is one KV move, with enough for the cost model to say what each link must carry.
type Transfer struct {
	// Tokens moved, and the pool-relative ranks that send and receive.
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
