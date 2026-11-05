// Package model defines a model as a directed acyclic graph of cost primitives.
//
// The representation is deliberately not a vendor configuration. blis-catalog
// already commits each model's config.json verbatim, and that file speaks one
// provider's dialect: hidden_size here, d_model there, n_embd elsewhere; a Mamba
// layer named by mamba_expand in one repo and by ssm_cfg in another. Re-spelling
// those fields would tie this schema to whichever vendor happened to define a
// model first, and would still not say what the GPU does.
//
// A ModelGraph says what the GPU does. Each node is one of the cost model's
// primitives, carrying the shape parameters that primitive prices and nothing
// else. Deriving a graph from a vendor config is a separate, mechanical step whose
// output is committed alongside the config with a digest, so a drift between the
// two is detectable rather than silent.
//
// Two rules keep the representation honest. A node exists when it launches GPU
// work, not when a config field exists: an architecture-specific scalar such as
// Granite's attention_multiplier replaces a softmax scale rather than adding an
// operation, so it changes no node. And a shape is derived the way the engine
// derives it: head_dim is hidden_size/num_heads, computed, because many configs
// never state it.
package model

// Op names a cost primitive. The ten members are the primitives that run inside
// a forward pass. The cost model's other five — WeightRead, HostOverhead,
// HostTransfer, StorageRead and P2P — are not graph nodes: the first is implied by
// the GEMMs, and the rest are priced by kernel methods the simulator calls
// directly rather than by traversing a layer.
type Op string

const (
	// OpGEMM is a dense matrix multiply: projections, the LM head.
	OpGEMM Op = "GEMM"
	// OpGroupedGEMM is the MoE expert matmul, one group per active expert.
	OpGroupedGEMM Op = "GroupedGEMM"
	// OpAttention is attention over a batch. Kind selects the cost structure.
	OpAttention Op = "Attention"
	// OpRecurrentUpdate is a linear-attention or state-space layer, whose state is
	// fixed-size in context rather than growing per token.
	OpRecurrentUpdate Op = "RecurrentUpdate"
	// OpElementwise is normalization, rope, activation, quantize/dequantize:
	// bandwidth-bound work with no reduction across ranks.
	OpElementwise Op = "Elementwise"
	// OpGather reads a few rows of a large table per token: a vocabulary embedding, or
	// DeepSeek-V4.1's Engram n-gram tables, 384 million rows per layer of which each
	// token reads 24. It is a primitive of its own because no other prices it: a GEMM
	// charges its whole n x k as per-step reads and FLOPs, which for a gather is wrong by
	// the ratio of the table to the rows read, and an Elementwise node has no table and
	// so no residency. The table may also live off the device (Residency).
	OpGather Op = "Gather"
	// OpAllReduce is a tensor-parallel reduction.
	OpAllReduce Op = "AllReduce"
	// OpAllGather and OpReduceScatter are the sequence-parallel pair, and the two
	// phases of an all-gather-family MoE dispatch.
	OpAllGather     Op = "AllGather"
	OpReduceScatter Op = "ReduceScatter"
	// OpAll2All is a routed MoE dispatch or combine. Named for the primitive, not
	// for the backend: an all-gather-family backend moves ring-shaped volume and
	// scales differently across nodes, which is why Scenario carries the backend
	// and this node does not assume one.
	OpAll2All Op = "All2All"
)

var ops = map[Op]bool{
	OpGEMM: true, OpGroupedGEMM: true, OpAttention: true, OpRecurrentUpdate: true,
	OpElementwise: true, OpGather: true, OpAllReduce: true, OpAllGather: true,
	OpReduceScatter: true, OpAll2All: true,
}

// Valid reports whether o is a recognized op.
func (o Op) Valid() bool { return ops[o] }

// Collective reports whether o moves bytes between ranks. A collective node
// carries a When condition, because whether it exists at all depends on the
// layout: a tp=1, ep=1 instantiation drops every one of them.
func (o Op) Collective() bool {
	switch o {
	case OpAllReduce, OpAllGather, OpReduceScatter, OpAll2All:
		return true
	}
	return false
}

// AttentionKind selects a cost structure within OpAttention. The distinction is
// about how much state a token occupies and how it shards, not about a vendor's
// naming: MLA stores one latent vector per token with no separate V and cannot be
// sharded below one head, where GQA stores per-KV-head K and V and shards with
// tensor parallelism until it runs out of heads.
type AttentionKind string

const (
	// AttentionGQA covers grouped-query and multi-head attention, which differ
	// only in how many query heads share a KV head.
	AttentionGQA AttentionKind = "gqa"
	// AttentionMLA is multi-head latent attention: one latent vector per token.
	AttentionMLA AttentionKind = "mla"
	// AttentionSparseMLA narrows the latent-cache read: a layer may select a top-k
	// of the cache (index_topk), compress the latent read (compress_ratio), bound it to
	// a sliding window (window), or combine them. All are the same latent-cache cost
	// structure narrowed by different mechanisms, run by one sparse kernel that reads
	// an explicit set of positions, so they are one kind rather than separate
	// compressed- or windowed-MLA kinds that would duplicate the MLA shape and sharding
	// rules. DeepSeek-V4.1's window-only layers are sparse_mla with a window alone.
	AttentionSparseMLA AttentionKind = "sparse_mla"
	// AttentionSWA is sliding-window attention, whose per-token read is bounded by
	// the window rather than by context length.
	AttentionSWA AttentionKind = "swa"
)

var attentionKinds = map[AttentionKind]bool{
	AttentionGQA: true, AttentionMLA: true, AttentionSparseMLA: true,
	AttentionSWA: true,
}

// Valid reports whether k is a recognized attention kind.
func (k AttentionKind) Valid() bool { return attentionKinds[k] }

// LatentKV reports whether the kind stores one latent vector per token rather
// than per-KV-head K and V. It is the property that decides whether tensor
// parallelism reduces KV bytes, so callers ask this rather than comparing kinds.
func (k AttentionKind) LatentKV() bool {
	return k == AttentionMLA || k == AttentionSparseMLA
}

// RecurrentKind selects a cost structure within OpRecurrentUpdate. Every member
// keeps constant-size state per sequence; they differ in what that state contains
// and therefore in how many bytes it occupies.
type RecurrentKind string

const (
	// RecurrentMamba2 carries a convolutional state and a temporal (SSM) state.
	RecurrentMamba2 RecurrentKind = "mamba2"
	// RecurrentKDA is Kimi Delta Attention: a single matrix-valued state.
	RecurrentKDA RecurrentKind = "kda"
	// RecurrentGDN is Gated DeltaNet, the same shape family as KDA.
	RecurrentGDN RecurrentKind = "gdn"
)

var recurrentKinds = map[RecurrentKind]bool{
	RecurrentMamba2: true, RecurrentKDA: true, RecurrentGDN: true,
}

// Valid reports whether k is a recognized recurrent kind.
func (k RecurrentKind) Valid() bool { return recurrentKinds[k] }
