package model

// TPLayout is how one node's work and weights divide across tensor-parallel ranks.
//
// A consumer without it has to guess, and the guess is wrong exactly where it matters.
// blis-latency-kernel infers a GEMM's split from whether k equals the hidden size and
// divides every other node by tp. DeepSeek-V4.1-Flash on the engine runs a large share
// of each layer replicated on every rank: the fused q/kv A-projection
// (MergedColumnParallelLinear with disable_tp=True, "fused ReplicatedLinear"), the
// compressor, the indexer's projections ("no tensor parallel, just replicated"), the
// router gate (GateLinear, a ReplicatedLinear), and all of mHC, whose FP32 parameters are
// plain tensors with no split. Dividing those by tp prices one tp-th of the work.
//
// Absent means the consumer's inference, so a graph opts in per node.
type TPLayout string

const (
	// TPDefault is the zero value: the consumer infers the layout.
	TPDefault TPLayout = ""
	// TPColumn, on a GEMM: the output width n splits across ranks
	// (ColumnParallelLinear). Each rank holds n/tp x k and computes its share.
	TPColumn TPLayout = "column"
	// TPRow, on a GEMM: the input width k splits across ranks (RowParallelLinear), and
	// the partial outputs are summed by a collective the graph states as its own node.
	TPRow TPLayout = "row"
	// TPSharded, on an Elementwise or Attention node: the work splits across ranks with
	// the heads or columns it operates on.
	TPSharded TPLayout = "sharded"
	// TPReplicated: every rank holds the node's whole weight and does the whole node's
	// work for every token it holds. Under a layout that shards tokens across the
	// tensor-parallel ranks (sequence parallelism), each rank holds fewer tokens; the
	// weight is still whole, and the token split is the consumer's to resolve from the
	// layout.
	TPReplicated TPLayout = "replicated"
)

// tpLayoutsByOp lists the layouts each op accepts. A GEMM splits along one of its two
// dimensions or not at all; an Elementwise or Attention node is sharded with its heads
// or replicated. Other ops have no per-node choice: a GroupedGEMM's split is decided by
// the expert- and tensor-parallel layout together (V4.1's shared expert is TP-sharded
// normally and replicated under sequence parallelism, deepseek_v4/nvidia/model.py), and
// a collective is defined by its group. A Gather states how its table splits with shard,
// which says more than a layout can: whether each token's row lives on one rank or each
// rank reads its own heads.
var tpLayoutsByOp = map[Op]map[TPLayout]bool{
	OpGEMM:        {TPColumn: true, TPRow: true, TPReplicated: true},
	OpElementwise: {TPSharded: true, TPReplicated: true},
	OpAttention:   {TPSharded: true, TPReplicated: true},
}

// ValidFor reports whether l is a layout op o accepts. The default is valid for every op.
func (l TPLayout) ValidFor(o Op) bool {
	return l == TPDefault || tpLayoutsByOp[o][l]
}
