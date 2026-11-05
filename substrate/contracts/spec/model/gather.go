package model

// Residency is where a gathered table lives.
type Residency string

const (
	// ResidencyDevice: the table is in GPU memory. An empty residency means this.
	ResidencyDevice Residency = "device"
	// ResidencyHost: the table is in pinned host memory and the GPU reads its rows over
	// the host link. DeepSeek-V4.1's Engram tables live here by default -- 264 B rows,
	// about 189 GiB across its two Engram layers -- read through UVA (the engine's
	// models/deepseek_v41/nvidia/engram.py).
	ResidencyHost Residency = "host"
)

var residencies = map[Residency]bool{"": true, ResidencyDevice: true, ResidencyHost: true}

// Valid reports whether r is a recognized residency. The empty value is valid and means
// device.
func (r Residency) Valid() bool { return residencies[r] }

// Effective returns the residency with its default applied, so a reader compares one
// spelling: the empty value is ResidencyDevice.
func (r Residency) Effective() Residency {
	if r == "" {
		return ResidencyDevice
	}
	return r
}

// ResidencyControl names the engine setting that decides a table's residency, so a
// consumer knows which deployment field to read. It is a closed list for the reason
// EmitCondition is: the graph states which question to ask, never how to answer it.
type ResidencyControl string

const (
	// ResidencyFixed is the zero value: no setting moves the table.
	ResidencyFixed ResidencyControl = ""
	// ResidencyEngramCPUOffload: the engine's EngramConfig.cpu_offload, set through
	// --engram-config (config/engram.py) and stated in a deployment as
	// engine.engram_config.cpu_offload. True keeps the table in host memory and false in
	// HBM. Its default is an engine-release fact, so the rules pack, not this schema,
	// checks that a graph's stated residency is the release's default.
	ResidencyEngramCPUOffload ResidencyControl = "engram_cpu_offload"
)

var residencyControls = map[ResidencyControl]bool{
	ResidencyFixed: true, ResidencyEngramCPUOffload: true,
}

// Valid reports whether c is a recognized control.
func (c ResidencyControl) Valid() bool { return residencyControls[c] }

// GatherShard is how a gathered table divides across tensor-parallel ranks, which
// decides both what each rank holds and how many rows each rank reads per token.
type GatherShard string

const (
	// ShardReplicated: every rank holds the whole table and reads every row. An empty
	// shard means this.
	ShardReplicated GatherShard = "replicated"
	// ShardRows: the table's rows split evenly across ranks, and each token's row is
	// read by the one rank that holds it; the others contribute zeros to a reduction.
	// the engine's VocabParallelEmbedding divides the padded vocabulary this way and ends
	// with a tensor-parallel all-reduce, which the graph states as its own collective
	// node after the gather.
	ShardRows GatherShard = "rows"
	// ShardHeads: the table is rows_per_token sub-tables, one per hash head, split across
	// S shards, each holding ceil(rows_per_token / S) whole heads and reading one row from
	// each for every token (common/engram.py: part_n_hash_cols = cdiv(n_hash_cols,
	// num_shards)). S is the tensor-parallel width, or that times the node-local
	// data-parallel replicas when they split the table rather than share one copy; the
	// deployment's Engram config decides which (in the pool's engine block). The heads'
	// sub-tables are of near-equal prime sizes, so a shard's share of rows is its share
	// of heads.
	ShardHeads GatherShard = "heads"
)

var gatherShards = map[GatherShard]bool{
	"": true, ShardReplicated: true, ShardRows: true, ShardHeads: true,
}

// Valid reports whether s is a recognized shard. The empty value is valid and means
// replicated.
func (s GatherShard) Valid() bool { return gatherShards[s] }

// Effective returns the shard with its default applied, so a reader compares one
// spelling: the empty value is ShardReplicated.
func (s GatherShard) Effective() GatherShard {
	if s == "" {
		return ShardReplicated
	}
	return s
}
