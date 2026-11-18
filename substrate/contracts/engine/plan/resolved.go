package plan

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/units"
)

// The settings an engine resolves that the cost model prices, stated in neutral words (design
// §6.1, SUB-6). The engine's lowering works out each one from its own settings and defaults, and
// the cost model prices what the plan says: it applies no default and reads no engine setting.
// Each field's zero value is stated in its comment.

// BatchLimits is the most one step may schedule, as the engine resolved it. Zero is not stated:
// the cost model refuses to price a term that needs a limit the plan leaves out.
type BatchLimits struct {
	// Requests is the most requests one step may hold.
	Requests int
	// Tokens is the most tokens one step may schedule.
	Tokens units.Tokens
}

// WeightLayout is the format each class of a checkpoint's weights is served in, in the catalog's
// dtype words (model.DType: bf16, fp8, nvfp4, ...). A class the model does not have is empty.
type WeightLayout struct {
	// Linear is the format of every dense projection and of the shared experts.
	Linear string
	// LinearScaleElements is how many weights share one resident scale byte (32 for MXFP8); zero
	// for none.
	LinearScaleElements int
	// RoutedExperts is the format of the routed experts.
	RoutedExperts string
	// Router is the format of the router, the GEMM with one row per routed expert.
	Router string
	// Head is the format of the output head.
	Head string
	// Embedding is the format of the token embedding table.
	Embedding string
}

// CacheFormat is how the engine stores KV and recurrent state, as its attention backends
// allocate it.
type CacheFormat struct {
	// Record is the key the catalog's cache records are looked up by (model.Cache.EntryBytes):
	// the format the KV cache is stored in, as the engine's attention backend resolves it. "" is
	// the model's own format.
	Record string
	// ElementBytes is the bytes of one KV element of a layer whose caches the catalog does not
	// state; 0 is not stated.
	ElementBytes float64
	// LatentCellBytes is the bytes one token of a latent-attention layer takes when the backend
	// packs the latent and its scales into one cell; 0 when the per-element width applies.
	LatentCellBytes float64
	// StateElementBytes is the bytes of one element of a recurrent layer's state; 0 is not
	// stated.
	StateElementBytes float64
	// Recurrent is how many states a recurrent layer keeps for a request.
	Recurrent RecurrentStates
	// TrackingSlots is, under RecurrentTracked, how many slots beside its working slot a request
	// holds to snapshot its state for the prefix cache; it is 0 under every other word.
	TrackingSlots int
}

// RecurrentStates is how many states a recurrent layer keeps for one request.
type RecurrentStates string

const (
	// RecurrentLatest keeps the latest state only, so a cached prefix cannot resume it.
	RecurrentLatest RecurrentStates = ""
	// RecurrentPageAligned keeps one state at the end of each full KV page, so a prefix that
	// ends on a page boundary resumes from it.
	RecurrentPageAligned RecurrentStates = "page_aligned"
	// RecurrentEveryPage keeps a state for every KV page, so the state grows with the context.
	RecurrentEveryPage RecurrentStates = "every_page"
	// RecurrentTracked keeps a working state and TrackingSlots more a request holds beside it to
	// snapshot the state at intervals of the engine's own, so a cached prefix resumes from a
	// snapshot. SGLang's mamba radix cache with the extra_buffer strategy holds two such slots
	// with the overlap schedule and one without it or with extra_buffer_lazy
	// (get_mamba_tracking_slots, mem_cache/allocation_sizing.py:11-19@297e54fd).
	RecurrentTracked RecurrentStates = "tracked"
)

// Valid reports whether r is a known word.
func (r RecurrentStates) Valid() bool {
	return r == RecurrentLatest || r == RecurrentPageAligned || r == RecurrentEveryPage || r == RecurrentTracked
}

// MoEPlan is how the engine runs a mixture-of-experts layer.
type MoEPlan struct {
	// Dispatch is the all-to-all kernel that moves tokens to their experts and back, when one
	// runs; the zero value is none, the experts then running on gathered or replicated tokens.
	Dispatch KernelImpl
	// Fused is true when the dispatch, the expert GEMMs and the combine run as one fused kernel.
	Fused bool
	// Experts is the kernel the engine runs the routed experts' grouped GEMMs with, as its MoE
	// backend selection resolves it (a stated backend, or the engine's own choice for the weight
	// format, part and parallelism), with the build it runs where the engine's kernel table
	// identifies it (CodeHash, Toolkit). The cost model prices the routed term on the ramp
	// measured for that kernel (ExpertsTriton, ExpertsFlashInferCutlass name the kernels) where the
	// registry carries one, and binds the coefficients to the build (VAL-8). The zero value is not
	// stated, and the routed term is then priced by weight format alone.
	Experts KernelImpl
	// RedundantExperts is how many extra copies of experts the engine places beside the model's
	// own to balance the load; 0 is none.
	RedundantExperts int
	// Routing is whether expert selection (top-k, or grouped top-k with a score-correction
	// bias) is a launch of its own in each routed MoE layer. It depends on the experts backend
	// the engine resolves, not on the checkpoint: a modular backend runs router.select_experts
	// first, a monolithic one takes the router logits and selects inside its kernel. The zero
	// value is not stated, and no selection launch is counted.
	Routing MoERouting
}

// MoERouting says where an MoE layer's expert selection runs.
type MoERouting string

const (
	// RoutingNotStated counts no launch for expert selection.
	RoutingNotStated MoERouting = ""
	// RoutingSeparate runs expert selection as its own kernel before the experts, one launch a
	// routed MoE layer.
	RoutingSeparate MoERouting = "separate"
	// RoutingFused selects experts inside a monolithic experts kernel: no launch of its own.
	RoutingFused MoERouting = "fused"
)

// Valid reports whether r is a known word.
func (r MoERouting) Valid() bool {
	return r == RoutingNotStated || r == RoutingSeparate || r == RoutingFused
}

// The MoE dispatch kernels a cost model prices by name (MoEPlan.Dispatch): the three kernel
// families of the DeepEP library. Any other dispatch kernel is named by its library alone.
var (
	DispatchDeepEPHighThroughput = KernelImpl{Library: "deepep", Variant: "high_throughput"}
	DispatchDeepEPLowLatency     = KernelImpl{Library: "deepep", Variant: "low_latency"}
	DispatchDeepEPV2             = KernelImpl{Library: "deepep", Variant: "v2"}
)

// The grouped-expert kernels a cost model prices by name (MoEPlan.Experts), by the kernel's own
// name: Triton's fused grouped GEMM (fused_moe_kernel) and FlashInfer's CUTLASS fused MoE. An
// engine states the build it runs in full (library, code hash, toolkit), and the cost model binds a
// coefficient to that build (design VAL-8); these name the kernels its ramps are measured for. Any
// other experts kernel is priced by its weight format.
var (
	ExpertsTriton            = KernelImpl{Variant: "fused_moe_kernel"}
	ExpertsFlashInferCutlass = KernelImpl{Library: "flashinfer", Variant: "cutlass_fused_moe"}
)

// CollectivePlan says which collectives a layer runs, where the engine decides it.
type CollectivePlan struct {
	// SequenceParallelMoE is true when an MoE layer's input is split by token across the
	// tensor-parallel ranks, so its all-reduce becomes a reduce-scatter and an all-gather.
	SequenceParallelMoE bool
	// SequenceParallelModel is true when the model's own residual stream is split by token across
	// the tensor-parallel ranks, so each attention all-reduce becomes an all-gather and a
	// reduce-scatter.
	SequenceParallelModel bool
	// AllReduce is the kernel the engine resolved for the tensor-parallel all-reduce, after
	// every request it declined (an unsupported width, a group that leaves the node). The cost
	// model prices the lane measured for that kernel and resolves nothing itself (SUB-6).
	// Unstated is read from ReportedSettings.SMAllReduce: true is AllReduceCustom, false is
	// AllReduceNCCL, which is what an engine that predates this field states.
	AllReduce AllReduceBackend
}

// AllReduceBackend is the kernel that runs a tensor-parallel all-reduce, in neutral words.
type AllReduceBackend string

const (
	// AllReduceUnstated leaves the choice to ReportedSettings.SMAllReduce.
	AllReduceUnstated AllReduceBackend = ""
	// AllReduceCustom is the engine's own streaming-multiprocessor kernel over peer memory.
	AllReduceCustom AllReduceBackend = "custom"
	// AllReduceNCCL is the communication library's all-reduce.
	AllReduceNCCL AllReduceBackend = "nccl"
	// AllReduceFlashInferTRTLLM is FlashInfer's all-reduce on the TensorRT-LLM kernels.
	AllReduceFlashInferTRTLLM AllReduceBackend = "flashinfer_trtllm"
	// AllReduceFlashInferMNNVL is FlashInfer's all-reduce over multi-node NVLink.
	AllReduceFlashInferMNNVL AllReduceBackend = "flashinfer_mnnvl"
)

// Valid reports whether b is a known word; the unstated word is valid.
func (b AllReduceBackend) Valid() bool {
	switch b {
	case AllReduceUnstated, AllReduceCustom, AllReduceNCCL, AllReduceFlashInferTRTLLM, AllReduceFlashInferMNNVL:
		return true
	}
	return false
}

// UsesSMs reports whether the kernel runs on the GPU's streaming multiprocessors over peer
// memory (every word but NCCL); it is meaningful for a stated word only.
func (b AllReduceBackend) UsesSMs() bool { return b != AllReduceNCCL }

// DecodeContextPlan is how a decode-context-parallel group combines its partial attention and
// stripes a sequence's KV across its ranks.
type DecodeContextPlan struct {
	// Combine is the collectives that merge the ranks' partial outputs.
	Combine DecodeCombine
	// Interleave is how many consecutive tokens one rank holds before the next takes over; 0 is
	// read as 1.
	Interleave units.Tokens
}

// DecodeCombine is how a decode-context-parallel group merges its partial outputs.
type DecodeCombine string

const (
	// CombineGatherScatter all-gathers the log-sum-exp, then reduce-scatters the output.
	CombineGatherScatter DecodeCombine = ""
	// CombineAllToAll exchanges the packed partial output and log-sum-exp in one all-to-all.
	CombineAllToAll DecodeCombine = "all_to_all"
)

// Valid reports whether c is a known word.
func (c DecodeCombine) Valid() bool { return c == CombineGatherScatter || c == CombineAllToAll }

// EngramPlacement is where a model's Engram lookup tables live.
type EngramPlacement struct {
	// Host is true when the tables are in host memory and read over the host link.
	Host bool
	// Shards is how many ways one copy of a table is split across the ranks that read it; 0 is
	// read as 1.
	Shards int
}

// ValidateResolved refuses resolved settings no engine can mean: a negative count, a fraction
// outside (0, 1], a word outside its list.
func (p PoolPlan) ValidateResolved() error {
	var bad []string
	negative := func(field string, v int64) {
		if v < 0 {
			bad = append(bad, fmt.Sprintf("%s is %d; it is never negative", field, v))
		}
	}
	width := func(field string, v float64) {
		if v < 0 {
			bad = append(bad, fmt.Sprintf("%s is %v bytes; it is never negative", field, v))
		}
	}
	word := func(field string, v string, ok bool) {
		if !ok {
			bad = append(bad, fmt.Sprintf("%s %q is not a known word", field, v))
		}
	}
	negative("Batch.Requests", int64(p.Batch.Requests))
	negative("Batch.Tokens", int64(p.Batch.Tokens))
	negative("MaxContext", int64(p.MaxContext))
	negative("PageTokens", int64(p.PageTokens))
	if p.MemoryFraction < 0 || p.MemoryFraction > 1 {
		bad = append(bad, fmt.Sprintf("MemoryFraction is %v; it is a share of the GPU's memory, in "+
			"(0, 1], or 0 for not stated", p.MemoryFraction))
	}
	word("Cache.Recurrent", string(p.Cache.Recurrent), p.Cache.Recurrent.Valid())
	negative("Cache.TrackingSlots", int64(p.Cache.TrackingSlots))
	if tracked := p.Cache.Recurrent == RecurrentTracked; tracked != (p.Cache.TrackingSlots > 0) && p.Cache.TrackingSlots >= 0 {
		bad = append(bad, fmt.Sprintf("Cache.TrackingSlots is %d under Cache.Recurrent %q; the tracked "+
			"word states at least one slot and every other word states none", p.Cache.TrackingSlots, p.Cache.Recurrent))
	}
	width("Cache.ElementBytes", p.Cache.ElementBytes)
	width("Cache.LatentCellBytes", p.Cache.LatentCellBytes)
	width("Cache.StateElementBytes", p.Cache.StateElementBytes)
	negative("MoE.RedundantExperts", int64(p.MoE.RedundantExperts))
	word("MoE.Routing", string(p.MoE.Routing), p.MoE.Routing.Valid())
	word("Collectives.AllReduce", string(p.Collectives.AllReduce), p.Collectives.AllReduce.Valid())
	word("DecodeContext.Combine", string(p.DecodeContext.Combine), p.DecodeContext.Combine.Valid())
	negative("DecodeContext.Interleave", int64(p.DecodeContext.Interleave))
	negative("Engram.Shards", int64(p.Engram.Shards))
	word("Graphs.Decode", string(p.Graphs.Decode), p.Graphs.Decode.Valid())
	word("Graphs.Other", string(p.Graphs.Other), p.Graphs.Other.Valid())
	negative("Graphs.Ceiling", int64(p.Graphs.Ceiling))
	negative("Drafts.RetainedTokens", int64(p.Drafts.RetainedTokens))
	negative("InFlight", int64(p.InFlight))
	for i, c := range p.Launches {
		switch {
		case c.FromTokens < 1 || c.ToTokens < c.FromTokens:
			bad = append(bad, fmt.Sprintf("Launches[%d] covers %d to %d tokens; a range starts at 1 "+
				"token or more and ends at or after its start", i, c.FromTokens, c.ToTokens))
		case c.PerLayer == 0:
			bad = append(bad, fmt.Sprintf("Launches[%d] changes no launch; a range the engine "+
				"fuses nothing in is left out", i))
		}
		for j := range i {
			if o := p.Launches[j]; c.FromTokens <= o.ToTokens && o.FromTokens <= c.ToTokens {
				bad = append(bad, fmt.Sprintf("Launches[%d] (%d to %d tokens) overlaps Launches[%d] "+
					"(%d to %d); a step's launches change one way", i, c.FromTokens, c.ToTokens, j,
					o.FromTokens, o.ToTokens))
			}
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("the plan's resolved settings: %s", strings.Join(bad, "; "))
	}
	return nil
}

// Lowered is what an engine's lowering hands the cost model for one pool: the plan with every
// setting the engine resolved, and the record of how it resolved them. The cost model prices the
// plan and reports the record as it received it, pricing nothing from it.
type Lowered struct {
	Plan PoolPlan
	// Overrides are the settings the deployment states that the engine resolves to something
	// else, each with the reason.
	Overrides []Override
	// Notes are the values the engine took where the deployment stated none, each with the rule
	// that set it.
	Notes []Note
	// ServedLayout is the engine's statement of the format each class of weights is served in,
	// which the cost model checks against the catalog graph and refuses when they differ; nil
	// states none. It is a check, not a price: the plan's WeightFormat and the graph's own
	// formats are what the cost model prices.
	ServedLayout *WeightLayout
	// Reported are settings the engine resolved that the cost model reports in its resolution
	// (kernel.Resolution) as it received them, and prices nothing from.
	Reported ReportedSettings
	// Drafter is the catalog model whose graph is the drafter, when the drafter is a checkpoint of
	// its own; "" is the target graph's own speculator. The cost model checks that the drafter
	// graph it is given is this one, and prices that graph.
	Drafter string
	// DefaultCacheRecord is the record the engine stores a cache in when the deployment states
	// no cache format; "" is the model's own format (an unstated cache, "auto"), the record the
	// graph's base description of each cache assumes. The cost model reports a stated cache sized
	// at its base record only when the cache runs under a record other than this one; it prices
	// nothing from it.
	DefaultCacheRecord string
	// Encoder states what the engine does with the graph's towers. Nil states nothing: the cost
	// model prices the decoder alone, holds no tower weights and refuses an encode step. It is
	// read only for a graph that prices an encoder.
	Encoder *EncoderRun
}

// EncoderRun is what an engine does with a graph's towers for one pool.
type EncoderRun struct {
	// Resident is true when the engine builds the towers and runs them for a request with
	// images. False is a deployment that serves the decoder only: no tower weights are held
	// and no image can be encoded.
	Resident bool
	// Mode is how a resident tower divides over the pool's tensor-parallel ranks.
	Mode EncoderMode
}

// EncoderMode is how the engine divides a tower over the tensor-parallel ranks.
type EncoderMode string

const (
	// EncoderWeightsSharded: the tower's column- and row-parallel linears are sharded over
	// the ranks and every rank encodes every image.
	EncoderWeightsSharded EncoderMode = "weights_sharded"
	// EncoderBatchSplit: every rank holds the whole tower and encodes its share of the images,
	// after which the output tokens are gathered.
	EncoderBatchSplit EncoderMode = "batch_split"
)

// Valid reports whether m is a recognized mode.
func (m EncoderMode) Valid() bool { return m == EncoderWeightsSharded || m == EncoderBatchSplit }

// ReportedSettings are settings the engine resolved that the cost model reports and does not
// price.
type ReportedSettings struct {
	// CascadeAttention is true when the engine may run attention over a shared prefix once for
	// the requests that share it.
	CascadeAttention bool
	// SMAllReduce is true when a tensor-parallel all-reduce runs on a kernel that uses the GPU's
	// streaming multiprocessors over peer memory, false when it runs through the communication
	// library.
	SMAllReduce bool
}

// Override is one stated setting the engine resolved to another value, and why. Field names
// the setting in the engine's own words.
type Override struct {
	Field, Requested, Resolved, Reason string
}

// Note is one value the engine took where the deployment stated none, and the rule that set it.
// Name is the setting in the engine's own words.
type Note struct {
	Name, Value, Reason string
}
