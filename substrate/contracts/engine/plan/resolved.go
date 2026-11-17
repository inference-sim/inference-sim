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
)

// Valid reports whether r is a known word.
func (r RecurrentStates) Valid() bool {
	return r == RecurrentLatest || r == RecurrentPageAligned || r == RecurrentEveryPage
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
	// format, part and parallelism), named by its kernel library (ExpertsTriton,
	// ExpertsFlashInferCutlass). The cost model prices the routed term on the ramp measured for
	// that kernel where the registry carries one. The zero value is not stated, and the routed
	// term is then priced by weight format alone.
	Experts KernelImpl
	// RedundantExperts is how many extra copies of experts the engine places beside the model's
	// own to balance the load; 0 is none.
	RedundantExperts int
}

// The MoE dispatch kernels a cost model prices by name (MoEPlan.Dispatch): the three kernel
// families of the DeepEP library. Any other dispatch kernel is named by its library alone.
var (
	DispatchDeepEPHighThroughput = KernelImpl{Library: "deepep", Variant: "high_throughput"}
	DispatchDeepEPLowLatency     = KernelImpl{Library: "deepep", Variant: "low_latency"}
	DispatchDeepEPV2             = KernelImpl{Library: "deepep", Variant: "v2"}
)

// The grouped-expert kernels a cost model prices by name (MoEPlan.Experts): Triton's fused
// experts, and FlashInfer's CUTLASS experts. Any other experts kernel is named by its library and
// variant, and priced by its weight format.
var (
	ExpertsTriton            = KernelImpl{Library: "triton", Variant: "fused_experts"}
	ExpertsFlashInferCutlass = KernelImpl{Library: "flashinfer", Variant: "cutlass_experts"}
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
}

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
	width("Cache.ElementBytes", p.Cache.ElementBytes)
	width("Cache.LatentCellBytes", p.Cache.LatentCellBytes)
	width("Cache.StateElementBytes", p.Cache.StateElementBytes)
	negative("MoE.RedundantExperts", int64(p.MoE.RedundantExperts))
	word("DecodeContext.Combine", string(p.DecodeContext.Combine), p.DecodeContext.Combine.Valid())
	negative("DecodeContext.Interleave", int64(p.DecodeContext.Interleave))
	negative("Engram.Shards", int64(p.Engram.Shards))
	word("Graphs.Decode", string(p.Graphs.Decode), p.Graphs.Decode.Valid())
	word("Graphs.Other", string(p.Graphs.Other), p.Graphs.Other.Valid())
	negative("Graphs.Ceiling", int64(p.Graphs.Ceiling))
	negative("Drafts.RetainedTokens", int64(p.Drafts.RetainedTokens))
	negative("InFlight", int64(p.InFlight))
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
}

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
