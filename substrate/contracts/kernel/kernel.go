// Package kernel is the interface a cost model implements and a discrete-event
// simulator calls.
//
// Every method is a pure function of its arguments and the resolved configuration:
// same inputs, same outputs, no internal mutation, safe for concurrent use. That is
// not a style preference. It is what lets the kernel be called at any simulated
// instant, tested without a scheduler attached, and memoized by batch shape.
//
// The boundary follows from purity. Anything whose value depends on WHEN a request
// arrives belongs to the simulator: queue waiting, admission ordering, preemption.
// Anything that follows from a batch and the deployment belongs here. A KV transfer
// is the instructive case — it is concurrent with execution and gates when a request
// becomes schedulable, so it sets an arrival time rather than a step time, and a
// kernel that added it to step time would charge for latency the engine already hid.
package kernel

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// Kernel prices one instantiated deployment. Obtain one from a constructor that
// resolves a scenario; there is deliberately no Init method on the interface,
// because an interface-level Init admits an unconstructed value and makes every
// other method's behaviour depend on whether it was called — the mutable state
// purity rules out.
type Kernel interface {
	// --- Memory occupancy ---------------------------------------------------

	// FixedBytes is occupancy independent of the request set, per rank.
	FixedBytes() MemoryBreakdown

	// SequenceFixedBytes is per-sequence occupancy that does not vary with prompt
	// length: recurrent state, per-sequence bookkeeping. Zero for a pure-attention
	// model. It is computed from the layout rather than stored, because it divides
	// with tensor-parallel width and its group count pads up to divide that width.
	SequenceFixedBytes() int64

	// SequenceVariableBytes is per-sequence occupancy as a function of token count,
	// quantized to the engine's page size. Two settings divide it: tensor-parallel
	// width, down to a floor of one KV head, and the cache dtype.
	SequenceVariableBytes(tokens int) int64

	// --- Step time -----------------------------------------------------------

	// StepTime prices one forward pass over a batch.
	StepTime(Batch) StepEstimate

	// --- Transfers -----------------------------------------------------------

	// TierTime prices one transfer between GPU and an offload tier. Two links are
	// crossed on a CPU tier — the host interconnect and host memory — and the slower
	// one binds, so an implementation takes the minimum rather than trusting either.
	TierTime(tier string, dir Direction, bytes int64, queueDepth int) time.Duration

	// PDTransferTime prices moving one request's KV between pools. The byte count is
	// not tokens times a constant: it depends on the attention kind and on which
	// parallelism axis shards KV.
	PDTransferTime(tokens int, from, to Placement) time.Duration

	// PDTransferDemand is the demand behind PDTransferTime: the same handoff, separated
	// into payload, floor and links so a caller running several at once can arbitrate
	// the links they share. The kernel never sees other transfers; dividing a shared
	// link among them is the simulator's work.
	//
	// The law, part of this contract, is that alone a demand costs the existing price:
	//
	//	PDTransferTime(t, f, g) == PDTransferDemand(t, f, g).AloneTime()
	//
	// for every argument, exactly. An implementation keeps it by computing
	// PDTransferTime from its own demand; one that priced in float seconds before
	// adopting this may move by the nanosecond rounding AloneTime fixes. Package
	// kerneltest checks the law against any implementation.
	PDTransferDemand(tokens int, from, to Placement) TransferDemand

	// --- Host overheads ------------------------------------------------------
	// These price engine work around a request rather than GPU work inside a step.
	// They belong here because the cost model already carries a host term for kernel
	// launch, and splitting host cost across two owners would leave a caller
	// sourcing some coefficients from the kernel and others from elsewhere.

	// AdmissionOverhead is CPU time before a request can be scheduled:
	// tokenization, template application, multimodal preprocessing. It scales with
	// prompt length. It excludes queue waiting, which depends on engine busyness and
	// therefore cannot live in a pure function.
	AdmissionOverhead(promptTokens int) time.Duration

	// OutputTokenOverhead is per-emitted-token host work: incremental
	// detokenization and streaming. It runs in the output path rather than the
	// forward pass, which is why it does not enter StepTime.
	OutputTokenOverhead() time.Duration

	// CompletionOverhead is fixed per-request work at finish: finish-reason
	// determination, serialization, teardown. It lands on end-to-end latency and not
	// on time-to-first-token.
	CompletionOverhead() time.Duration

	// --- Provenance ----------------------------------------------------------

	// Provenance reports every coefficient this kernel used, with its evidence. A
	// prediction that cannot state what it rests on cannot be audited.
	Provenance() []CoefficientOrigin

	// Resolved reports the configuration after resolution, including requested
	// settings the layout overrode.
	Resolved() Resolution

	// Deployment reports the pool this kernel prices: its role, its node count, its
	// parallelism and its engine settings, as the document declared them.
	//
	// It is on the interface because a simulator needs it and there is no other
	// non-duplicating source. The scheduler-facing engine settings — block_size,
	// max_num_seqs, max_num_batched_tokens — decide ADMISSION, which the package
	// comment assigns to the simulator; but they are properties of the pool the
	// kernel resolved, and the kernel is the only value that knows which pool that
	// was. A caller that instead retained the Deployment and a pool index would hold
	// a second answer to "which pool is this", and the two can disagree: a decode
	// pool priced at a prefill pool's parallelism is a wrong simulation that still
	// runs. Returning the pool removes the second answer rather than asking callers
	// to keep it consistent.
	//
	// A pool rather than an Engine because the role comes with it at no cost, and a
	// caller asking for admission settings is usually the same caller that needs to
	// know whether this engine prefills.
	//
	// What it returns is the REQUEST, not the resolution. The pool is the document as
	// loaded: where resolution settled a width or a backend differently, Resolved is
	// the answer and this is not. The two are separate values precisely so a reader is
	// never shown one while believing it is the other — so a caller wanting the
	// parallelism or the all-reduce backend reads Resolved, and reads the pool for the
	// settings resolution does not touch, which is what the admission settings are.
	//
	// It returns a copy, which is what a value return gives, but not a deep one: its
	// slices, maps and engine block are shared with whatever the implementation holds.
	// A caller must treat the result as read-only. Deep-copying instead would
	// cost an allocation on every call for a guarantee no caller in this design
	// needs, and the purity the package comment requires is a promise about what the
	// kernel does, not a defence against a caller that writes through a borrowed
	// pointer.
	Deployment() deployment.Pool
}

// MemoryBreakdown decomposes fixed occupancy rather than returning a scalar,
// because a caller deciding between "shard further" and "raise utilization" needs
// to know which term dominates.
type MemoryBreakdown struct {
	Weights        int64 // parameters resident on this rank
	ActivationPeak int64 // transient scratch at the batched-token bound
	CUDAGraph      int64 // capture cost; zero when capture is off
	EPLBRedundant  int64 // redundant-expert replicas; zero when EPLB is off
	CommBuffers    int64 // collective and all-to-all buffers
}

// Total returns the sum. Callers that only need a feasibility answer use it; the
// fields exist for callers that need to act on the result.
func (m MemoryBreakdown) Total() int64 {
	return m.Weights + m.ActivationPeak + m.CUDAGraph + m.EPLBRedundant + m.CommBuffers
}

// Resource is the axis a cost lands on. A communication op appears as ResourceSM
// when the resolved backend is an SM-consuming kernel, which is why the resolved
// backend and not the requested one determines it.
type Resource string

const (
	ResourceSM         Resource = "sm"
	ResourceHBM        Resource = "hbm"
	ResourceNVLink     Resource = "nvlink"
	ResourceNIC        Resource = "nic"
	ResourcePCIe       Resource = "pcie"
	ResourceCopyEngine Resource = "copy_engine"
	ResourceHost       Resource = "host"
)

// StepEstimate brackets a step and names its bottleneck.
type StepEstimate struct {
	// Overlap assumes perfect overlap WITHIN a stage; NoOverlap assumes none anywhere.
	//
	// The qualifier is the whole content of the estimate. A stage is a set of operations
	// that really do run concurrently — one layer, since the next layer needs this one's
	// output — so Overlap takes the max over resources inside a stage and sums across
	// stages. It is not a max over the step: that would assume every layer overlaps every
	// other, which understates a step badly wherever no single resource dominates, and
	// single-request decode is exactly that case.
	//
	// Reporting both matters because a configuration whose verdict flips between them
	// needs measurement rather than prediction, and a single number hides that.
	Overlap   time.Duration
	NoOverlap time.Duration

	// Bottleneck is the resource that did the most work over the step. It distinguishes
	// "add GPUs" from "raise the batch size" as the next action, and the answer changes
	// with batch size within one deployment.
	//
	// Not "the resource that set Overlap", which it cannot be. A step composes per stage
	// — max over resources within a stage, sum across stages, since one layer's output is
	// the next layer's input — so Overlap is a sum of per-stage maxima and no single
	// resource's total equals it. Which resource to relieve is the actionable question,
	// and answering it does not require that one resource dominate every stage.
	Bottleneck Resource

	// PerResource is the per-resource sum, for a caller that needs to see why.
	PerResource map[Resource]time.Duration

	// Stages splits the step by pipeline stage, in pipeline order, where the deployment
	// runs more than one (pp > 1); nil for a single stage.
	//
	// A pipeline-parallel step is not one forward pass on one set of GPUs but pp of them in
	// sequence, each on its own ranks: stage s runs its contiguous share of the layers and
	// hands its activations to stage s+1. Two batches in flight therefore overlap, one in
	// each stage, and a simulator needs every stage's time to know both when a batch's output
	// is ready (the sum) and when the next batch may enter a stage (that stage's own time).
	// Each stage's figure includes the send of its activations to the next stage, which is
	// where the inter-stage link is paid.
	//
	// The law, part of this contract: the stages' Overlap figures sum to Overlap and their
	// NoOverlap figures sum to NoOverlap, exactly. Package kerneltest checks it.
	Stages []StageEstimate
}

// StageEstimate is one pipeline stage's share of a step, bracketed as the step is.
type StageEstimate struct {
	Overlap   time.Duration
	NoOverlap time.Duration
}

// Batch is what the simulator must supply: per-request shape plus the engine and
// resource state a cost depends on. Nothing here is derivable from the deployment
// alone, which is why each field exists.
type Batch struct {
	Reqs []ReqShape

	// DecodeThreshold selects the attention kernel per request. It is engine
	// configuration rather than request state, and the classifier needs both.
	DecodeThreshold int

	// SMBudget is the SM count available this step: the chip's count, less any
	// withheld by a concurrent offload fetch. It is absorbed entirely when a step is
	// bandwidth-bound, which is the per-stage max rule working as intended.
	SMBudget int

	// PadTokens are the tokens this data-parallel rank pads its step with under the engine's DP
	// lockstep. Every rank of a data-parallel group runs each step at the group's largest
	// token count (v1/worker/dp_utils.py:99-111 _post_process_dp_padding),
	// passed to the forward context as num_tokens (v1/worker/gpu_model_runner.py:
	// 4407-4413); a rank with nothing scheduled runs a dummy batch of
	// num_tokens_unpadded = 0 (:6090, _dummy_run from :5784). Pad tokens run the GEMMs, the
	// MoE dispatch and combine, and the expert paths, but never the logits, the LM head,
	// the vocabulary all-gather or sampling, which read only the real requests' rows
	// (logits_indices, :2231). They are not requests, and pricing them as decode requests
	// overstates the work after the last layer and the attention.
	//
	// Zero means none. It must not be negative (Validate).
	PadTokens int

	// DraftPaddedTokens is the token count this rank's drafter launches, after its own graph
	// bucket and the width its data-parallel group agreed (the drafter coordinates its padding
	// apart from the target's: vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:455-471). The
	// drafter's passes run the GEMMs and the MoE paths over that many rows, of which only the
	// rank's own draft rows are real. Zero means the drafter launches only its own rows; a figure
	// below them is read the same way. It must not be negative (Validate).
	DraftPaddedTokens int
}

// ReqShape is one request's contribution. The three token counts are what an
// engine's own region classifier reads; passing a region enum instead would move an
// engine implementation detail into the caller.
type ReqShape struct {
	// Scheduled is tokens scheduled this step. Under speculation it is the draft
	// length plus one, so a value of one does not imply a decode.
	Scheduled int
	// Computed is tokens already computed. An engine advances it optimistically at
	// dispatch and rolls it back on speculative rejection, so a simulator passes the
	// value as of batch formation.
	Computed int
	// PromptLen is the prompt length. Computed against it is what separates a
	// chunked-prefill tail from a decode, and the two use different kernels.
	PromptLen int
	// CachedTokens is the prefix-cache hit, already subtracted from Scheduled. A
	// count rather than a rate: the simulator has computed it, and a cost model that
	// took a rate would have to guess.
	CachedTokens int
}

// Tokens returns the batch's total scheduled token count, the quantity collective
// thresholds compare against.
func (b Batch) Tokens() int {
	n := 0
	for _, r := range b.Reqs {
		n += r.Scheduled
	}
	return n
}

// UniformDecode reports whether every request schedules the same token count, equal
// to the uniform decode width. It decides which of two collective thresholds
// applies, and a single partially-rejected speculative request breaks it.
func (b Batch) UniformDecode(uniformWidth int) bool {
	if len(b.Reqs) == 0 {
		return false
	}
	for _, r := range b.Reqs {
		if r.Scheduled != uniformWidth {
			return false
		}
	}
	return true
}

// Direction distinguishes an offload write from a read back. They are separate
// because device read and write bandwidths differ materially on flash, and pricing
// an eviction at read bandwidth understates it.
type Direction string

const (
	DirectionToTier   Direction = "to_tier"
	DirectionFromTier Direction = "from_tier"
)

// Placement locates one rank. Comparing two placements is how a transfer's link is
// chosen: same node, same rack, or across the fabric. A two-tier placement cannot
// express a rack boundary, which is why Rack is present.
type Placement struct {
	Node int
	Rack int
	Pool deployment.Role
}

// CoefficientOrigin is one row of the provenance trail.
type CoefficientOrigin struct {
	Name   string // registry entry name
	Set    string // the set it came from
	Method string // the evidence vocabulary's term
	Scope  string // the scope as recorded
	// KernelMatch says how the entry met the kernel implementations that run (design VAL-8):
	// "exact" when it lists one of them by library, variant and code hash, "unkeyed" when it
	// lists none and so applies to every implementation without having been shown to match
	// the one that runs. Empty for a value the kernel supplied itself. A reader deciding that a
	// price rests on the running kernel counts only "exact".
	KernelMatch string
	// Primitive is the lowered primitive the entry prices, and Running the kernel implementations
	// the plan supplied for it, which the entry was matched against (VAL-8), as
	// "library/variant/code_hash" joined by ", " in sorted order. Both are empty for a value the
	// kernel supplied itself, and Running is empty when nothing was supplied. (A string, not a
	// list, so that CoefficientOrigin stays comparable.)
	Primitive string
	Running   string
}

// Resolution reports what the constructor decided where a deployment expressed a
// request. It exists so a prediction can be read without re-deriving the resolver's
// logic, and so a reader is never misled by a field the layout overrode.
type Resolution struct {
	// TensorParallelWidth, DataParallelWidth and ExpertParallelWidth are the
	// resolved layout. All three are decided by one constructor from one
	// deployment.Parallelism, which is why they are reported together: a consumer
	// that could read one and not the others would have to re-derive the rest or
	// read the Deployment and get the REQUEST rather than the resolution.
	//
	// Each states a width, so each has a floor of one — a layout that shards along
	// no axis still has one rank along it. Zero is not a narrower layout but an
	// unset field, and the distinction matters because a consumer multiplies by
	// these: a scheduler scales its sequence cap, token budget and KV budget by
	// DataParallelWidth, so a zero silently makes all three zero rather than
	// failing. An implementation fills all three; a reader that may see a
	// zero-valued Resolution should use TensorParallel, DataParallel and
	// ExpertParallel, which apply the floor.
	TensorParallelWidth int
	DataParallelWidth   int
	// ExpertParallelWidth is derived from the other two and the prefill-context
	// width rather than requested: tp x pcp x dp, or 1 when expert parallelism is
	// off. It is reported and not recomputed here because two of the facts it
	// needs — whether expert parallelism is enabled at all, and the prefill-context
	// width — are not part of this value, so a consumer recomputing it from the two
	// fields above may get a different answer. Expert parallelism off is the sharpest
	// case: the width is 1 however wide tp and dp are.
	ExpertParallelWidth int
	// AllReduceBackend is what will run, which may differ from what was requested.
	AllReduceBackend string
	// AsyncScheduling is the resolved value of a tri-state request.
	AsyncScheduling bool
	// CascadeAttention is whether it is available after every gate is applied.
	CascadeAttention bool
	// SequenceParallelMoE is whether the MoE input is made sequence-parallel, which
	// replaces an all-reduce with a reduce-scatter and all-gather pair.
	SequenceParallelMoE bool
	// Overrides records each request the layout could not honour, and why.
	Overrides []Override
}

// TensorParallel returns the resolved tensor-parallel width with its floor applied.
// It exists so a consumer reading a Resolution it did not construct cannot divide or
// multiply by a zero that means "unset" — see the field comments. An implementation
// should still fill the fields; this is the reader's guard, not a substitute.
func (r Resolution) TensorParallel() int { return atLeastOne(r.TensorParallelWidth) }

// DataParallel returns the resolved data-parallel width with its floor applied. An
// omitted dp means one instance, not none, and this is the field a scheduler
// multiplies its sequence cap, token budget and KV budget by.
func (r Resolution) DataParallel() int { return atLeastOne(r.DataParallelWidth) }

// ExpertParallel returns the resolved expert-parallel width with its floor applied.
// One is the correct floor here too: expert parallelism off means every rank holds
// every expert, which is a width of one group, not of none.
func (r Resolution) ExpertParallel() int { return atLeastOne(r.ExpertParallelWidth) }

func atLeastOne(w int) int {
	if w < 1 {
		return 1
	}
	return w
}

// Override is one request the resolver declined, with the reason.
type Override struct {
	Field     string
	Requested string
	Resolved  string
	Reason    string
}

// PaddedTokens returns the tokens the step runs through the layers: the scheduled tokens
// plus the data-parallel pad, the count the forward context carries as num_tokens.
func (b Batch) PaddedTokens() int { return b.Tokens() + b.PadTokens }

// Validate reports a batch no step can have: a negative pad, budget, threshold or
// per-request count.
func (b Batch) Validate() error {
	var bad []string
	for field, v := range map[string]int{
		"PadTokens": b.PadTokens, "SMBudget": b.SMBudget, "DecodeThreshold": b.DecodeThreshold,
		"DraftPaddedTokens": b.DraftPaddedTokens,
	} {
		if v < 0 {
			bad = append(bad, fmt.Sprintf("%s %d", field, v))
		}
	}
	for i, r := range b.Reqs {
		for field, v := range map[string]int{
			"Scheduled": r.Scheduled, "Computed": r.Computed, "PromptLen": r.PromptLen,
			"CachedTokens": r.CachedTokens,
		} {
			if v < 0 {
				bad = append(bad, fmt.Sprintf("Reqs[%d].%s %d", i, field, v))
			}
		}
	}
	if len(bad) == 0 {
		return nil
	}
	sort.Strings(bad)
	return fmt.Errorf("batch: must not be negative: %s", strings.Join(bad, ", "))
}
