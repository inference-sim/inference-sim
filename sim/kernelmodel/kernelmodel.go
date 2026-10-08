// Package kernelmodel adapts blis-latency-kernel to BLIS's sim.LatencyModel interface.
//
// # What this package is, and what it deliberately is not
//
// It is a translation layer. Every latency number it returns comes from the kernel; this
// package computes no cost of its own. The point of the exercise is to measure the kernel
// as committed in blis-latency-kernel, so any arithmetic added here would contaminate the
// result it is meant to measure.
//
// Three things do need care, and they are the whole of this file's substance:
//
//   - UNITS. BLIS's sim.LatencyModel is int64 MICROSECONDS ("ticks", see
//     sim/latency_model.go). The kernel returns time.Duration, which is nanoseconds.
//     Every boundary crossing converts, and nothing else in this file may see a Duration.
//
//   - THE >= 1 POSTCONDITION. BLIS requires StepTime >= 1 for every input including an
//     empty batch, because 0 stalls the simulation clock and violates INV-3. The kernel
//     charges an empty batch its host per-step cost, which is non-zero today, but relying
//     on a coefficient staying non-zero would make a clock invariant depend on a registry
//     value. So the floor is explicit here.
//
//   - BATCH TRANSLATION. BLIS's Request and the kernel's ReqShape carry the same facts
//     under different names. Getting this mapping wrong is silent: pricing a decode as a
//     chunked-prefill tail changes the kernel's choice of attention law, and the
//     simulation still runs.
//
// # What this package does NOT do
//
// It no longer loads artifacts or reads deployment documents. blis-latency-kernel exports
// Open, OpenPool and the accessors for everything a consumer needs from a resolved
// deployment, so this package holds only what is genuinely its own: adapting a kernel to
// sim.LatencyModel, translating a BLIS batch into the kernel's ReqShape, and turning the
// kernel's byte answers into a KV block count.
//
// That was not always true. The kernel kept its loading in internal/harness, which Go
// forbids importing across modules, so this package carried a copy of the sequence and its
// own loadBundle, and retained the Scenario, the Deployment and a pool index to index back
// into Pools[i] for engine settings and widths. Every line of that is gone. The copy was
// bounded -- loading, not modelling -- but the retained documents were worse than
// duplication: they were a second answer to "which pool is this", and when the two
// disagreed a decode pool was priced at a prefill pool's parallelism and the simulation
// still ran.
//
// Delegating also gained the kernel's validation. The copy did not validate, so a
// deployment whose pools do not fill its cluster built a model and simulated.
package kernelmodel

import (
	"fmt"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/inference-sim/sim"
)

// Repos is blis-latency-kernel's Repos.
//
// An alias rather than a second type: the kernel's Open takes one, so a distinct type here
// would mean converting at every call and would let the two drift if the kernel ever adds a
// root. A kernelmodel.Repos{...} literal still compiles unchanged.
type Repos = latencykernel.Repos

// Model adapts a kernel to sim.LatencyModel.
//
// decodeThreshold and smBudget are fixed at construction because they are engine and
// hardware configuration rather than per-step state, and re-deriving them inside the hot
// StepTime path would cost more than storing them.
type Model struct {
	k *latencykernel.Kernel

	decodeThreshold int
	smBudget        int

	// Host costs are invariant per kernel, so they are converted to ticks once. StepTime
	// is called once per simulated step; a Duration-to-ticks division per call is wasted.
	outputTokenTicks int64
	completionTicks  int64

	// No documents are retained. Everything this adapter needs about the deployment --
	// engine settings, the two parallel widths, the chip, the model name, the expert
	// geometry -- is read from the kernel, which resolved it.
	//
	// It used to hold the Scenario, the Deployment and the pool index so it could index
	// back into Pools[poolIndex]. That is a second source of truth for "which pool is
	// this": the kernel's answer and this bookkeeping could disagree, and a disagreement
	// prices a decode pool at a prefill pool's parallelism with nothing reporting it.
}

var _ sim.LatencyModel = (*Model)(nil)

// DecodeThreshold is the scheduled-token count at or below which the engine's classifier
// treats a request as a decode. It mirrors blis-latency-kernel's own constant; the kernel
// takes it per batch rather than holding it, because it is engine configuration.
const DecodeThreshold = 8

// Open builds a kernel from a scenario file and adapts it.
func Open(scenario string, r Repos) (*Model, error) {
	return OpenPool(scenario, r, 0)
}

// OpenPool builds a kernel for ONE pool of a scenario and adapts it.
//
// A disaggregated scenario states a prefill pool and a decode pool, and the two differ in
// the quantities that set step time -- tensor-parallel width, expert parallelism, the
// engine's token budget. Pricing both from pool 0 would charge the decode pool the prefill
// pool's parallelism, so a caller serving roles separately opens one model per pool. Open is
// this function at pool 0, which is what a colocated scenario has.
//
// The loading is blis-latency-kernel's own OpenPool. This used to reproduce that sequence --
// scenario, deployment, model graph, chip, fabric, coefficient sets, storage devices, rules
// pack -- because the kernel kept it in internal/ and Go forbids importing another module's
// internal packages. The kernel now exports it, so the copy is gone: one answer to "which
// files does this scenario imply", in the repository that owns the question.
//
// Delegating also gains the validation the kernel's New now runs. The copy here did not
// validate, so a deployment whose pools do not fill its cluster, or whose local
// data-parallel width does not divide a node, built a model and simulated.
func OpenPool(scenario string, r Repos, poolIndex int) (*Model, error) {
	k, err := latencykernel.OpenPool(scenario, r, poolIndex)
	if err != nil {
		return nil, err
	}
	return New(k, k.Chip().SMCount), nil
}

// New adapts an already-built kernel. Exported so a caller that constructed a kernel
// some other way -- a test with a hand-built coefficient set, say -- can adapt it without
// going through the filesystem.
func New(k *latencykernel.Kernel, smBudget int) *Model {
	return &Model{
		k:                k,
		decodeThreshold:  DecodeThreshold,
		smBudget:         smBudget,
		outputTokenTicks: ticks(k.OutputTokenOverhead()),
		completionTicks:  ticks(k.CompletionOverhead()),
	}
}

// Kernel exposes the adapted kernel. A caller that needs a memory or provenance answer
// asks the kernel directly rather than having this adapter grow a passthrough per method.
func (m *Model) Kernel() *latencykernel.Kernel { return m.k }

// StepTime prices one forward pass over the scheduled batch.
//
// StepEstimate.Expected is read rather than either band edge. The kernel reports a band
// -- Overlap sums the max over resources within each layer, NoOverlap sums every resource
// -- and Expected is the edge its own measured evidence selects, so this consumer does not
// re-decide. An earlier version of this comment claimed Overlap "is what a real engine
// achieves"; NVIDIA's FPM dataset, which measures one synchronized whole-forward iteration
// at a known batch and KV-token count, contradicts that. Over 219 points spanning two
// models, two parts and five parallelism topologies:
//
//	edge       mean|err|   signed
//	Overlap      14.70%   -10.80%
//	NoOverlap    10.16%    -0.73%
//
// NoOverlap is closer on four of the five cells and is nearly unbiased, where Overlap
// carries a one-sided deficit of the same magnitude BLIS shows end to end. The physical
// reason is PIECEWISE cudagraph mode: attention runs eagerly between captured segments,
// so per-layer overlap is structurally limited.
//
// The one dissenting cell is pure-tp2 -- the whole model on two GPUs, where the expert
// weight read dominates a single resource and per-stage max is the right composition.
// Even there NoOverlap wins at batch >= 128. That is a regime boundary worth revisiting
// with a resource-aware blend, not a reason to keep the optimistic edge everywhere.
//
// FPM is used only to choose between the kernel's own two edges. It is not fitted
// against, and the InferenceX corpus BLIS is scored on is never used for either.
func (m *Model) StepTime(batch []*sim.Request) int64 {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, 0, len(batch)),
		DecodeThreshold: m.decodeThreshold,
		SMBudget:        m.smBudget,
	}
	for _, req := range batch {
		b.Reqs = append(b.Reqs, shapeOf(req))
	}
	// NoOverlap rather than Overlap, and this is a measured choice rather than a
	// migration detail. schemas v0.2.0 removed StepEstimate.Expected, which the kernel
	// had set from evidence so a caller wanting one number did not have to re-decide.
	// The evidence it was set from, recorded in blis-registry's docs/band-selection.md
	// and in the kernel's own StepTime comment: over 219 points of NVIDIA's FPM dataset
	// spanning two models, two parts and five parallelism topologies, Overlap's signed
	// error is -13.45% and NoOverlap's is -3.44%, and NoOverlap is closer on 158 of
	// them. The physical reason is PIECEWISE cudagraph mode -- attention runs eagerly
	// between captured segments, so per-layer overlap is structurally limited.
	//
	// The kernel's StepTime comment states the conclusion directly: "A caller that wants
	// one figure should read NoOverlap." Reading Overlap here would silently under-price
	// every simulated step by about ten points.
	//
	// max(1) upholds BLIS's postcondition without depending on a registry value staying
	// non-zero. See the package comment.
	return max(1, ticks(m.k.StepTime(b).NoOverlap))
}

// QueueingTime is the host work before a request can be scheduled: tokenization and
// prompt preprocessing. It excludes queue waiting, which depends on engine busyness and
// is the simulator's own concern rather than a pure function's.
func (m *Model) QueueingTime(req *sim.Request) int64 {
	return ticks(m.k.AdmissionOverhead(int(req.InputLen())))
}

// OutputTokenProcessingTime is per-emitted-token host work: incremental detokenization
// and streaming.
func (m *Model) OutputTokenProcessingTime() int64 { return m.outputTokenTicks }

// PostDecodeFixedOverhead is fixed per-request work at completion.
func (m *Model) PostDecodeFixedOverhead() int64 { return m.completionTicks }

// shapeOf translates one BLIS request into the kernel's request shape.
//
// The mapping, and why each field is what it is:
//
//	Scheduled    <- NumNewTokens. Tokens this step, set by FormBatch. Under speculation
//	                it is 1+accepted, which is why a value of 1 does not imply a decode.
//	Computed     <- ProgressIndex. Tokens already computed. BLIS advances it after the
//	                step, so at batch formation it is the pre-step value the kernel wants.
//	PromptLen    <- InputLen(). Computed against PromptLen is what separates a
//	                chunked-prefill tail from a decode, and the two use different
//	                attention laws.
//	CachedTokens <- 0. A prefix-cache hit is already reflected in Computed: BLIS's
//	                batch_formation sets numNewTokens = InputLen() - ProgressIndex, and
//	                the cache-aware allocation path is what advances ProgressIndex. So the
//	                hit is excluded from Scheduled before the adapter sees it, and passing
//	                a cached count here would subtract the same tokens twice. The kernel's
//	                field exists for a caller that has NOT yet subtracted it.
//
//	                Note what this does NOT rest on: BLIS implements prefix caching
//	                unconditionally, in sim/kv, not behind a flag. An earlier version of
//	                this comment said BLIS "subtracts hits before setting NumNewTokens",
//	                which is the right conclusion from a premise that would have been
//	                false if caching were the thing being relied on.
func shapeOf(req *sim.Request) kernel.ReqShape {
	return kernel.ReqShape{
		Scheduled:    req.NumNewTokens,
		Computed:     int(req.ProgressIndex),
		PromptLen:    int(req.InputLen()),
		CachedTokens: 0,
	}
}

// ticks converts a kernel Duration to BLIS's int64 microsecond tick.
//
// Truncation rather than rounding, to match BLIS's existing backends: both call
// clampToInt64 on a float microsecond value. The largest error is 1 tick, which on the
// smallest step in the evaluation corpus (553 us for an empty batch) is under 0.2%.
func ticks(d interface{ Microseconds() int64 }) int64 { return d.Microseconds() }

// Engine reports the engine settings of the pool this kernel prices.
//
// A caller configuring a simulator needs the same block size, token budget and sequence cap
// the kernel resolved against, or the two would describe different deployments: a scheduler
// admitting 512 sequences while the kernel priced a 256-sequence engine is not a
// disagreement the numbers would reveal.
//
// It returns the deployment's own values rather than a copy with defaults filled in, and
// errors when a value a simulator requires is absent, so a missing setting is a failure
// rather than a silent zero.
//
// The settings live on the DEPLOYMENT as of blis-schemas v0.2.0: engine knobs are tunable
// configuration chosen against a scenario, not part of the immutable problem it states.
// The error text says "deployment" for the same reason -- a reader told the scenario lacks
// a block_size would look in the wrong document.
func (m *Model) Engine() (deployment.Engine, error) {
	e := m.k.Engine()
	if e.BlockSize <= 0 {
		return e, fmt.Errorf("kernelmodel: deployment states no block_size")
	}
	if e.MaxNumSeqs <= 0 {
		return e, fmt.Errorf("kernelmodel: deployment states no max_num_seqs")
	}
	if e.MaxNumBatchedTokens <= 0 {
		return e, fmt.Errorf("kernelmodel: deployment states no max_num_batched_tokens")
	}
	return e, nil
}

// DataParallelWidth is the pool's attention data-parallel width.
//
// vLLM runs this many independent EngineCores, each with its own sequence cap, token budget
// and KV budget, and splits requests disjointly across them. A single-instance simulator
// modelling the aggregate must scale all three.
func (m *Model) DataParallelWidth() int {
	return m.k.DataParallelWidth()
}

// StepEstimate exposes the kernel's full band for one batch: the overlap edge, the serialized
// edge, the binding resource and the per-resource breakdown.
//
// StepTime returns only the overlap edge, because sim.LatencyModel is a single int64. A
// caller diagnosing WHY a step costs what it does needs the rest, and asking the kernel again
// through this method is cheaper and less error-prone than rebuilding the batch translation.
func (m *Model) StepEstimate(batch []*sim.Request) kernel.StepEstimate {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, 0, len(batch)),
		DecodeThreshold: m.decodeThreshold,
		SMBudget:        m.smBudget,
	}
	for _, req := range batch {
		b.Reqs = append(b.Reqs, shapeOf(req))
	}
	return m.k.StepTime(b)
}

// Deployment reports the scenario facts an ALTERNATIVE latency backend needs to be
// constructed for the same deployment this kernel models.
//
// It exists for cmd/kernelscore's four-estimator comparison, which scores the kernel against
// BLIS's own roofline and trained-physics backends on one subset. Those backends are
// configured from a HuggingFace config.json and a hardware-calibration entry rather than from
// the catalog and registry, so the harness needs the scenario's model name, chip, tensor
// width and the two precisions -- and must read them from the SAME scenario the kernel was
// opened from, or the arms would describe different deployments.
//
// Nothing here is a latency coefficient: this is deployment identity, not calibration.
func (m *Model) Deployment() Deployment {
	// The layout comes from the Deployment and the identity from the Scenario, which is
	// exactly the v0.2.0 split: what traffic runs on which hardware is the problem, how
	// it is laid out is the choice. Both are the pair this model was opened from, so the
	// arms cannot describe different deployments.
	e := m.k.Engine()
	return Deployment{
		Model:    m.k.ModelName(),
		Hardware: m.k.Chip().Name,
		TP:       m.k.TensorParallelWidth(),
		DP:       m.k.DataParallelWidth(),
		// The RESOLVED width, not the request: expert parallelism is on when the group is
		// wider than one rank, which is what the kernel settled.
		ExpertParallel: m.k.Resolved().ExpertParallelWidth > 1,
		MoE:            m.k.Experts() > 0,
		Experts:        m.k.Experts(),
		ExpertsPerTok:  m.k.ExpertsPerToken(),
		Quantization:   e.Quantization,
		CacheDType:     e.CacheDType,
		// The chip's total memory, which is what vLLM resolves its batch defaults from.
		DeviceMemoryGiB: m.k.Chip().MemoryGiB,
		// Tri-state in the scenario, resolved here to the engine's default. vLLM caches
		// unless told not to, so nil means ON and only an explicit false disables it.
		PrefixCachingDisabled: e.EnablePrefixCaching != nil &&
			!*e.EnablePrefixCaching,
		BlockSize:  int64(e.BlockSize),
		GPUMemUtil: e.GPUMemoryUtilization,
	}
}

// Deployment is the scenario identity an alternative backend is built from.
type Deployment struct {
	Model    string
	Hardware string
	TP       int
	DP       int
	// ExpertParallel is the pool's parallel.enable_expert_parallel. Carried because a
	// caller deriving its deployment from the scenario needs every field that sets step
	// time, and expert parallelism changes which GEMM a routed layer runs.
	ExpertParallel bool
	// MoE reports whether the model routes tokens through experts, read from the graph
	// (a layer kind carrying a GroupedGEMM) rather than from an HF config. A caller
	// gating data parallelism on "is this MoE" needs it, and on this backend the graph
	// is the only thing that was parsed.
	MoE bool
	// Experts and ExpertsPerTok are the routed geometry, from the same graph node. A
	// caller whose library boundary validates "DP > 1 needs >= 2 experts" needs the
	// counts themselves, not just the boolean, so that invariant is checked against the
	// model rather than waived for this backend.
	Experts       int
	ExpertsPerTok int
	Quantization  string
	CacheDType    string
	// PrefixCachingDisabled is the engine's --no-enable-prefix-caching, already resolved
	// from the scenario's tri-state against vLLM's default of ON.
	PrefixCachingDisabled bool
	// DeviceMemoryGiB is the chip's total memory. vLLM resolves max_num_seqs and
	// max_num_batched_tokens from it for a deployment that passes neither, so reproducing
	// that resolution needs it.
	DeviceMemoryGiB float64
	BlockSize       int64
	GPUMemUtil      float64
}
