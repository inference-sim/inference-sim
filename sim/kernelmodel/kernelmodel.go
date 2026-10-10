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
// It does not load artifacts. blis-latency-kernel's OpenInputs loads a scenario's
// documents and its New builds the kernel from them, so this package holds only what is
// genuinely its own: adapting a kernel to sim.LatencyModel, translating a BLIS batch into
// the kernel's ReqShape, and turning the kernel's byte answers into a KV block count.
//
// It holds the kernel as the blis-schemas kernel.Kernel INTERFACE, not the concrete type.
// Everything the adapter reads about configuration comes through that interface: engine
// settings from Deployment(), widths from Resolved(). The one thing it does not -- the
// deployment's identity (model name, chip, expert geometry) -- is retained from the
// Inputs the kernel was built from, see identity.
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
	"math"
	"path/filepath"
	"time"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"

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
	k  kernel.Kernel
	id identity

	decodeThreshold int
	smBudget        int
	// specTokens is the pool's draft length (engine.speculative.num_spec_tokens), 0 when it
	// does not speculate. See batchOf.
	specTokens int

	// Host costs are invariant per kernel, so they are converted to ticks once. StepTime
	// is called once per simulated step; a Duration-to-ticks division per call is wasted.
	outputTokenTicks int64
	completionTicks  int64

	// No Scenario, Deployment or pool index is retained. Engine settings and widths are
	// read from the kernel, which resolved this pool: holding the documents to index back
	// into Pools[poolIndex] would be a second answer to "which pool is this", and a
	// disagreement prices a decode pool at a prefill pool's parallelism with nothing
	// reporting it. Do not reintroduce that while extending identity below.
}

// identity is the deployment identity an alternative backend is configured from: the
// model's name, the chip, and the routed expert geometry.
//
// It is NOT configuration the kernel resolved, so it is not read from the kernel: the
// kernel deliberately does not re-export identity (blis-schemas#36 draws that line --
// identity for comparison is the harness's concern, not the cost model's). It is taken
// from the same Inputs the kernel was built from, so it describes the chip and graph the
// kernel priced rather than a second load of files that could resolve differently.
type identity struct {
	model         string
	hardware      string
	memoryGiB     float64
	smCount       int
	experts       int
	expertsPerTok int

	// The pool's extent in the cluster, which places its instances. A deployment's pools fill
	// the cluster's nodes exactly (blis-schemas' placement contract); BLIS places them in
	// declaration order, so pool i occupies the nodes after pools 0..i-1.
	role        deployment.Role
	firstNode   int
	poolNodes   int
	gpusPerNode int
	gpusPerRack int
	// rankGPUs is the GPUs one data-parallel rank occupies: pp x tp x pcp.
	rankGPUs int
}

func identityOf(in latencykernel.Inputs) identity {
	experts, topK := expertCounts(in.Model)
	pool := in.Deployment.Pools[in.PoolIndex]
	first := 0
	for _, p := range in.Deployment.Pools[:in.PoolIndex] {
		first += p.Nodes
	}
	pl := pool.Parallel
	return identity{
		model:         in.Scenario.Model,
		hardware:      in.Chip.Name,
		memoryGiB:     in.Chip.MemoryGiB,
		smCount:       in.Chip.SMCount,
		experts:       experts,
		expertsPerTok: topK,
		role:          pool.Role,
		firstNode:     first,
		poolNodes:     pool.Nodes,
		gpusPerNode:   in.Scenario.Cluster.GPUsPerNode,
		gpusPerRack:   in.Scenario.Cluster.GPUsPerRack,
		rankGPUs:      max(1, pl.PP) * max(1, pl.TP) * max(1, pl.PCP),
	}
}

// expertCounts returns the routed expert count and top-k of the first routed layer kind,
// or zeros for a dense model. A layer kind carrying a GroupedGEMM node is what makes a
// model routed; the graph is the only model document this backend parses.
func expertCounts(g *model.Graph) (experts, topK int) {
	for _, kind := range g.LayerKinds {
		for _, n := range kind.Nodes {
			if n.Op == model.OpGroupedGEMM {
				return n.Experts, n.TopK
			}
		}
	}
	return 0, 0
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
// The loading is blis-latency-kernel's own: OpenInputs is the loading half of its OpenPool,
// and New is the same constructor OpenPool ends in, so this builds exactly the kernel
// OpenPool would -- validation included -- while keeping the Inputs the identity is read
// from. One answer to "which files does this scenario imply", in the repository that owns
// the question.
func OpenPool(scenario string, r Repos, poolIndex int) (*Model, error) {
	in, err := latencykernel.OpenInputs(scenario, r, poolIndex)
	if err != nil {
		return nil, err
	}
	k, err := latencykernel.New(in)
	if err != nil {
		return nil, err
	}
	return newModel(k, identityOf(in)), nil
}

func newModel(k kernel.Kernel, id identity) *Model {
	return &Model{
		k:                k,
		id:               id,
		decodeThreshold:  DecodeThreshold,
		smBudget:         id.smCount,
		specTokens:       specTokensOf(k),
		outputTokenTicks: ticks(k.OutputTokenOverhead()),
		completionTicks:  ticks(k.CompletionOverhead()),
	}
}

// OpenRole opens the pool of a scenario that serves role -- the prefill or the decode pool of
// a disaggregated deployment -- refusing a scenario that states no such pool or more than one.
func OpenRole(scenario string, r Repos, role deployment.Role) (*Model, error) {
	_, dep, err := latencykernel.LoadBundle(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return nil, err
	}
	index := -1
	for i, p := range dep.Pools {
		if p.Role == role {
			if index >= 0 {
				return nil, fmt.Errorf("%s states more than one %s pool; one engine layout per role is "+
					"what a simulated pool runs", scenario, role)
			}
			index = i
		}
	}
	if index < 0 {
		return nil, fmt.Errorf("%s states no %s pool", scenario, role)
	}
	return OpenPool(scenario, r, index)
}

// Role is the role of the pool this model prices.
func (m *Model) Role() deployment.Role { return m.id.role }

// RankCapacity is how many data-parallel ranks the pool's nodes hold at its layout: the pool's
// GPUs divided by the GPUs one rank occupies. A simulator running one replica per rank may run
// at most this many in the pool.
func (m *Model) RankCapacity() int {
	return m.id.poolNodes * m.id.gpusPerNode / m.id.rankGPUs
}

// PlacementOf places the pool's rank-th replica, its ranks packed onto the pool's nodes in
// order: rank r occupies GPUs [r*rankGPUs, (r+1)*rankGPUs) of the pool and sits on the node
// holding its first GPU. The rack is the node's position in GPUsPerRack-sized domains; a
// cluster stating none gives every node its own, so no two nodes are taken to share an NVLink
// domain the scenario never described.
func (m *Model) PlacementOf(rank int) kernel.Placement {
	node := m.id.firstNode
	if m.id.gpusPerNode > 0 {
		node += rank * m.id.rankGPUs / m.id.gpusPerNode
	}
	rack := node
	if m.id.gpusPerRack > 0 && m.id.gpusPerNode > 0 {
		rack = node * m.id.gpusPerNode / m.id.gpusPerRack
	}
	return kernel.Placement{Node: node, Rack: rack, Pool: m.id.role}
}

// PDTransferTicks prices moving tokens tokens of one request's KV between two placements, as
// the kernel prices it, rounded UP to a whole tick: a transfer is not complete until its last
// byte lands, and a zero-tick transfer would let decode start in the same instant.
//
// It panics when the kernel cannot price the handoff -- it reports an unbounded duration when
// the link it would cross has no bandwidth (a fabric or chip stating none) -- since charging
// any finite time for it would invent a number the kernel refused to give.
func (m *Model) PDTransferTicks(tokens int64, from, to kernel.Placement) int64 {
	d := m.k.PDTransferTime(int(tokens), from, to)
	if d < 0 || d >= time.Duration(math.MaxInt64/2) {
		panic(fmt.Sprintf("kernelmodel: the kernel cannot price a %d-token KV handoff from %+v to %+v "+
			"(it reports %v): the link between them states no bandwidth -- check the scenario's "+
			"cluster.fabric and the chip's IntraNodeBwGBps", tokens, from, to, d))
	}
	return max(1, (d.Nanoseconds()+999)/1000)
}

// Kernel exposes the adapted kernel. A caller that needs a memory or provenance answer
// asks the kernel directly rather than having this adapter grow a passthrough per method.
func (m *Model) Kernel() kernel.Kernel { return m.k }

// StepTime prices one forward pass over the scheduled batch.
//
// The kernel reports a band -- Overlap sums the max over resources within each stage,
// NoOverlap sums every resource -- and this returns the NoOverlap edge. That is a measured
// choice: over 219 points of NVIDIA's FPM dataset (one synchronized whole-forward
// iteration at a known batch and KV-token count, two models, two parts, five parallelism
// topologies), Overlap carries a one-sided deficit and NoOverlap is nearly unbiased and
// closer on most points. blis-registry's docs/band-selection.md records the figures. The
// physical reason is PIECEWISE cudagraph mode: attention runs eagerly between captured
// segments, so per-layer overlap is structurally limited. The kernel's own StepTime comment
// states the conclusion: a caller that wants one figure should read NoOverlap.
//
// FPM is used only to choose between the kernel's own two edges. It is not fitted
// against, and the InferenceX corpus BLIS is scored on is never used for either.
func (m *Model) StepTime(batch []*sim.Request) int64 {
	b := m.batchOf(batch)
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

// batchOf translates a BLIS batch into the kernel's.
//
// Under speculative decoding a decode verifies its whole draft every step: the engine runs
// 1 + num_spec_tokens positions per decoding request whatever was later accepted, and the
// blis-schemas contract states Scheduled that way ("the draft length plus one"). BLIS's
// NumNewTokens is the ADVANCE -- 1 + the accepted drafts, set by batch formation from the
// acceptance rate -- which decides how far the request moves, not what the forward pass
// cost. So a decode's Scheduled is the verify width here, and NumNewTokens keeps its
// meaning for the simulator.
func (m *Model) batchOf(batch []*sim.Request) kernel.Batch {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, 0, len(batch)),
		DecodeThreshold: m.decodeThreshold,
		SMBudget:        m.smBudget,
	}
	for _, req := range batch {
		r := shapeOf(req)
		if m.specTokens > 0 && req.ProgressIndex >= req.InputLen() {
			r.Scheduled = 1 + m.specTokens
		}
		b.Reqs = append(b.Reqs, r)
	}
	return b
}

// specTokensOf is the draft length of the pool a kernel prices.
func specTokensOf(k kernel.Kernel) int {
	if sp := k.Deployment().Engine.Speculative; sp != nil && sp.NumSpecTokens > 0 {
		return sp.NumSpecTokens
	}
	return 0
}

// shapeOf translates one BLIS request into the kernel's request shape.
//
// The mapping, and why each field is what it is:
//
//	Scheduled    <- NumNewTokens. Tokens this step, set by FormBatch -- the ADVANCE.
//	                Under speculation batchOf replaces it, for a decode, with the verify
//	                width 1 + num_spec_tokens, which is what the forward pass runs.
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
	e := m.k.Deployment().Engine
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
	return m.k.Resolved().DataParallel()
}

// StepEstimate exposes the kernel's full band for one batch: the overlap edge, the serialized
// edge, the binding resource and the per-resource breakdown.
//
// StepTime returns only the NoOverlap edge, because sim.LatencyModel is a single int64. A
// caller diagnosing WHY a step costs what it does needs the rest, and asking the kernel again
// through this method is cheaper and less error-prone than rebuilding the batch translation.
func (m *Model) StepEstimate(batch []*sim.Request) kernel.StepEstimate {
	b := m.batchOf(batch)
	return m.k.StepTime(b)
}

// Deployment reports the scenario facts that identify the deployment this kernel models:
// the model name, chip, parallel widths and precisions, read from the SAME scenario the
// kernel was opened from, so a caller configuring a simulator describes one deployment.
//
// Nothing here is a latency coefficient: this is deployment identity, not calibration.
func (m *Model) Deployment() Deployment {
	// The layout comes from the Deployment and the identity from the Scenario, which is
	// exactly the v0.2.0 split: what traffic runs on which hardware is the problem, how
	// it is laid out is the choice. Both are the pair this model was opened from, so the
	// arms cannot describe different deployments.
	e := m.k.Deployment().Engine
	r := m.k.Resolved()
	return Deployment{
		Model:    m.id.model,
		Hardware: m.id.hardware,
		TP:       r.TensorParallel(),
		DP:       r.DataParallel(),
		// The RESOLVED width, not the request: expert parallelism is on when the group is
		// wider than one rank, which is what the kernel settled.
		ExpertParallel: r.ExpertParallel() > 1,
		MoE:            m.id.experts > 0,
		Experts:        m.id.experts,
		ExpertsPerTok:  m.id.expertsPerTok,
		Quantization:   e.Quantization,
		CacheDType:     e.CacheDType,
		// The chip's total memory, which is what vLLM resolves its batch defaults from.
		DeviceMemoryGiB: m.id.memoryGiB,
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

// Shape is what a scenario's deployment states beyond any one pool, read without building a
// kernel: its pools' roles, in declaration order, and whether it states an offload hierarchy.
type Shape struct {
	Roles         []deployment.Role
	StatesOffload bool
}

// ShapeOf reads a scenario's Shape.
func ShapeOf(scenario string, r Repos) (Shape, error) {
	_, dep, err := latencykernel.LoadBundle(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return Shape{}, err
	}
	sh := Shape{Roles: make([]deployment.Role, len(dep.Pools)), StatesOffload: dep.Offload != nil}
	for i, p := range dep.Pools {
		sh.Roles[i] = p.Role
	}
	return sh, nil
}
