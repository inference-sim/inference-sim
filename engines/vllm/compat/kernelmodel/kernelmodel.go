// Moved from inference-sim/sim/kernelmodel/kernelmodel.go at ca532abca209af41425c007974a5f3ef336f5981.

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
	"sort"
	"strings"
	"time"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-latency-kernel/enginesettings"
	"github.com/inference-sim/blis-latency-kernel/vllmsettings"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/acceptance"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/workload"
	vllmrules "github.com/inference-sim/blis-vllm/rules"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
)

// Repos is blis-latency-kernel's Repos.
//
// An alias rather than a second type: the kernel's Open takes one, so a distinct type here
// would mean converting at every call and would let the two drift if the kernel ever adds a
// root. A kernelmodel.Repos{...} literal still compiles unchanged; every entry point below
// supplies the engine rules a literal leaves unset (withEngineRules).
type Repos = latencykernel.Repos

// withEngineRules returns r with the vLLM engine rules filled in where r states none: the
// packs of github.com/inference-sim/blis-vllm/rules, which the kernel takes from its caller
// rather than looking up itself (inference-sim#2021 B4). It is the one place this repository
// chooses the rules a scenario's engine_version selects, so a run, a replay and every scoring
// tool price a release against the same pack.
func withEngineRules(r Repos) Repos {
	if r.Rules == nil {
		r.Rules = latencykernel.RulesFrom(vllmrules.Lookup)
		r.RulesVersions = vllmrules.Versions
	}
	if r.EngineSettings == nil {
		// The kernel prices the engine settings its caller decodes, as it takes its rules from
		// its caller: vLLM's engine block, read by the kernel module's one conversion.
		r.EngineSettings = vllmsettings.OfPool
	}
	return r
}

// Model adapts a kernel to sim.LatencyModel.
//
// decodeThreshold and smBudget are fixed at construction because they are engine and
// hardware configuration rather than per-step state, and re-deriving them inside the hot
// StepTime path would cost more than storing them.
type Model struct {
	k  kernel.Kernel
	id identity
	// engine is the pool's engine block as this package decodes it, once at open: the kernel
	// prices the settings converted from it (Inputs.Engine), and the simulator's settings and
	// sizing read the rest (Engine).
	engine engineconfig.Engine

	decodeThreshold int
	smBudget        int
	// specTokens is the pool's draft length (engine.speculative.num_spec_tokens), 0 when it
	// does not speculate. See batchOf.
	specTokens int
	// pad is how a lockstep data-parallel step pads this pool's ranks (#1909).
	pad dpPadding

	// Host costs are invariant per kernel, so they are converted to ticks once. StepTime
	// is called once per simulated step; a Duration-to-ticks division per call is wasted.
	outputTokenTicks int64
	completionTicks  int64

	// graph is the model graph the kernel was built from, for V032Pieces.
	graph *model.Graph
	// rules is the release-rule report taken from the Inputs at open: the rules no one
	// enforces on this pool and what they find (#1942). See Provenance.
	rules ruleReport
	// routing is the pool's routing-imbalance resolution for the results file.
	routing *sim.RoutingRecord
	// cacheDType is the cache dtype that runs, as the release's rules resolve the pool's
	// requested one for the model's architecture, the chip's family and the TP width
	// (rules.EngineRules.ResolveCacheDType), the key a cache record is read by (KVLayers).
	cacheDType string

	// No Scenario, Deployment or pool index is retained. Engine settings and widths are
	// read from the kernel, which resolved this pool: holding the documents to index back
	// into Pools[poolIndex] would be a second answer to "which pool is this", and a
	// disagreement prices a decode pool at a prefill pool's parallelism with nothing
	// reporting it. Do not reintroduce that while extending identity below.
}

// identity is the deployment identity the simulator is configured from beside the kernel:
// the model's name, the chip, and the routed expert geometry.
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
	// defaultWindow is the graph's derived context window (GlobalShape.DefaultWindow), what an
	// unstated or auto max_model_len starts from; 0 when the graph states none.
	defaultWindow int

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
	// tpRanks is the pool's TP width; latentKV is whether every attention layer stores one
	// latent vector per token (MLA), whose cache every TP rank holds alike.
	tpRanks  int
	latentKV bool
	// prefillBackoff is NIXL's _prefill_backoff for this pool: the trailing prompt tokens a
	// disaggregated prefill does not compute and the decode side recomputes.
	prefillBackoff int
	// prefix is the KV cache group layout a prefix hit is cut by (PrefixHit).
	prefix prefixLayout
	// capabilityMajor is the chip's compute-capability major version, the family the kernel
	// resolves the model graph for (ForCapability); 0 when the chip states none.
	capabilityMajor int
}

func identityOf(in latencykernel.Inputs, engine engineconfig.Engine) identity {
	experts, topK := expertCounts(in.Model)
	pool := in.Deployment.Pools[in.PoolIndex]
	first := 0
	for _, p := range in.Deployment.Pools[:in.PoolIndex] {
		first += p.Nodes
	}
	pl := pool.Parallel
	return identity{
		model:           in.Scenario.Model,
		hardware:        in.Chip.Name,
		memoryGiB:       in.Chip.MemoryGiB,
		smCount:         in.Chip.SMCount,
		experts:         experts,
		expertsPerTok:   topK,
		defaultWindow:   in.Model.Global.DefaultWindow(),
		role:            pool.Role,
		firstNode:       first,
		poolNodes:       pool.Nodes,
		gpusPerNode:     in.Scenario.Cluster.GPUsPerNode,
		gpusPerRack:     in.Scenario.Cluster.GPUsPerRack,
		rankGPUs:        max(1, pl.PP) * max(1, pl.TP) * max(1, pl.PCP),
		tpRanks:         max(1, pl.TP),
		latentKV:        allLatent(in.Model),
		prefillBackoff:  prefillBackoffOf(in.Model, engine.Speculative),
		prefix:          prefixLayoutOf(in.Model, pool, engine),
		capabilityMajor: func() int { f, _ := in.Chip.CapabilityMajor(); return f }(),
	}
}

// prefillBackoffOf is vLLM v0.31.0's NixlConnector _prefill_backoff
// (nixl/base_scheduler.py:385-398): max(1 if the model has Mamba-style recurrent layers,
// num_prefill_lookahead_tokens - 1). The lookahead (config/vllm.py:657-678) is 0 without an
// Eagle-family drafter, num_speculative_tokens for multi-module MTP (method mtp with
// min(draft layers, K) > 1, config/speculative.py:1973-1979), and 1 for every other Eagle-family
// drafter -- DSpark included (speculative.py:1938-1942) -- so DSpark backs off nothing.
func prefillBackoffOf(g *model.Graph, sp *engineconfig.Speculative) int {
	backoff := 0
	g.EachNode(func(_ string, n model.Node) {
		if n.Op == model.OpRecurrentUpdate {
			backoff = 1
		}
	})
	if sp != nil && sp.Method == "mtp" && g.Speculator != nil &&
		min(g.Speculator.Stack.Layers(), sp.NumSpecTokens) > 1 {
		backoff = max(backoff, sp.NumSpecTokens-1)
	}
	return backoff
}

// PrefillBackoffTokens is the pool's NIXL prefill backoff (prefillBackoffOf): under P/D the
// decode side recomputes this many trailing prompt tokens instead of pulling them.
func (m *Model) PrefillBackoffTokens() int { return m.id.prefillBackoff }

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
	r = withEngineRules(r)
	in, err := latencykernel.OpenInputs(scenario, r, poolIndex)
	if err != nil {
		return nil, err
	}
	// One decoding of the pool's engine block, which the routing resolution, the model and the
	// kernel all read: this package keeps it (Model.engine), and the kernel prices the settings
	// converted from that same copy, never from a second read.
	engine, err := decodeEngine(in.Deployment.Pools[in.PoolIndex])
	if err != nil {
		return nil, fmt.Errorf("%s: %w", scenario, err)
	}
	if in.Routing, err = resolveRouting(in, engine, r.Registry, routingOverrides); err != nil {
		return nil, fmt.Errorf("%s: %w", scenario, err)
	}
	return fromDecoded(in, engine)
}

// fromInputs builds and adapts the kernel for loaded Inputs, decoding the priced pool's engine
// block first; see fromDecoded.
func fromInputs(in latencykernel.Inputs) (*Model, error) {
	engine, err := decodeEngine(in.Deployment.Pools[in.PoolIndex])
	if err != nil {
		return nil, err
	}
	return fromDecoded(in, engine)
}

// fromDecoded builds and adapts the kernel for loaded Inputs and the priced pool's decoded engine
// block, refusing a speculative method the scenario's engine release does not accept first.
func fromDecoded(in latencykernel.Inputs, engine engineconfig.Engine) (*Model, error) {
	if err := checkSpeculativeMethod(in, engine); err != nil {
		return nil, err
	}
	pool := in.Deployment.Pools[in.PoolIndex]
	if err := checkDPLocal(pool.Parallel, in.Scenario.Cluster.GPUsPerNode); err != nil {
		return nil, fmt.Errorf("%s pool: %w", pool.Role, err)
	}
	settings := vllmsettings.Of(engine)
	in.Engine = &settings
	k, err := latencykernel.New(in)
	if err != nil {
		return nil, err
	}
	m, err := newModel(k, identityOf(in, engine))
	if err != nil {
		return nil, err
	}
	m.engine = engine
	m.rules = reportRules(in)
	m.routing = routingRecordOf(in.Routing)
	m.graph = in.Model
	if in.Rules != nil {
		m.cacheDType = in.Rules.ResolveCacheDType(in.Model.Global.Architecture, m.id.capabilityMajor,
			max(pool.Parallel.TP, 1), engine.CacheDType)
	}
	return m, nil
}

func newModel(k kernel.Kernel, id identity) (*Model, error) {
	spec := specTokensOf(k)
	pad, err := dpPaddingOf(k, spec)
	if err != nil {
		return nil, err
	}
	return &Model{
		k:  k,
		id: id,
		// vLLM classifies a verify of up to 1 + K tokens as a decode (reorder_batch_threshold
		// is raised to it; for V4.1, vllm/models/deepseek_v41/sparse_mla.py:160), so a
		// speculating pool widens the threshold to cover its verify width (#1932).
		decodeThreshold:  max(DecodeThreshold, 1+spec),
		smBudget:         id.smCount,
		specTokens:       spec,
		pad:              pad,
		outputTokenTicks: ticks(k.OutputTokenOverhead()),
		completionTicks:  ticks(k.CompletionOverhead()),
	}, nil
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

// OffloadCopies is how many workers' copies of a block vLLM's CPU offload tier holds for
// this pool: 1 when every KV-holding attention layer is latent (MLA) and the engine runs one
// node's TP group (the replicated layout, offloading/config.py:158-171 at v0.31.0), else the
// engine's world size, PP x TP x PCP (kv_offload/cpu/spec.py:103-121).
func (m *Model) OffloadCopies() int64 {
	if m.id.latentKV && m.id.rankGPUs == m.id.tpRanks && m.id.rankGPUs <= max(1, m.id.gpusPerNode) {
		return 1
	}
	return int64(m.id.rankGPUs)
}

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

// EndpointPlacement reports whether one simulated engine rank fits on one
// node, and whether its rack was declared rather than inferred solely for
// conservative transfer pricing. A rank can cross a node boundary even when
// rankGPUs <= gpusPerNode, so check both ends of its GPU allocation.
func (m *Model) EndpointPlacement(rank int) (kernel.Placement, int, bool) {
	p := m.PlacementOf(rank)
	firstGPU := rank * m.id.rankGPUs
	lastGPU := firstGPU + m.id.rankGPUs - 1
	if m.id.gpusPerNode <= 0 {
		return p, 0, m.id.gpusPerRack > 0
	}
	nodeCount := lastGPU/m.id.gpusPerNode - firstGPU/m.id.gpusPerNode + 1
	return p, nodeCount, m.id.gpusPerRack > 0
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

// V032Pieces names the parts of the pool's deployment that blis-vllm's v0.32.0 engine
// (core/vllm) does not build yet, from the kernel's model graph and resolved layout:
//
//   - "hybrid KV groups": an attention layer whose KV cache spec KVLayers does not map, such as
//     a window or state cache or recurrent state, which vLLM manages as its own group
//     (HybridKVCacheCoordinator, vllm/v1/core/kv_cache_coordinator.py:609@5df63a72);
//   - "data parallelism (dp N)": more than one data-parallel rank, whose waves, lockstep and
//     balancer run across engines (vllm/v1/engine/core.py:2143-2414@5df63a72).
//
// It returns none for a pool the v0.32.0 engine builds as it stands.
func (m *Model) V032Pieces() []string {
	var out []string
	if _, err := m.KVLayers(); err != nil {
		out = append(out, "hybrid KV groups ("+err.Error()+")")
	}
	return out
}

// MTPDraftLayers is the pool's MTP drafter's layer count, the checkpoint's
// num_nextn_predict_layers, as the model graph's speculator stack holds it; 0 when the pool does
// not draft with method mtp or the graph describes no speculator. It sets the prefill lookahead
// vLLM reserves (use_multi_module_mtp, vllm/config/speculative.py:1982-1988@5df63a72), as
// prefillBackoffOf reads it.
func (m *Model) MTPDraftLayers() int {
	return mtpDraftLayersOf(m.graph, m.engine.Speculative)
}

func mtpDraftLayersOf(g *model.Graph, sp *engineconfig.Speculative) int {
	if sp == nil || sp.Method != "mtp" || g == nil || g.Speculator == nil {
		return 0
	}
	return g.Speculator.Stack.Layers()
}

// ReqShape is one request's share of a batch in the kernel's terms (blis-schemas
// kernel.ReqShape), for a caller pricing batches it formed itself, such as blis-vllm's v0.32.0
// engine through vllmengine.
type ReqShape = kernel.ReqShape

// StepTimeOfShapes prices a batch already in the kernel's shapes, exactly as StepTime prices a
// BLIS batch once batchOf has translated it: the kernel's NoOverlap step time in ticks, at
// least one. The shapes are the caller's: under speculation a decode's Scheduled is already
// its verify width.
func (m *Model) StepTimeOfShapes(reqs []ReqShape) int64 {
	b := kernel.Batch{Reqs: reqs, DecodeThreshold: m.decodeThreshold, SMBudget: m.smBudget}
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
//
// INV-K1 (no BLIS-side draft cost): the step price is the kernel's StepTime of exactly this
// verify-width batch, floored at one tick, and nothing else. blis-latency-kernel v0.3.0 prices
// the drafter's passes, its context stack and its head inside StepTime, so a draft term added
// here, or a Scheduled widened past 1 + K, would charge the drafts twice.
func (m *Model) batchOf(batch []*sim.Request) kernel.Batch {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, 0, len(batch)),
		DecodeThreshold: m.decodeThreshold,
		SMBudget:        m.smBudget,
	}
	for _, req := range batch {
		r := shapeOf(req)
		if m.specTokens > 0 && req.ProgressIndex >= req.PrefillEnd() {
			r.Scheduled = 1 + m.specTokens
		}
		b.Reqs = append(b.Reqs, r)
	}
	return b
}

// specTokensOf is the draft length of the pool a kernel prices, from the engine settings it
// prices with.
func specTokensOf(k kernel.Kernel) int {
	if e, ok := k.(interface {
		Engine() enginesettings.Settings
	}); ok {
		if sp := e.Engine().Speculative; sp != nil && sp.NumSpecTokens > 0 {
			return sp.NumSpecTokens
		}
		return 0
	}
	if e, err := decodeEngine(k.Deployment()); err == nil && e.Speculative != nil && e.Speculative.NumSpecTokens > 0 {
		return e.Speculative.NumSpecTokens
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
//	PromptLen    <- PrefillEnd(): the prompt, and for a request resumed after a preemption
//	                its re-prefilled output too. Computed against PromptLen is what separates a
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
		PromptLen:    int(req.PrefillEnd()),
		CachedTokens: 0,
	}
}

// ticks converts a kernel Duration to BLIS's int64 microsecond tick.
//
// Truncation rather than rounding, as the coefficient backends this kernel replaced did
// (they truncated a float microsecond value). The largest error is 1 tick, which on the
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
// DPLoadBalancing is the pool's parallel.data_parallel_load_balancing as stated ("" is
// internal; blis-schemas#76).
func (m *Model) DPLoadBalancing() deployment.DPLoadBalancing {
	return m.k.Deployment().Parallel.DataParallelLoadBalancing
}

func (m *Model) Engine() (engineconfig.Engine, error) {
	e := m.engine
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
	e := m.engine
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

// Deployment is the scenario identity the simulator's deployment is built from.
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
// kernel: its pools' roles, in declaration order, and whether it states an offload hierarchy;
// what its Scenario states that no kernel prices: its acceptance pin and its workload
// binding, which the simulator's speculative acceptance resolves from (#1939); and the llm-d
// router's inputs -- the router block and each pool's pod labels.
type Shape struct {
	Roles         []deployment.Role
	StatesOffload bool
	// PDConnector is the deployment's pd_transfer.connector, "" when it states none.
	PDConnector string
	// Offload is the deployment's offload block, nil when it states none.
	Offload *deployment.Offload
	// LongPrefillTokenThresholds and DPLoadBalancings are each pool's
	// engine.long_prefill_token_threshold (blis-schemas#75; 0 = unstated) and
	// parallel.data_parallel_load_balancing (#76; "" = unstated), in pool order.
	LongPrefillTokenThresholds []int
	DPLoadBalancings           []deployment.DPLoadBalancing
	// Router is the deployment's router block, or nil when it states none. Its paths are
	// relative to Dir.
	Router *deployment.Router
	// Labels are each pool's pod labels, parallel to Roles; nil for a pool that states none.
	Labels []map[string]string
	// Dir is the directory of the scenario file, against which the router block's paths resolve.
	Dir string

	Model           string
	Acceptance      *acceptance.Pin
	WorkloadBinding *workload.Binding
	// EngineVersion is the scenario's engine_version: the engine release its deployment was
	// written for.
	EngineVersion string
	// Unmodelled are each pool's stated engine settings that the compatibility engine does not
	// model, by their scenario field names (unmodelledSettings), parallel to Roles.
	Unmodelled [][]string
}

// unmodelledSettings are the engine settings a deployment states that the compatibility engine
// does not model: the ones blis-schemas added with the v0.32.0 engine (blis-schemas#102), which
// core/vllm reads. It returns their scenario field names, in declaration order.
func unmodelledSettings(e engineconfig.Engine) []string {
	var out []string
	add := func(stated bool, name string) {
		if stated {
			out = append(out, name)
		}
	}
	add(e.LongPrefillTokenThresholdAdaptive != nil, "long_prefill_token_threshold_adaptive")
	add(e.MaxNumScheduledTokens != 0, "max_num_scheduled_tokens")
	add(e.MaxNumActiveSeqs != 0, "max_num_active_seqs")
	add(e.Watermark != nil, "watermark")
	add(e.PrefillScheduleInterval != 0, "prefill_schedule_interval")
	add(e.StreamInterval != 0, "stream_interval")
	add(e.DisableHybridKVCacheManager != nil, "disable_hybrid_kv_cache_manager")
	add(e.KVCacheMemoryBytes != nil, "kv_cache_memory_bytes")
	add(e.PrefixMatchUnit != 0, "prefix_match_unit")
	// The compatibility engine models vLLM's default interval 0, which keeps only the semantic
	// checkpoints (compat/kv/prefix_hit.go), so a stated 0 is what it runs.
	add(e.PrefixCacheRetentionInterval != nil && *e.PrefixCacheRetentionInterval != 0, "prefix_cache_retention_interval")
	add(e.KVEvents != nil, "kv_events")
	add(e.MaxLoRAs != 0, "max_loras")
	add(e.MaxCPULoRAs != 0, "max_cpu_loras")
	add(e.Speculative != nil && e.Speculative.DraftModel != "", "speculative.draft_model")
	// engine.env: blis-schemas requires a consumer to refuse or disclose every key it cannot
	// honour (spec/deployment/deployment.go Engine.Env), and the compatibility engine reads none,
	// so each stated key is named.
	if len(e.Env) > 0 {
		keys := make([]string, 0, len(e.Env))
		for k := range e.Env {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out = append(out, "env ("+strings.Join(keys, ", ")+")")
	}
	return out
}

// ShapeOf reads a scenario's Shape.
func ShapeOf(scenario string, r Repos) (Shape, error) {
	path := filepath.Join(r.Scenarios, scenario)
	sc, dep, err := latencykernel.LoadBundle(path)
	if err != nil {
		return Shape{}, err
	}
	sh := Shape{Roles: make([]deployment.Role, len(dep.Pools)), StatesOffload: dep.Offload != nil,
		Model: sc.Model, Acceptance: sc.Acceptance, WorkloadBinding: sc.Workload, EngineVersion: sc.EngineVersion,
		Router: dep.Router, Labels: make([]map[string]string, len(dep.Pools)), Dir: filepath.Dir(path)}
	if dep.PDTransfer != nil {
		sh.PDConnector = dep.PDTransfer.Connector
	}
	if dep.Offload != nil {
		o := *dep.Offload
		o.Tiers = append([]deployment.Tier(nil), dep.Offload.Tiers...)
		sh.Offload = &o
	}
	for i, p := range dep.Pools {
		e, err := decodeEngine(p)
		if err != nil {
			return Shape{}, fmt.Errorf("%s: %w", scenario, err)
		}
		sh.Roles[i] = p.Role
		sh.Labels[i] = p.Labels
		sh.LongPrefillTokenThresholds = append(sh.LongPrefillTokenThresholds, e.LongPrefillTokenThreshold)
		sh.DPLoadBalancings = append(sh.DPLoadBalancings, p.Parallel.DataParallelLoadBalancing)
		sh.Unmodelled = append(sh.Unmodelled, unmodelledSettings(e))
	}
	return sh, nil
}

// checkSpeculativeMethod refuses a pool whose engine.speculative.method the scenario's engine
// release does not accept, read from that release's rules (rules.EngineRules
// SpeculativeMethodAccepted), so the accepted set moves with engine_version: dspark is
// accepted at vLLM 0.31.0 because that release ships it.
//
// The kernel does not check it itself. It refuses a method the model graph's speculator
// does not describe, and a combination vLLM refuses (MTP on DeepSeek-V4.1), but it applies
// only the field layer of blis-schemas validation, so on a graph with no speculator any
// label would price. The whole rule layer is not applied here either: it also checks the
// graph's own speculator name against the release, which refuses committed fixtures that do
// not speculate at all.
func checkSpeculativeMethod(in latencykernel.Inputs, engine engineconfig.Engine) error {
	sp := engine.Speculative
	if sp == nil {
		return nil
	}
	if in.Rules == nil {
		return fmt.Errorf("engine.speculative.method %q cannot be checked: no engine rules for "+
			"version %q", sp.Method, in.Scenario.EngineVersion)
	}
	if !in.Rules.SpeculativeMethodAccepted(sp.Method) {
		return fmt.Errorf("engine.speculative.method %q is not accepted by engine %s; it accepts %s",
			sp.Method, in.Rules.EngineVersion(), strings.Join(in.Rules.SpeculativeMethodNames(), ", "))
	}
	return nil
}

// checkDPLocal refuses a stated parallel.dp_local that places a replica's ranks on other nodes
// than PlacementOf's packing does. vLLM puts local data-parallel rank i at devices
// [i x world, i x world + local_world) of its node (get_physical_gpu_ids_for_local_dp_rank,
// vllm/v1/engine/utils.py:433-470@5df63a72), dp_local ranks to a node, while PlacementOf packs
// the ranks onto the pool's nodes in order. The two agree when every rank of a replica lands on
// the same node either way; otherwise endpoint hosts and P/D transfer prices would come from
// another layout than the deployment's, so the setting is refused rather than dropped.
// A rank wider than a node agrees only one replica to a node group (dp_local 1) of whole nodes.
// An unstated dp_local is the engine's own inference from the launch, which is the packing.
func checkDPLocal(pl deployment.Parallelism, gpusPerNode int) error {
	local := pl.DPLocal
	if local < 1 || gpusPerNode < 1 {
		return nil
	}
	rankGPUs := max(1, pl.PP) * max(1, pl.TP) * max(1, pl.PCP)
	refuse := func() error {
		return fmt.Errorf("parallel.dp_local %d puts %d data-parallel ranks of %d GPUs on each %d-GPU node, which "+
			"the compatibility engine does not model: it packs a replica's ranks onto the pool's nodes in order "+
			"(%d to a node), so it would price and host the ranks on other nodes. Leave dp_local unset, or state %d",
			local, local, rankGPUs, gpusPerNode, max(1, gpusPerNode/rankGPUs), max(1, gpusPerNode/rankGPUs))
	}
	if rankGPUs > gpusPerNode {
		if local != 1 || rankGPUs%gpusPerNode != 0 {
			return refuse()
		}
		return nil
	}
	for i := 0; i < max(1, pl.DP); i++ {
		node := i / local
		first, last := i*rankGPUs/gpusPerNode, ((i+1)*rankGPUs-1)/gpusPerNode
		if first != node || last != node {
			return refuse()
		}
	}
	return nil
}

// allLatent reports whether the model's attention layers all store a latent KV (MLA and its
// sparse form), and it has at least one.
func allLatent(g *model.Graph) bool {
	n := 0
	for _, k := range g.LayerKinds {
		for _, nd := range k.Nodes {
			// The DSA indexer's keys (block_index_scores) live in the MLA group and are one head
			// that every TP rank holds alike, so they keep the layout replicated.
			if nd.Op != model.OpAttention || nd.Role == "block_index_scores" {
				continue
			}
			if !nd.AttentionKind.LatentKV() {
				return false
			}
			n++
		}
	}
	return n > 0
}
