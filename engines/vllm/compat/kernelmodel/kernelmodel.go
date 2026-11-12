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

	blisschemas "github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-substrate/load"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	"github.com/inference-sim/blis-vllm/core/placement"
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
		r.RulesKind = "vllm"
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
	// assumed are the pricing assumptions this package discloses beside the kernel's own
	// (EngineAssumptions), each "name = value: reason".
	assumed []string
	// substitutions are the execution modes priced in place of the ones the deployment states
	// (Provenance().Substitutions).
	substitutions []sim.Substitution
	// indexerRules resolves the DeepSeek indexer's cache dtype, from the release's rules; nil when
	// they carry no such rule (indexerKVDTypeRules).
	indexerRules indexerKVDTypeRules

	decodeThreshold int
	smBudget        int
	// specTokens is the pool's draft length (engine.speculative_config.num_speculative_tokens), 0 when it
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
	// resolved is the pool's engine settings as its release resolves them (resolveEngine): for
	// vLLM 0.32.0 the batch settings a deployment leaves unstated, with each one's source.
	resolved resolvedEngine

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
	fabric        string
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
	// rackHandoff is whether a KV handoff between two nodes of one NVLink rack runs over NVLink.
	// NIXL moves KV over multi-node NVLink only when UCX_CUDA_IPC_ENABLE_MNNVL=y and the KV cache is
	// registered as VMM; otherwise it can only use RDMA or TCP across nodes
	// (docs/features/nixl_connector_usage.md:401@5df63a72). The engine block states the flag
	// (UCX_CUDA_IPC_ENABLE_MNNVL) and cannot yet state the VMM registration, so a stated "y" is read
	// as the run's NVLink path and is disclosed (rackHandoffAssumption); anything else is the NIC.
	rackHandoff bool
	// packaging is the cluster's packaging as blis-substrate resolved it, with each value's
	// source; gpusPerNode and gpusPerRack are its two counts.
	packaging scenario.Packaging
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

// identityOf reads the pool's placement from pk, the cluster's packaging as the scenario and
// the chip's catalog entry resolve it (scenario.Cluster.Packaging), never from the scenario's
// own gpus_per_node and gpus_per_rack, which are zero where it leaves them to the catalog.
func identityOf(in latencykernel.Inputs, engine engineconfig.Engine, pk scenario.Packaging) identity {
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
		fabric:          in.Scenario.Cluster.Fabric,
		memoryGiB:       in.Chip.MemoryGiB,
		smCount:         in.Chip.SMCount,
		experts:         experts,
		expertsPerTok:   topK,
		defaultWindow:   in.Model.Global.DefaultWindow(),
		role:            pool.Role,
		firstNode:       first,
		poolNodes:       pool.Nodes,
		gpusPerNode:     pk.GPUsPerNode,
		gpusPerRack:     pk.GPUsPerRack,
		rackHandoff:     engine.Env[envMNNVL] == "y",
		packaging:       pk,
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
	if in.Routing, err = resolveRouting(in, engine, r.Data, routingOverrides); err != nil {
		return nil, fmt.Errorf("%s: %w", scenario, err)
	}
	// Read by the kernel only under adaptive verification, which sizes its verify budget from it
	// (blis-substrate kernel/adaptive.go); nil otherwise, as the kernel then prices without it.
	if in.Acceptance, err = resolveAcceptance(in, engine, r.Data, acceptanceOverrides); err != nil {
		return nil, fmt.Errorf("%s: %w", scenario, err)
	}
	return fromDecoded(in, engine, r.Data)
}

// fromInputs builds and adapts the kernel for loaded Inputs, decoding the priced pool's engine
// block first; see fromDecoded.
func fromInputs(in latencykernel.Inputs) (*Model, error) {
	engine, err := decodeEngine(in.Deployment.Pools[in.PoolIndex])
	if err != nil {
		return nil, err
	}
	return fromDecoded(in, engine, nil)
}

// fromDecoded builds and adapts the kernel for loaded Inputs and the priced pool's decoded engine
// block, refusing a speculative method the scenario's engine release does not accept first.
//
// data is where in's documents came from: the kernel is built with New, so its coefficient
// lookups are recorded into data's selected-entry report here (design §3.3, VAL-6). A test
// that assembled in by hand passes nil.
func fromDecoded(in latencykernel.Inputs, engine engineconfig.Engine, data *load.Data) (*Model, error) {
	if err := checkSpeculativeMethod(in, engine); err != nil {
		return nil, err
	}
	pool := in.Deployment.Pools[in.PoolIndex]
	pk, err := in.Scenario.Cluster.Packaging(in.Chip)
	if err != nil {
		return nil, fmt.Errorf("the scenario's cluster: %w", err)
	}
	if err := checkDPLocal(pool.Parallel, pk.GPUsPerNode); err != nil {
		return nil, fmt.Errorf("%s pool: %w", pool.Role, err)
	}
	if err := checkMuseSparkParallel(architectureOf(in.Model), in.Chip, pool.Parallel); err != nil {
		return nil, fmt.Errorf("%s pool: %w", pool.Role, err)
	}
	settings := vllmsettings.Of(engine)
	// An unstated block_size, gpu_memory_utilization, max_num_seqs or max_num_batched_tokens the release resolves is priced at the
	// resolved value, so the kernel and Engine describe one engine (resolveEngine).
	r := resolveEngine(in, engine)
	if r.engine.BlockSize > 0 {
		settings.BlockSize = r.engine.BlockSize
	}
	if r.engine.GPUMemoryUtilization > 0 {
		settings.GPUMemoryUtilization = r.engine.GPUMemoryUtilization
	}
	// The cache dtype the architecture's hook resolves an unstated one to, priced as it runs.
	settings.CacheDType = r.engine.CacheDType
	// The batch settings Engine reports and the scheduler runs: left at zero the kernel would
	// fall back to vLLM v0.31.0's defaults, which skip the caps 0.32.0 applies (batched-DP 256,
	// min(max_num_seqs, max_num_batched_tokens), max_num_seqs x max_model_len).
	if r.engine.MaxNumSeqs > 0 {
		settings.MaxNumSeqs = r.engine.MaxNumSeqs
	}
	if r.engine.MaxNumBatchedTokens > 0 {
		settings.MaxNumBatchedTokens = r.engine.MaxNumBatchedTokens
	}
	in.Engine = &settings
	// How an autoregressive drafter's passes run is the engine's to state (SUB-6): vLLM v0.32.0's
	// speculator for eagle, eagle3 and mtp; any other method is left unstated and the kernel refuses it.
	in.DraftFirstPassRows, in.DraftPassRequests = vllmsettings.DraftPasses(engine)
	k, err := latencykernel.New(in)
	if err != nil {
		return nil, err
	}
	if data != nil {
		k.RecordSelected(data)
	}
	m, err := newModel(k, identityOf(in, engine, pk))
	if err != nil {
		return nil, err
	}
	// Every reader below sees the engine as its release resolves it (resolveEngine): the block
	// size, the memory budget and the batch settings the deployment left unstated.
	m.resolved = resolveEngine(in, engine)
	m.engine = m.resolved.engine
	m.assumed = append(languageModelOnlyAssumptions(in.Model, engine), cudagraphAssumptions(engine)...)
	m.assumed = append(m.assumed, rackHandoffAssumptions(m.id)...)
	if s := languageModelOnlySubstitution(in.Model, engine); s != nil {
		m.substitutions = append(m.substitutions, *s)
	}
	if s := cudagraphSubstitution(engine, k.Resolved().Overrides); s != nil {
		m.substitutions = append(m.substitutions, *s)
	}
	m.indexerRules, _ = in.Rules.(indexerKVDTypeRules)
	m.rules = reportRules(in)
	m.routing = routingRecordOf(in.Routing)
	// The graph the kernel priced: the scenario's, with a drafter named by
	// speculative.draft_model grafted on, so the KV groups sized from it are the priced layers.
	g := k.Graph()
	m.graph = &g
	if in.Rules != nil {
		m.cacheDType = in.Rules.ResolveCacheDType(architectureOf(in.Model), m.id.capabilityMajor,
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

// RolePools are the indexes, in the deployment's pools, of the pool serving each role, refusing
// a scenario that states one role in more than one pool. An engine.Spec selects its pool by this
// index (Spec.Pool).
func RolePools(scenario string, r Repos) (map[deployment.Role]int, error) {
	_, dep, err := blisschemas.LoadBundle(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return nil, err
	}
	out := map[deployment.Role]int{}
	for i, p := range dep.Pools {
		if _, dup := out[p.Role]; dup {
			return nil, fmt.Errorf("%s states more than one %s pool; one engine layout per role is "+
				"what a simulated pool runs", scenario, p.Role)
		}
		out[p.Role] = i
	}
	return out, nil
}

// OpenRole opens the pool of a scenario that serves role -- the prefill or the decode pool of
// a disaggregated deployment -- refusing a scenario that states no such pool or more than one.
func OpenRole(scenario string, r Repos, role deployment.Role) (*Model, error) {
	_, dep, err := blisschemas.LoadBundle(filepath.Join(r.Scenarios, scenario))
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

// EngineAssumptions are the assumptions this package makes in pricing the pool's engine block,
// each as "name = value: reason", beside the kernel's own (Provenance). They are empty when the
// block states everything the price depends on.
func (m *Model) EngineAssumptions() []string { return append([]string(nil), m.assumed...) }

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

// WorldSize is one engine's parallel world size, pp x tp x pcp (ParallelConfig.world_size): the
// workers a KV connector runs on, which the OffloadingConnector pends every job on
// (num_workers, vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:335@5df63a72).
func (m *Model) WorldSize() int { return m.id.rankGPUs }

// RankCapacity is how many data-parallel ranks the pool's nodes hold at its layout: the pool's
// GPUs divided by the GPUs one rank occupies. A simulator running one replica per rank may run
// at most this many in the pool.
func (m *Model) RankCapacity() int {
	return m.id.poolNodes * m.id.gpusPerNode / m.id.rankGPUs
}

// Packaging is the cluster's packaging this pool is placed and priced with, as
// blis-substrate resolves it from the scenario and the chip's catalog entry
// (scenario.Cluster.Packaging): the GPUs per node and per rack, and whether each was stated
// or taken from the catalog. A simulator reads it here rather than from the scenario, whose
// fields are zero where it leaves them to the catalog, and records it in the Result's inputs.
func (m *Model) Packaging() scenario.Packaging { return m.id.packaging }

// PlacementOf places the pool's rank-th replica, its ranks packed onto the pool's nodes in
// order: rank r occupies GPUs [r*rankGPUs, (r+1)*rankGPUs) of the pool and sits on the node
// holding its first GPU. The rack is the node's position in GPUsPerRack-sized domains; a
// cluster stating none gives every node its own, so no two nodes are taken to share an NVLink
// domain the scenario never described.
func (m *Model) PlacementOf(rank int) kernel.Placement { return m.placementAt(rank, m.id.rankGPUs) }

// placementAt places the pool's n-th unit of width GPUs, the units packed onto the nodes in order.
func (m *Model) placementAt(n, width int) kernel.Placement {
	node := m.id.firstNode
	if m.id.gpusPerNode > 0 {
		node += n * width / m.id.gpusPerNode
	}
	return kernel.Placement{Node: node, Rack: m.handoffRack(node), Pool: m.id.role}
}

// handoffRack is the rack a handoff reads for a pool's node. It decides which link a handoff between two nodes
// crosses (kernel.PDTransferDemand): one rack means the NVLink domain, two means the NICs. Without the NVLink path
// every node is its own rack, so the handoff is priced over the NIC. Only handoffs read this rack; the pool's
// collectives use EnginePlacementOf, whose racks are the hardware's. PlacementOf and GroupPlacementOf both place
// through it, so a host that runs a rank per instance and one that runs a group agree.
func (m *Model) handoffRack(node int) int {
	if m.id.rackHandoff && m.id.gpusPerRack > 0 && m.id.gpusPerNode > 0 {
		return node * m.id.gpusPerNode / m.id.gpusPerRack
	}
	return node
}

// GroupGPUs is the GPUs of one data-parallel group, dp x pp x tp x pcp: the pod a vLLM server runs on,
// whose ranks sit behind one endpoint and whose engine is one instance of a host that builds the
// whole group as one engine (core/vllm's DPGroup). It equals the GPUs of one rank at dp 1.
func (m *Model) GroupGPUs() int { return m.id.rankGPUs * max(1, m.k.Resolved().DataParallel()) }

// groupLayout is how the pool's nodes hold groups of GroupGPUs: perNode groups on each node when a group fits
// in one, else nodesPer whole nodes for each group. A group is never split across a node boundary it does not
// fill: a group of 6 GPUs on 8-GPU nodes takes 6 of a node's 8 and leaves 2 idle, rather than starting
// at GPU 6 of the node and running into the next, which would overlap the next group's GPUs. A group wider than a
// node must be a whole number of nodes (nodesPer), and one that is not is refused.
func (m *Model) groupLayout() (perNode, nodesPer int, err error) {
	return groupLayoutOf(m.GroupGPUs(), m.id.gpusPerNode)
}

// groupLayoutOf is groupLayout for groups of g GPUs on nodes of n.
func groupLayoutOf(g, n int) (perNode, nodesPer int, err error) {
	switch {
	case n <= 0:
		return 1, 1, nil
	case g <= n:
		return n / g, 1, nil
	case g%n != 0:
		return 0, 0, fmt.Errorf("a data-parallel group of %d GPUs does not fill whole %d-GPU nodes: it would share a node "+
			"with a neighbouring group; state a dp, tp or pp whose product is a divisor or a multiple of the node's GPUs", g, n)
	}
	return 0, g / n, nil
}

// groupCapacityOf is how many groups of g GPUs nodes pool nodes of n GPUs hold; 0 for a refused layout.
func groupCapacityOf(nodes, g, n int) int {
	perNode, nodesPer, err := groupLayoutOf(g, n)
	switch {
	case err != nil:
		return 0
	case n <= 0:
		return nodes
	case nodesPer > 1:
		return nodes / nodesPer
	}
	return nodes * perNode
}

// groupNodeOf is the node (counted from the pool's first) holding the first GPU of the pool's i-th group.
func groupNodeOf(i, g, n int) int {
	perNode, nodesPer, _ := groupLayoutOf(g, n)
	switch {
	case n <= 0:
		return 0
	case nodesPer > 1:
		return i * nodesPer
	}
	return i / max(perNode, 1)
}

// GroupLayoutError is the refusal for a group size the pool's nodes cannot hold without overlap, nil when they can.
func (m *Model) GroupLayoutError() error { _, _, err := m.groupLayout(); return err }

// GroupCapacity is how many groups the pool's nodes hold (groupLayout): 0 for a layout that is refused. A host that
// runs a group as one instance may run at most this many in the pool (RankCapacity counts ranks, for a host that
// runs one instance for each).
func (m *Model) GroupCapacity() int {
	return groupCapacityOf(m.id.poolNodes, m.GroupGPUs(), m.id.gpusPerNode)
}

// GroupPlacementOf places the pool's g-th group on the node holding its first GPU (groupLayout): the groups that
// fit in a node fill it before the next, and a wider group starts at a whole node.
func (m *Model) GroupPlacementOf(g int) kernel.Placement {
	node := m.id.firstNode + groupNodeOf(g, m.GroupGPUs(), m.id.gpusPerNode)
	return kernel.Placement{Node: node, Rack: m.handoffRack(node), Pool: m.id.role}
}

// EndpointGroupPlacement is EndpointPlacement for a host that runs a group as one instance: the first node and
// rack of the pool's g-th group, how many nodes the group spans, and whether the rack was declared.
func (m *Model) EndpointGroupPlacement(g int) (kernel.Placement, int, bool) {
	p := m.GroupPlacementOf(g)
	if m.id.gpusPerNode <= 0 {
		return p, 0, m.id.gpusPerRack > 0
	}
	_, nodesPer, _ := m.groupLayout()
	return p, max(nodesPer, 1), m.id.gpusPerRack > 0
}

// EnginePlacement is where the pool sits, in the engine's terms (core/placement): the chip and fabric the
// scenario names, the parallel widths the kernel resolved, the pool's nodes and the rack of each, and
// the node of the pool's first replica (PlacementOf(0), counted from the pool's first node). A
// Placement is what the engine states its transfers against, and the plan they are priced against.
func (m *Model) EnginePlacement() placement.Placement { return m.EnginePlacementOf(0) }

// EnginePlacementOf is the placement of the pool's rank-th replica: EnginePlacement with the node its ranks fill
// (PlacementOf), counted from the pool's first node. A host that builds one engine for each instance states it
// with the engine's SetPlacement (placement.Placeable).
func (m *Model) EnginePlacementOf(rank int) placement.Placement {
	return m.enginePlacement(m.PlacementOf(rank))
}

// EngineGroupPlacementOf is the placement of the pool's g-th group: EnginePlacement with the node the
// group's first rank fills (GroupPlacementOf), counted from the pool's first node. The group's own
// engine states where each of its ranks sits from it (core/vllm's DPGroup.SetPlacement).
func (m *Model) EngineGroupPlacementOf(g int) placement.Placement {
	return m.enginePlacement(m.GroupPlacementOf(g))
}

func (m *Model) enginePlacement(at kernel.Placement) placement.Placement {
	r := m.k.Resolved()
	p := placement.Placement{Hardware: m.id.hardware, Fabric: m.id.fabric, Tensor: m.id.tpRanks,
		Pipeline: m.PipelineStages(), Data: r.DataParallel(), Expert: r.ExpertParallel(),
		RanksPerNode: m.id.gpusPerNode, Nodes: m.id.poolNodes}
	for n := 0; n < m.id.poolNodes; n++ {
		rack := m.id.firstNode + n
		if m.id.gpusPerRack > 0 && m.id.gpusPerNode > 0 {
			rack = (m.id.firstNode + n) * m.id.gpusPerNode / m.id.gpusPerRack
		}
		p.NodeRacks = append(p.NodeRacks, rack)
	}
	p.Node = at.Node - m.id.firstNode
	return p
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

// RecordSelected registers the model's kernel with d, as the kernel's own RecordSelected does:
// d lists the kernel's selected coefficients, and (with a substrate that observes kernels) reads
// at report time which a priced step used outside its fit range. A host whose kernel is
// opened inside an engine calls it once per kernel; a nil d, or a kernel without the method,
// does nothing.
func (m *Model) RecordSelected(d *load.Data) {
	if d == nil {
		return
	}
	if r, ok := m.k.(interface{ RecordSelected(*load.Data) }); ok {
		r.RecordSelected(d)
	}
}

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

// HasRecurrentLayers reports whether the pool's model graph has a recurrent (Mamba-style) layer,
// which is when vLLM runs MambaModelConfig.verify_and_update_config for it
// (vllm/model_executor/models/config.py:507-521, 672-708@5df63a72; ModelConfig.is_hybrid,
// vllm/config/model.py:2023-2032@5df63a72). A Mamba-only architecture also has recurrent layers.
func (m *Model) HasRecurrentLayers() bool {
	has := false
	if m.graph != nil {
		m.graph.EachNode(func(_ string, n model.Node) {
			if n.Op == model.OpRecurrentUpdate {
				has = true
			}
		})
	}
	return has
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

// TierTransferTicks prices moving blocks of this pool's KV to (toTier) or from an offload tier
// device class, one transfer at a time, at the pool's per-block bytes: the kernel's TierTime,
// rounded up to whole ticks. An unknown tier prices at the kernel's sentinel, which a caller
// refuses at startup (compat/build priceOffload checks the tier first).
func (m *Model) TierTransferTicks(tier string, toTier bool, blocks int64) int64 {
	dir := kernel.DirectionFromTier
	if toTier {
		dir = kernel.DirectionToTier
	}
	bytes := blocks * m.k.SequenceVariableBytes(m.engine.BlockSize)
	return max(1, ticks(m.k.TierTime(tier, dir, bytes, 1)))
}

// TierTransferTicksOf is TierTransferTicks for a transfer of bytes, the caller's own per-block
// figure times the blocks (a hybrid pool's page bytes), or of blocks at the kernel's per-block
// bytes when bytes is 0.
func (m *Model) TierTransferTicksOf(tier string, toTier bool, bytes, blocks int64) int64 {
	if bytes <= 0 {
		return m.TierTransferTicks(tier, toTier, blocks)
	}
	dir := kernel.DirectionFromTier
	if toTier {
		dir = kernel.DirectionToTier
	}
	return max(1, ticks(m.k.TierTime(tier, dir, bytes, 1)))
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

// verifiedDrafter is the latency kernel's adaptive-verification answer (blis-substrate kernel/adaptive.go):
// for each request of a batch, how many drafts the verifier runs under the adaptive budget, or
// nil when the pool does not verify adaptively.
type verifiedDrafter interface {
	VerifiedDrafts(b kernel.Batch) []int
}

// VerifiedDraftsOfShapes is, for each request of a batch already in the kernel's shapes, how many
// of its scheduled drafts the verifier runs, or nil when every scheduled draft is verified: the
// pool does not turn adaptive verification on, or its kernel does not trim. Under adaptive
// verification vLLM's scheduler still budgets 1 + K tokens a decode, and the runner trims the
// verified drafts on the GPU to a confidence-chosen budget (get_num_tokens and reallocate_drafts,
// vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:282-350,392-448@5df63a72); the batch is
// the one StepTimeOfShapes prices, so the widths are the ones its step time was priced at.
func (m *Model) VerifiedDraftsOfShapes(reqs []ReqShape) []int {
	v, ok := m.k.(verifiedDrafter)
	if !ok {
		return nil
	}
	return v.VerifiedDrafts(kernel.Batch{Reqs: reqs, DecodeThreshold: m.decodeThreshold, SMBudget: m.smBudget})
}

// StageTimesOfShapes splits StepTimeOfShapes by pipeline stage, in pipeline order, where the
// pool runs more than one (the kernel's StepEstimate.Stages): each stage's NoOverlap figure in
// ticks, at least one, the last carrying the rounding so the stages sum exactly to
// StepTimeOfShapes. A single-stage pool returns that one figure.
func (m *Model) StageTimesOfShapes(reqs []ReqShape) []int64 {
	b := kernel.Batch{Reqs: reqs, DecodeThreshold: m.decodeThreshold, SMBudget: m.smBudget}
	e := m.k.StepTime(b)
	total := max(1, ticks(e.NoOverlap))
	if len(e.Stages) == 0 {
		return []int64{total}
	}
	out := make([]int64, len(e.Stages))
	var sum int64
	for i, st := range e.Stages {
		out[i] = max(1, ticks(st.NoOverlap))
		sum += out[i]
	}
	// Rounding each stage can drift from the rounded whole by a tick a stage; the last stage
	// absorbs it, and a total too short for one tick a stage widens to that.
	if d := total - sum; d != 0 {
		out[len(out)-1] = max(1, out[len(out)-1]+d)
	}
	return out
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

// DPLoadBalancing is the pool's parallel.data_parallel_load_balancing as stated ("" is
// internal; blis-schemas#76).
func (m *Model) DPLoadBalancing() deployment.DPLoadBalancing {
	return m.k.Deployment().Parallel.DataParallelLoadBalancing
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
	// EngineVersion is the engine release the scenario names (EngineRelease): engine.version,
	// or the deprecated engine_version.
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
	add(e.EnableChunkedPrefill != nil, "enable_chunked_prefill")
	add(e.MaxNumScheduledTokens != 0, "max_num_scheduled_tokens")
	add(e.MaxNumActiveSeqs != 0, "max_num_active_seqs")
	add(e.Watermark != nil, "watermark")
	add(e.PrefillScheduleInterval != 0, "prefill_schedule_interval")
	add(e.StreamInterval != 0, "stream_interval")
	add(e.DisableHybridKVCacheManager != nil, "disable_hybrid_kv_cache_manager")
	add(e.LanguageModelOnly != nil, "language_model_only")
	add(e.KVCacheMemoryBytes != nil, "kv_cache_memory_bytes")
	add(e.PrefixMatchUnit != 0, "prefix_match_unit")
	// The compatibility engine models vLLM's default interval 0, which keeps only the semantic
	// checkpoints (compat/kv/prefix_hit.go), so a stated 0 is what it runs.
	add(e.PrefixCacheRetentionInterval != nil && *e.PrefixCacheRetentionInterval != 0, "prefix_cache_retention_interval")
	add(e.KVEvents != nil, "kv_events_config")
	add(e.MaxLoRAs != 0, "max_loras")
	add(e.MaxCPULoRAs != 0, "max_cpu_loras")
	add(e.Speculative != nil && e.Speculative.DraftModel != "", "speculative_config.model")
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
	sc, dep, err := blisschemas.LoadBundle(path)
	if err != nil {
		return Shape{}, err
	}
	release, err := EngineRelease(sc)
	if err != nil {
		return Shape{}, fmt.Errorf("%s: %w", scenario, err)
	}
	sh := Shape{Roles: make([]deployment.Role, len(dep.Pools)), StatesOffload: dep.Offload != nil,
		Model: sc.Model, Acceptance: sc.Acceptance, WorkloadBinding: sc.Workload, EngineVersion: release,
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
		// The engine block's synthetic_acceptance_length is the run's supplied acceptance, at the
		// rank of the scenario's pin (AcceptanceWithEngine); every pool must state the same one.
		if sh.Acceptance, err = AcceptanceWithEngine(sh.Acceptance, e); err != nil {
			return Shape{}, fmt.Errorf("%s: pools[%d]: %w", scenario, i, err)
		}
	}
	return sh, nil
}

// checkSpeculativeMethod refuses a pool whose engine.speculative_config.method the scenario's engine
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
		return fmt.Errorf("engine.speculative_config.method %q cannot be checked: no engine rules for "+
			"version %q", sp.Method, in.Scenario.EngineRelease())
	}
	if !in.Rules.SpeculativeMethodAccepted(sp.Method) {
		return fmt.Errorf("engine.speculative_config.method %q is not accepted by engine %s; it accepts %s",
			sp.Method, in.Rules.EngineVersion(), strings.Join(in.Rules.SpeculativeMethodNames(), ", "))
	}
	return nil
}

// museSparkArchitecture is the architecture of the Muse-Spark checkpoints (the catalog graphs'
// global.architecture).
const museSparkArchitecture = "MuseSparkForConditionalGeneration"

// checkMuseSparkParallel refuses what vLLM refuses for Muse-Spark on a Hopper part: data or
// prefill-context parallelism above 1. Its MoE layers take the route-major Marlin and Triton path
// on every capability-90 device (MuseSparkModelOptNvFp4FusedMoE.__init__ for NVFP4 experts, and
// MuseSparkUnquantizedFusedMoE.__init__ for the layers the checkpoint excludes), and that path
// calls _validate_route_major_deployment, which raises NotImplementedError "MuseSpark1x
// Triton/Marlin currently requires DP1 and PCP1" (vllm/models/muse_spark/nvidia/muse_moe_runner.py
// :140-146,569-581,745-751@0f6b2619). The model is not in the pinned v0.32.0 release; the
// compatibility engine takes it from the commit the catalog's Muse graphs were derived from, as
// kvlayers.go does. Blackwell takes the TRT-LLM path, which has no such check.
func checkMuseSparkParallel(arch string, chip *hardware.Chip, pl deployment.Parallelism) error {
	if arch != museSparkArchitecture || chip == nil {
		return nil
	}
	if fam, ok := chip.CapabilityMajor(); !ok || fam != 9 {
		return nil
	}
	if pl.DP > 1 || pl.PCP > 1 {
		return fmt.Errorf("Muse-Spark on %s (compute capability %s) requires data parallelism 1 and prefill-context "+
			"parallelism 1 (got dp %d, pcp %d): its MoE layers take the route-major Marlin path on Hopper, which "+
			"vLLM refuses to build with more (\"MuseSpark1x Triton/Marlin currently requires DP1 and PCP1\", "+
			"vllm/models/muse_spark/nvidia/muse_moe_runner.py:140-146@0f6b2619). Blackwell is not affected",
			chip.Name, chip.ComputeCapability, pl.DP, pl.PCP)
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

// envMNNVL is the UCX variable that lets NIXL's cuda_ipc transport cross nodes over multi-node NVLink.
const envMNNVL = "UCX_CUDA_IPC_ENABLE_MNNVL"

// rackHandoffAssumptions discloses how a handoff between two nodes of one NVLink rack is priced.
// It speaks only for a part whose rack is one NVLink domain (gpus_per_rack above one node).
func rackHandoffAssumptions(id identity) []string {
	if id.gpusPerRack <= 0 || id.gpusPerNode <= 0 || id.gpusPerRack <= id.gpusPerNode {
		return nil
	}
	if id.rackHandoff {
		return []string{"kv_handoff_path: " + envMNNVL + "=y is stated, so a handoff between nodes of one rack is " +
			"priced over NVLink; that KV is registered as VMM, which NIXL needs for it " +
			"(docs/features/nixl_connector_usage.md:401@5df63a72), is not checked"}
	}
	return []string{"kv_handoff_path: " + envMNNVL + "=y is not stated, so a handoff between nodes is priced over " +
		"the NIC at the cluster fabric's rate even inside one NVLink rack; NIXL uses multi-node NVLink only with " +
		"that flag and VMM-registered KV (docs/features/nixl_connector_usage.md:401@5df63a72)"}
}
