package cmd

import (
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/env"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/acceptance"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/inference-sim/inference-sim/substrate/contracts/units"
	"github.com/inference-sim/inference-sim/substrate/load"

	blisvllm "github.com/inference-sim/inference-sim/engines/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/core/build"
	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel"
	cengine "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"
	coreengine "github.com/inference-sim/inference-sim/engines/vllm/vllmengine"

	"math"
	"time"

	"github.com/inference-sim/inference-sim/internal/stack"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	specmodel "github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	latencykernel "github.com/inference-sim/inference-sim/substrate/kernel"
)

// The one place inference-sim builds the vLLM engine (#2021 B2). The engine's configuration -- the
// scenario's engine-side blocks -- is decoded and resolved by the engine's build package (engines/vllm/core/build), which cmd
// calls only from here, handing the inputs over unread. A refusal comes back as an error; the CLI boundary turns it into the fatal exit.

// resolveKVOffloadConfig resolves the run's KV offload from the scenario's offload block
// (#1910), the only source (#2037): build.ResolveKVOffloadConfig is given no file.
// Single construction site (R4), shared by run and replay for INV-13 parity.
func resolveKVOffloadConfig() kvOffloadConfig { return resolveOffloadBlock(scenarioOffload) }

// resolveDecodeKVOffloadConfig is the decode pool's offload under P/D: its own block, else the
// deployment's, and none for a pool that states none. The run's one config (resolveKVOffloadConfig)
// is the prefill pool's (the colocated pool's for a colocated run).
func resolveDecodeKVOffloadConfig() kvOffloadConfig {
	return resolveOffloadBlock(scenarioDecodeOffload)
}

func resolveOffloadBlock(o *deployment.Offload) kvOffloadConfig {
	cfg, err := build.ResolveKVOffloadConfig(kernelScenario, o, "", blockSizeTokens, mustRunData())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return cfg
}

// runOffloads are the offload blocks the run's pools state, each with the deployment's applied:
// the first-served pool's (the prefill pool's under P/D, else the colocated pool's) and the decode
// pool's, nil for a pool with none. A shape that lists no pools (a deployment-less run) has the
// deployment's block for both.
func runOffloads(sh scenarioShapeT) (first, decode *deployment.Offload) {
	if len(sh.PoolOffloads) == 0 || len(sh.PoolOffloads) != len(sh.Roles) {
		return sh.Offload, sh.Offload
	}
	first, decode = sh.PoolOffloads[0], sh.PoolOffloads[0]
	for i, r := range sh.Roles {
		switch r {
		case deployment.RolePrefill:
			first = sh.PoolOffloads[i]
		case deployment.RoleDecode:
			decode = sh.PoolOffloads[i]
		}
	}
	return first, decode
}

// loadCatalogStorageDevices is the run's storage-device table, read by the engine from the
// catalog in use (build.StorageDevices).
// openEngineData opens the catalog and registry the options name: a fresh load.Data with an
// empty selected-entry report (kernelmodel.OpenData keeps no cache). runData calls it once for
// a run and hands that Data to the simulator and to every engine (newContractEngine). It is the
// build site's, as every use of the vLLM engine is.
var openEngineData = func(o load.Options) (*load.Data, error) { return kernelmodel.OpenData(o) }

func loadCatalogStorageDevices(data *load.Data) (map[string]build.KVOffloadDevice, error) {
	return build.StorageDevices(data)
}

// catalogStorageDevicesPath, catalogDevicesSubdir and catalogStorageDevicesFile locate the
// storage-device table inside a catalog directory, for the catalog-load gate's diagnostics.
func catalogStorageDevicesPath(catalog string) string {
	return filepath.Join(catalog, build.CatalogDevicesSubdir, build.CatalogStorageDevicesFile)
}

const (
	catalogDevicesSubdir      = build.CatalogDevicesSubdir
	catalogStorageDevicesFile = build.CatalogStorageDevicesFile
)

// catalogStorageDevicesRelPath is the table's location relative to the catalog clone root,
// for diagnostics that name it.
const catalogStorageDevicesRelPath = build.CatalogStorageDevicesRelPath

// engineKnobFlags are the command line's restatements of the scenario's per-pool engine knobs:
// none (#2021 B6, #2037). Every pool runs the long_prefill_token_threshold and
// data_parallel_load_balancing it states; the zero value tells the engine no flag was given,
// so a pool that states no mode runs vLLM's default.
func engineKnobFlags() build.EngineKnobFlags {
	return build.EngineKnobFlags{DPLoadBalancing: defaultDPLoadBalancing}
}

// runEngineKnobs is the run-wide long-prefill threshold and DP load-balancing mode
// (build.RunEngineKnobs).
func runEngineKnobs(shape kernelmodel.Shape, f build.EngineKnobFlags) (int64, string, error) {
	return build.RunEngineKnobs(kernelScenario, shape, f)
}

// enginePolicyFor is a pool's scheduling_policy in the engine's terms (build.EnginePolicyOf).
func enginePolicyFor(m *kernelmodel.Model) (build.EnginePolicy, error) {
	return build.EnginePolicyOf(kernelScenario, m)
}

// routingOverridesOf is the command line's routing layer (build.RoutingOverridesOf): the
// --routing-token-imbalance and --routing-expert-imbalance values, each with whether it was
// given.
func routingOverridesOf(token float64, tokenSet bool, expert float64, expertSet bool) (kernelmodel.RoutingOverrides, error) {
	return build.RoutingOverridesOf(build.RoutingFlags{
		TokenImbalance: token, TokenSet: tokenSet, ExpertImbalance: expert, ExpertSet: expertSet,
	})
}

// The acceptance flags' names, as the engine's refusals spell them.
const (
	acceptanceRateFlagName     = build.AcceptanceRateFlag
	acceptanceLengthFlagName   = build.AcceptanceLengthFlag
	acceptanceThinkingFlagName = build.AcceptanceThinkingFlag
	acceptanceCategoryFlagName = build.AcceptanceCategoryFlag
	acceptanceFallbackFlagName = build.AcceptanceFallbackFlag
)

// resolveAcceptance resolves the run's acceptance for a pool drafting k tokens with method
// (build.ResolveAcceptance), from the scenario, the registry, the run's hardware, what the
// workload states and the command line. CLI boundary: a refusal is fatal; a selector flag
// given to a run that does not speculate is warned about and ignored.
func resolveAcceptance(cmd *cobra.Command, k int, method string) (build.Acceptance, bool) {
	return resolveAcceptanceFor(cmd, kernelOpened, k, method)
}

// resolveAcceptanceFor is resolveAcceptance for the pool m (nil when no pool is open): the lookup's
// key takes m's own speculative block, so each P/D pool resolves its own acceptance
// (inference-sim#2125).
func resolveAcceptanceFor(cmd *cobra.Command, m *engineModel, k int, method string) (build.Acceptance, bool) {
	fl := cmd.Flags()
	if err := checkRealisationFlag(fl); err != nil {
		logrus.Fatalf("%v", err)
	}
	if sglangSpeculation.Algorithm != "" {
		for _, f := range []string{build.AcceptanceRateFlag, build.AcceptanceLengthFlag} {
			if fl.Changed(f) {
				logrus.Fatalf("%v", sglangSpeculationRefusal(f))
			}
		}
	}
	a, speculating, err := build.ResolveAcceptance(poolAcceptanceInputs(m, build.AcceptanceInputs{
		Scenario: kernelScenario, Shape: scenarioShape, Data: mustRunData(), Hardware: gpu,
		Workload: workloadAcceptance,
		Flags: build.AcceptanceFlags{
			Rate: speculativeAcceptance, Length: speculativeAcceptanceLength,
			RateSet: fl.Changed(build.AcceptanceRateFlag), LengthSet: fl.Changed(build.AcceptanceLengthFlag),
			Thinking: speculativeThinking, Category: speculativeCategory, FallbackCategory: speculativeFallbackCategory,
			ThinkingSet: fl.Changed(build.AcceptanceThinkingFlag), CategorySet: fl.Changed(build.AcceptanceCategoryFlag),
			FallbackSet: fl.Changed(build.AcceptanceFallbackFlag),
			Realisation: acceptance.Realisation(speculativeRealisation),
		},
	}), k, method)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	for _, f := range a.Ignored {
		logrus.Warnf("--%s is ignored: scenario %q does not speculate", f, kernelScenario)
	}
	// The command line's and the workload's acceptance layers reach the engine's kernel too: a
	// pool verifying adaptively sizes its draft budget from the run's acceptance
	// (kernelmodel.SetAcceptanceOverrides, https://github.com/inference-sim/blis-latency-kernel/issues/89), which must be the one the run
	// resolved here. The engines are built after this (engineDriverFactory).
	setAcceptanceOverrides(fl, k)
	return a, speculating
}

// acceptanceEngineVLLM is the engine name `blis acceptance --engine` takes.
const acceptanceEngineVLLM = build.Engine

// vllmAcceptanceRequest is req with the vLLM engine's part of the lookup's key stated as a run
// states it: the engine, its defaults, and the draft sampler and verifier the speculative block's
// sample methods (or the defaults) resolve to.
func vllmAcceptanceRequest(req acceptance.Request, draftSample, rejectionSample string) acceptance.Request {
	in := build.AcceptanceInputs{}.VLLMInputsSampling(draftSample, rejectionSample)
	req.Engine, req.DefaultDraftSampler, req.DefaultVerifier = in.Engine, in.DefaultDraftSampler, in.DefaultVerifier
	req.DraftSampler, req.Verifier = in.DraftSampler, in.Verifier
	return req
}

// poolAcceptanceInputs is in with pool m's part of the lookup's key, from the speculative
// block of the engine it opened (none when m is nil): the drafter it names
// (speculative_config.model, the catalog drafter; inference-sim/inference-sim#2142), its draft
// sampler and verifier as vLLM resolves them, defaults included, the engine "vllm" and vLLM's own
// defaults (build.AcceptanceInputs.VLLMInputs). A pool that names no drafter drafts with the
// target's own speculator.
func poolAcceptanceInputs(m *engineModel, in build.AcceptanceInputs) build.AcceptanceInputs {
	if m == nil {
		return in.VLLMInputs(nil)
	}
	e, err := m.Engine()
	if err != nil {
		logrus.Fatalf("reading the %s pool's speculative settings: %v", m.Role(), err)
	}
	in = in.VLLMInputs(e.Speculative)
	if e.Speculative != nil {
		in.Drafter = e.Speculative.DraftModel
	}
	return in
}

// The P/D pools (build.OpenPools and its pricing), as the run uses them.

// kernelPools is a disaggregated kernel run's per-role wiring: one kernel per pool.
type kernelPools struct {
	prefill, decode *kernelmodel.Model
	// connector is the prefill pool's KV connector and decodeConnector the decode pool's (each pool's
	// own pd_transfer.connector, else the deployment's).
	connector, decodeConnector string
	// p2p is whether the handoff is vLLM's p2p tier's (build.Pools.P2P).
	p2p bool
}

// pools is p as the engine's pools (nil for a colocated run).
func (p *kernelPools) pools() *build.Pools {
	if p == nil {
		return nil
	}
	return &build.Pools{Prefill: p.prefill, Decode: p.decode, Connector: p.connector, DecodeConnector: p.decodeConnector, P2P: p.p2p}
}

// runTopology is the run's instance counts as they stand.
func runTopology() build.Topology {
	return build.Topology{NumInstances: numInstances, Prefill: prefillInstances, Decode: decodeInstances,
		SharedOrEncode: prefillDecodeInstances > 0 || encodeInstances > 0}
}

// fatalf is the CLI's fatal exit, handed to the engine's pricing functions for a failure
// found while the run is under way.
func fatalf(format string, args ...any) { logrus.Fatalf(format, args...) }

// scenarioEndpointTopology uses the same role-specific kernel placements that price KV
// handoffs (build.EndpointGroupPlacements), as the cluster's endpoint topology.
func scenarioEndpointTopology(model *kernelmodel.Model, pools *kernelPools, total, prefill, decode int) (map[cluster.InstanceID]cluster.EndpointTopology, error) {
	// A group is one instance (EndpointGroupPlacements), except under external DP load balancing,
	// where every rank is its own router-visible instance, placed per rank (EndpointPlacements).
	place := build.EndpointGroupPlacements
	if externalDPActive() {
		place = build.EndpointPlacements
	}
	placed, err := place(model, pools.pools(), total, prefill, decode)
	if err != nil || placed == nil {
		return nil, err
	}
	out := make(map[cluster.InstanceID]cluster.EndpointTopology, len(placed))
	for id, p := range placed {
		out[cluster.InstanceID(id)] = cluster.EndpointTopology{Hostname: p.Hostname, Rack: p.Rack, NodeCount: p.NodeCount}
	}
	return out, nil
}

// openKernelPools opens the prefill and decode pools of the run's scenario (build.OpenPools):
// nil when no kernel is open or the run is not disaggregated, fatal when the pools cannot run.
// Its startup warnings are logged first, a refused run's included.
func openKernelPools() *kernelPools {
	repos := kernelmodel.Repos{Scenarios: kernelScenarioDir, Data: mustRunData()}
	bp, warnings, err := build.OpenPools(kernelScenario, repos, kernelOpened != nil, runTopology())
	for _, w := range warnings {
		logrus.Warnf("%s", w)
	}
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	if bp == nil {
		// A colocated run: no per-role kernels, whatever an earlier run in this process opened.
		kernelPDOpened = nil
		return nil
	}
	p := &kernelPools{prefill: bp.Prefill, decode: bp.Decode, connector: bp.Connector, decodeConnector: bp.DecodeConnector, p2p: bp.P2P}
	kernelPDOpened = p
	return p
}

// poolOfInstance is the engine.Spec pool index each instance's engine is built from: the
// colocated run's one pool, or the prefill or decode pool whose kernel the instance's role
// configuration carries (rolesOf), by its index in the deployment (build.PoolIndices).
func (p *kernelPools) poolOfInstance() func(sim.InstanceConfig) int {
	if p == nil {
		return func(sim.InstanceConfig) int { return 0 }
	}
	indices, err := build.PoolIndices(kernelScenario, kernelmodel.Repos{Scenarios: kernelScenarioDir,
		Data: mustRunData()})
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return func(c sim.InstanceConfig) int {
		switch engineConfigIn(c).LatencyModel {
		case latencyModel(p.prefill):
			return indices[deployment.RolePrefill]
		case latencyModel(p.decode):
			return indices[deployment.RoleDecode]
		}
		panic("engine build: a disaggregated run's instance takes neither the prefill nor the decode pool")
	}
}

// holdsPrefillKV reports whether the run's prefill engines hold a finished request's KV for
// the decode pull. False for a colocated run (nil receiver).
func (p *kernelPools) holdsPrefillKV() bool {
	return p.pools().HoldsPrefillKV()
}

// instanceEndpoint is where each instance's workers sit, in the numbering of its own pool: the
// node and the rack of its rank's placement, each counted from the pool's lowest (poolOrigin). It is what a decode
// engine is told about the prefill it pulls from (engine.Request.Peer). Instances are numbered
// prefill first, then decode (build.RankInPool). Nil for a colocated run (nil receiver).
func (p *kernelPools) instanceEndpoint() func(cluster.InstanceID) plan.Endpoint {
	if p == nil {
		return nil
	}
	firstDecode := runTopology().Prefill
	return func(id cluster.InstanceID) plan.Endpoint {
		n, err := strconv.Atoi(strings.TrimPrefix(string(id), "instance_"))
		if err != nil || n < 0 {
			panic(fmt.Sprintf("instance endpoint: unexpected instance id %q", id))
		}
		m := p.prefill
		if n >= firstDecode {
			m = p.decode
		}
		at := m.GroupPlacementOf(build.RankInPool(string(id), firstDecode))
		local, err := poolOrigin(m).Local(plan.Endpoint{Node: at.Node, Rack: at.Rack})
		if err != nil {
			panic(fmt.Sprintf("instance endpoint: instance %q: %v", id, err))
		}
		return local
	}
}

// requirePullPricing refuses, before the run starts, a run whose handoffs the cost model could not
// price: contention in a colocated run, and any prefill-decode pair whose link the fabric states no
// bandwidth for. Every pair is asked for one block, as the run's first handoff between them would
// be, so a pair the fabric cannot price is refused here and not by a stopped run.
func requirePullPricing(p *kernelPools, contention bool) {
	if p == nil {
		if contention {
			logrus.Fatalf("--pd-transfer-contention needs a disaggregated run: " +
				"pass --prefill-instances and --decode-instances with a P/D scenario")
		}
		return
	}
	e, err := newTransferEnvironment(p.decode, p.prefill)
	if err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	}
	top := runTopology()
	endpoint := p.instanceEndpoint()
	pl := plan.PoolPlan{Hardware: "pool", Layout: plan.Layout{Tensor: 1, RanksPerNode: 1, Nodes: 1 << 20}}
	for i := 0; i < top.Prefill; i++ {
		for j := 0; j < top.Decode; j++ {
			from, to := cluster.InstanceID(fmt.Sprintf("instance_%d", i)), cluster.InstanceID(fmt.Sprintf("instance_%d", top.Prefill+j))
			if _, err := e.TransferDemand(pl, plan.Transfer{Tokens: units.Tokens(e.blockSize),
				From: endpoint(from), To: endpoint(to), Kind: plan.TransferPeer}); err != nil {
				logrus.Fatalf("scenario %q: a KV handoff from prefill rank %d to decode rank %d: %v", kernelScenario, i, j, err)
			}
		}
	}
}

// setEngineFrontend gives the engine --renderer-num-workers and --api-server-count, vLLM's
// API-server settings (blis-schemas does not carry them yet). The engine resolves them into the
// admission queue its plan states. An engine that takes no such settings is refused. It lives
// here because this file is the one place the binary reaches the engine's package.
func setEngineFrontend(eng engine.Engine, version string) error {
	fe, ok := eng.(coreengine.Frontender)
	if !ok {
		// SGLang has no such settings; the flags at their defaults state nothing to refuse.
		if scenarioEngine.Kind == engineKindSGLang && rendererNumWorkers == 1 && apiServerCount == 0 {
			return nil
		}
		return fmt.Errorf("engine %s does not take API-server settings", version)
	}
	// An unstated count is vLLM's default for the balancing mode (serve.py:113-129): the ranks
	// behind one endpoint, which is every rank of an internal balancer, a node's under hybrid, and
	// one under external. The engine takes an unstated count as one client, so the host states it
	// (the vLLM engine api_server_count: "a caller states the count it ran").
	count := apiServerCount
	if g, ok := eng.(interface{ EndpointRanks(int) (int, int) }); ok && count == 0 {
		if _, n := g.EndpointRanks(0); n > 1 {
			count = n
		}
	}
	if err := fe.SetFrontend(coreengine.Frontend{RendererNumWorkers: rendererNumWorkers, APIServerCount: count}); err != nil {
		return fmt.Errorf("--renderer-num-workers / --api-server-count: %w", err)
	}
	return nil
}

// placeInstance states where instance index sits, through the engine contract (engine.Place): the
// engine's factory builds every instance of a pool from the same engine block, so each would state
// the pool's first node for itself, and a pull's near end would be priced there whichever instance
// pulls. The node and rack are the instance's endpoint (instanceEndpoint, the one source the
// cluster also states to a decode engine as its peer). A colocated run is left as built. A
// disaggregated run whose engine must be placed and cannot be (engine.ErrPlacementUnsupported), or
// refuses the place (engine.ErrPlacementRejected), is refused by that error and not run on the
// wrong link.
func placeInstance(eng engine.Engine, p *kernelPools, index int) error {
	if p == nil {
		return nil
	}
	m := p.prefill
	if index >= runTopology().Prefill {
		m = p.decode
	}
	at := p.instanceEndpoint()(cluster.InstanceID(fmt.Sprintf("instance_%d", index)))
	if err := engine.Place(eng, at, m.EnginePlacement().Nodes); err != nil {
		return fmt.Errorf("instance_%d: %w", index, err)
	}
	return nil
}

// environmentOf is the Environment of an engine built from the instance configuration c: the
// kernels of its own pool and, in a disaggregated run, of the pool its pulls read from.
func (p *kernelPools) environmentOf(c sim.InstanceConfig, colocated *kernelmodel.Model) env.Environment {
	own, peer := colocated, (*kernelmodel.Model)(nil)
	if p != nil {
		switch engineConfigIn(c).LatencyModel {
		case latencyModel(p.prefill):
			own, peer = p.prefill, p.prefill
		case latencyModel(p.decode):
			own, peer = p.decode, p.prefill
		}
	}
	if own == nil {
		return nil
	}
	e, err := newTransferEnvironment(own, peer)
	if err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	}
	if p != nil && own == p.prefill {
		// A prefill engine pulls from no one (its peer is its own pool), and its WRITEs land in the decode pool.
		e.across = p.decode
	}
	return wrapEnvironment(e)
}

// wrapEnvironment is the identity; a test sets it to watch every question an engine's host is
// asked, so that it can show a run and a replay turn each move into a demand through the
// Environment and through nothing else.
var wrapEnvironment = func(e env.Environment) env.Environment { return e }

// defaultDPLoadBalancing is vLLM's default DP load-balancing mode, which a pool that states no
// parallel.data_parallel_load_balancing runs (sim.DPLoadBalancingInternal, defined in
// sim/dp_modes.go; the engine's balancer is engines/vllm/core/engine/dp.go).
const defaultDPLoadBalancing = sim.DPLoadBalancingInternal

// adoptFirstPoolEngine takes the run's engine from the first pool's kernel
// (build.FirstPoolSettingsOf): the caps, pages and pool the kernel priced, so the simulated
// scheduler admits against them, per data-parallel rank: each rank of the engine's
// data-parallel group is sized as one EngineCore.
func adoptFirstPoolEngine(m *engineModel, settings kernelmodel.Settings) {
	fp, err := build.FirstPoolSettingsOf(kernelScenario, settings)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	totalKVBlocks = fp.TotalKVBlocks
	sequenceState = fp.SequenceState
	prefixHit = prefixHitOf("first pool", m)
	blockSizeTokens = fp.BlockSize
	maxNumSeqs = fp.MaxNumSeqs
	maxNumBatchedTokens = fp.MaxNumBatchedTokens
	maxModelLen = fp.MaxModelLen
	noEnablePrefixCaching = fp.PrefixCachingDisabled
	numSpeculativeTokens = fp.SpeculativeTokens
	speculativeMethod = fp.SpeculativeMethod
	logrus.Infof("deployment from scenario %s -- model %s, hardware %s, "+
		"tp %d, dp %d, expert-parallel %t; engine from the kernel -- %d KV blocks/rank of %d "+
		"tokens, max_num_seqs %d, max_num_batched_tokens %d, max_model_len %d (%s), prefix caching %t, "+
		"speculative tokens %d",
		kernelScenario, model, gpu,
		tensorParallelism, dataParallelism, enableExpertParallel,
		totalKVBlocks, blockSizeTokens, maxNumSeqs, maxNumBatchedTokens, maxModelLen, fp.MaxModelLenFrom,
		!noEnablePrefixCaching, numSpeculativeTokens)
	for _, note := range fp.Notes {
		logrus.Infof("%s", note)
	}
}

// applyKernelLoRAReservation re-sizes a kernel run's KV pool with the static LoRA adapter
// reservation set aside, once the LoRA config is known (it is resolved after the deployment
// is adopted). vLLM takes the reservation beside the weights before profiling the KV pool, so
// the kernel's budget shrinks by it (build.KVBlocksReserving). No-op when no kernel is open or
// with no reservation.
func applyKernelLoRAReservation() {
	if kernelOpened == nil || loraReservedBytesForKV <= 0 {
		return
	}
	blocks, err := build.KVBlocksReserving(kernelScenario, kernelOpened, loraReservedBytesForKV)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	totalKVBlocks = blocks
}

// prefixHitOf is how the pool's KV cache groups cut a prefix hit (build.PrefixHitOf), from the
// pool's engine.swa_bounded_replay or its resolution; no flag overrides it (#2037). It records
// the resolution for the results file and logs when the model's layout is not one BLIS maps.
func prefixHitOf(what string, m *engineModel) kvPrefixHit {
	r, warning := build.PrefixHitOf(what, m, nil)
	if warning != "" {
		logrus.Warnf("%s", warning)
	}
	if prefixCacheRecords == nil {
		prefixCacheRecords = map[string]sim.PrefixCacheRecord{}
	}
	prefixCacheRecords[string(m.Role())] = sim.PrefixCacheRecord{
		ModelRunner: r.Runner, ModelRunnerForcedBy: r.RunnerForcedBy,
		SWABoundedReplay: r.BoundedReplay, SWABoundedReplayFrom: r.BoundedReplayFrom,
		Layout: sim.KVPrefixHit(r.Hit), Note: r.Note,
	}
	return r.Hit
}

// overrides is each P/D role's engine, from its own pool's kernel and scenario pool.
func (p *kernelPools) overrides() (prefill, decode poolOverrides) {
	return poolOverridesOf(p.prefill), poolOverridesOf(p.decode)
}

// poolOverridesOf is a P/D pool's engine as the cluster takes it: the settings the vLLM engine
// resolves for the pool (build.PoolSettingsOf), copied into the cluster's per-pool overrides.
// The pool runs its own engine's scheduling policy.
func poolOverridesOf(m *engineModel) poolOverrides {
	hit := prefixHitOf(string(m.Role())+" pool", m)
	ps, err := build.PoolSettingsOf(kernelScenario, m, loraReservedBytesForKV, hit, engineKnobFlags())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	for _, note := range ps.Notes {
		logrus.Infof("%s", note)
	}
	o := poolOverrides{
		TP: &ps.TP, GPU: ps.GPU, LatencyModel: m, TotalKVBlocks: &ps.TotalKVBlocks, MaxModelLen: &ps.MaxModelLen,
		MaxNumSeqs: &ps.MaxNumSeqs, MaxNumBatchedTokens: &ps.MaxNumBatchedTokens,
		PrefixCachingDisabled: &ps.PrefixCachingDisabled, SequenceState: &ps.SequenceState, PrefixHit: &ps.PrefixHit,
		LongPrefillTokenThreshold: &ps.LongPrefillTokenThreshold, DPLoadBalancing: ps.DPLoadBalancing,
		Scheduler: ps.Policy.Scheduler, PreemptionPolicy: ps.Policy.Preemption,
	}
	return o
}

// poolsDraftDifferently reports whether the P/D pools state different speculative blocks
// (build.PoolsDraftDifferently).
func poolsDraftDifferently(p *kernelPools) bool {
	if p == nil {
		return false
	}
	different, err := build.PoolsDraftDifferently(p.prefill, p.decode)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return different
}

// resolvePoolSpeculation resolves each P/D pool's speculation from its own block (resolveSpeculation)
// and records it in poolSpeculation. It returns the run-wide config, the first drafting pool's,
// which names the run in the places that state one speculation. The command line's acceptance
// flags apply to every pool.
func resolvePoolSpeculation(cmd *cobra.Command, p *kernelPools) speculativeConfig {
	poolSpeculation = map[string]poolSpeculative{}
	var run speculativeConfig
	var have bool
	for _, pool := range []struct {
		role string
		m    *engineModel
	}{{"prefill", p.prefill}, {"decode", p.decode}} {
		d, err := build.PoolDrafting(pool.m)
		if err != nil {
			logrus.Fatalf("%v", err)
		}
		var c speculativeConfig
		var rec *sim.AcceptanceRecord
		if !d.Drafts() {
			// A pool that drafts nothing takes no acceptance (the other pool's selectors and
			// length are not refused for it).
			c = speculativeConfigOf(0, 0, d.Method)
		} else {
			c, rec = resolveSpeculation(cmd, pool.m, d.Tokens, d.Method, " in the "+pool.role+" pool")
		}
		poolSpeculation[pool.role] = poolSpeculative{Config: c, Record: rec}
		if !have && c.IsEnabled() {
			run, have = c, true
		}
	}
	if !have {
		return poolSpeculation["prefill"].Config
	}
	return run
}

// adoptPoolSpeculation puts each pool's own speculation on its overrides when the pools draft
// differently; a no-op otherwise, so same-config pools and colocated runs are unchanged.
func adoptPoolSpeculation(prefill, decode *poolOverrides) {
	if poolSpeculation == nil {
		return
	}
	pre, dec := poolSpeculation["prefill"].Config, poolSpeculation["decode"].Config
	prefill.Speculative, decode.Speculative = &pre, &dec
}

// rankInPool is an instance's rank within its pool (build.RankInPool).
func rankInPool(id cluster.InstanceID, firstDecode int) int {
	return build.RankInPool(string(id), firstDecode)
}

// requireKernelCapacity refuses a run whose instances do not fit the scenario's pools. Called
// once the instance counts are final. The host runs a data-parallel group as one instance
// (build.RequireGroupCapacity: a pool holds whole dp x pp x tp groups), except under external DP
// load balancing, where every rank is its own router-visible instance (build.RequireCapacity: a
// pool holds whole ranks).
func requireKernelCapacity(p *kernelPools) {
	check := build.RequireGroupCapacity
	if externalDPActive() {
		check = build.RequireCapacity
	}
	if err := check(kernelScenario, kernelOpened, p.pools(), runTopology()); err != nil {
		logrus.Fatalf("%v", err)
	}
}

// externalDP reports the data-parallel width to expose per rank under external load balancing and
// whether external balancing is active (inference-sim#2699). A colocated run is governed by the
// run-wide mode. A disaggregated (P/D) run resolves each pool's mode on its own: the run-wide
// dpLoadBalancing is the flag's value, which for a P/D run RunEngineKnobs leaves at the default
// (internal) whatever the pools state (engines/vllm/core/build/policy.go), so it cannot be trusted
// here -- reading it is what made a P/D external deployment build collapsed-internal while the
// Result reported external.
//
// External DP is driven for a colocated deployment and for a disaggregated (P/D) one: each rank is
// a router-visible endpoint, the shared group steps through the sim leader/member path, and a
// remote-prefilled decode sub-request completes on its rank through the leader/member split. A P/D
// deployment must run external at the same data-parallel width on both pools; a mixed-mode or
// mixed-width disaggregated deployment is refused (not run silently as one collapsed instance), its
// per-pool handling not driven yet.
func externalDP() (dp int, active bool) {
	if kernelPDOpened == nil {
		if dataParallelism > 1 && dpLoadBalancing == sim.DPLoadBalancingExternal {
			return dataParallelism, true
		}
		return 0, false
	}
	pExt, pDP := poolExternalDP(kernelPDOpened.prefill)
	dExt, dDP := poolExternalDP(kernelPDOpened.decode)
	if !pExt && !dExt {
		return 0, false
	}
	bothExternalSameWidth := pExt && dExt && pDP == dDP && pDP > 1
	if !bothExternalSameWidth {
		logrus.Fatalf("scenario %q: external data-parallel load balancing on a disaggregated deployment needs both pools "+
			"external at the same data-parallel width; got prefill external=%v/dp%d, decode external=%v/dp%d",
			kernelScenario, pExt, pDP, dExt, dDP)
	}
	return pDP, true
}

// poolExternalDP is a P/D pool's data-parallel width and whether it resolves to external balancing,
// from the pool's own kernel model (its stated mode, resolved as vLLM resolves it -- a dp_local=1
// hybrid resolves to external). A pool that is not a data-parallel group (dp <= 1) is not external.
func poolExternalDP(m *engineModel) (external bool, dp int) {
	dp = m.DataParallelWidth()
	if dp <= 1 {
		return false, dp
	}
	runs := m.ResolveDPLoadBalancing(m.Kernel().Deployment().Parallel)
	return runs == deployment.DPLBExternal, dp
}

// externalDPActive reports whether the run drives a data-parallel group under external load
// balancing: each rank is its own router-visible endpoint (inference-sim#2699).
func externalDPActive() bool {
	_, active := externalDP()
	return active
}

// effectiveDPLoadBalancing is the DeploymentConfig.DPLoadBalancing the run sets: external when the
// run drives external balancing (so the cluster's DPEachRankIsAnEndpoint check and the shared-group
// JoinGroup fire), else the run-wide mode. The run-wide dpLoadBalancing is the flag default for a
// P/D run and so is internal even when the pools state external; externalDPActive resolves the mode
// per pool, so this reflects it.
func effectiveDPLoadBalancing() string {
	if externalDPActive() {
		return sim.DPLoadBalancingExternal
	}
	return dpLoadBalancing
}

// externalDPLockstepRanks is the DeploymentConfig.DPLockstepRanks the run sets: the DP width under
// external balancing (so the cluster groups each run of that many instances into one shared-group
// replica), and 0 otherwise (a group is one collapsed instance, as before).
func externalDPLockstepRanks() int {
	dp, active := externalDP()
	if active {
		return dp
	}
	return 0
}

// scaleInstanceCountsForExternalDP multiplies the run's instance counts by the DP width when the
// run drives external DP load balancing: every rank of a group becomes its own instance. A no-op
// otherwise. Called once the group counts are final and before the capacity check reads them.
// External balancing requires every pool's data-parallel width to match (externalDP), so one
// width scales every pool's count alike.
func scaleInstanceCountsForExternalDP() {
	dp, active := externalDP()
	if !active {
		return
	}
	numInstances *= dp
	prefillInstances *= dp
	decodeInstances *= dp
	prefillDecodeInstances *= dp
	encodeInstances *= dp
}

// offloadPricing is what the engine's offload pricing reads from the run.
func offloadPricing() build.OffloadPricing {
	return build.OffloadPricing{BlockSizeTokens: blockSizeTokens, MaxNumBatchedTokens: maxNumBatchedTokens}
}

// sizeOffloadBlocks records the kernel's per-rank bytes of one GPU block on an enabled offload
// config and refuses a CPU tier smaller than one chunk (build.SizeOffloadBlocks). keepStated
// keeps the bytes a replayed trace header carries (INV-13).
func sizeOffloadBlocks(cfg *kvOffloadConfig, keepStated bool) {
	if err := build.SizeOffloadBlocks(kernelOpened, cfg, blockSizeTokens, keepStated); err != nil {
		logrus.Fatalf("%v", err)
	}
}

// applyKernelOffloadPricing has the kernel price every KV-offload transfer of a colocated
// run (build.PriceColocatedOffload). Shared by run and replay. The run has no legacy CPU tier
// (the scenario's offload block states the tiers), so there is no per-block charge to keep.
func applyKernelOffloadPricing(cfg *kvOffloadConfig) {
	if _, err := build.PriceColocatedOffload(kernelOpened, cfg, offloadPricing(), fatalf); err != nil {
		logrus.Fatalf("%v", err)
	}
}

// applyOffload prices each pool's KV offload by that pool's own kernel
// (build.Pools.PriceOffloadPools), and records it on the pool's overrides: a pool with no offload
// gets the inert configuration, not the run's. No-op when neither pool offloads.
func (p *kernelPools) applyOffload(prefillCfg, decodeCfg kvOffloadConfig, prefill, decode *poolOverrides) {
	pre, dec, ok, err := p.pools().PriceOffloadPools(kernelScenario, prefillCfg, decodeCfg, offloadPricing(), runTopology(), fatalf)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	if !ok {
		return
	}
	prefill.KVOffload, decode.KVOffload = &pre.Config, &dec.Config
}

// takePDTopologyFromPools sets the run's instance counts from the scenario's pools when the command
// line states no topology: a disaggregated scenario names a prefill pool and a decode pool, and the
// instances a pool holds are its layout's groups the pool's nodes fit (the same count the run would
// be refused above), so the scenario alone states how many of each the deployment ran
// (inference-sim#2126). Any topology flag, --num-instances included, keeps the command line's word:
// the counts are then the flags' and the existing checks apply to them. It reports whether it set
// the counts, which the caller records as the run's P/D topology.
func takePDTopologyFromPools(cmd *cobra.Command, a build.Adopted) (bool, error) {
	if !a.Disaggregated || kernelScenario == "" || cmd.Flags().Changed("num-instances") ||
		prefillInstances != 0 || decodeInstances != 0 || prefillDecodeInstances != 0 || encodeInstances != 0 {
		return false, nil
	}
	repos := kernelmodel.Repos{Scenarios: kernelScenarioDir, Data: mustRunData()}
	count := func(role deployment.Role) (int, error) {
		m, err := kernelmodel.OpenRole(kernelScenario, repos, role)
		if err != nil {
			return 0, fmt.Errorf("the scenario's %s pool: %w", role, err)
		}
		// A group is one instance (build.RequireGroupCapacity).
		n := m.GroupCapacity()
		if n < 1 {
			return 0, fmt.Errorf("the scenario's %s pool holds no instance at its layout (%v); give it the nodes its layout needs",
				role, m.GroupLayoutError())
		}
		return n, nil
	}
	p, err := count(deployment.RolePrefill)
	if err != nil {
		return false, err
	}
	d, err := count(deployment.RoleDecode)
	if err != nil {
		return false, err
	}
	prefillInstances, decodeInstances, numInstances = p, d, p+d
	logrus.Warnf("P/D topology taken from the scenario's pools: %d prefill and %d decode instance(s)", p, d)
	return true, nil
}

// adoptInputs are the scenario and roots the command line names, handed over unread, with the
// routing overrides applied just before the first pool opens (as before the move).
func adoptInputs(cmd *cobra.Command) build.AdoptInputs {
	return build.AdoptInputs{
		Scenario: kernelScenario, ScenariosDir: kernelScenarioDir,
		Data:       mustRunData(),
		PDTopology: prefillInstances != 0 || decodeInstances != 0,
		BeforeOpen: func(kernelmodel.Shape) error { applyRoutingOverrides(cmd); return nil },
	}
}

// adoptScenario reads the scenario's deployment (build.Adopt).
func adoptScenario(in build.AdoptInputs) (build.Adopted, error) { return build.Adopt(in) }

// openFirstPool opens and sizes the scenario's first pool (build.OpenFirstPool).
func openFirstPool(in build.AdoptInputs, a *build.Adopted) error { return build.OpenFirstPool(in, a) }

// openModel opens the scenario's first pool for a caller that resolves the latency model
// without adopting the deployment (build.OpenModel).
func openModel() (*kernelmodel.Model, error) {
	data, err := runData()
	if err != nil {
		return nil, err
	}
	return build.OpenModel(kernelScenario, kernelScenarioDir, data)
}

// openChipModel opens the pool of the scenario file name in dir and sizes it, so a chip the
// scenario cannot run on is refused when the model is opened (a node pool's variant).
func openChipModel(name, dir string) (*engineModel, error) {
	m, err := build.OpenModel(name, dir, mustRunData())
	if err != nil {
		return nil, err
	}
	if _, err := m.SettingsCheckedByGroups(); err != nil {
		return nil, fmt.Errorf("the scenario does not run on this chip: %w", err)
	}
	return m, nil
}

// expertModelConfig is the engine's model configuration as the scenario's graph states it: its
// expert geometry only (routed experts and experts per token, 0 for a dense model). Nothing
// else of ModelConfig is populated: no HF config is parsed, and a half-filled struct read as
// complete is how a dense model comes to be treated as MoE.
func expertModelConfig() modelConfig {
	var c modelConfig
	c.NumLocalExperts = kernelDeploymentExperts
	c.NumExpertsPerTok = kernelDeploymentTopK
	return c
}

// checkParallelism refuses data and expert parallelism a dense model cannot run
// (build.CheckParallelism).
func checkParallelism(moe bool) error {
	return build.CheckParallelism(kernelScenario, model, dataParallelism, enableExpertParallel, moe)
}

// The engine's types that the rest of cmd names: the kernel for one pool, the scenario's
// deployment shape, and the routing overrides handed to the pools; and the compatibility
// engine's configuration types, which only the build site opens (the host holds the engine's
// configuration unopened, sim.InstanceConfig.Engine).
type (
	engineModel      = kernelmodel.Model
	scenarioShapeT   = kernelmodel.Shape
	routingOverrides = kernelmodel.RoutingOverrides

	engineConfig      = cengine.SimConfig
	kvOffloadConfig   = cengine.KVOffloadConfig
	kvOffloadTier     = cengine.KVOffloadTier
	kvSequenceState   = cengine.KVSequenceState
	kvPrefixHit       = cengine.KVPrefixHit
	modelConfig       = cengine.ModelConfig
	loraConfig        = cengine.LoRAConfig
	batchConfig       = cengine.BatchConfig
	speculativeConfig = cengine.SpeculativeConfig
	latencyModel      = cengine.LatencyModel
	stepOverheadTier  = cengine.StepOverheadTier
)

// setRoutingOverrides hands the command line's routing layer to the pools about to open.
func setRoutingOverrides(o routingOverrides) { kernelmodel.SetRoutingOverrides(o) }

// setAcceptanceOverrides hands the kernel the acceptance layers beyond the scenario file, the
// ones build.ResolveAcceptance read: the command line's value, the workload's terms (a catalog
// preset, with the command line's category and thinking mode applied), and the fallback category.
func setAcceptanceOverrides(fl *pflag.FlagSet, k int) {
	kernelmodel.SetAcceptanceOverrides(acceptanceOverridesOf(fl, k))
}

// acceptanceOverridesOf is the run's acceptance layers beyond the scenario file
// (setAcceptanceOverrides).
func acceptanceOverridesOf(fl *pflag.FlagSet, k int) kernelmodel.AcceptanceOverrides {
	var cli *acceptance.Value
	switch {
	case fl.Changed(build.AcceptanceRateFlag):
		r := speculativeAcceptance
		cli = &acceptance.Value{NumSpecTokens: k, AcceptanceRate: &r, Source: "--" + build.AcceptanceRateFlag}
	case fl.Changed(build.AcceptanceLengthFlag):
		l := speculativeAcceptanceLength
		cli = &acceptance.Value{NumSpecTokens: k, AcceptanceLength: &l, Source: "--" + build.AcceptanceLengthFlag}
	}
	if cli != nil {
		cli.Realisation = acceptance.Realisation(speculativeRealisation)
	}
	var terms *acceptance.Workload
	if workloadAcceptance != nil || fl.Changed(build.AcceptanceThinkingFlag) || fl.Changed(build.AcceptanceCategoryFlag) {
		var w acceptance.Workload
		if workloadAcceptance != nil {
			w = *workloadAcceptance
		}
		if fl.Changed(build.AcceptanceThinkingFlag) {
			w.Thinking = acceptance.Thinking(speculativeThinking)
		}
		if fl.Changed(build.AcceptanceCategoryFlag) {
			w.Category = speculativeCategory
		}
		terms = &w
	}
	return kernelmodel.AcceptanceOverrides{CLI: cli, Workload: terms, FallbackCategory: speculativeFallbackCategory}
}

// currentRoutingOverridesOf is the routing layer the pools will open with.
func currentRoutingOverridesOf() routingOverrides { return kernelmodel.CurrentRoutingOverrides() }

// resolveLoRAConfig assembles the run's LoRA configuration (build.ResolveLoRAConfig) from
// defaults.yaml's cost coefficients, the optional --lora-config file and the scalar --lora-*
// flags (each only when Changed, R18), and validates it. The single LoRAConfig construction site
// (R4), called by both run and replay (INV-13); fatal on an invalid config (Principle V).
func resolveLoRAConfig(cmd *cobra.Command) loraConfig {
	var defs *build.LoRADefaults
	if d := loadRunDefaults(cmd.Flags().Changed("defaults-filepath")).LoRADefaults; d != nil {
		defs = &build.LoRADefaults{
			LoadBaseLatencyUs:     d.LoadBaseLatencyUs,
			LoadBandwidthBytesUs:  d.LoadBandwidthBytesUs,
			FootprintBytesPerRank: d.FootprintBytesPerRank,
		}
		if len(d.StepOverheadTiers) > 0 {
			defs.StepOverheadTiers = make(map[int][2]float64, len(d.StepOverheadTiers))
			for rank, t := range d.StepOverheadTiers {
				defs.StepOverheadTiers[rank] = [2]float64{t.K6, t.K7}
			}
		}
	}
	changed := cmd.Flags().Changed
	cfg, err := build.ResolveLoRAConfig(loraConfigPath, defs, build.LoRAFlags{
		AdapterCapacity: loraAdapterCapacity, AdapterCapacitySet: changed("lora-adapter-capacity"),
		LoadBaseLatencyUs: loraLoadBaseLatencyUs, LoadBaseLatencyUsSet: changed("lora-load-base-latency-us"),
		LoadBandwidthBytesUs: loraLoadBandwidthBytesUs, LoadBandwidthBytesUsSet: changed("lora-load-bandwidth-bytes-us"),
		FootprintBytesPerRank: loraFootprintBytesPerRank, FootprintBytesPerRankSet: changed("lora-footprint-bytes-per-rank"),
	})
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return cfg
}

// batchConfigFromCLI is the run's batch configuration, the part of engineConfigFor's the
// prefix-caching toggle reaches (build.BatchConfigOf).
func batchConfigFromCLI() batchConfig {
	return build.BatchConfigOf(engineSettings(modelConfig{}, 1, kvOffloadConfig{}, loraConfig{}, speculativeConfig{}))
}

// engineConfigFor is the engine's configuration for this run or replay (build.EngineConfigOf),
// from the settings the scenario, the kernel and the command line resolved: the one place both
// commands assemble it (INV-13). dp is the scenario's data parallelism.
// It resolves the speculative configuration, which records the acceptance for the results file.
func engineConfigFor(cmd *cobra.Command, m modelConfig, dp int, offload kvOffloadConfig,
	lora loraConfig) build.EngineConfig {
	return build.EngineConfigOf(engineSettings(m, dp, offload, lora, resolveSpeculativeConfig(cmd)))
}

// engineConfigOfRun is the run's engine configuration both commands build every instance's
// engine from: the engine's settings (ec, from engineConfigFor) and its front end, beside the
// simulator's own settings the engine reads (the horizon, the seed, the latency model, the SLO
// priority overrides and the server series). The one place cmd assembles it (INV-13).
func engineConfigOfRun(ec build.EngineConfig, horizon, seed int64, latency latencyModel, sloOverrides map[string]int) engineConfig {
	return engineConfig{
		Horizon: horizon,
		Seed:    seed,
		// The engine's configuration comes from the build site (engineConfigFor), the same for
		// run and replay (INV-13).
		KVCacheConfig:        ec.KVCache,
		BatchConfig:          ec.Batch,
		ModelHardwareConfig:  ec.ModelHardware,
		PolicyConfig:         ec.Policy,
		LoRAConfig:           ec.LoRA,
		SpeculativeConfig:    ec.Speculative,
		Frontend:             frontendConfigFromCLI(),
		SLOPriorityOverrides: sloOverrides,
		// The kernel prices every step.
		LatencyModel:           latency,
		ServerSeriesIntervalUs: serverSeriesIntervalUs(),
	}
}

// instanceConfigOf is an engine configuration as the host holds it: the settings the host reads,
// copied out, beside the engine's configuration itself, unopened (sim.InstanceConfig.Engine),
// which the engine factory opens again (engineConfigIn).
func instanceConfigOf(c engineConfig) sim.InstanceConfig {
	return sim.InstanceConfig{Horizon: c.Horizon, Seed: c.Seed, GPU: c.GPU, TP: c.TP, MaxNumSeqs: c.MaxNumSeqs,
		Model: c.Model, BlockSizeTokens: c.BlockSizeTokens, Engine: c}
}

// engineConfigIn is the engine configuration an instance configuration holds, with the host's
// settings the cluster may have changed for this instance applied: the GPU its node pool placed
// it on, and the horizon a run that learns its end while running set.
func engineConfigIn(c sim.InstanceConfig) engineConfig {
	ec, ok := c.Engine.(engineConfig)
	if !ok {
		panic(fmt.Sprintf("engine build: an instance configuration holds %T, not the engine's configuration", c.Engine))
	}
	ec.GPU, ec.Horizon = c.GPU, c.Horizon
	return ec
}

// hostOverheads sets a driver's host work from the latency model that prices the instance's
// engine: the post-decode and per-token output overheads. A request's admission time is not
// here: the driver asks the Environment what the request demands of the engine's admission queue.
func hostOverheads(dcfg *sim.EngineDriverConfig, m latencyModel) {
	if m == nil {
		return
	}
	dcfg.PostDecodeFixedOverhead = m.PostDecodeFixedOverhead()
	dcfg.OutputTokenProcessingTime = m.OutputTokenProcessingTime()
}

// streamIntervalOf is the pool's engine.stream_interval, how many tokens the API server holds
// before streaming them to the client (vllm/config/scheduler.py:213@5df63a72), which the driver
// applies to the delivery of tokens. Zero, unstated, is vLLM's default of 1. A negative value never
// gets here: the scenario's validation refuses it. A model that is not the engine's own reports
// none.
func streamIntervalOf(m latencyModel) (int, error) {
	km, ok := m.(*engineModel)
	if !ok {
		return 0, nil
	}
	e, err := km.Engine()
	if err != nil {
		return 0, fmt.Errorf("reading the pool's engine settings: %w", err)
	}
	return e.StreamInterval, nil
}

// Result closeness data (G1): the engine's resolutions, labelled by pool. This reads the pool
// models, so it is here at the build site.

// engineResolutions is the Result's resolutions: for each pool, the engine's own value of each
// setting the cost model resolves beside the host's, labelled with the pool (result.PoolLabels
// of the run's deployment) and sorted by setting, then pool.
func engineResolutions(models []*engineModel, dep deployment.Deployment) ([]result.Resolution, error) {
	if scenarioEngine.Kind == engineKindSGLang {
		return sglangResolutions(dep), nil
	}
	if len(models) == 0 {
		return nil, nil
	}
	indices, err := build.PoolIndices(kernelScenario, kernelmodel.Repos{Scenarios: kernelScenarioDir, Data: mustRunData()})
	if err != nil {
		return nil, err
	}
	labels := result.PoolLabels(dep)
	var out []result.Resolution
	for _, m := range models {
		out = append(out, result.ResolutionsOfPool(m.Kernel().Resolved(), labels[indices[m.Role()]])...)
	}
	result.SortResolutions(out)
	return out, nil
}

// provenanceOf is a pool's kernel evidence as the host's report records it, field for field.
func provenanceOf(p cengine.KernelPoolProvenance) sim.KernelPoolProvenance {
	out := sim.KernelPoolProvenance{MethodCounts: p.MethodCounts, Evidenced: p.Evidenced, Total: p.Total,
		EvidenceSource: p.EvidenceSource, NotEvidenced: p.NotEvidenced, Assumptions: p.Assumptions,
		SkippedRules: p.SkippedRules}
	if p.SkippedRuleFindings != nil {
		out.SkippedRuleFindings = make([]sim.RuleFinding, len(p.SkippedRuleFindings))
		for i, f := range p.SkippedRuleFindings {
			out.SkippedRuleFindings[i] = sim.RuleFinding(f)
		}
	}
	for _, s := range p.Substitutions {
		out.Substitutions = append(out.Substitutions, sim.Substitution(s))
	}
	if p.Routing != nil {
		r := sim.RoutingRecord(*p.Routing)
		out.Routing = &r
	}
	return out
}

// rolesOf is a disaggregated deployment's per-role instance configurations: each P/D pool's own
// engine settings (prefill, decode) applied to the run's engine configuration (resolvePoolConfig).
// Empty for a colocated run.
func rolesOf(global engineConfig, prefill, decode poolOverrides) cluster.PoolRoleConfigs {
	for _, o := range []struct {
		name string
		o    poolOverrides
	}{{"prefill pool", prefill}, {"decode pool", decode}} {
		if err := o.o.Validate(o.name); err != nil {
			logrus.Fatalf("%v", err)
		}
	}
	var r cluster.PoolRoleConfigs
	if !prefill.IsEmpty() {
		c := instanceConfigOf(resolvePoolConfig(global, prefill))
		r.Prefill = &c
	}
	if !decode.IsEmpty() {
		c := instanceConfigOf(resolvePoolConfig(global, decode))
		r.Decode = &c
	}
	return r
}

// engineSettings gathers the run's resolved engine settings, unread, for build.EngineConfigOf.
func engineSettings(m modelConfig, perRankDP int, offload kvOffloadConfig,
	lora loraConfig, spec speculativeConfig) build.EngineSettings {
	return build.EngineSettings{
		TotalKVBlocks: totalKVBlocks, BlockSizeTokens: blockSizeTokens,
		KVOffload:     offload,
		SequenceState: sequenceState, PrefixHit: prefixHit,
		MaxNumSeqs: maxNumSeqs, MaxNumBatchedTokens: maxNumBatchedTokens,
		LongPrefillTokenThreshold: longPrefillTokenThreshold, PrefixCachingDisabled: noEnablePrefixCaching,
		Model: m, ModelName: model, GPU: gpu, TP: tensorParallelism, DP: perRankDP,
		EnableExpertParallel: enableExpertParallel, MaxModelLen: maxModelLen,
		Scheduler: scheduler, PreemptionPolicy: preemptionPolicy,
		LoRA: lora, Speculative: spec,
	}
}

// speculativeConfigOf is the engine's speculative-decoding configuration
// (build.SpeculativeConfigOf), fatal on an invalid combination (CLI boundary, R6).
func speculativeConfigOf(k int, acceptance float64, method string) speculativeConfig {
	c, err := build.SpeculativeConfigOf(k, acceptance, method)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return c
}

// adapterReservedBytesFor is the static LoRA HBM reservation, in bytes, the kernel's KV budget
// sets aside for a resolved config (build.AdapterReservedBytes): 0 when the subsystem is inert
// (INV-6), fatal on a malformed cost config (Principle V). Both run and replay set
// loraReservedBytesForKV from it.
func adapterReservedBytesFor(cfg loraConfig) int64 {
	n, err := build.AdapterReservedBytes(cfg)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return n
}

// adapterRegistryOf is the declared adapters' registry (build.AdapterRegistryOf), nil when none
// are declared, for checking the workload's adapter references; fatal on an invalid registry.
func adapterRegistryOf(cfg loraConfig) workload.AdapterRegistry {
	r, err := build.AdapterRegistryOf(cfg)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return r
}

// adapterIDsOf lists the LoRA adapter IDs a run declares, in declaration order.
func adapterIDsOf(cfg loraConfig) []string {
	ids := make([]string, len(cfg.Adapters))
	for i, a := range cfg.Adapters {
		ids[i] = a.ID
	}
	return ids
}

// offloadRecord and offloadTierRecord are a trace header's record of the KV-offload config,
// in the engine's terms (build.OffloadRecord).
type (
	offloadRecord     = build.OffloadRecord
	offloadTierRecord = build.OffloadTierRecord
)

// kvOffloadOfRecord is the engine's offload config rebuilt from a trace's record of it
// (build.KVOffloadOfRecord); nil is the inert config.
func kvOffloadOfRecord(r *offloadRecord) kvOffloadConfig { return build.KVOffloadOfRecord(r) }

// reconcileReplayKVOffload is replay's KV-offload config: the trace header's, authoritative
// (BC-G6), reconciled with the scenario's offload block, which may only restate it
// (build.ReconcileReplayKVOffload); fatal on a conflict or an unreproducible record.
// allowScenarioAdd is for a converted trace or an agentic replay, which record no BLIS
// deployment (replayRecordsDeployment), so the scenario's block states their tiers.
func reconcileReplayKVOffload(headerBlock *workload.TraceKVOffloadConfig, allowScenarioAdd bool) kvOffloadConfig {
	return reconcileReplayBlock(headerBlock, scenarioOffload, allowScenarioAdd)
}

// reconcileReplayDecodeKVOffload is the same for the decode pool of a P/D replay. The trace header
// records one offload block, which both pools ran (a run whose pools offload differently writes no
// trace, refuseUnrecordableOffload), so the decode pool is reconciled against the same record.
func reconcileReplayDecodeKVOffload(headerBlock *workload.TraceKVOffloadConfig, allowScenarioAdd bool) kvOffloadConfig {
	return reconcileReplayBlock(headerBlock, scenarioDecodeOffload, allowScenarioAdd)
}

func reconcileReplayBlock(headerBlock *workload.TraceKVOffloadConfig, scenario *deployment.Offload, allowScenarioAdd bool) kvOffloadConfig {
	// The scenario's offload block is a source on replay as on run (resolveKVOffloadConfig),
	// and the result must match the header.
	stated := scenario != nil
	var scenarioCfg kvOffloadConfig
	if stated {
		// resolveOffloadBlock itself logrus.Fatalf's on a block the engine refuses.
		scenarioCfg = resolveOffloadBlock(scenario)
	}
	cfg, err := resolveReplayKVOffload(headerBlock, stated, scenarioCfg, allowScenarioAdd)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return cfg
}

// refuseUnrecordableOffload refuses to write a trace of a P/D run whose pools run different
// offload: the trace header records one offload block, so a replay could not rebuild the run
// (INV-13, never silent degradation). A run that writes no trace is unaffected.
func refuseUnrecordableOffload(prefill, decode kvOffloadConfig) error {
	if reflect.DeepEqual(prefill, decode) {
		return nil
	}
	return fmt.Errorf("the prefill and decode pools run different KV offload, which a trace header cannot record " +
		"(it holds one offload block, so a replay could not rebuild the run); run without --trace-output, or give both pools " +
		"the same offload")
}

// resolveReplayKVOffload is reconcileReplayKVOffload's error-returning core, over the trace
// header's record (build.ReconcileReplayKVOffload).
func resolveReplayKVOffload(headerBlock *workload.TraceKVOffloadConfig, stated bool, scenarioCfg kvOffloadConfig, allowScenarioAdd bool) (kvOffloadConfig, error) {
	return build.ReconcileReplayKVOffload(headerToSimOffload(headerBlock), stated, scenarioCfg, allowScenarioAdd)
}

// frontendConfigFromCLI is the engine's front-end config from --renderer-num-workers and
// --api-server-count (build.FrontendConfigOf), the single run/replay seam (INV-13); fatal on
// a negative count.
func frontendConfigFromCLI() cengine.FrontendConfig {
	c, err := build.FrontendConfigOf(rendererNumWorkers, apiServerCount)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return c
}

// registerContractEngines registers the engine kind a contract-built release is built by:
// the vLLM engine's translated v0.33.0 engine, with the latency kernel's step pricing and KV sizing
// (its Register). It is the one place this binary links an engine the contract reaches;
// later calls are no-ops.
func registerContractEngines() { coreengine.Register() }

// EngineReleases are the upstream engine releases this binary runs, one per kind, as each
// engine module records its own (for vLLM, blisvllm.Upstream). stack.json lists them (LIF-6).
func EngineReleases() []stack.Upstream {
	sk, sv, sc := sglangEngineRelease()
	return []stack.Upstream{
		{Kind: sk, Version: sv, Commit: sc},
		{Kind: blisvllm.Kind, Version: blisvllm.Upstream.Version, Commit: blisvllm.Upstream.Commit},
	}
}

// vllmEngineModel is the vLLM engine as this build runs it, for `blis version`. Two releases are
// recorded, and they are different facts:
//
//   - ModeledVersion is the release whose serving behaviour the kernel prices (the substrate's
//     EngineBehaviourVersion), which is the release every Result's kernel assumptions cite. A
//     benchmarked figure is comparable with it.
//   - BuildVersion is the upstream tree the engine's rules were ported from
//     (blisvllm.Upstream). It may be a commit with no release tag, so it is provenance only and a
//     price is never reported against it.
//
// A vLLM pool can take any deployment role: colocated, or one side of a prefill/decode
// deployment. Whether a particular model can disaggregate (its attention backend may have no KV
// connector) is a per-model fact the run itself refuses, not a fact about the kind.
func vllmEngineModel() engineReport {
	// The upstream tree the rules came from is the build's own pin, read where stack.json and
	// `blis version` read it, so the two reports cannot drift.
	upstream := builtEngines()[blisvllm.Kind]
	return engineReport{
		Kind:           upstream.Kind,
		ModeledVersion: latencykernel.EngineBehaviourVersion,
		ModeledCommit:  latencykernel.EngineBehaviourCommit,
		BuildVersion:   upstream.Version,
		BuildCommit:    upstream.Commit,
		Roles: []deployment.Role{
			deployment.RoleColocated,
			deployment.RolePrefill,
			deployment.RoleDecode,
		},
	}
}

// contractRoots are the roots the engine factory reads beside the run's Data: the scenario
// file's path, which it opens itself. The catalog, registry and overlay roots are absent on
// purpose: the engine reads the run's opened Data as it is and refuses a root that names other
// data (vllmengine.Input), so the Data is opened with the run's catalog, registry and overlays
// (dataOptions) and passed whole.
func contractRoots(scenarioPath string) map[string]string {
	return map[string]string{coreengine.RootScenario: scenarioPath}
}

// dpRankViews exposes a built external data-parallel group as one engine.Engine per rank — the
// per-endpoint views the router sees under external load balancing (inference-sim#2699, H3),
// through the engine module's host facade (vllmengine.DPRankViews). It reports false for any engine
// that is not such a shared group, so the caller keeps its single collapsed instance. cmd reaches
// the view engine only through engine.Engine and the facade, never the engine's core/vllm or
// dprankview packages (the reach-the-engine-only-through-the-contract boundary).
func dpRankViews(eng engine.Engine) ([]engine.Engine, bool) {
	return coreengine.DPRankViews(eng)
}

// newContractEngine builds one engine from spec over the run's opened data. Every engine of a
// run is built through here with the same Data, so the run has one selected-entry report and
// the data is opened once, not once for each build. A variable so a test can count the builds
// and the Data each was handed.
var newContractEngine = func(spec engine.Spec, data *load.Data) (engine.Engine, error) {
	if spec.Kind == engineKindSGLang {
		return newSGLangEngine(spec, data)
	}
	return coreengine.New(coreengine.Input{Spec: spec, Data: data})
}

// engineFieldList is the field list of the engine the scenario at scenarioPath names, as that
// engine reports it through engine.Engine.Fields (design section 7, X19). The engine is built the
// way a run builds its first pool's, over data, so the list is the one the run's engine would
// report; this binary holds no list of its own. The contract has no way to ask a kind for its
// list without building an engine, so a scenario is the input.
func engineFieldList(scenarioPath string, data *load.Data, seed int64) (fields.FieldList, error) {
	sel, err := selectEngine(scenarioPath)
	if err != nil {
		return fields.FieldList{}, err
	}
	registerContractEngines()
	eng, err := newContractEngine(engine.Spec{
		Kind:            sel.Kind,
		UpstreamVersion: sel.Version,
		Roots:           contractRoots(scenarioPath),
		Seed:            seed,
	}, data)
	if err != nil {
		return fields.FieldList{}, fmt.Errorf("scenario %s: engine %s: %w", scenarioPath, sel, err)
	}
	return eng.Fields(), nil
}

// Moved from sim/cluster/resolve.go at 739a519b, when the engine's configuration left the host
// (phase 2c): a P/D pool's engine settings are applied to the run's engine configuration here,
// at the build site, and the cluster receives each pool role's finished instance configuration
// (cluster.PoolRoleConfigs, rolesOf).

// poolOverrides holds optional per-pool hardware overrides for PD disaggregation.
// Nil pointer / empty string means "use global config" for that field.
// Pointer types for TP, MaxModelLen, TotalKVBlocks to distinguish "not set" (nil = use
// global) from an explicit value. The CLI fills them from each P/D pool's kernel
// (TotalKVBlocks is the kernel's per-rank budget).
//
// Contract for library callers constructing poolOverrides directly (bypassing CLI):
// - *TP must be > 0 when non-nil
// - *MaxModelLen must be > 0 when non-nil
type poolOverrides struct {
	TP            *int   // tensor parallelism (nil = use global)
	GPU           string // GPU type ("" = use global)
	MaxModelLen   *int64 // max sequence length (nil = use global)
	TotalKVBlocks *int64 // KV blocks (nil = use global; set by the CLI from the kernel)

	// LatencyModel prices this pool's steps (nil = use global). A disaggregated deployment
	// runs a different engine per role -- its own parallelism, token budget and graph mode
	// -- so each role is priced by the kernel for its own pool rather than by one copied to
	// every pool.
	LatencyModel latencyModel
	// MaxNumSeqs, MaxNumBatchedTokens and PrefixCachingDisabled are the pool's own engine
	// admission settings (nil = use global), for the same reason.
	MaxNumSeqs            *int64
	MaxNumBatchedTokens   *int64
	PrefixCachingDisabled *bool

	// KVOffload is the pool's own KV offload (nil = use global): the run's one tier
	// description, sized by the pool's bytes per block and priced by its own kernel, since
	// pools of different widths hold different bytes per block.
	KVOffload *kvOffloadConfig
	// SequenceState is the pool's bounded per-sequence state (nil = use global): pools of
	// different widths hold different state per sequence and per block (#1941).
	SequenceState *kvSequenceState
	// PrefixHit is how the pool's KV cache groups cut a prefix hit (nil = use global).
	PrefixHit *kvPrefixHit

	// Scheduler and PreemptionPolicy are the pool's own engine scheduling policy ("" = use
	// global): each P/D pool's engine states its own vLLM scheduling_policy, which decides both
	// its queue order and its preemption victim.
	Scheduler        string
	PreemptionPolicy string

	// LongPrefillTokenThreshold is the pool engine's long_prefill_token_threshold (nil = use
	// global; https://github.com/inference-sim/blis-schemas/issues/75), and DPLoadBalancing its data_parallel_load_balancing mode
	// ("" = use global; #76): vLLM sets both per engine, so prefill and decode pools may differ.
	LongPrefillTokenThreshold *int64
	DPLoadBalancing           string

	// Speculative is the pool's own speculative configuration (nil = use global): vLLM takes each
	// server's speculative_config, and the P/D connector's fingerprint carries none of its fields,
	// so the prefill and decode pools may draft differently (inference-sim#2125).
	Speculative *speculativeConfig
}

// Validate checks that non-nil pointer fields satisfy their constraints (R3).
// name is used in error messages (e.g., "prefill pool" or "decode pool").
// Library callers that construct poolOverrides directly (bypassing CLI validation)
// should call Validate before passing overrides to DeploymentConfig.
func (o poolOverrides) Validate(name string) error {
	if o.TP != nil && *o.TP <= 0 {
		return fmt.Errorf("%s: poolOverrides.TP must be > 0 when set, got %d", name, *o.TP)
	}
	if o.MaxModelLen != nil && *o.MaxModelLen <= 0 {
		return fmt.Errorf("%s: poolOverrides.MaxModelLen must be > 0 when set, got %d", name, *o.MaxModelLen)
	}
	if o.TotalKVBlocks != nil && *o.TotalKVBlocks <= 0 {
		return fmt.Errorf("%s: poolOverrides.TotalKVBlocks must be > 0 when set, got %d", name, *o.TotalKVBlocks)
	}
	if o.MaxNumSeqs != nil && *o.MaxNumSeqs <= 0 {
		return fmt.Errorf("%s: poolOverrides.MaxNumSeqs must be > 0 when set, got %d", name, *o.MaxNumSeqs)
	}
	if o.MaxNumBatchedTokens != nil && *o.MaxNumBatchedTokens <= 0 {
		return fmt.Errorf("%s: poolOverrides.MaxNumBatchedTokens must be > 0 when set, got %d", name, *o.MaxNumBatchedTokens)
	}
	if o.SequenceState != nil && (o.SequenceState.Hold == nil) != (o.SequenceState.Gate == nil) {
		return fmt.Errorf("%s: poolOverrides.SequenceState needs both Hold and Gate, or neither", name)
	}
	if o.LongPrefillTokenThreshold != nil && *o.LongPrefillTokenThreshold < 0 {
		return fmt.Errorf("%s: poolOverrides.LongPrefillTokenThreshold must be >= 0 when set, got %d", name, *o.LongPrefillTokenThreshold)
	}
	if !cluster.IsValidDPLoadBalancing(o.DPLoadBalancing) {
		return fmt.Errorf("%s: poolOverrides.DPLoadBalancing %q is not internal, hybrid or external", name, o.DPLoadBalancing)
	}
	if o.KVOffload != nil && o.KVOffload.IsEnabled() && o.KVOffload.PerBlockBytes <= 0 {
		return fmt.Errorf("%s: poolOverrides.KVOffload must state PerBlockBytes > 0 when enabled, got %d", name, o.KVOffload.PerBlockBytes)
	}
	return nil
}

// IsEmpty returns true when no overrides are set.
func (o poolOverrides) IsEmpty() bool {
	return o.TP == nil && o.GPU == "" &&
		o.MaxModelLen == nil && o.TotalKVBlocks == nil &&
		o.LatencyModel == nil && o.MaxNumSeqs == nil && o.MaxNumBatchedTokens == nil &&
		o.PrefixCachingDisabled == nil && o.KVOffload == nil &&
		o.SequenceState == nil && o.PrefixHit == nil && o.Scheduler == "" && o.PreemptionPolicy == "" &&
		o.LongPrefillTokenThreshold == nil && o.DPLoadBalancing == "" && o.Speculative == nil
}

// resolvePoolConfig applies per-pool overrides to a global SimConfig.
// Returns a new SimConfig with overridden fields; the global config is not mutated.
//
// Struct-copy safety: ModelConfig is a pure value type (safe to copy).
// SLOPriorityOverrides is a map[string]int that shares its backing map across copies.
// Safe because NewSLOPriorityMap only reads the map (for range), never mutates. The
// per-pool LatencyModel is shared by reference: the kernel adapter is read-only once built.
func resolvePoolConfig(global engineConfig, overrides poolOverrides) engineConfig {
	resolved := global // struct copy

	if overrides.TP != nil {
		resolved.TP = *overrides.TP
	}
	if overrides.GPU != "" {
		resolved.GPU = overrides.GPU
	}
	if overrides.MaxModelLen != nil {
		resolved.MaxModelLen = *overrides.MaxModelLen
	}
	if overrides.TotalKVBlocks != nil {
		resolved.TotalKVBlocks = *overrides.TotalKVBlocks
	}
	if overrides.LatencyModel != nil {
		resolved.LatencyModel = overrides.LatencyModel
	}
	if overrides.MaxNumSeqs != nil {
		resolved.MaxNumSeqs = *overrides.MaxNumSeqs
	}
	if overrides.MaxNumBatchedTokens != nil {
		resolved.MaxNumBatchedTokens = *overrides.MaxNumBatchedTokens
	}
	if overrides.PrefixCachingDisabled != nil {
		resolved.PrefixCachingDisabled = *overrides.PrefixCachingDisabled
	}
	if overrides.KVOffload != nil {
		resolved.Offload = *overrides.KVOffload
	}
	if overrides.SequenceState != nil {
		resolved.SequenceState = *overrides.SequenceState
	}
	if overrides.PrefixHit != nil {
		resolved.PrefixHit = *overrides.PrefixHit
	}
	if overrides.Scheduler != "" {
		resolved.Scheduler = overrides.Scheduler
	}
	if overrides.PreemptionPolicy != "" {
		resolved.PreemptionPolicy = overrides.PreemptionPolicy
	}
	if overrides.LongPrefillTokenThreshold != nil {
		resolved.LongPrefillTokenThreshold = *overrides.LongPrefillTokenThreshold
	}
	if overrides.Speculative != nil {
		resolved.SpeculativeConfig = *overrides.Speculative
	}

	return resolved
}

// The host's Environment (design 6.3): what the cost model answers about a transfer an engine
// started. The engine states each KV move it starts as engine.Transfer.Move, in the cost model's
// terms (the catalog device of a tier, the node and rack at each end, the direction, the tokens);
// the driver asks env.DemandFor(Environment, engine.Plan(), t) and gets the move's demand on the
// fabric: its payload, its floor and the links it crosses with their full rates. This is
// the cost model's side of that question for a kernel-backed run, and the only place the
// simulator turns a move into a demand. It prices nothing else: the engines at this pin open
// their pools' kernels themselves (engines/vllm/core/kernelmodel), so Graph, Chip, Memory, Price
// and Acceptance are refused here by name, never answered with a number nobody computed.
//
// One transferEnvironment is built for each engine, from the kernels of its own pool and, for a
// disaggregated run, of the pool it pulls from:
//
//   - A move between pools (plan.TransferPeer) is a decode engine's pull. The KV is the prefill
//     pool's, so the prefill pool's kernel states it (Kernel.PDTransferDemand) between the two
//     placements; the move's endpoints are each in their own pool's numbering, and a placement
//     is the pool's first node plus the endpoint's node. The KV moves in whole blocks, so the
//     tokens priced are the blocks' capacity.
//   - A move to or from a tier (plan.TransferToTier, plan.TransferFromTier) is an offload copy.
//     The kernel prices it as a time (Kernel.TierTime) and not as a demand, so the tier's link
//     is recovered from that price: the floor is the price of no bytes, and the rate is what a
//     payload large enough to be rate-bound gets (tierDemand).
type transferEnvironment struct {
	// own is the pool of the engine that asks, peer the pool its pulls read from (nil when
	// there is no other pool).
	own, peer *kernelmodel.Model
	// across is the pool a peer WRITE of this engine lands in when it leaves its own pool: the other
	// pool of a disaggregated run, nil in a colocated one (PeerWriteDemand).
	across *kernelmodel.Model
	// blockSize is the tokens in one KV block; the pools of a disaggregated run page alike.
	blockSize int64
	// rates are the tier links' rates, by tier and direction, recovered once (tierDemand).
	rates map[tierKey]float64
}

type tierKey struct {
	tier string
	dir  kernel.Direction
}

var _ env.Environment = (*transferEnvironment)(nil)

// newTransferEnvironment is the Environment of an engine in pool own, pulling from pool peer
// (nil for a colocated run). It fails when own states no block size.
func newTransferEnvironment(own, peer *kernelmodel.Model) (*transferEnvironment, error) {
	e, err := own.Engine()
	if err != nil {
		return nil, err
	}
	te := &transferEnvironment{own: own, peer: peer, blockSize: int64(e.BlockSize),
		rates: map[tierKey]float64{}}
	if peer != own {
		te.across = peer
	}
	return te, nil
}

// poolOrigin is where the pool's own numbering starts in the cluster's: the lowest node and the
// lowest rack among the places of the pool's groups (GroupPlacementOf, the places instanceEndpoint
// states). Every plan.Endpoint is counted from it in both coordinates (engine.PoolOrigin), so two
// pools whose local rack 0 are different racks of the cluster stay different when a pull's two
// ends are turned back into cluster placements. It is the minimum over the groups and not the
// first group's place, so that no group of the pool is below it.
func poolOrigin(m *kernelmodel.Model) engine.PoolOrigin {
	first := m.GroupPlacementOf(0)
	o := engine.PoolOrigin{Node: first.Node, Rack: first.Rack}
	for g := 1; g < m.GroupCapacity(); g++ {
		at := m.GroupPlacementOf(g)
		o.Node, o.Rack = min(o.Node, at.Node), min(o.Rack, at.Rack)
	}
	return o
}

// errHostEnvironment says which questions this Environment does not answer, and why.
func errHostEnvironment(what string) error {
	return fmt.Errorf("cmd: the host's Environment answers transfers only: %s is asked of the cost model by the "+
		"engine, which at this pin opens its pool's kernel itself (engines/vllm/core/kernelmodel)", what)
}

// Graph implements env.Environment; it is not answered here.
func (*transferEnvironment) Graph(string) (*specmodel.Graph, error) {
	return nil, errHostEnvironment("the model graph")
}

// Chip implements env.Environment; it is not answered here.
func (*transferEnvironment) Chip(string) (*hardware.Chip, error) {
	return nil, errHostEnvironment("the chip")
}

// Memory implements env.Environment; it is not answered here.
func (*transferEnvironment) Memory(plan.PoolPlan) (kernel.MemoryBreakdown, error) {
	return kernel.MemoryBreakdown{}, errHostEnvironment("memory")
}

// Price implements env.Environment; it is not answered here.
func (*transferEnvironment) Price(plan.PoolPlan, plan.StepShape) (kernel.StepEstimate, error) {
	return kernel.StepEstimate{}, errHostEnvironment("a step's price")
}

// HostDemand implements env.Environment: what a request of promptTokens asks of the admission
// queue the engine's plan states, from the pool's kernel (Kernel.AdmissionDemand).
func (e *transferEnvironment) HostDemand(_ plan.PoolPlan, promptTokens int) (kernel.HostDemand, error) {
	return e.own.Kernel().AdmissionDemand(promptTokens), nil
}

// Acceptance implements env.Environment; it is not answered here.
func (*transferEnvironment) Acceptance(string, string, string) (acceptance.Profile, error) {
	return acceptance.Profile{}, errHostEnvironment("an acceptance profile")
}

// TransferDemand implements env.Environment: what the move t asks of the fabric.
func (e *transferEnvironment) TransferDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error) {
	if err := p.CheckTransfer(t); err != nil {
		return kernel.TransferDemand{}, err
	}
	if t.Kind == plan.TransferPeer {
		return e.pullDemand(p, t)
	}
	return e.tierDemand(t)
}

// blocksOf is the whole KV blocks tokens occupy.
func (e *transferEnvironment) blocksOf(tokens int64) int64 {
	return (tokens + e.blockSize - 1) / e.blockSize
}

// placement is the cluster's placement of an endpoint in the numbering of pool m, node and rack
// both: the pool's origin added to each coordinate.
func placement(m *kernelmodel.Model, at plan.Endpoint) kernel.Placement {
	g := poolOrigin(m).Global(at)
	return kernel.Placement{Node: g.Node, Rack: g.Rack, Pool: m.PlacementOf(0).Pool}
}

// ownEndpoint refuses an endpoint of the engine's own pool that the pool's plan does not have;
// the far end of a pull is in another pool's plan, which this engine does not hold.
func ownEndpoint(p plan.PoolPlan, at plan.Endpoint) error {
	if nodes := max(1, p.Layout.Nodes); at.Node >= nodes {
		return fmt.Errorf("endpoint node %d of a pool with %d node(s)", at.Node, nodes)
	}
	return nil
}

// pullDemand is a decode engine's pull from the prefill pool: the prefill kernel's
// PDTransferDemand, in whole blocks, between the placements of the move's two ends. A pull of no
// tokens finds its KV already here and demands nothing.
func (e *transferEnvironment) pullDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error) {
	if t.Tokens == 0 {
		return kernel.TransferDemand{}, nil
	}
	if e.peer == nil {
		return kernel.TransferDemand{}, fmt.Errorf("a pull between pools, and this engine has no other pool to pull from")
	}
	if err := ownEndpoint(p, t.To); err != nil {
		return kernel.TransferDemand{}, fmt.Errorf("pull destination: %w", err)
	}
	from, to := placement(e.peer, t.From), placement(e.own, t.To)
	tokens := e.blocksOf(int64(t.Tokens)) * e.blockSize
	d := e.peer.Kernel().PDTransferDemand(int(tokens), from, to)
	if err := d.Check(); err != nil {
		return kernel.TransferDemand{}, fmt.Errorf("the kernel's demand for a %d-token pull from %+v to %+v is malformed: %w",
			tokens, from, to, err)
	}
	if at := d.AloneTime(); at < 0 || at >= time.Duration(math.MaxInt64/2) {
		return kernel.TransferDemand{}, fmt.Errorf("the kernel cannot price a %d-token KV pull from %+v to %+v "+
			"(it reports %v): the link between them states no bandwidth -- check the scenario's cluster.fabric "+
			"and the chip's IntraNodeBwGBps", tokens, from, to, at)
	}
	return d, nil
}

// PeerWriteDemand prices a p2p WRITE (sim.PeerWritePricer) of the bytes the engine states. The WRITE
// leaves the engine's own pool (from = placement(own, Move.From)). Its destination is in the
// peer's pool numbering: in a colocated run the engine's own pool, in a disaggregated run the
// other pool (placement of the opposite pool).
//
// Named limit (inference-sim/inference-sim#2439): a WRITE between two instances of one pool in a disaggregated run
// is read as crossing to the other pool, because a Move names an endpoint and not its pool.
//
// Between different global nodes the WRITE crosses the fabric: the kernel's PDTransferDemand links
// and floor (NIC egress and ingress, and uplinks across switch groups) with Bytes set to the
// WRITE's. Within one global node it is a copy between two places in host memory over one host
// link, "host/node/<node>", at the rate of the CPU tier's link recovered as tierDemand recovers
// it. That is a named substitution for design 6.1's host_copy_rate, which the registry does not
// hold at this pin (inference-sim#2203). A WRITE of no bytes, or one whose ends do not resolve,
// is refused.
func (e *transferEnvironment) PeerWriteDemand(p plan.PoolPlan, t engine.Transfer, bytes int64) (kernel.TransferDemand, error) {
	if bytes <= 0 {
		return kernel.TransferDemand{}, fmt.Errorf("a peer WRITE of %d bytes", bytes)
	}
	if err := ownEndpoint(p, t.Move.From); err != nil {
		return kernel.TransferDemand{}, fmt.Errorf("peer WRITE source: %w", err)
	}
	dest := e.own
	if e.peer != nil {
		dest = e.across
	}
	if dest == nil {
		return kernel.TransferDemand{}, fmt.Errorf("a peer WRITE in a disaggregated run, and this engine has no other pool to write to")
	}
	if t.Move.To.Node < 0 || t.Move.To.Rack < 0 || t.Move.From.Node < 0 || t.Move.From.Rack < 0 {
		return kernel.TransferDemand{}, fmt.Errorf("peer WRITE ends %+v -> %+v do not resolve", t.Move.From, t.Move.To)
	}
	from, to := placement(e.own, t.Move.From), placement(dest, t.Move.To)
	k := e.own.Kernel()
	if from.Node == to.Node {
		rate := e.hostRate()
		if rate <= 0 {
			return kernel.TransferDemand{}, fmt.Errorf("the CPU tier's link states no rate, so a same-node peer WRITE cannot be priced")
		}
		d := kernel.TransferDemand{Bytes: bytes, Floor: k.TierTime(cpuTierDevice, kernel.DirectionFromTier, 0, 1),
			Links: []kernel.LinkUse{{Link: kernel.LinkID(fmt.Sprintf("host/node/%d", from.Node)), BytesPerSecond: rate}}}
		return d, d.Check()
	}
	tokens := e.blocksOf(int64(t.Move.Tokens)) * e.blockSize
	d := k.PDTransferDemand(int(tokens), from, to)
	d.Bytes = bytes
	if err := d.Check(); err != nil {
		return kernel.TransferDemand{}, fmt.Errorf("the kernel's demand for a peer WRITE from %+v to %+v is malformed: %w", from, to, err)
	}
	if at := d.AloneTime(); at < 0 || at >= time.Duration(math.MaxInt64/2) {
		return kernel.TransferDemand{}, fmt.Errorf("the kernel cannot price a peer WRITE of %d bytes from %+v to %+v "+
			"(it reports %v): the link between them states no bandwidth", bytes, from, to, at)
	}
	return d, nil
}

// cpuTierDevice is the catalog device of the CPU tier.
const cpuTierDevice = build.OffloadCPUTier

// hostRate is the bytes per second of the CPU tier's link, recovered once as tierDemand does.
func (e *transferEnvironment) hostRate() float64 {
	key := tierKey{cpuTierDevice, kernel.DirectionFromTier}
	rate, ok := e.rates[key]
	if !ok {
		rate = tierRate(e.own.Kernel(), cpuTierDevice, kernel.DirectionFromTier)
		e.rates[key] = rate
	}
	return rate
}

// tierDemand is an offload copy to or from the catalog device t.Tier on the node that holds it:
// the whole blocks' bytes over one link at the rate the kernel prices the device. Alone, the
// demand costs exactly Kernel.TierTime of the same bytes (the kernel prices a copy at the
// device's bandwidth, the slower of it and the host link, and no less than its base latency),
// and the Environment refuses to state one that does not, rather than state a rate it cannot
// vouch for.
func (e *transferEnvironment) tierDemand(t plan.Transfer) (kernel.TransferDemand, error) {
	dir := kernel.DirectionFromTier
	if t.Kind == plan.TransferToTier {
		dir = kernel.DirectionToTier
	}
	k := e.own.Kernel()
	node := placement(e.own, t.From)
	bytes := e.blocksOf(int64(t.Tokens)) * k.SequenceVariableBytes(int(e.blockSize))
	price := k.TierTime(t.Tier, dir, bytes, 1)
	if price < 0 || price >= time.Duration(math.MaxInt64/2) {
		return kernel.TransferDemand{}, fmt.Errorf("the kernel cannot price a copy of %d bytes %s tier %q (it reports %v)",
			bytes, map[kernel.Direction]string{kernel.DirectionToTier: "to", kernel.DirectionFromTier: "from"}[dir], t.Tier, price)
	}
	key := tierKey{t.Tier, dir}
	rate, ok := e.rates[key]
	if !ok {
		// One rate for the link, which every copy over it states (the link arbiter refuses a link
		// stated at two). The recovered rate is a float that rounds once more than the kernel's
		// (tierRate's probe is 2^50 bytes priced in whole nanoseconds, so it is off by under one
		// part in 10^14), so it is raised by one part in 10^12: a copy alone then never drains later
		// than the kernel's price, and sooner by a nanosecond per 1000 s of copy at most, which
		// the copy's floor takes up below.
		rate = tierRate(k, t.Tier, dir) * (1 + 1e-12)
		e.rates[key] = rate
	}
	d := kernel.TransferDemand{
		Bytes: bytes,
		Floor: k.TierTime(t.Tier, dir, 0, 1),
		Links: []kernel.LinkUse{{
			Link:           kernel.LinkID(fmt.Sprintf("tier/%s/%s/node/%d", t.Tier, dirName(dir), node.Node)),
			BytesPerSecond: rate,
		}},
	}
	if d.AloneTime() < price {
		// The copy drains a little before the kernel's price: its floor is the price, so alone it
		// costs exactly that, and a copy that shares the link still drains at the link's one rate.
		d.Floor = price
	}
	if d.AloneTime() != price {
		return kernel.TransferDemand{}, fmt.Errorf("the demand of a copy of %d bytes %s tier %q costs %v alone but "+
			"the kernel prices it at %v", bytes, dirName(dir), t.Tier, d.AloneTime(), price)
	}
	if err := d.Check(); err != nil {
		return kernel.TransferDemand{}, err
	}
	return d, nil
}

func dirName(d kernel.Direction) string {
	if d == kernel.DirectionToTier {
		return "to"
	}
	return "from"
}

// tierRate is the bytes per second the kernel moves a copy at, recovered from its price of a
// payload so large that the rate binds, not the floor: bytes over price. The probe is the
// largest the price can state without saturating a Duration.
func tierRate(k kernel.Kernel, tier string, dir kernel.Direction) float64 {
	for probe := int64(1) << 50; probe > 1<<20; probe >>= 5 {
		if d := k.TierTime(tier, dir, probe, 1); d > 0 && d < time.Duration(math.MaxInt64/4) {
			return float64(probe) / d.Seconds()
		}
	}
	return 0
}

// Instance loading time from the instance's weights (#1215).
//
// A policy bundle's instance_lifecycle.weight_source names a catalog storage tier. An instance
// then loads in loading_delay (what does not depend on its weights) plus the time to read its
// resident weights from that tier. The weights are the cost model's own per-rank figure (the bytes
// it sizes KV against, design SUB-14), so a larger model, a wider tensor
// parallelism or a quantised checkpoint changes the delay with no extra input. The rate is the
// registry's offload_tier_<tier>_* entries for the instance's chip, read through the kernel's
// TierTime, so a faster chip or tier loads faster and a chip the registry has no entry for is
// refused by name.

// weightLoadTime is the time for an instance of tpDegree ranks to read its resident weights
// from tier on kernel k. The weights are the kernel's per-rank figure (FixedBytes, the bytes the
// cost model itself sizes KV against); every rank on a node reads at once and shares the tier
// (TierTime's queue depth), and the nodes of a multi-node instance read in parallel, so one
// node's time is the instance's. At this pin the engine's PoolPlan does not yet state resident
// weights (https://github.com/inference-sim/blis-latency-kernel/issues/84), so the kernel is the one source; when the plan does, this is
// the figure to replace. Weights kept in host memory are not in the kernel's figure and are not
// priced here.
func weightLoadTime(k kernel.Kernel, gpusPerNode int, tier string, tpDegree int) (time.Duration, error) {
	ranksOnNode := max(1, min(tpDegree, gpusPerNode))
	bytes := k.FixedBytes().Weights
	d := k.TierTime(tier, kernel.DirectionFromTier, bytes, ranksOnNode)
	if d < 0 || d >= time.Duration(math.MaxInt64/4) {
		return 0, fmt.Errorf("weight_source %q: the kernel cannot price a read of %d bytes from it", tier, bytes)
	}
	return d, nil
}

// weightLoader returns DeploymentConfig.WeightLoadTime: it prices an instance on the kernel of
// its pool's chip, the variant for a chip other than the scenario's own (withPoolChips) and the
// scenario's opened kernel otherwise.
func weightLoader(def *engineModel, variants map[string]*poolChip) func(gpu, tier string, tpDegree int) (time.Duration, error) {
	return func(gpu, tier string, tpDegree int) (time.Duration, error) {
		m := def
		if v, ok := variants[strings.ToLower(gpu)]; ok {
			m = v.model
		}
		return weightLoadTime(m.Kernel(), m.Packaging().GPUsPerNode, tier, tpDegree)
	}
}

// requireWeightSource refuses, before the run starts, a weight source that the kernel of any
// chip the run places instances on cannot price. The refusal names the chip and the registry
// entries the tier needs (the TierTime of a tier with no entry is unbounded, never zero).
func requireWeightSource(tier string, def *engineModel, variants map[string]*poolChip) {
	if tier == "" {
		return
	}
	chips := map[string]*engineModel{gpu: def}
	for _, v := range variants {
		chips[v.chip] = v.model
	}
	names := make([]string, 0, len(chips))
	for c := range chips {
		names = append(names, c)
	}
	slices.Sort(names)
	for _, c := range names {
		if d := chips[c].Kernel().TierTime(tier, kernel.DirectionFromTier, 1, 1); d < 0 || d >= time.Duration(math.MaxInt64/4) {
			logrus.Fatalf("policy bundle instance_lifecycle.weight_source %q cannot be priced on chip %q: the coefficient sets "+
				"of the scenario carry no %s, %s and %s for it (the registry set cost-model-offload-tiers; the catalog's "+
				"storage tiers are in catalog/devices/storage.yaml)", tier, c, coefficient.OffloadTierReadRate(tier),
				coefficient.OffloadTierWriteRate(tier), coefficient.OffloadTierBaseLatency(tier))
		}
	}
}

// Peer endpoints: where each instance sits, for the engines that send it KV (inference-sim/inference-sim#2439). They are
// here, with every other use of the vLLM engine, because only this file reaches it.

// peerEndpoint is where each instance's workers sit, in the numbering of its own pool, for an
// engine that sends it KV directly (engine.PeerPlaced; a p2p tier's WRITE ends there). A
// disaggregated run's instances are where instanceEndpoint puts them. A colocated run's instance n
// is the pool's n-th group, as the run's endpoint topology places it (build.EndpointGroupPlacements):
// the node holding the group's first GPU, counted from the pool's lowest node and rack
// (poolOrigin). Two instances on one node get one endpoint, and a WRITE between them is a move
// within the node. Nil when no kernel is open.
func peerEndpoint(pd *kernelPools, colocated *kernelmodel.Model) func(cluster.InstanceID) plan.Endpoint {
	if pd != nil {
		return pd.instanceEndpoint()
	}
	if colocated == nil {
		return nil
	}
	origin := poolOrigin(colocated)
	return func(id cluster.InstanceID) plan.Endpoint {
		n, err := strconv.Atoi(strings.TrimPrefix(string(id), "instance_"))
		if err != nil || n < 0 || n >= colocated.GroupCapacity() {
			panic(fmt.Sprintf("peer endpoint: instance %q is not one of the pool's %d groups", id, colocated.GroupCapacity()))
		}
		at := colocated.GroupPlacementOf(n)
		local, err := origin.Local(plan.Endpoint{Node: at.Node, Rack: at.Rack})
		if err != nil {
			panic(fmt.Sprintf("peer endpoint: instance %q: %v", id, err))
		}
		return local
	}
}

// placePeerInstance states where a colocated run's instance sits when its engine exchanges KV with
// peers (engine.Peered with an address): such an engine states its WRITEs from its own endpoint, so
// left as built every instance would state the pool's first node. Other colocated engines are left as
// built (placeInstance), and a disaggregated run's are placed there.
func placePeerInstance(eng engine.Engine, colocated *kernelmodel.Model, index int) error {
	p, ok := eng.(engine.Peered)
	if !ok || p.PeerAddr() == "" || colocated == nil {
		return nil
	}
	at := peerEndpoint(nil, colocated)(cluster.InstanceID(fmt.Sprintf("instance_%d", index)))
	if err := engine.Place(eng, at, colocated.EnginePlacement().Nodes); err != nil {
		return fmt.Errorf("instance_%d: %w", index, err)
	}
	return nil
}
