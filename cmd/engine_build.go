package cmd

import (
	"fmt"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/build"
	"github.com/inference-sim/blis-vllm/compat/contract"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	coreengine "github.com/inference-sim/blis-vllm/vllmengine"
	// Linking these registers the compatibility engine's constructors: compat/kv the KV store
	// sim.NewEngine builds (sim.NewKVStoreFromConfig), compat/lora the adapter registry
	// (sim.NewAdapterRegistryFunc). The build site is the one place that links them.
	_ "github.com/inference-sim/blis-vllm/compat/kv"
	_ "github.com/inference-sim/blis-vllm/compat/lora"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// The one place inference-sim builds blis-vllm (#2021 B2). The engine's configuration -- the
// scenario's engine-side blocks and the command line's restatements of them -- is decoded and
// resolved by blis-vllm's compat/build, which cmd calls only from here, handing the inputs over
// unread. A refusal comes back as an error; the CLI boundary turns it into the fatal exit.

// resolveKVOffloadConfig resolves the run's KV offload: the scenario's offload block (#1910)
// and the --kv-offload-config file, which may only restate it (build.ResolveKVOffloadConfig).
// Single construction site (R4), shared by run and replay for INV-13 parity.
func resolveKVOffloadConfig(cmd *cobra.Command) sim.KVOffloadConfig {
	cfg, err := build.ResolveKVOffloadConfig(kernelScenario, scenarioOffload, kvOffloadConfigPath,
		blockSizeTokens, resolveCatalogRoot)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return cfg
}

// loadCatalogStorageDevices reads the catalog's storage-device table through the engine's own
// reader, so the catalog-load gate checks the file the way a run reads it (C6).
func loadCatalogStorageDevices(catalog string) (map[string]build.KVOffloadDevice, error) {
	return build.LoadCatalogStorageDevices(catalog)
}

// catalogStorageDevicesPath, catalogDevicesSubdir and catalogStorageDevicesFile locate the
// storage-device table inside a catalog clone root, for the catalog-load gate's diagnostics.
func catalogStorageDevicesPath(catalog string) string {
	return build.CatalogStorageDevicesPath(catalog)
}

const (
	catalogDevicesSubdir      = build.CatalogDevicesSubdir
	catalogStorageDevicesFile = build.CatalogStorageDevicesFile
)

// catalogStorageDevicesRelPath is the table's location relative to the catalog clone root,
// for diagnostics that name it.
const catalogStorageDevicesRelPath = build.CatalogStorageDevicesRelPath

// engineKnobFlags gathers the command line's restatements of the scenario's per-pool engine
// knobs, unread, for the engine to check against the scenario.
func engineKnobFlags() build.EngineKnobFlags {
	return build.EngineKnobFlags{
		// long_prefill_token_threshold has no flag (#2021 B6): every pool runs what it states.
		DPLoadBalancing: dpLoadBalancing, DPLoadBalancingFromFlag: dpLoadBalancingFromFlag,
	}
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
	fl := cmd.Flags()
	a, speculating, err := build.ResolveAcceptance(build.AcceptanceInputs{
		Scenario: kernelScenario, Shape: scenarioShape, Registry: kernelRegistry, Hardware: gpu,
		Workload: workloadAcceptance,
		Flags: build.AcceptanceFlags{
			Rate: speculativeAcceptance, Length: speculativeAcceptanceLength,
			RateSet: fl.Changed(build.AcceptanceRateFlag), LengthSet: fl.Changed(build.AcceptanceLengthFlag),
			Thinking: speculativeThinking, Category: speculativeCategory, FallbackCategory: speculativeFallbackCategory,
			ThinkingSet: fl.Changed(build.AcceptanceThinkingFlag), CategorySet: fl.Changed(build.AcceptanceCategoryFlag),
			FallbackSet: fl.Changed(build.AcceptanceFallbackFlag),
		},
	}, k, method)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	for _, f := range a.Ignored {
		logrus.Warnf("--%s is ignored: scenario %q does not speculate", f, kernelScenario)
	}
	return a, speculating
}

// The P/D pools (build.OpenPools and its pricing), as the run uses them.

// kernelPools is a disaggregated kernel run's per-role wiring: one kernel per pool.
type kernelPools struct {
	prefill, decode *kernelmodel.Model
	// connector is the scenario's pd_transfer.connector.
	connector string
}

// pools is p as the engine's pools (nil for a colocated run).
func (p *kernelPools) pools() *build.Pools {
	if p == nil {
		return nil
	}
	return &build.Pools{Prefill: p.prefill, Decode: p.decode, Connector: p.connector}
}

// runTopology is the run's instance counts as they stand (after DP-as-placement once applied).
func runTopology() build.Topology {
	return build.Topology{NumInstances: numInstances, Prefill: prefillInstances, Decode: decodeInstances,
		SharedOrEncode: prefillDecodeInstances > 0 || encodeInstances > 0}
}

// fatalf is the CLI's fatal exit, handed to the engine's pricing functions for a failure
// found while the run is under way.
func fatalf(format string, args ...any) { logrus.Fatalf(format, args...) }

// scenarioEndpointTopology uses the same role-specific kernel placements that price KV
// handoffs (build.EndpointPlacements), as the cluster's endpoint topology.
func scenarioEndpointTopology(model *kernelmodel.Model, pools *kernelPools, total, prefill, decode int) (map[cluster.InstanceID]cluster.EndpointTopology, error) {
	placed, err := build.EndpointPlacements(model, pools.pools(), total, prefill, decode)
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
func openKernelPools(catalogRoot string) *kernelPools {
	repos := kernelmodel.Repos{Scenarios: kernelScenarioDir, Catalog: catalogRoot, Registry: kernelRegistry}
	bp, warnings, err := build.OpenPools(kernelScenario, repos, kernelOpened != nil, runTopology())
	for _, w := range warnings {
		logrus.Warnf("%s", w)
	}
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	if bp == nil {
		return nil
	}
	p := &kernelPools{prefill: bp.Prefill, decode: bp.Decode, connector: bp.Connector}
	kernelPDOpened = p
	return p
}

// holdsPrefillKV reports whether the run's prefill engines hold a finished request's KV for
// the decode pull. False for a colocated run (nil receiver).
func (p *kernelPools) holdsPrefillKV() bool {
	return p.pools().HoldsPrefillKV()
}

// handoffPricing returns the run's handoff price and, when contention is on, the demand behind
// it. Both nil for a colocated run, where contention is refused.
func (p *kernelPools) handoffPricing(contention bool) (
	func(int64, cluster.InstanceID, cluster.InstanceID) int64,
	func(int64, cluster.InstanceID, cluster.InstanceID) sim.Flow,
) {
	price, demand, err := p.pools().HandoffPricing(kernelScenario, contention, runTopology(), blockSizeTokens, fatalf)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	var priceFn func(int64, cluster.InstanceID, cluster.InstanceID) int64
	var demandFn func(int64, cluster.InstanceID, cluster.InstanceID) sim.Flow
	if price != nil {
		priceFn = func(tokens int64, from, to cluster.InstanceID) int64 { return price(tokens, string(from), string(to)) }
	}
	if demand != nil {
		demandFn = func(tokens int64, from, to cluster.InstanceID) sim.Flow {
			return demand(tokens, string(from), string(to))
		}
	}
	return priceFn, demandFn
}

// transferTime is the handoff price alone, between the placements of two instances in their
// pools (the price half of handoffPricing).
func (p *kernelPools) transferTime() func(int64, cluster.InstanceID, cluster.InstanceID) int64 {
	price, _ := p.handoffPricing(false)
	return price
}

// defaultDPLoadBalancing is vLLM's default DP load-balancing mode, the --dp-load-balancing
// flag's default (sim.DPLoadBalancingInternal, from blis-vllm compat/dplb).
const defaultDPLoadBalancing = sim.DPLoadBalancingInternal

// requireDPLoadBalancingMode refuses a --dp-load-balancing value that names no vLLM mode
// (sim.ValidDPLoadBalancing).
func requireDPLoadBalancingMode(mode string) {
	if !sim.ValidDPLoadBalancing(mode) {
		logrus.Fatalf("--dp-load-balancing must be %s, %s or %s, got %q", sim.DPLoadBalancingInternal,
			sim.DPLoadBalancingHybrid, sim.DPLoadBalancingExternal, mode)
	}
}

// adoptFirstPoolEngine takes the run's engine from the first pool's kernel
// (build.FirstPoolSettingsOf): the caps, pages and pool the kernel priced, so the simulated
// scheduler admits against them. Per data-parallel RANK: a dp>1 MoE deployment runs one replica
// per rank (DP-as-placement), each sized as one EngineCore, so the placement never divides it.
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

// prefixHitOf is how the pool's KV cache groups cut a prefix hit (build.PrefixHitOf), with the
// run's --swa-bounded-replay override. It records the resolution for the results file and logs
// when the model's layout is not one BLIS maps.
func prefixHitOf(what string, m *engineModel) sim.KVPrefixHit {
	r, warning := build.PrefixHitOf(what, m, swaBoundedReplay.v)
	if warning != "" {
		logrus.Warnf("%s", warning)
	}
	if prefixCacheRecords == nil {
		prefixCacheRecords = map[string]sim.PrefixCacheRecord{}
	}
	prefixCacheRecords[string(m.Role())] = sim.PrefixCacheRecord{
		ModelRunner: r.Runner, ModelRunnerForcedBy: r.RunnerForcedBy,
		SWABoundedReplay: r.BoundedReplay, SWABoundedReplayFrom: r.BoundedReplayFrom,
		Layout: r.Hit, Note: r.Note,
	}
	return r.Hit
}

// overrides is each P/D role's engine, from its own pool's kernel and scenario pool.
func (p *kernelPools) overrides() (prefill, decode cluster.PoolOverrides) {
	return poolOverridesOf(p.prefill), poolOverridesOf(p.decode)
}

// poolOverridesOf is a P/D pool's engine as the cluster takes it: the settings blis-vllm
// resolves for the pool (build.PoolSettingsOf), copied into the cluster's per-pool overrides.
// The pool runs its own engine's scheduling policy unless a flag or policy bundle chose one for
// the run, which then applies to every pool and is recorded where it departs.
func poolOverridesOf(m *engineModel) cluster.PoolOverrides {
	hit := prefixHitOf(string(m.Role())+" pool", m)
	ps, err := build.PoolSettingsOf(kernelScenario, m, loraReservedBytesForKV, hit, engineKnobFlags())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	for _, note := range ps.Notes {
		logrus.Infof("%s", note)
	}
	o := cluster.PoolOverrides{
		TP: &ps.TP, GPU: ps.GPU, LatencyModel: m, TotalKVBlocks: &ps.TotalKVBlocks, MaxModelLen: &ps.MaxModelLen,
		MaxNumSeqs: &ps.MaxNumSeqs, MaxNumBatchedTokens: &ps.MaxNumBatchedTokens,
		PrefixCachingDisabled: &ps.PrefixCachingDisabled, SequenceState: &ps.SequenceState, PrefixHit: &ps.PrefixHit,
		LongPrefillTokenThreshold: &ps.LongPrefillTokenThreshold, DPLoadBalancing: ps.DPLoadBalancing,
	}
	if schedulerSource != "" {
		recordPolicyOverride("scheduler", string(m.Role()), ps.Policy.Scheduler, scheduler, schedulerSource)
	} else {
		o.Scheduler = ps.Policy.Scheduler
	}
	if preemptionSource != "" {
		recordPolicyOverride("preemption_policy", string(m.Role()), ps.Policy.Preemption, preemptionPolicy, preemptionSource)
	} else {
		o.PreemptionPolicy = ps.Policy.Preemption
	}
	return o
}

// rankInPool is an instance's rank within its pool (build.RankInPool).
func rankInPool(id cluster.InstanceID, firstDecode int) int {
	return build.RankInPool(string(id), firstDecode)
}

// requireKernelCapacity refuses a run whose instances do not fit the scenario's pools
// (build.RequireCapacity). Called once the instance counts are final.
func requireKernelCapacity(p *kernelPools) {
	if err := build.RequireCapacity(kernelScenario, kernelOpened, p.pools(), runTopology()); err != nil {
		logrus.Fatalf("%v", err)
	}
}

// offloadPricing is what the engine's offload pricing reads from the run.
func offloadPricing() build.OffloadPricing {
	return build.OffloadPricing{BlockSizeTokens: blockSizeTokens, KVCPUBlocks: kvCPUBlocks,
		MaxNumBatchedTokens: maxNumBatchedTokens}
}

// sizeOffloadBlocks records the kernel's per-rank bytes of one GPU block on an enabled offload
// config and refuses a CPU tier smaller than one chunk (build.SizeOffloadBlocks). keepStated
// keeps the bytes a replayed trace header carries (INV-13).
func sizeOffloadBlocks(cfg *sim.KVOffloadConfig, keepStated bool) {
	if err := build.SizeOffloadBlocks(kernelOpened, cfg, blockSizeTokens, keepStated); err != nil {
		logrus.Fatalf("%v", err)
	}
}

// applyKernelOffloadPricing has the kernel price every KV-offload transfer of a colocated
// run (build.PriceColocatedOffload). It returns the legacy per-block charge. Shared by run and
// replay.
func applyKernelOffloadPricing(cfg *sim.KVOffloadConfig) int64 {
	ticks, err := build.PriceColocatedOffload(kernelOpened, cfg, offloadPricing(), fatalf)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return ticks
}

// applyOffload prices the run's KV offload once per pool, each by that pool's own kernel
// (build.Pools.PriceOffload), and records it on the pool's overrides. No-op when the run
// offloads nothing.
func (p *kernelPools) applyOffload(cfg sim.KVOffloadConfig, prefill, decode *cluster.PoolOverrides) {
	pre, dec, ok, err := p.pools().PriceOffload(kernelScenario, cfg, offloadPricing(), runTopology(), fatalf)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	if !ok {
		return
	}
	prefill.KVOffload, prefill.KVTransferTicksPerBlock = &pre.Config, &pre.TicksPerBlock
	decode.KVOffload, decode.KVTransferTicksPerBlock = &dec.Config, &dec.TicksPerBlock
}

// adoptInputs are the scenario and roots the command line names, handed over unread, with the
// routing overrides applied just before the first pool opens (as before the move).
func adoptInputs(cmd *cobra.Command) build.AdoptInputs {
	return build.AdoptInputs{
		Scenario: kernelScenario, ScenariosDir: kernelScenarioDir, Registry: kernelRegistry,
		Catalog:    resolveCatalogRoot,
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
func openModel(catalogRoot string) (*kernelmodel.Model, error) {
	return build.OpenModel(kernelScenario, kernelScenarioDir, catalogRoot, kernelRegistry)
}

// expertModelConfig is the engine's model configuration as the scenario's graph states it: its
// expert geometry only (routed experts and experts per token, 0 for a dense model). Nothing
// else of ModelConfig is populated: no HF config is parsed, and a half-filled struct read as
// complete is how a dense model comes to be treated as MoE.
func expertModelConfig() sim.ModelConfig {
	var c sim.ModelConfig
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
// deployment shape, and the routing overrides handed to the pools.
type (
	engineModel      = kernelmodel.Model
	scenarioShapeT   = kernelmodel.Shape
	routingOverrides = kernelmodel.RoutingOverrides
)

// setRoutingOverrides hands the command line's routing layer to the pools about to open.
func setRoutingOverrides(o routingOverrides) { kernelmodel.SetRoutingOverrides(o) }

// currentRoutingOverridesOf is the routing layer the pools will open with.
func currentRoutingOverridesOf() routingOverrides { return kernelmodel.CurrentRoutingOverrides() }

// checkPolicyNames refuses a scheduler or preemption policy the engine does not run
// (build.CheckPolicyNames), at the CLI boundary.
func checkPolicyNames(scheduler, preemption string) {
	if err := build.CheckPolicyNames(scheduler, preemption); err != nil {
		logrus.Fatalf("%v", err)
	}
}

// resolveLoRAConfig assembles the run's LoRA configuration (build.ResolveLoRAConfig) from
// defaults.yaml's cost coefficients, the optional --lora-config file and the scalar --lora-*
// flags (each only when Changed, R18), and validates it. The single LoRAConfig construction site
// (R4), called by both run and replay (INV-13); fatal on an invalid config (Principle V).
func resolveLoRAConfig(cmd *cobra.Command) sim.LoRAConfig {
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
func batchConfigFromCLI() sim.BatchConfig {
	return build.BatchConfigOf(engineSettings(sim.ModelConfig{}, 1, sim.KVOffloadConfig{}, 0, sim.LoRAConfig{},
		sim.SpeculativeConfig{}))
}

// engineConfigFor is the engine's configuration for this run or replay (build.EngineConfigOf),
// from the settings the scenario, the kernel and the command line resolved: the one place both
// commands assemble it (INV-13). perRankDP is DP-as-placement's per-replica DP (#1531, #1556).
// It resolves the speculative configuration, which records the acceptance for the results file.
func engineConfigFor(cmd *cobra.Command, m sim.ModelConfig, perRankDP int, offload sim.KVOffloadConfig,
	cpuTierTicks int64, lora sim.LoRAConfig) build.EngineConfig {
	return build.EngineConfigOf(engineSettings(m, perRankDP, offload, cpuTierTicks, lora, resolveSpeculativeConfig(cmd)))
}

// simConfigFor is the instance configuration both commands hand the cluster: the engine's
// configuration (ec, from engineConfigFor) and its front end, beside the simulator's own settings
// (the horizon, the seed, the latency model, the SLO priority overrides and the server series).
// The one place cmd assembles it (INV-13).
func simConfigFor(ec build.EngineConfig, horizon, seed int64, latency sim.LatencyModel, sloOverrides map[string]int) sim.SimConfig {
	return sim.SimConfig{
		Horizon: horizon,
		Seed:    seed,
		// The engine's configuration comes from the build site (engineConfigFor), the same for
		// run and replay (INV-13). DP-as-placement (#1531): the per-replica DP is in
		// ec.ModelHardware, from the same resolveDPPlacement for both (#1556).
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

// engineSettings gathers the run's resolved engine settings, unread, for build.EngineConfigOf.
func engineSettings(m sim.ModelConfig, perRankDP int, offload sim.KVOffloadConfig, cpuTierTicks int64,
	lora sim.LoRAConfig, spec sim.SpeculativeConfig) build.EngineSettings {
	return build.EngineSettings{
		TotalKVBlocks: totalKVBlocks, BlockSizeTokens: blockSizeTokens, KVCPUBlocks: kvCPUBlocks,
		KVOffloadThreshold: kvOffloadThreshold, KVTransferBandwidth: kvTransferBandwidth,
		KVTransferBaseLatency: kvTransferBaseLatency, KVOffload: offload, KVTransferTicksPerBlock: cpuTierTicks,
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
func speculativeConfigOf(k int, acceptance float64, method string) sim.SpeculativeConfig {
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
func adapterReservedBytesFor(cfg sim.LoRAConfig) int64 {
	n, err := build.AdapterReservedBytes(cfg)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return n
}

// adapterRegistryOf is the declared adapters' registry (build.AdapterRegistryOf), nil when none
// are declared, for checking the workload's adapter references; fatal on an invalid registry.
func adapterRegistryOf(cfg sim.LoRAConfig) sim.AdapterRegistry {
	r, err := build.AdapterRegistryOf(cfg)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return r
}

// offloadRecord and offloadTierRecord are a trace header's record of the KV-offload config,
// in the engine's terms (build.OffloadRecord).
type (
	offloadRecord     = build.OffloadRecord
	offloadTierRecord = build.OffloadTierRecord
)

// kvOffloadOfRecord is the engine's offload config rebuilt from a trace's record of it
// (build.KVOffloadOfRecord); nil is the inert config.
func kvOffloadOfRecord(r *offloadRecord) sim.KVOffloadConfig { return build.KVOffloadOfRecord(r) }

// reconcileReplayKVOffload is replay's KV-offload config: the trace header's, authoritative
// (BC-G6), reconciled with the scenario's offload block and --kv-offload-config, which may
// only restate it (build.ReconcileReplayKVOffload); fatal on a conflict or an unreproducible
// record. allowFlagAdd is for an observe trace or an agentic replay, which record no BLIS
// deployment (replayRecordsDeployment).
func reconcileReplayKVOffload(cmd *cobra.Command, headerBlock *workload.TraceKVOffloadConfig, allowFlagAdd bool) sim.KVOffloadConfig {
	// The scenario's offload block is a source on replay as on run: it and the flag resolve
	// together (resolveKVOffloadConfig), and the result must match the header.
	flagChanged := cmd.Flags().Changed("kv-offload-config") || scenarioOffload != nil
	var flagCfg sim.KVOffloadConfig
	if flagChanged {
		// resolveKVOffloadConfig itself logrus.Fatalf's on a malformed flag file.
		flagCfg = resolveKVOffloadConfig(cmd)
	}
	cfg, err := resolveReplayKVOffload(headerBlock, flagChanged, flagCfg, allowFlagAdd)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return cfg
}

// resolveReplayKVOffload is reconcileReplayKVOffload's error-returning core, over the trace
// header's record (build.ReconcileReplayKVOffload).
func resolveReplayKVOffload(headerBlock *workload.TraceKVOffloadConfig, flagChanged bool, flagCfg sim.KVOffloadConfig, allowFlagAdd bool) (sim.KVOffloadConfig, error) {
	return build.ReconcileReplayKVOffload(headerToSimOffload(headerBlock), flagChanged, flagCfg, allowFlagAdd)
}

// requireOffloadForTieredObservation refuses a replay whose trace's observed KV hit-rate came
// from the tiered offload counters but which reproduces no offload config (BC-10, #1583;
// build.RequireOffloadForTieredObservation).
func requireOffloadForTieredObservation(obs *workload.TraceObservedKVMetrics, offloadEnabled bool) error {
	if obs == nil {
		return nil
	}
	return build.RequireOffloadForTieredObservation(obs.Source == workload.ObservedKVSourceTiered, obs.Source, offloadEnabled)
}

// requireOneOffloadModel refuses the offload chain together with --kv-cpu-blocks (#1590 H1;
// build.RequireOneOffloadModel), fatal at the CLI boundary.
func requireOneOffloadModel(offloadEnabled bool, onReplay bool) {
	if err := build.RequireOneOffloadModel(offloadEnabled, kvCPUBlocks, onReplay); err != nil {
		logrus.Fatalf("%v", err)
	}
}

// frontendConfigFromCLI is the engine's front-end config from --renderer-num-workers and
// --api-server-count (build.FrontendConfigOf), the single run/replay seam (INV-13); fatal on
// a negative count.
func frontendConfigFromCLI() sim.FrontendConfig {
	c, err := build.FrontendConfigOf(rendererNumWorkers, apiServerCount)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return c
}

// groupFrontendConfigFromCLI is frontendConfigFromCLI for one endpoint whose engine is a whole
// data-parallel group of dp ranks (a contract-built release): vLLM's internal balancer defaults
// its API servers to one per rank (api_server_count = data_parallel_size,
// vllm/entrypoints/cli/serve.py:123-130@5df63a72), as the compatibility path's DP replica
// renders (sim.ReplicaAPIServers). A stated --api-server-count is kept.
func groupFrontendConfigFromCLI(dp int) sim.FrontendConfig {
	c := frontendConfigFromCLI()
	c.APIServerCount = sim.ReplicaAPIServers(c.APIServerCount, max(dp, 1))
	return c
}

// dpLockstepRanks is how many replicas of one logical instance form one vLLM DP group that
// steps in lockstep (build.DPLockstepRanks): dp for an MoE model at dp > 1.
func dpLockstepRanks(isMoE bool, dp int) int { return build.DPLockstepRanks(isMoE, dp) }

// checkHybridDPNodeLocal refuses --dp-load-balancing hybrid for a colocated replica whose ranks
// span nodes (build.CheckHybridDPNodeLocal). Only a colocated run with a kernel is checked: the
// kernel places the ranks.
func checkHybridDPNodeLocal(plan dpPlacementPlan) error {
	if !plan.Active || kernelOpened == nil ||
		prefillInstances > 0 || decodeInstances > 0 || prefillDecodeInstances > 0 || encodeInstances > 0 {
		return nil
	}
	return build.CheckHybridDPNodeLocal(dpLoadBalancing, plan.Replicas, numInstances,
		func(rank int) int { return kernelOpened.PlacementOf(rank).Node })
}

// compatContractFactory is the cluster's engine constructor for a compatibility release run
// through the engine contract under ContractCompatEnv (phase 2c's first stage): the
// compatibility engine behind engine.Engine (blis-vllm compat/contract), built from the run's
// instance configuration and driven by a sim.EngineDriver. nil otherwise, so the run is
// unchanged.
func compatContractFactory(version string, cfg sim.SimConfig, acceptance float64) func(int) (*sim.EngineDriver, error) {
	if !compatThroughContract(version) || cfg.Offload.IsEnabled() || cfg.HasAdapters() {
		// The first stage's adapter carries neither an offload tier nor LoRA, however the run
		// came by them (a replay takes its offload tier from its trace's header).
		return nil
	}
	build := func(index int) (*sim.EngineDriver, error) {
		eng, err := contract.New(fmt.Sprintf("instance_%d", index), cfg)
		if err != nil {
			return nil, err
		}
		dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon, Acceptance: acceptance, ResumeGapFromLastToken: true,
			ScrapeEngineMetrics: true}
		if cfg.LatencyModel != nil {
			dcfg.PostDecodeFixedOverhead = cfg.LatencyModel.PostDecodeFixedOverhead()
			dcfg.OutputTokenProcessingTime = cfg.LatencyModel.OutputTokenProcessingTime()
			dcfg.QueueingTime = cfg.LatencyModel.QueueingTime
		}
		return sim.NewEngineDriverWithFrontend(eng, dcfg, cfg.Frontend)
	}
	if _, err := build(0); err != nil {
		logrus.Fatalf("%v", err)
	}
	return build
}

// registerContractEngines registers the engine kind a contract-built release is built by:
// blis-vllm's translated v0.32.0 engine, with the latency kernel's step pricing and KV sizing
// (its Register). It is the one place this binary links an engine the contract reaches;
// later calls are no-ops.
func registerContractEngines() { coreengine.Register() }

// contractRoots are the roots the registered factory reads: the catalog and registry clone
// roots, and the scenario file's path, which it opens itself.
func contractRoots(catalog, registry, scenarioPath string) map[string]string {
	return map[string]string{coreengine.RootCatalog: catalog, coreengine.RootRegistry: registry,
		coreengine.RootScenario: scenarioPath}
}
