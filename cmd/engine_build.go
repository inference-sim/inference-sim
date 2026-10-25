package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/build"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
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
var catalogStorageDevicesRelPath = build.CatalogStorageDevicesRelPath

// engineKnobFlags gathers the command line's restatements of the scenario's per-pool engine
// knobs, unread, for the engine to check against the scenario.
func engineKnobFlags() build.EngineKnobFlags {
	return build.EngineKnobFlags{
		LongPrefillTokenThreshold: longPrefillTokenThreshold, LongPrefillFromFlag: longPrefillFromFlag,
		DPLoadBalancing: dpLoadBalancing, DPLoadBalancingFromFlag: dpLoadBalancingFromFlag,
	}
}

// runEngineKnobs is the run-wide long-prefill threshold and DP load-balancing mode
// (build.RunEngineKnobs).
func runEngineKnobs(shape kernelmodel.Shape, f build.EngineKnobFlags) (int64, string, error) {
	return build.RunEngineKnobs(kernelScenario, shape, f)
}

// poolEngineKnobsOf is a P/D pool's own threshold and mode (build.PoolEngineKnobs).
func poolEngineKnobsOf(m *kernelmodel.Model, f build.EngineKnobFlags) (int64, string, error) {
	return build.PoolEngineKnobs(kernelScenario, m, f)
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
func openKernelPools(catalogRoot string) *kernelPools {
	repos := kernelmodel.Repos{Scenarios: kernelScenarioDir, Catalog: catalogRoot, Registry: kernelRegistry}
	bp, err := build.OpenPools(kernelScenario, repos, kernelOpened != nil, runTopology())
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	if bp == nil {
		return nil
	}
	for _, w := range bp.Warnings {
		logrus.Warnf("%s", w)
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

// logSequenceFixedCharge reports the per-sequence term of an engine's KV capacity (#1941).
func logSequenceFixedCharge(what string, s kernelmodel.Settings) {
	if note := build.SequenceFixedCharge(what, s); note != "" {
		logrus.Infof("%s", note)
	}
}

// requireWindowFits refuses an engine with no window at all (build.RequireWindowFits).
func requireWindowFits(what string, s kernelmodel.Settings) {
	if err := build.RequireWindowFits(kernelScenario, what, s); err != nil {
		logrus.Fatalf("%v", err)
	}
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
