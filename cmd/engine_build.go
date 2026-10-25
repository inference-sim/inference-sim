package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/build"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim"
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
