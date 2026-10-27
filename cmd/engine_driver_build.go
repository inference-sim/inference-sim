package cmd

import (
	"fmt"
	"path/filepath"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/blis-schemas/engine"

	"github.com/inference-sim/inference-sim/sim"
)

// The 0.32.0 half of the one build site (#2021 B2, B3; GOAL.md): a scenario whose
// engine_version the compatibility engine does not model runs blis-vllm's translated vLLM
// v0.32.0 engine, which this binary reaches only through blis-schemas' engine.Engine contract.
// The engine is built by the factory registered for its kind (registerContractEngines, in
// engine_build.go) from the scenario file, unopened: the factory reads the pool's engine block,
// sizes the KV cache and prices every step with the latency kernel itself.

// coreEngineVersions are the releases that run through engine.Engine rather than the
// compatibility engine.
var coreEngineVersions = map[string]bool{"0.32.0": true}

// runsThroughContract reports whether the scenario's engine_version is one the engine.Engine
// path builds.
func runsThroughContract(version string) bool { return coreEngineVersions[version] }

// compatEngineVersions are the releases the compatibility engine models. Every run of one is
// built by compatContractFactory: the compatibility engine behind engine.Engine (blis-vllm
// compat/contract), driven by sim.EngineDriver, as a 0.32.0 run's engine is (GOAL.md phase 2c).
// Shared prefill-decode and encode pools are refused before any engine is built
// (compat/build.OpenPools), since no scenario describes the engine they would run.
var compatEngineVersions = map[string]bool{"0.29.0": true, "0.31.0": true}

// dpPlacementFor is the run's DP placement plan. A contract-built release places no rank
// replicas: the engine behind the contract is the whole data-parallel group, one endpoint
// whose ranks step in lockstep inside it, so the instance count stays the scenario's and each
// instance keeps the scenario's dp.
func dpPlacementFor(version string, plan dpPlacementPlan) dpPlacementPlan {
	if !runsThroughContract(version) {
		return plan
	}
	return dpPlacementPlan{Active: false, Replicas: 1, PerRankDP: dataParallelism}
}

// engineDriverFactory returns the cluster's engine constructor for a contract-built release,
// nil for one the compatibility engine runs. Each instance's engine is built from the
// scenario's first pool and driven by a sim.EngineDriver; the host overheads (queueing,
// per-token output processing, post-decode) are the latency model's, and the API-server front
// end (--renderer-num-workers, --api-server-count) is the run's, as on the compatibility path.
func engineDriverFactory(version string, horizon, seed int64, latency latencyModel, acceptance float64) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	if !runsThroughContract(version) {
		// A compatibility release: built by compatContractFactory, at the build site, from the
		// run's own instance configuration.
		return nil
	}
	// The contract path builds one engine per instance from the scenario's single colocated
	// pool. A disaggregated topology needs the cluster to drive P/D handoffs through the
	// contract, which it does not yet: refuse it, naming the piece, rather than run it on the
	// compatibility engine. A data-parallel replica is one instance whose engine is the whole DP
	// group, its ranks, waves and balancer the engine's (dpPlacementFor).
	if prefillInstances != 0 || decodeInstances != 0 || prefillDecodeInstances != 0 {
		logrus.Fatalf("scenario %q: engine_version %s runs through the engine contract, which does not "+
			"drive a P/D topology yet (P/D disaggregation)", kernelScenario, version)
	}
	// The DP group is vLLM's internal balancer, one endpoint routing to its ranks. Under the
	// external balancer each rank is its own endpoint and the router picks the rank
	// (vllm/entrypoints/cli/serve.py:107-116@5df63a72), which the group does not model: refuse
	// it, naming the mode, rather than run it as internal.
	if dataParallelism > 1 && dpLoadBalancing == sim.DPLoadBalancingExternal {
		logrus.Fatalf("scenario %q: engine_version %s runs a data-parallel group through the engine contract "+
			"behind vLLM's internal balancer, which does not drive --dp-load-balancing external yet "+
			"(one endpoint per rank, the router picking the rank)", kernelScenario, version)
	}
	registerContractEngines()
	scenarioPath := filepath.Join(kernelScenarioDir, kernelScenario)
	frontend := groupFrontendConfigFromCLI(dataParallelism)
	build := func(index int) (*sim.EngineDriver, error) {
		kinds := engine.Kinds()
		if len(kinds) != 1 {
			return nil, fmt.Errorf("scenario %q: engine_version %s: the build site registers one engine kind, "+
				"and the registry holds %v", kernelScenario, version, kinds)
		}
		// The Spec carries no configuration: the factory reads the pool's engine block from the
		// scenario file itself, so this binary opens no engine configuration.
		eng, err := engine.New(engine.Spec{
			Kind:            kinds[0],
			UpstreamVersion: version,
			Roots:           contractRoots(resolvedCatalogRoot, kernelRegistry, scenarioPath),
			Pool:            0,
			Seed:            seed,
		})
		if err != nil {
			return nil, fmt.Errorf("scenario %q: engine_version %s: %w", kernelScenario, version, err)
		}
		reportAssumptions(eng)
		if iv := serverSeriesIntervalUs(); iv > 0 {
			// --server-series-interval-s: the engine bins its own vllm:* series on the simulation
			// clock (the driver's ServerSeriesSpan and ServerSeriesBlock queries read it). An
			// engine that cannot record one is refused rather than leave the flag silently off.
			s, ok := eng.(interface{ EnableServerSeries(intervalUs int64) })
			if !ok {
				return nil, fmt.Errorf("scenario %q: engine_version %s: the engine records no server time series, "+
					"which --server-series-interval-s asks for", kernelScenario, version)
			}
			s.EnableServerSeries(iv)
		}
		cfg := sim.EngineDriverConfig{Horizon: horizon, Acceptance: acceptance}
		hostOverheads(&cfg, latency)
		return sim.NewEngineDriverWithFrontend(eng, cfg, frontend)
	}
	// Build the first instance's engine now: a deployment the engine refuses (one needing a
	// piece it does not build yet) is a CLI refusal, named, before the cluster is assembled.
	first, err := build(0)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return func(index int, _ sim.InstanceConfig) (*sim.EngineDriver, error) {
		if index == 0 && first != nil {
			d := first
			first = nil
			return d, nil
		}
		return build(index)
	}
}

// reportAssumptions writes an engine's disclosed assumptions (for blis-vllm's engine, that the
// latency kernel's 0.31.0 tuning prices 0.32.0) to stderr once.
func reportAssumptions(eng engine.Engine) {
	if assumptionsReported {
		return
	}
	assumptionsReported = true
	if a, ok := eng.(interface{ Assumptions() []string }); ok {
		for _, s := range a.Assumptions() {
			logrus.Warnf("engine assumption: %s", s)
		}
	}
}

var assumptionsReported bool
