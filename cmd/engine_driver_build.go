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

// engineDriverFactory returns the cluster's engine constructor for a contract-built release,
// nil for one the compatibility engine runs. Each instance's engine is built from the
// scenario's first pool and driven by a sim.EngineDriver; the host overheads (queueing,
// per-token output processing, post-decode) are the latency model's, as on the compatibility
// path.
func engineDriverFactory(version string, horizon, seed int64, latency sim.LatencyModel) func(int) (*sim.EngineDriver, error) {
	if !runsThroughContract(version) {
		return nil
	}
	// The contract path builds one engine per instance from the scenario's single colocated
	// pool. A disaggregated topology or a data-parallel replica needs the cluster to drive P/D
	// handoffs or lockstep waves through the contract, which it does not yet: refuse it, naming
	// the piece, rather than run it on the compatibility engine.
	if prefillInstances != 0 || decodeInstances != 0 || prefillDecodeInstances != 0 {
		logrus.Fatalf("scenario %q: engine_version %s runs through the engine contract, which does not "+
			"drive a P/D topology yet (P/D disaggregation)", kernelScenario, version)
	}
	if dataParallelism > 1 {
		logrus.Fatalf("scenario %q: engine_version %s runs through the engine contract, which does not "+
			"drive a data-parallel replica yet (data parallelism, dp %d)", kernelScenario, version, dataParallelism)
	}
	registerContractEngines()
	scenarioPath := filepath.Join(kernelScenarioDir, kernelScenario)
	return func(index int) (*sim.EngineDriver, error) {
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
		cfg := sim.EngineDriverConfig{Horizon: horizon}
		if latency != nil {
			cfg.PostDecodeFixedOverhead = latency.PostDecodeFixedOverhead()
			cfg.OutputTokenProcessingTime = latency.OutputTokenProcessingTime()
			cfg.QueueingTime = latency.QueueingTime
		}
		return sim.NewEngineDriver(eng, cfg), nil
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
