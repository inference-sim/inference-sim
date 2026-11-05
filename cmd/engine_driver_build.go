package cmd

import (
	"fmt"
	"path/filepath"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/blis-substrate/contracts/engine"

	"github.com/inference-sim/inference-sim/sim"
)

// The one build site of the engine (#2021 B2, B3): every run builds the engine the scenario
// selects (selectEngine, engine_select.go), which this binary reaches only through
// blis-schemas' engine.Engine contract. The engine is built by the factory registered for its
// kind (registerContractEngines, in engine_build.go) from the scenario file, unopened: the
// factory reads the pool's engine block, sizes the KV cache and prices every step with the
// latency kernel itself.

// engineDriverFactory returns the cluster's engine constructor. Each instance's engine is built
// from the scenario's first pool and driven by a sim.EngineDriver; the host overheads (queueing,
// per-token output processing, post-decode) are the latency model's, and the API-server front
// end (--renderer-num-workers, --api-server-count) is the run's.
func engineDriverFactory(ref engineRef, horizon, seed int64, latency latencyModel, acceptance float64) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	version := ref.String()
	// The contract path builds one engine per instance from its pool: the colocated pool, or
	// for a disaggregated topology the prefill or decode pool the instance's role takes (each
	// pool's kernel is its role's latency model, rolesOf). The engine runs NIXL as that pool's
	// kv_role, and the cluster drives the handoffs through the contract (sim.EngineDriver's
	// AddRemote, LandRemote and ReleaseHandoff). A shared prefill-decode pod has no pool of its
	// own, which blis-vllm's OpenPools refuses already. A data-parallel deployment is one
	// instance whose engine is the whole DP group, its ranks, waves and balancer the engine's.
	poolOf := kernelPDOpened.poolOfInstance()
	// The DP group is vLLM's internal balancer, one endpoint routing to its ranks. Under the
	// external balancer each rank is its own endpoint and the router picks the rank
	// (vllm/entrypoints/cli/serve.py:107-116@5df63a72), which the group does not model: refuse
	// it, naming the mode, rather than run it as internal.
	if dataParallelism > 1 && dpLoadBalancing == sim.DPLoadBalancingExternal {
		logrus.Fatalf("scenario %q: engine %s runs a data-parallel group through the engine contract "+
			"behind vLLM's internal balancer, which does not drive --dp-load-balancing external yet "+
			"(one endpoint per rank, the router picking the rank)", kernelScenario, version)
	}
	// The host does not carry request priority across the engine contract yet: the driver sends
	// every engine.Request at priority 0 (sim.EngineDriver, the SLO class only as a tag the engine
	// does not read), so a priority policy would order by arrival alone and silently run as FCFS.
	// Refuse the scenario's stated policy (vLLM's priority, BLIS's priority-fcfs: enginePolicyOf),
	// any SLO priority override, and a priority scheduler or preemption policy the run asks for
	// (--scheduler, --preemption-policy or a --policy-config bundle, none of which reaches the
	// engine), naming the setting, rather than run any as FCFS.
	refusePriority := func(setting string) {
		logrus.Fatalf("scenario %q: engine %s runs through the engine contract, and the host does not "+
			"carry request priority across it yet (every request reaches the engine at priority 0), so %s",
			kernelScenario, version, setting)
	}
	if scenarioScheduler == "priority-fcfs" {
		refusePriority("engine.scheduling_policy priority would run as FCFS")
	}
	if len(sloPriorityOverrides) > 0 {
		refusePriority("the --policy-config priority.slo_priorities override would be ignored")
	}
	if scheduler == "priority-fcfs" || scheduler == "reverse-priority" || preemptionPolicy == "priority" {
		refusePriority(fmt.Sprintf("scheduler %s with preemption policy %s would run as FCFS", scheduler, preemptionPolicy))
	}
	registerContractEngines()
	scenarioPath := filepath.Join(kernelScenarioDir, kernelScenario)
	roots, err := contractRoots(dataOptions(), scenarioPath)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	frontend := groupFrontendConfigFromCLI(dataParallelism)
	build := func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		// The Spec carries no configuration: the factory reads the pool's engine block from the
		// scenario file itself, so this binary opens no engine configuration.
		eng, err := engine.New(engine.Spec{
			Kind:            ref.Kind,
			UpstreamVersion: ref.Version,
			Roots:           roots,
			Pool:            poolOf(ic),
			Seed:            seed,
		})
		if err != nil {
			return nil, fmt.Errorf("scenario %q: engine %s: %w", kernelScenario, version, err)
		}
		reportAssumptions(eng)
		if iv := serverSeriesIntervalUs(); iv > 0 {
			// --server-series-interval-s: the engine bins its own vllm:* series on the simulation
			// clock (the driver's ServerSeriesSpan and ServerSeriesBlock queries read it). An
			// engine that cannot record one is refused rather than leave the flag silently off.
			s, ok := eng.(interface{ EnableServerSeries(intervalUs int64) })
			if !ok {
				return nil, fmt.Errorf("scenario %q: engine %s: the engine records no server time series, "+
					"which --server-series-interval-s asks for", kernelScenario, version)
			}
			s.EnableServerSeries(iv)
		}
		cfg := sim.EngineDriverConfig{Horizon: horizon, Acceptance: acceptance}
		m := latency
		if kernelPDOpened != nil {
			// Each P/D pool's host overheads are its own kernel's.
			m = engineConfigIn(ic).LatencyModel
		}
		hostOverheads(&cfg, m)
		return sim.NewEngineDriverWithFrontend(eng, cfg, frontend)
	}
	// Build the first instance's engine now: a deployment the engine refuses (one needing a
	// piece it does not build yet) is a CLI refusal, named, before the cluster is assembled.
	if kernelPDOpened != nil {
		// Each pool's engine is built with its instances, from the role's configuration.
		return build
	}
	first, err := build(0, sim.InstanceConfig{})
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		if index == 0 && first != nil {
			d := first
			first = nil
			return d, nil
		}
		return build(index, ic)
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
