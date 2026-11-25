package cmd

import (
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"

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
	// own, which the vLLM engine's OpenPools refuses already. A data-parallel deployment is one
	// instance whose engine is the whole DP group, its ranks, waves and balancer the engine's.
	poolOf := kernelPDOpened.poolOfInstance()
	// Under external DP load balancing each rank is its own router-visible endpoint and the router
	// picks the rank (vllm/entrypoints/cli/serve.py:107-116@5df63a72). The host builds the shared
	// group once per replica and exposes its ranks as per-rank endpoint views (dpRankViews over the
	// engine's dprankview, engine_build.go), one InstanceSimulator per rank; the sim leader/member
	// path (sim/cluster buildDPReplicas) steps the shared group and splits each step across the
	// views. Internal/unset builds the one collapsed instance, exactly as before.
	externalDP := dataParallelism > 1 && dpLoadBalancing == sim.DPLoadBalancingExternal
	// The host does not carry request priority across the engine contract yet: the driver sends
	// every engine.Request at priority 0 (sim.EngineDriver, the SLO class only as a tag the engine
	// does not read), so a priority policy would order by arrival alone and silently run as FCFS.
	// Refuse the scenario's stated policy (vLLM's priority, BLIS's priority-fcfs: enginePolicyOf)
	// and any SLO priority override, naming the setting, rather than run either as FCFS.
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
	registerContractEngines()
	scenarioPath := engineScenarioPath()
	// The run opens its catalog and registry once (runData) and every engine of the run reads
	// that Data, so the Result's hashes and its one selected-entry report cover the lookups of
	// every pool's engine (blis-vllm#90, design §3.3, VAL-6).
	data := mustRunData()
	// The run's file systems under the fs tiers' root_dirs, shared by every engine of the model
	// (#1698): one store for each (model, root_dir).
	storage := newTierStorage()
	// buildEngine builds one engine for the instance at index from its pool, placing it and setting
	// its front end, and returns the driver configuration its host overheads need. For an external
	// DP replica this builds the whole shared group once (at the replica's base index); the per-rank
	// views are wrapped from it below.
	buildEngine := func(index int, ic sim.InstanceConfig) (engine.Engine, sim.EngineDriverConfig, error) {
		m := latency
		accept := acceptance
		if kernelPDOpened != nil {
			// Each P/D pool's host overheads are its own kernel's, and its requests advance by
			// its own speculative acceptance (inference-sim#2125).
			m = engineConfigIn(ic).LatencyModel
			accept = engineConfigIn(ic).Acceptance
		}
		streamInterval, err := streamIntervalOf(m)
		if err != nil {
			return nil, sim.EngineDriverConfig{}, fmt.Errorf("scenario %q: %w", kernelScenario, err)
		}
		// The Spec carries no configuration: the factory reads the pool's engine block from the
		// scenario file itself, so this binary opens no engine configuration.
		eng, err := newContractEngine(engine.Spec{
			Kind:            ref.Kind,
			UpstreamVersion: ref.Version,
			Roots:           contractRoots(scenarioPath),
			Pool:            poolOf(ic),
			Seed:            seed,
		}, data)
		if err != nil {
			return nil, sim.EngineDriverConfig{}, fmt.Errorf("scenario %q: engine %s: %w", kernelScenario, version, err)
		}
		if err := placeInstance(eng, kernelPDOpened, index); err != nil {
			return nil, sim.EngineDriverConfig{}, fmt.Errorf("scenario %q: instance_%d: %w", kernelScenario, index, err)
		}
		if kernelPDOpened == nil {
			if err := placePeerInstance(eng, kernelOpened, index); err != nil {
				return nil, sim.EngineDriverConfig{}, fmt.Errorf("scenario %q: %w", kernelScenario, err)
			}
		}
		storage.share(eng, model)
		// --renderer-num-workers and --api-server-count are vLLM's API-server settings, which
		// blis-schemas does not carry yet; the engine resolves them into the admission queue its
		// plan states, and the driver queues requests on that shape.
		if err := setEngineFrontend(eng, version); err != nil {
			return nil, sim.EngineDriverConfig{}, fmt.Errorf("scenario %q: %w", kernelScenario, err)
		}
		reportAssumptions(eng)
		if iv := serverSeriesIntervalUs(); iv > 0 {
			// --server-series-interval-s: the engine bins its own vllm:* series on the simulation
			// clock (the driver's ServerSeriesSpan and ServerSeriesBlock queries read it). An
			// engine that cannot record one is refused rather than leave the flag silently off.
			s, ok := eng.(interface{ EnableServerSeries(intervalUs int64) })
			if !ok {
				return nil, sim.EngineDriverConfig{}, fmt.Errorf("scenario %q: engine %s: the engine records no server time series, "+
					"which --server-series-interval-s asks for", kernelScenario, version)
			}
			s.EnableServerSeries(iv)
		}
		cfg := sim.EngineDriverConfig{Horizon: horizon, Acceptance: accept, StreamInterval: streamInterval,
			Environment: kernelPDOpened.environmentOf(ic, kernelOpened)}
		hostOverheads(&cfg, m)
		return eng, cfg, nil
	}
	build := func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		eng, cfg, err := buildEngine(index, ic)
		if err != nil {
			return nil, err
		}
		return sim.NewEngineDriver(eng, cfg), nil
	}

	if externalDP {
		// Each run of dataParallelism consecutive instances is one shared group's ranks (the same
		// grouping sim/cluster's buildDPReplicas uses). The group is built once, at the replica's
		// base index, and exposed as one per-rank endpoint view; each view gets its own driver over
		// the shared group's configuration.
		dp := dataParallelism
		type replicaViews struct{ drivers []*sim.EngineDriver }
		cache := map[int]*replicaViews{}
		return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
			rep, rank := index/dp, index%dp
			rv := cache[rep]
			if rv == nil {
				eng, cfg, err := buildEngine(index, ic)
				if err != nil {
					return nil, err
				}
				views, ok := dpRankViews(eng)
				if !ok {
					return nil, fmt.Errorf("scenario %q: engine %s runs external DP load balancing but its engine is not a "+
						"data-parallel group that exposes per-rank endpoints", kernelScenario, version)
				}
				if len(views) != dp {
					// Fewer router-visible endpoints than ranks means the group resolved to a hybrid
					// shape (one endpoint fronting several ranks of a node), which this RFC scopes out
					// (inference-sim#2699); only true external balancing (one endpoint per rank,
					// dp_local 1) is driven.
					return nil, fmt.Errorf("scenario %q: parallel.data_parallel_load_balancing external exposes %d endpoint(s) for "+
						"%d data-parallel rank(s); the host drives external balancing only as one endpoint per rank (state "+
						"parallel.dp_local 1). An endpoint fronting several ranks is hybrid balancing, which is not driven yet",
						kernelScenario, len(views), dp)
				}
				rv = &replicaViews{drivers: make([]*sim.EngineDriver, dp)}
				for k, v := range views {
					rv.drivers[k] = sim.NewEngineDriver(v, cfg)
				}
				cache[rep] = rv
			}
			return rv.drivers[rank], nil
		}
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

// reportAssumptions writes an engine's disclosed assumptions (for the vLLM engine's engine, that the
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
