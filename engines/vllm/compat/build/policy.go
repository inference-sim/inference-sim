// Moved from inference-sim/cmd/root.go at 9c6ba5fa0054368bd1cb3007ced919cd69c5c369.
// Moved from inference-sim/cmd/scenario_engine_knobs.go at 9c6ba5fa0054368bd1cb3007ced919cd69c5c369.
// Moved from inference-sim/cmd/routing_imbalance.go at 9c6ba5fa0054368bd1cb3007ced919cd69c5c369.

package build

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/spec/routing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// EnginePolicy is a vLLM scheduling_policy in the compatibility engine's terms: it decides
// both the waiting queue's order and which running request is preempted under KV pressure
// (vLLM v0.31.0 v1/core/sched/scheduler.py: the preemption victim is chosen by priority when
// the policy is PRIORITY, else the last-scheduled request).
type EnginePolicy struct{ Scheduler, Preemption string }

// EnginePolicyOf resolves a pool's engine.scheduling_policy: fcfs is FCFS queueing and
// tail-of-batch preemption; priority is priority-fcfs queueing (lowest priority value first,
// then arrival) and priority preemption. Each pool resolves on its own, and an unstated
// policy is vLLM's default, fcfs (v0.31.0 vllm/config/scheduler.py:160) -- never another
// pool's. Any other value is refused; scenario names the scenario in the error.
func EnginePolicyOf(scenario string, m *kernelmodel.Model) (EnginePolicy, error) {
	eng, err := m.Engine()
	if err != nil {
		return EnginePolicy{}, fmt.Errorf("scenario %q: reading the %s pool's engine: %w", scenario, m.Role(), err)
	}
	policy := eng.SchedulingPolicy
	if policy == "" {
		policy = "fcfs"
	}
	p, ok := map[string]EnginePolicy{
		"fcfs":     {Scheduler: "fcfs", Preemption: "fcfs"},
		"priority": {Scheduler: "priority-fcfs", Preemption: "priority"},
	}[policy]
	if !ok {
		return EnginePolicy{}, fmt.Errorf("scenario %q states engine.scheduling_policy %q; BLIS simulates vLLM's "+
			"fcfs and priority", scenario, eng.SchedulingPolicy)
	}
	return p, nil
}

// EngineKnobFlags are the command line's restatements of the scenario's per-pool engine knobs:
// --dp-load-balancing, with whether it was given. engine.long_prefill_token_threshold has no
// flag (inference-sim#2021 B6): every pool runs what it states.
type EngineKnobFlags struct {
	DPLoadBalancing         string
	DPLoadBalancingFromFlag bool
}

// RunEngineKnobs makes the scenario the source of each pool's
// engine.long_prefill_token_threshold (blis-schemas#75) and
// parallel.data_parallel_load_balancing (#76), as #1910 made it the source of the offload
// tiers. vLLM sets both per engine, so each pool runs its own value. The balancing flag is a
// run-wide override: given, it applies to every pool and must agree with every mode a pool
// states, or the run is refused. A colocated run's one pool sets the run-wide value. It
// returns the run-wide threshold (the colocated pool's, else none) and mode (the flag's unless
// the colocated pool states its own).
func RunEngineKnobs(scenario string, shape kernelmodel.Shape, f EngineKnobFlags) (threshold int64, mode string, err error) {
	for i, m := range shape.DPLoadBalancings {
		if m == "" {
			continue
		}
		if f.DPLoadBalancingFromFlag && string(m.Mode()) != f.DPLoadBalancing {
			return 0, "", refuseRestatement(scenario, "dp-load-balancing", f.DPLoadBalancing,
				poolField(shape, i, "parallel.data_parallel_load_balancing"), string(m.Mode()))
		}
	}
	mode = f.DPLoadBalancing
	if len(shape.Roles) != 1 {
		return threshold, mode, nil
	}
	if v := shape.LongPrefillTokenThresholds[0]; v != 0 {
		threshold = int64(v)
	}
	if m := shape.DPLoadBalancings[0]; m != "" {
		mode = string(m.Mode())
	}
	return threshold, mode, nil
}

// PoolEngineKnobs is a P/D pool's own threshold and balancing mode: what the pool states, else
// vLLM's defaults (no threshold, internal), with the balancing flag's mode when given
// (RunEngineKnobs checked it agrees).
func PoolEngineKnobs(scenario string, m *kernelmodel.Model, f EngineKnobFlags) (threshold int64, mode string, err error) {
	eng, err := m.Engine()
	if err != nil {
		return 0, "", fmt.Errorf("scenario %q: reading the %s pool's engine: %w", scenario, m.Role(), err)
	}
	threshold = int64(eng.LongPrefillTokenThreshold)
	mode = string(m.DPLoadBalancing().Mode())
	if f.DPLoadBalancingFromFlag {
		mode = f.DPLoadBalancing
	}
	return threshold, mode, nil
}

func poolField(shape kernelmodel.Shape, i int, field string) string {
	return fmt.Sprintf("%s pool's %s", shape.Roles[i], field)
}

// refuseRestatement refuses a run-wide flag that disagrees with a value the scenario states.
func refuseRestatement(scenario, flag, flagValue, field, stated string) error {
	return fmt.Errorf("scenario %q: --%s %s disagrees with the scenario's %s %s; the scenario is the "+
		"deployment, so the flag may only restate it -- drop the flag, or change the scenario",
		scenario, flag, flagValue, field, stated)
}

// RoutingFlags are the command line's expert-parallel routing imbalance:
// --routing-token-imbalance and --routing-expert-imbalance, each with whether it was given.
type RoutingFlags struct {
	TokenImbalance  float64
	TokenSet        bool
	ExpertImbalance float64
	ExpertSet       bool
}

// RoutingOverridesOf is the command line's routing layer for the pools about to open
// (blis-schemas routing.Resolve: the command line is the highest layer, above the scenario's
// pin, the workload's value and the registry's default sets). The workload layer is the
// scenario's own binding: the pools open before a catalog preset or an agentic descriptor is
// read, so their category reaches routing through the scenario's pin (routing.profiles plus
// fallback_category), not here. An expert ratio without a token ratio, or an invalid value, is
// refused.
func RoutingOverridesOf(f RoutingFlags) (kernelmodel.RoutingOverrides, error) {
	o := kernelmodel.RoutingOverrides{}
	if f.ExpertSet && !f.TokenSet {
		return o, fmt.Errorf("--routing-expert-imbalance needs --routing-token-imbalance")
	}
	if f.TokenSet {
		v := &routing.Value{TokenImbalance: f.TokenImbalance, Source: "--routing-token-imbalance"}
		if f.ExpertSet {
			e := f.ExpertImbalance
			v.ExpertImbalance = &e
		}
		if p := v.Validate(); !p.OK() {
			return o, fmt.Errorf("--routing-token-imbalance: %s", p.Error())
		}
		o.CLI = v
	}
	return o, nil
}
