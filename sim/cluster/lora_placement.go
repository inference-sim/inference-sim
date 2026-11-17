package cluster

import (
	"fmt"
	"math"
	"sort"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// The LoRA placement loop (design SIM-14, inference-sim#1493 and #1495): a control loop beside the
// router that decides which adapters are loaded on which instances. It is a component (design
// section 8.1): it acts only on what the cluster publishes of each instance (the adapters its
// engine reports resident, and its capacity), and it acts only through the engine's own path for
// the purpose, the dynamic adapter load vLLM's serving API offers (/v1/load_lora_adapter), never
// through a hook inside the engine. Routing to the instances that hold an adapter is the router's
// lora-affinity scorer, reading the residency the engines publish.
//
// Two policies exist for when an adapter first becomes resident:
//
//   - on-demand (the default): nothing is placed; an adapter is loaded by the first request that
//     names it, as the engine's adapter cache does. The loop does nothing, and a run without a
//     lora_placement block is unchanged (byte-identical).
//   - pre-placement: the declared assignment (instance index -> adapter names) is loaded at
//     time zero, before any request, so a request for a placed adapter needs no load of its own.
//
// A periodic trigger repeats the placement every PeriodicIntervalUs of simulated time: an adapter
// the engine has since evicted or unloaded is loaded again. With no interval, or on-demand
// creation, no event is ever scheduled (inert when off).

// LoRA creation policy names.
const (
	LoRACreationOnDemand     = "on-demand"
	LoRACreationPrePlacement = "pre-placement"
)

// loraBundles are the named strategy bundles: a name stands for a creation policy. An explicit
// field of the config overrides the bundle's value for that field only.
var loraBundles = map[string]LoRAPlacementConfig{
	"on-demand":        {Creation: LoRACreationOnDemand},
	"static-placement": {Creation: LoRACreationPrePlacement},
}

// LoRAPlacementConfig configures the loop. The zero value is the baseline: on-demand, no
// assignment, no periodic trigger.
type LoRAPlacementConfig struct {
	// Bundle names a strategy bundle ("on-demand", "static-placement"); Creation, when set,
	// overrides its creation policy.
	Bundle   string `yaml:"bundle"`
	Creation string `yaml:"creation"`
	// Assignment is the adapters to hold on each instance, by the instance's index (0 is
	// instance_0), in the order they are loaded.
	Assignment map[int][]string `yaml:"assignment"`
	// PeriodicIntervalUs, when above 0 and the creation policy places adapters, repeats the
	// placement at that interval of simulated time.
	PeriodicIntervalUs float64 `yaml:"periodic_interval_us"`
}

// Resolve returns the effective config: the bundle's values with this config's own fields laid
// over them. It refuses a name that is not a bundle or a creation policy.
func (c LoRAPlacementConfig) Resolve() (LoRAPlacementConfig, error) {
	out := LoRAPlacementConfig{Creation: LoRACreationOnDemand, Assignment: c.Assignment, PeriodicIntervalUs: c.PeriodicIntervalUs, Bundle: c.Bundle}
	if c.Bundle != "" {
		b, ok := loraBundles[c.Bundle]
		if !ok {
			return out, fmt.Errorf("lora_placement.bundle %q is not a bundle; the bundles are %s", c.Bundle, bundleNames())
		}
		out.Creation = b.Creation
	}
	if c.Creation != "" {
		out.Creation = c.Creation
	}
	switch out.Creation {
	case LoRACreationOnDemand, LoRACreationPrePlacement:
	default:
		return out, fmt.Errorf("lora_placement.creation %q is not a creation policy; they are %s and %s",
			out.Creation, LoRACreationOnDemand, LoRACreationPrePlacement)
	}
	if math.IsNaN(out.PeriodicIntervalUs) || math.IsInf(out.PeriodicIntervalUs, 0) || out.PeriodicIntervalUs < 0 {
		return out, fmt.Errorf("lora_placement.periodic_interval_us must be a finite number >= 0, got %v", out.PeriodicIntervalUs)
	}
	if out.Creation == LoRACreationOnDemand && len(out.Assignment) > 0 {
		return out, fmt.Errorf("lora_placement.assignment is set but the creation policy is %s, which places nothing; "+
			"use creation: %s", out.Creation, LoRACreationPrePlacement)
	}
	if out.Creation == LoRACreationPrePlacement && len(out.Assignment) == 0 {
		return out, fmt.Errorf("lora_placement: creation %s needs an assignment", LoRACreationPrePlacement)
	}
	return out, nil
}

// Places reports whether the loop places any adapter.
func (c LoRAPlacementConfig) Places() bool { return c.Creation == LoRACreationPrePlacement }

func bundleNames() string {
	names := make([]string, 0, len(loraBundles))
	for k := range loraBundles {
		names = append(names, k)
	}
	sort.Strings(names)
	return fmt.Sprint(names)
}

// adapterPlacer is what an engine offers a placement loop: vLLM's dynamic adapter load and unload
// (/v1/load_lora_adapter, /v1/unload_lora_adapter). An engine that does not is refused at startup.
type adapterPlacer interface {
	LoadAdapter(name string) error
	UnloadAdapter(name string) (bool, error)
}

// loraPlacement is the loop's state: the effective config, and how many placements it has made.
type loraPlacement struct {
	cfg    LoRAPlacementConfig
	loaded int // adapter loads the loop has issued (Initial and periodic)
}

// placerOf is inst's engine as a placer, nil when it is not one.
func placerOf(inst *InstanceSimulator) adapterPlacer {
	if inst == nil || inst.drv == nil {
		return nil
	}
	p, _ := inst.drv.Engine().(adapterPlacer)
	return p
}

// startLoRAPlacement validates the config against the instances built at startup and, for
// pre-placement, loads the assignment at time zero. It returns the loop, nil when it places
// nothing. An assignment that names an instance the cluster does not have, an instance that cannot
// hold the adapters stated for it, a repeated adapter, or an engine that cannot load adapters is an
// error: nothing is truncated or dropped.
func (cs *ClusterSimulator) startLoRAPlacement() error {
	cfg, err := cs.config.LoRAPlacement.Resolve()
	if err != nil {
		return err
	}
	if !cfg.Places() {
		return nil
	}
	byIndex := map[int]*InstanceSimulator{}
	for _, inst := range cs.instances {
		var idx int
		if _, err := fmt.Sscanf(string(inst.ID()), "instance_%d", &idx); err == nil {
			byIndex[idx] = inst
		}
	}
	indexes := make([]int, 0, len(cfg.Assignment))
	for idx := range cfg.Assignment {
		indexes = append(indexes, idx)
	}
	sort.Ints(indexes)
	for _, idx := range indexes {
		names := cfg.Assignment[idx]
		inst := byIndex[idx]
		if inst == nil {
			return fmt.Errorf("lora_placement.assignment names instance %d, but the cluster has instances 0 to %d",
				idx, cs.config.NumInstances-1)
		}
		p := placerOf(inst)
		if p == nil {
			return fmt.Errorf("lora_placement: instance %d's engine cannot load adapters (vLLM's dynamic load); "+
				"placement needs an engine that models it", idx)
		}
		seen := map[string]bool{}
		for _, n := range names {
			if n == "" || seen[n] {
				return fmt.Errorf("lora_placement.assignment for instance %d lists adapter %q twice or empty", idx, n)
			}
			seen[n] = true
		}
		if capacity := inst.AdapterCapacity(); len(names) > capacity {
			return fmt.Errorf("lora_placement.assignment puts %d adapters on instance %d, which holds %d (max_cpu_loras)",
				len(names), idx, capacity)
		}
	}
	cs.loraPlacement = &loraPlacement{cfg: cfg}
	cs.placeAdapters(indexes, byIndex)
	return nil
}

// placeAdapters loads, on each assigned instance, the adapters it does not already hold, in the
// assignment's order.
func (cs *ClusterSimulator) placeAdapters(indexes []int, byIndex map[int]*InstanceSimulator) {
	lp := cs.loraPlacement
	for _, idx := range indexes {
		inst := byIndex[idx]
		if inst == nil || inst.State == sim.InstanceStateTerminated {
			continue
		}
		held := map[string]bool{}
		for _, n := range inst.ResidentAdapterIDs() {
			held[n] = true
		}
		for _, n := range lp.cfg.Assignment[idx] {
			if held[n] {
				continue
			}
			if err := placerOf(inst).LoadAdapter(n); err != nil {
				// Capacity was checked at start; an error now is an engine refusal worth seeing.
				logrus.Errorf("[lora-placement] load %q on %s: %v", n, inst.ID(), err)
				continue
			}
			lp.loaded++
		}
	}
}

// LoRAPlacementTickEvent is the periodic trigger: it repeats the placement and schedules the next
// tick while the run has work. Never scheduled when the loop is off or has no interval.
type LoRAPlacementTickEvent struct{ At int64 }

func (e *LoRAPlacementTickEvent) Timestamp() int64 { return e.At }
func (e *LoRAPlacementTickEvent) Priority() int    { return 10 }

// Execute reconciles the placement and schedules the next tick.
func (e *LoRAPlacementTickEvent) Execute(cs *ClusterSimulator) {
	lp := cs.loraPlacement
	if lp == nil {
		return
	}
	byIndex := map[int]*InstanceSimulator{}
	indexes := make([]int, 0, len(lp.cfg.Assignment))
	for idx := range lp.cfg.Assignment {
		indexes = append(indexes, idx)
		var found *InstanceSimulator
		for _, inst := range cs.instances {
			var i int
			if _, err := fmt.Sscanf(string(inst.ID()), "instance_%d", &i); err == nil && i == idx {
				found = inst
			}
		}
		byIndex[idx] = found
	}
	sort.Ints(indexes)
	cs.placeAdapters(indexes, byIndex)
	// Like the scaling tick: in a request-bounded run, stop once nothing is left to serve.
	if cs.config.Horizon == math.MaxInt64 && cs.pendingArrivals <= 0 {
		inFlight := 0
		for _, v := range cs.inFlightRequests {
			inFlight += v
		}
		if inFlight == 0 {
			return
		}
	}
	cs.schedule(&LoRAPlacementTickEvent{At: e.At + int64(lp.cfg.PeriodicIntervalUs)}, &cs.loraPlacementSender)
}

// scheduleFirstLoRAPlacementTick starts the periodic trigger at Run, when it is on.
func (cs *ClusterSimulator) scheduleFirstLoRAPlacementTick() {
	lp := cs.loraPlacement
	if lp == nil || lp.cfg.PeriodicIntervalUs <= 0 {
		return
	}
	cs.schedule(&LoRAPlacementTickEvent{At: cs.clock + int64(lp.cfg.PeriodicIntervalUs)}, &cs.loraPlacementSender)
}
