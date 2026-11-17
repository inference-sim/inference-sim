package cluster

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"time"

	"github.com/inference-sim/blis-vllm/compat/contract"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// placerEngine is the compatibility engine with the dynamic adapter load a v0.33 engine offers
// (blis-vllm core/vllm Engine.LoadAdapter): a cache of at most capacity adapters, oldest evicted.
type placerEngine struct {
	*contract.Engine
	capacity int
	resident []string
	loads    int
}

func (e *placerEngine) LoadAdapter(name string) error {
	e.loads++
	if i := slices.Index(e.resident, name); i >= 0 {
		e.resident = slices.Delete(e.resident, i, i+1)
	} else if len(e.resident) >= e.capacity {
		e.resident = e.resident[1:]
	}
	e.resident = append(e.resident, name)
	return nil
}

func (e *placerEngine) UnloadAdapter(name string) (bool, error) {
	i := slices.Index(e.resident, name)
	if i < 0 {
		return false, nil
	}
	e.resident = slices.Delete(e.resident, i, i+1)
	return true, nil
}

func (e *placerEngine) ResidentAdapterIDs() []string { return slices.Clone(e.resident) }
func (e *placerEngine) AdapterCapacity() int         { return e.capacity }

// placerCluster is a cluster of n instances whose engines are placerEngines of the given capacity,
// returned with those engines by instance index.
func placerCluster(t *testing.T, n, capacity int, cfg LoRAPlacementConfig, reqs []*sim.Request) (*ClusterSimulator, []*placerEngine) {
	t.Helper()
	d := newTestDeploymentConfig(n)
	d.LoRAPlacement = cfg
	var engines []*placerEngine
	d.NewEngine = func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		c := engineConfigIn(ic)
		eng, err := contract.New(fmt.Sprintf("instance_%d", index), c)
		if err != nil {
			return nil, err
		}
		pe := &placerEngine{Engine: eng, capacity: capacity}
		engines = append(engines, pe)
		dcfg := sim.EngineDriverConfig{Horizon: c.Horizon, ResumeGapFromLastToken: true, ScrapeEngineMetrics: true,
			ApplyAtStepStart: true, Environment: testutil.FixedPull(c.BlockSizeTokens, testPDTransferTime)}
		if m := c.LatencyModel; m != nil {
			dcfg.PostDecodeFixedOverhead = m.PostDecodeFixedOverhead()
			dcfg.OutputTokenProcessingTime = m.OutputTokenProcessingTime()
			dcfg.Environment = testutil.WithHost(dcfg.Environment, func(n int) kernel.HostDemand {
				return kernel.HostDemand{Service: time.Duration(m.QueueingTime(&cengine.Request{
					InputTokens: make([]cengine.TokenID, n)})) * time.Microsecond}
			})
		}
		return sim.NewEngineDriver(pe, dcfg), nil
	}
	return newTestCluster(d, NewSliceRequestSource(reqs), nil), engines
}

// Pre-placement loads the declared assignment on each instance at time zero, before any request.
func TestLoRAPlacement_PrePlacementLoadsTheAssignmentBeforeAnyRequest(t *testing.T) {
	cs, engines := placerCluster(t, 2, 4, LoRAPlacementConfig{Creation: LoRACreationPrePlacement,
		Assignment: map[int][]string{0: {"a", "b"}, 1: {"c"}}}, nil)
	if cs.startupErr != nil {
		t.Fatal(cs.startupErr)
	}
	if got := engines[0].resident; !slices.Equal(got, []string{"a", "b"}) {
		t.Errorf("instance 0 holds %v at time zero, want [a b]", got)
	}
	if got := engines[1].resident; !slices.Equal(got, []string{"c"}) {
		t.Errorf("instance 1 holds %v at time zero, want [c]", got)
	}
	if got := cs.instances[0].ResidentAdapterIDs(); !slices.Equal(got, []string{"a", "b"}) {
		t.Errorf("the cluster publishes %v as instance 0's residency, want [a b]", got)
	}
}

// An assignment the cluster cannot honour is refused before the run, never truncated.
func TestLoRAPlacement_RefusesWhatItCannotHonour(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		cfg        LoRAPlacementConfig
	}{
		{"more adapters than the cache holds", "holds 2", LoRAPlacementConfig{Creation: LoRACreationPrePlacement,
			Assignment: map[int][]string{0: {"a", "b", "c"}}}},
		{"an instance the cluster lacks", "instance 5", LoRAPlacementConfig{Creation: LoRACreationPrePlacement,
			Assignment: map[int][]string{5: {"a"}}}},
		{"an adapter listed twice", "twice", LoRAPlacementConfig{Creation: LoRACreationPrePlacement,
			Assignment: map[int][]string{0: {"a", "a"}}}},
		{"an assignment with on-demand creation", "places nothing", LoRAPlacementConfig{Creation: LoRACreationOnDemand,
			Assignment: map[int][]string{0: {"a"}}}},
		{"an unknown bundle", "not a bundle", LoRAPlacementConfig{Bundle: "nope"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cs, _ := placerCluster(t, 2, 2, tc.cfg, nil)
			if cs.startupErr == nil || !strings.Contains(cs.startupErr.Error(), tc.want) {
				t.Errorf("startup error = %v, want one naming %q", cs.startupErr, tc.want)
			}
		})
	}
	// An engine that cannot load adapters (the unwrapped compatibility engine) is refused by name.
	d := newTestDeploymentConfig(1)
	d.LoRAPlacement = LoRAPlacementConfig{Creation: LoRACreationPrePlacement, Assignment: map[int][]string{0: {"a"}}}
	cs := newTestCluster(d, NewSliceRequestSource(nil), nil)
	if cs.startupErr == nil || !strings.Contains(cs.startupErr.Error(), "cannot load adapters") {
		t.Errorf("an engine without a dynamic load: startup error = %v", cs.startupErr)
	}
}

// A bundle names a creation policy; an explicit field overrides it for that field only.
func TestLoRAPlacement_BundleResolution(t *testing.T) {
	r, err := LoRAPlacementConfig{Bundle: "static-placement", Assignment: map[int][]string{0: {"a"}}}.Resolve()
	if err != nil || r.Creation != LoRACreationPrePlacement {
		t.Fatalf("static-placement resolved to %q (%v), want pre-placement", r.Creation, err)
	}
	r, err = LoRAPlacementConfig{Bundle: "static-placement", Creation: LoRACreationOnDemand}.Resolve()
	if err != nil || r.Creation != LoRACreationOnDemand {
		t.Errorf("the explicit creation did not override the bundle: %q (%v)", r.Creation, err)
	}
	if r, err = (LoRAPlacementConfig{}).Resolve(); err != nil || r.Places() {
		t.Errorf("the zero config must be the baseline that places nothing (%v)", err)
	}
}

// The periodic trigger loads again what an engine has since lost, and is inert when off: with no
// placement policy, or an interval of zero, no event is ever scheduled and nothing is loaded.
func TestLoRAPlacement_PeriodicReconcilesAndIsInertWhenOff(t *testing.T) {
	reqs := newTestRequests(6)
	for i, r := range reqs {
		r.ArrivalTime = int64(i) * 2_000_000 // two seconds apart
	}

	t.Run("on reloads an adapter the engine lost", func(t *testing.T) {
		cs, engines := placerCluster(t, 1, 2, LoRAPlacementConfig{Creation: LoRACreationPrePlacement,
			Assignment: map[int][]string{0: {"a", "b"}}, PeriodicIntervalUs: 1_000_000}, reqs)
		engines[0].UnloadAdapter("a") // lost after the start-of-run placement
		mustRun(t, cs)
		if !slices.Contains(engines[0].resident, "a") {
			t.Errorf("the periodic trigger did not load a again; resident %v", engines[0].resident)
		}
	})

	t.Run("off schedules nothing", func(t *testing.T) {
		for name, cfg := range map[string]LoRAPlacementConfig{
			"no policy":            {},
			"on-demand + interval": {Creation: LoRACreationOnDemand, PeriodicIntervalUs: 1_000_000},
			"pre-placement without interval": {Creation: LoRACreationPrePlacement,
				Assignment: map[int][]string{0: {"a"}}},
		} {
			cs, engines := placerCluster(t, 1, 2, cfg, newTestRequests(3))
			mustRun(t, cs)
			if got := cs.loraPlacementSender.sent; got != 0 {
				t.Errorf("%s: %d placement tick(s) scheduled, want none", name, got)
			}
			if name != "pre-placement without interval" && engines[0].loads != 0 {
				t.Errorf("%s: %d adapter loads, want none", name, engines[0].loads)
			}
		}
	})
}
