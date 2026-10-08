package cluster

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// startRecordingPolicy is a round-robin RoutingPolicy that subscribes to
// first-token events (sim.RequestStartObserver) and records each one.
type startRecordingPolicy struct {
	inner     sim.RoundRobin
	subscribe bool
	events    []string
}

func (p *startRecordingPolicy) Route(req *sim.Request, s *sim.RouterState) sim.RoutingDecision {
	return p.inner.Route(req, s)
}
func (p *startRecordingPolicy) ObservesRequestStart() bool { return p.subscribe }
func (p *startRecordingPolicy) OnRequestStart(req *sim.Request, inst string, _ int64) {
	p.events = append(p.events, req.ID+"@"+inst)
}

// With no subscriber every instance's OnFirstToken stays nil, so the step loop
// is unchanged for every configuration that does not name a subscribing scorer.
func TestStartObserver_InertLeavesOnFirstTokenNil(t *testing.T) {
	for _, pol := range []string{"round-robin", "weighted", "route-to-holder"} {
		cfg := newTestDeploymentConfig(2)
		cfg.RoutingPolicy = pol
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		assert.Nil(t, cs.startObservers, pol)
		for _, inst := range cs.Instances() {
			assert.Nil(t, inst.sim.OnFirstToken, "%s: instance %s", pol, inst.ID())
		}
	}
	assert.Nil(t, collectStartObservers(&startRecordingPolicy{subscribe: false}, nil))
	sub := &startRecordingPolicy{subscribe: true}
	assert.Equal(t, []sim.RequestStartObserver{sub}, collectStartObservers(nil, sub, &startRecordingPolicy{}))
}

// A configured lora-residency scorer subscribes, so the startup path wires it.
func TestStartObserver_LoRAResidencyWired(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	cfg.RoutingPolicy = "weighted"
	cfg.RoutingScorerConfigs = []sim.ScorerConfig{{Name: "lora-residency", Weight: 1}}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	require.Len(t, cs.startObservers, 1)
	for _, inst := range cs.Instances() {
		assert.NotNil(t, inst.sim.OnFirstToken, "instance %s", inst.ID())
	}
}

// wireOnFirstToken notifies subscribers with the wired instance's id.
func TestStartObserver_WireHelperNotifiesWithInstanceID(t *testing.T) {
	cs := NewClusterSimulator(newTestDeploymentConfig(2), NewSliceRequestSource(nil), nil)
	rec := &startRecordingPolicy{subscribe: true}
	cs.startObservers = collectStartObservers(rec)
	for _, inst := range cs.Instances() {
		cs.wireOnFirstToken(inst)
		require.NotNil(t, inst.sim.OnFirstToken)
		inst.sim.OnFirstToken(&sim.Request{ID: "x"}, 7)
	}
	ids := cs.Instances()
	assert.Equal(t, []string{"x@" + string(ids[0].ID()), "x@" + string(ids[1].ID())}, rec.events)
}

// The live-added instance site is wired too.
func TestStartObserver_LiveAddedInstanceSite(t *testing.T) {
	dc := newTestDeploymentConfig(1)
	dc.Model = "test-model"
	dc.NodePools = []NodePoolConfig{
		{Name: "h100-pool", GPUType: "H100", GPUsPerNode: 4, InitialNodes: 1, MaxNodes: 2, GPUMemoryGiB: 80},
	}
	cs := NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)
	rec := &startRecordingPolicy{subscribe: true}
	cs.routingPolicy = rec
	cs.startObservers = collectStartObservers(cs.routingPolicy, cs.prefillRoutingPolicy, cs.decodeRoutingPolicy)

	require.NoError(t, NewDirectActuator(cs).Apply([]ScaleDecision{
		{ModelID: "test-model", Variant: NewVariantSpec("H100", 1), Delta: 1},
	}))
	var scaled *InstanceSimulator
	for _, inst := range cs.instances {
		if strings.HasPrefix(string(inst.ID()), "autoscale-") {
			scaled = inst
		}
	}
	require.NotNil(t, scaled, "precondition: scale-up created an instance")
	require.NotNil(t, scaled.sim.OnFirstToken, "live-added instance must be wired")
	scaled.sim.OnFirstToken(&sim.Request{ID: "y"}, 9)
	assert.Equal(t, []string{"y@" + string(scaled.ID())}, rec.events)
}
