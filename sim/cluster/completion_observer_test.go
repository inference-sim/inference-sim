package cluster

import (
	"container/heap"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// recordingPolicy is a round-robin RoutingPolicy that subscribes to request
// completions (sim.RequestCompletionObserver) and records each event.
type recordingPolicy struct {
	inner     sim.RoundRobin
	subscribe bool
	events    []string
}

func (p *recordingPolicy) Route(req *sim.Request, s *sim.RouterState) sim.RoutingDecision {
	return p.inner.Route(req, s)
}
func (p *recordingPolicy) ObservesRequestCompletion() bool { return p.subscribe }
func (p *recordingPolicy) OnRequestCompletion(req *sim.Request, inst string, _ int64) {
	p.events = append(p.events, req.ID+"@"+inst)
}

// Guard: with no subscriber (and no session/tenant/eviction consumer) the startup
// path leaves every instance's OnRequestDone nil — exactly as before.
func TestCompletionObserver_InertLeavesOnRequestDoneNil(t *testing.T) {
	for _, pol := range []string{"round-robin", "weighted", "route-to-holder"} {
		cfg := newTestDeploymentConfig(2)
		cfg.RoutingPolicy = pol
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		assert.Nil(t, cs.completionObservers, pol)
		for _, inst := range cs.Instances() {
			assert.Nil(t, inst.sim.OnRequestDone, "%s: instance %s", pol, inst.ID())
		}
	}
	// A policy implementing the interface but reporting no subscriber is not collected.
	assert.Nil(t, collectCompletionObservers(&recordingPolicy{subscribe: false}, nil))
	sub := &recordingPolicy{subscribe: true}
	assert.Equal(t, []sim.RequestCompletionObserver{sub}, collectCompletionObservers(nil, sub, &recordingPolicy{}))
}

// wireOnRequestDone notifies subscribers with the wired instance's id.
func TestCompletionObserver_WireHelperNotifiesWithInstanceID(t *testing.T) {
	cs := NewClusterSimulator(newTestDeploymentConfig(2), NewSliceRequestSource(nil), nil)
	rec := &recordingPolicy{subscribe: true}
	cs.completionObservers = collectCompletionObservers(rec)
	for _, inst := range cs.Instances() {
		cs.wireOnRequestDone(inst)
		require.NotNil(t, inst.sim.OnRequestDone)
		assert.Nil(t, inst.sim.OnRequestDone(&sim.Request{ID: "x"}, 7), "no follow-ups")
	}
	assert.Empty(t, rec.events, "delivered as a cluster event at the tick, not synchronously")
	drainClusterEvents(cs)
	ids := cs.Instances()
	assert.Equal(t, []string{"x@" + string(ids[0].ID()), "x@" + string(ids[1].ID())}, rec.events)
}

// Second wiring site: an instance added while the cluster is live
// (addLiveInstance, via the autoscaler's DirectActuator) is wired too.
func TestCompletionObserver_LiveAddedInstanceSite(t *testing.T) {
	dc := newTestDeploymentConfig(1)
	dc.Model = "test-model"
	dc.NodePools = []NodePoolConfig{
		{Name: "h100-pool", GPUType: "H100", GPUsPerNode: 4, InitialNodes: 1, MaxNodes: 2, GPUMemoryGiB: 80},
	}
	cs := NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)
	rec := &recordingPolicy{subscribe: true}
	cs.routingPolicy = rec
	cs.completionObservers = collectCompletionObservers(cs.routingPolicy, cs.prefillRoutingPolicy, cs.decodeRoutingPolicy)

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
	require.NotNil(t, scaled.sim.OnRequestDone, "live-added instance must be wired")
	scaled.sim.OnRequestDone(&sim.Request{ID: "y"}, 9)
	drainClusterEvents(cs)
	assert.Equal(t, []string{"y@" + string(scaled.ID())}, rec.events)
}

// drainClusterEvents executes every queued cluster event, as the run loop would.
func drainClusterEvents(cs *ClusterSimulator) {
	for len(cs.clusterEvents) > 0 {
		entry := heap.Pop(&cs.clusterEvents).(clusterEventEntry)
		cs.clock = entry.event.Timestamp()
		entry.event.Execute(cs)
	}
}
