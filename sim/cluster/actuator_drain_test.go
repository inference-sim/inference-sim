package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// drainFixture is a two-instance cluster for model "m1" with numSeeded requests queued on its
// first instance (the one a scale-down drains: the lowest ID), and the actuator that drains it
// under the named policy ("" states none).
func drainFixture(t *testing.T, policy string, numSeeded int) (*ClusterSimulator, *InstanceSimulator, *DirectActuator) {
	t.Helper()
	cfg := newTestDeploymentConfig(2)
	cfg.InstanceLifecycle = InstanceLifecycleConfig{DrainPolicy: policy}
	cs := newTestCluster(cfg, NewSliceRequestSource([]*sim.Request{}), nil)
	cs.gateway = newDefaultGateway() // a scale-down happens inside Run, when the gateway exists
	for _, inst := range cs.instances {
		inst.Model, inst.gpu, inst.TPDegree = "m1", "A100", 1
		inst.State = sim.InstanceStateActive
	}
	inst0 := cs.instances[0]
	for _, req := range newTestRequests(numSeeded) {
		req.ArrivalTime = 0
		admitToLedger(t, cs, req)
		inst0.InjectRequestOnline(req, 0)
		cs.inFlightRequests[string(inst0.ID())]++
	}
	for numSeeded > 0 && inst0.HasPendingEvents() && inst0.QueueDepth() < numSeeded {
		inst0.ProcessNextEvent()
	}
	if inst0.QueueDepth() != numSeeded {
		t.Fatalf("precondition: %d requests queued on the instance to drain, want %d", inst0.QueueDepth(), numSeeded)
	}
	return cs, inst0, NewDirectActuator(cs)
}

func scaleDownOne(t *testing.T, a *DirectActuator) {
	t.Helper()
	if err := a.Apply([]ScaleDecision{{ModelID: "m1", Variant: NewVariantSpec("A100", 1), Delta: -1}}); err != nil {
		t.Fatalf("scale-down: %v", err)
	}
}

// The scale-down path drains with the configured policy (#910). Before, DirectActuator.scaleDown
// moved every instance to Draining whatever drain_policy stated.
func TestScaleDownDrainsByTheConfiguredPolicy(t *testing.T) {
	const n = 3

	t.Run("IMMEDIATE terminates at once and abandons the work", func(t *testing.T) {
		cs, inst0, a := drainFixture(t, string(DrainPolicyImmediate), n)
		scaleDownOne(t, a)
		if inst0.State != sim.InstanceStateTerminated {
			t.Errorf("state = %q after an IMMEDIATE scale-down, want Terminated in the same tick", inst0.State)
		}
		if got := cs.inFlightRequests[string(inst0.ID())]; got != 0 {
			t.Errorf("in-flight count on the terminated instance = %d, want 0", got)
		}
		if got := inst0.Metrics().DroppedUnservable; got != n {
			t.Errorf("the instance counts %d requests dropped, want the %d it held", got, n)
		}
		for i := 0; i < n; i++ {
			id := fmt.Sprintf("request_%d", i)
			if f, ok := cs.Fates().Fate(id); !ok || f.Fate != sim.FateDropped {
				t.Errorf("%s: fate %+v (found %v), want dropped", id, f, ok)
			}
		}
		if err := cs.Run(); err != nil {
			t.Fatal(err)
		}
		if got := cs.AggregatedMetrics().CompletedRequests; got != 0 {
			t.Errorf("%d requests completed; those held by an IMMEDIATE-drained instance are abandoned", got)
		}
	})

	t.Run("WAIT keeps the instance until its work is done", func(t *testing.T) {
		cs, inst0, a := drainFixture(t, string(DrainPolicyWait), n)
		scaleDownOne(t, a)
		if inst0.State != sim.InstanceStateDraining {
			t.Fatalf("state = %q right after a WAIT scale-down, want Draining", inst0.State)
		}
		if inst0.IsRoutable() {
			t.Error("a draining instance is still routable")
		}
		if inst0.QueueDepth() != n {
			t.Errorf("queued = %d, want the instance's %d requests kept", inst0.QueueDepth(), n)
		}
		if err := cs.Run(); err != nil {
			t.Fatal(err)
		}
		if got := cs.AggregatedMetrics().CompletedRequests; got != n {
			t.Errorf("completed = %d, want all %d finished on the draining instance", got, n)
		}
		if inst0.State != sim.InstanceStateTerminated {
			t.Errorf("final state = %q, want Terminated once the work was done", inst0.State)
		}
		assertClusterINV1Conservation(t, cs, n, noRejections, "WAIT scale-down")
	})

	t.Run("REDIRECT re-injects the queued work elsewhere", func(t *testing.T) {
		cs, inst0, a := drainFixture(t, string(DrainPolicyRedirect), n)
		scaleDownOne(t, a)
		if inst0.State != sim.InstanceStateDraining {
			t.Errorf("state = %q right after a REDIRECT scale-down, want Draining", inst0.State)
		}
		if inst0.QueueDepth() != 0 {
			t.Errorf("queued = %d, want the queue handed back", inst0.QueueDepth())
		}
		if err := cs.Run(); err != nil {
			t.Fatal(err)
		}
		if got := cs.AggregatedMetrics().CompletedRequests; got != n {
			t.Errorf("completed = %d, want all %d served by the other instance", got, n)
		}
		assertClusterINV1Conservation(t, cs, n, noRejections, "REDIRECT scale-down")
	})

	t.Run("no policy stated is WAIT", func(t *testing.T) {
		_, inst0, a := drainFixture(t, "", n)
		scaleDownOne(t, a)
		if inst0.State != sim.InstanceStateDraining || inst0.QueueDepth() != n {
			t.Errorf("state %q with %d queued, want the WAIT outcome: Draining with the work kept", inst0.State, inst0.QueueDepth())
		}
	})

	t.Run("WAIT releases an idle instance at once", func(t *testing.T) {
		_, inst0, a := drainFixture(t, string(DrainPolicyWait), 0)
		scaleDownOne(t, a)
		if inst0.State != sim.InstanceStateTerminated {
			t.Errorf("state = %q, want an idle instance Terminated with its GPUs released at once", inst0.State)
		}
	})
}
