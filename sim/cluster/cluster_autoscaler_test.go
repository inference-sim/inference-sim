package cluster

import (
	"strings"
	"testing"
)

// The provisioner adds a node only for a placement that has waited the threshold, never past the
// ceiling, and not while a node of the pool is already on its way.
func TestPendingDrivenProvisioner_Decide(t *testing.T) {
	pool := func(total, provisioning, max int) []NodePoolState {
		return []NodePoolState{{PoolName: "p", GPUType: "A100", TotalNodes: total, ProvisioningNodes: provisioning, MaxNodes: max}}
	}
	waiting := func(since int64) []PendingPlacement {
		return []PendingPlacement{{ModelID: "m", Variant: NewVariantSpec("A100", 1), RequestedAt: since}}
	}
	p := NewPendingDrivenProvisioner(PendingDrivenConfig{PendingThresholdUs: 1000})
	for _, tc := range []struct {
		name string
		got  []NodeDecision
		want int
	}{
		{"under the threshold", p.Decide(1500, pool(1, 0, 4), waiting(1000)), 0},
		{"at the threshold", p.Decide(2000, pool(1, 0, 4), waiting(1000)), 1},
		{"nothing pending", p.Decide(9000, pool(1, 0, 4), nil), 0},
		{"a node is already coming", p.Decide(9000, pool(2, 1, 4), waiting(1000)), 0},
		{"the pool is at its ceiling", p.Decide(9000, pool(4, 0, 4), waiting(1000)), 0},
		{"another GPU type waits", p.Decide(9000, pool(1, 0, 4), []PendingPlacement{{Variant: NewVariantSpec("H100", 1), RequestedAt: 0}}), 0},
		{"several wait: one node", p.Decide(9000, pool(1, 0, 4), append(waiting(0), waiting(10)...)), 1},
	} {
		if len(tc.got) != tc.want {
			t.Errorf("%s: %d decision(s) %v, want %d", tc.name, len(tc.got), tc.got, tc.want)
		}
	}
	capped := NewPendingDrivenProvisioner(PendingDrivenConfig{PendingThresholdUs: 0, MaxNodesPerPool: 2})
	if got := capped.Decide(10, pool(2, 0, 8), waiting(0)); len(got) != 0 {
		t.Errorf("MaxNodesPerPool 2 with 2 nodes still added %v", got)
	}
}

func TestClusterAutoscalerConfig_Validate(t *testing.T) {
	for _, tc := range []struct {
		cfg  ClusterAutoscalerConfig
		want string
	}{
		{ClusterAutoscalerConfig{}, ""},
		{ClusterAutoscalerConfig{Policy: ClusterAutoscalerPendingDriven}, ""},
		{ClusterAutoscalerConfig{Policy: "bin-pack"}, "not a policy"},
		{ClusterAutoscalerConfig{Policy: ClusterAutoscalerPendingDriven, PendingDrivenConfig: PendingDrivenConfig{PendingThresholdUs: -1}}, "pending_threshold_us"},
		{ClusterAutoscalerConfig{Policy: ClusterAutoscalerPendingDriven, PendingDrivenConfig: PendingDrivenConfig{MaxNodesPerPool: -1}}, "max_nodes_per_pool"},
	} {
		err := tc.cfg.Validate()
		if (tc.want == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), tc.want)) {
			t.Errorf("%+v: error %v, want %q", tc.cfg, err, tc.want)
		}
	}
}

// scaleUpOnce asks for one more replica of the pool's variant on its first tick.
type scaleUpOnce struct{ fired bool }

func (e *scaleUpOnce) Optimize(_ []AnalyzerResult, _ GPUInventory) []ScaleDecision {
	if e.fired {
		return nil
	}
	e.fired = true
	return []ScaleDecision{{ModelID: "test-model", Variant: NewVariantSpec("A100", 1), Delta: 1}}
}

// autoscalerRun is a full pool of two GPUs running two instances, a model autoscaler that asks for
// a third replica, and the given cluster autoscaler. It returns the cluster after the run.
func autoscalerRun(t *testing.T, ca ClusterAutoscalerConfig) *ClusterSimulator {
	t.Helper()
	d := newTestDeploymentConfig(2)
	d.Model = "test-model"
	d.ModelAutoscalerIntervalUs = 1_000_000
	d.NodePools = []NodePoolConfig{{Name: "p", GPUType: "A100", GPUsPerNode: 2, InitialNodes: 1, MaxNodes: 2,
		GPUMemoryGiB: 80, ProvisioningDelay: DelaySpec{Mean: 1}}}
	d.ClusterAutoscaler = ca
	cs := newTestCluster(d, NewSliceRequestSource(newTestRequests(4)), nil)
	cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, &scaleUpOnce{}, NewDirectActuator(cs))
	mustRun(t, cs)
	return cs
}

// The two autoscalers meet in the pending queue: the model autoscaler asks for a replica, the pool
// is full, the placement waits, the cluster autoscaler provisions a node once it has waited the
// threshold, and the node coming up places the replica. With no cluster autoscaler the replica
// waits for ever, and a ceiling of one node stops the provisioning.
func TestClusterAutoscaler_ProvisionsANodeForAWaitingPlacement(t *testing.T) {
	with := autoscalerRun(t, ClusterAutoscalerConfig{Policy: ClusterAutoscalerPendingDriven,
		PendingDrivenConfig: PendingDrivenConfig{PendingThresholdUs: 500_000}})
	if n := with.placement.NodeCount(); n != 2 {
		t.Errorf("%d node(s) after the run, want the pool's second", n)
	}
	if n := len(with.instances); n != 3 {
		t.Errorf("%d instance(s), want the waiting replica placed on the new node (3)", n)
	}
	if q := with.placement.Pending(); len(q) != 0 {
		t.Errorf("pending queue after the node came up = %+v, want empty", q)
	}
	if err := with.placement.VerifyConservation(); err != nil {
		t.Errorf("GPU conservation after provisioning: %v", err)
	}

	for name, ca := range map[string]ClusterAutoscalerConfig{
		"no cluster autoscaler": {},
		"a ceiling of one node": {Policy: ClusterAutoscalerPendingDriven,
			PendingDrivenConfig: PendingDrivenConfig{MaxNodesPerPool: 1}},
	} {
		cs := autoscalerRun(t, ca)
		if n := cs.placement.NodeCount(); n != 1 {
			t.Errorf("%s: %d node(s), want the pool's one", name, n)
		}
		if q := cs.placement.Pending(); len(q) != 1 {
			t.Errorf("%s: pending queue %+v, want the replica still waiting", name, q)
		}
	}
}
