package cluster

import "testing"

// TestInstancesAreBilledAtTheirOwnPoolsPrice is SIM-17's price clause: with an H100 pool and an
// A100 pool under one workload, each placed instance reports the chip of the pool it sits in
// and is billed at that pool's price, not at another pool's.
func TestInstancesAreBilledAtTheirOwnPoolsPrice(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "fast", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 1, MinNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80, CostPerHour: 20},
		{Name: "older", GPUType: "A100", GPUsPerNode: 8, InitialNodes: 1, MinNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80, CostPerHour: 12},
	}
	cfg := deploymentForPlacement(4, pools, 9999)
	cfg.TP = 4
	cs := newTestCluster(cfg, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 4 {
		t.Fatalf("placed %d instances, want 4", len(cs.instances))
	}
	want := []struct {
		gpu  string
		cost float64
	}{{"H100", 20}, {"H100", 20}, {"A100", 12}, {"A100", 12}}
	for i, w := range want {
		inst := cs.instances[i]
		if inst.GPU() != w.gpu || inst.CostPerHour != w.cost {
			t.Errorf("instance %d: chip %q at %v/h, want %q at %v/h", i, inst.GPU(), inst.CostPerHour, w.gpu, w.cost)
		}
	}
}
