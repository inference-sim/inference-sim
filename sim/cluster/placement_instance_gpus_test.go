package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// An instance is the whole parallel group its engine runs, so it reserves, spans and is billed
// for tp x pp x pcp x dp GPUs, not tp of them (inference-sim#2260). The laws below hold for every
// layout: an instance never shares a node past its capacity, the GPUs a node pool holds for an
// instance are the GPUs its layout needs, and the instance's tensor-parallel degree stays tp.

func TestGPUsPerInstance(t *testing.T) {
	for _, c := range []struct {
		name            string
		tp, pp, pcp, dp int
		want            int
	}{
		{"tp only", 4, 1, 1, 1, 4},
		{"tp4 x dp2", 4, 1, 1, 2, 8},
		{"tp8 x dp1", 8, 1, 1, 1, 8},
		{"tp2 x pp2 x pcp2 x dp2", 2, 2, 2, 2, 16},
		{"unset widths count one", 0, 0, 0, 0, 1},
		{"unset dp counts one", 4, 0, 0, 0, 4},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := sim.GPUsPerInstance(c.tp, c.pp, c.pcp, c.dp); got != c.want {
				t.Errorf("GPUsPerInstance(%d,%d,%d,%d) = %d, want %d", c.tp, c.pp, c.pcp, c.dp, got, c.want)
			}
		})
	}
}

func TestInstanceConfig_GPUFootprint(t *testing.T) {
	for _, c := range []struct {
		name     string
		tp, gpus int
		want     int
	}{
		{"stated count wins", 4, 8, 8},
		{"unstated takes tp", 4, 0, 4},
		{"nothing stated is one GPU", 0, 0, 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := (sim.InstanceConfig{TP: c.tp, GPUs: c.gpus}).GPUFootprint(); got != c.want {
				t.Errorf("GPUFootprint() = %d, want %d", got, c.want)
			}
		})
	}
}

func b200Pool(initialNodes int) []NodePoolConfig {
	return []NodePoolConfig{{Name: "b200", GPUType: "B200", GPUsPerNode: 8, InitialNodes: initialNodes,
		MinNodes: 0, MaxNodes: 4, GPUMemoryGiB: 178, CostPerHour: 30}}
}

// Two instances of 8 GPUs each fit an 8-GPU node once, however the 8 are split between tp and dp:
// tp8 x dp1 and tp4 x dp2 place one instance and defer the other, while tp4 x dp1 (4 GPUs) places
// both. This is the issue's repro.
func TestStartupPlacement_ReservesTheInstancesWholeLayout(t *testing.T) {
	for _, c := range []struct {
		name       string
		tp, dp     int
		wantPlaced int
		wantGPUs   int
	}{
		{"tp8 dp1 takes the node", 8, 1, 1, 8},
		{"tp4 dp2 takes the node as tp8 does", 4, 2, 1, 8},
		{"tp2 dp4 takes the node as tp8 does", 2, 4, 1, 8},
		{"tp4 dp1 shares the node", 4, 1, 2, 4},
		{"tp2 dp2 shares the node", 2, 2, 2, 4},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := deploymentForPlacement(2, b200Pool(1), 9999)
			cfg.TP = c.tp
			cfg.GPUs = sim.GPUsPerInstance(c.tp, 1, 1, c.dp)
			cs := newTestCluster(cfg, NewSliceRequestSource(nil), nil)

			if len(cs.instances) != c.wantPlaced {
				t.Fatalf("placed %d instance(s), want %d", len(cs.instances), c.wantPlaced)
			}
			if got := len(cs.placement.pendingInsts); got != 2-c.wantPlaced {
				t.Errorf("%d instance(s) deferred, want %d", got, 2-c.wantPlaced)
			}
			total := 0
			for _, inst := range cs.instances {
				if got := len(inst.allocatedGPUIDs); got != c.wantGPUs {
					t.Errorf("%s holds %d GPUs, want %d", inst.ID(), got, c.wantGPUs)
				}
				if inst.TPDegree != c.tp {
					t.Errorf("%s TPDegree = %d, want the tensor-parallel degree %d", inst.ID(), inst.TPDegree, c.tp)
				}
				total += len(inst.allocatedGPUIDs)
			}
			if total > 8 {
				t.Errorf("instances hold %d GPUs of an 8-GPU pool", total)
			}
			if err := cs.placement.VerifyConservation(); err != nil {
				t.Errorf("VerifyConservation: %v", err)
			}
		})
	}
}

// A layout wider than a node spans whole nodes, and is billed and recorded for each: tp4 x dp4 is
// 16 GPUs, two 8-GPU nodes at the pool's rate each. Before the fix it reserved 4 GPUs on one node.
func TestStartupPlacement_NodeSpanAndCostFollowTheLayout(t *testing.T) {
	for _, c := range []struct {
		name     string
		tp, dp   int
		nodes    int
		wantCost float64
	}{
		{"tp4 dp2 is one node", 4, 2, 1, 30},
		{"tp4 dp4 is two nodes", 4, 4, 2, 60},
		{"tp8 dp2 is two nodes", 8, 2, 2, 60},
		{"tp16 dp1 is two nodes", 16, 1, 2, 60},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := deploymentForPlacement(1, b200Pool(2), 9999)
			cfg.TP = c.tp
			cfg.GPUs = sim.GPUsPerInstance(c.tp, 1, 1, c.dp)
			cs := newTestCluster(cfg, NewSliceRequestSource(nil), nil)
			if len(cs.instances) != 1 {
				t.Fatalf("placed %d instance(s), want 1", len(cs.instances))
			}
			if got := cs.MaxNodesSpanned(); got != c.nodes {
				t.Errorf("node span = %d, want %d", got, c.nodes)
			}
			if got := cs.instances[0].CostPerHour; got != c.wantCost {
				t.Errorf("CostPerHour = %v, want %v (%d node(s) x 30)", got, c.wantCost, c.nodes)
			}
		})
	}
}

// With no GPUs stated an instance takes tp GPUs, as every cluster built without the build site's
// layout does.
func TestStartupPlacement_UnstatedGPUsTakeTP(t *testing.T) {
	cfg := deploymentForPlacement(2, b200Pool(1), 9999)
	cfg.TP = 4
	cs := newTestCluster(cfg, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 2 {
		t.Fatalf("placed %d instance(s), want 2", len(cs.instances))
	}
	for _, inst := range cs.instances {
		if got := len(inst.allocatedGPUIDs); got != 4 {
			t.Errorf("%s holds %d GPUs, want 4", inst.ID(), got)
		}
	}
}

// A deferred instance keeps its whole layout: placed once a node is ready, it holds tp x dp GPUs
// and still reports tp as its tensor-parallel degree.
func TestDeferredPlacement_ReservesTheInstancesWholeLayout(t *testing.T) {
	cfg := deploymentForPlacement(2, b200Pool(0), 9999)
	cfg.TP = 4
	cfg.GPUs = 8
	cs := newTestCluster(cfg, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 0 {
		t.Fatalf("precondition: %d instance(s) placed with no node ready", len(cs.instances))
	}
	node, _ := cs.placement.ProvisionNode("b200", 0)
	(&NodeReadyEvent{timestamp: 0, nodeID: node.ID}).Execute(cs)

	if len(cs.instances) != 1 {
		t.Fatalf("placed %d instance(s) on one ready 8-GPU node, want 1 (8 GPUs each)", len(cs.instances))
	}
	inst := cs.instances[0]
	if got := len(inst.allocatedGPUIDs); got != 8 {
		t.Errorf("holds %d GPUs, want 8", got)
	}
	if inst.TPDegree != 4 {
		t.Errorf("TPDegree = %d, want 4", inst.TPDegree)
	}
	if err := cs.placement.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// A disaggregated deployment places each pool's instances at that pool's own layout: two prefill
// instances of tp2 x dp2 (4 GPUs) share one 8-GPU node and a decode instance of tp4 x dp2 (8 GPUs)
// takes the other.
func TestStartupPlacement_PDPoolsReserveTheirOwnLayout(t *testing.T) {
	cfg := newTestDisaggDeploymentConfig(3, 2, 1)
	cfg.NodePools = b200Pool(2)
	prefill, decode := cfg.InstanceConfig, cfg.InstanceConfig
	prefill.TP, prefill.GPUs = 2, sim.GPUsPerInstance(2, 1, 1, 2)
	decode.TP, decode.GPUs = 4, sim.GPUsPerInstance(4, 1, 1, 2)
	cfg.Roles.Prefill, cfg.Roles.Decode = &prefill, &decode
	cs := newTestCluster(cfg, NewSliceRequestSource(nil), nil)

	held := map[string]int{}
	for _, inst := range cs.instances {
		held[string(inst.ID())] = len(inst.allocatedGPUIDs)
	}
	want := map[string]int{"instance_0": 4, "instance_1": 4, "instance_2": 8}
	if len(held) != len(want) {
		t.Fatalf("placed %v, want %v", held, want)
	}
	for id, n := range want {
		if held[id] != n {
			t.Errorf("%s holds %d GPUs, want %d (all: %v)", id, held[id], n, held)
		}
	}
	if err := cs.placement.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}

// The autoscaler's free-GPU count subtracts the GPUs an instance holds, not its tensor-parallel
// degree: a tp4 x dp2 instance leaves no GPU of an 8-GPU node free.
func TestGPUInventory_CountsTheGPUsAnInstanceHolds(t *testing.T) {
	cfg := deploymentForPlacement(1, b200Pool(1), 9999)
	cfg.TP = 4
	cfg.GPUs = 8
	cs := newTestCluster(cfg, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 1 {
		t.Fatalf("placed %d instance(s), want 1", len(cs.instances))
	}
	inv := cs.gpuInventory()
	for _, v := range inv.Variants() {
		if free := inv.FreeSlots(v); free != 0 {
			t.Errorf("%+v has %d free GPU(s), want 0: the instance holds all 8", v, free)
		}
	}
}
