// infra_placement_test.go — BDD/TDD tests for bin-packing placement (US2).
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// ─── US2: Place Instances onto Nodes Using Bin-Packing ──────────────────────

// T019: 2 TP=4 instances on 8-GPU node — both placed, 0 free GPUs, INV-A holds.
func TestPlacement_TwoTP4InstancesOnEightGPUNode(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})

	var nodeID string
	for id := range pm.nodesByID {
		nodeID = id
	}

	// Place first instance
	nid1, gpus1, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 4)
	if err != nil {
		t.Fatalf("first PlaceInstance failed: %v", err)
	}
	if nid1 != nodeID {
		t.Errorf("first instance placed on node %q, want %q", nid1, nodeID)
	}
	if len(gpus1) != 4 {
		t.Errorf("first instance got %d GPUs, want 4", len(gpus1))
	}

	// Place second instance
	nid2, gpus2, _, err := pm.PlaceInstance("inst-1", "model-a", "H100", 4)
	if err != nil {
		t.Fatalf("second PlaceInstance failed: %v", err)
	}
	if nid2 != nodeID {
		t.Errorf("second instance placed on node %q, want %q", nid2, nodeID)
	}
	if len(gpus2) != 4 {
		t.Errorf("second instance got %d GPUs, want 4", len(gpus2))
	}

	t.Run("node has 0 free GPUs after two placements", func(t *testing.T) {
		if free := pm.FreeGPUCount(nodeID); free != 0 {
			t.Errorf("FreeGPUCount = %d, want 0", free)
		}
	})

	t.Run("no GPU assigned to both instances", func(t *testing.T) {
		set1 := make(map[string]struct{}, len(gpus1))
		for _, g := range gpus1 {
			set1[g] = struct{}{}
		}
		for _, g := range gpus2 {
			if _, conflict := set1[g]; conflict {
				t.Errorf("GPU %q assigned to both instances", g)
			}
		}
	})

	t.Run("INV-A holds", func(t *testing.T) {
		if err := pm.VerifyConservation(); err != nil {
			t.Errorf("VerifyConservation: %v", err)
		}
	})
}

// T020: Fully allocated node causes new instance to enter Scheduling (pending) state.
func TestPlacement_FullNodeCausesError(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})

	// Fill the node
	_, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 4)
	if err != nil {
		t.Fatalf("first PlaceInstance: %v", err)
	}
	_, _, _, err = pm.PlaceInstance("inst-1", "model-a", "H100", 4)
	if err != nil {
		t.Fatalf("second PlaceInstance: %v", err)
	}

	// Now the node is full
	_, _, _, err = pm.PlaceInstance("inst-2", "model-a", "H100", 4)
	if err == nil {
		t.Error("PlaceInstance on full node should return error, got nil")
	}

	t.Run("FreeGPUCount still 0 (no partial allocation, R5)", func(t *testing.T) {
		for id := range pm.nodesByID {
			if free := pm.FreeGPUCount(id); free != 0 {
				t.Errorf("FreeGPUCount = %d after failed placement, want 0", free)
			}
		}
	})

	t.Run("INV-A holds after failed placement", func(t *testing.T) {
		if err := pm.VerifyConservation(); err != nil {
			t.Errorf("VerifyConservation: %v", err)
		}
	})
}

// T021: ReleaseInstance returns GPUs to free pool; INV-A holds.
func TestPlacement_ReleaseReturnsGPUs(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})

	var nodeID string
	for id := range pm.nodesByID {
		nodeID = id
	}

	// Place and release
	_, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 4)
	if err != nil {
		t.Fatalf("PlaceInstance: %v", err)
	}

	if err := pm.ReleaseInstance("inst-0"); err != nil {
		t.Fatalf("ReleaseInstance: %v", err)
	}

	t.Run("FreeGPUCount restored to 8", func(t *testing.T) {
		if free := pm.FreeGPUCount(nodeID); free != 8 {
			t.Errorf("FreeGPUCount = %d, want 8", free)
		}
	})

	t.Run("second placement succeeds after release", func(t *testing.T) {
		_, _, _, err := pm.PlaceInstance("inst-1", "model-a", "H100", 4)
		if err != nil {
			t.Errorf("PlaceInstance after release failed: %v", err)
		}
	})

	t.Run("INV-A holds", func(t *testing.T) {
		if err := pm.VerifyConservation(); err != nil {
			t.Errorf("VerifyConservation: %v", err)
		}
	})
}

// T022: GPU type constraint — H100 instance not placed on A100 node.
func TestPlacement_GPUTypeConstraint(t *testing.T) {
	newMixedPM := func() *PlacementManager {
		return newTestPM([]NodePoolConfig{
			newTestPool("h100-pool", "H100", 8, 2),
			newTestPool("a100-pool", "A100", 4, 2),
		})
	}

	t.Run("H100 instance placed only on H100 node", func(t *testing.T) {
		pm := newMixedPM()
		nodeID, _, _, err := pm.PlaceInstance("inst-h", "model-a", "H100", 4)
		if err != nil {
			t.Fatalf("PlaceInstance H100: %v", err)
		}
		node := pm.nodesByID[nodeID]
		if node.GPUType != "H100" {
			t.Errorf("H100 instance placed on %s node, want H100", node.GPUType)
		}
		if node.PoolName != "h100-pool" {
			t.Errorf("H100 instance placed in pool %q, want h100-pool", node.PoolName)
		}
	})

	t.Run("A100 instance placed only on A100 node", func(t *testing.T) {
		pm := newMixedPM()
		nodeID, _, _, err := pm.PlaceInstance("inst-a", "model-b", "A100", 4)
		if err != nil {
			t.Fatalf("PlaceInstance A100: %v", err)
		}
		node := pm.nodesByID[nodeID]
		if node.GPUType != "A100" {
			t.Errorf("A100 instance placed on %s node, want A100", node.GPUType)
		}
		if node.PoolName != "a100-pool" {
			t.Errorf("A100 instance placed in pool %q, want a100-pool", node.PoolName)
		}
	})

	t.Run("H100 instance fails if no H100 capacity", func(t *testing.T) {
		pm := newMixedPM()
		// Fill all H100 slots: 2 nodes × 8 GPUs / TP=4 = 4 instances
		for i := 0; i < 4; i++ {
			id := InstanceID("inst-fill-h-" + string(rune('0'+i)))
			_, _, _, err := pm.PlaceInstance(id, "model-a", "H100", 4)
			if err != nil {
				t.Fatalf("filling H100 slots, placement %d failed: %v", i, err)
			}
		}
		_, _, _, err := pm.PlaceInstance("inst-overflow", "model-a", "H100", 4)
		if err == nil {
			t.Error("PlaceInstance on full H100 pool should fail, got nil")
		}
	})
}

// T005: AddPending stores the simCfg field in pendingInstance.
func TestAddPending_StoresSimCfg(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		{
			Name:         "h100-pool",
			GPUType:      "H100",
			GPUsPerNode:  8,
			MaxNodes:     4,
			GPUMemoryGiB: 80.0,
		},
	})

	sentinel := sim.SimConfig{}
	sentinel.MaxRunningReqs = 7 // recognizable sentinel value

	pm.AddPending("inst-0", "model-a", "H100", 4, sentinel)

	if len(pm.pendingInsts) != 1 {
		t.Fatalf("expected 1 pending instance, got %d", len(pm.pendingInsts))
	}
	if pm.pendingInsts[0].simCfg.MaxRunningReqs != 7 {
		t.Errorf("pendingInsts[0].simCfg.MaxRunningReqs = %d, want 7", pm.pendingInsts[0].simCfg.MaxRunningReqs)
	}
}

// T003: RetryPendingInstances populates gpuType on the placedInstance from the matched pool.
func TestRetryPendingInstances_PlacedInstanceHasGPUType(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		{
			Name:         "h100-prov",
			GPUType:      "H100",
			GPUsPerNode:  8,
			MaxNodes:     4,
			GPUMemoryGiB: 80.0,
		},
	})

	// Add pending instance before any node is ready
	pm.AddPending("inst-0", "model-a", "H100", 4, sim.SimConfig{})

	// Provision and ready a node
	node, _ := pm.ProvisionNode("h100-prov", 0)
	if err := pm.MarkNodeReady(node.ID); err != nil {
		t.Fatalf("MarkNodeReady: %v", err)
	}

	// Retry should now place the instance
	placed := pm.RetryPendingInstances()
	if len(placed) != 1 {
		t.Fatalf("RetryPendingInstances() placed %d instances, want 1", len(placed))
	}

	if placed[0].gpuType != "H100" {
		t.Errorf("placed[0].gpuType = %q, want %q", placed[0].gpuType, "H100")
	}
}

// T001: PlaceInstance returns matchedGPUType equal to the pool's gpu_type on success,
// and returns "" on error (no capacity).
func TestPlaceInstance_ReturnsMatchedPoolGPUType(t *testing.T) {
	tests := []struct {
		name             string
		poolGPUType      string
		requestGPUType   string
		tpDegree         int
		initialNodes     int
		wantMatchedGPU   string
		wantErr          bool
	}{
		{
			name:           "success path returns pool gpu type",
			poolGPUType:    "A100",
			requestGPUType: "A100",
			tpDegree:       4,
			initialNodes:   1,
			wantMatchedGPU: "A100",
			wantErr:        false,
		},
		{
			name:           "error path returns empty string",
			poolGPUType:    "A100",
			requestGPUType: "A100",
			tpDegree:       4,
			initialNodes:   0, // no nodes — no capacity
			wantMatchedGPU: "",
			wantErr:        true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pm := newTestPM([]NodePoolConfig{newTestPool("a100-pool", tc.poolGPUType, 8, tc.initialNodes)})
			if tc.initialNodes > 0 {
				// Ensure node is ready (newTestPM creates Ready nodes)
			}
			_, _, matchedGPUType, err := pm.PlaceInstance("inst-0", "model-a", tc.requestGPUType, tc.tpDegree)
			if tc.wantErr && err == nil {
				t.Errorf("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if matchedGPUType != tc.wantMatchedGPU {
				t.Errorf("matchedGPUType = %q, want %q", matchedGPUType, tc.wantMatchedGPU)
			}
		})
	}
}

// T022b: RetryPendingInstances places pending instances after a node becomes ready.
func TestPlacement_RetryPendingAfterNodeReady(t *testing.T) {
	// Start with no nodes
	pm := newTestPM([]NodePoolConfig{
		{
			Name:         "prov-pool",
			GPUType:      "H100",
			GPUsPerNode:  8,
			MaxNodes:     4,
			GPUMemoryGiB: 80.0,
		},
	})

	// Add an instance to pending
	pm.AddPending("inst-0", "model-a", "H100", 4, sim.SimConfig{})
	if len(pm.pendingInsts) != 1 {
		t.Fatalf("expected 1 pending instance, got %d", len(pm.pendingInsts))
	}

	// Provision a node
	node, _ := pm.ProvisionNode("prov-pool", 0)
	_ = pm.MarkNodeReady(node.ID)

	// Retry
	placed := pm.RetryPendingInstances()
	if len(placed) != 1 {
		t.Errorf("RetryPendingInstances() placed %d instances, want 1", len(placed))
	}
	if len(pm.pendingInsts) != 0 {
		t.Errorf("pendingInsts len = %d, want 0 after successful retry", len(pm.pendingInsts))
	}

	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation: %v", err)
	}
}
