// infra_node_test.go — BDD/TDD tests for Phase 1A node/GPU infrastructure.
// Tests are organized by user story and written BEFORE implementation per the BLIS constitution.
package cluster

import (
	"math/rand"
	"strings"
	"testing"
)

// ─── Helpers ────────────────────────────────────────────────────────────────

// newTestPool creates a NodePoolConfig for testing.
func newTestPool(name, gpuType string, gpusPerNode, initialNodes int) NodePoolConfig {
	return NodePoolConfig{
		Name:         name,
		GPUType:      gpuType,
		GPUsPerNode:  gpusPerNode,
		GPUMemoryGiB: 80.0,
		InitialNodes: initialNodes,
		MinNodes:     0,
		MaxNodes:     initialNodes + 4,
	}
}

// newTestPM creates a PlacementManager with a fixed-seed RNG for determinism.
func newTestPM(pools []NodePoolConfig) *PlacementManager {
	rng := rand.New(rand.NewSource(42))
	return NewPlacementManager(pools, rng, rng, 0)
}

// ─── US1: Node-Pool Infrastructure Topology ─────────────────────────────────

// T011: Two-pool initialization produces correct node count and GPU inventory.
// GPU IDs must be unique and traceable to their pool.
func TestNodePool_TwoPoolInitialization(t *testing.T) {
	pools := []NodePoolConfig{
		newTestPool("h100-pool", "H100", 8, 2),
		newTestPool("a100-pool", "A100", 4, 1),
	}
	pm := newTestPM(pools)

	t.Run("node count", func(t *testing.T) {
		if got := pm.NodeCount(); got != 3 {
			t.Errorf("NodeCount() = %d, want 3", got)
		}
	})

	t.Run("GPU count per pool", func(t *testing.T) {
		if got := pm.GPUCount("h100-pool"); got != 16 {
			t.Errorf("GPUCount(h100-pool) = %d, want 16", got)
		}
		if got := pm.GPUCount("a100-pool"); got != 4 {
			t.Errorf("GPUCount(a100-pool) = %d, want 4", got)
		}
	})

	t.Run("all GPU IDs are globally unique", func(t *testing.T) {
		seen := make(map[string]struct{})
		for _, node := range pm.nodesByID {
			for _, gpu := range node.GPUs {
				if _, dup := seen[gpu.ID]; dup {
					t.Errorf("duplicate GPU ID %q", gpu.ID)
				}
				seen[gpu.ID] = struct{}{}
			}
		}
		if got := len(seen); got != 20 {
			t.Errorf("total distinct GPU IDs = %d, want 20", got)
		}
	})

	t.Run("GPU IDs encode pool of origin", func(t *testing.T) {
		for _, node := range pm.nodesByID {
			for _, gpu := range node.GPUs {
				if !strings.HasPrefix(gpu.ID, gpu.PoolName) {
					t.Errorf("GPU %q ID does not start with pool name %q", gpu.ID, gpu.PoolName)
				}
			}
		}
	})
}

// T012: VerifyConservation invariant test (INV-A companion).
func TestNodePool_VerifyConservation(t *testing.T) {
	t.Run("fresh pool passes conservation", func(t *testing.T) {
		pm := newTestPM([]NodePoolConfig{newTestPool("p", "H100", 8, 2)})
		if err := pm.VerifyConservation(); err != nil {
			t.Errorf("VerifyConservation() on fresh pool returned error: %v", err)
		}
	})

	t.Run("corrupted pool fails conservation", func(t *testing.T) {
		pm := newTestPM([]NodePoolConfig{newTestPool("p", "H100", 8, 1)})
		// Manually corrupt: mark a GPU as allocated without going through PlaceInstance
		for _, node := range pm.nodesByID {
			node.GPUs[0].AllocatedTo = "fake-instance"
			node.GPUs[1].AllocatedTo = "fake-instance"
			// Deliberately leave TotalGPUs unchanged — conservation should fail
			// because the sum of actual free + allocated still equals TotalGPUs
			// but we can test by corrupting TotalGPUs instead
			node.TotalGPUs = 100 // mismatch
			break
		}
		if err := pm.VerifyConservation(); err == nil {
			t.Error("VerifyConservation() on corrupted pool should return error, got nil")
		}
	})
}

// T013: InitialNodes=0 produces empty pool.
func TestNodePool_ZeroInitialNodes(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		{
			Name:         "empty-pool",
			GPUType:      "H100",
			GPUsPerNode:  8,
			GPUMemoryGiB: 80.0,
			InitialNodes: 0,
			MinNodes:     0,
			MaxNodes:     4,
		},
	})

	if got := pm.NodeCount(); got != 0 {
		t.Errorf("NodeCount() = %d, want 0 for InitialNodes=0", got)
	}
	if got := pm.GPUCount("empty-pool"); got != 0 {
		t.Errorf("GPUCount(empty-pool) = %d, want 0 for InitialNodes=0", got)
	}
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation() on empty pool: %v", err)
	}
}

// T014: Cross-pool GPU IDs are globally unique.
func TestNodePool_CrossPoolGPUUniqueness(t *testing.T) {
	// Same GPUsPerNode in both pools — potential ID collision risk if pool name not used
	pools := []NodePoolConfig{
		newTestPool("alpha", "H100", 4, 2),
		newTestPool("beta", "A100", 4, 2),
	}
	pm := newTestPM(pools)

	allIDs := make([]string, 0)
	for _, node := range pm.nodesByID {
		for _, gpu := range node.GPUs {
			allIDs = append(allIDs, gpu.ID)
		}
	}

	seen := make(map[string]struct{}, len(allIDs))
	for _, id := range allIDs {
		if _, dup := seen[id]; dup {
			t.Errorf("duplicate GPU ID across pools: %q", id)
		}
		seen[id] = struct{}{}
	}
	if len(seen) != len(allIDs) {
		t.Errorf("duplicate IDs detected: unique=%d total=%d", len(seen), len(allIDs))
	}
}

// ─── US3: Node Provisioning and Termination Lifecycle ───────────────────────

// T028: ProvisionNode schedules NodeReadyEvent at T+delay; Provisioning blocks placement.
func TestNodeProvisioning_BlocksPlacementUntilReady(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		{
			Name:              "prov-pool",
			GPUType:           "H100",
			GPUsPerNode:       8,
			GPUMemoryGiB:      80.0,
			InitialNodes:      0,
			MinNodes:          0,
			MaxNodes:          4,
			ProvisioningDelay: DelaySpec{Mean: 120.0}, // 120 seconds = 120_000_000 ticks
		},
	})

	// Provision a node at clock T=0
	node, readyTime := pm.ProvisionNode("prov-pool", 0)

	t.Run("node starts in Provisioning state", func(t *testing.T) {
		if node.State != NodeStateProvisioning {
			t.Errorf("newly provisioned node state = %q, want Provisioning", node.State)
		}
	})

	t.Run("ready time matches provisioning delay", func(t *testing.T) {
		wantReadyTime := int64(120.0 * 1e6) // 120 seconds in microseconds
		if readyTime != wantReadyTime {
			t.Errorf("readyTime = %d, want %d", readyTime, wantReadyTime)
		}
	})

	t.Run("placement fails on Provisioning node", func(t *testing.T) {
		_, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 4)
		if err == nil {
			t.Error("PlaceInstance() on Provisioning node should fail, got nil error")
		}
	})

	t.Run("placement succeeds after MarkNodeReady", func(t *testing.T) {
		if err := pm.MarkNodeReady(node.ID); err != nil {
			t.Fatalf("MarkNodeReady: %v", err)
		}
		if node.State != NodeStateReady {
			t.Errorf("after MarkNodeReady, node state = %q, want Ready", node.State)
		}
		_, gpuIDs, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 4)
		if err != nil {
			t.Errorf("PlaceInstance() after MarkNodeReady failed: %v", err)
		}
		if len(gpuIDs) != 4 {
			t.Errorf("got %d GPU IDs, want 4", len(gpuIDs))
		}
	})
}

// T029: Ready node with no instances drains immediately to Terminated.
func TestNodeDrain_EmptyNodeDrainsImmediately(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("p", "H100", 8, 1)})

	// Find the one node
	var nodeID string
	for id := range pm.nodesByID {
		nodeID = id
	}

	drained := false
	if err := pm.DrainNode(nodeID, func() { drained = true }); err != nil {
		t.Fatalf("DrainNode: %v", err)
	}

	t.Run("callback fires immediately (no instances)", func(t *testing.T) {
		if !drained {
			t.Error("drain callback was not called immediately for empty node")
		}
	})

	t.Run("node transitions to Draining", func(t *testing.T) {
		node := pm.nodesByID[nodeID]
		if node.State != NodeStateDraining {
			t.Errorf("node state after DrainNode = %q, want Draining", node.State)
		}
	})

	// Simulate NodeDrainedEvent executing
	if err := pm.MarkNodeTerminated(nodeID); err != nil {
		t.Fatalf("MarkNodeTerminated: %v", err)
	}

	t.Run("node transitions to Terminated", func(t *testing.T) {
		node := pm.nodesByID[nodeID]
		if node.State != NodeStateTerminated {
			t.Errorf("node state after MarkNodeTerminated = %q, want Terminated", node.State)
		}
	})
}

// T030: Draining node with instances only terminates after last instance releases.
func TestNodeDrain_WaitsForInstanceRelease(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("p", "H100", 8, 1)})

	var nodeID string
	for id := range pm.nodesByID {
		nodeID = id
	}

	// Place an instance
	_, _, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 4)
	if err != nil {
		t.Fatalf("PlaceInstance: %v", err)
	}

	drained := false
	if err := pm.DrainNode(nodeID, func() { drained = true }); err != nil {
		t.Fatalf("DrainNode: %v", err)
	}

	t.Run("callback not called while instance exists", func(t *testing.T) {
		if drained {
			t.Error("drain callback should NOT fire while instance is still allocated")
		}
	})

	t.Run("node is Draining not Terminated", func(t *testing.T) {
		node := pm.nodesByID[nodeID]
		if node.State != NodeStateDraining {
			t.Errorf("node state = %q, want Draining", node.State)
		}
	})

	// Release the instance
	if err := pm.ReleaseInstance("inst-0"); err != nil {
		t.Fatalf("ReleaseInstance: %v", err)
	}

	t.Run("callback fires after last instance released", func(t *testing.T) {
		if !drained {
			t.Error("drain callback should fire after last instance is released")
		}
	})

	// INV-A must hold after release
	if err := pm.VerifyConservation(); err != nil {
		t.Errorf("VerifyConservation() after drain: %v", err)
	}
}

// ─── NodePoolConfig validation ───────────────────────────────────────────────

func TestNodePoolConfig_Validation(t *testing.T) {
	cases := []struct {
		name    string
		cfg     NodePoolConfig
		wantErr bool
	}{
		{
			name:    "valid config",
			cfg:     newTestPool("p", "H100", 8, 2),
			wantErr: false,
		},
		{
			name:    "empty name",
			cfg:     NodePoolConfig{Name: "", GPUType: "H100", GPUsPerNode: 8, GPUMemoryGiB: 80, MaxNodes: 2},
			wantErr: true,
		},
		{
			name:    "zero gpus per node",
			cfg:     NodePoolConfig{Name: "p", GPUType: "H100", GPUsPerNode: 0, GPUMemoryGiB: 80, MaxNodes: 2},
			wantErr: true,
		},
		{
			name:    "zero gpu memory",
			cfg:     NodePoolConfig{Name: "p", GPUType: "H100", GPUsPerNode: 8, GPUMemoryGiB: 0, MaxNodes: 2},
			wantErr: true,
		},
		{
			name:    "initial > max nodes",
			cfg:     NodePoolConfig{Name: "p", GPUType: "H100", GPUsPerNode: 8, GPUMemoryGiB: 80, InitialNodes: 5, MaxNodes: 3},
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.IsValid()
			if (err != nil) != tc.wantErr {
				t.Errorf("IsValid() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}
