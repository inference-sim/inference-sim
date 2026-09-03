// network_topology_test.go — tests for the placement-derived inter-node network
// topology signal (#1530). placedGPUsPerNode reports the size of the node(s) an
// instance's GPUs actually occupy; the latency model turns that into a cross-node
// collective cost. See docs/guide/latency-models.md.
package cluster

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
)

// TestPlacedGPUsPerNode_SingleNodePlacement verifies the topology signal for a
// placement contained in one node: it reports the NODE's size (not the instance's
// GPU count), so a collective larger than the instance is still scored correctly.
func TestPlacedGPUsPerNode_SingleNodePlacement(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})
	_, gpus, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 4)
	require.NoError(t, err)

	assert.Equal(t, 8, pm.placedGPUsPerNode(gpus),
		"a 4-GPU instance on an 8-GPU node must report the node size, not the instance size")
}

// TestPlacedGPUsPerNode_SpanningPlacement verifies the signal for a whole-node
// span: every hosting node has the pool's size, so that size is reported.
func TestPlacedGPUsPerNode_SpanningPlacement(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 2)})
	_, gpus, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 16)
	require.NoError(t, err)

	assert.Equal(t, 8, pm.placedGPUsPerNode(gpus))
}

// TestPlacedGPUsPerNode_SpanAgreesWithRealNodeCount verifies BC-5, the law that
// ties the derived signal back to placement: for EVERY placement shape the
// topology's NodesSpanned(tpDegree) must equal the number of distinct nodes the
// instance's GPUs actually occupy. This is what makes the signal placement-derived
// rather than a declared guess — it would fail if the ceiling arithmetic and the
// real span ever disagreed.
func TestPlacedGPUsPerNode_SpanAgreesWithRealNodeCount(t *testing.T) {
	tests := []struct {
		name         string
		gpusPerNode  int
		initialNodes int
		tpDegree     int
		wantNodes    int
	}{
		{"fits on one node", 8, 2, 4, 1},
		{"exactly one node", 8, 2, 8, 1},
		{"spans two nodes", 8, 2, 16, 2},
		{"spans three nodes", 4, 3, 12, 3},
		{"spans four small nodes", 2, 4, 8, 4},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pm := newTestPM([]NodePoolConfig{newTestPool("p", "H100", tc.gpusPerNode, tc.initialNodes)})
			_, gpus, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", tc.tpDegree)
			require.NoError(t, err)

			realNodes := len(pm.distinctNodesForGPUs(gpus))
			require.Equal(t, tc.wantNodes, realNodes, "precondition: expected node span")

			topo := sim.NewNetworkTopology(pm.placedGPUsPerNode(gpus))
			assert.Equal(t, realNodes, topo.NodesSpanned(tc.tpDegree),
				"NodesSpanned(tp) must equal the real distinct-node count")
		})
	}
}

// TestPlacedGPUsPerNode_UnknownGPUsAreInert verifies BC-8/R1: an unresolvable GPU
// set yields 0 (topology unknown ⇒ the network cost stays inert) rather than a
// plausible-looking wrong node size.
func TestPlacedGPUsPerNode_UnknownGPUsAreInert(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})

	assert.Equal(t, 0, pm.placedGPUsPerNode(nil), "empty GPU set")
	assert.Equal(t, 0, pm.placedGPUsPerNode([]string{}), "empty GPU slice")
	assert.Equal(t, 0, pm.placedGPUsPerNode([]string{"no-such-gpu"}), "unknown GPU ID")

	topo := sim.NewNetworkTopology(pm.placedGPUsPerNode([]string{"no-such-gpu"}))
	assert.False(t, topo.IsKnown())
	assert.Equal(t, 1, topo.NodesSpanned(16), "an unknown topology must never charge a cross-node cost")
}

// TestPlacedGPUsPerNode_MixedNodeSizesAreInert verifies the defensive branch: a
// GPU set drawn from nodes of different sizes cannot be described by one
// gpus-per-node figure, so the signal degrades to "unknown" instead of picking one
// arbitrarily (R1). PlaceInstance never produces such a set — it spans whole nodes
// within a single pool — so this is constructed by hand.
func TestPlacedGPUsPerNode_MixedNodeSizesAreInert(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{
		newTestPool("big", "H100", 8, 1),
		newTestPool("small", "L40S", 2, 1),
	})
	bigGPU := pm.nodesByPool["big"][0].GPUs[0].ID
	smallGPU := pm.nodesByPool["small"][0].GPUs[0].ID

	assert.Equal(t, 0, pm.placedGPUsPerNode([]string{bigGPU, smallGPU}),
		"a span over differently-sized nodes must report an unknown topology")
}

// TestPlacedGPUsPerNode_ReleasedGPUsStillResolve documents that the signal depends
// only on the GPU→node inventory, not on allocation state: it is a topology query,
// so releasing an instance does not change what node its GPUs sat on.
func TestPlacedGPUsPerNode_ReleasedGPUsStillResolve(t *testing.T) {
	pm := newTestPM([]NodePoolConfig{newTestPool("h100", "H100", 8, 1)})
	_, gpus, _, err := pm.PlaceInstance("inst-0", "model-a", "H100", 8)
	require.NoError(t, err)
	require.NoError(t, pm.ReleaseInstance("inst-0"))

	assert.Equal(t, 8, pm.placedGPUsPerNode(gpus))
}
