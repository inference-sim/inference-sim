package sim

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNetworkTopology_ZeroValueIsInert verifies BC-4: the zero value reports an
// unknown topology and a single-node span for every group size, so a build with
// no node-pool placement behaves exactly as it did before #1530 (INV-6).
func TestNetworkTopology_ZeroValueIsInert(t *testing.T) {
	var topo NetworkTopology
	assert.False(t, topo.IsKnown(), "zero value must report an unknown topology")
	for _, group := range []int{0, 1, 2, 8, 16, 1024} {
		assert.Equal(t, 1, topo.NodesSpanned(group),
			"unknown topology must never span more than one node (group=%d)", group)
	}
	require.NoError(t, topo.validate())
}

// TestNewNetworkTopology_ClampsNegative verifies BC-8: a negative node size is
// normalized to the inert unknown state rather than producing a negative divisor.
func TestNewNetworkTopology_ClampsNegative(t *testing.T) {
	topo := NewNetworkTopology(-4)
	assert.False(t, topo.IsKnown())
	assert.Equal(t, 1, topo.NodesSpanned(16))
	assert.Equal(t, 0, topo.PlacedGPUsPerNode)
	require.NoError(t, topo.validate())
}

// TestNetworkTopology_NodesSpanned verifies the span arithmetic (BC-5's
// arithmetic half) and BC-8's degenerate cases: a group that fits on one node
// never spans, a group larger than a node spans the ceiling, and an unknown
// topology always reports 1.
func TestNetworkTopology_NodesSpanned(t *testing.T) {
	tests := []struct {
		name            string
		gpusPerNode     int
		groupSize       int
		wantNodesSpan   int
		wantMembersNode int
	}{
		{"unknown topology, big group", 0, 16, 1, 16},
		{"group fits exactly", 8, 8, 1, 8},
		{"group smaller than node", 8, 4, 1, 4},
		{"group spans two nodes", 8, 16, 2, 8},
		{"group is four nodes", 4, 16, 4, 4},
		{"non-divisible group rounds up", 6, 16, 3, 6},
		{"single-GPU nodes", 1, 4, 4, 1},
		{"zero group", 8, 0, 1, 0},
		{"negative group", 8, -3, 1, 0},
		{"group of one", 8, 1, 1, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			topo := NewNetworkTopology(tc.gpusPerNode)
			assert.Equal(t, tc.wantNodesSpan, topo.NodesSpanned(tc.groupSize), "NodesSpanned")
			assert.Equal(t, tc.wantMembersNode, topo.MembersPerNode(tc.groupSize), "MembersPerNode")
		})
	}
}

// TestNetworkTopology_SpanTimesMembersCoversGroup verifies a conservation law
// rather than a table of values: the nodes spanned, times the members that share
// each node, must be enough to hold the whole group (and no more than one node's
// worth of slack). This holds for every (node size, group size) pair, so it would
// survive a rewrite of the ceiling arithmetic.
func TestNetworkTopology_SpanTimesMembersCoversGroup(t *testing.T) {
	for gpusPerNode := 1; gpusPerNode <= 16; gpusPerNode++ {
		topo := NewNetworkTopology(gpusPerNode)
		for group := 1; group <= 64; group++ {
			span := topo.NodesSpanned(group)
			perNode := topo.MembersPerNode(group)
			assert.GreaterOrEqual(t, span*perNode, group,
				"span(%d)×perNode(%d) must cover group=%d (gpusPerNode=%d)", span, perNode, group, gpusPerNode)
			assert.Less(t, (span-1)*perNode, group,
				"span must be minimal: (span-1)×perNode must not already cover group=%d (gpusPerNode=%d)", group, gpusPerNode)
		}
	}
}

// TestNetworkTopology_Validate verifies the validation boundary: only a
// hand-built negative value (bypassing the canonical constructor) is rejected.
func TestNetworkTopology_validate(t *testing.T) {
	require.NoError(t, NetworkTopology{PlacedGPUsPerNode: 0}.validate())
	require.NoError(t, NetworkTopology{PlacedGPUsPerNode: 8}.validate())
	err := NetworkTopology{PlacedGPUsPerNode: -1}.validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "PlacedGPUsPerNode")
}

// TestNewModelHardwareConfig_RejectsNegativeTopology verifies the validation has a
// caller: a hand-built negative node size supplied through the option is rejected at
// config construction, where the surrounding panics for MaxModelLen and DP already
// live, rather than silently producing a meaningless node span.
func TestNewModelHardwareConfig_RejectsNegativeTopology(t *testing.T) {
	mc := ModelConfig{NumLayers: 4, HiddenDim: 256, NumHeads: 4, NumKVHeads: 4, BytesPerParam: 2, IntermediateDim: 512}
	// Match on the field name rather than the whole message, so a reword of the error
	// does not break the test (the sibling validation tests use Contains for the same
	// reason).
	defer func() {
		r := recover()
		require.NotNil(t, r, "a negative node size supplied through the option must panic")
		assert.Contains(t, fmt.Sprint(r), "PlacedGPUsPerNode")
	}()
	NewModelHardwareConfig(mc, HardwareCalib{}, "m", "H100", 1, 1, false, "", "trained-physics", 0,
		WithNetworkTopology(NetworkTopology{PlacedGPUsPerNode: -8}))

}

// TestNewModelHardwareConfig_CanonicalConstructorNormalizes verifies the other half: the
// canonical NewNetworkTopology normalizes a negative node size instead of panicking, so
// the same intent expressed through it is accepted and simply inert.
func TestNewModelHardwareConfig_CanonicalConstructorNormalizes(t *testing.T) {
	mc := ModelConfig{NumLayers: 4, HiddenDim: 256, NumHeads: 4, NumKVHeads: 4, BytesPerParam: 2, IntermediateDim: 512}
	assert.NotPanics(t, func() {
		cfg := NewModelHardwareConfig(mc, HardwareCalib{}, "m", "H100", 1, 1, false, "", "trained-physics", 0,
			WithNetworkTopology(NewNetworkTopology(-8)))
		assert.False(t, cfg.NetworkTopology.IsKnown())
	})
}
