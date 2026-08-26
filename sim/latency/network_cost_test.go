package latency

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/require"
)

// newNetModel builds a trained-physics model through the production factory
// (NewLatencyModel + WithNetworkConfig) so the inter-node option-wiring and
// validation are exercised, then returns the concrete type for StepTime.
func newNetModel(t *testing.T, mc *sim.ModelConfig, tp, dp int, nc sim.NetworkConfig) *TrainedPhysicsModel {
	t.Helper()
	mhw := sim.NewModelHardwareConfig(*mc, dpepTestHW(), "m", "H100", tp, dp, false, "", "trained-physics", 0)
	lm, err := NewLatencyModel(*testCoeffs(), mhw, WithNetworkConfig(nc))
	require.NoError(t, err)
	m, ok := lm.(*TrainedPhysicsModel)
	require.True(t, ok, "expected *TrainedPhysicsModel")
	return m
}

// TestInterNodeCost_BC1_CrossNodeCharged is AC1: for an otherwise-identical config,
// a topology that spans a node boundary yields a strictly larger StepTime than one
// contained within a single node — for BOTH delivered legs (TP all-reduce and DEP
// all-to-all).
func TestInterNodeCost_BC1_CrossNodeCharged(t *testing.T) {
	batch := dpepMixedBatch() // has prefill + decode tokens → comm terms nonzero

	t.Run("TP all-reduce leg (dense, tp>gpus-per-node)", func(t *testing.T) {
		dense := testModelConfig()
		// tp=8 spanning 2 nodes of 4 vs contained in a node of 8.
		spanning := newNetModel(t, &dense, 8, 1, sim.NewNetworkConfig(4, 50.0, 0)).StepTime(batch)
		contained := newNetModel(t, &dense, 8, 1, sim.NewNetworkConfig(8, 50.0, 0)).StepTime(batch)
		inert := newNetModel(t, &dense, 8, 1, sim.NetworkConfig{}).StepTime(batch)
		require.Greater(t, spanning, contained, "cross-node TP all-reduce must cost strictly more than single-node")
		require.Equal(t, inert, contained, "a group that fits within one node must charge zero cross-node cost")
	})

	t.Run("DEP all-to-all leg (MoE, moeGroup>gpus-per-node)", func(t *testing.T) {
		moe := dpepMoEModelConfig()
		// tp=4, dp=2 → moeGroup=8. gpus-per-node=4: TP (4) fits, moeGroup (8) spans → isolates the dispatch leg.
		spanning := newNetModel(t, moe, 4, 2, sim.NewNetworkConfig(4, 50.0, 0)).StepTime(batch)
		contained := newNetModel(t, moe, 4, 2, sim.NewNetworkConfig(8, 50.0, 0)).StepTime(batch)
		inert := newNetModel(t, moe, 4, 2, sim.NetworkConfig{}).StepTime(batch)
		require.Greater(t, spanning, contained, "cross-node DEP all-to-all must cost strictly more than single-node")
		require.Equal(t, inert, contained, "a moeGroup that fits within one node must charge zero cross-node cost")
	})
}

// TestInterNodeCost_BC2_MonotonicFabricQuality is AC2: holding topology fixed,
// decreasing inter-node bandwidth OR increasing inter-node latency never decreases
// the charged cost (and strictly increases it while comm volume is nonzero).
func TestInterNodeCost_BC2_MonotonicFabricQuality(t *testing.T) {
	batch := dpepMixedBatch()
	moe := dpepMoEModelConfig()

	// Bandwidth: worse fabric (lower GB/s) ⇒ higher StepTime.
	fast := newNetModel(t, moe, 4, 2, sim.NewNetworkConfig(4, 100.0, 0)).StepTime(batch)
	mid := newNetModel(t, moe, 4, 2, sim.NewNetworkConfig(4, 50.0, 0)).StepTime(batch)
	slow := newNetModel(t, moe, 4, 2, sim.NewNetworkConfig(4, 25.0, 0)).StepTime(batch)
	require.Greater(t, slow, mid, "lower inter-node bandwidth must not decrease cost")
	require.Greater(t, mid, fast, "lower inter-node bandwidth must not decrease cost")

	// Latency: higher base latency ⇒ higher StepTime, holding bandwidth fixed.
	noLat := newNetModel(t, moe, 4, 2, sim.NewNetworkConfig(4, 50.0, 0)).StepTime(batch)
	someLat := newNetModel(t, moe, 4, 2, sim.NewNetworkConfig(4, 50.0, 0.01)).StepTime(batch)
	require.Greater(t, someLat, noLat, "higher inter-node latency must not decrease cost")
}

// TestInterNodeCost_BC3_NoOpByteIdentical is AC3 / INV-6 / INV-BC-DP1: any config
// that stays within one node — the inert default, and a declared topology whose
// groups fit — produces StepTime byte-identical to a model built with NO network
// option at all (the pre-feature value). Covers a matrix of dense/MoE × TP × DP.
func TestInterNodeCost_BC3_NoOpByteIdentical(t *testing.T) {
	dense := testModelConfig()
	moe := dpepMoEModelConfig()
	batch := dpepMixedBatch()

	cases := []struct {
		name string
		mc   *sim.ModelConfig
		tp   int
		dp   int
	}{
		{"dense tp1 dp1 (INV-BC-DP1)", &dense, 1, 1},
		{"dense tp2 dp1 (INV-BC-DP1)", &dense, 2, 1},
		{"dense tp8 dp1 (INV-BC-DP1)", &dense, 8, 1},
		{"moe tp2 dp1", moe, 2, 1},
		{"moe tp4 dp2", moe, 4, 2},
		{"moe tp8 dp1", moe, 8, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Reference: no network option applied at all (pre-feature construction).
			ref := newDPEPModel(t, tc.mc, tc.tp, tc.dp, false, "").StepTime(batch)

			// Inert network config (default zero value) must be byte-identical.
			inert := newNetModel(t, tc.mc, tc.tp, tc.dp, sim.NetworkConfig{}).StepTime(batch)
			require.Equal(t, ref, inert, "inert network config must be byte-identical to no-network")

			// A declared topology whose every group fits in one node (gpus-per-node
			// >= tp*dp) must also be byte-identical — the term is 0 unless a boundary
			// is crossed, even when the network model is "active".
			fits := newNetModel(t, tc.mc, tc.tp, tc.dp, sim.NewNetworkConfig(64, 50.0, 0.01)).StepTime(batch)
			require.Equal(t, ref, fits, "a topology that fits in one node must charge zero cross-node cost")
		})
	}
}

// TestInterNodeCost_BC4_Deterministic is AC4 / INV-6: the same config produces the
// same StepTime on repeated calls (the term is a pure function of config + tokens).
func TestInterNodeCost_BC4_Deterministic(t *testing.T) {
	batch := dpepMixedBatch()
	moe := dpepMoEModelConfig()
	nc := sim.NewNetworkConfig(4, 37.5, 0.003)
	first := newNetModel(t, moe, 4, 2, nc).StepTime(batch)
	for i := 0; i < 5; i++ {
		require.Equal(t, first, newNetModel(t, moe, 4, 2, nc).StepTime(batch), "StepTime must be deterministic")
	}
}

// TestInterNodeCost_BC6_BackendAndConfigGuards is BC-6 / R1: an active network
// config on a non-trained-physics backend is a hard error, and an invalid fabric
// config is rejected at construction — never a silent no-op.
func TestInterNodeCost_BC6_BackendAndConfigGuards(t *testing.T) {
	dense := testModelConfig()

	// Active network config on the roofline backend → error.
	roof := sim.NewModelHardwareConfig(dense, dpepTestHW(), "m", "H100", 8, 1, false, "", "roofline", 0)
	_, err := NewLatencyModel(*testCoeffs(), roof, WithNetworkConfig(sim.NewNetworkConfig(4, 50.0, 0)))
	require.Error(t, err, "active network config on roofline backend must error")

	// Inert network config on roofline is fine (no cross-node cost requested).
	_, err = NewLatencyModel(*testCoeffs(), roof, WithNetworkConfig(sim.NetworkConfig{}))
	require.NoError(t, err, "inert network config must not affect the roofline backend")

	// Invalid fabric config (active but zero bandwidth) → error at construction.
	tp := sim.NewModelHardwareConfig(dense, dpepTestHW(), "m", "H100", 8, 1, false, "", "trained-physics", 0)
	_, err = NewLatencyModel(*testCoeffs(), tp, WithNetworkConfig(sim.NetworkConfig{GPUsPerNode: 4}))
	require.Error(t, err, "active network config with zero bandwidth must error")
}
