// network_cost_test.go — inter-node network cost for cross-node collectives (#1530).
//
// The trained-physics comm bases divide byte volume by an EFFECTIVE link bandwidth:
// the on-package bwHbmUs for a collective contained in one node, or bwHbmUs scaled
// down by a cross-node penalty when the collective's participant group spans nodes.
// These tests verify the penalty's algebra from first principles, its monotonicity
// in fabric quality, its effect on StepTime through both comm legs, and — the tight
// regression guard — that every configuration expressible before #1530 is
// byte-identical.
package latency

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
)

// fabricHW returns a hardware calib with a calibrated interconnect: an on-node link
// `ratio` times faster than the fabric.
func fabricHW(ratio float64) sim.HardwareCalib {
	hw := dpepTestHW()
	hw.IntraNodeBwGBps = 450
	hw.InterNodeBwGBps = 450 / ratio
	return hw
}

// newNetModel builds a trained-physics model through NewLatencyModel (so the
// WithNetworkTopology option path is exercised) at the given parallelism, hardware
// and placed node size. gpusPerNode == 0 means "no node-pool placement" (inert).
func newNetModel(t *testing.T, mc sim.ModelConfig, hw sim.HardwareCalib, tp, dp int, ep bool, backend string, gpusPerNode int) *TrainedPhysicsModel {
	t.Helper()
	mhw := sim.NewModelHardwareConfig(mc, hw, "m", "H100", tp, dp, ep, backend, "trained-physics", 0,
		sim.WithNetworkTopology(sim.NewNetworkTopology(gpusPerNode)))
	lm, err := NewLatencyModel(*testCoeffs(), mhw)
	require.NoError(t, err)
	m, ok := lm.(*TrainedPhysicsModel)
	require.True(t, ok, "expected a TrainedPhysicsModel")
	return m
}

// ─── The penalty algebra ────────────────────────────────────────────────────

// TestSpanScale_NeutralCases verifies that the shared penalty form is exactly 1.0
// — no penalty, and therefore a bit-for-bit unchanged divisor — for every case in
// which nothing should be charged (BC-4, BC-8).
func TestSpanScale_NeutralCases(t *testing.T) {
	tests := []struct {
		name        string
		crossHops   int
		totalHops   int
		ratio       float64
		wantExactly float64
	}{
		{"no cross hops", 0, 15, 9.0, 1.0},
		{"negative cross hops", -1, 15, 9.0, 1.0},
		{"no total hops", 1, 0, 9.0, 1.0},
		{"negative total hops", 1, -3, 9.0, 1.0},
		{"neutral ratio", 1, 15, 1.0, 1.0},
		{"sub-unit ratio", 1, 15, 0.5, 1.0},
		{"zero ratio", 1, 15, 0.0, 1.0},
		{"NaN ratio", 1, 15, math.NaN(), 1.0},
		{"+Inf ratio", 1, 15, math.Inf(1), 1.0},
		{"-Inf ratio", 1, 15, math.Inf(-1), 1.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.wantExactly, spanScale(tc.crossHops, tc.totalHops, tc.ratio))
		})
	}
}

// TestSpanScale_MonotoneAndBounded verifies the two properties every consumer
// relies on: the penalty never shrinks as the fabric gets worse (BC-3), and it is
// always at least 1 — a spanning collective is never cheaper than an intra-node one.
func TestSpanScale_MonotoneAndBounded(t *testing.T) {
	prev := 1.0
	for _, ratio := range []float64{1, 1.5, 2, 4, 9, 12, 50} {
		got := spanScale(3, 15, ratio)
		assert.GreaterOrEqual(t, got, prev, "penalty must not shrink as the fabric worsens (ratio=%v)", ratio)
		assert.GreaterOrEqual(t, got, 1.0, "penalty must never make spanning cheaper than not spanning")
		prev = got
	}
	assert.Greater(t, prev, 1.0)
}

// TestRingSpanScale_MatchesHierarchicalDecomposition validates the closed form
// against the physics it was derived from, rather than against a captured number.
// A multi-node ring all-reduce is hierarchical: an intra-node reduce-scatter +
// all-gather over the g ranks on a node (2(g-1)/g of the volume on the fast link),
// then an inter-node all-reduce of the reduced 1/g chunk across the n nodes
// (2(n-1)/n of that chunk on the fabric). Normalizing by the flat single-node
// baseline 2(G-1)/G must reproduce ringSpanScale exactly.
func TestRingSpanScale_MatchesHierarchicalDecomposition(t *testing.T) {
	for _, gpusPerNode := range []int{1, 2, 4, 8} {
		for _, nodes := range []int{1, 2, 3, 4} {
			for _, ratio := range []float64{1, 2, 9, 25} {
				g := float64(gpusPerNode)
				n := float64(nodes)
				group := gpusPerNode * nodes
				bigG := float64(group)

				intraPhase := 2 * (g - 1) / g
				interPhase := 2 * (n - 1) / n / g * ratio
				baseline := 2 * (bigG - 1) / bigG
				var want float64
				if baseline == 0 { // G == 1: no collective at all
					want = 1.0
				} else {
					want = (intraPhase + interPhase) / baseline
				}

				topo := sim.NewNetworkTopology(gpusPerNode)
				got := ringSpanScale(topo, group, ratio)
				assert.InDelta(t, want, got, 1e-12,
					"gpusPerNode=%d nodes=%d ratio=%v", gpusPerNode, nodes, ratio)
			}
		}
	}
}

// TestRingSpanScale_NeutralAtEqualBandwidth verifies the physical sanity check the
// closed form makes explicit: if the fabric were as fast as the on-node link,
// spanning would cost nothing extra. This is what pins the formula's constant term
// — it is why the penalty is a pure function of (ratio-1).
func TestRingSpanScale_NeutralAtEqualBandwidth(t *testing.T) {
	for _, gpusPerNode := range []int{1, 2, 4, 8} {
		for _, group := range []int{1, 2, 4, 8, 16, 32} {
			topo := sim.NewNetworkTopology(gpusPerNode)
			assert.Equal(t, 1.0, ringSpanScale(topo, group, 1.0),
				"gpusPerNode=%d group=%d", gpusPerNode, group)
		}
	}
}

// TestRingSpanScale_SingleNodeIsNeutral verifies BC-4 at the formula level: a group
// that fits inside one node is charged nothing extra, for any fabric quality.
func TestRingSpanScale_SingleNodeIsNeutral(t *testing.T) {
	topo := sim.NewNetworkTopology(8)
	for _, group := range []int{1, 2, 4, 8} {
		for _, ratio := range []float64{1, 9, 100} {
			assert.Equal(t, 1.0, ringSpanScale(topo, group, ratio), "group=%d ratio=%v", group, ratio)
		}
	}
	// And an unknown topology is neutral even for a huge group.
	var unknown sim.NetworkTopology
	assert.Equal(t, 1.0, ringSpanScale(unknown, 64, 9))
}

// TestAll2AllSpanScale_MatchesPerPeerDecomposition validates the all-to-all closed
// form against its own derivation: a rank's egress splits over its G-1 peers, of
// which G-g are off-node and cost `ratio` times as much.
func TestAll2AllSpanScale_MatchesPerPeerDecomposition(t *testing.T) {
	for _, gpusPerNode := range []int{1, 2, 4, 8} {
		for _, nodes := range []int{1, 2, 4} {
			for _, ratio := range []float64{1, 2, 9} {
				group := gpusPerNode * nodes
				if group <= 1 {
					continue
				}
				g := float64(gpusPerNode)
				bigG := float64(group)
				want := ((g - 1) + ratio*(bigG-g)) / (bigG - 1)

				topo := sim.NewNetworkTopology(gpusPerNode)
				assert.InDelta(t, want, all2AllSpanScale(topo, group, ratio), 1e-12,
					"gpusPerNode=%d nodes=%d ratio=%v", gpusPerNode, nodes, ratio)
			}
		}
	}
}

// TestAll2AllSpanScale_ExceedsRingAtSameSpan verifies the qualitative claim that
// motivates modeling the two collective shapes separately: for the same group and
// the same span, an all-to-all pushes far more of its traffic across the fabric
// than a ring does, so it is penalized more. This is why the expert all-to-all —
// not the TP all-reduce — is the dominant cross-node cost for wide expert
// parallelism.
func TestAll2AllSpanScale_ExceedsRingAtSameSpan(t *testing.T) {
	topo := sim.NewNetworkTopology(8)
	const group, ratio = 16, 9.0
	ring := ringSpanScale(topo, group, ratio)
	a2a := all2AllSpanScale(topo, group, ratio)
	assert.Greater(t, ring, 1.0, "precondition: the ring must itself be penalized")
	assert.Greater(t, a2a, ring, "an all-to-all must be penalized more than a ring at the same span")
}

// ─── StepTime effects ───────────────────────────────────────────────────────

// stepBatch is a fixed mixed batch large enough that the comm terms are resolvable
// at microsecond granularity.
func stepBatch() []*sim.Request {
	return append(makePrefillBatch(4, 512), makeDecodeBatch(8, 1024)...)
}

// TestStepTime_SpanningCostsMoreThanSingleNode verifies BC-1 for the TP all-reduce
// leg: with the same model, TP degree, hardware and batch, an instance whose TP
// group spans two nodes has a strictly larger step time than one whose TP group
// fits on a single node. Nothing but the placed node size differs.
func TestStepTime_SpanningCostsMoreThanSingleNode(t *testing.T) {
	mc := testModelConfig() // dense
	hw := fabricHW(9)
	batch := stepBatch()

	singleNode := newNetModel(t, mc, hw, 8, 1, false, "", 8) // tp=8 on an 8-GPU node
	spanning := newNetModel(t, mc, hw, 8, 1, false, "", 4)   // tp=8 across two 4-GPU nodes

	got := singleNode.StepTime(batch)
	want := spanning.StepTime(batch)
	assert.Greater(t, want, got,
		"a TP group spanning two nodes must cost strictly more than the same group on one node")
}

// TestStepTime_MoreNodesCostMore verifies the penalty tracks the span, not just its
// existence: spreading the same TP group over more (smaller) nodes never costs
// less, and costs strictly more than the two-node span.
func TestStepTime_MoreNodesCostMore(t *testing.T) {
	mc := testModelConfig()
	hw := fabricHW(9)
	batch := stepBatch()

	prev := int64(0)
	for _, gpusPerNode := range []int{8, 4, 2, 1} { // 1, 2, 4, 8 nodes for tp=8
		got := newNetModel(t, mc, hw, 8, 1, false, "", gpusPerNode).StepTime(batch)
		assert.GreaterOrEqual(t, got, prev,
			"step time must not decrease as the same TP group spreads over more nodes (gpusPerNode=%d)", gpusPerNode)
		prev = got
	}
	single := newNetModel(t, mc, hw, 8, 1, false, "", 8).StepTime(batch)
	assert.Greater(t, prev, single, "the widest span must cost strictly more than the single-node placement")
}

// TestStepTime_MonotoneInFabricQuality verifies BC-3 (AC-2): holding the placement
// fixed, a slower fabric never lowers the step time, and over a realistic IB→slow
// range it strictly raises it.
func TestStepTime_MonotoneInFabricQuality(t *testing.T) {
	mc := testModelConfig()
	batch := stepBatch()

	prev := int64(0)
	for _, interBw := range []float64{450, 200, 100, 50, 25, 12.5, 5} {
		hw := dpepTestHW()
		hw.IntraNodeBwGBps = 450
		hw.InterNodeBwGBps = interBw
		got := newNetModel(t, mc, hw, 8, 1, false, "", 4).StepTime(batch)
		assert.GreaterOrEqual(t, got, prev,
			"step time must not decrease as the fabric gets slower (inter=%v GB/s)", interBw)
		prev = got
	}
	fastFabric := newNetModel(t, mc, fabricHW(1), 8, 1, false, "", 4).StepTime(batch)
	assert.Greater(t, prev, fastFabric, "the slowest fabric must cost strictly more than a fabric as fast as NVLink")
}

// TestStepTime_MoEReduceLegChargedCrossNode verifies the DP==1 MoE-FFN reduce is
// priced cross-node too. It flows through the same TP-group basis, so a spanning
// MoE instance at DP=1 must cost strictly more than a single-node one.
func TestStepTime_MoEReduceLegChargedCrossNode(t *testing.T) {
	mc := *dpepMoEModelConfig()
	hw := fabricHW(9)
	batch := stepBatch()

	single := newNetModel(t, mc, hw, 8, 1, false, "", 8).StepTime(batch)
	spanning := newNetModel(t, mc, hw, 8, 1, false, "", 4).StepTime(batch)
	assert.Greater(t, spanning, single,
		"a spanning MoE instance at DP=1 must pay a cross-node penalty on its MoE-FFN reduce")
}

// TestStepTime_MoEDispatchLegChargedCrossNode verifies BC-2: the expert
// dispatch/combine leg is priced cross-node when the flattened MoE group (TP·DP)
// does not fit inside one node. Exercised at the latency-model level because a
// node_pools + --dp>1 run is a fail-fast today (#1553); the placement-driven
// assertion for this leg lands with expert-parallel placement (#1548).
func TestStepTime_MoEDispatchLegChargedCrossNode(t *testing.T) {
	mc := *dpepMoEModelConfig()
	hw := fabricHW(9)
	batch := stepBatch()

	for _, backend := range []string{"allgather_reducescatter", "deepep_high_throughput"} {
		t.Run(backend, func(t *testing.T) {
			// moeGroup = TP·DP = 4·4 = 16. An 16-GPU node contains it; a 4-GPU node does not.
			contained := newNetModel(t, mc, hw, 4, 4, true, backend, 16).StepTime(batch)
			spanning := newNetModel(t, mc, hw, 4, 4, true, backend, 4).StepTime(batch)
			assert.Greater(t, spanning, contained,
				"an expert all-to-all/all-gather spanning nodes must cost strictly more than one contained in a node")
		})
	}
}

// ─── Inertness: the tight regression guard (BC-4, AC-3, INV-6, INV-BC-DP1) ──

// TestStepTime_NoTopologyIsByteIdentical verifies that omitting the option
// entirely — every configuration expressible before #1530 — leaves StepTime
// bit-identical, even when the hardware declares a calibrated interconnect.
func TestStepTime_NoTopologyIsByteIdentical(t *testing.T) {
	batch := stepBatch()
	for _, tc := range []struct {
		name string
		mc   sim.ModelConfig
		tp   int
		dp   int
		ep   bool
	}{
		{"dense tp1 dp1", testModelConfig(), 1, 1, false},
		{"dense tp8 dp1", testModelConfig(), 8, 1, false},
		{"moe tp8 dp1", *dpepMoEModelConfig(), 8, 1, false},
		{"moe tp4 dp4 ep", *dpepMoEModelConfig(), 4, 4, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hw := fabricHW(9)
			mhw := sim.NewModelHardwareConfig(tc.mc, hw, "m", "H100", tc.tp, tc.dp, tc.ep, "", "trained-physics", 0)

			noOption, err := NewLatencyModel(*testCoeffs(), mhw)
			require.NoError(t, err)
			mhwZero := sim.NewModelHardwareConfig(tc.mc, hw, "m", "H100", tc.tp, tc.dp, tc.ep, "", "trained-physics", 0,
				sim.WithNetworkTopology(sim.NetworkTopology{}))
			zeroTopology, err := NewLatencyModel(*testCoeffs(), mhwZero)
			require.NoError(t, err)

			assert.Equal(t, noOption.StepTime(batch), zeroTopology.StepTime(batch),
				"a zero-value topology must be indistinguishable from no option at all")
		})
	}
}

// TestStepTime_ContainedGroupIsByteIdentical verifies BC-4's second gate: with a
// KNOWN topology and a calibrated fabric, a collective that fits inside one node is
// still bit-identical to the unplaced model. This is the case that covers every
// existing node_pools config (single-node TP), so it is the guard that keeps #1529's
// placements from silently changing numbers.
func TestStepTime_ContainedGroupIsByteIdentical(t *testing.T) {
	batch := stepBatch()
	hw := fabricHW(9)
	for _, tc := range []struct {
		name        string
		mc          sim.ModelConfig
		tp          int
		gpusPerNode int
	}{
		{"tp1 on 8-GPU node", testModelConfig(), 1, 8},
		{"tp4 on 8-GPU node", testModelConfig(), 4, 8},
		{"tp8 on 8-GPU node", testModelConfig(), 8, 8},
		{"moe tp8 on 8-GPU node", *dpepMoEModelConfig(), 8, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			unplaced := newNetModel(t, tc.mc, hw, tc.tp, 1, false, "", 0)
			placed := newNetModel(t, tc.mc, hw, tc.tp, 1, false, "", tc.gpusPerNode)
			assert.Equal(t, unplaced.StepTime(batch), placed.StepTime(batch),
				"a collective contained in one node must not be charged anything extra")
		})
	}
}

// TestStepTime_UncalibratedFabricIsByteIdentical verifies BC-4's third gate: a
// spanning placement on hardware whose config declares no interconnect bandwidths
// is bit-identical to the pre-#1530 behavior. Cross-node cost requires calibration;
// it is never invented.
func TestStepTime_UncalibratedFabricIsByteIdentical(t *testing.T) {
	batch := stepBatch()
	mc := testModelConfig()
	bare := dpepTestHW() // no IntraNodeBwGBps / InterNodeBwGBps

	unplaced := newNetModel(t, mc, bare, 8, 1, false, "", 0)
	spanning := newNetModel(t, mc, bare, 8, 1, false, "", 4)
	assert.Equal(t, unplaced.StepTime(batch), spanning.StepTime(batch),
		"an uncalibrated interconnect must leave a spanning placement priced exactly as before")
}

// TestINVBCDP1_DenseDP1UnaffectedByTopology verifies INV-BC-DP1 explicitly: a dense
// DP=1 instance's step time is unchanged by the network feature across the TP
// matrix, whenever its TP group fits on a node.
func TestINVBCDP1_DenseDP1UnaffectedByTopology(t *testing.T) {
	batch := stepBatch()
	mc := testModelConfig()
	hw := fabricHW(9)
	for _, tp := range []int{1, 2, 4, 8} {
		unplaced := newNetModel(t, mc, hw, tp, 1, false, "", 0)
		placed := newNetModel(t, mc, hw, tp, 1, false, "", 8)
		assert.Equal(t, unplaced.StepTime(batch), placed.StepTime(batch), "tp=%d", tp)
	}
}

// TestStepTime_EmptyBatchUnaffected verifies the degenerate batch: no tokens means
// no collective, so the network feature cannot change the empty-batch floor.
func TestStepTime_EmptyBatchUnaffected(t *testing.T) {
	mc := testModelConfig()
	hw := fabricHW(9)
	unplaced := newNetModel(t, mc, hw, 8, 1, false, "", 0)
	spanning := newNetModel(t, mc, hw, 8, 1, false, "", 4)
	assert.Equal(t, unplaced.StepTime(nil), spanning.StepTime(nil))
	assert.Equal(t, unplaced.StepTime([]*sim.Request{}), spanning.StepTime([]*sim.Request{}))
}

// TestStepTime_HugeRatioStaysFinite verifies BC-8's robustness end: an absurdly
// slow fabric produces a large but finite, positive step time — never NaN, Inf, or
// a clock regression (INV-3).
func TestStepTime_HugeRatioStaysFinite(t *testing.T) {
	mc := testModelConfig()
	hw := dpepTestHW()
	hw.IntraNodeBwGBps = 450
	hw.InterNodeBwGBps = 1e-9 // pathologically slow fabric
	got := newNetModel(t, mc, hw, 8, 1, false, "", 4).StepTime(stepBatch())
	assert.Greater(t, got, int64(0), "step time must stay positive")
	assert.Less(t, got, int64(math.MaxInt64), "step time must stay below the clamp ceiling")
}

// ─── Backend scope ──────────────────────────────────────────────────────────

// TestRoofline_UnaffectedByNetworkTopology documents the deliberate backend
// asymmetry: the roofline backend models no communication term at all, so the
// cross-node penalty has nothing to scale and its step time is unchanged.
func TestRoofline_UnaffectedByNetworkTopology(t *testing.T) {
	mc := testModelConfig()
	hw := fabricHW(9)
	hw.MemoryGiB = 80
	mhw := sim.NewModelHardwareConfig(mc, hw, "m", "H100", 8, 1, false, "", "roofline", 0)
	mhwTopo := sim.NewModelHardwareConfig(mc, hw, "m", "H100", 8, 1, false, "", "roofline", 0,
		sim.WithNetworkTopology(sim.NewNetworkTopology(4)))
	batch := stepBatch()

	plain, err := NewLatencyModel(*testCoeffs(), mhw)
	require.NoError(t, err)
	withTopo, err := NewLatencyModel(*testCoeffs(), mhwTopo)
	require.NoError(t, err)

	assert.Equal(t, plain.StepTime(batch), withTopo.StepTime(batch),
		"roofline has no communication term, so a cross-node topology must not change its step time")
}

// ─── Calibration validation (R1: a typo must surface, not silently disable) ──

// TestNewTrainedPhysicsModel_RejectsUnusableInterconnectBw verifies that a value
// that was clearly meant to be a bandwidth but cannot be one is rejected, rather
// than being clamped into "no cross-node cost" where the user would never see it.
func TestNewTrainedPhysicsModel_RejectsUnusableInterconnectBw(t *testing.T) {
	tests := []struct {
		name  string
		intra float64
		inter float64
		want  string
	}{
		{"negative intra", -450, 50, "IntraNodeBwGBps"},
		{"NaN intra", math.NaN(), 50, "IntraNodeBwGBps"},
		{"Inf intra", math.Inf(1), 50, "IntraNodeBwGBps"},
		{"negative inter", 450, -50, "InterNodeBwGBps"},
		{"NaN inter", 450, math.NaN(), "InterNodeBwGBps"},
		{"Inf inter", 450, math.Inf(-1), "InterNodeBwGBps"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hw := dpepTestHW()
			hw.IntraNodeBwGBps = tc.intra
			hw.InterNodeBwGBps = tc.inter
			mhw := sim.NewModelHardwareConfig(testModelConfig(), hw, "m", "H100", 8, 1, false, "", "trained-physics", 0)
			_, err := NewLatencyModel(*testCoeffs(), mhw)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.want)
		})
	}
}

// TestNewTrainedPhysicsModel_RejectsHalfCalibratedInterconnect verifies that
// declaring one bandwidth without the other is rejected: it would produce no
// cross-node cost at all, which a user who bothered to set a value would not expect.
// Setting NEITHER stays valid — that is the normal inert case.
func TestNewTrainedPhysicsModel_RejectsHalfCalibratedInterconnect(t *testing.T) {
	for _, tc := range []struct {
		name         string
		intra, inter float64
	}{
		{"intra only", 450, 0},
		{"inter only", 0, 50},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hw := dpepTestHW()
			hw.IntraNodeBwGBps = tc.intra
			hw.InterNodeBwGBps = tc.inter
			mhw := sim.NewModelHardwareConfig(testModelConfig(), hw, "m", "H100", 8, 1, false, "", "trained-physics", 0)
			_, err := NewLatencyModel(*testCoeffs(), mhw)
			require.Error(t, err)
			assert.Contains(t, err.Error(), "incomplete")
		})
	}

	// Neither set: valid and inert.
	mhw := sim.NewModelHardwareConfig(testModelConfig(), dpepTestHW(), "m", "H100", 8, 1, false, "", "trained-physics", 0)
	_, err := NewLatencyModel(*testCoeffs(), mhw)
	require.NoError(t, err)
}

// TestTrainedPhysicsModel_StructLiteralKeepsCommTerm guards the one way this feature
// could silently DELETE cost instead of adding it: a model built by struct literal
// (as several tests in this package do) leaves the span scales at their zero value,
// and a divisor derived naively from a zero scale would be +Inf — zeroing the whole
// TP all-reduce term. The comm basis must stay positive for such a model.
func TestTrainedPhysicsModel_StructLiteralKeepsCommTerm(t *testing.T) {
	m := &TrainedPhysicsModel{
		tp:            8,
		hiddenDim:     4096,
		activationBPP: 2,
		bwHbmUs:       3.35e6,
		// tpSpanScale / moeSpanScale deliberately left at their zero value.
	}
	basis := m.tpAllReduceBasis(32, 1024)
	assert.Greater(t, basis, 0.0, "an unset span scale must not zero the TP all-reduce term")
	assert.False(t, math.IsInf(basis, 0), "the comm basis must stay finite")
	assert.False(t, math.IsNaN(basis), "the comm basis must not be NaN")

	// And it must equal the un-penalized value exactly.
	want := 32.0 * 1024.0 * 4096.0 * 2.0 * 2.0 * (7.0 / 8.0) / 3.35e6
	assert.Equal(t, want, basis)
}
