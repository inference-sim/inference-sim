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

// TestStepTime_MoEReduceLegChargedCrossNode verifies the DP==1 MoE-FFN reduce is priced
// cross-node too. Isolating it needs care: it flows through the same TP-group basis as the
// attention and dense-FFN all-reduces, so the naive check "spanning MoE > single-node MoE"
// would pass even if this leg alone were skipped — the attention all-reduce is penalized
// regardless.
//
// The sharp test is an EQUALITY. At DP=1 the MoE-FFN reduce replaces the dense-FFN reduce
// unit for unit (a uniform MoE model has numDenseLayers=0 and numMoELayers=numLayers,
// where a dense model has the reverse), so both models charge exactly 2·numLayers
// all-reduce units and must therefore gain exactly the same amount from spanning. If the
// MoE-FFN reduce were left unpenalized, the MoE model would be charged only numLayers
// units of penalty and would come out strictly CHEAPER — which is precisely the mutation
// this equality catches.
func TestStepTime_MoEReduceLegChargedCrossNode(t *testing.T) {
	hw := fabricHW(9)
	batch := stepBatch()

	moe := *dpepMoEModelConfig()
	// Same layers/hidden/heads/FFN as the MoE config, but dense (no experts), so the only
	// structural difference is which all-reduce unit the FFN contributes.
	dense := moe
	dense.NumLocalExperts = 0
	dense.NumExpertsPerTok = 0

	penalty := func(mc sim.ModelConfig) int64 {
		return newNetModel(t, mc, hw, 8, 1, false, "", 4).StepTime(batch) -
			newNetModel(t, mc, hw, 8, 1, false, "", 8).StepTime(batch)
	}
	moePenalty, densePenalty := penalty(moe), penalty(dense)

	assert.Greater(t, densePenalty, int64(0), "precondition: spanning must cost the dense model something")
	assert.Equal(t, densePenalty, moePenalty,
		"an MoE model at DP=1 all-reduces its MoE FFN in place of a dense FFN, so its spanning penalty "+
			"must equal an identically-shaped dense model's. A strictly smaller MoE penalty would mean "+
			"the MoE-FFN reduce is not being priced cross-node (moe=%d µs, dense=%d µs)",
		moePenalty, densePenalty)
}

// TestStepTime_CommFamilyDeterminesPenaltyShape guards the single most load-bearing
// modeling decision in this feature: WHICH penalty shape each MoE comm family gets.
//
// The all-gather family (vLLM's default) moves its volume as a ring — two phases at
// (G-1)/G efficiency, exactly like an all-reduce — so it takes the ring penalty, where
// only the reduced inter-node chunk crosses the fabric. A true all-to-all has no
// reduction, so a far larger share of every rank's egress leaves the node and it takes
// the per-peer penalty. At TP·DP=16 spanning two nodes at a 9x fabric ratio those differ
// by ~3.4x, so getting them backwards would badly over-price the default backend.
//
// Asserting only "spanning costs more" would NOT catch a swap: both shapes exceed 1. So
// this compares the two families against each other at identical group, span and fabric.
func TestStepTime_CommFamilyDeterminesPenaltyShape(t *testing.T) {
	mc := *dpepMoEModelConfig()
	hw := fabricHW(9)
	batch := stepBatch()

	// moeGroup = TP·DP = 4·4 = 16, spanning four 4-GPU nodes.
	ringFamily := newNetModel(t, mc, hw, 4, 4, true, "allgather_reducescatter", 4).StepTime(batch)
	a2aFamily := newNetModel(t, mc, hw, 4, 4, true, "deepep_high_throughput", 4).StepTime(batch)

	// Contained baselines isolate the cross-node penalty from the families' differing
	// byte volumes (all-gather moves dense hidden states; all-to-all moves top_k tokens).
	ringContained := newNetModel(t, mc, hw, 4, 4, true, "allgather_reducescatter", 16).StepTime(batch)
	a2aContained := newNetModel(t, mc, hw, 4, 4, true, "deepep_high_throughput", 16).StepTime(batch)

	ringPenalty := ringFamily - ringContained
	a2aPenalty := a2aFamily - a2aContained
	assert.Greater(t, ringPenalty, int64(0), "precondition: the ring family must be penalized at all")
	assert.Greater(t, a2aPenalty, ringPenalty,
		"a true all-to-all sends more of its data off-node than a ring, so it must be penalized more "+
			"for the same span; equal or inverted penalties mean the two comm families were given the "+
			"wrong collective shapes (ring=%d µs, all-to-all=%d µs)", ringPenalty, a2aPenalty)
}

// ─── The size-independent half: per-collective latency ──────────────────────

// TestStepTime_PerCollectiveLatencyIsChargedCrossNode verifies the second half of the
// cross-node cost: a fixed launch + fabric round-trip per collective that crosses a node
// boundary, independent of message size. A fabric as fast as the on-node link (ratio 1,
// no bandwidth penalty at all) isolates it — any increase must come from the latency.
func TestStepTime_PerCollectiveLatencyIsChargedCrossNode(t *testing.T) {
	mc := testModelConfig()
	batch := stepBatch()

	noLatency := fabricHW(1) // equal bandwidths ⇒ zero bandwidth penalty
	withLatency := noLatency
	withLatency.InterNodeLatencyUs = 5

	contained := newNetModel(t, mc, withLatency, 8, 1, false, "", 8).StepTime(batch)
	spanning := newNetModel(t, mc, withLatency, 8, 1, false, "", 4).StepTime(batch)
	assert.Greater(t, spanning, contained,
		"a per-collective latency must be charged when the TP group spans nodes, even with no "+
			"bandwidth penalty")

	// And with no latency declared, the same pair is byte-identical — the term is opt-in.
	assert.Equal(t,
		newNetModel(t, mc, noLatency, 8, 1, false, "", 8).StepTime(batch),
		newNetModel(t, mc, noLatency, 8, 1, false, "", 4).StepTime(batch),
		"with neither a bandwidth penalty nor a declared latency, spanning must cost nothing extra")
}

// TestStepTime_MonotoneInPerCollectiveLatency verifies AC-2's latency clause directly:
// holding the placement fixed, raising the per-collective latency never lowers step time,
// and strictly raises it across a realistic range.
func TestStepTime_MonotoneInPerCollectiveLatency(t *testing.T) {
	mc := testModelConfig()
	batch := stepBatch()

	prev := int64(0)
	for _, latencyUs := range []float64{0, 1, 2, 5, 10, 25} {
		hw := fabricHW(9)
		hw.InterNodeLatencyUs = latencyUs
		got := newNetModel(t, mc, hw, 8, 1, false, "", 4).StepTime(batch)
		assert.GreaterOrEqual(t, got, prev,
			"step time must not decrease as the per-collective latency rises (latency=%v µs)", latencyUs)
		prev = got
	}
	zeroLatency := func() int64 {
		hw := fabricHW(9)
		return newNetModel(t, mc, hw, 8, 1, false, "", 4).StepTime(batch)
	}()
	assert.Greater(t, prev, zeroLatency, "the largest latency must cost strictly more than none")
}

// TestStepTime_PerCollectiveLatencyScalesWithCollectiveCount verifies the latency is
// charged PER COLLECTIVE rather than once per step: a model with twice the layers runs
// twice the collectives and must pay about twice the latency. This is what distinguishes
// a per-collective cost from a flat per-step one, and it is why the term can dominate for
// a deep model on small messages.
func TestStepTime_PerCollectiveLatencyScalesWithCollectiveCount(t *testing.T) {
	batch := stepBatch()
	hw := fabricHW(1) // no bandwidth penalty — isolate the latency
	hw.InterNodeLatencyUs = 20

	shallow := testModelConfig()
	deep := shallow
	deep.NumLayers = shallow.NumLayers * 2

	penalty := func(mc sim.ModelConfig) int64 {
		return newNetModel(t, mc, hw, 8, 1, false, "", 4).StepTime(batch) -
			newNetModel(t, mc, hw, 8, 1, false, "", 8).StepTime(batch)
	}
	shallowPenalty, deepPenalty := penalty(shallow), penalty(deep)
	assert.Greater(t, shallowPenalty, int64(0), "precondition: the shallow model must pay a latency penalty")
	assert.Greater(t, deepPenalty, shallowPenalty,
		"twice the layers means twice the cross-node collectives, so the latency penalty must grow "+
			"(shallow=%d µs, deep=%d µs)", shallowPenalty, deepPenalty)
	// Roughly proportional: within 10% of 2x, confirming per-collective and not per-step.
	ratio := float64(deepPenalty) / float64(shallowPenalty)
	assert.InDelta(t, 2.0, ratio, 0.2, "the latency penalty should scale with the collective count")
}

// TestStepTime_NoLatencyChargedWithoutTokens verifies a step that communicates nothing
// pays no launch cost: with no tokens there is no collective to launch.
func TestStepTime_NoLatencyChargedWithoutTokens(t *testing.T) {
	mc := testModelConfig()
	hw := fabricHW(9)
	hw.InterNodeLatencyUs = 1000 // enormous, so any spurious charge would be obvious

	spanning := newNetModel(t, mc, hw, 8, 1, false, "", 4)
	contained := newNetModel(t, mc, hw, 8, 1, false, "", 8)
	assert.Equal(t, contained.StepTime(nil), spanning.StepTime(nil),
		"an empty batch runs no collective, so no launch cost may be charged")
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

	// And it must equal what an explicitly un-penalized model computes — asserted
	// against a sibling model rather than a re-implementation of the formula, so a
	// legitimate refactor of the basis does not break this guard.
	unpenalized := &TrainedPhysicsModel{
		tp: 8, hiddenDim: 4096, activationBPP: 2, bwHbmUs: 3.35e6, tpSpanScale: 1.0,
	}
	assert.Equal(t, unpenalized.tpAllReduceBasis(32, 1024), basis,
		"an unset span scale must behave exactly like an explicit no-penalty scale")
}

// ─── Hot-path guard ─────────────────────────────────────────────────────────

// BenchmarkTrainedPhysicsStepTime measures the path this feature modifies, which had no
// benchmark before. The three variants isolate what the feature costs: `inert` is the
// default every run without node pools takes, `spanning_bandwidth` adds the size-dependent
// penalty, and `spanning_bandwidth_latency` adds the size-independent one too. The
// penalties are frozen at construction, so the inert case should be indistinguishable
// from a pre-feature build and the spanning cases should differ only by a comparison and
// a division.
func BenchmarkTrainedPhysicsStepTime(b *testing.B) {
	mc := testModelConfig()
	batch := stepBatch()

	build := func(hw sim.HardwareCalib, gpusPerNode int) sim.LatencyModel {
		mhw := sim.NewModelHardwareConfig(mc, hw, "m", "H100", 8, 1, false, "", "trained-physics", 0,
			sim.WithNetworkTopology(sim.NewNetworkTopology(gpusPerNode)))
		lm, err := NewLatencyModel(*testCoeffs(), mhw)
		if err != nil {
			b.Fatal(err)
		}
		return lm
	}
	withLatency := fabricHW(9)
	withLatency.InterNodeLatencyUs = 5

	for _, variant := range []struct {
		name string
		lm   sim.LatencyModel
	}{
		{"inert", build(dpepTestHW(), 0)},
		{"spanning_bandwidth", build(fabricHW(9), 4)},
		{"spanning_bandwidth_latency", build(withLatency, 4)},
	} {
		b.Run(variant.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = variant.lm.StepTime(batch)
			}
		})
	}
}

// TestStepTime_SpanningPathAllocatesNothing asserts what the benchmark only reports: the
// cross-node path adds no heap allocation to StepTime. Both penalties are frozen at
// construction and the per-collective latency is a scalar add, so the spanning path should
// allocate exactly as much as the inert one — nothing. Asserting it rather than leaving it
// as a comment matters because StepTime runs once per simulated scheduler step, and an
// accidental allocation there would be invisible until someone profiled a long run.
func TestStepTime_SpanningPathAllocatesNothing(t *testing.T) {
	mc := testModelConfig()
	batch := stepBatch()
	withLatency := fabricHW(9)
	withLatency.InterNodeLatencyUs = 5

	for _, tc := range []struct {
		name        string
		hw          sim.HardwareCalib
		gpusPerNode int
	}{
		{"inert", dpepTestHW(), 0},
		{"spanning_bandwidth", fabricHW(9), 4},
		{"spanning_bandwidth_latency", withLatency, 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newNetModel(t, mc, tc.hw, 8, 1, false, "", tc.gpusPerNode)
			// Warm up once so any lazily-initialized state is not attributed to the measured runs.
			_ = m.StepTime(batch)
			allocs := testing.AllocsPerRun(50, func() { _ = m.StepTime(batch) })
			assert.Zero(t, allocs, "StepTime must not allocate on the %s path, got %.1f allocs/op", tc.name, allocs)
		})
	}
}
