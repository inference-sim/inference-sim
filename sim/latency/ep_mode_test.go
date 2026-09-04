package latency

// Behavioral contracts for the live expert-parallel MODE (#1548): what changes when
// --enable-expert-parallel is on, what must not change when it is off, and that the
// per-mode all-to-all profile is really the seam the backend selection travels through.
//
// The laws asserted here are relationships between two models that differ in exactly one
// input, never re-implementations of the step-time formula — so a legitimate refactor of
// the basis functions keeps them passing (refactor-survival).

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newEPModel builds a trained-physics model at (tp, dp, ep) with an explicit LOGICAL
// EP-group DP width, the way cmd stamps it onto a DP-as-placement replica. epGroupDP == 0
// omits the option entirely (the pre-#1548 construction).
func newEPModel(t *testing.T, mc sim.ModelConfig, tp, dp int, ep bool, backend string, epGroupDP int) *TrainedPhysicsModel {
	t.Helper()
	var opts []sim.ModelHardwareOption
	if epGroupDP > 0 {
		opts = append(opts, sim.WithExpertParallelGroupDP(epGroupDP))
	}
	mhw := sim.NewModelHardwareConfig(mc, dpepTestHW(), "m", "H100", tp, dp, ep, backend, "trained-physics", 0, opts...)
	m, err := NewTrainedPhysicsModel(*testCoeffs(), mhw)
	require.NoError(t, err)
	return m
}

// ─── AC-2: the toggle is no longer inert ────────────────────────────────────

// TestStepTime_EPModeIsLive is AC-2 / BC-2, the "no longer inert" pass condition: with
// every other input held fixed, toggling expert parallelism must change StepTime for a real
// MoE config. Two independent routes are asserted, because they exercise different halves of
// the change: the COLLECTIVE (a modular all-to-all moves top_k-routed tokens, not dense
// hidden states) and the WEIGHT footprint (a wider EP group puts fewer experts on each GPU).
func TestStepTime_EPModeIsLive(t *testing.T) {
	mc := *dpepMoEModelConfig()
	batch := stepBatch()

	for _, tc := range []struct {
		name      string
		tp        int
		backend   string
		epGroupDP int
	}{
		{"collective: tp8 dp1 on a modular all-to-all backend", 8, "deepep_high_throughput", 0},
		{"weights: tp2 replica of a logical dp4 EP group", 2, "", 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			off := newEPModel(t, mc, tc.tp, 1, false, tc.backend, tc.epGroupDP).StepTime(batch)
			on := newEPModel(t, mc, tc.tp, 1, true, tc.backend, tc.epGroupDP).StepTime(batch)
			assert.NotEqual(t, off, on,
				"--enable-expert-parallel must change MoE step time (EP off=%dµs, on=%dµs)", off, on)
		})
	}
}

// TestStepTime_EPOnAllGatherAtDP1EqualsAllReduce records — as a deliberate law, not an
// oversight — the one EP-on configuration whose step time does NOT move: vLLM's DEFAULT
// all-to-all backend at DP=1.
//
// It is a real property of the physics, not a gap in the wiring. allgather_reducescatter
// IS the ring all-reduce decomposition: dispatch all-gathers the dense hidden states and
// combine reduce-scatters them, so over a group of g the wire volume is
// 2·(g-1)/g·tokens·hidden either way. With β_EP defaulting to β₄ (the ≤10-coefficient
// default) the two terms are therefore numerically equal, and BLIS says so instead of
// inventing a difference.
//
// The toggle becomes observable through the three routes that carry real physics: a modular
// all-to-all backend (top_k volume — asserted above), a wider EP group than the TP group
// (weight sharding — asserted above and in
// TestStepTime_EPShardsExpertWeightsAcrossTheGroup), and a coefficient set that calibrates
// β_EP away from β₄.
//
// It doubles as the MUTUAL-EXCLUSIVITY guard for the two MoE-FFN comm terms, which is why it
// is an equality rather than a skipped case: the dispatch volume here equals the all-reduce
// volume, so if EP-on charged dispatch IN ADDITION to the reduction (the double-charge #1548
// exists to avoid) this would come out strictly greater, not equal.
func TestStepTime_EPOnAllGatherAtDP1EqualsAllReduce(t *testing.T) {
	mc := *dpepMoEModelConfig()
	batch := stepBatch()
	off := newEPModel(t, mc, 8, 1, false, "allgather_reducescatter", 0).StepTime(batch)
	on := newEPModel(t, mc, 8, 1, true, "allgather_reducescatter", 0).StepTime(batch)
	assert.Equal(t, off, on,
		"all-gather+reduce-scatter moves exactly the ring-all-reduce volume, so at DP=1 with "+
			"β_EP == β₄ the EP toggle is numerically neutral for this backend — a documented "+
			"property, not an inert toggle (see the other routes in this file)")
}

// TestStepTime_EPReplacesReduceWithDispatch pins the MECHANISM behind AC-2, so a future
// change cannot satisfy the "differs" assertion above by some unrelated route: at DP=1 the
// MoE-FFN communication switches from the TP all-reduce family to the dispatch/combine
// family. The discriminator is that the two families respond differently to the comm
// backend — an all-reduce does not read --moe-comm-backend at all, while dispatch/combine
// picks its byte volume from it (all-gather moves dense hidden states, a modular all-to-all
// moves top_k-routed tokens).
//
// So: with EP OFF, two different backends must give identical step time (no dispatch term
// exists to select). With EP ON they must differ (the term exists and the family matters).
func TestStepTime_EPReplacesReduceWithDispatch(t *testing.T) {
	mc := *dpepMoEModelConfig()
	batch := stepBatch()

	offAG := newEPModel(t, mc, 8, 1, false, "allgather_reducescatter", 0).StepTime(batch)
	offA2A := newEPModel(t, mc, 8, 1, false, "deepep_high_throughput", 0).StepTime(batch)
	assert.Equal(t, offAG, offA2A,
		"with EP off at DP=1 the MoE FFN all-reduces, so the comm backend must not matter")

	onAG := newEPModel(t, mc, 8, 1, true, "allgather_reducescatter", 0).StepTime(batch)
	onA2A := newEPModel(t, mc, 8, 1, true, "deepep_high_throughput", 0).StepTime(batch)
	assert.NotEqual(t, onAG, onA2A,
		"with EP on the MoE FFN dispatches/combines, so the comm backend must select the volume")
}

// TestStepTime_EPShardsExpertWeightsAcrossTheGroup is BC-3, the physics that makes expert
// parallelism worth deploying: EP-on holds num_experts/EP WHOLE experts per GPU instead of
// num_experts/TP tensor slices, so a WIDER logical EP group must reduce step time.
//
// It is asserted on a weight-dominated batch (a single decode token: no prefill compute,
// minimal KV) where the routed-expert weight term is the dominant cost, and across a
// SEQUENCE of group widths so the direction — monotone decrease — is the law rather than
// one magic pair.
func TestStepTime_EPShardsExpertWeightsAcrossTheGroup(t *testing.T) {
	mc := *dpepMoEModelConfig()
	// 64 experts so a group of 2/4/8 divides it without hitting the one-expert-per-rank
	// floor that would make the widths indistinguishable.
	mc.NumLocalExperts = 64
	batch := makeDecodeBatch(1, 64)

	prev := newEPModel(t, mc, 2, 1, true, "", 1).StepTime(batch) // EP group = TP = 2
	require.Positive(t, prev)
	for _, groupDP := range []int{2, 4, 8} {
		got := newEPModel(t, mc, 2, 1, true, "", groupDP).StepTime(batch)
		assert.Less(t, got, prev,
			"widening the EP group to TP·%d must put fewer experts on each GPU and so cost less "+
				"(previous=%dµs, got=%dµs)", groupDP, prev, got)
		prev = got
	}
}

// TestStepTime_EPDoesNotChangeExpertCompute is the counterpart law, and the one a naive
// implementation gets wrong: routed-expert COMPUTE is EP-mode-invariant. With EP on, the
// G-GPU group jointly processes the whole group's tokens, so per-GPU FLOPs land on the same
// value tensor-sharding gives — widening the EP group must NOT divide compute.
//
// The discriminator is the model's SENSITIVITY TO top_k, which makes the law exact rather
// than a magnitude bound. On the all-gather comm family (vLLM's default), `kEff` enters the
// step-time model in exactly one place — the routed-expert compute basis
// `tokens·kEff/moeGroup`. It does not enter the weight term (`numExperts/group`, no top_k)
// and it does not enter that family's dispatch volume (dense hidden states, no top_k).
//
// So d(step time)/d(kEff) is purely the compute term divided by whatever group scopes it:
//
//   - If compute is scoped by `moeGroup` (correct), that divisor is identical for both EP
//     widths, so the two sensitivities are EQUAL.
//   - If compute were scoped by the EP group (the defect), the sensitivities would differ by
//     the width ratio — here 4×.
//
// Nothing structural is asserted and there is no tolerance to tune beyond the ±1µs the int64
// step-time rounding allows.
func TestStepTime_EPDoesNotChangeExpertCompute(t *testing.T) {
	base := *dpepMoEModelConfig()
	base.NumLocalExperts = 8
	lowK, highK := base, base
	lowK.NumExpertsPerTok, highK.NumExpertsPerTok = 1, 4
	batch := makePrefillBatch(4, 1024)

	// epGroupDP 1 ⇒ EP group == TP == 2; epGroupDP 4 ⇒ EP group == 8. moeGroup is 2 in both.
	sensitivity := func(epGroupDP int) int64 {
		return newEPModel(t, highK, 2, 1, true, "", epGroupDP).StepTime(batch) -
			newEPModel(t, lowK, 2, 1, true, "", epGroupDP).StepTime(batch)
	}
	narrow, wide := sensitivity(1), sensitivity(4)

	require.Positive(t, narrow,
		"precondition: raising top_k must raise routed-expert compute, or this law tests nothing")
	assert.InDelta(t, narrow, wide, 1.0,
		"routed-expert COMPUTE must be scoped by moeGroup (identical for both EP widths), not by "+
			"the EP group: top_k sensitivity was %dµs at EP group 2 and %dµs at EP group 8 — a ratio "+
			"near the 4x width ratio means compute is being divided by the EP group", narrow, wide)
}

// ─── AC-6: the per-mode profile is the seam ─────────────────────────────────

// TestAll2AllProfile_SharedPlaceholderIsExact is AC-6's honesty condition. Every backend
// currently resolves to the same nominal profile, deliberately — differentiated DeepEP
// high-throughput vs low-latency curves need their own calibration (#1568). This test
// states that as a fact rather than leaving it to be discovered: the two DeepEP modes cost
// the SAME today, and commScale is exactly 1.0 so the placeholder cannot perturb any step
// time (INV-6).
//
// It is also the sentinel #1568 will trip: when the curves are populated, this test fails
// and must be replaced by a differentiation assertion.
func TestAll2AllProfile_SharedPlaceholderIsExact(t *testing.T) {
	for _, name := range ValidMoECommBackends {
		p, err := moeCommProfileFor(name)
		require.NoError(t, err)
		assert.Equal(t, 1.0, p.commScale,
			"backend %q must ship the exact nominal commScale placeholder until #1568 calibrates it", name)
	}

	mc := *dpepMoEModelConfig()
	batch := stepBatch()
	ht := newEPModel(t, mc, 8, 1, true, "deepep_high_throughput", 0).StepTime(batch)
	ll := newEPModel(t, mc, 8, 1, true, "deepep_low_latency", 0).StepTime(batch)
	assert.Equal(t, ht, ll,
		"DeepEP HT and LL share one placeholder cost today (#1568 differentiates them); if this "+
			"fails, the differentiation landed and this assertion must be replaced")
}

// TestAll2AllProfile_ScalesTheDispatchTerm proves the profile is WIRED, not merely stored —
// the point of AC-6 ("no re-plumbing when #1568 populates them"). It multiplies the field
// on an already-constructed model and asserts the dispatch-bearing step time moves, which
// no amount of table-filling can achieve if the multiplication site is missing.
func TestAll2AllProfile_ScalesTheDispatchTerm(t *testing.T) {
	mc := *dpepMoEModelConfig()
	batch := stepBatch()

	base := newEPModel(t, mc, 8, 1, true, "deepep_high_throughput", 0)
	nominal := base.StepTime(batch)

	scaled := newEPModel(t, mc, 8, 1, true, "deepep_high_throughput", 0)
	scaled.all2All.commScale = 100.0
	assert.Greater(t, scaled.StepTime(batch), nominal,
		"a larger per-mode commScale must raise the dispatch/combine cost — the profile has to be "+
			"multiplied into the basis, not just resolved and stored")

	// And it must reach ONLY the dispatch term: with EP off (no dispatch term) the same
	// scale must be inert.
	offBase := newEPModel(t, mc, 8, 1, false, "deepep_high_throughput", 0)
	offScaled := newEPModel(t, mc, 8, 1, false, "deepep_high_throughput", 0)
	offScaled.all2All.commScale = 100.0
	assert.Equal(t, offBase.StepTime(batch), offScaled.StepTime(batch),
		"commScale must not reach any term other than MoE dispatch/combine")
}

// TestMoECommProfileFor_UnknownIsHardError is R1: an unrecognized backend must not resolve
// to a nominal-looking profile.
func TestMoECommProfileFor_UnknownIsHardError(t *testing.T) {
	_, err := moeCommProfileFor("deepep_medium_throughput")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "deepep_medium_throughput")
}

// ─── BC-4 / INV-6: everything else is byte-identical ────────────────────────

// TestStepTime_EPGroupDPIsInertWithoutEP is the tight regression guard: the new option must
// change nothing unless expert parallelism is actually on. A dense model ignores it
// outright (EP is MoE-only), and an EP-OFF MoE model must be bit-for-bit unchanged however
// wide a group width is supplied.
func TestStepTime_EPGroupDPIsInertWithoutEP(t *testing.T) {
	batch := stepBatch()
	for _, tc := range []struct {
		name string
		mc   sim.ModelConfig
		ep   bool
	}{
		{"dense, EP off", testModelConfig(), false},
		{"MoE, EP off", *dpepMoEModelConfig(), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			noOption := newEPModel(t, tc.mc, 8, 1, tc.ep, "", 0).StepTime(batch)
			for _, groupDP := range []int{1, 2, 16} {
				assert.Equal(t, noOption, newEPModel(t, tc.mc, 8, 1, tc.ep, "", groupDP).StepTime(batch),
					"WithExpertParallelGroupDP(%d) must be inert here", groupDP)
			}
		})
	}
}

// TestStepTime_DenseEPGroupDPIsInert covers the dense case explicitly against INV-BC-DP1:
// a dense model must be byte-identical whatever is passed, even with the EP flag set (the
// CLI rejects that combination, but the latency model must not depend on the CLI for it).
func TestStepTime_DenseEPGroupDPIsInert(t *testing.T) {
	mc := testModelConfig()
	batch := stepBatch()
	base := newEPModel(t, mc, 8, 1, false, "", 0).StepTime(batch)
	assert.Equal(t, base, newEPModel(t, mc, 8, 1, true, "", 8).StepTime(batch),
		"expert parallelism has no meaning for a dense model, so step time must not move (INV-BC-DP1)")
}

// TestTpAllReduceBasis_LatencyIsNotSharedAcrossDPRanks pins the delegated #1530 fix: the
// per-collective launch latency is charged in full per DP rank, while the byte VOLUME is
// divided. Halving the volume divisor must therefore NOT halve a latency-dominated basis.
func TestTpAllReduceBasis_LatencyIsNotSharedAcrossDPRanks(t *testing.T) {
	mc := *dpepMoEModelConfig()
	hw := fabricHW(9)
	hw.InterNodeLatencyUs = 1000 // latency-dominated, so the two halves are separable
	// A 4-GPU node cannot contain the TP=8 group, so the cross-node latency is charged.
	mhw := sim.NewModelHardwareConfig(mc, hw, "m", "H100", 8, 1, false, "", "trained-physics", 0,
		sim.WithNetworkTopology(sim.NewNetworkTopology(4)))
	m, err := NewTrainedPhysicsModel(*testCoeffs(), mhw)
	require.NoError(t, err)
	require.Positive(t, m.tpCrossNodeLatencyUs, "precondition: the cross-node latency must be charged")

	at1 := m.tpAllReduceBasis(32, 1024, 1)
	at4 := m.tpAllReduceBasis(32, 1024, 4)
	assert.Less(t, at4, at1, "a larger volume divisor must reduce the basis")
	assert.Greater(t, at4, at1/4,
		"the per-collective launch latency must NOT be divided by dp — DP ranks launch their "+
			"collectives concurrently, so each pays it in full (dp=1: %.3fµs, dp=4: %.3fµs)", at1, at4)
}

// TestTpAllReduceBasis_ZeroDPDivisorIsCoerced extends the struct-literal guard
// (TestTrainedPhysicsModel_StructLiteralKeepsCommTerm) to the dp divisor #1548 introduces.
// A struct-literal model has dp == 0, so StepTime would hand this basis a divisor of 0 —
// which, unguarded, makes the whole comm term +Inf rather than merely mis-scaling it.
func TestTpAllReduceBasis_ZeroDPDivisorIsCoerced(t *testing.T) {
	m := &TrainedPhysicsModel{tp: 8, hiddenDim: 4096, activationBPP: 2, bwHbmUs: 3.35e6, tpSpanScale: 1.0}
	zero := m.tpAllReduceBasis(32, 1024, 0)
	assert.Equal(t, m.tpAllReduceBasis(32, 1024, 1), zero,
		"a zero dp divisor must behave exactly like the no-division case, not produce +Inf")
	assert.False(t, math.IsInf(zero, 0))
	assert.False(t, math.IsNaN(m.tpAllReduceBasis(32, 1024, math.NaN())), "a NaN divisor must not poison the term")
}

// ─── Attributing the Wide-EP speedup to a named term ────────────────────────

// TestStepTime_WideEPSpeedupIsWeightSharding attributes the large Wide-EP step-time
// improvement to a specific term, so the headline number is a physical claim rather than
// "the model says so". The shape is the real GLM/Kimi Wide-EP recipe: TP=1, DP=16, EP=16 —
// i.e. a per-replica config at TP=1, DP=1 carrying a logical EP-group width of 16.
//
// TP=1 forces the attribution, which is what makes this shape the right probe. There is no
// TP collective at all (tpAllReduceBasis returns 0 at tp <= 1) and tMoEReduce additionally
// requires tp > 1, so the all-reduce leg is 0 in BOTH modes — the gate change cannot improve
// this shape, it can only ADD the dispatch term. Routed-expert COMPUTE is scoped by
// moeGroup = TP·DP = 1 in both modes (asserted separately in
// TestStepTime_WideEPDoesNotAlsoReduceCompute). That leaves exactly two moving parts, and
// they move in OPPOSITE directions when the EP group widens to 16:
//
//	routed-expert WEIGHTS  numExperts/1 → numExperts/16   (cheaper)
//	dispatch/combine       none         → a 16-way all-to-all  (more expensive)
//
// So the law is a SIGN FLIP on the size of the expert weights, which is stronger than any
// magnitude comparison: with realistically large experts the same flag must be a speed-UP,
// and with the expert FFN shrunk to almost nothing it must become a slow-DOWN, because only
// the dispatch cost is left. That simultaneously proves the win IS the weight term and that
// the added collective is genuinely charged rather than silently zero.
//
// (An earlier attempt ablated by forcing expertShardGroup back to 1. That cannot separate the
// two: the dispatch collective runs over the expert-owning group, so a group of 1 zeroes the
// dispatch term as well — EP-on at TP=1 with an un-widened group is exactly EP-off, which is
// itself the right answer and is pinned below.)
func TestStepTime_WideEPSpeedupIsWeightSharding(t *testing.T) {
	base := *dpepMoEModelConfig()
	base.NumLocalExperts = 64 // deepseek-v2-lite / GLM-class routed-expert count
	// A token-HEAVY batch, deliberately: the dispatch/combine cost scales with tokens while
	// the weight saving does not, so this is where the added collective is large enough for
	// the sign flip to be a real test rather than a rounding artefact. See the note below on
	// why that asymmetry is itself the reason Wide-EP is a decode-phase strategy.
	batch := makePrefillBatch(16, 2048)

	delta := func(mc sim.ModelConfig) (off, on int64) {
		return newEPModel(t, mc, 1, 1, false, "", 0).StepTime(batch),
			newEPModel(t, mc, 1, 1, true, "", 16).StepTime(batch)
	}

	bigExperts := base
	bigExperts.MoEExpertFFNDim = 1408 // deepseek-v2-lite's real moe_intermediate_size
	offBig, onBig := delta(bigExperts)
	assert.Less(t, onBig, offBig,
		"with realistic expert weights, sharding them across the 16-GPU EP group must dominate the "+
			"all-to-all it introduces (EP off=%dµs, on=%dµs)", offBig, onBig)

	tinyExperts := base
	tinyExperts.MoEExpertFFNDim = 8 // expert weights ≈ 0, so only the dispatch cost remains
	offTiny, onTiny := delta(tinyExperts)
	assert.Greater(t, onTiny, offTiny,
		"with the expert weights shrunk away there is no weight saving left, so the SAME flag must "+
			"become a slow-down — proving the Wide-EP win is routed-expert WEIGHT sharding and that "+
			"the added all-to-all is really charged (EP off=%dµs, on=%dµs)", offTiny, onTiny)
}

// TestStepTime_EPWinIsLargerInDecodeThanPrefill records the asymmetry the sign-flip law above
// exposes, which is a genuine consequence of the two moving terms rather than a modelling
// artefact: the dispatch/combine cost scales with the step's TOKEN count, while the
// routed-expert weight saving is token-INDEPENDENT (numExperts/group). So expert parallelism
// pays best on token-sparse, weight-bound steps — decode — and pays worst on a large prefill
// step, where the all-to-all volume is at its largest and the weight saving is unchanged.
//
// That is a useful external sanity check on the whole feature: production Wide-EP recipes
// (DeepSeek's DEP guidance, and the GLM-5.2 recipe this issue is filed against) deploy wide
// expert parallelism on the DECODE side for exactly this reason. The model reproduces the
// qualitative shape of that recommendation without having been calibrated to it.
func TestStepTime_EPWinIsLargerInDecodeThanPrefill(t *testing.T) {
	mc := *dpepMoEModelConfig()
	mc.NumLocalExperts = 64
	mc.MoEExpertFFNDim = 1408

	relWin := func(batch []*sim.Request) float64 {
		off := newEPModel(t, mc, 1, 1, false, "", 16).StepTime(batch)
		on := newEPModel(t, mc, 1, 1, true, "", 16).StepTime(batch)
		return float64(off-on) / float64(off)
	}
	decodeWin := relWin(makeDecodeBatch(8, 512))
	prefillWin := relWin(makePrefillBatch(16, 2048))

	assert.Greater(t, decodeWin, prefillWin,
		"the relative EP win must be larger in decode (%.3f) than in a large prefill step (%.3f): the "+
			"dispatch cost grows with tokens while the weight saving does not", decodeWin, prefillWin)
}

// TestStepTime_EPOnAtTP1WithUnwidenedGroupIsInert pins the degenerate case the ablation above
// uncovered, because it is a correctness property in its own right: at TP=1 with the EP group
// NOT widened beyond the config's own degrees, the EP group is a single GPU — it owns every
// expert and has no peer to dispatch to — so EP-on must be exactly EP-off.
func TestStepTime_EPOnAtTP1WithUnwidenedGroupIsInert(t *testing.T) {
	mc := *dpepMoEModelConfig()
	mc.NumLocalExperts = 64
	batch := makeDecodeBatch(8, 512)
	assert.Equal(t,
		newEPModel(t, mc, 1, 1, false, "", 0).StepTime(batch),
		newEPModel(t, mc, 1, 1, true, "", 0).StepTime(batch),
		"a one-GPU expert-parallel group holds every expert and has nobody to exchange with, so the "+
			"EP toggle must be exactly neutral there")
}

// TestStepTime_WideEPDoesNotAlsoReduceCompute is the "not double-counting in the other
// direction" check for the shape above. Routed-expert compute is scoped by moeGroup, which
// is 1 for BOTH modes at TP=1/DP=1 — so the compute term must be computed identically, and
// the weight reduction must not be accompanied by a second, illegitimate compute reduction.
//
// The probe is again top_k sensitivity, which on the all-gather family reaches only the
// routed-expert compute basis: neither the weight term (numExperts/group, no top_k) nor that
// family's dispatch volume (dense hidden states, no top_k) carries kEff. Equal sensitivity in
// both modes therefore means the compute term is untouched.
func TestStepTime_WideEPDoesNotAlsoReduceCompute(t *testing.T) {
	base := *dpepMoEModelConfig()
	base.NumLocalExperts = 64
	lowK, highK := base, base
	lowK.NumExpertsPerTok, highK.NumExpertsPerTok = 1, 6
	batch := makePrefillBatch(4, 1024)

	sensitivity := func(ep bool, epGroupDP int) int64 {
		return newEPModel(t, highK, 1, 1, ep, "", epGroupDP).StepTime(batch) -
			newEPModel(t, lowK, 1, 1, ep, "", epGroupDP).StepTime(batch)
	}
	off, on := sensitivity(false, 0), sensitivity(true, 16)
	require.Positive(t, off, "precondition: top_k must move routed-expert compute")
	assert.InDelta(t, off, on, 1.0,
		"routed-expert COMPUTE must be identical in both EP modes at TP=1/DP=1 (moeGroup is 1 for "+
			"both): top_k sensitivity was %dµs EP-off and %dµs EP-on. A difference would mean the "+
			"weight saving is accompanied by an illegitimate compute reduction", off, on)
}

// ─── The "one whole expert per rank" clamp on the WEIGHT divisor ─────────────

// TestStepTime_EPWeightDivisorClampedToExpertCount is the boundary the review flagged as
// untested: an expert-parallel group WIDER than the model's routed-expert count. A rank that
// holds an expert holds one WHOLE expert, so num_experts is the widest divisor the weight
// footprint can support — charging num_experts/EP below 1 would model memory that does not
// exist (the optimistic direction), and the KV-capacity model already clamps this exact
// divisor, so leaving step time unclamped would falsify the agreement between them.
//
// The law: past the expert count, widening the EP group must stop reducing the WEIGHT term.
// It is asserted on a weight-dominated decode batch, and via a sibling model rather than a
// re-implemented formula — a group of exactly num_experts must cost the same as any wider one.
func TestStepTime_EPWeightDivisorClampedToExpertCount(t *testing.T) {
	mc := *dpepMoEModelConfig()
	mc.NumLocalExperts = 8 // Mixtral-class: an 8-expert model under a 16- or 64-wide EP group
	batch := makeDecodeBatch(4, 256)

	atCount := newEPModel(t, mc, 8, 1, true, "", 1).StepTime(batch) // EP group = TP = 8
	// Wider groups: 8·2 = 16 and 8·8 = 64, both beyond the 8 routed experts.
	for _, groupDP := range []int{2, 8} {
		wider := newEPModel(t, mc, 8, 1, true, "", groupDP).StepTime(batch)
		assert.GreaterOrEqual(t, wider, atCount,
			"an EP group of TP·%d exceeds the model's %d routed experts, so the WEIGHT divisor must be "+
				"clamped to the expert count — a wider group must not keep making weights cheaper "+
				"(at-count=%dµs, wider=%dµs)", groupDP, mc.NumLocalExperts, atCount, wider)
	}
}

// TestStepTime_EPOffFractionalExpertShareIsNotClamped is the counterpart, and the reason the
// clamp is gated on expert parallelism rather than applied unconditionally. With EP OFF the
// experts are TENSOR-sharded: a rank genuinely holds a FRACTION of every expert, so
// num_experts/group below one full-expert-equivalent is the CORRECT charge (8 experts
// tensor-sharded over a 16-GPU group is 0.5 each). Clamping there would be wrong physics AND
// would change the step time of every pre-#1548 MoE config whose flattened TP·DP group
// exceeds its expert count — an INV-6 break.
//
// The probe varies num_routed_experts at a FIXED group, which isolates the weight term
// exactly: in BalancedPlacement.Resolve, numExperts enters PerGPUExpertCount and nothing else
// (PerGPUComputeTokens and PerGPUCommTokens are functions of tokens, kEff and the group). So
// at a 16-wide group, an 8-expert model must be strictly cheaper than a 16-expert one (0.5 vs
// 1.0 full-expert-equivalents). Were the 8-expert case clamped to 8, its divisor would also
// yield 1.0 and the two would be EQUAL — which is exactly what this rules out.
func TestStepTime_EPOffFractionalExpertShareIsNotClamped(t *testing.T) {
	base := *dpepMoEModelConfig()
	batch := makeDecodeBatch(4, 256)
	withExperts := func(n int) int64 {
		mc := base
		mc.NumLocalExperts = n
		// EP off ⇒ the weight divisor is the flattened moeGroup = TP·DP = 8·2 = 16. (TP=16
		// directly is not expressible on this fixture: NumKVHeads=8 must divide TP.)
		return newEPModel(t, mc, 8, 2, false, "", 0).StepTime(batch)
	}
	eight, sixteen := withExperts(8), withExperts(16)
	assert.Less(t, eight, sixteen,
		"with EP off the experts are tensor-sharded, so 8 experts over a 16-wide group must charge "+
			"0.5 full-expert-equivalents per GPU — strictly less than a 16-expert model's 1.0. Equality "+
			"would mean the EP-off divisor was clamped to the expert count, which is wrong physics here "+
			"and an INV-6 break for pre-#1548 configs (8=%dµs, 16=%dµs)", eight, sixteen)
}

// TestStepTime_EPOnClampLandsExactlyOnTheExpertCount pins the clamp VALUE, not just its
// direction: under EP-on with a group wider than the expert count, the weight divisor must be
// exactly num_routed_experts. So an 8-expert model at EP=16 must cost precisely what a
// 16-expert model at EP=16 costs — both charge one whole expert per loaded rank.
//
// Same isolation as above (numExperts reaches only the weight term), and it is the assertion
// the EP-off test proves must NOT hold when expert parallelism is off.
func TestStepTime_EPOnClampLandsExactlyOnTheExpertCount(t *testing.T) {
	base := *dpepMoEModelConfig()
	batch := makeDecodeBatch(4, 256)
	withExperts := func(n int) int64 {
		mc := base
		mc.NumLocalExperts = n
		// EP on, group = TP·2 = 16, wider than either expert count.
		return newEPModel(t, mc, 8, 1, true, "", 2).StepTime(batch)
	}
	assert.Equal(t, withExperts(16), withExperts(8),
		"under EP-on the weight divisor must clamp to exactly num_routed_experts, so an 8-expert model "+
			"at EP=16 charges one whole expert per rank — identical to a 16-expert model at EP=16")
}

// TestStepTime_EPDispatchGroupIsNotClampedToExpertCount is the other half of the clamp
// contract, and the discriminating test the qa-review gate asked for: the WEIGHT divisor is
// clamped to the routed-expert count, but the DISPATCH/COMBINE collective is deliberately NOT
// — it genuinely spans every rank in the expert-parallel group, however few experts they hold.
//
// The isolation is exact. Past the expert count the weight term is PINNED (both groups below
// clamp to the same 8), so any remaining step-time difference between two wider groups can only
// come from the dispatch term. On the all-gather family the dispatch volume carries
// (group-1)/group, which keeps rising with the real group: 15/16 at EP=16, 63/64 at EP=64. So
// a strictly larger step time at the wider group proves the collective saw the unclamped group.
//
// Were the dispatch group clamped too, both would use 7/8 and the two step times would be
// EQUAL — which is exactly what this rules out. A token-heavy batch makes the dispatch term
// large enough for the comparison to be decisive rather than a rounding artefact.
func TestStepTime_EPDispatchGroupIsNotClampedToExpertCount(t *testing.T) {
	mc := *dpepMoEModelConfig()
	mc.NumLocalExperts = 8 // both groups below exceed this, so the weight term is pinned
	batch := makePrefillBatch(16, 2048)

	at16 := newEPModel(t, mc, 8, 1, true, "allgather_reducescatter", 2).StepTime(batch) // EP=16
	at64 := newEPModel(t, mc, 8, 1, true, "allgather_reducescatter", 8).StepTime(batch) // EP=64

	assert.Greater(t, at64, at16,
		"the dispatch/combine collective must span the FULL expert-parallel group, unclamped: past "+
			"the %d-expert clamp the weight term is identical for EP=16 and EP=64, so the wider group's "+
			"larger (group-1)/group dispatch volume must still show up as more step time. Equality "+
			"would mean the collective was clamped to the expert count too (EP=16: %dµs, EP=64: %dµs)",
		mc.NumLocalExperts, at16, at64)
}
