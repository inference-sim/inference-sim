package latency

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// hybridTestModelConfig returns a dense model config whose attention is split into
// `full` full-attention (KV-bearing, MLA-style) layers and `total - full` Kimi Delta
// Attention (linear-attention) layers. `full == total` (or `full == 0`, which
// EffectiveKVBearingLayers folds to NumLayers) reproduces a non-hybrid, all-full-
// attention model — the pre-#1636 behavior. Kept dense (NumLocalExperts == 0) so the
// tests isolate the attention/KV split from MoE-FFN scaling.
func hybridTestModelConfig(total, full int) *sim.ModelConfig {
	return &sim.ModelConfig{
		NumLayers:       total,
		KVBearingLayers: full,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		IntermediateDim: 14336,
		BytesPerParam:   2.0, // FP16
	}
}

// TestKDAStepTime_NonHybridByteIdentical is the INV-6 / INV-BC-DP1 no-op guard for
// #1636: the hybrid-attention step-time split must be a STRICT no-op for every
// non-hybrid model. A config with KVBearingLayers == 0 (fallback to NumLayers) and
// one with KVBearingLayers == NumLayers both have lKDA == 0, so every KDA-guarded
// branch is skipped and StepTime must be byte-identical across the batch and TP
// matrices. (The absolute golden values for the non-hybrid path are pinned by
// TestINVBCDP1_DenseStepTimeByteIdentical, which this change leaves untouched.)
func TestKDAStepTime_NonHybridByteIdentical(t *testing.T) {
	hw := testHardwareConfig()
	coeffs := testCoeffs()
	batches := [][]*sim.Request{
		makePrefillBatch(4, 512),
		makeDecodeBatch(8, 2048),
		append(makePrefillBatch(3, 128), makeDecodeBatch(5, 256)...),
	}
	for _, tp := range []int{1, 2, 4, 8} {
		mcZero := hybridTestModelConfig(32, 0)  // non-hybrid (EffectiveKVBearingLayers → NumLayers)
		mcFull := hybridTestModelConfig(32, 32) // explicit all-full-attention
		mhwZero := sim.NewModelHardwareConfig(*mcZero, hw, "m", "H100", tp, 1, false, "", "trained-physics", 0)
		mhwFull := sim.NewModelHardwareConfig(*mcFull, hw, "m", "H100", tp, 1, false, "", "trained-physics", 0)
		mZero, err := NewTrainedPhysicsModel(*coeffs, mhwZero)
		require.NoError(t, err)
		mFull, err := NewTrainedPhysicsModel(*coeffs, mhwFull)
		require.NoError(t, err)
		for _, b := range batches {
			assert.Equalf(t, mZero.StepTime(b), mFull.StepTime(b),
				"KVBearingLayers=0 and =NumLayers must be byte-identical (lKDA=0 no-op, INV-6/INV-BC-DP1); tp=%d", tp)
		}
	}
}

// TestKDAStepTime_HybridDecodeLowerThanAllFull is the #1636 acceptance behavior: for
// a K3-shaped layer split (24 full-attention of 93), decode step time must be LOWER
// than charging all 93 layers as full attention (the pre-#1636 estimate). The
// dominant decode term is the growing-KV read, which #1635 established lives only in
// the KV-bearing (full-attention) layers.
func TestKDAStepTime_HybridDecodeLowerThanAllFull(t *testing.T) {
	hw := testHardwareConfig()
	coeffs := testCoeffs()
	mHybrid := newTestTrainedPhysicsModel(t, hybridTestModelConfig(93, 24), hw, coeffs)
	mAllFull := newTestTrainedPhysicsModel(t, hybridTestModelConfig(93, 93), hw, coeffs)

	batch := makeDecodeBatch(16, 4096) // decode-heavy, long context → KV-read dominant
	hybrid := mHybrid.StepTime(batch)
	allFull := mAllFull.StepTime(batch)

	assert.Less(t, hybrid, allFull,
		"hybrid (24/93 full-attention) decode step time (%d µs) must be below all-93-full-attention (%d µs)", hybrid, allFull)
	assert.Greater(t, hybrid, int64(0), "hybrid decode step time must stay positive")
}

// TestKDAStepTime_ContextGrowthScalesWithFullAttentionLayers is the #1636 metamorphic
// law: growing sequence length raises the full-attention term but NOT the KDA term.
// The context-dependent decode growth (β₂ₐ compute is 0 in the default coefficients,
// so the growth is entirely the β₂ᵦ KV-read term) scales EXACTLY with the KV-bearing
// layer count — so the per-context step-time increase for a 24/93 hybrid is 24/93 of
// the all-93-full-attention increase. KDA layers contribute no context-dependent term.
func TestKDAStepTime_ContextGrowthScalesWithFullAttentionLayers(t *testing.T) {
	hw := testHardwareConfig()
	coeffs := testCoeffs()
	mHybrid := newTestTrainedPhysicsModel(t, hybridTestModelConfig(93, 24), hw, coeffs)
	mAllFull := newTestTrainedPhysicsModel(t, hybridTestModelConfig(93, 93), hw, coeffs)

	const (
		small = 2000
		large = 8000
		count = 16
	)
	dFull := mAllFull.StepTime(makeDecodeBatch(count, large)) - mAllFull.StepTime(makeDecodeBatch(count, small))
	dHybrid := mHybrid.StepTime(makeDecodeBatch(count, large)) - mHybrid.StepTime(makeDecodeBatch(count, small))

	require.Greater(t, dFull, int64(0), "all-full step time must grow with context")
	assert.Greater(t, dHybrid, int64(0),
		"hybrid step time must still grow with context (the 24 full-attention layers do scale)")
	assert.Less(t, dHybrid, dFull,
		"hybrid grows less steeply with context — the 69 KDA layers do not scale with sequence length")

	ratio := float64(dHybrid) / float64(dFull)
	assert.InDelta(t, 24.0/93.0, ratio, 0.02,
		"context-growth ratio (%.4f) must equal the KV-bearing layer fraction 24/93 (%.4f): KDA layers add no context-dependent term",
		ratio, 24.0/93.0)
}

// TestKDAStepTime_HybridPrefillGrowsSubQuadratically is the prefill leg of the #1636
// metamorphic law: KDA prefill is O(N) (linear), full-attention prefill is O(N²). As
// the prompt length grows, the hybrid model's prefill step time must rise LESS than
// the all-full-attention model's, because 69 of 93 layers no longer pay the quadratic
// attention-score cost.
func TestKDAStepTime_HybridPrefillGrowsSubQuadratically(t *testing.T) {
	hw := testHardwareConfig()
	coeffs := testCoeffs()
	mHybrid := newTestTrainedPhysicsModel(t, hybridTestModelConfig(93, 24), hw, coeffs)
	mAllFull := newTestTrainedPhysicsModel(t, hybridTestModelConfig(93, 93), hw, coeffs)

	dFull := mAllFull.StepTime(makePrefillBatch(1, 8000)) - mAllFull.StepTime(makePrefillBatch(1, 2000))
	dHybrid := mHybrid.StepTime(makePrefillBatch(1, 8000)) - mHybrid.StepTime(makePrefillBatch(1, 2000))

	require.Greater(t, dFull, int64(0), "all-full prefill step time must grow with prompt length")
	assert.Greater(t, dHybrid, int64(0), "hybrid prefill step time must still grow (24 full-attention layers are O(N²))")
	assert.Less(t, dHybrid, dFull,
		"hybrid prefill grows less: the 69 KDA layers are O(N), not O(N²)")
}

// rooflineDecodeStep builds a decode-only StepConfig of `count` requests, each at
// `ctx` context tokens.
func rooflineDecodeStep(count int, ctx int64) StepConfig {
	reqs := make([]DecodeRequestConfig, count)
	for i := range reqs {
		reqs[i] = DecodeRequestConfig{ProgressIndex: ctx, NumNewDecodeTokens: 1}
	}
	return StepConfig{DecodeRequests: reqs}
}

// TestRooflineKDA_NonHybridByteIdentical is the roofline INV-6 no-op guard: a config
// with KVBearingLayers == 0 and one with KVBearingLayers == NumLayers must produce
// byte-identical roofline step time (backend parity with the trained-physics no-op).
func TestRooflineKDA_NonHybridByteIdentical(t *testing.T) {
	hw := testHardwareConfig()
	steps := []StepConfig{
		rooflineDecodeStep(16, 4096),
		{PrefillRequests: []PrefillRequestConfig{{ProgressIndex: 0, NumNewPrefillTokens: 4096}}},
	}
	for _, tp := range []int{1, 2, 8} {
		mcZero := hybridTestModelConfig(93, 0)
		mcFull := hybridTestModelConfig(93, 93)
		for _, s := range steps {
			assert.Equalf(t, rooflineStepTime(*mcZero, hw, s, tp), rooflineStepTime(*mcFull, hw, s, tp),
				"roofline KVBearingLayers=0 and =NumLayers must be byte-identical (INV-6); tp=%d", tp)
		}
	}
}

// TestRooflineKDA_HybridDecodeLowerThanAllFull mirrors the trained-physics acceptance
// behavior in the roofline backend (#1636 backend parity): a 24/93 hybrid split has
// lower decode step time than charging all 93 layers as full attention.
func TestRooflineKDA_HybridDecodeLowerThanAllFull(t *testing.T) {
	hw := testHardwareConfig()
	step := rooflineDecodeStep(16, 8192)
	hybrid := rooflineStepTime(*hybridTestModelConfig(93, 24), hw, step, 1)
	allFull := rooflineStepTime(*hybridTestModelConfig(93, 93), hw, step, 1)

	assert.Less(t, hybrid, allFull,
		"roofline hybrid decode step time (%d µs) must be below all-93-full-attention (%d µs)", hybrid, allFull)
	assert.Greater(t, hybrid, int64(0), "roofline hybrid decode step time must stay positive")
}

// TestRooflineKDA_ContextGrowthLessForHybrid is the roofline metamorphic law: the
// growing-KV read (memory-bound decode) scales with the KV-bearing layer count, so a
// hybrid model's step time rises less with context than the all-full-attention model.
func TestRooflineKDA_ContextGrowthLessForHybrid(t *testing.T) {
	hw := testHardwareConfig()
	const (
		small = 2000
		large = 8000
		count = 16
	)
	dFull := rooflineStepTime(*hybridTestModelConfig(93, 93), hw, rooflineDecodeStep(count, large), 1) -
		rooflineStepTime(*hybridTestModelConfig(93, 93), hw, rooflineDecodeStep(count, small), 1)
	dHybrid := rooflineStepTime(*hybridTestModelConfig(93, 24), hw, rooflineDecodeStep(count, large), 1) -
		rooflineStepTime(*hybridTestModelConfig(93, 24), hw, rooflineDecodeStep(count, small), 1)

	require.Greater(t, dFull, int64(0), "all-full roofline step time must grow with context")
	assert.Greater(t, dHybrid, int64(0), "hybrid roofline step time must still grow (24 full-attention layers)")
	assert.Less(t, dHybrid, dFull, "hybrid grows less: KDA layers keep a fixed-size state, not a growing KV cache")
}
