package latency

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// BC-1: Empty batch returns >= 1
func TestTrainedPhysicsModel_EmptyBatchReturnsPositive(t *testing.T) {
	m := newTestTrainedPhysicsModel(t, trainedPhysicsTestModelConfig(), testHardwareConfig(), testCoeffs())

	emptyBatch := []*sim.Request{}

	stepTime := m.StepTime(emptyBatch)
	assert.GreaterOrEqual(t, stepTime, int64(1), "Empty batch must return stepTime >= 1")
}

// BC-2: Positive step time for all valid inputs
func TestTrainedPhysicsModel_PositiveStepTime(t *testing.T) {
	m := newTestTrainedPhysicsModel(t, trainedPhysicsTestModelConfig(), testHardwareConfig(), testCoeffs())

	tests := []struct {
		name  string
		batch []*sim.Request
	}{
		{"single_prefill", makePrefillBatch(1, 100)},
		{"single_decode", makeDecodeBatch(1, 100)},
		{"mixed_batch", append(makePrefillBatch(2, 50), makeDecodeBatch(3, 100)...)},
		{"large_prefill", makePrefillBatch(1, 2048)},
		{"large_decode_batch", makeDecodeBatch(32, 512)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stepTime := m.StepTime(tt.batch)
			assert.Greater(t, stepTime, int64(0), "StepTime must be positive for all valid inputs")
		})
	}
}

// BC-3: Monotonicity (more prefill/decode tokens → longer step)
func TestTrainedPhysicsModel_Monotonicity(t *testing.T) {
	m := newTestTrainedPhysicsModel(t, trainedPhysicsTestModelConfig(), testHardwareConfig(), testCoeffs())

	t.Run("prefill_token_monotonicity", func(t *testing.T) {
		batch100 := makePrefillBatch(1, 100)
		batch200 := makePrefillBatch(1, 200)
		batch400 := makePrefillBatch(1, 400)

		time100 := m.StepTime(batch100)
		time200 := m.StepTime(batch200)
		time400 := m.StepTime(batch400)

		assert.Less(t, time100, time200, "More prefill tokens must increase step time")
		assert.Less(t, time200, time400, "Monotonicity must hold across token counts")
	})

	t.Run("decode_token_monotonicity", func(t *testing.T) {
		batch1 := makeDecodeBatch(1, 100)
		batch4 := makeDecodeBatch(4, 100)
		batch8 := makeDecodeBatch(8, 100)

		time1 := m.StepTime(batch1)
		time4 := m.StepTime(batch4)
		time8 := m.StepTime(batch8)

		assert.Less(t, time1, time4, "More decode requests must increase step time")
		assert.Less(t, time4, time8, "Monotonicity must hold across batch sizes")
	})

	t.Run("sequence_length_monotonicity", func(t *testing.T) {
		batch50 := makeDecodeBatch(4, 50)
		batch100 := makeDecodeBatch(4, 100)
		batch200 := makeDecodeBatch(4, 200)

		time50 := m.StepTime(batch50)
		time100 := m.StepTime(batch100)
		time200 := m.StepTime(batch200)

		assert.Less(t, time50, time100, "Longer sequence must increase step time")
		assert.Less(t, time100, time200, "Monotonicity must hold across sequence lengths")
	})
}

// BC-4: Architecture-aware β₈ scaling (interleaved vs. uniform MoE vs. dense)
func TestTrainedPhysicsModel_Beta8Scaling(t *testing.T) {
	hw := testHardwareConfig()
	coeffs := testCoeffs()
	batch := append(makePrefillBatch(1, 100), makeDecodeBatch(4, 100)...)

	t.Run("interleaved_MoE_applies_beta8", func(t *testing.T) {
		// Interleaved: InterleaveMoELayerStep = 1 (alternating MoE/dense)
		interleavedCfg := trainedPhysicsTestModelConfig()
		interleavedCfg.NumLocalExperts = 8
		interleavedCfg.NumExpertsPerTok = 2
		interleavedCfg.InterleaveMoELayerStep = 1
		interleavedCfg.NumLayers = 48
		m := newTestTrainedPhysicsModel(t, interleavedCfg, hw, coeffs)

		stepTime := m.StepTime(batch)
		assert.Greater(t, stepTime, int64(0), "Interleaved MoE must produce positive step time")

		// β₈ contribution: numMoELayers = 48/(1+1) = 24
		require.Equal(t, 24, m.numMoELayers, "Interleaved arch must split layers correctly")
	})

	t.Run("uniform_MoE_skips_beta8", func(t *testing.T) {
		// Uniform: InterleaveMoELayerStep = 0 (all layers MoE)
		uniformCfg := trainedPhysicsTestModelConfig()
		uniformCfg.NumLocalExperts = 8
		uniformCfg.NumExpertsPerTok = 2
		uniformCfg.InterleaveMoELayerStep = 0
		uniformCfg.NumLayers = 32
		m := newTestTrainedPhysicsModel(t, uniformCfg, hw, coeffs)

		stepTime := m.StepTime(batch)
		assert.Greater(t, stepTime, int64(0), "Uniform MoE must produce positive step time")

		// β₈ should not apply (hasInterleavedMoE = false for uniform MoE)
		// numMoELayers = 32 for FLOPs/bandwidth, but moeScaling = 0 prevents β₈
		require.False(t, m.hasInterleavedMoE, "Uniform MoE must skip β₈ (no interleaving)")
		require.Equal(t, 32, m.numMoELayers, "Uniform MoE: all layers are MoE for FLOPs")
	})

	t.Run("dense_model_skips_beta8", func(t *testing.T) {
		// Dense: NumLocalExperts <= 1
		denseCfg := trainedPhysicsTestModelConfig()
		denseCfg.NumLocalExperts = 0
		denseCfg.NumLayers = 32
		m := newTestTrainedPhysicsModel(t, denseCfg, hw, coeffs)

		stepTime := m.StepTime(batch)
		assert.Greater(t, stepTime, int64(0), "Dense model must produce positive step time")

		require.False(t, m.hasInterleavedMoE, "Dense model must skip β₈")
		require.Equal(t, 0, m.numMoELayers, "Dense model has no MoE layers")
	})

	t.Run("uniform_more_expensive_than_interleaved", func(t *testing.T) {
		// Compare interleaved (24 MoE + 24 dense + β₈) vs uniform (48 MoE, no β₈)
		interleavedCfg := trainedPhysicsTestModelConfig()
		interleavedCfg.NumLocalExperts = 8
		interleavedCfg.NumExpertsPerTok = 2
		interleavedCfg.InterleaveMoELayerStep = 1
		interleavedCfg.NumLayers = 48

		uniformCfg := trainedPhysicsTestModelConfig()
		uniformCfg.NumLocalExperts = 8
		uniformCfg.NumExpertsPerTok = 2
		uniformCfg.InterleaveMoELayerStep = 0
		uniformCfg.NumLayers = 48

		mInterleaved := newTestTrainedPhysicsModel(t, interleavedCfg, hw, coeffs)
		mUniform := newTestTrainedPhysicsModel(t, uniformCfg, hw, coeffs)

		timeInterleaved := mInterleaved.StepTime(batch)
		timeUniform := mUniform.StepTime(batch)

		// Uniform MoE does more MoE work (48 MoE layers) than interleaved (24 MoE + 24 dense)
		// Even with β₈ overhead, interleaved is faster because MoE layers are expensive
		assert.Greater(t, timeUniform, timeInterleaved,
			"Uniform MoE (48 MoE layers) must be more expensive than interleaved (24 MoE + 24 dense + β₈)")

		// Verify β₈ is correctly gated
		require.True(t, mInterleaved.hasInterleavedMoE, "Interleaved should have β₈ enabled")
		require.False(t, mUniform.hasInterleavedMoE, "Uniform should have β₈ disabled")
	})
}

// BC-5: Overhead methods (QueueingTime, OutputTokenProcessingTime, PostDecodeFixedOverhead)
func TestTrainedPhysicsModel_OverheadMethods(t *testing.T) {
	m := newTestTrainedPhysicsModel(t, trainedPhysicsTestModelConfig(), testHardwareConfig(), testCoeffs())

	t.Run("QueueingTime_is_constant", func(t *testing.T) {
		req1 := &sim.Request{InputTokens: make([]int, 100)}
		req2 := &sim.Request{InputTokens: make([]int, 500)}

		time1 := m.QueueingTime(req1)
		time2 := m.QueueingTime(req2)

		assert.Equal(t, time1, time2, "QueueingTime must be constant per-request (α₀)")
		assert.Greater(t, time1, int64(0), "QueueingTime must be positive")
	})

	t.Run("OutputTokenProcessingTime_is_constant", func(t *testing.T) {
		time := m.OutputTokenProcessingTime()
		assert.Greater(t, time, int64(0), "OutputTokenProcessingTime must be positive (α₂)")
	})

	t.Run("PostDecodeFixedOverhead_is_constant", func(t *testing.T) {
		time := m.PostDecodeFixedOverhead()
		assert.Greater(t, time, int64(0), "PostDecodeFixedOverhead must be positive (α₁)")
	})
}

// BC-6: Factory construction validation
func TestTrainedPhysicsModel_FactoryConstruction(t *testing.T) {
	t.Run("valid_construction", func(t *testing.T) {
		hw := sim.ModelHardwareConfig{
			Backend:     "trained-physics",
			TP:          2,
			ModelConfig: *trainedPhysicsTestModelConfig(),
			HWConfig:    testHardwareConfig(),
		}
		coeffs := testCoeffs()

		m, err := NewTrainedPhysicsModel(*coeffs, hw)
		require.NoError(t, err, "Valid config must construct successfully")
		require.NotNil(t, m)
	})

	t.Run("invalid_TP_zero", func(t *testing.T) {
		hw := sim.ModelHardwareConfig{
			Backend:     "trained-physics",
			TP:          0, // Invalid
			ModelConfig: *trainedPhysicsTestModelConfig(),
			HWConfig:    testHardwareConfig(),
		}
		coeffs := testCoeffs()

		_, err := NewTrainedPhysicsModel(*coeffs, hw)
		require.Error(t, err, "TP=0 must return error")
		assert.Contains(t, err.Error(), "TP must be > 0")
	})

	t.Run("invalid_layers_zero", func(t *testing.T) {
		cfg := trainedPhysicsTestModelConfig()
		cfg.NumLayers = 0
		hw := sim.ModelHardwareConfig{
			Backend:     "trained-physics",
			TP:          1,
			ModelConfig: *cfg,
			HWConfig:    testHardwareConfig(),
		}
		coeffs := testCoeffs()

		_, err := NewTrainedPhysicsModel(*coeffs, hw)
		require.Error(t, err, "NumLayers=0 must return error")
		assert.Contains(t, err.Error(), "NumLayers must be > 0")
	})

	t.Run("invalid_NaN_coefficient", func(t *testing.T) {
		hw := sim.ModelHardwareConfig{
			Backend:     "trained-physics",
			TP:          1,
			ModelConfig: *trainedPhysicsTestModelConfig(),
			HWConfig:    testHardwareConfig(),
		}
		coeffs := testCoeffs()
		coeffs.BetaCoeffs[0] = math.NaN() // Invalid

		_, err := NewTrainedPhysicsModel(*coeffs, hw)
		require.Error(t, err, "NaN coefficient must return error")
		assert.Contains(t, err.Error(), "NaN")
	})

	t.Run("invalid_negative_coefficient", func(t *testing.T) {
		hw := sim.ModelHardwareConfig{
			Backend:     "trained-physics",
			TP:          1,
			ModelConfig: *trainedPhysicsTestModelConfig(),
			HWConfig:    testHardwareConfig(),
		}
		coeffs := testCoeffs()
		coeffs.BetaCoeffs[5] = -1.0 // Invalid (negative)

		_, err := NewTrainedPhysicsModel(*coeffs, hw)
		require.Error(t, err, "Negative coefficient must return error")
		assert.Contains(t, err.Error(), "negative")
	})
}

// BC-7: Config validation (TP > 0, required fields, coefficient length errors)
func TestTrainedPhysicsModel_ConfigValidation(t *testing.T) {
	t.Run("insufficient_beta_coefficients", func(t *testing.T) {
		hw := sim.ModelHardwareConfig{
			Backend:     "trained-physics",
			TP:          1,
			ModelConfig: *trainedPhysicsTestModelConfig(),
			HWConfig:    testHardwareConfig(),
		}
		coeffs := &sim.LatencyCoeffs{
			AlphaCoeffs: []float64{100, 50, 10},
			BetaCoeffs:  []float64{0.1, 0.2, 0.3}, // Only 3, need at least 7
		}

		_, err := NewTrainedPhysicsModel(*coeffs, hw)
		require.Error(t, err, "< 7 beta coefficients must return error")
		assert.Contains(t, err.Error(), "at least 7")
	})

	t.Run("insufficient_alpha_coefficients", func(t *testing.T) {
		hw := sim.ModelHardwareConfig{
			Backend:     "trained-physics",
			TP:          1,
			ModelConfig: *trainedPhysicsTestModelConfig(),
			HWConfig:    testHardwareConfig(),
		}
		coeffs := &sim.LatencyCoeffs{
			AlphaCoeffs: []float64{100}, // Only 1, need 3
			BetaCoeffs:  []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		}

		_, err := NewTrainedPhysicsModel(*coeffs, hw)
		require.Error(t, err, "< 3 alpha coefficients must return error")
		assert.Contains(t, err.Error(), "AlphaCoeffs requires at least 3")
	})
}

// Helper functions

func trainedPhysicsTestModelConfig() *sim.ModelConfig {
	return &sim.ModelConfig{
		NumLayers:              32,
		HiddenDim:              4096,
		NumHeads:               32,
		NumKVHeads:             8,
		IntermediateDim:        14336,
		NumLocalExperts:        0,  // Dense by default
		NumExpertsPerTok:       0,
		InterleaveMoELayerStep: 0,
		DenseIntermediateDim:   0,
		BytesPerParam:          2.0, // FP16
	}
}

func testHardwareConfig() sim.HardwareCalib {
	return sim.HardwareCalib{
		TFlopsPeak: 989.0,
		BwPeakTBs:  3.35,
		MfuPrefill: 0.55,
		MfuDecode:  0.30,
	}
}

func testCoeffs() *sim.LatencyCoeffs {
	return &sim.LatencyCoeffs{
		AlphaCoeffs: []float64{15563.199579, 777.3455, 45.907545},
		BetaCoeffs:  []float64{0.152128, 0.0, 1.36252915, 0.752037, 32.09546717, 4.41684444, 126.024825, 481.8613888, 0.0, 1.94710771},
	}
}

func newTestTrainedPhysicsModel(t *testing.T, mc *sim.ModelConfig, hw sim.HardwareCalib, coeffs *sim.LatencyCoeffs) *TrainedPhysicsModel {
	t.Helper()
	mhw := sim.ModelHardwareConfig{
		Backend:     "trained-physics",
		TP:          1,
		ModelConfig: *mc,
		HWConfig:    hw,
	}
	m, err := NewTrainedPhysicsModel(*coeffs, mhw)
	require.NoError(t, err, "Test setup must construct valid model")
	return m
}

func makePrefillBatch(count, tokens int) []*sim.Request {
	batch := make([]*sim.Request, count)
	for i := 0; i < count; i++ {
		batch[i] = &sim.Request{
			InputTokens:   make([]int, tokens),
			ProgressIndex: 0,
			NumNewTokens:  tokens,
			OutputTokens:  []int{},
		}
	}
	return batch
}

func makeDecodeBatch(count, seqLen int) []*sim.Request {
	batch := make([]*sim.Request, count)
	for i := 0; i < count; i++ {
		batch[i] = &sim.Request{
			InputTokens:   make([]int, seqLen),
			OutputTokens:  make([]int, 10), // Generated so far
			ProgressIndex: int64(seqLen + 10),
			NumNewTokens:  1, // Decode generates 1 token per step
		}
	}
	return batch
}
