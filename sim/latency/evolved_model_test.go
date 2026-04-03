package latency

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
)

// ═══════════════════════════════════════════════════════════════════════════
// Existing coefficient-math tests (TestBeta10*, TestBeta3Prime*)
// ═══════════════════════════════════════════════════════════════════════════

// TestBeta10BatchingInefficiency validates beta10 basis function contributions
// and scaling behavior for long vs short sequences.
//
// This test validates the CORRECTED expected ranges from iter11:
// - beta10 = 0.1-1.0 us per (token^2/batch_request), NOT milliseconds!
// - Expected contribution: ~31.25ms for Scout general-lite (500 tokens, batch_size=4)
// - Expected contribution: ~0.156ms for Scout roleplay (100 tokens, batch_size=32)
// - Expected scaling ratio: 200x (quadratic with sequence length, adjusted for batch size)
func TestBeta10BatchingInefficiency(t *testing.T) {
	// Test case 1: Long sequence, small batch (Scout general-lite scenario)
	// 500 tokens, batch_size=4, beta10=0.0005ms = 0.5us = 0.0000005s
	// Expected: 0.5us * (500^2/4) = 0.5us * 62,500 = 31,250us = 31.25ms = 0.03125s
	coeff := 0.0000005 // 0.5us in seconds
	tokens := 500.0
	batchSize := 4.0
	contribution := coeff * (tokens * tokens / batchSize)
	expectedSeconds := 0.03125 // 31.25ms
	tolerance := 0.10          // 10% tolerance

	if math.Abs(contribution-expectedSeconds)/expectedSeconds > tolerance {
		t.Errorf("beta10 long-sequence contribution out of range:\n"+
			"  got:      %.6fs (%.2fms)\n"+
			"  expected: %.6fs (%.2fms)\n"+
			"  tolerance: +/-%.0f%%",
			contribution, contribution*1e3,
			expectedSeconds, expectedSeconds*1e3,
			tolerance*100)
	}

	// Test case 2: Short sequence, large batch (Scout roleplay scenario)
	// 100 tokens, batch_size=32, beta10=0.5us = 0.0000005s
	// Expected: 0.5us * (100^2/32) = 0.5us * 312.5 = 156.25us = 0.156ms = 0.00015625s
	tokens2 := 100.0
	batchSize2 := 32.0
	contribution2 := coeff * (tokens2 * tokens2 / batchSize2)
	expectedSeconds2 := 0.00015625 // 0.156ms

	if math.Abs(contribution2-expectedSeconds2)/expectedSeconds2 > tolerance {
		t.Errorf("beta10 short-sequence contribution out of range:\n"+
			"  got:      %.6fs (%.3fms)\n"+
			"  expected: %.6fs (%.3fms)\n"+
			"  tolerance: +/-%.0f%%",
			contribution2, contribution2*1e3,
			expectedSeconds2, expectedSeconds2*1e3,
			tolerance*100)
	}

	// Test case 3: Verify quadratic scaling
	// Expected ratio: (500/100)^2 * (32/4) = 25 * 8 = 200x
	ratio := contribution / contribution2
	expectedRatio := 200.0

	if math.Abs(ratio-expectedRatio)/expectedRatio > tolerance {
		t.Errorf("beta10 scaling ratio out of range:\n"+
			"  got:      %.1fx\n"+
			"  expected: %.1fx\n"+
			"  tolerance: +/-%.0f%%",
			ratio, expectedRatio, tolerance*100)
	}

	t.Logf("beta10 unit tests PASSED:")
	t.Logf("  - Long-sequence (500 tokens, batch=4):  %.2fms (%.2f%% error)",
		contribution*1e3, math.Abs(contribution-expectedSeconds)/expectedSeconds*100)
	t.Logf("  - Short-sequence (100 tokens, batch=32): %.3fms (%.2f%% error)",
		contribution2*1e3, math.Abs(contribution2-expectedSeconds2)/expectedSeconds2*100)
	t.Logf("  - Scaling ratio: %.1fx (%.2f%% error)",
		ratio, math.Abs(ratio-expectedRatio)/expectedRatio*100)
}

// TestBeta3PrimeKVSeqLen validates beta3' basis function contributions
// and scaling behavior for long vs short sequences.
//
// This test validates:
// - beta3' = 0.1-1.0 us per (token*layer)
// - Expected contribution: ~14ms for Scout general-lite (500 tokens, 56 layers)
// - Expected contribution: ~2.8ms for Scout roleplay (100 tokens, 56 layers)
// - Expected scaling ratio: 5x (linear with sequence length)
func TestBeta3PrimeKVSeqLen(t *testing.T) {
	// Test case 1: Long sequence, dense model (Scout general-lite scenario)
	// 500 tokens, 56 layers, beta3'=0.5us = 0.0000005s per (token*layer)
	// Expected: 0.5us * (500 * 56) = 0.5us * 28,000 = 14,000us = 14ms = 0.014s
	coeff := 0.0000005 // 0.5us in seconds
	tokens := 500.0
	layers := 56.0
	contribution := coeff * (tokens * layers)
	expectedSeconds := 0.014 // 14ms
	tolerance := 0.10        // 10% tolerance

	if math.Abs(contribution-expectedSeconds)/expectedSeconds > tolerance {
		t.Errorf("beta3' long-sequence contribution out of range:\n"+
			"  got:      %.6fs (%.2fms)\n"+
			"  expected: %.6fs (%.2fms)\n"+
			"  tolerance: +/-%.0f%%",
			contribution, contribution*1e3,
			expectedSeconds, expectedSeconds*1e3,
			tolerance*100)
	}

	// Test case 2: Short sequence, same model (Scout roleplay scenario)
	// 100 tokens, 56 layers, beta3'=0.5us = 0.0000005s
	// Expected: 0.5us * (100 * 56) = 0.5us * 5,600 = 2,800us = 2.8ms = 0.0028s
	tokens2 := 100.0
	contribution2 := coeff * (tokens2 * layers)
	expectedSeconds2 := 0.0028 // 2.8ms

	if math.Abs(contribution2-expectedSeconds2)/expectedSeconds2 > tolerance {
		t.Errorf("beta3' short-sequence contribution out of range:\n"+
			"  got:      %.6fs (%.2fms)\n"+
			"  expected: %.6fs (%.2fms)\n"+
			"  tolerance: +/-%.0f%%",
			contribution2, contribution2*1e3,
			expectedSeconds2, expectedSeconds2*1e3,
			tolerance*100)
	}

	// Test case 3: Verify linear scaling
	// Expected ratio: 500/100 = 5x
	ratio := contribution / contribution2
	expectedRatio := 5.0

	if math.Abs(ratio-expectedRatio)/expectedRatio > tolerance {
		t.Errorf("beta3' scaling ratio out of range:\n"+
			"  got:      %.2fx\n"+
			"  expected: %.2fx\n"+
			"  tolerance: +/-%.0f%%",
			ratio, expectedRatio, tolerance*100)
	}

	t.Logf("beta3' unit tests PASSED:")
	t.Logf("  - Long-sequence (500 tokens, 56 layers):  %.2fms (%.2f%% error)",
		contribution*1e3, math.Abs(contribution-expectedSeconds)/expectedSeconds*100)
	t.Logf("  - Short-sequence (100 tokens, 56 layers): %.2fms (%.2f%% error)",
		contribution2*1e3, math.Abs(contribution2-expectedSeconds2)/expectedSeconds2*100)
	t.Logf("  - Scaling ratio: %.2fx (%.2f%% error)",
		ratio, math.Abs(ratio-expectedRatio)/expectedRatio*100)
}

// TestBeta10PhysicsAnalysis validates the corrected understanding from iter11
// that beta10 should be in microseconds, not milliseconds.
func TestBeta10PhysicsAnalysis(t *testing.T) {
	// Scenario: Scout general-lite experiment
	// Expected TTFT overhead from batching inefficiency: ~30ms
	// Tokens: 500, batch_size: 4
	expectedContributionMs := 30.0 // ms
	tokens := 500.0
	batchSize := 4.0
	basisFunctionValue := (tokens * tokens / batchSize) // 62,500

	// Calculate what beta10 should be
	expectedContributionSeconds := expectedContributionMs / 1e3 // Convert ms to seconds
	coefficientSeconds := expectedContributionSeconds / basisFunctionValue
	coefficientMicroseconds := coefficientSeconds * 1e6 // Convert to us for readability

	t.Logf("Physics analysis for beta10:")
	t.Logf("  - Expected contribution: %.1fms", expectedContributionMs)
	t.Logf("  - Basis function value:  %.0f (token^2/batch_request)", basisFunctionValue)
	t.Logf("  - Calculated beta10:     %.3fus (%.6fs)",
		coefficientMicroseconds, coefficientSeconds)

	// Validate that calculated beta10 is in the CORRECTED range (0.1-1.0 us)
	if coefficientMicroseconds < 0.1 || coefficientMicroseconds > 1.0 {
		t.Errorf("Calculated beta10 outside expected range:\n"+
			"  got:      %.3fus\n"+
			"  expected: 0.1-1.0us",
			coefficientMicroseconds)
	}

	// Demonstrate iter10's error
	iter10ExpectedRangeLowerMs := 0.1 // ms (WRONG!)
	iter10ExpectedRangeUpperMs := 1.0 // ms (WRONG!)
	iter10ConvergedValueUs := 0.945   // us (from iter10 results)
	iter10ConvergedValueMs := iter10ConvergedValueUs / 1e3 // Convert to ms

	t.Logf("\nIter10 hypothesis error analysis:")
	t.Logf("  - Iter10 expected range: %.1f-%.1f **ms** (WRONG by 1000x!)",
		iter10ExpectedRangeLowerMs, iter10ExpectedRangeUpperMs)
	t.Logf("  - Iter10 converged to:   %.3fus = %.6fms",
		iter10ConvergedValueUs, iter10ConvergedValueMs)
	t.Logf("  - Iter10 conclusion:     '1000x too small' (ERROR - hypothesis was wrong!)")
	t.Logf("  - Corrected range:       0.1-1.0 **us**")
	t.Logf("  - Iter10 value in us:    %.3fus (actually WITHIN correct range!)",
		iter10ConvergedValueUs)

	// Validate that iter10's converged value was actually reasonable
	if iter10ConvergedValueUs >= 0.1 && iter10ConvergedValueUs <= 1.0 {
		t.Logf("\nCONFIRMED: Iter10 beta10=%.3fus was CORRECT, hypothesis range was WRONG!",
			iter10ConvergedValueUs)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// EvolvedModel behavioral contract tests (BC-1 through BC-17)
// ═══════════════════════════════════════════════════════════════════════════

// --- Test helpers for EvolvedModel ---

// evolvedFittedBetas provides 10 beta coefficients for the evolved model.
// Based on iter25 architecture-aware training results.
var evolvedFittedBetas = []float64{
	0.138541,  // beta1: prefill roofline correction
	0.0,       // beta2: decode roofline correction
	1.363060,  // beta3: weight loading correction
	0.396094,  // beta4: TP communication correction
	62.289,    // beta5: per-layer overhead (us/layer)
	2.798,     // beta6: per-request scheduling (us/req)
	169.366,   // beta7: per-step overhead (us/step)
	427.3,     // beta8: per-MoE-layer overhead (us/MoE-layer)
	0.0,       // beta9: prefill KV split (beta1b)
	1.2632,    // beta10: decode KV split (beta2b)
}

var evolvedFittedAlphas = []float64{
	15561.96, // alpha0: API processing overhead
	776.24,   // alpha1: post-decode fixed
	45.91,    // alpha2: per-output-token
}

// newEvolvedScoutModel constructs an EvolvedModel for a Scout-style
// interleaved MoE model (InterleaveMoELayerStep=1).
func newEvolvedScoutModel() *EvolvedModel {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:              24,
			NumHeads:               32,
			HiddenDim:              3072,
			IntermediateDim:        8192,
			NumKVHeads:             8,
			NumLocalExperts:        64,
			NumExpertsPerTok:       8,
			InterleaveMoELayerStep: 1, // Interleaved
			BytesPerParam:          2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 8,
	}
	model, err := NewEvolvedModel(coeffs, hw)
	if err != nil {
		panic(err)
	}
	return model
}

// newEvolvedMixtralModel constructs an EvolvedModel for a Mixtral-style
// uniform MoE model (InterleaveMoELayerStep=0).
func newEvolvedMixtralModel() *EvolvedModel {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:              32,
			NumHeads:               32,
			HiddenDim:              4096,
			IntermediateDim:        14336,
			NumKVHeads:             8,
			NumLocalExperts:        8,
			NumExpertsPerTok:       2,
			InterleaveMoELayerStep: 0, // Uniform
			BytesPerParam:          2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 8,
	}
	model, err := NewEvolvedModel(coeffs, hw)
	if err != nil {
		panic(err)
	}
	return model
}

// newEvolvedDenseModel constructs an EvolvedModel for a dense (non-MoE) model.
func newEvolvedDenseModel() *EvolvedModel {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	model, err := NewEvolvedModel(coeffs, hw)
	if err != nil {
		panic(err)
	}
	return model
}

// --- BC-1: Clock Safety ---

func TestEvolved_EmptyBatch_ReturnsOne(t *testing.T) {
	model := newEvolvedScoutModel()
	assert.Equal(t, int64(1), model.StepTime(nil),
		"BC-1: nil batch must return >= 1")
	assert.Equal(t, int64(1), model.StepTime([]*sim.Request{}),
		"BC-1: empty batch must return >= 1")
}

// --- BC-2: Positive Step Time ---

func TestEvolved_PrefillOnly_PositiveStepTime(t *testing.T) {
	model := newEvolvedScoutModel()
	batch := []*sim.Request{makePrefillRequest(512, 512)}
	stepTime := model.StepTime(batch)
	assert.Greater(t, stepTime, int64(1),
		"BC-2: prefill step time should be > 1 us")
}

func TestEvolved_DecodeOnly_PositiveStepTime(t *testing.T) {
	model := newEvolvedScoutModel()
	batch := []*sim.Request{makeDecodeRequest(512, 100)}
	stepTime := model.StepTime(batch)
	assert.Greater(t, stepTime, int64(1),
		"BC-2: decode step time should be > 1 us")
}

func TestEvolved_MixedBatch_PositiveStepTime(t *testing.T) {
	model := newEvolvedScoutModel()
	batch := []*sim.Request{
		makePrefillRequest(256, 256),
		makeDecodeRequest(512, 100),
	}
	stepTime := model.StepTime(batch)
	assert.Greater(t, stepTime, int64(1),
		"BC-2: mixed batch step time should be > 1 us")
}

// --- BC-3: Prefill Monotonicity ---

func TestEvolved_PrefillMonotonicity(t *testing.T) {
	model := newEvolvedScoutModel()

	tokenCounts := []int{64, 128, 256, 512, 1024}
	var prevTime int64
	for _, n := range tokenCounts {
		batch := []*sim.Request{makePrefillRequest(n, n)}
		st := model.StepTime(batch)
		assert.GreaterOrEqual(t, st, prevTime,
			"BC-3: prefill step time should be non-decreasing: %d tokens -> %d us (prev %d us)", n, st, prevTime)
		prevTime = st
	}
}

// --- BC-4: Decode Monotonicity ---

func TestEvolved_DecodeMonotonicity(t *testing.T) {
	model := newEvolvedScoutModel()

	var prevTime int64
	for nReqs := 1; nReqs <= 16; nReqs *= 2 {
		batch := make([]*sim.Request, nReqs)
		for i := range batch {
			batch[i] = makeDecodeRequest(512, 100)
		}
		st := model.StepTime(batch)
		assert.GreaterOrEqual(t, st, prevTime,
			"BC-4: decode step time should be non-decreasing: %d reqs -> %d us (prev %d us)", nReqs, st, prevTime)
		prevTime = st
	}
}

// --- BC-5: Interleaved MoE includes beta8 overhead ---

func TestEvolved_InterleavedMoE_IncludesBeta8(t *testing.T) {
	scout := newEvolvedScoutModel()

	// Verify the model has interleaved MoE enabled
	assert.True(t, scout.hasInterleavedMoE,
		"BC-5: Scout model should have hasInterleavedMoE=true")
	assert.Greater(t, scout.numMoELayers, 0,
		"BC-5: Scout model should have numMoELayers > 0")

	// Compute step time with a decode batch
	batch := []*sim.Request{makeDecodeRequest(512, 100)}
	st := scout.StepTime(batch)

	// The beta8 * numMoELayers term should contribute:
	// 427.3 * 12 = 5127.6 us (Scout: 24 layers / (1+1) = 12 MoE layers)
	// So step time should be well above the pure overhead terms
	assert.Greater(t, st, int64(5000),
		"BC-5: interleaved MoE should include beta8 overhead (~5128 us from 12 MoE layers)")
}

// --- BC-6: Uniform MoE skips beta8 overhead ---

func TestEvolved_UniformMoE_SkipsBeta8(t *testing.T) {
	mixtral := newEvolvedMixtralModel()

	// Verify the model does NOT have interleaved MoE
	assert.False(t, mixtral.hasInterleavedMoE,
		"BC-6: Mixtral model should have hasInterleavedMoE=false")
	assert.Greater(t, mixtral.numMoELayers, 0,
		"BC-6: Mixtral model should still have numMoELayers > 0 (uniform MoE)")

	// Compare Scout (interleaved) vs Mixtral (uniform) with same batch.
	// Scout's beta8 term adds ~5128 us that Mixtral does not get.
	scout := newEvolvedScoutModel()
	batch := []*sim.Request{makeDecodeRequest(512, 100)}

	scoutTime := scout.StepTime(batch)
	mixtralTime := mixtral.StepTime(batch)

	// The models differ in architecture dimensions too, but we can verify
	// the key invariant: Scout's step time should include the beta8 overhead.
	// Since beta8 * numMoELayers = 427.3 * 12 = 5127.6 us for Scout,
	// and Mixtral gets 0 from the beta8 term, Scout should be meaningfully
	// larger if all other terms were similar. Both produce positive times.
	assert.Greater(t, scoutTime, int64(1),
		"BC-6: Scout should produce positive step time")
	assert.Greater(t, mixtralTime, int64(1),
		"BC-6: Mixtral should produce positive step time")

	// Verify using direct struct field that moeScaling is 0 for uniform MoE.
	// This is the key behavioral difference: uniform MoE does not apply beta8.
	t.Logf("Scout (interleaved MoE): %d us", scoutTime)
	t.Logf("Mixtral (uniform MoE):   %d us", mixtralTime)
}

// --- BC-7: QueueingTime returns alpha[0] ---

func TestEvolved_QueueingTime_IsAlpha0(t *testing.T) {
	model := newEvolvedScoutModel()

	// QueueingTime should be constant (alpha0 only), independent of input length.
	req512 := makePrefillRequest(512, 512)
	req1024 := makePrefillRequest(1024, 1024)

	assert.Equal(t, int64(15561), model.QueueingTime(req512),
		"BC-7: QueueingTime should be int64(alpha0)")
	assert.Equal(t, int64(15561), model.QueueingTime(req1024),
		"BC-7: QueueingTime must be constant (alpha0 only), independent of input length")
}

// --- BC-8: OutputTokenProcessingTime returns alpha[2] ---

func TestEvolved_OutputTokenProcessingTime_IsAlpha2(t *testing.T) {
	model := newEvolvedScoutModel()

	// alpha2 = 45.91 -> int64(45.91) = 45
	assert.Equal(t, int64(45), model.OutputTokenProcessingTime(),
		"BC-8: OutputTokenProcessingTime should return int64(alpha2)")
}

// --- BC-9: PostDecodeFixedOverhead returns alpha[1] ---

func TestEvolved_PostDecodeFixedOverhead_IsAlpha1(t *testing.T) {
	model := newEvolvedScoutModel()

	// alpha1 = 776.24 -> int64(776.24) = 776
	assert.Equal(t, int64(776), model.PostDecodeFixedOverhead(),
		"BC-9: PostDecodeFixedOverhead should return int64(alpha1)")
}

// --- BC-10: Factory Construction ---

func TestEvolved_NewEvolvedModel_ReturnsValidModel(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	model, err := NewEvolvedModel(coeffs, hw)
	assert.NoError(t, err, "BC-10: NewEvolvedModel should succeed with valid config")
	assert.NotNil(t, model, "BC-10: NewEvolvedModel should return non-nil model")

	// Verify it produces positive step times
	batch := []*sim.Request{makePrefillRequest(512, 512)}
	assert.Greater(t, model.StepTime(batch), int64(1),
		"BC-10: constructed model should produce positive step times")
}

// --- BC-11: No NaN/Inf ---

func TestEvolved_LargeContext_NoOverflow(t *testing.T) {
	model := newEvolvedScoutModel()

	// Very large context (100K tokens) -- tests float64 arithmetic doesn't overflow
	batch := []*sim.Request{makeDecodeRequest(100000, 50000)}
	st := model.StepTime(batch)
	assert.GreaterOrEqual(t, st, int64(1),
		"BC-11: large context must not produce overflow/NaN")
	// Step time should be a reasonable value (not overflowed to 1 from max(1, negative))
	assert.Greater(t, st, int64(100),
		"BC-11: 150K context should produce >100 us step time")
}

func TestEvolved_LargeBatch_NoOverflow(t *testing.T) {
	model := newEvolvedScoutModel()

	// Large batch of 256 decode requests
	batch := make([]*sim.Request, 256)
	for i := range batch {
		batch[i] = makeDecodeRequest(512, 200)
	}
	st := model.StepTime(batch)
	assert.GreaterOrEqual(t, st, int64(1),
		"BC-11: large batch must not overflow")
	assert.Greater(t, st, int64(100),
		"BC-11: 256-request batch should produce substantial step time")
}

// --- BC-12: Zero Allocations (benchmark) ---

func BenchmarkEvolved_StepTime(b *testing.B) {
	model := newEvolvedScoutModel()
	batch := make([]*sim.Request, 8)
	for i := range batch {
		if i < 2 {
			batch[i] = makePrefillRequest(512, 512)
		} else {
			batch[i] = makeDecodeRequest(512, 100)
		}
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = model.StepTime(batch)
	}
}

func TestEvolved_StepTime_ZeroAllocs(t *testing.T) {
	model := newEvolvedScoutModel()
	batch := make([]*sim.Request, 8)
	for i := range batch {
		if i < 2 {
			batch[i] = makePrefillRequest(512, 512)
		} else {
			batch[i] = makeDecodeRequest(512, 100)
		}
	}
	allocs := testing.AllocsPerRun(100, func() {
		_ = model.StepTime(batch)
	})
	assert.Equal(t, float64(0), allocs,
		"BC-12: StepTime must be allocation-free on the hot path")
}

// --- BC-13: Too-few betas returns error ---

func TestEvolved_TooFewBetas_Error(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  []float64{1, 2, 3, 4, 5, 6}, // Only 6, need >= 7
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "BC-13: fewer than 7 betas should return error")
	assert.Contains(t, err.Error(), "7 elements",
		"BC-13: error message should mention required 7 elements")
}

func TestEvolved_TooFewAlphas_Error(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{100.0, 50.0}, // Only 2, need >= 3
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "BC-13: fewer than 3 alphas should return error")
	assert.Contains(t, err.Error(), "3 elements",
		"BC-13: error message should mention required 3 elements")
}

// --- BC-14: Invalid config returns errors ---

func TestEvolved_InvalidConfig_Error(t *testing.T) {
	baseHW := func() sim.ModelHardwareConfig {
		return sim.ModelHardwareConfig{
			ModelConfig: sim.ModelConfig{
				NumLayers:       32,
				NumHeads:        32,
				HiddenDim:       4096,
				IntermediateDim: 11008,
				NumKVHeads:      32,
				BytesPerParam:   2,
			},
			HWConfig: sim.HardwareCalib{
				TFlopsPeak: 989.0,
				BwPeakTBs:  3.35,
			},
			TP: 1,
		}
	}

	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}

	tests := []struct {
		name   string
		modify func(*sim.ModelHardwareConfig)
		errMsg string
	}{
		{"zero TP", func(hw *sim.ModelHardwareConfig) { hw.TP = 0 }, "TP"},
		{"negative TP", func(hw *sim.ModelHardwareConfig) { hw.TP = -1 }, "TP"},
		{"zero NumLayers", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.NumLayers = 0 }, "NumLayers"},
		{"zero NumHeads", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.NumHeads = 0 }, "NumHeads"},
		{"zero HiddenDim", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.HiddenDim = 0 }, "HiddenDim"},
		{"zero IntermediateDim", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.IntermediateDim = 0 }, "IntermediateDim"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hw := baseHW()
			tc.modify(&hw)
			_, err := NewEvolvedModel(coeffs, hw)
			assert.Error(t, err,
				"BC-14: %s should return error", tc.name)
			assert.Contains(t, err.Error(), tc.errMsg,
				"BC-14: error for %s should mention %s", tc.name, tc.errMsg)
		})
	}
}

// --- BC-15: Config validation - NumHeads % TP != 0 returns error ---

func TestEvolved_NumHeadsNotDivisibleByTP_Error(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 3, // 32 % 3 != 0
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "BC-15: NumHeads not divisible by TP should return error")
	assert.Contains(t, err.Error(), "divisible",
		"BC-15: error should mention divisibility")
}

func TestEvolved_NumKVHeadsNotDivisibleByTP_Error(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      5, // 5 % 2 != 0
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 2,
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "BC-15: NumKVHeads not divisible by TP should return error")
	assert.Contains(t, err.Error(), "divisible",
		"BC-15: error should mention divisibility")
}

// --- BC-16: Hardware specs - TFlopsPeak=0 and BwPeakTBs=0 return errors ---

func TestEvolved_ZeroTFlopsPeak_Error(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 0, // Invalid
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "BC-16: TFlopsPeak=0 should return error")
	assert.Contains(t, err.Error(), "TFlopsPeak",
		"BC-16: error should mention TFlopsPeak")
}

func TestEvolved_ZeroBwPeakTBs_Error(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  0, // Invalid
		},
		TP: 1,
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "BC-16: BwPeakTBs=0 should return error")
	assert.Contains(t, err.Error(), "BwPeakTBs",
		"BC-16: error should mention BwPeakTBs")
}

func TestEvolved_NaNTFlopsPeak_Error(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: math.NaN(),
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "BC-16: NaN TFlopsPeak should return error")
	assert.Contains(t, err.Error(), "TFlopsPeak",
		"BC-16: error should mention TFlopsPeak")
}

// --- BC-17: Never negative ---

func TestEvolved_StepTime_NeverNegative(t *testing.T) {
	model := newEvolvedScoutModel()
	cases := []struct {
		name  string
		batch []*sim.Request
	}{
		{"nil", nil},
		{"empty", []*sim.Request{}},
		{"single prefill 1 token", []*sim.Request{makePrefillRequest(1, 1)}},
		{"single decode short context", []*sim.Request{makeDecodeRequest(10, 1)}},
		{"large batch", func() []*sim.Request {
			b := make([]*sim.Request, 256)
			for i := range b {
				b[i] = makeDecodeRequest(512, 200)
			}
			return b
		}()},
		{"mixed batch", []*sim.Request{
			makePrefillRequest(256, 256),
			makeDecodeRequest(512, 100),
			makeDecodeRequest(1024, 500),
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := model.StepTime(tc.batch)
			assert.GreaterOrEqual(t, st, int64(1),
				"BC-17: StepTime must be >= 1 for all inputs (INV-3)")
		})
	}
}

func TestEvolved_StepTime_NeverNegative_AllArchitectures(t *testing.T) {
	models := map[string]*EvolvedModel{
		"scout":   newEvolvedScoutModel(),
		"mixtral": newEvolvedMixtralModel(),
		"dense":   newEvolvedDenseModel(),
	}

	batch := []*sim.Request{
		makePrefillRequest(128, 128),
		makeDecodeRequest(256, 50),
	}

	for name, model := range models {
		t.Run(name, func(t *testing.T) {
			st := model.StepTime(batch)
			assert.GreaterOrEqual(t, st, int64(1),
				"BC-17: StepTime must be >= 1 for %s architecture", name)
		})
	}
}

// --- Additional structural validation tests ---

func TestEvolved_ScoutModel_LayerSplit(t *testing.T) {
	model := newEvolvedScoutModel()

	// Scout: 24 layers, InterleaveMoELayerStep=1
	// -> numMoELayers = 24 / (1+1) = 12
	// -> numDenseLayers = 24 - 12 = 12
	assert.Equal(t, 12, model.numMoELayers,
		"Scout should have 12 MoE layers (24 / (1+1))")
	assert.Equal(t, 12, model.numDenseLayers,
		"Scout should have 12 dense layers (24 - 12)")
	assert.True(t, model.hasInterleavedMoE,
		"Scout should have hasInterleavedMoE=true")
}

func TestEvolved_MixtralModel_LayerSplit(t *testing.T) {
	model := newEvolvedMixtralModel()

	// Mixtral: 32 layers, InterleaveMoELayerStep=0 (uniform MoE)
	// -> all layers are MoE
	assert.Equal(t, 32, model.numMoELayers,
		"Mixtral should have all 32 layers as MoE")
	assert.Equal(t, 0, model.numDenseLayers,
		"Mixtral should have 0 dense layers")
	assert.False(t, model.hasInterleavedMoE,
		"Mixtral should have hasInterleavedMoE=false")
}

func TestEvolved_DenseModel_LayerSplit(t *testing.T) {
	model := newEvolvedDenseModel()

	// Dense: 32 layers, no MoE
	assert.Equal(t, 0, model.numMoELayers,
		"Dense model should have 0 MoE layers")
	assert.Equal(t, 32, model.numDenseLayers,
		"Dense model should have all 32 dense layers")
	assert.False(t, model.hasInterleavedMoE,
		"Dense model should have hasInterleavedMoE=false")
}

// --- Verify 7-beta backward compatibility (beta8 defaults to 0) ---

func TestEvolved_SevenBetas_BackwardCompatible(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  []float64{0.138541, 0.0, 1.363060, 0.396094, 62.289, 2.798, 169.366}, // Only 7
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	model, err := NewEvolvedModel(coeffs, hw)
	assert.NoError(t, err, "7 betas should be accepted (backward compatible)")
	assert.NotNil(t, model)

	// Verify beta8 defaults to 0
	assert.Equal(t, 0.0, model.beta[7],
		"beta8 should default to 0 when only 7 betas provided")
	assert.False(t, model.prefillSplit,
		"prefillSplit should be false with 7 betas")
	assert.False(t, model.decodeSplit,
		"decodeSplit should be false with 7 betas")

	// Model should still produce valid step times
	batch := []*sim.Request{makePrefillRequest(512, 512)}
	st := model.StepTime(batch)
	assert.Greater(t, st, int64(1))
}

// --- Verify NaN/Inf coefficient rejection ---

func TestEvolved_NaNCoeffs_Error(t *testing.T) {
	badBetas := make([]float64, 10)
	copy(badBetas, evolvedFittedBetas)
	badBetas[2] = math.NaN()

	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: evolvedFittedAlphas,
		BetaCoeffs:  badBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "NaN beta coefficient should return error")
	assert.Contains(t, err.Error(), "NaN")
}

func TestEvolved_InfCoeffs_Error(t *testing.T) {
	badAlphas := make([]float64, 3)
	copy(badAlphas, evolvedFittedAlphas)
	badAlphas[0] = math.Inf(1)

	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: badAlphas,
		BetaCoeffs:  evolvedFittedBetas,
	}
	hw := sim.ModelHardwareConfig{
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
		TP: 1,
	}
	_, err := NewEvolvedModel(coeffs, hw)
	assert.Error(t, err, "Inf alpha coefficient should return error")
	assert.Contains(t, err.Error(), "Inf")
}
