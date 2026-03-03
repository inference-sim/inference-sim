package latency

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
)

// testCrossModelConfig returns a dense model config for crossmodel tests.
func testCrossModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:        32,
		HiddenDim:        4096,
		NumHeads:         32,
		NumKVHeads:       8,
		VocabSize:        128256,
		BytesPerParam:    2,
		IntermediateDim:  14336,
		NumLocalExperts:  0,
		NumExpertsPerTok: 0,
	}
}

// testMoEModelConfig returns a Mixtral-like MoE model config.
func testMoEModelConfig() sim.ModelConfig {
	mc := testCrossModelConfig()
	mc.NumLocalExperts = 8
	mc.NumExpertsPerTok = 2
	return mc
}

func TestCrossModelLatencyModel_StepTime_NonEmpty_Positive(t *testing.T) {
	// BC-4: non-empty batch → StepTime ≥ 1
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: (32.0 * 8.0 * 128.0 / 2.0) * 1e-6, // L*kvHeads*headDim/TP*1e-6
		isMoE:       0.0,
		isTP:        1.0,
	}
	batch := []*sim.Request{{
		InputTokens:   make([]int, 100),
		OutputTokens:  []int{1},
		ProgressIndex: 100,
		NumNewTokens:  1,
	}}
	result := m.StepTime(batch)
	assert.GreaterOrEqual(t, result, int64(1), "BC-4: non-empty batch must produce positive step time")
}

func TestCrossModelLatencyModel_StepTime_EmptyBatch_Zero(t *testing.T) {
	// BC-5: empty batch → 0
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: 0.001,
		isMoE:       0.0,
		isTP:        0.0,
	}
	result := m.StepTime([]*sim.Request{})
	assert.Equal(t, int64(0), result, "BC-5: empty batch must return 0")
}

func TestCrossModelLatencyModel_StepTime_Monotonic_Decode(t *testing.T) {
	// BC-3: more decode tokens → higher or equal step time
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: (32.0 * 8.0 * 128.0 / 2.0) * 1e-6,
		isMoE:       0.0,
		isTP:        1.0,
	}
	batchSmall := []*sim.Request{{
		InputTokens: make([]int, 10), OutputTokens: []int{1},
		ProgressIndex: 10, NumNewTokens: 1,
	}}
	batchLarge := []*sim.Request{
		{InputTokens: make([]int, 10), OutputTokens: []int{1}, ProgressIndex: 10, NumNewTokens: 1},
		{InputTokens: make([]int, 10), OutputTokens: []int{1}, ProgressIndex: 10, NumNewTokens: 1},
	}
	small := m.StepTime(batchSmall)
	large := m.StepTime(batchLarge)
	assert.GreaterOrEqual(t, large, small, "BC-3: more decode tokens must produce >= step time")
}

func TestCrossModelLatencyModel_MoE_IncreasesStepTime(t *testing.T) {
	// BC-6: MoE indicator contributes to step time
	baseCfg := func(isMoE float64) *CrossModelLatencyModel {
		return &CrossModelLatencyModel{
			betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
			alphaCoeffs: []float64{13732.0, 0.0, 860.6},
			numLayers:   32,
			kvDimScaled: (32.0 * 8.0 * 128.0 / 2.0) * 1e-6,
			isMoE:       isMoE,
			isTP:        0.0,
		}
	}
	batch := []*sim.Request{{
		InputTokens: make([]int, 100), OutputTokens: []int{1},
		ProgressIndex: 0, NumNewTokens: 100,
	}}
	dense := baseCfg(0.0).StepTime(batch)
	moe := baseCfg(1.0).StepTime(batch)
	assert.Greater(t, moe, dense, "BC-6: MoE model must have higher step time than dense (same tokens)")
}

func TestCrossModelLatencyModel_TP_IncreasesStepTime(t *testing.T) {
	// BC-7: TP > 1 adds synchronization overhead
	baseCfg := func(isTP float64) *CrossModelLatencyModel {
		return &CrossModelLatencyModel{
			betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
			alphaCoeffs: []float64{13732.0, 0.0, 860.6},
			numLayers:   32,
			kvDimScaled: (32.0 * 8.0 * 128.0 / 2.0) * 1e-6,
			isMoE:       0.0,
			isTP:        isTP,
		}
	}
	batch := []*sim.Request{{
		InputTokens: make([]int, 10), OutputTokens: []int{1},
		ProgressIndex: 10, NumNewTokens: 1,
	}}
	noTP := baseCfg(0.0).StepTime(batch)
	withTP := baseCfg(1.0).StepTime(batch)
	assert.Greater(t, withTP, noTP, "BC-7: TP > 1 must add overhead")
}

func TestCrossModelLatencyModel_MoE_PrefillMonotonicity(t *testing.T) {
	// BC-2: more prefill tokens on MoE model → higher or equal step time
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: (32.0 * 8.0 * 128.0 / 2.0) * 1e-6,
		isMoE:       1.0, // MoE model — prefill tokens contribute via β₂
		isTP:        0.0,
	}
	batchSmall := []*sim.Request{{
		InputTokens: make([]int, 50), ProgressIndex: 0, NumNewTokens: 50,
	}}
	batchLarge := []*sim.Request{{
		InputTokens: make([]int, 100), ProgressIndex: 0, NumNewTokens: 100,
	}}
	small := m.StepTime(batchSmall)
	large := m.StepTime(batchLarge)
	assert.GreaterOrEqual(t, large, small, "BC-2: more prefill tokens on MoE must produce >= step time")
}

func TestCrossModelLatencyModel_QueueingTime_MatchesBlackbox(t *testing.T) {
	// BC-8: QueueingTime identical to blackbox semantics
	alpha := []float64{13732.0, 0.0, 860.6}
	m := &CrossModelLatencyModel{
		betaCoeffs: []float64{116, 1226, 19, 9445}, alphaCoeffs: alpha,
		numLayers: 32, kvDimScaled: 0.001, isMoE: 0, isTP: 0,
	}
	bb := &BlackboxLatencyModel{
		betaCoeffs: []float64{1000, 10, 5}, alphaCoeffs: alpha,
	}
	req := &sim.Request{InputTokens: make([]int, 50)}
	assert.Equal(t, bb.QueueingTime(req), m.QueueingTime(req), "BC-8: QueueingTime must match blackbox")
	assert.Equal(t, bb.OutputTokenProcessingTime(), m.OutputTokenProcessingTime(), "BC-8: OutputTokenProcessingTime must match")
}
