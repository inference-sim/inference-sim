package latency

import (
	"os"
	"path/filepath"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeKernelProfile(t *testing.T) string {
	t.Helper()
	content := `
gpu: h100_sxm
backend: vllm
version: "0.14.0"
model: test-model
tp: 1
num_layers: 32
num_moe_layers: 0
num_dense_layers: 32
hidden_dim: 4096

gemm:
  tokens: [1.0, 16.0, 256.0, 4096.0]
  latency_us: [150.0, 155.0, 180.0, 2000.0]

context_attention:
  batch_size: [1.0, 4.0, 16.0]
  isl: [128.0, 1024.0, 4096.0]
  latency_us:
    - [0.3, 0.35, 0.4]
    - [0.5, 0.6, 0.75]
    - [1.0, 1.3, 1.8]

generation_attention:
  tokens: [1.0, 16.0, 256.0]
  context: [128.0, 1024.0, 4096.0]
  latency_us:
    - [0.24, 0.28, 0.36]
    - [0.4, 0.5, 0.65]
    - [0.8, 1.1, 1.5]

allreduce:
  tokens: [1.0, 256.0, 4096.0]
  latency_us: [0.0, 0.0, 0.0]
`
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.yaml")
	require.NoError(t, os.WriteFile(path, []byte(content), 0644))
	return path
}

func testKernelLookupModel(t *testing.T) (*KernelLookupModel, error) {
	t.Helper()
	profilePath := writeKernelProfile(t)
	// iter35 coefficients:
	// gamma[0]=γ₁=1(gemm), [1]=γ₂=1(pf_attn), [2]=γ₃=1(dc_attn),
	// [3]=reserved=0, [4]=γ₅=1(allreduce), [5]=γ₆=0(moe),
	// [6]=γ₇_pf=40(per-layer-per-seq), [7]=γ₈=3(per-req),
	// [8]=γ₉=100(per-step), [9]=γ₇_dc=10(per-layer-constant)
	gamma := []float64{1, 1, 1, 0, 1, 0, 40, 3, 100, 10}
	alpha := []float64{500, 0, 0}
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "h100", 1, "kernel-lookup", 0).
		WithKernelProfilePath(profilePath)
	coeffs := sim.NewLatencyCoeffs(gamma, alpha)
	m, err := NewKernelLookupModel(coeffs, hw)
	if err != nil {
		return nil, err
	}
	return m.(*KernelLookupModel), nil
}

func TestKernelLookupModel_EmptyBatch_ReturnsOne(t *testing.T) {
	model, err := testKernelLookupModel(t)
	require.NoError(t, err)
	assert.Equal(t, int64(1), model.StepTime([]*sim.Request{}))
}

func TestKernelLookupModel_QueueingTime(t *testing.T) {
	model, err := testKernelLookupModel(t)
	require.NoError(t, err)
	req := &sim.Request{}
	assert.Equal(t, int64(500), model.QueueingTime(req))
}

func TestKernelLookupModel_SinglePrefillRequest(t *testing.T) {
	model, err := testKernelLookupModel(t)
	require.NoError(t, err)

	req := &sim.Request{
		InputTokens:   make([]int, 128),
		ProgressIndex: 0,
		NumNewTokens:  128,
	}
	stepTime := model.StepTime([]*sim.Request{req})
	// γ₇_pf·L·1 + γ₈·batchSize + γ₉ = 40·32·1 + 3·1 + 100 = 1383 at minimum
	assert.Greater(t, stepTime, int64(1000))
}

// TestKernelLookupModel_Gamma9_ConstantDecodeOverhead verifies that γ₇_dc (gamma[9])
// contributes a CONSTANT per-step overhead (γ₇_dc × L) when decode requests are present,
// regardless of how many decode requests are in the batch.
//
// This is the iter35 fix: γ₇_dc × L (constant) instead of the iter33 form
// γ₇_dc × L / √batch (unphysical — overhead decreased with more work).
func TestKernelLookupModel_Gamma9_ConstantDecodeOverhead(t *testing.T) {
	profilePath := writeKernelProfile(t)
	// Only γ₇_dc non-zero so we can isolate its contribution
	gamma := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 10} // γ₇_dc=10µs/layer
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 1, "kernel-lookup", 0).
		WithKernelProfilePath(profilePath)
	model, err := NewKernelLookupModel(sim.NewLatencyCoeffs(gamma, []float64{0, 0, 0}), hw)
	require.NoError(t, err)

	makeDecodeReq := func() *sim.Request {
		return &sim.Request{
			InputTokens:   make([]int, 64),
			OutputTokens:  make([]int, 1),
			ProgressIndex: 64,
			NumNewTokens:  1,
		}
	}

	// γ₇_dc × L is constant regardless of batch size (not 1/√batch as in iter33).
	// Both batch=1 and batch=8 pay the same overhead = γ₇_dc × L = 10 × 32 = 320µs.
	t1 := model.StepTime([]*sim.Request{makeDecodeReq()})
	t8 := model.StepTime([]*sim.Request{
		makeDecodeReq(), makeDecodeReq(), makeDecodeReq(), makeDecodeReq(),
		makeDecodeReq(), makeDecodeReq(), makeDecodeReq(), makeDecodeReq(),
	})
	expected := int64(10 * 32) // γ₇_dc × L = 320µs
	assert.Equal(t, expected, t1, "γ₇_dc × L must be 320µs at decode batch=1")
	assert.Equal(t, expected, t8, "γ₇_dc × L must be 320µs at decode batch=8 (batch-independent)")
}

// TestKernelLookupModel_PrefixCacheHit_EqualToChunkedSecondStep verifies the
// FlashAttention prefix-cache fix (iter33, preserved in iter35).
func TestKernelLookupModel_PrefixCacheHit_EqualToChunkedSecondStep(t *testing.T) {
	model, err := testKernelLookupModel(t)
	require.NoError(t, err)

	cacheHit := &sim.Request{
		InputTokens:   make([]int, 512),
		ProgressIndex: 0,
		NumNewTokens:  256,
	}
	chunked2 := &sim.Request{
		InputTokens:   make([]int, 512),
		ProgressIndex: 256,
		NumNewTokens:  256,
	}

	assert.Equal(t, model.StepTime([]*sim.Request{cacheHit}),
		model.StepTime([]*sim.Request{chunked2}),
		"cache-hit first step and chunked second step with same (full_s, prefix, newT) must match")
}

func TestKernelLookupModel_FactoryError_NoProfile(t *testing.T) {
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 1, "kernel-lookup", 0)
	coeffs := sim.NewLatencyCoeffs(make([]float64, 10), make([]float64, 3))
	_, err := NewKernelLookupModel(coeffs, hw)
	assert.Error(t, err)
}

func TestKernelLookupModel_FactoryError_TooFewCoeffs(t *testing.T) {
	profilePath := writeKernelProfile(t)
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 1, "kernel-lookup", 0).
		WithKernelProfilePath(profilePath)
	coeffs := sim.NewLatencyCoeffs(make([]float64, 5), make([]float64, 3))
	_, err := NewKernelLookupModel(coeffs, hw)
	assert.Error(t, err)
}

func TestKernelLookupModel_FactoryError_TPMismatch(t *testing.T) {
	profilePath := writeKernelProfile(t)
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 2, "kernel-lookup", 0).
		WithKernelProfilePath(profilePath)
	coeffs := sim.NewLatencyCoeffs(make([]float64, 10), make([]float64, 3))
	_, err := NewKernelLookupModel(coeffs, hw)
	assert.Error(t, err)
}

func TestKernelLookupModel_PostDecodeFixedOverhead(t *testing.T) {
	model, err := testKernelLookupModel(t)
	require.NoError(t, err)
	assert.Equal(t, int64(0), model.PostDecodeFixedOverhead())
}
