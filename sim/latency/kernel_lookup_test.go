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
	// Warm-start gammas: γ₁=1(gemm), γ₂=1(pf_attn), γ₃=1(dc_attn), γ₄=0(unused),
	// γ₅=1(allreduce), γ₆=0(moe), γ₇=40(layer), γ₈=3(req), γ₉=100(step), γ₁₀=0
	gamma := []float64{1, 1, 1, 0, 1, 0, 40, 3, 100, 0}
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
	// γ₇_pf·L·1 + γ₈·batchSize + γ₉ = 40*32*1 + 3*1 + 100 = 1383 at minimum
	assert.Greater(t, stepTime, int64(1000))
}

// TestKernelLookupModel_PrefixCacheHit_EqualToChunkedSecondStep verifies that
// the FlashAttention basis function is physically consistent for prefix caching.
//
// Invariant: a first-step request with a KV cache prefix hit and a second-step
// chunked-prefill request must produce the SAME step time whenever they have
// identical (full_s, prefix, NumNewTokens). Both represent Q tokens attending
// to the same KV context with the same causal mask.
//
//   Cache-hit  (ProgressIndex=0, ISL=512, NumNewTokens=256, 256 cached):
//     full_s = len(InputTokens) = 512, prefix = 512 - 256 = 256
//
//   Chunked-2  (ProgressIndex=256, ISL=512, NumNewTokens=256, no cache):
//     full_s = ProgressIndex + NumNewTokens = 512, prefix = 256
//
// Without the fix, cache-hit would use full_s=256, prefix=0 — different from
// chunked-2 — and the two physically identical computations would give
// different step times.
func TestKernelLookupModel_PrefixCacheHit_EqualToChunkedSecondStep(t *testing.T) {
	model, err := testKernelLookupModel(t)
	require.NoError(t, err)

	// First-step with prefix cache hit: ISL=512, 256 cached, 256 new.
	cacheHit := &sim.Request{
		InputTokens:   make([]int, 512),
		ProgressIndex: 0,
		NumNewTokens:  256, // only new tokens; 256 cached
	}
	// Chunked second step: ISL=512, first 256 already done, processing next 256.
	chunked2 := &sim.Request{
		InputTokens:   make([]int, 512),
		ProgressIndex: 256, // processed in prior chunk
		NumNewTokens:  256,
	}

	tCacheHit := model.StepTime([]*sim.Request{cacheHit})
	tChunked2 := model.StepTime([]*sim.Request{chunked2})

	assert.Equal(t, tCacheHit, tChunked2,
		"prefix-cache-hit first step and chunked second step with identical "+
			"(full_s=512, prefix=256, newT=256) must produce the same step time")
}

func TestKernelLookupModel_FactoryError_NoProfile(t *testing.T) {
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 1, "kernel-lookup", 0)
	coeffs := sim.NewLatencyCoeffs(make([]float64, 10), make([]float64, 3))
	_, err := NewKernelLookupModel(coeffs, hw)
	assert.Error(t, err, "should fail without KernelProfilePath")
}

func TestKernelLookupModel_FactoryError_TooFewCoeffs(t *testing.T) {
	profilePath := writeKernelProfile(t)
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 1, "kernel-lookup", 0).
		WithKernelProfilePath(profilePath)
	coeffs := sim.NewLatencyCoeffs(make([]float64, 5), make([]float64, 3))
	_, err := NewKernelLookupModel(coeffs, hw)
	assert.Error(t, err, "should fail with fewer than 10 gamma coefficients")
}

func TestKernelLookupModel_FactoryError_TPMismatch(t *testing.T) {
	profilePath := writeKernelProfile(t) // profile has tp: 1
	// runtime TP=2 should fail
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "m", "h100", 2, "kernel-lookup", 0).
		WithKernelProfilePath(profilePath)
	coeffs := sim.NewLatencyCoeffs(make([]float64, 10), make([]float64, 3))
	_, err := NewKernelLookupModel(coeffs, hw)
	assert.Error(t, err, "should fail when runtime TP != profile TP")
}

func TestKernelLookupModel_PostDecodeFixedOverhead(t *testing.T) {
	model, err := testKernelLookupModel(t)
	require.NoError(t, err)
	// alpha[1] = 0 in our test setup
	assert.Equal(t, int64(0), model.PostDecodeFixedOverhead())
}
