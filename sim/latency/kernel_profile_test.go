package latency

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testProfileYAML = `
gpu: h100_sxm
backend: vllm
version: "0.14.0"
model: meta-llama/Llama-2-7b-hf
tp: 1
num_layers: 32
num_moe_layers: 0
num_dense_layers: 32
hidden_dim: 4096

gemm:
  tokens: [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0]
  latency_us: [151.6, 148.9, 157.3, 154.4, 181.4, 504.1, 2058.7]

context_attention:
  batch_size: [1.0, 4.0, 16.0, 64.0]
  isl: [128.0, 512.0, 2048.0, 4096.0]
  latency_us:
    - [0.31, 0.38, 0.54, 0.95]
    - [0.33, 0.42, 0.68, 1.38]
    - [0.41, 0.62, 1.24, 2.48]
    - [0.65, 1.14, 2.48, 5.01]

generation_attention:
  tokens: [1.0, 4.0, 16.0, 64.0]
  context: [128.0, 512.0, 2048.0, 4096.0]
  latency_us:
    - [0.25, 0.31, 0.46, 0.84]
    - [0.26, 0.33, 0.52, 1.04]
    - [0.31, 0.46, 0.98, 1.96]
    - [0.50, 0.87, 1.87, 3.72]

allreduce:
  tokens: [1.0, 16.0, 256.0, 4096.0]
  latency_us: [0.0, 0.0, 0.0, 0.0]
`

func writeTestProfile(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.yaml")
	require.NoError(t, os.WriteFile(path, []byte(content), 0644))
	return path
}

func TestLoadKernelProfile_ValidYAML(t *testing.T) {
	path := writeTestProfile(t, testProfileYAML)
	p, err := LoadKernelProfile(path)
	require.NoError(t, err)
	assert.Equal(t, "h100_sxm", p.GPU)
	assert.Equal(t, 1, p.TP)
	assert.Equal(t, 32, p.NumLayers)
	assert.Equal(t, 7, len(p.Gemm.Tokens))
	assert.Equal(t, 4, len(p.ContextAttention.BatchSize))
	assert.Equal(t, 4, len(p.ContextAttention.ISL))
	assert.Equal(t, 4, len(p.ContextAttention.LatencyUs))
	assert.Equal(t, 4, len(p.ContextAttention.LatencyUs[0]))
	assert.Nil(t, p.MoECompute) // dense model
}

func TestLoadKernelProfile_FileNotFound(t *testing.T) {
	_, err := LoadKernelProfile("/nonexistent/profile.yaml")
	assert.Error(t, err)
}

func TestLoadKernelProfile_InvalidTP(t *testing.T) {
	bad := `gpu: h100_sxm
tp: 0
num_layers: 32
gemm:
  tokens: [1.0]
  latency_us: [1.0]
context_attention:
  batch_size: [1.0]
  isl: [128.0]
  latency_us: [[1.0]]
generation_attention:
  tokens: [1.0]
  context: [128.0]
  latency_us: [[1.0]]
allreduce:
  tokens: [1.0]
  latency_us: [0.0]
`
	path := writeTestProfile(t, bad)
	_, err := LoadKernelProfile(path)
	assert.Error(t, err, "tp=0 should be rejected")
}

func TestInterp1D_LinearInterpolation(t *testing.T) {
	tbl := Lookup1D{Tokens: []float64{0, 100}, LatencyUs: []float64{0, 200}}
	assert.InDelta(t, 100.0, tbl.Interp1D(50), 0.01)
	assert.InDelta(t, 0.0, tbl.Interp1D(0), 0.01)
	assert.InDelta(t, 200.0, tbl.Interp1D(100), 0.01)
}

func TestInterp1D_Clamps(t *testing.T) {
	tbl := Lookup1D{Tokens: []float64{10, 100}, LatencyUs: []float64{50, 500}}
	assert.InDelta(t, 50.0, tbl.Interp1D(0), 0.01)    // below range → clamp
	assert.InDelta(t, 500.0, tbl.Interp1D(1000), 0.01) // above range → clamp
}

func TestInterp2D_Bilinear(t *testing.T) {
	tbl := Lookup2D{
		BatchSize: []float64{0, 100},
		ISL:       []float64{0, 100},
		LatencyUs: [][]float64{
			{0, 100},   // ISL=0
			{100, 200}, // ISL=100
		},
	}
	assert.InDelta(t, 100.0, tbl.Interp2D(50, 50), 0.01) // center → bilinear
	assert.InDelta(t, 0.0, tbl.Interp2D(0, 0), 0.01)
	assert.InDelta(t, 200.0, tbl.Interp2D(100, 100), 0.01)
}

func TestInterp2D_Clamps(t *testing.T) {
	tbl := Lookup2D{
		BatchSize: []float64{1, 10},
		ISL:       []float64{128, 1024},
		LatencyUs: [][]float64{
			{1.0, 5.0},  // ISL=128
			{3.0, 15.0}, // ISL=1024
		},
	}
	// Below range → clamp to first row/col
	assert.InDelta(t, 1.0, tbl.Interp2D(0, 0), 0.01)
	// Above range → clamp to last row/col
	assert.InDelta(t, 15.0, tbl.Interp2D(100, 9999), 0.01)
}
