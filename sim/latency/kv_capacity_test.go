package latency_test

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// --- Test helpers ---

// validDenseModelConfig returns a Llama-3.1-8B-like ModelConfig.
func validDenseModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		VocabSize:       128256,
		BytesPerParam:   2,
		IntermediateDim: 14336,
	}
}

// validHWConfig returns an H100-like HardwareCalib with 80 GiB memory.
func validHWConfig() sim.HardwareCalib {
	return sim.HardwareCalib{
		TFlopsPeak:    989.5,
		BwPeakTBs:     3.35,
		BwEffConstant: 0.72,
		MfuPrefill:    0.65,
		MfuDecode:     0.12,
		MemoryGiB:     80.0,
	}
}

// validDenseKVParams returns KVCapacityParams for a dense (non-MoE) model
// with SwiGLU activation.
func validDenseKVParams() latency.KVCapacityParams {
	return latency.KVCapacityParams{
		IsMoE:             false,
		NumLocalExperts:   0,
		TieWordEmbeddings: false,
		HiddenAct:         "silu",
	}
}

// --- Input validation tests ---

func TestCalculateKVBlocks_ZeroDenominators_ReturnError(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()
	tp := 1
	blockSize := int64(16)

	tests := []struct {
		name    string
		setup   func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams)
		errWant string // substring expected in the error
	}{
		{
			name: "zero TP",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				return mc, hc, 0, blockSize, params
			},
			errWant: "TP",
		},
		{
			name: "zero block size",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				return mc, hc, tp, 0, params
			},
			errWant: "block size",
		},
		{
			name: "zero NumHeads",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.NumHeads = 0
				return m, hc, tp, blockSize, params
			},
			errWant: "num_attention_heads",
		},
		{
			name: "zero BytesPerParam",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.BytesPerParam = 0
				return m, hc, tp, blockSize, params
			},
			errWant: "precision",
		},
		{
			name: "zero MemoryGiB",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				h := hc
				h.MemoryGiB = 0
				return mc, h, tp, blockSize, params
			},
			errWant: "GPU memory",
		},
		{
			name: "zero NumLayers",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.NumLayers = 0
				return m, hc, tp, blockSize, params
			},
			errWant: "num_layers",
		},
		{
			name: "zero HiddenDim",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.HiddenDim = 0
				return m, hc, tp, blockSize, params
			},
			errWant: "hidden_dim",
		},
		{
			name: "zero IntermediateDim",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.IntermediateDim = 0
				return m, hc, tp, blockSize, params
			},
			errWant: "intermediate_dim",
		},
		{
			name: "zero VocabSize",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.VocabSize = 0
				return m, hc, tp, blockSize, params
			},
			errWant: "vocab_size",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, h, tpVal, bs, p := tt.setup()
			_, err := latency.CalculateKVBlocks(m, h, tpVal, bs, p)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.errWant)
			}
			if !strings.Contains(err.Error(), tt.errWant) {
				t.Errorf("expected error containing %q, got: %v", tt.errWant, err)
			}
		})
	}
}

func TestCalculateKVBlocks_NaNInfInputs_ReturnError(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	tests := []struct {
		name    string
		setup   func() (sim.ModelConfig, sim.HardwareCalib)
		errWant string
	}{
		{
			name: "NaN GPU memory",
			setup: func() (sim.ModelConfig, sim.HardwareCalib) {
				h := hc
				h.MemoryGiB = math.NaN()
				return mc, h
			},
			errWant: "GPU memory",
		},
		{
			name: "Inf GPU memory",
			setup: func() (sim.ModelConfig, sim.HardwareCalib) {
				h := hc
				h.MemoryGiB = math.Inf(1)
				return mc, h
			},
			errWant: "GPU memory",
		},
		{
			name: "NaN precision",
			setup: func() (sim.ModelConfig, sim.HardwareCalib) {
				m := mc
				m.BytesPerParam = math.NaN()
				return m, hc
			},
			errWant: "precision",
		},
		{
			name: "Inf precision",
			setup: func() (sim.ModelConfig, sim.HardwareCalib) {
				m := mc
				m.BytesPerParam = math.Inf(-1)
				return m, hc
			},
			errWant: "precision",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, h := tt.setup()
			_, err := latency.CalculateKVBlocks(m, h, 1, 16, params)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.errWant)
			}
			if !strings.Contains(err.Error(), tt.errWant) {
				t.Errorf("expected error containing %q, got: %v", tt.errWant, err)
			}
		})
	}
}

func TestCalculateKVBlocks_HeadDimNotDivisible_ReturnError(t *testing.T) {
	mc := validDenseModelConfig()
	mc.HiddenDim = 4097 // not divisible by 32
	hc := validHWConfig()
	params := validDenseKVParams()

	_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, params)
	if err == nil {
		t.Fatal("expected error for non-divisible head dim, got nil")
	}
	if !strings.Contains(err.Error(), "divisible") {
		t.Errorf("expected error mentioning 'divisible', got: %v", err)
	}
}

func TestCalculateKVBlocks_BudgetExceeded_ReturnError(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	hc.MemoryGiB = 1.0 // too small for an 8B model
	params := validDenseKVParams()

	_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, params)
	if err == nil {
		t.Fatal("expected error for exceeded budget, got nil")
	}
	if !strings.Contains(err.Error(), "exceed") {
		t.Errorf("expected error mentioning 'exceed', got: %v", err)
	}
}

func TestCalculateKVBlocks_NonSwiGLU_ReturnError(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()
	params.HiddenAct = "relu"

	_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, params)
	if err == nil {
		t.Fatal("expected error for non-SwiGLU activation, got nil")
	}
	if !strings.Contains(err.Error(), "activation") {
		t.Errorf("expected error mentioning 'activation', got: %v", err)
	}
}

func TestCalculateKVBlocks_TPDivisibility_ReturnError(t *testing.T) {
	mc := validDenseModelConfig()
	mc.NumKVHeads = 8
	hc := validHWConfig()
	params := validDenseKVParams()

	_, err := latency.CalculateKVBlocks(mc, hc, 3, 16, params)
	if err == nil {
		t.Fatal("expected error for TP not dividing num_kv_heads, got nil")
	}
	errMsg := err.Error()
	if !strings.Contains(errMsg, "divisible") || !strings.Contains(errMsg, "TP") {
		t.Errorf("expected error mentioning 'divisible' and 'TP', got: %v", err)
	}
}
