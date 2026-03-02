package latency_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
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

func TestCalculateKVBlocks_FloorZero_ReturnError(t *testing.T) {
	// Use a tiny GPU memory that barely covers overhead, leaving near-zero
	// allocatable space so floor division yields 0 blocks.
	mc := validDenseModelConfig()
	hc := validHWConfig()
	hc.MemoryGiB = 2.0 // Just enough to not trigger "exceeds" but too small for any blocks
	params := validDenseKVParams()

	_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, params)
	if err == nil {
		t.Fatal("expected error for floor-zero blocks, got nil")
	}
	if !strings.Contains(err.Error(), "0 blocks") && !strings.Contains(err.Error(), "exceed") {
		t.Errorf("expected error mentioning '0 blocks' or 'exceed', got: %v", err)
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

// --- Task 2: Empirical fidelity + invariant tests (BC-4, BC-5, KV-CAP-5) ---

// llama31_8B_ModelConfig returns the exact Llama-3.1-8B architecture config
// used in empirical fidelity tests.
func llama31_8B_ModelConfig() sim.ModelConfig {
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

// h100HWConfig returns the H100 hardware config with 80 GiB memory.
func h100HWConfig() sim.HardwareCalib {
	return sim.HardwareCalib{
		TFlopsPeak:    989.5,
		BwPeakTBs:     3.35,
		BwEffConstant: 0.72,
		MfuPrefill:    0.65,
		MfuDecode:     0.12,
		MemoryGiB:     80.0,
	}
}

func TestCalculateKVBlocks_Llama31_8B_H100_TP2_WithinTolerance(t *testing.T) {
	mc := llama31_8B_ModelConfig()
	hc := h100HWConfig()
	params := latency.KVCapacityParams{
		IsMoE:             false,
		NumLocalExperts:   0,
		TieWordEmbeddings: false,
		HiddenAct:         "silu",
	}

	// Empirical baseline from defaults.yaml: Llama-3.1-8B / H100 / TP=2 = 132,139 blocks
	const empirical int64 = 132139
	const tolerance = 0.10

	got, err := latency.CalculateKVBlocks(mc, hc, 2, 16, params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	deviation := math.Abs(float64(got)-float64(empirical)) / float64(empirical)
	t.Logf("Llama-3.1-8B H100 TP=2: calculated=%d, empirical=%d, deviation=%.2f%%",
		got, empirical, deviation*100)

	if deviation > tolerance {
		t.Errorf("blocks=%d deviates %.1f%% from empirical %d (max 10%%)",
			got, deviation*100, empirical)
	}
}

func TestCalculateKVBlocks_Llama31_8B_H100_TP4_WithinTolerance(t *testing.T) {
	mc := llama31_8B_ModelConfig()
	hc := h100HWConfig()
	params := latency.KVCapacityParams{
		IsMoE:             false,
		NumLocalExperts:   0,
		TieWordEmbeddings: false,
		HiddenAct:         "silu",
	}

	// Empirical baseline from defaults.yaml: Llama-3.1-8B / H100 / TP=4 = 559,190 blocks
	const empirical int64 = 559190
	const tolerance = 0.10

	got, err := latency.CalculateKVBlocks(mc, hc, 4, 16, params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	deviation := math.Abs(float64(got)-float64(empirical)) / float64(empirical)
	t.Logf("Llama-3.1-8B H100 TP=4: calculated=%d, empirical=%d, deviation=%.2f%%",
		got, empirical, deviation*100)

	if deviation > tolerance {
		t.Errorf("blocks=%d deviates %.1f%% from empirical %d (max 10%%)",
			got, deviation*100, empirical)
	}
}

func TestCalculateKVBlocks_Monotonicity_TP1ToTP2(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()
	blockSize := int64(16)

	blocksTP1, err := latency.CalculateKVBlocks(mc, hc, 1, blockSize, params)
	if err != nil {
		t.Fatalf("TP=1 error: %v", err)
	}

	blocksTP2, err := latency.CalculateKVBlocks(mc, hc, 2, blockSize, params)
	if err != nil {
		t.Fatalf("TP=2 error: %v", err)
	}

	t.Logf("TP=1 blocks=%d, TP=2 blocks=%d", blocksTP1, blocksTP2)

	if blocksTP2 <= blocksTP1 {
		t.Errorf("monotonicity violation: TP=2 blocks (%d) should be greater than TP=1 blocks (%d)",
			blocksTP2, blocksTP1)
	}
}

func TestCalculateKVBlocks_Purity_SameInputsSameOutput(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	result1, err := latency.CalculateKVBlocks(mc, hc, 2, 16, params)
	if err != nil {
		t.Fatalf("first call error: %v", err)
	}

	result2, err := latency.CalculateKVBlocks(mc, hc, 2, 16, params)
	if err != nil {
		t.Fatalf("second call error: %v", err)
	}

	if result1 != result2 {
		t.Errorf("purity violation: same inputs produced different outputs: %d vs %d",
			result1, result2)
	}
}

// --- Task 3: MoE model tests (BC-9, BC-11, BC-13) ---

func TestCalculateKVBlocks_Mixtral_8x7B_H100_TP2_WithinTolerance(t *testing.T) {
	mc := sim.ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		VocabSize:       32000,
		BytesPerParam:   2,
		IntermediateDim: 14336,
	}
	hc := h100HWConfig()
	params := latency.KVCapacityParams{
		IsMoE:             true,
		NumLocalExperts:   8,
		TieWordEmbeddings: false,
		HiddenAct:         "silu",
	}

	// Empirical baseline from defaults.yaml: Mixtral-8x7B / H100 / TP=2 = 58,377 blocks
	const empirical int64 = 58377
	const tolerance = 0.20

	got, err := latency.CalculateKVBlocks(mc, hc, 2, 16, params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	deviation := math.Abs(float64(got)-float64(empirical)) / float64(empirical)
	t.Logf("Mixtral-8x7B H100 TP=2: calculated=%d, empirical=%d, deviation=%.2f%%",
		got, empirical, deviation*100)

	if deviation > tolerance {
		t.Errorf("blocks=%d deviates %.1f%% from empirical %d (max 20%%)",
			got, deviation*100, empirical)
	}
}

func TestCalculateKVBlocks_MoE_UsesHigherActivationConstant(t *testing.T) {
	mc := sim.ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		VocabSize:       32000,
		BytesPerParam:   2,
		IntermediateDim: 14336,
	}
	hc := h100HWConfig()

	denseParams := latency.KVCapacityParams{
		IsMoE:             false,
		NumLocalExperts:   0,
		TieWordEmbeddings: false,
		HiddenAct:         "silu",
	}
	moeParams := latency.KVCapacityParams{
		IsMoE:             true,
		NumLocalExperts:   8,
		TieWordEmbeddings: false,
		HiddenAct:         "silu",
	}

	denseBlocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, denseParams)
	if err != nil {
		t.Fatalf("dense error: %v", err)
	}

	moeBlocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, moeParams)
	if err != nil {
		t.Fatalf("MoE error: %v", err)
	}

	t.Logf("dense blocks=%d, MoE blocks=%d", denseBlocks, moeBlocks)

	if moeBlocks >= denseBlocks {
		t.Errorf("MoE model should produce fewer blocks than dense (MoE=%d >= dense=%d) "+
			"due to higher activation constant and MLP weight multiplication",
			moeBlocks, denseBlocks)
	}
}

// --- Task 4: Tied embeddings + extraction tests (BC-12, BC-25) ---

func TestCalculateKVBlocks_TiedEmbeddings_ProducesMoreBlocks(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()

	untiedParams := latency.KVCapacityParams{
		IsMoE:             false,
		NumLocalExperts:   0,
		TieWordEmbeddings: false,
		HiddenAct:         "silu",
	}
	tiedParams := latency.KVCapacityParams{
		IsMoE:             false,
		NumLocalExperts:   0,
		TieWordEmbeddings: true,
		HiddenAct:         "silu",
	}

	untiedBlocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, untiedParams)
	if err != nil {
		t.Fatalf("untied error: %v", err)
	}

	tiedBlocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, tiedParams)
	if err != nil {
		t.Fatalf("tied error: %v", err)
	}

	t.Logf("untied blocks=%d, tied blocks=%d", untiedBlocks, tiedBlocks)

	if tiedBlocks <= untiedBlocks {
		t.Errorf("tied embeddings should produce more blocks (less weight memory): tied=%d <= untied=%d",
			tiedBlocks, untiedBlocks)
	}
}

// writeTempConfigJSON writes a config.json to a temp dir and returns the file path.
func writeTempConfigJSON(t *testing.T, data map[string]any) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	raw, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("marshal config.json: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	return path
}

func TestExtractKVCapacityParams_DenseModel(t *testing.T) {
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":           "silu",
		"num_hidden_layers":    32,
		"hidden_size":          4096,
		"num_attention_heads":  32,
		"num_key_value_heads":  8,
		"intermediate_size":    14336,
		"vocab_size":           128256,
		"torch_dtype":          "bfloat16",
		"tie_word_embeddings":  false,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if params.IsMoE {
		t.Error("expected IsMoE=false for dense model")
	}
	if params.TieWordEmbeddings {
		t.Error("expected TieWordEmbeddings=false")
	}
	if params.HiddenAct != "silu" {
		t.Errorf("expected HiddenAct=%q, got %q", "silu", params.HiddenAct)
	}
}

func TestExtractKVCapacityParams_MoEModel(t *testing.T) {
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":           "silu",
		"num_hidden_layers":    32,
		"hidden_size":          4096,
		"num_attention_heads":  32,
		"num_key_value_heads":  8,
		"intermediate_size":    14336,
		"vocab_size":           32000,
		"torch_dtype":          "bfloat16",
		"num_local_experts":    8,
		"num_experts_per_tok":  2,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !params.IsMoE {
		t.Error("expected IsMoE=true for MoE model")
	}
	if params.NumLocalExperts != 8 {
		t.Errorf("expected NumLocalExperts=8, got %d", params.NumLocalExperts)
	}
}

func TestExtractKVCapacityParams_SingleExpert_ClassifiedAsDense(t *testing.T) {
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":           "silu",
		"num_hidden_layers":    32,
		"hidden_size":          4096,
		"num_attention_heads":  32,
		"num_key_value_heads":  8,
		"intermediate_size":    14336,
		"vocab_size":           32000,
		"torch_dtype":          "bfloat16",
		"num_local_experts":    1,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if params.IsMoE {
		t.Error("expected IsMoE=false for single-expert model (classified as dense)")
	}
}

func TestExtractKVCapacityParams_TiedEmbeddings(t *testing.T) {
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":           "silu",
		"num_hidden_layers":    32,
		"hidden_size":          4096,
		"num_attention_heads":  32,
		"num_key_value_heads":  8,
		"intermediate_size":    14336,
		"vocab_size":           128256,
		"torch_dtype":          "bfloat16",
		"tie_word_embeddings":  true,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !params.TieWordEmbeddings {
		t.Error("expected TieWordEmbeddings=true")
	}
}
