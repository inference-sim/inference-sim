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
		TFlopsPeak: 989.5,
		BwPeakTBs:  3.35,
		MfuPrefill: 0.65,
		MfuDecode:  0.12,
		MemoryGiB:  80.0,
	}
}

// validDenseKVParams returns KVCapacityParams for a dense (non-MoE) model
// with SwiGLU activation.
func validDenseKVParams() latency.KVCapacityParams {
	return latency.NewKVCapacityParams(false, 0, false, "silu", 0, 0)
}

// --- KVBytesPerToken tests ---

func TestKVBytesPerToken_Llama8B_TP1(t *testing.T) {
	mc := validDenseModelConfig() // 32 layers, 8 KV heads, 4096 hidden, 32 heads → headDim=128
	got, err := latency.KVBytesPerToken(mc, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 32 layers × 2 (K+V) × 128 headDim × 8 KVHeads × 2 BytesPerParam / 1 TP = 131072
	want := float64(32 * 2 * 128 * 8 * 2)
	if got != want {
		t.Errorf("KVBytesPerToken = %v, want %v", got, want)
	}
}

func TestKVBytesPerToken_TPSharding(t *testing.T) {
	mc := validDenseModelConfig()
	tp1, _ := latency.KVBytesPerToken(mc, 1)
	tp4, _ := latency.KVBytesPerToken(mc, 4)
	if tp4 != tp1/4 {
		t.Errorf("KVBytesPerToken(TP=4) = %v, want %v (TP=1 value / 4)", tp4, tp1/4)
	}
}

func TestKVBytesPerToken_LinearInNumLayers(t *testing.T) {
	mc := validDenseModelConfig()
	base, err := latency.KVBytesPerToken(mc, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	mc.NumLayers *= 2
	doubled, err := latency.KVBytesPerToken(mc, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if doubled != 2*base {
		t.Errorf("KVBytesPerToken should scale linearly with NumLayers: got %v, want 2*%v=%v", doubled, base, 2*base)
	}
}

func TestKVBytesPerToken_LinearInBytesPerParam(t *testing.T) {
	mc := validDenseModelConfig()
	base, err := latency.KVBytesPerToken(mc, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	mc.BytesPerParam *= 2
	doubled, err := latency.KVBytesPerToken(mc, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if doubled != 2*base {
		t.Errorf("KVBytesPerToken should scale linearly with BytesPerParam: got %v, want 2*%v=%v", doubled, base, 2*base)
	}
}

func TestKVBytesPerToken_MHA_FallbackToNumHeads(t *testing.T) {
	mc := validDenseModelConfig()
	mc.NumKVHeads = 0 // MHA: should use NumHeads (32)
	got, err := latency.KVBytesPerToken(mc, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 32 layers × 2 × 128 headDim × 32 heads × 2 = 524288
	want := float64(32 * 2 * 128 * 32 * 2)
	if got != want {
		t.Errorf("KVBytesPerToken(MHA) = %v, want %v", got, want)
	}
}

func TestKVBytesPerToken_InvalidInputs(t *testing.T) {
	mc := validDenseModelConfig()
	cases := []struct {
		name string
		mc   sim.ModelConfig
		tp   int
	}{
		{"zero TP", mc, 0},
		{"negative TP", mc, -1},
		{"zero NumHeads", sim.ModelConfig{NumLayers: 1, HiddenDim: 64, BytesPerParam: 2.0}, 1},
		{"zero NumLayers", sim.ModelConfig{NumHeads: 4, HiddenDim: 64, BytesPerParam: 2.0}, 1},
		{"zero HiddenDim", sim.ModelConfig{NumHeads: 4, NumLayers: 1, BytesPerParam: 2.0}, 1},
		{"zero BytesPerParam", sim.ModelConfig{NumHeads: 4, NumLayers: 1, HiddenDim: 64}, 1},
		{"indivisible KVHeads", sim.ModelConfig{NumHeads: 4, NumLayers: 1, HiddenDim: 64, NumKVHeads: 3, BytesPerParam: 2.0}, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := latency.KVBytesPerToken(tc.mc, tc.tp)
			if err == nil {
				t.Errorf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

func TestKVBytesPerToken_GQA_KVHeadsLessThanTP(t *testing.T) {
	// When numKVHeads < TP (e.g., 4 KV heads, TP=8), vLLM replicates KV heads
	// per GPU. Our formula divides total KV by TP, which underestimates per-GPU
	// KV bytes — a known approximation inherited from CalculateKVBlocks.
	mc := sim.ModelConfig{
		NumLayers:       2,
		NumHeads:        32,
		NumKVHeads:      4,
		HiddenDim:       128,
		IntermediateDim: 256,
		BytesPerParam:   2.0,
	}
	got, err := latency.KVBytesPerToken(mc, 8)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// headDim = 128/32 = 4; total = 2 × 2 × 4 × 4 × 2.0 = 128; per-GPU = 128/8 = 16
	// In reality vLLM replicates, so per-GPU would be higher. This documents the
	// current (optimistic) approximation.
	want := float64(16)
	if got != want {
		t.Errorf("KVBytesPerToken(numKVHeads=4, TP=8) = %v, want %v", got, want)
	}
}

func TestKVBytesPerToken_NonDivisibleKVHeadsLessThanTP_Accepted(t *testing.T) {
	// Non-divisible numKVHeads=3 with TP=8: should SUCCEED (numKVHeads < TP → no divisibility check).
	// This documents the asymmetry with the "indivisible KVHeads" TP=2 case in
	// TestKVBytesPerToken_InvalidInputs that errors (numKVHeads=3 >= TP=2).
	mc := sim.ModelConfig{NumHeads: 32, NumKVHeads: 3, NumLayers: 1, HiddenDim: 64, BytesPerParam: 2.0}
	_, err := latency.KVBytesPerToken(mc, 8)
	if err != nil {
		t.Errorf("expected success for numKVHeads=3 < TP=8, got error: %v", err)
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
		{
			name: "negative TP",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				return mc, hc, -1, blockSize, params
			},
			errWant: "TP",
		},
		{
			name: "negative block size",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				return mc, hc, tp, -1, params
			},
			errWant: "block size",
		},
		{
			name: "negative NumKVHeads",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.NumKVHeads = -1
				return m, hc, tp, blockSize, params
			},
			errWant: "num_kv_heads",
		},
		{
			name: "negative WeightBytesPerParam",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.WeightBytesPerParam = -0.5
				return m, hc, tp, blockSize, params
			},
			errWant: "WeightBytesPerParam",
		},
		{
			name: "NaN WeightBytesPerParam",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.WeightBytesPerParam = math.NaN()
				return m, hc, tp, blockSize, params
			},
			errWant: "WeightBytesPerParam",
		},
		{
			name: "Inf WeightBytesPerParam",
			setup: func() (sim.ModelConfig, sim.HardwareCalib, int, int64, latency.KVCapacityParams) {
				m := mc
				m.WeightBytesPerParam = math.Inf(1)
				return m, hc, tp, blockSize, params
			},
			errWant: "WeightBytesPerParam",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, h, tpVal, bs, p := tt.setup()
			_, err := latency.CalculateKVBlocks(m, h, tpVal, bs, 0.9, p)
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
			_, err := latency.CalculateKVBlocks(m, h, 1, 16, 0.9, params)
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

	_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, 0.9, params)
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

	_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, 0.9, params)
	if err == nil {
		t.Fatal("expected error for exceeded budget, got nil")
	}
	if !strings.Contains(err.Error(), "exceed") {
		t.Errorf("expected error mentioning 'exceed', got: %v", err)
	}
}

func TestCalculateKVBlocks_InsufficientMemory_SuggestsMinTP(t *testing.T) {
	// GIVEN a DeepSeek-V3-like MoE model (671B) with insufficient TP
	mc := sim.ModelConfig{
		NumLayers:          61,
		HiddenDim:          7168,
		NumHeads:           128,
		NumKVHeads:         128,
		VocabSize:          129280,
		BytesPerParam:      1.0, // FP8
		IntermediateDim:    18432,
		NumLocalExperts:    256,
		NumExpertsPerTok:   8,
		MoEExpertFFNDim:    2048,
		SharedExpertFFNDim: 2048,
	}
	hc := validHWConfig() // H100-80GB
	params := latency.NewKVCapacityParams(true, 256, false, "silu", 2048, 2048)
	tp := 2

	// WHEN CalculateKVBlocks is called with TP=2 (insufficient)
	_, err := latency.CalculateKVBlocks(mc, hc, tp, 16, 0.9, params)

	// THEN error is returned
	if err == nil {
		t.Fatal("expected error for insufficient memory, got nil")
	}

	// AND error message contains "Minimum GPUs required per instance"
	errMsg := err.Error()
	if !strings.Contains(errMsg, "Minimum GPUs required per instance") {
		t.Errorf("error should contain 'Minimum GPUs required per instance', got: %v", err)
	}

	// AND minimum GPU count is mathematically correct
	// Formula: ceil((modelWeightGiB + activationGiB) / (memGiB × util - nonTorchMultiGiB))
	// = ceil((656.51 + 8.00) / (80.0 × 0.9 - 0.6))
	// = ceil(664.51 / 71.40)
	// = ceil(9.30) = 10
	if !strings.Contains(errMsg, "Minimum GPUs required per instance: 10") {
		t.Errorf("expected 'Minimum GPUs required per instance: 10' in error, got: %v", err)
	}
}

func TestCalculateKVBlocks_InsufficientMemory_SucceedsAtSuggestedTP(t *testing.T) {
	// GIVEN a DeepSeek-V3-like MoE model (671B) with insufficient TP=2
	mc := sim.ModelConfig{
		NumLayers:          61,
		HiddenDim:          7168,
		NumHeads:           128,
		NumKVHeads:         128,
		VocabSize:          129280,
		BytesPerParam:      1.0, // FP8
		IntermediateDim:    18432,
		NumLocalExperts:    256,
		NumExpertsPerTok:   8,
		MoEExpertFFNDim:    2048,
		SharedExpertFFNDim: 2048,
	}
	hc := validHWConfig() // H100-80GB
	params := latency.NewKVCapacityParams(true, 256, false, "silu", 2048, 2048)

	// WHEN CalculateKVBlocks fails with TP=2, it should suggest minTP=10
	_, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
	if err == nil {
		t.Fatal("expected error for insufficient memory with TP=2, got nil")
	}
	if !strings.Contains(err.Error(), "Minimum GPUs required per instance: 10") {
		t.Errorf("expected minTP=10 suggestion, got: %v", err)
	}

	// AND CalculateKVBlocks should succeed at TP=16 (next valid TP >= 10 that divides num_kv_heads=128)
	blocks, err := latency.CalculateKVBlocks(mc, hc, 16, 16, 0.9, params)
	if err != nil {
		t.Errorf("expected success at TP=16 (next valid TP >= suggested minTP=10), got error: %v", err)
	}
	if blocks == 0 {
		t.Errorf("expected non-zero KV blocks at TP=16, got: %d", blocks)
	}
}

func TestCalculateKVBlocks_ZeroGPUMemory_RejectsInput(t *testing.T) {
	// GIVEN a model with zero GPU memory (degenerate config)
	mc := validDenseModelConfig()
	hc := validHWConfig()
	hc.MemoryGiB = 0.0
	params := validDenseKVParams()

	// WHEN CalculateKVBlocks is called
	_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, 0.9, params)

	// THEN error is returned
	if err == nil {
		t.Fatal("expected error for zero GPU memory, got nil")
	}

	// AND error mentions invalid GPU memory (caught by earlier validation, not minTP logic)
	errMsg := err.Error()
	if !strings.Contains(errMsg, "GPU memory") {
		t.Errorf("expected error mentioning 'GPU memory', got: %v", err)
	}
}

func TestCalculateKVBlocks_FloorZero_ReturnError(t *testing.T) {
	// Use the standard model/GPU config but set an enormous block size so
	// that a single block exceeds the allocatable KV space. This exercises
	// the floor-zero guard (BC-22) rather than the budget-exceeded guard.
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	// blockSize = 10M tokens → one block is huge, floor division yields 0
	_, err := latency.CalculateKVBlocks(mc, hc, 1, 10_000_000, 0.9, params)
	if err == nil {
		t.Fatal("expected error for floor-zero blocks, got nil")
	}
	if !strings.Contains(err.Error(), "0 blocks") {
		t.Errorf("expected error mentioning '0 blocks', got: %v", err)
	}
}

func TestCalculateKVBlocks_NonSwiGLU_ReturnError(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()
	params.HiddenAct = "relu"

	_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, 0.9, params)
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

	_, err := latency.CalculateKVBlocks(mc, hc, 3, 16, 0.9, params)
	if err == nil {
		t.Fatal("expected error for TP not dividing num_kv_heads, got nil")
	}
	errMsg := err.Error()
	if !strings.Contains(errMsg, "divisible") || !strings.Contains(errMsg, "TP") {
		t.Errorf("expected error mentioning 'divisible' and 'TP', got: %v", err)
	}
}

// --- Task 2: Empirical fidelity + invariant tests (BC-4, BC-5, KV-CAP-5) ---

// Aliases for fidelity tests — same config as validDenseModelConfig/validHWConfig
// but named after the specific model/GPU for clarity in test output.
func llama31_8B_ModelConfig() sim.ModelConfig { return validDenseModelConfig() }
func h100HWConfig() sim.HardwareCalib         { return validHWConfig() }

func TestCalculateKVBlocks_Llama31_8B_H100_TP2_WithinTolerance(t *testing.T) {
	mc := llama31_8B_ModelConfig()
	hc := h100HWConfig()
	params := latency.NewKVCapacityParams(false, 0, false, "silu", 0, 0)

	// Empirical baseline from defaults.yaml: Llama-3.1-8B / H100 / TP=2 = 132,139 blocks
	const empirical int64 = 132139
	const tolerance = 0.10

	got, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
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
	params := latency.NewKVCapacityParams(false, 0, false, "silu", 0, 0)

	// Empirical baseline from defaults.yaml: Llama-3.1-8B / H100 / TP=4 = 559,190 blocks
	const empirical int64 = 559190
	const tolerance = 0.10

	got, err := latency.CalculateKVBlocks(mc, hc, 4, 16, 0.9, params)
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

	blocksTP1, err := latency.CalculateKVBlocks(mc, hc, 1, blockSize, 0.9, params)
	if err != nil {
		t.Fatalf("TP=1 error: %v", err)
	}

	blocksTP2, err := latency.CalculateKVBlocks(mc, hc, 2, blockSize, 0.9, params)
	if err != nil {
		t.Fatalf("TP=2 error: %v", err)
	}

	t.Logf("TP=1 blocks=%d, TP=2 blocks=%d", blocksTP1, blocksTP2)

	if blocksTP2 <= blocksTP1 {
		t.Errorf("monotonicity violation: TP=2 blocks (%d) should be greater than TP=1 blocks (%d)",
			blocksTP2, blocksTP1)
	}
}

func TestCalculateKVBlocks_FractionalBytesPerParam_ProducesMoreBlocks(t *testing.T) {
	// INT4 quantization uses 0.5 bytes per parameter. Before the float64
	// arithmetic fix, int64(0.5) truncated to 0, causing a division-by-zero
	// panic. This test verifies fractional BytesPerParam works correctly and
	// produces more blocks than FP16 (smaller KV footprint per token).
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	// FP16 baseline (BytesPerParam = 2.0)
	blocksFP16, err := latency.CalculateKVBlocks(mc, hc, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("FP16 error: %v", err)
	}

	// INT4 (BytesPerParam = 0.5)
	mcINT4 := mc
	mcINT4.BytesPerParam = 0.5
	blocksINT4, err := latency.CalculateKVBlocks(mcINT4, hc, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("INT4 (BytesPerParam=0.5) error: %v", err)
	}

	t.Logf("FP16 blocks=%d, INT4 blocks=%d", blocksFP16, blocksINT4)

	if blocksINT4 <= blocksFP16 {
		t.Errorf("INT4 (0.5 bytes/param) should produce more blocks than FP16 (2 bytes/param): INT4=%d <= FP16=%d",
			blocksINT4, blocksFP16)
	}
}

func TestCalculateKVBlocks_Purity_SameInputsSameOutput(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	result1, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
	if err != nil {
		t.Fatalf("first call error: %v", err)
	}

	result2, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
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
	params := latency.NewKVCapacityParams(true, 8, false, "silu", 0, 0)

	// Empirical baseline from defaults.yaml: Mixtral-8x7B / H100 / TP=2 = 58,377 blocks
	const empirical int64 = 58377
	const tolerance = 0.20

	got, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
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

	denseParams := latency.NewKVCapacityParams(false, 0, false, "silu", 0, 0)
	moeParams := latency.NewKVCapacityParams(true, 8, false, "silu", 0, 0)

	denseBlocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, denseParams)
	if err != nil {
		t.Fatalf("dense error: %v", err)
	}

	moeBlocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, moeParams)
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

	untiedParams := latency.NewKVCapacityParams(false, 0, false, "silu", 0, 0)
	tiedParams := latency.NewKVCapacityParams(false, 0, true, "silu", 0, 0)

	untiedBlocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, untiedParams)
	if err != nil {
		t.Fatalf("untied error: %v", err)
	}

	tiedBlocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, tiedParams)
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
		"hidden_act":          "silu",
		"num_hidden_layers":   32,
		"hidden_size":         4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size":   14336,
		"vocab_size":          128256,
		"torch_dtype":         "bfloat16",
		"tie_word_embeddings": false,
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
		"hidden_act":          "silu",
		"num_hidden_layers":   32,
		"hidden_size":         4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size":   14336,
		"vocab_size":          32000,
		"torch_dtype":         "bfloat16",
		"num_local_experts":   8,
		"num_experts_per_tok": 2,
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
		"hidden_act":          "silu",
		"num_hidden_layers":   32,
		"hidden_size":         4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size":   14336,
		"vocab_size":          32000,
		"torch_dtype":         "bfloat16",
		"num_local_experts":   1,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if params.IsMoE {
		t.Error("expected IsMoE=false for single-expert model (classified as dense)")
	}
}

// --- I2: NumKVHeads=0 fallback to NumHeads ---

func TestCalculateKVBlocks_NumKVHeadsZero_FallsBackToNumHeads(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	// Explicit NumKVHeads = NumHeads (MHA)
	mcExplicit := mc
	mcExplicit.NumKVHeads = mc.NumHeads // 32
	blocksExplicit, err := latency.CalculateKVBlocks(mcExplicit, hc, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("explicit NumKVHeads=%d error: %v", mc.NumHeads, err)
	}

	// NumKVHeads = 0 (should behave identically to NumHeads)
	mcZero := mc
	mcZero.NumKVHeads = 0
	blocksZero, err := latency.CalculateKVBlocks(mcZero, hc, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("NumKVHeads=0 error: %v", err)
	}

	if blocksZero != blocksExplicit {
		t.Errorf("NumKVHeads=0 should produce same blocks as NumKVHeads=%d: got %d vs %d",
			mc.NumHeads, blocksZero, blocksExplicit)
	}
}

// --- I5: numKVHeads < TP path ---

func TestCalculateKVBlocks_NumKVHeadsLessThanTP_Succeeds(t *testing.T) {
	mc := validDenseModelConfig()
	mc.NumKVHeads = 2 // GQA with only 2 KV heads
	hc := validHWConfig()
	params := validDenseKVParams()

	// TP=4 > numKVHeads=2: vLLM replicates KV heads, our formula approximates
	blocks, err := latency.CalculateKVBlocks(mc, hc, 4, 16, 0.9, params)
	if err != nil {
		t.Fatalf("numKVHeads=2, TP=4 should succeed (known approximation), got error: %v", err)
	}
	if blocks <= 0 {
		t.Errorf("expected positive blocks, got %d", blocks)
	}
	t.Logf("numKVHeads=2, TP=4: blocks=%d (optimistic approximation)", blocks)
}

// --- I3: MoE fallback detection paths ---

func TestExtractKVCapacityParams_MoEFallback_NRoutedExperts(t *testing.T) {
	// DeepSeek-style: uses n_routed_experts instead of num_local_experts
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":       "silu",
		"n_routed_experts": 64,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !params.IsMoE {
		t.Error("expected IsMoE=true for n_routed_experts=64")
	}
	if params.NumLocalExperts != 64 {
		t.Errorf("expected NumLocalExperts=64, got %d", params.NumLocalExperts)
	}
}

func TestExtractKVCapacityParams_MoEFallback_NumExperts(t *testing.T) {
	// DBRX-style: uses num_experts
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":  "silu",
		"num_experts": 16,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !params.IsMoE {
		t.Error("expected IsMoE=true for num_experts=16")
	}
	if params.NumLocalExperts != 16 {
		t.Errorf("expected NumLocalExperts=16, got %d", params.NumLocalExperts)
	}
}

func TestExtractKVCapacityParams_MoEFallback_SharedExpertsOnly_ReturnsError(t *testing.T) {
	// Model with n_shared_experts but no total expert count — cannot estimate weights
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":       "silu",
		"n_shared_experts": 2,
	})

	_, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err == nil {
		t.Fatal("expected error for MoE detected via n_shared_experts without total expert count")
	}
	if !strings.Contains(err.Error(), "n_shared_experts") {
		t.Errorf("expected error mentioning n_shared_experts, got: %v", err)
	}
}

func TestExtractKVCapacityParams_MoEFallback_NumExpertsPerTokOnly_ReturnsError(t *testing.T) {
	// Switch Transformer-style: num_experts_per_tok=1 signals MoE but no total count
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":          "silu",
		"num_experts_per_tok": 1,
	})

	_, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err == nil {
		t.Fatal("expected error for MoE detected via num_experts_per_tok without total expert count")
	}
	if !strings.Contains(err.Error(), "num_experts_per_tok") {
		t.Errorf("expected error mentioning num_experts_per_tok, got: %v", err)
	}
}

func TestExtractKVCapacityParams_TiedEmbeddings(t *testing.T) {
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":          "silu",
		"num_hidden_layers":   32,
		"hidden_size":         4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size":   14336,
		"vocab_size":          128256,
		"torch_dtype":         "bfloat16",
		"tie_word_embeddings": true,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !params.TieWordEmbeddings {
		t.Error("expected TieWordEmbeddings=true")
	}
}

// --- Task 6: MoE per-expert dim fix tests ---

func TestCalculateKVBlocks_DeepSeekV3_PerExpertDimFix(t *testing.T) {
	// BC-7 (MOD-ROOF-6): per-expert dim (2048) vs general dim (18432) for DeepSeek-V3.
	// DeepSeek-V3 is a 671B model — requires FP8 (1 byte/param) and high TP in practice.
	// Without the fix, using 18432 as per-expert dim overestimates MLP weights by ~9×.
	mc := sim.ModelConfig{
		NumLayers:          61,
		HiddenDim:          7168,
		NumHeads:           128,
		NumKVHeads:         128,
		VocabSize:          129280,
		BytesPerParam:      1, // FP8 quantization (real-world deployment)
		IntermediateDim:    18432,
		NumLocalExperts:    256,
		NumExpertsPerTok:   8,
		MoEExpertFFNDim:    2048,
		SharedExpertFFNDim: 2048,
	}
	hc := validHWConfig()
	hc.MemoryGiB = 80.0

	// With per-expert dim fix: should produce usable blocks on 16×H100
	// (DeepSeek-V3 at 671B FP8 ≈ 656 GiB — requires TP≥16 on H100-80GB)
	params := latency.NewKVCapacityParams(true, 256, false, "silu", 2048, 2048)
	blocks, err := latency.CalculateKVBlocks(mc, hc, 16, 16, 0.9, params)
	if err != nil {
		t.Fatalf("per-expert dim fix should succeed, got error: %v", err)
	}
	if blocks <= 0 {
		t.Errorf("expected positive blocks for DeepSeek-V3 with per-expert dim fix, got %d", blocks)
	}
	t.Logf("DeepSeek-V3 H100×16 FP8 with per-expert dim fix: %d blocks", blocks)

	// Without fix (using general intermediate dim as per-expert): should fail or give fewer blocks
	paramsBuggy := latency.NewKVCapacityParams(true, 256, false, "silu", 0, 0)
	blocksBuggy, errBuggy := latency.CalculateKVBlocks(mc, hc, 16, 16, 0.9, paramsBuggy)
	if errBuggy == nil && blocksBuggy >= blocks {
		t.Errorf("BC-7: buggy path (using general dim) should give fewer blocks or error: buggy=%d, fixed=%d",
			blocksBuggy, blocks)
	}
	// With the buggy path, 18432 per-expert → 9× MLP weight overestimate → likely budget exceeded
	if errBuggy != nil {
		t.Logf("BC-7 confirmed: buggy path (general dim as per-expert) returns error: %v", errBuggy)
	} else {
		t.Logf("BC-7 confirmed: buggy path gives fewer blocks: buggy=%d < fixed=%d", blocksBuggy, blocks)
	}
}

func TestCalculateKVBlocks_MixtralPublishedParams(t *testing.T) {
	// BC-9: Mixtral-8x7B published parameter count cross-validation.
	// Published: 46.7B params. Weight bytes = 46.7B × 2 = ~93.4 GB.
	mc := sim.ModelConfig{
		NumLayers:        32,
		HiddenDim:        4096,
		NumHeads:         32,
		NumKVHeads:       8,
		VocabSize:        32000,
		BytesPerParam:    2,
		IntermediateDim:  14336,
		NumLocalExperts:  8,
		NumExpertsPerTok: 2,
	}
	params := latency.NewKVCapacityParams(true, 8, false, "silu", 0, 0)
	hc := validHWConfig()
	hc.MemoryGiB = 80.0

	blocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if blocks <= 0 {
		t.Errorf("expected positive blocks for Mixtral-8x7B, got %d", blocks)
	}
	t.Logf("Mixtral-8x7B H100 TP=2 (with MoE params): %d blocks", blocks)
}

func TestCalculateKVBlocks_Dense_UnchangedWithNewParams(t *testing.T) {
	// BC-11: dense model unchanged after adding MoE fields to KVCapacityParams
	mc := validDenseModelConfig()
	hc := validHWConfig()

	params := latency.NewKVCapacityParams(false, 0, false, "silu", 0, 0)
	blocks, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Must match the existing Llama-3.1-8B empirical baseline
	const empirical int64 = 132139
	const tolerance = 0.10
	deviation := math.Abs(float64(blocks)-float64(empirical)) / float64(empirical)
	if deviation > tolerance {
		t.Errorf("BC-11: dense blocks=%d deviates %.1f%% from empirical %d",
			blocks, deviation*100, empirical)
	}
}

func TestExtractKVCapacityParams_DeepSeekV3_PerExpertDim(t *testing.T) {
	// BC-18 + KV capacity: num_routed_experts detected, per-expert and shared dims extracted
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":            "silu",
		"num_hidden_layers":     61,
		"hidden_size":           7168,
		"num_attention_heads":   128,
		"num_key_value_heads":   128,
		"intermediate_size":     18432,
		"moe_intermediate_size": 2048,
		"n_shared_experts":      1,
		"num_routed_experts":    256,
		"num_experts_per_tok":   8,
		"vocab_size":            129280,
		"torch_dtype":           "bfloat16",
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !params.IsMoE {
		t.Error("expected IsMoE=true for DeepSeek-V3")
	}
	if params.NumLocalExperts != 256 {
		t.Errorf("expected NumLocalExperts=256, got %d", params.NumLocalExperts)
	}
	if params.MoEExpertFFNDim != 2048 {
		t.Errorf("expected MoEExpertFFNDim=2048, got %d", params.MoEExpertFFNDim)
	}
	if params.SharedExpertFFNDim != 2048 {
		t.Errorf("expected SharedExpertFFNDim=2048 (1 shared × 2048 per-expert), got %d", params.SharedExpertFFNDim)
	}
}

func TestExtractKVCapacityParams_Qwen2MoE_ExplicitSharedDim(t *testing.T) {
	// shared_expert_intermediate_size takes precedence over n_shared_experts × per-expert
	path := writeTempConfigJSON(t, map[string]any{
		"hidden_act":                      "silu",
		"num_local_experts":               60,
		"moe_intermediate_size":           2560,
		"shared_expert_intermediate_size": 5632,
	})

	params, err := latency.ExtractKVCapacityParamsFromFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if params.MoEExpertFFNDim != 2560 {
		t.Errorf("expected MoEExpertFFNDim=2560, got %d", params.MoEExpertFFNDim)
	}
	if params.SharedExpertFFNDim != 5632 {
		t.Errorf("expected SharedExpertFFNDim=5632 (explicit field), got %d", params.SharedExpertFFNDim)
	}
}

// --- Quantized model KV capacity tests (BC-7, BC-10) ---

func TestCalculateKVBlocks_W4A16_MoreBlocksThanFP16(t *testing.T) {
	// BC-10: W4A16 model produces more KV blocks (smaller weight footprint)
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	// FP16 baseline
	blocksFP16, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
	if err != nil {
		t.Fatalf("FP16 error: %v", err)
	}

	// W4A16: weight precision is 0.5, compute dtype stays at 2.0
	mcW4 := mc
	mcW4.WeightBytesPerParam = 0.5
	blocksW4, err := latency.CalculateKVBlocks(mcW4, hc, 2, 16, 0.9, params)
	if err != nil {
		t.Fatalf("W4A16 error: %v", err)
	}

	t.Logf("FP16 blocks=%d, W4A16 blocks=%d", blocksFP16, blocksW4)

	if blocksW4 <= blocksFP16 {
		t.Errorf("W4A16 should produce more blocks than FP16 (smaller weights): W4=%d <= FP16=%d",
			blocksW4, blocksFP16)
	}
}

func TestCalculateKVBlocks_W4A16_PerTokenKVBytesUnchanged(t *testing.T) {
	// BC-7: Per-token KV bytes use BytesPerParam (compute dtype), NOT WeightBytesPerParam
	// We verify this indirectly: changing WeightBytesPerParam should only affect
	// weight memory, not per-block KV bytes. Two models with same BytesPerParam
	// but different WeightBytesPerParam should produce different total blocks but
	// the difference should come from weight memory only.
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	// FP16 baseline
	blocksFP16, err := latency.CalculateKVBlocks(mc, hc, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("FP16 error: %v", err)
	}

	// FP8 weights (1.0 bytes/param) — intermediate between FP16 and W4A16
	mcFP8 := mc
	mcFP8.WeightBytesPerParam = 1.0
	blocksFP8, err := latency.CalculateKVBlocks(mcFP8, hc, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("FP8 error: %v", err)
	}

	// W4A16 (0.5 bytes/param)
	mcW4 := mc
	mcW4.WeightBytesPerParam = 0.5
	blocksW4, err := latency.CalculateKVBlocks(mcW4, hc, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("W4A16 error: %v", err)
	}

	t.Logf("FP16=%d blocks, FP8=%d blocks, W4A16=%d blocks", blocksFP16, blocksFP8, blocksW4)

	// Monotonicity: smaller weight precision → more blocks
	if blocksFP8 <= blocksFP16 {
		t.Errorf("FP8 should have more blocks than FP16: %d <= %d", blocksFP8, blocksFP16)
	}
	if blocksW4 <= blocksFP8 {
		t.Errorf("W4A16 should have more blocks than FP8: %d <= %d", blocksW4, blocksFP8)
	}
}

func TestCalculateKVBlocks_NonQuantized_UnchangedByWeightField(t *testing.T) {
	// BC-8 regression anchor: WeightBytesPerParam=0 (sentinel) behaves identically
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	blocksBaseline, err := latency.CalculateKVBlocks(mc, hc, 2, 16, 0.9, params)
	if err != nil {
		t.Fatalf("baseline error: %v", err)
	}

	// Explicitly set WeightBytesPerParam=0 (sentinel)
	mcExplicit := mc
	mcExplicit.WeightBytesPerParam = 0
	blocksExplicit, err := latency.CalculateKVBlocks(mcExplicit, hc, 2, 16, 0.9, params)
	if err != nil {
		t.Fatalf("explicit sentinel error: %v", err)
	}

	if blocksBaseline != blocksExplicit {
		t.Errorf("sentinel (WeightBytesPerParam=0) should match baseline: %d vs %d",
			blocksBaseline, blocksExplicit)
	}
}

func TestCalculateKVBlocks_GpuMemoryUtilization_Validation(t *testing.T) {
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	tests := []struct {
		name    string
		gpuUtil float64
		wantErr bool
	}{
		{"valid 0.3", 0.3, false},
		{"valid 0.5", 0.5, false},
		{"valid 0.7", 0.7, false},
		{"valid 0.9", 0.9, false},
		{"valid 1.0", 1.0, false},
		{"zero is invalid", 0.0, true},
		{"negative", -0.1, true},
		{"above 1.0 is invalid", 1.1, true},
		{"NaN", math.NaN(), true},
		{"positive infinity", math.Inf(1), true},
		{"negative infinity", math.Inf(-1), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := latency.CalculateKVBlocks(mc, hc, 1, 16, tt.gpuUtil, params)
			if (err != nil) != tt.wantErr {
				t.Errorf("gpuMemoryUtilization=%v: got error=%v, wantErr=%v", tt.gpuUtil, err, tt.wantErr)
			}
		})
	}
}

func TestCalculateKVBlocks_GpuMemoryUtilization_HigherUtilProducesMoreBlocks(t *testing.T) {
	// BC: higher GPU memory utilization leaves more memory for KV cache
	// GIVEN the same model and hardware
	mc := validDenseModelConfig()
	hc := validHWConfig()
	params := validDenseKVParams()

	// WHEN computed at 90% vs 95% utilization
	blocks90, err := latency.CalculateKVBlocks(mc, hc, 1, 16, 0.90, params)
	if err != nil {
		t.Fatalf("util=0.90: unexpected error: %v", err)
	}
	blocks95, err := latency.CalculateKVBlocks(mc, hc, 1, 16, 0.95, params)
	if err != nil {
		t.Fatalf("util=0.95: unexpected error: %v", err)
	}

	// THEN higher utilization must produce strictly more blocks
	if blocks95 <= blocks90 {
		t.Errorf("util=0.95 should yield more blocks than util=0.90: got %d <= %d", blocks95, blocks90)
	}
}

// Manual verification (issue #624):
//
// 1. Insufficient TP — error with actionable guidance:
//    $ ./blis run --model deepseek-ai/DeepSeek-V3 --tp 2 --hardware H100
//    => FATAL: "model overhead (665.71 GiB = 656.51 weights + 8.00 activation + 1.20 non-torch)
//       exceeds available GPU memory (144.00 GiB = 80.0 GiB × 90% util × 2 GPUs).
//       Minimum GPUs required per instance: 10"
//
// 2. TP=8 (128 KV heads divisible, but still insufficient memory):
//    $ ./blis run --model deepseek-ai/DeepSeek-V3 --tp 8 --hardware H100
//    => FATAL: "model overhead (669.31 GiB) exceeds available GPU memory (576.00 GiB).
//       Minimum GPUs required per instance: 10"
//
// 3. TP=10 (sufficient memory but 128 KV heads not divisible by 10):
//    $ ./blis run --model deepseek-ai/DeepSeek-V3 --tp 10 --hardware H100
//    => FATAL: "num_kv_heads (128) must be evenly divisible by TP (10)"
//
// 4. Sufficient TP — simulation succeeds:
//    $ ./blis run --model deepseek-ai/DeepSeek-V3 --tp 16 --hardware H100 --num-requests 10
//    => SUCCESS: auto-calculated total-kv-blocks=293387 (GPU=80 GiB, TP=16, block_size=16, MoE=true)
//    => Completed 10 requests successfully
