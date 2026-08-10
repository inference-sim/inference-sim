package latency_test

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/latency"
)

// modelConfigPath returns the path to a committed fixture config.json under
// model_configs/ (two levels up from sim/latency/).
func modelConfigPath(dir string) string {
	return filepath.Join("..", "..", "model_configs", dir, "config.json")
}

// TestGLM52FP8_ModelShape is the #1527 regression test over the committed
// GLM-5.2-FP8 fixture (glm_moe_dsa). It asserts the three shape facts the fixes
// turn on: F1 (explicit head_dim), F2 (MLA compressed-KV), F3 (dense/MoE split).
func TestGLM52FP8_ModelShape(t *testing.T) {
	cfg, err := latency.GetModelConfig(modelConfigPath("glm-5.2-fp8"))
	if err != nil {
		t.Fatalf("GetModelConfig(glm-5.2-fp8): %v", err)
	}

	// F1: explicit head_dim=192 is read (not hidden/heads = 6144/64 = 96).
	if cfg.HeadDim != 192 {
		t.Errorf("F1: HeadDim = %d, want 192 (explicit head_dim)", cfg.HeadDim)
	}
	if cfg.EffectiveHeadDim() != 192 {
		t.Errorf("F1: EffectiveHeadDim() = %d, want 192 (not 96=6144/64)", cfg.EffectiveHeadDim())
	}

	// F2: MLA latent fields are parsed.
	if cfg.KVLoraRank != 512 {
		t.Errorf("F2: KVLoraRank = %d, want 512", cfg.KVLoraRank)
	}
	if cfg.QKRopeHeadDim != 64 {
		t.Errorf("F2: QKRopeHeadDim = %d, want 64", cfg.QKRopeHeadDim)
	}

	// F3: dense-prefix count is parsed.
	if cfg.FirstKDenseReplace != 3 {
		t.Errorf("F3: FirstKDenseReplace = %d, want 3", cfg.FirstKDenseReplace)
	}

	// MoE detection: 256 routed experts.
	if !cfg.IsMoE() || cfg.NumLocalExperts != 256 {
		t.Errorf("expected MoE with 256 experts, got IsMoE=%v NumLocalExperts=%d", cfg.IsMoE(), cfg.NumLocalExperts)
	}

	// FP8 weight precision auto-detected (flat 1.0 byte/param — F4 documented approximation).
	if cfg.EffectiveWeightBytesPerParam() != 1.0 {
		t.Errorf("expected EffectiveWeightBytesPerParam=1.0 (FP8), got %v", cfg.EffectiveWeightBytesPerParam())
	}
}

// TestGLM52FP8_MLAKVBytes asserts the per-token MLA KV footprint over the
// committed fixture: (kv_lora_rank + qk_rope_head_dim) × num_layers × bytes,
// TP-invariant (F2 / BC-3 / BC-4). Compute bytes = bfloat16 = 2 (KV stays at
// compute dtype even though weights are FP8).
func TestGLM52FP8_MLAKVBytes(t *testing.T) {
	cfg, err := latency.GetModelConfig(modelConfigPath("glm-5.2-fp8"))
	if err != nil {
		t.Fatalf("GetModelConfig(glm-5.2-fp8): %v", err)
	}
	// BytesPerParam is the compute dtype (bfloat16 = 2), NOT the FP8 weight precision.
	if cfg.BytesPerParam != 2 {
		t.Fatalf("precondition: expected BytesPerParam=2 (bfloat16 compute), got %v", cfg.BytesPerParam)
	}
	// (512 + 64) × 78 layers × 2 bytes = 89856 bytes/token, replicated across TP.
	wantPerToken := float64((512 + 64) * 78 * 2)
	wantPerLayer := float64((512 + 64) * 2)
	for _, tp := range []int{1, 2, 4, 8, 16} {
		got, err := latency.KVBytesPerToken(*cfg, tp)
		if err != nil {
			t.Fatalf("KVBytesPerToken(TP=%d): %v", tp, err)
		}
		if got != wantPerToken {
			t.Errorf("MLA KVBytesPerToken(TP=%d) = %v, want %v (TP-invariant)", tp, got, wantPerToken)
		}
		if perLayer := got / 78; perLayer != wantPerLayer {
			t.Errorf("MLA per-layer bytes = %v, want %v", perLayer, wantPerLayer)
		}
	}
}

// TestGLM52FP8_WeightSplit asserts F3 (BC-6): the weight estimate reflects
// exactly 3 dense + 75 MoE layers. Because computeModelWeightBytes is unexported,
// the split is verified end-to-end via CalculateKVBlocks: the GLM-5.2 fixture at
// first_k_dense_replace=3 must fit STRICTLY MORE KV blocks than the same model
// with all 78 layers MoE (the pre-F3 behavior), because 3 expensive 256-expert
// layers are replaced by 3 cheap dense layers. A weak "< all-MoE" inequality is
// avoided: this compares the SAME model with and without the split, isolating the
// 3-layer delta, so a wrong dense-layer count would change the gap observably.
func TestGLM52FP8_WeightSplit(t *testing.T) {
	cfg, err := latency.GetModelConfig(modelConfigPath("glm-5.2-fp8"))
	if err != nil {
		t.Fatalf("GetModelConfig(glm-5.2-fp8): %v", err)
	}
	params, err := latency.ExtractKVCapacityParamsFromFile(modelConfigPath("glm-5.2-fp8"))
	if err != nil {
		t.Fatalf("ExtractKVCapacityParams(glm-5.2-fp8): %v", err)
	}

	// GLM-5.2-FP8 (~743B) requires very high TP to fit; use TP=64 to leave a
	// positive KV budget so the block counts are all computable.
	hc := validHWConfig()
	const tp = 64

	// Sweep FirstKDenseReplace = 0..4 and record the resulting block counts. This
	// pins the EXACT K=3 split, not merely "split > all-MoE": each dense-prefix
	// layer replaces one 256-expert MoE layer with a cheap dense layer, removing a
	// fixed weight delta, so the KV-block gain per dense layer must be CONSTANT
	// (block count linear in K). A wrong dense-layer count (off-by-one, wrong
	// clamp, or nonlinear split) would break the equal-marginals law by thousands
	// of blocks. This is a first-principles law, robust to the CalculateKVBlocks
	// overhead/truncation constants (they cancel in the per-layer differences).
	blocks := make([]int64, 5)
	for k := 0; k <= 4; k++ {
		mc := *cfg
		mc.FirstKDenseReplace = k
		b, err := latency.CalculateKVBlocks(mc, hc, tp, 1, 16, 0.9, params)
		if err != nil {
			t.Fatalf("K=%d CalculateKVBlocks: %v", k, err)
		}
		blocks[k] = b
	}
	t.Logf("GLM-5.2-FP8 TP=%d blocks by FirstKDenseReplace: %v", tp, blocks)

	// The committed fixture has first_k_dense_replace=3, so K=3 must land strictly
	// between all-MoE (K=0) and K=4.
	if blocks[0] >= blocks[3] || blocks[3] >= blocks[4] {
		t.Errorf("F3: block count must be strictly increasing in K around K=3; got K0=%d K3=%d K4=%d",
			blocks[0], blocks[3], blocks[4])
	}

	// Equal-marginals law: blocks[k]-blocks[k-1] is constant (±1 for the two integer
	// truncations inside CalculateKVBlocks). Verifies each dense layer contributes the
	// SAME weight delta ⇒ exactly K layers are converted, linearly.
	perLayer := blocks[1] - blocks[0]
	if perLayer <= 0 {
		t.Fatalf("F3: one dense layer must add KV blocks, got marginal %d", perLayer)
	}
	for k := 2; k <= 4; k++ {
		marginal := blocks[k] - blocks[k-1]
		if diff := marginal - perLayer; diff < -1 || diff > 1 {
			t.Errorf("F3: per-dense-layer block gain must be constant (linear split); marginal at K=%d was %d, first-layer marginal was %d",
				k, marginal, perLayer)
		}
	}

	// And the K=3 total gain equals 3× the per-layer marginal (±1 per truncation).
	if got, want := blocks[3]-blocks[0], 3*perLayer; got < want-2 || got > want+2 {
		t.Errorf("F3: K=3 gain (%d) must equal 3 dense layers × per-layer gain (%d)±2", got, want)
	}
}

// TestDeepSeekV2Lite_MLAAndDenseSplit exercises the committed deepseek-v2-lite
// fixture end-to-end: it is an MLA model (kv_lora_rank=512) with a dense prefix
// (first_k_dense_replace=1). This confirms the MLA branch is selected and the
// dense/MoE split applies for a second, independent architecture.
func TestDeepSeekV2Lite_MLAAndDenseSplit(t *testing.T) {
	cfg, err := latency.GetModelConfig(modelConfigPath("deepseek-v2-lite"))
	if err != nil {
		t.Fatalf("GetModelConfig(deepseek-v2-lite): %v", err)
	}
	if cfg.KVLoraRank != 512 || cfg.QKRopeHeadDim != 64 {
		t.Errorf("expected MLA fields (512, 64), got (%d, %d)", cfg.KVLoraRank, cfg.QKRopeHeadDim)
	}
	if cfg.FirstKDenseReplace != 1 {
		t.Errorf("expected FirstKDenseReplace=1, got %d", cfg.FirstKDenseReplace)
	}

	// MLA KV bytes: (512 + 64) × 27 layers × 2 bytes, TP-invariant.
	want := float64((512 + 64) * 27 * 2)
	for _, tp := range []int{1, 2, 4} {
		got, err := latency.KVBytesPerToken(*cfg, tp)
		if err != nil {
			t.Fatalf("KVBytesPerToken(TP=%d): %v", tp, err)
		}
		if got != want {
			t.Errorf("deepseek-v2-lite MLA KVBytesPerToken(TP=%d) = %v, want %v", tp, got, want)
		}
	}

	// Auto KV-block capacity path runs without error.
	params, err := latency.ExtractKVCapacityParamsFromFile(modelConfigPath("deepseek-v2-lite"))
	if err != nil {
		t.Fatalf("ExtractKVCapacityParams(deepseek-v2-lite): %v", err)
	}
	blocks, err := latency.CalculateKVBlocks(*cfg, validHWConfig(), 1, 1, 16, 0.9, params)
	if err != nil {
		t.Fatalf("CalculateKVBlocks(deepseek-v2-lite): %v", err)
	}
	if blocks <= 0 {
		t.Errorf("expected positive KV blocks for deepseek-v2-lite, got %d", blocks)
	}
}
