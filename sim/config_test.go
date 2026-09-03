package sim

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewKVCacheConfig_FieldEquivalence(t *testing.T) {
	got := NewKVCacheConfig(100, 16, 50, 0.9, 100.0, 500)
	want := KVCacheConfig{
		TotalKVBlocks:         100,
		BlockSizeTokens:       16,
		KVCPUBlocks:           50,
		KVOffloadThreshold:    0.9,
		KVTransferBandwidth:   100.0,
		KVTransferBaseLatency: 500,
	}
	assert.Equal(t, want, got)
}

func TestNewBatchConfig_FieldEquivalence(t *testing.T) {
	got := NewBatchConfig(10, 1000, 200)
	want := BatchConfig{
		MaxNumSeqs:                10,
		MaxNumBatchedTokens:       1000,
		LongPrefillTokenThreshold: 200,
	}
	assert.Equal(t, want, got)
}

func TestNewLatencyCoeffs_FieldEquivalence(t *testing.T) {
	beta := []float64{1000, 10, 2}
	alpha := []float64{500, 1, 1000}
	got := NewLatencyCoeffs(beta, alpha)
	want := LatencyCoeffs{BetaCoeffs: beta, AlphaCoeffs: alpha}
	assert.Equal(t, want, got)
}

func TestNewModelHardwareConfig_FieldEquivalence(t *testing.T) {
	mc := ModelConfig{NumLayers: 32}
	hw := HardwareCalib{TFlopsPeak: 1000.0, MemoryGiB: 80.0}
	got := NewModelHardwareConfig(mc, hw, "llama", "H100", 2, 1, false, "", "roofline", 8192)
	want := ModelHardwareConfig{
		ModelConfig:          mc,
		HWConfig:             hw,
		Model:                "llama",
		GPU:                  "H100",
		TP:                   2,
		DP:                   1,
		EnableExpertParallel: false,
		Backend:              "roofline",
		MaxModelLen:          8192,
	}
	assert.Equal(t, want, got)
}

// TestModelHardwareConfig_ParallelismHelpers verifies the DP/EP group-size
// helpers against vLLM's flattened-MoE-group semantics (#1417 / design §3).
//
// Laws asserted (behavioral, refactor-survivable):
//   - EffectiveDP clamps to >= 1.
//   - Dense EffectiveMoEGroupSize == TP regardless of DP/EP (dense never flattens).
//   - MoE EffectiveMoEGroupSize == TP·DP (the flattened group).
//   - EffectiveEP is in {1, EffectiveMoEGroupSize} and equals the group size
//     IFF expert parallelism is enabled on an MoE model.
func TestModelHardwareConfig_ParallelismHelpers(t *testing.T) {
	dense := ModelConfig{NumLayers: 32}                   // NumLocalExperts == 0 → dense
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8} // MoE

	tests := []struct {
		name         string
		mc           ModelConfig
		tp, dp       int
		ep           bool
		wantDP       int
		wantMoEGroup int
		wantEP       int
	}{
		// Degenerate / single-GPU.
		{"dense_tp1_dp1", dense, 1, 1, false, 1, 1, 1},
		{"moe_tp1_dp1_ep_off", moe, 1, 1, false, 1, 1, 1},
		{"moe_tp1_dp1_ep_on", moe, 1, 1, true, 1, 1, 1},

		// Dense never flattens: group stays TP, EP stays 1 even if requested.
		{"dense_tp2_dp1", dense, 2, 1, false, 1, 2, 1},
		{"dense_tp4_ep_on_ignored", dense, 4, 1, true, 1, 4, 1},

		// MoE EP-off: group flattens to TP·DP; EP predicate is 1.
		{"moe_tp2_dp1_ep_off", moe, 2, 1, false, 1, 2, 1},
		{"moe_tp2_dp2_ep_off", moe, 2, 2, false, 2, 4, 1},

		// MoE EP-on: group flattens to TP·DP; EP equals the group size.
		{"moe_tp2_dp1_ep_on", moe, 2, 1, true, 1, 2, 2},
		{"moe_tp2_dp2_ep_on", moe, 2, 2, true, 2, 4, 4},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := NewModelHardwareConfig(tc.mc, HardwareCalib{}, "m", "H100", tc.tp, tc.dp, tc.ep, "", "trained-physics", 0)
			assert.Equal(t, tc.wantDP, c.EffectiveDP(), "EffectiveDP")
			assert.Equal(t, tc.wantMoEGroup, c.EffectiveMoEGroupSize(), "EffectiveMoEGroupSize")
			assert.Equal(t, tc.wantEP, c.EffectiveEP(), "EffectiveEP")

			// Law: EP is either disabled (1) or exactly the flattened group.
			if c.EffectiveEP() != 1 {
				assert.Equal(t, c.EffectiveMoEGroupSize(), c.EffectiveEP(),
					"when EP is active it must equal the flattened MoE group size")
			}
		})
	}
}

// TestEffectiveDP_ClampsUnsetDP verifies that a zero/unset DP field (e.g. a
// zero-valued struct built outside the constructor) is treated as a single rank.
// The constructor rejects DP < 1, so this law is exercised via a direct literal.
func TestEffectiveDP_ClampsUnsetDP(t *testing.T) {
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8}
	c := ModelHardwareConfig{ModelConfig: moe, TP: 2, DP: 0} // DP unset
	assert.Equal(t, 1, c.EffectiveDP(), "unset DP must clamp to 1")
	assert.Equal(t, 2, c.EffectiveMoEGroupSize(), "TP·EffectiveDP = 2·1")
}

// TestNewModelHardwareConfig_DPValidation verifies the construction-time panics
// for invalid DP configurations (library boundary → panic).
func TestNewModelHardwareConfig_DPValidation(t *testing.T) {
	dense := ModelConfig{NumLayers: 32}
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8}

	tests := []struct {
		name         string
		mc           ModelConfig
		dp           int
		wantContains string
	}{
		{"dp_zero", moe, 0, "DP must be >= 1"},
		{"dp_negative", moe, -1, "DP must be >= 1"},
		{"dense_dp2", dense, 2, "only supported for MoE"},
		{"dense_dp8", dense, 8, "only supported for MoE"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected panic")
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, tc.wantContains) {
					t.Errorf("panic message %q should contain %q", msg, tc.wantContains)
				}
				if !strings.Contains(msg, "NewModelHardwareConfig") {
					t.Errorf("panic message %q should contain constructor name", msg)
				}
			}()
			NewModelHardwareConfig(tc.mc, HardwareCalib{}, "m", "H100", 2, tc.dp, false, "", "trained-physics", 0)
		})
	}
}

// TestNewModelHardwareConfig_MoE_DPAllowed verifies that DP > 1 is permitted for
// MoE models with either EP setting (no panic).
func TestNewModelHardwareConfig_MoE_DPAllowed(t *testing.T) {
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8}
	for _, ep := range []bool{false, true} {
		c := NewModelHardwareConfig(moe, HardwareCalib{}, "m", "H100", 2, 4, ep, "", "trained-physics", 0)
		assert.Equal(t, 4, c.DP)
		assert.Equal(t, ep, c.EnableExpertParallel)
		assert.Equal(t, 8, c.EffectiveMoEGroupSize()) // TP·DP = 2·4
	}
}

// TestEffectiveMoEGroupSize_EPModeIndependent locks in the design's load-bearing
// law (design §5 truth table): the flattened MoE group is TP·DP and is
// IDENTICAL whether expert parallelism is on or off — EP only relabels how that
// group is partitioned, never its size. A future latency-model change is most
// likely to break exactly this equality, so it is asserted directly rather than
// inferred from two independent expected numbers.
func TestEffectiveMoEGroupSize_EPModeIndependent(t *testing.T) {
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8}
	for _, tc := range []struct{ tp, dp int }{{2, 2}, {4, 2}, {1, 4}, {2, 1}} {
		off := NewModelHardwareConfig(moe, HardwareCalib{}, "m", "H100", tc.tp, tc.dp, false, "", "trained-physics", 0)
		on := NewModelHardwareConfig(moe, HardwareCalib{}, "m", "H100", tc.tp, tc.dp, true, "", "trained-physics", 0)
		assert.Equalf(t, off.EffectiveMoEGroupSize(), on.EffectiveMoEGroupSize(),
			"flattened MoE group must be EP-mode-independent at TP=%d,DP=%d", tc.tp, tc.dp)
		// And when EP is on, EP equals that same group.
		assert.Equal(t, on.EffectiveMoEGroupSize(), on.EffectiveEP(),
			"EP-on group must equal the flattened MoE group")
	}
}

// TestIsMoE_Boundary pins the canonical MoE-detection boundary (observable behavior,
// not the const value): 0 and 1 experts are dense; 2+ is MoE. This is the keystone
// guarding the intentional >= MoEMinExperts (not vLLM's > 0) threshold — see
// MoEMinExperts. A refactor that preserves the boundary keeps this green.
func TestIsMoE_Boundary(t *testing.T) {
	for _, tc := range []struct {
		experts int
		want    bool
	}{
		{0, false}, // dense (no expert fields)
		{1, false}, // single-expert is dense-equivalent in BLIS
		{2, true},  // smallest MoE
		{8, true},  // typical MoE (Mixtral)
	} {
		got := ModelConfig{NumLocalExperts: tc.experts}.IsMoE()
		assert.Equalf(t, tc.want, got, "IsMoE() for NumLocalExperts=%d", tc.experts)
	}
}

func TestNewPolicyConfig_FieldEquivalence(t *testing.T) {
	got := NewPolicyConfig("priority-fcfs", "")
	want := PolicyConfig{Scheduler: "priority-fcfs", PreemptionPolicy: ""}
	assert.Equal(t, want, got)
}

func TestNewPolicyConfig_DefaultPreemptionPolicy(t *testing.T) {
	cfg := NewPolicyConfig("fcfs", "")
	if cfg.PreemptionPolicy != "" {
		t.Errorf("default PreemptionPolicy: got %q, want empty", cfg.PreemptionPolicy)
	}
}

func TestNewWorkloadConfig_FieldEquivalence(t *testing.T) {
	got := NewWorkloadConfig()
	want := WorkloadConfig{}
	assert.Equal(t, want, got)
}

func TestNewKVCacheConfig_PanicsOnInvalid(t *testing.T) {
	tests := []struct {
		name            string
		totalKVBlocks   int64
		blockSizeTokens int64
		kvCPUBlocks     int64
		threshold       float64
		bandwidth       float64
		baseLatency     int64
		wantContains    string
	}{
		{"zero_total_kv_blocks", 0, 16, 0, 0, 0, 0, "TotalKVBlocks"},
		{"negative_total_kv_blocks", -1, 16, 0, 0, 0, 0, "TotalKVBlocks"},
		{"zero_block_size", 100, 0, 0, 0, 0, 0, "BlockSizeTokens"},
		{"negative_block_size", 100, -1, 0, 0, 0, 0, "BlockSizeTokens"},
		{"negative_cpu_blocks", 100, 16, -1, 0, 0, 0, "KVCPUBlocks"},
		{"tiered_bandwidth_zero", 100, 16, 10, 0.5, 0, 0, "KVTransferBandwidth"},
		{"tiered_bandwidth_negative", 100, 16, 10, 0.5, -1.0, 0, "KVTransferBandwidth"},
		{"tiered_bandwidth_nan", 100, 16, 10, 0.5, math.NaN(), 0, "KVTransferBandwidth"},
		{"tiered_bandwidth_pos_inf", 100, 16, 10, 0.5, math.Inf(1), 0, "KVTransferBandwidth"},
		{"tiered_bandwidth_neg_inf", 100, 16, 10, 0.5, math.Inf(-1), 0, "KVTransferBandwidth"},
		{"tiered_base_latency_negative", 100, 16, 10, 0.5, 100.0, -1, "KVTransferBaseLatency"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected panic")
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, tc.wantContains) {
					t.Errorf("panic message %q should contain %q", msg, tc.wantContains)
				}
				if !strings.Contains(msg, "NewKVCacheConfig") {
					t.Errorf("panic message %q should contain constructor name", msg)
				}
			}()
			NewKVCacheConfig(tc.totalKVBlocks, tc.blockSizeTokens, tc.kvCPUBlocks,
				tc.threshold, tc.bandwidth, tc.baseLatency)
		})
	}
}

func TestNewKVCacheConfig_SingleTier_SkipsTieredValidation(t *testing.T) {
	// BC-4: Single-tier mode (KVCPUBlocks=0) accepts any threshold/bandwidth/latency
	// without panicking. These fields are meaningless in single-tier mode.
	cfg := NewKVCacheConfig(100, 16, 0, -999.0, -999.0, -999)
	if cfg.TotalKVBlocks != 100 {
		t.Errorf("TotalKVBlocks = %d, want 100", cfg.TotalKVBlocks)
	}
	if cfg.KVOffloadThreshold != -999.0 {
		t.Errorf("KVOffloadThreshold = %f, want -999.0 (passed through)", cfg.KVOffloadThreshold)
	}
}

func TestNewKVCacheConfig_ValidTiered_ReturnsConfig(t *testing.T) {
	// BC-5: Valid tiered-mode parameters accepted
	cfg := NewKVCacheConfig(100, 16, 50, 0.9, 100.0, 500)
	if cfg.KVCPUBlocks != 50 {
		t.Errorf("KVCPUBlocks = %d, want 50", cfg.KVCPUBlocks)
	}
	if cfg.KVOffloadThreshold != 0.9 {
		t.Errorf("KVOffloadThreshold = %f, want 0.9", cfg.KVOffloadThreshold)
	}
}

func TestEffectiveWeightBytesPerParam_WhenSet_ReturnsWeightValue(t *testing.T) {
	// BC-4: GIVEN WeightBytesPerParam > 0, THEN returns WeightBytesPerParam
	mc := ModelConfig{BytesPerParam: 2.0, WeightBytesPerParam: 0.5}
	got := mc.EffectiveWeightBytesPerParam()
	if got != 0.5 {
		t.Errorf("expected 0.5 when WeightBytesPerParam set, got %v", got)
	}
}

func TestEffectiveWeightBytesPerParam_WhenZero_ReturnsBytesPerParam(t *testing.T) {
	// BC-5: GIVEN WeightBytesPerParam == 0 (sentinel), THEN returns BytesPerParam
	mc := ModelConfig{BytesPerParam: 2.0, WeightBytesPerParam: 0}
	got := mc.EffectiveWeightBytesPerParam()
	if got != 2.0 {
		t.Errorf("expected 2.0 (fallback to BytesPerParam), got %v", got)
	}
}

func TestEffectiveWeightBytesPerParam_BothZero_ReturnsZero(t *testing.T) {
	// Edge case: both zero → 0 (no panic, downstream validation catches it)
	mc := ModelConfig{BytesPerParam: 0, WeightBytesPerParam: 0}
	got := mc.EffectiveWeightBytesPerParam()
	if got != 0 {
		t.Errorf("expected 0 when both zero, got %v", got)
	}
}

func TestEffectiveKVBytesPerParam_WhenSet_ReturnsKVValue(t *testing.T) {
	// #1565: GIVEN KVBytesPerParam > 0 (e.g. --kv-cache-dtype fp8 → 1.0 under bf16
	// compute), THEN EffectiveKVBytesPerParam returns the explicit KV precision,
	// decoupled from the compute/activation dtype (BytesPerParam).
	mc := ModelConfig{BytesPerParam: 2.0, KVBytesPerParam: 1.0}
	got := mc.EffectiveKVBytesPerParam()
	if got != 1.0 {
		t.Errorf("expected 1.0 when KVBytesPerParam set, got %v", got)
	}
}

func TestEffectiveKVBytesPerParam_WhenZero_ReturnsBytesPerParam(t *testing.T) {
	// #1565: GIVEN KVBytesPerParam == 0 (the "auto" sentinel / flag absent), THEN
	// EffectiveKVBytesPerParam falls back to the compute dtype BytesPerParam — so the
	// KV footprint is byte-identical to a build without the flag (INV-6).
	mc := ModelConfig{BytesPerParam: 2.0, KVBytesPerParam: 0}
	got := mc.EffectiveKVBytesPerParam()
	if got != 2.0 {
		t.Errorf("expected 2.0 (fallback to BytesPerParam), got %v", got)
	}
}

func TestEffectiveKVBytesPerParam_IndependentOfWeightPrecision(t *testing.T) {
	// #1565: KV storage precision and weight quantization are independent vLLM engine
	// args. A W4A16 model (WeightBytesPerParam=0.5) with fp8 KV (KVBytesPerParam=1.0)
	// under bf16 compute reports 1.0 for KV and 0.5 for weights — neither leaks.
	mc := ModelConfig{BytesPerParam: 2.0, WeightBytesPerParam: 0.5, KVBytesPerParam: 1.0}
	if got := mc.EffectiveKVBytesPerParam(); got != 1.0 {
		t.Errorf("expected KV precision 1.0, got %v", got)
	}
	if got := mc.EffectiveWeightBytesPerParam(); got != 0.5 {
		t.Errorf("expected weight precision 0.5, got %v", got)
	}
}

func TestEffectiveHeadDim_WhenSet_ReturnsHeadDim(t *testing.T) {
	// F1 (BC-2): GIVEN explicit HeadDim > 0 (e.g. GLM-5.2: head_dim=192 while
	// hidden/heads=6144/64=96), THEN EffectiveHeadDim returns the explicit value.
	mc := ModelConfig{HiddenDim: 6144, NumHeads: 64, HeadDim: 192}
	if got := mc.EffectiveHeadDim(); got != 192 {
		t.Errorf("EffectiveHeadDim() = %d, want 192 (explicit head_dim)", got)
	}
}

func TestEffectiveHeadDim_WhenZero_ReturnsHiddenOverHeads(t *testing.T) {
	// F1 (BC-2, INV-6): GIVEN HeadDim == 0 (sentinel / key absent), THEN
	// EffectiveHeadDim falls back to hidden/heads — byte-identical to pre-change.
	mc := ModelConfig{HiddenDim: 4096, NumHeads: 32, HeadDim: 0}
	if got := mc.EffectiveHeadDim(); got != 128 {
		t.Errorf("EffectiveHeadDim() = %d, want 128 (hidden/heads fallback)", got)
	}
}

func TestEffectiveHeadDim_ZeroHeads_NoPanic(t *testing.T) {
	// Defensive: NumHeads == 0 on the implicit path must not divide by zero.
	mc := ModelConfig{HiddenDim: 4096, NumHeads: 0, HeadDim: 0}
	if got := mc.EffectiveHeadDim(); got != 0 {
		t.Errorf("EffectiveHeadDim() = %d, want 0 when NumHeads==0 and HeadDim==0", got)
	}
}

func TestIsMLA(t *testing.T) {
	// F2: IsMLA is true iff a positive compressed-KV latent rank is present.
	if (ModelConfig{KVLoraRank: 512}).IsMLA() != true {
		t.Error("expected IsMLA()=true when KVLoraRank>0")
	}
	if (ModelConfig{KVLoraRank: 0}).IsMLA() != false {
		t.Error("expected IsMLA()=false when KVLoraRank==0 (standard MHA/GQA)")
	}
}

func TestEffectiveKVBearingLayers_WhenSet_ReturnsField(t *testing.T) {
	// #1635 (BC-1): GIVEN a hybrid-attention model whose KV-bearing (full-attention)
	// layer count is fewer than its total layers (Kimi-K3: 24 MLA layers of 93; the
	// other 69 are linear-attention KDA layers with O(1)-in-sequence state and no
	// growing KV), THEN EffectiveKVBearingLayers returns the KV-bearing count, not
	// NumLayers.
	mc := ModelConfig{NumLayers: 93, KVBearingLayers: 24}
	if got := mc.EffectiveKVBearingLayers(); got != 24 {
		t.Errorf("EffectiveKVBearingLayers() = %d, want 24 (full-attention layers)", got)
	}
}

func TestEffectiveKVBearingLayers_WhenZero_ReturnsNumLayers(t *testing.T) {
	// #1635 (BC-2, INV-6): GIVEN KVBearingLayers == 0 (sentinel / not a hybrid
	// model), THEN EffectiveKVBearingLayers falls back to NumLayers — byte-identical
	// to pre-change for every standard-MHA and non-hybrid MLA model.
	mc := ModelConfig{NumLayers: 32, KVBearingLayers: 0}
	if got := mc.EffectiveKVBearingLayers(); got != 32 {
		t.Errorf("EffectiveKVBearingLayers() = %d, want 32 (fallback to NumLayers)", got)
	}
}

func TestEffectiveKVBearingLayers_ClampedToNumLayers(t *testing.T) {
	// #1635 (F2, defensive): a malformed config with KVBearingLayers > NumLayers must
	// clamp to NumLayers, never over-count KV worse than the all-layers default.
	mc := ModelConfig{NumLayers: 32, KVBearingLayers: 40}
	if got := mc.EffectiveKVBearingLayers(); got != 32 {
		t.Errorf("EffectiveKVBearingLayers() = %d, want 32 (clamped to NumLayers)", got)
	}
}

func TestNewBatchConfig_PanicsOnInvalid(t *testing.T) {
	tests := []struct {
		name          string
		maxRunning    int64
		maxTokens     int64
		prefillThresh int64
		wantContains  string
	}{
		{"zero_max_running", 0, 2048, 0, "MaxNumSeqs"},
		{"negative_max_running", -1, 2048, 0, "MaxNumSeqs"},
		{"zero_max_tokens", 256, 0, 0, "MaxNumBatchedTokens"},
		{"negative_max_tokens", 256, -1, 0, "MaxNumBatchedTokens"},
		{"negative_prefill", 256, 2048, -1, "LongPrefillTokenThreshold"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected panic")
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, tc.wantContains) {
					t.Errorf("panic message %q should contain %q", msg, tc.wantContains)
				}
			}()
			NewBatchConfig(tc.maxRunning, tc.maxTokens, tc.prefillThresh)
		})
	}
}

func TestHardwareCalib_MFUValidation(t *testing.T) {
	// BC-15: MFU values must be in valid ranges for capacity planning
	tests := []struct {
		name         string
		hw           HardwareCalib
		wantValid    bool
		wantContains string
	}{
		{
			name: "valid_h100_mfu",
			hw: HardwareCalib{
				TFlopsPeak: 989.5,
				TFlopsFP8:  1979.0,
				BwPeakTBs:  3.35,
				MfuPrefill: 0.45,
				MfuDecode:  0.30,
				MemoryGiB:  80.0,
			},
			wantValid: true,
		},
		{
			name: "valid_a100_mfu",
			hw: HardwareCalib{
				TFlopsPeak: 312,
				BwPeakTBs:  2.039,
				MfuPrefill: 0.38,
				MfuDecode:  0.18,
				MemoryGiB:  80.0,
			},
			wantValid: true,
		},
		{
			name: "valid_l40s_mfu",
			hw: HardwareCalib{
				TFlopsPeak: 362.05,
				BwPeakTBs:  0.864,
				MfuPrefill: 0.32,
				MfuDecode:  0.08,
				MemoryGiB:  48.0,
			},
			wantValid: true,
		},
		{
			name: "mfu_prefill_exceeds_one",
			hw: HardwareCalib{
				TFlopsPeak: 989.5,
				BwPeakTBs:  3.35,
				MfuPrefill: 1.1,
				MfuDecode:  0.30,
				MemoryGiB:  80.0,
			},
			wantValid:    false,
			wantContains: "MfuPrefill",
		},
		{
			name: "mfu_decode_exceeds_one",
			hw: HardwareCalib{
				TFlopsPeak: 989.5,
				BwPeakTBs:  3.35,
				MfuPrefill: 0.45,
				MfuDecode:  1.5,
				MemoryGiB:  80.0,
			},
			wantValid:    false,
			wantContains: "MfuDecode",
		},
		{
			name: "mfu_prefill_negative",
			hw: HardwareCalib{
				TFlopsPeak: 989.5,
				BwPeakTBs:  3.35,
				MfuPrefill: -0.1,
				MfuDecode:  0.30,
				MemoryGiB:  80.0,
			},
			wantValid:    false,
			wantContains: "MfuPrefill",
		},
		{
			name: "mfu_decode_negative",
			hw: HardwareCalib{
				TFlopsPeak: 989.5,
				BwPeakTBs:  3.35,
				MfuPrefill: 0.45,
				MfuDecode:  -0.1,
				MemoryGiB:  80.0,
			},
			wantValid:    false,
			wantContains: "MfuDecode",
		},
		{
			name: "mfu_decode_exceeds_prefill",
			hw: HardwareCalib{
				TFlopsPeak: 989.5,
				BwPeakTBs:  3.35,
				MfuPrefill: 0.30,
				MfuDecode:  0.45,
				MemoryGiB:  80.0,
			},
			wantValid:    false,
			wantContains: "MfuDecode",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateHardwareCalib(tc.hw)
			if tc.wantValid {
				if err != nil {
					t.Errorf("expected valid, got error: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("expected error containing %q, got nil", tc.wantContains)
				} else if !strings.Contains(err.Error(), tc.wantContains) {
					t.Errorf("error %q should contain %q", err.Error(), tc.wantContains)
				}
			}
		})
	}
}

// loraIntPtr is a local helper for building *int LoRAConfig fields in tests.
func loraIntPtr(v int) *int { return &v }

// TestLoRAConfig_Validate exercises the LoRAConfig validation contract
// (contracts/config-schema.md). Behavioral GIVEN/WHEN/THEN scenarios:
//   - adapters present + adapter_capacity == 0  => error (adapters forbidden)
//   - any adapter rank <= 0                      => error (R3)
//   - load_bandwidth_bytes_us <= 0               => error (R11 divisor guard)
//   - load_base_latency_us < 0                   => error (R3)
//   - footprint_bytes_per_rank <= 0              => error (R3)
//   - step_overhead_tiers k7 <= 0 / k6 < 0       => error (R3/R11)
//   - duplicate adapter id                       => error
//   - empty config                               => valid / inert (INV-6)
func TestLoRAConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     LoRAConfig
		wantErr bool
	}{
		{
			name:    "empty config is valid and inert",
			cfg:     LoRAConfig{},
			wantErr: false,
		},
		{
			name: "valid populated config",
			cfg: LoRAConfig{
				AdapterCapacity:       loraIntPtr(8),
				LoadBaseLatencyUs:     float64Ptr(1500.0),
				LoadBandwidthBytesUs:  float64Ptr(2.0e6),
				FootprintBytesPerRank: float64Ptr(2.0e6),
				Adapters: []AdapterSpec{
					{ID: "adapter_0", Rank: 8},
					{ID: "adapter_1", Rank: 16},
				},
			},
			wantErr: false,
		},
		{
			name: "adapters and positive capacity but no cost coefficients",
			cfg: LoRAConfig{
				AdapterCapacity: loraIntPtr(4),
				Adapters:        []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true, // gate consumes cost coefficients; CLI must catch the gap here (#1466)
		},
		{
			name: "adapters present but zero capacity",
			cfg: LoRAConfig{
				AdapterCapacity: loraIntPtr(0),
				Adapters:        []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true,
		},
		{
			name: "negative capacity",
			cfg: LoRAConfig{
				AdapterCapacity: loraIntPtr(-1),
				Adapters:        []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true,
		},
		{
			name: "adapter rank zero",
			cfg: LoRAConfig{
				AdapterCapacity: loraIntPtr(4),
				Adapters:        []AdapterSpec{{ID: "adapter_0", Rank: 0}},
			},
			wantErr: true,
		},
		{
			name: "adapter rank negative",
			cfg: LoRAConfig{
				AdapterCapacity: loraIntPtr(4),
				Adapters:        []AdapterSpec{{ID: "adapter_0", Rank: -8}},
			},
			wantErr: true,
		},
		{
			name: "load bandwidth zero",
			cfg: LoRAConfig{
				AdapterCapacity:      loraIntPtr(4),
				LoadBandwidthBytesUs: float64Ptr(0),
				Adapters:             []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true,
		},
		{
			name: "load bandwidth negative",
			cfg: LoRAConfig{
				AdapterCapacity:      loraIntPtr(4),
				LoadBandwidthBytesUs: float64Ptr(-1),
				Adapters:             []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true,
		},
		{
			name: "load base latency negative",
			cfg: LoRAConfig{
				AdapterCapacity:   loraIntPtr(4),
				LoadBaseLatencyUs: float64Ptr(-1),
				Adapters:          []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true,
		},
		{
			name: "footprint per rank zero",
			cfg: LoRAConfig{
				AdapterCapacity:       loraIntPtr(4),
				FootprintBytesPerRank: float64Ptr(0),
				Adapters:              []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true,
		},
		{
			name: "step overhead tier k7 zero (divisor guard)",
			cfg: LoRAConfig{
				AdapterCapacity:   loraIntPtr(4),
				StepOverheadTiers: map[int]StepOverheadTier{8: {K6: float64Ptr(0.02), K7: float64Ptr(0)}},
				Adapters:          []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true,
		},
		{
			name: "step overhead tier k6 negative",
			cfg: LoRAConfig{
				AdapterCapacity:   loraIntPtr(4),
				StepOverheadTiers: map[int]StepOverheadTier{8: {K6: float64Ptr(-0.1), K7: float64Ptr(1.0)}},
				Adapters:          []AdapterSpec{{ID: "adapter_0", Rank: 8}},
			},
			wantErr: true,
		},
		{
			name: "duplicate adapter id",
			cfg: LoRAConfig{
				AdapterCapacity: loraIntPtr(4),
				Adapters: []AdapterSpec{
					{ID: "adapter_0", Rank: 8},
					{ID: "adapter_0", Rank: 16},
				},
			},
			wantErr: true,
		},
		{
			name: "empty adapter id",
			cfg: LoRAConfig{
				AdapterCapacity: loraIntPtr(4),
				Adapters:        []AdapterSpec{{ID: "", Rank: 8}},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err, "expected validation error")
			} else {
				assert.NoError(t, err, "expected config to be valid")
			}
		})
	}
}

// validateHardwareCalib checks MFU value constraints for capacity planning.
// Returns error if values are outside physically plausible bounds.
func validateHardwareCalib(hw HardwareCalib) error {
	if hw.MfuPrefill < 0 || hw.MfuPrefill > 1 {
		return fmt.Errorf("MfuPrefill must be in [0,1], got %v", hw.MfuPrefill)
	}
	if hw.MfuDecode < 0 || hw.MfuDecode > 1 {
		return fmt.Errorf("MfuDecode must be in [0,1], got %v", hw.MfuDecode)
	}
	if hw.MfuDecode > hw.MfuPrefill {
		return fmt.Errorf("MfuDecode (%v) should not exceed MfuPrefill (%v) - decode is typically more memory-bound", hw.MfuDecode, hw.MfuPrefill)
	}
	return nil
}

func TestNewSpeculativeConfig_Validate(t *testing.T) {
	tests := []struct {
		name       string
		k          int
		acceptance float64
		method     string
		wantErr    bool
	}{
		{name: "inert zero value", k: 0, acceptance: 0, method: "", wantErr: false},
		{name: "valid mtp", k: 5, acceptance: 0.8, method: "mtp", wantErr: false},
		{name: "valid no method", k: 3, acceptance: 0.5, method: "", wantErr: false},
		{name: "valid alpha zero with k", k: 5, acceptance: 0.0, method: "eagle", wantErr: false},
		{name: "valid alpha one", k: 4, acceptance: 1.0, method: "", wantErr: false},
		{name: "valid at ceiling", k: MaxSpeculativeTokens, acceptance: 0.5, method: "", wantErr: false},
		{name: "negative k", k: -1, acceptance: 0, method: "", wantErr: true},
		{name: "k above ceiling", k: MaxSpeculativeTokens + 1, acceptance: 0.5, method: "", wantErr: true},
		{name: "alpha above one", k: 3, acceptance: 1.5, method: "", wantErr: true},
		{name: "alpha negative", k: 3, acceptance: -0.1, method: "", wantErr: true},
		{name: "alpha NaN", k: 3, acceptance: math.NaN(), method: "", wantErr: true},
		{name: "alpha Inf", k: 3, acceptance: math.Inf(1), method: "", wantErr: true},
		{name: "dangling acceptance k zero", k: 0, acceptance: 0.5, method: "", wantErr: true},
		{name: "dangling method k zero", k: 0, acceptance: 0, method: "mtp", wantErr: true},
		{name: "unknown method", k: 5, acceptance: 0.5, method: "bogus", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSpeculativeConfig(tt.k, tt.acceptance, tt.method)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSpeculativeConfig_Helpers(t *testing.T) {
	off := SpeculativeConfig{}
	assert.False(t, off.IsEnabled())
	// Off ⇒ one token per step, verify width 1 (byte-identity foundation).
	assert.Equal(t, 1.0, off.EffectiveTokensPerStep())
	assert.Equal(t, 1, off.VerifyWidth())

	on := SpeculativeConfig{K: 5, Acceptance: 0.8}
	assert.True(t, on.IsEnabled())
	// 1 + 0.8*5 = 5.0 mean accepted tokens/step.
	assert.InDelta(t, 5.0, on.EffectiveTokensPerStep(), 1e-9)
	// K+1 = 6 verified positions per forward pass.
	assert.Equal(t, 6, on.VerifyWidth())

	// α=0 with K>0: no throughput gain (g=1) but verify width still k+1 (cost applies).
	noGain := SpeculativeConfig{K: 4, Acceptance: 0.0}
	assert.InDelta(t, 1.0, noGain.EffectiveTokensPerStep(), 1e-9)
	assert.Equal(t, 5, noGain.VerifyWidth())
}

// TestEffectiveEPSize_LogicalVsPerReplica is the #1656 BC-5 law: the EP group size
// must be derived from the LOGICAL (user-requested) topology, not from a per-replica
// config whose DP has been collapsed to 1 by DP-as-placement (#1531).
//
// The two values are deliberately different, and the difference is the whole point:
// a consumer that sizes routed-expert weights per EP rank and reads the group size off
// a per-replica config would silently get "no sharding" for exactly the TP×DP topology
// that needs it. This test pins both values so that collapse can never be mistaken for
// the logical answer.
func TestEffectiveEPSize_LogicalVsPerReplica(t *testing.T) {
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8}

	// Logical topology as requested on the CLI: --tp 8 --dp 2 --enable-expert-parallel.
	logical := EffectiveEPSize(true, 8, 2, true)
	assert.Equal(t, 16, logical, "logical EP group is TP·DP = 8·2")

	// The same deployment after DP-as-placement expands it into 2 engine replicas,
	// each reconfigured with DP=1.
	perReplica := NewModelHardwareConfig(moe, HardwareCalib{}, "m", "H100", 8, 1, true, "", "trained-physics", 0)
	assert.Equal(t, 8, perReplica.EffectiveEP(), "a per-replica DP=1 config yields TP, not TP·DP")
	assert.NotEqual(t, logical, perReplica.EffectiveEP(),
		"BC-5: the per-replica EP group must not be mistaken for the logical group")
}

// TestEffectiveEPSize_MatchesAccessor is the R23 code-path-parity guard: the free
// function and the config-bound accessor are one formula, so they agree for every
// topology (the accessor is defined in terms of the function).
func TestEffectiveEPSize_MatchesAccessor(t *testing.T) {
	dense := ModelConfig{NumLayers: 32}
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8}

	tests := []struct {
		name   string
		mc     ModelConfig
		tp, dp int
		ep     bool
		want   int
	}{
		{"dense_ep_off", dense, 4, 1, false, 1},
		{"dense_ep_on_ignored", dense, 4, 1, true, 1},
		{"moe_ep_off", moe, 4, 1, false, 1},
		{"moe_ep_on_dp1", moe, 4, 1, true, 4},
		{"moe_ep_on_dp2", moe, 4, 2, true, 8},
		{"moe_ep_on_tp1_dp16", moe, 1, 16, true, 16},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := EffectiveEPSize(tc.mc.IsMoE(), tc.tp, tc.dp, tc.ep)
			assert.Equal(t, tc.want, got, "EffectiveEPSize")

			c := NewModelHardwareConfig(tc.mc, HardwareCalib{}, "m", "H100", tc.tp, tc.dp, tc.ep, "", "trained-physics", 0)
			assert.Equal(t, got, c.EffectiveEP(), "accessor must equal the free function")
		})
	}
}

// TestEffectiveEPSize_ClampsUnsetDP verifies the free function treats an unset (zero)
// DP as a single rank, matching EffectiveDP's clamp, so a zero-valued struct or a
// caller that forwards an unset degree cannot produce a zero divisor.
func TestEffectiveEPSize_ClampsUnsetDP(t *testing.T) {
	assert.Equal(t, 4, EffectiveEPSize(true, 4, 0, true), "dp=0 must clamp to 1 → TP")
	assert.Equal(t, 4, EffectiveEPSize(true, 4, -3, true), "negative dp must clamp to 1 → TP")
}

// TestEffectiveEPGroupDP_CarriesTheLogicalWidth is the #1548 companion to
// TestEffectiveEPSize_LogicalVsPerReplica: the per-replica collapse documented there is
// exactly what WithExpertParallelGroupDP repairs, and this pins how.
func TestEffectiveEPGroupDP_CarriesTheLogicalWidth(t *testing.T) {
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8}
	newCfg := func(tp, dp int, ep bool, opts ...ModelHardwareOption) ModelHardwareConfig {
		return NewModelHardwareConfig(moe, HardwareCalib{}, "m", "H100", tp, dp, ep, "", "trained-physics", 0, opts...)
	}

	// A DP-as-placement replica of the logical --tp 8 --dp 2 --enable-expert-parallel
	// deployment: its own DP is 1, but the option restores the 16-GPU group.
	replica := newCfg(8, 1, true, WithExpertParallelGroupDP(2))
	assert.Equal(t, 2, replica.EffectiveEPGroupDP())
	assert.Equal(t, 16, replica.EffectiveEP(), "the logical EP group is TP·EPGroupDP = 8·2")
	assert.Equal(t, EffectiveEPSize(true, 8, 2, true), replica.EffectiveEP(),
		"the repaired accessor must agree with the pure logical formula (R23)")

	// Omitting the option is the pre-#1548 behaviour, unchanged.
	assert.Equal(t, 8, newCfg(8, 1, true).EffectiveEP())

	// The option can only WIDEN: a stale or too-small width cannot shrink a group that
	// already has a real DP of its own.
	lumped := newCfg(8, 4, true, WithExpertParallelGroupDP(2))
	assert.Equal(t, 32, lumped.EffectiveEP(), "a smaller supplied width must not shrink TP·DP")
}

// TestEffectiveExpertShardGroupSize_SeparatesWeightsFromCompute is BC-3 at the config
// level: the routed-expert WEIGHT shard group widens under expert parallelism while the
// COMPUTE group does not. Conflating them is the specific defect this split exists to
// prevent (dividing compute by the EP group would under-charge it by DP).
func TestEffectiveExpertShardGroupSize_SeparatesWeightsFromCompute(t *testing.T) {
	dense := ModelConfig{NumLayers: 32}
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 8}

	tests := []struct {
		name        string
		mc          ModelConfig
		tp, dp      int
		ep          bool
		epGroupDP   int // 0 ⇒ option omitted
		wantCompute int
		wantWeights int
	}{
		// Every pre-#1548 shape: the two groups coincide, so nothing can move (INV-6).
		{"dense", dense, 8, 1, false, 0, 8, 8},
		{"dense ep flag ignored", dense, 8, 1, true, 0, 8, 8},
		{"moe ep off dp1", moe, 8, 1, false, 0, 8, 8},
		{"moe ep off dp2", moe, 8, 2, false, 0, 16, 16},
		{"moe ep on dp1", moe, 8, 1, true, 0, 8, 8},
		{"moe ep on dp2 lumped", moe, 8, 2, true, 0, 16, 16},
		{"moe ep on tp1 dp1 degenerate", moe, 1, 1, true, 0, 1, 1},
		// The #1548 shape: a per-replica config carrying the logical width. Compute stays
		// on the replica's own TP·DP; weights shard across the whole logical EP group.
		{"replica of logical tp8 dp2", moe, 8, 1, true, 2, 8, 16},
		{"replica of Wide-EP tp1 dp16", moe, 1, 1, true, 16, 1, 16},
		// EP off ⇒ the width is inert even when supplied.
		{"replica width without ep is inert", moe, 8, 1, false, 2, 8, 8},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var opts []ModelHardwareOption
			if tc.epGroupDP > 0 {
				opts = append(opts, WithExpertParallelGroupDP(tc.epGroupDP))
			}
			c := NewModelHardwareConfig(tc.mc, HardwareCalib{}, "m", "H100", tc.tp, tc.dp, tc.ep, "", "trained-physics", 0, opts...)
			assert.Equal(t, tc.wantCompute, c.EffectiveMoEGroupSize(), "compute (routed-expert FLOPs) group")
			assert.Equal(t, tc.wantWeights, c.EffectiveExpertShardGroupSize(), "weight (expert-ownership) group")
		})
	}
}

// TestEffectiveExpertShardGroupSize_ScalesWithPoolTP is BC-7: the option carries a DP
// WIDTH, not an absolute group size, so a per-pool TP override (cluster.ResolvePoolConfig
// rewrites TP on a struct copy) yields poolTP·width rather than contradicting the pool's
// own TP. An absolute group size stamped from the global TP would be wrong here.
func TestEffectiveExpertShardGroupSize_ScalesWithPoolTP(t *testing.T) {
	moe := ModelConfig{NumLayers: 32, NumLocalExperts: 64}
	global := NewModelHardwareConfig(moe, HardwareCalib{}, "m", "H100", 8, 1, true, "", "trained-physics", 0,
		WithExpertParallelGroupDP(2))
	assert.Equal(t, 16, global.EffectiveExpertShardGroupSize())

	pool := global // the struct copy ResolvePoolConfig makes
	pool.TP = 4    // ... then overrides TP, exactly as --prefill-tp does
	assert.Equal(t, 8, pool.EffectiveExpertShardGroupSize(),
		"the EP group must follow the pool's own TP (4·2), not stay pinned to the global TP")
}
