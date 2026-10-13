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
		PrefixCachingDisabled:     false,
	}
	assert.Equal(t, want, got)
}

func TestNewBatchConfig_WithPrefixCachingDisabled(t *testing.T) {
	got := NewBatchConfig(10, 1000, 200, WithPrefixCachingDisabled(true))
	if !got.PrefixCachingDisabled {
		t.Fatal("WithPrefixCachingDisabled(true) must reach BatchConfig")
	}
}

func TestNewModelHardwareConfig_FieldEquivalence(t *testing.T) {
	mc := ModelConfig{}
	got := NewModelHardwareConfig(mc, "llama", "H100", 2, 1, false, 8192)
	want := ModelHardwareConfig{
		ModelConfig:          mc,
		Model:                "llama",
		GPU:                  "H100",
		TP:                   2,
		DP:                   1,
		EnableExpertParallel: false,
		MaxModelLen:          8192,
	}
	assert.Equal(t, want, got)
}

// TestEffectiveDP_ClampsUnsetDP verifies that a zero/unset DP field (e.g. a
// zero-valued struct built outside the constructor) is treated as a single rank.
// The constructor rejects DP < 1, so this law is exercised via a direct literal.
func TestEffectiveDP_ClampsUnsetDP(t *testing.T) {
	moe := ModelConfig{NumLocalExperts: 8}
	c := ModelHardwareConfig{ModelConfig: moe, TP: 2, DP: 0} // DP unset
	assert.Equal(t, 1, c.EffectiveDP(), "unset DP must clamp to 1")
}

// TestNewModelHardwareConfig_DPValidation verifies the construction-time panics
// for invalid DP configurations (library boundary → panic).
func TestNewModelHardwareConfig_DPValidation(t *testing.T) {
	dense := ModelConfig{}
	moe := ModelConfig{NumLocalExperts: 8}

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
			NewModelHardwareConfig(tc.mc, "m", "H100", 2, tc.dp, false, 0)
		})
	}
}

// TestNewModelHardwareConfig_MoE_DPAllowed verifies that DP > 1 is permitted for
// MoE models with either EP setting (no panic).
func TestNewModelHardwareConfig_MoE_DPAllowed(t *testing.T) {
	moe := ModelConfig{NumLocalExperts: 8}
	for _, ep := range []bool{false, true} {
		c := NewModelHardwareConfig(moe, "m", "H100", 2, 4, ep, 0)
		assert.Equal(t, 4, c.DP)
		assert.Equal(t, ep, c.EnableExpertParallel)
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
