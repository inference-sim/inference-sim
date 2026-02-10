package cluster

import (
	"strings"
	"testing"
)

// TestHFModelConfig_IsMoE tests the IsMoE() method
func TestHFModelConfig_IsMoE(t *testing.T) {
	tests := []struct {
		name     string
		config   *HFModelConfig
		expected bool
	}{
		{
			name: "Dense model (no MoE)",
			config: &HFModelConfig{
				MoE: nil,
			},
			expected: false,
		},
		{
			name: "Dense model (MoE with 1 expert)",
			config: &HFModelConfig{
				MoE: &MoEConfig{
					NumLocalExperts: 1,
				},
			},
			expected: false,
		},
		{
			name: "MoE model (Mixtral-style)",
			config: &HFModelConfig{
				MoE: &MoEConfig{
					NumLocalExperts:  8,
					NumExpertsPerTok: 2,
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.config.IsMoE()
			if got != tt.expected {
				t.Errorf("IsMoE() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestHFModelConfig_HeadDim tests the HeadDim() computation
func TestHFModelConfig_HeadDim(t *testing.T) {
	tests := []struct {
		name              string
		hiddenSize        int
		numAttentionHeads int
		expected          int
	}{
		{
			name:              "Llama 8B",
			hiddenSize:        4096,
			numAttentionHeads: 32,
			expected:          128,
		},
		{
			name:              "Llama 70B",
			hiddenSize:        8192,
			numAttentionHeads: 64,
			expected:          128,
		},
		{
			name:              "Zero attention heads",
			hiddenSize:        4096,
			numAttentionHeads: 0,
			expected:          0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &HFModelConfig{
				HiddenSize:        tt.hiddenSize,
				NumAttentionHeads: tt.numAttentionHeads,
			}
			got := config.HeadDim()
			if got != tt.expected {
				t.Errorf("HeadDim() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestHFModelConfig_KVCacheBytesPerToken tests KV cache memory calculation
func TestHFModelConfig_KVCacheBytesPerToken(t *testing.T) {
	tests := []struct {
		name          string
		config        *HFModelConfig
		expectedBytes int64
	}{
		{
			name: "Llama 8B with FP16",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     2,
			},
			expectedBytes: 131072, // 2 * 32 * 8 * 128 * 2
		},
		{
			name: "Mixtral 8x7B with FP16",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     2,
				MoE: &MoEConfig{
					NumLocalExperts:  8,
					NumExpertsPerTok: 2,
				},
			},
			expectedBytes: 131072, // Same as dense - KV cache independent of MoE
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.config.KVCacheBytesPerToken()
			if got != tt.expectedBytes {
				t.Errorf("KVCacheBytesPerToken() = %v, want %v", got, tt.expectedBytes)
			}
		})
	}
}

// TestHFModelConfig_KVCacheIdenticalForDenseAndMoE verifies KV cache is same for dense and MoE
func TestHFModelConfig_KVCacheIdenticalForDenseAndMoE(t *testing.T) {
	denseConfig := &HFModelConfig{
		NumLayers:         32,
		HiddenSize:        4096,
		NumAttentionHeads: 32,
		NumKVHeads:        8,
		BytesPerParam:     2,
	}

	moeConfig := &HFModelConfig{
		NumLayers:         32,
		HiddenSize:        4096,
		NumAttentionHeads: 32,
		NumKVHeads:        8,
		BytesPerParam:     2,
		MoE: &MoEConfig{
			NumLocalExperts:  8,
			NumExpertsPerTok: 2,
		},
	}

	denseKV := denseConfig.KVCacheBytesPerToken()
	moeKV := moeConfig.KVCacheBytesPerToken()

	if denseKV != moeKV {
		t.Errorf("KV cache should be identical for dense and MoE with same attention config: dense=%d, moe=%d", denseKV, moeKV)
	}
}

// TestHFModelConfig_ActiveParametersLessThanTotal tests MoE active params
func TestHFModelConfig_ActiveParametersLessThanTotal(t *testing.T) {
	moeConfig := &HFModelConfig{
		ModelID:           "mixtral-test",
		NumLayers:         32,
		HiddenSize:        4096,
		IntermediateSize:  14336,
		NumAttentionHeads: 32,
		NumKVHeads:        8,
		VocabSize:         32000,
		BytesPerParam:     2,
		MoE: &MoEConfig{
			NumLocalExperts:        8,
			NumExpertsPerTok:       2,
			ExpertIntermediateSize: 14336,
		},
	}

	total := moeConfig.TotalParameters()
	active := moeConfig.ActiveParametersPerToken()

	if active >= total {
		t.Errorf("Active parameters (%d) should be less than total (%d) for MoE", active, total)
	}

	// For 8 experts with 2 active, we expect active to be roughly 25% of total (2/8)
	// (not exact due to shared components like attention)
	ratio := float64(active) / float64(total)
	if ratio < 0.2 || ratio > 0.5 {
		t.Errorf("Active/Total ratio (%.2f) outside expected range [0.2, 0.5]", ratio)
	}
}

// TestHFModelConfig_DenseActiveParametersEqualsTotal tests dense model active params
func TestHFModelConfig_DenseActiveParametersEqualsTotal(t *testing.T) {
	denseConfig := &HFModelConfig{
		ModelID:           "llama-test",
		NumLayers:         32,
		HiddenSize:        4096,
		IntermediateSize:  11008,
		NumAttentionHeads: 32,
		NumKVHeads:        8,
		VocabSize:         32000,
		BytesPerParam:     2,
	}

	total := denseConfig.TotalParameters()
	active := denseConfig.ActiveParametersPerToken()

	if active != total {
		t.Errorf("Active parameters (%d) should equal total (%d) for dense model", active, total)
	}
}

// TestHFModelConfig_Validate tests BC-1: model config validity invariants
func TestHFModelConfig_Validate(t *testing.T) {
	tests := []struct {
		name      string
		config    *HFModelConfig
		wantError bool
		errorMsg  string
	}{
		{
			name: "Valid dense model",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				IntermediateSize:  11008,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				VocabSize:         32000,
				BytesPerParam:     2,
			},
			wantError: false,
		},
		{
			name: "Valid MoE model",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				IntermediateSize:  14336,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				VocabSize:         32000,
				BytesPerParam:     2,
				MoE: &MoEConfig{
					NumLocalExperts:        8,
					NumExpertsPerTok:       2,
					ExpertIntermediateSize: 14336,
				},
			},
			wantError: false,
		},
		{
			name: "Invalid: NumLayers <= 0",
			config: &HFModelConfig{
				NumLayers:         0,
				HiddenSize:        4096,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     2,
			},
			wantError: true,
			errorMsg:  "NumLayers must be positive",
		},
		{
			name: "Invalid: HiddenSize <= 0",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        0,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     2,
			},
			wantError: true,
			errorMsg:  "HiddenSize must be positive",
		},
		{
			name: "Invalid: NumKVHeads > NumAttentionHeads",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				NumAttentionHeads: 8,
				NumKVHeads:        32,
				BytesPerParam:     2,
			},
			wantError: true,
			errorMsg:  "NumKVHeads (32) cannot exceed NumAttentionHeads (8)",
		},
		{
			name: "Invalid: HiddenSize not divisible by NumAttentionHeads",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4095,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     2,
			},
			wantError: true,
			errorMsg:  "HiddenSize (4095) must be divisible by NumAttentionHeads (32)",
		},
		{
			name: "Invalid: BytesPerParam not in {1,2,4}",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     3,
			},
			wantError: true,
			errorMsg:  "BytesPerParam must be 1, 2, or 4",
		},
		{
			name: "Valid: MoE with NumLocalExperts=1 treated as dense",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     2,
				MoE: &MoEConfig{
					NumLocalExperts:  1,
					NumExpertsPerTok: 1,
				},
			},
			wantError: false, // IsMoE() returns false, so no MoE validation
		},
		{
			name: "Invalid MoE: NumExpertsPerTok < 1",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     2,
				MoE: &MoEConfig{
					NumLocalExperts:  8,
					NumExpertsPerTok: 0,
				},
			},
			wantError: true,
			errorMsg:  "MoE NumExpertsPerTok must be >= 1",
		},
		{
			name: "Invalid MoE: NumExpertsPerTok > NumLocalExperts",
			config: &HFModelConfig{
				NumLayers:         32,
				HiddenSize:        4096,
				NumAttentionHeads: 32,
				NumKVHeads:        8,
				BytesPerParam:     2,
				MoE: &MoEConfig{
					NumLocalExperts:  8,
					NumExpertsPerTok: 10,
				},
			},
			wantError: true,
			errorMsg:  "MoE NumExpertsPerTok (10) cannot exceed NumLocalExperts (8)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantError {
				if err == nil {
					t.Errorf("Validate() expected error containing '%s', got nil", tt.errorMsg)
				} else if tt.errorMsg != "" && !contains(err.Error(), tt.errorMsg) {
					t.Errorf("Validate() error = '%v', want error containing '%s'", err, tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}
