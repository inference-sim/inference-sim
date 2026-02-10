package cluster

import "testing"

// TestDefaultVLLMEngineConfig tests that default config has sensible values
func TestDefaultVLLMEngineConfig(t *testing.T) {
	config := DefaultVLLMEngineConfig()

	if config.TensorParallelSize != 1 {
		t.Errorf("Expected TensorParallelSize = 1, got %d", config.TensorParallelSize)
	}
	if config.MaxNumSeqs != 256 {
		t.Errorf("Expected MaxNumSeqs = 256, got %d", config.MaxNumSeqs)
	}
	if config.BlockSize != 16 {
		t.Errorf("Expected BlockSize = 16, got %d", config.BlockSize)
	}
	if !config.EnableChunkedPrefill {
		t.Error("Expected EnableChunkedPrefill = true")
	}

	// Validate should pass
	if err := config.Validate(); err != nil {
		t.Errorf("Default config should be valid, got error: %v", err)
	}
}

// TestVLLMEngineConfig_TotalGPUs tests GPU count calculation
func TestVLLMEngineConfig_TotalGPUs(t *testing.T) {
	tests := []struct {
		name     string
		tp       int
		pp       int
		dp       int
		expected int
	}{
		{
			name:     "Single GPU",
			tp:       1,
			pp:       1,
			dp:       1,
			expected: 1,
		},
		{
			name:     "TP only",
			tp:       4,
			pp:       1,
			dp:       1,
			expected: 4,
		},
		{
			name:     "PP only",
			tp:       1,
			pp:       4,
			dp:       1,
			expected: 4,
		},
		{
			name:     "DP only",
			tp:       1,
			pp:       1,
			dp:       4,
			expected: 4,
		},
		{
			name:     "TP=2, PP=2, DP=4",
			tp:       2,
			pp:       2,
			dp:       4,
			expected: 16,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &VLLMEngineConfig{
				TensorParallelSize:   tt.tp,
				PipelineParallelSize: tt.pp,
				DataParallelSize:     tt.dp,
			}
			got := config.TotalGPUs()
			if got != tt.expected {
				t.Errorf("TotalGPUs() = %d, want %d", got, tt.expected)
			}
		})
	}
}

// TestVLLMEngineConfig_EffectiveExpertParallelism tests EP calculation for MoE
func TestVLLMEngineConfig_EffectiveExpertParallelism(t *testing.T) {
	tests := []struct {
		name     string
		tp       int
		dp       int
		moe      *VLLMMoEConfig
		expected int
	}{
		{
			name:     "Dense model (no MoE)",
			tp:       2,
			dp:       4,
			moe:      nil,
			expected: 0,
		},
		{
			name: "MoE model TP=2, DP=4",
			tp:   2,
			dp:   4,
			moe: &VLLMMoEConfig{
				EnableAllToAll: true,
			},
			expected: 8, // DP × TP
		},
		{
			name: "MoE model TP=1, DP=8",
			tp:   1,
			dp:   8,
			moe: &VLLMMoEConfig{
				EnableAllToAll: true,
			},
			expected: 8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &VLLMEngineConfig{
				TensorParallelSize: tt.tp,
				DataParallelSize:   tt.dp,
				MoEConfig:          tt.moe,
			}
			got := config.EffectiveExpertParallelism()
			if got != tt.expected {
				t.Errorf("EffectiveExpertParallelism() = %d, want %d", got, tt.expected)
			}
		})
	}
}

// TestVLLMEngineConfig_IsMoEDeployment tests MoE detection
func TestVLLMEngineConfig_IsMoEDeployment(t *testing.T) {
	tests := []struct {
		name     string
		moe      *VLLMMoEConfig
		expected bool
	}{
		{
			name:     "Dense model",
			moe:      nil,
			expected: false,
		},
		{
			name: "MoE model",
			moe: &VLLMMoEConfig{
				EnableAllToAll: true,
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &VLLMEngineConfig{
				MoEConfig: tt.moe,
			}
			got := config.IsMoEDeployment()
			if got != tt.expected {
				t.Errorf("IsMoEDeployment() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestVLLMEngineConfig_Validate tests BC-2: engine config validity invariants
func TestVLLMEngineConfig_Validate(t *testing.T) {
	tests := []struct {
		name      string
		config    *VLLMEngineConfig
		wantError bool
		errorMsg  string
	}{
		{
			name: "Valid config",
			config: &VLLMEngineConfig{
				TensorParallelSize:   2,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 0.90,
				TotalKVBlocks:        10000,
			},
			wantError: false,
		},
		{
			name: "Invalid: TensorParallelSize = 0",
			config: &VLLMEngineConfig{
				TensorParallelSize:   0,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 0.90,
			},
			wantError: true,
			errorMsg:  "TensorParallelSize must be >= 1",
		},
		{
			name: "Invalid: PipelineParallelSize = 0",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 0,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 0.90,
			},
			wantError: true,
			errorMsg:  "PipelineParallelSize must be >= 1",
		},
		{
			name: "Invalid: DataParallelSize = 0",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				DataParallelSize:     0,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 0.90,
			},
			wantError: true,
			errorMsg:  "DataParallelSize must be >= 1",
		},
		{
			name: "Invalid: MaxNumSeqs = 0",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           0,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 0.90,
			},
			wantError: true,
			errorMsg:  "MaxNumSeqs must be > 0",
		},
		{
			name: "Invalid: MaxNumBatchedTokens = 0",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  0,
				BlockSize:            16,
				GPUMemoryUtilization: 0.90,
			},
			wantError: true,
			errorMsg:  "MaxNumBatchedTokens must be > 0",
		},
		{
			name: "Invalid: BlockSize = 0",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            0,
				GPUMemoryUtilization: 0.90,
			},
			wantError: true,
			errorMsg:  "BlockSize must be > 0",
		},
		{
			name: "Invalid: GPUMemoryUtilization = 0",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 0.0,
			},
			wantError: true,
			errorMsg:  "GPUMemoryUtilization must be in (0, 1]",
		},
		{
			name: "Invalid: GPUMemoryUtilization > 1",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 1.5,
			},
			wantError: true,
			errorMsg:  "GPUMemoryUtilization must be in (0, 1]",
		},
		{
			name: "Invalid: TotalKVBlocks < 0",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 0.90,
				TotalKVBlocks:        -1,
			},
			wantError: true,
			errorMsg:  "TotalKVBlocks must be >= 0",
		},
		{
			name: "Valid: TotalKVBlocks = 0 (computed)",
			config: &VLLMEngineConfig{
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				DataParallelSize:     1,
				MaxNumSeqs:           256,
				MaxNumBatchedTokens:  8192,
				BlockSize:            16,
				GPUMemoryUtilization: 0.90,
				TotalKVBlocks:        0,
			},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantError {
				if err == nil {
					t.Errorf("Validate() expected error containing '%s', got nil", tt.errorMsg)
				} else if !contains(err.Error(), tt.errorMsg) {
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
