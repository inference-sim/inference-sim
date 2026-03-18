package sim

import (
	"fmt"
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
		MaxRunningReqs:            10,
		MaxScheduledTokens:        1000,
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
	got := NewModelHardwareConfig(mc, hw, "llama", "H100", 2, "roofline", 8192)
	want := ModelHardwareConfig{
		ModelConfig: mc,
		HWConfig:    hw,
		Model:       "llama",
		GPU:         "H100",
		TP:          2,
		Backend:     "roofline",
		MaxModelLen: 8192,
	}
	assert.Equal(t, want, got)
}

func TestNewPolicyConfig_FieldEquivalence(t *testing.T) {
	got := NewPolicyConfig("slo-based", "priority-fcfs")
	want := PolicyConfig{PriorityPolicy: "slo-based", Scheduler: "priority-fcfs"}
	assert.Equal(t, want, got)
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

func TestNewBatchConfig_PanicsOnInvalid(t *testing.T) {
	tests := []struct {
		name          string
		maxRunning    int64
		maxTokens     int64
		prefillThresh int64
		wantContains  string
	}{
		{"zero_max_running", 0, 2048, 0, "MaxRunningReqs"},
		{"negative_max_running", -1, 2048, 0, "MaxRunningReqs"},
		{"zero_max_tokens", 256, 0, 0, "MaxScheduledTokens"},
		{"negative_max_tokens", 256, -1, 0, "MaxScheduledTokens"},
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
