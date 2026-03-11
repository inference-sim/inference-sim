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
	got := NewBatchConfig(10, 1000, 200, 5.0, 7)
	want := BatchConfig{
		MaxRunningReqs:                10,
		MaxScheduledTokens:            1000,
		LongPrefillTokenThreshold:     200,
		PriorityPreemptionMargin:      5.0,
		MaxPriorityPreemptionsPerStep: 7,
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

func TestNewKVCacheConfig_ZeroValues_NoDefaults(t *testing.T) {
	// BC-4: Zero-value arguments must NOT inject non-zero defaults
	got := NewKVCacheConfig(0, 0, 0, 0, 0, 0)
	assert.Equal(t, KVCacheConfig{}, got)
}

func TestNewBatchConfig_PanicsOnInvalid(t *testing.T) {
	tests := []struct {
		name                string
		maxRunning          int64
		maxTokens           int64
		prefillThresh       int64
		preemptionMargin    float64
		maxPreemptionsPerStep int
		wantContains        string
	}{
		{"zero_max_running", 0, 2048, 0, 0, 0, "MaxRunningReqs"},
		{"negative_max_running", -1, 2048, 0, 0, 0, "MaxRunningReqs"},
		{"zero_max_tokens", 256, 0, 0, 0, 0, "MaxScheduledTokens"},
		{"negative_max_tokens", 256, -1, 0, 0, 0, "MaxScheduledTokens"},
		{"negative_prefill", 256, 2048, -1, 0, 0, "LongPrefillTokenThreshold"},
		{"negative_preemption_margin", 256, 2048, 0, -1.0, 0, "PriorityPreemptionMargin"},
		{"negative_max_preemptions_per_step", 256, 2048, 0, 0, -1, "MaxPriorityPreemptionsPerStep"},
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
			NewBatchConfig(tc.maxRunning, tc.maxTokens, tc.prefillThresh, tc.preemptionMargin, tc.maxPreemptionsPerStep)
		})
	}
}
