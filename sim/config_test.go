package sim

import (
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
	got := NewModelHardwareConfig(mc, hw, "llama", "H100", 2, true)
	want := ModelHardwareConfig{
		ModelConfig: mc,
		HWConfig:    hw,
		Model:       "llama",
		GPU:         "H100",
		TP:          2,
		Roofline:    true,
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
