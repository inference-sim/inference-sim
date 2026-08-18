package sim

import (
	"math"
	"strings"
	"testing"
)

// validFSTier returns a fully-resolved, valid fs tier for use as a test baseline.
func validFSTier() KVOffloadTier {
	return KVOffloadTier{
		Type:           "fs",
		RootDir:        "/mnt/kv",
		NReadThreads:   16,
		NWriteThreads:  16,
		Locality:       "LOCAL",
		EnableKVEvents: false,
		DirectIO:       true,
		DeviceClass:    "nvme_gen4",
		ReadBandwidth:  7000,
		WriteBandwidth: 5000,
		BaseLatency:    80,
	}
}

// validEnabledConfig returns a valid resolved multi-tier config.
func validEnabledConfig() KVOffloadConfig {
	return KVOffloadConfig{
		Enabled:                true,
		CPUBytesToUse:          17179869184,
		BlockSize:              16,
		BlocksPerChunk:         1,
		TokensPerHash:          16,
		EvictionPolicy:         "lru",
		OffloadPromptOnly:      true,
		SelfDescribingKVEvents: false,
		Tiers:                  []KVOffloadTier{validFSTier()},
	}
}

// T1: zero value is inert.
func TestKVOffloadConfig_ZeroValueInert(t *testing.T) {
	var c KVOffloadConfig
	if c.IsEnabled() {
		t.Fatalf("zero-value KVOffloadConfig must be inert (IsEnabled()==false)")
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("inert config must validate as a no-op, got %v", err)
	}
}

// T2: Validate accepts a valid config and rejects each invariant violation, naming the field.
func TestKVOffloadConfig_Validate(t *testing.T) {
	if err := validEnabledConfig().Validate(); err != nil {
		t.Fatalf("valid config must pass Validate, got %v", err)
	}

	tests := []struct {
		name     string
		mutate   func(*KVOffloadConfig)
		wantFrag string
	}{
		{"cpu_bytes zero", func(c *KVOffloadConfig) { c.CPUBytesToUse = 0 }, "cpu_bytes_to_use"},
		{"cpu_bytes negative", func(c *KVOffloadConfig) { c.CPUBytesToUse = -1 }, "cpu_bytes_to_use"},
		{"block_size zero", func(c *KVOffloadConfig) { c.BlockSize = 0 }, "block_size"},
		{"blocks_per_chunk zero", func(c *KVOffloadConfig) { c.BlocksPerChunk = 0 }, "blocks_per_chunk"},
		{"tokens_per_hash zero", func(c *KVOffloadConfig) { c.TokensPerHash = 0 }, "tokens_per_hash"},
		{"unknown eviction policy", func(c *KVOffloadConfig) { c.EvictionPolicy = "fifo" }, "eviction_policy"},
		{"tier type obj", func(c *KVOffloadConfig) { c.Tiers[0].Type = "obj" }, "type"},
		{"tier type p2p", func(c *KVOffloadConfig) { c.Tiers[0].Type = "p2p" }, "type"},
		{"tier type example", func(c *KVOffloadConfig) { c.Tiers[0].Type = "example" }, "type"},
		{"missing root_dir", func(c *KVOffloadConfig) { c.Tiers[0].RootDir = "" }, "root_dir"},
		{"bad locality", func(c *KVOffloadConfig) { c.Tiers[0].Locality = "somewhere" }, "locality"},
		{"read threads zero", func(c *KVOffloadConfig) { c.Tiers[0].NReadThreads = 0 }, "n_read_threads"},
		{"write threads zero", func(c *KVOffloadConfig) { c.Tiers[0].NWriteThreads = 0 }, "n_write_threads"},
		{"read bw zero", func(c *KVOffloadConfig) { c.Tiers[0].ReadBandwidth = 0 }, "read_bandwidth"},
		{"read bw NaN", func(c *KVOffloadConfig) { c.Tiers[0].ReadBandwidth = math.NaN() }, "read_bandwidth"},
		{"read bw Inf", func(c *KVOffloadConfig) { c.Tiers[0].ReadBandwidth = math.Inf(1) }, "read_bandwidth"},
		{"write bw negative", func(c *KVOffloadConfig) { c.Tiers[0].WriteBandwidth = -5 }, "write_bandwidth"},
		{"write bw NaN", func(c *KVOffloadConfig) { c.Tiers[0].WriteBandwidth = math.NaN() }, "write_bandwidth"},
		{"base latency negative", func(c *KVOffloadConfig) { c.Tiers[0].BaseLatency = -1 }, "base_latency"},
		{"base latency NaN", func(c *KVOffloadConfig) { c.Tiers[0].BaseLatency = math.NaN() }, "base_latency"},
		{"base latency Inf", func(c *KVOffloadConfig) { c.Tiers[0].BaseLatency = math.Inf(1) }, "base_latency"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := validEnabledConfig()
			tc.mutate(&c)
			err := c.Validate()
			if err == nil {
				t.Fatalf("expected Validate error for %q, got nil", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantFrag) {
				t.Fatalf("error %q must name the offending field %q", err.Error(), tc.wantFrag)
			}
		})
	}
}

// T2: a valid config with zero secondary tiers (CPU-only offload) is valid.
func TestKVOffloadConfig_Validate_NoSecondaryTiers(t *testing.T) {
	c := validEnabledConfig()
	c.Tiers = nil
	if err := c.Validate(); err != nil {
		t.Fatalf("enabled config with no secondary tiers (CPU-only) must validate, got %v", err)
	}
}

// T2: base_latency == 0 is allowed (finite, >= 0).
func TestKVOffloadConfig_Validate_ZeroBaseLatencyOK(t *testing.T) {
	c := validEnabledConfig()
	c.Tiers[0].BaseLatency = 0
	if err := c.Validate(); err != nil {
		t.Fatalf("zero base_latency must be allowed, got %v", err)
	}
}

// T3: NewKVCacheConfig without options yields an inert Offload (byte-identical behavior).
func TestNewKVCacheConfig_NoOffloadOption_Inert(t *testing.T) {
	cfg := NewKVCacheConfig(1000, 16, 0, 0, 0, 0)
	if cfg.Offload.IsEnabled() {
		t.Fatalf("NewKVCacheConfig without WithKVOffload must leave Offload inert")
	}
	if cfg.Offload.CPUBytesToUse != 0 || cfg.Offload.BlockSize != 0 || len(cfg.Offload.Tiers) != 0 {
		t.Fatalf("Offload must be the zero value, got %+v", cfg.Offload)
	}
}

// T3: WithKVOffload threads the resolved sub-config through the constructor.
func TestNewKVCacheConfig_WithKVOffload(t *testing.T) {
	want := validEnabledConfig()
	cfg := NewKVCacheConfig(1000, 16, 0, 0, 0, 0, WithKVOffload(want))
	if !cfg.Offload.IsEnabled() {
		t.Fatalf("WithKVOffload must set Offload enabled")
	}
	if cfg.Offload.CPUBytesToUse != want.CPUBytesToUse || len(cfg.Offload.Tiers) != 1 {
		t.Fatalf("Offload not threaded through: got %+v", cfg.Offload)
	}
}

// T3: an invalid enabled offload triggers factory validation (panic), matching the
// constructor's panic-on-invalid contract.
func TestNewKVCacheConfig_WithInvalidOffload_Panics(t *testing.T) {
	bad := validEnabledConfig()
	bad.CPUBytesToUse = 0 // invalid
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("NewKVCacheConfig must panic on an invalid enabled offload config")
		}
	}()
	_ = NewKVCacheConfig(1000, 16, 0, 0, 0, 0, WithKVOffload(bad))
}
