package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestResolvePoolConfig_NoOverrides_ReturnsGlobalUnchanged(t *testing.T) {
	// BC-P2-1: zero-valued overrides → identity
	global := sim.SimConfig{
		Horizon: 1000000,
		Seed:    42,
		KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "roofline", 8192),
	}
	overrides := PoolOverrides{} // all nil/zero

	resolved := ResolvePoolConfig(global, overrides)

	if resolved.TP != global.TP {
		t.Errorf("TP = %d, want %d", resolved.TP, global.TP)
	}
	if resolved.GPU != global.GPU {
		t.Errorf("GPU = %q, want %q", resolved.GPU, global.GPU)
	}
	if resolved.Backend != global.Backend {
		t.Errorf("Backend = %q, want %q", resolved.Backend, global.Backend)
	}
	if resolved.MaxModelLen != global.MaxModelLen {
		t.Errorf("MaxModelLen = %d, want %d", resolved.MaxModelLen, global.MaxModelLen)
	}
	if resolved.TotalKVBlocks != global.TotalKVBlocks {
		t.Errorf("TotalKVBlocks = %d, want %d", resolved.TotalKVBlocks, global.TotalKVBlocks)
	}
	// Non-overridden fields must also be identical
	if resolved.Horizon != global.Horizon {
		t.Errorf("Horizon = %d, want %d", resolved.Horizon, global.Horizon)
	}
	if resolved.Seed != global.Seed {
		t.Errorf("Seed = %d, want %d", resolved.Seed, global.Seed)
	}
}

func TestResolvePoolConfig_AllOverrides_Applied(t *testing.T) {
	// BC-P2-2: each override field applies independently
	global := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "roofline", 8192),
	}

	tp := 2
	maxLen := int64(4096)
	kvBlocks := int64(3000)
	overrides := PoolOverrides{
		TP:             &tp,
		GPU:            "A100",
		LatencyBackend: "crossmodel",
		MaxModelLen:    &maxLen,
		TotalKVBlocks:  &kvBlocks,
	}

	resolved := ResolvePoolConfig(global, overrides)

	if resolved.TP != 2 {
		t.Errorf("TP = %d, want 2", resolved.TP)
	}
	if resolved.GPU != "A100" {
		t.Errorf("GPU = %q, want %q", resolved.GPU, "A100")
	}
	if resolved.Backend != "crossmodel" {
		t.Errorf("Backend = %q, want %q", resolved.Backend, "crossmodel")
	}
	if resolved.MaxModelLen != 4096 {
		t.Errorf("MaxModelLen = %d, want 4096", resolved.MaxModelLen)
	}
	if resolved.TotalKVBlocks != 3000 {
		t.Errorf("TotalKVBlocks = %d, want 3000", resolved.TotalKVBlocks)
	}
	// Non-overridden fields stay global
	if resolved.Horizon != global.Horizon {
		t.Errorf("Horizon changed: %d, want %d", resolved.Horizon, global.Horizon)
	}
	if resolved.BlockSizeTokens != global.BlockSizeTokens {
		t.Errorf("BlockSizeTokens changed: %d, want %d", resolved.BlockSizeTokens, global.BlockSizeTokens)
	}
}

func TestResolvePoolConfig_PartialOverrides_OnlySpecifiedFieldsChange(t *testing.T) {
	global := sim.SimConfig{
		KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "roofline", 8192),
	}

	tp := 8
	overrides := PoolOverrides{TP: &tp} // only TP override

	resolved := ResolvePoolConfig(global, overrides)

	if resolved.TP != 8 {
		t.Errorf("TP = %d, want 8", resolved.TP)
	}
	// Everything else unchanged
	if resolved.GPU != "H100" {
		t.Errorf("GPU = %q, want %q", resolved.GPU, "H100")
	}
	if resolved.Backend != "roofline" {
		t.Errorf("Backend = %q, want %q", resolved.Backend, "roofline")
	}
	if resolved.MaxModelLen != 8192 {
		t.Errorf("MaxModelLen = %d, want 8192", resolved.MaxModelLen)
	}
	if resolved.TotalKVBlocks != 5000 {
		t.Errorf("TotalKVBlocks = %d, want 5000", resolved.TotalKVBlocks)
	}
}

func TestResolvePoolConfig_DoesNotMutateGlobal(t *testing.T) {
	global := sim.SimConfig{
		KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
	}
	origTP := global.TP

	tp := 8
	overrides := PoolOverrides{TP: &tp}
	_ = ResolvePoolConfig(global, overrides)

	if global.TP != origTP {
		t.Errorf("global.TP mutated: %d, want %d", global.TP, origTP)
	}
}

func TestResolveConfigForRole_Prefill(t *testing.T) {
	tp := 8
	dc := DeploymentConfig{
		SimConfig: sim.SimConfig{
			KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
		},
		PrefillOverrides: PoolOverrides{TP: &tp},
	}

	cfg := dc.resolveConfigForRole(PoolRolePrefill)
	if cfg.TP != 8 {
		t.Errorf("prefill TP = %d, want 8", cfg.TP)
	}
}

func TestResolveConfigForRole_Decode(t *testing.T) {
	tp := 2
	dc := DeploymentConfig{
		SimConfig: sim.SimConfig{
			KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
		},
		DecodeOverrides: PoolOverrides{TP: &tp},
	}

	cfg := dc.resolveConfigForRole(PoolRoleDecode)
	if cfg.TP != 2 {
		t.Errorf("decode TP = %d, want 2", cfg.TP)
	}
}

func TestResolveConfigForRole_NoRole_ReturnsGlobal(t *testing.T) {
	tp := 8
	dc := DeploymentConfig{
		SimConfig: sim.SimConfig{
			KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
		},
		PrefillOverrides: PoolOverrides{TP: &tp},
	}

	cfg := dc.resolveConfigForRole(PoolRole(0)) // no role
	if cfg.TP != 4 {
		t.Errorf("no-role TP = %d, want 4 (global)", cfg.TP)
	}
}
