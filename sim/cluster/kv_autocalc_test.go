package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// kvAutoCalcTestModel returns a model config with the fields CalculateKVBlocks
// requires (IntermediateDim, VocabSize > 0), plus SwiGLU-family activation via
// the KVCapacityParams. testRooflineModelConfig() lacks IntermediateDim/VocabSize,
// so per-instance KV tests use this dedicated config.
func kvAutoCalcTestModel() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:       4,
		HiddenDim:       256,
		NumHeads:        4,
		NumKVHeads:      4,
		BytesPerParam:   2.0,
		IntermediateDim: 512,
		VocabSize:       1000,
	}
}

// kvAutoCalcTestParams returns KVCapacityParams for a dense SwiGLU model.
func kvAutoCalcTestParams() latency.KVCapacityParams {
	return latency.NewKVCapacityParams(false, 0, false, "silu", 0, 0)
}

// baseSimCfgForKV builds a SimConfig with a starting (global) TotalKVBlocks and a
// valid model for per-instance KV recalculation.
func baseSimCfgForKV(globalBlocks, blockSize int64, maxModelLen int64) sim.SimConfig {
	return sim.SimConfig{
		Horizon:       1_000_000,
		Seed:          42,
		KVCacheConfig: sim.NewKVCacheConfig(globalBlocks, blockSize, 0, 0, 0, 0),
		BatchConfig:   sim.NewBatchConfig(8, 2048, 0),
		LatencyCoeffs: sim.NewLatencyCoeffs(nil, []float64{0, 0, 0}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(kvAutoCalcTestModel(), testRooflineHWCalib(),
			"test-model", "H100", 1, 1, false, "", "roofline", maxModelLen),
	}
}

// TestApplyPerInstanceKVCapacity_Disabled verifies BC-4/BC-5: when auto-calc is
// disabled, TotalKVBlocks is left unchanged regardless of the pool memory.
func TestApplyPerInstanceKVCapacity_Disabled(t *testing.T) {
	simCfg := baseSimCfgForKV(9999, 16, 0)
	cfg := KVAutoCalcConfig{
		Enabled:              false, // disabled
		GPUMemoryUtilization: 0.9,
		Params:               kvAutoCalcTestParams(),
	}
	applyPerInstanceKVCapacity(&simCfg, 48.0, cfg, "L40S")
	if simCfg.TotalKVBlocks != 9999 {
		t.Errorf("disabled auto-calc changed TotalKVBlocks: got %d, want 9999 (unchanged)", simCfg.TotalKVBlocks)
	}
}

// TestApplyPerInstanceKVCapacity_Enabled verifies BC-1 core law: the recomputed
// capacity equals an independent CalculateKVBlocks for the same GPU memory. This
// is a two-way computation law (not a golden constant), so it survives a rewrite.
func TestApplyPerInstanceKVCapacity_Enabled(t *testing.T) {
	const gpuMem = 48.0
	simCfg := baseSimCfgForKV(9999, 16, 0)
	cfg := KVAutoCalcConfig{
		Enabled:              true,
		GPUMemoryUtilization: 0.9,
		Params:               kvAutoCalcTestParams(),
	}

	// Independent expected value: what CalculateKVBlocks returns for this GPU.
	want, err := latency.CalculateKVBlocks(
		kvAutoCalcTestModel(),
		sim.HardwareCalib{MemoryGiB: gpuMem},
		1, 1, 16, 0.9, kvAutoCalcTestParams(),
	)
	if err != nil {
		t.Fatalf("setup: CalculateKVBlocks failed: %v", err)
	}

	applyPerInstanceKVCapacity(&simCfg, gpuMem, cfg, "L40S")

	if simCfg.TotalKVBlocks != want {
		t.Errorf("enabled auto-calc: TotalKVBlocks = %d, want %d (independent CalculateKVBlocks result)", simCfg.TotalKVBlocks, want)
	}
	if simCfg.TotalKVBlocks == 9999 {
		t.Errorf("enabled auto-calc did not change TotalKVBlocks from the global value")
	}
}

// TestApplyPerInstanceKVCapacity_DistinctPerGPU verifies BC-1: two different GPU
// memories yield different capacities (the essence of the bug fix).
func TestApplyPerInstanceKVCapacity_DistinctPerGPU(t *testing.T) {
	cfg := KVAutoCalcConfig{
		Enabled:              true,
		GPUMemoryUtilization: 0.9,
		Params:               kvAutoCalcTestParams(),
	}

	big := baseSimCfgForKV(1, 16, 0)
	applyPerInstanceKVCapacity(&big, 80.0, cfg, "H100")

	small := baseSimCfgForKV(1, 16, 0)
	applyPerInstanceKVCapacity(&small, 48.0, cfg, "L40S")

	if big.TotalKVBlocks <= small.TotalKVBlocks {
		t.Errorf("expected larger GPU memory (80 GiB → %d blocks) to yield MORE blocks than smaller (48 GiB → %d blocks)",
			big.TotalKVBlocks, small.TotalKVBlocks)
	}
}

// TestApplyPerInstanceKVCapacity_MemoryUnavailable verifies BC-7: pool memory <= 0
// falls back to the global capacity without panicking.
func TestApplyPerInstanceKVCapacity_MemoryUnavailable(t *testing.T) {
	simCfg := baseSimCfgForKV(7777, 16, 0)
	cfg := KVAutoCalcConfig{
		Enabled:              true,
		GPUMemoryUtilization: 0.9,
		Params:               kvAutoCalcTestParams(),
	}
	applyPerInstanceKVCapacity(&simCfg, 0.0, cfg, "unknown-gpu")
	if simCfg.TotalKVBlocks != 7777 {
		t.Errorf("memory<=0 fallback: TotalKVBlocks = %d, want 7777 (global, unchanged)", simCfg.TotalKVBlocks)
	}
}

// TestApplyPerInstanceKVCapacity_CalcError verifies BC-7: a CalculateKVBlocks error
// (here: GPU too small for the model) falls back to global, no panic.
func TestApplyPerInstanceKVCapacity_CalcError(t *testing.T) {
	simCfg := baseSimCfgForKV(5555, 16, 0)
	cfg := KVAutoCalcConfig{
		Enabled:              true,
		GPUMemoryUtilization: 0.9,
		Params:               kvAutoCalcTestParams(),
	}
	// A tiny GPU memory (0.01 GiB) cannot fit even the model overhead → error.
	applyPerInstanceKVCapacity(&simCfg, 0.01, cfg, "tiny-gpu")
	if simCfg.TotalKVBlocks != 5555 {
		t.Errorf("calc-error fallback: TotalKVBlocks = %d, want 5555 (global, unchanged)", simCfg.TotalKVBlocks)
	}
}

// TestApplyPerInstanceKVCapacity_MaxModelLenCap verifies BC-6: when the recomputed
// (smaller) capacity cannot hold MaxModelLen, MaxModelLen is capped to
// newBlocks*blockSize so the instance can construct.
func TestApplyPerInstanceKVCapacity_MaxModelLenCap(t *testing.T) {
	const gpuMem = 48.0
	const blockSize = 16
	// Determine the per-GPU capacity first, then set MaxModelLen larger than it can hold.
	blocks, err := latency.CalculateKVBlocks(
		kvAutoCalcTestModel(),
		sim.HardwareCalib{MemoryGiB: gpuMem},
		1, 1, blockSize, 0.9, kvAutoCalcTestParams(),
	)
	if err != nil {
		t.Fatalf("setup: CalculateKVBlocks failed: %v", err)
	}
	kvFeasibleMax := blocks * blockSize
	// MaxModelLen larger than the pool can serve.
	simCfg := baseSimCfgForKV(1, blockSize, kvFeasibleMax+blockSize*10)
	cfg := KVAutoCalcConfig{
		Enabled:              true,
		GPUMemoryUtilization: 0.9,
		Params:               kvAutoCalcTestParams(),
	}
	applyPerInstanceKVCapacity(&simCfg, gpuMem, cfg, "L40S")

	if simCfg.MaxModelLen != kvFeasibleMax {
		t.Errorf("MaxModelLen not capped: got %d, want %d (newBlocks*blockSize)", simCfg.MaxModelLen, kvFeasibleMax)
	}
}

// TestApplyPerInstanceKVCapacity_MaxModelLenNotCappedWhenFits verifies BC-6 boundary:
// when MaxModelLen fits within the recomputed capacity, it is left unchanged.
func TestApplyPerInstanceKVCapacity_MaxModelLenNotCappedWhenFits(t *testing.T) {
	const gpuMem = 80.0
	const blockSize = 16
	const smallMaxLen = int64(32) // trivially fits
	simCfg := baseSimCfgForKV(1, blockSize, smallMaxLen)
	cfg := KVAutoCalcConfig{
		Enabled:              true,
		GPUMemoryUtilization: 0.9,
		Params:               kvAutoCalcTestParams(),
	}
	applyPerInstanceKVCapacity(&simCfg, gpuMem, cfg, "H100")
	if simCfg.MaxModelLen != smallMaxLen {
		t.Errorf("MaxModelLen changed when it should fit: got %d, want %d", simCfg.MaxModelLen, smallMaxLen)
	}
}

// TestKVAutoCalcConfig_ZeroValueDisabled verifies the zero value is inert (BC-5).
func TestKVAutoCalcConfig_ZeroValueDisabled(t *testing.T) {
	var cfg KVAutoCalcConfig
	if cfg.Enabled {
		t.Error("zero-value KVAutoCalcConfig must be disabled (Enabled=false)")
	}
	var dc DeploymentConfig
	if dc.KVAutoCalc.Enabled {
		t.Error("zero-value DeploymentConfig.KVAutoCalc must be disabled")
	}
}
