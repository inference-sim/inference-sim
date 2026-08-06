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

// deploymentForPlacement builds a DeploymentConfig with two node pools of the given
// GPU memories and a global KVCacheConfig, for exercising per-instance KV recalc
// across placement paths. numInstances instances are placed (round-robin across pools
// by first-fit). Auto-calc is enabled iff enabled is true.
func deploymentForPlacement(numInstances int, enabled bool, pools []NodePoolConfig, globalBlocks int64) DeploymentConfig {
	return DeploymentConfig{
		SimConfig:    baseSimCfgForKV(globalBlocks, 16, 0),
		NumInstances: numInstances,
		NodePools:    pools,
		KVAutoCalc: KVAutoCalcConfig{
			Enabled:              enabled,
			GPUMemoryUtilization: 0.9,
			Params:               kvAutoCalcTestParams(),
		},
	}
}

// TestStartupPlacement_PerGPUKVCapacity verifies BC-1: at startup, instances placed
// on pools of different gpu_memory_gib receive different TotalKVBlocks, each equal to
// an independent CalculateKVBlocks for its own pool GPU.
func TestStartupPlacement_PerGPUKVCapacity(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 1, InitialNodes: 1, MinNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80},
		{Name: "l40s", GPUType: "L40S", GPUsPerNode: 1, InitialNodes: 1, MinNodes: 1, MaxNodes: 1, GPUMemoryGiB: 48},
	}
	cfg := deploymentForPlacement(2, true, pools, 9999)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)

	// Map placed instances by their GPU type.
	byGPU := map[string]*InstanceSimulator{}
	for _, inst := range cs.instances {
		byGPU[inst.GPU()] = inst
	}
	h100, okH := byGPU["H100"]
	l40s, okL := byGPU["L40S"]
	if !okH || !okL {
		t.Fatalf("expected instances placed on both H100 and L40S pools, got GPUs %v", func() []string {
			var g []string
			for k := range byGPU {
				g = append(g, k)
			}
			return g
		}())
	}

	wantH, err := latency.CalculateKVBlocks(kvAutoCalcTestModel(), sim.HardwareCalib{MemoryGiB: 80}, 1, 1, 16, 0.9, kvAutoCalcTestParams())
	if err != nil {
		t.Fatalf("setup H100: %v", err)
	}
	wantL, err := latency.CalculateKVBlocks(kvAutoCalcTestModel(), sim.HardwareCalib{MemoryGiB: 48}, 1, 1, 16, 0.9, kvAutoCalcTestParams())
	if err != nil {
		t.Fatalf("setup L40S: %v", err)
	}

	if got := h100.TotalKVBlocks(); got != wantH {
		t.Errorf("H100 instance TotalKVBlocks = %d, want %d (per-GPU CalculateKVBlocks)", got, wantH)
	}
	if got := l40s.TotalKVBlocks(); got != wantL {
		t.Errorf("L40S instance TotalKVBlocks = %d, want %d (per-GPU CalculateKVBlocks)", got, wantL)
	}
	if h100.TotalKVBlocks() == l40s.TotalKVBlocks() {
		t.Errorf("H100 and L40S instances got the same capacity %d — mixed-GPU bug not fixed", h100.TotalKVBlocks())
	}
	if h100.TotalKVBlocks() == 9999 || l40s.TotalKVBlocks() == 9999 {
		t.Errorf("an instance kept the global capacity 9999 instead of its per-GPU value")
	}
}

// TestStartupPlacement_DisabledKeepsGlobal verifies BC-4: with auto-calc disabled
// (explicit --total-kv-blocks), every placed instance keeps the global capacity even
// on pools of differing memory.
func TestStartupPlacement_DisabledKeepsGlobal(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 1, InitialNodes: 1, MinNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80},
		{Name: "l40s", GPUType: "L40S", GPUsPerNode: 1, InitialNodes: 1, MinNodes: 1, MaxNodes: 1, GPUMemoryGiB: 48},
	}
	cfg := deploymentForPlacement(2, false, pools, 9999) // disabled
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)

	for _, inst := range cs.instances {
		if got := inst.TotalKVBlocks(); got != 9999 {
			t.Errorf("instance on GPU %q: TotalKVBlocks = %d, want 9999 (global preserved when auto-calc disabled)", inst.GPU(), got)
		}
	}
}

// TestDeferredPlacement_PerGPUKVCapacity verifies BC-2: an instance constructed in the
// deferred NodeReadyEvent path (InitialNodes=0) receives its pool's per-GPU KV capacity.
func TestDeferredPlacement_PerGPUKVCapacity(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "l40s", GPUType: "L40S", GPUsPerNode: 1, InitialNodes: 0, MinNodes: 0, MaxNodes: 1, GPUMemoryGiB: 48},
	}
	cfg := deploymentForPlacement(1, true, pools, 9999)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 0 {
		t.Fatalf("precondition: expected 0 instances before NodeReadyEvent (InitialNodes=0), got %d", len(cs.instances))
	}

	// Trigger deferred construction.
	node, _ := cs.placement.ProvisionNode("l40s", 0)
	(&NodeReadyEvent{timestamp: 0, nodeID: node.ID}).Execute(cs)
	if len(cs.instances) != 1 {
		t.Fatalf("NodeReadyEvent.Execute constructed %d instances, want 1", len(cs.instances))
	}

	want, err := latency.CalculateKVBlocks(kvAutoCalcTestModel(), sim.HardwareCalib{MemoryGiB: 48}, 1, 1, 16, 0.9, kvAutoCalcTestParams())
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	if got := cs.instances[0].TotalKVBlocks(); got != want {
		t.Errorf("deferred L40S instance TotalKVBlocks = %d, want %d (per-GPU); global was 9999", got, want)
	}
}

// TestAutoscalerScaleUp_PerGPUKVCapacity verifies BC-3: an instance created by the
// autoscaler DirectActuator.scaleUp receives its placed pool's per-GPU KV capacity.
func TestAutoscalerScaleUp_PerGPUKVCapacity(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "l40s", GPUType: "L40S", GPUsPerNode: 2, InitialNodes: 1, MinNodes: 1, MaxNodes: 2, GPUMemoryGiB: 48},
	}
	cfg := deploymentForPlacement(1, true, pools, 9999)
	cfg.Model = "test-model"
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	cs.instances = []*InstanceSimulator{} // clear startup instance to simulate scale-up from empty

	actuator := NewDirectActuator(cs)
	if err := actuator.Apply([]ScaleDecision{
		{ModelID: "test-model", Variant: NewVariantSpec("L40S", 1), Delta: 1},
	}); err != nil {
		t.Fatalf("Apply returned error: %v", err)
	}
	if len(cs.instances) != 1 {
		t.Fatalf("scaleUp created %d instances, want 1", len(cs.instances))
	}

	want, err := latency.CalculateKVBlocks(kvAutoCalcTestModel(), sim.HardwareCalib{MemoryGiB: 48}, 1, 1, 16, 0.9, kvAutoCalcTestParams())
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	if got := cs.instances[0].TotalKVBlocks(); got != want {
		t.Errorf("autoscaler L40S instance TotalKVBlocks = %d, want %d (per-GPU); global was 9999", got, want)
	}
}

// TestApplyPerInstanceKVCapacity_AdapterReservedForwarded verifies that
// AdapterReservedBytes is forwarded to CalculateKVBlocks: a positive reservation
// yields fewer blocks than none (LoRA static HBM reservation shrinks usable KV).
// Guards against a future refactor silently dropping the reservation.
func TestApplyPerInstanceKVCapacity_AdapterReservedForwarded(t *testing.T) {
	const gpuMem = 48.0
	base := KVAutoCalcConfig{Enabled: true, GPUMemoryUtilization: 0.9, Params: kvAutoCalcTestParams()}

	noReserve := baseSimCfgForKV(1, 16, 0)
	applyPerInstanceKVCapacity(&noReserve, gpuMem, base, "L40S")

	withReserve := baseSimCfgForKV(1, 16, 0)
	rcfg := base
	rcfg.AdapterReservedBytes = 2 << 30 // 2 GiB
	applyPerInstanceKVCapacity(&withReserve, gpuMem, rcfg, "L40S")

	// Independent expectation: the reserved path must match CalculateKVBlocks with the option.
	want, err := latency.CalculateKVBlocks(kvAutoCalcTestModel(), sim.HardwareCalib{MemoryGiB: gpuMem},
		1, 1, 16, 0.9, kvAutoCalcTestParams(), latency.WithAdapterReservedBytes(2<<30))
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	if withReserve.TotalKVBlocks != want {
		t.Errorf("reserved TotalKVBlocks = %d, want %d (CalculateKVBlocks with reservation)", withReserve.TotalKVBlocks, want)
	}
	if withReserve.TotalKVBlocks >= noReserve.TotalKVBlocks {
		t.Errorf("AdapterReservedBytes not forwarded: reserved=%d not fewer than unreserved=%d",
			withReserve.TotalKVBlocks, noReserve.TotalKVBlocks)
	}
}

// TestApplyPerInstanceKVCapacity_MaxModelLenZeroUnchanged verifies BC-6 boundary:
// MaxModelLen=0 (unconstrained) is left at 0 after a successful recalc — the guard
// `if simCfg.MaxModelLen > 0` must not turn an unconstrained instance into a
// block-capacity-limited one.
func TestApplyPerInstanceKVCapacity_MaxModelLenZeroUnchanged(t *testing.T) {
	simCfg := baseSimCfgForKV(1, 16, 0) // MaxModelLen=0 (unconstrained)
	cfg := KVAutoCalcConfig{Enabled: true, GPUMemoryUtilization: 0.9, Params: kvAutoCalcTestParams()}
	applyPerInstanceKVCapacity(&simCfg, 48.0, cfg, "L40S")
	if simCfg.MaxModelLen != 0 {
		t.Errorf("MaxModelLen = %d after recalc, want 0 (unconstrained must stay unconstrained)", simCfg.MaxModelLen)
	}
	// Capacity should still have been recomputed.
	if simCfg.TotalKVBlocks == 1 {
		t.Error("TotalKVBlocks unchanged; recalc did not run")
	}
}

// TestDeferredPlacement_DistinctPerGPU verifies BC-2 with TWO distinct pools: a
// pool-lookup bug in the deferred path (picking the wrong GPU memory) would surface
// as equal or swapped capacities. Both pools start deferred (InitialNodes=0).
func TestDeferredPlacement_DistinctPerGPU(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "h100", GPUType: "H100", GPUsPerNode: 1, InitialNodes: 0, MinNodes: 0, MaxNodes: 1, GPUMemoryGiB: 80},
		{Name: "l40s", GPUType: "L40S", GPUsPerNode: 1, InitialNodes: 0, MinNodes: 0, MaxNodes: 1, GPUMemoryGiB: 48},
	}
	cfg := deploymentForPlacement(2, true, pools, 9999)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 0 {
		t.Fatalf("precondition: expected 0 instances before NodeReadyEvent, got %d", len(cs.instances))
	}

	// Provision both nodes and fire their NodeReadyEvents.
	for _, poolName := range []string{"h100", "l40s"} {
		node, _ := cs.placement.ProvisionNode(poolName, 0)
		(&NodeReadyEvent{timestamp: 0, nodeID: node.ID}).Execute(cs)
	}
	if len(cs.instances) != 2 {
		t.Fatalf("expected 2 deferred instances after NodeReadyEvents, got %d", len(cs.instances))
	}

	byGPU := map[string]*InstanceSimulator{}
	for _, inst := range cs.instances {
		byGPU[inst.GPU()] = inst
	}
	h, okH := byGPU["H100"]
	l, okL := byGPU["L40S"]
	if !okH || !okL {
		t.Fatalf("expected deferred instances on both H100 and L40S pools, got GPUs %v", byGPU)
	}
	wantH, _ := latency.CalculateKVBlocks(kvAutoCalcTestModel(), sim.HardwareCalib{MemoryGiB: 80}, 1, 1, 16, 0.9, kvAutoCalcTestParams())
	wantL, _ := latency.CalculateKVBlocks(kvAutoCalcTestModel(), sim.HardwareCalib{MemoryGiB: 48}, 1, 1, 16, 0.9, kvAutoCalcTestParams())
	if got := h.TotalKVBlocks(); got != wantH {
		t.Errorf("deferred H100 instance TotalKVBlocks = %d, want %d (a wrong-pool-memory lookup would surface here)", got, wantH)
	}
	if got := l.TotalKVBlocks(); got != wantL {
		t.Errorf("deferred L40S instance TotalKVBlocks = %d, want %d", got, wantL)
	}
}

// TestNoNodePools_EnabledIsInert verifies BC-5 boundary: even with KVAutoCalc.Enabled=true,
// a deployment with NO node pools never invokes the per-instance recalc (the recalc sites
// live only in the node-pool placement branch), so every instance keeps the global capacity.
// This guards the "inert by design" contract for the no-pool path.
func TestNoNodePools_EnabledIsInert(t *testing.T) {
	cfg := DeploymentConfig{
		SimConfig:    baseSimCfgForKV(1234, 16, 0),
		NumInstances: 2,
		// No NodePools.
		KVAutoCalc: KVAutoCalcConfig{
			Enabled:              true, // enabled, but must be inert without node pools
			GPUMemoryUtilization: 0.9,
			Params:               kvAutoCalcTestParams(),
		},
	}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 2 {
		t.Fatalf("expected 2 instances, got %d", len(cs.instances))
	}
	for _, inst := range cs.instances {
		if got := inst.TotalKVBlocks(); got != 1234 {
			t.Errorf("no-node-pools instance TotalKVBlocks = %d, want 1234 (global preserved; recalc must not fire without node pools)", got)
		}
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
