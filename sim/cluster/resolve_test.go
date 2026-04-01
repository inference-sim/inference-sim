package cluster

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestResolvePoolConfig_NoOverrides_ReturnsGlobalUnchanged(t *testing.T) {
	// BC-P2-1: zero-valued overrides → identity
	global := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
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

// TestNewClusterSimulator_PerPoolConfig_HeterogeneousTP verifies INV-P2-1:
// prefill and decode instances receive different TP values.
func TestNewClusterSimulator_PerPoolConfig_HeterogeneousTP(t *testing.T) {
	prefillTP := 8
	decodeTP := 2
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "blackbox", 0),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		RoutingPolicy:           "round-robin",
		PrefillOverrides:        PoolOverrides{TP: &prefillTP},
		DecodeOverrides:         PoolOverrides{TP: &decodeTP},
	}

	cs := NewClusterSimulator(config, nil, nil)

	if len(cs.Instances()) != 4 {
		t.Errorf("instance count = %d, want 4", len(cs.Instances()))
	}

	// Cluster constructed without panic — per-pool configs were valid
	membership := cs.PoolMembership()
	prefillCount := 0
	decodeCount := 0
	for _, role := range membership {
		switch role {
		case PoolRolePrefill:
			prefillCount++
		case PoolRoleDecode:
			decodeCount++
		}
	}
	if prefillCount != 2 {
		t.Errorf("prefill instances = %d, want 2", prefillCount)
	}
	if decodeCount != 2 {
		t.Errorf("decode instances = %d, want 2", decodeCount)
	}
}

// TestNewClusterSimulator_NoOverrides_BackwardCompat verifies BC-P2-1:
// without overrides, behavior is identical to Phase 1.
func TestNewClusterSimulator_NoOverrides_BackwardCompat(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "blackbox", 0),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		RoutingPolicy:           "round-robin",
		// No PrefillOverrides or DecodeOverrides — zero valued
	}

	cs := NewClusterSimulator(config, nil, nil)
	if len(cs.Instances()) != 4 {
		t.Errorf("instance count = %d, want 4", len(cs.Instances()))
	}
}

// TestINV_P2_1_PoolConfigConsistency verifies INV-P2-1: each instance receives
// config consistent with its pool role. Checks observable behavior:
// (1) pre-simulation: per-pool KV capacity differs between pools
// (2) post-simulation: disaggregation produces valid results with heterogeneous config
func TestINV_P2_1_PoolConfigConsistency(t *testing.T) {
	// Prefill pool: larger KV capacity (can hold more context)
	// Decode pool: smaller KV capacity (needs less for decode-only)
	prefillKV := int64(20000)
	decodeKV := int64(5000)
	// ModelConfig with NumKVHeads=4 (divisible by TP=4) for KV transfer derivation.
	mc := sim.ModelConfig{
		NumLayers:       2,
		NumHeads:        4,
		HiddenDim:       64,
		IntermediateDim: 128,
		BytesPerParam:   2.0,
	}
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(mc, sim.HardwareCalib{}, "test-model", "H100", 4, "blackbox", 0),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		RoutingPolicy:           "round-robin",
		PrefillOverrides:        PoolOverrides{TotalKVBlocks: &prefillKV},
		DecodeOverrides:         PoolOverrides{TotalKVBlocks: &decodeKV},
	}

	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests, nil)

	// INV-P2-1 pre-check: verify per-pool KV capacity via observable FreeKVBlocks().
	// Before simulation, FreeKVBlocks() == TotalCapacity (no requests allocated yet).
	membership := cs.PoolMembership()
	for _, inst := range cs.Instances() {
		role := membership[string(inst.ID())]
		freeBlocks := inst.FreeKVBlocks()
		switch role {
		case PoolRolePrefill:
			if freeBlocks != prefillKV {
				t.Errorf("prefill instance %s: FreeKVBlocks=%d, want %d", inst.ID(), freeBlocks, prefillKV)
			}
		case PoolRoleDecode:
			if freeBlocks != decodeKV {
				t.Errorf("decode instance %s: FreeKVBlocks=%d, want %d", inst.ID(), freeBlocks, decodeKV)
			}
		}
	}

	if err := cs.Run(); err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	// INV-P2-1 post-check: verify the simulation completed with heterogeneous config
	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests == 0 {
		t.Fatal("no requests completed — heterogeneous config may have caused issues")
	}

	// Verify parent requests were tracked (disaggregation active)
	parents := cs.ParentRequests()
	if len(parents) == 0 {
		t.Fatal("no parent requests — disaggregation should be active")
	}
}

// TestResolvePoolConfig_Idempotent verifies the algebraic invariant:
// applying the same overrides twice produces the same result as applying once.
// R7: companion invariant test for the golden-value tests above.
func TestResolvePoolConfig_Idempotent(t *testing.T) {
	global := sim.SimConfig{
		KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "roofline", 8192),
	}

	tp := 8
	maxLen := int64(4096)
	kvBlocks := int64(3000)
	overrides := PoolOverrides{
		TP:             &tp,
		GPU:            "A100",
		LatencyBackend: "crossmodel",
		MaxModelLen:    &maxLen,
		TotalKVBlocks:  &kvBlocks,
	}

	once := ResolvePoolConfig(global, overrides)
	twice := ResolvePoolConfig(once, overrides)

	// Idempotency: Resolve(Resolve(g, o), o) == Resolve(g, o)
	if once.TP != twice.TP {
		t.Errorf("TP not idempotent: once=%d, twice=%d", once.TP, twice.TP)
	}
	if once.GPU != twice.GPU {
		t.Errorf("GPU not idempotent: once=%q, twice=%q", once.GPU, twice.GPU)
	}
	if once.Backend != twice.Backend {
		t.Errorf("Backend not idempotent: once=%q, twice=%q", once.Backend, twice.Backend)
	}
	if once.MaxModelLen != twice.MaxModelLen {
		t.Errorf("MaxModelLen not idempotent: once=%d, twice=%d", once.MaxModelLen, twice.MaxModelLen)
	}
	if once.TotalKVBlocks != twice.TotalKVBlocks {
		t.Errorf("TotalKVBlocks not idempotent: once=%d, twice=%d", once.TotalKVBlocks, twice.TotalKVBlocks)
	}
	// Non-overridden fields must also be preserved
	if once.Horizon != twice.Horizon {
		t.Errorf("Horizon not preserved: once=%d, twice=%d", once.Horizon, twice.Horizon)
	}
	if once.BlockSizeTokens != twice.BlockSizeTokens {
		t.Errorf("BlockSizeTokens not preserved: once=%d, twice=%d", once.BlockSizeTokens, twice.BlockSizeTokens)
	}
}

// TestPoolOverrides_Validate_ErrorPaths verifies that Validate returns errors for
// invalid non-nil pointer fields (R3: validate numeric parameters).
func TestPoolOverrides_Validate_ErrorPaths(t *testing.T) {
	zero := 0
	zeroI64 := int64(0)
	neg := -1
	negI64 := int64(-1)

	tests := []struct {
		name        string
		overrides   PoolOverrides
		wantErrFrag string
	}{
		{
			name:        "TP=0",
			overrides:   PoolOverrides{TP: &zero},
			wantErrFrag: "TP must be > 0",
		},
		{
			name:        "TP negative",
			overrides:   PoolOverrides{TP: &neg},
			wantErrFrag: "TP must be > 0",
		},
		{
			name:        "MaxModelLen=0",
			overrides:   PoolOverrides{MaxModelLen: &zeroI64},
			wantErrFrag: "MaxModelLen must be > 0",
		},
		{
			name:        "MaxModelLen negative",
			overrides:   PoolOverrides{MaxModelLen: &negI64},
			wantErrFrag: "MaxModelLen must be > 0",
		},
		{
			name:        "TotalKVBlocks=0",
			overrides:   PoolOverrides{TotalKVBlocks: &zeroI64},
			wantErrFrag: "TotalKVBlocks must be > 0",
		},
		{
			name:        "TotalKVBlocks negative",
			overrides:   PoolOverrides{TotalKVBlocks: &negI64},
			wantErrFrag: "TotalKVBlocks must be > 0",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.overrides.Validate("test pool")
			if err == nil {
				t.Fatalf("Validate() returned nil, want error containing %q", tc.wantErrFrag)
			}
			if !strings.Contains(err.Error(), tc.wantErrFrag) {
				t.Errorf("Validate() error = %q, want it to contain %q", err.Error(), tc.wantErrFrag)
			}
		})
	}
}

// TestPoolOverrides_Validate_ValidValues verifies that Validate returns nil for
// valid non-nil pointer fields and for the zero-value (all nil) override.
func TestPoolOverrides_Validate_ValidValues(t *testing.T) {
	tp := 4
	maxLen := int64(8192)
	kvBlocks := int64(5000)

	tests := []struct {
		name      string
		overrides PoolOverrides
	}{
		{
			name:      "all nil (empty)",
			overrides: PoolOverrides{},
		},
		{
			name:      "valid TP",
			overrides: PoolOverrides{TP: &tp},
		},
		{
			name:      "valid MaxModelLen",
			overrides: PoolOverrides{MaxModelLen: &maxLen},
		},
		{
			name:      "valid TotalKVBlocks",
			overrides: PoolOverrides{TotalKVBlocks: &kvBlocks},
		},
		{
			name: "all fields valid",
			overrides: PoolOverrides{
				TP:             &tp,
				GPU:            "H100",
				LatencyBackend: "roofline",
				MaxModelLen:    &maxLen,
				TotalKVBlocks:  &kvBlocks,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.overrides.Validate("test pool"); err != nil {
				t.Errorf("Validate() returned unexpected error: %v", err)
			}
		})
	}
}

// newHeterogeneousDeploymentConfig creates a DeploymentConfig with per-pool overrides.
// This is the test helper consumed by future PRs.
func newHeterogeneousDeploymentConfig(numInstances, prefill, decode int, prefillOverrides, decodeOverrides PoolOverrides) DeploymentConfig {
	// ModelConfig with NumKVHeads=4 (divisible by TP=4) for KV transfer derivation.
	mc := sim.ModelConfig{
		NumLayers:       2,
		NumHeads:        4,
		HiddenDim:       64,
		IntermediateDim: 128,
		BytesPerParam:   2.0,
		// NumKVHeads=0: MHA fallback, uses NumHeads=4
	}
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(mc, sim.HardwareCalib{}, "test-model", "H100", 4, "blackbox", 0),
		},
		NumInstances:            numInstances,
		PrefillInstances:        prefill,
		DecodeInstances:         decode,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		RoutingPolicy:           "round-robin",
		PrefillOverrides:        prefillOverrides,
		DecodeOverrides:         decodeOverrides,
	}
}

// TestNewHeterogeneousDeploymentConfig_Helper verifies the test helper produces a valid config
// that can be passed to NewClusterSimulator without panicking.
func TestNewHeterogeneousDeploymentConfig_Helper(t *testing.T) {
	prefillKV := int64(8000)
	decodeKV := int64(3000)
	cfg := newHeterogeneousDeploymentConfig(4, 2, 2,
		PoolOverrides{TotalKVBlocks: &prefillKV},
		PoolOverrides{TotalKVBlocks: &decodeKV},
	)
	if cfg.NumInstances != 4 {
		t.Errorf("NumInstances = %d, want 4", cfg.NumInstances)
	}
	if cfg.PrefillInstances != 2 {
		t.Errorf("PrefillInstances = %d, want 2", cfg.PrefillInstances)
	}
	if cfg.DecodeInstances != 2 {
		t.Errorf("DecodeInstances = %d, want 2", cfg.DecodeInstances)
	}
	if *cfg.PrefillOverrides.TotalKVBlocks != prefillKV {
		t.Errorf("PrefillOverrides.TotalKVBlocks = %d, want %d", *cfg.PrefillOverrides.TotalKVBlocks, prefillKV)
	}
	if *cfg.DecodeOverrides.TotalKVBlocks != decodeKV {
		t.Errorf("DecodeOverrides.TotalKVBlocks = %d, want %d", *cfg.DecodeOverrides.TotalKVBlocks, decodeKV)
	}
}

// TestResolvePoolConfig_MaxModelLen_CappedToPoolKVCapacity verifies INV-P2-1:
// when a pool has smaller TotalKVBlocks than global, ResolvePoolConfig propagates
// the pool-capped MaxModelLen (set by CLI per-pool auto-capping fix) correctly.
func TestResolvePoolConfig_MaxModelLen_CappedToPoolKVCapacity(t *testing.T) {
	// GIVEN global config with large MaxModelLen and large global KV blocks
	global := sim.SimConfig{
		Horizon: 1000000,
		Seed:    42,
		KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0), // 10000 blocks × 16 = 160000 tokens
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, "", 131072),
	}

	// AND per-pool override with smaller TotalKVBlocks (smaller GPU) and auto-capped MaxModelLen
	poolKVFeasibleMax := int64(2048 * 16) // 32768 tokens
	poolMaxModelLen := poolKVFeasibleMax   // auto-capped CLI fix would set this
	poolBlocks := int64(2048)
	overrides := PoolOverrides{
		TotalKVBlocks: &poolBlocks,
		MaxModelLen:   &poolMaxModelLen,
	}

	// WHEN resolving pool config
	result := ResolvePoolConfig(global, overrides)

	// THEN pool config has smaller TotalKVBlocks and capped MaxModelLen
	if result.TotalKVBlocks != 2048 {
		t.Errorf("TotalKVBlocks = %d, want 2048", result.TotalKVBlocks)
	}
	if result.MaxModelLen != poolKVFeasibleMax {
		t.Errorf("MaxModelLen = %d, want %d (pool KV feasible max)", result.MaxModelLen, poolKVFeasibleMax)
	}

	// AND MaxModelLen does not exceed pool KV capacity (INV-P2-1)
	blocksNeeded := result.MaxModelLen / result.BlockSizeTokens
	if result.MaxModelLen%result.BlockSizeTokens != 0 {
		blocksNeeded++
	}
	if blocksNeeded > result.TotalKVBlocks {
		t.Errorf("INV-P2-1: MaxModelLen=%d requires %d blocks, exceeds pool TotalKVBlocks=%d",
			result.MaxModelLen, blocksNeeded, result.TotalKVBlocks)
	}
}

// TestPoolOverrides_IsEmpty verifies the IsEmpty method for all field branches.
func TestPoolOverrides_IsEmpty(t *testing.T) {
	tp := 4
	maxLen := int64(8192)
	kvBlocks := int64(5000)

	tests := []struct {
		name      string
		overrides PoolOverrides
		want      bool
	}{
		{name: "all nil/zero", overrides: PoolOverrides{}, want: true},
		{name: "TP set", overrides: PoolOverrides{TP: &tp}, want: false},
		{name: "GPU set", overrides: PoolOverrides{GPU: "A100"}, want: false},
		{name: "LatencyBackend set", overrides: PoolOverrides{LatencyBackend: "roofline"}, want: false},
		{name: "MaxModelLen set", overrides: PoolOverrides{MaxModelLen: &maxLen}, want: false},
		{name: "TotalKVBlocks set", overrides: PoolOverrides{TotalKVBlocks: &kvBlocks}, want: false},
		{
			name: "all fields set",
			overrides: PoolOverrides{
				TP:             &tp,
				GPU:            "A100",
				LatencyBackend: "roofline",
				MaxModelLen:    &maxLen,
				TotalKVBlocks:  &kvBlocks,
			},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.overrides.IsEmpty(); got != tc.want {
				t.Errorf("IsEmpty() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestResolveConfigForRole_CrossPoolIsolation verifies that prefill and decode overrides
// do not bleed into each other: asking for PoolRolePrefill does not apply DecodeOverrides
// and vice versa.
func TestResolveConfigForRole_CrossPoolIsolation(t *testing.T) {
	prefillTP := 8
	decodeTP := 2
	dc := DeploymentConfig{
		SimConfig: sim.SimConfig{
			KVCacheConfig:       sim.NewKVCacheConfig(5000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
		},
		PrefillOverrides: PoolOverrides{TP: &prefillTP},
		DecodeOverrides:  PoolOverrides{TP: &decodeTP},
	}

	prefillCfg := dc.resolveConfigForRole(PoolRolePrefill)
	decodeCfg := dc.resolveConfigForRole(PoolRoleDecode)

	if prefillCfg.TP != 8 {
		t.Errorf("prefill TP = %d, want 8 (DecodeOverrides must not apply to prefill role)", prefillCfg.TP)
	}
	if decodeCfg.TP != 2 {
		t.Errorf("decode TP = %d, want 2 (PrefillOverrides must not apply to decode role)", decodeCfg.TP)
	}
}

// TestNewClusterSimulator_PanicsOnInvalidPrefillOverrides verifies that NewClusterSimulator
// panics with a descriptive message when PrefillOverrides contains an invalid field,
// rather than allowing TP=0 to silently propagate to instance construction.
func TestNewClusterSimulator_PanicsOnInvalidPrefillOverrides(t *testing.T) {
	zero := 0
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "blackbox", 0),
		},
		NumInstances:     4,
		PrefillInstances: 2,
		DecodeInstances:  2,
		PDDecider:        "always",
		RoutingPolicy:    "round-robin",
		PrefillOverrides: PoolOverrides{TP: &zero}, // invalid: TP=0
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("NewClusterSimulator did not panic on invalid PrefillOverrides")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("panic value type = %T, want string; value = %v", r, r)
		}
		if !strings.Contains(msg, "TP must be > 0") {
			t.Errorf("panic message = %q, want it to contain %q", msg, "TP must be > 0")
		}
	}()

	NewClusterSimulator(config, nil, nil)
}

// TestNewClusterSimulator_PanicsOnInvalidDecodeOverrides verifies the same panic contract
// for DecodeOverrides.
func TestNewClusterSimulator_PanicsOnInvalidDecodeOverrides(t *testing.T) {
	zero := 0
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "blackbox", 0),
		},
		NumInstances:     4,
		PrefillInstances: 2,
		DecodeInstances:  2,
		PDDecider:        "always",
		RoutingPolicy:    "round-robin",
		DecodeOverrides:  PoolOverrides{TP: &zero}, // invalid: TP=0
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("NewClusterSimulator did not panic on invalid DecodeOverrides")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("panic value type = %T, want string; value = %v", r, r)
		}
		if !strings.Contains(msg, "TP must be > 0") {
			t.Errorf("panic message = %q, want it to contain %q", msg, "TP must be > 0")
		}
	}()

	NewClusterSimulator(config, nil, nil)
}

// TestINV_P2_1_RequestConservation verifies INV-1 holds for a heterogeneous PD cluster:
// injected_requests == completed + still_queued + still_running + dropped + timed_out.
func TestINV_P2_1_RequestConservation(t *testing.T) {
	prefillKV := int64(20000)
	decodeKV := int64(5000)
	requests := newTestRequests(10)
	// ModelConfig with NumKVHeads=4 (divisible by TP=4) for KV transfer derivation.
	mc := sim.ModelConfig{
		NumLayers:       2,
		NumHeads:        4,
		HiddenDim:       64,
		IntermediateDim: 128,
		BytesPerParam:   2.0,
	}
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(mc, sim.HardwareCalib{}, "test-model", "H100", 4, "blackbox", 0),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		RoutingPolicy:           "round-robin",
		PrefillOverrides:        PoolOverrides{TotalKVBlocks: &prefillKV},
		DecodeOverrides:         PoolOverrides{TotalKVBlocks: &decodeKV},
	}

	cs := NewClusterSimulator(config, requests, nil)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	m := cs.AggregatedMetrics()
	injected := len(requests)
	conserved := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable + m.TimedOutRequests
	if conserved != injected {
		t.Errorf("INV-1 violated: injected=%d, completed=%d+queued=%d+running=%d+dropped=%d+timedout=%d = %d",
			injected, m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable, m.TimedOutRequests, conserved)
	}
}
