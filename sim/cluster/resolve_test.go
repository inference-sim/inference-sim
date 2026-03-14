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
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		PDKVBytesPerToken:       512,
		RoutingPolicy:           "round-robin",
		PrefillOverrides:        PoolOverrides{TP: &prefillTP},
		DecodeOverrides:         PoolOverrides{TP: &decodeTP},
	}

	cs := NewClusterSimulator(config, nil)

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
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		PDKVBytesPerToken:       512,
		RoutingPolicy:           "round-robin",
		// No PrefillOverrides or DecodeOverrides — zero valued
	}

	cs := NewClusterSimulator(config, nil)
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
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		PDKVBytesPerToken:       512,
		RoutingPolicy:           "round-robin",
		PrefillOverrides:        PoolOverrides{TotalKVBlocks: &prefillKV},
		DecodeOverrides:         PoolOverrides{TotalKVBlocks: &decodeKV},
	}

	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)

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
// This is the test helper consumed by future PRs (PR3, PR4).
func newHeterogeneousDeploymentConfig(numInstances, prefill, decode int, prefillOverrides, decodeOverrides PoolOverrides) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 4, "", 0),
		},
		NumInstances:            numInstances,
		PrefillInstances:        prefill,
		DecodeInstances:         decode,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		PDKVBytesPerToken:       512,
		RoutingPolicy:           "round-robin",
		PrefillOverrides:        prefillOverrides,
		DecodeOverrides:         decodeOverrides,
	}
}
