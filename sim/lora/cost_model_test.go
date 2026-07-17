package lora

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// The adapter cost model derives the three Digital-Twin cost terms — cold-load
// latency, per-step compute-overhead factor, and HBM footprint — as pure,
// deterministic functions of adapter rank plus configured coefficients
// (data-model.md "Adapter Cost Model"). These contract tests assert the
// observable laws (non-negativity, monotonicity in rank, the no-adapter
// normalization, purity) independent of the internal representation.

// fptr is a small helper for building the pointer-typed LoRAConfig cost fields.
func fptr(v float64) *float64 { return &v }

// testCostModel builds a cost model over three ranked adapters (8/16/32) with
// simple, well-conditioned coefficients. base=1000µs, bandwidth=2e6 B/µs,
// footprint=2e6 B/rank.
func testCostModel(t *testing.T) *CostModel {
	t.Helper()
	cfg := sim.LoRAConfig{
		LoadBaseLatencyUs:     fptr(1000.0),
		LoadBandwidthBytesUs:  fptr(2.0e6),
		FootprintBytesPerRank: fptr(2.0e6),
		StepOverheadTiers: map[int]sim.StepOverheadTier{
			8:  {K6: fptr(0.02), K7: fptr(1.0)},
			16: {K6: fptr(0.04), K7: fptr(1.0)},
			32: {K6: fptr(0.06), K7: fptr(1.0)},
		},
		Adapters: []sim.AdapterSpec{
			{ID: "a8", Rank: 8},
			{ID: "a16", Rank: 16},
			{ID: "a32", Rank: 32},
		},
	}
	cm, err := NewCostModel(cfg)
	if err != nil {
		t.Fatalf("NewCostModel: unexpected error: %v", err)
	}
	return cm
}

// TestCostModel_LoadLatencyNonNegativeAndCharged verifies load latency is >= 0
// for every registered adapter and 0 for a base-model (empty id) or unknown id
// (SC-004 charge shape; T025).
func TestCostModel_LoadLatencyNonNegative(t *testing.T) {
	cm := testCostModel(t)

	for _, id := range []string{"a8", "a16", "a32"} {
		if got := cm.LoadLatency(id); got < 0 {
			t.Errorf("LoadLatency(%q) = %v, want >= 0", id, got)
		}
	}
	// base-model (empty) and unregistered ids carry no load cost.
	if got := cm.LoadLatency(""); got != 0 {
		t.Errorf("LoadLatency(\"\") = %v, want 0 (base model)", got)
	}
	if got := cm.LoadLatency("ghost"); got != 0 {
		t.Errorf("LoadLatency(ghost) = %v, want 0 (unregistered)", got)
	}

	// Exact value: base + ceil(footprint(rank)/bandwidth). rank=8 => footprint=1.6e7,
	// /2e6 = 8 => 1000 + 8 = 1008.
	if got, want := cm.LoadLatency("a8"), 1008.0; got != want {
		t.Errorf("LoadLatency(a8) = %v, want %v", got, want)
	}
}

// TestCostModel_LoadLatencyMonotonicInRank verifies a higher-rank adapter costs
// at least as much to load as a lower-rank one (footprint grows with rank).
func TestCostModel_LoadLatencyMonotonicInRank(t *testing.T) {
	cm := testCostModel(t)
	if !(cm.LoadLatency("a8") <= cm.LoadLatency("a16") && cm.LoadLatency("a16") <= cm.LoadLatency("a32")) {
		t.Errorf("LoadLatency not monotonic in rank: a8=%v a16=%v a32=%v",
			cm.LoadLatency("a8"), cm.LoadLatency("a16"), cm.LoadLatency("a32"))
	}
}

// TestCostModel_FootprintMonotonicInRank verifies footprint is >= 0 and strictly
// increases with rank (data-model.md; T025).
func TestCostModel_FootprintMonotonicInRank(t *testing.T) {
	cm := testCostModel(t)
	f8, f16, f32 := cm.FootprintBytes("a8"), cm.FootprintBytes("a16"), cm.FootprintBytes("a32")
	if f8 < 0 || f16 < 0 || f32 < 0 {
		t.Fatalf("footprint must be >= 0: a8=%v a16=%v a32=%v", f8, f16, f32)
	}
	if !(f8 < f16 && f16 < f32) {
		t.Errorf("footprint not strictly increasing in rank: a8=%v a16=%v a32=%v", f8, f16, f32)
	}
	if got := cm.FootprintBytes(""); got != 0 {
		t.Errorf("FootprintBytes(\"\") = %v, want 0", got)
	}
}

// TestCostModel_StepOverheadUnitWhenNoAdapters verifies the compute-overhead
// factor is exactly 1.0 for a batch with no adapters (INV-6 no-op), for any
// fitted K7 — a batch of base-model requests or an empty batch both normalize to
// 1.0 (T025).
func TestCostModel_StepOverheadUnitWhenNoAdapters(t *testing.T) {
	cm := testCostModel(t)

	if got := cm.StepOverheadFactor(nil); got != 1.0 {
		t.Errorf("StepOverheadFactor(nil) = %v, want 1.0", got)
	}
	baseOnly := []*sim.Request{{ID: "r1"}, {ID: "r2"}} // Adapter == "" for both
	if got := cm.StepOverheadFactor(baseOnly); got != 1.0 {
		t.Errorf("StepOverheadFactor(base-only batch) = %v, want 1.0", got)
	}

	// Non-unit K7 must still normalize to exactly 1.0 at A_B==0 (guards the no-op
	// regression: the factor is (K7 + K6*A_B)/K7, = 1.0 for any K7 when A_B==0).
	cfg := sim.LoRAConfig{
		LoadBaseLatencyUs:     fptr(1000.0),
		LoadBandwidthBytesUs:  fptr(2.0e6),
		FootprintBytesPerRank: fptr(2.0e6),
		StepOverheadTiers:     map[int]sim.StepOverheadTier{8: {K6: fptr(0.5), K7: fptr(3.7)}},
		Adapters:              []sim.AdapterSpec{{ID: "a8", Rank: 8}},
	}
	cm2, err := NewCostModel(cfg)
	if err != nil {
		t.Fatalf("NewCostModel: %v", err)
	}
	if got := cm2.StepOverheadFactor(baseOnly); got != 1.0 {
		t.Errorf("StepOverheadFactor with non-unit K7, no adapters = %v, want 1.0", got)
	}
}

// TestCostModel_StepOverheadGrowsWithAdapters verifies the factor is >= 1.0 and
// increases with the count of distinct adapters, counting a repeated adapter
// once (spec edge case; T025 / US3 scenario 2 precondition).
func TestCostModel_StepOverheadGrowsWithAdapters(t *testing.T) {
	cm := testCostModel(t)

	one := []*sim.Request{{ID: "r1", Adapter: "a8"}}
	// Same adapter twice => counted once => same factor as one.
	dup := []*sim.Request{{ID: "r1", Adapter: "a8"}, {ID: "r2", Adapter: "a8"}}
	two := []*sim.Request{{ID: "r1", Adapter: "a8"}, {ID: "r2", Adapter: "a16"}}

	fOne, fDup, fTwo := cm.StepOverheadFactor(one), cm.StepOverheadFactor(dup), cm.StepOverheadFactor(two)
	if fOne < 1.0 {
		t.Errorf("factor for one adapter = %v, want >= 1.0", fOne)
	}
	if fDup != fOne {
		t.Errorf("duplicate adapter counted more than once: dup=%v one=%v", fDup, fOne)
	}
	if !(fTwo > fOne) {
		t.Errorf("factor did not grow with distinct adapters: two=%v one=%v", fTwo, fOne)
	}
}

// TestCostModel_StepOverheadIgnoresUnregisteredAdapter verifies an unregistered
// adapter id is treated as base-model in the compute-overhead factor — consistent
// with LoadLatency and FootprintBytes, which both return 0 for an unregistered id.
// It must not inflate A_B (nor contribute a phantom rank), so a batch of only
// unregistered ids normalizes to 1.0 and adding an unregistered id alongside a
// registered one leaves the factor unchanged.
func TestCostModel_StepOverheadIgnoresUnregisteredAdapter(t *testing.T) {
	cm := testCostModel(t)

	ghostOnly := []*sim.Request{{ID: "r1", Adapter: "ghost"}, {ID: "r2", Adapter: "phantom"}}
	if got := cm.StepOverheadFactor(ghostOnly); got != 1.0 {
		t.Errorf("StepOverheadFactor(unregistered-only batch) = %v, want 1.0 (treated as base-model)", got)
	}

	one := []*sim.Request{{ID: "r1", Adapter: "a8"}}
	oneWithGhost := []*sim.Request{{ID: "r1", Adapter: "a8"}, {ID: "r2", Adapter: "ghost"}}
	if got, want := cm.StepOverheadFactor(oneWithGhost), cm.StepOverheadFactor(one); got != want {
		t.Errorf("unregistered id inflated the factor: with-ghost=%v, registered-only=%v", got, want)
	}
}

// TestCostModel_Deterministic verifies the queries are pure: repeated calls with
// the same inputs return identical results and do not mutate the model (INV-6, R7).
func TestCostModel_Deterministic(t *testing.T) {
	cm := testCostModel(t)
	batch := []*sim.Request{{ID: "r1", Adapter: "a8"}, {ID: "r2", Adapter: "a32"}}
	for i := 0; i < 3; i++ {
		if cm.LoadLatency("a16") != cm.LoadLatency("a16") ||
			cm.FootprintBytes("a16") != cm.FootprintBytes("a16") ||
			cm.StepOverheadFactor(batch) != cm.StepOverheadFactor(batch) {
			t.Fatalf("cost model not deterministic on iteration %d", i)
		}
	}
}

// TestNewCostModel_RejectsMalformedConfig verifies the constructor guards missing
// or non-positive divisor coefficients (R3/R11), mirroring LoRAConfig.Validate so
// a directly-built model is as safe as one from a validated config.
func TestNewCostModel_RejectsMalformedConfig(t *testing.T) {
	base := func() sim.LoRAConfig {
		return sim.LoRAConfig{
			LoadBaseLatencyUs:     fptr(1000.0),
			LoadBandwidthBytesUs:  fptr(2.0e6),
			FootprintBytesPerRank: fptr(2.0e6),
			Adapters:              []sim.AdapterSpec{{ID: "a8", Rank: 8}},
		}
	}
	tests := []struct {
		name  string
		mutate func(*sim.LoRAConfig)
	}{
		{"nil load base latency", func(c *sim.LoRAConfig) { c.LoadBaseLatencyUs = nil }},
		{"nil bandwidth (divisor)", func(c *sim.LoRAConfig) { c.LoadBandwidthBytesUs = nil }},
		{"zero bandwidth (divisor)", func(c *sim.LoRAConfig) { c.LoadBandwidthBytesUs = fptr(0) }},
		{"nil footprint", func(c *sim.LoRAConfig) { c.FootprintBytesPerRank = nil }},
		{"negative base latency", func(c *sim.LoRAConfig) { c.LoadBaseLatencyUs = fptr(-1) }},
		{"NaN base latency", func(c *sim.LoRAConfig) { c.LoadBaseLatencyUs = fptr(math.NaN()) }},
		{"Inf bandwidth", func(c *sim.LoRAConfig) { c.LoadBandwidthBytesUs = fptr(math.Inf(1)) }},
		{"NaN footprint", func(c *sim.LoRAConfig) { c.FootprintBytesPerRank = fptr(math.NaN()) }},
		{"rank overflows footprint to Inf", func(c *sim.LoRAConfig) {
			c.FootprintBytesPerRank = fptr(math.MaxFloat64)
			c.Adapters = []sim.AdapterSpec{{ID: "a8", Rank: 1 << 40}} // huge rank ⇒ footprint = Inf
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := base()
			tt.mutate(&cfg)
			if _, err := NewCostModel(cfg); err == nil {
				t.Errorf("NewCostModel(%s): expected error, got nil", tt.name)
			}
		})
	}
}
