package latency

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/lora"
	"github.com/stretchr/testify/require"
)

// This file holds the PR4 contract (T030) and companion invariant (T032) tests
// for the per-step adapter compute-overhead factor. They verify the OBSERVABLE
// effect on StepTime — that the factor derived by the cost model actually flows
// into both latency backends identically (R23) — rather than re-testing the
// factor math itself (that lives in sim/lora/cost_model_test.go).
//
// It imports sim/lora only as a test dependency to build a real cost model; the
// production latency package must never import sim/lora (the accessor arrives as
// the sim.AdapterCost interface).

func f64(v float64) *float64 { return &v }

// fakeAdapterCost is a minimal sim.AdapterCost that returns a fixed overhead
// factor regardless of batch. Used to assert both backends apply the SAME factor
// with identical arithmetic (R23), decoupled from cost-model rank/tier logic.
type fakeAdapterCost struct{ factor float64 }

func (f fakeAdapterCost) LoadLatency(string) float64                { return 0 }
func (f fakeAdapterCost) StepOverheadFactor([]*sim.Request) float64 { return f.factor }
func (f fakeAdapterCost) AdapterReservedBytes() float64             { return 0 }

// newAdapterCost builds a real *lora.CostModel (satisfying sim.AdapterCost) with
// the given rank tiers and adapter registry. Load-cost coefficients are inert
// (they do not affect StepOverheadFactor).
func newAdapterCost(t *testing.T, tiers map[int]sim.StepOverheadTier, adapters []sim.AdapterSpec) sim.AdapterCost {
	t.Helper()
	capacity := 16
	cfg := sim.LoRAConfig{
		AdapterCapacity:       &capacity,
		LoadBaseLatencyUs:     f64(0),
		LoadBandwidthBytesUs:  f64(1),
		FootprintBytesPerRank: f64(1),
		StepOverheadTiers:     tiers,
		Adapters:              adapters,
	}
	ac, err := lora.NewCostModel(cfg)
	require.NoError(t, err, "test setup: NewCostModel")
	return ac
}

// prefillReq builds a prefill-phase request of n tokens carrying adapter id.
func prefillReq(n int, adapter string) *sim.Request {
	return &sim.Request{
		InputTokens:   make([]sim.TokenID, n),
		ProgressIndex: 0,
		NumNewTokens:  n,
		Adapter:       adapter,
	}
}

// backends returns one roofline and one trained-physics model built via the
// production constructor with the supplied accessor (nil ⇒ no accessor).
func backends(t *testing.T, ac sim.AdapterCost) map[string]sim.LatencyModel {
	t.Helper()
	var opts []Option
	if ac != nil {
		opts = append(opts, WithAdapterCost(ac))
	}
	roofCoeffs := sim.NewLatencyCoeffs(nil, []float64{100, 1, 100})
	roofHW := sim.NewModelHardwareConfig(testModelConfig(), testHardwareCalib(), "", "", 2, 1, false, "", "roofline", 0)
	roof, err := NewLatencyModel(roofCoeffs, roofHW, opts...)
	require.NoError(t, err, "roofline NewLatencyModel")

	tpHW := sim.NewModelHardwareConfig(testModelConfig(), testHardwareCalib(), "", "", 1, 1, false, "", "trained-physics", 0)
	tp, err := NewLatencyModel(*testCoeffs(), tpHW, opts...)
	require.NoError(t, err, "trained-physics NewLatencyModel")

	return map[string]sim.LatencyModel{"roofline": roof, "trained-physics": tp}
}

// T030 / T032: factor is exactly 1.0 when A_B==0 (no adapter ids in batch), so
// StepTime is byte-identical to a model built without an accessor — for ANY
// fitted K7, not only K7==1 (INV-6, SC-001). This is the no-op regression guard.
func TestStepTime_NoAdapters_ByteIdentical_NonUnitK7(t *testing.T) {
	// Non-unit K7 tier — normalized factor must still be exactly 1.0 at A_B=0.
	ac := newAdapterCost(t,
		map[int]sim.StepOverheadTier{8: {K6: f64(3), K7: f64(7)}},
		[]sim.AdapterSpec{{ID: "a1", Rank: 8}},
	)
	withAC := backends(t, ac)
	noAC := backends(t, nil)

	batches := map[string][]*sim.Request{
		"empty":     {},
		"base-only": {prefillReq(100, ""), prefillReq(50, "")},
	}
	for name, model := range withAC {
		for bname, batch := range batches {
			got := model.StepTime(batch)
			want := noAC[name].StepTime(batch)
			if got != want {
				t.Errorf("%s StepTime(%s) with accessor = %d, want byte-identical %d (INV-6)", name, bname, got, want)
			}
			if got < 1 {
				t.Errorf("%s StepTime(%s) = %d, want >= 1 (INV-3)", name, bname, got)
			}
		}
	}
}

// T030: nil accessor is a no-op (INV-6) — an adapter-carrying batch times exactly
// as it would pre-feature when no accessor is wired.
func TestStepTime_NilAccessor_NoEffect(t *testing.T) {
	models := backends(t, nil)
	batch := []*sim.Request{prefillReq(100, "a1"), prefillReq(80, "a2")}
	bare := []*sim.Request{prefillReq(100, ""), prefillReq(80, "")}
	for name, model := range models {
		if got, want := model.StepTime(batch), model.StepTime(bare); got != want {
			t.Errorf("%s nil-accessor StepTime differs on adapter ids: %d vs %d (INV-6)", name, got, want)
		}
	}
}

// T030: the factor genuinely enters StepTime — an adapter-carrying batch is
// strictly slower than the identical batch with the ids stripped (same base).
func TestStepTime_AdapterBatch_SlowerThanBase(t *testing.T) {
	ac := newAdapterCost(t,
		map[int]sim.StepOverheadTier{8: {K6: f64(1), K7: f64(1)}}, // factor = 1 + A_B
		[]sim.AdapterSpec{{ID: "a1", Rank: 8}},
	)
	for name, model := range backends(t, ac) {
		withAdapter := model.StepTime([]*sim.Request{prefillReq(100, "a1"), prefillReq(100, "a1")})
		bare := model.StepTime([]*sim.Request{prefillReq(100, ""), prefillReq(100, "")})
		if withAdapter <= bare {
			t.Errorf("%s: adapter batch %d not slower than bare %d", name, withAdapter, bare)
		}
	}
}

// T030: monotonic in the count of DISTINCT adapters, and a duplicate adapter is
// counted once (A_B(2×a1) == A_B(1×a1) < A_B(a1,a2)). Same token structure
// isolates the factor from the base step time.
func TestStepTime_MonotonicInDistinctCount_DuplicateOnce(t *testing.T) {
	ac := newAdapterCost(t,
		map[int]sim.StepOverheadTier{8: {K6: f64(1), K7: f64(1)}}, // factor = 1 + A_B
		[]sim.AdapterSpec{{ID: "a1", Rank: 8}, {ID: "a2", Rank: 8}},
	)
	for name, model := range backends(t, ac) {
		one := model.StepTime([]*sim.Request{prefillReq(100, "a1"), prefillReq(100, "a1")}) // A_B=1
		two := model.StepTime([]*sim.Request{prefillReq(100, "a1"), prefillReq(100, "a2")}) // A_B=2
		if two <= one {
			t.Errorf("%s: A_B=2 StepTime %d not > A_B=1 StepTime %d (monotonic in distinct count)", name, two, one)
		}
	}
}

// T030: strictly increasing in max rank (FR-009) — a higher-rank tier with a
// larger K6/K7 ratio yields a strictly larger factor for the same A_B>0, proving
// rank actually selects the tier. Same single-request base isolates the factor.
func TestStepTime_StrictlyIncreasingInMaxRank(t *testing.T) {
	// ratios: rank 8 → 1/4 = 0.25 (factor 1.25); rank 64 → 3/2 = 1.5 (factor 2.5).
	ac := newAdapterCost(t,
		map[int]sim.StepOverheadTier{
			8:  {K6: f64(1), K7: f64(4)},
			64: {K6: f64(3), K7: f64(2)},
		},
		[]sim.AdapterSpec{{ID: "low", Rank: 8}, {ID: "high", Rank: 64}},
	)
	for name, model := range backends(t, ac) {
		low := model.StepTime([]*sim.Request{prefillReq(200, "low")})
		high := model.StepTime([]*sim.Request{prefillReq(200, "high")})
		if high <= low {
			t.Errorf("%s: higher max-rank StepTime %d not > lower-rank %d (FR-009)", name, high, low)
		}
	}
}

// T030: an out-of-envelope rank is clamped to the nearest calibrated tier — the
// factor stays >= 1 (no inversion below the base) and equals the nearest-tier
// factor rather than extrapolating.
func TestStepTime_OutOfEnvelopeRank_ClampedNoInversion(t *testing.T) {
	ac := newAdapterCost(t,
		map[int]sim.StepOverheadTier{
			8:  {K6: f64(1), K7: f64(4)},
			64: {K6: f64(3), K7: f64(2)},
		},
		[]sim.AdapterSpec{{ID: "huge", Rank: 1000}, {ID: "top", Rank: 64}},
	)
	for name, model := range backends(t, ac) {
		bare := model.StepTime([]*sim.Request{prefillReq(200, "")})
		huge := model.StepTime([]*sim.Request{prefillReq(200, "huge")})
		top := model.StepTime([]*sim.Request{prefillReq(200, "top")})
		if huge < bare {
			t.Errorf("%s: clamped out-of-envelope StepTime %d < base %d (inversion)", name, huge, bare)
		}
		if huge != top {
			t.Errorf("%s: rank 1000 StepTime %d != nearest-tier(64) StepTime %d (clamp)", name, huge, top)
		}
	}
}

// T032 (robustness, INV-3): a contract-violating accessor returning a non-finite
// factor (NaN/±Inf) must NOT stall the clock. applyAdapterOverhead treats such a
// factor as "no overhead" and returns the base step time unchanged (>= 1), rather
// than letting NaN reach clampToInt64 and saturate to MaxInt64.
func TestStepTime_NonFiniteFactor_TreatedAsNoOp(t *testing.T) {
	batch := []*sim.Request{prefillReq(150, "a1"), prefillReq(120, "a2")}
	base := backends(t, nil)
	for _, factor := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		withBad := backends(t, fakeAdapterCost{factor: factor})
		for name := range withBad {
			got := withBad[name].StepTime(batch)
			want := base[name].StepTime(batch)
			if got != want {
				t.Errorf("%s: non-finite factor %v gave StepTime %d, want base %d (no stall, INV-3)", name, factor, got, want)
			}
			if got < 1 {
				t.Errorf("%s: StepTime %d < 1 with factor %v (INV-3)", name, got, factor)
			}
		}
	}
}

// T030 (R23): both backends apply the identical factor with identical arithmetic.
// A fake accessor pins the factor so the assertion is exact: StepTime_with_adapter
// == max(1, floor(base * factor)) computed the same way for each backend.
func TestStepTime_BackendParity_IdenticalFactorApplication(t *testing.T) {
	const factor = 2.0
	fake := fakeAdapterCost{factor: factor}
	withAC := backends(t, fake)
	noAC := backends(t, nil)

	batch := []*sim.Request{prefillReq(150, "a1"), prefillReq(120, "a2")}
	for name := range withAC {
		base := noAC[name].StepTime(batch)
		got := withAC[name].StepTime(batch)
		want := int64(float64(base) * factor)
		if want < 1 {
			want = 1
		}
		if got != want {
			t.Errorf("%s: StepTime with factor %.1f = %d, want max(1, base %d * %.1f) = %d (R23 identical application)",
				name, factor, got, base, factor, want)
		}
	}
}
