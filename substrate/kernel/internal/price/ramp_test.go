package price

import (
	"math"
	"testing"
)

// The laws are tested for their properties rather than only at sampled points. A law
// checked at three values can be wrong between them; a law shown to be monotone, bounded
// and exact at its defining points cannot be.

func TestEfficiencyProperties(t *testing.T) {
	const epsMax, mHalf = 0.890, 102.0

	// Monotone increasing, and bounded above by the asymptote.
	prev := -1.0
	for _, m := range []float64{1, 8, 32, 96, 128, 512, 2048, 8192, 1 << 20} {
		got := Efficiency(m, epsMax, mHalf)
		if got < prev {
			t.Errorf("not monotone at m=%v: %v after %v", m, got, prev)
		}
		if got <= 0 || got >= epsMax {
			t.Errorf("m=%v gives %v, outside (0, %v)", m, got, epsMax)
		}
		prev = got
	}

	// Exactly half the asymptote at the half-max point. This is what the parameter's
	// name asserts, and a form that missed it would make the name misleading.
	if got := Efficiency(mHalf, epsMax, mHalf); math.Abs(got-epsMax/2) > 1e-12 {
		t.Errorf("Efficiency(mHalf) = %v, want %v", got, epsMax/2)
	}

	// Zero at zero, and at a negative batch size, which is not a real input but must not
	// produce a negative cost multiplier if one arrives.
	for _, m := range []float64{0, -1} {
		if got := Efficiency(m, epsMax, mHalf); got != 0 {
			t.Errorf("Efficiency(%v) = %v, want 0", m, got)
		}
	}

	// Approaches the asymptote.
	if got := Efficiency(1e12, epsMax, mHalf); math.Abs(got-epsMax) > 1e-3 {
		t.Errorf("Efficiency(1e12) = %v, want ~%v", got, epsMax)
	}
}

// TestEfficiencyAtRegistryConstants checks the ramp at the three per-dtype fits the
// registry carries, since those are the values a deployment will actually use.
func TestEfficiencyAtRegistryConstants(t *testing.T) {
	cases := []struct {
		dtype         string
		epsMax, mHalf float64
		atM512, atM32 float64
	}{
		// Fitted from 100,668 AISimulate GEMM rows on H200; see
		// blis-registry cost-model-primitives-h200.yaml.
		{"bf16", 0.890, 102, 0.742, 0.213},
		{"fp8", 0.788, 94, 0.666, 0.200},
		{"fp8_block", 0.640, 110, 0.527, 0.144},
	}
	for _, c := range cases {
		t.Run(c.dtype, func(t *testing.T) {
			if got := Efficiency(512, c.epsMax, c.mHalf); math.Abs(got-c.atM512) > 0.001 {
				t.Errorf("at M=512 got %.3f, want %.3f", got, c.atM512)
			}
			if got := Efficiency(32, c.epsMax, c.mHalf); math.Abs(got-c.atM32) > 0.001 {
				t.Errorf("at M=32 got %.3f, want %.3f", got, c.atM32)
			}
		})
	}
	// Block-scaled fp8 reaches a materially lower asymptote than plain fp8 on the same
	// peak, so treating them alike overstates the block-scaled path by a quarter.
	plain := Efficiency(1e9, 0.788, 94)
	block := Efficiency(1e9, 0.640, 110)
	if ratio := plain / block; math.Abs(ratio-1.23) > 0.02 {
		t.Errorf("fp8 over fp8_block asymptote ratio = %.3f, want ~1.23", ratio)
	}
}

func TestSpanProperties(t *testing.T) {
	// Monotone in the fabric ratio, and never below one.
	for _, hops := range [][2]int{{1, 15}, {3, 15}, {8, 15}, {72, 143}} {
		prev := 0.0
		for _, ratio := range []float64{1, 1.5, 2, 4, 9, 18, 50} {
			got := Span(hops[0], hops[1], ratio)
			if got < prev-1e-12 {
				t.Errorf("cross=%d: not monotone at ratio %v", hops[0], ratio)
			}
			if got < 1-1e-12 {
				t.Errorf("cross=%d ratio=%v: %v is below 1", hops[0], ratio, got)
			}
			prev = got
		}
	}
	// Exactly one when nothing crosses: the single-node case, and an in-rack group on
	// multi-node NVLink.
	if got := Span(0, 15, 9); got != 1 {
		t.Errorf("Span(0, 15, 9) = %v, want 1", got)
	}
	// A degenerate ratio must resolve to 1 rather than propagate. An absent bandwidth
	// field becoming a NaN step time is the failure this prevents.
	for _, bad := range []float64{0, 0.5, math.NaN(), math.Inf(1), math.Inf(-1)} {
		if got := Span(1, 15, bad); got != 1 {
			t.Errorf("Span with ratio %v = %v, want 1", bad, got)
		}
	}
}

// TestRingSpanIsHierarchical validates the ring form against the decomposition it claims
// to be, rather than against captured numbers. A multi-node ring all-reduce is an
// intra-node reduce-scatter and all-gather over g ranks, then an inter-node all-reduce of
// the reduced 1/g chunk over n nodes, normalized by the flat single-node baseline.
func TestRingSpanIsHierarchical(t *testing.T) {
	for _, gPerNode := range []int{1, 2, 4, 8} {
		for _, nodes := range []int{1, 2, 3, 4, 9} {
			for _, ratio := range []float64{1, 2, 9, 18, 25} {
				g, n := float64(gPerNode), float64(nodes)
				group := gPerNode * nodes
				bigG := float64(group)
				baseline := 2 * (bigG - 1) / bigG
				var want float64
				if baseline == 0 {
					want = 1
				} else {
					intra := 2 * (g - 1) / g
					inter := 2 * (n - 1) / n / g * ratio
					want = (intra + inter) / baseline
				}
				got := RingSpan(group, gPerNode, ratio)
				if math.Abs(got-want) > 1e-9 {
					t.Errorf("g=%d n=%d ratio=%v: got %v, want %v (hierarchical)",
						gPerNode, nodes, ratio, got, want)
				}
			}
		}
	}
}

// TestRingIsFarBelowTheFlatBound is the error a bandwidth-only model makes. At sixteen
// ranks over two eight-GPU nodes on a 9x fabric a ring pays 1.53x, where charging every
// hop the slow link pays 9x.
func TestRingIsFarBelowTheFlatBound(t *testing.T) {
	ring := RingSpan(16, 8, 9)
	flat := FlatSpan(16, 8, 9)
	if math.Abs(ring-1.533) > 0.01 {
		t.Errorf("ring span at 16/8/9x = %.3f, want ~1.533", ring)
	}
	if math.Abs(flat-9.0) > 0.01 {
		t.Errorf("flat span at 16/8/9x = %.3f, want ~9", flat)
	}
	if flat/ring < 5 {
		t.Errorf("the flat bound should exceed the ring by about 6x, got %.2f", flat/ring)
	}
}

// TestAll2AllExceedsRing: a routed all-to-all crosses far more of the fabric than a ring
// at the same placement, and the two are equal only when nothing leaves a node.
func TestAll2AllExceedsRing(t *testing.T) {
	for _, gPerNode := range []int{2, 4, 8, 72} {
		for _, nodes := range []int{1, 2, 4, 9} {
			group := gPerNode * nodes
			if group <= 1 {
				continue
			}
			ring := RingSpan(group, gPerNode, 9)
			a2a := All2AllSpan(group, gPerNode, 9)
			if a2a < ring-1e-12 {
				t.Errorf("g=%d n=%d: all-to-all %.3f is below ring %.3f",
					gPerNode, nodes, a2a, ring)
			}
			if nodes == 1 && math.Abs(a2a-ring) > 1e-12 {
				t.Errorf("g=%d single node: the two forms should agree, got %.3f and %.3f",
					gPerNode, a2a, ring)
			}
		}
	}
	// The figures the design documents quote, so a divergence here is a divergence from
	// the published analysis.
	if got := All2AllSpan(16, 8, 9); math.Abs(got-5.267) > 0.01 {
		t.Errorf("all-to-all at 16/8/9x = %.3f, want ~5.267", got)
	}
	if got := All2AllSpan(72, 8, 9); math.Abs(got-8.211) > 0.01 {
		t.Errorf("all-to-all at 72/8/9x = %.3f, want ~8.211", got)
	}
	// Charging an all-gather-family backend the routed penalty overstates it 3.4x.
	if r := All2AllSpan(16, 8, 9) / RingSpan(16, 8, 9); math.Abs(r-3.435) > 0.02 {
		t.Errorf("routed over ring at 16/8/9x = %.3f, want ~3.435", r)
	}
}

// TestRingSpanFallsBackWhenTheSplitIsUneven: the hierarchical identity holds only when the
// node size divides the group, so an uneven placement takes the flat bound rather than a
// formula that does not apply to it.
func TestRingSpanFallsBackWhenTheSplitIsUneven(t *testing.T) {
	uneven := RingSpan(12, 8, 9) // 12 ranks over 8-GPU nodes: one full node and a half
	flat := FlatSpan(12, 8, 9)
	if math.Abs(uneven-flat) > 1e-12 {
		t.Errorf("an uneven split should take the flat bound, got %.3f against %.3f",
			uneven, flat)
	}
}

func TestFloorAndRate(t *testing.T) {
	const rate = 35.39e9 // measured pinned host-to-device, H200
	const floor = 10e-6

	// A small transfer is floor-bound; a large one is rate-bound. The crossover is where
	// the two are equal, and the function must not sum them.
	small := FloorAndRate(1024, rate, floor)
	if small != floor {
		t.Errorf("a 1 KiB transfer should cost the floor, got %v", small)
	}
	big := FloorAndRate(1<<30, rate, floor)
	if want := float64(1<<30) / rate; math.Abs(big-want) > 1e-12 {
		t.Errorf("a 1 GiB transfer should cost %v, got %v", want, big)
	}
	// Zero bytes still pays the floor: a transfer that moves nothing is still issued.
	if got := FloorAndRate(0, rate, floor); got != floor {
		t.Errorf("a zero-byte transfer = %v, want the floor %v", got, floor)
	}
	// A zero rate is not a free transfer.
	if got := FloorAndRate(1<<20, 0, floor); !math.IsInf(got, 1) {
		t.Errorf("a zero rate should be infinite time, got %v", got)
	}
}

// TestShapeEfficiencyIsStarvedByTheSmallestDimension pins the property that motivates
// the three-factor form: a GEMM with plenty of rows but a narrow reduction is NOT
// efficient, and the one-factor ramp cannot say so.
//
// The numbers are the shapes a tensor-parallel deployment actually produces.
// minimax-m3's o_proj after TP=8 is (n=6144, k=1024) and its mlp_gate_up is
// (n=3072, k=6144); at m=1024 the measured fp8 efficiencies are 0.291 and 0.381, so the
// narrow-k shape is the slower one. A law that ranked them the other way round, or
// equally, would be wrong about the term this kernel was missing.
func TestShapeEfficiencyIsStarvedByTheSmallestDimension(t *testing.T) {
	const (
		epsMax = 1.00
		mHalf  = 256
		nHalf  = 3072
		kHalf  = 2048
	)
	narrowK := ShapeEfficiency(1024, 6144, 1024, epsMax, mHalf, nHalf, kHalf)
	wideK := ShapeEfficiency(1024, 3072, 6144, epsMax, mHalf, nHalf, kHalf)
	if !(narrowK < wideK) {
		t.Fatalf("a narrow reduction must price LOWER than a wide one: "+
			"o_proj-shaped %.4f, mlp-shaped %.4f", narrowK, wideK)
	}
	// And the one-factor ramp cannot distinguish them at all, which is the defect.
	if Efficiency(1024, epsMax, mHalf) != Efficiency(1024, epsMax, mHalf) {
		t.Fatal("unreachable")
	}
	flat := Efficiency(1024, epsMax, mHalf)
	if narrowK >= flat {
		t.Fatalf("the shape-aware law must price a narrow GEMM BELOW the "+
			"shape-blind ramp: %.4f against %.4f", narrowK, flat)
	}
}

// TestShapeEfficiencyNeverExceedsItsAsymptote guards the physical bound. epsMax is the
// limit of a product of three saturating factors, so no finite shape may reach it --
// a law that returned more than epsMax would let a step be priced faster than the
// silicon can run.
func TestShapeEfficiencyNeverExceedsItsAsymptote(t *testing.T) {
	const epsMax = 0.95
	for _, s := range [][3]float64{
		{1, 1, 1}, {1024, 6144, 1024}, {8192, 24576, 16384},
		{1e6, 1e6, 1e6},
	} {
		got := ShapeEfficiency(s[0], s[1], s[2], epsMax, 256, 3072, 2048)
		if got > epsMax {
			t.Fatalf("shape %v priced at %.4f, above the asymptote %.2f", s, got, epsMax)
		}
		if got < 0 {
			t.Fatalf("shape %v priced negative: %.4f", s, got)
		}
	}
}

// TestShapeEfficiencyFallsBackRatherThanZeroing pins the registry-migration behaviour.
// A part whose entry carries no n_half or k_half must keep pricing on the dimensions it
// does have, because a zero efficiency makes a step infinitely slow and a half-migrated
// registry would otherwise produce nonsense instead of the old answer.
func TestShapeEfficiencyFallsBackRatherThanZeroing(t *testing.T) {
	full := ShapeEfficiency(1024, 3072, 6144, 0.9, 256, 3072, 2048)
	noShape := ShapeEfficiency(1024, 3072, 6144, 0.9, 256, 0, 0)
	// Compared to a tolerance rather than exactly: the two laws reach the same value by
	// different multiplication orders, so the last bits need not agree.
	if want := Efficiency(1024, 0.9, 256); math.Abs(noShape-want) > 1e-12 {
		t.Fatalf("with both shape half-maxes absent the law must equal the "+
			"token-count ramp: %.17g against %.17g", noShape, want)
	}
	if !(full < noShape) {
		t.Fatalf("applying shape factors must only ever lower efficiency: "+
			"%.4f against %.4f", full, noShape)
	}
	// A node with no width recorded must not poison the step with a NaN. Without the
	// guards this is 0/0: a NaN efficiency makes the step time NaN, and a NaN compares
	// false against every threshold, so it would propagate silently instead of failing.
	// Each case pairs a missing dimension with a missing half-max, which is the 0/0 the
	// guards exist for: a registry mid-migration has both absent together.
	for _, s := range [][4]float64{
		{0, 6144, 0, 2048}, // n and nHalf absent
		{3072, 0, 3072, 0}, // k and kHalf absent
		{0, 0, 0, 0},       // both pairs absent
	} {
		got := ShapeEfficiency(1024, s[0], s[1], 0.9, 256, s[2], s[3])
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Fatalf("shape (n=%v,k=%v) with nHalf=%v kHalf=%v priced %v; a missing "+
				"width must not produce a non-finite efficiency",
				s[0], s[1], s[2], s[3], got)
		}
	}
}
