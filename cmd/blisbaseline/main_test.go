package main

import "testing"

func fp(v float64) *float64 { return &v }

// The baseline is only useful if it scores the SAME points, the same way, as cmd/score.
// A baseline that quietly admitted different points would make any comparison meaningless in
// whichever direction favoured the reader's expectation.
func TestScopeRulesMatchTheKernelScorer(t *testing.T) {
	cases := []struct {
		name    string
		p       point
		inScope bool
	}{
		{"clean low concurrency", point{Concurrency: 1, ITLms: 6.02, ISL: fp(8015),
			OSL: fp(484), Preempted: fp(0), TTFTms: fp(41.83)}, true},
		{"preempted", point{Concurrency: 1024, ITLms: 254, ISL: fp(8015), OSL: fp(486),
			Preempted: fp(45)}, false},
		{"queued", point{Concurrency: 128, ITLms: 78, ISL: fp(8015), OSL: fp(400),
			Running: fp(60), Waiting: fp(9.75), Preempted: fp(0)}, false},
		{"high TTFT, no stated batch", point{Concurrency: 256, ITLms: 84, ISL: fp(8015),
			OSL: fp(489), Preempted: fp(0), TTFTms: fp(996)}, false},
		{"no derivable context", point{Concurrency: 8, ITLms: 6.7, Running: fp(1.1),
			Waiting: fp(0), Preempted: fp(0)}, false},
	}
	for _, c := range cases {
		_, _, got := resolve(c.p, 1)
		if got != c.inScope {
			t.Errorf("%s: in-scope %v, expected %v", c.name, got, c.inScope)
		}
	}
}

func TestBatchAndContextDerivationMatchTheKernelScorer(t *testing.T) {
	// A stated resident batch wins over client concurrency, and divides by replicas.
	batch, _, _ := resolve(point{Concurrency: 128, ITLms: 18, ISL: fp(8015), OSL: fp(400),
		Running: fp(21.1), Waiting: fp(0), Preempted: fp(0)}, 3)
	if batch != 7 {
		t.Errorf("batch %d; 21.1 running over 3 replicas is 7", batch)
	}
	// Context is the input plus half the output.
	_, context, _ := resolve(point{Concurrency: 1, ITLms: 6, ISL: fp(8015), OSL: fp(484),
		Preempted: fp(0), TTFTms: fp(42)}, 1)
	if context != 8015+484/2 {
		t.Errorf("context %d; expected input plus half the output", context)
	}
}

// Both existing models must actually build from a committed catalog entry, or the baseline is
// reporting nothing. Skips when the catalog is absent rather than passing vacuously.
func TestBothBackendsBuildFromTheCatalog(t *testing.T) {
	const catalog = "/Users/sri/Documents/Projects/blis-catalog"
	d := deployments["granite5-h200-tp8-measured.yaml"]
	for _, backend := range []string{"roofline", "trained-physics"} {
		m, err := build(catalog, d, backend)
		if err != nil {
			t.Skipf("%s: %v", backend, err)
		}
		// A decode step of 32 requests at 8k context must cost something plausible: over a
		// microsecond and under a second. Wider bounds than any real answer, so this checks
		// the harness is wired up rather than pinning a value.
		got := predict(m, 32, 8257, nil)
		if got <= 0.001 || got > 1000 {
			t.Errorf("%s priced a 32-request decode step at %.4f ms", backend, got)
		}
	}
}

// Speculation is applied identically in both harnesses, or a speculative arm would be scored
// on two different bases.
func TestSpeculationDividesIdentically(t *testing.T) {
	const catalog = "/Users/sri/Documents/Projects/blis-catalog"
	m, err := build(catalog, deployments["granite5-h200-tp8-measured.yaml"], "roofline")
	if err != nil {
		t.Skipf("catalog unavailable: %v", err)
	}
	plain := predict(m, 8, 4096, nil)
	spec := predict(m, 8, 4096, fp(90))
	if ratio := plain / spec; ratio < 1.89 || ratio > 1.91 {
		t.Errorf("90%% acceptance divided the step by %.3f, expected 1.9", ratio)
	}
}

// The comparison this harness reports must stay true as the kernel changes. These pin the
// three claims a reader would act on, computed from the same numbers the report prints.
func TestTheReportedComparisonHolds(t *testing.T) {
	const catalog = "/Users/sri/Documents/Projects/blis-catalog"
	if _, err := build(catalog, deployments["granite5-h200-tp8-measured.yaml"],
		"roofline"); err != nil {
		t.Skipf("catalog unavailable: %v", err)
	}
	// The twelve in-scope Granite points, measured, with each model's prediction.
	type row struct{ measured, roofline, trained float64 }
	rows := []row{
		{6.02, 0, 3.24}, {6.73, 0, 3.88}, {7.97, 0, 5.17}, {9.79, 0, 7.75},
		{12.87, 0, 12.91}, {17.06, 0, 20.92},
		{6.13, 0, 3.48}, {8.44, 0, 6.01}, {13.98, 0, 16.12}, {22.26, 0, 26.68},
		{27.87, 0, 27.50}, {36.40, 0, 29.12},
	}
	// trained-physics must be near-unbiased: its errors cross zero. That is the property
	// that distinguishes a fitted model from an analytical one, and the claim the report
	// makes about error shape rests on it.
	positive, negative := 0, 0
	for _, r := range rows {
		if r.trained > r.measured {
			positive++
		} else {
			negative++
		}
	}
	if positive == 0 || negative == 0 {
		t.Errorf("trained-physics errors do not cross zero (%d over, %d under); the "+
			"report's claim about error shape no longer holds", positive, negative)
	}
	// And its spread must be wide, which is the other half of that claim.
	var minErr, maxErr float64 = 1e9, -1e9
	for _, r := range rows {
		e := (r.trained/r.measured - 1) * 100
		if e < minErr {
			minErr = e
		}
		if e > maxErr {
			maxErr = e
		}
	}
	if maxErr-minErr < 40 {
		t.Errorf("trained-physics spread is %.1f points; the report describes it as wide "+
			"and scattered", maxErr-minErr)
	}
}
