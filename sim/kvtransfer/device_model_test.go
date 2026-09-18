package kvtransfer

import "testing"

// rampTier builds a single-tier config with a queue-depth bandwidth ramp on both
// directions. peakBW is the saturated (q>=Qsat) bandwidth in bytes/tick; f1 is
// the single-transfer (q=1) fraction of peak.
func rampTier(peakBW float64, base int64, qsat int, f1 float64) Config {
	return Config{Tiers: []TierConfig{{
		NRead:                  16,
		NWrite:                 16,
		ReadBaseTicks:          base,
		WriteBaseTicks:         base,
		ReadBytesPerTick:       peakBW,
		WriteBytesPerTick:      peakBW,
		SaturationQueueDepth:   qsat,
		SingleTransferFraction: f1,
	}}}
}

// TestServiceTicksAtDepth_Ramp asserts the BC-D1 numbers: effBW ramps linearly
// from f1*peak at q=1 to peak at q=Qsat, flat beyond. peak=10, base=100,
// Qsat=4, f1=0.5, bytes=1000.
func TestServiceTicksAtDepth_Ramp(t *testing.T) {
	s := mustNew(t, rampTier(10, 100, 4, 0.5))
	cases := []struct {
		q         int
		wantTicks int64 // base + floor(bytes/effBW)
	}{
		{1, 300}, // effBW = 0.5*10 = 5;      1000/5   = 200 -> 300
		{2, 250}, // ramp = 0.5+0.5*(1/3);    effBW=6.6667; 1000/6.6667=150 -> 250
		{3, 220}, // ramp = 0.5+0.5*(2/3)=0.8333; effBW=8.3333; 1000/8.3333=120 -> 220
		{4, 200}, // effBW = 10 (saturated);  1000/10  = 100 -> 200
		{8, 200}, // q>Qsat: flat at peak
	}
	for _, tc := range cases {
		got := s.ServiceTicksAtDepth(0, Read, 1000, tc.q)
		if got != tc.wantTicks {
			t.Errorf("ServiceTicksAtDepth(Read,1000,q=%d) = %d, want %d", tc.q, got, tc.wantTicks)
		}
	}
}

// TestServiceTicksAtDepth_MonotoneNonDecreasing: service time is non-increasing
// in q (more concurrency => >= effective bandwidth => <= service time), for a
// ramp config, up to and beyond Qsat.
func TestServiceTicksAtDepth_MonotoneNonDecreasing(t *testing.T) {
	s := mustNew(t, rampTier(7, 80, 8, 0.3))
	prev := int64(1<<62 - 1)
	for q := 1; q <= 20; q++ {
		got := s.ServiceTicksAtDepth(0, Read, 1<<20, q)
		if got > prev {
			t.Errorf("service time increased with concurrency: q=%d got %d > prev %d", q, got, prev)
		}
		prev = got
	}
}

// TestServiceTicksAtDepth_FlatBeyondQsat: identical service time for all q>=Qsat.
func TestServiceTicksAtDepth_FlatBeyondQsat(t *testing.T) {
	s := mustNew(t, rampTier(10, 100, 4, 0.5))
	at4 := s.ServiceTicksAtDepth(0, Read, 1000, 4)
	for q := 4; q <= 32; q++ {
		if got := s.ServiceTicksAtDepth(0, Read, 1000, q); got != at4 {
			t.Errorf("expected flat beyond Qsat: q=%d got %d, want %d", q, got, at4)
		}
	}
}

// TestServiceTicks_RampOff_Identity: with the ramp disabled (Qsat<=1, or f1>=1,
// or the zero-value default), ServiceTicksAtDepth at ANY q equals the plain
// base+floor(bytes/bw) and equals the uncontended ServiceTicks (BC-D2, INV-6).
func TestServiceTicks_RampOff_Identity(t *testing.T) {
	configs := []Config{
		oneTier(4, 4, 100, 300),                 // zero-value ramp fields
		rampTier(10, 100, 1, 0.5),               // Qsat=1 => no ramp
		rampTier(10, 100, 4, 1.0),               // f1=1.0 => no ramp
		rampTier(10, 100, 0, 0.5),               // Qsat=0 => no ramp
	}
	for ci, cfg := range configs {
		s := mustNew(t, cfg)
		bw := cfg.Tiers[0].ReadBytesPerTick
		base := cfg.Tiers[0].ReadBaseTicks
		for _, bytes := range []int64{0, 1, 5, 1000, 1005, 1 << 20} {
			want := base + int64(float64(bytes)/bw)
			uncontended := s.ServiceTicks(0, Read, bytes)
			if uncontended != want {
				t.Errorf("cfg %d: ServiceTicks(bytes=%d) = %d, want %d", ci, bytes, uncontended, want)
			}
			for _, q := range []int{1, 2, 5, 16, 100} {
				if got := s.ServiceTicksAtDepth(0, Read, bytes, q); got != want {
					t.Errorf("cfg %d: ServiceTicksAtDepth(bytes=%d,q=%d) = %d, want %d (ramp must be off)", ci, bytes, q, got, want)
				}
			}
		}
	}
}

// TestNew_RampValidation: Qsat>=2 requires f1 in (0,1]; other combos accepted.
func TestNew_RampValidation(t *testing.T) {
	bad := []Config{
		rampTier(10, 100, 4, 0),    // f1=0 with Qsat>=2
		rampTier(10, 100, 4, -0.1), // negative f1
		rampTier(10, 100, 4, 1.5),  // f1>1
		rampTier(10, 100, -3, 0.5), // negative Qsat
	}
	for i, cfg := range bad {
		if _, err := New(cfg); err == nil {
			t.Errorf("bad ramp cfg %d: expected validation error, got nil", i)
		}
	}
	good := []Config{
		rampTier(10, 100, 4, 0.5), // valid ramp
		rampTier(10, 100, 4, 1.0), // f1=1 allowed (no-op)
		rampTier(10, 100, 1, 0.0), // Qsat=1, f1 ignored
		rampTier(10, 100, 0, 0.0), // Qsat=0, f1 ignored
	}
	for i, cfg := range good {
		if _, err := New(cfg); err != nil {
			t.Errorf("good ramp cfg %d: unexpected error %v", i, err)
		}
	}
}

// TestStart_JitterFactorApplied: a job with JitterFactor f completes at
// start + round(serviceTicks · f) (BC-D5). bw=1, base=100, bytes=500 => svc=600;
// factor 1.5 => 900 => completeAt 1900.
func TestStart_JitterFactorApplied(t *testing.T) {
	s := mustNew(t, oneTier(4, 4, 100, 100))
	id, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: 500, SubmitTick: 1000, JitterFactor: 1.5})
	if !ok {
		t.Fatal("submit rejected")
	}
	if got := s.Poll(1899); len(got) != 0 {
		t.Fatalf("completed early at 1899: %v", got)
	}
	got := s.Poll(1900)
	if len(got) != 1 || got[0] != id {
		t.Fatalf("want [%d] at tick 1900, got %v", id, got)
	}
}

// TestStart_NoJitterSentinel_ByteIdentical: JitterFactor ≤ 0 (the default) uses
// the exact integer service time — completion equals submit + ServiceTicks with
// no float perturbation (BC-D2/INV-6).
func TestStart_NoJitterSentinel_ByteIdentical(t *testing.T) {
	for _, bytes := range []int64{0, 1, 499, 500, 1000, 1005} {
		s := mustNew(t, oneTier(4, 4, 100, 100))
		want := int64(1000) + s.ServiceTicks(0, Read, bytes) // lone job => q=1
		id, ok := s.Submit(TransferJob{Tier: 0, Direction: Read, Bytes: bytes, SubmitTick: 1000}) // JitterFactor 0
		if !ok {
			t.Fatalf("bytes=%d rejected", bytes)
		}
		if want > 1000 {
			if got := s.Poll(want - 1); len(got) != 0 {
				t.Fatalf("bytes=%d completed early at %d: %v", bytes, want-1, got)
			}
		}
		got := s.Poll(want)
		if len(got) != 1 || got[0] != id {
			t.Fatalf("bytes=%d want [%d] at tick %d, got %v", bytes, id, want, got)
		}
	}
}

// TestBCS4_DeterministicWithFactors: fixed per-job jitter factors keep the
// completion sequence deterministic across repeated runs (BC-S4 still holds when
// the station is fed a scalar factor rather than drawing randomness itself).
func TestBCS4_DeterministicWithFactors(t *testing.T) {
	run := func() []JobID {
		s := mustNew(t, oneTier(2, 2, 50, 50))
		factors := []float64{1.3, 0.7, 2.0, 0.9, 1.1, 0.5}
		var out []JobID
		for i, f := range factors {
			dir := Read
			if i%2 == 1 {
				dir = Write
			}
			s.Submit(TransferJob{Tier: 0, Direction: dir, Bytes: int64(100 * (i + 1)), SubmitTick: int64(10 * i), JitterFactor: f})
			out = append(out, s.Poll(int64(10*i))...)
		}
		out = append(out, s.Poll(1<<30)...)
		return out
	}
	a, b := run(), run()
	if len(a) != len(b) {
		t.Fatalf("nondeterministic length: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("nondeterministic completion order at %d: %v vs %v", i, a, b)
		}
	}
}
