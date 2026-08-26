package kv

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// jitterCfg builds an enabled single-fs-tier offload config with the given
// relative latency-jitter stddev on its tier.
func jitterCfg(sigma float64) sim.KVOffloadConfig {
	cfg := enabledOffloadCfg(1<<20, 4096, 1)
	cfg.Tiers[0].LatencyJitterStddev = sigma
	return cfg
}

// BC-D5: with σ>0 and a seeded RNG, drawJitterFactor returns positive factors that
// vary around 1.0; the sequence is reproducible for the same seed and (almost
// surely) differs for a different seed.
func TestOffloadJitter_DeterministicPerSeed(t *testing.T) {
	drawN := func(seed int64, n int) []float64 {
		gpu := NewKVCacheState(64, 16)
		oc := NewOffloadCache(gpu, jitterCfg(0.2), WithOffloadRNG(rand.New(rand.NewSource(seed))))
		out := make([]float64, n)
		for i := range out {
			out[i] = oc.drawJitterFactor(0)
		}
		return out
	}
	a, b := drawN(42, 20), drawN(42, 20)
	varied := false
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed must reproduce draws: idx %d got %v vs %v", i, a[i], b[i])
		}
		if a[i] <= 0 {
			t.Fatalf("jitter factor must be positive, got %v", a[i])
		}
		if a[i] != 1.0 {
			varied = true
		}
	}
	if !varied {
		t.Fatal("expected jitter to vary from 1.0")
	}
	c := drawN(99, 20)
	allSame := true
	for i := range a {
		if a[i] != c[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Fatal("different seeds must (almost surely) produce different draws")
	}
}

// BC-D5/BC-D2: σ==0 returns the no-jitter sentinel (0) and does NOT consume the RNG
// stream — so an unrelated draw afterward matches a fresh RNG's first value (INV-6:
// a jitter-off run's RNG stream is byte-identical).
func TestOffloadJitter_OffNoDraw(t *testing.T) {
	gpu := NewKVCacheState(64, 16)
	rng := rand.New(rand.NewSource(7))
	oc := NewOffloadCache(gpu, jitterCfg(0), WithOffloadRNG(rng))
	for i := 0; i < 100; i++ {
		if f := oc.drawJitterFactor(0); f != 0 {
			t.Fatalf("σ=0 must return the no-jitter sentinel 0 (no draw), got %v", f)
		}
	}
	got := rng.NormFloat64()
	want := rand.New(rand.NewSource(7)).NormFloat64()
	if got != want {
		t.Fatalf("σ=0 must not consume the RNG stream: got %v want %v", got, want)
	}
}

// BC-D5: a tier with σ>0 but no RNG supplied is a construction error (jitter needs
// a seeded RNG for determinism). NewKVStore always supplies one.
func TestOffloadJitter_MissingRNGPanics(t *testing.T) {
	gpu := NewKVCacheState(64, 16)
	mustPanic(t, "σ>0 without RNG", func() {
		NewOffloadCache(gpu, jitterCfg(0.2)) // no WithOffloadRNG
	})
}
