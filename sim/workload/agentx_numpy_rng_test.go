package workload

import (
	"encoding/json"
	"os"
	"strconv"
	"testing"
)

type numpyGolden struct {
	NumpyVersion string `json:"numpy_version"`
	Cases        []struct {
		Seed string `json:"seed"`
		Lane *struct {
			Base    int64  `json:"base"`
			TraceID string `json:"trace_id"`
			Lane    int    `json:"lane"`
		} `json:"lane"`
		Raw       []string `json:"raw"`
		Random    []string `json:"random"`
		UniformLo float64  `json:"uniform_lo"`
		UniformHi float64  `json:"uniform_hi"`
		Uniform   string   `json:"uniform"`
	} `json:"cases"`
}

// TestNumpyGeneratorMatchesNumpy pins the Go port of default_rng(int) to vectors numpy
// produced (testdata/agentx/gen_numpy_rng_golden.py): raw PCG64 words, random() doubles,
// uniform(lo, hi), and aiperf's sha256 lane-seed derivation, all bit for bit.
func TestNumpyGeneratorMatchesNumpy(t *testing.T) {
	b, err := os.ReadFile("testdata/agentx/numpy_rng_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g numpyGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("no golden cases")
	}
	for _, c := range g.Cases {
		seed, err := strconv.ParseUint(c.Seed, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		if c.Lane != nil {
			if got := agentxLaneSeed(c.Lane.Base, c.Lane.TraceID, c.Lane.Lane); got != seed {
				t.Errorf("lane seed %+v: got %d want %d", *c.Lane, got, seed)
			}
		}
		gen := newNumpyGenerator(seed)
		for i, w := range c.Raw {
			want, _ := strconv.ParseUint(w, 10, 64)
			if got := gen.next64(); got != want {
				t.Errorf("seed %d raw[%d]: got %d want %d", seed, i, got, want)
			}
		}
		gen = newNumpyGenerator(seed)
		for i, h := range c.Random {
			want, _ := strconv.ParseFloat(h, 64)
			if got := gen.random(); got != want {
				t.Errorf("seed %d random[%d]: got %x want %x", seed, i, got, want)
			}
		}
		want, _ := strconv.ParseFloat(c.Uniform, 64)
		if got := newNumpyGenerator(seed).uniform(c.UniformLo, c.UniformHi); got != want {
			t.Errorf("seed %d uniform: got %x want %x", seed, got, want)
		}
	}
}
