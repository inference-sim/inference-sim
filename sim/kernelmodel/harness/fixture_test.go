package harness

import (
	"math/bits"
	"os"
	"testing"
)

// fixture_test.go checks the committed InferenceX length vectors without Python: a small,
// independent implementation of NumPy's legacy global generator recomputes them.
//
// What is reimplemented, from numpy/random (_legacy_seeding, mt19937.c, distributions.c):
//
//   - np.random.seed(s) for a 32-bit integer s is MT19937's init_genrand(s): mt[0] = s,
//     mt[i] = 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i, and the state is regenerated
//     before the first draw.
//   - np.random.randint(low, high, size=n) on the legacy RandomState draws, for a range
//     rng = high - 1 - low below 2^32, one 32-bit output per attempt, masks it with the
//     smallest all-ones mask covering rng, and rejects values above rng
//     (random_bounded_uint64_fill / buffered_bounded_masked_uint32, use_masked=True). The
//     same path serves the int32 and int64 dtypes for such a range.
//
// NEP 19 freezes the legacy RandomState stream ("the legacy RandomState ... will remain
// stable"), so the values do not depend on the numpy version that wrote the fixture.

// mt19937 is the reference Mersenne Twister (Matsumoto and Nishimura, genrand_int32).
type mt19937 struct {
	mt  [624]uint32
	idx int
}

func newMT19937(seed uint32) *mt19937 {
	m := &mt19937{idx: 624}
	m.mt[0] = seed
	for i := 1; i < 624; i++ {
		m.mt[i] = 1812433253*(m.mt[i-1]^(m.mt[i-1]>>30)) + uint32(i)
	}
	return m
}

func (m *mt19937) next32() uint32 {
	const upper, lower, matrixA = 0x80000000, 0x7fffffff, 0x9908b0df
	if m.idx >= 624 {
		for i := 0; i < 624; i++ {
			y := (m.mt[i] & upper) | (m.mt[(i+1)%624] & lower)
			v := m.mt[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				v ^= matrixA
			}
			m.mt[i] = v
		}
		m.idx = 0
	}
	y := m.mt[m.idx]
	m.idx++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// randint is np.random.randint(low, high, size=n) on the legacy generator, high exclusive.
func (m *mt19937) randint(low, high, n int) []int {
	rng := uint32(high - 1 - low)
	mask := uint32(0)
	if rng > 0 {
		mask = ^uint32(0) >> bits.LeadingZeros32(rng)
	}
	out := make([]int, n)
	for i := range out {
		if rng == 0 {
			out[i] = low
			continue
		}
		v := m.next32() & mask
		for v > rng {
			v = m.next32() & mask
		}
		out[i] = low + int(v)
	}
	return out
}

// The generator itself, against values printed by numpy 2.5.4 (np.random.seed then
// np.random.randint, in the order listed). Several ranges, so the mask and the rejection run at
// different widths: 206 values (rng 205, mask 255), 1640 (rng 1639, mask 2047), 2 (mask 1),
// 200000 (mask 2^18-1) and 1 (rng 0, which draws nothing); and three seeds, including the
// largest 32-bit one.
func TestTheGoGeneratorReproducesNumpysLegacyRandint(t *testing.T) {
	for _, tc := range []struct {
		name  string
		seed  uint32
		calls [][3]int // low, high (exclusive), size -- in call order
		want  [][]int
	}{
		{"seed 0, 1k inputs", 0, [][3]int{{819, 1025, 5}}, [][]int{{991, 866, 936, 1011, 886}}},
		{"seed 0, 8k then 1k", 0, [][3]int{{6553, 8193, 3}, {819, 1025, 3}},
			[][]int{{7237, 7112, 7769}, {886, 1014, 922}}},
		{"seed 0, size-0 draw first", 0, [][3]int{{0, 200000, 0}, {819, 1025, 5}},
			[][]int{{}, {991, 866, 936, 1011, 886}}},
		{"seed 0, two values", 0, [][3]int{{0, 2, 8}}, [][]int{{0, 1, 1, 0, 1, 1, 1, 1}}},
		{"seed 0, vocab-sized", 0, [][3]int{{0, 200000, 4}}, [][]int{{199340, 43567, 173685, 117952}}},
		{"seed 0, a one-value range draws nothing", 0, [][3]int{{5, 6, 3}, {819, 1025, 2}},
			[][]int{{5, 5, 5}, {991, 866}}},
		{"seed 3", 3, [][3]int{{819, 1025, 4}, {6553, 8193, 4}},
			[][]int{{925, 971, 950, 1003}, {7521, 6809, 7342, 7574}}},
		{"seed 2^32-1", 4294967295, [][3]int{{819, 1025, 3}}, [][]int{{982, 853, 831}}},
	} {
		g := newMT19937(tc.seed)
		for i, c := range tc.calls {
			got := g.randint(c[0], c[1], c[2])
			if len(got) != len(tc.want[i]) {
				t.Fatalf("%s call %d: %d values, want %d", tc.name, i, len(got), len(tc.want[i]))
			}
			for j := range got {
				if got[j] != tc.want[i][j] {
					t.Errorf("%s call %d: %v, numpy gives %v", tc.name, i, got, tc.want[i])
					break
				}
			}
		}
	}
}

// Every committed fixture has the benchmark's shape, and every point's vectors are exactly the
// draws the client makes: input vector then output vector, from a freshly seeded generator.
// Checked for EVERY point of every seed, so a regenerated fixture that drifted from the
// client's call order -- outputs before inputs, one generator shared across points, a
// different ratio -- fails here without Python.
func TestTheFixturesAreTheClientsDraws(t *testing.T) {
	for seed := 0; seed <= 4; seed++ {
		path := DefaultWorkloadsPath(seed)
		if seed > 0 {
			if _, err := os.Stat(path); err != nil {
				t.Errorf("seed %d fixture missing: %v", seed, err)
				continue
			}
		}
		w, err := LoadBenchmarkWorkloads(path)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if w.Seed != seed {
			t.Errorf("%s states seed %d", path, w.Seed)
		}
		for _, p := range w.Points {
			// Literals restated from the client and its launcher, not the package constants:
			// --num-prompts $((CONC * 10)) and RANDOM_RANGE_RATIO 0.8.
			n := 10 * p.Concurrency
			g := newMT19937(uint32(seed))
			wantIn := g.randint(int(float64(p.ISL)*0.8), p.ISL+1, n)
			wantOut := g.randint(int(float64(p.OSL)*0.8), p.OSL+1, n)
			if !equalInts(p.InputLens, wantIn) || !equalInts(p.OutputLens, wantOut) {
				t.Errorf("seed %d point %d:%d c=%d is not the client's draw", seed, p.ISL, p.OSL, p.Concurrency)
			}
		}
	}
}

// The loader refuses a fixture that does not state the protocol, so a stale or hand-edited
// file cannot be scored as the benchmark's workload.
func TestTheLoaderRefusesAFixtureOffProtocol(t *testing.T) {
	good := PointLengths{ISL: 10, OSL: 10, Concurrency: 1,
		InputLens: []int{8, 9, 10, 8, 9, 10, 8, 9, 10, 8}, OutputLens: []int{10, 10, 10, 10, 10, 10, 10, 10, 10, 10}}
	if err := good.validate(); err != nil {
		t.Fatalf("a well-formed point is refused: %v", err)
	}
	short := good
	short.InputLens = good.InputLens[:9]
	below := good
	below.OutputLens = append([]int{7}, good.OutputLens[1:]...) // int(10*0.8) = 8
	above := good
	above.InputLens = append([]int{11}, good.InputLens[1:]...)
	for name, p := range map[string]PointLengths{"9 of 10 requests": short, "below the interval": below,
		"above the interval": above} {
		if p.validate() == nil {
			t.Errorf("a point with %s is accepted", name)
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
