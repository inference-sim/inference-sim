package workload

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/bits"
)

// numpyGenerator reproduces numpy's default_rng(seed) for a non-negative integer seed:
// SeedSequence(seed) feeding PCG64 (XSL-RR 128/64), exactly as numpy 1.17+ computes it
// (numpy/random/bit_generator.pyx and numpy/random/src/pcg64/pcg64.h). aiperf draws each
// lane's snapshot instant t* with np.random.default_rng(seed).uniform(lo, hi)
// (timing/trajectory_source.py:602-637), so an exact replay needs the same stream. The
// golden vectors in testdata/agentx/numpy_rng_golden.json were produced by numpy itself.
type numpyGenerator struct {
	stateHi, stateLo uint64
	incHi, incLo     uint64
}

const (
	ssInitA    uint32 = 0x43b0d7e5
	ssMultA    uint32 = 0x931e8875
	ssInitB    uint32 = 0x8b51f9dd
	ssMultB    uint32 = 0x58f38ded
	ssMixMultL uint32 = 0xca01f9dd
	ssMixMultR uint32 = 0x4973f715
	ssXShift          = 16
	ssPoolSize        = 4

	pcgMultHi uint64 = 2549297995355413924
	pcgMultLo uint64 = 4865540595714422341
)

// seedSequencePool returns SeedSequence(entropy=seed).pool for an integer seed with no
// spawn key.
func seedSequencePool(seed uint64) [ssPoolSize]uint32 {
	var entropy []uint32
	if seed == 0 {
		entropy = []uint32{0}
	}
	for n := seed; n > 0; n >>= 32 {
		entropy = append(entropy, uint32(n))
	}
	hashConst := ssInitA
	hashmix := func(v uint32) uint32 {
		v ^= hashConst
		hashConst *= ssMultA
		v *= hashConst
		v ^= v >> ssXShift
		return v
	}
	mix := func(x, y uint32) uint32 {
		r := ssMixMultL*x - ssMixMultR*y
		return r ^ (r >> ssXShift)
	}
	var pool [ssPoolSize]uint32
	for i := range pool {
		if i < len(entropy) {
			pool[i] = hashmix(entropy[i])
		} else {
			pool[i] = hashmix(0)
		}
	}
	for src := 0; src < ssPoolSize; src++ {
		for dst := 0; dst < ssPoolSize; dst++ {
			if src != dst {
				pool[dst] = mix(pool[dst], hashmix(pool[src]))
			}
		}
	}
	for src := ssPoolSize; src < len(entropy); src++ {
		for dst := 0; dst < ssPoolSize; dst++ {
			pool[dst] = mix(pool[dst], hashmix(entropy[src]))
		}
	}
	return pool
}

// generateState64 is SeedSequence.generate_state(n, np.uint64).
func generateState64(pool [ssPoolSize]uint32, n int) []uint64 {
	words := make([]uint32, 2*n)
	hashConst := ssInitB
	for i := range words {
		v := pool[i%ssPoolSize]
		v ^= hashConst
		hashConst *= ssMultB
		v *= hashConst
		v ^= v >> ssXShift
		words[i] = v
	}
	out := make([]uint64, n)
	for i := range out {
		out[i] = uint64(words[2*i]) | uint64(words[2*i+1])<<32
	}
	return out
}

func newNumpyGenerator(seed uint64) *numpyGenerator {
	s := generateState64(seedSequencePool(seed), 4)
	g := &numpyGenerator{}
	// pcg_setseq_128_srandom_r(state = (s0, s1), initseq = (s2, s3)).
	g.incHi = s[2]<<1 | s[3]>>63
	g.incLo = s[3]<<1 | 1
	g.step()
	var c uint64
	g.stateLo, c = bits.Add64(g.stateLo, s[1], 0)
	g.stateHi, _ = bits.Add64(g.stateHi, s[0], c)
	g.step()
	return g
}

func (g *numpyGenerator) step() {
	hi, lo := bits.Mul64(g.stateLo, pcgMultLo)
	hi += g.stateHi*pcgMultLo + g.stateLo*pcgMultHi
	var c uint64
	lo, c = bits.Add64(lo, g.incLo, 0)
	hi, _ = bits.Add64(hi, g.incHi, c)
	g.stateHi, g.stateLo = hi, lo
}

// next64 is pcg64_next64: advance, then the XSL-RR output of the new state.
func (g *numpyGenerator) next64() uint64 {
	g.step()
	return bits.RotateLeft64(g.stateHi^g.stateLo, -int(g.stateHi>>58))
}

// random is Generator.random(): a 53-bit double in [0, 1).
func (g *numpyGenerator) random() float64 {
	return float64(g.next64()>>11) * (1.0 / 9007199254740992.0)
}

// uniform is Generator.uniform(low, high) for scalar bounds: low + (high-low)*random().
func (g *numpyGenerator) uniform(low, high float64) float64 {
	return low + float64((high-low)*g.random())
}

// agentxLaneSeed is aiperf's _seed_for_trace_lane: the first eight bytes, big-endian, of
// sha256("{base}:{trace_id}:{lane}") (timing/trajectory_source.py:172-180).
func agentxLaneSeed(base int64, traceID string, lane int) uint64 {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%d:%s:%d", base, traceID, lane)))
	return binary.BigEndian.Uint64(sum[:8])
}
