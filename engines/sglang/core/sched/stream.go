// Original to blis-sglang.

package sched

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// Stream is a named random stream (DEV-6). Its generator is SplitMix64, seeded from the first
// eight bytes of the SHA-256 of "<seed>/<name>", so a draw depends on the scenario seed and the
// stream's name and on nothing that happened in another stream or to another request.
//
// SGLang's random policy shuffles the waiting queue with Python's global generator, whose state
// every other user of that generator moves; a simulator cannot reproduce that, and need not: the
// policy promises a uniform shuffle, not a sequence. The replay test therefore supplies SGLang's
// own order (Scheduler.Shuffle) and compares everything that follows from it, and a separate test
// checks that the stream shuffles uniformly and repeatably.
type Stream struct{ state uint64 }

// NewStream is the stream named name under the scenario seed.
func NewStream(seed int64, name string) *Stream {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%d/%s", seed, name)))
	return &Stream{state: binary.BigEndian.Uint64(sum[:8])}
}

func (s *Stream) next() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// Intn is a uniform draw from [0, n), by rejection so no value is favoured.
func (s *Stream) Intn(n int) int {
	if n <= 0 {
		panic("sched: Intn of a non-positive bound")
	}
	limit := ^uint64(0) - (^uint64(0))%uint64(n)
	for {
		if v := s.next(); v < limit {
			return int(v % uint64(n))
		}
	}
}

// Shuffle permutes n items uniformly (Fisher-Yates, as random.shuffle's loop: from the last item
// down, each swapped with a draw from those not yet fixed).
func (s *Stream) Shuffle(n int, swap func(i, j int)) {
	for i := n - 1; i >= 1; i-- {
		swap(i, s.Intn(i+1))
	}
}
